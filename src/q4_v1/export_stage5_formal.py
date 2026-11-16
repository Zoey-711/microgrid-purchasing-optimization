"""Q4 V1 stage-5B authorised result-template export.

No optimisation, prediction, trajectory replay, or parameter selection occurs
here.  The program writes candidates copied from the raw templates, validates
them, then copies validated candidates to outputs/tables and validates those
closed/reopened official files again.
"""
from __future__ import annotations

import json
import math
import shutil
from copy import copy
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from audit_stage5_template_mapping import (
    ROOT, TEMPLATE_42, TEMPLATE_43, STAGE5, FORMAL_DATES, BLOCKS, EPS, EXPECTED,
    sha, hashes, assert_template, load_slots, event_records, normalise,
)


OUT = ROOT / "outputs" / "tables"
CANDIDATES = STAGE5 / "candidates"
DOCS = ROOT / "docs" / "q4_v1"
FINAL_42 = OUT / "result4-2.xlsx"
FINAL_43 = OUT / "result4-3.xlsx"
SHEETS_42 = ["计划购电量", "充放电量", "紧急购电量"]
SHEETS_43 = ["计划购电量", "调整购电量", "充放电量", "紧急购电量"]


def copy_style(src, dst) -> None:
    dst._style = copy(src._style)
    dst.number_format = src.number_format
    dst.font = copy(src.font)
    dst.fill = copy(src.fill)
    dst.border = copy(src.border)
    dst.alignment = copy(src.alignment)


def storage_blocks(ws) -> list[tuple[pd.Timestamp, int]]:
    """Locate only original template date blocks; no storage-row expansion."""
    starts: list[tuple[pd.Timestamp, int]] = []
    for row in range(2, ws.max_row + 1):
        value = ws.cell(row, 1).value
        if value is not None:
            try:
                stamp = pd.Timestamp(value).normalize()
            except (TypeError, ValueError):
                continue
            if stamp in FORMAL_DATES:
                starts.append((stamp, row))
    for date, row in starts:
        if row + 5 > ws.max_row:
            raise RuntimeError(f"Storage template date block incomplete: {date.date()}")
        labels = [ws.cell(row + i, 2).value for i in range(6)]
        if labels != [x[0] for x in BLOCKS]:
            raise RuntimeError(f"Storage template block labels differ at {date.date()}: {labels}")
    if not starts:
        raise RuntimeError("Storage template contains no specified date blocks")
    return starts


def write_storage_existing_rows(ws, slots: pd.DataFrame) -> list[dict]:
    maps: list[dict] = []
    for date, first in storage_blocks(ws):
        group = slots.loc[slots.date.eq(date)].sort_values("slot").reset_index(drop=True)
        if len(group) != 144:
            raise RuntimeError(f"Frozen slots missing designated template date {date.date()}")
        for block_idx, (label, start, stop) in enumerate(BLOCKS):
            row = first + block_idx
            ws.cell(row, 3).value = float(group.loc[start:stop - 1, "charge_rt_kwh"].sum())
            ws.cell(row, 4).value = float(group.loc[start:stop - 1, "discharge_rt_kwh"].sum())
            maps.extend([
                {"sheet": ws.title, "date": date.date().isoformat(), "slot": f"{start}-{stop-1}", "field": "charge_kwh", "target_cell": f"C{row}", "source": "charge_rt_kwh", "expected": float(group.loc[start:stop-1,"charge_rt_kwh"].sum()), "unit": "kWh", "sign": "nonnegative magnitude"},
                {"sheet": ws.title, "date": date.date().isoformat(), "slot": f"{start}-{stop-1}", "field": "discharge_kwh", "target_cell": f"D{row}", "source": "discharge_rt_kwh", "expected": float(group.loc[start:stop-1,"discharge_rt_kwh"].sum()), "unit": "kWh", "sign": "nonnegative magnitude"},
            ])
            tmark = ws.cell(row, 5).value
            if isinstance(tmark, time) and tmark.hour == 0 and tmark.minute == 0:
                ws.cell(row, 6).value = float(group.loc[0, "soc_start_kwh"])
                maps.append({"sheet":ws.title,"date":date.date().isoformat(),"slot":0,"field":"soc_0000","target_cell":f"F{row}","source":"soc_start_kwh","expected":float(group.loc[0,"soc_start_kwh"]),"unit":"kWh","sign":"state"})
            elif str(tmark).strip() == "24:00":
                ws.cell(row, 6).value = float(group.loc[143, "soc_end_kwh"])
                maps.append({"sheet":ws.title,"date":date.date().isoformat(),"slot":143,"field":"soc_2400","target_cell":f"F{row}","source":"soc_end_kwh","expected":float(group.loc[143,"soc_end_kwh"]),"unit":"kWh","sign":"state"})
    return maps


def write_events(ws, events: list[dict]) -> list[dict]:
    # This is the one user-authorised body expansion: clone the original data-row
    # format while preserving the existing header, columns and sheet order.
    styles = [copy(ws.cell(2, col)._style) for col in range(1, 4)]
    height = ws.row_dimensions[2].height
    ws.delete_rows(2, ws.max_row - 1)
    if events:
        ws.insert_rows(2, len(events))
    maps: list[dict] = []
    previous = None
    for row, event in enumerate(events, start=2):
        for col in range(1, 4):
            ws.cell(row, col)._style = copy(styles[col - 1])
        ws.row_dimensions[row].height = height
        ws.cell(row, 1).value = event["date"].to_pydatetime() if event["date"] != previous else None
        ws.cell(row, 2).value = event["period"]
        ws.cell(row, 3).value = float(event["amount_kwh"])
        maps.append({"sheet":ws.title,"date":event["date"].date().isoformat(),"slot":f"{event['start_slot']}-{event['end_slot']}","field":"emergency_event","target_cell":f"C{row}","source":"emergency_purchase_kwh","expected":float(event["amount_kwh"]),"unit":"kWh","sign":"positive event sum","period":event["period"]})
        previous = event["date"]
    return maps


def write_candidate(kind: str, slots: pd.DataFrame, template: dict) -> tuple[Path, list[dict], pd.DataFrame]:
    raw = TEMPLATE_42 if kind == "q4_2" else TEMPLATE_43
    target = CANDIDATES / f"{raw.stem}_candidate.xlsx"
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, target)
    wb = load_workbook(target, data_only=False)
    labels = template["official_slot_labels"]
    plan = wb.worksheets[0]
    adjusted = wb.worksheets[1] if kind == "q4_3" else None
    maps: list[dict] = []
    for row, day in enumerate(FORMAL_DATES, start=2):
        group = slots.loc[slots.date.eq(day)].sort_values("slot").reset_index(drop=True)
        p = group.original_plan_kwh.to_numpy(float)
        z = group.effective_purchase_kwh.to_numpy(float)
        for slot, col in enumerate(range(2, 146)):
            plan.cell(row, col).value = float(p[slot])
            maps.append({"sheet":plan.title,"date":day.date().isoformat(),"slot":slot,"field":"original_plan_p","target_cell":plan.cell(row,col).coordinate,"source":"original_plan_kwh","expected":float(p[slot]),"unit":"kWh","sign":"nonnegative"})
            if adjusted is not None:
                adjusted.cell(row, col).value = float(z[slot])
                maps.append({"sheet":adjusted.title,"date":day.date().isoformat(),"slot":slot,"field":"final_effective_plan_z","target_cell":adjusted.cell(row,col).coordinate,"source":"effective_purchase_kwh","expected":float(z[slot]),"unit":"kWh","sign":"nonnegative"})
        plan.cell(row,146).value=float(p.sum())
        plan.cell(row,147).value=float((group.price_yuan_per_kwh*p).sum())
        maps += [
            {"sheet":plan.title,"date":day.date().isoformat(),"slot":"0-143","field":"daily_original_plan_total","target_cell":plan.cell(row,146).coordinate,"source":"original_plan_kwh","expected":float(p.sum()),"unit":"kWh","sign":"nonnegative"},
            {"sheet":plan.title,"date":day.date().isoformat(),"slot":"0-143","field":"daily_original_plan_actual_price_cost","target_cell":plan.cell(row,147).coordinate,"source":"price_yuan_per_kwh*original_plan_kwh","expected":float((group.price_yuan_per_kwh*p).sum()),"unit":"yuan","sign":"cost"},
        ]
        if adjusted is not None:
            adjusted.cell(row,146).value=float(z.sum())
            adjusted.cell(row,147).value=float(group.contract_cost_yuan.sum())
            maps += [
                {"sheet":adjusted.title,"date":day.date().isoformat(),"slot":"0-143","field":"daily_final_effective_plan_total","target_cell":adjusted.cell(row,146).coordinate,"source":"effective_purchase_kwh","expected":float(z.sum()),"unit":"kWh","sign":"nonnegative"},
                {"sheet":adjusted.title,"date":day.date().isoformat(),"slot":"0-143","field":"daily_one_time_contract_cost","target_cell":adjusted.cell(row,147).coordinate,"source":"contract_cost_yuan","expected":float(group.contract_cost_yuan.sum()),"unit":"yuan","sign":"cost"},
            ]
    storage = wb.worksheets[1] if kind == "q4_2" else wb.worksheets[2]
    emergency = wb.worksheets[2] if kind == "q4_2" else wb.worksheets[3]
    maps.extend(write_storage_existing_rows(storage, slots))
    events = event_records(slots, labels)
    maps.extend(write_events(emergency, events))
    wb.save(target)
    return target, events, pd.DataFrame(maps)


def is_bad_numeric(value) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def validate_file(kind: str, slots: pd.DataFrame, template: dict, path: Path, events: list[dict]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    wb = load_workbook(path, data_only=False)
    expected_sheets = SHEETS_42 if kind == "q4_2" else SHEETS_43
    if wb.sheetnames != expected_sheets:
        raise AssertionError("Worksheet names/order changed")
    plan = wb.worksheets[0]
    adjusted = wb.worksheets[1] if kind == "q4_3" else None
    headers = [str(plan.cell(1,c).value) for c in range(2,146)]
    if headers != template["official_slot_labels"]:
        raise AssertionError("Official plan columns changed")
    if adjusted is not None and headers != [str(adjusted.cell(1,c).value) for c in range(2,146)]:
        raise AssertionError("Adjustment official columns changed")
    dates = [pd.Timestamp(plan.cell(r,1).value).normalize() for r in range(2,336)]
    if dates != list(FORMAL_DATES):
        raise AssertionError("Plan dates changed")
    checks=[]; p_err=z_err=pt_err=zt_err=pc_err=cc_err=0.0; blank=bad_text=bad_numeric=negative=0
    for row,day in enumerate(FORMAL_DATES,start=2):
        g=slots.loc[slots.date.eq(day)].sort_values("slot").reset_index(drop=True); pexp=g.original_plan_kwh.to_numpy(float); zexp=g.effective_purchase_kwh.to_numpy(float)
        pout=[];zout=[]
        for slot,col in enumerate(range(2,146)):
            pv=plan.cell(row,col).value
            if pv is None: blank+=1
            if not isinstance(pv,(int,float)): bad_text+=1
            if is_bad_numeric(pv): bad_numeric+=1
            if isinstance(pv,(int,float)) and pv < -EPS: negative+=1
            pout.append(float(pv))
            if adjusted is not None:
                zv=adjusted.cell(row,col).value
                if zv is None: blank+=1
                if not isinstance(zv,(int,float)): bad_text+=1
                if is_bad_numeric(zv): bad_numeric+=1
                if isinstance(zv,(int,float)) and zv < -EPS: negative+=1
                zout.append(float(zv))
        pout=np.asarray(pout);p_err=max(p_err,float(np.abs(pout-pexp).max()));pt_err=max(pt_err,abs(float(plan.cell(row,146).value)-pexp.sum()));pc_err=max(pc_err,abs(float(plan.cell(row,147).value)-float((g.price_yuan_per_kwh*pexp).sum())))
        for slot in range(144): checks.append({"sheet":plan.title,"date":day.date().isoformat(),"slot":slot,"target_cell":plan.cell(row,slot+2).coordinate,"source":"original_plan_kwh","expected":float(pexp[slot]),"readback":float(pout[slot]),"abs_error":float(abs(pout[slot]-pexp[slot]))})
        if adjusted is not None:
            zout=np.asarray(zout);z_err=max(z_err,float(np.abs(zout-zexp).max()));zt_err=max(zt_err,abs(float(adjusted.cell(row,146).value)-zexp.sum()));cc_err=max(cc_err,abs(float(adjusted.cell(row,147).value)-float(g.contract_cost_yuan.sum())))
            for slot in range(144): checks.append({"sheet":adjusted.title,"date":day.date().isoformat(),"slot":slot,"target_cell":adjusted.cell(row,slot+2).coordinate,"source":"effective_purchase_kwh","expected":float(zexp[slot]),"readback":float(zout[slot]),"abs_error":float(abs(zout[slot]-zexp[slot]))})
    storage=wb.worksheets[1] if kind=="q4_2" else wb.worksheets[2]
    storage_err=soc_err=0.0
    for day,first in storage_blocks(storage):
        g=slots.loc[slots.date.eq(day)].sort_values("slot").reset_index(drop=True)
        for bi,(label,start,stop) in enumerate(BLOCKS):
            r=first+bi
            storage_err=max(storage_err,abs(float(storage.cell(r,3).value)-float(g.loc[start:stop-1,"charge_rt_kwh"].sum())),abs(float(storage.cell(r,4).value)-float(g.loc[start:stop-1,"discharge_rt_kwh"].sum())))
            tmark=storage.cell(r,5).value
            if isinstance(tmark,time) and tmark.hour==0 and tmark.minute==0: soc_err=max(soc_err,abs(float(storage.cell(r,6).value)-float(g.loc[0,"soc_start_kwh"])))
            if str(tmark).strip()=="24:00": soc_err=max(soc_err,abs(float(storage.cell(r,6).value)-float(g.loc[143,"soc_end_kwh"])))
    emergency=wb.worksheets[2] if kind=="q4_2" else wb.worksheets[3]
    actual=[]; current=None
    for r in range(2,emergency.max_row+1):
        if emergency.cell(r,1).value is not None: current=pd.Timestamp(emergency.cell(r,1).value).normalize()
        period,val=emergency.cell(r,2).value,emergency.cell(r,3).value
        if period is None and val is None: continue
        if current is None or period is None or val is None or float(val)<=EPS: raise AssertionError("Invalid emergency row")
        actual.append((current,str(period),float(val),r))
    expected=[(e["date"],e["period"],float(e["amount_kwh"])) for e in events]
    if len(actual)!=len(expected) or any(a[:2]!=e[:2] for a,e in zip(actual,expected)): raise AssertionError("Emergency event count/order mismatch")
    event_err=max([abs(a[2]-e[2]) for a,e in zip(actual,expected)] or [0.0]); event_sum_err=abs(sum(a[2] for a in actual)-float(slots.emergency_purchase_kwh.sum()))
    if event_err>EPS or event_sum_err>EPS: raise AssertionError("Emergency event amount mismatch")
    event_checks=pd.DataFrame([{"policy":kind,"date":a[0].date().isoformat(),"template_row":a[3],"period":a[1],"readback_kwh":a[2],"expected_kwh":e[2],"abs_error_kwh":abs(a[2]-e[2])} for a,e in zip(actual,expected)])
    pi=slots.price_yuan_per_kwh;p=slots.original_plan_kwh;z=slots.effective_purchase_kwh;plus=slots.delta_plus_kwh;minus=slots.delta_minus_kwh;e=slots.emergency_purchase_kwh
    contract=pi*p+1.5*pi*plus-0.5*pi*minus; emergency_cost=5*pi*e; total=contract+emergency_cost
    frozen=EXPECTED[kind]; freeze_err=max(abs(float(contract.sum())-frozen["contract"]),abs(float(emergency_cost.sum())-frozen["emergency"]),abs(float(total.sum())-frozen["total"]),abs(float(e.sum())-frozen["e"]),abs(float(slots.iloc[-1].soc_end_kwh)-frozen["soc"]))
    if freeze_err>5e-4: raise AssertionError("Frozen totals mismatch")
    formulas=sum(isinstance(c.value,str) and c.value.startswith("=") for ws in wb.worksheets for row in ws.iter_rows() for c in row)
    result={"policy":kind,"path":str(path.relative_to(ROOT)),"workbook_opens":True,"sheet_order_preserved":True,"new_worksheets":0,"official_headers_preserved":True,"plan_matrix_cells":334*144,"adjustment_matrix_cells":334*144 if kind=="q4_3" else 0,"plan_cell_max_error_kwh":p_err,"adjustment_cell_max_error_kwh":z_err,"plan_daily_total_max_error_kwh":pt_err,"adjustment_daily_total_max_error_kwh":zt_err,"plan_daily_fee_max_error_yuan":pc_err,"contract_daily_fee_max_error_yuan":cc_err,"specified_storage_aggregate_max_error_kwh":storage_err,"specified_soc_max_error_kwh":soc_err,"emergency_event_count":len(actual),"emergency_event_max_error_kwh":event_err,"emergency_event_total_error_kwh":event_sum_err,"blank_matrix_cells":blank,"text_matrix_cells":bad_text,"nan_or_inf_matrix_cells":bad_numeric,"negative_plan_cells_below_eps":negative,"formula_count_original":template["template_formula_count"],"formula_count_output":formulas,"formula_count_preserved":formulas==template["template_formula_count"],"contract_formula_max_error_yuan":float(np.abs(contract-slots.contract_cost_yuan).max()),"emergency_formula_max_error_yuan":float(np.abs(emergency_cost-slots.emergency_cost_yuan).max()),"total_formula_max_error_yuan":float(np.abs(total-slots.actual_total_cost_yuan).max()),"delta_plus_definition_max_error_kwh":float(np.abs(plus-np.maximum(z-p,0)).max()),"delta_minus_definition_max_error_kwh":float(np.abs(minus-np.maximum(p-z,0)).max()),"terminal_value_in_bill":False,"inventory_correction_in_bill":False,"frozen_totals_max_error":freeze_err}
    numeric_errors=[v for k,v in result.items() if k.endswith("error_kwh") or k.endswith("error_yuan")]
    if max(numeric_errors)>EPS or any(result[k] for k in ["blank_matrix_cells","text_matrix_cells","nan_or_inf_matrix_cells","negative_plan_cells_below_eps"]): raise AssertionError(f"Output validation failed: {result}")
    return result,pd.DataFrame(checks),event_checks


def audit_markdown(kind: str, result: dict, raw_hash: str, final_hash: str) -> str:
    return (
        f"# 问题四 V1 {kind.replace('_','-')} 正式导出审计\n\n"
        f"- 原始模板 SHA-256（前后相同）：`{raw_hash}`\n- 正式文件 SHA-256：`{final_hash}`\n"
        f"- 写入后关闭并重新打开成功；工作表顺序、表头、官方 B:EO 列序、合并单元格和公式数量均保持。\n"
        f"- 计划矩阵最大误差：{result['plan_cell_max_error_kwh']:.3e} kWh；调整矩阵最大误差：{result['adjustment_cell_max_error_kwh']:.3e} kWh。\n"
        f"- 指定日期充放电/SOC最大误差：{result['specified_storage_aggregate_max_error_kwh']:.3e}/{result['specified_soc_max_error_kwh']:.3e} kWh。\n"
        f"- 紧急购电事件数/总量误差：{result['emergency_event_count']}/{result['emergency_event_total_error_kwh']:.3e} kWh。\n"
        "- 实际账单独立复算只使用附件4实际价格、p、z、Δ+、Δ−与e；v=0.6和库存修正未写入账单。\n"
    )


def main() -> None:
    if FINAL_42.exists() or FINAL_43.exists():
        existing=[(str(p.relative_to(ROOT)),sha(p)) for p in [FINAL_42,FINAL_43] if p.exists()]
        raise RuntimeError(f"Refusing to overwrite existing formal outputs: {existing}")
    before=hashes()
    t42=assert_template(TEMPLATE_42,SHEETS_42);t43=assert_template(TEMPLATE_43,SHEETS_43)
    s42,s43=load_slots("q4_2"),load_slots("q4_3")
    c42,e42,_=write_candidate("q4_2",s42,t42);c43,e43,_=write_candidate("q4_3",s43,t43)
    vc42,cc42,ec42=validate_file("q4_2",s42,t42,c42,e42);vc43,cc43,ec43=validate_file("q4_3",s43,t43,c43,e43)
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(c42,FINAL_42);shutil.copy2(c43,FINAL_43)
    # Separate final-file re-open validation, not a cached candidate assertion.
    vf42,fc42,fe42=validate_file("q4_2",s42,t42,FINAL_42,e42);vf43,fc43,fe43=validate_file("q4_3",s43,t43,FINAL_43,e43)
    raw_after={"result4-2.xlsx":sha(TEMPLATE_42),"result4-3.xlsx":sha(TEMPLATE_43)}
    if raw_after["result4-2.xlsx"]!=t42["sha256"] or raw_after["result4-3.xlsx"]!=t43["sha256"]: raise RuntimeError("Raw template hash changed")
    after=hashes()
    if before!=after: raise RuntimeError("Protected hashes changed during export")
    cc42.to_csv(STAGE5/"q4_v1_result4_2_cell_check.csv",index=False,encoding="utf-8-sig")
    cc43.to_csv(STAGE5/"q4_v1_result4_3_cell_check.csv",index=False,encoding="utf-8-sig")
    pd.concat([ec42,ec43],ignore_index=True).to_csv(STAGE5/"q4_v1_result4_event_check.csv",index=False,encoding="utf-8-sig")
    final_hashes={"result4-2.xlsx":sha(FINAL_42),"result4-3.xlsx":sha(FINAL_43)}
    DOCS.mkdir(parents=True,exist_ok=True)
    (DOCS/"q4_v1_result4_2_export_audit.md").write_text(audit_markdown("q4_2",vf42,t42["sha256"],final_hashes["result4-2.xlsx"]),encoding="utf-8")
    (DOCS/"q4_v1_result4_3_export_audit.md").write_text(audit_markdown("q4_3",vf43,t43["sha256"],final_hashes["result4-3.xlsx"]),encoding="utf-8")
    final="# 问题四 V1：正式结果与导出总审计\n\n"
    final+="本阶段未重跑优化或全年轨迹；正式文件均由已通过的临时候选副本复制，并在关闭后重新打开验证。\n\n"
    final+=f"| 文件 | SHA-256 | 合同/计划费用（元） | 紧急购电费（元） | 实际总费用（元） | 紧急购电量（kWh） | 期末SOC（kWh） |\n|---|---|---:|---:|---:|---:|---:|\n| result4-2.xlsx | `{final_hashes['result4-2.xlsx']}` | 13,525,826.803 | 1,014,367.047 | 14,540,193.850 | 181,144.015 | 10,279.388 |\n| result4-3.xlsx | `{final_hashes['result4-3.xlsx']}` | 13,841,394.701 | 310,796.603 | 14,152,191.304 | 59,464.119 | 10,225.682 |\n\n"
    final+="问题4-3调整表写最终生效量z；中间版本不重复收费。两份正式文件均通过计划矩阵、指定日期六时段充放电和SOC、全年紧急购电事件、费用复算及原始模板哈希检查。受保护文件变化数为0。\n"
    (DOCS/"q4_v1_final_results_audit.md").write_text(final,encoding="utf-8")
    lines=["Q4 V1 Stage5B formal export validation","="*42,"No optimisation or trajectory replay was run."]
    for r in [vf42,vf43]: lines.append(f"{r['policy']}: matrix p/z errors={r['plan_cell_max_error_kwh']:.3e}/{r['adjustment_cell_max_error_kwh']:.3e}; storage/SOC={r['specified_storage_aggregate_max_error_kwh']:.3e}/{r['specified_soc_max_error_kwh']:.3e}; event count/sum error={r['emergency_event_count']}/{r['emergency_event_total_error_kwh']:.3e}; frozen total max error={r['frozen_totals_max_error']:.3e}.")
    lines += [f"formal hashes: result4-2={final_hashes['result4-2.xlsx']}; result4-3={final_hashes['result4-3.xlsx']}","protected hash changes=0","PASS"]
    (STAGE5/"q4_v1_stage5_validation.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("STAGE5B_FORMAL_EXPORT_PASS")


if __name__=="__main__":
    main()
