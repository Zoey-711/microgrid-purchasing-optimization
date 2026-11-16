"""Q4 V1 stage-5A template audit and dry-run mapping.

Post-processing only.  Reads frozen stage-4 trajectories and original
templates, writes temporary workbooks only under outputs/q4_v1/stage5/dry_run,
and never creates outputs/tables/result4-2.xlsx or result4-3.xlsx.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from copy import copy
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(r"E:\数学建模\C题")
RAW = ROOT / "data" / "raw"
TEMPLATE_42 = RAW / "result4-2.xlsx"
TEMPLATE_43 = RAW / "result4-3.xlsx"
STAGE4 = ROOT / "outputs" / "q4_v1" / "stage4"
CHECKPOINTS = STAGE4 / "checkpoints"
STAGE5 = ROOT / "outputs" / "q4_v1" / "stage5"
DRY = STAGE5 / "dry_run"
DOCS = ROOT / "docs" / "q4_v1"
FORMAL_DATES = pd.date_range("2025-02-01", "2025-12-31", freq="D")
EPS = 1e-6
BLOCKS = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
          ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]
EXPECTED = {
    "q4_2": {"contract": 13525826.803, "emergency": 1014367.047, "total": 14540193.850, "e": 181144.015, "soc": 10279.388},
    "q4_3": {"contract": 13841394.701, "emergency": 310796.603, "total": 14152191.304, "e": 59464.119, "soc": 10225.682},
}
PROTECTED = [
    ROOT / "outputs" / "tables" / "result1.xlsx", ROOT / "outputs" / "tables" / "result2.xlsx", ROOT / "outputs" / "tables" / "result3.xlsx",
    TEMPLATE_42, TEMPLATE_43,
    ROOT / "outputs" / "q4_v1" / "stage3" / "q4_v1_frozen_config_4_2.json",
    ROOT / "outputs" / "q4_v1" / "stage3" / "q4_v1_frozen_config_4_3.json",
    CHECKPOINTS / "q4_2_checkpoint.json", CHECKPOINTS / "q4_2_slots.pkl", CHECKPOINTS / "q4_3_checkpoint.json", CHECKPOINTS / "q4_3_slots.pkl", CHECKPOINTS / "q4_3_plans.pkl",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def hashes() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha(p) if p.exists() else "MISSING" for p in PROTECTED}


def style_copy(src, dst) -> None:
    dst._style = copy(src._style)
    dst.number_format = src.number_format
    dst.font = copy(src.font)
    dst.fill = copy(src.fill)
    dst.border = copy(src.border)
    dst.alignment = copy(src.alignment)


def normalise(value: float, name: str) -> float:
    value = float(value)
    if -EPS <= value < 0:
        return 0.0
    if value < -EPS:
        raise ValueError(f"{name} is below -EPS: {value}")
    return value


def load_slots(policy: str) -> pd.DataFrame:
    x = pd.read_pickle(CHECKPOINTS / f"{policy}_slots.pkl").copy()
    x["date"] = pd.to_datetime(x.date).dt.normalize()
    x = x.sort_values(["date", "slot"]).reset_index(drop=True)
    if len(x) != len(FORMAL_DATES) * 144 or x.date.nunique() != len(FORMAL_DATES):
        raise RuntimeError(f"{policy}: frozen source is not 334 x 144")
    if list(x.date.drop_duplicates()) != list(FORMAL_DATES):
        raise RuntimeError(f"{policy}: date order does not match the official period")
    if not all(g.slot.astype(int).tolist() == list(range(144)) for _, g in x.groupby("date", sort=True)):
        raise RuntimeError(f"{policy}: slot order is not 0..143")
    for col in ["original_plan_kwh", "effective_purchase_kwh", "charge_rt_kwh", "discharge_rt_kwh", "emergency_purchase_kwh", "spill_kwh", "delta_plus_kwh", "delta_minus_kwh"]:
        x[col] = [normalise(v, col) for v in x[col]]
    return x


def workbook_audit(path: Path) -> dict:
    wb = load_workbook(path, data_only=False)
    result = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "sheets": []}
    for ws in wb.worksheets:
        nonempty = sum(cell.value is not None for row in ws.iter_rows() for cell in row)
        formulas = sum(isinstance(cell.value, str) and cell.value.startswith("=") for row in ws.iter_rows() for cell in row)
        result["sheets"].append({
            "name": ws.title, "max_row": ws.max_row, "max_column": ws.max_column,
            "header": [ws.cell(1, c).value for c in range(1, ws.max_column + 1)],
            "merged_ranges": [str(r) for r in ws.merged_cells.ranges], "formula_count": formulas,
            "data_validation_count": len(ws.data_validations.dataValidation) if ws.data_validations else 0,
            "freeze_panes": str(ws.freeze_panes),
            "hidden_rows": [r for r, d in ws.row_dimensions.items() if d.hidden],
            "hidden_columns": [c for c, d in ws.column_dimensions.items() if d.hidden],
            "nonempty_cells": nonempty,
            "blank_body_cells": sum(ws.cell(r, c).value is None for r in range(2, ws.max_row + 1) for c in range(1, ws.max_column + 1)),
            "row_heights": {str(r): d.height for r, d in ws.row_dimensions.items() if d.height is not None},
            "column_widths": {str(c): d.width for c, d in ws.column_dimensions.items() if d.width is not None},
            "format_samples": {coord: ws[coord].number_format for coord in ["A1", "B1", "B2", "EP1", "EP2", "EQ1", "EQ2"] if ws.max_column >= ws[coord].column and ws.max_row >= ws[coord].row},
        })
    return result


def assert_template(path: Path, expected_sheets: list[str]) -> dict:
    audit = workbook_audit(path)
    wb = load_workbook(path, data_only=False)
    if wb.sheetnames != expected_sheets:
        raise RuntimeError(f"Unexpected sheet order in {path.name}: {wb.sheetnames}")
    matrix_sheets = wb.worksheets[:2] if len(expected_sheets) == 4 else wb.worksheets[:1]
    labels = [str(matrix_sheets[0].cell(1, c).value) for c in range(2, 146)]
    if len(labels) != 144 or any(x in {"None", ""} for x in labels):
        raise RuntimeError(f"{path.name}: B:EO does not expose 144 official slot columns")
    dates = [pd.Timestamp(matrix_sheets[0].cell(r, 1).value).normalize() for r in range(2, 336)]
    if dates != list(FORMAL_DATES):
        raise RuntimeError(f"{path.name}: date matrix does not cover the official 334 days")
    for ws in matrix_sheets:
        if [str(ws.cell(1, c).value) for c in range(2, 146)] != labels:
            raise RuntimeError(f"{path.name}: official slot headers disagree across plan matrices")
        if any(ws.cell(r, c).value is not None for r in range(2, 336) for c in range(2, 146)):
            raise RuntimeError(f"{path.name}: expected empty B:EO output matrix is not empty")
    audit["official_slot_labels"] = labels
    audit["template_formula_count"] = sum(s["formula_count"] for s in audit["sheets"])
    return audit


def event_records(slots: pd.DataFrame, labels: list[str]) -> list[dict]:
    records: list[dict] = []
    for day, group in slots.groupby("date", sort=True):
        group = group.sort_values("slot").reset_index(drop=True)
        positive = group.loc[group.emergency_purchase_kwh > EPS, "slot"].astype(int).tolist()
        if not positive:
            continue
        start = last = positive[0]
        for slot in positive[1:] + [None]:
            if slot is not None and slot == last + 1:
                last = slot
                continue
            start_text = labels[start].split("-", 1)[0]
            end_text = labels[last].split("-", 1)[1]
            amount = float(group.loc[start:last, "emergency_purchase_kwh"].sum())
            records.append({"date": pd.Timestamp(day), "start_slot": start, "end_slot": last, "period": f"{start_text}-{end_text}", "amount_kwh": amount, "slot_count": last-start+1})
            if slot is None:
                break
            start = last = slot
    return records


def expand_storage(ws, slots: pd.DataFrame) -> list[dict]:
    styles = [[copy(ws.cell(r, c)._style) for c in range(1, 7)] for r in range(2, 8)]
    heights = [ws.row_dimensions[r].height for r in range(2, 8)]
    ws.delete_rows(2, ws.max_row - 1)
    ws.insert_rows(2, len(FORMAL_DATES) * 6)
    mapping: list[dict] = []
    for day_index, day in enumerate(FORMAL_DATES):
        group = slots[slots.date.eq(day)].sort_values("slot").reset_index(drop=True)
        for block_index, (block, start, stop) in enumerate(BLOCKS):
            row = 2 + day_index * 6 + block_index
            for col in range(1, 7):
                ws.cell(row, col)._style = copy(styles[block_index][col-1])
            ws.row_dimensions[row].height = heights[block_index]
            ws.cell(row, 1).value = day.to_pydatetime() if block_index == 0 else None
            ws.cell(row, 2).value = block
            ws.cell(row, 3).value = float(group.loc[start:stop-1, "charge_rt_kwh"].sum())
            ws.cell(row, 4).value = float(group.loc[start:stop-1, "discharge_rt_kwh"].sum())
            ws.cell(row, 5).value = time(0, 0) if block_index == 0 else ("24:00" if block_index == 1 else None)
            ws.cell(row, 6).value = float(group.loc[0, "soc_start_kwh"]) if block_index == 0 else (float(group.loc[143, "soc_end_kwh"]) if block_index == 1 else None)
            mapping.append({"sheet": ws.title, "date": day.date().isoformat(), "template_row": row, "field": "charge_kwh", "source": "charge_rt_kwh", "slot_start": start, "slot_end": stop-1, "target_cell": f"C{row}", "unit": "kWh", "sign": "nonnegative magnitude"})
            mapping.append({"sheet": ws.title, "date": day.date().isoformat(), "template_row": row, "field": "discharge_kwh", "source": "discharge_rt_kwh", "slot_start": start, "slot_end": stop-1, "target_cell": f"D{row}", "unit": "kWh", "sign": "nonnegative magnitude"})
            if block_index in (0, 1):
                mapping.append({"sheet": ws.title, "date": day.date().isoformat(), "template_row": row, "field": "SOC", "source": "soc_start_kwh" if block_index == 0 else "soc_end_kwh", "slot_start": 0 if block_index == 0 else 143, "slot_end": 0 if block_index == 0 else 143, "target_cell": f"F{row}", "unit": "kWh", "sign": "state"})
    return mapping


def expand_events(ws, events: list[dict]) -> list[dict]:
    styles = [copy(ws.cell(2, c)._style) for c in range(1, 4)]
    height = ws.row_dimensions[2].height
    ws.delete_rows(2, ws.max_row - 1)
    if events:
        ws.insert_rows(2, len(events))
    mapping: list[dict] = []
    previous = None
    for i, event in enumerate(events, start=2):
        for col in range(1, 4):
            ws.cell(i, col)._style = copy(styles[col-1])
        ws.row_dimensions[i].height = height
        ws.cell(i, 1).value = event["date"].to_pydatetime() if event["date"] != previous else None
        ws.cell(i, 2).value, ws.cell(i, 3).value = event["period"], event["amount_kwh"]
        mapping.append({"sheet": ws.title, "date": event["date"].date().isoformat(), "template_row": i, "field": "emergency_event_kwh", "source": "emergency_purchase_kwh", "slot_start": event["start_slot"], "slot_end": event["end_slot"], "target_cell": f"C{i}", "unit": "kWh", "sign": "strictly positive event sum", "period": event["period"]})
        previous = event["date"]
    return mapping


def write_dry_run(kind: str, slots: pd.DataFrame, template: dict) -> tuple[Path, list[dict], pd.DataFrame]:
    source = TEMPLATE_42 if kind == "q4_2" else TEMPLATE_43
    target = DRY / f"{source.stem}_dry_run.xlsx"
    DRY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    wb = load_workbook(target, data_only=False)
    labels = template["official_slot_labels"]
    matrices = wb.worksheets[:1] if kind == "q4_2" else wb.worksheets[:2]
    maps: list[dict] = []
    for row, day in enumerate(FORMAL_DATES, start=2):
        group = slots[slots.date.eq(day)].sort_values("slot").reset_index(drop=True)
        originals = group.original_plan_kwh.to_numpy(float)
        effective = group.effective_purchase_kwh.to_numpy(float)
        for slot, col in enumerate(range(2, 146)):
            matrices[0].cell(row, col).value = float(originals[slot])
            maps.append({"sheet": matrices[0].title, "date": day.date().isoformat(), "template_row": row, "field": "original_plan_p", "source": "original_plan_kwh", "slot_start": slot, "slot_end": slot, "target_cell": matrices[0].cell(row, col).coordinate, "unit": "kWh", "sign": "nonnegative"})
            if kind == "q4_3":
                matrices[1].cell(row, col).value = float(effective[slot])
                maps.append({"sheet": matrices[1].title, "date": day.date().isoformat(), "template_row": row, "field": "final_effective_plan_z", "source": "effective_purchase_kwh", "slot_start": slot, "slot_end": slot, "target_cell": matrices[1].cell(row, col).coordinate, "unit": "kWh", "sign": "nonnegative"})
        matrices[0].cell(row, 146).value = float(originals.sum())
        matrices[0].cell(row, 147).value = float((group.price_yuan_per_kwh * originals).sum())
        maps += [
            {"sheet": matrices[0].title, "date": day.date().isoformat(), "template_row": row, "field": "daily_original_plan_total", "source": "original_plan_kwh", "slot_start": 0, "slot_end": 143, "target_cell": matrices[0].cell(row,146).coordinate, "unit": "kWh", "sign": "nonnegative"},
            {"sheet": matrices[0].title, "date": day.date().isoformat(), "template_row": row, "field": "daily_original_plan_actual_price_cost", "source": "price_yuan_per_kwh*original_plan_kwh", "slot_start": 0, "slot_end": 143, "target_cell": matrices[0].cell(row,147).coordinate, "unit": "yuan", "sign": "cost"},
        ]
        if kind == "q4_3":
            matrices[1].cell(row, 146).value = float(effective.sum())
            matrices[1].cell(row, 147).value = float(group.contract_cost_yuan.sum())
            maps += [
                {"sheet": matrices[1].title, "date": day.date().isoformat(), "template_row": row, "field": "daily_final_effective_total", "source": "effective_purchase_kwh", "slot_start": 0, "slot_end": 143, "target_cell": matrices[1].cell(row,146).coordinate, "unit": "kWh", "sign": "nonnegative"},
                {"sheet": matrices[1].title, "date": day.date().isoformat(), "template_row": row, "field": "daily_one_time_contract_settlement", "source": "contract_cost_yuan", "slot_start": 0, "slot_end": 143, "target_cell": matrices[1].cell(row,147).coordinate, "unit": "yuan", "sign": "cost"},
            ]
    storage = wb.worksheets[1] if kind == "q4_2" else wb.worksheets[2]
    emergency = wb.worksheets[2] if kind == "q4_2" else wb.worksheets[3]
    maps += expand_storage(storage, slots)
    events = event_records(slots, labels)
    maps += expand_events(emergency, events)
    wb.save(target)
    return target, events, pd.DataFrame(maps)


def validate_dry_run(kind: str, slots: pd.DataFrame, template: dict, workbook: Path, events: list[dict]) -> tuple[dict, pd.DataFrame]:
    wb = load_workbook(workbook, data_only=False)
    expected_sheets = [s["name"] for s in template["sheets"]]
    if wb.sheetnames != expected_sheets:
        raise AssertionError("Worksheet names or order changed")
    matrices = wb.worksheets[:1] if kind == "q4_2" else wb.worksheets[:2]
    labels = [str(matrices[0].cell(1,c).value) for c in range(2,146)]
    if labels != template["official_slot_labels"]:
        raise AssertionError("Official 144-slot headers changed")
    if kind == "q4_3" and labels != [str(matrices[1].cell(1,c).value) for c in range(2,146)]:
        raise AssertionError("Plan and adjustment headers disagree")
    cell_rows = []
    errors = {"p":0.0,"z":0.0,"pt":0.0,"zt":0.0,"pc":0.0,"cc":0.0}
    for row, day in enumerate(FORMAL_DATES, start=2):
        group=slots[slots.date.eq(day)].sort_values("slot").reset_index(drop=True)
        p=np.array([matrices[0].cell(row,c).value for c in range(2,146)],dtype=float); exp=group.original_plan_kwh.to_numpy(float)
        errors["p"]=max(errors["p"],float(np.abs(p-exp).max())); errors["pt"]=max(errors["pt"],abs(float(matrices[0].cell(row,146).value)-exp.sum())); errors["pc"]=max(errors["pc"],abs(float(matrices[0].cell(row,147).value)-float((group.price_yuan_per_kwh*exp).sum())))
        for t in range(144): cell_rows.append({"sheet":matrices[0].title,"date":day.date().isoformat(),"slot":t,"target_cell":matrices[0].cell(row,t+2).coordinate,"source":"original_plan_kwh","expected_kwh":float(exp[t]),"readback_kwh":float(p[t]),"abs_error_kwh":float(abs(exp[t]-p[t]))})
        if kind == "q4_3":
            z=np.array([matrices[1].cell(row,c).value for c in range(2,146)],dtype=float); zexp=group.effective_purchase_kwh.to_numpy(float)
            errors["z"]=max(errors["z"],float(np.abs(z-zexp).max())); errors["zt"]=max(errors["zt"],abs(float(matrices[1].cell(row,146).value)-zexp.sum())); errors["cc"]=max(errors["cc"],abs(float(matrices[1].cell(row,147).value)-float(group.contract_cost_yuan.sum())) )
            for t in range(144): cell_rows.append({"sheet":matrices[1].title,"date":day.date().isoformat(),"slot":t,"target_cell":matrices[1].cell(row,t+2).coordinate,"source":"effective_purchase_kwh","expected_kwh":float(zexp[t]),"readback_kwh":float(z[t]),"abs_error_kwh":float(abs(zexp[t]-z[t]))})
    storage = wb.worksheets[1] if kind == "q4_2" else wb.worksheets[2]
    emergency = wb.worksheets[2] if kind == "q4_2" else wb.worksheets[3]
    if storage.max_row != 1+334*6 or storage.max_column != 6:
        raise AssertionError("Temporary storage table was not expanded into 334 x 6 blocks")
    storage_err=soc_err=0.0
    for di,day in enumerate(FORMAL_DATES):
        g=slots[slots.date.eq(day)].sort_values("slot").reset_index(drop=True); first=2+di*6
        if pd.Timestamp(storage.cell(first,1).value).normalize()!=day: raise AssertionError("Storage date mismatch")
        for bi,(label,start,stop) in enumerate(BLOCKS):
            r=first+bi
            if storage.cell(r,2).value!=label: raise AssertionError("Storage block label mismatch")
            storage_err=max(storage_err,abs(float(storage.cell(r,3).value)-float(g.loc[start:stop-1,"charge_rt_kwh"].sum())),abs(float(storage.cell(r,4).value)-float(g.loc[start:stop-1,"discharge_rt_kwh"].sum())) )
        soc_err=max(soc_err,abs(float(storage.cell(first,6).value)-float(g.loc[0,"soc_start_kwh"])),abs(float(storage.cell(first+1,6).value)-float(g.loc[143,"soc_end_kwh"])))
    actual=[]; current=None
    for r in range(2,emergency.max_row+1):
        if emergency.cell(r,1).value is not None: current=pd.Timestamp(emergency.cell(r,1).value).normalize()
        if emergency.cell(r,2).value is None and emergency.cell(r,3).value is None: continue
        actual.append((current,str(emergency.cell(r,2).value),float(emergency.cell(r,3).value)))
    expected=[(x["date"],x["period"],x["amount_kwh"]) for x in events]
    event_err=max([abs(a[2]-b[2]) for a,b in zip(actual,expected)] or [0.0])
    if len(actual)!=len(expected) or any(a[:2]!=b[:2] for a,b in zip(actual,expected)) or event_err>EPS: raise AssertionError("Emergency event readback mismatch")
    event_sum_err=abs(sum(x[2] for x in actual)-float(slots.emergency_purchase_kwh.sum()))
    if event_sum_err>EPS: raise AssertionError("All-day emergency event sum differs from frozen slots")
    pi=slots.price_yuan_per_kwh; p=slots.original_plan_kwh; z=slots.effective_purchase_kwh; plus=slots.delta_plus_kwh; minus=slots.delta_minus_kwh; e=slots.emergency_purchase_kwh
    contract=pi*p+1.5*pi*plus-0.5*pi*minus; emergency_cost=5*pi*e
    total=contract+emergency_cost
    expected_totals=EXPECTED[kind]
    freeze_errors={"contract_yuan":abs(float(contract.sum())-expected_totals["contract"]),"emergency_yuan":abs(float(emergency_cost.sum())-expected_totals["emergency"]),"total_yuan":abs(float(total.sum())-expected_totals["total"]),"emergency_kwh":abs(float(e.sum())-expected_totals["e"]),"terminal_soc_kwh":abs(float(slots.iloc[-1].soc_end_kwh)-expected_totals["soc"])}
    if max(freeze_errors.values())>5e-4: raise AssertionError(f"Frozen-total mismatch: {freeze_errors}")
    formulas=sum(isinstance(c.value,str) and c.value.startswith("=") for ws in wb.worksheets for row in ws.iter_rows() for c in row)
    result={"kind":kind,"dry_run_file":str(workbook.relative_to(ROOT)),"sheet_order_preserved":True,"formula_count_original":template["template_formula_count"],"formula_count_dry_run":formulas,"formula_count_preserved":formulas==template["template_formula_count"],"plan_cell_max_error_kwh":errors["p"],"effective_cell_max_error_kwh":errors["z"],"daily_plan_total_max_error_kwh":errors["pt"],"daily_effective_total_max_error_kwh":errors["zt"],"daily_plan_cost_max_error_yuan":errors["pc"],"daily_contract_cost_max_error_yuan":errors["cc"],"storage_aggregate_max_error_kwh":storage_err,"soc_writeback_max_error_kwh":soc_err,"emergency_event_count":len(actual),"emergency_event_max_error_kwh":event_err,"emergency_event_total_error_kwh":event_sum_err,"frozen_totals_max_error":max(freeze_errors.values()),"contract_formula_max_error_yuan":float(np.abs(contract-slots.contract_cost_yuan).max()),"emergency_formula_max_error_yuan":float(np.abs(emergency_cost-slots.emergency_cost_yuan).max()),"total_formula_max_error_yuan":float(np.abs(total-slots.actual_total_cost_yuan).max()),"delta_plus_definition_max_error_kwh":float(np.abs(plus-np.maximum(z-p,0)).max()),"delta_minus_definition_max_error_kwh":float(np.abs(minus-np.maximum(p-z,0)).max()),"formal_result_created":False}
    if max(v for k,v in result.items() if k.endswith("error_kwh") or k.endswith("error_yuan"))>EPS: raise AssertionError(f"Dry-run arithmetic mismatch: {result}")
    return result,pd.DataFrame(cell_rows)


def make_docs(a42: dict, a43: dict, v42: dict, v43: dict, protected_before: dict, protected_after: dict) -> None:
    DOCS.mkdir(parents=True,exist_ok=True)
    audit="# 问题四 V1：result4 模板结构审计\n\n"
    audit+="本审查直接读取 `data/raw/result4-2.xlsx` 和 `data/raw/result4-3.xlsx`，未依据 result2/result3 推测结构。两个原始模板均未改写。\n\n"
    for title,data in [("result4-2.xlsx",a42),("result4-3.xlsx",a43)]:
        audit+=f"## {title}\n\nSHA-256：`{data['sha256']}`。工作表顺序："+"、".join('`'+s['name']+'`' for s in data['sheets'])+"。\n\n"
        audit+="| 工作表 | 最大行×列 | 合并单元格 | 公式 | 数据验证 | 冻结窗格 | 隐藏行/列 |\n|---|---:|---|---:|---:|---|---|\n"
        for s in data['sheets']:
            audit+=f"| {s['name']} | {s['max_row']}×{s['max_column']} | {len(s['merged_ranges'])} | {s['formula_count']} | {s['data_validation_count']} | {s['freeze_panes']} | {len(s['hidden_rows'])}/{len(s['hidden_columns'])} |\n"
            samples=", ".join(f"{k}={v}" for k,v in s['format_samples'].items())
            audit+=f"\n格式样本：{samples or '无'}。行高、列宽的完整原始记录保存在 `outputs/q4_v1/stage5/q4_v1_template_audit.json`。\n"
        audit+="\n计划矩阵为第2—335行、B:EO 144列，日期为2025-02-01至2025-12-31。模板表头存在循环式官方列标签（首列为0:10—0:20、末列为次日0:00—0:10）；映射严格按 B:EO 的官方左至右顺序，不对时间索引重排。EP、EQ为模板既有日总量/日费用列。\n\n"
    audit+="`充放电量`和`紧急购电量`原表只预留示例/指定日期行与省略标记，不能同时容纳334天六时段量及全年全部紧急事件。因此演练副本按官方既有字段扩展数据体行；不新增工作表、不改表头、不改列序。这是正式导出前需确认的模板呈现解释。\n\n"
    audit+=f"演练回读：4-2/4-3均通过，文件分别为`{v42['dry_run_file']}`、`{v43['dry_run_file']}`；保护哈希变化数为 {sum(protected_before[k]!=protected_after[k] for k in protected_before)}。\n"
    (DOCS/"q4_v1_result4_template_audit.md").write_text(audit,encoding="utf-8")
    contract="# 问题四 V1：result4 字段映射契约（阶段5A冻结草案）\n\n"
    contract+="## 统一规则\n\n- 所有写入量均为每10分钟槽 **kWh**；不写入kW。官方B:EO列按既有左至右顺序对应内部槽0—143。\n- 充放电表 C/D 分别为实际外部充电量/对外放电量的**非负幅值**；效率仅在SOC递推中体现，不在写入时二次折算。F列仅按模板的0:00与24:00标记写入实际SOC。\n- 紧急购电按`e>1e-6 kWh`分日连续槽合并；相邻正槽同一事件，非连续或跨日期必分开；区间右端使用最后槽模板标签的右端，即右开表示。跨23:50—24:00的事件在当日结束，不与下一日0:00后的槽合并。无事件日期不新增零值行。\n\n"
    contract+="## 问题4-2\n\n`计划购电量` B:EO 写每日0:00锁定的 `original_plan_kwh=p`；EP写日计划量，EQ写实际目标槽价格下的 Σπp。费用总账不直接写入模板，因为模板没有全年费用字段，审计中独立复算 Σ(πp+5πe)。\n\n"
    contract+="## 问题4-3\n\n`计划购电量` B:EO 写0:00原计划 `p`；`调整购电量` B:EO 写槽执行前最后有效版本的最终生效量 `z`，**不写** Δ+−Δ−，也不累计6:00/12:00/18:00的中间版本。调整表EP写日Σz，EQ写每槽仅相对p结算一次的日合同费用Σ[πp+1.5πΔ+−0.5πΔ−]。\n\n"
    contract+="实际账单复算只使用附件4目标槽实际价格；预测价格、终端价值v=0.6和库存修正费用不进入模板账单。\n\n"
    contract+="## 尚存需人工确认的题面/模板呈现点\n\n模板没有单独的上调、下调或费用汇总表字段；本契约将“调整购电量”解释为最终生效量z，并将其日费用列解释为一次性合同结算。该解释与问题3冻结口径一致，但仍应在正式导出授权时确认。\n"
    (DOCS/"q4_v1_result4_mapping_contract.md").write_text(contract,encoding="utf-8")


def main() -> None:
    if (ROOT/"outputs"/"tables"/"result4-2.xlsx").exists() or (ROOT/"outputs"/"tables"/"result4-3.xlsx").exists():
        raise RuntimeError("Formal result4 output already exists; stage5A refuses to overwrite it")
    before=hashes()
    a42=assert_template(TEMPLATE_42,["计划购电量","充放电量","紧急购电量"])
    a43=assert_template(TEMPLATE_43,["计划购电量","调整购电量","充放电量","紧急购电量"])
    s42,s43=load_slots("q4_2"),load_slots("q4_3")
    f42,e42,m42=write_dry_run("q4_2",s42,a42); f43,e43,m43=write_dry_run("q4_3",s43,a43)
    v42,c42=validate_dry_run("q4_2",s42,a42,f42,e42); v43,c43=validate_dry_run("q4_3",s43,a43,f43,e43)
    m42.to_csv(STAGE5/"q4_v1_result4_2_cell_mapping.csv",index=False,encoding="utf-8-sig")
    m43.to_csv(STAGE5/"q4_v1_result4_3_cell_mapping.csv",index=False,encoding="utf-8-sig")
    pd.concat([pd.DataFrame(e42).assign(policy="q4_2"),pd.DataFrame(e43).assign(policy="q4_3")],ignore_index=True).to_csv(STAGE5/"q4_v1_emergency_event_check.csv",index=False,encoding="utf-8-sig")
    after=hashes()
    if before!=after: raise RuntimeError("Protected hashes changed during stage5A; stopping")
    make_docs(a42,a43,v42,v43,before,after)
    lines=["Q4 V1 Stage5A dry-run validation", "="*38, "Scope: original templates read only; no outputs/tables/result4-2.xlsx or result4-3.xlsx was created."]
    for name,v in [("q4-2",v42),("q4-3",v43)]:
        lines.append(f"{name}: plan/z cells max error={v['plan_cell_max_error_kwh']:.3e}/{v['effective_cell_max_error_kwh']:.3e} kWh; storage={v['storage_aggregate_max_error_kwh']:.3e}; SOC={v['soc_writeback_max_error_kwh']:.3e}; events={v['emergency_event_count']}, event sum error={v['emergency_event_total_error_kwh']:.3e}; frozen totals max error={v['frozen_totals_max_error']:.3e}.")
        x=EXPECTED[name.replace("-", "_")]
        lines.append(f"{name} frozen totals verified: contract={x['contract']:.3f} yuan; emergency={x['emergency']:.3f} yuan; total={x['total']:.3f} yuan; emergency volume={x['e']:.3f} kWh; terminal SOC={x['soc']:.3f} kWh.")
    lines += [f"template hashes: result4-2={a42['sha256']}; result4-3={a43['sha256']}", "protected hash changes=0", "formal export readiness: CONDITIONAL PASS — technical dry run passed; final authorisation should confirm the z interpretation and the intentional expansion of example-body rows."]
    (STAGE5/"q4_v1_dry_run_validation.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (STAGE5/"q4_v1_template_audit.json").write_text(json.dumps({"result4_2":a42,"result4_3":a43,"dry_validation":{"q4_2":v42,"q4_3":v43},"protected_before":before,"protected_after":after},ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print("STAGE5A_DRY_RUN_PASS")


if __name__=="__main__":
    main()
