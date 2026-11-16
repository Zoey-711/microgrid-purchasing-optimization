"""Stage-5 Q2-v2 submission exporter.

This exporter never solves a model.  It writes only the frozen main_dp50 slot
trajectory to a copy of the official template, validates that copy, and only
then replaces the authorised legacy submission workbook.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import copy
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .config import DELTA_T_HOURS, ENERGY_MAX_KWH, ETA_CHARGE, ETA_DISCHARGE, ROOT, SOC_MAX_KWH, SOC_MIN_KWH, TOL


RAW_TEMPLATE = ROOT / "data/raw/result2.xlsx"
LEGACY_RESULT = ROOT / "outputs/tables/result2.xlsx"
ARCHIVE_DIR = ROOT / "outputs/q2_v2/archive"
LEGACY_ARCHIVE = ARCHIVE_DIR / "result2_legacy_q2.xlsx"
STAGE4 = ROOT / "outputs/q2_v2/stage4"
SLOTS_PATH = STAGE4 / "q2_v2_slot_results.csv"
CANDIDATE = ROOT / "outputs/q2_v2/stage5/result2_v2_candidate.xlsx"
AUDIT_PATH = ROOT / "docs/q2_v2/q2_v2_export_audit.md"
ARCHIVE_RECORD = ROOT / "docs/q2_v2/q2_v2_legacy_result2_archive_record.md"
FREEZE_RECORD = ROOT / "docs/q2_v2/q2_v2_strategy_freeze_record.md"

SHEET_NAMES = ["计划购电量", "充放电量", "紧急购电量"]
FORMAL_DATES = pd.date_range("2025-02-01", "2025-12-31", freq="D")
BLOCKS = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
          ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_slots() -> pd.DataFrame:
    frame = pd.read_csv(SLOTS_PATH)
    frame = frame.loc[frame["policy_id"].eq("main_dp50")].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.loc[frame["date"].isin(FORMAL_DATES)].sort_values(["date", "slot"]).reset_index(drop=True)
    if len(frame) != 334 * 144 or not frame.groupby("date").size().eq(144).all():
        raise ValueError("Frozen main_dp50 slot source is incomplete")
    if frame["slot"].astype(int).groupby(frame["date"]).apply(lambda x: list(x) == list(range(144))).all() is False:
        raise ValueError("Frozen slot indices are not 0..143 for every official day")
    return frame


def assert_raw_template() -> dict:
    workbook = load_workbook(RAW_TEMPLATE, data_only=False)
    if workbook.sheetnames != SHEET_NAMES:
        raise ValueError(f"Unexpected template sheets: {workbook.sheetnames}")
    plan = workbook.worksheets[0]
    template_dates = [pd.Timestamp(plan.cell(row, 1).value).normalize() for row in range(2, 336)]
    if template_dates != list(FORMAL_DATES):
        raise ValueError("Template plan-sheet dates are not the official 334 dates")
    labels = [str(plan.cell(1, col).value) for col in range(2, 146)]
    if len(labels) != 144 or any(v is None for v in labels):
        raise ValueError("Template has no valid 144-slot header B:EO")
    if any(plan.cell(row, col).value is not None for row in range(2, 336) for col in range(2, 148)):
        raise ValueError("Raw template plan data area is not empty")
    formula_count = sum(
        1 for ws in workbook.worksheets for row in ws.iter_rows() for cell in row
        if isinstance(cell.value, str) and cell.value.startswith("=")
    )
    return {"labels": labels, "template_formula_count": formula_count, "raw_template_sha256": sha256(RAW_TEMPLATE)}


def archive_legacy() -> dict:
    if not LEGACY_RESULT.exists():
        raise FileNotFoundError(f"Authorised legacy result missing: {LEGACY_RESULT}")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    before = sha256(LEGACY_RESULT)
    shutil.copyfile(LEGACY_RESULT, LEGACY_ARCHIVE)
    copied = sha256(LEGACY_ARCHIVE)
    if before != copied:
        raise RuntimeError("Legacy archive hash mismatch; export is stopped before overwrite")
    record = {
        "archive_time": datetime.now().isoformat(timespec="seconds"),
        "legacy_original_path": str(LEGACY_RESULT.relative_to(ROOT)),
        "archive_path": str(LEGACY_ARCHIVE.relative_to(ROOT)),
        "legacy_original_sha256": before,
        "archive_sha256": copied,
        "statement": "旧问题二结果已被 V2 替代为正式提交版本；原字节副本已归档。",
    }
    ARCHIVE_RECORD.write_text(
        "# 旧问题二 result2.xlsx 归档记录\n\n```json\n" + json.dumps(record, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return record


def event_records(slots: pd.DataFrame, labels: list[str]) -> list[dict]:
    records: list[dict] = []
    for day, day_slots in slots.groupby("date", sort=True):
        day_slots = day_slots.sort_values("slot").reset_index(drop=True)
        positive = day_slots.loc[day_slots["emergency_purchase_kwh"] > TOL, "slot"].astype(int).tolist()
        if not positive:
            continue
        start = prev = positive[0]
        groups: list[tuple[int, int]] = []
        for slot in positive[1:]:
            if slot == prev + 1:
                prev = slot
            else:
                groups.append((start, prev))
                start = prev = slot
        groups.append((start, prev))
        for left_slot, right_slot in groups:
            left = labels[left_slot].split("-", 1)[0]
            right = labels[right_slot].split("-", 1)[1]
            amount = float(day_slots.loc[left_slot:right_slot, "emergency_purchase_kwh"].sum())
            if amount <= TOL:
                raise ValueError("A written emergency event must be strictly positive")
            records.append({"date": pd.Timestamp(day), "start_slot": left_slot, "end_slot": right_slot,
                            "period": f"{left}-{right}", "amount_kwh": amount})
    return records


def copy_style(source, target) -> None:
    target._style = copy(source._style)
    if source.number_format:
        target.number_format = source.number_format
    if source.alignment:
        target.alignment = copy(source.alignment)
    if source.font:
        target.font = copy(source.font)
    if source.fill:
        target.fill = copy(source.fill)
    if source.border:
        target.border = copy(source.border)


def write_candidate(slots: pd.DataFrame, template: dict) -> list[dict]:
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RAW_TEMPLATE, CANDIDATE)
    wb = load_workbook(CANDIDATE, data_only=False)
    if wb.sheetnames != SHEET_NAMES:
        raise ValueError("Copied template sheet structure changed")
    plan, storage, emergency = wb.worksheets
    labels = template["labels"]

    for row, day in enumerate(FORMAL_DATES, start=2):
        d = slots.loc[slots["date"].eq(day)].sort_values("slot").reset_index(drop=True)
        values = d["planned_purchase_kwh"].to_numpy(dtype=float)
        for col, value in enumerate(values, start=2):
            plan.cell(row, col).value = float(value)
        plan.cell(row, 146).value = float(values.sum())
        plan.cell(row, 147).value = float(d["plan_cost_yuan"].sum())

    storage_style = [[copy(storage.cell(row, col)._style) for col in range(1, 7)] for row in range(2, 8)]
    storage_height = [storage.row_dimensions[row].height for row in range(2, 8)]
    storage.delete_rows(2, storage.max_row - 1)
    storage.insert_rows(2, 334 * 6)
    for day_idx, day in enumerate(FORMAL_DATES):
        d = slots.loc[slots["date"].eq(day)].sort_values("slot").reset_index(drop=True)
        for block_idx, (label, start, stop) in enumerate(BLOCKS):
            row = 2 + day_idx * 6 + block_idx
            for col in range(1, 7):
                storage.cell(row, col)._style = copy(storage_style[block_idx][col - 1])
            storage.row_dimensions[row].height = storage_height[block_idx]
            storage.cell(row, 1).value = day.to_pydatetime() if block_idx == 0 else None
            storage.cell(row, 2).value = label
            # Template columns separate positive magnitudes: C=charge, D=discharge.
            storage.cell(row, 3).value = float(d.loc[start:stop - 1, "charge_kwh"].sum())
            storage.cell(row, 4).value = float(d.loc[start:stop - 1, "discharge_kwh"].sum())
            storage.cell(row, 5).value = time(0, 0) if block_idx == 0 else ("24:00" if block_idx == 1 else None)
            storage.cell(row, 6).value = (
                float(d.loc[0, "soc_start_kwh"]) if block_idx == 0 else
                (float(d.loc[143, "soc_end_kwh"]) if block_idx == 1 else None)
            )

    events = event_records(slots, labels)
    event_style = [copy(emergency.cell(2, col)._style) for col in range(1, 4)]
    event_height = emergency.row_dimensions[2].height
    emergency.delete_rows(2, emergency.max_row - 1)
    if events:
        emergency.insert_rows(2, len(events))
        last_date = None
        for row, event in enumerate(events, start=2):
            for col in range(1, 4):
                emergency.cell(row, col)._style = copy(event_style[col - 1])
            emergency.row_dimensions[row].height = event_height
            emergency.cell(row, 1).value = event["date"].to_pydatetime() if event["date"] != last_date else None
            emergency.cell(row, 2).value = event["period"]
            emergency.cell(row, 3).value = event["amount_kwh"]
            last_date = event["date"]
    wb.save(CANDIDATE)
    return events


def validate_candidate(slots: pd.DataFrame, template: dict, expected_events: list[dict]) -> dict:
    wb = load_workbook(CANDIDATE, data_only=False)
    if wb.sheetnames != SHEET_NAMES:
        raise AssertionError("Worksheet names changed")
    plan, storage, emergency = wb.worksheets
    formula_count = sum(1 for ws in wb.worksheets for row in ws.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("="))
    if formula_count != template["template_formula_count"]:
        raise AssertionError("Formula count differs from copied template")
    dates = [pd.Timestamp(plan.cell(row, 1).value).normalize() for row in range(2, 336)]
    if dates != list(FORMAL_DATES):
        raise AssertionError("Plan dates changed")
    if [str(plan.cell(1, col).value) for col in range(2, 146)] != template["labels"]:
        raise AssertionError("Official slot headers changed")

    plan_err = total_err = plan_cost_err = 0.0
    plan_blank = 0
    for row, day in enumerate(FORMAL_DATES, start=2):
        d = slots.loc[slots["date"].eq(day)].sort_values("slot").reset_index(drop=True)
        out = np.array([plan.cell(row, col).value for col in range(2, 146)], dtype=float)
        plan_blank += int(sum(plan.cell(row, col).value is None for col in range(2, 146)))
        expected = d["planned_purchase_kwh"].to_numpy(dtype=float)
        plan_err = max(plan_err, float(np.max(np.abs(out - expected))))
        total_err = max(total_err, abs(float(plan.cell(row, 146).value) - expected.sum()))
        plan_cost_err = max(plan_cost_err, abs(float(plan.cell(row, 147).value) - float(d["plan_cost_yuan"].sum())))
    if plan_blank or plan_err > TOL or total_err > TOL or plan_cost_err > TOL:
        raise AssertionError("Plan-sheet writeback mismatch")

    if storage.max_row != 1 + 334 * 6 or storage.max_column != 6:
        raise AssertionError("Storage-sheet dimensions mismatch")
    storage_err = soc_export_err = 0.0
    for day_idx, day in enumerate(FORMAL_DATES):
        d = slots.loc[slots["date"].eq(day)].sort_values("slot").reset_index(drop=True)
        first = 2 + day_idx * 6
        if pd.Timestamp(storage.cell(first, 1).value).normalize() != day:
            raise AssertionError("Storage date-block mapping mismatch")
        for block_idx, (label, start, stop) in enumerate(BLOCKS):
            row = first + block_idx
            if storage.cell(row, 2).value != label:
                raise AssertionError("Storage four-hour label mismatch")
            storage_err = max(storage_err, abs(float(storage.cell(row, 3).value) - float(d.loc[start:stop - 1, "charge_kwh"].sum())))
            storage_err = max(storage_err, abs(float(storage.cell(row, 4).value) - float(d.loc[start:stop - 1, "discharge_kwh"].sum())))
        soc_export_err = max(soc_export_err, abs(float(storage.cell(first, 6).value) - float(d.loc[0, "soc_start_kwh"])))
        soc_export_err = max(soc_export_err, abs(float(storage.cell(first + 1, 6).value) - float(d.loc[143, "soc_end_kwh"])))
    if storage_err > TOL or soc_export_err > TOL:
        raise AssertionError("Storage-sheet writeback mismatch")

    out_events = []
    current_date = None
    for row in range(2, emergency.max_row + 1):
        value = emergency.cell(row, 3).value
        period = emergency.cell(row, 2).value
        date_value = emergency.cell(row, 1).value
        if date_value is not None:
            current_date = pd.Timestamp(date_value).normalize()
        if value is None and period is None:
            continue
        if current_date is None or period is None or value is None or float(value) <= TOL:
            raise AssertionError("Invalid or zero emergency event record")
        out_events.append((current_date, str(period), float(value)))
    expected = [(x["date"], x["period"], x["amount_kwh"]) for x in expected_events]
    if len(out_events) != len(expected):
        raise AssertionError("Emergency-event count mismatch")
    event_err = max((abs(a[2] - b[2]) for a, b in zip(out_events, expected)), default=0.0)
    if any(a[0] != b[0] or a[1] != b[1] for a, b in zip(out_events, expected)) or event_err > TOL:
        raise AssertionError("Emergency-event mapping mismatch")
    event_sum_err = abs(sum(x[2] for x in out_events) - float(slots["emergency_purchase_kwh"].sum()))
    if event_sum_err > TOL:
        raise AssertionError("Emergency aggregate mismatch")

    # Costs and physical equations are independently recomputed from the frozen
    # per-slot records; the workbook is read back for plan and event equality.
    plan_cost = float((slots["price_yuan_per_kwh"] * slots["planned_purchase_kwh"]).sum())
    emergency_cost = float((5 * slots["price_yuan_per_kwh"] * slots["emergency_purchase_kwh"]).sum())
    total_cost = plan_cost + emergency_cost
    balance = slots["planned_purchase_kwh"] + slots["pv_actual_kwh"] + slots["discharge_kwh"] + slots["emergency_purchase_kwh"] - slots["load_actual_kwh"] - slots["charge_kwh"] - slots["spill_kwh"]
    transition = slots["soc_end_kwh"] - (slots["soc_start_kwh"] + ETA_CHARGE * slots["charge_kwh"] - slots["discharge_kwh"] / ETA_DISCHARGE)
    starts = slots.groupby("date").first()["soc_start_kwh"].to_numpy()
    ends = slots.groupby("date").last()["soc_end_kwh"].to_numpy()
    cross_day = np.max(np.abs(starts[1:] - ends[:-1]))
    soc_all = np.r_[slots["soc_start_kwh"].to_numpy(), slots["soc_end_kwh"].to_numpy()]
    checks = {
        "plan_cost_yuan": plan_cost,
        "emergency_cost_yuan": emergency_cost,
        "actual_total_cost_yuan": total_cost,
        "emergency_purchase_kwh": float(slots["emergency_purchase_kwh"].sum()),
        "plan_cell_max_error_kwh": plan_err,
        "daily_plan_total_max_error_kwh": total_err,
        "daily_plan_cost_max_error_yuan": plan_cost_err,
        "storage_aggregate_max_error_kwh": storage_err,
        "exported_soc_max_error_kwh": soc_export_err,
        "emergency_event_max_error_kwh": event_err,
        "emergency_event_sum_error_kwh": event_sum_err,
        "energy_balance_max_residual_kwh": float(np.max(np.abs(balance))),
        "soc_transition_max_residual_kwh": float(np.max(np.abs(transition))),
        "cross_day_soc_max_residual_kwh": float(cross_day),
        "soc_boundary_violations": int(((soc_all < SOC_MIN_KWH - TOL) | (soc_all > SOC_MAX_KWH + TOL)).sum()),
        "power_violations": int(((slots["charge_kwh"] > ENERGY_MAX_KWH + TOL) | (slots["discharge_kwh"] > ENERGY_MAX_KWH + TOL)).sum()),
        "simultaneous_charge_discharge_slots": int(((slots["charge_kwh"] > TOL) & (slots["discharge_kwh"] > TOL)).sum()),
        "emergency_while_charging_slots": int(((slots["emergency_purchase_kwh"] > TOL) & (slots["charge_kwh"] > TOL)).sum()),
        "formula_count_preserved": formula_count == template["template_formula_count"],
        "official_structure_preserved": True,
        "plan_empty_cells": plan_blank,
        "event_count": len(out_events),
    }
    if any(checks[k] > TOL for k in ("energy_balance_max_residual_kwh", "soc_transition_max_residual_kwh", "cross_day_soc_max_residual_kwh")):
        raise AssertionError("Physical recomputation failed")
    if any(checks[k] != 0 for k in ("soc_boundary_violations", "power_violations", "simultaneous_charge_discharge_slots", "emergency_while_charging_slots")):
        raise AssertionError("Physical constraint violation")
    return checks


def run() -> None:
    template = assert_raw_template()
    archive = archive_legacy()
    if sha256(RAW_TEMPLATE) != template["raw_template_sha256"]:
        raise RuntimeError("Raw template hash changed before export; overwrite is refused")
    slots = source_slots()
    events = write_candidate(slots, template)
    checks = validate_candidate(slots, template, events)
    # Copy only after all candidate checks pass.
    shutil.copy2(CANDIDATE, LEGACY_RESULT)
    final_hash = sha256(LEGACY_RESULT)
    if final_hash != sha256(CANDIDATE):
        raise RuntimeError("Final copy hash mismatch")
    raw_after = sha256(RAW_TEMPLATE)
    if raw_after != template["raw_template_sha256"]:
        raise RuntimeError("Raw template was changed during export")
    audit = {
        "export_time": datetime.now().isoformat(timespec="seconds"),
        "strategy_display_name": "历史联合残差情景—动态价值调度策略",
        "implementation_identifier": "main_dp50 (SOC grid = 50 kWh)",
        "source": str(SLOTS_PATH.relative_to(ROOT)),
        "period": ["2025-02-01", "2025-12-31"],
        "days": 334,
        "slots": 334 * 144,
        "archive": archive,
        "raw_template_sha256_before": template["raw_template_sha256"],
        "raw_template_sha256_after": raw_after,
        "exported_result2_sha256": final_hash,
        "candidate_sha256": sha256(CANDIDATE),
        "charge_discharge_convention": "模板 C、D 列分别记录实际充电量、实际放电量的非负 kWh 幅值；不采用正负号混写。",
        "emergency_event_convention": "只记录实际发生的正紧急购电事件；相邻正槽合并，数量为合并槽的实际 kWh 之和。",
        "checks": checks,
        "result2_exported": True,
        "protected_scope_note": "旧 outputs/tables/result2.xlsx 已先按授权归档；data/raw/result2.xlsx 未修改。",
    }
    AUDIT_PATH.write_text(
        "# 问题二 V2 正式模板导出审计\n\n```json\n" + json.dumps(audit, ensure_ascii=False, indent=2) + "\n```\n\n"
        "## 复核说明\n\n"
        "计划购电量表按官方 B:EO 的 144 列顺序逐槽写入；EP、EQ 分别写入日计划购电量和日计划购电费。充放电量表按六个四小时块扩展至 334 天，C、D 列分别为实际充电、放电的非负 kWh 幅值。紧急购电量表仅记录正事件，并合并相邻连续槽。\n\n"
        "费用复算以冻结逐槽记录的价格、计划购电量和紧急购电量独立进行；导出工作簿反读后逐单元格验证计划、四小时充放电汇总、首末 SOC 和紧急购电事件。因合并事件跨槽价格可能不同，紧急费用不能仅由事件总量反推，故以逐槽冻结源进行费用复算并以事件总量核对。\n",
        encoding="utf-8",
    )
    # Amend the freeze record with the authorised export transition without
    # altering its frozen model/result hashes.
    with FREEZE_RECORD.open("a", encoding="utf-8") as f:
        f.write("\n## 正式模板导出\n\n")
        f.write(f"已按授权以‘历史联合残差情景—动态价值调度策略’导出 `{LEGACY_RESULT.relative_to(ROOT)}`。")
        f.write(f"旧文件已字节归档至 `{LEGACY_ARCHIVE.relative_to(ROOT)}`，归档前后 SHA-256 均为 `{archive['legacy_original_sha256']}`；")
        f.write(f"新提交文件 SHA-256 为 `{final_hash}`。详见 `docs/q2_v2/q2_v2_export_audit.md`。\n")


if __name__ == "__main__":
    run()
