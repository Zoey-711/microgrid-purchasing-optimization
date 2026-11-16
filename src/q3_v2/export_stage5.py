"""Question 3 V2 stage-5 template export.

This module never solves or replays the model.  It reads the frozen Stage-4
A-policy slot trajectory, writes a candidate copy of the official template,
validates that copy, and only then writes ``outputs/tables/result3.xlsx``.
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


ROOT = Path(__file__).resolve().parents[2]
RAW_TEMPLATE = ROOT / "data" / "raw" / "result3.xlsx"
FINAL_FILE = ROOT / "outputs" / "tables" / "result3.xlsx"
STAGE4 = ROOT / "outputs" / "q3_v2" / "stage4"
STAGE5 = ROOT / "outputs" / "q3_v2" / "stage5"
DOCS = ROOT / "docs" / "q3_v2"
CANDIDATE = STAGE5 / "result3_candidate.xlsx"
SLOTS = STAGE4 / "q3_v2_slot_results.csv"
COMPARISON = STAGE4 / "q3_v2_full_year_comparison.csv"
PROTECTED = STAGE4 / "protected_hashes_after_stage4.json"

POLICY_A = "A_all_updates_dynamic_value"
POLICY_B = "B_no_adjustment_dynamic_value"
SHEET_NAMES = ["计划购电量", "调整购电量", "充放电量", "紧急购电量"]
FORMAL_DATES = pd.date_range("2025-02-01", "2025-12-31", freq="D")
BLOCKS = [
    ("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48),
    ("8:00-12:00", 48, 72), ("12:00-16:00", 72, 96),
    ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144),
]
EPS = 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def normalise_nonnegative(value: float, *, name: str) -> float:
    """Accept only a documented solver-level negative roundoff."""
    value = float(value)
    if -EPS <= value < 0:
        return 0.0
    if value < -EPS:
        raise ValueError(f"{name} has a real negative value: {value:.12g}")
    return value


def protected_hashes() -> dict[str, str]:
    manifest = json.loads(PROTECTED.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for relative, expected in manifest.items():
        path = ROOT / Path(relative)
        if not path.exists():
            mismatches.append(f"missing: {relative}")
        elif sha256(path) != expected:
            mismatches.append(f"hash mismatch: {relative}")
    if mismatches:
        raise RuntimeError("Protected-file check failed before export: " + "; ".join(mismatches))
    return manifest


def assert_template() -> dict:
    wb = load_workbook(RAW_TEMPLATE, data_only=False)
    if wb.sheetnames != SHEET_NAMES:
        raise RuntimeError(f"Unexpected worksheet structure: {wb.sheetnames}")
    plan, adjusted, storage, emergency = wb.worksheets
    labels = [str(plan.cell(1, col).value) for col in range(2, 146)]
    if len(labels) != 144 or labels != [str(adjusted.cell(1, col).value) for col in range(2, 146)]:
        raise RuntimeError("Official 144-slot headers are missing or inconsistent")
    dates = [pd.Timestamp(plan.cell(row, 1).value).normalize() for row in range(2, 336)]
    if dates != list(FORMAL_DATES):
        raise RuntimeError("Official plan-sheet dates are not 2025-02-01..2025-12-31")
    if any(plan.cell(row, col).value is not None or adjusted.cell(row, col).value is not None
           for row in range(2, 336) for col in range(2, 146)):
        raise RuntimeError("Raw template matrix is not empty; export refused")
    formulas = sum(
        isinstance(cell.value, str) and cell.value.startswith("=")
        for ws in wb.worksheets for row in ws.iter_rows() for cell in row
    )
    return {"labels": labels, "formula_count": formulas, "raw_sha256": sha256(RAW_TEMPLATE),
            "storage_template_rows": storage.max_row, "emergency_template_rows": emergency.max_row}


def load_a_slots() -> pd.DataFrame:
    frame = pd.read_csv(SLOTS)
    frame = frame.loc[frame["policy"].eq(POLICY_A)].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.loc[frame["date"].isin(FORMAL_DATES)].sort_values(["date", "slot"]).reset_index(drop=True)
    if len(frame) != len(FORMAL_DATES) * 144 or not frame.groupby("date").size().eq(144).all():
        raise RuntimeError("Frozen A-policy source does not contain 334 complete days")
    if not all(day["slot"].astype(int).tolist() == list(range(144)) for _, day in frame.groupby("date", sort=True)):
        raise RuntimeError("Frozen A-policy slots are not exactly 0..143")
    for column in ("original_plan_kwh", "effective_purchase_kwh", "charge_rt_kwh", "discharge_rt_kwh",
                   "emergency_purchase_kwh", "spill_kwh", "delta_plus_kwh", "delta_minus_kwh"):
        frame[column] = [normalise_nonnegative(v, name=column) for v in frame[column]]
    return frame


def event_records(slots: pd.DataFrame, labels: list[str]) -> list[dict]:
    events: list[dict] = []
    for day, group in slots.groupby("date", sort=True):
        group = group.sort_values("slot").reset_index(drop=True)
        positive = group.loc[group["emergency_purchase_kwh"] > EPS, "slot"].astype(int).tolist()
        if not positive:
            continue
        start = previous = positive[0]
        runs: list[tuple[int, int]] = []
        for slot in positive[1:]:
            if slot == previous + 1:
                previous = slot
            else:
                runs.append((start, previous))
                start = previous = slot
        runs.append((start, previous))
        for left, right in runs:
            start_text = labels[left].split("-", 1)[0]
            end_text = labels[right].split("-", 1)[1]
            amount = float(group.loc[left:right, "emergency_purchase_kwh"].sum())
            if amount <= EPS:
                raise RuntimeError("Non-positive emergency event would be written")
            events.append({"date": pd.Timestamp(day), "start_slot": left, "end_slot": right,
                           "period": f"{start_text}-{end_text}", "amount_kwh": amount})
    return events


def copy_cell_style(source, target) -> None:
    target._style = copy(source._style)
    target.number_format = source.number_format
    target.alignment = copy(source.alignment)
    target.font = copy(source.font)
    target.fill = copy(source.fill)
    target.border = copy(source.border)


def write_candidate(slots: pd.DataFrame, template: dict) -> list[dict]:
    STAGE5.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RAW_TEMPLATE, CANDIDATE)
    wb = load_workbook(CANDIDATE, data_only=False)
    plan, adjusted, storage, emergency = wb.worksheets
    labels = template["labels"]

    for row, day in enumerate(FORMAL_DATES, start=2):
        group = slots.loc[slots["date"].eq(day)].sort_values("slot").reset_index(drop=True)
        original = group["original_plan_kwh"].to_numpy(float)
        effective = group["effective_purchase_kwh"].to_numpy(float)
        for col, value in enumerate(original, start=2):
            plan.cell(row, col).value = float(value)
        for col, value in enumerate(effective, start=2):
            adjusted.cell(row, col).value = float(value)
        # The two existing total columns retain their template role.  For the
        # adjustment sheet they refer to final-effective quantity and the one-
        # time contract settlement, respectively.
        plan.cell(row, 146).value = float(original.sum())
        plan.cell(row, 147).value = float((group["price_yuan_per_kwh"] * original).sum())
        adjusted.cell(row, 146).value = float(effective.sum())
        adjusted.cell(row, 147).value = float(group["contract_cost_yuan"].sum())

    storage_styles = [[copy(storage.cell(row, col)._style) for col in range(1, 7)] for row in range(2, 8)]
    storage_heights = [storage.row_dimensions[row].height for row in range(2, 8)]
    storage.delete_rows(2, storage.max_row - 1)
    storage.insert_rows(2, len(FORMAL_DATES) * 6)
    for day_idx, day in enumerate(FORMAL_DATES):
        group = slots.loc[slots["date"].eq(day)].sort_values("slot").reset_index(drop=True)
        first = 2 + day_idx * 6
        for block_idx, (label, start, stop) in enumerate(BLOCKS):
            row = first + block_idx
            for col in range(1, 7):
                storage.cell(row, col)._style = copy(storage_styles[block_idx][col - 1])
            storage.row_dimensions[row].height = storage_heights[block_idx]
            storage.cell(row, 1).value = day.to_pydatetime() if block_idx == 0 else None
            storage.cell(row, 2).value = label
            # C/D are non-negative actual kWh magnitudes; the sign convention
            # is documented in the export audit rather than mixed in one cell.
            storage.cell(row, 3).value = float(group.loc[start:stop - 1, "charge_rt_kwh"].sum())
            storage.cell(row, 4).value = float(group.loc[start:stop - 1, "discharge_rt_kwh"].sum())
            storage.cell(row, 5).value = time(0, 0) if block_idx == 0 else ("24:00" if block_idx == 1 else None)
            storage.cell(row, 6).value = (float(group.loc[0, "soc_start_kwh"]) if block_idx == 0 else
                                          (float(group.loc[143, "soc_end_kwh"]) if block_idx == 1 else None))

    events = event_records(slots, labels)
    event_styles = [copy(emergency.cell(2, col)._style) for col in range(1, 4)]
    event_height = emergency.row_dimensions[2].height
    emergency.delete_rows(2, emergency.max_row - 1)
    if events:
        emergency.insert_rows(2, len(events))
        last_date = None
        for row, event in enumerate(events, start=2):
            for col in range(1, 4):
                emergency.cell(row, col)._style = copy(event_styles[col - 1])
            emergency.row_dimensions[row].height = event_height
            emergency.cell(row, 1).value = event["date"].to_pydatetime() if event["date"] != last_date else None
            emergency.cell(row, 2).value = event["period"]
            emergency.cell(row, 3).value = event["amount_kwh"]
            last_date = event["date"]
    wb.save(CANDIDATE)
    return events


def validate_candidate(slots: pd.DataFrame, template: dict, expected_events: list[dict], workbook_path: Path = CANDIDATE) -> tuple[dict, pd.DataFrame]:
    wb = load_workbook(workbook_path, data_only=False)
    if wb.sheetnames != SHEET_NAMES:
        raise AssertionError("Candidate worksheet names differ from official template")
    plan, adjusted, storage, emergency = wb.worksheets
    formula_count = sum(isinstance(c.value, str) and c.value.startswith("=") for ws in wb.worksheets for row in ws.iter_rows() for c in row)
    if formula_count != template["formula_count"]:
        raise AssertionError("Template formulas changed")
    labels = [str(plan.cell(1, col).value) for col in range(2, 146)]
    if labels != template["labels"] or labels != [str(adjusted.cell(1, col).value) for col in range(2, 146)]:
        raise AssertionError("Official headers changed")

    checks: list[dict] = []
    p_err = z_err = p_total_err = z_total_err = p_cost_err = contract_err = 0.0
    blanks = 0
    for row, day in enumerate(FORMAL_DATES, start=2):
        group = slots.loc[slots["date"].eq(day)].sort_values("slot").reset_index(drop=True)
        p_out = np.array([plan.cell(row, col).value for col in range(2, 146)], dtype=float)
        z_out = np.array([adjusted.cell(row, col).value for col in range(2, 146)], dtype=float)
        p_exp = group["original_plan_kwh"].to_numpy(float)
        z_exp = group["effective_purchase_kwh"].to_numpy(float)
        p_err = max(p_err, float(np.max(np.abs(p_out - p_exp))))
        z_err = max(z_err, float(np.max(np.abs(z_out - z_exp))))
        blanks += int(sum(plan.cell(row, col).value is None or adjusted.cell(row, col).value is None for col in range(2, 146)))
        p_total_err = max(p_total_err, abs(float(plan.cell(row, 146).value) - float(p_exp.sum())))
        z_total_err = max(z_total_err, abs(float(adjusted.cell(row, 146).value) - float(z_exp.sum())) )
        p_cost_err = max(p_cost_err, abs(float(plan.cell(row, 147).value) - float((group["price_yuan_per_kwh"] * p_exp).sum())))
        contract_err = max(contract_err, abs(float(adjusted.cell(row, 147).value) - float(group["contract_cost_yuan"].sum())))
        checks.extend({"sheet": "计划购电量", "date": day.date().isoformat(), "slot": int(slot), "expected_kwh": float(p_exp[slot]), "readback_kwh": float(p_out[slot]), "abs_error_kwh": float(abs(p_out[slot] - p_exp[slot]))} for slot in range(144))
        checks.extend({"sheet": "调整购电量（最终生效量）", "date": day.date().isoformat(), "slot": int(slot), "expected_kwh": float(z_exp[slot]), "readback_kwh": float(z_out[slot]), "abs_error_kwh": float(abs(z_out[slot] - z_exp[slot]))} for slot in range(144))
    if blanks or max(p_err, z_err, p_total_err, z_total_err, p_cost_err, contract_err) > EPS:
        raise AssertionError("Plan or adjustment writeback mismatch")

    storage_err = soc_err = 0.0
    if storage.max_row != 1 + len(FORMAL_DATES) * 6 or storage.max_column != 6:
        raise AssertionError("Storage worksheet has incorrect expanded dimensions")
    for day_idx, day in enumerate(FORMAL_DATES):
        group = slots.loc[slots["date"].eq(day)].sort_values("slot").reset_index(drop=True)
        first = 2 + day_idx * 6
        if pd.Timestamp(storage.cell(first, 1).value).normalize() != day:
            raise AssertionError("Storage date blocks are misaligned")
        for block_idx, (label, start, stop) in enumerate(BLOCKS):
            row = first + block_idx
            if storage.cell(row, 2).value != label:
                raise AssertionError("Storage time-block label changed")
            storage_err = max(storage_err, abs(float(storage.cell(row, 3).value) - float(group.loc[start:stop - 1, "charge_rt_kwh"].sum())))
            storage_err = max(storage_err, abs(float(storage.cell(row, 4).value) - float(group.loc[start:stop - 1, "discharge_rt_kwh"].sum())))
        soc_err = max(soc_err, abs(float(storage.cell(first, 6).value) - float(group.loc[0, "soc_start_kwh"])))
        soc_err = max(soc_err, abs(float(storage.cell(first + 1, 6).value) - float(group.loc[143, "soc_end_kwh"])))
    if max(storage_err, soc_err) > EPS:
        raise AssertionError("Storage aggregate or SOC writeback mismatch")

    actual_events: list[tuple[pd.Timestamp, str, float]] = []
    current_date: pd.Timestamp | None = None
    for row in range(2, emergency.max_row + 1):
        value, period, date_value = emergency.cell(row, 3).value, emergency.cell(row, 2).value, emergency.cell(row, 1).value
        if date_value is not None:
            current_date = pd.Timestamp(date_value).normalize()
        if value is None and period is None:
            continue
        if current_date is None or period is None or value is None or float(value) <= EPS:
            raise AssertionError("Emergency worksheet contains an invalid record")
        actual_events.append((current_date, str(period), float(value)))
    expected = [(event["date"], event["period"], event["amount_kwh"]) for event in expected_events]
    event_err = max((abs(got[2] - wanted[2]) for got, wanted in zip(actual_events, expected)), default=0.0)
    if len(actual_events) != len(expected) or event_err > EPS or any(got[:2] != wanted[:2] for got, wanted in zip(actual_events, expected)):
        raise AssertionError("Emergency-event reconstruction mismatch")
    event_sum_err = abs(sum(event[2] for event in actual_events) - float(slots["emergency_purchase_kwh"].sum()))
    if event_sum_err > EPS:
        raise AssertionError("Emergency event total differs from frozen slots")

    pi = slots["price_yuan_per_kwh"]
    p, z = slots["original_plan_kwh"], slots["effective_purchase_kwh"]
    plus, minus = slots["delta_plus_kwh"], slots["delta_minus_kwh"]
    c, b, e, spill = slots["charge_rt_kwh"], slots["discharge_rt_kwh"], slots["emergency_purchase_kwh"], slots["spill_kwh"]
    contract_recomputed = pi * p + 1.5 * pi * plus - 0.5 * pi * minus
    emergency_recomputed = 5.0 * pi * e
    balance = z + slots["pv_actual_kwh"] + b + e - slots["load_actual_kwh"] - c - spill
    transition = slots["soc_end_kwh"] - (slots["soc_start_kwh"] + 0.9 * c - b / 0.9)
    starts = slots.groupby("date", sort=True).first()["soc_start_kwh"].to_numpy(float)
    ends = slots.groupby("date", sort=True).last()["soc_end_kwh"].to_numpy(float)
    soc_all = np.r_[slots["soc_start_kwh"].to_numpy(float), slots["soc_end_kwh"].to_numpy(float)]
    delta_relation = np.maximum(z - p, 0.0) - plus
    minus_relation = np.maximum(p - z, 0.0) - minus
    result = {
        "plan_matrix_cells": len(FORMAL_DATES) * 144,
        "adjustment_matrix_cells": len(FORMAL_DATES) * 144,
        "plan_cell_max_error_kwh": p_err, "adjustment_cell_max_error_kwh": z_err,
        "plan_daily_total_max_error_kwh": p_total_err, "adjustment_daily_total_max_error_kwh": z_total_err,
        "plan_daily_fee_max_error_yuan": p_cost_err, "contract_daily_fee_max_error_yuan": contract_err,
        "storage_aggregate_max_error_kwh": storage_err, "exported_soc_max_error_kwh": soc_err,
        "emergency_event_max_error_kwh": event_err, "emergency_event_sum_error_kwh": event_sum_err,
        "contract_recompute_max_abs_error_yuan": float(np.max(np.abs(contract_recomputed - slots["contract_cost_yuan"]))),
        "emergency_recompute_max_abs_error_yuan": float(np.max(np.abs(emergency_recomputed - slots["emergency_cost_yuan"]))),
        "total_recompute_max_abs_error_yuan": float(np.max(np.abs(contract_recomputed + emergency_recomputed - slots["actual_total_cost_yuan"]))),
        "energy_balance_max_abs_residual_kwh": float(np.max(np.abs(balance))),
        "soc_transition_max_abs_residual_kwh": float(np.max(np.abs(transition))),
        "cross_day_soc_max_abs_error_kwh": float(np.max(np.abs(starts[1:] - ends[:-1]))),
        "soc_violation_count": int(((soc_all < 1200 - EPS) | (soc_all > 10800 + EPS)).sum()),
        "power_violation_count": int(((c > 2500 / 3 + EPS) | (b > 2500 / 3 + EPS)).sum()),
        "simultaneous_charge_discharge_count": int(((c > EPS) & (b > EPS)).sum()),
        "emergency_and_charge_count": int(((e > EPS) & (c > EPS)).sum()),
        "negative_plan_or_effective_count": int(((p < -EPS) | (z < -EPS)).sum()),
        "delta_plus_relation_max_error_kwh": float(np.max(np.abs(delta_relation))),
        "delta_minus_relation_max_error_kwh": float(np.max(np.abs(minus_relation))),
        "event_count": len(actual_events), "formula_count_preserved": formula_count == template["formula_count"],
    }
    numeric = [v for key, v in result.items() if key.endswith("error_kwh") or key.endswith("error_yuan") or key.endswith("residual_kwh")]
    count_checks = [result[key] for key in ("soc_violation_count", "power_violation_count", "simultaneous_charge_discharge_count", "emergency_and_charge_count", "negative_plan_or_effective_count")]
    if max(numeric, default=0.0) > EPS or any(count_checks):
        raise AssertionError("Independent post-write physical or accounting verification failed")
    return result, pd.DataFrame(checks)


def comparison_audit(slots: pd.DataFrame) -> dict:
    table = pd.read_csv(COMPARISON).set_index("policy")
    a, b = table.loc[POLICY_A], table.loc[POLICY_B]
    expected = {
        POLICY_A: (13152259.342, 116516.897, 13268776.239, 29540.082, 1462.576),
        POLICY_B: (13068148.860, 829567.131, 13897715.992, 180018.160, 1399.588),
    }
    for policy, values in expected.items():
        row = table.loc[policy]
        actual = (row["contract_cost_yuan"], row["emergency_cost_yuan"], row["actual_total_cost_yuan"], row["emergency_purchase_kwh"], row["terminal_soc_kwh"])
        if any(abs(float(x) - y) > 5e-4 for x, y in zip(actual, values)):
            raise AssertionError(f"Frozen comparison no longer agrees with stated {policy} values")
    from_slots = {
        "contract_cost_yuan": float(slots["contract_cost_yuan"].sum()),
        "emergency_cost_yuan": float(slots["emergency_cost_yuan"].sum()),
        "actual_total_cost_yuan": float(slots["actual_total_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(slots["emergency_purchase_kwh"].sum()),
        "terminal_soc_kwh": float(slots.iloc[-1]["soc_end_kwh"]),
    }
    source_err = {key: abs(from_slots[key] - float(a[key])) for key in from_slots}
    if max(source_err.values()) > EPS:
        raise AssertionError(f"A slot aggregation differs from frozen comparison: {source_err}")
    return {
        "A": {key: float(a[key]) for key in from_slots}, "B": {key: float(b[key]) for key in from_slots},
        "A_minus_B_contract_yuan": float(a["contract_cost_yuan"] - b["contract_cost_yuan"]),
        "A_minus_B_emergency_yuan": float(a["emergency_cost_yuan"] - b["emergency_cost_yuan"]),
        "B_minus_A_total_yuan": float(b["actual_total_cost_yuan"] - a["actual_total_cost_yuan"]),
        "B_minus_A_total_percent": float((b["actual_total_cost_yuan"] - a["actual_total_cost_yuan"]) / b["actual_total_cost_yuan"] * 100),
        "B_minus_A_emergency_kwh": float(b["emergency_purchase_kwh"] - a["emergency_purchase_kwh"]),
        "B_minus_A_emergency_percent": float((b["emergency_purchase_kwh"] - a["emergency_purchase_kwh"]) / b["emergency_purchase_kwh"] * 100),
        "A_minus_B_terminal_soc_kwh": float(a["terminal_soc_kwh"] - b["terminal_soc_kwh"]),
        "A_slot_aggregation_max_error": max(source_err.values()),
    }


def write_reports(template: dict, protected_before: dict[str, str], comparison: dict, checks: dict, final_hash: str) -> None:
    protected_after = protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Protected manifest changed during export")
    comparison_frame = pd.read_csv(COMPARISON)
    STAGE5.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(STAGE5 / "q3_v2_final_comparison.csv", index=False, encoding="utf-8-sig")
    payload = {
        "export_time": datetime.now().isoformat(timespec="seconds"),
        "strategy_name": "多时点预报更新—滚动购电调整与动态价值储能调度策略",
        "source_policy": POLICY_A,
        "source_slots": str(SLOTS.relative_to(ROOT)),
        "period": ["2025-02-01", "2025-12-31"], "days": 334, "slots": 48096,
        "raw_template_sha256_before": template["raw_sha256"], "raw_template_sha256_after": sha256(RAW_TEMPLATE),
        "result3_sha256": final_hash, "protected_file_count": len(protected_before),
        "protected_hashes_unchanged": protected_after == protected_before,
        "comparison": comparison, "writeback_checks": checks,
        "adjustment_sheet_interpretation": "按建模假设填写执行槽之前最后一个有效版本的最终生效购电量 z；不是各次调整量之和。",
        "charge_discharge_convention": "充放电量表 C/D 分别为实际充电/放电的非负 kWh 幅值。",
        "emergency_event_convention": "仅记录正紧急购电；同日相邻正槽合并，跨日不合并。",
        "optimization_rerun": False,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (DOCS / "q3_v2_export_audit.md").write_text("# 问题三 V2 正式模板导出审计\n\n```json\n" + text + "\n```\n", encoding="utf-8")
    (DOCS / "q3_v2_final_results_audit.md").write_text(
        "# 问题三 V2 最终结果审查\n\n"
        f"正式策略为**多时点预报更新—滚动购电调整与动态价值储能调度策略**。冻结 A 策略由逐槽源数据独立汇总：合同费用 {comparison['A']['contract_cost_yuan']:.3f} 元，紧急购电费用 {comparison['A']['emergency_cost_yuan']:.3f} 元，实际总费用 {comparison['A']['actual_total_cost_yuan']:.3f} 元，紧急购电量 {comparison['A']['emergency_purchase_kwh']:.3f} kWh，期末 SOC {comparison['A']['terminal_soc_kwh']:.3f} kWh。\n\n"
        f"相对禁止日内调整的 B 策略，A 的合同费用增加 {comparison['A_minus_B_contract_yuan']:.3f} 元，紧急购电费用减少 {-comparison['A_minus_B_emergency_yuan']:.3f} 元，实际总费用减少 {comparison['B_minus_A_total_yuan']:.3f} 元（{comparison['B_minus_A_total_percent']:.3f}%），紧急购电量减少 {comparison['B_minus_A_emergency_kwh']:.3f} kWh（{comparison['B_minus_A_emergency_percent']:.3f}%）；A 期末 SOC 高 {comparison['A_minus_B_terminal_soc_kwh']:.3f} kWh。\n\n"
        "上述结果为固定开发期参数下的因果回放，不主张理论全局最优。终端库存价值仅服务于优化，未从实际账单扣除。\n",
        encoding="utf-8",
    )
    (DOCS / "q3_v2_strategy_freeze_record.md").write_text(
        "# 问题三 V2 策略冻结记录\n\n"
        "- 正式策略：多时点预报更新—滚动购电调整与动态价值储能调度策略。\n"
        "- 冻结配置：同星期几负载预测；M=14 联合残差情景；终端库存价值 v=0.4 元/kWh；100 kWh SOC 价值网格；实际 SOC 跨日连续。\n"
        "- 更新时点：0:00、6:00、12:00、18:00。\n"
        "- 结算解释：每槽最终生效计划仅相对 0:00 原计划结算一次；上调部分为 1.5 倍电价、下调部分退还原价 50%；紧急购电为 5 倍电价。\n"
        "- 说明：以上为题意歧义下的建模解释与 1 月开发期冻结参数，不表示理论全局最优。\n"
        f"- 原模板 SHA-256：{template['raw_sha256']}；正式 result3.xlsx SHA-256：{final_hash}。\n"
        f"- 受保护文件：{len(protected_before)} 个，导出前后哈希一致。\n",
        encoding="utf-8",
    )
    (STAGE5 / "q3_v2_stage5_validation.txt").write_text(
        "Q3 V2 Stage 5 export validation\n"
        f"result3_sha256={final_hash}\nraw_template_sha256={template['raw_sha256']}\n"
        f"protected_hashes_unchanged={protected_after == protected_before}\n"
        f"contract={comparison['A']['contract_cost_yuan']:.12f}\nemergency={comparison['A']['emergency_cost_yuan']:.12f}\ntotal={comparison['A']['actual_total_cost_yuan']:.12f}\n"
        + "\n".join(f"{key}={value}" for key, value in checks.items()) + "\npassed=True\n",
        encoding="utf-8",
    )


def run() -> None:
    if FINAL_FILE.exists():
        raise FileExistsError(f"Authorised output already exists; refusing overwrite: {FINAL_FILE}")
    protected_before = protected_hashes()
    template = assert_template()
    comparison = comparison_audit(load_a_slots())
    if sha256(RAW_TEMPLATE) != template["raw_sha256"]:
        raise RuntimeError("Raw template changed during pre-export review")
    slots = load_a_slots()
    events = write_candidate(slots, template)
    checks, cell_checks = validate_candidate(slots, template, events)
    STAGE5.mkdir(parents=True, exist_ok=True)
    cell_checks.to_csv(STAGE5 / "q3_v2_result3_cell_check.csv", index=False, encoding="utf-8-sig")
    # Only a fully checked candidate can become the authorised output.
    FINAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANDIDATE, FINAL_FILE)
    final_hash = sha256(FINAL_FILE)
    if final_hash != sha256(CANDIDATE):
        raise RuntimeError("Candidate-to-final hash mismatch")
    if sha256(RAW_TEMPLATE) != template["raw_sha256"]:
        raise RuntimeError("Raw template changed during export")
    write_reports(template, protected_before, comparison, checks, final_hash)
    print(json.dumps({"result3": str(FINAL_FILE), "sha256": final_hash, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
