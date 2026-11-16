"""Create and verify the Question-2 submission workbook from the fixed A+ path.

Only the A+ strategy (w=7, q=0.75, Sres=3600 kWh) is replayed here.  The
baseline is never read for workbook values.  The raw template is copied before
writing and is checked again after export.
"""

from __future__ import annotations

from copy import copy
from datetime import time
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.config import (
    E_B_MAX,
    Q1_FOUR_HOUR_BLOCKS,
    Q1_TABLE1_PERIODS,
    Q2_FIXED_QUANTILE,
    Q2_FIXED_RHO,
    Q2_FIXED_WINDOW_DAYS,
    Q2_FULL_YEAR_SHADOW_COMPARISON_FILE,
    Q2_SHADOW_END,
    Q2_SHADOW_FORMAL_START,
    Q2_SHADOW_WARMUP_START,
    Q2_VALIDATION_TOLERANCE,
    RAW_DATA_DIR,
    S_INIT,
    S_MAX,
    S_MIN,
    TABLE_DIR,
    LOG_DIR,
)
from src.data_io import load_q2_actual_energy
from src.dispatch_q2_realtime import dispatch_q2_realtime
from src.forecast_q2 import forecast_net_load_quantile
from src.model_q2_day_ahead import get_highs_solver, solve_q2_day_ahead
from src.shadow_q2_full_year import _cold_start_plan


Q2_TEMPLATE = RAW_DATA_DIR / "result2.xlsx"
Q2_RESULT = TABLE_DIR / "result2.xlsx"
Q2_PAPER_TABLES = TABLE_DIR / "q2_paper_tables.csv"
Q2_SUBMISSION_CHECK = LOG_DIR / "q2_submission_check.txt"
APLUS_RESERVE_SOC = 3600.0
TABLE3_DATES = tuple(pd.Timestamp(value) for value in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"))


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assert_raw_template_empty() -> None:
    workbook = load_workbook(Q2_TEMPLATE, data_only=False)
    if workbook.sheetnames != ["计划购电量", "充放电量", "紧急购电量"]:
        raise ValueError(f"Unexpected result2 template sheets: {workbook.sheetnames}")
    plan = workbook["计划购电量"]
    storage = workbook["充放电量"]
    emergency = workbook["紧急购电量"]
    if any(plan.cell(row, col).value is not None for row in range(2, 336) for col in range(2, 148)):
        raise ValueError("Raw result2 plan template is not empty")
    if any(storage.cell(row, col).value is not None for row in list(range(2, 14)) + list(range(15, 21)) for col in (3, 4, 6)):
        raise ValueError("Raw result2 storage template is not empty")
    placeholder_values = {"…", "⋮", "⁝"}
    if any(
        emergency.cell(row, col).value is not None and str(emergency.cell(row, col).value).strip() not in placeholder_values
        for row in range(2, emergency.max_row + 1) for col in (2, 3)
    ):
        raise ValueError("Raw result2 emergency template is not empty")


def replay_aplus_for_export() -> tuple[list[dict], dict]:
    """Reproduce the validated A+ causal path, retaining only formal-day slots."""
    actual, price = load_q2_actual_energy()
    solver, _ = get_highs_solver()
    dates = pd.date_range(Q2_SHADOW_WARMUP_START, Q2_SHADOW_END, freq="D")
    formal_start = pd.Timestamp(Q2_SHADOW_FORMAL_START)
    state = float(S_INIT)
    previous_end = None
    formal_runs: list[dict] = []
    all_checks = {
        "no_future_data_leakage": True,
        "max_balance_error_kwh": 0.0,
        "soc_boundary_violation_count": 0,
        "power_violation_count": 0,
        "emergency_charge_overlap_count": 0,
        "max_cross_day_soc_error_kwh": 0.0,
        "max_plan_cost_recomputation_error_yuan": 0.0,
        "reserve_shortfall_all_365_kwh": 0.0,
        "reserve_trigger_dates_all_365": [],
    }
    for day_index, day in enumerate(dates):
        actual_day = actual.loc[actual["date"] == day].copy()
        if day_index == 0:
            forecast = None
            plan, metadata = _cold_start_plan(price)
        else:
            window = min(Q2_FIXED_WINDOW_DAYS, day_index)
            forecast = forecast_net_load_quantile(actual, day, window, Q2_FIXED_QUANTILE)
            if not forecast["history_end_date"].iloc[0] < day:
                all_checks["no_future_data_leakage"] = False
                raise RuntimeError(f"Future-data leakage detected for {day.date()}")
            plan, metadata = solve_q2_day_ahead(
                forecast=forecast,
                price=price,
                initial_soc_kwh=state,
                reserve_soc_kwh=APLUS_RESERVE_SOC,
                reserve_penalty_yuan_per_kwh=Q2_FIXED_RHO,
                solver=solver,
            )
        execution, _ = dispatch_q2_realtime(plan, actual_day, state, Q2_VALIDATION_TOLERANCE)
        soc_values = pd.concat([execution["rt_soc_start_kwh"], execution["rt_soc_end_kwh"]])
        all_checks["max_balance_error_kwh"] = max(
            all_checks["max_balance_error_kwh"], float(execution["balance_residual_kwh"].abs().max())
        )
        all_checks["soc_boundary_violation_count"] += int(
            ((soc_values < S_MIN - Q2_VALIDATION_TOLERANCE) | (soc_values > S_MAX + Q2_VALIDATION_TOLERANCE)).sum()
        )
        all_checks["power_violation_count"] += int(
            ((execution["rt_charge_kwh"] > E_B_MAX + Q2_VALIDATION_TOLERANCE) | (execution["rt_discharge_kwh"] > E_B_MAX + Q2_VALIDATION_TOLERANCE)).sum()
        )
        all_checks["emergency_charge_overlap_count"] += int(
            ((execution["emergency_purchase_kwh"] > Q2_VALIDATION_TOLERANCE) & (execution["rt_charge_kwh"] > Q2_VALIDATION_TOLERANCE)).sum()
        )
        if previous_end is not None:
            all_checks["max_cross_day_soc_error_kwh"] = max(
                all_checks["max_cross_day_soc_error_kwh"],
                abs(float(execution["rt_soc_start_kwh"].iloc[0]) - previous_end),
            )
        expected_plan_cost = float(metadata["plan_purchase_cost_yuan"])
        actual_plan_cost = float(execution["plan_purchase_cost_yuan"].sum())
        all_checks["max_plan_cost_recomputation_error_yuan"] = max(
            all_checks["max_plan_cost_recomputation_error_yuan"], abs(expected_plan_cost - actual_plan_cost)
        )
        all_checks["reserve_shortfall_all_365_kwh"] += float(metadata["reserve_shortfall_kwh"])
        if metadata["reserve_shortfall_kwh"] > Q2_VALIDATION_TOLERANCE:
            all_checks["reserve_trigger_dates_all_365"].append(
                (day.date().isoformat(), float(metadata["reserve_shortfall_kwh"]))
            )
        previous_end = float(execution["rt_soc_end_kwh"].iloc[-1])
        state = previous_end
        if day >= formal_start:
            formal_runs.append({"date": day, "forecast": forecast, "plan": plan, "metadata": metadata, "execution": execution})
    if len(formal_runs) != 334:
        raise RuntimeError("A+ export replay must contain exactly 334 formal days")
    return formal_runs, all_checks


def _template_headers(plan_sheet) -> list[str]:
    headers = [plan_sheet.cell(1, col).value for col in range(2, 146)]
    if len(headers) != 144 or any(value is None for value in headers):
        raise ValueError("Plan template must expose 144 official time columns B:EO")
    return [str(value) for value in headers]


def _event_records(formal_runs: list[dict], headers: list[str]) -> list[dict]:
    """Merge adjacent positive emergency slots without creating zero records."""
    events: list[dict] = []
    tol = Q2_VALIDATION_TOLERANCE
    for run in formal_runs:
        execution = run["execution"].sort_values("idx").reset_index(drop=True)
        positive = execution.loc[execution["emergency_purchase_kwh"] > tol, "idx"].astype(int).tolist()
        if not positive:
            continue
        start = previous = positive[0]
        groups: list[tuple[int, int]] = []
        for idx in positive[1:]:
            if idx == previous + 1:
                previous = idx
            else:
                groups.append((start, previous))
                start = previous = idx
        groups.append((start, previous))
        for start, stop in groups:
            left = headers[start].split("-", maxsplit=1)[0]
            right = headers[stop].split("-", maxsplit=1)[1]
            amount = float(execution.loc[start:stop, "emergency_purchase_kwh"].sum())
            if amount <= tol:
                raise RuntimeError("Merged emergency event cannot have zero amount")
            events.append(
                {
                    "date": run["date"], "start_idx": start, "end_idx": stop,
                    "time_period": f"{left}-{right}", "emergency_purchase_kwh": amount,
                }
            )
    return events


def export_q2_submission(formal_runs: list[dict]) -> tuple[Path, list[dict]]:
    """Copy the untouched official template and fill it only with A+ values."""
    _assert_raw_template_empty()
    Q2_RESULT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Q2_TEMPLATE, Q2_RESULT)
    workbook = load_workbook(Q2_RESULT)
    if workbook.sheetnames != ["计划购电量", "充放电量", "紧急购电量"]:
        raise ValueError("Copied template sheet names changed unexpectedly")
    plan_sheet = workbook["计划购电量"]
    storage_sheet = workbook["充放电量"]
    emergency_sheet = workbook["紧急购电量"]
    headers = _template_headers(plan_sheet)

    expected_dates = [run["date"].normalize() for run in formal_runs]
    template_dates = [pd.Timestamp(plan_sheet.cell(row, 1).value).normalize() for row in range(2, 336)]
    if template_dates != expected_dates:
        raise ValueError("Plan-template dates are not the official 2025-02-01..2025-12-31 range")
    for row, run in enumerate(formal_runs, start=2):
        plan = run["plan"].sort_values("idx").reset_index(drop=True)
        values = plan["plan_purchase_kwh"].to_numpy(dtype=float)
        if len(values) != 144:
            raise ValueError("Each A+ plan must have 144 slots")
        for col, value in enumerate(values, start=2):
            plan_sheet.cell(row, col).value = float(value)
        plan_sheet.cell(row, 146).value = float(values.sum())
        plan_sheet.cell(row, 147).value = float(plan["plan_purchase_cost_yuan"].sum())

    # Preserve official display styles by taking the first fully specified six-row block.
    storage_styles = [[copy(storage_sheet.cell(row, col)._style) for col in range(1, 7)] for row in range(2, 8)]
    storage_heights = [storage_sheet.row_dimensions[row].height for row in range(2, 8)]
    storage_sheet.delete_rows(2, storage_sheet.max_row - 1)
    storage_sheet.insert_rows(2, len(formal_runs) * 6)
    block_labels = [item[0] for item in Q1_FOUR_HOUR_BLOCKS]
    for day_index, run in enumerate(formal_runs):
        first_row = 2 + day_index * 6
        execution = run["execution"].sort_values("idx").reset_index(drop=True)
        for within_day in range(6):
            row = first_row + within_day
            for col in range(1, 7):
                storage_sheet.cell(row, col)._style = copy(storage_styles[within_day][col - 1])
            storage_sheet.row_dimensions[row].height = storage_heights[within_day]
            _, _, start, stop = Q1_FOUR_HOUR_BLOCKS[within_day]
            storage_sheet.cell(row, 1).value = run["date"].to_pydatetime() if within_day == 0 else None
            storage_sheet.cell(row, 2).value = block_labels[within_day]
            storage_sheet.cell(row, 3).value = float(execution.loc[start:stop - 1, "rt_charge_kwh"].sum())
            storage_sheet.cell(row, 4).value = float(execution.loc[start:stop - 1, "rt_discharge_kwh"].sum())
            storage_sheet.cell(row, 5).value = time(0, 0) if within_day == 0 else ("24:00" if within_day == 1 else None)
            storage_sheet.cell(row, 6).value = (
                float(execution["rt_soc_start_kwh"].iloc[0]) if within_day == 0
                else (float(execution["rt_soc_end_kwh"].iloc[-1]) if within_day == 1 else None)
            )

    events = _event_records(formal_runs, headers)
    emergency_style = [copy(emergency_sheet.cell(2, col)._style) for col in range(1, 4)]
    emergency_height = emergency_sheet.row_dimensions[2].height
    emergency_sheet.delete_rows(2, emergency_sheet.max_row - 1)
    if events:
        emergency_sheet.insert_rows(2, len(events))
        previous_date = None
        for row, event in enumerate(events, start=2):
            for col in range(1, 4):
                emergency_sheet.cell(row, col)._style = copy(emergency_style[col - 1])
            emergency_sheet.row_dimensions[row].height = emergency_height
            event_date = event["date"].normalize()
            emergency_sheet.cell(row, 1).value = event_date.to_pydatetime() if event_date != previous_date else None
            emergency_sheet.cell(row, 2).value = event["time_period"]
            emergency_sheet.cell(row, 3).value = float(event["emergency_purchase_kwh"])
            previous_date = event_date
    workbook.save(Q2_RESULT)
    return Q2_RESULT, events


def export_q2_paper_tables(formal_runs: list[dict], events: list[dict]) -> Path:
    """Write Table 1/2 formats and Table 3 emergency records for specified dates."""
    records = []
    run_by_date = {run["date"].normalize(): run for run in formal_runs}
    for date in TABLE3_DATES:
        run = run_by_date[date]
        plan = run["plan"].sort_values("idx").reset_index(drop=True)
        execution = run["execution"].sort_values("idx").reset_index(drop=True)
        for label, idx, _ in Q1_TABLE1_PERIODS:
            records.append({
                "table": "表1", "date": date.date().isoformat(), "item": "指定时段计划购电量",
                "time_period": label, "idx": idx, "purchase_kwh": float(plan.loc[idx, "plan_purchase_kwh"]),
                "charge_kwh": np.nan, "discharge_kwh": np.nan, "soc_kwh": np.nan,
                "emergency_purchase_kwh": np.nan, "cost_yuan": np.nan, "unit": "kWh",
            })
        records.extend([
            {"table": "表1", "date": date.date().isoformat(), "item": "全天计划购电量", "time_period": "全天", "idx": np.nan,
             "purchase_kwh": float(plan["plan_purchase_kwh"].sum()), "charge_kwh": np.nan, "discharge_kwh": np.nan,
             "soc_kwh": np.nan, "emergency_purchase_kwh": np.nan, "cost_yuan": np.nan, "unit": "kWh"},
            {"table": "表1", "date": date.date().isoformat(), "item": "全天计划购电费", "time_period": "全天", "idx": np.nan,
             "purchase_kwh": np.nan, "charge_kwh": np.nan, "discharge_kwh": np.nan,
             "soc_kwh": np.nan, "emergency_purchase_kwh": np.nan, "cost_yuan": float(execution["plan_purchase_cost_yuan"].sum()), "unit": "元"},
        ])
        for label, _, start, stop in Q1_FOUR_HOUR_BLOCKS:
            records.append({
                "table": "表2", "date": date.date().isoformat(), "item": "区间实际充放电量", "time_period": label,
                "idx": f"[{start}:{stop})", "purchase_kwh": np.nan,
                "charge_kwh": float(execution.loc[start:stop - 1, "rt_charge_kwh"].sum()),
                "discharge_kwh": float(execution.loc[start:stop - 1, "rt_discharge_kwh"].sum()),
                "soc_kwh": np.nan, "emergency_purchase_kwh": np.nan, "cost_yuan": np.nan, "unit": "kWh",
            })
        records.extend([
            {"table": "表2", "date": date.date().isoformat(), "item": "实际储电量", "time_period": "0:00", "idx": "soc[0]",
             "purchase_kwh": np.nan, "charge_kwh": np.nan, "discharge_kwh": np.nan,
             "soc_kwh": float(execution["rt_soc_start_kwh"].iloc[0]), "emergency_purchase_kwh": np.nan, "cost_yuan": np.nan, "unit": "kWh"},
            {"table": "表2", "date": date.date().isoformat(), "item": "实际储电量", "time_period": "24:00", "idx": "soc[144]",
             "purchase_kwh": np.nan, "charge_kwh": np.nan, "discharge_kwh": np.nan,
             "soc_kwh": float(execution["rt_soc_end_kwh"].iloc[-1]), "emergency_purchase_kwh": np.nan, "cost_yuan": np.nan, "unit": "kWh"},
        ])
    for event in events:
        if event["date"].normalize() in TABLE3_DATES:
            records.append({
                "table": "表3", "date": event["date"].date().isoformat(), "item": "紧急购电量",
                "time_period": event["time_period"], "idx": f"[{event['start_idx']}:{event['end_idx'] + 1})",
                "purchase_kwh": np.nan, "charge_kwh": np.nan, "discharge_kwh": np.nan, "soc_kwh": np.nan,
                "emergency_purchase_kwh": event["emergency_purchase_kwh"], "cost_yuan": np.nan, "unit": "kWh",
            })
    Q2_PAPER_TABLES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(Q2_PAPER_TABLES, index=False, encoding="utf-8-sig", float_format="%.10f")
    return Q2_PAPER_TABLES


def validate_q2_submission(
    formal_runs: list[dict], events: list[dict], replay_checks: dict, raw_hash_before: str
) -> dict:
    """Reopen output workbook and compare every written value against A+ replay."""
    tol = Q2_VALIDATION_TOLERANCE
    workbook = load_workbook(Q2_RESULT, data_only=False)
    sheets_ok = workbook.sheetnames == ["计划购电量", "充放电量", "紧急购电量"]
    plan_sheet = workbook["计划购电量"]
    storage_sheet = workbook["充放电量"]
    emergency_sheet = workbook["紧急购电量"]
    headers_ok = len(_template_headers(plan_sheet)) == 144
    plan_match = True
    plan_count = 0
    plan_cost_match = True
    for row, run in enumerate(formal_runs, start=2):
        expected = run["plan"].sort_values("idx").reset_index(drop=True)
        cells = [plan_sheet.cell(row, col).value for col in range(2, 146)]
        plan_count += sum(value is not None for value in cells)
        if not all(isinstance(value, (int, float)) for value in cells):
            plan_match = False
            continue
        plan_match &= bool(np.allclose(np.asarray(cells, dtype=float), expected["plan_purchase_kwh"].to_numpy(dtype=float), atol=tol, rtol=0.0))
        plan_match &= bool(np.isclose(float(plan_sheet.cell(row, 146).value), expected["plan_purchase_kwh"].sum(), atol=tol, rtol=0.0))
        plan_cost_match &= bool(np.isclose(float(plan_sheet.cell(row, 147).value), expected["plan_purchase_cost_yuan"].sum(), atol=tol, rtol=0.0))

    storage_match = storage_sheet.max_row == 1 + 6 * len(formal_runs)
    for day_index, run in enumerate(formal_runs):
        start_row = 2 + 6 * day_index
        execution = run["execution"].sort_values("idx").reset_index(drop=True)
        storage_match &= pd.Timestamp(storage_sheet.cell(start_row, 1).value).normalize() == run["date"].normalize()
        for block, (label, _, start, stop) in enumerate(Q1_FOUR_HOUR_BLOCKS):
            row = start_row + block
            storage_match &= storage_sheet.cell(row, 2).value == label
            storage_match &= np.isclose(float(storage_sheet.cell(row, 3).value), execution.loc[start:stop - 1, "rt_charge_kwh"].sum(), atol=tol, rtol=0.0)
            storage_match &= np.isclose(float(storage_sheet.cell(row, 4).value), execution.loc[start:stop - 1, "rt_discharge_kwh"].sum(), atol=tol, rtol=0.0)
        storage_match &= np.isclose(float(storage_sheet.cell(start_row, 6).value), execution["rt_soc_start_kwh"].iloc[0], atol=tol, rtol=0.0)
        storage_match &= np.isclose(float(storage_sheet.cell(start_row + 1, 6).value), execution["rt_soc_end_kwh"].iloc[-1], atol=tol, rtol=0.0)

    parsed_events = []
    current_date = None
    for row in range(2, emergency_sheet.max_row + 1):
        date_value = emergency_sheet.cell(row, 1).value
        if date_value is not None:
            current_date = pd.Timestamp(date_value).normalize()
        if emergency_sheet.cell(row, 2).value is None and emergency_sheet.cell(row, 3).value is None:
            continue
        parsed_events.append((current_date, emergency_sheet.cell(row, 2).value, float(emergency_sheet.cell(row, 3).value)))
    expected_events = [(event["date"].normalize(), event["time_period"], event["emergency_purchase_kwh"]) for event in events]
    events_match = len(parsed_events) == len(expected_events) and all(
        parsed[0] == expected[0] and parsed[1] == expected[1] and np.isclose(parsed[2], expected[2], atol=tol, rtol=0.0)
        for parsed, expected in zip(parsed_events, expected_events)
    )
    raw_hash_after = _hash(Q2_TEMPLATE)
    total_plan_excel = sum(float(plan_sheet.cell(row, 147).value) for row in range(2, 336))
    total_emergency_excel = sum(value for _, _, value in parsed_events)
    shadow = pd.read_csv(Q2_FULL_YEAR_SHADOW_COMPARISON_FILE)
    aplus = shadow.loc[shadow["strategy_key"] == "aplus"].iloc[0]
    total_plan_expected = sum(float(run["execution"]["plan_purchase_cost_yuan"].sum()) for run in formal_runs)
    total_emergency_expected = sum(float(run["execution"]["emergency_purchase_cost_yuan"].sum()) for run in formal_runs)
    report = {
        "sheet_names_exact": sheets_ok,
        "official_plan_date_rows_334": plan_sheet.max_row == 335,
        "official_plan_headers_144": headers_ok,
        "planned_purchase_nonempty_cells": plan_count,
        "planned_purchase_cells_expected_48096": plan_count == 334 * 144,
        "planned_purchase_and_daily_totals_match_aplus": plan_match,
        "planned_purchase_costs_match_aplus": plan_cost_match,
        "storage_six_blocks_per_day_match_actual_rt": bool(storage_match),
        "emergency_events_merged_and_match_actual": events_match,
        "emergency_events_total_kwh": total_emergency_excel,
        "emergency_event_total_matches_aplus": bool(np.isclose(total_emergency_excel, float(aplus["total_emergency_purchase_kwh"]), atol=tol, rtol=0.0)),
        "plan_cost_excel_yuan": total_plan_excel,
        "plan_cost_replay_yuan": total_plan_expected,
        "plan_cost_matches_shadow": bool(np.isclose(total_plan_excel, float(aplus["total_plan_purchase_cost_yuan"]), atol=tol, rtol=0.0)),
        "emergency_cost_replay_yuan": total_emergency_expected,
        "total_cost_replay_yuan": total_plan_expected + total_emergency_expected,
        "total_cost_matches_shadow": bool(np.isclose(total_plan_expected + total_emergency_expected, float(aplus["total_actual_cost_yuan"]), atol=tol, rtol=0.0)),
        "raw_template_hash_unchanged": raw_hash_after == raw_hash_before,
        "raw_template_still_empty": True,
        **replay_checks,
    }
    _assert_raw_template_empty()
    report["passed"] = bool(
        report["sheet_names_exact"] and report["official_plan_date_rows_334"] and report["official_plan_headers_144"]
        and report["planned_purchase_cells_expected_48096"] and report["planned_purchase_and_daily_totals_match_aplus"]
        and report["planned_purchase_costs_match_aplus"] and report["storage_six_blocks_per_day_match_actual_rt"]
        and report["emergency_events_merged_and_match_actual"] and report["emergency_event_total_matches_aplus"]
        and report["plan_cost_matches_shadow"] and report["total_cost_matches_shadow"]
        and report["raw_template_hash_unchanged"] and report["raw_template_still_empty"]
        and report["no_future_data_leakage"] and report["soc_boundary_violation_count"] == 0
        and report["power_violation_count"] == 0 and report["emergency_charge_overlap_count"] == 0
        and report["max_balance_error_kwh"] <= tol and report["max_cross_day_soc_error_kwh"] <= tol
        and report["max_plan_cost_recomputation_error_yuan"] <= tol
    )
    return report


def write_submission_check(report: dict) -> Path:
    Q2_SUBMISSION_CHECK.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Question 2 final submission check (A+ only)", "=" * 48]
    lines.extend(f"{key}: {value}" for key, value in report.items())
    Q2_SUBMISSION_CHECK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Q2_SUBMISSION_CHECK


def run_q2_submission_export() -> dict:
    """Replay A+, export official workbook, paper tables, and write-after checks."""
    raw_hash_before = _hash(Q2_TEMPLATE)
    _assert_raw_template_empty()
    formal_runs, replay_checks = replay_aplus_for_export()
    workbook_path, events = export_q2_submission(formal_runs)
    paper_path = export_q2_paper_tables(formal_runs, events)
    report = validate_q2_submission(formal_runs, events, replay_checks, raw_hash_before)
    check_path = write_submission_check(report)
    if not report["passed"]:
        raise RuntimeError("Question-2 submission check failed; inspect q2_submission_check.txt")
    return {
        "result_path": workbook_path, "paper_tables_path": paper_path,
        "submission_check_path": check_path, "report": report, "events": events,
    }


if __name__ == "__main__":
    outcome = run_q2_submission_export()
    print(f"Question-2 workbook: {outcome['result_path']}")
    print(f"Submission check passed: {outcome['report']['passed']}")
