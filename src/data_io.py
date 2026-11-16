"""Raw-file inspection for Question 1. This module never writes data/raw."""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
import pandas as pd

from src.config import (
    CHECK_DIR,
    ETA_C,
    ETA_D,
    N_SLOTS,
    RAW_DATA_DIR,
    Q2_INPUT_PATH,
    S_FINAL,
    S_INIT,
    S_MAX,
    S_MIN,
)


Q1_INPUT_PATH = RAW_DATA_DIR / "附件1.xlsx"
Q1_TEMPLATE_PATH = RAW_DATA_DIR / "result1.xlsx"
Q1_INSPECTION_PATH = CHECK_DIR / "template_check.txt"


def _display(value: Any) -> str:
    """Format an Excel cell value deterministically for a plain-text report."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    return repr(value)


def _rows(sheet, start: int, end: int) -> list[str]:
    return [
        f"row {row}: " + " | ".join(_display(sheet.cell(row, col).value) for col in range(1, sheet.max_column + 1))
        for row in range(start, end + 1)
    ]


def _time_to_minutes(value: Any) -> int | None:
    """Return minutes after representative-day start; 24:00 is 1440."""
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    text = str(value).strip()
    if text == "0:00+1":
        return 24 * 60
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return hour * 60 + minute


def _left_endpoint_minutes(label: Any) -> int | None:
    """Extract left endpoint from a result label such as 0:10-0:20."""
    text = str(label).strip()
    if "-" not in text:
        return None
    return _time_to_minutes(text.split("-", maxsplit=1)[0])


def _is_ten_minute_cycle(minutes: list[int | None]) -> bool:
    if len(minutes) != N_SLOTS or any(value is None for value in minutes):
        return False
    return all(minutes[i] == 10 * (i + 1) for i in range(N_SLOTS))


def inspect_q1_files(report_path: Path = Q1_INSPECTION_PATH) -> dict[str, Any]:
    """Inspect Attachment 1 and result1.xlsx before any model is built.

    The report records observed workbook structure, time-label mapping, result cells to
    fill, and the fixed storage-state convention. It does not alter the source files.
    """
    input_book = load_workbook(Q1_INPUT_PATH, read_only=True, data_only=False)
    template_book = load_workbook(Q1_TEMPLATE_PATH, read_only=True, data_only=False)
    input_sheet = input_book.active

    plan_sheet_name = "计划购电量"
    storage_sheet_name = "充放电量"
    if template_book.sheetnames != [plan_sheet_name, storage_sheet_name]:
        raise ValueError(f"Unexpected result1.xlsx sheets: {template_book.sheetnames}")
    plan_sheet = template_book[plan_sheet_name]
    storage_sheet = template_book[storage_sheet_name]

    input_headers = [input_sheet.cell(1, col).value for col in range(1, input_sheet.max_column + 1)]
    template_headers = [plan_sheet.cell(1, col).value for col in range(1, plan_sheet.max_column + 1)]
    storage_headers = [storage_sheet.cell(1, col).value for col in range(1, storage_sheet.max_column + 1)]
    input_times = [input_sheet.cell(row, 1).value for row in range(2, input_sheet.max_row + 1)]
    plan_labels = [plan_sheet.cell(row, 1).value for row in range(2, plan_sheet.max_row + 1)]
    input_minutes = [_time_to_minutes(value) for value in input_times]
    plan_left_minutes = [_left_endpoint_minutes(value) for value in plan_labels]

    input_count_ok = input_sheet.max_row - 1 == N_SLOTS
    plan_count_ok = plan_sheet.max_row - 1 == N_SLOTS
    input_continuous = _is_ten_minute_cycle(input_minutes)
    plan_continuous = _is_ten_minute_cycle(plan_left_minutes)
    mapping_by_order = input_count_ok and plan_count_ok and input_minutes == plan_left_minutes

    contract = {
        "decision_slots": N_SLOTS,
        "soc_slots": N_SLOTS + 1,
        "soc_start": S_INIT,
        "soc_end": S_FINAL,
        "soc_lower": S_MIN,
        "soc_upper": S_MAX,
        "eta_charge": ETA_C,
        "eta_discharge": ETA_D,
        "soc_equation": "soc[t+1] = soc[t] + eta_c * charge[t] - discharge[t] / eta_d",
    }
    report_lines = [
        "Question 1 attachment/template inspection",
        "=" * 42,
        "Scope: inspection only; no raw input file has been modified.",
        "",
        "[Attachment 1]",
        f"path: {Q1_INPUT_PATH}",
        f"sheets: {input_book.sheetnames}",
        f"active sheet: {input_sheet.title}",
        f"dimensions: {input_sheet.max_row} rows x {input_sheet.max_column} columns",
        f"headers: {input_headers}",
        "unit note: Excel headers have no units; the problem statement specifies price in yuan/kWh and load/PV in kW.",
        "first 5 data rows:",
        *_rows(input_sheet, 2, 6),
        "last 5 data rows:",
        *_rows(input_sheet, input_sheet.max_row - 4, input_sheet.max_row),
        "",
        "[result1 template]",
        f"path: {Q1_TEMPLATE_PATH}",
        f"sheets: {template_book.sheetnames}",
        f"plan-sheet dimensions: {plan_sheet.max_row} rows x {plan_sheet.max_column} columns",
        f"plan-sheet headers: {template_headers}",
        f"storage-sheet dimensions: {storage_sheet.max_row} rows x {storage_sheet.max_column} columns",
        f"storage-sheet headers: {storage_headers}",
        f"planned-purchase write range: B2:B{plan_sheet.max_row} ({plan_sheet.max_row - 1} cells)",
        "charge/discharge write ranges: B2:B7 and C2:C7 (six four-hour aggregates)",
        "SOC write cells: E2 for 0:00 and E3 for 24:00, according to D2/D3 labels",
        "",
        "[Time-label audit]",
        f"Attachment-1 data-row count is 144: {input_count_ok}",
        f"Template planned-purchase-row count is 144: {plan_count_ok}",
        f"Attachment-1 labels form 00:10, 00:20, ..., 24:00: {input_continuous}",
        f"Template left endpoints form 00:10, 00:20, ..., 24:00: {plan_continuous}",
        f"Order mapping input label -> template left endpoint is exact: {mapping_by_order}",
        f"first mapping: {_display(input_times[0])} -> {plan_labels[0]}",
        f"last mapping: {_display(input_times[-1])} -> {plan_labels[-1]}",
        "Implementation rule: use row order as the 144-period internal index; preserve template labels unchanged for export.",
        "Caveat: files establish exact row mapping but cannot independently prove timestamps are physical starts rather than ends.",
        "",
        "[Fixed Question-1 storage convention]",
        "decision variables: purchase[t], charge[t], discharge[t], curtailment[t], t = 0,...,143",
        "state variable: soc[t], t = 0,...,144; soc[0] is 0:00 and soc[144] is next-day 0:00",
        f"soc[0] = {contract['soc_start']}; soc[144] = {contract['soc_end']}",
        f"{contract['soc_lower']} <= soc[t] <= {contract['soc_upper']}",
        f"charge efficiency = {contract['eta_charge']}; discharge efficiency = {contract['eta_discharge']}",
        f"state equation: {contract['soc_equation']}",
        "",
        "Inspection verdict: PASS" if input_count_ok and plan_count_ok and input_continuous and plan_continuous and mapping_by_order else "Inspection verdict: REVIEW REQUIRED",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "passed": input_count_ok and plan_count_ok and input_continuous and plan_continuous and mapping_by_order,
        "report_path": report_path,
        "input_rows": input_sheet.max_row - 1,
        "template_rows": plan_sheet.max_row - 1,
        "first_mapping": (input_times[0], plan_labels[0]),
        "last_mapping": (input_times[-1], plan_labels[-1]),
        "storage_contract": contract,
    }


def load_q1_data() -> pd.DataFrame:
    """Return audited Attachment-1 data with internal index 0,...,143.

    The original time label is retained for traceability. Optimisation uses the
    index only, while export follows the official template row order.
    """
    inspection = inspect_q1_files()
    if not inspection["passed"]:
        raise RuntimeError("Question-1 input inspection did not pass")

    frame = pd.read_excel(Q1_INPUT_PATH)
    required_columns = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    if list(frame.columns) != required_columns:
        raise ValueError(f"Unexpected Attachment-1 headers: {list(frame.columns)}")
    if len(frame) != N_SLOTS or frame.isna().any().any():
        raise ValueError("Attachment 1 must have 144 complete data rows")

    result = pd.DataFrame(
        {
            "idx": range(N_SLOTS),
            "time_label": frame["时间"].map(_display),
            "price": pd.to_numeric(frame["电价"], errors="raise"),
            "load_kw": pd.to_numeric(frame["小区负载"], errors="raise"),
            "pv_kw": pd.to_numeric(frame["光伏发电预测功率"], errors="raise"),
        }
    )
    if (result[["price", "load_kw", "pv_kw"]] < 0).any().any():
        raise ValueError("Attachment 1 contains a negative price or power value")

    from src.config import DELTA_T

    result["load_kwh"] = result["load_kw"] * DELTA_T
    result["pv_kwh"] = result["pv_kw"] * DELTA_T
    return result


def load_q2_actual_energy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Question-2 actual load/PV energy and the daily-reused price curve.

    This function is read-only.  It preserves the official 144-column order and
    creates a long table indexed by (date, idx).  It intentionally exposes the
    full year only as historical observations; callers must filter dates to
    enforce the 0:00 information boundary.
    """
    attachment1 = pd.read_excel(Q1_INPUT_PATH)
    expected_q1_columns = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    if list(attachment1.columns) != expected_q1_columns or len(attachment1) != N_SLOTS:
        raise ValueError("Attachment 1 does not have the audited 144-slot price structure")
    price = pd.DataFrame(
        {
            "idx": range(N_SLOTS),
            "time_label": attachment1["时间"].map(_display),
            "price_yuan_per_kwh": pd.to_numeric(attachment1["电价"], errors="raise"),
        }
    )
    if price["price_yuan_per_kwh"].isna().any() or (price["price_yuan_per_kwh"] < 0).any():
        raise ValueError("Attachment 1 has invalid electricity prices")

    sheets = pd.read_excel(Q2_INPUT_PATH, sheet_name=None)
    required_sheets = ["小区负载", "光伏发电实际功率"]
    if list(sheets) != required_sheets:
        raise ValueError(f"Unexpected Attachment 2 sheets: {list(sheets)}")
    load_wide = sheets[required_sheets[0]]
    pv_wide = sheets[required_sheets[1]]
    if load_wide.shape != (365, N_SLOTS + 1) or pv_wide.shape != (365, N_SLOTS + 1):
        raise ValueError("Attachment 2 must contain 365 date rows and 144 slots per sheet")
    if load_wide.columns.tolist() != pv_wide.columns.tolist():
        raise ValueError("Load and PV sheets have different time-column orders")
    if load_wide.iloc[:, 0].tolist() != pv_wide.iloc[:, 0].tolist():
        raise ValueError("Load and PV sheets have different date orders")
    if load_wide.isna().any().any() or pv_wide.isna().any().any():
        raise ValueError("Attachment 2 contains missing values")

    dates = pd.to_datetime(load_wide.iloc[:, 0], errors="raise").dt.normalize()
    if not dates.is_monotonic_increasing or dates.nunique() != 365:
        raise ValueError("Attachment 2 dates must be a continuous, unique annual sequence")
    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    if not pd.DatetimeIndex(dates).equals(expected_dates):
        raise ValueError("Attachment 2 date range is not 2025-01-01 through 2025-12-31")

    load_values = load_wide.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    pv_values = pv_wide.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if (load_values < 0).any() or (pv_values < 0).any():
        raise ValueError("Attachment 2 contains a negative power value")

    date_values = dates.to_numpy().repeat(N_SLOTS)
    idx_values = list(range(N_SLOTS)) * len(dates)
    actual = pd.DataFrame(
        {
            "date": date_values,
            "idx": idx_values,
            "time_label": [str(value) for value in load_wide.columns[1:]] * len(dates),
            "load_kw": load_values.reshape(-1),
            "pv_kw": pv_values.reshape(-1),
        }
    )
    from src.config import DELTA_T

    actual["load_kwh"] = actual["load_kw"] * DELTA_T
    actual["pv_kwh"] = actual["pv_kw"] * DELTA_T
    actual["net_load_kwh"] = actual["load_kwh"] - actual["pv_kwh"]
    return actual, price


if __name__ == "__main__":
    result = inspect_q1_files()
    print(f"Inspection passed: {result['passed']}")
    print(f"Report: {result['report_path']}")
