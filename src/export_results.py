"""Write solved results into copies of official result templates."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from src.config import (
    N_PERIODS,
    Q1_BASELINE_INTERFACE_FILE,
    Q1_FOUR_HOUR_BLOCKS,
    Q1_PAPER_TABLES_FILE,
    Q1_RESULT_FILE,
    Q1_TABLE1_PERIODS,
    RAW_DATA_DIR,
)


def export_q1_result(solution: pd.DataFrame, output_path: Path = Q1_RESULT_FILE) -> Path:
    """Copy raw/result1.xlsx and fill only designated Question-1 result cells."""
    if len(solution) != N_PERIODS:
        raise ValueError(f"Question-1 solution must contain exactly {N_PERIODS} periods")

    template_path = RAW_DATA_DIR / "result1.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, output_path)
    workbook = load_workbook(output_path)
    plan_sheet = workbook["计划购电量"]
    storage_sheet = workbook["充放电量"]

    for t, value in enumerate(solution["purchase_kwh"]):
        plan_sheet.cell(row=t + 2, column=2, value=float(value))

    for label, row, start, stop in Q1_FOUR_HOUR_BLOCKS:
        if storage_sheet.cell(row=row, column=1).value != label:
            raise ValueError(f"Unexpected storage-template label at row {row}")
        storage_sheet.cell(row=row, column=2, value=float(solution["charge_kwh"].iloc[start:stop].sum()))
        storage_sheet.cell(row=row, column=3, value=float(solution["discharge_kwh"].iloc[start:stop].sum()))

    storage_sheet["E2"] = float(solution["soc_start_kwh"].iloc[0])
    storage_sheet["E3"] = float(solution["soc_end_kwh"].iloc[-1])
    for sheet in (plan_sheet, storage_sheet):
        for row in sheet.iter_rows():
            for cell in row:
                if cell.column != 1 and isinstance(cell.value, (int, float)):
                    cell.number_format = "0.000"
    workbook.save(output_path)
    return output_path


def export_q1_baseline_interface(
    solution: pd.DataFrame, output_path: Path = Q1_BASELINE_INTERFACE_FILE
) -> Path:
    """Persist the standard deterministic-baseline interface for Questions 2--4.

    soc_kwh is the storage state at the start of each of the 144 decision periods.
    """
    required = [
        "time_label",
        "price",
        "load_kwh",
        "pv_kwh",
        "purchase_kwh",
        "charge_kwh",
        "discharge_kwh",
        "soc_start_kwh",
    ]
    missing = [column for column in required if column not in solution.columns]
    if missing:
        raise ValueError(f"Missing baseline-interface fields: {missing}")
    interface = solution.loc[:, required].rename(
        columns={
            "pv_kwh": "pv_forecast_kwh",
            "soc_start_kwh": "soc_kwh",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interface.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def export_q1_paper_tables(
    solution: pd.DataFrame, output_path: Path = Q1_PAPER_TABLES_FILE
) -> Path:
    """Write Table 1 and Table 2 values in a machine-checkable paper-summary CSV."""
    records = []
    for label, index, excel_row in Q1_TABLE1_PERIODS:
        records.append(
            {
                "table": "表1",
                "item": "指定时段购电量",
                "time_period": label,
                "idx": index,
                "excel_row": excel_row,
                "purchase_kwh": float(solution.loc[index, "purchase_kwh"]),
                "charge_kwh": None,
                "discharge_kwh": None,
                "soc_kwh": None,
                "total_purchase_kwh": None,
                "total_purchase_cost_yuan": None,
                "unit": "kWh",
            }
        )
    records.extend(
        [
            {
                "table": "表1",
                "item": "全天购电量",
                "time_period": "全天",
                "idx": None,
                "excel_row": None,
                "purchase_kwh": None,
                "charge_kwh": None,
                "discharge_kwh": None,
                "soc_kwh": None,
                "total_purchase_kwh": float(solution["purchase_kwh"].sum()),
                "total_purchase_cost_yuan": None,
                "unit": "kWh",
            },
            {
                "table": "表1",
                "item": "全天购电费",
                "time_period": "全天",
                "idx": None,
                "excel_row": None,
                "purchase_kwh": None,
                "charge_kwh": None,
                "discharge_kwh": None,
                "soc_kwh": None,
                "total_purchase_kwh": None,
                "total_purchase_cost_yuan": float(solution["period_cost_yuan"].sum()),
                "unit": "元",
            },
        ]
    )
    for label, excel_row, start, stop in Q1_FOUR_HOUR_BLOCKS:
        records.append(
            {
                "table": "表2",
                "item": "区间充放电量",
                "time_period": label,
                "idx": f"[{start}:{stop})",
                "excel_row": excel_row,
                "purchase_kwh": None,
                "charge_kwh": float(solution["charge_kwh"].iloc[start:stop].sum()),
                "discharge_kwh": float(solution["discharge_kwh"].iloc[start:stop].sum()),
                "soc_kwh": None,
                "total_purchase_kwh": None,
                "total_purchase_cost_yuan": None,
                "unit": "kWh",
            }
        )
    records.extend(
        [
            {
                "table": "表2",
                "item": "储电量",
                "time_period": "0:00",
                "idx": "soc[0]",
                "excel_row": "E2",
                "purchase_kwh": None,
                "charge_kwh": None,
                "discharge_kwh": None,
                "soc_kwh": float(solution["soc_start_kwh"].iloc[0]),
                "total_purchase_kwh": None,
                "total_purchase_cost_yuan": None,
                "unit": "kWh",
            },
            {
                "table": "表2",
                "item": "储电量",
                "time_period": "24:00",
                "idx": "soc[144]",
                "excel_row": "E3",
                "purchase_kwh": None,
                "charge_kwh": None,
                "discharge_kwh": None,
                "soc_kwh": float(solution["soc_end_kwh"].iloc[-1]),
                "total_purchase_kwh": None,
                "total_purchase_cost_yuan": None,
                "unit": "kWh",
            },
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
