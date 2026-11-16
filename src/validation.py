"""Numerical validation for Question-1 optimisation results."""

from __future__ import annotations

from pathlib import Path
from numbers import Real

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.config import (
    E_B_MAX,
    N_PERIODS,
    Q1_FOUR_HOUR_BLOCKS,
    Q1_TABLE1_PERIODS,
    S_FINAL,
    S_INIT,
    S_MAX,
    S_MIN,
)


def validate_q1(solution: pd.DataFrame, solver_status: str, objective_cost: float, tolerance: float = 1e-6) -> dict:
    """Return physical and accounting checks after the LP is solved."""
    balance_error = (
        solution["purchase_kwh"] + solution["pv_kwh"] + solution["discharge_kwh"]
        - solution["load_kwh"] - solution["charge_kwh"] - solution["curtail_kwh"]
    ).abs()
    simultaneous = (solution["charge_kwh"] > tolerance) & (solution["discharge_kwh"] > tolerance)
    report = {
        "solver_status": solver_status,
        "objective_cost_yuan": float(objective_cost),
        "total_purchase_kwh": float(solution["purchase_kwh"].sum()),
        "max_balance_error_kwh": float(balance_error.max()),
        "soc_min_kwh": float(min(solution["soc_start_kwh"].min(), solution["soc_end_kwh"].min())),
        "soc_max_kwh": float(max(solution["soc_start_kwh"].max(), solution["soc_end_kwh"].max())),
        "soc_start_kwh": float(solution["soc_start_kwh"].iloc[0]),
        "soc_end_kwh": float(solution["soc_end_kwh"].iloc[-1]),
        "max_charge_kwh": float(solution["charge_kwh"].max()),
        "max_discharge_kwh": float(solution["discharge_kwh"].max()),
        "simultaneous_charge_discharge_count": int(simultaneous.sum()),
        "recomputed_cost_yuan": float(solution["period_cost_yuan"].sum()),
    }
    report["cost_difference_yuan"] = abs(report["objective_cost_yuan"] - report["recomputed_cost_yuan"])
    report["passed"] = bool(
        report["max_balance_error_kwh"] <= tolerance
        and report["soc_min_kwh"] >= S_MIN - tolerance
        and report["soc_max_kwh"] <= S_MAX + tolerance
        and abs(report["soc_start_kwh"] - S_INIT) <= tolerance
        and abs(report["soc_end_kwh"] - S_FINAL) <= tolerance
        and report["max_charge_kwh"] <= E_B_MAX + tolerance
        and report["max_discharge_kwh"] <= E_B_MAX + tolerance
        and report["cost_difference_yuan"] <= tolerance
    )
    return report


def write_validation_report(report: dict, path: Path) -> None:
    """Persist the validation report; simultaneous operation is diagnostic only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Question 1 validation", "=" * 30]
    lines.extend(f"{key}: {value}" for key, value in report.items())
    lines.append("Note: simultaneous charge/discharge is recorded but does not by itself fail the LP.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_q1_submission(
    solution: pd.DataFrame,
    physical_report: dict,
    result_path: Path,
    paper_tables_path: Path,
    tolerance: float = 1e-6,
) -> dict:
    """Re-open the final workbook and verify it against solution and paper tables."""
    workbook = load_workbook(result_path, data_only=False)
    required_sheets = ["计划购电量", "充放电量"]
    sheets_ok = workbook.sheetnames == required_sheets
    if not sheets_ok:
        return {
            "passed": False,
            "sheet_names_exact": False,
            "observed_sheet_names": workbook.sheetnames,
            **physical_report,
        }

    plan_sheet = workbook["计划购电量"]
    storage_sheet = workbook["充放电量"]
    purchase_cells = [plan_sheet.cell(row=row, column=2).value for row in range(2, N_PERIODS + 2)]
    nonempty_purchase_count = sum(value is not None for value in purchase_cells)
    numeric_purchase_cells = all(isinstance(value, Real) for value in purchase_cells)
    purchase_matches_solution = numeric_purchase_cells and np.allclose(
        np.asarray(purchase_cells, dtype=float),
        solution["purchase_kwh"].to_numpy(dtype=float),
        atol=tolerance,
        rtol=0.0,
    )

    target_labels_ok = all(
        plan_sheet.cell(row=excel_row, column=1).value == label
        and excel_row == index + 2
        for label, index, excel_row in Q1_TABLE1_PERIODS
    )
    storage_labels_ok = all(
        storage_sheet.cell(row=row, column=1).value == label
        for label, row, _, _ in Q1_FOUR_HOUR_BLOCKS
    )
    storage_values_ok = True
    for _, row, start, stop in Q1_FOUR_HOUR_BLOCKS:
        charge_value = storage_sheet.cell(row=row, column=2).value
        discharge_value = storage_sheet.cell(row=row, column=3).value
        expected_charge = float(solution["charge_kwh"].iloc[start:stop].sum())
        expected_discharge = float(solution["discharge_kwh"].iloc[start:stop].sum())
        storage_values_ok &= (
            isinstance(charge_value, Real)
            and isinstance(discharge_value, Real)
            and np.isclose(float(charge_value), expected_charge, atol=tolerance, rtol=0.0)
            and np.isclose(float(discharge_value), expected_discharge, atol=tolerance, rtol=0.0)
        )

    soc_start_cell = storage_sheet["E2"].value
    soc_end_cell = storage_sheet["E3"].value
    soc_cells_ok = (
        isinstance(soc_start_cell, Real)
        and isinstance(soc_end_cell, Real)
        and np.isclose(float(soc_start_cell), S_INIT, atol=tolerance, rtol=0.0)
        and np.isclose(float(soc_end_cell), S_FINAL, atol=tolerance, rtol=0.0)
    )

    paper = pd.read_csv(paper_tables_path)
    table1_periods = (
        paper[(paper["table"] == "表1") & (paper["item"] == "指定时段购电量")]
        .assign(_idx_numeric=lambda frame: pd.to_numeric(frame["idx"], errors="raise"))
        .sort_values("_idx_numeric")
    )
    table2_blocks = paper[(paper["table"] == "表2") & (paper["item"] == "区间充放电量")]
    paper_table1_ok = (
        len(table1_periods) == len(Q1_TABLE1_PERIODS)
        and table1_periods["time_period"].tolist() == [item[0] for item in Q1_TABLE1_PERIODS]
        and table1_periods["_idx_numeric"].astype(int).tolist() == [item[1] for item in Q1_TABLE1_PERIODS]
        and table1_periods["excel_row"].astype(int).tolist() == [item[2] for item in Q1_TABLE1_PERIODS]
        and np.allclose(
            table1_periods["purchase_kwh"].to_numpy(dtype=float),
            [solution.loc[index, "purchase_kwh"] for _, index, _ in Q1_TABLE1_PERIODS],
            atol=tolerance,
            rtol=0.0,
        )
    )
    total_purchase_row = paper[(paper["table"] == "表1") & (paper["item"] == "全天购电量")]
    total_cost_row = paper[(paper["table"] == "表1") & (paper["item"] == "全天购电费")]
    paper_totals_ok = (
        len(total_purchase_row) == 1
        and len(total_cost_row) == 1
        and np.isclose(
            float(total_purchase_row.iloc[0]["total_purchase_kwh"]),
            float(solution["purchase_kwh"].sum()),
            atol=tolerance,
            rtol=0.0,
        )
        and np.isclose(
            float(total_cost_row.iloc[0]["total_purchase_cost_yuan"]),
            float(solution["period_cost_yuan"].sum()),
            atol=tolerance,
            rtol=0.0,
        )
    )
    paper_table2_ok = len(table2_blocks) == len(Q1_FOUR_HOUR_BLOCKS)
    if paper_table2_ok:
        for (_, row, start, stop), (_, paper_row) in zip(Q1_FOUR_HOUR_BLOCKS, table2_blocks.iterrows()):
            paper_table2_ok &= (
                paper_row["time_period"] == storage_sheet.cell(row=row, column=1).value
                and np.isclose(float(paper_row["charge_kwh"]), solution["charge_kwh"].iloc[start:stop].sum(), atol=tolerance, rtol=0.0)
                and np.isclose(float(paper_row["discharge_kwh"]), solution["discharge_kwh"].iloc[start:stop].sum(), atol=tolerance, rtol=0.0)
            )
    paper_soc = paper[(paper["table"] == "表2") & (paper["item"] == "储电量")].set_index("time_period")
    paper_soc_ok = (
        set(paper_soc.index) == {"0:00", "24:00"}
        and np.isclose(float(paper_soc.loc["0:00", "soc_kwh"]), float(soc_start_cell), atol=tolerance, rtol=0.0)
        and np.isclose(float(paper_soc.loc["24:00", "soc_kwh"]), float(soc_end_cell), atol=tolerance, rtol=0.0)
    )
    paper_consistent = bool(paper_table1_ok and paper_totals_ok and paper_table2_ok and paper_soc_ok)
    report = {
        "sheet_names_exact": sheets_ok,
        "nonempty_planned_purchase_cells": nonempty_purchase_count,
        "planned_purchase_cells_exactly_144": nonempty_purchase_count == N_PERIODS,
        "planned_purchase_cells_numeric": numeric_purchase_cells,
        "planned_purchase_matches_solution": bool(purchase_matches_solution),
        "table1_index_and_template_mapping_exact": bool(target_labels_ok),
        "table2_template_labels_exact": bool(storage_labels_ok),
        "four_hour_charge_discharge_cells_match_solution": bool(storage_values_ok),
        "soc_cells_e2_e3_are_6000": bool(soc_cells_ok),
        "excel_and_paper_tables_consistent": paper_consistent,
        **physical_report,
    }
    report["passed"] = bool(
        physical_report["passed"]
        and report["sheet_names_exact"]
        and report["planned_purchase_cells_exactly_144"]
        and report["planned_purchase_cells_numeric"]
        and report["planned_purchase_matches_solution"]
        and report["table1_index_and_template_mapping_exact"]
        and report["table2_template_labels_exact"]
        and report["four_hour_charge_discharge_cells_match_solution"]
        and report["soc_cells_e2_e3_are_6000"]
        and report["excel_and_paper_tables_consistent"]
    )
    return report


def write_submission_check(report: dict, path: Path) -> None:
    """Write a complete final-submission audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Question 1 final submission check", "=" * 36]
    lines.extend(f"{key}: {value}" for key, value in report.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
