"""Read-only Stage-1 audit for the isolated Question-2 rebuild."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "q2_v2" / "audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def serial(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_label(value: Any) -> str:
    if isinstance(value, time):
        return value.strftime("%H:%M")
    text = str(value).strip().strip("'").strip('"')
    if text == "0:00+1":
        return text
    parts = text.split(":")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
    return text


def matrix_stats(ws, min_row: int, max_row: int, min_col: int, max_col: int) -> dict[str, Any]:
    values: list[float] = []
    missing = 0
    non_numeric = 0
    for row in ws.iter_rows(
        min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col, values_only=True
    ):
        for value in row:
            if value is None:
                missing += 1
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
            else:
                non_numeric += 1
    array = np.asarray(values, dtype=float)
    return {
        "cells": (max_row - min_row + 1) * (max_col - min_col + 1),
        "numeric_count": len(values),
        "missing_count": missing,
        "non_numeric_count": non_numeric,
        "finite": bool(np.isfinite(array).all()) if len(array) else True,
        "negative_count": int((array < 0).sum()) if len(array) else 0,
        "zero_count": int((array == 0).sum()) if len(array) else 0,
        "minimum": float(array.min()) if len(array) else None,
        "maximum": float(array.max()) if len(array) else None,
    }


def audit_attachment1() -> dict[str, Any]:
    path = RAW / "附件1.xlsx"
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    labels = [normalize_label(ws.cell(row=r, column=1).value) for r in range(2, 146)]
    prices = [ws.cell(row=r, column=2).value for r in range(2, 146)]
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "sheet_names": wb.sheetnames,
        "shape": [ws.max_row, ws.max_column],
        "headers": [serial(ws.cell(1, c).value) for c in range(1, ws.max_column + 1)],
        "period_count": len(labels),
        "first_labels": labels[:6],
        "last_labels": labels[-6:],
        "labels_unique": len(set(labels)) == len(labels),
        "price_missing_count": sum(v is None for v in prices),
        "price_minimum_yuan_per_kwh": float(min(prices)),
        "price_maximum_yuan_per_kwh": float(max(prices)),
        "q2_v2_allowed_columns": ["电价"],
        "q2_v2_forbidden_columns": ["小区负载", "光伏发电预测功率"],
    }


def audit_attachment2() -> dict[str, Any]:
    path = RAW / "附件2.xlsx"
    wb = load_workbook(path, read_only=True, data_only=True)
    sheets: dict[str, Any] = {}
    reference_labels: list[str] | None = None
    for ws in wb.worksheets:
        labels = [normalize_label(ws.cell(1, c).value) for c in range(2, 146)]
        dates = [ws.cell(r, 1).value for r in range(2, 367)]
        iso_dates = [v.date().isoformat() if isinstance(v, datetime) else str(v) for v in dates]
        expected = [
            (date(2025, 1, 1).fromordinal(date(2025, 1, 1).toordinal() + i)).isoformat()
            for i in range(365)
        ]
        sheet = {
            "shape": [ws.max_row, ws.max_column],
            "date_count": len(dates),
            "date_start": iso_dates[0],
            "date_end": iso_dates[-1],
            "dates_unique": len(set(iso_dates)) == len(iso_dates),
            "dates_continuous": iso_dates == expected,
            "period_count": len(labels),
            "first_labels": labels[:6],
            "last_labels": labels[-6:],
            "labels_unique": len(set(labels)) == len(labels),
            "data_stats": matrix_stats(ws, 2, 366, 2, 145),
            "unit": "kW",
        }
        if reference_labels is None:
            reference_labels = labels
        sheet["labels_match_first_sheet"] = labels == reference_labels
        sheets[ws.title] = sheet
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "sheet_names": wb.sheetnames,
        "sheets": sheets,
    }


def audit_result2_template() -> dict[str, Any]:
    path = RAW / "result2.xlsx"
    wb = load_workbook(path, read_only=False, data_only=False)
    plan = wb["计划购电量"]
    storage = wb["充放电量"]
    emergency = wb["紧急购电量"]
    dates = [plan.cell(r, 1).value for r in range(2, 336)]
    iso_dates = [v.date().isoformat() if isinstance(v, datetime) else str(v) for v in dates]
    labels = [str(plan.cell(1, c).value) for c in range(2, 146)]
    plan_values = [plan.cell(r, c).value for r in range(2, 336) for c in range(2, 146)]
    totals = [plan.cell(r, c).value for r in range(2, 336) for c in (146, 147)]
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "sheet_names": wb.sheetnames,
        "sheet_names_exact": wb.sheetnames == ["计划购电量", "充放电量", "紧急购电量"],
        "plan_sheet": {
            "shape": [plan.max_row, plan.max_column],
            "date_range": [iso_dates[0], iso_dates[-1]],
            "date_count": len(iso_dates),
            "dates_unique": len(set(iso_dates)) == len(iso_dates),
            "period_count": len(labels),
            "first_labels": labels[:6],
            "last_labels": labels[-6:],
            "write_range": "B2:EO335",
            "daily_total_range": "EP2:EP335",
            "daily_cost_range": "EQ2:EQ335",
            "plan_area_nonempty_count": sum(v is not None for v in plan_values),
            "total_area_nonempty_count": sum(v is not None for v in totals),
            "merged_ranges": [str(x) for x in plan.merged_cells.ranges],
        },
        "storage_sheet": {
            "shape": [storage.max_row, storage.max_column],
            "headers": [serial(storage.cell(1, c).value) for c in range(1, 7)],
            "four_hour_labels": [storage.cell(r, 2).value for r in range(2, 8)],
            "soc_labels": [serial(storage.cell(r, 5).value) for r in range(2, 4)],
            "template_example_dates": [
                serial(storage.cell(r, 1).value)
                for r in range(2, storage.max_row + 1)
                if storage.cell(r, 1).value is not None
            ],
            "charge_discharge_nonempty_count": sum(
                storage.cell(r, c).value is not None
                for r in range(2, storage.max_row + 1)
                for c in (3, 4)
            ),
            "soc_value_nonempty_count": sum(
                storage.cell(r, 6).value is not None for r in range(2, storage.max_row + 1)
            ),
            "merged_ranges": [str(x) for x in storage.merged_cells.ranges],
        },
        "emergency_sheet": {
            "shape": [emergency.max_row, emergency.max_column],
            "headers": [serial(emergency.cell(1, c).value) for c in range(1, 4)],
            "template_example_dates": [
                serial(emergency.cell(r, 1).value)
                for r in range(2, emergency.max_row + 1)
                if emergency.cell(r, 1).value is not None
            ],
            "event_fields_nonempty_count": sum(
                emergency.cell(r, c).value is not None
                for r in range(2, emergency.max_row + 1)
                for c in (2, 3)
            ),
            "merged_ranges": [str(x) for x in emergency.merged_cells.ranges],
        },
    }


def legacy_and_frozen_paths() -> list[Path]:
    paths: set[Path] = set()
    exact = [
        "data/raw/C题.pdf", "data/raw/附件1.xlsx", "data/raw/附件2.xlsx",
        "data/raw/result1.xlsx", "data/raw/result2.xlsx",
        "docs/q1_freeze_record.md", "outputs/tables/result1.xlsx",
        "outputs/tables/result2.xlsx", "data/processed/q1_day_data.csv",
        "data/processed/q1_baseline_dispatch.csv",
    ]
    for item in exact:
        path = ROOT / item
        if path.is_file():
            paths.add(path)
    patterns = [
        "src/*q2*.py", "docs/q2_*.md", "data/checks/q2*.txt",
        "outputs/tables/q2_*", "outputs/logs/q2_*",
        "outputs/figures/**/q2_*", "outputs/replay/q2_aplus/*",
    ]
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_file() and "q2_v2" not in path.parts:
                paths.add(path)
    return sorted(paths)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = legacy_and_frozen_paths()
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    audit = {
        "stage": "q2_v2_stage1_read_only_input_audit",
        "source_data_year": 2025,
        "periods_per_day": 144,
        "delta_t_hours": 1 / 6,
        "attachment1": audit_attachment1(),
        "attachment2": audit_attachment2(),
        "result2_template": audit_result2_template(),
    }
    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    audit["protected_file_count"] = len(protected)
    audit["protected_hashes_unchanged"] = before == after
    audit["result2_generated"] = any((ROOT / "outputs" / "q2_v2").glob("**/result2.xlsx"))
    (OUT / "q2_v2_input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=serial), encoding="utf-8"
    )
    (OUT / "q2_v2_protected_hashes_before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "q2_v2_protected_hashes_after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "Question 2 v2 - Stage 1 read-only audit",
        "=" * 48,
        f"protected_file_count: {len(protected)}",
        f"protected_hashes_unchanged: {before == after}",
        f"source_data_year: {audit['source_data_year']}",
        f"attachment1_periods: {audit['attachment1']['period_count']}",
        f"attachment2_sheets: {audit['attachment2']['sheet_names']}",
        f"template_sheets: {audit['result2_template']['sheet_names']}",
        f"template_plan_dates: {audit['result2_template']['plan_sheet']['date_count']}",
        f"template_plan_slots_per_day: {audit['result2_template']['plan_sheet']['period_count']}",
        f"template_plan_area_nonempty_count: {audit['result2_template']['plan_sheet']['plan_area_nonempty_count']}",
        f"result2_generated_in_q2_v2: {audit['result2_generated']}",
        "optimization_run: False",
        "full_year_run: False",
    ]
    (OUT / "q2_v2_stage1_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
