"""Run a three-day causal closure for Question 3 without result3 export."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    E_B_MAX,
    Q3_FIRST_MODIFIABLE_SLOT,
    Q3_ISSUE_HOURS,
    Q3_TEST_END,
    Q3_TEST_OUTPUT_DIR,
    Q3_TEST_RESERVE_PENALTY_YUAN_PER_KWH,
    Q3_TEST_RESERVE_SOC_KWH,
    Q3_TEST_START,
    Q3_VALIDATION_TOLERANCE,
    RAW_DATA_DIR,
    S_INIT,
    S_MAX,
    S_MIN,
)
from src.data_io import load_q2_actual_energy
from src.dispatch_q3_realtime import dispatch_q3_slot
from src.forecast_q3 import add_q3_slot_datetimes, build_q3_forecast_version, load_q3_pv_forecast_records
from src.model_q2_day_ahead import get_highs_solver
from src.model_q3_rolling import solve_q3_reference_plan
from src.q3_settlement import settle_adjusted_purchase, settlement_examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run_q3_small_validation() -> dict:
    """Execute the fixed engineering test on 2025-01-15 through 01-17."""
    output_dir = Q3_TEST_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    template = RAW_DATA_DIR / "result3.xlsx"
    template_hash_before = _sha256(template)
    actual, price = load_q2_actual_energy()
    actual = add_q3_slot_datetimes(actual)
    pv_records = load_q3_pv_forecast_records()
    price_map = price.set_index("idx")["price_yuan_per_kwh"]
    dates = pd.date_range(Q3_TEST_START, Q3_TEST_END, freq="D")
    if len(dates) != 3:
        raise RuntimeError("The minimal closure must remain a three-day test")
    solver, solver_name = get_highs_solver()

    forecast_versions: list[pd.DataFrame] = []
    plan_versions: list[pd.DataFrame] = []
    execution_rows: list[dict] = []
    day_rows: list[dict] = []
    soc = float(S_INIT)
    simultaneous_update_count = 0

    for day in dates:
        day_actual = actual.loc[actual["date"] == day].sort_values("idx").reset_index(drop=True)
        if len(day_actual) != 144:
            raise ValueError(f"{day.date()} does not have 144 actual slots")
        day_initial_soc = soc

        forecast0 = build_q3_forecast_version(actual, pv_records, day, 0)
        forecast_versions.append(forecast0.assign(version="00:00"))
        plan0, meta0 = solve_q3_reference_plan(
            forecast0, price, soc, Q3_TEST_RESERVE_SOC_KWH,
            Q3_TEST_RESERVE_PENALTY_YUAN_PER_KWH, original_purchase=None, solver=solver,
        )
        original = plan0.set_index("idx")["reference_purchase_kwh"].copy()
        effective = original.copy()
        previous_at_last_update = original.copy()
        reference_soc_end = plan0.set_index("idx")["reference_soc_end_kwh"].copy()
        active_issue = pd.Series(0, index=range(144), dtype=int)
        p0_record = plan0.copy()
        p0_record["version"] = "00:00_original"
        p0_record["original_purchase_kwh"] = p0_record["reference_purchase_kwh"]
        p0_record["previous_effective_purchase_kwh"] = p0_record["reference_purchase_kwh"]
        p0_record["delta_vs_previous_kwh"] = 0.0
        p0_record["decision_initial_soc_kwh"] = day_initial_soc
        p0_record["solver"] = solver_name
        p0_record["reserve_shortfall_kwh"] = meta0["reserve_shortfall_kwh"]
        plan_versions.append(p0_record)

        update_by_start = {Q3_FIRST_MODIFIABLE_SLOT[h]: h for h in Q3_ISSUE_HOURS if h != 0}
        for t in range(144):
            if t in update_by_start:
                issue_hour = update_by_start[t]
                version = build_q3_forecast_version(actual, pv_records, day, issue_hour)
                forecast_versions.append(version.assign(version=f"{issue_hour:02d}:00"))
                old_effective = effective.copy()
                updated, metadata = solve_q3_reference_plan(
                    version, price, soc, Q3_TEST_RESERVE_SOC_KWH,
                    Q3_TEST_RESERVE_PENALTY_YUAN_PER_KWH,
                    original_purchase=original, solver=solver,
                )
                idx = updated["idx"].astype(int)
                updated["version"] = f"{issue_hour:02d}:00_adjustment"
                updated["original_purchase_kwh"] = [float(original.at[i]) for i in idx]
                updated["previous_effective_purchase_kwh"] = [float(old_effective.at[i]) for i in idx]
                updated["delta_vs_previous_kwh"] = (
                    updated["reference_purchase_kwh"] - updated["previous_effective_purchase_kwh"]
                )
                updated["decision_initial_soc_kwh"] = soc
                updated["solver"] = solver_name
                updated["reserve_shortfall_kwh"] = metadata["reserve_shortfall_kwh"]
                plan_versions.append(updated)
                simultaneous_update_count += int(metadata["simultaneous_up_down_count"])
                for row in updated.itertuples(index=False):
                    i = int(row.idx)
                    previous_at_last_update.at[i] = float(old_effective.at[i])
                    effective.at[i] = float(row.reference_purchase_kwh)
                    reference_soc_end.at[i] = float(row.reference_soc_end_kwh)
                    active_issue.at[i] = issue_hour

            actual_row = day_actual.iloc[t]
            physical = dispatch_q3_slot(
                actual_row=actual_row,
                effective_purchase_kwh=float(effective.at[t]),
                current_soc_kwh=soc,
                reference_soc_end_kwh=float(reference_soc_end.at[t]),
                tolerance=Q3_VALIDATION_TOLERANCE,
            )
            contract = settle_adjusted_purchase(
                [float(original.at[t])], [float(effective.at[t])], [float(price_map.at[t])]
            ).iloc[0]
            emergency_cost = 5.0 * float(price_map.at[t]) * physical["emergency_purchase_kwh"]
            execution_rows.append(
                {
                    "date": day,
                    "idx": t,
                    "time_label": actual_row["time_label"],
                    "slot_datetime": actual_row["slot_datetime"],
                    "active_plan_issue_hour": int(active_issue.at[t]),
                    "original_purchase_kwh": float(original.at[t]),
                    "previous_effective_at_last_update_kwh": float(previous_at_last_update.at[t]),
                    "effective_purchase_kwh": float(effective.at[t]),
                    "delta_vs_original_kwh": float(effective.at[t] - original.at[t]),
                    "delta_vs_previous_kwh": float(effective.at[t] - previous_at_last_update.at[t]),
                    "price_yuan_per_kwh": float(price_map.at[t]),
                    **physical,
                    "upward_adjustment_kwh": float(contract.upward_adjustment_kwh),
                    "downward_adjustment_kwh": float(contract.downward_adjustment_kwh),
                    "retained_purchase_cost_yuan": float(contract.retained_purchase_cost_yuan),
                    "downward_breach_cost_yuan": float(contract.downward_breach_cost_yuan),
                    "upward_purchase_cost_yuan": float(contract.upward_purchase_cost_yuan),
                    "contract_settlement_cost_yuan": float(contract.contract_settlement_cost_yuan),
                    "emergency_purchase_cost_yuan": emergency_cost,
                    "actual_total_cost_yuan": float(contract.contract_settlement_cost_yuan) + emergency_cost,
                }
            )
            soc = physical["soc_end_kwh"]

        execution_day = pd.DataFrame(execution_rows).loc[lambda x: x["date"] == day]
        day_rows.append(
            {
                "date": day,
                "soc_start_kwh": day_initial_soc,
                "soc_end_kwh": soc,
                "contract_settlement_cost_yuan": execution_day["contract_settlement_cost_yuan"].sum(),
                "emergency_purchase_cost_yuan": execution_day["emergency_purchase_cost_yuan"].sum(),
                "actual_total_cost_yuan": execution_day["actual_total_cost_yuan"].sum(),
                "emergency_purchase_kwh": execution_day["emergency_purchase_kwh"].sum(),
                "upward_adjustment_kwh": execution_day["upward_adjustment_kwh"].sum(),
                "downward_adjustment_kwh": execution_day["downward_adjustment_kwh"].sum(),
            }
        )
    forecasts = pd.concat(forecast_versions, ignore_index=True)
    plans = pd.concat(plan_versions, ignore_index=True)
    execution = pd.DataFrame(execution_rows)
    daily = pd.DataFrame(day_rows)
    examples = settlement_examples()

    forecast_path = output_dir / "q3_forecast_versions.csv"
    plan_path = output_dir / "q3_plan_versions.csv"
    execution_path = output_dir / "q3_execution_settlement.csv"
    daily_path = output_dir / "q3_daily_summary.csv"
    example_path = output_dir / "q3_settlement_examples.csv"
    forecasts.to_csv(forecast_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    plans.to_csv(plan_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    execution.to_csv(execution_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    examples.to_csv(example_path, index=False, encoding="utf-8-sig", float_format="%.10f")

    tol = Q3_VALIDATION_TOLERANCE
    final_delta = execution["delta_vs_original_kwh"]
    branch_counts = {
        "upward_slots": int((final_delta > tol).sum()),
        "downward_slots": int((final_delta < -tol).sum()),
        "unchanged_slots": int((np.abs(final_delta) <= tol).sum()),
        "emergency_slots": int((execution["emergency_purchase_kwh"] > tol).sum()),
    }
    independent_contract = settle_adjusted_purchase(
        execution["original_purchase_kwh"],
        execution["effective_purchase_kwh"],
        execution["price_yuan_per_kwh"],
    )["contract_settlement_cost_yuan"].to_numpy()
    independent_emergency = (
        5.0 * execution["price_yuan_per_kwh"] * execution["emergency_purchase_kwh"]
    ).to_numpy()
    cost_error = float(np.max(np.abs(
        execution["actual_total_cost_yuan"].to_numpy()
        - independent_contract - independent_emergency
    )))
    if len(daily) > 1:
        cross_day_error = float(np.max(np.abs(
            daily["soc_start_kwh"].iloc[1:].to_numpy(dtype=float)
            - daily["soc_end_kwh"].iloc[:-1].to_numpy(dtype=float)
        )))
    else:
        cross_day_error = 0.0
    max_balance = float(execution["balance_residual_kwh"].abs().max())
    soc_min = float(min(execution["soc_start_kwh"].min(), execution["soc_end_kwh"].min()))
    soc_max = float(max(execution["soc_start_kwh"].max(), execution["soc_end_kwh"].max()))
    max_charge = float(execution["charge_rt_kwh"].max())
    max_discharge = float(execution["discharge_rt_kwh"].max())
    emergency_charge_overlap = int((
        (execution["emergency_purchase_kwh"] > tol) & (execution["charge_rt_kwh"] > tol)
    ).sum())
    simultaneous_rt = int((
        (execution["charge_rt_kwh"] > tol) & (execution["discharge_rt_kwh"] > tol)
    ).sum())
    version_counts = execution.groupby("active_plan_issue_hour").size().to_dict()
    expected_version_counts = {0: 35 * len(dates), 6: 36 * len(dates), 12: 36 * len(dates), 18: 37 * len(dates)}
    mapping_checks = {
        "t35_target": str(forecasts.loc[(forecasts["date"] == dates[0]) & (forecasts["issue_hour"] == 6) & (forecasts["idx"] == 35), "target_datetime"].iloc[0]),
        "t71_target": str(forecasts.loc[(forecasts["date"] == dates[0]) & (forecasts["issue_hour"] == 12) & (forecasts["idx"] == 71), "target_datetime"].iloc[0]),
        "t107_target": str(forecasts.loc[(forecasts["date"] == dates[0]) & (forecasts["issue_hour"] == 18) & (forecasts["idx"] == 107), "target_datetime"].iloc[0]),
        "t143_from_00_right_anchor": str(forecasts.loc[(forecasts["date"] == dates[0]) & (forecasts["issue_hour"] == 0) & (forecasts["idx"] == 143), "pv_right_anchor_datetime"].iloc[0]),
        "t143_from_18_right_anchor": str(forecasts.loc[(forecasts["date"] == dates[0]) & (forecasts["issue_hour"] == 18) & (forecasts["idx"] == 143), "pv_right_anchor_datetime"].iloc[0]),
    }
    example_expected = np.array([90.0, 130.0, 100.0, 115.0, 95.0])
    example_error = float(np.max(np.abs(examples["contract_settlement_cost_yuan"] - example_expected)))
    template_hash_after = _sha256(template)
    checks = {
        "forecast_rows": len(forecasts),
        "plan_version_rows": len(plans),
        "execution_rows": len(execution),
        "days": len(daily),
        **branch_counts,
        "forecast_version_switch_counts": version_counts,
        "expected_version_switch_counts": expected_version_counts,
        "max_balance_error_kwh": max_balance,
        "soc_min_kwh": soc_min,
        "soc_max_kwh": soc_max,
        "max_charge_kwh": max_charge,
        "max_discharge_kwh": max_discharge,
        "simultaneous_rt_charge_discharge_count": simultaneous_rt,
        "simultaneous_update_up_down_count": simultaneous_update_count,
        "emergency_charge_overlap_count": emergency_charge_overlap,
        "cross_day_soc_error_kwh": cross_day_error,
        "max_cost_recalculation_error_yuan": cost_error,
        "settlement_example_max_error_yuan": example_error,
        "no_future_load_leakage": bool(forecasts["no_future_actual_load_used"].all()),
        "no_actual_pv_used_for_forecast": bool((~forecasts["pv_actual_observation_used"]).all()),
        "pv_information_causal": bool(forecasts["pv_information_causal"].all()),
        "forecast_available_at_decision": bool(forecasts["forecast_visible_at_decision"].all()),
        "targets_not_before_issue": bool((forecasts["target_datetime"] >= forecasts["issue_datetime"]).all()),
        "template_hash_before": template_hash_before,
        "template_hash_after": template_hash_after,
        "template_unchanged": template_hash_before == template_hash_after,
        **mapping_checks,
    }
    passed = all(
        [
            len(execution) == len(dates) * 144,
            version_counts == expected_version_counts,
            max_balance <= tol,
            soc_min >= S_MIN - tol,
            soc_max <= S_MAX + tol,
            max_charge <= E_B_MAX + tol,
            max_discharge <= E_B_MAX + tol,
            simultaneous_rt == 0,
            simultaneous_update_count == 0,
            emergency_charge_overlap == 0,
            cross_day_error <= tol,
            cost_error <= tol,
            example_error <= tol,
            checks["no_future_load_leakage"],
            checks["no_actual_pv_used_for_forecast"],
            checks["pv_information_causal"],
            checks["forecast_available_at_decision"],
            checks["targets_not_before_issue"],
            checks["template_unchanged"],
            branch_counts["upward_slots"] > 0,
            branch_counts["downward_slots"] > 0,
            branch_counts["unchanged_slots"] > 0,
            branch_counts["emergency_slots"] > 0,
        ]
    )
    checks["passed"] = bool(passed)

    log_path = output_dir / "q3_small_validation.txt"
    lines = [
        "Question 3 minimal causal closure",
        "=" * 40,
        f"test_period: {dates[0].date()} through {dates[-1].date()}",
        "scope: engineering validation only; not parameter selection or formal evaluation",
        f"solver: {solver_name}",
        "settlement_assumption: each slot settled once, final effective amount versus 00:00 original plan",
        f"reserve_test_setting: Sres={Q3_TEST_RESERVE_SOC_KWH} kWh, rho={Q3_TEST_RESERVE_PENALTY_YUAN_PER_KWH} yuan/kWh (no extra reserve; not selected)",
        "load_forecast: 14-day exponentially weighted same-slot load, 7-day half-life; intraday median residual correction from last 12 observed slots",
        "pv_mapping: previous published forecast at issue + linear interpolation to current issue+1...24 hourly forecast anchors; no actual PV anchor",
        "",
        *[f"{key}: {value}" for key, value in checks.items()],
        "",
        "outputs:",
        str(forecast_path),
        str(plan_path),
        str(execution_path),
        str(daily_path),
        str(example_path),
        "result3.xlsx generated: False",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Question-3 minimal closure failed; see {log_path}")
    return {
        "checks": checks,
        "daily": daily,
        "outputs": [forecast_path, plan_path, execution_path, daily_path, example_path, log_path],
    }


if __name__ == "__main__":
    outcome = run_q3_small_validation()
    print(outcome["daily"].to_string(index=False))
    print(outcome["checks"])
