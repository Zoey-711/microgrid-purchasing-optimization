"""Validation checks for the causal Question-2 January backtest."""

from __future__ import annotations

import numpy as np

from src.config import E_B_MAX, N_PERIODS, S_MAX, S_MIN


def validate_q2_january_backtest(day_runs: list[dict], tolerance: float = 1e-6) -> dict:
    """Validate causality, physical feasibility, cross-day SOC, and accounting."""
    if not day_runs:
        raise ValueError("at least one daily run is required for validation")

    all_execution = []
    history_end_dates = []
    forecast_dates = []
    cross_day_errors = []
    previous_end = None
    for run in day_runs:
        forecast = run["forecast"]
        execution = run["execution"]
        if len(forecast) != N_PERIODS or len(execution) != N_PERIODS:
            raise ValueError("each run must contain exactly 144 forecast and execution slots")
        forecast_date = forecast["forecast_date"].iloc[0]
        history_end = forecast["history_end_date"].iloc[0]
        forecast_dates.append(forecast_date)
        history_end_dates.append(history_end)
        if previous_end is not None:
            cross_day_errors.append(abs(float(execution["rt_soc_start_kwh"].iloc[0]) - previous_end))
        previous_end = float(execution["rt_soc_end_kwh"].iloc[-1])
        all_execution.append(execution)

    import pandas as pd

    result = pd.concat(all_execution, ignore_index=True)
    balance_error = np.abs(result["balance_residual_kwh"])
    soc_values = pd.concat([result["rt_soc_start_kwh"], result["rt_soc_end_kwh"]])
    charge = result["rt_charge_kwh"]
    discharge = result["rt_discharge_kwh"]
    emergency = result["emergency_purchase_kwh"]
    plan_cost = float(result["plan_purchase_cost_yuan"].sum())
    emergency_cost = float(result["emergency_purchase_cost_yuan"].sum())
    recomputed_total = plan_cost + emergency_cost
    reported_total = float(sum(run["rt_metadata"]["total_actual_cost_yuan"] for run in day_runs))
    report = {
        "days_backtested": len(day_runs),
        "no_future_data_leakage": bool(all(history < day for history, day in zip(history_end_dates, forecast_dates))),
        "max_forecast_history_end_date": str(max(history_end_dates).date()),
        "min_forecast_date": str(min(forecast_dates).date()),
        "max_soc_cross_day_error_kwh": float(max(cross_day_errors, default=0.0)),
        "max_energy_balance_error_kwh": float(balance_error.max()),
        "soc_min_kwh": float(soc_values.min()),
        "soc_max_kwh": float(soc_values.max()),
        "max_charge_kwh": float(charge.max()),
        "max_discharge_kwh": float(discharge.max()),
        "simultaneous_charge_discharge_count": int(((charge > tolerance) & (discharge > tolerance)).sum()),
        "emergency_charge_overlap_count": int(((emergency > tolerance) & (charge > tolerance)).sum()),
        "plan_purchase_cost_yuan_recomputed": plan_cost,
        "emergency_purchase_cost_yuan_recomputed": emergency_cost,
        "total_cost_yuan_recomputed": recomputed_total,
        "total_cost_yuan_reported": reported_total,
        "cost_recomputation_difference_yuan": abs(recomputed_total - reported_total),
        "total_emergency_purchase_kwh": float(emergency.sum()),
    }
    report["passed"] = bool(
        report["no_future_data_leakage"]
        and report["max_soc_cross_day_error_kwh"] <= tolerance
        and report["max_energy_balance_error_kwh"] <= tolerance
        and report["soc_min_kwh"] >= S_MIN - tolerance
        and report["soc_max_kwh"] <= S_MAX + tolerance
        and report["max_charge_kwh"] <= E_B_MAX + tolerance
        and report["max_discharge_kwh"] <= E_B_MAX + tolerance
        and report["simultaneous_charge_discharge_count"] == 0
        and report["emergency_charge_overlap_count"] == 0
        and report["cost_recomputation_difference_yuan"] <= tolerance
    )
    return report
