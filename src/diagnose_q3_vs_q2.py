"""Read-only diagnostics explaining Q3 versus frozen Q2 January costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Q2_JANUARY_BACKTEST_FILE, TABLE_DIR
from src.data_io import load_q2_actual_energy
from src.forecast_q2 import forecast_net_load_quantile


def _metrics(name: str, predicted, actual) -> dict:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    error = predicted - actual
    return {
        "series": name,
        "slots": len(error),
        "mae_kwh_per_slot": float(np.mean(np.abs(error))),
        "rmse_kwh_per_slot": float(np.sqrt(np.mean(error ** 2))),
        "bias_pred_minus_actual_kwh_per_slot": float(np.mean(error)),
        "underprediction_rate": float(np.mean(error < 0)),
        "predicted_total_kwh": float(predicted.sum()),
        "actual_total_kwh": float(actual.sum()),
    }


def run_diagnostic():
    start, end = pd.Timestamp("2025-01-22"), pd.Timestamp("2025-01-31")
    q3 = pd.read_csv(
        "outputs/q3_january_backtest/q3_january_slot_trajectories.csv",
        parse_dates=["date"], low_memory=False,
    )
    q3 = q3.loc[
        (q3["candidate_id"] == "h21_b12_s6000_all_updates")
        & (q3["date"] >= start) & (q3["date"] <= end)
    ].sort_values(["date", "idx"])
    if len(q3) != 10 * 144:
        raise ValueError("Q3 selected January trajectory is incomplete")

    actual, _ = load_q2_actual_energy()
    q2_frames = []
    for day in pd.date_range(start, end, freq="D"):
        forecast = forecast_net_load_quantile(actual, day, 7, 0.75)
        observed = actual.loc[actual["date"] == day].sort_values("idx")
        q2_frames.append(pd.DataFrame({
            "predicted": forecast["forecast_net_load_kwh"].to_numpy(dtype=float),
            "actual": observed["net_load_kwh"].to_numpy(dtype=float),
        }))
    q2 = pd.concat(q2_frames, ignore_index=True)

    rows = [
        _metrics("q3_active_net_load_forecast", q3["active_net_load_forecast_kwh"], q3["net_load_actual_kwh"]),
        _metrics("q3_active_load_forecast", q3["active_load_forecast_kwh"], q3["load_kwh"]),
        _metrics("q3_active_pv_forecast", q3["active_pv_forecast_kwh"], q3["pv_actual_kwh"]),
        _metrics("q2_day_ahead_net_load_quantile", q2["predicted"], q2["actual"]),
    ]
    result = pd.DataFrame(rows)
    path = TABLE_DIR / "q3_q2_january_forecast_diagnostic.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")

    q2_costs = pd.read_csv(Q2_JANUARY_BACKTEST_FILE)
    q2_baseline = q2_costs.loc[q2_costs["selected"]].iloc[0]
    q2_reserve = q2_costs.loc[
        (q2_costs["w_days"] == 7) & (q2_costs["q"] == 0.75)
        & (q2_costs["Sres_kwh"] == 3600) & np.isclose(q2_costs["rho_yuan_per_kwh"], 0.6976)
    ].iloc[0]
    q3_cost = float(q3["actual_paid_cost_yuan"].sum())
    cost_rows = pd.DataFrame([
        {
            "scheme": "q3_selected_all_updates",
            "normal_or_contract_cost_yuan": float(q3["contract_settlement_cost_yuan"].sum()),
            "emergency_cost_yuan": float(q3["emergency_purchase_cost_yuan"].sum()),
            "actual_total_cost_yuan": q3_cost,
            "emergency_purchase_kwh": float(q3["emergency_purchase_kwh"].sum()),
        },
        {
            "scheme": "q2_january_cost_baseline",
            "normal_or_contract_cost_yuan": float(q2_baseline["plan_purchase_cost_yuan"]),
            "emergency_cost_yuan": float(q2_baseline["emergency_purchase_cost_yuan"]),
            "actual_total_cost_yuan": float(q2_baseline["total_cost_yuan"]),
            "emergency_purchase_kwh": float(q2_baseline["emergency_purchase_kwh"]),
        },
        {
            "scheme": "q2_january_reserve_3600",
            "normal_or_contract_cost_yuan": float(q2_reserve["plan_purchase_cost_yuan"]),
            "emergency_cost_yuan": float(q2_reserve["emergency_purchase_cost_yuan"]),
            "actual_total_cost_yuan": float(q2_reserve["total_cost_yuan"]),
            "emergency_purchase_kwh": float(q2_reserve["emergency_purchase_kwh"]),
        },
    ])
    cost_path = TABLE_DIR / "q3_q2_january_cost_diagnostic.csv"
    cost_rows.to_csv(cost_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    print(result.to_string(index=False))
    print(cost_rows.to_string(index=False))
    return result, cost_rows


if __name__ == "__main__":
    run_diagnostic()
