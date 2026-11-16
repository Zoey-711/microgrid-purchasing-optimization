"""Paired 50/100 kWh DP-grid check for refined competitive candidates."""

from __future__ import annotations

import pandas as pd

from .config import ROOT
from .data import load_inputs
from .execution import build_path_value_functions, execute_day
from .forecast import build_causal_replay
from .scenarios import build_joint_scenarios


OUT = ROOT / "outputs" / "q2_v2" / "stage3"
POLICIES = {
    "P2_M14_v0_dp100_refined": 0.0,
    "P3_M14_v04_dp100_refined": 0.4,
    "P4_M14_v08_dp100_refined": 0.8,
}


def run() -> None:
    data = load_inputs()
    replay = build_causal_replay(data)
    rows = []
    for policy, v in POLICIES.items():
        base = OUT / "refine" / policy
        daily = pd.read_csv(base / "daily_results.csv")
        plans = pd.read_csv(base / "day_ahead_plans.csv")
        slots100 = pd.read_csv(base / "slot_trajectories.csv")
        for date in pd.date_range("2025-01-22", "2025-01-31").strftime("%Y-%m-%d"):
            d = data.day_index(date)
            initial = float(daily.loc[daily["date"] == date, "soc_start_kwh"].iloc[0])
            plan = plans[plans["date"] == date].sort_values("slot")["planned_purchase_kwh"].to_numpy()
            scenarios = build_joint_scenarios(data, replay, d, 14)
            value50 = build_path_value_functions(scenarios, plan, data.price_yuan_per_kwh, 50.0, v)
            frame50, meta50 = execute_day(date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
                                          data.price_yuan_per_kwh, initial, "dp", value50)
            frame100 = slots100[slots100["date"] == date].sort_values("slot")
            rows.append({
                "policy_id": policy, "date": date, "terminal_value": v,
                "same_plan_same_initial_soc": True, "initial_soc_kwh": initial,
                "cost_grid100_yuan": float(frame100["actual_cost_yuan"].sum()),
                "cost_grid50_yuan": meta50["actual_total_cost_yuan"],
                "cost_50_minus_100_yuan": meta50["actual_total_cost_yuan"] - float(frame100["actual_cost_yuan"].sum()),
                "emergency_grid100_kwh": float(frame100["emergency_purchase_kwh"].sum()),
                "emergency_grid50_kwh": meta50["emergency_purchase_kwh"],
                "end_soc_grid100_kwh": float(frame100.iloc[-1]["soc_end_kwh"]),
                "end_soc_grid50_kwh": meta50["soc_end_kwh"],
                "sum_abs_net_action_difference_kwh": float(abs(
                    (frame50["charge_kwh"]-frame50["discharge_kwh"]).to_numpy()
                    - (frame100["charge_kwh"]-frame100["discharge_kwh"]).to_numpy()).sum()),
                "max_abs_soc_difference_kwh": float(abs(frame50["soc_end_kwh"].to_numpy()-frame100["soc_end_kwh"].to_numpy()).max()),
                "path_value_grid50_runtime_seconds": value50.runtime_seconds,
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "q2_v2_january_grid_precision_daily.csv", index=False)
    summary = detail.groupby("policy_id", as_index=False).agg(
        cost_grid100_yuan=("cost_grid100_yuan", "sum"), cost_grid50_yuan=("cost_grid50_yuan", "sum"),
        cost_50_minus_100_yuan=("cost_50_minus_100_yuan", "sum"),
        emergency_grid100_kwh=("emergency_grid100_kwh", "sum"), emergency_grid50_kwh=("emergency_grid50_kwh", "sum"),
        sum_abs_net_action_difference_kwh=("sum_abs_net_action_difference_kwh", "sum"),
        max_abs_soc_difference_kwh=("max_abs_soc_difference_kwh", "max"),
        path_value_grid50_runtime_seconds=("path_value_grid50_runtime_seconds", "sum"),
    )
    summary.to_csv(OUT / "q2_v2_january_grid_precision_summary.csv", index=False)


if __name__ == "__main__":
    run()
