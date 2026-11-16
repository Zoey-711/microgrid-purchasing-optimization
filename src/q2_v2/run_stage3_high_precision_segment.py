"""High-precision Jan-22--31 sequential check for the two closest candidates."""

from __future__ import annotations

import pandas as pd

from .config import ROOT
from .data import load_inputs
from .day_ahead_milp import solve_joint_scenario_milp
from .execution import build_path_value_functions, execute_day
from .forecast import build_causal_replay
from .scenarios import build_joint_scenarios


OUT = ROOT / "outputs" / "q2_v2" / "stage3"
TIME_LIMIT = 45.0
REL_GAP = 0.0001
POLICIES = {
    "P2_M14_v0_dp100_high_precision": ("P2_M14_v0_dp100_refined", 0.0),
    "P3_M14_v04_dp100_high_precision": ("P3_M14_v04_dp100_refined", 0.4),
}


def run() -> None:
    data = load_inputs()
    replay = build_causal_replay(data)
    summaries = []
    for policy_id, (refined_id, v) in POLICIES.items():
        print(f"START {policy_id}", flush=True)
        base = OUT / "refine" / refined_id
        warm = pd.read_csv(base / "daily_results.csv")
        soc = float(warm.loc[warm["date"] == "2025-01-22", "soc_start_kwh"].iloc[0])
        initial = soc
        slots_all, plans_all, daily_all, solver_all = [], [], [], []
        for date in pd.date_range("2025-01-22", "2025-01-31").strftime("%Y-%m-%d"):
            d = data.day_index(date)
            scenarios = build_joint_scenarios(data, replay, d, 14)
            solved = solve_joint_scenario_milp(
                scenarios, data.price_yuan_per_kwh, soc, v,
                time_limit_seconds=TIME_LIMIT, relative_gap=REL_GAP, random_seed=0,
            )
            plan = solved.plan["planned_purchase_kwh"].to_numpy()
            value = build_path_value_functions(scenarios, plan, data.price_yuan_per_kwh, 100.0, v)
            frame, meta = execute_day(date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
                                      data.price_yuan_per_kwh, soc, "dp", value)
            frame.insert(0, "policy_id", policy_id)
            slots_all.append(frame)
            plan_frame = solved.plan.copy(); plan_frame.insert(0, "date", date); plan_frame.insert(0, "policy_id", policy_id)
            plans_all.append(plan_frame)
            daily_all.append({"policy_id": policy_id, **meta})
            solver_all.append({"policy_id": policy_id, "date": date, **solved.metadata})
            soc = float(meta["soc_end_kwh"])
            print(f"{policy_id} {date} SOC={soc:.3f} cost={meta['actual_total_cost_yuan']:.3f}", flush=True)
        folder = OUT / "high_precision" / policy_id
        folder.mkdir(parents=True, exist_ok=True)
        slots = pd.concat(slots_all, ignore_index=True)
        plans = pd.concat(plans_all, ignore_index=True)
        daily = pd.DataFrame(daily_all)
        solvers = pd.DataFrame(solver_all)
        slots.to_csv(folder / "slot_trajectories.csv", index=False)
        plans.to_csv(folder / "day_ahead_plans.csv", index=False)
        daily.to_csv(folder / "daily_results.csv", index=False)
        solvers.to_csv(folder / "solver_results.csv", index=False)
        summaries.append({
            "policy_id": policy_id, "source_warmup_policy": refined_id,
            "development_initial_soc_kwh": initial, "development_terminal_soc_kwh": soc,
            "plan_cost_yuan": daily["plan_cost_yuan"].sum(),
            "emergency_cost_yuan": daily["emergency_cost_yuan"].sum(),
            "actual_total_cost_yuan": daily["actual_total_cost_yuan"].sum(),
            "emergency_purchase_kwh": daily["emergency_purchase_kwh"].sum(),
            "charge_kwh": daily["charge_kwh"].sum(), "discharge_kwh": daily["discharge_kwh"].sum(),
            "conversion_loss_kwh": daily["conversion_loss_kwh"].sum(), "spill_kwh": daily["spill_kwh"].sum(),
            "solver_time_seconds": solvers["solve_time_seconds"].sum(),
            "optimal_certified_days": solvers["optimal_solution_certified"].sum(),
            "time_limit_feasible_days": (solvers["termination_condition"] == "maxTimeLimit").sum(),
            "max_relative_gap": solvers["mip_gap"].max(), "max_absolute_gap_yuan": solvers["absolute_gap_yuan"].max(),
        })
        print(f"DONE {policy_id}", flush=True)
    pd.DataFrame(summaries).to_csv(OUT / "q2_v2_january_high_precision_candidates.csv", index=False)


if __name__ == "__main__":
    run()
