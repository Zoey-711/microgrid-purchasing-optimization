"""Independent read-back validation for Q2-v2 Stage 2 outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ENERGY_MAX_KWH, ETA_CHARGE, ETA_DISCHARGE, ROOT, SOC_MAX_KWH, SOC_MIN_KWH, TOL


OUT = ROOT / "outputs" / "q2_v2" / "stage2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run() -> dict:
    execution = pd.read_csv(OUT / "q2_v2_slot_trajectories.csv")
    plans = pd.read_csv(OUT / "q2_v2_day_ahead_plans.csv")
    dispatch = pd.read_csv(OUT / "q2_v2_milp_scenario_dispatch.csv")
    daily = pd.read_csv(OUT / "q2_v2_daily_results.csv")
    solvers = pd.read_csv(OUT / "q2_v2_solver_results.csv")
    info = pd.read_csv(OUT / "q2_v2_information_boundary.csv")
    paired = pd.read_csv(OUT / "q2_v2_paired_executor_comparison.csv")
    grid = pd.read_csv(OUT / "q2_v2_grid_sensitivity.csv")

    plan_lookup = plans[["run_id", "date", "slot", "planned_purchase_kwh"]].rename(
        columns={"planned_purchase_kwh": "locked_plan_kwh"}
    )
    merged = execution.merge(
        plan_lookup,
        left_on=["plan_run_id", "date", "slot"],
        right_on=["run_id", "date", "slot"],
        how="left",
        validate="many_to_one",
    )
    actual_plan_lock_error = float(
        np.max(np.abs(merged["planned_purchase_kwh"] - merged["locked_plan_kwh"]))
    )

    dp_plan = plans.rename(columns={"run_id": "plan_run_id"})[
        ["plan_run_id", "date", "slot", "planned_purchase_kwh"]
    ]
    scenario = dispatch.rename(columns={"run_id": "plan_run_id"}).merge(
        dp_plan, on=["plan_run_id", "date", "slot"], validate="many_to_one"
    )
    scenario_balance = (
        scenario["planned_purchase_kwh"]
        + scenario["pv_scenario_kwh"]
        + scenario["discharge_scenario_kwh"]
        + scenario["emergency_scenario_kwh"]
        - scenario["load_scenario_kwh"]
        - scenario["charge_scenario_kwh"]
        - scenario["spill_scenario_kwh"]
    )
    scenario_transition = (
        scenario["soc_end_scenario_kwh"]
        - scenario["soc_start_scenario_kwh"]
        - ETA_CHARGE * scenario["charge_scenario_kwh"]
        + scenario["discharge_scenario_kwh"] / ETA_DISCHARGE
    )

    cross_day_errors: list[float] = []
    for _, group in daily[daily["comparison_scope"] == "independent_cross_day"].groupby("strategy"):
        group = group.sort_values("date")
        cross_day_errors.extend(
            np.abs(group["soc_start_kwh"].to_numpy()[1:] - group["soc_end_kwh"].to_numpy()[:-1]).tolist()
        )

    before = json.loads((OUT / "protected_hashes_before.json").read_text(encoding="utf-8"))
    after_recorded = json.loads((OUT / "protected_hashes_after.json").read_text(encoding="utf-8"))
    current = {name: sha256(ROOT / name) for name in before}
    changed = sorted(name for name in before if before[name] != after_recorded[name] or before[name] != current[name])

    result = {
        "stage2_dates": sorted(execution["date"].unique().tolist()),
        "actual_slot_rows": int(len(execution)),
        "scenario_slot_rows": int(len(scenario)),
        "solver_runs": int(len(solvers)),
        "certified_optimal_runs": int(solvers["optimal_solution_certified"].astype(bool).sum()),
        "time_limit_feasible_runs": int((~solvers["optimal_solution_certified"].astype(bool)).sum()),
        "maximum_reported_mip_gap": float(solvers["mip_gap"].max()),
        "max_actual_balance_residual_kwh": float(execution["balance_residual_kwh"].abs().max()),
        "max_scenario_balance_residual_kwh": float(scenario_balance.abs().max()),
        "max_scenario_soc_transition_residual_kwh": float(scenario_transition.abs().max()),
        "actual_soc_min_kwh": float(min(execution["soc_start_kwh"].min(), execution["soc_end_kwh"].min())),
        "actual_soc_max_kwh": float(max(execution["soc_start_kwh"].max(), execution["soc_end_kwh"].max())),
        "actual_soc_boundary_violation_count": int(
            ((execution["soc_end_kwh"] < SOC_MIN_KWH - TOL) | (execution["soc_end_kwh"] > SOC_MAX_KWH + TOL)).sum()
        ),
        "actual_power_violation_count": int(
            ((execution["charge_kwh"] > ENERGY_MAX_KWH + TOL) | (execution["discharge_kwh"] > ENERGY_MAX_KWH + TOL)).sum()
        ),
        "actual_simultaneous_charge_discharge_count": int(
            ((execution["charge_kwh"] > TOL) & (execution["discharge_kwh"] > TOL)).sum()
        ),
        "actual_emergency_and_charge_count": int(
            ((execution["emergency_purchase_kwh"] > TOL) & (execution["charge_kwh"] > TOL)).sum()
        ),
        "scenario_simultaneous_charge_discharge_count": int(
            ((scenario["charge_scenario_kwh"] > TOL) & (scenario["discharge_scenario_kwh"] > TOL)).sum()
        ),
        "scenario_emergency_and_charge_count": int(
            ((scenario["emergency_scenario_kwh"] > TOL) & (scenario["charge_scenario_kwh"] > TOL)).sum()
        ),
        "negative_spill_count": int((execution["spill_kwh"] < -TOL).sum()),
        "plan_lock_max_error_kwh": actual_plan_lock_error,
        "cross_day_soc_max_error_kwh": float(max(cross_day_errors, default=0.0)),
        "actual_cost_recalculation_max_error_yuan": float(
            (execution["actual_cost_yuan"] - execution["plan_cost_yuan"] - execution["emergency_cost_yuan"]).abs().max()
        ),
        "information_date_violation_count": int((info.get("max_source_date") >= info["decision_date"]).fillna(False).sum())
        + int((info.get("scenario_dates_strictly_before_decision") == False).sum()),
        "protected_file_count": len(before),
        "protected_hash_change_count": len(changed),
        "protected_hash_changes": changed,
        "result2_created_inside_q2_v2": bool(any(OUT.parent.rglob("result2.xlsx"))),
        "paired_dp_minus_greedy_cost_yuan": float(paired["dp_minus_greedy_cost_yuan"].sum()),
        "paired_dp_minus_greedy_emergency_kwh": float(paired["dp_minus_greedy_emergency_kwh"].sum()),
        "paired_net_action_difference_kwh": float(paired["sum_abs_net_action_difference_kwh"].sum()),
        "grid50_minus_grid100_cost_yuan": float(grid["cost_50_minus_100_yuan"].sum()),
        "grid50_minus_grid100_emergency_kwh": float(grid["emergency_50_minus_100_kwh"].sum()),
        "grid_net_action_difference_kwh": float(grid["sum_abs_net_action_difference_kwh"].sum()),
        "grid_max_soc_difference_kwh": float(grid["max_abs_soc_difference_kwh"].max()),
    }
    critical_keys = [
        "actual_soc_boundary_violation_count", "actual_power_violation_count",
        "actual_simultaneous_charge_discharge_count", "actual_emergency_and_charge_count",
        "scenario_simultaneous_charge_discharge_count", "scenario_emergency_and_charge_count",
        "negative_spill_count", "information_date_violation_count", "protected_hash_change_count",
    ]
    residual_keys = [
        "max_actual_balance_residual_kwh", "max_scenario_balance_residual_kwh",
        "max_scenario_soc_transition_residual_kwh", "plan_lock_max_error_kwh",
        "cross_day_soc_max_error_kwh", "actual_cost_recalculation_max_error_yuan",
    ]
    result["validation_passed"] = (
        all(result[k] == 0 for k in critical_keys)
        and all(result[k] <= TOL for k in residual_keys)
        and not result["result2_created_inside_q2_v2"]
    )
    (OUT / "q2_v2_stage2_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["Q2-v2 Stage 2 independent validation", "=" * 42]
    lines.extend(f"{key}: {value}" for key, value in result.items())
    (OUT / "q2_v2_stage2_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
