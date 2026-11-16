"""Assemble Stage-3 comparisons, executor ablation and freeze validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, SOC_MAX_KWH, SOC_MIN_KWH, TOL
from .data import load_inputs
from .execution import execute_day


OUT = ROOT / "outputs" / "q2_v2" / "stage3"
SELECTED = OUT / "selected" / "selected_M14_v04_dp50"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run() -> dict:
    data = load_inputs()
    tables = []
    for stage, filename in [
        ("initial_screen", "q2_v2_january_screen_candidates.csv"),
        ("M_screen", "q2_v2_january_m_candidates.csv"),
        ("solver_refine", "q2_v2_january_refined_candidates.csv"),
        ("high_precision_segment", "q2_v2_january_high_precision_candidates.csv"),
    ]:
        frame = pd.read_csv(OUT / filename)
        frame.insert(0, "comparison_stage", stage)
        tables.append(frame)
    selected_summary = json.loads((OUT / "q2_v2_selected_development_summary.json").read_text(encoding="utf-8"))
    selected_frame = pd.DataFrame([{"comparison_stage": "selected_full_january", **selected_summary}])
    tables.append(selected_frame)
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined.to_csv(OUT / "q2_v2_january_candidate_comparison.csv", index=False)

    warmup_rows = []
    for policy_id, folder in [
        ("P2_M14_v0_dp100_refined", OUT / "refine" / "P2_M14_v0_dp100_refined"),
        ("P3_M14_v04_dp100_refined", OUT / "refine" / "P3_M14_v04_dp100_refined"),
        ("P4_M14_v08_dp100_refined", OUT / "refine" / "P4_M14_v08_dp100_refined"),
        ("selected_M14_v04_dp50", SELECTED),
    ]:
        all_days = pd.read_csv(folder / "daily_results.csv")
        warmup_rows.append({
            "policy_id": policy_id, "period": "2025-01-01..2025-01-31",
            "plan_cost_yuan": all_days["plan_cost_yuan"].sum(),
            "emergency_cost_yuan": all_days["emergency_cost_yuan"].sum(),
            "actual_total_cost_yuan": all_days["actual_total_cost_yuan"].sum(),
            "terminal_soc_kwh": all_days.iloc[-1]["soc_end_kwh"],
        })
    pd.DataFrame(warmup_rows).to_csv(OUT / "q2_v2_january_full_replay_cost_check.csv", index=False)

    selected_slots = pd.read_csv(SELECTED / "slot_trajectories.csv")
    selected_plans = pd.read_csv(SELECTED / "day_ahead_plans.csv")
    selected_daily = pd.read_csv(SELECTED / "daily_results.csv")
    selected_solver = pd.read_csv(SELECTED / "solver_results.csv")
    selected_info = pd.read_csv(SELECTED / "information_boundary.csv")

    ablation_rows = []
    for date in pd.date_range("2025-01-22", "2025-01-31").strftime("%Y-%m-%d"):
        d = data.day_index(date)
        dp = selected_slots[selected_slots["date"] == date].sort_values("slot")
        plan = selected_plans[selected_plans["date"] == date].sort_values("slot")["planned_purchase_kwh"].to_numpy()
        initial = float(dp.iloc[0]["soc_start_kwh"])
        greedy, gmeta = execute_day(date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
                                    data.price_yuan_per_kwh, initial, "greedy")
        ablation_rows.append({
            "date": date, "same_plan_same_initial_soc": True, "initial_soc_kwh": initial,
            "dp50_cost_yuan": float(dp["actual_cost_yuan"].sum()), "greedy_cost_yuan": gmeta["actual_total_cost_yuan"],
            "dp_minus_greedy_cost_yuan": float(dp["actual_cost_yuan"].sum()) - gmeta["actual_total_cost_yuan"],
            "dp50_emergency_kwh": float(dp["emergency_purchase_kwh"].sum()), "greedy_emergency_kwh": gmeta["emergency_purchase_kwh"],
            "dp_minus_greedy_emergency_kwh": float(dp["emergency_purchase_kwh"].sum()) - gmeta["emergency_purchase_kwh"],
            "dp50_end_soc_kwh": float(dp.iloc[-1]["soc_end_kwh"]), "greedy_end_soc_kwh": gmeta["soc_end_kwh"],
            "sum_abs_net_action_difference_kwh": float(np.abs(
                (dp["charge_kwh"]-dp["discharge_kwh"]).to_numpy()
                - (greedy["charge_kwh"]-greedy["discharge_kwh"]).to_numpy()).sum()),
        })
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(OUT / "q2_v2_january_executor_ablation.csv", index=False)

    dev_slots = selected_slots[selected_slots["is_development_evaluation"].astype(bool)].copy()
    dev_daily = selected_daily[selected_daily["is_development_evaluation"].astype(bool)].copy()
    plan_error = float(np.max(np.abs(
        selected_slots.merge(selected_plans[["date", "slot", "planned_purchase_kwh"]], on=["date", "slot"],
                             suffixes=("", "_locked"), validate="one_to_one")["planned_purchase_kwh"]
        - selected_slots.merge(selected_plans[["date", "slot", "planned_purchase_kwh"]], on=["date", "slot"],
                               suffixes=("", "_locked"), validate="one_to_one")["planned_purchase_kwh_locked"]
    )))
    cross = selected_daily.sort_values("date")
    cross_error = float(np.max(np.abs(cross["soc_start_kwh"].to_numpy()[1:] - cross["soc_end_kwh"].to_numpy()[:-1])))
    original_hashes = json.loads((ROOT / "outputs/q2_v2/audit/q2_v2_protected_hashes_after.json").read_text(encoding="utf-8"))
    current_hashes = {name: sha256(ROOT / name) for name in original_hashes}
    changed = [name for name in original_hashes if original_hashes[name] != current_hashes[name]]
    result = {
        "selected_policy": "selected_M14_v04_dp50", "development_period": "2025-01-22..2025-01-31",
        "development_days": 10, "selected_actual_cost_yuan": float(dev_daily["actual_total_cost_yuan"].sum()),
        "selected_plan_cost_yuan": float(dev_daily["plan_cost_yuan"].sum()),
        "selected_emergency_cost_yuan": float(dev_daily["emergency_cost_yuan"].sum()),
        "selected_emergency_kwh": float(dev_daily["emergency_purchase_kwh"].sum()),
        "selected_initial_soc_kwh": float(dev_daily.iloc[0]["soc_start_kwh"]),
        "selected_terminal_soc_kwh": float(dev_daily.iloc[-1]["soc_end_kwh"]),
        "max_balance_error_kwh": float(dev_slots["balance_residual_kwh"].abs().max()),
        "plan_lock_error_kwh": plan_error, "cross_day_soc_error_kwh": cross_error,
        "soc_violation_count": int(((dev_slots["soc_end_kwh"] < SOC_MIN_KWH-TOL) | (dev_slots["soc_end_kwh"] > SOC_MAX_KWH+TOL)).sum()),
        "power_violation_count": int(((dev_slots["charge_kwh"] > 5000/6+TOL) | (dev_slots["discharge_kwh"] > 5000/6+TOL)).sum()),
        "simultaneous_charge_discharge_count": int(((dev_slots["charge_kwh"] > TOL) & (dev_slots["discharge_kwh"] > TOL)).sum()),
        "emergency_and_charge_count": int(((dev_slots["emergency_purchase_kwh"] > TOL) & (dev_slots["charge_kwh"] > TOL)).sum()),
        "cost_recalculation_error_yuan": float(abs(dev_daily["actual_total_cost_yuan"].sum()-dev_daily["plan_cost_yuan"].sum()-dev_daily["emergency_cost_yuan"].sum())),
        "information_violation_count": int((~selected_info["all_prediction_sources_prior"].astype(bool)).sum() + (~selected_info["all_scenario_sources_prior"].astype(bool)).sum()),
        "development_optimal_certified_days": int(selected_solver[selected_solver["date"] >= "2025-01-22"]["optimal_solution_certified"].astype(bool).sum()),
        "development_max_mip_gap": float(selected_solver[selected_solver["date"] >= "2025-01-22"]["mip_gap"].max()),
        "protected_file_count": len(original_hashes), "protected_hash_change_count": len(changed),
        "protected_hash_changes": changed, "result2_inside_v2": bool(any((ROOT / "outputs/q2_v2").rglob("result2.xlsx"))),
        "ablation_dp_minus_greedy_cost_yuan": float(ablation["dp_minus_greedy_cost_yuan"].sum()),
        "ablation_dp_minus_greedy_emergency_kwh": float(ablation["dp_minus_greedy_emergency_kwh"].sum()),
    }
    critical = ["soc_violation_count", "power_violation_count", "simultaneous_charge_discharge_count",
                "emergency_and_charge_count", "information_violation_count", "protected_hash_change_count"]
    result["validation_passed"] = all(result[k] == 0 for k in critical) and max(
        result["max_balance_error_kwh"], result["plan_lock_error_kwh"], result["cross_day_soc_error_kwh"],
        result["cost_recalculation_error_yuan"]) <= TOL and not result["result2_inside_v2"]
    (OUT / "q2_v2_stage3_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "q2_v2_stage3_validation.txt").write_text("\n".join(["Q2-v2 Stage 3 validation", "="*32] + [f"{k}: {v}" for k,v in result.items()]) + "\n", encoding="utf-8")
    fixed = {
        "status": "frozen_after_january_development_only", "formal_evaluation_has_run": False,
        "load_forecast": "mean of same weekday/same slot in prior 35 days; fallback prior up-to-7 days",
        "pv_forecast": "same-slot mean of prior up-to-7 complete days", "joint_residual_scenario_cap": 14,
        "terminal_value_yuan_per_kwh": 0.4, "dp_soc_grid_kwh": 50.0,
        "actual_soc_is_continuous": True, "dp_primary_tie_tolerance_yuan": 1e-8,
        "tie_rule": "within primary tolerance minimize charge+discharge throughput; deterministic residual ordering",
        "solver": "HiGHS MILP", "solver_time_limit_seconds": 45.0, "solver_relative_gap": 0.0001,
        "solver_random_seed": 0, "solver_threads": 1, "cold_start": "Jan-01 zero plan and greedy actual execution",
        "initial_soc_2025_01_01_kwh": 6000.0, "cross_day_soc": True,
        "terminal_value_in_actual_bill": False, "result2_generation_authorized": False,
    }
    (OUT / "q2_v2_frozen_strategy_config.json").write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
