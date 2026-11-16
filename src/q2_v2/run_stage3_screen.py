"""Stage-3 January causal screen with predeclared, non-adaptive candidates."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, SOC_INITIAL_KWH, SOC_MAX_KWH, SOC_MIN_KWH, TOL
from .data import load_inputs
from .day_ahead_milp import solve_joint_scenario_milp
from .execution import build_path_value_functions, execute_day
from .forecast import build_causal_replay
from .scenarios import JointScenarios, build_joint_scenarios


OUT = ROOT / "outputs" / "q2_v2" / "stage3"
DEV_START = "2025-01-22"
DEV_END = "2025-01-31"
TIME_LIMIT = 8.0
REL_GAP = 0.002
RANDOM_SEED = 0


@dataclass(frozen=True)
class Policy:
    policy_id: str
    plan_kind: str
    scenario_cap: int
    terminal_value: float
    executor: str
    grid_step: float | None = None
    retention_soc: float | None = None


SCREEN_POLICIES = (
    Policy("P0_point_v0_greedy", "point", 1, 0.0, "greedy"),
    Policy("P1_M14_v0_greedy", "scenario", 14, 0.0, "greedy"),
    Policy("P2_M14_v0_dp100", "scenario", 14, 0.0, "dp", 100.0),
    Policy("P3_M14_v04_dp100", "scenario", 14, 0.4, "dp", 100.0),
    Policy("P4_M14_v08_dp100", "scenario", 14, 0.8, "dp", 100.0),
    Policy("P5_M14_v0_threshold3600", "scenario", 14, 0.0, "threshold", None, 3600.0),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def protected_current() -> dict[str, str]:
    original = json.loads((ROOT / "outputs/q2_v2/audit/q2_v2_protected_hashes_after.json").read_text(encoding="utf-8"))
    return {name: sha256(ROOT / name) for name in original}


def point_scenario(data, replay, day_index: int) -> JointScenarios:
    source = max(0, day_index - 1)
    return JointScenarios(
        source_indices=(source,),
        source_dates=(data.dates[source].date().isoformat(),),
        weights=np.ones(1),
        load_kwh=replay.load_forecast_kwh[day_index][None, :],
        pv_kwh=replay.pv_forecast_kwh[day_index][None, :],
    )


def run_policy(
    policy: Policy,
    data,
    replay,
    run_group: str = "screen",
    time_limit_seconds: float = TIME_LIMIT,
    relative_gap: float = REL_GAP,
) -> dict:
    policy_dir = OUT / run_group / policy.policy_id
    policy_dir.mkdir(parents=True, exist_ok=True)
    print(f"START {policy.policy_id}", flush=True)
    soc = SOC_INITIAL_KWH
    slot_frames: list[pd.DataFrame] = []
    plan_frames: list[pd.DataFrame] = []
    daily_rows: list[dict] = []
    solver_rows: list[dict] = []
    info_rows: list[dict] = []
    started_policy = time.perf_counter()
    for d in range(31):
        date = data.dates[d].date().isoformat()
        soc_start = soc
        if d == 0:
            plan = np.zeros(144)
            scenarios = None
            solver_meta = {
                "solver_status": "not_applicable", "termination_condition": "cold_start_zero_plan",
                "optimal_solution_certified": False, "best_bound_yuan": np.nan,
                "incumbent_objective_yuan": np.nan, "absolute_gap_yuan": np.nan,
                "mip_gap": np.nan, "solve_time_seconds": 0.0, "scenario_count": 0,
            }
            actual_executor = "greedy"
            path_value = None
            scenario_dates: tuple[str, ...] = ()
        else:
            scenarios = point_scenario(data, replay, d) if policy.plan_kind == "point" else build_joint_scenarios(data, replay, d, policy.scenario_cap)
            result = solve_joint_scenario_milp(
                scenarios, data.price_yuan_per_kwh, soc_start, policy.terminal_value,
                time_limit_seconds=time_limit_seconds, relative_gap=relative_gap, random_seed=RANDOM_SEED,
            )
            plan = result.plan["planned_purchase_kwh"].to_numpy(dtype=float)
            solver_meta = result.metadata
            scenario_dates = scenarios.source_dates
            if policy.executor == "dp":
                path_value = build_path_value_functions(
                    scenarios, plan, data.price_yuan_per_kwh, float(policy.grid_step), policy.terminal_value
                )
            else:
                path_value = None
            actual_executor = policy.executor
        frame, meta = execute_day(
            date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
            data.price_yuan_per_kwh, soc_start, actual_executor, path_value,
            retention_soc_kwh=policy.retention_soc,
        )
        frame.insert(0, "policy_id", policy.policy_id)
        frame["is_development_evaluation"] = DEV_START <= date <= DEV_END
        slot_frames.append(frame)
        plan_frames.append(pd.DataFrame({
            "policy_id": policy.policy_id, "date": date, "slot": np.arange(144),
            "time_label": data.time_labels, "planned_purchase_kwh": plan,
            "load_forecast_kwh": replay.load_forecast_kwh[d],
            "pv_forecast_kwh": replay.pv_forecast_kwh[d],
        }))
        daily_rows.append({"policy_id": policy.policy_id, "date": date,
                           "is_development_evaluation": DEV_START <= date <= DEV_END,
                           **meta, "actual_executor": actual_executor,
                           "scenario_count_effective": 0 if scenarios is None else scenarios.count})
        solver_rows.append({"policy_id": policy.policy_id, "date": date, "soc_start_kwh": soc_start,
                            "terminal_value_yuan_per_kwh": policy.terminal_value, **solver_meta})
        source_audit = replay.source_audit.iloc[d]
        info_rows.append({
            "policy_id": policy.policy_id, "decision_date": date,
            "load_history_dates": source_audit["load_history_dates"],
            "pv_history_dates": source_audit["pv_history_dates"],
            "scenario_dates": "|".join(scenario_dates),
            "scenario_count_effective": len(scenario_dates),
            "all_prediction_sources_prior": bool(source_audit["all_source_dates_strictly_before_decision"]),
            "all_scenario_sources_prior": all(pd.Timestamp(x) < pd.Timestamp(date) for x in scenario_dates),
        })
        soc = float(meta["soc_end_kwh"])
        print(f"{policy.policy_id} {date} SOC={soc:.3f} cost={meta['actual_total_cost_yuan']:.3f}", flush=True)

    slots = pd.concat(slot_frames, ignore_index=True)
    plans = pd.concat(plan_frames, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    solvers = pd.DataFrame(solver_rows)
    info = pd.DataFrame(info_rows)
    slots.to_csv(policy_dir / "slot_trajectories.csv", index=False)
    plans.to_csv(policy_dir / "day_ahead_plans.csv", index=False)
    daily.to_csv(policy_dir / "daily_results.csv", index=False)
    solvers.to_csv(policy_dir / "solver_results.csv", index=False)
    info.to_csv(policy_dir / "information_boundary.csv", index=False)
    dev = daily[daily["is_development_evaluation"]]
    dev_slots = slots[slots["is_development_evaluation"]]
    summary = {
        **asdict(policy),
        "development_start": DEV_START, "development_end": DEV_END,
        "development_days": int(len(dev)), "cold_start_rule": "Jan-01 zero plan plus greedy actual execution",
        "warmup_soc_not_reset": True,
        "plan_cost_yuan": float(dev["plan_cost_yuan"].sum()),
        "emergency_cost_yuan": float(dev["emergency_cost_yuan"].sum()),
        "actual_total_cost_yuan": float(dev["actual_total_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(dev["emergency_purchase_kwh"].sum()),
        "charge_kwh": float(dev["charge_kwh"].sum()),
        "discharge_kwh": float(dev["discharge_kwh"].sum()),
        "conversion_loss_kwh": float(dev["conversion_loss_kwh"].sum()),
        "spill_kwh": float(dev["spill_kwh"].sum()),
        "development_initial_soc_kwh": float(dev.iloc[0]["soc_start_kwh"]),
        "development_terminal_soc_kwh": float(dev.iloc[-1]["soc_end_kwh"]),
        "actual_soc_min_kwh": float(min(dev_slots["soc_start_kwh"].min(), dev_slots["soc_end_kwh"].min())),
        "actual_soc_max_kwh": float(max(dev_slots["soc_start_kwh"].max(), dev_slots["soc_end_kwh"].max())),
        "total_runtime_seconds": time.perf_counter() - started_policy,
        "solver_time_seconds": float(solvers["solve_time_seconds"].sum()),
        "optimal_certified_days": int(solvers["optimal_solution_certified"].fillna(False).astype(bool).sum()),
        "time_limit_feasible_days": int((solvers["termination_condition"] == "maxTimeLimit").sum()),
        "max_relative_gap": float(solvers["mip_gap"].max()),
        "max_absolute_gap_yuan": float(solvers["absolute_gap_yuan"].max()),
        "min_effective_scenarios_dev": int(dev["scenario_count_effective"].min()),
        "max_effective_scenarios_dev": int(dev["scenario_count_effective"].max()),
        "max_balance_residual_kwh": float(dev_slots["balance_residual_kwh"].abs().max()),
        "solver_time_limit_seconds": time_limit_seconds,
        "solver_requested_relative_gap": relative_gap,
        "soc_violation_count": int(((dev_slots["soc_end_kwh"] < SOC_MIN_KWH-TOL) | (dev_slots["soc_end_kwh"] > SOC_MAX_KWH+TOL)).sum()),
        "power_violation_count": int(((dev_slots["charge_kwh"] > 5000/6+TOL) | (dev_slots["discharge_kwh"] > 5000/6+TOL)).sum()),
        "simultaneous_charge_discharge_count": int(((dev_slots["charge_kwh"] > TOL) & (dev_slots["discharge_kwh"] > TOL)).sum()),
        "emergency_and_charge_count": int(((dev_slots["emergency_purchase_kwh"] > TOL) & (dev_slots["charge_kwh"] > TOL)).sum()),
        "information_violation_count": int((~info["all_prediction_sources_prior"]).sum() + (~info["all_scenario_sources_prior"]).sum()),
    }
    (policy_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE {policy.policy_id} dev_cost={summary['actual_total_cost_yuan']:.3f}", flush=True)
    return summary


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected_current()
    data = load_inputs()
    replay = build_causal_replay(data)
    summaries = [run_policy(policy, data, replay) for policy in SCREEN_POLICIES]
    pd.DataFrame(summaries).to_csv(OUT / "q2_v2_january_screen_candidates.csv", index=False)
    config = {
        "development_period": [DEV_START, DEV_END], "warmup_period": ["2025-01-01", "2025-01-21"],
        "policies": [asdict(x) for x in SCREEN_POLICIES], "solver_time_limit_seconds": TIME_LIMIT,
        "solver_relative_gap": REL_GAP, "solver_random_seed": RANDOM_SEED, "solver_threads": 1,
        "future_formal_period_used_for_selection": False, "result2_generated": False,
    }
    (OUT / "q2_v2_stage3_screen_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    after = protected_current()
    (OUT / "protected_hashes_stage3_before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "protected_hashes_stage3_after_screen.json").write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SCREEN COMPLETE", flush=True)


if __name__ == "__main__":
    run()
