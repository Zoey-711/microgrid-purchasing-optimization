"""Checkpointed Stage-4 runner for one frozen closed-loop policy."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, SOC_INITIAL_KWH
from .data import load_inputs
from .day_ahead_milp import solve_joint_scenario_milp
from .execution import build_path_value_functions, execute_day
from .forecast import build_causal_replay
from .scenarios import build_joint_scenarios


FROZEN_PATH = ROOT / "outputs/q2_v2/stage3/q2_v2_frozen_strategy_config.json"
OUT = ROOT / "outputs/q2_v2/stage4"


def _load_frozen() -> dict:
    cfg = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    required = {
        "joint_residual_scenario_cap": 14,
        "terminal_value_yuan_per_kwh": 0.4,
        "dp_soc_grid_kwh": 50.0,
        "solver_time_limit_seconds": 45.0,
        "solver_relative_gap": 0.0001,
        "solver_random_seed": 0,
        "solver_threads": 1,
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise RuntimeError(f"Frozen configuration mismatch: {key}={cfg.get(key)!r}")
    return cfg


def _write_day(folder: Path, name: str, date: str, frame: pd.DataFrame) -> None:
    target = folder / name
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / f"{date}.csv", index=False)


def _resume_state(folder: Path, default_soc: float, default_start: int) -> tuple[int, float]:
    state_path = folder / "state.json"
    if not state_path.exists():
        return default_start, default_soc
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") == "complete":
        return int(state["next_day_index"]), float(state["soc_kwh"])
    return int(state["next_day_index"]), float(state["soc_kwh"])


def run(policy: str) -> None:
    if policy not in {"main_dp50", "closed_loop_greedy"}:
        raise ValueError(policy)
    cfg = _load_frozen()
    data = load_inputs()
    replay = build_causal_replay(data)
    folder = OUT / policy
    folder.mkdir(parents=True, exist_ok=True)
    if policy == "main_dp50":
        jan = pd.read_csv(ROOT / "outputs/q2_v2/stage3/selected/selected_M14_v04_dp50/daily_results.csv")
        default_soc = float(jan.iloc[-1]["soc_end_kwh"])
        default_start = data.day_index("2025-02-01")
    else:
        default_soc = SOC_INITIAL_KWH
        default_start = 0
    start_index, soc = _resume_state(folder, default_soc, default_start)
    if start_index >= len(data.dates):
        print(f"{policy} already complete", flush=True)
        return
    config_record = {
        "policy": policy, "frozen_config_path": str(FROZEN_PATH.relative_to(ROOT)),
        "formal_start": "2025-02-01", "formal_end": "2025-12-31",
        "default_start_index": default_start, "default_initial_soc_kwh": default_soc,
        "closed_loop": True, "paired_ablation_generated": policy == "main_dp50",
        "no_solution_fallback": None, "on_no_solution": "write failure checkpoint and stop",
    }
    (folder / "run_config.json").write_text(json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8")
    for d in range(start_index, len(data.dates)):
        date = data.dates[d].date().isoformat()
        soc_start = soc
        try:
            if policy == "closed_loop_greedy" and d == 0:
                plan = np.zeros(144)
                scenarios = None
                solver_meta = {
                    "solver_status": "not_applicable", "termination_condition": "cold_start_zero_plan",
                    "optimal_solution_certified": False, "solution_class": "cold_start",
                    "best_bound_yuan": np.nan, "incumbent_objective_yuan": np.nan,
                    "absolute_gap_yuan": np.nan, "mip_gap": np.nan, "solve_time_seconds": 0.0,
                    "scenario_count": 0,
                }
            else:
                scenarios = build_joint_scenarios(data, replay, d, int(cfg["joint_residual_scenario_cap"]))
                solved = solve_joint_scenario_milp(
                    scenarios, data.price_yuan_per_kwh, soc_start,
                    float(cfg["terminal_value_yuan_per_kwh"]),
                    time_limit_seconds=float(cfg["solver_time_limit_seconds"]),
                    relative_gap=float(cfg["solver_relative_gap"]),
                    random_seed=int(cfg["solver_random_seed"]),
                )
                plan = solved.plan["planned_purchase_kwh"].to_numpy(dtype=float)
                solver_meta = solved.metadata
            if policy == "main_dp50":
                value = build_path_value_functions(
                    scenarios, plan, data.price_yuan_per_kwh,
                    float(cfg["dp_soc_grid_kwh"]), float(cfg["terminal_value_yuan_per_kwh"]),
                )
                executor = "dp"
            else:
                value = None
                executor = "greedy"
            frame, meta = execute_day(
                date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
                data.price_yuan_per_kwh, soc_start, executor, value,
            )
            frame.insert(0, "policy_id", policy)
            _write_day(folder, "slots", date, frame)
            plan_frame = pd.DataFrame({
                "policy_id": policy, "date": date, "slot": np.arange(144), "time_label": data.time_labels,
                "planned_purchase_kwh": plan, "load_forecast_kwh": replay.load_forecast_kwh[d],
                "pv_forecast_kwh": replay.pv_forecast_kwh[d],
            })
            _write_day(folder, "plans", date, plan_frame)
            _write_day(folder, "daily", date, pd.DataFrame([{**meta, "policy_id": policy,
                       "scenario_count_effective": 0 if scenarios is None else scenarios.count}]))
            _write_day(folder, "solver", date, pd.DataFrame([{**solver_meta, "policy_id": policy,
                       "date": date, "soc_start_kwh": soc_start}]))
            source = replay.source_audit.iloc[d]
            scenario_dates = () if scenarios is None else scenarios.source_dates
            info = pd.DataFrame([{
                "policy_id": policy, "decision_date": date,
                "load_history_dates": source["load_history_dates"], "pv_history_dates": source["pv_history_dates"],
                "scenario_dates": "|".join(scenario_dates), "scenario_count_effective": len(scenario_dates),
                "all_prediction_sources_prior": bool(source["all_source_dates_strictly_before_decision"]),
                "all_scenario_sources_prior": all(pd.Timestamp(x) < pd.Timestamp(date) for x in scenario_dates),
            }])
            _write_day(folder, "information", date, info)
            if policy == "main_dp50":
                paired, paired_meta = execute_day(
                    date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
                    data.price_yuan_per_kwh, soc_start, "greedy", None,
                )
                paired.insert(0, "policy_id", "paired_greedy_same_plan_soc")
                _write_day(folder, "paired_slots", date, paired)
                _write_day(folder, "paired_daily", date, pd.DataFrame([{**paired_meta,
                           "policy_id": "paired_greedy_same_plan_soc", "reference_policy": policy}]))
            soc = float(meta["soc_end_kwh"])
            state = {"status": "running", "last_completed_date": date, "next_day_index": d + 1,
                     "soc_kwh": soc, "updated_at_epoch": time.time()}
            (folder / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{policy} {date} soc={soc:.3f} cost={meta['actual_total_cost_yuan']:.3f} "
                  f"term={solver_meta['termination_condition']} gap={solver_meta.get('mip_gap')}", flush=True)
        except Exception as exc:
            failure = {"status": "failed", "date": date, "day_index": d, "soc_start_kwh": soc_start,
                       "error": repr(exc), "traceback": traceback.format_exc(), "no_fallback_used": True}
            (folder / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
    state = {"status": "complete", "last_completed_date": data.dates[-1].date().isoformat(),
             "next_day_index": len(data.dates), "soc_kwh": soc, "updated_at_epoch": time.time()}
    (folder / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=["main_dp50", "closed_loop_greedy"])
    run(parser.parse_args().policy)
