"""Run the authorized three-day Q2-v2 Stage-2 mechanism closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import (
    ENERGY_MAX_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    SOC_INITIAL_KWH,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    STAGE2_DP_GRIDS_KWH,
    STAGE2_END_DATE,
    STAGE2_SCENARIO_CAP,
    STAGE2_START_DATE,
    TOL,
    ROOT,
)
from .data import load_inputs
from .day_ahead_milp import MilpResult, solve_joint_scenario_milp
from .execution import PathValueResult, build_path_value_functions, execute_day
from .forecast import build_causal_replay
from .scenarios import JointScenarios, build_joint_scenarios


OUT = ROOT / "outputs" / "q2_v2" / "stage2"
AUDIT = ROOT / "outputs" / "q2_v2" / "audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _protected_paths() -> list[Path]:
    stored = json.loads((AUDIT / "q2_v2_protected_hashes_after.json").read_text(encoding="utf-8"))
    return [ROOT / name for name in stored]


def _save_milp(
    result: MilpResult,
    run_id: str,
    date: str,
    initial_soc: float,
    scenarios: JointScenarios,
    forecast_load: np.ndarray,
    forecast_pv: np.ndarray,
    plan_rows: list[pd.DataFrame],
    scenario_rows: list[pd.DataFrame],
    solver_rows: list[dict],
) -> np.ndarray:
    plan = result.plan.copy()
    plan.insert(0, "run_id", run_id)
    plan.insert(1, "date", date)
    plan["load_point_forecast_kwh"] = forecast_load
    plan["pv_point_forecast_kwh"] = forecast_pv
    plan["net_load_point_forecast_kwh"] = forecast_load - forecast_pv
    plan["initial_soc_kwh"] = initial_soc
    plan["scenario_source_dates"] = "|".join(scenarios.source_dates)
    plan_rows.append(plan)
    dispatch = result.scenario_dispatch.copy()
    dispatch.insert(0, "run_id", run_id)
    dispatch.insert(1, "date", date)
    scenario_rows.append(dispatch)
    solver_rows.append({"run_id": run_id, "date": date, "initial_soc_kwh": initial_soc, **result.metadata})
    return result.plan["planned_purchase_kwh"].to_numpy(dtype=float)


def _tag_execution(
    frame: pd.DataFrame,
    metadata: dict,
    strategy: str,
    comparison_scope: str,
    run_id: str,
    grid_step: float | None,
    terminal_value: float,
    scenarios: JointScenarios,
) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    frame.insert(0, "strategy", strategy)
    frame.insert(1, "comparison_scope", comparison_scope)
    frame.insert(2, "plan_run_id", run_id)
    frame["dp_grid_step_kwh"] = grid_step
    frame["terminal_value_yuan_per_kwh"] = terminal_value
    frame["scenario_count_effective"] = scenarios.count
    frame["scenario_source_dates"] = "|".join(scenarios.source_dates)
    meta = {
        "strategy": strategy,
        "comparison_scope": comparison_scope,
        "plan_run_id": run_id,
        "dp_grid_step_kwh": grid_step,
        "terminal_value_yuan_per_kwh": terminal_value,
        "scenario_count_effective": scenarios.count,
        **metadata,
    }
    return frame, meta


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = _protected_paths()
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    data = load_inputs()
    replay = build_causal_replay(data)
    start = data.day_index(STAGE2_START_DATE)
    end = data.day_index(STAGE2_END_DATE)
    day_indices = list(range(start, end + 1))
    v_test = float(np.median(data.price_yuan_per_kwh) * ETA_DISCHARGE)

    forecast_long = []
    for d in range(0, end + 1):
        for t in range(len(data.time_labels)):
            forecast_long.append(
                {
                    "date": data.dates[d].date().isoformat(),
                    "slot": t,
                    "time_label": data.time_labels[t],
                    "load_forecast_kwh": replay.load_forecast_kwh[d, t],
                    "pv_forecast_kwh": replay.pv_forecast_kwh[d, t],
                    "load_actual_kwh": data.load_kwh[d, t],
                    "pv_actual_kwh": data.pv_kwh[d, t],
                    "load_residual_kwh": replay.load_residual_kwh[d, t],
                    "pv_residual_kwh": replay.pv_residual_kwh[d, t],
                }
            )

    plan_rows: list[pd.DataFrame] = []
    scenario_rows: list[pd.DataFrame] = []
    solver_rows: list[dict] = []
    execution_rows: list[pd.DataFrame] = []
    daily_rows: list[dict] = []
    paired_rows: list[dict] = []
    grid_rows: list[dict] = []
    value_manifest: list[dict] = []
    dp_reference: dict[int, tuple[np.ndarray, float, JointScenarios, PathValueResult]] = {}

    # Main v=0 + DP100 closed loop.
    soc_dp = SOC_INITIAL_KWH
    for d in day_indices:
        date = data.dates[d].date().isoformat()
        scenarios = build_joint_scenarios(data, replay, d, STAGE2_SCENARIO_CAP)
        run_id = f"v0_dp_plan_{date}"
        result = solve_joint_scenario_milp(scenarios, data.price_yuan_per_kwh, soc_dp, 0.0)
        plan = _save_milp(
            result, run_id, date, soc_dp, scenarios,
            replay.load_forecast_kwh[d], replay.pv_forecast_kwh[d],
            plan_rows, scenario_rows, solver_rows,
        )
        value100 = build_path_value_functions(
            scenarios, plan, data.price_yuan_per_kwh, STAGE2_DP_GRIDS_KWH[0], 0.0
        )
        np.savez_compressed(
            OUT / f"path_values_{run_id}_grid100.npz",
            soc_grid_kwh=value100.soc_grid_kwh,
            path_values_yuan=value100.path_values_yuan,
            mean_value_yuan=value100.mean_value_yuan,
        )
        value_manifest.append(
            {"run_id": run_id, "date": date, "grid_step_kwh": 100.0, "runtime_seconds": value100.runtime_seconds, "file": f"path_values_{run_id}_grid100.npz"}
        )
        frame, meta = execute_day(
            date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
            data.price_yuan_per_kwh, soc_dp, "dp", value100,
        )
        frame, meta = _tag_execution(frame, meta, "scenario_v0_dp100", "independent_cross_day", run_id, 100.0, 0.0, scenarios)
        execution_rows.append(frame)
        daily_rows.append(meta)
        dp_reference[d] = (plan, soc_dp, scenarios, value100)
        soc_dp = float(meta["soc_end_kwh"])

    # v=0 + greedy independent cross-day loop; reuse day 1 plan because initial SOC is identical.
    soc_greedy = SOC_INITIAL_KWH
    for pos, d in enumerate(day_indices):
        date = data.dates[d].date().isoformat()
        scenarios = build_joint_scenarios(data, replay, d, STAGE2_SCENARIO_CAP)
        if pos == 0:
            plan, _, _, _ = dp_reference[d]
            run_id = f"v0_dp_plan_{date}"
        else:
            run_id = f"v0_greedy_plan_{date}"
            result = solve_joint_scenario_milp(scenarios, data.price_yuan_per_kwh, soc_greedy, 0.0)
            plan = _save_milp(
                result, run_id, date, soc_greedy, scenarios,
                replay.load_forecast_kwh[d], replay.pv_forecast_kwh[d],
                plan_rows, scenario_rows, solver_rows,
            )
        frame, meta = execute_day(
            date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
            data.price_yuan_per_kwh, soc_greedy, "greedy", None,
        )
        frame, meta = _tag_execution(frame, meta, "scenario_v0_greedy", "independent_cross_day", run_id, None, 0.0, scenarios)
        execution_rows.append(frame)
        daily_rows.append(meta)
        soc_greedy = float(meta["soc_end_kwh"])

    # Paired comparisons: same plan and same initial SOC for each day.
    for d in day_indices:
        date = data.dates[d].date().isoformat()
        plan, paired_soc, scenarios, value100 = dp_reference[d]
        greedy_frame, greedy_meta = execute_day(
            date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
            data.price_yuan_per_kwh, paired_soc, "greedy", None,
        )
        dp_frame, dp_meta = execute_day(
            date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
            data.price_yuan_per_kwh, paired_soc, "dp", value100,
        )
        paired_rows.append(
            {
                "date": date,
                "same_plan_run_id": f"v0_dp_plan_{date}",
                "same_initial_soc_kwh": paired_soc,
                "greedy_actual_cost_yuan": greedy_meta["actual_total_cost_yuan"],
                "dp100_actual_cost_yuan": dp_meta["actual_total_cost_yuan"],
                "dp_minus_greedy_cost_yuan": dp_meta["actual_total_cost_yuan"] - greedy_meta["actual_total_cost_yuan"],
                "greedy_emergency_kwh": greedy_meta["emergency_purchase_kwh"],
                "dp100_emergency_kwh": dp_meta["emergency_purchase_kwh"],
                "dp_minus_greedy_emergency_kwh": dp_meta["emergency_purchase_kwh"] - greedy_meta["emergency_purchase_kwh"],
                "greedy_end_soc_kwh": greedy_meta["soc_end_kwh"],
                "dp100_end_soc_kwh": dp_meta["soc_end_kwh"],
                "sum_abs_net_action_difference_kwh": float(np.abs((dp_frame["charge_kwh"] - dp_frame["discharge_kwh"]) - (greedy_frame["charge_kwh"] - greedy_frame["discharge_kwh"])).sum()),
            }
        )

        value50 = build_path_value_functions(scenarios, plan, data.price_yuan_per_kwh, 50.0, 0.0)
        frame50, meta50 = execute_day(
            date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
            data.price_yuan_per_kwh, paired_soc, "dp", value50,
        )
        frame50, meta50 = _tag_execution(frame50, meta50, "scenario_v0_dp50", "paired_grid_precision", f"v0_dp_plan_{date}", 50.0, 0.0, scenarios)
        execution_rows.append(frame50)
        value_manifest.append(
            {"run_id": f"v0_dp_plan_{date}", "date": date, "grid_step_kwh": 50.0, "runtime_seconds": value50.runtime_seconds, "file": "not_saved_full_paths_precision_only"}
        )
        grid_rows.append(
            {
                "date": date,
                "same_plan_run_id": f"v0_dp_plan_{date}",
                "same_initial_soc_kwh": paired_soc,
                "dp100_actual_cost_yuan": dp_meta["actual_total_cost_yuan"],
                "dp50_actual_cost_yuan": meta50["actual_total_cost_yuan"],
                "cost_50_minus_100_yuan": meta50["actual_total_cost_yuan"] - dp_meta["actual_total_cost_yuan"],
                "emergency_50_minus_100_kwh": meta50["emergency_purchase_kwh"] - dp_meta["emergency_purchase_kwh"],
                "sum_abs_net_action_difference_kwh": float(np.abs((frame50["charge_kwh"] - frame50["discharge_kwh"]) - (dp_frame["charge_kwh"] - dp_frame["discharge_kwh"])).sum()),
                "max_abs_soc_difference_kwh": float(np.max(np.abs(frame50["soc_end_kwh"] - dp_frame["soc_end_kwh"]))),
                "path_value_runtime_100_seconds": value100.runtime_seconds,
                "path_value_runtime_50_seconds": value50.runtime_seconds,
                "execution_runtime_100_seconds": dp_meta["runtime_seconds"],
                "execution_runtime_50_seconds": meta50["runtime_seconds"],
            }
        )

    # One-day nonzero-v functionality check; this is not parameter selection.
    d = day_indices[0]
    date = data.dates[d].date().isoformat()
    scenarios = build_joint_scenarios(data, replay, d, STAGE2_SCENARIO_CAP)
    run_id = f"vtest_dp_plan_{date}"
    v_result = solve_joint_scenario_milp(scenarios, data.price_yuan_per_kwh, SOC_INITIAL_KWH, v_test)
    v_plan = _save_milp(
        v_result, run_id, date, SOC_INITIAL_KWH, scenarios,
        replay.load_forecast_kwh[d], replay.pv_forecast_kwh[d],
        plan_rows, scenario_rows, solver_rows,
    )
    v_value = build_path_value_functions(scenarios, v_plan, data.price_yuan_per_kwh, 100.0, v_test)
    v_frame, v_meta = execute_day(
        date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], v_plan,
        data.price_yuan_per_kwh, SOC_INITIAL_KWH, "dp", v_value,
    )
    v_frame, v_meta = _tag_execution(v_frame, v_meta, "scenario_vtest_dp100", "one_day_terminal_value_function_test", run_id, 100.0, v_test, scenarios)
    execution_rows.append(v_frame)
    daily_rows.append(v_meta)
    np.savez_compressed(
        OUT / f"path_values_{run_id}_grid100.npz",
        soc_grid_kwh=v_value.soc_grid_kwh,
        path_values_yuan=v_value.path_values_yuan,
        mean_value_yuan=v_value.mean_value_yuan,
    )
    value_manifest.append(
        {"run_id": run_id, "date": date, "grid_step_kwh": 100.0, "runtime_seconds": v_value.runtime_seconds, "file": f"path_values_{run_id}_grid100.npz"}
    )

    forecasts = pd.DataFrame(forecast_long)
    plans = pd.concat(plan_rows, ignore_index=True)
    scenario_dispatch = pd.concat(scenario_rows, ignore_index=True)
    executions = pd.concat(execution_rows, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    paired = pd.DataFrame(paired_rows)
    grid = pd.DataFrame(grid_rows)
    solvers = pd.DataFrame(solver_rows)
    info = replay.source_audit.iloc[: end + 1].copy()
    for d in day_indices:
        scenarios = build_joint_scenarios(data, replay, d, STAGE2_SCENARIO_CAP)
        date = data.dates[d].date().isoformat()
        mask = info["decision_date"] == date
        info.loc[mask, "scenario_count_requested"] = STAGE2_SCENARIO_CAP
        info.loc[mask, "scenario_count_effective"] = scenarios.count
        info.loc[mask, "residual_scenario_dates"] = "|".join(scenarios.source_dates)
        info.loc[mask, "max_scenario_date"] = max(scenarios.source_dates)
        info.loc[mask, "scenario_dates_strictly_before_decision"] = all(pd.Timestamp(x) < pd.Timestamp(date) for x in scenarios.source_dates)

    forecasts.to_csv(OUT / "q2_v2_causal_forecast_replay.csv", index=False)
    plans.to_csv(OUT / "q2_v2_day_ahead_plans.csv", index=False)
    scenario_dispatch.to_csv(OUT / "q2_v2_milp_scenario_dispatch.csv", index=False)
    executions.to_csv(OUT / "q2_v2_slot_trajectories.csv", index=False)
    daily.to_csv(OUT / "q2_v2_daily_results.csv", index=False)
    paired.to_csv(OUT / "q2_v2_paired_executor_comparison.csv", index=False)
    grid.to_csv(OUT / "q2_v2_grid_sensitivity.csv", index=False)
    solvers.to_csv(OUT / "q2_v2_solver_results.csv", index=False)
    info.to_csv(OUT / "q2_v2_information_boundary.csv", index=False)
    pd.DataFrame(value_manifest).to_csv(OUT / "q2_v2_path_value_manifest.csv", index=False)

    config = {
        "stage": 2,
        "test_dates": [STAGE2_START_DATE, STAGE2_END_DATE],
        "test_initial_soc_kwh": SOC_INITIAL_KWH,
        "test_initial_soc_scope": "independent mechanism test; not annual carried SOC",
        "scenario_cap_fixed": STAGE2_SCENARIO_CAP,
        "scenario_padding_or_duplication": False,
        "dp_grid_steps_kwh": list(STAGE2_DP_GRIDS_KWH),
        "terminal_value_v0": 0.0,
        "terminal_value_test_yuan_per_kwh": v_test,
        "terminal_value_test_derivation": "0.9 times median Attachment-1 price; functionality only",
        "model_selection_performed": False,
        "parameter_search_performed": False,
        "full_year_run": False,
        "result2_generated": False,
    }
    (OUT / "q2_v2_stage2_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    (OUT / "protected_hashes_before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "protected_hashes_after.json").write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
