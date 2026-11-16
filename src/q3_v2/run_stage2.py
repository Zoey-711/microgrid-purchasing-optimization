"""Run the isolated three-day Q3 V2 causal closure; never writes result3.xlsx."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_q3 import add_q3_slot_datetimes, build_q3_forecast_version, load_q3_pv_forecast_records
from src.q2_v2.data import load_inputs
from src.q2_v2.execution import build_path_value_functions, greedy_action, select_dp_action

from .config import (
    EMERGENCY_MULTIPLIER, FIRST_MODIFIABLE_SLOT, ISSUE_HOURS, LOAD_BIAS_LOOKBACK_SLOTS_TEST,
    LOAD_HALF_LIFE_DAYS_TEST, LOAD_HISTORY_DAYS_TEST, NEXT_BOUNDARY_SLOT, OUT, ROOT,
    SCENARIO_CAP_TEST, SOC_GRID_TEST_KWH, SOC_INITIAL_TEST_KWH, TERMINAL_VALUE_TEST, TEST_DATES,
)
from .rolling_milp import VersionScenarios, solve_version_milp
from .settlement import manual_settlement_cases, settle_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def protected_manifest() -> dict[str, str]:
    fixed = [
        ROOT / "data/raw/C题.pdf", ROOT / "data/raw/附件1.xlsx", ROOT / "data/raw/附件2.xlsx",
        ROOT / "data/raw/附件3.xlsx", ROOT / "data/raw/result1.xlsx", ROOT / "data/raw/result2.xlsx",
        ROOT / "data/raw/result3.xlsx", ROOT / "outputs/tables/result1.xlsx", ROOT / "outputs/tables/result2.xlsx",
        ROOT / "docs/论文正文.md", ROOT / "docs/q1_freeze_record.md",
    ]
    groups = [ROOT / "docs/q2_v2", ROOT / "src/q2_v2"]
    # Old Q3 source/designs are historical evidence and remain read-only.
    fixed.extend(ROOT / "src" / name for name in (
        "forecast_q3.py", "model_q3_rolling.py", "dispatch_q3_realtime.py", "q3_settlement.py",
        "backtest_q3_january.py", "shadow_q3_full_year.py", "solve_q3.py",
    ))
    fixed.extend(ROOT / "docs" / name for name in (
        "q3_data_contract.md", "q3_model_design.md", "q3_information_boundary_audit.md",
        "q3_stage2_settlement_review.md", "q3_parameter_selection.md", "q3_full_year_analysis.md",
    ))
    paths = [p for p in fixed if p.exists()]
    for group in groups:
        if group.exists():
            paths.extend(p for p in group.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(ROOT)): _sha256(p) for p in sorted(set(paths))}


def _actual_long(data) -> pd.DataFrame:
    rows = []
    for d, date in enumerate(data.dates):
        for t in range(144):
            rows.append({"date": date, "idx": t, "time_label": data.time_labels[t],
                         "load_kwh": data.load_kwh[d, t], "pv_kwh": data.pv_kwh[d, t]})
    return add_q3_slot_datetimes(pd.DataFrame(rows))


def _forecast_cache(actual: pd.DataFrame, pv_records: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    cache = {}
    for day in pd.date_range("2025-01-02", TEST_DATES[-1], freq="D"):
        for issue in ISSUE_HOURS:
            frame = build_q3_forecast_version(
                actual, pv_records, day, issue,
                history_days=LOAD_HISTORY_DAYS_TEST,
                half_life_days=LOAD_HALF_LIFE_DAYS_TEST,
                bias_lookback_slots=LOAD_BIAS_LOOKBACK_SLOTS_TEST,
                pv_mapping_method="linear",
            )
            cache[(day.date().isoformat(), issue)] = frame
    return cache


def _joint_scenarios(data, cache, decision_date: str, issue: int) -> VersionScenarios:
    current = cache[(decision_date, issue)].sort_values("idx")
    start = FIRST_MODIFIABLE_SLOT[issue]
    source_dates = sorted({d for d, k in cache if k == issue and d < decision_date})[-SCENARIO_CAP_TEST:]
    if not source_dates:
        raise RuntimeError("No causal historical forecast residual is available")
    load_paths, pv_paths = [], []
    current_load = current["load_forecast_kwh"].to_numpy(float)
    current_pv = current["pv_forecast_kwh"].to_numpy(float)
    idx = current["idx"].to_numpy(int)
    for source in source_dates:
        hist = cache[(source, issue)].sort_values("idx")
        if not np.array_equal(hist["idx"].to_numpy(int), idx):
            raise AssertionError("Historical and current forecast suffixes are not aligned")
        di = data.day_index(source)
        load_residual = data.load_kwh[di, idx] - hist["load_forecast_kwh"].to_numpy(float)
        pv_residual = data.pv_kwh[di, idx] - hist["pv_forecast_kwh"].to_numpy(float)
        load_paths.append(np.maximum(current_load + load_residual, 0.0))
        pv_paths.append(np.maximum(current_pv + pv_residual, 0.0))
    count = len(source_dates)
    return VersionScenarios(tuple(source_dates), np.full(count, 1.0 / count), np.vstack(load_paths), np.vstack(pv_paths), start)


def _execute_main(data, cache) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slot_rows, plan_rows, solver_rows, scenario_rows, info_rows = [], [], [], [], []
    soc = SOC_INITIAL_TEST_KWH
    for date in TEST_DATES:
        d = data.day_index(date)
        day_soc_start = soc
        original = None
        effective = np.full(144, np.nan)
        for issue in ISSUE_HOURS:
            start = FIRST_MODIFIABLE_SLOT[issue]
            stop = NEXT_BOUNDARY_SLOT[issue]
            forecast = cache[(date, issue)].sort_values("idx")
            scenarios = _joint_scenarios(data, cache, date, issue)
            solve = solve_version_milp(scenarios, data.price_yuan_per_kwh, soc, TERMINAL_VALUE_TEST, original)
            version_values = solve.plan.sort_values("slot")["planned_purchase_kwh"].to_numpy(float)
            if issue == 0:
                original = np.full(144, np.nan)
                original[start:] = version_values
            prior_prefix = effective[:start].copy()
            effective[start:] = version_values
            prefix_error = float(np.nanmax(np.abs(effective[:start] - prior_prefix))) if start else 0.0
            if start and (np.isnan(effective[:start]).any() or prefix_error > 0.0):
                raise AssertionError("Executed plan prefix was modified")

            run_id = f"{date}_{issue:02d}00"
            solver_rows.append({"run_id": run_id, "date": date, "issue_hour": issue, "start_slot": start, **solve.metadata})
            dispatched = solve.scenario_dispatch.copy()
            dispatched.insert(0, "run_id", run_id)
            scenario_rows.append(dispatched)
            for _, row in solve.plan.iterrows():
                t = int(row.slot)
                plan_rows.append({
                    "run_id": run_id, "date": date, "issue_hour": issue, "decision_datetime": f"{date} {issue:02d}:00:00",
                    "slot": t, "time_label": data.time_labels[t], "original_plan_kwh": float(original[t]),
                    "version_plan_kwh": float(row.planned_purchase_kwh),
                    "delta_plus_vs_original_kwh": max(float(row.planned_purchase_kwh - original[t]), 0.0),
                    "delta_minus_vs_original_kwh": max(float(original[t] - row.planned_purchase_kwh), 0.0),
                    "modifiable": True, "executed_prefix_lock_error_kwh": prefix_error,
                })
            info_rows.append({
                "run_id": run_id, "date": date, "issue_hour": issue, "decision_datetime": f"{date} {issue:02d}:00:00",
                "first_modifiable_slot": start, "last_modifiable_slot": 143,
                "scenario_source_dates": "|".join(scenarios.source_dates),
                "scenario_count_requested": SCENARIO_CAP_TEST, "scenario_count_effective": scenarios.count,
                "all_scenario_dates_before_decision": all(s < date for s in scenarios.source_dates),
                "forecast_visible": bool(forecast["forecast_visible_at_decision"].all()),
                "pv_information_causal": bool(forecast["pv_information_causal"].all()),
                "no_future_actual_load_used": bool(forecast["no_future_actual_load_used"].all()),
                "actual_pv_used_in_forecast": bool(forecast["pv_actual_observation_used"].any()),
                "value_function_refresh_id": run_id,
                "value_function_plan_hash": hashlib.sha256(version_values.tobytes()).hexdigest().upper(),
            })

            values = build_path_value_functions(
                scenarios, version_values, data.price_yuan_per_kwh[start:], SOC_GRID_TEST_KWH, TERMINAL_VALUE_TEST
            )
            for t in range(start, stop):
                local = t - start
                z = float(effective[t])
                gap = float(data.load_kwh[d, t] - data.pv_kwh[d, t] - z)
                charge, discharge, emergency, spill, next_soc = select_dp_action(
                    gap, soc, float(data.price_yuan_per_kwh[t]), values.mean_value_yuan[local + 1], values.soc_grid_kwh
                )
                settled = settle_contract([original[t]], [z], [data.price_yuan_per_kwh[t]]).iloc[0]
                emergency_cost = EMERGENCY_MULTIPLIER * data.price_yuan_per_kwh[t] * emergency
                balance = z + data.pv_kwh[d, t] + discharge + emergency - data.load_kwh[d, t] - charge - spill
                slot_rows.append({
                    "strategy": "dynamic_value_main", "date": date, "slot": t, "time_label": data.time_labels[t],
                    "active_issue_hour": issue, "active_plan_run_id": run_id, "value_function_refresh_id": run_id,
                    "price_yuan_per_kwh": data.price_yuan_per_kwh[t], "load_actual_kwh": data.load_kwh[d, t],
                    "pv_actual_kwh": data.pv_kwh[d, t], "net_load_actual_kwh": data.load_kwh[d, t] - data.pv_kwh[d, t],
                    "original_plan_kwh": original[t], "effective_purchase_kwh": z,
                    "delta_plus_kwh": settled.delta_plus_kwh, "delta_minus_kwh": settled.delta_minus_kwh,
                    "charge_rt_kwh": charge, "discharge_rt_kwh": discharge, "emergency_purchase_kwh": emergency,
                    "spill_kwh": spill, "soc_start_kwh": soc, "soc_end_kwh": next_soc,
                    "contract_cost_yuan": settled.contract_cost_yuan, "emergency_cost_yuan": emergency_cost,
                    "actual_total_cost_yuan": settled.contract_cost_yuan + emergency_cost,
                    "balance_residual_kwh": balance,
                    "realtime_observation_assumption": "current_slot_actual_net_load_observable_at_execution",
                })
                soc = float(next_soc)
        if abs(day_soc_start - slot_rows[-144]["soc_start_kwh"]) > 1e-9:
            raise AssertionError("Day-start SOC record mismatch")
    return (pd.DataFrame(slot_rows), pd.DataFrame(plan_rows), pd.DataFrame(solver_rows),
            pd.concat(scenario_rows, ignore_index=True), pd.DataFrame(info_rows))


def _paired_greedy(main_slots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, day in main_slots.groupby("date", sort=True):
        day = day.sort_values("slot")
        soc = float(day.iloc[0].soc_start_kwh)  # Same main-strategy daily start; conditional ablation.
        for _, src in day.iterrows():
            gap = float(src.load_actual_kwh - src.pv_actual_kwh - src.effective_purchase_kwh)
            charge, discharge, emergency, spill, next_soc = greedy_action(gap, soc)
            emergency_cost = EMERGENCY_MULTIPLIER * float(src.price_yuan_per_kwh) * emergency
            balance = src.effective_purchase_kwh + src.pv_actual_kwh + discharge + emergency - src.load_actual_kwh - charge - spill
            rows.append({
                "strategy": "paired_greedy_fixed_main_plan_versions", "date": date, "slot": int(src.slot),
                "time_label": src.time_label, "active_issue_hour": int(src.active_issue_hour),
                "price_yuan_per_kwh": src.price_yuan_per_kwh, "load_actual_kwh": src.load_actual_kwh,
                "pv_actual_kwh": src.pv_actual_kwh, "original_plan_kwh": src.original_plan_kwh,
                "effective_purchase_kwh": src.effective_purchase_kwh, "charge_rt_kwh": charge,
                "discharge_rt_kwh": discharge, "emergency_purchase_kwh": emergency, "spill_kwh": spill,
                "soc_start_kwh": soc, "soc_end_kwh": next_soc, "contract_cost_yuan": src.contract_cost_yuan,
                "emergency_cost_yuan": emergency_cost, "actual_total_cost_yuan": src.contract_cost_yuan + emergency_cost,
                "balance_residual_kwh": balance,
                "comparison_scope": "same_main_plan_versions_and_same_main_daily_initial_soc_conditional_ablation",
            })
            soc = float(next_soc)
    return pd.DataFrame(rows)


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["strategy", "date"], as_index=False).agg(
        soc_start_kwh=("soc_start_kwh", "first"), soc_end_kwh=("soc_end_kwh", "last"),
        contract_cost_yuan=("contract_cost_yuan", "sum"), emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        actual_total_cost_yuan=("actual_total_cost_yuan", "sum"), emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
        charge_kwh=("charge_rt_kwh", "sum"), discharge_kwh=("discharge_rt_kwh", "sum"), spill_kwh=("spill_kwh", "sum"),
    )


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected_manifest()
    (OUT / "protected_hashes_before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "scope": list(TEST_DATES), "initial_soc_kwh": SOC_INITIAL_TEST_KWH,
        "initial_soc_provenance": "independent_three_day_closure_initialization_not_Jan1_continuous_state",
        "M_test": SCENARIO_CAP_TEST, "terminal_value_test_yuan_per_kwh": TERMINAL_VALUE_TEST,
        "soc_grid_test_kwh": SOC_GRID_TEST_KWH, "issue_hours": ISSUE_HOURS,
        "first_modifiable_slots_official_audited": FIRST_MODIFIABLE_SLOT,
        "settlement": "original plan plus 1.5*upward minus 0.5*downward; final effective version once per slot",
        "realtime_observation": "current-slot actual net load is observable at execution; not known earlier",
        "parameters_frozen_for_q3": False, "result3_generated": False,
    }
    (OUT / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    data = load_inputs()
    actual = _actual_long(data)
    pv_records = load_q3_pv_forecast_records()
    cache = _forecast_cache(actual, pv_records)
    main, plans, solver, scenario_dispatch, info = _execute_main(data, cache)
    paired = _paired_greedy(main)
    daily = pd.concat([_daily(main), _daily(paired)], ignore_index=True)
    manual = manual_settlement_cases()

    main.to_csv(OUT / "q3_v2_stage2_slot_results.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    plans.to_csv(OUT / "q3_v2_stage2_plan_versions.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    solver.to_csv(OUT / "q3_v2_stage2_solver_log.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    scenario_dispatch.to_csv(OUT / "q3_v2_stage2_scenario_dispatch.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    info.to_csv(OUT / "q3_v2_stage2_information_audit.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT / "q3_v2_stage2_paired_greedy_slots.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    daily.to_csv(OUT / "q3_v2_stage2_daily_results.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    manual.to_csv(OUT / "q3_v2_settlement_manual_cases.csv", index=False, encoding="utf-8-sig", float_format="%.10f")

    after = protected_manifest()
    (OUT / "protected_hashes_after.json").write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if before != after:
        raise RuntimeError("A protected file changed during Q3 V2 Stage 2")
    return {"main": main, "paired": paired, "daily": daily, "plans": plans, "solver": solver, "scenario": scenario_dispatch, "info": info, "manual": manual}


if __name__ == "__main__":
    started = time.perf_counter()
    outcome = run()
    print(outcome["daily"].to_string(index=False))
    print(f"elapsed_seconds={time.perf_counter()-started:.3f}")

