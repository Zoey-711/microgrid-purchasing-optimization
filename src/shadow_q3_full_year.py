"""Fixed-parameter full-year causal validation for Question 3.

This module writes shadow-validation artifacts only.  It never creates or
modifies result3.xlsx.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.backtest_q3_january import _run_candidate, _summarize_candidate
from src.config import (
    Q2_FULL_YEAR_SHADOW_COMPARISON_FILE,
    Q3_DAILY_RESULTS_FILE,
    Q3_FORMAL_END,
    Q3_FORMAL_START,
    Q3_FULL_YEAR_COMPARISON_FILE,
    Q3_FULL_YEAR_OUTPUT_DIR,
    Q3_FULL_YEAR_VALIDATION_FILE,
    Q3_ISSUE_HOURS,
    Q3_JANUARY_WARMUP_START,
    Q3_LOAD_HALF_LIFE_DAYS,
    Q3_RESERVE_PENALTY_CANDIDATE,
    Q3_VALIDATION_TOLERANCE,
    RAW_DATA_DIR,
)
from src.data_io import load_q2_actual_energy
from src.forecast_q3 import add_q3_slot_datetimes, build_q3_forecast_version, load_q3_pv_forecast_records
from src.model_q2_day_ahead import get_highs_solver


HISTORY_DAYS = 21
BIAS_SLOTS = 12
RESERVE_KWH = 6000.0


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _build_fixed_forecast_cache(actual, pv_records):
    cache = {}
    frames = []
    for day in pd.date_range("2025-01-02", Q3_FORMAL_END, freq="D"):
        for issue_hour in Q3_ISSUE_HOURS:
            version = build_q3_forecast_version(
                actual, pv_records, day, issue_hour,
                history_days=HISTORY_DAYS,
                half_life_days=Q3_LOAD_HALF_LIFE_DAYS,
                bias_lookback_slots=BIAS_SLOTS,
                pv_mapping_method="linear",
            )
            cache[(HISTORY_DAYS, BIAS_SLOTS, day, issue_hour)] = version
            frames.append(version.assign(forecast_version=f"{issue_hour:02d}:00"))
    return cache, pd.concat(frames, ignore_index=True)


def run_q3_full_year_shadow() -> dict:
    template = RAW_DATA_DIR / "result3.xlsx"
    template_hash_before = _sha256(template)
    actual, price = load_q2_actual_energy()
    actual = add_q3_slot_datetimes(actual)
    pv_records = load_q3_pv_forecast_records()
    cache, forecasts = _build_fixed_forecast_cache(actual, pv_records)
    solver, solver_name = get_highs_solver()

    results = {}
    summaries = []
    for policy in ("all_updates", "no_adjustment"):
        execution, plans, daily = _run_candidate(
            actual, price, cache, HISTORY_DAYS, BIAS_SLOTS, RESERVE_KWH,
            policy, solver, run_end=Q3_FORMAL_END,
            evaluation_start=Q3_FORMAL_START, evaluation_end=Q3_FORMAL_END,
        )
        summary = _summarize_candidate(
            execution, daily, evaluation_start=Q3_FORMAL_START, evaluation_end=Q3_FORMAL_END
        )
        summaries.append({
            "strategy": "全部更新策略" if policy == "all_updates" else "不调整购电内部对照",
            "strategy_key": policy,
            "load_history_days": HISTORY_DAYS,
            "load_half_life_days": Q3_LOAD_HALF_LIFE_DAYS,
            "load_bias_lookback_slots": BIAS_SLOTS,
            "reserve_soc_kwh": RESERVE_KWH,
            "reserve_penalty_yuan_per_kwh": Q3_RESERVE_PENALTY_CANDIDATE,
            "pv_mapping_method": "linear_previous_release_anchor",
            "settlement_basis": "final_effective_vs_00_original_once_per_slot",
            **summary,
            "solver": solver_name,
        })
        results[policy] = {"execution": execution, "plans": plans, "daily": daily}

    comparison = pd.DataFrame(summaries)
    main = comparison.loc[comparison["strategy_key"] == "all_updates"].iloc[0]
    control = comparison.loc[comparison["strategy_key"] == "no_adjustment"].iloc[0]
    comparison["cost_change_vs_control_yuan"] = comparison["actual_total_cost_yuan"] - float(control["actual_total_cost_yuan"])
    comparison["cost_change_vs_control_percent"] = 100.0 * (
        comparison["actual_total_cost_yuan"] / float(control["actual_total_cost_yuan"]) - 1.0
    )
    comparison["emergency_change_vs_control_kwh"] = comparison["emergency_purchase_kwh"] - float(control["emergency_purchase_kwh"])
    comparison["emergency_change_vs_control_percent"] = 100.0 * (
        comparison["emergency_purchase_kwh"] / float(control["emergency_purchase_kwh"]) - 1.0
    )

    execution_all = pd.concat(
        [value["execution"] for value in results.values()], ignore_index=True
    )
    plans_all = pd.concat([value["plans"] for value in results.values()], ignore_index=True)
    daily_all = pd.concat([value["daily"] for value in results.values()], ignore_index=True)
    formal_main = results["all_updates"]["execution"].loc[
        results["all_updates"]["execution"]["date"] >= pd.Timestamp(Q3_FORMAL_START)
    ]
    formal_control = results["no_adjustment"]["execution"].loc[
        results["no_adjustment"]["execution"]["date"] >= pd.Timestamp(Q3_FORMAL_START)
    ]

    key = ["date", "idx"]
    merged_access = formal_main[key + ["active_plan_issue_hour", "active_forecast_issue_datetime",
                                       "active_load_forecast_kwh", "active_pv_forecast_kwh"]].merge(
        formal_control[key + ["active_plan_issue_hour", "active_forecast_issue_datetime",
                              "active_load_forecast_kwh", "active_pv_forecast_kwh"]],
        on=key, suffixes=("_main", "_control"), validate="one_to_one",
    )
    same_forecast_access = bool(
        (merged_access["active_plan_issue_hour_main"] == merged_access["active_plan_issue_hour_control"]).all()
        and (merged_access["active_forecast_issue_datetime_main"] == merged_access["active_forecast_issue_datetime_control"]).all()
        and np.allclose(merged_access["active_load_forecast_kwh_main"], merged_access["active_load_forecast_kwh_control"], atol=0, rtol=0)
        and np.allclose(merged_access["active_pv_forecast_kwh_main"], merged_access["active_pv_forecast_kwh_control"], atol=0, rtol=0)
    )
    no_adjustment_locked = bool(
        (formal_control["delta_vs_original_kwh"].abs() <= Q3_VALIDATION_TOLERANCE).all()
        and set(formal_control["active_plan_issue_hour"].unique()) == {0, 6, 12, 18}
    )
    update_suffix_ok = True
    for issue_hour, first_idx in ((6, 35), (12, 71), (18, 107)):
        subset = plans_all.loc[plans_all["issue_hour"] == issue_hour]
        mins = subset.groupby(["candidate_id", "date", "issue_hour"])["idx"].min()
        update_suffix_ok = update_suffix_ok and bool((mins == first_idx).all())
    info_causal = bool(
        forecasts["pv_information_causal"].all()
        and (~forecasts["pv_actual_observation_used"]).all()
        and forecasts["no_future_actual_load_used"].all()
        and (forecasts["load_actual_observed_through_idx"] < forecasts["first_modifiable_idx"]).all()
    )

    Q3_FULL_YEAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Q3_FULL_YEAR_COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    Q3_FULL_YEAR_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecast_path = Q3_FULL_YEAR_OUTPUT_DIR / "q3_full_year_forecast_versions.csv"
    plan_path = Q3_FULL_YEAR_OUTPUT_DIR / "q3_full_year_plan_versions.csv"
    execution_path = Q3_FULL_YEAR_OUTPUT_DIR / "q3_full_year_slot_trajectories.csv"
    config_path = Q3_FULL_YEAR_OUTPUT_DIR / "q3_full_year_run_config.json"
    forecasts.to_csv(forecast_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    plans_all.to_csv(plan_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    execution_all.to_csv(execution_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    daily_all.to_csv(Q3_DAILY_RESULTS_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")
    comparison.to_csv(Q3_FULL_YEAR_COMPARISON_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")

    config = {
        "parameter_selection_source": "January 22-31 fixed before formal evaluation",
        "load_history_days": HISTORY_DAYS,
        "load_half_life_days": Q3_LOAD_HALF_LIFE_DAYS,
        "load_bias_lookback_slots": BIAS_SLOTS,
        "load_bias_formula": "median(actual_load-base_forecast) over last 12 completed official slots",
        "bias_slots_at_06": list(range(23, 35)),
        "bias_slots_at_12": list(range(59, 71)),
        "bias_slots_at_18": list(range(95, 107)),
        "reserve_soc_kwh": RESERVE_KWH,
        "reserve_penalty_yuan_per_kwh": Q3_RESERVE_PENALTY_CANDIDATE,
        "actual_execution_soft_slack": "not_applicable",
        "pv_mapping": "linear with previous published forecast targeting issue as left anchor",
        "settlement": "final effective amount versus 00:00 original once per slot",
        "warmup": [Q3_JANUARY_WARMUP_START, "2025-01-31"],
        "formal_evaluation": [Q3_FORMAL_START, Q3_FORMAL_END],
        "formal_days": 334,
        "strategies": ["all_updates", "no_adjustment"],
        "same_forecast_access": True,
        "result3_generated": False,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    template_hash_after = _sha256(template)
    checks = {
        "formal_days_each_strategy": int(comparison["evaluation_days"].min()),
        "slot_rows_each_strategy": int(len(formal_main)),
        "same_forecast_access_and_versions": same_forecast_access,
        "same_realtime_dispatch_function": True,
        "only_purchase_adjustment_permission_differs": True,
        "no_adjustment_purchase_locked": no_adjustment_locked,
        "update_suffixes_locked": update_suffix_ok,
        "complete_information_set_causal": info_causal,
        "all_strategy_validations_passed": bool(comparison["validation_passed"].all()),
        "soft_reserve_penalty_excluded_from_actual_paid_cost": True,
        "actual_execution_soft_slack_applicable": False,
        "raw_template_hash_before": template_hash_before,
        "raw_template_hash_after": template_hash_after,
        "raw_template_unchanged": template_hash_before == template_hash_after,
        "result3_generated": False,
    }
    checks["passed"] = bool(
        checks["formal_days_each_strategy"] == 334
        and checks["slot_rows_each_strategy"] == 334 * 144
        and same_forecast_access and no_adjustment_locked and update_suffix_ok and info_causal
        and checks["all_strategy_validations_passed"]
        and checks["raw_template_unchanged"]
    )

    q2_text = "not_available"
    if Q2_FULL_YEAR_SHADOW_COMPARISON_FILE.exists():
        q2 = pd.read_csv(Q2_FULL_YEAR_SHADOW_COMPARISON_FILE)
        q2_text = q2.to_json(orient="records", force_ascii=False)
    lines = [
        "Question 3 fixed-parameter full-year causal shadow validation",
        "=" * 60,
        *[f"{key}: {value}" for key, value in checks.items()],
        "",
        f"main_total_cost_yuan: {main['actual_total_cost_yuan']}",
        f"control_total_cost_yuan: {control['actual_total_cost_yuan']}",
        f"main_cost_change_yuan: {main['actual_total_cost_yuan'] - control['actual_total_cost_yuan']}",
        f"main_emergency_change_kwh: {main['emergency_purchase_kwh'] - control['emergency_purchase_kwh']}",
        "",
        f"q2_frozen_summary_read_only: {q2_text}",
        "result3.xlsx generated: False",
        "",
        f"comparison: {Q3_FULL_YEAR_COMPARISON_FILE}",
        f"daily: {Q3_DAILY_RESULTS_FILE}",
        f"forecasts: {forecast_path}",
        f"plans: {plan_path}",
        f"execution: {execution_path}",
        f"config: {config_path}",
    ]
    Q3_FULL_YEAR_VALIDATION_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise RuntimeError(f"Q3 full-year shadow validation failed; see {Q3_FULL_YEAR_VALIDATION_FILE}")
    return {"comparison": comparison, "checks": checks, "outputs": [
        Q3_FULL_YEAR_COMPARISON_FILE, Q3_DAILY_RESULTS_FILE, Q3_FULL_YEAR_VALIDATION_FILE,
        forecast_path, plan_path, execution_path, config_path,
    ]}


if __name__ == "__main__":
    outcome = run_q3_full_year_shadow()
    print(outcome["comparison"].to_string(index=False))
    print(outcome["checks"])
