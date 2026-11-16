"""Causal January parameter backtest for Question 3; never exports result3.xlsx."""

from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
import pandas as pd

from src.config import (
    E_B_MAX,
    Q3_ADJUSTMENT_POLICIES,
    Q3_FIRST_MODIFIABLE_SLOT,
    Q3_ISSUE_HOURS,
    Q3_JANUARY_EVALUATION_END,
    Q3_JANUARY_EVALUATION_START,
    Q3_JANUARY_OUTPUT_DIR,
    Q3_JANUARY_PARAMETER_FILE,
    Q3_JANUARY_WARMUP_START,
    Q3_LOAD_BIAS_LOOKBACK_CANDIDATES,
    Q3_LOAD_HALF_LIFE_DAYS,
    Q3_LOAD_HISTORY_DAY_CANDIDATES,
    Q3_RESERVE_PENALTY_CANDIDATE,
    Q3_RESERVE_SOC_CANDIDATES,
    Q3_VALIDATION_TOLERANCE,
    RAW_DATA_DIR,
    S_INIT,
    S_MAX,
    S_MIN,
)
from src.data_io import load_q2_actual_energy
from src.dispatch_q3_realtime import dispatch_q3_slot
from src.forecast_q3 import (
    add_q3_slot_datetimes,
    build_q3_forecast_version,
    load_q3_pv_forecast_records,
)
from src.model_q2_day_ahead import get_highs_solver
from src.model_q3_rolling import solve_q3_reference_plan
from src.q3_settlement import settle_adjusted_purchase


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _candidate_id(history_days: int, bias_slots: int, reserve: float, policy: str) -> str:
    return f"h{history_days}_b{bias_slots}_s{int(reserve)}_{policy}"


def _build_forecast_cache(actual, pv_records):
    cache = {}
    stored = []
    dates = pd.date_range("2025-01-02", Q3_JANUARY_EVALUATION_END, freq="D")
    for history_days, bias_slots in product(
        Q3_LOAD_HISTORY_DAY_CANDIDATES, Q3_LOAD_BIAS_LOOKBACK_CANDIDATES
    ):
        for day in dates:
            for issue_hour in Q3_ISSUE_HOURS:
                version = build_q3_forecast_version(
                    actual,
                    pv_records,
                    day,
                    issue_hour,
                    history_days=int(history_days),
                    half_life_days=float(Q3_LOAD_HALF_LIFE_DAYS),
                    bias_lookback_slots=int(bias_slots),
                )
                cache[(int(history_days), int(bias_slots), day, int(issue_hour))] = version
                stored.append(version.assign(
                    forecast_parameter_id=f"h{history_days}_b{bias_slots}",
                    forecast_version=f"{issue_hour:02d}:00",
                ))
    return cache, pd.concat(stored, ignore_index=True)


def _run_candidate(
    actual: pd.DataFrame,
    price: pd.DataFrame,
    forecast_cache: dict,
    history_days: int,
    bias_slots: int,
    reserve_soc_kwh: float,
    policy: str,
    solver,
    run_end: str = Q3_JANUARY_EVALUATION_END,
    evaluation_start: str = Q3_JANUARY_EVALUATION_START,
    evaluation_end: str = Q3_JANUARY_EVALUATION_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_id = _candidate_id(history_days, bias_slots, reserve_soc_kwh, policy)
    price_map = price.set_index("idx")["price_yuan_per_kwh"]
    dates = pd.date_range(Q3_JANUARY_WARMUP_START, run_end, freq="D")
    soc = float(S_INIT)
    execution_rows = []
    plan_frames = []
    daily_rows = []
    for day in dates:
        day_actual = actual.loc[actual["date"] == day].sort_values("idx").reset_index(drop=True)
        day_start_soc = soc
        if day == pd.Timestamp(Q3_JANUARY_WARMUP_START):
            # Common, fully causal cold start: no prior load history exists.
            original = pd.Series(0.0, index=range(144))
            effective = original.copy()
            previous_at_last_update = original.copy()
            reference_soc_end = pd.Series(S_MIN, index=range(144), dtype=float)
            active_issue = pd.Series(0, index=range(144), dtype=int)
            active_forecast_load = pd.Series(np.nan, index=range(144), dtype=float)
            active_forecast_pv = pd.Series(np.nan, index=range(144), dtype=float)
            active_forecast_net = pd.Series(np.nan, index=range(144), dtype=float)
            active_forecast_issue_datetime = pd.Series(pd.NaT, index=range(144), dtype="datetime64[ns]")
            day_ahead_shortfall = 0.0
            plan_frames.append(pd.DataFrame({
                "candidate_id": candidate_id,
                "date": day,
                "idx": range(144),
                "time_label": day_actual["time_label"],
                "version": "00:00_cold_start_zero",
                "issue_hour": 0,
                "original_purchase_kwh": 0.0,
                "previous_effective_purchase_kwh": 0.0,
                "reference_purchase_kwh": 0.0,
                "delta_vs_previous_kwh": 0.0,
                "upward_vs_original_kwh": 0.0,
                "downward_vs_original_kwh": 0.0,
                "decision_initial_soc_kwh": day_start_soc,
                "reserve_shortfall_kwh": 0.0,
                "soft_reserve_penalty_objective_yuan": 0.0,
            }))
        else:
            forecast0 = forecast_cache[(history_days, bias_slots, day, 0)]
            plan0, meta0 = solve_q3_reference_plan(
                forecast0, price, soc, reserve_soc_kwh,
                Q3_RESERVE_PENALTY_CANDIDATE, original_purchase=None, solver=solver,
            )
            original = plan0.set_index("idx")["reference_purchase_kwh"].copy()
            effective = original.copy()
            previous_at_last_update = original.copy()
            reference_soc_end = plan0.set_index("idx")["reference_soc_end_kwh"].copy()
            active_issue = pd.Series(0, index=range(144), dtype=int)
            forecast0_by_idx = forecast0.set_index("idx")
            active_forecast_load = forecast0_by_idx["load_forecast_kwh"].copy()
            active_forecast_pv = forecast0_by_idx["pv_forecast_kwh"].copy()
            active_forecast_net = forecast0_by_idx["net_load_forecast_kwh"].copy()
            active_forecast_issue_datetime = pd.Series(
                pd.Timestamp(forecast0["issue_datetime"].iloc[0]), index=range(144), dtype="datetime64[ns]"
            )
            day_ahead_shortfall = float(meta0["reserve_shortfall_kwh"])
            p0 = plan0.assign(
                candidate_id=candidate_id,
                version="00:00_original",
                original_purchase_kwh=plan0["reference_purchase_kwh"],
                previous_effective_purchase_kwh=plan0["reference_purchase_kwh"],
                delta_vs_previous_kwh=0.0,
                decision_initial_soc_kwh=day_start_soc,
                reserve_shortfall_kwh=day_ahead_shortfall,
                soft_reserve_penalty_objective_yuan=(
                    Q3_RESERVE_PENALTY_CANDIDATE * day_ahead_shortfall
                ),
            )
            plan_frames.append(p0)

        update_by_start = (
            {Q3_FIRST_MODIFIABLE_SLOT[h]: h for h in Q3_ISSUE_HOURS if h != 0}
            if day != pd.Timestamp(Q3_JANUARY_WARMUP_START) else {}
        )
        day_exec_start = len(execution_rows)
        for t in range(144):
            if t in update_by_start:
                issue_hour = update_by_start[t]
                version = forecast_cache[(history_days, bias_slots, day, issue_hour)]
                old_effective = effective.copy()
                updated, metadata = solve_q3_reference_plan(
                    version, price, soc, reserve_soc_kwh,
                    Q3_RESERVE_PENALTY_CANDIDATE,
                    original_purchase=original,
                    fixed_purchase=(original if policy == "no_adjustment" else None),
                    solver=solver,
                )
                updated_idx = updated["idx"].astype(int)
                updated = updated.assign(
                    candidate_id=candidate_id,
                    version=(
                        f"{issue_hour:02d}:00_adjustment"
                        if policy == "all_updates"
                        else f"{issue_hour:02d}:00_storage_update_purchase_locked"
                    ),
                    original_purchase_kwh=[float(original.at[i]) for i in updated_idx],
                    previous_effective_purchase_kwh=[float(old_effective.at[i]) for i in updated_idx],
                    delta_vs_previous_kwh=(
                        updated["reference_purchase_kwh"].to_numpy()
                        - np.array([float(old_effective.at[i]) for i in updated_idx])
                    ),
                    decision_initial_soc_kwh=soc,
                    reserve_shortfall_kwh=float(metadata["reserve_shortfall_kwh"]),
                    soft_reserve_penalty_objective_yuan=(
                        Q3_RESERVE_PENALTY_CANDIDATE * float(metadata["reserve_shortfall_kwh"])
                    ),
                )
                plan_frames.append(updated)
                for row in updated.itertuples(index=False):
                    i = int(row.idx)
                    previous_at_last_update.at[i] = float(old_effective.at[i])
                    effective.at[i] = float(row.reference_purchase_kwh)
                    reference_soc_end.at[i] = float(row.reference_soc_end_kwh)
                    active_issue.at[i] = issue_hour
                    active_forecast_load.at[i] = float(row.load_forecast_kwh)
                    active_forecast_pv.at[i] = float(row.pv_forecast_kwh)
                    active_forecast_net.at[i] = float(row.net_load_forecast_kwh)
                    active_forecast_issue_datetime.at[i] = pd.Timestamp(row.issue_datetime)

            actual_row = day_actual.iloc[t]
            physical = dispatch_q3_slot(
                actual_row,
                float(effective.at[t]),
                soc,
                float(reference_soc_end.at[t]),
                Q3_VALIDATION_TOLERANCE,
            )
            settlement = settle_adjusted_purchase(
                [float(original.at[t])], [float(effective.at[t])], [float(price_map.at[t])]
            ).iloc[0]
            emergency_cost = 5.0 * float(price_map.at[t]) * physical["emergency_purchase_kwh"]
            execution_rows.append({
                "candidate_id": candidate_id,
                "load_history_days": history_days,
                "load_bias_lookback_slots": bias_slots,
                "load_half_life_days": Q3_LOAD_HALF_LIFE_DAYS,
                "reserve_soc_kwh": reserve_soc_kwh,
                "reserve_penalty_yuan_per_kwh": Q3_RESERVE_PENALTY_CANDIDATE,
                "adjustment_policy": policy,
                "date": day,
                "idx": t,
                "time_label": actual_row["time_label"],
                "slot_datetime": actual_row["slot_datetime"],
                "active_plan_issue_hour": int(active_issue.at[t]),
                "active_forecast_issue_datetime": active_forecast_issue_datetime.at[t],
                "active_load_forecast_kwh": float(active_forecast_load.at[t]),
                "active_pv_forecast_kwh": float(active_forecast_pv.at[t]),
                "active_net_load_forecast_kwh": float(active_forecast_net.at[t]),
                "original_purchase_kwh": float(original.at[t]),
                "previous_effective_at_last_update_kwh": float(previous_at_last_update.at[t]),
                "final_effective_purchase_kwh": float(effective.at[t]),
                "delta_vs_original_kwh": float(effective.at[t] - original.at[t]),
                "delta_vs_previous_kwh": float(effective.at[t] - previous_at_last_update.at[t]),
                "price_yuan_per_kwh": float(price_map.at[t]),
                **physical,
                "upward_adjustment_kwh": float(settlement.upward_adjustment_kwh),
                "downward_adjustment_kwh": float(settlement.downward_adjustment_kwh),
                "original_plan_cost_yuan": float(price_map.at[t] * original.at[t]),
                "retained_purchase_cost_yuan": float(settlement.retained_purchase_cost_yuan),
                "downward_breach_cost_yuan": float(settlement.downward_breach_cost_yuan),
                "upward_purchase_cost_yuan": float(settlement.upward_purchase_cost_yuan),
                "contract_settlement_cost_yuan": float(settlement.contract_settlement_cost_yuan),
                "emergency_purchase_cost_yuan": emergency_cost,
                "actual_paid_cost_yuan": float(settlement.contract_settlement_cost_yuan) + emergency_cost,
                "day_ahead_reserve_shortfall_kwh": day_ahead_shortfall,
                "day_ahead_soft_reserve_penalty_yuan": (
                    Q3_RESERVE_PENALTY_CANDIDATE * day_ahead_shortfall
                ),
            })
            soc = physical["soc_end_kwh"]

        day_exec = pd.DataFrame(execution_rows[day_exec_start:])
        daily_rows.append({
            "candidate_id": candidate_id,
            "date": day,
            "evaluation_day": (
                pd.Timestamp(evaluation_start) <= day <= pd.Timestamp(evaluation_end)
            ),
            "soc_start_kwh": day_start_soc,
            "soc_end_kwh": soc,
            "actual_end_below_reserve": soc < reserve_soc_kwh - Q3_VALIDATION_TOLERANCE,
            "actual_end_reserve_shortfall_kwh": max(0.0, reserve_soc_kwh - soc),
            "original_plan_cost_yuan": day_exec["original_plan_cost_yuan"].sum(),
            "retained_purchase_cost_yuan": day_exec["retained_purchase_cost_yuan"].sum(),
            "downward_breach_cost_yuan": day_exec["downward_breach_cost_yuan"].sum(),
            "upward_purchase_cost_yuan": day_exec["upward_purchase_cost_yuan"].sum(),
            "contract_settlement_cost_yuan": day_exec["contract_settlement_cost_yuan"].sum(),
            "emergency_purchase_cost_yuan": day_exec["emergency_purchase_cost_yuan"].sum(),
            "actual_paid_cost_yuan": day_exec["actual_paid_cost_yuan"].sum(),
            "emergency_purchase_kwh": day_exec["emergency_purchase_kwh"].sum(),
            "upward_adjustment_kwh": day_exec["upward_adjustment_kwh"].sum(),
            "downward_adjustment_kwh": day_exec["downward_adjustment_kwh"].sum(),
            "day_ahead_reserve_shortfall_kwh": day_ahead_shortfall,
            "day_ahead_soft_reserve_penalty_yuan": (
                Q3_RESERVE_PENALTY_CANDIDATE * day_ahead_shortfall
            ),
            "actual_execution_soft_slack_applicable": False,
            "actual_execution_soft_slack_kwh": np.nan,
            "actual_execution_soft_penalty_yuan": np.nan,
        })
    return pd.DataFrame(execution_rows), pd.concat(plan_frames, ignore_index=True), pd.DataFrame(daily_rows)


def _summarize_candidate(
    execution: pd.DataFrame,
    daily: pd.DataFrame,
    evaluation_start: str = Q3_JANUARY_EVALUATION_START,
    evaluation_end: str = Q3_JANUARY_EVALUATION_END,
) -> dict:
    tol = Q3_VALIDATION_TOLERANCE
    evaluation = execution.loc[
        (execution["date"] >= pd.Timestamp(evaluation_start))
        & (execution["date"] <= pd.Timestamp(evaluation_end))
    ].copy()
    daily_eval = daily.loc[daily["evaluation_day"]].copy()
    soc_all = pd.concat([execution["soc_start_kwh"], execution["soc_end_kwh"]])
    cross_day = np.abs(
        daily["soc_start_kwh"].iloc[1:].to_numpy(dtype=float)
        - daily["soc_end_kwh"].iloc[:-1].to_numpy(dtype=float)
    )
    independent = settle_adjusted_purchase(
        evaluation["original_purchase_kwh"],
        evaluation["final_effective_purchase_kwh"],
        evaluation["price_yuan_per_kwh"],
    )["contract_settlement_cost_yuan"].to_numpy() + (
        5.0 * evaluation["price_yuan_per_kwh"] * evaluation["emergency_purchase_kwh"]
    ).to_numpy()
    cost_error = float(np.max(np.abs(independent - evaluation["actual_paid_cost_yuan"])))
    report = {
        "original_plan_cost_yuan": float(evaluation["original_plan_cost_yuan"].sum()),
        "retained_purchase_cost_yuan": float(evaluation["retained_purchase_cost_yuan"].sum()),
        "downward_breach_cost_yuan": float(evaluation["downward_breach_cost_yuan"].sum()),
        "upward_purchase_cost_yuan": float(evaluation["upward_purchase_cost_yuan"].sum()),
        "contract_settlement_cost_yuan": float(evaluation["contract_settlement_cost_yuan"].sum()),
        "emergency_purchase_cost_yuan": float(evaluation["emergency_purchase_cost_yuan"].sum()),
        "actual_total_cost_yuan": float(evaluation["actual_paid_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(evaluation["emergency_purchase_kwh"].sum()),
        "upward_adjustment_kwh": float(evaluation["upward_adjustment_kwh"].sum()),
        "downward_adjustment_kwh": float(evaluation["downward_adjustment_kwh"].sum()),
        "mean_end_soc_kwh": float(daily_eval["soc_end_kwh"].mean()),
        "min_end_soc_kwh": float(daily_eval["soc_end_kwh"].min()),
        "actual_end_below_reserve_days": int(daily_eval["actual_end_below_reserve"].sum()),
        "actual_end_reserve_shortfall_kwh": float(daily_eval["actual_end_reserve_shortfall_kwh"].sum()),
        "max_actual_end_reserve_shortfall_kwh": float(daily_eval["actual_end_reserve_shortfall_kwh"].max()),
        "day_ahead_reserve_shortfall_kwh": float(daily_eval["day_ahead_reserve_shortfall_kwh"].sum()),
        "day_ahead_reserve_shortfall_days": int((daily_eval["day_ahead_reserve_shortfall_kwh"] > tol).sum()),
        "soft_reserve_penalty_objective_yuan": float(daily_eval["day_ahead_soft_reserve_penalty_yuan"].sum()),
        "actual_execution_soft_slack_applicable": False,
        "actual_execution_soft_slack_kwh": np.nan,
        "actual_execution_soft_penalty_yuan": np.nan,
        "max_daily_emergency_purchase_kwh": float(daily_eval["emergency_purchase_kwh"].max()),
        "max_balance_error_kwh": float(execution["balance_residual_kwh"].abs().max()),
        "soc_min_kwh": float(soc_all.min()),
        "soc_max_kwh": float(soc_all.max()),
        "max_charge_kwh": float(execution["charge_rt_kwh"].max()),
        "max_discharge_kwh": float(execution["discharge_rt_kwh"].max()),
        "soc_violation_count": int(((soc_all < S_MIN - tol) | (soc_all > S_MAX + tol)).sum()),
        "power_violation_count": int(((execution["charge_rt_kwh"] > E_B_MAX + tol) | (execution["discharge_rt_kwh"] > E_B_MAX + tol)).sum()),
        "simultaneous_charge_discharge_count": int(((execution["charge_rt_kwh"] > tol) & (execution["discharge_rt_kwh"] > tol)).sum()),
        "emergency_charge_overlap_count": int(((execution["emergency_purchase_kwh"] > tol) & (execution["charge_rt_kwh"] > tol)).sum()),
        "max_cross_day_soc_error_kwh": float(cross_day.max(initial=0.0)),
        "max_cost_recalculation_error_yuan": cost_error,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "evaluation_days": int(len(daily_eval)),
    }
    report["validation_passed"] = bool(
        report["soc_violation_count"] == 0
        and report["power_violation_count"] == 0
        and report["simultaneous_charge_discharge_count"] == 0
        and report["emergency_charge_overlap_count"] == 0
        and report["max_balance_error_kwh"] <= tol
        and report["max_cross_day_soc_error_kwh"] <= tol
        and report["max_cost_recalculation_error_yuan"] <= tol
    )
    return report


def run_q3_january_backtest() -> dict:
    template = RAW_DATA_DIR / "result3.xlsx"
    hash_before = _sha256(template)
    actual, price = load_q2_actual_energy()
    actual = add_q3_slot_datetimes(actual)
    pv_records = load_q3_pv_forecast_records()
    forecast_cache, forecasts = _build_forecast_cache(actual, pv_records)
    solver, solver_name = get_highs_solver()

    comparisons = []
    executions = []
    plans = []
    daily_results = []
    for history_days, bias_slots, reserve, policy in product(
        Q3_LOAD_HISTORY_DAY_CANDIDATES,
        Q3_LOAD_BIAS_LOOKBACK_CANDIDATES,
        Q3_RESERVE_SOC_CANDIDATES,
        Q3_ADJUSTMENT_POLICIES,
    ):
        execution, plan_versions, daily = _run_candidate(
            actual, price, forecast_cache, int(history_days), int(bias_slots),
            float(reserve), str(policy), solver,
        )
        summary = _summarize_candidate(execution, daily)
        candidate_id = execution["candidate_id"].iloc[0]
        comparisons.append({
            "candidate_id": candidate_id,
            "load_history_days": int(history_days),
            "load_half_life_days": float(Q3_LOAD_HALF_LIFE_DAYS),
            "load_bias_lookback_slots": int(bias_slots),
            "reserve_soc_kwh": float(reserve),
            "reserve_penalty_yuan_per_kwh": float(Q3_RESERVE_PENALTY_CANDIDATE),
            "adjustment_policy": str(policy),
            **summary,
            "solver": solver_name,
        })
        executions.append(execution)
        plans.append(plan_versions)
        daily_results.append(daily.assign(
            load_history_days=int(history_days),
            load_bias_lookback_slots=int(bias_slots),
            reserve_soc_kwh=float(reserve),
            reserve_penalty_yuan_per_kwh=float(Q3_RESERVE_PENALTY_CANDIDATE),
            adjustment_policy=str(policy),
        ))

    comparison = pd.DataFrame(comparisons)
    eligible = comparison.loc[
        comparison["validation_passed"] & (comparison["adjustment_policy"] == "all_updates")
    ].sort_values(
        ["actual_total_cost_yuan", "emergency_purchase_kwh", "load_history_days",
         "load_bias_lookback_slots", "reserve_soc_kwh"],
        kind="stable",
    )
    if eligible.empty:
        raise RuntimeError("No valid all-update Q3 January candidate")
    selected_id = str(eligible.iloc[0]["candidate_id"])
    comparison["selected_main"] = comparison["candidate_id"] == selected_id
    selected = comparison.loc[comparison["selected_main"]].iloc[0]
    control_mask = (
        (comparison["adjustment_policy"] == "no_adjustment")
        & (comparison["load_history_days"] == selected["load_history_days"])
        & (comparison["load_bias_lookback_slots"] == selected["load_bias_lookback_slots"])
        & (comparison["reserve_soc_kwh"] == selected["reserve_soc_kwh"])
    )
    comparison["selected_internal_control"] = control_mask
    comparison = comparison.sort_values(
        ["adjustment_policy", "validation_passed", "actual_total_cost_yuan"],
        ascending=[True, False, True], kind="stable",
    )

    Q3_JANUARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    execution_all = pd.concat(executions, ignore_index=True)
    plans_all = pd.concat(plans, ignore_index=True)
    daily_all = pd.concat(daily_results, ignore_index=True)
    update_start_ok = True
    for issue_hour in (6, 12, 18):
        subset = plans_all.loc[plans_all["issue_hour"] == issue_hour]
        if not subset.empty:
            minimum_by_version = subset.groupby(["candidate_id", "date", "issue_hour"])["idx"].min()
            update_start_ok = update_start_ok and bool(
                (minimum_by_version == Q3_FIRST_MODIFIABLE_SLOT[issue_hour]).all()
            )
    plan_delta_identity_error = float(np.nanmax(np.abs(
        plans_all["reference_purchase_kwh"].to_numpy(dtype=float)
        - plans_all["original_purchase_kwh"].to_numpy(dtype=float)
        - plans_all["upward_vs_original_kwh"].to_numpy(dtype=float)
        + plans_all["downward_vs_original_kwh"].to_numpy(dtype=float)
    )))
    execution_delta_identity_error = float(np.max(np.abs(
        execution_all["final_effective_purchase_kwh"].to_numpy(dtype=float)
        - execution_all["original_purchase_kwh"].to_numpy(dtype=float)
        - execution_all["delta_vs_original_kwh"].to_numpy(dtype=float)
    )))
    no_adjustment_rows = execution_all.loc[execution_all["adjustment_policy"] == "no_adjustment"]
    no_adjustment_locked = bool(
        (no_adjustment_rows["delta_vs_original_kwh"].abs() <= Q3_VALIDATION_TOLERANCE).all()
        and set(no_adjustment_rows.loc[
            no_adjustment_rows["date"] > pd.Timestamp(Q3_JANUARY_WARMUP_START),
            "active_plan_issue_hour",
        ].unique()) == {0, 6, 12, 18}
    )
    complete_information_causal = bool(
        forecasts["no_future_actual_load_used"].all()
        and forecasts["pv_information_causal"].all()
        and (~forecasts["pv_actual_observation_used"]).all()
        and (forecasts["load_actual_observed_through_idx"] < forecasts["first_modifiable_idx"]).all()
    )
    forecast_path = Q3_JANUARY_OUTPUT_DIR / "q3_january_forecast_versions.csv"
    plan_path = Q3_JANUARY_OUTPUT_DIR / "q3_january_plan_versions.csv"
    execution_path = Q3_JANUARY_OUTPUT_DIR / "q3_january_slot_trajectories.csv"
    daily_path = Q3_JANUARY_OUTPUT_DIR / "q3_january_daily_results.csv"
    log_path = Q3_JANUARY_OUTPUT_DIR / "q3_january_backtest_validation.txt"
    Q3_JANUARY_PARAMETER_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(Q3_JANUARY_PARAMETER_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")
    forecasts.to_csv(forecast_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    plans_all.to_csv(plan_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    execution_all.to_csv(execution_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    daily_all.to_csv(daily_path, index=False, encoding="utf-8-sig", float_format="%.10f")

    hash_after = _sha256(template)
    all_passed = bool(
        comparison["validation_passed"].all()
        and update_start_ok
        and plan_delta_identity_error <= Q3_VALIDATION_TOLERANCE
        and execution_delta_identity_error <= Q3_VALIDATION_TOLERANCE
        and no_adjustment_locked
        and complete_information_causal
    )
    if hash_before != hash_after:
        all_passed = False
    lines = [
        "Question 3 January common-period causal backtest",
        "=" * 48,
        f"warmup: {Q3_JANUARY_WARMUP_START} through 2025-01-21",
        f"evaluation: {Q3_JANUARY_EVALUATION_START} through {Q3_JANUARY_EVALUATION_END}",
        "cold_start: 2025-01-01 purchase=0; actual execution initializes history and SOC",
        f"candidate_count: {len(comparison)}",
        f"all_candidate_validation_passed: {all_passed}",
        f"complete_information_set_causal: {complete_information_causal}",
        f"update_suffix_start_indices_correct: {update_start_ok}",
        f"plan_delta_identity_max_error_kwh: {plan_delta_identity_error}",
        f"execution_delta_identity_max_error_kwh: {execution_delta_identity_error}",
        f"no_adjustment_purchase_locked_but_forecast_updates_visible: {no_adjustment_locked}",
        "actual_pv_issue_anchor_used: False",
        "settlement: final effective amount versus 00:00 original, once per slot",
        "soft_reserve_penalty_excluded_from_actual_paid_cost: True",
        f"selected_main: {selected_id}",
        f"selected_internal_control: {comparison.loc[control_mask, 'candidate_id'].iloc[0]}",
        f"raw_result3_hash_before: {hash_before}",
        f"raw_result3_hash_after: {hash_after}",
        f"raw_template_unchanged: {hash_before == hash_after}",
        "result3.xlsx generated: False",
        "",
        f"parameter_table: {Q3_JANUARY_PARAMETER_FILE}",
        f"forecast_versions: {forecast_path}",
        f"plan_versions: {plan_path}",
        f"slot_trajectories: {execution_path}",
        f"daily_results: {daily_path}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all_passed:
        raise RuntimeError(f"Q3 January backtest failed; see {log_path}")
    return {
        "comparison": comparison,
        "selected": selected.to_dict(),
        "control": comparison.loc[control_mask].iloc[0].to_dict(),
        "outputs": [Q3_JANUARY_PARAMETER_FILE, forecast_path, plan_path, execution_path, daily_path, log_path],
    }


if __name__ == "__main__":
    outcome = run_q3_january_backtest()
    print("Selected main:")
    print(outcome["selected"])
    print("Matched no-adjustment control:")
    print(outcome["control"])
