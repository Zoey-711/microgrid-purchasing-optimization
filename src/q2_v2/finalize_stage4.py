"""Aggregate and validate the frozen Q2-v2 Stage-4 shadow evaluation.

This module is deliberately post-processing only.  It never solves a model and
never writes either the original or the submitted result2 workbook.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ENERGY_MAX_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    ROOT,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    TOL,
)


STAGE4 = ROOT / "outputs/q2_v2/stage4"
MAIN = STAGE4 / "main_dp50"
GREEDY = STAGE4 / "closed_loop_greedy"
FORMAL_START = pd.Timestamp("2025-02-01")
FORMAL_END = pd.Timestamp("2025-12-31")
EXPECTED_DATES = pd.date_range(FORMAL_START, FORMAL_END, freq="D")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_days(folder: Path, dates=EXPECTED_DATES) -> pd.DataFrame:
    frames = []
    for day in dates:
        path = folder / f"{day.date().isoformat()}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _summary(policy: str, daily: pd.DataFrame, slots: pd.DataFrame,
             solver: pd.DataFrame) -> dict:
    emergency_kwh = float(slots["emergency_purchase_kwh"].sum())
    emergency_cost = float(slots["emergency_cost_yuan"].sum())
    limit_mask = solver["termination_condition"].astype(str).eq("maxTimeLimit")
    return {
        "policy_id": policy,
        "formal_days": int(len(daily)),
        "plan_cost_yuan": float(slots["plan_cost_yuan"].sum()),
        "emergency_cost_yuan": emergency_cost,
        "actual_total_cost_yuan": float(slots["actual_cost_yuan"].sum()),
        "emergency_purchase_kwh": emergency_kwh,
        "emergency_weighted_unit_price_yuan_per_kwh": (
            emergency_cost / emergency_kwh if emergency_kwh > 0 else np.nan
        ),
        "max_daily_emergency_purchase_kwh": float(daily["emergency_purchase_kwh"].max()),
        "mean_daily_end_soc_kwh": float(daily["soc_end_kwh"].mean()),
        "min_daily_end_soc_kwh": float(daily["soc_end_kwh"].min()),
        "formal_terminal_soc_kwh": float(daily.iloc[-1]["soc_end_kwh"]),
        "actual_soc_min_kwh": float(slots[["soc_start_kwh", "soc_end_kwh"]].min().min()),
        "actual_soc_max_kwh": float(slots[["soc_start_kwh", "soc_end_kwh"]].max().max()),
        "charge_kwh": float(slots["charge_kwh"].sum()),
        "discharge_kwh": float(slots["discharge_kwh"].sum()),
        "conversion_loss_kwh": float(daily["conversion_loss_kwh"].sum()),
        "spill_kwh": float(slots["spill_kwh"].sum()),
        "day_ahead_solve_time_seconds": float(solver["solve_time_seconds"].sum()),
        "time_limit_feasible_days": int(limit_mask.sum()),
        "certified_optimal_days": int(solver["optimal_solution_certified"].fillna(False).astype(bool).sum()),
        "max_reported_mip_gap": float(pd.to_numeric(solver["mip_gap"], errors="coerce").max()),
    }


def _monthly(policy: str, daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x["month"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m")
    out = x.groupby("month", as_index=False).agg(
        days=("date", "count"),
        plan_cost_yuan=("plan_cost_yuan", "sum"),
        emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        actual_total_cost_yuan=("actual_total_cost_yuan", "sum"),
        emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
        max_daily_emergency_purchase_kwh=("emergency_purchase_kwh", "max"),
        mean_daily_end_soc_kwh=("soc_end_kwh", "mean"),
        min_daily_end_soc_kwh=("soc_end_kwh", "min"),
        terminal_soc_kwh=("soc_end_kwh", "last"),
        charge_kwh=("charge_kwh", "sum"),
        discharge_kwh=("discharge_kwh", "sum"),
        conversion_loss_kwh=("conversion_loss_kwh", "sum"),
        spill_kwh=("spill_kwh", "sum"),
    )
    out.insert(0, "policy_id", policy)
    return out


def _price_band(slots: pd.DataFrame, policy: str) -> pd.DataFrame:
    # Descriptive tariff intervals fixed for reporting only; never used by a
    # predictor, optimiser, controller, or strategy selection.
    x = slots.copy()
    x["price_band"] = pd.cut(
        x["price_yuan_per_kwh"], [-np.inf, 0.6, 1.0, np.inf], right=False,
        labels=["低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"],
    )
    out = x.groupby("price_band", observed=False, as_index=False).agg(
        emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
        emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        slots_with_emergency=("emergency_purchase_kwh", lambda s: int((s > TOL).sum())),
    )
    out.insert(0, "policy_id", policy)
    return out


def _validate(main_daily: pd.DataFrame, main_slots: pd.DataFrame,
              greedy_daily: pd.DataFrame, greedy_slots: pd.DataFrame,
              paired_daily: pd.DataFrame, paired_slots: pd.DataFrame,
              plans: pd.DataFrame, greedy_plans: pd.DataFrame,
              solver: pd.DataFrame, greedy_solver: pd.DataFrame,
              info: pd.DataFrame, greedy_info: pd.DataFrame) -> dict:
    dates = pd.to_datetime(main_daily["date"])
    paired_dates = pd.to_datetime(paired_daily["date"])
    main_start = main_daily["soc_start_kwh"].to_numpy(float)
    main_end = main_daily["soc_end_kwh"].to_numpy(float)
    pair_start = paired_daily["soc_start_kwh"].to_numpy(float)
    pair_end = paired_daily["soc_end_kwh"].to_numpy(float)
    greedy_start = greedy_daily["soc_start_kwh"].to_numpy(float)
    greedy_end = greedy_daily["soc_end_kwh"].to_numpy(float)
    main_cross = np.max(np.abs(main_start[1:] - main_end[:-1]))
    pair_cross = np.max(np.abs(pair_start[1:] - pair_end[:-1]))
    greedy_cross = np.max(np.abs(greedy_start[1:] - greedy_end[:-1]))
    state_equiv = np.max(np.abs(main_end - pair_end))
    paired_start_matches_main = np.max(np.abs(main_start - pair_start))
    plan_repeat = plans["planned_purchase_kwh"].to_numpy(float)
    slot_plan = main_slots["planned_purchase_kwh"].to_numpy(float)
    greedy_plan_repeat = greedy_plans["planned_purchase_kwh"].to_numpy(float)
    greedy_slot_plan = greedy_slots["planned_purchase_kwh"].to_numpy(float)
    balance = np.max(np.abs(main_slots["balance_residual_kwh"].to_numpy(float)))
    pair_balance = np.max(np.abs(paired_slots["balance_residual_kwh"].to_numpy(float)))
    def transition_error(x: pd.DataFrame) -> float:
        expected = (x["soc_start_kwh"].to_numpy(float)
                    + ETA_CHARGE * x["charge_kwh"].to_numpy(float)
                    - x["discharge_kwh"].to_numpy(float) / ETA_DISCHARGE)
        return float(np.max(np.abs(expected - x["soc_end_kwh"].to_numpy(float))))

    def bill_error(x: pd.DataFrame) -> float:
        expected = x["plan_cost_yuan"].to_numpy(float) + x["emergency_cost_yuan"].to_numpy(float)
        return float(np.max(np.abs(expected - x["actual_cost_yuan"].to_numpy(float))))

    def daily_sum_error(daily: pd.DataFrame, slots: pd.DataFrame, col_daily: str, col_slot: str) -> float:
        left = daily[["date", col_daily]].rename(columns={col_daily: "daily_value"})
        right = slots.groupby("date", as_index=False)[col_slot].sum().rename(columns={col_slot: "slot_value"})
        merged = left.merge(right, on="date", validate="one_to_one")
        return float(np.max(np.abs(merged["daily_value"] - merged["slot_value"])))
    soc_min = min(main_slots["soc_start_kwh"].min(), main_slots["soc_end_kwh"].min(),
                  greedy_slots["soc_start_kwh"].min(), greedy_slots["soc_end_kwh"].min(),
                  paired_slots["soc_start_kwh"].min(), paired_slots["soc_end_kwh"].min())
    soc_max = max(main_slots["soc_start_kwh"].max(), main_slots["soc_end_kwh"].max(),
                  greedy_slots["soc_start_kwh"].max(), greedy_slots["soc_end_kwh"].max(),
                  paired_slots["soc_start_kwh"].max(), paired_slots["soc_end_kwh"].max())
    power_viol = int(((main_slots["charge_kwh"] > ENERGY_MAX_KWH + TOL) |
                      (main_slots["discharge_kwh"] > ENERGY_MAX_KWH + TOL) |
                      (paired_slots["charge_kwh"] > ENERGY_MAX_KWH + TOL) |
                      (paired_slots["discharge_kwh"] > ENERGY_MAX_KWH + TOL)).sum())
    power_viol += int(((greedy_slots["charge_kwh"] > ENERGY_MAX_KWH + TOL) |
                       (greedy_slots["discharge_kwh"] > ENERGY_MAX_KWH + TOL)).sum())
    overlap = int((((main_slots["charge_kwh"] > TOL) & (main_slots["discharge_kwh"] > TOL)).sum()) +
                  (((paired_slots["charge_kwh"] > TOL) & (paired_slots["discharge_kwh"] > TOL)).sum()))
    overlap += int(((greedy_slots["charge_kwh"] > TOL) & (greedy_slots["discharge_kwh"] > TOL)).sum())
    emergency_charge = int((((main_slots["emergency_purchase_kwh"] > TOL) &
                              (main_slots["charge_kwh"] > TOL)).sum()) +
                           (((paired_slots["emergency_purchase_kwh"] > TOL) &
                              (paired_slots["charge_kwh"] > TOL)).sum()))
    emergency_charge += int(((greedy_slots["emergency_purchase_kwh"] > TOL) &
                              (greedy_slots["charge_kwh"] > TOL)).sum())
    before = json.loads((STAGE4 / "protected_hashes_before_stage4.json").read_text(encoding="utf-8"))
    after = {name: _sha256(ROOT / name) for name in before}
    (STAGE4 / "protected_hashes_after_stage4.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    changed = sorted(name for name in before if before[name] != after[name])
    jan = pd.read_csv(ROOT / "outputs/q2_v2/stage3/selected/selected_M14_v04_dp50/daily_results.csv")
    checks = {
        "formal_date_count": int(len(dates)),
        "formal_dates_exact": bool(dates.equals(pd.Series(EXPECTED_DATES))),
        "paired_dates_exact": bool(paired_dates.equals(pd.Series(EXPECTED_DATES))),
        "greedy_dates_exact": bool(pd.to_datetime(greedy_daily["date"]).equals(pd.Series(EXPECTED_DATES))),
        "main_slot_rows": int(len(main_slots)),
        "greedy_slot_rows": int(len(greedy_slots)),
        "paired_slot_rows": int(len(paired_slots)),
        "expected_slot_rows": int(334 * 144),
        "each_day_144_slots": bool(main_slots.groupby("date").size().eq(144).all()),
        "state_points_per_day_implicit": 145,
        "feb1_soc_matches_selected_jan31_end_error_kwh": abs(
            float(main_daily.iloc[0]["soc_start_kwh"]) - float(jan.iloc[-1]["soc_end_kwh"])
        ),
        "main_cross_day_soc_max_error_kwh": float(main_cross),
        "paired_cross_day_soc_max_difference_kwh_not_applicable": float(pair_cross),
        "paired_cross_day_soc_applicable": False,
        "paired_daily_start_soc_matches_main_max_error_kwh": float(paired_start_matches_main),
        "greedy_cross_day_soc_max_error_kwh": float(greedy_cross),
        "main_paired_daily_end_soc_max_difference_kwh_expected_executor_effect": float(state_equiv),
        "paired_is_closed_loop_equivalent_by_state_induction": False,
        "main_max_energy_balance_error_kwh": float(balance),
        "paired_max_energy_balance_error_kwh": float(pair_balance),
        "greedy_max_energy_balance_error_kwh": float(
            np.max(np.abs(greedy_slots["balance_residual_kwh"].to_numpy(float)))
        ),
        "main_soc_transition_max_error_kwh": transition_error(main_slots),
        "greedy_soc_transition_max_error_kwh": transition_error(greedy_slots),
        "paired_soc_transition_max_error_kwh": transition_error(paired_slots),
        "main_slot_bill_identity_max_error_yuan": bill_error(main_slots),
        "greedy_slot_bill_identity_max_error_yuan": bill_error(greedy_slots),
        "paired_slot_bill_identity_max_error_yuan": bill_error(paired_slots),
        "main_daily_total_vs_slot_sum_max_error_yuan": daily_sum_error(
            main_daily, main_slots, "actual_total_cost_yuan", "actual_cost_yuan"
        ),
        "greedy_daily_total_vs_slot_sum_max_error_yuan": daily_sum_error(
            greedy_daily, greedy_slots, "actual_total_cost_yuan", "actual_cost_yuan"
        ),
        "paired_daily_total_vs_slot_sum_max_error_yuan": daily_sum_error(
            paired_daily, paired_slots, "actual_total_cost_yuan", "actual_cost_yuan"
        ),
        "soc_min_observed_kwh": float(soc_min),
        "soc_max_observed_kwh": float(soc_max),
        "soc_boundary_violations": int((soc_min < SOC_MIN_KWH - TOL) or (soc_max > SOC_MAX_KWH + TOL)),
        "power_limit_violations": power_viol,
        "simultaneous_charge_discharge_violations": overlap,
        "emergency_purchase_while_charging_violations": emergency_charge,
        "plan_locked_slotwise_max_error_kwh": float(np.max(np.abs(plan_repeat - slot_plan))),
        "greedy_plan_locked_slotwise_max_error_kwh": float(
            np.max(np.abs(greedy_plan_repeat - greedy_slot_plan))
        ),
        "information_prediction_source_violations": int((~info["all_prediction_sources_prior"].astype(bool)).sum()),
        "information_scenario_source_violations": int((~info["all_scenario_sources_prior"].astype(bool)).sum()),
        "greedy_information_prediction_source_violations": int(
            (~greedy_info["all_prediction_sources_prior"].astype(bool)).sum()
        ),
        "greedy_information_scenario_source_violations": int(
            (~greedy_info["all_scenario_sources_prior"].astype(bool)).sum()
        ),
        "solver_days_each_closed_loop": int(len(solver)),
        "solver_days_with_usable_incumbent": int(solver["incumbent_objective_yuan"].notna().sum()),
        "greedy_solver_days_with_usable_incumbent": int(
            greedy_solver["incumbent_objective_yuan"].notna().sum()
        ),
        "protected_file_change_count": len(changed),
        "protected_file_changes": changed,
        "result2_v2_created": bool((ROOT / "outputs/q2_v2/result2.xlsx").exists()),
    }
    checks["all_required_checks_passed"] = bool(
        checks["formal_date_count"] == 334 and checks["formal_dates_exact"] and
        checks["paired_dates_exact"] and checks["greedy_dates_exact"] and
        checks["main_slot_rows"] == 334 * 144 and checks["greedy_slot_rows"] == 334 * 144 and
        checks["paired_slot_rows"] == 334 * 144 and
        checks["each_day_144_slots"] and checks["feb1_soc_matches_selected_jan31_end_error_kwh"] <= TOL and
        checks["main_cross_day_soc_max_error_kwh"] <= TOL and
        checks["greedy_cross_day_soc_max_error_kwh"] <= TOL and
        checks["paired_daily_start_soc_matches_main_max_error_kwh"] <= TOL and
        checks["main_max_energy_balance_error_kwh"] <= TOL and
        checks["paired_max_energy_balance_error_kwh"] <= TOL and
        checks["greedy_max_energy_balance_error_kwh"] <= TOL and
        checks["main_soc_transition_max_error_kwh"] <= TOL and
        checks["greedy_soc_transition_max_error_kwh"] <= TOL and
        checks["paired_soc_transition_max_error_kwh"] <= TOL and
        checks["main_slot_bill_identity_max_error_yuan"] <= TOL and
        checks["greedy_slot_bill_identity_max_error_yuan"] <= TOL and
        checks["paired_slot_bill_identity_max_error_yuan"] <= TOL and
        checks["main_daily_total_vs_slot_sum_max_error_yuan"] <= TOL and
        checks["greedy_daily_total_vs_slot_sum_max_error_yuan"] <= TOL and
        checks["paired_daily_total_vs_slot_sum_max_error_yuan"] <= TOL and
        checks["soc_boundary_violations"] == 0 and checks["power_limit_violations"] == 0 and
        checks["simultaneous_charge_discharge_violations"] == 0 and
        checks["emergency_purchase_while_charging_violations"] == 0 and
        checks["plan_locked_slotwise_max_error_kwh"] <= TOL and
        checks["greedy_plan_locked_slotwise_max_error_kwh"] <= TOL and
        checks["information_prediction_source_violations"] == 0 and
        checks["information_scenario_source_violations"] == 0 and
        checks["greedy_information_prediction_source_violations"] == 0 and
        checks["greedy_information_scenario_source_violations"] == 0 and
        checks["solver_days_with_usable_incumbent"] == 334 and
        checks["greedy_solver_days_with_usable_incumbent"] == 334 and
        checks["protected_file_change_count"] == 0 and not checks["result2_v2_created"]
    )
    return checks


def run() -> None:
    state = json.loads((MAIN / "state.json").read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError(f"Main Stage-4 run is not complete: {state}")
    greedy_state = json.loads((GREEDY / "state.json").read_text(encoding="utf-8"))
    if greedy_state.get("status") != "complete":
        raise RuntimeError(f"Greedy Stage-4 run is not complete: {greedy_state}")
    main_daily = _read_days(MAIN / "daily")
    main_slots = _read_days(MAIN / "slots")
    greedy_daily = _read_days(GREEDY / "daily")
    greedy_slots = _read_days(GREEDY / "slots")
    paired_daily = _read_days(MAIN / "paired_daily")
    paired_slots = _read_days(MAIN / "paired_slots")
    plans = _read_days(MAIN / "plans")
    greedy_plans = _read_days(GREEDY / "plans")
    solver = _read_days(MAIN / "solver")
    greedy_solver = _read_days(GREEDY / "solver")
    info = _read_days(MAIN / "information")
    greedy_info = _read_days(GREEDY / "information")
    main_daily["strategy_type"] = "independent_closed_loop"
    greedy_daily["strategy_type"] = "independent_closed_loop"
    paired_daily["strategy_type"] = "closed_loop_equivalent_and_paired_ablation"
    main_slots["strategy_type"] = "independent_closed_loop"
    greedy_slots["strategy_type"] = "independent_closed_loop"
    paired_slots["strategy_type"] = "closed_loop_equivalent_and_paired_ablation"
    main_summary = _summary("joint_scenario_milp_dp50", main_daily, main_slots, solver)
    greedy_summary = _summary("joint_scenario_milp_greedy", greedy_daily, greedy_slots, greedy_solver)
    comparison = pd.DataFrame([main_summary, greedy_summary])
    comparison.to_csv(STAGE4 / "q2_v2_full_year_strategy_comparison.csv", index=False)
    pd.concat([_monthly(main_summary["policy_id"], main_daily),
               _monthly(greedy_summary["policy_id"], greedy_daily)], ignore_index=True).to_csv(
        STAGE4 / "q2_v2_monthly_comparison.csv", index=False
    )
    pd.concat([main_daily, greedy_daily], ignore_index=True).to_csv(
        STAGE4 / "q2_v2_daily_results.csv", index=False
    )
    pd.concat([main_slots, greedy_slots], ignore_index=True).to_csv(
        STAGE4 / "q2_v2_slot_results.csv", index=False
    )
    plans["strategy_type"] = "independent_closed_loop"
    greedy_plans["strategy_type"] = "independent_closed_loop"
    pd.concat([plans, greedy_plans], ignore_index=True).to_csv(
        STAGE4 / "q2_v2_day_ahead_plans.csv", index=False
    )
    pd.concat([solver, greedy_solver], ignore_index=True).to_csv(STAGE4 / "q2_v2_solver_log.csv", index=False)
    pd.concat([info, greedy_info], ignore_index=True).to_csv(STAGE4 / "q2_v2_information_boundary.csv", index=False)
    paired_ablation = main_daily[["date", "soc_start_kwh", "soc_end_kwh",
                                  "plan_cost_yuan", "emergency_cost_yuan",
                                  "actual_total_cost_yuan", "emergency_purchase_kwh"]].merge(
        paired_daily[["date", "soc_start_kwh", "soc_end_kwh", "plan_cost_yuan",
                      "emergency_cost_yuan", "actual_total_cost_yuan", "emergency_purchase_kwh"]],
        on="date", suffixes=("_dp", "_greedy"), validate="one_to_one",
    )
    for col in ["emergency_cost_yuan", "actual_total_cost_yuan", "emergency_purchase_kwh", "soc_end_kwh"]:
        paired_ablation[f"dp_minus_greedy_{col}"] = paired_ablation[f"{col}_dp"] - paired_ablation[f"{col}_greedy"]
    paired_ablation.to_csv(STAGE4 / "q2_v2_paired_executor_ablation.csv", index=False)
    paired_slots.to_csv(STAGE4 / "q2_v2_paired_executor_slot_results.csv", index=False)
    pd.concat([_price_band(main_slots, main_summary["policy_id"]),
               _price_band(greedy_slots, greedy_summary["policy_id"])], ignore_index=True).to_csv(
        STAGE4 / "q2_v2_emergency_by_price_band.csv", index=False
    )
    checks = _validate(main_daily, main_slots, greedy_daily, greedy_slots,
                       paired_daily, paired_slots, plans, greedy_plans,
                       solver, greedy_solver, info, greedy_info)
    (STAGE4 / "q2_v2_full_year_validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["Q2 V2 Stage 4 full-year shadow validation", "=" * 48]
    lines += [f"{k}: {v}" for k, v in checks.items()]
    (STAGE4 / "q2_v2_full_year_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
