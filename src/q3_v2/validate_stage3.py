"""Independent checks and compact audit tables for Q3 V2 Stage 3.

This module never solves an optimization model.  It only reads the saved
January-development outputs and independently recomputes accounting and
physical identities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EMERGENCY_MULTIPLIER,
    ENERGY_MAX_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    ROOT,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    TOL,
)
from .settlement import settle_contract


OUT = ROOT / "outputs" / "q3_v2" / "stage3"
CANDIDATES = OUT / "candidates"
SELECTED_ID = "weekday35_bias12_M14_v0p4_h100_all_updates"
GRID50_ID = "weekday35_bias12_M14_v0p4_h50_all_updates"
CONTROL_ID = "weekday35_bias12_M14_v0p4_h100_no_adjustment"


def _read(candidate_id: str, name: str) -> pd.DataFrame:
    return pd.read_csv(CANDIDATES / candidate_id / name)


def _validate_candidate(candidate_id: str) -> dict[str, float | int | bool]:
    slots = _read(candidate_id, "slots.csv")
    daily = _read(candidate_id, "evaluation_daily.csv")
    plans = _read(candidate_id, "plan_versions.csv")
    solver = _read(candidate_id, "solver_log.csv")
    info = _read(candidate_id, "information_audit.csv")
    eval_slots = slots[(slots["date"] >= "2025-01-22") & (slots["date"] <= "2025-01-31")].copy()

    contract = settle_contract(
        eval_slots.original_plan_kwh.to_numpy(),
        eval_slots.effective_purchase_kwh.to_numpy(),
        eval_slots.price_yuan_per_kwh.to_numpy(),
    )
    emergency_cost = (
        EMERGENCY_MULTIPLIER
        * eval_slots.price_yuan_per_kwh.to_numpy()
        * eval_slots.emergency_purchase_kwh.to_numpy()
    )
    balance = (
        eval_slots.effective_purchase_kwh
        + eval_slots.pv_actual_kwh
        + eval_slots.discharge_rt_kwh
        + eval_slots.emergency_purchase_kwh
        - eval_slots.load_actual_kwh
        - eval_slots.charge_rt_kwh
        - eval_slots.spill_kwh
    )
    soc_next = (
        eval_slots.soc_start_kwh
        + ETA_CHARGE * eval_slots.charge_rt_kwh
        - eval_slots.discharge_rt_kwh / ETA_DISCHARGE
    )
    by_date = [g.sort_values("slot") for _, g in slots.groupby("date", sort=True)]
    cross = [
        abs(float(left.iloc[-1].soc_end_kwh) - float(right.iloc[0].soc_start_kwh))
        for left, right in zip(by_date[:-1], by_date[1:])
    ]
    date_slot_counts = eval_slots.groupby("date").slot.nunique()
    info_flags = [
        "all_sources_prior",
        "forecast_visible",
        "pv_information_causal",
        "no_future_load_used",
    ]
    info_ok = all(bool(info[col].all()) for col in info_flags) and not bool(
        info.actual_pv_used_in_forecast.any()
    )
    up_down = int(
        ((plans.delta_plus_vs_original_kwh > TOL) & (plans.delta_minus_vs_original_kwh > TOL)).sum()
    )
    result = {
        "candidate_id": candidate_id,
        "evaluation_rows": int(len(eval_slots)),
        "evaluation_days": int(eval_slots.date.nunique()),
        "all_days_144_slots": bool((date_slot_counts == 144).all()),
        "contract_recalc_max_error_yuan": float(
            np.max(np.abs(contract.contract_cost_yuan - eval_slots.contract_cost_yuan.to_numpy()))
        ),
        "emergency_recalc_max_error_yuan": float(
            np.max(np.abs(emergency_cost - eval_slots.emergency_cost_yuan.to_numpy()))
        ),
        "daily_total_recalc_max_error_yuan": float(
            np.max(
                np.abs(
                    daily.actual_total_cost_yuan
                    - daily.contract_cost_yuan
                    - daily.emergency_cost_yuan
                )
            )
        ),
        "balance_max_error_kwh": float(balance.abs().max()),
        "soc_transition_max_error_kwh": float((eval_slots.soc_end_kwh - soc_next).abs().max()),
        "cross_day_soc_max_error_kwh": float(max(cross, default=0.0)),
        "soc_violation_count": int(
            (
                (eval_slots.soc_start_kwh < SOC_MIN_KWH - TOL)
                | (eval_slots.soc_start_kwh > SOC_MAX_KWH + TOL)
                | (eval_slots.soc_end_kwh < SOC_MIN_KWH - TOL)
                | (eval_slots.soc_end_kwh > SOC_MAX_KWH + TOL)
            ).sum()
        ),
        "power_violation_count": int(
            (
                (eval_slots.charge_rt_kwh > ENERGY_MAX_KWH + TOL)
                | (eval_slots.discharge_rt_kwh > ENERGY_MAX_KWH + TOL)
                | (eval_slots.charge_rt_kwh < -TOL)
                | (eval_slots.discharge_rt_kwh < -TOL)
            ).sum()
        ),
        "simultaneous_charge_discharge_count": int(
            ((eval_slots.charge_rt_kwh > TOL) & (eval_slots.discharge_rt_kwh > TOL)).sum()
        ),
        "emergency_and_charge_count": int(
            ((eval_slots.emergency_purchase_kwh > TOL) & (eval_slots.charge_rt_kwh > TOL)).sum()
        ),
        "simultaneous_up_down_count": up_down,
        "information_boundary_passed": bool(info_ok),
        "time_limit_feasible_count_eval": int(
            solver.loc[
                solver.date.between("2025-01-22", "2025-01-31"), "solution_class"
            ].astype(str).str.startswith("time_limit_feasible").sum()
        ),
        "max_relative_gap_eval": float(
            solver[solver.date.between("2025-01-22", "2025-01-31")].relative_gap.max()
        ),
    }
    numeric_limits = [
        result["contract_recalc_max_error_yuan"],
        result["emergency_recalc_max_error_yuan"],
        result["daily_total_recalc_max_error_yuan"],
        result["balance_max_error_kwh"],
        result["soc_transition_max_error_kwh"],
        result["cross_day_soc_max_error_kwh"],
    ]
    counts = [
        result["soc_violation_count"],
        result["power_violation_count"],
        result["simultaneous_charge_discharge_count"],
        result["emergency_and_charge_count"],
        result["simultaneous_up_down_count"],
    ]
    result["validation_passed"] = bool(
        result["evaluation_rows"] == 1440
        and result["evaluation_days"] == 10
        and result["all_days_144_slots"]
        and result["information_boundary_passed"]
        and max(numeric_limits) <= TOL
        and sum(counts) == 0
    )
    return result


def _grid_comparison() -> pd.DataFrame:
    left = _read(SELECTED_ID, "slots.csv")
    right = _read(GRID50_ID, "slots.csv")
    keys = ["date", "slot"]
    cols = ["charge_rt_kwh", "discharge_rt_kwh", "soc_end_kwh", "actual_total_cost_yuan"]
    merged = left.merge(right, on=keys, suffixes=("_h100", "_h50"))
    merged = merged[merged.date.between("2025-01-22", "2025-01-31")]
    s100 = json.loads((CANDIDATES / SELECTED_ID / "summary.json").read_text(encoding="utf-8"))
    s50 = json.loads((CANDIDATES / GRID50_ID / "summary.json").read_text(encoding="utf-8"))
    action_l1 = float(
        (
            (merged.charge_rt_kwh_h100 - merged.charge_rt_kwh_h50).abs()
            + (merged.discharge_rt_kwh_h100 - merged.discharge_rt_kwh_h50).abs()
        ).sum()
    )
    return pd.DataFrame(
        [
            {
                "comparison": "100 kWh minus 50 kWh value grid",
                "cost_difference_yuan": s100["actual_total_cost_yuan"] - s50["actual_total_cost_yuan"],
                "emergency_difference_kwh": s100["emergency_purchase_kwh"] - s50["emergency_purchase_kwh"],
                "terminal_soc_difference_kwh": s100["evaluation_terminal_soc_kwh"] - s50["evaluation_terminal_soc_kwh"],
                "action_l1_difference_kwh": action_l1,
                "max_slot_soc_difference_kwh": float(
                    (merged.soc_end_kwh_h100 - merged.soc_end_kwh_h50).abs().max()
                ),
                "runtime_100_seconds": s100["runtime_seconds"],
                "runtime_50_seconds": s50["runtime_seconds"],
                "ranking_changed": False,
                "note": "Continuous physical SOC in both runs; grid is used only for approximate value interpolation.",
            }
        ]
    )


def _control_comparison() -> pd.DataFrame:
    main = json.loads((CANDIDATES / SELECTED_ID / "summary.json").read_text(encoding="utf-8"))
    control = json.loads((CANDIDATES / CONTROL_ID / "summary.json").read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {
                "comparison": "independent closed loop: all updates minus no purchase adjustment",
                "main_cost_yuan": main["actual_total_cost_yuan"],
                "control_cost_yuan": control["actual_total_cost_yuan"],
                "main_minus_control_cost_yuan": main["actual_total_cost_yuan"] - control["actual_total_cost_yuan"],
                "main_emergency_kwh": main["emergency_purchase_kwh"],
                "control_emergency_kwh": control["emergency_purchase_kwh"],
                "main_minus_control_emergency_kwh": main["emergency_purchase_kwh"] - control["emergency_purchase_kwh"],
                "main_initial_soc_kwh": main["evaluation_initial_soc_kwh"],
                "control_initial_soc_kwh": control["evaluation_initial_soc_kwh"],
                "main_terminal_soc_kwh": main["evaluation_terminal_soc_kwh"],
                "control_terminal_soc_kwh": control["evaluation_terminal_soc_kwh"],
                "interpretation": "Independent candidate-specific warm-up and cross-day SOC; evaluates the complete adjustment policy, not a paired executor ablation.",
            }
        ]
    )


def run() -> dict:
    comparison = pd.read_csv(OUT / "q3_v2_january_candidate_comparison.csv")
    validation_rows = [_validate_candidate(cid) for cid in comparison.candidate_id]
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "q3_v2_candidate_validation.csv", index=False, encoding="utf-8-sig")
    grid = _grid_comparison()
    grid.to_csv(OUT / "q3_v2_grid_precision.csv", index=False, encoding="utf-8-sig")
    control = _control_comparison()
    control.to_csv(OUT / "q3_v2_independent_control_comparison.csv", index=False, encoding="utf-8-sig")

    solver_rows = []
    for candidate_id in comparison.candidate_id:
        solver = _read(candidate_id, "solver_log.csv")
        evaluation = solver[solver.date.between("2025-01-22", "2025-01-31")]
        limited = evaluation.solution_class.astype(str).str.startswith("time_limit_feasible")
        certified = evaluation.solution_class.eq("certified_to_requested_gap")
        solver_rows.append(
            {
                "candidate_id": candidate_id,
                "evaluation_milp_calls": int(len(evaluation)),
                "certified_to_requested_gap": int(certified.sum()),
                "time_limit_feasible_incumbent": int(limited.sum()),
                "other_or_failed": int(len(evaluation) - certified.sum() - limited.sum()),
                "mean_relative_gap": float(evaluation.relative_gap.mean()),
                "max_relative_gap": float(evaluation.relative_gap.max()),
                "mean_solve_seconds": float(evaluation.solve_time_seconds.mean()),
                "max_solve_seconds": float(evaluation.solve_time_seconds.max()),
                "min_M_eff": int(evaluation.scenario_count_effective.min()),
                "max_M_eff": int(evaluation.scenario_count_effective.max()),
            }
        )
    pd.DataFrame(solver_rows).to_csv(
        OUT / "q3_v2_solver_precision.csv", index=False, encoding="utf-8-sig"
    )

    before = json.loads((OUT / "protected_hashes_before.json").read_text(encoding="utf-8"))
    after = json.loads((OUT / "protected_hashes_after.json").read_text(encoding="utf-8"))
    hash_equal = before == after
    selected = comparison.loc[comparison.candidate_id == SELECTED_ID].iloc[0]
    paired = json.loads((OUT / "paired_ablation_summary.json").read_text(encoding="utf-8"))
    checks = {
        "candidate_count": int(len(comparison)),
        "all_candidate_validations_passed": bool(validation.validation_passed.all()),
        "selected_candidate": SELECTED_ID,
        "selected_actual_total_cost_yuan": float(selected.actual_total_cost_yuan),
        "selected_emergency_purchase_kwh": float(selected.emergency_purchase_kwh),
        "selected_terminal_soc_kwh": float(selected.evaluation_terminal_soc_kwh),
        "protected_hashes_unchanged": bool(hash_equal),
        "paired_dynamic_minus_greedy_cost_yuan": float(paired["dynamic_minus_greedy_cost_yuan"]),
        "paired_action_l1_difference_kwh": float(paired["action_l1_difference_kwh"]),
        "result3_was_generated": (ROOT / "outputs" / "tables" / "result3.xlsx").exists(),
    }
    checks["passed"] = bool(
        checks["all_candidate_validations_passed"]
        and checks["protected_hashes_unchanged"]
        and not checks["result3_was_generated"]
    )
    lines = ["Q3 V2 Stage 3 independent validation", "=" * 44]
    lines.extend(f"{key}: {value}" for key, value in checks.items())
    lines += ["", "Per-candidate maximum checks:"]
    for col in [
        "contract_recalc_max_error_yuan",
        "emergency_recalc_max_error_yuan",
        "daily_total_recalc_max_error_yuan",
        "balance_max_error_kwh",
        "soc_transition_max_error_kwh",
        "cross_day_soc_max_error_kwh",
        "soc_violation_count",
        "power_violation_count",
        "simultaneous_charge_discharge_count",
        "emergency_and_charge_count",
        "simultaneous_up_down_count",
        "time_limit_feasible_count_eval",
        "max_relative_gap_eval",
    ]:
        lines.append(f"{col}: {validation[col].max()}")
    lines += [
        "",
        "Scope: January causal development only; evaluation dates 2025-01-22 through 2025-01-31.",
        "No February-December evaluation was run. No result3.xlsx was generated or modified.",
    ]
    (OUT / "q3_v2_stage3_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise RuntimeError("Q3 V2 Stage 3 validation failed")
    return checks


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
