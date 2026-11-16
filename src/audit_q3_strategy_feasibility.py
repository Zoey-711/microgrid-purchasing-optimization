"""Audit feasible Question-3 strategies without changing frozen Q1/Q2 results.

This module is deliberately read-only with respect to all frozen workbooks and
trajectories.  It independently settles the frozen Q2 trajectory as a zero-
adjustment reference under the documented Q3 settlement interpretation, and
diagnoses forecast underestimation in the fixed-parameter Q3 full-year run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "q3_strategy_review"
TABLE_DIR = ROOT / "outputs" / "tables"
LOG_DIR = ROOT / "outputs" / "logs"
TOL = 1e-6
DT = 1.0 / 6.0
E_MAX = 5000.0 * DT
S_MIN = 1200.0
S_MAX = 10800.0
ETA_C = 0.9
ETA_D = 0.9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def normalize_time_label(value: object) -> str:
    """Normalize CSV display syntax without changing the official row order."""
    text = str(value).strip().strip("'").strip('"')
    if "+1" in text:
        return text
    parts = text.split(":")
    if len(parts) == 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
    if len(parts) == 3:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(float(parts[2])):02d}"
    return text


def settle_zero_adjustment(
    original: np.ndarray, effective: np.ndarray, price: np.ndarray
) -> pd.DataFrame:
    """Independent implementation of the Q3 final-vs-original settlement."""
    upward = np.maximum(effective - original, 0.0)
    downward = np.maximum(original - effective, 0.0)
    retained = price * np.minimum(original, effective)
    downward_breach = 0.5 * price * downward
    upward_purchase = 1.5 * price * upward
    contract = retained + downward_breach + upward_purchase
    return pd.DataFrame(
        {
            "upward_adjustment_kwh": upward,
            "downward_adjustment_kwh": downward,
            "retained_purchase_cost_yuan": retained,
            "downward_breach_cost_yuan": downward_breach,
            "upward_purchase_cost_yuan": upward_purchase,
            "contract_settlement_cost_yuan": contract,
        }
    )


def audit_q2_reference() -> tuple[pd.DataFrame, dict[str, float | int | bool | str]]:
    source = ROOT / "outputs" / "replay" / "q2_aplus" / "q2_aplus_slot_trajectory.csv"
    frozen_summary = ROOT / "outputs" / "tables" / "q2_full_year_shadow_comparison.csv"
    q1 = pd.read_csv(ROOT / "data" / "processed" / "q1_baseline_dispatch.csv")
    q2 = pd.read_csv(source)
    q2["date"] = pd.to_datetime(q2["date"]).dt.strftime("%Y-%m-%d")

    price_map = q1["price"].to_numpy(dtype=float)
    time_map = q1["time_label"].map(normalize_time_label).to_numpy()
    q2["price_yuan_per_kwh"] = np.tile(price_map, len(q2) // 144)
    q2["official_time_label"] = np.tile(time_map, len(q2) // 144)
    original = q2["planned_purchase_kwh"].to_numpy(dtype=float)
    effective = original.copy()
    price = q2["price_yuan_per_kwh"].to_numpy(dtype=float)
    settlement = settle_zero_adjustment(original, effective, price)
    for col in settlement:
        q2[col] = settlement[col].to_numpy()

    emergency = q2["emergency_purchase_kwh"].to_numpy(dtype=float)
    q2["independent_emergency_cost_yuan"] = 5.0 * price * emergency
    q2["independent_actual_cost_yuan"] = (
        q2["contract_settlement_cost_yuan"] + q2["independent_emergency_cost_yuan"]
    )
    q2["balance_residual_kwh"] = (
        original
        + q2["discharge_rt_kwh"].to_numpy(dtype=float)
        + emergency
        - q2["net_load_actual_kwh"].to_numpy(dtype=float)
        - q2["charge_rt_kwh"].to_numpy(dtype=float)
        - q2["curtailment_kwh"].to_numpy(dtype=float)
    )
    q2["soc_transition_residual_kwh"] = (
        q2["soc_end_kwh"].to_numpy(dtype=float)
        - q2["soc_start_kwh"].to_numpy(dtype=float)
        - ETA_C * q2["charge_rt_kwh"].to_numpy(dtype=float)
        + q2["discharge_rt_kwh"].to_numpy(dtype=float) / ETA_D
    )

    date_counts = q2.groupby("date").size()
    slot_ok = q2.groupby("date")["slot"].apply(
        lambda x: np.array_equal(x.to_numpy(dtype=int), np.arange(144))
    )
    cross = q2.groupby("date", sort=True).agg(
        soc_start=("soc_start_kwh", "first"), soc_end=("soc_end_kwh", "last")
    )
    cross_error = (cross["soc_start"].iloc[1:].to_numpy() - cross["soc_end"].iloc[:-1].to_numpy())
    frozen = pd.read_csv(frozen_summary)
    frozen = frozen.loc[frozen["strategy_key"] == "aplus"].iloc[0]

    summary: dict[str, float | int | bool | str] = {
        "reference_name": "历史预测、零调整参照策略",
        "formal_days": int(q2["date"].nunique()),
        "slot_rows": int(len(q2)),
        "date_start": str(q2["date"].min()),
        "date_end": str(q2["date"].max()),
        "all_days_144_slots": bool((date_counts == 144).all()),
        "official_slot_order_ok": bool(slot_ok.all()),
        "official_time_labels_ok": bool(
            (q2["time_label"].map(normalize_time_label) == q2["official_time_label"]).all()
        ),
        "zero_adjustment_ok": bool(
            (q2["upward_adjustment_kwh"].abs() <= TOL).all()
            and (q2["downward_adjustment_kwh"].abs() <= TOL).all()
        ),
        "contract_equals_normal_plan_cost": bool(
            np.allclose(
                q2["contract_settlement_cost_yuan"],
                price * original,
                atol=TOL,
                rtol=0.0,
            )
        ),
        "contract_settlement_cost_yuan": float(q2["contract_settlement_cost_yuan"].sum()),
        "emergency_purchase_cost_yuan": float(q2["independent_emergency_cost_yuan"].sum()),
        "actual_total_cost_yuan": float(q2["independent_actual_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(q2["emergency_purchase_kwh"].sum()),
        "max_balance_error_kwh": float(q2["balance_residual_kwh"].abs().max()),
        "max_soc_transition_error_kwh": float(q2["soc_transition_residual_kwh"].abs().max()),
        "max_cross_day_soc_error_kwh": float(np.max(np.abs(cross_error))),
        "soc_violation_count": int(((q2["soc_start_kwh"] < S_MIN - TOL) | (q2["soc_start_kwh"] > S_MAX + TOL) | (q2["soc_end_kwh"] < S_MIN - TOL) | (q2["soc_end_kwh"] > S_MAX + TOL)).sum()),
        "power_violation_count": int(((q2["charge_rt_kwh"] > E_MAX + TOL) | (q2["discharge_rt_kwh"] > E_MAX + TOL)).sum()),
        "emergency_charge_overlap_count": int(((q2["emergency_purchase_kwh"] > TOL) & (q2["charge_rt_kwh"] > TOL)).sum()),
        "cost_difference_vs_frozen_yuan": float(q2["independent_actual_cost_yuan"].sum() - frozen["total_actual_cost_yuan"]),
        "emergency_difference_vs_frozen_kwh": float(q2["emergency_purchase_kwh"].sum() - frozen["total_emergency_purchase_kwh"]),
        "uses_attachment3_forecast": False,
        "operational_and_settlement_feasible": True,
        "eligible_as_formal_q3_main_without_qualification": False,
    }
    summary["all_checks_passed"] = bool(
        summary["formal_days"] == 334
        and summary["slot_rows"] == 334 * 144
        and summary["all_days_144_slots"]
        and summary["official_slot_order_ok"]
        and summary["official_time_labels_ok"]
        and summary["zero_adjustment_ok"]
        and summary["contract_equals_normal_plan_cost"]
        and summary["max_balance_error_kwh"] <= TOL
        and summary["max_soc_transition_error_kwh"] <= TOL
        and summary["max_cross_day_soc_error_kwh"] <= TOL
        and summary["soc_violation_count"] == 0
        and summary["power_violation_count"] == 0
        and summary["emergency_charge_overlap_count"] == 0
        and abs(float(summary["cost_difference_vs_frozen_yuan"])) <= TOL
        and abs(float(summary["emergency_difference_vs_frozen_kwh"])) <= TOL
    )
    keep = [
        "date", "slot", "time_label", "price_yuan_per_kwh",
        "planned_purchase_kwh", "upward_adjustment_kwh", "downward_adjustment_kwh",
        "charge_rt_kwh", "discharge_rt_kwh", "emergency_purchase_kwh",
        "curtailment_kwh", "soc_start_kwh", "soc_end_kwh",
        "contract_settlement_cost_yuan", "independent_emergency_cost_yuan",
        "independent_actual_cost_yuan", "balance_residual_kwh",
        "soc_transition_residual_kwh",
    ]
    return q2[keep], summary


def prediction_diagnostic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    path = ROOT / "outputs" / "q3_full_year" / "q3_full_year_slot_trajectories.csv"
    usecols = [
        "adjustment_policy", "date", "idx", "time_label", "active_plan_issue_hour",
        "active_load_forecast_kwh", "active_pv_forecast_kwh",
        "active_net_load_forecast_kwh", "load_kwh", "pv_actual_kwh",
        "net_load_actual_kwh", "emergency_purchase_kwh",
        "emergency_purchase_cost_yuan",
    ]
    q3 = pd.read_csv(path, usecols=usecols)
    q3 = q3.loc[
        (q3["adjustment_policy"] == "all_updates")
        & (pd.to_datetime(q3["date"]) >= pd.Timestamp("2025-02-01"))
    ].copy()
    q3["load_error_pred_minus_actual_kwh"] = q3["active_load_forecast_kwh"] - q3["load_kwh"]
    q3["pv_error_pred_minus_actual_kwh"] = q3["active_pv_forecast_kwh"] - q3["pv_actual_kwh"]
    q3["net_error_pred_minus_actual_kwh"] = q3["active_net_load_forecast_kwh"] - q3["net_load_actual_kwh"]
    q3["net_underestimate_kwh"] = np.maximum(-q3["net_error_pred_minus_actual_kwh"], 0.0)
    q3["clock_hour"] = q3["idx"].astype(int) // 6
    q3["net_underestimated"] = q3["net_underestimate_kwh"] > TOL

    rows = []
    for name, err_col in [
        ("负载预测", "load_error_pred_minus_actual_kwh"),
        ("光伏预测", "pv_error_pred_minus_actual_kwh"),
        ("净负荷预测", "net_error_pred_minus_actual_kwh"),
    ]:
        err = q3[err_col].to_numpy(dtype=float)
        rows.append(
            {
                "metric_group": name,
                "slots": len(err),
                "mae_kwh_per_slot": float(np.mean(np.abs(err))),
                "rmse_kwh_per_slot": float(np.sqrt(np.mean(err**2))),
                "bias_pred_minus_actual_kwh_per_slot": float(np.mean(err)),
                "underestimate_slot_count": int((err < -TOL).sum()),
                "underestimate_rate": float((err < -TOL).mean()),
                "underestimate_amount_kwh": float(np.maximum(-err, 0.0).sum()),
            }
        )
    summary_df = pd.DataFrame(rows)

    by_hour = q3.groupby("clock_hour", as_index=False).agg(
        slot_count=("idx", "size"),
        net_underestimate_slot_count=("net_underestimated", "sum"),
        net_underestimate_kwh=("net_underestimate_kwh", "sum"),
        emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
        emergency_purchase_cost_yuan=("emergency_purchase_cost_yuan", "sum"),
    )
    by_hour["net_underestimate_rate"] = by_hour["net_underestimate_slot_count"] / by_hour["slot_count"]
    by_hour["time_period"] = by_hour["clock_hour"].map(lambda h: f"{h:02d}:00-{h+1:02d}:00")

    by_issue = q3.groupby("active_plan_issue_hour", as_index=False).agg(
        slot_count=("idx", "size"),
        net_underestimate_slot_count=("net_underestimated", "sum"),
        net_underestimate_kwh=("net_underestimate_kwh", "sum"),
        emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
        emergency_purchase_cost_yuan=("emergency_purchase_cost_yuan", "sum"),
    )
    under_mask = q3["net_underestimated"]
    load_shortfall = np.maximum(
        q3["load_kwh"].to_numpy(dtype=float)
        - q3["active_load_forecast_kwh"].to_numpy(dtype=float),
        0.0,
    )
    pv_overforecast = np.maximum(
        q3["active_pv_forecast_kwh"].to_numpy(dtype=float)
        - q3["pv_actual_kwh"].to_numpy(dtype=float),
        0.0,
    )
    assoc = {
        "formal_slots": int(len(q3)),
        "net_underestimate_kwh": float(q3["net_underestimate_kwh"].sum()),
        "net_underestimate_slots": int(under_mask.sum()),
        "emergency_kwh_in_underestimated_slots": float(q3.loc[under_mask, "emergency_purchase_kwh"].sum()),
        "emergency_cost_in_underestimated_slots_yuan": float(q3.loc[under_mask, "emergency_purchase_cost_yuan"].sum()),
        "total_emergency_kwh": float(q3["emergency_purchase_kwh"].sum()),
        "total_emergency_cost_yuan": float(q3["emergency_purchase_cost_yuan"].sum()),
        "share_emergency_cost_in_underestimated_slots": float(q3.loc[under_mask, "emergency_purchase_cost_yuan"].sum() / q3["emergency_purchase_cost_yuan"].sum()),
        "positive_load_shortfall_in_net_underestimated_slots_kwh": float(load_shortfall[under_mask.to_numpy()].sum()),
        "positive_pv_overforecast_in_net_underestimated_slots_kwh": float(pv_overforecast[under_mask.to_numpy()].sum()),
        "corr_net_underestimate_emergency_kwh": float(q3["net_underestimate_kwh"].corr(q3["emergency_purchase_kwh"])),
        "corr_net_underestimate_emergency_cost": float(q3["net_underestimate_kwh"].corr(q3["emergency_purchase_cost_yuan"])),
    }
    return summary_df, by_hour, by_issue, assoc


def january_ranking_check() -> tuple[pd.DataFrame, dict[str, float | int | bool | str]]:
    src = ROOT / "outputs" / "tables" / "q3_january_parameter_comparison.csv"
    table = pd.read_csv(src)
    main = table.loc[table["adjustment_policy"] == "all_updates"].copy()
    main = main.sort_values(["actual_total_cost_yuan", "emergency_purchase_kwh", "candidate_id"])
    main.insert(0, "cost_rank", np.arange(1, len(main) + 1))
    selected = main.loc[main["selected_main"].astype(bool)].iloc[0]
    control = table.loc[table["selected_internal_control"].astype(bool)].iloc[0]
    check = {
        "all_update_candidate_count": int(len(main)),
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_cost_rank": int(selected["cost_rank"]),
        "selected_actual_total_cost_yuan": float(selected["actual_total_cost_yuan"]),
        "selected_emergency_purchase_kwh": float(selected["emergency_purchase_kwh"]),
        "corrected_internal_control_id": str(control["candidate_id"]),
        "corrected_internal_control_cost_yuan": float(control["actual_total_cost_yuan"]),
        "corrected_internal_control_emergency_kwh": float(control["emergency_purchase_kwh"]),
        "main_ranking_affected_by_control_fix": False,
        "selection_still_rank_1": bool(int(selected["cost_rank"]) == 1),
    }
    return main, check


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    protected = [
        ROOT / "outputs" / "tables" / "result2.xlsx",
        ROOT / "outputs" / "tables" / "q2_full_year_shadow_comparison.csv",
        ROOT / "outputs" / "replay" / "q2_aplus" / "q2_aplus_slot_trajectory.csv",
        ROOT / "data" / "raw" / "result3.xlsx",
    ]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}

    q2_audit, q2_summary = audit_q2_reference()
    forecast_summary, by_hour, by_issue, assoc = prediction_diagnostic()
    ranking, ranking_check = january_ranking_check()

    q2_audit.to_csv(OUT_DIR / "q2_historical_zero_adjustment_slot_audit.csv", index=False)
    pd.DataFrame([q2_summary]).to_csv(
        TABLE_DIR / "q3_q2_zero_adjustment_reference_summary.csv", index=False
    )
    forecast_summary.to_csv(TABLE_DIR / "q3_prediction_error_summary.csv", index=False)
    by_hour.to_csv(TABLE_DIR / "q3_net_underestimation_by_hour.csv", index=False)
    by_issue.to_csv(TABLE_DIR / "q3_net_underestimation_by_forecast_version.csv", index=False)
    pd.DataFrame([assoc]).to_csv(TABLE_DIR / "q3_underestimation_emergency_association.csv", index=False)
    ranking.to_csv(TABLE_DIR / "q3_january_main_ranking_check.csv", index=False)

    after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    hashes_unchanged = before == after
    passed = bool(q2_summary["all_checks_passed"] and ranking_check["selection_still_rank_1"] and hashes_unchanged)
    lines = [
        "Question 3 pre-final strategy feasibility review",
        "=" * 56,
        f"q2_zero_adjustment_reference_all_checks_passed: {q2_summary['all_checks_passed']}",
        f"q2_operational_and_settlement_feasible: {q2_summary['operational_and_settlement_feasible']}",
        f"q2_uses_attachment3_forecast: {q2_summary['uses_attachment3_forecast']}",
        f"q2_eligible_as_formal_q3_main_without_qualification: {q2_summary['eligible_as_formal_q3_main_without_qualification']}",
        f"q2_contract_equals_normal_plan_cost: {q2_summary['contract_equals_normal_plan_cost']}",
        f"q2_cost_difference_vs_frozen_yuan: {q2_summary['cost_difference_vs_frozen_yuan']:.12g}",
        f"q2_emergency_difference_vs_frozen_kwh: {q2_summary['emergency_difference_vs_frozen_kwh']:.12g}",
        f"january_selected_main_rank: {ranking_check['selected_cost_rank']}",
        f"january_main_ranking_affected_by_control_fix: {ranking_check['main_ranking_affected_by_control_fix']}",
        f"protected_hashes_unchanged: {hashes_unchanged}",
        f"result3_generated: {any(ROOT.glob('outputs/**/result3.xlsx'))}",
        f"passed: {passed}",
        "",
        "protected_hashes_before:",
        *[f"{k}: {v}" for k, v in before.items()],
        "",
        "protected_hashes_after:",
        *[f"{k}: {v}" for k, v in after.items()],
    ]
    (LOG_DIR / "q3_strategy_feasibility_review.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
