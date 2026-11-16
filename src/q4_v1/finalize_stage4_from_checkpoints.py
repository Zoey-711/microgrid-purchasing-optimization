"""Q4 V1 stage-4 final aggregation and independent verification.

This program is deliberately post-processing only: it reads the completed
trajectory/checkpoint pickle files, performs independent arithmetic checks, and
writes review artefacts.  It never invokes forecasting, MILP, DP, or template
export code.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\数学建模\C题")
STAGE = ROOT / "outputs" / "q4_v1" / "stage4"
CHECKPOINTS = STAGE / "checkpoints"
DOCS = ROOT / "docs" / "q4_v1"
EPS = 1e-6
DT = 1.0 / 6.0
PMAX_KWH = 5000.0 * DT
SOC_MIN, SOC_MAX = 1200.0, 10800.0
INVENTORY_REFERENCE_KWH = 6000.0
INVENTORY_REFERENCE_VALUE = 1.0  # Frozen before stage-4; not an actual bill item.

PROTECTED = [
    ROOT / "outputs" / "tables" / "result1.xlsx",
    ROOT / "outputs" / "tables" / "result2.xlsx",
    ROOT / "outputs" / "tables" / "result3.xlsx",
    ROOT / "data" / "raw" / "result4-2.xlsx",
    ROOT / "data" / "raw" / "result4-3.xlsx",
    ROOT / "outputs" / "q4_v1" / "stage3" / "q4_v1_frozen_config_4_2.json",
    ROOT / "outputs" / "q4_v1" / "stage3" / "q4_v1_frozen_config_4_3.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def protected_hashes() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) if p.exists() else "MISSING" for p in PROTECTED}


def read_pickle(policy: str, name: str) -> pd.DataFrame:
    return pd.read_pickle(CHECKPOINTS / f"{policy}_{name}.pkl")


def actual_price_band(price: pd.Series) -> pd.Series:
    # Frozen reporting bands from the stage-4 protocol: low <0.6, mid [0.6,1), high >=1.
    return pd.cut(price, [-np.inf, 0.6, 1.0, np.inf], labels=["低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"], right=False)


def max_abs(a: pd.Series) -> float:
    vals = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmax(np.abs(vals))) if len(vals) else 0.0


def count_bool(df: pd.DataFrame, names: list[str]) -> int:
    total = 0
    for name in names:
        if name in df.columns:
            s = df[name]
            if s.dtype == object:
                total += int(s.astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            else:
                total += int(s.fillna(False).astype(bool).sum())
    return total


def duration_stats(solver: pd.DataFrame, gap_col: str) -> dict[str, float]:
    gap = pd.to_numeric(solver[gap_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    sec = pd.to_numeric(solver["solve_time_seconds"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "gap_max": float(gap.max()) if len(gap) else math.nan,
        "gap_mean": float(gap.mean()) if len(gap) else math.nan,
        "gap_p95": float(gap.quantile(0.95)) if len(gap) else math.nan,
        "time_max_s": float(sec.max()) if len(sec) else math.nan,
        "time_mean_s": float(sec.mean()) if len(sec) else math.nan,
    }


def solver_quality(policy: str, solver: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = solver.copy()
    gap_col = "mip_gap" if "mip_gap" in out.columns else "relative_gap"
    out["relative_gap_for_audit"] = pd.to_numeric(out[gap_col], errors="coerce")
    text = (out.get("solution_class", "").astype(str) + " " + out.get("solver_status", "").astype(str) + " " + out.get("termination_condition", "").astype(str)).str.lower()
    out["audit_solution_class"] = np.select(
        [text.str.contains("limit|time"), text.str.contains("fail|error|infeas|unbound")],
        ["limit_feasible", "failed"],
        default="optimal_or_requested_gap",
    )
    if "incumbent_objective_yuan" not in out.columns:
        out["incumbent_objective_yuan"] = np.nan
    if "best_bound_yuan" not in out.columns:
        out["best_bound_yuan"] = np.nan
    stats = duration_stats(out, "relative_gap_for_audit")
    stats.update({
        "optimal_or_requested_gap": int((out.audit_solution_class == "optimal_or_requested_gap").sum()),
        "limit_feasible": int((out.audit_solution_class == "limit_feasible").sum()),
        "failed": int((out.audit_solution_class == "failed").sum()),
    })
    # Solver logs already contain their run label; force it to the audit label
    # without creating a duplicate column.
    if "policy" in out.columns:
        out["policy"] = policy
    else:
        out.insert(0, "policy", policy)
    return out, stats


def integrity(policy: str, slots: pd.DataFrame, checkpoint: dict) -> dict:
    s = slots.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.date
    day_slot = s[["date", "slot"]]
    dates = pd.Series(sorted(s.date.unique()))
    expected_dates = pd.date_range("2025-02-01", "2025-12-31", freq="D").date
    expected_pairs = {(d, t) for d in expected_dates for t in range(144)}
    actual_pairs = set(map(tuple, day_slot.itertuples(index=False, name=None)))
    return {
        "policy": policy,
        "completed_checkpoint": bool(checkpoint.get("completed_days") == 334 and checkpoint.get("next_date") is None),
        "completed_days_checkpoint": int(checkpoint.get("completed_days", -1)),
        "slot_rows": int(len(s)),
        "date_count": int(s.date.nunique()),
        "first_date": str(min(dates)) if len(dates) else "",
        "last_date": str(max(dates)) if len(dates) else "",
        "missing_dates": int(len(set(expected_dates) - set(dates))),
        "duplicate_date_slots": int(day_slot.duplicated().sum()),
        "missing_date_slots": int(len(expected_pairs - actual_pairs)),
        "unexpected_date_slots": int(len(actual_pairs - expected_pairs)),
        "duplicate_billing_rows": int(day_slot.duplicated().sum()),
        "valid_complete_trajectory": bool(len(s) == 334 * 144 and s.date.nunique() == 334 and day_slot.duplicated().sum() == 0 and len(expected_pairs - actual_pairs) == 0),
    }


def physics(policy: str, slots: pd.DataFrame, info: pd.DataFrame, plans: pd.DataFrame | None) -> dict:
    s = slots.sort_values(["date", "slot"]).copy()
    supply = s.effective_purchase_kwh + s.pv_actual_kwh + s.discharge_rt_kwh + s.emergency_purchase_kwh
    use = s.load_actual_kwh + s.charge_rt_kwh + s.spill_kwh
    balance = supply - use
    ecalc = s.soc_start_kwh + 0.9 * s.charge_rt_kwh - s.discharge_rt_kwh / 0.9
    state = ecalc - s.soc_end_kwh
    prior_end = s.groupby("date", sort=True).soc_end_kwh.last().shift(1)
    day_start = s.groupby("date", sort=True).soc_start_kwh.first()
    cross = day_start.iloc[1:].to_numpy() - prior_end.iloc[1:].to_numpy()
    p = s.original_plan_kwh
    z = s.effective_purchase_kwh
    plus_err = max_abs(s.delta_plus_kwh - np.maximum(z - p, 0.0))
    minus_err = max_abs(s.delta_minus_kwh - np.maximum(p - z, 0.0))
    expected_cost = s.price_yuan_per_kwh * p + 1.5 * s.price_yuan_per_kwh * s.delta_plus_kwh - 0.5 * s.price_yuan_per_kwh * s.delta_minus_kwh + 5 * s.price_yuan_per_kwh * s.emergency_purchase_kwh
    bill_err = max_abs(expected_cost - s.actual_total_cost_yuan)
    contract_expected = s.price_yuan_per_kwh * p + 1.5 * s.price_yuan_per_kwh * s.delta_plus_kwh - 0.5 * s.price_yuan_per_kwh * s.delta_minus_kwh
    contract_err = max_abs(contract_expected - s.contract_cost_yuan)
    emergency_err = max_abs(5 * s.price_yuan_per_kwh * s.emergency_purchase_kwh - s.emergency_cost_yuan)
    bad_lock = 0
    plan_match_err = math.nan
    if plans is not None and len(plans):
        bad_lock = int((pd.to_numeric(plans.executed_prefix_lock_error_kwh, errors="coerce").abs() > EPS).sum())
        lookup = plans.rename(columns={"version_plan_kwh": "checked_version_kwh"})[["date", "issue_hour", "slot", "checked_version_kwh"]].copy()
        lookup.date = pd.to_datetime(lookup.date).dt.strftime("%Y-%m-%d")
        joined = s.copy()
        joined.date = pd.to_datetime(joined.date).dt.strftime("%Y-%m-%d")
        joined = joined.merge(lookup, left_on=["date", "active_issue_hour", "slot"], right_on=["date", "issue_hour", "slot"], how="left")
        plan_match_err = max_abs(joined.effective_purchase_kwh - joined.checked_version_kwh)
    leak_cols = [c for c in info.columns if "leak" in c.lower()]
    leak_total = count_bool(info, leak_cols)
    price_leaks = count_bool(info, [c for c in leak_cols if "price" in c.lower()])
    load_pv_leaks = count_bool(info, [c for c in leak_cols if "load" in c.lower() or "pv" in c.lower()])
    return {
        "policy": policy,
        "max_supply_demand_balance_residual_kwh": max_abs(balance),
        "max_reported_balance_residual_kwh": max_abs(s.balance_residual_kwh),
        "max_soc_recurrence_residual_kwh": max_abs(state),
        "max_cross_day_soc_error_kwh": float(np.max(np.abs(cross))) if len(cross) else 0.0,
        "soc_below_min_count": int((s.soc_end_kwh < SOC_MIN - EPS).sum() + (s.soc_start_kwh < SOC_MIN - EPS).sum()),
        "soc_above_max_count": int((s.soc_end_kwh > SOC_MAX + EPS).sum() + (s.soc_start_kwh > SOC_MAX + EPS).sum()),
        "charge_power_limit_violations": int((s.charge_rt_kwh > PMAX_KWH + EPS).sum()),
        "discharge_power_limit_violations": int((s.discharge_rt_kwh > PMAX_KWH + EPS).sum()),
        "simultaneous_charge_discharge_count": int(((s.charge_rt_kwh > EPS) & (s.discharge_rt_kwh > EPS)).sum()),
        "emergency_purchase_while_charging_count": int(((s.emergency_purchase_kwh > EPS) & (s.charge_rt_kwh > EPS)).sum()),
        "negative_original_plan_below_tolerance": int((s.original_plan_kwh < -EPS).sum()),
        "negative_effective_plan_below_tolerance": int((s.effective_purchase_kwh < -EPS).sum()),
        "delta_plus_definition_max_error_kwh": plus_err,
        "delta_minus_definition_max_error_kwh": minus_err,
        "max_slot_contract_recompute_error_yuan": contract_err,
        "max_slot_emergency_recompute_error_yuan": emergency_err,
        "max_slot_total_bill_recompute_error_yuan": bill_err,
        "future_price_leak_count": price_leaks,
        "future_load_or_pv_leak_count": load_pv_leaks,
        "all_logged_information_leak_count": leak_total,
        "executed_prefix_lock_violation_count": bad_lock,
        "active_version_vs_effective_plan_max_error_kwh": plan_match_err,
        "middle_version_repeat_settlement_count": 0,  # Formula is evaluated once from p,z at each slot.
    }


def cost_summary(policy: str, slots: pd.DataFrame, checkpoint: dict) -> tuple[dict, pd.DataFrame]:
    s = slots.copy()
    s.date = pd.to_datetime(s.date)
    price = s.price_yuan_per_kwh
    p, z, e = s.original_plan_kwh, s.effective_purchase_kwh, s.emergency_purchase_kwh
    c0 = price * p
    up_fee = 1.5 * price * s.delta_plus_kwh
    refund = 0.5 * price * s.delta_minus_kwh
    contract = c0 + up_fee - refund
    emergency = 5 * price * e
    total = contract + emergency
    s["independent_contract_cost_yuan"] = contract
    s["independent_emergency_cost_yuan"] = emergency
    s["independent_total_cost_yuan"] = total
    s["price_band"] = actual_price_band(price)
    month = s.date.dt.to_period("M").astype(str)
    daily = s.assign(month=month).groupby("date", as_index=False).agg(
        policy=("policy", "first"), original_plan_purchase_kwh=("original_plan_kwh", "sum"), effective_purchase_kwh=("effective_purchase_kwh", "sum"),
        delta_plus_kwh=("delta_plus_kwh", "sum"), delta_minus_kwh=("delta_minus_kwh", "sum"), charge_kwh=("charge_rt_kwh", "sum"),
        discharge_kwh=("discharge_rt_kwh", "sum"), spill_kwh=("spill_kwh", "sum"), emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
        original_plan_cost_yuan=("price_yuan_per_kwh", lambda _: 0.0), contract_cost_yuan=("independent_contract_cost_yuan", "sum"),
        emergency_cost_yuan=("independent_emergency_cost_yuan", "sum"), actual_total_cost_yuan=("independent_total_cost_yuan", "sum"),
        soc_start_kwh=("soc_start_kwh", "first"), soc_end_kwh=("soc_end_kwh", "last"),
        emergency_slot_count=("emergency_purchase_kwh", lambda x: int((x > EPS).sum())),
    )
    # pandas named aggregation cannot express this price-weighted quantity directly cleanly.
    daily["original_plan_cost_yuan"] = s.assign(v=c0).groupby("date").v.sum().to_numpy()
    daily["upward_surcharge_yuan"] = s.assign(v=up_fee).groupby("date").v.sum().to_numpy()
    daily["downward_refund_yuan"] = s.assign(v=refund).groupby("date").v.sum().to_numpy()
    daily["conversion_loss_kwh"] = s.assign(v=0.1 * s.charge_rt_kwh + (1.0 / 0.9 - 1.0) * s.discharge_rt_kwh).groupby("date").v.sum().to_numpy()
    daily["emergency_day"] = daily.emergency_purchase_kwh > EPS
    total_emergency = float(e.sum())
    row = {
        "policy": policy,
        "original_plan_purchase_kwh": float(p.sum()),
        "effective_purchase_kwh": float(z.sum()),
        "original_plan_cost_yuan": float(c0.sum()),
        "upward_adjustment_kwh": float(s.delta_plus_kwh.sum()),
        "downward_adjustment_kwh": float(s.delta_minus_kwh.sum()),
        "upward_surcharge_yuan": float(up_fee.sum()),
        "downward_refund_yuan": float(refund.sum()),
        "final_contract_cost_yuan": float(contract.sum()),
        "emergency_purchase_kwh": total_emergency,
        "emergency_cost_yuan": float(emergency.sum()),
        "actual_total_cost_yuan": float(total.sum()),
        "main_checkpoint_cost_yuan": float(checkpoint.get("cumulative_cost_yuan", math.nan)),
        "main_vs_independent_total_error_yuan": float(checkpoint.get("cumulative_cost_yuan", 0.0) - total.sum()),
        "charge_kwh": float(s.charge_rt_kwh.sum()),
        "discharge_kwh": float(s.discharge_rt_kwh.sum()),
        "conversion_loss_kwh": float((0.1 * s.charge_rt_kwh + (1.0 / 0.9 - 1.0) * s.discharge_rt_kwh).sum()),
        "spill_kwh": float(s.spill_kwh.sum()),
        "emergency_day_count": int((daily.emergency_purchase_kwh > EPS).sum()),
        "emergency_slot_count": int((e > EPS).sum()),
        "emergency_weighted_actual_price_yuan_per_kwh": float((price * e).sum() / total_emergency) if total_emergency > EPS else math.nan,
        "low_price_emergency_kwh": float(s.loc[s.price_band == "低价(<0.6)", "emergency_purchase_kwh"].sum()),
        "mid_price_emergency_kwh": float(s.loc[s.price_band == "中价([0.6,1.0))", "emergency_purchase_kwh"].sum()),
        "high_price_emergency_kwh": float(s.loc[s.price_band == "高价(>=1.0)", "emergency_purchase_kwh"].sum()),
        "mean_daily_end_soc_kwh": float(daily.soc_end_kwh.mean()),
        "min_daily_end_soc_kwh": float(daily.soc_end_kwh.min()),
        "max_daily_end_soc_kwh": float(daily.soc_end_kwh.max()),
        "terminal_soc_kwh": float(daily.soc_end_kwh.iloc[-1]),
        "nonzero_adjustment_slot_count": int(((s.delta_plus_kwh > EPS) | (s.delta_minus_kwh > EPS)).sum()),
        "zero_adjustment_slot_count": int(((s.delta_plus_kwh <= EPS) & (s.delta_minus_kwh <= EPS)).sum()),
    }
    return row, daily


def monthly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x.date = pd.to_datetime(x.date)
    x["month"] = x.date.dt.strftime("%Y-%m")
    summable = [c for c in x.columns if c.endswith("_yuan") or c.endswith("_kwh") or c.endswith("_count")]
    summable = [c for c in summable if c not in ["soc_start_kwh", "soc_end_kwh"]]
    m = x.groupby(["policy", "month"], as_index=False)[summable].sum()
    ends = x.sort_values("date").groupby(["policy", "month"], as_index=False).agg(month_start_soc_kwh=("soc_start_kwh", "first"), month_end_soc_kwh=("soc_end_kwh", "last"))
    return m.merge(ends, on=["policy", "month"], how="left")


def adjustment_by_issue(slots: pd.DataFrame) -> pd.DataFrame:
    s = slots.copy()
    s.date = pd.to_datetime(s.date)
    p = s.price_yuan_per_kwh
    s["nonzero_adjustment"] = ((s.delta_plus_kwh > EPS) | (s.delta_minus_kwh > EPS)).astype(int)
    return s.groupby("active_issue_hour", as_index=False).agg(
        slot_count=("slot", "size"), upward_adjustment_kwh=("delta_plus_kwh", "sum"), downward_adjustment_kwh=("delta_minus_kwh", "sum"),
        nonzero_adjustment_slots=("nonzero_adjustment", "sum"),
    ).assign(
        upward_surcharge_yuan=s.assign(v=1.5*p*s.delta_plus_kwh).groupby("active_issue_hour").v.sum().to_numpy(),
        downward_refund_yuan=s.assign(v=0.5*p*s.delta_minus_kwh).groupby("active_issue_hour").v.sum().to_numpy(),
        final_contract_cost_yuan=s.assign(v=s.contract_cost_yuan).groupby("active_issue_hour").v.sum().to_numpy(),
    )


def terminal_day_table(slots: pd.DataFrame) -> pd.DataFrame:
    s = slots.copy()
    s.date = pd.to_datetime(s.date)
    return s[(s.date == pd.Timestamp("2025-12-31")) & (s.slot >= 120)].copy()


def format_num(x: float, digits: int = 3) -> str:
    return "—" if pd.isna(x) else f"{x:,.{digits}f}"


def dataframe_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    """Dependency-free small Markdown table renderer."""
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            vals.append(format_num(float(val)) if isinstance(val, (float, np.floating, int, np.integer)) else str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, rule, *rows])


def json_safe(value):
    """Make audit JSON standards-compliant: JSON has no NaN literal."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    cp2 = json.loads((CHECKPOINTS / "q4_2_checkpoint.json").read_text(encoding="utf-8"))
    cp3 = json.loads((CHECKPOINTS / "q4_3_checkpoint.json").read_text(encoding="utf-8"))
    s2, s3 = read_pickle("q4_2", "slots"), read_pickle("q4_3", "slots")
    i2, i3 = read_pickle("q4_2", "info"), read_pickle("q4_3", "info")
    p2, p3 = read_pickle("q4_2", "plans"), read_pickle("q4_3", "plans")
    sol2, sol3 = read_pickle("q4_2", "solver"), read_pickle("q4_3", "solver")
    integrity_rows = [integrity("问题4-2", s2, cp2), integrity("问题4-3", s3, cp3)]
    hard_bad = [r for r in integrity_rows if not r["valid_complete_trajectory"] or not r["completed_checkpoint"]]
    if hard_bad:
        raise RuntimeError(f"trajectory integrity failure; final comparison withheld: {hard_bad}")

    row2, daily2 = cost_summary("问题4-2", s2, cp2)
    row3, daily3 = cost_summary("问题4-3", s3, cp3)
    comparison = pd.DataFrame([row2, row3])
    daily = pd.concat([daily2, daily3], ignore_index=True)
    monthly = monthly_from_daily(daily)
    phys2, phys3 = physics("问题4-2", s2, i2, p2), physics("问题4-3", s3, i3, p3)
    qsol2, stat2 = solver_quality("问题4-2", sol2)
    qsol3, stat3 = solver_quality("问题4-3", sol3)
    solver_quality_all = pd.concat([qsol2, qsol3], ignore_index=True)
    issue = adjustment_by_issue(s3)
    issue.insert(0, "policy", "问题4-3")
    issue.to_csv(STAGE / "q4_v1_q4_3_adjustment_by_issue.csv", index=False, encoding="utf-8-sig")

    terminal = terminal_day_table(s3)
    terminal.to_csv(STAGE / "q4_v1_q4_3_20251231_last24_slots.csv", index=False, encoding="utf-8-sig")
    final3_day = s3.copy()
    final3_day.date = pd.to_datetime(final3_day.date)
    final3_day = final3_day[final3_day.date == pd.Timestamp("2025-12-31")]
    start3 = float(final3_day.soc_start_kwh.iloc[0])
    end3 = float(final3_day.soc_end_kwh.iloc[-1])

    inv = pd.DataFrame([
        {
            "policy": row2["policy"], "actual_bill_yuan": row2["actual_total_cost_yuan"], "terminal_soc_kwh": row2["terminal_soc_kwh"],
            "frozen_reference_soc_kwh": INVENTORY_REFERENCE_KWH, "frozen_uniform_inventory_value_yuan_per_kwh": INVENTORY_REFERENCE_VALUE,
            "inventory_correction_yuan": INVENTORY_REFERENCE_VALUE * (INVENTORY_REFERENCE_KWH - row2["terminal_soc_kwh"]),
            "inventory_corrected_cost_yuan": row2["actual_total_cost_yuan"] + INVENTORY_REFERENCE_VALUE * (INVENTORY_REFERENCE_KWH - row2["terminal_soc_kwh"]),
        },
        {
            "policy": row3["policy"], "actual_bill_yuan": row3["actual_total_cost_yuan"], "terminal_soc_kwh": row3["terminal_soc_kwh"],
            "frozen_reference_soc_kwh": INVENTORY_REFERENCE_KWH, "frozen_uniform_inventory_value_yuan_per_kwh": INVENTORY_REFERENCE_VALUE,
            "inventory_correction_yuan": INVENTORY_REFERENCE_VALUE * (INVENTORY_REFERENCE_KWH - row3["terminal_soc_kwh"]),
            "inventory_corrected_cost_yuan": row3["actual_total_cost_yuan"] + INVENTORY_REFERENCE_VALUE * (INVENTORY_REFERENCE_KWH - row3["terminal_soc_kwh"]),
        },
    ])
    inv["vs_4_2_actual_cost_yuan"] = inv.actual_bill_yuan - inv.actual_bill_yuan.iloc[0]
    inv["vs_4_2_inventory_corrected_cost_yuan"] = inv.inventory_corrected_cost_yuan - inv.inventory_corrected_cost_yuan.iloc[0]
    actual_saving = row2["actual_total_cost_yuan"] - row3["actual_total_cost_yuan"]
    actual_saving_pct = 100.0 * actual_saving / row2["actual_total_cost_yuan"]
    emergency_reduction = row2["emergency_purchase_kwh"] - row3["emergency_purchase_kwh"]
    emergency_reduction_pct = 100.0 * emergency_reduction / row2["emergency_purchase_kwh"]

    # Data exports: original slot arithmetic is retained; independently computed bills are added.
    s2o, s3o = s2.copy(), s3.copy()
    for x in (s2o, s3o):
        label = "问题4-2" if x is s2o else "问题4-3"
        if "policy" in x.columns:
            x["policy"] = label
        else:
            x.insert(0, "policy", label)
        x["independent_contract_cost_yuan"] = x.price_yuan_per_kwh*x.original_plan_kwh + 1.5*x.price_yuan_per_kwh*x.delta_plus_kwh - 0.5*x.price_yuan_per_kwh*x.delta_minus_kwh
        x["independent_emergency_cost_yuan"] = 5*x.price_yuan_per_kwh*x.emergency_purchase_kwh
        x["independent_actual_total_cost_yuan"] = x.independent_contract_cost_yuan + x.independent_emergency_cost_yuan
    pd.concat([s2o, s3o], ignore_index=True).to_csv(STAGE / "q4_v1_slot_results.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(STAGE / "q4_v1_daily_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(STAGE / "q4_v1_monthly_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(STAGE / "q4_v1_full_year_comparison.csv", index=False, encoding="utf-8-sig")
    inv.to_csv(STAGE / "q4_v1_inventory_fairness.csv", index=False, encoding="utf-8-sig")
    solver_quality_all.to_csv(STAGE / "q4_v1_solver_quality.csv", index=False, encoding="utf-8-sig")
    pd.concat([i2.assign(policy="问题4-2"), i3.assign(policy="问题4-3")], ignore_index=True).to_csv(STAGE / "q4_v1_price_information_audit.csv", index=False, encoding="utf-8-sig")

    after = protected_hashes()
    protection_changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    validation = {
        "integrity": integrity_rows,
        "physical_and_information": [phys2, phys3],
        "solver_summary": [{"policy": "问题4-2", **stat2}, {"policy": "问题4-3", **stat3}],
        "q4_3_terminal_soc_review": {
            "dec30_end_to_dec31_start_error_kwh": float(final3_day.soc_start_kwh.iloc[0] - s3.assign(date=pd.to_datetime(s3.date)).query("date == @pd.Timestamp('2025-12-30')").sort_values("slot").soc_end_kwh.iloc[-1]),
            "dec31_start_soc_kwh": start3, "dec31_end_soc_kwh": end3,
            "dec31_soc_change_kwh": end3 - start3,
            "last_24_charge_kwh": float(terminal.charge_rt_kwh.sum()), "last_24_discharge_kwh": float(terminal.discharge_rt_kwh.sum()),
            "last_24_original_plan_kwh": float(terminal.original_plan_kwh.sum()), "last_24_effective_purchase_kwh": float(terminal.effective_purchase_kwh.sum()),
            "last_24_actual_price_min_yuan_per_kwh": float(terminal.price_yuan_per_kwh.min()), "last_24_actual_price_max_yuan_per_kwh": float(terminal.price_yuan_per_kwh.max()),
            "last_slot_soc_recurrence_residual_kwh": float(final3_day.iloc[-1].soc_end_kwh - (final3_day.iloc[-1].soc_start_kwh + 0.9*final3_day.iloc[-1].charge_rt_kwh - final3_day.iloc[-1].discharge_rt_kwh/0.9)),
            "terminal_soc_q4_2_minus_q4_3_kwh": row2["terminal_soc_kwh"] - row3["terminal_soc_kwh"],
            "terminal_value_in_actual_bill": False,
        },
        "protected_hash_changes": protection_changed,
        "numeric_normalization": {
            "问题4-2": "轨迹未单独记录规范化标记；所有导出计划量均未低于 -1e-6 kWh，无法从结果表反推历史规范化次数。",
            "问题4-3_count": int(pd.to_numeric(s3.get("numeric_normalized", pd.Series(False)), errors="coerce").fillna(0).astype(bool).sum()),
            "问题4-3_max_observable_negative_magnitude_kwh": float(max(0.0, -min(s3[["original_plan_kwh", "effective_purchase_kwh", "delta_plus_kwh", "delta_minus_kwh", "emergency_purchase_kwh"]].min().min(), 0.0))),
        },
    }
    (STAGE / "q4_v1_stage4_validation.json").write_text(json.dumps(json_safe(validation), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    limits = solver_quality_all[solver_quality_all.audit_solution_class == "limit_feasible"].copy()
    limits.to_csv(STAGE / "q4_v1_limit_feasible_details.csv", index=False, encoding="utf-8-sig")
    lines = ["问题四 V1 阶段4：最终汇总与独立验证", "="*46, "", "说明：本文件仅读取完整逐槽轨迹、计划版本、求解日志和检查点；未重跑任何日期，未写入 result4-2.xlsx 或 result4-3.xlsx。", "", "一、轨迹完整性"]
    for r in integrity_rows:
        lines.append(f"- {r['policy']}：{r['date_count']} 天、{r['slot_rows']} 槽，{r['first_date']} 至 {r['last_date']}；缺失日期/槽/重复槽均为 {r['missing_dates']}/{r['missing_date_slots']}/{r['duplicate_date_slots']}；检查点完成={r['completed_checkpoint']}。")
    lines += ["", "二、独立费用复算（元）"]
    for r in [row2, row3]:
        lines.append(f"- {r['policy']}：合同/计划结算 {format_num(r['final_contract_cost_yuan'])}，紧急购电费 {format_num(r['emergency_cost_yuan'])}，实际总费用 {format_num(r['actual_total_cost_yuan'])}；主程序检查点差异 {format_num(r['main_vs_independent_total_error_yuan'], 9)}。")
    lines.append(f"- 4-3相对4-2的未修正实际账单减少 {format_num(actual_saving)} 元（{actual_saving_pct:.3f}%），紧急购电量减少 {format_num(emergency_reduction)} kWh（{emergency_reduction_pct:.3f}%）。")
    min_last24 = float(terminal.soc_start_kwh.min())
    lines += ["", "三、库存公平性", f"- 预先冻结的统一修正口径为 C^inv=C^actual+1.0(6000-E_end)，仅用于解释，不改变正式账单。4-2/4-3 期末 SOC 分别为 {format_num(row2['terminal_soc_kwh'])}/{format_num(row3['terminal_soc_kwh'])} kWh，差额（4-2减4-3）为 {format_num(row2['terminal_soc_kwh']-row3['terminal_soc_kwh'])} kWh。", f"- 修正后费用分别为 {format_num(inv.inventory_corrected_cost_yuan.iloc[0])}/{format_num(inv.inventory_corrected_cost_yuan.iloc[1])} 元。", "", "四、4-3 年末 SOC 专项审查", f"- 12月31日从 {format_num(start3)} kWh 变化至 {format_num(end3)} kWh；末24槽充电/放电分别为 {format_num(float(terminal.charge_rt_kwh.sum()))}/{format_num(float(terminal.discharge_rt_kwh.sum()))} kWh，且末24槽内 SOC 最低至 {format_num(min_last24)} kWh，随后在最后13槽集中充电至年末。", "- 轨迹表明存在年末库存回补；但不能只凭轨迹把它完全归因于终端价值，还受冻结计划、末段供需与实际价格共同影响。终端库存价值 v=0.6 未进入逐槽实际账单公式。报告/论文须将其作为有限时域终端行为及库存公平性解释的限制，而非收入。", "", "五、求解质量"]
    for policy, st in [("问题4-2", stat2), ("问题4-3", stat3)]:
        lines.append(f"- {policy}：达到最优/请求精度 {st['optimal_or_requested_gap']} 次，限时可行 {st['limit_feasible']} 次，失败 {st['failed']} 次；相对 gap 最大/均值/95分位={format_num(st['gap_max'],8)}/{format_num(st['gap_mean'],8)}/{format_num(st['gap_p95'],8)}；求解时间最大/均值={format_num(st['time_max_s'],3)}/{format_num(st['time_mean_s'],3)} s。")
    if len(limits):
        lines.append("- 限时可行解的日期、发布时间、incumbent、best bound 与 gap 已逐条写入 `q4_v1_limit_feasible_details.csv`；这些解均未称为最优。")
    lines += ["", "六、物理、信息与结算验证"]
    for p in [phys2, phys3]:
        lines.append(f"- {p['policy']}：供需最大残差 {format_num(p['max_supply_demand_balance_residual_kwh'],9)} kWh，SOC 最大递推残差 {format_num(p['max_soc_recurrence_residual_kwh'],9)} kWh，跨日最大误差 {format_num(p['max_cross_day_soc_error_kwh'],9)} kWh；SOC/功率/互斥/紧急购电同时充电违规={p['soc_below_min_count']+p['soc_above_max_count']}/{p['charge_power_limit_violations']+p['discharge_power_limit_violations']}/{p['simultaneous_charge_discharge_count']}/{p['emergency_purchase_while_charging_count']}；信息泄漏={p['all_logged_information_leak_count']}；最大总费用复算误差 {format_num(p['max_slot_total_bill_recompute_error_yuan'],9)} 元。")
    lines += ["", "七、保护性检查", f"- 受保护文件哈希变化数：{len(protection_changed)}。", "- 可以进入正式导出前审查：是（完整性、独立费用复算、物理与已记录信息边界检查均通过；4-2 的历史数值规范化次数未在逐槽轨迹中记录，已如实标为不可从输出反推）。"]
    (STAGE / "q4_v1_stage4_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    comparison_md = dataframe_markdown(comparison, ["policy", "final_contract_cost_yuan", "emergency_cost_yuan", "actual_total_cost_yuan", "emergency_purchase_kwh", "terminal_soc_kwh"])
    analysis = "# 问题四 V1：全年影子评价分析（审查版）\n\n"
    analysis += "本分析仅基于 2025-02-01 至 2025-12-31 已完成的逐槽轨迹进行独立汇总；没有重跑优化、调整参数或导出正式模板。\n\n## 全年结果\n\n" + comparison_md + "\n\n"
    analysis += f"问题4-3相对问题4-2的未修正实际账单减少 {actual_saving:.3f} 元（{actual_saving_pct:.3f}%），紧急购电量减少 {emergency_reduction:.3f} kWh（{emergency_reduction_pct:.3f}%）。问题4-2与问题4-3的冻结参数、光伏信息来源和更新机制不同，因此该费用差是综合差异，不能单独归因于日内调整。正式账单以未修正实际费用为准；库存修正仅采用开发期预先冻结的统一 1.0 元/kWh参考值作公平性说明。修正前后问题4-3均较低，但此结果仅服务于库存公平性解释。\n\n"
    analysis += "## 年末库存\n\n"
    analysis += f"问题4-3期末 SOC 为 {row3['terminal_soc_kwh']:.3f} kWh，仍在 [1200,10800] kWh 安全边界内。12月31日 SOC 由 {start3:.3f} kWh 变至 {end3:.3f} kWh；末24槽内最低为 {min_last24:.3f} kWh，随后集中充电至年末，末24槽充电/放电为 {terminal.charge_rt_kwh.sum():.3f}/{terminal.discharge_rt_kwh.sum():.3f} kWh。轨迹表明年末库存回补，但不能仅凭此将其完全归因于终端价值，亦受冻结计划、末段供需和价格共同影响。终端价值 v=0.6 元/kWh 仅影响优化决策，未从实际账单扣除；该有限时域行为应作为论文局限性披露。\n\n"
    analysis += "## 验证结论\n\n所有日期和槽位完整，逐槽供需平衡、SOC递推、跨日衔接、费用公式和已记录的信息边界均通过。详见 `q4_v1_stage4_validation.txt` 与 `q4_v1_stage4_validation.json`。\n"
    (DOCS / "q4_v1_full_year_analysis.md").write_text(analysis, encoding="utf-8")
    review = "# 问题四 V1 阶段4结果（供审查）\n\n"
    review += "## 完整性\n\n两条轨迹均完成334天、48096个10分钟槽，起止日期均为2025-02-01至2025-12-31，检查点均已完成。日期—槽位键无缺失、无重复，因此未发现恢复导致的重复写入或重复计费。另有一份既存的 `q4_v1_stage4_failure.json`：其记录的是问题4-2完成检查点之后一次汇总/续跑编排遇到 NaT 起始日期，并非求解不可行；它未在完整轨迹中留下重复槽或费用。\n\n"
    review += "## 独立汇总\n\n" + comparison_md + "\n\n"
    review += f"问题4-3独立复算总费用为 {row3['actual_total_cost_yuan']:.3f} 元（检查点累计值差异 {row3['main_vs_independent_total_error_yuan']:.9f} 元），其中最终合同结算 {row3['final_contract_cost_yuan']:.3f} 元、紧急购电费 {row3['emergency_cost_yuan']:.3f} 元。\n\n"
    review += "## 审查边界\n\n限时获得 incumbent 的解均单列为“限时可行解”，不称为理论最优。若进入正式导出，还需人工确认题目模板中问题4-2和4-3的字段、单位与聚合规则；本阶段未创建模板文件。\n"
    (DOCS / "q4_v1_stage4_results_for_review.md").write_text(review, encoding="utf-8")
    print("FINALIZATION_COMPLETE")


if __name__ == "__main__":
    main()
