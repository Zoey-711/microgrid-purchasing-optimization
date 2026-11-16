"""Paired Q4-3 ablation: identical information, only plan adjustment differs.

This is an isolated read-only experiment.  It never writes formal templates,
frozen configurations, or the original Q4 stage-4 trajectory.
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.q2_v2.execution import build_path_value_functions, execute_day, select_dp_action
from src.q2_v2.scenarios import JointScenarios
from src.q3_v2.config import MIP_RANDOM_SEED
from src.q3_v2.rolling_milp import solve_version_milp
from src.q4_v1 import run_stage4 as frozen
from src.q4_v1 import run_stage2 as data_source


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "outputs" / "q4_v1" / "stage3" / "q4_v1_frozen_config_4_3.json"
OUT = ROOT / "outputs" / "q4_v1" / "ablation" / "same_info"
FORMAL_START = pd.Timestamp("2025-02-01")
FORMAL_END = pd.Timestamp("2025-12-31")
N = 144
EPS = 1e-6
POLICIES = {
    "locked": False,
    "rolling": True,
}
PROTECTED = [
    "outputs/tables/result1.xlsx", "outputs/tables/result2.xlsx", "outputs/tables/result3.xlsx",
    "outputs/tables/result4-2.xlsx", "outputs/tables/result4-3.xlsx",
    "outputs/q4_v1/stage3/q4_v1_frozen_config_4_3.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_hash(*items: object) -> str:
    digest = hashlib.sha256()
    for item in items:
        if isinstance(item, np.ndarray):
            digest.update(np.ascontiguousarray(item, dtype=np.float64).tobytes())
        else:
            digest.update(str(item).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.10f")
    os.replace(tmp, path)


def frozen_config() -> dict:
    config = json.loads(FROZEN.read_text(encoding="utf-8"))
    expected = {"price_predictor": "P3", "supply_predictor": "weekday", "M": 14, "v": 0.6, "grid": 100}
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        raise RuntimeError(f"Q4-3 frozen configuration mismatch: {mismatch}")
    return config


def scenario_record(policy: str, date: pd.Timestamp, issue_hour: int, scenario, forecast_frame: pd.DataFrame, price_suffix: np.ndarray) -> dict:
    """External information signature; it deliberately excludes SOC and plans."""
    return {
        "policy": policy,
        "date": date.date().isoformat(),
        "issue_hour": issue_hour,
        "scenario_source_dates": "|".join(scenario.source_dates),
        "scenario_count": scenario.count,
        "weight_hash": stable_hash(scenario.weights),
        "load_scenario_hash": stable_hash(scenario.load_kwh),
        "pv_scenario_hash": stable_hash(scenario.pv_kwh),
        "forecast_hash": stable_hash(
            forecast_frame.idx.to_numpy(int),
            forecast_frame.load_forecast_kwh.to_numpy(float),
            forecast_frame.pv_forecast_kwh.to_numpy(float),
            price_suffix,
        ),
        "price_scenario": "not_used_by_frozen_q4_3_model",
        "random_seed": MIP_RANDOM_SEED,
    }


def cold_start_day(date: pd.Timestamp, labels, load, pv, price, soc: float, policy: str) -> tuple[pd.DataFrame, float, list[dict], list[dict], list[dict]]:
    """The frozen Q4-3 code uses this Jan-01 no-history fallback for preheating."""
    frame, _ = execute_day(date.date().isoformat(), labels, load, pv, np.zeros(N), price, soc, "greedy")
    frame = frame.rename(columns={
        "planned_purchase_kwh": "original_plan_kwh", "charge_kwh": "charge_rt_kwh",
        "discharge_kwh": "discharge_rt_kwh", "actual_cost_yuan": "actual_total_cost_yuan",
    })
    frame["effective_purchase_kwh"] = 0.0
    frame["delta_plus_kwh"] = 0.0
    frame["delta_minus_kwh"] = 0.0
    frame["contract_cost_yuan"] = 0.0
    frame["policy"] = policy
    frame["active_issue_hour"] = 0
    frame["numeric_normalized"] = False
    solver = [{"policy": policy, "date": date.date().isoformat(), "issue_hour": 0,
               "solution_class": "cold_start_greedy_no_history", "relative_gap": np.nan,
               "solve_time_seconds": 0.0, "random_seed": MIP_RANDOM_SEED}]
    plans = [{"policy": policy, "date": date.date().isoformat(), "issue_hour": 0, "slot": t,
              "original_plan_kwh": 0.0, "candidate_version_kwh": 0.0,
              "effective_plan_kwh": 0.0, "executed_prefix_lock_error_kwh": 0.0} for t in range(N)]
    audit = [{"policy": policy, "date": date.date().isoformat(), "issue_hour": 0,
              "scenario_source_dates": "cold_start_no_completed_history", "scenario_count": 0,
              "weight_hash": "cold_start", "load_scenario_hash": "cold_start", "pv_scenario_hash": "cold_start",
              "forecast_hash": "cold_start", "price_scenario": "not_used_by_frozen_q4_3_model",
              "random_seed": MIP_RANDOM_SEED}]
    return frame, float(frame.soc_end_kwh.iloc[-1]), solver, plans, audit


def run_day(i: int, dates, labels, load, pv, price, cache, config: dict, soc: float, policy: str, allow: bool):
    date = pd.Timestamp(dates[i])
    if i == 0:
        return cold_start_day(date, labels, load[i], pv[i], price[i], soc, policy)

    original: np.ndarray | None = None
    effective = np.zeros(N, dtype=float)
    rows: list[dict] = []
    solver_rows: list[dict] = []
    plan_rows: list[dict] = []
    audit_rows: list[dict] = []

    for hour in frozen.ISSUES:
        start, stop = frozen.START[hour], frozen.STOP[hour]
        scenario = frozen.scen43(i, hour, cache, load, pv, int(config["M"]))
        current = cache[(i, hour)].sort_values("idx")
        predicted_price = np.zeros(N, dtype=float)
        predicted_price[start:] = current.price_forecast_yuan_per_kwh.to_numpy(float)
        audit_rows.append(scenario_record(policy, date, hour, scenario, current, predicted_price[start:]))

        # Both policies solve the identical 0:00 problem.  At later updates the
        # locked policy still reads and records the same updated information, but
        # its effective suffix remains the 0:00 contract plan.
        result = solve_version_milp(scenario, predicted_price, soc, float(config["v"]), original)
        candidate = result.plan.planned_purchase_kwh.to_numpy(float)
        if hour == 0:
            original = np.zeros(N, dtype=float)
            original[start:] = candidate
            effective[start:] = candidate
            prefix_error = 0.0
            candidate_used = True
        elif allow:
            before = effective[:start].copy()
            effective[start:] = candidate
            prefix_error = float(np.max(np.abs(before - effective[:start]))) if start else 0.0
            if prefix_error > EPS:
                raise RuntimeError("Executed plan prefix was modified")
            candidate_used = True
        else:
            prefix_error = float(np.max(np.abs(effective[:start] - original[:start]))) if start else 0.0
            if prefix_error > EPS or np.max(np.abs(effective[start:] - original[start:])) > EPS:
                raise RuntimeError("Locked policy effective plan deviated from the 0:00 original plan")
            candidate_used = False

        meta = {"policy": policy, "date": date.date().isoformat(), "issue_hour": hour,
                **result.metadata, "candidate_plan_used": candidate_used,
                "candidate_plan_hash": stable_hash(candidate), "effective_suffix_hash": stable_hash(effective[start:]),
                "scenario_source_dates": "|".join(scenario.source_dates)}
        solver_rows.append(meta)

        # The value function is rebuilt at every published forecast time for both
        # policies.  Its only branch-specific input is the current effective plan
        # (and the independently propagated SOC), exactly as specified.
        joint = JointScenarios(tuple(range(scenario.count)), scenario.source_dates, scenario.weights,
                               scenario.load_kwh, scenario.pv_kwh)
        value = build_path_value_functions(joint, effective[start:].copy(), predicted_price[start:],
                                           float(config["grid"]), float(config["v"]))
        for slot in range(start, stop):
            relative = slot - start
            z, norm_z = frozen.normalize(float(effective[slot]))
            p = float(original[slot])
            gap = float(load[i, slot] - pv[i, slot] - z)
            charge, discharge, emergency, spill, next_soc = select_dp_action(
                gap, soc, float(price[i, slot]), value.mean_value_yuan[relative + 1], value.soc_grid_kwh
            )
            charge, nc = frozen.normalize(charge)
            discharge, nd = frozen.normalize(discharge)
            emergency, ne = frozen.normalize(emergency)
            spill, nw = frozen.normalize(spill)
            plus, minus, contract = frozen.settlement(p, z, float(price[i, slot]))
            rows.append({
                "policy": policy, "allow_intraday_plan_adjustment": allow, "date": date.date().isoformat(),
                "slot": slot, "time_label": labels[slot], "active_issue_hour": hour,
                "price_yuan_per_kwh": float(price[i, slot]), "load_actual_kwh": float(load[i, slot]),
                "pv_actual_kwh": float(pv[i, slot]), "original_plan_kwh": p,
                "effective_purchase_kwh": z, "delta_plus_kwh": plus, "delta_minus_kwh": minus,
                "charge_rt_kwh": charge, "discharge_rt_kwh": discharge,
                "emergency_purchase_kwh": emergency, "spill_kwh": spill,
                "soc_start_kwh": soc, "soc_end_kwh": next_soc, "contract_cost_yuan": contract,
                "emergency_cost_yuan": 5.0 * float(price[i, slot]) * emergency,
                "actual_total_cost_yuan": contract + 5.0 * float(price[i, slot]) * emergency,
                "balance_residual_kwh": z + pv[i, slot] + discharge + emergency - load[i, slot] - charge - spill,
                "numeric_normalized": bool(norm_z or nc or nd or ne or nw),
            })
            soc = next_soc
        plan_rows.extend({
            "policy": policy, "date": date.date().isoformat(), "issue_hour": hour, "slot": slot,
            "original_plan_kwh": float(original[slot]), "candidate_version_kwh": float(candidate[slot - start]),
            "effective_plan_kwh": float(effective[slot]), "executed_prefix_lock_error_kwh": prefix_error,
        } for slot in range(start, N))
    return pd.DataFrame(rows), soc, solver_rows, plan_rows, audit_rows


def save_checkpoint(policy: str, next_index: int, soc: float, slots, solver, plans, audit, config_hash: str) -> None:
    folder = OUT / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix, frame in [("slots", slots), ("solver", solver), ("plans", plans), ("audit", audit)]:
        pd.concat(frame, ignore_index=True).to_pickle(folder / f"{policy}_{suffix}.pkl")
    atomic_json(folder / f"{policy}.json", {
        "policy": policy, "config_sha256": config_hash, "next_index": next_index,
        "completed_through": None if next_index == 0 else str(next_index - 1), "end_soc_kwh": soc,
        "completed": next_index >= 365,
    })


def load_checkpoint(policy: str, config_hash: str):
    folder = OUT / "checkpoints"
    checkpoint = folder / f"{policy}.json"
    if not checkpoint.exists():
        return 0, 6000.0, [], [], [], []
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if state.get("config_sha256") != config_hash or state.get("policy") != policy:
        raise RuntimeError("Ablation checkpoint configuration mismatch")
    frames = [pd.read_pickle(folder / f"{policy}_{suffix}.pkl") for suffix in ("slots", "solver", "plans", "audit")]
    return int(state["next_index"]), float(state["end_soc_kwh"]), [frames[0]], [frames[1]], [frames[2]], [frames[3]]


def validate_trajectory(slots: pd.DataFrame, plans: pd.DataFrame, formal_only: bool) -> dict:
    x = slots[slots.date >= FORMAL_START.date().isoformat()].copy() if formal_only else slots.copy()
    if x.empty:
        raise RuntimeError("No trajectory rows available for validation")
    expected_days = 334 if formal_only else 365
    by_day = x.groupby("date", sort=True)
    ends, starts = by_day.soc_end_kwh.last().to_numpy(float), by_day.soc_start_kwh.first().to_numpy(float)
    lock = plans[plans.date >= FORMAL_START.date().isoformat()] if formal_only else plans
    return {
        "days": int(x.date.nunique()), "expected_days": expected_days, "slots": int(len(x)),
        "expected_slots": expected_days * N, "duplicate_slots": int(x.duplicated(["policy", "date", "slot"]).sum()),
        "max_balance_residual_kwh": float(abs(x.balance_residual_kwh).max()),
        "max_soc_transition_residual_kwh": float(abs(x.soc_end_kwh - x.soc_start_kwh - .9 * x.charge_rt_kwh + x.discharge_rt_kwh / .9).max()),
        "max_cross_day_soc_error_kwh": float(abs(ends[:-1] - starts[1:]).max()) if len(ends) > 1 else 0.0,
        "soc_min_kwh": float(min(x.soc_start_kwh.min(), x.soc_end_kwh.min())),
        "soc_max_kwh": float(max(x.soc_start_kwh.max(), x.soc_end_kwh.max())),
        "power_violation_count": int(((x.charge_rt_kwh > 5000 / 6 + EPS) | (x.discharge_rt_kwh > 5000 / 6 + EPS)).sum()),
        "simultaneous_charge_discharge_count": int(((x.charge_rt_kwh > EPS) & (x.discharge_rt_kwh > EPS)).sum()),
        "emergency_and_charge_count": int(((x.emergency_purchase_kwh > EPS) & (x.charge_rt_kwh > EPS)).sum()),
        "negative_effective_plan_count": int((x.effective_purchase_kwh < -EPS).sum()),
        "prefix_lock_max_error_kwh": float(abs(lock.executed_prefix_lock_error_kwh).max()) if len(lock) else 0.0,
    }


def cost_check(slots: pd.DataFrame) -> dict:
    recomputed = (slots.price_yuan_per_kwh * slots.original_plan_kwh
                  + 1.5 * slots.price_yuan_per_kwh * slots.delta_plus_kwh
                  - .5 * slots.price_yuan_per_kwh * slots.delta_minus_kwh
                  + 5.0 * slots.price_yuan_per_kwh * slots.emergency_purchase_kwh)
    daily_error = abs(pd.DataFrame({"date": slots.date, "cost": recomputed}).groupby("date").cost.sum()
                      - slots.groupby("date").actual_total_cost_yuan.sum())
    return {"max_slot_error_yuan": float(abs(recomputed - slots.actual_total_cost_yuan).max()),
            "max_daily_error_yuan": float(daily_error.max()),
            "total_error_yuan": float(abs(recomputed.sum() - slots.actual_total_cost_yuan.sum()))}


def period_summary(policy: str, slots: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    x = slots[(slots.date >= start.date().isoformat()) & (slots.date <= end.date().isoformat())].copy()
    if len(x) != len(pd.date_range(start, end)) * N:
        raise RuntimeError(f"Incomplete {policy} period summary")
    actual = float(x.actual_total_cost_yuan.sum())
    first_soc, end_soc = float(x.soc_start_kwh.iloc[0]), float(x.soc_end_kwh.iloc[-1])
    return {
        "policy": policy, "period": f"{start.date()}..{end.date()}", "days": int(x.date.nunique()),
        "start_soc_kwh": first_soc, "end_soc_kwh": end_soc,
        "contract_cost_yuan": float(x.contract_cost_yuan.sum()),
        "emergency_cost_yuan": float(x.emergency_cost_yuan.sum()),
        "actual_total_cost_yuan": actual, "emergency_purchase_kwh": float(x.emergency_purchase_kwh.sum()),
        "charge_kwh": float(x.charge_rt_kwh.sum()), "discharge_kwh": float(x.discharge_rt_kwh.sum()),
        "spill_kwh": float(x.spill_kwh.sum()),
        "conversion_loss_kwh": float(((.1 * x.charge_rt_kwh) + (1 / .9 - 1) * x.discharge_rt_kwh).sum()),
        "inventory_corrected_from_6000_yuan": actual + (6000.0 - end_soc),
        "inventory_corrected_interval_yuan": actual + (first_soc - end_soc),
        "upward_adjustment_kwh": float(x.delta_plus_kwh.sum()), "downward_adjustment_kwh": float(x.delta_minus_kwh.sum()),
    }


def paired_information_check(audit: pd.DataFrame) -> pd.DataFrame:
    cols = ["scenario_source_dates", "scenario_count", "weight_hash", "load_scenario_hash", "pv_scenario_hash", "forecast_hash", "price_scenario", "random_seed"]
    pivot = audit.pivot(index=["date", "issue_hour"], columns="policy", values=cols)
    rows = []
    for (date, hour), row in pivot.iterrows():
        same = all(row[(column, "locked")] == row[(column, "rolling")] for column in cols)
        rows.append({"date": date, "issue_hour": hour, "identical_exogenous_information": same,
                     **{column: row[(column, "locked")] for column in cols}})
    result = pd.DataFrame(rows)
    if not result.identical_exogenous_information.all():
        raise RuntimeError("Paired scenario/information audit failed")
    return result


def run_policy(policy: str, allow: bool, dates, labels, load, pv, price, cache, config, config_hash: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index, soc, slots, solver, plans, audit = load_checkpoint(policy, config_hash)
    for i in range(index, len(dates)):
        frame, soc, sol, plan, info = run_day(i, dates, labels, load, pv, price, cache, config, soc, policy, allow)
        slots.append(frame); solver.append(pd.DataFrame(sol)); plans.append(pd.DataFrame(plan)); audit.append(pd.DataFrame(info))
        save_checkpoint(policy, i + 1, soc, slots, solver, plans, audit, config_hash)
        print(f"{policy} completed {dates[i].date()} end_soc={soc:.6f}", flush=True)
    return (pd.concat(slots, ignore_index=True), pd.concat(solver, ignore_index=True),
            pd.concat(plans, ignore_index=True), pd.concat(audit, ignore_index=True))


def smoke_test(dates, labels, load, pv, price, cache, config, config_hash: str) -> None:
    marker = OUT / "smoke" / "validation.json"
    if marker.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior.get("passed") and prior.get("config_sha256") == config_hash:
            return
    results = {}
    for policy, allow in POLICIES.items():
        soc = 6000.0; frames = []; audits = []; plans = []
        for i in range(3):
            x, soc, _, p, a = run_day(i, dates, labels, load, pv, price, cache, config, soc, policy, allow)
            frames.append(x); plans.append(pd.DataFrame(p)); audits.append(pd.DataFrame(a))
        results[policy] = (pd.concat(frames), pd.concat(plans), pd.concat(audits))
    paired = paired_information_check(pd.concat([results["locked"][2], results["rolling"][2]], ignore_index=True))
    checks = {policy: validate_trajectory(x, p, formal_only=False) for policy, (x, p, _) in results.items()}
    hard = [policy for policy, check in checks.items() if check["slots"] != 3 * N or check["duplicate_slots"] or check["max_balance_residual_kwh"] > 1e-6 or check["max_soc_transition_residual_kwh"] > 1e-6]
    if hard:
        raise RuntimeError(f"Three-day smoke validation failed: {hard}")
    atomic_csv(OUT / "smoke" / "paired_information_audit.csv", paired)
    atomic_json(marker, {"passed": True, "config_sha256": config_hash, "checks": checks})


def main(smoke_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = {path: sha256(ROOT / path) for path in PROTECTED}
    config = frozen_config()
    config_hash = sha256(FROZEN)
    inherited = {
        "frozen_config_path": str(FROZEN.relative_to(ROOT)), "frozen_config_sha256": config_hash,
        "frozen_config": config, "reused_code": [
            "src/q4_v1/run_stage4.py: cache43, scen43, normalize, settlement",
            "src/q3_v2/rolling_milp.py: solve_version_milp and frozen solver settings",
            "src/q2_v2/execution.py: build_path_value_functions, select_dp_action",
        ],
        "experimental_difference": "allow_intraday_plan_adjustment=False versus True",
        "price_scenario_in_frozen_model": "not used; the frozen model uses causal price point forecasts",
        "cold_start": "Jan-01 uses the existing Q4-3 no-completed-history greedy fallback; both policies start at SOC=6000 kWh",
    }
    atomic_json(OUT / "inherited_configuration.json", inherited)
    atomic_json(OUT / "protected_hashes_before.json", before)
    dates, labels, load, pv, price = data_source.read_data()
    cache = frozen.cache43(dates, labels, load, pv, price)  # Shared read-only cache for both policies.
    try:
        smoke_test(dates, labels, load, pv, price, cache, config, config_hash)
        if smoke_only:
            after = {path: sha256(ROOT / path) for path in PROTECTED}
            changed = [path for path in before if before[path] != after[path]]
            if changed:
                raise RuntimeError(f"Protected files changed during smoke test: {changed}")
            atomic_json(OUT / "protected_hashes_after_smoke.json", after)
            print("Three-day paired smoke test passed; protected-file hash changes: 0", flush=True)
            return
        locked = run_policy("locked", False, dates, labels, load, pv, price, cache, config, config_hash)
        rolling = run_policy("rolling", True, dates, labels, load, pv, price, cache, config, config_hash)
    except Exception as exc:
        atomic_json(OUT / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise
    slot_frames = {"locked": locked[0], "rolling": rolling[0]}
    solver = pd.concat([locked[1], rolling[1]], ignore_index=True)
    plans = pd.concat([locked[2], rolling[2]], ignore_index=True)
    paired = paired_information_check(pd.concat([locked[3], rolling[3]], ignore_index=True))
    formal = [period_summary(name, frame, FORMAL_START, FORMAL_END) for name, frame in slot_frames.items()]
    warmup = [period_summary(name, frame, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")) for name, frame in slot_frames.items()]
    formal_by_policy = {row["policy"]: row for row in formal}
    locked_f, rolling_f = formal_by_policy["locked"], formal_by_policy["rolling"]
    saving = locked_f["actual_total_cost_yuan"] - rolling_f["actual_total_cost_yuan"]
    emergency_reduction = locked_f["emergency_purchase_kwh"] - rolling_f["emergency_purchase_kwh"]
    comparison = pd.DataFrame([{
        "period": "2025-02-01..2025-12-31", "locked_actual_cost_yuan": locked_f["actual_total_cost_yuan"],
        "rolling_actual_cost_yuan": rolling_f["actual_total_cost_yuan"], "cost_saving_yuan": saving,
        "cost_reduction_rate": saving / locked_f["actual_total_cost_yuan"],
        "locked_emergency_kwh": locked_f["emergency_purchase_kwh"], "rolling_emergency_kwh": rolling_f["emergency_purchase_kwh"],
        "emergency_reduction_kwh": emergency_reduction,
        "emergency_reduction_rate": emergency_reduction / locked_f["emergency_purchase_kwh"] if locked_f["emergency_purchase_kwh"] > EPS else np.nan,
        "cost_order_actual": "rolling_lower" if saving > EPS else "locked_lower" if saving < -EPS else "tie",
        "cost_order_6000_inventory": "rolling_lower" if locked_f["inventory_corrected_from_6000_yuan"] - rolling_f["inventory_corrected_from_6000_yuan"] > EPS else "locked_lower",
        "cost_order_interval_inventory": "rolling_lower" if locked_f["inventory_corrected_interval_yuan"] - rolling_f["inventory_corrected_interval_yuan"] > EPS else "locked_lower",
    }])
    daily = pd.concat([frame.groupby("date", as_index=False).agg(
        policy=("policy", "first"), contract_cost_yuan=("contract_cost_yuan", "sum"),
        emergency_cost_yuan=("emergency_cost_yuan", "sum"), actual_total_cost_yuan=("actual_total_cost_yuan", "sum"),
        emergency_purchase_kwh=("emergency_purchase_kwh", "sum"), charge_kwh=("charge_rt_kwh", "sum"),
        discharge_kwh=("discharge_rt_kwh", "sum"), spill_kwh=("spill_kwh", "sum"),
        soc_start_kwh=("soc_start_kwh", "first"), soc_end_kwh=("soc_end_kwh", "last"))
        for frame in slot_frames.values()], ignore_index=True)
    monthly = daily[daily.date >= FORMAL_START.date().isoformat()].assign(month=lambda x: x.date.str.slice(0, 7)).groupby(["policy", "month"], as_index=False).sum(numeric_only=True)
    checks = {name: {"all": validate_trajectory(frame, plans[plans.policy == name], False),
                      "formal": validate_trajectory(frame, plans[plans.policy == name], True),
                      "cost": cost_check(frame[frame.date >= FORMAL_START.date().isoformat()])}
              for name, frame in slot_frames.items()}
    locked_slots = slot_frames["locked"]
    checks["locked_effective_equals_original_max_error_kwh"] = float(abs(locked_slots.effective_purchase_kwh - locked_slots.original_plan_kwh).max())
    if checks["locked_effective_equals_original_max_error_kwh"] > EPS:
        raise RuntimeError("Locked policy violated effective_plan == original_00_plan")
    after = {path: sha256(ROOT / path) for path in PROTECTED}
    changed = [path for path in before if before[path] != after[path]]
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")
    atomic_csv(OUT / "warmup_summary.csv", pd.DataFrame(warmup))
    atomic_csv(OUT / "full_year_comparison.csv", comparison)
    atomic_csv(OUT / "daily_results.csv", daily)
    atomic_csv(OUT / "monthly_results.csv", monthly)
    atomic_csv(OUT / "slot_results.csv", pd.concat(slot_frames.values(), ignore_index=True))
    atomic_csv(OUT / "solver_quality.csv", solver)
    atomic_csv(OUT / "plan_versions.csv", plans)
    atomic_csv(OUT / "paired_information_audit.csv", paired)
    validation = {"config_sha256": config_hash, "formal_start": str(FORMAL_START.date()), "formal_end": str(FORMAL_END.date()),
                  "warmup": warmup, "formal": formal, "comparison": comparison.iloc[0].to_dict(),
                  "checks": checks, "protected_hashes_before": before, "protected_hashes_after": after,
                  "protected_changed": changed, "recovery": {name: json.loads((OUT / "checkpoints" / f"{name}.json").read_text(encoding="utf-8")) for name in POLICIES}}
    atomic_json(OUT / "validation.json", validation)
    (OUT / "validation.txt").write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": comparison.iloc[0].to_dict(), "checks": checks, "changed": changed}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the paired, same-information Q4-3 ablation.")
    parser.add_argument("--smoke-only", action="store_true", help="Run only the fixed Jan-01..Jan-03 paired validation.")
    main(parser.parse_args().smoke_only)
