"""Q3-V2 Stage 4: resumable Feb--Dec causal shadow evaluation.

This module is intentionally isolated from the frozen Q1/Q2 and legacy-Q3
folders.  It consumes only the Stage-3 frozen configuration plus previously
saved January warm-up trajectories.  It never writes ``result3.xlsx``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_q3 import load_q3_pv_forecast_records
from src.q2_v2.data import load_inputs
from src.q2_v2.execution import build_path_value_functions, greedy_action, select_dp_action

from .config import (
    EMERGENCY_MULTIPLIER, ENERGY_MAX_KWH, FIRST_MODIFIABLE_SLOT, ISSUE_HOURS,
    NEXT_BOUNDARY_SLOT, ROOT, SOC_MAX_KWH, SOC_MIN_KWH, TOL,
)
from .forecasting import build_forecast_cache
from .rolling_milp import solve_version_milp
from .run_stage2 import _actual_long, protected_manifest
from .run_stage3 import _scenarios, _validate_scenario
from .settlement import settle_contract


STAGE3 = ROOT / "outputs/q3_v2/stage3"
STAGE4 = ROOT / "outputs/q3_v2/stage4"
DOC = ROOT / "docs/q3_v2"
FORMAL_START = "2025-02-01"
FORMAL_END = "2025-12-31"
SPECIAL_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
NONNEGATIVE_NORMALIZATION_EPS_KWH = 1e-6
POLICIES = {
    "A_all_updates_dynamic_value": "all_updates",
    "B_no_adjustment_dynamic_value": "no_adjustment",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.10f")
    temporary.replace(path)


def _load_config() -> dict:
    path = STAGE3 / "q3_v2_frozen_config.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "load_method": "weekday35_bias12", "scenario_cap": 14,
        "terminal_value": 0.4, "soc_grid_kwh": 100.0,
        "adjustment_policy": "all_updates", "issue_hours": [0, 6, 12, 18],
        "first_modifiable_slots": [0, 35, 71, 107],
    }
    for key, expected in required.items():
        if frozen.get(key) != expected:
            raise RuntimeError(f"Frozen configuration mismatch for {key}: {frozen.get(key)!r}")
    return frozen


def _january_terminal_soc(candidate_id: str) -> tuple[float, Path]:
    path = STAGE3 / "candidates" / candidate_id / "slots.csv"
    rows = pd.read_csv(path, usecols=["date", "slot", "soc_end_kwh"])
    final = rows.loc[(rows["date"] == "2025-01-31") & (rows["slot"] == 143), "soc_end_kwh"]
    if len(final) != 1:
        raise RuntimeError(f"Cannot identify the single Jan-31 terminal SOC in {path}")
    return float(final.iloc[0]), path


def _policy_initialization() -> dict[str, dict]:
    configs = {
        "A_all_updates_dynamic_value": "weekday35_bias12_M14_v0p4_h100_all_updates",
        "B_no_adjustment_dynamic_value": "weekday35_bias12_M14_v0p4_h100_no_adjustment",
    }
    result = {}
    for policy, candidate in configs.items():
        soc, source = _january_terminal_soc(candidate)
        result[policy] = {
            "initial_soc_kwh": soc,
            "initial_soc_provenance": "candidate-specific causal Jan-01--Jan-31 warm-up terminal SOC; used unchanged as Feb-01 00:00 SOC",
            "stage3_candidate_id": candidate,
            "source_file": str(source.relative_to(ROOT)),
            "source_sha256": _sha256(source),
        }
    return result


def _day_dir(policy: str, date: str) -> Path:
    return STAGE4 / "policies" / policy / "days" / date


def _checkpoint_path(policy: str) -> Path:
    return STAGE4 / "checkpoints" / f"{policy}.json"


def _load_checkpoint(policy: str, config_hash: str, initialization: dict) -> tuple[pd.Timestamp, float]:
    checkpoint = _checkpoint_path(policy)
    if not checkpoint.exists():
        return pd.Timestamp(FORMAL_START), float(initialization["initial_soc_kwh"])
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if state.get("config_sha256") != config_hash or state.get("policy") != policy:
        raise RuntimeError(f"Checkpoint configuration does not match frozen Stage-4 run: {checkpoint}")
    completed = pd.Timestamp(state["completed_through"])
    expected = pd.date_range(FORMAL_START, completed, freq="D")
    disk_dates = sorted((STAGE4 / "policies" / policy / "days").glob("2025-*-*"))
    if [p.name for p in disk_dates] != [d.date().isoformat() for d in expected]:
        raise RuntimeError(f"Checkpoint/day-file mismatch for {policy}; refusing unsafe resume")
    return completed + pd.Timedelta(days=1), float(state["next_soc_kwh"])


def _slot_row(
    policy: str, date: str, t: int, issue: int, run_id: str, forecast: pd.DataFrame,
    original: np.ndarray, effective: np.ndarray, data, day_index: int, soc: float,
    charge: float, discharge: float, emergency: float, spill: float, next_soc: float,
) -> dict:
    z = float(effective[t])
    settled = settle_contract([original[t]], [z], [data.price_yuan_per_kwh[t]]).iloc[0]
    emergency_cost = float(EMERGENCY_MULTIPLIER * data.price_yuan_per_kwh[t] * emergency)
    balance = z + data.pv_kwh[day_index, t] + discharge + emergency - data.load_kwh[day_index, t] - charge - spill
    fcm = forecast.set_index("idx")
    return {
        "policy": policy, "date": date, "slot": t, "time_label": data.time_labels[t],
        "active_issue_hour": issue, "active_plan_run_id": run_id, "value_function_refresh_id": run_id,
        "price_yuan_per_kwh": float(data.price_yuan_per_kwh[t]),
        "load_actual_kwh": float(data.load_kwh[day_index, t]), "pv_actual_kwh": float(data.pv_kwh[day_index, t]),
        "load_forecast_kwh": float(fcm.loc[t, "load_forecast_kwh"]), "pv_forecast_kwh": float(fcm.loc[t, "pv_forecast_kwh"]),
        "net_load_actual_kwh": float(data.load_kwh[day_index, t] - data.pv_kwh[day_index, t]),
        "original_plan_kwh": float(original[t]), "effective_purchase_kwh": z,
        "delta_plus_kwh": float(settled.delta_plus_kwh), "delta_minus_kwh": float(settled.delta_minus_kwh),
        "charge_rt_kwh": float(charge), "discharge_rt_kwh": float(discharge),
        "emergency_purchase_kwh": float(emergency), "spill_kwh": float(spill),
        "soc_start_kwh": float(soc), "soc_end_kwh": float(next_soc),
        "contract_cost_yuan": float(settled.contract_cost_yuan), "emergency_cost_yuan": emergency_cost,
        "actual_total_cost_yuan": float(settled.contract_cost_yuan + emergency_cost),
        "balance_residual_kwh": float(balance),
        "realtime_observation_assumption": "current_slot_actual_net_load_observable_at_execution_not_known_before_slot",
    }


def _normalize_theoretical_nonnegative(
    values: np.ndarray,
    name: str,
    price: np.ndarray,
    context: str,
    audit: dict,
) -> np.ndarray:
    """Normalize only solver-scale negative noise on a nonnegative variable.

    This is deliberately asymmetric: values in [-epsilon, 0) become exactly
    zero; values below -epsilon remain a hard failure.  It never uses abs()
    and never changes positive values.  The conservative 1.5*pi*|x| bound is
    logged as a maximum possible one-slot contract-fee impact.
    """
    raw = np.asarray(values, dtype=float)
    if not np.isfinite(raw).all():
        raise RuntimeError(f"Non-finite {name} at {context}")
    severe = raw < -NONNEGATIVE_NORMALIZATION_EPS_KWH
    if severe.any():
        smallest = float(raw.min())
        raise RuntimeError(
            f"{name} contains a material negative value {smallest:.16g} kWh at {context}; "
            f"below epsilon={NONNEGATIVE_NORMALIZATION_EPS_KWH:g}"
        )
    tiny = (raw < 0.0) & ~severe
    if tiny.any():
        magnitude = np.abs(raw[tiny])
        price_array = np.broadcast_to(np.asarray(price, dtype=float), raw.shape)
        audit["count"] += int(tiny.sum())
        audit["max_abs_kwh"] = max(float(audit["max_abs_kwh"]), float(magnitude.max()))
        audit["max_cost_impact_yuan"] = max(
            float(audit["max_cost_impact_yuan"]),
            float(np.max(1.5 * price_array[tiny] * magnitude)),
        )
        audit["events"].append({
            "variable": name, "context": context, "count": int(tiny.sum()),
            "minimum_raw_kwh": float(raw[tiny].min()), "maximum_abs_kwh": float(magnitude.max()),
        })
        raw = raw.copy()
        raw[tiny] = 0.0
    return raw


def _run_day(policy: str, adjustment_policy: str, date: str, soc_start: float, data, cache: dict, frozen: dict):
    di = data.day_index(date)
    soc = float(soc_start)
    original: np.ndarray | None = None
    effective = np.full(144, np.nan)
    slots: list[dict] = []
    plans: list[dict] = []
    solvers: list[dict] = []
    infos: list[dict] = []
    normalization = {"count": 0, "max_abs_kwh": 0.0, "max_cost_impact_yuan": 0.0, "events": []}
    day_started = time.perf_counter()
    for issue in ISSUE_HOURS:
        start, stop = FIRST_MODIFIABLE_SLOT[issue], NEXT_BOUNDARY_SLOT[issue]
        forecast = cache[(date, issue)].sort_values("idx")
        scenarios = _scenarios(data, cache, date, issue, int(frozen["scenario_cap"]))
        solved = None
        run_id = f"{policy}_{date}_{issue:02d}00"
        if issue == 0 or adjustment_policy == "all_updates":
            solved = solve_version_milp(scenarios, data.price_yuan_per_kwh, soc, float(frozen["terminal_value"]), original)
            version = _normalize_theoretical_nonnegative(
                solved.plan.sort_values("slot")["planned_purchase_kwh"].to_numpy(float),
                "planned_purchase_kwh", data.price_yuan_per_kwh[start:],
                f"policy={policy};date={date};issue={issue}", normalization,
            )
            solvers.append({"policy": policy, "run_id": run_id, "date": date, "issue_hour": issue, "start_slot": start, **solved.metadata, **_validate_scenario(solved)})
        else:
            if original is None:
                raise AssertionError("No-adjustment policy lacks its 0:00 original plan")
            version = original[start:].copy()
            run_id += "_value_only"
        if issue == 0:
            original = version.copy()
        before = effective[:start].copy()
        effective[start:] = version
        prefix_error = float(np.nanmax(np.abs(before - effective[:start]))) if start else 0.0
        if start and (np.isnan(effective[:start]).any() or prefix_error > TOL):
            raise AssertionError("An executed plan prefix was retroactively changed")
        values = build_path_value_functions(scenarios, version, data.price_yuan_per_kwh[start:], float(frozen["soc_grid_kwh"]), float(frozen["terminal_value"]))
        for t in range(start, 144):
            plans.append({
                "policy": policy, "run_id": run_id, "date": date, "issue_hour": issue,
                "decision_datetime": f"{date} {issue:02d}:00:00", "slot": t, "time_label": data.time_labels[t],
                "original_plan_kwh": float(original[t]), "version_plan_kwh": float(version[t - start]),
                "delta_plus_vs_original_kwh": float(max(version[t - start] - original[t], 0.0)),
                "delta_minus_vs_original_kwh": float(max(original[t] - version[t - start], 0.0)),
                "executed_prefix_lock_error_kwh": prefix_error, "milp_solved": solved is not None,
            })
        infos.append({
            "policy": policy, "run_id": run_id, "date": date, "issue_hour": issue,
            "first_modifiable_slot": start, "scenario_count_requested": int(frozen["scenario_cap"]),
            "scenario_count_effective": scenarios.count, "scenario_source_dates": "|".join(scenarios.source_dates),
            "all_sources_prior": all(source < date for source in scenarios.source_dates),
            "forecast_visible": bool(forecast["forecast_visible_at_decision"].all()),
            "pv_information_causal": bool(forecast["pv_information_causal"].all()),
            "no_future_actual_load_used": bool(forecast["no_future_actual_load_used"].all()),
            "actual_pv_used_in_forecast": bool(forecast["pv_actual_observation_used"].any()),
            "value_function_plan_hash": hashlib.sha256(version.tobytes()).hexdigest().upper(),
            "value_runtime_seconds": values.runtime_seconds,
            "value_function_scope": "historical-path values averaged across causal joint scenarios; approximate future cost, not exact multistage stochastic DP",
        })
        for t in range(start, stop):
            local = t - start
            gap = float(data.load_kwh[di, t] - data.pv_kwh[di, t] - effective[t])
            charge, discharge, emergency, spill, next_soc = select_dp_action(
                gap, soc, float(data.price_yuan_per_kwh[t]), values.mean_value_yuan[local + 1], values.soc_grid_kwh
            )
            slots.append(_slot_row(policy, date, t, issue, run_id, forecast, original, effective, data, di, soc, charge, discharge, emergency, spill, next_soc))
            soc = float(next_soc)
    slot_frame = pd.DataFrame(slots)
    if len(slot_frame) != 144 or not np.array_equal(slot_frame.slot.to_numpy(int), np.arange(144)):
        raise AssertionError(f"Incomplete official-slot execution on {date}")
    daily = _daily_metrics(slot_frame)
    daily.update({
        "policy": policy, "date": date, "runtime_seconds": time.perf_counter() - day_started,
        "numeric_normalization_epsilon_kwh": NONNEGATIVE_NORMALIZATION_EPS_KWH,
        "numeric_normalization_count": normalization["count"],
        "numeric_normalization_max_abs_kwh": normalization["max_abs_kwh"],
        "numeric_normalization_max_cost_impact_yuan": normalization["max_cost_impact_yuan"],
        "numeric_normalization_events": json.dumps(normalization["events"], ensure_ascii=False),
    })
    return slot_frame, pd.DataFrame(plans), pd.DataFrame(solvers), pd.DataFrame(infos), daily


def _daily_metrics(frame: pd.DataFrame) -> dict:
    first, last = frame.iloc[0], frame.iloc[-1]
    c = frame.charge_rt_kwh
    b = frame.discharge_rt_kwh
    e = frame.emergency_purchase_kwh
    price = frame.price_yuan_per_kwh
    return {
        "soc_start_kwh": float(first.soc_start_kwh), "soc_end_kwh": float(last.soc_end_kwh),
        "contract_cost_yuan": float(frame.contract_cost_yuan.sum()), "emergency_cost_yuan": float(frame.emergency_cost_yuan.sum()),
        "actual_total_cost_yuan": float(frame.actual_total_cost_yuan.sum()), "emergency_purchase_kwh": float(e.sum()),
        "charge_kwh": float(c.sum()), "discharge_kwh": float(b.sum()), "spill_kwh": float(frame.spill_kwh.sum()),
        "conversion_loss_kwh": float(((1.0 - 0.9) * c + (1.0 / 0.9 - 1.0) * b).sum()),
        "upward_adjustment_kwh": float(frame.delta_plus_kwh.sum()), "downward_adjustment_kwh": float(frame.delta_minus_kwh.sum()),
        "emergency_low_price_kwh": float(e[price < 0.6].sum()),
        "emergency_mid_price_kwh": float(e[(price >= 0.6) & (price < 1.0)].sum()),
        "emergency_high_price_kwh": float(e[price >= 1.0].sum()),
        "max_balance_residual_kwh": float(frame.balance_residual_kwh.abs().max()),
        "minimum_actual_soc_kwh": float(frame[["soc_start_kwh", "soc_end_kwh"]].min().min()),
        "maximum_actual_soc_kwh": float(frame[["soc_start_kwh", "soc_end_kwh"]].max().max()),
        "soc_violation_count": int(((frame.soc_end_kwh < SOC_MIN_KWH - TOL) | (frame.soc_end_kwh > SOC_MAX_KWH + TOL)).sum()),
        "power_violation_count": int(((frame.charge_rt_kwh > ENERGY_MAX_KWH + TOL) | (frame.discharge_rt_kwh > ENERGY_MAX_KWH + TOL)).sum()),
        "simultaneous_charge_discharge_count": int(((frame.charge_rt_kwh > TOL) & (frame.discharge_rt_kwh > TOL)).sum()),
        "emergency_and_charge_count": int(((frame.emergency_purchase_kwh > TOL) & (frame.charge_rt_kwh > TOL)).sum()),
    }


def _save_day(policy: str, date: str, slot: pd.DataFrame, plans: pd.DataFrame, solvers: pd.DataFrame, infos: pd.DataFrame, daily: dict, config_hash: str, initialization: dict) -> None:
    directory = _day_dir(policy, date)
    _write_csv(directory / "slots.csv", slot)
    _write_csv(directory / "plan_versions.csv", plans)
    _write_csv(directory / "solver_log.csv", solvers)
    _write_csv(directory / "information_audit.csv", infos)
    _write_json(directory / "daily.json", daily)
    _write_json(_checkpoint_path(policy), {
        "policy": policy, "config_sha256": config_hash, "completed_through": date,
        "next_date": (pd.Timestamp(date) + pd.Timedelta(days=1)).date().isoformat(),
        "next_soc_kwh": daily["soc_end_kwh"], "initialization": initialization,
        "checkpoint_written_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    })


def run_policy(policy: str, adjustment_policy: str, data, cache: dict, frozen: dict, config_hash: str, initialization: dict) -> None:
    start_date, soc = _load_checkpoint(policy, config_hash, initialization)
    if start_date > pd.Timestamp(FORMAL_END):
        print(f"RESUME {policy}: already complete", flush=True)
        return
    all_days = pd.date_range(start_date, FORMAL_END, freq="D")
    total = len(pd.date_range(FORMAL_START, FORMAL_END, freq="D"))
    for day_no, day in enumerate(all_days, start=(start_date - pd.Timestamp(FORMAL_START)).days + 1):
        date = day.date().isoformat()
        try:
            slot, plans, solvers, infos, daily = _run_day(policy, adjustment_policy, date, soc, data, cache, frozen)
            _save_day(policy, date, slot, plans, solvers, infos, daily, config_hash, initialization)
            soc = float(daily["soc_end_kwh"])
            _write_json(STAGE4 / "progress.json", {
                "status": "running", "active_policy": policy, "completed_day": day_no,
                "total_days_per_policy": total, "completed_date": date, "next_soc_kwh": soc,
                "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            })
            print(f"PROGRESS policy={policy} day={day_no}/{total} date={date} cost={daily['actual_total_cost_yuan']:.3f} soc_end={soc:.3f} runtime={daily['runtime_seconds']:.2f}s", flush=True)
        except Exception as exc:
            _write_json(STAGE4 / "failed_run.json", {
                "policy": policy, "date": date, "error": repr(exc), "traceback": traceback.format_exc(),
                "instruction": "Stopped without fallback, as required. Prior completed-day checkpoints remain intact.",
            })
            raise


def _collect_policy(policy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(FORMAL_START, FORMAL_END, freq="D")
    parts = []
    for date in dates:
        directory = _day_dir(policy, date.date().isoformat())
        if not directory.exists():
            raise RuntimeError(f"Missing completed day for {policy}: {date.date().isoformat()}")
        parts.append((
            pd.read_csv(directory / "slots.csv"), pd.read_csv(directory / "plan_versions.csv"),
            pd.read_csv(directory / "solver_log.csv"), pd.read_csv(directory / "information_audit.csv"),
            json.loads((directory / "daily.json").read_text(encoding="utf-8")),
        ))
    return (
        pd.concat([x[0] for x in parts], ignore_index=True), pd.concat([x[1] for x in parts], ignore_index=True),
        pd.concat([x[2] for x in parts], ignore_index=True), pd.concat([x[3] for x in parts], ignore_index=True),
        pd.DataFrame([x[4] for x in parts]),
    )


def _paired_ablation(main_slots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for date, day in main_slots.groupby("date", sort=True):
        day = day.sort_values("slot")
        soc = float(day.iloc[0].soc_start_kwh)
        for _, source in day.iterrows():
            gap = float(source.load_actual_kwh - source.pv_actual_kwh - source.effective_purchase_kwh)
            charge, discharge, emergency, spill, next_soc = greedy_action(gap, soc)
            emergency_cost = float(EMERGENCY_MULTIPLIER * source.price_yuan_per_kwh * emergency)
            balance = source.effective_purchase_kwh + source.pv_actual_kwh + discharge + emergency - source.load_actual_kwh - charge - spill
            rows.append({
                "policy": "C_paired_greedy_fixed_A_plan_versions", "date": date, "slot": int(source.slot),
                "time_label": source.time_label, "price_yuan_per_kwh": source.price_yuan_per_kwh,
                "original_plan_kwh": source.original_plan_kwh, "effective_purchase_kwh": source.effective_purchase_kwh,
                "delta_plus_kwh": source.delta_plus_kwh, "delta_minus_kwh": source.delta_minus_kwh,
                "charge_rt_kwh": charge, "discharge_rt_kwh": discharge, "emergency_purchase_kwh": emergency,
                "spill_kwh": spill, "soc_start_kwh": soc, "soc_end_kwh": next_soc,
                "contract_cost_yuan": source.contract_cost_yuan, "emergency_cost_yuan": emergency_cost,
                "actual_total_cost_yuan": source.contract_cost_yuan + emergency_cost, "balance_residual_kwh": balance,
                "comparison_scope": "conditional_daily_pair_same_A_plan_versions_and_same_A_daily_initial_SOC_not_independent_cross_day_strategy",
            })
            soc = float(next_soc)
    paired = pd.DataFrame(rows)
    daily_rows = []
    for date, day in paired.groupby("date", sort=True):
        metrics = _daily_metrics(day)
        metrics.update({"policy": "C_paired_greedy_fixed_A_plan_versions", "date": date})
        daily_rows.append(metrics)
    daily = pd.DataFrame(daily_rows)
    main_daily_rows = []
    for date, day in main_slots.groupby("date", sort=True):
        metrics = _daily_metrics(day)
        metrics.update({"policy": "A_all_updates_dynamic_value", "date": date})
        main_daily_rows.append(metrics)
    main_daily = pd.DataFrame(main_daily_rows)
    summary = {
        "scope": "conditional daily paired executor ablation; plan-version trajectory and daily initial SOC fixed to A",
        "dynamic_cost_yuan": float(main_slots.actual_total_cost_yuan.sum()), "greedy_cost_yuan": float(paired.actual_total_cost_yuan.sum()),
        "dynamic_minus_greedy_cost_yuan": float(main_slots.actual_total_cost_yuan.sum() - paired.actual_total_cost_yuan.sum()),
        "dynamic_emergency_kwh": float(main_slots.emergency_purchase_kwh.sum()), "greedy_emergency_kwh": float(paired.emergency_purchase_kwh.sum()),
        "dynamic_minus_greedy_emergency_kwh": float(main_slots.emergency_purchase_kwh.sum() - paired.emergency_purchase_kwh.sum()),
        "dynamic_terminal_soc_kwh": float(main_daily.iloc[-1].soc_end_kwh), "greedy_terminal_soc_kwh": float(daily.iloc[-1].soc_end_kwh),
    }
    return paired, daily, summary


def _summary(policy: str, slots: pd.DataFrame, daily: pd.DataFrame, solver: pd.DataFrame, info: pd.DataFrame) -> dict:
    e = slots.emergency_purchase_kwh
    price = slots.price_yuan_per_kwh
    adjustment = slots.effective_purchase_kwh - slots.original_plan_kwh
    unchanged = adjustment.abs() <= TOL
    expected = slots.soc_start_kwh + 0.9 * slots.charge_rt_kwh - slots.discharge_rt_kwh / 0.9
    cross = daily.soc_start_kwh.iloc[1:].to_numpy() - daily.soc_end_kwh.iloc[:-1].to_numpy()
    return {
        "policy": policy, "formal_start": FORMAL_START, "formal_end": FORMAL_END, "formal_days": len(daily),
        "contract_cost_yuan": float(slots.contract_cost_yuan.sum()), "emergency_cost_yuan": float(slots.emergency_cost_yuan.sum()),
        "actual_total_cost_yuan": float(slots.actual_total_cost_yuan.sum()), "emergency_purchase_kwh": float(e.sum()),
        "max_daily_emergency_purchase_kwh": float(daily.emergency_purchase_kwh.max()),
        "weighted_emergency_price_yuan_per_kwh": float(slots.emergency_cost_yuan.sum() / e.sum()) if e.sum() > TOL else np.nan,
        "emergency_low_price_kwh": float(e[price < 0.6].sum()), "emergency_mid_price_kwh": float(e[(price >= 0.6) & (price < 1.0)].sum()), "emergency_high_price_kwh": float(e[price >= 1.0].sum()),
        "mean_daily_end_soc_kwh": float(daily.soc_end_kwh.mean()), "minimum_daily_end_soc_kwh": float(daily.soc_end_kwh.min()),
        "minimum_actual_soc_kwh": float(slots[["soc_start_kwh", "soc_end_kwh"]].min().min()), "maximum_actual_soc_kwh": float(slots[["soc_start_kwh", "soc_end_kwh"]].max().max()),
        "terminal_soc_kwh": float(daily.soc_end_kwh.iloc[-1]), "charge_kwh": float(slots.charge_rt_kwh.sum()),
        "discharge_kwh": float(slots.discharge_rt_kwh.sum()), "spill_kwh": float(slots.spill_kwh.sum()),
        "conversion_loss_kwh": float(((1.0 - 0.9) * slots.charge_rt_kwh + (1.0 / 0.9 - 1.0) * slots.discharge_rt_kwh).sum()),
        "upward_adjustment_kwh": float(slots.delta_plus_kwh.sum()), "downward_adjustment_kwh": float(slots.delta_minus_kwh.sum()),
        "upward_adjustment_slot_count": int((adjustment > TOL).sum()),
        "downward_adjustment_slot_count": int((adjustment < -TOL).sum()),
        "unchanged_slot_count": int(unchanged.sum()),
        "unchanged_effective_purchase_kwh": float(slots.loc[unchanged, "effective_purchase_kwh"].sum()),
        "milp_calls": int(len(solver)), "time_limit_feasible_count": int((solver.solution_class != "certified_to_requested_gap").sum()),
        "certified_to_requested_gap_count": int((solver.solution_class == "certified_to_requested_gap").sum()),
        "mean_mip_gap": float(solver.relative_gap.mean()), "max_mip_gap": float(solver.relative_gap.max()),
        "mean_solve_seconds": float(solver.solve_time_seconds.mean()), "total_solve_seconds": float(solver.solve_time_seconds.sum()),
        "total_value_seconds": float(info.value_runtime_seconds.sum()), "total_recorded_runtime_seconds": float(daily.runtime_seconds.sum()),
        "numeric_normalization_count": int(daily.numeric_normalization_count.fillna(0).sum()) if "numeric_normalization_count" in daily else 0,
        "numeric_normalization_max_abs_kwh": float(daily.numeric_normalization_max_abs_kwh.fillna(0).max()) if "numeric_normalization_max_abs_kwh" in daily else 0.0,
        "numeric_normalization_max_cost_impact_yuan": float(daily.numeric_normalization_max_cost_impact_yuan.fillna(0).max()) if "numeric_normalization_max_cost_impact_yuan" in daily else 0.0,
        "max_balance_residual_kwh": float(slots.balance_residual_kwh.abs().max()),
        "max_soc_transition_residual_kwh": float(np.max(np.abs(slots.soc_end_kwh - expected))),
        "cross_day_soc_max_error_kwh": float(np.max(np.abs(cross))) if len(cross) else 0.0,
        "soc_violation_count": int(((slots.soc_end_kwh < SOC_MIN_KWH - TOL) | (slots.soc_end_kwh > SOC_MAX_KWH + TOL)).sum()),
        "power_violation_count": int(((slots.charge_rt_kwh > ENERGY_MAX_KWH + TOL) | (slots.discharge_rt_kwh > ENERGY_MAX_KWH + TOL)).sum()),
        "simultaneous_charge_discharge_count": int(((slots.charge_rt_kwh > TOL) & (slots.discharge_rt_kwh > TOL)).sum()),
        "emergency_and_charge_count": int(((slots.emergency_purchase_kwh > TOL) & (slots.charge_rt_kwh > TOL)).sum()),
        "information_violation_count": int((~info.all_sources_prior).sum() + (~info.forecast_visible).sum() + (~info.pv_information_causal).sum() + (~info.no_future_actual_load_used).sum() + info.actual_pv_used_in_forecast.sum()),
    }


def _monthly(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["month"] = pd.to_datetime(daily.date).dt.to_period("M").astype(str)
    return daily.groupby(["policy", "month"], as_index=False).agg(
        days=("date", "count"), contract_cost_yuan=("contract_cost_yuan", "sum"), emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        actual_total_cost_yuan=("actual_total_cost_yuan", "sum"), emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
        upward_adjustment_kwh=("upward_adjustment_kwh", "sum"), downward_adjustment_kwh=("downward_adjustment_kwh", "sum"),
        charge_kwh=("charge_kwh", "sum"), discharge_kwh=("discharge_kwh", "sum"), spill_kwh=("spill_kwh", "sum"),
        soc_start_kwh=("soc_start_kwh", "first"), soc_end_kwh=("soc_end_kwh", "last"), mean_daily_end_soc_kwh=("soc_end_kwh", "mean"),
    )


def _finalize(frozen: dict, initializations: dict, protected_before: dict) -> None:
    policy_data = {policy: _collect_policy(policy) for policy in POLICIES}
    slots = pd.concat([x[0] for x in policy_data.values()], ignore_index=True)
    plans = pd.concat([x[1] for x in policy_data.values()], ignore_index=True)
    solver = pd.concat([x[2] for x in policy_data.values()], ignore_index=True)
    info = pd.concat([x[3] for x in policy_data.values()], ignore_index=True)
    daily = pd.concat([x[4] for x in policy_data.values()], ignore_index=True)
    paired, paired_daily, paired_summary = _paired_ablation(policy_data["A_all_updates_dynamic_value"][0])
    summaries = pd.DataFrame([
        _summary(policy, policy_data[policy][0], policy_data[policy][4], policy_data[policy][2], policy_data[policy][3])
        for policy in POLICIES
    ])
    monthly = _monthly(daily)
    _write_csv(STAGE4 / "q3_v2_full_year_comparison.csv", summaries)
    _write_csv(STAGE4 / "q3_v2_monthly_comparison.csv", monthly)
    _write_csv(STAGE4 / "q3_v2_daily_results.csv", daily)
    _write_csv(STAGE4 / "q3_v2_slot_results.csv", slots)
    _write_csv(STAGE4 / "q3_v2_plan_versions.csv", plans)
    _write_csv(STAGE4 / "q3_v2_solver_log.csv", solver)
    _write_csv(STAGE4 / "q3_v2_information_boundary_audit.csv", info)
    _write_csv(STAGE4 / "q3_v2_paired_ablation_slots.csv", paired)
    _write_csv(STAGE4 / "q3_v2_paired_ablation_daily.csv", paired_daily)
    _write_json(STAGE4 / "q3_v2_paired_ablation_summary.json", paired_summary)
    special = STAGE4 / "special_dates"
    for date in SPECIAL_DATES:
        directory = special / date
        _write_csv(directory / "slot_results.csv", slots[slots.date == date])
        _write_csv(directory / "plan_versions.csv", plans[plans.date == date])
        _write_csv(directory / "daily_results.csv", daily[daily.date == date])
    after = protected_manifest()
    _write_json(STAGE4 / "protected_hashes_before_stage4.json", protected_before)
    _write_json(STAGE4 / "protected_hashes_after_stage4.json", after)
    if protected_before != after:
        raise RuntimeError("A protected Q1/Q2V2/legacy-Q3 file changed during Stage 4")
    validation = ["Q3 V2 Stage 4 validation (causal shadow evaluation; no result3.xlsx)", f"Protected manifest unchanged: {protected_before == after}"]
    for _, row in summaries.iterrows():
        passed = all(float(row[k]) <= TOL for k in ("max_balance_residual_kwh", "max_soc_transition_residual_kwh", "cross_day_soc_max_error_kwh")) and all(int(row[k]) == 0 for k in ("soc_violation_count", "power_violation_count", "simultaneous_charge_discharge_count", "emergency_and_charge_count", "information_violation_count"))
        validation.append(f"{row.policy}: passed={passed}; balance={row.max_balance_residual_kwh:.3e}; cross_day={row.cross_day_soc_max_error_kwh:.3e}; information_violations={int(row.information_violation_count)}")
    _write_json(STAGE4 / "progress.json", {"status": "complete", "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), "result3_generated": False})
    (STAGE4 / "q3_v2_full_year_validation.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")
    analysis = f"""# 问题三 V2 阶段4全年影子评价说明

评价期为 {FORMAL_START} 至 {FORMAL_END}（334天）。配置直接读取阶段3冻结文件，未使用2—12月结果调参；实际账单仅为合同结算费加5倍电价紧急购电费，终端库存价值未抵扣账单。

策略A允许6:00、12:00、18:00调整购电；策略B只锁定0:00计划，但仍在同一信息边界内访问新预报并刷新储能未来价值。两者从各自合法的1月连续预热末SOC承接，因而期末SOC差异将随费用一并报告。

条件配对消融固定A的计划版本轨迹和日初SOC，只比较动态价值与贪心执行；它不是独立跨日闭环策略比较。

完整数值见 `q3_v2_full_year_comparison.csv`、逐月/逐日/逐槽文件及 `q3_v2_paired_ablation_summary.json`。本阶段未生成或修改 `result3.xlsx`。
"""
    (DOC / "q3_v2_full_year_analysis.md").write_text(analysis, encoding="utf-8")


def main() -> None:
    STAGE4.mkdir(parents=True, exist_ok=True)
    frozen = _load_config()
    frozen_path = STAGE3 / "q3_v2_frozen_config.json"
    config_hash = _sha256(frozen_path)
    initializations = _policy_initialization()
    protected_before = protected_manifest()
    protocol = {
        "stage": "Q3 V2 Stage 4", "formal_period": [FORMAL_START, FORMAL_END], "formal_days": 334,
        "frozen_config_source": str(frozen_path.relative_to(ROOT)), "frozen_config_sha256": config_hash,
        "frozen_parameters": frozen, "policies": POLICIES, "initializations": initializations,
        "paired_ablation": "fixed A plan-version trajectory and same A daily initial SOC; daily conditional comparison only",
        "result3_generated": False, "no_fallback_if_no_feasible_milp": True,
    }
    _write_json(STAGE4 / "q3_v2_stage4_run_protocol.json", protocol)
    data = load_inputs()
    actual = _actual_long(data)
    pv = load_q3_pv_forecast_records()
    if len(data.dates) < 365 or pd.Timestamp(data.dates.max()) < pd.Timestamp(FORMAL_END):
        raise RuntimeError("Attachment-2 data do not cover the required 2025-12-31 endpoint")
    print("PREPARE building causal forecast replay cache for 2025-01-01..2025-12-31", flush=True)
    cache = build_forecast_cache(actual, pv, frozen["load_method"], "2025-01-01", FORMAL_END)
    print(f"PREPARE cache complete: {len(cache)} forecast versions", flush=True)
    for policy, adjustment in POLICIES.items():
        run_policy(policy, adjustment, data, cache, frozen, config_hash, initializations[policy])
    _finalize(frozen, initializations, protected_before)
    print("COMPLETE Stage 4 finalized; result3.xlsx was not generated or modified", flush=True)


if __name__ == "__main__":
    main()
