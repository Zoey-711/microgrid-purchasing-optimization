"""Greedy and historical-path-value DP executors for Q2-v2."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ACTION_OBJECTIVE_TIE_TOL_YUAN,
    EMERGENCY_PRICE_MULTIPLIER,
    ENERGY_MAX_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    TOL,
)
from .scenarios import JointScenarios


@dataclass(frozen=True)
class PathValueResult:
    soc_grid_kwh: np.ndarray
    path_values_yuan: np.ndarray
    mean_value_yuan: np.ndarray
    runtime_seconds: float
    terminal_value_yuan_per_kwh: float


def make_soc_grid(step_kwh: float) -> np.ndarray:
    if step_kwh <= 0:
        raise ValueError("SOC grid step must be positive")
    count = int(round((SOC_MAX_KWH - SOC_MIN_KWH) / step_kwh))
    grid = SOC_MIN_KWH + step_kwh * np.arange(count + 1, dtype=float)
    if abs(grid[-1] - SOC_MAX_KWH) > 1e-9:
        grid = np.unique(np.append(grid[grid < SOC_MAX_KWH], SOC_MAX_KWH))
    return grid


def _transitions(net_gap_kwh: float, soc_kwh: float, grid: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """Return charge, discharge, emergency, spill, next SOC candidates."""
    candidates: list[tuple[float, float, float, float, float]] = []
    if net_gap_kwh > TOL:
        max_discharge = min(
            net_gap_kwh,
            ENERGY_MAX_KWH,
            ETA_DISCHARGE * max(0.0, soc_kwh - SOC_MIN_KWH),
        )
        discharge_values = {0.0, float(max_discharge)}
        low_next = soc_kwh - max_discharge / ETA_DISCHARGE
        for next_soc in grid[(grid >= low_next - TOL) & (grid <= soc_kwh + TOL)]:
            discharge_values.add(float(ETA_DISCHARGE * (soc_kwh - next_soc)))
        for discharge in sorted(discharge_values):
            discharge = min(max(discharge, 0.0), max_discharge)
            next_soc = soc_kwh - discharge / ETA_DISCHARGE
            candidates.append((0.0, discharge, net_gap_kwh - discharge, 0.0, next_soc))
    elif net_gap_kwh < -TOL:
        surplus = -net_gap_kwh
        max_charge = min(
            surplus,
            ENERGY_MAX_KWH,
            max(0.0, SOC_MAX_KWH - soc_kwh) / ETA_CHARGE,
        )
        charge_values = {0.0, float(max_charge)}
        high_next = soc_kwh + ETA_CHARGE * max_charge
        for next_soc in grid[(grid >= soc_kwh - TOL) & (grid <= high_next + TOL)]:
            charge_values.add(float((next_soc - soc_kwh) / ETA_CHARGE))
        for charge in sorted(charge_values):
            charge = min(max(charge, 0.0), max_charge)
            next_soc = soc_kwh + ETA_CHARGE * charge
            candidates.append((charge, 0.0, 0.0, surplus - charge, next_soc))
    else:
        candidates.append((0.0, 0.0, 0.0, 0.0, soc_kwh))
    # Remove numerical duplicates while keeping exact boundary and gap actions.
    unique: dict[tuple[float, ...], tuple[float, ...]] = {}
    for item in candidates:
        key = tuple(round(x, 10) for x in item)
        unique[key] = item
    return list(unique.values())


def build_path_value_functions(
    scenarios: JointScenarios,
    planned_purchase_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    grid_step_kwh: float,
    terminal_value_yuan_per_kwh: float,
) -> PathValueResult:
    plan = np.asarray(planned_purchase_kwh, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    n_slots = len(plan)
    if scenarios.load_kwh.shape[1] != n_slots or len(price) != n_slots:
        raise ValueError("Plan, price and scenarios must share the same horizon")
    grid = make_soc_grid(grid_step_kwh)
    started = time.perf_counter()
    values = np.empty((scenarios.count, n_slots + 1, len(grid)), dtype=float)
    values[:, n_slots, :] = -terminal_value_yuan_per_kwh * grid[None, :]
    soc_now = grid[:, None]
    soc_next = grid[None, :]
    discharge_matrix = ETA_DISCHARGE * (soc_now - soc_next)
    charge_matrix = (soc_next - soc_now) / ETA_CHARGE
    for w in range(scenarios.count):
        path_gap = scenarios.load_kwh[w] - scenarios.pv_kwh[w] - plan
        for t in range(n_slots - 1, -1, -1):
            next_value = values[w, t + 1]
            gap = float(path_gap[t])
            if gap > TOL:
                max_discharge = np.minimum.reduce(
                    [
                        np.full(len(grid), gap),
                        np.full(len(grid), ENERGY_MAX_KWH),
                        ETA_DISCHARGE * np.maximum(0.0, grid - SOC_MIN_KWH),
                    ]
                )
                valid = (discharge_matrix >= -TOL) & (
                    discharge_matrix <= max_discharge[:, None] + TOL
                )
                score = (
                    EMERGENCY_PRICE_MULTIPLIER
                    * price[t]
                    * (gap - np.clip(discharge_matrix, 0.0, None))
                    + next_value[None, :]
                )
                score[~valid] = np.inf
                best = np.min(score, axis=1)
                exact_next = grid - max_discharge / ETA_DISCHARGE
                exact_score = (
                    EMERGENCY_PRICE_MULTIPLIER * price[t] * (gap - max_discharge)
                    + np.interp(exact_next, grid, next_value)
                )
                values[w, t] = np.minimum(best, exact_score)
            elif gap < -TOL:
                surplus = -gap
                max_charge = np.minimum.reduce(
                    [
                        np.full(len(grid), surplus),
                        np.full(len(grid), ENERGY_MAX_KWH),
                        np.maximum(0.0, SOC_MAX_KWH - grid) / ETA_CHARGE,
                    ]
                )
                valid = (charge_matrix >= -TOL) & (charge_matrix <= max_charge[:, None] + TOL)
                score = np.broadcast_to(next_value[None, :], valid.shape).copy()
                score[~valid] = np.inf
                best = np.min(score, axis=1)
                exact_next = grid + ETA_CHARGE * max_charge
                exact_score = np.interp(exact_next, grid, next_value)
                values[w, t] = np.minimum(best, exact_score)
            else:
                values[w, t] = next_value
    return PathValueResult(
        soc_grid_kwh=grid,
        path_values_yuan=values,
        mean_value_yuan=np.tensordot(scenarios.weights, values, axes=(0, 0)),
        runtime_seconds=time.perf_counter() - started,
        terminal_value_yuan_per_kwh=terminal_value_yuan_per_kwh,
    )


def select_dp_action(
    net_gap_kwh: float,
    soc_kwh: float,
    price_yuan_per_kwh: float,
    next_mean_value: np.ndarray,
    soc_grid_kwh: np.ndarray,
) -> tuple[float, float, float, float, float]:
    best = None
    best_score = None
    for action in _transitions(net_gap_kwh, soc_kwh, soc_grid_kwh):
        charge, discharge, emergency, spill, next_soc = action
        score = (
            EMERGENCY_PRICE_MULTIPLIER * price_yuan_per_kwh * emergency
            + float(np.interp(next_soc, soc_grid_kwh, next_mean_value))
        )
        if _prefer_candidate(score, action, best_score, best):
            best_score = score
            best = action
    if best is None:
        raise RuntimeError("No feasible DP action")
    return best


def greedy_action(net_gap_kwh: float, soc_kwh: float) -> tuple[float, float, float, float, float]:
    if net_gap_kwh > TOL:
        discharge = min(
            net_gap_kwh,
            ENERGY_MAX_KWH,
            ETA_DISCHARGE * max(0.0, soc_kwh - SOC_MIN_KWH),
        )
        return 0.0, discharge, net_gap_kwh - discharge, 0.0, soc_kwh - discharge / ETA_DISCHARGE
    if net_gap_kwh < -TOL:
        surplus = -net_gap_kwh
        charge = min(
            surplus,
            ENERGY_MAX_KWH,
            max(0.0, SOC_MAX_KWH - soc_kwh) / ETA_CHARGE,
        )
        return charge, 0.0, 0.0, surplus - charge, soc_kwh + ETA_CHARGE * charge
    return 0.0, 0.0, 0.0, 0.0, soc_kwh


def threshold_action(
    net_gap_kwh: float, soc_kwh: float, retention_soc_kwh: float
) -> tuple[float, float, float, float, float]:
    """Greedy action with an economic discharge floor above the physical floor."""
    if not SOC_MIN_KWH <= retention_soc_kwh <= SOC_MAX_KWH:
        raise ValueError("Retention SOC must lie within physical bounds")
    if net_gap_kwh > TOL:
        discharge = min(
            net_gap_kwh,
            ENERGY_MAX_KWH,
            ETA_DISCHARGE * max(0.0, soc_kwh - retention_soc_kwh),
        )
        return 0.0, discharge, net_gap_kwh - discharge, 0.0, soc_kwh - discharge / ETA_DISCHARGE
    return greedy_action(net_gap_kwh, soc_kwh)


def execute_day(
    date: str,
    time_labels: tuple[str, ...],
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    planned_purchase_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    initial_soc_kwh: float,
    executor: str,
    path_value: PathValueResult | None = None,
    retention_soc_kwh: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    n_slots = len(planned_purchase_kwh)
    soc = float(initial_soc_kwh)
    rows = []
    for t in range(n_slots):
        net_gap = float(load_actual_kwh[t] - pv_actual_kwh[t] - planned_purchase_kwh[t])
        if executor == "greedy":
            action = greedy_action(net_gap, soc)
        elif executor == "threshold":
            if retention_soc_kwh is None:
                raise ValueError("Threshold executor requires retention_soc_kwh")
            action = threshold_action(net_gap, soc, retention_soc_kwh)
        elif executor == "dp":
            if path_value is None:
                raise ValueError("DP executor requires path values")
            action = select_dp_action(
                net_gap,
                soc,
                float(price_yuan_per_kwh[t]),
                path_value.mean_value_yuan[t + 1],
                path_value.soc_grid_kwh,
            )
        else:
            raise ValueError(f"Unknown executor: {executor}")
        charge, discharge, emergency, spill, next_soc = action
        plan_cost = float(price_yuan_per_kwh[t] * planned_purchase_kwh[t])
        emergency_cost = float(EMERGENCY_PRICE_MULTIPLIER * price_yuan_per_kwh[t] * emergency)
        balance = (
            planned_purchase_kwh[t] + pv_actual_kwh[t] + discharge + emergency
            - load_actual_kwh[t] - charge - spill
        )
        rows.append(
            {
                "date": date,
                "slot": t,
                "time_label": time_labels[t],
                "price_yuan_per_kwh": float(price_yuan_per_kwh[t]),
                "load_actual_kwh": float(load_actual_kwh[t]),
                "pv_actual_kwh": float(pv_actual_kwh[t]),
                "planned_purchase_kwh": float(planned_purchase_kwh[t]),
                "net_gap_before_storage_kwh": net_gap,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "emergency_purchase_kwh": emergency,
                "spill_kwh": spill,
                "soc_start_kwh": soc,
                "soc_end_kwh": next_soc,
                "plan_cost_yuan": plan_cost,
                "emergency_cost_yuan": emergency_cost,
                "actual_cost_yuan": plan_cost + emergency_cost,
                "balance_residual_kwh": balance,
                "executor": executor,
            }
        )
        soc = next_soc
    frame = pd.DataFrame(rows)
    metadata = {
        "date": date,
        "executor": executor,
        "soc_start_kwh": initial_soc_kwh,
        "soc_end_kwh": soc,
        "plan_cost_yuan": float(frame["plan_cost_yuan"].sum()),
        "emergency_cost_yuan": float(frame["emergency_cost_yuan"].sum()),
        "actual_total_cost_yuan": float(frame["actual_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(frame["emergency_purchase_kwh"].sum()),
        "charge_kwh": float(frame["charge_kwh"].sum()),
        "discharge_kwh": float(frame["discharge_kwh"].sum()),
        "spill_kwh": float(frame["spill_kwh"].sum()),
        "conversion_loss_kwh": float(
            ((1.0 - ETA_CHARGE) * frame["charge_kwh"] + (1.0 / ETA_DISCHARGE - 1.0) * frame["discharge_kwh"]).sum()
        ),
        "retention_soc_kwh": retention_soc_kwh,
        "runtime_seconds": time.perf_counter() - started,
    }
    return frame, metadata
def _prefer_candidate(
    score: float,
    action: tuple[float, float, float, float, float],
    best_score: float | None,
    best_action: tuple[float, float, float, float, float] | None,
) -> bool:
    """Compare actions without changing the monetary primary objective.

    Within a fixed absolute numerical tolerance, prefer lower storage
    throughput; remaining ties use emergency, spill and next SOC solely to
    make the result deterministic.
    """
    if best_score is None or best_action is None:
        return True
    if score < best_score - ACTION_OBJECTIVE_TIE_TOL_YUAN:
        return True
    if abs(score - best_score) > ACTION_OBJECTIVE_TIE_TOL_YUAN:
        return False
    charge, discharge, emergency, spill, next_soc = action
    bc, bd, be, bs, bn = best_action
    key = (charge + discharge, emergency, spill, next_soc, charge, discharge)
    best_key = (bc + bd, be, bs, bn, bc, bd)
    return key < best_key
