"""Versioned joint-residual scenario MILP for a Q3 remaining-day suffix."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from .config import (
    EMERGENCY_MULTIPLIER,
    ENERGY_MAX_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    MIP_RANDOM_SEED,
    MIP_REL_GAP,
    MIP_TIME_LIMIT_SECONDS,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    TOL,
)


@dataclass(frozen=True)
class VersionScenarios:
    source_dates: tuple[str, ...]
    weights: np.ndarray
    load_kwh: np.ndarray
    pv_kwh: np.ndarray
    start_slot: int

    @property
    def count(self) -> int:
        return len(self.source_dates)


@dataclass(frozen=True)
class RollingMilpResult:
    plan: pd.DataFrame
    scenario_dispatch: pd.DataFrame
    metadata: dict


def _solver():
    for name in ("appsi_highs", "highs"):
        solver = pyo.SolverFactory(name)
        if solver is not None and solver.available(exception_flag=False):
            solver.options["time_limit"] = MIP_TIME_LIMIT_SECONDS
            solver.options["mip_rel_gap"] = MIP_REL_GAP
            solver.options["random_seed"] = MIP_RANDOM_SEED
            solver.options["threads"] = 1
            return solver, name
    raise RuntimeError("HiGHS is unavailable")


def solve_version_milp(
    scenarios: VersionScenarios,
    price_full: np.ndarray,
    initial_soc_kwh: float,
    terminal_value_yuan_per_kwh: float,
    original_plan_full: np.ndarray | None,
) -> RollingMilpResult:
    start = scenarios.start_slot
    n = scenarios.load_kwh.shape[1]
    slots = np.arange(start, start + n, dtype=int)
    price = np.asarray(price_full, dtype=float)[slots]
    if scenarios.count < 1 or scenarios.pv_kwh.shape != scenarios.load_kwh.shape:
        raise ValueError("At least one aligned joint scenario is required")
    if not SOC_MIN_KWH <= initial_soc_kwh <= SOC_MAX_KWH:
        raise ValueError("Initial SOC is outside physical bounds")
    original = None if original_plan_full is None else np.asarray(original_plan_full, dtype=float)[slots]

    model = pyo.ConcreteModel("q3_v2_versioned_scenario_milp")
    model.W = pyo.RangeSet(0, scenarios.count - 1)
    model.T = pyo.RangeSet(0, n - 1)
    model.S = pyo.RangeSet(0, n)
    purchase_ub = {j: float(np.max(scenarios.load_kwh[:, j]) + ENERGY_MAX_KWH) for j in range(n)}
    model.purchase = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=lambda m, j: (0.0, purchase_ub[j]))
    model.charge = pyo.Var(model.W, model.T, domain=pyo.NonNegativeReals, bounds=(0.0, ENERGY_MAX_KWH))
    model.discharge = pyo.Var(model.W, model.T, domain=pyo.NonNegativeReals, bounds=(0.0, ENERGY_MAX_KWH))
    model.emergency = pyo.Var(model.W, model.T, domain=pyo.NonNegativeReals)
    model.spill = pyo.Var(model.W, model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.W, model.S, bounds=(SOC_MIN_KWH, SOC_MAX_KWH))
    model.charge_mode = pyo.Var(model.W, model.T, domain=pyo.Binary)
    if original is not None:
        model.up = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.down = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.adjustment = pyo.Constraint(model.T, rule=lambda m, j: m.purchase[j] == float(original[j]) + m.up[j] - m.down[j])
        model.down_limit = pyo.Constraint(model.T, rule=lambda m, j: m.down[j] <= float(original[j]))

    model.balance = pyo.Constraint(
        model.W,
        model.T,
        rule=lambda m, w, j: m.purchase[j] + float(scenarios.pv_kwh[w, j]) + m.discharge[w, j] + m.emergency[w, j]
        == float(scenarios.load_kwh[w, j]) + m.charge[w, j] + m.spill[w, j],
    )
    model.transition = pyo.Constraint(
        model.W,
        model.T,
        rule=lambda m, w, j: m.soc[w, j + 1] == m.soc[w, j] + ETA_CHARGE * m.charge[w, j] - m.discharge[w, j] / ETA_DISCHARGE,
    )
    model.initial = pyo.Constraint(model.W, rule=lambda m, w: m.soc[w, 0] == float(initial_soc_kwh))
    model.charge_exclusive = pyo.Constraint(model.W, model.T, rule=lambda m, w, j: m.charge[w, j] <= ENERGY_MAX_KWH * m.charge_mode[w, j])
    model.discharge_exclusive = pyo.Constraint(model.W, model.T, rule=lambda m, w, j: m.discharge[w, j] <= ENERGY_MAX_KWH * (1 - m.charge_mode[w, j]))
    model.emergency_no_charge = pyo.Constraint(
        model.W,
        model.T,
        rule=lambda m, w, j: m.emergency[w, j] <= float(scenarios.load_kwh[w, j]) * (1 - m.charge_mode[w, j]),
    )

    if original is None:
        contract = sum(float(price[j]) * model.purchase[j] for j in model.T)
    else:
        contract = sum(
            float(price[j]) * float(original[j]) + 1.5 * float(price[j]) * model.up[j] - 0.5 * float(price[j]) * model.down[j]
            for j in model.T
        )
    expected_emergency = sum(
        float(scenarios.weights[w]) * EMERGENCY_MULTIPLIER * float(price[j]) * model.emergency[w, j]
        for w in model.W for j in model.T
    )
    terminal_credit = terminal_value_yuan_per_kwh * sum(float(scenarios.weights[w]) * model.soc[w, n] for w in model.W)
    model.objective = pyo.Objective(expr=contract + expected_emergency - terminal_credit)

    solver, solver_name = _solver()
    started = time.perf_counter()
    raw = solver.solve(model, tee=False)
    elapsed = time.perf_counter() - started
    termination = raw.solver.termination_condition
    if termination not in {TerminationCondition.optimal, TerminationCondition.feasible, TerminationCondition.maxTimeLimit}:
        raise RuntimeError(f"Q3 V2 MILP failed: {raw.solver.status}/{termination}")
    try:
        plan_values = np.array([pyo.value(model.purchase[j]) for j in model.T], dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("MILP returned no usable incumbent") from exc
    if not np.isfinite(plan_values).all():
        raise RuntimeError("MILP incumbent is not finite")

    lower = getattr(raw.problem, "lower_bound", None)
    upper = getattr(raw.problem, "upper_bound", None)
    lower = float(lower) if lower is not None and math.isfinite(float(lower)) else np.nan
    upper = float(upper) if upper is not None and math.isfinite(float(upper)) else np.nan
    rel_gap = abs(upper - lower) / max(1.0, abs(upper)) if np.isfinite(lower) and np.isfinite(upper) else (0.0 if termination == TerminationCondition.optimal else np.nan)
    certified = termination == TerminationCondition.optimal and (not np.isfinite(rel_gap) or rel_gap <= MIP_REL_GAP + TOL)

    plan = pd.DataFrame({"slot": slots, "planned_purchase_kwh": plan_values, "price_yuan_per_kwh": price})
    rows = []
    for w in model.W:
        for j in model.T:
            rows.append(
                {
                    "scenario_index": int(w), "scenario_source_date": scenarios.source_dates[int(w)], "slot": int(slots[int(j)]),
                    "load_scenario_kwh": float(scenarios.load_kwh[w, j]), "pv_scenario_kwh": float(scenarios.pv_kwh[w, j]),
                    "charge_scenario_kwh": float(pyo.value(model.charge[w, j])), "discharge_scenario_kwh": float(pyo.value(model.discharge[w, j])),
                    "emergency_scenario_kwh": float(pyo.value(model.emergency[w, j])), "spill_scenario_kwh": float(pyo.value(model.spill[w, j])),
                    "soc_start_scenario_kwh": float(pyo.value(model.soc[w, j])), "soc_end_scenario_kwh": float(pyo.value(model.soc[w, j + 1])),
                    "charge_mode": int(round(pyo.value(model.charge_mode[w, j]))),
                }
            )
    dispatch = pd.DataFrame(rows)
    metadata = {
        "solver": solver_name, "solver_status": str(raw.solver.status), "termination_condition": str(termination),
        "solution_class": "certified_to_requested_gap" if certified else "time_limit_feasible_incumbent",
        "incumbent_objective_yuan": upper, "best_bound_yuan": lower,
        "absolute_gap_yuan": abs(upper - lower) if np.isfinite(lower) and np.isfinite(upper) else np.nan,
        "relative_gap": float(rel_gap), "solve_time_seconds": elapsed,
        "requested_relative_gap": MIP_REL_GAP, "time_limit_seconds": MIP_TIME_LIMIT_SECONDS,
        "random_seed": MIP_RANDOM_SEED, "scenario_count_effective": scenarios.count,
        "terminal_value_test_yuan_per_kwh": terminal_value_yuan_per_kwh,
        "objective_yuan": float(pyo.value(model.objective)),
        "expected_terminal_soc_kwh": float(sum(scenarios.weights[int(w)] * pyo.value(model.soc[w, n]) for w in model.W)),
    }
    return RollingMilpResult(plan, dispatch, metadata)

