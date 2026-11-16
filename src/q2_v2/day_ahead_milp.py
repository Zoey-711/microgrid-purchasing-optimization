"""Two-stage joint-scenario MILP for a common Q2 day-ahead purchase plan."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from .config import (
    EMERGENCY_PRICE_MULTIPLIER,
    ENERGY_MAX_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    MIP_REL_GAP,
    MIP_TIME_LIMIT_SECONDS,
    N_SLOTS,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    TOL,
)
from .scenarios import JointScenarios


@dataclass(frozen=True)
class MilpResult:
    plan: pd.DataFrame
    scenario_dispatch: pd.DataFrame
    metadata: dict


def _solver(time_limit_seconds: float, relative_gap: float, random_seed: int) -> tuple[object, str]:
    for name in ("appsi_highs", "highs"):
        solver = pyo.SolverFactory(name)
        if solver is not None and solver.available(exception_flag=False):
            solver.options["time_limit"] = time_limit_seconds
            solver.options["mip_rel_gap"] = relative_gap
            solver.options["random_seed"] = random_seed
            solver.options["threads"] = 1
            return solver, name
    raise RuntimeError("HiGHS is unavailable")


def solve_joint_scenario_milp(
    scenarios: JointScenarios,
    price_yuan_per_kwh: np.ndarray,
    initial_soc_kwh: float,
    terminal_value_yuan_per_kwh: float,
    time_limit_seconds: float = MIP_TIME_LIMIT_SECONDS,
    relative_gap: float = MIP_REL_GAP,
    random_seed: int = 0,
) -> MilpResult:
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    if price.shape != (N_SLOTS,) or scenarios.load_kwh.shape[1] != N_SLOTS:
        raise ValueError("Price and scenario trajectories must contain 144 slots")
    if not SOC_MIN_KWH <= initial_soc_kwh <= SOC_MAX_KWH:
        raise ValueError("Initial SOC is outside physical bounds")
    if terminal_value_yuan_per_kwh < 0:
        raise ValueError("Terminal inventory value must be nonnegative")
    k = scenarios.count
    model = pyo.ConcreteModel("q2_v2_joint_scenario_milp")
    model.W = pyo.RangeSet(0, k - 1)
    model.T = pyo.RangeSet(0, N_SLOTS - 1)
    model.S = pyo.RangeSet(0, N_SLOTS)
    purchase_ub = {
        t: float(np.max(scenarios.load_kwh[:, t]) + ENERGY_MAX_KWH) for t in range(N_SLOTS)
    }
    model.purchase = pyo.Var(
        model.T, domain=pyo.NonNegativeReals, bounds=lambda m, t: (0.0, purchase_ub[t])
    )
    model.charge = pyo.Var(model.W, model.T, domain=pyo.NonNegativeReals, bounds=(0, ENERGY_MAX_KWH))
    model.discharge = pyo.Var(model.W, model.T, domain=pyo.NonNegativeReals, bounds=(0, ENERGY_MAX_KWH))
    model.emergency = pyo.Var(model.W, model.T, domain=pyo.NonNegativeReals)
    model.spill = pyo.Var(model.W, model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.W, model.S, bounds=(SOC_MIN_KWH, SOC_MAX_KWH))
    # mode=1: charging is permitted; mode=0: discharge/emergency are permitted.
    model.charge_mode = pyo.Var(model.W, model.T, domain=pyo.Binary)

    def balance(m, w, t):
        return (
            m.purchase[t] + float(scenarios.pv_kwh[w, t]) + m.discharge[w, t] + m.emergency[w, t]
            == float(scenarios.load_kwh[w, t]) + m.charge[w, t] + m.spill[w, t]
        )

    def transition(m, w, t):
        return m.soc[w, t + 1] == m.soc[w, t] + ETA_CHARGE * m.charge[w, t] - m.discharge[w, t] / ETA_DISCHARGE

    model.balance = pyo.Constraint(model.W, model.T, rule=balance)
    model.transition = pyo.Constraint(model.W, model.T, rule=transition)
    model.initial = pyo.Constraint(model.W, rule=lambda m, w: m.soc[w, 0] == initial_soc_kwh)
    model.charge_exclusive = pyo.Constraint(
        model.W, model.T, rule=lambda m, w, t: m.charge[w, t] <= ENERGY_MAX_KWH * m.charge_mode[w, t]
    )
    model.discharge_exclusive = pyo.Constraint(
        model.W, model.T, rule=lambda m, w, t: m.discharge[w, t] <= ENERGY_MAX_KWH * (1 - m.charge_mode[w, t])
    )
    # Explicitly prevent scenario emergency energy from charging the battery.
    model.emergency_no_charge = pyo.Constraint(
        model.W,
        model.T,
        rule=lambda m, w, t: m.emergency[w, t]
        <= float(scenarios.load_kwh[w, t]) * (1 - m.charge_mode[w, t]),
    )
    plan_cost = sum(float(price[t]) * model.purchase[t] for t in model.T)
    emergency_cost = sum(
        float(scenarios.weights[w]) * EMERGENCY_PRICE_MULTIPLIER * float(price[t]) * model.emergency[w, t]
        for w in model.W
        for t in model.T
    )
    terminal_credit = terminal_value_yuan_per_kwh * sum(
        float(scenarios.weights[w]) * model.soc[w, N_SLOTS] for w in model.W
    )
    model.objective = pyo.Objective(expr=plan_cost + emergency_cost - terminal_credit)

    solver, solver_name = _solver(time_limit_seconds, relative_gap, random_seed)
    started = time.perf_counter()
    raw = solver.solve(model, tee=False)
    elapsed = time.perf_counter() - started
    termination = raw.solver.termination_condition
    status = raw.solver.status
    acceptable_terminations = {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.maxTimeLimit,
    }
    if termination not in acceptable_terminations:
        raise RuntimeError(f"Scenario MILP failed: status={status}, termination={termination}")

    # Appsi/HiGHS loads its incumbent on a time limit.  Verify that a usable
    # incumbent really exists before accepting a non-certified Stage-2 solve.
    try:
        incumbent_values = np.asarray([pyo.value(model.purchase[t]) for t in model.T], dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Scenario MILP ended at {termination} without a loadable incumbent"
        ) from exc
    if not np.isfinite(incumbent_values).all():
        raise RuntimeError(f"Scenario MILP ended at {termination} with a non-finite incumbent")

    lower = getattr(raw.problem, "lower_bound", None)
    upper = getattr(raw.problem, "upper_bound", None)
    lower = float(lower) if lower is not None and math.isfinite(float(lower)) else np.nan
    upper = float(upper) if upper is not None and math.isfinite(float(upper)) else np.nan
    if math.isfinite(lower) and math.isfinite(upper):
        gap = abs(upper - lower) / max(1.0, abs(upper))
    elif termination == TerminationCondition.optimal:
        gap = 0.0
    else:
        gap = np.nan
    optimal = termination == TerminationCondition.optimal and (not math.isfinite(gap) or gap <= relative_gap + TOL)

    plan = pd.DataFrame(
        {
            "slot": np.arange(N_SLOTS),
            "price_yuan_per_kwh": price,
            "planned_purchase_kwh": incumbent_values,
        }
    )
    rows = []
    for w in model.W:
        for t in model.T:
            rows.append(
                {
                    "scenario_index": int(w),
                    "scenario_source_date": scenarios.source_dates[int(w)],
                    "slot": int(t),
                    "load_scenario_kwh": float(scenarios.load_kwh[w, t]),
                    "pv_scenario_kwh": float(scenarios.pv_kwh[w, t]),
                    "charge_scenario_kwh": float(pyo.value(model.charge[w, t])),
                    "discharge_scenario_kwh": float(pyo.value(model.discharge[w, t])),
                    "emergency_scenario_kwh": float(pyo.value(model.emergency[w, t])),
                    "spill_scenario_kwh": float(pyo.value(model.spill[w, t])),
                    "soc_start_scenario_kwh": float(pyo.value(model.soc[w, t])),
                    "soc_end_scenario_kwh": float(pyo.value(model.soc[w, t + 1])),
                    "charge_mode": int(round(pyo.value(model.charge_mode[w, t]))),
                }
            )
    dispatch = pd.DataFrame(rows)
    metadata = {
        "solver": solver_name,
        "solver_status": str(status),
        "termination_condition": str(termination),
        "optimal_solution_certified": bool(optimal),
        "solution_class": "certified_optimal" if optimal else "time_limit_feasible_incumbent",
        "best_bound_yuan": lower,
        "incumbent_objective_yuan": upper,
        "mip_gap": float(gap),
        "solve_time_seconds": elapsed,
        "time_limit_seconds": time_limit_seconds,
        "requested_relative_gap": relative_gap,
        "random_seed": random_seed,
        "absolute_gap_yuan": abs(upper - lower) if math.isfinite(lower) and math.isfinite(upper) else np.nan,
        "scenario_count": k,
        "terminal_value_yuan_per_kwh": terminal_value_yuan_per_kwh,
        "objective_yuan": float(pyo.value(model.objective)),
        "plan_cost_yuan": float(np.dot(price, plan["planned_purchase_kwh"])),
        "expected_emergency_cost_yuan": float(
            sum(
                scenarios.weights[int(w)] * EMERGENCY_PRICE_MULTIPLIER * price[int(t)] * pyo.value(model.emergency[w, t])
                for w in model.W
                for t in model.T
            )
        ),
        "expected_terminal_soc_kwh": float(
            sum(scenarios.weights[int(w)] * pyo.value(model.soc[w, N_SLOTS]) for w in model.W)
        ),
        "terminal_value_objective_credit_yuan": float(
            terminal_value_yuan_per_kwh
            * sum(scenarios.weights[int(w)] * pyo.value(model.soc[w, N_SLOTS]) for w in model.W)
        ),
    }
    return MilpResult(plan=plan, scenario_dispatch=dispatch, metadata=metadata)
