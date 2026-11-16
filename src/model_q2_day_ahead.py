"""Question-2 day-ahead LP with a soft end-of-day SOC reserve."""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from src.config import E_B_MAX, ETA_C, ETA_D, N_PERIODS, S_MAX, S_MIN


def build_q2_day_ahead_model(
    forecast: pd.DataFrame,
    price: pd.DataFrame,
    initial_soc_kwh: float,
    reserve_soc_kwh: float,
    reserve_penalty_yuan_per_kwh: float,
) -> pyo.ConcreteModel:
    """Build the forecast-conditioned LP; no actual execution variable is used."""
    if len(forecast) != N_PERIODS or len(price) != N_PERIODS:
        raise ValueError("forecast and price must each contain 144 ordered slots")
    if not S_MIN <= initial_soc_kwh <= S_MAX:
        raise ValueError("initial SOC violates the safety interval")
    if not S_MIN <= reserve_soc_kwh <= S_MAX:
        raise ValueError("reserve SOC must lie in the safety interval")
    if reserve_penalty_yuan_per_kwh < 0:
        raise ValueError("reserve penalty must be non-negative")

    forecasted_net = forecast.set_index("idx")["forecast_net_load_kwh"]
    price_series = price.set_index("idx")["price_yuan_per_kwh"]
    if set(forecasted_net.index) != set(range(N_PERIODS)) or set(price_series.index) != set(range(N_PERIODS)):
        raise ValueError("forecast and price indices must be exactly 0,...,143")

    model = pyo.ConcreteModel("q2_day_ahead_lp")
    model.T = pyo.RangeSet(0, N_PERIODS - 1)
    model.T_SOC = pyo.RangeSet(0, N_PERIODS)
    model.purchase = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.plan_charge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, E_B_MAX))
    model.plan_discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, E_B_MAX))
    model.plan_curtail = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.plan_soc = pyo.Var(model.T_SOC, bounds=(S_MIN, S_MAX))
    model.reserve_shortfall = pyo.Var(domain=pyo.NonNegativeReals)

    def balance_rule(m, t):
        return (
            m.purchase[t] + m.plan_discharge[t]
            == float(forecasted_net.at[t]) + m.plan_charge[t] + m.plan_curtail[t]
        )

    def soc_rule(m, t):
        return m.plan_soc[t + 1] == m.plan_soc[t] + ETA_C * m.plan_charge[t] - m.plan_discharge[t] / ETA_D

    model.energy_balance = pyo.Constraint(model.T, rule=balance_rule)
    model.soc_transition = pyo.Constraint(model.T, rule=soc_rule)
    model.initial_soc = pyo.Constraint(expr=model.plan_soc[0] == float(initial_soc_kwh))
    model.soft_reserve = pyo.Constraint(
        expr=model.reserve_shortfall >= float(reserve_soc_kwh) - model.plan_soc[N_PERIODS]
    )
    model.objective = pyo.Objective(
        expr=sum(float(price_series.at[t]) * model.purchase[t] for t in model.T)
        + float(reserve_penalty_yuan_per_kwh) * model.reserve_shortfall,
        sense=pyo.minimize,
    )
    return model


def get_highs_solver():
    """Return an installed HiGHS interface, without silently changing solvers."""
    for solver_name in ("appsi_highs", "highs"):
        solver = pyo.SolverFactory(solver_name)
        if solver is not None and solver.available(exception_flag=False):
            return solver, solver_name
    raise RuntimeError("HiGHS is unavailable. Install highspy before running Question 2.")


def solve_q2_day_ahead(
    forecast: pd.DataFrame,
    price: pd.DataFrame,
    initial_soc_kwh: float,
    reserve_soc_kwh: float,
    reserve_penalty_yuan_per_kwh: float,
    solver=None,
) -> tuple[pd.DataFrame, dict]:
    """Solve and extract one day-ahead plan, retaining forecast/plan separation."""
    model = build_q2_day_ahead_model(
        forecast,
        price,
        initial_soc_kwh,
        reserve_soc_kwh,
        reserve_penalty_yuan_per_kwh,
    )
    solver, solver_name = get_highs_solver() if solver is None else (solver, type(solver).__name__)
    raw_result = solver.solve(model)
    termination = raw_result.solver.termination_condition
    if termination != TerminationCondition.optimal:
        raise RuntimeError(f"Question-2 day-ahead LP did not reach optimality: {termination}")

    indexed_price = price.set_index("idx")
    solution = forecast.copy().sort_values("idx").reset_index(drop=True)
    solution["price_yuan_per_kwh"] = [float(indexed_price.at[t, "price_yuan_per_kwh"]) for t in range(N_PERIODS)]
    solution["plan_purchase_kwh"] = [pyo.value(model.purchase[t]) for t in model.T]
    solution["plan_charge_kwh"] = [pyo.value(model.plan_charge[t]) for t in model.T]
    solution["plan_discharge_kwh"] = [pyo.value(model.plan_discharge[t]) for t in model.T]
    solution["plan_curtail_kwh"] = [pyo.value(model.plan_curtail[t]) for t in model.T]
    solution["plan_soc_start_kwh"] = [pyo.value(model.plan_soc[t]) for t in model.T]
    solution["plan_soc_end_kwh"] = [pyo.value(model.plan_soc[t + 1]) for t in model.T]
    solution["plan_purchase_cost_yuan"] = solution["price_yuan_per_kwh"] * solution["plan_purchase_kwh"]
    metadata = {
        "solver_name": solver_name,
        "solver_status": str(raw_result.solver.status),
        "termination_condition": str(termination),
        "objective_with_reserve_penalty_yuan": float(pyo.value(model.objective)),
        "plan_purchase_cost_yuan": float(solution["plan_purchase_cost_yuan"].sum()),
        "reserve_shortfall_kwh": float(pyo.value(model.reserve_shortfall)),
        "plan_soc_start_kwh": float(pyo.value(model.plan_soc[0])),
        "plan_soc_end_kwh": float(pyo.value(model.plan_soc[N_PERIODS])),
    }
    return solution, metadata
