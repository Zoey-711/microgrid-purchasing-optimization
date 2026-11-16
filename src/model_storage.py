"""Reusable storage constraints and the Question-1 linear program."""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo

from src.config import E_B_MAX, ETA_C, ETA_D, N_PERIODS, S_FINAL, S_INIT, S_MAX, S_MIN


def build_q1_model(data: pd.DataFrame) -> pyo.ConcreteModel:
    """Build the deterministic 144-period storage-dispatch LP for Question 1."""
    if len(data) != N_PERIODS:
        raise ValueError(f"Expected {N_PERIODS} periods, got {len(data)}")

    indexed = data.set_index("idx")
    model = pyo.ConcreteModel("q1_storage_lp")
    model.T = pyo.RangeSet(0, N_PERIODS - 1)
    model.T_SOC = pyo.RangeSet(0, N_PERIODS)
    model.purchase = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, E_B_MAX))
    model.discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, E_B_MAX))
    model.curtail = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T_SOC, bounds=(S_MIN, S_MAX))

    def balance_rule(m, t):
        return (
            m.purchase[t] + float(indexed.at[t, "pv_kwh"]) + m.discharge[t]
            == float(indexed.at[t, "load_kwh"]) + m.charge[t] + m.curtail[t]
        )

    def soc_rule(m, t):
        return m.soc[t + 1] == m.soc[t] + ETA_C * m.charge[t] - m.discharge[t] / ETA_D

    model.energy_balance = pyo.Constraint(model.T, rule=balance_rule)
    model.soc_transition = pyo.Constraint(model.T, rule=soc_rule)
    model.initial_soc = pyo.Constraint(expr=model.soc[0] == S_INIT)
    model.final_soc = pyo.Constraint(expr=model.soc[N_PERIODS] == S_FINAL)
    model.objective = pyo.Objective(
        expr=sum(float(indexed.at[t, "price"]) * model.purchase[t] for t in model.T),
        sense=pyo.minimize,
    )
    return model


def extract_q1_solution(model: pyo.ConcreteModel, data: pd.DataFrame) -> pd.DataFrame:
    """Extract an optimal Pyomo solution as a tabular result."""
    solution = data.copy()
    solution["purchase_kwh"] = [pyo.value(model.purchase[t]) for t in model.T]
    solution["charge_kwh"] = [pyo.value(model.charge[t]) for t in model.T]
    solution["discharge_kwh"] = [pyo.value(model.discharge[t]) for t in model.T]
    solution["curtail_kwh"] = [pyo.value(model.curtail[t]) for t in model.T]
    solution["soc_start_kwh"] = [pyo.value(model.soc[t]) for t in model.T]
    solution["soc_end_kwh"] = [pyo.value(model.soc[t + 1]) for t in model.T]
    solution["period_cost_yuan"] = solution["price"] * solution["purchase_kwh"]
    return solution
