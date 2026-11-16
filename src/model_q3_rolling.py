"""Day-ahead and forecast-update LPs for the Question-3 short closure."""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from src.config import E_B_MAX, ETA_C, ETA_D, S_MAX, S_MIN
from src.model_q2_day_ahead import get_highs_solver
from src.q3_settlement import settle_adjusted_purchase


def solve_q3_reference_plan(
    forecast: pd.DataFrame,
    price: pd.DataFrame,
    initial_soc_kwh: float,
    reserve_soc_kwh: float,
    reserve_penalty_yuan_per_kwh: float,
    original_purchase: pd.Series | None = None,
    fixed_purchase: pd.Series | None = None,
    solver=None,
) -> tuple[pd.DataFrame, dict]:
    """Solve one original or updated reference plan on ordered remaining slots.

    When ``original_purchase`` is supplied, adjustment variables and the
    objective are defined relative to the 00:00 original plan, not the previous
    forecast version. This is the documented main settlement assumption.
    """
    required = {"idx", "net_load_forecast_kwh"}
    if not required.issubset(forecast.columns):
        raise ValueError(f"Forecast lacks fields: {sorted(required - set(forecast.columns))}")
    data = forecast.sort_values("idx").reset_index(drop=True)
    indices = data["idx"].astype(int).tolist()
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Reference-plan indices must form one consecutive suffix")
    if not S_MIN <= initial_soc_kwh <= S_MAX:
        raise ValueError("Initial SOC is outside the physical interval")
    if not S_MIN <= reserve_soc_kwh <= S_MAX:
        raise ValueError("Reserve target is outside the physical interval")
    price_map = price.set_index("idx")["price_yuan_per_kwh"]
    if not set(indices).issubset(price_map.index):
        raise ValueError("Price table does not cover the planning suffix")
    original_map = None
    if original_purchase is not None:
        original_map = pd.Series(original_purchase, dtype=float)
        if not set(indices).issubset(original_map.index):
            raise ValueError("Original plan does not cover the adjustment suffix")
    fixed_map = None
    if fixed_purchase is not None:
        fixed_map = pd.Series(fixed_purchase, dtype=float)
        if not set(indices).issubset(fixed_map.index):
            raise ValueError("Fixed purchase does not cover the planning suffix")

    model = pyo.ConcreteModel("q3_reference_plan")
    model.T = pyo.Set(initialize=indices, ordered=True)
    model.J = pyo.RangeSet(0, len(indices))
    model.purchase = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, E_B_MAX))
    model.discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, E_B_MAX))
    model.curtail = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.J, bounds=(S_MIN, S_MAX))
    model.reserve_shortfall = pyo.Var(domain=pyo.NonNegativeReals)
    if fixed_map is not None:
        model.forecast_shortfall = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        def fixed_purchase_rule(m, t):
            return m.purchase[t] == float(fixed_map.at[t])

        model.fixed_purchase = pyo.Constraint(model.T, rule=fixed_purchase_rule)
    if original_map is not None:
        model.upward = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.downward = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        def adjustment_identity(m, t):
            return m.purchase[t] == float(original_map.at[t]) + m.upward[t] - m.downward[t]

        def downward_limit(m, t):
            return m.downward[t] <= float(original_map.at[t])

        model.adjustment_identity = pyo.Constraint(model.T, rule=adjustment_identity)
        model.downward_limit = pyo.Constraint(model.T, rule=downward_limit)

    net_map = data.set_index("idx")["net_load_forecast_kwh"]
    position = {t: j for j, t in enumerate(indices)}

    def balance_rule(m, t):
        shortage = m.forecast_shortfall[t] if fixed_map is not None else 0.0
        return m.purchase[t] + m.discharge[t] + shortage == float(net_map.at[t]) + m.charge[t] + m.curtail[t]

    def soc_rule(m, t):
        j = position[t]
        return m.soc[j + 1] == m.soc[j] + ETA_C * m.charge[t] - m.discharge[t] / ETA_D

    model.balance = pyo.Constraint(model.T, rule=balance_rule)
    model.soc_transition = pyo.Constraint(model.T, rule=soc_rule)
    model.initial_soc = pyo.Constraint(expr=model.soc[0] == float(initial_soc_kwh))
    model.soft_reserve = pyo.Constraint(
        expr=model.reserve_shortfall >= float(reserve_soc_kwh) - model.soc[len(indices)]
    )
    if original_map is None:
        contract_cost = sum(float(price_map.at[t]) * model.purchase[t] for t in indices)
    else:
        contract_cost = sum(
            float(price_map.at[t]) * float(original_map.at[t])
            - 0.5 * float(price_map.at[t]) * model.downward[t]
            + 1.5 * float(price_map.at[t]) * model.upward[t]
            for t in indices
        )
    forecast_shortfall_cost = (
        sum(5.0 * float(price_map.at[t]) * model.forecast_shortfall[t] for t in indices)
        if fixed_map is not None else 0.0
    )
    model.objective = pyo.Objective(
        expr=contract_cost + forecast_shortfall_cost
        + float(reserve_penalty_yuan_per_kwh) * model.reserve_shortfall,
        sense=pyo.minimize,
    )

    solver, solver_name = get_highs_solver() if solver is None else (solver, type(solver).__name__)
    raw = solver.solve(model)
    if raw.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(f"Q3 reference LP failed: {raw.solver.termination_condition}")

    result = data.copy()
    result["reference_purchase_kwh"] = [pyo.value(model.purchase[t]) for t in indices]
    result["reference_charge_kwh"] = [pyo.value(model.charge[t]) for t in indices]
    result["reference_discharge_kwh"] = [pyo.value(model.discharge[t]) for t in indices]
    result["reference_curtailment_kwh"] = [pyo.value(model.curtail[t]) for t in indices]
    result["reference_soc_start_kwh"] = [pyo.value(model.soc[j]) for j in range(len(indices))]
    result["reference_soc_end_kwh"] = [pyo.value(model.soc[j + 1]) for j in range(len(indices))]
    result["reference_forecast_shortfall_kwh"] = (
        [pyo.value(model.forecast_shortfall[t]) for t in indices]
        if fixed_map is not None else 0.0
    )
    result["price_yuan_per_kwh"] = [float(price_map.at[t]) for t in indices]
    if original_map is None:
        result["upward_vs_original_kwh"] = 0.0
        result["downward_vs_original_kwh"] = 0.0
        result["reference_contract_cost_yuan"] = (
            result["price_yuan_per_kwh"] * result["reference_purchase_kwh"]
        )
    else:
        result["upward_vs_original_kwh"] = [pyo.value(model.upward[t]) for t in indices]
        result["downward_vs_original_kwh"] = [pyo.value(model.downward[t]) for t in indices]
        settled = settle_adjusted_purchase(
            [float(original_map.at[t]) for t in indices],
            result["reference_purchase_kwh"],
            result["price_yuan_per_kwh"],
        )
        result["reference_contract_cost_yuan"] = settled["contract_settlement_cost_yuan"].to_numpy()

    simultaneous_adjustment = (
        (result["upward_vs_original_kwh"] > 1e-8)
        & (result["downward_vs_original_kwh"] > 1e-8)
    )
    metadata = {
        "solver": solver_name,
        "termination_condition": str(raw.solver.termination_condition),
        "objective_yuan": float(pyo.value(model.objective)),
        "reference_contract_cost_yuan": float(result["reference_contract_cost_yuan"].sum()),
        "reserve_shortfall_kwh": float(pyo.value(model.reserve_shortfall)),
        "reference_final_soc_kwh": float(result["reference_soc_end_kwh"].iloc[-1]),
        "reference_forecast_shortfall_kwh": float(result["reference_forecast_shortfall_kwh"].sum()),
        "simultaneous_up_down_count": int(simultaneous_adjustment.sum()),
    }
    return result, metadata
