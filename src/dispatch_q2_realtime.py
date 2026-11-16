"""Causal real-time execution of a fixed Question-2 purchase plan."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import E_B_MAX, ETA_C, ETA_D, N_PERIODS, S_MAX, S_MIN


def dispatch_q2_realtime(
    day_ahead_plan: pd.DataFrame,
    actual_day: pd.DataFrame,
    initial_soc_kwh: float,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, dict]:
    """Apply a fixed plan one slot at a time using only current actual values.

    ``plan_soc_end_kwh`` is a forecast-derived reference trajectory.  On an
    actual deficit, discharge never drives SOC below the reference for the
    current slot, which preserves the day-ahead risk reserve for later slots.
    On an actual surplus, charge whenever physical room exists.  This produces
    a causal receding execution rule without access to future actual values.
    """
    required_plan = {
        "idx", "plan_purchase_kwh", "plan_soc_end_kwh", "price_yuan_per_kwh",
    }
    required_actual = {"idx", "date", "load_kwh", "pv_kwh", "net_load_kwh"}
    if not required_plan.issubset(day_ahead_plan.columns):
        raise ValueError(f"day-ahead plan must contain {sorted(required_plan)}")
    if not required_actual.issubset(actual_day.columns):
        raise ValueError(f"actual day must contain {sorted(required_actual)}")
    plan = day_ahead_plan.sort_values("idx").reset_index(drop=True)
    actual = actual_day.sort_values("idx").reset_index(drop=True)
    if len(plan) != N_PERIODS or len(actual) != N_PERIODS:
        raise ValueError("plan and actual day must each contain 144 slots")
    if plan["idx"].tolist() != list(range(N_PERIODS)) or actual["idx"].tolist() != list(range(N_PERIODS)):
        raise ValueError("plan and actual indices must each be 0,...,143")
    if not S_MIN - tolerance <= initial_soc_kwh <= S_MAX + tolerance:
        raise ValueError("real-time initial SOC is outside bounds")

    records: list[dict] = []
    soc = float(initial_soc_kwh)
    for t in range(N_PERIODS):
        net = float(actual.at[t, "net_load_kwh"])
        purchase = float(plan.at[t, "plan_purchase_kwh"])
        price = float(plan.at[t, "price_yuan_per_kwh"])
        reserve_reference = float(plan.at[t, "plan_soc_end_kwh"])
        charge = discharge = emergency = curtail = 0.0

        if purchase >= net - tolerance:
            # Surplus branch: emergency purchase is forbidden; absorb excess first.
            surplus = max(0.0, purchase - net)
            charge = min(surplus, E_B_MAX, max(0.0, (S_MAX - soc) / ETA_C))
            curtail = surplus - charge
        else:
            # Deficit branch: charging is forbidden; discharge above the forecast
            # reference first, then purchase the remaining actual shortage urgently.
            deficit = net - purchase
            floor = max(S_MIN, min(S_MAX, reserve_reference))
            available_discharge = max(0.0, (soc - floor) * ETA_D)
            discharge = min(deficit, E_B_MAX, available_discharge)
            emergency = deficit - discharge

        soc_next = soc + ETA_C * charge - discharge / ETA_D
        balance = purchase + discharge + emergency - net - charge - curtail
        records.append(
            {
                "date": actual.at[t, "date"],
                "idx": t,
                "time_label": actual.at[t, "time_label"] if "time_label" in actual.columns else t,
                "price_yuan_per_kwh": price,
                "load_kwh": float(actual.at[t, "load_kwh"]),
                "pv_kwh": float(actual.at[t, "pv_kwh"]),
                "actual_net_load_kwh": net,
                "plan_purchase_kwh": purchase,
                "plan_soc_reference_end_kwh": reserve_reference,
                "rt_charge_kwh": charge,
                "rt_discharge_kwh": discharge,
                "emergency_purchase_kwh": emergency,
                "rt_curtail_kwh": curtail,
                "rt_soc_start_kwh": soc,
                "rt_soc_end_kwh": soc_next,
                "balance_residual_kwh": balance,
                "plan_purchase_cost_yuan": price * purchase,
                "emergency_purchase_cost_yuan": 5.0 * price * emergency,
            }
        )
        soc = soc_next

    result = pd.DataFrame(records)
    simultaneous = (result["rt_charge_kwh"] > tolerance) & (result["rt_discharge_kwh"] > tolerance)
    emergency_and_charge = (result["emergency_purchase_kwh"] > tolerance) & (result["rt_charge_kwh"] > tolerance)
    metadata = {
        "rt_initial_soc_kwh": float(initial_soc_kwh),
        "rt_final_soc_kwh": float(result["rt_soc_end_kwh"].iloc[-1]),
        "plan_purchase_cost_yuan": float(result["plan_purchase_cost_yuan"].sum()),
        "emergency_purchase_cost_yuan": float(result["emergency_purchase_cost_yuan"].sum()),
        "total_actual_cost_yuan": float(
            result["plan_purchase_cost_yuan"].sum() + result["emergency_purchase_cost_yuan"].sum()
        ),
        "total_emergency_purchase_kwh": float(result["emergency_purchase_kwh"].sum()),
        "max_balance_residual_kwh": float(np.abs(result["balance_residual_kwh"]).max()),
        "simultaneous_charge_discharge_count": int(simultaneous.sum()),
        "emergency_charge_overlap_count": int(emergency_and_charge.sum()),
    }
    return result, metadata
