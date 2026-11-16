"""One-slot causal physical execution for Question 3."""

from __future__ import annotations

from src.config import E_B_MAX, ETA_C, ETA_D, S_MAX, S_MIN


def dispatch_q3_slot(
    actual_row,
    effective_purchase_kwh: float,
    current_soc_kwh: float,
    reference_soc_end_kwh: float,
    tolerance: float = 1e-9,
) -> dict:
    """Execute one slot without reading any future actual value."""
    if not S_MIN - tolerance <= current_soc_kwh <= S_MAX + tolerance:
        raise ValueError("Current SOC is outside the physical interval")
    purchase = max(0.0, float(effective_purchase_kwh))
    load = float(actual_row["load_kwh"])
    pv = float(actual_row["pv_kwh"])
    net = load - pv
    charge = discharge = emergency = curtail = 0.0

    if purchase >= net - tolerance:
        surplus = max(0.0, purchase - net)
        charge = min(surplus, E_B_MAX, max(0.0, (S_MAX - current_soc_kwh) / ETA_C))
        curtail = surplus - charge
    else:
        deficit = net - purchase
        floor = max(S_MIN, min(S_MAX, float(reference_soc_end_kwh)))
        available = max(0.0, (current_soc_kwh - floor) * ETA_D)
        discharge = min(deficit, E_B_MAX, available)
        emergency = deficit - discharge

    next_soc = current_soc_kwh + ETA_C * charge - discharge / ETA_D
    residual = purchase + pv + discharge + emergency - load - charge - curtail
    return {
        "load_kwh": load,
        "pv_actual_kwh": pv,
        "net_load_actual_kwh": net,
        "effective_purchase_kwh": purchase,
        "charge_rt_kwh": charge,
        "discharge_rt_kwh": discharge,
        "emergency_purchase_kwh": emergency,
        "curtailment_rt_kwh": curtail,
        "soc_start_kwh": current_soc_kwh,
        "soc_end_kwh": next_soc,
        "reference_soc_end_kwh": float(reference_soc_end_kwh),
        "balance_residual_kwh": residual,
    }
