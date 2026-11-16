"""Single-settlement accounting for Q3 V2 under the documented assumption."""

from __future__ import annotations

import numpy as np
import pandas as pd


def settle_contract(original_kwh, effective_kwh, price_yuan_per_kwh) -> pd.DataFrame:
    p = np.asarray(original_kwh, dtype=float)
    z = np.asarray(effective_kwh, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    p, z, price = np.broadcast_arrays(p, z, price)
    if (p < -1e-10).any() or (z < -1e-10).any() or (price < -1e-10).any():
        raise ValueError("Purchase quantities and prices must be nonnegative")
    up = np.maximum(z - p, 0.0)
    down = np.maximum(p - z, 0.0)
    contract = price * p + 1.5 * price * up - 0.5 * price * down
    return pd.DataFrame(
        {
            "original_purchase_kwh": p.ravel(),
            "effective_purchase_kwh": z.ravel(),
            "delta_plus_kwh": up.ravel(),
            "delta_minus_kwh": down.ravel(),
            "original_plan_cost_yuan": (price * p).ravel(),
            "upward_cost_yuan": (1.5 * price * up).ravel(),
            "downward_refund_yuan": (0.5 * price * down).ravel(),
            "contract_cost_yuan": contract.ravel(),
        }
    )


def manual_settlement_cases() -> pd.DataFrame:
    cases = [
        ("无调整", 100.0, "100", 100.0, 0.0, 100.0),
        ("单次下调", 100.0, "80", 80.0, 0.0, 90.0),
        ("单次上调", 100.0, "120", 120.0, 0.0, 130.0),
        ("先下调后上调", 100.0, "80→110", 110.0, 0.0, 115.0),
        ("先上调后下调", 100.0, "120→80", 80.0, 0.0, 90.0),
        ("调整后回原计划", 100.0, "80→100", 100.0, 0.0, 100.0),
        ("下调并紧急购电", 100.0, "80", 80.0, 10.0, 140.0),
    ]
    rows = []
    for name, original, versions, final, emergency, expected in cases:
        settled = settle_contract([original], [final], [1.0]).iloc[0]
        emergency_cost = 5.0 * emergency
        actual = float(settled.contract_cost_yuan + emergency_cost)
        rows.append(
            {
                "case": name,
                "price_yuan_per_kwh": 1.0,
                "original_kwh": original,
                "versions": versions,
                "final_effective_kwh": final,
                "contract_cost_yuan": float(settled.contract_cost_yuan),
                "emergency_kwh": emergency,
                "emergency_cost_yuan": emergency_cost,
                "actual_total_cost_yuan": actual,
                "expected_total_yuan": expected,
                "absolute_error_yuan": abs(actual - expected),
                "passed": abs(actual - expected) <= 1e-10,
            }
        )
    return pd.DataFrame(rows)

