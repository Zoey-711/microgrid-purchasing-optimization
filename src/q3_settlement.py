"""Question-3 adjustment settlement under the documented main assumption.

The official statement gives the 0.5 and 1.5 price multipliers but does not
uniquely specify repeated-adjustment accounting. This module therefore keeps
the chosen non-duplicative interpretation explicit and independently testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def settle_adjusted_purchase(
    original_kwh,
    effective_kwh,
    price_yuan_per_kwh,
) -> pd.DataFrame:
    """Settle each slot once against its 00:00 original purchase.

    For a downward adjustment, delivered energy is paid normally and the
    cancelled part incurs a 50% breach charge. For an upward adjustment, the
    original amount is paid normally and the excess is paid at 150%.
    """
    original = np.asarray(original_kwh, dtype=float)
    effective = np.asarray(effective_kwh, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    original, effective, price = np.broadcast_arrays(original, effective, price)
    if (original < -1e-12).any() or (effective < -1e-12).any() or (price < -1e-12).any():
        raise ValueError("Purchase quantities and prices must be non-negative")

    upward = np.maximum(effective - original, 0.0)
    downward = np.maximum(original - effective, 0.0)
    retained = np.minimum(original, effective)
    retained_cost = price * retained
    downward_breach_cost = 0.5 * price * downward
    upward_purchase_cost = 1.5 * price * upward
    settlement = retained_cost + downward_breach_cost + upward_purchase_cost
    original_plan_cost = price * original
    downward_refund = 0.5 * price * downward
    equivalent = original_plan_cost - downward_refund + upward_purchase_cost
    if not np.allclose(settlement, equivalent, atol=1e-10, rtol=0.0):
        raise AssertionError("Equivalent settlement formulas disagree")

    return pd.DataFrame(
        {
            "original_purchase_kwh": original.ravel(),
            "effective_purchase_kwh": effective.ravel(),
            "upward_adjustment_kwh": upward.ravel(),
            "downward_adjustment_kwh": downward.ravel(),
            "retained_original_kwh": retained.ravel(),
            "price_yuan_per_kwh": price.ravel(),
            "original_plan_cost_yuan": original_plan_cost.ravel(),
            "downward_refund_yuan": downward_refund.ravel(),
            "retained_purchase_cost_yuan": retained_cost.ravel(),
            "downward_breach_cost_yuan": downward_breach_cost.ravel(),
            "upward_purchase_cost_yuan": upward_purchase_cost.ravel(),
            "contract_settlement_cost_yuan": settlement.ravel(),
        }
    )


def settlement_examples() -> pd.DataFrame:
    """Return hand-checkable single- and multi-adjustment examples."""
    cases = [
        ("单次下调", 1.0, 100.0, "80", 80.0),
        ("单次上调", 1.0, 100.0, "120", 120.0),
        ("无调整", 1.0, 100.0, "100", 100.0),
        ("多次调整后上调", 1.0, 100.0, "120→90→110", 110.0),
        ("多次调整后下调", 1.0, 100.0, "80→115→90", 90.0),
    ]
    rows: list[dict] = []
    for name, price, original, versions, final in cases:
        result = settle_adjusted_purchase([original], [final], [price]).iloc[0]
        rows.append(
            {
                "case": name,
                "price_yuan_per_kwh": price,
                "original_purchase_kwh": original,
                "intermediate_versions_kwh": versions,
                "final_effective_purchase_kwh": final,
                "upward_adjustment_kwh": result.upward_adjustment_kwh,
                "downward_adjustment_kwh": result.downward_adjustment_kwh,
                "retained_purchase_cost_yuan": result.retained_purchase_cost_yuan,
                "downward_breach_cost_yuan": result.downward_breach_cost_yuan,
                "upward_purchase_cost_yuan": result.upward_purchase_cost_yuan,
                "contract_settlement_cost_yuan": result.contract_settlement_cost_yuan,
                "charged_adjustment_versions": 1,
            }
        )
    return pd.DataFrame(rows)
