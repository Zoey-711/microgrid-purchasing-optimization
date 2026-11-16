"""Two-method PV mapping sensitivity for the selected January Q3 parameters."""

from __future__ import annotations

import pandas as pd

from src.backtest_q3_january import _run_candidate, _summarize_candidate
from src.config import (
    Q3_ISSUE_HOURS,
    Q3_JANUARY_EVALUATION_END,
    Q3_LOAD_HALF_LIFE_DAYS,
    TABLE_DIR,
)
from src.data_io import load_q2_actual_energy
from src.forecast_q3 import add_q3_slot_datetimes, build_q3_forecast_version, load_q3_pv_forecast_records
from src.model_q2_day_ahead import get_highs_solver


def run_q3_pv_mapping_sensitivity():
    actual, price = load_q2_actual_energy()
    actual = add_q3_slot_datetimes(actual)
    pv_records = load_q3_pv_forecast_records()
    solver, solver_name = get_highs_solver()
    rows = []
    for method in ("linear", "right_hold"):
        cache = {}
        for day in pd.date_range("2025-01-02", Q3_JANUARY_EVALUATION_END, freq="D"):
            for issue_hour in Q3_ISSUE_HOURS:
                cache[(21, 12, day, issue_hour)] = build_q3_forecast_version(
                    actual, pv_records, day, issue_hour,
                    history_days=21,
                    half_life_days=Q3_LOAD_HALF_LIFE_DAYS,
                    bias_lookback_slots=12,
                    pv_mapping_method=method,
                )
        execution, _, daily = _run_candidate(
            actual, price, cache, 21, 12, 6000.0, "all_updates", solver
        )
        rows.append({
            "pv_mapping_method": method,
            "load_history_days": 21,
            "load_bias_lookback_slots": 12,
            "reserve_soc_kwh": 6000.0,
            "adjustment_policy": "all_updates",
            **_summarize_candidate(execution, daily),
            "solver": solver_name,
        })
    result = pd.DataFrame(rows)
    path = TABLE_DIR / "q3_pv_mapping_sensitivity.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")
    print(result[["pv_mapping_method", "actual_total_cost_yuan", "emergency_purchase_kwh", "validation_passed"]].to_string(index=False))
    print(path)
    return result


if __name__ == "__main__":
    run_q3_pv_mapping_sensitivity()
