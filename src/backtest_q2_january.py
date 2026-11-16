"""January common-period causal backtest for Question 2 parameter selection."""

from __future__ import annotations

from itertools import product

import pandas as pd

from src.config import (
    Q2_BACKTEST_END,
    Q2_BACKTEST_INITIAL_SOC,
    Q2_BACKTEST_START,
    Q2_HISTORY_WINDOWS_DAYS,
    Q2_JANUARY_BACKTEST_FILE,
    Q2_NET_LOAD_QUANTILES,
    Q2_RESERVE_PENALTY_PRICE_MULTIPLIERS,
    Q2_RESERVE_SOC_CANDIDATES,
    Q2_VALIDATION_TOLERANCE,
)
from src.data_io import load_q2_actual_energy
from src.dispatch_q2_realtime import dispatch_q2_realtime
from src.forecast_q2 import forecast_net_load_quantile
from src.model_q2_day_ahead import get_highs_solver, solve_q2_day_ahead
from src.validation_q2 import validate_q2_january_backtest


def _common_backtest_dates() -> pd.DatetimeIndex:
    """Return the declared common January evaluation dates for all windows."""
    return pd.date_range(Q2_BACKTEST_START, Q2_BACKTEST_END, freq="D")


def _run_one_combination(
    actual: pd.DataFrame,
    price: pd.DataFrame,
    dates: pd.DatetimeIndex,
    window_days: int,
    quantile: float,
    reserve_soc_kwh: float,
    reserve_penalty_yuan_per_kwh: float,
    solver,
) -> tuple[dict, list[dict]]:
    """Run one causal cross-day path; no daily result file is written."""
    soc = float(Q2_BACKTEST_INITIAL_SOC)
    day_runs: list[dict] = []
    for day in dates:
        forecast = forecast_net_load_quantile(actual, day, window_days, quantile)
        plan, plan_metadata = solve_q2_day_ahead(
            forecast=forecast,
            price=price,
            initial_soc_kwh=soc,
            reserve_soc_kwh=reserve_soc_kwh,
            reserve_penalty_yuan_per_kwh=reserve_penalty_yuan_per_kwh,
            solver=solver,
        )
        actual_day = actual.loc[actual["date"] == day].copy()
        execution, rt_metadata = dispatch_q2_realtime(
            day_ahead_plan=plan,
            actual_day=actual_day,
            initial_soc_kwh=soc,
            tolerance=Q2_VALIDATION_TOLERANCE,
        )
        day_runs.append(
            {
                "date": day,
                "forecast": forecast,
                "plan": plan,
                "plan_metadata": plan_metadata,
                "execution": execution,
                "rt_metadata": rt_metadata,
            }
        )
        soc = rt_metadata["rt_final_soc_kwh"]
    return validate_q2_january_backtest(day_runs, Q2_VALIDATION_TOLERANCE), day_runs


def run_q2_january_backtest(output_path=Q2_JANUARY_BACKTEST_FILE) -> dict:
    """Select A+ candidates solely with the January 22--31 common backtest.

    The output is deliberately limited to one parameter-comparison CSV.  It
    neither runs the February--December workflow nor writes result2.xlsx.
    """
    actual, price = load_q2_actual_energy()
    dates = _common_backtest_dates()
    if dates[0] != pd.Timestamp("2025-01-22") or dates[-1] != pd.Timestamp("2025-01-31"):
        raise RuntimeError("Question-2 common backtest period must remain 2025-01-22 through 2025-01-31")
    max_price = float(price["price_yuan_per_kwh"].max())
    solver, solver_name = get_highs_solver()

    records: list[dict] = []
    retained_runs: dict[tuple[float, ...], list[dict]] = {}
    for window_days, quantile, reserve_soc, penalty_multiplier in product(
        Q2_HISTORY_WINDOWS_DAYS,
        Q2_NET_LOAD_QUANTILES,
        Q2_RESERVE_SOC_CANDIDATES,
        Q2_RESERVE_PENALTY_PRICE_MULTIPLIERS,
    ):
        rho = float(penalty_multiplier * max_price)
        validation, day_runs = _run_one_combination(
            actual=actual,
            price=price,
            dates=dates,
            window_days=int(window_days),
            quantile=float(quantile),
            reserve_soc_kwh=float(reserve_soc),
            reserve_penalty_yuan_per_kwh=rho,
            solver=solver,
        )
        key = (float(window_days), float(quantile), float(reserve_soc), float(rho))
        retained_runs[key] = day_runs
        records.append(
            {
                "w_days": int(window_days),
                "q": float(quantile),
                "Sres_kwh": float(reserve_soc),
                "rho_yuan_per_kwh": rho,
                "rho_price_multiplier": float(penalty_multiplier),
                "plan_purchase_cost_yuan": validation["plan_purchase_cost_yuan_recomputed"],
                "emergency_purchase_cost_yuan": validation["emergency_purchase_cost_yuan_recomputed"],
                "total_cost_yuan": validation["total_cost_yuan_recomputed"],
                "emergency_purchase_kwh": validation["total_emergency_purchase_kwh"],
                "soc_violation_count": int(
                    validation["soc_min_kwh"] < 1200.0 - Q2_VALIDATION_TOLERANCE
                    or validation["soc_max_kwh"] > 10800.0 + Q2_VALIDATION_TOLERANCE
                ),
                "validation_passed": bool(validation["passed"]),
                "no_future_data_leakage": bool(validation["no_future_data_leakage"]),
                "emergency_charge_overlap_count": validation["emergency_charge_overlap_count"],
                "max_balance_error_kwh": validation["max_energy_balance_error_kwh"],
                "soc_min_kwh": validation["soc_min_kwh"],
                "soc_max_kwh": validation["soc_max_kwh"],
                "final_soc_kwh": day_runs[-1]["rt_metadata"]["rt_final_soc_kwh"],
                "backtest_start": dates[0].date().isoformat(),
                "backtest_end": dates[-1].date().isoformat(),
                "solver": solver_name,
            }
        )

    comparison = pd.DataFrame(records)
    valid = comparison.loc[comparison["validation_passed"]].copy()
    if valid.empty:
        raise RuntimeError("No Question-2 candidate passed causal and physical validation")
    # Stable tie-break: total cost, emergency energy, then simpler/shorter window.
    selected = valid.sort_values(
        ["total_cost_yuan", "emergency_purchase_kwh", "w_days", "q", "Sres_kwh", "rho_yuan_per_kwh"],
        kind="stable",
    ).index[0]
    comparison["rank_among_valid"] = pd.NA
    valid_ranked = valid.sort_values(
        ["total_cost_yuan", "emergency_purchase_kwh", "w_days", "q", "Sres_kwh", "rho_yuan_per_kwh"],
        kind="stable",
    )
    comparison.loc[valid_ranked.index, "rank_among_valid"] = range(1, len(valid_ranked) + 1)
    comparison["selected"] = comparison.index == selected
    comparison = comparison.sort_values(["validation_passed", "rank_among_valid"], ascending=[False, True], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.10f")

    selected_row = comparison.loc[comparison["selected"]].iloc[0].to_dict()
    selected_key = (
        float(selected_row["w_days"]),
        float(selected_row["q"]),
        float(selected_row["Sres_kwh"]),
        float(selected_row["rho_yuan_per_kwh"]),
    )
    first_execution = retained_runs[selected_key][0]["execution"].copy()
    return {
        "comparison": comparison,
        "comparison_path": output_path,
        "selected_parameters": selected_row,
        "selected_first_day_execution": first_execution,
    }


if __name__ == "__main__":
    outcome = run_q2_january_backtest()
    print(f"Question-2 January comparison: {outcome['comparison_path']}")
    print("Selected parameters:")
    print({key: outcome["selected_parameters"][key] for key in ("w_days", "q", "Sres_kwh", "rho_yuan_per_kwh")})
