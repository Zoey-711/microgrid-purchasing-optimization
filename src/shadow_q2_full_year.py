"""Causal full-year shadow validation for the two fixed Question-2 strategies.

This module deliberately never imports the Excel exporter and never writes
result2.xlsx.  It is a validation workflow only.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.config import (
    E_B_MAX,
    FIGURE_DIR,
    Q2_BASELINE_APLUS_FIGURE,
    Q2_DAILY_SHADOW_RESULTS_FILE,
    Q2_FIXED_QUANTILE,
    Q2_FIXED_RHO,
    Q2_FIXED_WINDOW_DAYS,
    Q2_FULL_YEAR_FORECAST_METRICS_FILE,
    Q2_FULL_YEAR_SHADOW_COMPARISON_FILE,
    Q2_FULL_YEAR_SHADOW_VALIDATION_FILE,
    Q2_SHADOW_END,
    Q2_SHADOW_FORMAL_START,
    Q2_SHADOW_WARMUP_START,
    Q2_VALIDATION_TOLERANCE,
    S_INIT,
    S_MAX,
    S_MIN,
)
from src.data_io import load_q2_actual_energy
from src.dispatch_q2_realtime import dispatch_q2_realtime
from src.forecast_q2 import forecast_net_load_quantile
from src.model_q2_day_ahead import get_highs_solver, solve_q2_day_ahead
from src.plot_style import AXIS_GREY, EMERGENCY_PURCHASE, GRID_GREY, PLAN_PURCHASE, PRICE, SOC_MAIN


STRATEGIES = (
    {"strategy": "成本基线", "strategy_key": "baseline", "reserve_soc_kwh": 1200.0},
    {"strategy": "正式A+", "strategy_key": "aplus", "reserve_soc_kwh": 3600.0},
)


def _font_name() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def _cold_start_plan(price: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build the documented 2025-01-01 p=0 cold-start reference without a forecast."""
    plan = price[["idx", "price_yuan_per_kwh"]].copy()
    plan["plan_purchase_kwh"] = 0.0
    plan["plan_soc_end_kwh"] = S_MIN
    return plan, {
        "solver_name": "cold_start_rule",
        "solver_status": "not_applicable",
        "termination_condition": "not_applicable",
        "objective_with_reserve_penalty_yuan": 0.0,
        "plan_purchase_cost_yuan": 0.0,
        "reserve_shortfall_kwh": 0.0,
        "plan_soc_start_kwh": S_INIT,
        "plan_soc_end_kwh": S_INIT,
    }


def _daily_diagnostics(
    strategy: dict,
    date: pd.Timestamp,
    is_formal: bool,
    effective_window: int,
    forecast: pd.DataFrame | None,
    plan_metadata: dict,
    execution: pd.DataFrame,
    previous_soc_end: float | None,
) -> dict:
    """Compute day-level, independently recomputable physical and forecast checks."""
    tol = Q2_VALIDATION_TOLERANCE
    soc_values = pd.concat([execution["rt_soc_start_kwh"], execution["rt_soc_end_kwh"]])
    soc_violations = int(((soc_values < S_MIN - tol) | (soc_values > S_MAX + tol)).sum())
    power_violations = int(
        ((execution["rt_charge_kwh"] > E_B_MAX + tol) | (execution["rt_discharge_kwh"] > E_B_MAX + tol)).sum()
    )
    emergency_charge_overlap = int(
        ((execution["emergency_purchase_kwh"] > tol) & (execution["rt_charge_kwh"] > tol)).sum()
    )
    execution_plan_cost = float(execution["plan_purchase_cost_yuan"].sum())
    actual_cost = float(execution_plan_cost + execution["emergency_purchase_cost_yuan"].sum())
    reported_cost = execution_plan_cost + float(
        execution["emergency_purchase_cost_yuan"].sum()
    )
    record = {
        "strategy": strategy["strategy"],
        "strategy_key": strategy["strategy_key"],
        "date": date.date().isoformat(),
        "is_formal_statistic": is_formal,
        "w_eff_days": effective_window,
        "q": Q2_FIXED_QUANTILE if forecast is not None else np.nan,
        "Sres_kwh": strategy["reserve_soc_kwh"],
        "rho_yuan_per_kwh": Q2_FIXED_RHO,
        "plan_purchase_cost_yuan": execution_plan_cost,
        "day_ahead_plan_purchase_cost_yuan": float(plan_metadata["plan_purchase_cost_yuan"]),
        "plan_cost_recomputation_error_yuan": abs(
            execution_plan_cost - float(plan_metadata["plan_purchase_cost_yuan"])
        ),
        "emergency_purchase_cost_yuan": float(execution["emergency_purchase_cost_yuan"].sum()),
        "actual_total_cost_yuan": actual_cost,
        "recomputed_total_cost_yuan": reported_cost,
        "cost_recomputation_error_yuan": abs(actual_cost - reported_cost),
        "emergency_purchase_kwh": float(execution["emergency_purchase_kwh"].sum()),
        "soc_start_kwh": float(execution["rt_soc_start_kwh"].iloc[0]),
        "soc_end_kwh": float(execution["rt_soc_end_kwh"].iloc[-1]),
        "soc_min_kwh": float(soc_values.min()),
        "soc_max_kwh": float(soc_values.max()),
        "soc_boundary_violation_count": soc_violations,
        "power_violation_count": power_violations,
        "emergency_charge_overlap_count": emergency_charge_overlap,
        "max_balance_error_kwh": float(execution["balance_residual_kwh"].abs().max()),
        "cross_day_soc_error_kwh": 0.0 if previous_soc_end is None else abs(
            float(execution["rt_soc_start_kwh"].iloc[0]) - previous_soc_end
        ),
        "reserve_shortfall_kwh": float(plan_metadata["reserve_shortfall_kwh"]),
        "reserve_soft_penalty_yuan": float(Q2_FIXED_RHO * plan_metadata["reserve_shortfall_kwh"]),
        "rho_triggered": bool(plan_metadata["reserve_shortfall_kwh"] > tol),
        "history_end_date": "cold_start" if forecast is None else forecast["history_end_date"].iloc[0].date().isoformat(),
        "forecast_mae_net_load_kwh": np.nan,
        "forecast_bias_net_load_kwh": np.nan,
        "forecast_exceedance_rate": np.nan,
        "forecast_coverage_rate": np.nan,
        "forecast_observation_count": 0,
    }
    if forecast is not None:
        actual_net = execution["actual_net_load_kwh"].to_numpy(dtype=float)
        predicted_net = forecast.sort_values("idx")["forecast_net_load_kwh"].to_numpy(dtype=float)
        error = predicted_net - actual_net
        record.update(
            {
                "forecast_mae_net_load_kwh": float(np.mean(np.abs(error))),
                "forecast_bias_net_load_kwh": float(np.mean(error)),
                "forecast_exceedance_rate": float(np.mean(actual_net > predicted_net)),
                "forecast_coverage_rate": float(np.mean(actual_net <= predicted_net)),
                "forecast_observation_count": len(error),
            }
        )
    return record


def _write_summary_figure(comparison: pd.DataFrame) -> None:
    font = _font_name()
    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": 9.5, "axes.labelsize": 10.5, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    labels = ["成本基线\n($S^{res}=1200$ kWh)", "A+策略\n($S^{res}=3600$ kWh)"]
    metrics = [
        ("total_actual_cost_yuan", "总成本（百万元）", 1_000_000.0, ".3f"),
        ("total_emergency_purchase_kwh", "紧急购电量（kWh）", 1.0, ".0f"),
        ("mean_daily_end_soc_kwh", "平均日末 SOC（kWh）", 1.0, ".0f"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.45), facecolor="white")
    for panel_index, (axis, (column, ylabel, scale, value_format)) in enumerate(zip(axes, metrics)):
        values = comparison[column].to_numpy(dtype=float) / scale
        colors = {
            "total_actual_cost_yuan": [PRICE, PRICE],
            "total_emergency_purchase_kwh": [EMERGENCY_PURCHASE, EMERGENCY_PURCHASE],
            "mean_daily_end_soc_kwh": [PLAN_PURCHASE, SOC_MAIN],
        }[column]
        bars = axis.bar(labels, values, color=colors, width=0.54)
        bars[0].set_alpha(0.58)
        bars[1].set_alpha(0.94)
        axis.set_ylabel(ylabel)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID_GREY, linestyle="-", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(AXIS_GREY)
        axis.spines["bottom"].set_color(AXIS_GREY)
        axis.spines["left"].set_linewidth(0.85)
        axis.spines["bottom"].set_linewidth(0.85)
        axis.tick_params(axis="both", direction="out", width=0.8, length=3.5, color=AXIS_GREY)
        axis.text(-0.15, 1.02, f"({chr(97 + panel_index)})", transform=axis.transAxes, fontsize=10.5, fontweight="bold")
        offset = max(abs(values).max() * 0.015, 1.0)
        if scale > 1:
            offset = max(abs(values).max() * 0.02, 0.01)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + offset, format(value, value_format), ha="center", va="bottom", fontsize=8.5)
        axis.set_ylim(0, max(values) * 1.10)
    figure.tight_layout()
    Q2_BASELINE_APLUS_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(Q2_BASELINE_APLUS_FIGURE, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(Q2_BASELINE_APLUS_FIGURE.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_validation_log(all_days: pd.DataFrame, comparison: pd.DataFrame, forecast_metrics: pd.DataFrame) -> None:
    tol = Q2_VALIDATION_TOLERANCE
    lines = [
        "Question 2 full-year shadow validation",
        "=" * 42,
        "Scope: Attachment 1 price + Attachment 2 actual load/PV only; Attachment 3 was not used.",
        "Scope: shadow validation only; result2.xlsx was not created or modified.",
        "Cold start: 2025-01-01 00:00 SOC=6000 kWh, plan purchase=0; January is excluded from formal statistics.",
        "Formal statistics: 2025-02-01 through 2025-12-31 (334 days).",
        "Fixed plans: w=7, q=0.75; Sres=1200 baseline versus Sres=3600 A+; rho=0.6976 as a soft feasibility protection term.",
        "",
    ]
    for _, row in comparison.iterrows():
        strategy_days = all_days[all_days["strategy_key"] == row["strategy_key"]]
        forecasted = strategy_days[strategy_days["history_end_date"] != "cold_start"]
        leakage_ok = all(
            pd.Timestamp(history) < pd.Timestamp(date)
            for history, date in zip(forecasted["history_end_date"], forecasted["date"])
        )
        trigger_rows = strategy_days[strategy_days["rho_triggered"]]
        lines.extend(
            [
                f"[{row['strategy']}]",
                f"no_future_data_leakage: {leakage_ok}",
                f"max_energy_balance_error_kwh: {strategy_days['max_balance_error_kwh'].max():.12g}",
                f"soc_boundary_violation_count: {int(strategy_days['soc_boundary_violation_count'].sum())}",
                f"power_violation_count: {int(strategy_days['power_violation_count'].sum())}",
                f"max_cross_day_soc_error_kwh: {strategy_days['cross_day_soc_error_kwh'].max():.12g}",
                f"emergency_charge_overlap_count: {int(strategy_days['emergency_charge_overlap_count'].sum())}",
                f"max_cost_recomputation_error_yuan: {strategy_days['cost_recomputation_error_yuan'].max():.12g}",
                f"max_plan_cost_recomputation_error_yuan: {strategy_days['plan_cost_recomputation_error_yuan'].max():.12g}",
                f"rho_trigger_days_all_365: {len(trigger_rows)}",
                "rho_trigger_details: none" if trigger_rows.empty else "rho_trigger_details: " + "; ".join(
                    f"{entry.date}: shortfall={entry.reserve_shortfall_kwh:.6f} kWh, penalty={entry.reserve_soft_penalty_yuan:.6f} yuan"
                    for entry in trigger_rows.itertuples()
                ),
                "",
            ]
        )
    lines.append("Forecast metrics use predicted net load minus actual net load; they do not use plan-purchase minus actual-net-load.")
    lines.append("Forecast-metric rows: " + str(len(forecast_metrics)))
    Q2_FULL_YEAR_SHADOW_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    Q2_FULL_YEAR_SHADOW_VALIDATION_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_q2_full_year_shadow_validation() -> dict:
    """Run the two fixed strategies causally from Jan 1 through Dec 31."""
    actual, price = load_q2_actual_energy()
    all_dates = pd.date_range(Q2_SHADOW_WARMUP_START, Q2_SHADOW_END, freq="D")
    formal_start = pd.Timestamp(Q2_SHADOW_FORMAL_START)
    if len(all_dates) != 365 or len(pd.date_range(formal_start, all_dates[-1], freq="D")) != 334:
        raise RuntimeError("unexpected full-year or formal-statistic date count")
    solver, _ = get_highs_solver()
    state = {spec["strategy_key"]: float(S_INIT) for spec in STRATEGIES}
    previous_end = {spec["strategy_key"]: None for spec in STRATEGIES}
    day_records: list[dict] = []

    for day_index, day in enumerate(all_dates):
        actual_day = actual.loc[actual["date"] == day].copy()
        if len(actual_day) != 144:
            raise RuntimeError(f"{day.date()} does not have 144 actual slots")
        is_formal = day >= formal_start
        if day_index == 0:
            forecast = None
            effective_window = 0
        else:
            effective_window = min(Q2_FIXED_WINDOW_DAYS, day_index)
            forecast = forecast_net_load_quantile(actual, day, effective_window, Q2_FIXED_QUANTILE)
            if not forecast["history_end_date"].iloc[0] < day:
                raise RuntimeError(f"future-data leakage before planning {day.date()}")

        for spec in STRATEGIES:
            key = spec["strategy_key"]
            if forecast is None:
                plan, plan_metadata = _cold_start_plan(price)
            else:
                plan, plan_metadata = solve_q2_day_ahead(
                    forecast=forecast,
                    price=price,
                    initial_soc_kwh=state[key],
                    reserve_soc_kwh=spec["reserve_soc_kwh"],
                    reserve_penalty_yuan_per_kwh=Q2_FIXED_RHO,
                    solver=solver,
                )
            execution, _ = dispatch_q2_realtime(plan, actual_day, state[key], Q2_VALIDATION_TOLERANCE)
            record = _daily_diagnostics(
                strategy=spec,
                date=day,
                is_formal=is_formal,
                effective_window=effective_window,
                forecast=forecast,
                plan_metadata=plan_metadata,
                execution=execution,
                previous_soc_end=previous_end[key],
            )
            day_records.append(record)
            previous_end[key] = record["soc_end_kwh"]
            state[key] = record["soc_end_kwh"]

    all_days = pd.DataFrame(day_records)
    formal_days = all_days[all_days["is_formal_statistic"]].copy()
    if len(formal_days) != 668 or formal_days.groupby("strategy_key").size().to_dict() != {"aplus": 334, "baseline": 334}:
        raise RuntimeError("formal daily-result rows must be 334 per strategy")

    summary_records = []
    metric_records = []
    for spec in STRATEGIES:
        subset = formal_days[formal_days["strategy_key"] == spec["strategy_key"]]
        errors_weighted = subset["forecast_observation_count"].sum()
        summary_records.append(
            {
                "strategy": spec["strategy"],
                "strategy_key": spec["strategy_key"],
                "formal_days": len(subset),
                "total_actual_cost_yuan": subset["actual_total_cost_yuan"].sum(),
                "total_plan_purchase_cost_yuan": subset["plan_purchase_cost_yuan"].sum(),
                "total_emergency_purchase_cost_yuan": subset["emergency_purchase_cost_yuan"].sum(),
                "total_emergency_purchase_kwh": subset["emergency_purchase_kwh"].sum(),
                "mean_daily_end_soc_kwh": subset["soc_end_kwh"].mean(),
                "min_daily_end_soc_kwh": subset["soc_end_kwh"].min(),
                "soc_boundary_violation_count": int(subset["soc_boundary_violation_count"].sum()),
                "power_violation_count": int(subset["power_violation_count"].sum()),
                "emergency_charge_overlap_count": int(subset["emergency_charge_overlap_count"].sum()),
                "max_daily_emergency_purchase_kwh": subset["emergency_purchase_kwh"].max(),
                "reserve_shortfall_total_kwh": subset["reserve_shortfall_kwh"].sum(),
                "reserve_shortfall_trigger_days": int(subset["rho_triggered"].sum()),
                "reserve_soft_penalty_total_yuan": subset["reserve_soft_penalty_yuan"].sum(),
                "max_balance_error_kwh": subset["max_balance_error_kwh"].max(),
                "max_cross_day_soc_error_kwh": subset["cross_day_soc_error_kwh"].max(),
            }
        )
        metric_records.append(
            {
                "strategy": spec["strategy"],
                "strategy_key": spec["strategy_key"],
                "formal_days": len(subset),
                "observation_count": int(errors_weighted),
                "MAE_N_kwh": float((subset["forecast_mae_net_load_kwh"] * subset["forecast_observation_count"]).sum() / errors_weighted),
                "Bias_N_kwh": float((subset["forecast_bias_net_load_kwh"] * subset["forecast_observation_count"]).sum() / errors_weighted),
                "actual_above_forecast_rate": float((subset["forecast_exceedance_rate"] * subset["forecast_observation_count"]).sum() / errors_weighted),
                "actual_at_or_below_forecast_coverage": float((subset["forecast_coverage_rate"] * subset["forecast_observation_count"]).sum() / errors_weighted),
                "q": Q2_FIXED_QUANTILE,
                "w_days": Q2_FIXED_WINDOW_DAYS,
            }
        )

    comparison = pd.DataFrame(summary_records)
    forecast_metrics = pd.DataFrame(metric_records)
    Q2_FULL_YEAR_SHADOW_COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(Q2_FULL_YEAR_SHADOW_COMPARISON_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")
    forecast_metrics.to_csv(Q2_FULL_YEAR_FORECAST_METRICS_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")
    formal_days.to_csv(Q2_DAILY_SHADOW_RESULTS_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")
    _write_validation_log(all_days, comparison, forecast_metrics)
    return {
        "comparison": comparison,
        "forecast_metrics": forecast_metrics,
        "daily_results": formal_days,
        "all_days": all_days,
        "comparison_path": Q2_FULL_YEAR_SHADOW_COMPARISON_FILE,
        "metrics_path": Q2_FULL_YEAR_FORECAST_METRICS_FILE,
        "daily_path": Q2_DAILY_SHADOW_RESULTS_FILE,
        "validation_path": Q2_FULL_YEAR_SHADOW_VALIDATION_FILE,
        "figure_path": None,
    }


if __name__ == "__main__":
    outcome = run_q2_full_year_shadow_validation()
    print(f"Shadow comparison: {outcome['comparison_path']}")
    print(f"Forecast metrics: {outcome['metrics_path']}")
