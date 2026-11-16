"""Redraw the frozen Q2 V2 paper figures without re-solving any model.

All inputs are frozen Stage-4 replay tables.  This module only reads CSVs and
writes visual assets plus a short visual-check log.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.paper_figure_theme import (
    ACTUAL_BLUE,
    BOUND_GREY,
    CHARGE_GREEN,
    DISCHARGE_ORANGE,
    EMERGENCY_RED,
    FORECAST_BLUE,
    GRID_GREY,
    PLAN_BLUE,
    PV_GREEN,
    TEXT_GREY,
    compact_legend,
    configure_chinese_theme,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE4 = ROOT / "outputs" / "q2_v2" / "stage4"
FIGDIR = ROOT / "outputs" / "figures" / "q2_v2"
LOG = ROOT / "outputs" / "q2_v2" / "q2_v2_redesigned_figure_check.txt"
DT = 1.0 / 6.0
CASE_DATE = "2025-06-01"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slots = pd.read_csv(STAGE4 / "q2_v2_slot_results.csv")
    plans = pd.read_csv(STAGE4 / "q2_v2_day_ahead_plans.csv")
    daily = pd.read_csv(STAGE4 / "q2_v2_daily_comparison.csv")
    monthly = pd.read_csv(STAGE4 / "q2_v2_monthly_comparison.csv")
    bands = pd.read_csv(STAGE4 / "q2_v2_emergency_by_price_band.csv")
    for frame in (slots, plans, daily, monthly):
        if "date" in frame:
            frame["date"] = frame["date"].astype(str)
    return slots, plans, daily, monthly, bands


def save(fig: plt.Figure, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_bars(axis, bars, fmt: str, offset: float) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color=TEXT_GREY,
        )


def figure_annual_legacy(monthly: pd.DataFrame, bands: pd.DataFrame) -> None:
    row = monthly.copy()
    x = np.arange(2)
    names = ["动态价值\n调度", "贪心\n对照"]
    plan = np.array([row["plan_purchase_cost_yuan_dp50"].sum(), row["plan_purchase_cost_yuan_greedy"].sum()]) / 1e6
    emergency = np.array([row["emergency_purchase_cost_yuan_dp50"].sum(), row["emergency_purchase_cost_yuan_greedy"].sum()]) / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    a = ax.bar(x, plan, width=0.52, color=PLAN_BLUE, label="计划购电费")
    b = ax.bar(x, emergency, width=0.52, bottom=plan, color=EMERGENCY_RED, label="紧急购电费")
    annotate_bars(ax, [a[0], a[1]], "{:.3f}", 2)
    for i, total in enumerate(plan + emergency):
        ax.annotate(f"合计 {total:.3f}", (i, total), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=7.2)
    ax.set_xticks(x, names)
    ax.set_ylabel("费用（百万元）")
    ax.set_ylim(0, float(total.max()) * 1.085)
    panel_title(ax, "(a)", "全年费用构成")
    style_axis(ax)
    compact_legend(ax, loc="lower right", bbox_to_anchor=(1.0, 1.085), ncol=2)

    ax = axes[0, 1]
    band_order = ["低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"]
    label_map = {"低价(<0.6)": "低价\n<0.6", "中价([0.6,1.0))": "中价\n[0.6,1.0)", "高价(>=1.0)": "高价\n≥1.0"}
    dp = bands.set_index(["policy_id", "price_band"])["emergency_purchase_kwh"]
    vals_dp = [dp[("main_dp50", k)] / 1e4 for k in band_order]
    vals_g = [dp[("closed_loop_greedy", k)] / 1e4 for k in band_order]
    px = np.arange(3)
    p1 = ax.bar(px - 0.18, vals_dp, width=0.36, color=PV_GREEN, label="动态价值调度")
    p2 = ax.bar(px + 0.18, vals_g, width=0.36, color=BOUND_GREY, label="贪心对照")
    annotate_bars(ax, p1, "{:.2f}", 2)
    annotate_bars(ax, p2, "{:.2f}", 2)
    ax.set_xticks(px, [label_map[v] for v in band_order])
    ax.set_ylabel("紧急购电量（万 kWh）")
    panel_title(ax, "(b)", "紧急购电的电价区间分布")
    style_axis(ax)
    compact_legend(ax, loc="upper left", ncol=1)

    ax = axes[1, 0]
    saving = (row["actual_total_cost_yuan_greedy"] - row["actual_total_cost_yuan_dp50"]) / 1e4
    mx = np.arange(len(row))
    colors = np.where(saving >= 0, PV_GREEN, EMERGENCY_RED)
    bars = ax.bar(mx, saving, width=0.62, color=colors)
    ax.axhline(0, color=BOUND_GREY, linewidth=0.85)
    ax.set_xticks(mx, [str(v).split("-")[1] + "月" for v in row["month"]])
    ax.set_ylabel("费用节省额（万元）")
    panel_title(ax, "(c)", "各月费用节省额")
    style_axis(ax)
    ax.text(0.99, 0.96, "贪心实际总费用 − 动态价值实际总费用", transform=ax.transAxes, ha="right", va="top", fontsize=7.1, color=BOUND_GREY)

    ax = axes[1, 1]
    row["date"] = pd.to_datetime(row["month"] + "-01")
    cumulative = saving.cumsum()
    ax.plot(np.arange(1, len(row) + 1), cumulative, color=ACTUAL_BLUE, linewidth=1.8, marker="o", markersize=3.2)
    ax.fill_between(np.arange(1, len(row) + 1), 0, cumulative, color=ACTUAL_BLUE, alpha=0.10)
    ax.axhline(0, color=BOUND_GREY, linewidth=0.85)
    ax.set_xticks(np.arange(1, len(row) + 1), [str(v).split("-")[1] + "月" for v in row["month"]])
    ax.set_ylabel("累计费用节省（万元）")
    panel_title(ax, "(d)", "累计费用节省")
    style_axis(ax)
    ax.annotate(f"{cumulative.iloc[-1]:.2f}", (len(row), cumulative.iloc[-1]), xytext=(-3, 6), textcoords="offset points", ha="right", fontsize=7.5)
    save(fig, "q2_v2_annual_economic_mechanism")


def figure_annual(monthly: pd.DataFrame, bands: pd.DataFrame) -> None:
    """Draw the annual comparison directly from the frozen policy rows.

    The final comparison table is the source of truth for panel (a).  Monthly
    and price-band tables are independently checked against it before any
    figure is written, preventing stale labels or reused arrays from silently
    producing a visually plausible but numerically inconsistent chart.
    """
    comparison = pd.read_csv(STAGE4 / "q2_v2_final_comparison.csv")
    comparison = comparison.set_index("policy_id", verify_integrity=True)
    policy_ids = ["main_dp50", "closed_loop_greedy"]
    missing = [policy for policy in policy_ids if policy not in comparison.index]
    if missing:
        raise ValueError(f"Frozen comparison table is missing policies: {missing}")

    frozen = comparison.loc[policy_ids]
    plan_yuan = frozen["plan_purchase_cost_yuan"].to_numpy(float)
    emergency_yuan = frozen["emergency_purchase_cost_yuan"].to_numpy(float)
    total_yuan = frozen["actual_total_cost_yuan"].to_numpy(float)
    identity_error = np.abs(plan_yuan + emergency_yuan - total_yuan)
    if float(identity_error.max()) > 1e-6:
        raise ValueError(f"Annual billing identity failed: {identity_error}")

    annual_saving_yuan = float(total_yuan[1] - total_yuan[0])
    expected_saving_yuan = 106661.382243908
    if abs(annual_saving_yuan - expected_saving_yuan) > 1e-6:
        raise ValueError(
            f"Frozen annual saving mismatch: {annual_saving_yuan} vs {expected_saving_yuan}"
        )

    monthly_saving_yuan = (
        monthly["actual_total_cost_yuan_greedy"].to_numpy(float)
        - monthly["actual_total_cost_yuan_dp50"].to_numpy(float)
    )
    monthly_sum_error = abs(float(monthly_saving_yuan.sum()) - annual_saving_yuan)
    if monthly_sum_error > 1e-6:
        raise ValueError(f"Monthly savings do not sum to annual saving: {monthly_sum_error}")

    band_order = ["低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"]
    band_lookup = bands.set_index(["policy_id", "price_band"])["emergency_purchase_kwh"]
    band_values = {
        policy: np.array([float(band_lookup.loc[(policy, band)]) for band in band_order])
        for policy in policy_ids
    }
    band_errors = np.array(
        [
            abs(band_values[policy].sum() - float(frozen.loc[policy, "emergency_purchase_kwh"]))
            for policy in policy_ids
        ]
    )
    if float(band_errors.max()) > 1e-6:
        raise ValueError(f"Price-band emergency energy mismatch: {band_errors}")

    plan = plan_yuan / 1e6
    emergency = emergency_yuan / 1e6
    total = total_yuan / 1e6
    names = ["动态价值调度", "贪心对照"]
    x = np.arange(2)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.8), constrained_layout=True)

    ax = axes[0, 0]
    plan_bars = ax.bar(x, plan, width=0.52, color=PLAN_BLUE, label="计划购电费")
    emergency_bars = ax.bar(
        x, emergency, width=0.52, bottom=plan, color=EMERGENCY_RED, label="紧急购电费"
    )
    for i, (plan_bar, emergency_bar) in enumerate(zip(plan_bars, emergency_bars)):
        ax.text(
            plan_bar.get_x() + plan_bar.get_width() / 2,
            plan[i] * 0.52,
            f"{plan[i]:.3f}",
            ha="center",
            va="center",
            fontsize=7.4,
            color="white",
        )
        ax.text(
            emergency_bar.get_x() + emergency_bar.get_width() / 2,
            plan[i] + emergency[i] * 0.50,
            f"{emergency[i]:.3f}",
            ha="center",
            va="center",
            fontsize=7.2,
            color="white",
        )
        ax.annotate(
            f"合计 {total[i]:.3f}",
            (i, total[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color=TEXT_GREY,
        )
    ax.set_xticks(x, names)
    ax.set_ylabel("费用（百万元）")
    ax.set_ylim(0, float(total.max()) * 1.085)
    panel_title(ax, "(a)", "全年费用构成")
    style_axis(ax)
    compact_legend(ax, loc="lower right", bbox_to_anchor=(1.0, 1.085), ncol=2)

    ax = axes[0, 1]
    label_map = {
        "低价(<0.6)": "低价\n<0.6",
        "中价([0.6,1.0))": "中价\n[0.6,1.0)",
        "高价(>=1.0)": "高价\n≥1.0",
    }
    px = np.arange(3)
    vals_dp = band_values["main_dp50"] / 1e4
    vals_greedy = band_values["closed_loop_greedy"] / 1e4
    dp_bars = ax.bar(
        px - 0.18, vals_dp, width=0.36, color=PV_GREEN, label="动态价值调度"
    )
    greedy_bars = ax.bar(
        px + 0.18, vals_greedy, width=0.36, color=BOUND_GREY, label="贪心对照"
    )
    annotate_bars(ax, dp_bars, "{:.2f}", 2)
    annotate_bars(ax, greedy_bars, "{:.2f}", 2)
    ax.set_xticks(px, [label_map[band] for band in band_order])
    ax.set_ylabel("紧急购电量（万 kWh）")
    ax.set_ylim(0, float(max(vals_dp.max(), vals_greedy.max())) * 1.10)
    panel_title(ax, "(b)", "紧急购电的电价区间分布")
    style_axis(ax)
    compact_legend(ax, loc="upper left", ncol=1)

    ax = axes[1, 0]
    monthly_saving_10k = monthly_saving_yuan / 1e4
    mx = np.arange(len(monthly))
    colors = np.where(monthly_saving_10k >= 0, PV_GREEN, EMERGENCY_RED)
    bars = ax.bar(mx, monthly_saving_10k, width=0.62, color=colors)
    ax.axhline(0, color=BOUND_GREY, linewidth=0.85)
    ax.set_xticks(mx, [str(value).split("-")[1] + "月" for value in monthly["month"]])
    ax.set_ylabel("费用节省额（万元）")
    panel_title(ax, "(c)", "各月费用节省额")
    style_axis(ax)
    annotate_bars(ax, bars, "{:.2f}", 2)

    ax = axes[1, 1]
    cumulative_10k = np.cumsum(monthly_saving_yuan) / 1e4
    cx = np.arange(1, len(monthly) + 1)
    ax.plot(
        cx,
        cumulative_10k,
        color=ACTUAL_BLUE,
        linewidth=1.8,
        marker="o",
        markersize=3.2,
    )
    ax.fill_between(cx, 0, cumulative_10k, color=ACTUAL_BLUE, alpha=0.10)
    ax.axhline(0, color=BOUND_GREY, linewidth=0.85)
    ax.set_xticks(cx, [str(value).split("-")[1] + "月" for value in monthly["month"]])
    ax.set_ylabel("累计费用节省（万元）")
    ax.set_ylim(min(0.0, float(cumulative_10k.min())) - 0.15, float(cumulative_10k.max()) * 1.08)
    panel_title(ax, "(d)", "累计费用节省")
    style_axis(ax)
    for i, (month_x, cumulative_value) in enumerate(zip(cx, cumulative_10k)):
        # Alternate nearby labels vertically; keep edge labels inside the axes.
        y_offset = 4 if i % 2 == 0 else 7
        horizontal_alignment = "center"
        x_offset = 0
        if i == 0:
            horizontal_alignment = "left"
            x_offset = 2
        elif i == len(cx) - 1:
            horizontal_alignment = "right"
            x_offset = -2
        ax.annotate(
            f"{cumulative_value:.2f}",
            (month_x, cumulative_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="bottom",
            fontsize=7.2,
            color=TEXT_GREY,
        )

    save(fig, "q2_v2_annual_economic_mechanism")
    audit_lines = [
        "Q2 V2 annual economic mechanism figure data audit",
        "=" * 55,
        "source_annual: outputs/q2_v2/stage4/q2_v2_final_comparison.csv",
        "source_monthly: outputs/q2_v2/stage4/q2_v2_monthly_comparison.csv",
        "source_price_bands: outputs/q2_v2/stage4/q2_v2_emergency_by_price_band.csv",
        f"dynamic_plan_cost_yuan: {plan_yuan[0]:.12f}",
        f"dynamic_emergency_cost_yuan: {emergency_yuan[0]:.12f}",
        f"dynamic_total_cost_yuan: {total_yuan[0]:.12f}",
        f"greedy_plan_cost_yuan: {plan_yuan[1]:.12f}",
        f"greedy_emergency_cost_yuan: {emergency_yuan[1]:.12f}",
        f"greedy_total_cost_yuan: {total_yuan[1]:.12f}",
        f"annual_saving_yuan: {annual_saving_yuan:.12f}",
        f"billing_identity_max_error_yuan: {identity_error.max():.12g}",
        f"monthly_sum_error_yuan: {monthly_sum_error:.12g}",
        f"cumulative_endpoint_error_yuan: {abs(cumulative_10k[-1] * 1e4 - annual_saving_yuan):.12g}",
        f"price_band_energy_max_error_kwh: {band_errors.max():.12g}",
        "display_name: 动态价值调度",
        "optimization_rerun: false",
        "formal_result_files_modified: false",
        "passed: true",
    ]
    (ROOT / "outputs" / "q2_v2" / "q2_v2_annual_figure_data_check.txt").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )


def band_spans(axis: plt.Axes, price: np.ndarray):
    categories = np.select([price < 0.6, price < 1.0], [0, 1], default=2)
    colors = ["#EAF4ED", "#FFF4E0", "#FDEBEC"]
    labels = ["低价", "中价", "高价"]
    start = 0
    for i in range(1, len(categories) + 1):
        if i == len(categories) or categories[i] != categories[start]:
            axis.axvspan(start * DT, i * DT, color=colors[categories[start]], alpha=0.62, linewidth=0, zorder=0)
            start = i
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.62) for c in colors]
    return handles, labels


def day_soc(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = frame.sort_values("slot")
    soc = np.concatenate(([frame.iloc[0]["soc_start_kwh"]], frame["soc_end_kwh"].to_numpy(float)))
    return np.arange(145) * DT, soc


def figure_day(slots: pd.DataFrame) -> None:
    case = slots.loc[slots["date"].eq(CASE_DATE)].copy()
    dp = case.loc[case["policy_id"].eq("main_dp50")].sort_values("slot")
    greedy = case.loc[case["policy_id"].eq("closed_loop_greedy")].sort_values("slot")
    if len(dp) != 144 or len(greedy) != 144:
        raise ValueError(f"Expected 144 frozen slots for both strategies on {CASE_DATE}.")
    time = dp["slot"].to_numpy(float) * DT
    net = dp["load_actual_kwh"].to_numpy(float) - dp["pv_actual_kwh"].to_numpy(float)
    fig, axes = plt.subplots(3, 1, figsize=(7.15, 6.25), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.step(time, net, where="post", color=ACTUAL_BLUE, linewidth=1.7, label="实际净负荷")
    ax.step(time, dp["planned_purchase_kwh"], where="post", color=PLAN_BLUE, linewidth=1.55, label="计划购电量")
    width = DT * 0.38
    ax.bar(time - width / 2, dp["emergency_purchase_kwh"], width=width, color=EMERGENCY_RED, alpha=0.9, label="动态价值紧急购电")
    ax.bar(time + width / 2, greedy["emergency_purchase_kwh"], width=width, color=BOUND_GREY, alpha=0.78, label="贪心紧急购电")
    ax.set_ylabel("每槽电量（kWh）")
    panel_title(ax, "(a)", "实际净负荷、固定计划与紧急购电")
    style_axis(ax, time_axis=True)
    compact_legend(ax, loc="upper right", ncol=2, fontsize=7.1)

    ax = axes[1]
    band_handles, band_labels = band_spans(ax, dp["price_yuan_per_kwh"].to_numpy(float))
    charge_kw = dp["charge_kwh"].to_numpy(float) / DT
    discharge_kw = dp["discharge_kwh"].to_numpy(float) / DT
    ax.step(time, charge_kw, where="post", color=CHARGE_GREEN, linewidth=1.6, label="充电功率")
    ax.step(time, -discharge_kw, where="post", color=DISCHARGE_ORANGE, linewidth=1.6, label="放电功率")
    ax.axhline(5000, color=BOUND_GREY, linestyle="--", linewidth=0.85)
    ax.axhline(-5000, color=BOUND_GREY, linestyle="--", linewidth=0.85)
    ax.set_ylabel("功率（kW）")
    panel_title(ax, "(b)", "动态价值策略实际充放电功率")
    style_axis(ax, time_axis=True)
    line_handles, line_labels = ax.get_legend_handles_labels()
    compact_legend(
        ax,
        handles=line_handles + band_handles,
        labels=line_labels + band_labels,
        loc="lower left",
        ncol=5,
        fontsize=6.7,
        handlelength=1.25,
    )

    ax = axes[2]
    td, soc_dp = day_soc(dp)
    tg, soc_g = day_soc(greedy)
    ax.plot(td, soc_dp, color=ACTUAL_BLUE, linewidth=1.75, label="动态价值 SOC")
    ax.plot(tg, soc_g, color=BOUND_GREY, linewidth=1.35, linestyle="--", label="贪心 SOC")
    ax.axhline(1200, color=EMERGENCY_RED, linestyle="--", linewidth=1.0, label="安全下限")
    ax.axhline(10800, color=BOUND_GREY, linestyle="-.", linewidth=1.0, label="安全上限")
    ax.set_ylabel("储电量（kWh）")
    ax.set_xlabel("时间（h）")
    panel_title(ax, "(c)", "两种执行器的 SOC 轨迹")
    style_axis(ax, time_axis=True)
    compact_legend(ax, loc="upper right", ncol=2, fontsize=7.1)
    save(fig, "q2_v2_20250601_dispatch")


def figure_scenarios(slots: pd.DataFrame, plans: pd.DataFrame) -> None:
    case_plan = plans.loc[(plans["policy_id"].eq("main_dp50")) & (plans["date"].eq(CASE_DATE))].sort_values("slot")
    case_actual = slots.loc[(slots["policy_id"].eq("main_dp50")) & (slots["date"].eq(CASE_DATE))].sort_values("slot")
    info = pd.read_csv(STAGE4 / "main_dp50" / "information" / f"{CASE_DATE}.csv")
    dates = info.loc[0, "scenario_dates"].split("|")
    if len(case_plan) != 144 or len(case_actual) != 144 or len(dates) != 14:
        raise ValueError("Frozen June-01 scenario inputs are incomplete.")
    hist_slot = slots.loc[(slots["policy_id"].eq("main_dp50")) & (slots["date"].isin(dates))].copy()
    hist_plan = plans.loc[(plans["policy_id"].eq("main_dp50")) & (plans["date"].isin(dates))].copy()
    merged = hist_slot.merge(hist_plan[["date", "slot", "load_forecast_kwh", "pv_forecast_kwh"]], on=["date", "slot"], validate="one_to_one")
    scenario = []
    pred_net = case_plan["load_forecast_kwh"].to_numpy(float) - case_plan["pv_forecast_kwh"].to_numpy(float)
    for d in dates:
        h = merged.loc[merged["date"].eq(d)].sort_values("slot")
        residual_net = (h["load_actual_kwh"].to_numpy(float) - h["load_forecast_kwh"].to_numpy(float)) - (h["pv_actual_kwh"].to_numpy(float) - h["pv_forecast_kwh"].to_numpy(float))
        scenario.append(pred_net + residual_net)
    scenario_arr = np.asarray(scenario)
    actual_net = case_actual["load_actual_kwh"].to_numpy(float) - case_actual["pv_actual_kwh"].to_numpy(float)
    time = np.arange(144) * DT
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 5.25), sharex=True, constrained_layout=True)

    ax = axes[0]
    for i, path in enumerate(scenario_arr):
        ax.step(time, path, where="post", color=BOUND_GREY, alpha=0.25, linewidth=0.75, label="历史联合残差情景" if i == 0 else None)
    ax.step(time, pred_net, where="post", color=FORECAST_BLUE, linestyle="--", linewidth=1.65, label="日前点预测")
    ax.step(time, actual_net, where="post", color=ACTUAL_BLUE, linewidth=1.75, label="实际净负荷")
    ax.set_ylabel("每槽净负荷（kWh）")
    panel_title(ax, "(a)", "日前点预测与 14 条联合残差情景")
    style_axis(ax, time_axis=True)
    compact_legend(ax, loc="upper right", ncol=3, fontsize=7.1)

    ax = axes[1]
    q10, q25, q50, q75, q90 = np.quantile(scenario_arr, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    ax.fill_between(time, q10, q90, step="post", color=FORECAST_BLUE, alpha=0.13, label="10%–90% 情景带")
    ax.fill_between(time, q25, q75, step="post", color=PV_GREEN, alpha=0.22, label="25%–75% 情景带")
    ax.step(time, q50, where="post", color=BOUND_GREY, linewidth=1.25, label="情景中位数")
    ax.step(time, actual_net, where="post", color=ACTUAL_BLUE, linewidth=1.75, label="实际净负荷")
    ax.set_ylabel("每槽净负荷（kWh）")
    ax.set_xlabel("时间（h）")
    panel_title(ax, "(b)", "历史联合情景的分位带")
    style_axis(ax, time_axis=True)
    compact_legend(ax, loc="upper right", ncol=2, fontsize=7.1)
    ax.text(0.01, 0.04, "14 条情景仅由 2025-06-01 决策日前的历史数据构造", transform=ax.transAxes, fontsize=7.2, color=BOUND_GREY)
    save(fig, "q2_v2_joint_residual_scenarios")


def main() -> None:
    font, _ = configure_chinese_theme()
    slots, plans, daily, monthly, bands = read_inputs()
    figure_annual(monthly, bands)
    figure_day(slots)
    figure_scenarios(slots, plans)
    LOG.write_text(
        "问题二重新设计图表核查\n"
        f"中文字体：{font}\n"
        "数据来源：冻结 stage4 的 monthly_comparison、emergency_by_price_band、slot_results、day_ahead_plans。\n"
        "图1：费用单位为百万元、紧急购电量单位为万 kWh；图2/图3均保持官方144个10分钟槽顺序。\n"
        "图2代表日：2025-06-01；图3情景日期由当日冻结信息集记录的14个历史日期确定。\n"
        "仅对绘图副本完成 kWh→kW 转换；未回写任何冻结数据。\n"
        "输出：q2_v2_annual_economic_mechanism、q2_v2_20250601_dispatch、q2_v2_joint_residual_scenarios（PNG 300 dpi 与 PDF）。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
