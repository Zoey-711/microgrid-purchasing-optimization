"""Generate four Q1/Q2 paper figures strictly from frozen result tables.

This script performs presentation-only unit conversions on copies in memory. It
does not solve a model or modify any source CSV, workbook, or frozen result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from src.paper_figure_theme import (
    ACTUAL_BLUE,
    BORDER_DARK,
    BOUND_GREY,
    CHARGE_GREEN,
    DISCHARGE_ORANGE,
    EMERGENCY_RED,
    FORECAST_BLUE,
    GRID_GREY,
    PLAN_BLUE,
    PRICE_ORANGE,
    PV_GREEN,
    compact_legend,
    configure_chinese_theme,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
Q1_SOURCE = ROOT / "data" / "processed" / "q1_baseline_dispatch.csv"
Q2_PARAMETER_SOURCE = ROOT / "outputs" / "tables" / "q2_january_parameter_comparison.csv"
Q2_REPLAY_SOURCE = ROOT / "outputs" / "replay" / "q2_aplus" / "q2_aplus_slot_trajectory.csv"
OUTPUT_DIR = ROOT / "outputs" / "figures" / "paper_v2"
LOG_FILE = ROOT / "outputs" / "logs" / "q1_q2_figure_check_v2.txt"

DELTA_T = 1.0 / 6.0
N_SLOTS = 144
S_MIN = 1200.0
S_MAX = 10800.0
S_INIT = 6000.0
S_FINAL = 6000.0
P_LIMIT = 5000.0
ETA_C = 0.9
ETA_D = 0.9
CASE_DATE = "2025-09-23"
FIG_WIDTH = 16.0 / 2.54


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_columns(data: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{source} lacks required fields: {missing}")


def slot_step_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Represent slot t over [t*dt,(t+1)*dt] without shifting its official order."""
    values = np.asarray(values, dtype=float)
    if values.size != N_SLOTS:
        raise ValueError(f"Expected {N_SLOTS} slot values, got {values.size}")
    boundaries = np.arange(N_SLOTS + 1, dtype=float) * DELTA_T
    return boundaries, np.r_[values, values[-1]]


def state_series_from_start_and_actions(data: pd.DataFrame) -> np.ndarray:
    starts = data["soc_kwh"].to_numpy(dtype=float)
    terminal = (
        starts[-1]
        + ETA_C * float(data["charge_kwh"].iloc[-1])
        - float(data["discharge_kwh"].iloc[-1]) / ETA_D
    )
    return np.r_[starts, terminal]


def load_and_validate_q1() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    data = pd.read_csv(Q1_SOURCE)
    required = {
        "time_label", "price", "load_kwh", "pv_forecast_kwh", "purchase_kwh",
        "charge_kwh", "discharge_kwh", "soc_kwh",
    }
    require_columns(data, required, Q1_SOURCE)
    if len(data) != N_SLOTS:
        raise ValueError(f"Q1 requires 144 slots, found {len(data)}")
    numeric = data[list(required - {"time_label"})].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Q1 plotting fields contain non-finite values")
    states = state_series_from_start_and_actions(data)
    if states.size != 145:
        raise AssertionError("Q1 state vector is not 145 points")
    if not np.isclose(states[0], S_INIT, atol=1e-6) or not np.isclose(states[-1], S_FINAL, atol=1e-6):
        raise AssertionError("Q1 start/end SOC differs from the frozen 6000 kWh values")
    if states.min() < S_MIN - 1e-6 or states.max() > S_MAX + 1e-6:
        raise AssertionError("Q1 SOC lies outside frozen safety boundaries")
    checks = [
        f"Q1 rows: {len(data)}; official row order retained; first/last labels: {data.time_label.iloc[0]} / {data.time_label.iloc[-1]}",
        "Q1 load_kwh and pv_forecast_kwh divided by Delta_t=1/6 h for display in kW.",
        "Q1 charge_kwh and discharge_kwh divided by Delta_t=1/6 h for display in kW; efficiency was not reapplied.",
        f"Q1 SOC states: {states.size}; start/end: {states[0]:.10f}/{states[-1]:.10f} kWh; range: {states.min():.10f}--{states.max():.10f} kWh.",
    ]
    return data, states, checks


def load_and_validate_q2_case() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    all_data = pd.read_csv(Q2_REPLAY_SOURCE)
    required = {
        "date", "slot", "time_label", "net_load_pred_kwh", "net_load_actual_kwh",
        "planned_purchase_kwh", "charge_rt_kwh", "discharge_rt_kwh",
        "emergency_purchase_kwh", "curtailment_kwh", "soc_start_kwh", "soc_end_kwh",
    }
    require_columns(all_data, required, Q2_REPLAY_SOURCE)
    case = all_data.loc[all_data["date"].astype(str) == CASE_DATE].copy()
    case["slot"] = pd.to_numeric(case["slot"], errors="raise").astype(int)
    case = case.sort_values("slot", kind="stable").reset_index(drop=True)
    if len(case) != N_SLOTS or not np.array_equal(case["slot"].to_numpy(), np.arange(N_SLOTS)):
        raise ValueError(f"Q2 case {CASE_DATE} does not contain exact slots 0..143")
    if case["time_label"].duplicated().any():
        raise ValueError("Q2 case contains duplicate official time labels")
    numeric_fields = sorted(required - {"date", "time_label"})
    numeric = case[numeric_fields].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Q2 case plotting fields contain non-finite values")
    start = case["soc_start_kwh"].to_numpy(dtype=float)
    end = case["soc_end_kwh"].to_numpy(dtype=float)
    continuity_error = float(np.max(np.abs(start[1:] - end[:-1])))
    if continuity_error > 1e-6:
        raise AssertionError(f"Q2 SOC is not continuous: {continuity_error}")
    states = np.r_[start, end[-1]]
    if states.min() < S_MIN - 1e-6 or states.max() > S_MAX + 1e-6:
        raise AssertionError("Q2 case SOC lies outside safety boundaries")
    checks = [
        f"Q2 replay rows: {len(all_data)}; case {CASE_DATE} rows: {len(case)}; slots exactly 0..143.",
        f"Q2 case first/last official labels: {case.time_label.iloc[0]} / {case.time_label.iloc[-1]}.",
        "Q2 net load, planned purchase, and emergency purchase retained as per-slot kWh.",
        "Q2 charge_rt_kwh and discharge_rt_kwh divided by Delta_t=1/6 h for display in kW; efficiency was not reapplied.",
        f"Q2 actual SOC states: {states.size}; continuity max error: {continuity_error:.12g} kWh; start/end: {states[0]:.10f}/{states[-1]:.10f} kWh.",
    ]
    return case, states, checks


def load_and_validate_parameters() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data = pd.read_csv(Q2_PARAMETER_SOURCE)
    required = {
        "w_days", "q", "Sres_kwh", "rho_yuan_per_kwh", "total_cost_yuan",
        "emergency_purchase_kwh", "validation_passed", "backtest_start", "backtest_end",
    }
    require_columns(data, required, Q2_PARAMETER_SOURCE)
    numeric_fields = ["w_days", "q", "Sres_kwh", "rho_yuan_per_kwh", "total_cost_yuan", "emergency_purchase_kwh"]
    data[numeric_fields] = data[numeric_fields].apply(pd.to_numeric, errors="raise")
    if len(data) != 135 or not data["validation_passed"].astype(str).str.lower().eq("true").all():
        raise ValueError("Expected 135 physically valid January parameter records")
    if set(data["backtest_start"].astype(str)) != {"2025-01-22"} or set(data["backtest_end"].astype(str)) != {"2025-01-31"}:
        raise ValueError("Unexpected parameter backtest period")

    keys = ["w_days", "q", "Sres_kwh"]
    spread = data.groupby(keys)[["total_cost_yuan", "emergency_purchase_kwh"]].agg(np.ptp)
    max_duplicate_spread = float(spread.to_numpy().max())
    if max_duplicate_spread > 1e-6:
        raise AssertionError("rho variants are not numerically identical and cannot be merged")
    display = (
        data.sort_values("rho_yuan_per_kwh")
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )
    if len(display) != 45:
        raise AssertionError(f"Expected 45 unique (w,q,Sres) records, found {len(display)}")
    checks = [
        "Parameter backtest: 2025-01-22 through 2025-01-31; 135 valid records.",
        "rho values: 0.6976, 1.3952, 6.9760 yuan/kWh; all duplicate (w,q,Sres) outcomes are identical within 1e-6.",
        f"Collapsed to {len(display)} unique (w,q,Sres) points; displayed representative rho=0.6976 yuan/kWh.",
        f"Maximum duplicate metric spread: {max_duplicate_spread:.12g}.",
    ]
    return data, display, checks


def save_figure(figure, stem: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def plot_q1_supply(data: pd.DataFrame) -> tuple[Path, Path]:
    x, load_kw = slot_step_xy(data["load_kwh"].to_numpy(dtype=float) / DELTA_T)
    _, pv_kw = slot_step_xy(data["pv_forecast_kwh"].to_numpy(dtype=float) / DELTA_T)
    _, price = slot_step_xy(data["price"].to_numpy(dtype=float))
    _, purchase = slot_step_xy(data["purchase_kwh"].to_numpy(dtype=float))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(FIG_WIDTH, 7.7), constrained_layout=True)
    axes[0].step(x, load_kw, where="post", color=ACTUAL_BLUE, lw=1.7, label="小区负载")
    axes[0].step(x, pv_kw, where="post", color=PV_GREEN, lw=1.7, label="光伏预测功率")
    axes[0].set_ylabel("功率（kW）")
    panel_title(axes[0], "(a)", "小区负载与光伏功率", y=1.15)
    compact_legend(axes[0], loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2)

    axes[1].step(x, price, where="post", color=PRICE_ORANGE, lw=1.7, label="分时电价")
    axes[1].set_ylabel("电价（元/kWh）")
    panel_title(axes[1], "(b)", "分时电价", y=1.15)
    compact_legend(axes[1], loc="lower left", bbox_to_anchor=(0.0, 1.005))

    axes[2].step(x, purchase, where="post", color=PLAN_BLUE, lw=1.7, label="计划购电量")
    axes[2].fill_between(x, purchase, 0, step="post", color=PLAN_BLUE, alpha=0.10)
    axes[2].set_ylabel("每槽购电量（kWh）")
    axes[2].set_xlabel("时间（h）")
    panel_title(axes[2], "(c)", "每槽计划购电量（每槽10分钟）", y=1.15)
    compact_legend(axes[2], loc="lower left", bbox_to_anchor=(0.0, 1.005))

    for axis in axes:
        style_axis(axis, time_axis=True)
    return save_figure(fig, "q1_supply_price_purchase")


def plot_q1_storage(data: pd.DataFrame, states: np.ndarray) -> tuple[Path, Path]:
    x, charge_kw = slot_step_xy(data["charge_kwh"].to_numpy(dtype=float) / DELTA_T)
    _, discharge_kw = slot_step_xy(-data["discharge_kwh"].to_numpy(dtype=float) / DELTA_T)
    state_x = np.arange(145) * DELTA_T

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(FIG_WIDTH, 6.5), constrained_layout=True)
    axes[0].step(x, charge_kw, where="post", color=CHARGE_GREEN, lw=1.65, label="充电功率")
    axes[0].step(x, discharge_kw, where="post", color=DISCHARGE_ORANGE, lw=1.65, label="放电功率（负值）")
    axes[0].axhline(P_LIMIT, color=BOUND_GREY, ls="--", lw=0.9, label="充电功率上限（+5000 kW）")
    axes[0].axhline(-P_LIMIT, color=BOUND_GREY, ls=":", lw=0.9, label="放电功率下限（−5000 kW）")
    axes[0].axhline(0, color=BOUND_GREY, lw=0.75)
    axes[0].set_ylabel("充放电功率（kW）")
    axes[0].set_ylim(-5500, 5500)
    axes[0].set_yticks([-5000, -2500, 0, 2500, 5000])
    panel_title(axes[0], "(a)", "储能充放电功率（充电为正，放电为负）", y=1.19)
    compact_legend(axes[0], loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2)

    axes[1].plot(state_x, states, color=ACTUAL_BLUE, lw=1.75, label="储电量")
    axes[1].fill_between(state_x, states, S_MIN, color=ACTUAL_BLUE, alpha=0.08)
    axes[1].axhline(S_MIN, color=BOUND_GREY, ls="--", lw=0.95, label="安全下限（1200 kWh）")
    axes[1].axhline(S_MAX, color=BOUND_GREY, ls=":", lw=0.95, label="安全上限（10800 kWh）")
    axes[1].scatter([0, 24], [states[0], states[-1]], color=ACTUAL_BLUE, s=18, zorder=5)
    axes[1].annotate("0:00，6000 kWh", xy=(0, states[0]), xytext=(0.7, 6750), color=ACTUAL_BLUE,
                     arrowprops={"arrowstyle": "-", "color": ACTUAL_BLUE, "lw": 0.8}, fontsize=8.0)
    axes[1].annotate("24:00，6000 kWh", xy=(24, states[-1]), xytext=(23.2, 6750), ha="right", color=ACTUAL_BLUE,
                     arrowprops={"arrowstyle": "-", "color": ACTUAL_BLUE, "lw": 0.8}, fontsize=8.0)
    axes[1].set_ylabel("储电量（kWh）")
    axes[1].set_xlabel("时间（h）")
    axes[1].set_ylim(700, 11300)
    panel_title(axes[1], "(b)", "储电量及安全边界", y=1.15)
    compact_legend(axes[1], loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3)

    for axis in axes:
        style_axis(axis, time_axis=True)
    return save_figure(fig, "q1_storage_operation")


def plot_q2_tradeoff_appendix(display: pd.DataFrame) -> tuple[Path, Path]:
    line_colors = {7: ACTUAL_BLUE, 14: PV_GREEN, 21: PRICE_ORANGE}
    markers = {7: "o", 14: "s", 21: "^"}
    reserve_colors = {1200: ACTUAL_BLUE, 3600: PV_GREEN, 6000: PRICE_ORANGE}
    base = display[(display.w_days == 7) & np.isclose(display.q, 0.75) & np.isclose(display.Sres_kwh, 1200)].iloc[0]
    reserve = display[(display.w_days == 7) & np.isclose(display.q, 0.75) & np.isclose(display.Sres_kwh, 3600)].iloc[0]

    # The lower joint plot is deliberately centered and compact.  Its main
    # plotting box is physically square, but x/y numerical scales remain
    # independent because their units differ.
    fig = plt.figure(figsize=(FIG_WIDTH, 8.25), facecolor="white")
    grid = fig.add_gridspec(
        3, 2, height_ratios=[0.92, 1.38, 0.16],
        left=0.10, right=0.95, bottom=0.055, top=0.91,
        hspace=0.42, wspace=0.32,
    )
    ax_cost = fig.add_subplot(grid[0, 0])
    ax_emergency = fig.add_subplot(grid[0, 1])
    centered = grid[1, :].subgridspec(1, 3, width_ratios=[0.15, 1.0, 0.15])
    joint = centered[0, 1].subgridspec(
        2, 2, height_ratios=[0.18, 1.0], width_ratios=[1.0, 0.18],
        hspace=0.03, wspace=0.03,
    )
    ax_trade = fig.add_subplot(joint[1, 0])
    ax_trade.set_box_aspect(1)
    ax_marginal_x = fig.add_subplot(joint[0, 0], sharex=ax_trade)
    ax_marginal_y = fig.add_subplot(joint[1, 1], sharey=ax_trade)
    ax_legend = fig.add_subplot(grid[2, :])
    ax_legend.axis("off")

    baseline = display[np.isclose(display.Sres_kwh, 1200)]
    for w in [7, 14, 21]:
        part = baseline[baseline.w_days == w].sort_values("q")
        ax_cost.plot(part.q, part.total_cost_yuan / 10000.0, color=line_colors[w], marker=markers[w],
                     ms=4.2, lw=1.55, label=f"窗口 w={w} 天")
        ax_emergency.plot(part.q, part.emergency_purchase_kwh, color=line_colors[w], marker=markers[w],
                          ms=4.2, lw=1.55, label=f"窗口 w={w} 天")
    ax_cost.scatter([base.q], [base.total_cost_yuan / 10000.0], s=70, facecolors="none",
                    edgecolors=EMERGENCY_RED, lw=1.2, zorder=5)
    ax_emergency.scatter([base.q], [base.emergency_purchase_kwh], s=70, facecolors="none",
                         edgecolors=EMERGENCY_RED, lw=1.2, zorder=5)
    ax_cost.annotate("w=7，q=0.75", (base.q, base.total_cost_yuan / 10000.0), xytext=(-43, 12),
                     textcoords="offset points", fontsize=7.7, color=EMERGENCY_RED,
                     arrowprops={"arrowstyle": "-", "color": EMERGENCY_RED, "lw": 0.8})
    ax_emergency.annotate("w=7，q=0.75", (base.q, base.emergency_purchase_kwh), xytext=(-47, 12),
                          textcoords="offset points", fontsize=7.7, color=EMERGENCY_RED,
                          arrowprops={"arrowstyle": "-", "color": EMERGENCY_RED, "lw": 0.8})
    ax_cost.set_xlabel("分位数 q")
    ax_cost.set_ylabel("实际总成本（万元）")
    ax_emergency.set_xlabel("分位数 q")
    ax_emergency.set_ylabel("紧急购电量（kWh）")
    panel_title(ax_cost, "(a)", "总成本")
    panel_title(ax_emergency, "(b)", "紧急购电量")

    window_handles = [
        Line2D([0], [0], color=line_colors[w], marker=markers[w], lw=1.55,
               markersize=4.5, label=f"窗口 w={w} 天")
        for w in [7, 14, 21]
    ]
    fig.legend(
        handles=window_handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
        ncol=3, frameon=False, handlelength=2.0, handletextpad=0.45,
        columnspacing=1.2,
    )

    for sres in [1200, 3600, 6000]:
        for w in [7, 14, 21]:
            part = display[np.isclose(display.Sres_kwh, sres) & (display.w_days == w)]
            ax_trade.scatter(part.emergency_purchase_kwh, part.total_cost_yuan / 10000.0,
                             color=reserve_colors[sres], marker=markers[w], s=31, alpha=0.9,
                             edgecolors="white", linewidths=0.35)

    # Marginal histograms show counts of the same 45 completed schemes.  All
    # groups use identical equal-width bin edges; no KDE or fitted distribution
    # is added. Seven bins balance the 15 records available per reserve target.
    x_min = float(display.emergency_purchase_kwh.min())
    x_max = float(display.emergency_purchase_kwh.max())
    y_cost = display.total_cost_yuan / 10000.0
    y_min = float(y_cost.min())
    y_max = float(y_cost.max())
    x_bins = np.linspace(x_min, x_max, 8)
    y_bins = np.linspace(y_min, y_max, 8)
    for sres in [1200, 3600, 6000]:
        group = display[np.isclose(display.Sres_kwh, sres)]
        ax_marginal_x.hist(
            group.emergency_purchase_kwh, bins=x_bins,
            color=reserve_colors[sres], alpha=0.28,
            histtype="stepfilled", linewidth=0.0,
        )
        ax_marginal_x.hist(
            group.emergency_purchase_kwh, bins=x_bins,
            color=reserve_colors[sres], histtype="step", linewidth=0.85,
        )
        ax_marginal_y.hist(
            group.total_cost_yuan / 10000.0, bins=y_bins,
            orientation="horizontal", color=reserve_colors[sres],
            alpha=0.28, histtype="stepfilled", linewidth=0.0,
        )
        ax_marginal_y.hist(
            group.total_cost_yuan / 10000.0, bins=y_bins,
            orientation="horizontal", color=reserve_colors[sres],
            histtype="step", linewidth=0.85,
        )
    ax_trade.scatter([base.emergency_purchase_kwh], [base.total_cost_yuan / 10000.0], s=105,
                     facecolors="none", edgecolors=EMERGENCY_RED, lw=1.35, zorder=6)
    ax_trade.scatter([reserve.emergency_purchase_kwh], [reserve.total_cost_yuan / 10000.0], s=105,
                     facecolors="none", edgecolors=BOUND_GREY, lw=1.35, zorder=6)
    ax_trade.annotate("B", (base.emergency_purchase_kwh, base.total_cost_yuan / 10000.0),
                      xytext=(14, 11), textcoords="offset points", fontsize=8.0,
                      fontweight="bold", color=EMERGENCY_RED,
                      arrowprops={"arrowstyle": "-", "color": EMERGENCY_RED, "lw": 0.8},
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6})
    ax_trade.annotate("R", (reserve.emergency_purchase_kwh, reserve.total_cost_yuan / 10000.0),
                      xytext=(-18, 13), textcoords="offset points", fontsize=8.0,
                      fontweight="bold", color=BOUND_GREY,
                      arrowprops={"arrowstyle": "-", "color": BOUND_GREY, "lw": 0.8},
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6})
    ax_trade.set_xlabel("紧急购电量（kWh）")
    ax_trade.set_ylabel("实际总成本（万元）")
    panel_title(ax_marginal_x, "(c)", "成本与紧急购电权衡", y=1.45)

    reserve_handles = [Line2D([0], [0], marker="o", linestyle="none", color=reserve_colors[s],
                              label=f"$S^{{res}}={s}$ kWh", markersize=5.5) for s in [1200, 3600, 6000]]
    marker_handles = [Line2D([0], [0], marker=markers[w], linestyle="none", color=BOUND_GREY,
                             label=f"w={w} 天", markersize=5.5) for w in [7, 14, 21]]
    reserve_legend = ax_legend.legend(
        handles=reserve_handles, loc="center", bbox_to_anchor=(0.5, 0.72),
        frameon=False, ncol=3, handletextpad=0.35,
        columnspacing=0.75, borderaxespad=0.0,
    )
    ax_legend.add_artist(reserve_legend)
    ax_legend.legend(
        handles=marker_handles, loc="center", bbox_to_anchor=(0.5, 0.18),
        frameon=False, ncol=3, handletextpad=0.35,
        columnspacing=0.75, borderaxespad=0.0,
    )

    for axis in [ax_cost, ax_emergency, ax_trade]:
        style_axis(axis)
    for axis in [ax_marginal_x, ax_marginal_y]:
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    ax_marginal_x.tick_params(axis="both", which="both", left=False, bottom=False,
                              labelleft=False, labelbottom=False)
    ax_marginal_y.tick_params(axis="both", which="both", left=False, bottom=False,
                              labelleft=False, labelbottom=False)
    x_pad = max((x_max - x_min) * 0.045, 1.0)
    y_pad = max((y_max - y_min) * 0.055, 0.01)
    ax_trade.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_trade.set_ylim(y_min - y_pad, y_max + y_pad)
    return save_figure(fig, "q2_parameter_tradeoff_compact")


def plot_q2_parameter_selection_backup(display: pd.DataFrame) -> tuple[Path, Path]:
    """Draw the paper-facing Q2 parameter-selection figure.

    Panels (a) and (b) retain the Sres=1200 kWh sensitivity curves. Panel
    (c) deliberately reduces the encoding to the three tested reserve targets
    at the adopted prediction settings. All values come from ``display``;
    coordinates are neither jittered nor transformed apart from yuan to 10,000
    yuan for the cost axes.
    """
    line_colors = {7: ACTUAL_BLUE, 14: PV_GREEN, 21: PRICE_ORANGE}
    markers = {7: "o", 14: "s", 21: "^"}
    formal_rho = 0.6976

    baseline = display[np.isclose(display.Sres_kwh, 1200)].copy()
    focus = display[
        (display.w_days == 7)
        & np.isclose(display.q, 0.75)
        & np.isclose(display.rho_yuan_per_kwh, formal_rho)
        & display.Sres_kwh.isin([1200, 3600, 6000])
    ].sort_values("Sres_kwh")
    if len(focus) != 3 or set(focus.Sres_kwh.astype(int)) != {1200, 3600, 6000}:
        found = focus[["w_days", "q", "Sres_kwh", "rho_yuan_per_kwh"]].to_dict("records")
        raise ValueError(f"Missing fixed-parameter reserve records; found: {found}")

    base = focus[np.isclose(focus.Sres_kwh, 1200)].iloc[0]
    reserve = focus[np.isclose(focus.Sres_kwh, 3600)].iloc[0]
    delta_cost = float(reserve.total_cost_yuan - base.total_cost_yuan)
    delta_emergency = float(reserve.emergency_purchase_kwh - base.emergency_purchase_kwh)

    fig = plt.figure(figsize=(FIG_WIDTH, 6.9), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, height_ratios=[0.95, 1.10],
        left=0.10, right=0.96, bottom=0.09, top=0.90,
        hspace=0.48, wspace=0.31,
    )
    ax_cost = fig.add_subplot(grid[0, 0])
    ax_emergency = fig.add_subplot(grid[0, 1])
    lower = grid[1, :].subgridspec(1, 3, width_ratios=[0.07, 0.86, 0.07])
    ax_trade = fig.add_subplot(lower[0, 1])
    ax_trade.set_box_aspect(0.58)

    for w in [7, 14, 21]:
        part = baseline[baseline.w_days == w].sort_values("q")
        ax_cost.plot(
            part.q, part.total_cost_yuan / 10000.0,
            color=line_colors[w], marker=markers[w], ms=4.2, lw=1.55,
            label=f"窗口 w={w} 天",
        )
        ax_emergency.plot(
            part.q, part.emergency_purchase_kwh,
            color=line_colors[w], marker=markers[w], ms=4.2, lw=1.55,
            label=f"窗口 w={w} 天",
        )

    for axis, y in [
        (ax_cost, float(base.total_cost_yuan) / 10000.0),
        (ax_emergency, float(base.emergency_purchase_kwh)),
    ]:
        axis.scatter(
            [base.q], [y], s=66, facecolors="none",
            edgecolors=EMERGENCY_RED, linewidths=1.15, zorder=5,
        )
        axis.annotate(
            "w=7，q=0.75", (base.q, y), xytext=(-43, 11),
            textcoords="offset points", fontsize=7.6, color=EMERGENCY_RED,
            arrowprops={"arrowstyle": "-", "color": EMERGENCY_RED, "lw": 0.75},
        )

    ax_cost.set_xlabel("分位数 q")
    ax_cost.set_ylabel("实际总成本（万元）")
    ax_emergency.set_xlabel("分位数 q")
    ax_emergency.set_ylabel("紧急购电量（kWh）")
    panel_title(ax_cost, "(a)", "总成本")
    panel_title(ax_emergency, "(b)", "紧急购电量")

    window_handles = [
        Line2D(
            [0], [0], color=line_colors[w], marker=markers[w], lw=1.55,
            markersize=4.5, label=f"窗口 w={w} 天",
        )
        for w in [7, 14, 21]
    ]
    fig.legend(
        handles=window_handles, loc="upper center", bbox_to_anchor=(0.5, 0.982),
        ncol=3, frameon=False, handlelength=2.0, handletextpad=0.45,
        columnspacing=1.2,
    )

    comparison_grey = "#8A9199"
    point_colors = {1200: comparison_grey, 3600: PV_GREEN, 6000: comparison_grey}
    labels = {
        1200: "无额外保留：1200 kWh",
        3600: "选定日前保留：3600 kWh",
        6000: "较高保留目标：6000 kWh",
    }
    offsets = {1200: (10, -18), 3600: (10, 10), 6000: (-10, 12)}
    alignments = {1200: "left", 3600: "left", 6000: "right"}
    for _, row in focus.iterrows():
        sres = int(round(float(row.Sres_kwh)))
        x = float(row.emergency_purchase_kwh)
        y = float(row.total_cost_yuan) / 10000.0
        ax_trade.scatter(
            x, y, s=62 if sres == 3600 else 50,
            color=point_colors[sres], edgecolors="white", linewidths=0.6,
            alpha=0.98, zorder=4,
        )
        ax_trade.annotate(
            labels[sres], (x, y), xytext=offsets[sres],
            textcoords="offset points", ha=alignments[sres], va="center",
            fontsize=7.7, color=point_colors[sres],
            arrowprops={"arrowstyle": "-", "color": point_colors[sres], "lw": 0.75},
        )

    delta_emergency_text = f"{delta_emergency:+.3f} kWh"
    delta_text = (
        "3600 kWh 相对 1200 kWh：\n"
        f"成本 {delta_cost:+.3f} 元；紧急购电量 {delta_emergency_text}"
    )
    ax_trade.text(
        0.98, 0.58, delta_text, transform=ax_trade.transAxes,
        ha="right", va="center", fontsize=7.5, color=BOUND_GREY,
        bbox={"facecolor": "white", "edgecolor": GRID_GREY, "linewidth": 0.7, "pad": 2.5},
    )
    ax_trade.text(
        0.98, 0.04, "注：成本轴采用非零起点", transform=ax_trade.transAxes,
        ha="right", va="bottom", fontsize=7.2, color=BOUND_GREY,
    )
    ax_trade.set_xlabel("紧急购电量（kWh）")
    ax_trade.set_ylabel("实际总成本（万元）")
    panel_title(ax_trade, "(c)", "成本与紧急购电权衡")

    # Tight limits improve comparison without altering any point coordinate.
    xs = focus.emergency_purchase_kwh.to_numpy(dtype=float)
    ys = focus.total_cost_yuan.to_numpy(dtype=float) / 10000.0
    x_span = max(float(np.ptp(xs)), 1.0)
    y_span = max(float(np.ptp(ys)), 0.01)
    ax_trade.set_xlim(float(xs.min()) - 0.20 * x_span, float(xs.max()) + 0.20 * x_span)
    ax_trade.set_ylim(float(ys.min()) - 0.23 * y_span, float(ys.max()) + 0.26 * y_span)

    for axis in [ax_cost, ax_emergency, ax_trade]:
        style_axis(axis)
    for spine in ax_trade.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER_DARK)
        spine.set_linewidth(0.75)

    return save_figure(fig, "q2_parameter_selection_compare")


def plot_q2_sensitivity(display: pd.DataFrame) -> tuple[Path, Path]:
    """Plot w/q sensitivity under the no-extra-reserve condition."""
    line_colors = {7: ACTUAL_BLUE, 14: PV_GREEN, 21: PRICE_ORANGE}
    markers = {7: "o", 14: "s", 21: "^"}
    baseline = display[np.isclose(display.Sres_kwh, 1200)].copy()
    selected = baseline[(baseline.w_days == 7) & np.isclose(baseline.q, 0.75)]
    if len(selected) != 1:
        raise ValueError(f"Expected one w=7, q=0.75, Sres=1200 record; found {len(selected)}")
    selected = selected.iloc[0]

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, 3.55),
        gridspec_kw={"left": 0.10, "right": 0.97, "bottom": 0.19, "top": 0.82, "wspace": 0.32},
    )
    for w in [7, 14, 21]:
        part = baseline[baseline.w_days == w].sort_values("q")
        axes[0].plot(
            part.q, part.total_cost_yuan / 10000.0,
            color=line_colors[w], marker=markers[w], ms=4.2, lw=1.55,
            label=f"窗口 w={w} 天",
        )
        axes[1].plot(
            part.q, part.emergency_purchase_kwh,
            color=line_colors[w], marker=markers[w], ms=4.2, lw=1.55,
            label=f"窗口 w={w} 天",
        )
    for axis, y in [
        (axes[0], float(selected.total_cost_yuan) / 10000.0),
        (axes[1], float(selected.emergency_purchase_kwh)),
    ]:
        axis.scatter(
            [selected.q], [y], s=66, facecolors="none",
            edgecolors=EMERGENCY_RED, linewidths=1.15, zorder=5,
        )
        axis.annotate(
            "w=7，q=0.75", (selected.q, y), xytext=(-43, 11),
            textcoords="offset points", fontsize=7.6, color=EMERGENCY_RED,
            arrowprops={"arrowstyle": "-", "color": EMERGENCY_RED, "lw": 0.75},
        )
    axes[0].set_xlabel("分位数 q")
    axes[0].set_ylabel("实际总成本（万元）")
    axes[1].set_xlabel("分位数 q")
    axes[1].set_ylabel("紧急购电量（kWh）")
    panel_title(axes[0], "(a)", "总成本")
    panel_title(axes[1], "(b)", "紧急购电量")
    handles = [
        Line2D([0], [0], color=line_colors[w], marker=markers[w], lw=1.55,
               markersize=4.5, label=f"窗口 w={w} 天")
        for w in [7, 14, 21]
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
        ncol=3, frameon=False, handlelength=2.0, handletextpad=0.45,
        columnspacing=1.2,
    )
    for axis in axes:
        style_axis(axis)
    return save_figure(fig, "q2_prediction_parameter_sensitivity")


def plot_q2_tradeoff_rug(display: pd.DataFrame) -> tuple[Path, Path]:
    """Plot all 45 completed schemes with grouped marginal rug marks."""
    reserve_colors = {1200: ACTUAL_BLUE, 3600: PV_GREEN, 6000: PRICE_ORANGE}
    markers = {7: "o", 14: "s", 21: "^"}
    if len(display) != 45:
        raise ValueError(f"Expected 45 rho-collapsed schemes; found {len(display)}")
    base = display[
        (display.w_days == 7) & np.isclose(display.q, 0.75)
        & np.isclose(display.Sres_kwh, 1200)
    ]
    reserve = display[
        (display.w_days == 7) & np.isclose(display.q, 0.75)
        & np.isclose(display.Sres_kwh, 3600)
    ]
    if len(base) != 1 or len(reserve) != 1:
        raise ValueError("Baseline or selected-reserve record is missing")
    base, reserve = base.iloc[0], reserve.iloc[0]

    fig = plt.figure(figsize=(FIG_WIDTH, 7.15), facecolor="white")
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.19],
        left=0.11, right=0.90, bottom=0.035, top=0.94, hspace=0.16,
    )
    joint = outer[0].subgridspec(
        2, 2, height_ratios=[0.115, 1.0], width_ratios=[1.0, 0.115],
        hspace=0.025, wspace=0.025,
    )
    ax = fig.add_subplot(joint[1, 0])
    ax.set_box_aspect(1)
    ax_top = fig.add_subplot(joint[0, 0], sharex=ax)
    ax_right = fig.add_subplot(joint[1, 1], sharey=ax)
    ax_legend = fig.add_subplot(outer[1])
    ax_legend.axis("off")

    for sres in [1200, 3600, 6000]:
        for w in [7, 14, 21]:
            part = display[np.isclose(display.Sres_kwh, sres) & (display.w_days == w)]
            ax.scatter(
                part.emergency_purchase_kwh,
                part.total_cost_yuan / 10000.0,
                color=reserve_colors[sres], marker=markers[w], s=34,
                alpha=0.86, edgecolors="white", linewidths=0.4, zorder=3,
            )

    # Three separate color lanes preserve every real coordinate while making
    # the marginal locations legible. These are rug marks, not histograms or
    # probability distributions.
    for lane, sres in enumerate([1200, 3600, 6000]):
        part = display[np.isclose(display.Sres_kwh, sres)]
        ax_top.vlines(
            part.emergency_purchase_kwh, lane + 0.18, lane + 0.82,
            color=reserve_colors[sres], linewidth=0.8, alpha=0.72,
        )
        ax_right.hlines(
            part.total_cost_yuan / 10000.0, lane + 0.18, lane + 0.82,
            color=reserve_colors[sres], linewidth=0.8, alpha=0.72,
        )
    ax_top.set_ylim(0, 3)
    ax_right.set_xlim(0, 3)

    highlight_dark = "#4B5563"
    ax.scatter(
        [base.emergency_purchase_kwh], [base.total_cost_yuan / 10000.0],
        s=112, facecolors="none", edgecolors=highlight_dark,
        linewidths=1.25, zorder=6,
    )
    ax.scatter(
        [reserve.emergency_purchase_kwh], [reserve.total_cost_yuan / 10000.0],
        s=126, facecolors="none", edgecolors=PV_GREEN,
        linewidths=1.55, zorder=6,
    )
    ax.annotate(
        "无额外保留基线", (base.emergency_purchase_kwh, base.total_cost_yuan / 10000.0),
        xytext=(15, 11), textcoords="offset points", fontsize=7.5,
        color=highlight_dark,
        arrowprops={"arrowstyle": "-", "color": highlight_dark, "lw": 0.75},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
    )
    ax.annotate(
        "选定日前保留策略", (reserve.emergency_purchase_kwh, reserve.total_cost_yuan / 10000.0),
        xytext=(-18, 18), textcoords="offset points", ha="right", fontsize=7.5,
        color=PV_GREEN,
        arrowprops={"arrowstyle": "-", "color": PV_GREEN, "lw": 0.75},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
    )
    ax.set_xlabel("紧急购电量（kWh）")
    ax.set_ylabel("实际总成本（万元）")
    ax_top.set_title("成本与紧急购电权衡", fontsize=10.0, fontweight="bold", pad=5)

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=reserve_colors[sres],
               markersize=5.5, label=f"保留目标 {sres} kWh")
        for sres in [1200, 3600, 6000]
    ]
    shape_handles = [
        Line2D([0], [0], marker=markers[w], linestyle="none", color=BOUND_GREY,
               markersize=5.5, label=f"历史窗口 {w} 天")
        for w in [7, 14, 21]
    ]
    legend1 = ax_legend.legend(
        handles=color_handles, loc="center", bbox_to_anchor=(0.5, 0.70),
        ncol=3, frameon=False, handletextpad=0.35, columnspacing=0.8,
    )
    ax_legend.add_artist(legend1)
    ax_legend.legend(
        handles=shape_handles, loc="center", bbox_to_anchor=(0.5, 0.12),
        ncol=3, frameon=False, handletextpad=0.35, columnspacing=0.8,
    )

    style_axis(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER_DARK)
        spine.set_linewidth(0.75)
    for marginal in [ax_top, ax_right]:
        marginal.grid(False)
        marginal.set_facecolor("white")
        marginal.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in marginal.spines.values():
            spine.set_visible(False)
    return save_figure(fig, "q2_parameter_tradeoff_rug")


def plot_q2_tradeoff(display: pd.DataFrame) -> tuple[Path, Path]:
    """Reproduce the selected three-panel parameter figure with thin borders."""
    line_colors = {7: ACTUAL_BLUE, 14: PV_GREEN, 21: PRICE_ORANGE}
    markers = {7: "o", 14: "s", 21: "^"}
    reserve_colors = {1200: ACTUAL_BLUE, 3600: PV_GREEN, 6000: PRICE_ORANGE}
    baseline = display[np.isclose(display.Sres_kwh, 1200)].copy()
    base = display[
        (display.w_days == 7) & np.isclose(display.q, 0.75)
        & np.isclose(display.Sres_kwh, 1200)
    ].iloc[0]
    reserve = display[
        (display.w_days == 7) & np.isclose(display.q, 0.75)
        & np.isclose(display.Sres_kwh, 3600)
    ].iloc[0]

    fig = plt.figure(figsize=(FIG_WIDTH, 7.75), facecolor="white")
    outer = fig.add_gridspec(
        2, 2, height_ratios=[0.82, 1.18],
        left=0.09, right=0.965, bottom=0.075, top=0.965,
        hspace=0.37, wspace=0.25,
    )
    ax_cost = fig.add_subplot(outer[0, 0])
    ax_emergency = fig.add_subplot(outer[0, 1])
    joint = outer[1, :].subgridspec(
        2, 2, height_ratios=[0.19, 1.0], width_ratios=[1.0, 0.14],
        hspace=0.035, wspace=0.035,
    )
    ax_trade = fig.add_subplot(joint[1, 0])
    ax_top = fig.add_subplot(joint[0, 0], sharex=ax_trade)
    ax_right = fig.add_subplot(joint[1, 1], sharey=ax_trade)

    for w in [7, 14, 21]:
        part = baseline[baseline.w_days == w].sort_values("q")
        ax_cost.plot(
            part.q, part.total_cost_yuan / 10000.0,
            color=line_colors[w], marker=markers[w], ms=4.1, lw=1.5,
            label=f"窗口 w={w} 天",
        )
        ax_emergency.plot(
            part.q, part.emergency_purchase_kwh,
            color=line_colors[w], marker=markers[w], ms=4.1, lw=1.5,
            label=f"窗口 w={w} 天",
        )
    for axis, y in [
        (ax_cost, float(base.total_cost_yuan) / 10000.0),
        (ax_emergency, float(base.emergency_purchase_kwh)),
    ]:
        axis.scatter([base.q], [y], s=66, facecolors="none", edgecolors=EMERGENCY_RED,
                     linewidths=1.15, zorder=5)
        axis.annotate(
            "w=7，q=0.75", (base.q, y), xytext=(-43, 10), textcoords="offset points",
            fontsize=7.5, color=EMERGENCY_RED,
            arrowprops={"arrowstyle": "-", "color": EMERGENCY_RED, "lw": 0.75},
        )
    ax_cost.set_xlabel("分位数 q")
    ax_cost.set_ylabel("实际总成本（万元）")
    ax_emergency.set_xlabel("分位数 q")
    ax_emergency.set_ylabel("紧急购电量（kWh）")
    panel_title(ax_cost, "(a)", "分位数与实际总成本（无额外保留）")
    panel_title(ax_emergency, "(b)", "分位数与紧急购电量（无额外保留）")
    compact_legend(ax_cost, loc="upper right")
    compact_legend(ax_emergency, loc="upper right")

    for sres in [1200, 3600, 6000]:
        for w in [7, 14, 21]:
            part = display[np.isclose(display.Sres_kwh, sres) & (display.w_days == w)]
            ax_trade.scatter(
                part.emergency_purchase_kwh, part.total_cost_yuan / 10000.0,
                color=reserve_colors[sres], marker=markers[w], s=30, alpha=0.9,
                edgecolors="white", linewidths=0.35, zorder=3,
            )

    x_min = float(display.emergency_purchase_kwh.min())
    x_max = float(display.emergency_purchase_kwh.max())
    y_values = display.total_cost_yuan.to_numpy(dtype=float) / 10000.0
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_bins = np.linspace(x_min, x_max, 9)
    y_bins = np.linspace(y_min, y_max, 9)
    groups_x = [display[np.isclose(display.Sres_kwh, s)].emergency_purchase_kwh for s in [1200, 3600, 6000]]
    groups_y = [display[np.isclose(display.Sres_kwh, s)].total_cost_yuan / 10000.0 for s in [1200, 3600, 6000]]
    ax_top.hist(groups_x, bins=x_bins, stacked=True,
                color=[reserve_colors[s] for s in [1200, 3600, 6000]], alpha=0.40,
                edgecolor="white", linewidth=0.35)
    ax_right.hist(groups_y, bins=y_bins, stacked=True, orientation="horizontal",
                  color=[reserve_colors[s] for s in [1200, 3600, 6000]], alpha=0.40,
                  edgecolor="white", linewidth=0.35)

    ax_trade.scatter([base.emergency_purchase_kwh], [base.total_cost_yuan / 10000.0],
                     s=105, facecolors="none", edgecolors=EMERGENCY_RED, lw=1.35, zorder=6)
    ax_trade.scatter([reserve.emergency_purchase_kwh], [reserve.total_cost_yuan / 10000.0],
                     s=105, facecolors="none", edgecolors=BOUND_GREY, lw=1.35, zorder=6)
    ax_trade.annotate(
        "无额外保留基线\nw=7，q=0.75",
        (base.emergency_purchase_kwh, base.total_cost_yuan / 10000.0),
        xytext=(24, 18), textcoords="offset points", fontsize=7.3, color=EMERGENCY_RED,
        arrowprops={"arrowstyle": "-", "color": EMERGENCY_RED, "lw": 0.8},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
    )
    ax_trade.annotate(
        "选定日前保留策略\nw=7，q=0.75",
        (reserve.emergency_purchase_kwh, reserve.total_cost_yuan / 10000.0),
        xytext=(-58, 13), textcoords="offset points", fontsize=7.3, color=BOUND_GREY,
        arrowprops={"arrowstyle": "-", "color": BOUND_GREY, "lw": 0.8},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
    )
    ax_trade.set_xlabel("紧急购电量（kWh）")
    ax_trade.set_ylabel("实际总成本（万元）")
    panel_title(ax_top, "(c)", "不同日末保留目标下的成本—紧急购电权衡", y=1.72)

    reserve_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=reserve_colors[s],
               label=f"$S^{{res}}={s}$ kWh", markersize=5.3)
        for s in [1200, 3600, 6000]
    ]
    ax_top.legend(handles=reserve_handles, loc="lower left", bbox_to_anchor=(0.0, 1.03),
                  ncol=3, frameon=False, handletextpad=0.35, columnspacing=0.75)
    marker_handles = [
        Line2D([0], [0], marker=markers[w], linestyle="none", color=BOUND_GREY,
               label=f"w={w} 天", markersize=5.3)
        for w in [7, 14, 21]
    ]
    ax_trade.legend(handles=marker_handles, loc="lower right", ncol=3, frameon=False,
                    handletextpad=0.35, columnspacing=0.75)

    for axis in [ax_cost, ax_emergency, ax_trade]:
        style_axis(axis)
    for marginal in [ax_top, ax_right]:
        marginal.grid(False)
        marginal.tick_params(axis="both", which="both", left=False, right=False,
                             top=False, bottom=False, labelleft=False, labelright=False,
                             labeltop=False, labelbottom=False)
        for spine in marginal.spines.values():
            spine.set_visible(False)
    x_pad = max((x_max - x_min) * 0.04, 1.0)
    y_pad = max((y_max - y_min) * 0.05, 0.01)
    ax_trade.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_trade.set_ylim(y_min - y_pad, y_max + y_pad)
    return save_figure(fig, "q2_parameter_tradeoff")


def export_q2_reserve_comparison_table(display: pd.DataFrame) -> Path:
    """Export the three fixed-prediction reserve records as a concise table."""
    formal_rho = 0.6976
    focus = display[
        (display.w_days == 7) & np.isclose(display.q, 0.75)
        & np.isclose(display.rho_yuan_per_kwh, formal_rho)
        & display.Sres_kwh.isin([1200, 3600, 6000])
    ].sort_values("Sres_kwh").copy()
    if len(focus) != 3:
        raise ValueError(f"Expected three reserve-comparison records; found {len(focus)}")
    base = focus.iloc[0]
    names = {
        1200: "无额外保留",
        3600: "选定日前保留",
        6000: "较高保留目标",
    }
    table = pd.DataFrame({
        "策略": [names[int(round(v))] for v in focus.Sres_kwh],
        "w_days": focus.w_days.astype(int),
        "q": focus.q,
        "Sres_kwh": focus.Sres_kwh,
        "rho_yuan_per_kwh": focus.rho_yuan_per_kwh,
        "total_cost_yuan": focus.total_cost_yuan,
        "emergency_purchase_kwh": focus.emergency_purchase_kwh,
        "cost_change_vs_1200_yuan": focus.total_cost_yuan - float(base.total_cost_yuan),
        "emergency_change_vs_1200_kwh": focus.emergency_purchase_kwh - float(base.emergency_purchase_kwh),
    })
    path = ROOT / "outputs" / "tables" / "q2_fixed_prediction_reserve_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")
    return path


def plot_q2_case(case: pd.DataFrame, states: np.ndarray) -> tuple[Path, Path]:
    x, pred = slot_step_xy(case["net_load_pred_kwh"].to_numpy(dtype=float))
    _, actual = slot_step_xy(case["net_load_actual_kwh"].to_numpy(dtype=float))
    _, plan = slot_step_xy(case["planned_purchase_kwh"].to_numpy(dtype=float))
    _, charge_kw = slot_step_xy(case["charge_rt_kwh"].to_numpy(dtype=float) / DELTA_T)
    _, discharge_kw = slot_step_xy(-case["discharge_rt_kwh"].to_numpy(dtype=float) / DELTA_T)
    centers = (np.arange(N_SLOTS) + 0.5) * DELTA_T
    state_x = np.arange(145) * DELTA_T

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(FIG_WIDTH, 9.6), constrained_layout=True)
    axes[0].step(x, actual, where="post", color=ACTUAL_BLUE, lw=1.7, label="实际净负荷")
    axes[0].step(x, pred, where="post", color=FORECAST_BLUE, ls="--", lw=1.55, label="日前预测净负荷")
    axes[0].set_ylabel("每槽净负荷（kWh）")
    panel_title(axes[0], "(a)", "日前预测与实际净负荷", y=1.15)
    compact_legend(axes[0], loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2)

    axes[1].step(x, plan, where="post", color=PLAN_BLUE, lw=1.7, label="固定计划购电量")
    axes[1].bar(centers, case["emergency_purchase_kwh"], width=DELTA_T * 0.72,
                color=EMERGENCY_RED, label="实际紧急购电量", zorder=3)
    axes[1].set_ylabel("每槽购电量（kWh）")
    panel_title(axes[1], "(b)", "固定计划购电与实际紧急购电", y=1.15)
    compact_legend(axes[1], loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2)

    axes[2].step(x, charge_kw, where="post", color=CHARGE_GREEN, lw=1.6, label="充电功率")
    axes[2].step(x, discharge_kw, where="post", color=DISCHARGE_ORANGE, lw=1.6, label="放电功率（负值）")
    axes[2].axhline(P_LIMIT, color=BOUND_GREY, ls="--", lw=0.85, label="±5000 kW 功率限值")
    axes[2].axhline(-P_LIMIT, color=BOUND_GREY, ls="--", lw=0.85)
    axes[2].axhline(0, color=BOUND_GREY, lw=0.7)
    axes[2].set_ylabel("充放电功率（kW）")
    axes[2].set_ylim(-5500, 5500)
    axes[2].set_yticks([-5000, -2500, 0, 2500, 5000])
    panel_title(axes[2], "(c)", "实际充放电功率（充电为正，放电为负）", y=1.15)
    compact_legend(axes[2], loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3)

    axes[3].plot(state_x, states, color=ACTUAL_BLUE, lw=1.75, label="实际储电量")
    axes[3].fill_between(state_x, states, S_MIN, color=ACTUAL_BLUE, alpha=0.08)
    axes[3].axhline(S_MIN, color=BOUND_GREY, ls="--", lw=0.9, label="安全下限（1200 kWh）")
    axes[3].axhline(S_MAX, color=BOUND_GREY, ls=":", lw=0.9, label="安全上限（10800 kWh）")
    axes[3].scatter([0, 24], [states[0], states[-1]], color=ACTUAL_BLUE, s=17, zorder=5)
    axes[3].annotate(f"0:00，{states[0]:.3f} kWh", (0, states[0]), xytext=(0.65, states[0] + 850),
                     fontsize=7.8, color=ACTUAL_BLUE, arrowprops={"arrowstyle": "-", "color": ACTUAL_BLUE, "lw": 0.75})
    axes[3].annotate(f"24:00，{states[-1]:.3f} kWh", (24, states[-1]), xytext=(23.25, states[-1] + 850),
                     ha="right", fontsize=7.8, color=ACTUAL_BLUE,
                     arrowprops={"arrowstyle": "-", "color": ACTUAL_BLUE, "lw": 0.75})
    axes[3].set_ylabel("储电量（kWh）")
    axes[3].set_xlabel("时间（h）")
    axes[3].set_ylim(700, 11300)
    panel_title(axes[3], "(d)", "实际储电量与安全边界", y=1.15)
    compact_legend(axes[3], loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3)

    for axis in axes:
        style_axis(axis, time_axis=True)
    return save_figure(fig, "q2_case_dispatch")


def main() -> None:
    font_name, font_path = configure_chinese_theme()
    q1, q1_states, q1_checks = load_and_validate_q1()
    q2_case, q2_states, q2_checks = load_and_validate_q2_case()
    _, parameters, parameter_checks = load_and_validate_parameters()

    outputs = []
    outputs.extend(plot_q1_supply(q1))
    outputs.extend(plot_q1_storage(q1, q1_states))
    outputs.extend(plot_q2_sensitivity(parameters))
    outputs.extend(plot_q2_tradeoff(parameters))
    outputs.extend(plot_q2_case(q2_case, q2_states))
    reserve_table = export_q2_reserve_comparison_table(parameters)

    png_checks = []
    for path in outputs:
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                png_checks.append(
                    f"{path}: pixels={image.size[0]}x{image.size[1]}, dpi={image.info.get('dpi')}"
                )
    lines = [
        "Q1/Q2 paper figure v2 validation",
        "=" * 42,
        f"font: {font_name}",
        f"font_path: {font_path}",
        "font_glyph_check: PASS (Chinese, unit text, and Unicode minus covered)",
        f"Q1 source: {Q1_SOURCE}",
        f"Q1 source SHA-256: {sha256(Q1_SOURCE)}",
        f"Q2 parameter source: {Q2_PARAMETER_SOURCE}",
        f"Q2 parameter SHA-256: {sha256(Q2_PARAMETER_SOURCE)}",
        f"Q2 replay source: {Q2_REPLAY_SOURCE}",
        f"Q2 replay SHA-256: {sha256(Q2_REPLAY_SOURCE)}",
        "",
        *q1_checks,
        *q2_checks,
        *parameter_checks,
        "",
        "Slot mapping: official row t is drawn on [t/6,(t+1)/6] h; no row reorder or time-label parsing.",
        "State mapping: 145 states are drawn at 0,1/6,...,24 h with straight line segments.",
        "All kWh-to-kW conversions occurred on in-memory plotting copies only; source files were not written.",
        "No model solve, parameter search, interpolation, smoothing, fitting, or statistical inference was performed.",
        "",
        "Generated files:",
        *[str(path) for path in outputs],
        str(reserve_table),
        "",
        "PNG raster checks:",
        *png_checks,
    ]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
