"""Presentation-only rebuild of frozen Q1/Q3 paper figures.

This module only reads already frozen CSV trajectories and plan-version logs.
It deliberately contains no forecasting, optimisation, scheduling, or writes to
source data / workbooks.  All unit conversions occur in plotting arrays only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paper_figure_theme import configure_chinese_theme, style_axis, panel_title, compact_legend


ROOT = Path(__file__).resolve().parents[1]
Q1_SOURCE = ROOT / "data/processed/q1_baseline_dispatch.csv"
Q3_STAGE4 = ROOT / "outputs/q3_v2/stage4"
Q3_SLOTS = Q3_STAGE4 / "q3_v2_slot_results.csv"
Q3_PLANS = Q3_STAGE4 / "q3_v2_plan_versions.csv"
OUT = ROOT / "outputs/figures/paper_rebuild"
LOG = ROOT / "outputs/logs/q1_q3_paper_rebuild_check.json"

DT, N = 1 / 6, 144
S_MIN, S_MAX = 1200.0, 10800.0
P_MAX = 5000.0
Q3_POLICY, Q3_DATE = "A_all_updates_dynamic_value", "2025-03-20"

BLUE, PLAN_GREY, PLAN_CYAN, PLAN_ORANGE, PLAN_RED = "#4573B4", "#98A1AA", "#8ABFC4", "#D7A75E", "#D68F8B"
TEAL, ORANGE, RED, PRICE, BOUND, GRID, TEXT = "#66B2A3", "#E29D39", "#C44E52", "#C98B2D", "#B8BDC5", "#6B7280", "#30343B"


def step_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.size != N:
        raise ValueError(f"Expected {N} slots, got {values.size}")
    return np.arange(N + 1) * DT, np.r_[values, values[-1]]


def clean_axis(ax, *, time_axis: bool = True) -> None:
    style_axis(ax, time_axis=time_axis)
    ax.grid(axis="x", visible=False)


def save(fig: plt.Figure, stem: str) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, args in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
        path = OUT / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **args)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def q1_data() -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(Q1_SOURCE)
    required = {"load_kwh", "pv_forecast_kwh", "price", "purchase_kwh", "charge_kwh", "discharge_kwh", "soc_kwh"}
    if len(data) != N or not required.issubset(data.columns):
        raise ValueError("Frozen Q1 source does not have exactly 144 expected rows/fields")
    if not np.isfinite(data[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("Frozen Q1 plotting data contains non-finite values")
    soc = np.r_[data.soc_kwh.to_numpy(float), data.soc_kwh.iloc[-1] + .9 * data.charge_kwh.iloc[-1] - data.discharge_kwh.iloc[-1] / .9]
    if soc.size != 145 or not np.isclose(soc[0], 6000) or not np.isclose(soc[-1], 6000):
        raise ValueError("Q1 frozen SOC start/end or state count mismatch")
    return data, soc


def plot_q1_netload(data: pd.DataFrame) -> dict:
    net_kw = (data.load_kwh.to_numpy(float) - data.pv_forecast_kwh.to_numpy(float)) / DT
    price = data.price.to_numpy(float)
    purchase = data.purchase_kwh.to_numpy(float)
    x, net = step_xy(net_kw)
    _, price_step = step_xy(price)
    _, purchase_step = step_xy(purchase)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.30, 5.30), gridspec_kw={"hspace": .17})
    axes[0].step(x, net, where="post", color="#52677A", lw=1.55, label="净负荷")
    axes[0].axhline(0, color=TEXT, lw=.65)
    axes[0].set_ylabel("净负荷（kW）")
    panel_title(axes[0], "(a)", "净负荷")
    compact_legend(axes[0], loc="upper left")
    axes[1].step(x, price_step, where="post", color=PRICE, lw=1.55, label="分时电价")
    axes[1].set_ylabel("电价（元/kWh）")
    panel_title(axes[1], "(b)", "分时电价")
    compact_legend(axes[1], loc="upper left")
    axes[2].step(x, purchase_step, where="post", color=BLUE, lw=1.65, label="计划购电量", zorder=3)
    axes[2].fill_between(x, purchase_step, 0, step="post", color=BLUE, alpha=.10, zorder=1)
    axes[2].set_ylabel("购电量（kWh）")
    axes[2].set_xlabel("时间（h）")
    panel_title(axes[2], "(c)", "计划购电量")
    compact_legend(axes[2], loc="upper left")
    for ax in axes:
        clean_axis(ax)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    paths = save(fig, "q1_netload_price_purchase")
    return {"input": str(Q1_SOURCE.relative_to(ROOT)), "rows": len(data), "slots": N, "time_range_h": [0, 24],
            "net_load_kw_minmax": [float(net_kw.min()), float(net_kw.max())], "price_minmax": [float(price.min()), float(price.max())],
            "purchase_kwh_minmax": [float(purchase.min()), float(purchase.max())], "outputs": paths, "consistent": True}


def plot_q1_storage(data: pd.DataFrame, soc: np.ndarray) -> dict:
    x, charge = step_xy(data.charge_kwh.to_numpy(float) / DT)
    _, discharge = step_xy(-data.discharge_kwh.to_numpy(float) / DT)
    sx = np.arange(145) * DT
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.30, 4.65), gridspec_kw={"hspace": .20})
    axes[0].step(x, charge, where="post", color=TEAL, lw=1.55, label="充电功率")
    axes[0].step(x, discharge, where="post", color=ORANGE, lw=1.55, label="放电功率")
    axes[0].axhline(P_MAX, color=BOUND, ls="--", lw=.75, alpha=.82)
    axes[0].axhline(-P_MAX, color=BOUND, ls="--", lw=.75, alpha=.82)
    axes[0].axhline(0, color=TEXT, lw=.65)
    axes[0].set_ylim(-5500, 5500)
    axes[0].set_ylabel("功率（kW）")
    panel_title(axes[0], "(a)", "储能充放电功率（充电为正、放电为负）")
    compact_legend(axes[0], loc="upper right")
    axes[1].plot(sx, soc, color=BLUE, lw=1.75, label="SOC", zorder=3)
    axes[1].fill_between(sx, S_MIN, soc, where=soc >= S_MIN, color=BLUE, alpha=.10, zorder=1)
    axes[1].axhline(S_MIN, color=BOUND, ls="--", lw=.75)
    axes[1].axhline(S_MAX, color=BOUND, ls="--", lw=.75)
    axes[1].scatter([0, 24], [soc[0], soc[-1]], color=BLUE, s=13, zorder=4)
    axes[1].annotate("初始：6000 kWh", xy=(0, soc[0]), xytext=(.6, 6500), fontsize=7.2, color=TEXT,
                     arrowprops={"arrowstyle": "-", "lw": .6, "color": BOUND})
    axes[1].annotate("末端：6000 kWh", xy=(24, soc[-1]), xytext=(23.35, 6500), ha="right", fontsize=7.2, color=TEXT,
                     arrowprops={"arrowstyle": "-", "lw": .6, "color": BOUND})
    axes[1].set_ylim(700, 11300)
    axes[1].set_ylabel("储电量（kWh）")
    axes[1].set_xlabel("时间（h）")
    panel_title(axes[1], "(b)", "SOC 与安全边界")
    compact_legend(axes[1], loc="upper left")
    for ax in axes:
        clean_axis(ax)
    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    paths = save(fig, "q1_storage_response_soc")
    return {"input": str(Q1_SOURCE.relative_to(ROOT)), "rows": len(data), "slots": N, "soc_states": len(soc), "time_range_h": [0, 24],
            "charge_kw_minmax": [float(charge.min()), float(charge.max())], "discharge_kw_minmax": [float(discharge.min()), float(discharge.max())],
            "soc_minmax": [float(soc.min()), float(soc.max())], "soc_start_end": [float(soc[0]), float(soc[-1])], "outputs": paths, "consistent": True}


def q3_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    slots = pd.read_csv(Q3_SLOTS)
    plans = pd.read_csv(Q3_PLANS)
    d = slots[(slots.policy == Q3_POLICY) & (slots.date == Q3_DATE)].sort_values("slot").reset_index(drop=True)
    p = plans[(plans.policy == Q3_POLICY) & (plans.date == Q3_DATE)].copy()
    expected = {0: 144, 6: 109, 12: 73, 18: 37}
    if len(d) != N or not np.array_equal(d.slot.to_numpy(int), np.arange(N)) or p.groupby("issue_hour").size().to_dict() != expected:
        raise ValueError("Frozen Q3 representative data dimensions do not match the official 144-slot contract")
    return d, p


def q3_vertical_updates(axes: list[plt.Axes], *, labels: bool) -> None:
    for ax in axes:
        for hour in (6, 12, 18):
            ax.axvline(hour, color="#BFC5CC", lw=.72, ls=(0, (1.4, 2.2)), alpha=.75, zorder=1)
    if labels:
        for hour in (6, 12, 18):
            axes[0].annotate(f"{hour}:00更新", xy=(hour, .97), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=6.8, color=BOUND,
                             bbox={"facecolor": "white", "edgecolor": "none", "alpha": .80, "pad": .22})


def plot_q3_plan(d: pd.DataFrame, p: pd.DataFrame) -> dict:
    t = d.slot.to_numpy(float) * DT
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.55, 5.15), gridspec_kw={"hspace": .20})
    palette = [(0, PLAN_GREY, "0:00原计划"), (6, PLAN_CYAN, "6:00调整后计划"), (12, PLAN_ORANGE, "12:00调整后计划"), (18, PLAN_RED, "18:00调整后计划")]
    for issue, color, label in palette:
        q = p[p.issue_hour == issue].sort_values("slot")
        y = np.full(N, np.nan); y[q.slot.to_numpy(int)] = q.version_plan_kwh.to_numpy(float)
        axes[0].step(t, y, where="post", color=color, lw=1.15, label=label, zorder=3)
    axes[0].step(t, d.effective_purchase_kwh, where="post", color=BLUE, lw=2.25, label="最终执行计划", zorder=5)
    axes[0].set_ylabel("购电量（kWh）")
    panel_title(axes[0], "(a)", "各时点购电计划与最终执行轨迹")
    # Keep the five-version legend above the data region: the in-axis update
    # labels at 6/12/18 h must remain readable and no plan segment is covered.
    compact_legend(axes[0], loc="upper center", bbox_to_anchor=(.72, 1.21), ncol=3,
                   fontsize=6.7, columnspacing=.65, handlelength=1.7)
    axes[1].bar(t, d.delta_plus_kwh, width=DT*.70, color=TEAL, label="上调量", zorder=3)
    axes[1].bar(t, -d.delta_minus_kwh, width=DT*.70, color=ORANGE, label="下调量（负值）", zorder=3)
    axes[1].bar(t, d.emergency_purchase_kwh, width=DT*.40, color=RED, label="紧急购电", zorder=4)
    axes[1].axhline(0, color=TEXT, lw=.65)
    axes[1].set_ylabel("电量（kWh）")
    axes[1].set_xlabel("时间（h）")
    panel_title(axes[1], "(b)", "调整购电量与紧急购电量")
    compact_legend(axes[1], loc="upper left", ncol=3, fontsize=7.0)
    q3_vertical_updates(axes, labels=True)
    for ax in axes:
        clean_axis(ax)
    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    paths = save(fig, "q3_plan_evolution")
    return {"inputs": [str(Q3_SLOTS.relative_to(ROOT)), str(Q3_PLANS.relative_to(ROOT))], "date": Q3_DATE, "rows": len(d), "slots": N,
            "plan_version_rows": {str(k): int(v) for k, v in p.groupby("issue_hour").size().items()}, "time_range_h": [0, 24],
            "original_plan_minmax": [float(p[p.issue_hour == 0].version_plan_kwh.min()), float(p[p.issue_hour == 0].version_plan_kwh.max())],
            "effective_plan_minmax": [float(d.effective_purchase_kwh.min()), float(d.effective_purchase_kwh.max())], "outputs": paths, "consistent": True}


def plot_q3_storage(d: pd.DataFrame) -> dict:
    t = d.slot.to_numpy(float) * DT
    soc_x = np.r_[t, 24]
    soc = np.r_[d.soc_start_kwh.iloc[0], d.soc_end_kwh.to_numpy(float)]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.55, 6.05), gridspec_kw={"hspace": .18})
    axes[0].step(t, d.price_yuan_per_kwh, where="post", color=PRICE, lw=1.55, label="实际电价")
    axes[0].set_ylabel("电价（元/kWh）")
    panel_title(axes[0], "(a)", "分时电价")
    compact_legend(axes[0], loc="upper left")
    axes[1].step(t, d.charge_rt_kwh / DT, where="post", color=TEAL, lw=1.55, label="充电功率")
    axes[1].step(t, -d.discharge_rt_kwh / DT, where="post", color=ORANGE, lw=1.55, label="放电功率")
    axes[1].axhline(P_MAX, color=BOUND, lw=.75, ls="--")
    axes[1].axhline(-P_MAX, color=BOUND, lw=.75, ls="--")
    axes[1].axhline(0, color=TEXT, lw=.65)
    axes[1].set_ylabel("功率（kW）")
    panel_title(axes[1], "(b)", "储能充放电功率")
    compact_legend(axes[1], loc="upper right")
    axes[2].plot(soc_x, soc, color=BLUE, lw=1.75, label="实际SOC", zorder=3)
    axes[2].fill_between(soc_x, S_MIN, soc, where=soc >= S_MIN, color=BLUE, alpha=.09, zorder=1)
    axes[2].axhline(S_MIN, color=BOUND, lw=.75, ls="--", label="安全下限")
    axes[2].axhline(S_MAX, color=BOUND, lw=.75, ls="--", label="安全上限")
    axes[2].set_ylabel("储电量（kWh）")
    axes[2].set_xlabel("时间（h）")
    panel_title(axes[2], "(c)", "SOC")
    compact_legend(axes[2], loc="upper left", ncol=3, fontsize=6.8)
    q3_vertical_updates(axes, labels=False)
    for ax in axes:
        clean_axis(ax)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    paths = save(fig, "q3_storage_response")
    return {"input": str(Q3_SLOTS.relative_to(ROOT)), "date": Q3_DATE, "rows": len(d), "slots": N, "soc_states": len(soc), "time_range_h": [0, 24],
            "price_minmax": [float(d.price_yuan_per_kwh.min()), float(d.price_yuan_per_kwh.max())],
            "charge_kw_minmax": [float((d.charge_rt_kwh / DT).min()), float((d.charge_rt_kwh / DT).max())],
            "discharge_kw_minmax": [float((-d.discharge_rt_kwh / DT).min()), float((-d.discharge_rt_kwh / DT).max())],
            "soc_minmax": [float(soc.min()), float(soc.max())], "soc_start_end": [float(soc[0]), float(soc[-1])], "outputs": paths, "consistent": True}


def main() -> None:
    configure_chinese_theme()
    q1, q1_soc = q1_data()
    q3, q3_plans = q3_data()
    audit = {
        "scope": "presentation-only; no optimisation, forecast, scheduling, CSV/Excel/data modification",
        "q1_netload_price_purchase": plot_q1_netload(q1),
        "q1_storage_response_soc": plot_q1_storage(q1, q1_soc),
        "q3_plan_evolution": plot_q3_plan(q3, q3_plans),
        "q3_storage_response": plot_q3_storage(q3),
        "source_data_modified": False,
        "old_figures_overwritten": False,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
