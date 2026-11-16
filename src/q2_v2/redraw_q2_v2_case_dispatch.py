"""Redraw the frozen 2025-06-01 Q2 V2 representative-day figure only.

This visualization script reads the frozen Stage-4 slot trajectories.  It does
not solve, export templates, or write back numerical results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paper_figure_theme import (
    ACTUAL_BLUE,
    BOUND_GREY,
    CHARGE_GREEN,
    DISCHARGE_ORANGE,
    EMERGENCY_RED,
    GRID_GREY,
    PLAN_BLUE,
    TEXT_GREY,
    compact_legend,
    configure_chinese_theme,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
SLOT_FILE = ROOT / "outputs" / "q2_v2" / "stage4" / "q2_v2_slot_results.csv"
FIG_DIR = ROOT / "outputs" / "figures" / "q2_v2"
LOG_FILE = ROOT / "outputs" / "q2_v2" / "q2_v2_case_dispatch_redraw_check.txt"
DATE = "2025-06-01"
DT = 1.0 / 6.0

PLAN_GREY_BLUE = "#7D93AD"
PRICE_GREY = "#A4ACB7"


def soc_path(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(([frame.iloc[0]["soc_start_kwh"]], frame["soc_end_kwh"].to_numpy(float)))
    return np.arange(145, dtype=float) * DT, values


def place_top_legend(axis: plt.Axes, handles, labels, *, ncol: int, fontsize: float = 7.2) -> None:
    compact_legend(
        axis,
        handles=handles,
        labels=labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        ncol=ncol,
        fontsize=fontsize,
        borderaxespad=0.0,
    )


def main() -> None:
    font, _ = configure_chinese_theme()
    data = pd.read_csv(SLOT_FILE)
    data["date"] = data["date"].astype(str)
    case = data.loc[data["date"].eq(DATE)].copy()
    dp = case.loc[case["policy_id"].eq("main_dp50")].sort_values("slot").reset_index(drop=True)
    greedy = case.loc[case["policy_id"].eq("closed_loop_greedy")].sort_values("slot").reset_index(drop=True)
    if len(dp) != 144 or len(greedy) != 144:
        raise ValueError("Frozen representative-day trajectories must each contain 144 slots.")

    time = dp["slot"].to_numpy(float) * DT
    net = dp["load_actual_kwh"].to_numpy(float) - dp["pv_actual_kwh"].to_numpy(float)
    price = dp["price_yuan_per_kwh"].to_numpy(float)
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 6.55), sharex=True, constrained_layout=True)
    fig.suptitle("2025年6月1日（全年最大单日紧急购电日）", fontsize=10.2, fontweight="semibold", y=1.015)

    # (a) Lines and offset bars express distinct physical quantities separately.
    ax = axes[0]
    h_net = ax.step(time, net, where="post", color=ACTUAL_BLUE, linewidth=1.9, label="实际净负荷", zorder=4)[0]
    h_plan = ax.step(
        time,
        dp["planned_purchase_kwh"].to_numpy(float),
        where="post",
        color=PLAN_GREY_BLUE,
        linestyle=(0, (4, 2.2)),
        linewidth=1.32,
        label="计划购电量",
        zorder=3,
    )[0]
    width = DT * 0.24
    h_dp = ax.bar(
        time - width * 0.78,
        dp["emergency_purchase_kwh"].to_numpy(float),
        width=width,
        color=EMERGENCY_RED,
        alpha=0.90,
        label="动态价值",
        zorder=2,
    )
    h_greedy = ax.bar(
        time + width * 0.78,
        greedy["emergency_purchase_kwh"].to_numpy(float),
        width=width,
        color=BOUND_GREY,
        alpha=0.82,
        hatch="///",
        edgecolor=BOUND_GREY,
        linewidth=0.35,
        label="贪心",
        zorder=2,
    )
    ax.set_ylabel("每槽净负荷与购电量（kWh）")
    panel_title(ax, "(a)", "代表日净负荷、计划购电与紧急购电")
    style_axis(ax, time_axis=True)
    place_top_legend(ax, [h_net, h_plan, h_dp, h_greedy], ["实际净负荷", "计划购电量", "动态价值", "贪心"], ncol=2)
    ax.text(
        0.015, 0.065, "0:00—8:00 两种执行器\n紧急购电均为 0",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color=TEXT_GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5}, zorder=6,
    )

    # (b) A price line on a lightly styled right axis replaces fragmented bands.
    ax = axes[1]
    charge_kw = dp["charge_kwh"].to_numpy(float) / DT
    discharge_kw = dp["discharge_kwh"].to_numpy(float) / DT
    h_charge = ax.step(time, charge_kw, where="post", color=CHARGE_GREEN, linewidth=1.7, label="充电功率", zorder=3)[0]
    h_discharge = ax.step(time, -discharge_kw, where="post", color=DISCHARGE_ORANGE, linewidth=1.7, label="放电功率", zorder=3)[0]
    h_limit = ax.axhline(5000, color=BOUND_GREY, linestyle="--", linewidth=0.9, label="充放电功率限值（±5000 kW）", zorder=2)
    ax.axhline(-5000, color=BOUND_GREY, linestyle="--", linewidth=0.9, zorder=2)
    ax.set_ylabel("功率（kW）")
    panel_title(ax, "(b)", "动态价值策略实际充放电功率")
    style_axis(ax, time_axis=True)
    price_axis = ax.twinx()
    h_price = price_axis.step(time, price, where="post", color=PRICE_GREY, alpha=0.55, linestyle=(0, (2, 2)), linewidth=0.95, label="分时电价", zorder=1)[0]
    price_axis.set_ylabel("电价（元/kWh）", color=BOUND_GREY)
    price_axis.tick_params(axis="y", colors=BOUND_GREY, width=0.75, length=3.2)
    price_axis.spines["right"].set_color(BOUND_GREY)
    price_axis.spines["right"].set_linewidth(0.75)
    price_axis.spines["top"].set_visible(False)
    place_top_legend(ax, [h_charge, h_discharge, h_limit, h_price], ["充电功率", "放电功率", "充放电功率限值（±5000 kW）", "分时电价"], ncol=2, fontsize=6.8)

    # (c) Greedy path is drawn over the main path so overlaps remain explicit.
    ax = axes[2]
    td, soc_dp = soc_path(dp)
    tg, soc_greedy = soc_path(greedy)
    h_soc_dp = ax.plot(td, soc_dp, color=ACTUAL_BLUE, linewidth=2.2, label="动态价值 SOC", zorder=5)[0]
    ax.fill_between(td, soc_dp, soc_greedy, color=PLAN_BLUE, alpha=0.09, zorder=1, label="SOC 轨迹差")
    h_soc_greedy = ax.plot(tg, soc_greedy, color=BOUND_GREY, linewidth=1.75, linestyle=(0, (3, 1.7)), label="贪心 SOC", zorder=4)[0]
    h_lower = ax.axhline(1200, color=EMERGENCY_RED, linestyle="--", linewidth=1.0, label="安全下限", zorder=2)
    h_upper = ax.axhline(10800, color=BOUND_GREY, linestyle="-.", linewidth=1.0, label="安全上限", zorder=2)
    ax.set_ylabel("储电量（kWh）")
    ax.set_xlabel("时间（h）")
    panel_title(ax, "(c)", "两种执行器的 SOC 轨迹")
    style_axis(ax, time_axis=True)
    place_top_legend(ax, [h_soc_dp, h_soc_greedy, h_lower, h_upper], ["动态价值 SOC", "贪心 SOC", "安全下限", "安全上限"], ncol=2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "q2_v2_20250601_dispatch"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    LOG_FILE.write_text(
        "问题二 V2 代表日图重绘核查\n"
        f"中文字体：{font}\n"
        "数据来源：冻结 stage4/q2_v2_slot_results.csv；日期：2025-06-01。\n"
        "(a) 实际净负荷与计划购电量为阶梯曲线；两策略紧急购电为并列窄柱。\n"
        "0:00—8:00 两条冻结轨迹均无紧急购电；图 (a) 已增加该事实性标注。\n"
        "(b) 取消碎片化电价色带，改以右轴浅灰阶梯线展示分时电价；充放电功率由每槽 kWh 除以 1/6 h 得到。\n"
        "(c) 使用145个真实 SOC 状态点；浅填充仅展示两条轨迹的差异。\n"
        "未修改模型、冻结结果、Excel 或时间映射。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
