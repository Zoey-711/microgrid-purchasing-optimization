"""Redraw only the two frozen Q3 V2 result figures from stage-4 CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.paper_figure_theme import configure_chinese_theme

ROOT = Path(__file__).resolve().parents[2]
S4 = ROOT / "outputs/q3_v2/stage4"
OUT = ROOT / "outputs/figures/q3_v2"
AUDIT = OUT / "q3_v2_final_figure_redraw_audit.json"
A, B = "A_all_updates_dynamic_value", "B_no_adjustment_dynamic_value"
DT = 1 / 6
BLUE, TEAL, ORANGE, RED, GREY, GRID = "#4573B4", "#66B2A3", "#E29D39", "#C44E52", "#6B7280", "#E8E8E8"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#333333"); spine.set_linewidth(.75)
    ax.tick_params(labelsize=8, top=False, right=False)
    ax.grid(axis="y", color=GRID, linewidth=.7, zorder=0)


def title(ax: plt.Axes, label: str, text: str) -> None:
    ax.set_title(f"({label}) {text}", loc="left", fontsize=10, fontweight="semibold", pad=8)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
        fig.savefig(OUT / f"{stem}{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def annual() -> dict:
    cmp = pd.read_csv(S4 / "q3_v2_full_year_comparison.csv").set_index("policy")
    monthly = pd.read_csv(S4 / "q3_v2_monthly_comparison.csv")
    a, b = cmp.loc[A], cmp.loc[B]
    savings_yuan = b.actual_total_cost_yuan - a.actual_total_cost_yuan
    if abs(savings_yuan - 628939.7525409386) > 1e-6:
        raise ValueError("annual frozen cost mismatch")
    fig = plt.figure(figsize=(10.5, 6.15))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, .78), hspace=.42, wspace=.28)
    ax0, ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    labels = ["允许日内调整", "禁止日内调整"]
    contract = np.array([a.contract_cost_yuan, b.contract_cost_yuan]) / 1e6
    emergency = np.array([a.emergency_cost_yuan, b.emergency_cost_yuan]) / 1e6
    total = contract + emergency
    x = np.arange(2)
    bars_contract = ax0.bar(x, contract, .54, color=BLUE, label="合同购电费用", zorder=3)
    ax0.bar(x, emergency, .54, bottom=contract, color=RED, label="紧急购电费用", zorder=3)
    for i, bar in enumerate(bars_contract):
        xc = bar.get_x() + bar.get_width() / 2
        ax0.text(xc, contract[i] / 2, f"{contract[i]:.3f}", ha="center", va="center", fontsize=8, color="white")
        ax0.text(xc, contract[i] + emergency[i] / 2, f"{emergency[i]:.3f}", ha="center", va="center", fontsize=8, color="white")
        ax0.annotate(f"合计 {total[i]:.3f}", (xc, total[i]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.4)
    pct = savings_yuan / b.actual_total_cost_yuan * 100
    ax0.text(.5, 14.86, f"总费用减少 {savings_yuan / 1e4:.2f} 万元（{pct:.3f}%）", ha="center", va="bottom", fontsize=8.0, color=RED)
    ax0.set_ylim(0, 15.25)
    ax0.set_xticks(x, labels); ax0.set_ylabel("费用（百万元）")
    ax0.legend(frameon=False, fontsize=7.0, ncol=2, loc="upper left", bbox_to_anchor=(.01, 1.0),
               handlelength=1.35, columnspacing=.55, handletextpad=.4, borderaxespad=0)
    title(ax0, "a", "全年费用构成"); style(ax0)
    values = np.array([a.emergency_purchase_kwh, b.emergency_purchase_kwh]) / 1e4
    bars = ax1.bar(x, values, .54, color=[TEAL, GREY], zorder=3)
    for bar in bars:
        ax1.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
    pct_e = (b.emergency_purchase_kwh - a.emergency_purchase_kwh) / b.emergency_purchase_kwh * 100
    ax1.text(.5, 19.15, f"减少 {(b.emergency_purchase_kwh - a.emergency_purchase_kwh) / 1e4:.3f} 万kWh\n（{pct_e:.3f}%）", ha="center", va="bottom", fontsize=8.1, color=RED)
    ax1.set_ylim(0, 20.4)
    ax1.set_xticks(x, labels); ax1.set_ylabel("紧急购电量（万 kWh）")
    title(ax1, "b", "全年紧急购电量"); style(ax1)
    pivot = monthly.pivot(index="month", columns="policy", values="actual_total_cost_yuan").reset_index()
    cumulative = np.cumsum(pivot[B] - pivot[A]) / 1e4
    month_x = np.arange(len(pivot))
    ax2.plot(month_x, cumulative, color=BLUE, marker="o", lw=1.9, ms=4.2, zorder=3)
    ax2.fill_between(month_x, 0, cumulative, color=BLUE, alpha=.10, zorder=1)
    # Label every monthly cumulative point; stagger close neighbouring labels
    # without changing any plotted values.
    for i, value in enumerate(cumulative):
        offset = 4 if i not in (8, 9) else 7
        ax2.annotate(f"{value:.2f}", (i, value), xytext=(0, offset), textcoords="offset points",
                     ha="center", fontsize=7.2)
    ax2.set_xticks(month_x, [f"{str(m).split('-')[1]}月" for m in pivot.month])
    ax2.set_ylabel("累计费用节省（万元）")
    title(ax2, "c", "累计费用节省"); style(ax2)
    save(fig, "q3_v2_annual_adjustment_effect")
    return {"contract_plus_emergency_error_yuan": float(max(abs(total * 1e6 - np.array([a.actual_total_cost_yuan, b.actual_total_cost_yuan])))), "saving_error_yuan": float(abs(cumulative.iloc[-1] * 1e4 - savings_yuan)), "emergency_labels_10kwh": values.tolist(), "cumulative_final_10k_yuan": float(cumulative.iloc[-1])}


def representative() -> dict:
    slots = pd.read_csv(S4 / "q3_v2_slot_results.csv")
    plans = pd.read_csv(S4 / "q3_v2_plan_versions.csv")
    day = "2025-03-20"
    d = slots[(slots.policy == A) & (slots.date == day)].sort_values("slot").reset_index(drop=True)
    p = plans[(plans.policy == A) & (plans.date == day)].copy()
    if len(d) != 144 or p.groupby("issue_hour").size().to_dict() != {0: 144, 6: 109, 12: 73, 18: 37}:
        raise ValueError("representative day / plan-version dimensions mismatch")
    t = d.slot.to_numpy() * DT
    fig, axes = plt.subplots(4, 1, figsize=(7.35, 8.15), sharex=True)
    colors = [(0, "#9AB9D3", "0:00 原计划"), (6, "#8CBFB5", "6:00 调整后计划"), (12, "#D7AF71", "12:00 调整后计划"), (18, "#D99491", "18:00 调整后计划")]
    # Plot the realised effective plan first so that the version lines remain readable
    # wherever they coincide; it is an execution trace, not the visual focal series.
    axes[0].step(t, d.effective_purchase_kwh, where="post", color="#5E7184", lw=1.25,
                 alpha=.78, label="实际生效计划", zorder=2)
    for issue, color, label in colors:
        q = p[p.issue_hour == issue].sort_values("slot")
        y = np.full(144, np.nan); y[q.slot.to_numpy(int)] = q.version_plan_kwh.to_numpy(float)
        axes[0].step(t, y, where="post", lw=1.2, color=color, label=label, zorder=3)
    axes[0].set_ylabel("购电量（kWh）"); title(axes[0], "a", "原计划、调整后计划与实际生效计划"); style(axes[0])
    axes[0].legend(ncol=3, fontsize=6.8, frameon=True, facecolor="white", edgecolor="none", framealpha=.86,
                   loc="upper right", borderaxespad=.45, handlelength=1.8, columnspacing=.7, handletextpad=.35)
    axes[1].bar(t, d.delta_plus_kwh, width=DT*.72, color=TEAL, label="上调量")
    axes[1].bar(t, -d.delta_minus_kwh, width=DT*.72, color=ORANGE, label="下调量（负值）")
    axes[1].bar(t, d.emergency_purchase_kwh, width=DT*.42, color=RED, label="紧急购电", zorder=3)
    axes[1].axhline(0, color=GREY, lw=.8); axes[1].set_ylabel("电量（kWh）")
    title(axes[1], "b", "购电调整量与紧急购电量"); style(axes[1]); axes[1].legend(ncol=1, fontsize=7, frameon=False, loc="upper right")
    axes[2].step(t, d.charge_rt_kwh / DT, where="post", color=TEAL, lw=1.55, label="充电功率（正）")
    axes[2].step(t, -d.discharge_rt_kwh / DT, where="post", color=ORANGE, lw=1.55, label="放电功率（负）")
    axes[2].axhline(5000, color="#B8BDC5", lw=.85, ls="--")
    axes[2].axhline(-5000, color="#B8BDC5", lw=.85, ls="--")
    axes[2].set_ylabel("功率（kW）"); title(axes[2], "c", "储能充放电功率与实际电价（充电为正、放电为负）"); style(axes[2])
    right = axes[2].twinx(); right.step(t, d.price_yuan_per_kwh, where="post", color="#4B5563", lw=1.15, ls=(0,(3,2)), alpha=.85, label="实际电价（右轴）")
    right.set_ylabel("电价（元/kWh）", fontsize=8); right.tick_params(labelsize=8); right.spines["right"].set_color("#333333"); right.spines["right"].set_linewidth(.75)
    axes[2].legend(handles=[Line2D([], [], color=TEAL, lw=1.55, label="充电功率"),
                            Line2D([], [], color=ORANGE, lw=1.55, label="放电功率"),
                            Line2D([], [], color="#4B5563", lw=1.15, ls=(0, (3, 2)), label="实际电价（右轴）")],
                   fontsize=6.9, ncol=3, frameon=False, loc="upper left", handlelength=1.8, columnspacing=.7)
    axes[2].text(23.75, 5000, "功率上限", ha="right", va="bottom", fontsize=6.5, color=GREY)
    axes[2].text(23.75, -5000, "功率下限", ha="right", va="top", fontsize=6.5, color=GREY)
    soc_t = np.r_[t, 24]; soc = np.r_[d.soc_start_kwh.iloc[0], d.soc_end_kwh]
    axes[3].plot(soc_t, soc, color=BLUE, lw=2.0, zorder=3); axes[3].axhline(1200, color=RED, ls="--", lw=1); axes[3].axhline(10800, color=GREY, ls="-.", lw=1)
    axes[3].set_ylabel("储电量（kWh）"); axes[3].set_xlabel("时间（h)"); title(axes[3], "d", "SOC 与安全边界"); style(axes[3])
    axes[3].text(23.75, 10800, "安全上限", ha="right", va="bottom", fontsize=6.6, color=GREY)
    axes[3].text(23.75, 1200, "安全下限", ha="right", va="bottom", fontsize=6.6, color=RED)
    axes[3].annotate("实际SOC", xy=(24, soc[-1]), xytext=(-4, 5), textcoords="offset points", ha="right", va="bottom", fontsize=6.8, color=BLUE)
    for ax in axes:
        for hour in (6, 12, 18): ax.axvline(hour, color="#9CA3AF", lw=.75, ls=(0,(1.5,2.2)), alpha=.9, zorder=1)
    for hour in (6, 12, 18): axes[0].annotate(f"{hour}:00更新", xy=(hour, .68), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=6.6, color=GREY,
                                                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .72, "pad": .25})
    fig.suptitle("2025年3月20日代表日滚动调度结果", fontsize=12, fontweight="semibold", y=.987)
    for axis in axes[:3]:
        axis.tick_params(axis="x", labelbottom=False, bottom=False)
    axes[3].set_xlim(0, 24); axes[3].set_xticks(range(0, 25, 4)); fig.subplots_adjust(top=.93, bottom=.07, hspace=.43)
    save(fig, "q3_v2_20250320_rolling_dispatch")
    return {"slot_rows": int(len(d)), "plan_version_rows": {str(k): int(v) for k,v in p.groupby("issue_hour").size().items()}, "date": day, "update_hours": [6,12,18]}


def main() -> None:
    configure_chinese_theme()
    audit = {"source_files": [str(S4 / "q3_v2_full_year_comparison.csv"), str(S4 / "q3_v2_monthly_comparison.csv"), str(S4 / "q3_v2_slot_results.csv"), str(S4 / "q3_v2_plan_versions.csv")], "annual": annual(), "representative": representative(), "outputs": ["q3_v2_annual_adjustment_effect.{png,pdf,svg}", "q3_v2_20250320_rolling_dispatch.{png,pdf,svg}"], "data_modified": False}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
