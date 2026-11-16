"""Cross-strategy descriptive comparison based only on frozen annual summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paper_figure_theme import ACTUAL_BLUE, EMERGENCY_RED, PV_GREEN, configure_chinese_theme, panel_title, style_axis

ROOT = Path(__file__).resolve().parents[1]
Q2 = ROOT / "outputs/q2_v2/stage4/q2_v2_final_comparison.csv"
Q3 = ROOT / "outputs/q3_v2/stage4/q3_v2_full_year_comparison.csv"
Q4 = ROOT / "outputs/q4_v1/stage4/q4_v1_full_year_comparison.csv"
OUT = ROOT / "outputs/figures/paper_v2"
AUDIT = OUT / "strategy_total_cost_emergency_comparison.audit.md"


def save(fig: plt.Figure) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    result = []
    for ext, args in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
        p = OUT / f"strategy_total_cost_emergency_comparison{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white", **args)
        result.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return result


def main() -> None:
    configure_chinese_theme()
    q2, q3, q4 = pd.read_csv(Q2), pd.read_csv(Q3), pd.read_csv(Q4)
    q2dp = q2.loc[q2.policy_id.eq("main_dp50")].iloc[0]
    q2gr = q2.loc[q2.policy_id.eq("closed_loop_greedy")].iloc[0]
    q3allow = q3.loc[q3.policy.eq("A_all_updates_dynamic_value")].iloc[0]
    q3block = q3.loc[q3.policy.eq("B_no_adjustment_dynamic_value")].iloc[0]
    # Q4 policy strings contain Chinese labels; final-plan equality is the
    # frozen discriminant: q4-2 has no adjustments, q4-3 has adjustments.
    q42 = q4.loc[q4.upward_adjustment_kwh.abs() < 1e-9].iloc[0]
    q43 = q4.loc[q4.upward_adjustment_kwh.abs() >= 1e-9].iloc[0]
    labels = ["Q2\n动态价值", "Q2\n贪心对照", "Q3\n允许调整", "Q3\n禁止调整", "Q4-2\n日前锁定", "Q4-3\n滚动调整"]
    cost = np.array([q2dp.actual_total_cost_yuan, q2gr.actual_total_cost_yuan, q3allow.actual_total_cost_yuan, q3block.actual_total_cost_yuan, q42.actual_total_cost_yuan, q43.actual_total_cost_yuan]) / 1e6
    emerg = np.array([q2dp.emergency_purchase_kwh, q2gr.emergency_purchase_kwh, q3allow.emergency_purchase_kwh, q3block.emergency_purchase_kwh, q42.emergency_purchase_kwh, q43.emergency_purchase_kwh]) / 1e4
    colors = [ACTUAL_BLUE, "#6B7280", PV_GREEN, "#6B7280", "#6B7280", PV_GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), gridspec_kw={"wspace": .30})
    x = np.arange(6)
    for ax, values, ylabel, title in ((axes[0], cost, "实际总费用（百万元）", "不同策略的实际总费用"), (axes[1], emerg, "紧急购电量（万kWh）", "不同策略的紧急购电量")):
        bars = ax.bar(x, values, .62, color=colors, zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3f}", (bar.get_x()+bar.get_width()/2, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.1)
        ax.set_xticks(x, labels, rotation=0)
        ax.set_ylabel(ylabel)
        for boundary in (1.5, 3.5):
            ax.axvline(boundary, color="#B8BDC5", lw=.75, ls="--", zorder=1)
        ymax = values.max() * 1.13
        ax.set_ylim(0, ymax)
        panel_title(ax, "(a)" if ax is axes[0] else "(b)", title)
        style_axis(ax)
    axes[0].text(.5, .985, "固定电价", transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=7.3, color="#4B5563")
    axes[0].text(4.5, .985, "波动电价", transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=7.3, color="#4B5563")
    axes[1].text(.5, .985, "固定电价", transform=axes[1].get_xaxis_transform(), ha="center", va="top", fontsize=7.3, color="#4B5563")
    axes[1].text(4.5, .985, "波动电价", transform=axes[1].get_xaxis_transform(), ha="center", va="top", fontsize=7.3, color="#4B5563")
    fig.suptitle("不同策略的总费用与紧急购电量比较", fontsize=13, fontweight="semibold", y=.995)
    fig.text(.5, .004, "问题二、三为固定电价回放，问题四为波动电价回放；跨组柱高仅作描述性展示，不代表可直接归因的策略优劣。", ha="center", fontsize=7.25, color="#4B5563")
    paths = save(fig)
    audit = {"source_files": [str(Q2.relative_to(ROOT)), str(Q3.relative_to(ROOT)), str(Q4.relative_to(ROOT))], "strategies": labels,
             "total_cost_million_yuan": cost.tolist(), "emergency_10k_kwh": emerg.tolist(), "outputs": paths, "data_modified": False}
    AUDIT.write_text("# 不同策略总费用与紧急购电量比较：数据审计\n\n```json\n" + json.dumps(audit, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print("CROSS_STRATEGY_COMPARISON_PASS")


if __name__ == "__main__":
    main()
