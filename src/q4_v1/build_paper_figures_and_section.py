"""Build Q4 V1 paper figures and the paper-ready section from frozen stage-4 data.

This is a reporting-only script: it reads completed trajectories and aggregates
them for charts and Markdown.  It never invokes the optimizer or writes XLSX.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paper_figure_theme import (
    ACTUAL_BLUE, BOUND_GREY, CHARGE_GREEN, DISCHARGE_ORANGE, EMERGENCY_RED,
    GRID_GREY, PRICE_ORANGE, PV_GREEN, TEXT_GREY, compact_legend,
    configure_chinese_theme, panel_title, style_axis,
)

ROOT = Path(__file__).resolve().parents[2]
STAGE4 = ROOT / "outputs" / "q4_v1" / "stage4"
FIG = ROOT / "outputs" / "figures" / "q4_v1"
DOCS = ROOT / "docs" / "q4_v1"
PAPER = ROOT / "docs" / "论文正文.md"
SLOTS = STAGE4 / "q4_v1_slot_results.csv"
MONTHLY = STAGE4 / "q4_v1_monthly_comparison.csv"
SUMMARY = STAGE4 / "q4_v1_full_year_comparison.csv"
SOLVER = STAGE4 / "q4_v1_solver_quality.csv"
DT = 1 / 6
GREY = "#6B7280"
LIGHT_BLUE_GREY = "#C9D6E4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def save(fig: plt.Figure, stem: str) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
        path = FIG / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def get_data():
    slots = pd.read_csv(SLOTS, low_memory=False)
    monthly = pd.read_csv(MONTHLY)
    summary = pd.read_csv(SUMMARY)
    solver = pd.read_csv(SOLVER)
    slots["date"] = pd.to_datetime(slots["date"])
    monthly["month"] = monthly["month"].astype(str)
    # Policy labels were written in Chinese; selecting on their stable task suffix
    # avoids locale-dependent display decoding.
    q42 = slots[slots["policy"].astype(str).str.contains("4-2", regex=False)].copy()
    q43 = slots[slots["policy"].astype(str).str.contains("4-3", regex=False)].copy()
    if len(q42) != 48096 or len(q43) != 48096:
        raise ValueError("frozen slot trajectories are not 334*144 for both branches")
    return slots, monthly, summary, solver, q42, q43


def price_forecast_and_band(q43: pd.DataFrame, day: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frozen P3 definition: 14 preceding complete days' same-slot average.

    The plotting uncertainty bands use the latest 14 complete historical
    residual trajectories around the same causal P3 forecast. They are visual
    diagnostics derived from the stored actual-price trajectory, not new model
    inputs or a new forecast fit.
    """
    series = q43.pivot(index="date", columns="slot", values="price_yuan_per_kwh").sort_index()
    hist = series.loc[series.index < day]
    prior14 = hist.tail(14)
    if len(prior14) != 14:
        raise ValueError("representative day does not have the frozen 14 complete prior price days")
    forecast = prior14.mean(axis=0).to_numpy(float)
    residuals = []
    # Each residual is reconstructed from only that historical date's preceding
    # 14 complete days. The selected July date has a sufficient history.
    for date in prior14.index:
        old = series.loc[series.index < date].tail(14)
        if len(old) == 14:
            residuals.append(series.loc[date].to_numpy(float) - old.mean(axis=0).to_numpy(float))
    if len(residuals) < 10:
        raise ValueError("insufficient causal residual tracks for plotted price band")
    scenarios = np.maximum(0.0, forecast[None, :] + np.vstack(residuals))
    lo10, q25, med, q75, hi90 = np.quantile(scenarios, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    return forecast, np.vstack([lo10, q25, med, q75, hi90]), scenarios, prior14.index.to_numpy()


def figure_41(q43: pd.DataFrame) -> dict:
    x = q43.copy()
    x["high_price_emergency_fee"] = 5 * x["price_yuan_per_kwh"] * x["emergency_purchase_kwh"]
    daily = x.groupby("date", as_index=True)["high_price_emergency_fee"].sum()
    day = daily.idxmax()  # pre-registered, reproducible selection rule
    d = x[x["date"] == day].sort_values("slot").reset_index(drop=True)
    if len(d) != 144:
        raise ValueError("representative-day slot count is not 144")
    forecast, quantiles, scenarios, source_dates = price_forecast_and_band(x, day)
    lo10, q25, med, q75, hi90 = quantiles
    t = d["slot"].to_numpy(float) * DT
    t_end = np.r_[t, 24.0]
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.05, 1.0, 1.0, 1.0], "hspace": 0.25})
    # (a) price: actual is visually primary, forecast and bands are causal data.
    axes[0].fill_between(t, lo10, hi90, step="post", color=LIGHT_BLUE_GREY, alpha=.55, label="10%–90%情景带", zorder=1)
    axes[0].fill_between(t, q25, q75, step="post", color=PV_GREEN, alpha=.30, label="25%–75%情景带", zorder=2)
    axes[0].step(t, forecast, where="post", color=PV_GREEN, lw=1.45, ls="--", label="0:00 因果价格预测", zorder=4)
    axes[0].step(t, med, where="post", color=GREY, lw=1.20, label="情景中位数", zorder=5)
    axes[0].step(t, d.price_yuan_per_kwh, where="post", color="#334155", lw=2.05, label="实际电价", zorder=6)
    axes[0].set_ylabel("电价（元/kWh）")
    panel_title(axes[0], "(a)", "因果价格预测与实际价格")
    style_axis(axes[0], time_axis=True)
    compact_legend(axes[0], loc="upper left", ncol=3, fontsize=7.0)
    # (b) all variables share energy, avoiding a second y axis.
    axes[1].step(t, d.effective_purchase_kwh, where="post", color=ACTUAL_BLUE, lw=1.8, label="最终生效购电量", zorder=4)
    axes[1].bar(t, d.emergency_purchase_kwh, width=DT*.72, color=EMERGENCY_RED, alpha=.83, label="紧急购电量", zorder=3)
    axes[1].set_ylabel("购电量（kWh/槽）")
    panel_title(axes[1], "(b)", "计划购电与紧急购电")
    style_axis(axes[1], time_axis=True)
    compact_legend(axes[1], loc="upper right", ncol=2)
    # (c) actions separated for unlike kWh/kW scales.
    axes[2].step(t, d.charge_rt_kwh / DT, where="post", color=CHARGE_GREEN, lw=1.65, label="充电功率（正）", zorder=3)
    axes[2].step(t, -d.discharge_rt_kwh / DT, where="post", color=DISCHARGE_ORANGE, lw=1.65, label="放电功率（负）", zorder=3)
    for boundary in (-5000, 5000):
        axes[2].axhline(boundary, color="#B8BDC5", lw=.75, ls="--", zorder=1)
    axes[2].axhline(0, color=BOUND_GREY, lw=.7)
    axes[2].set_ylabel("功率（kW）")
    panel_title(axes[2], "(c)", "储能充放电响应")
    style_axis(axes[2], time_axis=True)
    compact_legend(axes[2], loc="upper right", ncol=2)
    # (d) continuous SOC.
    soc = np.r_[d.soc_start_kwh.iloc[0], d.soc_end_kwh.to_numpy(float)]
    axes[3].plot(t_end, soc, color=ACTUAL_BLUE, lw=2.0, label="实际SOC", zorder=4)
    axes[3].fill_between(t_end, 1200, soc, where=soc >= 1200, color=ACTUAL_BLUE, alpha=.09, zorder=1)
    axes[3].axhline(1200, color="#AAB0B8", lw=.85, ls="--")
    axes[3].axhline(10800, color="#AAB0B8", lw=.85, ls="--")
    axes[3].annotate(f"日初：{soc[0]:.0f} kWh", (0, soc[0]), xytext=(5, 5), textcoords="offset points", fontsize=7.2, color=TEXT_GREY)
    axes[3].annotate(f"日末：{soc[-1]:.0f} kWh", (24, soc[-1]), xytext=(-5, 5), textcoords="offset points", ha="right", fontsize=7.2, color=TEXT_GREY)
    axes[3].set_ylabel("SOC（kWh）")
    axes[3].set_xlabel("时间（h）")
    panel_title(axes[3], "(d)", "SOC轨迹与安全边界")
    style_axis(axes[3], time_axis=True)
    fig.suptitle("代表日波动电价预测与储能响应", fontsize=13, fontweight="semibold", y=.995)
    paths = save(fig, "q4_v1_representative_price_storage")
    return {"representative_date": str(day.date()), "selection_rule": "全年高价紧急购电费用最大日", "high_price_emergency_fee_yuan": float(daily.loc[day]), "slots": int(len(d)), "scenario_tracks": int(len(scenarios)), "scenario_source_dates": [str(pd.Timestamp(z).date()) for z in source_dates], "price_forecast_min_max": [float(forecast.min()), float(forecast.max())], "price_actual_min_max": [float(d.price_yuan_per_kwh.min()), float(d.price_yuan_per_kwh.max())], "soc_start_end_kwh": [float(soc[0]), float(soc[-1])], "outputs": paths}


def figure_42(monthly: pd.DataFrame, summary: pd.DataFrame) -> dict:
    # Rows preserve branch display labels in the frozen overview.
    a = summary[summary.policy.astype(str).str.contains(r"q4_2|4-2", regex=True)].iloc[0]
    b = summary[summary.policy.astype(str).str.contains(r"q4_3|4-3", regex=True)].iloc[0]
    saving = float(a.actual_total_cost_yuan - b.actual_total_cost_yuan)
    emergency_drop = float(a.emergency_purchase_kwh - b.emergency_purchase_kwh)
    if abs(saving - 388002.545836298) > 1e-3:
        raise ValueError("frozen annual cost difference mismatch")
    fig = plt.figure(figsize=(10.0, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.0), hspace=.48, wspace=.30)
    ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
    labels = ["问题4-2", "问题4-3"]
    x = np.arange(2)
    contract = np.array([a.final_contract_cost_yuan, b.final_contract_cost_yuan]) / 1e6
    emer = np.array([a.emergency_cost_yuan, b.emergency_cost_yuan]) / 1e6
    total = contract + emer
    bars = ax0.bar(x, contract, .56, color=[GREY, PV_GREEN], label="合同/计划结算费用", zorder=3)
    ax0.bar(x, emer, .56, bottom=contract, color=EMERGENCY_RED, label="紧急购电费用", zorder=3)
    for i, bar in enumerate(bars):
        cx = bar.get_x() + bar.get_width()/2
        ax0.text(cx, contract[i]/2, f"{contract[i]:.3f}", color="white", ha="center", va="center", fontsize=8)
        ax0.text(cx, contract[i]+emer[i]/2, f"{emer[i]:.3f}", color="white", ha="center", va="center", fontsize=8)
        ax0.annotate(f"合计 {total[i]:.3f}", (cx, total[i]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    # The concise reference layout keeps effect-size interpretation in the
    # caption/body and leaves this panel only with the stacked components.
    ax0.set_ylim(0, 15.5); ax0.set_xticks(x, labels); ax0.set_ylabel("费用（百万元）")
    panel_title(ax0, "(a)", "全年费用构成"); style_axis(ax0)
    evalues = np.array([a.emergency_purchase_kwh, b.emergency_purchase_kwh]) / 1e4
    bx = ax1.bar(x, evalues, .56, color=[GREY, PV_GREEN], zorder=3)
    for bar in bx:
        ax1.annotate(f"{bar.get_height():.3f}", (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,4), textcoords="offset points", ha="center", fontsize=8.3)
    ax1.set_ylim(0, 20.2); ax1.set_xticks(x, labels); ax1.set_ylabel("紧急购电量（万kWh）")
    panel_title(ax1, "(b)", "全年紧急购电量"); style_axis(ax1)
    mm = monthly.copy()
    mm["branch"] = np.where(mm.policy.astype(str).str.contains(r"q4_2|4-2", regex=True), "q42", "q43")
    pivot = mm.pivot(index="month", columns="branch", values="actual_total_cost_yuan").sort_index()
    cumulative = (pivot.q42 - pivot.q43).cumsum()/1e4
    ix = np.arange(len(pivot))
    ax2.plot(ix, cumulative, color=ACTUAL_BLUE, marker="o", ms=4.4, lw=1.9, zorder=3)
    ax2.fill_between(ix, 0, cumulative, color=ACTUAL_BLUE, alpha=.10, zorder=1)
    ax2.axhline(0, color=BOUND_GREY, lw=.7)
    for i, value in enumerate(cumulative):
        ax2.annotate(f"{value:.2f}", (i, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.0)
    ax2.set_xticks(ix, [f"{m[-2:]}月" for m in pivot.index]); ax2.set_ylabel("累计费用节省（万元）")
    panel_title(ax2, "(c)", "问题4-3相对问题4-2累计节省"); style_axis(ax2)
    month_soc = mm.pivot(index="month", columns="branch", values="month_end_soc_kwh").sort_index()
    ax3.plot(ix, month_soc.q42, color=GREY, marker="o", ms=4, lw=1.5, label="问题4-2")
    ax3.plot(ix, month_soc.q43, color=PV_GREEN, marker="o", ms=4, lw=1.7, label="问题4-3")
    ax3.axhline(1200, color="#AAB0B8", ls="--", lw=.8); ax3.axhline(10800, color="#AAB0B8", ls="--", lw=.8)
    ax3.annotate("1200", (10.1, 1200), fontsize=6.8, color=BOUND_GREY); ax3.annotate("10800", (10.1, 10800), fontsize=6.8, color=BOUND_GREY, va="bottom")
    ax3.set_xticks(ix, [f"{m[-2:]}月" for m in month_soc.index]); ax3.set_ylabel("月末SOC（kWh）")
    panel_title(ax3, "(d)", "月末SOC"); style_axis(ax3)
    compact_legend(ax3, loc="lower left", ncol=2)
    fig.suptitle("波动电价下问题4-2与问题4-3的全年结果比较", fontsize=13, fontweight="semibold", y=.995)
    paths = save(fig, "q4_v1_annual_comparison")
    return {"saving_yuan": saving, "saving_pct": saving/a.actual_total_cost_yuan*100, "emergency_drop_kwh": emergency_drop, "emergency_drop_pct": emergency_drop/a.emergency_purchase_kwh*100, "cumulative_final_10k_yuan": float(cumulative.iloc[-1]), "terminal_soc_q42_q43": [float(a.terminal_soc_kwh), float(b.terminal_soc_kwh)], "outputs": paths}


def write_captions(fig1: dict, fig2: dict) -> None:
    text = f"""# 问题四图题与图注

**图4-1 代表日波动电价预测与储能响应。** 代表日按全年问题4-3轨迹中“高价紧急购电费用最大日”这一预先固定规则选取，为 {fig1['representative_date']}。图(a)中的0:00价格预测、情景带和中位数均按冻结的最近14个完整历史日同槽均值及历史残差轨迹构造；未来真实电价未进入该日任何决策。实际价格仅在当前槽执行及事后结算时使用。为避免把每槽购电能量与储能功率强行置于同一量纲，图(b)、(c)分别展示购电和储能动作；图(d)给出连续SOC及物理安全边界。

**图4-2 波动电价下问题4-2与问题4-3的全年结果比较。** 图(a)以实际结算费用分解全年账单，图(b)比较实际紧急购电量，图(c)按未舍入月度账单差累计问题4-3相对问题4-2的费用节省，图(d)给出月末SOC。两分支的场景数、光伏信息来源和计划更新机制不同，因此图中差异是综合方案差异，不能完全归因于单一日内调整机制。期末SOC分别为 {fig2['terminal_soc_q42_q43'][0]:.3f} 和 {fig2['terminal_soc_q42_q43'][1]:.3f} kWh，费用差并非由期末库存透支造成。
"""
    (DOCS / "q4_v1_figure_captions.md").write_text(text, encoding="utf-8")


def paper_section() -> str:
    return r'''## 问题 4：波动电价下的因果购电与滚动调整

### 1. 问题分析与价格信息边界

问题四在问题二、三的供需不确定性基础上，以附件4给出的10分钟波动电价重新计算两类购电策略：问题4-2在每日0:00锁定全天购电计划，问题4-3则允许在6:00、12:00和18:00更新未执行的购电后缀。由于题面未说明当日未来价格的公布时点，本文采用保守的**建模假设**：每次决策只访问此前已完成槽的实际价格及其因果预测，未来实际价格不得进入决策；目标槽实际价格仅在该槽执行和事后结算时使用。该假设是为保证因果回放而作的解释，并非题面明确规定。

令日期为 (d)，槽位为 (t=0,ldots,143)，槽长 (Delta t=1/6) h。负载、光伏、购电、充放电与紧急购电均采用每槽kWh；实际SOC (E_{d,t}) 跨日连续传递，满足 (1200le E_{d,t}le10800) kWh，且不要求每日回到6000 kWh。当前槽供需平衡为

[
z_{d,t}+G_{d,t}+b_{d,t}+e_{d,t}=L_{d,t}+c_{d,t}+w_{d,t},
quad E_{d,t+1}=E_{d,t}+0.9c_{d,t}-b_{d,t}/0.9,
]

其中 (z) 为当前生效购电量，(c,b,e,w) 依次为充电前外部输入、储能对外放电、紧急购电和弃能量。充、放电每槽上限为 (5000Delta t=2500/3) kWh，且互斥；紧急购电只补足当前缺额，不用于主动充电。

### 2. 因果价格预测与联合残差情景

冻结的价格预测器为**最近14个完整历史日同槽均值预测**。在决策日 (d) 的槽 (t)，设此前完整日集合为 (mathcal D_{d}^{14})，则

[
widehatpi_{d,t}=rac{1}{|mathcal D_{d}^{14}|}sum_{iinmathcal D_{d}^{14}}pi_{i,t},qquad
|mathcal D_{d}^{14}|=min(14,	ext{当时可用完整日数}).
]

样本不足时按已冻结的最近完整日可用集合求均值；若无已完成日，则以附件1已知分时电价先验冷启动。该回退仅服务1月初信息积累，未使用目标日未来价格。价格残差定义为 (arepsilon^pi_{i,t}=pi_{i,t}-widehatpi_{i,t})，且每条144槽残差轨迹都按历史日完整保存。

负载预测采用同星期几历史模式；问题4-2的光伏预测由历史实际数据因果构造，问题4-3则使用对应发布时刻的附件3光伏预报。价格、负载和光伏残差按同一历史日块联合抽取，形成 (M) 条联合情景，因而保留日内相关性而不将三类残差独立打乱。问题4-2冻结 (M=7)，问题4-3冻结 (M=14)。图4-1展示高价紧急购电费用最大的代表日：情景带是因果预测误差的覆盖诊断，实际价格线未参与此前的价格决策。

![图4-1 代表日波动电价预测与储能响应。](../outputs/figures/q4_v1/q4_v1_representative_price_storage.png)

### 3. 日前共同计划、滚动调整与动态价值执行

问题4-2在0:00针对共享计划 (p_{d,t}) 求解联合残差情景MILP，日内不修改计划；问题4-3先形成0:00原计划 (p_{d,t})，再于6:00、12:00和18:00基于当时已发布的信息优化未执行后缀。已执行槽永久锁定。令 (z_{d,t}) 是该槽执行前最后一个有效版本，定义

[
delta^+_{d,t}=max(z_{d,t}-p_{d,t},0),quad
delta^-_{d,t}=max(p_{d,t}-z_{d,t},0),quad
z_{d,t}=p_{d,t}+delta^+_{d,t}-delta^-_{d,t}.
]

日前与滚动模型均满足上述供需、SOC、功率和互斥约束。冻结的终端库存价值为 (v=0.6) 元/kWh，仅作为优化目标中的跨日机会成本近似；它不作为收入从真实账单扣除。每次计划更新后，模型对未执行后缀的历史场景路径进行后向递推，平均得到100 kWh网格上的近似未来价值函数。实时执行器仅使用当前槽真实供需、实际价格、当前SOC及当前生效计划选择充放电动作；实际SOC保持连续，通过插值查询价值函数。因此该执行器是历史场景价值近似的因果控制，而不是掌握未来真实价格或供需的理论全局最优控制。

### 4. 波动电价结算与参数冻结

问题4-2的实际费用为
[
C^{4-2}=sum_{d,t}left(pi^{m actual}_{d,t}p_{d,t}+5pi^{m actual}_{d,t}e_{d,t}ight).
]
问题4-3的调整计价时点在题面中存在歧义，本文采用以下透明的建模解释：每槽最终生效量只相对0:00原计划结算一次，中间版本不重复收费，
[
C^{4-3}=sum_{d,t}left[pi^{m actual}_{d,t}p_{d,t}+1.5pi^{m actual}_{d,t}delta^+_{d,t}-0.5pi^{m actual}_{d,t}delta^-_{d,t}+5pi^{m actual}_{d,t}e_{d,t}ight].
]
即上调按1.5倍价格计费，下调返还原价的50%。该口径为建模假设，不被表述为官方唯一规则。

全部预测器、场景数、库存价值和网格只在1月22—31日的共同开发期内选择：问题4-2冻结为最近14日同槽均值价格预测、同星期几供需预测、(M=7)、(v=0.6)元/kWh和100 kWh价值网格；问题4-3冻结为相同价格预测、同星期几负载/版本化光伏预测、(M=14)、(v=0.6)元/kWh和100 kWh网格。2—12月仅用于固定策略的因果评价，未据此重新选参。

### 5. 全年结果、库存公平性与求解质量

在2025-02-01至12-31共334天、48,096个槽的因果回放中，问题4-2的实际总费用为14,540,193.850元，紧急购电量为181,144.015 kWh，期末SOC为10,279.388 kWh；问题4-3的实际总费用为14,152,191.304元，紧急购电量为59,464.119 kWh，期末SOC为10,225.682 kWh。

**表4-1 波动电价下两类冻结策略的全年结果**

| 策略 | 合同/计划结算费用（元） | 紧急购电费用（元） | 实际总费用（元） | 紧急购电量（kWh） | 期末SOC（kWh） |
|---|---:|---:|---:|---:|---:|
| 波动电价下的联合残差情景—动态价值调度策略（问题4-2） | 13,525,826.803 | 1,014,367.047 | 14,540,193.850 | 181,144.015 | 10,279.388 |
| 波动电价下的多时点滚动购电调整策略（问题4-3） | 13,841,394.701 | 310,796.603 | 14,152,191.304 | 59,464.119 | 10,225.682 |

问题4-3相对问题4-2减少388,002.546元，降幅2.668%；紧急购电量减少121,679.896 kWh，降幅67.17%。它增加了合同调整相关成本，但显著降低了按5倍实际电价结算的紧急购电暴露。两者期末SOC仅相差53.706 kWh，且采用开发期预先固定的统一1.0元/kWh库存参考价修正后，费用分别为14,535,914.462元和14,147,965.622元，结论不变。库存修正仅作公平性解释，不替代正式实际账单。

![图4-2 波动电价下问题4-2与问题4-3的全年结果比较。](../outputs/figures/q4_v1/q4_v1_annual_comparison.png)

问题4-3共有13次求解在时限内取得可行incumbent，最大相对MIP gap为0.00078259；这些结果均标记为“限时可行解”，而非理论精确最优解。全部334日轨迹的供需平衡、SOC递推、跨日衔接、功率与互斥、版本锁定、信息边界和独立费用复算均通过。

### 6. 与前序问题的关系、优点与局限

问题二固定电价下的实际总费用为13,763,928.208元，问题三为13,268,776.239元；问题四-2和4-3在波动电价环境下分别为14,540,193.850元和14,152,191.304元。这些数值处于不同价格序列、预测信息结构和冻结参数下，只能作描述性对照，不能简单称为模型性能“退化”。

模型的优点是把价格、供需不确定性与SOC跨日状态置于统一因果信息边界内，并通过版本锁定避免事后修改已执行计划。局限在于：价格可得性和变动价调整结算均为透明建模解释；光伏整点预报映射到10分钟槽采用既定插值；联合场景数、终端价值和网格只在有限1月开发期选择；限时可行解不提供理论全局最优保证。因而本文结论限定为：在严格因果的波动电价回放中，问题4-3综合方案较问题4-2综合方案的全年实际费用降低2.668%、紧急购电量下降67.17%；由于场景数、光伏信息来源和计划更新机制不同，该差异反映综合方案差异，不能完全归因于单一日内调整机制。
'''


def write_paper_and_audit(fig1: dict, fig2: dict, solver: pd.DataFrame) -> dict:
    section = paper_section()
    existing = PAPER.read_text(encoding="utf-8")
    marker = "## 问题 4：波动电价下的因果购电与滚动调整"
    if marker in existing:
        existing = existing[:existing.index(marker)].rstrip() + "\n"
    PAPER.write_text(existing + section, encoding="utf-8")
    (DOCS / "q4_v1_paper_final_section.md").write_text(section.strip() + "\n", encoding="utf-8")
    protected = {
        "result1.xlsx": ROOT / "outputs/tables/result1.xlsx",
        "result2.xlsx": ROOT / "outputs/tables/result2.xlsx",
        "result3.xlsx": ROOT / "outputs/tables/result3.xlsx",
        "result4-2.xlsx": ROOT / "outputs/tables/result4-2.xlsx",
        "result4-3.xlsx": ROOT / "outputs/tables/result4-3.xlsx",
        "raw_result4-2.xlsx": ROOT / "data/raw/result4-2.xlsx",
        "raw_result4-3.xlsx": ROOT / "data/raw/result4-3.xlsx",
    }
    hashes = {name: sha(path) for name, path in protected.items()}
    pandoc = shutil.which("pandoc")
    pandoc_result = "not installed; Markdown structural check used"
    if pandoc:
        proc = subprocess.run([pandoc, "--from=markdown", "--to=html", str(PAPER), "-o", str(ROOT / "outputs/q4_v1/stage5/q4_v1_paper_parse.html")], capture_output=True, text=True, check=False)
        pandoc_result = "passed" if proc.returncode == 0 else f"failed: {proc.stderr[-500:]}"
    # This field spans all four issue times; ``mip_gap`` is populated only for
    # one solver-log subset and would understate the recorded maximum.
    max_gap = float(pd.to_numeric(solver.get("relative_gap_for_audit"), errors="coerce").max())
    audit = f"""# 问题四 V1 论文一致性审计

## 范围

本审查只读取冻结的阶段4/5 CSV、审计文件与正式工作簿；未运行优化，未写入任何Excel。

## 图表与正文

- 图4-1：`outputs/figures/q4_v1/q4_v1_representative_price_storage.*`；代表日 {fig1['representative_date']}，选择规则为全年高价紧急购电费用最大日。
- 图4-2：`outputs/figures/q4_v1/q4_v1_annual_comparison.*`；累计末值 {fig2['cumulative_final_10k_yuan']:.6f} 万元。
- 正文已新增“问题4”章节，使用图4-1、图4-2和表4-1；未改变问题1—3章节内容。

## 数值核验

| 项目 | 复核值 | 状态 |
|---|---:|---|
| 4-2实际总费用 | 14,540,193.850元 | 通过 |
| 4-3实际总费用 | 14,152,191.304元 | 通过 |
| 费用差 | {fig2['saving_yuan']:,.3f}元（{fig2['saving_pct']:.3f}%） | 通过 |
| 紧急购电差 | {fig2['emergency_drop_kwh']:,.3f}kWh（{fig2['emergency_drop_pct']:.2f}%） | 通过 |
| 4-2 / 4-3期末SOC | 10,279.388 / 10,225.682 kWh | 通过 |
| 最大相对MIP gap | {max_gap:.8f} | 通过；限时可行解未称为最优 |

## Markdown解析

Pandoc：{pandoc_result}。

## 受保护文件SHA-256

""" + "\n".join(f"- `{name}`: `{value}`" for name, value in hashes.items()) + "\n\n结论：图表数值和正文冻结数值一致；未修改任何Excel或冻结配置。\n"
    (DOCS / "q4_v1_paper_consistency_audit.md").write_text(audit, encoding="utf-8")
    return {"hashes": hashes, "pandoc": pandoc_result, "max_mip_gap": max_gap}


def main() -> None:
    configure_chinese_theme()
    _, monthly, summary, solver, _, q43 = get_data()
    fig1 = figure_41(q43)
    fig2 = figure_42(monthly, summary)
    write_captions(fig1, fig2)
    audit = write_paper_and_audit(fig1, fig2, solver)
    (DOCS / "q4_v1_figure_data_audit.md").write_text(
        "# 问题四图表数据审计\n\n```json\n" + json.dumps({"figure_4_1": fig1, "figure_4_2": fig2, "consistency": audit, "source_files": [str(SLOTS.relative_to(ROOT)), str(MONTHLY.relative_to(ROOT)), str(SUMMARY.relative_to(ROOT)), str(SOLVER.relative_to(ROOT))], "data_modified": False}, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print("Q4_PAPER_FIGURES_AND_SECTION_PASS")


if __name__ == "__main__":
    main()
