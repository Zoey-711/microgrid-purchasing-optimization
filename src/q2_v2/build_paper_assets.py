"""Build paper-only Question-2 V2 figures, tables, and Markdown section.

Reads frozen records only; it never runs a solver or changes a workbook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.paper_figure_theme import configure_chinese_theme


ROOT = Path(__file__).resolve().parents[2]
STAGE3 = ROOT / "outputs/q2_v2/stage3"
STAGE4 = ROOT / "outputs/q2_v2/stage4"
FIG = ROOT / "outputs/figures/q2_v2"
DOC = ROOT / "docs/q2_v2"
PAPER = ROOT / "docs/论文正文.md"
TABLES = ROOT / "outputs/q2_v2/paper_tables"

BLUE = "#4573B4"
PLAN_BLUE = "#6090C1"
PRED_BLUE = "#6C9FB8"
GREEN = "#66B2A3"
ORANGE = "#E29D39"
RED = "#C44E52"
GREY = "#6B7280"
GRID = "#E8E8E8"
BLACK = "#333333"
FORMAL_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def stylize(ax, panel: str, title: str) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.75)
    ax.tick_params(top=False, right=False, labelsize=8, width=0.65, color=BLACK)
    ax.text(0.01, 0.95, f"({panel}) {title}", transform=ax.transAxes, va="top", ha="left", fontsize=8.8, fontweight="semibold")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slots = pd.read_csv(STAGE4 / "q2_v2_slot_results.csv")
    slots["date"] = pd.to_datetime(slots["date"])
    main = slots.loc[slots.policy_id.eq("main_dp50")].copy()
    greedy = slots.loc[slots.policy_id.eq("closed_loop_greedy")].copy()
    comp = pd.read_csv(STAGE4 / "q2_v2_final_comparison.csv")
    bands = pd.read_csv(STAGE4 / "q2_v2_emergency_by_price_band.csv")
    return main, greedy, comp, bands


def make_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.1), dpi=600)
    ax.set_axis_off()
    nodes = [
        (0.02, 0.60, 0.18, 0.22, "历史实际数据\n负载、光伏、电价", BLUE),
        (0.27, 0.60, 0.18, 0.22, "因果滚动预测\n与联合残差情景", GREEN),
        (0.52, 0.60, 0.18, 0.22, "日前随机 MILP\n统一计划购电量", PLAN_BLUE),
        (0.77, 0.60, 0.18, 0.22, "动态价值储能执行\n当前观测 + 近似价值", ORANGE),
        (0.39, 0.15, 0.23, 0.20, "实际费用评价\n计划购电费 + 5 倍紧急购电费", RED),
    ]
    for x, y, w, h, text, color in nodes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                             linewidth=0.9, edgecolor=color, facecolor=color, alpha=0.15)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=BLACK)
    for a, b in [((0.20, 0.71), (0.27, 0.71)), ((0.45, 0.71), (0.52, 0.71)), ((0.70, 0.71), (0.77, 0.71)), ((0.86, 0.60), (0.62, 0.35))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color=GREY))
    ax.text(0.02, 0.93, "每天 0:00：仅使用此前完整日的历史信息", fontsize=8.3, color=GREY)
    ax.text(0.77, 0.43, "每个 10 分钟槽：\n不使用未来实际值", fontsize=7.8, color=GREY, ha="center")
    fig.savefig(FIG / "q2_v2_method_flow.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "q2_v2_method_flow.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_results_figure(main: pd.DataFrame, comp: pd.DataFrame, bands: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.1, 10.0), dpi=600, facecolor="white")
    gs = fig.add_gridspec(5, 1, height_ratios=[1.05, 1.05, 1.15, 1.0, 1.1], hspace=0.46)
    ax1, ax2, ax3, ax4, ax5 = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    dp = comp.loc[comp.policy_id.eq("main_dp50")].iloc[0]
    gr = comp.loc[comp.policy_id.eq("closed_loop_greedy")].iloc[0]

    stylize(ax1, "a", "全年费用构成")
    names = ["动态价值\n调度", "贪心\n对照"]
    costs = np.array([[dp.plan_purchase_cost_yuan, dp.emergency_purchase_cost_yuan],
                      [gr.plan_purchase_cost_yuan, gr.emergency_purchase_cost_yuan]]) / 1e6
    x = np.arange(2)
    ax1.bar(x, costs[:, 0], color=PLAN_BLUE, width=0.52, label="计划购电费", zorder=3)
    ax1.bar(x, costs[:, 1], bottom=costs[:, 0], color=RED, width=0.52, label="紧急购电费", zorder=3)
    for i, total in enumerate(costs.sum(axis=1)):
        ax1.text(i, total + 0.07, f"{total:.3f}", ha="center", fontsize=8, color=BLACK)
    ax1.set_xticks(x, names); ax1.set_ylabel("费用（百万元）")
    ax1.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.98, 0.84))

    stylize(ax2, "b", "紧急购电的电价区间分布")
    order = ["低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"]
    bdp = bands.loc[bands.policy_id.eq("main_dp50")].set_index("price_band").loc[order]
    bgr = bands.loc[bands.policy_id.eq("closed_loop_greedy")].set_index("price_band").loc[order]
    xx = np.arange(3)
    ax2.bar(xx - 0.17, bdp.emergency_purchase_kwh / 1000, 0.34, color=GREEN, label="动态价值调度", zorder=3)
    ax2.bar(xx + 0.17, bgr.emergency_purchase_kwh / 1000, 0.34, color=GREY, label="贪心对照", zorder=3)
    ax2.set_xticks(xx, ["低价\n<0.6", "中价\n[0.6,1.0)", "高价\n≥1.0"])
    ax2.set_ylabel("紧急购电量（MWh）")
    ax2.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.98, 0.84))

    day = main.loc[main.date.eq(pd.Timestamp("2025-09-23"))].sort_values("slot").reset_index(drop=True)
    hours = day.slot.to_numpy() / 6
    stylize(ax3, "c", "代表日：固定计划与紧急购电")
    ax3.step(hours, day.planned_purchase_kwh, where="post", color=PLAN_BLUE, lw=1.6, label="计划购电量")
    ax3.fill_between(hours, 0, day.planned_purchase_kwh, step="post", color=PLAN_BLUE, alpha=0.10)
    ax3.bar(hours, day.emergency_purchase_kwh, width=1/7.2, color=RED, alpha=0.88, label="紧急购电量", zorder=4)
    ax3.set_ylabel("每槽电量（kWh）")
    ax3.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.98, 0.84))

    stylize(ax4, "d", "代表日：实际储能功率（充电为正、放电为负）")
    charge_kw = day.charge_kwh / (1 / 6)
    discharge_kw = -day.discharge_kwh / (1 / 6)
    ax4.step(hours, charge_kw, where="post", color=GREEN, lw=1.5, label="充电功率")
    ax4.step(hours, discharge_kw, where="post", color=ORANGE, lw=1.5, label="放电功率")
    ax4.axhline(5000, color=GREY, lw=0.9, ls="--")
    ax4.axhline(-5000, color=GREY, lw=0.9, ls="--")
    ax4.set_ylabel("功率（kW）")
    ax4.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.98, 0.84))

    stylize(ax5, "e", "代表日：实际 SOC 与安全边界")
    state_t = np.r_[hours, 24.0]
    state = np.r_[day.soc_start_kwh.to_numpy(), day.soc_end_kwh.iloc[-1]]
    ax5.plot(state_t, state, color=BLUE, lw=1.7, label="实际 SOC")
    ax5.fill_between(state_t, 1200, state, color=BLUE, alpha=0.08)
    ax5.axhline(1200, color=RED, lw=1.0, ls="--", label="安全下限")
    ax5.axhline(10800, color=GREY, lw=1.0, ls="--", label="安全上限")
    ax5.scatter([0, 24], [state[0], state[-1]], color=BLUE, s=14, zorder=5)
    ax5.set_ylabel("储电量（kWh）")
    ax5.set_xlabel("时间（h）")
    ax5.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(0.98, 0.84))

    for ax in (ax3, ax4, ax5):
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    ax3.tick_params(labelbottom=False); ax4.tick_params(labelbottom=False)
    fig.savefig(FIG / "q2_v2_results_composite.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "q2_v2_results_composite.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_tables(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in FORMAL_DATES:
        x = main.loc[main.date.eq(pd.Timestamp(date))].sort_values("slot")
        rows.append({
            "日期": date,
            "全天计划购电量（kWh）": x.planned_purchase_kwh.sum(),
            "计划购电费（元）": x.plan_cost_yuan.sum(),
            "紧急购电量（kWh）": x.emergency_purchase_kwh.sum(),
            "紧急购电费（元）": x.emergency_cost_yuan.sum(),
            "0:00 SOC（kWh）": x.soc_start_kwh.iloc[0],
            "24:00 SOC（kWh）": x.soc_end_kwh.iloc[-1],
        })
    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "q2_v2_specified_dates_summary.csv", index=False, encoding="utf-8-sig")
    return table


def table_md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---:"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        vals = [str(row.iloc[0])] + [f"{float(v):,.3f}" for v in row.iloc[1:]]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def rewrite_paper(table: pd.DataFrame) -> None:
    before = PAPER.read_text(encoding="utf-8").split("## 问题 2：", 1)[0]
    body = r'''## 问题 2：历史联合残差情景—动态价值调度

### 1. 问题分析与信息边界

问题 2 的电价曲线在每日 0:00 已知，但当天未来 144 个连续 10 分钟槽位的负载与光伏实际值尚未发生。因而，计划购电量必须在 0:00 一次性确定并在日内锁定；实际供需偏差只能通过储能的因果动作和紧急购电消化。本文提出**历史联合残差情景—动态价值调度策略**：日前阶段利用历史负载与光伏联合残差构造不确定净负荷情景，通过随机 MILP 确定统一计划购电量；实时阶段以历史情景后向递推得到的近似未来成本函数为依据，实施动态价值储能调度。

对日期 (d) 的日前决策，只允许使用 (d) 之前已经完整发生的负载、光伏及其因果预测记录、当日分时电价和日初 SOC；不得使用 (d) 日未来任何槽位的实际负载、实际光伏、紧急购电或 SOC。附件 3 的预报及问题 3 的调整机制不用于本问。实际支付费用严格定义为

\[
C^{\rm act}=\sum_d\sum_{t=0}^{143}\pi_t p_{d,t}
+\sum_d\sum_{t=0}^{143}5\pi_t e_{d,t},
\]

其中 (p_{d,t}) 是 0:00 锁定的计划购电量，(e_{d,t}) 仅补足真实缺额；未使用计划购电量不退款，紧急购电不得用于主动充电。

### 2. 历史滚动预测与联合残差情景

将负载与光伏功率均按 (\Delta t=1/6\) h 转为每槽电量。负载点预测优先取过去 35 天内同星期、同槽位观测的均值；样本不足时回退至最近可用同星期样本或最近 7 个完整日。光伏点预测取此前至多 7 个完整日的同槽位均值。以上规则均只取决于决策日之前的历史数据。

对历史日 (i) 的因果回放预测，定义残差

\[
\varepsilon^L_{i,t}=L_{i,t}-\widehat L_{i,t},\qquad
\varepsilon^{PV}_{i,t}=G_{i,t}-\widehat G_{i,t}.
\]

每日 0:00 从此前最多 (M=14) 个完整历史日取整日残差轨迹，而不分别打乱负载和光伏残差。第 (\omega) 条情景为

\[
L^\omega_{d,t}=\max\{0,\widehat L_{d,t}+\varepsilon^L_{\omega,t}\},\qquad
G^\omega_{d,t}=\max\{0,\widehat G_{d,t}+\varepsilon^{PV}_{\omega,t}\}.
\]

这样保留了负载—光伏预测偏差的联合关系及单日内时序相关性。图 3 给出从历史信息到实际执行的完整信息流；图中所有实时动作只使用当前实际观测，不使用未来实际值。

![图 3 历史联合残差情景—动态价值调度策略的信息流与决策流程。](../outputs/figures/q2_v2/q2_v2_method_flow.png)

### 3. 日前随机 MILP

以统一的 (p_{d,t}) 为第一阶段决策；每个情景分别配置充电 (c^\omega_{d,t})、放电 (b^\omega_{d,t})、紧急购电 (e^\omega_{d,t})、弃能 (w^\omega_{d,t}) 和 SOC (E^\omega_{d,t})。日前目标为

\[
\min J_d=\sum_t\pi_t p_{d,t}
+\frac{1}{M_d}\sum_{\omega=1}^{M_d}\sum_t5\pi_t e^\omega_{d,t}
-\frac{v}{M_d}\sum_{\omega=1}^{M_d}E^\omega_{d,144},
\]

其中 (M_d\le14) 是当日真实可得的情景数，终端库存价值 (v=0.4) 元/kWh 仅为跨日机会成本近似，不计入最终账单。由于充、放电模式以二元变量互斥，该日前模型为情景 MILP。每个情景满足

\[
p_{d,t}+G^\omega_{d,t}+b^\omega_{d,t}+e^\omega_{d,t}
=L^\omega_{d,t}+c^\omega_{d,t}+w^\omega_{d,t},
\]

\[
E^\omega_{d,t+1}=E^\omega_{d,t}+0.9c^\omega_{d,t}-\frac{b^\omega_{d,t}}{0.9},
\]

\[
1200\le E^\omega_{d,t}\le10800,quad
0\le c^\omega_{d,t},b^\omega_{d,t}\le\frac{2500}{3}.
\]

### 4. 近似未来价值函数与动态价值储能执行

日前计划锁定后，实时层面对当前真实缺额

\[
r_{d,t}=L_{d,t}-G_{d,t}-p_{d,t}.
\]

对每一条历史情景路径，针对离散 SOC 状态后向计算路径价值 (H^\omega_t(E))，终端条件为 (H^\omega_{144}(E)=-vE)，并以情景平均得到近似未来成本函数 (\bar H_t(E))。实时执行时保持实际 SOC 连续，不把实际状态强行吸附到网格；对候选可行动作插值得到下一状态价值，并选择使“当前紧急购电费用 + (\bar H_{t+1}(E_{t+1}))”最小的动作。正式实现的 SOC 网格为 50 kWh，代码标识中的 `DP50` 仅指这一实现参数，不是策略名称。

实际层始终满足

\[
p_{d,t}+G_{d,t}+b_{d,t}+e_{d,t}
=L_{d,t}+c_{d,t}+w_{d,t},
\qquad E_{d,t+1}=E_{d,t}+0.9c_{d,t}-\frac{b_{d,t}}{0.9}.
\]

当存在富余能量时才允许充电；发生紧急购电时强制 (c_{d,t}=0)。每日末 SOC 作为下一日初 SOC 传递，问题 2 未额外设置每日末 SOC 回到固定值的硬约束。

### 5. 参数选择与执行器消融

参数只在 2025-01-22 至 2025-01-31 的连续共同开发期确定，1 月 1 日至 21 日仅用于因果预热。以开发期的实际费用为评价量，逐步比较点预测基线、历史联合情景、终端价值候选和不同执行器。最终冻结 (M=14)、(v=0.4) 元/kWh 与 50 kWh SOC 网格；采用 HiGHS 单线程 MILP，时限 45 s、相对 gap 请求值 (10^{-4})。开发期选定策略的实际费用为 491,394.433 元，计划购电费为 483,561.858 元，紧急购电费为 7,832.575 元。

为隔离执行器影响，另做“同日计划、同日初 SOC，仅替换执行器”的配对消融。334 个正式日中，动态价值执行的费用比配对贪心低 106,662.322 元，方向与独立跨日闭环比较一致。该结果支持执行器在本数据回放中的作用，但不是对任意场景的理论保证。

### 6. 全年结果及机理分析

在 2025-02-01 至 2025-12-31 的 334 天因果回放中，动态价值调度策略的计划购电费为 12,847,359.131 元，紧急购电费为 916,569.077 元，实际总费用为 **13,763,928.208 元**。完整闭环贪心对照的实际总费用为 13,870,589.590 元，故本策略减少 106,661.382 元，降幅为 0.769%。

| 指标 | 动态价值调度策略 | 贪心对照 |
|---|---:|---:|
| 计划购电费（元） | 12,847,359.131 | 12,847,570.047 |
| 紧急购电费（元） | 916,569.077 | 1,023,019.543 |
| 实际总费用（元） | 13,763,928.208 | 13,870,589.590 |
| 紧急购电量（kWh） | 176,902.132 | 172,161.349 |
| 紧急购电加权平均单价（元/kWh） | 5.181221 | 5.942214 |
| 平均日末 SOC（kWh） | 2,030.403 | 2,029.414 |
| 全年末 SOC（kWh） | 1,404.800 | 1,404.800 |

紧急购电总量并未减少，反而增加 4,740.784 kWh；费用下降来自高价区间紧急购电由 124,578.392 kWh 降至 88,206.452 kWh，同时更多紧急缺额落在低、中价区间，因此加权紧急购电单价下降。两策略全年末 SOC 均为 1,404.800 kWh，且月末 SOC 一致，故费用优势不是期末库存透支造成。图 4 同时展示费用构成、电价区间分布及题面指定代表日的实际调度；2025-09-23 仅作为指定日期中的案例日，不表述为全年最差日。

![图 4 年度费用与电价区间紧急购电分布，以及 2025-09-23 的实际调度。](../outputs/figures/q2_v2/q2_v2_results_composite.png)

### 7. 指定日期结果

官方模板已按行列顺序写入 334 天 × 144 槽的计划购电量、六个四小时段的实际充放电量和日初/日末 SOC，并仅记录实际发生的连续紧急购电事件。表 3 汇总题面指定日期的日级结果；完整指定槽位与四小时区间数据见导出的 `result2.xlsx` 和冻结结果表。

''' + table_md(table) + r'''

其中，充电量、放电量均以各自列中的非负 kWh 幅值记录；在同一四小时汇总段内二者都非零，不表示同一 10 分钟槽同时充放电。

### 8. 模型验证与局限

导出前后对 48,096 个正式槽逐项复核：计划表无空值，日期和官方槽位顺序不变；最大能量平衡残差为 (6.821\times10^{-13}) kWh，SOC 递推最大残差为 (3.638\times10^{-12}) kWh，跨日 SOC 残差为 0。SOC、功率、同时充放电、紧急购电同时充电、计划锁定和信息边界违规数均为 0。

本方法的未来价值来自历史联合残差路径的近似后向计算，不掌握当天未来实际负载或光伏，也不构成理论全局最优控制。终端库存价值仅用于日前决策的跨日机会成本近似，未从实际账单中扣除。全年结果是冻结参数下的因果回放表现，不能被解释为对其他年份、其他误差分布或其他电价机制的普适结论。
'''
    PAPER.write_text(before.rstrip() + "\n\n" + body + "\n", encoding="utf-8")


def captions() -> None:
    (DOC / "q2_v2_figure_captions.md").write_text(
        "# 问题二 V2 图注\n\n"
        "**图 3** 历史联合残差情景—动态价值调度策略的信息流与决策流程。日前阶段只使用此前完整日的历史信息；日内执行仅使用当前槽实际观测和 0:00 构造的近似未来价值。\n\n"
        "**图 4** 年度费用构成、紧急购电电价区间分布及代表日调度。费用统计期为 2025-02-01 至 2025-12-31；代表日为题面指定日期 2025-09-23。紧急购电分布为冻结逐槽轨迹按当槽电价汇总，非概率分布。\n",
        encoding="utf-8",
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    configure_chinese_theme()
    main_slots, _, comparison, bands = load()
    table = make_tables(main_slots)
    make_flowchart()
    make_results_figure(main_slots, comparison, bands)
    rewrite_paper(table)
    captions()


if __name__ == "__main__":
    main()
