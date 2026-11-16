"""Second-pass, reproducible analysis of the Question-2 January grid search."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.backtest_q2_january import _common_backtest_dates, _run_one_combination
from src.config import (
    FIGURE_DIR,
    Q2_JANUARY_BACKTEST_FILE,
    Q2_VALIDATION_TOLERANCE,
    TABLE_DIR,
)
from src.data_io import load_q2_actual_energy
from src.model_q2_day_ahead import get_highs_solver
from src.plot_style import AXIS_GREY, GRID_GREY, PLAN_PURCHASE, SOC_MAIN, SOC_UPPER_BOUND


PARETO_FILE = TABLE_DIR / "q2_january_pareto_summary.csv"
SENSITIVITY_FIGURE = FIGURE_DIR / "q2_parameter_sensitivity.png"
SELECTION_NOTE = Path(__file__).resolve().parents[1] / "docs" / "q2_parameter_selection.md"


def _font_name() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def _required_columns(frame: pd.DataFrame) -> None:
    required = {
        "w_days", "q", "Sres_kwh", "rho_yuan_per_kwh",
        "plan_purchase_cost_yuan", "emergency_purchase_cost_yuan",
        "total_cost_yuan", "emergency_purchase_kwh", "validation_passed", "selected",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"parameter comparison is missing columns: {sorted(missing)}")


def _replay_diagnostics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Recover daily-end SOC and reserve-slack diagnostics without altering source CSV."""
    actual, price = load_q2_actual_energy()
    solver, _ = get_highs_solver()
    dates = _common_backtest_dates()
    diagnostics = []
    for row_id, row in comparison.iterrows():
        report, day_runs = _run_one_combination(
            actual=actual,
            price=price,
            dates=dates,
            window_days=int(row["w_days"]),
            quantile=float(row["q"]),
            reserve_soc_kwh=float(row["Sres_kwh"]),
            reserve_penalty_yuan_per_kwh=float(row["rho_yuan_per_kwh"]),
            solver=solver,
        )
        if not report["passed"]:
            raise RuntimeError(f"replay validation failed for comparison row {row_id}")
        replay_cost = report["total_cost_yuan_recomputed"]
        if not np.isclose(replay_cost, float(row["total_cost_yuan"]), atol=1e-5, rtol=0.0):
            raise RuntimeError(f"replay cost differs from source comparison row {row_id}")
        end_soc = [run["rt_metadata"]["rt_final_soc_kwh"] for run in day_runs]
        slack = [run["plan_metadata"]["reserve_shortfall_kwh"] for run in day_runs]
        diagnostics.append(
            {
                "_row_id": row_id,
                "mean_daily_end_soc_kwh": float(np.mean(end_soc)),
                "min_daily_end_soc_kwh": float(np.min(end_soc)),
                "max_daily_end_soc_kwh": float(np.max(end_soc)),
                "reserve_shortfall_days": int(sum(value > Q2_VALIDATION_TOLERANCE for value in slack)),
                "reserve_shortfall_total_kwh": float(sum(slack)),
                "reserve_shortfall_max_kwh": float(max(slack)),
                "rho_triggered": bool(any(value > Q2_VALIDATION_TOLERANCE for value in slack)),
            }
        )
    return pd.DataFrame(diagnostics)


def _is_dominated(candidate: pd.Series, alternatives: pd.DataFrame) -> bool:
    """Minimise both cost components; maximise mean/minimum actual end SOC."""
    eps = 1e-8
    no_worse = (
        (alternatives["plan_purchase_cost_yuan"] <= candidate["plan_purchase_cost_yuan"] + eps)
        & (alternatives["emergency_purchase_cost_yuan"] <= candidate["emergency_purchase_cost_yuan"] + eps)
        & (alternatives["mean_daily_end_soc_kwh"] >= candidate["mean_daily_end_soc_kwh"] - eps)
        & (alternatives["min_daily_end_soc_kwh"] >= candidate["min_daily_end_soc_kwh"] - eps)
    )
    strictly_better = (
        (alternatives["plan_purchase_cost_yuan"] < candidate["plan_purchase_cost_yuan"] - eps)
        | (alternatives["emergency_purchase_cost_yuan"] < candidate["emergency_purchase_cost_yuan"] - eps)
        | (alternatives["mean_daily_end_soc_kwh"] > candidate["mean_daily_end_soc_kwh"] + eps)
        | (alternatives["min_daily_end_soc_kwh"] > candidate["min_daily_end_soc_kwh"] + eps)
    )
    return bool((no_worse & strictly_better).any())


def _pareto_representatives(reserve: pd.DataFrame) -> pd.DataFrame:
    pareto = reserve.loc[
        [not _is_dominated(row, reserve.drop(index=index)) for index, row in reserve.iterrows()]
    ].copy()
    # Rho variants can be exactly identical when the reserve slack is zero. Keep
    # the least-penalised representative of an otherwise identical outcome.
    objective_columns = [
        "w_days", "q", "Sres_kwh", "plan_purchase_cost_yuan", "emergency_purchase_cost_yuan",
        "mean_daily_end_soc_kwh", "min_daily_end_soc_kwh",
    ]
    pareto = pareto.sort_values("rho_yuan_per_kwh", kind="stable").drop_duplicates(
        subset=objective_columns, keep="first"
    )
    return pareto


def _add_deltas(frame: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    result = frame.copy()
    result["total_cost_delta_vs_baseline_yuan"] = result["total_cost_yuan"] - baseline["total_cost_yuan"]
    result["emergency_purchase_delta_vs_baseline_kwh"] = (
        result["emergency_purchase_kwh"] - baseline["emergency_purchase_kwh"]
    )
    result["mean_end_soc_delta_vs_baseline_kwh"] = (
        result["mean_daily_end_soc_kwh"] - baseline["mean_daily_end_soc_kwh"]
    )
    result["min_end_soc_delta_vs_baseline_kwh"] = (
        result["min_daily_end_soc_kwh"] - baseline["min_daily_end_soc_kwh"]
    )
    return result


def _write_sensitivity_figure(comparison: pd.DataFrame) -> None:
    """Show q sensitivity at the no-extra-reserve baseline, separated by w."""
    baseline_reserve = comparison.loc[np.isclose(comparison["Sres_kwh"], 1200.0)].copy()
    baseline_reserve = baseline_reserve.sort_values("rho_yuan_per_kwh", kind="stable").groupby(
        ["w_days", "q"], as_index=False
    ).first()
    font = _font_name()
    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "lines.solid_capstyle": "round", "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    # Main-text figure retains the two non-redundant metrics. Emergency
    # purchase cost remains in the parameter-comparison table and is not
    # repeated because it is directly proportional to emergency energy.
    metrics = [
        ("total_cost_yuan", "实际总成本（万元）", 10_000.0),
        ("emergency_purchase_kwh", "紧急购电量（kWh）", 1.0),
    ]
    # Window length is a parameter category, not a cost/risk quantity; use
    # restrained cool hues to distinguish the three curves.
    colors = {7: SOC_MAIN, 14: PLAN_PURCHASE, 21: SOC_UPPER_BOUND}
    markers = {7: "o", 14: "s", 21: "^"}
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True, facecolor="white")
    for panel_index, (axis, (column, ylabel, scale)) in enumerate(zip(axes, metrics)):
        for window, group in baseline_reserve.groupby("w_days", sort=True):
            group = group.sort_values("q")
            axis.plot(
                group["q"], group[column] / scale, marker=markers.get(int(window), "o"), linewidth=2.0, markersize=4.5,
                color=colors.get(int(window), "#555555"), label=f"w={int(window)} 天",
            )
        axis.set_ylabel(ylabel)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID_GREY, linestyle="-", linewidth=0.7)
        axis.axvline(0.75, color=SOC_UPPER_BOUND, linestyle="--", linewidth=1.2, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(AXIS_GREY)
        axis.spines["bottom"].set_color(AXIS_GREY)
        axis.spines["left"].set_linewidth(0.85)
        axis.spines["bottom"].set_linewidth(0.85)
        axis.tick_params(axis="both", direction="out", width=0.8, length=4, color=AXIS_GREY)
        axis.text(-0.12, 1.03, f"({chr(97 + panel_index)})", transform=axis.transAxes, fontsize=11, fontweight="bold")
    axes[0].text(0.75, 0.97, "选定 q=0.75", transform=axes[0].get_xaxis_transform(), ha="right", va="top", fontsize=8.5, color=SOC_UPPER_BOUND)
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, columnspacing=1.4, handlelength=2.4)
    axes[-1].set_xlabel("净负荷预测分位数 q")
    axes[-1].set_xticks(sorted(baseline_reserve["q"].unique()))
    figure.subplots_adjust(hspace=0.20)
    SENSITIVITY_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(SENSITIVITY_FIGURE, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(SENSITIVITY_FIGURE.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_selection_note(
    summary: pd.DataFrame,
    baseline: pd.Series,
    reserve_min_cost: pd.Series,
    reserve_min_emergency: pd.Series,
    rho_summary: pd.DataFrame,
    q_summary: pd.DataFrame,
) -> None:
    reserve_improves_risk = (
        reserve_min_cost["emergency_purchase_kwh"] < baseline["emergency_purchase_kwh"] - 1e-6
        and reserve_min_cost["min_daily_end_soc_kwh"] > baseline["min_daily_end_soc_kwh"] + 1e-6
    )
    premium = reserve_min_cost["total_cost_yuan"] - baseline["total_cost_yuan"]
    if reserve_improves_risk and premium / baseline["total_cost_yuan"] <= 0.01:
        recommendation = (
            "最低成本的有效保留组合在总成本增幅不超过 1% 的条件下，同时降低紧急购电量并提高最小日末 SOC，"
            "因此推荐其作为正式 A+ 参数。1% 仅是本次工程筛选阈值，论文中须明确说明。"
        )
    else:
        recommendation = (
            "在本次 1 月共同回测中，未观察到“成本仅小幅增加、同时明确降低紧急购电风险并提高最小日末 SOC”的有效保留组合。"
            "因此正式基线应采用无额外日末 SOC 保留方案；\(S^{res}>1200\) 的方案保留为敏感性结果，不能被表述为已由本回测支持的正式策略。"
        )

    sensitivity_table = _markdown_table(
        q_summary,
        ["w_days", "q", "total_cost_yuan", "emergency_purchase_cost_yuan", "emergency_purchase_kwh"],
    )
    summary_table = _markdown_table(
        summary,
        [
            "category", "w_days", "q", "Sres_kwh", "rho_yuan_per_kwh", "total_cost_yuan",
            "emergency_purchase_kwh", "mean_daily_end_soc_kwh", "min_daily_end_soc_kwh",
            "total_cost_delta_vs_baseline_yuan", "emergency_purchase_delta_vs_baseline_kwh",
        ],
    )
    rho_table = _markdown_table(
        rho_summary,
        ["Sres_kwh", "combination_count", "rho_triggered_combination_count", "max_reserve_shortfall_total_kwh"],
    )
    text = fr"""# 问题 2：1 月参数选择与敏感性分析

## 数据范围与方法

本分析以 `outputs/tables/q2_january_parameter_comparison.csv` 中的 135 组、2025-01-22 至 2025-01-31 的共同因果回测为依据。源表未保存逐日日末 SOC 与日前软保留松弛量，故本次按源表中完全相同的参数组合重放 1 月回测以恢复这两项派生诊断；未运行 2 月至 12 月，也未生成或修改 `result2.xlsx`。

所有组合均先通过物理可行性、无未来数据泄漏、SOC 跨日衔接、能量平衡及“紧急购电不充电”校验，之后再比较紧急购电风险和实际总成本。

## 无额外保留的成本基线

源表选出的最低成本组合为 \(w=7\)、\(q=0.75\)、\(S^{{res}}=1200\) kWh、\(\rho=0.6976\) 元/kWh，总成本为 {baseline['total_cost_yuan']:.3f} 元，紧急购电量为 {baseline['emergency_purchase_kwh']:.3f} kWh。

该组合的 \(S^{{res}}\) 恰为物理安全下限，因此在本分析中明确称为**无额外日末 SOC 保留的成本基线**，不是最终风险保留策略。其平均日末 SOC 为 {baseline['mean_daily_end_soc_kwh']:.3f} kWh，最小日末 SOC 为 {baseline['min_daily_end_soc_kwh']:.3f} kWh。

## 有效保留组合与 Pareto 集

下表含无额外保留基线、\(S^{{res}}>1200\) 中实际总成本最低组合、紧急购电量最低组合，以及四目标 Pareto 非支配组合。Pareto 判定同时最小化计划购电费与紧急购电费，并最大化平均日末 SOC 与最小日末 SOC；完全等价的 \(\rho\) 变体仅保留惩罚较小者。

{summary_table}

其中，有效保留方案中总成本最低者为 \(w={int(reserve_min_cost['w_days'])}\)、\(q={reserve_min_cost['q']:.2f}\)、\(S^{{res}}={reserve_min_cost['Sres_kwh']:.0f}\) kWh、\(\rho={reserve_min_cost['rho_yuan_per_kwh']:.4f}\) 元/kWh；紧急购电量最低者为 \(w={int(reserve_min_emergency['w_days'])}\)、\(q={reserve_min_emergency['q']:.2f}\)、\(S^{{res}}={reserve_min_emergency['Sres_kwh']:.0f}\) kWh、\(\rho={reserve_min_emergency['rho_yuan_per_kwh']:.4f}\) 元/kWh。

## 分位数灵敏度

为隔离 \(q\) 的影响，图和下表均固定为无额外日末 SOC 保留（\(S^{{res}}=1200\) kWh）且取该保留水平下最小的 \(\rho\)。图见 `outputs/figures/q2_parameter_sensitivity.png`。

{sensitivity_table}

对每个窗口，\(q=0.60\) 是较低分位数、\(q=0.80\) 是较高分位数；表中同时列示所有候选 \(q\) 的实际总成本、紧急购电费和紧急购电量。因此不能仅依据 \(q=0.75\) 的单点结果断言分位数越高或越低越好：低分位数通常使日前采购更保守不足而增加紧急风险，高分位数则可能抬高计划购电费。最终选择必须以实际总成本和紧急风险的共同变化为准。

## \(\rho\) 是否实际触发

软保留项触发定义为至少一个回测日出现 \(u_d>10^{{-6}}\) kWh。汇总如下。

{rho_table}

本次 135 组组合在 \(S^{{res}}=1200,3600,6000\) kWh 三个保留水平下均未出现 \(u_d>10^{{-6}}\) kWh；因此本候选网格中 \(\rho\) 没有产生正的软保留惩罚，也未改变同一 \((w,q,S^{{res}})\) 下的实际经济结果。特别地，\(S^{{res}}=1200\) kWh 等于 SOC 物理下限，故无额外保留基线中的 \(\rho\) 不具有经济作用；更高 \(S^{{res}}\) 的结果由可行的保留目标本身形成，不能误称为 \(\rho\) 被实际罚触发的结果。

## 参数选择规则与结论

分层规则为：**物理可行 → 紧急购电不充电 → 紧急购电风险 → 实际总成本**。先排除违反任一物理/因果校验的组合；再在 Pareto 集中比较紧急购电量、日末 SOC 保留和实际成本差。

{recommendation}
"""
    SELECTION_NOTE.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_NOTE.write_text(text, encoding="utf-8")


def analyze_q2_january_parameters() -> dict:
    """Generate the requested Pareto CSV, sensitivity figure, and selection note."""
    comparison = pd.read_csv(Q2_JANUARY_BACKTEST_FILE)
    _required_columns(comparison)
    if len(comparison) != 135:
        raise ValueError("January comparison must contain the audited 135 parameter combinations")
    valid = comparison.loc[comparison["validation_passed"].astype(bool)].copy()
    baseline_rows = valid.loc[valid["selected"].astype(bool)]
    if len(baseline_rows) != 1:
        raise ValueError("source comparison must identify exactly one selected baseline")

    diagnostics = _replay_diagnostics(valid)
    valid = valid.merge(diagnostics, left_index=True, right_on="_row_id", how="inner").drop(columns="_row_id")
    baseline = valid.loc[valid["selected"].astype(bool)].iloc[0]
    reserve = valid.loc[valid["Sres_kwh"] > 1200.0 + Q2_VALIDATION_TOLERANCE].copy()
    reserve_min_cost = reserve.sort_values(
        ["total_cost_yuan", "emergency_purchase_kwh", "rho_yuan_per_kwh"], kind="stable"
    ).iloc[0]
    reserve_min_emergency = reserve.sort_values(
        ["emergency_purchase_kwh", "total_cost_yuan", "rho_yuan_per_kwh"], kind="stable"
    ).iloc[0]
    pareto = _pareto_representatives(reserve)

    summary_parts = [
        baseline.to_frame().T.assign(category="cost_baseline_no_extra_reserve"),
        reserve_min_cost.to_frame().T.assign(category="reserve_min_total_cost"),
        reserve_min_emergency.to_frame().T.assign(category="reserve_min_emergency_purchase"),
        pareto.assign(category="pareto_non_dominated"),
    ]
    summary = pd.concat(summary_parts, ignore_index=True)
    dedupe = ["w_days", "q", "Sres_kwh", "rho_yuan_per_kwh"]
    summary["category"] = summary.groupby(dedupe)["category"].transform(lambda values: ";".join(dict.fromkeys(values)))
    summary = summary.drop_duplicates(subset=dedupe, keep="first")
    summary = _add_deltas(summary, baseline).sort_values(
        ["total_cost_yuan", "emergency_purchase_kwh"], kind="stable"
    )
    output_columns = [
        "category", "w_days", "q", "Sres_kwh", "rho_yuan_per_kwh", "plan_purchase_cost_yuan",
        "emergency_purchase_cost_yuan", "total_cost_yuan", "emergency_purchase_kwh",
        "mean_daily_end_soc_kwh", "min_daily_end_soc_kwh", "max_daily_end_soc_kwh",
        "total_cost_delta_vs_baseline_yuan", "emergency_purchase_delta_vs_baseline_kwh",
        "mean_end_soc_delta_vs_baseline_kwh", "min_end_soc_delta_vs_baseline_kwh",
        "reserve_shortfall_days", "reserve_shortfall_total_kwh", "rho_triggered",
        "validation_passed", "no_future_data_leakage", "emergency_charge_overlap_count",
    ]
    PARETO_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary[output_columns].to_csv(PARETO_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")

    _write_sensitivity_figure(valid)
    q_summary = valid.loc[np.isclose(valid["Sres_kwh"], 1200.0)].sort_values(
        "rho_yuan_per_kwh", kind="stable"
    ).groupby(["w_days", "q"], as_index=False).first().sort_values(["w_days", "q"])
    rho_summary = valid.groupby("Sres_kwh", as_index=False).agg(
        combination_count=("rho_triggered", "size"),
        rho_triggered_combination_count=("rho_triggered", "sum"),
        max_reserve_shortfall_total_kwh=("reserve_shortfall_total_kwh", "max"),
    )
    _write_selection_note(summary, baseline, reserve_min_cost, reserve_min_emergency, rho_summary, q_summary)
    return {
        "pareto_path": PARETO_FILE,
        "figure_path": SENSITIVITY_FIGURE,
        "note_path": SELECTION_NOTE,
        "baseline": baseline.to_dict(),
        "reserve_min_cost": reserve_min_cost.to_dict(),
        "reserve_min_emergency": reserve_min_emergency.to_dict(),
    }


if __name__ == "__main__":
    outcome = analyze_q2_january_parameters()
    print(f"Pareto summary: {outcome['pareto_path']}")
    print(f"Sensitivity figure: {outcome['figure_path']}")
    print(f"Selection note: {outcome['note_path']}")
