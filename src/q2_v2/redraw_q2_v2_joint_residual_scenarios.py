"""Redraw frozen Q2 V2 historical joint-residual scenario figure only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paper_figure_theme import (
    ACTUAL_BLUE,
    BOUND_GREY,
    FORECAST_BLUE,
    GRID_GREY,
    PLAN_BLUE,
    PV_GREEN,
    compact_legend,
    configure_chinese_theme,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE4 = ROOT / "outputs" / "q2_v2" / "stage4"
FIG_DIR = ROOT / "outputs" / "figures" / "q2_v2"
LOG_FILE = ROOT / "outputs" / "q2_v2" / "q2_v2_joint_residual_scenarios_redraw_check.txt"
DATE = "2025-06-01"
DT = 1.0 / 6.0
RESIDUAL_BLUE_GREY = "#AEBFCC"
OUTER_BAND = "#C9D9E5"
INNER_BAND = "#8FC6C1"


def top_legend(axis: plt.Axes, handles, labels, ncol: int) -> None:
    compact_legend(
        axis,
        handles=handles,
        labels=labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        ncol=ncol,
        fontsize=7.1,
        borderaxespad=0.0,
    )


def main() -> None:
    font, _ = configure_chinese_theme()
    slots = pd.read_csv(STAGE4 / "q2_v2_slot_results.csv")
    plans = pd.read_csv(STAGE4 / "q2_v2_day_ahead_plans.csv")
    slots["date"] = slots["date"].astype(str)
    plans["date"] = plans["date"].astype(str)
    info = pd.read_csv(STAGE4 / "main_dp50" / "information" / f"{DATE}.csv")
    scenario_dates = info.loc[0, "scenario_dates"].split("|")
    if len(scenario_dates) != 14:
        raise ValueError("The frozen information record must contain exactly 14 scenario days.")

    case_plan = plans.loc[(plans["policy_id"].eq("main_dp50")) & (plans["date"].eq(DATE))].sort_values("slot")
    case_actual = slots.loc[(slots["policy_id"].eq("main_dp50")) & (slots["date"].eq(DATE))].sort_values("slot")
    historic_slots = slots.loc[(slots["policy_id"].eq("main_dp50")) & (slots["date"].isin(scenario_dates))].copy()
    historic_plans = plans.loc[(plans["policy_id"].eq("main_dp50")) & (plans["date"].isin(scenario_dates))].copy()
    history = historic_slots.merge(
        historic_plans[["date", "slot", "load_forecast_kwh", "pv_forecast_kwh"]],
        on=["date", "slot"],
        validate="one_to_one",
    )
    if len(case_plan) != 144 or len(case_actual) != 144 or len(history) != 14 * 144:
        raise ValueError("Frozen scenario source tables are incomplete.")

    pred_net = case_plan["load_forecast_kwh"].to_numpy(float) - case_plan["pv_forecast_kwh"].to_numpy(float)
    residual_paths = []
    for scenario_date in scenario_dates:
        hist = history.loc[history["date"].eq(scenario_date)].sort_values("slot")
        residual = (
            hist["load_actual_kwh"].to_numpy(float)
            - hist["load_forecast_kwh"].to_numpy(float)
            - hist["pv_actual_kwh"].to_numpy(float)
            + hist["pv_forecast_kwh"].to_numpy(float)
        )
        residual_paths.append(residual)
    residuals = np.asarray(residual_paths)
    scenarios = pred_net + residuals
    actual_net = case_actual["load_actual_kwh"].to_numpy(float) - case_actual["pv_actual_kwh"].to_numpy(float)
    q10, q25, q50, q75, q90 = np.quantile(scenarios, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    time = np.arange(144, dtype=float) * DT

    fig, axes = plt.subplots(2, 1, figsize=(7.25, 4.75), sharex=True, constrained_layout=True)

    ax = axes[0]
    line_handle = None
    for path in residuals:
        line_handle = ax.step(time, path, where="post", color=RESIDUAL_BLUE_GREY, alpha=0.35, linewidth=0.75, zorder=2)[0]
    zero_handle = ax.axhline(0, color=BOUND_GREY, linestyle="--", linewidth=0.95, zorder=1)
    ax.set_ylabel("净负荷残差（kWh）")
    panel_title(ax, "(a)", "历史联合残差轨迹")
    style_axis(ax, time_axis=True)
    top_legend(ax, [line_handle, zero_handle], ["14条历史联合残差轨迹", "零残差线"], ncol=2)

    ax = axes[1]
    outer = ax.fill_between(time, q10, q90, step="post", color=OUTER_BAND, alpha=0.34, label="10%–90% 情景带", zorder=1)
    inner = ax.fill_between(time, q25, q75, step="post", color=INNER_BAND, alpha=0.42, label="25%–75% 情景带", zorder=2)
    # The point forecast is numerically close to the scenario median.  Draw it
    # above the median with a darker cyan dashed stroke, while retaining the
    # actual trace as the visual top layer.  This is presentation-only.
    h_pred = ax.step(time, pred_net, where="post", color="#3F8FA3", linestyle=(0, (4, 2.2)), linewidth=1.85, label="日前点预测", zorder=5)[0]
    h_median = ax.step(time, q50, where="post", color=BOUND_GREY, linewidth=1.3, label="情景中位数", zorder=4)[0]
    h_actual = ax.step(time, actual_net, where="post", color=ACTUAL_BLUE, linewidth=2.05, label="实际净负荷", zorder=6)[0]
    zero_handle = ax.axhline(0, color=BOUND_GREY, linestyle="--", linewidth=0.9, zorder=0)
    ax.set_ylabel("每槽净负荷（kWh）")
    ax.set_xlabel("时间（h）")
    panel_title(ax, "(b)", "代表日净负荷情景带与实际轨迹")
    style_axis(ax, time_axis=True)
    top_legend(
        ax,
        [h_pred, h_median, outer, inner, h_actual, zero_handle],
        ["日前点预测", "情景中位数", "10%–90% 情景带", "25%–75% 情景带", "实际净负荷", "零净负荷线"],
        ncol=3,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "q2_v2_joint_residual_scenarios"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    LOG_FILE.write_text(
        "问题二 V2 历史联合残差情景图重绘核查\n"
        f"中文字体：{font}\n"
        "代表日：2025-06-01；冻结信息文件记录的情景日数量：14。\n"
        "(a) 仅显示14条净负荷残差轨迹及零残差线；未叠加点预测或实际轨迹。\n"
        "(b) 使用同一14条情景计算10%/25%/50%/75%/90%分位数，并叠加冻结的日前点预测和实际净负荷。\n"
        "所有时序按144个官方10分钟槽位，以 step(where='post') 绘制；未改动冻结数据、模型或情景。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
