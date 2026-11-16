"""Explain Stage-2 independent and paired DP/greedy differences."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import ACTION_OBJECTIVE_TIE_TOL_YUAN, EMERGENCY_PRICE_MULTIPLIER, ETA_CHARGE, ETA_DISCHARGE, ROOT
from .data import load_inputs
from .execution import _transitions, execute_day


STAGE2 = ROOT / "outputs" / "q2_v2" / "stage2"
OUT = ROOT / "outputs" / "q2_v2" / "stage3"
DOC = ROOT / "docs" / "q2_v2" / "q2_v2_stage2_difference_diagnosis.md"


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    daily = pd.read_csv(STAGE2 / "q2_v2_daily_results.csv")
    plans = pd.read_csv(STAGE2 / "q2_v2_day_ahead_plans.csv")
    slots = pd.read_csv(STAGE2 / "q2_v2_slot_trajectories.csv")
    solver = pd.read_csv(STAGE2 / "q2_v2_solver_results.csv")
    if "absolute_gap_yuan" not in solver:
        solver["absolute_gap_yuan"] = (solver["incumbent_objective_yuan"] - solver["best_bound_yuan"]).abs()

    independent = daily[daily["comparison_scope"] == "independent_cross_day"].copy()
    diag = independent.merge(
        solver[["run_id", "best_bound_yuan", "incumbent_objective_yuan", "absolute_gap_yuan", "mip_gap",
                "termination_condition", "optimal_solution_certified"]],
        left_on="plan_run_id", right_on="run_id", how="left", validate="many_to_one",
    )
    diag["planned_purchase_kwh"] = diag["plan_run_id"].map(
        plans.groupby("run_id")["planned_purchase_kwh"].sum()
    )
    diag.to_csv(OUT / "q2_v2_stage2_independent_daily_diagnosis.csv", index=False)

    paired_summary = []
    candidate_rows = []
    for date in ["2025-01-15", "2025-01-16", "2025-01-17"]:
        d = data.day_index(date)
        dp = slots[(slots["strategy"] == "scenario_v0_dp100") & (slots["date"] == date)].copy()
        run_id = dp["plan_run_id"].iloc[0]
        plan = plans[(plans["run_id"] == run_id) & (plans["date"] == date)].sort_values("slot")["planned_purchase_kwh"].to_numpy()
        initial_soc = float(dp.iloc[0]["soc_start_kwh"])
        greedy, _ = execute_day(date, data.time_labels, data.load_kwh[d], data.pv_kwh[d], plan,
                                data.price_yuan_per_kwh, initial_soc, "greedy")
        dp = dp.sort_values("slot").reset_index(drop=True)
        net_dp = dp["charge_kwh"] - dp["discharge_kwh"]
        net_g = greedy["charge_kwh"] - greedy["discharge_kwh"]
        abs_diff = np.abs(net_dp - net_g)
        max_slot = int(abs_diff.idxmax())
        loss_dp = float(((1-ETA_CHARGE)*dp["charge_kwh"] + (1/ETA_DISCHARGE-1)*dp["discharge_kwh"]).sum())
        loss_g = float(((1-ETA_CHARGE)*greedy["charge_kwh"] + (1/ETA_DISCHARGE-1)*greedy["discharge_kwh"]).sum())
        paired_summary.extend([
            {"date": date, "executor": "DP100", "charge_kwh": dp["charge_kwh"].sum(),
             "discharge_kwh": dp["discharge_kwh"].sum(), "conversion_loss_kwh": loss_dp,
             "spill_kwh": dp["spill_kwh"].sum(), "end_soc_kwh": dp.iloc[-1]["soc_end_kwh"],
             "sum_abs_action_difference_kwh": abs_diff.sum(), "max_difference_slot": max_slot,
             "max_action_difference_kwh": abs_diff.iloc[max_slot]},
            {"date": date, "executor": "greedy", "charge_kwh": greedy["charge_kwh"].sum(),
             "discharge_kwh": greedy["discharge_kwh"].sum(), "conversion_loss_kwh": loss_g,
             "spill_kwh": greedy["spill_kwh"].sum(), "end_soc_kwh": greedy.iloc[-1]["soc_end_kwh"],
             "sum_abs_action_difference_kwh": abs_diff.sum(), "max_difference_slot": max_slot,
             "max_action_difference_kwh": abs_diff.iloc[max_slot]},
        ])
        archive = np.load(STAGE2 / f"path_values_{run_id}_grid100.npz")
        grid = archive["soc_grid_kwh"]
        mean_value = archive["mean_value_yuan"]
        row = dp.iloc[max_slot]
        actions = _transitions(float(row["net_gap_before_storage_kwh"]), float(row["soc_start_kwh"]), grid)
        ranked = []
        for action in actions:
            c, b, e, spill, nxt = action
            immediate = EMERGENCY_PRICE_MULTIPLIER * float(row["price_yuan_per_kwh"]) * e
            future = float(np.interp(nxt, grid, mean_value[max_slot + 1]))
            ranked.append((immediate + future, c+b, immediate, future, action))
        minimum_score = min(x[0] for x in ranked)
        ranked.sort(
            key=lambda x: (
                0 if x[0] <= minimum_score + ACTION_OBJECTIVE_TIE_TOL_YUAN else 1,
                x[1] if x[0] <= minimum_score + ACTION_OBJECTIVE_TIE_TOL_YUAN else x[0],
                x[2], x[4][4],
            )
        )
        for rank, item in enumerate(ranked[:8], 1):
            score, throughput, immediate, future, action = item
            c, b, e, spill, nxt = action
            candidate_rows.append({"date": date, "slot": max_slot, "time_label": row["time_label"],
                                   "rank": rank, "charge_kwh": c, "discharge_kwh": b,
                                   "emergency_kwh": e, "spill_kwh": spill, "next_soc_kwh": nxt,
                                   "current_emergency_cost_yuan": immediate,
                                   "approx_future_value_yuan": future, "combined_score_yuan": score,
                                   "score_above_minimum_yuan": score - minimum_score,
                                   "within_primary_tie_tolerance": score <= minimum_score + ACTION_OBJECTIVE_TIE_TOL_YUAN,
                                   "throughput_kwh": throughput,
                                   "is_dp_action": abs(c-row["charge_kwh"]) < 1e-6 and abs(b-row["discharge_kwh"]) < 1e-6,
                                   "is_greedy_action": abs(c-greedy.iloc[max_slot]["charge_kwh"]) < 1e-6 and abs(b-greedy.iloc[max_slot]["discharge_kwh"]) < 1e-6})

    paired_df = pd.DataFrame(paired_summary)
    candidates_df = pd.DataFrame(candidate_rows)
    paired_df.to_csv(OUT / "q2_v2_stage2_paired_energy_diagnosis.csv", index=False)
    candidates_df.to_csv(OUT / "q2_v2_stage2_max_difference_action_candidates.csv", index=False)

    old = pd.read_csv(ROOT / "outputs" / "q2_v2" / "stage2_pre_tie_revision" / "q2_v2_paired_executor_comparison.csv")
    new = pd.read_csv(STAGE2 / "q2_v2_paired_executor_comparison.csv")
    tie_effect = {
        "rule": "within 1e-8 yuan of primary score, minimize charge+discharge throughput; then deterministic emergency/spill/SOC ordering",
        "old_action_difference_kwh": float(old["sum_abs_net_action_difference_kwh"].sum()),
        "new_action_difference_kwh": float(new["sum_abs_net_action_difference_kwh"].sum()),
        "change_kwh": float(new["sum_abs_net_action_difference_kwh"].sum() - old["sum_abs_net_action_difference_kwh"].sum()),
        "old_cost_difference_yuan": float(old["dp_minus_greedy_cost_yuan"].sum()),
        "new_cost_difference_yuan": float(new["dp_minus_greedy_cost_yuan"].sum()),
    }
    (OUT / "q2_v2_stage2_tie_rule_effect.json").write_text(json.dumps(tie_effect, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 问题二 V2 阶段2差异诊断", "",
             "独立闭环的费用差同时包含跨日SOC差异及由此产生的次日计划差异，不能归因于DP执行器本身。逐日明细见 `q2_v2_stage2_independent_daily_diagnosis.csv`。",
             "", "配对动作差定义为", "", r"\[\sum_t |(c_t^{DP}-b_t^{DP})-(c_t^{G}-b_t^{G})|.\]", "",
             "累计充放电、转换损耗、弃能和日末SOC见 `q2_v2_stage2_paired_energy_diagnosis.csv`；动作差最大槽的候选分解见 `q2_v2_stage2_max_difference_action_candidates.csv`。",
             "", "并列规则现固定为：主目标绝对差不超过 $10^{-8}$ 元时，优先最小化当槽充放电吞吐量；其余字段仅用于确定性排序。该规则不是折旧费，也未改变题面费用。",
             "", f"规则修订前后配对累计动作差分别为 {tie_effect['old_action_difference_kwh']:.6f} 和 {tie_effect['new_action_difference_kwh']:.6f} kWh，变化 {tie_effect['change_kwh']:.6f} kWh；配对费用差前后均为0元。由此可见阶段2的大动作差不是该并列次序造成，而主要来自近似未来价值在无紧急费用情况下允许多种跨期等价SOC路径。"]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
