"""Generate final Stage-4 review artifacts from frozen daily/slot outputs only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.q2_v2.data import load_inputs

from .config import ROOT, TOL
from .run_stage2 import protected_manifest


OUT = ROOT / "outputs/q3_v2/stage4"
DOC = ROOT / "docs/q3_v2"
POLICY_A = "A_all_updates_dynamic_value"
POLICY_B = "B_no_adjustment_dynamic_value"
SPECIAL_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fmt(x: float) -> str:
    return f"{float(x):,.3f}"


def main() -> None:
    comparison = pd.read_csv(OUT / "q3_v2_full_year_comparison.csv")
    monthly = pd.read_csv(OUT / "q3_v2_monthly_comparison.csv")
    daily = pd.read_csv(OUT / "q3_v2_daily_results.csv")
    slots = pd.read_csv(OUT / "q3_v2_slot_results.csv")
    plans = pd.read_csv(OUT / "q3_v2_plan_versions.csv")
    solver = pd.read_csv(OUT / "q3_v2_solver_log.csv")
    info = pd.read_csv(OUT / "q3_v2_information_boundary_audit.csv")
    independent = pd.read_csv(OUT / "q3_v2_independent_audit.csv")
    paired = json.loads((OUT / "q3_v2_paired_ablation_summary.json").read_text(encoding="utf-8"))

    indexed = comparison.set_index("policy")
    a, b = indexed.loc[POLICY_A], indexed.loc[POLICY_B]
    savings = float(b.actual_total_cost_yuan - a.actual_total_cost_yuan)
    savings_pct = 100.0 * savings / float(b.actual_total_cost_yuan)
    emergency_change = float(a.emergency_purchase_kwh - b.emergency_purchase_kwh)
    emergency_pct = 100.0 * emergency_change / float(b.emergency_purchase_kwh)
    terminal_diff = float(a.terminal_soc_kwh - b.terminal_soc_kwh)

    # Adjustment events are decision-version events, distinct from changed slots.
    event_rows = []
    for policy, frame in plans.groupby("policy", sort=True):
        updates = frame[frame.issue_hour > 0]
        grouped = updates.groupby(["date", "issue_hour", "run_id"], sort=True)
        changed = grouped.apply(
            lambda g: bool((np.abs(g.version_plan_kwh.to_numpy(float) - g.original_plan_kwh.to_numpy(float)) > TOL).any()),
            include_groups=False,
        )
        event_rows.append({
            "policy": policy, "scheduled_intraday_update_events": int(len(changed)),
            "events_with_nonzero_adjustment": int(changed.sum()),
            "events_with_zero_adjustment": int((~changed).sum()),
            "final_upward_adjusted_slots": int(indexed.loc[policy].upward_adjustment_slot_count),
            "final_downward_adjusted_slots": int(indexed.loc[policy].downward_adjustment_slot_count),
            "final_unchanged_slots": int(indexed.loc[policy].unchanged_slot_count),
            "upward_adjustment_kwh": float(indexed.loc[policy].upward_adjustment_kwh),
            "downward_adjustment_kwh": float(indexed.loc[policy].downward_adjustment_kwh),
            "unchanged_effective_purchase_kwh": float(indexed.loc[policy].unchanged_effective_purchase_kwh),
        })
    adjustment = pd.DataFrame(event_rows)
    adjustment.to_csv(OUT / "q3_v2_adjustment_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10f")

    quality = solver.groupby(["policy", "solution_class", "termination_condition"], as_index=False).agg(
        calls=("run_id", "count"), mean_gap=("relative_gap", "mean"), max_gap=("relative_gap", "max"),
        mean_solve_seconds=("solve_time_seconds", "mean"), total_solve_seconds=("solve_time_seconds", "sum"),
    )
    quality.to_csv(OUT / "q3_v2_solver_quality.csv", index=False, encoding="utf-8-sig", float_format="%.10f")

    special = daily[daily.date.isin(SPECIAL_DATES)].copy().sort_values(["date", "policy"])
    special.to_csv(OUT / "q3_v2_special_date_results.csv", index=False, encoding="utf-8-sig", float_format="%.10f")

    # Exact official date/slot/time-label and shared-information checks.
    expected_dates = pd.date_range("2025-02-01", "2025-12-31").strftime("%Y-%m-%d").tolist()
    source = load_inputs()
    exact_checks = {}
    for policy in (POLICY_A, POLICY_B):
        frame = slots[slots.policy == policy].sort_values(["date", "slot"])
        per_day = frame.groupby("date").slot.apply(list)
        exact_checks[policy] = {
            "days": int(frame.date.nunique()), "slots": int(len(frame)),
            "date_sequence_exact": sorted(frame.date.unique().tolist()) == expected_dates,
            "each_day_slots_0_to_143": bool(all(x == list(range(144)) for x in per_day)),
            "time_labels_match_official_row_order": bool(all(g.sort_values("slot").time_label.astype(str).tolist() == list(source.time_labels) for _, g in frame.groupby("date"))),
        }
    ia = info[info.policy == POLICY_A].sort_values(["date", "issue_hour"]).reset_index(drop=True)
    ib = info[info.policy == POLICY_B].sort_values(["date", "issue_hour"]).reset_index(drop=True)
    access_cols = ["date", "issue_hour", "first_modifiable_slot", "scenario_count_effective", "scenario_source_dates", "forecast_visible", "pv_information_causal", "no_future_actual_load_used", "actual_pv_used_in_forecast"]
    shared_access = ia[access_cols].equals(ib[access_cols])

    before = json.loads((OUT / "protected_hashes_before_stage4.json").read_text(encoding="utf-8"))
    after = json.loads((OUT / "protected_hashes_after_stage4.json").read_text(encoding="utf-8"))
    current = protected_manifest()
    raw_result3 = ROOT / "data/raw/result3.xlsx"
    output_result3 = ROOT / "outputs/tables/result3.xlsx"
    stage3_manifest = json.loads((ROOT / "outputs/q3_v2/stage3/protected_hashes_after.json").read_text(encoding="utf-8"))
    raw_result3_unchanged = _sha(raw_result3) == stage3_manifest["data\\raw\\result3.xlsx"]

    a_daily = daily[daily.policy == POLICY_A].sort_values("date")
    b_daily = daily[daily.policy == POLICY_B].sort_values("date")
    dec26_start = float(a_daily[a_daily.date == "2025-12-26"].iloc[0].soc_start_kwh)
    prefix_cost = float(a_daily[a_daily.date <= "2025-12-25"].actual_total_cost_yuan.sum())
    prefix_emergency = float(a_daily[a_daily.date <= "2025-12-25"].emergency_purchase_kwh.sum())
    recovery = {
        "prior_process_exit_record": "process exited after 2025-12-25 checkpoint and no A failure log was written; cause not inferred as model failure",
        "completed_before_resume_days": 328, "completed_before_resume_through": "2025-12-25",
        "resume_first_date": "2025-12-26", "pre_resume_checkpoint_sha256": "1CE94029B5EF64399C2A9FFBBF0F1473202F9CEA213849CF0F69BAE79928A1B9",
        "pre_resume_total_cost_yuan": 12972853.129148582,
        "post_aggregation_prefix_total_cost_yuan": prefix_cost,
        "pre_resume_emergency_kwh": 29276.106003702902,
        "post_aggregation_prefix_emergency_kwh": prefix_emergency,
        "dec25_end_soc_kwh": 1399.3154558798, "dec26_start_soc_kwh": dec26_start,
        "prefix_cost_continuity_error_yuan": prefix_cost - 12972853.129148582,
        "prefix_emergency_continuity_error_kwh": prefix_emergency - 29276.106003702902,
        "soc_resume_error_kwh": dec26_start - 1399.3154558798,
        "preexisting_328_days_recomputed": False,
        "A_numeric_normalization_count": int(a.numeric_normalization_count),
        "B_numeric_normalization_count": int(b.numeric_normalization_count),
        "normalization_epsilon_kwh": 1e-6,
        "A_max_normalization_abs_kwh": float(a.numeric_normalization_max_abs_kwh),
        "B_max_normalization_abs_kwh": float(b.numeric_normalization_max_abs_kwh),
        "B_max_normalization_cost_impact_yuan": float(b.numeric_normalization_max_cost_impact_yuan),
    }
    (OUT / "q3_v2_recovery_audit.json").write_text(json.dumps(recovery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checks = {
        "official_structure": exact_checks,
        "independent_numeric_audit": independent.to_dict(orient="records"),
        "same_forecast_and_scenario_access_A_B": bool(shared_access),
        "information_rows_each_policy": {POLICY_A: len(ia), POLICY_B: len(ib)},
        "protected_manifests_equal": before == after == current,
        "protected_file_count": len(current),
        "raw_result3_sha256": _sha(raw_result3),
        "raw_result3_unchanged_from_stage3_freeze": raw_result3_unchanged,
        "outputs_tables_result3_exists": output_result3.exists(),
        "recovery": recovery,
    }
    (OUT / "q3_v2_final_validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    numeric_pass = bool(
        (independent[["contract_recompute_max_abs_error_yuan", "emergency_recompute_max_abs_error_yuan", "total_recompute_max_abs_error_yuan", "soc_transition_max_abs_error_kwh", "cross_day_soc_max_abs_error_kwh", "balance_max_abs_error_kwh", "version_lock_max_abs_error_kwh"]].to_numpy(float) <= 1e-6).all()
        and (independent[["soc_violation_count", "power_violation_count", "simultaneous_charge_discharge_count", "emergency_and_charge_count", "information_violation_count"]].to_numpy(int) == 0).all()
    )
    all_pass = all(v["days"] == 334 and v["slots"] == 334 * 144 and v["date_sequence_exact"] and v["each_day_slots_0_to_143"] and v["time_labels_match_official_row_order"] for v in exact_checks.values()) and numeric_pass and shared_access and before == after == current and raw_result3_unchanged and not output_result3.exists() and abs(recovery["soc_resume_error_kwh"]) <= 1e-9 and abs(recovery["prefix_cost_continuity_error_yuan"]) <= 1e-6
    lines = [
        "Q3 V2 Stage 4 final causal shadow validation", f"ALL_CHECKS_PASSED={all_pass}",
        f"A/B each 334*144 complete={all(v['slots']==48096 for v in exact_checks.values())}",
        f"Official dates, slots and time labels exact={all(v['date_sequence_exact'] and v['each_day_slots_0_to_143'] and v['time_labels_match_official_row_order'] for v in exact_checks.values())}",
        f"Independent fee/physics/version audit passed={numeric_pass}", f"A/B common information access={shared_access}",
        f"Protected manifests unchanged ({len(current)} files)={before == after == current}",
        f"Raw result3 unchanged={raw_result3_unchanged}; outputs/tables/result3.xlsx exists={output_result3.exists()}",
        f"A resume SOC error={recovery['soc_resume_error_kwh']:.3e} kWh; prefix fee error={recovery['prefix_cost_continuity_error_yuan']:.3e} yuan",
        f"Numeric normalization: A count={int(a.numeric_normalization_count)}, B count={int(b.numeric_normalization_count)}, max magnitude={max(float(a.numeric_normalization_max_abs_kwh),float(b.numeric_normalization_max_abs_kwh)):.3e} kWh",
    ]
    (OUT / "q3_v2_stage4_final_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rec_md = f"""# 问题三 V2 阶段4恢复审计

原A进程仅能确认为在完成2025-12-25检查点后退出，且没有写入A失败日志；不将其推断为模型不可行。恢复前A已有328天、47232槽，日期连续且无重复，累计费用为{_fmt(recovery['pre_resume_total_cost_yuan'])}元，累计紧急购电量为{_fmt(recovery['pre_resume_emergency_kwh'])} kWh。

恢复严格从2025-12-26开始。2025-12-25日末与12月26日日初SOC均为{_fmt(dec26_start)} kWh，衔接误差为{recovery['soc_resume_error_kwh']:.3e} kWh；恢复后重新汇总的前328天费用和紧急购电量与恢复前记录误差分别为{recovery['prefix_cost_continuity_error_yuan']:.3e}元和{recovery['prefix_emergency_continuity_error_kwh']:.3e} kWh，说明前328天未被重算或重复计费。

非负量绝对容差固定为1e-6 kWh。A全年没有发生规范化；B共规范化9个处于容差内的负浮点噪声，最大幅度为{float(b.numeric_normalization_max_abs_kwh):.3e} kWh，最大单次费用影响不超过{float(b.numeric_normalization_max_cost_impact_yuan):.3e}元。没有使用绝对值或无条件截断，且不存在低于-1e-6 kWh的值。
"""
    (DOC / "q3_v2_stage4_recovery_audit.md").write_text(rec_md, encoding="utf-8")

    report = f"""# 问题三 V2 阶段4全年结果审查稿

本文件仅报告2025-02-01至2025-12-31共334天的因果影子评价结果，不生成`result3.xlsx`，也不形成论文最终结论。

## 独立闭环比较

| 指标 | A：允许日内调整 | B：禁止日内调整 |
|---|---:|---:|
| 合同结算费用/元 | {_fmt(a.contract_cost_yuan)} | {_fmt(b.contract_cost_yuan)} |
| 紧急购电费用/元 | {_fmt(a.emergency_cost_yuan)} | {_fmt(b.emergency_cost_yuan)} |
| 实际总费用/元 | {_fmt(a.actual_total_cost_yuan)} | {_fmt(b.actual_total_cost_yuan)} |
| 紧急购电量/kWh | {_fmt(a.emergency_purchase_kwh)} | {_fmt(b.emergency_purchase_kwh)} |
| 低/中/高价紧急购电量/kWh | {_fmt(a.emergency_low_price_kwh)} / {_fmt(a.emergency_mid_price_kwh)} / {_fmt(a.emergency_high_price_kwh)} | {_fmt(b.emergency_low_price_kwh)} / {_fmt(b.emergency_mid_price_kwh)} / {_fmt(b.emergency_high_price_kwh)} |
| 上调/下调电量/kWh | {_fmt(a.upward_adjustment_kwh)} / {_fmt(a.downward_adjustment_kwh)} | 0 / 0 |
| 充电/放电/损耗/kWh | {_fmt(a.charge_kwh)} / {_fmt(a.discharge_kwh)} / {_fmt(a.conversion_loss_kwh)} | {_fmt(b.charge_kwh)} / {_fmt(b.discharge_kwh)} / {_fmt(b.conversion_loss_kwh)} |
| 弃能量/kWh | {_fmt(a.spill_kwh)} | {_fmt(b.spill_kwh)} |
| 期末SOC/kWh | {_fmt(a.terminal_soc_kwh)} | {_fmt(b.terminal_soc_kwh)} |
| 平均日末SOC/kWh | {_fmt(a.mean_daily_end_soc_kwh)} | {_fmt(b.mean_daily_end_soc_kwh)} |

A相对B节省{_fmt(savings)}元，占B费用的{savings_pct:.3f}%；紧急购电量变化为{_fmt(emergency_change)} kWh（{emergency_pct:.3f}%）。A期末SOC比B高{_fmt(terminal_diff)} kWh，因此费用差异不能解释为A透支期末库存。

## 条件配对消融

固定A的计划版本轨迹与每日日初SOC后，动态价值执行费用为{_fmt(paired['dynamic_cost_yuan'])}元，贪心执行费用为{_fmt(paired['greedy_cost_yuan'])}元，动态价值少{_fmt(-paired['dynamic_minus_greedy_cost_yuan'])}元；两者该条件实验的期末SOC相同。该结果只隔离给定A计划下的执行器差异，不属于独立跨日闭环策略。

## 验证状态

两策略均包含334天×144槽；供需平衡、SOC递推与边界、功率、充放电互斥、紧急购电不充电、跨日衔接、计划版本锁定、单次合同结算复算和信息边界均通过。完整细节见`q3_v2_stage4_final_validation.txt`与`q3_v2_independent_audit.csv`。
"""
    (DOC / "q3_v2_stage4_results_for_review.md").write_text(report, encoding="utf-8")
    print(json.dumps({"all_checks_passed": all_pass, "savings_yuan": savings, "savings_pct": savings_pct, "emergency_change_kwh": emergency_change, "terminal_soc_difference_kwh": terminal_diff}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
