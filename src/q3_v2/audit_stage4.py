"""Independent read-only audit and narrative for completed Q3-V2 Stage 4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ENERGY_MAX_KWH, ROOT, SOC_MAX_KWH, SOC_MIN_KWH, TOL
from .settlement import settle_contract


OUT = ROOT / "outputs/q3_v2/stage4"
DOC = ROOT / "docs/q3_v2"


def _check(policy: str, slots: pd.DataFrame, plans: pd.DataFrame, info: pd.DataFrame) -> dict:
    rows = slots[slots.policy == policy].sort_values(["date", "slot"]).copy()
    plan = plans[plans.policy == policy].copy()
    audit = info[info.policy == policy].copy()
    settled = settle_contract(rows.original_plan_kwh, rows.effective_purchase_kwh, rows.price_yuan_per_kwh)
    emergency = 5.0 * rows.price_yuan_per_kwh.to_numpy(float) * rows.emergency_purchase_kwh.to_numpy(float)
    expected_soc = rows.soc_start_kwh + 0.9 * rows.charge_rt_kwh - rows.discharge_rt_kwh / 0.9
    grouped = list(rows.groupby("date", sort=True))
    cross = [abs(float(a.iloc[-1].soc_end_kwh) - float(b.iloc[0].soc_start_kwh)) for (_, a), (_, b) in zip(grouped[:-1], grouped[1:])]
    # Last issued plan for a slot must equal the plan used at execution.
    final_versions = plan.sort_values(["date", "slot", "issue_hour"]).groupby(["date", "slot"], as_index=False).tail(1)
    merged = rows.merge(final_versions[["date", "slot", "version_plan_kwh"]], on=["date", "slot"], how="left")
    contract_saved = rows.contract_cost_yuan.to_numpy(float)
    emergency_saved = rows.emergency_cost_yuan.to_numpy(float)
    total_saved = rows.actual_total_cost_yuan.to_numpy(float)
    contract_recomputed = settled.contract_cost_yuan.to_numpy(float)
    return {
        "policy": policy, "slots": len(rows), "days": rows.date.nunique(),
        "contract_recompute_max_abs_error_yuan": float(np.max(np.abs(contract_saved - contract_recomputed))),
        "emergency_recompute_max_abs_error_yuan": float(np.max(np.abs(emergency_saved - emergency))),
        "total_recompute_max_abs_error_yuan": float(np.max(np.abs(total_saved - contract_recomputed - emergency))),
        "soc_transition_max_abs_error_kwh": float(np.max(np.abs(rows.soc_end_kwh - expected_soc))),
        "cross_day_soc_max_abs_error_kwh": float(max(cross, default=0.0)),
        "balance_max_abs_error_kwh": float(rows.balance_residual_kwh.abs().max()),
        "soc_violation_count": int(((rows.soc_end_kwh < SOC_MIN_KWH - TOL) | (rows.soc_end_kwh > SOC_MAX_KWH + TOL)).sum()),
        "power_violation_count": int(((rows.charge_rt_kwh > ENERGY_MAX_KWH + TOL) | (rows.discharge_rt_kwh > ENERGY_MAX_KWH + TOL)).sum()),
        "simultaneous_charge_discharge_count": int(((rows.charge_rt_kwh > TOL) & (rows.discharge_rt_kwh > TOL)).sum()),
        "emergency_and_charge_count": int(((rows.emergency_purchase_kwh > TOL) & (rows.charge_rt_kwh > TOL)).sum()),
        "information_violation_count": int((~audit.all_sources_prior).sum() + (~audit.forecast_visible).sum() + (~audit.pv_information_causal).sum() + (~audit.no_future_actual_load_used).sum() + audit.actual_pv_used_in_forecast.sum()),
        "version_lock_max_abs_error_kwh": float(np.max(np.abs(merged.effective_purchase_kwh - merged.version_plan_kwh))),
    }


def main() -> None:
    slots = pd.read_csv(OUT / "q3_v2_slot_results.csv")
    plans = pd.read_csv(OUT / "q3_v2_plan_versions.csv")
    info = pd.read_csv(OUT / "q3_v2_information_boundary_audit.csv")
    policies = sorted(slots.policy.unique())
    checks = pd.DataFrame([_check(p, slots, plans, info) for p in policies])
    checks.to_csv(OUT / "q3_v2_independent_audit.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    compare = pd.read_csv(OUT / "q3_v2_full_year_comparison.csv")
    a = compare.set_index("policy").loc["A_all_updates_dynamic_value"]
    b = compare.set_index("policy").loc["B_no_adjustment_dynamic_value"]
    difference = float(b.actual_total_cost_yuan - a.actual_total_cost_yuan)
    terminal_difference = float(a.terminal_soc_kwh - b.terminal_soc_kwh)
    text = f"""# 问题三 V2 阶段4全年因果影子评价

## 比较协议

正式统计期为2025-02-01至2025-12-31，共334天。策略A允许在6:00、12:00、18:00对未执行槽位更新购电量；策略B固定0:00购电计划，但可在相同信息边界内使用已发布预报刷新储能未来价值函数。二者分别从各自在阶段3的合法1月连续预热末SOC承接，未在2月1日重置。

实际费用按每槽最后生效量相对0:00原计划的单次合同结算，加上5倍当槽电价紧急购电费；终端库存价值未从实际账单中扣除。历史路径价值函数是对因果历史联合情景逐路径后向值函数的加权平均，为近似未来成本，不是拥有未来实际信息的理论全局最优控制。

## 独立闭环结果

策略A实际总费用为{a.actual_total_cost_yuan:.3f}元，策略B为{b.actual_total_cost_yuan:.3f}元；B减A为{difference:.3f}元。两者期末SOC分别为{a.terminal_soc_kwh:.3f}和{b.terminal_soc_kwh:.3f} kWh，差值为{terminal_difference:.3f} kWh，解释费用差异时必须同时报告该库存差异。

完整年度、月度、日度、槽位、计划版本与求解质量结果见`outputs/q3_v2/stage4/`。条件配对消融固定A的日内计划版本和每日日初SOC，因此仅隔离实时执行器，不等同于两套独立跨日策略。

## 独立复算

逐项复算表位于`q3_v2_independent_audit.csv`。所有费用、状态转移、供需平衡、跨日SOC、物理边界、信息可用性和执行槽位版本锁定均要求在容差{TOL:g}内通过；若任一指标不通过，应停止后续模板导出。
"""
    (DOC / "q3_v2_full_year_analysis.md").write_text(text, encoding="utf-8")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
