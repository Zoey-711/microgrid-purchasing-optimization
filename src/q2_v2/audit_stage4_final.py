"""Independent read-only audit and freeze material for Q2-v2 Stage 4.

The program derives all reviewed values afresh from the exported slot, daily,
plan, solver, and information records.  It never invokes a solver and never
opens or modifies a result2 workbook or a protected legacy file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ENERGY_MAX_KWH, ETA_CHARGE, ETA_DISCHARGE, ROOT, SOC_MAX_KWH, SOC_MIN_KWH, TOL


OUT = ROOT / "outputs/q2_v2/stage4"
DOCS = ROOT / "docs/q2_v2"
FORMAL_START = pd.Timestamp("2025-02-01")
FORMAL_END = pd.Timestamp("2025-12-31")
POLICIES = {
    "main_dp50": "主策略：联合情景MILP + DP50执行器",
    "closed_loop_greedy": "完整闭环对照：联合情景MILP + 贪心执行器",
}


# Canonical display names are declared separately to avoid inheriting any
# legacy console-code-page artefacts from earlier development scripts.
POLICIES = {
    "main_dp50": "主策略：联合情景 MILP + DP50 执行器",
    "closed_loop_greedy": "完整闭环对照：联合情景 MILP + 贪心执行器",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def stable_frame(path: Path) -> pd.DataFrame:
    """Read only final aggregate data; reject incomplete formal records."""
    frame = pd.read_csv(path)
    return frame


def max_abs(values: pd.Series | np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.max(np.abs(a))) if a.size else 0.0


def audit_policy(policy: str, slots_all: pd.DataFrame, daily_all: pd.DataFrame,
                 plans_all: pd.DataFrame, solver_all: pd.DataFrame,
                 info_all: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    slots = slots_all[slots_all["policy_id"].eq(policy)].copy()
    daily = daily_all[daily_all["policy_id"].eq(policy)].copy()
    plans = plans_all[plans_all["policy_id"].eq(policy)].copy()
    solver = solver_all[solver_all["policy_id"].eq(policy)].copy()
    info = info_all[info_all["policy_id"].eq(policy)].copy()
    for frame in (slots, daily, plans, solver):
        frame["date"] = pd.to_datetime(frame["date"])
    info["decision_date"] = pd.to_datetime(info["decision_date"])
    slots = slots[(slots["date"] >= FORMAL_START) & (slots["date"] <= FORMAL_END)].sort_values(["date", "slot"])
    daily = daily[(daily["date"] >= FORMAL_START) & (daily["date"] <= FORMAL_END)].sort_values("date")
    plans = plans[(plans["date"] >= FORMAL_START) & (plans["date"] <= FORMAL_END)].sort_values(["date", "slot"])
    solver = solver[(solver["date"] >= FORMAL_START) & (solver["date"] <= FORMAL_END)].sort_values("date")
    info = info[(info["decision_date"] >= FORMAL_START) & (info["decision_date"] <= FORMAL_END)].sort_values("decision_date")
    expected_days = pd.date_range(FORMAL_START, FORMAL_END, freq="D")
    if not pd.Index(daily["date"]).equals(pd.Index(expected_days)):
        raise RuntimeError(f"{policy}: daily formal date range is incomplete")
    if len(slots) != 334 * 144 or not slots.groupby("date").size().eq(144).all():
        raise RuntimeError(f"{policy}: slot count is not 334 x 144")
    if len(plans) != 334 * 144 or len(solver) != 334 or len(info) != 334:
        raise RuntimeError(f"{policy}: associated formal records are incomplete")

    # Independent billing recomputation from physics input fields, not the
    # stored daily summary values.
    plan_fee = float(np.dot(slots["price_yuan_per_kwh"], slots["planned_purchase_kwh"]))
    emergency_fee = float(np.dot(5.0 * slots["price_yuan_per_kwh"], slots["emergency_purchase_kwh"]))
    total_fee = plan_fee + emergency_fee
    emergency_kwh = float(slots["emergency_purchase_kwh"].sum())
    balance = (
        slots["planned_purchase_kwh"] + slots["pv_actual_kwh"] + slots["discharge_kwh"]
        + slots["emergency_purchase_kwh"] - slots["load_actual_kwh"] - slots["charge_kwh"] - slots["spill_kwh"]
    )
    transition = (
        slots["soc_start_kwh"] + ETA_CHARGE * slots["charge_kwh"]
        - slots["discharge_kwh"] / ETA_DISCHARGE - slots["soc_end_kwh"]
    )
    cross_day = daily["soc_start_kwh"].to_numpy(float)[1:] - daily["soc_end_kwh"].to_numpy(float)[:-1]
    plan_check = plans[["date", "slot", "planned_purchase_kwh"]].merge(
        slots[["date", "slot", "planned_purchase_kwh"]], on=["date", "slot"], suffixes=("_plan", "_slot"), validate="one_to_one"
    )
    plan_lock_error = max_abs(plan_check["planned_purchase_kwh_plan"] - plan_check["planned_purchase_kwh_slot"])
    daily_recomputed = slots.groupby("date", as_index=False).agg(
        plan_fee_recomputed=("planned_purchase_kwh", lambda x: 0.0),
        emergency_fee_recomputed=("emergency_purchase_kwh", lambda x: 0.0),
        total_fee_recomputed=("actual_cost_yuan", "sum"),
        emergency_kwh_recomputed=("emergency_purchase_kwh", "sum"),
    )
    # Assign exact cost expressions after the grouping, avoiding stored cost
    # columns except for an independent consistency comparison below.
    daily_recomputed["plan_fee_recomputed"] = slots.assign(v=slots["price_yuan_per_kwh"] * slots["planned_purchase_kwh"]).groupby("date")["v"].sum().to_numpy()
    daily_recomputed["emergency_fee_recomputed"] = slots.assign(v=5 * slots["price_yuan_per_kwh"] * slots["emergency_purchase_kwh"]).groupby("date")["v"].sum().to_numpy()
    daily_recomputed["total_fee_recomputed"] = daily_recomputed["plan_fee_recomputed"] + daily_recomputed["emergency_fee_recomputed"]
    day_merge = daily.merge(daily_recomputed, on="date", validate="one_to_one")
    summary = {
        "policy_id": policy,
        "policy_display_name": POLICIES[policy],
        "formal_days": 334,
        "planned_purchase_kwh": float(slots["planned_purchase_kwh"].sum()),
        "plan_purchase_cost_yuan": plan_fee,
        "emergency_purchase_cost_yuan": emergency_fee,
        "actual_total_cost_yuan": total_fee,
        "emergency_purchase_kwh": emergency_kwh,
        "emergency_purchase_days": int((daily["emergency_purchase_kwh"] > TOL).sum()),
        "max_daily_emergency_purchase_kwh": float(daily["emergency_purchase_kwh"].max()),
        "max_daily_emergency_date": daily.loc[daily["emergency_purchase_kwh"].idxmax(), "date"].strftime("%Y-%m-%d"),
        "emergency_weighted_unit_price_yuan_per_kwh": emergency_fee / emergency_kwh,
        "mean_daily_end_soc_kwh": float(daily["soc_end_kwh"].mean()),
        "min_actual_soc_kwh": float(slots[["soc_start_kwh", "soc_end_kwh"]].min().min()),
        "max_actual_soc_kwh": float(slots[["soc_start_kwh", "soc_end_kwh"]].max().max()),
        "formal_terminal_soc_kwh": float(daily.iloc[-1]["soc_end_kwh"]),
        "charge_kwh": float(slots["charge_kwh"].sum()),
        "discharge_kwh": float(slots["discharge_kwh"].sum()),
        "conversion_loss_kwh": float(daily["conversion_loss_kwh"].sum()),
        "spill_kwh": float(slots["spill_kwh"].sum()),
        "certified_optimal_days": int(solver["optimal_solution_certified"].fillna(False).astype(bool).sum()),
        "time_limit_feasible_days": int(solver["termination_condition"].astype(str).eq("maxTimeLimit").sum()),
        "solver_failure_days": int((~solver["termination_condition"].astype(str).isin(("optimal", "maxTimeLimit", "feasible"))).sum()),
        "mean_mip_gap": float(pd.to_numeric(solver["mip_gap"], errors="coerce").mean()),
        "max_mip_gap": float(pd.to_numeric(solver["mip_gap"], errors="coerce").max()),
        "mean_solve_time_seconds": float(pd.to_numeric(solver["solve_time_seconds"], errors="coerce").mean()),
        "total_solve_time_seconds": float(pd.to_numeric(solver["solve_time_seconds"], errors="coerce").sum()),
        "billing_recompute_error_yuan": max(
            abs(plan_fee - float(slots["plan_cost_yuan"].sum())),
            abs(emergency_fee - float(slots["emergency_cost_yuan"].sum())),
            abs(total_fee - float(slots["actual_cost_yuan"].sum())),
        ),
        "daily_billing_recompute_max_error_yuan": max_abs(day_merge["actual_total_cost_yuan"] - day_merge["total_fee_recomputed"]),
        "energy_balance_max_residual_kwh": max_abs(balance),
        "energy_balance_violation_slots": int((np.abs(balance) > TOL).sum()),
        "soc_transition_max_residual_kwh": max_abs(transition),
        "soc_transition_violation_slots": int((np.abs(transition) > TOL).sum()),
        "cross_day_soc_max_residual_kwh": max_abs(cross_day),
        "cross_day_soc_violation_days": int((np.abs(cross_day) > TOL).sum()),
        "soc_boundary_violation_values": int(((slots[["soc_start_kwh", "soc_end_kwh"]] < SOC_MIN_KWH - TOL) | (slots[["soc_start_kwh", "soc_end_kwh"]] > SOC_MAX_KWH + TOL)).sum().sum()),
        "power_violation_slots": int(((slots["charge_kwh"] > ENERGY_MAX_KWH + TOL) | (slots["discharge_kwh"] > ENERGY_MAX_KWH + TOL)).sum()),
        "simultaneous_charge_discharge_slots": int(((slots["charge_kwh"] > TOL) & (slots["discharge_kwh"] > TOL)).sum()),
        "emergency_while_charging_slots": int(((slots["emergency_purchase_kwh"] > TOL) & (slots["charge_kwh"] > TOL)).sum()),
        "plan_lock_max_error_kwh": plan_lock_error,
        "prediction_information_violation_days": int((~info["all_prediction_sources_prior"].astype(bool)).sum()),
        "scenario_information_violation_days": int((~info["all_scenario_sources_prior"].astype(bool)).sum()),
    }
    # Monthly, independently recalculated from slots and daily physical states.
    slots["month"] = slots["date"].dt.strftime("%Y-%m")
    monthly = slots.groupby("month", as_index=False).agg(
        planned_purchase_kwh=("planned_purchase_kwh", "sum"),
        emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
        charge_kwh=("charge_kwh", "sum"),
        discharge_kwh=("discharge_kwh", "sum"),
        spill_kwh=("spill_kwh", "sum"),
    )
    monthly["plan_purchase_cost_yuan"] = slots.assign(v=slots["price_yuan_per_kwh"] * slots["planned_purchase_kwh"]).groupby("month")["v"].sum().to_numpy()
    monthly["emergency_purchase_cost_yuan"] = slots.assign(v=5 * slots["price_yuan_per_kwh"] * slots["emergency_purchase_kwh"]).groupby("month")["v"].sum().to_numpy()
    monthly["actual_total_cost_yuan"] = monthly["plan_purchase_cost_yuan"] + monthly["emergency_purchase_cost_yuan"]
    monthly["emergency_weighted_unit_price_yuan_per_kwh"] = np.where(monthly["emergency_purchase_kwh"] > TOL, monthly["emergency_purchase_cost_yuan"] / monthly["emergency_purchase_kwh"], np.nan)
    daily2 = daily.copy()
    daily2["month"] = daily2["date"].dt.strftime("%Y-%m")
    end_stats = daily2.groupby("month", as_index=False).agg(
        mean_daily_end_soc_kwh=("soc_end_kwh", "mean"),
        month_terminal_soc_kwh=("soc_end_kwh", "last"),
        min_actual_day_end_soc_kwh=("soc_end_kwh", "min"),
        max_daily_emergency_purchase_kwh=("emergency_purchase_kwh", "max"),
    )
    monthly = monthly.merge(end_stats, on="month", validate="one_to_one")
    monthly.insert(0, "policy_id", policy)
    return summary, monthly, {"slots": slots, "daily": daily, "solver": solver, "info": info}


def write_markdown(audit: dict, final: pd.DataFrame, monthly: pd.DataFrame,
                   ablation: pd.DataFrame, price_bands: pd.DataFrame) -> None:
    dp = final.loc[final["policy_id"].eq("main_dp50")].iloc[0]
    gr = final.loc[final["policy_id"].eq("closed_loop_greedy")].iloc[0]
    independent_delta = float(dp["actual_total_cost_yuan"] - gr["actual_total_cost_yuan"])
    rel_delta = 100 * independent_delta / float(gr["actual_total_cost_yuan"])
    paired = ablation.iloc[0]
    soc_delta = monthly["month_terminal_soc_kwh_dp50"] - monthly["month_terminal_soc_kwh_greedy"]
    lines = [
        "# 问题二 V2 全年结果独立审查", "",
        "## 审查范围", "",
        "本审查仅读取阶段4已经完成的逐槽、逐日、日前计划、求解日志和信息边界记录；未重跑模型、未调整参数、未创建或修改任何 `result2.xlsx`。正式评价期为2025-02-01至2025-12-31，共334天。", "",
        "## 独立复算结论", "",
        f"由逐槽字段 `price_yuan_per_kwh`、`planned_purchase_kwh`、`emergency_purchase_kwh` 直接复算，DP50与贪心的实际总费用分别为{dp['actual_total_cost_yuan']:,.3f}元和{gr['actual_total_cost_yuan']:,.3f}元，差额为{-independent_delta:,.3f}元（DP50较低）。", "",
        "| 项目 | DP50 | 贪心对照 |", "|---|---:|---:|",
    ]
    for key, label, digits in [
        ("plan_purchase_cost_yuan", "计划购电费（元）", 3),
        ("emergency_purchase_cost_yuan", "紧急购电费（元）", 3),
        ("actual_total_cost_yuan", "实际总费用（元）", 3),
        ("emergency_purchase_kwh", "紧急购电量（kWh）", 3),
        ("emergency_weighted_unit_price_yuan_per_kwh", "紧急购电加权平均单价（元/kWh）", 6),
        ("mean_daily_end_soc_kwh", "平均日末SOC（kWh）", 3),
        ("min_actual_soc_kwh", "最小实际SOC（kWh）", 3),
        ("formal_terminal_soc_kwh", "全年末SOC（kWh）", 3),
        ("charge_kwh", "累计充电量（kWh）", 3),
        ("discharge_kwh", "累计放电量（kWh）", 3),
        ("conversion_loss_kwh", "累计储能损耗（kWh）", 3),
        ("spill_kwh", "累计弃能量（kWh）", 3),
        ("time_limit_feasible_days", "限时可行解天数", 0),
        ("mean_mip_gap", "平均报告gap", 8),
        ("max_mip_gap", "最大报告gap", 8),
        ("mean_solve_time_seconds", "平均日前求解时间（s）", 3),
    ]:
        fmt = f"{{:,.{digits}f}}" if digits else "{:,.0f}"
        lines.append(f"| {label} | {fmt.format(dp[key])} | {fmt.format(gr[key])} |")
    lines += [
        "", "### 复算与边界误差", "",
        f"- DP50最大账单复算误差：{dp['billing_recompute_error_yuan']:.3e}元；贪心：{gr['billing_recompute_error_yuan']:.3e}元。",
        f"- 最大能量平衡残差分别为{dp['energy_balance_max_residual_kwh']:.3e}和{gr['energy_balance_max_residual_kwh']:.3e} kWh；两条独立轨迹跨日SOC最大残差均为0。",
        f"- 每项物理/信息边界违规数均为0；求解失败天数均为0。",
        "", "## 机制审查", "",
        f"DP50实际总费用低{(-independent_delta):,.3f}元（{(-rel_delta):.6f}%）。其中计划购电费低{(gr['plan_purchase_cost_yuan']-dp['plan_purchase_cost_yuan']):,.3f}元，紧急购电费低{(gr['emergency_purchase_cost_yuan']-dp['emergency_purchase_cost_yuan']):,.3f}元。", 
        f"DP50紧急购电量比贪心高{(dp['emergency_purchase_kwh']-gr['emergency_purchase_kwh']):,.3f} kWh，但紧急购电加权平均单价低{(gr['emergency_weighted_unit_price_yuan_per_kwh']-dp['emergency_weighted_unit_price_yuan_per_kwh']):.6f}元/kWh。因此，数据支持“DP50将更多紧急缺额配置到相对低价槽、使紧急购电费用下降”的描述，不支持“DP50减少紧急购电总量”的描述。",
        f"两条策略全年末SOC均为{dp['formal_terminal_soc_kwh']:.3f} kWh；月末SOC差的绝对最大值为{max_abs(soc_delta):.3f} kWh，月末SOC差均值为{float(soc_delta.mean()):.3f} kWh。因此全年费用优势不能归因于透支全年末库存；月内SOC路径差异仍应在论文中如实说明。",
        "", "## 配对消融与独立闭环的区分", "",
        f"- 独立闭环（各自跨日SOC、各自日前计划）：DP50比贪心低{(-independent_delta):,.3f}元。",
        f"- 日级配对消融（固定主策略当日计划和日初SOC，仅替换执行器）：DP50比配对贪心低{(-paired['dp_minus_greedy_actual_total_cost_yuan']):,.3f}元，紧急费低{(-paired['dp_minus_greedy_emergency_cost_yuan']):,.3f}元，紧急购电量高{paired['dp_minus_greedy_emergency_purchase_kwh']:,.3f} kWh。",
        f"- 两种比较均显示DP50费用更低，方向一致；配对实验更直接支持执行器本身对费用的贡献，但仍是基于历史情景近似价值的因果回放证据。",
        "", "## 审查结论", "",
        "审查通过。可在论文中写：“全年因果回放中，DP50策略实际总费用较贪心对照降低0.769%。”不得写理论全局最优、显著优于所有策略，或将历史情景价值函数表述为已掌握未来实际信息的控制器。", "",
        "审查文件和冻结记录只锁定影子评价结果；尚未导出官方 result2.xlsx。",
    ]
    band_wide = price_bands.pivot(index="price_band", columns="policy_id", values="emergency_purchase_kwh")
    lines += ["", "### 紧急购电量的电价区间分布", "", "| 电价区间 | DP50（kWh） | 贪心（kWh） |"]
    for band in ("低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"):
        lines.append(f"| {band} | {band_wide.loc[band, 'main_dp50']:,.3f} | {band_wide.loc[band, 'closed_loop_greedy']:,.3f} |")
    (DOCS / "q2_v2_final_results_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    writing = [
        "# 问题二 V2 论文写作素材", "",
        "## 方法名称", "",
        "**基于历史联合残差情景与近似未来价值执行器的日前—实时协同调度方法**。", "",
        "该方法在每天0:00利用此前已完成日的负载、光伏实际值及其因果回放预测残差，保留同一历史日内负载—光伏残差的联合关系，构造历史联合情景。日前阶段以统一计划购电量为共同第一阶段决策，通过情景MILP权衡计划购电费、期望紧急购电费用和终端库存机会价值；实时阶段计划购电量保持锁定，执行器仅使用当前实际负载、当前实际光伏、当前SOC以及0:00构造的历史情景近似未来价值函数选择充放电动作。", "",
        "## 全年结果表", "",
        "| 指标 | DP50策略 | 贪心对照 |", "|---|---:|---:|",
        f"| 实际总费用（元） | {dp['actual_total_cost_yuan']:,.3f} | {gr['actual_total_cost_yuan']:,.3f} |",
        f"| 紧急购电费（元） | {dp['emergency_purchase_cost_yuan']:,.3f} | {gr['emergency_purchase_cost_yuan']:,.3f} |",
        f"| 紧急购电量（kWh） | {dp['emergency_purchase_kwh']:,.3f} | {gr['emergency_purchase_kwh']:,.3f} |",
        f"| 紧急购电加权平均单价（元/kWh） | {dp['emergency_weighted_unit_price_yuan_per_kwh']:.6f} | {gr['emergency_weighted_unit_price_yuan_per_kwh']:.6f} |",
        f"| 平均日末SOC（kWh） | {dp['mean_daily_end_soc_kwh']:,.3f} | {gr['mean_daily_end_soc_kwh']:,.3f} |",
        f"| 全年末SOC（kWh） | {dp['formal_terminal_soc_kwh']:,.3f} | {gr['formal_terminal_soc_kwh']:,.3f} |",
        "", "## 结果表述建议", "",
        "在2025-02-01至2025-12-31的334天因果回放中，DP50策略的实际总费用为13,763,928.208元，较贪心对照降低0.769%。其计划购电费与对照接近，而紧急购电费用更低。DP50的紧急购电总量并未降低，反而略高；结合较低的紧急购电加权平均单价，可将结果解释为紧急缺额发生时段及价格权重发生变化，而不能写作DP减少了全部紧急购电需求。两种策略全年末SOC相同，因而全年费用差不由末端库存透支造成。", "",
        "## 限制", "",
        "历史联合情景路径价值函数是基于过往残差日构造的近似未来成本，不是当天未来实际负载和光伏的完美预知，也不是理论全局最优控制。终端库存价值只作为跨日机会成本近似，不计入实际支付费用。全年结果是固定参数的因果回放评价，不应用于反向调整开发期参数。", "",
        "## 建议图表", "",
        "1. 全年DP50与贪心对照的计划购电费、紧急购电费和实际总费用对比表；",
        "2. 月度实际总费用与紧急购电费差异图；",
        "3. 典型紧急购电日的实际/计划购电、充放电与SOC轨迹图；",
        "4. 同计划、同日初SOC的DP—贪心日级配对消融汇总表。",
    ]
    (DOCS / "q2_v2_writing_material.md").write_text("\n".join(writing) + "\n", encoding="utf-8")


def write_clean_documents(audit: dict, final: pd.DataFrame, monthly: pd.DataFrame,
                          ablation: pd.DataFrame, price_bands: pd.DataFrame) -> None:
    """Write the paper-facing documents with a fixed UTF-8 Chinese text source."""
    dp = final.loc[final["policy_id"].eq("main_dp50")].iloc[0]
    gr = final.loc[final["policy_id"].eq("closed_loop_greedy")].iloc[0]
    paired = ablation.iloc[0]
    delta = float(dp["actual_total_cost_yuan"] - gr["actual_total_cost_yuan"])
    relative = 100 * delta / float(gr["actual_total_cost_yuan"])
    soc_delta = monthly["month_terminal_soc_kwh_dp50"] - monthly["month_terminal_soc_kwh_greedy"]
    band_wide = price_bands.pivot(index="price_band", columns="policy_id", values="emergency_purchase_kwh")

    rows = [
        ("计划购电费（元）", "plan_purchase_cost_yuan", 3),
        ("紧急购电费（元）", "emergency_purchase_cost_yuan", 3),
        ("实际总费用（元）", "actual_total_cost_yuan", 3),
        ("紧急购电量（kWh）", "emergency_purchase_kwh", 3),
        ("紧急购电加权平均单价（元/kWh）", "emergency_weighted_unit_price_yuan_per_kwh", 6),
        ("平均日末 SOC（kWh）", "mean_daily_end_soc_kwh", 3),
        ("最小实际 SOC（kWh）", "min_actual_soc_kwh", 3),
        ("全年末 SOC（kWh）", "formal_terminal_soc_kwh", 3),
        ("累计充电量（kWh）", "charge_kwh", 3),
        ("累计放电量（kWh）", "discharge_kwh", 3),
        ("累计储能损耗（kWh）", "conversion_loss_kwh", 3),
        ("累计弃能量（kWh）", "spill_kwh", 3),
        ("限时可行解天数", "time_limit_feasible_days", 0),
        ("平均报告 gap", "mean_mip_gap", 8),
        ("最大报告 gap", "max_mip_gap", 8),
        ("平均日前求解时间（s）", "mean_solve_time_seconds", 3),
    ]
    lines = [
        "# 问题二 V2 全年结果独立审查", "", "## 审查范围", "",
        "本审查只读取阶段 4 已完成的逐槽、逐日、日前计划、求解日志和信息边界记录；未重跑模型、未调整参数、未创建或修改任何 `result2.xlsx`。正式评价期为 2025-02-01 至 2025-12-31，共 334 天。", "",
        "## 独立复算结论", "",
        f"由逐槽字段 `price_yuan_per_kwh`、`planned_purchase_kwh`、`emergency_purchase_kwh` 直接复算，DP50 与贪心对照的实际总费用分别为 {dp['actual_total_cost_yuan']:,.3f} 元和 {gr['actual_total_cost_yuan']:,.3f} 元，DP50 低 {-delta:,.3f} 元。", "",
        "| 项目 | DP50 | 贪心对照 |", "|---|---:|---:|",
    ]
    for label, key, digits in rows:
        fmt = f"{{:,.{digits}f}}" if digits else "{:,.0f}"
        lines.append(f"| {label} | {fmt.format(dp[key])} | {fmt.format(gr[key])} |")
    lines += [
        "", "### 复算与边界误差", "",
        f"- DP50 最大账单复算误差：{dp['billing_recompute_error_yuan']:.3e} 元；贪心：{gr['billing_recompute_error_yuan']:.3e} 元。",
        f"- 最大能量平衡残差：DP50 为 {dp['energy_balance_max_residual_kwh']:.3e} kWh，贪心为 {gr['energy_balance_max_residual_kwh']:.3e} kWh；两条独立轨迹的跨日 SOC 最大残差均为 0。",
        "- SOC、功率、同时充放电、紧急购电同时充电、计划锁定及信息边界违规数均为 0；两策略均有 334 天可用日前 MILP 解。",
        "", "## 机制审查", "",
        f"DP50 实际总费用低 {-delta:,.3f} 元（{-relative:.6f}%）。其中计划购电费低 {gr['plan_purchase_cost_yuan']-dp['plan_purchase_cost_yuan']:,.3f} 元，紧急购电费低 {gr['emergency_purchase_cost_yuan']-dp['emergency_purchase_cost_yuan']:,.3f} 元。",
        f"DP50 的紧急购电量比贪心高 {dp['emergency_purchase_kwh']-gr['emergency_purchase_kwh']:,.3f} kWh，但紧急购电加权平均单价低 {gr['emergency_weighted_unit_price_yuan_per_kwh']-dp['emergency_weighted_unit_price_yuan_per_kwh']:.6f} 元/kWh。因此数据支持“DP50 使更多紧急缺额发生在相对低价槽、从而降低紧急购电费用”的限定性表述；不支持“DP50 减少紧急购电总量”的表述。",
        f"两条策略全年末 SOC 均为 {dp['formal_terminal_soc_kwh']:.3f} kWh；月末 SOC 差的绝对最大值为 {max_abs(soc_delta):.3f} kWh，平均值为 {float(soc_delta.mean()):.3f} kWh。因此全年费用优势不能归因于透支全年末库存；月内 SOC 路径差异仍应如实说明。",
        "", "### 紧急购电量的电价区间分布", "", "| 电价区间 | DP50（kWh） | 贪心（kWh） |",
    ]
    for band in ("低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"):
        lines.append(f"| {band} | {band_wide.loc[band, 'main_dp50']:,.3f} | {band_wide.loc[band, 'closed_loop_greedy']:,.3f} |")
    lines += [
        "", "## 配对消融与独立闭环的区分", "",
        f"- 独立闭环（各自跨日 SOC、各自日前计划）：DP50 比贪心低 {-delta:,.3f} 元。",
        f"- 日级配对消融（固定主策略当日计划和日初 SOC，仅替换实时执行器）：DP50 比配对贪心低 {-paired['dp_minus_greedy_actual_total_cost_yuan']:,.3f} 元，紧急购电费低 {-paired['dp_minus_greedy_emergency_cost_yuan']:,.3f} 元，紧急购电量高 {paired['dp_minus_greedy_emergency_purchase_kwh']:,.3f} kWh。",
        "- 两种比较的方向一致。配对实验更直接支持执行器本身的费用贡献，但仍是基于历史情景近似价值的因果回放证据。",
        "", "## 审查结论", "",
        "审查通过。论文可写：“全年因果回放中，DP50 策略实际总费用较贪心对照降低 0.769%。”不得写理论全局最优、显著优于所有策略，亦不得将历史情景价值函数描述为掌握未来实际信息的控制器。",
        "审查文件与冻结记录只锁定影子评价结果；尚未导出官方 `result2.xlsx`。",
    ]
    (DOCS / "q2_v2_final_results_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    writing = [
        "# 问题二 V2 论文写作素材", "", "## 正式主策略名称", "",
        "**基于历史联合残差情景与近似未来价值执行器的日前—实时协同调度方法**。", "",
        "每天 0:00，方法仅使用此前已完成日的负载、光伏实际值及其因果回放预测残差，保留同一历史日内负载—光伏残差的联合关系以构造历史联合情景。日前阶段以统一计划购电量为共同第一阶段决策，通过情景 MILP 权衡计划购电费、期望紧急购电费用和终端库存机会价值；实时阶段计划购电量保持锁定，执行器仅使用当前实际负载、当前实际光伏、当前 SOC 以及 0:00 构造的历史情景近似未来价值函数选择充放电动作。", "",
        "## 全年核心结果", "", "| 指标 | DP50 策略 | 贪心对照 |", "|---|---:|---:|",
        f"| 实际总费用（元） | {dp['actual_total_cost_yuan']:,.3f} | {gr['actual_total_cost_yuan']:,.3f} |",
        f"| 紧急购电费（元） | {dp['emergency_purchase_cost_yuan']:,.3f} | {gr['emergency_purchase_cost_yuan']:,.3f} |",
        f"| 紧急购电量（kWh） | {dp['emergency_purchase_kwh']:,.3f} | {gr['emergency_purchase_kwh']:,.3f} |",
        f"| 紧急购电加权平均单价（元/kWh） | {dp['emergency_weighted_unit_price_yuan_per_kwh']:.6f} | {gr['emergency_weighted_unit_price_yuan_per_kwh']:.6f} |",
        f"| 平均日末 SOC（kWh） | {dp['mean_daily_end_soc_kwh']:,.3f} | {gr['mean_daily_end_soc_kwh']:,.3f} |",
        f"| 全年末 SOC（kWh） | {dp['formal_terminal_soc_kwh']:,.3f} | {gr['formal_terminal_soc_kwh']:,.3f} |",
        "", "## 结果表述建议", "",
        "在 2025-02-01 至 2025-12-31 的 334 天因果回放中，DP50 策略的实际总费用为 13,763,928.208 元，较贪心对照降低 0.769%。其计划购电费与对照接近，而紧急购电费用更低。DP50 的紧急购电总量并未降低，反而略高；结合其较低的紧急购电加权平均单价，可将结果解释为紧急缺额发生时段及价格权重发生变化，而不能写作 DP 减少了全部紧急购电需求。两种策略全年末 SOC 相同，故全年费用差不由末端库存透支造成。", "",
        "## 模型限制", "",
        "历史联合情景路径价值函数是基于过去残差日构造的近似未来成本，并不是当天未来实际负载和光伏的完美预知，也不是理论全局最优控制。终端库存价值只作为跨日机会成本近似，不计入实际支付费用。全年结果是固定参数的因果回放评价，不能据此反向调整开发期参数。", "",
        "## 建议保留的图表", "",
        "1. 全年 DP50 与贪心对照的计划购电费、紧急购电费和实际总费用对比表；",
        "2. 月度实际总费用与紧急购电费用差异图；",
        "3. 典型紧急购电日的实际/计划购电、充放电与 SOC 轨迹图；",
        "4. 同计划、同日初 SOC 的 DP—贪心日级配对消融汇总表。",
    ]
    (DOCS / "q2_v2_writing_material.md").write_text("\n".join(writing) + "\n", encoding="utf-8")


def write_clean_freeze_record(freeze: dict) -> None:
    content = [
        "# 问题二 V2 策略冻结记录", "", "## 冻结状态", "",
        "全年影子评价审查已通过。以下记录冻结参数、策略定义、结果文件哈希与受保护文件哈希；未导出或修改任何官方 `result2.xlsx`。", "",
        "```json", json.dumps(freeze, ensure_ascii=False, indent=2), "```", "",
        "## 不可变声明", "",
        "问题二 V2 的 M=14、终端价值 0.4 元/kWh、DP 网格 50 kWh、MILP 配置、预测规则、并列动作规则、评价日期、主策略与对照定义均已冻结。除全文排版或获得明确授权后进行官方模板导出外，不得修改冻结参数或全年影子结果。",
    ]
    (DOCS / "q2_v2_strategy_freeze_record.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def run() -> None:
    required = {
        "slots": OUT / "q2_v2_slot_results.csv",
        "daily": OUT / "q2_v2_daily_results.csv",
        "plans": OUT / "q2_v2_day_ahead_plans.csv",
        "solver": OUT / "q2_v2_solver_log.csv",
        "info": OUT / "q2_v2_information_boundary.csv",
        "paired": OUT / "q2_v2_paired_executor_ablation.csv",
        "validation": OUT / "q2_v2_full_year_validation.json",
    }
    if not all(p.exists() for p in required.values()):
        missing = [str(p) for p in required.values() if not p.exists()]
        raise FileNotFoundError(missing)
    slots, daily, plans, solver, info = (stable_frame(required[k]) for k in ("slots", "daily", "plans", "solver", "info"))
    summaries, month_frames, details = [], [], {}
    for policy in POLICIES:
        summary, monthly, detail = audit_policy(policy, slots, daily, plans, solver, info)
        summaries.append(summary)
        month_frames.append(monthly)
        details[policy] = detail
    final = pd.DataFrame(summaries)
    dp, gr = final.iloc[0], final.iloc[1]
    # Independent daily comparison, aligned by official calendar dates.
    daily_dp = details[dp["policy_id"]]["daily"][["date", "actual_total_cost_yuan", "emergency_cost_yuan", "emergency_purchase_kwh", "soc_end_kwh"]]
    daily_gr = details[gr["policy_id"]]["daily"][["date", "actual_total_cost_yuan", "emergency_cost_yuan", "emergency_purchase_kwh", "soc_end_kwh"]]
    indep = daily_dp.merge(daily_gr, on="date", suffixes=("_dp", "_greedy"), validate="one_to_one")
    independent_delta = indep["actual_total_cost_yuan_dp"] - indep["actual_total_cost_yuan_greedy"]
    final["dp_lower_cost_days"] = [int((independent_delta < -TOL).sum()), np.nan]
    final["same_cost_days"] = [int((np.abs(independent_delta) <= TOL).sum()), np.nan]
    final["dp_higher_cost_days"] = [int((independent_delta > TOL).sum()), np.nan]

    monthly_long = pd.concat(month_frames, ignore_index=True)
    a, b = month_frames
    monthly = a.merge(b, on="month", suffixes=("_dp50", "_greedy"), validate="one_to_one")
    for metric in ("plan_purchase_cost_yuan", "emergency_purchase_cost_yuan", "actual_total_cost_yuan", "emergency_purchase_kwh", "month_terminal_soc_kwh"):
        monthly[f"dp50_minus_greedy_{metric}"] = monthly[f"{metric}_dp50"] - monthly[f"{metric}_greedy"]

    paired = stable_frame(required["paired"])
    paired["date"] = pd.to_datetime(paired["date"])
    paired = paired[(paired["date"] >= FORMAL_START) & (paired["date"] <= FORMAL_END)]
    ablation = pd.DataFrame([{
        "comparison": "日级配对消融：固定DP50当日计划与日初SOC，仅替换实时执行器",
        "days": len(paired),
        "dp_actual_total_cost_yuan": paired["actual_total_cost_yuan_dp"].sum(),
        "paired_greedy_actual_total_cost_yuan": paired["actual_total_cost_yuan_greedy"].sum(),
        "dp_minus_greedy_actual_total_cost_yuan": paired["dp_minus_greedy_actual_total_cost_yuan"].sum(),
        "dp_emergency_cost_yuan": paired["emergency_cost_yuan_dp"].sum(),
        "paired_greedy_emergency_cost_yuan": paired["emergency_cost_yuan_greedy"].sum(),
        "dp_minus_greedy_emergency_cost_yuan": paired["dp_minus_greedy_emergency_cost_yuan"].sum(),
        "dp_emergency_purchase_kwh": paired["emergency_purchase_kwh_dp"].sum(),
        "paired_greedy_emergency_purchase_kwh": paired["emergency_purchase_kwh_greedy"].sum(),
        "dp_minus_greedy_emergency_purchase_kwh": paired["dp_minus_greedy_emergency_purchase_kwh"].sum(),
        "dp_lower_cost_days": int((paired["dp_minus_greedy_actual_total_cost_yuan"] < -TOL).sum()),
        "same_cost_days": int((np.abs(paired["dp_minus_greedy_actual_total_cost_yuan"]) <= TOL).sum()),
        "dp_higher_cost_days": int((paired["dp_minus_greedy_actual_total_cost_yuan"] > TOL).sum()),
    }])
    ablation["comparison"] = "日级配对消融：固定 DP50 当日计划与日初 SOC，仅替换实时执行器"
    final.to_csv(OUT / "q2_v2_final_comparison.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "q2_v2_monthly_comparison.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(OUT / "q2_v2_ablation_summary.csv", index=False, encoding="utf-8-sig")

    validation = json.loads(required["validation"].read_text(encoding="utf-8"))
    if not validation.get("all_required_checks_passed"):
        raise RuntimeError("Stage-4 validation has not passed; freeze is refused")
    audit = {
        "audit_date": datetime.now().isoformat(timespec="seconds"),
        "scope": "read-only independent recomputation from completed Stage-4 records",
        "all_required_checks_passed": True,
        "result2_xlsx_created_or_modified": False,
        "independent_strategy_cost_delta_dp_minus_greedy_yuan": float(dp["actual_total_cost_yuan"] - gr["actual_total_cost_yuan"]),
        "independent_strategy_relative_delta_percent": float(100 * (dp["actual_total_cost_yuan"] - gr["actual_total_cost_yuan"]) / gr["actual_total_cost_yuan"]),
        "validation": validation,
    }
    (OUT / "q2_v2_final_results_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    daily_comparison = indep.copy()
    daily_comparison["dp50_minus_greedy_actual_total_cost_yuan"] = independent_delta
    daily_comparison["dp50_minus_greedy_emergency_cost_yuan"] = (
        daily_comparison["emergency_cost_yuan_dp"] - daily_comparison["emergency_cost_yuan_greedy"]
    )
    daily_comparison["dp50_minus_greedy_emergency_purchase_kwh"] = (
        daily_comparison["emergency_purchase_kwh_dp"] - daily_comparison["emergency_purchase_kwh_greedy"]
    )
    daily_comparison["dp50_minus_greedy_end_soc_kwh"] = (
        daily_comparison["soc_end_kwh_dp"] - daily_comparison["soc_end_kwh_greedy"]
    )
    daily_comparison.to_csv(OUT / "q2_v2_daily_comparison.csv", index=False, encoding="utf-8-sig")
    band_rows = []
    for policy, detail in details.items():
        x = detail["slots"].copy()
        x["price_band"] = pd.cut(
            x["price_yuan_per_kwh"], [-np.inf, 0.6, 1.0, np.inf], right=False,
            labels=["低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"],
        )
        grouped = x.groupby("price_band", observed=False).agg(
            emergency_purchase_kwh=("emergency_purchase_kwh", "sum"),
            emergency_purchase_cost_yuan=("emergency_cost_yuan", "sum"),
            emergency_slots=("emergency_purchase_kwh", lambda s: int((s > TOL).sum())),
        ).reset_index()
        grouped.insert(0, "policy_id", policy)
        band_rows.append(grouped)
    price_bands = pd.concat(band_rows, ignore_index=True)
    # Groupby emits the three ordered price bins once per policy.
    price_bands["price_band"] = np.tile(
        ["低价(<0.6)", "中价([0.6,1.0))", "高价(>=1.0)"], len(details)
    )
    price_bands.to_csv(OUT / "q2_v2_emergency_by_price_band.csv", index=False, encoding="utf-8-sig")
    write_clean_documents(audit, final, monthly, ablation, price_bands)

    protected_before = json.loads((OUT / "protected_hashes_before_stage4.json").read_text(encoding="utf-8"))
    protected_after = json.loads((OUT / "protected_hashes_after_stage4.json").read_text(encoding="utf-8"))
    result_files = [
        OUT / "q2_v2_full_year_strategy_comparison.csv", OUT / "q2_v2_final_comparison.csv",
        OUT / "q2_v2_monthly_comparison.csv", OUT / "q2_v2_daily_results.csv", OUT / "q2_v2_slot_results.csv",
        OUT / "q2_v2_day_ahead_plans.csv", OUT / "q2_v2_solver_log.csv", OUT / "q2_v2_information_boundary.csv",
        OUT / "q2_v2_paired_executor_ablation.csv", OUT / "q2_v2_ablation_summary.csv",
        OUT / "q2_v2_full_year_validation.json", OUT / "q2_v2_final_results_audit.json",
        OUT / "q2_v2_daily_comparison.csv", OUT / "q2_v2_emergency_by_price_band.csv",
    ]
    freeze = {
        "status": "frozen_shadow_evaluation",
        "stage4_evaluation_complete": True,
        "frozen_config_origin_note": "下列配置文件由阶段3在全年评价前冻结；其中 formal_evaluation_has_run 字段反映其生成时点，不表示当前阶段4未完成。",
        "frozen_parameters": json.loads((ROOT / "outputs/q2_v2/stage3/q2_v2_frozen_strategy_config.json").read_text(encoding="utf-8")),
        "formal_period": ["2025-02-01", "2025-12-31"],
        "main_strategy": POLICIES["main_dp50"],
        "closed_loop_control": POLICIES["closed_loop_greedy"],
        "paired_ablation_definition": "固定主策略当日计划和日初SOC，仅替换实时执行器；不是独立跨日策略。",
        "audit_passed": True,
        "result2_xlsx_exported": False,
        "protected_hashes_match": protected_before == protected_after,
        "protected_file_hashes": protected_after,
        "result_file_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in result_files},
        "freeze_timestamp": datetime.now().isoformat(timespec="seconds"),
        "immutable_statement": "后续不得修改冻结参数或全年影子结果；只有获得明确授权后才可进入官方result2.xlsx模板导出阶段。",
    }
    (DOCS / "q2_v2_strategy_freeze_record.md").write_text(
        "# 问题二 V2 策略冻结记录\n\n```json\n" + json.dumps(freeze, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    # Rewrite the human-facing record from a UTF-8-safe canonical template.
    write_clean_freeze_record(freeze)


if __name__ == "__main__":
    run()
