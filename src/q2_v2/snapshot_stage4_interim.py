"""Read-only interim snapshot of concurrently running Stage-4 evaluations.

The runner writes one date per file.  This tool determines the common completed
date set from files present in *both* independent closed-loop directories and
never reads a date whose complete six-file record is absent.  It performs no
optimisation and does not alter state/checkpoint/result workbook files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ENERGY_MAX_KWH, ETA_CHARGE, ETA_DISCHARGE, ROOT, SOC_MAX_KWH, SOC_MIN_KWH, TOL


ROOT_OUT = ROOT / "outputs/q2_v2/stage4"
MAIN = ROOT_OUT / "main_dp50"
GREEDY = ROOT_OUT / "closed_loop_greedy"
SNAPSHOT_DIR = ROOT_OUT / "snapshots"
FORMAL_START = pd.Timestamp("2025-02-01")


def _available_dates(policy_dir: Path) -> set[str]:
    """A date is eligible only after all runner-produced records exist."""
    needed = ("daily", "slots", "plans", "solver", "information")
    out: set[str] | None = None
    for part in needed:
        dates = {p.stem for p in (policy_dir / part).glob("*.csv")}
        out = dates if out is None else out & dates
    return out or set()


def _read(policy_dir: Path, section: str, dates: list[str]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(policy_dir / section / f"{date}.csv") for date in dates],
        ignore_index=True,
    )


def _policy_summary(policy: str, daily: pd.DataFrame, slots: pd.DataFrame, solver: pd.DataFrame) -> dict:
    emergency_kwh = float(slots["emergency_purchase_kwh"].sum())
    emergency_cost = float(slots["emergency_cost_yuan"].sum())
    invalid_terms = ~solver["termination_condition"].astype(str).isin(("optimal", "maxTimeLimit", "feasible"))
    return {
        "strategy": policy,
        "days": int(daily.shape[0]),
        "plan_purchase_cost_yuan": float(slots["plan_cost_yuan"].sum()),
        "emergency_purchase_cost_yuan": emergency_cost,
        "actual_total_cost_yuan": float(slots["actual_cost_yuan"].sum()),
        "emergency_purchase_kwh": emergency_kwh,
        "emergency_weighted_unit_price_yuan_per_kwh": emergency_cost / emergency_kwh if emergency_kwh > TOL else np.nan,
        "mean_daily_end_soc_kwh": float(daily["soc_end_kwh"].mean()),
        "actual_soc_min_kwh": float(slots[["soc_start_kwh", "soc_end_kwh"]].min().min()),
        "common_period_terminal_soc_kwh": float(daily.iloc[-1]["soc_end_kwh"]),
        "solver_failure_days": int(invalid_terms.sum()),
        "time_limit_feasible_days": int(solver["termination_condition"].astype(str).eq("maxTimeLimit").sum()),
        "certified_optimal_days": int(solver["optimal_solution_certified"].fillna(False).astype(bool).sum()),
    }


def _violation_counts(slots: pd.DataFrame, info: pd.DataFrame) -> dict:
    soc = slots[["soc_start_kwh", "soc_end_kwh"]]
    expected_next = (
        slots["soc_start_kwh"].to_numpy(float)
        + ETA_CHARGE * slots["charge_kwh"].to_numpy(float)
        - slots["discharge_kwh"].to_numpy(float) / ETA_DISCHARGE
    )
    transition = np.abs(expected_next - slots["soc_end_kwh"].to_numpy(float))
    return {
        "energy_balance_violation_slots": int((np.abs(slots["balance_residual_kwh"].to_numpy(float)) > TOL).sum()),
        "soc_boundary_violation_values": int(((soc < SOC_MIN_KWH - TOL) | (soc > SOC_MAX_KWH + TOL)).sum().sum()),
        "power_violation_slots": int(((slots["charge_kwh"] > ENERGY_MAX_KWH + TOL) | (slots["discharge_kwh"] > ENERGY_MAX_KWH + TOL)).sum()),
        "simultaneous_charge_discharge_slots": int(((slots["charge_kwh"] > TOL) & (slots["discharge_kwh"] > TOL)).sum()),
        "emergency_while_charging_slots": int(((slots["emergency_purchase_kwh"] > TOL) & (slots["charge_kwh"] > TOL)).sum()),
        "soc_transition_violation_slots": int((transition > TOL).sum()),
        "prediction_information_violation_days": int((~info["all_prediction_sources_prior"].astype(bool)).sum()),
        "scenario_information_violation_days": int((~info["all_scenario_sources_prior"].astype(bool)).sum()),
    }


def _paired_summary(dates: list[str]) -> tuple[dict, pd.DataFrame]:
    # Paired result is created while the main policy runs: same locked daily
    # plan and same daily initial SOC, different executor.  It is not a third
    # cross-day strategy.
    paired_daily = _read(MAIN, "paired_daily", dates)
    paired_slots = _read(MAIN, "paired_slots", dates)
    dp_daily = _read(MAIN, "daily", dates)
    merged = dp_daily[["date", "actual_total_cost_yuan", "emergency_cost_yuan", "emergency_purchase_kwh", "soc_end_kwh"]].merge(
        paired_daily[["date", "actual_total_cost_yuan", "emergency_cost_yuan", "emergency_purchase_kwh", "soc_end_kwh"]],
        on="date", suffixes=("_dp", "_greedy"), validate="one_to_one",
    )
    out = {
        "comparison": "日级配对消融：同计划、同日初SOC；DP执行 vs 贪心执行（非独立跨日策略）",
        "days": int(len(merged)),
        "dp_actual_total_cost_yuan": float(merged["actual_total_cost_yuan_dp"].sum()),
        "paired_greedy_actual_total_cost_yuan": float(merged["actual_total_cost_yuan_greedy"].sum()),
        "dp_minus_paired_greedy_cost_yuan": float((merged["actual_total_cost_yuan_dp"] - merged["actual_total_cost_yuan_greedy"]).sum()),
        "dp_emergency_cost_yuan": float(merged["emergency_cost_yuan_dp"].sum()),
        "paired_greedy_emergency_cost_yuan": float(merged["emergency_cost_yuan_greedy"].sum()),
        "dp_emergency_purchase_kwh": float(merged["emergency_purchase_kwh_dp"].sum()),
        "paired_greedy_emergency_purchase_kwh": float(merged["emergency_purchase_kwh_greedy"].sum()),
        "dp_lower_cost_days": int((merged["actual_total_cost_yuan_dp"] < merged["actual_total_cost_yuan_greedy"] - TOL).sum()),
        "same_cost_days": int((np.abs(merged["actual_total_cost_yuan_dp"] - merged["actual_total_cost_yuan_greedy"]) <= TOL).sum()),
        "dp_higher_cost_days": int((merged["actual_total_cost_yuan_dp"] > merged["actual_total_cost_yuan_greedy"] + TOL).sum()),
        "paired_dp_min_soc_kwh": float(_read(MAIN, "slots", dates)[["soc_start_kwh", "soc_end_kwh"]].min().min()),
        "paired_greedy_min_soc_kwh": float(paired_slots[["soc_start_kwh", "soc_end_kwh"]].min().min()),
    }
    return out, merged


def run() -> Path:
    common = sorted(_available_dates(MAIN) & _available_dates(GREEDY))
    common = [d for d in common if pd.Timestamp(d) >= FORMAL_START]
    if not common:
        raise RuntimeError("No completed common formal dates are available for a snapshot.")
    # Preserve the continuous prefix only: do not allow a late isolated file
    # to appear as a completed evaluation date if an earlier date is absent.
    expected = pd.date_range(common[0], periods=len(common), freq="D").strftime("%Y-%m-%d").tolist()
    if common != expected:
        common = common[:next((i for i, (a, b) in enumerate(zip(common, expected)) if a != b), len(common))]

    main_daily, main_slots, main_solver, main_info = (
        _read(MAIN, part, common) for part in ("daily", "slots", "solver", "information")
    )
    greedy_daily, greedy_slots, greedy_solver, greedy_info = (
        _read(GREEDY, part, common) for part in ("daily", "slots", "solver", "information")
    )
    main_summary = _policy_summary("主策略：联合情景MILP + DP50", main_daily, main_slots, main_solver)
    greedy_summary = _policy_summary("独立贪心对照：联合情景MILP + 贪心执行", greedy_daily, greedy_slots, greedy_solver)
    cost_delta = main_summary["actual_total_cost_yuan"] - greedy_summary["actual_total_cost_yuan"]
    emergency_delta = main_summary["emergency_purchase_kwh"] - greedy_summary["emergency_purchase_kwh"]
    comparison = {
        "snapshot_label": "中间快照，非最终结果",
        "common_start_date": common[0],
        "common_end_date": common[-1],
        "common_days": len(common),
        "dp_minus_greedy_actual_cost_yuan": cost_delta,
        "relative_actual_cost_change_percent": 100 * cost_delta / greedy_summary["actual_total_cost_yuan"],
        "dp_minus_greedy_emergency_purchase_kwh": emergency_delta,
        "relative_emergency_purchase_change_percent": 100 * emergency_delta / greedy_summary["emergency_purchase_kwh"],
        "dp_lower_cost_days": int((main_daily["actual_total_cost_yuan"] < greedy_daily["actual_total_cost_yuan"] - TOL).sum()),
        "same_cost_days": int((np.abs(main_daily["actual_total_cost_yuan"] - greedy_daily["actual_total_cost_yuan"]) <= TOL).sum()),
        "dp_higher_cost_days": int((main_daily["actual_total_cost_yuan"] > greedy_daily["actual_total_cost_yuan"] + TOL).sum()),
    }
    paired, paired_daily_comparison = _paired_summary(common)
    validation = {
        "main": _violation_counts(main_slots, main_info),
        "greedy": _violation_counts(greedy_slots, greedy_info),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    base = SNAPSHOT_DIR / f"q2_v2_stage4_interim_snapshot_{stamp}"
    report = {
        "snapshot_label": "中间快照，非最终结果",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "read_only_source": [str(MAIN.relative_to(ROOT)), str(GREEDY.relative_to(ROOT))],
        "independent_cross_day_strategy_comparison": comparison,
        "main_strategy": main_summary,
        "greedy_strategy": greedy_summary,
        "validation": validation,
        "paired_daily_ablation_separate_from_independent_strategies": paired,
        "result2_xlsx_modified": False,
        "runner_processes_modified": False,
    }
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for section, values in (("独立跨日比较", comparison), ("主策略", main_summary), ("贪心对照", greedy_summary), ("主策略验证", validation["main"]), ("贪心对照验证", validation["greedy"]), ("日级配对消融（另列）", paired)):
        rows.extend({"snapshot_label": "中间快照，非最终结果", "section": section, "metric": k, "value": v} for k, v in values.items())
    pd.DataFrame(rows).to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    paired_daily_comparison.insert(0, "snapshot_label", "中间快照，非最终结果")
    paired_daily_comparison.to_csv(base.parent / f"{base.name}_paired_daily.csv", index=False, encoding="utf-8-sig")
    lines = ["# 问题二 V2 阶段4中间快照（非最终结果）", "", "此文件仅读取两个仍在运行的策略已完整落盘的共同日期；不包含任一策略独有的后续日期。", "", "## 独立跨日策略共同区间", "", f"- 日期：{common[0]} 至 {common[-1]}，共 {len(common)} 天。", ""]
    lines += ["| 指标 | 主策略 DP50 | 独立贪心对照 |", "|---|---:|---:|"]
    for key, label in (("plan_purchase_cost_yuan", "计划购电费（元）"), ("emergency_purchase_cost_yuan", "紧急购电费（元）"), ("actual_total_cost_yuan", "实际总费用（元）"), ("emergency_purchase_kwh", "紧急购电量（kWh）"), ("emergency_weighted_unit_price_yuan_per_kwh", "紧急购电加权平均单价（元/kWh）"), ("mean_daily_end_soc_kwh", "平均日末SOC（kWh）"), ("actual_soc_min_kwh", "最小实际SOC（kWh）"), ("common_period_terminal_soc_kwh", "共同区间末端SOC（kWh）"), ("solver_failure_days", "求解失败天数"), ("time_limit_feasible_days", "限时可行解天数")):
        lines.append(f"| {label} | {main_summary[key]:.6f} | {greedy_summary[key]:.6f} |" if isinstance(main_summary[key], float) else f"| {label} | {main_summary[key]} | {greedy_summary[key]} |")
    lines += ["", f"- DP50 − 贪心的实际总费用：{cost_delta:.6f} 元（{comparison['relative_actual_cost_change_percent']:.6f}%）。", f"- DP50 − 贪心的紧急购电量：{emergency_delta:.6f} kWh（{comparison['relative_emergency_purchase_change_percent']:.6f}%）。", f"- 日费用比较：DP50更低 {comparison['dp_lower_cost_days']} 天，相同 {comparison['same_cost_days']} 天，更高 {comparison['dp_higher_cost_days']} 天。", "", "## 物理与信息边界核验（共同区间）", ""]
    for name, vals in validation.items():
        lines.append(f"- {name}：" + "；".join(f"{k}={v}" for k, v in vals.items()) + "。")
    lines += ["", "## 日级配对消融（与独立跨日比较分开）", "", "同一日的主策略计划购电量与日初SOC固定，仅替换实时执行器；该实验不构成第三条跨日递推策略。", ""]
    lines += [f"- {k}：{v}" for k, v in paired.items()]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base


if __name__ == "__main__":
    print(run())
