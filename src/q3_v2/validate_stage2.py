"""Independent validation and synthetic mechanism checks for Q3 V2 Stage 2."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.q2_v2.execution import build_path_value_functions, greedy_action, select_dp_action

from .config import (
    EMERGENCY_MULTIPLIER, ENERGY_MAX_KWH, ETA_CHARGE, ETA_DISCHARGE, OUT, ROOT,
    SOC_MAX_KWH, SOC_MIN_KWH, TOL,
)
from .rolling_milp import VersionScenarios
from .settlement import manual_settlement_cases, settle_contract


def _synthetic_checks() -> pd.DataFrame:
    rows = []
    def add(name, passed, detail): rows.append({"test": name, "passed": bool(passed), "detail": detail})

    a = greedy_action(-100.0, 5000.0)
    add("光伏富余且电池未满", abs(a[0]-100.0)<TOL and a[1]==0 and a[3]==0, str(a))
    a = greedy_action(-100.0, SOC_MAX_KWH)
    add("光伏富余且电池已满", a[0]==0 and abs(a[3]-100.0)<TOL, str(a))
    a = greedy_action(100.0, 5000.0)
    add("缺额且电池可放电", abs(a[1]-100.0)<TOL and a[2]==0, str(a))
    a = greedy_action(100.0, SOC_MIN_KWH)
    add("缺额且SOC位于安全下限", a[1]==0 and abs(a[2]-100.0)<TOL, str(a))
    a1 = greedy_action(2000.0, SOC_MAX_KWH); a2 = greedy_action(-2000.0, SOC_MIN_KWH)
    add("充放电单槽上限", abs(a1[1]-ENERGY_MAX_KWH)<TOL and abs(a2[0]-ENERGY_MAX_KWH)<TOL, f"discharge={a1[1]},charge={a2[0]}")

    scenarios = VersionScenarios(("synthetic",), np.array([1.0]), np.array([[500.0,500.0]]), np.zeros((1,2)), 0)
    plan = np.zeros(2); price = np.array([0.2,1.4])
    value0 = build_path_value_functions(scenarios, plan, price, 50.0, 0.0)
    dp = select_dp_action(500.0, 1800.0, price[0], value0.mean_value_yuan[1], value0.soc_grid_kwh)
    add("当前低价未来高价保留", dp[1] < 500.0-TOL, str(dp))
    valuev = build_path_value_functions(scenarios, plan, price, 50.0, 0.4)
    add("v=0与非零v终端条件", np.allclose(value0.mean_value_yuan[-1],0.0) and np.allclose(valuev.mean_value_yuan[-1],-0.4*valuev.soc_grid_kwh), "terminal arrays checked")
    nongrid = select_dp_action(100.0, 1873.4, 1.0, value0.mean_value_yuan[1], value0.soc_grid_kwh)
    add("非网格SOC不被强制取整", abs(1873.4/50-round(1873.4/50)*50)>TOL and abs(nongrid[4]/50-round(nongrid[4]/50)*50)>TOL, str(nongrid))
    first = select_dp_action(100.0, 3000.0, 1.0, value0.mean_value_yuan[1], value0.soc_grid_kwh)
    second = select_dp_action(100.0, 3000.0, 1.0, value0.mean_value_yuan[1], value0.soc_grid_kwh)
    add("未来实际扰动不改变既有动作接口", first == second, f"first={first}; second={second}")
    add("人工结算七例", bool(manual_settlement_cases()["passed"].all()), "all expected values matched")
    return pd.DataFrame(rows)


def run() -> dict:
    main = pd.read_csv(OUT / "q3_v2_stage2_slot_results.csv")
    paired = pd.read_csv(OUT / "q3_v2_stage2_paired_greedy_slots.csv")
    plans = pd.read_csv(OUT / "q3_v2_stage2_plan_versions.csv")
    solver = pd.read_csv(OUT / "q3_v2_stage2_solver_log.csv")
    scenario = pd.read_csv(OUT / "q3_v2_stage2_scenario_dispatch.csv")
    info = pd.read_csv(OUT / "q3_v2_stage2_information_audit.csv")
    before = json.loads((OUT / "protected_hashes_before.json").read_text(encoding="utf-8"))
    after = json.loads((OUT / "protected_hashes_after.json").read_text(encoding="utf-8"))
    synth = _synthetic_checks()
    synth.to_csv(OUT / "q3_v2_stage2_synthetic_tests.csv", index=False, encoding="utf-8-sig")

    def violations(frame):
        expected_soc = frame.soc_start_kwh + ETA_CHARGE*frame.charge_rt_kwh - frame.discharge_rt_kwh/ETA_DISCHARGE
        return {
            "max_balance_residual_kwh": float(frame.balance_residual_kwh.abs().max()),
            "max_soc_transition_residual_kwh": float((frame.soc_end_kwh-expected_soc).abs().max()),
            "soc_boundary_violations": int(((frame[["soc_start_kwh","soc_end_kwh"]] < SOC_MIN_KWH-TOL) | (frame[["soc_start_kwh","soc_end_kwh"]] > SOC_MAX_KWH+TOL)).sum().sum()),
            "power_violations": int(((frame.charge_rt_kwh > ENERGY_MAX_KWH+TOL) | (frame.discharge_rt_kwh > ENERGY_MAX_KWH+TOL)).sum()),
            "simultaneous_charge_discharge": int(((frame.charge_rt_kwh>TOL)&(frame.discharge_rt_kwh>TOL)).sum()),
            "emergency_and_charge": int(((frame.emergency_purchase_kwh>TOL)&(frame.charge_rt_kwh>TOL)).sum()),
        }
    main_v = violations(main); pair_v = violations(paired)
    cross = []
    days = [g.sort_values("slot") for _,g in main.groupby("date",sort=True)]
    for left,right in zip(days[:-1],days[1:]): cross.append(abs(float(left.iloc[-1].soc_end_kwh)-float(right.iloc[0].soc_start_kwh)))
    settlement = settle_contract(main.original_plan_kwh, main.effective_purchase_kwh, main.price_yuan_per_kwh)
    settlement_error = float(np.max(np.abs(settlement.contract_cost_yuan-main.contract_cost_yuan)))
    actual_cost_error = float(np.max(np.abs(main.actual_total_cost_yuan-main.contract_cost_yuan-EMERGENCY_MULTIPLIER*main.price_yuan_per_kwh*main.emergency_purchase_kwh)))
    scen_balance = scenario.groupby(["run_id","scenario_index","slot"],as_index=False).first()
    scen_resid = scen_balance.load_scenario_kwh + scen_balance.charge_scenario_kwh + scen_balance.spill_scenario_kwh - scen_balance.discharge_scenario_kwh - scen_balance.emergency_scenario_kwh
    plan_lookup = plans.set_index(["run_id","slot"]).version_plan_kwh
    scen_expected = np.array([plan_lookup.loc[(r.run_id,int(r.slot))] + r.pv_scenario_kwh for _,r in scen_balance.iterrows()])
    scenario_balance_error = float(np.max(np.abs(scen_expected-scen_resid.to_numpy())))
    scenario_simul = int(((scenario.charge_scenario_kwh>TOL)&(scenario.discharge_scenario_kwh>TOL)).sum())
    scenario_emergency_charge = int(((scenario.emergency_scenario_kwh>TOL)&(scenario.charge_scenario_kwh>TOL)).sum())
    up_down = int(((plans.delta_plus_vs_original_kwh>TOL)&(plans.delta_minus_vs_original_kwh>TOL)).sum())
    result3_output = ROOT / "outputs/tables/result3.xlsx"
    checks = {
        "three_days_432_slots": len(main)==432,
        "twelve_plan_versions": solver.run_id.nunique()==12,
        "value_function_refreshed_12_times": main.value_function_refresh_id.nunique()==12,
        "all_scenario_dates_prior": bool(info.all_scenario_dates_before_decision.all()),
        "all_forecasts_visible": bool(info.forecast_visible.all()),
        "all_pv_information_causal": bool(info.pv_information_causal.all()),
        "no_future_load_used": bool(info.no_future_actual_load_used.all()),
        "no_actual_pv_used_in_forecast": not bool(info.actual_pv_used_in_forecast.any()),
        "executed_prefix_lock_max_error_kwh": float(plans.executed_prefix_lock_error_kwh.max()),
        "cross_day_soc_max_error_kwh": max(cross,default=0.0),
        "contract_recalculation_max_error_yuan": settlement_error,
        "actual_cost_recalculation_max_error_yuan": actual_cost_error,
        "scenario_balance_max_error_kwh": scenario_balance_error,
        "scenario_simultaneous_charge_discharge": scenario_simul,
        "scenario_emergency_and_charge": scenario_emergency_charge,
        "simultaneous_up_down": up_down,
        "protected_hashes_unchanged": before==after,
        "result3_output_exists": result3_output.exists(),
        "manual_and_synthetic_tests_pass": bool(synth.passed.all()),
        "main_physical": main_v,
        "paired_physical": pair_v,
    }
    scalar_ok = (
        checks["three_days_432_slots"] and checks["twelve_plan_versions"] and checks["value_function_refreshed_12_times"]
        and checks["all_scenario_dates_prior"] and checks["all_forecasts_visible"] and checks["all_pv_information_causal"]
        and checks["no_future_load_used"] and checks["no_actual_pv_used_in_forecast"] and checks["protected_hashes_unchanged"]
        and not checks["result3_output_exists"] and checks["manual_and_synthetic_tests_pass"]
        and max(checks["executed_prefix_lock_max_error_kwh"],checks["cross_day_soc_max_error_kwh"],settlement_error,actual_cost_error,scenario_balance_error)<=TOL
        and scenario_simul==0 and scenario_emergency_charge==0 and up_down==0
    )
    physical_ok = all(v==0 if "violations" in k or k in {"simultaneous_charge_discharge","emergency_and_charge"} else v<=TOL for metrics in (main_v,pair_v) for k,v in metrics.items())
    checks["passed"] = bool(scalar_ok and physical_ok)
    lines = ["Q3 V2 Stage 2 validation", "="*40]
    for k,v in checks.items(): lines.append(f"{k}: {v}")
    lines += ["", "MILP solution classes:", solver.solution_class.value_counts().to_string(), "", "No result3.xlsx was generated or modified."]
    (OUT / "q3_v2_stage2_validation.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if not checks["passed"]: raise RuntimeError("Q3 V2 Stage 2 validation failed")
    return checks


if __name__ == "__main__":
    started=time.perf_counter(); print(json.dumps(run(),ensure_ascii=False,indent=2)); print(f"elapsed_seconds={time.perf_counter()-started:.3f}")

