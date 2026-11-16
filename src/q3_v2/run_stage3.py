"""Causal January development, staged screening, controls, and Q3-V2 freeze."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_q3 import load_q3_pv_forecast_records
from src.q2_v2.data import load_inputs
from src.q2_v2.execution import build_path_value_functions, greedy_action, select_dp_action

from .config import (
    EMERGENCY_MULTIPLIER, ENERGY_MAX_KWH, FIRST_MODIFIABLE_SLOT, ISSUE_HOURS,
    NEXT_BOUNDARY_SLOT, ROOT, SOC_INITIAL_TEST_KWH, SOC_MAX_KWH, SOC_MIN_KWH, TOL,
)
from .forecasting import build_month_cache
from .rolling_milp import VersionScenarios, solve_version_milp
from .run_stage2 import _actual_long, protected_manifest
from .settlement import settle_contract


STAGE3 = ROOT / "outputs/q3_v2/stage3"
EVAL_START = "2025-01-22"
EVAL_END = "2025-01-31"


@dataclass(frozen=True)
class Candidate:
    load_method: str
    scenario_cap: int
    terminal_value: float
    soc_grid_kwh: float
    adjustment_policy: str = "all_updates"

    @property
    def id(self) -> str:
        v = str(self.terminal_value).replace(".", "p")
        h = str(int(self.soc_grid_kwh))
        return f"{self.load_method}_M{self.scenario_cap}_v{v}_h{h}_{self.adjustment_policy}"


def _scenarios(data, cache, date: str, issue: int, cap: int) -> VersionScenarios:
    current = cache[(date, issue)].sort_values("idx")
    idx = current.idx.to_numpy(int)
    sources = sorted({d for d,k in cache if k == issue and d < date})[-cap:]
    if not sources:
        raise RuntimeError("No causal residual scenarios")
    load, pv = [], []
    for source in sources:
        hist = cache[(source, issue)].sort_values("idx")
        if not np.array_equal(hist.idx.to_numpy(int), idx):
            raise AssertionError("Forecast suffix mismatch")
        di = data.day_index(source)
        load_resid = data.load_kwh[di,idx] - hist.load_forecast_kwh.to_numpy(float)
        pv_resid = data.pv_kwh[di,idx] - hist.pv_forecast_kwh.to_numpy(float)
        load.append(np.maximum(current.load_forecast_kwh.to_numpy(float)+load_resid,0.0))
        pv.append(np.maximum(current.pv_forecast_kwh.to_numpy(float)+pv_resid,0.0))
    n=len(sources)
    return VersionScenarios(tuple(sources),np.full(n,1.0/n),np.vstack(load),np.vstack(pv),FIRST_MODIFIABLE_SLOT[issue])


def _validate_scenario(solve) -> dict:
    d=solve.scenario_dispatch
    p=solve.plan.set_index("slot").planned_purchase_kwh
    lhs=np.array([p.loc[int(r.slot)]+r.pv_scenario_kwh+r.discharge_scenario_kwh+r.emergency_scenario_kwh for _,r in d.iterrows()])
    rhs=d.load_scenario_kwh+d.charge_scenario_kwh+d.spill_scenario_kwh
    trans=d.soc_start_scenario_kwh+0.9*d.charge_scenario_kwh-d.discharge_scenario_kwh/0.9
    return {
        "scenario_balance_max_error_kwh":float(np.max(np.abs(lhs-rhs))),
        "scenario_soc_max_error_kwh":float(np.max(np.abs(d.soc_end_scenario_kwh-trans))),
        "scenario_simultaneous_charge_discharge":int(((d.charge_scenario_kwh>TOL)&(d.discharge_scenario_kwh>TOL)).sum()),
        "scenario_emergency_and_charge":int(((d.emergency_scenario_kwh>TOL)&(d.charge_scenario_kwh>TOL)).sum()),
    }


def _execute_action(executor, gap, soc, price, value, local):
    if executor == "dp":
        return select_dp_action(gap,soc,price,value.mean_value_yuan[local+1],value.soc_grid_kwh)
    return greedy_action(gap,soc)


def run_candidate(candidate: Candidate, data, cache, save=True) -> dict:
    cdir=STAGE3/"candidates"/candidate.id
    summary_file=cdir/"summary.json"
    if summary_file.exists():
        summary=json.loads(summary_file.read_text(encoding="utf-8"))
        return {"summary":summary,"slots":pd.read_csv(cdir/"slots.csv"),"plans":pd.read_csv(cdir/"plan_versions.csv"),"solver":pd.read_csv(cdir/"solver_log.csv"),"info":pd.read_csv(cdir/"information_audit.csv")}
    cdir.mkdir(parents=True,exist_ok=True)
    slots=[]; plans=[]; solvers=[]; infos=[]
    soc=SOC_INITIAL_TEST_KWH
    run_started=time.perf_counter()
    for date in pd.date_range("2025-01-01","2025-01-31",freq="D"):
        ds=date.date().isoformat(); di=data.day_index(ds)
        if ds == "2025-01-01":
            # Explicit common cold start: no historical scenario and zero plan.
            for t in range(144):
                gap=float(data.load_kwh[di,t]-data.pv_kwh[di,t])
                charge,discharge,emergency,spill,next_soc=greedy_action(gap,soc)
                emergency_cost=EMERGENCY_MULTIPLIER*data.price_yuan_per_kwh[t]*emergency
                slots.append({"candidate_id":candidate.id,"date":ds,"slot":t,"time_label":data.time_labels[t],"active_issue_hour":0,
                    "price_yuan_per_kwh":data.price_yuan_per_kwh[t],"load_actual_kwh":data.load_kwh[di,t],"pv_actual_kwh":data.pv_kwh[di,t],
                    "load_forecast_kwh":np.nan,"pv_forecast_kwh":np.nan,"original_plan_kwh":0.0,"effective_purchase_kwh":0.0,
                    "delta_plus_kwh":0.0,"delta_minus_kwh":0.0,"charge_rt_kwh":charge,"discharge_rt_kwh":discharge,
                    "emergency_purchase_kwh":emergency,"spill_kwh":spill,"soc_start_kwh":soc,"soc_end_kwh":next_soc,
                    "contract_cost_yuan":0.0,"emergency_cost_yuan":emergency_cost,"actual_total_cost_yuan":emergency_cost,
                    "balance_residual_kwh":-gap+discharge+emergency-charge-spill,"cold_start":True})
                soc=float(next_soc)
            continue
        original=None; effective=np.full(144,np.nan)
        for issue in ISSUE_HOURS:
            start=FIRST_MODIFIABLE_SLOT[issue]; stop=NEXT_BOUNDARY_SLOT[issue]
            fc=cache[(ds,issue)].sort_values("idx")
            scenarios=_scenarios(data,cache,ds,issue,candidate.scenario_cap)
            solved=None
            if issue==0 or candidate.adjustment_policy=="all_updates":
                solved=solve_version_milp(scenarios,data.price_yuan_per_kwh,soc,candidate.terminal_value,original)
                version=solved.plan.sort_values("slot").planned_purchase_kwh.to_numpy(float)
                check=_validate_scenario(solved)
                run_id=f"{candidate.id}_{ds}_{issue:02d}00"
                solvers.append({"candidate_id":candidate.id,"run_id":run_id,"date":ds,"issue_hour":issue,"start_slot":start,**solved.metadata,**check})
            else:
                version=original[start:].copy(); run_id=f"{candidate.id}_{ds}_{issue:02d}00_value_only"
            if issue==0:
                original=np.full(144,np.nan); original[:]=version
            prefix=effective[:start].copy(); effective[start:]=version
            prefix_error=float(np.nanmax(np.abs(prefix-effective[:start]))) if start else 0.0
            if start and (np.isnan(prefix).any() or prefix_error>TOL): raise AssertionError("Executed plan prefix changed")
            value=build_path_value_functions(scenarios,version,data.price_yuan_per_kwh[start:],candidate.soc_grid_kwh,candidate.terminal_value)
            fcm=fc.set_index("idx")
            for t in range(start,144):
                plans.append({"candidate_id":candidate.id,"run_id":run_id,"date":ds,"issue_hour":issue,"slot":t,
                    "original_plan_kwh":original[t],"version_plan_kwh":version[t-start],
                    "delta_plus_vs_original_kwh":max(version[t-start]-original[t],0.0),"delta_minus_vs_original_kwh":max(original[t]-version[t-start],0.0),
                    "executed_prefix_lock_error_kwh":prefix_error,"milp_solved":solved is not None})
            infos.append({"candidate_id":candidate.id,"run_id":run_id,"date":ds,"issue_hour":issue,"first_modifiable_slot":start,
                "scenario_count_requested":candidate.scenario_cap,"scenario_count_effective":scenarios.count,
                "scenario_source_dates":"|".join(scenarios.source_dates),"all_sources_prior":all(s<ds for s in scenarios.source_dates),
                "forecast_visible":bool(fc.forecast_visible_at_decision.all()),"pv_information_causal":bool(fc.pv_information_causal.all()),
                "no_future_load_used":bool(fc.no_future_actual_load_used.all()),"actual_pv_used_in_forecast":bool(fc.pv_actual_observation_used.any()),
                "value_plan_hash":hashlib.sha256(version.tobytes()).hexdigest().upper(),"value_runtime_seconds":value.runtime_seconds})
            for t in range(start,stop):
                local=t-start; z=float(effective[t]); gap=float(data.load_kwh[di,t]-data.pv_kwh[di,t]-z)
                charge,discharge,emergency,spill,next_soc=_execute_action("dp",gap,soc,float(data.price_yuan_per_kwh[t]),value,local)
                settled=settle_contract([original[t]],[z],[data.price_yuan_per_kwh[t]]).iloc[0]
                emergency_cost=EMERGENCY_MULTIPLIER*data.price_yuan_per_kwh[t]*emergency
                balance=z+data.pv_kwh[di,t]+discharge+emergency-data.load_kwh[di,t]-charge-spill
                slots.append({"candidate_id":candidate.id,"date":ds,"slot":t,"time_label":data.time_labels[t],"active_issue_hour":issue,
                    "price_yuan_per_kwh":data.price_yuan_per_kwh[t],"load_actual_kwh":data.load_kwh[di,t],"pv_actual_kwh":data.pv_kwh[di,t],
                    "load_forecast_kwh":fcm.loc[t].load_forecast_kwh,"pv_forecast_kwh":fcm.loc[t].pv_forecast_kwh,
                    "original_plan_kwh":original[t],"effective_purchase_kwh":z,"delta_plus_kwh":settled.delta_plus_kwh,"delta_minus_kwh":settled.delta_minus_kwh,
                    "charge_rt_kwh":charge,"discharge_rt_kwh":discharge,"emergency_purchase_kwh":emergency,"spill_kwh":spill,
                    "soc_start_kwh":soc,"soc_end_kwh":next_soc,"contract_cost_yuan":settled.contract_cost_yuan,
                    "emergency_cost_yuan":emergency_cost,"actual_total_cost_yuan":settled.contract_cost_yuan+emergency_cost,
                    "balance_residual_kwh":balance,"cold_start":False})
                soc=float(next_soc)
    slot=pd.DataFrame(slots); plan=pd.DataFrame(plans); solver=pd.DataFrame(solvers); info=pd.DataFrame(infos)
    evaluation=slot[(slot.date>=EVAL_START)&(slot.date<=EVAL_END)].copy()
    daily=evaluation.groupby("date",as_index=False).agg(soc_start_kwh=("soc_start_kwh","first"),soc_end_kwh=("soc_end_kwh","last"),
        contract_cost_yuan=("contract_cost_yuan","sum"),emergency_cost_yuan=("emergency_cost_yuan","sum"),actual_total_cost_yuan=("actual_total_cost_yuan","sum"),
        emergency_purchase_kwh=("emergency_purchase_kwh","sum"),charge_kwh=("charge_rt_kwh","sum"),discharge_kwh=("discharge_rt_kwh","sum"),spill_kwh=("spill_kwh","sum"))
    price=evaluation.price_yuan_per_kwh
    emerg=evaluation.emergency_purchase_kwh
    expected_soc=evaluation.soc_start_kwh+0.9*evaluation.charge_rt_kwh-evaluation.discharge_rt_kwh/0.9
    max_balance=float(evaluation.balance_residual_kwh.abs().max()); max_soc=float((evaluation.soc_end_kwh-expected_soc).abs().max())
    eval_solver=solver[(solver.date>=EVAL_START)&(solver.date<=EVAL_END)]
    summary={**asdict(candidate),"candidate_id":candidate.id,"evaluation_start":EVAL_START,"evaluation_end":EVAL_END,"evaluation_days":10,
        "contract_cost_yuan":float(evaluation.contract_cost_yuan.sum()),"emergency_cost_yuan":float(evaluation.emergency_cost_yuan.sum()),
        "actual_total_cost_yuan":float(evaluation.actual_total_cost_yuan.sum()),"emergency_purchase_kwh":float(emerg.sum()),
        "emergency_low_price_kwh":float(emerg[price<0.6].sum()),"emergency_mid_price_kwh":float(emerg[(price>=0.6)&(price<1.0)].sum()),
        "emergency_high_price_kwh":float(emerg[price>=1.0].sum()),"evaluation_initial_soc_kwh":float(evaluation.iloc[0].soc_start_kwh),
        "evaluation_terminal_soc_kwh":float(evaluation.iloc[-1].soc_end_kwh),"mean_daily_end_soc_kwh":float(daily.soc_end_kwh.mean()),
        "minimum_actual_soc_kwh":float(evaluation[["soc_start_kwh","soc_end_kwh"]].min().min()),"maximum_actual_soc_kwh":float(evaluation[["soc_start_kwh","soc_end_kwh"]].max().max()),
        "charge_kwh":float(evaluation.charge_rt_kwh.sum()),"discharge_kwh":float(evaluation.discharge_rt_kwh.sum()),
        "storage_throughput_kwh":float(evaluation.charge_rt_kwh.sum()+evaluation.discharge_rt_kwh.sum()),"spill_kwh":float(evaluation.spill_kwh.sum()),
        "upward_adjustment_kwh":float(evaluation.delta_plus_kwh.sum()),"downward_adjustment_kwh":float(evaluation.delta_minus_kwh.sum()),
        "load_mae_kwh_per_slot":float(np.mean(np.abs(evaluation.load_actual_kwh-evaluation.load_forecast_kwh))),
        "pv_mae_kwh_per_slot":float(np.mean(np.abs(evaluation.pv_actual_kwh-evaluation.pv_forecast_kwh))),
        "net_load_mae_kwh_per_slot":float(np.mean(np.abs((evaluation.load_actual_kwh-evaluation.pv_actual_kwh)-(evaluation.load_forecast_kwh-evaluation.pv_forecast_kwh)))),
        "milp_calls_evaluation":int(len(eval_solver)),"mean_mip_gap":float(eval_solver.relative_gap.mean()),"max_mip_gap":float(eval_solver.relative_gap.max()),
        "mean_solve_seconds":float(eval_solver.solve_time_seconds.mean()),"total_solve_seconds_all_january":float(solver.solve_time_seconds.sum()),
        "total_value_seconds_all_january":float(info.value_runtime_seconds.sum()),"time_limit_feasible_count":int((solver.solution_class!="certified_to_requested_gap").sum()),
        "max_balance_residual_kwh":max_balance,"max_soc_transition_residual_kwh":max_soc,
        "soc_violation_count":int(((evaluation.soc_end_kwh<SOC_MIN_KWH-TOL)|(evaluation.soc_end_kwh>SOC_MAX_KWH+TOL)).sum()),
        "power_violation_count":int(((evaluation.charge_rt_kwh>ENERGY_MAX_KWH+TOL)|(evaluation.discharge_rt_kwh>ENERGY_MAX_KWH+TOL)).sum()),
        "simultaneous_charge_discharge_count":int(((evaluation.charge_rt_kwh>TOL)&(evaluation.discharge_rt_kwh>TOL)).sum()),
        "emergency_and_charge_count":int(((evaluation.emergency_purchase_kwh>TOL)&(evaluation.charge_rt_kwh>TOL)).sum()),
        "information_violation_count":int((~info.all_sources_prior).sum()+(~info.forecast_visible).sum()+(~info.pv_information_causal).sum()+(~info.no_future_load_used).sum()+info.actual_pv_used_in_forecast.sum()),
        "cross_day_soc_max_error_kwh":float(max(abs(g.sort_values("slot").iloc[-1].soc_end_kwh-nextg.sort_values("slot").iloc[0].soc_start_kwh) for (_,g),(_,nextg) in zip(list(slot.groupby("date",sort=True))[:-1],list(slot.groupby("date",sort=True))[1:]))),
        "runtime_seconds":time.perf_counter()-run_started}
    summary["validation_passed"]=bool(max_balance<=TOL and max_soc<=TOL and summary["soc_violation_count"]==0 and summary["power_violation_count"]==0 and summary["simultaneous_charge_discharge_count"]==0 and summary["emergency_and_charge_count"]==0 and summary["information_violation_count"]==0 and summary["cross_day_soc_max_error_kwh"]<=TOL)
    if save:
        slot.to_csv(cdir/"slots.csv",index=False,encoding="utf-8-sig",float_format="%.10f")
        plan.to_csv(cdir/"plan_versions.csv",index=False,encoding="utf-8-sig",float_format="%.10f")
        solver.to_csv(cdir/"solver_log.csv",index=False,encoding="utf-8-sig",float_format="%.10f")
        info.to_csv(cdir/"information_audit.csv",index=False,encoding="utf-8-sig")
        daily.to_csv(cdir/"evaluation_daily.csv",index=False,encoding="utf-8-sig",float_format="%.10f")
        summary_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"COMPLETE {candidate.id} cost={summary['actual_total_cost_yuan']:.3f} terminal={summary['evaluation_terminal_soc_kwh']:.3f} runtime={summary['runtime_seconds']:.1f}s",flush=True)
    return {"summary":summary,"slots":slot,"plans":plan,"solver":solver,"info":info}


def _paired_greedy(selected: dict) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    main=selected["slots"]
    rows=[]
    for date,day in main[main.date>=EVAL_START].groupby("date",sort=True):
        day=day.sort_values("slot"); soc=float(day.iloc[0].soc_start_kwh)
        for _,r in day.iterrows():
            gap=float(r.load_actual_kwh-r.pv_actual_kwh-r.effective_purchase_kwh)
            c,b,e,w,n=greedy_action(gap,soc); ec=EMERGENCY_MULTIPLIER*r.price_yuan_per_kwh*e
            rows.append({"date":date,"slot":int(r.slot),"soc_start_kwh":soc,"soc_end_kwh":n,"charge_rt_kwh":c,"discharge_rt_kwh":b,
                "emergency_purchase_kwh":e,"spill_kwh":w,"contract_cost_yuan":r.contract_cost_yuan,"emergency_cost_yuan":ec,
                "actual_total_cost_yuan":r.contract_cost_yuan+ec,"balance_residual_kwh":r.effective_purchase_kwh+r.pv_actual_kwh+b+e-r.load_actual_kwh-c-w})
            soc=float(n)
    paired=pd.DataFrame(rows)
    main_eval=main[main.date>=EVAL_START].sort_values(["date","slot"]); paired=paired.sort_values(["date","slot"])
    summary={"comparison":"fixed_selected_plan_versions_same_main_daily_initial_soc","dynamic_cost_yuan":float(main_eval.actual_total_cost_yuan.sum()),
        "greedy_cost_yuan":float(paired.actual_total_cost_yuan.sum()),"dynamic_minus_greedy_cost_yuan":float(main_eval.actual_total_cost_yuan.sum()-paired.actual_total_cost_yuan.sum()),
        "dynamic_emergency_kwh":float(main_eval.emergency_purchase_kwh.sum()),"greedy_emergency_kwh":float(paired.emergency_purchase_kwh.sum()),
        "dynamic_minus_greedy_emergency_kwh":float(main_eval.emergency_purchase_kwh.sum()-paired.emergency_purchase_kwh.sum()),
        "action_l1_difference_kwh":float((abs(main_eval.charge_rt_kwh.to_numpy()-paired.charge_rt_kwh.to_numpy())+abs(main_eval.discharge_rt_kwh.to_numpy()-paired.discharge_rt_kwh.to_numpy())).sum()),
        "different_action_slots":int(((abs(main_eval.charge_rt_kwh.to_numpy()-paired.charge_rt_kwh.to_numpy())+abs(main_eval.discharge_rt_kwh.to_numpy()-paired.discharge_rt_kwh.to_numpy()))>TOL).sum())}
    daily=paired.groupby("date",as_index=False).agg(soc_start_kwh=("soc_start_kwh","first"),soc_end_kwh=("soc_end_kwh","last"),actual_total_cost_yuan=("actual_total_cost_yuan","sum"),emergency_cost_yuan=("emergency_cost_yuan","sum"),emergency_purchase_kwh=("emergency_purchase_kwh","sum"))
    return paired,daily,summary


def main():
    STAGE3.mkdir(parents=True,exist_ok=True)
    before=protected_manifest(); (STAGE3/"protected_hashes_before.json").write_text(json.dumps(before,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    data=load_inputs(); actual=_actual_long(data); pv=load_q3_pv_forecast_records()
    caches={m:build_month_cache(actual,pv,m) for m in ("ewma14_bias12","weekday35_bias12")}
    runs={}; phase=[]
    def evaluate(c,stage):
        if c.id not in runs: runs[c.id]=run_candidate(c,data,caches[c.load_method])
        phase.append({"phase":stage,"candidate_id":c.id})
        return runs[c.id]
    # 1. Load predictors.
    load_candidates=[Candidate(m,14,0.4,100.0) for m in ("ewma14_bias12","weekday35_bias12")]
    for c in load_candidates:evaluate(c,"1_load_predictor")
    best_load=min(load_candidates,key=lambda c:runs[c.id]["summary"]["actual_total_cost_yuan"]).load_method
    # 2. Scenario caps.
    m_candidates=[Candidate(best_load,m,0.4,100.0) for m in (7,14,21)]
    for c in m_candidates:evaluate(c,"2_scenario_cap")
    ranked_m=sorted(m_candidates,key=lambda c:runs[c.id]["summary"]["actual_total_cost_yuan"]); best_m=ranked_m[0].scenario_cap
    # 3. Terminal values.
    v_candidates=[Candidate(best_load,best_m,v,100.0) for v in (0.0,0.4,0.8)]
    for c in v_candidates:evaluate(c,"3_terminal_value")
    ranked_v=sorted(v_candidates,key=lambda c:runs[c.id]["summary"]["actual_total_cost_yuan"]); best_v=ranked_v[0].terminal_value
    # 4. Cross checks fixed before seeing their outcomes.
    alternate_load="weekday35_bias12" if best_load=="ewma14_bias12" else "ewma14_bias12"
    cross=[Candidate(alternate_load,best_m,best_v,100.0),Candidate(best_load,ranked_m[1].scenario_cap,ranked_v[1].terminal_value,100.0)]
    for c in cross:evaluate(c,"4_cross_check")
    # Pick best validated h100 configuration tested so far, then h50 check; include required M14/v0.4/h50 reference.
    valid100=[Candidate(**{k:r["summary"][k] for k in ("load_method","scenario_cap","terminal_value","soc_grid_kwh","adjustment_policy")}) for r in runs.values() if r["summary"]["validation_passed"] and r["summary"]["soc_grid_kwh"]==100.0]
    best100=min(valid100,key=lambda c:runs[c.id]["summary"]["actual_total_cost_yuan"])
    grid_candidates=[Candidate(best100.load_method,best100.scenario_cap,best100.terminal_value,50.0),Candidate("ewma14_bias12",14,0.4,50.0)]
    for c in grid_candidates:evaluate(c,"5_grid_and_reference")
    # Final selection from the best100/h50 pair, not from unrelated screening configurations.
    final_pair=[best100,grid_candidates[0]]
    selected_c=min(final_pair,key=lambda c:runs[c.id]["summary"]["actual_total_cost_yuan"])
    selected=runs[selected_c.id]
    control_c=Candidate(selected_c.load_method,selected_c.scenario_cap,selected_c.terminal_value,selected_c.soc_grid_kwh,"no_adjustment")
    control=evaluate(control_c,"6_independent_no_adjustment_control")
    paired,paired_daily,paired_summary=_paired_greedy(selected)
    paired.to_csv(STAGE3/"paired_greedy_slots.csv",index=False,encoding="utf-8-sig",float_format="%.10f")
    paired_daily.to_csv(STAGE3/"paired_greedy_daily.csv",index=False,encoding="utf-8-sig",float_format="%.10f")
    (STAGE3/"paired_ablation_summary.json").write_text(json.dumps(paired_summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    comparison=pd.DataFrame([r["summary"] for r in runs.values()]).sort_values("actual_total_cost_yuan")
    phase_df=pd.DataFrame(phase).drop_duplicates()
    comparison=comparison.merge(phase_df.groupby("candidate_id").phase.agg(lambda x:"|".join(x)).reset_index(),on="candidate_id",how="left")
    comparison.to_csv(STAGE3/"q3_v2_january_candidate_comparison.csv",index=False,encoding="utf-8-sig",float_format="%.10f")
    phase_df.to_csv(STAGE3/"q3_v2_selection_path.csv",index=False,encoding="utf-8-sig")
    freeze={"selected_candidate_id":selected_c.id,**asdict(selected_c),"development_period":[EVAL_START,EVAL_END],
        "cold_start":"2025-01-01 00:00 SOC=6000 kWh; zero plan; greedy physical execution",
        "warmup":"2025-01-02 through 2025-01-21; candidate-specific closed-loop SOC",
        "selection_primary_metric":"actual contract settlement plus emergency purchase cost",
        "parameters_selected_using_Feb_Dec":False,"result3_generated":False}
    (STAGE3/"q3_v2_frozen_config.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    after=protected_manifest(); (STAGE3/"protected_hashes_after.json").write_text(json.dumps(after,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if before!=after:raise RuntimeError("Protected files changed during Stage 3")
    print("SELECTED",json.dumps(freeze,ensure_ascii=False),flush=True)
    print("CONTROL",json.dumps(control["summary"],ensure_ascii=False),flush=True)
    print("PAIRED",json.dumps(paired_summary,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
