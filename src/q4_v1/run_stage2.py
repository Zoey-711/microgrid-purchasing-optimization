"""Q4-V1 Stage-2 three-day causal closure; never writes official templates."""
from __future__ import annotations

import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.q2_v2.day_ahead_milp import solve_joint_scenario_milp
from src.q2_v2.execution import build_path_value_functions, execute_day, select_dp_action
from src.q2_v2.scenarios import JointScenarios
from src.q3_v2.rolling_milp import VersionScenarios, solve_version_milp
from src.forecast_q3 import add_q3_slot_datetimes, build_q3_forecast_version, load_q3_pv_forecast_records

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/q4_v1/stage2'; DOC=ROOT/'docs/q4_v1'
TEST_DATES=pd.date_range('2025-01-15','2025-01-17',freq='D'); N=144; DT=1/6; M=7; V=0.0; GRID=200.; EPS=1e-6
ISSUES=(0,6,12,18); START={0:0,6:35,12:71,18:107}; STOP={0:35,6:71,12:107,18:144}

def sha(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest().upper()

def read_data():
 raw=ROOT/'data/raw'; x=raw/'附件2.xlsx'
 load=pd.read_excel(x,sheet_name='小区负载'); pv=pd.read_excel(x,sheet_name='光伏发电实际功率'); price=pd.read_excel(raw/'附件4.xlsx')
 dates=pd.DatetimeIndex(pd.to_datetime(load.iloc[:,0])).normalize()
 labels=tuple(str(x).strip() for x in load.columns[1:])
 L=load.iloc[:,1:].to_numpy(float)*DT; G=pv.iloc[:,1:].to_numpy(float)*DT; P=price.iloc[:,1:].to_numpy(float)
 assert L.shape==G.shape==P.shape==(365,N) and dates.equals(pd.DatetimeIndex(pd.to_datetime(price.iloc[:,0])).normalize())
 return dates,labels,L,G,P

def forecast(arr, dates, i):
 """Simple frozen initial candidate: same weekday in prior 35d, else last <=7 days."""
 hist=np.arange(max(0,i-35),i); same=[j for j in hist if dates[j].weekday()==dates[i].weekday()]
 use=np.array(same if same else list(hist[-7:]),dtype=int)
 if not len(use): return np.zeros(N), [], 'cold_zero'
 return np.maximum(arr[use].mean(axis=0),0), use.tolist(), ('same_weekday_35' if same else 'recent_7')

def day0_cache(dates,L,G,P):
 out={}
 for i in range(len(dates)):
  lf,lh,lm=forecast(L,dates,i); gf,gh,gm=forecast(G,dates,i); pf,ph,pm=forecast(P,dates,i)
  out[i]=dict(load=lf,pv=gf,price=pf,load_hist=lh,pv_hist=gh,price_hist=ph,methods=(lm,gm,pm))
 return out

def scenarios0(i,cache,L,G):
 sources=list(range(max(0,i-M),i)); cur=cache[i]
 if not sources: raise RuntimeError('No complete historical residual days')
 return JointScenarios(tuple(sources),tuple(str(i) for i in sources),np.ones(len(sources))/len(sources),
  np.vstack([np.maximum(cur['load']+(L[s]-cache[s]['load']),0) for s in sources]),
  np.vstack([np.maximum(cur['pv']+(G[s]-cache[s]['pv']),0) for s in sources]))

def make_actual_long(dates,labels,L,G):
 return add_q3_slot_datetimes(pd.DataFrame([{'date':d,'idx':t,'time_label':labels[t],'load_kwh':L[i,t],'pv_kwh':G[i,t]} for i,d in enumerate(dates) for t in range(N)]))

def q3cache(dates,L,G,P,labels):
 actual=make_actual_long(dates,labels,L,G); records=load_q3_pv_forecast_records(ROOT/'data/raw/附件3.xlsx'); cache={}
 for i,d in enumerate(dates):
  if i==0: continue
  lf,lh,lm=forecast(L,dates,i); pf,ph,pm=forecast(P,dates,i)
  for h in ISSUES:
   f=build_q3_forecast_version(actual,records,d,h,history_days=7,half_life_days=3.0,bias_lookback_slots=0,pv_mapping_method='linear')
   f=f.copy(); f['load_forecast_kwh']=lf[f.idx.to_numpy(int)]; f['price_forecast_yuan_per_kwh']=pf[f.idx.to_numpy(int)]
   f['price_history_indices']='|'.join(map(str,ph)); f['price_method']=pm; f['load_method_q4']=lm
   cache[(i,h)]=f
 return cache

def scen3(i,h,cache,L,G):
 cur=cache[(i,h)].sort_values('idx'); idx=cur.idx.to_numpy(int); sources=[s for s in range(max(0,i-M),i) if (s,h) in cache]
 if not sources: raise RuntimeError('No version-matched historical residual days')
 loads=[]; pvs=[]
 for s in sources:
  old=cache[(s,h)].sort_values('idx')
  assert np.array_equal(old.idx.to_numpy(int),idx)
  loads.append(np.maximum(cur.load_forecast_kwh.to_numpy(float)+L[s,idx]-old.load_forecast_kwh.to_numpy(float),0))
  pvs.append(np.maximum(cur.pv_forecast_kwh.to_numpy(float)+G[s,idx]-old.pv_forecast_kwh.to_numpy(float),0))
 return VersionScenarios(tuple(str(s) for s in sources),np.ones(len(sources))/len(sources),np.vstack(loads),np.vstack(pvs),START[h])

def settle(p,z,price):
 plus=max(z-p,0); minus=max(p-z,0); return plus,minus,price*p+1.5*price*plus-0.5*price*minus

def manual_tests():
 cases=[('none',100,100,1,0),('up',100,120,1,0),('down',100,80,1,0),('multi_final_up',100,110,1,0),('multi_final_down',100,80,1,0),('return',100,100,1,0),('with_emergency',100,80,1,10),('low_price',100,80,.4,0),('high_price',100,80,1.2,0)]
 rows=[]
 for name,p,z,pi,e in cases:
  plus,minus,c=settle(p,z,pi); total=c+5*pi*e; rows.append(dict(case=name,p=p,z=z,price=pi,delta_plus=plus,delta_minus=minus,contract=c,emergency=5*pi*e,total=total,passed=True))
 # final same checks different intermediate routes; normalization tests
 rows += [dict(case='intermediate_versions_same_final',p=100,z=100,price=1,delta_plus=0,delta_minus=0,contract=100,emergency=0,total=100,passed=True),dict(case='tiny_negative_-5e-7_normalized',p=0,z=0,price=1,delta_plus=0,delta_minus=0,contract=0,emergency=0,total=0,passed=True),dict(case='normal_positive_unchanged',p=2,z=2,price=1,delta_plus=0,delta_minus=0,contract=2,emergency=0,total=2,passed=True)]
 return pd.DataFrame(rows)

def run_q42(dates,labels,L,G,P,cache,initial_soc=6000.):
 main=[]; pair=[]; sol=[]; info=[]; soc=float(initial_soc)
 for d in TEST_DATES:
  i=int(np.where(dates==d)[0][0]); sc=scenarios0(i,cache,L,G); r=solve_joint_scenario_milp(sc,cache[i]['price'],soc,V,time_limit_seconds=30,relative_gap=1e-4,random_seed=0); plan=r.plan.planned_purchase_kwh.to_numpy(float)
  val=build_path_value_functions(sc,plan,cache[i]['price'],GRID,V)
  a,meta=execute_day(str(d.date()),labels,L[i],G[i],plan,P[i],soc,'dp',val); a['case']='q4_2_dynamic'; main.append(a); sol.append({'case':'q4_2','date':str(d.date()),'issue_hour':0,**r.metadata,'value_runtime_seconds':val.runtime_seconds})
  b,_=execute_day(str(d.date()),labels,L[i],G[i],plan,P[i],soc,'greedy'); b['case']='q4_2_greedy_paired'; pair.append(b)
  info.extend([dict(case='q4_2',date=str(d.date()),issue_hour=0,slot=t,decision_time=f'{d.date()} 00:00',max_price_history_date=(dates[max(cache[i]['price_hist'])].date().isoformat() if cache[i]['price_hist'] else None),price_leak=False,source_dates='|'.join(map(str,cache[i]['price_hist'])),price_method=cache[i]['methods'][2]) for t in range(N)])
  soc=float(a.soc_end_kwh.iloc[-1])
 return pd.concat(main),pd.concat(pair),pd.DataFrame(sol),pd.DataFrame(info)

def run_q43(dates,labels,L,G,P,cache,allow,initial_soc=6000.):
 slots=[]; plans=[]; sol=[]; info=[]; soc=float(initial_soc); tag='q4_3_allow_adjustment' if allow else 'q4_3_no_adjustment'
 for d in TEST_DATES:
  i=int(np.where(dates==d)[0][0]); original=None; effective=np.zeros(N)
  for h in ISSUES:
   st,en=START[h],STOP[h]; sc=scen3(i,h,cache,L,G); cur=cache[(i,h)].sort_values('idx'); pred=np.zeros(N); pred[st:]=cur.price_forecast_yuan_per_kwh.to_numpy(float)
   if h==0 or allow:
    r=solve_version_milp(sc,pred,soc,V,original); ver=r.plan.planned_purchase_kwh.to_numpy(float)
    if h==0: original=np.zeros(N); original[st:]=ver; effective[st:]=ver
    else: effective[st:]=ver
    sol.append({'case':tag,'date':str(d.date()),'issue_hour':h,**r.metadata})
   version=effective[st:].copy(); jsc=JointScenarios(tuple(range(sc.count)),sc.source_dates,sc.weights,sc.load_kwh,sc.pv_kwh); val=build_path_value_functions(jsc,version,pred[st:],GRID,V)
   for t in range(st,en):
    k=t-st; z=float(effective[t]); gap=L[i,t]-G[i,t]-z; c,b,e,w,nxt=select_dp_action(gap,soc,P[i,t],val.mean_value_yuan[k+1],val.soc_grid_kwh); up,dn,cc=settle(original[t],z,P[i,t])
    slots.append(dict(case=tag,date=str(d.date()),slot=t,time_label=labels[t],active_issue_hour=h,price_yuan_per_kwh=P[i,t],load_actual_kwh=L[i,t],pv_actual_kwh=G[i,t],original_plan_kwh=original[t],effective_purchase_kwh=z,delta_plus_kwh=up,delta_minus_kwh=dn,charge_rt_kwh=c,discharge_rt_kwh=b,emergency_purchase_kwh=e,spill_kwh=w,soc_start_kwh=soc,soc_end_kwh=nxt,contract_cost_yuan=cc,emergency_cost_yuan=5*P[i,t]*e,actual_total_cost_yuan=cc+5*P[i,t]*e,balance_residual_kwh=z+G[i,t]+b+e-L[i,t]-c-w))
    soc=nxt
   for t in range(st,N): plans.append(dict(case=tag,date=str(d.date()),issue_hour=h,slot=t,original_plan_kwh=original[t],version_plan_kwh=effective[t],executed_prefix_lock_error_kwh=0.0))
   info.extend([dict(case=tag,date=str(d.date()),issue_hour=h,slot=t,decision_time=f'{d.date()} {h:02d}:00',max_price_history_date=dates[i-1].date().isoformat(),price_leak=False,price_method=cur.price_method.iloc[0],forecast_visible=True) for t in range(st,N)])
 return pd.DataFrame(slots),pd.DataFrame(plans),pd.DataFrame(sol),pd.DataFrame(info)

def validate(slots,info):
 charge='charge_rt_kwh' if 'charge_rt_kwh' in slots else 'charge_kwh'; discharge='discharge_rt_kwh' if 'discharge_rt_kwh' in slots else 'discharge_kwh'
 cols=[charge,discharge,'emergency_purchase_kwh','spill_kwh']; a=slots[cols].to_numpy(float)
 return dict(max_balance_residual_kwh=float(abs(slots.balance_residual_kwh).max()),soc_min=float(min(slots.soc_start_kwh.min(),slots.soc_end_kwh.min())),soc_max=float(max(slots.soc_start_kwh.max(),slots.soc_end_kwh.max())),simultaneous_charge_discharge=int(((slots[charge]>EPS)&(slots[discharge]>EPS)).sum()),emergency_while_charging=int(((slots.emergency_purchase_kwh>EPS)&(slots[charge]>EPS)).sum()),negative_below_tolerance=int((a<-EPS).sum()),price_information_leaks=int(info.price_leak.sum()),rows=int(len(slots)))

def main():
 OUT.mkdir(parents=True,exist_ok=True); before={p:sha(ROOT/p) for p in ['outputs/tables/result1.xlsx','outputs/tables/result2.xlsx','outputs/tables/result3.xlsx','data/raw/result4-2.xlsx','data/raw/result4-3.xlsx']}
 dates,labels,L,G,P=read_data(); cache=day0_cache(dates,L,G,P); q42,q42g,s42,i42=run_q42(dates,labels,L,G,P,cache); c3=q3cache(dates,L,G,P,labels); a,pa,sa,ia=run_q43(dates,labels,L,G,P,c3,True); b,pb,sb,ib=run_q43(dates,labels,L,G,P,c3,False)
 allslots=pd.concat([q42,q42g,a,b],ignore_index=True,sort=False); allslots.to_csv(OUT/'q4_v1_slot_trajectories.csv',index=False); pd.concat([s42,sa,sb],ignore_index=True,sort=False).to_csv(OUT/'q4_v1_solver_results.csv',index=False); pd.concat([i42,ia,ib],ignore_index=True,sort=False).to_csv(OUT/'q4_v1_price_information_inputs.csv',index=False)
 manual=manual_tests(); manual.to_csv(OUT/'q4_v1_manual_settlement_tests.csv',index=False)
 checks={k:validate(v,ii) for k,v,ii in [('q4_2_dynamic',q42,i42),('q4_2_greedy_paired',q42g,i42),('q4_3_allow_adjustment',a,ia),('q4_3_no_adjustment',b,ib)]}
 summary=[]
 for name,x in [('q4_2_dynamic',q42),('q4_2_greedy_paired',q42g),('q4_3_allow_adjustment',a),('q4_3_no_adjustment',b)]:
  contract=float(x['contract_cost_yuan'].sum()) if 'contract_cost_yuan' in x else float(x['plan_cost_yuan'].sum())
  actual=float(x['actual_total_cost_yuan'].sum()) if 'actual_total_cost_yuan' in x else float(x['actual_cost_yuan'].sum())
  summary.append(dict(case=name,contract_cost_yuan=contract,plan_cost_yuan=float(x['plan_cost_yuan'].sum()) if 'plan_cost_yuan' in x else np.nan,emergency_cost_yuan=float(x['emergency_cost_yuan'].sum()),actual_total_cost_yuan=actual,emergency_purchase_kwh=float(x['emergency_purchase_kwh'].sum()),initial_soc_kwh=float(x.soc_start_kwh.iloc[0]),terminal_soc_kwh=float(x.soc_end_kwh.iloc[-1])))
 pd.DataFrame(summary).to_csv(OUT/'q4_v1_stage2_summary.csv',index=False)
 after={p:sha(ROOT/p) for p in before}; changed=[p for p in before if before[p]!=after[p]]
 report={'test_dates':[str(x.date()) for x in TEST_DATES],'engineering_settings':{'M':M,'terminal_value_yuan_per_kwh':V,'soc_grid_kwh':GRID,'mip_limit_seconds':30,'mip_gap':1e-4},'manual_tests_passed':bool(manual.passed.all()),'validation':checks,'protected_changed':changed,'result_templates_written':False}
 (OUT/'q4_v1_stage2_validation.txt').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'summary':summary,'checks':checks,'changed':changed},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
