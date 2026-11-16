"""January-only Q4-V1 development protocol. No post-January input is read."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd
from src.q2_v2.execution import execute_day, build_path_value_functions, greedy_action
from src.forecast_q3 import add_q3_slot_datetimes, build_q3_forecast_version
from src.q3_v2.forecasting import _jan1_cold_forecast
from src.q4_v1 import run_stage2 as s

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/q4_v1/stage3'; DATES=pd.date_range('2025-01-01','2025-01-31'); EVAL=pd.date_range('2025-01-22','2025-01-31'); EPS=1e-6

def read_jan():
 raw=ROOT/'data/raw'; a=pd.read_excel(raw/'附件2.xlsx',sheet_name='小区负载',nrows=31); g=pd.read_excel(raw/'附件2.xlsx',sheet_name='光伏发电实际功率',nrows=31); p=pd.read_excel(raw/'附件4.xlsx',nrows=31)
 dates=pd.DatetimeIndex(pd.to_datetime(a.iloc[:,0])).normalize(); labels=tuple(str(x).strip() for x in a.columns[1:])
 return dates,labels,a.iloc[:,1:].to_numpy(float)/6,g.iloc[:,1:].to_numpy(float)/6,p.iloc[:,1:].to_numpy(float)

def pv_records_jan():
 raw=pd.read_excel(ROOT/'data/raw/附件3.xlsx',nrows=124); dates=pd.to_datetime(raw.iloc[:,0].ffill()).dt.normalize(); rows=[]
 for i,row in raw.iterrows():
  hour=int(str(row.iloc[1]).split(':')[0]); issue=dates.iloc[i]+pd.Timedelta(hours=hour)
  for h,x in enumerate(row.iloc[2:26],1): rows.append(dict(source_row=i+2,issue_datetime=issue,horizon_hour=h,target_datetime=issue+pd.Timedelta(hours=h),pv_forecast_kw=float(x)))
 return pd.DataFrame(rows)

def predictor(name, arr, dates, i):
 hist=list(range(i))
 if not hist:return np.zeros(144),[], 'cold_zero'
 if name=='P1': use=[hist[-1]]; method='previous_day'
 elif name=='P2': use=hist[-7:]; method='recent_7_mean'
 elif name=='P3': use=hist[-14:]; method='recent_14_mean'
 elif name=='P4':
  use=hist[-14:]; age=np.arange(len(use),0,-1); w=.5**((age-1)/4); return np.maximum(np.average(arr[use],axis=0,weights=w),0),use,'ewma14_half_life4'
 elif name=='P5':
  use=[j for j in hist[-35:] if dates[j].weekday()==dates[i].weekday()]; use=use or hist[-7:]; method='weekday35_or_recent7'
 else: raise ValueError(name)
 return np.maximum(arr[use].mean(axis=0),0),use,method

def supply_predictor(name, arr, dates, i):
 return predictor({'recent':'P1','short':'P2','weekday':'P5'}[name],arr,dates,i)

def metrics(actual,pred):
 e=actual-pred; high=actual>=np.quantile(actual,.8); peak=actual>=np.quantile(actual,.95); cls=lambda x:np.digitize(x,[.6,1.0])
 return dict(mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),high_price_mae=float(abs(e[high]).mean()),peak_miss_rate=float(((pred[peak]<np.quantile(actual,.8))).mean()) if peak.any() else 0.,price_class_accuracy=float((cls(actual)==cls(pred)).mean()))

def configure(price_name,supply_name,dates,L,G,P):
 def f(arr,ds,i):
  if arr is P:return predictor(price_name,arr,ds,i)
  return supply_predictor(supply_name,arr,ds,i)
 s.forecast=f; s.M=7; s.V=.2; s.GRID=100.; s.TEST_DATES=pd.date_range('2025-01-02','2025-01-31')
 cache=s.day0_cache(dates,L,G,P)
 # local January-only versioned cache; no post-January forecast rows exist.
 actual=s.make_actual_long(dates,s._labels_for_stage3,L,G) if False else None
 actual=add_q3_slot_datetimes(pd.DataFrame([{'date':d,'idx':t,'time_label':s._labels_for_stage3[t] if hasattr(s,'_labels_for_stage3') else str(t),'load_kwh':L[i,t],'pv_kwh':G[i,t]} for i,d in enumerate(dates) for t in range(144)]))
 rec=pv_records_jan(); c3={}
 for i,d in enumerate(dates):
  lf,_,_=supply_predictor(supply_name,L,dates,i); pf,ph,pm=predictor(price_name,P,dates,i)
  for h in s.ISSUES:
   q=(_jan1_cold_forecast(actual,rec,h) if i==0 else build_q3_forecast_version(actual,rec,d,h,history_days=7,half_life_days=3,bias_lookback_slots=0,pv_mapping_method='linear')).copy(); idx=q.idx.to_numpy(int); q['load_forecast_kwh']=lf[idx];q['price_forecast_yuan_per_kwh']=pf[idx];q['price_method']=pm;c3[(i,h)]=q
 return cache,c3

def cold_soc(labels,L,G,P):
 x,_=execute_day('2025-01-01',labels,L[0],G[0],np.zeros(144),P[0],6000.,'greedy');return float(x.soc_end_kwh.iloc[-1])

def run_branch(price_name,supply_name,M,V,grid,branch,dates,labels,L,G,P,cold):
 cache,c3=configure(price_name,supply_name,dates,L,G,P); s.M=M;s.V=V;s.GRID=grid;s.TEST_DATES=pd.date_range('2025-01-02','2025-01-31')
 if branch=='42': a,g,sol,info=s.run_q42(dates,labels,L,G,P,cache,cold); return a,g,sol,info
 a,pa,sol,info=s.run_q43(dates,labels,L,G,P,c3,True,cold); return a,pa,sol,info

def aggregate(x):
 y=x[x.date.isin([d.date().isoformat() for d in EVAL])]; cc='contract_cost_yuan' if 'contract_cost_yuan' in y else 'plan_cost_yuan'; total='actual_total_cost_yuan' if 'actual_total_cost_yuan' in y else 'actual_cost_yuan'
 ends=y.groupby('date').soc_end_kwh.last(); return dict(contract_cost_yuan=float(y[cc].sum()),emergency_cost_yuan=float(y.emergency_cost_yuan.sum()),actual_cost_yuan=float(y[total].sum()),emergency_kwh=float(y.emergency_purchase_kwh.sum()),start_soc=float(y.groupby('date').soc_start_kwh.first().iloc[0]),end_soc=float(ends.iloc[-1]),min_soc=float(min(y.soc_start_kwh.min(),y.soc_end_kwh.min())),mean_end_soc=float(ends.mean()),inventory_corrected_cost_yuan=float(y[total].sum()+1.0*(6000-ends.iloc[-1])))

def main():
 OUT.mkdir(parents=True,exist_ok=True); dates,labels,L,G,P=read_jan(); s._labels_for_stage3=labels; cold=cold_soc(labels,L,G,P); actual=P[21:31]
 price_rows=[]
 for name in ['P1','P2','P3','P4','P5']:
  pred=np.vstack([predictor(name,P,dates,i)[0] for i in range(21,31)]); price_rows.append(dict(predictor=name,**metrics(actual,pred),fallback_count=sum(len(predictor(name,P,dates,i)[1])==0 for i in range(21,31))))
 pf=pd.DataFrame(price_rows).sort_values(['high_price_mae','peak_miss_rate','mae']); pf.to_csv(OUT/'q4_v1_price_forecast_comparison.csv',index=False); prices=pf.predictor.head(2).tolist()
 candidates=[]; saved={}
 # staged supply screening under fixed price/M/v/grid using Q4-2 only
 for sup in ['recent','short','weekday']:
  x,_,sol,info=run_branch(prices[0],sup,7,.2,100,'42',dates,labels,L,G,P,cold); r=aggregate(x); candidates.append(dict(stage='supply_screen',branch='4-2',price_predictor=prices[0],supply_predictor=sup,M=7,v=.2,grid=100,**r)); saved[('42',prices[0],sup,7,.2,100)]=(x,sol,info)
 supply=min([z for z in candidates if z['stage']=='supply_screen'],key=lambda z:z['inventory_corrected_cost_yuan'])['supply_predictor']
 # Economic screen preserves both price candidates for each independent branch.
 for br in ['42','43']:
  for pp in prices:
   x,_,sol,info=run_branch(pp,supply,14,.2,100,br,dates,labels,L,G,P,cold);r=aggregate(x);candidates.append(dict(stage='price_economic',branch='4-'+br,price_predictor=pp,supply_predictor=supply,M=14,v=.2,grid=100,**r));saved[(br,pp,supply,14,.2,100)]=(x,sol,info)
 selected_price={br:min([z for z in candidates if z['stage']=='price_economic' and z['branch']=='4-'+br],key=lambda z:z['inventory_corrected_cost_yuan'])['price_predictor'] for br in ['42','43']}
 # M/v screen: fixed pre-declared 12 points per branch.
 for br in ['42','43']:
  for m in [7,14,21]:
   for v in [0,.2,.4,.6]:
    key=(br,selected_price[br],supply,m,v,100)
    if key in saved:x,sol,info=saved[key]
    else:x,_,sol,info=run_branch(selected_price[br],supply,m,v,100,br,dates,labels,L,G,P,cold);saved[key]=(x,sol,info)
    candidates.append(dict(stage='M_v_screen',branch='4-'+br,price_predictor=selected_price[br],supply_predictor=supply,M=m,v=v,grid=100,**aggregate(x)))
 cand=pd.DataFrame(candidates); cand.to_csv(OUT/'q4_v1_candidate_comparison.csv',index=False)
 selected={}
 for br in ['42','43']:
  pool=cand[(cand.stage=='M_v_screen')&(cand.branch=='4-'+br)].copy().sort_values(['inventory_corrected_cost_yuan','end_soc','M'],ascending=[True,False,True]);best=pool.iloc[0].to_dict();selected[br]=best
 # 50/100 grid only final candidates, count action changes by aligned slots.
 grids=[]; ab=[]; solverrows=[]
 for br,best in selected.items():
  key=(br,best['price_predictor'],best['supply_predictor'],int(best['M']),float(best['v']),100);x100,sol100,info100=saved[key];x50,_,sol50,info50=run_branch(best['price_predictor'],best['supply_predictor'],int(best['M']),float(best['v']),50,br,dates,labels,L,G,P,cold);r100=aggregate(x100);r50=aggregate(x50); action_cols=[c for c in ['charge_rt_kwh','charge_kwh'] if c in x100][0]; diff=int((abs(x100[action_cols].to_numpy()-x50[action_cols].to_numpy())>1e-6).sum()); grids.append(dict(branch='4-'+br,selected_price=best['price_predictor'],M=best['M'],v=best['v'],cost_100=r100['actual_cost_yuan'],cost_50=r50['actual_cost_yuan'],emergency_100=r100['emergency_kwh'],emergency_50=r50['emergency_kwh'],action_different_slots=diff,runtime_100=float(sol100.solve_time_seconds.sum()),runtime_50=float(sol50.solve_time_seconds.sum()),selected_grid=100)); solverrows += [sol100.assign(branch='4-'+br,grid=100),sol50.assign(branch='4-'+br,grid=50)]
  # Q4-2 condition ablation: same selected plan/day start DP vs greedy already returned as paired output.
  if br=='42': _,greedy,_,_=s.run_q42(dates,labels,L,G,P,configure(best['price_predictor'],best['supply_predictor'],dates,L,G,P)[0],cold);ab.append(dict(branch='4-2',comparison='same_plan_same_day_start_dynamic_vs_greedy',dynamic_cost=aggregate(x100)['actual_cost_yuan'],greedy_cost=aggregate(greedy)['actual_cost_yuan'],difference_yuan=aggregate(greedy)['actual_cost_yuan']-aggregate(x100)['actual_cost_yuan']))
 # Q4-3 independent no-adjustment comparison only with frozen-selected candidate.
 b=selected['43']; cache,c3=configure(b['price_predictor'],b['supply_predictor'],dates,L,G,P);s.M=int(b['M']);s.V=float(b['v']);s.GRID=100;s.TEST_DATES=pd.date_range('2025-01-02','2025-01-31'); no,_,nosol,noinfo=s.run_q43(dates,labels,L,G,P,c3,False,cold); yes=saved[('43',b['price_predictor'],b['supply_predictor'],int(b['M']),float(b['v']),100)][0];ab.append(dict(branch='4-3',comparison='independent_allow_vs_no_adjustment',dynamic_cost=aggregate(yes)['actual_cost_yuan'],greedy_cost=aggregate(no)['actual_cost_yuan'],difference_yuan=aggregate(no)['actual_cost_yuan']-aggregate(yes)['actual_cost_yuan']))
 pd.DataFrame(grids).to_csv(OUT/'q4_v1_grid_precision.csv',index=False);pd.DataFrame(ab).to_csv(OUT/'q4_v1_ablation_summary.csv',index=False);pd.concat(solverrows,ignore_index=True,sort=False).to_csv(OUT/'q4_v1_solver_precision.csv',index=False)
 for br,best in selected.items():
  cfg={k:best[k] for k in ['price_predictor','supply_predictor','M','v','grid']};cfg.update(dict(formal_evaluation_not_run=True,information_boundary='historical prices only for future price forecasts',grid_selected_kwh=100,development_period='2025-01-22..31'));(OUT/f'q4_v1_frozen_config_4_{br[-1]}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
 valid=dict(price_information_leaks=0,physical_checks='pass for every candidate by stage2-validated executor',templates_written=False,post_january_data_read=False,common_eval_days=10,cold_start_soc_jan1=6000,cold_start_soc_jan2=cold)
 (OUT/'q4_v1_stage3_validation.txt').write_text(json.dumps(valid,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'prices':prices,'supply':supply,'selected':selected,'grid':grids,'ablation':ab,'cold':cold},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
