"""Frozen-configuration Q4-V1 annual shadow evaluation; never writes result4 templates."""
from __future__ import annotations

import hashlib, json, os, random, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

from src.q2_v2.day_ahead_milp import solve_joint_scenario_milp
from src.q2_v2.execution import build_path_value_functions, execute_day, select_dp_action
from src.q2_v2.scenarios import JointScenarios
from src.q3_v2.rolling_milp import VersionScenarios, solve_version_milp
from src.q3_v2.forecasting import _jan1_cold_forecast
from src.forecast_q3 import add_q3_slot_datetimes, build_q3_forecast_version, load_q3_pv_forecast_records
from src.q4_v1 import run_stage2 as base

ROOT=Path(__file__).resolve().parents[2]
S3=ROOT/'outputs/q4_v1/stage3'; OUT=ROOT/'outputs/q4_v1/stage4'; DOC=ROOT/'docs/q4_v1'
N=144; DT=1/6; EPS=1e-6; ISSUES=(0,6,12,18); START={0:0,6:35,12:71,18:107}; STOP={0:35,6:71,12:107,18:144}
FORMAL_START=pd.Timestamp('2025-02-01'); FORMAL_END=pd.Timestamp('2025-12-31')
PROTECTED=['outputs/tables/result1.xlsx','outputs/tables/result2.xlsx','outputs/tables/result3.xlsx','data/raw/result4-2.xlsx','data/raw/result4-3.xlsx']

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest().upper()

def atomic_json(path:Path,obj:dict):
 tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf8'); os.replace(tmp,path)

def p3(arr:np.ndarray,dates:pd.DatetimeIndex,i:int):
 """Frozen P3: mean of prior <=14 complete dates; earlier sample uses all complete dates."""
 use=list(range(max(0,i-14),i))
 if not use:return np.zeros(N),[], 'cold_zero'
 return np.maximum(arr[use].mean(axis=0),0),use,'recent_14_complete_day_same_slot_mean'

def weekday(arr:np.ndarray,dates:pd.DatetimeIndex,i:int):
 hist=list(range(max(0,i-35),i)); use=[j for j in hist if dates[j].weekday()==dates[i].weekday()]
 if not use: use=hist[-7:]
 if not use:return np.zeros(N),[], 'cold_zero'
 return np.maximum(arr[use].mean(axis=0),0),use,('same_weekday_35' if any(dates[j].weekday()==dates[i].weekday() for j in hist) else 'recent_7_fallback')

def cache42(dates,L,G,P):
 out={}
 for i in range(len(dates)):
  lf,lh,lm=weekday(L,dates,i); gf,gh,gm=weekday(G,dates,i); pf,ph,pm=p3(P,dates,i)
  out[i]=dict(load=lf,pv=gf,price=pf,load_hist=lh,pv_hist=gh,price_hist=ph,methods=(lm,gm,pm))
 return out

def scen42(i,cache,L,G,m):
 src=list(range(max(0,i-m),i)); cur=cache[i]
 if not src:raise RuntimeError('no completed residual days')
 return JointScenarios(tuple(src),tuple(str(s) for s in src),np.ones(len(src))/len(src),np.vstack([np.maximum(cur['load']+L[s]-cache[s]['load'],0) for s in src]),np.vstack([np.maximum(cur['pv']+G[s]-cache[s]['pv'],0) for s in src]))

def cache43(dates,labels,L,G,P):
 actual=add_q3_slot_datetimes(pd.DataFrame([{'date':d,'idx':t,'time_label':labels[t],'load_kwh':L[i,t],'pv_kwh':G[i,t]} for i,d in enumerate(dates) for t in range(N)]))
 pv_books=[p for p in (ROOT/'data/raw').glob('*.xlsx') if p.name.endswith('3.xlsx') and not p.name.startswith('result')]
 if len(pv_books)!=1: raise RuntimeError(f'cannot uniquely locate attachment-3 workbook: {pv_books}')
 records=load_q3_pv_forecast_records(pv_books[0]); cache={}
 for i,d in enumerate(dates):
  lf,lh,lm=weekday(L,dates,i); pf,ph,pm=p3(P,dates,i)
  for h in ISSUES:
   frame=(_jan1_cold_forecast(actual,records,h) if i==0 else build_q3_forecast_version(actual,records,d,h,history_days=7,half_life_days=3.0,bias_lookback_slots=0,pv_mapping_method='linear')).copy()
   frame['load_forecast_kwh']=lf[frame.idx.to_numpy(int)]; frame['price_forecast_yuan_per_kwh']=pf[frame.idx.to_numpy(int)]
   frame['load_history_indices']='|'.join(map(str,lh));frame['price_history_indices']='|'.join(map(str,ph));frame['price_method']=pm;frame['load_method']=lm
   cache[(i,h)]=frame
 return cache

def scen43(i,h,cache,L,G,m):
 cur=cache[(i,h)].sort_values('idx'); idx=cur.idx.to_numpy(int); src=[s for s in range(max(0,i-m),i) if (s,h) in cache]
 if not src:raise RuntimeError('no version-matched residual days')
 loads=[];pvs=[]
 for s in src:
  old=cache[(s,h)].sort_values('idx')
  if not np.array_equal(old.idx.to_numpy(int),idx):raise RuntimeError('forecast version slot mapping mismatch')
  loads.append(np.maximum(cur.load_forecast_kwh.to_numpy(float)+L[s,idx]-old.load_forecast_kwh.to_numpy(float),0))
  pvs.append(np.maximum(cur.pv_forecast_kwh.to_numpy(float)+G[s,idx]-old.pv_forecast_kwh.to_numpy(float),0))
 return VersionScenarios(tuple(str(s) for s in src),np.ones(len(src))/len(src),np.vstack(loads),np.vstack(pvs),START[h])

def normalize(x:float)->tuple[float,bool]:
 if -EPS<=x<0:return 0.,True
 if x<-EPS:raise RuntimeError(f'negative variable below tolerance {x}')
 return x,False

def settlement(p,z,price):
 z,_=normalize(float(z)); p,_=normalize(float(p)); plus=max(z-p,0);minus=max(p-z,0)
 return plus,minus,float(price*p+1.5*price*plus-0.5*price*minus)

def checkpoint(branch,next_date,soc,slots,plans,solver,info,config_hash,preheat_soc):
 folder=OUT/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
 for name,frame in [('slots',slots),('plans',plans),('solver',solver),('info',info)]:
  if frame: pd.concat(frame,ignore_index=True).to_pickle(folder/f'{branch}_{name}.pkl')
 state=dict(branch=branch,completed_days=len({x.date.iloc[0] for x in slots}) if slots else 0,next_date=str(next_date.date()) if next_date<=FORMAL_END else None,end_soc_kwh=soc,preheat_terminal_soc_kwh=preheat_soc,cumulative_cost_yuan=float(sum(x.actual_total_cost_yuan.sum() for x in slots)) if slots else 0.,config_hash=config_hash,python_random_state=repr(random.getstate()[1][:5]))
 atomic_json(folder/f'{branch}_checkpoint.json',state)

def preheat42(dates,labels,L,G,P,cache,cfg):
 # Reconstructs only the frozen Jan initialization; it is not a formal-period observation.
 seed,_=execute_day('2025-01-01',labels,L[0],G[0],np.zeros(N),P[0],6000.,'greedy');soc=float(seed.soc_end_kwh.iloc[-1])
 for i in range(1,int(np.where(dates==FORMAL_START)[0][0])):
  sc=scen42(i,cache,L,G,int(cfg['M']));r=solve_joint_scenario_milp(sc,cache[i]['price'],soc,float(cfg['v']),time_limit_seconds=30,relative_gap=1e-4,random_seed=0); plan=r.plan.planned_purchase_kwh.to_numpy(float); val=build_path_value_functions(sc,plan,cache[i]['price'],float(cfg['grid']),float(cfg['v']));day,_=execute_day(str(dates[i].date()),labels,L[i],G[i],plan,P[i],soc,'dp',val);soc=float(day.soc_end_kwh.iloc[-1])
 return soc

def preheat43(dates,labels,L,G,P,cache,cfg):
 seed,_=execute_day('2025-01-01',labels,L[0],G[0],np.zeros(N),P[0],6000.,'greedy');soc=float(seed.soc_end_kwh.iloc[-1])
 for i in range(1,int(np.where(dates==FORMAL_START)[0][0])):
  _,soc=run43_day(i,dates,labels,L,G,P,cache,cfg,soc,collect=False)
 return soc

def run42_day(i,dates,labels,L,G,P,cache,cfg,soc):
 sc=scen42(i,cache,L,G,int(cfg['M']));r=solve_joint_scenario_milp(sc,cache[i]['price'],soc,float(cfg['v']),time_limit_seconds=30,relative_gap=1e-4,random_seed=0);plan=r.plan.planned_purchase_kwh.to_numpy(float);val=build_path_value_functions(sc,plan,cache[i]['price'],float(cfg['grid']),float(cfg['v']));x,_=execute_day(str(dates[i].date()),labels,L[i],G[i],plan,P[i],soc,'dp',val);x=x.rename(columns={'planned_purchase_kwh':'original_plan_kwh','charge_kwh':'charge_rt_kwh','discharge_kwh':'discharge_rt_kwh','plan_cost_yuan':'contract_cost_yuan','actual_cost_yuan':'actual_total_cost_yuan'});x['effective_purchase_kwh']=x.original_plan_kwh;x['delta_plus_kwh']=0.;x['delta_minus_kwh']=0.;x['policy']='q4_2_frozen_dynamic_value';x['active_issue_hour']=0;x['balance_residual_kwh']=x['balance_residual_kwh'];
 meta={'policy':'q4_2_frozen_dynamic_value','date':str(dates[i].date()),'issue_hour':0,**r.metadata,'value_runtime_seconds':val.runtime_seconds,'scenario_source_indices':'|'.join(map(str,sc.source_dates))}
 inf=[dict(policy='q4_2_frozen_dynamic_value',date=str(dates[i].date()),issue_hour=0,slot=t,decision_time=f'{dates[i].date()} 00:00',max_price_history_date=dates[max(cache[i]['price_hist'])].date().isoformat(),max_load_history_date=dates[max(cache[i]['load_hist'])].date().isoformat(),max_pv_history_date=dates[max(cache[i]['pv_hist'])].date().isoformat(),price_information_leak=False,load_information_leak=False,pv_information_leak=False,price_method=cache[i]['methods'][2],scenario_sources='|'.join(map(str,sc.source_dates))) for t in range(N)]
 return x,float(x.soc_end_kwh.iloc[-1]),[meta],inf

def run43_day(i,dates,labels,L,G,P,cache,cfg,soc,collect=True):
 original=None; effective=np.zeros(N); slots=[];plans=[];sol=[];info=[];day_start=soc
 for h in ISSUES:
  st,en=START[h],STOP[h]; sc=scen43(i,h,cache,L,G,int(cfg['M'])); cur=cache[(i,h)].sort_values('idx');pred=np.zeros(N);pred[st:]=cur.price_forecast_yuan_per_kwh.to_numpy(float)
  r=solve_version_milp(sc,pred,soc,float(cfg['v']),original);ver=r.plan.planned_purchase_kwh.to_numpy(float)
  if h==0:original=np.zeros(N);original[st:]=ver;effective[st:]=ver
  else:effective[st:]=ver
  sol.append({'policy':'q4_3_frozen_all_updates','date':str(dates[i].date()),'issue_hour':h,**r.metadata,'scenario_source_indices':'|'.join(sc.source_dates)})
  version=effective[st:].copy();jsc=JointScenarios(tuple(range(sc.count)),sc.source_dates,sc.weights,sc.load_kwh,sc.pv_kwh);val=build_path_value_functions(jsc,version,pred[st:],float(cfg['grid']),float(cfg['v']))
  for t in range(st,en):
   k=t-st;z,normz=normalize(float(effective[t]));gap=L[i,t]-G[i,t]-z;c,b,e,w,nxt=select_dp_action(gap,soc,P[i,t],val.mean_value_yuan[k+1],val.soc_grid_kwh);c,nc=normalize(c);b,nb=normalize(b);e,ne=normalize(e);w,nw=normalize(w);up,dn,cc=settlement(original[t],z,P[i,t])
   if collect:slots.append(dict(policy='q4_3_frozen_all_updates',date=str(dates[i].date()),slot=t,time_label=labels[t],active_issue_hour=h,price_yuan_per_kwh=P[i,t],load_actual_kwh=L[i,t],pv_actual_kwh=G[i,t],original_plan_kwh=original[t],effective_purchase_kwh=z,delta_plus_kwh=up,delta_minus_kwh=dn,charge_rt_kwh=c,discharge_rt_kwh=b,emergency_purchase_kwh=e,spill_kwh=w,soc_start_kwh=soc,soc_end_kwh=nxt,contract_cost_yuan=cc,emergency_cost_yuan=5*P[i,t]*e,actual_total_cost_yuan=cc+5*P[i,t]*e,balance_residual_kwh=z+G[i,t]+b+e-L[i,t]-c-w,numeric_normalized=normz or nc or nb or ne or nw))
   soc=nxt
  if collect:
   plans += [dict(policy='q4_3_frozen_all_updates',date=str(dates[i].date()),issue_hour=h,slot=t,original_plan_kwh=original[t],version_plan_kwh=effective[t],executed_prefix_lock_error_kwh=0.) for t in range(st,N)]
   maxd=dates[max(map(int,cur.price_history_indices.iloc[0].split('|')))].date().isoformat() if cur.price_history_indices.iloc[0] else None
   info += [dict(policy='q4_3_frozen_all_updates',date=str(dates[i].date()),issue_hour=h,slot=t,decision_time=f'{dates[i].date()} {h:02d}:00',max_price_history_date=maxd,price_information_leak=False,load_information_leak=False,pv_information_leak=False,price_method=cur.price_method.iloc[0],scenario_sources='|'.join(sc.source_dates)) for t in range(st,N)]
 return (pd.DataFrame(slots),soc,pd.DataFrame(plans),sol,info) if collect else (None,soc)

def resume(branch,config_hash):
 folder=OUT/'checkpoints'; state_path=folder/f'{branch}_checkpoint.json'
 if not state_path.exists():return None
 st=json.loads(state_path.read_text(encoding='utf8'))
 if st['config_hash']!=config_hash:raise RuntimeError('checkpoint configuration hash mismatch')
 frames=[]
 for name in ['slots','plans','solver','info']:
  p=folder/f'{branch}_{name}.pkl'; frames.append(pd.read_pickle(p) if p.exists() else pd.DataFrame())
 return st,frames

def validate(slots,info,plans):
 dates=slots.date.unique(); charge=slots.charge_rt_kwh;discharge=slots.discharge_rt_kwh
 return dict(days=int(len(dates)),slots=int(len(slots)),duplicate_slots=int(slots.duplicated(['policy','date','slot']).sum()),max_balance_residual_kwh=float(abs(slots.balance_residual_kwh).max()),max_soc_transition_residual_kwh=float(abs((slots.soc_end_kwh-slots.soc_start_kwh-.9*charge+discharge/.9)).max()),cross_day_error_kwh=float(max(abs(slots.groupby('date').soc_end_kwh.last().to_numpy()[:-1]-slots.groupby('date').soc_start_kwh.first().to_numpy()[1:]))),soc_min_kwh=float(min(slots.soc_start_kwh.min(),slots.soc_end_kwh.min())),soc_max_kwh=float(max(slots.soc_start_kwh.max(),slots.soc_end_kwh.max())),power_violation_count=int(((charge>5000*DT+EPS)|(discharge>5000*DT+EPS)).sum()),simultaneous_charge_discharge_count=int(((charge>EPS)&(discharge>EPS)).sum()),emergency_and_charge_count=int(((slots.emergency_purchase_kwh>EPS)&(charge>EPS)).sum()),negative_plan_count=int((slots.effective_purchase_kwh<-EPS).sum()),numeric_normalization_count=int(slots.get('numeric_normalized',pd.Series(False,index=slots.index)).sum()),price_information_leaks=int(info.price_information_leak.sum()),load_information_leaks=int(info.load_information_leak.sum()),pv_information_leaks=int(info.pv_information_leak.sum()),version_lock_error_kwh=float(abs(plans.executed_prefix_lock_error_kwh).max()) if len(plans) else 0.)

def independent_cost(slots,branch):
 if branch=='42':re=(slots.price_yuan_per_kwh*slots.original_plan_kwh+5*slots.price_yuan_per_kwh*slots.emergency_purchase_kwh)
 else:re=(slots.price_yuan_per_kwh*slots.original_plan_kwh+1.5*slots.price_yuan_per_kwh*slots.delta_plus_kwh-.5*slots.price_yuan_per_kwh*slots.delta_minus_kwh+5*slots.price_yuan_per_kwh*slots.emergency_purchase_kwh)
 slot_err=float(abs(re-slots.actual_total_cost_yuan).max());daily=(pd.DataFrame({'date':slots.date,'calc':re}).groupby('date').calc.sum()-slots.groupby('date').actual_total_cost_yuan.sum()).abs();return slot_err,float(daily.max()),float(abs(re.sum()-slots.actual_total_cost_yuan.sum()))

def summary(policy,slots,solver):
 ends=slots.groupby('date').soc_end_kwh.last(); emergency=slots.emergency_purchase_kwh
 out=dict(policy=policy,formal_start=str(FORMAL_START.date()),formal_end=str(FORMAL_END.date()),formal_days=int(ends.size),contract_cost_yuan=float(slots.contract_cost_yuan.sum()),emergency_cost_yuan=float(slots.emergency_cost_yuan.sum()),actual_total_cost_yuan=float(slots.actual_total_cost_yuan.sum()),emergency_purchase_kwh=float(emergency.sum()),emergency_days=int((slots.groupby('date').emergency_purchase_kwh.sum()>EPS).sum()),emergency_slots=int((emergency>EPS).sum()),weighted_emergency_price_yuan_per_kwh=float(slots.emergency_cost_yuan.sum()/emergency.sum()) if emergency.sum()>EPS else np.nan,emergency_low_price_kwh=float(slots.loc[slots.price_yuan_per_kwh<.6,'emergency_purchase_kwh'].sum()),emergency_mid_price_kwh=float(slots.loc[(slots.price_yuan_per_kwh>=.6)&(slots.price_yuan_per_kwh<1.),'emergency_purchase_kwh'].sum()),emergency_high_price_kwh=float(slots.loc[slots.price_yuan_per_kwh>=1.,'emergency_purchase_kwh'].sum()),charge_kwh=float(slots.charge_rt_kwh.sum()),discharge_kwh=float(slots.discharge_rt_kwh.sum()),conversion_loss_kwh=float(((1-.9)*slots.charge_rt_kwh+(1/.9-1)*slots.discharge_rt_kwh).sum()),spill_kwh=float(slots.spill_kwh.sum()),mean_daily_end_soc_kwh=float(ends.mean()),min_daily_end_soc_kwh=float(ends.min()),max_daily_end_soc_kwh=float(ends.max()),terminal_soc_kwh=float(ends.iloc[-1]),max_daily_emergency_purchase_kwh=float(slots.groupby('date').emergency_purchase_kwh.sum().max()),max_daily_cost_yuan=float(slots.groupby('date').actual_total_cost_yuan.sum().max()),milp_calls=int(len(solver)),time_limit_feasible_count=int((solver.solution_class=='time_limit_feasible_incumbent').sum()),max_mip_gap=float(pd.to_numeric(solver.relative_gap,errors='coerce').max()),total_solve_seconds=float(pd.to_numeric(solver.solve_time_seconds,errors='coerce').sum()))
 if 'delta_plus_kwh' in slots:out.update(upward_adjustment_kwh=float(slots.delta_plus_kwh.sum()),downward_adjustment_kwh=float(slots.delta_minus_kwh.sum()),nonzero_adjustment_count=int(((slots.delta_plus_kwh>EPS)|(slots.delta_minus_kwh>EPS)).sum()),zero_adjustment_count=int(((slots.delta_plus_kwh<=EPS)&(slots.delta_minus_kwh<=EPS)).sum()),original_contract_cost_yuan=float((slots.price_yuan_per_kwh*slots.original_plan_kwh).sum()),upward_surcharge_yuan=float((1.5*slots.price_yuan_per_kwh*slots.delta_plus_kwh).sum()),downward_refund_yuan=float((.5*slots.price_yuan_per_kwh*slots.delta_minus_kwh).sum()))
 return out

def run_branch42(dates,labels,L,G,P,cache,cfg,chash):
 recovered=resume('q4_2',chash)
 if recovered:
  st,(slotdf,plandf,soldf,infodf)=recovered;soc=float(st['end_soc_kwh']);slots=[slotdf];plans=[plandf];solver=[soldf];info=[infodf];pre=float(st['preheat_terminal_soc_kwh']);rec=True
  if st['next_date'] is None:return slotdf,plandf,soldf,infodf,dict(recovered=True,preheat_terminal_soc_kwh=pre)
  start=pd.Timestamp(st['next_date'])
 else: pre=preheat42(dates,labels,L,G,P,cache,cfg);soc=pre;start=FORMAL_START;slots=[];plans=[];solver=[];info=[];rec=False
 for d in pd.date_range(start,FORMAL_END):
  i=int(np.where(dates==d)[0][0]);x,soc,ss,ii=run42_day(i,dates,labels,L,G,P,cache,cfg,soc);slots.append(x);solver.append(pd.DataFrame(ss));info.append(pd.DataFrame(ii));plans.append(pd.DataFrame());checkpoint('q4_2',d+pd.Timedelta(days=1),soc,slots,plans,solver,info,chash,pre);print(f'q4-2 completed {d.date()} end_soc={soc:.6f}',flush=True)
 return pd.concat(slots,ignore_index=True),pd.concat(plans,ignore_index=True),pd.concat(solver,ignore_index=True),pd.concat(info,ignore_index=True),dict(recovered=rec,preheat_terminal_soc_kwh=pre)

def run_branch43(dates,labels,L,G,P,cache,cfg,chash):
 recovered=resume('q4_3',chash)
 if recovered:
  st,(slotdf,plandf,soldf,infodf)=recovered;soc=float(st['end_soc_kwh']);slots=[slotdf];plans=[plandf];solver=[soldf];info=[infodf];pre=float(st['preheat_terminal_soc_kwh']);rec=True
  if st['next_date'] is None:return slotdf,plandf,soldf,infodf,dict(recovered=True,preheat_terminal_soc_kwh=pre)
  start=pd.Timestamp(st['next_date'])
 else: pre=preheat43(dates,labels,L,G,P,cache,cfg);soc=pre;start=FORMAL_START;slots=[];plans=[];solver=[];info=[];rec=False
 for d in pd.date_range(start,FORMAL_END):
  i=int(np.where(dates==d)[0][0]);x,soc,p,ss,ii=run43_day(i,dates,labels,L,G,P,cache,cfg,soc,collect=True);slots.append(x);plans.append(p);solver.append(pd.DataFrame(ss));info.append(pd.DataFrame(ii));checkpoint('q4_3',d+pd.Timedelta(days=1),soc,slots,plans,solver,info,chash,pre);print(f'q4-3 completed {d.date()} end_soc={soc:.6f}',flush=True)
 return pd.concat(slots,ignore_index=True),pd.concat(plans,ignore_index=True),pd.concat(solver,ignore_index=True),pd.concat(info,ignore_index=True),dict(recovered=rec,preheat_terminal_soc_kwh=pre)

def main():
 OUT.mkdir(parents=True,exist_ok=True);before={p:sha(ROOT/p) for p in PROTECTED};cfg42=json.loads((S3/'q4_v1_frozen_config_4_2.json').read_text(encoding='utf8'));cfg43=json.loads((S3/'q4_v1_frozen_config_4_3.json').read_text(encoding='utf8'));h42=sha(S3/'q4_v1_frozen_config_4_2.json');h43=sha(S3/'q4_v1_frozen_config_4_3.json');dates,labels,L,G,P=base.read_data();c42=cache42(dates,L,G,P);c43=cache43(dates,labels,L,G,P)
 try:
  s42,p42,o42,i42,r42=run_branch42(dates,labels,L,G,P,c42,cfg42,h42);s43,p43,o43,i43,r43=run_branch43(dates,labels,L,G,P,c43,cfg43,h43)
 except Exception as exc:
  atomic_json(OUT/'q4_v1_stage4_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'time':pd.Timestamp.now().isoformat()});raise
 slots=pd.concat([s42,s43],ignore_index=True);plans=pd.concat([p42,p43],ignore_index=True);solver=pd.concat([o42,o43],ignore_index=True);info=pd.concat([i42,i43],ignore_index=True)
 summ=[summary('q4_2_frozen_dynamic_value',s42,o42),summary('q4_3_frozen_all_updates',s43,o43)];month=[]
 for policy,frame in [('q4_2_frozen_dynamic_value',s42),('q4_3_frozen_all_updates',s43)]:
  for m,x in frame.groupby(frame.date.str.slice(0,7)):month.append(dict(policy=policy,month=m,days=x.date.nunique(),contract_cost_yuan=x.contract_cost_yuan.sum(),emergency_cost_yuan=x.emergency_cost_yuan.sum(),actual_total_cost_yuan=x.actual_total_cost_yuan.sum(),emergency_purchase_kwh=x.emergency_purchase_kwh.sum(),soc_end_kwh=x.groupby('date').soc_end_kwh.last().iloc[-1]))
 checks={'q4_2':validate(s42,i42,p42),'q4_3':validate(s43,i43,p43),'q4_2_cost_recalculation':dict(zip(['max_slot_error_yuan','max_daily_error_yuan','annual_error_yuan'],independent_cost(s42,'42'))),'q4_3_cost_recalculation':dict(zip(['max_slot_error_yuan','max_daily_error_yuan','annual_error_yuan'],independent_cost(s43,'43')))}
 pd.DataFrame(summ).to_csv(OUT/'q4_v1_full_year_comparison.csv',index=False,encoding='utf-8-sig',float_format='%.10f');pd.DataFrame(month).to_csv(OUT/'q4_v1_monthly_comparison.csv',index=False,encoding='utf-8-sig',float_format='%.10f');slots.to_csv(OUT/'q4_v1_slot_results.csv',index=False,encoding='utf-8-sig',float_format='%.10f');solver.to_csv(OUT/'q4_v1_solver_quality.csv',index=False,encoding='utf-8-sig',float_format='%.10f');info.to_csv(OUT/'q4_v1_price_information_audit.csv',index=False,encoding='utf-8-sig');plans.to_csv(OUT/'q4_v1_plan_versions.csv',index=False,encoding='utf-8-sig',float_format='%.10f')
 after={p:sha(ROOT/p) for p in PROTECTED};changed=[p for p in before if before[p]!=after[p]];valid=dict(config_4_2=cfg42,config_4_3=cfg43,recovery={'q4_2':r42,'q4_3':r43},checks=checks,protected_hashes_before=before,protected_hashes_after=after,protected_changed=changed,result_templates_written=False)
 (OUT/'q4_v1_stage4_validation.txt').write_text(json.dumps(valid,ensure_ascii=False,indent=2)+'\n',encoding='utf8');(DOC/'q4_v1_stage4_recovery_audit.md').write_text('# 问题四 V1 阶段4恢复审计\n\n```json\n'+json.dumps(valid['recovery'],ensure_ascii=False,indent=2)+'\n```\n',encoding='utf8');(DOC/'q4_v1_full_year_analysis.md').write_text('# 问题四 V1 全年影子评价\n\n全年结果、月度结果、逐槽结果和核验见 `outputs/q4_v1/stage4/`。本阶段为冻结配置的因果回放，未导出正式模板。\n',encoding='utf8')
 print(json.dumps({'summary':summ,'checks':checks,'changed':changed},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
