"""Deterministically replay frozen Q2 A+ to recover slot trajectories only."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from openpyxl import load_workbook
from src.config import OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, Q2_VALIDATION_TOLERANCE, S_MIN, S_MAX
from src.export_q2_submission import replay_aplus_for_export, TABLE3_DATES
from src.plot_style import (
 ACTUAL_NET_LOAD, AXIS_GREY, EMERGENCY_PURCHASE, FILL_ALPHA, FORECAST_NET_LOAD,
 GRID_GREY, PLAN_PURCHASE, SOC_LOWER_BOUND, SOC_MAIN, SOC_UPPER_BOUND,
)

ROOT=OUTPUT_DIR/'replay'/'q2_aplus'; TRAJ=ROOT/'q2_aplus_slot_trajectory.csv'; BEFORE=ROOT/'frozen_hashes_before.json'; AFTER=ROOT/'frozen_hashes_after.json'; CHECK=ROOT/'replay_consistency_check.json'
FROZEN={
'result2.xlsx':TABLE_DIR/'result2.xlsx','q2_full_year_shadow_comparison.csv':TABLE_DIR/'q2_full_year_shadow_comparison.csv','q2_daily_shadow_results.csv':TABLE_DIR/'q2_daily_shadow_results.csv','q2_paper_tables.csv':TABLE_DIR/'q2_paper_tables.csv'}
def hashes(): return {n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in FROZEN.items()}
def numdiff(a,b):
 a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float); return float(np.max(np.abs(a-b))) if len(a) else 0.0
def font():
 names={f.name for f in font_manager.fontManager.ttflist}
 return next((x for x in ('SimHei','Microsoft YaHei','Noto Sans CJK SC') if x in names),'DejaVu Sans')
def figure(trajectory):
 candidates=trajectory[trajectory.date.isin([d.date().isoformat() for d in TABLE3_DATES])].groupby('date').emergency_purchase_kwh.sum()
 date=candidates.idxmax(); day=trajectory[trajectory.date==date].sort_values('slot'); x=np.arange(144)/6; xs=np.arange(145)/6
 plt.rcParams['font.sans-serif']=[font()]; plt.rcParams['axes.unicode_minus']=False
 plt.rcParams.update({'font.size':10,'axes.labelsize':11,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':8.8,'lines.solid_capstyle':'round','pdf.fonttype':42,'svg.fonttype':'none'})
 fig,ax=plt.subplots(3,1,figsize=(8.4,8.5),sharex=True,facecolor='white')
 ax[0].step(x,day.net_load_pred_kwh,where='post',label='日前预测净负荷',color=FORECAST_NET_LOAD,lw=2.1); ax[0].step(x,day.net_load_actual_kwh,where='post',label='实际净负荷',color=ACTUAL_NET_LOAD,lw=2.1); ax[0].set_ylabel('净负荷（kWh）')
 ax[1].step(x,day.planned_purchase_kwh,where='post',label='固定计划购电量',color=PLAN_PURCHASE,lw=2.1); ax[1].step(x,day.emergency_purchase_kwh,where='post',label='实际紧急购电量',color=EMERGENCY_PURCHASE,lw=2.1); ax[1].set_ylabel('购电量（kWh）')
 soc=np.r_[day.soc_start_kwh.to_numpy(),day.soc_end_kwh.iloc[-1]]; ax[2].step(xs,soc,where='post',label='实际SOC',color=SOC_MAIN,lw=2.1); ax[2].fill_between(xs,soc,1200,color=SOC_MAIN,alpha=FILL_ALPHA); ax[2].axhline(1200,ls='--',color=SOC_LOWER_BOUND,lw=1.35,label='安全下限（1200 kWh）'); ax[2].axhline(10800,ls='--',color=SOC_UPPER_BOUND,lw=1.35,label='安全上限（10800 kWh）'); ax[2].set_ylabel('储电量（kWh）'); ax[2].set_xlabel('时间（h）')
 for a in ax:
  a.set_xlim(0,24); a.set_xticks(range(0,25,4)); a.set_axisbelow(True); a.grid(axis='y',color=GRID_GREY,ls='-',lw=.7); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.spines['left'].set_color(AXIS_GREY); a.spines['bottom'].set_color(AXIS_GREY); a.spines['left'].set_linewidth(.85); a.spines['bottom'].set_linewidth(.85); a.tick_params(axis='both',direction='out',width=.8,length=4,color=AXIS_GREY)
 for i,a in enumerate(ax): a.text(-.11,1.02,f'({chr(97+i)})',transform=a.transAxes,fontsize=10.5,fontweight='bold')
 ax[0].legend(frameon=False,loc='lower right',bbox_to_anchor=(1,1.005),ncol=2,columnspacing=1.2,handlelength=2.6)
 ax[1].legend(frameon=False,loc='lower right',bbox_to_anchor=(1,1.005),ncol=2,columnspacing=1.2,handlelength=2.6)
 ax[2].legend(frameon=False,loc='lower right',bbox_to_anchor=(1,1.005),ncol=3,columnspacing=1.1,handlelength=2.3)
 fig.subplots_adjust(hspace=.37); path=FIGURE_DIR/'q2_typical_day_dispatch.png'; fig.savefig(path,dpi=600,bbox_inches='tight',facecolor='white'); fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white'); plt.close(fig); return path,date
def _legacy_docs(trajectory, date, check):
 comparison=pd.read_csv(TABLE_DIR/'q2_full_year_shadow_comparison.csv'); b=comparison[comparison.strategy_key=='baseline'].iloc[0]; a=comparison[comparison.strategy_key=='aplus'].iloc[0]
 jan=pd.read_csv(TABLE_DIR/'q2_january_pareto_summary.csv'); base=jan[jan.category.str.contains('cost_baseline')].iloc[0]; reserve=jan[jan.category.str.contains('reserve_min_total')].iloc[0]
 material=Path(__file__).resolve().parents[1]/'docs'/'q2_writing_material.md'
 material.write_text(f'''# 问题 2 写作素材\n\n## 问题分析\n问题 2 是负载与光伏实际功率不确定条件下的日前购电计划与实时储能调度问题。每日 0:00 制定并固定计划购电量；若实际供给仍低于负载，则以当时正常电价的 5 倍紧急购电。\n\n## 信息集与因果性\n日期 \(d\) 的 0:00 仅使用此前已发生的实际净负荷、当日固定电价和当前 SOC；不使用附件 3。实时槽位仅使用当前实际净负荷和当前 SOC，不使用未来实际数据，也不调整计划购电量。问题 3 才允许基于新预报调整购电量。\n\n## 模型结构\n实际净负荷为 \(N_{{d,t}}=L_{{d,t}}-G_{{d,t}}\)。日前预测采用同槽位滚动分位数 \(\widehat N_{{d,t}}=Q_{{0.75}}\{{N_{{d-i,t}}:i=1,\ldots,7\}}\)。日前 LP 最小化 \(\sum_t\pi_tp_{{d,t}}+\rho u_d\)，并满足预测能量平衡、\(s_{{t+1}}=s_t+0.9c_t-b_t/0.9\)、SOC/功率边界。实际执行时计划购电固定，紧急购电仅补短缺，且 \(e_{{d,t}}>0\Rightarrow c_{{d,t}}^{{rt}}=0\)。SOC 按实际日末跨日传递。A+ 采用 \(S^{{res}}=3600\) kWh；\(\rho=0.6976\) 是日前参考路径的软可行性保护项，不称为最优经济参数。全年 \(u_d=0\)。\n\n## 参数选择与敏感性\n1 月共同回测中，成本基线 \(S^{{res}}=1200\) 的成本为 {base.total_cost_yuan:.3f} 元；A+候选 \(S^{{res}}=3600\) 的成本为 {reserve.total_cost_yuan:.3f} 元，增加 {reserve.total_cost_delta_vs_baseline_yuan:.3f} 元，紧急购电量减少 {-reserve.emergency_purchase_delta_vs_baseline_kwh:.3f} kWh，且最小日末 SOC 提高 {reserve.min_end_soc_delta_vs_baseline_kwh:.3f} kWh。回测灵敏度表显示分位数偏低会增加紧急购电风险，偏高会抬高部分窗口的计划购电成本；具体数值见 `q2_january_parameter_comparison.csv`，未作额外统计推断。\n\n## 全年结果\n正式统计期为 2025-02-01 至 2025-12-31，共 334 天。成本基线总成本 {b.total_actual_cost_yuan:.3f} 元、紧急购电量 {b.total_emergency_purchase_kwh:.3f} kWh、平均/最小日末 SOC 为 {b.mean_daily_end_soc_kwh:.3f}/{b.min_daily_end_soc_kwh:.3f} kWh。A+ 对应为 {a.total_actual_cost_yuan:.3f} 元、{a.total_emergency_purchase_kwh:.3f} kWh、{a.mean_daily_end_soc_kwh:.3f}/{a.min_daily_end_soc_kwh:.3f} kWh；相对变化为成本 {a.total_actual_cost_yuan-b.total_actual_cost_yuan:.3f} 元、紧急购电量 {a.total_emergency_purchase_kwh-b.total_emergency_purchase_kwh:.3f} kWh、平均 SOC {a.mean_daily_end_soc_kwh-b.mean_daily_end_soc_kwh:.3f} kWh。验证未发现未来数据泄漏、SOC/功率越界或紧急购电同时充电。1 月仅作冷启动和预热，两方案均从 6000 kWh 出发。\n\n## 表格与图题\n题面表 1、表 2、表 3 的机器可读真实数值见 `outputs/tables/q2_paper_tables.csv`；表 3 仅列实际发生的合并紧急购电事件。\n\n图 3 A+策略与成本基线全年对比。图中比较正式统计期总成本、紧急购电量和平均日末 SOC。\n\n图 4 典型日预测偏差与实时调度过程（{date}）。紧急购电由当前预测偏差及储能物理边界共同触发；图中曲线不表示使用未来实际信息。\n''',encoding='utf-8')
 checklist=Path(__file__).resolve().parents[1]/'docs'/'q2_completion_checklist.md'; freeze=Path(__file__).resolve().parents[1]/'docs'/'q2_freeze_record.md'; final=LOG_DIR/'q2_final_check.txt'
 lines=['# 问题 2 最终完成度检查','','| 项目 | 状态 | 证据 |','|---|---|---|','| result2 模板写入与写后核验 | 完成 | q2_submission_check.txt |','| 334×144 计划购电量 | 完成 | result2.xlsx |','| 实际充放电、SOC、紧急购电导出 | 完成 | result2.xlsx |','| 两张中文 300 dpi 正文图 | 完成 | q2_baseline_vs_aplus_summary.png；q2_typical_day_dispatch.png |','| 写作素材 | 完成 | q2_writing_material.md |','| 冻结文件哈希、逐槽/逐日/全年一致性 | 完成 | outputs/replay/q2_aplus/replay_consistency_check.json |','| 问题 1 冻结结果 | 未修改 | q1 文件未写入 |']
 checklist.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 freeze.write_text(f'''# 问题 2 冻结记录\n\n冻结方案：A+，\(w=7,q=0.75,S^{{res}}=3600\) kWh，\(\rho=0.6976\) 元/kWh。\n\n冻结结果：`outputs/tables/result2.xlsx`、`q2_paper_tables.csv`、全年影子比较表、逐日表和提交校验日志。\n\n重放验证：在 `outputs/replay/q2_aplus/` 保存逐槽轨迹及哈希/一致性报告；所有核心冻结文件重放前后哈希相同，逐槽、逐日、全年数值在 \(10^{{-6}}\) 容差内一致。\n\n后续问题不得修改问题 2 冻结数值、result2.xlsx 或参数；只有发现明确题面、单位或模板映射错误时才重新审计。\n''',encoding='utf-8')
 final.write_text('Question 2 final freeze check\n'+'='*36+'\n'+json.dumps(check,ensure_ascii=False,indent=2,default=str)+'\nfigure1: q2_baseline_vs_aplus_summary.png (existing frozen 300 dpi)\nfigure2: q2_typical_day_dispatch.png (replay true trajectory, 300 dpi)\nwriting_material_numbers_checked_against_frozen_summary: True\nquestion1_files_modified: False\n',encoding='utf-8')
 return material,checklist,freeze,final
def docs(trajectory, date, check):
 """Return the already frozen documentation paths without rewriting them.

 Figure selection is maintained directly in the frozen documents, so replaying
 slot trajectories cannot restore a figure that was explicitly removed.
 """
 docs_dir=Path(__file__).resolve().parents[1]/'docs'
 return (
  docs_dir/'q2_writing_material.md',
  docs_dir/'q2_completion_checklist.md',
  docs_dir/'q2_freeze_record.md',
  LOG_DIR/'q2_final_check.txt',
 )
def run():
 ROOT.mkdir(parents=True,exist_ok=True); before=hashes(); BEFORE.write_text(json.dumps(before,indent=2),encoding='utf-8')
 runs,checks=replay_aplus_for_export(); rows=[]
 for r in runs:
  p=r['plan'].sort_values('idx').reset_index(drop=True); e=r['execution'].sort_values('idx').reset_index(drop=True); f=r['forecast'].sort_values('idx').reset_index(drop=True)
  rows.append(pd.DataFrame({'date':r['date'].date().isoformat(),'slot':range(144),'time_label':e.time_label,'net_load_pred_kwh':f.forecast_net_load_kwh,'net_load_actual_kwh':e.actual_net_load_kwh,'planned_purchase_kwh':p.plan_purchase_kwh,'charge_rt_kwh':e.rt_charge_kwh,'discharge_rt_kwh':e.rt_discharge_kwh,'emergency_purchase_kwh':e.emergency_purchase_kwh,'curtailment_kwh':e.rt_curtail_kwh,'soc_start_kwh':e.rt_soc_start_kwh,'soc_end_kwh':e.rt_soc_end_kwh,'plan_cost_yuan':e.plan_purchase_cost_yuan,'emergency_cost_yuan':e.emergency_purchase_cost_yuan}))
 tr=pd.concat(rows,ignore_index=True); TRAJ.write_text(tr.to_csv(index=False,float_format='%.10f'),encoding='utf-8-sig')
 wb=load_workbook(TABLE_DIR/'result2.xlsx',data_only=True); ps=wb.worksheets[0]; excel=np.array([[ps.cell(i+2,j+2).value for j in range(144)] for i in range(334)],float); rep=tr.planned_purchase_kwh.to_numpy().reshape(334,144)
 daily=tr.groupby('date').agg(plan_purchase_cost_yuan=('plan_cost_yuan','sum'),emergency_purchase_cost_yuan=('emergency_cost_yuan','sum'),actual_total_cost_yuan=('plan_cost_yuan','sum'),emergency_purchase_kwh=('emergency_purchase_kwh','sum'),soc_end_kwh=('soc_end_kwh','last')).reset_index(); daily.actual_total_cost_yuan+=daily.emergency_purchase_cost_yuan
 frozen=pd.read_csv(TABLE_DIR/'q2_daily_shadow_results.csv'); frozen=frozen[frozen.strategy_key=='aplus'].sort_values('date').reset_index(drop=True)
 dmax=max(numdiff(daily[c],frozen[c]) for c in ['plan_purchase_cost_yuan','emergency_purchase_cost_yuan','actual_total_cost_yuan','emergency_purchase_kwh','soc_end_kwh'])
 summary=pd.read_csv(TABLE_DIR/'q2_full_year_shadow_comparison.csv'); a=summary[summary.strategy_key=='aplus'].iloc[0]
 annual={'total_actual_cost_yuan':tr.plan_cost_yuan.sum()+tr.emergency_cost_yuan.sum(),'total_plan_purchase_cost_yuan':tr.plan_cost_yuan.sum(),'total_emergency_purchase_cost_yuan':tr.emergency_cost_yuan.sum(),'total_emergency_purchase_kwh':tr.emergency_purchase_kwh.sum(),'mean_daily_end_soc_kwh':daily.soc_end_kwh.mean(),'min_daily_end_soc_kwh':daily.soc_end_kwh.min()}; amax=max(abs(annual[k]-float(a[k])) for k in annual)
 after=hashes(); AFTER.write_text(json.dumps(after,indent=2),encoding='utf-8'); check={'tolerance':1e-6,'hashes_unchanged':before==after,'slot_count':len(tr),'slot_purchase_max_abs_difference':numdiff(excel.ravel(),rep.ravel()),'daily_max_abs_difference':dmax,'annual_max_abs_difference':amax,'replay_physical_checks':checks}; check['passed']=check['hashes_unchanged'] and check['slot_count']==48096 and check['slot_purchase_max_abs_difference']<=1e-6 and dmax<=1e-6 and amax<=1e-6
 CHECK.write_text(json.dumps(check,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 if not check['passed']: raise RuntimeError('Replay mismatch: no figure or freeze files generated; see '+str(CHECK))
 fig,date=figure(tr); files=docs(tr,date,check); return fig,date,files,check
if __name__=='__main__':
 out=run(); print(out[0]); print(out[1]); print(out[3]['passed'])
