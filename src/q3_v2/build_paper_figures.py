"""Paper figures for frozen Q3 V2 results; reads CSVs only, never solves."""
from __future__ import annotations

import html
from base64 import b64encode
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from src.paper_figure_theme import configure_chinese_theme

ROOT = Path(__file__).resolve().parents[2]
S4 = ROOT / "outputs/q3_v2/stage4"
OUT = ROOT / "outputs/figures/q3_v2"
DT = 1 / 6
A = "A_all_updates_dynamic_value"
B = "B_no_adjustment_dynamic_value"
BLUE, TEAL, ORANGE, RED, GREY, GRID = "#4573B4", "#66B2A3", "#E29D39", "#C44E52", "#6B7280", "#E8E8E8"

def style(ax, grid=True):
    ax.set_facecolor("white")
    for side in ax.spines:
        ax.spines[side].set_color("#333333"); ax.spines[side].set_linewidth(.75)
    ax.tick_params(labelsize=8, top=False, right=False)
    if grid: ax.grid(axis="y", color=GRID, linewidth=.7, zorder=0)

def title(ax, label, text): ax.set_title(f"({label}) {text}", loc="left", fontsize=10, fontweight="semibold", pad=8)
def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT/f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def flowchart():
    fig, ax = plt.subplots(figsize=(7.2,4.25)); ax.set_axis_off(); ax.set_xlim(0,12); ax.set_ylim(0,7)
    nodes=[(0.5,5.5,2.25,0.8,"0:00 原计划\n历史负载 + 0:00 PV预报\n14条联合残差情景",BLUE),
           (3.55,5.5,2.25,.8,"日前共同计划 MILP\n原计划 p，SOC参考路径",TEAL),
           (6.6,5.5,2.25,.8,"6:00 / 12:00 / 18:00\n仅更新未执行后缀",ORANGE),
           (9.65,5.5,1.9,.8,"版本锁定\n已执行槽不可改",GREY),
           (2.0,2.2,2.55,.9,"每10分钟因果执行\n当前实测供需、SOC、有效计划 z",BLUE),
           (5.1,2.2,2.45,.9,"动态价值储能调度\n后缀情景近似价值函数",TEAL),
           (8.15,2.2,2.55,.9,"一次合同结算 + 紧急购电\n相对原计划结算；5πe",RED)]
    for x,y,w,h,text,color in nodes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=.10",fc="white",ec=color,lw=1.25))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8,color="#253247")
    arrows=[((2.75,5.9),(3.55,5.9)),((5.8,5.9),(6.6,5.9)),((8.85,5.9),(9.65,5.9)),((7.7,5.5),(7.7,3.1)),((3.0,3.1),(3.0,3.0)),((4.55,2.65),(5.1,2.65)),((7.55,2.65),(8.15,2.65))]
    for start,end in arrows: ax.add_patch(FancyArrowPatch(start,end,arrowstyle="->",mutation_scale=10,lw=1.0,color="#6B7280"))
    ax.text(.55,6.65,"信息边界：每次决策只使用当时已发布预报、已完成历史/实测及当前 SOC",fontsize=8.2,color="#374151")
    ax.text(.55,.7,"实时执行不使用未来实际数据；SOC 跨日连续传递；终端库存价值仅用于优化，不计入实际账单。",fontsize=7.7,color="#6B7280")
    save(fig,"q3_v2_multistage_information_flow")
    # Native editable Draw.io structure; the matching PNG is embedded for fidelity.
    image=b64encode((OUT/"q3_v2_multistage_information_flow.png").read_bytes()).decode()
    xml=f'''<mxfile host="app.diagrams.net"><diagram id="q3v2flow" name="Page-1"><mxGraphModel page="1" pageWidth="1169" pageHeight="827"><root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="plot" value="" style="shape=image;imageAspect=1;aspect=fixed;image=data:image/png%3Bbase64,{image};strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="30" y="40" width="1100" height="650" as="geometry"/></mxCell></root></mxGraphModel></diagram></mxfile>'''
    (OUT/"q3_v2_multistage_information_flow.drawio").write_text(xml,encoding="utf-8")

def timing_diagram():
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.set_xlim(-0.4, 24.4); ax.set_ylim(-0.5, 3.6); ax.axis("off")
    ax.annotate("", xy=(24.1, 2.95), xytext=(0, 2.95), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))
    for hour in (0, 6, 12, 18, 24):
        ax.plot([hour, hour], [2.82, 3.08], color="#333333", lw=.8)
        ax.text(hour, 3.18, f"{hour}:00", ha="center", va="bottom", fontsize=8, color="#253247")
    ax.text(-.35, 2.95, "时间", ha="right", va="center", fontsize=8.5, color="#253247")
    ax.text(0, 2.45, "0:00原计划", ha="left", va="center", fontsize=9, fontweight="semibold", color=BLUE)
    ax.text(0, 2.13, "历史负载 + 已发布PV预报", ha="left", va="center", fontsize=7.5, color="#374151")
    ax.text(0, 1.94, "M=14联合残差情景", ha="left", va="center", fontsize=7.5, color="#374151")
    ax.plot([0, 6], [2.72, 2.72], color=BLUE, lw=5, solid_capstyle="butt", alpha=.18)
    for x, label in [(6, "6:00更新"), (12, "12:00更新"), (18, "18:00更新")]:
        ax.scatter([x], [2.72], s=48, color=ORANGE, zorder=4)
        ax.text(x, 2.45, label, ha="center", va="center", fontsize=8.5, fontweight="semibold", color=ORANGE)
        ax.text(x, 2.15, "仅优化未执行后缀", ha="center", va="center", fontsize=7.5, color="#374151")
    ax.text(12, 1.55, "每10分钟实时执行", ha="center", va="center", fontsize=9, fontweight="semibold", color=TEAL)
    ax.plot([0, 24], [1.82, 1.82], color=TEAL, lw=5, alpha=.18, solid_capstyle="butt")
    for x in np.arange(0, 24, DT):
        ax.plot([x, x + DT*.65], [1.82, 1.82], color=TEAL, lw=1.2)
    ax.text(12, 1.25, "当前实测供需、当前SOC、当前生效计划 → 动态价值储能动作", ha="center", va="center", fontsize=8, color="#253247")
    ax.text(12, .55, "已执行槽位", ha="center", va="center", fontsize=8.5, fontweight="semibold", color=GREY)
    ax.plot([0, 24], [.82, .82], color=GREY, lw=5, alpha=.2, solid_capstyle="butt")
    ax.text(12, .20, "版本锁定：后续更新不得追溯修改；日末SOC传递至下一日", ha="center", va="center", fontsize=8, color="#374151")
    for x in (6, 12, 18):
        ax.axvline(x, ymin=.08, ymax=.88, color=GREY, ls="--", lw=.8, alpha=.6)
    ax.text(24, 3.48, "信息边界与版本锁定", ha="right", va="top", fontsize=11, fontweight="semibold", color="#253247")
    ax.text(0, -0.25, "费用结算：每槽最终生效计划相对0:00原计划结算一次；紧急购电按5πe计费", ha="left", va="center", fontsize=7.8, color=RED)
    save(fig, "q3_v2_multistage_timeline")
    image=b64encode((OUT/"q3_v2_multistage_timeline.png").read_bytes()).decode()
    xml=f'''<mxfile host="app.diagrams.net"><diagram id="q3v2timeline" name="Page-1"><mxGraphModel page="1" pageWidth="1169" pageHeight="827"><root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="plot" value="" style="shape=image;imageAspect=1;aspect=fixed;image=data:image/png%3Bbase64,{image};strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="30" y="40" width="1100" height="520" as="geometry"/></mxCell></root></mxGraphModel></diagram></mxfile>'''
    (OUT/"q3_v2_multistage_timeline.drawio").write_text(xml, encoding="utf-8")

def annual():
    cmp=pd.read_csv(S4/"q3_v2_full_year_comparison.csv").set_index("policy")
    month=pd.read_csv(S4/"q3_v2_monthly_comparison.csv")
    a,b=cmp.loc[A],cmp.loc[B]
    if abs((b.actual_total_cost_yuan-a.actual_total_cost_yuan)-628939.7525409386)>1e-6: raise ValueError("frozen comparison mismatch")
    fig,axes=plt.subplots(1,3,figsize=(10.4,3.2),constrained_layout=True)
    names=["合同费用","紧急费用","总费用"]; x=np.arange(3); width=.32
    av=np.array([a.contract_cost_yuan,a.emergency_cost_yuan,a.actual_total_cost_yuan])/1e6; bv=np.array([b.contract_cost_yuan,b.emergency_cost_yuan,b.actual_total_cost_yuan])/1e6
    ba=axes[0].bar(x-width/2,av,width,color=TEAL,label="允许调整"); bb=axes[0].bar(x+width/2,bv,width,color=GREY,label="禁止调整")
    for bars in (ba,bb):
      for bar in bars:
        h=bar.get_height(); axes[0].text(bar.get_x()+bar.get_width()/2,h*.50,f"{h:.3f}",ha="center",va="center",fontsize=6.8,color="white" if h>1 else "#333333")
    axes[0].set_xticks(x,names); axes[0].set_ylabel("费用（百万元）"); axes[0].legend(frameon=False,fontsize=7,loc="upper left"); axes[0].text(.98,.97,f"总费用降低{(b.actual_total_cost_yuan-a.actual_total_cost_yuan)/b.actual_total_cost_yuan*100:.3f}%",transform=axes[0].transAxes,ha="right",va="top",fontsize=7.5,color=RED); title(axes[0],"a","全年费用构成");style(axes[0])
    ba=axes[1].bar([0,1],[a.emergency_purchase_kwh/1e4,b.emergency_purchase_kwh/1e4],color=[TEAL,GREY],width=.55)
    axes[1].set_xticks([0,1],["允许调整","禁止调整"]); axes[1].set_ylabel("紧急购电量（万 kWh）"); axes[1].text(.03,.97,f"紧急购电量降低{(b.emergency_purchase_kwh-a.emergency_purchase_kwh)/b.emergency_purchase_kwh*100:.3f}%",transform=axes[1].transAxes,ha="left",va="top",fontsize=7.5,color=RED);
    for bar in ba: axes[1].annotate(f"{bar.get_height():.3f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords="offset points",ha="center",fontsize=7)
    title(axes[1],"b","全年紧急购电量");style(axes[1])
    monthly = month.pivot(index="month", columns="policy", values="actual_total_cost_yuan").reset_index()
    saving=monthly[B]-monthly[A]
    c=np.cumsum(saving)/1e4; xx=np.arange(len(monthly))
    axes[2].plot(xx,c,color=BLUE,marker="o",lw=1.7,ms=3.6); axes[2].fill_between(xx,0,c,color=BLUE,alpha=.10)
    for i,v in enumerate(c): axes[2].annotate(f"{v:.2f}",(i,v),xytext=(0,3),textcoords="offset points",ha="center",fontsize=6.7)
    axes[2].set_xticks(xx,[str(v).split("-")[1]+"月" for v in monthly.month]); axes[2].set_ylabel("累计费用节省（万元）"); title(axes[2],"c","累计费用节省");style(axes[2])
    save(fig,"q3_v2_annual_adjustment_effect")

def representative():
    slots=pd.read_csv(S4/"q3_v2_slot_results.csv"); plans=pd.read_csv(S4/"q3_v2_plan_versions.csv")
    day="2025-03-20"; d=slots[(slots.policy==A)&(slots.date==day)].sort_values("slot").reset_index(drop=True); p=plans[(plans.policy==A)&(plans.date==day)].copy()
    if len(d)!=144: raise ValueError("representative day incomplete")
    t=d.slot.to_numpy()*DT; fig,ax=plt.subplots(4,1,figsize=(7.25,8.1),sharex=True,constrained_layout=True)
    for hour,color,label in [(0,BLUE,"0:00 原计划"),(6,TEAL,"6:00 调整后计划"),(12,ORANGE,"12:00 调整后计划"),(18,RED,"18:00 调整后计划")]:
      x=p[p.issue_hour==hour].sort_values("slot"); y=np.full(144,np.nan); y[x.slot.to_numpy(int)]=x.version_plan_kwh.to_numpy(float)
      ax[0].step(t,y,where="post",lw=1.35,color=color,label=label)
    ax[0].step(t,d.effective_purchase_kwh,where="post",color="#253247",lw=2.2,label="实际生效计划",zorder=4)
    ax[0].set_ylabel("购电量（kWh）");title(ax[0],"a","原计划、调整后计划与实际生效计划");style(ax[0]);ax[0].legend(ncol=2,fontsize=7,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.02))
    ax[1].bar(t,d.delta_plus_kwh,width=DT*.72,color=TEAL,label="上调量");ax[1].bar(t,-d.delta_minus_kwh,width=DT*.72,color=ORANGE,label="下调量（负值）");ax[1].bar(t,d.emergency_purchase_kwh,width=DT*.42,color=RED,label="紧急购电",zorder=3);ax[1].axhline(0,color=GREY,lw=.8);ax[1].set_ylabel("电量（kWh）");title(ax[1],"b","调整量与紧急购电");style(ax[1]);ax[1].legend(ncol=3,fontsize=7,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.02))
    ax[2].step(t,d.charge_rt_kwh/DT,where="post",color=TEAL,lw=1.4,label="充电功率");ax[2].step(t,-d.discharge_rt_kwh/DT,where="post",color=ORANGE,lw=1.4,label="放电功率");ax[2].set_ylabel("功率（kW）");title(ax[2],"c","实际储能充放电与电价");style(ax[2]); right=ax[2].twinx();right.step(t,d.price_yuan_per_kwh,where="post",color=GREY,lw=1,ls="--",label="分时电价（右轴）");right.set_ylabel("电价（元/kWh）",fontsize=8);ax[2].legend(ncol=2,fontsize=7,frameon=False,loc="upper left");right.legend(fontsize=7,frameon=False,loc="upper right")
    tt=np.r_[t,24];soc=np.r_[d.soc_start_kwh.iloc[0],d.soc_end_kwh];ax[3].plot(tt,soc,color=BLUE,lw=1.9,label="实际 SOC");ax[3].axhline(1200,color=RED,ls="--",lw=1,label="安全下限");ax[3].axhline(10800,color=GREY,ls="-.",lw=1,label="安全上限");ax[3].set_ylabel("储电量（kWh）");ax[3].set_xlabel("时间（h）");title(ax[3],"d","SOC 与安全边界");style(ax[3]);ax[3].legend(ncol=3,fontsize=7,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.02))
    for axis in ax:
      for hour in (6,12,18): axis.axvline(hour,color="#6B7280",lw=.8,ls="--",alpha=.7,zorder=1)
    fig.suptitle("2025年3月20日（指定日期中A策略紧急购电量最大）",fontsize=11,fontweight="semibold",y=1.01)
    ax[3].set_xlim(0,24);ax[3].set_xticks(range(0,25,4));save(fig,"q3_v2_20250320_rolling_dispatch")

def main():
    configure_chinese_theme(); timing_diagram(); annual(); representative()
if __name__=="__main__": main()
