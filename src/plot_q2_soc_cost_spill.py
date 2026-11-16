"""Q2 paper figure from frozen stage-4 trajectories; no optimisation or data edits."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ROOT / "outputs/q2_v2/stage4/q2_v2_slot_results.csv"
MONTHLY = ROOT / "outputs/q2_v2/stage4/q2_v2_monthly_comparison.csv"
OUT = ROOT / "outputs/figures/q2_v2"
OUT.mkdir(parents=True, exist_ok=True)

slots = pd.read_csv(SLOTS)
monthly = pd.read_csv(MONTHLY)
slots = slots.loc[slots["policy_id"].eq("main_dp50")].copy()
if slots.shape[0] != 334 * 144 or slots.groupby(["date", "slot"]).size().ne(1).any():
    raise ValueError("Frozen main strategy must contain exactly 334×144 unique slots.")
if len(monthly) != 11 or not monthly["policy_id_dp50"].eq("main_dp50").all():
    raise ValueError("Monthly frozen data mismatch.")

# Actual SOC at starts of 0/4/8/12/16/20 h and at end of day (24 h).
nodes = [0, 24, 48, 72, 96, 120]
node_data = slots.loc[slots["slot"].isin(nodes), ["date", "slot", "soc_start_kwh"]].copy()
node_data["hour"] = node_data["slot"] / 6
end = (slots.loc[slots["slot"].eq(143), ["date", "soc_end_kwh"]]
       .rename(columns={"soc_end_kwh": "soc_start_kwh"}))
end["hour"] = 24
node_data = pd.concat([node_data[["date", "hour", "soc_start_kwh"]], end], ignore_index=True)
q = (node_data.groupby("hour")["soc_start_kwh"]
     .quantile([.10, .25, .50, .75, .90]).unstack())
q.columns = ["q10", "q25", "q50", "q75", "q90"]
assert list(q.index) == [0, 4, 8, 12, 16, 20, 24]

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10})
blue, teal, red, gray, band1, band2, grid = "#4573B4", "#66B2A3", "#C44E52", "#6B7280", "#DCE7F2", "#AFC7DF", "#D9DEE7"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"wspace": .30})
h = q.index.to_numpy(dtype=float)
ax1.fill_between(h, q.q10.to_numpy(), q.q90.to_numpy(), color=band1, alpha=.75, label="10%–90%区间", zorder=1)
ax1.fill_between(h, q.q25.to_numpy(), q.q75.to_numpy(), color=band2, alpha=.72, label="25%–75%区间", zorder=2)
ax1.plot(h, q.q50, color=blue, linewidth=2.2, marker="o", markersize=3.8, label="SOC中位轨迹", zorder=4)
ax1.axhline(1200, color=gray, linestyle="--", linewidth=.8)
ax1.axhline(10800, color=gray, linestyle="--", linewidth=.8)
ax1.text(24.1, 1200, "安全下限", va="center", fontsize=8, color=gray)
ax1.text(24.1, 10800, "安全上限", va="center", fontsize=8, color=gray)
ax1.set(xlim=(0, 24.9), ylim=(500, 11500), xticks=h, xlabel="时刻（h）", ylabel="SOC（kWh）")
ax1.grid(axis="y", color=grid, linewidth=.8); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.set_title("(a) 主策略全年SOC包络", loc="left", fontweight="semibold")
ax1.legend(loc="lower right", bbox_to_anchor=(.99, .985), ncol=2, frameon=False, fontsize=8)

months = monthly["month"].str[-2:].astype(int).to_numpy()
plan = monthly["plan_purchase_cost_yuan_dp50"].to_numpy() / 1e4
emg = monthly["emergency_purchase_cost_yuan_dp50"].to_numpy() / 1e4
spill = monthly["spill_kwh_dp50"].to_numpy() / 1e4
x = np.arange(len(months))
ax2.bar(x, plan, .62, color=blue, label="计划购电费", zorder=3)
ax2.bar(x, emg, .62, bottom=plan, color=red, label="紧急购电费", zorder=3)
ax2.set(xticks=x, xticklabels=[f"{m:02d}月" for m in months], ylabel="费用（万元）", xlabel="月份")
ax2.grid(axis="y", color=grid, linewidth=.8, zorder=0); ax2.spines["top"].set_visible(False)
ax2r = ax2.twinx()
ax2r.plot(x, spill, color=teal, linewidth=1.8, marker="o", markersize=3.5, label="弃能量", zorder=4)
ax2r.set_ylabel("弃能量（万kWh）", color="#397D72"); ax2r.tick_params(axis="y", colors="#397D72"); ax2r.spines["top"].set_visible(False)
for xi, total in zip(x, plan + emg):
    ax2.text(xi, total, f"{total:.1f}", ha="center", va="bottom", fontsize=7.5, color="#24364E")
handles = ax2.get_legend_handles_labels()[0] + ax2r.get_legend_handles_labels()[0]
labels = ax2.get_legend_handles_labels()[1] + ax2r.get_legend_handles_labels()[1]
ax2.legend(handles, labels, loc="upper left", ncol=3, frameon=False, fontsize=8)
ax2.set_title("(b) 月度费用构成与弃能量", loc="left", fontweight="semibold")
fig.suptitle("问题二储能SOC全年包络与月度费用、弃能量结构", y=.995, fontweight="semibold")
fig.subplots_adjust(top=.82, bottom=.17, left=.07, right=.94)
base = OUT / "q2_v2_soc_envelope_monthly_cost_spill"
for ext, kwargs in [("png", {"dpi": 300}), ("pdf", {}), ("svg", {})]:
    fig.savefig(base.with_suffix("." + ext), bbox_inches="tight", facecolor="white", **kwargs)
plt.close(fig)

audit = {
 "slot_source": str(SLOTS.relative_to(ROOT)).replace("\\", "/"),
 "monthly_source": str(MONTHLY.relative_to(ROOT)).replace("\\", "/"),
 "policy": "main_dp50", "unique_slots": int(len(slots)), "days": int(slots.date.nunique()),
 "soc_nodes": [int(i) for i in h], "soc_quantiles_kwh": q.round(9).reset_index().to_dict(orient="records"),
 "monthly_plan_cost_yuan_sum": float(monthly.plan_purchase_cost_yuan_dp50.sum()),
 "monthly_emergency_cost_yuan_sum": float(monthly.emergency_purchase_cost_yuan_dp50.sum()),
 "monthly_spill_kwh_sum": float(monthly.spill_kwh_dp50.sum()),
 "check": "PASS: 334×144 slots; SOC nodes drawn from actual trajectories; monthly totals use frozen monthly table."
}
(OUT / "q2_v2_soc_envelope_monthly_cost_spill.audit.md").write_text(
 "# 问题二 SOC 包络与月度费用图：数据审计\n\n" + json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False))
