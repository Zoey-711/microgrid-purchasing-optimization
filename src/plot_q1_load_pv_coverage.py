"""Paper figure only: aggregates frozen Q1 typical-day inputs; no optimization."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/q1_baseline_dispatch.csv"
OUT = ROOT / "outputs/figures/paper_v2"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT = OUT / "q1_load_pv_coverage.audit.md"
df = pd.read_csv(SRC)
required = {"time_label", "load_kwh", "pv_forecast_kwh"}
missing = required - set(df.columns)
if len(df) != 144 or missing:
    raise ValueError(f"Expected 144 frozen Q1 slots and {required}; missing={missing}, n={len(df)}")

# The official Q1 paper tables use six 4-hour periods; aggregate only, never resample.
period = np.arange(len(df)) // 24
labels = ["00:00–04:00", "04:00–08:00", "08:00–12:00", "12:00–16:00", "16:00–20:00", "20:00–24:00"]
agg = (df.assign(period=period).groupby("period", sort=True)
       .agg(load_kwh=("load_kwh", "sum"), pv_kwh=("pv_forecast_kwh", "sum")))
agg["coverage_pct"] = np.where(agg["load_kwh"] > 0, 100 * agg["pv_kwh"] / agg["load_kwh"], 0)
load_mwh = agg["load_kwh"] / 1000
pv_mwh = agg["pv_kwh"] / 1000

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10})
blue, teal, red, grid = "#4573B4", "#66B2A3", "#A66161", "#D9DEE7"
fig, ax = plt.subplots(figsize=(9.4, 4.8))
x = np.arange(6); width = .32
b1 = ax.bar(x-width/2, load_mwh, width, color=blue, label="小区负载", zorder=3)
b2 = ax.bar(x+width/2, pv_mwh, width, color=teal, label="光伏预测电量", zorder=3)
ax.set_ylabel("电量（MWh）"); ax.set_xticks(x, labels); ax.set_xlabel("时间区段")
ax.grid(axis="y", color=grid, linewidth=.8, zorder=0); ax.spines["top"].set_visible(False)
ax2 = ax.twinx()
line = ax2.plot(x, agg["coverage_pct"], color=red, marker="o", linewidth=1.8, markersize=4.5, label="光伏覆盖比例", zorder=4)
ax2.set_ylabel("覆盖比例（%）"); ax2.spines["top"].set_visible(False)
for rect, value in zip(b1, load_mwh):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8, color="#23344D")
for rect, value in zip(b2, pv_mwh):
    if value > 0:
        if value >= 10:
            ax.text(rect.get_x()+rect.get_width()/2, value * .56, f"{value:.2f}",
                    ha="center", va="center", fontsize=8, color="white")
        else:
            ax.text(rect.get_x()+rect.get_width()/2, value, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#2E756A")
for xi, value in zip(x, agg["coverage_pct"]):
    ax2.annotate(f"{value:.1f}%", (xi, value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color=red)
handles = [b1, b2, line[0]]
ax.legend(handles, [h.get_label() for h in handles], loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=3, frameon=False)
fig.suptitle("典型日负载、光伏预测电量及覆盖比例", y=.99, fontweight="semibold")
fig.subplots_adjust(top=.82, bottom=.16, left=.08, right=.92)
base = OUT / "q1_load_pv_coverage"
for ext, kwargs in [("png", {"dpi": 300}), ("pdf", {}), ("svg", {})]:
    fig.savefig(base.with_suffix("." + ext), bbox_inches="tight", facecolor="white", **kwargs)
plt.close(fig)
audit = {
    "source": str(SRC.relative_to(ROOT)).replace("\\", "/"),
    "source_slots": int(len(df)), "aggregation": "six consecutive 24-slot (4-hour) sums; no interpolation or smoothing",
    "load_total_kwh": float(agg["load_kwh"].sum()), "pv_total_kwh": float(agg["pv_kwh"].sum()),
    "coverage_total_pct": float(100 * agg["pv_kwh"].sum() / agg["load_kwh"].sum()),
    "six_period_rows": agg.reset_index().round(9).to_dict(orient="records"),
    "check": "PASS: six load/PV sums equal frozen 144-slot totals"}
AUDIT.write_text("# 问题一负载—光伏覆盖图数据审计\n\n" + json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False))
