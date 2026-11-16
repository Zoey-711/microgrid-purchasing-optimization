"""Draw a reporting-only annual heatmap from frozen Q4 actual trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

from src.paper_figure_theme import configure_chinese_theme, panel_title, style_axis

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "q4_v1" / "stage4" / "q4_v1_slot_results.csv"
OUT = ROOT / "outputs" / "figures" / "q4_v1"
AUDIT = OUT / "q4_v1_load_pv_price_annual_heatmap.audit.md"


def save(fig: plt.Figure, stem: str) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kwargs in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
        path = OUT / f"{stem}{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def decorate(ax: plt.Axes, dates: pd.DatetimeIndex, label: str, title: str) -> None:
    month_ticks = [i for i, d in enumerate(dates) if d.day == 1]
    ax.set_yticks(month_ticks)
    ax.set_yticklabels([f"{dates[i].month}月" for i in month_ticks])
    ax.set_xticks([0, 36, 72, 108, 143])
    ax.set_xticklabels(["0", "6", "12", "18", "24"])
    ax.set_ylabel("日期（2025年）")
    panel_title(ax, label, title)
    style_axis(ax)


def main() -> None:
    configure_chinese_theme()
    raw = pd.read_csv(SOURCE, low_memory=False)
    # Actual load/PV/price are common to the two branches. Use exactly one
    # branch so every date-slot cell occurs once and no values are averaged.
    data = raw[raw.policy.astype(str).str.contains("4-3", regex=False)].copy()
    data["date"] = pd.to_datetime(data.date)
    if len(data) != 334 * 144 or data.duplicated(["date", "slot"]).any():
        raise ValueError("frozen q4-3 trajectory is not a complete 334×144 date-slot grid")
    data = data.sort_values(["date", "slot"])
    matrices = {
        "load": data.pivot(index="date", columns="slot", values="load_actual_kwh"),
        "pv": data.pivot(index="date", columns="slot", values="pv_actual_kwh"),
        "price": data.pivot(index="date", columns="slot", values="price_yuan_per_kwh"),
    }
    if any(x.shape != (334, 144) or x.isna().any().any() for x in matrices.values()):
        raise ValueError("missing or malformed frozen actual-variable matrix")
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 8.2), sharex=True, gridspec_kw={"hspace": .28})
    # Sequential, low-saturation palettes make within-variable temporal
    # structure legible without claiming a shared scale across three units.
    # Reuse the paper's fixed visual semantics: main quantity blue, PV teal,
    # and price orange-gold. Only the palette changes; matrices are untouched.
    load_cmap = LinearSegmentedColormap.from_list("paper_load", ["#F4F7FB", "#A9C4E0", "#4573B4", "#234A78"])
    pv_cmap = LinearSegmentedColormap.from_list("paper_pv", ["#F3FAF8", "#B9E0D7", "#66B2A3", "#28786F"])
    price_cmap = LinearSegmentedColormap.from_list("paper_price", ["#FFF7E8", "#F5D49A", "#E29D39", "#A85D1D"])
    specs = [
        ("load", load_cmap, "(a)", "全年实际负载时序热力图", "实际负载（kWh/槽）"),
        ("pv", pv_cmap, "(b)", "全年实际光伏时序热力图", "实际光伏（kWh/槽）"),
        ("price", price_cmap, "(c)", "全年实际波动电价时序热力图", "实际电价（元/kWh）"),
    ]
    ranges = {}
    for ax, (key, cmap, label, title, cbar_label) in zip(axes, specs):
        values = matrices[key].to_numpy(float)
        im = ax.imshow(values, aspect="auto", interpolation="nearest", origin="upper", cmap=cmap)
        decorate(ax, matrices[key].index, label, title)
        cb = fig.colorbar(im, ax=ax, pad=.012, shrink=.95)
        cb.set_label(cbar_label)
        ranges[key] = [float(values.min()), float(values.max())]
    axes[-1].set_xlabel("时间（h）")
    fig.suptitle("负载、光伏与波动电价的全年时序热力图", fontsize=13, fontweight="semibold", y=.995)
    fig.text(.5, .006, "横轴为每日144个10分钟槽，纵轴为2025年2月1日至12月31日；三幅图分别采用自身变量量纲与色标。",
             ha="center", fontsize=7.5, color="#4B5563")
    paths = save(fig, "q4_v1_load_pv_price_annual_heatmap")
    audit = {"source": str(SOURCE.relative_to(ROOT)), "branch_used_for_unique_actuals": "问题4-3", "matrix_shape": [334, 144],
             "date_range": [str(matrices['load'].index.min().date()), str(matrices['load'].index.max().date())], "ranges": ranges,
             "outputs": paths, "data_modified": False}
    AUDIT.write_text("# 负载、光伏与波动电价全年时序热力图：数据审计\n\n```json\n" + json.dumps(audit, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print("Q4_ANNUAL_LOAD_PV_PRICE_HEATMAP_PASS")


if __name__ == "__main__":
    main()
