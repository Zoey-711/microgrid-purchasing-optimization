"""Chinese publication figures for the existing Question-1 dispatch result."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.config import (
    DAY_HOURS,
    DELTA_T,
    ETA_C,
    ETA_D,
    FIGURE_DIR,
    LOG_DIR,
    Q1_BASELINE_INTERFACE_FILE,
    S_FINAL,
    S_INIT,
    S_MAX,
    S_MIN,
)
from src.plot_style import (
    AXIS_GREY, FILL_ALPHA, GRID_GREY, PLAN_PURCHASE, PRICE, SOC_LOWER_BOUND,
    SOC_MAIN, SOC_UPPER_BOUND,
)


VISUALIZATION_LOG_FILE = LOG_DIR / "q1_visualization_log.txt"
CN_FONT_CANDIDATES = ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS")


def _configure_chinese_font() -> str:
    """Select an installed Chinese-capable font, with a generic fallback."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in CN_FONT_CANDIDATES if name in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [selected, *CN_FONT_CANDIDATES, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return selected


def _paper_axes(axis) -> None:
    """Apply a restrained white-background paper style."""
    axis.set_facecolor("white")
    axis.grid(axis="y", color=GRID_GREY, linestyle="-", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(AXIS_GREY)
    axis.spines["bottom"].set_color(AXIS_GREY)
    axis.spines["left"].set_linewidth(0.85)
    axis.spines["bottom"].set_linewidth(0.85)
    axis.tick_params(axis="both", direction="out", width=0.8, length=4, color=AXIS_GREY)
    axis.set_xlim(0, DAY_HOURS)
    axis.set_xticks(np.arange(0, DAY_HOURS + 0.1, 4))


def _load_existing_q1_dispatch(data_path: Path) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Read, but never modify, the 144-slot baseline interface."""
    data = pd.read_csv(data_path)
    required = {
        "time_label",
        "price",
        "purchase_kwh",
        "charge_kwh",
        "discharge_kwh",
        "soc_kwh",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Baseline dispatch lacks required columns: {sorted(missing)}")
    if len(data) != 144:
        raise ValueError(f"Baseline dispatch must contain 144 rows, found {len(data)}")

    values = data[["price", "purchase_kwh", "charge_kwh", "discharge_kwh", "soc_kwh"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Baseline dispatch contains non-finite plotting values")

    # Official row order is retained exactly: row i remains dispatch slot i.
    # The terminal SOC is reconstructed only from the final stored slot transition.
    terminal_soc = (
        float(data["soc_kwh"].iloc[-1])
        + ETA_C * float(data["charge_kwh"].iloc[-1])
        - float(data["discharge_kwh"].iloc[-1]) / ETA_D
    )
    soc_series = np.append(data["soc_kwh"].to_numpy(dtype=float), terminal_soc)
    return data, soc_series, terminal_soc


def generate_q1_chinese_figures(
    data_path: Path = Q1_BASELINE_INTERFACE_FILE,
) -> tuple[list[Path], dict]:
    """Create the two requested Chinese figures from the existing CSV only."""
    data, soc_series, terminal_soc = _load_existing_q1_dispatch(data_path)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    font_name = _configure_chinese_font()

    # Dispatch values retain their official row order, without parsing or shifting labels.
    slot_hours = (np.arange(len(data)) + 1) * DELTA_T
    state_hours = np.arange(len(soc_series)) * DELTA_T
    purchase_color = PLAN_PURCHASE
    soc_color = SOC_MAIN

    figure, (price_axis, purchase_axis) = plt.subplots(
        2, 1, sharex=True, figsize=(8.1, 5.6), gridspec_kw={"height_ratios": [1, 1.12]}
    )
    figure.patch.set_facecolor("white")
    price_axis.step(slot_hours, data["price"], where="post", color=PRICE, linewidth=2.2)
    price_axis.set_ylabel("电价（元/kWh）")
    _paper_axes(price_axis)
    price_axis.text(0.01, 0.98, "(a)", transform=price_axis.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6})
    price_axis.tick_params(labelbottom=False)

    purchase_axis.step(
        slot_hours,
        data["purchase_kwh"],
        where="post",
        color=purchase_color,
        linewidth=2.25,
    )
    purchase_axis.fill_between(
        slot_hours,
        data["purchase_kwh"],
        0,
        step="post",
        color=purchase_color,
        alpha=0.07,
    )
    purchase_axis.set_xlabel("时间（h）")
    purchase_axis.set_ylabel("计划购电量（kWh）")
    _paper_axes(purchase_axis)
    purchase_axis.text(0.01, 0.98, "(b)", transform=purchase_axis.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6})
    figure.subplots_adjust(hspace=0.14)
    purchase_path = FIGURE_DIR / "q1_purchase_price_cn.png"
    figure.savefig(purchase_path, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(purchase_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.1, 5.0))
    figure.patch.set_facecolor("white")
    axis.step(state_hours, soc_series, where="post", color=soc_color, linewidth=2.2, label="储电量")
    axis.fill_between(state_hours, soc_series, S_MIN, color=soc_color, alpha=FILL_ALPHA)
    axis.axhline(S_MIN, color=SOC_LOWER_BOUND, linestyle="--", linewidth=1.35, label=f"安全下限（{S_MIN:.0f} kWh）")
    axis.axhline(S_MAX, color=SOC_UPPER_BOUND, linestyle="--", linewidth=1.35, label=f"安全上限（{S_MAX:.0f} kWh）")
    axis.scatter([0, DAY_HOURS], [soc_series[0], soc_series[-1]], color=soc_color, s=28, zorder=4)
    axis.annotate(
        "初始：6000 kWh",
        xy=(0, soc_series[0]),
        xytext=(18, 20),
        textcoords="offset points",
        fontsize=8.5,
        color=soc_color,
        arrowprops={"arrowstyle": "->", "color": soc_color, "lw": 0.9},
    )
    axis.annotate(
        "末端：6000 kWh",
        xy=(DAY_HOURS, soc_series[-1]),
        xytext=(-10, 22),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=soc_color,
        arrowprops={"arrowstyle": "->", "color": soc_color, "lw": 0.9},
    )
    axis.set_xlabel("时间（h）")
    axis.set_ylabel("储电量（kWh）")
    _paper_axes(axis)
    axis.set_ylim(800, 11250)
    axis.set_yticks([800, 2000, 4000, 6000, 8000, 10000, 11250])
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.7,
    )
    soc_path = FIGURE_DIR / "q1_storage_soc_cn.png"
    figure.savefig(soc_path, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(soc_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    checks = {
        "source_data": str(data_path),
        "font": font_name,
        "row_count": len(data),
        "price_unit": "元/kWh",
        "purchase_unit": "kWh",
        "soc_unit": "kWh",
        "soc_min_kwh": float(soc_series.min()),
        "soc_max_kwh": float(soc_series.max()),
        "soc_start_kwh": float(soc_series[0]),
        "soc_end_kwh": float(terminal_soc),
        "soc_bounds_ok": bool(soc_series.min() >= S_MIN and soc_series.max() <= S_MAX),
        "soc_endpoints_ok": bool(np.isclose(soc_series[0], S_INIT) and np.isclose(terminal_soc, S_FINAL)),
        "official_row_order_preserved": True,
        "generated_files": [str(purchase_path), str(soc_path)],
    }
    log_lines = ["Question 1 Chinese-figure generation", "=" * 36]
    log_lines.extend(f"{key}: {value}" for key, value in checks.items())
    VISUALIZATION_LOG_FILE.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return [purchase_path, soc_path], checks


def plot_q1_results(_solution: pd.DataFrame | None = None) -> list[Path]:
    """Compatibility wrapper: plot only from the existing baseline CSV."""
    paths, _ = generate_q1_chinese_figures()
    return paths


if __name__ == "__main__":
    paths, checks = generate_q1_chinese_figures()
    print("\n".join(str(path) for path in paths))
    print(f"SOC checks passed: {checks['soc_bounds_ok'] and checks['soc_endpoints_ok']}")
