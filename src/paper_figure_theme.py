"""Shared Chinese publication theme for the second-generation paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ACTUAL_BLUE = "#4573B4"
PLAN_BLUE = "#6090C1"
FORECAST_BLUE = "#6C9FB8"
PV_GREEN = "#66B2A3"
PRICE_ORANGE = "#E29D39"
EMERGENCY_RED = "#C44E52"
CHARGE_GREEN = "#66B2A3"
DISCHARGE_ORANGE = "#E29D39"
BOUND_GREY = "#6B7280"
GRID_GREY = "#E8E8E8"
TEXT_GREY = "#30343B"
BORDER_DARK = "#333333"

FONT_CANDIDATES = (
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
)


def configure_chinese_theme() -> tuple[str, Path]:
    """Select one installed CJK font, verify key glyphs, and configure Matplotlib."""
    installed: dict[str, Path] = {}
    for item in font_manager.fontManager.ttflist:
        installed.setdefault(item.name, Path(item.fname))
    selected = next((name for name in FONT_CANDIDATES if name in installed), None)
    if selected is None:
        raise RuntimeError(
            "No approved Chinese font found. Expected Source Han Sans, Noto Sans CJK SC, "
            "Microsoft YaHei, or SimHei."
        )

    font_path = installed[selected]
    charmap = font_manager.get_font(str(font_path)).get_charmap()
    test_text = "问题储能购电量预测光伏安全边界万元充放电−上下标"
    missing = sorted({char for char in test_text if ord(char) not in charmap})
    if missing:
        raise RuntimeError(f"Selected font {selected} lacks required glyphs: {missing}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected],
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "axes.unicode_minus": True,
            "axes.edgecolor": BORDER_DARK,
            "axes.labelcolor": TEXT_GREY,
            "xtick.color": TEXT_GREY,
            "ytick.color": TEXT_GREY,
            "text.color": TEXT_GREY,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return selected, font_path


def style_axis(axis, *, time_axis: bool = False) -> None:
    """Apply the common white, minimal academic axis style."""
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_GREY, linestyle="-", linewidth=0.65)
    axis.grid(axis="x", visible=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER_DARK)
        spine.set_linewidth(0.75)
    axis.tick_params(
        axis="both", which="both", direction="out", width=0.75, length=3.2,
        top=False, right=False, labeltop=False, labelright=False,
        bottom=True, left=True,
    )
    if time_axis:
        axis.set_xlim(0, 24)
        axis.set_xticks(np.arange(0, 25, 4))


def panel_title(axis, label: str, title: str, *, y: float = 1.025) -> None:
    axis.text(
        0.0,
        y,
        f"{label} {title}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="semibold",
        clip_on=False,
    )


def compact_legend(axis, **kwargs):
    defaults = {
        "frameon": False,
        "facecolor": "white",
        "edgecolor": "none",
        "framealpha": 0.90,
        "handlelength": 2.2,
        "handletextpad": 0.5,
        "columnspacing": 1.0,
        "borderaxespad": 0.2,
    }
    defaults.update(kwargs)
    return axis.legend(**defaults)
