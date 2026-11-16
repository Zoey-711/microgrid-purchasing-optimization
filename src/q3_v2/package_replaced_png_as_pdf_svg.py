"""Package user-supplied replacement PNGs as matching PDF/SVG raster containers."""
from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "outputs/figures/q3_v2"
STEMS = ("q3_v2_annual_adjustment_effect", "q3_v2_20250320_rolling_dispatch")


def package(stem: str) -> None:
    png = FIG / f"{stem}.png"
    with Image.open(png) as image:
        image.convert("RGB").save(png.with_suffix(".pdf"), "PDF", resolution=300.0)
        width, height = image.size
    payload = base64.b64encode(png.read_bytes()).decode("ascii")
    png.with_suffix(".svg").write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        f"<image width=\"{width}\" height=\"{height}\" href=\"data:image/png;base64,{payload}\"/>"
        "</svg>\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    for item in STEMS:
        package(item)
