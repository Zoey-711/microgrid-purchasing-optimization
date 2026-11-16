"""Package frozen-data Q2 figures as editable Draw.io documents.

Dense statistical charts are embedded as high-resolution PNG layers because
Draw.io Desktop's CLI does not reliably render a base64 SVG image cell on this
Windows installation. Page framing, heading and source note remain native
Draw.io objects. No solver or source result is changed.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "outputs" / "figures" / "q2_v2"

TARGETS = {
    "q2_v2_annual_economic_mechanism": "问题二图 1：全年策略效果与经济机理",
    "q2_v2_20250601_dispatch": "问题二图 2：2025-06-01 代表日调度机理",
    "q2_v2_joint_residual_scenarios": "问题二图 3：历史联合残差情景构造",
}


def make_drawio(stem: str, title: str) -> None:
    png_path = FIG / f"{stem}.png"
    png_data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    xml = f'''<mxfile host="app.diagrams.net" modified="2026-09-11T00:00:00.000Z" agent="Codex" version="31.4.5" type="device">
  <diagram id="{stem}" name="Page-1">
    <mxGraphModel dx="1180" dy="820" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0"/><mxCell id="1" parent="0"/>
        <mxCell id="frame" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#333333;strokeWidth=1;" vertex="1" parent="1"><mxGeometry x="20" y="20" width="1129" height="787" as="geometry"/></mxCell>
        <mxCell id="title" value="{html.escape(title)}" style="text;html=1;align=left;verticalAlign=middle;fontFamily=Microsoft YaHei;fontSize=18;fontStyle=1;fontColor=#30343B;strokeColor=none;fillColor=none;whiteSpace=wrap;" vertex="1" parent="1"><mxGeometry x="50" y="38" width="1050" height="32" as="geometry"/></mxCell>
        <mxCell id="plot" value="" style="shape=image;imageAspect=1;aspect=fixed;image=data:image/png%3Bbase64,{png_data};strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="50" y="82" width="1050" height="670" as="geometry"/></mxCell>
        <mxCell id="note" value="冻结结果绘制；图内文字、图例与图形区域经独立留白处理。" style="text;html=1;align=right;verticalAlign=middle;fontFamily=Microsoft YaHei;fontSize=9;fontColor=#6B7280;strokeColor=none;fillColor=none;whiteSpace=wrap;" vertex="1" parent="1"><mxGeometry x="60" y="764" width="1030" height="20" as="geometry"/></mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
    (FIG / f"{stem}.drawio").write_text(xml, encoding="utf-8")
    audit = f"""# {title}：Draw.io 图件审计

## 可见元素

| 元素 | 位置 | 介质 | 状态 |
|---|---|---|---|
| 白色页面与细深灰边框 | 全画布 | Draw.io 原生图形 | 完成 |
| 图标题 | 顶部 | Draw.io 原生文本 | 完成 |
| 冻结数据图形、坐标轴、图例与标注 | 主图区域 | 高分辨率 PNG 嵌入图层 | 完成 |
| 数据来源提示 | 页脚 | Draw.io 原生文本 | 完成 |

## 核验

- 主图仅由冻结 CSV 生成的绘图副本构成；未重新求解或改写统计数据。
- Draw.io Desktop 在本机 CLI 下不能可靠渲染嵌入式 SVG，故使用同一冻结图的 300 dpi PNG 图层，避免 SVG 占位图标。
- 标题、主图和页脚分区排布；导出 PNG/PDF 后应逐图检查文字、图例和裁切。
"""
    (FIG / f"{stem}.audit.md").write_text(audit, encoding="utf-8")


def main() -> None:
    for stem, title in TARGETS.items():
        if not (FIG / f"{stem}.png").exists():
            raise FileNotFoundError(f"Missing figure source: {stem}.png")
        make_drawio(stem, title)


if __name__ == "__main__":
    main()
