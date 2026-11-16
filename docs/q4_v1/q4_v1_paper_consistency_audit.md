# 问题四 V1 论文一致性审计

## 范围

本审查只读取冻结的阶段4/5 CSV、审计文件与正式工作簿；未运行优化，未写入任何Excel。

## 图表与正文

- 图4-1：`outputs/figures/q4_v1/q4_v1_representative_price_storage.*`；代表日 2025-07-02，选择规则为全年高价紧急购电费用最大日。
- 图4-2：`outputs/figures/q4_v1/q4_v1_annual_comparison.*`；累计末值 38.800255 万元。
- 正文已新增“问题4”章节，使用图4-1、图4-2和表4-1；未改变问题1—3章节内容。

## 数值核验

| 项目 | 复核值 | 状态 |
|---|---:|---|
| 4-2实际总费用 | 14,540,193.850元 | 通过 |
| 4-3实际总费用 | 14,152,191.304元 | 通过 |
| 费用差 | 388,002.546元（2.668%） | 通过 |
| 紧急购电差 | 121,679.896kWh（67.17%） | 通过 |
| 4-2 / 4-3期末SOC | 10,279.388 / 10,225.682 kWh | 通过 |
| 最大相对MIP gap | 0.00078259 | 通过；限时可行解未称为最优 |

## Markdown解析

Pandoc：passed。

## 受保护文件SHA-256

- `result1.xlsx`: `E594F3CD449C5634941EC3A647C7E46C40A9F881451CE8C97AD197AA62693585`
- `result2.xlsx`: `3D21511D2D411F92E293F79524F80EEF482BA7D884B403C8C29B75DBE23C3870`
- `result3.xlsx`: `A1DFFD4EB38D80431DD711C60D4BD405A2E00A0936C12264DD1029AB45BD33DA`
- `result4-2.xlsx`: `C20384AC037B984296AF084963E125A5B581729AD32EFA465FD9B0C7C7CE166D`
- `result4-3.xlsx`: `339C42BB93442261A914B9BCF5B7D00B3B9860D62A1087BF560A04F937A264EE`
- `raw_result4-2.xlsx`: `1C26494CFC6D754E0BD9BFF7E13E1126A73D2D2DA6C5336EB251D89B9A1A1A47`
- `raw_result4-3.xlsx`: `C59DA470CABD0BE23F602C95C8AA9D11EC224A0CDAC216B3E1F218E65D006BDC`

结论：图表数值和正文冻结数值一致；未修改任何Excel或冻结配置。
