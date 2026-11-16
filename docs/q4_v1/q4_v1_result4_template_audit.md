# 问题四 V1：result4 模板结构审计

本审查直接读取 `data/raw/result4-2.xlsx` 和 `data/raw/result4-3.xlsx`，未依据 result2/result3 推测结构。两个原始模板均未改写。

## result4-2.xlsx

SHA-256：`1C26494CFC6D754E0BD9BFF7E13E1126A73D2D2DA6C5336EB251D89B9A1A1A47`。工作表顺序：`计划购电量`、`充放电量`、`紧急购电量`。

| 工作表 | 最大行×列 | 合并单元格 | 公式 | 数据验证 | 冻结窗格 | 隐藏行/列 |
|---|---:|---|---:|---:|---|---|
| 计划购电量 | 335×147 | 0 | 0 | 0 | None | 0/0 |

格式样本：A1=General, B1=General, B2=General, EP1=General, EP2=General, EQ1=General, EQ2=General。行高、列宽的完整原始记录保存在 `outputs/q4_v1/stage5/q4_v1_template_audit.json`。
| 充放电量 | 20×6 | 0 | 0 | 0 | None | 0/0 |

格式样本：A1=General, B1=General, B2=General, EP2=General, EQ2=General。行高、列宽的完整原始记录保存在 `outputs/q4_v1/stage5/q4_v1_template_audit.json`。
| 紧急购电量 | 11×3 | 0 | 0 | 0 | None | 0/0 |

格式样本：A1=General, B1=General, B2=General, EP2=General, EQ2=General。行高、列宽的完整原始记录保存在 `outputs/q4_v1/stage5/q4_v1_template_audit.json`。

计划矩阵为第2—335行、B:EO 144列，日期为2025-02-01至2025-12-31。模板表头存在循环式官方列标签（首列为0:10—0:20、末列为次日0:00—0:10）；映射严格按 B:EO 的官方左至右顺序，不对时间索引重排。EP、EQ为模板既有日总量/日费用列。

## result4-3.xlsx

SHA-256：`C59DA470CABD0BE23F602C95C8AA9D11EC224A0CDAC216B3E1F218E65D006BDC`。工作表顺序：`计划购电量`、`调整购电量`、`充放电量`、`紧急购电量`。

| 工作表 | 最大行×列 | 合并单元格 | 公式 | 数据验证 | 冻结窗格 | 隐藏行/列 |
|---|---:|---|---:|---:|---|---|
| 计划购电量 | 335×147 | 0 | 0 | 0 | None | 0/0 |

格式样本：A1=General, B1=General, B2=General, EP1=General, EP2=General, EQ1=General, EQ2=General。行高、列宽的完整原始记录保存在 `outputs/q4_v1/stage5/q4_v1_template_audit.json`。
| 调整购电量 | 335×147 | 0 | 0 | 0 | None | 0/0 |

格式样本：A1=General, B1=General, B2=General, EP1=General, EP2=General, EQ1=General, EQ2=General。行高、列宽的完整原始记录保存在 `outputs/q4_v1/stage5/q4_v1_template_audit.json`。
| 充放电量 | 26×6 | 0 | 0 | 0 | None | 0/0 |

格式样本：A1=General, B1=General, B2=General, EP2=General, EQ2=General。行高、列宽的完整原始记录保存在 `outputs/q4_v1/stage5/q4_v1_template_audit.json`。
| 紧急购电量 | 11×3 | 0 | 0 | 0 | None | 0/0 |

格式样本：A1=General, B1=General, B2=General, EP2=General, EQ2=General。行高、列宽的完整原始记录保存在 `outputs/q4_v1/stage5/q4_v1_template_audit.json`。

计划矩阵为第2—335行、B:EO 144列，日期为2025-02-01至2025-12-31。模板表头存在循环式官方列标签（首列为0:10—0:20、末列为次日0:00—0:10）；映射严格按 B:EO 的官方左至右顺序，不对时间索引重排。EP、EQ为模板既有日总量/日费用列。

`充放电量`和`紧急购电量`原表只预留示例/指定日期行与省略标记，不能同时容纳334天六时段量及全年全部紧急事件。因此演练副本按官方既有字段扩展数据体行；不新增工作表、不改表头、不改列序。这是正式导出前需确认的模板呈现解释。

演练回读：4-2/4-3均通过，文件分别为`outputs\q4_v1\stage5\dry_run\result4-2_dry_run.xlsx`、`outputs\q4_v1\stage5\dry_run\result4-3_dry_run.xlsx`；保护哈希变化数为 0。
