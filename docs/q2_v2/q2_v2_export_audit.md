# 问题二 V2 正式模板导出审计

```json
{
  "export_time": "2026-09-11T18:39:09",
  "strategy_display_name": "历史联合残差情景—动态价值调度策略",
  "implementation_identifier": "main_dp50 (SOC grid = 50 kWh)",
  "source": "outputs\\q2_v2\\stage4\\q2_v2_slot_results.csv",
  "period": [
    "2025-02-01",
    "2025-12-31"
  ],
  "days": 334,
  "slots": 48096,
  "archive": {
    "archive_time": "2026-09-11T18:39:07",
    "legacy_original_path": "outputs\\tables\\result2.xlsx",
    "archive_path": "outputs\\q2_v2\\archive\\result2_legacy_q2.xlsx",
    "legacy_original_sha256": "58CEA3400BE56E4E3FFABAD02434AC1F6B888165C7F058517EABB317846AFCEE",
    "archive_sha256": "58CEA3400BE56E4E3FFABAD02434AC1F6B888165C7F058517EABB317846AFCEE",
    "statement": "旧问题二结果已被 V2 替代为正式提交版本；原字节副本已归档。"
  },
  "raw_template_sha256_before": "1C26494CFC6D754E0BD9BFF7E13E1126A73D2D2DA6C5336EB251D89B9A1A1A47",
  "raw_template_sha256_after": "1C26494CFC6D754E0BD9BFF7E13E1126A73D2D2DA6C5336EB251D89B9A1A1A47",
  "exported_result2_sha256": "3D21511D2D411F92E293F79524F80EEF482BA7D884B403C8C29B75DBE23C3870",
  "candidate_sha256": "3D21511D2D411F92E293F79524F80EEF482BA7D884B403C8C29B75DBE23C3870",
  "charge_discharge_convention": "模板 C、D 列分别记录实际充电量、实际放电量的非负 kWh 幅值；不采用正负号混写。",
  "emergency_event_convention": "只记录实际发生的正紧急购电事件；相邻正槽合并，数量为合并槽的实际 kWh 之和。",
  "checks": {
    "plan_cost_yuan": 12847359.13083032,
    "emergency_cost_yuan": 916569.0772929109,
    "actual_total_cost_yuan": 13763928.208123231,
    "emergency_purchase_kwh": 176902.13245527353,
    "plan_cell_max_error_kwh": 4.547473508864641e-13,
    "daily_plan_total_max_error_kwh": 7.275957614183426e-12,
    "daily_plan_cost_max_error_yuan": 7.275957614183426e-12,
    "storage_aggregate_max_error_kwh": 5.4569682106375694e-12,
    "exported_soc_max_error_kwh": 9.094947017729282e-13,
    "emergency_event_max_error_kwh": 1.8189894035458565e-12,
    "emergency_event_sum_error_kwh": 2.9103830456733704e-11,
    "energy_balance_max_residual_kwh": 6.821210263296962e-13,
    "soc_transition_max_residual_kwh": 3.637978807091713e-12,
    "cross_day_soc_max_residual_kwh": 0.0,
    "soc_boundary_violations": 0,
    "power_violations": 0,
    "simultaneous_charge_discharge_slots": 0,
    "emergency_while_charging_slots": 0,
    "formula_count_preserved": true,
    "official_structure_preserved": true,
    "plan_empty_cells": 0,
    "event_count": 589
  },
  "result2_exported": true,
  "protected_scope_note": "旧 outputs/tables/result2.xlsx 已先按授权归档；data/raw/result2.xlsx 未修改。"
}
```

## 复核说明

计划购电量表按官方 B:EO 的 144 列顺序逐槽写入；EP、EQ 分别写入日计划购电量和日计划购电费。充放电量表按六个四小时块扩展至 334 天，C、D 列分别为实际充电、放电的非负 kWh 幅值。紧急购电量表仅记录正事件，并合并相邻连续槽。

费用复算以冻结逐槽记录的价格、计划购电量和紧急购电量独立进行；导出工作簿反读后逐单元格验证计划、四小时充放电汇总、首末 SOC 和紧急购电事件。因合并事件跨槽价格可能不同，紧急费用不能仅由事件总量反推，故以逐槽冻结源进行费用复算并以事件总量核对。
