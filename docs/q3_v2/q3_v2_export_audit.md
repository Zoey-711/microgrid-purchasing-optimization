# 问题三 V2 正式模板导出审计

```json
{
  "export_time": "2026-09-12T04:23:06",
  "strategy_name": "多时点预报更新—滚动购电调整与动态价值储能调度策略",
  "source_policy": "A_all_updates_dynamic_value",
  "source_slots": "outputs\\q3_v2\\stage4\\q3_v2_slot_results.csv",
  "period": [
    "2025-02-01",
    "2025-12-31"
  ],
  "days": 334,
  "slots": 48096,
  "raw_template_sha256_before": "C59DA470CABD0BE23F602C95C8AA9D11EC224A0CDAC216B3E1F218E65D006BDC",
  "raw_template_sha256_after": "C59DA470CABD0BE23F602C95C8AA9D11EC224A0CDAC216B3E1F218E65D006BDC",
  "result3_sha256": "A1DFFD4EB38D80431DD711C60D4BD405A2E00A0936C12264DD1029AB45BD33DA",
  "protected_file_count": 72,
  "protected_hashes_unchanged": true,
  "comparison": {
    "A": {
      "contract_cost_yuan": 13152259.341800263,
      "emergency_cost_yuan": 116516.8973672058,
      "actual_total_cost_yuan": 13268776.239167469,
      "emergency_purchase_kwh": 29540.082258968,
      "terminal_soc_kwh": 1462.5760325657
    },
    "B": {
      "contract_cost_yuan": 13068148.860439423,
      "emergency_cost_yuan": 829567.1312689825,
      "actual_total_cost_yuan": 13897715.991708407,
      "emergency_purchase_kwh": 180018.1597610229,
      "terminal_soc_kwh": 1399.5883261834
    },
    "A_minus_B_contract_yuan": 84110.48136083968,
    "A_minus_B_emergency_yuan": -713050.2339017767,
    "B_minus_A_total_yuan": 628939.7525409386,
    "B_minus_A_total_percent": 4.525490036752613,
    "B_minus_A_emergency_kwh": 150478.0775020549,
    "B_minus_A_emergency_percent": 83.5904987040291,
    "A_minus_B_terminal_soc_kwh": 62.98770638229985,
    "A_slot_aggregation_max_error": 1.862645149230957e-09
  },
  "writeback_checks": {
    "plan_matrix_cells": 48096,
    "adjustment_matrix_cells": 48096,
    "plan_cell_max_error_kwh": 0.0,
    "adjustment_cell_max_error_kwh": 0.0,
    "plan_daily_total_max_error_kwh": 1.4551915228366852e-11,
    "adjustment_daily_total_max_error_kwh": 7.275957614183426e-12,
    "plan_daily_fee_max_error_yuan": 7.275957614183426e-12,
    "contract_daily_fee_max_error_yuan": 7.275957614183426e-12,
    "storage_aggregate_max_error_kwh": 1.8189894035458565e-12,
    "exported_soc_max_error_kwh": 0.0,
    "emergency_event_max_error_kwh": 2.2737367544323206e-13,
    "emergency_event_sum_error_kwh": 1.0913936421275139e-11,
    "contract_recompute_max_abs_error_yuan": 1.772377800079994e-10,
    "emergency_recompute_max_abs_error_yuan": 2.899014361901209e-10,
    "total_recompute_max_abs_error_yuan": 3.6970959627069533e-10,
    "energy_balance_max_abs_residual_kwh": 1.9986146071460098e-10,
    "soc_transition_max_abs_residual_kwh": 1.4551915228366852e-10,
    "cross_day_soc_max_abs_error_kwh": 0.0,
    "soc_violation_count": 0,
    "power_violation_count": 0,
    "simultaneous_charge_discharge_count": 0,
    "emergency_and_charge_count": 0,
    "negative_plan_or_effective_count": 0,
    "delta_plus_relation_max_error_kwh": 1.0015810403274372e-10,
    "delta_minus_relation_max_error_kwh": 1.0018652574217413e-10,
    "event_count": 228,
    "formula_count_preserved": true
  },
  "adjustment_sheet_interpretation": "按建模假设填写执行槽之前最后一个有效版本的最终生效购电量 z；不是各次调整量之和。",
  "charge_discharge_convention": "充放电量表 C/D 分别为实际充电/放电的非负 kWh 幅值。",
  "emergency_event_convention": "仅记录正紧急购电；同日相邻正槽合并，跨日不合并。",
  "optimization_rerun": false
}
```

## 正式文件写后复核

```json
{
  "postwrite_audit_time": "2026-09-12T04:24:40",
  "result3_sha256": "A1DFFD4EB38D80431DD711C60D4BD405A2E00A0936C12264DD1029AB45BD33DA",
  "raw_template_sha256": "C59DA470CABD0BE23F602C95C8AA9D11EC224A0CDAC216B3E1F218E65D006BDC",
  "candidate_byte_identical_to_result3": true,
  "protected_manifest_unchanged": true,
  "protected_file_count": 72,
  "matrix_and_physical_checks": {
    "plan_matrix_cells": 48096,
    "adjustment_matrix_cells": 48096,
    "plan_cell_max_error_kwh": 0.0,
    "adjustment_cell_max_error_kwh": 0.0,
    "plan_daily_total_max_error_kwh": 1.4551915228366852e-11,
    "adjustment_daily_total_max_error_kwh": 7.275957614183426e-12,
    "plan_daily_fee_max_error_yuan": 7.275957614183426e-12,
    "contract_daily_fee_max_error_yuan": 7.275957614183426e-12,
    "storage_aggregate_max_error_kwh": 1.8189894035458565e-12,
    "exported_soc_max_error_kwh": 0.0,
    "emergency_event_max_error_kwh": 2.2737367544323206e-13,
    "emergency_event_sum_error_kwh": 1.0913936421275139e-11,
    "contract_recompute_max_abs_error_yuan": 1.772377800079994e-10,
    "emergency_recompute_max_abs_error_yuan": 2.899014361901209e-10,
    "total_recompute_max_abs_error_yuan": 3.6970959627069533e-10,
    "energy_balance_max_abs_residual_kwh": 1.9986146071460098e-10,
    "soc_transition_max_abs_residual_kwh": 1.4551915228366852e-10,
    "cross_day_soc_max_abs_error_kwh": 0.0,
    "soc_violation_count": 0,
    "power_violation_count": 0,
    "simultaneous_charge_discharge_count": 0,
    "emergency_and_charge_count": 0,
    "negative_plan_or_effective_count": 0,
    "delta_plus_relation_max_error_kwh": 1.0015810403274372e-10,
    "delta_minus_relation_max_error_kwh": 1.0018652574217413e-10,
    "event_count": 228,
    "formula_count_preserved": true
  },
  "version_lock_and_time_mapping": {
    "active_plan_version_cell_count": 48096,
    "active_effective_plan_max_error_kwh": 0.0,
    "executed_prefix_lock_max_error_kwh": 0.0,
    "official_slot_time_mapping_errors": 0
  },
  "passed": true
}
```
