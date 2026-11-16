# locked_same_info vs reusable formal Q4-3

正式 Q4-3 已按 inheritance_equivalence_audit 只读登记为 rolling 分支，未复制或修改正式文件。

{
  "locked_days": 365,
  "locked_formal_slots": 52560,
  "formal_reused_slots": 48096,
  "daily_cost_sum_matches": "True",
  "rolling_reused": true,
  "rolling_reuse_evidence": "inheritance_equivalence_audit reusable=true, max_field_error=1.8189894035458565e-12",
  "locked_validation": {
    "days": 334,
    "expected_days": 334,
    "slots": 48096,
    "expected_slots": 48096,
    "duplicate_slots": 0,
    "max_balance_residual_kwh": 4.547473508864641e-13,
    "max_soc_transition_residual_kwh": 9.094947017729282e-13,
    "max_cross_day_soc_error_kwh": 0.0,
    "soc_min_kwh": 1200.0,
    "soc_max_kwh": 10800.0,
    "power_violation_count": 0,
    "simultaneous_charge_discharge_count": 0,
    "emergency_and_charge_count": 0,
    "negative_effective_plan_count": 0,
    "prefix_lock_max_error_kwh": 0.0
  },
  "locked_cost_check": {
    "max_slot_error_yuan": 2.0446741461022788e-11,
    "max_daily_error_yuan": 2.1827872842550278e-11,
    "total_error_yuan": 0.0
  },
  "rolling_formal_cost_check": {
    "max_slot_error_yuan": 7.592873733121623e-12,
    "max_daily_error_yuan": 1.4551915228366852e-11,
    "total_error_yuan": 0.0
  }
}
