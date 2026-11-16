# Formal Q4-3 versus same_info rolling inheritance audit

reusable = true

两分支均从 2025-01-01 00:00 SOC=6000 kWh 出发，经相同预热过程后在正式评价期起点 SOC=10800 kWh。

## Checks
{
  "row_count_equal": true,
  "date_slot_order_equal": true,
  "initial_soc_formal_kwh": 10800.0,
  "initial_soc_replay_kwh": 10800.0,
  "raw_initial_soc_2025_01_01_00_kwh": 6000.0,
  "same_january_warmup": true,
  "formal_period_start_soc_is_10800_kwh": true,
  "max_field_error": 1.8189894035458565e-12,
  "all_fields_within_tolerance": true,
  "reusable": true
}

## Field comparison
                 field  max_abs_error  mean_abs_error  within_tolerance
     original_plan_kwh   2.273737e-13    9.186068e-15              True
    effective_plan_kwh   2.273737e-13    7.368591e-15              True
            charge_kwh   5.684342e-14    1.994932e-16              True
         discharge_kwh   5.684342e-14    1.033333e-15              True
emergency_purchase_kwh   0.000000e+00    0.000000e+00              True
         soc_start_kwh   1.818989e-12    1.068446e-13              True
           soc_end_kwh   1.818989e-12    1.068446e-13              True
    contract_cost_yuan   1.136868e-13    5.131698e-15              True
actual_total_cost_yuan   1.136868e-13    5.131698e-15              True
