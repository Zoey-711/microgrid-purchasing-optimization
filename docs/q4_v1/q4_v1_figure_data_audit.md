# 问题四图表数据审计

```json
{
  "figure_4_1": {
    "representative_date": "2025-07-02",
    "selection_rule": "全年高价紧急购电费用最大日",
    "high_price_emergency_fee_yuan": 29654.369429666316,
    "slots": 144,
    "scenario_tracks": 14,
    "scenario_source_dates": [
      "2025-06-18",
      "2025-06-19",
      "2025-06-20",
      "2025-06-21",
      "2025-06-22",
      "2025-06-23",
      "2025-06-24",
      "2025-06-25",
      "2025-06-26",
      "2025-06-27",
      "2025-06-28",
      "2025-06-29",
      "2025-06-30",
      "2025-07-01"
    ],
    "price_forecast_min_max": [
      0.32852857142857145,
      1.3738500000000002
    ],
    "price_actual_min_max": [
      0.4184,
      1.727
    ],
    "soc_start_end_kwh": [
      8208.53329728898,
      7849.312039567901
    ],
    "outputs": [
      "outputs\\figures\\q4_v1\\q4_v1_representative_price_storage.png",
      "outputs\\figures\\q4_v1\\q4_v1_representative_price_storage.pdf",
      "outputs\\figures\\q4_v1\\q4_v1_representative_price_storage.svg"
    ]
  },
  "figure_4_2": {
    "saving_yuan": 388002.54583629593,
    "saving_pct": 2.668482620268821,
    "emergency_drop_kwh": 121679.89601963121,
    "emergency_drop_pct": 67.17301484249506,
    "cumulative_final_10k_yuan": 38.80025458362983,
    "terminal_soc_q42_q43": [
      10279.388139406394,
      10225.682061740748
    ],
    "outputs": [
      "outputs\\figures\\q4_v1\\q4_v1_annual_comparison.png",
      "outputs\\figures\\q4_v1\\q4_v1_annual_comparison.pdf",
      "outputs\\figures\\q4_v1\\q4_v1_annual_comparison.svg"
    ]
  },
  "consistency": {
    "hashes": {
      "result1.xlsx": "E594F3CD449C5634941EC3A647C7E46C40A9F881451CE8C97AD197AA62693585",
      "result2.xlsx": "3D21511D2D411F92E293F79524F80EEF482BA7D884B403C8C29B75DBE23C3870",
      "result3.xlsx": "A1DFFD4EB38D80431DD711C60D4BD405A2E00A0936C12264DD1029AB45BD33DA",
      "result4-2.xlsx": "C20384AC037B984296AF084963E125A5B581729AD32EFA465FD9B0C7C7CE166D",
      "result4-3.xlsx": "339C42BB93442261A914B9BCF5B7D00B3B9860D62A1087BF560A04F937A264EE",
      "raw_result4-2.xlsx": "1C26494CFC6D754E0BD9BFF7E13E1126A73D2D2DA6C5336EB251D89B9A1A1A47",
      "raw_result4-3.xlsx": "C59DA470CABD0BE23F602C95C8AA9D11EC224A0CDAC216B3E1F218E65D006BDC"
    },
    "pandoc": "passed",
    "max_mip_gap": 0.0007825911757777
  },
  "source_files": [
    "outputs\\q4_v1\\stage4\\q4_v1_slot_results.csv",
    "outputs\\q4_v1\\stage4\\q4_v1_monthly_comparison.csv",
    "outputs\\q4_v1\\stage4\\q4_v1_full_year_comparison.csv",
    "outputs\\q4_v1\\stage4\\q4_v1_solver_quality.csv"
  ],
  "data_modified": false
}
```
