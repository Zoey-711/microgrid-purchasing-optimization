# 问题二 V2 策略冻结记录

## 冻结状态

全年影子评价审查已通过。以下记录冻结参数、策略定义、结果文件哈希与受保护文件哈希；未导出或修改任何官方 `result2.xlsx`。

```json
{
  "status": "frozen_shadow_evaluation",
  "stage4_evaluation_complete": true,
  "frozen_config_origin_note": "下列配置文件由阶段3在全年评价前冻结；其中 formal_evaluation_has_run 字段反映其生成时点，不表示当前阶段4未完成。",
  "frozen_parameters": {
    "status": "frozen_after_january_development_only",
    "formal_evaluation_has_run": false,
    "load_forecast": "mean of same weekday/same slot in prior 35 days; fallback prior up-to-7 days",
    "pv_forecast": "same-slot mean of prior up-to-7 complete days",
    "joint_residual_scenario_cap": 14,
    "terminal_value_yuan_per_kwh": 0.4,
    "dp_soc_grid_kwh": 50.0,
    "actual_soc_is_continuous": true,
    "dp_primary_tie_tolerance_yuan": 1e-08,
    "tie_rule": "within primary tolerance minimize charge+discharge throughput; deterministic residual ordering",
    "solver": "HiGHS MILP",
    "solver_time_limit_seconds": 45.0,
    "solver_relative_gap": 0.0001,
    "solver_random_seed": 0,
    "solver_threads": 1,
    "cold_start": "Jan-01 zero plan and greedy actual execution",
    "initial_soc_2025_01_01_kwh": 6000.0,
    "cross_day_soc": true,
    "terminal_value_in_actual_bill": false,
    "result2_generation_authorized": false
  },
  "formal_period": [
    "2025-02-01",
    "2025-12-31"
  ],
  "main_strategy": "主策略：联合情景 MILP + DP50 执行器",
  "closed_loop_control": "完整闭环对照：联合情景 MILP + 贪心执行器",
  "paired_ablation_definition": "固定主策略当日计划和日初SOC，仅替换实时执行器；不是独立跨日策略。",
  "audit_passed": true,
  "result2_xlsx_exported": false,
  "protected_hashes_match": true,
  "protected_file_hashes": {
    "data\\checks\\q2_template_check.txt": "6A94F34473C73A5810BDD6B618768D31A6B82AF0D79CF58FFDDEA638152775FF",
    "data\\processed\\q1_baseline_dispatch.csv": "89B0328F967E152D22AF9479793E428C61C41D3D7BD0CBD557C556DA156E965A",
    "data\\processed\\q1_day_data.csv": "3D84C5744A71C1F1FF976EDBE8CE230C789B19959DB3A9913ABD50D1826775C9",
    "data\\raw\\C题.pdf": "2C098F6AE9DD47AE965AEBDEC3B9FACF3DE6C173F999783C1012C08FC5FB9D2D",
    "data\\raw\\result1.xlsx": "28360E0974E7D6065394A8ABA4E14A86773AE0036CC7D3EA1B211B515B03D688",
    "data\\raw\\result2.xlsx": "1C26494CFC6D754E0BD9BFF7E13E1126A73D2D2DA6C5336EB251D89B9A1A1A47",
    "data\\raw\\附件1.xlsx": "66B87134F5ECCCD68184D3539BB1293EF039F9E0FDD955A589B9BFA7F227C377",
    "data\\raw\\附件2.xlsx": "2E95FD446BFAFA0D8C59577B5C2E2EA8B3F1DEF20DDE54A3062556F4DA9B4C72",
    "docs\\q1_freeze_record.md": "BFE74FFE6DDD3ABBBD4128FDD8FBED6531A21136D4AE7016F9F4AE4C6C53DCA4",
    "docs\\q2_completion_checklist.md": "37BF759CDED13866B87961FE6747078A3DDD26BC49E87FEEFA150A380E3439A5",
    "docs\\q2_data_contract.md": "B9ED23D1A3B6F9E9802C27ED5930810C4327699A70B09CB43A616094785430A6",
    "docs\\q2_freeze_record.md": "E12C3064A7E93DA44843B6B495E7FC96C97ECF5D92DC825055CC47652B052352",
    "docs\\q2_model_design.md": "8AED2D5100BD404DAE3A9BB364153358F223B5EA0F92064EBADE9E269612CB15",
    "docs\\q2_parameter_selection.md": "8A76A8BCC72E950EDFDC7C8B43B1B16B425787AEB816FC14F9436B629CD14094",
    "docs\\q2_reserve_audit.md": "58DCD56C8AB0977AEA77A4648D680A7789348E8BEDE76B3D02AAA1578B1488D5",
    "docs\\q2_writing_material.md": "F5345A9C8C5A173240B3F797E4366FD049FDFFFB069BA81CF93E342B56B1AB05",
    "outputs\\logs\\q2_final_check.txt": "93535E90AB3AFBDF2A5F5BA5D45CB6E281C103521A04826FE7EC2C5D041DF95A",
    "outputs\\logs\\q2_full_year_shadow_validation.txt": "DAFF89A0FBFF81162C2604516DC861F8952B6E1A45355F6957297B60BD0E8FA8",
    "outputs\\logs\\q2_submission_check.txt": "C12AE21B580029A3AC48AA60E40E396CD62DE7E967E592FCAB749769865AE335",
    "outputs\\replay\\q2_aplus\\frozen_hashes_after.json": "74EE16AA3A2A1CF83831ABA304ECB5B401F126B91705EB85118C59C912EEBC44",
    "outputs\\replay\\q2_aplus\\frozen_hashes_before.json": "74EE16AA3A2A1CF83831ABA304ECB5B401F126B91705EB85118C59C912EEBC44",
    "outputs\\replay\\q2_aplus\\q2_aplus_slot_trajectory.csv": "BB3C8409A7C22ECBF199016C9A75506DFC7A301AF5653BEAE882F60FA9183ED7",
    "outputs\\replay\\q2_aplus\\replay_consistency_check.json": "41BCDCCA0CE942E8789B8A73E729F2CE9D4C0980A921B1687638807E390D4B33",
    "outputs\\tables\\q2_daily_shadow_results.csv": "7F7649421C9572DF102BC345B96125B96EEF6945042564B579CB9AE8F53BDFE8",
    "outputs\\tables\\q2_fixed_prediction_reserve_comparison.csv": "53FF964EF8D991ACE3D64DB9F5CBAB7C82564F97813CEAAFE23C79ADC736509F",
    "outputs\\tables\\q2_fixed_prediction_reserve_comparison.md": "25EBD82864D40346C935B8AB362EF542C56BF82723AA0E2C48BF5189525EB898",
    "outputs\\tables\\q2_full_year_forecast_metrics.csv": "C2E11536E7133D9FA8064B7F56F12ABD9FFD954A5448AB6E6DFBC2ECE85985E7",
    "outputs\\tables\\q2_full_year_shadow_comparison.csv": "7C62F8AFB8AE679E715C8469633C7905D4A15B84B4182AC4B7A5D0214906C07A",
    "outputs\\tables\\q2_january_parameter_comparison.csv": "22FF3F192B0278966D3CCF63893982423D853ECC6122E9679E3C3FD81418FBFC",
    "outputs\\tables\\q2_january_pareto_summary.csv": "B02897CAD5E93CB9994BD1D1DD495B60A8CCC0F8D0CB0CEE15622854865D4393",
    "outputs\\tables\\q2_paper_tables.csv": "1680FB7E84B45CB0D268F5C13B8DDFC8F0C151080D58898AB85C6F4DBFB6A34C",
    "outputs\\tables\\result1.xlsx": "E594F3CD449C5634941EC3A647C7E46C40A9F881451CE8C97AD197AA62693585",
    "outputs\\tables\\result2.xlsx": "58CEA3400BE56E4E3FFABAD02434AC1F6B888165C7F058517EABB317846AFCEE",
    "src\\analyze_q2_january.py": "9CB9BD35A1D57BF39ADF7B9F316E8FABB196CCA135527B56D63C4D02C80B9D38",
    "src\\backtest_q2_january.py": "25824D89CDF83BAEC33F48574F4432334BC3D81698CDC6D6DA2BFE4BE33B1521",
    "src\\diagnose_q3_vs_q2.py": "5042E1A1261FF0FB8B28E927E9279E48357849BA43A0CA3D0F62A0E14D3E8F1A",
    "src\\dispatch_q2_realtime.py": "66CF0F402D7378EDB7C88F784FB2D8631BB56CE6ED577FF4B678CC648D89697F",
    "src\\export_q2_submission.py": "B22030D8B58AB8B9480B034BEBF4519136E25465E27FDAF03576595588C96195",
    "src\\forecast_q2.py": "4A897EA9464DC9CF2F1CE0ED0D25F1A43161FBC41C207EDAF75DB6E99A39C45E",
    "src\\generate_q1_q2_figures_v2.py": "9E62D10CE32BB946BD05C8DD5E0D2CF729183034AEFE373F6C21ECCE710A06A5",
    "src\\model_q2_day_ahead.py": "C4DB1B25829B7BF1C9146A542A8C0DE9DF83383ABABE1C880E9138DDAE5D27E7",
    "src\\replay_q2_aplus_freeze.py": "D02286B0BE1A7C27DDC684278F55452E38E3FC082A0755428C0A10CCB00E459B",
    "src\\shadow_q2_full_year.py": "0A17FF197BA899259ED6CA7D839CD144794B261C8C3E2EA85A00247A7921F192",
    "src\\solve_q2.py": "35507E0C06B06A72A02E2A5446B9712CC2F3E5144403786BD4C3119908C94811",
    "src\\validation_q2.py": "146423636CC06570FFD9D33420C99D855D0C21D3113F11F724F5A31B35103845"
  },
  "result_file_hashes": {
    "outputs\\q2_v2\\stage4\\q2_v2_full_year_strategy_comparison.csv": "9915E16F68F99B98ACC7DB0057FC8FD78F1FD348B0538F390422EFFD05949F25",
    "outputs\\q2_v2\\stage4\\q2_v2_final_comparison.csv": "FABA171562D42B695C4F06C0BF0D666EBEF4DFB76FEBBA1380CDFAB41ED572D0",
    "outputs\\q2_v2\\stage4\\q2_v2_monthly_comparison.csv": "FADAA5468C562BD83F2467309755CA64BF455528555F2499B44FADFAB7575060",
    "outputs\\q2_v2\\stage4\\q2_v2_daily_results.csv": "518245BE91B55AA5C1560B6C45F94A8B21DEDBAA8865F87A43FA5B29E6A85939",
    "outputs\\q2_v2\\stage4\\q2_v2_slot_results.csv": "C6A806F09A57191735E8FCC6D197CD967EEADD22D513C034E67F1051873ED5C4",
    "outputs\\q2_v2\\stage4\\q2_v2_day_ahead_plans.csv": "ED5C57EDE77E17B5E4C7932B89B1C6D384659764B0B147062987760F22B6CC88",
    "outputs\\q2_v2\\stage4\\q2_v2_solver_log.csv": "C9E6FD635556C3B2EC4F7841C217BE8710F4CF87266BDD07F330C1D518729D7E",
    "outputs\\q2_v2\\stage4\\q2_v2_information_boundary.csv": "4D9E1DC1C1632015A5769E04EE8942ADC12E2CC9AEF34F9500A52ED663E4C9AE",
    "outputs\\q2_v2\\stage4\\q2_v2_paired_executor_ablation.csv": "E181F7443BA95256BEED4EB9410BD5B7478C5D50394F178C0FFE0FA8761AE91A",
    "outputs\\q2_v2\\stage4\\q2_v2_ablation_summary.csv": "D560FFCF75AE2AF3C0D687BF8522FACB3B45581FAA5628E4935A04A359B3084A",
    "outputs\\q2_v2\\stage4\\q2_v2_full_year_validation.json": "EA04B385EF427E57BBD530FD2220CE05D2996BB6D82DD5CF6DB445C498377830",
    "outputs\\q2_v2\\stage4\\q2_v2_final_results_audit.json": "A02FE276BB1AEAD1A28B23DAC32C169E21A1BE1FCA4816B4CDE7A1A74D62AF10",
    "outputs\\q2_v2\\stage4\\q2_v2_daily_comparison.csv": "F26C16A3DE0EFFD92DE325151475F9F9924EA3778A0F5B952C57C17BD810C0A4",
    "outputs\\q2_v2\\stage4\\q2_v2_emergency_by_price_band.csv": "6D448603E4B867D780DBE79093E9C5B87BED234E37E53E59B394231085B9F71B"
  },
  "freeze_timestamp": "2026-09-11T18:27:52",
  "immutable_statement": "后续不得修改冻结参数或全年影子结果；只有获得明确授权后才可进入官方result2.xlsx模板导出阶段。"
}
```

## 不可变声明

问题二 V2 的 M=14、终端价值 0.4 元/kWh、DP 网格 50 kWh、MILP 配置、预测规则、并列动作规则、评价日期、主策略与对照定义均已冻结。除全文排版或获得明确授权后进行官方模板导出外，不得修改冻结参数或全年影子结果。

## 正式模板导出

已按授权以‘历史联合残差情景—动态价值调度策略’导出 `outputs\tables\result2.xlsx`。旧文件已字节归档至 `outputs\q2_v2\archive\result2_legacy_q2.xlsx`，归档前后 SHA-256 均为 `58CEA3400BE56E4E3FFABAD02434AC1F6B888165C7F058517EABB317846AFCEE`；新提交文件 SHA-256 为 `3D21511D2D411F92E293F79524F80EEF482BA7D884B403C8C29B75DBE23C3870`。详见 `docs/q2_v2/q2_v2_export_audit.md`。
