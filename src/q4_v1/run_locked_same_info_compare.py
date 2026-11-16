"""Resume locked same_info branch and compare it with reusable formal Q4-3."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.q4_v1 import run_stage2 as data_source
from src.q4_v1 import run_stage4 as frozen
from src.q4_v1.run_same_info_ablation import OUT, sha256, frozen_config, run_policy, paired_information_check, validate_trajectory, cost_check, period_summary

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "outputs/q4_v1/stage4/q4_v1_slot_results.csv"

def main() -> None:
    cfg = frozen_config(); cfg_hash = sha256(ROOT / "outputs/q4_v1/stage3/q4_v1_frozen_config_4_3.json")
    dates, labels, load, pv, price = data_source.read_data(); cache = frozen.cache43(dates, labels, load, pv, price)
    locked = run_policy("locked", False, dates, labels, load, pv, price, cache, cfg, cfg_hash)
    lf, ls, lp, la = locked
    formal = pd.read_csv(FORMAL); formal = formal[formal.policy == "问题4-3"].copy()
    formal = formal.sort_values(["date", "slot"]).reset_index(drop=True)
    lf = lf.sort_values(["date", "slot"]).reset_index(drop=True)
    # Formal Q4-3 is the reusable rolling branch; register it without copying or modifying the source file.
    formal["policy"] = "rolling_formal_reused"
    formal["allow_intraday_plan_adjustment"] = True
    formal_sum = period_summary("rolling_formal_reused", formal, pd.Timestamp("2025-02-01"), pd.Timestamp("2025-12-31"))
    locked_sum = period_summary("locked_same_info", lf, pd.Timestamp("2025-02-01"), pd.Timestamp("2025-12-31"))
    daily_locked = lf.groupby("date", as_index=False).agg(policy=("policy","first"), actual_total_cost_yuan=("actual_total_cost_yuan","sum"), contract_cost_yuan=("contract_cost_yuan","sum"), emergency_cost_yuan=("emergency_cost_yuan","sum"), emergency_purchase_kwh=("emergency_purchase_kwh","sum"), soc_start_kwh=("soc_start_kwh","first"), soc_end_kwh=("soc_end_kwh","last"))
    daily_formal = formal.groupby("date", as_index=False).agg(actual_total_cost_yuan=("actual_total_cost_yuan","sum"), contract_cost_yuan=("contract_cost_yuan","sum"), emergency_cost_yuan=("emergency_cost_yuan","sum"), emergency_purchase_kwh=("emergency_purchase_kwh","sum"), soc_start_kwh=("soc_start_kwh","first"), soc_end_kwh=("soc_end_kwh","last"))
    daily = daily_locked.merge(daily_formal, on="date", suffixes=("_locked", "_rolling")); daily["cost_difference_locked_minus_rolling_yuan"] = daily.actual_total_cost_yuan_locked - daily.actual_total_cost_yuan_rolling; daily["emergency_difference_locked_minus_rolling_kwh"] = daily.emergency_purchase_kwh_locked - daily.emergency_purchase_kwh_rolling
    fields = ["original_plan_kwh","effective_purchase_kwh","charge_rt_kwh","discharge_rt_kwh","emergency_purchase_kwh","soc_start_kwh","soc_end_kwh","contract_cost_yuan","actual_total_cost_yuan"]
    # Compare locked against the formal rolling trajectory as an audit reference; unequal plans are expected for the ablation.
    checks = {"locked_days": int(lf.date.nunique()), "locked_formal_slots": int(len(lf)), "formal_reused_slots": int(len(formal)), "daily_cost_sum_matches": abs(daily.cost_difference_locked_minus_rolling_yuan.sum() - (locked_sum["actual_total_cost_yuan"] - formal_sum["actual_total_cost_yuan"])) < 1e-6, "rolling_reused": True, "rolling_reuse_evidence": "inheritance_equivalence_audit reusable=true, max_field_error=1.8189894035458565e-12", "locked_validation": validate_trajectory(lf, lp, True), "locked_cost_check": cost_check(lf[lf.date >= "2025-02-01"]), "rolling_formal_cost_check": cost_check(formal[formal.date >= "2025-02-01"])}
    comparison = pd.DataFrame([{**locked_sum, "policy":"locked_same_info"}, {**formal_sum, "policy":"rolling_formal_reused"}])
    comparison["inventory_corrected_from_6000_yuan"] = comparison.actual_total_cost_yuan + 6000 - comparison.end_soc_kwh
    comparison["inventory_corrected_interval_yuan"] = comparison.actual_total_cost_yuan + comparison.start_soc_kwh - comparison.end_soc_kwh
    comparison.to_csv(OUT / "locked_vs_reused_rolling_comparison.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "locked_vs_reused_rolling_daily_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([checks]).to_json(OUT / "locked_vs_reused_rolling_validation.json", orient="records", force_ascii=False, indent=2)
    (OUT / "locked_vs_reused_rolling_validation.md").write_text("# locked_same_info vs reusable formal Q4-3\n\n正式 Q4-3 已按 inheritance_equivalence_audit 只读登记为 rolling 分支，未复制或修改正式文件。\n\n" + json.dumps(checks, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
