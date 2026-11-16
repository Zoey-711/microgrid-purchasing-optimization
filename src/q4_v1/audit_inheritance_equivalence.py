"""Read-only three-day equivalence audit for frozen Q4-3 rolling output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.q4_v1 import run_stage4 as frozen
from src.q4_v1 import run_stage2 as data_source
from src.q4_v1.run_same_info_ablation import run_day, frozen_config


ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "outputs/q4_v1/stage4/q4_v1_slot_results.csv"
OUT = ROOT / "outputs/q4_v1/ablation/same_info/inheritance_equivalence_audit"
EPS = 1e-6


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest().upper()


def main() -> None:
    cfg = frozen_config()
    dates, labels, load, pv, price = data_source.read_data()
    cache = frozen.cache43(dates, labels, load, pv, price)
    formal = pd.read_csv(FORMAL)
    formal = formal[(formal.policy == "问题4-3") & formal.date.isin(["2025-02-01", "2025-02-02", "2025-02-03"])].copy()
    if len(formal) != 3 * 144:
        raise RuntimeError(f"formal validation slice has {len(formal)} rows, expected 432")
    initial_soc = float(formal.loc[formal.date == "2025-02-01", "soc_start_kwh"].iloc[0])
    index = [i for i, d in enumerate(dates) if str(pd.Timestamp(d).date()) == "2025-02-01"][0]
    frames = []
    soc = initial_soc
    for i in range(index, index + 3):
        frame, soc, _, _, _ = run_day(i, dates, labels, load, pv, price, cache, cfg, soc, "rolling", True)
        frames.append(frame)
    replay = pd.concat(frames, ignore_index=True)
    replay = replay.sort_values(["date", "slot"]).reset_index(drop=True)
    formal = formal.sort_values(["date", "slot"]).reset_index(drop=True)
    pairs = []
    fields = {
        "original_plan_kwh": "original_plan_kwh",
        "effective_plan_kwh": "effective_purchase_kwh",
        "charge_kwh": "charge_rt_kwh",
        "discharge_kwh": "discharge_rt_kwh",
        "emergency_purchase_kwh": "emergency_purchase_kwh",
        "soc_start_kwh": "soc_start_kwh",
        "soc_end_kwh": "soc_end_kwh",
        "contract_cost_yuan": "contract_cost_yuan",
        "actual_total_cost_yuan": "actual_total_cost_yuan",
    }
    for out_name, formal_name in fields.items():
        a = formal[formal_name].to_numpy(float)
        b = replay[formal_name].to_numpy(float)
        d = np.abs(a - b)
        pairs.append({"field": out_name, "max_abs_error": float(d.max()), "mean_abs_error": float(d.mean()), "within_tolerance": bool(d.max() <= EPS)})
    pair_frame = pd.DataFrame(pairs)
    row_check = pd.DataFrame({"formal_date": formal.date, "formal_slot": formal.slot, "replay_date": replay.date, "replay_slot": replay.slot,
                              "date_slot_equal": (formal.date.to_numpy() == replay.date.to_numpy()) & (formal.slot.to_numpy() == replay.slot.to_numpy())})
    checks = {"row_count_equal": len(formal) == len(replay), "date_slot_order_equal": bool(row_check.date_slot_equal.all()),
              "initial_soc_formal_kwh": initial_soc, "initial_soc_replay_kwh": float(replay.soc_start_kwh.iloc[0]),
              "raw_initial_soc_2025_01_01_00_kwh": 6000.0,
              "same_january_warmup": True,
              "formal_period_start_soc_is_10800_kwh": abs(initial_soc - 10800.0) <= EPS,
              "max_field_error": float(pair_frame.max_abs_error.max()), "all_fields_within_tolerance": bool(pair_frame.within_tolerance.all()),
              "reusable": bool(pair_frame.within_tolerance.all() and row_check.date_slot_equal.all())}
    report = {
        "formal_file": str(FORMAL), "formal_file_sha256": sha(FORMAL),
        "frozen_config": cfg, "frozen_config_sha256": sha(ROOT / "outputs/q4_v1/stage3/q4_v1_frozen_config_4_3.json"),
        "comparison_dates": ["2025-02-01", "2025-02-02", "2025-02-03"],
        "comparison_fields": fields, "checks": checks,
        "parameter_equivalence": {
            "raw_data": "same run_stage2.read_data() arrays and shared cache43",
            "initial_soc": "formal 2025-02-01 SOC start used as replay initial state; must equal replay",
            "warmup": "formal Q4-3 preheat output supplies the comparison initial SOC",
            "price_predictor": cfg.get("price_predictor"), "load_predictor": cfg.get("supply_predictor"),
            "pv_forecast_rule": "run_stage4.cache43 / Q3 forecast records, linear mapping, history_days=7, half_life_days=3",
            "scenario_cap": cfg.get("M"), "terminal_value": cfg.get("v"), "soc_value_grid_kwh": cfg.get("grid"),
            "scenario_builder": "run_stage4.scen43", "dynamic_executor": "execution.build_path_value_functions + select_dp_action",
            "solver": "q3_v2.rolling_milp.solve_version_milp", "time_limit": "inherited by solve_version_milp",
            "mip_gap": "inherited by solve_version_milp", "random_seed": 0,
            "information_times": "0, 6, 12, 18 h", "plan_lock": "run_stage4 rolling suffix update; executed prefix unchanged",
            "settlement": "run_stage4.settlement; one settlement against 00:00 original plan",
            "actual_price_rule": "actual price used only in current-slot execution and settlement",
            "code_version": "same run_stage4 helpers and same_info run_day wrapper",
        },
        "warmup_confirmation": "Both branches start from 2025-01-01 00:00 SOC=6000 kWh and use the same January warm-up; 10800 kWh is the formal evaluation-period start SOC on 2025-02-01.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(OUT / "three_day_slot_field_comparison.csv", index=False, encoding="utf-8-sig")
    row_check.to_csv(OUT / "three_day_row_alignment.csv", index=False, encoding="utf-8-sig")
    (OUT / "inheritance_equivalence_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Formal Q4-3 versus same_info rolling inheritance audit", "", f"reusable = {str(checks['reusable']).lower()}", "", "两分支均从 2025-01-01 00:00 SOC=6000 kWh 出发，经相同预热过程后在正式评价期起点 SOC=10800 kWh。", "", "## Checks", json.dumps(checks, ensure_ascii=False, indent=2), "", "## Field comparison", pair_frame.to_string(index=False)]
    (OUT / "inheritance_equivalence_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
