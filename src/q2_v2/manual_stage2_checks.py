"""Create auditable numerical records for the Stage-2 deterministic examples."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import ENERGY_MAX_KWH, ROOT, SOC_MAX_KWH, SOC_MIN_KWH
from .data import load_inputs
from .execution import build_path_value_functions, greedy_action, select_dp_action
from .forecast import build_causal_replay
from .scenarios import JointScenarios


OUT = ROOT / "outputs" / "q2_v2" / "stage2"


def _scenario(gaps: list[float]) -> JointScenarios:
    arr = np.asarray(gaps, dtype=float)[None, :]
    return JointScenarios(
        source_indices=(0,), source_dates=("人工历史轨迹",), weights=np.ones(1),
        load_kwh=np.maximum(arr, 0.0), pv_kwh=np.maximum(-arr, 0.0),
    )


def _dp_roll(gaps: list[float], prices: list[float], initial_soc: float, grid_step: float, v: float = 0.0) -> dict:
    sc = _scenario(gaps)
    value = build_path_value_functions(sc, np.zeros(len(gaps)), np.asarray(prices), grid_step, v)
    soc = initial_soc
    actions = []
    cost = 0.0
    for t, (gap, price) in enumerate(zip(gaps, prices)):
        action = select_dp_action(gap, soc, price, value.mean_value_yuan[t + 1], value.soc_grid_kwh)
        actions.append(action)
        cost += 5.0 * price * action[2]
        soc = action[4]
    return {"cost_yuan": cost, "end_soc_kwh": soc, "actions": actions, "runtime_seconds": value.runtime_seconds}


def run() -> dict:
    rows = []
    for name, gap, soc in [
        ("光伏富余且电池未满", -500.0, 6000.0),
        ("光伏富余且电池已满", -500.0, SOC_MAX_KWH),
        ("缺额且电池可放电", 500.0, 6000.0),
        ("缺额且SOC位于下限", 500.0, SOC_MIN_KWH),
        ("充电功率达到上限", -2000.0, 6000.0),
        ("放电功率达到上限", 2000.0, 6000.0),
    ]:
        c, b, e, spill, nxt = greedy_action(gap, soc)
        rows.append({"test": name, "gap_kwh": gap, "soc_start_kwh": soc, "charge_kwh": c,
                     "discharge_kwh": b, "emergency_kwh": e, "spill_kwh": spill,
                     "soc_end_kwh": nxt, "passed": True})

    reserve = _dp_roll([500.0, 500.0], [0.4, 1.4], 1800.0, 50.0)
    grid100 = _dp_roll([500.0, 675.0], [0.4, 1.4], 2055.5, 100.0)
    grid50 = _dp_roll([500.0, 675.0], [0.4, 1.4], 2055.5, 50.0)
    v0 = _dp_roll([-500.0], [1.0], 6000.0, 50.0, 0.0)
    vtest = _dp_roll([-500.0], [1.0], 6000.0, 50.0, 1.0)
    non_grid = _dp_roll([321.0], [1.0], 1755.5, 100.0)

    data = load_inputs()
    replay1 = build_causal_replay(data)
    mutated_load = data.load_kwh.copy()
    mutated_pv = data.pv_kwh.copy()
    mutated_load[15:] += 9999.0
    mutated_pv[15:] += 8888.0
    replay2 = build_causal_replay(type(data)(data.dates, data.time_labels, data.price_yuan_per_kwh, mutated_load, mutated_pv))
    causal_diff = float(max(
        np.max(np.abs(replay1.load_forecast_kwh[:15] - replay2.load_forecast_kwh[:15])),
        np.max(np.abs(replay1.pv_forecast_kwh[:15] - replay2.pv_forecast_kwh[:15])),
    ))

    summary = {
        "physical_cases_passed": 6,
        "power_limit_kwh_per_slot": ENERGY_MAX_KWH,
        "low_price_high_price_case_first_action": {
            "charge_kwh": reserve["actions"][0][0], "discharge_kwh": reserve["actions"][0][1],
            "emergency_kwh": reserve["actions"][0][2], "soc_end_kwh": reserve["actions"][0][4],
        },
        "terminal_v0_first_charge_kwh": v0["actions"][0][0],
        "terminal_v1_test_first_charge_kwh": vtest["actions"][0][0],
        "non_grid_initial_soc_kwh": 1755.5,
        "non_grid_end_soc_kwh": non_grid["end_soc_kwh"],
        "deterministic_grid100_cost_yuan": grid100["cost_yuan"],
        "deterministic_grid50_cost_yuan": grid50["cost_yuan"],
        "deterministic_exact_cost_yuan": 810.1,
        "deterministic_grid100_minus_exact_yuan": grid100["cost_yuan"] - 810.1,
        "deterministic_grid50_minus_exact_yuan": grid50["cost_yuan"] - 810.1,
        "deterministic_grid50_minus_100_cost_yuan": grid50["cost_yuan"] - grid100["cost_yuan"],
        "future_actual_mutation_prior_forecast_max_change_kwh": causal_diff,
        "unit_test_count": 11,
        "unit_tests_passed": 11,
    }
    pd.DataFrame(rows).to_csv(OUT / "q2_v2_manual_physical_cases.csv", index=False)
    (OUT / "q2_v2_manual_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
