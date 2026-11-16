"""Independent Stage-4 worker for the frozen no-adjustment control.

It deliberately calls the same ``run_policy`` implementation as the primary
runner.  A and B write to separate policy/day folders; the shared progress
file is informational only.  The primary runner performs final aggregation
after both policy checkpoints are complete.
"""

from src.forecast_q3 import load_q3_pv_forecast_records
from src.q2_v2.data import load_inputs

from .forecasting import build_forecast_cache
from .run_stage2 import _actual_long
from .run_stage4 import (
    FORMAL_END, STAGE3, _load_config, _policy_initialization, _sha256, run_policy,
)


def main() -> None:
    frozen = _load_config()
    config_hash = _sha256(STAGE3 / "q3_v2_frozen_config.json")
    initialization = _policy_initialization()["B_no_adjustment_dynamic_value"]
    data = load_inputs()
    cache = build_forecast_cache(_actual_long(data), load_q3_pv_forecast_records(), frozen["load_method"], "2025-01-01", FORMAL_END)
    run_policy("B_no_adjustment_dynamic_value", "no_adjustment", data, cache, frozen, config_hash, initialization)


if __name__ == "__main__":
    main()
