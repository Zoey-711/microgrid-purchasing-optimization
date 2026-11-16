"""Higher precision rerun for the three competitive M=14 DP candidates."""

from __future__ import annotations

import json

import pandas as pd

from .config import ROOT
from .data import load_inputs
from .forecast import build_causal_replay
from .run_stage3_screen import Policy, protected_current, run_policy


OUT = ROOT / "outputs" / "q2_v2" / "stage3"
TIME_LIMIT = 20.0
REL_GAP = 0.0002
POLICIES = (
    Policy("P2_M14_v0_dp100_refined", "scenario", 14, 0.0, "dp", 100.0),
    Policy("P3_M14_v04_dp100_refined", "scenario", 14, 0.4, "dp", 100.0),
    Policy("P4_M14_v08_dp100_refined", "scenario", 14, 0.8, "dp", 100.0),
)


def run() -> None:
    before = protected_current()
    data = load_inputs()
    replay = build_causal_replay(data)
    rows = [run_policy(p, data, replay, run_group="refine", time_limit_seconds=TIME_LIMIT,
                       relative_gap=REL_GAP) for p in POLICIES]
    pd.DataFrame(rows).to_csv(OUT / "q2_v2_january_refined_candidates.csv", index=False)
    after = protected_current()
    (OUT / "q2_v2_stage3_refine_hash_check.json").write_text(
        json.dumps({"changed": [k for k in before if before[k] != after[k]], "before": before, "after": after},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
