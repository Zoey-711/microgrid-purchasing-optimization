"""Second Stage-3 step: compare M on the competitive v=0.4 DP candidate."""

from __future__ import annotations

import json

import pandas as pd

from .config import ROOT
from .data import load_inputs
from .forecast import build_causal_replay
from .run_stage3_screen import Policy, protected_current, run_policy


OUT = ROOT / "outputs" / "q2_v2" / "stage3"
POLICIES = (
    Policy("M21_v04_dp100", "scenario", 21, 0.4, "dp", 100.0),
    Policy("M30_v04_dp100", "scenario", 30, 0.4, "dp", 100.0),
)


def run() -> None:
    before = protected_current()
    data = load_inputs()
    replay = build_causal_replay(data)
    rows = [run_policy(p, data, replay, run_group="m_screen") for p in POLICIES]
    pd.DataFrame(rows).to_csv(OUT / "q2_v2_january_m_candidates.csv", index=False)
    after = protected_current()
    (OUT / "q2_v2_stage3_m_hash_check.json").write_text(
        json.dumps({"changed": [k for k in before if before[k] != after[k]], "before": before, "after": after},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
