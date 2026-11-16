"""Final January causal replay for the selected Stage-3 configuration."""

from __future__ import annotations

import json

from .config import ROOT
from .data import load_inputs
from .forecast import build_causal_replay
from .run_stage3_screen import Policy, protected_current, run_policy


def run() -> None:
    before = protected_current()
    data = load_inputs()
    replay = build_causal_replay(data)
    policy = Policy("selected_M14_v04_dp50", "scenario", 14, 0.4, "dp", 50.0)
    summary = run_policy(policy, data, replay, run_group="selected", time_limit_seconds=45.0,
                         relative_gap=0.0001)
    out = ROOT / "outputs" / "q2_v2" / "stage3"
    (out / "q2_v2_selected_development_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    after = protected_current()
    (out / "q2_v2_stage3_selected_hash_check.json").write_text(
        json.dumps({"changed": [k for k in before if before[k] != after[k]], "before": before, "after": after},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
