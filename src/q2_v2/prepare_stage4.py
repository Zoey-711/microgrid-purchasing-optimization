"""Write the Stage-4 preregistration record before formal evaluation."""

from __future__ import annotations

import hashlib
import json

from .config import ROOT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run() -> None:
    out = ROOT / "outputs/q2_v2/stage4"
    out.mkdir(parents=True, exist_ok=True)
    protected = json.loads((ROOT / "outputs/q2_v2/audit/q2_v2_protected_hashes_after.json").read_text(encoding="utf-8"))
    current = {name: sha256(ROOT / name) for name in protected}
    frozen_path = ROOT / "outputs/q2_v2/stage3/q2_v2_frozen_strategy_config.json"
    record = {
        "created_before_formal_evaluation": True,
        "frozen_config_path": str(frozen_path.relative_to(ROOT)),
        "frozen_config_sha256": sha256(frozen_path),
        "formal_period": ["2025-02-01", "2025-12-31"],
        "formal_days": 334,
        "policies": ["main_dp50", "closed_loop_greedy", "paired_greedy_same_plan_soc"],
        "no_solution_fallback": None,
        "on_no_solution": "checkpoint and stop",
        "parameter_changes_after_results_allowed": False,
        "result2_generation_authorized": False,
        "protected_hash_changes_before_run": [k for k in protected if protected[k] != current[k]],
    }
    (out / "q2_v2_stage4_preregistered_protocol.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "protected_hashes_before_stage4.json").write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
