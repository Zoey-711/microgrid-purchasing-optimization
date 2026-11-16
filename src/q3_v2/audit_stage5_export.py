"""Independent post-write audit for the authorised Q3 V2 result3 workbook."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .export_stage5 import (
    CANDIDATE, DOCS, EPS, FINAL_FILE, FORMAL_DATES, POLICY_A, PROTECTED,
    ROOT, SLOTS, STAGE4, STAGE5, assert_template, event_records, load_a_slots,
    protected_hashes, sha256, validate_candidate,
)


def _canonical_time(value: str) -> str:
    value = value.replace("+1", "")
    hour, minute = value.split(":", 1)
    return f"{int(hour):02d}:{int(minute):02d}"


def version_and_mapping_audit(slots: pd.DataFrame, labels: list[str]) -> dict:
    plans = pd.read_csv(STAGE4 / "q3_v2_plan_versions.csv")
    plans = plans.loc[plans["policy"].eq(POLICY_A)].copy()
    plans["date"] = pd.to_datetime(plans["date"]).dt.normalize()
    plans = plans.loc[plans["date"].isin(FORMAL_DATES)]
    active = slots[["date", "slot", "time_label", "active_plan_run_id", "effective_purchase_kwh"]].copy()
    active = active.merge(
        plans[["run_id", "slot", "version_plan_kwh", "executed_prefix_lock_error_kwh"]],
        left_on=["active_plan_run_id", "slot"], right_on=["run_id", "slot"], how="left", validate="one_to_one",
    )
    if active["version_plan_kwh"].isna().any():
        raise AssertionError("An active frozen plan version is absent from the version table")
    effective_error = float(np.max(np.abs(active["effective_purchase_kwh"] - active["version_plan_kwh"])))
    lock_error = float(np.max(np.abs(plans["executed_prefix_lock_error_kwh"])))
    label_errors = []
    sample = slots.loc[slots["date"].eq(FORMAL_DATES[0])].sort_values("slot")
    for slot, (_, row) in enumerate(sample.iterrows()):
        template_left = str(labels[slot]).split("-", 1)[0].replace("+1", "")
        label_errors.append(_canonical_time(template_left) != _canonical_time(str(row["time_label"])))
    if any(label_errors):
        raise AssertionError("Official template column order differs from frozen source time labels")
    if max(effective_error, lock_error) > EPS:
        raise AssertionError("Executed plan-version lock audit failed")
    return {
        "active_plan_version_cell_count": int(len(active)),
        "active_effective_plan_max_error_kwh": effective_error,
        "executed_prefix_lock_max_error_kwh": lock_error,
        "official_slot_time_mapping_errors": int(sum(label_errors)),
    }


def run() -> None:
    if not FINAL_FILE.exists():
        raise FileNotFoundError(f"Missing formal workbook: {FINAL_FILE}")
    protected_before = json.loads(PROTECTED.read_text(encoding="utf-8"))
    current_protected = protected_hashes()
    template = assert_template()
    slots = load_a_slots()
    checks, cells = validate_candidate(slots, template, event_records(slots, template["labels"]), workbook_path=FINAL_FILE)
    version = version_and_mapping_audit(slots, template["labels"])
    candidate_match = CANDIDATE.exists() and sha256(CANDIDATE) == sha256(FINAL_FILE)
    if not candidate_match:
        raise AssertionError("Formal workbook does not byte-match the validated candidate")
    if sha256(ROOT / "data/raw/result3.xlsx") != template["raw_sha256"]:
        raise AssertionError("Raw result3 template hash changed")
    STAGE5.mkdir(parents=True, exist_ok=True)
    cells.to_csv(STAGE5 / "q3_v2_result3_cell_check.csv", index=False, encoding="utf-8-sig")
    payload = {
        "postwrite_audit_time": datetime.now().isoformat(timespec="seconds"),
        "result3_sha256": sha256(FINAL_FILE),
        "raw_template_sha256": template["raw_sha256"],
        "candidate_byte_identical_to_result3": candidate_match,
        "protected_manifest_unchanged": current_protected == protected_before,
        "protected_file_count": len(current_protected),
        "matrix_and_physical_checks": checks,
        "version_lock_and_time_mapping": version,
        "passed": True,
    }
    audit = DOCS / "q3_v2_export_audit.md"
    with audit.open("a", encoding="utf-8") as file:
        file.write("\n## 正式文件写后复核\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n")
    with (DOCS / "q3_v2_strategy_freeze_record.md").open("a", encoding="utf-8") as file:
        file.write(f"\n- 写后复核时间：{payload['postwrite_audit_time']}；正式文件反读、计划版本锁定、官方时间列映射和受保护文件哈希均通过。\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
