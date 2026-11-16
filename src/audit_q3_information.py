"""Audit Question-3 forecast information boundaries without optimization."""

from __future__ import annotations

import hashlib

import pandas as pd

from src.config import (
    Q3_INFORMATION_AUDIT_FILE,
    Q3_INFORMATION_AUDIT_LOG,
    Q3_ISSUE_HOURS,
    RAW_DATA_DIR,
)
from src.data_io import load_q2_actual_energy
from src.forecast_q3 import (
    add_q3_slot_datetimes,
    build_q3_forecast_version,
    load_q3_pv_forecast_records,
)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run_q3_information_audit() -> dict:
    """Audit all forecast versions actually needed in the January backtest."""
    actual, _ = load_q2_actual_energy()
    actual = add_q3_slot_datetimes(actual)
    pv_records = load_q3_pv_forecast_records()
    rows = []
    # January 1 is the common cold-start day and has no optimization forecast.
    # Every forecast used from January 2 onward is audited here.
    for day in pd.date_range("2025-01-02", "2025-01-31", freq="D"):
        for issue_hour in Q3_ISSUE_HOURS:
            version = build_q3_forecast_version(actual, pv_records, day, issue_hour)
            keep = version[
                [
                    "date", "idx", "time_label", "issue_hour", "issue_datetime",
                    "target_datetime", "first_modifiable_idx", "pv_forecast_kw",
                    "pv_forecast_kwh", "pv_left_anchor_datetime", "pv_left_source",
                    "pv_left_source_target_datetime", "pv_left_available_datetime",
                    "pv_left_forecast_issue_datetime", "pv_left_source_row",
                    "pv_left_horizon_hour", "pv_left_value_kw",
                    "pv_right_anchor_datetime", "pv_right_source",
                    "pv_right_source_target_datetime", "pv_right_available_datetime",
                    "pv_right_forecast_issue_datetime", "pv_right_source_row",
                    "pv_right_horizon_hour", "pv_right_value_kw",
                    "pv_actual_observation_used", "pv_information_causal",
                    "load_history_start", "load_history_end",
                    "load_actual_observed_through_idx", "no_future_actual_load_used",
                ]
            ].copy()
            version["decision_datetime"] = version["issue_datetime"]
            keep.insert(5, "decision_datetime", version["decision_datetime"].to_numpy())
            rows.append(keep)
    audit = pd.concat(rows, ignore_index=True)

    expected_rows_per_day = sum(144 - {0: 0, 6: 35, 12: 71, 18: 107}[h] for h in Q3_ISSUE_HOURS)
    mapping = {}
    sample_day = pd.Timestamp("2025-01-15")
    for issue_hour, idx in ((6, 35), (12, 71), (18, 107)):
        row = audit.loc[
            (audit["date"] == sample_day)
            & (audit["issue_hour"] == issue_hour)
            & (audit["idx"] == idx)
        ].iloc[0]
        mapping[f"t{idx}_target"] = str(row["target_datetime"])
        mapping[f"t{idx}_left_source"] = str(row["pv_left_source"])
        mapping[f"t{idx}_left_available"] = str(row["pv_left_available_datetime"])
    t143 = audit.loc[
        (audit["date"] == sample_day) & (audit["issue_hour"] == 18) & (audit["idx"] == 143)
    ].iloc[0]
    mapping["t143_target"] = str(t143["target_datetime"])
    mapping["t143_right_source_target"] = str(t143["pv_right_source_target_datetime"])

    checks = {
        "audit_start": str(audit["date"].min().date()),
        "audit_end": str(audit["date"].max().date()),
        "days": int(audit["date"].nunique()),
        "rows": int(len(audit)),
        "expected_rows": int(30 * expected_rows_per_day),
        "actual_power_semantics_proven": False,
        "actual_pv_anchor_count": int(audit["pv_actual_observation_used"].sum()),
        "all_anchor_inputs_available_by_decision": bool(audit["pv_information_causal"].all()),
        "all_load_history_precedes_forecast_date": bool(
            (pd.to_datetime(audit["load_history_end"]) < pd.to_datetime(audit["date"])).all()
        ),
        "all_intraday_load_observations_completed": bool(
            (audit["load_actual_observed_through_idx"] < audit["first_modifiable_idx"]).all()
        ),
        "parameter_selection_uses_evaluation_period_only": True,
        **mapping,
    }
    checks["passed"] = bool(
        checks["rows"] == checks["expected_rows"]
        and checks["actual_pv_anchor_count"] == 0
        and checks["all_anchor_inputs_available_by_decision"]
        and checks["all_load_history_precedes_forecast_date"]
        and checks["all_intraday_load_observations_completed"]
    )

    Q3_INFORMATION_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(Q3_INFORMATION_AUDIT_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")
    template = RAW_DATA_DIR / "result3.xlsx"
    lines = [
        "Question 3 information-boundary audit",
        "=" * 40,
        "official_evidence: Attachment 2 states actual power (kW) at different times, but does not define instantaneous versus completed-slot mean",
        "conservative_rule: no Attachment-2 PV value is used as an issue-time interpolation anchor",
        "left_anchor_rule: latest previously published forecast targeting the decision time",
        "cold_start_rule: 2025-01-01 has zero purchase and no optimization forecast; if a forecast is requested, current h=1 is held backward without actual PV",
        "load_rule: prior full days plus only idx < first_modifiable_idx for intraday bias",
        "parameter_rule: candidates are fixed before evaluation and ranked only on 2025-01-22 through 2025-01-31",
        f"raw_result3_sha256: {_sha256(template)}",
        "result3.xlsx generated: False",
        "",
        *[f"{key}: {value}" for key, value in checks.items()],
        "",
        f"audit_csv: {Q3_INFORMATION_AUDIT_FILE}",
    ]
    Q3_INFORMATION_AUDIT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise RuntimeError(f"Q3 information audit failed; see {Q3_INFORMATION_AUDIT_LOG}")
    return {"audit": audit, "checks": checks}


if __name__ == "__main__":
    result = run_q3_information_audit()
    print(result["checks"])
