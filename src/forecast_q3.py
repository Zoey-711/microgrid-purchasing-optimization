"""Causal load and multi-release PV forecasts for Question 3."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from src.config import (
    DELTA_T,
    N_PERIODS,
    Q3_FIRST_MODIFIABLE_SLOT,
    Q3_FORECAST_INPUT_PATH,
    Q3_LOAD_BIAS_LOOKBACK_SLOTS,
    Q3_LOAD_DECAY_HALF_LIFE_DAYS,
    Q3_LOAD_HISTORY_DAYS,
)


def _hour_from_value(value) -> int:
    if isinstance(value, time):
        if value.minute or value.second:
            raise ValueError(f"Forecast issue time is not an exact hour: {value}")
        return int(value.hour)
    parsed = pd.to_datetime(str(value), format="%H:%M", errors="raise")
    return int(parsed.hour)


def load_q3_pv_forecast_records(path=Q3_FORECAST_INPUT_PATH) -> pd.DataFrame:
    """Read Attachment 3 into a versioned issue-target long table."""
    raw = pd.read_excel(path, sheet_name=0)
    if raw.shape != (1460, 26):
        raise ValueError(f"Attachment 3 must have 1460 rows x 26 columns, found {raw.shape}")
    dates = pd.to_datetime(raw.iloc[:, 0].ffill(), errors="raise").dt.normalize()
    issue_hours = raw.iloc[:, 1].map(_hour_from_value)
    if set(issue_hours.unique()) != {0, 6, 12, 18}:
        raise ValueError(f"Unexpected issue hours: {sorted(issue_hours.unique())}")
    values = raw.iloc[:, 2:26].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError("Attachment 3 contains missing or negative forecasts")

    rows: list[dict] = []
    for row_idx in range(len(raw)):
        issue = dates.iloc[row_idx] + pd.Timedelta(hours=int(issue_hours.iloc[row_idx]))
        for h in range(1, 25):
            rows.append(
                {
                    "source_row": row_idx + 2,
                    "issue_date": dates.iloc[row_idx],
                    "issue_hour": int(issue_hours.iloc[row_idx]),
                    "issue_datetime": issue,
                    "horizon_hour": h,
                    "target_datetime": issue + pd.Timedelta(hours=h),
                    "pv_forecast_kw": float(values[row_idx, h - 1]),
                }
            )
    result = pd.DataFrame(rows)
    if result.duplicated(["issue_datetime", "horizon_hour"]).any():
        raise ValueError("Duplicate Attachment-3 issue/horizon keys")
    return result


def add_q3_slot_datetimes(actual: pd.DataFrame) -> pd.DataFrame:
    """Map official row t to its audited Attachment-2 timestamp.

    Attachment 2 columns are 00:10, ..., 23:50, 00:00+1. Therefore row t
    receives timestamp date + (t+1)*10 minutes. This is a mapping convention,
    not an extra interpretation of the template interval label.
    """
    result = actual.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["slot_datetime"] = result["date"] + pd.to_timedelta((result["idx"] + 1) * 10, unit="min")
    if result.duplicated("slot_datetime").any():
        raise ValueError("Actual slot timestamps are not unique")
    return result


def _load_history_forecast(
    actual_with_time: pd.DataFrame,
    forecast_date: pd.Timestamp,
    issue_hour: int,
    history_days: int = Q3_LOAD_HISTORY_DAYS,
    half_life_days: float = Q3_LOAD_DECAY_HALF_LIFE_DAYS,
    bias_lookback_slots: int = Q3_LOAD_BIAS_LOOKBACK_SLOTS,
) -> tuple[np.ndarray, float, pd.Timestamp, pd.Timestamp, int]:
    """Exponentially weight historical same-slot load, then apply causal bias."""
    history = actual_with_time.loc[actual_with_time["date"] < forecast_date].copy()
    available_dates = pd.DatetimeIndex(history["date"].drop_duplicates().sort_values())
    if len(available_dates) == 0:
        raise ValueError("History-only load forecast requires at least one prior day")
    if history_days <= 0 or half_life_days <= 0 or bias_lookback_slots < 0:
        raise ValueError("Load forecast parameters must be nonnegative, with positive history/half-life")
    selected_dates = available_dates[-min(history_days, len(available_dates)):]
    selected = history.loc[history["date"].isin(selected_dates)].copy()
    age_days = (forecast_date - selected["date"]).dt.days.to_numpy(dtype=float)
    selected["weight"] = np.power(0.5, age_days / half_life_days)
    selected["weighted_load"] = selected["weight"] * selected["load_kwh"]
    grouped = selected.groupby("idx", sort=True)
    base = (grouped["weighted_load"].sum() / grouped["weight"].sum()).reindex(range(N_PERIODS))
    if base.isna().any():
        raise ValueError("Historical load window is incomplete")

    start_idx = Q3_FIRST_MODIFIABLE_SLOT[issue_hour]
    bias = 0.0
    if start_idx > 0 and bias_lookback_slots > 0:
        observed = actual_with_time.loc[
            (actual_with_time["date"] == forecast_date) & (actual_with_time["idx"] < start_idx),
            ["idx", "load_kwh"],
        ].sort_values("idx")
        if len(observed) != start_idx:
            raise ValueError("Elapsed current-day load observations are incomplete")
        observed = observed.tail(bias_lookback_slots)
        residual = observed["load_kwh"].to_numpy(dtype=float) - base.iloc[observed["idx"]].to_numpy(dtype=float)
        bias = float(np.median(residual))
    adjusted = np.maximum(base.to_numpy(dtype=float) + bias, 0.0)
    return adjusted, bias, selected_dates[0], selected_dates[-1], len(selected_dates)


def build_q3_forecast_version(
    actual_with_time: pd.DataFrame,
    pv_records: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    issue_hour: int,
    history_days: int = Q3_LOAD_HISTORY_DAYS,
    half_life_days: float = Q3_LOAD_DECAY_HALF_LIFE_DAYS,
    bias_lookback_slots: int = Q3_LOAD_BIAS_LOOKBACK_SLOTS,
    pv_mapping_method: str = "linear",
) -> pd.DataFrame:
    """Build one causal forecast version for current-day modifiable slots."""
    if issue_hour not in Q3_FIRST_MODIFIABLE_SLOT:
        raise ValueError(f"Unsupported issue hour: {issue_hour}")
    day = pd.Timestamp(forecast_date).normalize()
    issue = day + pd.Timedelta(hours=issue_hour)
    start_idx = Q3_FIRST_MODIFIABLE_SLOT[issue_hour]
    day_rows = actual_with_time.loc[actual_with_time["date"] == day].sort_values("idx")
    if len(day_rows) != N_PERIODS or day_rows["idx"].tolist() != list(range(N_PERIODS)):
        raise ValueError(f"Actual day {day.date()} does not have 144 ordered slots")
    future_rows = day_rows.loc[day_rows["idx"] >= start_idx].copy()
    if not (future_rows["slot_datetime"] >= issue).all():
        raise AssertionError("A modifiable target precedes its issue time")

    issue_forecasts = pv_records.loc[pv_records["issue_datetime"] == issue].sort_values("horizon_hour")
    if len(issue_forecasts) != 24:
        raise ValueError(f"Missing 24-hour PV forecast at {issue}")
    # Attachment 2 calls these values "actual power" but does not establish
    # whether a value stamped at an issue time is instantaneous or the mean of
    # a completed interval.  It is therefore not used as the issue-time anchor.
    prior_anchor = pv_records.loc[
        (pv_records["target_datetime"] == issue)
        & (pv_records["issue_datetime"] < issue)
    ].sort_values("issue_datetime")
    if not prior_anchor.empty:
        anchor_record = prior_anchor.iloc[-1]
        anchor_kw = float(anchor_record["pv_forecast_kw"])
        anchor_source = "previously_published_forecast"
        anchor_source_issue = pd.Timestamp(anchor_record["issue_datetime"])
        anchor_source_target = pd.Timestamp(anchor_record["target_datetime"])
        anchor_available = anchor_source_issue
        anchor_source_row = int(anchor_record["source_row"])
        anchor_horizon_hour = int(anchor_record["horizon_hour"])
    elif issue == pd.Timestamp("2025-01-01 00:00:00"):
        # No earlier attachment record exists.  Hold the already released h=1
        # value backwards over the first hour; this uses no actual observation.
        anchor_record = issue_forecasts.iloc[0]
        anchor_kw = float(anchor_record["pv_forecast_kw"])
        anchor_source = "current_release_h1_cold_start_hold"
        anchor_source_issue = issue
        anchor_source_target = pd.Timestamp(anchor_record["target_datetime"])
        anchor_available = issue
        anchor_source_row = int(anchor_record["source_row"])
        anchor_horizon_hour = int(anchor_record["horizon_hour"])
    else:
        raise ValueError(f"No previously published PV anchor at {issue}")

    # Force nanosecond resolution before integer conversion. Pandas may retain
    # Excel datetimes at microsecond resolution, whose raw integers cannot be
    # mixed with Timestamp.value (nanoseconds).
    forecast_target_ns = (
        issue_forecasts["target_datetime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    )
    anchor_times = np.concatenate(([issue.value], forecast_target_ns))
    anchor_values = np.concatenate(([anchor_kw], issue_forecasts["pv_forecast_kw"].to_numpy(dtype=float)))
    anchor_sources = np.concatenate(([anchor_source], np.repeat("current_release_forecast", 24)))
    anchor_source_issues = np.concatenate((
        [anchor_source_issue.to_datetime64()],
        issue_forecasts["issue_datetime"].to_numpy(dtype="datetime64[ns]"),
    )).astype("datetime64[ns]")
    anchor_source_targets = np.concatenate((
        [anchor_source_target.to_datetime64()],
        issue_forecasts["target_datetime"].to_numpy(dtype="datetime64[ns]"),
    )).astype("datetime64[ns]")
    anchor_available_times = np.concatenate((
        [anchor_available.to_datetime64()],
        issue_forecasts["issue_datetime"].to_numpy(dtype="datetime64[ns]"),
    )).astype("datetime64[ns]")
    anchor_source_rows = np.concatenate(([anchor_source_row], issue_forecasts["source_row"].to_numpy(dtype=int)))
    anchor_horizons = np.concatenate(([anchor_horizon_hour], issue_forecasts["horizon_hour"].to_numpy(dtype=int)))
    target_ns = future_rows["slot_datetime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if target_ns.min() < anchor_times.min() or target_ns.max() > anchor_times.max():
        raise ValueError("Current-day target lies outside the visible 24-hour forecast")
    right_index = np.searchsorted(anchor_times, target_ns, side="left")
    right_index = np.clip(right_index, 1, len(anchor_times) - 1)
    exact = anchor_times[right_index] == target_ns
    left_index = np.where(exact, right_index, right_index - 1)
    if pv_mapping_method == "linear":
        pv_kw = np.interp(target_ns.astype(float), anchor_times.astype(float), anchor_values)
    elif pv_mapping_method == "right_hold":
        # Each forecast for an integer-hour target is held over the preceding
        # six 10-minute target rows.  This is a sensitivity convention only.
        pv_kw = anchor_values[right_index]
    else:
        raise ValueError(f"Unsupported PV mapping method: {pv_mapping_method}")
    load_kwh, bias, hist_start, hist_end, hist_days = _load_history_forecast(
        actual_with_time, day, issue_hour, history_days, half_life_days, bias_lookback_slots
    )
    idx = future_rows["idx"].to_numpy(dtype=int)
    result = pd.DataFrame(
        {
            "date": day,
            "idx": idx,
            "time_label": future_rows["time_label"].to_numpy(),
            "issue_hour": issue_hour,
            "issue_datetime": issue,
            "target_datetime": future_rows["slot_datetime"].to_numpy(),
            "forecast_horizon_minutes": (
                (future_rows["slot_datetime"] - issue).dt.total_seconds() / 60.0
            ).to_numpy(),
            "first_modifiable_idx": start_idx,
            "load_forecast_kwh": load_kwh[idx],
            "load_bias_correction_kwh": bias,
            "pv_forecast_kw": pv_kw,
            "pv_forecast_kwh": pv_kw * DELTA_T,
            "pv_mapping_method": pv_mapping_method,
            "pv_left_anchor_datetime": pd.to_datetime(anchor_times[left_index]),
            "pv_right_anchor_datetime": pd.to_datetime(anchor_times[right_index]),
            "pv_issue_anchor_source": anchor_source,
            "pv_left_source": anchor_sources[left_index],
            "pv_left_source_target_datetime": pd.to_datetime(anchor_source_targets[left_index]),
            "pv_left_available_datetime": pd.to_datetime(anchor_available_times[left_index]),
            "pv_left_forecast_issue_datetime": pd.to_datetime(anchor_source_issues[left_index]),
            "pv_left_source_row": anchor_source_rows[left_index],
            "pv_left_horizon_hour": anchor_horizons[left_index],
            "pv_left_value_kw": anchor_values[left_index],
            "pv_right_source": anchor_sources[right_index],
            "pv_right_source_target_datetime": pd.to_datetime(anchor_source_targets[right_index]),
            "pv_right_available_datetime": pd.to_datetime(anchor_available_times[right_index]),
            "pv_right_forecast_issue_datetime": pd.to_datetime(anchor_source_issues[right_index]),
            "pv_right_source_row": anchor_source_rows[right_index],
            "pv_right_horizon_hour": anchor_horizons[right_index],
            "pv_right_value_kw": anchor_values[right_index],
            "pv_actual_observation_used": False,
            "load_history_start": hist_start,
            "load_history_end": hist_end,
            "load_history_days": hist_days,
            "load_actual_observed_through_idx": start_idx - 1,
            "load_history_parameter_days": history_days,
            "load_half_life_days": half_life_days,
            "load_bias_lookback_slots": bias_lookback_slots,
        }
    )
    result["net_load_forecast_kwh"] = result["load_forecast_kwh"] - result["pv_forecast_kwh"]
    result["no_future_actual_load_used"] = result["load_history_end"] < day
    result["forecast_visible_at_decision"] = result["issue_datetime"] <= issue
    result["pv_information_causal"] = (
        (result["pv_left_available_datetime"] <= result["issue_datetime"])
        & (result["pv_right_available_datetime"] <= result["issue_datetime"])
        & ~result["pv_actual_observation_used"]
    )
    return result
