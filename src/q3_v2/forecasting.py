"""Causal Q3-V2 forecast replay variants sharing the audited PV mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast_q3 import build_q3_forecast_version

from .config import FIRST_MODIFIABLE_SLOT, LOAD_BIAS_LOOKBACK_SLOTS_TEST, LOAD_HALF_LIFE_DAYS_TEST


def _jan1_cold_forecast(actual: pd.DataFrame, pv_records: pd.DataFrame, issue_hour: int) -> pd.DataFrame:
    """Causal Jan-1 replay: zero load forecast and the actually released PV version."""
    day = pd.Timestamp("2025-01-01")
    issue = day + pd.Timedelta(hours=issue_hour)
    start = FIRST_MODIFIABLE_SLOT[issue_hour]
    future = actual.loc[(actual.date == day) & (actual.idx >= start)].sort_values("idx").copy()
    released = pv_records.loc[pv_records.issue_datetime == issue].sort_values("horizon_hour")
    if len(released) != 24:
        raise ValueError("Jan-1 release lacks 24 forecast hours")
    prior = pv_records.loc[(pv_records.target_datetime == issue) & (pv_records.issue_datetime < issue)].sort_values("issue_datetime")
    if len(prior):
        anchor = prior.iloc[-1]
        anchor_value = float(anchor.pv_forecast_kw)
        anchor_available = pd.Timestamp(anchor.issue_datetime)
        anchor_source = "previously_published_forecast"
    else:
        anchor = released.iloc[0]
        anchor_value = float(anchor.pv_forecast_kw)
        anchor_available = issue
        anchor_source = "current_release_h1_cold_start_hold"
    times = np.concatenate(([issue.value], released.target_datetime.to_numpy(dtype="datetime64[ns]").astype(np.int64)))
    values = np.concatenate(([anchor_value], released.pv_forecast_kw.to_numpy(float)))
    target = future.slot_datetime.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    pv_kw = np.interp(target.astype(float), times.astype(float), values)
    return pd.DataFrame({
        "date": day, "idx": future.idx.to_numpy(int), "time_label": future.time_label.to_numpy(),
        "issue_hour": issue_hour, "issue_datetime": issue, "target_datetime": future.slot_datetime.to_numpy(),
        "first_modifiable_idx": start, "load_forecast_kwh": 0.0, "load_bias_correction_kwh": 0.0,
        "pv_forecast_kw": pv_kw, "pv_forecast_kwh": pv_kw / 6.0, "pv_mapping_method": "linear",
        "pv_issue_anchor_source": anchor_source, "pv_actual_observation_used": False,
        "load_history_start": pd.NaT, "load_history_end": pd.NaT, "load_history_days": 0,
        "load_predictor_id": "jan1_cold_start_zero", "no_future_actual_load_used": True,
        "forecast_visible_at_decision": True, "pv_information_causal": anchor_available <= issue,
    })


def _weekday_load(actual: pd.DataFrame, day: pd.Timestamp, issue_hour: int) -> tuple[np.ndarray, dict]:
    history = actual.loc[actual["date"] < day].copy()
    dates = pd.DatetimeIndex(history["date"].drop_duplicates().sort_values())
    earliest = day - pd.Timedelta(days=35)
    selected_dates = [d for d in dates if d >= earliest and d.weekday() == day.weekday()]
    method = "mean_same_weekday_within_35_days"
    if not selected_dates:
        selected_dates = list(dates[-min(7, len(dates)):])
        method = "fallback_recent_up_to_7_days"
    if not selected_dates:
        raise ValueError("Weekday forecast needs at least one prior complete day")
    selected = history.loc[history["date"].isin(selected_dates)]
    base = selected.groupby("idx")["load_kwh"].mean().reindex(range(144))
    if base.isna().any():
        raise ValueError("Historical load days are incomplete")
    start = FIRST_MODIFIABLE_SLOT[issue_hour]
    bias = 0.0
    if start > 0:
        observed = actual.loc[(actual["date"] == day) & (actual["idx"] < start), ["idx", "load_kwh"]].sort_values("idx").tail(LOAD_BIAS_LOOKBACK_SLOTS_TEST)
        bias = float(np.median(observed.load_kwh.to_numpy() - base.iloc[observed.idx].to_numpy()))
    values = np.maximum(base.to_numpy(float) + bias, 0.0)
    return values, {"load_method": method, "load_bias_correction_kwh": bias,
                    "load_history_start": min(selected_dates), "load_history_end": max(selected_dates),
                    "load_history_days": len(selected_dates)}


def build_forecast_version(actual, pv_records, day, issue_hour: int, load_method: str) -> pd.DataFrame:
    day = pd.Timestamp(day).normalize()
    # This call supplies the already-audited PV version mapping and its source timestamps.
    frame = build_q3_forecast_version(
        actual, pv_records, day, issue_hour, history_days=14,
        half_life_days=LOAD_HALF_LIFE_DAYS_TEST,
        bias_lookback_slots=LOAD_BIAS_LOOKBACK_SLOTS_TEST, pv_mapping_method="linear",
    )
    if load_method == "ewma14_bias12":
        frame["load_predictor_id"] = load_method
        return frame
    if load_method != "weekday35_bias12":
        raise ValueError(f"Unknown load predictor: {load_method}")
    load, meta = _weekday_load(actual, day, issue_hour)
    idx = frame.idx.to_numpy(int)
    frame["load_forecast_kwh"] = load[idx]
    frame["net_load_forecast_kwh"] = frame.load_forecast_kwh - frame.pv_forecast_kwh
    frame["load_predictor_id"] = load_method
    for key, value in meta.items():
        frame[key] = value
    frame["no_future_actual_load_used"] = pd.to_datetime(frame.load_history_end) < day
    return frame


def build_forecast_cache(
    actual: pd.DataFrame,
    pv_records: pd.DataFrame,
    load_method: str,
    start_date: str = "2025-01-01",
    end_date: str = "2025-01-31",
) -> dict[tuple[str, int], pd.DataFrame]:
    """Replay only forecasts that would have been available at each issue time.

    The function deliberately builds one immutable forecast record for every
    historical decision version.  Later scenario construction can therefore
    use residuals against the *then available* forecast rather than a
    hindsight re-fit.  ``start_date``/``end_date`` are inclusive.
    """
    cache = {}
    for day in pd.date_range(start_date, end_date, freq="D"):
        for issue in (0, 6, 12, 18):
            if day == pd.Timestamp("2025-01-01"):
                frame = _jan1_cold_forecast(actual, pv_records, issue)
            else:
                frame = build_forecast_version(actual, pv_records, day, issue, load_method)
            cache[(day.date().isoformat(), issue)] = frame
    return cache


def build_month_cache(actual, pv_records, load_method: str) -> dict[tuple[str, int], pd.DataFrame]:
    """Backward-compatible January wrapper used by the frozen Stage-3 run."""
    return build_forecast_cache(actual, pv_records, load_method, "2025-01-01", "2025-01-31")
