"""Causal rolling net-load forecasts for Question 2."""

from __future__ import annotations

import pandas as pd

from src.config import N_PERIODS


def forecast_net_load_quantile(
    actual: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    window_days: int,
    quantile: float,
) -> pd.DataFrame:
    """Forecast one day using only its preceding ``window_days`` observations.

    The explicit date filter is the data-leakage safeguard: observations from
    the forecast day or any later day are never included, even though Attachment
    2 is loaded as a complete annual table for offline backtesting.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between 0 and 1")
    required = {"date", "idx", "net_load_kwh"}
    if not required.issubset(actual.columns):
        raise ValueError(f"actual data must contain {sorted(required)}")

    date = pd.Timestamp(forecast_date).normalize()
    history = actual.loc[actual["date"] < date, ["date", "idx", "net_load_kwh"]].copy()
    history_dates = pd.DatetimeIndex(history["date"].drop_duplicates().sort_values())
    if len(history_dates) < window_days:
        raise ValueError(
            f"{date.date()} has only {len(history_dates)} prior dates; {window_days} are required"
        )
    selected_dates = history_dates[-window_days:]
    selected = history[history["date"].isin(selected_dates)]
    counts = selected.groupby("idx").size()
    if len(counts) != N_PERIODS or not (counts == window_days).all():
        raise ValueError("history window does not have one observation per slot and prior day")

    prediction = selected.groupby("idx", sort=True)["net_load_kwh"].quantile(quantile)
    result = pd.DataFrame(
        {
            "idx": range(N_PERIODS),
            "forecast_net_load_kwh": prediction.reindex(range(N_PERIODS)).to_numpy(dtype=float),
            "forecast_date": date,
            "history_start_date": selected_dates[0],
            "history_end_date": selected_dates[-1],
            "history_days": window_days,
            "quantile": quantile,
        }
    )
    if result["forecast_net_load_kwh"].isna().any():
        raise ValueError("net-load quantile forecast has missing slots")
    if not result["history_end_date"].iloc[0] < date:
        raise RuntimeError("future-data leakage detected in forecast history")
    return result
