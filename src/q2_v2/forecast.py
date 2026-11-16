"""Causal historical forecasts and replayed residuals for Q2-v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Q2InputData


@dataclass(frozen=True)
class ForecastReplay:
    load_forecast_kwh: np.ndarray
    pv_forecast_kwh: np.ndarray
    load_residual_kwh: np.ndarray
    pv_residual_kwh: np.ndarray
    source_audit: pd.DataFrame


def build_causal_replay(data: Q2InputData) -> ForecastReplay:
    n_days, n_slots = data.load_kwh.shape
    load_fc = np.zeros((n_days, n_slots), dtype=float)
    pv_fc = np.zeros((n_days, n_slots), dtype=float)
    audit_rows: list[dict] = []
    for d in range(n_days):
        decision_date = data.dates[d]
        if d == 0:
            load_sources: list[int] = []
            pv_sources: list[int] = []
            load_method = "cold_start_zero"
            pv_method = "cold_start_zero"
        else:
            earliest = max(0, d - 35)
            same_weekday = [i for i in range(earliest, d) if data.dates[i].weekday() == decision_date.weekday()]
            if same_weekday:
                load_sources = same_weekday
                load_method = "mean_same_weekday_within_35_days"
            else:
                load_sources = list(range(max(0, d - 7), d))
                load_method = "fallback_recent_up_to_7_days"
            pv_sources = list(range(max(0, d - 7), d))
            pv_method = "mean_recent_up_to_7_days"
            load_fc[d] = data.load_kwh[load_sources].mean(axis=0)
            pv_fc[d] = data.pv_kwh[pv_sources].mean(axis=0)
        load_dates = [data.dates[i].date().isoformat() for i in load_sources]
        pv_dates = [data.dates[i].date().isoformat() for i in pv_sources]
        audit_rows.append(
            {
                "decision_date": decision_date.date().isoformat(),
                "load_method": load_method,
                "load_history_dates": "|".join(load_dates),
                "pv_method": pv_method,
                "pv_history_dates": "|".join(pv_dates),
                "load_history_count": len(load_dates),
                "pv_history_count": len(pv_dates),
                "max_load_source_date": max(load_dates) if load_dates else "",
                "max_pv_source_date": max(pv_dates) if pv_dates else "",
                "all_source_dates_strictly_before_decision": all(
                    pd.Timestamp(x) < decision_date for x in load_dates + pv_dates
                ),
            }
        )
    return ForecastReplay(
        load_forecast_kwh=load_fc,
        pv_forecast_kwh=pv_fc,
        load_residual_kwh=data.load_kwh - load_fc,
        pv_residual_kwh=data.pv_kwh - pv_fc,
        source_audit=pd.DataFrame(audit_rows),
    )
