"""Read only the Q2-authorized columns and convert kW to slot kWh."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from .config import DELTA_T_HOURS, N_SLOTS, ROOT


def _label(value: object) -> str:
    if isinstance(value, time):
        return value.strftime("%H:%M")
    return str(value).strip().strip("'").strip('"')


@dataclass(frozen=True)
class Q2InputData:
    dates: pd.DatetimeIndex
    time_labels: tuple[str, ...]
    price_yuan_per_kwh: np.ndarray
    load_kwh: np.ndarray
    pv_kwh: np.ndarray

    def day_index(self, day: str | pd.Timestamp) -> int:
        target = pd.Timestamp(day)
        positions = np.flatnonzero(self.dates == target)
        if len(positions) != 1:
            raise KeyError(f"Date not found exactly once: {target.date()}")
        return int(positions[0])


def load_inputs() -> Q2InputData:
    price_raw = pd.read_excel(ROOT / "data" / "raw" / "附件1.xlsx", sheet_name=0)
    if len(price_raw) != N_SLOTS:
        raise ValueError("Attachment 1 must contain exactly 144 price rows")
    price = pd.to_numeric(price_raw.iloc[:, 1], errors="raise").to_numpy(dtype=float)

    workbook = ROOT / "data" / "raw" / "附件2.xlsx"
    load_raw = pd.read_excel(workbook, sheet_name="小区负载")
    pv_raw = pd.read_excel(workbook, sheet_name="光伏发电实际功率")
    dates = pd.DatetimeIndex(pd.to_datetime(load_raw.iloc[:, 0]))
    pv_dates = pd.DatetimeIndex(pd.to_datetime(pv_raw.iloc[:, 0]))
    if not dates.equals(pv_dates):
        raise ValueError("Load and PV dates differ")
    load = load_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    pv = pv_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if load.shape != (365, N_SLOTS) or pv.shape != (365, N_SLOTS):
        raise ValueError(f"Unexpected Attachment-2 shapes: {load.shape}, {pv.shape}")
    if not np.isfinite(load).all() or not np.isfinite(pv).all() or (load < 0).any() or (pv < 0).any():
        raise ValueError("Attachment 2 contains invalid power data")
    labels_load = tuple(_label(v) for v in load_raw.columns[1:])
    labels_pv = tuple(_label(v) for v in pv_raw.columns[1:])
    if labels_load != labels_pv or len(labels_load) != N_SLOTS:
        raise ValueError("Load/PV time labels are not aligned")
    return Q2InputData(
        dates=dates,
        time_labels=labels_load,
        price_yuan_per_kwh=price,
        load_kwh=load * DELTA_T_HOURS,
        pv_kwh=pv * DELTA_T_HOURS,
    )
