"""Joint historical load/PV residual scenarios without resampling or padding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Q2InputData
from .forecast import ForecastReplay


@dataclass(frozen=True)
class JointScenarios:
    source_indices: tuple[int, ...]
    source_dates: tuple[str, ...]
    weights: np.ndarray
    load_kwh: np.ndarray
    pv_kwh: np.ndarray

    @property
    def count(self) -> int:
        return len(self.source_indices)


def build_joint_scenarios(
    data: Q2InputData, replay: ForecastReplay, decision_index: int, scenario_cap: int
) -> JointScenarios:
    if decision_index <= 0:
        raise ValueError("At least one completed historical day is required for scenarios")
    if scenario_cap <= 0:
        raise ValueError("scenario_cap must be positive")
    start = max(0, decision_index - scenario_cap)
    sources = tuple(range(start, decision_index))
    load = np.maximum(
        replay.load_forecast_kwh[decision_index][None, :]
        + replay.load_residual_kwh[list(sources)],
        0.0,
    )
    pv = np.maximum(
        replay.pv_forecast_kwh[decision_index][None, :]
        + replay.pv_residual_kwh[list(sources)],
        0.0,
    )
    return JointScenarios(
        source_indices=sources,
        source_dates=tuple(data.dates[i].date().isoformat() for i in sources),
        weights=np.full(len(sources), 1.0 / len(sources), dtype=float),
        load_kwh=load,
        pv_kwh=pv,
    )
