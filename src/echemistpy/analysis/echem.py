"""Electrochemistry-specific analytics."""

from __future__ import annotations

import numpy as np

from ..io.reorganization import MeasurementRecord
from ..math import numeric_gradient, trapezoidal_integral


def _require_variables(record: MeasurementRecord, *variables: str) -> None:
    missing = set(variables) - set(record.data.data_vars)
    if missing:
        raise KeyError(f"record missing required variables: {sorted(missing)}")


def peak_current(record: MeasurementRecord) -> float:
    """Return the magnitude of the largest absolute current."""

    _require_variables(record, "current")
    currents = record.data["current"].values.astype(float)
    if currents.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(currents)))


def anodic_cathodic_split(record: MeasurementRecord) -> tuple[MeasurementRecord, MeasurementRecord]:
    """Split a voltammogram into anodic/cathodic sweeps based on potential derivative."""

    _require_variables(record, "potential")
    dim = next(iter(record.data["potential"].dims), "index")
    potential = record.data["potential"].values.astype(float)
    derivative = numeric_gradient(potential)
    anodic_selector = np.asarray(derivative >= 0)
    cathodic_selector = np.asarray(derivative < 0)
    anodic = record.data.isel({dim: np.flatnonzero(anodic_selector)})
    cathodic = record.data.isel({dim: np.flatnonzero(cathodic_selector)})
    return (
        MeasurementRecord(metadata=record.metadata, data=anodic, annotations=record.annotations),
        MeasurementRecord(metadata=record.metadata, data=cathodic, annotations=record.annotations),
    )


def coulombic_efficiency(record: MeasurementRecord) -> float:
    """Estimate coulombic efficiency by integrating current over time if present."""

    _require_variables(record, "current", "time")
    current = record.data["current"].values.astype(float)
    time = record.data["time"].values.astype(float)
    if current.size == 0 or time.size == 0:
        return float("nan")
    charge = trapezoidal_integral(current, time)
    discharge_mask = current < 0
    discharge = trapezoidal_integral(-current[discharge_mask], time[discharge_mask]) if discharge_mask.any() else 0.0
    if discharge == 0:
        return 1.0
    return float(charge / discharge)


__all__ = ["peak_current", "anodic_cathodic_split", "coulombic_efficiency"]
