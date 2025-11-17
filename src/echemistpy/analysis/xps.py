"""XPS-specific analysis helpers."""

from __future__ import annotations

import numpy as np

from ..io.reorganization import MeasurementRecord


def peak_characteristics(record: MeasurementRecord) -> dict[str, float]:
    """Return peak binding energy/intensity metrics for an XPS record."""

    required = {"binding_energy", "intensity"}
    missing = required - set(record.data.data_vars)
    if missing:
        raise KeyError(f"XPS record missing required variables: {sorted(missing)}")
    intensities = record.data["intensity"].values.astype(float)
    binding = record.data["binding_energy"].values.astype(float)
    if intensities.size == 0 or np.isnan(intensities).all():
        return {"peak_binding_energy": float("nan"), "peak_intensity": float("nan"), "mean_intensity": float("nan")}
    idx = int(np.nanargmax(intensities))
    return {
        "peak_binding_energy": float(binding[idx]),
        "peak_intensity": float(intensities[idx]),
        "mean_intensity": float(np.nanmean(intensities)),
    }


__all__ = ["peak_characteristics"]
