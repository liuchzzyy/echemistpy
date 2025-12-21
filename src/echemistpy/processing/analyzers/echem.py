"""Electrochemical analysis helpers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import xarray as xr

from .base import TechniqueAnalyzer


class CyclicVoltammetryAnalyzer(TechniqueAnalyzer):
    """Compute simple oxidation/reduction metrics for CV traces."""

    technique = "echem"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("potential", "current")

    def preprocess(self, measurement):
        data = measurement.data.sortby("potential")
        dim = data["current"].dims[0]
        baseline = float(data["current"].values.mean())
        normalized = data["current"].values - baseline
        measurement.data = data.assign(baseline_corrected=(dim, normalized))
        return measurement

    def compute(self, measurement) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        data = measurement.data
        potential = data["potential"].values
        corrected = data["baseline_corrected"].values
        dim = data["baseline_corrected"].dims[0]

        oxidation_idx = int(np.argmax(corrected))
        reduction_idx = int(np.argmin(corrected))
        potential_step = np.gradient(potential)
        cumulative_charge = np.cumsum(corrected * potential_step)
        scale = np.max(np.abs(corrected)) or 1.0
        normalized = corrected / scale

        table = data.assign(
            normalized_current=(dim, normalized),
            cumulative_charge=(dim, cumulative_charge),
        )

        summary: Dict[str, Any] = {
            "oxidation_peak_potential": float(potential[oxidation_idx]),
            "oxidation_peak_current": float(corrected[oxidation_idx]),
            "reduction_peak_potential": float(potential[reduction_idx]),
            "reduction_peak_current": float(corrected[reduction_idx]),
            "net_charge": float(cumulative_charge[-1]),
        }
        return summary, {"cv_trace": table}
