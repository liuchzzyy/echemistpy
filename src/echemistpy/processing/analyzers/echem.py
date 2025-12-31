"""Electrochemical analysis helpers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import xarray as xr

from echemistpy.io.structures import RawData
from .registry import TechniqueAnalyzer


class CyclicVoltammetryAnalyzer(TechniqueAnalyzer):
    """Compute simple oxidation/reduction metrics for CV traces."""

    technique = "echem"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("potential", "current")

    def preprocess(self, raw_data: RawData) -> RawData:
        """Sort by potential and calculate baseline corrected current."""
        # Ensure we are working with a Dataset
        if raw_data.is_tree:
            ds = raw_data.data.dataset
            if ds is None:
                raise ValueError("DataTree has no root dataset for CV analysis.")
        else:
            ds = raw_data.data

        ds = ds.sortby("potential")

        # Find the main dimension (usually 'record' or 'row')
        dim = ds["current"].dims[0]

        baseline = float(ds["current"].mean())
        normalized = ds["current"].values - baseline

        # Update the data
        raw_data.data = ds.assign(baseline_corrected=(dim, normalized))
        return raw_data

    def compute(self, raw_data: RawData) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        """Calculate peak potentials and currents."""
        ds = raw_data.data
        if isinstance(ds, xr.DataTree):
            ds = ds.dataset
            if ds is None:
                raise ValueError("DataTree has no root dataset for CV analysis.")

        potential = ds["potential"].values
        corrected = ds["baseline_corrected"].values
        dim = ds["baseline_corrected"].dims[0]

        oxidation_idx = int(np.argmax(corrected))
        reduction_idx = int(np.argmin(corrected))

        # Calculate charge
        potential_step = np.gradient(potential)
        cumulative_charge = np.cumsum(corrected * potential_step)

        scale = np.max(np.abs(corrected)) or 1.0
        normalized = corrected / scale

        table = ds.assign(
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
