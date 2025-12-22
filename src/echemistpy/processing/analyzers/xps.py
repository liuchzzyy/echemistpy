"""Photoelectron spectroscopy helpers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import xarray as xr
from scipy.signal import savgol_filter

from .base import TechniqueAnalyzer


class XPSAnalyzer(TechniqueAnalyzer):
    """Basic peak metrics and background estimation for XPS spectra."""

    technique = "xps"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("binding_energy", "counts")

    def preprocess(self, raw_data):
        raw_data.data = raw_data.data.sortby("binding_energy", ascending=False)
        counts = raw_data.data["counts"].values
        smoothed = savgol_filter(counts, window_length=9, polyorder=2)
        dim = raw_data.data["counts"].dims[0]
        raw_data.data["smoothed"] = (dim, smoothed)
        return raw_data

    def compute(self, raw_data) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        data = raw_data.data
        energy = data["binding_energy"].values
        smoothed = data["smoothed"].values
        derivative = np.gradient(smoothed, energy)
        dim = data["binding_energy"].dims[0]
        enriched = data.assign(derivative=(dim, derivative))
        max_idx = int(np.argmax(smoothed))
        summary = {
            "peak_energy": float(energy[max_idx]),
            "peak_intensity": float(smoothed[max_idx]),
            "counts_mean": float(data["counts"].values.mean()),
        }
        return summary, {"spectrum": enriched}
