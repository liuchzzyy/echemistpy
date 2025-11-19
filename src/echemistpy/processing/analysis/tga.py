"""Thermogravimetric analysis helpers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import xarray as xr

from .base import TechniqueAnalyzer


class TGAAnalyzer(TechniqueAnalyzer):
    """Determine weight loss steps for TGA profiles."""

    technique = "tga"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("temperature", "mass")

    def preprocess(self, measurement):
        measurement.data = measurement.data.sortby("temperature")
        dim = measurement.data["mass"].dims[0]
        mass_fraction = measurement.data["mass"].values / measurement.data["mass"].values.max()
        measurement.data["mass_fraction"] = (dim, mass_fraction)
        return measurement

    def compute(self, measurement) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        data = measurement.data
        temperature = data["temperature"].values
        mass_fraction = data["mass_fraction"].values
        derivative = np.gradient(mass_fraction, temperature)
        dim = data["temperature"].dims[0]
        enriched = data.assign(dm_dt=(dim, derivative))
        min_idx = int(np.argmin(derivative))
        summary = {
            "total_mass_loss": float(1 - mass_fraction[-1]),
            "max_loss_rate_temp": float(temperature[min_idx]),
            "max_loss_rate": float(derivative[min_idx]),
        }
        return summary, {"tga_profile": enriched}
