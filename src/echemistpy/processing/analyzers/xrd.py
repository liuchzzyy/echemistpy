"""X-ray diffraction utilities."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import xarray as xr

from .base import TechniqueAnalyzer


class XRDPowderAnalyzer(TechniqueAnalyzer):
    """Compute simple peak statistics for powder XRD traces."""

    technique = "xrd"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("2theta", "intensity")

    def preprocess(self, summary_data):
        summary_data.data = summary_data.data.sortby("2theta")
        summary_data.data["intensity"] = summary_data.data["intensity"].clip(min=0)
        return summary_data

    def compute(self, summary_data) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        data = summary_data.data
        dim = data["intensity"].dims[0]
        theta = data["2theta"].values
        intensity = data["intensity"].values
        max_idx = int(np.argmax(intensity))
        integrated_intensity = float(np.trapz(intensity, theta))
        background = float(np.quantile(intensity, 0.05))
        normalized = data.assign(norm_intensity=(dim, intensity - background))

        summary: Dict[str, Any] = {
            "max_position": float(theta[max_idx]),
            "max_intensity": float(intensity[max_idx]),
            "integrated_intensity": integrated_intensity,
            "background": background,
        }
        tables = {"normalized_pattern": normalized}
        return summary, tables
