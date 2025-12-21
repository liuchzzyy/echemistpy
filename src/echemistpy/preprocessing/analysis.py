"""Common analysis functions for signal processing.

This module contains general-purpose analysis functions that can be used
across different techniques.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from scipy.integrate import simpson

from echemistpy.core.structures import Measurement, AnalysisResult


def find_peaks_in_measurement(
    measurement: Measurement, x_var: str, y_var: str, height: Optional[float] = None, distance: Optional[int] = None, prominence: Optional[float] = None, **kwargs
) -> AnalysisResult:
    """Find peaks in the measurement data.

    Args:
        measurement: The input Measurement object.
        x_var: The name of the variable to use for the x-axis.
        y_var: The name of the variable to use for the y-axis (signal).
        height: Required height of peaks.
        distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
        prominence: Required prominence of peaks.
        **kwargs: Additional keyword arguments passed to scipy.signal.find_peaks.

    Returns:
        An AnalysisResult object containing peak information.
    """
    x_data = measurement.data[x_var].values
    y_data = measurement.data[y_var].values

    peaks, properties = find_peaks(y_data, height=height, distance=distance, prominence=prominence, **kwargs)

    peak_x = x_data[peaks]
    peak_y = y_data[peaks]

    summary = {"n_peaks": len(peaks), "peak_indices": peaks.tolist(), "parameters": {"height": height, "distance": distance, "prominence": prominence, **kwargs}}

    # Create a table for peak data
    peaks_table = xr.Dataset(
        {
            "peak_index": (["peak"], peaks),
            "x": (["peak"], peak_x),
            "y": (["peak"], peak_y),
        },
        coords={"peak": np.arange(len(peaks))},
    )

    # Add properties to the table
    for key, value in properties.items():
        # Handle properties that might have different lengths or types
        try:
            peaks_table[key] = (["peak"], value)
        except Exception:
            pass

    return AnalysisResult(data=peaks_table)


def integrate_signal(measurement: Measurement, x_var: str, y_var: str, x_range: Optional[Tuple[float, float]] = None) -> float:
    """Integrate the signal using Simpson's rule.

    Args:
        measurement: The input Measurement object.
        x_var: The name of the variable to use for the x-axis.
        y_var: The name of the variable to use for the y-axis.
        x_range: Optional range (min, max) to integrate over.

    Returns:
        The integral value.
    """
    data = measurement.data

    if x_range:
        mask = (data[x_var] >= x_range[0]) & (data[x_var] <= x_range[1])
        subset = data.isel(row=mask)
        x = subset[x_var].values
        y = subset[y_var].values
    else:
        x = data[x_var].values
        y = data[y_var].values

    return simpson(y, x=x)
