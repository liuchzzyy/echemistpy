"""Data normalization module for echemistpy."""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from echemistpy.io.structures import Measurement


def normalize_min_max(measurement: Measurement, variable: str, feature_range: tuple[float, float] = (0, 1)) -> Measurement:
    """Normalize a variable in the measurement using Min-Max scaling.

    Args:
        measurement: The input Measurement object.
        variable: The name of the variable to normalize.
        feature_range: The desired range of transformed data.

    Returns:
        A new Measurement object with the normalized variable.
    """
    new_measurement = measurement.copy()
    data = new_measurement.data[variable]
    
    min_val = data.min()
    max_val = data.max()
    
    if min_val == max_val:
        # Avoid division by zero if data is constant
        normalized_data = xr.zeros_like(data)
    else:
        std = (data - min_val) / (max_val - min_val)
        normalized_data = std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    new_measurement.data[variable] = normalized_data
    new_measurement.metadata.extras[f"normalization_{variable}"] = f"MinMax{feature_range}"
    
    return new_measurement


def normalize_z_score(measurement: Measurement, variable: str) -> Measurement:
    """Normalize a variable in the measurement using Z-score standardization.

    Args:
        measurement: The input Measurement object.
        variable: The name of the variable to normalize.

    Returns:
        A new Measurement object with the normalized variable.
    """
    new_measurement = measurement.copy()
    data = new_measurement.data[variable]
    
    mean_val = data.mean()
    std_val = data.std()
    
    if std_val == 0:
        normalized_data = xr.zeros_like(data)
    else:
        normalized_data = (data - mean_val) / std_val
    
    new_measurement.data[variable] = normalized_data
    new_measurement.metadata.extras[f"normalization_{variable}"] = "ZScore"
    
    return new_measurement
