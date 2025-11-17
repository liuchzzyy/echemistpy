"""Data validation utilities."""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.core.exceptions import ValidationError


def validate_data(
    data: Any,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> bool:
    """Validate data for common issues.

    Parameters
    ----------
    data : Any
        Data to validate (numpy array, pandas DataFrame, xarray Dataset)
    allow_nan : bool, optional
        Whether to allow NaN values, by default False
    allow_inf : bool, optional
        Whether to allow infinite values, by default False

    Returns
    -------
    bool
        True if data is valid

    Raises
    ------
    ValidationError
        If data validation fails
    """
    try:
        # Convert to numpy array for validation
        if isinstance(data, pd.DataFrame) or isinstance(data, (xr.Dataset, xr.DataArray)):
            arr = data.values
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise ValidationError(msg)

        # Check for NaN values
        if not allow_nan and np.any(np.isnan(arr)):
            msg = "Data contains NaN values"
            raise ValidationError(msg)

        # Check for infinite values
        if not allow_inf and np.any(np.isinf(arr)):
            msg = "Data contains infinite values"
            raise ValidationError(msg)

        return True
    except ValidationError:
        raise
    except Exception as e:
        msg = f"Data validation failed: {e}"
        raise ValidationError(msg) from e


def check_dimensions(
    data: Any,
    expected_dims: int | tuple[int, ...] | None = None,
    expected_shape: tuple[int, ...] | None = None,
) -> bool:
    """Check if data has expected dimensions or shape.

    Parameters
    ----------
    data : Any
        Data to check
    expected_dims : int or tuple of int, optional
        Expected number of dimensions
    expected_shape : tuple of int, optional
        Expected shape

    Returns
    -------
    bool
        True if dimensions match expectations

    Raises
    ------
    ValidationError
        If dimensions don't match expectations
    """
    try:
        # Get shape
        if isinstance(data, (pd.DataFrame, pd.Series)) or isinstance(data, (xr.Dataset, xr.DataArray)) or isinstance(data, np.ndarray):
            shape = data.shape
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise ValidationError(msg)

        # Check dimensions
        if expected_dims is not None:
            if isinstance(expected_dims, int):
                if len(shape) != expected_dims:
                    msg = f"Expected {expected_dims} dimensions, got {len(shape)}"
                    raise ValidationError(msg)
            elif len(shape) not in expected_dims:
                msg = f"Expected dimensions {expected_dims}, got {len(shape)}"
                raise ValidationError(msg)

        # Check shape
        if expected_shape is not None:
            if shape != expected_shape:
                msg = f"Expected shape {expected_shape}, got {shape}"
                raise ValidationError(msg)

        return True
    except ValidationError:
        raise
    except Exception as e:
        msg = f"Dimension check failed: {e}"
        raise ValidationError(msg) from e
