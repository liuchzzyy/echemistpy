"""Data processing utilities."""

from typing import Literal

import numpy as np
from scipy import signal

from echemistpy.core.exceptions import PreprocessingError


def normalize(
    data: np.ndarray,
    method: Literal["minmax", "zscore", "l2"] = "minmax",
) -> np.ndarray:
    """Normalize data using various methods.

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize
    method : {'minmax', 'zscore', 'l2'}, optional
        Normalization method, by default 'minmax'
        - 'minmax': Scale to [0, 1] range
        - 'zscore': Standardize to mean=0, std=1
        - 'l2': L2 normalization

    Returns
    -------
    np.ndarray
        Normalized data

    Raises
    ------
    PreprocessingError
        If normalization fails
    """
    try:
        if method == "minmax":
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if data_max == data_min:
                return np.zeros_like(data)
            return (data - data_min) / (data_max - data_min)
        if method == "zscore":
            mean = np.nanmean(data)
            std = np.nanstd(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - mean) / std
        if method == "l2":
            norm = np.linalg.norm(data)
            if norm == 0:
                return np.zeros_like(data)
            return data / norm
        msg = f"Unknown normalization method: {method}"
        raise PreprocessingError(msg)
    except Exception as e:
        msg = f"Normalization failed: {e}"
        raise PreprocessingError(msg) from e


def smooth(
    data: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
    method: Literal["savgol", "moving_average"] = "savgol",
) -> np.ndarray:
    """Smooth data using various methods.

    Parameters
    ----------
    data : np.ndarray
        Input data to smooth
    window_length : int, optional
        Length of the smoothing window, by default 5
    polyorder : int, optional
        Order of polynomial for Savitzky-Golay filter, by default 2
    method : {'savgol', 'moving_average'}, optional
        Smoothing method, by default 'savgol'

    Returns
    -------
    np.ndarray
        Smoothed data

    Raises
    ------
    PreprocessingError
        If smoothing fails
    """
    try:
        if method == "savgol":
            if window_length % 2 == 0:
                window_length += 1
            return signal.savgol_filter(data, window_length, polyorder)
        if method == "moving_average":
            kernel = np.ones(window_length) / window_length
            return np.convolve(data, kernel, mode="same")
        msg = f"Unknown smoothing method: {method}"
        raise PreprocessingError(msg)
    except Exception as e:
        msg = f"Smoothing failed: {e}"
        raise PreprocessingError(msg) from e


def baseline_correction(
    data: np.ndarray,
    method: Literal["polynomial", "linear"] = "polynomial",
    degree: int = 2,
) -> np.ndarray:
    """Remove baseline from data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    method : {'polynomial', 'linear'}, optional
        Baseline correction method, by default 'polynomial'
    degree : int, optional
        Degree of polynomial for fitting, by default 2

    Returns
    -------
    np.ndarray
        Baseline-corrected data

    Raises
    ------
    PreprocessingError
        If baseline correction fails
    """
    try:
        x = np.arange(len(data))
        if method == "polynomial":
            coeffs = np.polyfit(x, data, degree)
            baseline = np.polyval(coeffs, x)
        elif method == "linear":
            coeffs = np.polyfit(x, data, 1)
            baseline = np.polyval(coeffs, x)
        else:
            msg = f"Unknown baseline correction method: {method}"
            raise PreprocessingError(msg)
        return data - baseline
    except Exception as e:
        msg = f"Baseline correction failed: {e}"
        raise PreprocessingError(msg) from e
