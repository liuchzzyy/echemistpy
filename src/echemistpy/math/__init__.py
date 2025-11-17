"""Reusable math helpers shared across electrochemistry workflows."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray | Iterable[float]


def numeric_gradient(values: ArrayLike, *, spacing: float | None = None) -> np.ndarray:
    """Return the numerical gradient of *values*.

    Parameters
    ----------
    values:
        Iterable of numeric samples whose gradient should be computed.
    spacing:
        Optional scalar spacing between samples.  When omitted, numpy infers an
        even spacing of ``1``.
    """

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    if spacing is None:
        return np.gradient(array)
    return np.gradient(array, spacing)


def trapezoidal_integral(y_values: ArrayLike, x_values: ArrayLike) -> float:
    """Return the trapezoidal integral of *y_values* with respect to *x_values*."""

    y_array = np.asarray(list(y_values), dtype=float)
    x_array = np.asarray(list(x_values), dtype=float)
    if y_array.shape != x_array.shape:
        msg = "x_values and y_values must share the same length for integration"
        raise ValueError(msg)
    return float(np.trapz(y_array, x_array))


__all__ = ["numeric_gradient", "trapezoidal_integral"]
