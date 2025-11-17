"""Mathematical helpers for spectrum conditioning."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    """Compute a centered moving average."""

    values = np.asarray(tuple(values), dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    if window > values.size:
        raise ValueError("window cannot exceed input length")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


__all__ = ["moving_average"]
