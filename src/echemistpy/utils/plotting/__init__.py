"""Plotting hooks for future visualization features."""

from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr


def plot_dataset(dataset: xr.Dataset, *, x: str, y: str, ax: plt.Axes | None = None) -> plt.Axes:
    """Quick-look line plot for a dataset variable."""

    if x not in dataset.variables or y not in dataset.variables:
        raise ValueError(f"Both '{x}' and '{y}' must exist in the dataset.")
    axis = ax or plt.gca()
    axis.plot(dataset[x].values, dataset[y].values)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    return axis


__all__ = ["plot_dataset"]
