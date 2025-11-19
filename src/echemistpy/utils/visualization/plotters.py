"""Visualization helpers for echemistpy."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from echemistpy.io.structures import Measurement, AnalysisResult


def plot_measurement(
    measurement: Measurement,
    x_var: str,
    y_vars: str | list[str],
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """Plot data from a Measurement object.

    Args:
        measurement: The Measurement object to plot.
        x_var: The name of the variable to use for the x-axis.
        y_vars: The name(s) of the variable(s) to use for the y-axis.
        title: Optional title for the plot.
        ax: Optional matplotlib Axes to plot on.
        **kwargs: Additional keyword arguments passed to the plot function.

    Returns:
        A tuple containing the Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if isinstance(y_vars, str):
        y_vars = [y_vars]

    for y_var in y_vars:
        if y_var not in measurement.data:
            continue
            
        x_data = measurement.data[x_var]
        y_data = measurement.data[y_var]
        
        ax.plot(x_data, y_data, label=y_var, **kwargs)

    ax.set_xlabel(x_var)
    ax.set_ylabel(", ".join(y_vars))
    
    if title:
        ax.set_title(title)
    elif measurement.metadata.sample_name:
        ax.set_title(f"{measurement.metadata.sample_name} ({measurement.metadata.technique})")

    if len(y_vars) > 1:
        ax.legend()

    return fig, ax


def plot_comparison(
    measurement1: Measurement,
    measurement2: Measurement,
    x_var: str,
    y_var: str,
    labels: Tuple[str, str] = ("Original", "Processed"),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """Plot a comparison between two Measurement objects (e.g., before and after normalization).

    Args:
        measurement1: The first Measurement object.
        measurement2: The second Measurement object.
        x_var: The name of the variable to use for the x-axis.
        y_var: The name of the variable to use for the y-axis.
        labels: Labels for the two measurements.
        title: Optional title for the plot.
        ax: Optional matplotlib Axes to plot on.
        **kwargs: Additional keyword arguments passed to the plot function.

    Returns:
        A tuple containing the Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot first measurement
    if y_var in measurement1.data:
        ax.plot(measurement1.data[x_var], measurement1.data[y_var], label=labels[0], linestyle='-', **kwargs)

    # Plot second measurement
    if y_var in measurement2.data:
        ax.plot(measurement2.data[x_var], measurement2.data[y_var], label=labels[1], linestyle='--', **kwargs)

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    
    if title:
        ax.set_title(title)
    
    ax.legend()

    return fig, ax


def plot_peaks(
    measurement: Measurement,
    analysis_result: AnalysisResult,
    x_var: str,
    y_var: str,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """Plot measurement data with detected peaks highlighted.

    Args:
        measurement: The Measurement object.
        analysis_result: The AnalysisResult object containing peak data.
        x_var: The name of the variable to use for the x-axis.
        y_var: The name of the variable to use for the y-axis.
        title: Optional title for the plot.
        ax: Optional matplotlib Axes to plot on.
        **kwargs: Additional keyword arguments passed to the plot function.

    Returns:
        A tuple containing the Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot the signal
    if y_var in measurement.data:
        ax.plot(measurement.data[x_var], measurement.data[y_var], label="Signal", **kwargs)

    # Plot the peaks
    if "peaks" in analysis_result.tables:
        peaks_table = analysis_result.tables["peaks"]
        ax.plot(peaks_table["x"], peaks_table["y"], "rx", label="Peaks")

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    
    if title:
        ax.set_title(title)
    
    ax.legend()

    return fig, ax
