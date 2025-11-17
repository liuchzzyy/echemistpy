"""Plotting utilities for data visualization."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def setup_plotting_style(style: str = "seaborn-v0_8-darkgrid") -> None:
    """Setup matplotlib plotting style.

    Parameters
    ----------
    style : str, optional
        Matplotlib style to use, by default 'seaborn-v0_8-darkgrid'
    """
    try:
        plt.style.use(style)
    except OSError:
        # Fallback to default if style not found
        plt.style.use("default")


def plot_line(
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a line plot.

    Parameters
    ----------
    x : np.ndarray
        X-axis data
    y : np.ndarray
        Y-axis data
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure
    xlabel : str, optional
        X-axis label, by default 'X'
    ylabel : str, optional
        Y-axis label, by default 'Y'
    title : str, optional
        Plot title, by default ''
    **kwargs : Any
        Additional arguments passed to ax.plot

    Returns
    -------
    Figure, Axes
        Matplotlib figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.plot(x, y, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_heatmap(
    data: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    ax: Axes | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "",
    cmap: str = "viridis",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a heatmap plot.

    Parameters
    ----------
    data : np.ndarray
        2D data array to plot
    x : np.ndarray, optional
        X-axis coordinates
    y : np.ndarray, optional
        Y-axis coordinates
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure
    xlabel : str, optional
        X-axis label, by default 'X'
    ylabel : str, optional
        Y-axis label, by default 'Y'
    title : str, optional
        Plot title, by default ''
    cmap : str, optional
        Colormap name, by default 'viridis'
    **kwargs : Any
        Additional arguments passed to ax.imshow or ax.pcolormesh

    Returns
    -------
    Figure, Axes
        Matplotlib figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    if x is not None and y is not None:
        im = ax.pcolormesh(x, y, data, cmap=cmap, **kwargs)
    else:
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower", **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.colorbar(im, ax=ax)

    return fig, ax


def plot_contour(
    data: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    ax: Axes | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "",
    levels: int = 10,
    cmap: str = "viridis",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a contour plot.

    Parameters
    ----------
    data : np.ndarray
        2D data array to plot
    x : np.ndarray, optional
        X-axis coordinates
    y : np.ndarray, optional
        Y-axis coordinates
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure
    xlabel : str, optional
        X-axis label, by default 'X'
    ylabel : str, optional
        Y-axis label, by default 'Y'
    title : str, optional
        Plot title, by default ''
    levels : int, optional
        Number of contour levels, by default 10
    cmap : str, optional
        Colormap name, by default 'viridis'
    **kwargs : Any
        Additional arguments passed to ax.contourf

    Returns
    -------
    Figure, Axes
        Matplotlib figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    if x is None:
        x = np.arange(data.shape[1])
    if y is None:
        y = np.arange(data.shape[0])

    cs = ax.contourf(x, y, data, levels=levels, cmap=cmap, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.colorbar(cs, ax=ax)

    return fig, ax
