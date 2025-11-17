"""Plotting helpers and default visualization styles."""

from __future__ import annotations

from typing import Mapping

DEFAULT_STYLE: Mapping[str, str | float] = {
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
}


def apply_default_style() -> Mapping[str, str | float]:
    """Apply :data:`DEFAULT_STYLE` to matplotlib and return the active rcParams."""

    import matplotlib as mpl

    mpl.rcParams.update(DEFAULT_STYLE)
    return mpl.rcParams


__all__ = ["DEFAULT_STYLE", "apply_default_style"]
