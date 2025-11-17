"""Helpers that load built-in plugins."""

from __future__ import annotations


def load_builtin_readers() -> None:
    """Import modules that register default readers."""

    from . import reorganization as _reorganization

    _reorganization.register_builtin_readers()


def initialize_analysis_plugins() -> None:
    """Import analysis packages so that analyzers register themselves."""

    from ..analysis import xps as _xps_analysis  # noqa: F401  # ensures module side effects run


__all__ = ["load_builtin_readers", "initialize_analysis_plugins"]
