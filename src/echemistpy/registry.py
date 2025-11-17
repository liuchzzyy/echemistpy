"""Compatibility shim for legacy registry imports.

The canonical registry implementation now lives in
:mod:`echemistpy.io.reading.registry`.  This module simply re-exports those
helpers so that older code importing :mod:`echemistpy.registry` continues to
function without modification.
"""

from .io.reading.registry import *  # noqa: F401,F403
