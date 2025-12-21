"""Plugin specifications for echemistpy io system using pluggy.

This module defines the plugin hooks and specifications for loading and saving
different data file formats in echemistpy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pluggy
import xarray as xr

from echemistpy.io.structures import RawData, RawDataInfo

# Plugin namespace
hookspec = pluggy.HookspecMarker("echemistpy_io")
hookimpl = pluggy.HookimplMarker("echemistpy_io")


# ============================================================================
# Loader Plugin Specification
# ============================================================================


class LoaderSpec:
    """Specification for data loader plugins."""

    @hookspec
    def get_supported_extensions(self) -> list[str]:
        """Return list of file extensions supported by this loader.

        Returns:
            List of extensions without dots (e.g., ['mpt', 'mpr'])
        """

    @hookspec
    def load_file(
        self,
        filepath: Path,
        **kwargs: Any,
    ) -> tuple[RawData, RawDataInfo]:
        """Load a data file and return RawData and RawDataInfo.

        Args:
            filepath: Path to the file to load
            **kwargs: Additional loader-specific parameters

        Returns:
            Tuple of (RawData, RawDataInfo) containing the loaded data and metadata
        """

    @hookspec(firstresult=True)
    def can_load(self, filepath: Path) -> bool:
        """Check if this loader can handle the given file.

        Args:
            filepath: Path to the file to check

        Returns:
            True if this loader can handle the file, False otherwise
        """


# ============================================================================
# Saver Plugin Specification
# ============================================================================


class SaverSpec:
    """Specification for data saver plugins."""

    @hookspec
    def get_supported_formats(self) -> list[str]:
        """Return list of output formats supported by this saver.

        Returns:
            List of format names (e.g., ['csv', 'json', 'netcdf'])
        """

    @hookspec
    def save_data(
        self,
        data: xr.Dataset,
        metadata: dict[str, Any],
        filepath: Path,
        fmt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Save data to file in the specified format.

        Args:
            data: xarray.Dataset to save
            metadata: Metadata dictionary to include
            filepath: Destination path
            fmt: Optional format override
            **kwargs: Additional saver-specific parameters
        """


__all__ = [
    "LoaderSpec",
    "SaverSpec",
    "hookimpl",
    "hookspec",
]
