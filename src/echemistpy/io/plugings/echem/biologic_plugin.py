"""BioLogic MPT/MPR file loader plugin using pluggy interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from traitlets import HasTraits, Unicode

from echemistpy.io.plugin_specs import hookimpl
from echemistpy.io.structures import RawData, RawDataInfo

# Import the BiologicDataReader from the existing module
from .BiologicMPTReader import BiologicDataReader

logger = logging.getLogger(__name__)


class BiologicLoaderPlugin(HasTraits):
    """BioLogic file loader plugin with traitlets parameter control.

    This plugin wraps the BiologicDataReader to provide a pluggy-compatible
    interface for loading BioLogic MPT/MPR files.
    """

    # Traitlets properties for configuration
    encoding = Unicode(default_value="latin1", help="File encoding").tag(config=True)

    def __init__(self, **kwargs):
        """Initialize the plugin with optional configuration."""
        super().__init__(**kwargs)

    @hookimpl
    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return ["mpt", "mpr"]

    @hookimpl(tryfirst=True)
    def can_load(self, filepath: Path) -> bool:
        """Check if this loader can handle the given file."""
        return filepath.suffix.lower() in {".mpt", ".mpr"}

    @hookimpl
    def load_file(
        self,
        filepath: Path,
        **kwargs: Any,
    ) -> tuple[RawData, RawDataInfo]:
        """Load a BioLogic file and return RawData and RawDataInfo.

        Args:
            filepath: Path to the BioLogic file
            **kwargs: Additional parameters (currently unused)

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        # Use the existing BiologicDataReader
        metadata, data_dict = BiologicDataReader.load(filepath)

        # Convert data_dict to xarray.Dataset
        dataset = self._dict_to_dataset(data_dict)

        # Create RawData and RawDataInfo
        raw_data = RawData(data=dataset)
        raw_data_info = RawDataInfo(meta=metadata)

        return raw_data, raw_data_info

    @staticmethod
    def _dict_to_dataset(data_dict: dict[str, Any]) -> xr.Dataset:
        """Convert data dictionary to xarray.Dataset.

        Args:
            data_dict: Dictionary with column names as keys and lists as values

        Returns:
            xarray.Dataset with 'row' dimension
        """
        # Remove metadata from data_dict
        data_dict = {k: v for k, v in data_dict.items() if k != "_metadata"}

        if not data_dict:
            return xr.Dataset()

        # Get the length of data (assume all columns have same length)
        first_key = next(iter(data_dict.keys()))
        n_rows = len(data_dict[first_key])

        # Create data variables
        data_vars = {}
        for name, values in data_dict.items():
            # Convert to numpy array if it's not already
            if not isinstance(values, np.ndarray):
                values = np.array(values)
            data_vars[name] = ("row", values)

        # Create dataset with row coordinate
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={"row": np.arange(n_rows)}
        )

        return dataset


__all__ = ["BiologicLoaderPlugin"]
