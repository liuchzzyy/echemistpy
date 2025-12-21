"""LANHE XLSX file loader plugin using pluggy interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from traitlets import HasTraits, Unicode, validate

from echemistpy.io.plugin_specs import hookimpl
from echemistpy.io.structures import RawData, RawDataInfo

# Import the XLSXDataReader from the existing module
from .LanheXLSXReader import XLSXDataReader

logger = logging.getLogger(__name__)


class LanheLoaderPlugin(HasTraits):
    """LANHE XLSX file loader plugin with traitlets parameter control.
    
    This plugin wraps the XLSXDataReader to provide a pluggy-compatible
    interface for loading LANHE Excel files.
    """

    # Traitlets properties for configuration
    encoding = Unicode(default_value="utf-8", help="File encoding").tag(config=True)
    
    def __init__(self, **kwargs):
        """Initialize the plugin with optional configuration."""
        super().__init__(**kwargs)

    @hookimpl
    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return ["xlsx", "xls"]

    @hookimpl(tryfirst=True)
    def can_load(self, filepath: Path) -> bool:
        """Check if this loader can handle the given file."""
        # Additional check: LANHE files typically have specific sheet names
        # For now, just check extension
        return filepath.suffix.lower() in [".xlsx", ".xls"]

    @hookimpl
    def load_file(
        self,
        filepath: Path,
        **kwargs: Any,
    ) -> tuple[RawData, RawDataInfo]:
        """Load a LANHE XLSX file and return RawData and RawDataInfo.

        Args:
            filepath: Path to the LANHE XLSX file
            **kwargs: Additional parameters (currently unused)

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        # Use the existing XLSXDataReader
        metadata, data_dict = XLSXDataReader.load(filepath)
        
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


__all__ = ["LanheLoaderPlugin"]
