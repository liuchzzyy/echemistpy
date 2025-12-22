"""Simple plugin registry for echemistpy io system.

This module provides a simple registry to manage loader and saver plugins
without external dependencies like pluggy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from echemistpy.io.structures import RawData, RawDataInfo


class IOPluginManager:
    """Simple registry for io plugins."""

    def __init__(self):
        """Initialize the plugin manager."""
        self._loaders: Dict[str, Any] = {}
        self._savers: Dict[str, Any] = {}

    def register_loader(self, extensions: list[str], loader_class: Any) -> None:
        """Register a loader class for specific extensions.

        Args:
            extensions: List of file extensions (e.g., ['mpt', 'mpr'])
            loader_class: The class or factory to handle these files
        """
        for ext in extensions:
            ext = ext.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            self._loaders[ext] = loader_class

    def register_saver(self, formats: list[str], saver_class: Any) -> None:
        """Register a saver class for specific formats.

        Args:
            formats: List of format names (e.g., ['csv', 'json'])
            saver_class: The class or factory to handle saving
        """
        for fmt in formats:
            self._savers[fmt.lower()] = saver_class

    def get_loader(self, extension: str) -> Optional[Any]:
        """Get the loader for a given extension."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return self._loaders.get(ext)

    def get_saver(self, fmt: str) -> Optional[Any]:
        """Get the saver for a given format."""
        return self._savers.get(fmt.lower())

    def list_supported_extensions(self) -> list[str]:
        """List all supported file extensions."""
        return list(self._loaders.keys())


# Global instance
_instance = None


def get_plugin_manager() -> IOPluginManager:
    """Get or create the global plugin manager instance."""
    global _instance
    if _instance is None:
        _instance = IOPluginManager()
    return _instance
        extension = (fmt or filepath.suffix.lstrip(".")).lower()

        if extension not in self._loaders:
            raise ValueError(
                f"No loader found for extension '{extension}'. "
                f"Supported formats: {', '.join(sorted(self._loaders.keys()))}"
            )

        loader = self._loaders[extension]
        return loader.load_file(filepath, **kwargs)

    def save_data(
        self,
        data: xr.Dataset,
        metadata: dict[str, Any],
        filepath: str | Path,
        fmt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Save data using the appropriate plugin.

        Args:
            data: xarray.Dataset to save
            metadata: Metadata dictionary
            filepath: Destination path
            fmt: Optional format override
            **kwargs: Additional arguments passed to the saver plugin

        Raises:
            ValueError: If no saver is found for the format
        """
        filepath = Path(filepath)
        extension = (fmt or filepath.suffix.lstrip(".")).lower()

        if extension not in self._savers:
            raise ValueError(
                f"No saver found for format '{extension}'. "
                f"Supported formats: {', '.join(sorted(self._savers.keys()))}"
            )

        saver = self._savers[extension]
        saver.save_data(data, metadata, filepath, fmt=fmt, **kwargs)

    def get_supported_loaders(self) -> dict[str, str]:
        """Get dictionary of supported loader extensions.

        Returns:
            Dictionary mapping extensions to loader names
        """
        return {ext: type(loader).__name__ for ext, loader in self._loaders.items()}

    def get_supported_savers(self) -> dict[str, str]:
        """Get dictionary of supported saver formats.

        Returns:
            Dictionary mapping formats to saver names
        """
        return {fmt: type(saver).__name__ for fmt, saver in self._savers.items()}


# Global plugin manager instance
_plugin_manager: Optional[IOPluginManager] = None


def get_plugin_manager() -> IOPluginManager:
    """Get the global plugin manager instance.

    Returns:
        Global IOPluginManager instance
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = IOPluginManager()
    return _plugin_manager


__all__ = [
    "IOPluginManager",
    "get_plugin_manager",
]
