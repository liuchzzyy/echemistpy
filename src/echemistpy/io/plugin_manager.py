"""Plugin manager for echemistpy io system using pluggy.

This module provides the plugin manager that discovers and manages
loader and saver plugins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pluggy
import xarray as xr

from echemistpy.io.plugin_specs import LoaderSpec, SaverSpec
from echemistpy.io.structures import RawData, RawDataInfo


class IOPluginManager:
    """Manager for io plugins using pluggy."""

    def __init__(self):
        """Initialize the plugin manager."""
        self.pm = pluggy.PluginManager("echemistpy_io")
        self.pm.add_hookspecs(LoaderSpec)
        self.pm.add_hookspecs(SaverSpec)
        self._loaders: dict[str, Any] = {}
        self._savers: dict[str, Any] = {}

    def register_plugin(self, plugin: Any, name: Optional[str] = None) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance or module
            name: Optional plugin name
        """
        # Check if plugin is already registered by checking existing plugins
        if name:
            existing_plugins = dict(self.pm.list_name_plugin())
            if name in existing_plugins:
                # Unregister the old plugin with this name
                old_plugin = existing_plugins[name]
                if old_plugin is not None:
                    self.pm.unregister(old_plugin)
        
        self.pm.register(plugin, name=name)
        self._refresh_mappings()

    def unregister_plugin(self, plugin: Any) -> None:
        """Unregister a plugin.

        Args:
            plugin: Plugin instance or module to unregister
        """
        self.pm.unregister(plugin)
        self._refresh_mappings()

    def _refresh_mappings(self) -> None:
        """Refresh the extension to loader and format to saver mappings."""
        self._loaders.clear()
        self._savers.clear()

        # Build loader mappings
        for plugin in self.pm.get_plugins():
            if hasattr(plugin, "get_supported_extensions"):
                try:
                    extensions = plugin.get_supported_extensions()
                    for ext in extensions:
                        self._loaders[ext.lower()] = plugin
                except Exception:
                    pass

            if hasattr(plugin, "get_supported_formats"):
                try:
                    formats = plugin.get_supported_formats()
                    for fmt in formats:
                        self._savers[fmt.lower()] = plugin
                except Exception:
                    pass

    def load_file(
        self,
        filepath: str | Path,
        fmt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[RawData, RawDataInfo]:
        """Load a file using the appropriate plugin.

        Args:
            filepath: Path to the file to load
            fmt: Optional format override (file extension without dot)
            **kwargs: Additional arguments passed to the loader plugin

        Returns:
            Tuple of (RawData, RawDataInfo)

        Raises:
            ValueError: If no loader is found for the file format
        """
        filepath = Path(filepath)
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
