"""Simple plugin registry for echemistpy io system.

This module provides a simple registry to manage loader and saver plugins
without external dependencies like pluggy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


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

    def get_supported_loaders(self) -> dict[str, str]:
        """Get dictionary of supported loader extensions.

        Returns:
            Dictionary mapping extensions to loader names
        """
        return {ext: loader.__name__ if hasattr(loader, "__name__") else str(loader) for ext, loader in self._loaders.items()}

    def get_supported_savers(self) -> dict[str, str]:
        """Get dictionary of supported saver formats.

        Returns:
            Dictionary mapping formats to saver names
        """
        return {fmt: saver.__name__ if hasattr(saver, "__name__") else str(saver) for fmt, saver in self._savers.items()}


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
