"""Simple plugin registry for echemistpy io system.

This module provides a simple registry to manage loader and saver plugins
without external dependencies like pluggy.
"""

from __future__ import annotations

from typing import Any, Optional

from traitlets import Bool, Dict, HasTraits


class IOPluginManager(HasTraits):
    """Simple registry for io plugins using traitlets for variable management."""

    loaders = Dict(help="Dictionary mapping file extensions to loader classes")
    savers = Dict(help="Dictionary mapping format names to saver classes")
    initialized = Bool(False, help="Whether default plugins have been initialized")

    def register_loader(self, extensions: list[str], loader_class: Any) -> None:
        """Register a loader class for specific extensions.

        Args:
            extensions: List of file extensions (e.g., ['mpt', 'mpr'])
            loader_class: The class or factory to handle these files
        """
        for ext in extensions:
            ext_clean = ext.lower()
            if not ext_clean.startswith("."):
                ext_clean = f".{ext_clean}"
            self.loaders[ext_clean] = loader_class

    def register_saver(self, formats: list[str], saver_class: Any) -> None:
        """Register a saver class for specific formats.

        Args:
            formats: List of format names (e.g., ['csv', 'json'])
            saver_class: The class or factory to handle saving
        """
        for fmt in formats:
            self.savers[fmt.lower()] = saver_class

    def get_loader(self, extension: str) -> Optional[Any]:
        """Get the loader for a given extension."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return self.loaders.get(ext)

    def get_saver(self, fmt: str) -> Optional[Any]:
        """Get the saver for a given format."""
        return self.savers.get(fmt.lower())

    def list_supported_extensions(self) -> list[str]:
        """List all supported file extensions."""
        return list(self.loaders.keys())

    def get_supported_loaders(self) -> dict[str, str]:
        """Get dictionary of supported loader extensions.

        Returns:
            Dictionary mapping extensions to loader names
        """
        return {ext: loader.__name__ if hasattr(loader, "__name__") else str(loader) for ext, loader in self.loaders.items()}

    def get_supported_savers(self) -> dict[str, str]:
        """Get dictionary of supported saver formats.

        Returns:
            Dictionary mapping formats to saver names
        """
        return {fmt: saver.__name__ if hasattr(saver, "__name__") else str(saver) for fmt, saver in self.savers.items()}


def get_plugin_manager() -> IOPluginManager:
    """Get the global plugin manager instance.

    Returns:
        Global IOPluginManager instance
    """
    if not hasattr(get_plugin_manager, "_instance"):
        get_plugin_manager._instance = IOPluginManager()  # type: ignore
    return get_plugin_manager._instance  # type: ignore


__all__ = [
    "IOPluginManager",
    "get_plugin_manager",
]
