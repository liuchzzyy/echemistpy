"""Simple plugin registry for echemistpy io system.

This module provides a simple registry to manage loader and saver plugins
without external dependencies like pluggy.
"""

from __future__ import annotations

from typing import Any, Optional

from traitlets import Bool, Dict, HasTraits


class IOPluginManager(HasTraits):
    """Simple registry for io plugins using traitlets for variable management."""

    _instance = None

    loaders = Dict(help="Dictionary mapping file extensions to a list of loader classes")
    savers = Dict(help="Dictionary mapping format names to saver classes")
    initialized = Bool(False, help="Whether default plugins have been initialized")

    @classmethod
    def get_instance(cls) -> IOPluginManager:
        """Get the global plugin manager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

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

            # Create a copy of the dict to ensure traitlets detects the change
            current_loaders = dict(self.loaders)
            if ext_clean not in current_loaders:
                current_loaders[ext_clean] = []

            # Avoid duplicate registration
            if loader_class not in current_loaders[ext_clean]:
                new_list = list(current_loaders[ext_clean])
                new_list.append(loader_class)
                current_loaders[ext_clean] = new_list
                self.loaders = current_loaders

    def register_saver(self, formats: list[str], saver_class: Any) -> None:
        """Register a saver class for specific formats.

        Args:
            formats: List of format names (e.g., ['csv', 'json'])
            saver_class: The class or factory to handle saving
        """
        current_savers = dict(self.savers)
        for fmt in formats:
            current_savers[fmt.lower()] = saver_class
        self.savers = current_savers

    def get_loader(self, extension: str, instrument: Optional[str] = None) -> Optional[Any]:
        """Get the loader for a given extension, optionally filtered by instrument.

        If multiple loaders exist for an extension and no instrument is provided,
        the first one registered is returned.
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        loaders = self.loaders.get(ext, [])
        if not loaders:
            return None

        if instrument:
            # Try to find a loader that matches the instrument name
            for loader in loaders:
                # Check class attribute 'instrument'
                loader_inst = getattr(loader, "instrument", None)
                inst_name = ""
                if hasattr(loader_inst, "default_value"):
                    inst_name = str(loader_inst.default_value)
                elif loader_inst is not None:
                    inst_name = str(loader_inst)

                if inst_name and inst_name.lower() == instrument.lower():
                    return loader

                # Also check if the class name contains the instrument name as a fallback
                if instrument.lower() in loader.__name__.lower():
                    return loader

            # If instrument was specified but no match found, return None
            return None

        # Default to the first one if no instrument provided
        return loaders[0]

    def get_saver(self, fmt: str) -> Optional[Any]:
        """Get the saver for a given format."""
        return self.savers.get(fmt.lower())

    def list_supported_extensions(self) -> list[str]:
        """List all supported file extensions."""
        return list(self.loaders.keys())

    def get_supported_loaders(self) -> dict[str, list[str]]:
        """Get dictionary of supported loader extensions.

        Returns:
            Dictionary mapping extensions to a list of loader names
        """
        return {ext: [loader.__name__ if hasattr(loader, "__name__") else str(loader) for loader in loaders] for ext, loaders in self.loaders.items()}

    def get_loader_instruments(self, extension: str) -> list[str]:
        """Get a list of available instrument names for a given extension.

        Args:
            extension: File extension (e.g., '.xlsx')

        Returns:
            List of instrument names or class names
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        loaders = self.loaders.get(ext, [])
        instruments = []
        for loader in loaders:
            loader_inst = getattr(loader, "instrument", None)
            if hasattr(loader_inst, "default_value"):
                instruments.append(str(loader_inst.default_value))
            elif loader_inst is not None:
                instruments.append(str(loader_inst))
            else:
                instruments.append(loader.__name__)
        return instruments


def get_plugin_manager() -> IOPluginManager:
    """Get the global plugin manager instance.

    Returns:
        Global IOPluginManager instance
    """
    return IOPluginManager.get_instance()


__all__ = [
    "IOPluginManager",
    "get_plugin_manager",
]
