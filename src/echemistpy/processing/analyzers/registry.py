"""Registries that keep track of available analyzers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, TYPE_CHECKING

from traitlets import Dict as TDict, HasTraits

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .base import TechniqueAnalyzer


class TechniqueRegistry(HasTraits):
    """Map technique identifiers to analyzer instances."""

    _registry = TDict(help="Internal registry mapping technique identifiers to analyzer instances")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def register(self, technique: str, analyzer: "TechniqueAnalyzer") -> None:
        """Register an analyzer for a technique.

        Args:
            technique: Technique identifier (case-insensitive)
            analyzer: TechniqueAnalyzer instance
        """
        technique_lower = technique.lower()
        self._registry[technique_lower] = analyzer

    def unregister(self, technique: str) -> None:
        """Unregister an analyzer for a technique.

        Args:
            technique: Technique identifier (case-insensitive)
        """
        self._registry.pop(technique.lower(), None)

    def get(self, technique: str) -> "TechniqueAnalyzer":
        """Get analyzer for a technique.

        Args:
            technique: Technique identifier (case-insensitive)

        Returns:
            TechniqueAnalyzer instance

        Raises:
            KeyError: If technique not found
        """
        try:
            return self._registry[technique.lower()]
        except KeyError as exc:
            raise KeyError(f"No analyzer registered for '{technique}'.") from exc

    def available(self) -> Iterable[str]:
        """Get list of registered techniques.

        Returns:
            Tuple of available technique identifiers
        """
        return tuple(sorted(self._registry.keys()))

    def __contains__(self, technique: str) -> bool:
        """Check if technique is registered.

        Args:
            technique: Technique identifier (case-insensitive)

        Returns:
            True if technique is registered
        """
        return technique.lower() in self._registry

    def __len__(self) -> int:
        """Get number of registered techniques.

        Returns:
            Number of analyzers in registry
        """
        return len(self._registry)


def create_default_registry() -> TechniqueRegistry:
    """Return a registry populated with the built-in analyzers.

    Returns:
        TechniqueRegistry with standard analyzers
    """
    from .echem import CyclicVoltammetryAnalyzer
    from .tga import TGAAnalyzer
    from .xps import XPSAnalyzer
    from .xrd import XRDPowderAnalyzer

    registry = TechniqueRegistry()
    registry.register("xrd", XRDPowderAnalyzer())
    registry.register("xps", XPSAnalyzer())
    registry.register("tga", TGAAnalyzer())
    registry.register("echem", CyclicVoltammetryAnalyzer())
    return registry
