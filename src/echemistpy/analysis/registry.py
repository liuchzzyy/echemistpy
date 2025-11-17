"""Registries that keep track of available analyzers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict

from echemistpy.io.structures import Measurement

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .base import TechniqueAnalyzer


class TechniqueRegistry:
    """Map technique identifiers to analyzer instances."""

    def __init__(self) -> None:
        self._registry: Dict[str, "TechniqueAnalyzer"] = {}

    def register(self, technique: str, analyzer: "TechniqueAnalyzer") -> None:
        technique_lower = technique.lower()
        self._registry[technique_lower] = analyzer

    def unregister(self, technique: str) -> None:
        self._registry.pop(technique.lower(), None)

    def get(self, technique: str) -> "TechniqueAnalyzer":
        try:
            return self._registry[technique.lower()]
        except KeyError as exc:
            raise KeyError(f"No analyzer registered for '{technique}'.") from exc

    def available(self) -> Iterable[str]:
        return tuple(sorted(self._registry.keys()))

    def __contains__(self, technique: str) -> bool:  # pragma: no cover - trivial
        return technique.lower() in self._registry

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._registry)


def create_default_registry() -> TechniqueRegistry:
    """Return a registry populated with the built-in analyzers."""

    from echemistpy.analysis.echem import CyclicVoltammetryAnalyzer
    from echemistpy.analysis.tga import TGAAnalyzer
    from echemistpy.analysis.xps import XPSAnalyzer
    from echemistpy.analysis.xrd import XRDPowderAnalyzer

    registry = TechniqueRegistry()
    registry.register("xrd", XRDPowderAnalyzer())
    registry.register("xps", XPSAnalyzer())
    registry.register("tga", TGAAnalyzer())
    registry.register("echem", CyclicVoltammetryAnalyzer())
    return registry


class TechniqueAnalyzer:
    """Avoid circular imports in annotations."""

    def analyze(self, measurement: Measurement):  # pragma: no cover - interface
        raise NotImplementedError
