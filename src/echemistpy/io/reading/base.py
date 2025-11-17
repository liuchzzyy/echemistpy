"""Base classes shared by all data readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from ..reorganization import MeasurementRecord


class BaseFileReader(ABC):
    """Abstract reader that converts instrument files into measurement records."""

    #: Canonical name for the technique (e.g. "echem", "xps").
    technique: str

    def __init__(self, **default_kwargs: Any) -> None:
        self._default_kwargs = default_kwargs

    @abstractmethod
    def read(self, path: str | Path, **kwargs: Any) -> "MeasurementRecord":
        """Parse *path* and return a :class:`MeasurementRecord`."""

    def __call__(self, path: str | Path, **overrides: Any) -> "MeasurementRecord":
        """Allow reader instances to be used as callables."""

        call_kwargs = {**self._default_kwargs, **overrides}
        return self.read(path, **call_kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(technique={self.technique!r})"
