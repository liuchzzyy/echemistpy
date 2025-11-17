"""Namespace dedicated to community provided extensions."""

from __future__ import annotations

from importlib import import_module
from typing import Iterable, Sequence


def load_external_modules(modules: Iterable[str]) -> list[str]:
    """Import each module path in *modules* and return the loaded names."""

    loaded: list[str] = []
    for module_path in modules:
        module_path = module_path.strip()
        if not module_path:
            continue
        import_module(module_path)
        loaded.append(module_path)
    return loaded


def load_default_external_modules(extra: Sequence[str] | None = None) -> list[str]:
    """Load third-party modules listed in ``extra`` or an ``ECHEMISTPY_EXT`` env var."""

    import os

    requested = list(extra or [])
    env_value = os.getenv("ECHEMISTPY_EXT")
    if env_value:
        requested.extend(part.strip() for part in env_value.split(","))
    return load_external_modules(requested)


__all__ = ["load_external_modules", "load_default_external_modules"]
