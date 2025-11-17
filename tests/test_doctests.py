"""Smoke-test doctest snippets exposed via public documentation."""

from __future__ import annotations

import doctest
import importlib
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - simple path guard
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - simple path guard
    sys.path.insert(0, str(SRC))

MODULES: Iterable[str] = (
    "echemistpy.utils.external.echem.lanhe_reader",
    "echemistpy.io.loaders",
    "echemistpy.pipelines.manager",
)


def test_doctests() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        failures, _ = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
        assert failures == 0, f"Doctests failed for {module_name}"
