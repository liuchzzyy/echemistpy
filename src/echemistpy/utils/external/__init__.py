"""Abstractions for connecting to external services (left for future work)."""

from __future__ import annotations

from typing import Protocol


class ExternalService(Protocol):
    """Protocol describing a remote computation endpoint."""

    name: str

    def run(self, payload: dict) -> dict:
        """Execute a remote task and return structured results."""


__all__ = ["ExternalService"]
