"""Per-kind handler registry.

Gate 2 populates this dict with the real email briefing handler. Gate 1
leaves it empty — the runner's no-op dispatch path logs `would_execute`
and exits, exercising the lifecycle without any handler.
"""

from __future__ import annotations

from typing import Dict

from .base_handler import RoutineHandler


KIND_HANDLERS: Dict[str, RoutineHandler] = {}


def register_handler(handler: RoutineHandler) -> None:
    """Register a handler by its declared `kind`. Last-in wins so tests can
    swap a mock in without restarting the registry."""
    KIND_HANDLERS[handler.kind] = handler
