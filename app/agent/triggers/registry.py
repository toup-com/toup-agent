"""Per-kind trigger handler registry.

Mirror of `app.agent.routines.registry`. Handlers register themselves
at import time via `register_handler(handler)`. The runner dispatches
via `KIND_HANDLERS[trigger.kind]`.
"""

from __future__ import annotations

from typing import Dict

from .base_handler import TriggerHandler


KIND_HANDLERS: Dict[str, TriggerHandler] = {}


def register_handler(handler: TriggerHandler) -> None:
    """Register a handler by its declared `kind`. Last-in wins so
    tests can swap a mock in without restarting the registry."""
    KIND_HANDLERS[handler.kind] = handler
