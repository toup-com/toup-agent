"""Trigger runtime — event-driven sibling of `app.agent.routines`.

Routines: APScheduler-driven, fire on a cron. Triggers: event-driven,
fire when the platform-side webhook dispatches an envelope. Both
share the Day-as-Chat write contract (`role=assistant`,
`channel=trigger`/`routine`, `source=<kind>`) so Mission Control can
render either with one component.

Public surface:
  - `TriggerRunner` — singleton runtime, started at agent boot.
  - `KIND_HANDLERS` — registry populated by the per-kind handler
    modules. The runner dispatches via `KIND_HANDLERS[trigger.kind]`.
  - `register_handler` — used by handler modules at import time.
"""

from .base_handler import TriggerHandler, TriggerResult
from .registry import KIND_HANDLERS, register_handler
from .runner import TriggerRunner

__all__ = [
    "TriggerHandler", "TriggerResult",
    "KIND_HANDLERS", "register_handler",
    "TriggerRunner",
]
