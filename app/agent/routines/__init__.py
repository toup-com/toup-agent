"""Agent routines — system-managed scheduled actions.

`Routine` rows describe what should run on what schedule. `RoutineRunner`
owns the APScheduler instance that fires them. Per-kind handlers in
`registry.KIND_HANDLERS` carry out the actual work and write the result
into Day-as-Chat.

The runner is intentionally separate from `CronService` (user-authored
prompts that post to Telegram) and `HeartbeatService` (proactive nudges
sharing CronService's scheduler). Three concerns, two schedulers feels
right — Routines stand alone.
"""

from .base_handler import RoutineHandler, RoutineResult, RoutineStatus
from .email_briefing_handler import EmailBriefingHandler
from .registry import KIND_HANDLERS, register_handler
from .runner import RoutineRunner


# Register the email briefing handler so any routine with
# kind="email_briefing" gets a real dispatch (not the no-op fallback).
# Gating happens at `RoutineRunner._load_enabled_routines` via the
# `routines_email_briefing_enabled` per-tenant feature flag.
register_handler(EmailBriefingHandler())

__all__ = [
    "EmailBriefingHandler",
    "RoutineHandler",
    "RoutineResult",
    "RoutineStatus",
    "KIND_HANDLERS",
    "register_handler",
    "RoutineRunner",
]
