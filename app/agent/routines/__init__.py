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

from .agent_task_handler import AgentTaskHandler
from .autopilot_handler import AutopilotHandler
from .base_handler import RoutineHandler, RoutineResult, RoutineStatus
from .email_briefing_handler import EmailBriefingHandler
from .reminder_handler import ReminderHandler
from .registry import KIND_HANDLERS, register_handler
from .runner import RoutineRunner


# Default handler registrations. Each kind has its own feature flag
# gate (see `RoutineRunner._kind_enabled`).
#   - `email_briefing` — Gmail-specialised preset (LLM + MCP)
#   - `agent_task`     — generic prompt-driven, runs through the agent (LLM)
#   - `reminder`       — text-only delivery, no LLM/MCP (mig 042)
#   - `autopilot`      — autonomous mission heartbeat (Autopilot PR6)
register_handler(EmailBriefingHandler())
register_handler(AgentTaskHandler())
register_handler(ReminderHandler())
register_handler(AutopilotHandler())

# Automations engine (Round 26) — `automation_poll` / `automation_schedule`.
# Registered ONLY behind the flag so a dark tenant's handler registry —
# and everything derived from it (API kind validation, runner gating) —
# is byte-identical to today's. The runner's `_kind_enabled` gates
# fires a second time, so a flag flipped off after registration also
# goes quiet.
try:
    from app.config import settings as _settings
    if getattr(_settings, "automations_enabled", False):
        from app.agent.automations.handlers import register_automation_handlers
        register_automation_handlers()
except Exception as _e:  # noqa: BLE001 — never break routine boot
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "automations handler registration failed: %s", _e,
    )

__all__ = [
    "AgentTaskHandler",
    "AutopilotHandler",
    "EmailBriefingHandler",
    "ReminderHandler",
    "RoutineHandler",
    "RoutineResult",
    "RoutineStatus",
    "KIND_HANDLERS",
    "register_handler",
    "RoutineRunner",
]
