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
from .base_handler import RoutineHandler, RoutineResult, RoutineStatus
from .email_briefing_handler import EmailBriefingHandler
from .registry import KIND_HANDLERS, register_handler
from .runner import RoutineRunner


# Default handler registrations. Both kinds gate on the per-tenant
# feature flag (see `RoutineRunner._kind_enabled`).
#   - `email_briefing` — Gmail-specialised preset
#   - `agent_task` — generic prompt-driven, runs through the agent
register_handler(EmailBriefingHandler())
register_handler(AgentTaskHandler())

__all__ = [
    "AgentTaskHandler",
    "EmailBriefingHandler",
    "RoutineHandler",
    "RoutineResult",
    "RoutineStatus",
    "KIND_HANDLERS",
    "register_handler",
    "RoutineRunner",
]
