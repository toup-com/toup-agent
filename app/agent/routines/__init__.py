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
from .reminder_handler import ReminderHandler
from .registry import KIND_HANDLERS, register_handler
from .runner import RoutineRunner


# Default handler registrations. Each kind has its own feature flag
# gate (see `RoutineRunner._kind_enabled`).
#   - `email_briefing` — Gmail-specialised preset (LLM + MCP)
#   - `agent_task`     — generic prompt-driven, runs through the agent (LLM)
#   - `reminder`       — text-only delivery, no LLM/MCP (mig 042)
register_handler(EmailBriefingHandler())
register_handler(AgentTaskHandler())
register_handler(ReminderHandler())

__all__ = [
    "AgentTaskHandler",
    "EmailBriefingHandler",
    "ReminderHandler",
    "RoutineHandler",
    "RoutineResult",
    "RoutineStatus",
    "KIND_HANDLERS",
    "register_handler",
    "RoutineRunner",
]
