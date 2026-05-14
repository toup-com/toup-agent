"""Routine handler contract.

A handler is a class implementing `execute()` for one routine kind. The
runner dispatches to handlers via `KIND_HANDLERS` in `registry.py`. The
handler does NOT write `routine_runs` rows — the runner owns idempotency
and the success/failure bookkeeping. The handler returns a structured
`RoutineResult` and the runner updates the run row from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession


# Status values that match `routine_runs.status` exactly.
RoutineStatus = str  # "running" | "success" | "partial" | "failed" | "skipped_reauth"

# Phase 2 / Ticket 2.1 — outcome state machine. See model docstring for
# semantics. The `status` field above stays as the legacy compatibility
# shim; new code branches on `outcome`. Mapping derived in `_outcome_for_result`.
RoutineOutcome = str  # "success" | "success_empty" | "partial" | "tool_error" | "failure"

VALID_OUTCOMES: frozenset[str] = frozenset({
    "success", "success_empty", "partial", "tool_error", "failure",
})


@dataclass
class RoutineResult:
    """Structured outcome of one handler.execute() call.

    Runner uses these fields to finalize the `routine_runs` row and (on
    success) advance the parent routine's `last_state_json` watermark.
    """

    status: RoutineStatus
    emails_fetched: int = 0
    summary_message_id: Optional[str] = None
    # On success, replaces `routines.last_state_json`. None = leave watermark
    # untouched (e.g., empty-result run that still completed cleanly should
    # advance; failed run must not).
    new_watermark: Optional[dict[str, Any]] = None
    error_class: Optional[str] = None
    error_detail: Optional[str] = None
    # Ticket 2.1 additions:
    # Per-channel delivery confirmations populated by the handler /
    # broadcaster pipeline. Shape:
    #   {"website": {"status": "delivered", "message_id": "...", ...},
    #    "telegram": {"status": "skipped", "reason": "no_recipient"}, ...}
    channel_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    # MCP tool names invoked while handling this run. Useful for forensics
    # when a tool_error outcome surfaces — operator sees exactly which
    # gmail__* call failed.
    tools_invoked: list[str] = field(default_factory=list)
    # Optional explicit outcome override. When set, runner uses this
    # value verbatim instead of deriving from `status` + channel_results.
    # Reserved for handlers that have richer outcome knowledge than the
    # generic mapping (e.g., agent_task with structured agent feedback).
    outcome: Optional[RoutineOutcome] = None
    # Free-form per-handler metrics for the structured log line (latency,
    # token counts, model name, etc.). Logged but not persisted.
    metrics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RoutineHandler(Protocol):
    """One handler per routine kind. Registered in `registry.KIND_HANDLERS`.

    `kind` is the string discriminator used in `Routine.kind` and the
    `source` column on the resulting Message row.

    `execute()` receives the loaded `Routine`, the already-claimed
    `RoutineRun` row (status='running'), and a DB session. It must NOT
    commit the run row — the runner owns the lifecycle.
    """

    kind: str

    async def execute(
        self,
        routine: Any,   # Routine — typed Any to avoid circular import
        run: Any,       # RoutineRun
        db: AsyncSession,
    ) -> RoutineResult: ...
