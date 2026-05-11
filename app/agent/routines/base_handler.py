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
