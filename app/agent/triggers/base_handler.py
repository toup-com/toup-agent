"""Trigger handler contract.

Mirrors `app.agent.routines.base_handler.RoutineHandler` in spirit but
distinct enough that we keep them separate:

  - Routines pass `(routine, run, db)`. Triggers pass `(trigger, events,
    db)` where `events` is a *batch* — coalesced siblings arrive in one
    handler invocation so an email-burst becomes one digest, not N
    LLM calls.
  - Routines bookkeep one `RoutineRun` per fire. Triggers bookkeep N
    `TriggerEvent` rows (one per inbound event); the handler marks each
    success/failure.
  - Routines have an idempotency gate on `(routine_id, local_date)`.
    Triggers have one on `(trigger_id, event_dedupe_id)`. Different
    semantics → different protocol.

The runner owns the run-row lifecycle (claim → run → finalise). The
handler returns a `TriggerResult` and the runner persists its outcome
into the `TriggerEvent` rows + the parent `Trigger.last_*` fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession


# Status values that mirror `trigger_events.status`.
TriggerEventStatus = str  # "queued" | "running" | "success" | "failed" | …


@dataclass
class TriggerResult:
    """Structured outcome of one handler.execute() call over a batch.

    `per_event_status` is the load-bearing field: maps each
    `trigger_event.id` in the input batch to its final status string.
    The runner uses this to finalise each row. If an event id is
    missing from the map the runner treats it as failed with
    `error_class='handler_did_not_report'`.

    `summary_message_id` is the single Day-as-Chat Message produced
    by the handler. Each event row's `summary_message_id` points at
    it — multiple events pointing to one Message is the normal
    coalesced shape.

    `new_provider_state` (if not None) is merged into
    `triggers.provider_state_json` — used by handlers that update
    a per-handler watermark (none in v1, but the slot exists so
    future kinds don't need a schema change).
    """

    status: TriggerEventStatus  # overall handler outcome
    per_event_status: dict[str, str] = field(default_factory=dict)
    summary_message_id: Optional[str] = None
    new_provider_state: Optional[dict[str, Any]] = None
    error_class: Optional[str] = None
    error_detail: Optional[str] = None
    # Free-form per-handler metrics — logged but not persisted.
    metrics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TriggerHandler(Protocol):
    """One handler per trigger kind + action pair (or kind-only when
    the handler internally dispatches on action). Registered in
    `registry.KIND_HANDLERS`.

    `kind` is the string discriminator used in `Trigger.kind` and the
    `source` column on the resulting Message row.

    `execute()` receives the loaded `Trigger`, the batch of
    `TriggerEvent` rows already claimed by the runner (status='running'),
    and a DB session. It must NOT commit the rows — the runner owns
    that lifecycle.
    """

    kind: str

    async def execute(
        self,
        trigger: Any,           # Trigger — typed Any to avoid circular import
        events: list[Any],      # list[TriggerEvent]
        db: AsyncSession,
    ) -> TriggerResult: ...
