"""One-active-per-kind uniqueness check for routines.

Extracted from `app/api/routines.py` so the same logic is callable
from:
  - the FastAPI route handler (`create_routine`, `update_routine`)
  - the agent's routines skill (`routines__create`)
  - tests (without dragging in the FastAPI router init that's broken
    in the test env's Starlette version)

The DB-level guarantee lives in the partial UNIQUE index added by
migration 041 (`uq_routines_one_per_kind`, Postgres-only). This
function is the friendly-error path: it returns the existing routine
ids when a collision would occur so the caller can render a clean 409
with actionable detail.

Pre-2026-05-14 the API used `.scalar_one_or_none()` here, which
raises `MultipleResultsFound` if ≥2 enabled routines of the same kind
already exist — i.e., it crashed in exactly the state it was supposed
to guard against. This module replaces that with a multi-row-safe
query.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


# Kinds that are NOT singleton-per-user.
#
# - `agent_task`: generic "run this prompt" — users want many ("morning
#   briefing", "noon GitHub check", "evening calendar review"). One-per-
#   kind would collapse them into a single row, useless.
# - `reminder`: users naturally have many ("call mom at 6", "water the
#   plants Tuesday", "standup ping at 9:30") — the routines__remind
#   skill was designed to support N reminders per user. Adding only
#   `agent_task` was an oversight in mig 041; without `reminder` here,
#   creating a second reminder returns 409 and the agent surfaces a
#   confusing error to the user.
#
# Other kinds (email_briefing, future calendar_briefing, etc.) are
# presets where one active instance per user is the design.
KINDS_EXEMPT_FROM_ONE_PER_KIND: frozenset[str] = frozenset({"agent_task", "reminder"})


async def find_conflicting_enabled_routine_ids(
    db: AsyncSession,
    *,
    user_id: str,
    kind: str,
    exclude_routine_id: Optional[str] = None,
) -> List[str]:
    """Return the ids of every enabled routine for this (user, kind)
    that would collide with creating-or-enabling a routine of that kind.

    Empty list means "no conflict, the create/enable can proceed."
    Non-empty means the caller MUST 409.

    `exclude_routine_id` is used by the update path — when enabling a
    previously-disabled routine, exclude its own id so a self-enable
    doesn't false-positive.

    Returns a list (not an Optional) so the multi-row case is handled
    cleanly: even when the DB already has duplicates (legacy state
    before migration 041's cleanup), this returns ALL the ids so the
    operator / agent can see the full mess instead of one arbitrary row.
    """
    if kind in KINDS_EXEMPT_FROM_ONE_PER_KIND:
        return []

    # Avoid circular import — Routine lives in app.db.models.
    from app.db.models import Routine

    stmt = (
        select(Routine.id)
        .where(
            Routine.user_id == user_id,
            Routine.kind == kind,
            Routine.enabled == True,  # noqa: E712
        )
    )
    if exclude_routine_id is not None:
        stmt = stmt.where(Routine.id != exclude_routine_id)

    result = await db.execute(stmt)
    return [row[0] for row in result.all()]
