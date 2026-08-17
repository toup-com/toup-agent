"""Emit one row into ``product_events``. The whole module is this.

TELEMETRY MAY NOT BREAK DELIVERY. That is the entire design constraint,
and it forces two decisions that would otherwise look like over-caution:

**It opens its OWN session.** Handing it the caller's ``AsyncSession`` is
the obvious cheap thing and it is the dangerous thing: a failed INSERT
poisons the enclosing transaction on Postgres, so every subsequent
statement on that session raises ``InFailedSqlTransaction`` — and the only
way out is a ROLLBACK that discards the caller's own uncommitted work. A
metric would then be able to unsend an operator's message. The fan-out
holds one long-lived session across N recipients (
``admin_dispatch_worker.run_dispatch_fanout``), so the blast radius of
that mistake is the rest of the broadcast, not one row. Its own session
costs a pooled connection checkout and one round trip, against an agent
hop capped at 15s on the same code path.

**It swallows.** ``except Exception`` and log, never re-raise. Note the
type: ``asyncio.CancelledError`` is a ``BaseException`` since 3.8 and so
passes through — a shutdown must still be able to cancel the task this
runs inside.

Cheap, because it fires once per RECIPIENT of a broadcast: one INSERT,
no SELECT. Dedupe is the UNIQUE constraint plus a caught
``IntegrityError``, not a read-then-write — the read would double the
round trips AND still race two replicas.

R3: nothing here imports an LLM client, a credit service or a turn
runner, and nothing here can be reached by the agent. An operator's
message costs no credits and this must not become the exception.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)


async def emit_product_event(
    event: str,
    *,
    user_id: Optional[str] = None,
    actor_user_id: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    dedupe_key: Optional[str] = None,
    occurred_at: Optional[datetime] = None,
) -> None:
    """Record that ``event`` happened. Never raises.

    ``dedupe_key`` must be derived from the FACT, not from the attempt:
    ``f"{event}:{dispatch_id}:{user_id}"`` and friends. Pass None only
    where a repeat genuinely is a second occurrence.

    ``occurred_at`` exists because a caller that commits first and emits
    second (which is the correct order — see the call sites) would
    otherwise stamp the row with the time of the emit rather than of the
    fact.
    """
    # Imported inside the function, not at module scope: this module is
    # imported by app/api/notices.py and app/api/admin/dispatch.py, and a
    # module-level `from app.db.database import ...` there re-enters the
    # DB package during router import.
    from app.db.database import async_session_maker
    from app.db.models import ProductEvent

    try:
        async with async_session_maker() as db:
            db.add(ProductEvent(
                # Explicit rather than the column default: the default fires
                # at flush, and `occurred_at` is the whole point of the
                # parameter above.
                id=str(uuid.uuid4()),
                event=event,
                user_id=user_id,
                actor_user_id=actor_user_id,
                entity_type=entity_type,
                entity_id=entity_id,
                payload_json=payload or None,
                dedupe_key=dedupe_key,
                created_at=occurred_at or datetime.utcnow(),
            ))
            try:
                await db.commit()
            except IntegrityError:
                # The expected, boring outcome of a Retry press or of the
                # second replica reconciling the same dispatch: the fact is
                # already recorded. Not logged at warning — on a re-run of a
                # broadcast this is one line per recipient.
                await db.rollback()
                logger.debug(
                    "[product_events] %s already recorded (key=%s)",
                    event, dedupe_key,
                )
    except Exception:  # noqa: BLE001 — telemetry must never reach the caller
        logger.warning(
            "[product_events] emit failed for %s (entity=%s/%s)",
            event, entity_type, entity_id, exc_info=True,
        )
