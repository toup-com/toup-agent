"""Persistent dedupe for WhatsApp Cloud API inbound webhooks.

Two operations on the ``whatsapp_inbound_dedupe`` table:

* :func:`claim` — single ``INSERT ... ON CONFLICT DO NOTHING``. Returns
  True only the first time the composite key is seen. The channel
  adapter calls this before dispatching the inbound; if it returns
  False the message is a Meta retry and must be skipped.
* :func:`sweep_expired` — deletes rows older than ``DEFAULT_TTL`` so the
  table stays small. Called periodically from a background task that
  starts with the channel.

Both functions accept an optional ``session_maker`` so tests can inject
a SQLite-backed maker without monkey-patching the global. In production
the default points at the agent's async session factory.

Why DB-backed and not in-memory:
  An in-memory ``set`` of seen message ids would lose state on every
  container restart. Meta is most likely to retry exactly during a
  restart window (the previous instance dropped the connection), so an
  in-memory dedupe would fail the one case it most needs to handle.
  The DB is per-tenant (each Toup container has its own Postgres role),
  so isolation is automatic and writes are <1 ms.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Optional

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# How long a claimed message stays in the table. Meta retries a webhook
# delivery on a curve that starts within seconds and tapers over the
# next half hour; 20 minutes covers the entire dense burst with margin
# while keeping the table compact.
DEFAULT_TTL: timedelta = timedelta(minutes=20)

# How often the sweep loop wakes up. Every 5 minutes is plenty — the
# window is 20 minutes and we don't need second-level precision.
SWEEP_INTERVAL_SECONDS: int = 300


# Type alias for the async-session-maker shape: a zero-arg callable
# that returns an async context manager yielding an AsyncSession.
SessionMaker = Callable[[], "AsyncSession"]


def _default_session_maker() -> SessionMaker:
    """Lazy-import the agent's session factory so this module stays
    importable in unit-test contexts that haven't loaded ``app.db``.
    """
    from app.db.database import async_session_maker

    return async_session_maker


async def claim(
    user_id: str,
    message_id: str,
    *,
    session_maker: Optional[SessionMaker] = None,
) -> bool:
    """Claim ``(user_id, message_id)``. Returns True iff first claim.

    Uses Postgres ``INSERT ... ON CONFLICT DO NOTHING`` and inspects
    rowcount: 1 means we inserted (first time), 0 means a row already
    exists for the key (already seen, retry must be skipped).

    Returns True (let-the-message-through) on any unexpected error so a
    DB blip can never silently drop legitimate inbound traffic. The
    error is logged for ops visibility.
    """
    if not user_id or not message_id:
        # Defensive: a Cloud API inbound without an ``id`` is rare but
        # possible (read receipts, system events). Treat as
        # already-claimed so the caller skips dedupe handling — those
        # frames have nothing to dispatch anyway.
        return False

    maker = session_maker or _default_session_maker()
    try:
        from app.db.models.agent import WhatsAppInboundDedupe

        async with maker() as db:  # type: AsyncSession
            stmt = (
                pg_insert(WhatsAppInboundDedupe.__table__)
                .values(
                    user_id=user_id,
                    message_id=message_id,
                    received_at=datetime.utcnow(),
                )
                .on_conflict_do_nothing(
                    constraint="pk_whatsapp_inbound_dedupe",
                )
            )
            result = await db.execute(stmt)
            await db.commit()
            return bool(getattr(result, "rowcount", 0))
    except Exception as exc:
        logger.warning(
            "whatsapp.dedupe.claim_failed user=%s msg_id=%s err=%s — letting through",
            user_id[:8] if user_id else "-",
            message_id[:16] if message_id else "-",
            exc,
        )
        return True


async def sweep_expired(
    *,
    older_than: timedelta = DEFAULT_TTL,
    session_maker: Optional[SessionMaker] = None,
) -> int:
    """Delete rows whose ``received_at`` is older than ``older_than``.

    Returns the number of rows pruned (0 on error). Safe to call on a
    short interval — the indexed ``received_at`` column makes this an
    index range delete.
    """
    maker = session_maker or _default_session_maker()
    cutoff = datetime.utcnow() - older_than
    try:
        from app.db.models.agent import WhatsAppInboundDedupe

        async with maker() as db:  # type: AsyncSession
            stmt = delete(WhatsAppInboundDedupe).where(
                WhatsAppInboundDedupe.received_at < cutoff
            )
            result = await db.execute(stmt)
            await db.commit()
            n = int(getattr(result, "rowcount", 0) or 0)
            if n:
                logger.info("whatsapp.dedupe.sweep pruned=%d cutoff=%s", n, cutoff.isoformat())
            return n
    except Exception as exc:
        logger.warning("whatsapp.dedupe.sweep_failed err=%s", exc)
        return 0


async def run_sweep_loop(
    *,
    interval_seconds: int = SWEEP_INTERVAL_SECONDS,
    older_than: timedelta = DEFAULT_TTL,
    session_maker: Optional[SessionMaker] = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Background loop: ``sweep_expired`` every ``interval_seconds``.

    Cancellable via ``asyncio.Task.cancel()``. Tests can inject a fast
    ``sleep`` to drive a deterministic number of iterations and then
    cancel.
    """
    import asyncio

    _sleep = sleep or asyncio.sleep
    while True:
        try:
            await _sleep(interval_seconds)
        except asyncio.CancelledError:
            return
        await sweep_expired(older_than=older_than, session_maker=session_maker)
