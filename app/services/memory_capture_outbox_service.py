"""Park and replay TURNS the memory curator could not write (v3 §2.1.6).

See app/db/models/memory_capture_outbox.py for why this exists. In short: the
curator runs fire-and-forget after the reply is streamed, so when it fails
there is no request to return an error to and no user to retry it — everything
the user stated that turn is simply gone.

What round 8 parked here was a set of already-extracted facts, which made the
retry free. v3 parks the TURN, because the curator's whole job is to decide
what to change ABOUT THE CURRENT FILES: an op set computed against yesterday's
bodies would `rewrite` bullets that have since been merged away, and its
`match` strings would no longer exist. So a replay re-runs the writer, which
costs one model call — hence `REPLAY_PER_TURN = 1`.

The retry rides the per-turn post-processing path rather than a scheduler:
this is one small read on a turn that is already doing background work, and
the memory-maintenance scheduler has historically been flag-gated off.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.memory_capture_outbox import MemoryCaptureOutbox

logger = logging.getLogger(__name__)

# After this many failures the row is abandoned. A turn that fails five times
# over five separate turns is not failing transiently, and retrying it forever
# would turn one bad row into a permanent per-turn tax — now a per-turn LLM
# CALL, which makes the cap matter more than it did in round 8.
MAX_ATTEMPTS = 5

# Exponential-ish backoff, indexed by attempts already made.
_BACKOFF_MINUTES = (0, 1, 5, 30, 180)

# ONE replay per turn. Each one is a curator call against the real files; two
# parked turns replayed together would also race each other's ops through the
# same bodies. They drain one per turn instead.
REPLAY_PER_TURN = 1

#: A parked turn older than this is dropped unreplayed. The curator resolves
#: relative dates against TODAY ("tomorrow at 9"), so replaying a week-old
#: turn writes a wrong date as a confident fact.
MAX_AGE_HOURS = 48


def _backoff(attempts: int) -> datetime:
    idx = min(attempts, len(_BACKOFF_MINUTES) - 1)
    return datetime.utcnow() + timedelta(minutes=_BACKOFF_MINUTES[idx])


def serialize_turn(
    user_text: str, assistant_text: str = "", channel: str = "app"
) -> Dict[str, Any]:
    """The parked payload. Text is capped — the writer's own prompt caps it
    too, and an unbounded turn in a JSONB column is a row nobody can read."""
    return {
        "user_text": (user_text or "")[:8000],
        "assistant_text": (assistant_text or "")[:8000],
        "channel": channel or "app",
        "ts": datetime.utcnow().isoformat(),
    }


async def record_turn_failure(
    db: AsyncSession,
    user_id: str,
    user_text: str,
    assistant_text: str,
    error: BaseException,
    *,
    channel: str = "app",
    source_message_id: Optional[str] = None,
) -> Optional[str]:
    """Park a turn whose curation failed, so it is not lost.

    Never raises: this runs in the failure handler of the write path, and a
    failure to record a failure must not replace the original error.
    """
    if not (user_text or "").strip():
        return None

    def _row():
        # Built fresh each attempt: an instance that failed to flush is
        # already attached to the session that failed, and must not be reused.
        return MemoryCaptureOutbox(
            user_id=user_id,
            source_message_id=source_message_id,
            payload_json=serialize_turn(user_text, assistant_text, channel),
            attempts=0,
            next_attempt_at=_backoff(0),
            last_error=f"{type(error).__name__}: {str(error)[:400]}",
        )

    # First attempt: the caller's session.
    row = None
    try:
        row = _row()
        db.add(row)
        await db.flush()
        logger.warning(
            "[memory_outbox] parked a turn for user=%s after %s",
            str(user_id)[:8], type(error).__name__,
        )
        return row.id
    except Exception as exc:  # noqa: BLE001
        # Detach the pending row before falling back.
        #
        # A failed flush does NOT remove the object from the session — it
        # stays in `db.new`, and any later successful commit on that session
        # inserts it. Combined with the rescue copy below, that parked the
        # same turn TWICE and replayed it twice.
        if row is not None:
            try:
                db.expunge(row)
            except Exception:  # noqa: BLE001
                pass
        logger.warning(
            "[memory_outbox] park on the caller's session failed (%s); "
            "retrying on a fresh one", type(exc).__name__,
        )

    # Second attempt: a NEW session.
    #
    # This is the case the outbox exists for, and the first attempt cannot
    # serve it. If the write failed because of a database error, the caller's
    # transaction is already poisoned — every further statement on that
    # session raises PendingRollbackError, including the INSERT that parks the
    # row. So the turn would be lost precisely when the safety net was
    # supposed to catch it.
    #
    # A fresh session rather than a rollback of the caller's, because rolling
    # back someone else's transaction would silently discard whatever else
    # that turn had pending.
    try:
        from app.db.database import async_session_maker

        async with async_session_maker() as rescue:
            row = _row()
            rescue.add(row)
            await rescue.commit()
            logger.warning(
                "[memory_outbox] parked a turn for user=%s on a rescue session "
                "after %s", str(user_id)[:8], type(error).__name__,
            )
            return row.id
    except Exception as exc:  # noqa: BLE001
        # Both the caller's session and a fresh one failed, which means the
        # database is unreachable rather than merely unhappy.
        logger.error("[memory_outbox] could not park failed curation: %s", exc)
        return None


async def replay_pending(
    db: AsyncSession, user_id: str, limit: int = REPLAY_PER_TURN,
    *, api_key: Optional[str] = None,
) -> int:
    """Re-run the curator over parked turns. Returns rows resolved.

    Never raises, for the same reason `record_turn_failure` does not: it is
    called from the per-turn background block and must not break a turn.
    """
    from app.services import memory_curator

    resolved = 0
    try:
        now = datetime.utcnow()
        rows = (await db.execute(
            select(MemoryCaptureOutbox)
            .where(and_(
                MemoryCaptureOutbox.user_id == user_id,
                MemoryCaptureOutbox.resolved_at.is_(None),
                or_(
                    MemoryCaptureOutbox.next_attempt_at.is_(None),
                    MemoryCaptureOutbox.next_attempt_at <= now,
                ),
            ))
            .order_by(MemoryCaptureOutbox.created_at)
            .limit(limit)
        )).scalars().all()

        if not rows:
            return 0

        for row in rows:
            row.attempts += 1
            payload = row.payload_json
            if isinstance(payload, str):  # sqlite JSON round-trip
                try:
                    payload = json.loads(payload)
                except Exception:  # noqa: BLE001
                    payload = {}
            payload = payload if isinstance(payload, dict) else {}
            user_text = payload.get("user_text") or ""

            age_h = (now - (row.created_at or now)).total_seconds() / 3600.0
            if not user_text.strip() or age_h > MAX_AGE_HOURS:
                # Stale or empty: resolve it rather than replay it. The
                # curator resolves "tomorrow" against TODAY, so a two-day-old
                # turn would write a date that was never true.
                row.resolved_at = datetime.utcnow()
                row.last_error = (
                    "dropped: no user text" if not user_text.strip()
                    else f"dropped: parked {age_h:.0f}h ago (> {MAX_AGE_HOURS}h)"
                )
                logger.info("[memory_outbox] %s user=%s", row.last_error, str(user_id)[:8])
                continue

            try:
                await memory_curator.curate_turn(
                    db, user_id,
                    user_text=user_text,
                    assistant_text=payload.get("assistant_text") or "",
                    channel=payload.get("channel") or "app",
                    api_key=api_key,
                )
                row.resolved_at = datetime.utcnow()
                resolved += 1
                logger.info(
                    "[memory_outbox] replayed a turn for user=%s (attempt %d)",
                    str(user_id)[:8], row.attempts,
                )
            except Exception as exc:  # noqa: BLE001
                row.last_error = f"{type(exc).__name__}: {str(exc)[:400]}"
                if row.attempts >= MAX_ATTEMPTS:
                    # Abandoned, not silently dropped: resolved_at stops the
                    # retries and last_error records why. The row stays as the
                    # audit trail.
                    row.resolved_at = datetime.utcnow()
                    logger.error(
                        "[memory_outbox] ABANDONED after %d attempts user=%s: %s",
                        row.attempts, str(user_id)[:8], row.last_error,
                    )
                else:
                    row.next_attempt_at = _backoff(row.attempts)
                    logger.warning(
                        "[memory_outbox] replay failed (attempt %d/%d) user=%s: %s",
                        row.attempts, MAX_ATTEMPTS, str(user_id)[:8], row.last_error,
                    )
        await db.flush()
    except Exception as exc:  # noqa: BLE001
        logger.error("[memory_outbox] replay sweep failed: %s", exc)
    return resolved


async def pending_count(db: AsyncSession, user_id: str) -> int:
    """Rows still owed. Exposed for tests and for the health block."""
    from sqlalchemy import func

    return int((await db.execute(
        select(func.count(MemoryCaptureOutbox.id)).where(and_(
            MemoryCaptureOutbox.user_id == user_id,
            MemoryCaptureOutbox.resolved_at.is_(None),
        ))
    )).scalar() or 0)
