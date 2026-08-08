"""Record and replay memory captures whose write failed.

See app/db/models/memory_capture_outbox.py for why this exists. In short:
extraction happens fire-and-forget after the reply is streamed, so when the
write fails there is no request to return an error to and no user to retry it —
the facts stated that turn are simply gone.

The retry deliberately rides the per-turn maintenance path
(agent_runner's post-processing block, alongside expire_stale_memories and
decay_expired_tasks) rather than the memory-maintenance scheduler, which is
behind a flag that has never been enabled in production.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.memory_capture_outbox import MemoryCaptureOutbox

logger = logging.getLogger(__name__)

# After this many failures the row is abandoned. A payload that fails five
# times over five separate turns is not failing transiently, and retrying it
# forever would turn one bad row into a permanent per-turn tax.
MAX_ATTEMPTS = 5

# Exponential-ish backoff, indexed by attempts already made.
_BACKOFF_MINUTES = (0, 1, 5, 30, 180)

# How many rows one turn is allowed to replay. Capture is not urgent and the
# per-turn path is shared with the user's reply latency budget.
REPLAY_PER_TURN = 3


def _backoff(attempts: int) -> datetime:
    idx = min(attempts, len(_BACKOFF_MINUTES) - 1)
    return datetime.utcnow() + timedelta(minutes=_BACKOFF_MINUTES[idx])


def serialize_extracted(memories: Sequence[Any]) -> List[Dict[str, Any]]:
    """Flatten ExtractedMemory objects into JSON the retry can replay.

    Only the fields the write path actually consumes. Entities are carried
    because they are written in the same block.
    """
    out: List[Dict[str, Any]] = []
    for m in memories:
        out.append({
            "content": m.content,
            "summary": getattr(m, "summary", None),
            "category": str(getattr(m.category, "value", m.category)),
            "memory_type": str(getattr(m.memory_type, "value", m.memory_type)),
            "importance": float(getattr(m, "importance", 0.5) or 0.5),
            "confidence": float(getattr(m, "confidence", 0.8) or 0.8),
            "tags": list(getattr(m, "tags", []) or []),
            "metadata": dict(getattr(m, "metadata", {}) or {}),
            "ttl_days": getattr(m, "ttl_days", None),
        })
    return out


async def record_failure(
    db: AsyncSession,
    user_id: str,
    memories: Sequence[Any],
    error: BaseException,
    source_message_id: Optional[str] = None,
) -> Optional[str]:
    """Park an extraction whose write failed, so it is not lost.

    Never raises: this runs in the failure handler of the capture path, and a
    failure to record a failure must not replace the original error.
    """
    if not memories:
        return None

    def _row():
        # Built fresh each attempt: an instance that failed to flush is already
        # attached to the session that failed, and must not be reused.
        return MemoryCaptureOutbox(
            user_id=user_id,
            source_message_id=source_message_id,
            payload_json=serialize_extracted(memories),
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
            "[memory_outbox] parked %d fact(s) for user=%s after %s",
            len(memories), user_id, type(error).__name__,
        )
        return row.id
    except Exception as exc:  # noqa: BLE001
        # Detach the pending row before falling back.
        #
        # A failed flush does NOT remove the object from the session — it stays
        # in `db.new`, and any later successful commit on that session inserts
        # it. Combined with the rescue copy below, that parked the same facts
        # TWICE and replayed them twice. Caught by
        # test_parking_survives_a_poisoned_transaction, which found 2 rows where
        # it asserted 1.
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
    # This is the case the outbox exists for, and the first attempt cannot serve
    # it. If the write failed because of a database error, the caller's
    # transaction is already poisoned — every further statement on that session
    # raises PendingRollbackError, including the INSERT that parks the row. So
    # the facts would be lost precisely when the safety net was supposed to
    # catch them.
    #
    # A fresh session rather than a rollback of the caller's, because rolling
    # back someone else's transaction would silently discard whatever else that
    # turn had pending. This function's job is to not lose data, not to decide
    # what other work is expendable.
    try:
        from app.db.database import async_session_maker

        async with async_session_maker() as rescue:
            row = _row()
            rescue.add(row)
            await rescue.commit()
            logger.warning(
                "[memory_outbox] parked %d fact(s) for user=%s on a rescue "
                "session after %s", len(memories), user_id, type(error).__name__,
            )
            return row.id
    except Exception as exc:  # noqa: BLE001
        # Both the caller's session and a fresh one failed, which means the
        # database is unreachable rather than merely unhappy. Nothing further
        # can be persisted; §5.2 of the verification report states this limit.
        logger.error("[memory_outbox] could not park failed capture: %s", exc)
        return None


async def replay_pending(
    db: AsyncSession, user_id: str, limit: int = REPLAY_PER_TURN
) -> int:
    """Retry parked captures for this user. Returns the number of rows resolved.

    Never raises, for the same reason record_failure does not: it is called
    from the per-turn maintenance block and must not break a turn.
    """
    from app.schemas import BrainType, MemoryCreate
    from app.services.memory_dedup_service import MemoryDedupService

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

        dedup = MemoryDedupService(db)
        for row in rows:
            row.attempts += 1
            try:
                payload = row.payload_json
                if isinstance(payload, str):     # sqlite JSON round-trip
                    payload = json.loads(payload)
                # Carry the lease back across the replay. serialize_extracted
                # deliberately stores `ttl_days`, and this rebuild used to drop
                # it — MemoryCreate.expires_at then defaulted to None, so every
                # parked transient memory came back PERMANENT. Worse, if the
                # replayed content deduped against the row that did land, the
                # reinforce path reads `expires_at is None` as "restated as
                # durable" and clears the lease on the survivor too.
                #
                # Anchor on when the fact was STATED (row.created_at), not on
                # when the retry happened to run: a fact parked for two days is
                # two days into its life, not starting fresh.
                _anchor = row.created_at or datetime.utcnow()
                creates = []
                for item in (payload or []):
                    _ttl = item.get("ttl_days")
                    creates.append(
                        MemoryCreate(
                            content=item["content"],
                            summary=item.get("summary"),
                            brain_type=BrainType.USER,
                            category=item.get("category") or "other",
                            memory_type=item.get("memory_type") or "fact",
                            importance=item.get("importance", 0.5),
                            confidence=item.get("confidence", 0.8),
                            tags=item.get("tags") or [],
                            metadata=item.get("metadata") or {},
                            source_type="outbox_replay",
                            expires_at=(
                                _anchor + timedelta(days=float(_ttl))
                                if _ttl else None
                            ),
                        )
                    )
                if creates:
                    await dedup.smart_create_memories(
                        new_memories=creates, user_id=user_id
                    )
                row.resolved_at = datetime.utcnow()
                resolved += 1
                logger.info(
                    "[memory_outbox] replayed %d fact(s) for user=%s (attempt %d)",
                    len(creates), user_id, row.attempts,
                )
            except Exception as exc:  # noqa: BLE001
                row.last_error = f"{type(exc).__name__}: {str(exc)[:400]}"
                if row.attempts >= MAX_ATTEMPTS:
                    # Abandoned, not silently dropped: resolved_at is set so the
                    # row stops being retried, and last_error records why. The
                    # row itself stays as the audit trail.
                    row.resolved_at = datetime.utcnow()
                    logger.error(
                        "[memory_outbox] ABANDONED after %d attempts user=%s: %s",
                        row.attempts, user_id, row.last_error,
                    )
                else:
                    row.next_attempt_at = _backoff(row.attempts)
                    logger.warning(
                        "[memory_outbox] replay failed (attempt %d/%d) user=%s: %s",
                        row.attempts, MAX_ATTEMPTS, user_id, row.last_error,
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
