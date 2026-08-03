"""Agent-side notify client — durable outbox + flush loop (Autopilot PR4).

The ONE way agent-side code sends a proactive notification::

    from app.services.agent_notify_client import notify
    await notify(
        event_kind="needs_approval",
        title="Autopilot needs a decision",
        body="Send the draft email to Sam?",
        data={"route": "mission-control", "mission_id": ...},
        priority="high",
    )

``notify()`` persists an ``agent_notify_outbox`` row (tenant DB —
survives restarts and blue-green swaps) and kicks an opportunistic
flush; the background loop retries anything the opportunistic pass
missed. A row is done only when the platform's POST /api/agent/notify
acks it (queued/duplicate) — the row id IS the platform idempotency
key, so at-least-once flushing is safe by construction.

Endpoint/key gating copies credit_reporter (_platform_endpoint /
_agent_key): silently no-ops when the container isn't configured with
a platform URL + agent key (dev monolith), and defends against tenants
whose platform_api_url lacks the /api suffix.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx
from sqlalchemy import or_, select, update

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentNotifyOutbox
from app.services.background_tasks import spawn as _spawn_bg

logger = logging.getLogger(__name__)

_FLUSH_BATCH = 25
_FLUSH_INTERVAL_S = 30
# A notification that couldn't be delivered for a day has lost its
# moment — expire instead of retrying forever.
_EXPIRE_AFTER = timedelta(hours=24)
_BACKOFF_BASE = timedelta(seconds=30)
_BACKOFF_CAP = timedelta(minutes=15)

# Kick an immediate flush from notify() (latency optimization — the
# loop is the guarantee). Tests set this False to keep flushes
# deterministic.
OPPORTUNISTIC_FLUSH = True


def _platform_endpoint() -> Optional[str]:
    platform_url = (getattr(settings, "platform_api_url", "") or "").strip()
    if not platform_url:
        return None
    base = platform_url.rstrip("/")
    # Some tenants' platform_api_url lacks the /api suffix (see
    # agent_main's register normalization) — append when missing.
    if not base.endswith("/api"):
        base = f"{base}/api"
    return f"{base}/agent/notify"


def _agent_key() -> str:
    return (getattr(settings, "agent_api_key", "") or "").strip()


async def notify(
    *,
    event_kind: str,
    title: str,
    body: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    priority: str = "default",
    dedup_key: Optional[str] = None,
) -> Optional[str]:
    """Durably enqueue a notification; returns the outbox row id.
    Returns None (and logs) when the tenant DB isn't writable yet
    (pool-lobby pre-bind) — producers treat that as best-effort."""
    row_id = str(uuid.uuid4())
    try:
        async with async_session_maker() as db:
            db.add(AgentNotifyOutbox(
                id=row_id,
                event_kind=event_kind,
                title=title[:200],
                body=body,
                data_json=data,
                priority=priority,
                dedup_key=dedup_key,
                created_at=datetime.utcnow(),
            ))
            await db.commit()
    except Exception as e:  # noqa: BLE001 — pre-bind lobby / DB hiccup
        logger.warning("notify(): outbox write failed (%s) — dropped", e)
        return None

    # Opportunistic immediate flush so the happy path is sub-second;
    # the background loop is the guarantee, this is just latency.
    if OPPORTUNISTIC_FLUSH:
        try:
            _spawn_bg(flush_notify_outbox())
        except RuntimeError:
            pass  # no running loop (sync caller) — loop will catch it
    return row_id


async def flush_notify_outbox(limit: int = _FLUSH_BATCH) -> Dict[str, int]:
    """Flush unacked rows. Safe to call concurrently — the worst case
    is a duplicate POST, which the platform dedupes on the row id."""
    endpoint = _platform_endpoint()
    key = _agent_key()
    user_id = (getattr(settings, "user_id", "") or "").strip()
    if not endpoint or not key or not user_id:
        return {"skipped_unconfigured": 1}

    now = datetime.utcnow()
    stats = {"acked": 0, "retried": 0, "expired": 0}
    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentNotifyOutbox)
            .where(
                AgentNotifyOutbox.flushed_at.is_(None),
                or_(
                    AgentNotifyOutbox.next_attempt_at.is_(None),
                    AgentNotifyOutbox.next_attempt_at <= now,
                ),
            )
            .order_by(AgentNotifyOutbox.created_at)
            .limit(limit)
        )
        rows = list(result.scalars().all())

        for row in rows:
            if now - row.created_at > _EXPIRE_AFTER:
                row.flushed_at = now
                row.last_error = "expired"
                stats["expired"] += 1
                continue

            outcome, err = await _post_row(endpoint, key, user_id, row)
            if outcome == "acked":
                row.flushed_at = now
                row.last_error = None
                stats["acked"] += 1
            elif outcome == "permanent":
                # 4xx contract rejection — a poison row; retrying
                # forever would wedge the queue behind it.
                row.flushed_at = now
                row.last_error = f"rejected: {err}"
                logger.error("notify outbox row %s rejected: %s", row.id, err)
                stats["expired"] += 1
            else:  # transient
                row.attempts += 1
                backoff = min(
                    _BACKOFF_BASE * (2 ** max(0, row.attempts - 1)), _BACKOFF_CAP,
                )
                row.next_attempt_at = now + backoff
                row.last_error = str(err)[:300]
                stats["retried"] += 1
        await db.commit()
    return stats


async def _post_row(
    endpoint: str, key: str, user_id: str, row: AgentNotifyOutbox,
) -> tuple[str, Optional[str]]:
    """→ ("acked"|"permanent"|"transient", error)."""
    payload = {
        "user_id": user_id,
        "idempotency_key": row.id,
        "event_kind": row.event_kind,
        "title": row.title,
        "body": row.body,
        "data": row.data_json,
        "priority": row.priority,
        "dedup_key": row.dedup_key,
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                endpoint, json=payload, headers={"X-Agent-Key": key},
            )
        if resp.status_code == 200:
            status = (resp.json() or {}).get("status")
            if status in ("queued", "duplicate"):
                return ("acked", None)
            return ("transient", f"unexpected body: {resp.text[:120]}")
        if resp.status_code in (401, 403, 422):
            return ("permanent", f"http_{resp.status_code}: {resp.text[:200]}")
        return ("transient", f"http_{resp.status_code}")
    except Exception as e:  # noqa: BLE001 — network
        return ("transient", f"{type(e).__name__}: {str(e)[:150]}")


async def notify_outbox_loop() -> None:
    """Background flusher — started from the agent lifespan. Tolerates
    pre-bind lobby mode (DB raises) by just sleeping through it."""
    logger.info("notify outbox loop started (interval %ss)", _FLUSH_INTERVAL_S)
    while True:
        try:
            await flush_notify_outbox()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — never die
            logger.debug("notify outbox flush skipped: %s", e)
        await asyncio.sleep(_FLUSH_INTERVAL_S)
