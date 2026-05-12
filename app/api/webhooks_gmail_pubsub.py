"""Platform-side Gmail Pub/Sub webhook.

Endpoint: `POST /api/v1/webhooks/gmail`

Flow per push:
  1. Verify the Pub/Sub JWT (Bearer header, RS256 against Google
     JWKS, audience + signer pinned). Bad JWT → 401, Pub/Sub stops
     retrying.
  2. Decode `(emailAddress, historyId)` from the base64 message.data
     blob. Malformed → 400.
  3. Resolve the email to a `connector_identity` row → user_id.
     No match → 200 with `unknown_email=true` in the log. We answer
     200 so Pub/Sub stops retrying; a user disconnecting Gmail
     while a push is in flight is not an error.
  4. Look up enabled email_received triggers for the user. The
     platform doesn't actually need the trigger rows here (the
     agent owns those) — it just needs to know there's *something*
     to deliver to. We use the most recent `provider_state_json`
     watermark to drive the history.list call. (Multi-trigger
     watermark coordination is documented in the runbook.)
  5. Fetch new gmail_message_ids via history.list from the stored
     baseline.
  6. Dispatch the message-ids to the user's per-tenant agent
     container via the inbound endpoint. The agent dedupes on
     UNIQUE(trigger_id, event_dedupe_id) — Pub/Sub retries are
     idempotent.
  7. Update the watermark + return 200.

Latency budget: ~2 s p50, hard cap 10 s (Pub/Sub timeout). Tokens
live in the connector vault — fetch is fast, history.list is a
single Google API call (or a handful for big deltas). Dispatch to
agent is a single httpx call.

Per the trigger spec: filtering happens at the AGENT (Gate T2), not
here. The webhook delivers every new message-id; the runner applies
filters before invoking handler.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from sqlalchemy import select

from app.config import settings
from app.db.database import async_session_maker
from app.services.gmail_pubsub import (
    PubSubAuthError,
    GmailWatchError,
    decode_pubsub_envelope,
    list_new_message_ids,
    verify_pubsub_jwt,
)
from app.services.trigger_dispatch import (
    TriggerDispatchError,
    dispatch_events,
)


router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)


@router.post("/gmail")
async def gmail_pubsub_webhook(
    request: Request,
    authorization: str = Header(default="", alias="Authorization"),
) -> dict[str, Any]:
    """Production Gmail Pub/Sub push receiver.

    Returns 200 + structured JSON on every successful authenticated
    delivery — Pub/Sub treats 200 as "stop retrying." Returns 401 on
    JWT failure (forged / wrong audience / expired), 400 on malformed
    envelope, 503 on transient downstream errors (agent unreachable,
    Gmail API rate-limited) so Pub/Sub retries.

    Never logs PII. Sender addresses, subjects, bodies — none of these
    are touched by this endpoint. Only message-ids, which are opaque
    handles, get propagated.
    """
    started_ms = _ms_now()

    # Feature flag — fail-closed gate so a legacy push during a
    # rollback can't run the dispatch code path. Trigger flag mirrors
    # the routine flag pattern.
    if not getattr(settings, "triggers_email_enabled", False):
        logger.info("[gmail_pubsub] dropped: feature disabled")
        return {"status": "ok", "dropped": "feature_disabled"}

    # 1. JWT verify. Pub/Sub uses `Authorization: Bearer <JWT>`.
    try:
        claims = await verify_pubsub_jwt(authorization)
    except PubSubAuthError as e:
        logger.warning("[gmail_pubsub] jwt_verify_failed reason=%s", str(e)[:200])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"pubsub auth failed: {e}",
        )

    # 2. Decode envelope.
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    try:
        email, history_id = decode_pubsub_envelope(body)
    except ValueError as e:
        # Malformed = permanent error. 400 stops Pub/Sub retries.
        logger.warning("[gmail_pubsub] envelope_invalid reason=%s", str(e)[:200])
        raise HTTPException(status_code=400, detail=f"envelope invalid: {e}")

    # 3. Resolve email → user_id via connector_identities.
    user_id = await _resolve_user_for_email(email)
    if user_id is None:
        # Common, not an error — user disconnected Gmail, or the push
        # is for an email we never had a connector for. Return 200 so
        # Pub/Sub stops retrying. Log so ops can spot a misconfigured
        # subscription that's spraying us with foreign emails.
        logger.info(
            "[gmail_pubsub] unknown_email signer=%s history_id=%s",
            (claims.get("email") or "")[:64], history_id,
        )
        return {
            "status": "ok",
            "dropped": "unknown_email",
            "latency_ms": _ms_now() - started_ms,
        }

    # 4. Look up the trigger watermark for this user. We need it to
    # know where to resume history.list from. In Gate T1 we store this
    # in `triggers.provider_state_json.gmail_history_id`; multiple
    # triggers for the same Gmail account share the watermark (= the
    # furthest-advanced cursor). For the v1 single-trigger case this
    # is straightforward.
    watermark, trigger_count = await _read_watermark(user_id)
    if trigger_count == 0:
        # No enabled email_received trigger — push delivered but no
        # action. Don't fetch history (saves a Google API call), just
        # acknowledge so Pub/Sub stops.
        logger.info(
            "[gmail_pubsub] no_active_trigger user_id=%s history_id=%s",
            user_id[:8], history_id,
        )
        return {
            "status": "ok",
            "dropped": "no_active_trigger",
            "latency_ms": _ms_now() - started_ms,
        }
    if not watermark:
        # First push after the watch was just provisioned — use the
        # push's history_id as the baseline so we don't replay
        # arbitrary history.
        watermark = history_id

    # 5. Fetch new message-ids since the watermark.
    try:
        msg_ids, new_watermark = await list_new_message_ids(
            user_id, start_history_id=watermark,
        )
    except GmailWatchError as e:
        # Two cases:
        #   - 404 (history too old): caller needs to re-arm watch.
        #     We return 200 (don't retry this exact push) and the
        #     refresh job repairs it on its next pass.
        #   - Other 5xx / token mint failure: transient → 503.
        msg = str(e)
        if "too old" in msg:
            logger.warning(
                "[gmail_pubsub] history_too_old user_id=%s history_id=%s",
                user_id[:8], watermark,
            )
            await _flag_watch_needs_refresh(user_id)
            return {
                "status": "ok",
                "dropped": "history_too_old",
                "latency_ms": _ms_now() - started_ms,
            }
        logger.exception(
            "[gmail_pubsub] history_fetch_failed user_id=%s err=%s",
            user_id[:8], str(e)[:200],
        )
        raise HTTPException(
            status_code=503, detail=f"history fetch failed: {e}"
        )

    if not msg_ids:
        # The push fired but our delta was empty — common when the
        # message was filtered out by `historyTypes=messageAdded`
        # (label changes, drafts, etc).
        if new_watermark:
            await _advance_watermark(user_id, new_watermark)
        return {
            "status": "ok",
            "dispatched": 0,
            "watermark": new_watermark or watermark,
            "latency_ms": _ms_now() - started_ms,
        }

    # 6. Dispatch to the user's agent container.
    events = [{"event_dedupe_id": mid, "external_payload": {
        "gmail_message_id": mid,
        "history_id_at_push": history_id,
    }} for mid in msg_ids]
    try:
        agent_response = await dispatch_events(
            user_id=user_id,
            trigger_kind="email_received",
            events=events,
        )
    except TriggerDispatchError as e:
        # 4xx = permanent agent rejection; surface to Pub/Sub so it
        # stops. 5xx = transient; Pub/Sub retries.
        logger.warning(
            "[gmail_pubsub] dispatch_failed user_id=%s status=%d msg=%s",
            user_id[:8], e.status_code, str(e)[:200],
        )
        raise HTTPException(status_code=e.status_code, detail=str(e))

    # 7. Advance the watermark only on successful dispatch — partial
    # failure means we want the next retry to redo this delta. The
    # agent-side dedupe gate makes redo idempotent.
    if new_watermark:
        await _advance_watermark(user_id, new_watermark)

    latency_ms = _ms_now() - started_ms
    logger.info(
        "[gmail_pubsub] ok user_id=%s history_delta=%s→%s msgs=%d "
        "agent_inserted=%s dedupe_hits=%s latency_ms=%d",
        user_id[:8], watermark, new_watermark, len(msg_ids),
        agent_response.get("inserted"),
        agent_response.get("dedupe_hits"),
        latency_ms,
    )
    return {
        "status": "ok",
        "dispatched": len(msg_ids),
        "agent": agent_response,
        "watermark": new_watermark,
        "latency_ms": latency_ms,
    }


# ── Helpers (DB-local) ───────────────────────────────────────────────


async def _resolve_user_for_email(email: str) -> str | None:
    """`connector_identities.provider_account_id == email` →
    `user_id`. Returns None if no active Gmail identity for this email
    (user disconnected or never connected)."""
    from app.db.models import ConnectorIdentity

    async with async_session_maker() as db:
        result = (await db.execute(
            select(ConnectorIdentity.user_id).where(
                ConnectorIdentity.connector_id == "gmail",
                ConnectorIdentity.provider_account_id == email,
                ConnectorIdentity.status == "active",
            )
        )).scalar_one_or_none()
    return result


async def _read_watermark(user_id: str) -> tuple[str | None, int]:
    """Pull the highest `provider_state_json.gmail_history_id` across
    this user's enabled email_received triggers, plus the count of
    enabled triggers. The platform's `triggers` table is agent-side —
    we proxy this via a direct DB call only when the agent and
    platform share a DB (monolith mode); in split-DB mode this
    function queries the agent over the bridge.

    For Gate T1 we use the direct-DB path. The bridge-proxied path is
    a small follow-up that wraps the same SQL.
    """
    try:
        from app.db.models import Trigger
    except ImportError:
        return None, 0

    async with async_session_maker() as db:
        try:
            rows = (await db.execute(
                select(Trigger.provider_state_json).where(
                    Trigger.user_id == user_id,
                    Trigger.kind == "email_received",
                    Trigger.enabled.is_(True),
                )
            )).all()
        except Exception as e:
            # Platform-mode DB doesn't have the triggers table —
            # caller will fall through to dispatch with no
            # baseline (uses push's history_id). Logged at debug
            # since this is the expected platform DB shape.
            logger.debug("[gmail_pubsub] triggers table not in this DB: %s", e)
            return None, 0
    if not rows:
        return None, 0

    history_ids: list[int] = []
    for (ps,) in rows:
        if not isinstance(ps, dict):
            continue
        hid = ps.get("gmail_history_id")
        if hid is None:
            continue
        try:
            history_ids.append(int(str(hid)))
        except ValueError:
            pass
    if not history_ids:
        return None, len(rows)
    # Resume from the LOWEST baseline — Gmail's history.list is
    # idempotent so re-fetching messages another trigger already saw
    # is fine (the agent's UNIQUE gate dedupes). Resuming from the
    # highest would silently skip events newer than one trigger's
    # cursor.
    return str(min(history_ids)), len(rows)


async def _advance_watermark(user_id: str, new_history_id: str) -> None:
    """Persist the new history_id baseline. Updates every enabled
    email_received trigger for this user — they're all consuming the
    same Gmail account so their cursors march together."""
    from sqlalchemy import update
    from app.db.models import Trigger

    async with async_session_maker() as db:
        try:
            rows = (await db.execute(
                select(Trigger.id, Trigger.provider_state_json).where(
                    Trigger.user_id == user_id,
                    Trigger.kind == "email_received",
                    Trigger.enabled.is_(True),
                )
            )).all()
            for tid, ps in rows:
                new_state = dict(ps or {})
                new_state["gmail_history_id"] = new_history_id
                await db.execute(
                    update(Trigger).where(Trigger.id == tid).values(
                        provider_state_json=new_state,
                    )
                )
            await db.commit()
        except Exception as e:
            logger.debug(
                "[gmail_pubsub] watermark advance no-op (split DB?): %s",
                str(e)[:200],
            )


async def _flag_watch_needs_refresh(user_id: str) -> None:
    """Mark every email_received trigger for this user as needing a
    watch re-arm. The refresh job picks these up on its next pass.
    """
    from sqlalchemy import update
    from app.db.models import Trigger

    async with async_session_maker() as db:
        try:
            rows = (await db.execute(
                select(Trigger.id, Trigger.provider_state_json).where(
                    Trigger.user_id == user_id,
                    Trigger.kind == "email_received",
                    Trigger.enabled.is_(True),
                )
            )).all()
            for tid, ps in rows:
                new_state = dict(ps or {})
                new_state["needs_refresh"] = True
                await db.execute(
                    update(Trigger).where(Trigger.id == tid).values(
                        provider_state_json=new_state,
                    )
                )
            await db.commit()
        except Exception as e:
            logger.debug(
                "[gmail_pubsub] needs_refresh flag no-op (split DB?): %s",
                str(e)[:200],
            )


def _ms_now() -> int:
    return int(time.perf_counter() * 1000)
