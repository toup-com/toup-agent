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
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status
from sqlalchemy import select, update

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

    # Decode the body BEFORE JWT verify so we can record
    # `last_push_received_at` on the matching ConnectorIdentity even
    # when JWT verification subsequently fails (audience mismatch /
    # expired token / forged push). That signal is what tells the
    # operator "Pub/Sub IS pushing, the issue is downstream" — exactly
    # the diagnostic that was missing in the previous fix and a
    # prerequisite for the `/probe` endpoint's `push_rejected_at_platform`
    # diagnosis.
    #
    # We pre-fetch the JSON here; the verify_pubsub_jwt + decode_pubsub_
    # envelope blocks below still own the actual validation. A forged
    # body whose envelope happens to decode to a real email address
    # will get its timestamp updated — that's intentional: the timestamp
    # only proves "something hit this endpoint claiming to be for X",
    # which is the signal we want for diagnosis. Real ops follow-up
    # uses the JWT verify outcome (logged separately) to distinguish
    # legitimate from forged pushes.
    body: dict[str, Any] = {}
    pre_decoded_email: Optional[str] = None
    try:
        body = await request.json()
    except Exception:
        # Defer the 400 until after the JWT verify so we don't leak
        # "we accept anything as long as it has a Bearer token" — but
        # we have no body so nothing to record; skip and let the decode
        # block below raise 400.
        body = {}
    if body:
        try:
            pre_decoded_email, _pre_history_id = decode_pubsub_envelope(body)
        except ValueError:
            pre_decoded_email = None
    if pre_decoded_email:
        try:
            await _record_push_received(pre_decoded_email)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "[gmail_pubsub] record_push_received failed email=%s err=%s",
                (pre_decoded_email or "")[:32], e,
            )

    # 1. JWT verify. Pub/Sub uses `Authorization: Bearer <JWT>`.
    try:
        claims = await verify_pubsub_jwt(authorization)
    except PubSubAuthError as e:
        logger.warning("[gmail_pubsub] jwt_verify_failed reason=%s", str(e)[:200])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"pubsub auth failed: {e}",
        )

    # 2. Decode envelope (re-run so the validation branches below
    # behave exactly as before — pre-decode above is observability-only).
    if not body:
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

    # 4. Look up the watermark. Single source of truth is
    # `ConnectorIdentity.metadata_json.gmail_history_id` on the
    # platform DB — written when the OAuth callback armed the watch,
    # refreshed daily, and advanced on every successful dispatch
    # here. Triggers themselves live agent-side; using the connector
    # row decouples watermark from trigger existence and avoids the
    # cross-DB lookup the prior code attempted (which silently
    # returned (None, 0) and dropped every push as no_active_trigger
    # in split-DB mode).
    watermark = await _read_watermark_from_connector(user_id, email)
    if not watermark:
        # First push after watch arm — use the push's history_id as
        # the baseline so we don't replay arbitrary history. Whoever
        # armed the watch (OAuth callback / provision RPC / daily
        # refresh) should have written one already; if we land here
        # it means the arm event landed BEFORE we wrote the metadata
        # row (race), or this is the first push after a manual REPL
        # arm. Either way, history_id from the envelope is the
        # correct baseline.
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
            await _flag_connector_needs_refresh(user_id, email)
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
            await _advance_watermark_on_connector(user_id, email, new_watermark)
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
        await _advance_watermark_on_connector(user_id, email, new_watermark)
    # Trigger rows live on the per-tenant agent DB in split-DB mode,
    # not on the platform DB. The historical `_stamp_triggers_dispatched`
    # call queried `triggers` here and produced a warning on every push
    # ("relation triggers does not exist"). The agent's
    # `triggers/inbound` handler already stamps `last_fired_at` on the
    # Trigger row when it inserts a TriggerEvent — that's the source of
    # truth for `/probe`. No platform-side stamp needed.

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


async def _read_watermark_from_connector(
    user_id: str, email: str,
) -> str | None:
    """Pull `gmail_history_id` from this user's Gmail
    `ConnectorIdentity.metadata_json`. None means "no baseline yet"
    — caller falls through to the push's own history_id, which is
    the right behaviour for a freshly-armed watch.

    Lives on the platform DB — same database the OAuth callback +
    provision RPC + daily refresh write to. No cross-DB lookup, no
    silent drop on split-DB mode.
    """
    from app.db.models import ConnectorIdentity

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity.metadata_json).where(
                ConnectorIdentity.user_id == user_id,
                ConnectorIdentity.connector_id == "gmail",
                ConnectorIdentity.provider_account_id == email,
            )
        )).first()
    if row is None or not row[0]:
        return None
    try:
        import json
        meta = json.loads(row[0])
        if not isinstance(meta, dict):
            return None
        hid = meta.get("gmail_history_id")
        return str(hid) if hid is not None else None
    except (ValueError, TypeError) as e:
        logger.warning(
            "[gmail_pubsub] metadata_json parse failed user=%s err=%s",
            user_id[:8], e,
        )
        return None


async def _advance_watermark_on_connector(
    user_id: str, email: str, new_history_id: str,
) -> None:
    """Persist the new baseline on `ConnectorIdentity.metadata_json`.
    No-op if the identity row vanished between push and update
    (user disconnected mid-flight — Pub/Sub will stop on the next
    push when email→user resolution misses)."""
    import json
    from sqlalchemy import update
    from app.db.models import ConnectorIdentity

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == user_id,
                ConnectorIdentity.connector_id == "gmail",
                ConnectorIdentity.provider_account_id == email,
            )
        )).scalar_one_or_none()
        if row is None:
            return
        try:
            meta = json.loads(row.metadata_json or "{}")
            if not isinstance(meta, dict):
                meta = {}
        except (ValueError, TypeError):
            meta = {}
        meta["gmail_history_id"] = str(new_history_id)
        meta.pop("needs_refresh", None)             # advancing resets the flag
        await db.execute(
            update(ConnectorIdentity)
            .where(ConnectorIdentity.id == row.id)
            .values(metadata_json=json.dumps(meta))
        )
        await db.commit()


async def _flag_connector_needs_refresh(user_id: str, email: str) -> None:
    """Mark this Gmail identity for re-arm on the next refresh pass.
    Set when `history.list` returns `historyId too old` — the watch
    is still live but the cursor is past Gmail's retention window
    (typically ~7 days). Re-arming generates a fresh baseline."""
    import json
    from sqlalchemy import update
    from app.db.models import ConnectorIdentity

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == user_id,
                ConnectorIdentity.connector_id == "gmail",
                ConnectorIdentity.provider_account_id == email,
            )
        )).scalar_one_or_none()
        if row is None:
            return
        try:
            meta = json.loads(row.metadata_json or "{}")
            if not isinstance(meta, dict):
                meta = {}
        except (ValueError, TypeError):
            meta = {}
        meta["needs_refresh"] = True
        await db.execute(
            update(ConnectorIdentity)
            .where(ConnectorIdentity.id == row.id)
            .values(metadata_json=json.dumps(meta))
        )
        await db.commit()


def _ms_now() -> int:
    return int(time.perf_counter() * 1000)


def _utc_iso_z() -> str:
    """ISO-8601 UTC string with a `Z` suffix. Matches the format used
    by `_utc_iso` in triggers.py so frontend parsing is consistent."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def _record_push_received(email: str) -> None:
    """Stamp `last_push_received_at` on the matching ConnectorIdentity.

    Called BEFORE JWT verification so even a forged / mis-audienced
    push leaves a footprint the operator can use to confirm "Pub/Sub
    is reaching us; the issue is JWT/audience/IAM". The `/probe`
    endpoint reads this timestamp to compute its diagnosis.

    Idempotent: every push overwrites with a fresher timestamp.
    Missing connector identity → no-op (forged push or stale
    subscription).
    """
    import json as _json
    from app.db.models import ConnectorIdentity

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.connector_id == "gmail",
                ConnectorIdentity.provider_account_id == email,
            )
        )).scalar_one_or_none()
        if row is None:
            return
        try:
            meta = _json.loads(row.metadata_json or "{}")
            if not isinstance(meta, dict):
                meta = {}
        except (ValueError, TypeError):
            meta = {}
        meta["last_push_received_at"] = _utc_iso_z()
        await db.execute(
            update(ConnectorIdentity)
            .where(ConnectorIdentity.id == row.id)
            .values(metadata_json=_json.dumps(meta))
        )
        await db.commit()


