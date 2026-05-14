"""Platform-side watch-provisioning RPC for tenant agents.

The agent's `triggers__create` (kind=email_received) needs Gmail's
`users.watch` to be armed before any event can fire. The Gmail watch
call requires a refresh token, which is platform-only (architecture
guarantee — agent containers never see refresh tokens). So the agent
calls this endpoint, and the platform does the work.

Auth contract — symmetric to `triggers_inbound` but reversed:
  • Agent POSTs `X-Agent-Key: <its agent_api_key>` + body `{user_id}`.
  • Platform verifies the key matches `AgentConfig.agent_api_key` AND
    `AgentConfig.user_id == body.user_id`. Both wrong = forged or
    routing bug; 401.

Endpoint surface:
  POST /api/v1/triggers/_provision_watch
       Arms (or re-arms) a Gmail watch for the calling tenant.
       Writes the new historyId + expiration to
       `ConnectorIdentity.metadata_json` so the webhook's watermark
       reader has a baseline regardless of whether triggers exist on
       the platform DB.
       Returns the provisioned state so the agent can surface it.

  GET  /api/v1/triggers/_watch_status?connector_id=gmail
       Read-only: what's the current watch state for this tenant's
       Gmail account? The agent uses this when listing triggers so
       the UI's "watch_provisioned" flag is honest about the live
       state, not whatever the last cached agent-side write said.

Failure modes returned in `error`:
  - "needs_reauth"        — no active Gmail identity / refresh token
  - "ops_blocked"         — GCP project/topic unset platform-side
  - "scope_error"         — Gmail rejected the watch call (4xx)
  - "transient"           — 5xx / network — caller may retry

Returning HTTP 200 even on application failures (with `error` set) is
intentional: this is an agent-to-platform RPC, not an end-user
endpoint. The agent surfaces the structured outcome to the chat
agent verbatim; failing with HTTP 5xx would mask the underlying
reason behind "platform unreachable."
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from app.db.database import async_session_maker
from app.db.models import AgentConfig, ConnectorIdentity


router = APIRouter(prefix="/v1/triggers", tags=["triggers (provision)"])
logger = logging.getLogger(__name__)


class ProvisionWatchReq(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=36)
    connector_id: str = Field(default="gmail", min_length=1, max_length=64)


class ProvisionWatchResponse(BaseModel):
    provisioned: bool
    history_id: Optional[str] = None
    expires_at: Optional[str] = None
    error: Optional[str] = None          # see "Failure modes" in module doc
    detail: Optional[str] = None         # human-readable, safe to log


class WatchStatusResponse(BaseModel):
    provisioned: bool
    history_id: Optional[str] = None
    expires_at: Optional[str] = None
    # gmail_connector_present is what differentiates "Gmail isn't
    # connected at all" from "Gmail connected but watch isn't armed yet."
    # The agent uses this to pick the right user-facing message.
    gmail_connector_present: bool = False


async def _auth_agent(
    x_agent_key: Optional[str], body_user_id: str,
) -> AgentConfig:
    """Resolve and validate the calling tenant. Returns the AgentConfig
    row so the caller has access to user_id + agent_url etc. without a
    second query. 401 on mismatch — matches the heartbeat pattern in
    extension.py."""
    if not x_agent_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Agent-Key required",
        )
    async with async_session_maker() as db:
        row = (await db.execute(
            select(AgentConfig).where(
                AgentConfig.user_id == body_user_id,
                AgentConfig.agent_api_key == x_agent_key,
            )
        )).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="agent key / user_id mismatch",
        )
    return row


@router.post("/_provision_watch", response_model=ProvisionWatchResponse)
async def provision_watch(
    req: ProvisionWatchReq,
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
) -> ProvisionWatchResponse:
    """Arm (or re-arm) the user's Gmail watch and persist the new
    baseline. Idempotent at the Gmail end — a fresh `users.watch` over
    a live one just resets the expiration clock."""
    await _auth_agent(x_agent_key, req.user_id)

    # Only Gmail is supported in v1. The kind enum will gain more
    # values; the actual provisioning differs per provider, so we keep
    # the dispatch explicit instead of metaprogramming.
    if req.connector_id != "gmail":
        return ProvisionWatchResponse(
            provisioned=False,
            error="unsupported_connector",
            detail=f"connector_id={req.connector_id!r} not supported",
        )

    # Cheap fail-fast: if the platform doesn't have GCP creds, we'll
    # never be able to arm — surface that as ops_blocked immediately
    # instead of waiting for start_watch to fail.
    from app.config import settings
    if not settings.gcp_project or not settings.pubsub_topic:
        logger.warning(
            "[trigger_provision] ops_blocked user=%s gcp_project=%r topic=%r",
            req.user_id[:8], bool(settings.gcp_project), bool(settings.pubsub_topic),
        )
        return ProvisionWatchResponse(
            provisioned=False,
            error="ops_blocked",
            detail="GCP project or Pub/Sub topic not configured on platform",
        )
    if not getattr(settings, "triggers_email_enabled", True):
        return ProvisionWatchResponse(
            provisioned=False,
            error="feature_disabled",
            detail="triggers_email_enabled is off",
        )

    # Call the existing watch arm. Maps known failures to structured
    # errors so the agent can surface a specific user-facing message.
    from app.services.gmail_pubsub import start_watch, GmailWatchError
    try:
        handle = await start_watch(req.user_id)
    except GmailWatchError as e:
        msg = str(e)
        if "no active gmail identity" in msg.lower():
            return ProvisionWatchResponse(
                provisioned=False, error="needs_reauth",
                detail="Gmail is not connected (or refresh token is gone).",
            )
        # Token-mint failure — typically expired refresh token.
        if "token refresh failed" in msg.lower():
            return ProvisionWatchResponse(
                provisioned=False, error="needs_reauth",
                detail="Gmail refresh token is rejected — reconnect Gmail.",
            )
        # GCP misconfig caught earlier; if we get here it's an upstream
        # 4xx from Gmail — typically scope or topic IAM. Surface for ops.
        if "status=4" in msg:
            return ProvisionWatchResponse(
                provisioned=False, error="scope_error", detail=msg[:300],
            )
        logger.exception(
            "[trigger_provision] watch_start_failed user=%s", req.user_id[:8],
        )
        return ProvisionWatchResponse(
            provisioned=False, error="transient", detail=msg[:300],
        )

    # Persist watermark + expiration on ConnectorIdentity.metadata_json.
    # This is the single source of truth the webhook reads — independent
    # of agent-side trigger rows.
    expires_iso = datetime.fromtimestamp(
        handle.expiration_unix_ms / 1000, tz=timezone.utc,
    ).isoformat()
    await _write_watch_state(
        user_id=req.user_id,
        connector_id=req.connector_id,
        history_id=handle.history_id,
        expires_at_iso=expires_iso,
    )

    logger.info(
        "[trigger_provision] armed user=%s history_id=%s expires=%s",
        req.user_id[:8], handle.history_id, expires_iso,
    )
    return ProvisionWatchResponse(
        provisioned=True,
        history_id=handle.history_id,
        expires_at=expires_iso,
    )


@router.get("/_watch_status", response_model=WatchStatusResponse)
async def watch_status(
    user_id: str,
    connector_id: str = "gmail",
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
) -> WatchStatusResponse:
    """Cheap read of the current watch state. The agent uses this on
    `triggers__list` so the UI's `watch_provisioned` flag reflects
    live truth (not whatever was cached when the trigger row was last
    written)."""
    await _auth_agent(x_agent_key, user_id)

    async with async_session_maker() as db:
        ident = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == user_id,
                ConnectorIdentity.connector_id == connector_id,
            )
        )).scalar_one_or_none()

    if ident is None:
        return WatchStatusResponse(
            provisioned=False, gmail_connector_present=False,
        )
    # A revoked / provider_down identity = no live watch regardless of
    # cached state.
    if ident.status not in ("active", "reauth_required"):
        return WatchStatusResponse(
            provisioned=False, gmail_connector_present=True,
        )
    meta = _parse_metadata(ident.metadata_json)
    history_id = meta.get("gmail_history_id")
    expires_at = meta.get("gmail_watch_expires_at")

    provisioned = bool(history_id and expires_at and not _expired(expires_at))
    return WatchStatusResponse(
        provisioned=provisioned,
        history_id=history_id,
        expires_at=expires_at,
        gmail_connector_present=True,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_metadata(raw: Optional[str]) -> dict[str, Any]:
    """`ConnectorIdentity.metadata_json` is Text-encoded JSON (Postgres
    JSONB would be nicer; spec is Text for SQLite-test parity). Treat
    parse failure as empty — we never crash on a malformed row."""
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}


def _expired(iso: str) -> bool:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return True
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt <= now


async def _write_watch_state(
    *, user_id: str, connector_id: str,
    history_id: str, expires_at_iso: str,
) -> None:
    """Merge gmail_history_id + gmail_watch_expires_at into the
    ConnectorIdentity row's metadata_json. Preserves any other keys the
    OAuth flow stashed there."""
    async with async_session_maker() as db:
        ident = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == user_id,
                ConnectorIdentity.connector_id == connector_id,
            )
        )).scalar_one_or_none()
        if ident is None:
            logger.warning(
                "[trigger_provision] write_watch_state user=%s connector=%s "
                "— identity not found, skipping metadata write",
                user_id[:8], connector_id,
            )
            return
        meta = _parse_metadata(ident.metadata_json)
        meta["gmail_history_id"] = str(history_id)
        meta["gmail_watch_expires_at"] = expires_at_iso
        meta["gmail_watch_armed_at"] = datetime.now(timezone.utc).isoformat()
        await db.execute(
            update(ConnectorIdentity)
            .where(ConnectorIdentity.id == ident.id)
            .values(metadata_json=json.dumps(meta))
        )
        await db.commit()
