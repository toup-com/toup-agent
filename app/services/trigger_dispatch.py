"""Platform → per-tenant agent dispatch for trigger events.

The platform's Pub/Sub webhook authenticates the push, resolves the
user, pulls new gmail_message_ids from history.list, then calls each
user's per-tenant agent container's `/api/triggers/inbound` endpoint.

Why platform calls agent (not the reverse): tokens live platform-side,
the per-tenant trigger config + event audit trail lives agent-side.
Dispatching this way keeps the isolation guarantee — agent containers
never see refresh tokens, platform never directly writes to
agent-owned tables.

Auth: X-Agent-Key header, value = `agent_configs.agent_api_key` for
the recipient. The agent's existing X-Agent-Key middleware validates
against `settings.agent_api_key` (the env-var copy of the same key);
mismatch returns 401.

Failure model:
  - Transport error (timeout, refused, 5xx): raise; the webhook
    returns 503. Pub/Sub WILL retry — on the next retry the agent's
    UNIQUE(trigger_id, event_dedupe_id) gate dedupes.
  - 4xx from agent (401 = wrong key, 422 = malformed envelope):
    raise; webhook returns the same 4xx upstream. Pub/Sub doesn't
    retry on 4xx — the event dies, which is correct for permanent
    failures (a wrong tenant key needs an ops fix, not a retry storm).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from sqlalchemy import select

from app.db.database import async_session_maker

logger = logging.getLogger(__name__)


class TriggerDispatchError(RuntimeError):
    """Raised when we couldn't deliver to the agent. `status_code`
    distinguishes transport (5xx) from contract (4xx) failures so
    the webhook can choose the right response status."""

    def __init__(self, message: str, *, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


async def resolve_agent_target(user_id: str) -> tuple[str, str]:
    """Return `(agent_url, agent_api_key)` for the tenant. Raises
    TriggerDispatchError with status 404 if no active agent —
    that's a "no recipient" signal the webhook surfaces as a
    successful 200 (the push was valid, just nowhere to send)."""
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        row = (await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
                AgentConfig.user_id == user_id,
                AgentConfig.deploy_status == "active",
            )
        )).first()
    if row is None or not row.agent_url or not row.agent_api_key:
        raise TriggerDispatchError(
            f"no active agent for user_id={user_id[:8]}…",
            status_code=404,
        )
    return row.agent_url, row.agent_api_key


async def dispatch_events(
    *,
    user_id: str,
    trigger_kind: str,
    events: list[dict[str, Any]],
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Send the events to the user's agent container.

    `events` is a list of `{event_dedupe_id, external_payload?}`
    dicts — same shape as the agent endpoint's pydantic model.

    Returns the agent's `TriggerInboundResponse` body verbatim so the
    webhook's structured log can pin down per-event outcomes.

    Raises TriggerDispatchError on transport or contract failure.
    """
    agent_url, agent_key = await resolve_agent_target(user_id)
    url = f"{agent_url.rstrip('/')}/api/triggers/inbound"
    payload = {
        "trigger_kind": trigger_kind,
        "user_id": user_id,
        "events": events,
    }
    headers = {
        "X-Agent-Key": agent_key,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.RequestError as e:
        # Transport-level failure: agent down, network blip. Webhook
        # responds 503 so Pub/Sub retries (dedupe gate at agent side
        # collapses the retry).
        raise TriggerDispatchError(
            f"dispatch transport error: {type(e).__name__}: {e}",
            status_code=503,
        ) from e

    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            return {"accepted_events": len(events), "raw": r.text[:500]}

    # 4xx from agent → contract violation (wrong key, malformed body,
    # cross-tenant routing bug). Forward the status so Pub/Sub stops
    # retrying.
    if 400 <= r.status_code < 500:
        raise TriggerDispatchError(
            f"agent rejected dispatch status={r.status_code} "
            f"body={r.text[:200]}",
            status_code=r.status_code,
        )
    # 5xx from agent → transient; let Pub/Sub retry.
    raise TriggerDispatchError(
        f"agent dispatch failed status={r.status_code} "
        f"body={r.text[:200]}",
        status_code=503,
    )
