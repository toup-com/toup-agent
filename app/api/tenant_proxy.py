"""Platform → tenant-agent proxy for routers whose store is AGENT_ONLY.

This is the same mechanism `app/api/memories.py` and `app/api/media_playlists.py`
already use, factored out so `documents.py` and `ingest.py` can reuse it rather
than growing a third hand-rolled copy.

WHY IT EXISTS
    `documents`, `document_chunks`, `media`, `memories`, `conversations`,
    `messages` and `entities` are all in `AGENT_ONLY_TABLES`
    (`app/db/models/base.py`). Platform-mode `init_db` excludes them
    (`app/db/database.py:270-271`) and no alembic revision creates them, so on
    the platform side those tables simply are not there. A handler mounted only
    by `platform_main.py` that queries them either 500s with
    `UndefinedTableError` or — on an older platform DB that still carries the
    monolith leftovers — writes rows into a database the tenant agent never
    reads.

THE CONTRACT
    * Only the PLATFORM proxies. `agent_configs` is a SHARED table, so a tenant
      DB carrying a stray row must never make an agent proxy to itself —
      `serving_locally()` is the guard.
    * NOTHING here falls back to a local query. `memories.py` and
      `media_playlists.py` can fall back on reads because the platform DB keeps
      a `memories` row set / a playlist mirror table. For documents and media
      there is no mirror: falling back would mean either a 500 from the missing
      relation or a fabricated empty list. An unreachable tenant is reported as
      502, never as "you have no documents".
    * Agent-side 4xx answers (409 duplicate upload, 413 too large, 415
      unsupported type, 422 refused by the memory gate, 404 not found) are
      propagated verbatim. Only transport failures and agent-side 5xx collapse
      into 502.
    * READS are retried once. A write never is: a POST that timed out may well
      have landed, and a second one makes two of whatever it created.

THE THREE HAND-ROLLED COPIES, AND WHAT THEY COST (R40, 2026-08-31)
    `day_chats.py`, `sessions.py` and `messages_recover.py` predate this module
    and each grew its own proxy helper. All three broke BOTH rules above: they
    had no `serving_locally()` guard, and their helpers returned `None` for
    every failure — which the callers read as "nothing to say" and answered
    with a local SELECT against the platform DB.

    On 2026-08-31 a neighbouring tenant's container recreate saturated the
    shared VPS host, one user's agent went slow for about two minutes, and the
    day-chats proxy timed out at 10 s. The platform answered his phone:

        GET /api/day-chats                     -> 200 10099ms   (empty)
        GET /api/day-chats/2026-08-31/messages -> 200 10125ms   (empty)

    Two hundreds, both empty, both lies. The app drew "Beginning of your
    history" over an untouched account and the user reported that his messages
    had been deleted. Nothing had been.

    All three now answer 503 + `Retry-After` + `X-Toup-Reason:
    agent_unreachable`, exactly as this module has always specified for
    documents. `backend/tests/test_r40_tenant_read_honesty.py` pins it, and
    asserts that no proxy call site can reach a local SELECT after a failed
    hop. If you add a fourth router with an AGENT_ONLY store, use THIS module.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import AgentConfig

logger = logging.getLogger(__name__)

# An upload can be 500 MB of video and the agent still has to parse it, so this
# budget is much larger than the 10-25s the memories/playlists proxies use.
DEFAULT_PROXY_TIMEOUT_S = 15.0
UPLOAD_PROXY_TIMEOUT_S = 180.0
# A READ is retried once, so its ladder is what has to fit inside the mobile
# client's 15 s abort: 2 × 6 s + 0.4 s = 12.4 s worst case. Using
# DEFAULT_PROXY_TIMEOUT_S for both attempts would run to 30 s and put the
# platform back to answering after the phone has stopped listening.
READ_PROXY_TIMEOUT_S = 6.0
READ_PROXY_BACKOFF_S = 0.4


def serving_locally() -> bool:
    """True where the AGENT_ONLY tables are in THIS process's database.

    Agent containers and monolith/dev runs own the store outright; only the
    platform has to reach across the network for it.
    """
    return (settings.run_mode or "").strip().lower() in ("agent", "monolith")


async def agent_proxy_info(
    user_id: str, db: AsyncSession
) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) when this call must be proxied."""
    if serving_locally():
        return None
    try:
        async with db.begin_nested():
            row = (
                await db.execute(
                    select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
                        AgentConfig.user_id == user_id,
                        AgentConfig.deploy_status == "active",
                    )
                )
            ).first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass  # agent_configs may be absent/empty on a tenant DB
    return None


def _agent_detail(resp, default: str) -> str:
    try:
        payload = resp.json()
    except Exception:
        return default
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return default


async def proxy_to_agent(
    agent_url: str,
    agent_api_key: str,
    path: str,
    method: str = "GET",
    *,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    data: Optional[dict] = None,
    files: Optional[dict] = None,
    timeout: float = DEFAULT_PROXY_TIMEOUT_S,
) -> Any:
    """Forward one request to the tenant agent. Never falls back locally.

    `path` is appended to the agent's `/api` prefix, e.g. "ingest/document".
    Returns the decoded JSON body (or None for 204/empty). Raises
    HTTPException(502) when the tenant cannot be reached or answered 5xx, and
    re-raises the agent's own status for any 4xx.
    """
    import asyncio

    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url.rstrip('/')}/api/{path.lstrip('/')}"
    # One cheap retry, READS ONLY. The failure worth surviving is a transient
    # host stall — on 2026-08-31 a neighbouring tenant's container recreate
    # pushed one agent's response times from sub-second to ten and twenty-two
    # seconds for about two minutes, and every read against it failed once.
    # A write is never repeated: a POST that timed out may well have landed,
    # and a second one makes two of whatever it created.
    read_only = method.upper() in ("GET", "HEAD")
    attempts = 2 if read_only else 1
    # An explicit per-call `timeout=` still wins (uploads pass their own); the
    # shorter read budget applies only when the caller took the default.
    if read_only and timeout == DEFAULT_PROXY_TIMEOUT_S:
        timeout = READ_PROXY_TIMEOUT_S
    resp = None
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            client = get_agent_http_client()
            resp = await client.request(
                method,
                url,
                headers={"X-Agent-Key": agent_api_key},
                params=params or {},
                json=json_body,
                data=data,
                files=files,
                timeout=timeout,
            )
            break
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                # `repr`, not `str`: an httpx timeout stringifies to "".
                logger.info("Agent proxy %s %s attempt %d/%d failed (%r) — retrying",
                            method, url, attempt, attempts, e)
                await asyncio.sleep(READ_PROXY_BACKOFF_S)
    if resp is None:
        logger.warning("Agent proxy %s %s failed after %d attempt(s): %r",
                       method, url, attempts, last_exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach your agent. Please try again.",
            headers={"Retry-After": "3", "X-Toup-Reason": "agent_unreachable"},
        )

    if 200 <= resp.status_code < 300:
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    if 400 <= resp.status_code < 500:
        # The agent is the authority on duplicates, unsupported types, size
        # limits and gate refusals — surfacing those as 502 would tell the user
        # "your agent is down" when in fact it answered precisely.
        raise HTTPException(
            status_code=resp.status_code,
            detail=_agent_detail(resp, "Your agent rejected this request."),
        )

    logger.warning("Agent proxy %s %s returned %s", method, url, resp.status_code)
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Your agent could not process this request. Please try again.",
    )
