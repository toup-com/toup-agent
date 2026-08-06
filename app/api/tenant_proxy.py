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
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url.rstrip('/')}/api/{path.lstrip('/')}"
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
    except Exception as e:
        logger.warning("Agent proxy %s %s failed: %s", method, url, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach your agent. Please try again.",
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
