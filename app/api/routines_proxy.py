"""Platform-side proxy for `/api/routines/*`.

The agent container owns the Routines storage + scheduler — the platform
DB doesn't have the `routines` tables. When the frontend (served from
toup.ai) calls `/api/routines/...`, the platform's `app/api/__init__`
mounts this proxy router. We resolve the logged-in user, look up their
`agent_url + agent_api_key`, forward the request, return the response.

Same pattern `messages_recover.py` uses for `/api/messages/since`. The
agent container's `routines.py` does the real work.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.auth import get_current_user
from app.db.database import get_db
from app.db.models import User


router = APIRouter(prefix="/routines", tags=["routines (proxy)"])
logger = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 7230 §6.1) that must NOT be forwarded.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length",
}


async def _get_agent_target(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
    """Resolve (agent_url, agent_api_key) for the logged-in user.

    Mirrors `messages_recover._get_agent_proxy_info`. Local copy avoids
    cross-import + keeps the swallow-on-missing-table semantics scoped
    to this file (platform DB is the source of truth for AgentConfig).
    """
    try:
        from app.db.models import AgentConfig
        result = await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
                AgentConfig.user_id == user_id,
                AgentConfig.deploy_status == "active",
            )
        )
        row = result.first()
        if row and row.agent_url and row.agent_api_key:
            return (row.agent_url, row.agent_api_key)
    except Exception as e:
        logger.warning("routines_proxy: failed to resolve agent target for %s: %s", user_id, e)
    return None


async def _proxy(
    request: Request,
    sub_path: str,
    *,
    current_user: User,
    db: AsyncSession,
) -> Response:
    """Forward the request to the user's agent, return the response.

    Preserves status code, body, and JSON content-type. Avoids leaking
    hop-by-hop headers; injects X-Agent-Key from the tenant config.
    """
    target = await _get_agent_target(current_user.id, db)
    if target is None:
        # No active agent → treat as 'feature not available' so the
        # frontend renders the empty state cleanly instead of a 5xx.
        raise HTTPException(status_code=404, detail="No active agent for this user")

    agent_url, agent_api_key = target
    url = f"{agent_url.rstrip('/')}/api/routines{sub_path}"
    headers = {
        "X-Agent-Key": agent_api_key,
        "content-type": request.headers.get("content-type", "application/json"),
        "accept": "application/json",
    }

    body = await request.body()
    method = request.method.upper()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                method, url,
                params=dict(request.query_params),
                headers=headers,
                content=body if body else None,
            )
    except httpx.RequestError as e:
        logger.warning("routines_proxy %s %s failed: %s", method, url, e)
        raise HTTPException(status_code=502, detail="Agent unreachable")

    # Filter hop-by-hop response headers
    out_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=out_headers,
        media_type=resp.headers.get("content-type"),
    )


# ── Routes ─────────────────────────────────────────────────────────────
# Catch-all forwarder. We list each method+path explicitly (rather than a
# single `/{path:path}`) so the OpenAPI surface mirrors the agent's
# routines.py — easier to debug from the API docs.


@router.get("/_runner_status")
async def proxy_runner_status(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "/_runner_status", current_user=current_user, db=db)


@router.get("")
async def proxy_list(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "", current_user=current_user, db=db)


@router.post("")
async def proxy_create(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, "", current_user=current_user, db=db)


@router.patch("/{routine_id}")
async def proxy_update(
    routine_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{routine_id}", current_user=current_user, db=db)


@router.delete("/{routine_id}")
async def proxy_delete(
    routine_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{routine_id}", current_user=current_user, db=db)


@router.post("/{routine_id}/run")
async def proxy_run(
    routine_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{routine_id}/run", current_user=current_user, db=db)


@router.get("/{routine_id}/runs")
async def proxy_runs(
    routine_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{routine_id}/runs", current_user=current_user, db=db)
