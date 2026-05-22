"""Platform-side proxy for `/api/triggers/*`.

Sibling to `routines_proxy.py`. The tenant agent container owns the
Triggers storage + runner — the platform DB doesn't have the
`triggers` tables. When the frontend (served from toup.ai) calls
`/api/triggers/...`, the platform's `platform_main` mounts this proxy
router, which resolves the logged-in user's `agent_url + agent_api_key`
and forwards the request.

Without this file the dashboard's Triggers panel hits the SPA's
index.html and explodes with "Unexpected token '<', '<!DOCTYPE'..."
because `/api/triggers/*` only exists on `agent_main`, not on
`platform_main`. Classic three-main.py trap.
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


router = APIRouter(prefix="/triggers", tags=["triggers (proxy)"])
logger = logging.getLogger(__name__)

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length",
}


async def _get_agent_target(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
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
        logger.warning("triggers_proxy: failed to resolve agent target for %s: %s", user_id, e)
    return None


async def _proxy(
    request: Request,
    sub_path: str,
    *,
    current_user: User,
    db: AsyncSession,
) -> Response:
    target = await _get_agent_target(current_user.id, db)
    if target is None:
        raise HTTPException(status_code=404, detail="No active agent for this user")

    agent_url, agent_api_key = target
    url = f"{agent_url.rstrip('/')}/api/triggers{sub_path}"
    headers = {
        "X-Agent-Key": agent_api_key,
        "content-type": request.headers.get("content-type", "application/json"),
        "accept": "application/json",
    }

    body = await request.body()
    method = request.method.upper()

    # TKT-LAT-007 (wave 3): shared agent_http client.
    from app.services.agent_http import get_agent_http_client

    try:
        client = get_agent_http_client()
        resp = await client.request(
            method, url,
            params=dict(request.query_params),
            headers=headers,
            content=body if body else None,
            timeout=30.0,
        )
    except httpx.RequestError as e:
        logger.warning("triggers_proxy %s %s failed: %s", method, url, e)
        raise HTTPException(status_code=502, detail="Agent unreachable")

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


@router.patch("/{trigger_id}")
async def proxy_update(
    trigger_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{trigger_id}", current_user=current_user, db=db)


@router.delete("/{trigger_id}")
async def proxy_delete(
    trigger_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{trigger_id}", current_user=current_user, db=db)


@router.get("/{trigger_id}/events")
async def proxy_events(
    trigger_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{trigger_id}/events", current_user=current_user, db=db)


@router.post("/{trigger_id}/test")
async def proxy_test(
    trigger_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _proxy(request, f"/{trigger_id}/test", current_user=current_user, db=db)
