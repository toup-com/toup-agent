"""Autopilot proxy — platform forwards Mission Control's autopilot
calls to the user's tenant agent (Autopilot PR8).

Missions/approvals live in AGENT_ONLY tables; the platform can never
SQL-read them (split-DB rule). Same recipe as apps_proxy: resolve
AgentConfig (agent_url + agent_api_key, deploy_status='active'),
forward with X-Agent-Key, pass the query string (the /jobs/events
lesson), JWT auth platform-side via get_current_user.

Action segments are ALLOWLISTED — this is not an open proxy into the
agent's API surface.

Mounted in platform_main (never agent_main / app.main).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import AgentConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/autopilot", tags=["Autopilot Proxy"])

_MISSION_ACTIONS = frozenset({"pause", "resume", "cancel"})


async def _agent_info(user_id: str, db: AsyncSession) -> Tuple[str, str]:
    result = await db.execute(
        select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
            AgentConfig.user_id == user_id,
            AgentConfig.deploy_status == "active",
        )
    )
    row = result.first()
    if not row or not row.agent_url or not row.agent_api_key:
        raise HTTPException(503, "Agent not available")
    return row.agent_url.rstrip("/"), row.agent_api_key


async def _forward(
    request: Request,
    user_id: str,
    db: AsyncSession,
    path: str,
    method: str = "GET",
    body: Optional[dict] = None,
) -> JSONResponse:
    from app.services.agent_http import get_agent_http_client

    agent_url, key = await _agent_info(user_id, db)
    qs = f"?{request.url.query}" if request.url.query else ""
    url = f"{agent_url}/api/autopilot/{path}{qs}"
    try:
        client = get_agent_http_client()
        headers = {"X-Agent-Key": key}
        if method == "GET":
            resp = await client.get(url, headers=headers, timeout=15.0)
        else:
            resp = await client.post(url, headers=headers, json=body or {}, timeout=15.0)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — agent recycling / cold start
        logger.warning("autopilot proxy %s %s failed: %s", method, url, e)
        raise HTTPException(502, "Agent unreachable")


@router.get("/missions")
async def list_missions_proxy(
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _forward(request, current_user.id, db, "missions")


@router.post("/missions")
async def create_mission_proxy(
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    body = await request.json()
    return await _forward(request, current_user.id, db, "missions",
                          method="POST", body=body)


@router.get("/missions/{mission_id}")
async def get_mission_proxy(
    mission_id: str,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _forward(request, current_user.id, db, f"missions/{mission_id}")


@router.post("/missions/{mission_id}/{action}")
async def mission_action_proxy(
    mission_id: str,
    action: str,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if action not in _MISSION_ACTIONS:
        raise HTTPException(404, "Unknown action")
    return await _forward(
        request, current_user.id, db, f"missions/{mission_id}/{action}",
        method="POST",
    )


@router.get("/approvals")
async def list_approvals_proxy(
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _forward(request, current_user.id, db, "approvals")


@router.post("/approvals/{approval_id}/decision")
async def decide_approval_proxy(
    approval_id: str,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    body = await request.json()
    return await _forward(
        request, current_user.id, db, f"approvals/{approval_id}/decision",
        method="POST", body=body,
    )
