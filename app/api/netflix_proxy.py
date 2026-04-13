"""
Netflix Stream Proxy — proxies HLS from user's VPS agent to browser.

Routes:
  POST /api/netflix-stream/start → proxy to agent, returns stream info
  POST /api/netflix-stream/stop  → proxy to agent
  GET  /api/netflix-stream/{stream_id}/{filename} → proxy HLS segments
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.api.auth import get_current_user
from app.db.models import AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/netflix-stream", tags=["Netflix Stream Proxy"])


async def _get_agent_info(user_id: str, db: AsyncSession):
    """Get agent URL and API key for a user."""
    try:
        result = await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key)
            .where(AgentConfig.user_id == user_id)
        )
        row = result.first()
    except Exception:
        raise HTTPException(status_code=404, detail="No active agent")
    if not row or not row.agent_url or not row.agent_api_key:
        raise HTTPException(status_code=404, detail="No active agent")
    return row.agent_url, row.agent_api_key


@router.post("/start")
async def proxy_start(
    request: Request,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start Netflix stream — proxy to user's agent."""
    agent_url, agent_key = await _get_agent_info(user.id, db)
    body = await request.json()

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{agent_url}/api/netflix-stream/start",
            json=body,
            headers={"X-Agent-Key": agent_key},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    # Rewrite HLS URL to go through platform proxy
    data["hls_url"] = f"/api/netflix-stream/{data['stream_id']}/stream.m3u8"
    data["_agent_url"] = agent_url  # for segment proxy
    return data


@router.post("/stop")
async def proxy_stop(
    request: Request,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stop Netflix stream — proxy to user's agent."""
    agent_url, agent_key = await _get_agent_info(user.id, db)
    body = await request.json()

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{agent_url}/api/netflix-stream/stop",
            params={"stream_id": body.get("stream_id", "")},
            headers={"X-Agent-Key": agent_key},
        )
    return resp.json()


@router.get("/{stream_id}/{filename}")
async def proxy_hls(
    stream_id: str,
    filename: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Proxy HLS segments from user's agent."""
    agent_url, agent_key = await _get_agent_info(user.id, db)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{agent_url}/api/netflix-stream/{stream_id}/{filename}",
            headers={"X-Agent-Key": agent_key},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Segment not found")

    content_type = "application/vnd.apple.mpegurl" if filename.endswith(".m3u8") else "video/MP2T"
    cache = "no-cache" if filename.endswith(".m3u8") else "max-age=3600"

    return Response(
        content=resp.content,
        media_type=content_type,
        headers={
            "Cache-Control": cache,
            "Access-Control-Allow-Origin": "*",
        },
    )
