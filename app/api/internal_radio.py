"""Internal RPC: tenant agents fetch YouTube-Music data via the platform.

Per-tenant agents run on Contabo VPS IPs that YouTube Music anti-bot-blocks,
and the agent containers have NO egress proxy (YT_DLP_PROXY is not in
AgentEnvContract). So `ytmusicapi.get_watch_playlist` / `search` return empty
from the agent → the radio toggle is rejected with `station_unavailable`, even
for a perfectly valid Song seed.

The PLATFORM, by contrast, has a working residential proxy (settings.yt_dlp_proxy
→ the Tailscale exit node, the same path yt-dlp audio extraction already uses).
Verified: the platform's proxied egress IP is residential and music.youtube.com
returns 200 through it.

So agents POST their YT-Music calls here; the platform runs them through the
proxy and returns the RAW ytmusicapi result. The agent parses it exactly as if
it had called ytmusicapi locally (see app/agent/radio/playlist.py `_yt_remote`).

Auth: `X-Agent-Key` header matched against `AgentConfig.agent_api_key` for the
posted `user_id` — the same pattern as triggers_provision.py.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, List, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import AgentConfig

router = APIRouter(prefix="/internal/radio", tags=["internal (radio)"])
logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,16}$")


class YtReq(BaseModel):
    user_id: str = Field(..., min_length=8, max_length=36)
    op: str = Field(..., description="get_watch_playlist | search")
    video_id: Optional[str] = None
    query: Optional[str] = None
    filter: Optional[str] = None
    limit: Optional[int] = None
    # get_watch_playlist only: request YT Music's own "Start Radio" variant
    # (params=wAEB), which varies its picks per call — the lever that makes a
    # repeated "play me Drake" build a FRESH station instead of the same 50
    # tracks in the same order.
    radio: Optional[bool] = None


class WarmReq(BaseModel):
    user_id: str = Field(..., min_length=8, max_length=36)
    video_ids: List[str] = Field(..., max_length=3)
    # "build"   — download + remux the whole track (UPCOMING tracks only; it
    #             saturates the residential proxy, so aiming it at the track
    #             currently playing would starve that track's own stream).
    # "extract" — yt-dlp metadata handshake only, no media bytes. Safe for the
    #             track playing RIGHT NOW, which is where the latency is.
    # Defaults to "build" so an older agent image keeps its exact behaviour.
    mode: str = Field(default="build", pattern="^(build|extract)$")


async def _auth_agent(x_agent_key: Optional[str], user_id: str) -> None:
    """401 unless X-Agent-Key matches the AgentConfig row for user_id."""
    if not x_agent_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "X-Agent-Key required")
    async with async_session_maker() as db:
        row = (await db.execute(
            select(AgentConfig).where(
                AgentConfig.user_id == user_id,
                AgentConfig.agent_api_key == x_agent_key,
            )
        )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "agent key / user_id mismatch")


@router.post("/yt")
async def yt(req: YtReq, x_agent_key: Optional[str] = Header(default=None)) -> dict:
    """Run a single ytmusicapi call through the platform's proxy on behalf of
    an agent. Returns {"result": <raw ytmusicapi result>}. A 5xx here tells the
    agent to fall back to its own (direct, likely-blocked) call — no regression.
    """
    await _auth_agent(x_agent_key, req.user_id)

    # _ytmusic() reads settings.yt_dlp_proxy, which IS set on the platform, so
    # these calls egress through the residential exit node.
    from app.agent.radio.playlist import _ytmusic

    ytm = _ytmusic()
    op = (req.op or "").strip()
    try:
        if op == "get_watch_playlist":
            if not req.video_id:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "video_id required")
            result: Any = await asyncio.to_thread(
                lambda: ytm.get_watch_playlist(
                    videoId=req.video_id, limit=req.limit or 50, radio=bool(req.radio),
                )
            )
        elif op == "search":
            if not req.query:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "query required")
            result = await asyncio.to_thread(
                lambda: ytm.search(req.query, filter=req.filter or None, limit=req.limit or 5) or []
            )
        else:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown op {op!r}")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[internal/radio] yt op=%s failed: %s: %s", op, type(e).__name__, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"ytmusic failed: {type(e).__name__}")

    return {"result": result}


# Separate router because the warm endpoint is media-scoped, not radio-scoped
# (/api/internal/media/warm). Registered alongside internal_radio_router in
# platform_main.
media_warm_router = APIRouter(prefix="/internal/media", tags=["internal (media)"])


@media_warm_router.post("/warm")
async def warm(req: WarmReq, x_agent_key: Optional[str] = Header(default=None)) -> dict:
    """Pre-build the audio remux (L1/L2 cache) for up to 3 tracks.

    Called by tenant agents at media_play broadcast time (radio/player.py
    `warm_audio_cache`) so the phone's native request — which lands within a
    second or two on the audio-first app — hits a build already in flight or
    done instead of a cold yt-dlp extraction. The build path is deduped
    (`_remux_tasks`) and concurrency-capped (`_build_sem`), so this can never
    stampede the residential proxy; a warm for an already-cached track is a
    no-op.
    """
    await _auth_agent(x_agent_key, req.user_id)
    try:
        from app.api.media_proxy import _ensure_extract_bg, _ensure_remux_bg, _SPOOL_ENABLED
    except Exception as e:  # media proxy unavailable in this process shape
        logger.warning("[internal/media] warm unavailable: %s", e)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "media proxy unavailable")
    # `spool` advertises the single-flight spool capability (2026-08-09).
    # Agents read it off the ack to decide whether a NOW-PLAYING build warm is
    # safe (radio/player.py::warm_audio_cache) — on a pre-spool image such a
    # build is a duplicate download racing the live stream. Older platform
    # images simply omit the key, which agents treat as "no".
    ids = [v for v in req.video_ids if isinstance(v, str) and _VIDEO_ID_RE.match(v)][:3]
    if not ids:
        return {"ok": True, "warmed": 0, "spool": bool(_SPOOL_ENABLED)}
    warmer = _ensure_extract_bg if req.mode == "extract" else _ensure_remux_bg
    for vid in ids:
        asyncio.create_task(warmer(vid))
    logger.info(
        "[internal/media] warm accepted mode=%s ids=%s user=%s",
        req.mode, ids, req.user_id[:8],
    )
    return {"ok": True, "warmed": len(ids), "mode": req.mode, "spool": bool(_SPOOL_ENABLED)}
