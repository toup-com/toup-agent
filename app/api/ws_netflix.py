"""
Netflix Stream API — start/stop streams + serve HLS segments.

Endpoints:
  POST /api/netflix-stream/start  → start a stream, returns stream_id + HLS URL
  POST /api/netflix-stream/stop   → stop a stream
  GET  /api/netflix-stream/{stream_id}/{filename} → serve HLS segments
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/netflix-stream", tags=["Netflix Stream"])


class StartStreamRequest(BaseModel):
    netflix_url: str
    email: str
    password: str
    profile: str = ""


class StartStreamResponse(BaseModel):
    stream_id: str
    hls_url: str


@router.post("/start", response_model=StartStreamResponse)
async def start_stream(
    body: StartStreamRequest,
    agent_key: Optional[str] = Query(None, alias="agent_key"),
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
):
    """Start a Netflix streaming session."""
    from app.config import settings

    # Auth: agent key
    key = agent_key or x_agent_key
    if not key or key != settings.agent_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from app.services.netflix_stream import start_netflix_stream

    stream_id = str(uuid.uuid4())[:8]

    try:
        hls_dir = await start_netflix_stream(
            stream_id=stream_id,
            netflix_url=body.netflix_url,
            email=body.email,
            password=body.password,
            profile=body.profile,
        )
    except Exception as e:
        logger.exception("[NF-STREAM] Failed to start")
        raise HTTPException(status_code=500, detail=str(e))

    hls_url = f"/api/netflix-stream/{stream_id}/stream.m3u8"
    return StartStreamResponse(stream_id=stream_id, hls_url=hls_url)


@router.post("/stop")
async def stop_stream(
    stream_id: str,
    agent_key: Optional[str] = Query(None, alias="agent_key"),
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
):
    """Stop a Netflix streaming session."""
    from app.config import settings

    key = agent_key or x_agent_key
    if not key or key != settings.agent_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from app.services.netflix_stream import stop_netflix_stream
    await stop_netflix_stream(stream_id)
    return {"status": "stopped"}


@router.get("/{stream_id}/{filename}")
async def serve_hls(stream_id: str, filename: str):
    """Serve HLS playlist and segments."""
    from app.services.netflix_stream import get_stream_hls_dir

    hls_dir = get_stream_hls_dir(stream_id)
    if not hls_dir:
        raise HTTPException(status_code=404, detail="Stream not found")

    # `filename` is a path parameter from the client. Only a plain segment
    # name inside THIS stream's directory may be served — a `..` or nested
    # path resolves outside it and is a 404, never a read (2026-08-19
    # file-endpoint audit: this was `hls_dir / filename` unchecked, with
    # the extension allow-list applied only afterwards).
    from pathlib import Path as _Path
    base = _Path(hls_dir).resolve()
    file_path = (base / _Path(filename).name).resolve()
    if _Path(filename).name != filename or file_path.parent != base:
        raise HTTPException(status_code=404, detail="Segment not found")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Segment not found")

    # Set correct content types
    if filename.endswith(".m3u8"):
        return Response(
            content=file_path.read_bytes(),
            media_type="application/vnd.apple.mpegurl",
            headers={
                "Cache-Control": "no-cache, no-store",
                "Access-Control-Allow-Origin": "*",
            },
        )
    elif filename.endswith(".ts"):
        return Response(
            content=file_path.read_bytes(),
            media_type="video/MP2T",
            headers={
                "Cache-Control": "max-age=3600",
                "Access-Control-Allow-Origin": "*",
            },
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown file type")
