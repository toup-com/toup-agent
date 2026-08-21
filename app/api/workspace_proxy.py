"""File library proxy — the platform's ONLY role in the file system.

Every ``/api/library/*`` and ``/api/workspace/*`` request is authenticated
here and forwarded, verbatim, to the caller's own agent container, which
executes it (``app/api/library.py``) against the virtual file tree. Body,
query string, method and content-type pass straight through; the agent's
status, body and content headers come straight back.

What this replaces (2026-08-19): a per-route module that listed the
container filesystem over SSH (``find`` on ``/data/agents/<uuid8>/workspace``),
read files with ``docker exec … cat``, wrote/moved/deleted with
``base64 -d >``/``mv``/``rm -rf`` on HOST paths, "sanitised" client paths with
``.replace("..", "")``, and merged ``apps/`` + ``vibecoding/`` build trees into
the listing. Pool tenants have no host bind, so most of that was dead; what
worked leaked container internals into the UI. None of it survives: there is
no SSH, no docker exec, no host path and no client-supplied filesystem path
anywhere in this file.

Failure semantics: no agent, or an unreachable one, is a **503** with a
plain message — never an empty listing (an empty library reads as data
loss). 4xx from the agent are relayed as-is (a 404 for an id the agent does
not know is the correct answer for a cross-tenant id, too).
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user, security
from app.api.files import _get_agent_proxy_info, get_user_for_file
from app.config import settings
from app.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Files"])

_ALLOWED_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]

# Timeouts. Listings are manifest reads; uploads/downloads stream up to
# 50 MB; a first DOCX preview is a LibreOffice cold convert (measured
# 10.3 s on a real file — see files.py).
_JSON_READ_S = 20.0
_STREAM_READ_S = 120.0

# Route suffixes that clients embed (<img>, <iframe>, WebView, expo
# downloads) — these accept ?token= through the same explicit-before-ambient
# ladder as /api/files; everything else needs the normal session auth
# (Bearer / SSO cookie, with token revocation checks).
_EMBED_SUFFIXES = ("/download", "/preview", "file-download")
# Never relay these to the caller: hop-by-hop, framing (we re-frame as
# chunked), and anything that could pin the browser to the agent's origin.
_DROP_RESPONSE_HEADERS = {
    "content-length", "transfer-encoding", "connection", "keep-alive",
    "server", "date", "set-cookie", "www-authenticate", "access-control-allow-origin",
    "access-control-allow-credentials",
}
_FORWARD_REQUEST_HEADERS = ("content-type", "accept", "content-length", "if-none-match", "range")


def _is_embed_route(path: str) -> bool:
    p = "/" + path.strip("/")
    return p.endswith(_EMBED_SUFFIXES[0]) or p.endswith(_EMBED_SUFFIXES[1]) or p.endswith(_EMBED_SUFFIXES[2])


async def _authenticate(request: Request, path: str, credentials: Optional[HTTPAuthorizationCredentials],
                        token: Optional[str], db: AsyncSession):
    if _is_embed_route(path):
        return await get_user_for_file(request, token, db)
    return await get_current_user(request, credentials, db)


async def _forward(request: Request, user_id: str, upstream_path: str, db: AsyncSession) -> Response:
    proxy = await _get_agent_proxy_info(user_id, db)
    if not proxy:
        raise HTTPException(503, "Your agent is not available right now. Please try again in a moment.")
    agent_url, agent_key = proxy

    from app.services.agent_http import get_agent_http_client
    client = get_agent_http_client()

    streamed_body = request.method in ("POST", "PUT", "PATCH")
    headers = {"X-Agent-Key": agent_key}
    for h in _FORWARD_REQUEST_HEADERS:
        if h == "content-length" and not streamed_body:
            continue
        v = request.headers.get(h)
        if v:
            headers[h] = v
    # Never forward the user's own credentials to the agent — X-Agent-Key
    # is the agent-side identity, and the agent's ladder would otherwise
    # see a platform JWT it cannot validate.
    query = [(k, v) for k, v in request.query_params.multi_items() if k not in ("token", "agent_key")]

    is_embed = _is_embed_route(upstream_path)
    read_timeout = _STREAM_READ_S if (streamed_body or is_embed) else _JSON_READ_S

    url = f"{agent_url.rstrip('/')}{settings.api_prefix}/{upstream_path.lstrip('/')}"
    try:
        req = client.build_request(
            request.method, url, headers=headers, params=query,
            content=request.stream() if streamed_body else None,
            timeout=httpx.Timeout(connect=5.0, read=read_timeout, write=read_timeout, pool=5.0),
        )
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        logger.warning("[library-proxy] %s %s unreachable: %s", request.method, url, e)
        raise HTTPException(503, "Could not reach your agent. Please try again.")

    if resp.status_code == 404:
        # FastAPI's default 404 for an UNMOUNTED path is exactly
        # {"detail":"Not Found"} (capital F); the library's own 404s never
        # use that string. An agent image that predates the library router
        # must read as "being updated", not as "no such file".
        body = await resp.aread()
        await resp.aclose()
        if body.strip() == b'{"detail":"Not Found"}':
            raise HTTPException(503, "Your agent is being updated. Please try again in a few minutes.")
        return Response(content=body, status_code=404,
                        media_type=resp.headers.get("content-type", "application/json"))

    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    media_type = resp.headers.get("content-type")

    if resp.status_code == 304:
        # `if-none-match` has always been forwarded (_FORWARD_REQUEST_HEADERS),
        # but until the library routes carried an ETag the agent had nothing to
        # match and this was unreachable. Now that it is reachable, a 304 must
        # not come back as a StreamingResponse: that frames a body the status
        # forbids and stamps a content-type on a response that has no content.
        await resp.aclose()
        out_headers.pop("content-type", None)
        return Response(status_code=304, headers=out_headers)

    async def _iter():
        try:
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(_iter(), status_code=resp.status_code, media_type=media_type, headers=out_headers)


@router.api_route("/library/{path:path}", methods=_ALLOWED_METHODS)
async def proxy_library(
    request: Request,
    path: str,
    token: Optional[str] = Query(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    upstream = f"library/{path}"
    user = await _authenticate(request, upstream, credentials, token, db)
    return await _forward(request, user.id, upstream, db)


@router.api_route("/workspace/{path:path}", methods=_ALLOWED_METHODS)
async def proxy_workspace(
    request: Request,
    path: str,
    token: Optional[str] = Query(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    upstream = f"workspace/{path}"
    user = await _authenticate(request, upstream, credentials, token, db)
    return await _forward(request, user.id, upstream, db)
