"""Platform-side artifact serving — the sandbox boundary.

An artifact is HTML the model wrote. It is untrusted markup running untrusted
script, so the whole security posture is "this page must be able to do
nothing except paint itself":

* **Its own origin.** ``settings.artifact_origin`` should be a host that
  holds no Toup cookie and no Toup localStorage. Same-origin with the SPA
  would give injected app code the account's own storage.
* **No cookies, in either direction.** This route never reads a cookie for
  auth (query token only) and never sets one. That is the one difference
  from ``apps_proxy.preview_proxy``, which *does* set ``preview_token`` —
  fine for an Expo dev server, wrong for a page the model authored.
* **A strict CSP**, including ``frame-ancestors`` limited to Toup origins so
  nobody else can embed a user's app.
* **Sandboxed frame without ``allow-same-origin``** — enforced by the
  embedder (``AppArtifactFrame.tsx``); this module's job is to make sure
  nothing here *depends* on same-origin working.

The token is artifact-scoped (``auth_service.create_artifact_token``): it
fetches one static file. It is deliberately not the ``app_preview`` token,
which can also reach ``/api/apps/{id}/chat``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


def artifact_csp() -> str:
    """The Content-Security-Policy every artifact response carries.

    ``script-src`` keeps ``'unsafe-inline'`` and ``'unsafe-eval'`` on purpose:
    the whole format is inline script, and Babel-standalone (the supported
    React path) compiles JSX at runtime, which is ``eval``. They are safe to
    grant *here* precisely because everything else is closed — the page has
    no origin worth stealing from, no cookies, and no network.

    ``connect-src 'self'`` reads as permissive but is not: the frame is
    sandboxed without ``allow-same-origin``, so its origin is opaque and
    ``'self'`` matches nothing. It is the tightest expressible value —
    ``'none'`` is equivalent in effect and breaks same-origin debugging of
    the page outside a frame.
    """
    cdn = settings.artifact_cdn_origin
    ancestors = " ".join(settings.artifact_frame_ancestors) or "'none'"
    return "; ".join([
        "default-src 'self'",
        f"script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: {cdn}",
        f"style-src 'self' 'unsafe-inline' {cdn}",
        "img-src 'self' data: blob:",
        f"font-src 'self' data: {cdn}",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        f"frame-ancestors {ancestors}",
    ])


def artifact_headers() -> dict:
    return {
        "Content-Security-Policy": artifact_csp(),
        # Belt-and-braces for the same-origin dev fallback: even if the page
        # somehow ran with a real origin, these keep it from being reused as
        # a springboard.
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Resource-Policy": "same-site",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Permissions-Policy": (
            "geolocation=(), microphone=(), camera=(), payment=(), "
            "usb=(), interest-cohort=()"
        ),
        # An artifact changes under the same URL on every edit. A cached copy
        # would show the user their previous revision after present_app.
        "Cache-Control": "no-store, must-revalidate",
        "Pragma": "no-cache",
    }


def artifact_url(slug: str, token: str) -> str:
    """Absolute URL the SPA should put in the iframe ``src``."""
    base = (settings.artifact_origin or "").rstrip("/")
    path = f"{settings.api_prefix}/artifacts/{slug}?token={token}"
    return f"{base}{path}" if base else path


async def _get_agent(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
    """(agent_url, api_key) for an ACTIVE agent, or None. Mirrors
    ``apps_proxy._get_agent`` — the deploy_status filter matters: a
    provisioning or drained container answers, badly."""
    from sqlalchemy import select
    from app.db import AgentConfig
    row = (await db.execute(
        select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
            AgentConfig.user_id == user_id,
            AgentConfig.deploy_status == "active",
        )
    )).first()
    if not row or not row.agent_url or not row.agent_api_key:
        return None
    return row.agent_url.rstrip("/"), row.agent_api_key


@router.post("/{slug}/token")
async def mint_artifact_token(slug: str, current_user=Depends(get_current_user)):
    """Exchange the account bearer for a slug-scoped artifact token.

    The SPA calls this, then puts the RETURNED token in the iframe ``src`` —
    never the account JWT. Same shape as ``/apps/{id}/preview-token``.
    """
    from app.services.auth_service import create_artifact_token
    # Minted ONCE and reused for both fields — two calls would return two
    # different jti's and leave the caller guessing which one the url carries.
    tok = create_artifact_token(str(current_user.id), slug)
    return {
        "token": tok,
        "expires_in": settings.artifact_token_expire_minutes * 60,
        "url": artifact_url(slug, tok),
    }


@router.get("/{slug}/state")
async def get_artifact_state(
    slug: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Read an app's persisted state.

    Authenticated with the ACCOUNT bearer, not the artifact token, and
    called by the Toup shell — never by the artifact itself. The artifact
    has no network at all; it asks the shell over ``postMessage``. That is
    the whole point: model-authored code never holds a credential.
    """
    return await _agent_json(str(current_user.id), db, "GET", f"/{slug}/state")


@router.put("/{slug}/state")
async def put_artifact_state(
    slug: str,
    body: dict,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _agent_json(
        str(current_user.id), db, "PUT", f"/{slug}/state", body=body,
    )


async def _agent_json(user_id: str, db: AsyncSession, method: str,
                      path: str, body: Optional[dict] = None):
    agent = await _get_agent(user_id, db)
    if not agent:
        raise HTTPException(503, "agent unavailable")
    agent_url, key = agent
    from app.services.agent_http import get_agent_http_client
    client = get_agent_http_client()
    url = f"{agent_url}{settings.api_prefix}/artifacts{path}"
    try:
        if method == "PUT":
            resp = await client.put(
                url, headers={"X-Agent-Key": key}, json=body or {}, timeout=15.0,
            )
        else:
            resp = await client.get(url, headers={"X-Agent-Key": key}, timeout=15.0)
    except Exception as exc:
        logger.warning("[artifact] state %s %s failed: %s", method, path, exc)
        raise HTTPException(502, "artifact state unreachable")
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, "artifact state error")
    return resp.json()


@router.get("/{slug}")
async def serve_artifact(
    slug: str,
    request: Request,
    token: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Serve one artifact's HTML, sandboxed.

    Auth is the query token ONLY. Cookies are never consulted — on the
    same-origin dev fallback the browser will still *send* the session
    cookie, and honouring it would mean an attacker-supplied ``<iframe
    src=".../artifacts/x">`` on another site renders a victim's app with
    their ambient session. Requiring the scoped token makes the request
    unforgeable regardless of what the browser attaches.
    """
    from app.services.auth_service import decode_artifact_token

    user_id = decode_artifact_token(token or "", slug)
    if not user_id:
        # No WWW-Authenticate: this origin must never trigger a browser
        # credential prompt on an untrusted page.
        raise HTTPException(401, "artifact token required")

    agent = await _get_agent(user_id, db)
    if not agent:
        raise HTTPException(503, "agent unavailable")
    agent_url, key = agent

    from app.services.agent_http import get_agent_http_client
    try:
        client = get_agent_http_client()
        resp = await client.get(
            f"{agent_url}{settings.api_prefix}/artifacts/{slug}",
            headers={"X-Agent-Key": key},
            timeout=30.0,
        )
    except Exception as exc:
        logger.warning("[artifact] fetch failed for %s: %s", slug, exc)
        raise HTTPException(502, "artifact unreachable")

    if resp.status_code == 404:
        raise HTTPException(404, "no such app")
    if resp.status_code >= 400:
        logger.warning("[artifact] agent returned %s for %s", resp.status_code, slug)
        raise HTTPException(502, "artifact unreachable")

    headers = artifact_headers()
    rev = resp.headers.get("x-toup-artifact-revision")
    if rev:
        headers["X-Toup-Artifact-Revision"] = rev

    # Response, not StreamingResponse, and no set_cookie anywhere on this
    # path — see the module docstring. Nothing below may add one.
    return Response(
        content=resp.content,
        status_code=200,
        media_type="text/html; charset=utf-8",
        headers=headers,
    )
