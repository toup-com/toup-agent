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
from fastapi.responses import JSONResponse
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

    **``media-src`` is why generated apps were silent (round 20).** There was
    no ``media-src`` directive, so it fell back to ``default-src 'self'`` —
    and on an opaque origin ``'self'`` matches nothing, which means every
    ``data:`` and ``blob:`` sound an app makes was refused. Reproduced with a
    control: with the directive absent the browser fires
    ``securitypolicyviolation: media-src`` and the element rejects with
    ``NotSupportedError — Failed to load because no supported source was
    found``; with ``media-src 'self' data: blob:`` present, the identical
    page plays. The error is the reason nobody caught it: it is what a
    browser says about a CORRUPT FILE, so it reads as a bad sound rather than
    as a policy decision, and the app's own `play()` rejection is a promise
    generated code never awaits.

    The mobile runner has always sent ``media-src data: blob:`` in its own
    ``<meta>`` policy (`appArtifacts.sandboxCsp`); this is the web half
    catching up, and the two are now the same policy in that respect. Nothing
    is loosened: ``data:`` and ``blob:`` are bytes the page already has,
    reachable with no network, which is why ``img-src`` has granted them
    since the beginning.

    **The directives themselves live in `app.artifact_policy`, not here.**
    The bug was two runners disagreeing about one artifact, so a second copy
    of the list — even a correct one — is the same mistake again. That module
    imports nothing but `app.config` precisely so the agent-side publish gate
    can load the app under this exact policy; the platform image does not ship
    `app/agent/` and the agent does not run this router, so the shared thing
    has to sit above them both.
    """
    from app.artifact_policy import sandbox_csp
    return sandbox_csp(
        settings.artifact_cdn_origin,
        frame_ancestors=settings.artifact_frame_ancestors,
    )


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
        # `autoplay=(self)` is stated rather than left to the default, which
        # is the same value. A sound an app makes on a tap is not autoplay in
        # the sense the policy is about, but the feature gates
        # `HTMLMediaElement.play()` and `AudioContext` alike — so the one
        # directive nobody may quietly tighten while debugging a noisy page is
        # written down, next to the ones that ARE closed. Delegation into the
        # cross-origin frame is the embedder's half (`AppArtifactFrame`'s
        # `allow="autoplay"`); this is the document's half.
        "Permissions-Policy": (
            "geolocation=(), microphone=(), camera=(), payment=(), "
            "usb=(), interest-cohort=(), autoplay=(self)"
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


@router.get("")
@router.get("/")
async def list_artifacts(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Every single-file app this account has, newest update first.

    The account bearer, not an artifact token: this is the SHELL asking, the
    way it asks for the file list — an artifact itself can never reach here.

    It exists because the `apps` table cannot answer the question. `AppResponse`
    has no size (it lives inside an opaque `files_json` blob) and no revision,
    and the manifest in the container is the only place both are kept. The
    mobile Files page renders name · modified · size, so without this route an
    app could only ever be listed without two of its three facts.
    """
    return await _agent_json(str(current_user.id), db, "GET", "/")


@router.patch("/{slug}")
async def rename_artifact(
    slug: str,
    body: dict,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename an app. The slug — every card's handle — never moves."""
    out = await _agent_json(
        str(current_user.id), db, "PATCH", f"/{slug}", body={"title": body.get("title")},
    )
    # Keep the `apps` row in step, so the Workspace list and the artifact list
    # do not disagree about what the same app is called. Best effort: the
    # manifest is the authority the runner and the card read, and failing the
    # rename because a mirror row is missing would be the wrong trade.
    try:
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import App
        title = (out or {}).get("title")
        if title:
            async with async_session_maker() as s:
                row = (await s.execute(
                    select(App).where(App.user_id == str(current_user.id), App.slug == slug)
                )).scalar_one_or_none()
                if row is not None:
                    row.name = title
                    await s.commit()
    except Exception:
        logger.warning("[artifact] apps-row rename mirror failed for %s", slug, exc_info=True)
    return out


@router.delete("/{slug}")
async def delete_artifact(
    slug: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an app: the file, its history, its state and its manifest row.

    The `apps` row goes with it. Leaving it behind would put a dead app in the
    Workspace list whose every open answers 404 — which is exactly the "no
    playlists yet for a full library" failure mode in reverse: a row for
    something that is gone reads as breakage, not as deletion.
    """
    out = await _agent_json(str(current_user.id), db, "DELETE", f"/{slug}")
    try:
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import App
        async with async_session_maker() as s:
            row = (await s.execute(
                select(App).where(App.user_id == str(current_user.id), App.slug == slug)
            )).scalar_one_or_none()
            if row is not None:
                await s.delete(row)
                await s.commit()
    except Exception:
        logger.warning("[artifact] apps-row delete mirror failed for %s", slug, exc_info=True)
    return out


@router.get("/{slug}/icon")
async def get_artifact_icon(
    slug: str,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """An app's icon, for a card in the SHELL.

    The account bearer, not an artifact token, and deliberately so: this is
    Toup's own UI asking for a tile, the same way it asks for the file list.
    An artifact can never reach here — it has no network at all.

    Served with the artifact CSP's spirit but its own, tighter policy: an SVG
    is a document, and a document served from an origin the shell trusts must
    not be able to run anything even if `logo.sanitize_svg` were one day
    fooled. `sandbox` with no allowances and `script-src 'none'` make that
    structural rather than a matter of the validator being right.
    """
    agent = await _get_agent(str(current_user.id), db)
    if not agent:
        raise HTTPException(503, "agent unavailable")
    agent_url, key = agent
    from app.services.agent_http import get_agent_http_client
    headers = {"X-Agent-Key": key}
    inm = request.headers.get("if-none-match")
    if inm:
        headers["If-None-Match"] = inm
    try:
        resp = await get_agent_http_client().get(
            f"{agent_url}{settings.api_prefix}/artifacts/{slug}/icon",
            headers=headers, timeout=10.0,
        )
    except Exception as exc:
        logger.warning("[artifact] icon fetch failed for %s: %s", slug, exc)
        raise HTTPException(502, "icon unreachable")
    if resp.status_code == 304:
        return Response(status_code=304, headers={
            "ETag": resp.headers.get("etag", ""), "Cache-Control": "no-cache"})
    if resp.status_code == 404:
        raise HTTPException(404, "no such app")
    if resp.status_code >= 400:
        raise HTTPException(502, "icon unreachable")
    return Response(
        content=resp.content,
        media_type="image/svg+xml",
        headers={
            "Cache-Control": "no-cache",
            "ETag": resp.headers.get("etag", ""),
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy":
                "default-src 'none'; style-src 'unsafe-inline'; sandbox",
            "Cross-Origin-Resource-Policy": "same-site",
        },
    )


@router.get("/{slug}/preview")
async def get_artifact_preview(
    slug: str,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """An app's publish-time snapshot, for a card in the SHELL.

    The account bearer, not an artifact token — same posture as the icon
    route, and for the same reason: this is Toup's own UI asking for a
    picture, and an artifact itself has no network at all. A 404 means no
    publish has stored a snapshot yet; the card shows its placeholder.
    """
    agent = await _get_agent(str(current_user.id), db)
    if not agent:
        raise HTTPException(503, "agent unavailable")
    agent_url, key = agent
    from app.services.agent_http import get_agent_http_client
    headers = {"X-Agent-Key": key}
    inm = request.headers.get("if-none-match")
    if inm:
        headers["If-None-Match"] = inm
    try:
        resp = await get_agent_http_client().get(
            f"{agent_url}{settings.api_prefix}/artifacts/{slug}/preview",
            headers=headers, timeout=10.0,
        )
    except Exception as exc:
        logger.warning("[artifact] preview fetch failed for %s: %s", slug, exc)
        raise HTTPException(502, "preview unreachable")
    if resp.status_code == 304:
        return Response(status_code=304, headers={
            "ETag": resp.headers.get("etag", ""), "Cache-Control": "no-cache"})
    if resp.status_code == 404:
        raise HTTPException(404, "no preview")
    if resp.status_code >= 400:
        raise HTTPException(502, "preview unreachable")
    return Response(
        content=resp.content,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache",
            "ETag": resp.headers.get("etag", ""),
            "X-Content-Type-Options": "nosniff",
            "Cross-Origin-Resource-Policy": "same-site",
        },
    )


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
    headers = {"X-Agent-Key": key}
    try:
        if method == "PUT":
            resp = await client.put(url, headers=headers, json=body or {}, timeout=15.0)
        elif method == "PATCH":
            resp = await client.patch(url, headers=headers, json=body or {}, timeout=15.0)
        elif method == "DELETE":
            resp = await client.delete(url, headers=headers, timeout=15.0)
        else:
            resp = await client.get(url, headers=headers, timeout=15.0)
    except Exception as exc:
        logger.warning("[artifact] %s %s failed: %s", method, path, exc)
        raise HTTPException(502, "artifact store unreachable")
    # The agent's own status is passed through where it is MEANINGFUL: a 404
    # for an app that is gone must reach the client as a 404 (the Files list
    # drops the row) rather than as a generic 502 (which reads as "the network
    # is down, try again" for something that will never come back).
    if resp.status_code == 404:
        raise HTTPException(404, "no such app")
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, "artifact store error")
    return resp.json()


async def _bearer_user_id(request: Request, db: AsyncSession) -> Optional[str]:
    """The account behind an ``Authorization: Bearer`` header, or None.

    **Header only.** ``get_current_user`` also honours the SSO cookie, and that
    is precisely what this module refuses to do: a cookie is attached by the
    browser to any request any page can cause, so honouring one here would let
    an attacker's ``<img>``/``<iframe>``/``fetch`` reach a victim's artifact on
    their ambient session. A Bearer header cannot be attached by a third-party
    page, which is what makes this branch unforgeable in the same way the
    scoped token is.
    """
    header = request.headers.get("authorization") or ""
    scheme, _, raw = header.partition(" ")
    if scheme.lower() != "bearer" or not raw.strip():
        return None
    try:
        from app.services.auth_service import decode_access_token, get_user_by_id
        user_id = decode_access_token(raw.strip())
        if not user_id:
            return None
        user = await get_user_by_id(db, user_id)
        if user is None or not getattr(user, "is_active", True):
            return None
        return str(user.id)
    except Exception:  # noqa: BLE001 - a bad token is not a server error
        logger.debug("[artifact] bearer decode failed", exc_info=True)
        return None


@router.get("/{slug}")
async def serve_artifact(
    slug: str,
    request: Request,
    token: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Serve one artifact — the document to a frame, the handle to a client.

    Two callers, one URL, told apart by HOW they authenticate:

    * **``?token=``** → the sandboxed HTML document. This is the iframe/WebView
      src. Auth is the query token ONLY; cookies are never consulted, because
      on the same-origin dev fallback the browser will still *send* the session
      cookie and honouring it would mean an attacker-supplied ``<iframe
      src=".../artifacts/x">`` on another site renders a victim's app with
      their ambient session. Requiring the scoped token makes the request
      unforgeable regardless of what the browser attaches.
    * **``Authorization: Bearer``, no token** → JSON: slug, title, revision,
      updated_at and the unwrapped body. This is the shape the mobile client
      has always asked this URL for (``api.getAppArtifact``), and answering it
      with 401 is half of why an edit never reached the user: told that
      revision 2 existed, the runner dropped its cached body, asked here for
      the new one, and was refused — so it kept showing revision 1. The body is
      the file as the model wrote it, NOT the runtime-wrapped document, because
      this caller applies its own sandbox wrapper.

    A request with neither is refused, and a request with both takes the token
    branch — the narrower credential wins, so a page cannot upgrade itself to
    the account's view by also sending a header.
    """
    from app.services.auth_service import decode_artifact_token

    user_id = decode_artifact_token(token or "", slug)
    if not user_id and not token:
        account_id = await _bearer_user_id(request, db)
        if account_id:
            return JSONResponse(
                await _agent_json(account_id, db, "GET", f"/{slug}/source"),
                headers={"Cache-Control": "no-store"},
            )
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
