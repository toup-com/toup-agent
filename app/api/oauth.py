"""
T1d — Connector OAuth API.

Three user-facing endpoints + two stub-only endpoints:

  POST /api/oauth/connect/<connector_id>      (auth required)
  GET  /api/oauth/callback                    (provider hits this)
  POST /api/oauth/disconnect/<connector_id>   (auth required)

  GET  /api/oauth/_stub/authorize             (test harness only)
  POST /api/oauth/_stub/token                 (test harness only)

Flow (PKCE per architecture §3.1):

  1. /connect generates code_verifier, code_challenge, state. Inserts
     ConnectorOAuthSession row keyed by sha256(state). Returns the
     provider's authorize_url with the appropriate query params.
  2. Browser navigates to authorize_url → user consents at provider →
     provider 302s back to /callback?code=...&state=....
  3. /callback verifies the state HMAC, atomic-deletes the session row
     (single-use), exchanges code+verifier for tokens, persists via
     vault.put, redirects to /agent/integrations?connected=<id>.

PKCE token exchange itself is provider-agnostic standard OAuth 2.0 —
the same POST shape works against Google, GitHub, Slack, Notion, etc.
We keep it in the OAuth API rather than the provider class because
the only variation is the token URL (already in ProviderAppConfig);
provider-specific quirks like Slack's response shape can be handled
in a follow-up if/when needed.

`provider.refresh()`, `provider.revoke()`, and `provider.execute()`
ARE provider-specific (different URLs, different params per service)
and live in each connector's provider.py.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import urllib.parse
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db
from app.db.models import (
    ConnectorOAuthSession,
    EVENT_CONNECTED,
    EVENT_REVOCATION_PROVIDER_FAILED,
    EVENT_REVOCATION_PROVIDER_SUCCEEDED,
)
from app.services import connector_vault as vault
from app.services.connector_registry import get_registry
from app.services.connector_vault import _audit_then_commit
from app.services.oauth_state import (
    StateVerificationError,
    sign_state,
    state_hash,
    verify_state,
)
from app.services.provider_apps import get_provider_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/oauth", tags=["OAuth"])


# ─── Response shapes ─────────────────────────────────────────────────


class ConnectStartResponse(BaseModel):
    authorize_url: str


class DisconnectResponse(BaseModel):
    status: str
    revoke_outcome: str  # "succeeded" | "failed" | "no_identity"


# ─── Helpers ─────────────────────────────────────────────────────────


def _flag_or_503() -> None:
    if not settings.enable_connector_oauth:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Connector OAuth is unavailable. Most often this is a "
                "missing OAUTH_STATE_SECRET env var on the platform — "
                "set it (e.g. `python -c \"import secrets; "
                "print(secrets.token_urlsafe(32))\"`) and redeploy. "
                "ENABLE_CONNECTOR_OAUTH must also be true."
            ),
        )


def _make_pkce() -> tuple[str, str]:
    """PKCE: cryptographically random verifier (43–128 chars per RFC
    7636) → code_challenge = base64url(sha256(verifier))."""
    verifier = secrets.token_urlsafe(64)[:96]  # well within RFC bounds
    challenge_bytes = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _build_authorize_url(
    *,
    base_url: str,
    client_id: str,
    redirect_uri: str,
    scopes: list[str],
    state: str,
    code_challenge: str,
    use_pkce: bool,
) -> str:
    """Compose the standard OAuth 2.0 authorize URL."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes) if scopes else "",
        "state": state,
    }
    if use_pkce:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{urllib.parse.urlencode(params)}"


async def _exchange_code_for_tokens(
    *,
    token_url: str,
    code: str,
    code_verifier: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict:
    """Standard PKCE token exchange. Returns the parsed JSON response.

    For local stub (`token_url` starts with `/api/...`), the call goes
    in-process via httpx ASGITransport. In production, hits the real
    provider endpoint over the public internet.
    """
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": code_verifier,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }
    # If the URL is relative (stub), we resolve against the platform's
    # own host. The caller (callback handler) passes its request.url for
    # this resolution.
    if token_url.startswith("/"):
        # In-process: caller wires us through directly. Production never
        # hits this branch — real provider URLs are absolute.
        from app.api.oauth import _stub_token_exchange  # self-import OK
        return await _stub_token_exchange(body)

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        r = await client.post(token_url, data=body)
        r.raise_for_status()
        return r.json()


# ─── /connect ────────────────────────────────────────────────────────


@router.post("/connect/{connector_id}", response_model=ConnectStartResponse)
async def oauth_connect(
    connector_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start an OAuth dance for (current_user, connector_id).

    Returns the URL the browser should navigate to. The frontend opens
    this in a popup; the user consents at the provider; the provider
    redirects back to /api/oauth/callback (handled below)."""
    _flag_or_503()

    # 1. Manifest lookup.
    entry = get_registry().get(connector_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown connector {connector_id!r}",
        )

    # 2. Provider-app config lookup.
    app_cfg = get_provider_app(entry.manifest.oauth.provider_app)
    if app_cfg is None:
        provider = entry.manifest.oauth.provider_app
        # Map provider name → the exact env vars the operator must set.
        # Keeps the 503 actionable instead of generic "config missing".
        env_hints = {
            "google": "GOOGLE_OAUTH_CLIENT_ID + GOOGLE_OAUTH_CLIENT_SECRET",
            "github": "GITHUB_OAUTH_CLIENT_ID + GITHUB_OAUTH_CLIENT_SECRET",
        }
        hint = env_hints.get(provider, f"{provider.upper()}_OAUTH_CLIENT_ID + ..._SECRET")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"{entry.manifest.name} OAuth credentials not configured "
                f"on the platform. Set {hint} and redeploy."
            ),
        )

    # 3. PKCE + state.
    code_verifier, code_challenge = _make_pkce()
    state = sign_state(str(current_user.id), connector_id)
    expires_at_ts = secrets.token_urlsafe(0)  # placeholder for type
    from datetime import datetime, timedelta
    expires_at = datetime.utcnow() + timedelta(seconds=settings.oauth_state_ttl_seconds)

    # 4. Persist the in-flight session so /callback can look up the
    #    code_verifier when the browser comes back. Single-use: the
    #    callback DELETEs this row in the same transaction as token exchange.
    db.add(ConnectorOAuthSession(
        state_hash=state_hash(state),
        user_id=str(current_user.id),
        connector_id=connector_id,
        code_verifier=code_verifier,
        expires_at=expires_at,
    ))
    await db.commit()

    # 5. Build the authorize URL with the manifest's declared scopes.
    authorize_url = _build_authorize_url(
        base_url=app_cfg.authorize_url,
        client_id=app_cfg.client_id,
        redirect_uri=settings.oauth_callback_url,
        scopes=entry.manifest.oauth.scopes,
        state=state,
        code_challenge=code_challenge,
        use_pkce=app_cfg.use_pkce,
    )

    # T1d Q2: absolutise stub authorize URLs so the frontend's
    # window.open contract is "every authorize_url is absolute, period"
    # — matching real providers (which return absolute URLs by nature
    # of being external). Resolved against the parsed origin of
    # oauth_callback_url so the stub stays same-origin with the
    # callback regardless of test vs production host.
    if authorize_url.startswith("/"):
        cb_parsed = urllib.parse.urlparse(settings.oauth_callback_url)
        origin = f"{cb_parsed.scheme}://{cb_parsed.netloc}"
        authorize_url = origin + authorize_url

    return ConnectStartResponse(authorize_url=authorize_url)


# ─── /callback ───────────────────────────────────────────────────────


@router.get("/callback")
async def oauth_callback(
    request: Request,
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Provider hits this with `?code=...&state=...` (or `?error=...`).

    No FastAPI auth — the state token IS the trust anchor. We verify
    HMAC + atomic-delete the session row + exchange code for tokens +
    persist via vault.put + redirect to the frontend success page.
    """
    # User declined consent at the provider, OR provider returned an
    # error. Surface to the frontend so it can show the right state.
    if error:
        logger.info("[oauth.callback] provider returned error=%s", error)
        # Same bridge route — popup gets the postMessage in error mode,
        # full-page fallback gets the URL params on the bridge page
        # and bounces back to /agent/integrations via the parent's
        # `?error=` handler when the bridge can't postMessage.
        return RedirectResponse(
            url=f"/oauth/callback-bridge?error={urllib.parse.quote(error)}",
            status_code=status.HTTP_302_FOUND,
        )

    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="callback missing code or state",
        )

    # 1. Verify state HMAC + TTL.
    try:
        payload = verify_state(state)
    except StateVerificationError as e:
        logger.warning("[oauth.callback] state verification failed: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid state")

    # 2. Atomic single-use lookup + delete on the session row.
    sh = state_hash(state)
    session_row = (await db.execute(
        select(ConnectorOAuthSession).where(ConnectorOAuthSession.state_hash == sh)
    )).scalar_one_or_none()
    if session_row is None:
        # Either never existed (forged state with valid HMAC somehow)
        # or already redeemed (replay).
        logger.warning(
            "[oauth.callback] no session row for state_hash=%s — replay or forgery",
            sh[:12],
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid state")

    # Cross-check: the session's user_id must match the state's user_id.
    # Defense in depth — the HMAC verify already proved the state wasn't
    # forged, but if the session and state disagree on user_id, refuse.
    if session_row.user_id != payload.user_id:
        logger.warning(
            "[oauth.callback] session user_id %s != state user_id %s",
            session_row.user_id[:8], payload.user_id[:8],
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid state")

    # Delete in the same transaction as the rest of the work — single-use.
    await db.execute(
        delete(ConnectorOAuthSession).where(ConnectorOAuthSession.state_hash == sh)
    )

    # 3. Look up connector + provider-app config.
    entry = get_registry().get(payload.connector_id)
    if entry is None:
        await db.rollback()
        logger.error(
            "[oauth.callback] connector %s not in registry — was it removed mid-flow?",
            payload.connector_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="connector unknown",
        )
    app_cfg = get_provider_app(entry.manifest.oauth.provider_app)
    if app_cfg is None:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="provider app unknown",
        )

    # 4. Exchange code for tokens.
    try:
        tokens = await _exchange_code_for_tokens(
            token_url=app_cfg.token_url,
            code=code,
            code_verifier=session_row.code_verifier,
            client_id=app_cfg.client_id,
            client_secret=app_cfg.client_secret,
            redirect_uri=settings.oauth_callback_url,
        )
    except httpx.HTTPError as e:
        await db.rollback()
        logger.exception("[oauth.callback] token exchange failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"token exchange failed: {type(e).__name__}",
        )

    access_token = tokens.get("access_token")
    if not access_token:
        await db.rollback()
        logger.error("[oauth.callback] token endpoint did not return access_token: %s", tokens)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="provider did not return access_token",
        )

    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in")
    from datetime import datetime, timedelta
    expires_at = (
        datetime.utcnow() + timedelta(seconds=int(expires_in))
        if expires_in else None
    )
    granted_scopes = (
        tokens.get("scope", "").split() if tokens.get("scope")
        else entry.manifest.oauth.scopes
    )

    # 5. Commit the session-delete BEFORE vault.put, since vault.put
    #    runs its own transactions (audit-then-act). If we left the
    #    delete pending, vault.put's commits would also commit the
    #    delete — which is what we want — but isolating it makes
    #    intent clearer and matches the streaming.py pattern.
    await db.commit()

    await vault.put(
        db,
        payload.user_id,
        payload.connector_id,
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=expires_at,
        scopes=granted_scopes,
        provider_account_id=tokens.get("account") or tokens.get("login"),
        metadata={"raw_token_response_keys": sorted(tokens.keys())},
        event_type=EVENT_CONNECTED,
    )

    # 6. T1h: notify the tenant container's /api/agent/refresh-tools so
    # it drops its 60s TTL list_tools cache and the new connector's
    # tools appear immediately. Best-effort — failure logs WARN and
    # the TTL acts as the safety net (the OAuth flow must succeed
    # regardless).
    from app.services.docker_host_service import refresh_tenant_tools
    await refresh_tenant_tools(db, payload.user_id)
    logger.info(
        "[oauth.callback] connected user=%s connector=%s",
        payload.user_id[:8], payload.connector_id,
    )

    # 7. Redirect to the frontend OAuth callback bridge. When the
    #    OAuth flow ran in a popup, the bridge posts back to the
    #    parent via BroadcastChannel and closes; when the flow ran
    #    full-page (popup blocked), the bridge's render still works
    #    and the parent's URL-param effect on /agent/integrations
    #    handles the success.
    return RedirectResponse(
        url=f"/oauth/callback-bridge?connected={urllib.parse.quote(payload.connector_id)}",
        status_code=status.HTTP_302_FOUND,
    )


# ─── /disconnect ─────────────────────────────────────────────────────


@router.post("/disconnect/{connector_id}", response_model=DisconnectResponse)
async def oauth_disconnect(
    connector_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke + zero. Per architecture §3.5: best-effort revoke, then
    same-transaction status flip + ciphertext zero. Disconnect proceeds
    even if revoke fails (the token is dead-to-us regardless)."""
    _flag_or_503()

    entry = get_registry().get(connector_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown connector {connector_id!r}",
        )

    user_id = str(current_user.id)
    revoke_outcome = "no_identity"

    # 1. Try to revoke at the provider. Best-effort.
    identity = await vault.get(db, user_id, connector_id)
    if identity is not None:
        try:
            await entry.provider.revoke(
                user_id=user_id,
                access_token=identity.access_token,
                refresh_token=identity.refresh_token,
            )
            revoke_outcome = "succeeded"
            await _audit_then_commit(
                db,
                user_id=user_id,
                connector_id=connector_id,
                event_type=EVENT_REVOCATION_PROVIDER_SUCCEEDED,
            )
        except Exception as e:
            revoke_outcome = "failed"
            logger.warning(
                "[oauth.disconnect] provider revoke failed for user=%s connector=%s: %s",
                user_id[:8], connector_id, e,
            )
            await _audit_then_commit(
                db,
                user_id=user_id,
                connector_id=connector_id,
                event_type=EVENT_REVOCATION_PROVIDER_FAILED,
                metadata={"error_class": type(e).__name__, "error_str": str(e)[:300]},
            )

    # 2. Vault disconnect — atomic status flip + ciphertext zero
    #    (writes its own EVENT_DISCONNECTED audit row).
    await vault.disconnect(db, user_id, connector_id)

    # 3. T1h: notify tenant /api/agent/refresh-tools so the
    # connector's tools disappear from list_tools immediately.
    # Best-effort — see oauth.callback for failure semantics.
    from app.services.docker_host_service import refresh_tenant_tools
    await refresh_tenant_tools(db, user_id)
    logger.info(
        "[oauth.disconnect] disconnected user=%s connector=%s revoke=%s",
        user_id[:8], connector_id, revoke_outcome,
    )

    return DisconnectResponse(status="disconnected", revoke_outcome=revoke_outcome)


# ─── Stub-only endpoints (test harness) ──────────────────────────────


@router.get("/_stub/authorize")
async def stub_authorize(
    request: Request,
    state: Optional[str] = Query(default=None),
    code_challenge: Optional[str] = Query(default=None),
    redirect_uri: Optional[str] = Query(default=None),
):
    """Stub authorize endpoint. Used by `stub_provider_app` only.

    Real providers show a consent screen here. The stub immediately
    302s back to `/api/oauth/callback?code=stub_code&state=...` with
    a deterministic code so tests can complete the round-trip without
    leaving the platform.
    """
    if not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stub authorize requires state",
        )
    callback = redirect_uri or settings.oauth_callback_url
    code = f"stub_code_{secrets.token_urlsafe(8)}"
    return RedirectResponse(
        url=f"{callback}?code={code}&state={urllib.parse.quote(state)}",
        status_code=status.HTTP_302_FOUND,
    )


async def _stub_token_exchange(form: dict) -> dict:
    """Stub token endpoint logic. Returns deterministic tokens.
    Called from `_exchange_code_for_tokens` when the URL is relative
    (avoids needing a network round-trip in-process)."""
    # Minimal validation — real providers would check client_id/secret
    # and verify the PKCE code_verifier hashes to the stored challenge.
    # The stub trusts the caller because it only runs in tests.
    if not form.get("code") or not form.get("code_verifier"):
        raise httpx.HTTPError("stub token: missing code or code_verifier")
    suffix = secrets.token_urlsafe(8)
    return {
        "access_token": f"stub_at_{suffix}",
        "refresh_token": f"stub_rt_{suffix}",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "",
        "account": "stub_user@example.com",
    }


@router.post("/_stub/token")
async def stub_token_endpoint(request: Request) -> dict:
    """HTTP wrapper around `_stub_token_exchange` for the unlikely
    case where token exchange is routed through HTTP rather than the
    in-process shortcut. Production never hits this — it's only here
    for completeness so the stub provider_app's token_url has a real
    backing route if a test wants to call it directly."""
    form = dict(await request.form())
    return await _stub_token_exchange(form)
