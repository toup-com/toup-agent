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
import re
import secrets
import urllib.parse
from typing import Mapping, Optional

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
    AgentConfig,
    ConnectorIdentity,
    ConnectorOAuthSession,
    EVENT_CONNECTED,
    EVENT_REVOCATION_PROVIDER_FAILED,
    EVENT_REVOCATION_PROVIDER_SUCCEEDED,
)
from sqlalchemy import update as sa_update
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
from app.services.background_tasks import spawn as _spawn_bg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/oauth", tags=["OAuth"])


# ─── Response shapes ─────────────────────────────────────────────────


class ConnectStartResponse(BaseModel):
    authorize_url: str


class DisconnectResponse(BaseModel):
    status: str
    revoke_outcome: str  # "succeeded" | "failed" | "no_identity"


class ReadOnlyRequest(BaseModel):
    read_only: bool


class ReadOnlyResponse(BaseModel):
    connector_id: str
    read_only: bool


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
    provider_name: str = "",
    force_account_selection: bool = False,
    extra_params: Optional[Mapping[str, str]] = None,
    scope_param: str = "scope",
) -> str:
    """Compose the standard OAuth 2.0 authorize URL.

    `force_account_selection=True` adds `prompt=select_account` so the
    provider re-shows its account chooser even when the user is already
    signed in. Drives the connected-card "Switch account" action;
    Google/Microsoft/GitHub all honour the standard OAuth 2.0 `prompt`
    parameter.

    `scope_param` names the parameter the scope list rides in. It is
    `scope` per RFC 6749 §3.1 for everyone except Slack, which issues a
    BOT token for `scope` and a USER token for `user_scope` — see
    `ProviderAppConfig.scope_param`. Only the chosen key is emitted; a
    provider that partitions its scopes this way rejects an empty
    counterpart rather than ignoring it.

    `provider_name` is the registered provider-app key (`google`,
    `microsoft`, `github`, `linkedin`). Used to apply provider-specific
    quirks below — most importantly Google's `access_type=offline` +
    `prompt=consent` pair, WITHOUT which Google does not return a
    refresh_token and every connected Gmail/Calendar/Drive identity
    flips to `reauth_required` exactly one hour after the user
    connects. Pin this rigorously; the symptom is "the connector
    disconnected itself overnight" reported every hour, on the hour.
    """
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        scope_param: " ".join(scopes) if scopes else "",
        "state": state,
    }
    if use_pkce:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"

    # Provider-specific quirks that MUST be applied to issue refresh
    # tokens. Without these, the access token expires in ~1 h with
    # no recovery path → identity flips to reauth_required and the
    # MCP layer drops the tool, which sends the agent to its browser
    # fallback. Same root cause hit every Google connector at the
    # 1-h mark since the initial connectors commit (126bb77, never
    # patched). See `provider_apps.py:272` for the original
    # aspirational comment that promised these would be added here.
    if provider_name == "google":
        # `access_type=offline` is REQUIRED — Google omits the
        # refresh_token in the token response otherwise.
        params["access_type"] = "offline"
        # `prompt=consent` ensures Google re-issues a refresh_token on
        # re-auth too (Google's default is to skip the consent screen
        # AND skip the refresh_token if the user has previously
        # consented — same hourly-disconnect symptom on re-connect).
        # Overridden by `force_account_selection=True` below since the
        # caller's intent is "show the account chooser, not consent".
        if not force_account_selection:
            params["prompt"] = "consent"

    if force_account_selection:
        # Overrides Google's `prompt=consent` — `select_account` is
        # the right thing for "Switch account", and Google still
        # honours `access_type=offline` so refresh_token is preserved.
        params["prompt"] = "select_account"

    # Provider-declared required params (`ProviderAppConfig.
    # extra_authorize_params`). Applied LAST so a provider that declares
    # a param as REQUIRED wins over the generic handling above — the
    # motivating case is Atlassian, which documents `prompt=consent` as
    # required and would otherwise be silently overwritten with
    # `select_account` by the Switch-account path, turning a documented
    # requirement into an intermittent one that only breaks for users
    # who switch accounts.
    for _k, _v in (extra_params or {}).items():
        params[_k] = _v

    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{urllib.parse.urlencode(params)}"


class TokenResponseUnparseable(Exception):
    """The token endpoint answered, but not in any shape we can read.

    Distinct from `httpx.HTTPError` (transport/status) on purpose: this
    one means the provider is reachable and answering, so the operator
    fix is a parser or a header, not a network probe.
    """


def _parse_token_response(r: httpx.Response) -> dict:
    """Read a token response as JSON *or* form-urlencoded.

    RFC 6749 §5.1 mandates JSON, and Google / Microsoft / LinkedIn all
    comply unconditionally. **GitHub does not**: its token endpoint
    content-negotiates and defaults to `application/x-www-form-
    urlencoded`, so a bare `r.json()` raised `JSONDecodeError: Expecting
    value: line 1 column 1` — an unhandled 500 on the callback, for
    every GitHub connect attempt ever made. Sending `Accept:
    application/json` (below) is the real fix; this parser is the belt
    to that pair of braces, so a provider that ignores the header can
    never again turn a *successful* authorization into a 500.
    """
    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()

    def _form() -> dict:
        # `parse_qs` drops blank values and returns a list per key; the
        # OAuth token response is flat, so collapse to the first.
        return {k: v[0] for k, v in urllib.parse.parse_qs(r.text).items() if v}

    if ctype.endswith("x-www-form-urlencoded"):
        parsed = _form()
        if parsed:
            return parsed
        # Mislabelled — fall through and try JSON below.

    try:
        payload = r.json()
    except ValueError:
        # Not JSON either. A form body sent under the wrong content-type
        # is the realistic case; an HTML error page is the other, and
        # `parse_qs` yields nothing useful for it, which is what we want.
        parsed = _form()
        if parsed:
            return parsed
        raise TokenResponseUnparseable(
            f"content-type={ctype or 'none'!r} body={r.text[:120]!r}"
        ) from None

    if not isinstance(payload, dict):
        raise TokenResponseUnparseable(f"expected an object, got {type(payload).__name__}")
    return payload


def _lift_nested_token(tokens: dict, key: str) -> dict:
    """Overlay `tokens[key]`'s keys onto `tokens`, if it holds a dict.

    OVERLAY, not replace, and that is the whole design. Slack's response
    is `{ok, access_token: <bot>, team: {...}, authed_user: {id, scope,
    access_token: <user>}}` on success and `{ok: false, error: …}` on
    failure — with no `authed_user` at all. Replacing would delete the
    error channel exactly when it is needed, and dropping `team` costs
    the only workspace identifier in the response.

    Nested keys win: that is the point, since the outer `access_token`
    and the outer `scope` both describe the bot.
    """
    if not key:
        return tokens
    nested = tokens.get(key)
    if not isinstance(nested, dict) or not nested:
        # Not an error worth failing on. The caller's missing-
        # `access_token` branch already reports the provider's own words,
        # and it does so with more context than this function has.
        return tokens
    merged = dict(tokens)
    merged.update(nested)
    return merged


async def _exchange_code_for_tokens(
    *,
    token_url: str,
    code: str,
    code_verifier: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    use_pkce: bool = True,
    token_auth_style: str = "body",
    token_body_format: str = "form",
    extra_headers: Optional[Mapping[str, str]] = None,
    token_lift_key: str = "",
) -> dict:
    """Standard PKCE token exchange. Returns the parsed token response.

    For local stub (`token_url` starts with `/api/...`), the call goes
    in-process via httpx ASGITransport. In production, hits the real
    provider endpoint over the public internet.

    Note the response is not necessarily JSON on the wire — see
    `_parse_token_response`. Nor does a 2xx mean success: GitHub returns
    HTTP 200 with an `error=` body, so the caller must check for
    `access_token` rather than trusting the status.

    `token_lift_key` names a nested object whose keys are overlaid onto
    the result (`ProviderAppConfig.token_lift_key`). Slack nests the USER
    token under `authed_user` and keeps the top level for the bot token,
    so without the lift the callback stores the wrong principal's
    credential — a token that authenticates fine and then cannot see any
    of the user's conversations.
    """
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    # `code_verifier` belongs to PKCE. It used to be sent unconditionally,
    # which made `manifest.oauth.pkce` and `ProviderAppConfig.use_pkce`
    # decorative on this leg — Google/GitHub/LinkedIn happen to ignore a
    # stray field, so nothing broke and nothing revealed that the flag had
    # no reader here. A provider that validates its inputs would reject
    # the whole exchange.
    if use_pkce:
        body["code_verifier"] = code_verifier

    headers = {"Accept": "application/json"}
    headers.update(extra_headers or {})

    # RFC 6749 §2.3.1: a client may authenticate with HTTP Basic instead
    # of body parameters, and some providers accept ONLY that. Notion
    # 401s on body credentials.
    auth = None
    if token_auth_style == "basic":
        auth = (client_id, client_secret)
    else:
        body["client_id"] = client_id
        body["client_secret"] = client_secret
    # If the URL is relative (stub), we resolve against the platform's
    # own host. The caller (callback handler) passes its request.url for
    # this resolution.
    if token_url.startswith("/"):
        # In-process: caller wires us through directly. Production never
        # hits this branch — real provider URLs are absolute.
        from app.api.oauth import _stub_token_exchange  # self-import OK
        return _lift_nested_token(await _stub_token_exchange(body), token_lift_key)

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        # `Accept` is load-bearing for GitHub and a no-op everywhere
        # else — see `_parse_token_response`. RFC 6749 §4.1.3 mandates a
        # form body and every provider here uses one except Notion, which
        # documents JSON and answers 400 invalid_json to a form body.
        if token_body_format == "json":
            r = await client.post(token_url, json=body, headers=headers, auth=auth)
        else:
            r = await client.post(token_url, data=body, headers=headers, auth=auth)
        r.raise_for_status()
        return _lift_nested_token(_parse_token_response(r), token_lift_key)


# ─── /connect ────────────────────────────────────────────────────────


@router.post("/connect/{connector_id}", response_model=ConnectStartResponse)
async def oauth_connect(
    connector_id: str,
    switch_account: bool = Query(
        default=False,
        description=(
            "When true, ask the provider to re-show its account chooser "
            "(OAuth `prompt=select_account`). Used by the connected-card "
            "'Switch account' action so the user can swap to a different "
            "Google/GitHub/etc. account without first disconnecting."
        ),
    ),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start an OAuth dance for (current_user, connector_id).

    Returns the URL the browser should navigate to. The frontend opens
    this in a popup; the user consents at the provider; the provider
    redirects back to /api/oauth/callback (handled below)."""
    _flag_or_503()

    # 0. Pre-flight: the user MUST have an agent_config row, otherwise
    #    the OAuth completes successfully and writes a connector_identity
    #    row that no agent will ever see — silent orphan that surfaces
    #    in chat as the agent insisting the tool isn't available.
    #    Burned by exactly this on 2026-05-10 — adding the guard so
    #    other users don't lose hours to the same trap.
    _agent_check = (
        await db.execute(
            select(AgentConfig.user_id).where(AgentConfig.user_id == str(current_user.id))
        )
    ).scalar_one_or_none()
    if _agent_check is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "reason": "no_agent_config",
                "message": (
                    "Your Toup agent isn't set up yet. Finish onboarding "
                    "and provision your agent before connecting external "
                    "tools — otherwise the OAuth row would be stored for "
                    "an account that has no agent to read it."
                ),
            },
        )

    # 1. Manifest lookup.
    entry = get_registry().get(connector_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown connector {connector_id!r}",
        )

    # 2. Provider-app config lookup. Async path checks the
    #    provider_app_credentials DB table first (admin-UI saves take
    #    effect immediately), then falls back to env-var registration.
    from app.services.provider_apps import get_provider_app_async
    app_cfg = await get_provider_app_async(entry.manifest.oauth.provider_app)
    if app_cfg is None:
        # Structured 503 — `provider` lets the frontend render the
        # inline setup form for the right OAuth client without parsing
        # the human message. `reason: "credentials_missing"` lets the
        # client distinguish this case from a generic 503.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "reason": "credentials_missing",
                "provider": entry.manifest.oauth.provider_app,
                "provider_label": entry.manifest.name,
                "message": (
                    f"{entry.manifest.name} OAuth credentials are not "
                    f"configured yet. The platform owner can paste them "
                    f"in the admin panel and you can connect immediately."
                ),
            },
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
        provider_name=app_cfg.name,
        force_account_selection=switch_account,
        extra_params=app_cfg.extra_authorize_params,
        scope_param=app_cfg.scope_param,
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


# The standard codes a provider may hand back on the authorize leg
# (RFC 6749 §4.1.2.1), in the user's words rather than the spec's. Any
# code not listed falls through to the frontend's generic copy.
_PROVIDER_ERROR_COPY = {
    "access_denied": "You cancelled the connection. Nothing was changed.",
    "invalid_scope": "This connector asked for a permission the provider refused.",
    "server_error": "The provider hit an error on their side. Please try again.",
    "temporarily_unavailable": "The provider is temporarily unavailable. Please try again shortly.",
}


def _bridge_error(
    code: str,
    *,
    connector_id: Optional[str] = None,
    message: Optional[str] = None,
) -> RedirectResponse:
    """Send a callback failure to the frontend bridge instead of raising.

    `connector_id` is not decoration. The bridge posts one `connectorId`
    field, and `useOAuthHandoff` drops any message whose id doesn't match
    the connector it opened the popup for — so an error redirect carrying
    only `?error=<code>` was discarded on arrival and the sheet sat until
    its cancellation timeout, reporting a backend failure to the user as
    "Cancelled — popup closed before completion". Always pass the
    connector when it is known.

    Raising HTTPException here is the other trap: the popup is a browser
    window, so a 4xx/5xx renders as a bare "Internal Server Error" page
    with no way back — which is precisely how the GitHub token-exchange
    crash presented for its entire life.
    """
    params = {"error": code}
    if connector_id:
        params["connector"] = connector_id
    if message:
        params["message"] = message
    return RedirectResponse(
        url=f"/oauth/callback-bridge?{urllib.parse.urlencode(params)}",
        status_code=status.HTTP_302_FOUND,
    )


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
        # Best-effort connector id so the bridge's message survives the
        # id match in `useOAuthHandoff` (see `_bridge_error`). The state
        # is untrusted at this point and there is no code to redeem, so a
        # failed verify costs only the connector label.
        _cid = None
        if state:
            try:
                _cid = verify_state(state).connector_id
            except Exception:
                # Deliberately broad. This lookup is best-effort and its
                # only output is a label, so nothing it can raise should
                # be able to fail the request it is decorating — and
                # `verify_state` did leak a UnicodeEncodeError past
                # `StateVerificationError` until it was fixed at the
                # source. Narrow this back and a second such leak turns
                # a branch that always redirected into an unhandled 500.
                pass
        # Same bridge route — popup gets the postMessage in error mode,
        # full-page fallback gets the URL params on the bridge page
        # and bounces back to /agent/integrations via the parent's
        # `?error=` handler when the bridge can't postMessage.
        #
        # Carry a sentence, because the frontend's fallback for a
        # message-less error is the literal string "OAuth flow failed" —
        # which is a lie for `access_denied`, by far the commonest code
        # here. The user pressed Cancel; nothing failed.
        return _bridge_error(error, connector_id=_cid, message=_PROVIDER_ERROR_COPY.get(error))

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

    # Defense in depth: the /connect entry already refused if no
    # agent_config existed, but a stale state token from a user whose
    # agent was deleted mid-flow could still land here. Refusing the
    # vault.put avoids the orphan-row class entirely. Mirror error
    # shape on /connect so the frontend bridge page can render the
    # same actionable message.
    _agent_check = (
        await db.execute(
            select(AgentConfig.user_id).where(AgentConfig.user_id == payload.user_id)
        )
    ).scalar_one_or_none()
    if _agent_check is None:
        await db.commit()  # commit the session-delete before bailing
        logger.warning(
            "[oauth.callback] user_id %s has no agent_config — refusing vault.put",
            payload.user_id[:8],
        )
        return _bridge_error(
            "no_agent_config",
            connector_id=payload.connector_id,
            message="Your agent isn't set up yet. Finish onboarding, then connect again.",
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
    # Same DB-first lookup as /connect — admin saves between connect
    # and callback take effect (they shouldn't, the OAuth state is
    # 10-minute TTL, but consistency wins).
    from app.services.provider_apps import get_provider_app_async
    app_cfg = await get_provider_app_async(entry.manifest.oauth.provider_app)
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
            use_pkce=app_cfg.use_pkce,
            token_auth_style=app_cfg.token_auth_style,
            token_body_format=app_cfg.token_body_format,
            extra_headers=app_cfg.extra_token_headers,
            token_lift_key=app_cfg.token_lift_key,
        )
    except (httpx.HTTPError, TokenResponseUnparseable) as e:
        await db.rollback()
        logger.exception(
            "[oauth.callback] token exchange failed connector=%s: %s",
            payload.connector_id, e,
        )
        return _bridge_error(
            "token_exchange_failed",
            connector_id=payload.connector_id,
            message=f"{entry.manifest.name} did not complete the handshake. Please try again.",
        )

    access_token = tokens.get("access_token")
    if not access_token:
        await db.rollback()
        # A 2xx is not success: GitHub answers HTTP 200 with an `error=`
        # body, so this branch is the real error channel for it. Log the
        # provider's own words and the KEYS only — never the values, a
        # partial response can still carry a refresh_token.
        provider_error = tokens.get("error") or "missing_access_token"
        provider_detail = tokens.get("error_description") or ""
        logger.error(
            "[oauth.callback] no access_token connector=%s provider_error=%s detail=%s keys=%s",
            payload.connector_id, provider_error, provider_detail, sorted(tokens.keys()),
        )
        return _bridge_error(
            "token_exchange_failed",
            connector_id=payload.connector_id,
            message=(
                provider_detail
                or f"{entry.manifest.name} refused the connection ({provider_error})."
            ),
        )

    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in")
    from datetime import datetime, timedelta
    expires_at = (
        datetime.utcnow() + timedelta(seconds=int(expires_in))
        if expires_in else None
    )
    # RFC 6749 §5.1 says space-delimited; GitHub sends `repo,read:user`.
    # A bare `.split()` stored that as ONE scope named "repo,read:user",
    # so the audit record of what the user actually granted was a string
    # no scope check could ever match. Split on either.
    granted_scopes = (
        [s for s in re.split(r"[,\s]+", tokens["scope"]) if s] if tokens.get("scope")
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

    # 6b. Gmail post-connect: arm a `users.watch` so any
    # `email_received` trigger the user creates (now or later) starts
    # firing as soon as Gmail produces a Pub/Sub event. Without this,
    # a freshly-connected Gmail has no subscription and the first
    # message after connect is silently dropped until the next daily
    # refresh tick. Fire-and-forget — the OAuth redirect must not
    # block on an external Google API call. Failures (token, scope,
    # GCP misconfig) are logged + retried by the daily refresh.
    if payload.connector_id == "gmail":
        _spawn_bg(_arm_gmail_watch_post_connect(payload.user_id))

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


# ─── /read-only ──────────────────────────────────────────────────────


@router.post("/{connector_id}/read-only", response_model=ReadOnlyResponse)
async def oauth_read_only(
    connector_id: str,
    req: ReadOnlyRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Toggle the per-identity read-only flag.

    `read_only=true`  → every mutating tool (`mutates: true` in the
                        manifest) disappears from the agent's tool list
                        and the dispatcher refuses any call to one as
                        defense-in-depth.
    `read_only=false` → tools come back the next time the agent
                        refreshes its tool cache (T1h push below).

    Drives the connected-card "Switch to read-only" action. Idempotent:
    repeated calls with the same value succeed and return the current
    state.
    """
    _flag_or_503()

    entry = get_registry().get(connector_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown connector {connector_id!r}",
        )

    user_id = str(current_user.id)

    # Identity must exist + be in a usable state. We allow toggling on
    # active and reauth_required identities (the user might be locking
    # down a connection they're about to re-auth) but reject revoked /
    # provider_down where the flag is meaningless.
    ident_row = (
        await db.execute(
            select(ConnectorIdentity)
            .where(ConnectorIdentity.user_id == user_id)
            .where(ConnectorIdentity.connector_id == connector_id)
        )
    ).scalar_one_or_none()
    if ident_row is None or ident_row.status not in ("active", "reauth_required"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active {connector_id!r} identity to toggle",
        )

    await db.execute(
        sa_update(ConnectorIdentity)
        .where(ConnectorIdentity.id == ident_row.id)
        .values(read_only=bool(req.read_only))
    )
    await db.commit()

    # Push the tool list to the tenant so write-tools disappear (or
    # reappear) without a page reload.
    from app.services.docker_host_service import refresh_tenant_tools
    await refresh_tenant_tools(db, user_id)
    logger.info(
        "[oauth.read_only] user=%s connector=%s → read_only=%s",
        user_id[:8], connector_id, bool(req.read_only),
    )
    return ReadOnlyResponse(connector_id=connector_id, read_only=bool(req.read_only))


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


# ─── Gmail post-connect watch arming (T2c) ───────────────────────────


async def _arm_gmail_watch_post_connect(user_id: str) -> None:
    """Fire-and-forget watch arm immediately after a successful Gmail
    OAuth callback. Writes `gmail_history_id` + `gmail_watch_expires_at`
    into `ConnectorIdentity.metadata_json` so the Pub/Sub webhook has
    a baseline to resume `history.list` from on the first push.

    Best-effort: every failure mode is recoverable by the daily refresh
    job. Logged at WARN (token/scope) or ERROR (unexpected) so ops can
    see when a user's first email after connect won't fire until
    midnight UTC.

    Idempotent: safe to call again if a retry mechanism ever fires it
    a second time — Gmail accepts a fresh watch over a live one and
    resets the clock.
    """
    from app.config import settings
    if not settings.gcp_project or not settings.pubsub_topic:
        logger.warning(
            "[oauth.gmail_post_connect] skipped user=%s — GCP not configured",
            user_id[:8],
        )
        return
    if not getattr(settings, "triggers_email_enabled", True):
        logger.info(
            "[oauth.gmail_post_connect] skipped user=%s — triggers feature off",
            user_id[:8],
        )
        return

    from app.services.gmail_pubsub import start_watch, GmailWatchError
    from app.api.triggers_provision import _write_watch_state
    from datetime import datetime, timezone

    try:
        handle = await start_watch(user_id)
    except GmailWatchError as e:
        # Common: token rot, scope miss, IAM. Don't crash the callback
        # — daily refresh is the safety net. BUT: notify the agent so
        # the trigger rows surface "Setup error" / "Reconnect Gmail"
        # to the user instead of silently sitting dormant until the
        # next refresh tick.
        msg = str(e)
        if "no active gmail identity" in msg.lower() or "token refresh failed" in msg.lower():
            err_code = "needs_reauth"
        elif "gcp_project or pubsub_topic unset" in msg.lower():
            err_code = "ops_blocked"
        elif "status=4" in msg.lower():
            err_code = "scope_error"
        else:
            err_code = "transient"
        logger.warning(
            "[oauth.gmail_post_connect] arm_failed user=%s err=%s code=%s",
            user_id[:8], msg[:200], err_code,
        )
        await _notify_agent_provision_failure(
            user_id, error=err_code, detail=msg[:300],
        )
        return
    except Exception as e:
        logger.exception(
            "[oauth.gmail_post_connect] arm_crash user=%s err=%s",
            user_id[:8], e,
        )
        await _notify_agent_provision_failure(
            user_id, error="transient", detail=f"unexpected: {type(e).__name__}",
        )
        return

    expires_iso = datetime.fromtimestamp(
        handle.expiration_unix_ms / 1000, tz=timezone.utc,
    ).isoformat()
    try:
        await _write_watch_state(
            user_id=user_id,
            connector_id="gmail",
            history_id=handle.history_id,
            expires_at_iso=expires_iso,
        )
    except Exception as e:
        logger.exception(
            "[oauth.gmail_post_connect] write_state_crash user=%s err=%s",
            user_id[:8], e,
        )
        return

    logger.info(
        "[oauth.gmail_post_connect] armed user=%s history_id=%s expires=%s",
        user_id[:8], handle.history_id, expires_iso,
    )

    # Persist provider_account_id (the Gmail address). The generic
    # vault.put at OAuth callback used to set this from the token
    # response, but Google's token endpoint doesn't return account/login
    # — those are GitHub-style fields. Result: provider_account_id stayed
    # NULL on every Gmail OAuth, and the Pub/Sub webhook's email→user
    # resolver (`_resolve_user_for_email`) returned None, silently
    # dropping every push as `unknown_email`. We fetch the actual email
    # via Gmail's users.getProfile and write it here so the next push
    # resolves correctly. Idempotent — every arm rewrites with the same
    # address. (2026-05-17 — burned hours on the silent-drop diagnosis.)
    try:
        from sqlalchemy import update as _sa_update
        from app.db.database import async_session_maker as _asm
        from app.db.models import ConnectorIdentity as _CI
        from app.services.gmail_pubsub import (
            _resolve_token_for as _gmail_token,
            GMAIL_API_BASE as _gmail_base,
        )
        import httpx as _httpx
        access = await _gmail_token(user_id)
        async with _httpx.AsyncClient(timeout=10.0) as _c:
            _r = await _c.get(
                f"{_gmail_base}/profile",
                headers={"Authorization": f"Bearer {access}"},
            )
        if _r.status_code == 200:
            _email = (_r.json().get("emailAddress") or "").strip().lower()
            if _email:
                async with _asm() as _db:
                    await _db.execute(
                        _sa_update(_CI).where(
                            _CI.user_id == user_id,
                            _CI.connector_id == "gmail",
                        ).values(provider_account_id=_email)
                    )
                    await _db.commit()
                logger.info(
                    "[oauth.gmail_post_connect] provider_account_id_set user=%s email=%s",
                    user_id[:8], _email,
                )
        else:
            logger.warning(
                "[oauth.gmail_post_connect] getProfile status=%s for user=%s",
                _r.status_code, user_id[:8],
            )
    except Exception as _e:
        # Best-effort — the watch is already armed, the daily refresh
        # will re-run this path. Don't crash the callback.
        logger.warning(
            "[oauth.gmail_post_connect] provider_account_id write failed user=%s err=%s",
            user_id[:8], _e,
        )


async def _notify_agent_provision_failure(
    user_id: str, *, error: str, detail: str,
) -> None:
    """POST to the agent's `/api/triggers/_record_provision_failure`
    endpoint so the user's email_received triggers flip to
    `last_status='provisioning_failed'` with a specific `provision_error`
    code. Without this, an arm failure was invisible on the dashboard
    until the user manually clicked Verify or the 6h refresh fan-out
    (Gap 1) ran.

    Best-effort: if the agent is unreachable we log and return — the
    daily refresh will re-attempt arming, and Gap 1's fan-out will
    clear stale provision_error fields when arming succeeds.
    """
    import httpx
    from app.services.trigger_dispatch import resolve_agent_target, TriggerDispatchError

    try:
        agent_url, agent_key = await resolve_agent_target(user_id)
    except TriggerDispatchError as e:
        logger.warning(
            "[oauth.notify_agent] no agent target user=%s err=%s",
            user_id[:8], e,
        )
        return

    url = f"{agent_url.rstrip('/')}/api/triggers/_record_provision_failure"
    payload = {
        "user_id": user_id,
        "connector_id": "gmail",
        "error": error,
        "detail": detail,
    }
    headers = {"X-Agent-Key": agent_key, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.RequestError as e:
        logger.warning(
            "[oauth.notify_agent] transport failed user=%s err=%s",
            user_id[:8], e,
        )
        return
    if r.status_code != 200:
        logger.warning(
            "[oauth.notify_agent] non-200 user=%s status=%d body=%s",
            user_id[:8], r.status_code, r.text[:200],
        )
        return
    try:
        body = r.json()
        logger.info(
            "[oauth.notify_agent] recorded user=%s triggers_updated=%s",
            user_id[:8], body.get("triggers_updated"),
        )
    except Exception:
        pass
