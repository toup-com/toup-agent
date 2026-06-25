"""Authentication endpoints"""

import asyncio
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
from jose import jwt, JWTError

from app.db import get_db
from app.db.models import UserSession
from app.schemas import (
    UserCreate, UserLogin, UserResponse, Token,
    ChangePasswordRequest, UpdateProfileRequest,
)
from app.services import (
    authenticate_user, create_user, get_user_by_email,
    get_user_by_id, create_access_token, decode_access_token,
    verify_password, change_user_password,
)
from app.services.rate_limiter import login_rate_limiter, signup_rate_limiter
from app.services.turnstile import verify_turnstile_token
from app.services.session_tracker import (
    parse_device_label, record_login_session, JTI_REVOCATION_GRACE_SECONDS,
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)

# Fire-and-forget background tasks (e.g. the warm-pool claim on OAuth signup).
# Strong refs are held here so the event loop can't GC a still-running task; the
# done-callback discards them on completion.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _bg_finalize_signup(user_id: str) -> None:
    """Finish a fresh OAuth signup OFF the request path: mint the free-tier
    bundle creds, then claim the warm pool.

    Both used to run inline in the Google callback. The free-tier activation in
    particular provisions a per-user OpenAI project via a SLOW synchronous
    Admin-API call (~3-6s) — pure dead weight on the redirect the user waits on.
    Neither has to finish before that redirect: the first chat message is ~30s
    out (after onboarding), and the onboarding flow re-activates the free tier
    as a backstop. Runs on its own DB session, with activation BEFORE the claim
    so the bridge bind still injects TOUP_TOKEN on first boot. Idempotent;
    failures only log.
    """
    try:
        from app.db import async_session_maker
        from app.services.free_tier_activation import activate_free_tier
        from app.services.pool_service import claim_or_prewarm
        async with async_session_maker() as bg_db:
            try:
                await activate_free_tier(bg_db, user_id, force_env_push=False)
            except Exception as ae:
                logger.warning(
                    "[google-auth] bg free-tier activation failed user=%s: %s",
                    user_id[:8], ae,
                )
            await claim_or_prewarm(bg_db, user_id)
    except Exception as e:
        logger.warning(
            "[google-auth] background finalize failed user=%s: %s",
            user_id[:8], e,
        )


async def _bg_apple_refresh_capture(user_id: str, authorization_code: str) -> None:
    """Capture the Apple refresh token OFF the sign-in path.

    Needed only later for account-deletion revocation (Guideline 5.1.1(v)). The
    exchange is an httpx round-trip to Apple (up to 10s) — the user shouldn't
    wait on it at sign-in. The one-time code is exchanged on the next loop tick,
    well inside its lifetime. Runs on its own DB session. Best-effort; a failure
    only means we can't proactively revoke at deletion (delete_account tolerates
    it), so it only logs.
    """
    try:
        from app.services.apple_auth import (
            exchange_authorization_code, siwa_revocation_configured,
        )
        if not siwa_revocation_configured():
            return
        tokens = await exchange_authorization_code(authorization_code)
        refresh = (tokens or {}).get("refresh_token")
        if not refresh:
            return
        from app.db import async_session_maker
        from app.services import get_user_by_id
        from app.services.credential_crypto import encrypt_str
        async with async_session_maker() as bg_db:
            user = await get_user_by_id(bg_db, user_id)
            if user is None:
                return
            user.apple_refresh_token = encrypt_str(refresh)
            await bg_db.commit()
    except Exception as e:
        logger.warning(
            "[apple-auth] background refresh-token capture failed user=%s: %s",
            user_id[:8], e,
        )

# SSO cookie config
SSO_COOKIE_NAME = "hex_sso_token"
SSO_COOKIE_DOMAIN = ".toup.ai"
SSO_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 1 week


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Dependency to get the current authenticated user.
    Checks Bearer token first, then falls back to SSO cookie.
    Validates token was issued after last password change."""
    token = None
    user_id = None

    # 1. Try Bearer token
    if credentials and credentials.credentials:
        token = credentials.credentials
        user_id = decode_access_token(token)

    # 2. Fall back to SSO cookie
    if not user_id:
        token = request.cookies.get(SSO_COOKIE_NAME)
        if token:
            user_id = decode_access_token(token)

    # 3. Fall back to agent mode
    if not user_id:
        agent_key = request.headers.get("x-agent-key", "")
        if settings.agent_api_key and agent_key == settings.agent_api_key and settings.user_id:
            user_id = settings.user_id
            token = None  # Skip token revocation check for agent mode

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_user_by_id(db, user_id)
    if not user:
        # In agent mode, auto-create a stub user
        if settings.agent_api_key and settings.user_id and user_id == settings.user_id:
            from app.db.models import User
            user = User(id=user_id, email=f"{user_id[:8]}@agent.local", hashed_password="", name="Agent Owner")
            db.add(user)
            await db.commit()
            await db.refresh(user)
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled")

    # Token revocation: reject tokens issued before last password change.
    # Also: per-session revocation via the UserSession table (account
    # page "sign out this device"). The two checks are complementary —
    # password change scorches everything via password_changed_at; the
    # per-session table lets users sign out one device at a time.
    if token and getattr(user, 'password_changed_at', None):
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
            token_iat = datetime.utcfromtimestamp(payload.get("iat", 0))
            if token_iat < user.password_changed_at:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token invalidated by password change. Please log in again.",
                )
        except JWTError:
            pass  # Token already validated by decode_access_token

    # Per-session revocation. Skip for agent-mode (no token, no jti).
    # The grace window swallows the race where login returns the JWT
    # before the session-row commit lands in the read connection.
    if token:
        try:
            payload = jwt.decode(
                token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
            )
            jti = payload.get("jti")
            token_iat = datetime.utcfromtimestamp(payload.get("iat", 0))
            if jti:
                from app.services.session_tracker import (
                    get_session_by_jti, maybe_bump_last_seen,
                )
                session_row = await get_session_by_jti(db, jti)
                if session_row is None:
                    # No matching session row. Could be: legacy token
                    # issued before the table existed (let through —
                    # grace), or a freshly-issued token whose commit
                    # hasn't been observed yet (grace window), or a
                    # forged/tampered jti (reject after grace).
                    age = (datetime.utcnow() - token_iat).total_seconds()
                    if age > JTI_REVOCATION_GRACE_SECONDS:
                        # Treat as legacy/missing. Don't reject —
                        # we'd nuke every pre-rollout session. The
                        # password_changed_at gate above is still the
                        # blunt-but-reliable revocation lever.
                        pass
                elif session_row.is_revoked:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="This session was signed out. Please log in again.",
                    )
                else:
                    # Live session — keep last_seen_at fresh.
                    request.state.user_session_jti = jti  # for endpoints that need it
                    await maybe_bump_last_seen(db, session_row)
        except HTTPException:
            raise
        except JWTError:
            pass

    return user


# ── Register ─────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user.

    Phase-3 hardening (docs/onboarding/prewarm-phase0.md):
      * IP-keyed rate limit (5 signups per IP per hour by default).
        Caps the "spam signups burn pre-warmed containers" attack now
        that Soul.save provisions a Docker container.
      * Cloudflare Turnstile verification when TURNSTILE_SECRET_KEY is
        set. Skipped in dev / CI to keep tests deterministic.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Rate-limit FIRST so we don't waste DB / Turnstile work on
    # already-blocked IPs.
    retry_after = signup_rate_limiter.check(client_ip)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail=f"Too many signup attempts from this network. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Turnstile gate — skipped when no secret configured (dev / CI).
    # Verifies BEFORE we touch the DB so a failing token never hits
    # the User uniqueness constraint.
    if not await verify_turnstile_token(
        user_data.cf_turnstile_token, remote_ip=client_ip,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CAPTCHA verification failed. Please refresh and try again.",
        )

    existing = await get_user_by_email(db, user_data.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    # Record the attempt only after both gates pass — we don't count a
    # bot's failed Turnstile attempts toward the rate limit (otherwise
    # one bot can lock out a shared IP). Successful create is the
    # signal that consumes a slot.
    user = await create_user(db, email=user_data.email, password=user_data.password, name=user_data.name)
    signup_rate_limiter.record(client_ip)

    # Kick off the prewarm IMMEDIATELY at signup — before the user
    # even navigates to /onboarding/welcome. Previously the prewarm
    # fired from OnboardingShell.mount, which was 1-2s later (auto-
    # login round-trip + React Router navigate + chunk hydrate +
    # mount). Moving it here:
    #   - Cuts ~1-2s off the boot budget the user-visible flow can
    #     overlap with prewarm. By Install time, the container is
    #     ~1-2s further along its boot.
    #   - Provides a more deterministic start point for telemetry —
    #     [PREWARM-START] fires synchronously with the User row insert.
    # Fire-and-forget; failures are logged + non-blocking. The
    # OnboardingShell-mount fallback at OnboardingShell.tsx remains in
    # place as a defense (schedule_prewarm short-circuits on already-
    # running/provisioning containers, so the second call is a cheap
    # no-op rather than a duplicate provision).
    if settings.prewarm_on_soul_save:
        try:
            from app.api.agent_setup import _get_or_create_config
            config = await _get_or_create_config(str(user.id), db)
            # Default to managed for unified onboarding. The signup
            # form has no hosting choice, and the frontend's
            # Welcome.advance writes hosting_mode='managed' anyway —
            # priming here lets the prewarm container boot once with
            # the correct env so we don't recreate at Welcome→Soul.
            dirty = False
            if config.hosting_mode in (None, "self-hosted"):
                config.hosting_mode = "managed"
                dirty = True
            if not config.whatsapp_mode:
                config.whatsapp_mode = "qr_link"
                dirty = True
            if dirty:
                await db.commit()
            # Hand the free-tier activation AND the bridge claim to the
            # background so neither blocks the signup response (mirrors the
            # Google/Apple flow). Activation mints the bundle creds + provisions
            # a per-user OpenAI project (a slow sync Admin-API call, ~3-6s) and
            # claim_or_prewarm awaits the bridge bind (up to 30s) — none of which
            # the user waits on: the first chat is post-onboarding,
            # _bg_finalize_signup activates BEFORE it claims so the bind still
            # injects TOUP_TOKEN, and onboarding re-activates the free tier as a
            # backstop.
            _spawn_background(_bg_finalize_signup(str(user.id)))
        except Exception as e:
            logger.warning(
                "[REGISTER] Prewarm/claim schedule failed for user %s: %s",
                str(user.id)[:8], e,
            )
    return user


# ── Google OAuth sign-in / sign-up ───────────────────────────────
#
# Lets users skip the username/password form by clicking "Continue
# with Google." Same Google OAuth client we already use for the Gmail
# connector — different scope set (openid + email + profile, no
# Gmail access). One callback URL `/api/auth/google/callback`, which
# MUST be registered as an Authorized Redirect URI in the same Google
# Cloud Console project that owns GOOGLE_OAUTH_CLIENT_ID.
#
# Flow:
#   1. Frontend → GET /api/auth/google/start?mode=signup&redirect=...
#      → 302 to https://accounts.google.com/o/oauth2/v2/auth?... with
#      HMAC-signed state carrying mode + redirect.
#   2. User consents → Google → GET /api/auth/google/callback?code&state
#   3. Backend exchanges code for token, fetches userinfo, finds-or-
#      creates the User, issues a JWT, redirects to the frontend at
#      /auth/google/complete#token=<jwt>&new=<1|0>&redirect=<path>.
#   4. Frontend reads the hash, stores the JWT, navigates the user to
#      /onboarding/welcome (new) or the redirect param (returning).
#
# CSRF protection: state is HMAC-signed with the existing oauth_state
# secret; tampered or expired states are rejected before we touch the
# DB. Same primitive used by the connector OAuth path.
#
# Email verification: Google's userinfo carries `email_verified=true`
# for verified accounts; we stamp User.email_verified_at on signup so
# the email-verification gate (F13) doesn't block the new user from
# spending their credits.

import base64 as _base64
import hashlib as _hashlib
import hmac as _hmac
import json as _json
import secrets as _secrets
import time as _time

_GOOGLE_AUTH_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_AUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_AUTH_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
# OpenID Connect minimum scope set — `openid` triggers OIDC mode,
# `email` and `profile` give us the address + display name. No Gmail
# or Drive scopes — sign-in is read-only of the user's identity.
_GOOGLE_AUTH_SCOPES = "openid email profile"


def _signin_state_secret() -> bytes:
    """Same JWT secret reused for HMAC-signing the sign-in state token.
    Distinct purpose from the JWT (no claims, no expiration encoded in
    the token itself — we add iat + verify a TTL on the server side),
    same trust root. Rotating jwt_secret invalidates in-flight sign-ins,
    which is the same blast radius rotating the JWT has — acceptable."""
    raw = (settings.jwt_secret or "").strip()
    if not raw:
        raise HTTPException(500, "JWT_SECRET not configured")
    return raw.encode("utf-8")


def _b64url_encode(b: bytes) -> str:
    return _base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return _base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _sign_signin_state(*, mode: str, redirect: Optional[str], platform: str = "web") -> str:
    """Mint a state token carrying (mode, redirect, platform, nonce, iat).

    Format: `<body_b64>.<sig_b64>`. Same shape as
    services/oauth_state.py uses for the connector flow — distinct
    payload keys so a cross-flow replay is rejected by the
    `kind` field check on verify.

    `platform` selects the post-callback redirect target: "web" lands on
    the SPA at /auth/google/complete, "native" deep-links back into the
    mobile app via the toup:// scheme. It only picks the redirect — it
    never grants auth — so it's safe even though the app passes it."""
    payload = {
        "kind": "auth_signin",
        "mode": (mode or "signup").strip().lower()[:16],
        "redirect": (redirect or "").strip()[:512] or None,
        "platform": "native" if (platform or "web").strip().lower() == "native" else "web",
        "nonce": _secrets.token_urlsafe(16),
        "iat": int(_time.time()),
    }
    body = _json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body_b64 = _b64url_encode(body)
    sig = _hmac.new(_signin_state_secret(), body_b64.encode("ascii"), _hashlib.sha256).digest()
    return f"{body_b64}.{_b64url_encode(sig)}"


def _verify_signin_state(token: str, *, max_age_seconds: int = 600) -> dict:
    """Parse + verify the sign-in state. Raises HTTPException on any
    failure so the callback can fall straight through to its error
    redirect."""
    if not token or "." not in token:
        raise HTTPException(400, "invalid_state")
    try:
        body_b64, sig_b64 = token.rsplit(".", 1)
    except ValueError:
        raise HTTPException(400, "invalid_state")
    expected = _hmac.new(_signin_state_secret(), body_b64.encode("ascii"), _hashlib.sha256).digest()
    try:
        provided = _b64url_decode(sig_b64)
    except Exception:
        raise HTTPException(400, "invalid_state")
    if not _hmac.compare_digest(expected, provided):
        raise HTTPException(400, "invalid_state")
    try:
        payload = _json.loads(_b64url_decode(body_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(400, "invalid_state")
    if not isinstance(payload, dict) or payload.get("kind") != "auth_signin":
        raise HTTPException(400, "wrong_state_kind")
    iat = payload.get("iat")
    if not isinstance(iat, int):
        raise HTTPException(400, "invalid_state")
    age = int(_time.time()) - iat
    if age < -60 or age > max_age_seconds:
        raise HTTPException(400, "state_expired")
    return payload


def _google_auth_callback_url() -> str:
    """The exact redirect_uri we send to Google AND register in the
    Cloud Console. Override via GOOGLE_AUTH_CALLBACK_URL if a deploy
    fronts the API at a non-default host."""
    override = (getattr(settings, "google_auth_callback_url", "") or "").strip()
    if override:
        return override
    base = (settings.app_public_base_url or "https://toup.ai").rstrip("/")
    return f"{base}/api/auth/google/callback"


async def _google_auth_credentials() -> tuple[str, str]:
    """Resolve (client_id, client_secret).

    Reads from the same source the Gmail connector uses
    (`provider_apps.get_provider_app_async`), which transparently
    consults the `provider_app_credentials` table FIRST then falls back
    to the env-var snapshot. The admin UI saves Google credentials to
    that table — reusing the resolver means a single Google OAuth
    client powers both connector OAuth (Gmail / Drive / Calendar) and
    sign-in OAuth without the operator having to set duplicate env
    vars.

    Raises HTTPException(400) on missing config — NOT 503 — because
    Cloudflare's origin-shield Worker substitutes a branded "Updating
    Toup" page for any 5xx response on toup.ai/*. A 400 surfaces the
    real reason to the operator; sign-in misconfiguration is a
    client/config problem, not a server outage.
    """
    from app.services.provider_apps import get_provider_app_async
    app_cfg = await get_provider_app_async("google")
    cid = (app_cfg.client_id if app_cfg else "").strip()
    csec = (app_cfg.client_secret if app_cfg else "").strip()
    if not cid or not csec:
        raise HTTPException(
            status_code=400,
            detail=(
                "Google sign-in isn't configured. An operator needs to "
                "save Google OAuth credentials via the admin Connectors "
                "UI (or set GOOGLE_OAUTH_CLIENT_ID + "
                "GOOGLE_OAUTH_CLIENT_SECRET in env) AND add "
                f"{_google_auth_callback_url()} to the Google Cloud "
                "Console project's Authorized Redirect URIs."
            ),
        )
    return cid, csec


@router.get("/google/start")
async def google_auth_start(
    request: Request,
    mode: str = "signup",
    redirect: Optional[str] = None,
    platform: str = "web",
):
    """Begin the Google OAuth flow. Redirects to Google's consent screen.

    `mode` is informational only (sign-in vs sign-up behave identically
    server-side — if the email is new we create the user, if it's known
    we just log them in). The frontend uses it to choose post-callback
    copy ("Welcome!" vs "Welcome back!"); we round-trip it through state.

    Returns JSON `{"authorize_url": ...}` so the frontend can do a
    top-level `window.location.assign(authorize_url)`. We deliberately
    do NOT 307-redirect from this endpoint anymore:

      * Cloudflare's origin-shield Worker on toup.ai/* substitutes a
        branded "Updating Toup" page for ANY 5xx response. Letting the
        frontend handle the navigation puts a same-origin XHR / fetch
        in front of that Worker — the Worker only catches navigations,
        so a fetch sees the real backend response (errors and all).
      * Browser back-button behavior is cleaner: the user is never
        parked on `/api/auth/google/start` in their history.
    """
    import urllib.parse

    client_id, _ = await _google_auth_credentials()
    signed_state = _sign_signin_state(mode=mode, redirect=redirect, platform=platform)

    params = {
        "client_id": client_id,
        "redirect_uri": _google_auth_callback_url(),
        "response_type": "code",
        "scope": _GOOGLE_AUTH_SCOPES,
        "state": signed_state,
        # `select_account` so users with multiple Google accounts see
        # the chooser rather than getting silently signed in with the
        # last-used account. Matches the GitHub `/select_account` UX.
        "prompt": "select_account",
        # `access_type=online` — we don't need refresh tokens here.
        # The user signs in, we issue our own JWT, Google's token is
        # discarded immediately. Refresh tokens would just be cruft.
        "access_type": "online",
    }
    authorize_url = (
        f"{_GOOGLE_AUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    )
    return {"authorize_url": authorize_url}


# Deep link the mobile app registers (CFBundleURLSchemes: "toup" in the
# app's Info.plist). The callback redirects here — instead of the web
# SPA — when the sign-in was started with platform=native, so
# ASWebAuthenticationSession on the device catches the redirect and hands
# the JWT (in the fragment) back to the app.
_NATIVE_AUTH_COMPLETE_URL = "toup://auth/google/complete"


def _peek_state_platform(state: Optional[str]) -> str:
    """Best-effort read of `platform` from an UNVERIFIED state token.

    The early error branches (provider cancel, missing code, bad state)
    bail before the signature is verified, but still need to pick the
    right redirect target so the mobile session catches the bounce-back.
    `platform` only selects web-URL vs toup:// — it never grants auth —
    so reading it pre-verification is safe. Defaults to "web"."""
    if not state or "." not in state:
        return "web"
    try:
        body_b64 = state.rsplit(".", 1)[0]
        payload = _json.loads(_b64url_decode(body_b64).decode("utf-8"))
        return "native" if (payload.get("platform") or "web") == "native" else "web"
    except Exception:
        return "web"


def _frontend_complete_url(*, token: str, is_new: bool, redirect: Optional[str], error: Optional[str] = None, platform: str = "web") -> str:
    """Build the post-callback redirect URL. The JWT lives in the URL
    fragment so it stays out of HTTP logs / referer headers (the
    fragment isn't sent to the server on the GET that loads the page).

    When `platform == "native"` the landing is the app's toup:// deep
    link rather than the web SPA, so the mobile auth session resolves."""
    import urllib.parse
    if (platform or "web") == "native":
        landing = _NATIVE_AUTH_COMPLETE_URL
    else:
        base = (settings.app_public_base_url or "https://toup.ai").rstrip("/")
        landing = f"{base}/auth/google/complete"
    parts: list[str] = []
    if token:
        parts.append(f"token={urllib.parse.quote(token)}")
    parts.append(f"new={'1' if is_new else '0'}")
    if redirect:
        parts.append(f"redirect={urllib.parse.quote(redirect)}")
    if error:
        parts.append(f"error={urllib.parse.quote(error)}")
    return f"{landing}#{'&'.join(parts)}"


def _pick_google_display_name(info: dict, email: str) -> str:
    # Google's userinfo can ship any subset of {given_name, family_name,
    # name}. Prefer the structured pair when we have at least one half —
    # joining lets locales that order surname-first (e.g. Japanese,
    # Hungarian) keep Google's own ordering rather than us guessing.
    # Final fallback is the email local-part so the User row's `name`
    # column never lands NULL on a Google signup (admin user list +
    # avatar initials both render `name || email`, so NULL works, but
    # populated is friendlier and matches what the user typed into
    # Google).
    #
    # No transliteration: `name` is stored as UTF-8 end-to-end so
    # Persian / Arabic / CJK names render correctly in the admin panel
    # and chat sidebar. Only sanitisation is whitespace trimming.
    given = (info.get("given_name") or "").strip()
    family = (info.get("family_name") or "").strip()
    if given or family:
        return f"{given} {family}".strip()
    full = (info.get("name") or "").strip()
    if full:
        return full
    local = (email or "").split("@", 1)[0].strip()
    # Cap to the column's String(255) limit defensively; Gmail addresses
    # max out at 64 chars locally so this is mostly belt-and-braces.
    return local[:64] if local else "User"


@router.get("/google/callback")
async def google_auth_callback(
    request: Request,
    response: Response,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Handle Google's OAuth redirect. Exchanges the code for tokens,
    fetches userinfo, finds or creates the User, issues a Toup JWT,
    and redirects the browser to the frontend's `/auth/google/complete`
    landing with the JWT in the URL fragment.

    All failure paths redirect with `error=<reason>` in the fragment so
    the frontend can surface a usable message — never returns a bare
    500 or a JSON blob the user can't read.
    """
    import httpx

    # Pick the redirect target (web SPA vs toup:// deep link) up front so
    # even the pre-verification error branches bounce back to the right
    # place. Re-derived authoritatively from the verified payload below.
    platform = _peek_state_platform(state)

    def _dest(**kw):
        return _frontend_complete_url(platform=platform, **kw)

    # Provider-side error — user clicked "Cancel" on Google's screen.
    if error:
        return Response(
            status_code=307,
            headers={"Location": _dest(
                token="", is_new=False, redirect=None, error=error,
            )},
        )
    if not code or not state:
        return Response(
            status_code=307,
            headers={"Location": _dest(
                token="", is_new=False, redirect=None,
                error="missing_code_or_state",
            )},
        )

    client_id, client_secret = await _google_auth_credentials()

    # Verify state — rejects expired (>10min) and tampered states.
    try:
        payload = _verify_signin_state(state)
    except HTTPException as e:
        logger.warning("[google-auth] state verification failed: %s", e.detail)
        return Response(
            status_code=307,
            headers={"Location": _dest(
                token="", is_new=False, redirect=None,
                error=str(e.detail or "invalid_state"),
            )},
        )
    caller_redirect = payload.get("redirect")
    # Authoritative platform from the verified state (overrides the
    # best-effort peek above; `_dest` closes over this name).
    platform = "native" if payload.get("platform") == "native" else "web"

    # Exchange code → tokens.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(
                _GOOGLE_AUTH_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": _google_auth_callback_url(),
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if token_resp.status_code != 200:
            logger.warning(
                "[google-auth] token exchange %d: %s",
                token_resp.status_code, token_resp.text[:300],
            )
            return Response(
                status_code=307,
                headers={"Location": _dest(
                    token="", is_new=False, redirect=caller_redirect,
                    error="token_exchange_failed",
                )},
            )
        tokens = token_resp.json()
    except Exception:
        logger.exception("[google-auth] token exchange exception")
        return Response(
            status_code=307,
            headers={"Location": _dest(
                token="", is_new=False, redirect=caller_redirect,
                error="token_exchange_failed",
            )},
        )

    access_token = (tokens or {}).get("access_token", "")
    if not access_token:
        return Response(
            status_code=307,
            headers={"Location": _dest(
                token="", is_new=False, redirect=caller_redirect,
                error="no_access_token",
            )},
        )

    # Fetch userinfo (email + name).
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            info_resp = await client.get(
                _GOOGLE_AUTH_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if info_resp.status_code != 200:
            logger.warning(
                "[google-auth] userinfo %d: %s",
                info_resp.status_code, info_resp.text[:300],
            )
            return Response(
                status_code=307,
                headers={"Location": _dest(
                    token="", is_new=False, redirect=caller_redirect,
                    error="userinfo_failed",
                )},
            )
        info = info_resp.json() or {}
    except Exception:
        logger.exception("[google-auth] userinfo exception")
        return Response(
            status_code=307,
            headers={"Location": _dest(
                token="", is_new=False, redirect=caller_redirect,
                error="userinfo_failed",
            )},
        )

    email = (info.get("email") or "").strip().lower()
    if not email:
        return Response(
            status_code=307,
            headers={"Location": _dest(
                token="", is_new=False, redirect=caller_redirect,
                error="no_email",
            )},
        )
    email_verified = bool(info.get("email_verified", False))
    display_name = _pick_google_display_name(info, email)

    # Find or create user.
    existing = await get_user_by_email(db, email)
    is_new = False
    if existing is not None:
        user = existing
    else:
        # OAuth signup — set a random unguessable password so the user
        # can never log in via /login until they explicitly set one via
        # the password-change flow. (`get_password_hash` would otherwise
        # be called with an empty string which the password verifier
        # would reject — but defense-in-depth: random secret.)
        random_password = _secrets.token_urlsafe(48)
        user = await create_user(
            db, email=email, password=random_password, name=display_name,
        )
        is_new = True
        # Google verified the email → stamp it. The credit-verification
        # gate (F13) trusts this flag for charge eligibility.
        if email_verified:
            user.email_verified_at = datetime.utcnow()
            await db.commit()
            await db.refresh(user)

        # Same prewarm-on-signup behavior as the /register path so the
        # tenant container is warming while the user finishes onboarding.
        if settings.prewarm_on_soul_save:
            try:
                from app.api.agent_setup import _get_or_create_config
                config = await _get_or_create_config(str(user.id), db)
                dirty = False
                if config.hosting_mode in (None, "self-hosted"):
                    config.hosting_mode = "managed"
                    dirty = True
                if not config.whatsapp_mode:
                    config.whatsapp_mode = "qr_link"
                    dirty = True
                if dirty:
                    await db.commit()
                # Hand the free-tier activation AND the bridge claim to the
                # background so neither blocks the OAuth redirect. Activation
                # mints the bundle creds and provisions a per-user OpenAI project
                # (a slow sync Admin-API call, ~3-6s) — none of which the user
                # waits on: the first chat message is ~30s out (post-onboarding),
                # the finalize runs activation BEFORE the claim so the bind still
                # injects TOUP_TOKEN, and the onboarding flow re-activates the
                # free tier as a backstop.
                _spawn_background(_bg_finalize_signup(str(user.id)))
            except Exception as e:
                logger.warning(
                    "[google-auth] prewarm/claim failed user=%s: %s",
                    str(user.id)[:8], e,
                )

    if not user.is_active:
        return Response(
            status_code=307,
            headers={"Location": _dest(
                token="", is_new=False, redirect=caller_redirect,
                error="account_disabled",
            )},
        )

    # Issue a Toup JWT. Same path as the /login endpoint — record the
    # session so the per-device "Sign out" UI sees this login.
    jwt_token = create_access_token(user.id)
    try:
        await record_login_session(db, user.id, jwt_token, request)
    except Exception:
        # Session-tracker is non-critical for the auth flow — log and
        # continue. The user can still log out via password change.
        logger.exception("[google-auth] record_login_session failed")

    # Set the SSO cookie too, matching the /login + /register UX so
    # cross-subdomain navigation stays authenticated without the
    # frontend having to forward the bearer token via JS.
    response.set_cookie(
        key=SSO_COOKIE_NAME, value=jwt_token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True,
        samesite="none", path="/",
    )

    redirect_url = _dest(
        token=jwt_token, is_new=is_new, redirect=caller_redirect,
    )
    return Response(
        status_code=307,
        headers={
            "Location": redirect_url,
            "Cache-Control": "no-store",
            **{k: v for k, v in response.headers.items() if k.lower() == "set-cookie"},
        },
    )


# ── Login (with rate limiting) ───────────────────────────────────

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """Login and get access token. Rate-limited: 5 attempts per 5 minutes."""
    login_id = credentials.email.strip()
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit check
    retry_after = login_rate_limiter.check(client_ip, login_id)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    user = await authenticate_user(db, login_id, credentials.password)
    if not user and '@' not in login_id:
        user = await authenticate_user(db, f"{login_id}@toup.ai", credentials.password)

    if not user:
        login_rate_limiter.record(client_ip, login_id)
        # Check if user exists at all to give a more helpful message
        from app.services import get_user_by_email
        exists = await get_user_by_email(db, login_id)
        if not exists and '@' not in login_id:
            exists = await get_user_by_email(db, f"{login_id}@toup.ai")
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No account found with this email. Please sign up first.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    login_rate_limiter.clear(client_ip, login_id)
    token = create_access_token(user.id)

    # Record the session BEFORE returning the token. This guarantees
    # the row is committed before the client uses the JWT to call
    # protected endpoints — avoids the otherwise-tiny race window where
    # get_current_user sees a jti with no matching row. The grace
    # window in session_tracker is a belt-and-suspenders safeguard.
    await record_login_session(db, user.id, token, request)

    response.set_cookie(
        key=SSO_COOKIE_NAME, value=token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True,
        samesite="none", path="/",
    )
    return Token(access_token=token)


# ── Sign in with Apple (native, iOS) ─────────────────────────────
# Guideline 4.8: the privacy-preserving login required because the app
# also offers Google Sign-In. The iOS client runs AppleAuthentication
# .signInAsync() and POSTs the resulting identity_token here; we verify
# it against Apple's JWKS, link/create the user by email (mirroring the
# Google flow), and return a Toup JWT. No web redirect (native exchange).
class AppleAuthRequest(BaseModel):
    identity_token: str
    authorization_code: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None


class AppleAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    is_new: bool = False


@router.post("/apple", response_model=AppleAuthResponse)
async def apple_auth(
    body: AppleAuthRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Sign in with Apple. Verify Apple's identity token, link/create the
    user by email, return a Toup JWT (Guideline 4.8)."""
    import secrets as _secrets
    from app.services.apple_auth import (
        verify_apple_identity_token, AppleTokenError,
    )

    try:
        claims = verify_apple_identity_token(
            body.identity_token, settings.apple_client_id,
        )
    except AppleTokenError as e:
        logger.warning("[apple-auth] token verification failed: %s", e)
        raise HTTPException(status_code=401, detail="Apple sign-in verification failed.")

    # Apple includes the email in the identity token when the user grants
    # it (a private-relay address for Hide-My-Email); fall back to the
    # email the client forwarded from the first-authorization credential.
    email = (claims.get("email") or body.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Apple did not provide an email address.")
    ev = claims.get("email_verified")
    email_verified = (ev is True) or (str(ev).lower() == "true")

    existing = await get_user_by_email(db, email)
    is_new = False
    if existing is not None:
        user = existing
    else:
        # OAuth signup — random unguessable password (no /login until the
        # user sets one via password-change). Mirrors the Google path.
        random_password = _secrets.token_urlsafe(48)
        # Apple sends `fullName` ONLY on the very first authorization for an
        # (Apple ID, app) pair; every later sign-in — and any signup after a
        # prior failed/reset attempt (e.g. the pre-PLA-acceptance failures) —
        # returns it empty. Mirror the Google path (_pick_google_display_name)
        # and fall back to the email local-part so the User row's `name` never
        # lands empty. An empty name doesn't just show the raw email in the
        # sidebar/avatar — it also makes the Soul→VPS owner-name sync
        # (soul.py: owner_name=current_user.name) push None, which is what
        # surfaces as the agent calling its owner "Agent"/"Owner".
        _apple_name = (body.full_name or "").strip()
        if not _apple_name:
            _local = (email or "").split("@", 1)[0].strip()
            _apple_name = _local[:64] if _local else "User"
        user = await create_user(
            db, email=email, password=random_password, name=_apple_name,
        )
        is_new = True
        if email_verified:
            user.email_verified_at = datetime.utcnow()
            await db.commit()
            await db.refresh(user)
        # New signup: prime managed config, then hand free-tier activation +
        # the bridge claim to the background so neither blocks this response.
        # Mirrors the Google flow — the inline claim_or_prewarm awaited the
        # bridge bind for up to 30s, which is what made Sign-in-with-Apple hang
        # before onboarding. The first chat is post-onboarding,
        # _bg_finalize_signup activates BEFORE it claims so the bind still
        # injects TOUP_TOKEN, and onboarding re-activates the free tier as a
        # backstop.
        if settings.prewarm_on_soul_save:
            try:
                from app.api.agent_setup import _get_or_create_config
                config = await _get_or_create_config(str(user.id), db)
                dirty = False
                if config.hosting_mode in (None, "self-hosted"):
                    config.hosting_mode = "managed"
                    dirty = True
                if not config.whatsapp_mode:
                    config.whatsapp_mode = "qr_link"
                    dirty = True
                if dirty:
                    await db.commit()
                _spawn_background(_bg_finalize_signup(str(user.id)))
            except Exception as e:
                logger.warning("[apple-auth] prewarm/claim failed user=%s: %s", str(user.id)[:8], e)

    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account has been disabled.")

    # Capture an Apple refresh token so account deletion can revoke the grant
    # (Guideline 5.1.1(v)). The refresh token is only needed LATER at deletion
    # time — it has zero bearing on this sign-in response — so run the Apple
    # token exchange (an httpx round-trip, up to 10s) off the path instead of
    # awaiting it on every Sign-in-with-Apple.
    if body.authorization_code:
        _spawn_background(_bg_apple_refresh_capture(str(user.id), body.authorization_code))

    token = create_access_token(user.id)
    try:
        await record_login_session(db, user.id, token, request)
    except Exception:
        logger.exception("[apple-auth] record_login_session failed")

    response.set_cookie(
        key=SSO_COOKIE_NAME, value=token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True,
        samesite="none", path="/",
    )
    return AppleAuthResponse(access_token=token, is_new=is_new)


# ── Profile ──────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_user)):
    """Get current user info"""
    return current_user


@router.patch("/profile")
async def update_profile(
    body: UpdateProfileRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user profile (name, timezone).

    Timezone updates are silent: frontend captures the OS-resolved IANA
    name via `Intl` on app boot and PATCHes here. We validate against
    zoneinfo and no-op the write if the value already matches — so the
    boot-time call is cheap and idempotent.
    """
    changed = False
    if body.name is not None and body.name != current_user.name:
        current_user.name = body.name
        changed = True
    if body.timezone is not None and body.timezone != current_user.timezone:
        # Validate IANA name. zoneinfo is stdlib (3.9+); raises
        # ZoneInfoNotFoundError for typos / spoofed values.
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(body.timezone)
        except ZoneInfoNotFoundError:
            raise HTTPException(status_code=400, detail="Invalid timezone")
        current_user.timezone = body.timezone
        changed = True
        # TKT-LAT-004: drop the in-process tz cache so the next agent
        # turn picks up the new value instead of serving stale.
        try:
            from app.agent._user_tz_cache import invalidate_cached_user_tz
            invalidate_cached_user_tz(current_user.id)
        except Exception:
            pass
    if changed:
        current_user.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(current_user)
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "timezone": current_user.timezone,
    }


# ── Timezone-from-coordinates ────────────────────────────────────
#
# Account-page "Share precise location" flow. The browser asks the
# user for navigator.geolocation permission, then POSTs the resolved
# (lat, lng) here. We derive the IANA timezone via the bundled
# timezonefinder shapefile (no external API, no per-request network).
# This is the explicit-consent counterpart to the silent Intl capture
# wired into /auth/profile.

# Module-level instance — TimezoneFinder() loads the shapefile on
# construction, so reusing one instance across requests avoids paying
# that cost (~50-200ms) per call.
_tz_finder = None


def _get_tz_finder():
    global _tz_finder
    if _tz_finder is None:
        from timezonefinder import TimezoneFinder
        _tz_finder = TimezoneFinder()
    return _tz_finder


class TimezoneFromCoordsRequest(BaseModel):
    # Latitude bounds match the WGS84 valid range. Anything outside
    # this is either a typo or hostile — reject early so timezonefinder
    # never sees garbage.
    lat: float
    lng: float


@router.post("/timezone-from-coords")
async def timezone_from_coords(
    body: TimezoneFromCoordsRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Resolve precise IANA timezone from (lat, lng) and persist on
    User.timezone. Called by the account page after the user grants
    browser geolocation permission. Returns the resolved zone so the
    frontend can show "Now using America/Toronto" in the share card.
    """
    if not (-90.0 <= body.lat <= 90.0) or not (-180.0 <= body.lng <= 180.0):
        raise HTTPException(status_code=400, detail="Coordinates out of range")

    tf = _get_tz_finder()
    tz_name = tf.timezone_at(lat=body.lat, lng=body.lng)
    if not tz_name:
        # timezonefinder returns None only for a handful of unmapped
        # polar / disputed-water polygons. Unlikely for real user
        # devices but the frontend has a fallback to keep the silently
        # captured Intl value, so we surface the failure honestly.
        raise HTTPException(
            status_code=404, detail="Could not determine timezone for these coordinates"
        )

    # Defense in depth: zoneinfo confirms the string timezonefinder
    # returned is one Python (and the rest of our scheduling stack)
    # accepts. If a bundled shapefile ever drifts ahead of stdlib
    # tzdata this catches it before we corrupt User.timezone.
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        raise HTTPException(
            status_code=500, detail="Resolved zone is not a known IANA name"
        )

    if current_user.timezone != tz_name:
        current_user.timezone = tz_name
        current_user.updated_at = datetime.utcnow()
        await db.commit()
        # TKT-LAT-004: drop the in-process tz cache so the next agent
        # turn picks up the new value instead of serving stale.
        try:
            from app.agent._user_tz_cache import invalidate_cached_user_tz
            invalidate_cached_user_tz(current_user.id)
        except Exception:
            pass

    return {"timezone": tz_name}


# ── Change Password ──────────────────────────────────────────────

@router.post("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    response: Response,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change own password. Requires current password. Returns new token."""
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    await change_user_password(db, current_user, body.new_password)
    new_token = create_access_token(current_user.id)

    # Revoke every existing session — including this one — then record
    # the freshly-minted token. password_changed_at already invalidates
    # old JWTs via iat comparison; this also clears the
    # account-page sessions list so the user sees a clean "just one
    # active device" state after changing their password.
    from app.services.session_tracker import revoke_all_user_sessions_except
    await revoke_all_user_sessions_except(db, current_user.id, except_jti=None)
    await record_login_session(db, current_user.id, new_token, request=None)

    response.set_cookie(
        key=SSO_COOKIE_NAME, value=new_token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True,
        samesite="none", path="/",
    )
    return {"access_token": new_token, "message": "Password changed successfully"}


# ── Token Validation ─────────────────────────────────────────────

class ValidateRequest(BaseModel):
    token: Optional[str] = None

class ValidateResponse(BaseModel):
    valid: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None


@router.post("/validate", response_model=ValidateResponse)
async def validate_token(request: Request, body: Optional[ValidateRequest] = None, db: AsyncSession = Depends(get_db)):
    """Validate a JWT token and return user info."""
    token = None
    if body and body.token:
        token = body.token
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        token = request.cookies.get(SSO_COOKIE_NAME)
    if not token:
        return ValidateResponse(valid=False)

    user_id = decode_access_token(token)
    if not user_id:
        return ValidateResponse(valid=False)

    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        return ValidateResponse(valid=False)

    # Token revocation check
    if getattr(user, 'password_changed_at', None):
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
            token_iat = datetime.utcfromtimestamp(payload.get("iat", 0))
            if token_iat < user.password_changed_at:
                return ValidateResponse(valid=False)
        except JWTError:
            return ValidateResponse(valid=False)

    return ValidateResponse(valid=True, user_id=str(user.id), email=user.email, name=user.name)


# ── SSO Exchange ─────────────────────────────────────────────────

class SSOExchangeRequest(BaseModel):
    token: str

@router.post("/sso", response_model=Token)
async def sso_exchange(body: SSOExchangeRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """Exchange an SSO token for a fresh JWT."""
    user_id = decode_access_token(body.token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired SSO token")
    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    new_token = create_access_token(user.id)
    # New token = new session row, same way as /login does it.
    await record_login_session(db, user.id, new_token, request)
    response.set_cookie(
        key=SSO_COOKIE_NAME, value=new_token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True, samesite="none", path="/",
    )
    return Token(access_token=new_token)


# ── Logout ───────────────────────────────────────────────────────

@router.post("/logout")
async def logout_user(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """Logout — clear SSO cookie + revoke current session."""
    # Revoke the current session row so the JWT can't be reused (e.g.
    # if the user is on a shared device and the cookie is still
    # readable elsewhere). Best-effort — if we can't decode the token
    # we still clear the cookie.
    token = (
        request.cookies.get(SSO_COOKIE_NAME)
        or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    )
    if token:
        try:
            payload = jwt.decode(
                token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
            )
            jti = payload.get("jti")
            if jti:
                result = await db.execute(
                    select(UserSession).where(UserSession.jti == jti)
                )
                row = result.scalar_one_or_none()
                if row and not row.is_revoked:
                    row.is_revoked = True
                    await db.commit()
        except (JWTError, Exception) as e:  # noqa: BLE001
            logger.debug("logout session revoke swallowed: %s", e)
    response.delete_cookie(key=SSO_COOKIE_NAME, domain=SSO_COOKIE_DOMAIN, path="/")
    return {"success": True}


# ── Delete Account ───────────────────────────────────────────────
# Apple App Store Guideline 5.1.1(v): users must be able to initiate
# account deletion from within the app. Runs the SAME cascade as the
# admin DELETE /users/{id} endpoint — destroys the managed container +
# tenant DB + Caddy route via the bridge, deletes the Stripe customer
# (cancels every subscription tied to it), archives the per-user
# OpenAI bundle project, wipes 17+ tables (messages, memories, agent
# configs with API keys, etc.), then DELETEs the user row.
#
# Authentication is two-step (per §1.3 sign-off, 2026-05-01):
#   1. Client calls POST /auth/reauth with its existing access token
#      to obtain a 5-minute single-use sensitive-action token.
#   2. Client calls POST /auth/delete-account passing the token in
#      the X-Sensitive-Action-Token header.
#
# Why fresh-JWT instead of password re-entry: avoids storing or
# transmitting the plaintext password on mobile (SecureStore doesn't
# expose it). The token has a stable replay defense via the
# sensitive_action_redemptions table.
#
# Single source of truth for "delete me everywhere" lives in
# `app.services.user_deletion.delete_user_completely`.

from app.services.sensitive_action_token import (
    SensitiveActionPurpose,
    issue_sensitive_action_token,
    verify_sensitive_action_token,
    DEFAULT_TTL_SECONDS,
)
from app.services.user_deletion import (
    DeletionActor,
    DeletionAbortedError,
    delete_user_completely,
)
from dataclasses import asdict


class ReauthRequest(BaseModel):
    # Optional — defaults to delete_account so the existing mobile flow,
    # which calls /reauth with no body, keeps working unchanged. New
    # destructive endpoints pass the matching purpose explicitly.
    purpose: Optional[str] = "delete_account"


class ReauthResponse(BaseModel):
    sensitive_action_token: str
    expires_in: int
    purpose: str


@router.post("/reauth", response_model=ReauthResponse)
async def reauth(
    body: Optional[ReauthRequest] = None,
    current_user=Depends(get_current_user),
):
    """Issue a 5-minute single-use sensitive-action token, bound to the
    calling user and to one specific destructive purpose.

    Defaults to `delete_account` for backward compatibility with the
    mobile delete-account flow that POSTs an empty body. New endpoints
    pass `{"purpose": "<value>"}` explicitly.
    """
    purpose_str = (body.purpose if body else None) or SensitiveActionPurpose.DELETE_ACCOUNT.value
    try:
        purpose = SensitiveActionPurpose(purpose_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown sensitive-action purpose: {purpose_str!r}",
        )
    token, _exp = issue_sensitive_action_token(
        user_id=str(current_user.id),
        purpose=purpose,
    )
    return ReauthResponse(
        sensitive_action_token=token,
        expires_in=DEFAULT_TTL_SECONDS,
        purpose=purpose.value,
    )


@router.post("/delete-account")
async def delete_account(
    request: Request,
    response: Response,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete the current user's account and ALL associated data.

    Requires a fresh sensitive-action token issued via POST /auth/
    reauth within the last 5 minutes. Synchronous — by the time this
    returns, Stripe customer is canceled, the managed container is
    destroyed, OpenAI project is archived, and the user row is gone.

    Last admin / cannot-delete-self guards do not apply here: this
    is the user's own account. Apple's guideline doesn't let us
    refuse deletion just because the user is an admin.

    Failure modes:
      - 401: missing / invalid / replayed sensitive-action token
      - 502: cascade aborted (Stripe or container teardown failed).
        User's data is intact; client should surface a step-keyed
        retry message and let the user try again.
      - 200 with cookie cleared: deletion succeeded.
    """
    sat = request.headers.get("X-Sensitive-Action-Token", "")
    await verify_sensitive_action_token(
        db,
        sat,
        expected_user_id=str(current_user.id),
        expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    # Persist the redemption row before we touch destructive state, so
    # a token that authorized this run is irreversibly burned even if
    # the cascade aborts. Otherwise a Stripe-fail abort would let the
    # user re-issue + re-redeem the same token.
    await db.commit()

    try:
        receipt = await delete_user_completely(
            db,
            current_user,
            actor=DeletionActor.SELF,
            request_ip=(request.client.host if request.client else None),
            request_user_agent=request.headers.get("user-agent"),
        )
    except DeletionAbortedError as e:
        # Don't clear the cookie — the user is still logged in, their
        # data is intact, and they can retry once the underlying
        # service recovers.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"step": e.step.value, "message": e.detail},
        )

    # Success — clear the SSO cookie so the now-deleted user's session
    # is invalidated immediately on the client.
    response.delete_cookie(
        key=SSO_COOKIE_NAME, domain=SSO_COOKIE_DOMAIN, path="/",
    )
    return {
        "success": True,
        "message": "Account and all associated data deleted.",
        "receipt": asdict(receipt),
    }


# ── Demo ─────────────────────────────────────────────────────────

@router.post("/demo", response_model=Token)
async def demo_login(db: AsyncSession = Depends(get_db)):
    """Create or login as demo user (for testing)"""
    demo_email = "demo@toup.local"
    demo_password = "demo123456"
    user = await get_user_by_email(db, demo_email)
    if not user:
        user = await create_user(db, email=demo_email, password=demo_password, name="Demo User")
    token = create_access_token(user.id)
    return Token(access_token=token)


# ── Direct-to-agent session token ────────────────────────────────

@router.post("/agent-session-token")
async def agent_session_token(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Issue a short-lived (5min) token the browser uses to open a
    WebSocket directly to the agent at `agent-<prefix>.agents.toup.ai`,
    bypassing the Railway platform proxy entirely.

    Why: a platform redeploy (Railway swap) would otherwise drop every
    chat WebSocket for ~30s. Direct-to-agent connections live between
    the browser and Contabo VPS, completely independent of Railway,
    so chat keeps streaming through any number of platform deploys.

    Token format: HS256 JWT, secret = the user's `agent_api_key`
    (already known to both platform and agent — the same key the
    platform uses for X-Agent-Key proxy auth, so no new bootstrap).
    Claims:
        sub: user_id
        iss: "toup-platform"
        aud: "toup-agent-session"
        exp: now + 300s

    Refresh: client calls again before expiry. The 5-min window keeps
    the token short enough that leaks don't matter much, while long
    enough that a refresh during a long deploy still succeeds.

    Returns 200 with `{token, agent_url, expires_in}`. Returns 404 if
    the user has no provisioned agent — caller falls back to the
    legacy platform-proxy chat path."""
    from sqlalchemy import select
    from app.db.models import AgentConfig

    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == current_user.id)
    )
    cfg = result.scalar_one_or_none()
    if not cfg or not cfg.agent_url or not cfg.agent_api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No provisioned agent for this user",
        )
    if cfg.deploy_status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent is not active (status={cfg.deploy_status})",
        )

    expires_in = 300  # 5 min
    now = int(datetime.utcnow().timestamp())
    payload = {
        "sub": str(current_user.id),
        "iss": "toup-platform",
        "aud": "toup-agent-session",
        "iat": now,
        "exp": now + expires_in,
    }
    # python-jose, already in requirements (`from jose import jwt` at top).
    token = jwt.encode(payload, cfg.agent_api_key, algorithm="HS256")
    return {
        "token": token,
        "agent_url": cfg.agent_url,
        "expires_in": expires_in,
    }
