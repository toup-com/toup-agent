"""
T1d — OAuth provider-app registry.

Connectors are keyed by `manifest.id` (gmail, gcal, gdrive, github, ...);
provider APPS are keyed by `manifest.oauth.provider_app` (google, github,
stub_provider_app). One provider app can back many connectors — the
canonical example is Google: Gmail + Calendar + Drive all share the
single google_oauth_client_id (architecture §6.1).

This module is the source of truth for OAuth-app config (client_id,
client_secret, authorize/token/revoke URLs). Real provider-app configs
land in T3a (Google) and T4a (GitHub). T1d ships only the stub.

Why an explicit module rather than a settings field per provider:
  - One place to look for "what providers does the platform know about?"
  - Bootstrap-on-demand: real providers are only registered when their
    env vars are set (a deploy without GOOGLE_OAUTH_CLIENT_ID simply
    does not register google).
  - Tests can clear-and-re-register without polluting global config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderAppConfig:
    """OAuth client config for one provider app.

    `client_secret` is sensitive — never logged, never surfaced through
    any HTTP endpoint.
    """

    name: str               # e.g. "google", "github", "stub_provider_app"
    client_id: str
    client_secret: str
    authorize_url: str      # base URL — query params appended at /connect
    token_url: str          # POST endpoint for code-exchange and refresh
    use_pkce: bool = True


# Module-level registry. Populated at platform lifespan; cleared via
# `reset_for_tests` when needed.
_apps: dict[str, ProviderAppConfig] = {}

# Per-name "skeleton" template — endpoint URLs, PKCE policy. Populated
# at boot by `bootstrap_provider_apps`. We use these to assemble a
# ProviderAppConfig when DB-backed credentials override env-var ones,
# without re-typing the URL set per provider.
_TEMPLATES: dict[str, dict] = {
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "use_pkce": True,
    },
    "github": {
        # `/select_account` (not `/authorize`) so multi-account users see
        # GitHub's account chooser before consenting — matches the
        # Base44 / Vercel / "Sign in with GitHub" UX. GitHub's plain
        # `/authorize` silently authorizes the currently-signed-in
        # account, which is confusing for anyone who keeps a personal
        # + work GitHub logged in at the same time. Same query params,
        # same token-exchange URL — only the front-door page differs.
        "authorize_url": "https://github.com/login/oauth/select_account",
        "token_url": "https://github.com/login/oauth/access_token",
        "use_pkce": False,
    },
    "microsoft": {
        # Microsoft Identity Platform v2. `common` tenant accepts both
        # personal Microsoft accounts (Outlook.com, Live, Xbox) and
        # work/school Azure AD accounts. Endpoint is templated with
        # the tenant id at registration time (see
        # `register_microsoft_provider_app` below). Microsoft always
        # shows an account chooser when `prompt=select_account` is
        # set or when the user has multiple signed-in accounts — but
        # by default it can short-circuit to the active account. We
        # add `prompt=select_account` at the connect call site for
        # both first-time + "Switch account" flows so the experience
        # matches the GitHub `/select_account` choice (see above).
        "authorize_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "use_pkce": True,
    },
    "linkedin": {
        # LinkedIn OAuth 2.0. No PKCE support today. Authorize URL
        # supports `client_id`, `redirect_uri`, `response_type=code`,
        # `state`, and `scope` (space-separated; we URL-encode at
        # send time). Token endpoint returns access_token +
        # refresh_token + expires_in.
        "authorize_url": "https://www.linkedin.com/oauth/v2/authorization",
        "token_url": "https://www.linkedin.com/oauth/v2/accessToken",
        "use_pkce": False,
    },
    "stub_provider_app": {
        "authorize_url": "/api/oauth/_stub/authorize",
        "token_url": "/api/oauth/_stub/token",
        "use_pkce": True,
    },
}


def register_provider_app(config: ProviderAppConfig) -> None:
    """Idempotent — re-registering with the same name overwrites."""
    _apps[config.name] = config
    logger.info(
        "[provider_apps] Registered %r (client_id=%s..., authorize=%s)",
        config.name, config.client_id[:8], config.authorize_url,
    )


def _from_db_credentials(name: str) -> Optional[ProviderAppConfig]:
    """Best-effort load of provider-app credentials from the platform
    DB. Returns None on any failure (no row, decryption error, DB down)
    so the caller falls through to env-var registration.

    Cheap to call: a single PK lookup. We do NOT cache here — admin UI
    saves should take effect on the very next /connect request without
    a process restart, and the ProviderAppCredential table is tiny.
    For very high-traffic deploys we can add a TTL cache later.
    """
    template = _TEMPLATES.get(name)
    if template is None:
        # Unknown provider — no template to assemble against.
        return None
    try:
        # Local imports — provider_apps.py loads at module import in
        # config / lifespan paths where the DB layer isn't ready yet.
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import ProviderAppCredential
        from app.services.credential_crypto import decrypt_str
        # Sync wrapper — get_provider_app is sync. Use the running
        # event loop's run_until_complete equivalent. Simpler: use
        # an async helper and have callers `await` instead. Falling
        # back to sync DB query via a tiny asyncio.run since callers
        # are inside FastAPI's event loop already → can't asyncio.run.
        # We instead expose an async sibling and have the OAuth
        # endpoint use that path. Sync callers (lifespan boot) get
        # env-var-only behaviour; that's correct.
        return None  # sync path falls through; async sibling handles DB
    except Exception as e:
        logger.warning("[provider_apps] DB credential load failed: %s", e)
        return None


async def get_provider_app_async(name: str) -> Optional[ProviderAppConfig]:
    """Async resolver that consults the DB first (admin-UI overrides),
    then falls back to whatever was registered at boot from env vars.

    The OAuth /connect endpoint awaits this so admin saves take effect
    immediately. Lifespan code that just wants the env-var snapshot
    keeps using `get_provider_app(name)`.
    """
    template = _TEMPLATES.get(name)
    if template is None:
        return _apps.get(name)

    try:
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import ProviderAppCredential
        from app.services.credential_crypto import decrypt_str

        async with async_session_maker() as db:
            row = (await db.execute(
                select(ProviderAppCredential).where(
                    ProviderAppCredential.name == name,
                ),
            )).scalar_one_or_none()
        if row is not None:
            return ProviderAppConfig(
                name=name,
                client_id=row.client_id,
                client_secret=decrypt_str(row.client_secret_enc),
                authorize_url=template["authorize_url"],
                token_url=template["token_url"],
                use_pkce=template["use_pkce"],
            )
    except Exception as e:
        # Never block /connect on a transient DB hiccup — fall through
        # to the env-var registration. Logged so operators can spot
        # decryption issues (wrong PLATFORM_ENCRYPTION_KEY rotation).
        logger.warning(
            "[provider_apps] DB credential load failed for %r: %s",
            name, e,
        )

    return _apps.get(name)


def get_provider_app(name: str) -> Optional[ProviderAppConfig]:
    """Sync resolver — env-var snapshot only. Use `get_provider_app_async`
    from request handlers so admin-UI overrides apply immediately."""
    return _apps.get(name)


def list_registered() -> list[str]:
    """Names of providers registered from ENV VARS at boot. Sync, for
    callers in module-init paths that run before the DB is ready.

    Not the whole picture — admin-UI credentials live in the DB and
    are invisible here. Anything user-facing wants
    `list_configured_async()`."""
    return sorted(_apps.keys())


async def list_configured_async() -> Optional[set[str]]:
    """Every provider app that has usable OAuth credentials right now:
    DB rows (admin-UI saves) unioned with the env-var registrations.

    One query, not one per provider — the catalogue calls this once per
    request and `provider_app_credentials` is a handful of rows.

    A connector whose provider app is NOT in this set cannot complete a
    connect: `/api/oauth/connect` answers 503 `credentials_missing`. The
    catalogue uses that to stop advertising a Connect button that is
    guaranteed to fail, which is what LinkedIn and Outlook did to every
    user for months.

    Returns **None** when the answer is UNKNOWN — i.e. the DB read
    failed — and the caller must then gate nothing.

    That distinction is the whole safety of this function. Returning the
    env-var set on failure looks like a reasonable fallback and is a trap
    in production: platform-api runs with no `GOOGLE_OAUTH_*` /
    `GITHUB_OAUTH_*` env vars at all (credentials live in the DB), so
    `_apps` is EMPTY there. A single transient DB hiccup would therefore
    answer "nothing is configured" with total confidence and grey out
    every connector on the page, Google and GitHub included.

    None means "don't know, so don't act". The worst case is then a
    Connect button that 503s — exactly the behaviour we had before this
    function existed, which is the correct thing to degrade back to.
    """
    try:
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import ProviderAppCredential

        async with async_session_maker() as db:
            rows = (await db.execute(
                select(ProviderAppCredential.name),
            )).scalars().all()
    except Exception as e:
        logger.warning(
            "[provider_apps] configured-list DB read failed (%s) — "
            "catalogue will not gate any connector this request", e,
        )
        return None
    return set(_apps.keys()) | {r for r in rows if r}


def reset_for_tests() -> None:
    """Drop all registered provider apps. Tests only — production
    boots once and never resets."""
    _apps.clear()


# ─── Stub provider-app (T1d test harness) ────────────────────────────


def register_stub_provider_app() -> None:
    """Register the stub provider app whose authorize+token endpoints
    are local routes on the platform itself (`/api/oauth/_stub/*`).

    Always safe to call — the routes are gated by run_mode / test
    contexts and the experimental-status filter on the stub manifest
    keeps it out of production load.
    """
    register_provider_app(ProviderAppConfig(
        name="stub_provider_app",
        client_id="stub_client",
        client_secret="stub_secret_not_used_anywhere_real",
        # Resolved relative to oauth_callback_url's host (set at /connect
        # call time) — the stub authorize endpoint lives on the same
        # FastAPI app as the OAuth API, so the browser stays in-process.
        authorize_url="/api/oauth/_stub/authorize",
        token_url="/api/oauth/_stub/token",
        use_pkce=True,
    ))


def register_google_provider_app() -> None:
    """T3a — Register the shared Google OAuth client.

    One client backs Gmail, Calendar, and Drive (architecture §6.1):
    distinct manifests, distinct scope sets, ONE OAuth credential pair.
    Authorize URL is Google's standard endpoint; PKCE always on per
    Google's modern security guidance for confidential clients with
    public-facing platforms.

    Required env vars:
      - GOOGLE_OAUTH_CLIENT_ID
      - GOOGLE_OAUTH_CLIENT_SECRET

    Without both set, this function silently no-ops — the registry's
    boot-time lint will skip any connector that names this provider
    so the boot doesn't fail loudly on a missing-env case (production
    may run without Google connectors when the verification clock is
    still pending).
    """
    cid = getattr(settings, "google_oauth_client_id", "") or ""
    csec = getattr(settings, "google_oauth_client_secret", "") or ""
    if not cid or not csec:
        logger.info(
            "[provider_apps] google not registered — "
            "GOOGLE_OAUTH_CLIENT_ID/SECRET not set",
        )
        return
    register_provider_app(ProviderAppConfig(
        name="google",
        client_id=cid,
        client_secret=csec,
        # `prompt=consent` and `access_type=offline` get appended at the
        # connect step (oauth.py builds the query string) so we always
        # get a refresh_token and re-show the consent screen on
        # re-auth — Google otherwise short-circuits and skips both.
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        use_pkce=True,
    ))


def register_github_provider_app() -> None:
    """T4a — Register the GitHub OAuth client. Same shape as Google
    minus PKCE (GitHub doesn't support PKCE on the standard OAuth
    app type; would need a GitHub App for that. PKCE off here is
    intentional — the architecture §6.1 PKCE-default does not apply
    when the provider doesn't support it).

    Required env vars:
      - GITHUB_OAUTH_CLIENT_ID
      - GITHUB_OAUTH_CLIENT_SECRET
    """
    cid = getattr(settings, "github_oauth_client_id", "") or ""
    csec = getattr(settings, "github_oauth_client_secret", "") or ""
    if not cid or not csec:
        logger.info(
            "[provider_apps] github not registered — "
            "GITHUB_OAUTH_CLIENT_ID/SECRET not set",
        )
        return
    register_provider_app(ProviderAppConfig(
        name="github",
        client_id=cid,
        client_secret=csec,
        # See the rationale in `_TEMPLATES["github"]` above — we ALWAYS
        # route through GitHub's `/select_account` so multi-account
        # users see the chooser instead of being silently authorized
        # as whichever GitHub user happens to be logged in.
        authorize_url="https://github.com/login/oauth/select_account",
        token_url="https://github.com/login/oauth/access_token",
        use_pkce=False,
    ))


def register_microsoft_provider_app() -> None:
    """Register the Microsoft Identity Platform v2 OAuth client.

    Backs the Outlook connector. Tenant id is templated into the
    authorize + token URLs at registration time so an operator can
    lock OAuth to a specific Azure AD organisation by setting
    `MICROSOFT_OAUTH_TENANT` to that org's GUID (default `common`
    accepts both personal + work accounts).

    Required env vars:
      - MICROSOFT_OAUTH_CLIENT_ID
      - MICROSOFT_OAUTH_CLIENT_SECRET
    Optional:
      - MICROSOFT_OAUTH_TENANT (default: `common`)
    """
    cid = getattr(settings, "microsoft_oauth_client_id", "") or ""
    csec = getattr(settings, "microsoft_oauth_client_secret", "") or ""
    tenant = getattr(settings, "microsoft_oauth_tenant", "") or "common"
    if not cid or not csec:
        logger.info(
            "[provider_apps] microsoft not registered — "
            "MICROSOFT_OAUTH_CLIENT_ID/SECRET not set",
        )
        return
    register_provider_app(ProviderAppConfig(
        name="microsoft",
        client_id=cid,
        client_secret=csec,
        authorize_url=f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
        token_url=f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        use_pkce=True,
    ))


def register_linkedin_provider_app() -> None:
    """Register the LinkedIn OAuth 2.0 client.

    Backs the LinkedIn connector (profile read + post share). LinkedIn
    doesn't support PKCE yet, so this provider opts out — same
    treatment as the GitHub OAuth app.

    Required env vars:
      - LINKEDIN_OAUTH_CLIENT_ID
      - LINKEDIN_OAUTH_CLIENT_SECRET
    """
    cid = getattr(settings, "linkedin_oauth_client_id", "") or ""
    csec = getattr(settings, "linkedin_oauth_client_secret", "") or ""
    if not cid or not csec:
        logger.info(
            "[provider_apps] linkedin not registered — "
            "LINKEDIN_OAUTH_CLIENT_ID/SECRET not set",
        )
        return
    register_provider_app(ProviderAppConfig(
        name="linkedin",
        client_id=cid,
        client_secret=csec,
        authorize_url="https://www.linkedin.com/oauth/v2/authorization",
        token_url="https://www.linkedin.com/oauth/v2/accessToken",
        use_pkce=False,
    ))


def bootstrap_provider_apps() -> None:
    """Called from platform_main lifespan. Always registers stub;
    registers real providers only when their env vars are present."""
    register_stub_provider_app()
    register_google_provider_app()
    register_github_provider_app()
    register_microsoft_provider_app()
    register_linkedin_provider_app()
