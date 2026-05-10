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


def register_provider_app(config: ProviderAppConfig) -> None:
    """Idempotent — re-registering with the same name overwrites."""
    _apps[config.name] = config
    logger.info(
        "[provider_apps] Registered %r (client_id=%s..., authorize=%s)",
        config.name, config.client_id[:8], config.authorize_url,
    )


def get_provider_app(name: str) -> Optional[ProviderAppConfig]:
    return _apps.get(name)


def list_registered() -> list[str]:
    return sorted(_apps.keys())


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
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        use_pkce=False,
    ))


def bootstrap_provider_apps() -> None:
    """Called from platform_main lifespan. Always registers stub;
    registers real providers only when their env vars are present."""
    register_stub_provider_app()
    register_google_provider_app()
    register_github_provider_app()
