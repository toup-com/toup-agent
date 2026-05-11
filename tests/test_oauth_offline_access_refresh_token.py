"""Regression tests for the 2026-05-12 "Gmail disconnects every hour" fix.

Two contracts pinned here. Both have to hold for refresh tokens to
work — drop either and Google access tokens expire silently at the
1-hour mark with no recovery path, the dispatcher flips identity to
`reauth_required`, the MCP filter drops the tool, and the agent
falls back to the `browser` tool (which fails because Chromium isn't
installed in the runtime container).

  1. The Google authorize URL MUST include `access_type=offline`.
     Without it Google's token response omits the refresh_token
     entirely. This was the silent-since-day-one bug — the original
     `provider_apps.py:272` comment promised these would be appended
     "at the connect step" but they never were.

  2. The Google authorize URL MUST include `prompt=consent` on a
     normal connect (and `prompt=select_account` on a switch-account
     connect, which Google honours WITH access_type=offline so
     refresh_token still comes back). Without consent on the first
     re-auth, Google can silently skip the consent screen AND skip
     re-issuing a refresh_token even when access_type=offline is set.

Microsoft uses the `offline_access` scope (already pinned in the
outlook manifest), not these query params; LinkedIn returns a
refresh_token without any extra params. GitHub doesn't expire.
"""
from __future__ import annotations

import urllib.parse

from app.api.oauth import _build_authorize_url


def _parse_query(url: str) -> dict[str, str]:
    """Return the URL's query as a flat dict. Multi-value keys would
    collapse but every OAuth param here is single-valued."""
    return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))


def test_google_authorize_url_includes_access_type_offline():
    """`access_type=offline` is REQUIRED for Google to issue a
    refresh_token. Drop it and every Gmail / Calendar / Drive identity
    flips to `reauth_required` exactly 1 hour after the user connects."""
    url = _build_authorize_url(
        base_url="https://accounts.google.com/o/oauth2/v2/auth",
        client_id="testclient",
        redirect_uri="https://app.example/callback",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        state="state-token",
        code_challenge="challenge",
        use_pkce=True,
        provider_name="google",
    )
    params = _parse_query(url)
    assert params.get("access_type") == "offline", (
        "Google authorize URL must include access_type=offline. "
        "Without it Google's token response omits refresh_token and "
        "every connected Gmail/Calendar/Drive identity flips to "
        "reauth_required at the 1-hour mark."
    )


def test_google_authorize_url_includes_prompt_consent_on_normal_connect():
    """`prompt=consent` re-shows Google's consent screen and ensures
    the refresh_token is re-issued even on re-auth (Google can
    otherwise silently skip both). Pair with access_type=offline."""
    url = _build_authorize_url(
        base_url="https://accounts.google.com/o/oauth2/v2/auth",
        client_id="testclient",
        redirect_uri="https://app.example/callback",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        state="state-token",
        code_challenge="challenge",
        use_pkce=True,
        provider_name="google",
        force_account_selection=False,
    )
    params = _parse_query(url)
    assert params.get("prompt") == "consent", (
        "Google normal-connect authorize URL must include "
        "prompt=consent so a re-auth re-issues the refresh_token. "
        "Without it the second connect can silently skip both consent "
        "and the refresh_token, replaying the hourly-disconnect bug."
    )


def test_google_switch_account_uses_select_account_not_consent():
    """The connected-card 'Switch account' path passes
    force_account_selection=True. That should set prompt=select_account
    (not consent) — and access_type=offline must STILL be present so
    the refresh_token still comes back on the new account's first
    connect."""
    url = _build_authorize_url(
        base_url="https://accounts.google.com/o/oauth2/v2/auth",
        client_id="testclient",
        redirect_uri="https://app.example/callback",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        state="state-token",
        code_challenge="challenge",
        use_pkce=True,
        provider_name="google",
        force_account_selection=True,
    )
    params = _parse_query(url)
    assert params.get("prompt") == "select_account", (
        "switch-account path must use prompt=select_account so the "
        "user sees Google's account chooser"
    )
    assert params.get("access_type") == "offline", (
        "switch-account must STILL include access_type=offline — "
        "without it the new account's first-time connect won't return "
        "a refresh_token and will hit the same 1-hour disconnect."
    )


def test_non_google_providers_dont_get_google_specific_params():
    """`access_type=offline` and `prompt=consent` are Google-specific.
    Other providers should not receive them — they either don't
    recognize them (GitHub) or have their own equivalent (Microsoft's
    `offline_access` scope, declared in the outlook manifest)."""
    for provider in ("github", "microsoft", "linkedin"):
        url = _build_authorize_url(
            base_url="https://example/authorize",
            client_id="testclient",
            redirect_uri="https://app.example/callback",
            scopes=["read"],
            state="state",
            code_challenge="challenge",
            use_pkce=True,
            provider_name=provider,
        )
        params = _parse_query(url)
        assert "access_type" not in params, (
            f"{provider!r}: access_type is Google-specific and should "
            f"not be added for other providers."
        )
        # GitHub doesn't expect prompt=consent. Microsoft accepts it
        # but doesn't need it (offline_access scope is the right
        # mechanism). LinkedIn ignores it.
        assert params.get("prompt") != "consent", (
            f"{provider!r}: prompt=consent should be Google-only — "
            f"other providers either ignore it or have a better path."
        )


def test_outlook_manifest_declares_offline_access_scope():
    """Microsoft's equivalent of Google's `access_type=offline` is the
    `offline_access` scope. Pin it in the manifest — drop it and
    Outlook refresh_tokens disappear with the same symptom (hourly
    disconnect) but a different fix point."""
    import yaml
    from pathlib import Path
    manifest = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "app/connectors/outlook/manifest.yaml").read_text()
    )
    scopes = manifest["oauth"]["scopes"]
    assert "offline_access" in scopes, (
        "outlook manifest must declare the `offline_access` scope. "
        "Without it Microsoft's token response omits refresh_token "
        "and every Outlook identity hits the 1-hour disconnect bug."
    )
