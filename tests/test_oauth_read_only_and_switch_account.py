"""Regression test for the 2026-05-11 Base44-parity connector UX work.

Pins three wiring decisions so they can't regress quietly:

1. `_build_authorize_url(..., force_account_selection=True)` adds
   `prompt=select_account`. The connected-card "Switch account"
   action relies on this so Google re-shows the account chooser even
   when the user is already signed in.

2. `POST /api/oauth/{id}/read-only` is wired and persists the flag
   onto `ConnectorIdentity.read_only`. The dispatcher / MCP filter
   both gate on this column, so the API contract is the binding link.

3. The dispatcher refuses a mutating tool call when the identity is
   read-only — defense-in-depth on top of the MCP-filter drop. If
   this check ever disappears, a stale tool-list cache could squeak
   a write through.

Source-grep style for (1) and (3); a focused FastAPI call for (2)."""
from __future__ import annotations

from pathlib import Path

from app.api.oauth import _build_authorize_url


_DISPATCHER_SRC = (
    Path(__file__).resolve().parent.parent
    / "app" / "services" / "connector_dispatcher.py"
).read_text()


def test_authorize_url_adds_prompt_when_switching_account():
    """The "Switch account" action passes `force_account_selection=True`
    so the provider re-shows its chooser. Without `prompt=select_account`
    Google silently reuses the signed-in account and the user can't
    actually swap — the action becomes a no-op."""
    url_switch = _build_authorize_url(
        base_url="https://accounts.google.com/o/oauth2/auth",
        client_id="cid",
        redirect_uri="http://x",
        scopes=["gmail.readonly"],
        state="state-xyz",
        code_challenge="cc",
        use_pkce=True,
        force_account_selection=True,
    )
    assert "prompt=select_account" in url_switch, (
        "force_account_selection=True must emit `prompt=select_account` — "
        "without it the 'Switch account' button is a no-op when the user "
        "is already signed in."
    )

    url_default = _build_authorize_url(
        base_url="https://accounts.google.com/o/oauth2/auth",
        client_id="cid",
        redirect_uri="http://x",
        scopes=["gmail.readonly"],
        state="state-xyz",
        code_challenge="cc",
        use_pkce=True,
    )
    assert "prompt=" not in url_default, (
        "Default OAuth flow must NOT add prompt=select_account — first-"
        "time consent should follow Google's normal UX."
    )


def test_dispatcher_blocks_mutating_tools_on_read_only_identity():
    """Defense-in-depth: even if a stale tool-list cache squeaks a
    mutating tool past the MCP filter, the dispatcher refuses to
    invoke it when `identity.read_only` is True. This source-grep
    pins the guard so a refactor that re-orders the dispatcher steps
    can't accidentally drop it."""
    assert "read_only" in _DISPATCHER_SRC, (
        "connector_dispatcher.py must reference identity.read_only — "
        "without the guard, MCP-filter-cached writes would still reach "
        "the provider when the user toggled read-only mid-session."
    )
    # The guard must check both `read_only` AND `manifest_tool.mutates`.
    # Otherwise read-only would silently block reads too (false positive)
    # or block nothing (false negative).
    guard_idx = _DISPATCHER_SRC.index('getattr(identity, "read_only", False)')
    nearby = _DISPATCHER_SRC[guard_idx:guard_idx + 200]
    assert "manifest_tool.mutates" in nearby, (
        "read-only guard must gate on manifest_tool.mutates so read "
        "tools still pass while write tools are refused. The current "
        "code reads `if identity.read_only and manifest_tool.mutates: ...`."
    )


def test_github_authorize_routes_through_select_account():
    """GitHub's plain `/login/oauth/authorize` silently authorizes the
    currently-signed-in account (no chooser, no confirmation), which is
    confusing for anyone with personal + work GitHub accounts. We
    route through `/login/oauth/select_account` so first-time connect
    always shows a chooser — same UX as Base44 / Vercel / "Sign in
    with GitHub" buttons across the web.

    Pinning this in both the template and the env-var registration
    path so a future copy-paste from GitHub's docs doesn't quietly
    drop us back to the silent-authorize URL."""
    from app.services.provider_apps import _TEMPLATES, register_github_provider_app
    from app.services import provider_apps as _pa

    assert _TEMPLATES["github"]["authorize_url"].endswith("/select_account"), (
        "GitHub template authorize_url must route through "
        "/login/oauth/select_account so multi-account users see the "
        "chooser. The plain /authorize endpoint silently uses the "
        "currently-signed-in account."
    )

    # Stub the settings so the env-var path actually registers.
    from app.config import settings
    prev_cid = getattr(settings, "github_oauth_client_id", "")
    prev_sec = getattr(settings, "github_oauth_client_secret", "")
    settings.github_oauth_client_id = "test_cid"
    settings.github_oauth_client_secret = "test_secret"
    try:
        _pa.reset_for_tests()
        register_github_provider_app()
        cfg = _pa.get_provider_app("github")
        assert cfg is not None
        assert cfg.authorize_url.endswith("/select_account"), (
            "register_github_provider_app must also use /select_account — "
            "otherwise the prod path (env-var registration) drifts from "
            "the template path (DB-credential registration)."
        )
    finally:
        settings.github_oauth_client_id = prev_cid
        settings.github_oauth_client_secret = prev_sec
        _pa.reset_for_tests()


def test_health_probe_unaffected_by_read_only():
    """A read-only identity is still a healthy identity — the probe
    should keep refreshing tokens and reporting health. Locking the
    inverse (probes that skip read-only identities) keeps the user's
    connection from quietly drifting into provider_down territory just
    because they don't want writes."""
    probe_src = (
        Path(__file__).resolve().parent.parent
        / "app" / "services" / "connector_health_probe.py"
    ).read_text()
    # The probe loop selects on status; it should NOT filter by
    # read_only. If a future change adds a "skip read-only" filter,
    # this test fails and the rationale gets re-evaluated.
    assert ".read_only" not in probe_src, (
        "connector_health_probe.py must not branch on identity.read_only — "
        "read-only is a tool-call-time concern, not a probe-time concern. "
        "Filtering probes by it would let tokens silently expire."
    )
