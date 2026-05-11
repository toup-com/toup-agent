"""Regression test for the 2026-05-11 connector roster expansion.

Three new connectors ship together: docs (Google Docs), outlook
(Microsoft Graph mail), and linkedin (profile read + share post).
Each pairs with a provider_app:

  docs     → google      (existing OAuth client)
  outlook  → microsoft   (new OAuth client, env vars required)
  linkedin → linkedin    (new OAuth client, env vars required)

The connector registry validates each manifest at load time. A
manifest that references a `provider_app` not listed in
`KNOWN_PROVIDER_APPS` is rejected with an alarm — that's how
outlook + linkedin got rejected on the first load. Pin all four
constraints here so a future cleanup of `KNOWN_PROVIDER_APPS` can't
silently drop a connector:

  1. docs / outlook / linkedin all register on a clean boot.
  2. KNOWN_PROVIDER_APPS contains microsoft + linkedin.
  3. _TEMPLATES has microsoft + linkedin entries.
  4. Outlook's manifest declares `offline_access` so we actually
     get a refresh_token — Microsoft mints one ONLY when that
     scope is requested.
"""
from __future__ import annotations

from app.services.connector_registry import (
    KNOWN_PROVIDER_APPS,
    get_registry,
    reset_registry_for_tests,
)
from app.services.provider_apps import _TEMPLATES


def test_known_provider_apps_includes_microsoft_and_linkedin():
    """Without these, the registry rejects outlook + linkedin
    manifests at boot — connectors silently disappear from the
    catalog. This was the failure mode caught during initial wiring."""
    assert "microsoft" in KNOWN_PROVIDER_APPS, (
        "KNOWN_PROVIDER_APPS must include 'microsoft' — Outlook (and "
        "future MS 365 connectors) reference provider_app='microsoft'."
    )
    assert "linkedin" in KNOWN_PROVIDER_APPS, (
        "KNOWN_PROVIDER_APPS must include 'linkedin'."
    )


def test_provider_app_templates_include_microsoft_and_linkedin():
    """`get_provider_app_async` uses these templates to assemble a
    ProviderAppConfig when credentials come from the admin DB. Drop
    the template and the admin UI's "Save Microsoft credentials" path
    silently breaks (no error, just no app registered)."""
    assert "microsoft" in _TEMPLATES
    assert "linkedin" in _TEMPLATES
    # Microsoft authorize URL must point at the v2 oauth endpoint —
    # /common or a templated tenant id. Easy to regress to v1 if a
    # future copy/paste from Microsoft docs lands on the older URL.
    assert "oauth2/v2.0/authorize" in _TEMPLATES["microsoft"]["authorize_url"]
    assert _TEMPLATES["microsoft"]["use_pkce"] is True
    # LinkedIn doesn't support PKCE today — if we ever set use_pkce
    # True they reject the authorize request with `invalid_request`.
    assert _TEMPLATES["linkedin"]["use_pkce"] is False


def test_docs_outlook_linkedin_register_on_clean_boot():
    """Smoke load the registry and confirm all three new connectors
    end up in the loaded set. A manifest YAML typo or a provider
    class with the wrong manifest_id ClassVar would surface here as
    a missing entry."""
    reset_registry_for_tests()
    reg = get_registry()
    reg.load_all()
    ids = {entry.manifest.id for entry in reg.list_all()}
    for required in ("docs", "outlook", "linkedin"):
        assert required in ids, (
            f"{required!r} not in registry. Check the alarm list — "
            f"likely manifest validation rejected it (run "
            f"reg.alarms() to see why)."
        )


def test_outlook_manifest_includes_offline_access_scope():
    """Microsoft Identity Platform mints a refresh_token ONLY when
    `offline_access` is in the scope set. Without it, the OAuth flow
    succeeds, the agent works for ~1h, then refresh fails with
    `invalid_grant` and the user has to re-OAuth every hour. The
    initial Outlook draft missed this exact scope — pin it."""
    reset_registry_for_tests()
    reg = get_registry()
    reg.load_all()
    entry = reg.get("outlook")
    assert entry is not None, "outlook connector failed to register"
    scopes = entry.manifest.oauth.scopes
    assert "offline_access" in scopes, (
        "outlook manifest must request `offline_access` — without it "
        "Microsoft doesn't issue a refresh_token and the user has to "
        "re-OAuth every hour."
    )


def test_linkedin_manifest_includes_openid_scope():
    """LinkedIn's `/v2/userinfo` endpoint requires the `openid` scope
    AND returns the user's `sub` field which the share-post tool
    needs to construct `urn:li:person:<sub>` as the post author.
    Without `openid` in the scope set, share_post can't find the
    author and 401s every call. Pin the scope explicitly."""
    reset_registry_for_tests()
    reg = get_registry()
    reg.load_all()
    entry = reg.get("linkedin")
    assert entry is not None, "linkedin connector failed to register"
    scopes = entry.manifest.oauth.scopes
    assert "openid" in scopes, (
        "linkedin manifest must request `openid` — /v2/userinfo "
        "needs it, and share_post depends on the userinfo `sub` to "
        "construct the `urn:li:person:<sub>` author."
    )
    assert "w_member_social" in scopes, (
        "linkedin manifest must request `w_member_social` — share_post "
        "won't work without it (403 from the ugcPosts endpoint)."
    )
