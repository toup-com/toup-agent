"""
Regression tests for app.services.bundle_client.

Pinned because regressions here have caused live customer outages
(matin incident, 2026-04-27 — bundle subscribers' agents bypassed the
proxy entirely when llm_mode wasn't honored, and Cloudflare WAF blocked
the SDK's default user-agent when they finally hit the proxy).

These tests assert on the *constructor surface*: base_url, api_key
prefix, and the user-agent header carried by the underlying httpx
client. They never make a network call.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _bundle_settings(llm_mode: str = "bundle", toup_token: str = "toup_ct_test_xyz",
                     platform_api_url: str = "https://toup.ai/api"):
    """Patch settings to simulate a bundle-mode tenant."""
    from app.config import settings
    return patch.multiple(
        settings,
        llm_mode=llm_mode,
        toup_token=toup_token,
        platform_api_url=platform_api_url,
    )


def test_anthropic_client_in_bundle_mode_routes_through_proxy():
    """The single most important contract: bundle mode → proxy base_url
    + TOUP_TOKEN as api_key. Regressing this leaks the platform master
    Anthropic key directly to api.anthropic.com from every tenant."""
    from app.services.bundle_client import make_anthropic_client

    with _bundle_settings():
        client = make_anthropic_client()

    assert client is not None
    assert str(client.base_url) == "https://toup.ai/api/llm/"
    # Anthropic SDK stores the key on .api_key
    assert client.api_key == "toup_ct_test_xyz"


def test_anthropic_client_carries_waf_safe_user_agent():
    """Cloudflare in front of toup.ai blocks the SDK's default
    "Anthropic/Python X.Y.Z" user-agent as a bot. Bundle clients MUST
    rewrite UA at the httpx layer via event hook. (matin incident —
    fourth chained root cause.)"""
    from app.services.bundle_client import make_anthropic_client

    with _bundle_settings():
        client = make_anthropic_client()

    # The event hook is what does the override; assert it's installed and
    # rewrites both UA and stainless headers when invoked.
    hooks = client._client.event_hooks.get("request", [])
    assert hooks, "expected request event hook to be installed"

    import httpx, asyncio
    req = httpx.Request(
        "POST", "https://toup.ai/x",
        headers={
            "user-agent": "AsyncAnthropic/Python 0.96.0",
            "x-stainless-package-version": "0.96.0",
            "x-stainless-lang": "python",
        },
    )
    asyncio.run(hooks[0](req))
    assert req.headers["user-agent"] == "toup-agent/1.0 (bundle-proxy)"
    assert "x-stainless-package-version" not in req.headers
    assert "x-stainless-lang" not in req.headers


def test_openai_client_in_bundle_mode_routes_through_proxy():
    """OpenAI bundle path: proxy base_url + /openai/v1 subpath +
    TOUP_TOKEN as api_key."""
    from app.services.bundle_client import make_openai_client

    with _bundle_settings():
        client = make_openai_client()

    assert client is not None
    assert str(client.base_url) == "https://toup.ai/api/llm/openai/v1/"
    assert client.api_key == "toup_ct_test_xyz"


def test_openai_client_carries_waf_safe_user_agent():
    from app.services.bundle_client import make_openai_client
    import httpx, asyncio

    with _bundle_settings():
        client = make_openai_client()

    hooks = client._client.event_hooks.get("request", [])
    assert hooks, "expected request event hook to be installed"
    req = httpx.Request(
        "POST", "https://toup.ai/x",
        headers={
            "user-agent": "AsyncOpenAI/Python 1.50.0",
            "x-stainless-package-version": "1.50.0",
        },
    )
    asyncio.run(hooks[0](req))
    assert req.headers["user-agent"] == "toup-agent/1.0 (bundle-proxy)"
    assert "x-stainless-package-version" not in req.headers


def test_byok_anthropic_falls_back_to_direct_when_not_bundle():
    """llm_mode != bundle → BYOK construction with the caller's key,
    no proxy involved."""
    from app.services.bundle_client import make_anthropic_client

    with _bundle_settings(llm_mode="manual", toup_token=""):
        client = make_anthropic_client(byok_key="sk-ant-api03-USERKEY")

    assert client is not None
    assert str(client.base_url).rstrip("/") == "https://api.anthropic.com"
    assert client.api_key == "sk-ant-api03-USERKEY"


def test_byok_oauth_token_uses_claude_code_headers():
    """sk-ant-oat tokens are Claude Code OAuth — caller wants the
    claude-code beta headers + claude-code user-agent. Only meaningful
    in BYOK; bundle proxy never sees these tokens."""
    from app.services.bundle_client import make_anthropic_client

    with _bundle_settings(llm_mode="manual", toup_token=""):
        client = make_anthropic_client(byok_key="sk-ant-oat01-OAUTHTOKEN")

    assert client is not None
    # OAuth path sets static headers on the http_client; assert at that layer.
    ua = client._client.headers.get("user-agent", "")
    assert "claude-code" in ua, f"OAuth path must use claude-code UA, got: {ua!r}"
    # Beta headers come via SDK default_headers
    default_h = getattr(client, "default_headers", {}) or {}
    assert "anthropic-beta" in default_h
    assert "claude-code-20250219" in default_h["anthropic-beta"]


def test_returns_none_when_not_bundle_and_no_byok_key():
    """Sane fallback when there's literally no way to talk to a provider."""
    from app.services.bundle_client import make_anthropic_client, make_openai_client

    with _bundle_settings(llm_mode="manual", toup_token=""):
        with patch("app.config.settings.anthropic_api_key", ""), \
             patch("app.config.settings.openai_api_key", ""):
            assert make_anthropic_client() is None
            assert make_openai_client() is None


def test_bundle_takes_precedence_over_byok_key():
    """If a tenant has llm_mode=bundle AND a BYOK key (e.g. legacy from
    pre-bundle setup), the proxy still wins. We don't want bundle
    subscribers leaking through to direct provider calls."""
    from app.services.bundle_client import make_anthropic_client

    with _bundle_settings(llm_mode="bundle", toup_token="toup_ct_xyz"):
        client = make_anthropic_client(byok_key="sk-ant-api03-LEGACY")

    assert str(client.base_url) == "https://toup.ai/api/llm/"
    assert client.api_key == "toup_ct_xyz"  # NOT the BYOK key
