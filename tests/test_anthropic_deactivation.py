"""Platform-wide Anthropic deactivation (2026-05-29).

The bundle Anthropic path rides ONE shared platform Claude account; when
it ran out of credit every bundle user's Claude call hard-400'd. Until
that account is funded we run OpenAI-only. These tests pin the three
enforcement layers:

  - model_router: never picks Claude when anthropic_enabled is False, and
    defaults bundle preference to OpenAI (not Anthropic) when enabled.
  - llm_proxy._route_chat: hard-rejects a Claude model when disabled
    (backstop for stale agents whose router still picks Claude).
  - api/models: omits Claude from the picker when disabled.
"""
from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "test")

import pytest


def _set_anthropic(monkeypatch, enabled: bool):
    from app.config import settings
    monkeypatch.setattr(settings, "anthropic_enabled", enabled, raising=False)


def test_router_routes_openai_when_anthropic_disabled(monkeypatch):
    from app.services import model_router
    from app.services.model_resolver import is_openai_model
    _set_anthropic(monkeypatch, False)
    for pref in ("anthropic", "openai", None):
        d = model_router.classify_request("hi", preferred_provider=pref)
        assert is_openai_model(d.model), (
            f"anthropic disabled must route OpenAI, got {d.model!r} for pref={pref!r}"
        )


def test_router_bundle_default_is_openai_when_enabled(monkeypatch):
    """Even with Anthropic ENABLED, an unset preferred_provider now defaults
    to OpenAI (was anthropic) — each user has their own OpenAI key, Anthropic
    is the shared-account path."""
    from app.services import model_router
    from app.services.model_resolver import is_openai_model
    from app.config import settings
    _set_anthropic(monkeypatch, True)
    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "toup_ct_test", raising=False)
    d = model_router.classify_request("hi", preferred_provider=None)
    assert is_openai_model(d.model), f"bundle default should be OpenAI, got {d.model!r}"


def test_router_explicit_anthropic_still_works_when_enabled(monkeypatch):
    """When Anthropic is enabled, an explicit preferred=anthropic opt-in
    still selects a Claude model (so re-enabling restores choice)."""
    from app.services import model_router
    from app.services.model_resolver import is_claude_model
    from app.config import settings
    _set_anthropic(monkeypatch, True)
    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "toup_ct_test", raising=False)
    d = model_router.classify_request("hi", preferred_provider="anthropic")
    assert is_claude_model(d.model), f"explicit anthropic should pick Claude, got {d.model!r}"


def test_proxy_rejects_claude_when_anthropic_disabled(monkeypatch):
    """The proxy backstop: a Claude model must be rejected (503), never
    routed to the unfunded shared Claude account, when disabled."""
    from fastapi import HTTPException
    from app.api import llm_proxy
    _set_anthropic(monkeypatch, False)

    class _Cfg:
        bundle_openai_api_key = "sk-proj-x"
        user_id = "u1"

    with pytest.raises(HTTPException) as ei:
        llm_proxy._route_chat("claude-opus-4-7", _Cfg())
    assert ei.value.status_code == 503
    assert "anthropic" in str(ei.value.detail).lower()


def test_proxy_allows_openai_when_anthropic_disabled(monkeypatch):
    """OpenAI models must route normally when Anthropic is disabled —
    the deactivation must not break the path we actually use."""
    from app.api import llm_proxy
    _set_anthropic(monkeypatch, False)

    class _Cfg:
        bundle_openai_api_key = "sk-proj-mine"
        user_id = "u1"

    backend, key = llm_proxy._route_chat("gpt-5.5", _Cfg())
    assert backend.name == "openai"
    assert key == "sk-proj-mine"  # per-tenant key preferred


def test_proxy_allows_claude_when_anthropic_enabled(monkeypatch):
    """Sanity: when re-enabled, Claude routes to the Anthropic backend."""
    from app.api import llm_proxy
    from app.config import settings
    _set_anthropic(monkeypatch, True)
    monkeypatch.setattr(settings, "platform_anthropic_api_key", "sk-ant-master", raising=False)

    class _Cfg:
        bundle_openai_api_key = None
        user_id = "u1"

    backend, key = llm_proxy._route_chat("claude-opus-4-7", _Cfg())
    assert backend.name == "anthropic"
