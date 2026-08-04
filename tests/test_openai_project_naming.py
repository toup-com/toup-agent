"""Regression for the OpenAI project-naming change (2026-05-24).

The operator's dashboard was showing opaque `toup-tenant-<8hex>` project
names, which made it impossible to tell which project belonged to which
user without grepping the DB. This commit threads the user's display
name into the project name so the dashboard reads as
`toup-<NAME>-<prefix>`.

These tests are pure-function (no DB, no network) and just exercise the
naming helper + the wrapper. The OpenAI Admin API itself is mocked in
the operator's staging smoke run.
"""
from __future__ import annotations

import inspect

from app.services import openai_admin_service


def test_sanitize_strips_non_alphanumeric():
    fn = openai_admin_service._sanitize_for_project_name
    assert fn("NARIMAN") == "NARIMAN"
    assert fn("john doe") == "john-doe"
    assert fn("Arman_Hosseini!") == "Arman-Hosseini"
    # Internationalised characters are scrubbed — OpenAI project-name
    # validator accepts a narrow ASCII set, and leaving them in would
    # cause create_project() to 400.
    assert fn("نریمان") == ""
    assert fn("张伟") == ""


def test_sanitize_trims_and_strips_hyphen_artifacts():
    fn = openai_admin_service._sanitize_for_project_name
    # Leading/trailing hyphens left after substitution must be stripped
    # — OpenAI rejects names that start or end with separator chars.
    assert fn("---Nariman---") == "Nariman"
    assert fn("!Nariman!") == "Nariman"
    # Length cap (default 30) prevents the user from blowing past
    # OpenAI's name validator on long display names.
    longname = "A" * 50
    assert len(fn(longname)) == 30


def test_provision_tenant_signature_accepts_user_name():
    """The provision_tenant wrapper must accept an optional `user_name`
    kwarg so callers can request the new readable naming without
    breaking existing call sites that haven't been updated yet.
    """
    sig = inspect.signature(openai_admin_service.provision_tenant)
    assert "user_name" in sig.parameters
    # Default must be None (or some falsy sentinel) so existing
    # positional `provision_tenant(prefix)` calls don't break.
    assert sig.parameters["user_name"].default is None


def test_provision_tenant_uses_user_name_in_project_label():
    """When `user_name` is passed, the project name interpolates the
    sanitised name; when omitted, falls back to the legacy
    `toup-tenant-<prefix>` form.
    """
    src = inspect.getsource(openai_admin_service.provision_tenant)
    # Both naming branches must remain in source.
    assert 'f"toup-{safe_name}-{prefix}"' in src, (
        "Named-project form must use 'toup-<name>-<prefix>' so the "
        "operator's OpenAI dashboard can identify users at a glance."
    )
    assert 'f"toup-tenant-{prefix}"' in src, (
        "Legacy fallback must still be present so callers that don't "
        "have the user's name (or whose name sanitises to '') keep "
        "the existing naming."
    )


def test_billing_helper_accepts_user_name_kwarg():
    """billing._provision_openai_project_if_needed must accept the
    user_name kwarg and forward it. Without this, free_tier_activation's
    call site has nowhere to inject the name.
    """
    from app.api import billing
    sig = inspect.signature(billing._provision_openai_project_if_needed)
    assert "user_name" in sig.parameters
    src = inspect.getsource(billing._provision_openai_project_if_needed)
    assert "provision_tenant(prefix, user_name=user_name)" in src, (
        "The billing wrapper must forward user_name to provision_tenant; "
        "otherwise the naming change silently no-ops on the activation path."
    )


def test_free_tier_activation_prefers_agent_name_over_user_name(monkeypatch):
    """The OpenAI project label must be the AGENT persona name, falling back
    to the user's display name when there is no agent name yet (activation can
    run before the Soul step).

    This used to be three `inspect.getsource(activate_free_tier)` substring
    checks, and it went red without anything breaking: the logic was moved
    into `_run_openai_project_provision` (free_tier_activation.py:88-100),
    where it still does exactly this. A test that greps for a line is really
    testing where the line lives.

    So call the thing and look at what it passes. This version would have
    stayed green through that refactor, and goes red if the preference order
    is ever reversed — which the grep could not tell you.
    """
    import asyncio as _asyncio

    from app.services import free_tier_activation as fta

    seen: dict = {}

    class _FakeCfg:
        def __init__(self, agent_name):
            self.agent_name = agent_name
            self.bundle_openai_project_id = None
            self.bundle_openai_api_key = None

    class _FakeUser:
        name = "NARIMAN"

    def _fake_session(cfg):
        class _Res:
            def scalar_one_or_none(self_inner):
                return cfg

        class _DB:
            async def execute(self_inner, *a, **k):
                return _Res()

            async def get(self_inner, model, ident):
                return _FakeUser()

            async def commit(self_inner):
                return None

            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *a):
                return False

        def _maker():
            return _DB()

        return _maker

    import app.api.billing as billing

    def _record(cfg, user_name=None):
        seen["user_name"] = user_name

    monkeypatch.setattr(billing, "_provision_openai_project_if_needed", _record)

    # 1. agent_name present -> the persona name wins
    import app.db.database as dbmod
    monkeypatch.setattr(dbmod, "async_session_maker", _fake_session(_FakeCfg("Aria")))
    _asyncio.run(fta._run_openai_project_provision("u-1"))
    assert seen["user_name"] == "Aria", (
        "the agent persona name must label the OpenAI project — it is what "
        f"makes the operator's dashboard readable; got {seen['user_name']!r}"
    )

    # 2. no agent_name yet (activation before the Soul step) -> user's name
    seen.clear()
    monkeypatch.setattr(dbmod, "async_session_maker", _fake_session(_FakeCfg(None)))
    _asyncio.run(fta._run_openai_project_provision("u-1"))
    assert seen["user_name"] == "NARIMAN", (
        f"must fall back to the user's display name; got {seen['user_name']!r}"
    )

