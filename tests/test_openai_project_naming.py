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


def test_free_tier_activation_prefers_agent_name_over_user_name():
    """activate_free_tier must prefer the AGENT name (e.g. 'Aria') over
    the user's display name (e.g. 'NARIMAN') when labelling the OpenAI
    project. The OpenAI project represents the agent's API consumption,
    so the persona name is the more meaningful identifier when scanning
    the operator's dashboard. Falls back to the user's display name when
    the agent_name is missing (e.g. activation runs before Soul step).
    """
    from app.services import free_tier_activation
    src = inspect.getsource(free_tier_activation.activate_free_tier)
    # Both lookup branches must remain in source.
    assert "cfg.agent_name" in src, (
        "activate_free_tier must check cfg.agent_name first."
    )
    assert "from app.db.models import User" in src, (
        "activate_free_tier must still fall back to User.name when "
        "agent_name is missing."
    )
    assert "user_name=project_label" in src, (
        "The resolved label must be forwarded to "
        "_provision_openai_project_if_needed so the OpenAI dashboard "
        "shows readable per-agent project names."
    )
