"""R29-D — what the platform's env push renders, and for whom.

Two bugs pinned here, both found live on the founder tenant while
reviving the environment (2026-08-24):

1. The tenant-side automations flag had NO delivery path. The
   documented lever ("append AUTOMATIONS_ENABLED=true to the tenant
   .env") does not survive boot: `tunnel_client.py` replaces the whole
   file with the platform's rendered env (`_build_env`) on connect.
   Fix: the renderer itself emits the flag, resolved per-tenant from
   the same feature-flag readout the API gate uses (allowlist first,
   then pct bucketing). Dark tenants render byte-identically to
   before the flag existed.

2. `_resolve_model` rendered `_auto_model(...)` into AGENT_MODEL even
   when the platform truth was "no pin" (`agent_model=''`) — baking a
   concrete model into the pushed .env and silently overriding the
   fleet default on the tenant's next boot (observed: gpt-5.6-terra
   before a restart, claude-opus-4-7 after). In bundle mode an empty
   pin now renders NO AGENT_MODEL line, so the agent resolves its own
   fleet default — which is what `generate_env_content`'s contract
   said all along.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _cfg(**over):
    """A duck-typed AgentConfig with every attr _build_env touches."""
    base = dict(
        agent_api_key="k-agent", openai_api_key="", anthropic_api_key="",
        google_api_key="", mistral_api_key="", groq_api_key="",
        xai_api_key="", deepseek_api_key="", agent_model="",
        llm_mode="bundle", telegram_bot_token="", discord_bot_token="",
        slack_bot_token="", slack_app_token="",
        whatsapp_phone_number_id="", whatsapp_access_token="",
        whatsapp_verify_token="", whatsapp_app_secret="",
        whatsapp_mode="", whatsapp_self_e164="",
        whatsapp_baileys_allowlist="", brave_api_key="",
        elevenlabs_api_key="", connect_token="tok", db_mode="auto",
        supabase_url="",
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── 1. Flag delivery through the renderer ────────────────────────────


def test_generate_env_content_emits_flag_only_when_on():
    from app.services.ssh_deploy_service import generate_env_content

    on = generate_env_content(
        user_id="u1", agent_api_key="k", automations_enabled=True,
    )
    assert "AUTOMATIONS_ENABLED=true" in on

    off = generate_env_content(user_id="u1", agent_api_key="k")
    assert "AUTOMATIONS" not in off, (
        "a dark tenant's env must not mention the flag at all — "
        "absent key = agent-side default (False)"
    )


def test_build_env_threads_flag_through():
    from app.api.agent_setup import _build_env

    env_on = _build_env(_cfg(), "u1", automations_enabled=True)
    assert "AUTOMATIONS_ENABLED=true" in env_on
    env_off = _build_env(_cfg(), "u1")
    assert "AUTOMATIONS" not in env_off


@pytest.mark.asyncio
async def test_automations_env_flag_reads_the_real_allowlist():
    """The renderer's readout is the SAME one the API gate uses:
    allowlisted user ON at pct 0, everyone else dark."""
    from app.api.agent_setup import _automations_env_flag
    from app.db.database import async_session_maker
    from app.services.feature_flags import set_allowlist

    listed = str(uuid.uuid4())
    unlisted = str(uuid.uuid4())
    async with async_session_maker() as db:
        await set_allowlist(db, "automations", [listed])
        await db.commit()
    try:
        assert await _automations_env_flag(listed) is True
        assert await _automations_env_flag(unlisted) is False
    finally:
        async with async_session_maker() as db:
            await set_allowlist(db, "automations", [])
            await db.commit()


@pytest.mark.asyncio
async def test_automations_env_flag_fails_dark():
    """A broken flag read renders the tenant DARK, never lit."""
    from app.api import agent_setup as mod

    with patch(
        "app.services.feature_flags.is_enabled",
        side_effect=RuntimeError("db down"),
    ):
        assert await mod._automations_env_flag("any") is False


# ── 2. Empty model pin renders NO AGENT_MODEL line in bundle mode ────


def test_resolve_model_bundle_no_pin_omits_the_line():
    from app.api.agent_setup import _resolve_model
    from app.services.ssh_deploy_service import generate_env_content

    cfg = _cfg(llm_mode="bundle", agent_model="", anthropic_api_key="a-key")
    assert _resolve_model(cfg) == "", (
        "bundle + no pin must defer to the agent's fleet default, not "
        "bake _auto_model's pick into the pushed env"
    )
    env = generate_env_content(
        user_id="u1", agent_api_key="k", agent_model=_resolve_model(cfg),
    )
    assert "AGENT_MODEL" not in env


def test_resolve_model_explicit_pin_survives():
    from app.api.agent_setup import _resolve_model

    cfg = _cfg(llm_mode="bundle", agent_model="gpt-5.6-terra",
               openai_api_key="o-key")
    assert _resolve_model(cfg) == "gpt-5.6-terra"


def test_resolve_model_byok_no_pin_still_automodels():
    """Manual/BYOK keeps the old behavior: pick a model matching the
    keys the user actually holds — the agent default may need a key
    they don't have."""
    from app.api.agent_setup import _resolve_model

    cfg = _cfg(llm_mode="manual", agent_model="", anthropic_api_key="a-key")
    assert _resolve_model(cfg) != "", "BYOK must still get a concrete model"


def test_resolve_model_key_mismatch_still_falls_back():
    """The pre-existing guard is untouched: a pinned model whose
    provider key is missing falls back to _auto_model."""
    from app.api.agent_setup import _resolve_model

    cfg = _cfg(llm_mode="manual", agent_model="claude-opus-4-7",
               anthropic_api_key="", openai_api_key="o-key")
    resolved = _resolve_model(cfg)
    assert "claude" not in resolved.lower()
