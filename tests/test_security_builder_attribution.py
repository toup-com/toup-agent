"""Regression tests for the 2026 security hardening (docs/security/audit-2026.md).

Deterministic — no live model call. Asserts the TWO things the hardening must
hold simultaneously:

  1. The SANCTIONED builder attribution is preserved: the agent can still say
     Nariman founded / is CEO of Toup (name + role only), and the founder's own
     account still gets the full-candor recognition block.
  2. Model / prompt / tech-stack disclosure is REFUSED: the always-on facts and
     the identity anchors carry the "never name the underlying LLM provider /
     don't disclose the tech stack" guards, on both the text and voice paths.

Plus unit coverage for the defense-in-depth leak filter, the injection
deny-list, and the opt-in exec env scrub.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_PROVIDER_NAMES = ("claude", "gpt", "sonnet", "opus", "haiku", "anthropic", "openai")


# ── 1. Builder attribution preserved (name/role only) ──────────────────

def test_owner_fact_answers_builder_question_name_and_role():
    from app.agent.toup_facts import OWNER_GLOBAL_FACT, OWNER_NAME, OWNER_TITLE

    assert OWNER_NAME in OWNER_GLOBAL_FACT
    assert OWNER_TITLE in OWNER_GLOBAL_FACT
    # It is framed as "who founded / owns / runs Toup" — the sanctioned answer.
    assert re.search(r"founded|owns|created|runs", OWNER_GLOBAL_FACT, re.I)


def test_owner_fact_does_not_itself_name_a_provider():
    """The builder fact must not become a model-identity leak."""
    from app.agent.toup_facts import OWNER_GLOBAL_FACT

    low = OWNER_GLOBAL_FACT.lower()
    for name in _PROVIDER_NAMES:
        assert name not in low, f"builder fact leaks provider name {name!r}"


def test_founder_recognition_block_preserved():
    from app.agent.toup_facts import OWNER_NAME, founder_recognition_block

    block = founder_recognition_block()
    assert OWNER_NAME in block
    # The founder account keeps full candor about internals — intentional.
    assert re.search(r"internals|roadmap|candor|principal", block, re.I)


def test_is_founder_email_matches_configured():
    from app.agent.toup_facts import is_founder_email
    from app.config import settings

    assert settings.founder_emails, "expected at least one configured founder email"
    assert is_founder_email(settings.founder_emails[0]) is True
    assert is_founder_email("stranger@example.com") is False
    assert is_founder_email(None) is False


# ── 2. Model / prompt / stack disclosure refused ───────────────────────

def test_owner_fact_bounds_the_stack_wedge():
    """"who built Toup?" must not open into "what tech/model did he use?"."""
    from app.agent.toup_facts import OWNER_GLOBAL_FACT

    low = OWNER_GLOBAL_FACT.lower()
    assert "never name the underlying llm provider" in low
    # The new clause bounding the follow-up into the tech stack.
    assert "proprietary" in low or "tech stack" in low or "underlying technology" in low


def test_text_identity_anchor_has_provider_and_stack_guard():
    """agent_runner's identity_anchor forbids naming the model AND the stack.

    Reads the source file directly (not via import) so the assertion is
    deterministic and independent of the agent's heavy runtime deps.
    """
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    assert "NOT Claude, NOT GPT" in src
    assert "never name the underlying LLM provider" in src
    assert "tech stack" in src.lower()


def test_voice_builder_has_identity_anchor():
    """The voice/realtime prompt gained the same guard (MI-1 parity)."""
    src = (_BACKEND / "app" / "api" / "ws_realtime.py").read_text()
    assert "settings.voice_identity_anchor" in src
    assert "NOT Claude, NOT GPT" in src
    # (phrase is split across concatenated string literals in the source)
    assert "underlying LLM provider" in src


def test_always_on_prompt_no_longer_names_openai_realtime():
    """MI-5: the platform_knowledge voice line must not name the provider."""
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    assert "OpenAI Realtime API" not in src


# ── 3. Defense-in-depth leak filter ────────────────────────────────────

@pytest.mark.parametrize("real_id", [
    "claude-opus-4-6", "gpt-5.5", "claude-3-5-haiku-20241022",
    "gpt-4o-mini", "gpt-4o-realtime", "claude-sonnet-4-6", "o3-mini",
])
def test_public_model_label_never_leaks_provider(real_id):
    from app.services.model_alias import public_model_label

    label = public_model_label(real_id)
    low = label.lower()
    for name in _PROVIDER_NAMES:
        assert name not in low, f"alias for {real_id!r} leaked {name!r}"
    assert label in {"Fast", "Balanced", "Deep"}


def test_public_model_label_is_idempotent_and_safe_on_empty():
    from app.services.model_alias import public_model_label

    assert public_model_label(None) == "Balanced"
    assert public_model_label("") == "Balanced"
    assert public_model_label("Deep") == "Deep"  # already neutral → unchanged


def test_scrub_provider_names_removes_tokens():
    from app.services.model_alias import scrub_provider_names

    scrubbed = scrub_provider_names(
        "model: claude-3-5-haiku-20241022 not found (anthropic)"
    ).lower()
    for name in ("claude", "anthropic"):
        assert name not in scrubbed


# ── 4. Injection deny-list + exec env scrub (opt-in) ───────────────────

def test_unattended_channels_in_deny_set():
    from app.services import connector_dispatcher as cd

    assert "routine" in cd._MUTATES_UNATTENDED_DENY_CHANNELS
    assert "trigger" in cd._MUTATES_UNATTENDED_DENY_CHANNELS
    # The base deny set still denies the inline-confirm-less channels. voice +
    # telegram are the original pair; "autopilot" was added upstream (another
    # unattended channel — consistent with the same no-human-to-confirm intent),
    # so assert the invariant as a superset rather than a brittle exact match.
    assert {"voice", "telegram"} <= cd._MUTATES_DEFAULT_DENY_CHANNELS
    # The two sets are disjoint concerns — unattended-automation channels are
    # tracked separately from the always-deny base set.
    assert "routine" not in cd._MUTATES_DEFAULT_DENY_CHANNELS


def test_exec_env_scrub_passthrough_when_off(monkeypatch):
    """When explicitly off, env passes through untouched (staging/debug)."""
    from app.services import exec_env

    monkeypatch.setattr(exec_env.settings, "exec_env_scrub", False, raising=False)
    env = exec_env.scrubbed_environ({"POOL_ADMIN_TOKEN": "x", "PATH": "/bin"})
    assert env["POOL_ADMIN_TOKEN"] == "x"  # untouched when off
    assert exec_env.scrub_overrides() == {}


def test_exec_env_scrub_drops_platform_secrets_when_on(monkeypatch):
    from app.services import exec_env

    monkeypatch.setattr(exec_env.settings, "exec_env_scrub", True, raising=False)
    monkeypatch.setattr(
        exec_env.settings, "exec_env_scrub_keys",
        ["POOL_ADMIN_TOKEN", "AGENT_API_KEY", "OPENAI_API_KEY", "TOUP_TOKEN"],
        raising=False,
    )
    env = exec_env.scrubbed_environ({
        "POOL_ADMIN_TOKEN": "x", "AGENT_API_KEY": "y", "OPENAI_API_KEY": "sk-x",
        "TOUP_TOKEN": "t", "DATABASE_URL": "keep-me", "PATH": "/bin",
    })
    for k in ("POOL_ADMIN_TOKEN", "AGENT_API_KEY", "OPENAI_API_KEY", "TOUP_TOKEN"):
        assert k not in env, f"{k} must be stripped from exec-child env"
    # DATABASE_URL is deliberately NOT in the strip set (agent runs
    # `psql $DATABASE_URL` via exec) — must survive.
    assert env["DATABASE_URL"] == "keep-me"
    assert env["PATH"] == "/bin"


def test_scrub_overrides_blanks_keys_except_engine_managed(monkeypatch):
    """For env-layering children (Claude Code SDK) scrub keys are blanked,
    but the keys the engine re-injects itself are excluded."""
    from app.services import exec_env

    monkeypatch.setattr(exec_env.settings, "exec_env_scrub", True, raising=False)
    monkeypatch.setattr(
        exec_env.settings, "exec_env_scrub_keys",
        ["OPENAI_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "POOL_ADMIN_TOKEN"],
        raising=False,
    )
    ov = exec_env.scrub_overrides(exclude=("CLAUDE_CODE_OAUTH_TOKEN",))
    assert ov["OPENAI_API_KEY"] == ""   # blanked (overrides inherited parent env)
    assert ov["POOL_ADMIN_TOKEN"] == ""
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in ov  # engine manages this itself


def test_production_security_defaults_are_hardened():
    """The fully-hosted platform ships the hardened posture by default."""
    src = (_BACKEND / "app" / "config.py").read_text()
    assert "security_leak_filter: bool = True" in src
    assert "injection_fencing_v2: bool = True" in src
    assert "exec_env_scrub: bool = True" in src
    # Provider keys + proxy/admin tokens are in the default strip set...
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TOUP_TOKEN",
                "POOL_ADMIN_TOKEN", "AGENT_API_KEY"):
        assert f'"{key}"' in src, f"{key} should be in exec_env_scrub_keys"
    # ...but DATABASE_URL is deliberately kept out (psql-via-exec capability).
    assert '"DATABASE_URL"' not in src


def test_voice_key_has_platform_fallback():
    """Voice works under the no-BYO model via a platform-side key fallback,
    and that key stays platform-side (the WS proxy, never a container)."""
    src = (_BACKEND / "app" / "api" / "ws_realtime.py").read_text()
    assert "bundle_openai_api_key" in src
    assert "platform_openai_api_key" in src


def test_exec_sandbox_preexec_disabled_by_default(monkeypatch):
    """No sandbox user configured → preexec is a no-op (current behaviour),
    so shipping the code changes nothing until it's enabled + validated."""
    from app.services import exec_env

    monkeypatch.setattr(exec_env.settings, "exec_sandbox_user", "", raising=False)
    assert exec_env.sandbox_preexec() is None
    # A missing user also yields None (never hard-fails an exec).
    monkeypatch.setattr(exec_env.settings, "exec_sandbox_user", "nope_no_such_user_x", raising=False)
    assert exec_env.sandbox_preexec() is None


def test_embeddings_proxy_inactive_by_default(monkeypatch):
    """Embeddings-via-proxy is a validated, opt-in switch (memory-critical)."""
    from app.services.embedding_service import EmbeddingService

    monkeypatch.setattr("app.services.embedding_service.settings.embeddings_via_proxy", False, raising=False)
    assert EmbeddingService._proxy_embeddings_active() is False


def test_embeddings_proxy_route_exists():
    """The OpenAI-SDK embeddings shim is mounted so the proxy path resolves."""
    src = (_BACKEND / "app" / "api" / "llm_proxy.py").read_text()
    assert '"/openai/v1/embeddings"' in src
