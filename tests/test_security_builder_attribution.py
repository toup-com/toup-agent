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


def test_sandbox_preexec_resolves_existing_user(monkeypatch):
    """When exec_sandbox_user names a real OS user, preexec returns a callable
    (the drop fn). Uses the test-runner's own user so it resolves everywhere."""
    import getpass
    from app.services import exec_env
    monkeypatch.setattr(exec_env.settings, "exec_sandbox_user", getpass.getuser(), raising=False)
    fn = exec_env.sandbox_preexec()
    assert callable(fn)


def test_agent_image_enables_exec_sandbox_as_toup():
    """The agent image drops exec to the workspace-owning `toup` uid and makes
    /app traversable-but-not-readable so exec keeps workspace R/W while losing
    /app source + /proc secret access (docs/security/audit-2026.md EXF-3)."""
    df = (_BACKEND / "Dockerfile.agent").read_text()
    assert "ENV EXEC_SANDBOX_USER=toup" in df
    assert "chmod -R o-rwx /app" in df        # source unreadable to non-root
    assert "chmod o+x /app" in df             # ...but /app dir is traversable
    # The boot step keeps the workspace owned by toup (the exec uid).
    assert "chown -R toup:toup /app/workspace" in df


def test_embeddings_proxy_inactive_by_default(monkeypatch):
    """Embeddings-via-proxy is a validated, opt-in switch (memory-critical)."""
    from app.services.embedding_service import EmbeddingService

    monkeypatch.setattr("app.services.embedding_service.settings.embeddings_via_proxy", False, raising=False)
    assert EmbeddingService._proxy_embeddings_active() is False


def test_embeddings_proxy_route_exists():
    """The OpenAI-SDK embeddings shim is mounted so the proxy path resolves."""
    src = (_BACKEND / "app" / "api" / "llm_proxy.py").read_text()
    assert '"/openai/v1/embeddings"' in src


# ── Round 2: residual gaps found by the adversarial verification pass ───
# (docs/security/audit-2026.md — WS/voice model-id, analyze_image fence,
# doctor disclosure, app-builder env scrub, telegram siblings, latent channels)


def test_ws_chat_done_frame_aliases_model():
    """The live WS chat `done` frame must alias the model id under the flag."""
    src = (_BACKEND / "app" / "api" / "ws_chat.py").read_text()
    assert "public_model_label" in src and "security_leak_filter" in src
    # The raw response.model must NOT be assigned directly into the done frame.
    assert '"model": response.model' not in src


def test_sse_done_frame_aliases_model():
    src = (_BACKEND / "app" / "api" / "api_v1.py").read_text()
    assert '"model": response.model' not in src
    assert "public_model_label" in src


def test_voice_response_done_aliases_model():
    src = (_BACKEND / "app" / "api" / "ws_realtime.py").read_text()
    assert '"model": turn_model' not in src
    assert "public_model_label" in src


def test_analyze_image_is_fenced():
    """analyze_image (OCR) output is treated as untrusted external content."""
    src = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    # It must appear inside the _EXTERNAL_CONTENT_TOOLS set.
    marker = src.split("_EXTERNAL_CONTENT_TOOLS", 1)[1][:400]
    assert '"analyze_image"' in marker


def test_scrub_stack_terms_neutralizes_infra_names():
    from app.services.model_alias import scrub_stack_terms
    out = scrub_stack_terms("running on Docker with FastAPI + pgvector and Playwright")
    for tok in ("docker", "fastapi", "pgvector", "playwright"):
        assert tok not in out.lower()
    assert "[component]" in out
    assert scrub_stack_terms("") == ""


def test_doctor_output_scrubbed_and_no_key_fragment():
    """doctor tool scrubs provider/stack names; key checks never echo material."""
    te = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    assert "scrub_stack_terms" in te and "scrub_provider_names" in te
    doc = (_BACKEND / "app" / "agent" / "cli_doctor.py").read_text()
    # The key-prefix leak (key[:8]) must be gone from the key checks.
    assert "key[:8]" not in doc


def test_app_builder_subprocess_env_scrubbed():
    """Every app_manager child gets a scrubbed env (no raw os.environ spawn)."""
    src = (_BACKEND / "app" / "agent" / "app_manager.py").read_text()
    assert "from app.services.exec_env import scrubbed_environ" in src
    # No spawn should splat the full os.environ into a child.
    assert "{**os.environ," not in src


def test_telegram_sibling_commands_alias_model():
    src = (_BACKEND / "app" / "agent" / "telegram_bot.py").read_text()
    assert "_tg_model_label" in src
    # The raw settings.agent_model must not be printed unaliased in status.
    assert "Model: <code>{settings.agent_model}</code>" not in src


def test_unattended_executors_tag_channel():
    """cron + heartbeat pass an unattended channel so the mutating deny applies."""
    cron = (_BACKEND / "app" / "agent" / "cron_service.py").read_text()
    hb = (_BACKEND / "app" / "agent" / "heartbeat_service.py").read_text()
    assert 'channel="cron"' in cron
    assert 'channel="heartbeat"' in hb
    # And those channels are in the deny set.
    from app.services import connector_dispatcher as cd
    assert "cron" in cd._MUTATES_UNATTENDED_DENY_CHANNELS
    assert "heartbeat" in cd._MUTATES_UNATTENDED_DENY_CHANNELS


def test_tool_descriptions_drop_provider_names():
    src = (_BACKEND / "app" / "agent" / "tool_definitions.py").read_text()
    assert "GPT vision" not in src
    assert "gpt-image-1" not in src
    assert "ChatGPT" not in src


def test_parallel_web_trim_reseals_fence():
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    assert "</external_content>" in src
    assert "_r.rstrip().endswith" in src


def test_public_health_omits_embedding_model():
    src = (_BACKEND / "platform_main.py").read_text()
    # The health payload must not name the embedding model.
    assert '"embedding_model": settings.embedding_model' not in src


def test_models_endpoint_gated_when_leak_filter_on():
    """GET /api/models requires a valid session when security_leak_filter is on,
    so anonymous callers can't enumerate the real model catalogue (audit MI)."""
    src = (_BACKEND / "app" / "api" / "models.py").read_text()
    assert "security_leak_filter" in src
    assert "decode_access_token" in src
    assert "status_code=401" in src


def test_web_models_hook_sends_credentials():
    """The web model hook sends the session cookie so the gated endpoint works
    for logged-in users (paired with the backend gate above)."""
    hook = (_BACKEND.parent / "frontend" / "src" / "hooks" / "useModels.ts").read_text()
    assert "credentials: 'include'" in hook
    assert "credentials: 'omit'" not in hook


# ── Round 5: gaps found by the final adversarial re-audit (16 confirmed) ───


def test_mcp_known_channels_superset_of_deny_sets():
    """The transport channel allow-list must include every unattended channel
    the dispatcher denies, else the deny is silently bypassed (channel clamped
    to web). Fail-closed clamp target must be a deny-set member."""
    from app.mcp_auth import _KNOWN_CHANNELS, _UNKNOWN_CHANNEL_CLAMP
    from app.services import connector_dispatcher as cd
    deny = cd._MUTATES_DEFAULT_DENY_CHANNELS | cd._MUTATES_UNATTENDED_DENY_CHANNELS
    assert deny <= _KNOWN_CHANNELS, f"deny channels not all known: {deny - _KNOWN_CHANNELS}"
    assert _UNKNOWN_CHANNEL_CLAMP in cd._MUTATES_UNATTENDED_DENY_CHANNELS


def test_scrub_stack_terms_covers_embedding_names():
    from app.services.model_alias import scrub_stack_terms
    out = scrub_stack_terms("pgvector + all-MiniLM-L6-v2 / text-embedding-3-small")
    assert "MiniLM".lower() not in out.lower()
    assert "text-embedding-3-small" not in out


def test_file_tools_path_jail_present():
    """_resolve_path must jail file tools out of /proc + /app source (the
    in-process file-tool EXF-3 hole the exec sandbox did not cover)."""
    src = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    assert "_guard_path" in src
    assert "/proc" in src and "platform source" in src
    # grep gets a NUL/binary guard so /proc/*/environ can't be line-dumped.
    assert "Binary file skipped" in src


def test_app_builder_spawns_drop_privileges():
    """app_manager + app_builder npm/expo spawns must drop to the sandbox uid
    (preexec) and no longer inherit the full root os.environ."""
    am = (_BACKEND / "app" / "agent" / "app_manager.py").read_text()
    assert "sandbox_preexec" in am and "preexec_fn=sandbox_preexec()" in am
    assert "os.chown(APPS_DIR" in am
    skill = (_BACKEND / "app" / "agent" / "skills" / "builtins" / "app_builder" / "skill.py").read_text()
    assert "sandbox_preexec" in skill
    assert '{**os.environ, "EXPO_NO_TELEMETRY"' not in skill  # no raw root env


def test_credits_ledger_and_v1_chat_alias_model():
    cr = (_BACKEND / "app" / "api" / "credits.py").read_text()
    assert "public_model_label" in cr and "public_provider_label" in cr
    assert "model=r.model, provider=r.provider" not in cr
    v1 = (_BACKEND / "app" / "api" / "api_v1.py").read_text()
    # non-streaming ChatResponse must not pass the raw model straight through
    assert "model=response.model," not in v1


def test_tool_descriptions_scrubbed_at_runtime():
    ar = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    assert "_scrub_tool_descriptions" in ar
    # system prompt no longer names the browser engine / db brand
    assert "headless Chromium" not in ar
    assert "tenant PostgreSQL database" not in ar


def test_web_fetch_has_ssrf_guard():
    rd = (_BACKEND / "app" / "agent" / "smart_fetch" / "reader.py").read_text()
    assert "_assert_public_url" in rd
    assert "is_private" in rd and "is_link_local" in rd
    assert "follow_redirects=False" in rd  # redirects guarded per hop


# ── Round 6: gaps the focused re-audit found (incl. round-5 regressions) ───


def test_mcp_known_channels_include_orchestration_channels():
    """The round-5 fail-closed clamp must NOT over-deny legit internal channels
    (subagent/app_builder/…) — they pass through so the dispatcher applies its
    normal policy (re-audit round 6 regression)."""
    from app.mcp_auth import _KNOWN_CHANNELS
    for ch in ("subagent", "app_builder", "netflix", "chat_intent", "app", "api", "agent"):
        assert ch in _KNOWN_CHANNELS, f"legit channel {ch} would be wrongly clamped"


def test_path_jail_is_allow_list_and_covers_walk():
    """The file-tool jail is deny-by-default (allow-list) so grep path='/' or
    '/etc' is rejected, and grep's recursive walk guards each file (symlink to
    /proc can't leak) — re-audit round 6 CRITICAL."""
    src = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    assert "_allowed_path_roots" in src
    assert "outside the allowed workspace" in src  # allow-list deny message
    # grep's recursive walk guards each file before opening it.
    assert "self._guard_path(fpath" in src


def test_web_fetch_ssrf_guard_before_browser_fallback():
    src = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    fn = src.split("async def _tool_web_fetch", 1)[1].split("async def ", 1)[0]
    # guard runs before the browser fallback (browser_fetch_enabled)
    assert "_assert_public_url" in fn
    assert fn.index("_assert_public_url") < fn.index("browser_fetch_enabled")


def test_browser_tool_has_ssrf_guard():
    src = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    fn = src.split("async def _tool_browser(", 1)[1].split("async def ", 1)[0]
    assert "_assert_public_url" in fn


def test_tunnel_endpoints_gated():
    src = (_BACKEND / "app" / "api" / "ws_agent_tunnel.py").read_text()
    # tunnel_status now authenticates + refuses cross-tenant disclosure
    st = src.split('async def tunnel_status(', 1)[1].split("@router", 1)[0]
    assert "_authenticate_tunnel" in st and "never disclose another tenant" in st
    # tunnel_debug no longer seeds tunnels_active before auth
    assert 'result: dict = {"tunnels_active": list(_tunnels.keys())}' not in src
