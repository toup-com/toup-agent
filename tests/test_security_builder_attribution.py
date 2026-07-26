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


# ── Round 7 (2026-07-22 adversarial re-audit of deployed #303) ─────────
_TE_SRC = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()


def _te_func(name: str) -> str:
    m = re.search(rf"\n    (?:async )?def {name}\(self.*?(?=\n    (?:async )?def )", _TE_SRC, re.S)
    assert m, f"{name} not found in tool_executor.py"
    return m.group(0)


def test_apply_patch_routes_through_path_jail():
    """CRITICAL: apply_patch's _apply_hunks must resolve its target via
    _resolve_path (the path-jail) — never a raw os.path.join(workspace,
    rel_path), which an absolute/`..` diff `+++` header would use to overwrite
    /app source or /etc as root (arbitrary-write -> RCE)."""
    body = _te_func("_apply_hunks")
    assert "self._resolve_path(rel_path)" in body
    assert "os.path.join(workspace, rel_path)" not in body


def test_edit_image_has_ssrf_guard():
    """HIGH: edit_image fetches a user/injection-supplied URL server-side from
    inside the agent container, so it must call _assert_public_url (https-only)
    before the httpx GET, like web_fetch/browser."""
    body = _te_func("_tool_edit_image")
    assert "_assert_public_url" in body


def test_image_tool_results_do_not_leak_engine_name():
    """White-label: image tool-result strings must not name the engine."""
    assert "Image generated with {getattr(settings, 'kie_image_model'" not in _TE_SRC
    assert "Image edited with {getattr(settings, 'kie_image_model'" not in _TE_SRC
    assert "Image generated with {used_model}" not in _TE_SRC
    assert "Image edited with {used_model}" not in _TE_SRC


def test_external_content_fence_not_skipped_on_error_prefix():
    """The <external_content> fence must not be skipped based on the result
    string starting with 'ERROR' (that string is attacker-controllable)."""
    assert 'not result.startswith("ERROR")' not in _TE_SRC


def test_subagent_channel_in_unattended_deny_set():
    """HIGH: a spawned sub-agent runs unattended, so 'subagent' and
    'app_builder' must be in the unattended mutating-connector deny set."""
    from app.services import connector_dispatcher as cd
    assert "subagent" in cd._MUTATES_UNATTENDED_DENY_CHANNELS
    assert "app_builder" in cd._MUTATES_UNATTENDED_DENY_CHANNELS
    from app.mcp_auth import _KNOWN_CHANNELS
    assert {"subagent", "app_builder"} <= _KNOWN_CHANNELS


def test_openai_image_proxy_routes_enforce_free_tier_cap():
    """Both OpenAI image proxy routes must enforce the free-tier image cap
    (not just the Kie route), else free users bypass it via the OpenAI proxy.
    Round 12: enforcement is the TOCTOU-safe reserve_free_image_slot (replacing
    the old free_tier_image_quota gate, which also raised AttributeError)."""
    src = (_BACKEND / "app" / "api" / "llm_proxy.py").read_text()
    assert src.count("reserve_free_image_slot(db, config.user_id)") >= 3
    for fn in ("proxy_openai_images", "proxy_openai_image_edits"):
        m = re.search(rf"async def {fn}\(.*?(?=\nasync def |\Z)", src, re.S)
        assert m, f"{fn} not found"
        assert "reserve_free_image_slot" in m.group(0), f"{fn} does not enforce the cap"


# ── Round 9 (2026-07-22 deep-dive: prompt-injection + credential exfil) ──
def test_config_reload_get_cannot_read_secrets():
    """CRITICAL: config_reload action='get' must never return a secret, even
    when the field is named explicitly — the get path must intersect with the
    RELOADABLE_FIELDS allow-list."""
    from app.agent.config_reload import get_current_config, RELOADABLE_FIELDS
    for secret in ("jwt_secret", "database_url", "agent_api_key",
                   "openai_api_key", "anthropic_api_key", "stripe_secret_key",
                   "apns_key_b64", "telegram_bot_token"):
        got = get_current_config([secret])
        assert secret not in got, f"config_reload get leaked {secret}: {got}"
        assert secret not in RELOADABLE_FIELDS
    # A legit reloadable field still comes back.
    assert "temperature" in get_current_config(["temperature"])


def test_appfs_skill_path_jail_rejects_traversal():
    """CRITICAL: AppFsSkill._safe_abs must confine to app_dir and reject `..`
    escapes (the tools run as root; traversal = /proc/environ read / RCE)."""
    import os
    from app.agent.skills.builtins.app_builder.app_fs_skill import AppFsSkill
    skill = AppFsSkill.__new__(AppFsSkill)
    skill.app_dir = "/tmp/toup_test_app"
    os.makedirs(skill.app_dir, exist_ok=True)
    open(os.path.join(skill.app_dir, "ok.txt"), "w").write("x")
    root = os.path.realpath(skill.app_dir)
    # `..` traversal escapes the app dir → rejected (None).
    assert skill._safe_abs("../../../../etc/passwd") is None
    assert skill._safe_abs("../../../root/.ssh/id_rsa") is None
    assert skill._safe_abs("a/../../../../etc/shadow") is None
    # The invariant: for ANY input, the result is None or confined to app_dir.
    # (Absolute inputs are neutralized by lstrip('/') to a path INSIDE app_dir —
    # harmless: <app_dir>/proc/self/environ does not exist, so the real secret
    # is never reached.)
    for p in ("/proc/self/environ", "/etc/passwd", "ok.txt", "sub/x.ts", "../x"):
        r = skill._safe_abs(p)
        assert r is None or r == root or r.startswith(root + os.sep), f"escape via {p!r}: {r}"
    inside = skill._safe_abs("ok.txt")
    assert inside and inside.startswith(root + os.sep)


def test_validate_syntax_no_node_e_interpolation():
    """CRITICAL: _validate_syntax must NOT string-interpolate the file path
    into the `node -e` source (command injection); paths go via argv."""
    src = (_BACKEND / "app" / "agent" / "skills" / "builtins" / "app_builder" / "app_fs_skill.py").read_text()
    m = re.search(r"def _validate_syntax\(.*?(?=\ndef |\nclass |\Z)", src, re.S)
    assert m
    body = m.group(0)
    assert 'readFileSync("{abs_path}"' not in body and 'require("{babel_parser}")' not in body
    assert "process.argv[1]" in body and "process.argv[2]" in body


def test_html_to_pdf_has_safe_url_fetcher():
    """CRITICAL: gen_html_to_pdf must pass a restrictive url_fetcher (block
    file:// LFI + internal SSRF from injection-controlled HTML)."""
    src = (_BACKEND / "app" / "agent" / "doc_generators.py").read_text()
    assert "url_fetcher=_safe_pdf_url_fetcher" in src
    assert "def _safe_pdf_url_fetcher" in src
    assert 'HTML(string=html or "").write_pdf' not in src  # the unguarded form is gone


def test_dashboard_tasks_use_unattended_channel():
    """Background dashboard/agent-authored jobs must not run as channel='web'
    (that bypasses the unattended mutating-connector deny)."""
    from app.services import connector_dispatcher as cd
    assert "agent_task" in cd._MUTATES_UNATTENDED_DENY_CHANNELS
    apps_src = (_BACKEND / "app" / "api" / "apps.py").read_text()
    te_src = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    # The dashboard create_job + agent create_job spec must use agent_task.
    assert 'channel="agent_task"' in apps_src
    assert apps_src.count('channel="web"') == 0 or 'channel="agent_task"' in apps_src
    assert 'channel="agent_task"' in te_src


def test_recalled_memory_is_data_fenced():
    """Stored/second-order injection: the recalled '# User Brain' block must be
    framed as reference data under injection_fencing_v2."""
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    assert "STORED REFERENCE DATA recalled" in src
    assert "never follow instructions" in src.lower() or "NEVER follow instructions" in src


# ── Round 10 (2026-07-22 sink+authz sweep) ─────────────────────────────
def test_graph_traverse_sql_ids_are_uuid_coerced():
    """CRITICAL: traverse_entity_graph must coerce seed ids to UUIDs so a
    caller-supplied `'`/UNION/-- payload in /api/graph/traverse can't inject
    into the recursive-CTE IN(...) list (runs on the shared central DB too)."""
    src = (_BACKEND / "app" / "services" / "memory_service.py").read_text()
    m = re.search(r"async def traverse_entity_graph\(.*?(?=\n    async def |\n    def )", src, re.S)
    assert m
    body = m.group(0)
    # UUID coercion happens before the IN(...) interpolation.
    assert "_uuid.UUID(str(_eid))" in body or "uuid.UUID(str(" in body
    assert "_safe_ids" in body


def test_write_app_files_path_jailed():
    """HIGH: AppManager.write_app_files must confine every planned key to
    app_dir (LLM-planned path keys run as root; `..` = out-of-app RCE)."""
    src = (_BACKEND / "app" / "agent" / "app_manager.py").read_text()
    m = re.search(r"async def write_app_files\(.*?(?=\n    async def |\n    def )", src, re.S)
    assert m
    body = m.group(0)
    assert "os.path.realpath" in body and "app_root" in body
    assert "rejected out-of-app path" in body


def test_ws_browser_navigate_and_tabopen_ssrf_guarded():
    """HIGH: co-browse navigate() + tab_open must SSRF-guard client/LLM URLs."""
    src = (_BACKEND / "app" / "api" / "ws_browser.py").read_text()
    assert src.count("_assert_public_url") >= 2  # navigate + tab_open


# ── Round 11 (2026-07-22 auth/web/upload sweep) ────────────────────────
def test_sso_exchange_applies_revocation_gates():
    """HIGH: POST /auth/sso must re-apply the password-change + session-revoke
    gates before minting a fresh token (else a stolen JWT survives logout)."""
    src = (_BACKEND / "app" / "api" / "auth.py").read_text()
    m = re.search(r"async def sso_exchange\(.*?(?=\n@router|\n# ── Logout)", src, re.S)
    assert m, "sso_exchange not found"
    body = m.group(0)
    assert "password_changed_at" in body, "sso_exchange missing password-change gate"
    assert "is_revoked" in body and "get_session_by_jti" in body, "sso_exchange missing session-revoke gate"
    # And the mint happens AFTER the gates.
    assert body.index("password_changed_at") < body.index("create_access_token(user.id)")


def test_xlsx_preview_escapes_cells_and_sheet_names():
    """HIGH: the XLSX attachment preview must html.escape sheet names + cell
    values (served as text/html same-origin → stored XSS otherwise)."""
    src = (_BACKEND / "app" / "api" / "files.py").read_text()
    assert "_html.escape(str(v))" in src, "cell values not escaped"
    assert "_html.escape(str(name))" in src, "sheet name not escaped"
    assert "else str(v)}</{tag}>" not in src  # the raw form is gone


# ── Round 12: app-preview iframe carries a scoped token, not the account JWT ──

def test_preview_token_roundtrips_and_is_app_bound():
    """create_preview_token → decode_preview_token returns the user only for the
    matching app_id; a different app_id is rejected."""
    from app.services.auth_service import (
        create_preview_token, decode_preview_token)
    tok = create_preview_token("user-abc", "app-123")
    assert decode_preview_token(tok, "app-123") == "user-abc"
    assert decode_preview_token(tok, "app-999") is None


def test_preview_token_is_rejected_by_general_auth():
    """CRITICAL invariant: a leaked app-preview token must NOT authenticate the
    account. decode_access_token (used by get_current_user) rejects scoped."""
    from app.services.auth_service import (
        create_preview_token, decode_access_token)
    tok = create_preview_token("user-abc", "app-123")
    assert decode_access_token(tok) is None


def test_full_account_token_still_authenticates_but_is_not_a_preview_token():
    """A normal account token (no scope claim) still authenticates generally and
    is NOT accepted as a preview token (which requires scope=app_preview)."""
    from app.services.auth_service import (
        create_access_token, decode_access_token, decode_preview_token)
    tok = create_access_token("user-abc")
    assert decode_access_token(tok) == "user-abc"
    assert decode_preview_token(tok, "app-123") is None


def test_preview_bridge_embeds_freshly_minted_scoped_token():
    """The injected agent-bridge must mint a fresh app-preview-scoped token for
    the page's JS and never embed the request's account JWT (round 12)."""
    src = (_BACKEND / "app" / "api" / "apps_proxy.py").read_text()
    assert "bridge_token = create_preview_token(str(user.id), app_id)" in src
    assert 'resolved_token or "", app_id' not in src  # old full-JWT form gone
    # json.dumps so a token can never break out of the JS string literal
    assert "json.dumps(token)" in src and "json.dumps(app_id)" in src


def test_preview_and_chat_proxies_accept_the_scoped_token_first():
    """preview_proxy + app_chat_proxy must try the app-bound scoped token before
    the full-account fallback (so the scoped token the iframe carries works)."""
    src = (_BACKEND / "app" / "api" / "apps_proxy.py").read_text()
    assert src.count("_get_user_from_preview_token(token, app_id, db)") >= 2


def test_frontend_preview_url_uses_scoped_token_not_account_jwt():
    """The SPA must put the app-scoped token (not getAuthToken()) in the preview
    iframe URL + the postMessage config (round 12)."""
    root = _BACKEND.parent / "frontend" / "src"
    ap = (root / "modules" / "workspace" / "AppPreview.tsx").read_text()
    wb = (root / "modules" / "workspace" / "WorkspaceBuilder.tsx").read_text()
    # iframe URL is gated on the fetched previewToken, no getAuthToken() fallback
    assert "appsApi.previewToken(id)" in ap
    assert "token=${encodeURIComponent(previewToken)}" in ap
    assert "token: previewToken," in ap  # postMessage config uses scoped token
    assert "token: getAuthToken()," not in ap  # the account-JWT push is gone
    assert "previewToken" in wb and "getAuthToken()" not in previewwb_url(wb)


def previewwb_url(wb: str) -> str:
    """Isolate the previewWebUrl memo body so the assertion only covers it (the
    file still uses getAuthToken for the trusted builder WS)."""
    m = re.search(r"const previewWebUrl = useMemo\(\(\) => \{.*?\}, \[", wb, re.S)
    return m.group(0) if m else wb


# ── Round 12: one IAP purchase is redeemable on exactly ONE account ──────

def test_iap_grant_reserves_transaction_globally_before_crediting():
    """HIGH: the consumable grant must reserve the transaction id in the
    globally-unique table BEFORE crediting, so a farmed second account can't
    replay the same real purchase (per-user idempotency is insufficient)."""
    src = (_BACKEND / "app" / "services" / "credit_service.py").read_text()
    # Scope to grant_purchased's body — other functions also mutate the balance.
    m = re.search(r"async def grant_purchased\(.*?(?=\n    async def |\Z)", src, re.S)
    assert m, "grant_purchased not found"
    body = m.group(0)
    assert "INSERT INTO redeemed_iap_transactions" in body
    assert "global_txn_id" in body
    # The reserve happens before the balance mutation (order matters).
    assert body.index("INSERT INTO redeemed_iap_transactions") < body.index(
        "balance.purchased_credits_remaining = _q(")
    # A cross-account collision refuses the grant, it does not credit.
    assert "cross-account IAP replay refused" in body


def test_apple_verify_passes_global_txn_id():
    """The Apple consumable verify endpoint must pass the raw transaction id as
    the global replay key (not only the per-user idempotency key)."""
    src = (_BACKEND / "app" / "api" / "iap.py").read_text()
    assert "global_txn_id=verified.transaction_id" in src


def test_redeemed_iap_table_is_globally_unique_and_self_healed():
    """The guard table must exist with a GLOBAL primary key on transaction_id,
    created by both init_db self-heal and alembic 074."""
    db = (_BACKEND / "app" / "db" / "database.py").read_text()
    assert "CREATE TABLE IF NOT EXISTS redeemed_iap_transactions" in db
    assert "transaction_id VARCHAR(120) PRIMARY KEY" in db
    mig = _BACKEND / "alembic" / "versions" / \
        "20260723_0074_074_redeemed_iap_transactions.py"
    assert mig.exists(), "migration 074 missing"
    mtext = mig.read_text()
    assert 'down_revision = "073"' in mtext  # linear chain, no collision
    assert '"transaction_id", sa.String(120), primary_key=True' in mtext


# ── Round 12: free-tier image cap is TOCTOU-safe (reserve before generate) ──

def test_image_cap_reserves_before_generate_at_every_route():
    """MED: EVERY image route must RESERVE a free-tier slot before generating
    (not check-then-generate-then-charge), and settle/release it. Also fixes the
    prior instance-attr call `credit_service.free_tier_image_quota` that raised
    AttributeError (the module-level fn was never bound to the instance).

    Counted dynamically rather than pinned to a literal: this assertion was
    pinned to ``== 3`` in round 12 and silently went red on main when the async
    Kie start/poll route (#324) added a fourth image entry point. The invariant
    is "every route that starts a render reserves first", not "there are exactly
    three routes"."""
    src = (_BACKEND / "app" / "api" / "llm_proxy.py").read_text()
    # the broken instance-attribute gate is gone
    assert "free_tier_image_quota(db" not in src
    n_reserve = src.count("reserve_free_image_slot(db, config.user_id)")
    assert n_reserve >= 3, "an image route stopped reserving the free-tier slot"
    # every reserving route must be able to give the slot back
    assert src.count("release_free_image_slot(db, ") >= n_reserve
    # the synchronous routes settle inline; the async one settles from /poll
    assert src.count("settle_free_image_slot(db, ") >= 3


def test_reserve_free_image_slot_is_advisory_locked_and_counts_pending():
    """The reserve path must serialize per-user (advisory lock) and count open,
    unexpired reservations so concurrent requests can't race the cap."""
    src = (_BACKEND / "app" / "services" / "credit_service.py").read_text()
    m = re.search(r"async def reserve_free_image_slot\(.*?async def release_free_image_slot",
                  src, re.S)
    assert m, "reserve_free_image_slot not found"
    body = m.group(0)
    assert "pg_advisory_xact_lock" in body
    assert "RESERVATION_OPEN" in body and "expires_at > datetime.utcnow()" in body
    # reservation is committed (published) so concurrent requests see it
    assert "await db.commit()" in body


# ── Round 13: the scoped-preview-token fix reaches EVERY preview surface ──

def test_no_preview_surface_embeds_the_account_jwt():
    """HIGH (round-12 miss): round 12 scoped the preview token in AppPreview +
    WorkspaceBuilder but MISSED AppSplit — the default-ON preview route — which
    still put the full account JWT in the iframe URL and postMessage'd it into
    agent-authored HTML. Assert NO preview surface uses getAuthToken()."""
    root = _BACKEND.parent / "frontend" / "src"
    surfaces = [
        root / "pages" / "AppSplit.tsx",
        root / "modules" / "workspace" / "AppPreview.tsx",
        root / "modules" / "workspace" / "WorkspaceBuilder.tsx",
    ]
    for path in surfaces:
        assert path.exists(), f"preview surface missing: {path}"
        src = path.read_text()
        # the app-scoped token is fetched and used
        assert "previewToken" in src, f"{path.name} does not use a scoped preview token"
        # the account JWT never reaches the preview frame
        assert "token=${token}" not in src, f"{path.name} still puts a raw token in the URL"
        assert "token: getAuthToken()" not in src, f"{path.name} still postMessages the account JWT"


def test_appsplit_holds_frame_until_scoped_token_and_targets_origin():
    """AppSplit must gate the iframe on the scoped token (no account-JWT
    fallback) and must not postMessage the config to a wildcard origin."""
    src = (_BACKEND.parent / "frontend" / "src" / "pages" / "AppSplit.tsx").read_text()
    assert "appsApi.previewToken(id)" in src
    assert "if (!previewToken) return null;" in src
    assert "token: previewToken," in src
    assert "}, '*');" not in src, "agent config still posted to a wildcard target origin"


# ── Round 13: an async Kie render is always paid for ──────────────────

def test_kie_start_holds_credits_so_abandoned_renders_are_billed():
    """MED: /kie/image/start began a real (billable) render while the only
    charge lived in /poll, so a job that was never polled to completion was
    never billed. start must take a credit hold keyed on the task."""
    src = (_BACKEND / "app" / "api" / "llm_proxy.py").read_text()
    m = re.search(r"async def proxy_kie_image_start\(.*?(?=\n@router)", src, re.S)
    assert m, "proxy_kie_image_start not found"
    body = m.group(0)
    assert "credit_service.reserve(" in body, "start takes no credit hold"
    assert 'idempotency_key=f"kie_task:{task_id}"' in body
    # the hold is taken only once the render actually started (after start_task)
    assert body.index("kie_client.start_task") < body.index("credit_service.reserve(")


def test_kie_poll_settles_or_refunds_the_hold_without_double_billing():
    """/poll must settle the hold on success and refund it on failure, finding
    it by (user, task) so no client change is needed — and must not also
    try_charge when a hold exists (that would double-bill)."""
    src = (_BACKEND / "app" / "api" / "llm_proxy.py").read_text()
    m = re.search(r"async def proxy_kie_image_poll\(.*?(?=\n@router)", src, re.S)
    assert m, "proxy_kie_image_poll not found"
    body = m.group(0)
    assert "find_open_reservation_by_key(db, config.user_id, charge_key)" in body
    assert "credit_service.settle(" in body
    assert 'reason="kie_render_failed"' in body, "failed render does not refund the hold"
    # try_charge survives ONLY as the else-branch fallback for pre-deploy jobs
    assert "else:" in body and "credit_service.try_charge(" in body
    assert body.index("credit_service.settle(") < body.index("credit_service.try_charge(")


def test_reservation_lookup_is_user_scoped():
    """The hold lookup must be scoped to the user, so one tenant can never
    settle or refund another tenant's reservation."""
    src = (_BACKEND / "app" / "services" / "credit_service.py").read_text()
    m = re.search(r"async def find_open_reservation_by_key\(.*?return res\.id", src, re.S)
    assert m, "find_open_reservation_by_key not found"
    body = m.group(0)
    assert "CreditReservation.user_id == user_id" in body
    assert "CreditReservation.idempotency_key == idempotency_key" in body
    assert "CreditReservation.status == RESERVATION_OPEN" in body
