"""
Runtime identity — late-bound USER_ID + tenant credentials.

Phase A (never-sleep plan) replaces env-time `settings.user_id` reads
with a runtime-resolved getter so a single agent container image can
serve any user once it's bound. The bridge starts containers with
`TOUP_POOL_GENERIC=1` (lobby mode); on `POST /admin/bind`, the agent
writes the tenant payload to `/etc/toup-agent/runtime.json` and every
subsequent `get_user_id()` call returns the bound id.

Three modes:

1. **Legacy env-mode** (`TOUP_POOL_GENERIC` unset, `USER_ID` env present):
   `get_user_id()` returns `os.environ['USER_ID']` (cached). Existing
   tenants — whose containers were created with `--env USER_ID=...` —
   continue to work without touching `/etc/toup-agent/runtime.json`.

2. **Pool-bound mode** (`TOUP_POOL_GENERIC=1`, runtime.json present):
   Reads from `/etc/toup-agent/runtime.json`. Bind-time atomic write
   (write-tmp + rename) means a partial-write never leaks a half-bound
   container.

3. **Pool-lobby mode** (`TOUP_POOL_GENERIC=1`, runtime.json absent):
   `get_user_id()` returns None. Lobby endpoints check this and 503
   any non-admin route. The bridge holds the container in this mode
   until claim.

The functions here are the SINGLE SOURCE OF TRUTH for tenant identity
inside the agent. Code that imports `settings.user_id` directly will be
caught by the Phase A.3 grep gate and rewritten.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RUNTIME_PATH = Path("/etc/toup-agent/runtime.json")
POOL_GENERIC_ENV = "TOUP_POOL_GENERIC"

# Process-local cache. Read from disk on first access, then on every
# explicit reload (post-bind) the loader bumps `_cache_token` so any
# memoized `_cached_*` values invalidate.
_lock = threading.Lock()
_runtime: Optional[Dict[str, Any]] = None
_runtime_loaded = False
_cache_token: int = 0


def _load_from_disk() -> Optional[Dict[str, Any]]:
    """Load /etc/toup-agent/runtime.json. None if absent or malformed.

    Soft-fails: malformed JSON logs a warning and returns None, which
    pushes the agent into lobby mode. The bridge will retry the bind.
    Crashing the agent here would kill any in-flight users on the same
    container during a hot-bind retry."""
    if not RUNTIME_PATH.exists():
        return None
    try:
        return json.loads(RUNTIME_PATH.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[runtime_identity] Failed to read %s: %s", RUNTIME_PATH, e)
        return None


def _ensure_loaded() -> None:
    global _runtime, _runtime_loaded
    if _runtime_loaded:
        return
    with _lock:
        if _runtime_loaded:
            return
        _runtime = _load_from_disk()
        _runtime_loaded = True


def is_pool_generic() -> bool:
    """True when this container was started in pool-lobby mode.

    Pool-lobby mode is set by the bridge via `TOUP_POOL_GENERIC=1` in
    docker run env. Existing legacy tenants don't have this var and
    continue to operate exactly as today.
    """
    val = os.environ.get(POOL_GENERIC_ENV, "").strip().lower()
    return val in {"1", "true", "yes"}


def is_bound() -> bool:
    """True when the container has tenant identity (env or pool-bind)."""
    return get_user_id() is not None and bool(get_user_id())


def reload() -> None:
    """Force a fresh disk read. Called by `/admin/bind` after writing
    runtime.json so subsequent `get_*` calls see the new tenant.

    Bumps `_cache_token` so any module that wants to cheaply observe
    rebinds (e.g., LLM client builders that key per-tenant) can compare
    tokens cheaply."""
    global _runtime, _runtime_loaded, _cache_token
    with _lock:
        _runtime = _load_from_disk()
        _runtime_loaded = True
        _cache_token += 1
    logger.info("[runtime_identity] Reloaded — token=%d, bound=%s", _cache_token, is_bound())


def cache_token() -> int:
    """Monotonic counter that increments on every `reload()`. Useful for
    callers that maintain per-tenant memoized state."""
    return _cache_token


def restore_at_boot() -> int:
    """Re-apply the persisted bind payload (runtime.json) to the live
    settings instance. Called ONCE from the agent lifespan, before any
    traffic is served.

    This closes the keyless-restart class (2026-07-04): `/admin/bind`
    writes the full payload to runtime.json AND mutates `settings`, but
    the settings mutation is process-local — a container restart (host
    reboot, docker restart, OOM kill) reverts every `settings.X` read
    (API-key middleware, session-JWT verification, LLM keys, TOUP_TOKEN)
    to the spawn-time env values while runtime.json still holds the
    authoritative bind. The agent then looks healthy and bound on
    /agent/health while rejecting ALL chat auth. Re-applying at boot
    makes bind state genuinely restart-proof instead of relying on the
    platform's 180s reclaim sweep to notice and force a re-bind.

    Returns the number of fields applied; 0 when there is no persisted
    bind (fresh generic lobby container, or env-mode tenant whose
    identity comes from .env and needs no restore).
    """
    _ensure_loaded()
    with _lock:
        payload = dict(_runtime or {})
    if not payload.get("user_id"):
        return 0
    applied = apply_to_settings(payload)
    logger.info(
        "[runtime_identity] Boot restore: re-applied %d bind fields for user=%s",
        applied, str(payload.get("user_id"))[:8],
    )
    return applied


def write_runtime(payload: Dict[str, Any]) -> None:
    """Atomically write the bind payload to disk and reload.

    Caller (the `/admin/bind` handler) is responsible for validating the
    payload shape. We only enforce: must be a dict, must contain
    `user_id`. The atomic write (tmp + rename) is mandatory because the
    file is read by every other module on get_*; a half-written file
    during a re-bind would silently corrupt identity.
    """
    if not isinstance(payload, dict) or not payload.get("user_id"):
        raise ValueError("write_runtime: payload missing user_id")
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RUNTIME_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, RUNTIME_PATH)
    reload()


# ── Per-field accessors ──────────────────────────────────────────────


def _get(key: str) -> Optional[str]:
    """Resolve `key` from runtime.json first, else from env (uppercased)."""
    _ensure_loaded()
    if _runtime is not None:
        v = _runtime.get(key)
        if v is not None and v != "":
            return str(v)
    env_val = os.environ.get(key.upper(), "")
    return env_val if env_val else None


def get_user_id() -> Optional[str]:
    """Owner user id for this container.

    None means lobby mode (pool container, not yet bound). Callers that
    need a guaranteed-bound id should check `is_bound()` or rely on the
    lobby-mode middleware to 503 the request before they're called."""
    return _get("user_id")


def get_agent_api_key() -> Optional[str]:
    """The X-Agent-Key the platform sends. Used by the API key
    middleware to authenticate platform → agent calls."""
    return _get("agent_api_key")


def get_database_url() -> Optional[str]:
    """Per-tenant DATABASE_URL. In pool-bound mode this is the renamed
    DB (`postgresql+asyncpg://.../toup_agent_<prefix>`). In env-mode
    it's whatever cloud-init wrote."""
    return _get("database_url")


def get_runtime_field(name: str) -> Optional[str]:
    """Generic accessor for non-core fields (channel tokens, agent_color,
    agent_name, llm_mode, etc.). Same env-fallback semantics."""
    return _get(name)


def all_runtime_fields() -> Dict[str, Any]:
    """Return a snapshot of the full runtime payload for diagnostics.

    Drops obvious secrets when serializing for /agent/health. The
    health endpoint should call `redact_secrets(all_runtime_fields())`
    rather than this directly."""
    _ensure_loaded()
    return dict(_runtime or {})


_SECRET_KEY_TOKENS = (
    "key", "token", "secret", "password", "pwd", "credential",
)


# ── Mutation of the pydantic Settings instance ──────────────────────
# Pre-Phase-A approach was to refactor every `settings.user_id` call
# site to `runtime_identity.get_user_id()`. That's ~80 sites, mostly
# in agent code paths, and a mechanical pass risked silently dropping
# a reference. Instead we mutate the live `settings` instance so the
# existing reads continue to resolve correctly. Pydantic v2
# BaseSettings is mutable by default; we use `object.__setattr__` to
# bypass any validators that would reject mid-process-life mutation.

# Map of bind-payload key -> Settings attribute name. When a new
# bridge field is added that needs to land in settings, append here.
# Keys NOT in this map (channel-specific tokens stored only in
# runtime.json, e.g.) remain accessible via `get_runtime_field()`.
_PAYLOAD_TO_SETTING = {
    "user_id": "user_id",
    "agent_api_key": "agent_api_key",
    "database_url": "database_url",
    "agent_color": "agent_color",
    "agent_name": "agent_name",
    "llm_mode": "llm_mode",
    "openai_api_key": "openai_api_key",
    "anthropic_api_key": "anthropic_api_key",
    "google_api_key": "google_api_key",
    "mistral_api_key": "mistral_api_key",
    "groq_api_key": "groq_api_key",
    "xai_api_key": "xai_api_key",
    "deepseek_api_key": "deepseek_api_key",
    "agent_model": "agent_model",
    "telegram_bot_token": "telegram_bot_token",
    "discord_bot_token": "discord_bot_token",
    "slack_bot_token": "slack_bot_token",
    "slack_app_token": "slack_app_token",
    "whatsapp_phone_number_id": "whatsapp_phone_number_id",
    "whatsapp_access_token": "whatsapp_access_token",
    "whatsapp_verify_token": "whatsapp_verify_token",
    "whatsapp_app_secret": "whatsapp_app_secret",
    "whatsapp_mode": "whatsapp_mode",
    # Baileys ACL — restart_whatsapp_channel reads these from settings
    # to filter inbound messages. Missing here meant pool-bound users'
    # inbound WhatsApp messages were silently dropped by the empty
    # allowlist even when /admin/bind successfully forwarded them.
    "whatsapp_baileys_allowlist": "whatsapp_baileys_allowlist",
    "whatsapp_self_e164": "whatsapp_self_e164",
    "whatsapp_session_status": "whatsapp_session_status",
    "connect_token": "toup_token",
    "supabase_url": "supabase_url",
    "supabase_anon_key": "supabase_anon_key",
}


def apply_to_settings(payload: Dict[str, Any]) -> int:
    """Mutate the live `app.config.settings` instance from a bind payload.

    Returns the number of fields successfully written. Unknown payload
    keys are silently ignored (defensive against future bridge
    changes leaking unexpected data — no surprise side effects).

    Each field is set via `object.__setattr__` so a pydantic validator
    that rejects post-init mutation doesn't crash the bind. If a field
    is missing from settings (rare; usually means a config refactor),
    we log a warning and continue — the bind still succeeds, just with
    one less field applied. Caller can compare the return count with
    `len(_PAYLOAD_TO_SETTING)` to detect drift.

    ALSO sets the corresponding env var (uppercase). Modules that read
    `os.environ` directly (a handful of places) see the new value.
    Existing process objects that cached env at import time are NOT
    updated — those need lazy init or a hot-restart path.
    """
    from app.config import settings as _settings
    written = 0
    for payload_key, attr in _PAYLOAD_TO_SETTING.items():
        if payload_key not in payload:
            continue
        value = payload[payload_key]
        if value is None:
            continue
        try:
            object.__setattr__(_settings, attr, value)
            os.environ[payload_key.upper()] = str(value)
            written += 1
        except Exception as e:
            logger.warning(
                "[runtime_identity] Could not apply %s -> settings.%s: %s",
                payload_key, attr, e,
            )
    return written


def redact_secrets(d: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort secret redaction for diagnostic outputs.

    Any key whose name (case-insensitive) contains one of the tokens
    above is replaced with a length tag. Conservative on purpose —
    false positives (a non-secret key that happens to have 'token' in
    its name) are preferable to leaking a real secret in /agent/health.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kl = k.lower()
        if any(t in kl for t in _SECRET_KEY_TOKENS) and isinstance(v, str) and v:
            out[k] = f"<redacted:{len(v)}>"
        else:
            out[k] = v
    return out
