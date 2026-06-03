"""
Pool admin endpoints — `/admin/bind` and `/admin/drain`.

These are the only routes a pool-lobby agent serves besides
`/agent/health`. Both are protected by `X-Pool-Admin-Token` (a static
token held by the bridge), NOT by `X-Agent-Key` — the platform's
agent-key isn't known to a generic pool container yet.

Bind transitions the container from GENERIC to ASSIGNED (specific
user). Drain is the inverse signal for blue-green rollouts: the bridge
calls `/admin/drain` on the slot whose Caddy route just got flipped
away, the slot stops accepting new WS connections, and exits when
in-flight handlers finish (or after the timeout).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request

from app.services import drain_state, runtime_identity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


POOL_ADMIN_TOKEN_ENV = "POOL_ADMIN_TOKEN"


def _check_admin_token(token: Optional[str]) -> None:
    """Validate the pool admin token. 401 on missing/mismatch.

    Constant-time compare to avoid leaking the token via response-time
    side-channels. The token MUST be set via env on every container
    (set by the bridge at `docker run`) — without it the container
    refuses bind, which is the safe default for a misconfigured
    deploy."""
    expected = os.environ.get(POOL_ADMIN_TOKEN_ENV, "").strip()
    if not expected:
        # Fail closed: no token configured = nobody can bind. This is
        # what we want in a misconfigured deploy.
        raise HTTPException(status_code=503, detail="Pool admin token not configured")
    provided = (token or "").strip()
    # Constant-time compare. hmac.compare_digest is the right primitive
    # but we don't import hmac elsewhere in this module — use a manual
    # length-tolerant compare via hmac directly.
    import hmac as _hmac
    if not _hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid pool admin token")


# ── /admin/bind ────────────────────────────────────────────────────


# Bind payload size cap. Channel tokens (Slack app token, WhatsApp
# access token, etc.) plus a base64 connect_token can run a few KB;
# 64KB is generous and bounds the worst case where a misbehaving
# bridge tries to ship MBs of garbage.
_MAX_BIND_BODY_BYTES = 64 * 1024


# Keys we accept from the bridge. Anything else is silently dropped —
# defensive against future bridge changes leaking unexpected data.
_BIND_FIELDS = (
    "user_id",
    "agent_api_key",
    # User-side identity (real human, not the agent persona). Threaded
    # through so lazy-create paths in the agent's local DB stop stubbing
    # `name='Agent Owner'` / `email='<prefix>@agent.local'`. Without
    # these, the user's first-message greeting was generic forever.
    "user_name",
    "user_email",
    "agent_color",
    "agent_name",
    "llm_mode",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "mistral_api_key",
    "groq_api_key",
    "xai_api_key",
    "deepseek_api_key",
    "agent_model",
    "telegram_bot_token",
    "discord_bot_token",
    "slack_bot_token",
    "slack_app_token",
    "whatsapp_phone_number_id",
    "whatsapp_access_token",
    "whatsapp_verify_token",
    "whatsapp_app_secret",
    "whatsapp_mode",
    # WhatsApp Baileys (QR-link) ACL — without these in the whitelist,
    # the platform's bind payload was silently dropped here and the
    # agent restarted its WhatsApp channel with allowlist_size=0.
    # Result: every inbound WhatsApp message hit lid.allowlist_empty
    # and was ignored, so users saw "linked" in settings but their
    # agent never replied. Pool path only — slow path went through
    # `_build_env` which propagated these fields via env vars.
    "whatsapp_baileys_allowlist",
    "whatsapp_self_e164",
    "whatsapp_session_status",
    "connect_token",
    "supabase_url",
    "supabase_anon_key",
)


@router.post("/bind")
async def admin_bind(
    request: Request,
    x_pool_admin_token: Optional[str] = Header(None, alias="X-Pool-Admin-Token"),
) -> Dict[str, Any]:
    """Bind a generic pool container to a specific tenant.

    Idempotent in the practical sense: if called twice with the same
    payload, the second call replaces the first. In ASSIGNED→ASSIGNED
    the bridge would only call this for re-binds (e.g., recovery from
    a crashed in-progress bind), and it's safe.

    Architecture note — DB stays as-is:
        Generic pool containers boot with `DATABASE_URL` pointing at a
        pre-allocated pool DB (`toup_pool_db_NN`). On bind, the DB name
        does NOT change — the bridge owns the user→pool-slot mapping in
        its own `pool.db`, and the platform's `agent_url` indirection
        keeps the DB name an internal detail. This avoids the 141-file
        late-bind engine refactor.

    Order of operations:
    1. Validate token, size, payload shape.
    2. Mutate `settings` so `settings.user_id`, channel tokens, etc.
       reflect the bound identity (no call-site refactor needed).
    3. Write `/etc/toup-agent/runtime.json` for durability across
       process restarts (bridge re-binds on restart from pool.db).
    4. Wake lazy channel adapters.

    Returns 200 + resolution snapshot so the bridge can confirm.
    """
    _check_admin_token(x_pool_admin_token)

    # Body cap. Reading the raw body avoids relying on Content-Length
    # which a misbehaving client can lie about.
    raw = await request.body()
    if len(raw) > _MAX_BIND_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Bind payload too large")

    import json as _json
    try:
        payload = _json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, _json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    user_id = payload.get("user_id")
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(status_code=400, detail="user_id required")
    agent_api_key = payload.get("agent_api_key")
    if not agent_api_key or not isinstance(agent_api_key, str):
        raise HTTPException(status_code=400, detail="agent_api_key required")

    # Filter to whitelisted fields only.
    filtered = {k: payload[k] for k in _BIND_FIELDS if k in payload and payload[k] is not None}

    logger.info(
        "[admin/bind] Binding container to user_id=%s (fields: %d)",
        str(user_id)[:8], len(filtered),
    )

    # 1. Mutate settings so legacy `settings.<field>` reads see the
    #    new identity. This is the alternative to a 141-file refactor
    #    of `settings.user_id` -> `runtime_identity.get_user_id()`.
    try:
        applied = runtime_identity.apply_to_settings(filtered)
        logger.info("[admin/bind] Applied %d fields to settings", applied)
    except Exception as e:
        logger.exception("[admin/bind] apply_to_settings failed")
        raise HTTPException(status_code=500, detail=f"Settings apply failed: {e}") from e

    # 2. Write runtime.json (durable). After this returns, every
    #    runtime_identity.is_bound() check returns True and the lobby
    #    middleware lets traffic through.
    try:
        runtime_identity.write_runtime(filtered)
    except Exception as e:
        logger.exception("[admin/bind] runtime.json write failed")
        raise HTTPException(status_code=500, detail=f"Runtime write failed: {e}") from e

    # 2b. Ensure owner user row exists in the (now-bound) DB with the
    #     real name + email. Without `user_name`/`user_email` in the
    #     bind payload (legacy callers), falls back to the stub values
    #     that legacy bind used. Existing rows are UPDATED on every
    #     bind so a migration from stub → real name happens automatically
    #     on the next config-refresh, no manual SQL needed.
    #
    #     The agent_main lifespan also creates this row at startup gated
    #     on `settings.user_id`, but `user_id` is empty in lobby mode,
    #     so the row was never created there. Without it, the first
    #     chat WS hits a Foreign Key violation when trying to write the
    #     user's message into a session whose user_id references users.id.
    real_user_name = (filtered.get("user_name") or "").strip() or "Agent Owner"
    real_user_email = (filtered.get("user_email") or "").strip() or f"{user_id[:8]}@agent.local"
    try:
        from app.db.database import async_session_maker as _sm
        from app.services.auth_service import get_user_by_id
        from app.db.models import User
        async with _sm() as _udb:
            existing = await get_user_by_id(_udb, user_id)
            if not existing:
                _udb.add(User(
                    id=user_id,
                    email=real_user_email,
                    hashed_password="",
                    name=real_user_name,
                ))
                await _udb.commit()
                logger.info(
                    "[admin/bind] Owner user row created for %s (name=%r)",
                    user_id[:8], real_user_name,
                )
            else:
                # Backfill the real values if the row was previously
                # stubbed via a lazy-create path (bind path's stub or
                # the WS auth path's stub). Cheap UPDATE — only writes
                # when the value actually changes.
                changed = False
                if existing.name != real_user_name and real_user_name != "Agent Owner":
                    existing.name = real_user_name
                    changed = True
                if existing.email != real_user_email and real_user_email != f"{user_id[:8]}@agent.local":
                    existing.email = real_user_email
                    changed = True
                if changed:
                    await _udb.commit()
                    logger.info(
                        "[admin/bind] Owner user row backfilled for %s (name=%r)",
                        user_id[:8], real_user_name,
                    )
    except Exception as e:
        logger.exception("[admin/bind] Owner-user creation failed (non-fatal)")
        # Don't fail the bind on this — the user row will be created
        # lazily by ws_chat.py's stub-user path on first chat. Logging
        # only so an operator can spot DB-down conditions.

    # 2b-2. Reset the agent IDENTITY in the local AgentConfig row to match
    #       THIS bind. The agent runtime reads its display name solely from
    #       AgentConfig.agent_name (agent_runner._build_system_prompt) and
    #       renders a clean "you don't have a name yet" when it's empty — it
    #       never invents one. So a claimed/refreshed container must carry
    #       only the current owner's chosen name: we write agent_name/
    #       agent_color from the bind payload, and CLEAR a stale agent_name
    #       when the payload doesn't carry one (a fresh signup hasn't picked
    #       a name yet at claim time). This kills the "wrong agent name shows
    #       first, then flips to the chosen one" symptom — a freshly-claimed
    #       container can no longer surface a leftover/leaked name; the user's
    #       Soul choice arrives via /api/soul/sync. Savepoint-guarded because
    #       very old tenant DBs may predate the agent_configs table.
    try:
        from app.db.database import async_session_maker as _sm2
        from app.db.models import AgentConfig as _AgentConfig
        from sqlalchemy import select
        _bound_agent_name = (filtered.get("agent_name") or "").strip() or None
        _bound_agent_color = (filtered.get("agent_color") or "").strip() or None
        async with _sm2() as _adb:
            async with _adb.begin_nested():
                _ac = (await _adb.execute(
                    select(_AgentConfig).where(_AgentConfig.user_id == user_id)
                )).scalar_one_or_none()
                if _ac is not None:
                    # Only overwrite the name (incl. clearing a stale one).
                    # Color is only set when provided (no point clearing it).
                    if _ac.agent_name != _bound_agent_name:
                        _ac.agent_name = _bound_agent_name
                    if _bound_agent_color and _ac.agent_color != _bound_agent_color:
                        _ac.agent_color = _bound_agent_color
                elif _bound_agent_name or _bound_agent_color:
                    _adb.add(_AgentConfig(
                        user_id=user_id,
                        agent_name=_bound_agent_name,
                        agent_color=_bound_agent_color,
                    ))
            await _adb.commit()
            logger.info(
                "[admin/bind] Local AgentConfig identity reset for %s (name=%r)",
                user_id[:8], _bound_agent_name,
            )
    except Exception as e:
        logger.warning("[admin/bind] AgentConfig identity reset failed (non-fatal): %s", e)

    # 2c. Refresh the LLM key cache. OpenAI/Anthropic clients are
    #     constructed once at agent boot and cached on KeyProvider's
    #     version counter; without bumping that counter the LLM
    #     services keep using the (empty) key the lobby agent had at
    #     startup. The agent then sends every chat completion with no
    #     api_key header and OpenAI returns 401 'Incorrect API key
    #     provided: missing'. Calling refresh() reads the freshly-
    #     mutated settings and bumps version → next chat call rebuilds
    #     the client with the user's real key.
    try:
        from app.services.key_provider import keys as _llm_keys
        _llm_keys.refresh()
        logger.info("[admin/bind] LLM key cache refreshed")
    except Exception as e:
        logger.warning("[admin/bind] LLM key refresh failed (non-fatal): %s", e)

    # 3. Wake any lazy channel adapters. Best-effort — adapter init
    #    can take seconds (Telegram getMe, WhatsApp QR pairing) and we
    #    want /admin/bind to return fast. The wake_lazy_channels module
    #    may not be present in early-Phase-A builds; treat as no-op.
    try:
        from app.services.channel_init import wake_lazy_channels  # type: ignore
        wake_lazy_channels()
    except Exception as e:
        logger.info("[admin/bind] channel_init not available yet: %s", e)

    return {
        "ok": True,
        "ready": True,
        "user_id": user_id,
        "fields_applied": len(filtered),
    }


# ── /admin/drain ───────────────────────────────────────────────────


@router.post("/drain")
async def admin_drain(
    request: Request,
    x_pool_admin_token: Optional[str] = Header(None, alias="X-Pool-Admin-Token"),
) -> Dict[str, Any]:
    """Engage drain mode. Used by Phase B blue-green rollouts.

    After this returns, new WS connections close with code 1012 and
    the container exits when active connections drop to zero (or after
    the timeout, whichever first). Caller (bridge) uses this signal to
    coordinate cutover: flip Caddy → call /admin/drain on the old slot
    → docker stop the old slot once it exits.

    Default timeout 60s — same value blessed in the plan.
    """
    _check_admin_token(x_pool_admin_token)

    timeout_s = 60
    try:
        body = await request.body()
        if body:
            import json as _json
            data = _json.loads(body)
            if isinstance(data, dict) and isinstance(data.get("drain_timeout_s"), (int, float)):
                timeout_s = max(1, int(data["drain_timeout_s"]))
    except Exception:
        # Drain ignores body parse failures — the default timeout is
        # always safe.
        pass

    drain_state.set_draining(timeout_s)
    return {"ok": True, **drain_state.status(), "timeout_s": timeout_s}


# ── /admin/status ──────────────────────────────────────────────────


@router.get("/status")
async def admin_status(
    x_pool_admin_token: Optional[str] = Header(None, alias="X-Pool-Admin-Token"),
) -> Dict[str, Any]:
    """Diagnostic: report the container's bind/drain state.

    Bridge polls this to verify drain progress and to check whether a
    just-spawned generic container is ready for claim. Token-protected
    so it can't be enumerated externally to map tenant identities."""
    _check_admin_token(x_pool_admin_token)
    return {
        "is_pool_generic": runtime_identity.is_pool_generic(),
        "is_bound": runtime_identity.is_bound(),
        "user_id": runtime_identity.get_user_id(),
        "cache_token": runtime_identity.cache_token(),
        "drain": drain_state.status(),
        "runtime": runtime_identity.redact_secrets(runtime_identity.all_runtime_fields()),
    }
