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

import asyncio
import logging
import os
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Header, HTTPException, Request

from app.services import drain_state, runtime_identity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

# Strong references for fire-and-forget tasks. `asyncio.create_task` keeps only
# a WEAK reference, so a task nobody holds can be garbage-collected while it is
# still running.
_BACKGROUND_TASKS: set = set()


POOL_ADMIN_TOKEN_ENV = "POOL_ADMIN_TOKEN"

#: Serialises the identity-mutation section of `/admin/bind`. Rebuilt when the
#: running loop changes: an `asyncio.Lock` binds to the loop it is first
#: awaited on, and a lock held against a dead loop is a permanent wedge.
_BIND_LOCK: "tuple[Any, asyncio.Lock] | None" = None


def _bind_lock() -> asyncio.Lock:
    global _BIND_LOCK
    loop = asyncio.get_running_loop()
    if _BIND_LOCK is None or _BIND_LOCK[0] is not loop:
        _BIND_LOCK = (loop, asyncio.Lock())
    return _BIND_LOCK[1]


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
    # The platform's copy of `users.timezone`. NULL-FILL ONLY on this side —
    # db/models/base.py declares the TENANT row authoritative and ws_chat is
    # the high-frequency writer, so a bind that overwrote it would fight the
    # client on every refresh-config push. Deliberately NOT in
    # runtime_identity._PAYLOAD_TO_SETTING: Settings has no `timezone` field,
    # and registering one would log a warning per bind and pollute
    # os.environ['USER_TIMEZONE'].
    "user_timezone",
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

    # Steps 1-2 are the IDENTITY MUTATION, and they must land as a unit:
    # settings, the per-process caches keyed off them, and runtime.json all
    # describe one tenant, and a reader that sees half of that sees a
    # container bound to nobody (503s every route) or to a mix. Concurrent
    # binds are the norm — both Railway replicas call /v1/pool/refresh-config,
    # observed 25 ms apart twice on pool-81 — so say it in code rather than
    # relying on the fact that nothing here currently awaits.
    #
    # Deliberately NOT held across the rest of the handler: step 2d's MCP
    # bootstrap makes a network call, and a lock spanning it would let one
    # slow platform round-trip wedge every future bind on this container.
    async with _bind_lock():
        # 1. Mutate settings so legacy `settings.<field>` reads see the
        #    new identity. This is the alternative to a 141-file refactor
        #    of `settings.user_id` -> `runtime_identity.get_user_id()`.
        try:
            applied = runtime_identity.apply_to_settings(filtered)
            logger.info("[admin/bind] Applied %d fields to settings", applied)
        except Exception as e:
            logger.exception("[admin/bind] apply_to_settings failed")
            raise HTTPException(status_code=500, detail=f"Settings apply failed: {e}") from e

        # 1b. Identity just changed — drop any per-process smart_fetch caches so a
        #     re-bound pool container can never serve the previous tenant's cached
        #     search results or fetched pages (defense in depth for tenant isolation).
        try:
            from app.agent.smart_fetch import clear_caches
            clear_caches()
        except Exception:
            logger.warning("[admin/bind] smart_fetch cache clear failed", exc_info=True)

        # 2. Write runtime.json (durable). After this returns, every
        #    runtime_identity.is_bound() check returns True and the lobby
        #    middleware lets traffic through.
        try:
            runtime_identity.write_runtime(filtered)
        except Exception as e:
            logger.exception("[admin/bind] runtime.json write failed")
            raise HTTPException(status_code=500, detail=f"Runtime write failed: {e}") from e

        # Bounded agent pool (R46/L8): connections opened under the PREVIOUS
        # identity must not serve the new tenant. No-op on NullPool; never
        # fails the bind.
        try:
            from app.db.database import dispose_engine
            await dispose_engine(reason="admin_bind")
        except Exception as _de:  # noqa: BLE001
            logger.warning("[admin/bind] engine dispose skipped: %s", type(_de).__name__)

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
    # A tz this tzdata cannot load is worse than none — it would be stored and
    # then raise in every date calculation that reads it. And 'UTC' is never
    # stored: ws_chat's rebucket gate reads a stored 'UTC' as a real prior
    # zone, so writing it would permanently disable the repair path for the
    # exact users who need it.
    _bind_tz = (filtered.get("user_timezone") or "").strip()
    if _bind_tz:
        if _bind_tz.upper() == "UTC":
            _bind_tz = ""
        else:
            try:
                ZoneInfo(_bind_tz)
            except Exception:
                logger.warning(
                    "[admin/bind] ignoring unresolvable timezone %r", _bind_tz[:64]
                )
                _bind_tz = ""
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
                    timezone=_bind_tz or None,
                ))
                await _udb.commit()
                # The user's real name used to be written here verbatim
                # (round 46 A15). Whether a name ARRIVED is the operational
                # question; which name it was is not.
                logger.info(
                    "[admin/bind] Owner user row created for %s (name=%s)",
                    user_id[:8], "set" if real_user_name else "none",
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
                # NULL-FILL ONLY. A tenant that already knows its zone knows
                # it better than the platform does.
                if _bind_tz and not existing.timezone:
                    existing.timezone = _bind_tz
                    changed = True
                if changed:
                    await _udb.commit()
                    logger.info(
                        "[admin/bind] Owner user row backfilled for %s (name=%s)",
                        user_id[:8], "set" if real_user_name else "none",
                    )
        if _bind_tz:
            # A tz change can move the user's local_date, so the (user_id,
            # local_date) day-chat cache has to go with the tz cache — a turn
            # already in flight would otherwise keep resolving the pre-fill
            # bucket for the full 300 s TTL. This is the `_with_day_chat`
            # variant, which until now had zero callers repo-wide.
            from app.agent._user_tz_cache import (
                invalidate_cached_user_tz_with_day_chat,
            )

            invalidate_cached_user_tz_with_day_chat(user_id)
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
    #       only the current owner's chosen name; the user's Soul choice
    #       arrives via /api/soul/sync. Savepoint-guarded because very old
    #       tenant DBs may predate the agent_configs table.
    #
    #       A BIND MAY NOT CLEAR A NAME. This block used to read the name out
    #       of `filtered` and write the result unconditionally, so a payload
    #       that said nothing about the name NULLed whatever the user had
    #       chosen. The obvious reading of that — "every bind erases the name"
    #       — is FALSE and was checked: the bridge does carry `agent_name`
    #       when the platform holds one (pool-17's runtime.json reads 'Aria'
    #       and its tenant row still did after binds on 10 Sep), and
    #       f261b564's NULL was inherited from a platform column that was
    #       itself NULL because his Soul save rolled back.
    #
    #       The real defect is narrower: the platform's column is TRANSIENTLY
    #       empty around a failed or racing Soul save, and both Railway
    #       replicas push `refresh-config` on their own schedule, so a bind
    #       lands in that window and copies the blank down over a name the
    #       agent already had. So an empty value — absent, null or "" alike —
    #       now means "the platform has nothing to tell me", never "clear it".
    #       `POST /api/soul` is the only writer that may clear this column,
    #       on both sides, which is the same rule the platform follows.
    #
    #       No isolation risk: the row is selected by `user_id`, so a slot
    #       re-bound to a DIFFERENT user finds no row and starts clean. That,
    #       not the unconditional write, is what stops a leftover name leaking.
    try:
        from app.db.database import async_session_maker as _sm2
        from app.db.models import AgentConfig as _AgentConfig
        from sqlalchemy import select

        def _bound_identity(key: str) -> Optional[str]:
            raw = payload.get(key, None)
            return (raw.strip() or None) if isinstance(raw, str) else None

        _bound_agent_name = _bound_identity("agent_name")
        _bound_agent_color = _bound_identity("agent_color")
        _skipped_blank_name = False
        async with _sm2() as _adb:
            async with _adb.begin_nested():
                _ac = (await _adb.execute(
                    select(_AgentConfig).where(_AgentConfig.user_id == user_id)
                )).scalar_one_or_none()
                if _ac is not None:
                    if _bound_agent_name and _ac.agent_name != _bound_agent_name:
                        _ac.agent_name = _bound_agent_name
                    elif not _bound_agent_name and _ac.agent_name:
                        _skipped_blank_name = True
                    if _bound_agent_color and _ac.agent_color != _bound_agent_color:
                        _ac.agent_color = _bound_agent_color
                elif _bound_agent_name or _bound_agent_color:
                    _adb.add(_AgentConfig(
                        user_id=user_id,
                        agent_name=_bound_agent_name,
                        agent_color=_bound_agent_color,
                    ))
            await _adb.commit()
            if _skipped_blank_name:
                # Worth a line: it means the platform's column was empty while
                # this tenant had a name, which is the Soul-save race.
                logger.info(
                    "[admin/bind] Kept existing agent_name for %s — bind carried none",
                    user_id[:8],
                )
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

    # 2c-1. Warm the LLM WIRE (round 46, A1). The refresh above deliberately
    #     defers the client rebuild to "the next chat call" — and on a freshly
    #     claimed container that call is the user's FIRST MESSAGE. On pool-82
    #     (2026-09-15) the rebuild landed 111 s after the key refresh, inside
    #     the turn, and `[OPENAI] Client rebuilt` is the last line before a
    #     9.47 s whole-process freeze.
    #
    #     The SDK's once-per-process lazy work is what runs there: the
    #     `client.responses` resource import (134-141 ms measured) and the
    #     first pydantic parse of a `response.completed` (202-214 ms, ~400x
    #     its warm cost — core-schema construction). Both hold the import
    #     lock / GIL, so they must be paid on a WORKER THREAD, and a bind
    #     must never wait for or fail on them.
    try:
        import asyncio as _asyncio

        async def _warm_llm_wire() -> None:
            try:
                from app.services.openai_agent_service import get_openai_agent_service
                from app.services.anthropic_service import get_anthropic_service

                oai_ms = await _asyncio.to_thread(get_openai_agent_service().warm)
                ant_ms = await _asyncio.to_thread(get_anthropic_service().warm)
                logger.info(
                    "[PERF] llm_wire_warm_ms=%d openai_ms=%d anthropic_ms=%d",
                    int(oai_ms + ant_ms), int(oai_ms), int(ant_ms),
                )
            except Exception as _e:
                logger.warning("[admin/bind] LLM wire warm failed (non-fatal): %s", _e)

        # Keep a reference: asyncio documents a task with no strong reference
        # as collectable MID-EXECUTION, and this task is the only producer of
        # `llm_wire_warm` (a `turn_ready_detail` diagnostic — never a term of
        # `serving`, D1) and the only thing paying the SDK's lazy work off the
        # first turn.
        _warm_task = _asyncio.get_running_loop().create_task(_warm_llm_wire())
        _BACKGROUND_TASKS.add(_warm_task)
        _warm_task.add_done_callback(_BACKGROUND_TASKS.discard)
    except Exception as e:
        logger.warning("[admin/bind] LLM wire warm not scheduled (non-fatal): %s", e)

    # 2c-2. Reset the embedding-provider cache. The agent_main boot
    #     pre-load resolved the embedding provider while this container
    #     was still in LOBBY mode (llm_mode="manual", no toup_token), so
    #     the EmbeddingService singleton cached "local" — and bind always
    #     loses that race because the bridge only calls /admin/bind after
    #     lobby health reports ready (end of boot). This bind just flipped
    #     llm_mode/toup_token on the live settings; without the reset the
    #     provider stays "local" for the whole process lifetime and every
    #     memory write on a freshly claimed / blue-green-recreated pool
    #     container degrades to an unembedded row (dedup off, vector
    #     recall blind — the silent follow-up to canary 533354ce). Reset →
    #     the next embed re-resolves and lands on the proxy client.
    try:
        from app.services.embedding_service import EmbeddingService as _EmbSvc
        _EmbSvc.reset_provider_cache()
        logger.info("[admin/bind] Embedding provider cache reset")
    except Exception as e:
        logger.warning("[admin/bind] Embedding provider reset failed (non-fatal): %s", e)

    # 2d. Bootstrap the MCP connector-tools client. A pool container
    #     boots in lobby mode without agent_api_key, so the lifespan's
    #     MCP init was a no-op — this bind just supplied the key.
    #     Without this call the container never registers a single
    #     connector tool (gmail__*, calendar__*, …): users see the
    #     integration "Connected" while the agent, offered no gmail__
    #     tools, improvises with the browser-extension tools and tells
    #     them to pair the desktop Chrome extension. Deferred refresh
    #     keeps bind fast; the 60s periodic loop is the retry net.
    try:
        from app.agent.mcp_bootstrap import ensure_mcp_initialized
        mcp_status = await ensure_mcp_initialized(
            request.app, defer_initial_refresh=True
        )
        logger.info("[admin/bind] MCP bootstrap: %s", mcp_status)
    except Exception as e:
        logger.warning("[admin/bind] MCP bootstrap failed (non-fatal): %s", e)

    # 3. Wake any lazy channel adapters. Best-effort — adapter init
    #    can take seconds (Telegram getMe, WhatsApp QR pairing) and we
    #    want /admin/bind to return fast. The wake_lazy_channels module
    #    may not be present in early-Phase-A builds; treat as no-op.
    try:
        from app.services.channel_init import wake_lazy_channels  # type: ignore
        wake_lazy_channels()
    except Exception as e:
        logger.info("[admin/bind] channel_init not available yet: %s", e)

    # 4. Warm the verify browser. Boot only does this for an already-bound
    #    container now, because a resident Brave in an unclaimed lobby spare
    #    is ~104 MiB PSS for a build nobody can request. This is the other
    #    half: the moment a container HAS a user, warm it — still earlier
    #    than the app-build that needs it. Idempotent, which is required
    #    rather than merely tidy: `refresh-config` is routine and both
    #    Railway replicas push one.
    try:
        from app.agent.skills.builtins.app_html.verify import schedule_warm_browser
        if schedule_warm_browser("bind"):
            logger.info("[admin/bind] Verify browser warm-up scheduled")
    except Exception as e:
        logger.info("[admin/bind] browser warm-up not scheduled: %s", e)

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


# ── /admin/whatsapp-logout ─────────────────────────────────────────


@router.post("/whatsapp-logout")
async def admin_whatsapp_logout(
    x_pool_admin_token: Optional[str] = Header(None, alias="X-Pool-Admin-Token"),
) -> Dict[str, Any]:
    """Log the tenant's WhatsApp device out and wipe the on-disk session.

    R46 F9. The bridge needs this on a USER RELEASE — deleting a Toup account
    used to leave the linked device sitting in the user's WhatsApp forever,
    and the pool's workspace save/restore is a `docker cp` MERGE, so
    `.whatsapp_auth` could reach the next tenant of the slot. The capability
    already existed (`force_logout` → sidecar `/pair/logout` → `sock.logout()`
    + `wipeAuthDir()`); its only caller was the user's own "Disconnect"
    button, which nothing calls on deletion.

    Pool-admin token, not a user JWT: the bridge has no JWT, and by the time
    a slot is released the user may already be gone. Idempotent — an agent
    with no WhatsApp adapter reports `logged_out: false` and 200, because a
    teardown must never fail on a channel that was never up.
    """
    _check_admin_token(x_pool_admin_token)
    try:
        from app.agent.channels.whatsapp_baileys import get_active_baileys_channel
        channel = get_active_baileys_channel()
    except Exception:
        channel = None
    if channel is None:
        return {"ok": True, "logged_out": False, "reason": "no_active_channel"}
    try:
        await channel.force_logout()
    except Exception as exc:
        logger.warning("[admin] whatsapp logout failed: %s", type(exc).__name__)
        return {"ok": False, "logged_out": False, "reason": "logout_failed"}
    logger.info("[admin] whatsapp logged out on release")
    return {"ok": True, "logged_out": True}


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
