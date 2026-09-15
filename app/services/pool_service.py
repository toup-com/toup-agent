"""
Platform-side pool service (Phase A.2 — never-sleep plan).

Calls the bridge's `POST /v1/pool/claim` to acquire a pre-booted
generic agent container for a freshly-registered user. Falls back to
the slow `provision_container` path if the pool is exhausted or the
feature flag is off.

Wires into `app.api.auth.register` — the user signs up, this service
fires fire-and-forget, and by the time the user reaches Welcome the
container is bound + ready.

The key difference from `schedule_prewarm`:

- `schedule_prewarm` calls `provision_container(recreate=False)`,
  which spawns a fresh container from `toup-agent:<image>`.
  Cold-boot takes ~15s end-to-end.
- `claim_for_user` asks the bridge to bind an already-running pool
  container. Wall-clock <1s if a generic is available; ~15s fallback
  if not.

After claim succeeds, the resulting `ManagedContainer` row + AgentConfig
are populated identically to the provision path so existing code
(rollouts, reaper, telemetry) sees no shape difference.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import AgentConfig, ManagedContainer
from app.services.background_tasks import spawn as _spawn_bg

logger = logging.getLogger(__name__)


# Postgres `lock_timeout` fires SQLSTATE 55P03 (lock_not_available). asyncpg
# raises LockNotAvailableError; SQLAlchemy wraps it in an OperationalError with
# the original hanging off `.orig`. Match on the sqlstate first — the class
# names differ across drivers, the sqlstate does not.
_LOCK_TIMEOUT_SQLSTATE = "55P03"


def _is_lock_timeout(exc: BaseException) -> bool:
    """True only for "somebody else holds this lock", never for a DB failure."""
    for e in (exc, getattr(exc, "orig", None)):
        if e is None:
            continue
        if getattr(e, "sqlstate", None) == _LOCK_TIMEOUT_SQLSTATE:
            return True
        if getattr(e, "pgcode", None) == _LOCK_TIMEOUT_SQLSTATE:
            return True
        if "LockNotAvailable" in type(e).__name__:
            return True
    return False


class ProvisionDriveTaken(Exception):
    """Another driver (another replica, or another task on this one) already
    holds this user's claim lock.

    Raised instead of returning None so callers can tell "nobody is doing this,
    fall back to the slow path" from "somebody is doing this, do NOT start a
    second one". The 2026-09-06 incident is what the distinction is for: two
    prewarms 19 ms apart both drove a cold named provision for a user the pool
    was already binding.
    """


class ClaimOutcomeUnknown(ProvisionDriveTaken):
    """The bridge never gave a usable answer, so we do not know what it did.

    A timeout or a 5xx on POST /v1/pool/claim is not "no slot". On 2026-09-06
    the bridge finished that exact bind two seconds after the platform stopped
    listening, and the named create the platform started instead reached the
    host: `docker network tnt_aec1977b` is stamped 18:18:00 with zero
    containers, eleven seconds BEFORE the pool bind completed at 18:18:11. It
    only failed to produce a second container because Postgres was at 300/300.
    Two completed versions of the same race are still on the VPS: prefixes
    667cf3de and 51d4ed2f each have a named container serving them AND a pool
    slot the bridge reports ASSIGNED+bound that serves nobody.

    A SUBCLASS of ProvisionDriveTaken, deliberately, and not a sibling: every
    `except ProvisionDriveTaken` in this tree already means "somebody else owns
    this user's outcome — observe, do not start a second path", which is
    exactly what an unknown outcome demands. A sibling would have had to be
    added to three handlers by hand and the one that got missed would be this
    incident again.

    `ensure_discovery` is always started before this is raised, so something
    IS still watching for the late bind.
    """


# The advisory key `claim_for_user` takes for the duration of a bind. Named
# once so `try_take_claim_drive` and the lock below can never drift apart.
def _claim_drive_key(user_id: str) -> str:
    return f"pool_claim:{user_id}"


async def _release_claim_drive(db: AsyncSession) -> None:
    """Hand `pool_claim:<user>` back before returning from claim_for_user
    without a container.

    An xact-scoped advisory lock lives until the session's next commit or
    rollback, and every early return below this point does neither — so on the
    pool-exhausted path the lock was still held while the caller went on to
    schedule a prewarm, and the prewarm now refuses to drive while somebody
    holds it. Rolling back discards only the uncommitted CAS that mints
    `agent_api_key`; `activate_free_tier` commits its own work, and the key is
    re-minted and re-pushed by the bridge's idempotent claim on the next
    attempt — which is what already happened when the session simply closed.
    """
    try:
        await db.rollback()
    except Exception:
        logger.warning("[pool_service] could not release the claim drive lock")


async def ensure_agent_api_key(user_id: str) -> Optional[str]:
    """CAS-mint the per-tenant `agent_api_key` and COMMIT it, in its own short
    transaction. Returns the committed key, or None if it could not be done.

    Why a separate transaction rather than reordering the caller's commit
    (D-5): the claim path holds `pg_advisory_xact_lock('pool_claim:<uid>')`,
    which is transaction-scoped. Committing the caller's transaction to make
    the key visible would RELEASE that lock before the bridge call, and the
    lock is the thing that makes two near-simultaneous claims produce one bind
    instead of two containers with different keys (the 2026-06-30 and
    2026-07-01 double-bind incidents). A second session commits the key while
    the first keeps the lock.

    Deadlock-proof by construction, not by inspection: every call site reaches
    this with no uncommitted write on the `agent_configs` row (the claim path
    has done only SELECTs, or `activate_free_tier` has already committed), but
    "no caller ever will" is not a property a future edit preserves. So the
    UPDATE runs under a short `lock_timeout` and ANY failure — contention,
    a non-Postgres dialect, a DB blip — answers None and the caller falls back
    to the in-transaction CAS, i.e. to exactly today's behaviour.

    Safe to leave behind on a failed claim: the key is idempotent once set
    (`WHERE agent_api_key IS NULL`), and a committed key with no bind is
    strictly better than the state it replaces — the platform holding NULL
    while the agent already holds K.
    """
    try:
        from app.db.database import async_session_maker, get_engine
        from sqlalchemy import update as _sa_update, text as _text

        candidate = secrets.token_urlsafe(48)
        async with async_session_maker() as kdb:
            if get_engine().dialect.name == "postgresql":
                await kdb.execute(_text("SET LOCAL lock_timeout = '2s'"))
            await kdb.execute(
                _sa_update(AgentConfig)
                .where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.agent_api_key.is_(None),
                )
                .values(agent_api_key=candidate)
            )
            await kdb.commit()
            # Re-read rather than trusting `candidate`: the CAS is designed so
            # exactly ONE racer sets the key and every racer reads back the
            # SAME winning value. Returning our own candidate after losing the
            # race is how the platform and the routed container end up holding
            # different keys.
            key = (await kdb.execute(
                select(AgentConfig.agent_api_key).where(
                    AgentConfig.user_id == user_id
                )
            )).scalar_one_or_none()
        return key or None
    except Exception as e:                              # noqa: BLE001
        logger.info(
            "[pool_service] pre-commit key mint unavailable for %s (%s) — "
            "falling back to the in-transaction CAS",
            str(user_id)[:8], type(e).__name__,
        )
        return None


async def _build_bind_payload(
    db: AsyncSession,
    user_id: str,
    agent_config: AgentConfig,
) -> dict:
    # User-side identity for the agent's local DB lazy-create path.
    # Without these, the agent stubs `name='Agent Owner'` /
    # `email='<prefix>@agent.local'` and the chat greeting reads as
    # generic forever. Loaded from the platform User row, with sane
    # fallback if the row is somehow missing (won't normally happen
    # for a managed tenant — if it does, /admin/bind degrades to the
    # legacy stub values).
    from app.db.models import User as _PlatformUser
    _u_row = (await db.execute(
        select(_PlatformUser).where(_PlatformUser.id == user_id)
    )).scalar_one_or_none()
    _user_name = (_u_row.name if _u_row and _u_row.name else None) or ""
    _user_email = (_u_row.email if _u_row and _u_row.email else None) or ""
    """Compose the body for `POST /v1/pool/claim`. Mirrors the
    bridge's /admin/bind contract — see backend/app/api/admin_pool.py
    `_BIND_FIELDS`."""
    prefix = user_id[:8]
    # Per-tenant agent_api_key — STABLE across concurrent claims via an atomic
    # compare-and-set. The browser's chat WebSocket authenticates with an HS256
    # JWT signed with this key (ws_chat._validate_session_token); the agent
    # verifies with its bound copy. If two near-simultaneous claims each read
    # agent_api_key=NULL and minted a DIFFERENT random key, the platform DB kept
    # one while Caddy could route to the container bound with the other — the JWT
    # then fails signature verification on the routed agent and EVERY chat 401s
    # "Authentication required" (the 2026-06-30 + 2026-07-01 double-bind incidents;
    # the txn-scoped advisory lock above was meant to prevent it but doesn't hold
    # reliably across every bind path under the :6543 pooler). The CAS below wins
    # at the DB row-lock level with no advisory-lock/pooler caveats: exactly one
    # racer sets the key (WHERE agent_api_key IS NULL), and both re-read the SAME
    # winning value, so every claim for this user carries an identical key and the
    # routed container can always verify the session JWT. Idempotent once set.
    agent_api_key = agent_config.agent_api_key
    if not agent_api_key:
        # COMMITTED BEFORE THE PUSH (D-5, 2026-09-12). The CAS used to run on
        # the caller's transaction, which does not commit until AFTER the
        # bridge call — and the bridge's `/v1/pool/claim` reaches the agent's
        # `/api/admin/bind`, whose handler calls `ensure_mcp_initialized`
        # SYNCHRONOUSLY. That fires `POST /api/mcp/mcp` inside the uncommitted
        # window, `MCPAuthMiddleware` resolves `X-Agent-Key` with a fresh,
        # deliberately UNCACHED read, sees nothing, and 401s. Measured on
        # 12 Sep: key pushed 19:12:38.70, committed ~19:12:40.3, first MCP
        # request 19:12:40.142 — inside it. `mcp_tools_cache` then fell back
        # to an empty "stale" list and the agent ran with ZERO connector tools
        # until 19:14:02, 82 s later. ~1.7 s per claim in the healthy case,
        # and hit EVERY time because the bind handler fires immediately.
        #
        # Committed in its OWN short transaction, not by committing the
        # caller's: the caller holds `pg_advisory_xact_lock('pool_claim:<uid>')`
        # and that lock is xact-scoped, so committing there would release the
        # very guard that makes two racing replicas produce one bind (the
        # 2026-06-30 double-bind incident). A second session leaves it held.
        minted = await ensure_agent_api_key(user_id)
        if minted:
            agent_api_key = minted
            agent_config.agent_api_key = minted
    if not agent_api_key:
        # Fallback: the separate-session mint could not run (SQLite in tests,
        # a lock timeout, a DB blip). Behave exactly as before — a key that is
        # late-visible is still far better than no key at all.
        from sqlalchemy import update as _sa_update
        candidate = secrets.token_urlsafe(48)
        await db.execute(
            _sa_update(AgentConfig)
            .where(AgentConfig.user_id == user_id, AgentConfig.agent_api_key.is_(None))
            .values(agent_api_key=candidate)
        )
        agent_api_key = (await db.execute(
            select(AgentConfig.agent_api_key).where(AgentConfig.user_id == user_id)
        )).scalar_one()
        agent_config.agent_api_key = agent_api_key

    # Image tag for the bridge's refill spawn after this claim. Use the
    # last successful rollout's SHA — same source the slow provision path
    # uses. `settings.docker_agent_image` ("toup-agent:latest") is a
    # fresh-install sentinel, NOT a published GHCR tag, so passing it as
    # the hint causes refill to spawn pool members on whatever the host
    # cached weeks ago. See May 2026 incident: stale `211f0b4beabd`
    # pool members had no /api/admin/bind, every claim 404'd.
    from app.services.docker_host_service import _latest_known_good_image_tag
    image_hint = await _latest_known_good_image_tag(db) or settings.docker_agent_image

    payload = {
        "user_id": str(user_id),
        "agent_api_key": agent_api_key,
        "prefix": prefix,
        "_image_tag": image_hint,
    }
    if _user_name:
        payload["user_name"] = _user_name
    if _user_email:
        payload["user_email"] = _user_email

    # `users.timezone` is authority='tenant' (db/models/base.py), but the
    # PLATFORM is where it is first LEARNED — the web captures it via Intl on
    # boot and PATCHes /auth/profile. Without carrying it, a pool tenant's
    # `users.timezone` stays NULL until its first web/mobile WS turn, and
    # every tz-less channel (WhatsApp, Telegram, voice, routines) buckets its
    # day in UTC — which is how a WhatsApp message at 01:40 UTC landed in a
    # day chat dated tomorrow (2026-09-14). The key is `user_timezone`, NOT
    # `timezone`: runtime_identity._get falls back to os.environ[KEY.upper()]
    # and TIMEZONE/TZ are real container env vars.
    #
    # A falsy platform value emits NO key at all, which is what makes "never
    # overwrite a non-null tenant value with null" structurally true on the
    # receiving side rather than a promise.
    _user_tz = (getattr(_u_row, "timezone", None) or "").strip() if _u_row else ""
    if _user_tz:
        payload["user_timezone"] = _user_tz

    # Channel + identity fields — same set as
    # `_agent_config_to_bridge_body` in docker_host_service. Pulled
    # one-by-one (not a kwarg loop) so adding a new field is an
    # explicit edit here AND in admin_pool._BIND_FIELDS.
    for field in (
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
        # WhatsApp Baileys ACL fields. The pool path used to OMIT both,
        # so pool-bound users had every inbound WhatsApp message silently
        # dropped by `lid.allowlist_empty` even when their settings page
        # showed numbers in "Who can talk to your agent". The legacy slow
        # path's `_agent_config_to_bridge_body` always passed them
        # through; pool drift was an unintentional regression.
        "whatsapp_baileys_allowlist",
        "whatsapp_self_e164",
        "whatsapp_session_status",
        "connect_token",
        "supabase_url",
        "supabase_anon_key",
    ):
        v = getattr(agent_config, field, None)
        if v:
            payload[field] = v

    return payload


async def claim_for_user(
    db: AsyncSession, user_id: str, *, force: bool = False,
    origin: str = "signup",
) -> Optional[ManagedContainer]:
    """Claim a pool container for `user_id`. Returns the populated
    ManagedContainer row on success, None on pool-exhausted-or-disabled.

    Failure modes:
    - Feature flag off (`settings.use_container_pool=False`): None.
    - Pool exhausted (bridge returns 503): None — caller falls back to
      `provision_container`.
    - Bridge unreachable: None + log; caller falls back.
    - Bridge OK but DB write fails: rolls back the bind via
      `bridge.POST /v1/pool/release` (TODO; for now logged as
      ASSIGNED-without-DB-row drift, reconciler picks up).

    Caller is responsible for the fallback decision. We never fall
    back ourselves — the platform's signup path may not always want
    the slow path on pool failure (e.g., during a planned maintenance
    window when the pool is intentionally drained).

    `force=True` skips the existing-row early-return and drives a fresh
    bridge claim even when the row says running. Used by the reclaim
    backstop's keyless sweep: after a container restart the agent keeps
    its bound identity (restored from the tenant DB) but loses every
    bind-injected secret, so the row is honest ('running') while chat
    auth is dead — the bridge's idempotent claim path re-pushes the full
    bind (with this payload's secrets) to the SAME container.
    """
    if not getattr(settings, "use_container_pool", False):
        logger.info("[pool_service] use_container_pool=False — skipping claim")
        return None

    # Serialize concurrent claims for the SAME user. Two near-simultaneous
    # claims (e.g. a ws/chat connect that finds no active agent firing within
    # seconds of a background signup-finalize) would BOTH read existing=None
    # below and BOTH call the bridge /v1/pool/claim — binding two pool
    # containers with DIFFERENT agent keys. Only one survives the
    # managed_containers.user_id unique constraint at commit, leaving the
    # subdomain routed to one container while the DB stores the other's key, so
    # every chat 4001s "Authentication required" (the 2026-06-30 incident). A
    # txn-scoped advisory lock makes the loser BLOCK until the winner commits;
    # it then sees the winner's container via the existing-check below and
    # returns it WITHOUT a second bind. xact-scoped → auto-released on
    # commit/rollback, safe under the :6543 transaction pooler. The open txn
    # already spans the bridge call today, so this adds no new connection hold.
    # Postgres-only; sqlite test paths never reach here (use_container_pool off).
    from app.db.database import get_engine as _get_engine
    if _get_engine().dialect.name == "postgresql":
        from sqlalchemy import text as _text
        # BOUNDED. The wait used to be unbounded, so a second replica sat here
        # for the entire length of the first replica's bridge call — up to 30 s
        # holding a request worker and a DB connection to learn something it
        # could have observed instead. `lock_timeout` applies to advisory-lock
        # acquisition in Postgres, so a loser now raises promptly and we switch
        # to discovery, which is the answer we actually wanted. 0 restores the
        # legacy unbounded wait.
        _wait_s = int(getattr(settings, "pool_claim_lock_wait_s", 0) or 0)
        try:
            if _wait_s > 0:
                await db.execute(_text(f"SET LOCAL lock_timeout = '{_wait_s}s'"))
            await db.execute(
                _text("SELECT pg_advisory_xact_lock(hashtext(:k)::bigint)"),
                {"k": _claim_drive_key(user_id)},
            )
            if _wait_s > 0:
                # Scope the timeout to the lock, not to the bridge call's own
                # statements further down this transaction.
                await db.execute(_text("SET LOCAL lock_timeout = 0"))
        except Exception as _le:
            if _wait_s <= 0 or not _is_lock_timeout(_le):
                # NOT contention. A connection reset, a dead pool, a syntax
                # error — anything else here must keep its own meaning.
                # Swallowing it as "somebody else is driving" would make
                # claim_or_prewarm return True and provision NOBODY, turning a
                # DB blip into a user with no agent at all.
                raise
            await db.rollback()
            logger.info(
                "[pool_service] another driver holds the claim lock for %s (%s) "
                "— observing instead of duplicating",
                str(user_id)[:8], type(_le).__name__,
            )
            signup_trace(user_id, "claim_contended", type(_le).__name__,
                         origin=origin)
            ensure_discovery(user_id, reason="claim_lock_contended")
            raise ProvisionDriveTaken(
                f"claim lock for {str(user_id)[:8]} held by another driver"
            )

    # Existing container check — same idempotency the slow path has. Re-read
    # AFTER the lock so a loser that just blocked sees the winner's freshly
    # committed row and returns it instead of double-binding.
    existing = (await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )).scalar_one_or_none()
    if existing and existing.status in ("running", "provisioning") and not force:
        return existing

    # AgentConfig must already exist (created by /api/agent-setup or
    # in auth.register's prewarm block). If it doesn't, bail — pool
    # claim assumes the row is there for the bind payload.
    agent_config = (await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )).scalar_one_or_none()
    if agent_config is None:
        logger.warning("[pool_service] No AgentConfig for user %s — skipping claim", user_id[:8])
        return None

    # ── Credential guarantee (production invariant) ──────────────────────
    # NEVER bind a managed bundle container whose AgentConfig still lacks LLM
    # credentials. _build_bind_payload forwards `connect_token` / `llm_mode`
    # ONLY when truthy, so a config still on the signup defaults
    # (bundle_status='none', connect_token=NULL, llm_mode='manual') would bind
    # a container with an EMPTY TOUP_TOKEN — and the user's FIRST chat message
    # then 401s at the LLM proxy → "Your API key is invalid" (the 2026-05-30
    # new-signup incident). This is the single chokepoint EVERY container-claim
    # path funnels through (auth.register, Google web + native callback, the
    # agent-setup prewarm endpoint, and any future caller), so minting here
    # makes "the bind always carries a valid token" an invariant rather than a
    # property of each individual call site.
    #
    # Idempotent: activate_free_tier early-returns for already-active users
    # (the explicit signup-path calls already ran for the common case, so this
    # is usually a cheap no-op). Paid users awaiting their Stripe webhook are
    # skipped — their creds are minted by the webhook and they can't chat until
    # they pay anyway (mirrors the same guard in agent_setup.provision).
    # force_env_push=False because the bind below injects the token directly —
    # no recreate, the warm-pool fast path is preserved.
    if agent_config.bundle_status not in ("active", "cancelling"):
        try:
            from app.db.models import CreditBalance
            cb = (await db.execute(
                select(CreditBalance).where(CreditBalance.user_id == user_id)
            )).scalar_one_or_none()
            is_paid_awaiting_stripe = bool(cb and cb.plan_id and cb.plan_id != "free")
            if not is_paid_awaiting_stripe:
                from app.services.free_tier_activation import activate_free_tier
                await activate_free_tier(db, str(user_id), force_env_push=False)
                agent_config = (await db.execute(
                    select(AgentConfig).where(AgentConfig.user_id == user_id)
                )).scalar_one_or_none()
                if agent_config is None:
                    logger.warning(
                        "[pool_service] AgentConfig vanished after activation for %s",
                        user_id[:8],
                    )
                    return None
        except Exception as _ae:
            # Never let activation failure abort the claim — log loudly so the
            # backfill/reconciler (PR #149/#151/#153) can cure a tokenless bind.
            logger.warning(
                "[pool_service] pre-claim free-tier activation failed for %s: %s "
                "— bind may lack TOUP_TOKEN; reconciler will re-converge",
                user_id[:8], _ae,
            )

    payload = await _build_bind_payload(db, user_id, agent_config)

    # Call bridge FIRST — only persist the ManagedContainer row once
    # we have a real host_port (the column is NOT NULL on the schema).
    # Pre-Phase-A code did a placeholder insert+commit before bridge,
    # which 500'd the whole register endpoint via NotNullViolation
    # AND poisoned the session so the User row's response build also
    # blew up. Lesson: never insert a row with required fields you
    # don't yet have.
    from app.services.docker_host_service import _bridge_client
    try:
        async with _bridge_client() as client:
            resp = await client.post("/v1/pool/claim", json=payload)
            if resp.status_code == 503:
                logger.info(
                    "[pool_service] Pool exhausted — fallback to provision for user %s",
                    user_id[:8],
                )
                # DEFINITE: the bridge answered, and it bound nothing. Hand the
                # `pool_claim:` advisory lock back before returning, because
                # the caller's next move is schedule_prewarm and the prewarm
                # now refuses to drive while that lock is held. An xact lock
                # lives until this session's next commit/rollback, and nothing
                # below commits on this path.
                await _release_claim_drive(db)
                return None
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(
            "[pool_service] bridge claim returned %s: %s",
            e.response.status_code, e.response.text[:200],
        )
        signup_trace(user_id, "claim_timeout", f"http_{e.response.status_code}",
                     origin=origin)
        # A 5xx can still leave a completed bind behind (the bind and the
        # response are not the same event), so ask rather than assume — and
        # RAISE, so no caller reads it as "the pool had nothing for this user"
        # and opens the cold named door beside the bind we may already own.
        if e.response.status_code >= 500:
            ensure_discovery(user_id, reason="claim_5xx")
            await _release_claim_drive(db)
            raise ClaimOutcomeUnknown(
                f"bridge claim for {str(user_id)[:8]} answered "
                f"{e.response.status_code}; the bind may have completed"
            )
        # A 4xx IS an answer: the bridge rejected the request and bound
        # nothing. Definite → the named fallback is correct.
        return None
    except httpx.HTTPError as e:
        # `str(httpx.ReadTimeout())` is the EMPTY STRING, which is how the
        # 2026-09-06 trail came to read "bridge unreachable: " and name
        # nothing at all. repr() carries the class; `%r` on the exception
        # would too. Same fix already applied at
        # docker_host_service._update_container_env's retry loop.
        logger.warning("[pool_service] bridge unreachable: %r", e)
        signup_trace(user_id, "claim_timeout", type(e).__name__, origin=origin)
        # THE FIX. A timed-out claim is not a failed claim: on 2026-09-06 the
        # bridge finished this exact bind 2 s after we stopped listening.
        ensure_discovery(user_id, reason="claim_timeout")
        # ...and it is not a claim that FOUND NOTHING either, which is what
        # `return None` used to tell claim_or_prewarm. Discovery owns this
        # user's outcome now; a second driver taking the named path would
        # produce the duplicate the class docstring documents.
        await _release_claim_drive(db)
        raise ClaimOutcomeUnknown(
            f"bridge claim for {str(user_id)[:8]} timed out ({type(e).__name__}); "
            f"the bind may have completed"
        )

    # Bridge response shape: {ok, container_name, host_port, db_pool_slot}
    container_name = data.get("container_name")
    host_port = data.get("host_port")
    if not container_name or not host_port:
        logger.error("[pool_service] bridge claim returned bad shape: %s", data)
        return None

    # Now we have everything required by the schema — upsert the row.
    if existing:
        container = existing
    else:
        import uuid as _uuid
        container = ManagedContainer(
            id=str(_uuid.uuid4()),
            user_id=user_id,
            container_name=container_name,
            host_port=int(host_port),
        )
        db.add(container)
    container.container_name = container_name
    container.host_port = int(host_port)
    # Persist the bridge's real docker container_id (added to the claim
    # response 2026-06). Without this the row is born container_id=NULL and the
    # container-backfill reconciler (predicate `container_id IS NULL`,
    # docker_host_service.backfill_sentinel_image_containers) COLD-REBUILDS this
    # perfectly-healthy warm container ~one reconciler tick later — the
    # warm-claim-abandoned-then-cold-build latency bug. `.get` tolerates an old
    # bridge image that doesn't yet return the field (stays NULL, A3 health-gates
    # the rebuild as the safety net).
    container.container_id = data.get("container_id")
    # Stamp the REAL image_tag the pool container is actually running —
    # NOT settings.docker_agent_image which is the "toup-agent:latest"
    # sentinel ("intentionally a safe sentinel, not a deployable tag" per
    # docker_host_service docstring). Pre-fix every pool-claimed user got
    # the sentinel in the DB, which (a) made `_latest_known_good_image_tag`
    # signals divergent from reality, (b) confused the rollout reconciler
    # into orphan-quarantining containers that were actually fine on the
    # bridge, and (c) propagated through provision_container's fallback so
    # new signups got "image=toup-agent:latest, container_id=None" rows —
    # the user-couldn't-chat root cause for the 2026-05-30 Nariman signup.
    # The bridge already runs pool members on its `current_image_tag` (kept
    # in sync via notify_pool_image_refresh on every successful rollout),
    # so the last-known-good rollout SHA IS the running image. If no
    # rollout has ever completed we still fall back to the sentinel as a
    # last resort + emit a loud WARN so the backfill can cure it.
    from app.services.docker_host_service import _latest_known_good_image_tag
    real_image_tag = await _latest_known_good_image_tag(db)
    if real_image_tag:
        container.image_tag = real_image_tag
    else:
        logger.warning(
            "[POOL-IMAGE-MISS] no known-good rollout — stamping sentinel "
            "for user=%s; backfill will re-provision on next platform boot",
            user_id[:8],
        )
        container.image_tag = settings.docker_agent_image
    container.status = "running"
    container.started_at = datetime.utcnow()
    container.error_message = None
    container.db_name = data.get("db_pool_slot") or container.db_name

    # AgentConfig: agent_url is the same shape as the slow path.
    prefix = user_id[:8]
    agent_url = f"https://agent-{prefix}.agents.toup.ai"
    agent_config.agent_url = agent_url
    agent_config.agent_api_key = payload["agent_api_key"]
    agent_config.hosting_mode = "managed"
    agent_config.deploy_status = "active"

    try:
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("[pool_service] DB commit after bind failed for %s", user_id[:8])
        return None

    logger.info(
        "[pool_service] Claimed %s for user %s (agent_url=%s)",
        container_name, user_id[:8], agent_url,
    )
    signup_trace(user_id, "claim_ok", f"slot={container_name}", origin=origin)
    # The registry just changed; a poller reading a 2 s-old cache would
    # otherwise miss its own success.
    _invalidate_pool_list_cache()
    return container


async def notify_pool_image_refresh(image_tag: str) -> bool:
    """Tell the bridge a new image SHA has rolled out.

    Bridge persists this as `current_image_tag` and the reconciler will
    drain stale-image GENERIC pool members on its next tick (≤30 s),
    respawning them on the new SHA. ASSIGNED members are untouched —
    those go through Phase B blue-green per-tenant.

    Returns True on success, False on bridge unreachable / 4xx / 5xx.
    Failure is non-fatal: the rollout itself succeeds even if the pool
    refresh notification doesn't go through; the reconciler's
    live-tenant fallback eventually catches up. Caller logs and moves on."""
    if not image_tag or image_tag.endswith(":latest"):
        # :latest is a fresh-install sentinel; rollout always provides a
        # real SHA tag, so this guards against accidental misuse.
        logger.info("[pool_service] skip pool refresh (no real tag): %s", image_tag)
        return False
    # Normalize bare 'toup-agent:<sha>' → fully-qualified GHCR ref so
    # the bridge can `docker pull` it directly.
    tag = image_tag.strip()
    if tag.startswith("toup-agent:"):
        tag = "ghcr.io/toup-com/toup-agent:" + tag.split(":", 1)[1]
    from app.services.docker_host_service import _bridge_client
    try:
        async with _bridge_client() as client:
            resp = await client.post("/v1/pool/refresh-image", json={"image_tag": tag})
            if resp.status_code != 200:
                logger.warning(
                    "[pool_service] bridge refresh-image %s: %s",
                    resp.status_code, resp.text[:200],
                )
                return False
            data = resp.json()
            logger.info(
                "[pool_service] pool image refreshed: changed=%s stale_to_drain=%s tag=%s",
                data.get("changed"), data.get("stale_generic_to_drain"), tag,
            )
            return True
    except Exception as e:
        logger.warning("[pool_service] bridge refresh-image unreachable: %s", e)
        return False


_POOL_QUIESCE_POLL_S = 15.0
# Bridge call inside a poll loop — the shared client's 30s default is a
# lifetime here, and a hung snapshot stalls the whole wait.
_POOL_SNAPSHOT_TIMEOUT_S = 10
# Consecutive identical snapshots after which the pool is judged WEDGED rather
# than churning. 4 x 15s = 60s: long enough that a slow batch is not mistaken
# for a stall, short enough that a permanently-stuck slot cannot hold a rollout
# for the whole timeout — which is what slot 26 did on 2026-08-01.
_POOL_STUCK_POLLS = 4


async def pool_churn_snapshot() -> Optional[dict]:
    """How much of the pool is mid-recycle right now, per the bridge.

    Returns None when the bridge can't be reached or answers oddly. The caller
    must treat None as UNKNOWN and proceed — never as "busy" — so losing
    telemetry can never wedge a rollout.
    """
    from app.services.docker_host_service import _bridge_client
    try:
        # Explicit short timeout: this runs in a poll loop, and the whole point
        # of the loop is to stay responsive. The shared client's 30s default is
        # a lifetime here.
        async with _bridge_client(timeout_s=_POOL_SNAPSHOT_TIMEOUT_S) as client:
            resp = await client.get("/v1/pool/health")
            if resp.status_code != 200:
                logger.warning("[pool_service] pool health %s", resp.status_code)
                return None
            data = resp.json()
        summary = data.get("last_reconciler_summary") or {}
        members = data.get("members") or {}
        return {
            "upgrading": list(summary.get("assigned_upgrading") or []),
            "stale": int(summary.get("assigned_stale") or 0),
            "spawning": int(members.get("spawning") or 0),
            "draining": int(members.get("draining") or 0),
        }
    except Exception as e:
        logger.warning("[pool_service] pool churn snapshot unavailable: %s", e)
        return None


def pool_is_busy(snap: dict) -> bool:
    return bool(snap["upgrading"]) or snap["stale"] > 0 or snap["spawning"] > 0 or snap["draining"] > 0


async def wait_for_pool_quiescence(timeout_s: float) -> tuple[bool, str]:
    """Block until the bridge has stopped recycling pool containers.

    WHY THIS EXISTS (measured on production 2026-08-01)

    A rollout that COMPLETES calls notify_pool_image_refresh, and the bridge
    then recycles every pool member — 49 of 50 within 30 minutes, in batches of
    3 assigned upgrades plus generic drains, each spawning a transient extra
    container. The NEXT rollout's canary upgrade lands inside that window and
    loses to it, in whichever way the contention happens to bite:

        17:15  ConnectError after 8.9s
        17:19  heartbeat stale       -> aborted_orphan
        17:27  0 health checks/259s  -> aborted_canary_failed

    Five consecutive rollouts died there, on diffs that were a config int, a
    prompt string and one db.commit() — nothing that can fail a boot. The same
    image then upgraded the same canary in 97s with 3 health checks once the
    pool had settled. Merges arrive in bursts (5 PRs in ~45 min that day), so
    every rollout after the first was landing in its predecessor's churn.

    BUSY IS NOT THE SAME AS MOVING. `assigned_upgrading: ["26"]` with
    `assigned_stale: 1` was reported unchanged at 18:40 and again at 19:23 —
    slot 26 was WEDGED, not recycling. The first cut of this waited on that,
    which is a condition that never clears; the gate then blocked every rollout
    for its whole timeout. So the loop watches for CHANGE, and a snapshot that
    has not moved for `_POOL_STUCK_POLLS` polls is treated as stuck rather than
    busy — the rollout proceeds and says so.

    Returns (quiet, reason). Never raises. Not-quiet returns quiet=False so the
    caller can record it and PROCEED — a late rollout beats a wedged one, and
    the canary gate still protects the fleet if it turns out to be a bad time.

    The CALLER owns heartbeating (rollout_service wraps this in
    `_heartbeating`). An earlier version took a callback and beat through the
    caller's session; it went silent after ~5.5 min and the rollout was orphaned
    at 3.1 min idle. `_heartbeating` exists precisely because an inline beat
    sharing the caller's pool is the bug.
    """
    if timeout_s <= 0:
        return True, "wait disabled"

    deadline = time.time() + timeout_s
    first = await pool_churn_snapshot()
    if first is None:
        return True, "bridge unreachable — proceeding without the check"
    if not pool_is_busy(first):
        return True, "pool already quiescent"

    logger.info("[pool_service] waiting for pool quiescence: %s", first)
    unchanged = 0
    prev = first
    while time.time() < deadline:
        await asyncio.sleep(_POOL_QUIESCE_POLL_S)
        snap = await pool_churn_snapshot()
        if snap is None:
            return True, "bridge went unreachable mid-wait — proceeding"

        if snap == prev:
            unchanged += 1
            if unchanged >= _POOL_STUCK_POLLS:
                return False, (
                    f"pool state unchanged for "
                    f"{int(_POOL_STUCK_POLLS * _POOL_QUIESCE_POLL_S)}s — treating "
                    f"as wedged, not churning: {snap}"
                )
        else:
            unchanged = 0
            prev = snap

        if not pool_is_busy(snap):
            waited = int(timeout_s - (deadline - time.time()))
            logger.info("[pool_service] pool quiescent after %ss", waited)
            return True, f"quiescent after {waited}s"

    last = await pool_churn_snapshot()
    return False, f"still busy after {int(timeout_s)}s: {last}"


@dataclass(frozen=True)
class PoolRelease:
    """What the bridge said when we asked it to release a slot.

    `ok` and `found` are DIFFERENT questions and the caller needs both.
    "The bridge answered and no slot matched" is success with nothing to do;
    "the bridge never answered" is a slot we may still be holding. The old
    signature was a bare bool that collapsed the two, and `destroy_container`
    discarded even that — which is how three pool slots ended up ASSIGNED to
    accounts that no longer exist, their containers still running and their
    databases still on disk, while the deletion receipt said
    `container_destroyed: true`.
    """

    ok: bool
    found: bool = False
    detail: str = ""

    def __bool__(self) -> bool:      # back-compat for truthiness call sites
        return self.ok


async def release_pool_member(
    prefix: Optional[str] = None, user_id: Optional[str] = None
) -> PoolRelease:
    """Tell the bridge a user has been deleted / churned.

    Bridge marks the pool member DRAINING; the reconciler completes the
    destroy on its next tick. The DB role is revoked so any leftover
    connection attempts fail; the DB itself is preserved (cleanup is
    out-of-band). Slot is reusable once the reconciler reaps it; the
    next allocation creates a fresh DB role + password.

    NOTE the consequence of that last paragraph for deletion: a slot that is
    never released is never reaped, so its database is never dropped. A failed
    release is therefore not a tidiness problem, it is the user's data staying
    on disk after they asked for it to be erased.
    """
    if not prefix and not user_id:
        return PoolRelease(ok=False, detail="neither prefix nor user_id given")
    body: dict = {}
    if prefix:
        body["prefix"] = prefix
    if user_id:
        body["user_id"] = str(user_id)
    from app.services.docker_host_service import _bridge_client
    try:
        async with _bridge_client() as client:
            resp = await client.post("/v1/pool/release", json=body)
            if resp.status_code != 200:
                logger.warning(
                    "[pool_service] bridge release %s: %s",
                    resp.status_code, resp.text[:200],
                )
                return PoolRelease(
                    ok=False, detail=f"bridge {resp.status_code}: {resp.text[:120]}"
                )
            data = resp.json()
            logger.info(
                "[pool_service] pool release: found=%s slot=%s prefix=%s user=%s",
                data.get("found"), data.get("slot"), prefix, str(user_id)[:8] if user_id else "",
            )
            return PoolRelease(ok=True, found=bool(data.get("found")))
    except Exception as e:
        # %r, not %s: str(httpx.ReadTimeout()) is the EMPTY STRING, which is why
        # the production trail for this class of failure names no cause at all.
        logger.warning("[pool_service] bridge release unreachable: %r", e)
        return PoolRelease(ok=False, detail=repr(e))


# A pool claim is "fresh" for as long as the signup that made it could still
# plausibly be in flight. Inside that window the slot database holds nothing —
# the user has not sent a message — so the named recreate strands no data.
# Outside it, the same call is the R40 data-loss defect.
POOL_CLAIM_FRESH_WINDOW = timedelta(minutes=15)


async def _load_container(db, user_id: str):
    """The user's ManagedContainer row, or None. Never raises: this runs inside
    a fire-and-forget guard and a failed lookup must degrade to "unknown", not
    to a swallowed exception that skips the heal entirely."""
    try:
        from app.db.models import ManagedContainer
        row = await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == user_id)
        )
        return row.scalar_one_or_none()
    except Exception:
        logger.warning("[pool-heal] could not read managed_containers for %s", user_id[:8])
        return None


def _claim_is_fresh(mc) -> bool:
    """Was this container claimed recently enough that its database is empty?

    UNKNOWN answers FALSE. A lookup that failed, a row with no timestamps, a
    mock in a test — none of those are evidence that the slot holds nothing,
    and the cost of guessing wrong is the user's whole account.
    """
    ts = getattr(mc, "started_at", None) or getattr(mc, "created_at", None)
    if not isinstance(ts, datetime):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts) < POOL_CLAIM_FRESH_WINDOW


async def _verify_and_heal_pool_claim(
    user_id: str,
    *,
    budget_s: float = 30.0,
    interval_s: float = 3.0,
) -> None:
    """Background guard for a just-completed pool claim.

    The pool fast-path's bridge `/v1/pool/claim` response carries no
    container_id and the platform never verifies the bound member is
    actually reachable at the tenant agent_url — so a stale pool member, a
    missing Caddy route, or a half-finished bind leaves the user with a row
    that says status='running' but an agent they can't reach (the 2026-05
    "new user can't talk to his agent" class). The 180s container reconciler
    eventually heals this, but a brand-new user shouldn't eat a multi-minute
    dead window on their very first message.

    This polls the agent's real `/agent/health` for a short budget. A miss is
    observation only: it never restarts or cold-swaps the assigned member.
    Fire-and-forget; never raises into the caller. The authenticated periodic
    reconciler remains the durable repair backstop.
    """
    try:
        from app.services.prewarm_service import _is_agent_actually_healthy
        deadline = asyncio.get_event_loop().time() + budget_s
        while asyncio.get_event_loop().time() < deadline:
            if await _is_agent_actually_healthy(user_id):
                # Bound + reachable (Change 4 made the health check bind-aware).
                # Re-fire the owner/soul sync now that the bind is CONFIRMED:
                # the onboarding soul-save may have raced the bind (fired while
                # the container was still GENERIC), leaving the agent's local
                # owner User row as the stub → "Hey Agent Owner". This
                # deterministic post-bind push corrects owner_name/email +
                # agent_name/color regardless of whether the bridge forwarded
                # user_name into /admin/bind. Fire-and-forget; never blocks.
                try:
                    from sqlalchemy import select as _select
                    from app.db.models import AgentConfig as _AC
                    from app.db.database import async_session_maker as _sm
                    from app.services.docker_host_service import _sync_soul_after_start
                    async with _sm() as _sdb:
                        _cfg = (await _sdb.execute(
                            _select(_AC).where(_AC.user_id == user_id)
                        )).scalar_one_or_none()
                    if _cfg and _cfg.agent_url and _cfg.agent_api_key:
                        asyncio.create_task(
                            _sync_soul_after_start(user_id, _cfg.agent_url, _cfg.agent_api_key)
                        )
                except Exception:
                    logger.exception(
                        "[pool-heal] post-bind owner soul re-sync schedule failed user=%s",
                        user_id[:8],
                    )
                return  # pool member is genuinely reachable — fast path won
            await asyncio.sleep(interval_s)
        logger.warning(
            "[pool-heal] pool claim for user=%s never became reachable in %.0fs "
            "— preserving its canonical assignment for reconciliation",
            user_id[:8], budget_s,
        )
        # A 30-second public-route miss is uncertainty, not proof the
        # assigned member is broken. In particular, restarting and then
        # cold-swapping to `toup-agent-{prefix}` creates a second database and
        # overwrites AgentConfig while the bridge still routes the pool slot.
        # The authenticated periodic reconciler owns any later repair.
        return
    except Exception:
        # Never let the guard crash — the periodic container reconciler is the
        # durable backstop if this fails or a redeploy kills it mid-flight.
        logger.exception("[pool-heal] verify-and-heal failed for user=%s", user_id[:8])


async def claim_or_prewarm(db: AsyncSession, user_id: str) -> bool:
    """Try the pool first; fall back to schedule_prewarm.

    Used by `auth.register` so the existing prewarm-on-register
    behavior is preserved when the pool isn't ready or is exhausted.
    Returns True on any successful claim/prewarm-schedule, False if
    everything failed (caller logs and continues — registration
    itself doesn't depend on this).

    On a successful pool claim we additionally spawn a fire-and-forget
    `_verify_and_heal_pool_claim` guard to observe readiness without mutating
    the just-created ownership mapping. Registration latency is unaffected.
    """
    try:
        c = await claim_for_user(db, user_id, origin="signup")
        if c is not None:
            try:
                _spawn_bg(_verify_and_heal_pool_claim(user_id))
            except Exception:
                # create_task should never fail in an async context, but if it
                # somehow does, the periodic reconciler still covers us.
                logger.warning("[pool_service] could not spawn pool-heal guard for %s", user_id[:8])
            return True
    except ProvisionDriveTaken as e:
        # Somebody else owns this user's outcome — either another driver holds
        # the claim lock, or (ClaimOutcomeUnknown) the bridge never told us
        # what OUR OWN call did. Falling through to schedule_prewarm here is
        # the 2026-09-06 duplicate: a SECOND driver taking the cold NAMED path
        # (POST /v1/tenants) for a user the pool is already binding. Observe
        # instead — claim_for_user has already started discovery, which adopts
        # whatever the bind produces.
        logger.info(
            "[pool_service] provisioning for %s is already being driven (%s) "
            "— observing",
            user_id[:8], type(e).__name__,
        )
        return True
    except Exception:
        logger.exception("[pool_service] claim_for_user raised; falling through to prewarm")
        # Critical: roll back the session so the rest of auth.register
        # (the User-row response build) doesn't fail on a poisoned
        # session. SQLAlchemy raises InvalidRequestError or "current
        # transaction is aborted" on any further use until rollback.
        try:
            await db.rollback()
        except Exception:
            pass

    try:
        from app.services.prewarm_service import schedule_prewarm
        await schedule_prewarm(user_id)
        return True
    except Exception as e:
        logger.warning(
            "[pool_service] schedule_prewarm fallback failed for %s: %s",
            user_id[:8], e,
        )
        return False


# How far back the reclaim backstop will "adopt" a signup whose finalize died
# before it even primed the AgentConfig. The died-early class only occurs AT
# signup (a Railway redeploy killing the fire-and-forget finalize), so a
# 30-day window covers every real case while guaranteeing the reconciler can
# never mass-provision containers for years-old abandoned accounts.
RECLAIM_FRESH_SIGNUP_WINDOW_DAYS = 30

# How stale a status='provisioning' row must be before the backstop treats it
# as dead rather than in-flight. schedule_prewarm marks 'provisioning' BEFORE
# its bridge call; a platform redeploy mid-flight strands the row there, and
# claim_for_user's existing-row check then short-circuits EVERY later claim
# for that user — a permanent dead-end unless something unsticks it.
RECLAIM_PROVISIONING_STALE_MIN = 15

PROBE_STRIKES_BEFORE_RESTART = 2
# Restart at most N containers per tick — a sweep must never become a
# restart storm (mass failures are guarded separately by the quorum check).
RESTARTS_PER_TICK = 2
# …and at most this many restarts for ONE user inside a rolling hour, across
# replicas and across redeploys. The bridge's `tenant_health` already caps
# itself at 3 per 1800 s and then logs "CRASH-LOOP … giving up"; on 12 Sep the
# platform's UNCAPPED restarter overrode that cap every 195 s and re-fired its
# give-up alert three times. The one safety device in the system that said
# "stop" was defeated by a loop that had never heard of it (L3-5).
MAX_RESTARTS_PER_USER_PER_HOUR = 3
RESTART_WINDOW_S = 3600

# After this many consecutive deferred ticks against one subject, the
# per-subject alert escalates from "watching for convergence" to critical
# "healing is NOT happening — needs operator reconciliation".
KEYLESS_NAMED_ESCALATE_TICKS = 3


# ── Sweep safety state (agent_probe_state) ────────────────────────────
#
# This was two module-level dicts, `_PROBE_STRIKES` and `_KEYLESS_NAMED_TICKS`,
# and the 2026-09-12 incident is what per-process state costs (L3-11, §11.3):
#
#   * "two CONSECUTIVE sick ticks before a restart" was really "two ticks on
#     EITHER replica", so a genuinely sick container restarted at twice the
#     intended rate;
#   * every Railway redeploy reset the safety state, so no cap could ever be
#     reached across a deploy;
#   * and the one sentence an operator needed — "f261b564: 46th consecutive
#     tick, class=401, 0 successful probes since 19:12" — lived in one
#     process's memory and in no query. `SELECT * FROM agent_probe_state WHERE
#     consecutive_failures > 3` is the dashboard that did not exist.
#
# Read-modify-write is safe here WITHOUT row locking for one reason and only
# one: the sweep runs under the `container_reconciler` lease, so exactly one
# replica executes it per tick. If that gate is ever removed, this needs a CAS.


def _probe_state_blank(uid: str) -> dict:
    return {
        "user_id": str(uid),
        "consecutive_failures": 0,
        "last_class": None,
        "restarts_in_window": 0,
        "restart_window_started_at": None,
        "first_seen_at": None,
        "escalated_at": None,
        "dirty": False,
        "delete": False,
    }


async def _probe_state_load(uids: list) -> dict:
    """Load every row this tick could touch, in one query. Never raises —
    an unreadable state table must degrade the sweep's MEMORY, never its
    safety: with a blank state nothing has strikes, so nothing restarts."""
    out = {str(u): _probe_state_blank(u) for u in uids}
    if not uids:
        return out
    try:
        from app.db.database import async_session_maker
        from app.db.models import AgentProbeState
        async with async_session_maker() as db:
            rows = (await db.execute(
                select(AgentProbeState).where(
                    AgentProbeState.user_id.in_([str(u) for u in uids])
                )
            )).scalars().all()
        for r in rows:
            out[str(r.user_id)] = {
                "user_id": str(r.user_id),
                "consecutive_failures": int(r.consecutive_failures or 0),
                "last_class": r.last_class,
                "restarts_in_window": int(r.restarts_in_window or 0),
                "restart_window_started_at": r.restart_window_started_at,
                "first_seen_at": r.first_seen_at,
                "escalated_at": r.escalated_at,
                "dirty": False,
                "delete": False,
            }
    except Exception:
        logger.warning("[pool-reclaim] probe-state load failed; this tick runs "
                       "with no memory", exc_info=True)
    return out


async def _probe_state_flush(states: dict) -> None:
    """Persist the rows this tick changed. Never raises."""
    changed = [s for s in states.values() if s.get("dirty") or s.get("delete")]
    if not changed:
        return
    try:
        from app.db.database import async_session_maker
        from app.db.models import AgentProbeState
        from sqlalchemy import delete as _delete
        now = datetime.utcnow()
        async with async_session_maker() as db:
            for s in changed:
                uid = s["user_id"]
                if s.get("delete"):
                    await db.execute(
                        _delete(AgentProbeState).where(
                            AgentProbeState.user_id == uid
                        )
                    )
                    continue
                row = (await db.execute(
                    select(AgentProbeState).where(AgentProbeState.user_id == uid)
                )).scalar_one_or_none()
                if row is None:
                    row = AgentProbeState(user_id=uid)
                    db.add(row)
                row.consecutive_failures = int(s["consecutive_failures"])
                row.last_class = s["last_class"]
                row.restarts_in_window = int(s["restarts_in_window"])
                row.restart_window_started_at = s["restart_window_started_at"]
                row.first_seen_at = s["first_seen_at"]
                row.escalated_at = s["escalated_at"]
                row.updated_at = now
            await db.commit()
    except Exception:
        logger.warning("[pool-reclaim] probe-state flush failed; strikes and "
                       "restart caps fall back to this tick only", exc_info=True)


def _probe_state_ok(st: dict) -> None:
    """A 200. The streak is over — drop the row rather than keep a zero."""
    if (st["consecutive_failures"] or st["last_class"] or st["first_seen_at"]
            or st["escalated_at"] or st["restarts_in_window"]):
        st["delete"] = True


def _probe_state_fail(st: dict, cls: str) -> dict:
    """Record one non-200 sweep of class `cls`. Returns the mutated state."""
    now = datetime.utcnow()
    if st["last_class"] != cls:
        # A different failure class is a different fault. Start the streak
        # again so "46 consecutive 401s" can never be 20 timeouts plus 26 401s.
        st["consecutive_failures"] = 0
        st["first_seen_at"] = now
        st["escalated_at"] = None
    st["consecutive_failures"] = int(st["consecutive_failures"]) + 1
    st["last_class"] = cls
    if st["first_seen_at"] is None:
        st["first_seen_at"] = now
    st["dirty"] = True
    return st


def _restart_allowed(st: dict) -> bool:
    """Is a restart for this user within MAX_RESTARTS_PER_USER_PER_HOUR?

    The window is rolling-by-reset: it opens on the first restart and is
    cleared once RESTART_WINDOW_S has passed since then. Crude on purpose —
    the value of this cap is that it EXISTS and is shared across replicas and
    redeploys, which the in-memory version never was.
    """
    now = datetime.utcnow()
    started = st.get("restart_window_started_at")
    if started is None or (now - started).total_seconds() >= RESTART_WINDOW_S:
        return True
    return int(st.get("restarts_in_window") or 0) < MAX_RESTARTS_PER_USER_PER_HOUR


def _record_restart(st: dict) -> None:
    now = datetime.utcnow()
    started = st.get("restart_window_started_at")
    if started is None or (now - started).total_seconds() >= RESTART_WINDOW_S:
        st["restart_window_started_at"] = now
        st["restarts_in_window"] = 0
    st["restarts_in_window"] = int(st.get("restarts_in_window") or 0) + 1
    st["dirty"] = True


async def _restart_sick_container(user_id: str, container_name: str) -> bool:
    """Restart a managed container via the bridge. Pool members go through
    the pool addon (which re-applies the persisted bind + route after the
    restart); named tenants use the per-tenant restart (keys re-enter from
    .env at boot). Never raises."""
    prefix = str(user_id)[:8]
    try:
        from app.services.docker_host_service import _bridge_client
        async with _bridge_client() as client:
            if (container_name or "").startswith("toup-agent-pool-"):
                r = await client.post(
                    "/v1/pool/restart-member",
                    json={"user_id": str(user_id), "prefix": prefix},
                )
            else:
                r = await client.post(f"/v1/tenants/{prefix}/restart")
            r.raise_for_status()
        return True
    except Exception as e:
        logger.warning(
            "[pool-reclaim] bridge restart failed user=%s container=%s: %s",
            prefix, container_name, e,
        )
        return False


async def _stranded_user_ids(db: AsyncSession, limit: int = 15) -> list[str]:
    """Active users whose agent container is missing or pool-orphaned.

    Stranded classes, all observed in production 2026-07-03/04:
      * Managed-config users with NO ManagedContainer row — the signup's
        fire-and-forget finalize died AFTER priming the config but before
        the claim (a Railway redeploy kills in-flight background tasks).
        11 real users were in this state, including Apple's App Store
        review accounts.
      * Managed-config users with a pool-bound row knocked out of
        status='running' — the rollout whois-404 quarantine (fixed in
        rollout_service._running_tenants; damaged rows persist).
      * FRESH signups (≤30 days) stranded in ANY dead-end shape: config
        never created / never primed off the 'self-hosted' schema default
        (Apple signup 8dwm74…@privaterelay), a row in a terminal
        non-running state (error / orphan / stopped — a schedule_prewarm
        fallback that died leaves these, NAMED name included), or a row
        stuck in 'provisioning' past the stale window (the legacy
        reconcile_stuck_provisioning re-fire demonstrably never reached
        the bridge for days). Guarded: real self-hosters (ssh_host set)
        and hosting_mode='local' users are never matched.

    OLD named (non-pool) containers in orphan/error states are deliberately
    NOT matched — re-homing a heavy-state named tenant is an operator
    decision (see the c47c5b4b recovery runbook). A ≤30-day signup has no
    heavy state, so adopting it onto a pool slot is strictly better than
    leaving it dead.
    """
    from datetime import timedelta
    from sqlalchemy import and_, or_
    from app.db.models import User

    fresh_cutoff = datetime.utcnow() - timedelta(days=RECLAIM_FRESH_SIGNUP_WINDOW_DAYS)
    provisioning_stale_cutoff = datetime.utcnow() - timedelta(
        minutes=RECLAIM_PROVISIONING_STALE_MIN
    )
    config_default_unprimed = and_(
        or_(
            AgentConfig.hosting_mode.is_(None),
            and_(
                AgentConfig.hosting_mode == "self-hosted",
                AgentConfig.ssh_host.is_(None),
            ),
        ),
    )
    rows = await db.execute(
        select(User.id)
        .outerjoin(AgentConfig, AgentConfig.user_id == User.id)
        .outerjoin(ManagedContainer, ManagedContainer.user_id == User.id)
        .where(
            User.is_active == True,  # noqa: E712
            User.id.notin_(_deletion_in_flight_user_ids()),
            or_(
                # Managed users: heal a missing row or a non-running
                # pool-bound row, regardless of signup age.
                and_(
                    AgentConfig.hosting_mode == "managed",
                    or_(
                        ManagedContainer.id.is_(None),
                        and_(
                            ManagedContainer.container_name.like("toup-agent-pool-%"),
                            ManagedContainer.status.notin_(("running", "provisioning")),
                        ),
                    ),
                ),
                # Fresh signups stranded in any dead-end shape: adopt them
                # (config is created and primed managed by reclaim,
                # mirroring the signup path). Real self-hosters and 'local'
                # are excluded by the config guard on the no-row class and
                # by hosting_mode on the row classes below.
                and_(
                    User.created_at >= fresh_cutoff,
                    or_(
                        # finalize died before config prime, no row ever
                        and_(
                            ManagedContainer.id.is_(None),
                            or_(AgentConfig.user_id.is_(None), config_default_unprimed),
                        ),
                        # a row in a terminal non-running state (named OR
                        # pool — error/orphan/stopped): provisioning died
                        and_(
                            ManagedContainer.status.notin_(("running", "provisioning")),
                            or_(
                                AgentConfig.hosting_mode == "managed",
                                config_default_unprimed,
                            ),
                        ),
                        # stuck in 'provisioning' past the stale window
                        and_(
                            ManagedContainer.status == "provisioning",
                            ManagedContainer.updated_at.isnot(None),
                            ManagedContainer.updated_at < provisioning_stale_cutoff,
                            or_(
                                AgentConfig.hosting_mode == "managed",
                                config_default_unprimed,
                            ),
                        ),
                    ),
                ),
            ),
        )
        .order_by(User.created_at.desc())
        .limit(limit)
    )
    return [str(uid) for uid in rows.scalars().all()]


# ── Accounts the backstop must not touch ─────────────────────────────────
#
# `destroy_container` leaves the row at status='deleted' while it still names
# the pool slot, and the User row survives (still `is_active`) until the very
# last step of `delete_user_completely`. That is EXACTLY the shape both
# stranded predicates look for, so a deletion in flight reads as a stranded
# signup — measured on 2026-09-07, where the 15 s fast pass put
# `GET /v1/pool/whois?user_id=c7905c52-…` on the bridge four seconds after that
# account's slot had been released. The fast pass only adopts and the member
# was already DRAINING, so nothing happened; the 180 s scan does not adopt, it
# CLAIMS, and a tick landing in the same window binds a fresh pool container,
# database and Caddy route to an account whose deletion has already passed the
# point where anything would release them.
#
# `deletion_audit_events` is the right marker and needs no new column: it is
# INSERTed and committed as the FIRST transaction of `delete_user_completely`,
# before Stripe and before the container, and it outlives the user row.
#
# 'failed' is split by step on purpose. Stripe runs first and aborts before any
# local teardown — "the user is still functional and an admin can retry" — so
# those users are ordinary customers and must still be healed. From the
# container step onward, teardown was owed and did not land
# (`ContainerTeardownIncomplete`), and claiming a NEW slot on top of that is
# how a released-but-not-reaped slot becomes a permanently leaked one.
_DELETION_PAST_TEARDOWN_STEPS = ("container", "openai", "cascade", "user_row")


def _deletion_in_flight_user_ids():
    """Scalar subquery of user ids whose account deletion must not be undone.

    A clean deletion removes the user row, so 'succeeded' never matches
    anything the predicates could join to; it is left out rather than excluded
    so the intent stays readable."""
    from app.db.models import DeletionAuditEvent
    from sqlalchemy import and_, or_
    return (
        select(DeletionAuditEvent.user_id)
        .where(
            or_(
                DeletionAuditEvent.status == "in_progress",
                and_(
                    DeletionAuditEvent.status == "failed",
                    DeletionAuditEvent.failure_step.in_(
                        _DELETION_PAST_TEARDOWN_STEPS
                    ),
                ),
            )
        )
    )


async def reclaim_stranded_users(max_per_tick: int = 5) -> dict:
    """Reconciler backstop: replay the signup finalize for stranded users.

    Per user: ensure the AgentConfig exists and is primed to managed (only
    when it is still on the untouched schema default — real self-hosters
    are excluded by the predicate), then re-run the pool claim. Safe to
    re-run: claim_for_user takes a per-user advisory lock, reuses the
    stable agent_api_key (CAS), and the bridge's per-user idempotent claim
    returns the user's EXISTING container when already bound — the
    platform row flips back to status='running' and the Caddy route is
    re-asserted. Truly containerless users get a fresh warm claim, exactly
    as at signup. claim_for_user's credential guarantee mints free-tier
    creds internally, so no separate activation step is needed here.

    Runs enumeration and each heal in its own narrow session; never raises.
    """
    if not getattr(settings, "use_container_pool", False):
        return {"skipped": "pool_disabled"}
    summary: dict = {"candidates": 0, "claimed": 0, "failed": 0}
    try:
        from app.db.database import async_session_maker
        # Window is deliberately much larger than max_per_tick: enumeration
        # is newest-first, so with a small window a cluster of permanently-
        # failing fresh signups could starve older stranded users forever.
        # 60 candidates + shuffle means starvation needs >60 simultaneously
        # ever-failing users — at which point the failure alerts fire anyway.
        async with async_session_maker() as db:
            candidates = await _stranded_user_ids(db, limit=60)
        summary["candidates"] = len(candidates)
        # NO early return on empty — phase 2 (the keyless-agent sweep below)
        # must run EVERY tick regardless. The original `return summary` here
        # made phase 2 unreachable in the steady state (zero stranded users
        # is the NORMAL condition), which left the 2026-07-04 keyless agents
        # broken while this function reported all-quiet.
        # Shuffle so a persistently-failing user can't starve the rest of
        # the backlog (enumeration is newest-first within the window).
        import random
        random.shuffle(candidates)
        for uid in candidates[:max_per_tick]:
            try:
                async with async_session_maker() as udb:
                    # Finalize replay step 1: config exists + primed. Mirrors
                    # _bg_finalize_oauth_signup / register's prime block.
                    from app.api.agent_setup import _get_or_create_config
                    cfg = await _get_or_create_config(uid, udb)
                    dirty = False
                    if cfg.hosting_mode in (None, "self-hosted") and not cfg.ssh_host:
                        cfg.hosting_mode = "managed"
                        dirty = True
                    if not cfg.whatsapp_mode:
                        cfg.whatsapp_mode = "qr_link"
                        dirty = True
                    if dirty:
                        await udb.commit()
                    # Step 2: unstick a stale 'provisioning' row. schedule_prewarm
                    # stamps 'provisioning' before its bridge call; if that task
                    # died (redeploy), the row wedges there forever and
                    # claim_for_user's existing-row check short-circuits every
                    # subsequent claim. Flip stale ones to 'error' so the claim
                    # below proceeds; genuinely in-flight rows are skipped (the
                    # predicate only surfaces them past the stale window, but
                    # re-check here in case provisioning restarted meanwhile).
                    from datetime import timedelta as _td
                    mc = (await udb.execute(
                        select(ManagedContainer).where(ManagedContainer.user_id == uid)
                    )).scalar_one_or_none()
                    if mc and mc.status == "provisioning":
                        stale = (
                            mc.updated_at is not None
                            and mc.updated_at
                            < datetime.utcnow() - _td(minutes=RECLAIM_PROVISIONING_STALE_MIN)
                        )
                        if not stale:
                            continue  # in-flight — let it finish
                        mc.status = "error"
                        mc.error_message = "[pool-reclaim] unstuck stale provisioning"
                        await udb.commit()
                    # Step 3: the claim (activation happens inside).
                    c = await claim_for_user(udb, uid, origin="reclaim")
                if c is not None:
                    summary["claimed"] += 1
                    logger.warning(
                        "[pool-reclaim] healed stranded user=%s -> %s",
                        uid[:8], c.container_name,
                    )
                else:
                    summary["failed"] += 1
            except Exception as e:
                summary["failed"] += 1
                logger.warning("[pool-reclaim] claim failed user=%s: %s", uid[:8], e)
    except Exception:
        logger.exception("[pool-reclaim] enumeration failed")

    # Phase 2 — authenticated agent sweep, ALL managed containers. The only
    # honest "can this user chat?" check is an AUTHENTICATED probe with the
    # platform's stored key (2026-07-04: /agent/health showed green while
    # keyless agents 401'd every message and a poisoned-DB agent 500'd every
    # message for three days). Every running managed container is probed each
    # tick — pool-bound AND named (the named ones are the OLDEST users and
    # were invisible to the original pool-only sweep):
    #
    #   200          → healthy, clear strikes.
    #   401/403      → keyless/desynced. Pool: force re-claim (bridge
    #                  idempotent claim re-pushes the full bind — no-op if
    #                  healthy). Named: alert and preserve for ownership
    #                  reconciliation; a route/config split cannot be repaired
    #                  by restarting the named container.
    #   404/4xx      → routing problem (Caddy "unknown tenant") — the bridge
    #                  route reconciler owns that; never restart a healthy
    #                  agent over a missing route.
    #   5xx/timeout  → sick agent (poisoned DB pool, wedged process). Two
    #                  CONSECUTIVE sick ticks → restart via bridge + alert.
    #                  Would have auto-healed tenant 3134fece days before the
    #                  user reported it.
    #
    # Mass-failure guard: when the majority of probes fail at the transport
    # layer in one tick, that's a platform-egress or host-level problem, not
    # N individually-sick agents — no strikes are recorded (a restart storm
    # across the fleet would turn a blip into an outage) and we alert instead.
    try:
        from app.db.database import async_session_maker
        import httpx as _httpx
        async with async_session_maker() as db:
            rows = (await db.execute(
                select(
                    ManagedContainer.user_id,
                    ManagedContainer.container_name,
                    AgentConfig.agent_url,
                    AgentConfig.agent_api_key,
                )
                .join(AgentConfig, AgentConfig.user_id == ManagedContainer.user_id)
                .where(
                    ManagedContainer.status == "running",
                    AgentConfig.agent_url.isnot(None),
                    AgentConfig.agent_api_key.isnot(None),
                )
            )).all()

        async def _probe(uid: str, cname: str, url: str, key: str, client) -> tuple:
            try:
                r = await client.get(
                    f"{url}/api/sessions",
                    params={"limit": 1},
                    headers={"X-Agent-Key": key},
                )
                return (str(uid), cname or "", r.status_code, None)
            except Exception as e:
                return (str(uid), cname or "", None, type(e).__name__)

        results: list = []
        if rows:
            async with _httpx.AsyncClient(timeout=8) as client:
                results = await asyncio.gather(
                    *[_probe(u, n or "", a, k, client) for u, n, a, k in rows]
                )

        transport_errors = sum(1 for _, _, s, _ in results if s is None)
        mass_failure = bool(results) and transport_errors > len(results) / 2

        # Shared, durable strike/cap state. One query for every row this tick
        # could touch; one flush at the end. Safe to read-modify-write without
        # a row lock ONLY because the sweep is leader-gated (W1) — one replica
        # per tick. See _probe_state_load.
        states = await _probe_state_load([uid for uid, _, _, _ in results])

        keyless_pool: list = []
        keyless_named: list = []
        sick: list = []
        for uid, cname, status, err in results:
            is_pool = cname.startswith("toup-agent-pool-")
            st = states.get(uid) or _probe_state_blank(uid)
            states[uid] = st
            if status == 200:
                _probe_state_ok(st)
                continue
            if status in (401, 403):
                _probe_state_fail(st, "401")
                if is_pool:
                    keyless_pool.append((uid, cname))
                else:
                    keyless_named.append((uid, cname))
                continue
            if status is not None and status < 500:
                # 404 etc. — routing, owned by the bridge route reconciler.
                _probe_state_fail(st, "4xx")
                continue
            if mass_failure and status is None:
                continue  # platform-egress blip; no per-agent strikes
            _probe_state_fail(st, "5xx" if status is not None else "transport")
            if st["consecutive_failures"] >= PROBE_STRIKES_BEFORE_RESTART:
                sick.append((uid, cname, status if status is not None else err))

        summary["keyless"] = len(keyless_pool) + len(keyless_named)
        summary["sick"] = len(sick)

        if mass_failure:
            logger.warning(
                "[pool-reclaim] sweep quorum failed: %d/%d probes hit transport "
                "errors — skipping strikes this tick",
                transport_errors, len(results),
            )
            try:
                from app.services.alerting import send_infra_alert
                await send_infra_alert(
                    "sweep-quorum", "critical",
                    f"Agent sweep: {transport_errors}/{len(results)} probes failed "
                    "at transport level — platform egress or agent host problem?",
                )
            except Exception:
                pass

        # Keyless pool members → force re-claim (full bind refresh).
        for uid, cname in keyless_pool[:max_per_tick]:
            try:
                async with async_session_maker() as udb:
                    c = await claim_for_user(udb, uid, force=True, origin="heal")
                if c is not None:
                    summary["rebound"] = summary.get("rebound", 0) + 1
                    logger.warning(
                        "[pool-reclaim] re-bound keyless agent user=%s -> %s",
                        uid[:8], c.container_name,
                    )
                else:
                    summary["failed"] += 1
            except Exception as e:
                summary["failed"] += 1
                logger.warning(
                    "[pool-reclaim] keyless re-bind failed user=%s: %s", uid[:8], e
                )

        # A named 401 is an OWNERSHIP question, and it is answerable: ask the
        # bridge which container it believes owns this user and what Caddy
        # dials for the hostname, then act only on what that says. Restarting
        # the named container can never repair a platform-side row, which is
        # why the pre-738 code restarted forever and the post-738 code deferred
        # forever. Neither converged; this does, in one tick, for the one shape
        # that produced the 12 Sep incident.
        from app.services.alerting import send_infra_alert as _send_alert
        for uid, cname in keyless_named:
            st = states[uid]
            ticks = st["consecutive_failures"]
            outcome, ev = await _reconcile_named_401(uid, cname)
            summary[f"named_401_{outcome}"] = (
                summary.get(f"named_401_{outcome}", 0) + 1
            )

            if outcome == OWNERSHIP_ADOPT:
                adopted = await _adopt_bridge_truth(uid, ev)
                if adopted:
                    summary["repaired"] = summary.get("repaired", 0) + 1
                    _probe_state_ok(st)          # converged; drop the streak
                    st["delete"] = True
                    logger.warning(
                        "[pool-reclaim] REPAIRED named/pool mismatch user=%s: "
                        "platform row said %s, bridge says %s — adopted",
                        uid[:8], cname, ev.get("pool_container"),
                    )
                    continue
                summary["failed"] += 1
                outcome = OWNERSHIP_UNKNOWN      # fall through to the alert

            summary["keyless_named_deferred"] = (
                summary.get("keyless_named_deferred", 0) + 1
            )
            logger.warning(
                "[pool-reclaim] named tenant 401 user=%s container=%s "
                "outcome=%s (tick %d) evidence=%s",
                uid[:8], cname, outcome, ticks, ev,
            )
            try:
                if outcome == OWNERSHIP_AMBIGUOUS:
                    # The state nobody owned. The bridge's route restorer
                    # refuses it by design and reports it into a field that is
                    # never logged; four accounts sat here for a day and one
                    # went dark on the next Caddy restart with nothing paging.
                    # This is that page.
                    await _send_alert(
                        "pool-ownership-ambiguous", "critical",
                        f"{uid[:8]}: named AND pool ownership both present — "
                        f"NOT acting. named_container="
                        f"{ev.get('named_container')} db={ev.get('named_db')}; "
                        f"pool_container={ev.get('pool_container')} "
                        f"db={ev.get('pool_db')}; caddy_route="
                        f"{ev.get('route_container') or ev.get('route_upstream')}; "
                        f"platform_row={ev.get('platform_container')}. "
                        f"401 for {ticks} consecutive ticks. Only an operator "
                        "can decide which database holds this user's life.",
                        subject=uid[:8],
                    )
                    st["escalated_at"] = st["escalated_at"] or datetime.utcnow()
                    st["dirty"] = True
                elif outcome == OWNERSHIP_NO_BIND:
                    # NEVER force-claim here. An established user with no
                    # bridge bind is an escalation, not a re-claim: a
                    # force-claim binds an empty database (R40).
                    await _send_alert(
                        "pool-selfheal-stuck", "critical",
                        f"{uid[:8]} answered 401/403 for {ticks} consecutive "
                        f"ticks (container={cname}) and the bridge reports NO "
                        "bind for this user at all. NOT re-claiming — a "
                        "force-claim would bind an established account to a "
                        "fresh empty slot. Needs an operator.",
                        subject=uid[:8],
                    )
                    st["escalated_at"] = st["escalated_at"] or datetime.utcnow()
                    st["dirty"] = True
                elif ticks >= KEYLESS_NAMED_ESCALATE_TICKS:
                    await _send_alert(
                        "pool-selfheal-stuck", "critical",
                        f"NOT healing: {uid[:8]} has answered 401/403 for {ticks} "
                        f"consecutive reconciliation ticks (container={cname}, "
                        f"since {st.get('first_seen_at')}). Bridge ownership "
                        f"evidence: {ev}. A restart cannot repair a platform "
                        "key/route mismatch — this needs operator "
                        "reconciliation of named/pool ownership.",
                        subject=uid[:8],
                    )
                    st["escalated_at"] = st["escalated_at"] or datetime.utcnow()
                    st["dirty"] = True
                else:
                    await _send_alert(
                        "pool-selfheal-stuck", "warning",
                        f"{uid[:8]} answered 401/403 (container={cname}); restart "
                        f"REFUSED — a restart can't fix a key/route mismatch. "
                        f"Tick {ticks}; watching for convergence.",
                        subject=uid[:8],
                    )
            except Exception:
                pass

        # Sick (2 consecutive 5xx/timeout ticks) → restart via bridge, bounded
        # per tick AND per user per hour. The per-user cap is the half that was
        # missing: the bridge caps itself at 3 restarts per 1800 s and then
        # gives up loudly, and the platform's uncapped restarter overrode that
        # every 195 s while re-firing the bridge's give-up alert (L3-5).
        for uid, cname, why in sick[:RESTARTS_PER_TICK]:
            st = states[uid]
            if not _restart_allowed(st):
                summary["restart_capped"] = summary.get("restart_capped", 0) + 1
                logger.warning(
                    "[pool-reclaim] restart CAPPED for user=%s container=%s "
                    "(%d in the last hour, max %d) — still probing so recovery "
                    "is noticed", uid[:8], cname,
                    st.get("restarts_in_window") or 0,
                    MAX_RESTARTS_PER_USER_PER_HOUR,
                )
                if not st.get("escalated_at"):
                    st["escalated_at"] = datetime.utcnow()
                    st["dirty"] = True
                    try:
                        await _send_alert(
                            "pool-selfheal-stuck", "critical",
                            f"{uid[:8]} is still {why} after "
                            f"{MAX_RESTARTS_PER_USER_PER_HOUR} restarts in an "
                            f"hour (container={cname}). Restarting has stopped; "
                            "probing continues. A container that does not come "
                            "back from three restarts needs an operator.",
                            subject=uid[:8],
                        )
                    except Exception:
                        pass
                continue
            ok = await _restart_sick_container(uid, cname)
            _record_restart(st)
            st["consecutive_failures"] = 0
            st["dirty"] = True
            summary["restarted"] = summary.get("restarted", 0) + (1 if ok else 0)
            logger.warning(
                "[pool-reclaim] restarted sick agent user=%s container=%s "
                "(reason=%s) ok=%s", uid[:8], cname, why, ok,
            )

        await _probe_state_flush(states)
    except Exception:
        logger.exception("[pool-reclaim] authenticated sweep failed")

    # Law 4: healing is good; REPEATED healing means something upstream broke.
    # The rollup distinguishes actions REQUESTED (a claim/rebind/restart that
    # returned 2xx) from state OBSERVED (probe classes this tick). A
    # keyless_named_deferred entry was NOT healed — calling it "healed" is the
    # exact lie PR 738 removed — so it lives under "observed", and the stuck
    # subjects behind it are alerted PER-SUBJECT (with a consecutive-tick count
    # and a critical escalation) in the loop above.
    try:
        requested = {
            k: summary[k] for k in ("claimed", "rebound", "restarted")
            if summary.get(k)
        }
        observed = {
            k: summary[k]
            for k in ("keyless", "sick", "keyless_named_deferred", "failed")
            if summary.get(k)
        }
        if requested or observed:
            from app.services.alerting import send_infra_alert
            await send_infra_alert(
                "pool-selfheal", "warning",
                f"Reconciliation tick — requested: {requested or '{}'}; "
                f"observed: {observed or '{}'}. 'requested' counts confirm only "
                "that a claim/rebind/restart request succeeded; end-user chat "
                "readiness still requires verification. 'observed' are probe "
                "states this tick — keyless_named_deferred entries were NOT "
                "healed (a restart cannot fix a key/route mismatch) and are "
                "alerted per-subject.",
            )
    except Exception:
        pass
    return summary


# ══════════════════════════════════════════════════════════════════════════
# Late-success discovery (2026-09-06 incident)
#
# The claim that "failed" at 18:18:09 had in fact SUCCEEDED at 18:18:11: the
# bridge finished binding toup-agent-pool-73 two seconds after the platform's
# flat 30 s httpx timeout fired on both callers. Nothing on the platform asked
# again. The user's app (build 109) gave up at 18:18:45 and the 180 s
# container reconciler only noticed at 18:19:12.
#
# A timed-out bridge call is not evidence that nothing happened — it is
# evidence that we do not know. Everything below exists to go and ask,
# promptly, cheaply, and without ever binding a second slot.
# ══════════════════════════════════════════════════════════════════════════


def _t_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


# user prefix -> monotonic clock at registration. Bounded FIFO: this is a log
# convenience, never a source of truth.
#
# It used to stamp t0 on FIRST SIGHTING, which made the number meaningless:
# nothing seeded it at registration (`seed_signup_trace` had no caller at all),
# so on the claim path the origin WAS the claim and every hop printed
# `elapsed_ms=0`. Production printed exactly that for a 4 s register->bound on
# 2026-09-07, and for two reclaims of months-old users during the P0 walk.
_TRACE_T0: dict = {}
_TRACE_T0_MAX = 512


def _trace_put(key: str, t0: float) -> None:
    if len(_TRACE_T0) >= _TRACE_T0_MAX:
        # FIFO evict — dicts preserve insertion order.
        for k in list(_TRACE_T0)[: _TRACE_T0_MAX // 4]:
            _TRACE_T0.pop(k, None)
    _TRACE_T0[key] = t0


def _trace_origin(user_id: str) -> Optional[float]:
    """t0 for this user, or None when this process never saw their
    registration. None is a real answer and prints as `elapsed_ms=null`.

    With two Railway replicas the claim routinely completes in the process that
    did NOT serve the registration, and a fabricated t0 there reports a long
    hop as instantaneous — which is the same string a genuinely instant hop
    produces, so the two could not be told apart."""
    return _TRACE_T0.get(str(user_id)[:8])


def seed_signup_trace(user_id: str, *, only_if_absent: bool = False) -> bool:
    """Stamp t0 for a user's signup trace. Call from the registration path so
    every later `elapsed_ms` is measured from registration rather than from
    whenever this process first heard of them.

    Returns True iff this call stamped it. `only_if_absent=True` is for the
    background finalizers, which run AFTER `auth.register` in the same process
    on the password path: re-seeding there would move t0 forward past the work
    it is supposed to measure, and re-emitting `registered` would double-count
    one signup. On the OAuth paths there is no earlier seed and they are the
    registration, so they stamp it and emit the hop.
    """
    key = str(user_id)[:8]
    if only_if_absent and key in _TRACE_T0:
        return False
    _trace_put(key, time.monotonic())
    return True


def signup_trace(
    user_id: str,
    hop: str,
    detail: str = "",
    *,
    origin: str = "signup",
    created_at=None,
) -> None:
    """One structured line per provisioning hop, keyed by the user's 8-hex
    prefix. Deliberately boring: no secrets, no message content, no PII — a
    prefix, a hop name, an origin, an elapsed_ms and a short mechanical detail.

    Hops: registered | prewarm_start | claim_ok | claim_timeout |
          discovered | adopted | bound | ready
    (plus a few mechanical ones: prewarm_dedupe, discover_start, discover_give_up,
     config_sync_deferred).

    `origin` says WHY this hop happened — signup | reclaim | discovery | heal.
    Without it `hop=claim_ok` covers a fresh registration, the 180 s stranded
    backstop re-claiming a months-old user, a discovery adopt and the keyless
    sweep's force re-bind, and on-call cannot tell which they are looking at.

    `created_at` is the one way to get an elapsed on a replica that did not
    serve the registration, and it takes a VALUE the caller already holds —
    never a user id to look up. A logging helper must not issue a DB round trip.
    """
    try:
        pfx = str(user_id)[:8]
        t0 = _trace_origin(pfx)
        if t0 is None and created_at is not None:
            # Wall-clock delta, converted to this process's monotonic frame and
            # cached so later hops for the same user stay consistent.
            try:
                now = datetime.utcnow()
                ref = created_at
                if getattr(ref, "tzinfo", None) is not None:
                    ref = ref.replace(tzinfo=None)
                age = (now - ref).total_seconds()
                if age >= 0:
                    t0 = time.monotonic() - age
                    _trace_put(pfx, t0)
            except Exception:
                t0 = None
        elapsed = "null" if t0 is None else str(_t_ms(t0))
        logger.info(
            "[signup-trace] user=%s hop=%s origin=%s elapsed_ms=%s detail=%s",
            pfx, hop, origin or "unknown", elapsed, (detail or "")[:120],
        )
    except Exception:
        pass


# ── The cheap bridge read ────────────────────────────────────────────────
#
# `GET /v1/pool/list` is the only route on the DEPLOYED bridge that can answer
# "which slot is user X bound to" without shelling out to docker: its handler
# is a plain `def` (FastAPI threadpool, so it cannot block the bridge's event
# loop) and its body is one members.json read. It answers for every user at
# once, which is why the result is cached and single-flighted below — twenty
# concurrent pollers must cost the bridge one call, not twenty.
#
# `bridge_pool_whois_route` swaps in a per-user route when one exists.

_POOL_LIST_CACHE: dict = {"ts": -1e9, "data": None}
_POOL_LIST_LOCK: Optional[asyncio.Lock] = None
_WHOIS_UNAVAILABLE = False
_TENANT_TRUTH_UNAVAILABLE = False


# ── Ownership truth (GET /v1/pool/tenant-truth) ───────────────────────
#
# The 4xx branch of the sweep used to defer routing to the bridge's route
# reconciler, and the bridge's route reconciler refuses to act on exactly this
# class of account (`named_and_pool_ownership_ambiguous`). Two healers, each
# correctly deferring to the other, is zero healers — and four accounts sat in
# that hole, one of which went dark on 13 Sep 01:59 with nothing paging (L3-3,
# L3 §5). The platform's authenticated probe is the only thing that can see
# "the key I hold does not open the door this hostname leads to"; to act on it
# safely it needs to know WHICH container legitimately owns the user, and the
# bridge is the authority on that.

OWNERSHIP_ADOPT = "adopt"
OWNERSHIP_AMBIGUOUS = "ambiguous"
OWNERSHIP_NO_BIND = "no_bind"
OWNERSHIP_UNKNOWN = "unknown"


def _first_str(body: dict, *keys: str) -> Optional[str]:
    for k in keys:
        v = body.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _tenant_truth_view(body: dict) -> Optional[dict]:
    """Normalise `/v1/pool/tenant-truth` into the four facts this decision
    needs. Returns None when the body carries none of them.

    Tolerant of spelling because the bridge is another lane's code and is not
    in this repository — but never GUESSES. An unrecognised body answers None,
    which routes to OWNERSHIP_UNKNOWN, which does nothing and alerts. The
    alternative (assume "no named container" from a body we could not read)
    would authorise an adopt on the strength of a parse failure, and this
    branch's whole purpose is to stop acting on beliefs we cannot evidence.
    """
    if not isinstance(body, dict):
        return None
    pool = body.get("pool") if isinstance(body.get("pool"), dict) else {}
    named = body.get("named") if isinstance(body.get("named"), dict) else {}
    route = body.get("route") if isinstance(body.get("route"), dict) else {}

    view = {
        "prefix": _first_str(body, "prefix", "tenant", "user_prefix"),
        "pool_container": (
            _first_str(pool, "container_name", "container", "name")
            or _first_str(body, "pool_container", "pool_container_name")
        ),
        "pool_db": (
            _first_str(pool, "db_name", "db", "db_pool_slot")
            or _first_str(body, "pool_db", "pool_db_name")
        ),
        "named_container": (
            _first_str(named, "container_name", "container", "name")
            or _first_str(body, "named_container", "named_container_name")
        ),
        "named_db": (
            _first_str(named, "db_name", "db")
            or _first_str(body, "named_db", "named_db_name")
        ),
        "route_container": (
            _first_str(route, "container_name", "container", "upstream_container")
            or _first_str(body, "route_container", "route_target_container")
        ),
        "route_upstream": (
            _first_str(route, "upstream", "target", "dial")
            or _first_str(body, "route_upstream", "route_target")
        ),
    }
    if not any(view[k] for k in
               ("pool_container", "named_container", "route_container",
                "route_upstream")):
        # The bridge may legitimately answer "I know this prefix and it has
        # nothing" — but only if it SAYS so. An explicit empty answer is an
        # answer; a body we could not read is not.
        if body.get("found") is False or body.get("known") is True:
            return view
        return None
    return view


async def bridge_tenant_truth(
    prefix: str, *, timeout_s: Optional[float] = None,
) -> Optional[dict]:
    """ADAPTER. `GET /v1/pool/tenant-truth?prefix=…` — the bridge's own view of
    which container owns a tenant prefix and what Caddy dials for it.

    Returns the normalised view, or None when the bridge cannot answer, does
    not implement the route, or answers something this adapter cannot read.
    None means "no evidence", never "no named container". Never raises — every
    caller's safe action on no-evidence is to do nothing and alert.
    """
    global _TENANT_TRUTH_UNAVAILABLE
    if _TENANT_TRUTH_UNAVAILABLE:
        return None
    if timeout_s is None:
        timeout_s = float(
            getattr(settings, "provision_discovery_read_timeout_s", 8) or 8
        )
    route = (
        getattr(settings, "bridge_tenant_truth_route", "/v1/pool/tenant-truth")
        or ""
    ).strip()
    if not route:
        return None
    try:
        from app.services.docker_host_service import get_bridge_client
        client = await get_bridge_client()
        r = await client.get(
            route, params={"prefix": str(prefix)}, timeout=timeout_s,
        )
        if r.status_code in (404, 405, 501):
            # Configured but not deployed. Degrade ONCE, not once per subject
            # per tick — 79 rows x 2 replicas x every 195 s is a lot of 404s.
            _TENANT_TRUTH_UNAVAILABLE = True
            logger.info(
                "[pool-reclaim] bridge tenant-truth route not deployed (%s) — "
                "named-401 repair degrades to alert-only", r.status_code,
            )
            return None
        r.raise_for_status()
        return _tenant_truth_view(r.json() or {})
    except Exception as e:
        logger.info(
            "[pool-reclaim] tenant-truth read failed for %s: %s",
            str(prefix)[:8], type(e).__name__,
        )
        return None


def classify_named_401_ownership(
    *,
    platform_container: Optional[str],
    slot: Optional[dict],
    truth: Optional[dict],
) -> tuple:
    """Decide what a NAMED tenant's 401 means. Pure — the whole point.

    Returns `(outcome, evidence)`. Three actionable outcomes and one
    deliberate non-answer:

      * ADOPT     — the bridge reports a POOL bind for this user, reports NO
                    named container for the prefix, the Caddy route (when it
                    reports one) names that same pool container, and the
                    platform row names something ELSE. The platform is simply
                    wrong; adopting the bridge's answer converges in one tick.
                    This is the 12 Sep shape, and this branch would have ended
                    that incident at 19:18.
      * AMBIGUOUS — the bridge reports BOTH a pool bind and a named container
                    for the prefix. Two databases, two candidate owners, and
                    only a human can say which one holds the user's life. Do
                    NOT act; page with both container names and both DB names.
      * NO_BIND   — the bridge has no bind at all. A force-claim here binds an
                    established user to a fresh EMPTY slot (R40), so this is an
                    escalation, never a re-claim.
      * UNKNOWN   — no evidence. Defer and alert, exactly as wave 1 did.
    """
    if truth is None:
        return OWNERSHIP_UNKNOWN, {"reason": "no_bridge_truth"}

    pool_c = truth.get("pool_container") or (
        (slot or {}).get("container_name")
        if str((slot or {}).get("container_name") or "").startswith(
            "toup-agent-pool-")
        else None
    )
    named_c = truth.get("named_container")
    route_c = truth.get("route_container")
    ev = {
        "pool_container": pool_c,
        "pool_db": truth.get("pool_db") or (slot or {}).get("db_name"),
        "named_container": named_c,
        "named_db": truth.get("named_db"),
        "route_container": route_c,
        "route_upstream": truth.get("route_upstream"),
        "platform_container": platform_container,
    }

    if pool_c and named_c:
        return OWNERSHIP_AMBIGUOUS, ev
    if not pool_c and not named_c:
        return OWNERSHIP_NO_BIND, ev
    if not pool_c:
        # A named container the bridge owns, and the platform row may or may
        # not name it. Neither adopt (claim_for_user binds a POOL slot) nor
        # "no bind" applies; an operator owns a named tenant's key drift.
        return OWNERSHIP_UNKNOWN, dict(ev, reason="named_only")
    if route_c and route_c != pool_c:
        # The route disagrees with the bind. That IS the split, even though
        # only one side reported a container.
        return OWNERSHIP_AMBIGUOUS, dict(ev, reason="route_disagrees_with_bind")
    if platform_container and platform_container == pool_c:
        # Everyone already agrees on the container; the 401 is a key fault on
        # a correctly-mapped pool member — the `keyless_pool` branch's job, and
        # this row would not be here unless the name were mis-classified.
        return OWNERSHIP_UNKNOWN, dict(ev, reason="platform_already_agrees")
    return OWNERSHIP_ADOPT, ev


async def _reconcile_named_401(user_id: str, container_name: str) -> tuple:
    """Gather the bridge's ownership evidence for one named-401 subject and
    classify it. Never raises: every read failure becomes "no evidence"."""
    prefix = str(user_id)[:8]
    slot = None
    try:
        slot = await bridge_lookup_user_slot(user_id)
    except Exception as e:
        # A raise means the bridge did not ANSWER (transport), which is not the
        # same as "no bind" — `bridge_lookup_user_slot` draws that distinction
        # deliberately and this branch must preserve it.
        logger.info(
            "[pool-reclaim] whois unavailable for %s: %s", prefix, type(e).__name__,
        )
    truth = await bridge_tenant_truth(prefix)
    return classify_named_401_ownership(
        platform_container=container_name or None, slot=slot, truth=truth,
    )


_IDENTITY_REPAIR_LOCKS: dict = {}


async def reconcile_agent_identity(user_id: str) -> str:
    """Ownership-aware repair for ONE user, callable off the chat path.

    Returns the outcome string (`adopt` once repaired, else `ambiguous`,
    `no_bind` or `unknown`). Single-flighted per user: two replicas hit this
    simultaneously on a first message, as they did all through 12 Sep.

    DIRECTION, and this is the invariant the chat path must never violate
    (L4 skeptic C4): this ADOPTS the container the BRIDGE says owns the user
    INTO the platform row. It never pushes the platform's stored key at
    whatever the route happens to hit. At 19:14:42 on 12 Sep the route pointed
    at pool-81 while the platform held the NAMED key; the other direction
    would have bound that named key onto a live pool member and manufactured
    the ambiguous-ownership state an operator then had to unpick by hand — a
    fix that creates the P0 it is fixing.

    Deliberately does NOT force-claim on any outcome. A force-claim binds an
    established account to a fresh empty slot when the bridge has no bind
    (R40), and for a correctly-mapped pool member with a stale key the sweep's
    own `keyless_pool` branch is already the repair, on a 15 s cadence.
    """
    uid = str(user_id)
    lock = _IDENTITY_REPAIR_LOCKS.get(uid)
    if lock is None:
        lock = _IDENTITY_REPAIR_LOCKS[uid] = asyncio.Lock()
    async with lock:
        try:
            from app.db.database import async_session_maker
            async with async_session_maker() as db:
                mc = (await db.execute(
                    select(ManagedContainer).where(
                        ManagedContainer.user_id == uid
                    )
                )).scalar_one_or_none()
            cname = (mc.container_name if mc else "") or ""
            outcome, ev = await _reconcile_named_401(uid, cname)
            if outcome == OWNERSHIP_ADOPT:
                if await _adopt_bridge_truth(uid, ev):
                    logger.warning(
                        "[identity-repair] user=%s adopted %s over platform "
                        "row %s", uid[:8], ev.get("pool_container"), cname,
                    )
                    return OWNERSHIP_ADOPT
                return OWNERSHIP_UNKNOWN
            logger.info(
                "[identity-repair] user=%s outcome=%s — not acting (%s)",
                uid[:8], outcome, ev,
            )
            return outcome
        except Exception:
            logger.warning(
                "[identity-repair] user=%s failed", uid[:8], exc_info=True,
            )
            return OWNERSHIP_UNKNOWN
        finally:
            if len(_IDENTITY_REPAIR_LOCKS) > 512:
                _IDENTITY_REPAIR_LOCKS.clear()


async def _adopt_bridge_truth(user_id: str, evidence: dict) -> bool:
    """Rewrite the platform row from the bridge's answer.

    DIRECTION IS THE WHOLE POINT (L4 skeptic C4). This adopts the ROUTED
    container's identity INTO the platform row. It never pushes the platform's
    stored belief at whatever the route happens to hit: on 12 Sep the route
    pointed at pool-81 while the platform held the NAMED key, and binding that
    named key onto a live pool member would have manufactured the very
    ambiguous-ownership state an operator then had to unpick by hand.
    """
    container = evidence.get("pool_container")
    if not container:
        return False
    slot = {
        "container_name": container,
        "db_name": evidence.get("pool_db"),
        "state": "ASSIGNED",
        "bound": True,
        "prefix": str(user_id)[:8],
    }
    try:
        url = await _adopt_discovered_bind(
            user_id, slot, bridge_confirmed_sole_owner=True,
        )
    except Exception:
        logger.warning(
            "[pool-reclaim] adopt from bridge truth failed for %s",
            str(user_id)[:8], exc_info=True,
        )
        return False
    return bool(url)


def _pool_list_lock() -> asyncio.Lock:
    # Created lazily: a module-level asyncio.Lock binds to the loop running at
    # import time, and this module is imported before the app's loop exists.
    global _POOL_LIST_LOCK
    if _POOL_LIST_LOCK is None:
        _POOL_LIST_LOCK = asyncio.Lock()
    return _POOL_LIST_LOCK


def _invalidate_pool_list_cache() -> None:
    _POOL_LIST_CACHE["ts"] = -1e9
    _POOL_LIST_CACHE["data"] = None


async def _pool_list(timeout_s: float) -> Optional[dict]:
    ttl = float(getattr(settings, "provision_discovery_cache_ttl_s", 2.0) or 0)
    now = time.monotonic()
    cached = _POOL_LIST_CACHE.get("data")
    if cached is not None and (now - _POOL_LIST_CACHE["ts"]) < ttl:
        return cached
    async with _pool_list_lock():
        now = time.monotonic()
        cached = _POOL_LIST_CACHE.get("data")
        if cached is not None and (now - _POOL_LIST_CACHE["ts"]) < ttl:
            return cached
        from app.services.docker_host_service import get_bridge_client
        client = await get_bridge_client()
        r = await client.get("/v1/pool/list", timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        _POOL_LIST_CACHE["data"] = data
        _POOL_LIST_CACHE["ts"] = time.monotonic()
        return data


def _is_adoptable(slot: Optional[dict]) -> bool:
    """Is this slot a COMPLETED bind we may adopt?

    `ASSIGNED` alone is not enough. The bridge stamps `assigned_user_id` and
    moves the slot to ASSIGNING/ASSIGNED around the bind, and `/v1/pool/whois`
    reports `bound` separately for exactly that reason — it is the stronger
    signal of the two. Adopting an unbound slot publishes an agent_url for a
    container that cannot answer yet. `/v1/pool/list` carries no `bound` key,
    so absence means "not contradicted", never "false".
    """
    if not slot:
        return False
    if slot.get("state") != "ASSIGNED":
        return False
    return slot.get("bound") is not False


def _whois_to_member(body: dict) -> Optional[dict]:
    """Translate `GET /v1/pool/whois`'s CLAIM-shaped body into the registry
    shape `_member_to_slot` reads.

    The two are genuinely different vocabularies and the whois one is NOT a
    member: `slot` is a string, the ids are `user_id`/`prefix` rather than
    `assigned_user_id`/`assigned_prefix`, and the port/db/container fields are
    the claim response's names. A previous version of this adapter guessed
    (`body.get("member") or body.get("slot") or body`) and picked the string
    "73", so every whois answer raised AttributeError — a crash reachable only
    once the route was deployed and this module's own tests were all green.
    """
    if not body or body.get("found") is False:
        return None
    if not body.get("container_name"):
        return None
    return {
        "slot": body.get("slot"),
        "container_name": body.get("container_name"),
        "docker_id": body.get("container_id"),
        "port": body.get("host_port"),
        "db_name": body.get("db_pool_slot"),
        "state": body.get("state"),
        "assigned_user_id": body.get("user_id"),
        "assigned_prefix": body.get("prefix"),
        "bound": body.get("bound"),
    }


def _member_to_slot(m: dict) -> dict:
    """Normalise one `/v1/pool/list` member into the shape the adopt path
    wants. Field names mirror bridge/pool_addon.py's members.json writer."""
    return {
        "slot": m.get("slot"),
        "container_name": m.get("container_name"),
        "container_id": m.get("docker_id"),
        "host_port": m.get("port"),
        "db_name": m.get("db_name"),
        "state": str(m.get("state") or "").upper(),
        "prefix": m.get("assigned_prefix"),
        # Only /v1/pool/whois reports this; a registry member carries no such
        # key and must therefore read None, not False. See `_is_adoptable`.
        "bound": m.get("bound"),
    }


async def bridge_lookup_user_slot(
    user_id: str, *, timeout_s: Optional[float] = None,
) -> Optional[dict]:
    """ADAPTER. The one place the bridge read is chosen — swap the endpoint
    here, nowhere else.

    Returns a normalised slot dict (see `_member_to_slot`) or None when the
    bridge does not know this user. Raises on transport failure so callers can
    tell "the bridge says no" from "the bridge did not answer": those have
    different meanings and only the first one is an answer.
    """
    if timeout_s is None:
        timeout_s = float(getattr(settings, "provision_discovery_read_timeout_s", 8) or 8)
    global _WHOIS_UNAVAILABLE
    route = (getattr(settings, "bridge_pool_whois_route", "") or "").strip()
    if route and not _WHOIS_UNAVAILABLE:
        try:
            from app.services.docker_host_service import get_bridge_client
            client = await get_bridge_client()
            r = await client.get(
                route, params={"user_id": str(user_id)}, timeout=timeout_s,
            )
            if r.status_code in (404, 405, 501):
                # Configured but not deployed. Degrade once, not every poll.
                _WHOIS_UNAVAILABLE = True
            else:
                r.raise_for_status()
                m = _whois_to_member(r.json() or {})
                if m is None:
                    # The bridge ANSWERED "no bind for this user". That is an
                    # answer, not a failure — do not fall through to the list.
                    return None
                return _member_to_slot(m)
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError:
            raise

    data = await _pool_list(timeout_s)
    if not data:
        return None
    uid = str(user_id)
    for m in (data.get("members") or []):
        if str(m.get("assigned_user_id") or "") == uid:
            return _member_to_slot(m)
    return None


# ── Adoption ─────────────────────────────────────────────────────────────

async def _bound_agent_url(db: AsyncSession, user_id: str) -> Optional[str]:
    """The user's agent_url IFF the platform's own row set is already
    consistent: a running container AND a config carrying both the url and the
    key. Anything less is not "bound" — a row without the key is the shape that
    401s every chat, which is what makes a partial adopt worse than none."""
    try:
        mc = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == user_id)
        )).scalar_one_or_none()
        if not mc or mc.status != "running":
            return None
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )).scalar_one_or_none()
        if not cfg or not cfg.agent_url or not cfg.agent_api_key:
            return None
        return cfg.agent_url
    except Exception:
        return None


def _is_named_container(name: Optional[str]) -> bool:
    n = name or ""
    return n.startswith("toup-agent-") and not n.startswith("toup-agent-pool-")


async def _adopt_discovered_bind(
    user_id: str, slot: dict, *, bridge_confirmed_sole_owner: bool = False,
) -> Optional[str]:
    """Adopt a bind the bridge has confirmed, through the SAME path
    `[pool-reclaim] healed stranded` uses: `claim_for_user`.

    Going through claim_for_user rather than writing the row from `slot` is not
    ceremony. `_build_bind_payload` mints `agent_api_key` with a compare-and-set
    and the claim that timed out never committed it — the bridge pushed key K to
    the agent while the platform kept NULL. A DB-only adopt would therefore
    produce a row that looks healthy and 401s on the first chat message. The
    bridge's per-user idempotent claim re-pushes the bind, so platform and agent
    end up holding the same key.

    Returns the agent_url on success, None otherwise. Never raises.
    """
    from app.db.database import async_session_maker
    try:
        async with async_session_maker() as db:
            already = await _bound_agent_url(db, user_id)
            if already:
                return already

            mc = (await db.execute(
                select(ManagedContainer).where(ManagedContainer.user_id == user_id)
            )).scalar_one_or_none()

            # NEVER move a named tenant onto a pool slot on the strength of a
            # registry read. The named tenant's database is toup_agent_<prefix>
            # and the pool slot's is the slot's own — re-homing is data loss
            # (see provision_container's PoolMemberSwapRefused for the mirror
            # image of this refusal). An operator owns that migration.
            if mc is not None and _is_named_container(mc.container_name) and (
                slot.get("container_name") != mc.container_name
            ) and not bridge_confirmed_sole_owner:
                logger.error(
                    "[discovery] user=%s bridge reports pool slot %s while the "
                    "platform row names %s. NOT adopting — a named tenant's data "
                    "lives in its own database. Reconcile by hand.",
                    str(user_id)[:8], slot.get("container_name"), mc.container_name,
                )
                return None
            if bridge_confirmed_sole_owner and mc is not None and (
                _is_named_container(mc.container_name)
            ):
                # The ONE way past the refusal above, and it is not a weakening
                # of it: the caller has asked the bridge for the prefix's
                # ownership and been told there is NO named container — the
                # platform row names a container that does not exist, so there
                # is no second database to lose. `classify_named_401_ownership`
                # is the only producer of that evidence and it answers
                # AMBIGUOUS (which never reaches here) the moment a named
                # container IS reported. Keep those two facts together: a
                # caller that sets this flag without the tenant-truth read is
                # re-opening the R40 data-loss door.
                logger.warning(
                    "[discovery] user=%s adopting bridge truth %s over a "
                    "platform row naming %s — the bridge reports NO named "
                    "container for this prefix",
                    str(user_id)[:8], slot.get("container_name"),
                    mc.container_name,
                )

            # A row wedged at 'provisioning' short-circuits every future
            # claim_for_user (its existing-row check returns it untouched), and
            # reclaim only unsticks it after 15 minutes. The bridge saying
            # ASSIGNED is proof the provisioning it describes is over, so the
            # unstick here is evidence-gated rather than clock-gated.
            if mc is not None and mc.status == "provisioning":
                mc.status = "error"
                mc.error_message = "[discovery] unstuck: bridge reports ASSIGNED"
                await db.commit()

            try:
                c = await claim_for_user(db, user_id, origin="discovery")
            except ProvisionDriveTaken:
                # The winner is mid-bind. Its own commit is the adoption; the
                # next discovery tick will see the converged rows. Racing it
                # here is exactly the second assignment we must not create.
                return None
            if c is None:
                return None
            cfg = (await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == user_id)
            )).scalar_one_or_none()
            return cfg.agent_url if cfg else None
    except Exception:
        logger.exception("[discovery] adopt failed for user=%s", str(user_id)[:8])
        return None


# Per-replica dedupe of discovery loops. Cross-replica duplicates are harmless:
# every adopt funnels through claim_for_user's per-user advisory lock and the
# bridge's own per-user idempotent claim, so the second one is a no-op.
_DISCOVERY_INFLIGHT: set = set()


def ensure_discovery(user_id: str, *, reason: str = "") -> None:
    """Start the bounded discovery loop for `user_id` unless one is already
    running in this process. Safe to call from anywhere, including from an
    exception handler. Never raises, never blocks."""
    if not getattr(settings, "provision_discovery_enabled", True):
        return
    uid = str(user_id)
    if uid in _DISCOVERY_INFLIGHT:
        return
    try:
        _DISCOVERY_INFLIGHT.add(uid)
        _spawn_bg(
            discover_and_adopt_bind(uid, reason=reason),
            name=f"discover:{uid[:8]}",
        )
    except Exception:
        _DISCOVERY_INFLIGHT.discard(uid)
        logger.warning("[discovery] could not start loop for %s", uid[:8])


async def discover_and_adopt_bind(
    user_id: str,
    *,
    reason: str = "",
    budget_s: Optional[float] = None,
    interval_s: Optional[float] = None,
) -> Optional[str]:
    """Poll the bridge for a bind that completed after we stopped listening,
    and adopt it. Bounded; the 180 s reconciler remains the backstop.

    Returns the agent_url once the platform's rows are consistent, else None.
    """
    uid = str(user_id)
    if budget_s is None:
        budget_s = float(getattr(settings, "provision_discovery_max_s", 120) or 120)
    if interval_s is None:
        interval_s = float(getattr(settings, "provision_discovery_interval_s", 5) or 5)
    read_timeout = float(getattr(settings, "provision_discovery_read_timeout_s", 8) or 8)
    from app.db.database import async_session_maker
    started = time.monotonic()
    signup_trace(uid, "discover_start", f"reason={reason} budget_s={int(budget_s)}",
                 origin="discovery")
    try:
        while (time.monotonic() - started) < budget_s:
            # Someone else (the reconciler, another replica, the original
            # claim finishing late) may have converged already.
            try:
                async with async_session_maker() as db:
                    url = await _bound_agent_url(db, uid)
                if url:
                    signup_trace(uid, "bound", "already_converged", origin="discovery")
                    return url
            except Exception:
                logger.warning("[discovery] DB check failed for %s", uid[:8])

            slot = None
            try:
                slot = await bridge_lookup_user_slot(uid, timeout_s=read_timeout)
            except Exception as e:
                # The bridge did not answer. That is not "no bind" — keep asking.
                logger.info(
                    "[discovery] user=%s lookup failed (%s) — will retry",
                    uid[:8], type(e).__name__,
                )

            if _is_adoptable(slot):
                # ASSIGNED only. `assigned_user_id` is stamped at ASSIGNING,
                # i.e. BEFORE the agent has been bound, so adopting on that
                # state would publish an agent_url for a container that cannot
                # yet answer.
                signup_trace(uid, "discovered", f"slot={slot.get('container_name')}",
                             origin="discovery")
                url = await _adopt_discovered_bind(uid, slot)
                if url:
                    signup_trace(uid, "adopted", f"slot={slot.get('container_name')}",
                                 origin="discovery")
                    return url
                # Adopt refused or failed — the next tick re-reads the truth.
                _invalidate_pool_list_cache()

            await asyncio.sleep(interval_s)
    finally:
        _DISCOVERY_INFLIGHT.discard(uid)
    signup_trace(uid, "discover_give_up", f"after_s={int(time.monotonic() - started)}",
                 origin="discovery")
    return None


async def try_adopt_stranded(db: AsyncSession, user_id: str) -> Optional[str]:
    """ONE bounded discovery attempt. Returns the user's agent_url if the
    platform now holds a consistent bind for them, else None.

    CONTRACT (Lane D / ws_chat_proxy may import this defensively):

    * **Bounded.** Total wall clock is capped by
      `settings.provision_adopt_budget_s` (default 3 s). If the work overruns
      it is NOT cancelled — it keeps running in the background and this call
      answers None. Cancelling mid-adopt would abort a DB commit or a bridge
      bind, which is worse than answering "not yet".
    * **Cheap in the common case.** The first thing it does is a pure-DB read;
      if the rows are already consistent it returns the url with zero bridge
      traffic. The bridge read it may then do is docker-free and shared with
      every other caller within a ~2 s window.
    * **Safe to call while a client is waiting**, and safe to call repeatedly:
      adoption funnels through `claim_for_user`'s per-user Postgres advisory
      lock and the bridge's per-user idempotent claim, so N concurrent calls
      produce at most one bind and exactly one managed_containers row.
    * **Never raises.** Every failure is None.
    * `db` is used only for the initial read. All writes happen on a private
      session so this can never commit or poison the caller's transaction.
    * Returns None when discovery is disabled
      (`settings.provision_discovery_enabled=False`).
    """
    if not getattr(settings, "provision_discovery_enabled", True):
        return None
    uid = str(user_id)
    try:
        url = await _bound_agent_url(db, uid)
        if url:
            return url
    except Exception:
        pass

    budget = float(getattr(settings, "provision_adopt_budget_s", 3.0) or 3.0)
    read_timeout = min(budget, float(
        getattr(settings, "provision_discovery_read_timeout_s", 8) or 8
    ))

    async def _once() -> Optional[str]:
        slot = await bridge_lookup_user_slot(uid, timeout_s=read_timeout)
        if not _is_adoptable(slot):
            return None
        signup_trace(uid, "discovered", "via=try_adopt_stranded", origin="discovery")
        url = await _adopt_discovered_bind(uid, slot)
        if url:
            signup_trace(uid, "adopted", "via=try_adopt_stranded", origin="discovery")
        return url

    task = _spawn_bg(_once(), name=f"adopt-once:{uid[:8]}")
    done, _pending = await asyncio.wait({task}, timeout=budget)
    if task in done:
        try:
            return task.result()
        except Exception:
            return None
    # Overran the budget. Leave it running — it holds the only in-flight
    # adopt for this user and cancelling it mid-write is the one genuinely
    # dangerous option.
    return None


# ── Cross-replica dedupe ─────────────────────────────────────────────────

async def try_take_provision_drive(db: AsyncSession, user_id: str) -> bool:
    """Non-blocking, cross-replica "am I the one driving provisioning for this
    user?" Returns True iff this transaction now owns the drive.

    A Postgres transaction-scoped advisory lock, which is the right primitive
    here for three reasons: it is visible to every replica (unlike the
    in-process `_ENV_PUSH_LOCKS`), it needs no schema and no TTL bookkeeping
    because it dies with its transaction, and a replica that crashes or is
    redeployed mid-bridge-call releases it automatically — so "retry after a
    lost response" is safe by construction rather than by a timeout guess.

    The caller MUST keep this session's transaction open for the duration of
    the work it guards; on commit or rollback the lock is released.

    Non-Postgres engines (the sqlite test paths) always answer True — there is
    only one process there, so there is nothing to dedupe.
    """
    try:
        from app.db.database import get_engine as _get_engine
        if _get_engine().dialect.name != "postgresql":
            return True
        from sqlalchemy import text as _text
        got = (await db.execute(
            _text("SELECT pg_try_advisory_xact_lock(hashtext(:k)::bigint)"),
            {"k": f"provision_drive:{user_id}"},
        )).scalar()
        return bool(got)
    except Exception:
        # A lock we cannot take is not a reason to skip provisioning. Fail
        # OPEN: a duplicate drive is idempotent, a skipped one is a dead agent.
        logger.warning(
            "[provision-drive] advisory lock check failed for %s — proceeding",
            str(user_id)[:8],
        )
        return True


async def try_take_claim_drive(db: AsyncSession, user_id: str) -> bool:
    """Non-blocking probe of the SAME advisory key `claim_for_user` holds for
    the length of a bind. False means a pool claim for this user is in flight.

    `try_take_provision_drive` is not enough on its own: it keys on
    `provision_drive:<uid>` while claim_for_user keys on `pool_claim:<uid>`,
    so nothing at all excluded a soul-save prewarm from driving
    POST /v1/tenants for a user the pool was mid-bind on. That is the door the
    2026-09-06 incident actually used — the two PREWARM-STARTs at 18:17:39
    reached the bridge's create_tenant at 18:18:00, eleven seconds before the
    pool bind completed.

    Non-Postgres engines answer True: one process, nothing to serialise.
    """
    try:
        from app.db.database import get_engine as _get_engine
        if _get_engine().dialect.name != "postgresql":
            return True
        from sqlalchemy import text as _text
        got = (await db.execute(
            _text("SELECT pg_try_advisory_xact_lock(hashtext(:k)::bigint)"),
            {"k": _claim_drive_key(user_id)},
        )).scalar()
        return bool(got)
    except Exception:
        # Fail OPEN, same as try_take_provision_drive: a duplicate drive is
        # idempotent on the bridge, a skipped one is a user with no agent.
        logger.warning(
            "[provision-drive] claim-lock probe failed for %s — proceeding",
            str(user_id)[:8],
        )
        return True


# ── The fast stranded pass ───────────────────────────────────────────────

async def _recently_stranded_user_ids(db: AsyncSession, limit: int = 10) -> list:
    """Users who signed up in the last `stranded_fast_window_min` minutes and
    still have no running container. Deliberately much narrower than
    `_stranded_user_ids`: this runs every 15 s, so it must be a handful of rows
    and it must find nothing in the steady state (in which case the pass costs
    one indexed query and ZERO bridge calls)."""
    from datetime import timedelta as _td
    from sqlalchemy import and_, or_
    from app.db.models import User

    cutoff = datetime.utcnow() - _td(
        minutes=int(getattr(settings, "stranded_fast_window_min", 30) or 30)
    )
    rows = await db.execute(
        select(User.id)
        .outerjoin(AgentConfig, AgentConfig.user_id == User.id)
        .outerjoin(ManagedContainer, ManagedContainer.user_id == User.id)
        .where(
            User.is_active == True,  # noqa: E712
            User.id.notin_(_deletion_in_flight_user_ids()),
            User.created_at >= cutoff,
            AgentConfig.hosting_mode == "managed",
            or_(
                ManagedContainer.id.is_(None),
                ManagedContainer.status != "running",
            ),
        )
        .order_by(User.created_at.desc())
        .limit(limit)
    )
    return [str(uid) for uid in rows.scalars().all()]


async def reclaim_stranded_fast() -> dict:
    """15 s sub-tick of the container reconciler: for RECENT signups only, ask
    the bridge whether a bind already exists and adopt it.

    This is not a second reconciliation system — every adopt goes through
    `claim_for_user`, exactly as `reclaim_stranded_users` does, so the two
    racing produces one bind and one row. What it changes is latency: on
    2026-09-06 the truth existed at 18:18:11 and the 180 s scan reported it at
    18:19:12. Never raises.
    """
    summary: dict = {"candidates": 0, "adopted": 0}
    if not getattr(settings, "provision_discovery_enabled", True):
        return {"skipped": "discovery_disabled"}
    if not getattr(settings, "use_container_pool", False):
        return {"skipped": "pool_disabled"}
    try:
        from app.db.database import async_session_maker
        async with async_session_maker() as db:
            candidates = await _recently_stranded_user_ids(db)
        summary["candidates"] = len(candidates)
        if not candidates:
            return summary
        read_timeout = float(
            getattr(settings, "provision_discovery_read_timeout_s", 8) or 8
        )
        for uid in candidates:
            try:
                slot = await bridge_lookup_user_slot(uid, timeout_s=read_timeout)
            except Exception as e:
                logger.info(
                    "[stranded-fast] lookup failed (%s) — 180s scan still covers it",
                    type(e).__name__,
                )
                break  # bridge is not answering; do not hammer it per user
            if _is_adoptable(slot):
                signup_trace(uid, "discovered", "via=stranded_fast", origin="discovery")
                if await _adopt_discovered_bind(uid, slot):
                    summary["adopted"] += 1
                    signup_trace(uid, "adopted", "via=stranded_fast", origin="discovery")
                    logger.warning(
                        "[stranded-fast] adopted late bind user=%s -> %s",
                        uid[:8], slot.get("container_name"),
                    )
    except Exception:
        logger.exception("[stranded-fast] pass failed")
    return summary
