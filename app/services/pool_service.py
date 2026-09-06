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
    db: AsyncSession, user_id: str, *, force: bool = False
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
                {"k": f"pool_claim:{user_id}"},
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
            signup_trace(user_id, "claim_contended", type(_le).__name__)
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
                return None
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(
            "[pool_service] bridge claim returned %s: %s",
            e.response.status_code, e.response.text[:200],
        )
        signup_trace(user_id, "claim_timeout", f"http_{e.response.status_code}")
        # A 5xx can still leave a completed bind behind (the bind and the
        # response are not the same event), so ask rather than assume.
        if e.response.status_code >= 500:
            ensure_discovery(user_id, reason="claim_5xx")
        return None
    except httpx.HTTPError as e:
        # `str(httpx.ReadTimeout())` is the EMPTY STRING, which is how the
        # 2026-09-06 trail came to read "bridge unreachable: " and name
        # nothing at all. repr() carries the class; `%r` on the exception
        # would too. Same fix already applied at
        # docker_host_service._update_container_env's retry loop.
        logger.warning("[pool_service] bridge unreachable: %r", e)
        signup_trace(user_id, "claim_timeout", type(e).__name__)
        # THE FIX. A timed-out claim is not a failed claim: on 2026-09-06 the
        # bridge finished this exact bind 2 s after we stopped listening.
        ensure_discovery(user_id, reason="claim_timeout")
        return None

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
    signup_trace(user_id, "claim_ok", f"slot={container_name}")
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

    This polls the agent's real `/agent/health` for a short budget (pool
    members are pre-warmed, so a healthy one passes in a few seconds). If it
    never comes up, we re-provision via the reliable slow path
    (`provision_container(recreate=True)` — the same call the reconciler/
    backfill use, which creates a real `toup-agent-{prefix}` container AND
    records container_id). Fire-and-forget: never raises into the caller, so
    it can't affect the signup request. The reconciler remains the durable
    backstop for the case where a platform redeploy kills this task mid-flight.
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
            "— re-provisioning via slow path (recreate=True)",
            user_id[:8], budget_s,
        )
        # HEAL THE POOL MEMBER IN PLACE FIRST.
        #
        # `provision_container(recreate=True)` binds `toup_agent_<prefix>`,
        # while a pool member's data lives in the slot's own
        # `toup_agent_feedNNNN` — so the "reliable slow path" this used to take
        # unconditionally is also the path that empties an account (R40; see
        # `docker_host_service.PoolMemberSwapRefused`). A wedged member is
        # usually just wedged: `/v1/pool/restart-member` restarts the container
        # and re-applies its bind, keeping the database it already has.
        from app.services.docker_host_service import provision_container, restart_container
        from app.db.database import async_session_maker

        async with async_session_maker() as heal_db:
            mc = await _load_container(heal_db, user_id)
            is_pool = bool(mc and (mc.container_name or "").startswith("toup-agent-pool-"))
            if is_pool:
                logger.warning(
                    "[pool-heal] restarting pool member %s for user=%s in place "
                    "(a named recreate would bind an empty toup_agent_%s)",
                    mc.container_name, user_id[:8], user_id[:8],
                )
                try:
                    await restart_container(heal_db, user_id)
                    await heal_db.commit()
                except Exception:
                    logger.exception("[pool-heal] pool restart failed user=%s", user_id[:8])
                # Give the restart the same budget the claim got, then re-check.
                deadline2 = asyncio.get_event_loop().time() + budget_s
                while asyncio.get_event_loop().time() < deadline2:
                    if await _is_agent_actually_healthy(user_id):
                        logger.info("[pool-heal] user=%s recovered after an in-place restart",
                                    user_id[:8])
                        return
                    await asyncio.sleep(interval_s)

            # Still unreachable. The named recreate is the last resort, and it
            # is only safe while the claim is FRESH — at that point the user
            # has never sent a message and the slot database holds nothing to
            # strand. An established member reaching here is exactly the
            # data-loss case, so it is left for an operator instead.
            if is_pool and not _claim_is_fresh(mc):
                logger.error(
                    "[pool-heal] REFUSING the named recreate for user=%s: pool member %s "
                    "is not a fresh claim, and re-provisioning would bind an empty "
                    "toup_agent_%s and strand the slot's data. It has been restarted in "
                    "place; operator action required if it is still unreachable.",
                    user_id[:8], mc.container_name, user_id[:8],
                )
                return
            # `allow_pool_swap=is_pool`, not a bare True: a NAMED tenant taking
            # this path never needed the override, and passing it anyway states
            # an intent the call does not have.
            await provision_container(
                heal_db, user_id, recreate=True, allow_pool_swap=is_pool,
            )
            await heal_db.commit()
        logger.info("[pool-heal] re-provisioned user=%s to a reachable container", user_id[:8])
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
    `_verify_and_heal_pool_claim` guard so a member that's stale/unreachable
    self-heals within ~30s instead of leaving a brand-new user stuck on their
    first message (see that function). Registration latency is unaffected —
    the guard runs in the background.
    """
    try:
        c = await claim_for_user(db, user_id)
        if c is not None:
            try:
                _spawn_bg(_verify_and_heal_pool_claim(user_id))
            except Exception:
                # create_task should never fail in an async context, but if it
                # somehow does, the periodic reconciler still covers us.
                logger.warning("[pool_service] could not spawn pool-heal guard for %s", user_id[:8])
            return True
    except ProvisionDriveTaken:
        # Somebody else owns this user's provisioning. Falling through to
        # schedule_prewarm here would be the 2026-09-06 duplicate: a SECOND
        # driver taking the cold NAMED path (POST /v1/tenants) for a user the
        # pool is already binding. Observe instead — claim_for_user has already
        # started discovery, which adopts whatever the winner produces.
        logger.info(
            "[pool_service] provisioning for %s is already being driven — observing",
            user_id[:8],
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

# Authenticated-sweep strike ledger: consecutive 5xx/timeout probe failures
# per user id. In-memory on purpose — a platform redeploy resets strikes,
# which only delays a restart by one 180s tick. Two consecutive sick ticks
# (≈6 min genuinely unresponsive) before restarting keeps one slow request
# or GC pause from bouncing a healthy container.
_PROBE_STRIKES: dict = {}
PROBE_STRIKES_BEFORE_RESTART = 2
# Restart at most N containers per tick — a sweep must never become a
# restart storm (mass failures are guarded separately by the quorum check).
RESTARTS_PER_TICK = 2


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
                    c = await claim_for_user(udb, uid)
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
    #                  healthy). Named: restart via bridge (keys re-enter from
    #                  .env at boot) + alert, because named 401s mean key
    #                  drift, which should be impossible.
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

        keyless_pool: list = []
        keyless_named: list = []
        sick: list = []
        for uid, cname, status, err in results:
            is_pool = cname.startswith("toup-agent-pool-")
            if status == 200:
                _PROBE_STRIKES.pop(uid, None)
                continue
            if status in (401, 403):
                _PROBE_STRIKES.pop(uid, None)
                (keyless_pool if is_pool else keyless_named).append((uid, cname))
                continue
            if status is not None and status < 500:
                # 404 etc. — routing, owned by the bridge route reconciler.
                _PROBE_STRIKES.pop(uid, None)
                continue
            if mass_failure and status is None:
                continue  # platform-egress blip; no per-agent strikes
            strikes = _PROBE_STRIKES.get(uid, 0) + 1
            _PROBE_STRIKES[uid] = strikes
            if strikes >= PROBE_STRIKES_BEFORE_RESTART:
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
                    c = await claim_for_user(udb, uid, force=True)
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

        # Keyless NAMED tenants → restart (env re-injects keys at boot).
        for uid, cname in keyless_named[:RESTARTS_PER_TICK]:
            ok = await _restart_sick_container(uid, cname)
            summary["restarted"] = summary.get("restarted", 0) + (1 if ok else 0)
            logger.warning(
                "[pool-reclaim] named tenant 401 user=%s container=%s restart=%s",
                uid[:8], cname, ok,
            )

        # Sick (2 consecutive 5xx/timeout ticks) → restart via bridge.
        for uid, cname, why in sick[:RESTARTS_PER_TICK]:
            ok = await _restart_sick_container(uid, cname)
            _PROBE_STRIKES.pop(uid, None)
            summary["restarted"] = summary.get("restarted", 0) + (1 if ok else 0)
            logger.warning(
                "[pool-reclaim] restarted sick agent user=%s container=%s "
                "(reason=%s) ok=%s", uid[:8], cname, why, ok,
            )
    except Exception:
        logger.exception("[pool-reclaim] authenticated sweep failed")

    # Law 4: healing is good; REPEATED healing means something upstream broke.
    # Tell the operator whenever a self-heal actually fired (rate-limited so a
    # flapping loop can't flood the chat).
    try:
        healed = {
            k: v for k, v in summary.items()
            if k in ("claimed", "rebound", "restarted", "keyless", "sick", "failed") and v
        }
        if healed:
            from app.services.alerting import send_infra_alert
            await send_infra_alert(
                "pool-selfheal", "warning",
                f"Self-heal fired: {healed}. Users were healed automatically — "
                "if this repeats, something upstream is broken.",
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


# user prefix -> monotonic clock at registration (or at first sighting when
# registration is not in this process's history). Bounded FIFO: this is a log
# convenience, never a source of truth.
_TRACE_T0: dict = {}
_TRACE_T0_MAX = 512


def _trace_origin(user_id: str) -> float:
    key = str(user_id)[:8]
    t0 = _TRACE_T0.get(key)
    if t0 is None:
        if len(_TRACE_T0) >= _TRACE_T0_MAX:
            # FIFO evict — dicts preserve insertion order.
            for k in list(_TRACE_T0)[: _TRACE_T0_MAX // 4]:
                _TRACE_T0.pop(k, None)
        t0 = time.monotonic()
        _TRACE_T0[key] = t0
    return t0


def seed_signup_trace(user_id: str) -> None:
    """Stamp t0 for a user's signup trace. Call from the registration path so
    every later `elapsed_ms` is measured from registration rather than from
    whenever this process first heard of them."""
    key = str(user_id)[:8]
    _TRACE_T0.pop(key, None)
    _trace_origin(key)


def signup_trace(user_id: str, hop: str, detail: str = "") -> None:
    """One structured line per provisioning hop, keyed by the user's 8-hex
    prefix. Deliberately boring: no secrets, no message content, no PII — a
    prefix, a hop name, an elapsed_ms and a short mechanical detail.

    Hops: registered | prewarm_start | claim_ok | claim_timeout |
          discovered | adopted | bound | ready
    (plus a few mechanical ones: prewarm_dedupe, discover_start, discover_give_up,
     config_sync_deferred).
    """
    try:
        pfx = str(user_id)[:8]
        logger.info(
            "[signup-trace] user=%s hop=%s elapsed_ms=%d detail=%s",
            pfx, hop, _t_ms(_trace_origin(pfx)), (detail or "")[:120],
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


async def _adopt_discovered_bind(user_id: str, slot: dict) -> Optional[str]:
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
            ):
                logger.error(
                    "[discovery] user=%s bridge reports pool slot %s while the "
                    "platform row names %s. NOT adopting — a named tenant's data "
                    "lives in its own database. Reconcile by hand.",
                    str(user_id)[:8], slot.get("container_name"), mc.container_name,
                )
                return None

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
                c = await claim_for_user(db, user_id)
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
    signup_trace(uid, "discover_start", f"reason={reason} budget_s={int(budget_s)}")
    try:
        while (time.monotonic() - started) < budget_s:
            # Someone else (the reconciler, another replica, the original
            # claim finishing late) may have converged already.
            try:
                async with async_session_maker() as db:
                    url = await _bound_agent_url(db, uid)
                if url:
                    signup_trace(uid, "bound", "already_converged")
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
                signup_trace(uid, "discovered", f"slot={slot.get('container_name')}")
                url = await _adopt_discovered_bind(uid, slot)
                if url:
                    signup_trace(uid, "adopted", f"slot={slot.get('container_name')}")
                    return url
                # Adopt refused or failed — the next tick re-reads the truth.
                _invalidate_pool_list_cache()

            await asyncio.sleep(interval_s)
    finally:
        _DISCOVERY_INFLIGHT.discard(uid)
    signup_trace(uid, "discover_give_up", f"after_s={int(time.monotonic() - started)}")
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
        signup_trace(uid, "discovered", "via=try_adopt_stranded")
        url = await _adopt_discovered_bind(uid, slot)
        if url:
            signup_trace(uid, "adopted", "via=try_adopt_stranded")
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
                signup_trace(uid, "discovered", "via=stranded_fast")
                if await _adopt_discovered_bind(uid, slot):
                    summary["adopted"] += 1
                    signup_trace(uid, "adopted", "via=stranded_fast")
                    logger.warning(
                        "[stranded-fast] adopted late bind user=%s -> %s",
                        uid[:8], slot.get("container_name"),
                    )
    except Exception:
        logger.exception("[stranded-fast] pass failed")
    return summary
