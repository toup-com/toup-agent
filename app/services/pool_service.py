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
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import AgentConfig, ManagedContainer

logger = logging.getLogger(__name__)


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
    # Generate the per-tenant agent_api_key here. The bridge would
    # do this in the slow path; for the pool path the platform owns
    # generation so the key lands in our DB BEFORE the bind succeeds.
    # If claim fails after this, the key is harmless (bridge never
    # gets it).
    agent_api_key = agent_config.agent_api_key or secrets.token_urlsafe(48)

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


async def claim_for_user(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
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
        await db.execute(
            _text("SELECT pg_advisory_xact_lock(hashtext(:k)::bigint)"),
            {"k": f"pool_claim:{user_id}"},
        )

    # Existing container check — same idempotency the slow path has. Re-read
    # AFTER the lock so a loser that just blocked sees the winner's freshly
    # committed row and returns it instead of double-binding.
    existing = (await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )).scalar_one_or_none()
    if existing and existing.status in ("running", "provisioning"):
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
        return None
    except httpx.HTTPError as e:
        logger.warning("[pool_service] bridge unreachable: %s", e)
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


async def release_pool_member(prefix: Optional[str] = None, user_id: Optional[str] = None) -> bool:
    """Tell the bridge a user has been deleted / churned.

    Bridge marks the pool member DRAINING; the reconciler completes the
    destroy on its next tick. The DB role is revoked so any leftover
    connection attempts fail; the DB itself is preserved (cleanup is
    out-of-band). Slot is reusable once the reconciler reaps it; the
    next allocation creates a fresh DB role + password.

    Returns True if the bridge accepted the release (including the
    "no pool member matches" case — that's idempotent success).
    """
    if not prefix and not user_id:
        return False
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
                return False
            data = resp.json()
            logger.info(
                "[pool_service] pool release: found=%s slot=%s prefix=%s user=%s",
                data.get("found"), data.get("slot"), prefix, str(user_id)[:8] if user_id else "",
            )
            return True
    except Exception as e:
        logger.warning("[pool_service] bridge release unreachable: %s", e)
        return False


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
        from app.db.database import async_session_maker
        from app.services.docker_host_service import provision_container
        async with async_session_maker() as heal_db:
            await provision_container(heal_db, user_id, recreate=True)
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
                asyncio.create_task(_verify_and_heal_pool_claim(user_id))
            except Exception:
                # create_task should never fail in an async context, but if it
                # somehow does, the periodic reconciler still covers us.
                logger.warning("[pool_service] could not spawn pool-heal guard for %s", user_id[:8])
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
