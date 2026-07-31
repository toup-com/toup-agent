"""
Managed Docker Host Service — Phase 3 rewrite.

Tenant lifecycle operations route through the typed mTLS provisioning bridge
at settings.bridge_url (e.g. https://bridge.agents.toup.ai). Platform never
SSHes into the Docker host.

Public API is stable across the Phase 2 → Phase 3 transition. Callers
(`agent_setup.py`, `admin/users.py`, `admin/infrastructure.py`, etc.) use
the same function signatures; only the internals changed.

Gone in Phase 3:
  - `_run_ssh` — replaced by `_bridge_client()` (httpx with mTLS)
  - `_get_ssh_key_file` — SSH key material no longer needed
  - `upgrade_container`, `upgrade_all_containers` — rollouts are orchestrated
    by `rollout_service.py` now; this module no longer owns them
  - `_build_env` — bridge's `create_tenant` builds the env internally from
    the agent_config payload we send it (see AgentEnvContract for the shape)

See docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md §4 for the platform ↔
bridge protocol rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
import tempfile
import time
from datetime import datetime
from typing import Optional

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ManagedContainer, AgentConfig, Rollout

logger = logging.getLogger(__name__)


# ─── Image-tag resolution for new tenants ────────────────────────


async def _latest_known_good_image_tag(db: AsyncSession) -> Optional[str]:
    """Image_tag of the most recent successfully-completed rollout.

    The "last-good SHA" already running in the tenant fleet — what new
    tenants should be provisioned onto. Skips `aborted_canary_failed`,
    `cancelled`, `pending`, and `running` rollouts; `complete` and
    `complete_with_failures` both mean "this SHA passed canary + reached
    the fleet" — with_failures still put MOST of the fleet on the tag
    (per-tenant failures are re-driven by the convergence sweep), so
    excluding it would pin new provisions to an older SHA than the fleet.

    Returns None only on a fresh platform install with no rollout
    history. Callers fall back to settings.docker_agent_image in that
    case (which is intentionally a safe sentinel, not a deployable tag).

    See: 2026-04-26 Alireza incident — first-time bundle subscriber's
    provision_container call hit the bridge with `toup-agent:latest` (the
    sentinel) → 404 because the rollout pipeline only publishes SHA tags.
    """
    result = await db.execute(
        select(Rollout.image_tag)
        .where(Rollout.status.in_(["complete", "complete_with_failures"]))
        .order_by(Rollout.completed_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ─── Bridge httpx client (mTLS) ───────────────────────────────────

_cert_tmpfiles: dict[str, str] = {}


def _write_cert_tmpfile(name: str, pem: str) -> str:
    """Write a PEM-encoded cert/key to a private tempfile once per process.
    Returns the path. httpx requires file paths for ssl_context loading.

    The files live in /tmp with mode 600 and are never deleted — they die
    with the platform process. Rewriting per-request would create thousands
    of tempfiles under sustained load.
    """
    if name in _cert_tmpfiles:
        return _cert_tmpfiles[name]
    fd, path = tempfile.mkstemp(prefix=f"bridge_{name}_", suffix=".pem")
    with os.fdopen(fd, "w") as f:
        f.write(pem)
    os.chmod(path, 0o600)
    _cert_tmpfiles[name] = path
    return path


_BRIDGE_CLIENT: Optional[httpx.AsyncClient] = None
_BRIDGE_CLIENT_LOCK = asyncio.Lock()


def _build_bridge_client(timeout_s: Optional[int] = None) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient configured for mTLS to the bridge.

    Uses cert material from settings.bridge_{ca,client}_{cert,key}. Those
    are PEM strings stored as Railway env vars (multi-line via Raw Editor).

    Most callers should use :func:`get_bridge_client` instead — the
    module-level singleton holds a connection pool open for the life of
    the process, which avoids paying the TLS+mTLS handshake (~30–50 ms)
    on every bridge call. Use this raw builder only for the rare
    custom-timeout case where you need a fresh client.
    """
    if not settings.bridge_url:
        raise RuntimeError(
            "settings.bridge_url is empty — Phase 3 platform requires the bridge to be configured"
        )
    if not (settings.bridge_ca_cert and settings.bridge_client_cert and settings.bridge_client_key):
        raise RuntimeError(
            "bridge mTLS certs missing from settings — set bridge_ca_cert, bridge_client_cert, bridge_client_key"
        )

    cert_path = _write_cert_tmpfile("client_cert", settings.bridge_client_cert)
    key_path = _write_cert_tmpfile("client_key", settings.bridge_client_key)

    # TLS architecture has TWO cert surfaces:
    #   1. server cert — Caddy presents a Let's Encrypt wildcard for
    #      *.agents.toup.ai. We verify it via the SYSTEM CA bundle.
    #   2. client cert auth — Caddy requires a client cert signed by our
    #      bridge CA (bridge_ca_cert). We PRESENT bridge_client_cert/key;
    #      the server side verifies against bridge_ca_cert on the VPS.
    # So we do NOT need bridge_ca_cert in the platform client's SSL context
    # at all — using it as cafile would reject the Let's Encrypt server cert
    # with "unable to get local issuer certificate". The CA only lives on
    # the server side, burned into /etc/caddy/bridge-ca.crt.
    ctx = ssl.create_default_context()
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)

    return httpx.AsyncClient(
        base_url=settings.bridge_url.rstrip("/"),
        verify=ctx,
        timeout=httpx.Timeout(timeout_s or settings.bridge_request_timeout_s),
    )


async def get_bridge_client() -> httpx.AsyncClient:
    """Return the shared mTLS bridge AsyncClient (TKT-LAT-006).

    The client is constructed lazily on first use and kept open for the
    life of the process so subsequent bridge calls reuse the same
    connection pool — eliminating the ~30–50 ms TLS+mTLS handshake that
    the previous per-call _bridge_client pattern paid on every
    provision, prewarm, rollout, and health check.

    Callers should NOT close the returned client. Pass per-request
    timeouts via the ``timeout=`` kwarg on the individual call instead
    of building a fresh client.
    """
    global _BRIDGE_CLIENT
    if _BRIDGE_CLIENT is not None and not _BRIDGE_CLIENT.is_closed:
        return _BRIDGE_CLIENT
    async with _BRIDGE_CLIENT_LOCK:
        if _BRIDGE_CLIENT is None or _BRIDGE_CLIENT.is_closed:
            t = time.perf_counter()
            _BRIDGE_CLIENT = _build_bridge_client()
            logger.info(
                "[PERF] bridge_client_init %.1fms",
                (time.perf_counter() - t) * 1000,
            )
    return _BRIDGE_CLIENT


async def _close_bridge_client_for_test() -> None:
    """Test hook: dispose the singleton so a fresh one is built next call."""
    global _BRIDGE_CLIENT
    if _BRIDGE_CLIENT is not None:
        try:
            await _BRIDGE_CLIENT.aclose()
        except Exception:
            pass
    _BRIDGE_CLIENT = None


# Legacy alias: existing call sites import _bridge_client. Keep it as a
# thin wrapper around the singleton so the diff is bounded and the
# `async with` pattern continues to work — __aexit__ on the singleton's
# AsyncClient.__aenter__ context is a no-op close because we override
# __aexit__ in this returned wrapper.
class _BridgeClientLease:
    """Async context manager that hands out the shared client without
    closing it on exit. Drops the `async with _bridge_client() as c:`
    pattern in place without churning every caller."""

    def __init__(self, timeout_s: Optional[int] = None):
        self._timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None
        # Fresh-client mode: caller asked for a non-default timeout, so
        # we cannot hand out the shared pool. Build a one-shot client
        # and close it on exit.
        self._owned = timeout_s is not None

    async def __aenter__(self) -> httpx.AsyncClient:
        if self._owned:
            self._client = _build_bridge_client(self._timeout_s)
        else:
            self._client = await get_bridge_client()
        return self._client

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owned and self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass


def _bridge_client(timeout_s: Optional[int] = None) -> _BridgeClientLease:
    """Backward-compatible entrypoint — returns a lease wrapper.

    Without ``timeout_s``: yields the shared singleton (no close on exit).
    With ``timeout_s``: yields a fresh client and closes it on exit.
    """
    return _BridgeClientLease(timeout_s=timeout_s)


# ─── Lifecycle operations (stable signatures) ────────────────────


def _agent_config_to_bridge_body(agent_config: Optional[AgentConfig]) -> dict:
    """Flatten an AgentConfig row into the shape bridge's CreateTenantReq expects.

    Whitelist-based: every field a tenant container needs in its .env
    must be explicitly listed. Forgetting a field is silent — the
    bridge writes the .env without it and the agent boots with that
    env var unset (e.g. WHATSAPP_MODE missing → BaileysWhatsAppChannel
    never spawns even though the platform DB has whatsapp_mode='qr_link').

    Trip wires that previously caused incidents:
    * llm_mode missing → bundle subscribers got "API key invalid"
      (matin incident, 2026-04-27)
    * whatsapp_* fields missing → QR-mode rollout failed silently
      (this incident, 2026-04-30)

    Slack tokens, service keys (brave, elevenlabs) and all WhatsApp
    fields are added here — they were latent bugs from earlier work
    that hadn't surfaced because nobody was using those features on
    managed-mode containers yet.
    """
    if not agent_config:
        return {}
    out: dict = {}
    # API keys + agent identity
    for field in (
        "openai_api_key", "anthropic_api_key", "google_api_key",
        "mistral_api_key", "xai_api_key", "deepseek_api_key",
        "agent_color", "agent_model",
    ):
        val = getattr(agent_config, field, None)
        if val:
            out[field] = val
    # Channel tokens + tunnel
    for field in (
        "telegram_bot_token", "discord_bot_token",
        "slack_bot_token", "slack_app_token",
    ):
        val = getattr(agent_config, field, None)
        if val:
            out[field] = val
    # connect_token lives as agent_config.connect_token but the bridge names
    # the env var TOUP_TOKEN; the bridge's CreateTenantReq exposes it as
    # `connect_token`.
    if getattr(agent_config, "connect_token", None):
        out["connect_token"] = agent_config.connect_token
    # llm_mode tells the bridge to write LLM_MODE into the tenant container
    # env. Without it the agent falls back to BYOK and bundle subscribers
    # see "API key invalid" on every chat. The matin incident on 2026-04-27
    # was caused by this field never being forwarded.
    if getattr(agent_config, "llm_mode", None):
        out["llm_mode"] = agent_config.llm_mode
    # Service keys (Brave search, ElevenLabs TTS) — TENANT columns only.
    # The PLATFORM Brave key is deliberately no longer forwarded here: it now
    # lives only in the search gateway (app/api/search_proxy.py), which every
    # container reaches with the TOUP_TOKEN it already holds, so the fleet
    # secret exists in one process instead of 42 .env files.
    # Caveat worth knowing before someone "fixes" a tenant's missing results:
    # a tenant's OWN brave_api_key still ships but is currently INERT — the
    # agent's direct-Brave call was removed with the gateway, so nothing in
    # the container reads BRAVE_API_KEY any more.
    for field in ("brave_api_key", "elevenlabs_api_key"):
        val = getattr(agent_config, field, None)
        if val:
            out[field] = val
    # WhatsApp (Cloud API / Path A — BYOA Meta App).
    for field in (
        "whatsapp_phone_number_id", "whatsapp_access_token",
        "whatsapp_verify_token", "whatsapp_app_secret",
    ):
        val = getattr(agent_config, field, None)
        if val:
            out[field] = val
    # WhatsApp (QR-link / Path C — Baileys via neonize). whatsapp_mode
    # is the gate that decides which channel adapter agent_main.py
    # spawns at boot; without it in the .env, the BaileysWhatsAppChannel
    # never starts even after the user toggles QR mode in Settings.
    if getattr(agent_config, "whatsapp_mode", None):
        out["whatsapp_mode"] = agent_config.whatsapp_mode
    if getattr(agent_config, "whatsapp_self_e164", None):
        out["whatsapp_self_e164"] = agent_config.whatsapp_self_e164
    if getattr(agent_config, "whatsapp_baileys_allowlist", None):
        out["whatsapp_baileys_allowlist"] = agent_config.whatsapp_baileys_allowlist
    # Sub-agent spawning kill-switch (alembic 057). Forwarded ONLY
    # when True so existing tenants with the default-False column
    # don't get a redundant SUBAGENT_SPAWNING_ENABLED=false line in
    # their .env (the agent's pydantic-settings default is already
    # False). Flipping this column for a tenant + recreate-container
    # writes SUBAGENT_SPAWNING_ENABLED=true into the .env so spawn
    # routes to the new orchestrator instead of legacy.
    if getattr(agent_config, "subagent_spawning_enabled", False):
        out["subagent_spawning_enabled"] = True
    return out


async def provision_container(
    db: AsyncSession,
    user_id: str,
    agent_config: Optional[AgentConfig] = None,
    recreate: bool = False,
    override_image_tag: Optional[str] = None,
) -> ManagedContainer:
    """Provision a new Docker container for a user's agent, via the bridge.

    Idempotent for new provisions: if a running container already exists for
    this user, returns the existing row WITHOUT re-calling the bridge.

    Pass `recreate=True` to force the bridge POST /v1/tenants call even when
    a container is already running. The bridge's create flow is itself
    idempotent (force-removes the stale container, rewrites .env, re-runs
    with fresh env). This is the path used by `update_container_env` after
    bundle activation flips llm_mode / connect_token, since the existing
    container started with the pre-activation env and won't pick up the
    new values otherwise (Arshia incident, 2026-04-28).
    """
    prefix = user_id[:8]
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    existing = result.scalar_one_or_none()
    if existing and existing.status in ("running", "provisioning") and not recreate:
        return existing

    # Fetch agent_config if caller didn't pass one
    if not agent_config:
        result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        agent_config = result.scalar_one_or_none()

    # Resolve image_tag from the latest successful rollout. The settings
    # default ("toup-agent:latest") is only a fresh-install sentinel — it
    # is NOT published to GHCR, so falling through to it for new tenants
    # is what caused the 2026-04-26 Alireza incident.
    #
    # Override priority: explicit ``override_image_tag`` (operator
    # recovery) > pin_image_tag on the existing managed_containers row
    # > latest known-good rollout > settings sentinel.
    if override_image_tag:
        image_tag = override_image_tag
    elif existing and existing.pin_image_tag:
        image_tag = existing.pin_image_tag
    else:
        image_tag = await _latest_known_good_image_tag(db) or settings.docker_agent_image
        # The sentinel ("toup-agent:latest" / any ":latest" tag) is
        # NOT a deployable tag — the rollout pipeline only publishes
        # SHA tags. Falling through to it produced the 2026-04-26
        # Alireza incident (bridge 404 because the local sentinel
        # isn't published to GHCR) AND the 2026-05-30 Nariman signup
        # incident (container_id=None, user couldn't chat). Refuse to
        # ship a bare ":latest" — caller must wait for a rollout to
        # complete, or pass an explicit override_image_tag.
        if image_tag.endswith(":latest"):
            logger.error(
                "[CONTAINER-SENTINEL-REFUSE] refusing to provision user=%s "
                "with sentinel image %r — no completed rollout has been "
                "recorded. Trigger a rollout or pass override_image_tag.",
                user_id[:8], image_tag,
            )
            raise HTTPException(
                503,
                detail=(
                    "Agent container can't be provisioned right now — "
                    "the platform has no known-good image SHA to deploy. "
                    "An operator must complete a rollout first."
                ),
            )
    body = {
        "prefix": prefix,
        "user_id": user_id,
        "image_tag": image_tag,
        "agent_config": _agent_config_to_bridge_body(agent_config),
    }

    if existing:
        existing.status = "provisioning"
        existing.error_message = None
        await db.commit()

    try:
        async with _bridge_client() as client:
            r = await client.post("/v1/tenants", json=body)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        err = f"bridge create_tenant failed: {e.response.status_code} {e.response.text[:200]}"
        logger.error(err)
        if existing:
            existing.status = "error"
            existing.error_message = err[:500]
            await db.commit()
        raise RuntimeError(err)
    except httpx.HTTPError as e:
        err = f"bridge unreachable: {e}"
        logger.error(err)
        if existing:
            existing.status = "error"
            existing.error_message = err[:500]
            await db.commit()
        raise RuntimeError(err)

    # Upsert ManagedContainer row. INSERT-then-fall-through used to
    # race two concurrent provisions (e.g. user clicks Next twice in
    # /setup/channel — frontend retry on transient errors makes this
    # easier to hit) — both saw existing=None, both INSERT'd, second
    # hit `managed_containers_user_id_key` UniqueViolation and
    # poisoned the session. Now we re-SELECT after the bridge call so
    # whichever request committed first is reused; only if there
    # really is no row do we INSERT.
    if not existing:
        existing = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == user_id)
        )).scalar_one_or_none()

    if existing:
        container = existing
    else:
        import uuid
        container = ManagedContainer(
            id=str(uuid.uuid4()),
            user_id=user_id,
            container_name=data["container_name"],
        )
        db.add(container)

    container.container_id = data["container_id"]
    container.container_name = data["container_name"]
    container.host_port = int(data["host_port"])
    container.image_tag = data["image_tag"]
    container.status = "running"
    container.started_at = datetime.utcnow()
    container.error_message = None
    # db_name is deterministic from prefix; bridge's create_tenant_db produces it
    container.db_name = f"toup_agent_{prefix}"

    # Update AgentConfig with the HTTPS agent URL + bridge-issued api_key
    if agent_config:
        agent_config.agent_url = data["agent_url"]
        agent_config.agent_api_key = data["agent_api_key"]
        agent_config.hosting_mode = "managed"
        agent_config.deploy_status = "active"

    try:
        await db.commit()
    except Exception:
        # Last-line backstop for the same race — if a different request
        # raced in between our re-SELECT and commit, the INSERT path
        # would still UniqueViolate. Recover by rolling back to a fresh
        # transaction and merging onto whatever row is now there.
        await db.rollback()
        existing = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == user_id)
        )).scalar_one_or_none()
        if not existing:
            raise
        existing.container_id = data["container_id"]
        existing.container_name = data["container_name"]
        existing.host_port = int(data["host_port"])
        existing.image_tag = data["image_tag"]
        existing.status = "running"
        existing.started_at = datetime.utcnow()
        existing.error_message = None
        existing.db_name = f"toup_agent_{prefix}"
        if agent_config:
            agent_config.agent_url = data["agent_url"]
            agent_config.agent_api_key = data["agent_api_key"]
            agent_config.hosting_mode = "managed"
            agent_config.deploy_status = "active"
        await db.commit()
        container = existing
    await db.refresh(container)
    logger.info(
        "provisioned_container user=%s prefix=%s port=%s agent_url=%s",
        user_id[:8], prefix, container.host_port, data["agent_url"],
    )

    # Post-provision soul sync (background, non-blocking)
    asyncio.create_task(
        _sync_soul_after_start(user_id, data["agent_url"], data["agent_api_key"])
    )
    return container


async def backfill_sentinel_image_containers(
    db: AsyncSession,
    *,
    limit: int = 200,
) -> dict:
    """Re-provision any managed_container that's stuck on the sentinel image
    ("toup-agent:latest" / any bare `:latest` tag) or has a NULL
    container_id — both of which indicate the user was bound via the pool
    fast-path before that bug was fixed and currently CAN'T chat (DB lies
    about the image, no real bridge-tracked container_id).

    Runs on platform boot as the eventually-consistent guarantee behind
    "every signup gets a real GHCR image" — cures any stragglers from
    before the pool_service fix or from a transient
    `_latest_known_good_image_tag` miss. Bounded + ~0 candidates in steady
    state so it's a cheap query on most boots.

    Idempotent: only touches rows that match the broken-state predicate;
    successful re-provision moves them off the predicate so subsequent
    boots find nothing to fix.
    """
    rows = (await db.execute(
        select(ManagedContainer).where(
            (ManagedContainer.image_tag.endswith(":latest"))
            | (ManagedContainer.container_id.is_(None))
        ).limit(limit)
    )).scalars().all()

    fixed = 0
    failed = 0
    skipped = 0
    for mc in rows:
        # Skip rows whose user has no AgentConfig (e.g., deleted user with
        # a dangling container row) — provision_container would just bail.
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == mc.user_id)
        )).scalar_one_or_none()
        if cfg is None:
            skipped += 1
            continue
        # A healthy warm pool claim legitimately has container_id=NULL until the
        # claim path records it (the bridge only began returning container_id in
        # 2026-06). Do NOT cold-rebuild such a row merely because the id is NULL —
        # that destroys the warm container and pays a full cold boot, AND drops it
        # onto the cold-provision path that races identity sync (the
        # warm-claim-abandoned → "Agent Owner" / unnamed-agent bug). The
        # sentinel-image arm still forces an unconditional rebuild (the original
        # purpose); a NULL-id-ONLY match is health-gated and skipped when the agent
        # is actually reachable — `_verify_and_heal_pool_claim` owns claim health,
        # and A2 backfills the id on the next fresh claim.
        image_is_sentinel = (mc.image_tag or "").endswith(":latest")
        if not image_is_sentinel and mc.container_id is None:
            try:
                from app.services.prewarm_service import _is_agent_actually_healthy
                if await _is_agent_actually_healthy(mc.user_id):
                    skipped += 1
                    continue
            except Exception:
                # Health probe itself failed → fall through and re-provision so a
                # genuinely-broken row still self-heals (safe default).
                pass
        try:
            new_mc = await provision_container(db, mc.user_id, recreate=True)
            await db.commit()
            if new_mc.container_id and not new_mc.image_tag.endswith(":latest"):
                fixed += 1
            else:
                failed += 1
        except Exception as e:
            await db.rollback()
            logger.warning(
                "[container-backfill] re-provision failed for user=%s: %s",
                mc.user_id[:8], e,
            )
            failed += 1

    summary = {
        "candidates": len(rows), "fixed": fixed,
        "failed": failed, "skipped_no_agent_config": skipped,
    }
    if rows:
        logger.info("[container-backfill] %s", summary)
    return summary


async def container_reconciler_loop() -> None:
    """Long-running background task — re-runs `backfill_sentinel_image_containers`
    on an interval so a signup that lands in the broken state (container_id
    NULL or a ":latest" sentinel image) self-heals within minutes.

    Why this exists: the pool fast-path's bridge `/v1/pool/claim` response
    carries no container_id, so every pool-claimed signup is born with
    container_id NULL and CAN'T chat until re-provisioned. The backfill cured
    this only at platform boot, which left users who signed up between deploys
    stuck for hours (2026-05-31: mrvviinn@gmail.com signed up post-boot and had
    no reachable agent at all). Same durability pattern as
    rollout_service.rollout_reconciler_loop: asyncio.create_task dies on a
    Railway redeploy → the DB row is the durable signal → the reconciler
    retries on its next tick. Each tick opens its own narrow session and never
    lets an exception kill the loop.
    """
    interval = int(getattr(settings, "container_reconciler_interval_s", 180) or 0)
    if interval <= 0:
        logger.info("[container-reconciler] disabled (interval<=0)")
        return
    logger.info("[container-reconciler] started (tick=%ss)", interval)
    from app.db.database import async_session_maker
    while True:
        # Sleep first: the boot path already runs one backfill before this
        # loop starts, so an immediate tick would be redundant churn.
        await asyncio.sleep(interval)
        try:
            async with async_session_maker() as db:
                summary = await backfill_sentinel_image_containers(db)
            if summary.get("fixed") or summary.get("failed"):
                logger.info("[container-reconciler] tick: %s", summary)
        except Exception:
            # Never let a tick exception kill the loop — log and keep going.
            logger.exception("[container-reconciler] tick failed; will retry")
        # Stranded-user backstop: managed users with NO container row (their
        # signup finalize died — Railway redeploys kill fire-and-forget
        # tasks) or a pool row knocked out of 'running' (rollout orphan
        # quarantine). backfill_sentinel_image_containers can't see either
        # class: its predicate needs an existing row with a NULL
        # container_id or sentinel image. Own try/except + own sessions so
        # a failure here can't affect the backfill above (and vice versa).
        try:
            from app.services.pool_service import reclaim_stranded_users
            reclaim = await reclaim_stranded_users()
            if reclaim.get("claimed") or reclaim.get("failed"):
                logger.info("[container-reconciler] reclaim: %s", reclaim)
        except Exception:
            logger.exception("[container-reconciler] reclaim failed; will retry")


async def stop_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Stop a tenant container via the bridge.

    Note: bridge doesn't have a /stop endpoint in Phase 3 v1 — stopping is
    equivalent to leaving the container in place but marking our DB row
    as stopped. If future ops need `docker stop`, we add a bridge endpoint
    then. For now, this is a DB-status-only op.
    """
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    container.status = "stopped"
    container.stopped_at = datetime.utcnow()
    await db.commit()
    logger.info("marked_stopped user=%s", user_id[:8])
    return container


async def start_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Start/re-provision a stopped container. Delegates to provision_container."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    return await provision_container(db, user_id)


async def restart_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Restart tenant container via bridge."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    prefix = user_id[:8]
    try:
        async with _bridge_client() as client:
            r = await client.post(f"/v1/tenants/{prefix}/restart")
            r.raise_for_status()
    except httpx.HTTPError as e:
        err = f"bridge restart failed: {e}"
        logger.error(err)
        container.status = "error"
        container.error_message = err[:500]
        await db.commit()
        return container
    container.status = "running"
    container.started_at = datetime.utcnow()
    container.error_message = None
    await db.commit()
    return container


async def destroy_container(db: AsyncSession, user_id: str) -> bool:
    """Remove a tenant container (DB + network + Caddy route) via bridge.

    Pool members vs ad-hoc tenants:
    - If the container's name is `toup-agent-pool-NN`, it was bound from
      the warm pool. Use `/v1/pool/release` so the bridge drains via
      /admin/drain, revokes the DB role, and the reconciler reaps and
      respawns the slot to maintain target.
    - Otherwise it's a slow-path tenant container; use the legacy
      `/v1/tenants/<prefix>` DELETE which drops DB, network, route.
    """
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return False
    prefix = user_id[:8]
    is_pool_member = (
        container.container_name
        and container.container_name.startswith("toup-agent-pool-")
    )
    try:
        if is_pool_member:
            from app.services.pool_service import release_pool_member
            await release_pool_member(prefix=prefix, user_id=user_id)
        else:
            async with _bridge_client() as client:
                r = await client.delete(f"/v1/tenants/{prefix}")
                # 204 is the expected response; 404 (already gone) is also OK
                if r.status_code not in (204, 404):
                    r.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("bridge destroy failed for %s: %s", user_id[:8], e)
        # Continue — mark DB row regardless. Operator can reconcile if bridge
        # state drifts.
    container.status = "deleted"
    container.stopped_at = datetime.utcnow()
    await db.commit()
    return True


async def get_container_status(db: AsyncSession, user_id: str) -> Optional[dict]:
    """Get runtime status via bridge + DB row.

    Pool-bound containers keep their `toup-agent-pool-NN` name even
    after claim — the bridge's `/v1/tenants/{prefix}/status` endpoint
    looks up `toup-agent-{prefix}` and 404s on every pool-bound user.
    Without the fallback below, frontend's `docker_status === 'running'`
    check stayed false forever after a successful pool claim, producing
    "Agent took too long to start" on the WhatsApp QR onboarding step
    despite the container being healthy. The DB row's status field is
    set authoritatively by `claim_for_user` AFTER a successful bridge
    bind, so it's the right fallback when the bridge can't resolve the
    tenant prefix.
    """
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    prefix = user_id[:8]

    bridge_status: dict = {}
    try:
        async with _bridge_client() as client:
            r = await client.get(f"/v1/tenants/{prefix}/status")
            if r.status_code == 200:
                bridge_status = r.json()
    except httpx.HTTPError as e:
        logger.warning("bridge status unreachable for %s: %s", user_id[:8], e)

    # Pool-bound fallback: bridge can't resolve `toup-agent-pool-NN` by
    # tenant prefix, so docker_status comes back missing. The DB row is
    # authoritative for pool-bound containers — use it.
    docker_status = bridge_status.get("status")
    if docker_status is None and container.container_name and \
            container.container_name.startswith("toup-agent-pool-"):
        docker_status = container.status

    return {
        "id": container.id,
        "status": container.status,
        "docker_status": docker_status,
        "docker_health": bridge_status.get("health"),
        "port": container.host_port,
        "image_tag": container.image_tag,
        "pin_image_tag": container.pin_image_tag,
        "container_name": container.container_name,
        "created_at": container.created_at.isoformat() if container.created_at else None,
        "started_at": container.started_at.isoformat() if container.started_at else None,
        "error": container.error_message,
    }


async def update_container_env(
    db: AsyncSession,
    user_id: str,
    agent_config: AgentConfig,
) -> Optional[ManagedContainer]:
    """Push a new agent_config to the bridge and recreate the container.

    In Phase 2 this SSH'd to rewrite the .env file then `docker rm/run`.
    In Phase 3 we re-provision via bridge (idempotent — bridge detects
    existing container, force-removes it, rewrites the .env file, and
    re-runs with fresh env).

    Must pass `recreate=True` — without it, provision_container's
    new-tenant idempotency early-return would skip the bridge call
    entirely and leave the container running with stale env (the bug that
    let Arshia's freshly-activated bundle tenant keep `LLM_MODE=manual`
    and an empty `TOUP_TOKEN` after Stripe checkout completed).
    """
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None

    # Phase A (never-sleep): if this user is on a pool container, route
    # the env update through the bridge's /v1/pool/refresh-config which
    # calls /admin/bind on the running agent — preserves the warm
    # state. Recreate would force-destroy the pool container and the
    # next chat would hit a cold-booting fresh tenant ("waking..."
    # indicator), exactly the regression Phase A was built to prevent.
    if container.container_name and container.container_name.startswith("toup-agent-pool-"):
        prefix = user_id[:8]
        # Build the same bind payload the platform pool_service builds
        # at signup, then forward to the bridge.
        from app.services.pool_service import _build_bind_payload
        payload = await _build_bind_payload(db, user_id, agent_config)
        try:
            async with _bridge_client() as client:
                resp = await client.post("/v1/pool/refresh-config", json=payload)
                if resp.status_code == 404:
                    # Bridge no longer tracks this prefix as pool-bound
                    # (pool registry drift). Fall through to recreate
                    # path — the user will eat a cold boot once, then
                    # be on a regular tenant container.
                    logger.warning(
                        "[update_container_env] pool drift for %s: bridge says no pool member. Falling through.",
                        user_id[:8],
                    )
                    return await provision_container(
                        db, user_id, agent_config=agent_config, recreate=True,
                    )
                resp.raise_for_status()
                logger.info(
                    "[update_container_env] pool refresh OK for %s (no recreate)",
                    user_id[:8],
                )
                return container
        except httpx.HTTPError as e:
            logger.warning(
                "[update_container_env] pool refresh failed for %s: %s. Falling through to recreate.",
                user_id[:8], e,
            )
            return await provision_container(
                db, user_id, agent_config=agent_config, recreate=True,
            )

    return await provision_container(
        db, user_id, agent_config=agent_config, recreate=True,
    )


async def upgrade_tenant_image(
    db: Optional[AsyncSession],
    user_id: str,
    image_tag: str,
    rollout_id: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> dict:
    """Upgrade one tenant to a specific image tag.

    Phase B (never-sleep plan): when `settings.use_blue_green_rollouts`
    is True, calls the bridge's `/v1/tenants/<prefix>/blue-green-upgrade`
    endpoint — zero WS drops, in-flight streams complete on the old
    slot. When False (default during the bake window), falls back to
    the legacy `/v1/tenants/<prefix>/upgrade` recreate-flow.

    Called by rollout_service.py. Raises on failure; caller decides
    rollback strategy.

    The ``db`` parameter is OPTIONAL. When None, this function opens
    a fresh narrow session for the post-call ManagedContainer.image_tag
    update — callers should pass None to avoid holding their own
    session across the bridge call. The held-session-across-network
    pattern caused the 547-min stall on rollout ed53f0d89a11
    (2026-05-25): when the bridge stopped responding, each hung
    attempt held a DB connection, and the reconciler's own pool was
    starved on the Supabase pooler so it couldn't mark the rollout
    orphan. Passing a real db is supported for backward compatibility
    with tests.
    """
    prefix = user_id[:8]
    use_blue_green = bool(getattr(settings, "use_blue_green_rollouts", False))
    endpoint = (
        f"/v1/tenants/{prefix}/blue-green-upgrade"
        if use_blue_green
        else f"/v1/tenants/{prefix}/upgrade"
    )
    body = {"image_tag": image_tag, "rollout_id": rollout_id}
    if use_blue_green:
        body["drain_timeout_s"] = int(getattr(settings, "blue_green_drain_timeout_s", 60))

    async with _bridge_client(timeout_s or settings.bridge_upgrade_timeout_s) as client:
        r = await client.post(endpoint, json=body)
        if r.status_code == 404 and use_blue_green:
            # The bridge may not have the blue-green endpoint deployed
            # yet (older bridge image, or it was rolled back). Distinguish
            # "endpoint missing" (the route itself is unknown) from
            # "container not found" (tenant has no container) by checking
            # the tenant whois endpoint first. If the tenant exists on
            # the bridge, the 404 is the route — fall back to legacy
            # /upgrade so canary doesn't false-abort with
            # "orphan tenant" when the real issue is bridge capability.
            # Self-heals once the bridge is updated.
            whois_status = await _whois_status(client, prefix)
            if whois_status == 200:
                logger.warning(
                    "[rollout] bridge has no /blue-green-upgrade route; "
                    "falling back to legacy /upgrade for tenant=%s",
                    prefix,
                )
                # Drop blue-green-only field before retrying legacy endpoint
                body.pop("drain_timeout_s", None)
                r = await client.post(f"/v1/tenants/{prefix}/upgrade", json=body)
                if r.status_code == 404:
                    # Now it's a real "container not found" — surface it.
                    # Whois disagreed (200), so don't claim it's a confirmed
                    # orphan; the caller will alert without quarantining.
                    raise BridgeContainerNotFound(
                        prefix=prefix, raw=r.text[:200], confirmed_orphan=False,
                    )
            else:
                # Whois 404 = bridge agrees no such tenant → confirmed orphan.
                # Whois exception / other status = bridge ambiguous; alert
                # but don't quarantine.
                raise BridgeContainerNotFound(
                    prefix=prefix,
                    raw=r.text[:200],
                    confirmed_orphan=(whois_status == 404),
                )
        elif r.status_code == 404:
            # Legacy /upgrade returned 404. Corroborate with /whois before
            # claiming a confirmed orphan — a transient bridge hiccup that
            # only affects /upgrade would otherwise auto-quarantine real
            # tenants when the caller acts on `confirmed_orphan`.
            whois_status = await _whois_status(client, prefix)
            raise BridgeContainerNotFound(
                prefix=prefix,
                raw=r.text[:200],
                confirmed_orphan=(whois_status == 404),
            )
        if r.status_code == 503:
            try:
                detail = r.json().get("detail", {})
            except ValueError:
                detail = {"error": "unhealthy", "raw": r.text[:200]}
            raise BridgeUpgradeUnhealthy(detail)
        if r.status_code == 504 and use_blue_green:
            # Blue-green specific: new slot didn't become healthy.
            # Bridge already cleaned up the new slot; old slot keeps
            # serving. Surface as BridgeUpgradeUnhealthy so the
            # rollout reconciler treats it the same as legacy 503.
            raise BridgeUpgradeUnhealthy({
                "error": "blue_green_health_timeout",
                "raw": r.text[:200],
            })
        r.raise_for_status()
        data = r.json()

    # Platform-side record-keeping: update ManagedContainer.image_tag.
    # Use the caller's session when provided (legacy tests do); otherwise
    # open a fresh narrow session so the caller doesn't need to hold one
    # across the bridge call above.
    async def _persist_image_tag(session: AsyncSession) -> None:
        result = await session.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == user_id)
        )
        container = result.scalar_one_or_none()
        if container:
            container.image_tag = image_tag
            await session.commit()

    if db is not None:
        await _persist_image_tag(db)
    else:
        from app.db.database import async_session_maker
        async with async_session_maker() as own_db:
            await _persist_image_tag(own_db)

    return data


class BridgeUpgradeUnhealthy(RuntimeError):
    """Raised when bridge /upgrade returns 503 — tenant ended up unhealthy.
    Wraps the structured detail body so the rollout service can decide on
    rollback with prior_tag.
    """

    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(f"bridge upgrade unhealthy: {detail}")


class BridgeContainerNotFound(RuntimeError):
    """Raised when bridge /upgrade returns 404 — no container exists for
    this prefix. The tenant is orphaned (was deprovisioned without our
    record being updated, or never provisioned). Rollback is impossible
    because there's nothing to roll back to. Caller should mark the
    attempt failed and skip the rollback path.

    ``confirmed_orphan`` is True only when the bridge ``/whois`` endpoint
    also returned 404 for this prefix — i.e. two independent bridge
    endpoints agreed the tenant doesn't exist. The rollout service uses
    this to gate auto-quarantine of the platform-side ``ManagedContainer``
    row: a confirmed orphan flips ``status='running' → 'orphan'`` so
    subsequent rollouts skip it (eliminating the per-rollout alert
    spam observed for tenants 9ef741e3 / 6d560209 / 4930493b / 6fefba0a
    on 2026-05-24..25); an unconfirmed orphan (whois timed out or returned
    a different status) only alerts so a transient bridge hiccup can't
    auto-evict real tenants.

    Caught 2026-05-10: tenant 9ef741e3 produced 422 on every rollback
    because the rollout service was sending image_tag="unknown" (the
    sentinel for missing prior_tag) which fails Pydantic validation
    on the bridge.
    """

    def __init__(self, prefix: str, raw: str = "", confirmed_orphan: bool = False):
        self.prefix = prefix
        self.raw = raw
        self.confirmed_orphan = confirmed_orphan
        super().__init__(f"bridge container not found for prefix={prefix}: {raw}")


async def _whois_status(client, prefix: str) -> int:
    """Return the HTTP status code from the bridge's ``/whois`` endpoint,
    or 0 if the request itself failed (DNS, connect, timeout, etc.).

    Used to corroborate 404s from /upgrade / /blue-green-upgrade before
    claiming a tenant is a confirmed orphan. We only treat a tenant as a
    confirmed orphan when whois ALSO returns 404 — anything else (200 =
    tenant exists, 0 = bridge unreachable, 5xx = bridge errored) leaves
    confirmed_orphan=False so callers don't quarantine on ambiguous signal.
    """
    try:
        r = await client.get(f"/v1/tenants/{prefix}/whois")
        return r.status_code
    except Exception as e:  # noqa: BLE001 — any HTTP/transport error
        logger.warning("[rollout] whois probe failed for %s: %s", prefix, e)
        return 0


# ─── Post-provision soul sync (unchanged from Phase 2) ────────────


async def refresh_tenant_tools(db: AsyncSession, user_id: str) -> bool:
    """T1h — Notify a user's tenant agent that its connector tools
    cache must be invalidated immediately.

    Called by the OAuth callback (after `vault.put` + audit) and the
    OAuth disconnect handler (after `vault.disconnect` + audit). The
    agent's `PUT /api/agent/refresh-tools` drops its 60s TTL cache
    and refetches via the platform's per-user filtered tools/list.

    Best-effort:
      * 3-second timeout — the user is waiting for the OAuth flow to
        complete. If a tenant container is slow or down, we bail and
        the cache's 60s TTL becomes the safety net.
      * Failure logs WARN, returns False. NEVER raises — the OAuth
        flow must succeed regardless of whether this notification
        lands. The cache will self-heal on the next TTL tick.
      * Returns True on 2xx, False on anything else (including no
        AgentConfig row, missing agent_url, missing key).
    """
    import httpx as _httpx
    from app.db.models import AgentConfig

    cfg = (
        await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
    ).scalar_one_or_none()
    if cfg is None or not cfg.agent_url or not cfg.agent_api_key:
        # Self-hosted agents (no managed AgentConfig) and
        # mid-provisioning users (no agent_url yet) both land here.
        # Not an error — just nothing to notify.
        logger.info(
            "[refresh_tools] no agent_url/key for user=%s — skipping notify",
            user_id[:8],
        )
        return False

    url = f"{cfg.agent_url.rstrip('/')}/api/agent/refresh-tools"
    # Two-pass notify: a fast 3 s attempt right after OAuth so the user
    # gets immediate access to the connector tools, and a 10 s fallback
    # if the agent container was just warming up (image pull, app
    # startup, etc.). Without the retry, a single slow agent boot meant
    # the cache only updated via the 60 s TTL — the user saw "Gmail
    # tool isn't exposed" responses for up to a minute.
    headers = {"X-Agent-Key": cfg.agent_api_key}
    for attempt, timeout_s in enumerate((3.0, 10.0), start=1):
        try:
            async with _httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.put(url, headers=headers)
            if 200 <= resp.status_code < 300:
                logger.info(
                    "[refresh_tools] notified user=%s (status=%d, attempt=%d)",
                    user_id[:8], resp.status_code, attempt,
                )
                return True
            # Non-2xx — agent reachable but rejected the call. Retrying
            # won't fix a 401 / 404; bail to the TTL safety net.
            logger.warning(
                "[refresh_tools] tenant returned %d for user=%s — "
                "TTL is the safety net (attempt=%d, body=%s)",
                resp.status_code, user_id[:8], attempt, resp.text[:200],
            )
            return False
        except Exception as e:
            # Transport-level failure (timeout, DNS, container booting).
            # Loop once with the longer timeout before bailing.
            if attempt == 1:
                logger.info(
                    "[refresh_tools] first attempt timed out for user=%s "
                    "(%s) — retrying with longer timeout",
                    user_id[:8], type(e).__name__,
                )
                continue
            logger.warning(
                "[refresh_tools] notify failed for user=%s after %d "
                "attempts: %s — TTL is the safety net",
                user_id[:8], attempt, e,
            )
            return False
    return False


async def _sync_soul_after_start(
    user_id: str, agent_url: str, agent_api_key: str, db_session_maker=None,
):
    """Wait for container health, then push soul config.

    Runs as a background task after provision or recreate. Unchanged from
    Phase 2 except the agent_url is now the HTTPS subdomain (via Caddy)
    rather than http://ip:port.
    """
    import httpx as _httpx
    from app.db.database import async_session_maker
    from app.db.models import SoulConfig

    for _ in range(10):
        await asyncio.sleep(3)
        try:
            async with _httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{agent_url}/agent/health")
                if resp.status_code == 200:
                    break
        except Exception:
            continue
    else:
        logger.warning("[SOUL] container not healthy after 30s for %s", user_id[:8])
        return

    try:
        _owner_name = None
        _owner_email = None
        # Poll for the SoulConfig instead of giving up on the first read. The
        # container can be (re)provisioned BEFORE the user finishes the Soul step
        # — most importantly when the container-backfill reconciler cold-rebuilds
        # a freshly-claimed container mid-onboarding. A single read + early-return
        # then permanently regresses the agent to the "Agent Owner" stub greeting
        # AND an unnamed agent (sidebar shows generic "agent" not the chosen name),
        # because this HTTP push is the cold path's ONLY identity carrier. Retry
        # briefly so a soul-save that races the (re)provision still lands.
        soul = None
        for _soul_attempt in range(8):
            async with async_session_maker() as db:
                soul = (await db.execute(
                    select(SoulConfig).where(SoulConfig.user_id == user_id)
                )).scalar_one_or_none()
            if soul:
                break
            await asyncio.sleep(3)

        # Owner identity — pushed REGARDLESS of whether a SoulConfig exists yet.
        # A freshly (re)provisioned container boots with a stub
        # User(name="Agent Owner"); coupling this push to the Soul step meant any
        # agent reached before onboarding (or whose container was rebuilt
        # mid-onboarding) greeted "Agent Owner" forever. Captured as plain
        # strings before the session closes.
        try:
            from app.db.models import User as _User
            async with async_session_maker() as db:
                _owner = (await db.execute(
                    select(_User).where(_User.id == user_id)
                )).scalar_one_or_none()
                _owner_name = (getattr(_owner, "name", None) or None)
                _owner_email = (getattr(_owner, "email", None) or None)
        except Exception:
            _owner_name = _owner_email = None

        if not soul and not (_owner_name or _owner_email):
            logger.warning(
                "[SOUL] no SoulConfig and no owner identity for %s — skipping sync",
                user_id[:8],
            )
            return

        # When the soul isn't ready yet, send an OWNER-ONLY sync: name and
        # compiled_text are omitted (the agent skips the soul upsert and keeps
        # its existing/default soul) but the owner name still lands, so the
        # agent stops greeting "Agent Owner". The full soul lands on the
        # user-initiated soul-save path (soul.py:_sync_soul_to_vps) at onboarding.
        _body = {
            "user_id": user_id,
            "name": soul.name if soul else None,
            "compiled_text": soul.compiled_text if soul else None,
            "deactivate_agent_soul_memories": False,
            "owner_name": _owner_name,
            "owner_email": _owner_email,
        }
        if soul:
            _body["agent_config_updates"] = {
                "agent_name": soul.name,
                "agent_color": soul.color,
            }

        async with _httpx.AsyncClient(timeout=15) as client:
            resp = await client.put(
                f"{agent_url}/api/soul/sync",
                json=_body,
                headers={"X-Agent-Key": agent_api_key},
            )
            if resp.status_code == 200:
                logger.info(
                    "[SOUL] synced for %s (%s)",
                    user_id[:8], "soul+owner" if soul else "owner-only",
                )
            else:
                logger.warning("[SOUL] sync failed for %s: %s", user_id[:8], resp.status_code)
    except Exception as e:
        logger.warning("[SOUL] sync error for %s: %s", user_id[:8], e)
