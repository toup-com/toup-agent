"""Shared SQLAlchemy base and imports for all model files."""

from datetime import datetime
from typing import Optional, List
from enum import Enum
import uuid

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, BigInteger, Boolean,
    ForeignKey, Table, Enum as SQLEnum, JSON, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR

try:
    from pgvector.sqlalchemy import Vector as _PgVector
except ImportError:
    _PgVector = None  # Fallback for environments without pgvector


def _get_embedding_dim() -> int:
    """Get configured embedding dimension (avoids circular import at class-body time)."""
    try:
        from app.config import settings
        return settings.embedding_dimension
    except Exception:
        return 1536  # safe default


def Vector(dim: int = 0):
    """Return a pgvector Vector column type with the configured dimension."""
    if _PgVector is None:
        return None
    if dim <= 0:
        dim = _get_embedding_dim()
    return _PgVector(dim)


Base = declarative_base()


# ── Table Partitioning: Platform vs Agent DBs ─────────────────────────
#
# Toup has a hybrid database architecture:
#   - Platform DB (Railway/Supabase): central, shared across all users
#   - Agent DB (per-user Docker container): isolated per user
#
# When you add a new model, add its __tablename__ to exactly ONE of these
# sets, or to SHARED_TABLES if it genuinely lives in both databases.
# If you skip this step, test_table_partitioning.py will fail.
#
# init_db() uses these sets to decide which tables to create in each mode.
# ──────────────────────────────────────────────────────────────────────

AGENT_ONLY_TABLES: set[str] = {
    # Day-as-Chat context architecture
    "day_chats",
    "context_budget_logs",
    "migration_status",
    # Conversation & messages (agent stores all chat data)
    "conversations",
    "messages",
    # Durable exactly-once ledger for inbound chat (see ProcessedMessage).
    "processed_messages",
    # Durable agent→platform notification outbox (Autopilot PR4) —
    # flushed to the platform's /api/agent/notify, acked rows only.
    "agent_notify_outbox",
    # Autopilot approvals — durable ask-the-user store (Autopilot PR7);
    # survives WS disconnects/restarts unlike the in-memory
    # PermissionBroker. Platform reads via the agent HTTP API only.
    "autopilot_approvals",
    # Memory system
    "memories",
    # Durable outbox for facts extracted but not yet stored — capture is
    # fire-and-forget, so without this a failed write loses the turn silently.
    "memory_capture_outbox",
    "memory_relationships",
    "brain_stats",
    "memory_events",
    "retrieval_events",
    # Entity & knowledge graph
    "entities",
    "entity_links",
    "entity_relationships",
    # Documents & media
    "documents",
    "document_chunks",
    "media",
    # Toup Media library — saved, replayable playlists (captured radio
    # stations). Platform reads via the memories-style proxy; the platform DB
    # keeps a mirror table only as the read-fallback target.
    "media_playlists",
    # Apps & build jobs
    "apps",
    "build_jobs",
    "build_usage",
    "reconciliation_logs",
    # The per-job event log. `app/api/jobs_events.py` lives under app/api/ but
    # is mounted ONLY by agent_main.py:1679 — platform_main never imports it —
    # and every writer is agent-side (job_reaper, job_logger, tool_executor,
    # app_builder/skill). Grepped for `JobEvent` across platform_main and every
    # api module platform_main mounts, plus app/services: zero hits.
    "job_events",
    # Agent runtime
    "cron_jobs",
    "telegram_user_mappings",
    "agent_errors",
    "api_keys",
    # System-managed routines (email briefing, calendar briefing, …).
    # User-facing config in /agent/settings/routines; operator monitoring
    # in Mission Control reads via authenticated API, not direct DB query.
    "routines",
    "routine_runs",
    "routine_notification_dedupe",
    # Event-driven automations (Gmail Pub/Sub etc.). Platform-side
    # webhook authenticates the Pub/Sub push and dispatches an envelope
    # to the user's per-tenant agent container via the bridge; the
    # agent owns the trigger config + event audit trail. Tokens never
    # leave the platform — same isolation guarantee as Routines.
    "triggers",
    "trigger_events",
    # WhatsApp Cloud API webhook dedupe (agent-side; survives container
    # restarts so Meta retries within a 7-day window cannot re-run the LLM)
    "whatsapp_inbound_dedupe",
    # Identity & soul
    "identities",
    "soul_configs",
}

PLATFORM_ONLY_TABLES: set[str] = {
    # VPS & infrastructure
    "vps_plans",
    "vps_instances",
    "managed_containers",
    # Automated deployment pipeline (Phase 3)
    "rollouts",
    "rollout_attempts",
    # Billing & invites
    "invites",
    "llm_bundle_allocations",
    "llm_usage_records",
    "llm_proxy_events",
    # Credits, plans and IAP. The platform is the sole accountant: the agent
    # computes cost with PURE functions (`tokens_to_credits_raw`, `FLAT_FEES`,
    # `image_generation_cost_cents`, `underlying_cost_to_credits` — each checked
    # for `select(`/`await`/session use and clean) and then REPORTS the charge
    # over HTTP via credit_reporter. Grepped `CreditLedger|CreditBalance|
    # CreditReservation|SubscriptionPlan|AppleSubscription` across agent_main,
    # all of app/agent/, and all 30 api modules agent_main mounts: zero hits.
    #
    # These were uncategorized until 2026-08-04, and uncategorized means created
    # in BOTH databases — so every tenant carries empty copies. That is not
    # harmless bookkeeping: app_builder/skill.py:848 records a gate that called
    # `credit_service.check_balance()` against the agent's own DB, hit
    # "subscription_plans 'free' row missing", had it swallowed by an except,
    # and silently no-op'd. An empty table is a worse failure surface than an
    # absent one, because it answers queries instead of refusing them.
    "credit_ledger",
    "credit_balances",
    "credit_reservations",
    "subscription_plans",
    "apple_subscriptions",
    # Search gateway telemetry. Platform-only by construction: the whole point
    # of the gateway is that the tenant container no longer sees the upstream
    # call, so it has nothing to write here.
    "search_events",
    # Streaming credentials (platform stores, agent fetches via API)
    "streaming_credentials",
    "credential_access_log",
    # Platform-wide admin settings (editable via admin panel)
    "platform_settings",
    # Account deletion audit + sensitive-action replay defense (§1.3).
    "deletion_audit_events",
    "sensitive_action_redemptions",
    # Per-tenant security audit (T0b — agent_api_key rotation, future
    # connector quarantine, future security primitives). Append-only.
    "agent_security_events",
    # Third-party connector vault (T1a). Tokens stay platform-side
    # only — the bridge never sees them, tenant containers never see
    # them. Reaches providers via the platform's MCP server (T0c-gated
    # auth) → connector dispatcher (T1e).
    "connector_identities",
    "connector_oauth_sessions",
    "connector_events",
    "connector_user_preferences",
    # Staged elevation:true tool calls awaiting the user's tap on the
    # confirmation card. Platform-side because the tokens that execute
    # them are, so approval never round-trips to a tenant container.
    "connector_pending_actions",
    # OAuth provider-app credentials (Google/GitHub/etc client_id+secret).
    # Persisted in DB so the operator can paste them through the admin
    # UI instead of editing env vars + redeploying.
    "provider_app_credentials",
    # Active user JWT sessions. Auth router is platform-only (users
    # don't log into agent containers — agent_api_key auth is used
    # there instead), so the table only exists on the platform side.
    "user_sessions",
    # Maintenance / support agent: ingests user-reported problems,
    # diagnoses against docs/skills, and (after admin approval) opens a
    # fix PR. Operator/admin tool — never lives on tenant agent DBs.
    "support_issues",
    "support_issue_events",
    # Support card attachments (e.g. mobile screenshots) — bytes live in the
    # platform DB, served only via an auth'd reporter/admin endpoint.
    "support_attachments",
    # Free-credit grant eligibility tombstone + persistent signup-attempt
    # log. Signup, the free-credit grant, and IP rate limiting all live on
    # the platform — agent containers neither create accounts nor grant
    # credits — so these are platform-only (mirrors the migration guards
    # that skip agent DBs).
    "grant_eligibility",
    "signup_attempts",
    # Proactive notifications (Autopilot arc PR2). Expo push tokens are
    # per-device credentials the platform owns — agent containers must
    # never hold them (same isolation stance as connector tokens above).
    # The queue is claimed/sent by the platform dispatcher; agents only
    # reach it through POST /api/agent/notify.
    "push_devices",
    "notification_queue",
    # iOS Live Activity push-to-start tokens + per-mission activity
    # state (Autopilot phone surface). APNs tokens are platform-only
    # for the same reason as Expo tokens above.
    "live_activity_devices",
    "live_activities",
    # Operator→user announcements. The dispatch + its per-recipient
    # ledger are platform bookkeeping, and the admin↔user thread is a
    # conversation with the OPERATOR, not agent data: the agent must
    # never read it, and the panel reads every user's replies without
    # fanning out to N containers. The only thing that reaches a tenant
    # DB is the chat card itself, written over HTTP by the fan-out
    # worker (and with day_chat_id NULL, so the agent cannot see it).
    "admin_dispatches",
    "admin_dispatch_targets",
    "admin_thread_messages",
}

SHARED_TABLES: set[str] = {
    # Users table exists in both: platform has all users, agent has its owner
    "users",
    # agent_configs was PLATFORM_ONLY on the theory "agent reads it via API",
    # but the agent chat/runner/tool hot path in fact reads it by DIRECT query:
    # SELECT AgentConfig WHERE user_id=… at ws_chat.py:1564/2548,
    # agent_runner.py:587/3091/3439, tool_executor.py:1091, chat.py:189/438
    # (BYOK OpenAI key + per-user disabled-tools). Older agent DBs carried the
    # table as a monolith leftover, so this was latent; newer partitioned agent
    # DBs that init_db never created it on hit `UndefinedTable`, which aborts the
    # surrounding transaction and — because the callers catch the Python error
    # without rolling back — poisons the very next write, so chat persistence
    # broke fleet-wide (2026-07 incident). SHARED creates an EMPTY table on the
    # agent: every touch site is a read that already handles the no-row case by
    # falling back (platform key / no disabled tools). No secret is ever written
    # agent-side (no INSERT/UPDATE against AgentConfig runs in agent mode), so the
    # platform-owns-the-config isolation still holds. This is the permanent fix
    # for that incident; the live fleet was already hot-patched with the table.
    "agent_configs",
    # Browser-extension device registrations. `app/api/extension.py` touches
    # ExtensionDevice 28 times and is mounted by BOTH platform_main.py:843 and
    # agent_main.py:1734, so the table is genuinely bi-resident — the extension
    # pairs against whichever surface answers it. Categorizing this one to a
    # single side is precisely the mistake that took `agent_configs` down.
    "extension_devices",
}


# ── Shared-column authority map (L-1b) ────────────────────────────────
#
# A SHARED table exists in BOTH databases, which means every one of its
# columns is TWO columns: a platform copy and a tenant copy. Twice now a
# writer updated one copy while the reader read the other, silently:
#
#   * users.timezone (2026-08-10, #488 class): chat wrote the TENANT
#     copy, voice read the PLATFORM copy → 23 active users' "today"
#     computed in UTC.
#   * agent_configs.agent_name (2026-08-11): the Soul page wrote the
#     PLATFORM copy, text chat + the W-6 assembler read the TENANT copy
#     → agents named at bind were nameless in chat.
#
# This map makes the decision explicit and CI-enforced: every column of
# every SHARED table must declare which database owns its value. Adding
# a column to a shared table without an entry here fails
# tests/test_table_partitioning.py with instructions.
#
# authority values:
#   "platform"               — the platform DB copy is the truth; the
#                              tenant copy is empty/stub/dead unless a
#                              named sync delivers it.
#   "tenant"                 — the tenant DB copy is the truth; the
#                              platform copy is stub/self-healed/dead.
#   "both-synced"            — both sides legitimately write their own
#                              copy. Either the named sync converges
#                              them, or (sync="none") the value is
#                              row-local bookkeeping where divergence is
#                              expected — see the note.
#   "immutable-after-create" — set at row INSERT and never mutated;
#                              cross-DB equality holds by construction
#                              (or is explicitly N/A — see note).
#
# write_paths are short human pointers ("<code path> (platform|tenant)"),
# not exhaustive call graphs — enough to find the writer with one grep.
# sync names the mechanism that moves the value across the seam, or
# "none". NEVER put a secret or a live value in this map.
# ──────────────────────────────────────────────────────────────────────

_CFG_PATCH = "agent-setup config PATCH (app/api/agent_setup.py update_config, platform)"
_BRIDGE_ENV = "bridge env push (update_container_env → container process env; tenant column stays empty)"
_BYOK_NOTE = (
    "secret is platform-vaulted; never written tenant-side (see SHARED_TABLES "
    "note) — the agent receives it via container env, and agent-side direct "
    "reads of the tenant row fall back when empty"
)
_DEAD_TENANT = "dead on tenant — no tenant-side writer (grep 2026-08-12)"

SHARED_COLUMN_AUTHORITY: dict[str, dict] = {
    # ── users ─────────────────────────────────────────────────────────
    "users.id": {
        "authority": "immutable-after-create",
        "write_paths": [
            "platform signup/oauth (app/services/auth_service.py)",
            "tenant stub create (auth.get_current_user agent-mode, ws_chat stub, /admin/bind owner upsert)",
        ],
        "sync": "none",
        "note": "primary key = the user uuid on both sides; it is the cross-DB join key and never mutates",
    },
    "users.email": {
        "authority": "platform",
        "write_paths": [
            "platform signup/oauth (auth_service, app/api/auth.py)",
            "soul-sync owner upsert (app/api/soul.py sync_soul receiver, tenant)",
            "/admin/bind owner upsert (app/api/admin_pool.py, tenant)",
        ],
        "sync": "soul-sync owner upsert + bind payload",
        "note": "platform is identity authority; tenant boots with an @agent.local stub healed on bind/soul save",
    },
    "users.hashed_password": {
        "authority": "platform",
        "write_paths": ["auth_service signup + change_password (platform)"],
        "sync": "none",
        "note": "tenant stub rows carry an empty string; login exists only on the platform — " + _DEAD_TENANT,
    },
    "users.name": {
        "authority": "platform",
        "write_paths": [
            "PATCH /auth/profile (platform)",
            "admin users update (app/api/admin/users.py, platform)",
            "soul-sync owner upsert (tenant)",
            "/admin/bind owner upsert (tenant)",
        ],
        "sync": "soul-sync owner upsert + bind payload",
        "note": "platform authority; tenant stub 'Agent Owner' is healed on bind/soul save (the 'Hey Agent Owner' fix)",
    },
    "users.password_changed_at": {
        "authority": "platform",
        "write_paths": ["auth_service change_password (platform)"],
        "sync": "none",
        "note": "token-revocation cutoff, read platform-side only; " + _DEAD_TENANT,
    },
    "users.role": {
        "authority": "platform",
        "write_paths": [
            "admin users role update (platform)",
            "invite accept (app/api/admin/users.py, platform)",
        ],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "users.created_at": {
        "authority": "immutable-after-create",
        "write_paths": ["row INSERT on either side"],
        "sync": "none",
        "note": "row-local: platform signup time vs tenant stub-create time legitimately differ — never compare across DBs",
    },
    "users.updated_at": {
        "authority": "both-synced",
        "write_paths": ["every local UPDATE on either side"],
        "sync": "none",
        "note": "ORM bookkeeping — each database stamps its own copy on local writes; divergence expected, exclude from drift probes",
    },
    "users.is_active": {
        "authority": "platform",
        "write_paths": ["admin users update (platform)"],
        "sync": "none",
        "note": "enforced at platform auth; tenant copy defaults True and " + _DEAD_TENANT,
    },
    "users.stripe_customer_id": {
        "authority": "platform",
        "write_paths": ["app/api/billing.py customer create (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "users.timezone": {
        "authority": "tenant",
        "write_paths": [
            "chat WS tz persist (app/api/ws_chat.py ~2880, tenant)",
            "routines skill tz heal (app/agent/skills/builtins/routines, tenant)",
            "PATCH /auth/profile Intl capture (platform)",
            "POST /auth/timezone-from-coords (platform)",
            "admin users update (platform)",
            "voice-relay NULL-fill (ws_realtime._persist_user_tz — local DB on either surface)",
        ],
        "sync": "voice-relay NULL-fill + boot-time Intl PATCH (added 2026-08-10)",
        "note": "THE 2026-08-10 defect column (#488 class): chat wrote tenant while voice read platform → 23 active "
                "users' day computed in UTC. Tenant copy is authoritative (chat is the high-frequency writer); the "
                "platform copy self-heals from device tz when NULL",
    },
    "users.email_verified_at": {
        "authority": "platform",
        "write_paths": [
            "app/api/email_verification.py verify (platform)",
            "auth_service oauth signup (platform)",
        ],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "users.email_verification_token": {
        "authority": "platform",
        "write_paths": ["app/api/email_verification.py issue/clear (platform)"],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "users.email_verification_sent_at": {
        "authority": "platform",
        "write_paths": ["app/api/email_verification.py issue (platform)"],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "users.apple_refresh_token": {
        "authority": "platform",
        "write_paths": ["app/api/auth.py apple oauth (platform, stored encrypted)"],
        "sync": "none",
        "note": "platform-held credential; " + _DEAD_TENANT,
    },
    "users.apple_sub": {
        "authority": "platform",
        "write_paths": ["app/api/auth.py + auth_service apple oauth (platform)"],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "users.notification_preferences": {
        "authority": "platform",
        "write_paths": [
            "app/api/account.py notification prefs + DND (platform)",
            "app/api/extension.py memory-optout (BOTH-mounted router — writes whichever DB is local)",
        ],
        "sync": "none",
        "note": "platform copy is the truth (notification_dispatcher reads PLATFORM); the extension memory-optout "
                "write on a tenant surface lands tenant-side — known hazard, audit target",
    },
    "users.is_canary": {
        "authority": "platform",
        "write_paths": ["operator SQL only — no code writer (grep 2026-08-12)"],
        "sync": "none",
        "note": "read by rollout_service canary selection; " + _DEAD_TENANT,
    },
    "users.first_media_played_at": {
        "authority": "platform",
        "write_paths": [
            "POST /api/account/media-played (app/api/account.py, platform — the only writer)",
            "alembic 086 legacy backfill (platform)",
        ],
        "sync": "none",
        "note": "write-once monotonic latch (COALESCE) behind the progressively-disclosed Media nav "
                "entry; read by GET /auth/me. The agent never reads or writes it — " + _DEAD_TENANT
                + " — but the column must exist tenant-side because the ORM model declares it "
                  "(init_db _alter_statements carries the mirror)",
    },
    # ── agent_configs ─────────────────────────────────────────────────
    "agent_configs.id": {
        "authority": "immutable-after-create",
        "write_paths": [
            "platform row create (checkout/toup_code/agent_setup _get_or_create_config)",
            "tenant row create (/admin/bind identity reset, soul-sync receiver — fresh uuid)",
        ],
        "sync": "none",
        "note": "NOT a cross-DB join key: the tenant row is materialized with its OWN uuid — probes must join on user_id",
    },
    "agent_configs.user_id": {
        "authority": "immutable-after-create",
        "write_paths": ["same row-create paths as agent_configs.id"],
        "sync": "none",
        "note": "the cross-DB join key for this table",
    },
    "agent_configs.hosting_mode": {
        "authority": "platform",
        "write_paths": [
            "signup priming (app/api/auth.py)",
            _CFG_PATCH,
            "app/api/vps.py (platform)",
        ],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "agent_configs.ssh_host": {
        "authority": "platform",
        "write_paths": ["app/api/vps.py + provisioning services (hostinger/aws/hetzner, platform)"],
        "sync": "none",
        "note": "self-hosted VPS provisioning state; " + _DEAD_TENANT,
    },
    "agent_configs.ssh_port": {
        "authority": "platform",
        "write_paths": ["app/api/vps.py + provisioning services (platform)"],
        "sync": "none",
        "note": "self-hosted VPS provisioning state; " + _DEAD_TENANT,
    },
    "agent_configs.ssh_user": {
        "authority": "platform",
        "write_paths": ["app/api/vps.py + provisioning services (platform)"],
        "sync": "none",
        "note": "self-hosted VPS provisioning state; " + _DEAD_TENANT,
    },
    "agent_configs.ssh_password": {
        "authority": "platform",
        "write_paths": ["app/api/vps.py + provisioning services (platform)"],
        "sync": "none",
        "note": "platform-held credential; " + _DEAD_TENANT,
    },
    "agent_configs.ssh_key": {
        "authority": "platform",
        "write_paths": ["app/api/vps.py + provisioning services (platform)"],
        "sync": "none",
        "note": "platform-held credential; " + _DEAD_TENANT,
    },
    "agent_configs.target_os": {
        "authority": "platform",
        "write_paths": ["app/api/vps.py (platform)"],
        "sync": "none",
        "note": "self-hosted VPS provisioning state; " + _DEAD_TENANT,
    },
    "agent_configs.llm_mode": {
        "authority": "platform",
        "write_paths": [
            "billing bundle webhook (app/api/billing.py, platform)",
            "free_tier_activation (platform)",
            _CFG_PATCH,
        ],
        "sync": _BRIDGE_ENV,
        "note": "platform decides billing mode; the agent learns it from container env, not the tenant column",
    },
    "agent_configs.llm_mode_pre_v2": {
        "authority": "platform",
        "write_paths": ["no live writer found (grep 2026-08-12)"],
        "sync": "none",
        "note": "legacy snapshot from the llm-mode v2 migration; no live writer found (grep 2026-08-12); default platform",
    },
    "agent_configs.openai_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE,
    },
    "agent_configs.anthropic_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE,
    },
    "agent_configs.google_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE,
    },
    "agent_configs.mistral_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE,
    },
    "agent_configs.groq_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE + " (groq is NOT in the bridge body whitelist — env delivery unverified, audit target)",
    },
    "agent_configs.xai_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE,
    },
    "agent_configs.deepseek_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE,
    },
    "agent_configs.agent_name": {
        "authority": "platform",
        "write_paths": [
            "PUT /api/soul save (app/api/soul.py, platform)",
            _CFG_PATCH,
            "/admin/bind identity reset (app/api/admin_pool.py, tenant — from bind payload)",
            "PUT /api/soul/sync receiver (app/api/soul.py, tenant)",
        ],
        "sync": "soul-sync write-through",
        "note": "THE 2026-08-11 defect column: Soul page wrote platform, chat + W-6 assembler read tenant → "
                "nameless agents. Platform Soul save is authoritative; _sync_soul_to_vps pushes it through "
                "PUT /api/soul/sync (PR #599 makes that write-through fail-closed)",
    },
    "agent_configs.agent_model": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "model config is platform-owned (and FROZEN); the agent reads AGENT_MODEL from container env — "
                "the tenant column is dead (the D-2 fossil was a stale container env var, not this column)",
    },
    "agent_configs.app_builder_planner_model": {
        "authority": "platform",
        "write_paths": ["no live writer found (grep 2026-08-12)"],
        "sync": "none",
        "note": "model overrides come from env-level settings via model_resolver, not this column; default platform",
    },
    "agent_configs.app_builder_builder_model": {
        "authority": "platform",
        "write_paths": ["no live writer found (grep 2026-08-12)"],
        "sync": "none",
        "note": "model overrides come from env-level settings via model_resolver, not this column; default platform",
    },
    "agent_configs.preferred_provider": {
        "authority": "platform",
        "write_paths": ["agent-setup provider endpoint (app/api/agent_setup.py:1455, platform)"],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "agent_configs.bundle_stripe_subscription_id": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_status": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)", "scheduled_tasks bundle jobs (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_started_at": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_current_period_end": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)", "scheduled_tasks renewal (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_cancelled_at": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)", "scheduled_tasks cancel-grace job (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.llm_token_hash": {
        "authority": "platform",
        "write_paths": [
            "billing/vps token mint (platform)",
            "free_tier_activation (platform)",
            "llm_proxy + search_proxy self-heal (platform)",
        ],
        "sync": "none",
        "note": "hash of the TOUP_TOKEN the proxy validates; the token itself reaches the agent via bind env; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_anthropic_budget_cents": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)", "free_tier_activation (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_openai_budget_cents": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)", "free_tier_activation (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_anthropic_daily_cap_cents": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_period_start": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)", "scheduled_tasks renewal (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_period_end": {
        "authority": "platform",
        "write_paths": ["billing bundle webhook (platform)", "scheduled_tasks renewal (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_openai_project_id": {
        "authority": "platform",
        "write_paths": ["free_tier_activation / billing project provisioning (platform)"],
        "sync": "none",
        "note": "billing is platform-only; " + _DEAD_TENANT,
    },
    "agent_configs.bundle_openai_api_key": {
        "authority": "platform",
        "write_paths": ["free_tier_activation / billing project provisioning (platform)"],
        "sync": "none",
        "note": "platform-held credential (per-user OpenAI project key); " + _DEAD_TENANT,
    },
    "agent_configs.telegram_bot_token": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "channel " + _BYOK_NOTE,
    },
    "agent_configs.discord_bot_token": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "channel " + _BYOK_NOTE,
    },
    "agent_configs.slack_bot_token": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "channel " + _BYOK_NOTE,
    },
    "agent_configs.slack_app_token": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "channel " + _BYOK_NOTE,
    },
    "agent_configs.whatsapp_phone_number_id": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "WhatsApp Cloud API config; " + _BYOK_NOTE,
    },
    "agent_configs.whatsapp_access_token": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "WhatsApp Cloud API " + _BYOK_NOTE,
    },
    "agent_configs.whatsapp_verify_token": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "WhatsApp Cloud API " + _BYOK_NOTE,
    },
    "agent_configs.whatsapp_app_secret": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "WhatsApp Cloud API " + _BYOK_NOTE,
    },
    "agent_configs.whatsapp_connected_at": {
        "authority": "tenant",
        "write_paths": [
            "first inbound Cloud API webhook (app/agent/channels/whatsapp_channel.py:271, tenant)",
        ],
        "sync": "none",
        "note": "LIVE SPLIT of the class this map exists to catch (flagged 2026-08-12, Cloud API mode only): the "
                "tenant writes it, but agent_setup's status response reads the PLATFORM copy, which has NO writer "
                "— platform-side 'connected' state can never flip. Audit target",
    },
    "agent_configs.whatsapp_mode": {
        "authority": "platform",
        "write_paths": [
            "signup priming (app/api/auth.py)",
            "agent-setup qr-start/pair-code (platform)",
            _CFG_PATCH,
        ],
        "sync": _BRIDGE_ENV,
        "note": "platform picks the mode and pushes it to the container env; tenant column receive-only/dead",
    },
    "agent_configs.whatsapp_self_e164": {
        "authority": "platform",
        "write_paths": ["agent-setup qr-status writeback from agent snapshot (platform)"],
        "sync": _BRIDGE_ENV,
        "note": "platform persists the sidecar-reported linked number; the agent's truth is the sidecar session, not the tenant column",
    },
    "agent_configs.whatsapp_baileys_allowlist": {
        "authority": "platform",
        "write_paths": [
            "agent-setup pair-code pre-seed + qr-status post-link seed (platform)",
            _CFG_PATCH,
        ],
        "sync": _BRIDGE_ENV,
        "note": "enforced by the sidecar from env; tenant column receive-only/dead",
    },
    "agent_configs.whatsapp_session_status": {
        "authority": "platform",
        "write_paths": ["agent-setup qr-start/pair-code/qr-status writeback (platform)"],
        "sync": "none",
        "note": "platform-side UI cache of the sidecar-reported session state; " + _DEAD_TENANT,
    },
    "agent_configs.claude_code_oauth_token": {
        "authority": "platform",
        "write_paths": ["toup-code token save (app/api/toup_code.py:168, platform)"],
        "sync": "none",
        "note": "platform-held credential; " + _DEAD_TENANT,
    },
    "agent_configs.openai_codex_token": {
        "authority": "platform",
        "write_paths": ["toup-code token save (app/api/toup_code.py:168, platform)"],
        "sync": "none",
        "note": "platform-held credential; " + _DEAD_TENANT,
    },
    "agent_configs.brave_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE,
    },
    "agent_configs.elevenlabs_api_key": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": _BRIDGE_ENV,
        "note": "BYOK " + _BYOK_NOTE,
    },
    "agent_configs.agent_api_key": {
        "authority": "platform",
        "write_paths": [
            "agent-setup register (platform)",
            "agent_key_rotation service (platform)",
            "app/api/vps.py + pool_service (platform)",
        ],
        "sync": "delivered to the agent process via bind env (AGENT_API_KEY), not via the tenant column",
        "note": "platform-held credential; " + _DEAD_TENANT,
    },
    "agent_configs.agent_url": {
        "authority": "platform",
        "write_paths": ["agent-setup register (platform)", "vps/docker_host provisioning (platform)"],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "agent_configs.deploy_status": {
        "authority": "platform",
        "write_paths": ["vps/docker_host provisioning (platform)"],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "agent_configs.deploy_log": {
        "authority": "platform",
        "write_paths": ["vps/docker_host provisioning (platform)"],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "agent_configs.setup_completed": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": "none",
        "note": "onboarding UI state; " + _DEAD_TENANT,
    },
    "agent_configs.setup_step": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": "none",
        "note": "onboarding UI state; " + _DEAD_TENANT,
    },
    "agent_configs.setup_type": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": "none",
        "note": "onboarding UI state; " + _DEAD_TENANT,
    },
    "agent_configs.onboarding_completed": {
        "authority": "platform",
        "write_paths": [
            "PUT /api/soul save + agent-setup register (platform)",
            _CFG_PATCH,
            "PUT /api/soul/sync receiver (tenant)",
        ],
        "sync": "soul-sync write-through (carried in agent_config_updates on the onboarding-completing save)",
        "note": "platform owns onboarding state; tenant copy receive-only",
    },
    "agent_configs.agent_color": {
        "authority": "platform",
        "write_paths": [
            "PUT /api/soul save (platform)",
            _CFG_PATCH,
            "/admin/bind identity reset (tenant — from bind payload)",
            "PUT /api/soul/sync receiver (tenant)",
        ],
        "sync": "soul-sync write-through",
        "note": "same authority + write-through as agent_name (PR #599 fail-closed); also pushed as container env for the orb",
    },
    "agent_configs.disabled_tools": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": "none",
        "note": "SUSPECTED LIVE SPLIT (flagged 2026-08-12): written platform-side only, but agent_runner.py:1334 and "
                "ws_realtime.py:2793 read the TENANT copy (empty) — per-user tool disables may not apply in tenant "
                "chat. Not in the bridge body whitelist either. Audit target",
    },
    "agent_configs.subagent_spawning_enabled": {
        "authority": "platform",
        "write_paths": ["admin users toggle (app/api/admin/users.py:1542, platform)"],
        "sync": "none",
        "note": "admin toggle; agent-side enforcement reads env-level settings — tenant column dead",
    },
    "agent_configs.connect_token": {
        "authority": "platform",
        "write_paths": [
            "billing bundle webhook (platform)",
            "agent-setup generate_connect_token (platform)",
            "vps/pool_service/free_tier_activation (platform)",
        ],
        "sync": _BRIDGE_ENV,
        "note": "terminal-agent pairing credential, validated platform-side (ws_agent_tunnel); " + _DEAD_TENANT,
    },
    "agent_configs.db_mode": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH, "app/api/vps.py (platform)"],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "agent_configs.supabase_url": {
        "authority": "platform",
        "write_paths": [_CFG_PATCH],
        "sync": "none",
        "note": _DEAD_TENANT,
    },
    "agent_configs.created_at": {
        "authority": "immutable-after-create",
        "write_paths": ["row INSERT on either side"],
        "sync": "none",
        "note": "row-local; platform row and tenant row are created at different times by design — never compare",
    },
    "agent_configs.updated_at": {
        "authority": "both-synced",
        "write_paths": ["every local UPDATE on either side"],
        "sync": "none",
        "note": "ORM bookkeeping — each database stamps its own copy; divergence expected, exclude from drift probes",
    },
    # ── extension_devices ─────────────────────────────────────────────
    # Rows are SURFACE-LOCAL: a device pairs against exactly one surface
    # and its row lives in that surface's DB (the module docstring says
    # rows persist on the platform DB — tenant-side /ws/extension
    # heartbeats reach the platform over HTTP). The same row never
    # exists in both DBs, so cross-DB value drift is structurally N/A;
    # the declared authority names where rows are EXPECTED to live.
    "extension_devices.id": {
        "authority": "immutable-after-create",
        "write_paths": ["_record_pairing / legacy-row materialize (app/api/extension.py, both-mounted)"],
        "sync": "none",
        "note": "surface-local row uuid; rows are not mirrored across DBs",
    },
    "extension_devices.user_id": {
        "authority": "immutable-after-create",
        "write_paths": ["_record_pairing (extension.py, both-mounted)"],
        "sync": "none",
        "note": "owner of the paired device; set once at pairing",
    },
    "extension_devices.device_name": {
        "authority": "platform",
        "write_paths": ["_record_pairing pair-approve/pair-issue (extension.py — JWT-authed, runs platform-side)"],
        "sync": "none",
        "note": "pairing endpoints require platform JWT auth, so rows land in the platform DB",
    },
    "extension_devices.extension_version": {
        "authority": "platform",
        "write_paths": ["_record_pairing (extension.py, platform-side in practice)"],
        "sync": "none",
        "note": "set at pairing; same surface-local reasoning as device_name",
    },
    "extension_devices.token_jti": {
        "authority": "immutable-after-create",
        "write_paths": ["_record_pairing (extension.py)"],
        "sync": "none",
        "note": "token id (not the token) — set at pairing, used for revoke matching",
    },
    "extension_devices.paired_at": {
        "authority": "immutable-after-create",
        "write_paths": ["_record_pairing (extension.py)"],
        "sync": "none",
        "note": "set once at pairing",
    },
    "extension_devices.last_seen_at": {
        "authority": "platform",
        "write_paths": ["_touch_last_seen heartbeat upsert (extension.py — tenant WS heartbeats arrive over HTTP)"],
        "sync": "none",
        "note": "heartbeat freshness for the settings page, which reads the platform DB",
    },
    "extension_devices.revoked_at": {
        "authority": "platform",
        "write_paths": ["device revoke endpoints (extension.py:858/867, platform-side)"],
        "sync": "none",
        "note": "revocation must be visible to the settings page (platform reader)",
    },
}
