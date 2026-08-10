"""Platform-specific models — VPS provisioning, invites, LLM billing."""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List
import uuid

from sqlalchemy import String, Text, DateTime, Float, Integer, Boolean, ForeignKey, Index, Numeric
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base


class VPSPlan(Base):
    """Available VPS plans that users can choose during signup."""
    __tablename__ = "vps_plans"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    instance_type: Mapped[str] = mapped_column(String(20), nullable=False)
    vcpu: Mapped[int] = mapped_column(Integer, nullable=False)
    ram_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    stripe_price_id: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    provider: Mapped[str] = mapped_column(String(20), nullable=False, default="aws")
    hostinger_plan_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    hetzner_server_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    instances: Mapped[List["VPSInstance"]] = relationship("VPSInstance", back_populates="plan")


class VPSInstance(Base):
    """A provisioned EC2 instance assigned to a user."""
    __tablename__ = "vps_instances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(20), ForeignKey("vps_plans.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    provider: Mapped[str] = mapped_column(String(20), default="aws")
    aws_instance_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    aws_region: Mapped[str] = mapped_column(String(20), default="us-east-1")
    hostinger_vm_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    public_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    public_dns: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ami_id: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    ssh_password: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    agent_api_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stripe_session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, unique=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    provisioned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    terminated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    plan: Mapped["VPSPlan"] = relationship("VPSPlan", back_populates="instances")

    __table_args__ = (
        Index("ix_vps_instances_user_id", "user_id"),
        Index("ix_vps_instances_status", "status"),
        Index("ix_vps_instances_stripe_session", "stripe_session_id"),
    )


class ManagedContainer(Base):
    """A Docker container running a user's agent on the managed host."""
    __tablename__ = "managed_containers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # ON DELETE CASCADE: account deletion removes the container row; the
    # container reconciler / provision path can re-create it during the
    # delete window, so CASCADE keeps DELETE FROM users atomic (mig 060).
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    container_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)  # Docker container ID
    container_name: Mapped[str] = mapped_column(String(100), nullable=False)
    host_port: Mapped[int] = mapped_column(Integer, nullable=False)  # Mapped port on Docker host
    db_name: Mapped[str] = mapped_column(String(100), nullable=False)  # Per-user PostgreSQL database
    status: Mapped[str] = mapped_column(String(20), default="provisioning")  # provisioning|running|stopped|error
    # image_tag: currently-running agent image for this tenant.
    # Phase 2 defaulted to "toup-agent:latest"; Phase 3 makes it nullable and
    # populates it with ghcr.io/toup-com/toup-agent:<sha> via the provisioning
    # bridge on create + every rollout attempt.
    image_tag: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # pin_image_tag: if set, the rollout service SKIPS this tenant (status
    # 'skipped_pinned'). Used to hold a user on a specific SHA — e.g. for
    # regression isolation or branch testing. Admin SQL only in Phase 3; UI
    # deferred until there's real need.
    pin_image_tag: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Auto-updated on any row mutation. Powers the prewarm reconciler's
    # stuck-row detection (status='provisioning' AND updated_at older
    # than ~5min = the in-flight asyncio.create_task died, retry it).
    # `created_at` reflects only the first-ever provision; `started_at`
    # is only set on successful start. Migration 037 added the column
    # with server-default and a (status, updated_at) composite index.
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True,
    )

    __table_args__ = (
        Index("ix_managed_containers_user_id", "user_id"),
        Index("ix_managed_containers_status", "status"),
        Index("ix_managed_containers_host_port", "host_port", unique=True),
        Index("ix_managed_containers_status_updated_at", "status", "updated_at"),
    )


class Invite(Base):
    """Closed-beta invite tokens."""
    __tablename__ = "invites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_by: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="beta_user")
    note: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    used_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_invites_status", "status"),
    )


class LLMBundleAllocation(Base):
    """Per-user, per-provider budget allocation for LLM bundle subscription."""
    __tablename__ = "llm_bundle_allocations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    allocation_cents: Mapped[int] = mapped_column(Integer, default=0)
    used_cents: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_alloc_user_provider", "user_id", "provider", unique=True),
    )


class LLMUsageRecord(Base):
    """Individual LLM API call usage record for bundle billing."""
    __tablename__ = "llm_usage_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_usage_user_created", "user_id", "created_at"),
        Index("ix_llm_usage_user_provider", "user_id", "provider"),
    )


class LLMProxyEvent(Base):
    """LLM proxy usage event — tracks every request through the proxy for metering."""
    __tablename__ = "llm_proxy_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(20), nullable=False)  # anthropic | openai
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(20), nullable=False)  # chat | embeddings | tts
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    # Numeric since alembic 084 (R-3): the 1¢/call floor is gone, so a
    # sub-cent call records its true fractional cost. An Integer column
    # here would silently re-floor every fraction on INSERT.
    cost_cents: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=0)
    was_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(10), default="ok")  # ok | error
    # Operation classification for billing + budget exemption.
    # NULL or "user.*" = counts toward user caps. "system.*" (e.g. "system.day_archival")
    # = platform-side operations, tracked for cost dashboards but exempt from user caps.
    operation_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Prompt-cache read hits for this call (F-7 / A9-1, alembic 075). OpenAI's
    # usage.prompt_tokens_details.cached_tokens / Anthropic's
    # cache_read_input_tokens. Telemetry only — never enters cost_cents or
    # credit math. NULL = recorded before 075 or usage wasn't inspectable.
    cached_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Prompt-cache WRITE tokens (alembic 083). Billed at a premium on part of
    # the gpt-5.6 family (sol/luna 1.25x input; terra's measured write rate
    # equals list input — config.pricing_per_1k). Extracted on both wires and
    # already priced into cost_cents since G1 prep; until 083 it was then
    # DROPPED — the one cache number recoverable only by grepping [CACHE]
    # platform logs. NULL = pre-083 row, or usage wasn't inspectable, or the
    # provider reported no write field (chat wire reports none today; the
    # Responses wire does).
    cache_write_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Surface the turn came from — "web", "voice", "telegram", … (alembic 082).
    #
    # Prompt caching is prefix-exact and the wire TOOLS ARRAY heads the prefix,
    # so a channel that strips a tool starts a separate provider cache lineage
    # and re-bills the whole request. That is a per-channel cost question, and
    # until this column existed it was unanswerable: the value was resolved on
    # the agent (agent_runner passes `channel` through the whole turn) and then
    # dropped at the agent→proxy boundary, and it is NOT recoverable from
    # anything else on the row — operation_type is NULL for every user-facing
    # chat call by design, and the [CACHE] log line carries only an 8-char
    # hash of the cache key.
    #
    # NULL means "not reported": every non-agent caller (embeddings, images,
    # internal_llm system ops) and any agent old enough to predate the header.
    channel: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_proxy_user_created", "user_id", "created_at"),
        Index("ix_llm_proxy_user_provider_created", "user_id", "provider", "created_at"),
        Index("ix_llm_proxy_operation_type", "operation_type"),
        # The query this column exists for: cache hit rate per channel over a
        # window. Leads with created_at because every such query is time-boxed.
        Index("ix_llm_proxy_created_channel", "created_at", "channel"),
    )


class SearchEvent(Base):
    """One row per search served through the platform search gateway.

    Deliberately NOT folded into ``llm_proxy_events``: that table's
    ``provider``/``endpoint`` are String(20) and would have to carry tier
    semantics they were never named for. And deliberately not left in
    ``credit_ledger`` alone — that table is an immutable billing audit trail
    shared by 18 event types, so per-search telemetry there is a JSONB scan
    (``metadata->>'tier'``) with no index, which is how "searches per user per
    day" became a full ledger scan.

    Every column here is a real typed column so the five questions the founder
    asked — how many searches per user per day, how slow, which tier answered,
    what was throttled or fell back, and what did it cost whom — are each one
    indexed query.

    ``query_sha256`` is a 16-hex truncation, never the query text. A search
    query is the most sensitive string a private agent handles; the hash exists
    only to spot a runaway loop repeating one query, which is the failure this
    telemetry is for.
    """
    __tablename__ = "search_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)

    # Which rung of the ladder actually produced the answer.
    # brave_api | cache | httpx_race | browser_brave | none
    tier: Mapped[str] = mapped_column(String(20), nullable=False)
    engine: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)

    # ok | throttled | error
    status: Mapped[str] = mapped_column(String(16), default="ok", nullable=False)
    # Why Brave did not serve this one: tenant_rate_limit | fleet_headroom |
    # cooldown_after_429 | http_429 | unconfigured | upstream_error | empty_result
    degraded_reason: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    # True when a rung BELOW brave_api answered — the founder's "fallback event".
    was_fallback: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Cost attribution. `credits` is the unit users are billed in.
    # `cost_cents` is our own upstream cost and is RESERVED, not populated:
    # Brave is flat-rate per query on this plan, so there is no per-call figure
    # to record. A metered per-call engine would fill it; do not read it as
    # zero-cost. `charged` is False while metering is in dry-run
    # (web_tool_metering_charge), so the usage series exists before the billing
    # does — the same meter_only pattern the credit ledger already uses.
    credits: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    cost_cents: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    charged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # voice | chat | job | routine | trigger — which surface asked.
    channel: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    query_sha256: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # Fleet headroom Brave itself reported on this call (x-ratelimit-remaining,
    # first bucket). This is the ONLY fleet-wide view of the shared 50 rps
    # account ceiling that exists — Brave computes it across every key on the
    # plan, so no coordinator of ours can reproduce it. Recording it is what
    # makes quota alerting possible without a second source of truth.
    brave_remaining: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_search_events_user_created", "user_id", "created_at"),
        Index("ix_search_events_created", "created_at"),
        Index("ix_search_events_status_created", "status", "created_at"),
    )


class PlatformSetting(Base):
    """Generic key-value table for admin-editable runtime settings.

    Used for things that change occasionally and live outside the deploy
    cycle: monthly fixed costs (Anthropic Max sub, VPS bill, Railway plan)
    that the admin needs to edit when contracts change, without touching
    Railway env vars or shipping a deploy.

    Why not env vars: business numbers shouldn't require infra access to
    update. The env-var defaults from settings.py are still honored as
    fallbacks when no DB row exists yet — so newly-deployed environments
    pick up sensible defaults from settings.platform_cost_*_monthly_usd
    and only diverge once an admin saves a custom value.

    Stored as TEXT so the same table can also hold non-numeric settings
    (feature flags, banner messages, etc.) in the future.
    """

    __tablename__ = "platform_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )
    # ON DELETE SET NULL: this is an audit reference, not user-owned data —
    # deleting the admin who last touched a global setting must not be
    # blocked, and must not delete the setting. (mig 060)
    updated_by_user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
