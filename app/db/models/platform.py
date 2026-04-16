"""Platform-specific models — VPS provisioning, invites, LLM billing."""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import String, Text, DateTime, Float, Integer, Boolean, ForeignKey, Index
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
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, unique=True)
    container_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)  # Docker container ID
    container_name: Mapped[str] = mapped_column(String(100), nullable=False)
    host_port: Mapped[int] = mapped_column(Integer, nullable=False)  # Mapped port on Docker host
    db_name: Mapped[str] = mapped_column(String(100), nullable=False)  # Per-user PostgreSQL database
    status: Mapped[str] = mapped_column(String(20), default="provisioning")  # provisioning|running|stopped|error
    image_tag: Mapped[str] = mapped_column(String(100), default="toup-agent:latest")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_managed_containers_user_id", "user_id"),
        Index("ix_managed_containers_status", "status"),
        Index("ix_managed_containers_host_port", "host_port", unique=True),
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
    cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    was_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(10), default="ok")  # ok | error
    # Operation classification for billing + budget exemption.
    # NULL or "user.*" = counts toward user caps. "system.*" (e.g. "system.day_archival")
    # = platform-side operations, tracked for cost dashboards but exempt from user caps.
    operation_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_proxy_user_created", "user_id", "created_at"),
        Index("ix_llm_proxy_user_provider_created", "user_id", "provider", "created_at"),
        Index("ix_llm_proxy_operation_type", "operation_type"),
    )
