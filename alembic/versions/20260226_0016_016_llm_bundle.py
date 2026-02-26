"""LLM bundle: new provider keys, bundle subscription, allocation & usage tables

Revision ID: 20260226_0016
Revises: 015_agent_config
Create Date: 2026-02-26

Adds:
- agent_configs: llm_mode, new provider API key columns, bundle subscription fields
- llm_bundle_allocations: per-user per-provider budget allocation
- llm_usage_records: individual API call usage records
"""
from alembic import op
import sqlalchemy as sa

revision = "016_llm_bundle"
down_revision = "015_agent_config"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── New columns on agent_configs ─────────────────────────
    op.add_column("agent_configs", sa.Column("llm_mode", sa.String(20), nullable=False, server_default="manual"))
    op.add_column("agent_configs", sa.Column("google_api_key", sa.String(500), nullable=True))
    op.add_column("agent_configs", sa.Column("mistral_api_key", sa.String(500), nullable=True))
    op.add_column("agent_configs", sa.Column("groq_api_key", sa.String(500), nullable=True))
    op.add_column("agent_configs", sa.Column("xai_api_key", sa.String(500), nullable=True))
    op.add_column("agent_configs", sa.Column("deepseek_api_key", sa.String(500), nullable=True))
    op.add_column("agent_configs", sa.Column("bundle_stripe_subscription_id", sa.String(255), nullable=True))
    op.add_column("agent_configs", sa.Column("bundle_status", sa.String(20), nullable=False, server_default="none"))
    op.add_column("agent_configs", sa.Column("bundle_started_at", sa.DateTime, nullable=True))
    op.add_column("agent_configs", sa.Column("bundle_current_period_end", sa.DateTime, nullable=True))

    # ── llm_bundle_allocations ───────────────────────────────
    op.create_table(
        "llm_bundle_allocations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("allocation_cents", sa.Integer, nullable=False, server_default="0"),
        sa.Column("used_cents", sa.Integer, nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_llm_alloc_user_provider", "llm_bundle_allocations", ["user_id", "provider"], unique=True)

    # ── llm_usage_records ────────────────────────────────────
    op.create_table(
        "llm_usage_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("session_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_llm_usage_user_created", "llm_usage_records", ["user_id", "created_at"])
    op.create_index("ix_llm_usage_user_provider", "llm_usage_records", ["user_id", "provider"])


def downgrade() -> None:
    op.drop_index("ix_llm_usage_user_provider", "llm_usage_records")
    op.drop_index("ix_llm_usage_user_created", "llm_usage_records")
    op.drop_table("llm_usage_records")
    op.drop_index("ix_llm_alloc_user_provider", "llm_bundle_allocations")
    op.drop_table("llm_bundle_allocations")
    op.drop_column("agent_configs", "bundle_current_period_end")
    op.drop_column("agent_configs", "bundle_started_at")
    op.drop_column("agent_configs", "bundle_status")
    op.drop_column("agent_configs", "bundle_stripe_subscription_id")
    op.drop_column("agent_configs", "deepseek_api_key")
    op.drop_column("agent_configs", "xai_api_key")
    op.drop_column("agent_configs", "groq_api_key")
    op.drop_column("agent_configs", "mistral_api_key")
    op.drop_column("agent_configs", "google_api_key")
    op.drop_column("agent_configs", "llm_mode")
