"""LLM proxy: usage events table + AgentConfig budget fields

Revision ID: 020
Revises: 019
Create Date: 2026-04-09

Adds llm_proxy_events table for per-request metering, and budget/auth
fields on agent_configs for the LLM proxy.
"""
from alembic import op
import sqlalchemy as sa

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade():
    # New table for proxy usage events
    op.create_table(
        "llm_proxy_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("endpoint", sa.String(20), nullable=False),
        sa.Column("input_tokens", sa.Integer, default=0),
        sa.Column("output_tokens", sa.Integer, default=0),
        sa.Column("cost_cents", sa.Integer, default=0),
        sa.Column("was_fallback", sa.Boolean, default=False),
        sa.Column("latency_ms", sa.Integer, default=0),
        sa.Column("status", sa.String(10), default="ok"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_llm_proxy_user_created", "llm_proxy_events", ["user_id", "created_at"])
    op.create_index("ix_llm_proxy_user_provider_created", "llm_proxy_events", ["user_id", "provider", "created_at"])

    # AgentConfig budget + auth fields
    op.add_column("agent_configs", sa.Column("llm_token_hash", sa.String(128), nullable=True))
    op.add_column("agent_configs", sa.Column("bundle_anthropic_budget_cents", sa.Integer, server_default="3000"))
    op.add_column("agent_configs", sa.Column("bundle_openai_budget_cents", sa.Integer, server_default="1000"))
    op.add_column("agent_configs", sa.Column("bundle_anthropic_daily_cap_cents", sa.Integer, server_default="100"))
    op.add_column("agent_configs", sa.Column("bundle_period_start", sa.DateTime, nullable=True))
    op.add_column("agent_configs", sa.Column("bundle_period_end", sa.DateTime, nullable=True))


def downgrade():
    op.drop_column("agent_configs", "bundle_period_end")
    op.drop_column("agent_configs", "bundle_period_start")
    op.drop_column("agent_configs", "bundle_anthropic_daily_cap_cents")
    op.drop_column("agent_configs", "bundle_openai_budget_cents")
    op.drop_column("agent_configs", "bundle_anthropic_budget_cents")
    op.drop_column("agent_configs", "llm_token_hash")
    op.drop_index("ix_llm_proxy_user_provider_created", table_name="llm_proxy_events")
    op.drop_index("ix_llm_proxy_user_created", table_name="llm_proxy_events")
    op.drop_table("llm_proxy_events")
