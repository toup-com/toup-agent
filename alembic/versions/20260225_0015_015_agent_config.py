"""Agent config table + missing vps_instances.agent_api_key

Revision ID: 20260225_0015
Revises: 20260218_0014
Create Date: 2026-02-25

Adds:
- agent_configs: Per-user agent setup configuration (wizard state, SSH creds, API keys, channels)
- vps_instances.agent_api_key: Missing from migration 014 but present in model
"""
from alembic import op
import sqlalchemy as sa

revision = "015_agent_config"
down_revision = "014_vps_instances"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── agent_configs ──────────────────────────────────────────
    op.create_table(
        "agent_configs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), unique=True, nullable=False),
        # Step 1: Machine / Hosting
        sa.Column("hosting_mode", sa.String(20), nullable=False, server_default="self-hosted"),
        sa.Column("ssh_host", sa.String(255), nullable=True),
        sa.Column("ssh_port", sa.Integer, nullable=False, server_default="22"),
        sa.Column("ssh_user", sa.String(100), nullable=True, server_default="ubuntu"),
        sa.Column("ssh_password", sa.String(255), nullable=True),
        sa.Column("ssh_key", sa.Text, nullable=True),
        # Step 2: LLM
        sa.Column("openai_api_key", sa.String(500), nullable=True),
        sa.Column("anthropic_api_key", sa.String(500), nullable=True),
        sa.Column("agent_model", sa.String(50), nullable=False, server_default="gpt-5.2"),
        # Step 3: Channels
        sa.Column("telegram_bot_token", sa.String(255), nullable=True),
        sa.Column("discord_bot_token", sa.String(255), nullable=True),
        sa.Column("slack_bot_token", sa.String(255), nullable=True),
        sa.Column("slack_app_token", sa.String(255), nullable=True),
        sa.Column("whatsapp_phone_number_id", sa.String(100), nullable=True),
        sa.Column("whatsapp_access_token", sa.String(500), nullable=True),
        # Step 4: Services
        sa.Column("brave_api_key", sa.String(255), nullable=True),
        sa.Column("elevenlabs_api_key", sa.String(255), nullable=True),
        # Deploy state
        sa.Column("agent_api_key", sa.String(100), nullable=True),
        sa.Column("agent_url", sa.String(500), nullable=True),
        sa.Column("deploy_status", sa.String(20), nullable=False, server_default="none"),
        sa.Column("deploy_log", sa.Text, nullable=True),
        # Wizard state
        sa.Column("setup_completed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("setup_step", sa.Integer, nullable=False, server_default="1"),
        # Timestamps
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_agent_configs_user_id", "agent_configs", ["user_id"])

    # ── vps_instances: add missing agent_api_key column ────────
    op.add_column("vps_instances", sa.Column("agent_api_key", sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column("vps_instances", "agent_api_key")
    op.drop_index("ix_agent_configs_user_id", "agent_configs")
    op.drop_table("agent_configs")
