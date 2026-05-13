"""openai_codex_token — second provider for Toup Code IDE

Revision ID: 039
Revises: 038
Create Date: 2026-05-13

Adds a sibling column to `claude_code_oauth_token` for the OpenAI Codex
CLI. Toup Code's first-time screen now shows two cards (Claude Code +
GPT Codex) and routes /code/spawn to whichever CLI the user picked.

Additive change; safe to roll out alongside running pods. Same trust
model as the existing channel-token columns (plaintext, rotate to
column-level encryption before opening Toup Code to the public).
"""

from alembic import op
import sqlalchemy as sa


revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_configs",
        sa.Column("openai_codex_token", sa.String(length=2000), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_configs", "openai_codex_token")
