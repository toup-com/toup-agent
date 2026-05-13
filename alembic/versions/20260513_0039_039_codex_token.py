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
    # Idempotent: skip the add when the column already exists. Without
    # this guard, a half-succeeded prior deploy (column added but
    # alembic_version row not bumped) crashes every subsequent boot
    # with "column already exists" — the container exits before
    # uvicorn binds, Railway's /health never responds, healthcheck
    # times out at 5 min. Was the actual cause of the
    # 2026-05-13 platform-api deploy loop.
    conn = op.get_bind()
    exists = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='agent_configs' "
            "AND column_name='openai_codex_token'"
        )
    ).scalar()
    if not exists:
        op.add_column(
            "agent_configs",
            sa.Column("openai_codex_token", sa.String(length=2000), nullable=True),
        )


def downgrade() -> None:
    # Tolerate "column already gone" — same idempotency logic, inverted.
    conn = op.get_bind()
    exists = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='agent_configs' "
            "AND column_name='openai_codex_token'"
        )
    ).scalar()
    if exists:
        op.drop_column("agent_configs", "openai_codex_token")
