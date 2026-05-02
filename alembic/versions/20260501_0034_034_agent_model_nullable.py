"""Drop NOT NULL on agent_configs.agent_model.

Migration 029 dropped the `server_default='gpt-5.2'` and backfilled the
column to NULL when callers wanted "follow the platform default via
model_resolver", but it never dropped the NOT NULL constraint on the
column itself. Existing rows had `agent_model` set so the gap was
invisible; new signups crashed inside `_get_or_create_config` because
the model declares `nullable=True` and SQLAlchemy INSERTs `NULL`. The
500 propagated up, the frontend's fail-open `.catch` in
`HubPageV2.config()` set `setupComplete=true`, and the user landed on
the chat page with no AgentConfig at all — onboarding silently skipped.

Reproduced 2026-05-01 when a fresh signup ("Arshia") hit chat with no
agent provisioned. Fix is one line; this migration codifies it so
redeploys / fresh DBs don't regress.

Revision ID: 034
Revises: 033
Create Date: 2026-05-01
"""

from alembic import op


revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("agent_configs", "agent_model", nullable=True)


def downgrade():
    op.alter_column("agent_configs", "agent_model", nullable=False)
