"""agent_configs: per-tenant sub-agent spawning kill-switch.

Follow-up to mig 056 (sub-agent spawning columns on build_jobs).
056 shipped the schema for the new path, but the path itself stays
gated by `settings.subagent_spawning_enabled` — which is an env-var-
backed boolean read by the per-tenant agent process at boot.

Tenant agent containers source their env vars from
``/data/agents/<prefix>/.env`` on the VPS, populated by the bridge
from the dict that ``_agent_config_to_bridge_body`` returns. The dict
is a strict whitelist — fields not in it never reach the .env file,
so the agent's pydantic-settings keeps the False default and spawn
keeps routing to the legacy SubAgentManager (Telegram-only).

This migration adds the per-row toggle column so ops can flip it for
specific tenants without redeploying. The whitelist + AgentEnvContract
wirings land alongside this in the same PR.

Revision ID: 057
Revises: 056
Create Date: 2026-05-23

Purely additive. Reversible. Default ``false`` so existing rows and
new self-served tenants stay on the legacy path until ops opts them
in. SQLite and Postgres both accept ``ADD COLUMN flag BOOLEAN NOT
NULL DEFAULT FALSE`` in one shot; existing rows backfill atomically
via the server_default.

The data step that flips the column to True for the rolling-out
tenants is intentionally NOT in this migration — it's an ops action
(targeted UPDATE) so a future ``alembic downgrade 056 && upgrade 057``
cycle doesn't silently re-enable the feature for an already-rolled-
back tenant. See docs/runbooks/subagent-rollout.md for the targeted
UPDATE statement.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "057"
down_revision = "056"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.057")


_COL_NAME = "subagent_spawning_enabled"


def upgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "agent_configs", _COL_NAME):
        logger.info(
            "[alembic.057] agent_configs.%s already present; skipping",
            _COL_NAME,
        )
        return
    op.add_column(
        "agent_configs",
        sa.Column(
            _COL_NAME,
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    logger.info("[alembic.057] added agent_configs.%s", _COL_NAME)


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "agent_configs", _COL_NAME):
        with op.batch_alter_table("agent_configs") as batch:
            batch.drop_column(_COL_NAME)
        logger.info("[alembic.057] dropped agent_configs.%s", _COL_NAME)


def _column_exists(conn, table: str, column: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        cols = {c["name"] for c in insp.get_columns(table)}
        return column in cols
    except Exception:
        return False
