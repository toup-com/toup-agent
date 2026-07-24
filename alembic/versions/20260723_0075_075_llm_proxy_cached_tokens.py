"""Cache telemetry — llm_proxy_events.cached_tokens (F-7 / A9-1).

Revision ID: 075
Revises: 074
Create Date: 2026-07-23

One nullable column from the token-efficiency audit (audit_A9 / verify_F-F):
``usage.prompt_tokens_details.cached_tokens`` (OpenAI) and
``cache_read_input_tokens`` (Anthropic) were captured on every chat call but
never persisted anywhere — the only cache-hit observable was grepping
``[PERF] cache_read=`` in tenant container logs. Storing the per-call count on
``llm_proxy_events`` makes cache-hit rate queryable for all bundle-mode
traffic. Telemetry only: it does NOT change cost_cents or credit math
(A9-2 documented out of scope).

Idempotent + guarded like 070-074: platform DBs only (skips DBs without
``users``); ``llm_proxy_events`` is PLATFORM_ONLY in base.py so the agent-side
init_db create_all path never touches it. The init_db ``_alter_statements``
mirror carries the same ADD COLUMN IF NOT EXISTS statement as the
authoritative heal (the Dockerfile ``alembic upgrade head`` is best-effort).
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "075"
down_revision = "074"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.075")

_TABLE = "llm_proxy_events"
_COLUMN = "cached_tokens"


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.075] users absent; skipping (not a platform DB)")
        return
    if _TABLE not in tables:
        logger.info("[alembic.075] %s absent; skipping", _TABLE)
        return
    existing = {c["name"] for c in insp.get_columns(_TABLE)}
    if _COLUMN in existing:
        logger.info("[alembic.075] %s.%s already present", _TABLE, _COLUMN)
        return
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.Integer(), nullable=True))
    logger.info("[alembic.075] added %s.%s", _TABLE, _COLUMN)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _TABLE in set(insp.get_table_names()):
        existing = {c["name"] for c in insp.get_columns(_TABLE)}
        if _COLUMN in existing:
            op.drop_column(_TABLE, _COLUMN)
