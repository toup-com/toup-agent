"""Grandfather BYO (`llm_mode='manual'`) users into the bundle proxy.

Revision ID: 058
Revises: 057
Create Date: 2026-05-23

Renamed from `057` → `058` on 2026-05-24 after the discovery that a
sibling migration (`057_agent_configs_subagent_spawning_flag.py`) had
already claimed the `057` revision id. Alembic refused to upgrade
through the duplicate-head state, logging `Multiple head revisions are
present` on every boot — so neither migration ever ran via
`alembic upgrade head`. Both columns landed via `init_db()`'s
`_alter_statements` (the agent self-heal path), but the alembic_version
table stayed pinned at 056. Re-linking this revision to 058 makes the
chain linear again so future deploys advance the version pointer
cleanly.

Onboarding v2 removes the binary "Toup bundle / Bring-your-own" choice
from the LLM step. Going forward every user is on the platform-owned
proxy (`llm_mode='bundle'`) and the credit-tier subscription determines
caps/throughput. This migration:

  1. Adds `agent_configs.llm_mode_pre_v2 VARCHAR(20)` (nullable) so the
     pre-v2 value of every row is preserved verbatim.
  2. Backfills `llm_mode_pre_v2 = llm_mode` for every row.
  3. Coerces `llm_mode = 'bundle'` everywhere it was `'manual'`.

Reversibility: `downgrade()` restores `llm_mode = llm_mode_pre_v2` from
the backfill column. We deliberately do NOT drop the `llm_mode_pre_v2`
column on downgrade — operators may want it as forensic evidence of
who was on BYO before the cutover. A separate cleanup migration can
drop it once we're confident the rollout is permanent.

Why a coerce instead of a flag: the backend has 11+ branches that
gate on `llm_mode == 'manual'`. Coercing the column to `'bundle'`
neutralises every branch in one shot (every consumer's "is manual"
check evaluates to False) without us having to amend each call site
in this PR. The deeper code-path removal lives in a follow-up cleanup
PR scheduled after live data confirms no user remains on `'manual'`.

Pre-/post-migration counts admins should record:

  SELECT llm_mode, COUNT(*) FROM agent_configs GROUP BY llm_mode;

If `'manual'` count is non-trivial (>50 in production) STOP and
review the grandfather strategy — operators may want to email those
users a heads-up before the auto-coerce lands.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.058")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "agent_configs" not in tables:
        # Platform DBs always have this table; agent (per-tenant) DBs
        # don't. Skipping is correct — agent DBs never carry the
        # llm_mode column the platform's webhook handlers do.
        logger.info("[alembic.058] agent_configs absent; skipping (platform-only table)")
        return

    existing_cols = {c["name"] for c in insp.get_columns("agent_configs")}
    if "llm_mode_pre_v2" not in existing_cols:
        op.add_column(
            "agent_configs",
            sa.Column("llm_mode_pre_v2", sa.String(20), nullable=True),
        )
        # Refresh inspector cache after DDL or the subsequent backfill
        # SELECT can hit a stale column list on some dialects.
        insp.clear_cache()

    # Idempotent backfill: only stamp pre_v2 for rows where it's still
    # NULL. Re-runs are safe; rows that already have the snapshot
    # are skipped, which means `downgrade() → upgrade()` cycles don't
    # destroy the original value.
    conn.execute(sa.text(
        "UPDATE agent_configs SET llm_mode_pre_v2 = llm_mode "
        "WHERE llm_mode_pre_v2 IS NULL AND llm_mode IS NOT NULL"
    ))

    # The coerce. Every row that was on 'manual' moves to 'bundle'.
    # Snapshot the count so the migration log is loud enough that
    # operators can verify against their pre-migration audit.
    pre = conn.execute(sa.text(
        "SELECT COUNT(*) FROM agent_configs WHERE llm_mode = 'manual'"
    )).scalar() or 0
    if pre:
        logger.info("[alembic.058] coercing %d BYO users to bundle", pre)
        conn.execute(sa.text(
            "UPDATE agent_configs SET llm_mode = 'bundle' WHERE llm_mode = 'manual'"
        ))
    else:
        logger.info("[alembic.058] no BYO users to coerce")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "agent_configs" not in tables:
        return

    existing_cols = {c["name"] for c in insp.get_columns("agent_configs")}
    if "llm_mode_pre_v2" not in existing_cols:
        # Nothing to restore from.
        logger.warning("[alembic.058] llm_mode_pre_v2 column absent; cannot reverse coerce")
        return

    # Restore every row's pre-v2 value. Rows whose pre_v2 is NULL stay
    # NULL — they were freshly-created (post-v2) users who never had
    # the old binary choice in the first place.
    restored = conn.execute(sa.text(
        "UPDATE agent_configs SET llm_mode = llm_mode_pre_v2 "
        "WHERE llm_mode_pre_v2 IS NOT NULL"
    )).rowcount
    logger.info("[alembic.058] restored llm_mode for %s rows", restored)

    # Deliberately NOT dropping the llm_mode_pre_v2 column here. It's
    # cheap (20 chars per row), serves as forensic evidence, and
    # dropping it makes future re-upgrades non-deterministic. The
    # cleanup PR scheduled after the rollout window can DROP it.
