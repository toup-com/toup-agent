"""Maintenance / support agent: support_issues + support_issue_events.

Revision ID: 064
Revises: 063
Create Date: 2026-06-14

Adds the two PLATFORM-only tables backing the admin maintenance-and-support
system (see app.support + docs/support-agent/README.md):

* support_issues       — one row per user-reported problem moving through
                         intake → classify → triage → diagnose → approve →
                         implement → verify → PR.
* support_issue_events — append-only audit trail of state transitions and
                         admin decisions, FK→support_issues ON DELETE CASCADE.

Idempotent + platform-DB-guarded (skips agent DBs, which never carry these).
Agents boot via init_db create_all, not alembic; these tables are in
PLATFORM_ONLY_TABLES so create_all already excludes them from agent DBs and
this migration short-circuits there too.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "064"
down_revision = "063"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.064")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    table_names = set(insp.get_table_names())

    # Platform DB only: user_sessions is a platform-only sentinel table.
    # Absent ⇒ this is a per-tenant agent DB ⇒ nothing to do here.
    if "user_sessions" not in table_names:
        logger.info("[alembic.064] user_sessions absent; not a platform DB — skipping")
        return

    if "support_issues" not in table_names:
        op.create_table(
            "support_issues",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("reporter_user_id", sa.String(36), nullable=True),
            sa.Column("reporter_email", sa.String(255), nullable=True),
            sa.Column("tenant_id", sa.String(36), nullable=True),
            sa.Column("channel", sa.String(32), nullable=False, server_default="api"),
            sa.Column("raw_report", sa.Text(), nullable=False),
            sa.Column("repro_info", sa.Text(), nullable=True),
            sa.Column("symptom", sa.Text(), nullable=True),
            sa.Column("severity", sa.String(16), nullable=False, server_default="medium"),
            sa.Column("status", sa.String(32), nullable=False, server_default="intake"),
            sa.Column("classification", sa.String(24), nullable=True),
            sa.Column("classification_rationale", sa.Text(), nullable=True),
            sa.Column("routed_subsystems", sa.JSON(), nullable=True),
            sa.Column("routing_rationale", sa.Text(), nullable=True),
            sa.Column("skills_baseline_sha", sa.String(40), nullable=True),
            sa.Column("diagnosis_report", sa.JSON(), nullable=True),
            sa.Column("diagnosis_summary", sa.Text(), nullable=True),
            sa.Column("decision", sa.String(24), nullable=True),
            sa.Column("decision_by_user_id", sa.String(36), nullable=True),
            sa.Column("decision_at", sa.DateTime(), nullable=True),
            sa.Column("decision_notes", sa.Text(), nullable=True),
            sa.Column("branch_name", sa.String(255), nullable=True),
            sa.Column("pr_url", sa.String(512), nullable=True),
            sa.Column("verification", sa.JSON(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.text("now()")),
        )
        op.create_index("ix_support_issues_status", "support_issues", ["status"])
        op.create_index("ix_support_issues_classification", "support_issues", ["classification"])
        op.create_index("ix_support_issues_created_at", "support_issues", ["created_at"])
        logger.info("[alembic.064] created support_issues")
    else:
        logger.info("[alembic.064] support_issues already present; skipping")

    if "support_issue_events" not in table_names:
        op.create_table(
            "support_issue_events",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "issue_id", sa.String(36),
                sa.ForeignKey("support_issues.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("event_type", sa.String(40), nullable=False),
            sa.Column("actor", sa.String(16), nullable=False, server_default="system"),
            sa.Column("actor_user_id", sa.String(36), nullable=True),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("detail", sa.JSON(), nullable=True),
        )
        op.create_index("ix_support_issue_events_issue_id", "support_issue_events", ["issue_id"])
        op.create_index("ix_support_issue_events_created_at", "support_issue_events", ["created_at"])
        logger.info("[alembic.064] created support_issue_events")
    else:
        logger.info("[alembic.064] support_issue_events already present; skipping")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    table_names = set(insp.get_table_names())
    if "user_sessions" not in table_names:
        return

    if "support_issue_events" in table_names:
        op.drop_table("support_issue_events")
    if "support_issues" in table_names:
        op.drop_table("support_issues")
