"""Support Phase-0 diagnosis-quality grade columns on support_issues (PLATFORM-only).

Revision ID: 066
Revises: 065
Create Date: 2026-06-15

Adds four PLATFORM-only columns to support_issues backing the Phase-0
diagnosis-quality grading harness (see app/api/support.py grade_issue + the
Phase-0 section of docs/support-agent/README.md):

    grade_verdict        VARCHAR(40)   -- app.support.enums.SupportGradeVerdict
    grade_note           TEXT
    graded_by_user_id    VARCHAR(36)
    graded_at            TIMESTAMP

Grading is an admin ANNOTATION of how good a diagnosis was — strictly separate
from the approval gate; it never advances the lifecycle. Indexed on
grade_verdict for the corpus tally's grouped count.

Idempotent + platform-DB-guarded (skips agent DBs). support_issues is in
PLATFORM_ONLY_TABLES; init_db create_all already excludes it from agent DBs and
this migration short-circuits there too. Mirrored in init_db._alter_statements
because the platform-api boots via init_db and runs alembic only best-effort.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "066"
down_revision = "065"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.066")

_COLUMNS = {
    "grade_verdict": sa.Column("grade_verdict", sa.String(40), nullable=True),
    "grade_note": sa.Column("grade_note", sa.Text(), nullable=True),
    "graded_by_user_id": sa.Column("graded_by_user_id", sa.String(36), nullable=True),
    "graded_at": sa.Column("graded_at", sa.DateTime(), nullable=True),
}


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    table_names = set(insp.get_table_names())

    # Platform DB only: user_sessions is a platform-only sentinel table.
    if "user_sessions" not in table_names:
        logger.info("[alembic.066] user_sessions absent; not a platform DB — skipping")
        return
    if "support_issues" not in table_names:
        logger.info("[alembic.066] support_issues absent; mig 064 not applied — skipping")
        return

    existing = {c["name"] for c in insp.get_columns("support_issues")}
    for name, column in _COLUMNS.items():
        if name not in existing:
            op.add_column("support_issues", column)
            logger.info("[alembic.066] added support_issues.%s", name)

    index_names = {ix["name"] for ix in insp.get_indexes("support_issues")}
    if "ix_support_issues_grade_verdict" not in index_names:
        op.create_index(
            "ix_support_issues_grade_verdict", "support_issues", ["grade_verdict"],
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    table_names = set(insp.get_table_names())
    if "user_sessions" not in table_names or "support_issues" not in table_names:
        return

    index_names = {ix["name"] for ix in insp.get_indexes("support_issues")}
    if "ix_support_issues_grade_verdict" in index_names:
        op.drop_index("ix_support_issues_grade_verdict", table_name="support_issues")

    existing = {c["name"] for c in insp.get_columns("support_issues")}
    for name in ("graded_at", "graded_by_user_id", "grade_note", "grade_verdict"):
        if name in existing:
            op.drop_column("support_issues", name)
