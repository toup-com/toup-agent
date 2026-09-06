"""101 — durable BuildJob claims and voice-task delivery cursors.

Revision ID: 101
Revises: 100
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "101"
down_revision = "100"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "build_jobs"):
        return
    existing = {c["name"] for c in sa.inspect(bind).get_columns("build_jobs")}
    additions = (
        ("claim_owner", sa.String(64), True, None),
        ("claim_token", sa.String(64), True, None),
        ("claim_expires_at", sa.DateTime(), True, None),
        ("state_revision", sa.Integer(), False, "0"),
        ("delivery_revision", sa.Integer(), False, "0"),
        ("receipt_revision", sa.Integer(), False, "0"),
        ("spoken_revision", sa.Integer(), False, "0"),
    )
    for name, typ, nullable, default in additions:
        if name not in existing:
            op.add_column("build_jobs", sa.Column(
                name, typ, nullable=nullable, server_default=default,
            ))
    indexes = {i["name"] for i in sa.inspect(bind).get_indexes("build_jobs")}
    if "ix_build_jobs_source_claim" not in indexes:
        op.create_index(
            "ix_build_jobs_source_claim", "build_jobs",
            ["source_kind", "status", "claim_expires_at"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "build_jobs"):
        return
    indexes = {i["name"] for i in sa.inspect(bind).get_indexes("build_jobs")}
    if "ix_build_jobs_source_claim" in indexes:
        op.drop_index("ix_build_jobs_source_claim", table_name="build_jobs")
    existing = {c["name"] for c in sa.inspect(bind).get_columns("build_jobs")}
    for name in (
        "spoken_revision", "receipt_revision", "delivery_revision",
        "state_revision", "claim_expires_at", "claim_token", "claim_owner",
    ):
        if name in existing:
            op.drop_column("build_jobs", name)
