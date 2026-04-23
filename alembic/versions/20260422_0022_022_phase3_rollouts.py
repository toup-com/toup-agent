"""Phase 3: rollouts + rollout_attempts + pin_image_tag + is_canary

Revision ID: 022
Revises: 021
Create Date: 2026-04-22

Lands all DB schema changes for the Phase 3 automated deployment pipeline:

  1. `rollouts` — audit log of every deployment attempt
  2. `rollout_attempts` — per-tenant sub-record (attempt + optional rollback)
  3. `managed_containers.pin_image_tag` — skip this tenant during rollouts
  4. `managed_containers.image_tag` → nullable (was NOT NULL default
     "toup-agent:latest"); the bridge now populates with real SHA tags
  5. `users.is_canary` — single-row flag for the designated rollout canary
     enforced via a partial unique index

The partial unique index is Postgres-specific (SQLite fallback in tests
doesn't enforce it, which is OK — tests don't care about multi-canary).

See docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md for the schema rationale
and docs/new-vps/15-PHASE-3-FULL-SCOPE.md W0 for acceptance criteria.
"""
from alembic import op
import sqlalchemy as sa


revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade():
    # ── 1. rollouts table ──────────────────────────────────────────
    op.create_table(
        "rollouts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("image_tag", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("trigger", sa.String(32), nullable=False),
        sa.Column("triggered_by", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("canary_prefix", sa.String(8), nullable=True),
        sa.Column("canary_wait_minutes", sa.Integer, nullable=False, server_default="10"),
        sa.Column("started_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
    )
    op.create_index("ix_rollouts_image_tag", "rollouts", ["image_tag"])
    op.create_index("ix_rollouts_status", "rollouts", ["status"])
    op.create_index("ix_rollouts_started_at", "rollouts", ["started_at"])

    # ── 2. rollout_attempts table ──────────────────────────────────
    op.create_table(
        "rollout_attempts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "rollout_id",
            sa.String(36),
            sa.ForeignKey("rollouts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "container_id",
            sa.String(36),
            sa.ForeignKey("managed_containers.id"),
            nullable=False,
        ),
        sa.Column("prior_tag", sa.String(255), nullable=False),
        sa.Column("new_tag", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("health_checks_passed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Integer, nullable=True),
    )
    op.create_index("ix_rollout_attempts_rollout_id", "rollout_attempts", ["rollout_id"])
    op.create_index("ix_rollout_attempts_container_id", "rollout_attempts", ["container_id"])
    op.create_index("ix_rollout_attempts_status", "rollout_attempts", ["status"])

    # ── 3. managed_containers.pin_image_tag ────────────────────────
    op.add_column(
        "managed_containers",
        sa.Column("pin_image_tag", sa.String(255), nullable=True),
    )

    # ── 4. managed_containers.image_tag → nullable + widen to 255 ──
    # Was: String(100), NOT NULL DEFAULT "toup-agent:latest"
    # Now: String(255), NULL, no server default (bridge populates on create)
    # Any existing "toup-agent:latest" rows stay as-is; the rollout service
    # treats them as "legacy, upgrade on first rollout."
    op.alter_column(
        "managed_containers",
        "image_tag",
        existing_type=sa.String(100),
        type_=sa.String(255),
        nullable=True,
        server_default=None,
        existing_server_default="toup-agent:latest",
    )

    # ── 5. users.is_canary + partial unique index ─────────────────
    op.add_column(
        "users",
        sa.Column("is_canary", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    # Partial unique index: only rows WHERE is_canary=TRUE participate in
    # the uniqueness check. Postgres-specific; SQLite ignores the WHERE
    # (harmless for test environments).
    op.create_index(
        "uq_users_is_canary_partial",
        "users",
        ["is_canary"],
        unique=True,
        postgresql_where=sa.text("is_canary = true"),
    )


def downgrade():
    op.drop_index("uq_users_is_canary_partial", table_name="users")
    op.drop_column("users", "is_canary")

    op.alter_column(
        "managed_containers",
        "image_tag",
        existing_type=sa.String(255),
        type_=sa.String(100),
        nullable=False,
        server_default="toup-agent:latest",
    )
    op.drop_column("managed_containers", "pin_image_tag")

    op.drop_index("ix_rollout_attempts_status", table_name="rollout_attempts")
    op.drop_index("ix_rollout_attempts_container_id", table_name="rollout_attempts")
    op.drop_index("ix_rollout_attempts_rollout_id", table_name="rollout_attempts")
    op.drop_table("rollout_attempts")

    op.drop_index("ix_rollouts_started_at", table_name="rollouts")
    op.drop_index("ix_rollouts_status", table_name="rollouts")
    op.drop_index("ix_rollouts_image_tag", table_name="rollouts")
    op.drop_table("rollouts")
