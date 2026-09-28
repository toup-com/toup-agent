"""107 — desktop_connections: phone pairing by QR, token re-issue and remote tasks.

Revision ID: 107
Revises: 106
Create Date: 2026-09-21

Schema only, and platform only. **Nothing here turns anything on.** The new
routes answer 404 until the `desktop_connections` registry flag is on for an
account, and that flag starts at 0 % with an empty allowlist
(`desktop/docs/CONNECTIONS.md` §12).

Columns added:

  * ``desktop_pairings``: the Mac's model, the session that started the
    pairing, and the QR challenge's lifecycle. Only ``sha256(challenge)`` is
    stored, so a leaked row cannot be replayed; ``qr_claimed_*`` is what makes
    a challenge work once; ``claim_expires_at`` bounds the claiming phone's
    window to decide.
  * ``desktop_devices``: model, how and by which session it was paired, and
    the previous token ``jti`` with its grace deadline (re-issue).
  * ``desktop_pending_actions``: the remote task a card belongs to, the
    session that decided it, and when the Mac said it was asking locally.

Table created: ``desktop_tasks`` (one phone request for one Mac). Unique on
``(user_id, client_request_id)`` so a retried submit cannot create a second
task. ``ON DELETE CASCADE`` to users, like every other ``desktop_*`` table.

No ALTER mirror in ``app/db/database.py`` and no ``SHARED_COLUMN_AUTHORITY``
entries, for the reason 106 gives: these tables have no tenant copy.

Downgrade drops the table and the columns. A downgrade loses in-flight QR
pairings and task history, which is the correct loss for a rollback of the
feature that created them.
"""

from alembic import op
import sqlalchemy as sa

revision = "107"
down_revision = "106"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("desktop_pairings") as b:
        b.add_column(sa.Column("model", sa.String(64), nullable=True))
        b.add_column(sa.Column("init_session_jti", sa.String(64), nullable=True))
        b.add_column(sa.Column("qr_challenge_hash", sa.String(64), nullable=True))
        b.add_column(sa.Column("qr_minted_at", sa.DateTime(), nullable=True))
        b.add_column(sa.Column("qr_expires_at", sa.DateTime(), nullable=True))
        b.add_column(sa.Column("qr_claimed_at", sa.DateTime(), nullable=True))
        b.add_column(
            sa.Column("qr_claimed_session_jti", sa.String(64), nullable=True),
        )
        b.add_column(sa.Column("qr_claimed_label", sa.String(60), nullable=True))
        b.add_column(sa.Column("claim_expires_at", sa.DateTime(), nullable=True))
    op.create_index(
        "ix_desktop_pairings_qr_challenge_hash",
        "desktop_pairings", ["qr_challenge_hash"],
    )

    with op.batch_alter_table("desktop_devices") as b:
        b.add_column(sa.Column("model", sa.String(64), nullable=True))
        b.add_column(sa.Column("paired_via", sa.String(16), nullable=True))
        b.add_column(sa.Column("paired_session_jti", sa.String(64), nullable=True))
        b.add_column(sa.Column("prev_token_jti", sa.String(64), nullable=True))
        b.add_column(sa.Column("prev_jti_expires_at", sa.DateTime(), nullable=True))
    op.create_index(
        "ix_desktop_devices_prev_token_jti",
        "desktop_devices", ["prev_token_jti"],
    )

    with op.batch_alter_table("desktop_pending_actions") as b:
        b.add_column(sa.Column("remote_task_id", sa.String(36), nullable=True))
        b.add_column(
            sa.Column("decided_session_jti", sa.String(64), nullable=True),
        )
        b.add_column(sa.Column("mac_prompt_at", sa.DateTime(), nullable=True))
    op.create_index(
        "ix_desktop_pending_actions_remote_task_id",
        "desktop_pending_actions", ["remote_task_id"],
    )

    op.create_table(
        "desktop_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("device_id", sa.String(36), nullable=False),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="queued",
        ),
        sa.Column(
            "turn_state", sa.String(16), nullable=False, server_default="queued",
        ),
        sa.Column("outcome", sa.String(240), nullable=True),
        sa.Column("conversation_id", sa.String(36), nullable=True),
        sa.Column("client_request_id", sa.String(64), nullable=False),
        sa.Column("submitted_session_jti", sa.String(64), nullable=True),
        sa.Column("submitted_via", sa.String(16), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(), nullable=True),
        sa.Column("cancel_reason", sa.String(16), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'dispatched', 'running', 'waiting_on_you', "
            "'waiting_on_mac', 'done', 'failed', 'cancelled', 'mac_offline')",
            name="ck_desktop_tasks_status",
        ),
        sa.UniqueConstraint(
            "user_id", "client_request_id",
            name="uq_desktop_tasks_user_client_request",
        ),
    )
    op.create_index("ix_desktop_tasks_user_id", "desktop_tasks", ["user_id"])
    op.create_index("ix_desktop_tasks_status", "desktop_tasks", ["status"])
    op.create_index(
        "ix_desktop_tasks_user_device_created",
        "desktop_tasks", ["user_id", "device_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_desktop_tasks_user_device_created", table_name="desktop_tasks",
    )
    op.drop_index("ix_desktop_tasks_status", table_name="desktop_tasks")
    op.drop_index("ix_desktop_tasks_user_id", table_name="desktop_tasks")
    op.drop_table("desktop_tasks")

    op.drop_index(
        "ix_desktop_pending_actions_remote_task_id",
        table_name="desktop_pending_actions",
    )
    with op.batch_alter_table("desktop_pending_actions") as b:
        b.drop_column("mac_prompt_at")
        b.drop_column("decided_session_jti")
        b.drop_column("remote_task_id")

    op.drop_index(
        "ix_desktop_devices_prev_token_jti", table_name="desktop_devices",
    )
    with op.batch_alter_table("desktop_devices") as b:
        b.drop_column("prev_jti_expires_at")
        b.drop_column("prev_token_jti")
        b.drop_column("paired_session_jti")
        b.drop_column("paired_via")
        b.drop_column("model")

    op.drop_index(
        "ix_desktop_pairings_qr_challenge_hash", table_name="desktop_pairings",
    )
    with op.batch_alter_table("desktop_pairings") as b:
        b.drop_column("claim_expires_at")
        b.drop_column("qr_claimed_label")
        b.drop_column("qr_claimed_session_jti")
        b.drop_column("qr_claimed_at")
        b.drop_column("qr_expires_at")
        b.drop_column("qr_minted_at")
        b.drop_column("qr_challenge_hash")
        b.drop_column("init_session_jti")
        b.drop_column("model")
