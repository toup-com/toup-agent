"""extension_devices — persistent Chrome extension pairings

Revision ID: 038
Revises: 037
Create Date: 2026-05-12

Replaces the in-memory `extension_bridge.is_connected` check that powers
the /settings/extensions "paired" badge. That check answered the
question "is this user's WS socket currently registered on THIS process"
— but the platform process and the tenant agent process each have their
own bridge module with their own in-memory dict, so the platform-side
endpoint always returned False for a tenant-side socket, even when the
user was actively chatting with their agent via the sidepanel.

This table is the durable answer to "has the user paired the
extension?". `paired_at` is set on pair_approve / pair_issue,
`last_seen_at` is bumped from the tenant agent over a heartbeat
endpoint, `revoked_at` is set when the user clicks Unpair.

Additive change; safe to roll out alongside running pods.
"""

from alembic import op
import sqlalchemy as sa


revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "extension_devices",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("device_name", sa.String(length=200), nullable=False),
        sa.Column("extension_version", sa.String(length=40), nullable=True),
        sa.Column("token_jti", sa.String(length=64), nullable=True),
        sa.Column("paired_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_seen_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_extension_devices_user_id", "extension_devices", ["user_id"])
    op.create_index("ix_extension_devices_token_jti", "extension_devices", ["token_jti"])
    op.create_index("ix_extension_devices_revoked_at", "extension_devices", ["revoked_at"])
    op.create_index(
        "ix_extension_devices_user_revoked",
        "extension_devices",
        ["user_id", "revoked_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_extension_devices_user_revoked", table_name="extension_devices")
    op.drop_index("ix_extension_devices_revoked_at", table_name="extension_devices")
    op.drop_index("ix_extension_devices_token_jti", table_name="extension_devices")
    op.drop_index("ix_extension_devices_user_id", table_name="extension_devices")
    op.drop_table("extension_devices")
