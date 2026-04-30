"""Add whatsapp_inbound_dedupe table.

Persistent dedupe for WhatsApp Cloud API inbound webhooks. Each row is
keyed on (user_id, message_id); insertion is the only write, deletion
happens via a periodic sweep that drops rows older than the retention
window. See `app/db/models/agent.py::WhatsAppInboundDedupe` for the
full motivation.

Revision ID: 031
Revises: 030
Create Date: 2026-04-29
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "whatsapp_inbound_dedupe",
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("message_id", sa.String(200), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint(
            "user_id", "message_id", name="pk_whatsapp_inbound_dedupe"
        ),
    )
    op.create_index(
        "ix_whatsapp_inbound_dedupe_received_at",
        "whatsapp_inbound_dedupe",
        ["received_at"],
    )


def downgrade():
    op.drop_index(
        "ix_whatsapp_inbound_dedupe_received_at",
        table_name="whatsapp_inbound_dedupe",
    )
    op.drop_table("whatsapp_inbound_dedupe")
