"""Add Path C / QR-link WhatsApp mode columns to agent_configs.

Four nullable columns — one column per concept, no JSON blob — so the
Settings UI can render Pending / Linked / Logged-out states by reading
plain SQL. Existing tenants are valid as-is; default is NULL (= no
WhatsApp mode picked, falls back to whatever the user configured for
Cloud API).

* whatsapp_mode             "cloud_api" | "qr_link" | NULL
* whatsapp_self_e164        the dedicated number holding the QR session
* whatsapp_baileys_allowlist comma-separated E.164 of senders the agent
                             responds to (inbound from anyone else is
                             silently dropped at the ACL gate)
* whatsapp_session_status   "not_linked" | "linking" | "linked" | "logged_out"

Revision ID: 032
Revises: 031
Create Date: 2026-04-30
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "agent_configs",
        sa.Column("whatsapp_mode", sa.String(20), nullable=True),
    )
    op.add_column(
        "agent_configs",
        sa.Column("whatsapp_self_e164", sa.String(20), nullable=True),
    )
    op.add_column(
        "agent_configs",
        sa.Column("whatsapp_baileys_allowlist", sa.Text, nullable=True),
    )
    op.add_column(
        "agent_configs",
        sa.Column("whatsapp_session_status", sa.String(20), nullable=True),
    )


def downgrade():
    op.drop_column("agent_configs", "whatsapp_session_status")
    op.drop_column("agent_configs", "whatsapp_baileys_allowlist")
    op.drop_column("agent_configs", "whatsapp_self_e164")
    op.drop_column("agent_configs", "whatsapp_mode")
