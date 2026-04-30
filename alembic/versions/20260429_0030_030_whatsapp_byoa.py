"""Add whatsapp_verify_token, whatsapp_app_secret, whatsapp_connected_at to agent_configs.

These three columns are the per-tenant credentials and status for the
WhatsApp BYOA (Bring Your Own Meta App) onboarding flow. Each user
creates their own Meta App in Meta Business Manager, configures the
webhook to point at their own agent's subdomain, and pastes the
credentials into the platform Settings page. The platform writes them
into agent_configs and the env-sync path injects them into the user's
agent container via .env.

* whatsapp_verify_token  — user-chosen string echoed back during the
  GET handshake to /api/whatsapp/webhook.
* whatsapp_app_secret    — used to HMAC-verify the X-Hub-Signature-256
  on every inbound POST.
* whatsapp_connected_at  — set the first time the agent receives a
  real (non-handshake) webhook; surfaces the green "Connected" badge.

All three are nullable so existing rows remain valid without backfill.
The phone_number_id and access_token columns already exist (created in
migration 015).

Revision ID: 030
Revises: 029
Create Date: 2026-04-29
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "agent_configs",
        sa.Column("whatsapp_verify_token", sa.String(100), nullable=True),
    )
    op.add_column(
        "agent_configs",
        sa.Column("whatsapp_app_secret", sa.String(200), nullable=True),
    )
    op.add_column(
        "agent_configs",
        sa.Column("whatsapp_connected_at", sa.DateTime(), nullable=True),
    )


def downgrade():
    op.drop_column("agent_configs", "whatsapp_connected_at")
    op.drop_column("agent_configs", "whatsapp_app_secret")
    op.drop_column("agent_configs", "whatsapp_verify_token")
