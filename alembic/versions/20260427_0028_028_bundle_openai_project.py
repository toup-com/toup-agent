"""Add bundle_openai_project_id + bundle_openai_api_key to agent_configs.

Per-user OpenAI projects are auto-provisioned on bundle activation via the
OpenAI Admin API. The proxy reads `bundle_openai_api_key` on outbound (so
OpenAI bills per project), the agent never sees the key. On admin
user-delete, `bundle_openai_project_id` is used to archive the project so
billing stops.

Distinct from the BYOK `openai_api_key` column above (which flows to the
agent's env in manual/BYOK mode). Bundle keys are PROXY-side only.

Revision ID: 028
Revises: 027
Create Date: 2026-04-27
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "agent_configs",
        sa.Column("bundle_openai_project_id", sa.String(64), nullable=True),
    )
    op.add_column(
        "agent_configs",
        sa.Column("bundle_openai_api_key", sa.String(255), nullable=True),
    )


def downgrade():
    op.drop_column("agent_configs", "bundle_openai_api_key")
    op.drop_column("agent_configs", "bundle_openai_project_id")
