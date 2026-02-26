"""010 — Schema-enforced extraction: add schema_type to entities

Adds schema_type column to entities table so we can track which
Pydantic schema was used to extract each entity.

Revision ID: 010_schema_extraction
Revises: 009_entity_graph_v2
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "010_schema_extraction"
down_revision = "009_entity_graph_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add schema_type to entities
    op.add_column(
        "entities",
        sa.Column("schema_type", sa.String(50), nullable=True),
    )
    # Index for fast lookup by schema_type
    op.create_index(
        "ix_entities_schema_type",
        "entities",
        ["schema_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_entities_schema_type", table_name="entities")
    op.drop_column("entities", "schema_type")
