"""009 - Entity graph improvements: relationship_label + entity name tsvector

Adds:
1. relationship_label column to entity_relationships for human-readable labels
2. name_search tsvector generated column on entities for fast name lookup
3. GIN index on entities name_search

Revision ID: 009_entity_graph_v2
Revises: 008_hybrid_retrieval
Create Date: 2026-02-13
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "009_entity_graph_v2"
down_revision = "008_hybrid_retrieval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------------
    # 1. Add relationship_label to entity_relationships
    # ---------------------------------------------------------------
    op.add_column(
        "entity_relationships",
        sa.Column("relationship_label", sa.String(255), nullable=True),
    )

    # ---------------------------------------------------------------
    # 2. Add name_search tsvector column on entities for fast lookup
    # ---------------------------------------------------------------
    # PostgreSQL GENERATED ALWAYS AS ... STORED for auto-maintained tsvector
    op.execute("""
        ALTER TABLE entities
        ADD COLUMN name_search tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', COALESCE(name, ''))) STORED
    """)

    # GIN index for fast full-text search on entity names
    op.execute(
        "CREATE INDEX ix_entities_name_search ON entities USING GIN (name_search)"
    )

    # ---------------------------------------------------------------
    # 3. Backfill relationship_label for existing rows
    # ---------------------------------------------------------------
    op.execute("""
        UPDATE entity_relationships er
        SET relationship_label = (
            SELECT e1.name || ' ' || REPLACE(er.relationship_type, '_', ' ') || ' ' || e2.name
            FROM entities e1, entities e2
            WHERE e1.id = er.source_entity_id
              AND e2.id = er.target_entity_id
        )
        WHERE relationship_label IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_entities_name_search")
    op.execute("ALTER TABLE entities DROP COLUMN IF EXISTS name_search")
    op.drop_column("entity_relationships", "relationship_label")
