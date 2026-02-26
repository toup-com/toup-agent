"""008 - Hybrid retrieval: entity_relationships table + tsvector full-text search

Adds:
1. entity_relationships table - direct entity-to-entity links for knowledge graph
2. search_vector tsvector column on memories with GIN index for full-text search
3. Trigger to auto-update search_vector on INSERT/UPDATE
4. Backfill existing memories' search_vector

Revision ID: 008_hybrid_retrieval
Revises: 007_pgvector_embedding
Create Date: 2026-02-13
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "008_hybrid_retrieval"
down_revision = "007_pgvector_embedding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------------
    # 1. entity_relationships table
    # ---------------------------------------------------------------
    op.create_table(
        "entity_relationships",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("source_entity_id", sa.String(36), sa.ForeignKey("entities.id"), nullable=False),
        sa.Column("target_entity_id", sa.String(36), sa.ForeignKey("entities.id"), nullable=False),
        sa.Column("relationship_type", sa.String(100), nullable=False),
        sa.Column("properties_json", sa.Text, nullable=True),
        sa.Column("confidence", sa.Float, default=0.7),
        sa.Column("mention_count", sa.Integer, default=1),
        sa.Column("first_seen_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("last_seen_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Unique constraint: one relationship type per entity pair per user
    op.create_unique_constraint(
        "uq_entity_rel_src_tgt_type",
        "entity_relationships",
        ["source_entity_id", "target_entity_id", "relationship_type"],
    )

    # Indexes for graph traversal
    op.create_index("ix_entity_rel_source", "entity_relationships", ["source_entity_id"])
    op.create_index("ix_entity_rel_target", "entity_relationships", ["target_entity_id"])
    op.create_index("ix_entity_rel_user", "entity_relationships", ["user_id"])
    op.create_index("ix_entity_rel_type", "entity_relationships", ["relationship_type"])

    # ---------------------------------------------------------------
    # 2. search_vector tsvector column on memories
    # ---------------------------------------------------------------
    op.execute("ALTER TABLE memories ADD COLUMN search_vector tsvector")

    # GIN index for fast full-text search
    op.execute(
        "CREATE INDEX ix_memories_search_vector ON memories USING GIN (search_vector)"
    )

    # ---------------------------------------------------------------
    # 3. Trigger to auto-update search_vector on INSERT/UPDATE
    # ---------------------------------------------------------------
    op.execute("""
        CREATE OR REPLACE FUNCTION memories_search_vector_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', COALESCE(NEW.summary, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B') ||
                setweight(to_tsvector('english', COALESCE(NEW.category, '')), 'C');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER memories_search_vector_trigger
        BEFORE INSERT OR UPDATE OF content, summary, category
        ON memories
        FOR EACH ROW
        EXECUTE FUNCTION memories_search_vector_update();
    """)

    # ---------------------------------------------------------------
    # 4. Backfill existing memories
    # ---------------------------------------------------------------
    op.execute("""
        UPDATE memories SET search_vector =
            setweight(to_tsvector('english', COALESCE(summary, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(content, '')), 'B') ||
            setweight(to_tsvector('english', COALESCE(category, '')), 'C');
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS memories_search_vector_trigger ON memories")
    op.execute("DROP FUNCTION IF EXISTS memories_search_vector_update()")
    op.execute("DROP INDEX IF EXISTS ix_memories_search_vector")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS search_vector")
    op.drop_table("entity_relationships")
