"""007 — Enable pgvector and add native embedding columns

Revision ID: 007_pgvector_embedding
Revises: 006_dm_pairing
Create Date: 2026-02-13

Migrates from embedding_json (TEXT, ~15KB JSON per row, Python-side cosine similarity)
to native pgvector embedding columns with HNSW index for 100-1000x faster searches.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '007_pgvector_embedding'
down_revision = '006_dm_pairing'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Enable pgvector extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Add native vector(1536) column to memories table
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding vector(1536)")

    # 3. Migrate existing data: convert JSON text → native vector
    #    The embedding_json column stores a JSON array like "[0.123, -0.456, ...]"
    #    We need to strip the JSON brackets and cast to vector
    op.execute("""
        UPDATE memories
        SET embedding = embedding_json::vector
        WHERE embedding_json IS NOT NULL
          AND embedding IS NULL
    """)

    # 4. Create HNSW index for fast approximate nearest neighbor search
    #    Using cosine distance (vector_cosine_ops) since that's what we use for similarity
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_memories_embedding_hnsw
        ON memories USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # 5. Add vector column to entities table
    op.execute("ALTER TABLE entities ADD COLUMN IF NOT EXISTS embedding vector(1536)")

    # 6. Migrate entities data
    op.execute("""
        UPDATE entities
        SET embedding = embedding_json::vector
        WHERE embedding_json IS NOT NULL
          AND embedding IS NULL
    """)

    # 7. Add vector column to document_chunks table
    op.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector(1536)")

    # 8. Migrate document_chunks data
    op.execute("""
        UPDATE document_chunks
        SET embedding = embedding_json::vector
        WHERE embedding_json IS NOT NULL
          AND embedding IS NULL
    """)

    # 9. Add vector column to messages table
    op.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS embedding vector(1536)")

    # 10. Migrate messages data
    op.execute("""
        UPDATE messages
        SET embedding = embedding_json::vector
        WHERE embedding_json IS NOT NULL
          AND embedding IS NULL
    """)


def downgrade() -> None:
    # Remove vector columns (data preserved in embedding_json)
    op.execute("DROP INDEX IF EXISTS ix_memories_embedding_hnsw")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE entities DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS embedding")
    # Note: We don't drop the vector extension as other things might use it
