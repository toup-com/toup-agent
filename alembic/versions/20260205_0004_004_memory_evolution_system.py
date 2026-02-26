"""Memory Evolution System - Add canonical_content, history, merge tracking

Revision ID: 004_memory_evolution
Revises: 003_agent_platform_identity
Create Date: 2026-02-05

This migration adds:
1. canonical_content: The current, most up-to-date version of the memory
2. history_json: JSON array of all versions with timestamps (stored as Text)
3. merged_from_json: Array of memory IDs that were merged into this one (stored as Text)
4. superseded_by: If this memory was merged into another, points to the new one
5. is_active: False if this memory was merged into another (soft archive)
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '004_memory_evolution'
down_revision = '003_agent_platform'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add canonical_content column
    op.add_column('memories', sa.Column('canonical_content', sa.Text(), nullable=True))
    
    # Add history_json column (stored as Text for cross-DB compatibility)
    op.add_column('memories', sa.Column('history_json', sa.Text(), nullable=True))
    
    # Add merged_from_json column (stored as Text)
    op.add_column('memories', sa.Column('merged_from_json', sa.Text(), nullable=True))
    
    # Add superseded_by column
    op.add_column('memories', sa.Column('superseded_by', sa.String(36), nullable=True))
    
    # Add is_active column (may already exist from previous migration)
    try:
        op.add_column('memories', sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'))
    except Exception:
        pass  # Column already exists
    
    # Create index on superseded_by for faster lookups
    op.create_index('ix_memories_superseded_by', 'memories', ['superseded_by'])
    
    # Create index on is_active for filtering active memories
    try:
        op.create_index('ix_memories_is_active', 'memories', ['is_active'])
    except Exception:
        pass  # Index may already exist
    
    # Backfill canonical_content from content for existing memories
    op.execute("""
        UPDATE memories 
        SET canonical_content = content 
        WHERE canonical_content IS NULL
    """)
    
    # Create initial history entries for existing memories (as JSON text)
    op.execute("""
        UPDATE memories 
        SET history_json = '[{"date": "' || to_char(created_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') || '", "content": "' || REPLACE(REPLACE(content, '"', '\\"'), E'\\n', '\\n') || '", "source": "' || COALESCE(source_type, 'manual') || '", "action": "created"}]'
        WHERE history_json IS NULL
    """)
    
    # Initialize merged_from_json as empty array
    op.execute("""
        UPDATE memories 
        SET merged_from_json = '[]'
        WHERE merged_from_json IS NULL
    """)
    
    # Set is_active to true for all existing non-deleted memories
    op.execute("""
        UPDATE memories 
        SET is_active = true 
        WHERE is_active IS NULL AND is_deleted = false
    """)
    
    # Set is_active to false for deleted memories
    op.execute("""
        UPDATE memories 
        SET is_active = false 
        WHERE is_active IS NULL AND is_deleted = true
    """)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_memories_superseded_by', table_name='memories')
    try:
        op.drop_index('ix_memories_is_active', table_name='memories')
    except Exception:
        pass
    
    # Drop columns
    op.drop_column('memories', 'superseded_by')
    op.drop_column('memories', 'merged_from_json')
    op.drop_column('memories', 'history_json')
    op.drop_column('memories', 'canonical_content')
