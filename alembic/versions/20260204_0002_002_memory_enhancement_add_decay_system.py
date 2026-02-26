"""Add memory decay and consolidation system

Revision ID: 002_memory_enhancement
Revises: 001_initial
Create Date: 2026-02-04

This migration adds the "10-Year Memory Architecture" features:
- Memory strength/decay fields
- Memory level (episodic/semantic/procedural/meta)
- Emotional salience
- MemoryEvent audit log table
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002_memory_enhancement'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to memories table for decay/reinforcement system
    # All columns have defaults for backward compatibility with existing data
    
    op.add_column('memories', sa.Column(
        'strength', sa.Float, server_default='1.0', nullable=False,
        comment='Memory strength (0-1), decays over time based on Ebbinghaus curve'
    ))
    
    op.add_column('memories', sa.Column(
        'memory_level', sa.String(20), server_default='episodic', nullable=False,
        comment='Cognitive level: episodic, semantic, procedural, meta'
    ))
    
    op.add_column('memories', sa.Column(
        'emotional_salience', sa.Float, server_default='0.5', nullable=False,
        comment='Emotional weight (0-1), affects decay resistance'
    ))
    
    op.add_column('memories', sa.Column(
        'last_reinforced_at', sa.DateTime, nullable=True,
        comment='Last time memory was strengthened via access/recall'
    ))
    
    op.add_column('memories', sa.Column(
        'consolidation_count', sa.Integer, server_default='0', nullable=False,
        comment='Number of times this memory has been consolidated'
    ))
    
    op.add_column('memories', sa.Column(
        'decay_rate', sa.Float, server_default='0.1', nullable=False,
        comment='Individual decay rate modifier (higher = faster decay)'
    ))
    
    # Create index for memory level queries
    op.create_index('ix_memories_user_level', 'memories', ['user_id', 'memory_level'])
    
    # Create index for strength-based queries (find weak memories for cleanup)
    op.create_index('ix_memories_strength', 'memories', ['strength'])
    
    # Create the memory_events table for immutable audit logging
    op.create_table(
        'memory_events',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('memory_id', sa.String(36), sa.ForeignKey('memories.id'), index=True, nullable=False),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), index=True, nullable=False),
        sa.Column('event_type', sa.String(20), index=True, nullable=False,
                  comment='Event type: created, accessed, reinforced, decayed, consolidated, updated, deleted, linked, unlinked'),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now(), index=True, nullable=False),
        sa.Column('event_data_json', sa.Text, nullable=True,
                  comment='JSON with event-specific data (old/new values, context, etc.)'),
        sa.Column('trigger_source', sa.String(50), nullable=True,
                  comment='What triggered this event: api, scheduled, consolidation, etc.'),
    )
    
    # Create composite indexes for efficient event queries
    op.create_index('ix_memory_events_memory_time', 'memory_events', ['memory_id', 'timestamp'])
    op.create_index('ix_memory_events_user_time', 'memory_events', ['user_id', 'timestamp'])
    op.create_index('ix_memory_events_type_time', 'memory_events', ['event_type', 'timestamp'])


def downgrade() -> None:
    # Drop memory_events table and its indexes
    op.drop_index('ix_memory_events_type_time', table_name='memory_events')
    op.drop_index('ix_memory_events_user_time', table_name='memory_events')
    op.drop_index('ix_memory_events_memory_time', table_name='memory_events')
    op.drop_table('memory_events')
    
    # Drop new indexes on memories
    op.drop_index('ix_memories_strength', table_name='memories')
    op.drop_index('ix_memories_user_level', table_name='memories')
    
    # Remove new columns from memories
    op.drop_column('memories', 'decay_rate')
    op.drop_column('memories', 'consolidation_count')
    op.drop_column('memories', 'last_reinforced_at')
    op.drop_column('memories', 'emotional_salience')
    op.drop_column('memories', 'memory_level')
    op.drop_column('memories', 'strength')
