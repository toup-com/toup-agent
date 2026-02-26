"""Initial schema - baseline for existing HexBrain tables

Revision ID: 001_initial
Revises: 
Create Date: 2026-02-04

This migration establishes the baseline schema for the HexBrain memory system.
It creates all the existing tables that were previously created via init_db.py.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, index=True, nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('is_active', sa.Boolean, default=True),
    )

    # Conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), index=True, nullable=False),
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('started_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('ended_at', sa.DateTime, nullable=True),
        sa.Column('message_count', sa.Integer, default=0),
        sa.Column('metadata_json', sa.Text, nullable=True),
    )

    # Messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id'), index=True, nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('embedding_json', sa.Text, nullable=True),
    )

    # Memories table (core)
    op.create_table(
        'memories',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), index=True, nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('summary', sa.String(500), nullable=True),
        sa.Column('category', sa.String(20), index=True, nullable=False),
        sa.Column('memory_type', sa.String(20), index=True, nullable=False),
        sa.Column('embedding_json', sa.Text, nullable=True),
        sa.Column('importance', sa.Float, default=0.5),
        sa.Column('confidence', sa.Float, default=1.0),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), index=True),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('last_accessed_at', sa.DateTime, nullable=True),
        sa.Column('access_count', sa.Integer, default=0),
        sa.Column('source_message_id', sa.String(36), sa.ForeignKey('messages.id'), nullable=True),
        sa.Column('source_type', sa.String(50), default='conversation'),
        sa.Column('metadata_json', sa.Text, nullable=True),
        sa.Column('tags_json', sa.Text, nullable=True),
        sa.Column('is_deleted', sa.Boolean, default=False, index=True),
        sa.Column('deleted_at', sa.DateTime, nullable=True),
    )

    # Create composite indexes for memories
    op.create_index('ix_memories_user_category', 'memories', ['user_id', 'category'])
    op.create_index('ix_memories_user_type', 'memories', ['user_id', 'memory_type'])
    op.create_index('ix_memories_user_created', 'memories', ['user_id', 'created_at'])

    # Memory relationships (many-to-many self-referential)
    op.create_table(
        'memory_relationships',
        sa.Column('source_id', sa.String(36), sa.ForeignKey('memories.id'), primary_key=True),
        sa.Column('target_id', sa.String(36), sa.ForeignKey('memories.id'), primary_key=True),
        sa.Column('relationship_type', sa.String(50), nullable=True),
        sa.Column('strength', sa.Float, default=1.0),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    # Entities table
    op.create_table(
        'entities',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), index=True, nullable=False),
        sa.Column('name', sa.String(255), index=True, nullable=False),
        sa.Column('entity_type', sa.String(50), index=True, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('embedding_json', sa.Text, nullable=True),
        sa.Column('attributes_json', sa.Text, nullable=True),
        sa.Column('mention_count', sa.Integer, default=1),
        sa.Column('first_seen_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('last_seen_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create composite indexes for entities
    op.create_index('ix_entities_user_name', 'entities', ['user_id', 'name'])
    op.create_index('ix_entities_user_type', 'entities', ['user_id', 'entity_type'])

    # Entity links table
    op.create_table(
        'entity_links',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('memory_id', sa.String(36), sa.ForeignKey('memories.id'), index=True, nullable=False),
        sa.Column('entity_id', sa.String(36), sa.ForeignKey('entities.id'), index=True, nullable=False),
        sa.Column('role', sa.String(50), default='mentioned'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    # Brain stats table (cached visualization data)
    op.create_table(
        'brain_stats',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), unique=True, index=True, nullable=False),
        sa.Column('region_counts_json', sa.Text, default='{}'),
        sa.Column('region_sizes_json', sa.Text, default='{}'),
        sa.Column('total_memories', sa.Integer, default=0),
        sa.Column('total_entities', sa.Integer, default=0),
        sa.Column('total_connections', sa.Integer, default=0),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('brain_stats')
    op.drop_table('entity_links')
    op.drop_index('ix_entities_user_type', table_name='entities')
    op.drop_index('ix_entities_user_name', table_name='entities')
    op.drop_table('entities')
    op.drop_table('memory_relationships')
    op.drop_index('ix_memories_user_created', table_name='memories')
    op.drop_index('ix_memories_user_type', table_name='memories')
    op.drop_index('ix_memories_user_category', table_name='memories')
    op.drop_table('memories')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('users')
