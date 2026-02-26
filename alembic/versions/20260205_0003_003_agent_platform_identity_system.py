"""Agent Platform - Identity System and Message Enhancements

Revision ID: 003_agent_platform
Revises: 002_memory_enhancement
Create Date: 2026-02-05

Adds:
- identities table for agent personality/behavior documents
- Enhances messages table with token tracking fields
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003_agent_platform'
down_revision = '002_memory_enhancement'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create identities table
    op.create_table(
        'identities',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('identity_type', sa.String(50), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('priority', sa.Integer, default=0),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create composite index for user_id + identity_type
    op.create_index('ix_identities_user_type', 'identities', ['user_id', 'identity_type'])
    
    # Add new columns to messages table for token tracking
    op.add_column('messages', sa.Column('tokens_prompt', sa.Integer, nullable=True))
    op.add_column('messages', sa.Column('tokens_completion', sa.Integer, nullable=True))
    op.add_column('messages', sa.Column('model_used', sa.String(50), nullable=True))
    op.add_column('messages', sa.Column('memories_retrieved_json', sa.Text, nullable=True))
    op.add_column('messages', sa.Column('processing_time_ms', sa.Integer, nullable=True))


def downgrade() -> None:
    # Remove columns from messages
    op.drop_column('messages', 'processing_time_ms')
    op.drop_column('messages', 'memories_retrieved_json')
    op.drop_column('messages', 'model_used')
    op.drop_column('messages', 'tokens_completion')
    op.drop_column('messages', 'tokens_prompt')
    
    # Drop identities table
    op.drop_index('ix_identities_user_type', table_name='identities')
    op.drop_table('identities')
