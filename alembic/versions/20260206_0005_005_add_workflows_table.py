"""005 — Add workflows table for /work page

Revision ID: 005_add_workflows_table
Revises: 004_memory_evolution_system
Create Date: 2026-02-06
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '005_add_workflows_table'
down_revision = '004_memory_evolution'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'workflows',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), server_default='draft'),
        sa.Column('nodes_json', sa.Text(), server_default='[]'),
        sa.Column('edges_json', sa.Text(), server_default='[]'),
        sa.Column('run_count', sa.Integer(), server_default='0'),
        sa.Column('last_run_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('workflows')
