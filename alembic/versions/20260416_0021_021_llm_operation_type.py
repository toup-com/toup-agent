"""LLM operation_type + DayChat archival summary columns

Revision ID: 021
Revises: 020
Create Date: 2026-04-16

1. Adds operation_type to llm_proxy_events so platform-side system operations
   (e.g. end-of-day archival summaries) can be logged for cost tracking without
   counting toward user budget caps. NULL or "user.*" counts toward caps;
   "system.*" is exempt.
2. Adds archival_summary + archival_summary_generated_at + archival_summary_status
   to day_chats for the end-of-day archival pass. Distinct from rolling_summary.
"""
from alembic import op
import sqlalchemy as sa


revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "llm_proxy_events",
        sa.Column("operation_type", sa.String(50), nullable=True),
    )
    op.create_index(
        "ix_llm_proxy_operation_type",
        "llm_proxy_events",
        ["operation_type"],
    )

    op.add_column(
        "day_chats",
        sa.Column("archival_summary", sa.Text, nullable=True),
    )
    op.add_column(
        "day_chats",
        sa.Column("archival_summary_generated_at", sa.DateTime, nullable=True),
    )
    op.add_column(
        "day_chats",
        sa.Column(
            "archival_summary_status",
            sa.String(20),
            nullable=False,
            server_default="not_needed",
        ),
    )


def downgrade():
    op.drop_column("day_chats", "archival_summary_status")
    op.drop_column("day_chats", "archival_summary_generated_at")
    op.drop_column("day_chats", "archival_summary")
    op.drop_index("ix_llm_proxy_operation_type", table_name="llm_proxy_events")
    op.drop_column("llm_proxy_events", "operation_type")
