"""011 — Retrieval feedback events table for self-improvement loop.

Tracks every retrieval event: which memories were retrieved, which were
used in the response, which were irrelevant, and when extraction gaps
are detected. Powers the feedback loop that improves extraction over time.

Revision ID: 011_retrieval_feedback
Revises: 010_schema_extraction
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "011_retrieval_feedback"
down_revision = "010_schema_extraction"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrieval_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), nullable=True),
        # The query that triggered retrieval
        sa.Column("query", sa.Text, nullable=False),
        # Which strategies were used (JSON array: ["vector", "keyword", "graph"])
        sa.Column("strategies_used_json", sa.Text, nullable=True),
        # IDs of memories retrieved (JSON array)
        sa.Column("retrieved_memory_ids_json", sa.Text, nullable=False),
        # IDs of memories actually referenced/used in the response (JSON array)
        sa.Column("used_memory_ids_json", sa.Text, nullable=True),
        # IDs of memories that were irrelevant (JSON array)
        sa.Column("irrelevant_memory_ids_json", sa.Text, nullable=True),
        # Number of results returned
        sa.Column("result_count", sa.Integer, nullable=False, server_default="0"),
        # Was there an extraction gap? (response contained info not in any retrieved memory)
        sa.Column("has_extraction_gap", sa.Boolean, nullable=False, server_default="false"),
        # Gap topics detected (JSON array of topic strings)
        sa.Column("gap_topics_json", sa.Text, nullable=True),
        # Was the user correcting the agent?
        sa.Column("is_correction", sa.Boolean, nullable=False, server_default="false"),
        # Correction details (JSON: {"wrong_fact": "...", "correct_fact": "..."})
        sa.Column("correction_data_json", sa.Text, nullable=True),
        # Overall quality signal: "good", "partial", "miss", "empty"
        sa.Column("quality_signal", sa.String(20), nullable=True),
        # Processing time for retrieval (ms)
        sa.Column("retrieval_time_ms", sa.Integer, nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
    )

    # Indexes for analysis queries
    op.create_index("ix_retrieval_events_user_created", "retrieval_events", ["user_id", "created_at"])
    op.create_index("ix_retrieval_events_quality", "retrieval_events", ["user_id", "quality_signal"])
    op.create_index("ix_retrieval_events_gaps", "retrieval_events", ["user_id", "has_extraction_gap"])


def downgrade() -> None:
    op.drop_index("ix_retrieval_events_gaps")
    op.drop_index("ix_retrieval_events_quality")
    op.drop_index("ix_retrieval_events_user_created")
    op.drop_table("retrieval_events")
