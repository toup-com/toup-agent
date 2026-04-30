"""Add app_builder_planner_model + app_builder_builder_model + null
agent_model server_default + backfill gpt-5.2 -> NULL.

Phase 3 commit E of the LLM model-selection refactor (see
docs/llm-model-refactor-plan.md §5).

Three changes, all additive at the row level (no data loss):

1. New nullable columns `app_builder_planner_model` and
   `app_builder_builder_model` on `agent_configs`. NULL means "fall
   through to agent_model" — this is the documented behaviour, not a
   missing-data state. Resolver code (`app.services.model_resolver`)
   already reads these via getattr with a None default, so the gap
   between this migration and commit F is safe.

2. `agent_configs.agent_model` server_default flipped from `'gpt-5.2'`
   (set in migration 015) to NULL. The old default predated the
   resolver layer; new rows now correctly inherit the platform default
   from `settings.agent_model` at use-time.

3. Backfill: rows still holding the literal `'gpt-5.2'` mean the user
   never customised — that value came from the stale server_default,
   not from a deliberate choice. Setting them to NULL routes them
   through the resolver instead. Rows with any other value (including
   `claude-opus-4-6`, customised choices, etc.) are untouched.

Revision ID: 029
Revises: 028
Create Date: 2026-04-28
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade():
    # 1. New auto-builder split columns (nullable; resolver inherits from agent_model on null).
    op.add_column(
        "agent_configs",
        sa.Column("app_builder_planner_model", sa.String(50), nullable=True),
    )
    op.add_column(
        "agent_configs",
        sa.Column("app_builder_builder_model", sa.String(50), nullable=True),
    )

    # 2. Drop the stale server_default on agent_model. New rows now follow
    #    the platform default via the resolver instead of being baked with
    #    a specific model id.
    op.alter_column(
        "agent_configs",
        "agent_model",
        server_default=None,
        existing_type=sa.String(50),
    )

    # 3. Backfill rows still holding the old default. NULL routes through
    #    the resolver — what every other value already does at runtime.
    op.execute(
        "UPDATE agent_configs SET agent_model = NULL WHERE agent_model = 'gpt-5.2'"
    )


def downgrade():
    # Restoring the old server_default is reversible. The backfill is not
    # — those NULL rows can't be distinguished from rows that were always
    # NULL, so we leave them alone on downgrade. (At time of writing, the
    # affected row count was small enough to enumerate manually if rollback
    # ever needed perfect parity.)
    op.alter_column(
        "agent_configs",
        "agent_model",
        server_default="gpt-5.2",
        existing_type=sa.String(50),
    )
    op.drop_column("agent_configs", "app_builder_builder_model")
    op.drop_column("agent_configs", "app_builder_planner_model")
