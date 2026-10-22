"""108 — durable derived chat titles and short generation leases.

Platform only. No message text, no messages/day_chats dependency, no tenant
ALTER mirror. Existing users are not rewritten. Old replicas ignore the new
table, so this additive schema can land before the reader/writer rollout.
"""

from alembic import op
import sqlalchemy as sa

revision = "108"
down_revision = "107"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Tenant DBs do not run this platform chain. Keep a direct invocation from
    # creating a platform cache there as well (same guard as platform-only DDL).
    from app.config import settings
    if settings.run_mode == "agent":
        return
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("chat_topic_titles"):
        # init_db/create_all or a narrowly scoped additive rollout can precede
        # Alembic on a platform whose historical revision marker lags. Adopt
        # only the expected schema; never silently stamp an unrelated table.
        columns = {column["name"]: column for column in inspector.get_columns("chat_topic_titles")}
        expected = {"user_id", "day", "canonical_id", "fingerprint", "version", "topic",
                    "lease_token", "lease_expires_at", "created_at", "updated_at", "expires_at"}
        lengths = {"user_id": 36, "canonical_id": 50, "fingerprint": 64, "topic": 160, "lease_token": 64}
        foreign_keys = inspector.get_foreign_keys("chat_topic_titles")
        if (set(columns) != expected
                or any(getattr(columns[name]["type"], "length", None) != length for name, length in lengths.items())
                or inspector.get_pk_constraint("chat_topic_titles")["constrained_columns"] != ["user_id", "day", "canonical_id"]
                or not any(fk["constrained_columns"] == ["user_id"] and fk["referred_table"] == "users"
                           and fk.get("options", {}).get("ondelete") == "CASCADE" for fk in foreign_keys)
                or not any(index["name"] == "ix_chat_topic_titles_expires_at" and index["column_names"] == ["expires_at"]
                           for index in inspector.get_indexes("chat_topic_titles"))):
            raise RuntimeError("chat_topic_titles does not match migration 108")
        return
    op.create_table(
        "chat_topic_titles",
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("canonical_id", sa.String(50), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("topic", sa.String(160), nullable=True),
        sa.Column("lease_token", sa.String(64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("version > 0", name="ck_chat_topic_titles_version"),
    )
    op.create_index("ix_chat_topic_titles_expires_at", "chat_topic_titles", ["expires_at"])


def downgrade() -> None:
    from app.config import settings
    if settings.run_mode == "agent":
        return
    op.drop_table("chat_topic_titles")
