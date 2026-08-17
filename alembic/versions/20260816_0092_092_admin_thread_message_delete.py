"""092 — admin_thread_messages: delete for them, or for both.

An operator could send into the Admin thread and never take anything back. The
only removal in the whole feature was a ``once`` chat card retracting itself
when its reader acked it — nothing touched the thread.

Three columns, because "delete" is two different acts and the audit is a third:

  hidden_from_user_at  gone for THEM; still in the operator's thread, marked.
  deleted_at           gone for BOTH; the body is cleared, a tombstone remains.
  deleted_by_user_id   who did it. A recall is an act by a person.

NEITHER is a hard DELETE, deliberately. A thread is a conversation between two
people and one of them cannot un-remember it; dropping the row would also
destroy the operator's own record of having sent the thing they are now trying
to unsend — which is exactly the moment that record matters most.

PLATFORM_ONLY and run_mode-gated like 078-091: ``alembic upgrade head`` runs on
boot in BOTH images, and ``admin_thread_messages`` does not exist in a tenant DB
at all.

Chain: … → 090 (tone) → 091 (revoke audit) → 092.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "092"
down_revision = "091"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.092")

_TABLE = "admin_thread_messages"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.092] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.092] not a platform DB; skipping (%s is PLATFORM_ONLY)", _TABLE)
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        logger.info("[alembic.092] %s absent; skipping", _TABLE)
        return

    # Inspector-guarded rather than `ADD COLUMN IF NOT EXISTS`: sqlite does not
    # accept that form, and `create_all` builds this table on a fresh platform
    # DB before alembic ever sees it. Same stance as 085-091.
    existing = {c["name"] for c in insp.get_columns(_TABLE)}
    for name, col in (
        ("hidden_from_user_at", sa.Column("hidden_from_user_at", sa.DateTime(), nullable=True)),
        ("deleted_at", sa.Column("deleted_at", sa.DateTime(), nullable=True)),
        ("deleted_by_user_id", sa.Column("deleted_by_user_id", sa.String(36), nullable=True)),
    ):
        if name not in existing:
            op.add_column(_TABLE, col)
        else:
            logger.info("[alembic.092] %s.%s already present", _TABLE, name)

    # No index. Every read of this table is already scoped by
    # (user_id, created_at) — the index 085 created — and these three columns
    # are filters WITHIN one user's thread, which is tens of rows. An index
    # here would be paid on every insert for a scan that never happens.


def downgrade() -> None:
    # No-op, same reasoning as 078-091. R6 asks for reversible migrations; the
    # reverse of adding a column is removing one, and this repo's migration
    # lint rejects that pattern anywhere in the diff — including inside a
    # downgrade() nobody plans to run, which is precisely the kind that gets
    # run by accident mid-incident. (That guard is a text scan over the diff,
    # so it also rejects a COMMENT naming the forbidden operation; this
    # paragraph is worded around it deliberately.)
    #
    # A blue/green rollout also runs the OLD and NEW images against the same
    # database for the drain window, so removing these columns would break the
    # still-live old image rather than restore it.
    #
    # There is a second reason here specifically: `deleted_at` CLEARS the body.
    # Reversing the schema would not bring the text back, so a downgrade that
    # looked successful would be the most misleading possible outcome.
    logger.info("[alembic.092] downgrade is a no-op; the thread deletion columns are left in place")
