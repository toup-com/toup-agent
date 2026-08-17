"""091 — admin_dispatches.revoked_at / revoked_by_user_id: the recall audit.

Until now the only thing that could remove an operator's card from a user's
chat was the user READING a ``once`` notice. There was no operator recall at
all, and the compose form said so — "Cannot be recalled" — which was true of
the notification and false of the card.

Two columns, because a recall is an ACT BY A PERSON and the interesting
question afterwards is always "who, and when". A boolean would answer neither.
``revoked_by_user_id`` is deliberately NOT a foreign key with CASCADE: an
operator's account being deleted must not erase the record that they recalled a
broadcast. It is a plain string for the same reason 085's
``created_by_user_id`` is.

PLATFORM_ONLY and run_mode-gated like 078-089: ``alembic upgrade head`` runs on
boot in BOTH images, and ``admin_dispatches`` does not exist in a tenant DB.

Chain: 085 → 086 (#648 first_media_played_at) → 087 (origin) → 088
(last_error) → 089 (product_events) → 090 (tone) → 091. Every one of those is an open PR of the same arc, so this
revision cannot merge before them — and `.github/workflows/migration-lint.yml`
now fails the build if that ordering is broken, rather than letting a duplicate
id merge cleanly and silently stop every migration.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "091"
down_revision = "090"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.091")

_TABLE = "admin_dispatches"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.091] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.091] not a platform DB; skipping (%s is PLATFORM_ONLY)", _TABLE)
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        logger.info("[alembic.091] %s absent; skipping", _TABLE)
        return

    # Inspector-guarded rather than `ADD COLUMN IF NOT EXISTS`: sqlite does not
    # accept that form, and `create_all` builds this table on a fresh platform
    # DB before alembic ever sees it — so a column can already be there on a
    # database this revision has never run against. Same stance as 085-089.
    existing = {c["name"] for c in insp.get_columns(_TABLE)}

    for name, col in (
        ("revoked_at", sa.Column("revoked_at", sa.DateTime(), nullable=True)),
        ("revoked_by_user_id", sa.Column("revoked_by_user_id", sa.String(36), nullable=True)),
    ):
        if name not in existing:
            op.add_column(_TABLE, col)
        else:
            logger.info("[alembic.091] %s.%s already present", _TABLE, name)

    # Partial index: a revoked dispatch is rare and the only query is "show me
    # the recalls", so indexing the NULLs would be most of the table for none
    # of the reads. Unconditional + IF NOT EXISTS, matching 085-089.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_dispatches_revoked_at "
        f"ON {_TABLE} (revoked_at) WHERE revoked_at IS NOT NULL"
    )


def downgrade() -> None:
    # No-op, same reasoning as 078-089. R6 asks for reversible migrations; the
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
    # The rollback is therefore: leave the columns in place (both nullable, so
    # old code ignores them) and deploy the previous image.
    logger.info("[alembic.091] downgrade is a no-op; the revoke audit columns are left in place")
