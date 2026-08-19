"""094 — admin_thread_messages: a screenshot report is a message in the thread.

A user filing a report from the app (POST /support/issues — a note, a severity,
the screen they were on, and usually a screenshot) reached the operator by
EMAIL only. The Admin thread — the one operator↔user conversation the product
has, with a reply box on both ends — never heard about it, so the operator's
answer had nowhere conversational to go.

Three nullable columns, all display-side, on the row a report now opens:

  kind         NULL for text (every existing row); 'report' for a report row.
               A column, not a JSON flag: the Conversations list badges the
               loudest UNANSWERED report per user in SQL over the whole table.
  severity     the reporter's own rating (low|medium|high|critical, the
               support enum's values). Same reason.
  report_json  {support_issue_id, channel, context:{screen, app_version,
               build, platform, device, os, raw}}. Nothing filters on it.

The screenshot itself is an `admin_thread_attachments` row (093) hanging off
the report row — no second picture table. The row's id is a uuid5 of the
support issue id, so the screenshot upload (a separate request from the app)
finds its message without a lookup column, and a replayed intake collides on
the primary key instead of filing the same report twice.

No thread entity is added (D3, Unit 1): a second report from the same user is
a second report CARD in that user's one conversation, never a second
conversation. This migration is what "no persistence-schema rewrites beyond
what the report thread needs" comes to.

PLATFORM_ONLY and run_mode-gated like 078-093: ``alembic upgrade head`` runs on
boot in BOTH images, and ``admin_thread_messages`` does not exist in a tenant DB
at all.

Chain: … → 092 (thread deletion) → 093 (thread attachments) → 094.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "094"
down_revision = "093"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.094")

_TABLE = "admin_thread_messages"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.094] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.094] not a platform DB; skipping (%s is PLATFORM_ONLY)", _TABLE)
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        logger.info("[alembic.094] %s absent; skipping", _TABLE)
        return

    # Inspector-guarded rather than `ADD COLUMN IF NOT EXISTS`: sqlite does not
    # accept that form, and `create_all` builds this table on a fresh platform
    # DB before alembic ever sees it. Same stance as 085-093.
    existing = {c["name"] for c in insp.get_columns(_TABLE)}
    for name, col in (
        ("kind", sa.Column("kind", sa.String(16), nullable=True)),
        ("severity", sa.Column("severity", sa.String(16), nullable=True)),
        ("report_json", sa.Column("report_json", sa.JSON(), nullable=True)),
    ):
        if name not in existing:
            op.add_column(_TABLE, col)
        else:
            logger.info("[alembic.094] %s.%s already present", _TABLE, name)

    # No index. The list's severity pass runs over the PAGE of users it has
    # already aggregated (≤ 500 user_ids, IN-list on the indexed user_id), and
    # `kind` is a filter within those rows. An index on `kind` alone would be
    # paid on every insert for a scan nothing performs.


def downgrade() -> None:
    # No-op, same stance as 078-093: a blue/green rollout runs the OLD and NEW
    # images against one database for the drain window, and dropping the
    # columns would break the still-live old image rather than restore it. The
    # columns are nullable and unread by the old image, so leaving them is free.
    logger.info("[alembic.094] downgrade is a no-op; the report columns are left in place")
