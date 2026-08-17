"""090 — admin_dispatches.tone: the notification sound an operator chose.

One nullable String(32) holding a catalogue id from
``app/services/dispatch_tones.py`` (``ping``, ``chime``, … or
``default``). NULL is the pre-090 history and reads as the system
default — the read path (`dispatch_tones.normalize`) collapses anything
unknown, so no backfill is needed and none is done.

PLATFORM_ONLY, gated on ``settings.run_mode`` like 078-085: ``alembic
upgrade head`` runs on boot in BOTH images, and ``admin_dispatches``
does not exist in a tenant DB at all.

⚠️ ORDERING WITH THE SIBLING ARC BRANCHES. This revision hangs off 085
because that is the last revision on ``main`` at the time of writing.
086 (``admin_dispatch_origin``) and 087 (``admin_dispatch_last_error``)
exist on other in-flight branches of this same arc, and every one of
those also hangs off its own predecessor. Whichever of them merges
AFTER this one must re-point its ``down_revision`` (or this one must be
re-pointed at 087) — two revisions sharing a parent is two heads, and
two heads makes ``alembic upgrade head`` fail at ENVIRONMENT LOAD, i.e.
NO migration runs at all, on both images, on the next deploy.
``tests/test_migration_088_dispatch_tone.py::test_exactly_one_head``
fails the moment that happens, which is the only reason this is safe to
land before its siblings.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "090"
# Chain: 085 → 086 (#646) → 087 (#641) → 088 (#651 product_events) → 089.
#
# Authored as 088-off-085 while those siblings were in flight, which collided
# head-on: #651 had ALSO taken 088. A duplicate revision id merges cleanly in
# git — different filenames, no conflict — and then alembic raises at
# ENVIRONMENT LOAD, so `upgrade head` runs NOTHING, on every boot, in both
# images. This repo has hit that before with two parallel PRs numbered 080.
#
# 088 is not in this tree yet, so `test_exactly_one_head` fails here until
# #651 lands and this branch is rebased. That failure is CORRECT. Do not
# "fix" it by re-pointing at 085: two revisions sharing a parent is two
# heads, which is the same environment-load failure by another route.
down_revision = "089"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.090")

_TABLE = "admin_dispatches"
_COLUMN = "tone"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.090] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.090] not a platform DB; skipping (%s is PLATFORM_ONLY)", _TABLE)
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        logger.info("[alembic.090] %s absent; skipping", _TABLE)
        return

    # Inspector-guarded rather than `ADD COLUMN IF NOT EXISTS`: sqlite
    # rejects that form, and `create_all` builds this table on a fresh
    # platform DB before alembic ever sees it — so "already present" is
    # an ordinary state, not an error. (085 does the same for
    # `admin_thread_messages.sender_name`.)
    if _COLUMN in {c["name"] for c in insp.get_columns(_TABLE)}:
        logger.info("[alembic.090] %s.%s already present", _TABLE, _COLUMN)
        return

    op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(32), nullable=True))


def downgrade() -> None:
    # No-op, same reasoning as 078-085: a blue/green rollout runs the OLD
    # and NEW images against one DB for the drain window, so removing a
    # column the new image writes would break the old one mid-drain — and
    # the migration-lint guard rejects destructive patterns anywhere in
    # the diff, including inside a downgrade nobody meant to run.
    logger.info("[alembic.090] downgrade is a no-op; %s.%s is left in place", _TABLE, _COLUMN)
