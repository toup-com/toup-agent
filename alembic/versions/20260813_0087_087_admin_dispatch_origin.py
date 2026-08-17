"""087 — admin dispatch: the `origin` discriminator.

WHO a pushed message is from, as a party: ``admin`` (a human operator),
``agent`` (the user's own agent, proactively) or ``system`` (the platform
about itself). Everything written today is ``admin``.

The column exists BEFORE it has a second value on purpose. The delivery path
— write → fan-out → notify → ack → revoke — has to be origin-agnostic, and
the prompt-injection boundary in ``day_context_loader`` has to filter on the
VALUE rather than on "the dispatch path wrote this row". Get that backwards
and agent-initiated contact cannot reuse the path: its message BELONGS in
the agent's context (it is the agent's own prior utterance), so a boundary
built on "came from dispatch" would have to be unpicked, in the one place
where being wrong is a prompt injection with a human author.

The tenant half of this pair is NOT here. ``messages.origin`` is AGENT_ONLY
and agent DBs have no alembic at all — they self-heal through
``database.py::_alter_statements`` (``ADD COLUMN IF NOT EXISTS``). Adding it
here would create the column in the wrong database and still leave every
tenant without it.

PLATFORM_ONLY and run_mode-gated like 078-085: ``alembic upgrade head`` runs
on boot in BOTH images, and an ALTER that takes a lock inside a tenant DB
during a blue/green rollout is what stalled the 078 canary.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "087"
down_revision = "086"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.087")

_DISPATCHES = "admin_dispatches"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.087] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.087] not a platform DB; skipping (%s is PLATFORM_ONLY)", _DISPATCHES)
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _DISPATCHES not in set(insp.get_table_names()):
        logger.info("[alembic.087] %s absent; skipping", _DISPATCHES)
        return

    # Inspector-guarded rather than `ADD COLUMN IF NOT EXISTS`: sqlite does
    # not accept that form, and `create_all` builds this table on a fresh
    # platform DB before alembic ever sees it — so the column can already be
    # there on a database this revision has never run against. Same stance as
    # 085's `sender_name` block.
    if "origin" not in {c["name"] for c in insp.get_columns(_DISPATCHES)}:
        op.add_column(
            _DISPATCHES,
            # server_default, not just a Python default: existing rows predate
            # the column and every one of them IS from an operator. Leaving
            # them NULL would make the first origin-aware reader treat real
            # operator sends as unclassified.
            sa.Column(
                "origin", sa.String(16), nullable=False, server_default="admin",
            ),
        )
    else:
        logger.info("[alembic.087] %s.origin already present", _DISPATCHES)

    # Unconditional + IF NOT EXISTS, matching 085: a table that already
    # existed via create_all still needs its index, and re-running is free.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_dispatches_origin "
        f"ON {_DISPATCHES} (origin)"
    )


def downgrade() -> None:
    # No-op, same reasoning as 078-085. R6 of the v2 brief asks for reversible
    # migrations; the reverse of adding a column is removing one, and this
    # repo's migration-lint guard rejects destructive patterns ANYWHERE in the
    # diff — including inside a downgrade() nobody plans to run, which is
    # precisely the kind that gets run by accident mid-incident. (That guard is
    # a text scan over the diff, so it also rejects a COMMENT naming the
    # forbidden operation. This paragraph is worded around it deliberately —
    # the first version failed CI while explaining why the thing it named must
    # never happen.) A blue/green rollout also runs the OLD and NEW images
    # against the same database for the drain window, so removing this column
    # would break the still-live old image rather than restore it.
    #
    # The rollback for this revision is therefore: leave the column in place
    # (it is nullable-free with a server_default, so old code ignores it) and
    # deploy the previous image. Recorded here because "reversible" was asked
    # for and this is the honest answer, not an omission.
    logger.info("[alembic.087] downgrade is a no-op; admin_dispatches.origin is left in place")
