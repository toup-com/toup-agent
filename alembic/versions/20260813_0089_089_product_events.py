"""089 — product_events: the funnel table Admin Dispatch is the first producer of.

One append-only row per product fact worth counting. Generic on purpose
(``event`` + ``entity_type``/``entity_id`` rather than ``dispatch_id``):
the questions it answers are funnel questions, and a funnel that has to
UNION one table per feature stops being asked.

It is not a duplicate of anything. ``admin_dispatch_targets`` is the
delivery LEDGER — one row per (dispatch, user), UPDATEd in place — so it
can say a notice was read but never WHEN it was delivered, nor that a
Retry press delivered it a second time. ``credit_ledger`` is an immutable
billing trail and this path spends no credits.

``dedupe_key`` + its UNIQUE index are the retry guard: the fan-out is
idempotent by construction, so a retry legitimately re-walks every
recipient, and without a key per FACT "recipients reached" would climb
with the number of times an operator pressed a button.

PLATFORM_ONLY (``app/db/models/base.py``). Gated on ``settings.run_mode``
like revisions 078-085, for the same hard-learned reason: ``alembic
upgrade head`` runs on boot in BOTH images, ``CREATE TABLE … REFERENCES
users`` takes a lock on ``users``, and taking that lock inside a tenant DB
during a blue/green rollout is what stalled the 078 canary.

Revision ID: 088
Revises: 085
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op
# Explicit, not `sa.dialects.postgresql`: `import sqlalchemy as sa` does not
# import the dialect subpackages, so the attribute form raises AttributeError
# — and only on the branch that CREATES the table, i.e. only in production.
from sqlalchemy.dialects.postgresql import JSONB

revision = "089"
# Chained onto 087 (#641), which chains onto 086 (#646). This branch was
# authored against 085 because those two had not landed; re-pointing it is
# the one-line change its own note anticipated.
#
# The note did NOT anticipate the other half, which is the fatal one: the
# sibling ringtone branch ALSO claimed `revision = "088"`. A duplicate
# revision id merges cleanly in git — no conflict, both files land — and
# then alembic raises at ENVIRONMENT LOAD, so `upgrade head` runs NOTHING.
# Not "one migration fails": nothing runs, on every boot, in both images.
# That has bitten this repo before (two parallel PRs both numbered 080).
# The ringtone branch is now 089. Before merging anything here, run
# `alembic heads` and require exactly one.
down_revision = "088"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.089")

_EVENTS = "product_events"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.089] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.089] not a platform DB; skipping (%s is PLATFORM_ONLY)", _EVENTS,
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.089] users absent; skipping")
        return

    if _EVENTS not in tables:
        op.create_table(
            _EVENTS,
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("event", sa.String(48), nullable=False),
            # SET NULL on both: a deleted account takes its identity out of
            # the series without taking the counts with it. A funnel whose
            # denominator shrinks retroactively is worse than no funnel.
            #
            # `user_id` is the SUBJECT and is nullable for a second reason
            # too: a broadcast's `dispatch_created` has no single subject.
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column(
                "actor_user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("entity_type", sa.String(32), nullable=True),
            sa.Column("entity_id", sa.String(64), nullable=True),
            sa.Column(
                "payload_json",
                sa.JSON().with_variant(JSONB(), "postgresql"),
                nullable=True,
            ),
            sa.Column("dedupe_key", sa.String(200), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            # The retry guard: the fan-out is idempotent by construction, so
            # a Retry press legitimately re-walks every recipient, and
            # without this "recipients reached" would climb with the number
            # of times an operator pressed a button. Multiple NULLs are
            # allowed on both backends, which is the intended escape for an
            # event where every occurrence really is its own fact.
            #
            # Declared INLINE, matching 085's `uq_admin_dispatch_target`: a
            # fresh platform DB gets this table from `create_all` before
            # alembic ever sees it, and the model declares the same named
            # constraint — so both build orders end up with one object of
            # one name, rather than a constraint on one path and a bare
            # index on the other.
            sa.UniqueConstraint("dedupe_key", name="uq_product_events_dedupe"),
        )
    else:
        logger.info("[alembic.089] %s already present", _EVENTS)

    # Unconditional with IF NOT EXISTS: on a fresh platform DB `create_all`
    # builds the table before alembic ever sees it (085 records the same
    # ordering), so the table-existence guard above must not gate the
    # indexes — and re-running these is free.
    #
    # The reason the table exists: one event's rate over a window
    # ("dispatch_read last 7 days"). Every funnel query is this shape.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_product_events_event_created "
        f"ON {_EVENTS} (event, created_at)"
    )
    # The per-dispatch timeline: everything that happened to THIS message.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_product_events_entity "
        f"ON {_EVENTS} (entity_type, entity_id)"
    )
    # Support's question the other way round: what has this account been
    # sent, and what did they do with it.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_product_events_user_created "
        f"ON {_EVENTS} (user_id, created_at)"
    )
    # Deliberately NO bare index on created_at: every INSERT pays for every
    # index and this table takes one row per RECIPIENT of a broadcast. The
    # only reader that would want it is age-based pruning, an occasional ops
    # action that can afford a scan.


def downgrade() -> None:
    # No-op, same reasoning as revisions 078-085: a blue-green rollout runs
    # the OLD and NEW images against the same DB for the drain window, and
    # the migration-lint guard rejects destructive patterns anywhere in the
    # diff — including ones that "only" sit in downgrade(), which is exactly
    # the kind that gets run by accident during an incident.
    logger.info(
        "[alembic.089] downgrade is a no-op; %s is left in place (see comment)",
        _EVENTS,
    )
