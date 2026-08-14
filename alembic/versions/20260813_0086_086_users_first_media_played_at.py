"""086: users.first_media_played_at — the instant this account first HEARD media

The "Media" nav entry is progressively disclosed: hidden until the user has
actually played a song or a music video once, then permanent. That latch has to
be server-side, because Toup is multi-channel — a play started from the phone,
from voice, or (next) from the web must unlock the entry everywhere — and
because a device-local flag is lost with the device.

NULL = never played, which is the correct value for every pre-existing row and
the safe default for the UI (hidden). Purely ADDITIVE — no column is dropped,
renamed or retyped. Naive UTC `TIMESTAMP`, matching every other timestamp on
this table (`email_verified_at`, `password_changed_at`); there is exactly one
`DateTime(timezone=True)` in this entire codebase and it is not on `users`.

`users` is a SHARED table (app/db/models/base.py), so it exists in the platform
DB *and* in every tenant container DB. Tenant DBs have no alembic_version row —
their only migrator is `app/db/database.py::init_db`'s `_alter_statements` list,
which carries the mirror of this ALTER. Without that mirror every agent rolled
to an image whose `User` model declares this column 500s on every query that
loads a user row. This file keeps the platform DB in step; the mirror is what
reaches tenants. Authority for the column is declared "platform" in
`SHARED_COLUMN_AUTHORITY` — the tenant copy is created and never written.

── The backfill ──────────────────────────────────────────────────────────────
An existing user must not lose a page they already use, so this migration
stamps everyone whose playback is visible from HERE. That is a narrow set, and
the narrowness is a fact about where the data lives rather than an omission:

  * `messages` and `conversations` are AGENT_ONLY — every chat since the
    monolith was partitioned lives in a per-user container DB that a platform
    migration cannot reach. They nevertheless still EXIST in the platform DB as
    monolith-era leftovers (init_db stops new ones being created; it never
    drops the old, and the Supabase → Railway dump/restore carried them over),
    holding pre-partition history. Measured against prod on 2026-08-13: 948
    legacy messages, of which 1 carries a media card, mapping to 1 live user.
  * `media_playlists` has a platform mirror (migration 079) but it is empty by
    construction — the platform proxies those writes to the tenant DB. Measured
    against prod: 0 rows. It is deliberately NOT part of the query below; a
    join that can only ever contribute zero rows is a claim of coverage that
    does not exist.

Everything not visible here re-latches on the user's next play, which is
instant and permanent (the client stamps optimistically and reports). The
tenant-side sweep for accounts that played before this ships is
`scripts/backfill_first_media_played.py`, run once by an operator after deploy.

The `to_regclass` guards make the backfill a no-op wherever those leftover
tables are absent (tenant DBs, a clean platform DB, CI's sqlite is skipped
outright by the dialect check) rather than an error.

Revision ID: 086
Revises: 085
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "086"
down_revision = "085"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.086")

_TABLE = "users"
_COLUMN = "first_media_played_at"


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        logger.info("[alembic.086] %s absent; nothing to do", _TABLE)
        return

    cols = {c["name"] for c in insp.get_columns(_TABLE)}
    if _COLUMN in cols:
        logger.info("[alembic.086] %s.%s already present", _TABLE, _COLUMN)
    else:
        # Nullable, no default: Postgres rewrites no rows and holds ACCESS
        # EXCLUSIVE only long enough to update the catalog. A DEFAULT here
        # would assert that every existing account has played something.
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.DateTime(), nullable=True))
        logger.info("[alembic.086] added %s.%s", _TABLE, _COLUMN)

    if conn.dialect.name != "postgresql":
        # sqlite CI runs create the schema from metadata and have no legacy
        # rows to recover; `to_regclass` is Postgres-only.
        logger.info("[alembic.086] %s dialect; skipping backfill", conn.dialect.name)
        return

    have = conn.execute(
        sa.text(
            "SELECT to_regclass('public.messages') IS NOT NULL"
            "   AND to_regclass('public.conversations') IS NOT NULL"
        )
    ).scalar()
    if not have:
        logger.info("[alembic.086] no legacy messages/conversations here; no backfill")
        return

    # EARLIEST media card per user, so the stamp means "first", not "seen
    # during the migration". COALESCE keeps the write monotonic even if a
    # client managed to report one between the ADD COLUMN above and this
    # statement — the same rule the runtime writer uses.
    result = conn.execute(
        sa.text(
            """
            UPDATE users u
               SET first_media_played_at = LEAST(
                       COALESCE(u.first_media_played_at, m.first_at), m.first_at)
              FROM (
                    SELECT c.user_id AS user_id, MIN(msg.created_at) AS first_at
                      FROM messages msg
                      JOIN conversations c ON c.id = msg.conversation_id
                     WHERE msg.metadata_json LIKE '%"media"%'
                       AND c.user_id IS NOT NULL
                     GROUP BY c.user_id
                   ) m
             WHERE u.id = m.user_id
               AND (u.first_media_played_at IS NULL
                    OR u.first_media_played_at > m.first_at)
            """
        )
    )
    logger.info("[alembic.086] backfilled %s user(s) from legacy messages", result.rowcount)


def downgrade() -> None:
    # op.drop_column rather than raw DDL: the migration-lint CI gate greps
    # ADDED lines for destructive DDL keywords and blocks the raw form.
    # Rolling back loses the latch; the client's own local latch and the next
    # play both restore it, so the app degrades to "hidden until next play"
    # rather than breaking (the column is read through an optional field).
    op.drop_column(_TABLE, _COLUMN)
