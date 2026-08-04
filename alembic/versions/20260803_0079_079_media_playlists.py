"""079 — media_playlists, the Toup Media library (saved, replayable playlists).

``media_playlists`` is an AGENT_ONLY table (see ``app/db/models/base.py``): the
live store is each user's tenant container DB, written by the playlist API
running there, and every tenant gets the table from ``init_db``'s
``create_all`` on next boot — no migration needed on that side.

This revision creates the PLATFORM mirror only. The platform serves the
read-fallback when a user's agent is unreachable (the memories pattern:
``app/api/media_playlists.py`` reads proxy-with-fallback, writes
proxy-or-502), and ``init_db`` on the platform *excludes* AGENT_ONLY tables
from ``create_all`` — so without this revision the fallback SELECT would hit a
missing relation.

Gated on ``settings.run_mode`` like revision 078, and for the same
hard-learned reason: ``alembic upgrade head`` runs on boot in BOTH images, a
``CREATE TABLE … REFERENCES users`` takes a lock on ``users``, and taking that
lock inside a tenant DB during a blue/green rollout is what stalled the 078
canary. Tenants must get this table from ``create_all``, never from here.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "079"
down_revision = "078"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.079")

_TABLE = "media_playlists"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.079] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.079] not a platform DB; skipping (%s reaches tenants via create_all)",
            _TABLE,
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.079] users absent; skipping")
        return
    if _TABLE in tables:
        logger.info("[alembic.079] %s already present", _TABLE)
        return

    op.create_table(
        _TABLE,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("media_type", sa.String(20), nullable=False, server_default="music"),
        sa.Column("source", sa.String(20), nullable=False, server_default="radio"),
        sa.Column("seed_intent", sa.Text(), nullable=True),
        sa.Column("seed_video_id", sa.String(32), nullable=True),
        sa.Column("tracks_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("track_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("artwork_url", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_media_playlists_user_created "
        f"ON {_TABLE} (user_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_media_playlists_seed "
        f"ON {_TABLE} (user_id, seed_video_id)"
    )


def downgrade() -> None:
    # Deliberately a no-op, and the migration-lint guard is why.
    #
    # A blue-green rollout runs the OLD and NEW images against the same tenant
    # DB for the ~60s drain window, so removing the table here would take it
    # out from under whichever slot is still serving. The guard rejects the
    # pattern anywhere in the diff — it cannot tell that this one sat in
    # downgrade(), and it is right not to try: a destructive step that is
    # "only on downgrade" is exactly the kind that gets run by accident
    # during an incident.
    #
    # Leaving the table costs one unused table on the platform DB. Removing it
    # can cost the fleet. If it ever genuinely needs to go, that is a
    # maintenance-window operation, not an automatic downgrade step.
    logger.info(
        "[alembic.079] downgrade is a no-op; %s is left in place (see comment)",
        _TABLE,
    )
