"""105 — unlimited_grants: the admin override on the Unlimited entitlement.

Revision ID: 105
Revises: 104
Create Date: 2026-09-13

Schema only. **This migration grants nobody anything.** It creates the table
that 104's HOOK note pointed at, so ``entitlement.py`` stops degrading to
"no override" and the admin router / CLI have somewhere to write.

Why a dedicated table rather than the "tagging mechanism" the sketch asked
to reuse: there is no tagging mechanism. ``users.role`` is a single-valued
``String(20)`` with two production values, and ``role == 'admin'`` is the
EXACT string ``require_admin`` tests — so "tagging" a paying customer would
hand them broadcast dispatch, fleet rollouts, other users' data audits,
invites, role changes and account deletion. It would also *work* as an
entitlement (``_is_unlimited_user`` is the same test), which is what would
let the mistake survive long enough to matter. See
``docs/billing/UNLIMITED_ENTITLEMENT_DESIGN.md`` §4.1.

Two details that are load-bearing rather than stylistic:

  * **The sponsor link carries NO foreign key.**
    ``apple_subscriptions.user_id`` is ON DELETE CASCADE, so deleting the
    sponsor's account destroys the subscription row — and an FK here would
    take the audit trail with it. "Sponsored by original transaction X,
    which no longer exists" is a finding the reconciler must be able to
    read, not a dangling pointer to be cleaned up.

  * **``uq_unlimited_grant_live`` is a PARTIAL unique index.** A grant row
    is never deleted; revoke and lapse are recorded on the row. A plain
    unique index on ``granted_to_user_id`` would therefore make every user
    un-re-grantable forever after their first revoke. Postgres gets the
    partial index; SQLite (the test lane) supports partial indexes too and
    gets the same one, so tests exercise the real constraint. Any other
    dialect falls back to a plain non-unique index and the service layer's
    409 is the only guard there.

``downgrade()`` REFUSES to drop the table while it holds rows. Billing
history is never deleted; a loud failure telling the operator to export it
first beats silently destroying an audit trail.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "105"
down_revision = "104"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.105")

_TABLE = "unlimited_grants"
_LIVE_WHERE = "revoked_at IS NULL AND lapsed_at IS NULL"


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    tables = set(insp.get_table_names())

    if "users" not in tables:
        # Agent (per-tenant) DBs carry no user or credit state. unlimited_grants
        # is PLATFORM_ONLY (app/db/models/base.py) for the same reason the five
        # credit tables are: an empty copy answers "no grant" to a question the
        # tenant must never be asked.
        logger.info("[alembic.105] users absent; skipping (agent-only DB)")
        return

    if _TABLE in tables:
        logger.info("[alembic.105] %s already present; skipping create", _TABLE)
    else:
        op.create_table(
            _TABLE,
            sa.Column("id", sa.String(36), primary_key=True),
            # WHO holds it.
            sa.Column(
                "granted_to_user_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
            ),
            # WHAT sponsors it — denormalised, no FK. See the module docstring.
            sa.Column("sponsor_kind", sa.String(16), nullable=False),
            sa.Column("sponsor_user_id", sa.String(36), nullable=True),
            sa.Column("sponsor_original_txn_id", sa.String(64), nullable=True),
            sa.Column("sponsor_stripe_sub_id", sa.String(120), nullable=True),
            # WHY, and a backstop independent of the sponsor.
            sa.Column("reason", sa.Text(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            # WHO granted it. The email is a SNAPSHOT: the operator's account
            # can be deleted and the trail must still name them.
            sa.Column("granted_by_user_id", sa.String(36), nullable=False),
            sa.Column("granted_by_email", sa.String(320), nullable=False),
            sa.Column("granted_at", sa.DateTime(), nullable=False),
            # HOW it ended — exactly one of revoked_at / lapsed_at on a dead row.
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
            sa.Column("revoked_by_user_id", sa.String(36), nullable=True),
            sa.Column("revoked_by_email", sa.String(320), nullable=True),
            sa.Column("revoked_reason", sa.Text(), nullable=True),
            sa.Column("lapsed_at", sa.DateTime(), nullable=True),
            sa.Column("lapsed_reason", sa.String(120), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        logger.info("[alembic.105] %s created", _TABLE)

    # Re-inspect: the table may have just been created, and index creation
    # below must be idempotent against a partially-applied run.
    insp = sa.inspect(conn)
    insp.clear_cache()
    existing = {ix["name"] for ix in insp.get_indexes(_TABLE)}

    dialect = conn.dialect.name
    if "uq_unlimited_grant_live" not in existing:
        if dialect in ("postgresql", "sqlite"):
            # Partial unique: one LIVE grant per user, while revoked/lapsed
            # rows stay and stay unique-index-invisible.
            conn.execute(sa.text(
                f"CREATE UNIQUE INDEX uq_unlimited_grant_live ON {_TABLE} "
                f"(granted_to_user_id) WHERE {_LIVE_WHERE}"
            ))
            logger.info("[alembic.105] partial unique index created (%s)", dialect)
        else:
            op.create_index(
                "uq_unlimited_grant_live", _TABLE, ["granted_to_user_id"],
                unique=False,
            )
            logger.warning(
                "[alembic.105] dialect %s has no partial index — created a plain "
                "index. The one-live-grant invariant rests on the service layer's "
                "409 here.", dialect,
            )

    for name, col in (
        ("ix_unlimited_grant_user", "granted_to_user_id"),
        ("ix_unlimited_grant_sponsor_txn", "sponsor_original_txn_id"),
        ("ix_unlimited_grant_sponsor_usr", "sponsor_user_id"),
    ):
        if name not in existing:
            op.create_index(name, _TABLE, [col])
    logger.info("[alembic.105] %s indexes present", _TABLE)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    if _TABLE not in set(insp.get_table_names()):
        return

    held = conn.execute(sa.text(f"SELECT COUNT(*) FROM {_TABLE}")).scalar() or 0
    if held:
        # Never delete billing history. A grant row records who comped whom,
        # why, on whose subscription, and how it ended — including grants that
        # have already lapsed. Dropping the table would erase all of it with no
        # trace that it ever existed.
        raise RuntimeError(
            f"[alembic.105] {_TABLE} holds {held} row(s) — refusing to drop an "
            f"audit trail. Export it (SELECT * FROM {_TABLE}) and drop the table "
            f"manually if you really mean to."
        )
    op.drop_table(_TABLE)
    logger.info("[alembic.105] %s dropped (was empty)", _TABLE)
