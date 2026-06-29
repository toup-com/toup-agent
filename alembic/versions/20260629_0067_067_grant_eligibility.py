"""Add grant_eligibility tombstone for free-credit Sybil resistance.

Revision ID: 067
Revises: 066
Create Date: 2026-06-29

One row per canonical email identity that has received the one-time free
grant. Decoupled from `users` (no FK) so it survives account deletion and
blocks delete -> re-signup re-granting. See
app/db/models/grant_eligibility.py and app/services/email_canonical.py.

Backfill is NON-DESTRUCTIVE: it seeds a tombstone for every distinct
canonical email among users that already hold a credit_balances row
(i.e. already granted), keyed to the EARLIEST such account. Existing
accounts that canonicalize together (pre-existing multi-accounts) are
reported in the log as a collision REPORT — they are never merged or
deleted, only recorded so the first one owns the tombstone.

Idempotent + guarded: skipped entirely on DBs without `users` /
`credit_balances` (per-tenant agent DBs), and re-runnable (uses the
table's own presence + INSERT-if-absent semantics).
"""
from __future__ import annotations

import logging
from datetime import datetime

import sqlalchemy as sa
from alembic import op


revision = "067"
down_revision = "066"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.067")


def _backfill(conn) -> None:
    """Seed tombstones from already-granted users. Best-effort: any failure
    here is logged and swallowed so a backfill hiccup never blocks the
    schema change (the table is the load-bearing part; the backfill only
    protects pre-existing accounts)."""
    try:
        from app.services.email_canonical import canonical_email_hash
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[alembic.067] cannot import canonicalizer; skipping backfill: %s", e)
        return

    rows = conn.execute(sa.text(
        "SELECT u.id, u.email FROM users u "
        "JOIN credit_balances b ON b.user_id = u.id "
        "ORDER BY u.created_at ASC, u.id ASC"
    )).fetchall()

    seen: dict[str, str] = {}      # canonical_hash -> first_user_id
    collisions: dict[str, int] = {}  # canonical_hash -> count of extra accounts
    inserted = 0
    for uid, email in rows:
        if not email:
            continue
        h = canonical_email_hash(email)
        if h in seen:
            collisions[h] = collisions.get(h, 1) + 1
            continue
        seen[h] = uid
        # INSERT-if-absent: re-runs and any pre-seeded rows are no-ops.
        existing = conn.execute(
            sa.text("SELECT 1 FROM grant_eligibility WHERE canonical_email_hash = :h"),
            {"h": h},
        ).first()
        if existing:
            continue
        conn.execute(
            sa.text(
                "INSERT INTO grant_eligibility (canonical_email_hash, granted_at, first_user_id) "
                "VALUES (:h, :ts, :uid)"
            ),
            {"h": h, "ts": datetime.utcnow(), "uid": uid},
        )
        inserted += 1

    logger.info(
        "[alembic.067] backfill: %d users -> %d tombstones seeded, %d canonical collisions",
        len(rows), inserted, len(collisions),
    )
    if collisions:
        # Collision REPORT — do NOT merge/delete. Operators review these as
        # pre-existing multi-accounts; the earliest account owns the grant.
        total_extra = sum(collisions.values())
        logger.warning(
            "[alembic.067] COLLISION REPORT: %d canonical identities have multiple "
            "existing accounts (%d extra accounts total). These were NOT merged or "
            "deleted. Hashes (first 12 chars): %s",
            len(collisions), total_extra,
            ", ".join(sorted(h[:12] for h in collisions)),
        )


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables or "credit_balances" not in tables:
        logger.info("[alembic.067] users/credit_balances absent; skipping (not a platform DB)")
        return

    if "grant_eligibility" not in tables:
        op.create_table(
            "grant_eligibility",
            sa.Column("canonical_email_hash", sa.String(64), primary_key=True),
            sa.Column("granted_at", sa.DateTime(), nullable=False),
            sa.Column("first_user_id", sa.String(36), nullable=True),
        )
        logger.info("[alembic.067] created grant_eligibility")
    else:
        logger.info("[alembic.067] grant_eligibility already present; skipping create")

    _backfill(conn)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    if "grant_eligibility" in set(insp.get_table_names()):
        op.drop_table("grant_eligibility")
