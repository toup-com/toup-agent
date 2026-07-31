"""077 — job error taxonomy, archival, and progress columns.

Mission Control overhaul, Phase 2. Additive and fully reversible.

Context (docs/audits/mission-control-audit.md): a raw Python ``repr(exc)``
travelled five hops from the job handler to a React Native ``<Text>`` node
with zero mapping. Users read 402 JSON blobs, ``BadRequestError`` reprs and
``AttributeError("'BuildJob' object has no attribute …")``. These columns
split that single free-text field into a routing key, humanized copy, and an
internal-only detail string.

``error_message`` is deliberately NOT dropped:
  * the down-migration must restore the previous rendering exactly, and
  * legacy history is preserved rather than rewritten.

NOTE ON REACH — this migration alone is NOT sufficient.
``build_jobs`` is AGENT_ONLY (see app/db/models/base.py): it lives in each
tenant's agent database, which boots via ``init_db`` and never runs alembic.
The authoritative DDL for agent DBs is the mirrored ``ADD COLUMN IF NOT
EXISTS`` block in ``app/db/database.py::_alter_statements``. This revision
exists so the platform DB (which carries the table from the pre-split era)
stays consistent and so the change is reversible from one place. Historic
precedent for the split: migration 046 created
``uq_build_jobs_source_idempotency`` here and it consequently never reached a
single agent DB — the root cause of the unguarded-INSERT finding.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "077"
down_revision = "076"
branch_labels = None
depends_on = None


_COLUMNS = (
    ("error_class", sa.String(40)),
    ("user_message", sa.Text()),
    ("technical_detail", sa.Text()),
    ("archived_at", sa.DateTime()),
    ("progress_step", sa.Integer()),
    ("progress_total", sa.Integer()),
)


def _has_table(name: str) -> bool:
    bind = op.get_bind()
    return sa.inspect(bind).has_table(name)


def _existing_columns(table: str) -> set[str]:
    bind = op.get_bind()
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    # No-op on databases where build_jobs was never created (a
    # cleanly-partitioned platform DB), so this is safe fleet-wide.
    if not _has_table("build_jobs"):
        return

    existing = _existing_columns("build_jobs")
    for name, type_ in _COLUMNS:
        if name not in existing:
            op.add_column("build_jobs", sa.Column(name, type_, nullable=True))

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_build_jobs_archived_at "
        "ON build_jobs (archived_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_build_jobs_user_active "
        "ON build_jobs (user_id, created_at DESC) WHERE archived_at IS NULL"
    )


def downgrade() -> None:
    if not _has_table("build_jobs"):
        return

    op.execute("DROP INDEX IF EXISTS ix_build_jobs_user_active")
    op.execute("DROP INDEX IF EXISTS ix_build_jobs_archived_at")

    existing = _existing_columns("build_jobs")
    for name, _type in reversed(_COLUMNS):
        if name in existing:
            op.drop_column("build_jobs", name)
