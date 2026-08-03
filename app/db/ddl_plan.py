"""Decide which boot-time DDL statements actually need to run.

WHY THIS EXISTS

``init_db()`` replays the whole hand-maintained ``_alter_statements`` list on
every container start — ~358 statements against a LIVE tenant database. Nearly
all of them are no-ops on a converged tenant, and the list is written to be
idempotent (``ADD COLUMN IF NOT EXISTS``, ``CREATE INDEX IF NOT EXISTS``), so
this looked free.

It is not free. **Postgres takes the ACCESS EXCLUSIVE lock BEFORE it evaluates
IF NOT EXISTS.** Measured on postgres:16 (2026-08-03), isolated container:

    -- nothing else touching the table
    ALTER TABLE t ADD COLUMN IF NOT EXISTS c1 int;
    NOTICE:  column "c1" of relation "t" already exists, skipping   -- instant

    -- ONE plain `BEGIN; SELECT count(*) FROM t;` held open
    ALTER TABLE t ADD COLUMN IF NOT EXISTS c1 int;
    ERROR:  canceling statement due to lock timeout

    -- and pg_locks during the wait:
    mode                | granted
    AccessShareLock     | t        <- the plain SELECT
    AccessExclusiveLock | f        <- the "no-op" ALTER, queued behind it

So every no-op statement still queues for an exclusive lock on a table the
blue container is actively serving. That is how a blue-green green spent
**251 seconds getting through 41 of 358 statements** on 2026-08-01 while three
sibling greens on the same host finished all 358 in under a minute each — the
rollout aborted `aborted_canary_failed` with `health_checks_passed: 0`, and
the health budget had already been raised 30 -> 120 -> 240 chasing it.

THE APPROACH, AND WHY NOT A VERSION MARKER

The obvious fix — stamp a schema version in the tenant DB and skip the pass
when it matches — is the one to avoid. Tenant databases have no
``alembic_version`` (see the ALTER-mirror design), a marker can be written
when a statement actually failed, and it cannot see drift applied out of band.
Every one of those failure modes is silent, and the symptom arrives weeks
later as a missing column.

This plans against the LIVE CATALOG instead. A statement is skipped only when
the catalog says its effect is already present, which is the same question
Postgres itself would ask — we just ask it once, cheaply, in a read-only query
that takes no table locks, instead of 358 times behind an exclusive lock. It
is self-correcting by construction: drop a column out of band and the next
boot plans the statement back in.

CONSERVATISM IS THE WHOLE CONTRACT

Skipping a statement that was needed is a silent schema bug. Running one that
was not needed costs a lock. Those are not symmetric, so every uncertainty
resolves to RUN:

  * only three statement shapes are skippable at all — everything else
    (backfills, ``DO $$`` blocks, functions, triggers, ``DROP INDEX``,
    ``ALTER COLUMN``) runs untouched, every boot, exactly as before;
  * the shape must match a strict anchored pattern, with no second statement
    and no second action riding along;
  * if the snapshot could not be taken, nothing is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

# ── the three skippable shapes ────────────────────────────────────────
# Anchored at the start; the statement is whitespace-normalised first.

_ADD_COLUMN = re.compile(
    r"^ALTER\s+TABLE\s+(?:ONLY\s+)?(?:IF\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s+"
    r"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s|$)",
    re.IGNORECASE,
)
_CREATE_INDEX = re.compile(
    r"^CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?IF\s+NOT\s+EXISTS\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s+ON\s",
    re.IGNORECASE,
)
_CREATE_TABLE = re.compile(
    r"^CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.IGNORECASE,
)

# A second ALTER action (`..., ADD COLUMN y int`) would be invisible to the
# single-column match above, so any statement carrying more than one action is
# disqualified rather than half-understood.
_EXTRA_ACTION = re.compile(r"\b(ADD|DROP|ALTER|RENAME)\s+(COLUMN|CONSTRAINT)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SchemaSnapshot:
    """What the tenant database already has. All names lowercased."""

    columns: frozenset  # {(table, column)}
    indexes: frozenset  # {index_name}
    tables: frozenset  # {table_name}

    @property
    def is_empty(self) -> bool:
        return not (self.columns or self.indexes or self.tables)


def normalize(stmt: str) -> str:
    """Collapse whitespace and strip a single trailing semicolon."""
    one = " ".join(stmt.split()).strip()
    return one[:-1].rstrip() if one.endswith(";") else one


def _carries_a_second_statement(one_line: str) -> bool:
    """`CREATE TABLE ...; INSERT ...` must never be judged on its first half.

    Only meaningful for the skippable shapes — none of them legitimately
    contain a semicolon, so any interior `;` is disqualifying. (`DO $$` bodies
    are full of semicolons and match no shape, so they never reach here.)
    """
    return ";" in one_line


def is_satisfied(stmt: str, snap: SchemaSnapshot) -> bool:
    """True only when the catalog already shows this statement's effect.

    Every path that is not a confident yes returns False, i.e. run it.
    """
    if snap.is_empty:
        return False

    one = normalize(stmt)
    if _carries_a_second_statement(one):
        return False

    m = _ADD_COLUMN.match(one)
    if m:
        table, column = m.group(1).lower(), m.group(2).lower()
        # Exactly one action, and it is the ADD COLUMN we just matched.
        if len(_EXTRA_ACTION.findall(one)) != 1:
            return False
        return (table, column) in snap.columns

    m = _CREATE_INDEX.match(one)
    if m:
        return m.group(1).lower() in snap.indexes

    m = _CREATE_TABLE.match(one)
    if m:
        return m.group(1).lower() in snap.tables

    return False


def plan(
    statements: Sequence[str], snap: SchemaSnapshot
) -> Tuple[List[str], List[str]]:
    """Split `statements` into (to_run, skipped), preserving order.

    Order matters: the list is a migration sequence, and the statements that
    survive planning must reach the executor in their original relative order.
    """
    to_run: List[str] = []
    skipped: List[str] = []
    for stmt in statements:
        (skipped if is_satisfied(stmt, snap) else to_run).append(stmt)
    return to_run, skipped


def snapshot_from_rows(
    column_rows: Iterable[Tuple[str, str]],
    index_rows: Iterable[str],
    table_rows: Iterable[str],
) -> SchemaSnapshot:
    """Build a snapshot from raw catalog rows (kept pure for testing)."""
    columns: Set[Tuple[str, str]] = {
        (str(t).lower(), str(c).lower()) for t, c in column_rows
    }
    return SchemaSnapshot(
        columns=frozenset(columns),
        indexes=frozenset(str(i).lower() for i in index_rows),
        tables=frozenset(str(t).lower() for t in table_rows),
    )


# Read-only catalog queries. No table locks, one round trip each, and scoped
# to the schema the app actually uses — an unexpected search_path yields an
# empty snapshot, which disables planning rather than mis-planning.
SNAPSHOT_SQL = {
    "columns": (
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema()"
    ),
    "indexes": (
        "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
    ),
    "tables": (
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = current_schema()"
    ),
}
