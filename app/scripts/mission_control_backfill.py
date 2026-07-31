"""Mission Control Phase 2f — legacy backfill, retention, idempotency index.

**DRY RUN BY DEFAULT.** Nothing is written unless ``--apply`` is passed.
This is the only step of the Mission Control overhaul that can lose data, so
it is a reviewable report first and a mutation second.

    # report only (safe, read-only)
    python -m app.scripts.mission_control_backfill

    # each stage is opt-in and independent
    python -m app.scripts.mission_control_backfill --apply --classify
    python -m app.scripts.mission_control_backfill --apply --archive-infra
    python -m app.scripts.mission_control_backfill --apply --retention
    python -m app.scripts.mission_control_backfill --apply --index

Run against a TENANT AGENT database — ``build_jobs`` is AGENT_ONLY.

What each stage does
--------------------
``--classify``      Fill ``error_class`` / ``user_message`` / ``technical_detail``
                    on rows that predate the taxonomy. ``error_message`` is
                    copied verbatim into ``technical_detail`` and never
                    cleared, so the change is reversible and no history is
                    lost. NOTE: the API also classifies on read, so this
                    stage is an optimisation, not a correctness fix.
``--archive-infra`` Stamp ``archived_at`` on rows whose only failure was our
                    own infrastructure (``infra_restart`` /
                    ``infra_interrupted``) older than the cutoff. Archive is
                    NOT delete — the rows stay queryable and a deep link
                    still resolves.
``--retention``     Archive terminal rows older than ``--retention-days``.
                    Also archive-only.
``--index``         Create the partial UNIQUE
                    ``uq_build_jobs_source_idempotency``. This is the index
                    every dedupe comment in the codebase cites as "the gate"
                    — it exists only in alembic 046 and therefore never
                    reached a single agent DB. Creation FAILS if duplicates
                    exist, so the report lists them first; resolving them is
                    a deliberate human decision, never automatic.

Deletion is not implemented on purpose. Nothing here removes a row.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text

INDEX_NAME = "uq_build_jobs_source_idempotency"

#: Failure classes that are our fault and carry no user value once past.
INFRA_CLASSES = ("infra_restart", "infra_interrupted", "infra_unrecoverable")


def _hr(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


async def _report_classify(db: Any) -> list[tuple[str, int]]:
    """Group un-classified failures by what they WOULD become."""
    from app.agent.job_status import classify
    from app.db.models import BuildJob

    rows = (await db.execute(
        select(BuildJob.id, BuildJob.error_message)
        .where(
            BuildJob.error_message.isnot(None),
            BuildJob.error_class.is_(None),
        )
    )).all()

    buckets: dict[str, int] = {}
    for _jid, msg in rows:
        buckets[classify(msg).error_class] = buckets.get(classify(msg).error_class, 0) + 1
    return sorted(buckets.items(), key=lambda kv: -kv[1])


async def _apply_classify(db: Any) -> int:
    from app.agent.job_status import classify, technical_detail
    from app.db.models import BuildJob

    rows = (await db.execute(
        select(BuildJob).where(
            BuildJob.error_message.isnot(None),
            BuildJob.error_class.is_(None),
        )
    )).scalars().all()

    for job in rows:
        verdict = classify(job.error_message)
        job.error_class = verdict.error_class
        job.user_message = verdict.user_message
        # Preserve the original verbatim; `error_message` is left intact so
        # the change stays reversible.
        if not job.technical_detail:
            job.technical_detail = technical_detail(job.error_message)
    await db.commit()
    return len(rows)


async def _report_duplicates(db: Any) -> list[tuple[str, str, int]]:
    """(source_id, idempotency_key) pairs that would break the UNIQUE index."""
    from app.db.models import BuildJob

    return (await db.execute(
        select(BuildJob.source_id, BuildJob.idempotency_key, func.count().label("n"))
        .where(
            BuildJob.source_id.isnot(None),
            BuildJob.idempotency_key.isnot(None),
        )
        .group_by(BuildJob.source_id, BuildJob.idempotency_key)
        .having(func.count() > 1)
        .order_by(func.count().desc())
    )).all()


async def _report_archivable(db: Any, *, infra_before: datetime,
                             terminal_before: datetime) -> dict[str, int]:
    from app.agent.job_status import TERMINAL_STATUSES
    from app.db.models import BuildJob

    infra = (await db.execute(
        select(func.count()).select_from(BuildJob).where(
            BuildJob.archived_at.is_(None),
            BuildJob.error_class.in_(INFRA_CLASSES),
            BuildJob.created_at < infra_before,
        )
    )).scalar() or 0

    terminal = (await db.execute(
        select(func.count()).select_from(BuildJob).where(
            BuildJob.archived_at.is_(None),
            BuildJob.status.in_(tuple(TERMINAL_STATUSES)),
            BuildJob.created_at < terminal_before,
        )
    )).scalar() or 0

    tests = (await db.execute(
        select(func.count()).select_from(BuildJob).where(
            BuildJob.archived_at.is_(None),
            BuildJob.idempotency_key.like("test:%"),
        )
    )).scalar() or 0

    return {"infra": infra, "terminal": terminal, "test_artifacts": tests}


async def _apply_archive(db: Any, *, where_clauses: list[Any], now: datetime) -> int:
    from sqlalchemy import update

    from app.db.models import BuildJob

    total = 0
    for clause in where_clauses:
        res = await db.execute(
            update(BuildJob)
            .where(BuildJob.archived_at.is_(None), clause)
            .values(archived_at=now)
            .execution_options(synchronize_session=False)
        )
        total += getattr(res, "rowcount", 0) or 0
    await db.commit()
    return total


async def _apply_index(db: Any) -> str:
    """Create the partial UNIQUE index. Fails loudly on duplicates."""
    dups = await _report_duplicates(db)
    if dups:
        return (
            f"REFUSED — {len(dups)} duplicate (source_id, idempotency_key) "
            "pair(s) exist. Resolve them deliberately first; this script will "
            "not choose which row to sacrifice."
        )
    # CONCURRENTLY cannot run in a transaction; agent DBs are small enough
    # that a brief lock is acceptable, and IF NOT EXISTS keeps it re-runnable.
    await db.execute(text(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {INDEX_NAME} "
        "ON build_jobs (source_id, idempotency_key) "
        "WHERE idempotency_key IS NOT NULL"
    ))
    await db.commit()
    return "created"


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    p.add_argument("--classify", action="store_true")
    p.add_argument("--archive-infra", action="store_true")
    p.add_argument("--retention", action="store_true")
    p.add_argument("--index", action="store_true")
    p.add_argument("--infra-days", type=int, default=7)
    p.add_argument("--retention-days", type=int, default=30)
    args = p.parse_args()

    from app.agent.job_status import TERMINAL_STATUSES
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    now = datetime.utcnow()
    infra_before = now - timedelta(days=args.infra_days)
    terminal_before = now - timedelta(days=args.retention_days)

    async with async_session_maker() as db:
        total = (await db.execute(
            select(func.count()).select_from(BuildJob)
        )).scalar() or 0

        print(f"\nbuild_jobs rows: {total}")
        print(f"mode: {'APPLY (writes!)' if args.apply else 'DRY RUN (read-only)'}")

        _hr("1. Un-classified failures → what they would become")
        buckets = await _report_classify(db)
        if not buckets:
            print("  nothing to classify")
        for cls, n in buckets:
            print(f"  {n:6d}  {cls}")

        _hr("2. Archivable (archive ≠ delete; rows stay queryable)")
        counts = await _report_archivable(
            db, infra_before=infra_before, terminal_before=terminal_before,
        )
        print(f"  {counts['infra']:6d}  our-fault infra failures older than {args.infra_days}d")
        print(f"  {counts['terminal']:6d}  terminal rows older than {args.retention_days}d")
        print(f"  {counts['test_artifacts']:6d}  test-fire artifacts (idempotency_key LIKE 'test:%')")

        _hr(f"3. Duplicates blocking {INDEX_NAME}")
        dups = await _report_duplicates(db)
        if not dups:
            print("  none — the UNIQUE index can be created safely")
        else:
            print(f"  {len(dups)} colliding pair(s):")
            for src, key, n in dups[:20]:
                print(f"    {n}x  source_id={src}  idempotency_key={key}")
            if len(dups) > 20:
                print(f"    … and {len(dups) - 20} more")
            print("\n  These must be resolved DELIBERATELY. This script never")
            print("  picks a row to delete.")

        if not args.apply:
            _hr("DRY RUN — nothing was written")
            print("  re-run with --apply plus the stage flags you want")
            return

        # ── mutations, each opt-in ────────────────────────────────────
        if args.classify:
            n = await _apply_classify(db)
            print(f"\n✓ classified {n} row(s) (error_message preserved verbatim)")

        if args.archive_infra:
            n = await _apply_archive(db, now=now, where_clauses=[
                (BuildJob.error_class.in_(INFRA_CLASSES))
                & (BuildJob.created_at < infra_before),
                BuildJob.idempotency_key.like("test:%"),
            ])
            print(f"✓ archived {n} infra/test row(s)")

        if args.retention:
            n = await _apply_archive(db, now=now, where_clauses=[
                (BuildJob.status.in_(tuple(TERMINAL_STATUSES)))
                & (BuildJob.created_at < terminal_before),
            ])
            print(f"✓ archived {n} aged terminal row(s)")

        if args.index:
            print(f"✓ index: {await _apply_index(db)}")


if __name__ == "__main__":
    asyncio.run(main())