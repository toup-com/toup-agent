"""One fleet-wide runner for the platform's singleton background loops.

The 2026-09-12 onboarding incident, in one sentence: every infra-mutating
loop on `platform-api` ran on BOTH Railway replicas with no leader election,
so a single bad platform-side mapping produced 92 container restarts over
2 h 28 m — 46 ticks, each carrying exactly one restart POST from each replica
(and one duplicated Telegram alert pair every ~13 minutes).

Why a lease ROW and not `pg_advisory_lock` (L3 §11.1):

  * Session-scoped `pg_advisory_lock` is bound to a CONNECTION. This platform
    talks to Postgres through a transaction pooler (`pool_service.py` says so
    where it picks the xact-scoped variant), where the connection under a
    session is not stable. A session-level lock there is a lock you cannot
    reliably release.
  * `pg_advisory_xact_lock` IS safe under that pooler — and is exactly right
    for the short per-user critical sections `pool_service` already guards —
    but holding one for the length of a tick means holding an open transaction,
    and therefore a pooler connection, for 15-60 s per tick per replica. Wrong
    trade for a loop, and invisible to an operator.

So: one row per loop, claimed and renewed in ONE statement (no long
transaction), handed over by TTL expiry when a holder dies, and queryable —
`SELECT * FROM infra_leases` answers "which replica has been running the
reconciler, and for how many ticks", which nothing answered before.

The contract every caller relies on:

  * `acquire_lease` is a compare-and-set. At most ONE holder can hold a given
    name at any instant, because the WHERE clause on the upsert only lets the
    current holder renew or a NEW holder take an EXPIRED lease.
  * A replica that loses the race SKIPS the tick and sleeps. It does not wait,
    and it does not do half the work.
  * `ttl = 3 x interval`, so one missed renewal is tolerated and a dead holder
    is replaced within one TTL.
  * Clock skew between replicas cannot produce two runners: the comparison
    happens inside one atomic statement against one row. Skew only changes how
    EAGERLY a handover happens, which is why the TTL is 3x and not 1.05x.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Naive UTC, like every other timestamp column in this schema.

    `datetime.utcnow()` is deprecated in 3.12 and the warning fired on EVERY
    acquire — about fourteen lines a minute across two replicas into the
    Railway logs that are the platform's forensic store. The columns are
    `sa.DateTime()` without a timezone, so the replacement must stay naive:
    an aware value would compare against `:now` differently on the two
    dialects the upsert supports.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


_HOLDER_ID: Optional[str] = None


def holder_id() -> str:
    """This process's identity as a lease holder.

    `RAILWAY_REPLICA_ID` when the platform is deployed on Railway (so the row
    names something an operator can find in the dashboard); otherwise a
    per-process uuid, which is still correct — two processes never share one.
    Cached so a renewal cannot silently change identity mid-tick.
    """
    global _HOLDER_ID
    if _HOLDER_ID is None:
        env = (
            os.environ.get("RAILWAY_REPLICA_ID")
            or os.environ.get("RAILWAY_INSTANCE_ID")
            or ""
        ).strip()
        _HOLDER_ID = (env or f"proc-{uuid.uuid4().hex[:12]}")[:80]
    return _HOLDER_ID


def reset_for_tests() -> None:
    global _HOLDER_ID
    _HOLDER_ID = None


async def acquire_lease(
    name: str, holder: Optional[str] = None, ttl_s: float = 540.0,
) -> bool:
    """Claim or renew the lease `name` for `holder`. True iff we hold it.

    One statement:

        INSERT INTO infra_leases (...) VALUES (...)
        ON CONFLICT (name) DO UPDATE SET holder = :me, expires_at = :exp, ...
         WHERE infra_leases.expires_at < :now OR infra_leases.holder = :me
        RETURNING holder

    If the WHERE fails, the upsert updates nothing and RETURNING yields no
    row — which is precisely "somebody else holds a live lease", i.e. not
    leader. There is no read-then-write window for two replicas to slip
    through, which is the whole reason this is one statement and not three.

    Never raises: a DB blip must not take a background loop down with it. It
    answers False, the tick is skipped, and the next tick asks again — the
    same durability model every one of these loops already has. (Answering
    True on error would re-create the two-runner bug on exactly the day the
    database is unhappy.)
    """
    me = (holder or holder_id())[:80]
    ttl = max(1.0, float(ttl_s))
    try:
        from app.db.database import async_session_maker, get_engine
        from app.db.models import InfraLease

        now = _utcnow()
        expires = now + timedelta(seconds=ttl)

        dialect = get_engine().dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as _insert
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as _insert
        else:                                   # pragma: no cover
            logger.warning(
                "[infra-lease] dialect %s has no upsert — refusing to guess; "
                "loop '%s' will not run", dialect, name,
            )
            return False

        stmt = _insert(InfraLease).values(
            name=name, holder=me, acquired_at=now, expires_at=expires,
            tick_seq=1,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[InfraLease.name],
            set_={
                "holder": me,
                # Keep our own acquired_at across renewals so the row answers
                # "since when has THIS replica been the runner"; reset it when
                # the lease changes hands.
                "acquired_at": now,
                "expires_at": expires,
                "tick_seq": InfraLease.tick_seq + 1,
            },
            where=(
                (InfraLease.expires_at < now) | (InfraLease.holder == me)
            ),
        ).returning(InfraLease.holder, InfraLease.tick_seq)

        async with async_session_maker() as db:
            row = (await db.execute(stmt)).first()
            await db.commit()

        if row is None:
            return False
        return str(row[0]) == me
    except Exception:
        logger.warning(
            "[infra-lease] acquire failed for '%s' — skipping this tick",
            name, exc_info=True,
        )
        return False


async def release_lease(name: str, holder: Optional[str] = None) -> bool:
    """Give up a lease we hold, so the next replica takes over immediately
    instead of after a TTL. Best-effort — shutdown paths only. Never raises.
    """
    me = (holder or holder_id())[:80]
    try:
        from app.db.database import async_session_maker
        from app.db.models import InfraLease
        from sqlalchemy import delete

        async with async_session_maker() as db:
            res = await db.execute(
                delete(InfraLease).where(
                    InfraLease.name == name, InfraLease.holder == me,
                )
            )
            await db.commit()
        return bool(res.rowcount)
    except Exception:
        logger.info("[infra-lease] release failed for '%s'", name, exc_info=True)
        return False


async def lease_holder(name: str) -> Optional[str]:
    """Who holds `name` right now (expired or not). Operator/diagnostic use."""
    try:
        from app.db.database import async_session_maker
        from app.db.models import InfraLease

        async with async_session_maker() as db:
            return (await db.execute(
                select(InfraLease.holder).where(InfraLease.name == name)
            )).scalar_one_or_none()
    except Exception:
        return None


def lease_ttl_for(interval_s: float) -> float:
    """TTL = 3 x the loop's interval.

    One missed renewal (a slow tick, a GC pause, a DB blip) must not hand the
    lease to the other replica mid-work, and a DEAD holder must be replaced
    without an operator. 3x is the smallest multiple that buys both.
    """
    return max(30.0, float(interval_s) * 3.0)
