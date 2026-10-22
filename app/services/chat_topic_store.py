"""Bounded platform title cache with fenced, cross-replica generation leases.

The caller first verifies anchors against the requesting user's visible day.
This module accepts only IDs, fingerprints and generated labels; it never reads
tenant messages. Every public operation closes its own short transaction before
returning. In particular no connection remains checked out during model work.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
import re
import time

from sqlalchemy import and_, delete, func, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.db.database import async_session_maker
from app.db.models.chat_topic import ChatTopicTitle

DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_TTL_SECONDS = 90 * 24 * 60 * 60
MAX_BATCH_SIZE = 20
_OPERATION_TIMEOUT_SECONDS = 2.0
_CLEANUP_INTERVAL_SECONDS = 300
_CLEANUP_BATCH_SIZE = 200
_next_cleanup_at = 0.0
_HEX = re.compile(r"[0-9a-f]{16,64}\Z")
_table = ChatTopicTitle.__table__


class TopicStoreUnavailable(Exception):
    """The optional durable cache cannot be used; local fallback is allowed."""


def _scope(user_id: str, day: date, version: int) -> None:
    if not isinstance(user_id, str) or not 1 <= len(user_id) <= 36:
        raise ValueError("invalid topic owner")
    if not isinstance(day, date) or isinstance(day, datetime):
        raise ValueError("invalid topic day")
    if type(version) is not int or not 1 <= version <= 2_147_483_647:
        raise ValueError("invalid topic version")


def _source(canonical_id: str, fingerprint: str) -> None:
    if not isinstance(canonical_id, str) or not 1 <= len(canonical_id) <= 50:
        raise ValueError("invalid topic anchor")
    if not isinstance(fingerprint, str) or not _HEX.fullmatch(fingerprint):
        raise ValueError("invalid topic fingerprint")


def _lease(lease_token: str) -> None:
    if not isinstance(lease_token, str) or not 16 <= len(lease_token) <= 64:
        raise ValueError("invalid topic lease")


def _duration(value: int, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("invalid topic retention duration")
    return value


@asynccontextmanager
async def _session():
    if settings.run_mode == "agent":
        # Tenant metadata caching remains local; never query a missing platform
        # table there or create it as a side effect of a request.
        raise TopicStoreUnavailable("topic store is platform-only")
    try:
        # Bound the whole optional operation: connection acquisition, queries,
        # and commit. Context unwinding rolls back/closes before timeout is
        # translated. A caller's own cancellation still propagates unchanged.
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            async with async_session_maker() as db:
                async with db.begin():
                    yield db
    except (SQLAlchemyError, RuntimeError, OSError, TimeoutError):
        # No SQL/parameters, title text or credentials escape in a cache error.
        raise TopicStoreUnavailable("topic store unavailable") from None


async def _database_now(db) -> datetime:
    now = (await db.execute(select(func.current_timestamp()))).scalar_one()
    return now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo else now


def _identity(user_id, day, canonical_id):
    return and_(
        _table.c.user_id == user_id, _table.c.day == day,
        _table.c.canonical_id == canonical_id,
    )


def _owned_lease(user_id, day, canonical_id, fingerprint, version, lease_token):
    return and_(
        _identity(user_id, day, canonical_id),
        _table.c.fingerprint == fingerprint, _table.c.version == version,
        _table.c.lease_token == lease_token,
    )


async def load_topic_titles(*, user_id: str, day: date, version: int,
                            sources: dict[str, str]) -> dict[str, str]:
    _scope(user_id, day, version)
    if len(sources) > MAX_BATCH_SIZE:
        raise ValueError("too many topic anchors")
    for canonical_id, fingerprint in sources.items():
        _source(canonical_id, fingerprint)
    if not sources:
        return {}
    async with _session() as db:
        now = await _database_now(db)
        rows = (await db.execute(select(
            _table.c.canonical_id, _table.c.fingerprint, _table.c.topic,
        ).where(
            _table.c.user_id == user_id, _table.c.day == day,
            _table.c.version == version,
            _table.c.canonical_id.in_(sources),
            _table.c.topic.is_not(None), _table.c.expires_at > now,
        ))).all()
        return {row.canonical_id: row.topic for row in rows
                if row.fingerprint == sources[row.canonical_id]}


async def _prune(db, now: datetime, limit: int) -> int:
    expired = and_(
        _table.c.expires_at <= now,
        or_(_table.c.lease_expires_at.is_(None), _table.c.lease_expires_at <= now),
    )
    keys = select(_table.c.user_id, _table.c.day, _table.c.canonical_id).where(
        expired,
    ).order_by(_table.c.expires_at).limit(limit)
    result = await db.execute(delete(_table).where(
        expired,
        tuple_(_table.c.user_id, _table.c.day, _table.c.canonical_id).in_(keys),
    ))
    return result.rowcount


async def cleanup_expired_titles(*, limit: int = _CLEANUP_BATCH_SIZE) -> int:
    """Delete at most 200 expired rows; live generation leases are preserved."""
    _duration(limit, _CLEANUP_BATCH_SIZE)
    async with _session() as db:
        return await _prune(db, await _database_now(db), limit)


async def claim_generation(*, user_id: str, day: date, canonical_id: str,
                           fingerprint: str, version: int, lease_token: str,
                           lease_seconds: int = 90) -> bool:
    """Atomically claim absent/stale work; False means cached or already leased.

    PostgreSQL ON CONFLICT locks the canonical row and re-evaluates its WHERE
    condition after a competing writer commits. SQLite uses the same conditional
    UPSERT under its writer lock. No check-then-insert race or advisory lock.
    """
    global _next_cleanup_at
    _scope(user_id, day, version)
    _source(canonical_id, fingerprint)
    _lease(lease_token)
    _duration(lease_seconds, 300)
    async with _session() as db:
        dialect = db.bind.dialect.name
        insert = {"postgresql": pg_insert, "sqlite": sqlite_insert}.get(dialect)
        if insert is None:
            raise TopicStoreUnavailable("topic store dialect unsupported")
        now = await _database_now(db)
        # Amortized, indexed, bounded maintenance. The process clock only gates
        # sweep frequency; row expiry and lease ownership use database time.
        if time.monotonic() >= _next_cleanup_at:
            _next_cleanup_at = time.monotonic() + _CLEANUP_INTERVAL_SECONDS
            await _prune(db, now, _CLEANUP_BATCH_SIZE)
        expires = now + timedelta(seconds=lease_seconds)
        values = dict(
            user_id=user_id, day=day, canonical_id=canonical_id,
            fingerprint=fingerprint, version=version, topic=None,
            lease_token=lease_token, lease_expires_at=expires,
            created_at=now, updated_at=now, expires_at=expires,
        )
        statement = insert(_table).values(**values).on_conflict_do_update(
            index_elements=[_table.c.user_id, _table.c.day, _table.c.canonical_id],
            set_={key: value for key, value in values.items()
                  if key not in {"user_id", "day", "canonical_id", "created_at"}},
            where=and_(
                or_(_table.c.lease_expires_at.is_(None), _table.c.lease_expires_at <= now),
                or_(_table.c.topic.is_(None), _table.c.expires_at <= now,
                    _table.c.fingerprint != fingerprint, _table.c.version != version),
            ),
        ).returning(_table.c.canonical_id)
        return (await db.execute(statement)).scalar_one_or_none() is not None


async def complete_generation(*, user_id: str, day: date, canonical_id: str,
                              fingerprint: str, version: int, lease_token: str,
                              topic: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> bool:
    _scope(user_id, day, version)
    _source(canonical_id, fingerprint)
    _lease(lease_token)
    _duration(ttl_seconds, MAX_TTL_SECONDS)
    if (not isinstance(topic, str) or not 1 <= len(topic) <= 160
            or topic != topic.strip() or any(ord(char) < 32 or ord(char) == 127 for char in topic)
            or "\u2028" in topic or "\u2029" in topic):
        raise ValueError("invalid generated topic")
    async with _session() as db:
        now = await _database_now(db)
        result = await db.execute(update(_table).where(
            _owned_lease(user_id, day, canonical_id, fingerprint, version, lease_token),
            _table.c.lease_expires_at > now,
        ).values(
            topic=topic, lease_token=None, lease_expires_at=None,
            updated_at=now, expires_at=now + timedelta(seconds=ttl_seconds),
        ))
        return result.rowcount == 1


async def release_generation(*, user_id: str, day: date, canonical_id: str,
                             fingerprint: str, version: int, lease_token: str) -> None:
    _scope(user_id, day, version)
    _source(canonical_id, fingerprint)
    _lease(lease_token)
    async with _session() as db:
        now = await _database_now(db)
        await db.execute(update(_table).where(
            _owned_lease(user_id, day, canonical_id, fingerprint, version, lease_token),
        ).values(lease_token=None, lease_expires_at=None, updated_at=now, expires_at=now))
