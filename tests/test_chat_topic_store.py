"""Real SQLite transactions for the platform-only derived title store."""

import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
import importlib.util
from pathlib import Path
import uuid

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event, inspect, select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.services import chat_topic_store as store
from app.db.models.chat_topic import ChatTopicTitle

DAY = date(2026, 10, 22)
SOURCE = "a" * 24
OTHER_SOURCE = "b" * 24
USER = "user-one"
OTHER_USER = "user-two"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    # This suite builds only users + the new platform table. It must not rely
    # on messages/day_chats or conftest's combined-mode create_all.
    yield


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/topics.db", poolclass=NullPool,
        connect_args={"timeout": 10},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE users (id VARCHAR(36) PRIMARY KEY)"))
        await conn.execute(text("INSERT INTO users (id) VALUES (:one), (:two)"),
                           {"one": USER, "two": OTHER_USER})
        await conn.run_sync(ChatTopicTitle.__table__.create)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(store, "async_session_maker", maker)
    monkeypatch.setattr(store.settings, "run_mode", "platform")
    monkeypatch.setattr(store, "_next_cleanup_at", 0.0)
    yield maker
    await engine.dispose()


def identity(canonical_id="turn-one", **overrides):
    return dict(user_id=USER, day=DAY, canonical_id=canonical_id,
                fingerprint=SOURCE, version=3) | overrides


async def publish(canonical_id="turn-one", **overrides):
    args = identity(canonical_id, **overrides)
    token = uuid.uuid4().hex
    assert await store.claim_generation(**args, lease_token=token)
    assert await store.complete_generation(**args, lease_token=token, topic="Geometry definitions")
    return args


async def load(**overrides):
    args = dict(user_id=USER, day=DAY, version=3, sources={"turn-one": SOURCE})
    return await store.load_topic_titles(**(args | overrides))


async def test_durable_read_user_day_fingerprint_and_version_isolation(db):
    await publish()
    # Every call already uses a separate file-backed connection, modeling a
    # second replica reading only committed state, without any process cache.
    assert await load() == {"turn-one": "Geometry definitions"}
    assert await load(user_id=OTHER_USER) == {}
    assert await load(day=DAY - timedelta(days=1)) == {}
    assert await load(version=4) == {}
    assert await load(sources={"turn-one": OTHER_SOURCE}) == {}
    assert not await store.claim_generation(**identity(), lease_token=uuid.uuid4().hex)
    async with db() as session:
        row = (await session.execute(select(ChatTopicTitle))).scalar_one()
        assert row.created_at <= row.updated_at < row.expires_at
        assert row.lease_token is None


async def test_concurrent_replicas_only_one_generation_lease(db):
    tokens = [uuid.uuid4().hex for _ in range(10)]
    claimed = await asyncio.gather(*[
        store.claim_generation(**identity(), lease_token=token) for token in tokens
    ])
    assert sum(claimed) == 1
    winner = tokens[claimed.index(True)]
    assert not await store.complete_generation(
        **identity(), lease_token=uuid.uuid4().hex, topic="Wrong worker",
    )
    assert await store.complete_generation(
        **identity(), lease_token=winner, topic="The winning label",
    )
    assert await load() == {"turn-one": "The winning label"}


async def test_expired_lease_takeover_fences_old_publish_and_release(db):
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    assert await store.claim_generation(**identity(), lease_token=first)
    async with db.begin() as session:
        await session.execute(update(ChatTopicTitle).values(
            lease_expires_at=datetime(2000, 1, 1),
        ))
    next_identity = identity(fingerprint=OTHER_SOURCE, version=4)
    assert await store.claim_generation(**next_identity, lease_token=second)
    assert not await store.complete_generation(
        **identity(), lease_token=first, topic="Stale writer",
    )
    await store.release_generation(**identity(), lease_token=first)
    assert await store.complete_generation(
        **next_identity, lease_token=second, topic="Updated topic",
    )
    assert await load() == {}
    assert await load(version=4, sources={"turn-one": OTHER_SOURCE}) == {"turn-one": "Updated topic"}


async def test_expired_lease_cannot_publish_even_without_takeover(db):
    token = uuid.uuid4().hex
    assert await store.claim_generation(**identity(), lease_token=token)
    async with db.begin() as session:
        await session.execute(update(ChatTopicTitle).values(lease_expires_at=datetime(2000, 1, 1)))
    assert not await store.complete_generation(**identity(), lease_token=token, topic="Too late")
    assert await load() == {}


async def test_changed_source_invalidates_cache_and_can_be_claimed(db):
    await publish()
    async with db() as session:
        created = (await session.execute(select(ChatTopicTitle.created_at))).scalar_one()
    token = uuid.uuid4().hex
    args = identity(fingerprint=OTHER_SOURCE, version=4)
    assert await store.claim_generation(**args, lease_token=token)
    assert await load() == {}
    assert await store.complete_generation(**args, lease_token=token, topic="Revised subject")
    async with db() as session:
        row = (await session.execute(select(ChatTopicTitle))).scalar_one()
        assert row.created_at == created
        assert row.version == 4 and row.fingerprint == OTHER_SOURCE


async def test_title_expiry_and_bounded_cleanup_preserve_live_lease(db):
    for anchor in ("turn-one", "turn-two", "turn-three"):
        await publish(anchor)
    token = uuid.uuid4().hex
    assert await store.claim_generation(**identity("live-turn"), lease_token=token)
    async with db.begin() as session:
        # A live lease must survive even a stale cache-retention timestamp.
        await session.execute(update(ChatTopicTitle).values(expires_at=datetime(2000, 1, 1)))
    assert await load() == {}
    assert await store.cleanup_expired_titles(limit=2) == 2
    assert await store.cleanup_expired_titles(limit=2) == 1
    async with db() as session:
        row = (await session.execute(select(ChatTopicTitle))).scalar_one()
        assert row.canonical_id == "live-turn" and row.lease_token == token
    assert await store.complete_generation(**identity("live-turn"), lease_token=token, topic="Live title")


async def test_release_allows_retry_and_account_deletion_cascades(db):
    token = uuid.uuid4().hex
    assert await store.claim_generation(**identity(), lease_token=token)
    await store.release_generation(**identity(), lease_token=token)
    await publish()
    await publish(user_id=OTHER_USER)
    async with db.begin() as session:
        await session.execute(text("DELETE FROM users WHERE id = :id"), {"id": USER})
    assert await load() == {}
    assert await load(user_id=OTHER_USER) == {"turn-one": "Geometry definitions"}


async def test_unavailable_schema_is_distinct_from_lease_contention(db):
    async with db.begin() as session:
        await session.execute(text("DROP TABLE chat_topic_titles"))
    with pytest.raises(store.TopicStoreUnavailable, match="topic store unavailable"):
        await load()
    with pytest.raises(store.TopicStoreUnavailable):
        await store.claim_generation(**identity(), lease_token=uuid.uuid4().hex)


async def test_agent_mode_never_opens_platform_store(db, monkeypatch):
    monkeypatch.setattr(store.settings, "run_mode", "agent")

    def forbidden():
        pytest.fail("tenant mode opened a platform cache connection")

    monkeypatch.setattr(store, "async_session_maker", forbidden)
    with pytest.raises(store.TopicStoreUnavailable, match="platform-only"):
        await load()


async def test_optional_store_timeout_includes_connection_acquisition(db, monkeypatch):
    monkeypatch.setattr(store, "_OPERATION_TIMEOUT_SECONDS", 0.03)
    exited = asyncio.Event()

    @asynccontextmanager
    async def stalled_connection():
        try:
            await asyncio.Future()
            yield  # pragma: no cover — the connection never arrives.
        finally:
            exited.set()

    monkeypatch.setattr(store, "async_session_maker", stalled_connection)
    with pytest.raises(store.TopicStoreUnavailable):
        await asyncio.wait_for(load(), timeout=1)
    assert exited.is_set()


async def test_stalled_transaction_times_out_rolls_back_and_recovers(db, monkeypatch):
    original_now = store._database_now
    monkeypatch.setattr(store, "_OPERATION_TIMEOUT_SECONDS", 0.03)

    async def stalled_query(session):
        await session.execute(text("INSERT INTO users (id) VALUES ('uncommitted-timeout')"))
        await asyncio.Future()

    monkeypatch.setattr(store, "_database_now", stalled_query)
    with pytest.raises(store.TopicStoreUnavailable):
        await asyncio.wait_for(store.claim_generation(**identity(), lease_token=uuid.uuid4().hex), timeout=1)
    async with db() as session:
        assert (await session.execute(text("SELECT count(*) FROM users WHERE id='uncommitted-timeout'"))).scalar_one() == 0
    monkeypatch.setattr(store, "_database_now", original_now)
    monkeypatch.setattr(store, "_OPERATION_TIMEOUT_SECONDS", 2.0)
    await publish()
    assert await load() == {"turn-one": "Geometry definitions"}


async def test_external_cancellation_is_not_reported_as_cache_outage(db, monkeypatch):
    entered = asyncio.Event()

    async def stalled_query(session):
        await session.execute(text("INSERT INTO users (id) VALUES ('uncommitted-cancel')"))
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(store, "_database_now", stalled_query)
    task = asyncio.create_task(store.claim_generation(**identity(), lease_token=uuid.uuid4().hex))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with db() as session:
        assert (await session.execute(text("SELECT count(*) FROM users WHERE id='uncommitted-cancel'"))).scalar_one() == 0


async def test_timeout_also_covers_transaction_commit_phase(db, monkeypatch):
    monkeypatch.setattr(store, "_OPERATION_TIMEOUT_SECONDS", 0.03)
    reached_commit = asyncio.Event()

    class CommitStall:
        def __init__(self, session):
            self.session = session

        def __getattr__(self, name):
            return getattr(self.session, name)

        @asynccontextmanager
        async def begin(self):
            async with self.session.begin():
                yield self
                reached_commit.set()
                await asyncio.Future()  # A stalled driver before COMMIT completes.

    @asynccontextmanager
    async def stalled_session():
        async with db() as session:
            yield CommitStall(session)

    monkeypatch.setattr(store, "async_session_maker", stalled_session)
    with pytest.raises(store.TopicStoreUnavailable):
        await asyncio.wait_for(store.claim_generation(**identity(), lease_token=uuid.uuid4().hex), timeout=1)
    assert reached_commit.is_set()
    async with db() as session:
        assert (await session.execute(select(ChatTopicTitle))).first() is None


async def test_boundaries_reject_unbounded_retention_and_payloads(db):
    await publish("x" * 50)  # Message.id permits 50, even though UUIDs use 36.
    with pytest.raises(ValueError, match="too many"):
        await load(sources={str(index): SOURCE for index in range(21)})
    with pytest.raises(ValueError, match="retention"):
        await store.claim_generation(**identity(), lease_token=uuid.uuid4().hex, lease_seconds=301)
    token = uuid.uuid4().hex
    assert await store.claim_generation(**identity(), lease_token=token)
    for topic in ("raw\nmessage", "x" * 161):
        with pytest.raises(ValueError, match="generated topic"):
            await store.complete_generation(**identity(), lease_token=token, topic=topic)
    with pytest.raises(ValueError, match="retention"):
        await store.complete_generation(**identity(), lease_token=token, topic="Title", ttl_seconds=store.MAX_TTL_SECONDS + 1)


def test_additive_migration_platform_only_roundtrip(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "alembic/versions/20261022_0108_108_chat_topic_titles.py"
    spec = importlib.util.spec_from_file_location("chat_topic_migration_108", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.down_revision == "107"
    engine = create_engine(f"sqlite:///{tmp_path}/migration.db")
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE users (id VARCHAR(36) PRIMARY KEY)"))
            with Operations.context(MigrationContext.configure(conn)):
                monkeypatch.setattr(store.settings, "run_mode", "agent")
                migration.upgrade()
                assert inspect(conn).get_table_names() == ["users"]
                monkeypatch.setattr(store.settings, "run_mode", "platform")
                migration.upgrade()
                assert set(inspect(conn).get_table_names()) == {"users", "chat_topic_titles"}
                migration.upgrade()  # A targeted create_all rollout is adoptable.
                assert {column["name"] for column in inspect(conn).get_columns("chat_topic_titles")} == set(ChatTopicTitle.__table__.columns.keys())
                fks = inspect(conn).get_foreign_keys("chat_topic_titles")
                assert len(fks) == 1 and fks[0]["referred_table"] == "users"
                assert fks[0]["options"]["ondelete"] == "CASCADE"
                migration.downgrade()
                assert inspect(conn).get_table_names() == ["users"]
    finally:
        engine.dispose()
