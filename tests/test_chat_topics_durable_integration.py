"""Canonical topic resolution against the real durable store, without an LLM."""

import asyncio
from datetime import date
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db.models.chat_topic import ChatTopicTitle
from app.services import chat_topics as topics
from app.services import chat_topic_store as store

DAY = date(2026, 10, 22)
USER = "owner"
ROWS = [
    {"id": "user-anchor", "role": "user", "conversation_id": "conversation",
     "client_msg_id": "client-turn", "content": "Test the pause button with 200 geometry sentences."},
    {"id": "reply", "role": "assistant", "conversation_id": "conversation",
     "client_msg_id": "client-turn", "content": "1. A circle has no corners."},
]


@pytest.fixture(autouse=True)
def _reset_database():
    # Avoid conftest's unrelated combined-mode schema; model calls are mocked.
    topics._RECENT.clear()
    topics._RETRY_AFTER.clear()
    topics._USER_LOCKS.clear()
    topics._REMOTE_CACHE.clear()
    topics._HISTORY_CACHE.clear()
    topics._HISTORY_READS.clear()
    topics._HISTORY_BYTES = 0


@pytest_asyncio.fixture
async def durable_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/integration.db", poolclass=NullPool)
    async with engine.begin() as connection:
        await connection.execute(text("CREATE TABLE users (id VARCHAR(36) PRIMARY KEY)"))
        await connection.execute(text("INSERT INTO users (id) VALUES ('owner'), ('stranger')"))
        await connection.run_sync(ChatTopicTitle.__table__.create)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(store, "async_session_maker", maker)
    monkeypatch.setattr(store.settings, "run_mode", "platform")
    monkeypatch.setattr(store, "_next_cleanup_at", 0.0)
    yield maker
    await engine.dispose()


def group():
    groups, _ = topics._canonical_groups(topics._history_rows(ROWS))
    return {"user-anchor": groups["user-anchor"]}


def lease_identity(*, lease_token):
    return dict(user_id=USER, day=DAY, canonical_id="user-anchor",
                fingerprint=group()["user-anchor"]["fingerprint"],
                version=topics._VERSION, lease_token=lease_token)


async def fetch(ids=None, rows=None, **kwargs):
    return await topics.get_remote_chat_topics(
        user_id=USER, day=DAY, message_ids=ids or ["reply"],
        messages=ROWS if rows is None else rows, **kwargs,
    )


async def test_replica_local_cache_reset_reads_durable_canonical_title(durable_db, monkeypatch):
    generate = AsyncMock(return_value={"user-anchor": "Testing the chat pause button"})
    monkeypatch.setattr(topics, "_generate", generate)
    first = await fetch()
    assert first["topics"][0]["message_id"] == "reply"
    assert first["topics"][0]["canonical_message_id"] == "user-anchor"
    assert {"reply", "user-anchor"} <= set(first["topics"][0]["alias_message_ids"])
    topics._REMOTE_CACHE.clear()  # Another replica/restart knows no process cache.
    second = await fetch(["user-anchor"])
    assert second["topics"][0]["topic"] == first["topics"][0]["topic"]
    assert generate.await_count == 1
    async with durable_db() as db:
        row = (await db.execute(select(ChatTopicTitle))).scalar_one()
        assert row.canonical_id == "user-anchor" and row.lease_token is None


async def test_requested_anchors_must_be_in_owned_visible_rows_before_cache(durable_db, monkeypatch):
    load = AsyncMock(wraps=store.load_topic_titles)
    generate = AsyncMock(return_value={})
    monkeypatch.setattr(store, "load_topic_titles", load)
    monkeypatch.setattr(topics, "_generate", generate)
    result = await fetch(["foreign", "hidden"], rows=[*ROWS, {
        "id": "hidden", "role": "user", "content": "Operator internal text", "origin": "admin",
    }])
    assert result == {"topics": [], "unavailable": True}
    load.assert_not_awaited()
    generate.assert_not_awaited()
    result = await fetch(["foreign", "reply"])
    assert result["topics"] == []
    assert load.await_args_list[0].kwargs["sources"].keys() == {"user-anchor"}
    assert "foreign" not in str(generate.await_args_list)


async def test_identical_anchor_ids_do_not_share_titles_across_accounts(durable_db, monkeypatch):
    async def generate(user_id, _turns):
        return {"user-anchor": "Owner pause test" if user_id == USER else "Stranger pause test"}

    monkeypatch.setattr(topics, "_generate", generate)
    owner = await fetch()
    stranger = await topics.get_remote_chat_topics(
        user_id="stranger", day=DAY, message_ids=["reply"], messages=ROWS,
    )
    assert owner["topics"][0]["topic"] == "Owner pause test"
    assert stranger["topics"][0]["topic"] == "Stranger pause test"
    topics._REMOTE_CACHE.clear()
    monkeypatch.setattr(topics, "_generate", AsyncMock(side_effect=AssertionError("durable hit expected")))
    assert (await fetch())["topics"][0]["topic"] == "Owner pause test"


async def test_lease_contention_waits_for_other_replicas_durable_winner(durable_db, monkeypatch):
    args = lease_identity(lease_token=uuid.uuid4().hex)
    assert await store.claim_generation(**args)
    generate = AsyncMock(side_effect=AssertionError("contender must not invoke model"))
    monkeypatch.setattr(topics, "_generate", generate)

    async def winner():
        await asyncio.sleep(0.03)
        assert await store.complete_generation(**args, topic="Testing pause control")

    winning = asyncio.create_task(winner())
    result = await fetch()
    await winning
    assert result["unavailable"] is False
    assert result["topics"][0]["topic"] == "Testing pause control"
    generate.assert_not_awaited()


async def test_expired_publish_never_returns_or_caches_the_losing_title(durable_db, monkeypatch):
    calls = 0

    async def read(**kwargs):
        nonlocal calls
        calls += 1
        return {} if calls == 1 else {"user-anchor": "The durable winning title"}

    monkeypatch.setattr(store, "load_topic_titles", read)
    monkeypatch.setattr(store, "claim_generation", AsyncMock(return_value=True))
    monkeypatch.setattr(store, "complete_generation", AsyncMock(return_value=False))
    release = AsyncMock()
    monkeypatch.setattr(store, "release_generation", release)
    monkeypatch.setattr(topics, "_generate", AsyncMock(return_value={"user-anchor": "The stale losing title"}))
    result = await fetch()
    assert result["topics"][0]["topic"] == "The durable winning title"
    assert all(value[0] != "The stale losing title" for value in topics._REMOTE_CACHE.values())
    release.assert_awaited_once()


async def test_model_cancellation_releases_committed_lease(durable_db, monkeypatch):
    entered = asyncio.Event()

    async def generate(*_args):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(topics, "_generate", generate)
    task = asyncio.create_task(fetch())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with durable_db() as db:
        row = (await db.execute(select(ChatTopicTitle))).scalar_one()
        assert row.lease_token is None
    assert await store.claim_generation(**lease_identity(lease_token=uuid.uuid4().hex))


async def test_unavailable_store_uses_bounded_process_cache(durable_db, monkeypatch):
    monkeypatch.setattr(store, "load_topic_titles", AsyncMock(side_effect=store.TopicStoreUnavailable()))
    claim = AsyncMock(side_effect=AssertionError("do not claim after failed cache read"))
    monkeypatch.setattr(store, "claim_generation", claim)
    generate = AsyncMock(return_value={"user-anchor": "Testing pause control"})
    monkeypatch.setattr(topics, "_generate", generate)
    first, second = await fetch(), await fetch()
    assert first == second and first["unavailable"] is False
    assert generate.await_count == 1
    claim.assert_not_awaited()


async def test_publish_cache_outage_preserves_valid_local_result_and_releases(durable_db, monkeypatch):
    monkeypatch.setattr(store, "complete_generation", AsyncMock(side_effect=store.TopicStoreUnavailable()))
    monkeypatch.setattr(topics, "_generate", AsyncMock(return_value={"user-anchor": "Testing pause control"}))
    result = await fetch()
    assert result["unavailable"] is False
    async with durable_db() as db:
        row = (await db.execute(select(ChatTopicTitle))).scalar_one()
        assert row.topic is None and row.lease_token is None


async def test_generation_rate_limit_releases_claim_without_model_work(durable_db, monkeypatch):
    for _ in range(6):
        assert topics._allow_generation(USER)
    generate = AsyncMock(side_effect=AssertionError("user generation budget exhausted"))
    monkeypatch.setattr(topics, "_generate", generate)
    result = await fetch()
    assert result["unavailable"] is True
    generate.assert_not_awaited()
    async with durable_db() as db:
        row = (await db.execute(select(ChatTopicTitle))).scalar_one()
        assert row.lease_token is None
