"""The optional shared cache must not multiply request latency by batch size."""

import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import chat_topic_store as store
from app.services import chat_topics as topics


@pytest.fixture(autouse=True)
def _reset_database():
    topics._REMOTE_CACHE.clear()
    topics._RECENT.clear()
    topics._RETRY_AFTER.clear()


def groups(count=20):
    rows = [{"id": f"task{i}", "role": "user", "content": f"Work on project {i}",
             "conversation_id": f"conversation{i}"} for i in range(count)]
    return topics._canonical_groups(rows)[0]


async def test_store_outage_opens_circuit_before_per_item_claims(monkeypatch):
    load = AsyncMock(side_effect=store.TopicStoreUnavailable("unavailable"))
    claim = AsyncMock()
    complete = AsyncMock()
    monkeypatch.setattr(store, "load_topic_titles", load)
    monkeypatch.setattr(store, "claim_generation", claim)
    monkeypatch.setattr(store, "complete_generation", complete)
    monkeypatch.setattr(topics, "_generate", AsyncMock(return_value={f"task{i}": "A useful task title" for i in range(20)}))
    result, busy = await topics._resolve_remote_titles("owner", date(2026, 10, 22), groups())
    assert len(result) == 20 and not busy
    load.assert_awaited_once()
    claim.assert_not_awaited()
    complete.assert_not_awaited()


async def test_twenty_publishes_share_one_store_budget(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(topics, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(store, "load_topic_titles", AsyncMock(return_value={}))
    monkeypatch.setattr(store, "claim_generation", AsyncMock(return_value=True))
    publishes = []

    async def publish(**kwargs):
        publishes.append(kwargs["canonical_id"])
        now[0] += 2
        return True

    release = AsyncMock()
    monkeypatch.setattr(store, "complete_generation", publish)
    monkeypatch.setattr(store, "release_generation", release)
    monkeypatch.setattr(topics, "_generate", AsyncMock(return_value={f"task{i}": "A useful task title" for i in range(20)}))
    result, busy = await topics._resolve_remote_titles("owner", date(2026, 10, 22), groups())
    assert len(result) == 20 and not busy
    assert len(publishes) == 3  # The third operation exhausts the shared 5s budget.
    release.assert_not_awaited()  # Remaining leases expire; no unbounded cleanup ladder.


async def test_cancellation_during_claim_batch_releases_committed_claim(monkeypatch):
    committed = asyncio.Event()
    pending = asyncio.Event()
    monkeypatch.setattr(store, "load_topic_titles", AsyncMock(return_value={}))

    async def claim(**kwargs):
        if kwargs["canonical_id"] == "task0":
            committed.set()
            return True
        pending.set()
        await asyncio.Event().wait()

    release = AsyncMock()
    generate = AsyncMock()
    monkeypatch.setattr(store, "claim_generation", claim)
    monkeypatch.setattr(store, "release_generation", release)
    monkeypatch.setattr(topics, "_generate", generate)
    task = asyncio.create_task(topics._resolve_remote_titles("owner", date(2026, 10, 22), groups(2)))
    await committed.wait()
    await pending.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.assert_awaited_once()
    assert release.await_args.kwargs["canonical_id"] == "task0"
    generate.assert_not_awaited()
