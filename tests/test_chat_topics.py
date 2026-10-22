"""Owned, day-scoped, tool-free topic generation and durable cache behavior."""

import asyncio
import json
from datetime import date
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.services import chat_topics as topics


@pytest.fixture(autouse=True)
def _reset_database():
    # These tests own a tiny DB; don't run the billing suite's global fixture.
    topics._RECENT.clear()
    topics._RETRY_AFTER.clear()
    topics._USER_LOCKS.clear()
    topics._HISTORY_LOCKS.clear()
    topics._REMOTE_CACHE.clear()
    topics._HISTORY_CACHE.clear()
    topics._HISTORY_READS.clear()
    topics._HISTORY_BYTES = 0


@pytest.fixture
async def sessions(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/topics.sqlite")
    async with engine.begin() as db:
        for sql in (
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, user_id TEXT, channel TEXT, day_chat_id TEXT)",
            "CREATE TABLE day_chats (id TEXT PRIMARY KEY, user_id TEXT, local_date DATE)",
            "CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT, day_chat_id TEXT, channel TEXT, origin TEXT, role TEXT, content TEXT, created_at DATETIME, occurred_at DATETIME, metadata_json TEXT, client_msg_id TEXT, source TEXT)",
            "INSERT INTO conversations VALUES ('conv','owner','web','old'), ('other','stranger','web','today'), ('hidden','owner','automation','today')",
            "INSERT INTO day_chats VALUES ('today','owner','2026-10-22'), ('old','owner','2026-10-21'), ('foreign','stranger','2026-10-22')",
        ):
            await db.execute(text(sql))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def seed(db, id, content="Discuss geometric shapes", role="user", conv="conv", day="today", at="2026-10-22 14:00:00", origin=None, meta=None):
    at = at + ".000000" if "." not in at else at  # SQLAlchemy's SQLite DateTime representation.
    await db.execute(text("INSERT INTO messages VALUES (:id,:conv,:day,NULL,:origin,:role,:content,:at,NULL,:meta,NULL,NULL)"),
                     dict(id=id, conv=conv, day=day, origin=origin, role=role, content=content, at=at, meta=meta))
    await db.commit()


async def fetch(db, ids, user="owner"):
    return await topics.get_chat_topics(db, user_id=user, day=date(2026, 10, 22),
                                       message_ids=ids, tz_name="America/Toronto")


async def test_owned_day_turns_only_and_context_excludes_admin(sessions, monkeypatch):
    async with sessions() as db:
        await seed(db, "first", content="Explain triangles", at="2026-10-22 13:00:00")
        await seed(db, "admin", content="INTERNAL OPERATOR TEXT", origin="admin", at="2026-10-22 13:30:00")
        await seed(db, "anchor", content="And squares?")
        await seed(db, "answer", content="Squares have four equal sides", role="assistant", at="2026-10-22 14:01:00")
        await seed(db, "wrong-owner", conv="other")
        await seed(db, "wrong-day", day="old")
        await seed(db, "hidden", conv="hidden")
        await seed(db, "wrong-day-owner", day="foreign")
        await seed(db, "legacy-today", day=None, at="2026-10-23 03:30:00")
        await seed(db, "legacy-tomorrow", day=None, at="2026-10-23 04:30:00")
        calls = []

        async def model(**kwargs):
            assert not db.in_transaction(), "Never hold a DB transaction across model work"
            calls.append(kwargs)
            turns = json.loads(kwargs["messages"][0]["content"])
            assert {turn["message_id"] for turn in turns} == {"anchor", "legacy-today"}
            anchor = next(turn for turn in turns if turn["message_id"] == "anchor")
            assert anchor["context_before"] == [{"role": "user", "content": "Explain triangles"}]
            assert "Squares have four equal sides" not in kwargs["messages"][0]["content"]
            assert "INTERNAL OPERATOR TEXT" not in kwargs["messages"][0]["content"]
            assert kwargs["operation_type"] == "system.chat_topic"
            assert "tools" not in kwargs
            return json.dumps({"topics": [{"message_id": t["message_id"], "topic": "Understanding squares and their properties"} for t in turns]})

        monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
        result = await fetch(db, ["anchor", "wrong-owner", "wrong-day", "hidden", "wrong-day-owner", "legacy-today", "legacy-tomorrow", "missing"])
        assert [row["message_id"] for row in result["topics"]] == ["anchor", "legacy-today"]
        assert len(calls) == 1


async def test_cache_preserves_concurrent_metadata_and_invalidates_edited_request(sessions, monkeypatch):
    async with sessions() as db:
        await seed(db, "anchor", meta=json.dumps({"attachments": ["original"]}))
        calls = []

        async def model(**kwargs):
            calls.append(kwargs)
            async with sessions() as concurrent:
                await concurrent.execute(text("UPDATE messages SET metadata_json=:meta WHERE id='anchor'"),
                                         {"meta": json.dumps({"attachments": ["original"], "tool_events": ["new"]})})
                await concurrent.commit()
            return '{"topics":[{"message_id":"anchor","topic":"Reviewing geometric shapes and properties"}]}'

        monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
        first = await fetch(db, ["anchor"])
        assert first["unavailable"] is False
        assert await fetch(db, ["anchor"]) == first
        assert len(calls) == 1
        meta = json.loads((await db.execute(text("SELECT metadata_json FROM messages WHERE id='anchor'"))).scalar())
        assert meta["attachments"] == ["original"]
        assert meta["tool_events"] == ["new"]
        await db.execute(text("UPDATE messages SET content='Plan tomorrow' WHERE id='anchor'"))
        await db.commit()
        await fetch(db, ["anchor"])
        assert len(calls) == 2


async def test_overlapping_requests_share_cached_generation(sessions, monkeypatch):
    async with sessions() as db:
        await seed(db, "anchor")
    count = 0

    async def model(**kwargs):
        nonlocal count
        count += 1
        await asyncio.sleep(0.02)
        return '{"topics":[{"message_id":"anchor","topic":"Reviewing geometric shapes and properties"}]}'

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    async with sessions() as left, sessions() as right:
        first, second = await asyncio.gather(fetch(left, ["anchor"]), fetch(right, ["anchor"]))
    assert count == 1
    assert first == second


async def test_failed_generation_backoff_and_untrusted_model_output(sessions, monkeypatch):
    async with sessions() as db:
        await seed(db, "anchor")
        count = 0

        async def model(**kwargs):
            nonlocal count
            count += 1
            return '{"topics":[{"message_id":"foreign","topic":"Stolen topic"},{"message_id":"anchor","topic":"<script>bad</script>"}]}'

        monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
        assert await fetch(db, ["anchor"]) == {"topics": [], "unavailable": True}
        assert await fetch(db, ["anchor"]) == {"topics": [], "unavailable": True}
        assert count == 1
        assert (await db.execute(text("SELECT metadata_json FROM messages WHERE id='anchor'"))).scalar() is None


async def test_model_failure_and_rate_limit_leave_history_intact(sessions, monkeypatch):
    async with sessions() as db:
        await seed(db, "anchor", meta="legacy non-json metadata")

        async def model(**kwargs):
            return None

        monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
        result = await fetch(db, ["anchor"])
        assert result == {"topics": [], "unavailable": True}
        assert (await db.execute(text("SELECT metadata_json FROM messages WHERE id='anchor'"))).scalar() == "legacy non-json metadata"
        assert all(topics._allow_generation("another") for _ in range(6))
        assert not topics._allow_generation("another")


@pytest.mark.parametrize("raw", ["not JSON", '{"topics":null}', '{"topics":[{"message_id":[],"topic":"invalid"}]}',
                                  'Here is JSON:\n```json\n{"topics":[]}\n```',
                                  '```json\n{"topics":[]}\n```\nExtra text'])
async def test_malformed_completions_are_not_saved(sessions, monkeypatch, raw):
    async with sessions() as db:
        await seed(db, "anchor")

        async def model(**kwargs):
            return raw

        monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
        assert await fetch(db, ["anchor"]) == {"topics": [], "unavailable": True}


async def test_endpoint_auth_input_bounds_and_owned_history_read(sessions, monkeypatch):
    from app.api import day_chats
    from app.api.auth import get_current_user
    from app.db import get_db

    app = FastAPI()
    app.include_router(day_chats.router, prefix="/api")

    async def database():
        async with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = database
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/api/day-chats/2026-10-22/topics", json={"message_ids": ["anchor"]})).status_code == 401
        assert (await client.post("/api/day-chats/2026-10-22/topics", json={"message_ids": ["anchor"], "cache_only": True})).status_code == 401
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="owner", timezone="America/Toronto")
        assert (await client.post("/api/day-chats/2026-10-22/topics", json={"message_ids": ["x"] * 21})).status_code == 422
        assert (await client.post("/api/day-chats/bad/topics", json={"message_ids": ["x"]})).status_code == 400
        assert (await client.post("/api/day-chats/2026-10-22/topics", json={"message_ids": ["x" * 51]})).status_code == 400

        async def proxy_info(*args):
            return "https://tenant.test", "test-agent-key"

        calls = []

        async def proxy_history(url, key, path, params=None, **kwargs):
            calls.append((url, key, path, params))
            return [{"id": "anchor", "role": "assistant", "content": "Weekly recap of writing projects", "conversation_id": "routine", "channel": "web"}]

        async def model(**kwargs):
            turns = json.loads(kwargs["messages"][0]["content"])
            assert turns[0]["source_role"] == "assistant"
            assert turns[0]["source_text"] == "Weekly recap of writing projects"
            assert turns[0]["context_before"] == []
            return '{"topics":[{"message_id":"anchor","topic":"Understanding geometric shapes"}]}'

        monkeypatch.setattr("app.api.tenant_proxy.serving_locally", lambda: False)
        monkeypatch.setattr(day_chats, "_get_agent_proxy_info", proxy_info)
        monkeypatch.setattr(day_chats, "_proxy_day_chats", proxy_history)
        monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
        response = await client.post("/api/day-chats/2026-10-22/topics", json={"message_ids": ["anchor"]})
        assert response.status_code == 200
        assert response.json()["topics"][0]["message_id"] == "anchor"
        assert calls == [("https://tenant.test", "test-agent-key", "2026-10-22/messages", {"limit": 2000})]
        cached = await client.post("/api/day-chats/2026-10-22/topics", json={"message_ids": ["anchor", "missing"], "cache_only": True})
        assert cached.status_code == 200
        assert cached.json()["cache_only"] is True and cached.json()["unavailable"] is False
        assert cached.json()["topics"] == response.json()["topics"]
        assert len(calls) == 1
        assert len(topics._RECENT["owner"]) == 1
        assert (await client.post("/api/day-chats/2026-10-22/topics", json={"message_ids": ["anchor"], "cache_only": "yes"})).status_code == 422


def remote_row(id, content="Explain geometric shapes", role="user", **kwargs):
    return {"id": id, "content": content, "role": role, "conversation_id": "conv", "channel": "web", **kwargs}


async def remote(ids, messages, user="owner", day=date(2026, 10, 22)):
    return await topics.get_remote_chat_topics(user_id=user, day=day, message_ids=ids, messages=messages)


async def test_remote_response_is_id_scoped_and_excludes_admin_and_hidden(monkeypatch):
    messages = [remote_row("prior", "Discuss triangles"), remote_row("admin", "operator secret", channel="admin"),
                remote_row("notice", "notice secret", admin_notice={"body": "notice"}),
                remote_row("hidden", "hidden secret", channel="automation"),
                remote_row("anchor", "And squares?"), remote_row("answer", "Squares have four equal sides", role="assistant")]
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        turns = json.loads(kwargs["messages"][0]["content"])
        assert [row["message_id"] for row in turns] == ["anchor"]
        assert turns[0]["context_before"] == [{"role": "user", "content": "Discuss triangles"}]
        assert "Squares have four equal sides" not in kwargs["messages"][0]["content"]
        assert "secret" not in kwargs["messages"][0]["content"]
        return '{"topics":[{"message_id":"anchor","topic":"Understanding squares and their properties"},{"message_id":"foreign","topic":"Ignore foreign IDs"}]}'

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    result = await remote(["anchor", "foreign", "admin", "notice", "hidden"], messages)
    assert result["topics"] == [{"message_id": "anchor", "topic": "Understanding squares and their properties", "canonical_message_id": "anchor", "alias_message_ids": ["anchor", "answer"]}]
    assert result["unavailable"] is True
    assert len(calls) == 1


async def test_remote_cache_scopes_account_day_and_content(monkeypatch):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '{"topics":[{"message_id":"anchor","topic":"Understanding geometric shapes"}]}'

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    rows = [remote_row("anchor")]
    result = await remote(["anchor"], rows)
    assert result == await remote(["anchor"], rows)
    assert len(calls) == 1
    assert (await remote(["anchor"], []))["topics"] == [], "Cache cannot serve an ID absent from the owned-day read"
    await remote(["anchor"], rows, user="another")
    await remote(["anchor"], rows, day=date(2026, 10, 21))
    await remote(["anchor"], [remote_row("anchor", "A changed request")])
    assert len(calls) == 4


async def test_remote_overlap_rate_bound_and_cache_limit(monkeypatch):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0.01)
        turns = json.loads(kwargs["messages"][0]["content"])
        return json.dumps({"topics": [{"message_id": row["message_id"], "topic": "Understanding geometric shapes"} for row in turns]})

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    monkeypatch.setattr(topics, "_REMOTE_CACHE_MAX", 2)
    rows = [remote_row("anchor")]
    first, second = await asyncio.gather(remote(["anchor"], rows), remote(["anchor"], rows))
    assert first == second and len(calls) == 1
    for i in range(6):
        await remote([str(i)], [remote_row(str(i))])
    assert len(calls) == 6
    assert len(topics._REMOTE_CACHE) == 2
    assert (await remote(["last"], [remote_row("last")]))["unavailable"] is True


@pytest.mark.parametrize("payload", [{"error": "unavailable"}, None, [None, {"id": "anchor", "role": "user", "content": []}]])
async def test_remote_malformed_history_never_reaches_model(monkeypatch, payload):
    async def model(**kwargs):
        raise AssertionError("No model call without an owned user message")

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    assert await remote(["anchor"], payload) == {"topics": [], "unavailable": True}


@pytest.mark.parametrize("fence", ["```json", "```"])
async def test_complete_json_fence_is_accepted(monkeypatch, fence):
    async def model(**kwargs):
        return fence + '\n{"topics":[{"message_id":"anchor","topic":"Understanding geometric shapes"}]}\n```'

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    assert (await remote(["anchor"], [remote_row("anchor")]))["unavailable"] is False


async def test_remote_history_singleflight_cache_ttl_and_new_anchor(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(topics, "time", SimpleNamespace(monotonic=lambda: now[0]))
    reads, generations = [], []
    rows = [remote_row("anchor")]

    async def loader():
        reads.append(True)
        await asyncio.sleep(0.01)
        return list(rows)

    async def model(**kwargs):
        generations.append(True)
        turns = json.loads(kwargs["messages"][0]["content"])
        return json.dumps({"topics": [{"message_id": turn["message_id"], "topic": "Understanding geometric shapes"} for turn in turns]})

    async def fetch(ids=("anchor",), user="owner", day=date(2026, 10, 22)):
        return await topics.get_remote_chat_topics(user_id=user, day=day, message_ids=list(ids), load_messages=loader)

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    first, second = await asyncio.gather(fetch(), fetch())
    assert first == second and not first["unavailable"]
    assert len(reads) == len(generations) == 1
    assert await fetch() == first
    assert len(reads) == len(generations) == 1
    rows.append(remote_row("new"))
    assert (await fetch(("new",)))["unavailable"] is True
    assert len(reads) == 1
    now[0] += 6
    assert (await fetch(("new",)))["unavailable"] is False
    assert len(reads) == 2
    now[0] += 31
    assert await fetch() == first
    assert len(reads) == 3 and len(generations) == 2
    await fetch(user="another")
    await fetch(day=date(2026, 10, 21))
    assert len(reads) == 5 and len(generations) == 4
    assert not topics._USER_LOCKS, "Weak locks must not retain account IDs"


async def test_remote_history_read_budget_does_not_serve_expired_context(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(topics, "time", SimpleNamespace(monotonic=lambda: now[0]))
    reads = []

    async def loader():
        reads.append(True)
        return [remote_row("anchor")]

    for i in range(30):
        await topics._remote_history("owner", date(2026, 10, i + 1), ["anchor"], loader)
    assert len(reads) == 30
    now[0] += 31
    with pytest.raises(topics.TopicHistoryUnavailable):
        await topics._remote_history("owner", date(2026, 10, 1), ["anchor"], loader)
    assert len(reads) == 30
    now[0] += 30
    assert await topics._remote_history("owner", date(2026, 10, 1), ["anchor"], loader)
    assert len(reads) == 31


async def test_remote_history_projection_and_byte_entry_caps(monkeypatch):
    monkeypatch.setattr(topics, "_HISTORY_MAX_BYTES", 4000)
    monkeypatch.setattr(topics, "_HISTORY_MAX_ENTRIES", 2)
    content = "x" * 100000

    async def loader():
        return [remote_row("anchor", content, attachments=["secret"] * 10000, tool_events=["secret"]),
                remote_row("hidden", "hidden secret", channel="automation"),
                remote_row("answer", content, role="assistant")]

    for i in range(4):
        rows = await topics._remote_history(str(i), date(2026, 10, 22), ["anchor"], loader)
        assert [row["id"] for row in rows] == ["anchor", "answer"]
        assert len(rows[0]["content"]) == 1600 and len(rows[1]["content"]) == 1600
        assert rows[0]["_topic_source_hash"] == topics._fingerprint(content)
        assert "secret" not in json.dumps(rows)
        assert topics._HISTORY_BYTES <= 4000 and len(topics._HISTORY_CACHE) <= 2
    assert topics._HISTORY_BYTES == sum(record[3] for record in topics._HISTORY_CACHE.values())
    assert len(topics._HISTORY_CACHE) == 1


async def test_remote_history_cancelled_loader_releases_account_lock(monkeypatch):
    entered = asyncio.Event()

    async def loader():
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(topics.get_remote_chat_topics(
        user_id="owner", day=date(2026, 10, 22), message_ids=["anchor"], load_messages=loader))
    await entered.wait()
    lock = topics._USER_LOCKS["owner"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not lock.locked() and not topics._HISTORY_CACHE


async def test_remote_assistant_titles_own_content_and_same_conversation_context(monkeypatch):
    rows = [remote_row("unrelated", "Read Rick Rubin", conversation_id="older-thread"),
            remote_row("request", "Summarize this week's writing", conversation_id="writing"),
            remote_row("admin", "operator secret", channel="admin", conversation_id="writing"),
            remote_row("recap", "Weekly recap: finished a draft and outlined an essay", role="assistant", conversation_id="writing"),
            remote_row("standalone", "Weekly recap of saved recipes", role="assistant", conversation_id="routine"),
            remote_row("missing-conversation", "Weekly recap of garden tasks", role="assistant", conversation_id=None),
            remote_row("hidden", "secret routine", role="assistant", channel="automation"),
            remote_row("notice", "secret notice", role="assistant", admin_notice={"text": "private"})]
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        turns = json.loads(kwargs["messages"][0]["content"])
        by_id = {turn["message_id"]: turn for turn in turns}
        assert set(by_id) == {"t0", "t1", "t2"}
        assert by_id["t0"]["source_role"] == "user"
        assert by_id["t0"]["source_text"] == "Summarize this week's writing"
        assert by_id["t1"]["source_role"] == "assistant"
        assert by_id["t1"]["context_before"] == by_id["t2"]["context_before"] == []
        assert "Rick Rubin" not in kwargs["messages"][0]["content"]
        assert "secret" not in kwargs["messages"][0]["content"]
        return json.dumps({"topics": [{"message_id": turn["message_id"], "topic": "Reviewing the week's completed projects"} for turn in turns]})

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    result = await remote(["recap", "standalone", "missing-conversation", "hidden", "notice", "foreign"], rows)
    assert [item["message_id"] for item in result["topics"]] == ["recap", "standalone", "missing-conversation"]
    assert result["unavailable"] is True and len(calls) == 1


async def test_remote_fingerprint_tracks_role_full_source_and_selected_context(monkeypatch):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return json.dumps({"topics": [{"message_id": turn["message_id"], "topic": "Reviewing the weekly writing recap"} for turn in json.loads(kwargs["messages"][0]["content"])]})

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    rows = [remote_row("unrelated", "Unrelated text", conversation_id="another"),
            remote_row("request", "x" * 1000), remote_row("recap", "y" * 2000, role="assistant")]
    first = await remote(["recap"], rows)
    assert first == await remote(["recap"], rows) and len(calls) == 1
    rows[0]["content"] = "Changed unrelated text"
    await remote(["recap"], rows)
    assert len(calls) == 1
    rows[1]["content"] += "changed beyond context prompt cap"
    await remote(["recap"], rows)
    assert len(calls) == 2
    rows[2]["content"] += "changed beyond source prompt cap"
    await remote(["recap"], rows)
    assert len(calls) == 2, "Assistant streaming does not change the user intent fingerprint"
    rows[2]["role"] = "user"
    await remote(["recap"], rows)
    assert len(calls) == 3


async def test_local_assistant_anchor_ownership_day_context_and_cache(sessions, monkeypatch):
    async with sessions() as db:
        await db.execute(text("INSERT INTO conversations VALUES ('routine','owner','web','today'), ('unrelated','owner','web','today')"))
        await db.commit()
        await seed(db, "old-request", content="Read Rick Rubin", conv="routine", day="old", at="2026-10-21 10:00:00")
        await seed(db, "other-request", content="Unrelated topic", conv="unrelated", at="2026-10-22 13:00:00")
        await seed(db, "recap", content="Weekly recap of saved recipes", role="assistant", conv="routine")
        await seed(db, "request", content="Summarize this week's writing", at="2026-10-22 13:00:00")
        await seed(db, "answer", content="Weekly writing recap", role="assistant")
        await seed(db, "wrong-owner", role="assistant", conv="other")
        await seed(db, "wrong-day", role="assistant", day="old")
        await seed(db, "hidden", role="assistant", conv="hidden")
        await seed(db, "admin", role="assistant", origin="admin")
        calls = []

        async def model(**kwargs):
            assert not db.in_transaction()
            turns = json.loads(kwargs["messages"][0]["content"])
            calls.append(turns)
            by_id = {turn["message_id"]: turn for turn in turns}
            if "recap" in by_id:
                assert by_id["recap"]["context_before"] == []
                assert by_id["recap"]["source_text"] == "Weekly recap of saved recipes"
            if "request" in by_id:
                assert by_id["request"]["source_text"].startswith("Summarize")
                assert by_id["request"]["source_role"] == "user"
            assert "Rick Rubin" not in json.dumps(turns) and "Unrelated topic" not in json.dumps(turns)
            return json.dumps({"topics": [{"message_id": turn["message_id"], "topic": "Reviewing the weekly recap"} for turn in turns]})

        monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
        result = await fetch(db, ["recap", "answer", "wrong-owner", "wrong-day", "hidden", "admin"])
        assert [row["message_id"] for row in result["topics"]] == ["recap", "answer"]
        assert result["unavailable"] is True
        assert (await fetch(db, ["recap", "answer"]))["unavailable"] is False
        assert len(calls) == 1
        await db.execute(text("UPDATE messages SET content='Summarize the revised writing plan' WHERE id='request'"))
        await db.commit()
        await fetch(db, ["recap", "answer"])
        assert len(calls) == 2 and [turn["message_id"] for turn in calls[-1]] == ["request"]


async def test_local_cache_only_bypasses_generation_lock_and_writes(sessions, monkeypatch):
    async with sessions() as db:
        await seed(db, 'cached', content='Plan a product launch')
        await seed(db, 'missing', content='Find a restaurant', at='2026-10-22 15:00:00')
        await seed(db, 'foreign', conv='other')
        await seed(db, 'hidden', conv='hidden')
        rows = [{'id': 'cached', 'role': 'user', 'content': 'Plan a product launch', 'conversation_id': 'conv'}]
        fingerprint = topics._canonical_groups(rows)[0]['cached']['fingerprint']
        metadata = json.dumps({'attachments': ['keep'], 'chat_topic': {'version': topics._VERSION, 'source': fingerprint, 'topic': 'Plan a product launch'}})
        await db.execute(text("UPDATE messages SET metadata_json=:metadata WHERE id='cached'"), {'metadata': metadata})
        await db.commit()
        lock = asyncio.Lock()
        await lock.acquire()
        topics._USER_LOCKS['owner'] = lock
        async def forbidden(*args, **kwargs):
            raise AssertionError('Cache-only must not generate')
        monkeypatch.setattr(topics, '_generate', forbidden)
        try:
            result = await asyncio.wait_for(topics.get_chat_topics(db, user_id='owner', day=date(2026, 10, 22),
                message_ids=['cached', 'missing', 'foreign', 'hidden'], tz_name='America/Toronto', cache_only=True), .5)
        finally:
            lock.release()
        assert result['cache_only'] and not result['unavailable']
        assert [item['message_id'] for item in result['topics']] == ['cached']
        assert not topics._RECENT and not topics._RETRY_AFTER
        assert (await db.execute(text("SELECT metadata_json FROM messages WHERE id='cached'"))).scalar() == metadata
        assert (await db.execute(text("SELECT metadata_json FROM messages WHERE id='missing'"))).scalar() is None


async def test_local_cache_only_database_outage_is_not_normal_miss(monkeypatch):
    from unittest.mock import AsyncMock
    from sqlalchemy.exc import OperationalError
    db = SimpleNamespace(commit=AsyncMock(side_effect=OperationalError('read', {}, Exception('unavailable'))), rollback=AsyncMock())
    result = await topics.get_chat_topics(db, user_id='owner', day=date(2026, 10, 22), message_ids=['anchor'], tz_name=None, cache_only=True)
    assert result == {'topics': [], 'unavailable': True, 'cache_only': True}
    db.rollback.assert_awaited_once()


async def test_cache_only_history_singleflight_is_independent_of_generation(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services import chat_topic_store as store
    reads = []
    entered, finish = asyncio.Event(), asyncio.Event()
    async def loader():
        reads.append(True)
        entered.set()
        await finish.wait()
        return [remote_row('anchor')]
    monkeypatch.setattr(store, 'load_topic_titles', AsyncMock(return_value={}))
    lock = asyncio.Lock()
    await lock.acquire()
    topics._USER_LOCKS['owner'] = lock
    async def fetch(user='owner', day=date(2026, 10, 22)):
        return await topics.get_remote_chat_topics(user_id=user, day=day, message_ids=['anchor'], load_messages=loader, cache_only=True)
    tasks = [asyncio.create_task(fetch()) for _ in range(2)]
    await entered.wait()
    finish.set()
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), .5)
    finally:
        lock.release()
    assert results == [{'topics': [], 'unavailable': False, 'cache_only': True}] * 2
    assert len(reads) == 1
    await fetch(user='other')
    await fetch(day=date(2026, 10, 21))
    assert len(reads) == 3
    assert not topics._HISTORY_LOCKS and not topics._RECENT and not topics._RETRY_AFTER


@pytest.mark.parametrize('failure', ['exception', 'malformed', 'timeout', 'budget'])
async def test_cache_only_history_failure_is_not_normal_miss(monkeypatch, failure):
    from unittest.mock import AsyncMock
    from app.services import chat_topic_store as store
    async def loader():
        if failure == 'exception':
            raise RuntimeError('tenant unavailable')
        if failure == 'malformed':
            return {'error': 'unavailable'}
        if failure == 'timeout':
            await asyncio.Event().wait()
        return [remote_row('anchor')]
    if failure == 'timeout':
        monkeypatch.setattr(topics, '_HISTORY_READ_TIMEOUT', .02)
    if failure == 'budget':
        topics._HISTORY_READS['owner'] = [topics.time.monotonic()] * 30
    load = AsyncMock()
    monkeypatch.setattr(store, 'load_topic_titles', load)
    result = await topics.get_remote_chat_topics(user_id='owner', day=date(2026, 10, 22), message_ids=['anchor'], load_messages=loader, cache_only=True)
    assert result == {'topics': [], 'unavailable': True, 'cache_only': True}
    load.assert_not_awaited()
    assert not topics._HISTORY_LOCKS and not topics._RECENT and not topics._RETRY_AFTER
