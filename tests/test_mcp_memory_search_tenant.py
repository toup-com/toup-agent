"""MCP memory tools must read the TENANT, and must not answer from the
platform's stale table (v3 §2.1.6, §4).

`memory_files` is AGENT_ONLY, exactly as `memories` was. `memory_create` /
`update` / `delete` / `list` were routed to the tenant (#377, a1d65a61);
`memory_search` was left behind on the platform session. That does not
error — it silently matches whatever monolith-era rows are still sitting in
the platform's table, so an MCP client could write into the tenant and then
fail to find it, because the write and the read hit different databases.

Measured on the live canary, 2026-08-05:

    platform DB, user 533354ce : 2 rows, both from 2026-07-28, one carrying
                                 category='context' (not in the taxonomy)
    tenant DB,   user 533354ce : 19 real memories, none of them those two

**v3 changes the answer to the fall-back question.** Round 8's rule was "a
READ may fall back to the platform session, a WRITE may not", because rows
were mirrored platform-side so the fall-back could at least return
something true-ish. Files are not mirrored: there is nothing on the platform
to fall back TO, and synthesising a file view from nothing is a lie rather
than a degradation (the same reasoning that made the REST file reads answer
503 instead of a virtual view). So an unreachable tenant is now an explicit
error on every leg, and the anti-vacuity control below asserts THAT.

These are behavioural tests: they call the tool and assert on what it
RETURNS. A source-text assertion cannot tell a proxied read from an
unproxied one that happens to mention the helper's name.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. pytest tests/test_mcp_memory_search_tenant.py
"""
from __future__ import annotations

import pytest

TENANT_HIT = {
    "slug": "topics/places",
    "title": "Places",
    "snippet": "works out of a co-working space on Queen West",
}

PLATFORM_ROW_CONTENT = "User's locker codeword is kestrel-5ca0."


class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _install(monkeypatch, *, proxy_result):
    """Point the tools at a fake tenant, and leave a platform session that
    would be WRONG to use so "which database answered" stays observable."""
    import app.mcp_server as m

    monkeypatch.setattr(m, "_get_user_id", lambda: "533354ce-40c5-41b6-81a8-e87e33a08a24")
    monkeypatch.setattr(m, "async_session_maker", lambda: _FakeDB())

    calls: list[dict] = []

    async def fake_proxy(db, user_id, path, params=None):
        calls.append({"user_id": user_id, "params": params or {}, "path": path})
        return proxy_result

    monkeypatch.setattr(m, "_proxy_files_read", fake_proxy)

    class _PlatformSvc:
        def __init__(self, db):
            pass

        async def search_memories(self, user_id, request):  # pragma: no cover
            raise AssertionError(
                "an MCP memory tool reached the platform session — there are "
                "no memory files there to read"
            )

    monkeypatch.setattr(m, "MemoryService", _PlatformSvc)
    return calls


@pytest.mark.asyncio
async def test_search_returns_the_tenants_files(monkeypatch):
    """The whole bug in one assertion: the tenant's file must come back.

    MUTATION: delete the `_proxy_files_read(...)` call from `memory_search`
    -> `data` is unbound / the platform stub raises, and this goes red.
    """
    import app.mcp_server as m

    calls = _install(monkeypatch, proxy_result={"results": [TENANT_HIT]})

    out = await m.memory_search.fn(query="where do I work", limit=5)

    assert out["results"] == [TENANT_HIT]
    assert out["total"] == 1
    assert calls and calls[0]["path"] == "search", (
        f"tenant was not asked for /search: {calls}"
    )
    assert calls[0]["params"].get("q") == "where do I work", (
        "the v3 search route's query parameter is `q`, not `query` — a "
        "mismatched name is an empty result set, not an error"
    )


@pytest.mark.asyncio
async def test_a_result_names_the_FILE_it_came_from(monkeypatch):
    """The unit is a file. A snippet with no slug is unopenable: the model
    cannot follow it up, which is what `memory_read_file` exists for."""
    import app.mcp_server as m

    _install(monkeypatch, proxy_result={"results": [TENANT_HIT]})
    out = await m.memory_search.fn(query="q", limit=5)
    assert out["results"][0]["slug"] == "topics/places"
    assert "id" not in out["results"][0], "there are no memory ids in v3"


@pytest.mark.asyncio
async def test_memory_files_lists_and_reads_through_the_tenant(monkeypatch):
    import app.mcp_server as m

    listing = {"sections": [{"section": "topics", "files": [{"slug": "topics/places"}]}]}
    calls = _install(monkeypatch, proxy_result=listing)
    assert await m.memory_files.fn() == listing
    assert calls[0]["path"] == "files"

    calls.clear()
    await m.memory_files.fn(slug="topics/places")
    assert calls[0]["path"] == "files/topics/places"


@pytest.mark.asyncio
async def test_an_unreachable_tenant_is_an_ERROR_not_a_platform_answer(monkeypatch):
    """Anti-vacuity control, and the v3 reversal of the read asymmetry.

    Round 8 let a READ fall back to the platform session. There is nothing
    there now — `memory_files` is AGENT_ONLY with no mirror — so a fall-back
    could only ever return an empty list, and "no memory files yet" for a
    user with a full library reads as data loss. It says so instead.

    The platform stub above RAISES, so if a fall-back ever came back this
    test fails loudly rather than quietly asserting the wrong thing.
    """
    import app.mcp_server as m

    _install(monkeypatch, proxy_result=None)  # tenant unreachable

    search = await m.memory_search.fn(query="q", limit=5)
    assert search["results"] == [] and "reachable" in search["error"]

    files = await m.memory_files.fn()
    assert "reachable" in files["error"]


@pytest.mark.asyncio
async def test_the_write_refuses_without_a_tenant(monkeypatch):
    """A WRITE never falls back. "Succeeding" against the wrong database is
    how a user comes to believe something was saved that was not."""
    import app.mcp_server as m

    monkeypatch.setattr(m, "_get_user_id", lambda: "533354ce-40c5-41b6-81a8-e87e33a08a24")
    monkeypatch.setattr(m, "async_session_maker", lambda: _FakeDB())

    async def _no_agent(user_id, db):
        return None

    import app.api.memories as memories_api

    monkeypatch.setattr(memories_api, "_get_agent_proxy_info", _no_agent)

    out = await m.memory_remember.fn(instruction="remember I like Googoosh")
    assert out["applied"] == 0 and "reachable" in out["error"]


@pytest.mark.asyncio
async def test_the_write_proxies_the_instruction_to_the_v3_route(monkeypatch):
    import app.mcp_server as m
    import app.api.memories as memories_api

    monkeypatch.setattr(m, "_get_user_id", lambda: "533354ce-40c5-41b6-81a8-e87e33a08a24")
    monkeypatch.setattr(m, "async_session_maker", lambda: _FakeDB())

    async def _agent(user_id, db):
        return ("http://agent", "key")

    seen: dict = {}

    async def _write(url, key, path, method, body=None, params=None):
        seen.update({"path": path, "method": method, "body": body})
        return {"applied": 1, "rejected": [], "note": "Saved 1 change."}

    monkeypatch.setattr(memories_api, "_get_agent_proxy_info", _agent)
    monkeypatch.setattr(memories_api, "_proxy_memories_write", _write)

    out = await m.memory_remember.fn(instruction="remember I like Googoosh")
    assert out["applied"] == 1
    assert seen["path"] == "instruct" and seen["method"] == "POST"
    assert seen["body"] == {"instruction": "remember I like Googoosh"}
