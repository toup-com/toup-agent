"""MCP `memory_search` must read the tenant, not the platform table.

`memories` is AGENT_ONLY. `memory_create` / `update` / `delete` / `list` were
routed to the tenant (#377, a1d65a61); `memory_search` was left behind on the
platform session. That does not error — it silently matches whatever
monolith-era rows are still sitting in the platform's `memories` table, so an
MCP client could create a memory into the tenant and then fail to find it,
because the write and the read hit different databases.

Measured on the live canary, 2026-08-05:

    platform DB, user 533354ce : 2 rows, both from 2026-07-28, one carrying
                                 category='context' (not in the taxonomy)
    tenant DB,   user 533354ce : 19 real memories, none of them those two

These are behavioural tests: they call the tool and assert on what it
RETURNS. A source-text assertion cannot tell a proxied read from an
unproxied one that happens to mention the helper's name.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. pytest tests/test_mcp_memory_search_tenant.py
"""
from __future__ import annotations

import pytest


TENANT_ROW = {
    "id": "11111111-1111-1111-1111-111111111111",
    "content": "USER works at co-working space",
    "summary": None,
    "category": "locations",
    "importance": 0.7,
    "similarity_score": 0.91,
    "explanation": None,
}

PLATFORM_ROW_CONTENT = "User's locker codeword is kestrel-5ca0."


class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _install(monkeypatch, *, proxy_result):
    """Point the tool at a fake tenant and a platform session that would be
    WRONG to use, so 'which database answered' is observable in the result."""
    import app.mcp_server as m

    monkeypatch.setattr(m, "_get_user_id", lambda: "533354ce-40c5-41b6-81a8-e87e33a08a24")
    monkeypatch.setattr(m, "async_session_maker", lambda: _FakeDB())

    calls: list[dict] = []

    async def fake_proxy(db, user_id, params=None, path=""):
        calls.append({"user_id": user_id, "params": params or {}, "path": path})
        return proxy_result

    monkeypatch.setattr(m, "_proxy_memory_list_from_tenant", fake_proxy)

    class _PlatformSvc:
        def __init__(self, db):
            pass

        async def search_memories(self, user_id, request):
            row = type("M", (), {
                "id": "99999999-9999-9999-9999-999999999999",
                "content": PLATFORM_ROW_CONTENT,
                "summary": None, "category": "context", "importance": 0.5,
            })()
            res = type("R", (), {"memory": row, "score": 0.4, "explanation": None})()
            return [res], 1, 3.0

    monkeypatch.setattr(m, "MemoryService", _PlatformSvc)
    return calls


@pytest.mark.asyncio
async def test_search_returns_the_tenants_rows_not_the_platforms(monkeypatch):
    """The whole bug in one assertion: the tenant's row must come back.

    MUTATION: delete the `_proxy_memory_list_from_tenant(...)` block from
    `memory_search` -> this returns the platform row and the test goes red.
    """
    import app.mcp_server as m

    calls = _install(monkeypatch, proxy_result={
        "results": [TENANT_ROW], "total_count": 1, "search_time_ms": 7.5,
    })

    out = await m.memory_search.fn(query="where do I work", limit=5)

    contents = [r["content"] for r in out["results"]]
    assert contents == ["USER works at co-working space"], (
        f"memory_search did not return the tenant's rows: {contents}"
    )
    assert PLATFORM_ROW_CONTENT not in contents, (
        "memory_search answered from the platform's stale monolith table"
    )
    assert calls and calls[0]["path"] == "search", (
        f"tenant was not asked for /search: {calls}"
    )
    assert calls[0]["params"].get("query") == "where do I work"


@pytest.mark.asyncio
async def test_score_comes_from_the_tenants_similarity_score_field(monkeypatch):
    """The tenant serialises `similarity_score`; this tool's contract says
    `score`. Mapping the wrong key yields a silent 0.0 on every result, which
    ranks correctly-retrieved memories as worthless."""
    import app.mcp_server as m

    _install(monkeypatch, proxy_result={
        "results": [TENANT_ROW], "total_count": 1, "search_time_ms": 7.5,
    })

    out = await m.memory_search.fn(query="q", limit=5)
    assert out["results"][0]["score"] == pytest.approx(0.91), (
        "score was not mapped from the tenant's similarity_score"
    )
    assert out["total"] == 1, "total was not mapped from total_count"


@pytest.mark.asyncio
async def test_unreachable_tenant_falls_back_to_the_platform_session(monkeypatch):
    """Anti-vacuity control — and the documented read asymmetry.

    A READ may fall back (worst case: a stale or empty answer); a WRITE may
    not. If this test ever fails because the fallback vanished, searches would
    start raising when a tenant is briefly unreachable. It also proves the two
    tests above are not passing merely because the platform path is dead code:
    here the platform path IS exercised and returns its own distinct row.
    """
    import app.mcp_server as m

    _install(monkeypatch, proxy_result=None)   # tenant unreachable

    out = await m.memory_search.fn(query="q", limit=5)
    assert [r["content"] for r in out["results"]] == [PLATFORM_ROW_CONTENT], (
        "the platform fall-back stopped working"
    )
