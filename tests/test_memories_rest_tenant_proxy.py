"""Every /api/memories route must reach the tenant (G-1, audit 2026-08-09).

`memories` is AGENT_ONLY: the real rows live in the tenant container's DB and
the platform's copy is stale monolith-era data. 11 of 19 routes proxied; the
other 8 quietly served (or wrote!) the platform DB:

  reads : /related  /region  /type  /{id}/events  /{id}/history
          /duplicates/report
  writes: /deduplicate  /{id}/merge

Two of the reads are called by the web app today (api.ts: /related, /type),
so the memory UI showed 2026-05 data or 404s. The two writes violated the
writes-must-502 contract (`_proxy_memories_write` docstring): /deduplicate
"succeeded" against stale rows while the user's real duplicates sat
untouched; /merge rewrote a platform row the agent never reads.

The structural test pins the invariant for FUTURE routes too — the failure
mode was never "someone removed a proxy call", it was "someone added an
endpoint and forgot the proxy exists".
"""

from __future__ import annotations

import inspect
import re
import types

import pytest

from app.api import memories as mem_api


def _route_blocks():
    """Split memories.py source into (method, path, handler_name, body)."""
    src = inspect.getsource(mem_api)
    blocks = re.split(r'(@router\.(?:get|post|patch|delete)\("[^"]*"[^)]*\))', src)
    out = []
    for i in range(1, len(blocks), 2):
        deco = blocks[i]
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        m = re.match(r'@router\.(\w+)\("([^"]*)"', deco)
        fn = re.search(r"async def (\w+)", body)
        # body of THIS handler only — stop at the next decorator if regex
        # split ever changes; today each block is exactly one handler.
        out.append((m.group(1).upper(), m.group(2), fn.group(1) if fn else "?", body))
    return out


# ── Structural: no route may skip the tenant ─────────────────────────────

def test_every_memories_route_consults_the_tenant_proxy():
    routes = _route_blocks()
    assert len(routes) >= 19, f"route sweep looks broken: found {len(routes)}"
    missing = [
        f"{meth} /memories{path or ''} ({name})"
        for meth, path, name, body in routes
        if "_get_agent_proxy_info" not in body
    ]
    assert not missing, (
        "routes that never reach the tenant (the platform copy of `memories` "
        f"is stale monolith data): {missing}"
    )


def test_write_routes_use_the_write_proxy_not_the_read_fallback():
    """A write that falls back to the platform DB on failure 'succeeds'
    against the wrong database. Writes go through _proxy_memories_write,
    which 502s instead of falling back."""
    routes = {name: body for _, _, name, body in _route_blocks()}
    for name in ("deduplicate_memories", "merge_into_memory", "create_memory",
                 "update_memory", "delete_memory"):
        assert "_proxy_memories_write" in routes[name], (
            f"{name} is a write and must use the 502-on-failure proxy"
        )


# ── Behavioral: the two new write proxies route correctly ────────────────

class _User:
    id = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def proxied(monkeypatch):
    calls = []

    async def fake_info(user_id, db):
        return ("http://agent.test", "key-123")

    async def fake_write(agent_url, agent_api_key, path, method,
                         body=None, params=None):
        calls.append({"path": path, "method": method, "params": params})
        return {"proxied": True}

    async def fake_read(agent_url, agent_api_key, path, params=None,
                        method="GET", body=None):
        calls.append({"path": path, "method": method, "params": params})
        return {"proxied": True}

    monkeypatch.setattr(mem_api, "_get_agent_proxy_info", fake_info)
    monkeypatch.setattr(mem_api, "_proxy_memories_write", fake_write)
    monkeypatch.setattr(mem_api, "_proxy_memories", fake_read)
    return calls


async def test_deduplicate_routes_to_the_tenant_with_its_params(proxied):
    resp = await mem_api.deduplicate_memories(
        category="preferences", brain_type=None, dry_run=False,
        current_user=_User(), db=None,
    )
    assert resp.status_code == 200
    assert proxied == [{
        "path": "deduplicate", "method": "POST",
        "params": {"dry_run": False, "category": "preferences"},
    }], "brain_type=None must be omitted, dry_run must pass through"


async def test_merge_routes_to_the_tenant(proxied):
    resp = await mem_api.merge_into_memory(
        memory_id="mem-1", new_content="the office moved to floor 3",
        source_type="manual", current_user=_User(), db=None,
    )
    assert resp.status_code == 200
    assert proxied[0]["path"] == "mem-1/merge"
    assert proxied[0]["params"]["new_content"] == "the office moved to floor 3"


async def test_merge_refuses_never_store_content_before_proxying(proxied):
    """The secret gate is a pure function; it must fire before the network,
    preserving the 422 + reason instead of a generic tenant 502."""
    from fastapi import HTTPException

    pan = "4111 1111 1111 1111"  # test PAN, passes Luhn
    with pytest.raises(HTTPException) as exc:
        await mem_api.merge_into_memory(
            memory_id="mem-1", new_content=f"my card number is {pan}",
            source_type="manual", current_user=_User(), db=None,
        )
    assert exc.value.status_code == 422
    assert proxied == [], "refused content must never leave the platform"


async def test_related_read_proxies_with_depth(proxied):
    resp = await mem_api.get_memory_with_relations(
        memory_id="mem-9", depth=2, current_user=_User(), db=None,
    )
    assert resp.status_code == 200
    assert proxied == [{"path": "mem-9/related", "method": "GET",
                        "params": {"depth": 2}}]


async def test_events_read_falls_back_to_local_when_proxy_returns_none(monkeypatch):
    """Reads MAY fall back (worst case: stale/empty) — pin the asymmetry."""
    async def fake_info(user_id, db):
        return ("http://agent.test", "key-123")

    async def fake_read(*a, **kw):
        return None  # tenant unreachable

    class _Svc:
        def __init__(self, db):
            pass

        async def get_memory(self, memory_id, user_id):
            return None  # local platform DB has no such row

    monkeypatch.setattr(mem_api, "_get_agent_proxy_info", fake_info)
    monkeypatch.setattr(mem_api, "_proxy_memories", fake_read)
    monkeypatch.setattr(mem_api, "MemoryService", _Svc)

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await mem_api.get_memory_events(
            memory_id="mem-9", limit=100, current_user=_User(), db=None,
        )
    assert exc.value.status_code == 404, "read fallback runs the local path"


# ── Structural: the file routes must precede /{memory_id} ────────────────
# Starlette matches in declaration order; a `/files` route declared after
# `/{memory_id}` is silently captured by it ("files" parses as a memory id)
# and every files request 404s. Same class of pin as the /breakdown and
# /search ordering notes in memories.py.

def test_file_routes_precede_the_memory_id_capture():
    routes = _route_blocks()
    order = [path for _, path, _, _ in routes]
    id_capture = next(i for i, p in enumerate(order) if p.startswith("/{memory_id}"))
    file_routes = [i for i, p in enumerate(order) if p.startswith("/files")]
    assert file_routes, "the memory-file routes are gone"
    late = [order[i] for i in file_routes if i > id_capture]
    assert not late, f"file routes declared after /{{memory_id}} capture them: {late}"
