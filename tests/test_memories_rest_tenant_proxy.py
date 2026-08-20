"""Every /api/memories route must reach the tenant (G-1, audit 2026-08-09).

`memory_files` is AGENT_ONLY: the real memory lives in the tenant
container's DB. The original failure mode was never "someone removed a
proxy call", it was "someone added an endpoint and forgot the proxy
exists" — so the pin is structural and applies to future routes.

MEMORY V3 (2026-08-20) moved the behavioural half of this file to
`tests/test_memories_v3_api.py`: the routes it drove — `/deduplicate`,
`/{id}/merge`, `/{id}/related`, `/{id}/events` — served the retired ROW
layer and no longer exist. What remains here is the invariant those tests
were an instance of, re-derived against the v3 surface, plus the one
asymmetry that CHANGED and must not drift back:

  round 8: a read that could not reach the tenant fell back to the
           platform DB, because rows were mirrored there.
  v3:      there is no mirror. A read that cannot reach the tenant is a
           503, and an empty library would read as data loss.
"""

from __future__ import annotations

import inspect
import re

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
        out.append((m.group(1).upper(), m.group(2), fn.group(1) if fn else "?", body))
    return out


# ── Structural: no route may skip the tenant ─────────────────────────────

def test_every_memories_route_consults_the_tenant_proxy():
    routes = _route_blocks()
    # The floor is now an EXACT count. Round 8 asserted `>= 19` because the
    # surface was still growing; v3's surface is closed (contract §4), so a
    # new route is a decision someone has to make here on purpose. It moved
    # to 11 once (WS-5): §7's rollback has to be executable, the tenant's
    # tables are only reachable through the tenant, and a rollback nobody
    # can run is a paragraph rather than a plan.
    assert len(routes) == 11, f"route sweep looks broken: found {len(routes)}"
    missing = [
        f"{meth} /memories{path or ''} ({name})"
        for meth, path, name, body in routes
        if "_get_agent_proxy_info" not in body
    ]
    assert not missing, (
        "routes that never reach the tenant (memory_files is AGENT_ONLY and "
        f"the platform has NO copy at all): {missing}"
    )


def test_write_routes_use_the_write_proxy_not_the_read_fallback():
    """A write that falls back to the platform DB on failure 'succeeds'
    against the wrong database. Writes go through _proxy_memories_write,
    which 502s instead of falling back."""
    routes = {name: body for _, _, name, body in _route_blocks()}
    for name in ("instruct_memory_file", "instruct_memory", "delete_memory_file",
                 "forget_all_memories", "admin_migrate_v3",
                 "admin_migrate_v3_rollback"):
        assert "_proxy_memories_write" in routes[name], (
            f"{name} is a write and must use the 502-on-failure proxy"
        )


def test_reads_no_longer_fall_back_to_the_platform_database():
    """The one contract that CHANGED. `_proxy_memories` still returns None
    on failure, but every caller now raises 503 instead of running a local
    branch — there is nothing left in the platform DB to run it against."""
    routes = {name: body for _, _, name, body in _route_blocks()}
    for name in ("list_memory_files", "get_memory_file", "search_memory_files",
                 "memory_log", "admin_migrate_v3_report"):
        body = routes[name]
        assert "_agent_unreachable()" in body, name
        assert "MemoryService" not in body, (
            f"{name} runs a platform-DB branch — that is the round-8 virtual "
            "view, and under v3 it can only produce an empty library"
        )


# ── Structural: ordering ─────────────────────────────────────────────────
# Starlette matches in declaration order. `{slug:path}` is greedy, so every
# literal route must either be declared BEFORE the captures or be
# unreachable-by-construction from them. Round 8's version of this test
# looked for `/{memory_id}` — the capture that route no longer exists.

def test_no_literal_route_is_shadowed_by_the_slug_capture():
    order = [path for _, path, _, _ in _route_blocks()]
    captures = [i for i, p in enumerate(order) if "{slug:path}" in p]
    assert captures, "the memory-file routes are gone"
    for i, path in enumerate(order):
        if "{" in path:
            continue
        for c in captures:
            if c < i:
                assert order[c].startswith("/files/"), (
                    f"literal route {path!r} is declared after the greedy "
                    f"capture {order[c]!r} and can never be reached"
                )


def test_the_row_era_routes_are_gone_not_merely_unused():
    """`/breakdown`, `/search` (POST), `/{memory_id}` and friends were the
    row product. Leaving one behind leaves a way back to it."""
    paths = {path for _, path, _, _ in _route_blocks()}
    assert not [p for p in paths if "{memory_id}" in p]
    assert "/breakdown" not in paths
    assert "/deduplicate" not in paths
    assert "/duplicates/report" not in paths
