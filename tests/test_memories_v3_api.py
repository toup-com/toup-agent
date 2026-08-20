"""The v3 /api/memories surface (rebuild-2026-08-v3 §4).

Three things this file exists to catch, each of which shipped once:

1. **An engine field reaching a client.** Round 8 served 29-field rows and
   both Memory pages became database-row inspectors. `test_no_engine_
   metadata_reaches_a_client` walks every response body this router can
   produce — including the tenant-proxy pass-through — and fails on a
   banned key at any depth.
2. **A read falling back to the platform DB.** With rows retired there is
   nothing to synthesize a file from, so an unreachable agent must 503.
   Round 8's virtual view would now be an empty library, which reads as
   data loss.
3. **A write "succeeding" against the wrong database.** Writes go through
   `_proxy_memories_write`, which 502s and never falls back.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.api import memories as mem_api


class _User:
    id = "00000000-0000-0000-0000-000000000001"


# Every field the round-8 payload carried that the product must never see
# again, plus the internal columns v3 adds. A key at ANY depth fails.
BANNED_KEYS = {
    "strength", "memory_level", "emotional_salience", "decay_rate",
    "consolidation_count", "last_reinforced_at", "last_accessed_at",
    "access_count", "importance", "confidence", "expires_at", "is_active",
    "is_deleted", "superseded_by", "merged_from", "history", "brain_type",
    "category", "memory_type", "source_type", "file_position", "entry_count",
    # NB: "entries" is NOT banned globally — the memory LOG's shape is
    # {days: [{day, entries: […]}]}, and those entries are change lines,
    # not rows. The round-8 sense of the word (a file's memory rows) is
    # pinned separately, on the file payload itself.
    "similarity_score", "explanation", "event_type", "trigger_source",
    "memories", "organized", "purpose", "consolidated_at",
    "pinned_meta_json", "related_json", "links_json", "body", "metadata",
    "tags", "canonical_content",
}


def _banned_in(payload, path="$"):
    """Every banned key found anywhere in a response, with its path."""
    found = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in BANNED_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_banned_in(value, f"{path}.{key}"))
    elif isinstance(payload, list):
        for i, value in enumerate(payload):
            found.extend(_banned_in(value, f"{path}[{i}]"))
    return found


# ── Structural: the surface itself ───────────────────────────────────

def test_the_surface_is_exactly_the_v3_routes():
    """The contract names nine routes plus forget-everything, plus the
    migration admin set. A route that is not on this list is round-8
    machinery that came back.

    The admin set grew by one deliberately (WS-5): the rollback of §7 has
    to be executable, the tenant's tables are only reachable through the
    tenant, and a rollback nobody can run is a paragraph rather than a
    plan. It is behind an exact confirmation string, like the other
    destructive route on this router."""
    actual = {
        (sorted(r.methods)[0], r.path) for r in mem_api.router.routes
    }
    assert actual == {
        ("GET", "/memories/files"),
        ("GET", "/memories/files/{slug:path}"),
        ("POST", "/memories/files/{slug:path}/instruct"),
        ("DELETE", "/memories/files/{slug:path}"),
        ("POST", "/memories/instruct"),
        ("GET", "/memories/search"),
        ("GET", "/memories/log"),
        ("POST", "/memories/admin/migrate-v3"),
        ("GET", "/memories/admin/migrate-v3/report"),
        ("POST", "/memories/admin/migrate-v3/rollback"),
        ("DELETE", "/memories"),
    }


def test_no_row_route_survives():
    """`GET /{memory_id}`, `PATCH`, `/reinforce`, `/events`, `/breakdown`,
    `/deduplicate`, `/duplicates/report`, `/{id}/merge`, `/category/{c}`,
    `/region/{r}`, `/type/{t}` — all of them served the retired row layer."""
    paths = {r.path for r in mem_api.router.routes}
    for gone in (
        "/memories/{memory_id}", "/memories/breakdown", "/memories/deduplicate",
        "/memories/{memory_id}/events", "/memories/{memory_id}/reinforce",
        "/memories/{memory_id}/related", "/memories/{memory_id}/history",
        "/memories/duplicates/report", "/memories/{memory_id}/merge",
        "/memories/category/{category}", "/memories/type/{memory_type}",
        "/memories/region/{region}",
    ):
        assert gone not in paths, gone
    # And nothing POSTs a row into existence any more.
    assert ("POST", "/memories") not in {
        (sorted(r.methods)[0], r.path) for r in mem_api.router.routes
    }


def test_literal_routes_precede_the_greedy_slug_capture():
    """Starlette matches in declaration order and `{slug:path}` is greedy.
    `/memories/instruct`, `/search`, `/log` and `/admin/*` are only
    reachable because every capture above them is prefixed by `/files/`."""
    order = [r.path for r in mem_api.router.routes]
    captures = [i for i, p in enumerate(order) if "{slug:path}" in p]
    for literal in (
        "/memories/instruct", "/memories/search", "/memories/log",
        "/memories/admin/migrate-v3", "/memories/admin/migrate-v3/rollback",
    ):
        i = order.index(literal)
        for c in captures:
            if c < i:
                assert order[c].startswith("/memories/files/"), (
                    f"{literal} is shadowed by the capture {order[c]}"
                )


# ── Structural: the proxy invariants (the round-8 pins) ──────────────

def _route_blocks():
    """Split memories.py source into (method, path, handler_name, body)."""
    import inspect
    import re

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


def test_every_route_consults_the_tenant_proxy():
    routes = _route_blocks()
    assert len(routes) == 11, f"route sweep looks broken: found {len(routes)}"
    missing = [
        f"{meth} /memories{path or ''} ({name})"
        for meth, path, name, body in routes
        if "_get_agent_proxy_info" not in body
    ]
    assert not missing, (
        "routes that never reach the tenant (memory_files is AGENT_ONLY and "
        f"the platform has no copy at all): {missing}"
    )


def test_write_routes_use_the_write_proxy_not_the_read_fallback():
    routes = {name: body for _, _, name, body in _route_blocks()}
    for name in (
        "instruct_memory_file", "instruct_memory", "delete_memory_file",
        "forget_all_memories", "admin_migrate_v3", "admin_migrate_v3_rollback",
    ):
        assert "_proxy_memories_write" in routes[name], (
            f"{name} is a write and must use the 502-on-failure proxy"
        )


def test_read_routes_degrade_to_503_not_to_a_platform_view():
    routes = {name: body for _, _, name, body in _route_blocks()}
    for name in ("list_memory_files", "get_memory_file", "search_memory_files",
                 "memory_log", "admin_migrate_v3_report"):
        assert "_agent_unreachable()" in routes[name], (
            f"{name} must 503 when the tenant does not answer — there are no "
            "rows left to synthesize a file from"
        )


# ── Behavioural: proxying ────────────────────────────────────────────

class _Calls(list):
    """A call log that also carries the canned tenant replies."""
    replies: dict


@pytest.fixture
def proxied(monkeypatch):
    calls = _Calls()
    replies = {}

    async def fake_info(user_id, db):
        return ("http://agent.test", "key-123")

    async def fake_write(agent_url, agent_api_key, path, method,
                         body=None, params=None):
        calls.append({"path": path, "method": method, "params": params, "body": body})
        return replies.get(("write", path), {"applied": 1, "rejected": [], "note": "Saved 1 change."})

    async def fake_read(agent_url, agent_api_key, path, params=None,
                        method="GET", body=None):
        calls.append({"path": path, "method": method, "params": params})
        return replies.get(("read", path), {"sections": []})

    monkeypatch.setattr(mem_api, "_get_agent_proxy_info", fake_info)
    monkeypatch.setattr(mem_api, "_proxy_memories_write", fake_write)
    monkeypatch.setattr(mem_api, "_proxy_memories", fake_read)
    calls.replies = replies
    return calls


async def test_each_route_forwards_the_path_the_tenant_expects(proxied):
    await mem_api.list_memory_files(current_user=_User(), db=None)
    await mem_api.get_memory_file(slug="people/majid-tajik", current_user=_User(), db=None)
    await mem_api.search_memory_files(q="teams", limit=5, current_user=_User(), db=None)
    await mem_api.memory_log(month="2026-08", current_user=_User(), db=None)
    assert [c["path"] for c in proxied] == [
        "files", "files/people/majid-tajik", "search", "log",
    ]
    assert proxied[2]["params"] == {"q": "teams", "limit": 5}
    assert proxied[3]["params"] == {"month": "2026-08"}


async def test_instruct_forwards_the_instruction_as_a_write(proxied):
    from app.schemas import MemoryFileInstructRequest

    payload = MemoryFileInstructRequest(instruction="forget the old exam date")
    await mem_api.instruct_memory_file(
        slug="areas/ielts", payload=payload, current_user=_User(), db=None,
    )
    await mem_api.instruct_memory(payload=payload, current_user=_User(), db=None)
    assert [(c["path"], c["method"]) for c in proxied] == [
        ("files/areas/ielts/instruct", "POST"), ("instruct", "POST"),
    ]
    assert proxied[0]["body"] == {"instruction": "forget the old exam date"}


async def test_delete_forwards_and_answers_the_client_contract(proxied):
    resp = await mem_api.delete_memory_file(
        slug="topics/music", current_user=_User(), db=None,
    )
    assert proxied == [{
        "path": "files/topics/music", "method": "DELETE",
        "params": None, "body": None,
    }]
    import json as _json
    assert _json.loads(resp.body) == {"deleted": True}


async def test_forget_everything_still_needs_the_exact_confirmation(proxied):
    with pytest.raises(HTTPException) as exc:
        await mem_api.forget_all_memories(
            confirm="forget everything", current_user=_User(), db=None,
        )
    assert exc.value.status_code == 400
    assert proxied == [], "a mistyped confirmation must never reach the tenant"

    await mem_api.forget_all_memories(
        confirm="FORGET EVERYTHING", current_user=_User(), db=None,
    )
    assert proxied[0]["params"] == {"confirm": "FORGET EVERYTHING"}


async def test_an_unreachable_tenant_is_a_503_not_an_empty_library(monkeypatch):
    async def fake_info(user_id, db):
        return ("http://agent.test", "key-123")

    async def fake_read(*a, **kw):
        return None  # tenant unreachable

    monkeypatch.setattr(mem_api, "_get_agent_proxy_info", fake_info)
    monkeypatch.setattr(mem_api, "_proxy_memories", fake_read)

    for call in (
        mem_api.list_memory_files(current_user=_User(), db=None),
        mem_api.get_memory_file(slug="you/profile", current_user=_User(), db=None),
        mem_api.search_memory_files(q="x", limit=5, current_user=_User(), db=None),
        mem_api.memory_log(month="2026-08", current_user=_User(), db=None),
    ):
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 503
        assert exc.value.detail == "Your agent isn't reachable right now."


async def test_a_tenant_404_stays_a_404(monkeypatch):
    """Distinguishable from unreachable: the file genuinely is not there."""
    async def fake_info(user_id, db):
        return ("http://agent.test", "key-123")

    async def fake_read(*a, **kw):
        return {"__status__": 404}

    monkeypatch.setattr(mem_api, "_get_agent_proxy_info", fake_info)
    monkeypatch.setattr(mem_api, "_proxy_memories", fake_read)
    with pytest.raises(HTTPException) as exc:
        await mem_api.get_memory_file(
            slug="topics/ghost", current_user=_User(), db=None,
        )
    assert exc.value.status_code == 404


async def test_the_platform_has_no_local_branch_to_fall_back_to(monkeypatch):
    """No agent row at all + RUN_MODE=platform. Round 8 answered from the
    platform's `memories` copy here; v3 has no such copy."""
    async def no_agent(user_id, db):
        return None

    monkeypatch.setattr(mem_api, "_get_agent_proxy_info", no_agent)
    monkeypatch.setattr(mem_api, "_platform_mode", lambda: True)
    with pytest.raises(HTTPException) as exc:
        await mem_api.list_memory_files(current_user=_User(), db=None)
    assert exc.value.status_code == 503


async def test_the_migration_routes_never_answer_from_the_platform(monkeypatch):
    """They were 501 stubs; WS-5 implemented them agent-side.

    The platform half stays a pure proxy, and that is the property worth
    pinning: `migration_status`, `memory_files` and `memory_file_changes`
    are all AGENT_ONLY, so a platform-local branch would report "migrated
    0 rows" from a database that never held any — which a fleet driver
    cannot tell apart from a tenant that really has nothing.
    """
    async def no_agent(user_id, db):
        return None

    monkeypatch.setattr(mem_api, "_get_agent_proxy_info", no_agent)
    monkeypatch.setattr(mem_api, "_platform_mode", lambda: True)
    for call in (
        mem_api.admin_migrate_v3(dry_run=True, current_user=_User(), db=None),
        mem_api.admin_migrate_v3_report(current_user=_User(), db=None),
        mem_api.admin_migrate_v3_rollback(
            confirm="ROLLBACK MEMORY V3", hard=False, current_user=_User(), db=None,
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 503


async def test_rollback_needs_the_exact_confirmation(monkeypatch):
    """Destructive, and reachable with a key a proxy already holds — the
    same reason `DELETE /api/memories` carries a confirmation string. The
    check runs BEFORE the proxy lookup, so a typo never reaches a tenant."""
    called = []

    async def boom(user_id, db):
        called.append(user_id)
        return ("http://agent", "k")

    monkeypatch.setattr(mem_api, "_get_agent_proxy_info", boom)
    with pytest.raises(HTTPException) as exc:
        await mem_api.admin_migrate_v3_rollback(
            confirm="rollback memory v3", hard=False, current_user=_User(), db=None,
        )
    assert exc.value.status_code == 400
    assert not called, "a bad confirmation must not reach the tenant at all"


# ── The zero-metadata contract ───────────────────────────────────────

async def test_no_engine_metadata_reaches_a_client(monkeypatch, proxied):
    """Walk every response this router can produce, at every depth."""
    import json as _json

    # The tenant answers with realistic v3 bodies…
    proxied.replies[("read", "files")] = {
        "sections": [{"section": "you", "files": [{
            "slug": "you/profile", "section": "you", "title": "Profile",
            "description": "Who this person is — setup; read when it matters.",
            "updated_at": "2026-08-20T10:00:00",
        }]}]
    }
    proxied.replies[("read", "files/you/profile")] = {
        "slug": "you/profile", "section": "you", "title": "Profile",
        "description": "Who this person is — setup; read when it matters.",
        "body_md": "- uses an Android phone",
        "links": [{"slug": "topics/music", "title": "Music"}],
        "updated_at": "2026-08-20T10:00:00",
    }
    proxied.replies[("read", "search")] = {
        "results": [{"slug": "topics/music", "title": "Music", "snippet": "…"}]
    }
    proxied.replies[("read", "log")] = {
        "days": [{"day": "2026-08-20", "entries": [{
            "file_slug": "topics/music", "file_title": "Music",
            "kind": "updated", "summary": "Added Music: likes Googoosh.",
            "at": "2026-08-20T10:00:00",
        }]}]
    }

    responses = [
        await mem_api.list_memory_files(current_user=_User(), db=None),
        await mem_api.get_memory_file(slug="you/profile", current_user=_User(), db=None),
        await mem_api.search_memory_files(q="music", limit=5, current_user=_User(), db=None),
        await mem_api.memory_log(month="2026-08", current_user=_User(), db=None),
        await mem_api.delete_memory_file(slug="topics/music", current_user=_User(), db=None),
        await mem_api.forget_all_memories(
            confirm="FORGET EVERYTHING", current_user=_User(), db=None,
        ),
    ]
    from app.schemas import MemoryFileInstructRequest
    responses.append(await mem_api.instruct_memory(
        payload=MemoryFileInstructRequest(instruction="remember I use Android"),
        current_user=_User(), db=None,
    ))

    # The round-8 sense of "entries": a file detail carried its memory ROWS
    # under that key, and both clients built the file page from them.
    file_detail = _json.loads(responses[1].body)
    assert "entries" not in file_detail

    offenders = []
    for resp in responses:
        body = resp
        if hasattr(resp, "body"):
            body = _json.loads(resp.body)
        offenders.extend(_banned_in(body))
    assert not offenders, f"engine metadata reached a client: {offenders}"


def test_the_row_serializer_is_not_wired_to_any_route():
    """`memory_to_response` still exists — ingest.py, agent.py and stats.py
    import it for the DOCUMENT/MEDIA leg (§3.4). It must not be reachable
    from a /api/memories route, or the 29-field payload comes straight
    back."""
    for _, path, name, body in _route_blocks():
        assert "memory_to_response" not in body, (
            f"{name} ({path}) serializes rows — the memory product is files"
        )


# ── The `{slug:path}` capture is user input on an AUTHENTICATED hop ───

_HOSTILE_SLUGS = [
    "../admin/whatever",              # already decoded by Starlette
    "..%2f..%2fadmin/whatever",       # the encoded form, if it survives
    "you/profile/../../admin",
    "you/profile?x=1",                # a query the tenant would parse
    "you/profile#frag",
    "/absolute/path",
    "a/b/c",                          # two namespaces: no such shape exists
    "absolute/path",                  # well-formed, but in NO v3 section
    "you/pro file",                   # a space
    "you/profile\\..\\admin",
    "x" * 200,                        # over the 120 cap
    "",
    "you/pro_file",                   # underscore: banned by is_valid_slug
]


@pytest.mark.parametrize("slug", _HOSTILE_SLUGS, ids=range(len(_HOSTILE_SLUGS)))
def test_a_malformed_slug_is_refused_before_it_reaches_the_tenant(slug):
    """`_slug_path` used to be `slug.lstrip("/")`, and its result was
    interpolated raw into an X-Agent-Key-AUTHENTICATED proxy URL. A caller
    could therefore shape the path of an authenticated request to the tenant
    agent.

    Rejection, not sanitation: a sanitizer that "fixes" a hostile path still
    forwards a request the caller shaped. 404 rather than 400 because a
    malformed slug and an unknown one are the same answer to a client, and a
    distinguishable 400 tells a prober which shapes reach the parser.
    """
    with pytest.raises(HTTPException) as exc:
        mem_api._slug_path(slug)
    assert exc.value.status_code == 404
    assert "not found" in str(exc.value.detail).lower()


@pytest.mark.parametrize("slug,expect", [
    ("you/profile", "you/profile"),
    ("/you/profile", "you/profile"),      # a leading slash is still tolerated
    ("learned", "learned"),
    ("people/majid-tajik", "people/majid-tajik"),
    ("people/گوگوش", "people/گوگوش"),      # a Persian name makes a Persian slug
])
def test_a_real_slug_still_passes(slug, expect):
    """ANTI-VACUITY. A validator that refused everything would pass every
    test above and break the whole product."""
    assert mem_api._slug_path(slug) == expect


async def test_every_slug_route_validates_before_proxying(proxied, monkeypatch):
    """All THREE call sites, driven through the handlers.

    A guard applied at two of three is the same defect with a smaller blast
    radius, and the three are separately reachable (GET, POST instruct,
    DELETE).
    """
    from app.schemas import MemoryFileInstructRequest

    payload = MemoryFileInstructRequest(instruction="anything")
    hostile = "../admin/whatever"

    for coro in (
        mem_api.get_memory_file(slug=hostile, current_user=_User(), db=None),
        mem_api.instruct_memory_file(
            slug=hostile, payload=payload, current_user=_User(), db=None,
        ),
        mem_api.delete_memory_file(slug=hostile, current_user=_User(), db=None),
    ):
        with pytest.raises(HTTPException) as exc:
            await coro
        assert exc.value.status_code == 404

    assert not proxied, (
        "a hostile slug reached the authenticated proxy: "
        f"{[c['path'] for c in proxied]}"
    )
