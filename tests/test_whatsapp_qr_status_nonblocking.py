"""R46 F4 — a status POLL is a read, and a link is announced, not discovered.

Incident 2026-09-15. The sidecar witnessed `connection.open` at 14:48:57.919.
The platform learned of it at 14:49:18.692 — 21 s later — because that is when
a backgrounded phone resumed polling. The agent knew all along and nothing
told anyone.

Two changes here:

(a) `whatsapp_qr_status` returns the agent's snapshot BEFORE it persists it and
    before any bridge push. It is polled by every client on every foreground
    for every not-yet-linked user; making a read cost a write (and, on the
    link tick, an AWAITED env push) is latency the user pays for nothing.
(b) The agent announces the link to a new internal route the moment it sees
    it, so the stored state is right before anyone asks.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test \\
      pytest tests/test_whatsapp_qr_status_nonblocking.py -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import pathlib
import time
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException

_HERE = pathlib.Path(__file__).resolve().parents[1]
_BAILEYS_SRC = (_HERE / "app" / "agent" / "channels" / "whatsapp_baileys.py").read_text()


class _Cfg:
    def __init__(self, status=None, self_e164=None, allowlist=""):
        self.whatsapp_session_status = status
        self.whatsapp_self_e164 = self_e164
        self.whatsapp_baileys_allowlist = allowlist
        self.updated_at = None


class _Db:
    def __init__(self, row=None):
        self.commits = 0
        self._row = row

    async def commit(self):
        self.commits += 1

    async def execute(self, _stmt):
        row = self._row

        class _R:
            def first(self_inner):
                return row
        return _R()


class _Row:
    def __init__(self, user_id):
        self.user_id = user_id


class _User:
    id = "11111111-2222-3333-4444-555555555555"


# ── (a) the poll answers first ────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_poll_returns_before_the_persist_resolves(monkeypatch):
    from app.api import agent_setup as mod

    snapshot = {"session_status": "linked", "self_e164": "+14155552671"}
    spawned = []

    async def fake_proxy(method, path, user_id, db, **kw):
        return snapshot

    def fake_spawn(coro, name=None):
        spawned.append(name)
        coro.close()
        return object()

    async def never(*a, **kw):
        await asyncio.sleep(3600)

    monkeypatch.setattr(mod, "_agent_qr_proxy", fake_proxy)
    monkeypatch.setattr(mod, "_spawn_bg", fake_spawn)
    monkeypatch.setattr(mod, "_wa_persist_snapshot", never)

    t0 = time.monotonic()
    out = await mod.whatsapp_qr_status(current_user=_User(), db=_Db())
    elapsed = time.monotonic() - t0

    assert out is snapshot
    assert elapsed < 1.0, "the poll waited for its own bookkeeping"
    assert spawned, "the persist was dropped, not deferred"


@pytest.mark.asyncio
async def test_the_persist_still_runs_behind_the_response(monkeypatch):
    """Deferred, not deleted: the durable record is the whole reason the
    Settings card survives a reload."""
    from app.api import agent_setup as mod

    cfg = _Cfg(status="linking")

    async def fake_get_or_create(user_id, db):
        return cfg

    @asynccontextmanager
    async def fake_sm():
        yield _Db()

    monkeypatch.setattr(mod, "_get_or_create_config", fake_get_or_create)
    monkeypatch.setattr("app.db.database.async_session_maker", fake_sm)

    await mod._wa_persist_snapshot(
        str(_User.id),
        {"session_status": "linked", "self_e164": "+14155552671"},
    )

    assert cfg.whatsapp_session_status == "linked"
    assert cfg.whatsapp_self_e164 == "+14155552671"


@pytest.mark.asyncio
async def test_the_persist_seeds_the_self_allowlist_only_when_empty(monkeypatch):
    from app.api import agent_setup as mod

    pushed = []

    async def fake_push(user_id):
        pushed.append(user_id)
        return True

    @asynccontextmanager
    async def fake_sm():
        yield _Db()

    monkeypatch.setattr("app.db.database.async_session_maker", fake_sm)
    monkeypatch.setattr(mod, "_env_push_worker", fake_push)

    empty = _Cfg(status="linking", allowlist="")

    async def get_empty(user_id, db):
        return empty
    monkeypatch.setattr(mod, "_get_or_create_config", get_empty)
    await mod._wa_persist_snapshot(
        str(_User.id), {"session_status": "linked", "self_e164": "+14155552671"},
    )
    assert empty.whatsapp_baileys_allowlist == "+14155552671"
    assert len(pushed) == 1

    full = _Cfg(status="linking", allowlist="+14155559999")

    async def get_full(user_id, db):
        return full
    monkeypatch.setattr(mod, "_get_or_create_config", get_full)
    await mod._wa_persist_snapshot(
        str(_User.id), {"session_status": "linked", "self_e164": "+14155552671"},
    )
    assert full.whatsapp_baileys_allowlist == "+14155559999", "an existing allowlist was overwritten"
    assert len(pushed) == 1, "a second env push for an allowlist that was already set"


@pytest.mark.asyncio
async def test_a_persist_failure_never_escapes(monkeypatch):
    """The poll has already answered; a DB hiccup behind it must not become an
    unhandled task exception."""
    from app.api import agent_setup as mod

    @asynccontextmanager
    async def boom():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr("app.db.database.async_session_maker", boom)
    await mod._wa_persist_snapshot(str(_User.id), {"session_status": "linked"})


# ── (b) the agent announces the link ──────────────────────────────────


@pytest.mark.asyncio
async def test_the_link_route_refuses_a_request_without_an_agent_key():
    from app.api import agent_setup as mod

    body = mod._WhatsAppLinkAnnounce(session_status="linked", self_e164="+1")
    # The db is armed with a row that WOULD match, so an empty key falling
    # through to a lookup (`key = key or "anon"`) cannot pass this as a 401
    # for the other reason.
    with pytest.raises(HTTPException) as exc:
        await mod.whatsapp_link_announce(
            body, db=_Db(row=_Row("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")),
            agent_key=None,
        )
    assert exc.value.status_code == 401
    assert "X-Agent-Key" in str(exc.value.detail), (
        "a missing key was refused by the lookup rather than by the check — "
        "an agent key that happens to be empty in the column would then "
        "authenticate an anonymous caller"
    )
    with pytest.raises(HTTPException) as exc2:
        await mod.whatsapp_link_announce(
            body, db=_Db(row=_Row("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")),
            agent_key="   ",
        )
    assert exc2.value.status_code == 401
    assert "X-Agent-Key" in str(exc2.value.detail)


@pytest.mark.asyncio
async def test_the_link_route_refuses_an_unknown_agent_key():
    from app.api import agent_setup as mod

    body = mod._WhatsAppLinkAnnounce(session_status="linked", self_e164="+1")
    with pytest.raises(HTTPException) as exc:
        await mod.whatsapp_link_announce(body, db=_Db(row=None), agent_key="nope")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_the_link_route_refuses_an_unknown_status():
    from app.api import agent_setup as mod

    body = mod._WhatsAppLinkAnnounce(session_status="banana")
    with pytest.raises(HTTPException) as exc:
        await mod.whatsapp_link_announce(
            body, db=_Db(row=_Row("u-1")), agent_key="k",
        )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_the_link_route_persists_for_the_key_s_own_owner(monkeypatch):
    """The body carries no user id, so a caller cannot name a tenant it does
    not hold the key for."""
    from app.api import agent_setup as mod

    seen = []

    def fake_spawn(coro, name=None):
        seen.append(name)
        coro.close()
        return object()

    monkeypatch.setattr(mod, "_spawn_bg", fake_spawn)
    body = mod._WhatsAppLinkAnnounce(session_status="linked", self_e164="+14155552671")
    out = await mod.whatsapp_link_announce(
        body, db=_Db(row=_Row("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")), agent_key="k",
    )

    assert out == {"ok": True}
    assert seen and "aaaaaaaa" in seen[0]


@pytest.mark.asyncio
async def test_a_repeated_announcement_is_a_no_op(monkeypatch):
    """Idempotent by construction: it runs the same persist a poll runs, and
    that path already refuses a write that changes nothing."""
    from app.api import agent_setup as mod

    cfg = _Cfg(status="linked", self_e164="+14155552671", allowlist="+14155552671")
    db = _Db()

    async def fake_get_or_create(user_id, _db):
        return cfg

    @asynccontextmanager
    async def fake_sm():
        yield db

    monkeypatch.setattr(mod, "_get_or_create_config", fake_get_or_create)
    monkeypatch.setattr("app.db.database.async_session_maker", fake_sm)

    snapshot = {
        "session_status": "linked", "self_e164": "+14155552671",
        "session_status_source": "sse", "session_status_stable_s": 0,
    }
    await mod._wa_persist_snapshot(str(_User.id), snapshot)
    await mod._wa_persist_snapshot(str(_User.id), snapshot)

    assert db.commits == 0, "an unchanged announcement wrote to the row"


# ── the agent half ────────────────────────────────────────────────────


def test_the_agent_announces_from_the_one_state_write_path():
    """A link the SSE stream missed is found by the reconciler; announcing
    only from the SSE branch would miss exactly the case nobody else can
    report (the sidecar's stream has no replay)."""
    assert "_apply_sidecar_state" in _BAILEYS_SRC
    body = _BAILEYS_SRC.split("def _apply_sidecar_state", 1)[1].split("\n    async def ", 1)[0]
    assert "self._announce_link()" in body, (
        "the link announcement is not on the one state write path"
    )


def test_the_announcement_is_deduped_and_best_effort():
    assert "_last_link_announced" in _BAILEYS_SRC
    body = _BAILEYS_SRC.split("async def _post_link_announce", 1)[1][:2000]
    assert "X-Agent-Key" in body
    assert "except Exception" in body, (
        "a platform that has not deployed the route yet would break the link"
    )


def test_the_announcement_never_logs_the_number():
    """The self number is data for the platform's own column, never a log
    line."""
    body = _BAILEYS_SRC.split("def _announce_link", 1)[1].split("\n    # ──", 1)[0]
    assert "self._self_e164" in body  # it is SENT
    for line in body.splitlines():
        if "logger." in line:
            assert "self_e164" not in line or "redact" in line


def test_the_platform_logs_a_link_seen_breadcrumb():
    src = (_HERE / "app" / "api" / "agent_setup.py").read_text()
    assert "[wa] link_seen" in src
    assert "self=%s" in src.split("[wa] link_seen", 1)[1][:300]
    # counts and booleans only — never the number itself
    seg = src.split("[wa] link_seen", 1)[1][:400]
    assert "bool(payload.self_e164)" in seg


# ── fix lane A: deferred is not the same as unbounded ─────────────────
#
# The poll runs every 2 s per pairing client. Deferring the persist replaced
# "a write on the response path" with "a request-detached task holding its OWN
# platform DB session, on every tick, including the ~99% where nothing
# changed" — an authenticated user could hold platform connections by polling.


@pytest.fixture
def coalesce(monkeypatch):
    from app.api import agent_setup as mod

    mod._wa_snapshot_written.clear()
    mod._wa_snapshot_inflight.clear()
    state = {"spawned": [], "persisted": [], "release": asyncio.Event()}

    def fake_spawn(coro, name=None):
        state["spawned"].append(name)
        return asyncio.get_event_loop().create_task(coro)

    async def fake_persist(user_id, snapshot, *, where="X"):
        state["persisted"].append(dict(snapshot))
        await state["release"].wait()
        return True

    monkeypatch.setattr(mod, "_spawn_bg", fake_spawn)
    monkeypatch.setattr(mod, "_wa_persist_snapshot", fake_persist)
    try:
        yield mod, state
    finally:
        state["release"].set()
        mod._wa_snapshot_written.clear()
        mod._wa_snapshot_inflight.clear()


async def _poll(mod, snapshot, monkeypatch):
    async def fake_proxy(method, path, user_id, db, **kw):
        return snapshot

    monkeypatch.setattr(mod, "_agent_qr_proxy", fake_proxy)
    return await mod.whatsapp_qr_status(current_user=_User(), db=_Db())


@pytest.mark.asyncio
async def test_an_unchanged_snapshot_spawns_nothing(coalesce, monkeypatch):
    mod, state = coalesce
    snap = {"session_status": "linking", "self_e164": None}

    await _poll(mod, snap, monkeypatch)
    state["release"].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(state["spawned"]) == 1

    for _ in range(5):
        await _poll(mod, dict(snap), monkeypatch)
        await asyncio.sleep(0)
    assert len(state["spawned"]) == 1, (
        f"{len(state['spawned'])} detached tasks for six identical polls — "
        "each one opens its own platform DB session"
    )


@pytest.mark.asyncio
async def test_a_changed_snapshot_is_still_written(coalesce, monkeypatch):
    mod, state = coalesce
    state["release"].set()

    await _poll(mod, {"session_status": "linking", "self_e164": None}, monkeypatch)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await _poll(mod, {"session_status": "linked", "self_e164": "+1"}, monkeypatch)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(state["spawned"]) == 2
    assert [p["session_status"] for p in state["persisted"]] == ["linking", "linked"]


@pytest.mark.asyncio
async def test_concurrent_polls_collapse_to_one_writer(coalesce, monkeypatch):
    """Two polls 2 s apart also RACED each other and the agent's own
    `/internal/whatsapp-link` announce on the same row."""
    mod, state = coalesce
    snap = {"session_status": "linking", "self_e164": None}

    await _poll(mod, snap, monkeypatch)
    await asyncio.sleep(0)
    assert len(state["persisted"]) == 1
    # The first write is still in flight (never released); a second poll with a
    # DIFFERENT snapshot must not open a second session behind it.
    await _poll(mod, {"session_status": "linked", "self_e164": "+1"}, monkeypatch)
    await asyncio.sleep(0)
    assert len(state["spawned"]) == 1, "a second writer started under the first"


@pytest.mark.asyncio
async def test_a_failed_write_is_retried_by_the_next_poll(coalesce, monkeypatch):
    """The fingerprint is recorded only for a write that reported success —
    otherwise one swallowed failure makes the row permanently stale."""
    mod, state = coalesce
    calls = []

    async def failing(user_id, snapshot, *, where="X"):
        calls.append(snapshot)
        return False

    monkeypatch.setattr(mod, "_wa_persist_snapshot", failing)
    snap = {"session_status": "linking", "self_e164": None}
    for _ in range(3):
        await _poll(mod, dict(snap), monkeypatch)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert len(calls) == 3, "a write that failed was remembered as done"


def test_the_persist_reports_whether_it_wrote():
    """`_wa_claim_snapshot_persist`'s memory is only safe because the persist
    distinguishes success from a swallowed failure."""
    import inspect
    from app.api import agent_setup as mod

    src = inspect.getsource(mod._wa_persist_snapshot)
    assert "return True" in src and "return False" in src


def test_the_announced_path_is_a_route_the_platform_actually_registers():
    """The agent builds this URL by string concatenation and swallows a 404 as
    "the platform has not deployed the route yet" — so a prefix drift makes the
    announcement permanently and silently inert, which is the exact failure
    mode F4b exists to remove. Read the suffix out of the SOURCE (never
    retyped) and resolve it against the real router.
    """
    import re as _re

    from app.api.agent_setup import router as _router

    m = _re.search(
        r'url = f"\{base\}(/[A-Za-z0-9/_\-]+)"',
        _BAILEYS_SRC.split("async def _post_link_announce", 1)[1][:800],
    )
    assert m, "the announcement no longer builds a literal path"
    suffix = m.group(1)

    posts = {
        r.path for r in _router.routes if "POST" in getattr(r, "methods", set())
    }
    assert suffix in posts, (
        f"the agent announces to {suffix}, which the platform's agent-setup "
        f"router does not register as a POST route: {sorted(posts)[:6]}…"
    )

    # And the base the agent joins it onto already carries the API prefix, so
    # the full URL is <platform>/api/agent-setup/... — the router's own mount.
    from app.config import settings as _s

    assert (_s.platform_api_url or "").rstrip("/").endswith(_s.api_prefix), (
        "platform_api_url no longer ends with the API prefix — the announced "
        "URL would miss the mount point"
    )
