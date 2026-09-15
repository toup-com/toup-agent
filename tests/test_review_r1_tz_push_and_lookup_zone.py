"""Review round 1 (2026-09-14) — the two findings that touched production data
paths, pinned by behaviour.

P0  A timezone push must never RECREATE a container. `_push_tz_to_agent`
    fires from PATCH /auth/profile, which the new mobile build calls on every
    launch; `update_container_env` routes a pool member through the warm
    /v1/pool/refresh-config and every other container through
    `provision_container(recreate=True)` — a force-remove and re-run that
    drops the WhatsApp socket, loses the in-flight turn and (R40) can bind a
    different database. Two guards, both pinned here: the caller skips named
    tenants, and the service refuses the recreate under `pool_only=True`.

P1  The channel session lookup must resolve TODAY with the SAME zone the
    runner buckets the turn in, and must never create a day row. It used to
    call `resolve_day_chat_id_for_now(tz_override=None)` — User.timezone or
    UTC, and a CREATE — while the runner went on to sources 3/4 (bind payload,
    owner's phone). A phone-only tenant got an empty future-dated DayChat and a
    fresh Conversation on every inbound message.

Also pins the counter placements the persist/orphan alerts depend on.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_review_r1_tz_push_and_lookup_zone.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.agent.channels.base import BaseChannel, ChannelType, InboundMessage

_BACKEND = Path(__file__).resolve().parents[1]
_APP = _BACKEND / "app"

# UTC-11 and UTC+14: their local dates differ at EVERY instant, so a case
# built on the pair is deterministic whatever the wall clock says.
TZ_ROW = "Pacific/Pago_Pago"
TZ_RUNNER = "Pacific/Kiritimati"
CHAT = "+14155552671"


# ── fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def day_tables():
    from sqlalchemy import inspect as sa_inspect
    from app.db.database import engine
    from app.db.models.base import Base

    names = ["day_chats", "conversations", "messages"]
    async with engine.begin() as conn:
        for n in names:
            table = Base.metadata.tables.get(n)
            if table is None:
                continue
            try:
                await conn.run_sync(table.create, checkfirst=True)
            except Exception:
                pass
    async with engine.connect() as conn:
        existing = await conn.run_sync(lambda c: set(sa_inspect(c).get_table_names()))
    missing = [n for n in names if n not in existing]
    if missing:
        pytest.skip(f"cannot create {missing} on this backend")
    yield


@pytest.fixture(autouse=True)
def _clear_caches():
    from app.agent._user_tz_cache import _USER_TZ_CACHE
    from app.agent import _day_chat_cache

    _USER_TZ_CACHE.clear()
    yield
    _USER_TZ_CACHE.clear()
    try:
        _day_chat_cache._CACHE.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


async def _mk_user(tz=TZ_ROW):
    from app.db.database import async_session_maker
    from app.db.models import User

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"r1-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="R1", timezone=tz))
        await db.commit()
    return uid


async def _seed_today_in(uid, tz_name, *, chat_id=CHAT):
    """A DayChat for the user's CURRENT local date in `tz_name`, holding one
    active WhatsApp Conversation for `chat_id`."""
    from app.agent.day_chat_resolver import resolve_local_date
    from app.db.database import async_session_maker
    from app.db.models import Conversation
    from app.db.models.day_chat import DayChat

    local_date, _ = resolve_local_date(datetime.now(timezone.utc), tz_name)
    dc_id, conv_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(DayChat(id=dc_id, user_id=uid, local_date=local_date, timezone=tz_name))
        await db.flush()
        db.add(Conversation(
            id=conv_id, user_id=uid, channel="whatsapp", day_chat_id=dc_id,
            is_active=True, started_at=datetime.utcnow(), updated_at=datetime.utcnow(),
            metadata_json=f'{{"channel_chat_id": "{chat_id}"}}',
        ))
        await db.commit()
    return dc_id, conv_id


async def _count_days(uid):
    from sqlalchemy import func, select
    from app.db.database import async_session_maker
    from app.db.models.day_chat import DayChat

    async with async_session_maker() as db:
        return (await db.execute(
            select(func.count()).select_from(DayChat).where(DayChat.user_id == uid)
        )).scalar_one()


# ── P0: a timezone push never recreates ──────────────────────────────


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeDB:
    def __init__(self, row):
        self._row = row
        self.commits = 0

    async def execute(self, *_a, **_k):
        return _FakeResult(self._row)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_pool_only_push_leaves_a_named_tenant_untouched(monkeypatch):
    from app.services import docker_host_service as dhs

    recreated = []

    async def _provision(*a, **k):
        recreated.append(k)
        return "RECREATED"

    monkeypatch.setattr(dhs, "provision_container", _provision)
    named = SimpleNamespace(container_name="toup-agent-named-tenant", user_id="u" * 36)
    db = _FakeDB(named)

    out = await dhs._update_container_env(db, "u" * 36, MagicMock(), pool_only=True)

    assert out is named, "pool_only must hand back the container untouched"
    assert recreated == [], "a pool_only push reached provision_container(recreate=True)"


@pytest.mark.asyncio
async def test_a_plain_push_still_recreates_a_named_tenant(monkeypatch):
    """ANTI-VACUITY for the case above: the named-tenant env push is what
    `recreate=True` exists for (Stripe activation, bundle changes). The new
    flag narrows one caller; it must not disable the path."""
    from app.services import docker_host_service as dhs

    recreated = []

    async def _provision(*a, **k):
        recreated.append(k)
        return "RECREATED"

    monkeypatch.setattr(dhs, "provision_container", _provision)
    named = SimpleNamespace(container_name="toup-agent-named-tenant", user_id="u" * 36)

    out = await dhs._update_container_env(_FakeDB(named), "u" * 36, MagicMock())

    assert out == "RECREATED"
    assert recreated and recreated[0].get("recreate") is True


@pytest.mark.asyncio
async def test_pool_only_does_not_short_circuit_a_pool_member(monkeypatch):
    """The flag must only bite on NAMED containers — a pool member still has
    to reach the warm refresh branch, or the seed never arrives."""
    from app.services import docker_host_service as dhs

    reached = []

    class _Stop(Exception):
        pass

    def _boom(*a, **k):
        reached.append(True)
        raise _Stop()

    # The pool branch's first call is `_build_bind_payload`; trip on it.
    import app.services.pool_service as ps
    monkeypatch.setattr(ps, "_build_bind_payload", _boom)
    pool = SimpleNamespace(container_name="toup-agent-pool-18", user_id="u" * 36)

    with pytest.raises(_Stop):
        await dhs._update_container_env(_FakeDB(pool), "u" * 36, MagicMock(), pool_only=True)
    assert reached, "a pool member never reached the refresh branch under pool_only"


def test_the_service_gate_uses_the_same_prefix_as_the_pool_branch():
    src = (_APP / "services" / "docker_host_service.py").read_text()
    assert '_POOL_NAME_PREFIX = "toup-agent-pool-"' in src
    gate = src.index("if pool_only and not (container.container_name or \"\").startswith(_POOL_NAME_PREFIX)")
    branch = src.index('if container.container_name and container.container_name.startswith("toup-agent-pool-")')
    assert gate < branch, "the pool_only gate must sit ABOVE the pool branch, not after it"


@pytest.mark.asyncio
async def test_auth_tz_push_skips_a_named_tenant_and_marks_a_pool_push(monkeypatch):
    """The caller-side guard. Real rows in the platform tables; the bridge
    call is recorded, never made."""
    from app.api import auth as auth_mod
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig, ManagedContainer, User
    from app.services import docker_host_service as dhs

    calls = []

    async def _record(db, user_id, cfg, **kw):
        calls.append((user_id, kw))
        return None

    monkeypatch.setattr(dhs, "update_container_env", _record)

    async def _mk(container_name, port):
        uid = str(uuid.uuid4())
        async with async_session_maker() as db:
            db.add(User(id=uid, email=f"r1auth-{uuid.uuid4().hex[:8]}@example.com",
                        hashed_password="x", name="A", timezone="America/Toronto"))
            await db.flush()
            # host_port is UNIQUE on the platform table.
            db.add(ManagedContainer(user_id=uid, container_name=container_name,
                                    host_port=port, db_name="x", status="running"))
            db.add(AgentConfig(user_id=uid))
            await db.commit()
        return uid

    base = 40000 + (uuid.uuid4().int % 10000)
    named_uid = await _mk("toup-agent-named-1", base)
    pool_uid = await _mk("toup-agent-pool-18", base + 1)

    auth_mod._push_tz_to_agent(named_uid)
    auth_mod._push_tz_to_agent(pool_uid)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if any(c[0] == pool_uid for c in calls):
            break

    assert all(c[0] != named_uid for c in calls), (
        "a profile PATCH on a named tenant reached update_container_env — "
        "that is provision_container(recreate=True) on a live container"
    )
    pool_calls = [c for c in calls if c[0] == pool_uid]
    assert pool_calls, "the pool tenant's push never fired"
    assert pool_calls[0][1].get("pool_only") is True


def test_the_tz_push_is_paced_across_users():
    src = (_APP / "api" / "auth.py").read_text()
    assert re.search(r"_TZ_PUSH_GATE\s*=\s*asyncio\.Semaphore\(\s*[12]\s*\)", src), (
        "the cross-user pacing gate is gone — a release day queues every tenant's "
        "bind at once against a single-threaded bridge"
    )
    fn = src.split("def _push_tz_to_agent(")[1].split("\n@router")[0]
    assert "async with _TZ_PUSH_GATE" in fn


# ── P1: the lookup shares the runner's zone and never creates a day ──


@pytest.mark.asyncio
async def test_the_lookup_never_creates_a_day_row(day_tables):
    from app.agent.channels.shared.message_handler import _resolve_session_id

    uid = await _mk_user()
    assert await _count_days(uid) == 0

    got = await _resolve_session_id(uid, ChannelType.WHATSAPP, CHAT, {})

    assert got is None
    assert await _count_days(uid) == 0, (
        "the session lookup minted a DayChat — that is the empty, future-dated "
        "day of the 2026-09-14 incident, created on every inbound message"
    )


@pytest.mark.asyncio
async def test_the_lookup_uses_the_zone_it_is_handed(day_tables):
    """Today in the RUNNER's zone holds the conversation; today in the users
    row's zone is a different date. The lookup must find it through the
    resolver and miss it through the row."""
    from app.agent.channels.shared.message_handler import _resolve_session_id

    uid = await _mk_user(TZ_ROW)
    _, conv = await _seed_today_in(uid, TZ_RUNNER)

    async def _runner_zone(_db):
        return TZ_RUNNER

    via_resolver = await _resolve_session_id(uid, ChannelType.WHATSAPP, CHAT, {}, _runner_zone)
    via_row = await _resolve_session_id(uid, ChannelType.WHATSAPP, CHAT, {})

    assert via_resolver == conv, "the lookup ignored the zone it was handed"
    assert via_row is None, (
        "ANTI-VACUITY: the row's zone names a different date, so the row path "
        "must miss — if it hits, the two zones do not disagree and the case proves nothing"
    )


@pytest.mark.asyncio
async def test_a_new_local_day_is_a_cache_miss_but_a_degraded_day_is_not(day_tables):
    from app.agent.channels.shared import message_handler as mh

    uid = await _mk_user(TZ_ROW)
    dc_today, conv = await _seed_today_in(uid, TZ_RUNNER)

    async def _runner_zone(_db):
        return TZ_RUNNER

    # Cached under a day that is NOT today's id → miss, then re-resolved from the DB.
    cache = {("whatsapp", CHAT): ("some-old-day-id", "conv-from-yesterday")}
    got = await mh._resolve_session_id(uid, ChannelType.WHATSAPP, CHAT, cache, _runner_zone)
    assert got == conv and cache[("whatsapp", CHAT)] == (dc_today, conv)

    # Cached under today's id → hit without a query.
    got = await mh._resolve_session_id(uid, ChannelType.WHATSAPP, CHAT, cache, _runner_zone)
    assert got == conv

    # Day resolution DEGRADED → keep the cache; do not mint a session per turn.
    async def _boom(_db, _uid, _tz):
        return mh._DAY_UNRESOLVED

    orig = mh._todays_day_chat_id
    mh._todays_day_chat_id = _boom
    try:
        got = await mh._resolve_session_id(uid, ChannelType.WHATSAPP, CHAT, cache, _runner_zone)
    finally:
        mh._todays_day_chat_id = orig
    assert got == conv


class _FakeChannel(BaseChannel):
    def __init__(self):
        super().__init__(ChannelType.WHATSAPP)
        self.sent = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_text(self, chat_id, text, parse_mode=None):
        self.sent.append((chat_id, text))

    async def send_typing(self, chat_id):
        pass


@pytest.mark.asyncio
async def test_the_handler_asks_the_runner_for_the_zone(day_tables):
    """End to end through `make_channel_handler`: the runner's own
    `_resolve_effective_tz` is what the lookup consults, with the runner's
    argument shape, and the turn is threaded into today's conversation."""
    from app.agent.channels.shared import make_channel_handler

    uid = await _mk_user(TZ_ROW)
    _, conv = await _seed_today_in(uid, TZ_RUNNER)

    seen = []

    async def _resolve_effective_tz(db, user_id, client_tz, channel):
        seen.append((user_id, client_tz, channel))
        return TZ_RUNNER

    ran = []

    async def _run(**kw):
        ran.append(kw)
        resp = MagicMock()
        resp.text = "ok"
        resp.session_id = kw.get("session_id") or str(uuid.uuid4())
        resp.persisted = {}
        resp.day_chat_id = None
        resp.tokens_total = 0
        resp.tool_calls = []
        resp.processing_time_ms = 1
        return resp

    runner = MagicMock()
    runner._resolve_effective_tz = _resolve_effective_tz
    runner.run = AsyncMock(side_effect=_run)

    handler = make_channel_handler(channel=_FakeChannel(), agent_runner=runner, user_id=uid)
    await handler(InboundMessage(channel=ChannelType.WHATSAPP, channel_user_id=CHAT,
                                 channel_chat_id=CHAT, text="hi", media_paths=[]))

    assert seen and seen[0] == (uid, None, "whatsapp"), seen
    assert ran and ran[0]["session_id"] == conv, "the turn was not threaded into today's conversation"


@pytest.mark.asyncio
async def test_a_stub_runner_without_the_method_falls_back_to_the_row(day_tables):
    """`MagicMock().anything` is truthy and callable; the handler must not
    mistake it for the resolver. Awaiting it would raise on every lookup and
    the row's zone would be silently replaced by a degraded miss."""
    from app.agent.channels.shared import make_channel_handler

    uid = await _mk_user(TZ_ROW)
    _, conv = await _seed_today_in(uid, TZ_ROW)  # today in the ROW's zone
    ran = []

    async def _run(**kw):
        ran.append(kw)
        resp = MagicMock()
        resp.text = "ok"
        resp.session_id = kw.get("session_id") or str(uuid.uuid4())
        resp.persisted = {}
        resp.day_chat_id = None
        resp.tokens_total = 0
        resp.tool_calls = []
        resp.processing_time_ms = 1
        return resp

    runner = MagicMock()  # `_resolve_effective_tz` is a MagicMock attribute
    runner.run = AsyncMock(side_effect=_run)
    handler = make_channel_handler(channel=_FakeChannel(), agent_runner=runner, user_id=uid)
    await handler(InboundMessage(channel=ChannelType.WHATSAPP, channel_user_id=CHAT,
                                 channel_chat_id=CHAT, text="hi", media_paths=[]))
    assert ran and ran[0]["session_id"] == conv, "the row-zone fallback did not resolve today's conversation"


# ── counter placement pins ───────────────────────────────────────────


def test_claimed_is_counted_after_the_kind_gate():
    src = (_APP / "agent" / "channels" / "whatsapp_baileys.py").read_text()
    gate = src.index("inbound.unsupported_kind")
    claimed = src.index('_signal("channel_events_claimed")')
    assert claimed > gate, (
        "channel_events_claimed is counted before the non-text gate — a sticker "
        "claims and never dispatches, and the orphan alert pages forever"
    )


def test_persisted_is_counted_for_whatsapp_alone():
    src = (_APP / "agent" / "channels" / "shared" / "message_handler.py").read_text()
    idx = src.index('_signal("channel_events_persisted")')
    window = src[max(0, idx - 200):idx]
    assert "ChannelType.WHATSAPP" in window, (
        "channel_events_persisted counts every channel while channel_events_claimed "
        "is WhatsApp-only — a Telegram turn hides a real WhatsApp orphan"
    )


def test_ws_chat_does_not_count_media_expected():
    src = (_APP / "api" / "ws_chat.py").read_text()
    assert re.search(r'incr\(\s*"media_expected"\s*\)', src) is None, (
        "media_expected is counted in ws_chat AND in _run_inner for the same card — "
        "the persist-gap alert fires forever for anyone who plays a song"
    )
    runner = (_APP / "agent" / "agent_runner.py").read_text()
    assert re.search(r'incr\(\s*"media_expected"\s*\)', runner), "the runner-side count is gone"


# ── the P3s: task strong refs, the bounded lock dict ─────────────────


@pytest.mark.asyncio
async def test_the_tz_push_task_is_held_until_it_finishes(monkeypatch):
    """asyncio keeps only a weak reference to a task; a fire-and-forget
    `create_task` with no strong ref can be collected mid-await. The push is
    parked in `_TZ_PUSH_TASKS` while it runs and dropped when done."""
    from app.api import auth as auth_mod

    gate = asyncio.Event()

    class _Blocking:
        async def __aenter__(self):
            await gate.wait()
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(auth_mod, "_TZ_PUSH_GATE", _Blocking())
    before = set(auth_mod._TZ_PUSH_TASKS)
    auth_mod._push_tz_to_agent("u" * 36)
    await asyncio.sleep(0)
    held = set(auth_mod._TZ_PUSH_TASKS) - before
    assert len(held) == 1 and not next(iter(held)).done(), "the push task is not held while pending"
    gate.set()
    await asyncio.gather(*held)
    await asyncio.sleep(0)
    assert not (set(auth_mod._TZ_PUSH_TASKS) & held), "a finished push task is kept forever"


def _closure_var(fn, name):
    """Find a closure cell by name, one level down through nested closures."""
    for f in [fn] + [c.cell_contents for c in (fn.__closure__ or ()) if callable(c.cell_contents)]:
        if f.__code__.co_freevars and name in f.__code__.co_freevars:
            return f.__closure__[f.__code__.co_freevars.index(name)].cell_contents
    raise AssertionError(f"{name} not found in the handler's closures")


@pytest.mark.asyncio
async def test_the_per_chat_lock_dict_is_bounded(day_tables):
    """A Discord/Slack tenant sees an unbounded set of chat ids over a
    process lifetime; idle locks past 512 entries are dropped and simply
    re-minted on the next turn."""
    from app.agent.channels.shared import make_channel_handler

    uid = await _mk_user(TZ_ROW)

    async def _run(**kw):
        resp = MagicMock()
        resp.text = "ok"
        resp.session_id = str(uuid.uuid4())
        resp.persisted = {}
        resp.day_chat_id = None
        resp.tokens_total = 0
        resp.tool_calls = []
        resp.processing_time_ms = 1
        return resp

    runner = MagicMock()
    runner.run = AsyncMock(side_effect=_run)
    handler = make_channel_handler(channel=_FakeChannel(), agent_runner=runner, user_id=uid)
    locks = _closure_var(handler, "session_locks")
    for i in range(600):
        await handler(InboundMessage(channel=ChannelType.WHATSAPP, channel_user_id=f"+1555{i:07d}",
                                     channel_chat_id=f"+1555{i:07d}", text="hi", media_paths=[]))
    assert len(locks) <= 512, f"{len(locks)} per-chat locks retained — the dict is unbounded again"
    assert len(locks) >= 256, "the prune dropped LIVE bookkeeping wholesale, not just the idle overflow"
