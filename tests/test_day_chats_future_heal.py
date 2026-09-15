"""A heal that cannot run is a missing repair, never a broken day index (D5/D10).

`list_day_chats` healed future-dated day chats INLINE, on the request's own
session, inside a bare `except Exception: logger.warning(...)` with no
rollback (day_chats.py:606-641). On 2026-09-14 the merge branch updated
`messages` and `conversations`, then deleted a day chat that a
`context_budget_logs` row still referenced. Postgres aborted the transaction;
the except swallowed it; the enclosing `for dc in list(day_chats)` kept
dereferencing ORM attributes on an aborted session; every subsequent statement
raised PendingRollbackError. Result: 65 consecutive 500s, and no day index for
any client for three hours.

D5 moves the heal into `day_chat_rebucket.rebucket_user_days` called in a
SEPARATE session, so "a heal failure must never poison the request" is a
property of the shape rather than of remembering a rollback. This file pins the
route's side of that, plus the two cheap gates that keep it off the hot path
(`before is None`, and a 60 s per-user cooldown).

Lane: RUN_MODE=agent — day_chats / conversations / messages /
context_budget_logs are all AGENT_ONLY, and the route SELECTs them. Registered
in tests/COVERAGE_DEBT.txt with the literal `# agent-mode` marker.

Local run (from backend/):
    RUN_MODE=agent PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_day_chats_future_heal.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import json
import uuid
from datetime import date as Date, datetime, timedelta, timezone

import pytest

TORONTO = "America/Toronto"


def _svc():
    return pytest.importorskip(
        "app.services.day_chat_rebucket",
        reason="day_chat_rebucket service not landed yet — lane B3 (D5)",
    )


async def _seed(*, user_tz, future_days=1, with_cbl=True):
    """A user, a day chat for today and `future_days` dated ahead of them."""
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    try:
        from app.db.models.day_chat import ContextBudgetLog
    except Exception:  # noqa: BLE001
        ContextBudgetLog = None

    uid = str(uuid.uuid4())
    today = datetime.now(timezone.utc).date()
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"heal-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="Heal", timezone=user_tz))
        db.add(DayChat(id=f"{uid}-today", user_id=uid, local_date=today,
                       timezone=user_tz or "UTC"))
        await db.flush()
        for n in range(future_days):
            dc = f"{uid}-fut{n}"
            db.add(DayChat(id=dc, user_id=uid, local_date=today + timedelta(days=n + 1),
                           timezone="UTC"))
            await db.flush()
            conv = f"{uid}-conv{n}"
            db.add(Conversation(id=conv, user_id=uid, channel="whatsapp", day_chat_id=dc,
                                started_at=datetime.utcnow()))
            db.add(Message(id=f"{uid}-m{n}", conversation_id=conv, day_chat_id=dc,
                           channel="whatsapp", role="user", content="hi",
                           created_at=datetime.utcnow()))
            if with_cbl and ContextBudgetLog is not None:
                db.add(ContextBudgetLog(id=f"{uid}-cbl{n}", user_id=uid, day_chat_id=dc,
                                        created_at=datetime.utcnow()))
        await db.commit()
    return uid


async def _list(uid, *, before=None):
    """The REAL route function, with the platform proxy probe stubbed out —
    this tenant has no managed_containers row and the proxy is not the
    subject."""
    from sqlalchemy import select
    import app.api.day_chats as api
    from app.db.database import async_session_maker
    from app.db.models.user import User

    real = api._get_agent_proxy_info

    async def _no_proxy(*a, **k):
        return None

    api._get_agent_proxy_info = _no_proxy
    try:
        async with async_session_maker() as db:
            user = (await db.execute(select(User).where(User.id == uid))).scalar_one()
            out = await api.list_day_chats(limit=30, before=before, current_user=user, db=db)
            # A SECOND statement on the SAME session: this is the assertion
            # that the heal did not leave the request in PendingRollback.
            await db.execute(select(User).where(User.id == uid))
    finally:
        api._get_agent_proxy_info = real

    if isinstance(out, list):
        return out
    return json.loads(bytes(out.body).decode())


@pytest.fixture
def cooldown_off(monkeypatch):
    """The per-user cooldown is exactly what a test calling the route twice
    in one process would trip."""
    import app.api.day_chats as api

    for name in ("_HEAL_COOLDOWN", "_heal_cooldown"):
        d = getattr(api, name, None)
        if isinstance(d, dict):
            d.clear()
            monkeypatch.setattr(api, name, {}, raising=False)
    yield


# ── the 500 ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_raising_rebucket_still_returns_the_day_list(cooldown_off):
    """The invariant that matters most operationally. The heal is a repair;
    the response is the product."""
    _svc()
    import app.services.day_chat_rebucket as mod
    import app.api.day_chats as api

    uid = await _seed(user_tz=TORONTO)

    async def boom(*a, **k):
        raise RuntimeError("injected heal failure")

    real = mod.rebucket_user_days
    mod.rebucket_user_days = boom
    had = hasattr(api, "rebucket_user_days")
    if had:
        api.rebucket_user_days = boom
    try:
        rows = await _list(uid)
    finally:
        mod.rebucket_user_days = real
        if had:
            api.rebucket_user_days = real

    assert rows, (
        "a failed heal returned an EMPTY day list — the released client's "
        "history-honesty guard reads a 200-with-nothing as data loss"
    )
    assert any(r.get("id", "").endswith("-fut0") for r in rows), (
        "the unhealed rows must still be LISTED; hiding them turns a missing "
        "repair into missing history"
    )


@pytest.mark.asyncio
async def test_two_consecutive_calls_both_succeed(cooldown_off):
    """The 3-hour outage in one line: the second call is the one that proves
    the first did not poison anything process-wide."""
    _svc()
    uid = await _seed(user_tz=TORONTO)
    assert await _list(uid)
    assert await _list(uid)


@pytest.mark.asyncio
async def test_a_context_budget_log_child_does_not_break_the_request(cooldown_off):
    """The exact dependent the shipped heal did not know about."""
    _svc()
    uid = await _seed(user_tz=TORONTO, with_cbl=True)
    rows = await _list(uid)
    assert rows


# ── the gates ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_heal_is_skipped_for_a_paginated_request(cooldown_off, monkeypatch):
    """Future days only ever appear on page 1. Running the heal for a cursor
    fetch spends it on the scroll."""
    _svc()
    import app.services.day_chat_rebucket as mod

    uid = await _seed(user_tz=TORONTO)
    calls = []

    async def spy(*a, **k):
        calls.append(1)
        return mod.RebucketResult(changed=False, moved_messages=0, moved_conversations=0,
                                  moved_cbls=0, days_created=[], days_deleted=[],
                                  days_retained={}, per_date_errors={}, before_after=[])

    monkeypatch.setattr(mod, "rebucket_user_days", spy)
    import app.api.day_chats as api
    if hasattr(api, "rebucket_user_days"):
        monkeypatch.setattr(api, "rebucket_user_days", spy)

    await _list(uid, before=(datetime.now(timezone.utc).date()).isoformat())
    assert calls == []


@pytest.mark.asyncio
async def test_the_heal_is_skipped_when_the_user_has_no_timezone(cooldown_off, monkeypatch):
    """And this is the ordering trap D5 calls out: the stranded day exists
    BECAUSE the tz was NULL, so a heal gated on a known tz cannot repair the
    window that created the problem. It must not CRASH either — that is all
    this case asserts; the repair arrives with the tz (lane B1's seed)."""
    _svc()
    import app.services.day_chat_rebucket as mod

    uid = await _seed(user_tz=None)
    calls = []

    async def spy(*a, **k):
        calls.append(1)
        raise AssertionError("rebucket must not run with an unknown tz")

    monkeypatch.setattr(mod, "rebucket_user_days", spy)
    import app.api.day_chats as api
    if hasattr(api, "rebucket_user_days"):
        monkeypatch.setattr(api, "rebucket_user_days", spy)

    rows = await _list(uid)
    assert rows
    assert calls == []


@pytest.mark.asyncio
async def test_the_per_user_cooldown_runs_the_heal_once(cooldown_off, monkeypatch):
    """A retrying client calls this route on every focus. Six heals per action
    is how a repair becomes the outage."""
    _svc()
    import app.api.day_chats as api

    if not any(hasattr(api, n) for n in ("_HEAL_COOLDOWN", "_heal_cooldown")):
        pytest.skip("heal cooldown not landed yet — lane B3 (D5)")

    import app.services.day_chat_rebucket as mod
    uid = await _seed(user_tz=TORONTO)
    calls = []

    async def spy(*a, **k):
        calls.append(1)
        return mod.RebucketResult(changed=False, moved_messages=0, moved_conversations=0,
                                  moved_cbls=0, days_created=[], days_deleted=[],
                                  days_retained={}, per_date_errors={}, before_after=[])

    monkeypatch.setattr(mod, "rebucket_user_days", spy)
    if hasattr(api, "rebucket_user_days"):
        monkeypatch.setattr(api, "rebucket_user_days", spy)

    await _list(uid)
    await _list(uid)
    assert len(calls) == 1, f"the heal ran {len(calls)} times inside the cooldown"


# ── the counter ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_future_dated_day_chats_is_counted(cooldown_off):
    """D10. The number that would have made this incident visible on the day
    it happened rather than three days later."""
    hs = pytest.importorskip(
        "app.services.health_signals",
        reason="health_signals not landed yet — lane B3 (D10)",
    )
    _svc()
    hs.reset_for_tests()

    uid = await _seed(user_tz=TORONTO, future_days=3)
    await _list(uid)

    # The monotonic COUNTER records that the rows existed at all …
    assert hs.get("future_dated_day_chats_seen") >= 3, (
        f"saw {hs.get('future_dated_day_chats_seen')} future-dated day chats, "
        "seeded 3"
    )
    # … and the GAUGE reads the current state, re-measured after the heal —
    # a repaired tenant must not page for an hour on a stale positive.
    assert hs.get("future_dated_day_chats") == 0, (
        f"gauge still reads {hs.get('future_dated_day_chats')} after the heal "
        "repaired every future-dated day"
    )
