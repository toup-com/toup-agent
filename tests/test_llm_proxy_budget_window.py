"""Per-tenant model budget: the rolling monthly window and the typed refusal.

Spec A4 of the 2026-09-28 document-analysis incident. A tenant's agent was
refused every OpenAI call because the proxy's "monthly" cap summed spend since
activation — just over the 1000c budget, most of it months old — and the
refusal was a bare 429 {"detail": "Monthly openai budget exceeded"} that the
OpenAI SDK retried as a rate limit and the platform never logged. These tests
pin the window (`budget_period_bounds`, whole-second aligned), the gate
(`_check_budget` on real rows), the typed refusal (`_raise_budget_exceeded`:
its detail carries no spend or budget — those are in the log line only) and
what `/api/llm/usage` reports.

Lane: platform
(users and agent_configs are SHARED tables but llm_proxy_events is
PLATFORM_ONLY, so this file must stay out of the RUN_MODE=agent sweep.)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import types
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import openai
import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy import func, select

from app.api import llm_proxy as lp
from app.config import settings
from app.services import budget_refusal

REASON = "monthly_model_budget_exceeded"
ANCHOR = datetime(2026, 5, 24, 19, 10)   # a bundle_started_at (naive UTC, like the column)
NOW = datetime(2026, 9, 29, 1, 30)       # the incident probe time
WINDOW = (datetime(2026, 9, 24, 19, 10), datetime(2026, 10, 24, 19, 10))
WINDOW_ISO = ("2026-09-24T19:10:00+00:00", "2026-10-24T19:10:00+00:00")
SPLIT_EVENTS = [(datetime(2026, 9, 20, 12, 0), 975),          # before the rolling window
                (datetime(2026, 9, 28, 22, 0), 31.2575)]      # inside it; lifetime 1006.2575c
_REFUSED_LINE = re.compile(r"spent=(\d+\.\d\d) budget=(\d+) ")


class _Clock(datetime):
    """Frozen clock for everything in llm_proxy that asks `datetime` for now."""
    value = NOW

    @classmethod
    def utcnow(cls):
        return cls.value

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.value
        return cls.value.replace(tzinfo=timezone.utc).astimezone(tz)


@pytest.fixture(autouse=True)
def _proxy_state(monkeypatch):
    lp._budget_cache.clear()
    lp._rate_windows.clear()
    lp._refusal_logged_at.clear()
    _Clock.value = NOW
    monkeypatch.setattr(lp, "datetime", _Clock)
    monkeypatch.setattr(settings, "bundle_budget_rolling_month", True)
    monkeypatch.setattr(settings, "credit_enforcement_enabled", False)
    yield
    lp._budget_cache.clear()
    lp._rate_windows.clear()
    lp._refusal_logged_at.clear()


# ── budget_period_bounds (pure) ──────────────────────────────────────


def _cfg(**kw):
    base = dict(user_id="u-pure-0000", bundle_period_start=None,
                bundle_period_end=None, bundle_started_at=ANCHOR)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize("anchor, now, start, end", [
    # the incident probe time: the window since the latest anniversary
    (ANCHOR, NOW, *WINDOW),
    # a 31st anchor is clamped in short months and never drifts to the 28th
    (datetime(2026, 1, 31, 23, 30), datetime(2026, 2, 28, 23, 29, 59),
     datetime(2026, 1, 31, 23, 30), datetime(2026, 2, 28, 23, 30)),
    (datetime(2026, 1, 31, 23, 30), datetime(2026, 2, 28, 23, 30),
     datetime(2026, 2, 28, 23, 30), datetime(2026, 3, 31, 23, 30)),
    (datetime(2026, 1, 31, 23, 30), datetime(2026, 3, 31, 23, 30),
     datetime(2026, 3, 31, 23, 30), datetime(2026, 4, 30, 23, 30)),
    (datetime(2026, 1, 31, 10), datetime(2026, 4, 15),
     datetime(2026, 3, 31, 10), datetime(2026, 4, 30, 10)),
    (datetime(2024, 1, 31, 10), datetime(2024, 3, 1),           # leap February
     datetime(2024, 2, 29, 10), datetime(2024, 3, 31, 10)),
    (datetime(2025, 12, 31, 23), datetime(2026, 2, 10),         # across a year
     datetime(2026, 1, 31, 23), datetime(2026, 2, 28, 23)),
    # a leap-day anchor: 2025-02-28, then back on the 29th
    (datetime(2024, 2, 29, 12), datetime(2025, 3, 1),
     datetime(2025, 2, 28, 12), datetime(2025, 3, 29, 12)),
    (datetime(2024, 2, 29, 12), datetime(2025, 3, 29, 12),
     datetime(2025, 3, 29, 12), datetime(2025, 4, 29, 12)),
    # the anniversary opens the new window; one microsecond earlier is the old one
    (ANCHOR, datetime(2026, 9, 24, 19, 10), *WINDOW),
    (ANCHOR, datetime(2026, 9, 24, 19, 9, 59, 999999),
     datetime(2026, 8, 24, 19, 10), datetime(2026, 9, 24, 19, 10)),
    # first month, and a replica whose clock is behind the anchor
    (datetime(2026, 9, 20, 8), NOW, datetime(2026, 9, 20, 8), datetime(2026, 10, 20, 8)),
    (datetime(2026, 9, 29, 2), NOW, datetime(2026, 9, 29, 2), datetime(2026, 10, 29, 2)),
])
def test_rolling_bounds_oracle(anchor, now, start, end):
    got = lp.budget_period_bounds(_cfg(bundle_started_at=anchor), now=now)
    assert got == (start, end)
    assert got[0].tzinfo is None and got[1].tzinfo is None


def test_bounds_normalise_aware_and_mixed_inputs():
    toronto = timezone(timedelta(hours=-4))
    aware_anchor = ANCHOR.replace(tzinfo=timezone.utc).astimezone(toronto)   # 15:10-04:00
    aware_now = NOW.replace(tzinfo=timezone.utc).astimezone(toronto)         # 21:30-04:00 on the 28th
    for anchor, now in [(aware_anchor, NOW), (ANCHOR, aware_now), (aware_anchor, aware_now)]:
        got = lp.budget_period_bounds(_cfg(bundle_started_at=anchor), now=now)
        assert got == WINDOW
        assert got[0].tzinfo is None and got[1].tzinfo is None
    live = lp.budget_period_bounds(_cfg(
        bundle_period_start=datetime(2026, 9, 10, tzinfo=timezone.utc),
        bundle_period_end=datetime(2026, 10, 9, 20, tzinfo=toronto)), now=aware_now)
    assert live == (datetime(2026, 9, 10), datetime(2026, 10, 10))
    assert live[0].tzinfo is None and live[1].tzinfo is None


def test_bounds_default_now_is_the_module_clock():
    assert lp.budget_period_bounds(_cfg()) == WINDOW
    _Clock.value = datetime(2026, 10, 24, 19, 10)
    assert lp.budget_period_bounds(_cfg()) == (datetime(2026, 10, 24, 19, 10),
                                               datetime(2026, 11, 24, 19, 10))


def test_kill_switch_restores_the_lifetime_window(monkeypatch):
    monkeypatch.setattr(settings, "bundle_budget_rolling_month", False)
    assert lp.budget_period_bounds(_cfg(), now=NOW) == (ANCHOR, None)
    ended = (datetime(2026, 7, 10), datetime(2026, 8, 10))
    assert lp.budget_period_bounds(
        _cfg(bundle_period_start=ended[0], bundle_period_end=ended[1]), now=NOW) == (ended[0], None)
    live = (datetime(2026, 9, 10), datetime(2026, 10, 10))
    assert lp.budget_period_bounds(
        _cfg(bundle_period_start=live[0], bundle_period_end=live[1]), now=NOW) == live


def test_live_stripe_period_is_unchanged_and_a_stale_one_rolls():
    ps, pe = datetime(2026, 9, 10), datetime(2026, 10, 10)
    assert lp.budget_period_bounds(_cfg(bundle_period_start=ps, bundle_period_end=pe), now=NOW) == (ps, pe)
    # period_end is exclusive: an unrenewed period rolls on from its start
    assert lp.budget_period_bounds(
        _cfg(bundle_period_start=ps, bundle_period_end=pe), now=pe) == (pe, datetime(2026, 11, 10))
    # ended and never re-stamped (Stripe renewal does not write these columns)
    assert lp.budget_period_bounds(
        _cfg(bundle_period_start=datetime(2026, 6, 10), bundle_period_end=datetime(2026, 7, 10)),
        now=NOW) == (datetime(2026, 9, 10), datetime(2026, 10, 10))
    # founder-admin shape: an end before the start is not a live period
    assert lp.budget_period_bounds(
        _cfg(bundle_period_start=datetime(2026, 8, 1), bundle_period_end=datetime(2026, 7, 1)),
        now=NOW) == (datetime(2026, 9, 1), datetime(2026, 10, 1))
    # a start without an end anchors the rolling month
    assert lp.budget_period_bounds(_cfg(bundle_period_start=ps), now=NOW) == (ps, pe)
    assert lp.budget_period_bounds(_cfg(bundle_started_at=None), now=NOW) == (None, None)


def test_spend_cache_prunes_expired_entries_and_keeps_the_user_prefix():
    lp._budget_cache["u1:openai:monthly@2026-08-24T19:10:00"] = (0.0, Decimal(5))  # long expired
    lp._set_cached_spend("u1:openai:monthly@2026-09-24T19:10:00", Decimal(1))
    assert list(lp._budget_cache) == ["u1:openai:monthly@2026-09-24T19:10:00"]
    lp._invalidate_cache("u1")
    assert lp._budget_cache == {}


# ── _check_budget on real rows ───────────────────────────────────────


async def _seed(*, role="beta_user", started_at=ANCHOR, period_start=None, period_end=None,
                budget=1000, events=()):
    """A tenant with a working proxy token. `events` are (at, cents) for
    openai or (at, cents, provider)."""
    from app.db import AgentConfig, LLMProxyEvent, User, async_session_maker

    uid, token = str(uuid.uuid4()), f"tok-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"b-{uid[:8]}@example.test", hashed_password="x", role=role))
        await db.flush()
        db.add(AgentConfig(
            user_id=uid, llm_token_hash=lp._hash_token(token), bundle_status="active",
            llm_mode="bundle", bundle_started_at=started_at, bundle_period_start=period_start,
            bundle_period_end=period_end, bundle_openai_budget_cents=budget,
            bundle_openai_api_key="sk-test-outbound",
        ))
        for at, cents, *provider in events:
            db.add(LLMProxyEvent(user_id=uid, provider=(provider or ["openai"])[0],
                                 model="gpt-5.5", endpoint="chat",
                                 cost_cents=Decimal(str(cents)), created_at=at))
        await db.commit()
    return uid, token


async def _config(db, uid):
    from app.db import AgentConfig
    return (await db.execute(select(AgentConfig).where(AgentConfig.user_id == uid))).scalar_one()


async def test_in_window_spend_admitted_and_lifetime_spend_refused_without_clearing_the_cache(monkeypatch):
    from app.db import async_session_maker

    uid, _ = await _seed(events=SPLIT_EVENTS)
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        assert await lp._check_budget(cfg, "openai", db) is None                  # 31.26c this window
        monkeypatch.setattr(settings, "bundle_budget_rolling_month", False)
        # Same user inside the 30 s TTL: only a key that carries the window
        # start can tell the lifetime sum from the cached window sum.
        assert await lp._check_budget(cfg, "openai", db) == "monthly_exceeded"    # over 1000c lifetime
    assert len(lp._budget_cache) == 2 and all(k.startswith(uid) for k in lp._budget_cache)


async def test_admin_is_exempt():
    from app.db import async_session_maker

    uid, _ = await _seed(role="admin", events=[(datetime(2026, 9, 28), 5000)])
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        assert await lp._is_budget_exempt(cfg, db) is True
        assert await lp._check_budget(cfg, "openai", db) is None


async def test_tenants_are_isolated():
    from app.db import async_session_maker

    a, _ = await _seed(events=[(datetime(2026, 9, 28), 10)])
    b, _ = await _seed(events=[(datetime(2026, 9, 28), 5000)])
    async with async_session_maker() as db:
        cfg_a, cfg_b = await _config(db, a), await _config(db, b)
        assert await lp._check_budget(cfg_b, "openai", db) == "monthly_exceeded"
        assert await lp._check_budget(cfg_a, "openai", db) is None
        assert await lp._check_budget(cfg_b, "openai", db) == "monthly_exceeded"   # warm cache, still B's


async def test_window_roll_is_not_masked_by_the_spend_cache():
    from app.db import async_session_maker

    _Clock.value = datetime(2026, 9, 24, 19, 9, 50)          # 10 s before the anniversary
    uid, _ = await _seed(events=[(datetime(2026, 9, 1), 1000)])
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        assert await lp._check_budget(cfg, "openai", db) == "monthly_exceeded"
        _Clock.value = datetime(2026, 9, 24, 19, 10, 5)      # new window, cache still warm
        assert await lp._check_budget(cfg, "openai", db) is None


async def test_a_window_error_falls_back_to_the_legacy_window_and_never_500s(monkeypatch, caplog):
    from app.db import async_session_maker

    def broken(config, now=None):
        raise OverflowError("date value out of range")

    monkeypatch.setattr(lp, "budget_period_bounds", broken)
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    uid, _ = await _seed(events=SPLIT_EVENTS)
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        assert await lp._check_budget(cfg, "openai", db) == "monthly_exceeded"   # since activation
        with pytest.raises(HTTPException) as exc:
            await lp._raise_budget_exceeded(cfg, "openai", db)
    assert exc.value.status_code == 429
    assert exc.value.detail["period_start"] == "2026-05-24T19:10:00+00:00"
    assert exc.value.detail["period_end"] is None and "Retry-After" not in exc.value.headers
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("[budget] window_error")]
    assert lines and f"user={uid[:8]}" in lines[0] and uid not in lines[0]


# ── The typed refusal ────────────────────────────────────────────────


class _FakeRequest:  # tests/test_llm_proxy_rate_limit.py:85
    def __init__(self, body, headers=None):
        self._body, self.headers = body, headers or {}

    async def json(self):
        return self._body


def _refused_log_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "app.api.llm_proxy" and r.getMessage().startswith("[budget] refused")]


_STRIPE_START = NOW - timedelta(days=10)


@pytest.mark.parametrize("seed, retry_after, period", [
    # rolling window, 25 days ahead: capped at 7 days
    (dict(), "604800", WINDOW_ISO),
    # a live Stripe period ending in 2 h, and in 10 s (no 60 s floor)
    (dict(period_start=_STRIPE_START, period_end=NOW + timedelta(hours=2)), "7200",
     ("2026-09-19T01:30:00+00:00", "2026-09-29T03:30:00+00:00")),
    (dict(period_start=_STRIPE_START, period_end=NOW + timedelta(seconds=10)), "10",
     ("2026-09-19T01:30:00+00:00", "2026-09-29T01:30:10+00:00")),
    # kill switch off: the lifetime window has no end, so no Retry-After
    (dict(rolling=False), None, ("2026-05-24T19:10:00+00:00", None)),
], ids=["rolling-far", "stripe-2h", "stripe-10s", "no-end"])
async def test_refusal_shape_via_real_proxy_chat(monkeypatch, caplog, seed, retry_after, period):
    from app.db import LLMProxyEvent, async_session_maker

    seed = dict(seed)
    monkeypatch.setattr(settings, "bundle_budget_rolling_month", seed.pop("rolling", True))
    uid, _ = await _seed(events=[(datetime(2026, 9, 28), 1000.5)], **seed)   # over, inside the window
    upstream = []

    async def fake_chat(body, api_key):
        upstream.append(body)

    monkeypatch.setattr(lp, "_route_chat",
                        lambda model, cfg: (types.SimpleNamespace(name="openai", chat=fake_chat), "k"))
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    async with async_session_maker() as db:
        cfg = await _config(db, uid)

        async def fake_auth(request, db_):
            return cfg

        monkeypatch.setattr(lp, "_auth_agent", fake_auth)
        count = select(func.count()).select_from(LLMProxyEvent)
        rows_before = (await db.execute(count)).scalar_one()
        with pytest.raises(HTTPException) as exc:
            await lp.proxy_chat(_FakeRequest({"model": "gpt-5.5", "messages": []}), db=db)
        rows_after = (await db.execute(count)).scalar_one()

    e = exc.value
    assert e.status_code == 429
    assert upstream == [] and rows_before == rows_after          # no upstream call, no event row
    assert e.headers["X-Toup-Reason"] == REASON
    assert e.headers["x-should-retry"] == "false"
    assert e.headers.get("Retry-After") == retry_after
    d = e.detail
    # Spend and budget are Toup's provider cost: never on the wire (str(exc)
    # reaches job rows and model context), only in the log line below.
    assert set(d) == {"error", "message", "provider", "period_start", "period_end"}
    assert d["error"] == REASON
    assert d["message"] == "Monthly openai budget exceeded"      # the legacy sentence, kept
    assert d["provider"] == "openai"
    assert (d["period_start"], d["period_end"]) == period
    assert json.loads(json.dumps(d)) == d                        # JSON-safe: no Decimal, no datetime
    assert "1000" not in json.dumps(d) and "cents" not in json.dumps(d)
    assert budget_refusal.is_budget_refusal(d)
    lines = _refused_log_lines(caplog)
    assert len(lines) == 1 and f"user={uid[:8]}" in lines[0] and uid not in lines[0]
    assert "provider=openai" in lines[0] and "spent=1000.50" in lines[0] and "budget=1000" in lines[0]
    spent, budget = _REFUSED_LINE.search(lines[0]).groups()
    assert float(spent) >= int(budget)                            # the gate's own comparison, logged


async def test_an_anthropic_monthly_refusal_uses_the_anthropic_budget(monkeypatch, caplog):
    from app.db import async_session_maker

    uid, _ = await _seed(events=[(datetime(2026, 9, 28), 3000, "anthropic")])   # 3000c: the month's cap
    upstream = []

    async def fake_chat(body, api_key):
        upstream.append(body)

    monkeypatch.setattr(lp, "_route_chat",
                        lambda model, cfg: (types.SimpleNamespace(name="anthropic", chat=fake_chat), "k"))
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        assert await lp._check_budget(cfg, "openai", db) is None          # the other provider is untouched

        async def fake_auth(request, db_):
            return cfg

        monkeypatch.setattr(lp, "_auth_agent", fake_auth)
        with pytest.raises(HTTPException) as exc:
            await lp.proxy_chat(_FakeRequest({"model": "claude-sonnet-4-6", "messages": []}), db=db)
    d = exc.value.detail
    assert exc.value.status_code == 429 and upstream == []
    assert d["provider"] == "anthropic" and "budget_cents" not in d and "spent_cents" not in d
    assert d["message"] == "Monthly anthropic budget exceeded"
    assert (d["period_start"], d["period_end"]) == WINDOW_ISO
    lines = _refused_log_lines(caplog)
    assert len(lines) == 1 and "provider=anthropic" in lines[0]
    assert _REFUSED_LINE.search(lines[0]).groups() == ("3000.00", "3000")   # the anthropic budget


async def test_refusal_log_is_one_line_per_tenant_window(caplog):
    from app.db import async_session_maker

    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    uid, _ = await _seed(events=[(datetime(2026, 9, 28), 1200)])
    other, _ = await _seed(events=[(datetime(2026, 9, 28), 1200)])
    async with async_session_maker() as db:
        for tenant in (uid, uid, uid, other):
            cfg = await _config(db, tenant)
            with pytest.raises(HTTPException):
                await lp._raise_budget_exceeded(cfg, "openai", db)
    lines = _refused_log_lines(caplog)
    assert len(lines) == 2
    assert f"user={uid[:8]}" in lines[0] and f"user={other[:8]}" in lines[1]


async def test_the_refusal_rechecks_and_admits_after_a_window_roll():
    from app.db import async_session_maker

    _Clock.value = datetime(2026, 9, 24, 19, 9, 59)
    uid, _ = await _seed(events=[(datetime(2026, 9, 1), 1000)])
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        assert await lp._check_budget(cfg, "openai", db) == "monthly_exceeded"
        with pytest.raises(HTTPException) as exc:                # one verdict, one now
            await lp._raise_budget_exceeded(cfg, "openai", db)
        assert exc.value.detail["period_end"] == "2026-09-24T19:10:00+00:00"
        assert exc.value.headers["Retry-After"] == "1"
        # The anniversary passes between the check and the raise: admit rather
        # than refuse with next month's window (a reset a month out).
        _Clock.value = datetime(2026, 9, 24, 19, 10, 1)
        assert await lp._raise_budget_exceeded(cfg, "openai", db) is None


async def test_a_past_window_end_is_never_emitted(client, monkeypatch):
    # budget_period_bounds never returns an end that has passed today; this
    # pins the refusal's and /usage's own guard against a future change that
    # would (a reset in the past is a false promise and a Retry-After of 1 s).
    past_end = NOW - timedelta(minutes=5)
    monkeypatch.setattr(lp, "budget_period_bounds",
                        lambda config, now=None: (datetime(2026, 9, 1), past_end))
    from app.db import async_session_maker

    uid, token = await _seed(events=[(datetime(2026, 9, 28), 1000)])
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        assert await lp._check_budget(cfg, "openai", db) == "monthly_exceeded"
        with pytest.raises(HTTPException) as exc:
            await lp._raise_budget_exceeded(cfg, "openai", db)
    assert exc.value.detail["period_start"] == "2026-09-01T00:00:00+00:00"
    assert exc.value.detail["period_end"] is None and "Retry-After" not in exc.value.headers
    usage = (await client.get("/api/llm/usage", headers={"Authorization": f"Bearer {token}"})).json()
    assert usage["period_end"] is None and usage["openai_blocked"] is True


# ── Whole-second bounds ──────────────────────────────────────────────
#
# Production bundle_started_at values carry microseconds and iso_utc drops
# them. With the anchor unaligned, a refusal in the first .9 s after the
# emitted second was still enforced under the OLD window, so it named an end
# that had already passed (Retry-After 1). Aligned, the enforced boundary is
# the emitted second.

ANCHOR_US = datetime(2026, 3, 15, 6, 45, 12, 900000)       # synthetic, with microseconds
ANNIVERSARY = datetime(2026, 10, 15, 6, 45, 12)             # ANCHOR_US + 7 months, whole second
NEXT_ANNIVERSARY = datetime(2026, 11, 15, 6, 45, 12)
_AROUND_THE_ANNIVERSARY = [ANNIVERSARY + timedelta(microseconds=us) for us in (
    -999999, -500000, -1, 0, 1, 500000, 899999, 900000, 999999)]


@pytest.mark.parametrize("now", _AROUND_THE_ANNIVERSARY)
def test_a_microsecond_anchor_gives_whole_second_bounds(now):
    start, end = lp.budget_period_bounds(_cfg(bundle_started_at=ANCHOR_US), now=now)
    assert start.microsecond == 0 and end.microsecond == 0
    assert start <= now < end
    assert end == (ANNIVERSARY if now < ANNIVERSARY else NEXT_ANNIVERSARY)
    assert budget_refusal.to_naive_utc(budget_refusal.iso_utc(end)) == end   # the wire second IS the bound


def test_a_live_stripe_period_and_the_kill_switch_end_are_whole_seconds(monkeypatch):
    cfg = _cfg(bundle_period_start=datetime(2026, 9, 10, 8, 0, 0, 700000),
               bundle_period_end=datetime(2026, 10, 10, 8, 0, 0, 900000))
    live = (datetime(2026, 9, 10, 8), datetime(2026, 10, 10, 8))
    assert lp.budget_period_bounds(cfg, now=datetime(2026, 10, 10, 7, 59, 59, 500000)) == live
    # half a second past the emitted end, the period is over — not live until .9
    assert lp.budget_period_bounds(cfg, now=datetime(2026, 10, 10, 8, 0, 0, 500000)) == (
        datetime(2026, 10, 10, 8), datetime(2026, 11, 10, 8))
    monkeypatch.setattr(settings, "bundle_budget_rolling_month", False)
    legacy = _cfg(bundle_started_at=ANCHOR_US, bundle_period_end=datetime(2026, 10, 1, 0, 0, 0, 900000))
    assert lp.budget_period_bounds(legacy, now=datetime(2026, 9, 30, 23, 59, 59, 500000)) == (
        datetime(2026, 3, 15, 6, 45, 12), datetime(2026, 10, 1))
    assert lp.budget_period_bounds(legacy, now=datetime(2026, 10, 1, 0, 0, 0, 500000)) == (
        datetime(2026, 3, 15, 6, 45, 12), None)


@pytest.mark.parametrize("now", [ANNIVERSARY - timedelta(microseconds=500000),
                                 ANNIVERSARY - timedelta(microseconds=1),
                                 ANNIVERSARY + timedelta(microseconds=500000),
                                 ANNIVERSARY + timedelta(microseconds=899999)],
                         ids=["last-second-.5", "last-microsecond", "next-second-.5", "next-second-.899999"])
async def test_a_refusal_around_the_anniversary_never_names_a_past_reset(client, now):
    from app.db import async_session_maker

    _Clock.value = now
    # over budget in the window that is ending and in the one that begins
    uid, token = await _seed(started_at=ANCHOR_US,
                             events=[(datetime(2026, 10, 10), 1000), (ANNIVERSARY, 1000)])
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        enforced = lp.budget_period_bounds(cfg)
        assert await lp._check_budget(cfg, "openai", db) == "monthly_exceeded"
        with pytest.raises(HTTPException) as exc:
            await lp._raise_budget_exceeded(cfg, "openai", db)
    d, headers = exc.value.detail, exc.value.headers
    assert budget_refusal.parse_utc(d["period_end"]) > now.replace(tzinfo=timezone.utc)
    assert (d["period_start"], d["period_end"]) == tuple(budget_refusal.iso_utc(b) for b in enforced)
    assert budget_refusal.to_naive_utc(d["period_end"]) == enforced[1]   # the very second enforced
    assert 1 <= int(headers["Retry-After"]) <= 604800
    usage = (await client.get("/api/llm/usage", headers={"Authorization": f"Bearer {token}"})).json()
    assert usage["openai_blocked"] is True
    assert (usage["period_start"], usage["period_end"]) == (d["period_start"], d["period_end"])


async def test_proxy_chat_proceeds_when_the_recheck_admits(monkeypatch):
    from app.db import async_session_maker

    uid, _ = await _seed(events=SPLIT_EVENTS)                    # the fresh verdict admits
    calls = []

    async def fake_chat(body, api_key):
        calls.append(body)
        return types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    async def stale_check(cfg, provider, db):                    # decided just before a roll
        return "monthly_exceeded"

    async def no_log(*a, **kw):
        return None

    monkeypatch.setattr(lp, "_route_chat",
                        lambda model, cfg: (types.SimpleNamespace(name="openai", chat=fake_chat), "k"))
    monkeypatch.setattr(lp, "_check_budget", stale_check)
    monkeypatch.setattr(lp, "_log_event", no_log)
    async with async_session_maker() as db:
        cfg = await _config(db, uid)

        async def fake_auth(request, db_):
            return cfg

        monkeypatch.setattr(lp, "_auth_agent", fake_auth)
        resp = await lp.proxy_chat(_FakeRequest({"model": "gpt-5.5", "messages": []}), db=db)
    assert resp.status_code == 200 and len(calls) == 1


async def _anthropic_daily_cap_tenant(openai_cents):
    """Anthropic: 150c today (over the 100c daily cap, under the 3000c month)."""
    return await _seed(events=[(datetime(2026, 9, 28), openai_cents),
                               (datetime(2026, 9, 29, 0, 45), 150, "anthropic")])


def _anthropic_then_fake_openai(monkeypatch):
    calls = {"anthropic": [], "openai": []}

    async def anthropic_chat(body, api_key):
        calls["anthropic"].append(body)

    async def openai_chat(body, api_key):
        calls["openai"].append(body)
        return types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    async def no_log(*a, **kw):
        return None

    monkeypatch.setattr(lp, "_route_chat", lambda model, cfg: (
        types.SimpleNamespace(name="anthropic", chat=anthropic_chat), "k-anthropic"))
    monkeypatch.setattr(lp, "_openai", types.SimpleNamespace(name="openai", chat=openai_chat))
    monkeypatch.setattr(lp, "_log_event", no_log)
    return calls


async def test_anthropic_daily_fallback_is_refused_by_an_exhausted_openai_budget(monkeypatch):
    from app.db import async_session_maker

    uid, _ = await _anthropic_daily_cap_tenant(openai_cents=1000)
    calls = _anthropic_then_fake_openai(monkeypatch)
    async with async_session_maker() as db:
        cfg = await _config(db, uid)
        assert await lp._check_budget(cfg, "anthropic", db) == "daily_exceeded"

        async def fake_auth(request, db_):
            return cfg

        monkeypatch.setattr(lp, "_auth_agent", fake_auth)
        with pytest.raises(HTTPException) as exc:
            await lp.proxy_chat(_FakeRequest({"model": "claude-sonnet-4-6", "messages": []}), db=db)
    assert exc.value.status_code == 429
    assert exc.value.headers["X-Toup-Reason"] == REASON
    assert exc.value.detail["provider"] == "openai"
    assert calls == {"anthropic": [], "openai": []}              # no silent switch


async def test_anthropic_daily_fallback_still_runs_under_the_openai_budget(monkeypatch):
    from app.db import async_session_maker

    uid, _ = await _anthropic_daily_cap_tenant(openai_cents=10)
    calls = _anthropic_then_fake_openai(monkeypatch)
    async with async_session_maker() as db:
        cfg = await _config(db, uid)

        async def fake_auth(request, db_):
            return cfg

        monkeypatch.setattr(lp, "_auth_agent", fake_auth)
        resp = await lp.proxy_chat(_FakeRequest({
            "model": "claude-sonnet-4-6", "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]}), db=db)
    assert resp.status_code == 200
    assert len(calls["openai"]) == 1 and calls["anthropic"] == []


# ── End to end over HTTP ─────────────────────────────────────────────


async def test_ordinary_account_end_to_end(client, monkeypatch):
    async def fake_chat(self, body, api_key):
        return types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(lp.OpenAIBackend, "chat", fake_chat)
    url = "/api/llm/openai/v1/chat/completions"
    body = {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]}

    _, token = await _seed(events=SPLIT_EVENTS)
    res = await client.post(url, json=body, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200, res.text                      # 31c this window, >1000c lifetime

    _, over = await _seed(events=[(datetime(2026, 9, 28), 1000)])
    res = await client.post(url, json=body, headers={"Authorization": f"Bearer {over}"})
    assert res.status_code == 429
    assert res.headers["x-toup-reason"] == REASON
    assert res.headers["x-should-retry"] == "false"
    assert res.headers["retry-after"] == "604800"
    detail = res.json()["detail"]
    assert detail["error"] == REASON
    assert (detail["period_start"], detail["period_end"]) == WINDOW_ISO

    usage = (await client.get("/api/llm/usage", headers={"Authorization": f"Bearer {token}"})).json()
    assert (usage["period_start"], usage["period_end"]) == WINDOW_ISO
    assert usage["openai_monthly_cents"] == pytest.approx(31.2575, abs=0.1)
    assert usage["openai_remaining_cents"] == pytest.approx(1000 - 31.2575, abs=0.1)
    assert usage["openai_blocked"] is False and usage["budget_exempt"] is False

    blocked = (await client.get("/api/llm/usage", headers={"Authorization": f"Bearer {over}"})).json()
    assert blocked["openai_blocked"] is True
    assert blocked["openai_remaining_cents"] == 0
    assert (blocked["period_start"], blocked["period_end"]) == WINDOW_ISO


@pytest.mark.parametrize("path, body, message", [
    ("/api/llm/openai/v1/responses", {"model": "gpt-5.5", "input": "hi"},
     "Monthly openai budget exceeded"),
    ("/api/llm/openai/v1/embeddings", {"model": "text-embedding-3-small", "input": "hi"},
     "Monthly OpenAI budget exceeded"),
], ids=["responses", "embeddings"])
async def test_the_other_openai_routes_refuse_with_the_typed_429(client, monkeypatch, path, body, message):
    async def never(*a, **kw):
        raise AssertionError("a refused call reached the provider")

    for name in ("chat", "chat_stream", "responses", "responses_stream", "embeddings"):
        monkeypatch.setattr(lp.OpenAIBackend, name, never)
    _, token = await _seed(events=[(datetime(2026, 9, 28), 1000)])
    res = await client.post(path, json=body, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 429, res.text
    assert res.headers["x-toup-reason"] == REASON and res.headers["x-should-retry"] == "false"
    assert res.headers["retry-after"] == "604800"
    detail = res.json()["detail"]
    assert detail["error"] == REASON and detail["message"] == message
    assert (detail["period_start"], detail["period_end"]) == WINDOW_ISO


async def test_usage_for_exempt_and_unanchored_tenants(client):
    _, admin = await _seed(role="admin", events=[(datetime(2026, 9, 28), 5000)])
    usage = (await client.get("/api/llm/usage", headers={"Authorization": f"Bearer {admin}"})).json()
    assert usage["budget_exempt"] is True and usage["openai_blocked"] is False
    assert usage["openai_remaining_cents"] == 1e12 and usage["anthropic_remaining_cents"] == 1e12
    assert usage["openai_monthly_cents"] == pytest.approx(5000)   # spend is still reported
    assert (usage["period_start"], usage["period_end"]) == WINDOW_ISO

    _, fresh = await _seed(started_at=None)
    usage = (await client.get("/api/llm/usage", headers={"Authorization": f"Bearer {fresh}"})).json()
    assert usage["period_start"] is None and usage["period_end"] is None
    assert usage["openai_remaining_cents"] == 1000 and usage["anthropic_remaining_cents"] == 3000
    assert usage["openai_blocked"] is False and usage["budget_exempt"] is False


class _CountingTransport(httpx.ASGITransport):
    def __init__(self, app):
        super().__init__(app=app)
        self.paths: list[str] = []

    async def handle_async_request(self, request):
        self.paths.append(request.url.path)
        return await super().handle_async_request(request)


@pytest.mark.parametrize("seed, retry_after", [
    # 30 s ahead: without x-should-retry the SDK would sleep 30 s and retry
    (dict(period_start=_STRIPE_START, period_end=NOW + timedelta(seconds=30)), "30"),
    (dict(), "604800"),                                          # 25 days ahead
    (dict(rolling=False), None),                                 # no end at all
], ids=["30s", "25d", "none"])
async def test_openai_sdk_with_default_retries_makes_exactly_one_request(monkeypatch, caplog, seed, retry_after):
    seed = dict(seed)
    monkeypatch.setattr(settings, "bundle_budget_rolling_month", seed.pop("rolling", True))
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    uid, token = await _seed(events=[(datetime(2026, 9, 28), 1000.25)], **seed)

    async def never(self, body, api_key):
        raise AssertionError("a refused call reached the provider")

    monkeypatch.setattr(lp.OpenAIBackend, "chat", never)
    app = FastAPI()
    app.include_router(lp.router, prefix=settings.api_prefix)
    transport = _CountingTransport(app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        sdk = openai.AsyncOpenAI(api_key=token, http_client=http,
                                 base_url=f"http://test{settings.api_prefix}/llm/openai/v1")
        assert sdk.max_retries == 2                              # the SDK default the chat client keeps
        with pytest.raises(openai.RateLimitError) as exc:
            await asyncio.wait_for(sdk.chat.completions.create(
                model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]), timeout=20)
    assert transport.paths == [f"{settings.api_prefix}/llm/openai/v1/chat/completions"]
    err = exc.value
    assert err.response.headers.get("retry-after") == retry_after
    assert budget_refusal.is_budget_refusal(err)
    detail = budget_refusal.budget_refusal_detail(err)
    assert detail is not None and detail["error"] == REASON
    assert set(detail) == {"error", "message", "provider", "period_start", "period_end"}
    # What job rows, trigger/routine surfaces and model context get from
    # str()/repr() of the SDK error: no spend, no budget.
    for text in (str(err), repr(err)):
        assert "cents" not in text and "1000" not in text, text
        assert budget_refusal.is_budget_refusal(text)
        assert budget_refusal.budget_refusal_detail(text) == detail
    lines = _refused_log_lines(caplog)
    assert len(lines) == 1 and f"user={uid[:8]}" in lines[0]
    assert _REFUSED_LINE.search(lines[0]).groups() == ("1000.25", "1000")


@pytest.mark.parametrize("route", ["generations", "edits"])
async def test_a_budget_refused_image_leaves_the_free_image_count_unchanged(client, monkeypatch, route):
    from app.services import credit_service

    held: dict[str, str] = {}                                    # open free-image reservations

    async def fake_reserve(db, user_id):
        rid = f"slot-{len(held) + 1}"
        held[rid] = user_id
        return (False, len(held), 10, rid)

    async def fake_release(db, reservation_id):
        held.pop(reservation_id, None)

    async def never(*a, **kw):
        raise AssertionError("a refused image reached the provider")

    monkeypatch.setattr(credit_service, "reserve_free_image_slot", fake_reserve)
    monkeypatch.setattr(credit_service, "release_free_image_slot", fake_release)
    monkeypatch.setattr(lp.OpenAIBackend, "images", never)
    monkeypatch.setattr(lp.OpenAIBackend, "images_edit", never)
    _, token = await _seed(events=[(datetime(2026, 9, 28), 1000)])
    auth = {"Authorization": f"Bearer {token}"}
    url = f"/api/llm/openai/v1/images/{route}"
    if route == "edits":
        res = await client.post(url, headers=auth, data={"prompt": "a hat", "model": "gpt-image-1"},
                                files={"image": ("in.png", b"\x89PNG\r\n\x1a\n", "image/png")})
    else:
        res = await client.post(url, headers=auth, json={"prompt": "a cat", "model": "gpt-image-1"})
    assert res.status_code == 429, res.text
    assert res.headers["x-toup-reason"] == REASON
    assert res.json()["detail"]["message"] == "Monthly OpenAI budget exceeded"
    assert held == {}


# ── Structural pins ──────────────────────────────────────────────────


def _handler(name: str) -> str:
    src = inspect.getsource(lp)
    start = src.index(f"async def {name}(")
    nxt = src.find("\nasync def ", start + 1)
    return src[start: nxt if nxt != -1 else len(src)]


def test_every_monthly_refusal_is_the_typed_one():
    for name in ("proxy_chat", "proxy_responses", "proxy_embeddings",
                 "proxy_openai_images", "proxy_openai_image_edits"):
        body = _handler(name)
        assert "_check_budget(" in body and "_raise_budget_exceeded(" in body, name
    src = inspect.getsource(lp)
    assert not re.search(r'HTTPException\(\s*429\s*,\s*f?"Monthly', src)
    assert "_log_event(" not in inspect.getsource(lp._raise_budget_exceeded)


def test_the_anthropic_daily_fallback_rechecks_the_openai_budget():
    body = _handler("proxy_chat")
    daily = body.index('if budget_result == "daily_exceeded":')
    segment = body[daily: body.index("_anthropic_to_openai_request(body)", daily)]
    # R4b (Option A branch): the fallback gate carries the request's kind.
    assert '_check_budget(config, "openai", db,\n' in segment
    assert "**budget_kind_kw)" in segment
    assert "_raise_budget_exceeded(" in segment


def test_image_routes_check_the_budget_before_reserving_a_free_image():
    for name in ("proxy_openai_images", "proxy_openai_image_edits"):
        body = _handler(name)
        assert body.index("_raise_budget_exceeded(") < body.index(
            "reserve_free_image_slot(db, config.user_id)"), name


def test_the_gate_and_usage_share_one_exemption():
    # One live standing lookup (admin exemption + Unlimited plan, one query)
    # and one refusal rule, read by both the gate and /usage.
    gate = inspect.getsource(lp._check_budget)
    usage = _handler("get_proxy_usage")
    for body in (gate, usage):
        assert "_budget_standing(" in body and "_monthly_allocation_refuses(" in body
    assert "_enforce_rate_limit(" not in usage
    assert "_budget_standing(" in inspect.getsource(lp._is_budget_exempt)
