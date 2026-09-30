"""Option A (prepared for a PENDING owner decision): the proxy's monthly
model allocation honours the Unlimited plan on TEXT calls.

The Unlimited design of record (docs/billing/UNLIMITED_ENTITLEMENT_DESIGN.md,
2026-09-13) says a human on Unlimited never meets a spend refusal: log ->
alert -> pace, refusal off. The proxy's per-tenant allocation
(`llm_proxy._check_budget`, 2026-04-09) predates it and refused an Unlimited
tenant like any other once its $10 OpenAI allocation was spent. These tests
pin the prepared behaviour:

* an Unlimited tenant (credit_balances.plan_id == 'unlimited') over its
  allocation is admitted on chat/responses/embeddings and reported once per
  1x/2x/5x multiple per window (one `[budget] unlimited over allocation`
  line + one infra alert, pending until delivered);
* R1: the standing is read live, so an upgrade or a downgrade at an
  exhausted cap takes effect on the very next call (no clock jump);
* R2: an undelivered alert (False or an exception) is retried on later gate
  calls with a doubling backoff, and never again once delivered;
* R2b: a multiple crossed while another alert's send is in flight is sent
  when that send completes, with NO further gate call (held on an
  asyncio.Event, no sleeps); after a failure both are delivered once, in
  order, by the (fake, explicitly fired) backoff timer; the timer is bounded;
* R3: the alert uses a nonzero alerting window, so alerting.py's
  per-subject window and per-category subject cap hold (real
  send_infra_alert, fake Telegram);
* R4: the OpenAI image generation and edit routes keep the monthly stop for
  Unlimited (typed 429), and an image request above `_IMAGE_MAX_N` images is
  refused before anything is reserved or spent;
* R4b: a chat/Responses request carrying an image_generation tool (in
  `tools`, forced or allow-listed via `tool_choice`, streamed or not) or any
  other hosted tool gets the typed 429 for an Unlimited tenant over its
  allocation with ZERO upstream calls; the same request under the cap, the
  agent's own function-tool traffic over the cap, other plans and admins
  are unchanged;
* `unlimited_proxy_budget_refusal_enabled` restores the typed 429 for it,
  exactly as for every other tenant;
* every other plan, a failed plan lookup (fail closed) and admins are
  unchanged; the Anthropic daily soft cap still routes to OpenAI;
* `GET /api/llm/usage` reports `unlimited` and does not report an admitted
  Unlimited tenant as blocked (the agent's document preflight reads it),
  while `openai_image_blocked` says the image routes would refuse.

Lane: platform
(llm_proxy_events and credit_balances are PLATFORM tables.)
"""
from __future__ import annotations

import asyncio
from collections import deque
import logging
import types
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import select, update

from app.api import llm_proxy as lp
from app.config import settings
from app.services import alerting

REASON = "monthly_model_budget_exceeded"
ANCHOR = datetime(2026, 5, 24, 19, 10)   # bundle_started_at (naive UTC, like the column)
NOW = datetime(2026, 9, 29, 1, 30)
WINDOW_ISO = ("2026-09-24T19:10:00+00:00", "2026-10-24T19:10:00+00:00")
IN_WINDOW = datetime(2026, 9, 28, 12, 0)
OVER_LINE = "[budget] unlimited over allocation"


class _Clock(datetime):
    value = NOW

    @classmethod
    def utcnow(cls):
        return cls.value

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.value
        return cls.value.replace(tzinfo=timezone.utc).astimezone(tz)


class _Mono:
    """The alert-retry clock (``lp._monotonic``)."""
    value = 1_000_000.0


class _Timer:
    def __init__(self, delay, callback, args):
        self.delay, self.callback, self.args = delay, callback, args
        self.due = _Mono.value + delay
        self.cancelled = self.fired = False

    def cancel(self):
        self.cancelled = True


class _Timers(list):
    """Every alert backoff timer armed through ``lp._call_later`` — never a
    real one: a test fires them explicitly (``fire_due``) after moving the
    retry clock, so nothing sleeps and nothing races."""

    def armed(self):
        return [t for t in self if not (t.cancelled or t.fired)]

    def fire_due(self):
        due = [t for t in self.armed() if t.due <= _Mono.value]
        for t in due:
            t.fired = True
            t.callback(*t.args)
        return len(due)


_TIMERS = _Timers()


@pytest.fixture(autouse=True)
def _proxy_state(monkeypatch):
    def _clear():
        lp._budget_cache.clear()
        lp._rate_windows.clear()
        lp._refusal_logged_at.clear()
        lp._unlimited_alerts.clear()
        lp._unlimited_alert_tasks.clear()   # a failed test's held send stays on its own loop
        alerting.reset_for_tests()
        _TIMERS.clear()

    _clear()
    _Clock.value = NOW
    _Mono.value = 1_000_000.0
    monkeypatch.setattr(lp, "datetime", _Clock)
    monkeypatch.setattr(lp, "_monotonic", lambda: _Mono.value)
    # raising=False: the seam is new in R2b, and the R2b tests must be able
    # to run (and fail on their assertions) against the code before it.
    monkeypatch.setattr(lp, "_call_later",
                        lambda delay, cb, *args: _TIMERS.append(_Timer(delay, cb, args)) or _TIMERS[-1],
                        raising=False)
    monkeypatch.setattr(settings, "bundle_budget_rolling_month", True)
    monkeypatch.setattr(settings, "credit_enforcement_enabled", False)
    monkeypatch.setattr(settings, "unlimited_proxy_budget_refusal_enabled", False)
    yield
    _clear()


@pytest.fixture
def alerts(monkeypatch):
    """Every send_infra_alert the proxy makes, as (category, level, message,
    kw). Delivered (True) unless a test queues other outcomes in
    ``alerts.outcomes`` (True, False or an exception instance)."""

    class _Sent(list):
        outcomes: list = []

    sent = _Sent()
    sent.outcomes = []

    async def record(category, level, message, **kw):
        sent.append((category, level, message, kw))
        outcome = sent.outcomes.pop(0) if sent.outcomes else True
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(alerting, "send_infra_alert", record)
    monkeypatch.setattr(alerting, "infra_alerts_configured", lambda: True)
    return sent


@pytest.fixture
def timers():
    return _TIMERS


async def _drain_alerts():
    """Until no alert send is running — a finished send may start the next
    owed one (R2b), so one gather is not enough."""
    while lp._unlimited_alert_tasks:
        await asyncio.gather(*list(lp._unlimited_alert_tasks))


def _over_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "app.api.llm_proxy" and r.getMessage().startswith(OVER_LINE)]


async def _seed(*, plan="unlimited", role="beta_user", events=(), budget=1000,
                anthropic_budget=3000):
    """A tenant with a proxy token and (unless plan is None) a credit balance
    row on `plan`. `events` are (at, cents) for openai or (at, cents, provider)."""
    from app.db import AgentConfig, CreditBalance, LLMProxyEvent, User, async_session_maker

    uid, token = str(uuid.uuid4()), f"tok-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"u-{uid[:8]}@example.test", hashed_password="x", role=role))
        await db.flush()
        db.add(AgentConfig(
            user_id=uid, llm_token_hash=lp._hash_token(token), bundle_status="active",
            llm_mode="bundle", bundle_started_at=ANCHOR, bundle_openai_budget_cents=budget,
            bundle_anthropic_budget_cents=anthropic_budget,
            bundle_openai_api_key="sk-test-outbound",
        ))
        if plan is not None:
            db.add(CreditBalance(
                user_id=uid, plan_id=plan, day_anchor_local_date="2026-09-29",
                period_start=datetime(2026, 9, 24), period_end=datetime(2026, 10, 24),
            ))
        await db.commit()
    await _add_events(uid, events)
    return uid, token


async def _add_events(uid, events):
    from app.db import LLMProxyEvent, async_session_maker

    async with async_session_maker() as db:
        for at, cents, *provider in events:
            db.add(LLMProxyEvent(user_id=uid, provider=(provider or ["openai"])[0],
                                 model="gpt-5.5", endpoint="chat",
                                 cost_cents=Decimal(str(cents)), created_at=at))
        await db.commit()
    lp._invalidate_cache(uid)   # what _log_event does after a user-attributable event


async def _config(db, uid):
    from app.db import AgentConfig
    return (await db.execute(select(AgentConfig).where(AgentConfig.user_id == uid))).scalar_one()


async def _gate(uid, provider="openai"):
    from app.db import async_session_maker

    async with async_session_maker() as db:
        return await lp._check_budget(await _config(db, uid), provider, db)


# ── The gate ─────────────────────────────────────────────────────────


async def test_unlimited_over_allocation_is_admitted_and_reported_once_per_multiple(caplog, alerts):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    uid, _ = await _seed(events=[(IN_WINDOW, 999.99)])
    assert await _gate(uid) is None
    await _drain_alerts()
    assert _over_lines(caplog) == [] and alerts == []            # under the allocation: silent

    await _add_events(uid, [(IN_WINDOW, 0.01)])                   # exactly 1000c: the gate's >=
    for _ in range(3):
        assert await _gate(uid) is None
    await _drain_alerts()
    assert _over_lines(caplog) == [
        f"{OVER_LINE} user={uid[:8]} provider=openai spent=1000.00 budget=1000 multiple=1"]
    assert len(alerts) == 1
    category, level, message, kw = alerts[0]
    assert category == "unlimited_over_allocation" and level == "warning"
    assert kw["subject"] == uid[:8]
    assert kw["min_interval_s"] == lp._UNLIMITED_ALERT_INTERVAL_S > 0   # R3: never 0
    assert uid not in message and uid[:8] in message and "NOT refused" in message
    assert all(uid not in line for line in _over_lines(caplog))   # the 8-char prefix only

    await _add_events(uid, [(IN_WINDOW, 1000)])                   # 2x
    assert await _gate(uid) is None and await _gate(uid) is None
    await _add_events(uid, [(IN_WINDOW, 2500)])                   # 4.5x: no new multiple
    assert await _gate(uid) is None
    await _add_events(uid, [(IN_WINDOW, 500)])                    # 5x
    assert await _gate(uid) is None and await _gate(uid) is None
    await _drain_alerts()
    assert [line.rsplit("multiple=", 1)[1] for line in _over_lines(caplog)] == ["1", "2", "5"]
    assert "spent=5000.00 budget=1000 multiple=5" in _over_lines(caplog)[-1]
    assert len(alerts) == 3
    assert not [r for r in caplog.records if r.getMessage().startswith("[budget] refused")]


async def test_a_tenant_first_seen_past_several_multiples_is_reported_once_at_the_highest(caplog, alerts):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    uid, _ = await _seed(events=[(IN_WINDOW, 6000)])
    other, _ = await _seed(events=[(IN_WINDOW, 1500)])
    for _ in range(2):
        assert await _gate(uid) is None and await _gate(other) is None
    await _drain_alerts()
    lines = _over_lines(caplog)
    assert len(lines) == 2 and len(alerts) == 2
    assert f"user={uid[:8]} provider=openai spent=6000.00 budget=1000 multiple=5" in lines[0]
    assert f"user={other[:8]} provider=openai spent=1500.00 budget=1000 multiple=1" in lines[1]
    assert [a[3]["subject"] for a in alerts] == [uid[:8], other[:8]]   # per-tenant dedupe


async def test_a_new_window_reports_again(caplog, alerts):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    _Clock.value = datetime(2026, 9, 24, 19, 9, 50)               # the old window's last seconds
    uid, _ = await _seed(events=[(datetime(2026, 9, 20), 1000)])
    assert await _gate(uid) is None
    _Clock.value = datetime(2026, 9, 29)                          # next window
    await _add_events(uid, [(IN_WINDOW, 1000)])
    assert await _gate(uid) is None and await _gate(uid) is None
    await _drain_alerts()
    assert len(_over_lines(caplog)) == 2 and len(alerts) == 2
    assert "2026-08-24T19:10:00+00:00" in alerts[0][2] and WINDOW_ISO[0] in alerts[1][2]


async def test_the_refusal_flag_refuses_unlimited_exactly_like_other_tenants(client, monkeypatch, caplog, alerts):
    async def fake_chat(self, body, api_key):
        return types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(lp.OpenAIBackend, "chat", fake_chat)
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    url = "/api/llm/openai/v1/chat/completions"
    body = {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]}
    _, unlimited = await _seed(events=[(IN_WINDOW, 1000)])
    _, builder = await _seed(plan="builder", events=[(IN_WINDOW, 1000)])

    def auth(token):
        return {"Authorization": f"Bearer {token}"}

    monkeypatch.setattr(settings, "unlimited_proxy_budget_refusal_enabled", True)
    refusals = [await client.post(url, json=body, headers=auth(t)) for t in (unlimited, builder)]
    for res in refusals:
        assert res.status_code == 429, res.text
        assert res.headers["x-toup-reason"] == REASON and res.headers["x-should-retry"] == "false"
        assert res.headers["retry-after"] == "604800"
        detail = res.json()["detail"]
        assert set(detail) == {"error", "message", "provider", "period_start", "period_end"}
        assert detail["error"] == REASON and detail["message"] == "Monthly openai budget exceeded"
        assert (detail["period_start"], detail["period_end"]) == WINDOW_ISO
    assert refusals[0].json() == refusals[1].json()               # indistinguishable on the wire
    usage = (await client.get("/api/llm/usage", headers=auth(unlimited))).json()
    assert usage["openai_blocked"] is True and usage["unlimited"] is True
    await _drain_alerts()
    assert _over_lines(caplog) == [] and alerts == []             # refused, not "admitted over"

    # Off (the default): the same Unlimited tenant is admitted; builder is not.
    monkeypatch.setattr(settings, "unlimited_proxy_budget_refusal_enabled", False)
    res = await client.post(url, json=body, headers=auth(unlimited))
    assert res.status_code == 200, res.text
    assert (await client.post(url, json=body, headers=auth(builder))).status_code == 429


@pytest.mark.parametrize("plan", ["free", "starter", "builder", "pro", "elite", None],
                         ids=["free", "starter", "builder", "pro", "elite", "no-balance-row"])
async def test_only_the_unlimited_plan_is_honoured(plan, caplog, alerts):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    other, _ = await _seed(plan=plan, events=[(IN_WINDOW, 1000)])
    unlimited, _ = await _seed(events=[(IN_WINDOW, 1000)])
    assert await _gate(other) == "monthly_exceeded"
    assert await _gate(unlimited) is None
    assert await _gate(other) == "monthly_exceeded"               # warm caches, still per tenant
    await _drain_alerts()
    lines = _over_lines(caplog)
    assert len(lines) == 1 and f"user={unlimited[:8]}" in lines[0]
    assert [a[3]["subject"] for a in alerts] == [unlimited[:8]]


async def test_a_failed_plan_lookup_fails_closed_and_is_not_cached(monkeypatch, caplog, alerts):
    from app.db import async_session_maker

    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    unlimited, _ = await _seed(events=[(IN_WINDOW, 1000)])
    free, _ = await _seed(plan="free", events=[(IN_WINDOW, 10)])
    admin, _ = await _seed(role="admin", events=[(IN_WINDOW, 5000)])
    real = lp._standing_statement

    def broken(user_id):
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(lp, "_standing_statement", broken)
    assert await _gate(unlimited) == "monthly_exceeded"           # not unlimited: refused when over
    assert await _gate(free) is None                              # under: still admitted, never a 500
    assert await _gate(admin) == "monthly_exceeded"               # same as the old role lookup
    async with async_session_maker() as db:
        cfg = await _config(db, unlimited)
        with pytest.raises(HTTPException) as exc:
            await lp._raise_budget_exceeded(cfg, "openai", db)
    assert exc.value.status_code == 429 and exc.value.detail["error"] == REASON
    errors = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("[budget] standing_lookup_error")]
    assert errors and f"user={unlimited[:8]}" in errors[0] and unlimited not in errors[0]

    monkeypatch.setattr(lp, "_standing_statement", real)          # the lookup recovers
    assert await _gate(unlimited) is None
    assert await _gate(admin) is None


async def test_the_alert_path_never_raises(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")

    async def exploding(*a, **kw):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(alerting, "send_infra_alert", exploding)
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    assert await _gate(uid) is None
    await _drain_alerts()
    assert any(r.getMessage().startswith(f"[budget] unlimited alert failed user={uid[:8]}")
               for r in caplog.records)

    def broken_multiple(spent, budget):
        raise ArithmeticError("boom")

    monkeypatch.setattr(lp, "_unlimited_multiple", broken_multiple)
    other, _ = await _seed(events=[(IN_WINDOW, 2000)])
    assert await _gate(other) is None                             # the report failed, the call did not
    assert any(r.getMessage().startswith(f"[budget] unlimited report failed user={other[:8]}")
               for r in caplog.records)


async def test_admins_are_unchanged(client, caplog, alerts):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    plain, plain_tok = await _seed(role="admin", plan="free", events=[(IN_WINDOW, 5000)])
    both, both_tok = await _seed(role="admin", plan="unlimited", events=[(IN_WINDOW, 5000)])
    for uid in (plain, both):
        assert await _gate(uid) is None
    await _drain_alerts()
    assert _over_lines(caplog) == [] and alerts == []             # exempt: never "over allocation"
    for token, unlimited in ((plain_tok, False), (both_tok, True)):
        usage = (await client.get("/api/llm/usage",
                                  headers={"Authorization": f"Bearer {token}"})).json()
        assert usage["budget_exempt"] is True and usage["openai_blocked"] is False
        assert usage["openai_remaining_cents"] == 1e12
        assert usage["unlimited"] is unlimited                    # the plan, reported as is


# ── The Anthropic daily soft cap still routes ────────────────────────


class _FakeRequest:
    def __init__(self, body, headers=None):
        self._body, self.headers = body, headers or {}

    async def json(self):
        return self._body


async def test_the_anthropic_daily_cap_still_falls_back_to_openai_for_unlimited(monkeypatch, caplog, alerts):
    from app.db import async_session_maker

    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    # Over the Anthropic month (3000c), over its daily cap (100c today), and
    # over the OpenAI month (1000c): an Unlimited tenant is still served, by
    # the OpenAI fallback, not refused.
    uid, _ = await _seed(events=[(IN_WINDOW, 1000),
                                 (IN_WINDOW, 2900, "anthropic"),
                                 (datetime(2026, 9, 29, 0, 45), 150, "anthropic")])
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
    assert await _gate(uid, "anthropic") == "daily_exceeded"      # routing cap, not a refusal
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
    await _drain_alerts()
    assert sorted(line.split(" provider=")[1].split(" ")[0] for line in _over_lines(caplog)) == [
        "anthropic", "openai"]


# ── /api/llm/usage and the agent's document preflight ────────────────


async def test_usage_does_not_block_an_admitted_unlimited_tenant(client, monkeypatch, alerts):
    from app.agent import attachment_analysis as AA
    from app.services import bundle_client

    _, unlimited = await _seed(events=[(IN_WINDOW, 1200)])
    _, free = await _seed(plan="free", events=[(IN_WINDOW, 1200)])

    async def usage_of(token):
        return (await client.get("/api/llm/usage",
                                 headers={"Authorization": f"Bearer {token}"})).json()

    u = await usage_of(unlimited)
    assert u["unlimited"] is True and u["budget_exempt"] is False
    assert u["openai_blocked"] is False                           # admitted, so not blocked
    assert u["openai_remaining_cents"] == 0                       # spend accounting only
    assert u["openai_monthly_cents"] == pytest.approx(1200)
    assert (u["period_start"], u["period_end"]) == WINDOW_ISO
    f = await usage_of(free)
    assert f["unlimited"] is False and f["openai_blocked"] is True

    # The agent's document-analysis preflight, against this very endpoint.
    monkeypatch.setattr(settings, "llm_mode", "bundle")
    monkeypatch.setattr(settings, "platform_api_url", "http://platform.test/api")
    transport = client._transport

    def factory(timeout=120.0):
        return httpx.AsyncClient(transport=transport, timeout=timeout)

    monkeypatch.setattr(bundle_client, "_proxy_http_client", factory)
    verdicts = {}
    for name, token in (("unlimited", unlimited), ("free", free)):
        monkeypatch.setattr(settings, "toup_token", token)
        verdicts[name] = await AA._budget_preflight()
    assert verdicts["unlimited"]["blocked"] is False and verdicts["unlimited"]["known"] is True
    assert verdicts["free"]["blocked"] is True and verdicts["free"]["known"] is True


# ── R1: the standing that decides at the cap is never stale ──────────

CHAT_URL = "/api/llm/openai/v1/chat/completions"
CHAT_BODY = {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]}


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _ok_response():
    return types.SimpleNamespace(
        status_code=200, headers={}, content=b"",
        json=lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                "input_tokens": 1, "output_tokens": 1, "total_tokens": 1},
                      "output": [], "data": []})


def _fake_text_provider(monkeypatch):
    """The OpenAI text backends answer 200; returns the routes they served."""
    served: list = []

    def make(name):
        async def fake(self, body, api_key):
            served.append(name)
            return _ok_response()
        return fake

    for name in ("chat", "responses", "embeddings"):
        monkeypatch.setattr(lp.OpenAIBackend, name, make(name))
    return served


async def _set_plan(uid, plan):
    """Another replica / the billing webhook changes the plan (no cache is
    told)."""
    from app.db import CreditBalance, async_session_maker

    async with async_session_maker() as db:
        await db.execute(update(CreditBalance).where(CreditBalance.user_id == uid)
                         .values(plan_id=plan))
        await db.commit()


def _assert_hosted_tool_refusal(res, kind=None):
    """A provider-hosted tool inside chat/Responses/Messages is refused for
    EVERY tenant before the budget gate: the proxy meters only mainline
    tokens, so the per-call / per-image charges of hosted tools would never
    reach the spend the retained ceiling is enforced on."""
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert detail["code"] == "hosted_tool_unsupported"
    assert "images endpoint" in detail["message"]
    if kind is not None:
        assert detail["kind"] == kind


_assert_image_tool_refusal = _assert_hosted_tool_refusal


def _assert_typed_refusal(res):
    assert res.status_code == 429, res.text
    assert res.headers["x-toup-reason"] == REASON and res.headers["x-should-retry"] == "false"
    detail = res.json()["detail"]
    assert set(detail) == {"error", "message", "provider", "period_start", "period_end"}
    assert detail["error"] == REASON
    assert (detail["period_start"], detail["period_end"]) == WINDOW_ISO


async def test_an_upgrade_at_an_exhausted_cap_is_admitted_on_the_very_next_call(client, monkeypatch, alerts):
    served = _fake_text_provider(monkeypatch)
    uid, token = await _seed(plan="free", events=[(IN_WINDOW, 1000)])
    _assert_typed_refusal(await client.post(CHAT_URL, json=CHAT_BODY, headers=_auth(token)))
    usage = (await client.get("/api/llm/usage", headers=_auth(token))).json()
    assert usage["openai_blocked"] is True and usage["unlimited"] is False

    await _set_plan(uid, "unlimited")                              # no clock jump at all
    res = await client.post(CHAT_URL, json=CHAT_BODY, headers=_auth(token))
    assert res.status_code == 200, res.text
    assert served == ["chat"]
    usage = (await client.get("/api/llm/usage", headers=_auth(token))).json()
    assert usage["openai_blocked"] is False and usage["unlimited"] is True


async def test_a_downgrade_over_the_cap_is_refused_on_the_very_next_call(client, monkeypatch, alerts):
    served = _fake_text_provider(monkeypatch)
    uid, token = await _seed(events=[(IN_WINDOW, 1000)])
    assert (await client.post(CHAT_URL, json=CHAT_BODY, headers=_auth(token))).status_code == 200
    usage = (await client.get("/api/llm/usage", headers=_auth(token))).json()
    assert usage["openai_blocked"] is False and usage["unlimited"] is True

    await _set_plan(uid, "free")                                   # revoked, no clock jump
    _assert_typed_refusal(await client.post(CHAT_URL, json=CHAT_BODY, headers=_auth(token)))
    assert served == ["chat"]
    usage = (await client.get("/api/llm/usage", headers=_auth(token))).json()
    assert usage["openai_blocked"] is True and usage["unlimited"] is False


async def test_the_standing_is_one_live_query_per_gate_call(monkeypatch, alerts):
    calls = []
    real = lp._standing_statement

    def counting(user_id):
        calls.append(user_id)
        return real(user_id)

    monkeypatch.setattr(lp, "_standing_statement", counting)
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    for _ in range(3):
        assert await _gate(uid) is None
    assert calls == [uid] * 3                                      # one query each, none cached


# ── R2: an alert stays owed until it is delivered ────────────────────


async def _crossed(uid, times=1):
    for _ in range(times):
        assert await _gate(uid) is None
    await _drain_alerts()


async def test_an_undelivered_alert_is_retried_with_backoff_until_delivered(caplog, alerts):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    alerts.outcomes[:] = [False, False, True]

    await _crossed(uid)
    assert len(alerts) == 1                                        # tried, not delivered
    _Mono.value += 59
    await _crossed(uid, 3)
    assert len(alerts) == 1                                        # no retry before 60 s
    _Mono.value += 1
    await _crossed(uid)
    assert len(alerts) == 2                                        # retry 1: still False
    _Mono.value += 119
    await _crossed(uid)
    assert len(alerts) == 2                                        # backoff doubled to 120 s
    _Mono.value += 1
    await _crossed(uid)
    assert len(alerts) == 3                                        # retry 2: delivered
    _Mono.value += 10_000
    await _crossed(uid, 3)
    assert len(alerts) == 3                                        # never again for 1x
    assert all(a[0] == "unlimited_over_allocation" and "1x its" in a[2] for a in alerts)
    assert len(_over_lines(caplog)) == 1                           # the log line stays one
    not_delivered = [r.getMessage() for r in caplog.records
                     if r.getMessage().startswith("[budget] unlimited alert not delivered")]
    assert [m.split("retry_after_s=")[1] for m in not_delivered] == ["60", "120"]


async def test_a_raising_sender_is_swallowed_and_retried(caplog, alerts):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    alerts.outcomes[:] = [RuntimeError("telegram down"), True]
    await _crossed(uid)                                            # the call is admitted anyway
    assert any(r.getMessage().startswith(f"[budget] unlimited alert failed user={uid[:8]}")
               for r in caplog.records)
    _Mono.value += 60
    await _crossed(uid)
    _Mono.value += 10_000
    await _crossed(uid, 2)
    assert len(alerts) == 2


async def test_the_backoff_is_capped_and_a_new_crossing_sends_the_owed_alerts_at_once_in_order(alerts):
    # R2b changed the second half: a higher crossing no longer supersedes an
    # undelivered lower alert. Each multiple is delivered once, lowest first.
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    alerts.outcomes[:] = [False] * 8
    await _crossed(uid)
    for _ in range(7):
        _Mono.value += lp._UNLIMITED_ALERT_RETRY_MAX_S
        await _crossed(uid)
    assert len(alerts) == 8                                        # delay never exceeds the cap
    assert lp._unlimited_retry_delay(30) == lp._UNLIMITED_ALERT_RETRY_MAX_S
    await _add_events(uid, [(IN_WINDOW, 1000)])                    # 2x while 1x is still owed
    await _crossed(uid)                                            # a new crossing goes now...
    assert len(alerts) == 10
    assert "1x its" in alerts[-2][2] and "2x its" in alerts[-1][2]  # ...owed 1x first, then 2x
    _Mono.value += 10_000
    await _crossed(uid, 2)
    assert len(alerts) == 10                                       # each delivered once


async def test_no_telegram_configured_settles_without_retrying(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    monkeypatch.setattr(settings, "infra_alert_telegram_token", "")
    monkeypatch.setattr(settings, "admin_alert_telegram_token", "")
    calls = []
    real = alerting.send_infra_alert

    async def counting(*a, **kw):
        calls.append(a)
        return await real(*a, **kw)

    monkeypatch.setattr(alerting, "send_infra_alert", counting)
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    await _crossed(uid)
    _Mono.value += 10_000
    await _crossed(uid, 2)
    assert len(calls) == 1 and len(_over_lines(caplog)) == 1


# ── R2b: an alert owed while another is in flight is not lost ────────


def _held_sender(monkeypatch, first_outcome):
    """A send_infra_alert whose FIRST send is held on an asyncio.Event until
    the test releases it, then returns ``first_outcome``; later sends return
    True at once. Also counts the gate's report calls."""
    held = types.SimpleNamespace(sent=[], started=asyncio.Event(),
                                 release=asyncio.Event(), reports=[])

    async def sender(category, level, message, **kw):
        held.sent.append(message)
        if len(held.sent) == 1:
            held.started.set()
            await held.release.wait()
            return first_outcome
        return True

    monkeypatch.setattr(alerting, "send_infra_alert", sender)
    monkeypatch.setattr(alerting, "infra_alerts_configured", lambda: True)
    real_report = lp._report_unlimited_over_allocation

    def counting(*a, **kw):
        held.reports.append(a)
        return real_report(*a, **kw)

    monkeypatch.setattr(lp, "_report_unlimited_over_allocation", counting)
    return held


def _multiples(messages):
    import re
    return [re.search(r"past (\d+x) its", m).group(1) for m in messages]


async def _hold_1x_then_cross_2x(held):
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    assert await _gate(uid) is None                                # 1x: its send starts...
    await asyncio.wait_for(held.started.wait(), 5)                 # ...and is held in flight
    await _add_events(uid, [(IN_WINDOW, 1000)])                    # spend passes 2x
    assert await _gate(uid) is None                                # seen while 1x is in flight
    assert _multiples(held.sent) == ["1x"] and len(held.reports) == 2
    return uid


async def test_a_multiple_crossed_while_a_send_is_in_flight_is_sent_when_it_completes(
        monkeypatch, caplog, timers):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    held = _held_sender(monkeypatch, True)
    await _hold_1x_then_cross_2x(held)

    held.release.set()                                             # the 1x send completes
    await _drain_alerts()
    assert len(held.reports) == 2                                  # NO further gate call...
    assert _multiples(held.sent) == ["1x", "2x"]                   # ...and 2x went, exactly once
    assert timers == []                                            # no retry was needed
    [state] = lp._unlimited_alerts.values()
    assert state.owed == {} and state.delivered == 2 and not state.in_flight
    assert [line.rsplit("multiple=", 1)[1] for line in _over_lines(caplog)] == ["1", "2"]

    _Mono.value += 10_000
    timers.fire_due()
    await _drain_alerts()
    assert _multiples(held.sent) == ["1x", "2x"]                   # and never again


async def test_a_failed_send_with_a_higher_multiple_owed_delivers_both_once_in_order(
        monkeypatch, caplog, timers):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    held = _held_sender(monkeypatch, False)
    await _hold_1x_then_cross_2x(held)

    held.release.set()                                             # the 1x send fails
    await _drain_alerts()
    assert _multiples(held.sent) == ["1x"]

    _Mono.value += 59
    timers.fire_due()
    await _drain_alerts()
    assert _multiples(held.sent) == ["1x"]                         # not before the 60 s backoff
    _Mono.value += 1
    timers.fire_due()                                              # the backoff elapses; no gate call
    await _drain_alerts()
    assert len(held.reports) == 2                                  # still no further gate call
    assert _multiples(held.sent) == ["1x", "1x", "2x"]             # 1x retried, then 2x: in order
    assert [t.delay for t in timers] == [lp._UNLIMITED_ALERT_RETRY_FIRST_S]   # one timer did it
    [state] = lp._unlimited_alerts.values()
    assert state.owed == {} and state.delivered == 2 and timers.armed() == []

    _Mono.value += 10_000
    timers.fire_due()
    await _drain_alerts()
    assert _multiples(held.sent) == ["1x", "1x", "2x"]             # each delivered once


async def test_the_backoff_timer_is_bounded_and_a_gate_call_still_retries(alerts, timers):
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    alerts.outcomes[:] = [False] * (lp._UNLIMITED_ALERT_TIMER_RETRIES + 1)
    await _crossed(uid)                                            # attempt 1 fails, a timer is armed
    for attempt in range(2, lp._UNLIMITED_ALERT_TIMER_RETRIES + 2):
        [timer] = timers.armed()
        assert timer.delay == lp._unlimited_retry_delay(attempt - 1)
        _Mono.value += timer.delay
        assert timers.fire_due() == 1
        await _drain_alerts()
        assert len(alerts) == attempt
    assert timers.armed() == []                                    # bounded: no more timers
    _Mono.value += lp._UNLIMITED_ALERT_RETRY_MAX_S
    await _crossed(uid)                                            # a gate call still retries
    assert len(alerts) == lp._UNLIMITED_ALERT_TIMER_RETRIES + 2
    _Mono.value += 10_000
    await _crossed(uid, 2)
    assert len(alerts) == lp._UNLIMITED_ALERT_TIMER_RETRIES + 2    # delivered: done


async def test_a_gate_call_retry_disarms_the_timer_so_nothing_is_sent_twice(alerts, timers):
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    alerts.outcomes[:] = [False, True]
    await _crossed(uid)
    [timer] = timers.armed()
    _Mono.value += timer.delay
    await _crossed(uid)                                            # the gate call retries first
    assert timer.cancelled and len(alerts) == 2
    assert timers.fire_due() == 0
    await _drain_alerts()
    assert len(alerts) == 2


# ── R3: alerting.py's rate limit and per-category cap stay in force ──


@pytest.fixture
def telegram(monkeypatch):
    """The REAL send_infra_alert, configured, against a fake Telegram.
    ``telegram.statuses`` queues response codes (default 200);
    ``telegram.now`` is alerting.py's clock."""

    class _Telegram:
        posts: list = []
        statuses: list = []
        now = 5_000_000.0

    tg = _Telegram()
    tg.posts, tg.statuses = [], []

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            for _ in range(3):                                     # a real POST yields
                await asyncio.sleep(0)
            status = tg.statuses.pop(0) if tg.statuses else 200
            tg.posts.append((status, json["text"]))
            return types.SimpleNamespace(status_code=status)

    monkeypatch.setattr(settings, "infra_alert_telegram_token", "test-bot-token")
    monkeypatch.setattr(settings, "infra_alert_telegram_chat_id", "test-chat")
    monkeypatch.setattr(settings, "infra_alert_category_subject_cap", 5)
    monkeypatch.setattr(alerting, "httpx", types.SimpleNamespace(AsyncClient=_Client))
    monkeypatch.setattr(alerting, "time", types.SimpleNamespace(time=lambda: tg.now))
    return tg


def _delivered(tg):
    return [text for status, text in tg.posts if status < 300]


async def test_many_tenants_crossing_at_once_page_at_most_the_category_cap(telegram):
    uids = [(await _seed(events=[(IN_WINDOW, 1000)]))[0] for _ in range(12)]
    results = await asyncio.gather(*(_gate(u) for u in uids))     # concurrent crossings
    assert results == [None] * 12
    await _drain_alerts()
    assert len(telegram.posts) == 5                                # the cap, not 12

    _Mono.value += 60                                              # the owed ones retry...
    telegram.now += 60                                             # ...inside the same window
    await asyncio.gather(*(_gate(u) for u in uids))
    await _drain_alerts()
    assert len(telegram.posts) == 5                                # still capped

    _Mono.value += 600
    telegram.now += 600                                            # the category window rolls
    for u in uids:
        assert await _gate(u) is None
    await _drain_alerts()
    delivered = _delivered(telegram)
    assert len(delivered) == 10                                    # at most the cap per window
    assert "per-category cap" in delivered[5]                      # the capped ones are named
    assert len({t.split("Unlimited account ")[1][:8] for t in delivered}) == 10


async def test_the_per_subject_window_holds_and_the_suppressed_alert_is_delivered_later(telegram):
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    await _crossed(uid)
    assert len(telegram.posts) == 1 and "1x its" in telegram.posts[0][1]
    await _add_events(uid, [(IN_WINDOW, 1000)])                    # 2x, 100 s later
    _Mono.value += 100
    telegram.now += 100
    await _crossed(uid)
    _Mono.value += 60
    telegram.now += 60
    await _crossed(uid)
    assert len(telegram.posts) == 1                                # per-subject window: held
    _Mono.value += 600
    telegram.now += 600
    await _crossed(uid)
    assert len(telegram.posts) == 2 and "2x its" in telegram.posts[1][1]
    assert "similar suppressed" in telegram.posts[1][1]
    _Mono.value += 10_000
    telegram.now += 10_000
    await _crossed(uid, 2)
    assert len(telegram.posts) == 2


async def test_a_real_delivery_failure_is_retried(telegram):
    uid, _ = await _seed(events=[(IN_WINDOW, 1000)])
    telegram.statuses[:] = [502]
    await _crossed(uid)
    assert [s for s, _ in telegram.posts] == [502]
    _Mono.value += 30
    await _crossed(uid)
    assert len(telegram.posts) == 1                                # backoff: not yet
    _Mono.value += 30
    await _crossed(uid)
    assert [s for s, _ in telegram.posts] == [502, 200]
    _Mono.value += 10_000
    telegram.now += 10_000
    await _crossed(uid, 2)
    assert len(telegram.posts) == 2


# ── R4: images keep the monthly stop; n is bounded ───────────────────

TEXT_ROUTES = [
    ("/api/llm/openai/v1/chat/completions", CHAT_BODY),
    ("/api/llm/openai/v1/responses", {"model": "gpt-5.5", "input": "hi"}),
    ("/api/llm/openai/v1/embeddings", {"model": "text-embedding-3-small", "input": "hi"}),
]


@pytest.fixture
def free_image_slots(monkeypatch):
    """Open free-image reservations, by id (the real ones need the credit
    system; the budget question is what these tests are about)."""
    from app.services import credit_service

    held: dict[str, str] = {}

    async def fake_reserve(db, user_id):
        rid = f"slot-{len(held) + 1}"
        held[rid] = user_id
        return (False, len(held), 10, rid)

    async def fake_release(db, reservation_id):
        held.pop(reservation_id, None)

    async def fake_settle(db, reservation_id):
        held.pop(reservation_id, None)

    monkeypatch.setattr(credit_service, "reserve_free_image_slot", fake_reserve)
    monkeypatch.setattr(credit_service, "release_free_image_slot", fake_release)
    monkeypatch.setattr(credit_service, "settle_free_image_slot", fake_settle)
    return held


def _image_provider(monkeypatch):
    """The OpenAI image backends; records the n each request carried and
    answers 400 (no image: nothing charged, the slot is released)."""
    seen: list = []

    def rejected():
        return types.SimpleNamespace(status_code=400, headers={}, content=b"{}", text="{}",
                                     json=lambda: {"error": {"message": "test stop"}})

    async def images(self, body, api_key):
        seen.append(("generations", body.get("n")))
        return rejected()

    async def images_edit(self, data, files, api_key):
        seen.append(("edits", data.get("n")))
        return rejected()

    monkeypatch.setattr(lp.OpenAIBackend, "images", images)
    monkeypatch.setattr(lp.OpenAIBackend, "images_edit", images_edit)
    return seen


async def _post_image(client, token, route, n=None):
    url = f"/api/llm/openai/v1/images/{route}"
    if route == "edits":
        data = {"prompt": "a hat", "model": "gpt-image-1"}
        if n is not None:
            data["n"] = str(n)
        return await client.post(url, headers=_auth(token), data=data,
                                 files={"image": ("in.png", b"\x89PNG\r\n\x1a\n", "image/png")})
    body = {"prompt": "a cat", "model": "gpt-image-1"}
    if n is not None:
        body["n"] = n
    return await client.post(url, headers=_auth(token), json=body)


async def test_unlimited_over_allocation_text_is_admitted_and_images_are_refused(
        client, monkeypatch, caplog, alerts, free_image_slots):
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    served = _fake_text_provider(monkeypatch)
    seen = _image_provider(monkeypatch)
    _, token = await _seed(events=[(IN_WINDOW, 1000)])
    for url, body in TEXT_ROUTES:
        res = await client.post(url, json=body, headers=_auth(token))
        assert res.status_code == 200, (url, res.text)
    assert served == ["chat", "responses", "embeddings"]
    for route in ("generations", "edits"):
        res = await _post_image(client, token, route)
        _assert_typed_refusal(res)
        assert res.json()["detail"]["message"] == "Monthly OpenAI budget exceeded"
    assert seen == [] and free_image_slots == {}                   # nothing reached OpenAI or held a slot
    usage = (await client.get("/api/llm/usage", headers=_auth(token))).json()
    assert usage["unlimited"] is True
    assert usage["openai_blocked"] is False and usage["openai_image_blocked"] is True
    await _drain_alerts()
    assert len(_over_lines(caplog)) == 1 and len(alerts) == 1      # text crossing reported once


async def test_unlimited_under_allocation_images_are_admitted(client, monkeypatch, alerts, free_image_slots):
    seen = _image_provider(monkeypatch)
    _, token = await _seed(events=[(IN_WINDOW, 999)])
    for route in ("generations", "edits"):
        res = await _post_image(client, token, route)
        assert res.status_code == 400 and res.json() == {"error": {"message": "test stop"}}
    assert [r for r, _ in seen] == ["generations", "edits"]
    usage = (await client.get("/api/llm/usage", headers=_auth(token))).json()
    assert usage["openai_blocked"] is False and usage["openai_image_blocked"] is False


@pytest.mark.parametrize("plan", ["free", "builder", "elite", None],
                         ids=["free", "builder", "elite", "no-balance-row"])
async def test_non_unlimited_over_allocation_is_refused_on_every_route(
        client, monkeypatch, alerts, free_image_slots, plan):
    served = _fake_text_provider(monkeypatch)
    seen = _image_provider(monkeypatch)
    _, token = await _seed(plan=plan, events=[(IN_WINDOW, 1000)])
    for url, body in TEXT_ROUTES:
        _assert_typed_refusal(await client.post(url, json=body, headers=_auth(token)))
    for route in ("generations", "edits"):
        _assert_typed_refusal(await _post_image(client, token, route))
    assert served == [] and seen == [] and free_image_slots == {}
    usage = (await client.get("/api/llm/usage", headers=_auth(token))).json()
    assert usage["openai_blocked"] is True and usage["openai_image_blocked"] is True
    assert alerts == []


async def test_the_refusal_flag_does_not_change_images_and_admins_stay_exempt(
        client, monkeypatch, alerts, free_image_slots):
    seen = _image_provider(monkeypatch)
    _, unlimited = await _seed(events=[(IN_WINDOW, 1000)])
    _, admin = await _seed(role="admin", plan="unlimited", events=[(IN_WINDOW, 5000)])
    for flag in (True, False):
        monkeypatch.setattr(settings, "unlimited_proxy_budget_refusal_enabled", flag)
        for route in ("generations", "edits"):
            _assert_typed_refusal(await _post_image(client, unlimited, route))
            assert (await _post_image(client, admin, route)).status_code == 400   # reached OpenAI
    assert len(seen) == 4
    usage = (await client.get("/api/llm/usage", headers=_auth(admin))).json()
    assert usage["openai_image_blocked"] is False and usage["budget_exempt"] is True


@pytest.mark.parametrize("route", ["generations", "edits"])
async def test_an_image_request_above_the_n_bound_is_refused_before_anything_is_held(
        client, monkeypatch, free_image_slots, route):
    seen = _image_provider(monkeypatch)
    _, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    res = await _post_image(client, token, route, n=5)
    assert seen == []                                              # never reached OpenAI
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert detail["code"] == "image_count_too_large"
    assert detail["max"] == lp._IMAGE_MAX_N == 4 and detail["requested"] == 5
    assert "at most 4 images per request" in detail["message"]
    assert seen == [] and free_image_slots == {}                   # nothing reserved or sent
    for n in (None, 1, 4):                                         # what clients send: unchanged
        assert (await _post_image(client, token, route, n=n)).status_code == 400
    expected = [None, 1, 4] if route == "generations" else [1, 1, 4]
    assert seen == [(route, n) for n in expected]                  # forwarded as sent


def test_only_the_image_routes_opt_out_of_the_unlimited_exemption():
    import inspect
    import re

    src = inspect.getsource(lp)

    def body_of(name):
        start = src.index(f"async def {name}(")
        nxt = src.find("\nasync def ", start + 1)
        return src[start: nxt if nxt != -1 else len(src)]

    for name in ("proxy_openai_images", "proxy_openai_image_edits"):
        body = body_of(name)
        assert '_check_budget(config, "openai", db, kind=BUDGET_KIND_IMAGE)' in body, name
        assert body.index("_enforce_image_count(") < body.index("reserve_free_image_slot(db, config.user_id)")
    calls = re.findall(r"_check_budget\(([^)]*)\)", src)
    image_calls = [c for c in calls if "BUDGET_KIND_IMAGE" in c]
    assert len(image_calls) == 2
    # R4b: the chat and Responses routes classify their body before the gate.
    responses = body_of("proxy_responses")
    assert '_check_budget(config, "openai", db, **_budget_kind_kw(body))' in responses
    assert responses.index("_budget_kind_kw(body)") < responses.index("backend.responses")
    chat = body_of("proxy_chat")
    assert "budget_kind_kw = _budget_kind_kw(body)" in chat
    assert chat.count("**budget_kind_kw)") == 2                    # the gate and the fallback gate
    assert lp._UNLIMITED_HONOURED_KINDS == frozenset({lp.BUDGET_KIND_TEXT})
    assert lp._monthly_allocation_refuses(lp._BudgetStanding(False, True), "anything-new") is True


# ── R4b: a hosted tool on an admitted text route is not text ─────────

RESPONSES_URL = "/api/llm/openai/v1/responses"
_FN = {"type": "function", "name": "read_file", "description": "Read a file",
       "parameters": {"type": "object", "properties": {}}}
IMAGE_TOOL_BODIES = {
    "in-tools": {"model": "gpt-5.5", "input": "draw a cat",
                 "tools": [{"type": "image_generation"}]},
    "forced": {"model": "gpt-5.5", "input": "draw a cat",
               "tools": [_FN, {"type": "image_generation", "quality": "high"}],
               "tool_choice": {"type": "image_generation"}},
    "forced-only": {"model": "gpt-5.5", "input": "draw a cat",
                    "tool_choice": {"type": "image_generation"}},
    "allow-listed": {"model": "gpt-5.5", "input": "draw a cat",
                     "tools": [_FN, {"type": "image_generation"}],
                     "tool_choice": {"type": "allowed_tools", "mode": "required",
                                     "tools": [{"type": "image_generation"}]}},
    "streamed": {"model": "gpt-5.5", "input": "draw a cat", "stream": True,
                 "tools": [{"type": "image_generation"}]},
}


def _upstream_spy(monkeypatch):
    """Every OpenAI/Anthropic text backend call (non-stream answers 200).
    The R4b refusals must leave this EMPTY: OpenAI is never called."""
    calls: list = []

    def make(name):
        async def fake(self, body, api_key, *a, **kw):
            calls.append((name, body.get("model")))
            return _ok_response()
        return fake

    def make_stream(name):
        async def fake(self, body, api_key, *a, **kw):
            calls.append((name, body.get("model")))
            yield b"data: [DONE]\n\n"
        return fake

    for cls in (lp.OpenAIBackend, lp.AnthropicBackend):
        for name in ("chat", "responses", "embeddings", "images"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, make(f"{cls.__name__}.{name}"))
        for name in ("chat_stream", "responses_stream"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, make_stream(f"{cls.__name__}.{name}"))
    return calls


@pytest.mark.parametrize("shape", list(IMAGE_TOOL_BODIES))
async def test_unlimited_over_the_cap_cannot_generate_images_through_responses(
        client, monkeypatch, caplog, alerts, shape):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    calls = _upstream_spy(monkeypatch)
    _, token = await _seed(events=[(IN_WINDOW, 1000)])
    res = await client.post(RESPONSES_URL, json=IMAGE_TOOL_BODIES[shape], headers=_auth(token))
    _assert_image_tool_refusal(res)                                # refused before the gate
    assert calls == []                                             # ZERO upstream calls
    assert _over_lines(caplog) == [] and alerts == []              # never reached the gate


async def test_the_image_tool_refusal_holds_with_the_refusal_flag_either_way(client, monkeypatch, alerts):
    calls = _upstream_spy(monkeypatch)
    _, token = await _seed(events=[(IN_WINDOW, 1000)])
    for flag in (False, True):
        monkeypatch.setattr(settings, "unlimited_proxy_budget_refusal_enabled", flag)
        _assert_image_tool_refusal(await client.post(
            RESPONSES_URL, json=IMAGE_TOOL_BODIES["in-tools"], headers=_auth(token)))
    assert calls == []


@pytest.mark.parametrize("plan,spend", [("unlimited", 999), ("builder", 10), ("free", 0)],
                         ids=["unlimited-under-cap", "builder", "free"])
async def test_the_image_tool_is_refused_for_every_tenant_even_under_the_cap(
        client, monkeypatch, alerts, plan, spend):
    """Image-tool charges are not metered by the proxy (only mainline tokens
    are), so an under-cap request could generate images that never count
    toward the image ceiling the owner kept. Refused for everyone, before
    any upstream call, every shape, streamed included; other traffic is
    unaffected (the next test)."""
    calls = _upstream_spy(monkeypatch)
    _, token = await _seed(plan=plan, events=[(IN_WINDOW, spend)])
    for shape in IMAGE_TOOL_BODIES:
        res = await client.post(RESPONSES_URL, json=IMAGE_TOOL_BODIES[shape], headers=_auth(token))
        _assert_image_tool_refusal(res)
    assert calls == [] and alerts == []
    # A plain text request from the same tenant still goes through.
    res = await client.post(RESPONSES_URL, json={"model": "gpt-5.5", "input": "hi"}, headers=_auth(token))
    assert res.status_code == 200, res.text
    assert calls == [("OpenAIBackend.responses", "gpt-5.5")]


async def test_the_agents_own_responses_traffic_over_the_cap_is_still_admitted(client, monkeypatch, alerts):
    """Built with the agent's own converters (openai_agent_service): function
    tools plus a translated tool_choice — the only shapes it sends."""
    from app.services.openai_agent_service import (
        _anthropic_tools_to_responses, _chat_tool_choice_to_responses,
    )

    calls = _upstream_spy(monkeypatch)
    _, token = await _seed(events=[(IN_WINDOW, 5000)])
    tools = _anthropic_tools_to_responses([
        {"name": "web_search", "description": "Search the web",
         "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}},
        {"name": "read_file", "description": "Read", "input_schema": {"type": "object"}},
    ])
    choices = [None, "auto", "required",
               _chat_tool_choice_to_responses({"type": "function", "function": {"name": "read_file"}}),
               _chat_tool_choice_to_responses({"type": "allowed_tools", "allowed_tools": {
                   "mode": "auto", "tools": [{"type": "function", "function": {"name": "web_search"}}]}})]
    for choice in choices:
        body = {"model": "gpt-5.5", "input": "hi", "tools": tools}
        if choice is not None:
            body["tool_choice"] = choice
        res = await client.post(RESPONSES_URL, json=body, headers=_auth(token))
        assert res.status_code == 200, (choice, res.text)
    plain = await client.post(RESPONSES_URL, json={"model": "gpt-5.5", "input": "hi"}, headers=_auth(token))
    assert plain.status_code == 200
    assert len(calls) == len(choices) + 1


@pytest.mark.parametrize("plan", ["free", "builder", None], ids=["free", "builder", "no-balance-row"])
async def test_non_unlimited_image_tool_requests_are_refused_too(client, monkeypatch, alerts, plan):
    """Unmetered: the image tool's per-image charges never reach the window,
    so it is refused for every tenant, over or under the cap."""
    calls = _upstream_spy(monkeypatch)
    _, over = await _seed(plan=plan, events=[(IN_WINDOW, 1000)])
    _, under = await _seed(plan=plan, events=[(IN_WINDOW, 10)])
    for token in (over, under):
        _assert_hosted_tool_refusal(await client.post(
            RESPONSES_URL, json=IMAGE_TOOL_BODIES["forced"], headers=_auth(token)), kind=lp.BUDGET_KIND_IMAGE)
    assert calls == []


async def test_admins_are_refused_the_image_tool_as_well(client, monkeypatch, alerts):
    calls = _upstream_spy(monkeypatch)
    _, admin = await _seed(role="admin", plan="unlimited", events=[(IN_WINDOW, 5000)])
    _assert_hosted_tool_refusal(await client.post(
        RESPONSES_URL, json=IMAGE_TOOL_BODIES["forced"], headers=_auth(admin)))
    assert calls == []


# Spelled out (not read from lp) so collection works on the code before R4b;
# test_the_request_kind_table pins that it matches lp._GATED_TOOL_TYPES.
_ENUMERATED_HOSTED_TOOL_TYPES = [
    "apply_patch", "bash", "code_execution", "code_interpreter", "computer", "computer_use",
    "computer_use_preview", "file_search", "local_shell", "mcp", "memory", "shell",
    "text_editor", "web_fetch", "web_search", "web_search_preview"]
_HOSTED_TOOL_TYPES = _ENUMERATED_HOSTED_TOOL_TYPES + [
    "web_search_preview_2025_03_11", "web_search_2025_08_26", "some_future_tool"]


@pytest.mark.parametrize("tool_type", _HOSTED_TOOL_TYPES)
async def test_every_other_hosted_tool_is_refused_for_every_tenant(
        client, monkeypatch, alerts, tool_type):
    """web search, file search, code interpreter, MCP, computer use, unknown
    types: billed per call by the provider, unmetered by the proxy — refused
    before upstream whatever the tenant or its spend."""
    calls = _upstream_spy(monkeypatch)
    _, over = await _seed(events=[(IN_WINDOW, 1000)])
    _, builder = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    for token in (over, builder):
        for body in ({"model": "gpt-5.5", "input": "hi", "tools": [_FN, {"type": tool_type}]},
                     {"model": "gpt-5.5", "input": "hi", "tools": [_FN],
                      "tool_choice": {"type": tool_type}}):
            _assert_hosted_tool_refusal(await client.post(RESPONSES_URL, json=body, headers=_auth(token)),
                                        kind=lp.BUDGET_KIND_HOSTED_TOOL)
    assert calls == []


async def test_unlimited_over_the_cap_chat_with_a_hosted_tool_is_refused(client, monkeypatch, alerts):
    calls = _upstream_spy(monkeypatch)
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "platform_anthropic_api_key", "sk-ant-test-outbound")
    _, token = await _seed(events=[(IN_WINDOW, 1000), (IN_WINDOW, 3000, "anthropic")])
    fn = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    # OpenAI chat completions: web search is switched on by web_search_options.
    _assert_hosted_tool_refusal(await client.post(CHAT_URL, headers=_auth(token), json={
        **CHAT_BODY, "tools": [fn], "web_search_options": {}}))
    # Anthropic Messages: a server tool would be forwarded as sent — refused too.
    res = await client.post("/api/llm/v1/messages", headers=_auth(token), json={
        "model": "claude-sonnet-4-6", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]})
    _assert_hosted_tool_refusal(res)
    assert calls == []
    # The same chat without the hosted tool is admitted (text, Option A).
    res = await client.post(CHAT_URL, headers=_auth(token), json={**CHAT_BODY, "tools": [fn]})
    assert res.status_code == 200, res.text
    assert calls == [("OpenAIBackend.chat", "gpt-5.5")]


def test_the_request_kind_table():
    kind = lp._request_budget_kind
    TEXT, IMAGE, HOSTED = lp.BUDGET_KIND_TEXT, lp.BUDGET_KIND_IMAGE, lp.BUDGET_KIND_HOSTED_TOOL
    fn_chat = {"type": "function", "function": {"name": "f"}}
    anthropic_client_tool = {"name": "f", "description": "d", "input_schema": {"type": "object"}}
    assert kind({"model": "gpt-5.5", "input": "hi"}) == TEXT
    assert kind({"tools": [_FN, fn_chat, anthropic_client_tool, {"type": "custom", "name": "c"}],
                 "tool_choice": "required"}) == TEXT
    assert kind({"tools": [_FN], "tool_choice": {"type": "function", "name": "read_file"}}) == TEXT
    assert kind({"tools": [anthropic_client_tool], "tool_choice": {"type": "tool", "name": "f"}}) == TEXT
    assert kind({"tools": [fn_chat], "tool_choice": {"type": "allowed_tools", "allowed_tools": {
        "mode": "auto", "tools": [fn_chat]}}}) == TEXT
    assert kind({"prompt": "a plain string field", "tools": []}) == TEXT
    assert kind({"tools": [{"type": "image_generation"}]}) == IMAGE
    assert kind({"tool_choice": {"type": "image_generation"}}) == IMAGE
    assert kind({"tools": [{"type": "web_search"}, {"type": "image_generation"}]}) == IMAGE   # image wins
    assert kind({"tool_choice": {"type": "allowed_tools", "allowed_tools": {
        "tools": [{"type": "image_generation"}]}}}) == IMAGE          # chat-shaped allow-list
    for t in ("web_search", "web_search_preview_2025_03_11", "file_search", "code_interpreter",
              "mcp", "computer_use_preview", "local_shell", "web_search_20250305",
              "web_fetch_20250910", "code_execution_20250825", "bash_20250124",
              "text_editor_20250728", "computer_20250124", "memory_20250818", "never_heard_of_it"):
        assert kind({"tools": [{"type": t}]}) == HOSTED, t
    assert kind({"tools": ["not-a-dict"]}) == HOSTED
    assert kind({"tools": [{"type": {"unhashable": True}}]}) == HOSTED
    assert kind({"web_search_options": {}}) == HOSTED
    assert kind({"mcp_servers": [{"type": "url", "url": "https://mcp.example.test"}]}) == HOSTED
    assert kind({"prompt": {"id": "pmpt_123"}}) == HOSTED
    assert kind(None) == TEXT and kind([]) == TEXT
    assert lp._budget_kind_kw({"input": "hi"}) == {}
    for body in ({"tools": [{"type": "image_generation"}]}, {"tools": [{"type": "web_search"}]},
                 {"web_search_options": {}}, {"prompt": {"id": "pmpt_123"}}):
        with pytest.raises(HTTPException) as info:                 # refused outright, before the gate
            lp._budget_kind_kw(body)
        assert info.value.status_code == 400 and info.value.detail["code"] == "hosted_tool_unsupported"
    # Every enumerated type carries a kind and a cost rationale.
    for t, (k, why) in lp._GATED_TOOL_TYPES.items():
        assert k in (IMAGE, HOSTED) and len(why) > 20, t
    assert lp._GATED_TOOL_TYPES["image_generation"][0] == IMAGE
    assert sorted(set(lp._GATED_TOOL_TYPES) - {"image_generation"}) == _ENUMERATED_HOSTED_TOOL_TYPES
    assert HOSTED not in lp._UNLIMITED_HONOURED_KINDS and IMAGE not in lp._UNLIMITED_HONOURED_KINDS


# ── R4c: a malformed image count is a truthful 400, before anything is held ──

_BAD_JSON_N = ["abc", 0, -1, True, False, 1.5, 2.0, "RAW:Infinity", "RAW:NaN", [], {}]
_BAD_FORM_N = ["abc", "0", "-1", "1.5", "true", "inf", "1e3", " -2 "]


@pytest.mark.parametrize("bad", _BAD_JSON_N, ids=[repr(v) for v in _BAD_JSON_N])
async def test_a_malformed_image_count_is_refused_on_generations(
        client, monkeypatch, free_image_slots, bad):
    seen = _image_provider(monkeypatch)
    _, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    url = "/api/llm/openai/v1/images/generations"
    if isinstance(bad, str) and bad.startswith("RAW:"):
        # httpx will not encode a non-finite float; a client can still send the
        # token, and the server's JSON parser accepts it as a float.
        raw = b'{"prompt": "a cat", "model": "gpt-image-1", "n": ' + bad[4:].encode() + b"}"
        res = await client.post(url, headers={**_auth(token), "Content-Type": "application/json"},
                                content=raw)
    else:
        res = await client.post(url, headers=_auth(token),
                                json={"prompt": "a cat", "model": "gpt-image-1", "n": bad})
    assert res.status_code == 400, res.text                       # never a 500, never silently 1
    detail = res.json()["detail"]
    assert detail["code"] == "image_count_invalid" and detail["max"] == 4
    assert "whole number from 1 to 4" in detail["message"]
    assert seen == [] and free_image_slots == {}                   # nothing reserved or sent


@pytest.mark.parametrize("bad", _BAD_FORM_N, ids=[repr(v) for v in _BAD_FORM_N])
async def test_a_malformed_image_count_is_refused_on_edits(
        client, monkeypatch, free_image_slots, bad):
    seen = _image_provider(monkeypatch)
    _, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    res = await client.post("/api/llm/openai/v1/images/edits", headers=_auth(token),
                            data={"prompt": "a hat", "model": "gpt-image-1", "n": bad},
                            files={"image": ("in.png", b"\x89PNG\r\n\x1a\n", "image/png")})
    assert res.status_code == 400, res.text
    assert res.json()["detail"]["code"] == "image_count_invalid"
    assert seen == [] and free_image_slots == {}


@pytest.mark.parametrize("route", ["generations", "edits"])
async def test_valid_image_counts_are_forwarded_unchanged(client, monkeypatch, free_image_slots, route):
    seen = _image_provider(monkeypatch)
    _, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    for n in (None, 1, 2, 4, "3"):
        assert (await _post_image(client, token, route, n=n)).status_code == 400   # reached OpenAI
    got = [n for _, n in seen]
    assert got == ([None, 1, 2, 4, "3"] if route == "generations" else [1, 1, 2, 4, 3])



# ── R4d: a malformed tool_choice is a truthful 400, never a 500 ─────────────

_BAD_TOOL_CHOICE_TYPES = [[], {}, 7, None, ["image_generation"]]


@pytest.mark.parametrize("bad", _BAD_TOOL_CHOICE_TYPES, ids=[repr(v) for v in _BAD_TOOL_CHOICE_TYPES])
def test_a_tool_choice_whose_type_is_not_a_string_is_a_400_not_a_500(bad):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as info:
        lp._request_budget_kind({"model": "gpt-5", "input": "hi", "tool_choice": {"type": bad}})
    assert info.value.status_code == 400
    assert info.value.detail["code"] == "tool_choice_invalid"


@pytest.mark.parametrize("route,body", [
    ("/api/llm/openai/v1/responses", {"model": "gpt-5", "input": "hi"}),
    ("/api/llm/openai/v1/chat/completions", {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}),
], ids=["responses", "chat-completions"])
async def test_a_malformed_tool_choice_over_the_wire_is_400_with_zero_upstream_calls(
        client, monkeypatch, alerts, route, body):
    served = _fake_text_provider(monkeypatch)
    _, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    res = await client.post(route, headers=_auth(token), json={**body, "tool_choice": {"type": []}})
    assert res.status_code == 400, res.text
    assert res.json()["detail"]["code"] == "tool_choice_invalid"
    assert served == []


# ── the call that CROSSES a multiple reports it (post-spend) ────────────────

async def _settle_alerts(n: int = 20):
    for _ in range(n):
        await asyncio.sleep(0)


async def test_a_single_call_that_crosses_1x_is_reported_with_no_later_call(monkeypatch, alerts, caplog):
    """The preflight sees the spend BEFORE the call; the event that crosses
    1x used to go unreported until the next gate call — which may never
    come. `_log_event` now reports after the spend is recorded."""
    from app.db import async_session_maker
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    uid, _ = await _seed(events=[(IN_WINDOW, 999)])            # 1c under the allocation
    async with async_session_maker() as db:
        await lp._log_event(db, uid, "openai", "gpt-5.5", "chat", 100, 10, Decimal("2"), 12)
    await _settle_alerts()
    assert [m for m in _over_lines(caplog)] and any("multiple=1" in m for m in _over_lines(caplog))
    assert len(alerts) == 1 and alerts[0][0] == "unlimited_over_allocation"
    # The same crossing is not reported twice by the next gate call.
    async with async_session_maker() as db:
        from app.db import AgentConfig
        cfg = (await db.execute(select(AgentConfig).where(AgentConfig.user_id == uid))).scalar_one()
        assert await lp._check_budget(cfg, "openai", db) is None
    await _settle_alerts()
    assert len(alerts) == 1


async def test_a_system_operation_or_a_failed_call_reports_nothing_after_spend(monkeypatch, alerts):
    from app.db import async_session_maker
    uid, _ = await _seed(events=[(IN_WINDOW, 999)])
    async with async_session_maker() as db:
        await lp._log_event(db, uid, "openai", "gpt-5.5", "chat", 100, 10, Decimal("2"), 12,
                            operation_type="system.cache_warm")
        await lp._log_event(db, uid, "openai", "gpt-5.5", "chat", 100, 10, Decimal("2"), 12,
                            status="error")
    await _settle_alerts()
    assert alerts == []


# ── the free "warm" exemption: OFF by default; bounded and allow-listed when on ──
#
# `X-Toup-Operation-Type: system.cache_warm` exempts a call from the credit
# charge and the monthly window. Any agent-token holder can send it, so
# (2026-09-30 audit) the exemption is a platform switch that defaults OFF —
# a warm is served, billed and counted like any request — and when ON the
# body must be exactly the shape the agent's warm sends: no hidden provider
# context (previous_response_id / conversation / prompt / files), function
# tools only, gpt-* model, a serialised-size and a daily-rate bound.

WARM = "system.cache_warm"


def _warm_body(chars: int) -> dict:
    return {"model": "gpt-5.5", "max_output_tokens": 16, "tool_choice": "none",
            "input": [{"role": "user", "content": "x" * chars}]}


@pytest.fixture
def exemption_on(monkeypatch):
    monkeypatch.setattr(settings, "llm_proxy_system_operation_exemption", True)
    monkeypatch.setattr(lp, "_warm_calls", {})


def test_the_exemption_is_off_by_default_so_a_perfect_warm_is_billed(monkeypatch):
    from app.config import Settings
    assert Settings.model_fields["llm_proxy_system_operation_exemption"].default is False
    monkeypatch.setattr(settings, "llm_proxy_system_operation_exemption", False)
    monkeypatch.setattr(lp, "_warm_calls", {})
    assert lp._system_operation_for(WARM, _warm_body(1), user_id="u-off") is None
    monkeypatch.setattr(settings, "llm_proxy_system_operation_exemption", True)
    assert lp._system_operation_for(WARM, _warm_body(1), user_id="u-off") == WARM


async def test_with_the_switch_off_a_warm_over_the_wire_is_charged_and_counted(client, monkeypatch):
    """Over the wire, default settings: the header changes nothing — the row
    is user-attributable (so it is charged and counts toward the window)."""
    from app.db import LLMProxyEvent, async_session_maker
    served = _fake_text_provider(monkeypatch)
    monkeypatch.setattr(lp, "_warm_calls", {})
    assert settings.llm_proxy_system_operation_exemption is False
    uid, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    res = await client.post(RESPONSES_URL, headers={**_auth(token), lp.OPERATION_TYPE_HEADER: WARM},
                            json=_warm_body(1))
    assert res.status_code == 200, res.text
    async with async_session_maker() as db:
        rows = (await db.execute(select(LLMProxyEvent).where(LLMProxyEvent.user_id == uid,
                                                             LLMProxyEvent.endpoint == "responses"))).scalars().all()
    assert [r.operation_type for r in rows if r.model == "gpt-5.5"] == [None]
    assert len(served) == 1
    # …and the same warm at the allocation is refused like any other text
    # call for a non-Unlimited tenant: no free path past the window.
    uid2, token2 = await _seed(plan="builder", events=[(IN_WINDOW, 1000)])
    res = await client.post(RESPONSES_URL, headers={**_auth(token2), lp.OPERATION_TYPE_HEADER: WARM},
                            json=_warm_body(1))
    assert res.status_code == 429, res.text
    assert res.json()["detail"]["error"] == REASON
    assert len(served) == 1


def test_a_genuine_warm_is_exempt_but_a_huge_one_is_billed(exemption_on):
    assert lp._system_operation_for(WARM, _warm_body(160_000), user_id="u-warm-1") == WARM
    assert lp._system_operation_for(WARM, _warm_body(lp._WARM_MAX_INPUT_CHARS + 1),
                                    user_id="u-warm-2") is None
    nested = {**_warm_body(0), "input": [{"role": "user", "content": [{"type": "input_text", "text": "y" * 500_000}]}]}
    assert lp._system_operation_for(WARM, nested, user_id="u-warm-3") is None
    # Text hidden OUTSIDE the input item is billed too: instructions and
    # tools are forwarded verbatim and charged as mainline input.
    for field, value in (("instructions", "i" * 500_000),
                         ("tools", [{"type": "function", "name": "f", "description": "d" * 500_000}])):
        body = {**_warm_body(1), field: value}
        assert lp._system_operation_for(WARM, body, user_id=f"u-warm-{field}") is None, field
    # The bound is on serialised UTF-8 BYTES, not characters: 200k
    # three-byte characters are 600k bytes on the wire.
    assert lp._system_operation_for(WARM, _warm_body(0) | {"instructions": "\u4e2d" * 200_000},
                                    user_id="u-warm-utf8") is None
    genuine = {**_warm_body(1), "instructions": "h" * 150_000,
               "tools": [{"type": "function", "name": "read", "description": "r" * 10_000}],
               "stream": True, "store": False, "include": ["reasoning.encrypted_content"],
               "temperature": 0.7, "prompt_cache_key": "k", "prompt_cache_retention": "24h",
               "safety_identifier": "s"}
    assert lp._system_operation_for(WARM, genuine, user_id="u-warm-genuine") == WARM
    gpt6 = {**genuine, "model": "gpt-6-sol", "prompt_cache_options": {"ttl": "30m"}}
    gpt6.pop("prompt_cache_retention")
    assert lp._system_operation_for(WARM, gpt6, user_id="u-warm-gpt6") == WARM


@pytest.mark.parametrize("extra", [
    # Hidden provider-side context: a tiny body continuing a stored
    # response, a conversation or a stored prompt template is a full-sized
    # billed request the proxy cannot see.
    {"previous_response_id": "resp_abc"},
    {"conversation": "conv_abc"},
    {"conversation": {"id": "conv_abc"}},
    {"prompt": {"id": "pmpt_abc", "variables": {"doc": "x"}}},
    {"include": ["file_search_call.results"]},
    {"include": "reasoning.encrypted_content"},          # not a list: not the warm's shape
    {"store": True},
    # Inputs that are fetched and billed provider-side, or replay stored items.
    {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file_abc"}]}]},
    {"input": [{"role": "user", "content": [{"type": "input_file", "file_url": "https://x/y.pdf"}]}]},
    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]}]},
    {"input": [{"role": "user", "content": [{"type": "input_text", "text": "a", "extra": 1}]}]},
    {"input": [{"type": "item_reference", "id": "msg_abc"}]},
    {"input": [{"role": "user", "content": ".", "attachments": [{"file_id": "f"}]}]},
    # Hosted tools are billed per use by the provider.
    {"tools": [{"type": "web_search"}]},
    {"tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]},
    {"tools": [{"type": "function", "name": "f"}, {"type": "mcp", "server_url": "https://x"}]},
    {"tools": "none"},
    # Not a KNOWN warm model: a pro tier ($30/M input, no cached-input
    # rate, absent from the pricing table on purpose), an image/audio id
    # wearing the gpt- prefix, an unpriced id, or another family.
    {"model": "gpt-5.5-pro"},
    {"model": "gpt-5.6-terra-pro"},
    {"model": "gpt-4.5-preview"},
    {"model": "gpt-image-2"},
    {"model": "gpt-4o-realtime-preview"},
    {"model": "gpt-4o-transcribe"},
    {"model": "gpt-6"},
    {"model": "claude-opus-5-5"},
    {"model": "o3"},
    {"model": ["gpt-5.5"]},
    # Any field outside the exact allowlist.
    {"metadata": {"k": "v"}},
    {"reasoning": {"effort": "high"}},
    {"user": "u"},
    {"text": {"format": {"type": "text"}}},
    {"truncation": "auto"},
    {"service_tier": "priority"},
    {"background": True},
    # Allowed fields pinned to the VALUE the warm sends.
    {"max_output_tokens": 33},
    {"max_output_tokens": "16"},
    {"max_output_tokens": 16.0},
    {"max_output_tokens": True},
    {"tool_choice": "required"},
    {"tool_choice": {"type": "none"}},
    {"store": 0},
    {"stream": "yes"},
    {"instructions": ["x"]},
    {"temperature": 2.5},
    {"temperature": "0.7"},
    {"prompt_cache_retention": "7d"},
    {"prompt_cache_options": {"ttl": "30m", "x": "y"}},
    {"prompt_cache_options": {"ttl": "24h"}},
    {"prompt_cache_key": "k" * 257},
    {"safety_identifier": ""},
    {"include": [1]},
    {"tools": [{"type": "function", "name": "f", "server_url": "https://x"}]},
    {"tools": [{"type": "function"}]},
    {"input": [{"role": "system", "content": "."}]},
    {"input": [{"role": "developer", "content": "."}]},
])
def test_a_warm_that_reaches_outside_its_exact_shape_is_billed(exemption_on, extra):
    body = {**_warm_body(1), **extra}
    assert lp._system_operation_for(WARM, body, user_id="u-shape") is None, extra
    # The failing shape did not spend one of the day's warm slots.
    assert lp._warm_calls.get("u-shape") in (None, deque()) or len(lp._warm_calls["u-shape"]) == 0


def test_more_warms_than_a_day_allows_are_billed(exemption_on, monkeypatch):
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(lp, "_monotonic", lambda: clock["t"])
    for i in range(lp._WARM_MAX_PER_DAY):
        assert lp._system_operation_for(WARM, _warm_body(1000), user_id="u-rate") == WARM, i
    assert lp._system_operation_for(WARM, _warm_body(1000), user_id="u-rate") is None
    assert lp._system_operation_for(WARM, _warm_body(1000), user_id="u-other") == WARM
    clock["t"] += 86_401                                             # a day later the window is clear
    assert lp._system_operation_for(WARM, _warm_body(1000), user_id="u-rate") == WARM


async def test_a_warm_the_provider_bills_as_more_than_a_warm_is_filed_as_user_traffic(exemption_on):
    """Second decision from the provider's usage: adversarial text under
    the byte cap still tokenises at ~1 token/byte (measured), ten times a
    genuine ~40k-token head. The row is then user-attributable — charged
    and counted — whatever the header said."""
    from app.db import LLMProxyEvent, async_session_maker
    uid, _ = await _seed(plan="builder", events=[])
    async with async_session_maker() as db:
        await lp._log_event(db, uid, "openai", "gpt-5.5", "responses", 40_000, 16, Decimal("20"), 12,
                            operation_type=WARM)
        await lp._log_event(db, uid, "openai", "gpt-5.5", "responses", lp._WARM_MAX_INPUT_TOKENS + 1, 16,
                            Decimal("30"), 12, operation_type=WARM)
        await lp._log_event(db, uid, "openai", "gpt-5.5", "responses", 1_000, 33, Decimal("1"), 12,
                            operation_type=WARM)
        await db.commit()
    async with async_session_maker() as db:
        rows = (await db.execute(select(LLMProxyEvent).where(LLMProxyEvent.user_id == uid,
                                                             LLMProxyEvent.endpoint == "responses"))).scalars().all()
    ops = sorted((r.input_tokens, r.output_tokens, r.operation_type) for r in rows)
    assert ops == [(1_000, 33, None), (40_000, 16, WARM), (lp._WARM_MAX_INPUT_TOKENS + 1, 16, None)]
    assert lp._warm_operation_after_usage(None, 10**9, 10**6) is None
    assert lp._warm_operation_after_usage("system.other", 10**9, 1) == "system.other"   # only the warm re-decides
    assert lp._warm_operation_after_usage(WARM, "many", 1) is None


async def test_over_the_wire_a_warm_with_large_provider_usage_is_billed(client, monkeypatch, exemption_on):
    from app.db import LLMProxyEvent, async_session_maker

    async def big(self, body, api_key):
        return types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"input_tokens": lp._WARM_MAX_INPUT_TOKENS + 1, "output_tokens": 16,
                                    "total_tokens": lp._WARM_MAX_INPUT_TOKENS + 17}, "output": []})
    monkeypatch.setattr(lp.OpenAIBackend, "responses", big)
    uid, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    res = await client.post(RESPONSES_URL, headers={**_auth(token), lp.OPERATION_TYPE_HEADER: WARM},
                            json=_warm_body(1))
    assert res.status_code == 200, res.text
    async with async_session_maker() as db:
        rows = (await db.execute(select(LLMProxyEvent).where(LLMProxyEvent.user_id == uid,
                                                             LLMProxyEvent.endpoint == "responses"))).scalars().all()
    assert [(r.input_tokens, r.operation_type) for r in rows] == [(lp._WARM_MAX_INPUT_TOKENS + 1, None)]


async def test_the_daily_warm_count_is_fleet_wide_from_the_rows(client, monkeypatch, exemption_on):
    """The in-memory counter is per replica and resets on restart; the rows
    the proxy writes are the durable count. Eleven warms in the last day:
    the twelfth is exempt; twelve: the next is billed."""
    from app.db import LLMProxyEvent, async_session_maker
    served = _fake_text_provider(monkeypatch)
    uid, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])

    async def add_warm_rows(n, when):
        async with async_session_maker() as db:
            for _ in range(n):
                db.add(LLMProxyEvent(user_id=uid, provider="openai", model="gpt-5.5", endpoint="responses",
                                     input_tokens=1, output_tokens=1, cost_cents=Decimal("0.1"), latency_ms=1,
                                     status="ok", operation_type=WARM, created_at=when))
            await db.commit()

    async def ops():
        async with async_session_maker() as db:
            rows = (await db.execute(select(LLMProxyEvent).where(
                LLMProxyEvent.user_id == uid, LLMProxyEvent.endpoint == "responses"))).scalars().all()
        return [r.operation_type for r in rows]

    hdr = {**_auth(token), lp.OPERATION_TYPE_HEADER: WARM}
    now = lp.datetime.utcnow()                                                 # the proxy's (frozen) clock
    await add_warm_rows(lp._WARM_MAX_PER_DAY - 1, now - timedelta(hours=1))
    await add_warm_rows(5, now - timedelta(hours=25))                          # yesterday: not counted
    res = await client.post(RESPONSES_URL, headers=hdr, json=_warm_body(1))
    assert res.status_code == 200, res.text
    assert (await ops()).count(WARM) == lp._WARM_MAX_PER_DAY + 5              # the 12th of the day was exempt
    res = await client.post(RESPONSES_URL, headers=hdr, json=_warm_body(1))   # the 13th is billed
    assert res.status_code == 200, res.text
    recorded = await ops()
    assert recorded.count(WARM) == lp._WARM_MAX_PER_DAY + 5 and recorded.count(None) == 1
    assert len(served) == 2


async def test_over_the_wire_only_the_exact_warm_shape_is_exempt(client, monkeypatch, exemption_on, alerts):
    """Switch ON. Same agent token, five requests wearing the header: the
    genuine warm is filed as system.cache_warm; a huge input, huge
    instructions, a previous_response_id continuation and a conversation
    continuation are all filed as user traffic (charged and counted)."""
    from app.db import LLMProxyEvent, async_session_maker
    served = _fake_text_provider(monkeypatch)
    uid, token = await _seed(plan="builder", events=[(IN_WINDOW, 10)])
    hdr = {**_auth(token), lp.OPERATION_TYPE_HEADER: WARM}

    async def ops():
        async with async_session_maker() as db:
            rows = (await db.execute(select(LLMProxyEvent).where(
                LLMProxyEvent.user_id == uid, LLMProxyEvent.endpoint == "responses"))).scalars().all()
        return [r.operation_type for r in rows if r.model == "gpt-5.5"]

    res = await client.post(RESPONSES_URL, headers=hdr, json=_warm_body(1))
    assert res.status_code == 200, res.text
    assert await ops() == [WARM]
    for body in (
        _warm_body(lp._WARM_MAX_INPUT_CHARS + 1),
        {**_warm_body(1), "instructions": "i" * (lp._WARM_MAX_INPUT_CHARS + 1)},
        {**_warm_body(1), "previous_response_id": "resp_hidden_context"},
        {**_warm_body(1), "conversation": "conv_hidden_context"},
    ):
        res = await client.post(RESPONSES_URL, headers=hdr, json=body)
        assert res.status_code == 200, res.text
    recorded = await ops()
    assert recorded.count(WARM) == 1 and recorded.count(None) == 4 and len(recorded) == 5
    assert len(served) == 5
    # The four billed rows are the ones the window gate sums (its filter:
    # operation_type NULL or not "system.*"); the warm row is outside it.
    # That a billed warm is then refused at the allocation is proven by
    # test_with_the_switch_off_a_warm_over_the_wire_is_charged_and_counted.
    async with async_session_maker() as db:
        counted = (await db.execute(select(LLMProxyEvent).where(
            LLMProxyEvent.user_id == uid, LLMProxyEvent.endpoint == "responses",
            (LLMProxyEvent.operation_type.is_(None))
            | (~LLMProxyEvent.operation_type.startswith("system."))))).scalars().all()
    assert len(counted) == 4


# ── low-cost images advance the window spend and stop at the ceiling ──────

def _image_success_provider(monkeypatch):
    """The OpenAI image backends answering a real (tiny) image so the route
    records the cost; records each request's n."""
    seen: list = []

    def ok():
        return types.SimpleNamespace(status_code=200, headers={}, content=b"{}", text="{}",
                                     json=lambda: {"created": 1, "data": [{"b64_json": "AAAA"}]})

    async def images(self, body, api_key):
        seen.append(("generations", body.get("n")))
        return ok()

    async def images_edit(self, data, files, api_key):
        seen.append(("edits", data.get("n")))
        return ok()

    monkeypatch.setattr(lp.OpenAIBackend, "images", images)
    monkeypatch.setattr(lp.OpenAIBackend, "images_edit", images_edit)
    return seen


@pytest.mark.parametrize("route", ["generations", "edits"])
async def test_repeated_low_quality_images_advance_spend_and_stop_at_the_ceiling(
        client, monkeypatch, alerts, free_image_slots, route):
    """A 0.6c low-quality image was recorded as int(float(0.6)) = 0c, so
    repeated low images never reached the allocation. The event column
    stores fractional cents; the cost is recorded as the Decimal it is."""
    from app.db import LLMProxyEvent, async_session_maker
    from app.services.credit_service import image_generation_cost_cents
    seen = _image_success_provider(monkeypatch)
    each = Decimal(str(image_generation_cost_cents("1024x1024", "low", "gpt-image-2")))
    assert 0 < each < 1, each                                      # sub-cent: the bug's precondition
    budget = 2
    uid, token = await _seed(plan="builder", budget=budget)
    url = f"/api/llm/openai/v1/images/{route}"

    async def post():
        if route == "edits":
            return await client.post(url, headers=_auth(token),
                                     data={"prompt": "a hat", "model": "gpt-image-2", "size": "1024x1024", "quality": "low"},
                                     files={"image": ("in.png", b"\x89PNG\r\n\x1a\n", "image/png")})
        return await client.post(url, headers=_auth(token),
                                 json={"prompt": "a cat", "model": "gpt-image-2", "size": "1024x1024", "quality": "low"})

    admitted = 0
    for _ in range(int(budget / each) + 3):
        res = await post()
        if res.status_code == 429:
            break
        assert res.status_code == 200, res.text
        admitted += 1
    else:
        pytest.fail("never stopped at the ceiling")
    _assert_typed_refusal(res)
    assert admitted == int(budget / each) + (1 if budget % each else 0)   # the crossing call itself is served
    async with async_session_maker() as db:
        rows = (await db.execute(select(LLMProxyEvent).where(
            LLMProxyEvent.user_id == uid, LLMProxyEvent.endpoint == "images"))).scalars().all()
    assert len(rows) == admitted and all(Decimal(str(r.cost_cents)) == each for r in rows)
    assert len(seen) == admitted
