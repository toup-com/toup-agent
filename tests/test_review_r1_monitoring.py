"""Review round 1 (2026-09-14) — the monitoring findings, pinned by behaviour.

  * persist-gap was structurally dead: `turns_completed` and
    `assistant_rows_written` were incremented on adjacent lines, so the
    difference the alert reads was identically zero. Now: turns on ENTRY to
    `_save_messages`, rows after the caller's COMMIT (`_note_turn_persisted`).
  * `_probe_agent_health` returned a 4-tuple on the no-URL branch; the caller
    unpacks three, and one provisioning row without a port aborted the sweep.
  * agent-output / day-chats alerted on cumulative lifetime counters, so one
    event paged every rate-limit window for the life of the container.
  * `day_chat_resolve_failures` was published and folded by nobody.
  * alerts were awaited inside the per-container loop with the platform DB
    session held.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_review_r1_monitoring.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import alerting, container_monitor as cm, health_signals as hs

_APP = Path(__file__).resolve().parents[1] / "app"


@pytest.fixture(autouse=True)
def _clean():
    hs.reset_for_tests()
    alerting.reset_for_tests()
    cm.reset_signal_state_for_tests()
    yield
    hs.reset_for_tests()
    alerting.reset_for_tests()
    cm.reset_signal_state_for_tests()


@pytest.fixture
def alerts(monkeypatch):
    got: list[dict] = []

    async def recorder(category, level, message, *, subject=None, min_interval_s=600):
        got.append({"category": category, "level": level, "message": message,
                    "subject": subject})
        return True

    monkeypatch.setattr(alerting, "send_infra_alert", recorder)
    monkeypatch.setattr(cm, "send_infra_alert", recorder)
    return got


# ── persist-gap counters ─────────────────────────────────────────────


def _runner():
    from app.agent.agent_runner import AgentRunner

    r = AgentRunner.__new__(AgentRunner)
    r.tools = SimpleNamespace(_last_media=None, _last_pending_action=None, pending_attachments=[])
    return r


@pytest.mark.asyncio
async def test_a_save_that_fails_leaves_turns_ahead_of_rows():
    """The anti-vacuity case the old wiring could never produce: the turn is
    counted on entry, the row is not counted because nothing was committed."""
    r = _runner()
    db = SimpleNamespace(execute=AsyncMock(side_effect=RuntimeError("db down")))

    with pytest.raises(RuntimeError):
        await r._save_messages(
            db=db, session_id="s", user_id="u" * 36, user_message="hi",
            assistant_response="yo", tokens_input=1, tokens_output=1,
            model="m", processing_time_ms=1,
        )

    assert hs.get("turns_completed") == 1
    assert hs.get("assistant_rows_written") == 0, (
        "a row was counted for a save that never reached a commit"
    )


def test_rows_are_counted_after_the_commit_and_media_with_them():
    from app.agent.agent_runner import _note_turn_persisted

    _note_turn_persisted({"asst_message_id": "a", "media": None})
    assert (hs.get("assistant_rows_written"), hs.get("media_persisted")) == (1, 0)
    _note_turn_persisted({"asst_message_id": "b", "media": {"video_id": "x"}})
    assert (hs.get("assistant_rows_written"), hs.get("media_persisted")) == (2, 1)
    _note_turn_persisted(None)  # a caller with nothing persisted still counts the row it committed
    assert hs.get("assistant_rows_written") == 3


def test_the_two_counters_are_not_on_adjacent_lines():
    """Structure, in addition to behaviour: the row counter must live AFTER
    the `await db.commit()` at the single call site, not inside the save."""
    src = (_APP / "agent" / "agent_runner.py").read_text()
    save = src.split("async def _save_messages(")[1].split("\n    async def ")[0]
    assert 'incr("turns_completed")' in save
    assert 'incr("assistant_rows_written")' not in save, (
        "assistant_rows_written is counted inside _save_messages again — before the commit"
    )
    call = src.index("_persisted = await self._save_messages(")
    commit = src.index("await db.commit()", call)
    noted = src.index("_note_turn_persisted(_persisted)", call)
    assert commit < noted, "the row is counted before the commit"
    assert "[media-persist] MISS" not in src, "the unreachable MISS guard is back"


# ── the probe's shape ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_probe_answers_three_wide_without_a_url(monkeypatch):
    class _Res:
        def scalar_one_or_none(self):
            return None

    class _DB:
        async def execute(self, *a, **k):
            return _Res()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr("app.db.database.async_session_maker", lambda: _DB())
    monkeypatch.setattr(cm.settings, "docker_host_ip", "10.0.0.1", raising=False)
    container = SimpleNamespace(id="c0", user_id="u" * 36, host_port=None,
                                container_name="toup-agent-provisioning")

    out = await cm._probe_agent_health(container)

    assert len(out) == 3, f"probe returned {len(out)} values; the caller unpacks three"
    healthy, db_ok, turn_ready = out
    assert healthy is False and db_ok is None and turn_ready is None


# ── the fold: deltas, and decisions rather than sends ────────────────


def _c(i=0):
    return SimpleNamespace(id=f"c{i}", user_id=f"user-{i}-abcdefgh", container_name=f"toup-agent-{i}")


def test_the_fold_decides_and_does_not_send(alerts):
    out = cm._fold_health_signals(_c(), {"channel_tag_prefixed_replies": 1})
    assert inspect.iscoroutine(out) is False and isinstance(out, list)
    assert out and out[0]["category"] == "agent-output"
    assert alerts == [], "the fold sent an alert itself — it must only decide"


@pytest.mark.asyncio
async def test_alerts_are_dispatched_after_the_sweep(alerts):
    payload = [dict(category="agent-output", level="warning", message="m", subject="s",
                    min_interval_s=1)]
    await cm._dispatch_signal_alerts(payload)
    assert [a["category"] for a in alerts] == ["agent-output"]


def test_a_single_leak_pages_once_not_forever():
    """Lifetime counter 1 on tick 1 and still 1 on ticks 2..5: one decision."""
    decided = []
    for _ in range(5):
        decided += cm._fold_health_signals(_c(), {"channel_tag_prefixed_replies": 1})
    assert len([d for d in decided if d["category"] == "agent-output"]) == 1
    # A NEW leak pages again.
    decided = cm._fold_health_signals(_c(), {"channel_tag_prefixed_replies": 2})
    assert any(d["category"] == "agent-output" for d in decided)


def test_a_new_impossible_day_pages_at_once_and_a_repaired_one_stops():
    c = _c()
    # Tick 1: tenant produced 3 impossible rows, the heal repaired them (gauge 0).
    d1 = cm._fold_health_signals(c, {"future_dated_day_chats_seen": 3, "future_dated_day_chats": 0})
    assert any(d["category"] == "day-chats" for d in d1)
    # Tick 2: nothing new — no decision, even though the lifetime counter is 3.
    d2 = cm._fold_health_signals(c, {"future_dated_day_chats_seen": 3, "future_dated_day_chats": 0})
    assert not any(d["category"] == "day-chats" for d in d2)


def test_an_unrepaired_day_pages_after_three_ticks():
    c = _c()
    snap = {"future_dated_day_chats_seen": 1, "future_dated_day_chats": 1}
    first = cm._fold_health_signals(c, snap)          # the SEEN growth pages on tick 1
    assert any(d["category"] == "day-chats" for d in first)
    second = cm._fold_health_signals(c, snap)         # tick 2: streak 2 — quiet
    assert not any(d["category"] == "day-chats" for d in second)
    third = cm._fold_health_signals(c, snap)          # tick 3: still there — page
    assert any(d["category"] == "day-chats" for d in third)


def test_resolve_failures_page_when_they_repeat():
    c = _c()
    assert not any(d["category"] == "day-chats"
                   for d in cm._fold_health_signals(c, {"day_chat_resolve_failures": 1}))
    assert any(d["category"] == "day-chats"
               for d in cm._fold_health_signals(c, {"day_chat_resolve_failures": 2}))
    # Two at once on a fresh container also pages.
    assert any(d["category"] == "day-chats"
               for d in cm._fold_health_signals(_c(7), {"day_chat_resolve_failures": 2}))


def test_the_new_counters_are_declared():
    snap = hs.snapshot()
    for name in ("day_chat_resolve_failures", "future_dated_day_chats_seen"):
        assert name in snap, f"{name} is incremented but not declared — an old image and a zero look alike"


@pytest.mark.asyncio
async def test_a_forced_gauge_refresh_ignores_the_ttl():
    calls = []

    async def fn():
        calls.append(1)
        return len(calls)

    hs.register_gauge("_t_gauge", fn, ttl_s=3600)
    try:
        await hs.refresh_gauges(only="_t_gauge")
        await hs.refresh_gauges(only="_t_gauge")           # inside the TTL: no re-measure
        assert len(calls) == 1
        await hs.refresh_gauges(only="_t_gauge", force=True)
        assert len(calls) == 2 and hs.get("_t_gauge") == 2
    finally:
        hs._GAUGES.pop("_t_gauge", None)


def test_retired_containers_leave_the_per_container_state():
    cm._fold_health_signals(_c(1), {"channel_tag_prefixed_replies": 1})
    cm._fold_health_signals(_c(2), {"future_dated_day_chats": 1})
    assert "c1" in cm._signal_prev and "c2" in cm._future_days_streak
    cm._prune_signal_state({"c2"})
    assert "c1" not in cm._signal_prev, "a retired container's snapshot is kept forever"
    assert "c2" in cm._future_days_streak, "a live container's streak was pruned"


@pytest.mark.parametrize("mode,registers", [("platform", False), ("agent", True), ("monolith", True)])
def test_the_future_day_gauge_registers_only_where_a_tenant_table_lives(monkeypatch, mode, registers):
    """The gauge measures a TENANT table. The platform process only proxies,
    and its `day_chats` table is a monolith leftover shared across every
    user — scanning it once a minute for a counter nobody reads was the R3
    finding."""
    from app.api import day_chats as dc
    from app.config import settings

    monkeypatch.setattr(settings, "run_mode", mode, raising=False)
    assert dc._gauge_process() is registers


@pytest.mark.asyncio
async def test_a_new_gauge_is_due_even_on_a_host_that_just_booted(monkeypatch):
    """CI caught this one: `at` started at 0.0, and on a runner whose
    monotonic clock read less than the TTL a brand-new gauge was never due —
    `time.monotonic() - 0.0 < ttl`. A gauge that has never been measured is
    due by definition."""
    monkeypatch.setattr(hs.time, "monotonic", lambda: 100.0)  # a host up for 100 s
    calls = []

    async def fn():
        calls.append(1)
        return 7

    hs.register_gauge("_t_fresh", fn, ttl_s=3600)
    try:
        assert hs.gauge_due("_t_fresh") is True
        await hs.refresh_gauges(only="_t_fresh")
        assert calls == [1] and hs.get("_t_fresh") == 7
        hs.reset_for_tests()                       # zeroes AND marks never-measured
        assert hs.gauge_due("_t_fresh") is True
    finally:
        hs._GAUGES.pop("_t_fresh", None)
