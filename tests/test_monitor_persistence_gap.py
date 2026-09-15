"""A turn that completes and writes no assistant row must page somebody (D10).

The 2026-09-14 incident produced no alert of any kind. Turns completed, rows
landed in a day chat no client could reach, the media card was dropped on every
reload and `/api/day-chats` answered 500 sixty-five times in three hours — and
the monitor's verdict for every one of those containers was "healthy", because
`/agent/health` answers about the PROCESS.

D10 folds the process-lifetime counters into `check_all_containers` and alerts
through `send_infra_alert` with four categories: `persist-gap`,
`channel-orphan`, `agent-output`, `day-chats`.

Three properties this file exists to hold, each of which is a way the alert
could be worse than nothing:

  * a gap of one is NOISE. The counters are sampled between ticks and a turn
    in flight at sample time is an off-by-one, not an outage.
  * the alert is keyed per SUBJECT (an 8-char user prefix). One permanently
    stuck tenant must not blanket-suppress every other tenant's alert for the
    whole window — that is L3-8, already learned once in alerting.py.
  * the loop is LEADER-GATED. It restarts containers and writes
    `status='error'`; it ran unelected on both Railway replicas until
    2026-09-12.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_monitor_persistence_gap.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import pytest

from app.services import alerting, container_monitor as cm


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    alerting.reset_for_tests()
    for name in ("_persist_gap_counts", "_signal_gap_counts", "_failure_counts",
                 "_db_down_counts", "_turn_down_counts"):
        d = getattr(cm, name, None)
        if isinstance(d, dict):
            d.clear()
    yield
    alerting.reset_for_tests()


@pytest.fixture
def alerts(monkeypatch):
    """Record every send_infra_alert call, in both namespaces — the module
    defines it and container_monitor may have imported the name directly."""
    got: list[dict] = []

    async def recorder(category, level, message, *, subject=None, min_interval_s=600):
        got.append({"category": category, "level": level, "message": message,
                    "subject": subject})
        return True

    monkeypatch.setattr(alerting, "send_infra_alert", recorder)
    if hasattr(cm, "send_infra_alert"):
        monkeypatch.setattr(cm, "send_infra_alert", recorder)
    # The streaks count ticks across calls; every scenario starts cold.
    reset = getattr(cm, "reset_signal_state_for_tests", None)
    if callable(reset):
        reset()
    return got


def _containers(monkeypatch, n=1):
    """Stand `check_all_containers` up over a fake DB returning `n` rows."""
    from app.db.models import ManagedContainer

    rows = [
        ManagedContainer(
            id=f"c{i}", user_id=f"user-{i}-abcdefgh", container_name=f"toup-agent-{i}",
            status="running", host_port=9000 + i,
        )
        for i in range(n)
    ]

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return rows

    class _DB:
        async def execute(self, *a, **k):
            return _Result()

        async def commit(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(cm.settings, "docker_host_ip", "10.0.0.1", raising=False)
    monkeypatch.setattr("app.db.database.async_session_maker", lambda: _DB())
    return rows


def _probe(monkeypatch, per_tick):
    """`per_tick` is a list of `health_signals` dicts, one per tick, reused
    for every container."""
    state = {"i": -1}

    async def probe(container):
        return True, True, True

    async def signals(container):
        return per_tick[min(state["i"], len(per_tick) - 1)]

    def advance():
        state["i"] += 1

    monkeypatch.setattr(cm, "_probe_agent_health", probe)
    if hasattr(cm, "_probe_health_signals"):
        monkeypatch.setattr(cm, "_probe_health_signals", signals)
    return advance


def _require_gap_alerts():
    if not hasattr(cm, "_probe_health_signals"):
        pytest.skip("container_monitor does not fold health_signals yet — lane B3 (D10)")


@pytest.mark.asyncio
async def test_a_sustained_persistence_gap_alerts_as_critical(monkeypatch, alerts):
    _require_gap_alerts()
    _containers(monkeypatch)
    advance = _probe(monkeypatch, [
        {"turns_completed": 10, "assistant_rows_written": 3},
        {"turns_completed": 14, "assistant_rows_written": 3},
    ])

    advance()
    await cm.check_all_containers()
    advance()
    await cm.check_all_containers()

    gap = [a for a in alerts if a["category"] == "persist-gap"]
    assert gap, f"no persist-gap alert; got {[a['category'] for a in alerts]}"
    assert gap[0]["level"] == "critical"


@pytest.mark.asyncio
async def test_a_gap_of_one_on_a_single_tick_is_not_an_alert(monkeypatch, alerts):
    """A turn in flight at sample time is an off-by-one. Paging on it teaches
    the operator to ignore the category."""
    _require_gap_alerts()
    _containers(monkeypatch)
    advance = _probe(monkeypatch, [{"turns_completed": 10, "assistant_rows_written": 9}])

    advance()
    await cm.check_all_containers()

    assert [a for a in alerts if a["category"] == "persist-gap"] == []


@pytest.mark.asyncio
async def test_a_gap_that_closes_resets_the_streak(monkeypatch, alerts):
    """ANTI-VACUITY for the case above: the counter must be per-container and
    must actually reset, or the second incident on a tenant that once had an
    off-by-one alerts immediately."""
    _require_gap_alerts()
    _containers(monkeypatch)
    advance = _probe(monkeypatch, [
        {"turns_completed": 10, "assistant_rows_written": 9},
        {"turns_completed": 11, "assistant_rows_written": 11},
        {"turns_completed": 12, "assistant_rows_written": 11},
    ])

    for _ in range(3):
        advance()
        await cm.check_all_containers()

    assert [a for a in alerts if a["category"] == "persist-gap"] == []


@pytest.mark.asyncio
async def test_the_alert_is_keyed_per_tenant(monkeypatch, alerts):
    """L3-8, applied to the new category: `subject` carries the user prefix so
    one stuck tenant cannot suppress another's alert for the window — and the
    prefix, not the id, because this string reaches Telegram."""
    _require_gap_alerts()
    _containers(monkeypatch, n=2)
    advance = _probe(monkeypatch, [
        {"turns_completed": 10, "assistant_rows_written": 0},
        {"turns_completed": 20, "assistant_rows_written": 0},
    ])

    advance()
    await cm.check_all_containers()
    advance()
    await cm.check_all_containers()

    subjects = {a["subject"] for a in alerts if a["category"] == "persist-gap"}
    assert len(subjects) == 2, f"expected one subject per tenant, got {subjects}"
    for s in subjects:
        assert s and len(s) <= 16, f"subject {s!r} looks like a full user id"


@pytest.mark.asyncio
async def test_no_alert_carries_a_message_body_or_a_phone_number(monkeypatch, alerts):
    """The alert goes to a Telegram chat. Counts and a user prefix; never a
    transcript fragment and never an e164."""
    _require_gap_alerts()
    _containers(monkeypatch)
    advance = _probe(monkeypatch, [
        {"turns_completed": 10, "assistant_rows_written": 0,
         "channel_tag_prefixed_replies": 2},
        {"turns_completed": 20, "assistant_rows_written": 0,
         "channel_tag_prefixed_replies": 5},
    ])

    advance()
    await cm.check_all_containers()
    advance()
    await cm.check_all_containers()

    import re
    for a in alerts:
        assert not re.search(r"\+\d{7,}", a["message"]), a["message"]


# ── the gate above all of it ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_loop_does_not_run_without_the_lease(monkeypatch):
    """It restarts containers and writes `status='error'`. It ran unelected on
    both Railway replicas until 2026-09-12, and the new alerting only makes a
    double-run louder."""
    called = []

    async def never():
        called.append(1)

    async def no_lease(name, ttl_s=None):
        return False

    monkeypatch.setattr(cm, "check_all_containers", never)
    monkeypatch.setattr("app.services.infra_lease.acquire_lease", no_lease)

    import asyncio

    async def _sleep(_):
        raise asyncio.CancelledError

    monkeypatch.setattr(cm.asyncio, "sleep", _sleep)
    with pytest.raises(asyncio.CancelledError):
        await cm.monitor_loop()

    assert called == [], "check_all_containers ran on a replica with no lease"
