"""Process-lifetime counters, and a reader that tolerates an image without them (D10).

The 2026-09-14 incident was silent on every dashboard: turns completed, rows
were written into a day chat nobody could reach, a media card was dropped on
every reload, and `/api/day-chats` 500ed 65 times over three hours. Nothing
counted any of it.

D10 adds `app/services/health_signals.py` — ints only, no ids and no text,
exposed under `/agent/health.health_signals`. Two properties decide whether it
is safe to ship:

  * it may NEVER make a tenant look unhealthy. The counters are diagnostics;
    a raise inside them must not change the health verdict, or a bug in the
    instrumentation restarts containers.
  * the READER must tolerate a body with no `health_signals` key. The fleet
    runs image 7edaed3ab644 and a rollout is gradual, so for the length of
    that rollout the monitor reads old bodies and new ones in the same tick.

Local run (from backend/):
    RUN_MODE=agent PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_agent_health_signals.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import pytest


def hs():
    return pytest.importorskip(
        "app.services.health_signals",
        reason="health_signals not landed yet — lane B3 (D10)",
    )


# D10's list, verbatim. Named rather than discovered: a counter that is
# incremented but never declared is one nobody will think to graph.
EXPECTED = [
    "future_dated_day_chats", "day_chats_5xx", "turns_completed",
    "assistant_rows_written", "media_expected", "media_persisted",
    "channel_events_claimed", "channel_events_deduped", "channel_events_persisted",
    "wa_sidecar_spawns", "channel_tag_prefixed_replies",
    "rebucket_runs", "rebucket_failures",
    # Review round 1: a NULL-day message is the incident's own shape, and the
    # gauge needed a monotonic partner so a heal cannot erase the evidence.
    "day_chat_resolve_failures", "future_dated_day_chats_seen",
    "wa_sidecar_adopted",
    # Round 44: the cached WhatsApp status was advanced by the reconciler
    # rather than by SSE — i.e. a `connection_open` reached nobody.
    "wa_status_reconciled",
]


@pytest.fixture(autouse=True)
def _reset():
    """Counters are process-lifetime by design, so they leak between tests.

    Deliberately NOT an importorskip: this fixture runs for every test in the
    file, including the reader cases that exercise container_monitor and can
    run today. Skipping here would hide them behind a module that has not
    landed.
    """
    try:
        from app.services import health_signals as m
    except Exception:
        yield
        return
    m.reset_for_tests()
    yield
    m.reset_for_tests()


def test_the_snapshot_declares_every_documented_counter():
    snap = hs().snapshot()
    missing = [k for k in EXPECTED if k not in snap]
    assert not missing, f"health_signals does not declare {missing}"


def test_every_value_is_an_int_and_nothing_carries_text():
    """The body is served publicly at /agent/health with no auth. A user id,
    a phone number or a message fragment in here is a data leak on a route
    that exists to be polled."""
    snap = hs().snapshot()
    for k, v in snap.items():
        assert isinstance(v, int) and not isinstance(v, bool), f"{k}={v!r} is not an int"


def test_incr_and_get_round_trip():
    m = hs()
    m.incr("turns_completed")
    m.incr("turns_completed", 4)
    assert m.get("turns_completed") == 5
    assert m.snapshot()["turns_completed"] == 5


def test_an_unknown_counter_never_raises():
    """Call sites are `health_signals.incr(...)` sprinkled through the hot
    path and imported lazily; a typo must cost a lost number, never a turn."""
    m = hs()
    m.incr("a_counter_nobody_declared")
    assert m.get("a_counter_that_does_not_exist") == 0


def test_reset_for_tests_clears_everything():
    m = hs()
    m.incr("media_expected", 3)
    m.reset_for_tests()
    assert m.get("media_expected") == 0


# ── the endpoint ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_health_exposes_the_signals():
    import agent_main

    hs()
    body = await agent_main.agent_health()
    if "health_signals" not in body:
        pytest.skip("/agent/health does not expose health_signals yet — lane B2 (D10)")
    assert all(isinstance(v, int) for v in body["health_signals"].values())


@pytest.mark.asyncio
async def test_agent_health_still_answers_when_the_counters_raise(monkeypatch):
    """Best-effort, in the strict sense: an exception inside the diagnostics
    must not make a healthy tenant report unhealthy — that would have the
    monitor restarting containers because a counter module broke."""
    import agent_main

    m = hs()

    def boom():
        raise RuntimeError("counters exploded")

    monkeypatch.setattr(m, "snapshot", boom)
    body = await agent_main.agent_health()
    assert body.get("status") in ("healthy", "ok"), (
        "a failing counter module changed the health verdict"
    )


# ── the reader ────────────────────────────────────────────────────────


def test_verdict_from_health_body_tolerates_an_old_image():
    """The fleet runs 7edaed3ab644 and a rollout is gradual: the monitor
    reads bodies with and without this key in the same tick."""
    from app.services.container_monitor import verdict_from_health_body

    assert verdict_from_health_body({"status": "healthy"}) == (True, None, None)


def test_the_signals_reader_answers_none_for_a_body_without_them():
    from app.services import container_monitor as cm

    reader = getattr(cm, "signals_from_health_body", None)
    if reader is None:
        pytest.skip("signals_from_health_body not landed yet — lane B3 (D10)")
    assert reader({"status": "healthy"}) is None


def test_the_signals_reader_parses_a_new_image_body():
    from app.services import container_monitor as cm

    reader = getattr(cm, "signals_from_health_body", None)
    if reader is None:
        pytest.skip("signals_from_health_body not landed yet — lane B3 (D10)")
    got = reader({"status": "healthy", "health_signals": {"turns_completed": 7}})
    assert got == {"turns_completed": 7}


@pytest.mark.parametrize("body", [
    {"status": "healthy", "health_signals": None},
    {"status": "healthy", "health_signals": "not-a-dict"},
    {"status": "healthy", "health_signals": []},
    {},
])
def test_the_signals_reader_never_raises_on_a_malformed_body(body):
    """A tenant answering nonsense is a tenant with a problem, not a reason
    for the MONITOR to stop checking everybody else in the loop."""
    from app.services import container_monitor as cm

    reader = getattr(cm, "signals_from_health_body", None)
    if reader is None:
        pytest.skip("signals_from_health_body not landed yet — lane B3 (D10)")
    assert reader(body) is None or isinstance(reader(body), dict)
