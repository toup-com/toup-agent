"""R14.3 RELAY-START — what the caller waits through, and what their first
words are allowed to interrupt.

Measured in production on d3ebda17, every founder session (n=5): WS accept →
`ready` min 1.99 s / median 4.43 s / max 7.53 s.  On the 15:06:23Z call the
breakdown was session lookup +1.0 s → voice-context +2.0 s → history seed
+3.5 s → provider started +4.4 s, and on the attempt ten seconds earlier the
seed timed out and the client had hung up 14 ms before `ready`.  Two defects
sit in that number and one sits right after it:

* the reconnect seed was AWAITED between the config wait and `session.start`,
  so every caller paid a tenant round trip they did not need;
* the tz write sat in front of the provider handshake rather than beside it;
* the audio a caller speaks while the client still says "connecting" queues on
  the socket and is read in one burst the moment `client_loop` starts — and
  the relay read those words as the caller talking over the agent's opening
  reply (`played_ms=300`).

Every latency number here is a HARNESS number: fake provider, fake client,
sleeps standing in for tenant round trips.  It measures the relay's own
sequencing, never production.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect

from app.config import settings
from tests.test_live_harness import (
    FakeClient,
    FakeProvider,
    config,
    fast_clocks,
    out_audio,
    out_text,
    patch_relay,
    run_relay,
    user_delta,
)

import app.services.live_voice_protocol as live


SRC = Path(live.__file__)
WS_SRC = Path(live.__file__).parent.parent / "api" / "ws_realtime.py"


def audio(data="AAAA"):
    return {"type": "audio", "data": data}


def push(provider, event):
    """A script step that feeds the provider without consuming a client read."""

    return lambda: provider.push(event)


def close_on_close(provider, event):
    if event.get("type") == "session.close":
        provider.push({"type": "session.closed", "reason": "close_requested",
                       "usage": {"seconds": 1}})


def startup_clocks(monkeypatch, **overrides):
    """`fast_clocks`, plus the R14 knobs shrunk to test scale."""

    defaults = {
        "voice_live_preready_gap_ms": 20,
        "voice_live_preready_window_ms": 500,
        "voice_live_first_turn_grace_ms": 250,
        "voice_live_history_budget_ms": 400,
        "voice_live_bargein_min_ms": 600,
        "voice_live_bargein_min_tokens": 3,
    }
    defaults.update(overrides)
    fast_clocks(monkeypatch, **defaults)


def ready_ms(client, t0) -> float:
    assert "ready" in client.stamps, "the relay never became ready"
    return (client.stamps["ready"] - t0) * 1000.0


@pytest.mark.asyncio
async def test_preready_client_close_cancels_silent_provider_handshake(monkeypatch):
    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    provider_cancelled = asyncio.Event()

    class SilentProvider(FakeProvider):
        async def recv(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise

    def disconnected():
        raise WebSocketDisconnect(code=1000)

    client = FakeClient([config(), 0.02, disconnected])
    provider = SilentProvider()
    started = time.monotonic()
    await run_relay(client, provider, timeout=1.0)
    assert time.monotonic() - started < 0.5
    assert not client.of("ready")
    assert provider_cancelled.is_set()


@pytest.mark.asyncio
async def test_preready_provider_silence_uses_overall_deadline(monkeypatch):
    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    provider_cancelled = asyncio.Event()

    class SilentProvider(FakeProvider):
        async def recv(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise

    client = FakeClient([config()])
    provider = SilentProvider()
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await run_relay(
            client, provider, timeout=1.0,
            startup_deadline_monotonic=started + 0.08,
        )
    assert time.monotonic() - started < 0.5
    assert not client.of("ready")
    assert provider_cancelled.is_set()
    await asyncio.sleep(0)
    assert not [task for task in asyncio.all_tasks()
                if task is not asyncio.current_task() and not task.done()
                and task.get_name() in {"live-history-seed", "live-tz"}]


@pytest.mark.asyncio
async def test_preready_client_receive_keeps_its_pending_frame_after_ready(monkeypatch):
    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    client = FakeClient([config(), 0.30, {"type": "stop"}])
    provider = FakeProvider(on_send=close_on_close, start_delay=0.20)
    started = time.monotonic()
    await run_relay(client, provider, timeout=2.0)
    assert client.of("ready")
    assert client.reads == 3
    assert time.monotonic() - started >= 0.30


@pytest.mark.asyncio
async def test_preready_audio_burst_is_buffered_through_provider_handshake(monkeypatch):
    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    client = FakeClient([config(), 0.16, *([audio("QUJD")] * 80),
                         0.20, {"type": "stop"}])
    provider = FakeProvider(on_send=close_on_close, start_delay=0.25)
    await run_relay(client, provider, timeout=2.0)
    assert client.of("ready")
    assert len(provider.of("session.input_audio.append")) == 80


@pytest.mark.asyncio
async def test_post_ready_work_can_outlive_startup_deadline(monkeypatch):
    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    entered_post_ready = asyncio.Event()
    release_post_ready = asyncio.Event()
    original_send_state = live._LiveSession.send_state

    async def delayed_send_state(self, state, *args, **kwargs):
        if state == "listening" and self.ready_at and not entered_post_ready.is_set():
            entered_post_ready.set()
            await release_post_ready.wait()
        return await original_send_state(self, state, *args, **kwargs)

    monkeypatch.setattr(live._LiveSession, "send_state", delayed_send_state)
    client = FakeClient([config(), {"type": "stop"}])
    provider = FakeProvider(on_send=close_on_close)
    deadline = time.monotonic() + 0.5
    relay_task = asyncio.create_task(run_relay(
        client, provider, timeout=2.0,
        startup_deadline_monotonic=deadline,
    ))
    try:
        await asyncio.wait_for(entered_post_ready.wait(), timeout=0.4)
        await asyncio.sleep(max(0.0, deadline - time.monotonic() + 0.05))
        assert client.of("ready")
        assert not relay_task.done(), "post-ready work must survive the startup deadline"
        release_post_ready.set()
        await relay_task
    finally:
        release_post_ready.set()
        if not relay_task.done():
            relay_task.cancel()
        await asyncio.gather(relay_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_preready_timeout_stops_both_persistence_workers(monkeypatch):
    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    created = []
    original_queue = live.PersistenceQueue

    class TrackedQueue(original_queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    async def stalled_day_date(self):
        # This point is after both workers start and before session_id/ready.
        assert len(created) == 2
        assert all(queue._worker is not None for queue in created)
        self.startup_timeout.reschedule(time.monotonic() + 0.05)
        await asyncio.Event().wait()

    monkeypatch.setattr(live, "PersistenceQueue", TrackedQueue)
    monkeypatch.setattr(live._LiveSession, "day_date", stalled_day_date)
    client = FakeClient([config()])
    provider = FakeProvider()
    with pytest.raises(TimeoutError):
        await run_relay(
            client, provider, timeout=1.0,
            startup_deadline_monotonic=time.monotonic() + 2.0,
        )
    assert not client.of("ready")
    assert len(created) == 2
    assert all(queue._stopped and queue._worker is None for queue in created)
    assert not [task for task in asyncio.all_tasks()
                if not task.done() and task is not asyncio.current_task()
                and task.get_name() in {"live-persist-rows", "live-persist-curate"}]


async def _no_stored_zone(_user_id):
    """Stand-in for `ws_realtime._get_user_tz_name` in the tests that time
    `ready`: the account has no stored zone, answered without a DB round
    trip."""

    return None


# ── (a) readiness never waits on the history seed ───────────────────────


@pytest.mark.asyncio
async def test_a_history_fetch_that_never_returns_does_not_delay_ready(monkeypatch):
    """The seed is a TENANT read. A tenant that never answers is a cold model,
    never a caller staring at "Connecting to the voice engine…"."""

    startup_clocks(monkeypatch, voice_live_history_rows=20)
    patch_relay(monkeypatch)

    async def never(_user_id):
        await asyncio.sleep(30)
        return ("http://agent", "key")

    monkeypatch.setattr(live, "_vcount", lambda *a, **k: None)
    from app.api import ws_realtime as rt
    monkeypatch.setattr(rt, "_get_vps_info", never)
    # `day_date` reads the STORED zone (R14V-1) before `ready`, bounded at
    # 0.5 s. Stubbed so this stays a measurement of the relay's own sequencing
    # and not of whatever the local DB does: CI run 35792648976 failed this
    # test at 482.5 ms on a contended runner with the seed nowhere near the
    # critical path, and a 450 ms delay on that one read reproduces the number.
    # The real read stays covered by
    # test_day_date_is_the_account_zone_not_the_handsets.
    monkeypatch.setattr(rt, "_get_user_tz_name", _no_stored_zone)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([config(), 0.05, {"type": "stop"}])
    t0 = time.monotonic()
    await run_relay(client, provider, timeout=5.0)

    assert ready_ms(client, t0) < 400, ready_ms(client, t0)
    # And it never rode into `session.start` — there was nothing to ride.
    start = provider.of("session.start")[0]
    assert "input" not in start["session"]


@pytest.mark.asyncio
async def test_a_seed_that_beat_the_handshake_rides_into_session_start(monkeypatch):
    """Collected, not discarded: when the tenant IS fast the provider's own
    history slot is still the better carrier than a context append."""

    startup_clocks(monkeypatch, voice_live_history_rows=20)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    async def vps(_user_id):
        return ("http://agent", "key")

    async def rows(*_args, **_kwargs):
        return [
            {"role": "user", "content": "where did we leave the invoice"},
            {"role": "assistant", "content": "with the accountant"},
        ]

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", rows)

    # The seed is instant and the config frame takes a round trip to arrive —
    # the 1.5 s window the config wait already spends idle is exactly what
    # starting the seed on line one buys.
    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([0.05, config(), 0.05, {"type": "stop"}])
    await run_relay(client, provider, timeout=5.0)

    items = provider.of("session.start")[0]["session"]["input"]
    assert [i["role"] for i in items] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_a_late_seed_still_reaches_the_provider(monkeypatch):
    """It lost the race to `session.start`, whose fields are immutable — so it
    arrives as a context append instead. The facts are not thrown away."""

    startup_clocks(monkeypatch, voice_live_history_rows=20)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    seed_answered: list[float] = []

    async def slow_vps(_user_id):
        await asyncio.sleep(0.08)
        seed_answered.append(time.monotonic())
        return ("http://agent", "key")

    async def rows(*_args, **_kwargs):
        return [{"role": "user", "content": "the blue folder"}]

    monkeypatch.setattr(rt, "_get_vps_info", slow_vps)
    monkeypatch.setattr(rt, "_vps_api", rows)
    # Same reason as the never-returning seed above: not a DB measurement.
    monkeypatch.setattr(rt, "_get_user_tz_name", _no_stored_zone)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([config(), 0.3, {"type": "stop"}])
    await run_relay(client, provider, timeout=5.0)

    # An ORDER, not a wall-clock limit: `ready` went out before the tenant
    # had even answered the seed's first leg. That is the property ("ready
    # never waits on the seed"), and unlike `< 70 ms` it holds on a runner
    # where everything is uniformly slow. A relay that waited on the seed —
    # or on a history budget — sends `ready` after this stamp and fails here.
    assert "ready" in client.stamps, "the relay never became ready"
    assert seed_answered, "the seed never reached the tenant"
    assert client.stamps["ready"] < seed_answered[0], (
        "ready went out AFTER the history seed's tenant lookup answered — "
        "readiness waited on the seed",
        (client.stamps["ready"] - seed_answered[0]) * 1000.0,
    )
    assert "input" not in provider.of("session.start")[0]["session"]
    seeded = [
        e for e in provider.of("session.thinking.append")
        if "the blue folder" in e["content"]
    ]
    assert seeded, [e["content"][:40] for e in provider.of("session.thinking.append")]


@pytest.mark.asyncio
async def test_a_seed_that_touches_the_tenant_at_all_takes_the_late_carrier(monkeypatch):
    """R14V-2. The native `history=` slot is a fast path PRODUCTION DOES NOT
    TAKE, and the contract the feature owes is the weaker one.

    `take_ready_history` gives the seed one event-loop tick.  A real seed opens
    a DB session and makes an HTTP request, each of which yields many times —
    so it is suspended when the collection happens, and the sibling test above
    (`…rides_into_session_start`) is green only because its fakes return
    without ever yielding.  Model this with the FASTEST possible real tenant:
    zero delay, one yield.  It still misses `session.start`, and it still
    reaches the provider before the caller's second turn, which is what A8-4
    actually needs."""

    startup_clocks(monkeypatch, voice_live_history_rows=20)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    async def vps(_user_id):
        await asyncio.sleep(0)  # one yield: the cheapest real tenant read
        return ("http://agent", "key")

    async def rows(*_args, **_kwargs):
        return [{"role": "user", "content": "the blue folder"}]

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", rows)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        0.02,
        {"type": "audio", "data": "QUJD"},   # the caller's first turn
        0.05,
        {"type": "audio", "data": "REVG"},   # …and their second
        0.05,
        {"type": "stop"},
    ])
    await run_relay(client, provider, timeout=5.0)

    assert "input" not in provider.of("session.start")[0]["session"], (
        "a seed that yielded once still made the native slot — then the tick "
        "above is buying reachability, and this test is measuring the fakes"
    )
    kinds = [e.get("type") for e in provider.sent]
    seeded = [
        i for i, e in enumerate(provider.sent)
        if e.get("type") == "session.thinking.append"
        and "the blue folder" in str(e.get("content") or "")
    ]
    assert seeded, kinds
    second_turn = [i for i, k in enumerate(kinds) if k == "session.input_audio.append"]
    assert len(second_turn) >= 2, kinds
    assert seeded[0] < second_turn[1], (
        f"the seed reached the provider at {seeded[0]} but the caller's second "
        f"turn was already at {second_turn[1]} — the model answered it cold"
    )


@pytest.mark.asyncio
async def test_a_seed_past_its_budget_is_dropped_not_left_pending(monkeypatch, caplog):
    """A tenant that never answers must not leave a task poking it for the
    length of the call."""

    startup_clocks(
        monkeypatch, voice_live_history_rows=20, voice_live_history_budget_ms=120,
    )
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    async def never(_user_id):
        await asyncio.sleep(30)
        return None

    monkeypatch.setattr(rt, "_get_vps_info", never)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([config(), 0.35, {"type": "stop"}])
    with caplog.at_level(logging.INFO, logger="app.services.live_voice_protocol"):
        await run_relay(client, provider, timeout=5.0)

    assert any("history seed dropped reason=budget" in r.getMessage()
               for r in caplog.records)
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "live-history-seed"]


@pytest.mark.asyncio
async def test_a_failed_handshake_abandons_the_tz_write_too(monkeypatch):
    """R14V-3. The path where a startup task is genuinely still in flight is a
    provider handshake that FAILS — the config frame has been read, the tz
    write is running, and the rejection propagates to `run()`, which has none
    of `start_provider`'s locals to hand its cleanup.

    So both tasks hang off the session.  Before this, `abandon_startup` was
    passed a `tz_task` that was always None (its only branch sits above the
    assignment) and the live one leaked."""

    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    async def slow_tz(_user_id, _raw):
        await asyncio.sleep(30)
        return "Europe/London"

    monkeypatch.setattr(rt, "_apply_client_tz", slow_tz)

    class RejectingProvider(FakeProvider):
        """`FakeProvider.recv` always answers `session.started`, so a startup
        that FAILS has to override it — `push` feeds the provider LOOP, which
        the handshake never reaches."""

        async def recv(self):
            await asyncio.sleep(0)
            return json.dumps({"type": "error", "error": {"code": "model_unavailable"}})

    provider = RejectingProvider()
    client = FakeClient([config(tz="Europe/London"), 0.05, {"type": "stop"}])
    with pytest.raises(RuntimeError):
        await run_relay(client, provider, timeout=5.0)

    # The cancel needs a tick to land; 30 s of sleep does not.
    await asyncio.sleep(0.05)
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "live-tz"], (
        "the tz write outlived the call it belonged to — it is harmless today "
        "only because nothing reads it, and the next tenant read added beside "
        "it inherits the leak"
    )


# ── (b) independent tenant reads overlap rather than add ────────────────


@pytest.mark.asyncio
async def test_slow_tenant_reads_overlap_the_handshake_rather_than_adding(monkeypatch):
    """HARNESS number. Three independent legs of 120 ms each — the seed, the tz
    write, the provider handshake. Serialized they are 360 ms; the caller
    should wait for the longest, not for the sum.

    Each leg opens its OWN DB session: a gather over one AsyncSession kills
    every leg but the first, which is the reason this is three tasks and not
    one `asyncio.gather` over a shared session.
    """

    leg = 0.12
    startup_clocks(monkeypatch, voice_live_history_rows=20)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    async def slow_vps(_user_id):
        await asyncio.sleep(leg)
        return None

    async def slow_tz(_user_id, _raw):
        await asyncio.sleep(leg)
        return "Europe/London"

    async def tz_read(_user_id):
        return "Europe/London"

    monkeypatch.setattr(rt, "_get_vps_info", slow_vps)
    monkeypatch.setattr(rt, "_apply_client_tz", slow_tz)
    # `day_date` reads the STORED zone (R14V-1). Stubbed so this stays a
    # measurement of the three legs and not of whatever the local DB does.
    monkeypatch.setattr(rt, "_get_user_tz_name", tz_read)

    provider = FakeProvider(on_send=close_on_close, start_delay=leg)
    client = FakeClient([config(tz="Europe/London"), 0.05, {"type": "stop"}])
    t0 = time.monotonic()
    await run_relay(client, provider, timeout=5.0)
    elapsed = ready_ms(client, t0)

    # Not a tautology in either direction: the handshake leg is genuinely on
    # the critical path (so the floor holds), and the other two are not (so the
    # ceiling would fail the moment either is awaited in front of it).
    assert elapsed >= leg * 1000 * 0.9, elapsed
    assert elapsed < leg * 1000 * 2, elapsed
    # The zone the config frame carried was USED — overlapping it must not cost
    # the thing it was read for.
    assert any(
        "Europe/London" in e["content"]
        for e in provider.of("session.thinking.append")
    )


@pytest.mark.asyncio
async def test_day_date_is_the_account_zone_not_the_handsets(monkeypatch):
    """R14V-1. `_apply_client_tz` returns the string the CONFIG FRAME carried,
    and the write underneath it is `WHERE users.timezone IS NULL` on purpose —
    so that return equals the account's zone only when the column was blank.

    Reading it back is the difference between the relay and every other
    `day_date` producer agreeing on the caller's day and disagreeing about it.
    The two zones here are 25 h apart, so their local dates can never coincide:
    if the handset's zone were used, this test could not pass by luck."""

    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    async def tz(_user_id, raw):
        # The real one: validate, attempt the IS NULL write, return the raw
        # string. The account row here already HAS a zone, so the write is a
        # no-op and this return is the device's, not the account's.
        return raw

    async def tz_read(_user_id):
        return "Pacific/Niue"

    monkeypatch.setattr(rt, "_apply_client_tz", tz)
    monkeypatch.setattr(rt, "_get_user_tz_name", tz_read)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([config(tz="Pacific/Kiritimati"), 0.05, {"type": "stop"}])
    await run_relay(client, provider, timeout=5.0)

    day = client.of("session_id")[0].get("day_date")
    assert day == rt._local_today_str("Pacific/Niue"), day
    assert day != rt._local_today_str("Pacific/Kiritimati"), (
        "the relay keyed the caller's day to the HANDSET's zone — the client "
        "caches the session id per local day, so this call is filed under a "
        "day nothing else in the relay agrees with"
    )


@pytest.mark.asyncio
async def test_day_date_falls_back_to_the_client_zone_when_the_read_fails(monkeypatch):
    """The read is the source of truth, not a hard requirement: a validated
    device zone is a better answer than no `day_date` at all, and it is what
    the account row would hold anyway whenever the column was blank."""

    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    async def tz(_user_id, raw):
        return raw

    async def tz_read(_user_id):
        raise RuntimeError("tenant unreachable")

    monkeypatch.setattr(rt, "_apply_client_tz", tz)
    monkeypatch.setattr(rt, "_get_user_tz_name", tz_read)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([config(tz="Pacific/Kiritimati"), 0.05, {"type": "stop"}])
    await run_relay(client, provider, timeout=5.0)

    day = client.of("session_id")[0].get("day_date")
    assert day == rt._local_today_str("Pacific/Kiritimati"), day


@pytest.mark.asyncio
async def test_day_date_still_reads_the_db_when_no_zone_was_sent(monkeypatch):
    """The read is not deleted — it is the fallback for a client that sent no
    zone at all, which is every web session."""

    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    reads: list[str] = []

    async def tz_read(user_id):
        reads.append(user_id)
        return "Asia/Tehran"

    monkeypatch.setattr(rt, "_get_user_tz_name", tz_read)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([config(), 0.05, {"type": "stop"}])
    await run_relay(client, provider, timeout=5.0)

    assert reads == ["user-1"]
    assert client.of("session_id")[0].get("day_date")


def test_the_voice_context_fan_out_is_still_a_fan_out():
    """SOURCE probe, ORDER-sensitive. The session lookup and the voice-context
    build are independent tenant reads and are started as tasks BEFORE either
    is awaited. Awaiting one in front of the other would add its full duration
    to every caller's wait, and no behavioural test in this file crosses the
    `ws_realtime` fan-out to notice."""

    src = WS_SRC.read_text()
    spawn_session = src.index("_session_t = asyncio.create_task(_session_step())")
    spawn_instr = src.index("_instructions_t = asyncio.create_task(_instructions_step())")
    joined = src.index("db_session_id, live_instructions = await asyncio.wait_for(")
    assert spawn_session < joined
    assert spawn_instr < joined
    assert "asyncio.gather(_session_t, _instructions_t)" in src[joined:joined + 180]


def test_the_history_seed_starts_before_the_config_wait():
    """SOURCE probe, ORDER-sensitive. Starting the seed AFTER the config wait
    would still overlap the handshake, but it would forfeit the 1.5 s window
    the config wait already spends idle — and a seed spawned after
    `build_session_start` could never ride in natively at all."""

    src = SRC.read_text()
    body = src[src.index("    async def start_provider(self)"):]
    body = body[:body.index("\n    async def tick_loop")]
    spawn = body.index('self.fetch_history(), name="live-history-seed"')
    config_wait = body.index("self.ws.receive_text(), timeout=1.5")
    handshake = body.index("await self.provider_send(build_session_start(")
    assert spawn < config_wait < handshake
    # And it is never awaited inline anywhere in the startup path.
    assert "await self.fetch_history()" not in body


# ── (c) the accept→ready line ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_accept_to_ready_line_is_present_with_numbers_and_no_text(
    monkeypatch, caplog,
):
    """`elapsed_ms` starts after the slow part, so on its own it understated
    startup by 1.5-6.7 s. The line that replaces it carries the caller's whole
    wait and its breakdown — and, like every line this relay writes, not one
    word of what was said."""

    secret = "the invoice from the accountant"
    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        push(provider, out_text("I can help with that.", 0, 800)),
        push(provider, user_delta(secret, 0, 900)),
        0.1,
        {"type": "stop"},
    ])
    with caplog.at_level(logging.INFO, logger="app.services.live_voice_protocol"):
        await run_relay(client, provider, timeout=5.0)

    lines = [r.getMessage() for r in caplog.records if "accept_to_ready_ms=" in r.getMessage()]
    assert len(lines) == 1, lines
    line = lines[0]
    for key in ("config_ms=", "provider_ms=", "day_date_ms=", "history_rows="):
        assert key in line, line
    assert re.search(r"accept_to_ready_ms=\d+", line), line
    # Numbers only. Every value on the line is an integer or the session stem.
    for field in line.split("[LIVE] ")[1].split():
        name, _, value = field.partition("=")
        if name == "session":
            continue
        assert value.isdigit(), field
    for record in caplog.records:
        assert secret not in record.getMessage()
        assert "I can help with that." not in record.getMessage()


# ── (d) the caller's first utterance ────────────────────────────────────


@pytest.mark.asyncio
async def test_pre_ready_speech_is_the_first_turn_not_a_barge_in(monkeypatch):
    """The 15:06:28Z shape. The caller speaks while the client still says
    "connecting"; that audio queues on the socket and is read in one burst the
    moment `client_loop` starts, which is when the agent's opening reply
    begins. The provider transcribes it and — before this — the relay called
    those words an interruption and cut the reply at 300 ms.

    They came FIRST. A reply that began before that speech was processed is
    not something it interrupted."""

    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        # Spoken during "Connecting to the voice engine…", delivered now.
        audio("QUJD"),
        audio("REVG"),
        # The opening reply, which began before any of that was processed.
        push(provider, out_text("Hey — good to hear you.", 0, 1100)),
        push(provider, out_audio()),
        # …and the transcript of the burst, arriving over that reply.
        push(provider, user_delta("چه کار", 0, 700)),
        push(provider, user_delta(" میکنید", 700, 1400)),
        push(provider, user_delta(" امروز", 1400, 2100)),
        0.12,
        {"type": "stop"},
    ])
    await run_relay(client, provider, timeout=5.0)

    assert client.of("speech_started") == []
    # DELIVERED AND ANSWERED: the fence withholds barge-in evidence, never the
    # audio and never the words.
    assert [e["audio"] for e in provider.of("session.input_audio.append")] == [
        "QUJD", "REVG",
    ]


@pytest.mark.asyncio
async def test_the_same_words_after_ready_do_interrupt(monkeypatch):
    """The control. Nothing about those words was special — only when the
    audio behind them arrived. With no pre-`ready` backlog the fence never
    arms and the identical deltas cut the reply, as they must."""

    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        # A real gap before the first client frame: this caller said nothing
        # while connecting.
        0.05,
        audio("QUJD"),
        push(provider, out_text("Hey — good to hear you.", 0, 1100)),
        push(provider, out_audio()),
        push(provider, user_delta("چه کار", 0, 700)),
        push(provider, user_delta(" میکنید", 700, 1400)),
        push(provider, user_delta(" امروز", 1400, 2100)),
        0.12,
        {"type": "stop"},
    ])
    await run_relay(client, provider, timeout=5.0)

    assert len(client.of("speech_started")) == 1


@pytest.mark.asyncio
async def test_the_fence_expires_and_a_real_interruption_still_fires(monkeypatch):
    """The fence is bounded. Once the backlog has drained and the provider's
    transcription lag has passed, the caller is talking over the agent in the
    ordinary way and barge-in is theirs again."""

    startup_clocks(monkeypatch, voice_live_first_turn_grace_ms=60)
    patch_relay(monkeypatch)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        audio("QUJD"),
        # A real-time gap ends the drain; the grace then expires.
        0.15,
        {"type": "audio_ready"},
        0.12,
        push(provider, out_text("Here is the long answer you asked for.", 0, 2000)),
        push(provider, out_audio()),
        push(provider, user_delta("no stop", 3000, 3600)),
        push(provider, user_delta(" that is wrong", 3600, 4300)),
        0.12,
        {"type": "stop"},
    ])
    await run_relay(client, provider, timeout=5.0)

    assert len(client.of("speech_started")) == 1


@pytest.mark.asyncio
async def test_a_caller_who_speaks_once_and_goes_quiet_cannot_wedge_the_fence(
    monkeypatch,
):
    """No later frame ever ends the drain for this caller, so the fence has a
    second bound — from `ready` itself. Without it one pre-`ready` chunk would
    disable barge-in for the rest of the call."""

    startup_clocks(
        monkeypatch,
        voice_live_preready_window_ms=80,
        voice_live_first_turn_grace_ms=60,
    )
    patch_relay(monkeypatch)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        audio("QUJD"),
        # Nothing further from the client at all until the very end.
        0.3,
        push(provider, out_text("Here is the long answer you asked for.", 0, 2000)),
        push(provider, out_audio()),
        push(provider, user_delta("no stop", 3000, 3600)),
        push(provider, user_delta(" that is wrong", 3600, 4300)),
        0.12,
        {"type": "stop"},
    ])
    await run_relay(client, provider, timeout=5.0)

    assert len(client.of("speech_started")) == 1


@pytest.mark.asyncio
async def test_the_fence_does_not_arm_without_pre_ready_audio(monkeypatch):
    """Frames that are not audio arrive in the same burst (`audio_ready`,
    `config`). Arming on those would cost every call its first seconds of
    barge-in for nothing."""

    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        {"type": "audio_ready"},
        {"type": "audio_ready"},
        push(provider, out_text("Hey — good to hear you.", 0, 1100)),
        push(provider, out_audio()),
        push(provider, user_delta("چه کار", 0, 700)),
        push(provider, user_delta(" میکنید", 700, 1400)),
        push(provider, user_delta(" امروز", 1400, 2100)),
        0.12,
        {"type": "stop"},
    ])
    await run_relay(client, provider, timeout=5.0)

    assert len(client.of("speech_started")) == 1


@pytest.mark.asyncio
async def test_the_shielded_first_turn_is_still_a_user_turn(monkeypatch):
    """R13.3.4 again, from the other side: withholding barge-in evidence must
    never drop the caller's words. Whatever the fence decides, the utterance
    assembler still sees the whole turn and the row is still written."""

    startup_clocks(monkeypatch)
    calls = patch_relay(monkeypatch)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        audio("QUJD"),
        push(provider, out_text("Hey — good to hear you.", 0, 1100)),
        push(provider, out_audio()),
        push(provider, user_delta("where is", 0, 700)),
        push(provider, user_delta(" the invoice", 700, 1400)),
        0.25,
        {"type": "stop"},
    ])
    await run_relay(client, provider, timeout=5.0)

    said = " ".join(
        str(row.get("user_text") or "") for row in calls["saves"]
    )
    assert "where is the invoice" in said, calls["saves"]


@pytest.mark.asyncio
async def test_a_tap_inside_the_fence_still_moves_the_interrupt_floor(monkeypatch):
    """R14V-4. The fence withholds BARGE-IN evidence and nothing else.

    A caller who TAPS to interrupt inside the fence window is interrupting —
    explicitly, with a button.  Every further delta of theirs has to push the
    interrupt's resume floor forward, or provider output generated while they
    were still talking is accepted and the agent resumes over their tail.  The
    fence used to skip `note_user_input` wholesale, and that bookkeeping is its
    first statement."""

    startup_clocks(
        monkeypatch,
        voice_live_first_turn_grace_ms=2000,
        voice_live_preready_window_ms=2000,
        voice_live_interrupt_suppress_ms=2000,
    )
    patch_relay(monkeypatch)

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([
        config(),
        audio("QUJD"),                                    # arms the fence
        push(provider, out_text("Hey — good to hear you.", 0, 1100)),
        push(provider, out_audio()),
        push(provider, user_delta("no", 0, 500)),         # first words, fenced
        0.05,
        {"type": "interrupt", "played_ms": 300},          # the caller TAPS
        push(provider, user_delta(" stop please", 500, 1800)),  # still talking
        0.05,
        # Output the provider had already generated while they were talking.
        push(provider, out_text(" as I was saying", 900, 1400)),
        0.05,
        {"type": "stop"},
    ])
    await run_relay(client, provider, timeout=5.0)

    spoken = [str(f.get("text") or "") for f in client.of("response_text")]
    assert any("good to hear you" in t for t in spoken), spoken
    assert not any("as I was saying" in t for t in spoken), (
        "queued output from before the caller finished was let through — the "
        "resume floor froze at the tap instead of following their voice"
    )


# ── (d)/R12.4: the reconnect note the startup path owns ─────────────────


@pytest.mark.asyncio
async def test_the_r12_4_detached_result_note_still_arrives(monkeypatch):
    """A task the caller started on an EARLIER socket and hung up on. Its
    answer is in the thread and nothing ever said it out loud, so the startup
    path hands the model one thinking note per result. It sits directly below
    the code this lane rewrote; moving the history seed past `ready` must not
    have moved it."""

    startup_clocks(monkeypatch)
    patch_relay(monkeypatch)
    live.remember_detached("db-session", "find the invoice", "It is with the accountant.")

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([config(), 0.08, {"type": "stop"}])
    await run_relay(client, provider, timeout=5.0)

    notes = [
        e for e in provider.of("session.thinking.append")
        if "It is with the accountant." in e["content"]
    ]
    assert notes, [e["content"][:60] for e in provider.of("session.thinking.append")]
    # Delivered as a NOTE, never spoken on arrival.
    assert live.take_detached_results("db-session") == []


@pytest.mark.asyncio
async def test_the_detached_note_survives_a_history_seed_that_never_returns(
    monkeypatch,
):
    """The two now share the startup tail. A seed left awaiting a dead tenant
    used to be in front of this note; it must no longer be in front of
    anything."""

    startup_clocks(monkeypatch, voice_live_history_rows=20)
    patch_relay(monkeypatch)
    from app.api import ws_realtime as rt

    async def never(_user_id):
        await asyncio.sleep(30)
        return None

    monkeypatch.setattr(rt, "_get_vps_info", never)
    # Not a DB measurement — see the never-returning seed test at the top.
    monkeypatch.setattr(rt, "_get_user_tz_name", _no_stored_zone)
    live.remember_detached("db-session", "find the invoice", "It is with the accountant.")

    provider = FakeProvider(on_send=close_on_close)
    client = FakeClient([config(), 0.1, {"type": "stop"}])
    t0 = time.monotonic()
    await run_relay(client, provider, timeout=5.0)

    assert ready_ms(client, t0) < 400, ready_ms(client, t0)
    assert any(
        "It is with the accountant." in e["content"]
        for e in provider.of("session.thinking.append")
    )
