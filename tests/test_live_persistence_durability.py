"""What a Live call leaves behind when the tenant is slow, flaky, or busy.

Round R2, persistence P1–P6. Production (RUNTIME_EVIDENCE §10, one 12-minute
call): 273 "Saving to VPS session" POSTs, a `LOST 1 voice message(s)` ERROR
logged on an attempt the queue then RETRIED, and at hang-up
`persistence drain timed out; 25 queued` with no loss signal after it — the
worker was cancelled and whatever it still held was never written and never
counted. Each test below states the property that production broke.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone

import pytest

import test_live_harness as H


def _msg(key, *, rev=None, text="x", side="user", **extra):
    import app.services.live_voice_protocol as live

    payload = {"user_text": "", "assistant_text": "", f"{side}_ref": key}
    payload[f"{side}_text"] = text
    if rev is not None:
        payload[f"{side}_revision"] = rev
    payload.update(extra)
    return live.PersistenceRecord(kind="message", key=key, payload=payload)


class _Resp:
    """The one shape `_vps_api` reads off an httpx response."""

    def __init__(self, status, body=b'{"id": "m1"}'):
        self.status_code = status
        self.content = body

    def json(self):
        import json
        return json.loads(self.content)


def _fake_tenant(monkeypatch, answers):
    """Real `_save_voice_messages` + real `_vps_api`, fake HTTP underneath.

    `answers` is consumed one per POST: an int is a status code, an exception
    instance is raised. Returns the list of POSTed bodies.
    """

    from app.api import ws_realtime as rt
    import app.services.agent_http as ah

    posts: list[dict] = []
    script = list(answers)

    async def vps(_uid):
        return ("http://agent.test", "k")

    class _Client:
        async def post(self, url, **kw):
            posts.append(dict(kw.get("json") or {}))
            answer = script.pop(0) if len(script) > 1 else script[0]
            if isinstance(answer, BaseException):
                raise answer
            return _Resp(answer)

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(ah, "get_agent_http_client", lambda: _Client())
    return posts


# ── P1: nothing queued at hang-up vanishes ────────────────────────────

@pytest.mark.asyncio
async def test_records_left_when_the_queue_stops_are_counted_lost_never_vanish(monkeypatch):
    """`stop()` used to cancel the worker and walk away from the queue: of five
    records one was written, none was counted lost and no counter fired."""

    import app.services.live_voice_protocol as live

    async def slow_save(user_id, session_id, **kw):
        await asyncio.sleep(0.2)
        return 0

    H.patch_relay(monkeypatch, save=slow_save)
    counters = H.capture_counters(monkeypatch)
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    for i in range(5):
        queue.submit(_msg(f"live-utt:psid:{i}", text=f"sentence {i}"))
    assert await queue.drain(0.3) is False
    await queue.stop()

    assert queue.written + queue.lost == 5, (
        f"{5 - queue.written - queue.lost} record(s) vanished: "
        f"written={queue.written} lost={queue.lost}"
    )
    lost = [fields for name, fields in counters if name == "live_persist_lost"]
    assert len(lost) == queue.lost
    assert {f.get("reason") for f in lost} == {"teardown_budget"}


@pytest.mark.asyncio
async def test_twenty_five_rows_queued_at_hang_up_all_land_after_the_socket_closes(monkeypatch):
    """The production shape: a slow tenant, a backlog at hang-up, and a short
    inline drain. The handler must not wait for the tenant, and the backlog is
    written by the detached finisher instead of being thrown away."""

    H.fast_clocks(
        monkeypatch, voice_live_utterance_gap_ms=200,
        voice_live_persist_drain_s=0.3,
        voice_live_persist_background_drain_s=30.0,
    )
    written: dict[str, str] = {}

    async def slow_save(user_id, session_id, **kw):
        await asyncio.sleep(0.05)
        ref = kw.get("user_ref") or kw.get("assistant_ref")
        written[ref] = kw.get("user_text") or kw.get("assistant_text")
        return 0

    H.patch_relay(monkeypatch, save=slow_save)
    counters = H.capture_counters(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            for i in range(25):
                provider.push(H.user_delta(f"sentence {i} ends", i * 1000, i * 1000 + 400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.3, {"type": "stop"}])
    started = time.monotonic()
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=10, drain=False)
    returned = time.monotonic() - started
    # The finisher is a detached task; join it (bounded) like the harness does.
    await H.drain_detached(timeout=20.0)

    texts = list(written.values())
    missing = [i for i in range(25) if not any(f"sentence {i} ends" in t for t in texts)]
    assert not missing, f"{len(missing)}/25 turns never reached the thread: {missing}"
    assert returned < 5.0, f"the socket handler waited {returned:.1f}s on the tenant"
    assert not [n for n, _ in counters if n in ("live_persist_lost", "voice_transcript_lost")]


@pytest.mark.asyncio
async def test_a_finisher_cancelled_at_shutdown_counts_its_backlog_instead_of_waiting(monkeypatch):
    """The finisher outlives the socket, not the process. When shutdown
    cancels it, it must neither sit out its two-minute budget nor let the
    cancelled drain skip the accounting."""

    import app.services.live_voice_protocol as live

    async def hung_save(user_id, session_id, **kw):
        await asyncio.sleep(3600)

    H.patch_relay(monkeypatch, save=hung_save)
    counters = H.capture_counters(monkeypatch)
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    for i in range(3):
        queue.submit(_msg(f"live-utt:p:{i}", text=f"sentence {i}"))
    finisher = asyncio.create_task(live._finish_detached(queue, [], 60.0, 120.0))
    await asyncio.sleep(0.05)
    started = time.monotonic()
    finisher.cancel()
    await asyncio.gather(finisher, return_exceptions=True)

    assert time.monotonic() - started < 1.0
    assert queue.written == 0 and queue.lost == 3
    lost = [f.get("reason") for name, f in counters if name == "live_persist_lost"]
    assert lost == ["teardown_budget"] * 3


# ── P3: memory curation cannot hold a transcript row ──────────────────

@pytest.mark.asyncio
async def test_a_curator_call_hanging_for_a_minute_never_delays_a_transcript_row(monkeypatch):
    """`curate` records shared the one serial lane with transcript rows, so a
    tenant LLM call (60 s budget) held every later row — and the teardown
    drain — behind it."""

    H.fast_clocks(
        monkeypatch, voice_live_utterance_gap_ms=300, auto_extract_memories=True,
        voice_live_persist_drain_s=2.0, voice_live_curate_drain_s=0.2,
    )
    started = time.monotonic()
    saves: list[tuple[str, float]] = []
    curating = asyncio.Event()

    async def save(user_id, session_id, **kw):
        saves.append((kw.get("user_text") or kw.get("assistant_text"), time.monotonic() - started))
        return 0

    async def curate(user_id, user_text, assistant_text):
        curating.set()
        await asyncio.sleep(60)

    H.patch_relay(monkeypatch, save=save, curate=curate)
    counters = H.capture_counters(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("I prefer mornings", 0, 500))
            provider.push(H.user_delta("and I hate coriander", 3000, 3600))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.0, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=15)
    returned = time.monotonic() - started

    assert curating.is_set(), "the first turn must have been handed to the curator"
    second = [t for text, t in saves if text == "and I hate coriander"]
    assert second, "the second turn's row was never written"
    assert second[0] < 1.0, (
        f"the second row waited for the curator: written at {second[0]:.2f}s"
    )
    assert returned < 5.0, f"hang-up waited {returned:.1f}s on a memory write"
    # Not silently discarded either: the hung curation is counted.
    assert any(name == "live_curate_dropped" for name, _ in counters)


@pytest.mark.asyncio
async def test_the_curation_lane_is_bounded_and_sheds_its_oldest_waiting_turn(monkeypatch):
    """Low priority means bounded: a tenant whose curator never answers must
    not grow a per-call backlog without limit, and what is shed is counted."""

    import app.services.live_voice_protocol as live

    counters = H.capture_counters(monkeypatch)
    seen: list[str] = []

    async def curate(user_id, user_text, assistant_text):
        seen.append(user_text)
        await asyncio.sleep(60)

    H.patch_relay(monkeypatch, curate=curate)
    lane = live.PersistenceQueue("user-1", "db-session", maxsize=2, lane="curate",
                                 drop_oldest=True)
    lane.start()
    for i in range(4):
        lane.submit(live.PersistenceRecord(
            kind="curate", key=f"curate:live-utt:p:{i}",
            payload={"user_text": f"turn {i}", "assistant_text": ""},
        ))
        await asyncio.sleep(0.01)          # turn 0 is on the wire from here on
    await lane.stop()

    assert seen == ["turn 0"]
    shed = [f.get("reason") for name, f in counters if name == "live_curate_dropped"]
    # turn 1 was shed for turn 3; turn 0 (in flight) and 2, 3 at stop.
    assert shed == ["queue_full", "teardown_budget", "teardown_budget", "teardown_budget"]
    assert not [n for n, _ in counters if n in ("live_persist_lost", "voice_transcript_lost")], (
        "a memory write that never ran is not a lost transcript"
    )


# ── P2: one row, newest revision wins, never out of order ─────────────

@pytest.mark.asyncio
async def test_an_older_revision_never_lands_after_a_newer_one(monkeypatch):
    """The tenant stores no revision and overwrites on every upsert, so the
    relay is the only thing keeping a row's writes in order. A revision queued
    behind an in-flight one of the same row waits for it; revisions queued
    together collapse into the newest."""

    import app.services.live_voice_protocol as live

    posted: list[int] = []
    inflight = {"now": 0, "max": 0}
    hold = asyncio.Event()

    async def save(user_id, session_id, **kw):
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        if kw.get("user_revision") == 1:
            await hold.wait()
        posted.append(kw.get("user_revision"))
        inflight["now"] -= 1
        return 0

    H.patch_relay(monkeypatch, save=save)
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(_msg("K", rev=1))
    await asyncio.sleep(0.02)                  # revision 1 is on the wire
    for rev in (2, 3, 4):
        queue.submit(_msg("K", rev=rev, text="x" * rev))
    hold.set()
    assert await queue.drain(2.0)
    await queue.stop()

    assert posted == [1, 4], posted
    assert inflight["max"] == 1, "two revisions of one row were in flight together"
    assert queue.written == 2 and queue.lost == 0


@pytest.mark.asyncio
async def test_queued_revisions_of_one_row_coalesce_to_the_latest(monkeypatch):
    import app.services.live_voice_protocol as live

    posted: list[tuple] = []

    async def save(user_id, session_id, **kw):
        await asyncio.sleep(0.1)
        posted.append((kw.get("user_ref"), kw.get("user_revision"), kw.get("user_text")))
        return 0

    H.patch_relay(monkeypatch, save=save)
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(_msg("other", rev=1, text="blocker"))
    await asyncio.sleep(0.02)                  # the lane is busy from here on
    for rev in range(1, 6):
        queue.submit(_msg("K", rev=rev, text="x" * rev))
    queue.submit(_msg("after", rev=1, text="later row"))
    assert await queue.drain(3.0)
    await queue.stop()

    assert [p for p in posted if p[0] == "K"] == [("K", 5, "xxxxx")]
    # The coalesced row kept its place in line: before the row queued after it.
    assert [p[0] for p in posted] == ["other", "K", "after"]


@pytest.mark.asyncio
async def test_a_slow_tenant_gets_fewer_writes_and_the_newest_revision_lands_last(monkeypatch):
    """Through the real relay: a turn revised faster than the tenant answers.
    Every revision used to be POSTed (273 for one call in production); now the
    revisions that queued behind a slow write collapse, and the row still ends
    on its newest text."""

    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=5000)
    submitted: list[tuple] = []
    original_submit = live.PersistenceQueue.submit

    def spy_submit(self, record):
        if record.kind == "message" and record.payload.get("user_ref"):
            submitted.append((record.key, record.payload.get("user_revision")))
        return original_submit(self, record)

    monkeypatch.setattr(live.PersistenceQueue, "submit", spy_submit)
    posts: list[tuple] = []

    async def slow_save(user_id, session_id, **kw):
        await asyncio.sleep(0.25)
        if kw.get("user_ref"):
            posts.append((kw["user_ref"], kw.get("user_revision"), kw.get("user_text")))
        return 0

    H.patch_relay(monkeypatch, save=slow_save)
    provider = H.FakeProvider(on_send=lambda p, e: (
        p.push(H.closed()) if e["type"] == "session.close" else None))
    words = ["one ", "two ", "three ", "four ", "five ", "six"]

    def says(i):
        return lambda: provider.push(H.user_delta(words[i], i * 300, i * 300 + 200))

    script = [H.config(), 0.05]
    for i in range(len(words)):
        script += [says(i), 0.07]
    client = H.FakeClient(script + [0.3, {"type": "stop"}])
    await H.run_relay(client, provider, timeout=10)

    assert posts and len({ref for ref, _rev, _text in posts}) == 1
    revisions = [rev for _ref, rev, _text in posts]
    assert revisions == sorted(set(revisions)), f"a row's writes went out of order: {revisions}"
    assert posts[-1][2] == "one two three four five six"
    assert posts[-1][1] == max(rev for _key, rev in submitted)
    assert len(posts) < len(submitted), (
        f"{len(submitted)} revisions, {len(posts)} POSTs: nothing coalesced"
    )


@pytest.mark.asyncio
async def test_a_coalesced_revision_keeps_the_first_writes_occurred_at(monkeypatch):
    """`write_spoken_flag` omits `occurred_at` deliberately (the row already
    exists). If the row's FIRST write is still queued when that revision
    replaces it, dropping the stamp would create the row with the tenant's
    arrival time instead of its place in the call."""

    import app.services.live_voice_protocol as live

    posted: list[dict] = []

    async def save(user_id, session_id, **kw):
        await asyncio.sleep(0.05)
        posted.append(kw)
        return 0

    H.patch_relay(monkeypatch, save=save)
    stamp = datetime(2026, 9, 23, 2, 59, 25, tzinfo=timezone.utc)
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(_msg("blocker", text="first"))
    await asyncio.sleep(0.01)                  # the lane is busy from here on
    queue.submit(_msg("live-delegation:p:d1", rev=1, text="answer", side="assistant",
                      assistant_occurred_at=stamp))
    queue.submit(_msg("live-delegation:p:d1", rev=2, text="answer", side="assistant",
                      assistant_voice={"spoken": True}))
    assert await queue.drain(2.0)
    await queue.stop()

    rows = [p for p in posted if p.get("assistant_ref") == "live-delegation:p:d1"]
    assert len(rows) == 1
    assert rows[0]["assistant_revision"] == 2
    assert rows[0]["assistant_voice"] == {"spoken": True}
    assert rows[0]["assistant_occurred_at"] == stamp


# ── P4: a failure is LOST only when it is final ───────────────────────

@pytest.mark.asyncio
async def test_a_transient_failure_then_success_is_not_logged_as_lost(monkeypatch, caplog):
    """Production V3+01:09.4: `VPS API POST … failed: ` (an httpx ReadTimeout
    prints an empty message) → `LOST 1 voice message(s)` → the queue retried
    and the row landed. The ERROR and `voice_transcript_lost` had already
    claimed a transcript was not persisted."""

    import httpx
    import app.services.live_voice_protocol as live

    counters = H.capture_counters(monkeypatch)
    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0, 0.0, 0.0))
    posts = _fake_tenant(monkeypatch, [httpx.ReadTimeout(""), 201])
    caplog.set_level(logging.INFO)

    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(_msg("live-utt:p:1", rev=1, text="hello"))
    assert await queue.drain(3.0)
    await queue.stop()

    assert len(posts) == 2
    assert queue.written == 1 and queue.lost == 0
    names = [name for name, _ in counters]
    assert "voice_transcript_lost" not in names, "a recovered row must not be counted lost"
    assert "live_persist_lost" not in names
    assert names.count("live_persist_retry") == 1
    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if "LOST" in m], messages
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    # The failure names its class: an empty `str(ReadTimeout)` is not a cause.
    assert any("ReadTimeout" in m for m in messages if "VPS API" in m), messages


@pytest.mark.asyncio
async def test_a_permanent_failure_is_not_retried_and_is_lost_exactly_once(monkeypatch, caplog):
    """A 400 ("Content is required") will be a 400 on every retry; the ladder
    only lengthens head-of-line blocking. It is final at once, and final is
    reported once."""

    import app.services.live_voice_protocol as live

    counters = H.capture_counters(monkeypatch)
    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0, 0.0, 0.0))
    posts = _fake_tenant(monkeypatch, [400])
    caplog.set_level(logging.INFO)

    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(_msg("live-output:p:3", rev=1, text="an answer", side="assistant"))
    assert await queue.drain(3.0)
    await queue.stop()

    assert len(posts) == 1, f"a permanent failure was retried {len(posts) - 1} time(s)"
    assert queue.written == 0 and queue.lost == 1
    lost = [fields for name, fields in counters if name == "voice_transcript_lost"]
    assert len(lost) == 1 and lost[0].get("n") == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len([m for m in errors if "LOST 1" in m]) == 1, errors
    assert any("400" in m for m in errors), errors
    # Key and counts only — the row's words never reach the log.
    assert not [m for m in errors if "an answer" in m]


@pytest.mark.asyncio
async def test_a_transient_failure_that_never_recovers_is_lost_once_at_the_end(monkeypatch, caplog):
    """F33 changed WHEN "the end" is: a refused connection is an outage, and it
    is retried until the outage window has passed (not after three attempts
    in 1.25 s, which lost a row to a 2 s tenant restart). It is still lost
    exactly once, loudly, when the window is spent. This pinned `len(posts)
    == 3` / two retries — the defect — and now pins the window."""

    import httpx
    import app.services.live_voice_protocol as live

    counters = H.capture_counters(monkeypatch)
    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.05,))
    posts = _fake_tenant(monkeypatch, [httpx.ConnectError("refused")])
    caplog.set_level(logging.INFO)

    queue = live.PersistenceQueue("user-1", "db-session")
    queue.retry_window_s = 0.3
    queue.start()
    started = time.monotonic()
    queue.submit(_msg("live-utt:p:2", rev=1, text="hello"))
    assert await queue.drain(3.0)
    elapsed = time.monotonic() - started
    await queue.stop()

    # More than the old three attempts, bounded by the window (and by the
    # hard `MAX_OUTAGE_ATTEMPTS` cap, the literal 12, when backoff is ~0).
    assert 3 < len(posts) <= 12, len(posts)
    assert 0.25 <= elapsed < 1.0, elapsed
    assert queue.written == 0 and queue.lost == 1
    names = [name for name, _ in counters]
    assert names.count("voice_transcript_lost") == 1, names
    assert names.count("live_persist_retry") == len(posts) - 1, names
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len([m for m in errors if "LOST" in m]) == 1, errors
    assert any("ConnectError" in m for m in errors), errors


# ── P5: a row keeps its place ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_user_row_keeps_its_occurred_at_across_revisions(monkeypatch):
    """Every revision re-ran `stamp()`, which clamps to last+1 ms, and the
    tenant overwrites `occurred_at` on every upsert: the row drifted later in
    the saved thread with each rewrite."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=5000)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("که ", 0, 300))
            provider.push(H.user_delta("ایونت ", 700, 1100))
            provider.push(H.user_delta("معروف", 1500, 1900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.5, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    rows = [s for s in recorded["saves"] if s.get("user_text")]
    assert len(rows) >= 2, "the test needs at least two revisions of the row"
    stamps = {r["user_occurred_at"] for r in rows}
    assert len(stamps) == 1, f"one row, {len(stamps)} different positions"


@pytest.mark.asyncio
async def test_a_spoken_row_is_stamped_at_its_epoch_start_and_keeps_it(monkeypatch):
    """A spoken row was stamped when it was PERSISTED (`provider_clock_ms()`),
    not where the reply began, and every late ack re-stamped it again."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=250)
    recorded = H.patch_relay(monkeypatch)
    provider = H.FakeProvider(on_send=lambda p, e: (
        p.push(H.user_delta("hello there", 0, 400)) if e["type"] == "session.start"
        else p.push(H.closed()) if e["type"] == "session.close" else None
    ))

    def speaks():
        provider.push(H.out_text("Hi, how can I help?", 1500, 2300))
        provider.push(H.out_audio())

    oid = f"live:{H.PSID}:1"
    client = H.FakeClient([
        H.config(), 0.3, speaks, 0.7,
        {"type": "playback_idle", "response_id": oid, "item_id": oid, "played_ms": 800},
        0.2, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)

    user = [s for s in recorded["saves"] if s.get("user_text")]
    spoken = [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_spoken"
    ]
    assert user and spoken
    assert len(spoken) >= 2, "the late ack must revise the spoken row"
    assert len({s["assistant_occurred_at"] for s in spoken}) == 1
    # The user row is the call's first stamp (provider 0 ms); the reply began
    # at provider 1500 ms, and its row says exactly that.
    offset = spoken[0]["assistant_occurred_at"] - user[0]["user_occurred_at"]
    assert offset.total_seconds() == pytest.approx(1.5, abs=0.001), offset


@pytest.mark.asyncio
async def test_a_barge_in_row_and_the_answer_it_interrupted_never_swap_places(monkeypatch):
    """V1 03:37 shape (U:35/U:36 over a 19.8 s answer): the saved order after
    the first writes must be the saved order at the end. The tenant sorts by
    `occurred_at` and overwrites it on every upsert, so a re-stamped revision
    is a row that moved."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=5000,
                  voice_live_output_epoch_gap_ms=3000)
    recorded = H.patch_relay(monkeypatch)
    provider = H.FakeProvider(on_send=lambda p, e: (
        p.push(H.closed()) if e["type"] == "session.close" else None))

    def assistant_speaks():
        provider.push(H.out_text("Here is a long list of professors ", 100, 900))
        provider.push(H.out_audio())

    def user_starts():
        provider.push(H.user_delta("is their ", 1000, 1300))

    def user_continues():
        provider.push(H.user_delta("work on LLMs", 1400, 1800))

    oid = f"live:{H.PSID}:1"
    client = H.FakeClient([
        H.config(), 0.1, assistant_speaks, 0.2, user_starts, 0.1,
        {"type": "interrupt", "response_id": oid, "item_id": oid, "played_ms": 700},
        0.2, user_continues, 0.4, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)

    first: dict = {}
    last: dict = {}
    for save in recorded["saves"]:
        side = "user" if save.get("user_text") else "assistant"
        ref, at = save.get(f"{side}_ref"), save.get(f"{side}_occurred_at")
        if ref and at is not None:
            first.setdefault(ref, at)
            last[ref] = at
    assert len(first) >= 2, first
    assert sorted(first, key=first.get) == sorted(last, key=last.get), (
        "a row changed places between its first and its last write"
    )
    assert first == last


# ── P6: backend time, decomposed (counts only) ────────────────────────

@pytest.mark.asyncio
async def test_think_reports_the_agents_own_time_and_tokens(monkeypatch):
    """The done payload carries `processing_time_ms` and token counts; the
    relay threw them away, so 5.4-16 s of backend time could not be split
    into agent time and transport from production logs."""

    from app.api import ws_realtime as rt
    from app.config import settings

    monkeypatch.setattr(rt, "_agent_runner", None)
    monkeypatch.setattr(rt, "_v2_active", lambda *a, **k: True)
    monkeypatch.setattr(settings, "voice_realtime_tool_events", False, raising=False)

    class _Decision:
        model = "gpt-5.5"

    monkeypatch.setattr("app.services.model_router.classify_request", lambda task: _Decision())

    async def vps(_user_id):
        return ("https://u.agents.test", "agent-key")

    async def vps_api(agent_url, key, method, path, params=None, json_body=None, timeout=15.0):
        return {"text": "Booked.", "model": "gpt-5.5", "processing_time_ms": 4200,
                "tokens_input": 26000, "tokens_output": 150}

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)

    out: dict = {}
    text, _model = await rt._think("user-1", "book it", "sess-1", out=out)
    assert text == "Booked."
    assert out["processing_time_ms"] == 4200
    assert out["tokens_input"] == 26000 and out["tokens_output"] == 150


@pytest.mark.asyncio
async def test_delegation_finished_splits_backend_time(monkeypatch, caplog):
    H.fast_clocks(monkeypatch)
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.1)
        await relay.on_event({"type": "tool.start", "call_id": "i1", "name": "web_search",
                              "args": {"query": "uoft llm professors"}})
        await asyncio.sleep(0.1)
        if out is not None:
            out.update({"processing_time_ms": 120, "tokens_input": 900, "tokens_output": 40})
        return "Three professors work on LLMs.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("which uoft professors work on llms", 0, 600))
            provider.push(H.delegation("d1", 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    caplog.set_level(logging.INFO)
    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send, auto_ack=True), timeout=6)

    lines = [r.getMessage() for r in caplog.records if "delegation finished" in r.getMessage()]
    assert lines, "no 'delegation finished' line"
    line = lines[0]
    fields = dict(re.findall(r"(\w+)=(-?\d+)", line))
    assert 50 <= int(fields["first_tool_ms"]) <= 1000, line
    assert int(fields["agent_ms"]) == 120
    assert int(fields["tokens_in"]) == 900 and int(fields["tokens_out"]) == 40
    assert int(fields["transport_ms"]) == int(fields["elapsed_ms"]) - 120
    # Counts only: neither the request nor the answer is in the line.
    assert "professors" not in line
    latency = [f for name, f in counters if name == "live_delegation_latency"]
    assert latency and latency[0].get("bucket")
