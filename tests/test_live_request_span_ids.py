"""`delegation.request_turn_ids`: WHICH caller turns a task's request consumed.

R2 supervisor correction (causal request-span identity).  The app's
unanswered-request watch used to treat ANY delegation frame — including a late
`completed` for task A — as the answer to the caller's newest request B, so an
independent B ("separately, find tomorrow's weather") spoken while A ran was
never recovered once A finished.  The app now covers a turn with a task only
when the relay SAYS the task consumed it.  The relay already knows exactly
which turns that is (spec §B, `UtteranceAssembler.request_span` → `consume`):
this file pins that it ships them, additively and only under `task_lifecycle`:

  • every span turn in SPOKEN order (ascending ordinal), the ANCHOR
    (`parent_user_turn_id`) wherever it falls — last for an ordinary span,
    whose anchor is its newest turn, but first when a check-in is overtaken by
    the older request it swept in (addendum 2 item 2; this file pinned "anchor
    last" before, which put the check-in ahead of the question it followed);
  • bounded (`REQUEST_TURN_IDS_MAX`), the anchor always kept, then the newest;
  • on EVERY lifecycle observation of the task (created … completed), because
    it is identity, like `title`;
  • never a turn the task did not consume — a later independent request's
    turns are not in an earlier task's list, and vice versa;
  • absent for a client that did not negotiate `task_lifecycle`, and absent
    on an orphan that no turn was bound to (no id is invented).
"""

import asyncio
import time

import pytest

import test_live_harness as H


CLOCKS = dict(
    voice_live_transcript_settle_ms=40,
    voice_live_utterance_gap_ms=400,
    voice_live_utterance_hard_gap_ms=900,
    voice_live_output_epoch_gap_ms=250,
    voice_live_delegation_ttl_s=1.0,
    voice_live_interrupt_suppress_ms=300,
)

#: A request spoken across five closed turns (each pause > gap, < span gap).
PIECES = [
    ("What", 500, 650), (" is the best", 1450, 1750), (" professor in UofT", 2550, 2950),
    (" who is working on", 3750, 4150), (" LLM", 4950, 5100),
]


async def _call(monkeypatch, script, *, config=None, timeout=30.0, include_saves=False):
    H.fast_clocks(monkeypatch, **CLOCKS)
    calls: list[str] = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append(kwargs.get("display_request") or task)
        return "An answer.", "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        if event["type"] == "session.start":
            box["provider"] = provider
            box["t0"] = asyncio.get_running_loop().time()
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def at(wall_s):
        delay = box["t0"] + wall_s - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)

    client = H.FakeClient([
        config or H.config(), lambda: script(box["provider"], at, box), {"type": "stop"},
    ])
    box["client"] = client
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=timeout,
        db_session_id=f"db-span-ids-{time.monotonic_ns()}",
    )
    return (client, calls, recorded) if include_saves else (client, calls)


def _finals(client):
    return [f for f in client.of("transcript") if f.get("final")]


def _frames(client, delegation_id):
    return [f for f in client.of("delegation") if f.get("delegation_id") == delegation_id]


def _lifecycle(client, delegation_id):
    """The task's lifecycle OBSERVATIONS.  The relay's later delivery verdict
    (`phase:completed` + `spoken`, no state/revision) is deliberately not one:
    the app recognises it by exactly that absence, so it carries no identity
    fields at all — `relation` and `request_turn_ids` alike."""

    return [f for f in _frames(client, delegation_id) if "task_revision" in f]


async def _wait_for(box, predicate, *, tries=200):
    for _ in range(tries):
        if predicate(box["client"]):
            return True
        await asyncio.sleep(0.02)
    return False


# ── units: the tracker records the span it consumed ───────────────────────

def _turn(ordinal, text, start, end, **extra):
    from app.services.live_voice_protocol import Utterance

    fields = dict(causal_speech=True, closed=True)
    fields.update(extra)
    return Utterance(
        turn_id=f"live-utt:p:{ordinal}", ordinal=ordinal,
        start_ms=start, end_ms=end, text=text, **fields,
    )


def test_consume_records_every_span_turn_in_spoken_order():
    from app.services.live_voice_protocol import LiveDelegationTracker, UtteranceAssembler

    asm = UtteranceAssembler("p")
    answered = _turn(1, "hello", 0, 300, dispatched=True)
    a = _turn(2, "find the ر", 1000, 1400)
    b = _turn(3, "وبات", 2000, 2200)
    c = _turn(4, " lab at U of T", 3000, 3500)
    asm.closed_queue = [answered, a, b, c]
    span = asm.request_span(c, max_chars=600, max_gap_ms=8000)

    tracker = LiveDelegationTracker()
    tracker.add_delegation("d1", 3600)
    task = tracker.consume("d1", c, span=span)
    assert task is not None
    assert task.request_turn_ids == [a.turn_id, b.turn_id, c.turn_id]
    # An ordinary span's anchor is its newest turn, so spoken order ends on it.
    assert task.request_turn_ids[-1] == task.turn_id, "the newest (anchor) turn is not last"
    assert answered.turn_id not in task.request_turn_ids, "an answered turn joined the request"


def test_request_turn_ids_are_chronological_when_the_anchor_is_older():
    """Addendum 2 item 2: a check-in overtaken by the request it swept in
    anchors on that OLDER request (`rebind_to_requests`); the list stays in
    spoken order — question, then check-in — whatever order it is built in,
    and the bound keeps the anchor and then the newest turns."""

    from app.services.live_voice_protocol import REQUEST_TURN_IDS_MAX, span_turn_ids

    question = _turn(4, "who is the dean there?", 3000, 3800)
    check_in = _turn(5, "what happened?", 5000, 5500)
    assert span_turn_ids([question, check_in], question.turn_id) == [
        question.turn_id, check_in.turn_id,
    ]
    assert span_turn_ids([check_in, question], question.turn_id) == [
        question.turn_id, check_in.turn_id,
    ]
    assert span_turn_ids([check_in, question], check_in.turn_id) == [
        question.turn_id, check_in.turn_id,
    ]

    turns = [_turn(i, f" w{i}", i * 500, i * 500 + 100) for i in range(1, 21)]
    ids = span_turn_ids(list(reversed(turns)), turns[0].turn_id)
    assert len(ids) == REQUEST_TURN_IDS_MAX
    assert ids == [turns[0].turn_id] + [t.turn_id for t in turns[-(REQUEST_TURN_IDS_MAX - 1):]]


def test_a_single_turn_request_names_only_its_anchor():
    from app.services.live_voice_protocol import LiveDelegationTracker

    only = _turn(7, "what time is it", 0, 900)
    tracker = LiveDelegationTracker()
    tracker.add_delegation("d7", 1000)
    task = tracker.consume("d7", only)
    assert task is not None
    assert task.request_turn_ids == [only.turn_id]


def test_request_turn_ids_are_bounded_and_keep_the_anchor():
    from app.services.live_voice_protocol import (
        REQUEST_TURN_IDS_MAX, LiveDelegationTracker,
    )

    turns = [_turn(i, f" w{i}", i * 500, i * 500 + 100) for i in range(1, 21)]
    anchor = turns[-1]
    tracker = LiveDelegationTracker()
    tracker.add_delegation("d20", 20_000)
    task = tracker.consume("d20", anchor, span=turns)
    assert task is not None
    assert len(task.request_turn_ids) == REQUEST_TURN_IDS_MAX == 16
    assert task.request_turn_ids[-1] == anchor.turn_id  # the newest turn, in spoken order
    # The NEWEST priors survive the bound, in order.
    assert task.request_turn_ids == [t.turn_id for t in turns[-16:]]
    # …while every span turn is still consumed (at-most-once is unchanged).
    assert all(t.consumed_by == "d20" for t in turns)


# ── end to end: the frame carries it, on every observation ────────────────

@pytest.mark.asyncio
async def test_every_lifecycle_frame_names_the_whole_request_span(monkeypatch):
    async def script(p, at, box):
        for text, start, end in PIECES:
            await at(start / 1000.0)
            p.push(H.user_delta(text, start, end))
        p.push(H.delegation("d1", 5140))
        await _wait_for(box, lambda c: any(
            f.get("state") == "completed" for f in _frames(c, "d1")))
        await asyncio.sleep(0.3)

    client, calls = await _call(monkeypatch, script)

    finals = _finals(client)
    assert [f["text"] for f in finals] == [t.strip() for t, _s, _e in PIECES]
    ids = [f["turn_id"] for f in finals]
    frames = _lifecycle(client, "d1")
    assert frames, "no delegation frames"
    assert {f.get("state") for f in frames} >= {"queued", "completed"}
    for frame in frames:
        assert frame.get("request_turn_ids") == ids, frame
        # Spoken order; this span's anchor is its newest turn, so it is last.
        assert frame["request_turn_ids"][-1] == frame["parent_user_turn_id"] == frame["turn_id"]
    for verdict in [f for f in _frames(client, "d1") if "task_revision" not in f]:
        assert "request_turn_ids" not in verdict, "the auxiliary delivery verdict grew identity fields"
    assert calls == ["What is the best professor in UofT who is working on LLM"]


@pytest.mark.asyncio
async def test_saved_result_names_the_same_exact_user_span(monkeypatch):
    async def script(p, at, box):
        for text, start, end in PIECES:
            await at(start / 1000.0)
            p.push(H.user_delta(text, start, end))
        p.push(H.delegation("d1", 5140))
        await _wait_for(box, lambda c: any(
            f.get("state") == "completed" for f in _frames(c, "d1")))
        await asyncio.sleep(0.3)

    client, _calls, recorded = await _call(monkeypatch, script, include_saves=True)
    ids = [f["turn_id"] for f in _finals(client)]
    rows = [s for s in recorded["saves"]
            if str(s.get("assistant_ref") or "").startswith("live-delegation:")]
    assert rows
    from app.api.sessions import _clean_voice
    for row in rows:
        voice = row["assistant_voice"]
        assert voice["request_turn_ids"] == ids
        assert _clean_voice(dict(voice))["request_turn_ids"] == ids
    persisted_turn_ids = {
        (s.get("user_voice") or {}).get("turn_id") for s in recorded["saves"]
        if s.get("user_voice")
    }
    assert set(ids) <= persisted_turn_ids


@pytest.mark.asyncio
async def test_an_independent_later_request_never_shares_the_earlier_tasks_turns(monkeypatch):
    """Task A consumes its turns; B, spoken while A runs, is its OWN request:
    B's frames name only B's turn and A's frames never name B's — so a late A
    frame is provably not an answer to B."""

    release = asyncio.Event()

    async def script(p, at, box):
        p.push(H.user_delta("Search for robotics professors.", 500, 1500))
        p.push(H.delegation("dA", 1560))
        await _wait_for(box, lambda c: bool(_frames(c, "dA")))
        await at(3.0)
        p.push(H.user_delta("Separately, find tomorrow's weather.", 3000, 4200))
        p.push(H.delegation("dB", 4260))
        await _wait_for(box, lambda c: bool(_frames(c, "dB")))
        release.set()
        await _wait_for(box, lambda c: all(
            any(f.get("state") in {"completed", "failed", "canceled"} for f in _frames(c, d))
            for d in ("dA", "dB")))
        await asyncio.sleep(0.3)

    async def think(_user_id, task, _session_id, **kwargs):
        if "robotics" in (kwargs.get("display_request") or task):
            await release.wait()
        return "An answer.", "test-model"

    H.fast_clocks(monkeypatch, **CLOCKS)
    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        if event["type"] == "session.start":
            box["provider"] = provider
            box["t0"] = asyncio.get_running_loop().time()
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def at(wall_s):
        delay = box["t0"] + wall_s - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)

    client = H.FakeClient([H.config(), lambda: script(box["provider"], at, box), {"type": "stop"}])
    box["client"] = client
    await H.run_relay(
        client, H.FakeProvider(on_send=on_send, auto_ack=True), timeout=30.0,
        db_session_id=f"db-span-ids-ab-{time.monotonic_ns()}",
    )

    finals = _finals(client)
    assert [f["text"] for f in finals] == [
        "Search for robotics professors.", "Separately, find tomorrow's weather.",
    ]
    turn_a, turn_b = finals[0]["turn_id"], finals[1]["turn_id"]
    frames_a, frames_b = _lifecycle(client, "dA"), _lifecycle(client, "dB")
    assert frames_a and frames_b
    assert all(f.get("request_turn_ids") == [turn_a] for f in frames_a), frames_a
    assert all(f.get("request_turn_ids") == [turn_b] for f in frames_b), frames_b
    # The late terminal frame of A names A only.
    late_a = [f for f in frames_a if f.get("state") in {"completed", "failed", "canceled"}]
    assert late_a and all(turn_b not in f["request_turn_ids"] for f in late_a)


@pytest.mark.asyncio
async def test_no_request_turn_ids_without_task_lifecycle(monkeypatch):
    legacy_features = H.config()
    legacy_features["features"] = [f for f in legacy_features["features"] if f != "task_lifecycle"]

    async def script(p, at, box):
        p.push(H.user_delta("What time is it in Tokyo", 500, 1500))
        p.push(H.delegation("d1", 1560))
        await _wait_for(box, lambda c: any(
            f.get("phase") == "completed" for f in _frames(c, "d1")))
        await asyncio.sleep(0.2)

    client, _calls = await _call(monkeypatch, script, config=legacy_features)
    frames = _frames(client, "d1")
    assert frames, "the legacy delegation family went dark"
    assert all("request_turn_ids" not in f for f in frames), frames


@pytest.mark.asyncio
async def test_an_orphan_delegation_invents_no_turn_ids(monkeypatch):
    async def script(p, at, box):
        # No caller speech at all: nothing this delegation could be bound to.
        p.push(H.delegation("d-orphan", 100))
        await _wait_for(box, lambda c: bool(_frames(c, "d-orphan")))
        await asyncio.sleep(0.2)

    client, _calls = await _call(monkeypatch, script)
    frames = _frames(client, "d-orphan")
    assert [f.get("phase") for f in frames] == ["expired"]
    assert all("request_turn_ids" not in f for f in frames), frames
