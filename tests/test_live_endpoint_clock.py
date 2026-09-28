"""Where a caller's turn ENDS on the GPT-Live relay, and what a delegation binds to.

Spec v0.3 §A and §B, from the 2026-09-22 recordings: one Persian question
became seven turns ('چرا','نمی‌ت','ونی',...), the task was the last fragment
('کنی؟'), a status question split as 'خب چی' + 'شد' ran a second research turn
titled 'شد', and the agent's own voice held its own audio and prefixed the
caller's next request.

The relay measured the silence that closes a user turn on `provider_clock_ms()`,
a wall-anchored clock that assistant output positions and delegation offsets
also push forward.  When the provider's INPUT timeline sits behind that clock
by more than the gap, every fragment already reads as silent and closes on the
first tick.  These tests drive the real relay against the fake GPT-Live
harness with provider-shaped events on scaled clocks (settle 40 / gap 400 /
hard gap 900 ms — the production 350 / 1200 / 2600 relationship: settle much
shorter than the inter-word pause, gap much shorter than a real silence).

Every scenario has a positive twin: a genuinely separate utterance after real
silence is still its own turn, and exact provider replays can neither move
the input clock nor duplicate a turn.
"""

import asyncio
import logging
import re
import time

import pytest

import test_live_harness as H


CLOCKS = dict(
    voice_live_transcript_settle_ms=40,
    voice_live_utterance_gap_ms=400,
    voice_live_utterance_hard_gap_ms=900,
    voice_live_output_epoch_gap_ms=250,
    voice_live_delegation_ttl_s=3.0,
    voice_live_interrupt_suppress_ms=300,
)


class TimedClient(H.FakeClient):
    """The phone, with the arrival instant of EVERY frame (not just the first)."""

    def __init__(self, script=None):
        super().__init__(script)
        self.times: list[float] = []

    async def send_json(self, frame):
        self.times.append(time.monotonic())
        await super().send_json(frame)


async def _call(monkeypatch, script, *, clocks=None, think=None, timeout=30.0, box=None):
    """Run one relay call; `script(provider, at)` drives it on wall time since
    `session.start` (≈ the relay's provider-clock origin).  `box["client"]`
    exposes the phone to a script that has to wait on a frame."""

    H.fast_clocks(monkeypatch, **dict(CLOCKS, **(clocks or {})))
    calls: list[dict] = []

    async def default_think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, "display": kwargs.get("display_request")})
        return "An answer.", "test-model"

    H.patch_relay(monkeypatch, think=think or default_think)
    box = {} if box is None else box

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

    client = TimedClient([
        H.config(), lambda: script(box["provider"], at), {"type": "stop"},
    ])
    box["client"] = client
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=timeout,
        db_session_id=f"db-endpoint-clock-{time.monotonic_ns()}",
    )
    return client, provider, calls


def _finals(client):
    return [f["text"] for f in client.of("transcript") if f.get("final")]


def _created(client):
    return [f for f in client.of("delegation") if f.get("phase") == "created"]


async def _say(p, at, words, *, wall_s, position_ms, step_s=0.12, span_ms=120):
    """Push `words` 120 ms apart; returns the provider position after the last."""

    position = position_ms
    for index, word in enumerate(words):
        await at(wall_s + index * step_s)
        p.push(H.user_delta(word, position, position + span_ms))
        position += span_ms
    return position


# ── §A: the endpoint is measured on the INPUT timeline ───────────────────

FA = ["چرا", " نمی‌ت", "ونی", " آهنگ", " رو", " عوض", " کنی؟"]
FA_SENTENCE = "".join(FA).strip()


@pytest.mark.asyncio
async def test_input_timeline_behind_the_relay_clock_keeps_one_persian_turn(monkeypatch):
    """Production V1+01:12 onward: the input timeline ~0.9 s (gap < lag)
    behind the relay clock.  The sub-word split 'نمی‌ت' + 'ونی' must rejoin in
    ONE turn, and the delegation must carry the whole sentence."""

    async def script(p, at):
        # First word arrives at wall 1.0 s carrying input position 100 ms.
        end = await _say(p, at, FA, wall_s=1.0, position_ms=100)
        p.push(H.delegation("d-fa", end + 40))
        await at(1.0 + len(FA) * 0.12 + 1.6)

    client, _provider, calls = await _call(monkeypatch, script)

    assert _finals(client) == [FA_SENTENCE]
    turn_ids = {f["turn_id"] for f in client.of("transcript")}
    assert len(turn_ids) == 1, turn_ids
    created = _created(client)
    assert [c["title"] for c in created] == [FA_SENTENCE]
    assert created[0]["turn_id"] in turn_ids
    assert [c["display"] for c in calls] == [FA_SENTENCE]
    assert calls[0]["task"].startswith(FA_SENTENCE)


EN = ["What", " is the", " most", " famous", " building", " at", " U of T?"]


@pytest.mark.asyncio
async def test_assistant_output_ahead_of_wall_time_does_not_split_the_next_turn(
    monkeypatch, caplog,
):
    """A long reply whose transcript positions run to ~14 s while 0.2 s of wall
    time has passed pushes the WIRE clock forward.  The caller's next turn,
    at real input positions, is still one turn — and the wire clock is never
    pulled back (lifecycle `updated_ms` stays monotonic for the shipped app)."""

    async def script(p, at):
        await at(0.2)
        for i in range(10):
            p.push(H.out_text(
                f" sentence {i} of a long spoken reply.", 200 + i * 1400, 200 + (i + 1) * 1400,
            ))
        p.push(H.out_audio("AAAA"))
        end = await _say(p, at, EN, wall_s=1.5, position_ms=1500)
        p.push(H.delegation("d-en", end + 40))
        await at(1.5 + len(EN) * 0.12 + 1.6)

    from app.services import live_voice_protocol as live

    with caplog.at_level(logging.INFO, logger=live.logger.name):
        client, _provider, calls = await _call(monkeypatch, script)

    sentence = "".join(EN)
    assert _finals(client) == [sentence]
    assert [c["title"] for c in _created(client)] == [sentence]
    assert [c["display"] for c in calls] == [sentence]
    stamps = [f["updated_ms"] for f in client.of("delegation") if "updated_ms" in f]
    assert stamps and stamps == sorted(stamps), stamps
    assert stamps[0] >= 14200, "the wire clock must never be re-anchored backwards"
    # Counts-only diagnostic at turn close: how far the wire clock ran ahead of
    # the input clock.  Never the caller's words.
    closes = [r.getMessage() for r in caplog.records if "turn closed" in r.getMessage()]
    assert closes, "no turn-close log line"
    skew = re.search(r"skew_ms=(-?\d+)", closes[-1])
    assert skew is not None, closes[-1]
    assert int(skew.group(1)) >= 5000
    assert not any("famous" in line for line in closes)


HESITANT = [
    "What", " the most", ", uh... the best", " professor in UofT",
    " who is", " working on", " LLM",
]


@pytest.mark.asyncio
async def test_english_hesitation_on_a_lagging_input_timeline_is_one_turn(monkeypatch):
    """Production V3 U:91–U:100: the UofT/LLM question split into ten turns and
    no delegation ever ran.  Uneven pacing (two 250 ms hesitations — longer
    than the settle, shorter than the gap) on a timeline 1.7 s behind."""

    delays = [0.0, 0.12, 0.25, 0.12, 0.12, 0.12, 0.25]

    async def script(p, at):
        wall, position = 2.0, 300
        for word, delay in zip(HESITANT, delays):
            wall += delay
            position += int(delay * 1000)
            await at(wall)
            p.push(H.user_delta(word, position, position + 100))
        p.push(H.delegation("d-hes", position + 140))
        await at(wall + 1.6)

    client, _provider, calls = await _call(monkeypatch, script)

    sentence = "".join(HESITANT)
    assert _finals(client) == [sentence]
    assert [c["title"] for c in _created(client)] == [sentence]
    assert [c["display"] for c in calls] == [sentence]


@pytest.mark.asyncio
async def test_a_real_silence_still_separates_two_utterances(monkeypatch):
    """The positive twin: on the same lagging timeline, 1.2 s of real silence
    (on both clocks) between two requests is still two turns, and the second
    delegation is not joined to the first request its own delegation owns."""

    first = ["Find", " the", " Robarts", " library", " hours."]
    second = [" And", " the", " Gerstein", " library", " too."]

    async def script(p, at):
        end = await _say(p, at, first, wall_s=1.0, position_ms=100)
        p.push(H.delegation("d-one", end + 40))
        second_wall = 1.0 + len(first) * 0.12 + 1.2
        end = await _say(p, at, second, wall_s=second_wall, position_ms=end + 1200)
        p.push(H.delegation("d-two", end + 40))
        await at(second_wall + len(second) * 0.12 + 1.6)

    client, _provider, calls = await _call(monkeypatch, script)

    one, two = "".join(first).strip(), "".join(second).strip()
    assert _finals(client) == [one, two]
    assert [c["title"] for c in _created(client)] == [one, two]
    assert [c["display"] for c in calls] == [one, two]


@pytest.mark.asyncio
async def test_exact_replays_cannot_move_the_input_clock_or_duplicate_a_turn(monkeypatch):
    """A provider retry is not new speech.  Replays of the last fragment for
    1.2 s after the caller stopped must not postpone the endpoint, and replays
    or stale out-of-order fragments must not open a second turn."""

    words = ["Play", " something", " calm."]
    box: dict = {}

    async def script(p, at):
        position = 100  # lagging, as in production
        for index, word in enumerate(words):
            await at(1.0 + index * 0.12)
            box[word] = H.user_delta(word, position, position + 120)
            p.push(dict(box[word]))
            position += 120
        box["last_real"] = time.monotonic()
        for k in range(15):
            await at(1.0 + len(words) * 0.12 + 0.08 * (k + 1))
            p.push(dict(box[" calm."]))
            if k == 7:
                p.push(dict(box[" something"]))
                p.push(H.user_delta(" stale", 50, 90))
        await at(1.0 + len(words) * 0.12 + 1.2 + 0.8)

    client, _provider, _calls = await _call(monkeypatch, script)

    finals = [
        (f, t) for f, t in zip(client.frames, client.times)
        if f.get("type") == "transcript" and f.get("final")
    ]
    assert [f["text"] for f, _t in finals] == ["Play something calm."]
    closed_after = finals[0][1] - box["last_real"]
    # gap 400 ms + one 250 ms poll; had a replay re-anchored the clock the turn
    # could only close 400 ms after the replays stopped (≥ 1.6 s).
    assert closed_after < 1.1, closed_after
    assert not any("stale" in f.get("text", "") for f in client.of("transcript"))


@pytest.mark.asyncio
async def test_the_input_clock_is_separate_from_the_monotonic_wire_clock():
    """Unit form of §A: the silence clock re-anchors to the newest accepted
    user fragment with NO max() against the wire clock, which stays monotonic."""

    from app.config import settings
    from app.services.live_voice_protocol import (
        LiveDelegationTracker,
        UtteranceAssembler,
        _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.settings = settings
    session.tracker = LiveDelegationTracker()
    session.utterances = UtteranceAssembler("p-clock", gap_ms=400, hard_gap_ms=900)
    session.adapter = None
    session.settle_deadline = 0.0
    session.tick_wake = asyncio.Event()
    # Assistant output already pushed the wire clock 20 s ahead.
    session.provider_now_ms = 20000
    session.provider_now_at = time.monotonic()

    await session.on_input_transcript(H.user_delta("hello", 1000, 1200))

    assert session.provider_clock_ms() >= 20000
    heard = session.input_clock_ms()
    assert 1200 <= heard < 1700, heard
    assert not session.utterances.should_close_on_gap(heard)
    assert session.user_is_mid_utterance()
    # A replay is rejected before either clock is touched.
    anchor = session.input_now_at
    await session.on_input_transcript(H.user_delta("hello", 1000, 1200))
    assert session.input_now_at == anchor


# ── §A: an echo-only open turn neither holds output nor absorbs a request ──

AGENT_WORDS = (
    "the weather in toronto today is sunny with a light breeze and "
    "a high of twenty two degrees so it is a good day for a walk "
    "along the lake shore later this afternoon"
).split()
ECHO = {
    # every echoed delta carries a content word the agent just said
    "content": [
        " the weather in", " toronto today is", " sunny with a",
        " light breeze and", " high of twenty", " two degrees so",
    ],
    # …and the commonest shape: a delta of function words only, which is
    # evidence of nobody in particular (it used to make the echo "causal")
    "function_tail": [
        " the weather in", " toronto today is", " sunny with a",
        " light breeze and", " high of twenty", " it is a",
    ],
}
REQUEST = ["Search", " the", " Robarts", " library", " opening", " hours."]


async def _agent_speaks(p, at):
    """The agent talks from 0.2 s to 2.2 s: a word + an audio chunk every 50 ms."""

    for i in range(40):
        await at(0.2 + i * 0.05)
        position = 200 + i * 50
        p.push(H.out_text(" " + AGENT_WORDS[i % len(AGENT_WORDS)], position, position + 50))
        p.push(H.out_audio("AAAA"))


async def _echo_comes_back(p, at, echo):
    """Its own words come back through the hot mic, ~0.3–0.5 s behind."""

    for k, words in enumerate(ECHO[echo]):
        wall = 0.6 + k * 0.2
        await at(wall)
        position = int(wall * 1000) - 150
        p.push(H.user_delta(words, position, position + 150))


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", sorted(ECHO))
async def test_an_echo_only_open_turn_does_not_hold_the_agents_audio(monkeypatch, echo):
    """Before the fix EVERY output event was deferred while any user turn was
    open — including one made only of the agent's own echoed voice — so the
    agent went silent mid-sentence until the echo turn's endpoint."""

    async def script(p, at):
        await asyncio.gather(_agent_speaks(p, at), _echo_comes_back(p, at, echo))
        await at(3.5)

    client, _provider, _calls = await _call(monkeypatch, script)

    echo_frames = [
        index for index, f in enumerate(client.frames)
        if f.get("type") == "transcript" and f.get("text", "").startswith("the weather")
    ]
    assert echo_frames, "the echo is still transcribed as a user turn (R13.3.4)"
    first, final = echo_frames[0], echo_frames[-1]
    assert client.frames[final].get("final")
    between = [f for f in client.frames[first:final] if f.get("type") == "audio_delta"]
    assert between, "agent audio was held behind an echo-only user turn"
    # The agent's audio keeps its own 50 ms cadence the whole time it talks.
    audio = [t for f, t in zip(client.frames, client.times) if f.get("type") == "audio_delta"]
    assert len(audio) == 40
    longest = max(b - a for a, b in zip(audio, audio[1:]))
    assert longest < 0.45, f"agent audio stalled {longest:.2f}s behind its own echo"


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", sorted(ECHO))
async def test_the_callers_first_real_words_after_echo_start_a_new_turn(monkeypatch, echo):
    """Echo still open when the caller starts a real request used to be merged
    into it: the task's title was the agent's own words.  The first non-echo
    content fragment is a turn boundary; the echo is not part of the request."""

    async def voices(p, at):
        await _echo_comes_back(p, at, echo)
        end = await _say(p, at, REQUEST, wall_s=1.8, position_ms=1800)
        p.push(H.delegation("d-after-echo", end + 40))

    async def script(p, at):
        await asyncio.gather(_agent_speaks(p, at), voices(p, at))
        await at(4.5)

    client, _provider, calls = await _call(monkeypatch, script)

    request = "".join(REQUEST)
    finals = _finals(client)
    assert finals[-1] == request
    assert any(f.startswith("the weather in") for f in finals[:-1]), finals
    assert [c["title"] for c in _created(client)] == [request]
    assert [c["display"] for c in calls] == [request]


def test_a_non_causal_open_turn_is_never_a_direct_reply_parent():
    from app.services.live_voice_protocol import UtteranceAssembler, _LiveSession

    session = _LiveSession.__new__(_LiveSession)
    session.utterances = UtteranceAssembler("p-direct")
    _, echo = session.utterances.add(" the weather in", 500, 800)
    assert not echo.causal_speech  # nothing marked it as the caller's

    assert session.consume_direct_turn(900) == ""
    assert not echo.dispatched, "fenced as answered, a real request could never bind"

    echo.causal_speech = True
    assert session.consume_direct_turn(900) == echo.turn_id


@pytest.mark.asyncio
async def test_a_delegation_waiting_on_the_open_turn_is_not_expired(monkeypatch):
    """GPT-Live may decide mid-utterance.  `dispatch_pending` correctly waits for
    the open turn, but the TTL expired the entry on age alone while the caller
    was still talking, and the request was lost as `no_causal_turn`."""

    words = (
        "find me the email from my professor about the thesis draft and "
        "also check whether the meeting moved"
    ).split()
    spoken = [(" " if i else "") + w for i, w in enumerate(words)]

    async def script(p, at):
        position = 500
        for index, word in enumerate(spoken):
            await at(0.5 + index * 0.12)
            p.push(H.user_delta(word, position, position + 120))
            if index == 2:
                p.push(H.delegation("d-early", position + 120))
            position += 120
        await at(0.5 + len(spoken) * 0.12 + 1.5)

    client, _provider, calls = await _call(
        monkeypatch, script, clocks={"voice_live_delegation_ttl_s": 0.6},
    )

    phases = [(f["phase"], f.get("reason")) for f in client.of("delegation")]
    assert ("expired", "no_causal_turn") not in phases, phases
    sentence = " ".join(words)
    assert _finals(client) == [sentence]
    assert [c["display"] for c in calls] == [sentence]


# ── §A: structural unfinished tails (no content vocabulary) ──────────────

@pytest.mark.parametrize(
    "text",
    [
        "and...",
        "I was thinking…",
        "What the most, uh...",
        "Search for the professor,",
        "آهنگ رو،",
        "first this;",
        "the list is:",
        "Toronto -",
        "نمی‌",
    ],
)
def test_structural_tails_wait_for_the_hard_gap(text):
    from app.services.live_voice_protocol import UtteranceAssembler, looks_unfinished

    assert looks_unfinished(text)
    asm = UtteranceAssembler("p-tail", gap_ms=400, hard_gap_ms=900)
    asm.add(text, 0, 400)
    assert not asm.should_close_on_gap(400 + 400)
    assert asm.should_close_on_gap(400 + 900)


@pytest.mark.parametrize(
    "text",
    [
        "Search the Robarts library opening hours.",
        "What is the most famous building at U of T?",
        "چرا نمی‌تونی آهنگ رو عوض کنی؟",
        "ok",
        "Play something calm",
    ],
)
def test_finished_sentences_keep_the_ordinary_gap(text):
    from app.services.live_voice_protocol import looks_unfinished

    assert not looks_unfinished(text)


# ── §B: a delegation binds to the whole unanswered request span ──────────

@pytest.mark.asyncio
async def test_a_split_status_question_is_bound_whole_and_not_researched_again(monkeypatch):
    """Production V2+00:11: 'خب چی' (U:50) + 'شد' (U:51) — the delegation bound
    to 'شد', which was classified as a new request and ran a second 13.6 s
    research turn.  Here the two turns are split by a REAL pause (so the
    endpoint is right to split them); binding the unanswered span makes the
    classifier see 'خب چی شد' — a status question about the running task."""

    counters = H.capture_counters(monkeypatch)
    release = asyncio.Event()
    calls: list[str] = []

    async def think(_user_id, _task, _session_id, **kwargs):
        calls.append(kwargs.get("display_request"))
        await asyncio.wait_for(release.wait(), timeout=10)
        return "دفتر استاد در پردیس داون‌تاون است.", "test-model"

    async def script(p, at):
        end = await _say(
            p, at, ["آدرس", " دفتر", " استاد", " رو", " پیدا", " کن"],
            wall_s=0.5, position_ms=500,
        )
        p.push(H.delegation("d1", end + 40))
        for _ in range(100):
            if calls:
                break
            await asyncio.sleep(0.02)
        await at(2.5)
        p.push(H.user_delta("خب", 2500, 2650))
        await at(2.65)
        p.push(H.user_delta(" چی", 2650, 2800))
        # A real 0.9 s pause on both clocks: the endpoint closes 'خب چی'.
        await at(3.55)
        p.push(H.user_delta(" شد", 3550, 3700))
        p.push(H.delegation("d2", 3740))
        await at(5.0)
        release.set()
        await at(5.6)

    client, _provider, _calls = await _call(monkeypatch, script, think=think)

    assert _finals(client) == ["آدرس دفتر استاد رو پیدا کن", "خب چی", "شد"]
    assert calls == ["آدرس دفتر استاد رو پیدا کن"], "a status question re-ran the research"
    assert any(name == "live_status_answered" for name, _fields in counters)
    d1 = [f["phase"] for f in client.of("delegation") if f["delegation_id"] == "d1"]
    assert "superseded" not in d1 and "cancelled" not in d1


PIECES = [
    ("What", 500, 650), (" is the best", 1450, 1750), (" professor in UofT", 2550, 2950),
    (" who is working on", 3750, 4150), (" LLM", 4950, 5100),
]


@pytest.mark.asyncio
async def test_a_request_spoken_across_turns_is_one_task_and_every_turn_is_consumed(
    monkeypatch,
):
    """Five closed turns, each after a real (> gap, < span gap) pause and none
    answered: the one delegation carries all of them, keeps the ANCHOR turn as
    its causal parent, and consumes every span turn so a second delegation
    cannot run the same words again (at-most-once)."""

    async def script(p, at):
        for text, start, end in PIECES:
            await at(start / 1000.0)
            p.push(H.user_delta(text, start, end))
        p.push(H.delegation("d1", 5140))
        for _ in range(150):
            if _created(box["client"]):
                break
            await asyncio.sleep(0.02)
        # An exact replay of d1, and a second provider delegation whose offset
        # points into the first span turn.
        p.push(H.delegation("d1", 5140))
        p.push(H.delegation("d2", 600))
        await at(8.5)

    box: dict = {}
    client, _provider, calls = await _call(
        monkeypatch, script, clocks={"voice_live_delegation_ttl_s": 1.0}, box=box,
    )

    request = "What is the best professor in UofT who is working on LLM"
    finals = [f for f in client.of("transcript") if f.get("final")]
    assert [f["text"] for f in finals] == [text.strip() for text, _s, _e in PIECES]
    created = _created(client)
    assert [(c["delegation_id"], c["title"]) for c in created] == [("d1", request)]
    anchor = finals[-1]["turn_id"]
    assert created[0]["turn_id"] == anchor
    assert created[0]["parent_user_turn_id"] == anchor
    assert [c["display"] for c in calls] == [request]
    d2 = [(f["phase"], f.get("reason")) for f in client.of("delegation") if f["delegation_id"] == "d2"]
    assert d2 == [("expired", "no_causal_turn")]


@pytest.mark.asyncio
async def test_a_long_request_title_is_cut_at_a_word_boundary(monkeypatch):
    words = (
        "Find me the three most cited papers about retrieval augmented generation "
        "published by researchers at the University of Toronto this year"
    ).split()
    spoken = [(" " if i else "") + w for i, w in enumerate(words)]

    async def script(p, at):
        end = await _say(p, at, spoken, wall_s=0.5, position_ms=500, step_s=0.06, span_ms=60)
        p.push(H.delegation("d-long", end + 40))
        await at(0.5 + len(spoken) * 0.06 + 1.5)

    client, _provider, calls = await _call(monkeypatch, script)

    request = " ".join(words)
    assert [c["display"] for c in calls] == [request]
    title = _created(client)[0]["title"]
    assert len(title) <= 80 and title.endswith("…"), title
    kept = title[:-1]
    assert request.startswith(kept) and request[len(kept)] == " ", title


# ── §B units ──────────────────────────────────────────────────────────────

def _turn(ordinal, text, start, end, **extra):
    from app.services.live_voice_protocol import Utterance

    fields = dict(causal_speech=True, closed=True)
    fields.update(extra)
    return Utterance(
        turn_id=f"live-utt:p:{ordinal}", ordinal=ordinal,
        start_ms=start, end_ms=end, text=text, **fields,
    )


def test_request_span_walks_back_over_contiguous_unanswered_causal_turns():
    from app.services.live_voice_protocol import LiveDelegationTracker, UtteranceAssembler

    asm = UtteranceAssembler("p")
    answered = _turn(1, "hello", 0, 300, dispatched=True)
    a = _turn(2, "find the ر", 1000, 1400)
    b = _turn(3, "وبات", 2000, 2200)       # a word split ACROSS turns
    c = _turn(4, " lab at U of T", 3000, 3500)
    asm.closed_queue = [answered, a, b, c]

    span = asm.request_span(c, max_chars=600, max_gap_ms=8000)
    assert span == [a, b, c]

    tracker = LiveDelegationTracker()
    tracker.add_delegation("d1", 3600)
    task = tracker.consume("d1", c, span=span)
    assert task is not None
    assert task.transcript == "find the روبات lab at U of T"
    assert task.turn_id == c.turn_id
    assert all(u.consumed_by == "d1" and u.dispatched for u in span)
    assert not answered.consumed_by
    # Consumed turns are never bound again.
    assert asm.causal_candidate(1200) is None


def test_span_join_inserts_a_space_only_across_a_real_pause():
    """A new provider transcript item after a real pause may start without its
    own leading space; the span must not read "LLM.What".  A boundary closer
    than the utterance gap keeps the exact join (a split word rejoins)."""

    from app.services.live_voice_protocol import LiveDelegationTracker

    first = _turn(1, "Tell me about LLM.", 0, 900)
    second = _turn(2, "What is the best lab", 2400, 3200)   # 1.5 s real pause
    tracker = LiveDelegationTracker()
    tracker.add_delegation("d1", 3300)
    task = tracker.consume("d1", second, span=[first, second], join_gap_ms=1200)
    assert task is not None
    assert task.transcript == "Tell me about LLM. What is the best lab"

    close_a = _turn(3, "find the ر", 4000, 4400)
    close_b = _turn(4, "وبات lab", 4600, 4900)             # 200 ms: same word
    tracker.add_delegation("d2", 5000)
    task = tracker.consume("d2", close_b, span=[close_a, close_b], join_gap_ms=1200)
    assert task is not None
    assert task.transcript == "find the روبات lab"

    spaced_a = _turn(5, "play something ", 6000, 6400)
    spaced_b = _turn(6, "calm", 8000, 8300)                # already has a space
    tracker.add_delegation("d3", 8400)
    task = tracker.consume("d3", spaced_b, span=[spaced_a, spaced_b], join_gap_ms=1200)
    assert task is not None
    assert task.transcript == "play something calm"


@pytest.mark.parametrize(
    "blocker",
    [
        {"dispatched": True},          # answered by direct output
        {"consumed_by": "d0"},         # another delegation owns it
        {"causal_speech": False},      # echo / content-free
    ],
)
def test_request_span_stops_at_the_first_ineligible_turn(blocker):
    from app.services.live_voice_protocol import UtteranceAssembler

    asm = UtteranceAssembler("p")
    older = _turn(1, "older", 0, 300)
    stop = _turn(2, "blocked", 1000, 1300, **blocker)
    a = _turn(3, "what is", 2000, 2300)
    b = _turn(4, " the best", 3000, 3300)
    asm.closed_queue = [older, stop, a, b]
    assert asm.request_span(b, max_chars=600, max_gap_ms=8000) == [a, b]


def test_request_span_is_bounded_by_gap_and_size():
    from app.services.live_voice_protocol import UtteranceAssembler

    asm = UtteranceAssembler("p")
    far = _turn(1, "an hour ago", 0, 300)
    a = _turn(2, "x" * 50, 9000, 9300)
    b = _turn(3, " y" * 20, 10000, 10300)
    asm.closed_queue = [far, a, b]
    # 8.7 s of silence before `a` is beyond the span gap.
    assert asm.request_span(b, max_chars=600, max_gap_ms=8000) == [a, b]
    # The size bound never drops the anchor itself.
    assert asm.request_span(b, max_chars=60, max_gap_ms=8000) == [b]


def test_request_title_cuts_at_a_word_boundary():
    from app.services.live_voice_protocol import request_title

    short = "What is the best professor in UofT who is working on LLM"
    assert request_title(short) == short
    long = "word " * 30
    title = request_title(long)
    assert len(title) <= 80 and title.endswith("…")
    # "word"×16 is exactly 79 characters and ends where a word ends.
    assert title == ("word " * 16).rstrip() + "…"
    # No whitespace to cut at: a hard cut, still bounded and marked.
    unbroken = "x" * 200
    assert request_title(unbroken) == "x" * 79 + "…"
    assert request_title("  padded  ") == "padded"
