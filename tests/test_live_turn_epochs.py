"""PR-7: where a caller's turn ends, what happens to the output held behind
it, and where an output epoch ends (the 2026-10-02 call, P4 and P5).

What the call showed, in relay terms (FINDINGS §2 P4/P5):

* One sentence ("Everything is good. Can you search Who is" | "most
  experienced" | "officer in LLM in U of T") became three turns, and in each
  thinking pause GPT-Live's AUDIO-ONLY output — no transcript at all
  (`lang=-`) — was released after the turn closed and played into the
  caller's pause as a parentless epoch: "Speaking" while the caller was still
  finishing the question (F1, F3).
* Output held inside the interrupt fence was judged at RELEASE, after the
  fence had expired, so it played (F2).
* Held output never refreshed the output epoch's idle timer, so the relay
  retired an epoch while its own continuation sat in the held queue (P5).
* A parentless epoch was cut mid-sentence where its parent became known
  ("Hey! I'm doing" | "good, thanks.", F6a), an epoch was cut on 700 ms of
  stall with no regard for the sentence (F6b), and a tail the caller spoke a
  new request over was replayed after that request (F7).

Every scenario here drives the REAL relay against the fake GPT-Live harness
on scaled clocks (settle 40 / gap 400 / hard gap 900 ms, output gap 250 / hard
gap 800 ms: the production 350 / 1200 / 2600 and 700 / 2500 relationships),
and every fix has a twin that pins what must NOT change.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

import test_live_harness as H
from app.services import live_voice_protocol as live


CLOCKS = dict(
    voice_live_transcript_settle_ms=40,
    voice_live_utterance_gap_ms=400,
    voice_live_utterance_hard_gap_ms=900,
    voice_live_output_epoch_gap_ms=250,
    voice_live_output_epoch_hard_gap_ms=800,
    voice_live_delegation_ttl_s=3.0,
    voice_live_interrupt_suppress_ms=300,
)

#: `H.config()`'s list plus `output_attribution`, which the v0.3 app sends.
ATTRIBUTED = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    live.FEATURE_OUTPUT_ATTRIBUTION,
]


class TimedClient(H.FakeClient):
    """The phone, with the arrival instant of EVERY frame."""

    def __init__(self, script=None):
        super().__init__(script)
        self.times: list[float] = []

    async def send_json(self, frame):
        self.times.append(time.monotonic())
        await super().send_json(frame)


async def _run(
    monkeypatch, make_steps, *, clocks=None, think=None, features=None, timeout=30.0,
):
    """One relay call.  `make_steps(box, at)` returns the phone's script
    between `config` and `stop`: frames, sleeps, or zero-argument callables
    (awaited) that drive the provider through `box["provider"]` on wall time
    since `session.start` (`await at(seconds)`)."""

    H.fast_clocks(monkeypatch, **dict(CLOCKS, **(clocks or {})))
    calls: list[dict] = []

    async def default_think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, "display": kwargs.get("display_request")})
        return "An answer.", "test-model"

    recorded = H.patch_relay(monkeypatch, think=think or default_think)
    box: dict = {}

    def on_send(provider, event):
        if event["type"] == "session.start":
            box["provider"] = provider
            box["t0"] = asyncio.get_running_loop().time()
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def at(wall_s):
        while "t0" not in box:
            await asyncio.sleep(0.005)
        delay = box["t0"] + wall_s - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)

    config = H.config() if features is None else H.config(features=list(features))
    client = TimedClient([config, *make_steps(box, at), {"type": "stop"}])
    box["client"] = client
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=timeout,
        db_session_id=f"db-pr7-{time.monotonic_ns()}",
    )
    return client, provider, calls, recorded


async def _provider(box):
    while "provider" not in box:
        await asyncio.sleep(0.005)
    return box["provider"]


async def _say(p, at, words, *, wall_s, position_ms, step_s=0.12, span_ms=120):
    """Push `words` `step_s` apart on wall time and `span_ms` apart on the
    provider's input timeline; returns the input position after the last."""

    position = position_ms
    for index, word in enumerate(words):
        await at(wall_s + index * step_s)
        p.push(H.user_delta(word, position, position + span_ms))
        position += span_ms
    return position


async def _until(predicate, *, timeout=5.0, label="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.01)


def _finals(client):
    return [f for f in client.of("transcript") if f.get("final")]


def _final_count(box, n):
    return lambda: len(_finals(box["client"])) >= n


def _audio(client, prefix):
    return [f for f in client.of("audio_delta") if str(f.get("data") or "").startswith(prefix)]


def _texts(client, needle):
    return [f for f in client.of("response_text") if needle in str(f.get("text") or "")]


def _dropped(counters, reason):
    return [
        fields for name, fields in counters
        if name == "live_deferred_output_dropped" and fields.get("reason") == reason
    ]


def _stop_instructions(provider):
    return [
        e for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-interrupt-")
    ]


# ══════════════════════════════════════════════════════════════════════
# F1 — audio with no transcript is never played into the caller's pause
# ══════════════════════════════════════════════════════════════════════

SEG_A = ["Everything", " is", " good.", " Can", " you", " search", " Who", " is"]
SEG_B = [" most", " experienced"]
SEG_C = [" officer", " in", " LLM", " in", " U", " of", " T"]


@pytest.mark.asyncio
async def test_audio_only_output_held_in_a_thinking_pause_is_never_played(monkeypatch):
    """The call's three-segment sentence, with its real shape: each pause is
    longer than the hard gap (so each segment is its own turn, as on the
    call), and in each pause GPT-Live streams audio with no transcript.

    Old: the audio held behind each segment was released at the 1200 ms close
    as a parentless epoch — `state: speaking` and audio between the caller's
    own transcripts (epochs 4 and 5, `lang=-`).  New: audio that no
    transcript ever timed is dropped (`untimed_audio`), and the one delegation
    still spans all three turns."""

    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await _say(p, at, SEG_A, wall_s=0.5, position_ms=500)
            for k in range(11):                  # thinking pause 1: audio only
                await at(1.5 + k * 0.1)
                p.push(H.out_audio(f"PAUSE1-{k}"))
            await _say(p, at, SEG_B, wall_s=2.7, position_ms=2700)
            for k in range(11):                  # thinking pause 2: audio only
                await at(3.1 + k * 0.1)
                p.push(H.out_audio(f"PAUSE2-{k}"))
            end = await _say(p, at, SEG_C, wall_s=4.3, position_ms=4300)
            p.push(H.delegation("d-call", end + 40))
            await at(7.0)
        return [drive]

    client, _provider_, calls, _rec = await _run(monkeypatch, steps)

    finals = _finals(client)
    assert [f["text"] for f in finals] == [
        "Everything is good. Can you search Who is", "most experienced",
        "officer in LLM in U of T",
    ]
    assert not _audio(client, "PAUSE"), "untimed audio was played into a pause"
    first = client.frames.index(finals[0])
    last = client.frames.index(finals[-1])
    between = client.frames[first:last]
    assert not [f for f in between if f.get("type") == "audio_delta"]
    assert not [f for f in between if f.get("type") == "state" and f.get("state") == "speaking"]
    created = [f for f in client.of("delegation") if f.get("phase") == "created"]
    assert len(created) == 1
    assert created[0]["request_turn_ids"] == [f["turn_id"] for f in finals]
    request = "Everything is good. Can you search Who is most experienced officer in LLM in U of T"
    # Titled from the caller's words — since PR-10 (light P7) from the first
    # sentence that asks something, so without the leading "Everything is
    # good." (the job-1 title defect of the same call).
    display = "Can you search Who is most experienced officer in LLM in U of T"
    assert live.display_request_text(request) == display and request.endswith(display)
    assert created[0]["title"].rstrip("…") and display.startswith(created[0]["title"].rstrip("…"))
    assert [c["display"] for c in calls] == [display]
    assert _dropped(counters, "untimed_audio"), counters


@pytest.mark.asyncio
async def test_audio_whose_transcript_follows_within_the_gap_still_plays_parented(monkeypatch):
    """The positive twin (passes before and after the fix): audio that
    arrived just before the close and whose transcript lands just after it
    is NOT untimed — it is released, in order, parented to the turn."""

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            end = await _say(p, at, ["Play", " something", " calm."], wall_s=0.5, position_ms=500)
            await at(1.05)
            p.push(H.out_audio("REPLY-0"))           # the turn is still open
            await _until(_final_count(box, 1), label="the turn's close")
            p.push(H.out_text("Here is something calm.", end + 500, end + 1500))
            p.push(H.out_audio("REPLY-1"))
            await at(3.0)
        return [drive]

    client, _p, _calls, _rec = await _run(
        monkeypatch, steps, clocks={"voice_live_output_epoch_gap_ms": 600},
    )

    turn = _finals(client)[0]["turn_id"]
    text = _texts(client, "something calm")
    assert len(text) == 1 and text[0]["parent_user_turn_id"] == turn
    audio = _audio(client, "REPLY")
    assert [f["data"] for f in audio] == ["REPLY-0", "REPLY-1"]
    assert {f["response_id"] for f in audio} == {text[0]["assistant_turn_id"]}


@pytest.mark.asyncio
async def test_audio_leading_its_transcript_is_measured(monkeypatch):
    """F1 depends on audio leading its transcript by less than one output gap,
    so the lead is counted (`live_untimed_audio_timed lag_ms=`) wherever a
    transcript times held audio (fails before: no such counter)."""

    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            end = await _say(p, at, ["Play", " something", " calm."], wall_s=0.5, position_ms=500)
            await at(0.9)
            p.push(H.out_audio("REPLY-0"))           # the turn is still open
            await asyncio.sleep(0.15)
            p.push(H.out_text("Here is something calm.", end + 100, end + 1100))
            p.push(H.out_audio("REPLY-1"))
            await at(2.6)
        return [drive]

    client, _p, _calls, _rec = await _run(
        monkeypatch, steps, clocks={"voice_live_output_epoch_gap_ms": 600},
    )

    assert [f["data"] for f in _audio(client, "REPLY")] == ["REPLY-0", "REPLY-1"]
    lags = [f["lag_ms"] for n, f in counters if n == "live_untimed_audio_timed"]
    assert len(lags) == 1 and 100 <= lags[0] < 600, counters


@pytest.mark.asyncio
async def test_the_untimed_audio_hold_has_a_no_build_rollback(monkeypatch):
    """`voice_live_untimed_audio_hold_enabled = False` puts back the old
    release: audio no transcript timed plays parentless when the caller's
    turn closes, and nothing is dropped as untimed."""

    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await _say(p, at, ["Play", " something", " calm."], wall_s=0.5, position_ms=500)
            await at(0.9)
            p.push(H.out_audio("UNTIMED-0"))         # the turn is still open
            await at(2.5)
        return [drive]

    client, _p, _calls, _rec = await _run(
        monkeypatch, steps, clocks={"voice_live_untimed_audio_hold_enabled": False},
    )

    played = _audio(client, "UNTIMED")
    assert [f["data"] for f in played] == ["UNTIMED-0"]
    assert not _dropped(counters, "untimed_audio"), counters


# ══════════════════════════════════════════════════════════════════════
# F2 — the interrupt fence is judged when output ARRIVES, not at release
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_output_held_inside_the_interrupt_fence_is_not_released_after_it(monkeypatch):
    """The interrupted reply's late words arrive inside the 2.5 s fence while
    the caller is talking, so they are held behind the caller's turn — and
    released after that turn closed, by which time the fence had lifted on
    its deadline (02:28:50.843: "suppression lifted reason=deadline" in the
    same millisecond epoch 4 opened).  Judged at arrival they are fenced; a
    real reply that arrives after the fence still plays."""

    box_holder: dict = {}

    def steps(box, at):
        box_holder.update(box=box)

        async def ask():
            p = await _provider(box)
            await _say(p, at, ["What", " is", " the", " weather", " in", " Toronto?"],
                       wall_s=0.3, position_ms=300)
            await _until(_final_count(box, 1), label="the question's close")
            p.push(H.out_text("It is sunny and warm in Toronto today", 1500, 2600))
            p.push(H.out_audio("OLD-0"))
            await _until(lambda: _audio(box["client"], "OLD"), label="the reply playing")

        async def talk_over_it():
            p = box["provider"]
            loop = asyncio.get_running_loop()
            box["t_interrupt"] = loop.time()
            # The caller keeps talking; the model's already-generated tail of
            # the interrupted reply streams in INSIDE the fence (300 ms here).
            p.push(H.user_delta("Actually", 2700, 2950))
            await asyncio.sleep(0.05)
            p.push(H.user_delta(" wait", 2950, 3200))
            await asyncio.sleep(0.05)
            p.push(H.out_text(" and the high is twenty two", 2600, 2900))
            p.push(H.out_audio("FENCED"))
            # After the fence, while the caller's turn is still open: a real
            # reply to what they just said.
            await asyncio.sleep(0.35)
            p.push(H.out_text("Sure, go ahead.", 3300, 3900))
            p.push(H.out_audio("REPLY"))
            await asyncio.sleep(2.2)

        return [ask, {"type": "interrupt", "played_ms": 400}, talk_over_it]

    client, _p, _calls, _rec = await _run(monkeypatch, steps)

    assert not _texts(client, "twenty"), "a fenced tail played after the fence lifted"
    assert not _audio(client, "FENCED")
    reply = _texts(client, "Sure, go ahead.")
    assert len(reply) == 1
    assert reply[0]["parent_user_turn_id"] == _finals(client)[-1]["turn_id"]
    assert _audio(client, "REPLY")


# ══════════════════════════════════════════════════════════════════════
# F3a — a clause cannot end on a closed-class word
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "text",
    [
        "Everything is good. Can you search Who is",
        "Can you search who is",
        "where are",
        "the name of",
        "what was the score in",
        "is it on",
        "find the lab at",
        "the paper written by",
        "tell me about my",
        "what happened to their",
        "who is the most",
        "the food was very",
        "can you ask his",
        "I want the cheap one but",
        "could you check if it has",
        "we should be",
    ],
)
def test_copula_tail_waits_for_the_hard_gap(text):
    """Old: "…search Who is" closed on the ordinary 1200 ms gap because `is`
    was not a tail word, and the caller's next words became a second turn."""

    from app.services.live_voice_protocol import UtteranceAssembler, looks_unfinished

    assert looks_unfinished(text)
    asm = UtteranceAssembler("p-tail", gap_ms=400, hard_gap_ms=900)
    asm.add(text, 0, 400)
    assert not asm.should_close_on_gap(400 + 400)
    assert asm.should_close_on_gap(400 + 900)
    # A fragment inside the hard gap continues the same turn on the
    # provider timeline as well.
    rotated, current = asm.add(" next", 400 + 600, 400 + 700)
    assert rotated is None and current.text == text + " next"


@pytest.mark.parametrize("text", ["call her", "turn it on", "tell me more", "log in"])
def test_a_teardown_turn_ending_on_an_endpoint_only_tail_is_not_a_fragment(text):
    """The broader endpoint list is the ENDPOINT's: it only makes the relay
    wait longer before closing.  A turn the session end cut off is judged by
    R6-3's own closed classes, which deliberately keep object pronouns and
    phrasal particles complete — so "call her" said just before a hang-up is
    still a request, exactly as before."""

    from app.services.live_voice_protocol import looks_unfinished, teardown_fragment

    assert looks_unfinished(text)          # the endpoint waits for the hard gap
    assert not teardown_fragment(text)     # but the request is not discarded


# ══════════════════════════════════════════════════════════════════════
# F3b — the turn closes on the gap, the floor is given back only after the
# hard gap, and a caller who resumes inside it drops the premature reply.
# The window is OFF by default (`voice_live_phrase_window_enabled`): GPT-Live's
# ASR rarely punctuates, so it would delay nearly every direct reply.  These
# scenarios turn it on.
# ══════════════════════════════════════════════════════════════════════

WINDOW = {"voice_live_phrase_window_enabled": True}

@pytest.mark.asyncio
@pytest.mark.parametrize(("span_ms", "stops"), [(120, 0), (600, 1)])
async def test_caller_resuming_inside_the_hard_gap_drops_the_premature_reply(monkeypatch, span_ms, stops):
    """…and the model is told to stop only when the resumed words are
    SUSTAINED (D-P9: 3 × 600 ms is past the 1500 ms floor; 3 × 120 ms is not)."""

    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            end = await _say(p, at, ["Find", " me", " a", " good", " restaurant"],
                             wall_s=0.5, position_ms=500)
            # GPT-Live answers the half sentence while the turn is still open.
            await at(1.1)
            p.push(H.out_text("Sure, here are some restaurants nearby.", end + 50, end + 800))
            p.push(H.out_audio("EARLY"))
            await _until(_final_count(box, 1), label="the first turn's close")
            # The caller resumes 1.5x the gap after their last word: inside the
            # hard gap, after the relay already closed the turn.
            resume = end + 600
            end2 = await _say(p, at, [" in", " Kyoto", " tonight."], wall_s=0.0,
                              position_ms=resume, step_s=0.05, span_ms=span_ms)
            await asyncio.sleep(0.1)
            p.push(H.out_text("Here is one in Kyoto for tonight.", end2 + 200, end2 + 1200))
            p.push(H.out_audio("ANSWER"))
            await asyncio.sleep(2.0)
        return [drive]

    client, provider, _calls, _rec = await _run(monkeypatch, steps, clocks=WINDOW)

    finals = _finals(client)
    assert [f["text"] for f in finals] == ["Find me a good restaurant", "in Kyoto tonight."]
    assert not _texts(client, "restaurants nearby"), "the premature reply was played"
    assert not _audio(client, "EARLY")
    assert len(_stop_instructions(provider)) == stops
    answer = _texts(client, "in Kyoto for tonight")
    assert len(answer) == 1 and answer[0]["parent_user_turn_id"] == finals[-1]["turn_id"]
    assert _audio(client, "ANSWER")
    assert _dropped(counters, "caller_resumed"), counters


@pytest.mark.asyncio
async def test_a_premature_reply_still_streaming_when_the_caller_resumes_is_dropped_to_its_end(
    monkeypatch, caplog,
):
    """GPT-Live is mid-way through its reply to the half sentence when the
    caller resumes: what is held is dropped, and so are the words of that same
    reply still arriving after it (contiguous on its own timeline) — they must
    not become the answer to the caller's full sentence.  The real answer,
    after a pause, is the resumed turn's."""

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            end = await _say(p, at, ["Find", " me", " a", " good", " restaurant"],
                             wall_s=0.5, position_ms=500)
            await _until(_final_count(box, 1), label="the first turn's close")
            reply = "Sure here are some good restaurants nearby with great reviews".split()
            position = end + 300
            for i, word in enumerate(reply):
                if i == 3:
                    # The caller resumes, inside the hard gap.
                    await _say(p, at, [" in", " Kyoto", " tonight."], wall_s=0.0,
                               position_ms=end + 600, step_s=0.0)
                p.push(H.out_text(" " + word, position, position + 100))
                p.push(H.out_audio("EARLY"))
                position += 100
                await asyncio.sleep(0.04)
            await asyncio.sleep(1.2)
            p.push(H.out_text("Here is one in Kyoto for tonight.", position + 1500, position + 2500))
            p.push(H.out_audio("ANSWER"))
            await asyncio.sleep(2.0)
        return [drive]

    client, provider, _calls, _rec = await _run(monkeypatch, steps, clocks=WINDOW)

    finals = _finals(client)
    assert [f["text"] for f in finals] == ["Find me a good restaurant", "in Kyoto tonight."]
    assert not _audio(client, "EARLY"), "the premature reply's remainder was played"
    assert [f["text"] for f in client.of("response_text")] == ["Here is one in Kyoto for tonight."]
    assert client.of("response_text")[0]["parent_user_turn_id"] == finals[-1]["turn_id"]
    # D-P9 re-baseline: the caller resumed for well under
    # `voice_live_stop_instruction_min_speech_ms` — the dropped reply never
    # reached the phone, and a permanent stop note would only risk muting
    # the answer they are owed.
    assert _stop_instructions(provider) == []
    assert _dropped(counters, "caller_resumed"), counters
    # D-R1 (h), voice round 2026-10-06: the words of the dropped reply that
    # kept arriving were dropped with nothing but a counter; now one line per
    # dropped RUN (never one per delta) says so.
    lines = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("[LIVE] deferred output dropped reason=continues_dropped_run")
    ]
    assert len(lines) == 1, lines


@pytest.mark.asyncio
async def test_a_reply_to_an_unfinished_turn_is_released_after_the_hard_gap(monkeypatch):
    """Twin: nobody resumes, so the held reply plays — but only once the hard
    gap has passed without the caller's voice, never at the 1200 ms close."""

    marks: dict = {}

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            end = await _say(p, at, ["Find", " me", " a", " good", " restaurant"],
                             wall_s=0.5, position_ms=500)
            marks["last_word"] = time.monotonic()
            await at(1.1)
            p.push(H.out_text("Sure, here are some restaurants nearby.", end + 50, end + 800))
            p.push(H.out_audio("EARLY"))
            await at(3.0)
        return [drive]

    client, _p, _calls, _rec = await _run(monkeypatch, steps, clocks=WINDOW)

    final = _finals(client)[0]
    text = _texts(client, "restaurants nearby")
    assert len(text) == 1 and text[0]["parent_user_turn_id"] == final["turn_id"]
    released = client.times[client.frames.index(text[0])]
    closed = client.times[client.frames.index(final)]
    # Hard gap 900 ms (a little slack for the clock); the old release came at
    # the 400 ms close plus one tick.
    assert released - marks["last_word"] >= 0.85, "released before the hard gap"
    assert released > closed


@pytest.mark.asyncio
async def test_the_phrase_window_is_off_by_default(monkeypatch):
    """Control (passes before and after): with the shipped settings the held
    reply to an unpunctuated turn is released at the ordinary-gap close, as
    before PR-7.  The window costs about +1.4 s on most direct replies, so it
    stays off until the punctuation rate (`live_turn_gap_close`) and the
    arrival-jitter counter justify it; that rate is counted either way."""

    from app.config import Settings

    assert Settings.model_fields["voice_live_phrase_window_enabled"].default is False
    counters = H.capture_counters(monkeypatch)
    marks: dict = {}

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            end = await _say(p, at, ["Find", " me", " a", " good", " restaurant"],
                             wall_s=0.5, position_ms=500)
            marks["last_word"] = time.monotonic()
            await at(1.1)
            p.push(H.out_text("Sure, here are some restaurants nearby.", end + 50, end + 800))
            p.push(H.out_audio("EARLY"))
            await at(2.5)
        return [drive]

    client, _p, _calls, _rec = await _run(monkeypatch, steps)

    text = _texts(client, "restaurants nearby")
    assert len(text) == 1
    released = client.times[client.frames.index(text[0])]
    closed = client.times[client.frames.index(_finals(client)[0])]
    # Released in the very tick that closed the turn, not a hard gap later.
    assert released - closed < 0.25, "the default still waited for the hard gap"
    closes = [fields for name, fields in counters if name == "live_turn_gap_close"]
    assert closes == [{"phrase_end": "no"}], counters


@pytest.mark.asyncio
@pytest.mark.parametrize("resumed", [["Hmm."], ["um"], ["please"], ["um,", " please"]])
async def test_a_resumed_word_that_asks_nothing_never_drops_the_held_reply(monkeypatch, resumed):
    """Review of PR-7: any letter used to count as the caller resuming, so
    "Hmm.", "um" or "please" inside the hard gap dropped the held answer and
    told the model to stop.  GPT-Live counts that answer as said, so it was
    lost for good.  Words that ask nothing now leave the reply held behind
    their turn, and it plays once that turn closes (fails before)."""

    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            end = await _say(
                p, at, ["What", " is", " the", " weather", " in", " Toronto"],
                wall_s=0.5, position_ms=500,
            )
            await at(1.3)
            p.push(H.out_text("It is sunny in Toronto today.", end + 50, end + 900))
            p.push(H.out_audio("ANSWER"))
            await _until(_final_count(box, 1), label="the question's close")
            await _say(p, at, resumed, wall_s=0.0, position_ms=end + 600, step_s=0.05)
            await asyncio.sleep(2.0)
        return [drive]

    client, provider, _calls, _rec = await _run(monkeypatch, steps, clocks=WINDOW)

    finals = _finals(client)
    assert finals[0]["text"] == "What is the weather in Toronto"
    answer = _texts(client, "sunny in Toronto")
    assert len(answer) == 1, "the held answer was dropped"
    assert answer[0]["parent_user_turn_id"] == finals[0]["turn_id"]
    assert _audio(client, "ANSWER")
    assert _stop_instructions(provider) == []
    assert not _dropped(counters, "caller_resumed"), counters
    kept = [f for n, f in counters if n == "live_deferred_output_kept"]
    assert kept == [{"reason": "resumed_asks_nothing"}], counters


@pytest.mark.asyncio
async def test_a_filler_then_real_words_still_drops_the_premature_reply(monkeypatch):
    """Twin: the caller's resumed turn starts with "um" and then finishes the
    sentence.  The decision waits on the filler and is made on the real words:
    the premature reply is dropped once, as before the review fix."""

    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            end = await _say(p, at, ["Find", " me", " a", " good", " restaurant"],
                             wall_s=0.5, position_ms=500)
            await at(1.1)
            p.push(H.out_text("Sure, here are some restaurants nearby.", end + 50, end + 800))
            p.push(H.out_audio("EARLY"))
            await _until(_final_count(box, 1), label="the first turn's close")
            end2 = await _say(p, at, ["um", " in", " Kyoto", " tonight."], wall_s=0.0,
                              position_ms=end + 600, step_s=0.08)
            await asyncio.sleep(0.1)
            p.push(H.out_text("Here is one in Kyoto for tonight.", end2 + 200, end2 + 1200))
            p.push(H.out_audio("ANSWER"))
            await asyncio.sleep(2.0)
        return [drive]

    client, provider, _calls, _rec = await _run(monkeypatch, steps, clocks=WINDOW)

    finals = _finals(client)
    assert [f["text"] for f in finals] == ["Find me a good restaurant", "um in Kyoto tonight."]
    assert not _texts(client, "restaurants nearby"), "the premature reply was played"
    assert not _audio(client, "EARLY")
    # D-P9 re-baseline: the caller resumed for well under
    # `voice_live_stop_instruction_min_speech_ms` — the dropped reply never
    # reached the phone, and a permanent stop note would only risk muting
    # the answer they are owed.
    assert _stop_instructions(provider) == []
    answer = _texts(client, "in Kyoto for tonight")
    assert len(answer) == 1 and answer[0]["parent_user_turn_id"] == finals[-1]["turn_id"]
    assert len(_dropped(counters, "caller_resumed")) == 1, counters


# ══════════════════════════════════════════════════════════════════════
# F3c — a turn split by arrival jitter is measured
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_turn_split_by_arrival_jitter_is_logged(monkeypatch, caplog):
    """The silence that closes a turn is measured from when the last fragment
    ARRIVED.  When the next fragment then starts, on the provider's own
    timeline, closer than the gap to the closed turn's end, the split was made
    by transcription latency, not by the caller.  Logged and counted only."""

    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await at(0.3)
            p.push(H.user_delta("What is the best", 300, 700))
            # 0.9 s of wall silence closes the turn; the next words start
            # 200 ms after it on the provider timeline.
            await at(1.2)
            p.push(H.user_delta(" professor here", 900, 1300))
            await at(2.4)
        return [drive]

    with caplog.at_level(logging.INFO, logger=live.logger.name):
        client, _p, _calls, _rec = await _run(monkeypatch, steps)

    assert [f["text"] for f in _finals(client)] == ["What is the best", "professor here"]
    lines = [r.getMessage() for r in caplog.records if "split by arrival jitter" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "provider_gap_ms=200" in lines[0]
    assert "best" not in lines[0] and "professor" not in lines[0]
    assert [f for n, f in counters if n == "live_turn_split_jitter"]


# ══════════════════════════════════════════════════════════════════════
# P5 — held output and the output epoch's idle timer
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_held_continuation_released_after_a_backchannel_is_its_own_epoch(monkeypatch):
    """D-P7 R1 re-baseline (was "deferred continuation does not gap-retire the
    open epoch", P5): a backchannel over the reply held the reply's next
    sentence.  "Mm-hm." is the caller's turn and became final ABOVE that
    sentence on the phone, so the sentence is a NEW epoch below it — never
    grown into the bubble above (the 0:14 defect).  It is still the
    question's answer relay-side, but its wire names no parent: the phone
    already holds the question's answer, and draws this one after "Mm-hm.".
    The rows follow the epochs (D-P7 R2)."""

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await _say(p, at, ["When", " does", " the", " library", " open?"],
                       wall_s=0.3, position_ms=300)
            await _until(_final_count(box, 1), label="the question's close")
            await at(1.6)
            p.push(H.out_text("It opens at nine.", 1500, 1850))
            p.push(H.out_audio("A1"))
            await at(1.7)
            p.push(H.user_delta("Mm-hm.", 1700, 1950))
            await at(1.75)
            # The next sentence started while the caller was still saying
            # "Mm-hm." on the provider timeline: it is held behind their turn.
            p.push(H.out_text(" It closes at ten at night.", 1900, 2700))
            p.push(H.out_audio("A2"))
            await at(3.5)
        return [drive]

    # A 350 ms output gap: the held words arrive well inside it, and the old
    # relay's timer still fired (at ~1.95 s) before "Mm-hm." closed (~2.1 s).
    client, _p, _calls, _rec = await _run(
        monkeypatch, steps, clocks={"voice_live_output_epoch_gap_ms": 350},
    )

    question, backchannel = _finals(client)
    texts = client.of("response_text")
    assert [f["text"] for f in texts] == ["It opens at nine.", " It closes at ten at night."]
    first, tail = texts
    assert first["assistant_turn_id"] != tail["assistant_turn_id"], "a bubble grew above a final turn"
    assert client.frames.index(backchannel) < client.frames.index(tail)
    assert first["parent_user_turn_id"] == question["turn_id"]
    assert tail["parent_user_turn_id"] is None
    segments = client.of("speech_segment_complete")
    assert [s["text"] for s in segments][:1] == ["It opens at nine."]


def _reply_over_last_words_steps(box, at):
    """P5a's shape: an audio-only epoch is playing, the caller says "Hey,
    how's it going", and the reply's first words overlap their last ones."""

    async def drive():
        p = await _provider(box)
        # An audio-only epoch is playing when the caller starts (its last
        # chunk at 0.54 s keeps it open to 0.94 s on a 400 ms gap).
        for k, wall in enumerate((0.2, 0.3, 0.4, 0.5, 0.54)):
            await at(wall)
            p.push(H.out_audio(f"GREETING-{k}"))
        # "Hey, how's it going": long and wordy enough for one barge-in
        # shot, which the phone leaves unanswered.
        await at(0.55)
        p.push(H.user_delta("Hey,", 400, 650))
        await at(0.6)
        p.push(H.user_delta(" how's it", 650, 950))
        await at(0.65)
        p.push(H.user_delta(" going", 950, 1150))
        # The reply's first words overlap the caller's last ones: judged
        # the open (parentless) epoch's, and held behind the turn.
        await at(0.7)
        p.push(H.out_text("Hey! I'm doing", 1100, 1300))
        p.push(H.out_audio("REPLY-0"))
        # The next words begin after the caller's turn ended — and after
        # the old relay's idle timer had already retired the epoch.
        await at(1.05)
        p.push(H.out_text(" good, thanks.", 1350, 1700))
        p.push(H.out_audio("REPLY-1"))
        await at(3.2)
    return [drive]


_P5A_CLOCKS = {"voice_live_utterance_gap_ms": 600, "voice_live_output_epoch_gap_ms": 400}


@pytest.mark.asyncio
async def test_a_reply_that_started_over_the_callers_last_words_is_one_epoch(monkeypatch):
    """P5a: "Hey! I'm doing" | "good, thanks." — the reply's first words
    overlapped the caller's turn and were judged the parentless open epoch's;
    the next word began after the turn and was its answer, and the relay cut
    the sentence there.  For a client that negotiated `output_attribution` the
    parentless epoch now ADOPTS that turn relay-side (one epoch, one row,
    parented): its wire frames stay parentless for its whole life and the
    settled parent arrives as `output_attribution` (fails before)."""

    client, _p, _calls, recorded = await _run(
        monkeypatch, _reply_over_last_words_steps, features=ATTRIBUTED, clocks=_P5A_CLOCKS,
    )

    turn = _finals(client)[0]
    assert turn["text"] == "Hey, how's it going"
    texts = client.of("response_text")
    assert [f["text"] for f in texts] == ["Hey! I'm doing", " good, thanks."]
    assert len({f["assistant_turn_id"] for f in texts}) == 1, "the sentence was cut in two"
    # The wire never names a parent mid-epoch.
    assert all(f.get("parent_user_turn_id") is None for f in texts)
    epoch = texts[0]["epoch"]
    assert client.of("output_attribution") == [{
        "type": "output_attribution",
        "assistant_turn_id": texts[0]["assistant_turn_id"],
        "response_id": texts[0]["assistant_turn_id"],
        "epoch": epoch,
        "parent_user_turn_id": turn["turn_id"],
        "settled": True,
    }]
    rows = [s for s in recorded["saves"] if str(s.get("assistant_ref") or "").startswith("live-output:")]
    assert rows, recorded["saves"]
    assert rows[-1]["assistant_text"] == "Hey! I'm doing good, thanks."
    assert {r["assistant_ref"] for r in rows} == {f"live-output:{H.PSID}:{epoch}"}
    assert rows[-1]["assistant_voice"]["parent_user_turn_id"] == turn["turn_id"]


@pytest.mark.asyncio
async def test_a_client_without_output_attribution_still_sees_the_reply_parented(monkeypatch):
    """Review of PR-7: adoption must not change the wire for a client that
    did not negotiate `output_attribution`.  Such a build learns that its turn
    was answered only from an epoch naming it (`parent_user_turn_id`); an
    adopted epoch never names it, so its turn would look unanswered while the
    agent talks.  It keeps the old shape: the parentless epoch ends where the
    cause becomes known, the next epoch names the turn, and no
    `output_attribution` frame is sent."""

    client, _p, _calls, _recorded = await _run(
        monkeypatch, _reply_over_last_words_steps, clocks=_P5A_CLOCKS,
    )

    turn = _finals(client)[0]
    assert turn["text"] == "Hey, how's it going"
    texts = client.of("response_text")
    assert [f["text"] for f in texts] == ["Hey! I'm doing", " good, thanks."]
    assert texts[0]["assistant_turn_id"] != texts[1]["assistant_turn_id"]
    assert texts[0].get("parent_user_turn_id") is None
    assert texts[1]["parent_user_turn_id"] == turn["turn_id"]
    assert client.of("output_attribution") == []


def _unbroken_reply_over_last_words_steps(box, at):
    """P5a's shape with the reply UNBROKEN: its next words start exactly where
    "Hey! I'm doing" ended on the provider timeline, and arrive inside the
    output gap — one stream that never paused."""

    async def drive():
        p = await _provider(box)
        for k, wall in enumerate((0.2, 0.3, 0.4, 0.5, 0.54)):
            await at(wall)
            p.push(H.out_audio(f"GREETING-{k}"))
        await at(0.55)
        p.push(H.user_delta("Hey,", 400, 650))
        await at(0.6)
        p.push(H.user_delta(" how's it", 650, 950))
        await at(0.65)
        p.push(H.user_delta(" going", 950, 1150))
        await at(0.7)
        p.push(H.out_text("Hey! I'm doing", 1100, 1300))
        p.push(H.out_audio("REPLY-0"))
        await at(0.95)
        p.push(H.out_text(" good, thanks.", 1300, 1650))
        p.push(H.out_audio("REPLY-1"))
        await at(3.2)
    return [drive]


@pytest.mark.asyncio
async def test_an_unbroken_reply_keeps_the_legacy_wire_without_output_attribution(monkeypatch):
    """Integration gate on F6a (contract: parent adoption must not change the
    wire for a client that did not negotiate `output_attribution`).  The reply
    that began over the caller's last words never paused, and f49c49a6 kept it
    ONE parentless epoch (the unbroken-run rule).  Before the gate, F6a's mark
    made its next words the turn's answer for such a client too — a second
    epoch naming the turn.  Now the old wire stands
    (`legacy_judges_turn_answer`)."""

    client, _p, _calls, _recorded = await _run(
        monkeypatch, _unbroken_reply_over_last_words_steps, clocks=_P5A_CLOCKS,
    )

    texts = client.of("response_text")
    assert [f["text"] for f in texts] == ["Hey! I'm doing", " good, thanks."]
    assert len({f["assistant_turn_id"] for f in texts}) == 1, texts
    assert all(f.get("parent_user_turn_id") is None for f in texts), texts
    assert client.of("output_attribution") == []


@pytest.mark.asyncio
async def test_an_unbroken_reply_is_still_adopted_for_output_attribution(monkeypatch):
    """The twin: a client that negotiated `output_attribution` keeps F6a — one
    epoch, parentless on the wire, and the settled parent sent as
    `output_attribution`."""

    client, _p, _calls, _recorded = await _run(
        monkeypatch, _unbroken_reply_over_last_words_steps, features=ATTRIBUTED,
        clocks=_P5A_CLOCKS,
    )

    turn = _finals(client)[0]
    texts = client.of("response_text")
    assert [f["text"] for f in texts] == ["Hey! I'm doing", " good, thanks."]
    assert len({f["assistant_turn_id"] for f in texts}) == 1, texts
    assert all(f.get("parent_user_turn_id") is None for f in texts)
    attribution = client.of("output_attribution")
    assert [a["parent_user_turn_id"] for a in attribution] == [turn["turn_id"]], attribution
    assert attribution[0]["settled"] is True


def test_an_epoch_without_a_sentence_end_survives_a_short_stall():
    """F6b, adapter unit, production numbers (700 ms gap, 2500 ms hard gap):
    "…the LLM" + 1.0 s stall + " program." is one epoch; a sentence that
    ended keeps the 700 ms gap."""

    adapter = live.LiveClientAdapter("s-f6b", gap_ms=700, hard_gap_ms=2500)
    adapter.provider_event(H.out_text("Here is what I found about the LLM", 0, 900), now=10.0)
    first = adapter.output_id
    frames, closed = adapter.rotate_on_gap(now=11.0)
    assert closed is None and frames == []
    adapter.provider_event(H.out_text(" program.", 1900, 2300), now=11.0)
    assert adapter.output_id == first
    frames, closed = adapter.rotate_on_gap(now=11.8)
    assert closed is not None and closed.output_id == first
    assert closed.text == "Here is what I found about the LLM program."

    # The control: a finished sentence still closes on the ordinary gap.
    adapter.provider_event(H.out_text("It is finished.", 3000, 3500), now=20.0)
    second = adapter.output_id
    frames, closed = adapter.rotate_on_gap(now=21.0)
    assert closed is not None and closed.output_id == second
    adapter.provider_event(H.out_text(" Anything else?", 4500, 5000), now=21.0)
    assert adapter.output_id != second

    # …and past the hard gap even an unfinished sentence closes.
    adapter.provider_event(H.out_text("And then the", 6000, 6400), now=30.0)
    third = adapter.output_id
    assert adapter.rotate_on_gap(now=32.0)[1] is None
    closed = adapter.rotate_on_gap(now=32.6)[1]
    assert closed is not None and closed.output_id == third


def test_a_spoken_over_epoch_logs_its_audio_and_text_counts(caplog):
    """Integration of PR-1 and PR-7: every `output epoch closed` line carries
    PR-1's counts (audio deltas, transcript characters), and PR-7's new
    `spoken_over` retirement is one of them."""

    adapter = live.LiveClientAdapter("s-f7-counts", gap_ms=700, hard_gap_ms=2500)
    adapter.provider_event(H.out_text("It opens at nine", 0, 600), now=10.0)
    adapter.provider_event(H.out_audio("A1"), now=10.0)
    adapter.provider_event(H.out_audio("A2"), now=10.1)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _frames, closed = adapter.retire_spoken_over(now=10.2)
    assert closed is not None and closed.retire_reason == "spoken_over"
    lines = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("[LIVE] output epoch closed ")
    ]
    assert len(lines) == 1 and "reason=spoken_over" in lines[0], lines
    assert lines[0].endswith(" audio_chunks=2 text_chars=16"), lines[0]


def test_an_adapter_built_without_a_hard_gap_keeps_the_ordinary_gap():
    """Unit callers that never name the new gap keep today's behaviour."""

    adapter = live.LiveClientAdapter("s-f6b-default", gap_ms=700)
    adapter.provider_event(H.out_text("Here is what I found about the", 0, 900), now=10.0)
    assert adapter.rotate_on_gap(now=10.8)[1] is not None


@pytest.mark.asyncio
async def test_a_mid_sentence_epochs_successor_is_folded_onto_its_row(monkeypatch):
    """F6b, the record half: a stall longer than the hard gap still splits
    the epoch (the phone is told the first one ended), but with no caller
    turn in between, the successor's words are written onto the SAME row
    (revision bumped), so the day chat reads one sentence ("…LLM" | "program."
    were two rows on the call)."""

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await _say(p, at, ["Tell", " me", " about", " the", " program."],
                       wall_s=0.3, position_ms=300)
            await _until(_final_count(box, 1), label="the question's close")
            await at(1.5)
            p.push(H.out_text("The program covers the LLM", 1500, 2200))
            p.push(H.out_audio("P1"))
            # A stall longer than the hard gap (800 ms here).
            await at(2.6)
            p.push(H.out_text(" program in depth.", 3300, 3900))
            p.push(H.out_audio("P2"))
            await at(4.6)
        return [drive]

    client, _p, _calls, recorded = await _run(monkeypatch, steps)

    texts = client.of("response_text")
    assert len({f["assistant_turn_id"] for f in texts}) == 2, "a stall past the hard gap still splits"
    first_epoch = texts[0]["epoch"]
    rows = [s for s in recorded["saves"] if str(s.get("assistant_ref") or "").startswith("live-output:")]
    assert {r["assistant_ref"] for r in rows} == {f"live-output:{H.PSID}:{first_epoch}"}, rows
    assert rows[-1]["assistant_text"] == "The program covers the LLM program in depth."
    revisions = [r["assistant_revision"] for r in rows]
    assert revisions == sorted(revisions) and len(set(revisions)) == len(revisions)


@pytest.mark.asyncio
async def test_a_mid_sentence_epoch_is_not_folded_across_a_caller_turn(monkeypatch):
    """Twin: the caller spoke between the two epochs — the second is not the
    first one's sentence any more, and keeps its own row."""

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await _say(p, at, ["Tell", " me", " about", " the", " program."],
                       wall_s=0.3, position_ms=300)
            await _until(_final_count(box, 1), label="the question's close")
            await at(1.5)
            p.push(H.out_text("The program covers the LLM", 1500, 2200))
            p.push(H.out_audio("P1"))
            await at(2.5)
            await _say(p, at, ["And", " the", " cost?"], wall_s=2.5, position_ms=2500)
            await _until(_final_count(box, 2), label="the second question's close")
            p.push(H.out_text("It costs nothing.", 3600, 4100))
            p.push(H.out_audio("P2"))
            await asyncio.sleep(1.5)
        return [drive]

    client, _p, _calls, recorded = await _run(monkeypatch, steps)

    texts = client.of("response_text")
    epochs = sorted({f["epoch"] for f in texts})
    assert len(epochs) == 2
    rows = [s for s in recorded["saves"] if str(s.get("assistant_ref") or "").startswith("live-output:")]
    assert {r["assistant_ref"] for r in rows} == {f"live-output:{H.PSID}:{e}" for e in epochs}


@pytest.mark.asyncio
async def test_a_mid_sentence_epoch_is_not_folded_across_a_long_silence(monkeypatch):
    """Review of PR-7: the fold had no time bound, so words born any time
    later with no caller turn between (a muted caller, a withdrawn relay
    line) were written onto the old row.  Past `_FOLD_MAX_GAP_MS` on the
    provider's output timeline they keep a row of their own (fails before)."""

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await _say(p, at, ["Tell", " me", " about", " the", " program."],
                       wall_s=0.3, position_ms=300)
            await _until(_final_count(box, 1), label="the question's close")
            await at(1.5)
            p.push(H.out_text("The program covers the LLM", 1500, 2200))
            p.push(H.out_audio("P1"))
            await at(2.6)
            gap = 9000                       # > `_FOLD_MAX_GAP_MS` (8000)
            p.push(H.out_text(" that real quick.", 2200 + gap, 2700 + gap))
            p.push(H.out_audio("P2"))
            await at(4.6)
        return [drive]

    client, _p, _calls, recorded = await _run(monkeypatch, steps)

    texts = client.of("response_text")
    epochs = sorted({f["epoch"] for f in texts})
    assert len(epochs) == 2
    rows = [s for s in recorded["saves"] if str(s.get("assistant_ref") or "").startswith("live-output:")]
    assert {r["assistant_ref"] for r in rows} == {f"live-output:{H.PSID}:{e}" for e in epochs}, rows
    assert live._FOLD_MAX_GAP_MS < 9000


# ══════════════════════════════════════════════════════════════════════
# F7 — a tail the caller spoke a new request over is not replayed after it
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("words,keeps_tail", [
    (["What", " happened"], False),
    (["Mm-hm."], True),
    # A bare assent is no request of its own here (`asks_nothing` keeps it out
    # of the phatic class because it may consent to a proposal; over a reply
    # that is still talking it is a backchannel).
    (["Yeah."], True),
])
async def test_a_tail_spoken_over_by_a_new_request_is_not_replayed_after_it(
    monkeypatch, words, keeps_tail,
):
    """P5d: epoch 10 was saying "…That didn't go through." when the caller
    asked "What happened" (two words: under the barge-in floor).  The rest of
    the sentence was held behind that turn and played AFTER it, as a
    parentless epoch, ahead of the answer.  A real request now drops the held
    tail and retires the epoch; a backchannel ("Mm-hm.") is no request, and
    the tail plays on — D-P7 R1 re-baseline: in a NEW epoch after the
    backchannel's final, so the phone never grows a bubble above it."""

    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await at(0.3)
            p.push(H.out_text("That didn't go through.", 300, 1300))
            for k, wall in enumerate((0.3, 0.4, 0.5, 0.6)):
                await at(wall)
                p.push(H.out_audio(f"HEAD-{k}"))
            position = 600
            for index, word in enumerate(words):
                await at(0.61 + index * 0.09)
                span = 500 // len(words)
                p.push(H.user_delta(word, position, position + span))
                position += span
                if index == 0:
                    # GPT-Live's audio leads its transcript (the order the
                    # relay's held-audio repair assumes): the sentence goes on
                    # while the caller talks — then, below, the answer.
                    await at(0.65)
                    p.push(H.out_audio("TAIL"))
                    p.push(H.out_text(" Ask me again and I'll retry.", 1050, 1900))
            if not keeps_tail:
                await at(0.9)
                p.push(H.out_audio("ANSWER"))
                p.push(H.out_text("It failed because the agent was unreachable.", 2000, 3000))
            await at(3.0)
        return [drive]

    client, provider, _calls, _rec = await _run(monkeypatch, steps)

    final = _finals(client)[0]
    head = _texts(client, "didn't go through")
    assert len(head) == 1
    tail = _texts(client, "retry")
    if keeps_tail:
        assert len(tail) == 1, "a backchannel must not cost the caller the sentence"
        assert tail[0]["assistant_turn_id"] != head[0]["assistant_turn_id"]
        assert client.frames.index(tail[0]) > client.frames.index(final)
        assert not _dropped(counters, "spoken_over")
    else:
        assert tail == [], "the tail the caller spoke over was replayed after them"
        assert not _audio(client, "TAIL")
        answer = _texts(client, "unreachable")
        assert len(answer) == 1 and answer[0]["parent_user_turn_id"] == final["turn_id"]
        assert answer[0]["assistant_turn_id"] != head[0]["assistant_turn_id"]
        assert _dropped(counters, "spoken_over"), counters
        # The model had already moved on to the answer: telling it to stop
        # now would cut THAT off.
        assert _stop_instructions(provider) == []


@pytest.mark.asyncio
async def test_a_spoken_over_tail_that_ended_is_dropped_without_stopping_the_model(monkeypatch):
    """The held tail ended before the caller's turn closed and nothing
    followed it: it is dropped all the same — and the model is NOT told to
    stop, because it is no longer saying anything, and the instruction would
    only risk muting the answer the caller just asked for."""

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await at(0.3)
            p.push(H.out_text("That didn't go through.", 300, 1300))
            for k, wall in enumerate((0.3, 0.4, 0.5, 0.6)):
                await at(wall)
                p.push(H.out_audio(f"HEAD-{k}"))
            await at(0.61)
            p.push(H.user_delta("What", 600, 850))
            await at(0.65)
            p.push(H.out_text(" Ask me again and I'll retry.", 1050, 1900))
            p.push(H.out_audio("TAIL"))
            await at(0.7)
            p.push(H.user_delta(" happened", 850, 1100))
            await at(2.6)
        return [drive]

    client, provider, _calls, _rec = await _run(monkeypatch, steps)

    assert _texts(client, "retry") == []
    assert not _audio(client, "TAIL")
    assert _stop_instructions(provider) == []


@pytest.mark.asyncio
async def test_a_tail_still_streaming_when_the_request_closes_is_never_given_to_it(monkeypatch):
    """The boundary of F7: when the old sentence is STILL arriving as the
    caller's turn closes, dropping what is held would leave the rest to arrive
    with no epoch open and be read as the answer to the caller's question (F0
    Sequence C).  It is left exactly as before: the old sentence keeps its own
    parent, and only the real answer is the question's."""

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            words = ("the library opens at nine in the morning on weekdays and it closes "
                     "at ten at night except on holidays when hours are shorter").split()
            for i, word in enumerate(words):
                await at(0.2 + i * 0.08)
                position = 200 + i * 80
                p.push(H.out_text(" " + word, position, position + 80))
                p.push(H.out_audio("OLD"))
                if i == 8:
                    p.push(H.user_delta("What about Sunday", 900, 1200))
            await at(3.0)
            p.push(H.out_text("On Sunday it opens at noon.", 3000, 3400))
            p.push(H.out_audio("ANSWER"))
            await at(4.0)
        return [drive]

    client, _p, _calls, _rec = await _run(monkeypatch, steps)

    question = _finals(client)[0]["turn_id"]
    answer = _texts(client, "On Sunday")
    assert len(answer) == 1 and answer[0]["parent_user_turn_id"] == question
    old = [f for f in client.of("response_text") if "On Sunday" not in f["text"]]
    assert all(f.get("parent_user_turn_id") != question for f in old), old
    assert "".join(f["text"] for f in old).count("shorter") == 1


# ══════════════════════════════════════════════════════════════════════
# D-P7 R1 (2026-10-05) — a caller turn that became final ends the epoch
# ══════════════════════════════════════════════════════════════════════

def _reply_then_turn_steps(turn_words, *, turn_at_ms=2600):
    """The question, reply A (left open: the output gap is long), a caller
    turn while A is still open, then the model's next words — inside A's
    gap, the window in which they used to join A's epoch."""

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await _say(p, at, ["When", " does", " it", " open?"], wall_s=0.3, position_ms=300)
            await _until(_final_count(box, 1), label="the question's close")
            p.push(H.out_text("It opens at nine on weekdays.", 1500, 2400))
            p.push(H.out_audio("A1"))
            await asyncio.sleep(0.1)
            for index, word in enumerate(turn_words):
                p.push(H.user_delta(word, turn_at_ms + index * 200, turn_at_ms + (index + 1) * 200))
                await asyncio.sleep(0.05)
            await _until(_final_count(box, 2), label="the caller turn's close")
            p.push(H.out_text(" And at noon on Sundays.", 3600, 4400))
            p.push(H.out_audio("A2"))
            await asyncio.sleep(1.0)
        return [drive]
    return steps


_R1_CLOCKS = {"voice_live_output_epoch_gap_ms": 3000, "voice_live_output_epoch_hard_gap_ms": 3000}


@pytest.mark.asyncio
async def test_a_reply_after_a_closed_caller_turn_opens_a_new_epoch_inside_the_gap(monkeypatch, caplog):
    """D-P7 R1, the 0:14 defect: the reply to «عالی ام» joined the agent's
    previous bubble ABOVE that turn, because its first delta arrived inside
    the open epoch's gap (probe 3: `live:live-psid:1` twice, no parent).  A
    caller turn that became final retires the open epoch before the next
    delta: the words after it are a new epoch, and answer that turn."""

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    client, _p, _calls, recorded = await _run(
        monkeypatch, _reply_then_turn_steps(["great", " thanks"]), clocks=_R1_CLOCKS,
    )

    question, thanks = _finals(client)
    texts = client.of("response_text")
    assert [f["text"] for f in texts] == ["It opens at nine on weekdays.", " And at noon on Sundays."]
    first, second = texts
    assert first["assistant_turn_id"] != second["assistant_turn_id"]
    assert first["parent_user_turn_id"] == question["turn_id"]
    assert second["parent_user_turn_id"] == thanks["turn_id"]
    # The first epoch ends on the wire before the second begins.
    ended = next(
        f for f in client.of("speech_segment_complete") if f["response_id"] == first["assistant_turn_id"]
    )
    assert client.frames.index(thanks) < client.frames.index(ended) < client.frames.index(second)
    assert any(
        "output epoch closed" in r.getMessage() and "reason=caller_turn_closed" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_an_echo_only_turn_never_ends_the_epoch(monkeypatch):
    """The twin: the agent's own words coming back through the mic close an
    echo-only turn — not the caller's, so the reply keeps its epoch."""

    client, _p, _calls, _rec = await _run(
        monkeypatch,
        _reply_then_turn_steps([" it opens at", " nine on weekdays"], turn_at_ms=1700),
        clocks=_R1_CLOCKS,
    )

    echo = _finals(client)[1]
    assert echo.get("echo_only") is True, echo
    texts = client.of("response_text")
    assert [f["text"] for f in texts] == ["It opens at nine on weekdays.", " And at noon on Sundays."]
    assert texts[0]["assistant_turn_id"] == texts[1]["assistant_turn_id"]


# ── D-R1 (h), voice round 2026-10-06: no drop path is silent ─────────


@pytest.mark.asyncio
async def test_held_output_over_its_byte_bound_is_dropped_with_a_line(monkeypatch, caplog):
    """The fail-closed bound of the held-output buffer threw the whole held
    reply away with only a counter."""

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    monkeypatch.setattr(live, "_DEFERRED_OUTPUT_MAX_BYTES", 40)
    counters = H.capture_counters(monkeypatch)

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await at(0.3)
            p.push(H.user_delta("tell me about", 500, 900))
            await asyncio.sleep(0.1)
            # Output while the caller's turn is still open is HELD behind it.
            p.push(H.out_text("Sure, here is a long premature answer.", 950, 1300))
            p.push(H.out_text(" And it keeps going past the bound.", 1300, 1700))
            await asyncio.sleep(1.5)
        return [drive]

    await _run(monkeypatch, steps)

    lines = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("[LIVE] deferred output dropped reason=capacity")
    ]
    assert len(lines) == 1 and " events=" in lines[0], lines
    assert _dropped(counters, "capacity"), counters


@pytest.mark.asyncio
async def test_a_refused_output_transcript_is_logged_once_per_epoch(monkeypatch, caplog):
    """An output transcript the tracker refuses (it ends before text already
    accepted, or repeats a fragment) was dropped with no trace at all."""

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    def steps(box, at):
        async def drive():
            p = await _provider(box)
            await at(0.2)
            p.push(H.out_text("Hello there, how can I help?", 1000, 2000))
            p.push(H.out_audio())
            await asyncio.sleep(0.05)
            p.push(H.out_text(" stale", 500, 900))
            p.push(H.out_text(" staler", 400, 800))
            await asyncio.sleep(0.6)
        return [drive]

    client, _provider_obj, _calls, _rec = await _run(monkeypatch, steps)

    lines = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("[LIVE] deferred output dropped reason=transcript_refused")
    ]
    assert len(lines) == 1, lines
    assert not [f for f in client.of("response_text") if "stale" in str(f.get("text") or "")]


# ── Voice round 2026-10-07 (D-R3(5)): one sentence across ONE request ──────
# The 12:04 call: the model's «باشه، حتماً، دارم تو وب می‌گردم.» was held behind
# the open «کیه» turn and released as two deltas stamped with their own
# definitive parents — «باشه» → the request turn, the rest → «کیه».  The second
# parent rotated the epoch («باشه», no audio, `causal_user_turn`) and the thread
# got two rows.  Both turns are ONE request once «کیه» joined it (Layer A); a
# piece cut anyway is carried into the next row, never written alone (Layer B,
# `carry_spoken_fragment` at `persist_spoken`'s slot).

import app.services.live_voice_protocol as _live  # noqa: E402
from app.config import settings as _settings  # noqa: E402

_PROF = "قوی‌ترین استاد یوافتی توی بحث ال‌ال‌ام رو پیدا کن"
_ACK_HEAD, _ACK_TAIL = "باشه", "، حتماً، دارم تو وب می‌گردم."


async def _released_ack_call(monkeypatch, *, join: bool, carry: float = 0.0):
    """d1 runs the request; «کیه» follows 1.4 s later; then the two released
    ack deltas arrive stamped with their recorded parents."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(_settings, "voice_live_addendum_window_ms", 5000 if join else 0)
    monkeypatch.setattr(_settings, "voice_live_fragment_carry_s", carry)
    calls: list = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        await asyncio.sleep(1.5)
        return "کالین رافل.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    def finals():
        return {f["text"]: f["turn_id"] for f in client.of("transcript") if f.get("final")}

    async def script():
        p = box["p"]
        p.push(H.user_delta(_PROF, 1000, 2500))
        p.push(H.delegation("d1", 2550))
        for _ in range(150):
            if calls:
                break
            await asyncio.sleep(0.02)
        p.push(H.user_delta("کیه", 3900, 4100))
        for _ in range(200):
            if "کیه" in finals():
                break
            await asyncio.sleep(0.02)
        request, tail = finals()[_PROF], finals()["کیه"]
        p.push(dict(H.out_text(_ACK_HEAD, 4200, 4280),
                    _toup_parent_user_turn_id=request, _toup_parent_resolved=True))
        p.push(dict(H.out_text(_ACK_TAIL, 4300, 4900),
                    _toup_parent_user_turn_id=tail, _toup_parent_resolved=True))
        p.push(H.out_audio("SPEAKING"))
        await asyncio.sleep(2.0)

    client = H.FakeClient([H.config(), 0.1, script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=15,
                      db_session_id=f"db-r3-ack-{int(join)}-{carry}")
    rows = [s["assistant_text"] for s in recorded["saves"]
            if str(s.get("assistant_ref") or "").startswith("live-output:")
            and s.get("assistant_text")]
    return client, rows


@pytest.mark.asyncio
async def test_one_sentence_across_one_request_is_one_epoch_and_one_row(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    # Control: «کیه» is NOT the request's (no join) — the recorded split.
    client, rows = await _released_ack_call(monkeypatch, join=False)
    assert rows == [_ACK_HEAD, _ACK_TAIL]
    caplog.clear()
    client, rows = await _released_ack_call(monkeypatch, join=True)
    assert rows == [_ACK_HEAD + _ACK_TAIL]
    opened = [r.getMessage() for r in caplog.records if "[LIVE] output epoch opened" in r.getMessage()]
    assert len(opened) == 1, opened
    assert any(r.getMessage().startswith("[LIVE] output epoch kept across one request epoch=1 ")
               for r in caplog.records)


@pytest.mark.asyncio
async def test_a_cut_head_rides_into_the_next_row_and_the_wire_marks_it(monkeypatch):
    """Layer B end to end, through the REAL `persist_spoken` (its order: bind
    → media fold → fragment carry, unbound only → own row): the rotated
    «باشه» is held and rides into the next row; the live wire marks its
    segment `fragment: true`."""

    client, rows = await _released_ack_call(monkeypatch, join=False, carry=3.0)
    assert rows == [_ACK_HEAD + _ACK_TAIL], rows
    segments = {f["epoch"]: f for f in client.of("speech_segment_complete")}
    assert segments[1].get("fragment") is True and segments[1]["text"] == _ACK_HEAD
    assert "fragment" not in segments[2]


def test_layer_a_keeps_only_an_unfinished_sentence_of_one_request():
    def session_with(text, *, second_owner="d1", addendum=False):
        session = _live._LiveSession.__new__(_live._LiveSession)
        session.utterances = _live.UtteranceAssembler("p-a", gap_ms=400, hard_gap_ms=900)
        first = session.utterances
        first.add(_PROF, 1000, 2500)
        u1 = first.close("gap")
        first.add("کیه", 3900, 4100)
        u2 = first.close("gap")
        u1.consumed_by = "d1"
        u2.consumed_by = None if addendum else second_owner
        session.addendum_turns = {u2.turn_id: "d1"} if addendum else {}
        adapter = _live.LiveClientAdapter("p-a")
        adapter._epoch, adapter._output_id = 1, "live:p-a:1"
        adapter._output_text, adapter._parent_user_turn_id = text, u1.turn_id
        session.adapter = adapter
        return session, u2.turn_id

    session, tail = session_with(_ACK_HEAD)
    assert session.keep_epoch_across_request(tail) is True
    session, tail = session_with(_ACK_HEAD, addendum=True)
    assert session.keep_epoch_across_request(tail) is True
    session, tail = session_with(_ACK_HEAD + ".")
    assert session.keep_epoch_across_request(tail) is False, "a finished sentence still rotates"
    session, tail = session_with(_ACK_HEAD, second_owner="d2")
    assert session.keep_epoch_across_request(tail) is False, "two requests are two replies"
    session, tail = session_with(_ACK_HEAD, second_owner=None)
    assert session.keep_epoch_across_request(tail) is False
