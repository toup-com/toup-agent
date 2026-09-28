"""F0 Sequence C: which caller turn a direct reply answers.

The supervisor's reproduction (protocol sha256 4913f4cc…): the agent is
mid-sentence when the caller asks "What about Sunday?" over it, below the
barge-in thresholds.  The tail of that sentence ("… except on holidays when
hours are shorter") comes back through the hot mic as an echo-only turn.
Seconds later the model answers "On Sunday it opens at noon."  The answer was
parented to the ECHO turn.

Root cause, traced: the question's turn was still open, so the rest of the
agent's sentence was held behind it.  The next held word began 40 ms after the
question ENDED, and the held-output rule read "begins after the new turn,
not over it" per delta — so the old sentence's continuation was given the
question as its parent and fenced it.  Only the echo was left for the real
answer, and `consume_direct_turn` then fell back to it.

The fix, and what these tests hold:

* a held delta that continues an unbroken output run keeps that run's parent
  (`_LiveSession.unbroken_output_parent`): no gap on the provider timeline
  since the previous output delta, and no idle on the relay clock.  A real
  pause is still a boundary, and output after it still answers the open turn;
* a direct reply is parented only to the caller's own speech.  With no causal
  turn eligible it is PARENTLESS, never the echo's — and the echo turn is not
  fenced by it, so a delegation's last-resort fallback can still bind a caller
  who repeated the agent's words;
* once a decision claims a causal turn, the echo-only turns between that turn
  and the decision are settled: a second provider decision can no longer run
  the agent's own echoed words.

Clocks are the supervisor probe's (the three-videos scale: settle 40 / gap 600
/ hard gap 1300 / output-epoch gap 300).
"""

import asyncio
import time

import pytest

import test_live_harness as H
import app.services.live_voice_protocol as live


CLOCKS = dict(
    voice_live_transcript_settle_ms=40,
    voice_live_utterance_gap_ms=600,
    voice_live_utterance_hard_gap_ms=1300,
    voice_live_output_epoch_gap_ms=300,
    voice_live_delegation_ttl_s=2.5,
    voice_live_interrupt_suppress_ms=400,
    voice_live_result_defer_max_ms=1500,
    voice_live_speak_timeout_s=1.5,
    voice_live_progress_speak_after_s=600.0,
)

REQUEST = ["Search", " the", " Robarts", " library", " opening", " hours."]
REQUEST_TEXT = "Search the Robarts library opening hours."
ACK = "Sure, let me look up the Robarts library opening hours for you right now.".split()
ACK_ECHO = [" sure let me look", " up the robarts library", " opening hours for you"]

AGENT = ("the library opens at nine in the morning on weekdays and it closes at ten "
         "at night except on holidays when hours are shorter").split()
QUESTION = "What about Sunday?"
REPLY = "On Sunday it opens at noon.".split()


async def _run(monkeypatch, script, *, think_answer="Robarts opens at 8:30."):
    """One relay call; `script(provider, at)` runs on wall time since
    `session.start`.  Returns (phone, display requests the tenant ran)."""

    H.fast_clocks(monkeypatch, **CLOCKS)
    calls: list[str] = []
    box: dict = {}

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append(kwargs.get("display_request") or task)
        return think_answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            box["p"] = provider
            box["t0"] = asyncio.get_running_loop().time()
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def at(wall_s):
        delay = box["t0"] + wall_s - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)

    client = H.FakeClient([H.config(), lambda: script(box["p"], at), {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=30.0, db_session_id=f"db-f0-direct-{time.monotonic_ns()}",
    )
    return client, calls


def _finals(client):
    return [(f["turn_id"], f["text"], f.get("echo_only")) for f in client.of("transcript") if f.get("final")]


def _turn_of(client, prefix):
    return next(t for t, text, _e in _finals(client) if text.startswith(prefix))


def _epochs(client):
    """(assistant_turn_id, parent_user_turn_id, text) per output epoch, in order."""

    order: list[str] = []
    parent: dict[str, object] = {}
    text: dict[str, str] = {}
    for frame in client.of("response_text"):
        epoch = frame["assistant_turn_id"]
        if epoch not in parent:
            order.append(epoch)
            parent[epoch] = frame.get("parent_user_turn_id")
            text[epoch] = ""
        assert frame.get("parent_user_turn_id") == parent[epoch], "one epoch, one parent"
        text[epoch] += frame.get("text") or ""
    return [(e, parent[e], text[e].strip()) for e in order]


def _created(client, delegation_id=None):
    return [
        (f.get("delegation_id"), f.get("title"), f.get("turn_id"), f.get("request_turn_ids"))
        for f in client.of("delegation")
        if f.get("phase") == "created"
        and (delegation_id is None or f.get("delegation_id") == delegation_id)
    ]


# ── the agent, the caller and the echo, as provider events ─────────────

def _speak(words, *, wall_s, position_ms, step_s, span_ms, audio="AAAA"):
    async def run(p, at):
        for i, word in enumerate(words):
            await at(wall_s + i * step_s)
            position = position_ms + i * span_ms
            p.push(H.out_text(" " + word, position, position + span_ms))
            p.push(H.out_audio(audio))
    return run


def _say(pieces, *, wall_s, position_ms, step_s=0.12, span_ms=120):
    async def run(p, at):
        position = position_ms
        for i, piece in enumerate(pieces):
            await at(wall_s + i * step_s)
            p.push(H.user_delta(piece, position, position + span_ms))
            position += span_ms
    return run


def _echo(chunks, *, wall_s, step_s, lag_ms, span_ms):
    """The agent's voice back through the hot mic, on the INPUT timeline."""

    async def run(p, at):
        for k, chunk in enumerate(chunks):
            wall = wall_s + k * step_s
            await at(wall)
            position = int(wall * 1000) - lag_ms
            p.push(H.user_delta(chunk, position, position + span_ms))
    return run


def _delegate(delegation_id, *, wall_s, offset_ms):
    async def run(p, at):
        await at(wall_s)
        p.push(H.delegation(delegation_id, offset_ms))
    return run


def _gather(*parts, until_s):
    async def script(p, at):
        await asyncio.gather(*(part(p, at) for part in parts))
        await at(until_s)
    return script


# The supervisor's Sequence C, event for event.
SEQUENCE_C = (
    _speak(AGENT, wall_s=0.2, position_ms=200, step_s=0.08, span_ms=80),
    _say([QUESTION], wall_s=0.9, position_ms=900, span_ms=300),
    _echo([" except on holidays", " when hours are", " shorter"],
          wall_s=1.85, step_s=0.12, lag_ms=120, span_ms=120),
    _speak(REPLY, wall_s=3.2, position_ms=3200, step_s=0.0, span_ms=60, audio="BBBB"),
)


# ══════════════════════════════════════════════════════════════════════
# The permanent regression: the exact supervisor sequence
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_sequence_c_the_direct_reply_answers_the_question_not_the_echo(monkeypatch):
    client, _calls = await _run(monkeypatch, _gather(*SEQUENCE_C, until_s=5.0))

    finals = _finals(client)
    question = _turn_of(client, "What about")
    echo = _turn_of(client, "except on holidays")

    # The supervisor probe's own assertion, verbatim in substance.
    reply_parents = sorted({
        f.get("parent_user_turn_id") for f in client.frames
        if f.get("parent_user_turn_id") and "On" in str(f.get("text") or "")
    })
    assert reply_parents == [question], (reply_parents, finals, _epochs(client))

    assert [(t, e) for t, _x, e in finals] == [(question, None), (echo, True)], finals
    epochs = _epochs(client)
    # The answer is the ONLY output parented to the question …
    assert [text for _e, parent, text in epochs if parent == question] == [
        "On Sunday it opens at noon.",
    ], epochs
    # … the sentence it was asked over keeps its own (greeting: null) parent,
    # including the words held behind the question and released after it …
    assert all(parent is None for _e, parent, text in epochs if "holidays" in text), epochs
    assert "".join(text for _e, parent, text in epochs if parent is None).count("shorter") == 1
    # … and nothing is ever parented to the agent's own echoed words.
    assert all(parent != echo for _e, parent, _t in epochs), epochs


@pytest.mark.asyncio
async def test_sequence_c_a_real_pause_is_still_a_boundary_for_the_held_answer(monkeypatch):
    """The positive twin.  The agent STOPS mid-sentence when the caller asks
    (a gap on the provider timeline) and answers while the question's turn is
    still open.  That held answer began after a real boundary: it answers the
    question, exactly as before the fix."""

    stopped = AGENT[:12]  # "… on weekdays and it closes" ends at wall ~1.1

    client, _calls = await _run(monkeypatch, _gather(
        _speak(stopped, wall_s=0.2, position_ms=200, step_s=0.08, span_ms=80),
        _say([QUESTION], wall_s=0.9, position_ms=900, span_ms=300),
        # 1300 > the last agent position (1160): a pause, while the turn is open.
        _speak(REPLY, wall_s=1.3, position_ms=1300, step_s=0.02, span_ms=60, audio="BBBB"),
        until_s=3.5,
    ))

    question = _turn_of(client, "What about")
    epochs = _epochs(client)
    assert [text for _e, parent, text in epochs if parent == question] == [
        "On Sunday it opens at noon.",
    ], epochs
    assert all(parent is None for _e, parent, text in epochs if "weekdays" in text), epochs


# ══════════════════════════════════════════════════════════════════════
# Parentless, never the echo: a reply with no causal turn to answer
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_reply_with_only_echo_left_is_parentless(monkeypatch):
    """The caller's request is answered directly; the answer echoes back as an
    echo-only turn; the model then speaks again.  Nothing causal is left to
    answer, so that output carries an explicit null parent — the app's watch
    reads null as covering the newest non-echo final — and never the echo."""

    answer = "Robarts opens at nine in the morning and closes at ten at night.".split()
    client, calls = await _run(monkeypatch, _gather(
        _say(["What", " time", " does", " Robarts", " open?"], wall_s=0.3, position_ms=300),
        _speak(answer, wall_s=1.7, position_ms=1700, step_s=0.05, span_ms=60),
        _echo([" robarts opens at nine", " in the morning and", " closes at ten at night"],
              wall_s=2.0, step_s=0.25, lag_ms=150, span_ms=150),
        _speak("Anything else I can help with?".split(), wall_s=3.8, position_ms=3800,
               step_s=0.05, span_ms=60, audio="BBBB"),
        until_s=5.0,
    ))

    request = _turn_of(client, "What time")
    echo = _turn_of(client, "robarts opens")
    epochs = _epochs(client)
    assert [(parent, text) for _e, parent, text in epochs] == [
        (request, "Robarts opens at nine in the morning and closes at ten at night."),
        (None, "Anything else I can help with?"),
    ], (epochs, echo)
    frames = [f for f in client.of("response_text") if "Anything" in (f.get("text") or "")]
    assert frames and all(
        "parent_user_turn_id" in f and f["parent_user_turn_id"] is None for f in frames
    ), "parentless must be an explicit null on the wire"
    assert dict((t, e) for t, _x, e in _finals(client))[echo] is True
    assert calls == []


# ══════════════════════════════════════════════════════════════════════
# Delegations: the caller's words run, exactly once; the agent's never
# ══════════════════════════════════════════════════════════════════════

REQUEST_TURN = _say(REQUEST, wall_s=0.3, position_ms=300)
ACK_SPOKEN = _speak(ACK, wall_s=1.7, position_ms=1700, step_s=0.05, span_ms=60)
ACK_ECHOED = _echo(ACK_ECHO, wall_s=2.0, step_s=0.25, lag_ms=150, span_ms=150)


@pytest.mark.asyncio
async def test_ack_then_delegation_runs_the_request_once_and_a_replay_runs_nothing(monkeypatch):
    client, calls = await _run(monkeypatch, _gather(
        REQUEST_TURN, ACK_SPOKEN,
        _delegate("d-ack", wall_s=2.45, offset_ms=2420),
        _delegate("d-ack", wall_s=3.6, offset_ms=2420),   # the same id, replayed
        until_s=5.5,
    ))

    request = _turn_of(client, "Search the Robarts")
    assert calls == [REQUEST_TEXT]
    assert _created(client) == [("d-ack", REQUEST_TEXT, request, [request])]


@pytest.mark.asyncio
async def test_sequence_a_the_echoed_ack_never_becomes_the_anchor(monkeypatch):
    client, calls = await _run(monkeypatch, _gather(
        REQUEST_TURN, ACK_SPOKEN, ACK_ECHOED,
        _delegate("d-ack", wall_s=2.45, offset_ms=2420),
        until_s=5.5,
    ))

    request = _turn_of(client, "Search the Robarts")
    assert calls == [REQUEST_TEXT], "the task ran on the agent's echoed words"
    assert _created(client) == [("d-ack", REQUEST_TEXT, request, [request])]


@pytest.mark.asyncio
async def test_a_second_decision_after_the_echoed_ack_never_runs_the_echo(monkeypatch):
    """GPT-Live deciding twice for one utterance (recorded in V1).  The first
    decision binds the request; the ack's echo, between that request and the
    decision, is settled with it — so the second decision runs nothing rather
    than the agent's own words."""

    client, calls = await _run(monkeypatch, _gather(
        REQUEST_TURN, ACK_SPOKEN, ACK_ECHOED,
        _delegate("d-ack", wall_s=2.45, offset_ms=2420),
        _delegate("d-second", wall_s=3.4, offset_ms=3380),
        until_s=6.5,
    ))

    request = _turn_of(client, "Search the Robarts")
    assert calls == [REQUEST_TEXT], calls
    assert _created(client) == [("d-ack", REQUEST_TEXT, request, [request])]
    expired = [
        f for f in client.of("delegation")
        if f.get("delegation_id") == "d-second" and f.get("phase") == "expired"
    ]
    assert expired and expired[-1].get("reason") == "no_causal_turn", expired


@pytest.mark.asyncio
async def test_sequence_b_a_question_over_the_reply_is_not_rebound_to_its_echo_tail(monkeypatch):
    """The caller asks for work over the agent's sentence (below barge-in);
    the sentence's tail echoes back after the question closed; the delegation
    offset lands after that echo started.  The question runs, alone."""

    ask = ["Find", " the", " Sunday", " hours", " for", " Gerstein."]
    client, calls = await _run(monkeypatch, _gather(
        _speak(AGENT, wall_s=0.2, position_ms=200, step_s=0.08, span_ms=80),
        _say(ask, wall_s=0.9, position_ms=900, step_s=0.05, span_ms=50),
        _echo([" except on holidays", " when hours are", " shorter"],
              wall_s=1.85, step_s=0.12, lag_ms=120, span_ms=120),
        _delegate("dB", wall_s=2.3, offset_ms=2200),
        until_s=5.0,
    ))

    question = _turn_of(client, "Find the Sunday")
    assert calls == ["Find the Sunday hours for Gerstein."], calls
    assert _created(client, "dB") == [
        ("dB", "Find the Sunday hours for Gerstein.", question, [question]),
    ]


@pytest.mark.asyncio
async def test_a_caller_who_repeats_the_agents_words_still_gets_the_work_once(monkeypatch):
    """The echo fallback that REMAINS for delegations, and why.

    The caller repeats the agent's own offer while it is still speaking; the
    relay's verdict is word overlap, so the turn is judged echo-only.  The
    model acknowledges (parentless: no causal turn to answer) and delegates.
    With no causal turn eligible, the delegation binds the echo-judged turn:
    dropping it would silently lose a request the caller really made, and the
    app never re-asks an `echo_only` turn.  It runs once; a replay runs
    nothing."""

    offer = "I can look up the Robarts library opening hours for you if you like.".split()
    repeat = [" look up the", " robarts library", " opening hours"]
    client, calls = await _run(monkeypatch, _gather(
        _speak(offer, wall_s=0.2, position_ms=200, step_s=0.08, span_ms=80),
        _say(repeat, wall_s=1.0, position_ms=1000, step_s=0.12, span_ms=120),
        _speak("Sure, one moment.".split(), wall_s=2.4, position_ms=2400,
               step_s=0.05, span_ms=60, audio="BBBB"),
        _delegate("d-rep", wall_s=2.6, offset_ms=2550),
        _delegate("d-rep", wall_s=3.4, offset_ms=2550),   # replayed id
        until_s=5.0,
    ))

    repeated = _turn_of(client, "look up the robarts")
    assert calls == ["look up the robarts library opening hours"], calls
    assert _created(client) == [
        ("d-rep", "look up the robarts library opening hours", repeated, [repeated]),
    ]
    # The acknowledgement answered no causal turn: parentless, not the echo.
    assert [(p, t) for _e, p, t in _epochs(client) if t.startswith("Sure")] == [
        (None, "Sure, one moment."),
    ]
    # …and it WAS judged echo: this is the fallback, not a causal bind.
    assert dict((t, e) for t, _x, e in _finals(client))[repeated] is True


# ══════════════════════════════════════════════════════════════════════
# The rules, directly
# ══════════════════════════════════════════════════════════════════════

def _turn(ordinal, text, start, end, **extra):
    fields = dict(causal_speech=True, closed=True)
    fields.update(extra)
    return live.Utterance(
        turn_id=f"live-utt:p:{ordinal}", ordinal=ordinal,
        start_ms=start, end_ms=end, text=text, **fields,
    )


def _session(*turns):
    session = live._LiveSession.__new__(live._LiveSession)
    session.utterances = live.UtteranceAssembler("p")
    session.utterances.closed_queue = list(turns)
    return session


def test_a_direct_reply_takes_the_caller_and_never_the_echo():
    question = _turn(1, "What about Sunday?", 0, 300)
    later_echo = _turn(2, " on holidays", 400, 900, causal_speech=False)
    session = _session(question, later_echo)

    assert session.consume_direct_turn(1000) == question.turn_id
    assert question.dispatched
    # The echo between the answered question and the reply is settled with it.
    assert later_echo.dispatched and later_echo.consumed_by is None
    # Nothing causal left: the next reply is parentless.
    assert session.consume_direct_turn(1100) == ""


def test_with_only_echo_eligible_a_reply_is_parentless_and_the_echo_stays_bindable():
    answered = _turn(1, "What time does Robarts open?", 0, 300, dispatched=True)
    echo = _turn(2, " robarts opens at nine", 400, 900, causal_speech=False)
    session = _session(answered, echo)

    assert session.consume_direct_turn(1000) == ""
    assert not echo.dispatched, "a parentless reply fences nothing"
    # …so a delegation's last resort (a caller repeating the agent) still has it.
    assert session.utterances.causal_candidate(1000) is echo
    # A causal turn that starts only AFTER the reply began is not its cause.
    later = _turn(3, "And Gerstein?", 1200, 1500)
    session.utterances.closed_queue.append(later)
    assert session.consume_direct_turn(1000) == ""
    assert not later.dispatched


def test_the_delegation_fallback_is_exactly_no_causal_turn_eligible():
    asm = live.UtteranceAssembler("p")
    caller = _turn(1, "Search the Robarts library opening hours.", 0, 800)
    echo = _turn(2, " sure let me look up the robarts library", 1600, 2400, causal_speech=False)
    asm.closed_queue = [caller, echo]
    # A causal turn at/before the offset always wins, however late the echo ends.
    assert asm.causal_candidate(2500) is caller
    # Fenced by the acknowledgement: still the anchor while that epoch may claim it.
    caller.dispatched = True
    assert asm.causal_candidate(2500, revocable_turn_id=caller.turn_id) is caller
    # Not revocable any more, nothing causal eligible: the echo is the last resort.
    assert asm.causal_candidate(2500) is echo
    # A consumed caller turn never comes back, revocable or not.
    caller.consumed_by = "d-1"
    assert asm.causal_candidate(2500, revocable_turn_id=caller.turn_id) is echo
    # Settled echo (a decision already claimed the turn before it) is not bindable.
    asm.settle_echo_after(caller, 2500)
    assert echo.dispatched and asm.causal_candidate(2500) is None


def test_settling_touches_only_echo_between_the_anchor_and_the_decision():
    asm = live.UtteranceAssembler("p")
    older_echo = _turn(1, " the weather in toronto", 0, 400, causal_speech=False)
    anchor = _turn(2, "Search the Robarts library.", 600, 1000)
    between = _turn(3, " sure let me look", 1100, 1400, causal_speech=False)
    caller_after = _turn(4, "Also Gerstein.", 1500, 1800)
    after_decision = _turn(5, " one moment", 2600, 2900, causal_speech=False)
    asm.closed_queue = [older_echo, anchor, between, caller_after, after_decision]

    assert asm.settle_echo_after(anchor, 2000) == 1
    assert between.dispatched
    assert not older_echo.dispatched
    assert not caller_after.dispatched
    assert not after_decision.dispatched
    # A fallback (echo) anchor settles nothing.
    assert asm.settle_echo_after(older_echo, 5000) == 0


class _Adapter:
    """Only what `unbroken_output_parent` reads from the adapter."""

    def __init__(self, *, open_=False, text=False, end_ms=0, parent="", at=0.0, gap_ms=300):
        self.output_open = open_
        self.output_has_text = text
        self.output_end_ms = end_ms
        self.output_parent_user_turn_id = parent
        self.last_output_monotonic = at
        self.gap_ms = gap_ms


def _held(start, end, parent, at, kind="session.output_transcript.delta"):
    return {
        "type": kind, "delta": "x", "start_ms": start, "end_ms": end,
        "_toup_parent_resolved": True, "_toup_parent_tentative": False,
        "_toup_parent_user_turn_id": parent, "_toup_arrived_at": at,
    }


def _run_session(adapter, held=()):
    session = live._LiveSession.__new__(live._LiveSession)
    session.adapter = adapter
    session.deferred_output = list(held)
    return session


def test_a_held_delta_continues_the_run_only_when_neither_clock_broke():
    now = 100.0
    held = [_held(1160, 1240, "", now - 0.08)]
    session = _run_session(_Adapter(), held)
    # Touching the previous held delta, 80 ms later: the same sentence.
    assert session.unbroken_output_parent(1240, now) == ""
    # Overlapping it: the same sentence.
    assert session.unbroken_output_parent(1200, now) == ""
    # One provider millisecond of pause: a boundary (the pre-fix rule decides).
    assert session.unbroken_output_parent(1241, now) is None
    # Touching, but the stream idled for the output-epoch gap: a boundary.
    assert session.unbroken_output_parent(1240, now + 0.3) is None
    # No position: no evidence of anything.
    assert session.unbroken_output_parent(None, now) is None
    # A held run already answering the caller's turn keeps answering it.
    held_answer = [_held(1300, 1360, "live-utt:p:1", now - 0.02)]
    assert _run_session(_Adapter(), held_answer).unbroken_output_parent(1360, now) == "live-utt:p:1"
    # Held audio after the text is newer activity, not a position.
    audio_last = held + [dict(_held(0, 0, "", now - 0.01), type="session.output_audio.delta")]
    assert _run_session(_Adapter(), audio_last).unbroken_output_parent(1240, now + 0.28) == ""


def test_with_nothing_held_the_run_is_the_live_epoch_on_the_phone():
    now = 50.0
    live_epoch = _Adapter(open_=True, text=True, end_ms=1160, parent="live-utt:p:3", at=now - 0.05)
    assert _run_session(live_epoch).unbroken_output_parent(1160, now) == "live-utt:p:3"
    assert _run_session(live_epoch).unbroken_output_parent(1200, now) is None
    # A retired epoch (gap, playback_idle, interrupt) is a boundary.
    retired = _Adapter(open_=False, text=False, end_ms=0, at=now - 0.05)
    assert _run_session(retired).unbroken_output_parent(0, now) is None
    # An epoch with only untimed audio has no position to continue.
    audio_only = _Adapter(open_=True, text=False, end_ms=0, at=now - 0.05)
    assert _run_session(audio_only).unbroken_output_parent(0, now) is None
    # Idle past the gap on the relay clock: a boundary.
    stale = _Adapter(open_=True, text=True, end_ms=1160, parent="", at=now - 0.5)
    assert _run_session(stale).unbroken_output_parent(1160, now) is None
