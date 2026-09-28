"""What a delegation binds to, what it is classified by, and whose words it runs.

R2 fix round (addendum 1–3, findings F0 F1 F10 F2 F3 F4 F26 F5 F6 F15 F18/F37,
relay half).  The request span (§B) supplies the REQUEST TEXT; authority —
status, cancel, media control, correction — is read from the anchor's own words
plus a split continuation of the same phrase, never because an unanswered
lead-in was swept into the span.  An echo-only turn is never a direct reply's
parent (the reply is parentless instead), is a delegation's anchor only as the
last resort when no causal turn is eligible, and its transcript frames say so
(`echo_only`).

Every test drives the real relay against the fake GPT-Live harness with
provider-shaped events on scaled clocks (settle 40 / gap 400 / hard gap 900 ms,
the production 350 / 1200 / 2600 relationship) unless it says otherwise, and
every behaviour has a positive twin that must keep working.
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
    voice_live_delegation_ttl_s=3.0,
    voice_live_interrupt_suppress_ms=300,
)

V03 = [
    "live_turns", "state_seq", "outcomes", "delegation_frames", "task_lifecycle",
    "playback_frames", "turn_timing", "media_control", "media_transport", "reask_turns",
]


class TimedClient(H.FakeClient):
    """The phone, with the arrival instant of EVERY frame."""

    def __init__(self, script=None):
        super().__init__(script)
        self.times: list[float] = []

    async def send_json(self, frame):
        self.times.append(time.monotonic())
        await super().send_json(frame)


async def _until(pred, timeout=6.0, label="condition"):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred():
        if loop.time() > end:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.01)


async def _wait_quietly(pred, timeout=4.0):
    """Wait for `pred` without raising: the ASSERTIONS decide the verdict."""

    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred() and loop.time() < end:
        await asyncio.sleep(0.01)
    return bool(pred())


async def _call(
    monkeypatch, script, *, clocks=None, think=None, control=None, features=None,
    timeout=30.0, box=None, speak_on_commentary=False,
):
    """One relay call.  `script(provider, at)` runs on wall time since
    `session.start`; `box["client"]` is the phone."""

    H.fast_clocks(monkeypatch, **dict(CLOCKS, **(clocks or {})))
    calls: list[dict] = []

    async def default_think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, **kwargs})
        return "An answer.", "test-model"

    controls: list[str] = []

    async def default_control(_user_id, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped" if action == "stop" else "ok"}

    H.patch_relay(
        monkeypatch, think=think or default_think, control=control or default_control,
    )
    box = {} if box is None else box
    box["calls"], box["controls"] = calls, controls

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

    frame = H.config(features=list(features or V03))
    client = TimedClient([frame, lambda: script(box["provider"], at), {"type": "stop"}])
    box["client"] = client
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=speak_on_commentary,
    )
    await H.run_relay(
        client, provider, timeout=timeout,
        db_session_id=f"db-span-authority-{time.monotonic_ns()}",
    )
    return client, provider, calls, controls


def _finals(client):
    return [f for f in client.of("transcript") if f.get("final")]


def _final_id(client, text):
    return next(f["turn_id"] for f in _finals(client) if f["text"] == text)


def _created(client, delegation_id=None):
    return [
        f for f in client.of("delegation")
        if f.get("phase") == "created"
        and (delegation_id is None or f.get("delegation_id") == delegation_id)
    ]


def _phases(client, delegation_id):
    return [f["phase"] for f in client.of("delegation") if f.get("delegation_id") == delegation_id]


def _displays(calls):
    return [c.get("display_request") for c in calls]


async def _say(p, at, words, *, wall_s, position_ms, step_s=0.12, span_ms=120):
    position = position_ms
    for index, word in enumerate(words):
        await at(wall_s + index * step_s)
        p.push(H.user_delta(word, position, position + span_ms))
        position += span_ms
    return position


def _turn(ordinal, text, start, end, **extra):
    from app.services.live_voice_protocol import Utterance

    fields = dict(causal_speech=True, closed=True)
    fields.update(extra)
    return Utterance(
        turn_id=f"live-utt:p:{ordinal}", ordinal=ordinal,
        start_ms=start, end_ms=end, text=text, **fields,
    )


# ══════════════════════════════════════════════════════════════════════
# F0 — an echo-only closed turn is never the anchor or the reply's parent
# ══════════════════════════════════════════════════════════════════════

ROBARTS = "Search the Robarts library opening hours."
ACK = "Sure, let me look up the Robarts library opening hours for you."
ACK_ECHO = [" sure let me look", " up the robarts library", " opening hours for you"]
ECHO_CLOCKS = dict(
    voice_live_utterance_gap_ms=200, voice_live_utterance_hard_gap_ms=400,
    voice_live_output_epoch_gap_ms=5000, voice_live_delegation_ttl_s=3.0,
    voice_live_transcript_settle_ms=10,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("delegation_first", [False, True])
async def test_f0_the_echoed_acknowledgement_never_becomes_the_anchor(monkeypatch, delegation_first):
    """Sequence A: the agent acknowledges, its acknowledgement comes back through
    the hot mic as an echo-only turn, and GPT-Live's delegation offset lands
    after that echo started.  The delegation must still run the caller's words,
    not the agent's.  (`delegation_first` is the ordering that always worked.)"""

    async def script(p, _at):
        p.push(H.user_delta(ROBARTS, 0, 800))
        client = box["client"]
        await _until(lambda: len(_finals(client)) == 1, label="request final")
        p.push(H.out_text(ACK, 900, 1500))
        p.push(H.out_audio("AAAA"))
        await _until(lambda: bool(client.of("response_text")), label="ack on the phone")
        if delegation_first:
            p.push(H.delegation("d1", 2000))
            await asyncio.sleep(0.05)
        position = 1700
        for chunk in ACK_ECHO:
            p.push(H.user_delta(chunk, position, position + 200))
            position += 200
            await asyncio.sleep(0.03)
        if not delegation_first:
            p.push(H.delegation("d1", 2000))
        await _until(lambda: "completed" in _phases(client, "d1"), label="d1 done")

    box: dict = {}
    client, _p, calls, _c = await _call(monkeypatch, script, clocks=ECHO_CLOCKS, box=box)

    assert _displays(calls) == [ROBARTS], _displays(calls)
    caller_turn = _final_id(client, ROBARTS)
    created = _created(client, "d1")[0]
    assert created["turn_id"] == caller_turn
    assert created["request_turn_ids"] == [caller_turn]
    assert created["title"] == ROBARTS


GERSTEIN_Q0 = "When is Robarts open?"
GERSTEIN_E1 = "Robarts opens at nine in the morning and closes at ten at night except on holidays."
GERSTEIN_ECHO = [" closes at ten", " at night except", " on holidays"]


@pytest.mark.asyncio
async def test_f0_a_request_asked_over_the_reply_is_not_rebound_to_the_echo_tail(monkeypatch):
    """Sequence B: the caller asks a new question over the reply (below the
    barge-in thresholds), the reply's tail echoes back as a later turn, and the
    delegation offset lands after that echo started."""

    question = "Find the Sunday hours for Gerstein."

    async def script(p, _at):
        client = box["client"]
        p.push(H.user_delta(GERSTEIN_Q0, 0, 600))
        await _until(lambda: len(_finals(client)) == 1)
        p.push(H.out_text(GERSTEIN_E1, 700, 1400))
        p.push(H.out_audio("AAAA"))
        await _until(lambda: bool(client.of("response_text")))
        p.push(H.user_delta(question, 2000, 2400))
        await _until(lambda: len(_finals(client)) == 2)
        position = 2700
        for chunk in GERSTEIN_ECHO:
            p.push(H.user_delta(chunk, position, position + 150))
            position += 150
            await asyncio.sleep(0.02)
        p.push(H.delegation("dB", 2900))
        await _until(lambda: "completed" in _phases(client, "dB"))

    box: dict = {}
    client, _p, calls, _c = await _call(monkeypatch, script, clocks=ECHO_CLOCKS, box=box)

    assert _displays(calls) == [question], _displays(calls)
    created = _created(client, "dB")[0]
    assert created["request_turn_ids"] == [_final_id(client, question)]


@pytest.mark.asyncio
async def test_f0_a_direct_reply_is_parented_to_the_question_not_a_later_echo(monkeypatch):
    """Sequence C: an echo turn that closed AFTER the caller's question must not
    take the model's direct reply to that question."""

    H.fast_clocks(monkeypatch, **dict(CLOCKS, **ECHO_CLOCKS))
    H.patch_relay(monkeypatch)
    box: dict = {}

    def on_send(provider, event):
        if event["type"] == "session.start":
            box["p"] = provider
            provider.push(H.user_delta(GERSTEIN_Q0, 0, 600))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def question_then_echo():
        await _until(lambda: "p" in box)
        p = box["p"]
        await _until(lambda: len(_finals(client)) == 1)
        p.push(H.out_text(GERSTEIN_E1, 700, 1400))
        p.push(H.out_audio("AAAA"))
        await _until(lambda: bool(client.of("response_text")))
        p.push(H.user_delta("What about Sunday?", 2000, 2300))
        await _until(lambda: len(_finals(client)) == 2)
        position = 2600
        for chunk in GERSTEIN_ECHO:
            p.push(H.user_delta(chunk, position, position + 150))
            position += 150
            await asyncio.sleep(0.02)
        await _until(lambda: len(_finals(client)) == 3)
        # The phone reports the first reply played out: its epoch retires.
        rid = client.of("response_text")[0]["assistant_turn_id"]
        client._script.insert(0, {
            "type": "playback_idle", "response_id": rid, "item_id": rid, "played_ms": 4000,
        })

    async def the_direct_reply():
        await asyncio.sleep(0.05)
        for i, word in enumerate("On Sunday it opens at noon.".split()):
            box["p"].push(H.out_text(" " + word, 3300 + i * 60, 3360 + i * 60))
            box["p"].push(H.out_audio("BBBB"))
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(features=V03), question_then_echo, the_direct_reply, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-f0-direct-parent")

    parents = []
    for frame in client.of("response_text"):
        key = (frame.get("assistant_turn_id"), frame.get("parent_user_turn_id"))
        if key not in parents:
            parents.append(key)
    assert parents[-1][1] == _final_id(client, "What about Sunday?"), parents


def test_f0_causal_turns_win_and_an_echo_turn_is_only_a_fallback():
    """Causal turns always win.  The echo fallback is a DELEGATION's last
    resort only (a caller who repeats the agent's words keeps their request);
    a direct reply never takes it.

    F0 Sequence C inverted the direct half of this pin: it used to assert that
    once the question was consumed, the next direct reply was parented to the
    later echo turn.  That is exactly the supervisor's failure (the model's
    answer parented to the agent's own echoed words), so the reply is now
    PARENTLESS — see tests/test_live_f0_direct_parent.py."""

    from app.services.live_voice_protocol import UtteranceAssembler, _LiveSession

    asm = UtteranceAssembler("p")
    caller = _turn(1, "What about Sunday?", 0, 300)
    echo = _turn(2, " closes at ten at night", 400, 900, causal_speech=False)
    asm.closed_queue = [caller, echo]
    assert asm.causal_candidate(1000) is caller
    # The revocable (acknowledged) caller turn still beats a later echo.
    caller.dispatched = True
    assert asm.causal_candidate(1000, revocable_turn_id=caller.turn_id) is caller
    # With no causal turn eligible at all, the echo is still bindable by a
    # DELEGATION: a caller who repeats the agent's words keeps their request.
    assert asm.causal_candidate(1000) is echo

    session = _LiveSession.__new__(_LiveSession)
    session.utterances = UtteranceAssembler("p2")
    question = _turn(1, "What about Sunday?", 0, 300)
    later_echo = _turn(2, " on holidays", 400, 900, causal_speech=False)
    session.utterances.closed_queue = [question, later_echo]
    assert session.consume_direct_turn(1000) == question.turn_id
    # The echo between the question and its answer is settled with it …
    assert question.dispatched and later_echo.dispatched
    assert later_echo.consumed_by is None
    # … and with nothing causal left, the next reply is parentless, never the echo.
    assert session.consume_direct_turn(1000) == ""

    session.utterances = UtteranceAssembler("p3")
    only_echo = _turn(1, " on holidays", 400, 900, causal_speech=False)
    session.utterances.closed_queue = [only_echo]
    assert session.consume_direct_turn(1000) == ""
    # A parentless reply fences nothing: a delegation's fallback still has it.
    assert not only_echo.dispatched
    assert session.utterances.causal_candidate(1000) is only_echo


# ══════════════════════════════════════════════════════════════════════
# F1 / F10 — a status check-in never swallows the request it swept in
# ══════════════════════════════════════════════════════════════════════

DEAN = "Who is the dean there?"


def _held_research_think(release, *, tool=False, first_answer="Robotics professors: A, B."):
    calls: list[dict] = []

    async def think(_user_id, task, _session_id, relay=None, out=None, **kwargs):
        calls.append({"task": task, **kwargs})
        if len(calls) == 1:
            if tool and relay is not None:
                await relay.on_event({
                    "type": "tool.start", "call_id": "t1", "name": "web_search",
                    "args": {"query": "q"},
                })
            await asyncio.wait_for(release.wait(), timeout=15)
            return first_answer, "test-model"
        return "The dean is Professor X.", "test-model"

    return think, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("late", ["none", "after_s"])
async def test_f1_a_status_check_in_dispatches_the_question_it_swept_in_while_work_runs(
    monkeypatch, late,
):
    """Task A runs; the caller asks a question the model neither answers nor
    delegates; after a real pause "What happened?" is delegated.  The span
    binds both.  The status words must not answer for the question: THAT
    request runs, with its own words as the task and title (addendum 2/3), and
    the running task is left alone.  A later provider delegation for the same
    question never runs it twice."""

    release = asyncio.Event()
    think, calls = _held_research_think(release)
    counters = H.capture_counters(monkeypatch)

    async def script(p, at):
        end = await _say(p, at, ["Find", " robotics", " professors", " at", " UofT."],
                         wall_s=0.5, position_ms=500)
        p.push(H.delegation("d1", end + 40))
        await _until(lambda: calls, label="d1 running")
        q_end = await _say(p, at, ["Who", " is", " the", " dean", " there?"],
                           wall_s=2.5, position_ms=2500)
        s_end = await _say(p, at, [" What", " happened?"], wall_s=5.6, position_ms=q_end + 2500)
        p.push(H.delegation("d2", s_end + 60))
        await at(6.6)
        if late == "after_s":
            p.push(H.delegation("d3", s_end + 900))
        await at(8.0)
        release.set()
        await at(9.0)

    client, _p, _calls, _c = await _call(monkeypatch, script, think=think)

    displays = _displays(calls)
    assert displays.count(DEAN) == 1, displays
    assert displays[0] == "Find robotics professors at UofT."
    created = _created(client, "d2")
    assert [c["title"] for c in created] == [DEAN], created
    # The status turn is covered by the same task (the app never re-asks it).
    # Identity (addendum 2 item 1): the task takes the REQUEST's own anchor
    # turn — the question — and the check-in's turn follows it in spoken order
    # (item 2).  This pin read `turn_id == status_turn` before the merge chose
    # one identity (`rebind_to_requests`).
    status_turn = _final_id(client, "What happened?")
    assert created[0]["request_turn_ids"] == [_final_id(client, DEAN), status_turn]
    assert created[0]["turn_id"] == _final_id(client, DEAN)
    assert "superseded" not in _phases(client, "d1")
    assert "cancelled" not in _phases(client, "d1")
    assert not any(name == "live_status_answered" for name, _f in counters)
    assert ("live_status_dispatched", {"reason": "request_in_span"}) in counters


@pytest.mark.asyncio
async def test_f10_a_persian_question_swept_into_a_check_in_while_research_runs(monkeypatch):
    """F10: «رئیس دانشکده مهندسی کیه» (no answer, no delegation), then «چی شد؟»
    3 s later while research runs.  The dean question is worked on once; the
    research is not cancelled."""

    release = asyncio.Event()
    think, calls = _held_research_think(release, tool=True)
    dean = "رئیس دانشکده مهندسی کیه"

    async def script(p, at):
        p.push(H.user_delta("research the toronto robotics faculty", 0, 900))
        p.push(H.delegation("d1", 950))
        await _until(lambda: calls, label="d1 running")
        await at(1.5)
        p.push(H.user_delta(dean, 3000, 3800))
        await at(3.0)
        p.push(H.user_delta("چی شد؟", 6800, 7300))
        p.push(H.delegation("d2", 7350))
        await at(5.0)
        release.set()
        await at(6.0)

    client, _p, _calls, _c = await _call(monkeypatch, script, think=think)

    worked = [d for d in _displays(calls)[1:] if dean in (d or "")]
    assert len(worked) == 1, _displays(calls)
    assert "superseded" not in _phases(client, "d1")
    assert "cancelled" not in _phases(client, "d1")


@pytest.mark.asyncio
async def test_f1_an_unheard_result_is_not_replayed_over_a_question_in_the_span(monkeypatch):
    """The unheard-result branch: A finished but its result was never SAID.
    "What happened?" joined to an unanswered question in one span dispatches
    the question instead of re-reading A's result over it."""

    first_answer = "Professor Ada Lovelace works on robotics."
    calls: list = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, **kwargs})
        if len(calls) == 1:
            return first_answer, "test-model"
        return "The dean is Professor X.", "test-model"

    async def script(p, at):
        p.push(H.user_delta("find robotics professors at toronto", 0, 900))
        p.push(H.delegation("d1", 950))
        client = box["client"]
        # No speech for the commentary: nudge, then an honest spoken:false.
        await _until(lambda: any(
            f.get("spoken") is False for f in client.of("delegation")
            if f.get("delegation_id") == "d1"
        ), label="d1 unheard")
        p.push(H.user_delta(DEAN, 6000, 6800))
        await asyncio.sleep(0.6)
        p.push(H.user_delta("What happened?", 9300, 9700))
        p.push(H.delegation("d2", 9750))
        await _wait_quietly(lambda: len(calls) >= 2 or any(
            str(e.get("event_id") or "").startswith("toup-live-status-")
            for e in box["provider"].sent
        ))
        await asyncio.sleep(0.4)

    box: dict = {}
    client, provider, _calls, _c = await _call(
        monkeypatch, script, think=think, box=box,
        clocks={"voice_live_delegation_ttl_s": 2.0, "voice_live_speak_timeout_s": 0.5},
    )

    assert _displays(calls)[1:] == [DEAN], _displays(calls)
    replays = [
        e for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-status-")
    ]
    assert replays == [], "the unheard result was re-read over the newer question"


@pytest.mark.asyncio
async def test_f1_an_interjection_before_the_check_in_is_still_a_status_question(monkeypatch):
    """The positive twin: "Okay." is not a request, so "Okay." + "What
    happened?" while A runs is answered from state — no card, no second run."""

    release = asyncio.Event()
    think, calls = _held_research_think(release)
    counters = H.capture_counters(monkeypatch)

    async def script(p, at):
        end = await _say(p, at, ["Find", " robotics", " professors", " at", " UofT."],
                         wall_s=0.5, position_ms=500)
        p.push(H.delegation("d1", end + 40))
        await _until(lambda: calls, label="d1 running")
        await _say(p, at, ["Okay."], wall_s=2.5, position_ms=2500)
        await _say(p, at, ["What", " happened?"], wall_s=3.5, position_ms=3600)
        p.push(H.delegation("d2", 3900))
        await at(5.0)
        release.set()
        await at(5.8)

    client, _p, _calls, _c = await _call(monkeypatch, script, think=think)

    assert len(calls) == 1, _displays(calls)
    assert _phases(client, "d2") == []
    assert any(name == "live_status_answered" for name, _f in counters)


# ══════════════════════════════════════════════════════════════════════
# F2 — an accepted reask's answer fences the reasked turn
# ══════════════════════════════════════════════════════════════════════

CAPITAL = "What is the capital of Australia?"
FLIGHTS = "Find me flights to Canberra next week."
CANBERRA = "The capital is Canberra."


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_first", [True, False])
async def test_f2_a_reasked_and_answered_turn_is_not_absorbed_by_the_next_request(
    monkeypatch, audio_first,
):
    H.fast_clocks(monkeypatch)
    thinks: list[str] = []

    async def think(_user_id, _task, _session_id, **kwargs):
        thinks.append(kwargs.get("display_request"))
        return "Found flights.", "test-model"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(CAPITAL, 0, 900))
        elif (
            event["type"] == "session.instructions.append"
            and str(event.get("event_id") or "").startswith("toup-live-reask-")
        ):
            if audio_first:
                provider.push(H.out_audio("QUFB"))
                provider.push(H.out_text(CANBERRA, 1000, 1600))
            else:
                provider.push(H.out_text(CANBERRA, 1000, 1600))
                provider.push(H.out_audio("QUFB"))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def reask():
        await _until(lambda: _finals(client), label="Q1 final")
        client._script.insert(0, {
            "type": "inject_text", "text": CAPITAL, "reason": "no_response",
            "reask_of_user_turn_id": _finals(client)[0]["turn_id"],
        })

    async def answered_then_q2():
        await _until(lambda: any(
            CANBERRA in f.get("text", "") for f in client.of("response_text")
        ), label="answer on the phone")
        await asyncio.sleep(0.6)
        box["p"].push(H.user_delta(FLIGHTS, 7000, 7900))
        await _until(lambda: len(_finals(client)) >= 2, label="Q2 final")
        box["p"].push(H.delegation("d-flights", 7940))
        await _until(lambda: _created(client), label="created")
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(features=V03), reask, answered_then_q2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-f2-{audio_first}")

    finals = _finals(client)
    q1, q2 = finals[0]["turn_id"], finals[1]["turn_id"]
    assert [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")] == [
        (q1, "accepted"),
    ]
    assert any(
        f.get("parent_user_turn_id") == q1 and CANBERRA in (f.get("text") or "")
        for f in client.of("response_text")
    )
    created = _created(client)[0]
    assert created["title"] == FLIGHTS, created
    assert created["request_turn_ids"] == [q2]
    assert thinks == [FLIGHTS]


@pytest.mark.asyncio
async def test_f2_an_acknowledged_reask_still_binds_its_own_delegation(monkeypatch):
    """The positive twin: the model answers the reask with "Let me look that
    up." and then delegates.  The fence keeps the turn claimable by that same
    response's delegation (it is the open epoch's parent)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []
    request = "Find the best LLM professor at UofT."

    async def think(_user_id, _task, _session_id, **kwargs):
        thinks.append(kwargs.get("display_request"))
        return "An answer.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def reask():
        await _until(lambda: _finals(client), label="q1")
        client._script.insert(0, {
            "type": "inject_text", "text": request,
            "reask_of_user_turn_id": _finals(client)[0]["turn_id"], "reason": "no_response",
        })

    async def ack_then_delegate():
        p = box["p"]
        await _until(lambda: client.of("reask_result"), label="reask_result")
        p.push(H.out_audio("AAAA"))
        p.push(H.out_text(" Let me look that up.", 1000, 1600))
        await asyncio.sleep(0.02)
        p.push(H.delegation("d1", 1650))
        await _until(lambda: thinks, label="the request ran")
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(features=V03), reask, ack_then_delegate, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-f2-ack")

    assert thinks == [request]


# ══════════════════════════════════════════════════════════════════════
# F3 / F26 — authority is the anchor's own words, not a swept-in lead-in
# ══════════════════════════════════════════════════════════════════════

async def _lead_then_command(p, at, lead, command, *, lead_ms=(300, 500), command_at=1300):
    """An unanswered lead-in turn, a real pause, then the command and its
    delegation.  Returns nothing; the relay's frames are the evidence."""

    await at(lead_ms[0] / 1000)
    p.push(H.user_delta(lead, *lead_ms))
    await at(command_at / 1000)
    p.push(H.user_delta(command, command_at, command_at + 400))
    await at((command_at + 400 + 700) / 1000)
    p.push(H.delegation("d-cmd", command_at + 500))
    await at((command_at + 400 + 700) / 1000 + 1.2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lead", "command", "action"),
    [
        ("Hmm.", "Next song.", "next"),
        ("مرسی.", "آهنگ بعدی", "next"),
        ("Wait.", "Stop the music.", "stop"),
        ("One second.", "stop the music", "stop"),
        ("یه لحظه.", "آهنگ رو قطع کن", "stop"),
        ("ببین", "آهنگ رو قطع کن", "stop"),
        ("Hello?", "stop the music", "stop"),
    ],
)
async def test_f26_a_bare_media_command_after_an_unanswered_lead_in_is_executed(
    monkeypatch, lead, command, action,
):
    async def script(p, at):
        await _lead_then_command(p, at, lead, command)

    client, _p, calls, controls = await _call(monkeypatch, script)

    assert controls == [action], (controls, _displays(calls))
    assert calls == [], "a bare media command went to a full agent turn"
    created = _created(client, "d-cmd")
    assert [c["title"] for c in created] == [command.strip()]
    assert created[0]["turn_id"] == _final_id(client, command.strip())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lead", "command", "command_at"),
    [
        # A negation that is half of the same phrase is never dropped.
        ("Don't", "stop the music", 1300),
        # A question whose tail happens to be a media phrase.
        ("What is the name of the", "next song", 1800),
    ],
)
async def test_f26_a_lead_in_that_is_half_of_the_phrase_keeps_it_a_request(
    monkeypatch, lead, command, command_at,
):
    async def script(p, at):
        await _lead_then_command(p, at, lead, command, command_at=command_at)

    client, _p, calls, controls = await _call(monkeypatch, script)

    assert controls == []
    assert _displays(calls) == [f"{lead} {command}"], _displays(calls)


@pytest.mark.asyncio
async def test_f26_a_command_split_across_turns_is_still_one_command(monkeypatch):
    async def script(p, at):
        await _lead_then_command(p, at, "آهنگ رو", "قطع کن", command_at=1500)

    client, _p, calls, controls = await _call(monkeypatch, script)

    assert controls == ["stop"], _displays(calls)
    created = _created(client, "d-cmd")[0]
    assert created["title"] == "آهنگ رو قطع کن"
    assert len(created["request_turn_ids"]) == 2


@pytest.mark.parametrize(
    ("texts", "unit"),
    [
        # A lead-in that ends its own phrase is not part of the command.
        (["Hmm.", " Next song."], [" Next song."]),
        (["ببین", " آهنگ رو قطع کن"], [" آهنگ رو قطع کن"]),
        (["Okay,", " next song"], [" next song"]),
        (["Who is the dean there?", " What happened?"], [" What happened?"]),
        # A turn with no phrase end is half of the anchor's phrase.
        (["خب چی", " شد"], ["خب چی", " شد"]),
        (["آهنگ رو", " قطع کن"], ["آهنگ رو", " قطع کن"]),
        (["Don't", " stop the music"], ["Don't", " stop the music"]),
        (["No,", " only downtown."], ["No,", " only downtown."]),
        (["Find restaurants in Toronto", " not too expensive"],
         ["Find restaurants in Toronto", " not too expensive"]),
    ],
)
def test_f3_the_authority_unit_is_the_anchors_own_phrase(texts, unit):
    from app.services.live_voice_protocol import authority_unit

    turns = [_turn(i + 1, t, i * 2000, i * 2000 + 500) for i, t in enumerate(texts)]
    assert [t.text for t in authority_unit(turns)] == unit


def _running_first_task(release):
    calls: list[dict] = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, **kwargs})
        if len(calls) == 1:
            try:
                await asyncio.wait_for(release.wait(), timeout=15)
            except asyncio.CancelledError:
                raise
        return "An answer.", "test-model"

    return think, calls


async def _running_then(p, at, calls, lead, anchor, *, anchor_at=3300):
    end = await _say(p, at, ["Find", " apartments", " in", " Toronto."], wall_s=0.3, position_ms=300)
    p.push(H.delegation("d1", end + 40))
    await _until(lambda: calls, label="d1 running")
    await at(2.3)
    p.push(H.user_delta(lead, 2300, 2500))
    await at(anchor_at / 1000)
    p.push(H.user_delta(anchor, anchor_at, anchor_at + 400))
    await at((anchor_at + 400 + 700) / 1000)
    p.push(H.delegation("d2", anchor_at + 500))
    await at((anchor_at + 400 + 700) / 1000 + 1.0)


@pytest.mark.asyncio
async def test_f3_an_interjection_does_not_strip_a_correction_of_its_authority(monkeypatch):
    release = asyncio.Event()
    think, calls = _running_first_task(release)

    async def script(p, at):
        await _running_then(p, at, calls, "Hmm.", "No, only downtown.")
        release.set()
        await at(6.0)

    client, _p, _calls, _c = await _call(monkeypatch, script, think=think)

    lifecycle = [f for f in client.of("delegation") if f.get("delegation_id") == "d2" and "task_revision" in f]
    assert lifecycle and all(
        f.get("relation") == {"kind": "replaces", "task_id": "d1"} for f in lifecycle
    ), lifecycle
    assert _created(client, "d2")[0]["title"] == "No, only downtown."
    assert "superseded" in _phases(client, "d1")
    assert "Find apartments in Toronto." in str(calls[-1]["task"])


@pytest.mark.asyncio
async def test_f3_an_interjection_does_not_strip_a_cancel_of_its_authority(monkeypatch):
    release = asyncio.Event()
    think, calls = _running_first_task(release)

    async def script(p, at):
        await _running_then(p, at, calls, "Hmm.", "Cancel that.")
        await at(6.0)
        release.set()

    client, _p, _calls, _c = await _call(monkeypatch, script, think=think)

    assert _phases(client, "d2") == [], "a relay-answered cancel is not a card"
    assert _phases(client, "d1")[-1] in {"cancelled", "superseded"}, _phases(client, "d1")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_f3_a_lead_in_never_grants_authority_the_anchor_does_not_have(monkeypatch):
    """«No.» swept in front of an ordinary request must not make it a
    correction that supersedes the running task."""

    release = asyncio.Event()
    think, calls = _running_first_task(release)

    async def script(p, at):
        await _running_then(p, at, calls, "No.", "Find restaurants downtown.")
        release.set()
        await at(6.0)

    client, _p, _calls, _c = await _call(monkeypatch, script, think=think)

    assert "superseded" not in _phases(client, "d1")
    assert "cancelled" not in _phases(client, "d1")
    frames = [f for f in client.of("delegation") if f.get("delegation_id") == "d2"]
    assert frames and not any(f.get("relation") for f in frames), frames
    assert "No. Find restaurants downtown." in _displays(calls)


@pytest.mark.asyncio
async def test_f3_a_split_request_whose_second_half_reads_like_a_correction_stays_one_request(
    monkeypatch,
):
    release = asyncio.Event()
    think, calls = _running_first_task(release)

    async def script(p, at):
        await _running_then(p, at, calls, "Find restaurants in Toronto", "not too expensive")
        release.set()
        await at(6.0)

    client, _p, _calls, _c = await _call(monkeypatch, script, think=think)

    assert "superseded" not in _phases(client, "d1")
    assert "Find restaurants in Toronto not too expensive" in _displays(calls)


# ══════════════════════════════════════════════════════════════════════
# F4 — a stray interjection never picks the reply language
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    ("texts", "lang"),
    [
        (["خب", " Who works on LLMs at UofT?"], "en"),
        (["خب،", " پس", " Who works on LLMs at UofT?"], "en"),
        (["یه استاد پیدا کن که روی", " Large Language Models at University of Toronto"], "fa"),
        (["find the ر", "وبات", " lab at U of T"], "fa"),
        (["Geoffrey Hinton at University of Toronto", " کیه؟"], "fa"),
        (["Who works on LLMs at UofT?"], "en"),
    ],
)
def test_f4_the_reply_language_follows_the_anchors_own_phrase(texts, lang):
    from app.services.live_voice_protocol import LiveDelegationTracker

    turns = [
        _turn(index + 1, text, index * 2000, index * 2000 + 500)
        for index, text in enumerate(texts)
    ]
    tracker = LiveDelegationTracker()
    tracker.add_delegation("d1", len(texts) * 2000)
    task = tracker.consume("d1", turns[-1], span=turns, join_gap_ms=400)
    assert task is not None
    assert all(text.strip() in task.transcript for text in texts), task.transcript
    assert task.reply_language == lang


@pytest.mark.asyncio
async def test_f4_a_persian_interjection_before_an_english_request_gets_an_english_task(
    monkeypatch,
):
    async def script(p, at):
        await at(0.3)
        p.push(H.user_delta("خب", 300, 420))
        await _say(p, at, ["Who", " works", " on", " LLMs", " at", " UofT?"],
                   wall_s=1.2, position_ms=1200)
        p.push(H.delegation("d1", 2000))
        await at(3.5)

    client, _p, calls, _c = await _call(monkeypatch, script)

    assert len(calls) == 1
    assert calls[0]["reply_language"] == "en"
    assert "Persian" not in str(calls[0]["task"])
    assert calls[0]["display_request"] == "خب Who works on LLMs at UofT?"


# ══════════════════════════════════════════════════════════════════════
# F5 — the echo boundary keeps the caller's first words
# ══════════════════════════════════════════════════════════════════════

AGENT = (
    "the weather in toronto today is sunny with a light breeze and "
    "a high of twenty two degrees so it is a good day for a walk "
    "along the lake shore later this afternoon"
).split()
ECHO_OPEN = [" the weather in", " toronto today is", " sunny with a", " light breeze and"]
ECHO_LATE = [" the weather in", " toronto today is", " sunny with a", " light breeze and", " high of twenty"]


def _agent(words, start, step):
    async def run(p, at):
        for i, word in enumerate(words):
            await at(start + i * step)
            position = int((start + i * step) * 1000)
            p.push(H.out_text(" " + word, position, position + int(step * 1000)))
            p.push(H.out_audio("AAAA"))
    return run


def _echo(chunks, start, step):
    async def run(p, at):
        for k, chunk in enumerate(chunks):
            wall = start + k * step
            await at(wall)
            position = int(wall * 1000) - 150
            p.push(H.user_delta(chunk, position, position + 150))
    return run


def _caller(words, start, step=0.12):
    async def run(p, at):
        position = int(start * 1000)
        for k, word in enumerate(words):
            await at(start + k * step)
            p.push(H.user_delta(word, position, position + 120))
            position += 120
        p.push(H.delegation("d-echo", position + 40))
    return run


ECHO_CASES = {
    "content_head": ((AGENT, 0.2, 0.05), (ECHO_OPEN, 0.6, 0.2),
                     ["Search", " the", " Robarts", " library", " opening", " hours."], 1.5),
    "en_stopword_head": ((AGENT, 0.2, 0.05), (ECHO_OPEN, 0.6, 0.2),
                         ["Do", " I", " have", " a", " meeting", " with", " Sara", " tomorrow?"], 1.5),
    "fa_stopword_head": ((AGENT, 0.2, 0.05), (ECHO_OPEN, 0.6, 0.2),
                         ["با", " کی", " فردا", " جلسه", " دارم؟"], 1.5),
    # Generation faster than real time: the epoch gap-retires while the echo
    # is still arriving, then the caller speaks (c) or first waits past the
    # utterance gap (c2).  The agent's words never become the caller's.
    "late_echo_after_retire": ((AGENT[:20], 0.2, 0.02), (ECHO_LATE, 0.6, 0.2),
                               ["Search", " the", " Robarts", " library", " opening", " hours."], 1.7),
    "late_echo_then_silence": ((AGENT[:20], 0.2, 0.02), (ECHO_LATE, 0.6, 0.2),
                               ["Search", " the", " Robarts", " library", " opening", " hours."], 2.3),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(ECHO_CASES))
async def test_f5_the_callers_first_words_after_echo_stay_with_the_caller(monkeypatch, case):
    agent, echo, words, caller_at = ECHO_CASES[case]

    async def script(p, at):
        await asyncio.gather(_agent(*agent)(p, at), _echo(*echo)(p, at), _caller(words, caller_at)(p, at))
        await at(4.0)

    client, _p, calls, _c = await _call(monkeypatch, script)

    request = "".join(words).strip()
    assert _displays(calls) == [request], ([f["text"] for f in _finals(client)], _displays(calls))
    assert _finals(client)[-1]["text"] == request


# ══════════════════════════════════════════════════════════════════════
# F6 — a backchannel's punctuation never holds the agent's reply
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", ["Mm-hm,", "Hmm...", "آره،", "Okay,", "Um,", "hm -"])
def test_f6_a_content_free_turn_is_never_an_unfinished_clause(text):
    from app.services.live_voice_protocol import looks_unfinished

    assert not looks_unfinished(text)


@pytest.mark.parametrize(
    "text", ["Find me the professor,", "Um, find me the", "and...", "آهنگ رو،", "What the most, uh..."],
)
def test_f6_a_real_clause_keeps_its_structural_tail(text):
    from app.services.live_voice_protocol import looks_unfinished

    assert looks_unfinished(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("backchannel", ["Mm-hm,", "Hmm...", "آره،"])
async def test_f6_a_backchannel_over_the_reply_holds_it_no_longer_than_the_gap(
    monkeypatch, backchannel,
):
    words = ("the library opens at nine in the morning and closes at ten at night on "
             "weekdays and it keeps shorter hours on the weekend so plan ahead").split()

    async def script(p, at):
        await _say(p, at, ["What", " time", " does", " the", " library", " open?"],
                   wall_s=0.2, position_ms=200)

        async def agent():
            for i in range(60):
                await at(1.4 + i * 0.05)
                position = 1500 + i * 50
                p.push(H.out_text(" " + words[i % len(words)], position, position + 50))
                p.push(H.out_audio("AAAA"))

        async def caller():
            await at(1.4 + 0.6)
            p.push(H.user_delta(backchannel, 1950, 2200))

        await asyncio.gather(agent(), caller())
        await at(1.4 + 60 * 0.05 + 1.5)

    client, _p, _calls, _c = await _call(monkeypatch, script)

    audio = [t for f, t in zip(client.frames, client.times) if f.get("type") == "audio_delta"]
    assert len(audio) == 60
    longest = max(b - a for a, b in zip(audio, audio[1:]))
    # gap 400 + settle 40 + one 250 ms tick, with slack: never the 900 ms hard gap.
    assert longest < 0.8, f"the reply stalled {longest:.2f}s behind «{backchannel}»"
    assert not client.of("speech_started")


# ══════════════════════════════════════════════════════════════════════
# F15 — a queued task whose anchor aged out still starts (or fails honestly)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_f15_a_queued_task_starts_after_more_than_twenty_turns(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_max_concurrent_delegations=2)
    release = asyncio.Event()
    thinks: list[str] = []
    box: dict = {}

    async def think(_user_id, task, _session_id, **_kwargs):
        thinks.append(str(task))
        if len(thinks) <= 2:
            await release.wait()
        return "Done.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("book the flight to Paris", 0, 300))
            provider.push(H.delegation("d1", 320))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def drive():
        await asyncio.sleep(0.25)
        p = box["p"]
        p.push(H.user_delta("what temperature is it outside", 2000, 2400))
        p.push(H.delegation("d2", 2420))
        await asyncio.sleep(0.25)
        p.push(H.user_delta("book a table at a thai place downtown", 4000, 4600))
        p.push(H.delegation("d3", 4620))
        await asyncio.sleep(0.3)
        position = 6000
        for i in range(24):
            p.push(H.user_delta(f"okay sure number {i}", position, position + 200))
            position += 1000
            await asyncio.sleep(0.03)
        await asyncio.sleep(0.4)
        release.set()
        await _wait_quietly(lambda: len(thinks) >= 3, timeout=3.0)
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(), drive, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=15)

    d3 = _phases(client, "d3")
    assert d3[:2] == ["created", "pending"], d3
    assert "started" in d3 and d3[-1] in {"completed", "failed"}, d3
    assert len(thinks) == 3


# ══════════════════════════════════════════════════════════════════════
# F18 / F37 (relay half) — an echo-only turn says so on the wire
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_f18_echo_only_transcript_frames_are_marked_and_caller_frames_are_not(monkeypatch):
    async def script(p, at):
        await asyncio.gather(
            _agent(AGENT, 0.2, 0.05)(p, at),
            _echo(ECHO_OPEN, 0.6, 0.2)(p, at),
            _caller(["Search", " the", " Robarts", " library", " opening", " hours."], 1.8)(p, at),
        )
        await at(4.0)

    client, _p, calls, _c = await _call(monkeypatch, script)

    transcripts = client.of("transcript")
    echo_id = next(f["turn_id"] for f in transcripts if f["text"].startswith("the weather"))
    echo_frames = [f for f in transcripts if f["turn_id"] == echo_id]
    assert echo_frames and all(f.get("echo_only") is True for f in echo_frames), echo_frames
    assert echo_frames[-1].get("final")
    caller_frames = [f for f in transcripts if f["turn_id"] != echo_id]
    assert caller_frames and not any("echo_only" in f for f in caller_frames), caller_frames
    assert _displays(calls) == [ROBARTS]
