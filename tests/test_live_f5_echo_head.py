"""F5 (round 2): the caller who opens with the agent's own words keeps them.

The supervisor's reproduction on integrated sha256 980fe1d4…: the agent says
"the robarts library opens at nine in the morning and closes at ten tonight",
its echo comes back through the hot mic as an echo-only turn, and while that
turn is still open the caller asks "Robarts library hours on Sunday?".  The
first delta ("Robarts library") repeated the agent's words, so the bag-of-words
echo verdict kept it in the echo turn: the task ran as "hours on Sunday?" and
the echo turn's final read "…ten tonightRobarts library" — with the epoch open
(caller at 1.3 s) and after it retired (1.55 s) alike.

Vocabulary cannot separate the two; what the relay CAN observe is when and
where the agent's words could be coming back (`EchoEvidence`):

* each word the phone played echoes ONCE — "robarts library" had already come
  back 800 ms earlier on the input timeline, so these words are the caller's;
* the phone's receipts (`playback_idle` / `interrupt` / `playback_failed`)
  end an epoch's playback — after one (+ a bounded tail) its words cannot echo;
* echo lands at a stable lag behind the provider position of the word it
  repeats — measured on this call's echo — which is evidence, not proof.

Proof makes the words the caller's.  Position alone cannot, so a fragment only
position doubts stays echo, and — when the caller's own words run straight on
from it — rides into the request as MARKED, uncertain possible-head context,
never as the caller's words or title (the documented residual).

Controls: echo during playback stays echo (including a phrase the agent says
twice), a function-word echo tail stays echo, late echo after the epoch
retires stays echo, the F0 Sequences A/B/C, and replays run nothing twice.

Clocks are the §A scale (settle 40 / gap 400 / hard gap 900 / epoch gap 250).
"""

import asyncio
import inspect
import json
import time

import pytest

import test_live_harness as H
import app.services.live_voice_protocol as live
from test_live_endpoint_clock import AGENT_WORDS, ECHO, REQUEST, _agent_speaks, _echo_comes_back


CLOCKS = dict(
    voice_live_transcript_settle_ms=40,
    voice_live_utterance_gap_ms=400,
    voice_live_utterance_hard_gap_ms=900,
    voice_live_output_epoch_gap_ms=250,
    voice_live_delegation_ttl_s=3.0,
    voice_live_interrupt_suppress_ms=300,
)


class Phone(H.FakeClient):
    """The phone; a script item ``("frame", make)`` sends ``make()`` built at
    that moment (a receipt must name the epoch the relay minted)."""

    async def receive_text(self):
        while self._script:
            item = self._script.pop(0)
            self.reads += 1
            if isinstance(item, tuple) and item and item[0] == "frame":
                return json.dumps(item[1]())
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
                continue
            if callable(item):
                result = item()
                if inspect.isawaitable(result):
                    await result
                continue
            return json.dumps(item)
        await asyncio.sleep(3600)


async def _call(monkeypatch, script, *, receipts=()):
    """One relay call.  `script(provider, at)` runs on wall time since
    `session.start`; `receipts` is [(wall_s, make_frame)] the phone sends
    while it runs.  Returns (phone, [{"task", "display"}])."""

    H.fast_clocks(monkeypatch, **CLOCKS)
    calls: list[dict] = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, "display": kwargs.get("display_request")})
        return "An answer.", "test-model"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

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

    def start():
        box["run"] = asyncio.ensure_future(script(box["p"], at))

    items: list = [H.config(), start]
    for wall_s, make in receipts:
        items.append(lambda wall_s=wall_s: at(wall_s))
        items.append(("frame", lambda make=make: make(box["client"])))
    items.append(lambda: box["run"])
    items.append({"type": "stop"})
    client = Phone(items)
    box["client"] = client
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=30.0, db_session_id=f"db-f5-echo-head-{time.monotonic_ns()}",
    )
    return client, calls


def _finals(client):
    return [(f["text"], f.get("echo_only")) for f in client.of("transcript") if f.get("final")]


def _created(client):
    """The task cards' titles: the words a card shows are the certain words."""

    return [f.get("title") for f in client.of("delegation") if f.get("phase") == "created"]


def _agent(words, *, wall_s=0.2, step_s=0.075, position_ms=200):
    async def run(p, at):
        for i, word in enumerate(words):
            await at(wall_s + i * step_s)
            position = position_ms + int(i * step_s * 1000)
            p.push(H.out_text(" " + word, position, position + int(step_s * 1000)))
            p.push(H.out_audio("AAAA"))
    return run


def _echo(chunks, *, wall_s=0.5, step_s=0.18, lag_ms=150):
    async def run(p, at):
        for k, chunk in enumerate(chunks):
            wall = wall_s + k * step_s
            await at(wall)
            position = int(wall * 1000) - lag_ms
            p.push(H.user_delta(chunk, position, position + lag_ms))
    return run


def _caller(pieces, *, wall_s, delegation="d1", step_s=0.12, span_ms=120, replay=()):
    async def run(p, at):
        position = int(wall_s * 1000)
        for k, piece in enumerate(pieces):
            await at(wall_s + k * step_s)
            p.push(H.user_delta(piece, position, position + span_ms))
            if k in replay:
                p.push(H.user_delta(piece, position, position + span_ms))
            position += span_ms
        p.push(H.delegation(delegation, position + 40))
    return run


def _script(*parts, until_s=4.0):
    async def script(p, at):
        await asyncio.gather(*(part(p, at) for part in parts))
        await at(until_s)
    return script


def _playback_idle(client):
    """The phone finished playing the newest epoch it was sent."""

    epoch = client.of("audio_delta")[-1]["response_id"]
    return {"type": "playback_idle", "response_id": epoch, "item_id": epoch}


# The supervisor probe, event for event
# (/private/tmp/toup-voice-r2-review/probes/endpoint-binding/test_probe_echo_overlap_head.py).
LIBRARY = "the robarts library opens at nine in the morning and closes at ten tonight".split()
LIBRARY_ECHO = [" the robarts library", " opens at nine", " in the morning", " and closes at", " ten tonight"]
ASK = ["Robarts library", " hours on", " Sunday?"]
ASK_TEXT = "Robarts library hours on Sunday?"


# ══════════════════════════════════════════════════════════════════════
# The reproduction: the caller's head words are the caller's
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("caller_at", [1.3, 1.55])
async def test_the_callers_words_that_repeat_the_agent_stay_with_the_caller(monkeypatch, caller_at):
    """Epoch still open (1.3) and retired (1.55): "robarts library" already
    came back as echo, so the same words 800 ms later are the caller's."""

    client, calls = await _call(monkeypatch, _script(
        _agent(LIBRARY), _echo(LIBRARY_ECHO), _caller(ASK, wall_s=caller_at),
    ))

    assert [c["display"] for c in calls] == [ASK_TEXT], (_finals(client), calls)
    assert _finals(client) == [(" ".join(LIBRARY), True), (ASK_TEXT, None)]
    assert _created(client) == [ASK_TEXT]
    # Proven, not guessed: nothing is carried as an uncertain head.
    assert "UNCERTAIN" not in calls[0]["task"]


@pytest.mark.asyncio
async def test_the_callers_words_after_the_phone_reported_playback_over(monkeypatch):
    """Words the agent said but whose echo never came back: after the phone's
    `playback_idle` (+ the tail) they cannot be echo — the caller is repeating
    them.  "morning" was said, never echoed; the caller asks about it."""

    # The scaled clocks' tail (this harness has no transcription delay).
    monkeypatch.setattr(live, "_ECHO_RECEIPT_TAIL_MS", 100, raising=False)
    skipped = [" the robarts library", " opens at nine", " and closes at", " ten tonight"]
    ask = ["Morning", " hours on", " Sunday?"]
    client, calls = await _call(
        monkeypatch,
        _script(_agent(LIBRARY, step_s=0.05), _echo(skipped, wall_s=0.5), _caller(ask, wall_s=1.3)),
        receipts=[(1.0, _playback_idle)],
    )

    assert [c["display"] for c in calls] == ["Morning hours on Sunday?"], (_finals(client), calls)
    assert _finals(client)[-1] == ("Morning hours on Sunday?", None)
    assert "UNCERTAIN" not in calls[0]["task"]


@pytest.mark.asyncio
async def test_without_a_receipt_the_same_words_are_an_uncertain_head_not_a_certain_one(monkeypatch):
    """The receipt's twin with no `playback_idle`: nothing PROVES "Morning" is
    not the echo of the agent's "morning" — it only lands later than this
    call's echo does.  It stays in the echo turn, and the request carries it
    as marked, uncertain context: never as the caller's words, never in the
    title — the task is not handed over as if "hours on Sunday?" were whole."""

    skipped = [" the robarts library", " opens at nine", " and closes at", " ten tonight"]
    ask = ["Morning", " hours on", " Sunday?"]
    client, calls = await _call(monkeypatch, _script(
        _agent(LIBRARY, step_s=0.05), _echo(skipped, wall_s=0.5), _caller(ask, wall_s=1.3),
    ))

    assert [c["display"] for c in calls] == ["hours on Sunday?"], (_finals(client), calls)
    assert _created(client) == ["hours on Sunday?"]
    task = calls[0]["task"]
    assert "UNCERTAIN" in task and "«Morning»" in task, task
    assert task.index("«Morning»") < task.index("Current caller request: hours on Sunday?")


# ══════════════════════════════════════════════════════════════════════
# The documented residual: ambiguous by what the relay can observe
# ══════════════════════════════════════════════════════════════════════

WEATHER = AGENT_WORDS[:20]


@pytest.mark.parametrize("shape", ["retired_repeat", "open_repeat"])
@pytest.mark.asyncio
async def test_an_ambiguous_head_rides_as_marked_context_never_as_the_callers_words(monkeypatch, shape):
    """The verifiers' (b)/(d) residual.  The caller opens with words the agent
    said that have NOT yet come back ("Twenty two degrees" while the echo had
    reached "light breeze"; "Is it open" while a second "opens" was still
    unechoed).  Nothing proves them the caller's, and where they land only
    argues against echo — the relay cannot know.  So they stay in the echo
    turn, and the request is not handed over as if it were whole: it carries
    them as marked possible-head context."""

    if shape == "retired_repeat":
        agent = _agent(WEATHER, step_s=0.05)
        echo = _echo([" the weather in", " toronto today is", " sunny with a", " light breeze and"])
        ask, caller_at = ["Twenty two degrees", " in", " Montreal", " too?"], 1.55
        certain, possible = "in Montreal too?", "Twenty two degrees"
    else:
        agent = _agent((LIBRARY * 3)[:30], step_s=0.05)
        echo = _echo(LIBRARY_ECHO)
        ask, caller_at = ["Is", " it", " open", " on", " Sunday?"], 1.5
        certain, possible = "on Sunday?", "Is it open"

    client, calls = await _call(monkeypatch, _script(agent, echo, _caller(ask, wall_s=caller_at), until_s=4.5))

    assert [c["display"] for c in calls] == [certain], (_finals(client), calls)
    assert _created(client) == [certain]
    assert _finals(client)[-1] == (certain, None)
    task = calls[0]["task"]
    assert f"«{possible}»" in task and "UNCERTAIN" in task, task
    assert task.index(f"«{possible}»") < task.index(f"Current caller request: {certain}")
    # The uncertain words stayed where they were heard: in the echo turn.
    assert possible in _finals(client)[0][0] and _finals(client)[0][1] is True


# ══════════════════════════════════════════════════════════════════════
# Controls: echo stays echo
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_echo_of_a_phrase_the_agent_says_twice_stays_echo(monkeypatch):
    """True echo during playback, including the SECOND "robarts library": each
    occurrence echoes once, so the repeat is the second occurrence coming back,
    not the caller."""

    said = "the robarts library is open today and the robarts library closes at ten".split()
    echo = [" the robarts library", " is open today", " and the robarts library", " closes at ten"]
    ask = ["Search", " the", " Gerstein", " hours."]
    client, calls = await _call(monkeypatch, _script(
        _agent(said, step_s=0.05), _echo(echo), _caller(ask, wall_s=1.5),
    ))

    assert _finals(client) == [(" ".join(said), True), ("Search the Gerstein hours.", None)]
    assert [c["display"] for c in calls] == ["Search the Gerstein hours."]
    assert "UNCERTAIN" not in calls[0]["task"]


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", sorted(ECHO))
async def test_echo_and_a_function_word_echo_tail_stay_echo(monkeypatch, echo):
    """The committed §A shapes: content echo and an echoed function tail
    ("… it is a"), then the caller after a pause.  The echo keeps every word;
    the request is only the caller's; a pause means no possible head."""

    async def voices(p, at):
        await _echo_comes_back(p, at, echo)
        position = 1800
        for k, piece in enumerate(REQUEST):
            await at(1.8 + k * 0.12)
            p.push(H.user_delta(piece, position, position + 120))
            position += 120
        p.push(H.delegation("d-after-echo", position + 40))

    client, calls = await _call(monkeypatch, _script(_agent_speaks, voices, until_s=4.5))

    request = "".join(REQUEST)
    finals = _finals(client)
    assert finals[-1] == (request, None)
    assert finals[0][0] == "".join(ECHO[echo]).strip() and finals[0][1] is True, finals
    assert [c["display"] for c in calls] == [request]
    assert calls[0]["task"].startswith(request)


@pytest.mark.asyncio
async def test_late_echo_after_the_epoch_retired_stays_echo(monkeypatch):
    """Generation faster than playback: the epoch gap-retires while its echo is
    still arriving, and the echo's lag grows as it trails.  Still echo — it
    never becomes the caller's, never prefixes the request."""

    late = [" the weather in", " toronto today is", " sunny with a", " light breeze and", " high of twenty"]
    client, calls = await _call(monkeypatch, _script(
        _agent(WEATHER, step_s=0.02), _echo(late, wall_s=0.6, step_s=0.2),
        _caller(["Search", " the", " Robarts", " library", " opening", " hours."], wall_s=1.7),
    ))

    assert [c["display"] for c in calls] == ["Search the Robarts library opening hours."]
    assert _finals(client)[0] == ("".join(late).strip(), True)
    assert "UNCERTAIN" not in calls[0]["task"]


# ══════════════════════════════════════════════════════════════════════
# Controls: F0 Sequences A/B/C and replays
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_sequence_a_the_echoed_ack_is_never_the_request(monkeypatch):
    import test_live_f0_direct_parent as F0

    client, calls = await F0._run(monkeypatch, F0._gather(
        F0.REQUEST_TURN, F0.ACK_SPOKEN, F0.ACK_ECHOED,
        F0._delegate("d-ack", wall_s=2.45, offset_ms=2420),
        until_s=5.5,
    ))

    request = F0._turn_of(client, "Search the Robarts")
    assert calls == [F0.REQUEST_TEXT]
    assert F0._created(client) == [("d-ack", F0.REQUEST_TEXT, request, [request])]
    echo = F0._turn_of(client, "sure let me look")
    assert dict((t, e) for t, _x, e in F0._finals(client))[echo] is True


@pytest.mark.asyncio
async def test_sequence_b_the_question_over_the_reply_runs_alone(monkeypatch):
    import test_live_f0_direct_parent as F0

    ask = ["Find", " the", " Sunday", " hours", " for", " Gerstein."]
    client, calls = await F0._run(monkeypatch, F0._gather(
        F0._speak(F0.AGENT, wall_s=0.2, position_ms=200, step_s=0.08, span_ms=80),
        F0._say(ask, wall_s=0.9, position_ms=900, step_s=0.05, span_ms=50),
        F0._echo([" except on holidays", " when hours are", " shorter"],
                 wall_s=1.85, step_s=0.12, lag_ms=120, span_ms=120),
        F0._delegate("dB", wall_s=2.3, offset_ms=2200),
        until_s=5.0,
    ))

    assert calls == ["Find the Sunday hours for Gerstein."], calls


@pytest.mark.asyncio
async def test_sequence_c_the_reply_answers_the_question_not_the_echo(monkeypatch):
    import test_live_f0_direct_parent as F0

    client, _calls = await F0._run(monkeypatch, F0._gather(*F0.SEQUENCE_C, until_s=5.0))

    question = F0._turn_of(client, "What about")
    echo = F0._turn_of(client, "except on holidays")
    reply_parents = sorted({
        f.get("parent_user_turn_id") for f in client.frames
        if f.get("parent_user_turn_id") and "On" in str(f.get("text") or "")
    })
    assert reply_parents == [question]
    assert [(t, e) for t, _x, e in F0._finals(client)] == [(question, None), (echo, True)]


@pytest.mark.asyncio
async def test_a_caller_who_repeats_unechoed_agent_words_over_the_reply_is_still_the_fallback(monkeypatch):
    """F0's kept fallback is unchanged: words the agent is still saying, which
    have not come back yet, are judged echo where echo lands — the turn stays
    `echo_only`, and the delegation's last resort still runs it once."""

    import test_live_f0_direct_parent as F0

    offer = "I can look up the Robarts library opening hours for you if you like.".split()
    repeat = [" look up the", " robarts library", " opening hours"]
    client, calls = await F0._run(monkeypatch, F0._gather(
        F0._speak(offer, wall_s=0.2, position_ms=200, step_s=0.08, span_ms=80),
        F0._say(repeat, wall_s=1.0, position_ms=1000, step_s=0.12, span_ms=120),
        F0._speak("Sure, one moment.".split(), wall_s=2.4, position_ms=2400,
                  step_s=0.05, span_ms=60, audio="BBBB"),
        F0._delegate("d-rep", wall_s=2.6, offset_ms=2550),
        F0._delegate("d-rep", wall_s=3.4, offset_ms=2550),
        until_s=5.0,
    ))

    repeated = F0._turn_of(client, "look up the robarts")
    assert calls == ["look up the robarts library opening hours"], calls
    assert dict((t, e) for t, _x, e in F0._finals(client))[repeated] is True


@pytest.mark.asyncio
async def test_replayed_fragments_and_a_replayed_delegation_run_the_request_once(monkeypatch):
    """The reproduction with the caller's first delta replayed verbatim and the
    delegation replayed: one turn, one execution, the whole request."""

    async def replay_delegation(p, at):
        await at(2.6)
        p.push(H.delegation("d1", 1300 + 3 * 120 + 40))

    client, calls = await _call(monkeypatch, _script(
        _agent(LIBRARY), _echo(LIBRARY_ECHO), _caller(ASK, wall_s=1.3, replay=(0, 1)),
        replay_delegation,
    ))

    assert [c["display"] for c in calls] == [ASK_TEXT], calls
    assert _finals(client) == [(" ".join(LIBRARY), True), (ASK_TEXT, None)]
    assert _created(client) == [ASK_TEXT]


# ══════════════════════════════════════════════════════════════════════
# The evidence, directly
# ══════════════════════════════════════════════════════════════════════

def _adapter(words, *, step=50, start=200):
    adapter = live.LiveClientAdapter("s")
    for i, word in enumerate(words):
        position = start + i * step
        adapter.provider_event(H.out_text(" " + word, position, position + step))
    return adapter


def test_a_word_whose_echo_already_came_back_cannot_come_back_again():
    adapter = _adapter(LIBRARY)
    first = adapter.echo_evidence(" the robarts library", 350, 500)
    assert first.kind == "echo"
    adapter.note_echo(first, 350, 500)

    again = adapter.echo_evidence("Robarts library", 1300, 1420)
    assert (again.kind, again.exonerated, again.vocab, again.strong) == ("caller", True, 1.0, 0.0)
    # The SAME audio transcribed twice (overlapping input spans) is not proof.
    overlap = adapter.echo_evidence(" robarts library", 400, 520)
    assert overlap.kind == "echo"


def test_the_phones_receipt_ends_an_epochs_echo():
    adapter = _adapter(LIBRARY)
    adapter.rotate_on_gap(now=time.monotonic() + 10)
    (closed,) = adapter._history
    closed.receipt_monotonic = 100.0

    def over(epoch):
        return 1000 if epoch.receipt_monotonic else None

    before = adapter.echo_evidence("morning", 900, 1000, playback_over_ms=over)
    after = adapter.echo_evidence("morning", 1100, 1200, playback_over_ms=over)
    assert before.kind == "echo"
    assert (after.kind, after.exonerated) == ("caller", True)


def test_position_alone_is_doubt_never_proof():
    adapter = _adapter(LIBRARY)
    for text, start in ((" the robarts library", 350), (" opens at nine", 530)):
        evidence = adapter.echo_evidence(text, start, start + 150)
        assert evidence.kind == "echo"
        adapter.note_echo(evidence, start, start + 150)

    # "morning" (said at 600-650) arriving 1.2 s after it: far off this call's
    # echo lag, but nothing PROVES it is not the echo.
    late = adapter.echo_evidence("morning", 1800, 1900)
    assert (late.kind, late.exonerated) == ("ambiguous", False)
    # Where echo lands, it is echo.
    on_time = adapter.echo_evidence(" in the morning", 710, 860)
    assert on_time.kind == "echo"


def test_without_positions_or_occurrences_nothing_is_proven_either_way():
    adapter = live.LiveClientAdapter("s")
    adapter.provider_event(H.out_text(" the robarts library opens"))   # no positions
    evidence = adapter.echo_evidence("robarts library", 1300, 1420)
    assert (evidence.kind, evidence.vocab) == ("echo", 1.0)
    adapter.note_echo(evidence, 1300, 1420)
    # Untimed words calibrate nothing; the next echo is still echo.
    assert adapter._echo_lags == []
    assert adapter.echo_evidence(" opens", 1500, 1600).kind == "echo"
    # A token known only through held (deferred) text stays "could be echo".
    held = live.LiveClientAdapter("h")
    held.note_deferred_output(" the weather in toronto")
    assert held.echo_evidence("toronto weather", 100, 200).kind == "echo"


def test_the_bag_of_words_verdict_is_unchanged():
    reference = frozenset(live._echo_reference_tokens("the robarts library opens at nine"))
    assert live.echo_overlap("Robarts library hours on Sunday?", reference) == (4, 0.5)
    assert live.echo_overlap(" it is a", reference) == (0, 0.0)
    assert live.echo_overlap("robarts", reference, unfinished_tail=True) == (1, 1.0)


def test_the_possible_head_is_labelled_uncertain_and_never_the_request():
    plain = live.delegated_agent_input("hours on Sunday?", [])
    assert plain == "hours on Sunday?"
    marked = live.delegated_agent_input("hours on Sunday?", [], possible_head="Robarts  library")
    assert marked.endswith("Current caller request: hours on Sunday?")
    assert "UNCERTAIN" in marked and "«Robarts library»" in marked
    assert marked.index("«Robarts library»") < marked.index("Current caller request:")
    # Bounded.
    long = live.delegated_agent_input("x?", [], possible_head="word " * 100)
    assert len(long) < 600


def test_the_possible_head_comes_from_the_first_turn_of_the_request():
    first = live.Utterance(turn_id="u1", ordinal=1, start_ms=0, end_ms=1, text="in", possible_head="Twenty")
    second = live.Utterance(turn_id="u2", ordinal=2, start_ms=2, end_ms=3, text="Montreal")
    task = live.DelegatedTask(delegation_id="d", offset_ms=0, transcript="in Montreal")
    task.span = [second, first]
    assert live.request_possible_head(task) == "Twenty"
    task.span = [second]
    assert live.request_possible_head(task) == ""
