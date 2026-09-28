"""R2 round 3 (addendum 3, the independent final critic): the relay half.

Evidence: /private/tmp/toup-voice-r2-review/final-matrix.json (regressions R1,
R2, R3 and the V3 split-stop gap) and the critic's probes under
/private/tmp/toup-voice-r2-review/probes/final-critic/.

  item 1  Phatic turns are not requests.  A presence check ("can you hear
          me?", «هستی؟»), a continuer ("okay go on", «ادامه بده») or a closing
          ("okay thanks", «مرسی عالیه») never overtakes a check-in — the
          status path dispatched «هستی؟» as research, with a card titled by
          it, and never answered the check-in (R1) — and is never a reask
          target (R3: the relay accepted a reask of "okay thanks").  "is it
          done?" / «تموم شد؟» / «کارت تموم شد؟» are check-ins.
  item 2  Final transcript frames of a phatic turn carry additive
          `"phatic": true` (gated on `live_turns`, like `echo_only`).
  item 3  The media backstop fires only on a command that names its media
          object, read over the anchor's authority unit, and only while the
          phone reports playback: a bare "next" answering the agent's own
          question skipped the track (R2), a named command with nothing
          playing still called the tenant, and the recorded V3 split stop
          («آهنگ رو نمی‌تونی» + «قطع») stopped nothing.
  item 4  A correction of a FINISHED task needs the same evidence as one of a
          running task: "no, what time is it in tokyo?" after an answer does
          not replace that answer's task.

Everything that is behaviour is driven through the real relay
(`test_live_harness`) with provider-shaped events; the pure seams that exist on
every build (`_LiveSession.is_request_turn`, `is_status_question`,
`_LiveSession.modifies_record`) are called directly, so each test fails on the
pre-round-3 relay by ASSERTION, not by an import error.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import test_live_harness as H
from app.config import settings
from app.services import live_voice_protocol as live


BASE = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
]
V03 = BASE + ["media_transport", "reask_turns"]
TF132 = [
    "delegation_frames", "heard_text", "live_turns", "media_control",
    "playback_frames", "task_lifecycle", "turn_timing",
]
RESEARCH = "research the toronto robotics faculty"
SUBJECT = "find Iranian computer science professors at UofT downtown who work on robotics"

#: Addendum 3 item 1's closed class, en + fa, as the addendum lists it, plus
#: the critic's probe_is_request_turn backchannels that are phatic.
PHATIC = [
    # presence
    "are you there?", "can you hear me?", "hello?", "Hello? Hello?",
    "هستی؟", "الو", "صدامو داری؟",
    # continuers
    "okay go on", "go on", "mm-hm", "keep going", "ادامه بده", "باشه ادامه بده", "خب",
    # closings
    "okay thanks", "thanks", "great, thank you", "thank you so much", "alright thanks",
    "مرسی", "مرسی عالیه", "ممنون", "باشه ممنون", "اوکی مرسی", "دستت درد نکنه",
    # backchannels and holds
    "got it", "sounds good", "cool", "that's great", "I see", "no worries",
    "hold on", "one sec", "hmm let me think", "آهان", "عالیه", "خیلی خوبه", "صبر کن", "یه لحظه",
]

#: CONTROLS: requests (and answers that ARE requests) stay requests.
REQUESTS = [
    "find the dean of engineering at toronto",
    "thanks, and find the dean of engineering",
    "هستی؟ ساعت توکیو چنده",
    "یه استاد رباتیک تو تورنتو پیدا کن",
    "can you hear the music in the video",
    "go to the toronto admissions page",
]


def _turn(text: str, n: int = 1) -> "live.Utterance":
    return live.Utterance(
        turn_id=f"U:{n}", ordinal=n, start_ms=0, end_ms=500, text=text,
        closed=True, causal_speech=True,
    )


def _frames_for(client, did):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _terminal(client, did):
    return any(
        f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
        for f in _frames_for(client, did)
    )


def _lifecycle(frames):
    return [f for f in frames if "task_revision" in f]


def _finals(client) -> list[dict]:
    return [f for f in client.of("transcript") if f.get("final")]


def _send(client, frame) -> None:
    """Queue `frame` as the next thing the phone sends (turn ids exist only at
    run time)."""

    client._script.insert(0, frame)


async def _wait(pred, timeout=4.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


# ══════════════════════════════════════════════════════════════════════
# item 1 — phatic turns are not requests
# ══════════════════════════════════════════════════════════════════════

def test_phatic_turns_are_never_requests_and_requests_still_are():
    """The status path's own test (`is_request_turn`): which words the caller
    is waiting on.  The critic measured 25/35 common backchannels as requests."""

    judged_requests = [
        text for text in PHATIC
        if live._LiveSession.is_request_turn(None, _turn(text), SUBJECT)
    ]
    assert judged_requests == [], f"phatic turns read as requests: {judged_requests}"
    missed = [
        text for text in REQUESTS
        if not live._LiveSession.is_request_turn(None, _turn(text), SUBJECT)
    ]
    assert missed == [], f"real requests no longer read as requests: {missed}"


def test_is_it_done_is_a_check_in_and_a_request_holding_it_is_not():
    for check_in in ("is it done?", "تموم شد؟", "کارت تموم شد؟", "are you done?"):
        assert live.is_status_question(check_in, SUBJECT), check_in
        assert live.classify_followup(check_in, RESEARCH, status_context=SUBJECT) == "status"
    for request in (
        "is it done raining in Toronto?", "what happened in Iran today",
        "وقتی تموم شد بهم ایمیلش کن",
    ):
        assert not live.is_status_question(request, SUBJECT), request


def _opener(box, first=RESEARCH, first_id="d1"):
    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(first, 0, 900))
            p.push(H.delegation(first_id, 950))
        elif e["type"] == "session.close":
            p.push(H.closed())
    return on_send


@pytest.mark.asyncio
@pytest.mark.parametrize("gap_ms", [11000, 1500], ids=["outside_span", "inside_span"])
@pytest.mark.parametrize(("filler", "check_in"), [
    ("are you there", "is it done?"),
    ("okay go on", "any update?"),
    ("الو", "کارت تموم شد؟"),
    ("باشه ادامه بده", "چی شد؟"),
])
async def test_a_phatic_turn_never_overtakes_a_check_in(monkeypatch, filler, check_in, gap_ms):
    """R1: research runs, the caller checks presence (the model is silent), then
    checks in.  The check-in is ANSWERED from state; nothing is dispatched a
    second time and no card is titled by the filler.  Unpunctuated fillers
    («الو», "okay go on" with no «؟») included: a phatic turn is a phrase of
    its own and never lends its words to the check-in after it."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    counters = H.capture_counters(monkeypatch)
    displays: list[str] = []
    started = asyncio.Event()
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        if len(displays) == 1:
            started.set()
            await asyncio.sleep(2.0)
            return "Robotics list.", "m"
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta(filler, 3000, 3600))
        await asyncio.sleep(0.4)
        start = 3600 + gap_ms
        p.push(H.user_delta(check_in, start, start + 500))
        p.push(H.delegation("d2", start + 550))
        await _wait(lambda: _terminal(client, "d1"), 5.0)
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=12, db_session_id=f"db-r3-phatic-{abs(hash((filler, gap_ms)))}",
    )

    assert displays == [RESEARCH], f"the filler {filler!r} was run: {displays}"
    created = [f.get("title") for f in client.of("delegation") if f.get("phase") == "created"]
    assert filler not in created and _frames_for(client, "d2") == [], created
    assert not [c for c in counters if c[0] == "live_status_dispatched"]
    assert ("live_status_answered", {"source": "running"}) in counters


@pytest.mark.asyncio
async def test_control_an_unanswered_request_still_overtakes_a_check_in(monkeypatch):
    """CONTROL (addendum 2 item 1 stays): a real request the model left
    unanswered is what the caller waits on; the check-in dispatches IT."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    displays: list[str] = []
    started = asyncio.Event()
    box: dict = {}
    request = "who is the dean of engineering there"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        if len(displays) == 1:
            started.set()
            await asyncio.sleep(2.0)
            return "Robotics list.", "m"
        return "Dean X.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta(request, 3000, 3900))
        await asyncio.sleep(0.4)
        p.push(H.user_delta("what happened?", 15000, 15500))
        p.push(H.delegation("d2", 15550))
        await _wait(lambda: _terminal(client, "d1") and _terminal(client, "d2"), 5.0)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-r3-phatic-control")

    assert displays == [RESEARCH, request], displays


def _answered_then(box, first: str, answer: str):
    """U1 `first` is answered directly by the model; the script adds the rest."""

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(first, 0, 900))
            provider.push(H.out_text(answer, 1000, 1600))
            provider.push(H.out_audio("A"))
        elif event["type"] == "session.close":
            provider.push(H.closed())
    return on_send


@pytest.mark.asyncio
@pytest.mark.parametrize(("closing", "by_id", "expected"), [
    ("okay thanks", True, "stale"),
    ("مرسی عالیه", True, "stale"),
    ("can you hear me?", True, "stale"),
    ("مرسی عالیه", False, "stale"),            # a TF132-style text-only reask
    ("and what about london?", True, "accepted"),   # CONTROL: a real request
])
async def test_a_phatic_turn_is_never_a_reask_target(monkeypatch, caplog, closing, by_id, expected):
    """R3: after an answered request the caller closes ("okay thanks"); the
    model rightly says nothing, and the app's watch re-asks the closing.  The
    relay refuses it — `stale`, logged `not_a_request` — and writes no reask
    instruction, so the model is never told to reply to «okay thanks»."""

    H.fast_clocks(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    async def closing_turn():
        await _wait(lambda: len(_finals(client)) >= 1)
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(closing, 3000, 3600))
        await _wait(lambda: len(_finals(client)) >= 2)
        frame = {"type": "inject_text", "text": closing, "reason": "no_response"}
        if by_id:
            frame["reask_of_user_turn_id"] = _finals(client)[1]["turn_id"]
        _send(client, frame)

    client = H.FakeClient([
        H.config(features=V03), closing_turn, 0.3, {"type": "stop"},
    ])
    provider = H.FakeProvider(
        on_send=_answered_then(box, "what's the weather in paris", "It is sunny in Paris."),
        auto_ack=True,
    )
    with caplog.at_level(logging.INFO, logger="app.services.live_voice_protocol"):
        await H.run_relay(client, provider, timeout=8, db_session_id=f"db-r3-reask-{abs(hash((closing, by_id)))}")

    u2 = _finals(client)[1]["turn_id"]
    results = [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")]
    assert results == [(u2, expected)], results
    reasks = [
        e for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-reask-")
    ]
    if expected == "stale":
        assert reasks == [], "the model was told to reply to a phatic turn"
        assert ("live_reask", {"outcome": "stale", "reason": "not_a_request"}) in counters
        assert any(
            "outcome=stale reason=not_a_request" in r.getMessage() for r in caplog.records
        )
    else:
        assert len(reasks) == 1


# ══════════════════════════════════════════════════════════════════════
# item 2 — `transcript.phatic` on the wire
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("features", [BASE, TF132, None], ids=["v02", "tf132", "legacy"])
async def test_final_frames_of_a_phatic_turn_carry_phatic(monkeypatch, features):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}
    turns = ["can you hear me?", "find the dean of engineering at toronto", "مرسی عالیه"]

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def script():
        at = 0
        for text in turns:
            box["p"].push(H.user_delta(text, at, at + 600))
            count = len([f for f in client.of("transcript") if f.get("final")])
            await _wait(lambda: len([f for f in client.of("transcript") if f.get("final")]) > count)
            at += 3000

    config = H.legacy_config() if features is None else H.config(features=features)
    client = H.FakeClient([config, 0.1, script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=f"db-r3-wire-{features and len(features)}")

    finals = {f["text"]: f for f in client.of("transcript") if f.get("final")}
    assert set(finals) == set(turns), finals
    if features is None:
        # Build 129 / web negotiated nothing: the frame is exactly as before.
        assert not any("phatic" in f for f in finals.values())
        return
    assert finals["can you hear me?"].get("phatic") is True
    assert finals["مرسی عالیه"].get("phatic") is True
    assert "phatic" not in finals["find the dean of engineering at toronto"]
    # Final frames only; the key is never on a partial.
    assert not any("phatic" in f for f in client.of("transcript") if not f.get("final"))


# ══════════════════════════════════════════════════════════════════════
# item 3 — the media backstop's scope
# ══════════════════════════════════════════════════════════════════════

def _silent_model(box):
    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())
    return on_send


async def _backstop_call(
    monkeypatch, turns, *, features=V03, now_playing=("Fadat Sham - Mahasti",),
    ack="چشم.", delegate_at=None, question=None,
):
    """The caller says `turns` (each its own closed turn); the model only
    acknowledges (`ack`), optionally after asking `question`, and optionally
    delegates late.  Returns (client, provider, controls, thinks)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    controls: list[str] = []
    thinks: list[str] = []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped" if action == "stop" else "executed",
                "title": "Bahaneh"}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "Professor Alan Turing.", "m"

    H.patch_relay(monkeypatch, think=think, control=control)
    box: dict = {}

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        if question:
            p.push(H.out_text(question, 100, 1500))
            p.push(H.out_audio("Q"))
            await asyncio.sleep(0.4)
        at = 3000
        for text in turns:
            p.push(H.user_delta(text, at, at + 500))
            at += 3500
            await asyncio.sleep(0.4)
        if ack:
            # After the caller's last turn (it ended at `at - 3000`).
            p.push(H.out_text(ack, at - 2900, at - 2700))
            p.push(H.out_audio("OK"))
        if delegate_at is not None:
            await asyncio.sleep(0.8)
            p.push(H.delegation("d-late", at - 2500))
        await asyncio.sleep(1.4)

    frames = [H.config(features=features)]
    for title in now_playing:
        frames.append(
            {"type": "now_playing", "title": "", "state": "stopped", "source": "media_x"}
            if title is None else {"type": "now_playing", "title": title}
        )
    client = H.FakeClient(frames + [script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_silent_model(box), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=12,
        db_session_id=f"db-r3-backstop-{abs(hash((tuple(turns), tuple(now_playing), delegate_at, question)))}",
    )
    return client, provider, controls, thinks


@pytest.mark.asyncio
@pytest.mark.parametrize("words", ["next", "بعدی", "skip"])
async def test_a_bare_media_word_is_never_backstopped(monkeypatch, words):
    """R2: music plays, the caller says a bare "next" (the model only answers).
    With no media noun the relay never executes it itself."""

    client, _provider, controls, _thinks = await _backstop_call(monkeypatch, [words], ack="Sure.")
    assert controls == [], f"a bare {words!r} skipped the track"
    assert client.of("media_control") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("words", "now_playing"), [
    ("next song", ()),                                        # nothing ever reported
    ("آهنگ رو قطع کن", ("Fadat Sham - Mahasti", None)),         # the phone reported a stop
    ("آهنگ بعدی", ("Fadat Sham - Mahasti", None)),
])
async def test_the_backstop_never_calls_the_tenant_without_reported_playback(
    monkeypatch, words, now_playing,
):
    client, _provider, controls, _thinks = await _backstop_call(
        monkeypatch, [words], now_playing=now_playing,
    )
    assert controls == [], "the tenant was called with nothing playing"
    assert client.of("media_control") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("words", "action"), [
    ("next song", "next"), ("آهنگ بعدی", "next"), ("آهنگ رو قطع کن", "stop"),
])
async def test_control_a_named_command_while_music_plays_is_backstopped(monkeypatch, words, action):
    """CONTROL: addendum 6's production shape still works."""

    client, _provider, controls, _thinks = await _backstop_call(monkeypatch, [words])
    assert controls == [action]
    assert [(f["action"], f["status"]) for f in client.of("media_control")] == [
        (action, "requested"), (action, "executed"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second"), [
    ("آهنگ رو نمی‌تونی", "قطع"),      # RUNTIME_EVIDENCE #4, V3 U:83-U:86
    ("آهنگ رو", "قطعش کن"),
    ("stop the", "music"),
])
async def test_the_recorded_split_stop_stops_the_music_and_consumes_every_turn(
    monkeypatch, first, second,
):
    """V3: «آهنگ رو نمی‌تونی» closes, «قطع» closes as its own turn, GPT-Live
    only says «چشم» and never delegates.  The anchor's authority unit is the
    command: the stop runs once, parented on the anchor, and BOTH turns are
    consumed — the model's late delegation is absorbed, never run."""

    client, _provider, controls, thinks = await _backstop_call(
        monkeypatch, [first, second], delegate_at=True,
    )
    assert controls == ["stop"], controls
    assert thinks == [], "the late delegation for the stop ran as an agent turn"
    frames = client.of("media_control")
    anchor = [f["turn_id"] for f in _finals(client) if f["text"] == second][0]
    assert [(f["action"], f["status"]) for f in frames] == [("stop", "requested"), ("stop", "executed")]
    assert {f["parent_user_turn_id"] for f in frames} == {anchor}
    assert client.of("delegation") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second"), [
    ("don't", "stop the music"),
    ("what's the weather in paris this weekend", "آهنگ رو قطع کن"),
])
async def test_the_backstop_reads_the_authority_unit_not_the_anchors_last_words(
    monkeypatch, first, second,
):
    """An unfinished preceding turn is part of the anchor's phrase: «Don't» +
    "stop the music" is not a stop, and an unpunctuated request in front of a
    stop makes the unit a request, not a bare command.  The relay never acts
    on such a unit itself (a model delegation still can)."""

    client, _provider, controls, _thinks = await _backstop_call(
        monkeypatch, [first, second], ack="Okay.",
    )
    assert controls == [], f"the relay acted on {second!r} alone"
    assert client.of("media_control") == []


@pytest.mark.asyncio
async def test_control_tf132_never_gets_a_relay_stop_for_the_split_shape(monkeypatch):
    client, _provider, controls, _thinks = await _backstop_call(
        monkeypatch, ["آهنگ رو نمی‌تونی", "قطع"], features=TF132,
    )
    assert controls == []
    assert client.of("media_control") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("words", ["next", "بعدی"])
async def test_a_bare_next_answering_the_agents_question_runs_as_the_request(monkeypatch, words):
    """R2 (late delegation): "Want me to find the next professor?" — "next";
    the model delegates the work.  It is the caller's request: it runs, and the
    track is not skipped (no backstop absorbed it, no fast path skipped it)."""

    client, _provider, controls, thinks = await _backstop_call(
        monkeypatch, [words], ack="Sure, let me look.", delegate_at=True,
        question="I found Professor Ada Lovelace. Want me to find the next professor?",
    )
    assert controls == [], "the answer to the agent's question skipped the track"
    assert thinks == [words], thinks


@pytest.mark.asyncio
async def test_control_a_bare_next_with_no_question_is_still_a_delegated_skip(monkeypatch):
    """CONTROL: a bare "next" the model DELEGATES while music plays, with no
    question from the agent before it, is still the fast media skip."""

    client, _provider, controls, thinks = await _backstop_call(
        monkeypatch, ["next"], ack="Okay.", delegate_at=True,
    )
    assert controls == ["next"], controls
    assert thinks == []


# ══════════════════════════════════════════════════════════════════════
# item 4 — a correction of a FINISHED task needs evidence
# ══════════════════════════════════════════════════════════════════════

ORIGINAL = "find iranian computer science professors at the university of toronto"
RECORD = {
    "transcript": ORIGINAL, "related_request": "",
    "answer": "Professor X and Professor Y, both at St. George.",
}


def test_modifies_record_applies_the_running_target_evidence_rule():
    for no_evidence in (
        "no, what time is it in tokyo right now?",
        "نه بابا، ساعت توکیو چنده؟",
        "not now, set a timer for ten minutes",
    ):
        assert not live._LiveSession.modifies_record(None, no_evidence, RECORD), no_evidence
    for evidenced in (
        "no, only the downtown campus",          # a restriction follows the negation
        "no, thursday",                          # a bare fragment
        "no, only iranian professors in robotics",   # shared words
        "actually only the computer science ones",
    ):
        assert live._LiveSession.modifies_record(None, evidenced, RECORD), evidenced


@pytest.mark.asyncio
@pytest.mark.parametrize(("followup", "related"), [
    ("no, what time is it in tokyo right now?", False),
    ("no, only the downtown campus", True),     # CONTROL
])
async def test_a_negating_opener_after_an_answer_relates_only_with_evidence(monkeypatch, followup, related):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[str] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        return "Professor X, Professor Y.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def follow():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(followup, 5000, 5800))
        box["p"].push(H.delegation("d2", 5850))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(), follow, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_opener(box, ORIGINAL), auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-r3-finished-{related}")

    assert len(calls) == 2
    d2 = _lifecycle(_frames_for(client, "d2"))
    assert d2
    if related:
        assert all(f.get("relation") == {"kind": "replaces", "task_id": "d1"} for f in d2), d2
    else:
        assert all("relation" not in f for f in d2), d2
        assert "Request the caller is correcting" not in calls[1]


# ══════════════════════════════════════════════════════════════════════
# integrator (round 3 verification) — a phatic turn never SUPERSEDES
# ══════════════════════════════════════════════════════════════════════
#
# Item 1 closed the phatic turn as a TARGET; the reask fences still counted it
# as a NEWER turn.  "find me a dentist" (model silent) → "are you there?" →
# the v0.3 app re-asks the dentist request by id → `stale newer_turn`, which
# the watch reads as covered: the request was dropped without a word.  The
# same held across a socket repair: "hello?" on the new socket staled the
# carried request, and "hello?" as the last thing said before a drop was
# carried IN PLACE of the unanswered request (then refused as not_a_request).

UNANSWERED = "find me a dentist near union station"


def _silent_after(box, first: str):
    """U1 `first` gets NO answer (the scripts add what follows)."""

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(first, 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())
    return on_send


def _reask_appends(provider):
    return [
        e for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-reask-")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("later", "model_answers_later", "by_id", "expected"), [
    ("are you there?", False, True, "accepted"),
    ("هستی؟", False, True, "accepted"),
    ("hello?", True, True, "accepted"),          # the model said "yes, I'm here" to it
    ("can you hear me?", False, False, "accepted"),  # a text-only reask of U1
    ("and what about london?", False, True, "stale"),  # CONTROL: a real newer request
])
async def test_a_phatic_turn_never_supersedes_an_unanswered_request(
    monkeypatch, later, model_answers_later, by_id, expected,
):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    async def later_turn():
        await _wait(lambda: len(_finals(client)) >= 1)
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(later, 4000, 4600))
        await _wait(lambda: len(_finals(client)) >= 2)
        if model_answers_later:
            box["p"].push(H.out_text("Yes, I'm here.", 4800, 5300))
            box["p"].push(H.out_audio("A"))
            await asyncio.sleep(0.4)
        frame = {"type": "inject_text", "text": UNANSWERED, "reason": "no_response"}
        if by_id:
            frame["reask_of_user_turn_id"] = _finals(client)[0]["turn_id"]
        _send(client, frame)

    client = H.FakeClient([H.config(features=V03), later_turn, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_silent_after(box, UNANSWERED), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=8,
        db_session_id=f"db-r3-supersede-{abs(hash((later, model_answers_later, by_id)))}",
    )

    u1 = _finals(client)[0]["turn_id"]
    assert [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")] == [
        (u1, expected),
    ]
    appends = _reask_appends(provider)
    if expected == "accepted":
        assert len(appends) == 1 and UNANSWERED in appends[0]["content"], appends
    else:
        assert appends == []


def _drop():
    from fastapi import WebSocketDisconnect
    raise WebSocketDisconnect(code=1006)


async def _two_sockets(monkeypatch, db, *, before_drop, after_reconnect):
    """Socket 1 hears `before_drop` turns (model silent) and drops; socket 2
    hears `after_reconnect` turns, then the phone re-asks the carried turn by
    the id it holds (U1 of socket 1)."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def first_send(provider, event):
        if event["type"] == "session.start":
            for i, text in enumerate(before_drop):
                provider.push(H.user_delta(text, i * 3000, i * 3000 + 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    c1 = H.FakeClient([H.config(features=V03), 0.9, _drop])
    await H.run_relay(c1, H.FakeProvider(on_send=first_send, auto_ack=True), timeout=6,
                      db_session_id=db)
    stranded = _finals(c1)[0]["turn_id"]

    def second_send(provider, event):
        if event["type"] == "session.start":
            for i, text in enumerate(after_reconnect):
                provider.push(H.user_delta(text, i * 3000, i * 3000 + 700))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    # A repaired socket is a NEW provider session (its own turn ids).
    p2 = H.FakeProvider(on_send=second_send, session_id="live-psid-2", auto_ack=True)
    c2 = H.FakeClient([
        H.config(features=V03), 0.7,
        {"type": "inject_text", "text": before_drop[0], "reason": "no_response",
         "reask_of_user_turn_id": stranded},
        0.4, {"type": "stop"},
    ])
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    return stranded, c2, p2


@pytest.mark.asyncio
@pytest.mark.parametrize(("hello", "expected"), [
    ("hello?", "accepted"),
    ("الو", "accepted"),
    ("actually, book a table for two instead", "stale"),   # CONTROL: a real request
])
async def test_a_presence_check_after_a_repair_never_stales_the_carried_request(
    monkeypatch, hello, expected,
):
    stranded, c2, p2 = await _two_sockets(
        monkeypatch, f"db-r3-carry-{abs(hash(hello))}",
        before_drop=[UNANSWERED], after_reconnect=[hello],
    )
    assert [(f["user_turn_id"], f["outcome"]) for f in c2.of("reask_result")] == [
        (stranded, expected),
    ]
    assert len(_reask_appends(p2)) == (1 if expected == "accepted" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(("last", "carried_is_request"), [
    ("are you there?", True),
    ("صدامو داری؟", True),
    ("and what about a pharmacy?", False),   # CONTROL: the newest real request is carried
])
async def test_a_presence_check_before_a_drop_is_never_carried_in_place_of_the_request(
    monkeypatch, last, carried_is_request,
):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    db = f"db-r3-carry-last-{abs(hash(last))}"

    def first_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(UNANSWERED, 0, 900))
            provider.push(H.user_delta(last, 3000, 3700))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    c1 = H.FakeClient([H.config(features=V03), 0.9, _drop])
    await H.run_relay(c1, H.FakeProvider(on_send=first_send, auto_ack=True), timeout=6,
                      db_session_id=db)
    finals = _finals(c1)
    assert len(finals) == 2, finals
    request_id, last_id = finals[0]["turn_id"], finals[1]["turn_id"]
    wanted = request_id if carried_is_request else last_id

    p2 = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
        session_id="live-psid-2", auto_ack=True,
    )
    c2 = H.FakeClient([
        H.config(features=V03), 0.5,
        {"type": "inject_text", "text": UNANSWERED if carried_is_request else last,
         "reason": "no_response", "reask_of_user_turn_id": wanted},
        0.4, {"type": "stop"},
    ])
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)

    assert [(f["user_turn_id"], f["outcome"]) for f in c2.of("reask_result")] == [
        (wanted, "accepted"),
    ]
    appends = _reask_appends(p2)
    assert len(appends) == 1
    assert (UNANSWERED if carried_is_request else last) in appends[0]["content"]
