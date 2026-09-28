"""R2 round 4 (addendum 4, the round-3 critic's 16 findings): the relay half.

Binding spec: /private/tmp/toup-voice-r2-spec-addendum-4.md.  Findings and
their independent verification: /private/tmp/toup-voice-r2-gates/
verify-r3.9uCiCM/{critic-findings,verify-results}.txt.

  §1  ONE per-turn "asks nothing" verdict, judged at close with dialogue
      state: a consent to the agent's own question is a request ("yes,
      sounds good", «آره عالیه»); presence checks, closings and bare prompts
      ("so?", «پس؟») ask nothing — on the wire (`phatic`), in every reask and
      carry fence, the status path, and a greeting never spawns research.
  §2  An asks-nothing tail never anchors or titles the request before it.
  §3  "Answered" is HEARD, not received; an UNCERTAIN request (a reply to a
      later phatic turn was heard) is re-asked conditionally, never with
      "no reply reached them".
  §4.2 The turn open at a drop is closed and judged before the carry.
  §5  Value fragments and question subjects decide negation evidence; a
      PARTIAL correction of running work asks "Should I stop «A»?" first.
  §6  The media backstop's evidence is causal play evidence; the wording is
      honest ("I won't start «X»"); a newer play is never halted.
  §7  A fired command's tail is part of it; a hesitation inside a phrase is
      transparent; content-free tokens are media filler.

Old-fail is proven by running THIS file against integrated-snap5 (the
pre-round-4 relay): every behaviour test fails there by assertion.  Controls
pass on both.  Cross-layer: with ROUND4_FRAMES_OUT set, the key scenarios'
relay frames are written to that JSON file for the app's real turnWatch/hook
replay (never written otherwise).
"""

from __future__ import annotations

import asyncio
import json
import os

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
SUBJECT = "find Iranian computer science professors at UofT downtown who work on robotics"
RESEARCH = "research the toronto robotics faculty"
UNANSWERED = "find me a dentist near union station"


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════

def _finals(client) -> list[dict]:
    return [f for f in client.of("transcript") if f.get("final")]


def _frames_for(client, did):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _phases(client, did):
    return [f.get("phase") for f in _frames_for(client, did)]


def _terminal(client, did):
    return any(
        f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
        for f in _frames_for(client, did)
    )


def _lifecycle(frames):
    return [f for f in frames if "task_revision" in f]


class Phone(H.FakeClient):
    """`FakeClient` whose scenario runs in the background, so the phone can
    send a frame NOW (`push_frame`) — a receipt for the epoch just played, a
    reask — while the scenario keeps going.  A pushed callable is called (a
    drop raises from it)."""

    def __init__(self, script=None):
        super().__init__(script)
        self.inbox: list = []
        self._bg = None

    def push_frame(self, frame) -> None:
        self.inbox.append(frame)

    async def receive_text(self):
        import inspect

        while True:
            if self.inbox:
                item = self.inbox.pop(0)
                self.reads += 1
                if callable(item):
                    item()
                    continue
                return json.dumps(item)
            if self._bg is not None:
                if not self._bg.done():
                    await asyncio.sleep(0.01)
                    continue
                bg, self._bg = self._bg, None
                bg.result()
            if not self._script:
                await asyncio.sleep(0.01)
                continue
            item = self._script.pop(0)
            self.reads += 1
            if isinstance(item, (int, float)):
                self._bg = asyncio.ensure_future(asyncio.sleep(item))
                continue
            if callable(item):
                result = item()
                if inspect.isawaitable(result):
                    self._bg = asyncio.ensure_future(result)
                continue
            return json.dumps(item)


def _send(client, frame) -> None:
    """The phone sends `frame` now (or, for a plain FakeClient, next)."""

    if isinstance(client, Phone):
        client.push_frame(frame)
    else:
        client._script.insert(0, frame)


async def _wait(pred, timeout=4.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def _reask_appends(provider):
    return [
        e for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-reask-")
    ]


def _commentary(provider) -> list[str]:
    return [str(e.get("content") or "") for e in provider.of("session.commentary.append")]


def _thinking(provider) -> list[dict]:
    return provider.of("session.thinking.append")


def _db(tag, *parts) -> str:
    return f"db-r4-{tag}-{abs(hash(parts))}"


def _dump(name: str, client) -> None:
    """Cross-layer: the relay's frames for the app's real replay, written only
    under ROUND4_FRAMES_OUT (like CLOSE_FRAMES_OUT)."""

    path = os.environ.get("ROUND4_FRAMES_OUT")
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    keep = {
        "transcript", "delegation", "reask_result", "response_text", "audio_delta",
        "speech_segment_complete", "media_control", "playback_interrupted", "state",
    }
    data[name] = [f for f in client.frames if f.get("type") in keep]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


def _turn(text: str, n: int = 1, **extra) -> "live.Utterance":
    return live.Utterance(
        turn_id=f"U:{n}", ordinal=n, start_ms=0, end_ms=500, text=text,
        closed=True, causal_speech=True, **extra,
    )


def _silent_after(box, first: str, *, extra=None):
    """U1 `first` gets NO answer; `extra` (provider events) follow it."""

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(first, 0, 900))
            for item in extra or ():
                provider.push(item)
        elif event["type"] == "session.close":
            provider.push(H.closed())
    return on_send


def _drop():
    from fastapi import WebSocketDisconnect
    raise WebSocketDisconnect(code=1006)


# ══════════════════════════════════════════════════════════════════════
# §1.3 — the structural grammar (pure seams that exist on every build)
# ══════════════════════════════════════════════════════════════════════

#: relay-phatic-2: everyday presence / wellbeing checks, en + fa.
PRESENCE = [
    "can you still hear me?", "hello? anyone?", "did you hear me?", "are you still on the line?",
    "you still with me?", "are you still with me?", "did you understand what I said?",
    "how are you?", "hi toup, how are you?", "hello, are you still there?",
    "هستی هنوز؟", "کجایی؟", "سلام خوبی؟", "اونجا هستی؟", "تو هستی؟", "هنوز اونجا هستی؟",
    "الو هنوز صدامو داری؟", "صدامو شنیدی؟", "حالت چطوره؟", "متوجه شدی؟",
]
#: …and what must stay a request (third person, 'yet', content, a question).
PRESENCE_CONTROLS = [
    "is it still there?", "are you there yet?", "تو کجا هستی", "where is the nearest pharmacy",
    "can you hear the music in the video", "هستی؟ ساعت توکیو چنده", "نه، کجایی هستش استاده؟",
    "is anyone there at the clinic on sundays?", "anyone", "did you find anyone?",
    "thanks, and find the dean", "how are you going to fix this", "yes", "okay",
    "تو کجایی هستی؟",   # "where are you from?": two predicates in one clause
]
#: relay-phatic-4: closings are THANKS_HEAD [PREP OBJECT] (+ address words).
CLOSINGS = [
    "thank you for your help", "thanks for your help", "خیلی ممنونم", "مرسی خسته نباشی",
    "thanks man", "thank you sir", "thanks buddy", "thanks for that",
    "thank you so much for your help", "thanks for everything, bye", "مرسی از کمکت",
    "ممنون بابت همه چی", "مرسی داداش", "خسته نباشی", "ممنونم ازت", "خیلی‌ممنونم",
]
CLOSING_CONTROLS = [
    "thanks for the list", "thanks for your help with the booking",
    "thanks for your help, now find the dean", "thank you for that, can you also email it to me",
    "مرسی، حالا یه ایمیل بفرست", "ممنون از کمکت، ساعت توکیو چنده؟",
    "خیلی ممنون میشم اگه ایمیلش کنی", "man", "خیلی", "yes sir", "خسته‌ام", "خسته شدم",
]


@pytest.mark.parametrize("text", PRESENCE + CLOSINGS)
def test_presence_checks_and_closings_are_phatic(text):
    assert live.is_phatic(text), text
    assert not live._LiveSession.is_request_turn(None, _turn(text), SUBJECT), text


@pytest.mark.parametrize("text", PRESENCE_CONTROLS + CLOSING_CONTROLS)
def test_control_requests_and_answers_stay_non_phatic(text):
    assert not live.is_phatic(text), text


@pytest.mark.parametrize("text", ["so?", "well?", "and?", "پس؟", "Um.", "and then?"])
def test_a_bare_prompt_asks_nothing_and_is_never_a_request(text):
    fn = getattr(live, "text_asks_nothing", None)
    assert fn is not None and fn(text), text
    assert not live._LiveSession.is_request_turn(None, _turn(text), SUBJECT)


@pytest.mark.parametrize("text", ["okay", "yes", "okay?", "find it", "tell me", "no", "نه", "باشه"])
def test_control_a_bare_answer_is_not_a_prompt(text):
    """§1.4: bare 'okay'/'yes'/«باشه» stay NOT in the class (a false request
    costs one reask, a false phatic loses a consent)."""

    fn = getattr(live, "text_asks_nothing", None)
    assert fn is None or not fn(text), text


@pytest.mark.parametrize("fused", [
    "hello? any update?", "الو؟ چی شد؟", "هستی؟ چی شد؟", "can you hear me? any update?",
    "hello? what happened?",
])
def test_a_check_in_fused_to_a_presence_check_is_a_check_in(fused):
    """§1.6 (relay-phatic-2, fused variant)."""

    assert live.is_status_question(fused, SUBJECT), fused


@pytest.mark.parametrize("utterance", [
    "هستی؟ ساعت توکیو چنده", "hello? what's the weather in paris", "are you there? find the dean",
])
def test_control_a_request_after_a_presence_check_is_a_request(utterance):
    assert not live.is_status_question(utterance, SUBJECT), utterance


# ══════════════════════════════════════════════════════════════════════
# §1.1/§1.4 — a consent to the agent's own question is a request (wire)
# ══════════════════════════════════════════════════════════════════════

QUESTION = "I found Union Dental on Front Street. Should I book you in for tomorrow at ten?"
STATEMENT = "I found Union Dental on Front Street. It is open until six today."
QUESTION_FA = "دندونپزشکی یونیون رو پیدا کردم. برای فردا ساعت ده وقت بگیرم؟"
STATEMENT_FA = "دندونپزشکی یونیون رو پیدا کردم. امروز تا ساعت شش بازه."
CONSENTS = ["yes, sounds good", "sure, that's fine", "yes, perfect", "آره عالیه", "باشه خوبه"]


async def _consent_call(monkeypatch, answer, agent_line, *, features=V03, before=(), db=""):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(UNANSWERED, 0, 900))
            p.push(H.out_text(agent_line, 1000, 2600))
            p.push(H.out_audio("A"))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def confirm():
        await _wait(lambda: len(_finals(client)) >= 1)
        await asyncio.sleep(0.3)
        at = 4000
        for text in before:
            box["p"].push(H.user_delta(text, at, at + 300))
            count = len(_finals(client))
            await _wait(lambda: len(_finals(client)) > count)
            at += 1500
        box["p"].push(H.user_delta(answer, at, at + 600))
        count = len(_finals(client))
        await _wait(lambda: len(_finals(client)) > count)
        _send(client, {
            "type": "inject_text", "text": answer, "reason": "no_response",
            "reask_of_user_turn_id": _finals(client)[-1]["turn_id"],
        })

    client = Phone([H.config(features=features), confirm, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=db or _db("consent", answer, agent_line, before))
    return client, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", CONSENTS)
async def test_a_consent_to_the_agents_question_is_a_request(monkeypatch, answer):
    """relay-phatic-1: the booking the caller confirmed stays recoverable —
    no `phatic` flag, and the by-id reask is accepted (old: phatic + stale)."""

    line = QUESTION_FA if answer[0] > "z" else QUESTION
    client, provider = await _consent_call(monkeypatch, answer, line)
    last = _finals(client)[-1]
    assert last.get("phatic") is not True, f"{answer!r} after a question was flagged phatic"
    assert [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")] == [
        (last["turn_id"], "accepted"),
    ]
    assert len(_reask_appends(provider)) == 1
    _dump(f"consent_after_question:{answer}", client)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", CONSENTS)
async def test_control_the_same_words_after_a_statement_close_the_exchange(monkeypatch, answer):
    line = STATEMENT_FA if answer[0] > "z" else STATEMENT
    client, provider = await _consent_call(monkeypatch, answer, line)
    last = _finals(client)[-1]
    assert last.get("phatic") is True
    assert [f["outcome"] for f in client.of("reask_result")] == ["stale"]
    assert _reask_appends(provider) == []
    _dump(f"closing_after_statement:{answer}", client)


@pytest.mark.asyncio
async def test_a_content_free_turn_does_not_close_the_agents_question(monkeypatch):
    """§1.1 window: question → "hmm" (its own turn) → "yes, sounds good"."""

    client, provider = await _consent_call(
        monkeypatch, "yes, sounds good", QUESTION, before=("hmm",),
    )
    last = _finals(client)[-1]
    assert last["text"] == "yes, sounds good"
    assert last.get("phatic") is not True
    assert [f["outcome"] for f in client.of("reask_result")] == ["accepted"]


@pytest.mark.asyncio
async def test_a_question_held_behind_the_callers_turn_still_counts(monkeypatch):
    """§1.1: the agent's question arrives while the caller's answer is still
    open (deferred output) — it was spoken BEFORE the answer began."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(UNANSWERED, 0, 900))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: len(_finals(client)) >= 1)
        await asyncio.sleep(0.2)
        p = box["p"]
        p.push(H.user_delta("yes, sounds", 2700, 2900))
        p.push(H.out_text(QUESTION, 1000, 2600))       # spoken before, held behind "yes"
        p.push(H.out_audio("A"))
        p.push(H.user_delta(" good", 2900, 3000))
        await _wait(lambda: len(_finals(client)) >= 2)
        await asyncio.sleep(0.4)   # the released question's epoch retires
        _send(client, {
            "type": "inject_text", "text": "yes, sounds good", "reason": "no_response",
            "reask_of_user_turn_id": _finals(client)[1]["turn_id"],
        })

    client = Phone([H.config(features=V03), script, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=_db("deferred-q"))
    answer = _finals(client)[1]
    assert answer.get("phatic") is not True
    assert [f["outcome"] for f in client.of("reask_result")] == ["accepted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", ["yes, perfect", "آره عالیه", "okay thanks"])
async def test_control_a_closing_after_an_answered_question_stays_phatic(monkeypatch, closing):
    """'Should I…?' → 'yes' → 'Done, you're booked…' → 'yes, perfect': the
    question was answered; the closing asks nothing."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(UNANSWERED, 0, 900))
            p.push(H.out_text(QUESTION, 1000, 2600))
            p.push(H.out_audio("A"))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: len(_finals(client)) >= 1)
        p = box["p"]
        await asyncio.sleep(0.2)
        p.push(H.user_delta("yes", 3000, 3300))
        await _wait(lambda: len(_finals(client)) >= 2)
        p.push(H.out_text("Done, you're booked for tomorrow at ten.", 3500, 4500))
        p.push(H.out_audio("B"))
        await asyncio.sleep(0.4)
        p.push(H.user_delta(closing, 6000, 6600))
        await _wait(lambda: len(_finals(client)) >= 3)
        _send(client, {
            "type": "inject_text", "text": closing, "reason": "no_response",
            "reask_of_user_turn_id": _finals(client)[2]["turn_id"],
        })

    client = Phone([H.config(features=V03), script, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=_db("answered-q", closing))
    last = _finals(client)[2]
    assert last.get("phatic") is True
    assert [f["outcome"] for f in client.of("reask_result")] == ["stale"]
    assert _reask_appends(provider) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes, sounds good", "آره عالیه", "yes"])
async def test_a_consent_before_a_drop_is_carried(monkeypatch, answer):
    """relay-phatic-1 across a repair: the consent is carried; the repaired
    socket (a new provider session) accepts its by-id reask.  'yes' is the
    control (accepted on both builds)."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    db = _db("consent-drop", answer)
    line = QUESTION_FA if answer[0] > "z" else QUESTION

    def first_send(p, e):
        if e["type"] == "session.start":
            p.push(H.user_delta(UNANSWERED, 0, 900))
            p.push(H.out_text(line, 1000, 2600))
            p.push(H.out_audio("A"))
            p.push(H.user_delta(answer, 4000, 4600))
        elif e["type"] == "session.close":
            p.push(H.closed())

    c1 = Phone([H.config(features=V03), 0.9, _drop])
    await H.run_relay(c1, H.FakeProvider(on_send=first_send, auto_ack=True), timeout=6,
                      db_session_id=db)
    consent = _finals(c1)[-1]
    assert consent["text"] == answer and consent.get("phatic") is not True
    p2 = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
        session_id="live-psid-r4-2", auto_ack=True,
    )
    c2 = Phone([
        H.config(features=V03), 0.5,
        {"type": "inject_text", "text": answer, "reason": "no_response",
         "reask_of_user_turn_id": consent["turn_id"]},
        0.4, {"type": "stop"},
    ])
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    assert [(f["user_turn_id"], f["outcome"]) for f in c2.of("reask_result")] == [
        (consent["turn_id"], "accepted"),
    ]


# ══════════════════════════════════════════════════════════════════════
# §1.5 — the flag and every fence read the same verdict (wire)
# ══════════════════════════════════════════════════════════════════════

PROMPTS = [
    "so?", "well?", "and?", "پس؟", "Um.", "can you still hear me?", "hello? anyone?",
    "هستی هنوز؟", "did you hear me?",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("later", PROMPTS + ["and what about london?"])
async def test_a_turn_that_asks_nothing_never_supersedes_the_unanswered_request(monkeypatch, later):
    """relay-phatic-3: flagged on the wire AND not newer for the fence; the
    request the caller is waiting on is re-asked (the control — a real newer
    request — still stales it)."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    async def later_turn():
        await _wait(lambda: len(_finals(client)) >= 1)
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(later, 4000, 4600))
        await _wait(lambda: len(_finals(client)) >= 2)
        f = _finals(client)
        # The v0.3 watch re-asks its newest UNFLAGGED final.
        target = f[0] if f[1].get("phatic") else f[1]
        _send(client, {
            "type": "inject_text", "text": target["text"], "reason": "no_response",
            "reask_of_user_turn_id": target["turn_id"],
        })

    client = Phone([H.config(features=V03), later_turn, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_silent_after(box, UNANSWERED), auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=_db("fence", later))

    finals = _finals(client)
    appends = _reask_appends(provider)
    if later == "and what about london?":
        assert finals[1].get("phatic") is None
        assert len(appends) == 1 and later in appends[0]["content"]
        return
    assert finals[1].get("phatic") is True, finals[1]
    assert [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")] == [
        (finals[0]["turn_id"], "accepted"),
    ]
    assert len(appends) == 1 and UNANSWERED in appends[0]["content"], appends
    _dump(f"prompt_after_request:{later}", client)


@pytest.mark.asyncio
@pytest.mark.parametrize("later", ["so?", "can you still hear me?", "hello? anyone?"])
async def test_a_by_id_reask_is_never_staled_by_a_turn_that_asks_nothing(monkeypatch, later):
    """Probe C shape: the request is re-asked by id while the prompt exists."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    async def later_turn():
        await _wait(lambda: len(_finals(client)) >= 1)
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(later, 4000, 4600))
        await _wait(lambda: len(_finals(client)) >= 2)
        _send(client, {
            "type": "inject_text", "text": UNANSWERED, "reason": "no_response",
            "reask_of_user_turn_id": _finals(client)[0]["turn_id"],
        })

    client = Phone([H.config(features=V03), later_turn, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_silent_after(box, UNANSWERED), auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=_db("byid", later))
    assert [f["outcome"] for f in client.of("reask_result")] == ["accepted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", [
    "thank you for your help", "thanks for your help", "خیلی ممنونم", "مرسی خسته نباشی", "thanks man",
])
async def test_a_closing_after_an_answered_request_is_never_reasked(monkeypatch, closing):
    """relay-phatic-4 (R3 class)."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta("what's the weather in paris", 0, 900))
            p.push(H.out_text("It is sunny in Paris.", 1000, 1600))
            p.push(H.out_audio("A"))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def closing_turn():
        await _wait(lambda: len(_finals(client)) >= 1)
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(closing, 3000, 3600))
        await _wait(lambda: len(_finals(client)) >= 2)
        _send(client, {
            "type": "inject_text", "text": closing, "reason": "no_response",
            "reask_of_user_turn_id": _finals(client)[1]["turn_id"],
        })

    client = Phone([H.config(features=V03), closing_turn, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=_db("closing", closing))
    assert _finals(client)[1].get("phatic") is True
    assert [f["outcome"] for f in client.of("reask_result")] == ["stale"]
    assert _reask_appends(provider) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("last", ["can you still hear me?", "هستی هنوز؟", "so?", "and?"])
async def test_a_turn_that_asks_nothing_is_never_carried_in_place_of_the_request(monkeypatch, last):
    """Probe F: the request is carried across the drop, not the prompt."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    db = _db("carry", last)

    def first_send(p, e):
        if e["type"] == "session.start":
            p.push(H.user_delta(UNANSWERED, 0, 900))
            p.push(H.user_delta(last, 3000, 3700))
        elif e["type"] == "session.close":
            p.push(H.closed())

    c1 = Phone([H.config(features=V03), 0.9, _drop])
    await H.run_relay(c1, H.FakeProvider(on_send=first_send, auto_ack=True), timeout=6,
                      db_session_id=db)
    request_id = _finals(c1)[0]["turn_id"]
    p2 = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
        session_id="live-psid-r4-carry", auto_ack=True,
    )
    c2 = Phone([
        H.config(features=V03), 0.5,
        {"type": "inject_text", "text": UNANSWERED, "reason": "no_response",
         "reask_of_user_turn_id": request_id},
        0.4, {"type": "stop"},
    ])
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    assert [(f["user_turn_id"], f["outcome"]) for f in c2.of("reask_result")] == [
        (request_id, "accepted"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("features", [V03, TF132, None], ids=["v03", "tf132", "legacy"])
async def test_the_phatic_flag_is_final_only_gated_and_follows_the_verdict(monkeypatch, features):
    """Wire contract unchanged: final frames only, `live_turns` only (legacy
    gets no key), `echo_only` wins; the NEW members are flagged too."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}
    turns = ["can you still hear me?", "find the dean of engineering at toronto", "so?", "خیلی ممنونم"]

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def script():
        at = 0
        for text in turns:
            box["p"].push(H.user_delta(text, at, at + 600))
            count = len(_finals(client))
            await _wait(lambda: len(_finals(client)) > count)
            at += 3000

    config = H.legacy_config() if features is None else H.config(features=features)
    client = Phone([config, 0.1, script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=_db("wire", str(features)))
    finals = {f["text"]: f for f in _finals(client)}
    assert set(finals) == set(turns), finals
    if features is None:
        assert not any("phatic" in f for f in finals.values())
        return
    assert finals["can you still hear me?"].get("phatic") is True
    assert finals["so?"].get("phatic") is True
    assert finals["خیلی ممنونم"].get("phatic") is True
    assert "phatic" not in finals["find the dean of engineering at toronto"]
    assert not any("phatic" in f for f in client.of("transcript") if not f.get("final"))


# ══════════════════════════════════════════════════════════════════════
# §1.5/§1.6 — presence variants never overtake a check-in (R1 shape)
# ══════════════════════════════════════════════════════════════════════

def _opener(box, first=RESEARCH, first_id="d1"):
    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(first, 0, 900))
            p.push(H.delegation(first_id, 950))
        elif e["type"] == "session.close":
            p.push(H.closed())
    return on_send


async def _research_then(monkeypatch, turns, *, gap_ms=11000, delegate=True, db=""):
    """Research runs (d1, 2 s); the caller says `turns` (the model silent),
    the last one delegated as d2.  Returns (client, provider, displays,
    counters)."""

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
        at = 3000
        for index, text in enumerate(turns):
            p.push(H.user_delta(text, at, at + 500))
            if index == len(turns) - 1 and delegate:
                p.push(H.delegation("d2", at + 550))
            await asyncio.sleep(0.4)
            at += 500 + gap_ms
        await _wait(lambda: _terminal(client, "d1"), 5.0)
        await asyncio.sleep(0.3)

    client = Phone([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=db or _db("r1", tuple(turns), gap_ms))
    return client, provider, displays, counters


@pytest.mark.asyncio
@pytest.mark.parametrize("gap_ms", [11000, 1500], ids=["outside_span", "inside_span"])
@pytest.mark.parametrize(("filler", "check_in"), [
    ("can you still hear me?", "any update?"),
    ("hello? anyone?", "any update?"),
    ("هستی هنوز؟", "چی شد؟"),
    ("کجایی؟", "چی شد؟"),
    ("سلام خوبی؟", "چی شد؟"),
])
async def test_a_presence_variant_never_overtakes_a_check_in(monkeypatch, filler, check_in, gap_ms):
    """relay-phatic-2 (probe B): the check-in is answered from state; nothing
    is dispatched as research and no card is titled by the presence check."""

    client, _provider, displays, counters = await _research_then(
        monkeypatch, [filler, check_in], gap_ms=gap_ms,
    )
    assert displays == [RESEARCH], displays
    assert _frames_for(client, "d2") == []
    assert not [c for c in counters if c[0] == "live_status_dispatched"]
    assert ("live_status_answered", {"source": "running"}) in counters


@pytest.mark.asyncio
@pytest.mark.parametrize("fused", ["hello? any update?", "الو؟ چی شد؟", "هستی؟ چی شد؟"])
async def test_a_fused_presence_and_check_in_is_answered_from_state(monkeypatch, fused):
    """§1.6: one utterance, a presence check fused to a check-in."""

    client, _provider, displays, counters = await _research_then(monkeypatch, [fused])
    assert displays == [RESEARCH], displays
    assert _frames_for(client, "d2") == []
    assert ("live_status_answered", {"source": "running"}) in counters


# ══════════════════════════════════════════════════════════════════════
# §1.7 — greetings and presence checks never spawn research
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("greeting", ["hello, how are you?", "سلام خوبی؟", "are you there?", "thanks man"])
async def test_a_greeting_the_model_delegates_starts_no_task(monkeypatch, greeting):
    """Owner req 13: nothing unanswered before it — no card, no agent run; the
    function call is completed with a note that it is conversation."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    counters = H.capture_counters(monkeypatch)
    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "An answer.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        if e["type"] == "session.start":
            p.push(H.user_delta(greeting, 0, 900))
            p.push(H.delegation("d-hello", 950))
        elif e["type"] == "session.close":
            p.push(H.closed())

    client = Phone([H.config(), 0.8, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=_db("greet", greeting))
    assert thinks == [], f"{greeting!r} ran as research"
    assert client.of("delegation") == [], "a card for a greeting"
    assert ("live_followup", {"verdict": "phatic_no_task"}) in counters
    notes = [e for e in _thinking(provider) if e.get("delegation_id") == "d-hello"]
    assert notes and "conversation" in notes[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reminder", ["are you there?", "هستی؟", "did you hear me?", "متوجه شدی؟"])
async def test_a_reminder_delegation_runs_the_request_it_reminds_of(monkeypatch, reminder):
    """Owner req 6 shape: the model ignored U1 and delegated only after the
    caller's reminder; the work is U1 — its words, its title — never the
    reminder's (rebind reason `phatic_anchor`)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    counters = H.capture_counters(monkeypatch)
    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "Union Dental, Front Street.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        await _wait(lambda: len(_finals(client)) >= 1)
        box["p"].push(H.user_delta(reminder, 12000, 12600))
        box["p"].push(H.delegation("d-remind", 12650))
        await _wait(lambda: _terminal(client, "d-remind"), 4.0)

    client = Phone([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_silent_after(box, UNANSWERED), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=_db("remind", reminder))
    assert thinks == [UNANSWERED], thinks
    created = [f for f in client.of("delegation") if f.get("phase") == "created"]
    assert [f["title"] for f in created] == [UNANSWERED]
    assert created[0]["turn_id"] == _finals(client)[0]["turn_id"]
    assert ("live_status_dispatched", {"reason": "phatic_anchor"}) in counters


# ══════════════════════════════════════════════════════════════════════
# §2 — an asks-nothing tail never anchors or titles the request
# ══════════════════════════════════════════════════════════════════════

async def _span_call(monkeypatch, first, second, *, gap=1500, db=""):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[dict] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"display": kw.get("display_request") or "", "lang": kw.get("reply_language")})
        return "Union Dental, Front Street.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        await _wait(lambda: len(_finals(client)) >= 1)
        p = box["p"]
        start = 900 + gap
        p.push(H.user_delta(second, start, start + 600))
        p.push(H.delegation("d-span", start + 650))
        await _wait(lambda: _terminal(client, "d-span"), 4.0)
        # A second decision after the tail must not bind the consumed tail.
        p.push(H.delegation("d-again", start + 700))
        await asyncio.sleep(0.8)

    client = Phone([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_silent_after(box, first), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=db or _db("span", first, second))
    return client, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["Are you there?", "هستی؟", "Okay, thanks."])
async def test_a_phatic_tail_never_anchors_or_titles_the_request(monkeypatch, tail):
    """app-5 / relay: anchor R, request_turn_ids [R], title and language from
    R only; the tail stays consumed (a later decision cannot run it)."""

    client, calls = await _span_call(monkeypatch, UNANSWERED, tail)
    r_id = _finals(client)[0]["turn_id"]
    created = [f for f in _frames_for(client, "d-span") if f.get("phase") == "created"]
    assert created and created[0]["turn_id"] == r_id
    assert created[0]["title"] == UNANSWERED
    assert created[0].get("request_turn_ids") == [r_id]
    assert [c["display"] for c in calls] == [UNANSWERED], calls
    assert calls[0]["lang"] == "en", "an English request took its language from «هستی؟»"
    assert not [f for f in _frames_for(client, "d-again") if f.get("phase") == "created"]
    _dump(f"span_tail:{tail}", client)


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second", "title"), [
    (UNANSWERED, "one that is open on sunday", UNANSWERED + " one that is open on sunday"),
    ("find me a restaurant that is", "good", "find me a restaurant that is good"),
    (UNANSWERED, "okay thanks", UNANSWERED + " okay thanks"),   # conscious choice: stays joined
])
async def test_control_continuations_keep_every_word(monkeypatch, first, second, title):
    client, calls = await _span_call(monkeypatch, first, second)
    created = [f for f in _frames_for(client, "d-span") if f.get("phase") == "created"]
    assert created and created[0]["title"] == title, created
    assert [c["display"] for c in calls] == [title]


# ══════════════════════════════════════════════════════════════════════
# §3 — heard, not received; the UNCERTAIN request is re-asked conditionally
# ══════════════════════════════════════════════════════════════════════

LIBRARY_Q = "what time does the union station library close"


def _epoch_of(client) -> str:
    return client.of("audio_delta")[-1]["response_id"]


async def _hello_then(
    monkeypatch, reply, *, receipt=None, request=LIBRARY_Q, features=V03, by_id=True,
    later="hello?", db="",
):
    """R unanswered; the caller says `later` (asks nothing); the model's
    reply is parented to that turn; `receipt(eid)` is the phone's frame for
    the reply's epoch (None: no receipt yet); then R is re-asked."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    async def script():
        await _wait(lambda: len(_finals(client)) >= 1)
        await asyncio.sleep(0.3)
        p = box["p"]
        p.push(H.user_delta(later, 8000, 8400))
        await _wait(lambda: len(_finals(client)) >= 2)
        if reply:
            p.push(H.out_text(reply, 8600, 9900))
            p.push(H.out_audio("A"))
            await _wait(lambda: client.of("audio_delta"))
            if receipt is not None:
                _send(client, receipt(_epoch_of(client)))
            await asyncio.sleep(0.5)
        frame = {"type": "inject_text", "text": request, "reason": "no_response"}
        if by_id:
            frame["reask_of_user_turn_id"] = _finals(client)[0]["turn_id"]
        _send(client, frame)

    client = Phone([H.config(features=features), script, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_silent_after(box, request), auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=db or _db("hello", reply, str(receipt), later))
    return client, provider


def _idle(eid):
    return {"type": "playback_idle", "response_id": eid, "item_id": eid}


def _cut(played_ms):
    return lambda eid: {"type": "interrupt", "response_id": eid, "item_id": eid, "played_ms": played_ms}


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    "Yes, I'm here.",
    "Yes, I'm here, let me check the library hours.",
    "The Union Station library closes at nine tonight.",
    "It's open until nine this evening.",          # a paraphrase sharing no word
])
async def test_a_request_after_a_reply_to_a_phatic_turn_is_reasked_conditionally(monkeypatch, reply):
    """app-1 (supervisor 14:19): the reply to "hello?" was HEARD; whether it
    answered R is not the relay's to guess (no lexical overlap rule).  R stays
    recoverable, `accepted` + `conditional: true`, and the instruction never
    claims nothing reached the caller."""

    client, provider = await _hello_then(monkeypatch, reply, receipt=_idle)
    r_id = _finals(client)[0]["turn_id"]
    results = client.of("reask_result")
    assert [(f["user_turn_id"], f["outcome"]) for f in results] == [(r_id, "accepted")]
    assert results[0].get("conditional") is True
    appends = _reask_appends(provider)
    assert len(appends) == 1
    content = appends[0]["content"]
    assert "Earlier the caller asked" in content and LIBRARY_Q in content and "hello?" in content
    assert "No reply" not in content and "did not reach" not in content
    _dump(f"conditional_reask:{reply}", client)


@pytest.mark.asyncio
async def test_a_reply_with_no_receipt_yet_is_presumed_heard(monkeypatch):
    """The pinned round-3 shape ("Yes, I'm here." with no receipt yet) → R is
    uncertain → conditional.

    Supervisor 2026-09-23 18:16: the RATIONALE this pin carried ("the
    delivery presumption stands", i.e. no receipt = heard) is withdrawn — a
    missing receipt is UNKNOWN, never heard.  The verdict is unchanged for a
    different reason: a reply to the asks-nothing turn that MAY have been
    heard (HEARD or UNKNOWN) makes R uncertain, so `accepted` +
    `conditional: true`; only a provably UNHEARD reply leaves R plain (next
    test).  The name is kept so the pin's id stays stable across rounds."""

    client, provider = await _hello_then(monkeypatch, "Yes, I'm here.", receipt=None)
    results = client.of("reask_result")
    assert [f["outcome"] for f in results] == ["accepted"] and results[0].get("conditional") is True
    # The P-reply wording (the reply to "hello?"), not R's own-answer wording:
    # R itself had no output parented to it.
    content = _reask_appends(provider)[0]["content"]
    assert "Earlier the caller asked" in content and "never confirmed" not in content


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [_cut(0), _cut(None)], ids=["played_0", "no_played_ms"])
async def test_an_unheard_reply_to_the_phatic_turn_leaves_a_plain_reask(monkeypatch, receipt):
    """§3.2 pin: P-parented output entirely unheard → plain (not conditional)."""

    client, provider = await _hello_then(monkeypatch, "Yes, I'm here.", receipt=receipt)
    results = client.of("reask_result")
    assert [f["outcome"] for f in results] == ["accepted"]
    assert "conditional" not in results[0]
    content = _reask_appends(provider)[0]["content"]
    assert content.startswith("No reply to the caller's most recent words reached them")


@pytest.mark.asyncio
async def test_control_a_plain_unanswered_request_carries_no_conditional_key(monkeypatch):
    client, provider = await _hello_then(monkeypatch, "", receipt=None)
    results = client.of("reask_result")
    assert [f["outcome"] for f in results] == ["accepted"] and "conditional" not in results[0]


@pytest.mark.asyncio
async def test_tf132_gets_no_new_key_but_the_same_conditional_recovery(monkeypatch):
    """Feature negotiation: no `reask_turns`, no reask_result at all; the
    text-only reask still gets the conditional instruction."""

    client, provider = await _hello_then(
        monkeypatch, "Yes, I'm here.", receipt=_idle, features=TF132, by_id=False,
    )
    assert client.of("reask_result") == []
    appends = _reask_appends(provider)
    assert len(appends) == 1 and "Earlier the caller asked" in appends[0]["content"]


async def _answered_then_reask(monkeypatch, receipt, *, features=V03, db=""):
    """R answered by an R-parented reply; `receipt(eid)` for that epoch; then
    R is re-asked by id."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}
    request = "what's the weather in paris"

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(request, 0, 900))
            p.push(H.out_text("It is sunny in Paris.", 1000, 1600))
            p.push(H.out_audio("A"))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: client.of("audio_delta"))
        await _wait(lambda: len(_finals(client)) >= 1)
        if receipt is not None:
            _send(client, receipt(_epoch_of(client)))
        await asyncio.sleep(0.5)
        _send(client, {
            "type": "inject_text", "text": request, "reason": "no_response",
            "reask_of_user_turn_id": _finals(client)[0]["turn_id"],
        })

    client = Phone([H.config(features=features), script, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=db or _db("rheard", str(receipt), str(features)))
    return client, provider


def _failed(eid):
    return {"type": "playback_failed", "response_id": eid, "item_id": eid, "reason": "decode_error"}


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [_cut(0), _failed], ids=["cut_before_audio", "playback_failed"])
async def test_an_answer_that_was_never_heard_leaves_the_request_recoverable(monkeypatch, receipt):
    """§3.2 pin: R-parented output entirely unheard → R recoverable, a plain
    reask with TRUTHFUL wording (old: `answered`, the caller never heard it)."""

    client, provider = await _answered_then_reask(monkeypatch, receipt)
    results = client.of("reask_result")
    assert [f["outcome"] for f in results] == ["accepted"], results
    assert "conditional" not in results[0]
    content = _reask_appends(provider)[0]["content"]
    assert "did not reach them" in content and "what's the weather in paris" in content


@pytest.mark.asyncio
async def test_an_empty_heard_receipt_is_unheard(monkeypatch):
    """The validated receipt decides: "" means nothing was heard."""

    features = V03 + ["heard_text"]
    client, provider = await _answered_then_reask(
        monkeypatch,
        lambda eid: {"type": "interrupt", "response_id": eid, "item_id": eid,
                     "played_ms": 900, "heard_text": ""},
        features=features,
    )
    assert [f["outcome"] for f in client.of("reask_result")] == ["accepted"]


@pytest.mark.asyncio
# Supervisor 2026-09-23 18:16: the `no_receipt` case is REMOVED from this pin.
# It asserted `answered` for an answer the phone never confirmed playing —
# receipt absence treated as proof of hearing, which let the relay's
# `answered` verdict silently clear the app's bounded recovery of a request
# nobody heard.  Its new verdict (accepted + conditional, reason
# `unconfirmed`) is pinned by the next test and by test_live_no_receipt_r4.py.
@pytest.mark.parametrize("receipt", [_cut(1200), _idle], ids=["clipped", "drained"])
async def test_control_an_answer_heard_in_part_or_whole_answers_the_request(monkeypatch, receipt):
    """§3.2 pin: clipped after some heard audio → today's receipt rule decides
    (heard → answered); drained is answered as before."""

    client, provider = await _answered_then_reask(monkeypatch, receipt)
    assert [f["outcome"] for f in client.of("reask_result")] == ["answered"]
    assert _reask_appends(provider) == []


@pytest.mark.asyncio
async def test_an_answer_with_no_receipt_is_neither_heard_nor_unheard(monkeypatch):
    """Supervisor 2026-09-23 18:16 — formerly
    `test_control_an_answer_heard_in_part_or_whole_answers_the_request
    [no_receipt]`, which expected `answered`.  No receipt is UNKNOWN: R stays
    recoverable, `accepted` + `conditional: true`, and the one instruction
    claims neither that the answer was heard nor that it "did not reach"
    the caller, and forbids redoing work."""

    client, provider = await _answered_then_reask(monkeypatch, None)
    results = client.of("reask_result")
    assert [f["outcome"] for f in results] == ["accepted"], results
    assert results[0].get("conditional") is True
    appends = _reask_appends(provider)
    assert len(appends) == 1
    content = appends[0]["content"]
    assert "what's the weather in paris" in content and "never confirmed playing it" in content
    assert "did not reach" not in content and "No reply" not in content


@pytest.mark.asyncio
async def test_a_reconnect_carries_an_uncertain_request_conditionally(monkeypatch):
    """§3.4: R, "hello?", a heard reply, then the drop: R is carried with
    `conditional`, and its by-id reask on the new provider session is the
    conditional instruction."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    db = _db("carry-cond")
    box: dict = {}

    async def first_script():
        await _wait(lambda: len(_finals(c1)) >= 1)
        p = box["p"]
        p.push(H.user_delta("hello?", 8000, 8400))
        await _wait(lambda: len(_finals(c1)) >= 2)
        p.push(H.out_text("Yes, I'm here.", 8600, 9200))
        p.push(H.out_audio("A"))
        await _wait(lambda: c1.of("audio_delta"))
        _send(c1, _idle(_epoch_of(c1)))
        await asyncio.sleep(0.3)
        _send(c1, _drop)

    c1 = Phone([H.config(features=V03), first_script, 3.0])
    await H.run_relay(c1, H.FakeProvider(on_send=_silent_after(box, LIBRARY_Q), auto_ack=True),
                      timeout=6, db_session_id=db)
    r_id = _finals(c1)[0]["turn_id"]
    p2 = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
        session_id="live-psid-r4-cond", auto_ack=True,
    )
    c2 = Phone([
        H.config(features=V03), 0.5,
        {"type": "inject_text", "text": LIBRARY_Q, "reason": "no_response",
         "reask_of_user_turn_id": r_id},
        0.4, {"type": "stop"},
    ])
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    results = c2.of("reask_result")
    assert [(f["user_turn_id"], f["outcome"]) for f in results] == [(r_id, "accepted")]
    assert results[0].get("conditional") is True
    content = _reask_appends(p2)[0]["content"]
    assert "Earlier the caller asked" in content and "hello?" in content


@pytest.mark.asyncio
async def test_control_an_old_tasks_output_never_covers_a_new_request(monkeypatch):
    """Watch-causality property on the relay: task A's delivered result,
    parented to A, never answers (or makes uncertain) B — B is plain."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    started = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        started.set()
        await asyncio.sleep(1.2)
        return "Professor A and Professor B.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}
    request_b = "what's the weather in paris tomorrow"

    async def script():
        await started.wait()
        box["p"].push(H.user_delta(request_b, 3000, 3700))
        await _wait(lambda: _terminal(client, "d1"), 4.0)
        await asyncio.sleep(0.8)   # A's result is spoken (its own epoch)
        b = [f for f in _finals(client) if f["text"] == request_b][0]
        _send(client, {
            "type": "inject_text", "text": request_b, "reason": "no_response",
            "reask_of_user_turn_id": b["turn_id"],
        })

    client = Phone([H.config(features=V03), script, 0.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box), auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=_db("old-task"))
    results = client.of("reask_result")
    assert [f["outcome"] for f in results] == ["accepted"], results
    assert "conditional" not in results[0]


# ══════════════════════════════════════════════════════════════════════
# §4.2 — the turn open at the drop is closed and judged before the carry
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("open_at_drop", "carried"), [
    ("can you still hear me?", "request"),    # judged asks-nothing at teardown
    ("so?", "request"),
    ("and one that is open on sunday", "open"),  # CONTROL: a real request open at the drop
])
async def test_the_turn_open_at_a_drop_is_judged_before_the_carry(monkeypatch, open_at_drop, carried):
    H.fast_clocks(
        monkeypatch, voice_live_utterance_gap_ms=3000, voice_live_utterance_hard_gap_ms=6000,
    )
    H.patch_relay(monkeypatch)
    db = _db("teardown", open_at_drop)

    def first_send(p, e):
        if e["type"] == "session.start":
            p.push(H.user_delta(UNANSWERED, 0, 900))
            p.push(H.user_delta(open_at_drop, 5000, 5600))   # still open at the drop
        elif e["type"] == "session.close":
            p.push(H.closed())

    c1 = Phone([H.config(features=V03), 0.6, _drop])
    await H.run_relay(c1, H.FakeProvider(on_send=first_send, auto_ack=True), timeout=6,
                      db_session_id=db)
    finals = _finals(c1)
    assert [f["text"] for f in finals] == [UNANSWERED, open_at_drop], finals
    wanted = finals[0] if carried == "request" else finals[1]
    p2 = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
        session_id="live-psid-r4-td", auto_ack=True,
    )
    c2 = Phone([
        H.config(features=V03), 0.5,
        {"type": "inject_text", "text": wanted["text"], "reason": "no_response",
         "reask_of_user_turn_id": wanted["turn_id"]},
        0.4, {"type": "stop"},
    ])
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    assert [(f["user_turn_id"], f["outcome"]) for f in c2.of("reask_result")] == [
        (wanted["turn_id"], "accepted"),
    ]


# ══════════════════════════════════════════════════════════════════════
# §5.1/§5.2 — value fragments and question subjects (pure + wire)
# ══════════════════════════════════════════════════════════════════════

BOOKING = {
    "transcript": "book a dentist appointment for wednesday at 2pm", "related_request": "",
    "answer": "Booked for Wednesday at 2pm with Dr. Lee.",
}
PROFESSORS = "find iranian computer science professors at the university of toronto"
PROF_RECORD = {
    "transcript": PROFESSORS, "related_request": "",
    "answer": "Professor X and Professor Y, both at St. George.",
}


@pytest.mark.parametrize(("text", "record"), [
    ("no, thursday at 3pm", BOOKING),
    ("no, make it thursday at three", BOOKING),
    ("no, next tuesday at 10am", BOOKING),
    ("نه فردا صبح", {"transcript": "برای امروز وقت دندونپزشکی بگیر", "answer": "برای امروز وقت گرفتم."}),
    ("نه، فردا ساعت پنج", {"transcript": "برای امروز وقت دندونپزشکی بگیر", "answer": "برای امروز وقت گرفتم."}),
    ("نه، پنجشنبه ساعت ۳", {"transcript": "برای چهارشنبه وقت دندونپزشکی بگیر", "answer": "گرفتم."}),
    ("no, eight thirty", {"transcript": "set an alarm for 7am", "answer": "Alarm set for 7am."}),
    ("no, 8", {"transcript": "set an alarm for 7am", "answer": "Alarm set for 7am."}),   # any digit token
])
def test_a_value_only_fragment_corrects_the_finished_task(text, record):
    """relay-phatic-5 (regression): negation + a day/time substitution."""

    assert live._LiveSession.modifies_record(None, text, record), text


@pytest.mark.parametrize(("text", "record"), [
    ("no, what's the weather in toronto right now?", PROF_RECORD),
    ("no, what's the weather in toronto?", PROF_RECORD),
    ("no, how do i get to the university of toronto from union station?", PROF_RECORD),
    ("no, what's on wednesday?", BOOKING),
    ("نه، هوای تورنتو چطوره؟", {"transcript": "استادهای کامپیوتر دانشگاه تورنتو رو پیدا کن", "answer": ""}),
])
def test_a_question_with_a_subject_of_its_own_relates_to_nothing(text, record):
    """relay-phatic-6: a shared place word is no evidence for a question."""

    assert not live._LiveSession.modifies_record(None, text, record), text


@pytest.mark.parametrize(("text", "record"), [
    ("no, where is he a professor now?", PROF_RECORD),
    ("no, which professors are at st george?", PROF_RECORD),
    ("no, can you book it for thursday instead?", BOOKING),
    ("no, thursday", BOOKING),
    ("no, only the downtown campus", PROF_RECORD),
    ("no, what time is it in tokyo right now?", None),
])
def test_control_same_subject_questions_still_link_and_tokyo_does_not(text, record):
    if record is None:
        assert not live._LiveSession.modifies_record(None, text, PROF_RECORD)
        return
    assert live._LiveSession.modifies_record(None, text, record), text


async def _correction_call(monkeypatch, first, followup, *, running, db=""):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[str] = []
    killed: list[str] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        calls.append(task)
        if did == "d1" and running:
            try:
                await asyncio.sleep(4.0)
            except asyncio.CancelledError:
                killed.append(did)
                raise
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def follow():
        if running:
            await _wait(lambda: len(calls) == 1)
        else:
            await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(followup, 5000, 5800))
        box["p"].push(H.delegation("d2", 5850))
        await _wait(lambda: _terminal(client, "d2"), 4.0)
        await asyncio.sleep(0.3)

    client = Phone([H.config(), follow, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, first), auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=db or _db("corr", first, followup, running))
    return client, provider, calls, killed


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", ["no, thursday at 3pm", "no, make it thursday at three"])
@pytest.mark.parametrize("running", [False, True], ids=["finished", "running"])
async def test_a_day_and_time_correction_replaces_the_booking(monkeypatch, followup, running):
    """relay-phatic-5 on both paths: linked `replaces` (finished — never
    cancelled) / superseded (running — never a second booking)."""

    client, _provider, calls, killed = await _correction_call(
        monkeypatch, BOOKING["transcript"], followup, running=running,
    )
    d2 = _lifecycle(_frames_for(client, "d2"))
    assert d2 and all(f.get("relation") == {"kind": "replaces", "task_id": "d1"} for f in d2), d2
    assert "Request the caller is correcting" in calls[-1]
    if running:
        assert "superseded" in _phases(client, "d1")
    else:
        assert "superseded" not in _phases(client, "d1") and "cancelled" not in _phases(client, "d1")


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [False, True], ids=["finished", "running"])
async def test_a_question_sharing_a_place_word_never_links_or_supersedes(monkeypatch, running):
    """relay-phatic-6: destructive on the running path before."""

    client, provider, calls, killed = await _correction_call(
        monkeypatch, PROFESSORS, "no, what's the weather in toronto right now?", running=running,
    )
    assert killed == []
    assert "superseded" not in _phases(client, "d1")
    assert all("relation" not in f for f in _lifecycle(_frames_for(client, "d2")))
    assert "Request the caller is correcting" not in calls[-1]
    # A question of its own is not an ambiguous correction: nothing is asked.
    assert not any("Should I stop" in c for c in _commentary(provider))


# ══════════════════════════════════════════════════════════════════════
# §5.3 — PARTIAL evidence: ask "Should I stop «A»?" before cancelling
# ══════════════════════════════════════════════════════════════════════

# PIN CHANGED (R2 addendum 6 R6-8, the integrator's resolution of the C11
# spec conflict): 'no, book me a hotel in toronto' was this suite's PARTIAL
# trigger, but it is SELF-CONTAINED (its own predicate and head) and shares
# only 'toronto' — one of its three own words — so it is now NONE: no stop
# question, no binding, it runs normally (pinned in test_live_round6.py and
# test_live_c11_r5.py as `PARTIAL_R5`).  The §5.3 confirmation suites keep
# exercising a real PARTIAL: a self-contained correction sharing at least half
# of its own words (university + toronto of dorm / university / toronto +
# its predicate) with the running search — and with the SHORT/DOC variants
# the probes use ("find professors at the university of toronto", "find the
# office hours of dr. smith at the university of toronto").
PARTIAL = "no, find a dorm at the university of toronto"
PARTIAL_FA = "نه، یه خوابگاه تو دانشگاه تورنتو پیدا کن"
#: The round-5 triggers, NONE since R6-8 (kept for the pins that say so).
PARTIAL_R5 = "no, book me a hotel in toronto"
PARTIAL_FA_R5 = "نه، یه هتل تو تورنتو برام رزرو کن"
A_FA = "استادهای ایرانی کامپیوتر دانشگاه تورنتو رو پیدا کن"
QUESTION_MARKS = ("Should I stop", "Sorry — should I stop", "رو متوقف کنم")


def _is_question(content: str) -> bool:
    return any(mark in content for mark in QUESTION_MARKS)


async def _confirm_call(
    monkeypatch, steps, *, features=V03, first=PROFESSORS, partial=PARTIAL,
    release_a_early=False, db="",
):
    """A (d1) runs; the caller's PARTIAL correction is delegated (d2); the
    model says the relay's question aloud; `steps(ctx)` then plays the phone
    and the caller.  Returns ctx with client/provider/calls/killed."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    # The relay acts on an unclaimed answer after the backstop grace.
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    ctx: dict = {"calls": [], "killed": [], "questions": []}
    a_done = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        ctx["calls"].append((did, kw.get("display_request") or ""))
        if did == "d1":
            try:
                await asyncio.wait_for(a_done.wait(), timeout=6.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                ctx["killed"].append(did)
                raise
            return "Professor X.", "m"
        if did == "d2":
            await asyncio.sleep(2.5)
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)
    ctx["a_done"] = a_done
    box: dict = {}
    clock = {"at": 20000}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(first, 0, 900))
            p.push(H.delegation("d1", 950))
        elif e["type"] == "session.commentary.append" and _is_question(str(e.get("content") or "")):
            # The model says the relay's question (its own epoch).
            start = clock["at"]
            ctx["questions"].append((start, start + 1500))
            p.push(H.out_text(str(e["content"]), start, start + 1500))
            p.push(H.out_audio("Q"))
            clock["at"] += 6000
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: len(ctx["calls"]) >= 1)
        p = box["p"]
        p.push(H.user_delta(partial, 5000, 5800))
        p.push(H.delegation("d2", 5850))
        ok = await _wait(lambda: ctx["questions"] and client.of("audio_delta"), 3.0)
        ctx["asked"] = bool(ok)
        if release_a_early:
            a_done.set()
            await _wait(lambda: _terminal(client, "d1"), 3.0)
        await steps(ctx, p, client)
        await asyncio.sleep(0.6)
        a_done.set()
        await _wait(lambda: _terminal(client, "d1") and _terminal(client, "d2"), 5.0)
        await asyncio.sleep(0.3)

    client = Phone([H.config(features=features), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    ctx["client"], ctx["provider"] = client, provider
    await H.run_relay(client, provider, timeout=20, db_session_id=db or _db("confirm", repr(steps), partial))
    return ctx


def _heard(client):
    """The phone played the question epoch to the end."""

    return _idle(_epoch_of(client))


def _answer(text, at_offset=2500, *, receipt=_heard, delegate=False):
    async def steps(ctx, p, client):
        if receipt is not None:
            _send(client, receipt(client))
            await asyncio.sleep(0.2)
        q_start, q_end = ctx["questions"][-1]
        p.push(H.user_delta(text, q_end + at_offset, q_end + at_offset + 500))
        if delegate:
            p.push(H.delegation("d-answer", q_end + at_offset + 550))
        await asyncio.sleep(0.8)
    return steps


def _cancelled_a(ctx) -> bool:
    frames = _frames_for(ctx["client"], "d1")
    return any(f.get("phase") == "cancelled" and f.get("reason") == "user_cancel" for f in frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes", "yes stop it", "cancel it"])
async def test_a_partial_correction_asks_and_a_yes_stops_the_old_task(monkeypatch, answer):
    ctx = await _confirm_call(monkeypatch, _answer(answer))
    client, provider = ctx["client"], ctx["provider"]
    assert ctx["asked"], "the relay never asked before cancelling"
    questions = [c for c in _commentary(provider) if _is_question(c)]
    assert len(questions) == 1 and "find iranian computer science professors" in questions[0]
    assert _cancelled_a(ctx), _phases(client, "d1")
    d2 = _lifecycle(_frames_for(client, "d2"))
    # Separate at first (no relation), then the LATEST relation replaces A.
    assert "relation" not in d2[0]
    assert d2[-1].get("relation") == {"kind": "replaces", "task_id": "d1"}, d2[-1]
    assert any("Okay, I stopped" in c for c in _commentary(provider))
    assert [did for did, _d in ctx["calls"]] == ["d1", "d2"]
    _dump(f"confirm_yes:{answer}", client)


@pytest.mark.asyncio
async def test_a_persian_partial_correction_asks_in_persian_and_a_yes_stops(monkeypatch):
    ctx = await _confirm_call(
        monkeypatch, _answer("آره قطعش کن"), first=A_FA, partial=PARTIAL_FA,
    )
    questions = [c for c in _commentary(ctx["provider"]) if _is_question(c)]
    assert questions and "رو متوقف کنم؟" in questions[0]
    assert _cancelled_a(ctx)
    assert any("رو متوقف کردم" in c for c in _commentary(ctx["provider"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["no", "keep it", "both", "نه"])
async def test_a_no_keeps_both_tasks(monkeypatch, answer):
    ctx = await _confirm_call(monkeypatch, _answer(answer))
    assert not _cancelled_a(ctx)
    assert _phases(ctx["client"], "d1")[-1] == "completed"
    assert all("relation" not in f for f in _lifecycle(_frames_for(ctx["client"], "d2")))
    assert any("keep both" in c for c in _commentary(ctx["provider"]))


@pytest.mark.asyncio
async def test_a_mixed_answer_stops_a_and_runs_the_rest_under_its_own_title(monkeypatch):
    """'yes, and …': the leading answer is applied, the remainder is the
    caller's own request — never titled "yes, and …"."""

    ctx = await _confirm_call(
        monkeypatch, _answer("yes, and search for waterloo robotics professors", delegate=True),
    )
    client = ctx["client"]
    assert _cancelled_a(ctx)
    created = [f for f in _frames_for(client, "d-answer") if f.get("phase") == "created"]
    assert created and created[0]["title"] == "search for waterloo robotics professors", created
    assert ("d-answer", "search for waterloo robotics professors") in ctx["calls"]


@pytest.mark.asyncio
async def test_a_pure_answer_the_model_delegates_is_absorbed(monkeypatch):
    ctx = await _confirm_call(monkeypatch, _answer("yes", delegate=True))
    client, provider = ctx["client"], ctx["provider"]
    assert _cancelled_a(ctx)
    assert _frames_for(client, "d-answer") == [], "a card for the answer"
    assert [did for did, _d in ctx["calls"]] == ["d1", "d2"]
    notes = [e for e in _thinking(provider) if e.get("delegation_id") == "d-answer"]
    assert notes and "handled" in notes[0]["content"]


@pytest.mark.asyncio
async def test_assent_to_a_newer_intervening_question_never_cancels(monkeypatch):
    async def steps(ctx, p, client):
        _send(client, _heard(client))
        await asyncio.sleep(0.2)
        q_start, q_end = ctx["questions"][-1]
        p.push(H.out_text("Do you also want their email addresses?", q_end + 500, q_end + 1500))
        p.push(H.out_audio("N"))
        await asyncio.sleep(0.4)
        p.push(H.user_delta("yes", q_end + 3000, q_end + 3400))
        await asyncio.sleep(0.8)

    ctx = await _confirm_call(monkeypatch, steps)
    assert not _cancelled_a(ctx)
    assert _phases(ctx["client"], "d1")[-1] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [
    lambda client: _cut(0)(_epoch_of(client)),
    None,
], ids=["interrupted", "no_receipt"])
async def test_an_unheard_question_binds_no_answer(monkeypatch, receipt):
    ctx = await _confirm_call(monkeypatch, _answer("yes", receipt=receipt))
    assert not _cancelled_a(ctx)
    assert _phases(ctx["client"], "d1")[-1] == "completed"


@pytest.mark.asyncio
async def test_a_task_that_finished_before_the_answer_cancels_nothing(monkeypatch):
    ctx = await _confirm_call(monkeypatch, _answer("yes"), release_a_early=True)
    assert _phases(ctx["client"], "d1")[-1] == "completed"
    assert not _cancelled_a(ctx)
    assert not any("Okay, I stopped" in c for c in _commentary(ctx["provider"]))


@pytest.mark.asyncio
async def test_an_answer_after_the_expiry_cancels_nothing(monkeypatch):
    monkeypatch.setattr(live, "_CONFIRM_EXPIRY_MS", 1000, raising=False)
    ctx = await _confirm_call(monkeypatch, _answer("yes", at_offset=4000))
    assert not _cancelled_a(ctx)


@pytest.mark.asyncio
async def test_yes_but_wait_never_cancels_and_asks_once_more(monkeypatch):
    ctx = await _confirm_call(monkeypatch, _answer("yes but wait"))
    questions = [c for c in _commentary(ctx["provider"]) if _is_question(c)]
    assert len(questions) == 2 and questions[1].startswith("Sorry — should I stop"), questions
    assert not _cancelled_a(ctx)


@pytest.mark.asyncio
async def test_after_one_reclarification_a_clear_yes_still_stops(monkeypatch):
    async def steps(ctx, p, client):
        await _answer("yes but wait")(ctx, p, client)
        await _wait(lambda: len(ctx["questions"]) >= 2, 3.0)
        await _wait(lambda: len(client.of("audio_delta")) >= 2, 2.0)
        _send(client, _heard(client))
        await asyncio.sleep(0.2)
        q_start, q_end = ctx["questions"][-1]
        p.push(H.user_delta("yes", q_end + 1500, q_end + 1900))
        await asyncio.sleep(0.8)

    ctx = await _confirm_call(monkeypatch, steps)
    assert _cancelled_a(ctx)


@pytest.mark.asyncio
async def test_the_echo_of_the_relays_question_is_not_an_answer(monkeypatch):
    """The caller's mic picks up the question itself; that is echo, never an
    answer (and never voids the question) — the caller's own 'yes' after it
    still stops A."""

    async def steps(ctx, p, client):
        q_start, q_end = ctx["questions"][-1]
        question = [c for c in _commentary(ctx["provider"]) if _is_question(c)][-1]
        p.push(H.user_delta(question, q_start + 150, q_end + 150))   # echo while it plays
        await asyncio.sleep(0.3)
        _send(client, _heard(client))
        await asyncio.sleep(0.3)
        p.push(H.user_delta("yes", q_end + 3000, q_end + 3400))
        await asyncio.sleep(0.8)

    ctx = await _confirm_call(monkeypatch, steps)
    echo = [f for f in _finals(ctx["client"]) if "Should I stop" in f["text"]]
    assert echo and echo[0].get("echo_only") is True, _finals(ctx["client"])
    assert _cancelled_a(ctx)


@pytest.mark.asyncio
async def test_tf132_voice_only_confirmation_works(monkeypatch):
    ctx = await _confirm_call(monkeypatch, _answer("yes"), features=TF132)
    assert _cancelled_a(ctx)
    assert ctx["client"].of("reask_result") == []


@pytest.mark.asyncio
async def test_a_confirmation_is_never_reapplied_by_a_reask(monkeypatch):
    async def steps(ctx, p, client):
        await _answer("yes")(ctx, p, client)
        await asyncio.sleep(1.0)
        answer = [f for f in _finals(client) if f["text"] == "yes"][0]
        _send(client, {
            "type": "inject_text", "text": "yes", "reason": "no_response",
            "reask_of_user_turn_id": answer["turn_id"],
        })
        await asyncio.sleep(0.5)

    ctx = await _confirm_call(monkeypatch, steps)
    assert _cancelled_a(ctx)
    assert [f["outcome"] for f in ctx["client"].of("reask_result")] == ["answered"]
    assert _reask_appends(ctx["provider"]) == []
    stopped = [c for c in _commentary(ctx["provider"]) if "Okay, I stopped" in c]
    assert len(stopped) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("followup", "superseded"), [
    ("no, only the downtown campus", True),              # FULL evidence: supersede, no question
    ("what time is it in tokyo right now", False),       # unrelated: separate, no question
])
async def test_control_full_corrections_and_unrelated_requests_ask_nothing(
    monkeypatch, followup, superseded,
):
    client, provider, calls, killed = await _correction_call(
        monkeypatch, PROFESSORS, followup, running=True,
    )
    assert not any(_is_question(c) for c in _commentary(provider))
    assert ("superseded" in _phases(client, "d1")) is superseded


# ══════════════════════════════════════════════════════════════════════
# §6 — causal play evidence, the full fire, honest wording, the fence
# ══════════════════════════════════════════════════════════════════════

class Tenant:
    """A realistic tenant: a play takes the halt mark when it starts and is
    dropped (superseded) if a stop reached the tenant meanwhile; a stop with
    nothing audible answers `nothing_playing`."""

    def __init__(self, *, search_s=1.6, audible=False):
        self.search_s = search_s
        self.audible = audible
        self.halted = 0
        self.timeline: list[tuple[str, str]] = []

    async def play(self, _uid, query, variety=False):
        from app.api import ws_realtime as rt

        mark = self.halted
        self.timeline.append(("play_requested", query))
        await asyncio.sleep(self.search_s)
        if self.halted != mark:
            self.timeline.append(("play_dropped", query))
            return rt.PLAY_MEDIA_SUPERSEDED, None
        self.timeline.append(("play_broadcast", query))
        self.audible = True
        return (f"Starting {query}.", {"type": "youtube", "video_id": "JAZZ0000001", "title": "Jazz Mix"})

    async def control(self, _uid, action):
        self.timeline.append(("control", action))
        self.halted += 1
        if not self.audible:
            return {"ok": False, "action": action, "reason": "nothing_playing"}
        self.audible = False
        return {"ok": True, "action": action, "reason": "stopped"}


async def _media_call(monkeypatch, steps, *, tenant, frames=(), features=V03, db=""):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    H.patch_relay(monkeypatch, play=tenant.play, control=tenant.control)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await asyncio.sleep(0.2)
        await steps(box["p"], client)

    client = Phone([H.config(features=features), *frames, script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=14, db_session_id=db or _db("media", repr(steps), repr(frames)))
    return client, provider


def _said(provider) -> list[str]:
    return _commentary(provider) + [
        str(e.get("content") or "") for e in provider.of("session.instructions.append")
    ]


def _play_then_stop(stop, ack, *, stop_at=2000):
    async def steps(p, client):
        p.push(H.user_delta("play some jazz", 0, 500))
        p.push(H.delegation("dPlay", 520))
        await asyncio.sleep(0.4)
        p.push(H.user_delta(stop, stop_at, stop_at + 600))
        p.push(H.out_text(ack, stop_at + 700, stop_at + 900))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(3.0)
    return steps


@pytest.mark.asyncio
@pytest.mark.parametrize(("stop", "ack", "lang"), [
    ("stop the music", "Okay.", "en"), ("آهنگ رو قطع کن", "چشم.", "fa"),
])
async def test_an_acknowledged_stop_prevents_a_play_still_searching(monkeypatch, stop, ack, lang):
    """relay-media-1: the relay's own in-flight play is causal evidence; the
    backstop fires fully (halt + tenant stop), the play never starts, and the
    wording is honest: "I won't start «…»" — never "Nothing was playing"."""

    tenant = Tenant(search_s=1.6)
    client, provider = await _media_call(monkeypatch, _play_then_stop(stop, ack), tenant=tenant)
    assert ("control", "stop") in tenant.timeline, tenant.timeline
    assert ("play_broadcast", "some jazz") not in tenant.timeline, tenant.timeline
    assert "completed" not in _phases(client, "dPlay")
    said = _said(provider)
    assert not any("Jazz Mix" in s for s in said)
    wont = "won't start «some jazz»" if lang == "en" else "«some jazz» رو پخش نمی‌کنم"
    assert any(wont in s for s in said), said
    assert not any("Nothing was playing" in s or "چیزی در حال پخش نبود" in s for s in said)
    _dump(f"media_pending_stop:{lang}", client)


@pytest.mark.asyncio
async def test_a_queued_play_is_dropped_by_an_acknowledged_stop(monkeypatch):
    """relay-media-1 (queued at capacity): never searched."""

    release = asyncio.Event()
    tenant = Tenant(search_s=0.1)
    started: list[str] = []

    async def think(_u, _t, _s, **kwargs):
        started.append(kwargs["delegation_id"])
        await release.wait()
        return "A long answer.", "m"

    async def steps(p, client):
        p.push(H.user_delta("find the history of jazz in new orleans", 0, 400))
        p.push(H.delegation("d1", 450))
        await _wait(lambda: "d1" in started)
        p.push(H.user_delta("what is the weather in paris tomorrow", 1500, 1900))
        p.push(H.delegation("d2", 1950))
        await _wait(lambda: "d2" in started)
        p.push(H.user_delta("play some jazz", 3000, 3400))
        p.push(H.delegation("dPlay", 3450))
        await _wait(lambda: "pending" in _phases(client, "dPlay"))
        p.push(H.user_delta("آهنگ رو قطع کن", 4500, 4900))
        p.push(H.out_text("چشم.", 5000, 5200))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.2)
        release.set()
        await asyncio.sleep(1.0)

    H.fast_clocks(monkeypatch)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    H.patch_relay(monkeypatch, think=think, play=tenant.play, control=tenant.control)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await asyncio.sleep(0.2)
        await steps(box["p"], client)

    client = Phone([H.config(features=V03), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    try:
        await H.run_relay(client, provider, timeout=14, db_session_id=_db("queued-play"))
    finally:
        release.set()
    assert not any(kind.startswith("play") for kind, _q in tenant.timeline), tenant.timeline
    frames = _frames_for(client, "dPlay")
    assert any(f.get("phase") == "cancelled" and f.get("reason") == "media_stopped" for f in frames)


@pytest.mark.asyncio
async def test_a_stop_right_after_an_announced_play_is_carried_out(monkeypatch):
    """relay-media-2: the play completed and was announced; the phone is still
    loading (no `now_playing` yet); the stop reaches the tenant."""

    tenant = Tenant(search_s=0.05)

    async def steps(p, client):
        p.push(H.user_delta("play some jazz", 0, 500))
        p.push(H.delegation("dPlay", 520))
        await _wait(lambda: "completed" in _phases(client, "dPlay"))
        await asyncio.sleep(0.2)
        p.push(H.user_delta("stop the music", 3000, 3500))
        p.push(H.out_text("Okay.", 3600, 3800))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.6)

    client, provider = await _media_call(monkeypatch, steps, tenant=tenant)
    assert ("control", "stop") in tenant.timeline, tenant.timeline
    assert any("Stopped the music." in s for s in _said(provider))


@pytest.mark.asyncio
@pytest.mark.parametrize("contradiction", ["stopped", "failed"])
async def test_an_announced_play_the_phone_contradicted_is_no_evidence(monkeypatch, contradiction):
    tenant = Tenant(search_s=0.05)

    async def steps(p, client):
        p.push(H.user_delta("play some jazz", 0, 500))
        p.push(H.delegation("dPlay", 520))
        await _wait(lambda: "completed" in _phases(client, "dPlay"))
        await asyncio.sleep(0.2)
        if contradiction == "stopped":
            _send(client, {"type": "now_playing", "title": "", "state": "stopped", "source": "media_x"})
        else:
            _send(client, {"type": "playback_failed", "video_id": "JAZZ0000001", "reason": "blocked"})
        await asyncio.sleep(0.3)
        p.push(H.user_delta("stop the music", 3000, 3500))
        p.push(H.out_text("Okay.", 3600, 3800))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.4)

    client, provider = await _media_call(monkeypatch, steps, tenant=tenant)
    assert ("control", "stop") not in tenant.timeline, tenant.timeline


@pytest.mark.asyncio
async def test_control_nothing_reported_and_nothing_started_never_calls_the_tenant(monkeypatch):
    tenant = Tenant()

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 3000, 3500))
        p.push(H.out_text("Okay.", 3600, 3800))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.4)

    client, provider = await _media_call(monkeypatch, steps, tenant=tenant)
    assert tenant.timeline == []


@pytest.mark.asyncio
async def test_a_play_asked_after_the_stop_is_never_halted_by_it(monkeypatch):
    """§6 newer-play fence: music is playing; "stop the music" (acknowledged
    only), then — within the grace — "play halo".  The newer intent wins: the
    backstop never stops the new play."""

    tenant = Tenant(search_s=0.2, audible=True)

    async def steps(p, client):
        # Production grace: the newer request lands inside it.
        settings.voice_live_media_backstop_grace_ms = 1500
        p.push(H.user_delta("stop the music", 3000, 3500))
        await _wait(lambda: _finals(client))
        p.push(H.out_text("Okay.", 3600, 3700))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.05)
        p.push(H.user_delta("play halo by beyonce", 4000, 4500))
        p.push(H.delegation("dHalo", 4550))
        await asyncio.sleep(2.2)

    client, provider = await _media_call(
        monkeypatch, steps, tenant=tenant, frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    assert ("play_broadcast", "halo by beyonce") in tenant.timeline, tenant.timeline
    # PIN CHANGED (R2 addendum 6 R6-4: every stop is sent at fire time, scoped
    # to the plays asked for before it — never withheld): the old "the tenant
    # is never called" pinned the §6 fence that withheld the stop.  The stop
    # now reaches the tenant BEFORE the newer play's request and never after
    # its broadcast, so the newer play is never halted by it.
    kinds = [kind for kind, _what in tenant.timeline]
    assert "control" not in kinds[kinds.index("play_broadcast"):], tenant.timeline
    if "control" in kinds:
        assert kinds.index("control") < kinds.index("play_requested"), tenant.timeline


@pytest.mark.asyncio
async def test_a_titled_paused_report_is_not_playback(monkeypatch):
    """§6.3/§6.4: a titled 'paused' frame is not evidence and adds no "Now
    playing" note."""

    tenant = Tenant()

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 3000, 3500))
        p.push(H.out_text("Okay.", 3600, 3800))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.4)

    client, provider = await _media_call(
        monkeypatch, steps, tenant=tenant,
        frames=({"type": "now_playing", "title": "Track A — Artist", "state": "paused"},),
    )
    assert tenant.timeline == []
    assert not any("Now playing" in str(e.get("content") or "") for e in _thinking(provider))


# ══════════════════════════════════════════════════════════════════════
# §7 — media units
# ══════════════════════════════════════════════════════════════════════

async def _backstop(monkeypatch, turns, *, ack="Okay.", late=False, step_ms=1800, db=""):
    """Music is playing; the caller says `turns` (each its own turn); the
    model only acknowledges; optionally delegates late."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    controls, thinks = [], []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think, control=control)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        at = 3000
        for text in turns:
            p.push(H.user_delta(text, at, at + 500))
            at += step_ms
            await asyncio.sleep(0.5)
        p.push(H.out_text(ack, at, at + 200))
        p.push(H.out_audio("OK"))
        if late:
            await asyncio.sleep(0.6)
            p.push(H.delegation("d-late", at + 300))
        await asyncio.sleep(1.4)

    client = Phone([
        H.config(features=V03), {"type": "now_playing", "title": "Fadat Sham - Mahasti"},
        script, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=db or _db("bs", tuple(turns), late))
    return client, controls, thinks


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["کن", "کن لطفا"])
async def test_the_tail_of_a_fired_command_is_part_of_it(monkeypatch, tail):
    """relay-media-3: «آهنگ رو قطع» fires; «کن» follows; the late delegation
    for the command is absorbed — never run as «کن», no card."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    controls, thinks = [], []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or task)
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think, control=control)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        p.push(H.user_delta("آهنگ رو قطع", 3000, 3600))
        await asyncio.sleep(0.8)
        p.push(H.user_delta(tail, 4900, 5200))
        p.push(H.out_text("چشم.", 5300, 5500))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.6)
        p.push(H.delegation("d-late", 5600))
        await asyncio.sleep(1.4)

    client = Phone([
        H.config(features=V03), {"type": "now_playing", "title": "Fadat Sham - Mahasti"},
        script, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=_db("tail", tail))
    assert controls == ["stop"], controls
    assert thinks == []
    assert client.of("delegation") == []


@pytest.mark.parametrize(("head", "tail", "absorbed"), [
    ("آهنگ رو قطع", "کن", True),
    ("آهنگ رو قطع", "نکن", False),            # a negation is never absorbed
    ("آهنگ رو قطع", "قطع کن", False),         # a repeated command is its own
])
def test_a_tail_joins_only_when_it_keeps_the_same_named_command(head, tail, absorbed):
    joined = f"{head} {tail}"
    same = live.media_command_evidence(joined) == ("stop", True)
    own = live.media_command_evidence(tail)[0] is not None or getattr(
        live, "media_heads_present", lambda _t: True)(tail)
    assert (same and not own) is absorbed


@pytest.mark.asyncio
@pytest.mark.parametrize(("turns", "stops"), [
    (["stop the", "um", "music"], True),                  # a hesitation inside the phrase
    (["آهنگ رو", "خب", "قطع کن"], True),
    (["Stop.", "The music."], True),                      # punctuated halves: whole-span fallback
    (["Hmm.", "stop the music"], True),                   # CONTROL: lead-in unchanged
    (["don't", "um", "stop the music"], False),           # the safety mirror (old: EXECUTED)
    (["What is the weather in Paris?", "The music."], False),
    (["Stop.", "The search."], False),
    (["Don't.", "The music."], False),
    (["don't", "stop the music"], False),
    (["Hmm.", "The music."], False),
])
async def test_media_units_read_the_whole_phrase_and_nothing_else(monkeypatch, turns, stops):
    client, controls, _thinks = await _backstop(monkeypatch, turns)
    assert controls == (["stop"] if stops else []), (turns, controls)


@pytest.mark.parametrize("words", [
    "باشه آهنگ رو قطع کن", "آره آهنگ رو قطع کن", "ببین آهنگ رو قطع کن", "مرسی آهنگ رو قطع کن",
    "yes stop the music", "yes, stop the music", "thanks, stop the music",
])
def test_content_free_tokens_are_media_filler(words):
    """relay-media-5: one closed class, two grammars."""

    assert live.media_command_evidence(words) == ("stop", True), words


@pytest.mark.parametrize(("words", "expected"), [
    ("yeah next song", ("next", True)),
    ("alright, next song", ("next", True)),
    ("yes next", ("next", False)),                 # bare: never backstopped
    ("آره بعدی", ("next", False)),
    ("no, stop the music", (None, False)),         # negations stay out
    ("don't stop the music", (None, False)),
])
def test_control_negations_stay_out_and_bare_words_stay_bare(words, expected):
    assert live.media_command_evidence(words) == expected, words


@pytest.mark.asyncio
@pytest.mark.parametrize("words", ["باشه آهنگ رو قطع کن", "yes, stop the music"])
async def test_an_acknowledgement_lead_in_inside_the_stop_turn_is_backstopped(monkeypatch, words):
    client, controls, _thinks = await _backstop(monkeypatch, [words], ack="چشم." if words[0] > "z" else "Okay.")
    assert controls == ["stop"]
