"""R2 addendum 6 (round-5 findings) — the relay's classifiers and readers.

R6-1  ONE confirmation-answer classifier: (kind, request) over the ORIGINAL
      clause boundaries with the ONE asks-nothing verdict; consumers follow
      `request` only (a pure answer — thanks, a presence check or a restated
      answer after it — consumes the turn, is answered by the outcome line and
      is never run as a task); a real status question after an answer stays a
      request ('yes, what did you find?'); begin() defense in depth.
R6-2  The §5.3 question reader: a question sentence ENDS in ?/؟ (titles
      masked, no split on "."), inversion only for unpunctuated text, the
      designed-count rule; real newer questions still void.  The same reader
      binds C11's which-question.
R6-3  The teardown fragment classifier: punctuation closes the clause, the
      Persian clitic reading needs OBJECT position, and a bare NP with the
      clitic letter shape is complete only when the call's own history vouches
      for the token (else A4's "cannot place": truncated, visible).
R6-8  C11 corrections: pleasantry/copula framing is never anaphora; «چطوریه»
      is a predicate; repair markers are context-dependent; self-contained
      follow-ups keep the proportion grade (the resolved conflict: 'no, book
      me a hotel in toronto' is NONE); target selection over running AND
      finished jobs with lexical evidence first; which-question answers are
      only selection clauses; C12's continuation claim is tentative until the
      caller's input for its window settled (output_first, both ways).

Wire tests run the REAL relay (`test_live_harness`, a distinct provider
session id per socket); pure tests grade the shared helpers.  Old-fail: this
file run against a copy of fx4-relay-snapF (lvp 052f1459…): 153 failed, every
one by assertion; the 104 that pass there are the controls and the unchanged
round-5 pins carried in the same tables (real newer questions still void,
truncated turns stay fragments, the C1 agreeing answers, the NONE controls
with two jobs whose one shared word ties).
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

import test_live_harness as H
import test_live_no_receipt_r4 as NR
import test_live_round4 as R4
from app.config import settings
from app.services import live_voice_protocol as live


PROFESSORS = R4.PROFESSORS
A_FA = R4.A_FA
HOTEL = "find me a hotel near union station"
HOTEL_FA = "یه هتل نزدیک یونیون استیشن پیدا کن"
CAMPUS = "no, what about the downtown campus?"
DOC = "find the office hours of dr. smith at the university of toronto"
#: A real PARTIAL correction of the professors search (R6-8: self-contained,
#: shares university + toronto — at least half of its own words).  Explicit
#: here, so this file means the same on a tree whose R4.PARTIAL is older.
PARTIAL_EN = "no, find a dorm at the university of toronto"
PARTIAL_FA = "نه، یه خوابگاه تو دانشگاه تورنتو پیدا کن"
BINDING = "refer back to"
V03 = R4.V03
TF132 = R4.TF132
Phone = R4.Phone
_frames_for = R4._frames_for
_phases = R4._phases
_lifecycle = R4._lifecycle
_commentary = R4._commentary
_thinking = R4._thinking
_wait = R4._wait
_send = R4._send
_finals = R4._finals
WHICH_MARKS = ("Which one do you mean", "Sorry — which one", "منظورت کدومه", "ببخشید — کدومش")


def _tag(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _db(name: str, *parts) -> str:
    return f"db-r6-{name}-{_tag(*parts)}"


def _psid(name: str, *parts) -> str:
    return f"live-psid-r6-{name}-{_tag(*parts)}"


def _is_which(content: str) -> bool:
    return any(mark in content for mark in WHICH_MARKS)


def _which(provider) -> list[str]:
    return [c for c in _commentary(provider) if _is_which(c)]


def _stops(provider) -> list[str]:
    return [c for c in _commentary(provider) if R4._is_question(c)]


def _relations(client, did) -> list:
    return [f.get("relation") for f in _lifecycle(_frames_for(client, did))]


def _errors(client) -> list[dict]:
    """Error frames other than the session's own close at the end."""

    return [f for f in client.of("error") if not str(f.get("code") or "").startswith("live_closed")]


def _notes(provider, did) -> list[str]:
    return [str(e.get("content") or "") for e in _thinking(provider) if e.get("delegation_id") == did]


# ══════════════════════════════════════════════════════════════════════
# The §5.3 confirmation harness: A (d1) runs; the PARTIAL correction is
# delegated (d2); the model SAYS each relay line as its own epoch — rendered
# by `render` (what GPT-Live's transcript says: verbatim, paraphrased, with a
# lead-in or a tail); `steps(ctx, p, client)` plays the phone and the caller.
# ══════════════════════════════════════════════════════════════════════

async def _confirm(
    monkeypatch, steps, *, d2_s=2.5, features=V03, first=PROFESSORS, partial=PARTIAL_EN,
    render=None, db="",
):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    ctx: dict = {"calls": [], "killed": [], "questions": [], "lines": [], "spoken": []}
    a_done = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        ctx["calls"].append((did, kw.get("display_request") or ""))
        if did == "d1":
            try:
                await asyncio.wait_for(a_done.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                ctx["killed"].append(did)
                raise
            return "Professor X.", "m"
        if did == "d2":
            await asyncio.sleep(d2_s)
            return "Booked the Hotel Ocho.", "m"
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}
    clock = {"at": 20000}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(first, 0, 900))
            p.push(H.delegation("d1", 950))
        elif e["type"] == "session.commentary.append":
            content = str(e.get("content") or "")
            spoken = render(content) if render is not None else content
            start = clock["at"]
            if R4._is_question(content):
                ctx["questions"].append((start, start + 1500))
            ctx["lines"].append((content, start, start + 1500))
            ctx["spoken"].append(spoken)
            p.push(H.out_text(spoken, start, start + 1500))
            p.push(H.out_audio("L"))
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
        await steps(ctx, p, client)
        await asyncio.sleep(0.6)
        a_done.set()
        await _wait(lambda: R4._terminal(client, "d1") and R4._terminal(client, "d2"), 6.0)
        await asyncio.sleep(0.5)

    client = Phone([H.config(features=features), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, session_id=_psid("conf", db))
    ctx["client"], ctx["provider"] = client, provider
    await H.run_relay(client, provider, timeout=24, db_session_id=db)
    return ctx


def _epoch_saying(client, marks) -> str:
    said = [
        f for f in client.of("response_text")
        if any(m in (f.get("text") or "") for m in marks) and f.get("assistant_turn_id")
    ]
    if said:
        return said[-1]["assistant_turn_id"]
    return client.of("audio_delta")[-1]["response_id"]


def _question_epoch(client) -> str:
    return _epoch_saying(client, R4.QUESTION_MARKS + ("Should I stop", "stop looking up"))


def _answer(text, *, delegate=False, at_offset=2500, heard=True):
    async def steps(ctx, p, client):
        if heard:
            _send(client, R4._idle(_question_epoch(client)))
            await asyncio.sleep(0.2)
        at = max(end for _line, _start, end in ctx["lines"]) + at_offset
        p.push(H.user_delta(text, at, at + 500))
        if delegate:
            p.push(H.delegation("d-answer", at + 550))
        await asyncio.sleep(0.8)
    return steps


def _cancelled_a(ctx) -> bool:
    return R4._cancelled_a(ctx)


def _questions_said(ctx) -> list[str]:
    return [c for c, *_ in ctx["lines"] if R4._is_question(c)]


STOPPED_MARKS = ("Okay, I stopped", "رو متوقف کردم")
KEPT_MARKS = ("keep both", "هر دو")


def _fa(text: str) -> dict:
    return {} if text.isascii() else {"first": A_FA, "partial": PARTIAL_FA}


# ══════════════════════════════════════════════════════════════════════
# R6-1 — ONE confirmation-answer classifier (pure)
# ══════════════════════════════════════════════════════════════════════

PURE_ANSWERS = [
    ("yes, thank you so much", "assent"), ("okay, thank you so much", "assent"),
    ("آره، خیلی ممنون", "assent"), ("آره، مرسی، خیلی ممنون", "assent"),
    ("yes thank you very much", "assent"), ("yes, hello?", "assent"),
    ("yes, can you hear me?", "assent"),                    # decision: a presence tail is answered
    ("Yes, you can stop it.", "assent"), ("Yes, I want you to stop it.", "assent"),
    ("آره، می‌تونی قطعش کنی", "assent"), ("آره، دیگه لازم نیست", "assent"),
    ("آره، لازمش ندارم", "assent"), ("Yes, I don't need it anymore.", "assent"),
    ("No, let it continue.", "keep"), ("نه، بذار ادامه بده", "keep"),
    ("No, thank you so much.", "keep"), ("نه، خیلی ممنون", "keep"),
    ("Yes, stop it, thank you so much.", "assent"), ("Yes, stop it, thanks so much.", "assent"),
    ("Sure, go ahead, thanks a lot.", "assent"), ("آره، قطعش کن، خیلی ممنون.", "assent"),
    ("No, keep both, thanks a lot.", "keep"), ("No, keep both, thanks so much!", "keep"),
    ("نه، هر دو رو نگه دار.", "keep"), ("نه، هر دو رو نگه‌دار.", "keep"), ("نه، نگهش دار.", "keep"),
    ("No, keep them both.", "keep"),
    # the round-5 C1 pins, unchanged
    ("Yes, stop it.", "assent"), ("Yes, cancel it.", "assent"), ("Sure, go ahead.", "assent"),
    ("آره، قطعش کن.", "assent"), ("No, keep both.", "keep"), ("No, keep it.", "keep"),
    ("Yes, sounds good.", "assent"), ("Yes. Stop it.", "assent"), ("No, don't stop it.", "keep"),
    ("yes", "assent"), ("cancel it", "assent"), ("no", "keep"), ("both", "keep"), ("نه", "keep"),
]


@pytest.mark.parametrize(("text", "kind"), PURE_ANSWERS)
def test_r61_a_pure_answer_has_no_request(text, kind):
    """Old: 'thank you so much' / 'you can stop it' / «دیگه لازم نیست» came
    back as a MIXED remainder, the thanks tails after an agreeing clause (and
    the Persian keep verb «نگه دار») as AMBIGUOUS."""

    assert live.parse_confirmation_answer(text) == (kind, ""), text


@pytest.mark.parametrize(("text", "kind", "words"), [
    ("yes, what did you find?", "assent", "what did you find?"),      # the status tail
    ("yes, I changed my mind", "assent", "I changed my mind"),        # decision: MIXED
    ("yes, thanks for the list", "assent", "thanks for the list"),    # a content object
    ("آره، خیلی ممنون میشم اگه ایمیلش کنی", "assent", "خیلی ممنون میشم اگه ایمیلش کنی"),
    ("yes, and search for waterloo robotics professors", "assent", "search for waterloo robotics professors"),
    ("آره و یه هتل ارزون هم پیدا کن", "assent", "یه هتل ارزون هم پیدا کن"),
    ("Yes, stop it, and find the dean of engineering", "assent", "find the dean of engineering"),
    ("yes please find the dean", "assent", "find the dean"),
    ("stop it please find the dean", "assent", "find the dean"),
    ("yes stop the hotel one", "assent", "stop the hotel one"),      # a cancel of OTHER work
])
def test_r61_a_request_after_the_answer_is_the_request_in_the_callers_words(text, kind, words):
    assert live.parse_confirmation_answer(text) == (kind, words), text


@pytest.mark.parametrize("text", [
    "No, stop it.", "yes. no, keep it", "yes but wait", "Yes, but wait.", "No, that's fine.",
])
def test_r61_control_opposite_polarity_or_a_hedge_stays_ambiguous(text):
    assert live.parse_confirmation_answer(text)[0] == "ambiguous", text


@pytest.mark.parametrize("text", [
    "stop the hotel one", "cancel the search for hotels", "what's the weather?",
    "find me a pharmacy near union station",
])
def test_r61_control_a_request_of_its_own_is_no_answer(text):
    assert live.parse_confirmation_answer(text)[0] == "none", text


def test_r61_ambiguous_readings_say_whether_the_turn_holds_words_of_its_own():
    read = getattr(live, "read_confirmation_answer", None)
    assert read is not None, "no single confirmation classifier"
    assert read("yes but wait").residue is False
    assert read("yes but find the dean first").residue is True


# ══════════════════════════════════════════════════════════════════════
# R6-1 — wire: a pure answer is consumed, answered and never run
# ══════════════════════════════════════════════════════════════════════

def _then_reask_the_answer(text):
    async def steps(ctx, p, client):
        await _answer(text)(ctx, p, client)
        await _wait(lambda: any(any(m in c for m in STOPPED_MARKS) for c, *_ in ctx["lines"]), 3.0)
        await _wait(lambda: len(client.of("audio_delta")) >= 2, 2.0)
        await asyncio.sleep(0.2)
        _send(client, R4._idle(client.of("audio_delta")[-1]["response_id"]))   # the outcome line heard
        await asyncio.sleep(0.4)
        answer = [f for f in _finals(client) if f["text"] == text][0]
        ctx["answer_turn"] = answer["turn_id"]
        _send(client, {"type": "inject_text", "text": text, "reason": "no_response",
                       "reask_of_user_turn_id": answer["turn_id"]})
        await asyncio.sleep(0.6)
    return steps


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "yes, thank you so much", "okay, thank you so much", "آره، خیلی ممنون",
    "Yes, you can stop it.", "آره، دیگه لازم نیست.",
])
async def test_r61_a_pure_answer_is_answered_by_the_outcome_line(monkeypatch, text):
    """Old (fx4/snapF): A cancelled, but the outcome line was parented to B's
    turn and the by-id reask of the answer turn came back 'accepted' with the
    plain "No reply … reached them" instruction."""

    ctx = await _confirm(monkeypatch, _then_reask_the_answer(text), db=_db("r61-pure", text), **_fa(text))
    client, provider = ctx["client"], ctx["provider"]
    assert _cancelled_a(ctx), _phases(client, "d1")
    assert len(_questions_said(ctx)) == 1, _questions_said(ctx)
    outcome = [
        f for f in client.of("response_text")
        if any(m in (f.get("text") or "") for m in STOPPED_MARKS)
    ]
    assert outcome and outcome[0].get("parent_user_turn_id") == ctx["answer_turn"], outcome[:1]
    results = [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")]
    assert results == [(ctx["answer_turn"], "answered")], results
    assert R4._reask_appends(provider) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "yes, thank you so much", "آره، خیلی ممنون", "Yes, you can stop it.", "آره، دیگه لازم نیست.",
])
async def test_r61_a_delegated_pure_answer_never_runs_as_a_task(monkeypatch, text):
    """Old: a card titled 'thank you so much' / «دیگه لازم نیست» ran as its
    own agent task (or 'you can stop it' hit the cancel path)."""

    ctx = await _confirm(monkeypatch, _answer(text, delegate=True), db=_db("r61-del", text), **_fa(text))
    client, provider = ctx["client"], ctx["provider"]
    assert _cancelled_a(ctx), _phases(client, "d1")
    assert _frames_for(client, "d-answer") == [], _frames_for(client, "d-answer")
    assert [did for did, _d in ctx["calls"]] == ["d1", "d2"], ctx["calls"]
    notes = _notes(provider, "d-answer")
    assert notes and "handled" in notes[0], notes
    assert not _errors(client), _errors(client)


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "stops"), [
    ("Yes, stop it, thank you so much.", True),
    ("Sure, go ahead, thanks a lot.", True),
    ("آره، قطعش کن، خیلی ممنون.", True),
    ("No, keep both, thanks a lot.", False),
    ("نه، هر دو رو نگه دار.", False),
])
async def test_r61_an_agreeing_answer_is_applied_on_the_first_reply(monkeypatch, text, stops):
    """Old: every one was AMBIGUOUS — the relay asked again ('Sorry — should
    I stop …? Yes or no?')."""

    ctx = await _confirm(monkeypatch, _answer(text), db=_db("r61-first", text), **_fa(text))
    assert len(_questions_said(ctx)) == 1, _questions_said(ctx)
    assert _cancelled_a(ctx) is stops, _phases(ctx["client"], "d1")
    if not stops:
        assert _phases(ctx["client"], "d1")[-1] == "completed"


@pytest.mark.asyncio
async def test_r61_a_status_question_after_the_answer_is_never_swallowed(monkeypatch):
    """'yes, what did you find?': the answer applies AND the status question
    is the caller's own request.  Old: parsed as a pure 'yes' and its
    delegation absorbed with "Nothing else to do: do not delegate it again"."""

    text = "yes, what did you find?"
    ctx = await _confirm(monkeypatch, _answer(text, delegate=True), db=_db("r61-status"))
    client, provider = ctx["client"], ctx["provider"]
    assert _cancelled_a(ctx), _phases(client, "d1")
    notes = _notes(provider, "d-answer")
    assert not any("do not delegate it again" in n for n in notes), notes
    stop_lines = [c for c in _commentary(provider) if any(m in c for m in STOPPED_MARKS)]
    assert stop_lines, _commentary(provider)
    # The outcome line is parented to B's turn (C4: a MIXED answer) — the
    # status question stays recoverable.
    answer = [f for f in _finals(client) if f["text"] == text][0]
    outcome = [
        f for f in client.of("response_text")
        if any(m in (f.get("text") or "") for m in STOPPED_MARKS)
    ]
    assert outcome and outcome[0].get("parent_user_turn_id") != answer["turn_id"]


@pytest.mark.asyncio
async def test_r61_control_a_mixed_answer_stops_a_and_runs_the_rest(monkeypatch):
    text = "yes, and search for waterloo robotics professors"
    ctx = await _confirm(monkeypatch, _answer(text, delegate=True), db=_db("r61-mixed"))
    client = ctx["client"]
    assert _cancelled_a(ctx)
    created = [f for f in _frames_for(client, "d-answer") if f.get("phase") == "created"]
    assert created and created[0]["title"] == "search for waterloo robotics professors", created


@pytest.mark.asyncio
@pytest.mark.parametrize(("forced", "absorbed"), [
    ("thank you so much", True),          # asks nothing
    ("you can stop it", True),            # a restated stop of A
    ("stop it", True),                    # a bare cancel: its object is A
    ("search for waterloo robotics professors", False),   # CONTROL: a real request runs
])
async def test_r61_begin_defense_in_depth_one_answer_one_action(monkeypatch, forced, absorbed):
    """Even if the classifier ever handed back a remainder that is the answer
    itself, begin() absorbs it with the C5 'handled' result — never a second
    card, never a second action."""

    text = f"yes, {forced}"
    real = getattr(live, "read_confirmation_answer", None)
    assert real is not None, "no single confirmation classifier"

    def reading(value, title_words=frozenset()):
        if value == text:
            return live.ConfirmationReading("assent", forced, True)
        return real(value, title_words)

    monkeypatch.setattr(live, "read_confirmation_answer", reading)
    ctx = await _confirm(monkeypatch, _answer(text, delegate=True), db=_db("r61-dd", forced))
    client, provider = ctx["client"], ctx["provider"]
    assert _cancelled_a(ctx), _phases(client, "d1")
    if absorbed:
        assert _frames_for(client, "d-answer") == [], _frames_for(client, "d-answer")
        assert [did for did, _d in ctx["calls"]] == ["d1", "d2"], ctx["calls"]
        notes = _notes(provider, "d-answer")
        assert notes and "handled" in notes[0], notes
        assert not _errors(client), _errors(client)
    else:
        assert ("d-answer", forced) in ctx["calls"], ctx["calls"]


# ══════════════════════════════════════════════════════════════════════
# R6-2 — the §5.3 question reader (pure)
# ══════════════════════════════════════════════════════════════════════

def _line(title_request, *, lang="en", again=False):
    return live.relay_line(
        "confirm_stop_again" if again else "confirm_stop_ask", lang,
        title=live.request_title(title_request, 60),
    )


def _bare(text):
    return text.replace("«", "").replace("»", "")


L = _line(PROFESSORS)
LD = _line(DOC)
LQ = _line("can you check whether the robarts library is open tomorrow?")
LF = _line(A_FA, lang="fa")
LA = _line(PROFESSORS, again=True)
LAF = _line(A_FA, lang="fa", again=True)
LS = _line("find me a pharmacy near union station")
LAS = _line("find me a pharmacy near union station", again=True)

READER_MATRIX = [
    # the relay asked ONE question; the model asked none → never void
    ("Will do. " + L, L, False), (L + " Do let me know.", L, False), (L + " Can do either.", L, False),
    (L + " Will keep the hotel search going.", L, False), (_bare(L), L, False), (_bare(LD), LD, False),
    ("Should I stop looking up Dr. Smith's office hours?", LD, False),
    ("Should I stop the St. Michael's search? It's past 5 p.m. there.", L, False),
    ("Should I stop the U.S. search for the 2.5 star hotels?", L, False),
    (_bare(LAS), LAS, False), (_bare(LA), LA, False), (_bare(LAF), LAF, False),
    ("Sorry, I didn't catch that. Should I stop the pharmacy search? Yes or no?", LAS, False),
    (_bare(LQ), LQ, False), ("Sure. " + L, L, False), (L, L, False), (LF, LF, False), (LA, LA, False),
    # REAL questions of the model's own → void (kept)
    (L + " Also, do you want me to include their email addresses?", L, True),
    (L + " And would you like hotels near the university?", L, True),
    (L + " Or do you want me to keep both running?", L, True),
    ("Do you want their emails? " + L, L, True),
    (LF + " ایمیلشون رو هم می‌خوای؟", LF, True),
    (_bare(L) + " Also, do you want me to include their email addresses?", L, True),
    (_bare(LA) + " Or do you want me to keep both running?", LA, True),
    ("Should I stop the pharmacy search? Or should I keep both?", LS, True),
    (_bare(LF) + " ایمیلشون رو هم می‌خوای؟", LF, True),
    ("Should I stop looking up Dr. Smith? Do you want his email too?", LD, True),
]


@pytest.mark.parametrize(("said", "line", "void"), READER_MATRIX)
def test_r62_the_reader_counts_question_sentences(said, line, void):
    """Old: '…' / 'Dr.' split into sentences, 'Will do.' / 'Do let me know.'
    read as questions, and the re-ask line's own two '?' voided it."""

    assert live._LiveSession.question_beside_relay_line(said, line) is void, said


@pytest.mark.parametrize(("text", "asks"), [
    (L + " Either way is fine.", True), (L + " Just say yes or no.", True),
    ("I'll stop the professors search.", False), ("Should be all set for Friday.", False),
    ("Will send it over.", False), ("should i stop the professors search", True),
    ("do you want me to keep both", True), ("will do", False), ("do let me know", False),
    ("«can you check it?» is booked.", False), ("آیا جستجو رو متوقف کنم", True),
])
def test_r62_an_epoch_asks_by_its_question_marks_or_inversion(text, asks):
    fn = getattr(live, "epoch_asks", None)
    assert fn is not None and fn(text) is asks, text


def test_r62_question_sentences_ignore_dots_and_quoted_titles():
    assert getattr(live, "question_sentences", None) is not None
    assert live.question_sentences("Should I stop «find … ok?»?") == 1
    assert live.question_sentences("Dr. Smith… p.m. 2.5 U.S.") == 0
    assert live.question_sentences("Yes?? Really?") == 2
    assert live.question_sentences(LA) == 2 and live.question_sentences(L) == 1


# ══════════════════════════════════════════════════════════════════════
# R6-2 — wire: one question binds the yes; a real own question voids
# ══════════════════════════════════════════════════════════════════════

def _q(fn):
    return lambda content: fn(content) if R4._is_question(content) else content


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "render", "first", "cancels"), [
    ("will_do_lead_in", _q(lambda c: "Will do. " + c), PROFESSORS, True),
    ("do_let_me_know", _q(lambda c: c + " Do let me know."), PROFESSORS, True),
    ("bare_ellipsis_title", _q(_bare), PROFESSORS, True),
    ("paraphrase_dr", _q(lambda c: "Should I stop looking up Dr. Smith's office hours?"), DOC, True),
    ("tail_either_way", _q(lambda c: c + " Either way is fine."), PROFESSORS, True),
    ("control_real_q_verbatim", _q(lambda c: c + " Also, do you want me to include their email addresses?"),
     PROFESSORS, False),
    ("control_real_q_bare", _q(lambda c: _bare(c) + " Or do you want me to keep both running?"),
     PROFESSORS, False),
])
async def test_r62_a_yes_to_the_only_question_stops_a(monkeypatch, name, render, first, cancels):
    ctx = await _confirm(monkeypatch, _answer("yes"), first=first, render=render, db=_db("r62", name))
    assert ctx["questions"], "the relay never asked"
    assert _cancelled_a(ctx) is cancels, (name, _phases(ctx["client"], "d1"), ctx["spoken"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "result", "cancels"), [
    ("b_result_should_be_set", "Should be all set for Friday.", True),
    ("b_result_will_send", "Will send the booking details shortly.", True),
    ("control_b_result_real_q", "Booked it. Do you want me to email the receipt?", False),
])
async def test_r62_a_statement_after_the_question_is_no_newer_question(monkeypatch, name, result, cancels):
    """Fast B: its result is said between the question and the answer.  Old:
    'Should be all set …' opened with an auxiliary and read as a question."""

    def render(content):
        return result if "Booked the Hotel Ocho" in content else content

    async def steps(ctx, p, client):
        # The answer comes after B's result was said (the verifier's order).
        await _wait(lambda: any(result in s for s in ctx["spoken"]), 4.0)
        await _wait(lambda: len(client.of("audio_delta")) >= 2, 2.0)
        await asyncio.sleep(0.2)
        await _answer("yes")(ctx, p, client)

    ctx = await _confirm(monkeypatch, steps, d2_s=1.0, render=render, db=_db("r62b", name))
    assert ctx["questions"]
    assert any(result in s for s in ctx["spoken"]), ctx["spoken"]
    assert _cancelled_a(ctx) is cancels, (name, _phases(ctx["client"], "d1"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "first", "partial", "second", "cancels"), [
    ("reask_bare_en", PROFESSORS, PARTIAL_EN, "yes", True),
    ("reask_bare_fa", A_FA, PARTIAL_FA, "آره", True),
])
async def test_r62_a_clear_yes_after_a_bare_reask_stops_a(monkeypatch, name, first, partial, second, cancels):
    """The relay's re-ask holds TWO question marks by design ('…? Yes or
    no?'); spoken without «», the old reader counted them as a newer
    question and dropped the clear second answer."""

    async def steps(ctx, p, client):
        await _answer("yes but wait" if second == "yes" else "آره ولی صبر کن")(ctx, p, client)
        await _wait(lambda: len(ctx["questions"]) >= 2, 3.0)
        await _wait(lambda: len(client.of("audio_delta")) >= 2, 2.0)
        _send(client, R4._idle(client.of("audio_delta")[-1]["response_id"]))
        await asyncio.sleep(0.2)
        _q_start, q_end = ctx["questions"][-1]
        p.push(H.user_delta(second, q_end + 1500, q_end + 1900))
        await asyncio.sleep(0.9)

    ctx = await _confirm(monkeypatch, steps, first=first, partial=partial, render=_q(_bare),
                         db=_db("r62r", name))
    assert len(ctx["questions"]) == 2, ctx["spoken"]
    assert _cancelled_a(ctx) is cancels, _phases(ctx["client"], "d1")


# ══════════════════════════════════════════════════════════════════════
# R6-3 — the teardown fragment classifier (pure)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "یه هتل ارزون پیدا کن تو تورنتو.", "یه هتل ارزون پیدا کن تو تورنتو", "بلیط قطار برای کیوتو",
    "پروازهای ارزون به تورنتو", "یه پالتو", "یه رستوران خوب تو کیوتو", "استادای ایرانی دانشگاه تورنتو.",
    "هوای تورنتو؟", "do you know where the nearest pharmacy is?", "who is the reservation for?",
    "what's the meeting about?", "what should I do?",
])
def test_r63_a_complete_turn_is_no_fragment(text):
    """Old: every one read as a fragment (the letter-suffix clitic test, and
    end words that ignored the '?')."""

    assert not live.teardown_fragment(text), text


@pytest.mark.parametrize("text", [
    "can you", "are you still", "الو هنوز صدامو", "thank you for", "did you", "صدامو", "آهنگ رو",
    "find me a", "where is the", "can you hear", "book it for", "یه دندونپزشک با",
    "can you?", "are you still?", "الو هنوز صدامو؟", "صدامو؟", "did you?", "thank you for.",
    "ایمیلشو", "شمارشو", "خب شمارشو", "آدرس دفترشو", "آهنگمو", "کارشو", "صدای بچه‌مو",
])
def test_r63_control_a_truncated_turn_stays_a_fragment(text):
    assert live.teardown_fragment(text), text


@pytest.mark.parametrize("text", ["هوای تورنتو", "دانشگاه تورنتو", "قیمت پالتو", "استادای ایرانی دانشگاه تورنتو"])
def test_r63_a_bare_np_is_complete_only_when_the_call_vouches_for_it(text):
    """Integrator decision (c): corroborated by this call's own history →
    complete; otherwise A4's "cannot place" (truncated, visible)."""

    def vouched(value, tokens):
        try:
            return live.teardown_fragment(value, tokens)
        except TypeError:          # a relay with no notion of the call's history
            return None

    token = live._leading_words(text)[-1]
    assert live.teardown_fragment(text)
    assert vouched(text, {token}) is False
    assert vouched("آدرس دفترشو", {token}) is True     # another token vouches for nothing


# ══════════════════════════════════════════════════════════════════════
# R6-3 — wire: a turn open at the drop (two sockets, two provider sessions)
# ══════════════════════════════════════════════════════════════════════

TD_REQUEST = "what time does the union station library close"


def _td_provider(psid, start=None):
    def on_send(p, e):
        if e["type"] == "session.start" and start is not None:
            start(p)
        elif e["type"] == "session.close":
            p.push(H.closed())
    return H.FakeProvider(on_send=on_send, auto_ack=True, session_id=psid)


async def _drop_mid(monkeypatch, older, open_text, asks, tag, *, agent_said="", caller_said=""):
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=3000, voice_live_utterance_hard_gap_ms=6000)
    H.patch_relay(monkeypatch)
    db = _db("td", tag)

    def start(p):
        at = 0
        if caller_said:
            p.push(H.user_delta(caller_said, 0, 900))
            at = 5000
        if agent_said:
            p.push(H.out_text(agent_said, at + 1000, at + 1800))
            at += 5000
        if older:
            p.push(H.user_delta(older, at, at + 900))
            at += 5000
        p.push(H.user_delta(open_text, at, at + 400))

    c1 = Phone([H.config(features=V03), 0.8, R4._drop])
    await H.run_relay(c1, _td_provider(_psid("td1", tag), start), timeout=6, db_session_id=db)
    ids = {}
    for f in c1.of("transcript"):
        t = (f.get("text") or "").strip()
        if older and t == older:
            ids["R"] = f["turn_id"]
        elif t == open_text:
            ids["Q"] = f["turn_id"]
    script: list = [H.config(features=V03), 0.5]
    for label, words in asks:
        script += [{"type": "inject_text", "text": words, "reason": "no_response",
                    "reask_of_user_turn_id": ids[label]}, 0.4]
    script += [{"type": "stop"}]
    c2 = Phone(script)
    p2 = _td_provider(_psid("td2", tag))
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    labels = {v: k for k, v in ids.items()}
    verdicts = [(labels.get(f["user_turn_id"], "?"), f["outcome"]) for f in c2.of("reask_result")]
    contents = [str(a.get("content") or "") for a in R4._reask_appends(p2)]
    return verdicts, contents


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "q", "history"), [
    ("en_pharmacy_is_q", "do you know where the nearest pharmacy is?", {}),
    ("en_reservation_for_q", "who is the reservation for?", {}),
    ("fa_profs_toronto_dot", "استادای ایرانی دانشگاه تورنتو.", {}),
    ("fa_train_kyoto", "بلیط قطار برای کیوتو", {}),
    # (the agent's words share ONLY the token — never the caller's whole
    # phrase, which the relay would rightly judge the agent's own echo)
    ("fa_weather_after_agent", "هوای تورنتو", {"agent_said": "کتابخونه‌ی مرکزی تورنتو ساعت نه باز میشه."}),
    ("fa_weather_after_caller", "هوای تورنتو", {"caller_said": "تورنتو الان ساعت چنده؟"}),
])
async def test_r63_a_complete_turn_at_the_drop_is_the_owed_request(monkeypatch, name, q, history):
    """Old: Q 'unknown' and R carried with 'ask them to finish' — the complete
    request was demoted to a fragment."""

    verdicts, contents = await _drop_mid(
        monkeypatch, TD_REQUEST, q, [("Q", q), ("R", TD_REQUEST)], "complete-" + name, **history,
    )
    assert verdicts == [("Q", "accepted"), ("R", "unknown")], (verdicts, contents)
    assert len(contents) == 1 and f"«{q}»" in contents[0], contents
    assert "تماس قطع شد" not in contents[0] and "call dropped" not in contents[0], contents


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "q"), [
    ("sedamo", "صدامو"), ("alo_sedamo", "الو هنوز صدامو"),
    ("ahang_ro", "آهنگ رو"), ("emailesho", "ایمیلشو"), ("address_office", "آدرس دفترشو"),
    ("weather_unvouched", "هوای تورنتو"),                    # decision (c): cannot place
])
async def test_r63_control_a_truncated_turn_keeps_r_owed(monkeypatch, name, q):
    verdicts, contents = await _drop_mid(
        monkeypatch, TD_REQUEST, q, [("Q", q), ("R", TD_REQUEST)], "trunc-" + name,
    )
    assert verdicts == [("Q", "unknown"), ("R", "accepted")], (verdicts, contents)
    assert len(contents) == 1 and ("call dropped" in contents[0] or "تماس قطع شد" in contents[0]), contents


# ══════════════════════════════════════════════════════════════════════
# R6-8 — the C11 follow-up classification (pure)
# ══════════════════════════════════════════════════════════════════════

PROF_WORDS = live._content_words(PROFESSORS)
FA_WORDS = live._content_words(A_FA)
HOTEL_WORDS = live._content_words(HOTEL)


@pytest.mark.parametrize(("text", "held"), [
    ("no, book me a hotel in toronto", PROF_WORDS),              # the resolved conflict
    ("no whats the weather in toronto right now", PROF_WORDS),
    ("نه هوای تورنتو الان چطوره", FA_WORDS),
    ("نه، هوای تورنتو چطوریه؟", FA_WORDS),                        # copula enclitic
    ("no, it's okay, find me a pharmacy near union station", PROF_WORDS),
    ("no it's fine, book me a taxi home", PROF_WORDS),
    ("نه، اون خوبه، یه داروخونه نزدیک یونیون استیشن پیدا کن", FA_WORDS),
    ("no, what's the weather in toronto right now?", PROF_WORDS),
    ("no, it's fine", PROF_WORDS), ("no, great", PROF_WORDS),
])
def test_r68_a_self_contained_follow_up_is_none(text, held):
    """Old: the pleasantry's 'it' read as anaphora, «چطوریه» as a bare NP,
    and a '?'-less weather question as a partial statement."""

    assert live.negation_evidence(text, held) == "none", text
    assert live.contextual_reference(text) == "", text


@pytest.mark.parametrize(("text", "held", "kind"), [
    # PIN CHANGE, R2 addendum 6 R6-10 ('I mean NP' parity, which names it):
    # "no, I mean the downtown campus" joined "i meant" / «منظورم» in the
    # explicit-modification table — FULL, pinned in the grade table below and
    # in tests/test_live_round6b.py.  The other repair phrasings stay PARTIAL.
    ("no, I'm asking about the downtown campus", PROF_WORDS, "repair"),
    ("no, the downtown campus I mean", PROF_WORDS, "repair"),
    ("no, i said the downtown campus", PROF_WORDS, "repair"),
    ("نه، اونایی که تو پردیس داون‌تاون هستن", FA_WORDS, "repair"),
    ("no, what about the downtown campus?", PROF_WORDS, "ellipsis"),
    ("نه، پردیس داون‌تاون چطور؟", FA_WORDS, "ellipsis"),
    ("no, these aren't downtown", PROF_WORDS, "anaphora"),
    ("نه، اینا یوآفتی داون‌تاون نیستند", FA_WORDS, "anaphora"),
    ("no, it's okay, these aren't downtown", PROF_WORDS, "anaphora"),   # after a pleasantry
])
def test_r68_a_context_dependent_follow_up_is_partial(text, held, kind):
    """Old: the repair phrasings graded NONE — both ran as strangers."""

    assert live.contextual_reference(text) == kind, text
    assert live.negation_evidence(text, held) == "partial", text


@pytest.mark.parametrize(("text", "held", "grade"), [
    (PARTIAL_EN, PROF_WORDS, "partial"),           # ≥ half of its own words shared
    (PARTIAL_FA, FA_WORDS, "partial"),
    ("no, I meant the downtown campus", PROF_WORDS, "full"),     # FULL cases unchanged
    # R2 addendum 6 R6-10 ('I mean NP' parity): present-tense "i mean" + NP
    # is the same explicit modification as "i meant" (was PARTIAL in round 6).
    ("no, I mean the downtown campus", PROF_WORDS, "full"),
    ("no, only the downtown campus", PROF_WORDS, "full"),
    ("no, thursday", live._content_words(R4.BOOKING["transcript"]), "full"),
    ("no, what about associate professors?", PROF_WORDS, "partial"),
    ("no, it's okay, find me a pharmacy near union station", HOTEL_WORDS, "partial"),
])
def test_r68_control_the_proportion_and_full_grades(text, held, grade):
    assert live.negation_evidence(text, held) == grade, text


def test_r68_control_dependent_and_restriction_shapes():
    assert live.sequencing_marker("then email me the list")
    assert live.contextual_reference("then email me the list") == ""
    assert live.explicit_modification("actually only the robotics faculty") == "correction"


@pytest.mark.parametrize(("text", "want"), [
    ("what's the weather in toronto right now?", ("none", -1, "")),
    ("find me a pharmacy near union station", ("none", -1, "")),
    ("یه داروخونه نزدیک یونیون استیشن پیدا کن", ("none", -1, "")),
    ("neither, I'm asking about the weather", ("none", -1, "")),
    ("the first one", ("selected", 0, "")), ("the professors one", ("selected", 0, "")),
    ("the hotel search", ("selected", 1, "")), ("okay, the second one", ("selected", 1, "")),
    ("I mean the professors one", ("selected", 0, "")),
    ("the professors one, and find me a pharmacy near union station",
     ("selected", 0, "find me a pharmacy near union station")),
    ("both", ("ambiguous", -1, "")), ("either", ("ambiguous", -1, "")), ("yes", ("ambiguous", -1, "")),
    ("hmm, which?", ("ambiguous", -1, "")),
])
def test_r68_only_a_selection_clause_answers_the_which_question(text, want):
    """Old: a request sharing one word with a job ('toronto', 'union
    station') graded 'ambiguous' and was consumed as an unclear answer."""

    words = [live._content_words(PROFESSORS), live._content_words(HOTEL)]
    assert live.parse_target_answer(text, words) == want, text


# ══════════════════════════════════════════════════════════════════════
# R6-8 — wire: one running job
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "followup"), [
    (PROFESSORS, "no, it's okay, find me a pharmacy near union station"),
    (PROFESSORS, "no it's fine, book me a taxi home"),
    (A_FA, "نه، اون خوبه، یه داروخونه نزدیک یونیون استیشن پیدا کن"),
    (A_FA, "نه، هوای تورنتو چطوریه؟"),
    (PROFESSORS, "no whats the weather in toronto right now"),
    (A_FA, "نه هوای تورنتو الان چطوره"),
    (PROFESSORS, "no, book me a hotel in toronto"),
])
async def test_r68_a_self_contained_request_runs_normally(monkeypatch, first, followup):
    """Old: asked 'Should I stop «…»?' and bound the request to the running
    job as its referent."""

    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, first, followup, running=True, db=_db("r68-self", first, followup),
    )
    head = calls[-1].split("Current caller request:")[0]
    assert killed == [] and "superseded" not in _phases(client, "d1")
    assert not _stops(provider) and not _which(provider), _commentary(provider)
    assert BINDING not in head
    assert _phases(client, "d2")[:2] == ["created", "started"], _phases(client, "d2")


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "followup"), [
    # PIN CHANGE, R2 addendum 6 R6-10 ('I mean NP' parity, which names it):
    # "no, I mean the downtown campus" is an explicit modification now — FULL
    # with exactly one determinable running target (supersede, no question),
    # pinned in tests/test_live_round6b.py.  The other repairs still ask.
    (PROFESSORS, "no, I'm asking about the downtown campus"),
    (PROFESSORS, "no, the downtown campus I mean"),
    (A_FA, "نه، اونایی که تو پردیس داون‌تاون هستن"),
])
async def test_r68_a_repair_of_running_work_asks_and_binds(monkeypatch, first, followup):
    """Old: both ran as strangers — d1 left started, no relation, no question
    (the supervisor's 21:54 shape)."""

    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, first, followup, running=True, db=_db("r68-repair", first, followup),
    )
    assert killed == [] and "superseded" not in _phases(client, "d1")
    questions = _stops(provider)
    assert len(questions) == 1 and live.request_title(first, 60) in questions[0], _commentary(provider)
    assert BINDING in calls[-1].split("Current caller request:")[0]


# ══════════════════════════════════════════════════════════════════════
# R6-8 — wire: several jobs (running / finished); target selection and the
# which-question (harness: two jobs, then the follow-up d3)
# ══════════════════════════════════════════════════════════════════════

async def _jobs_call(monkeypatch, steps, *, first=PROFESSORS, second=HOTEL, followup=CAMPUS,
                     schedule="both_running", features=V03, render=None, db=""):
    """schedule: both_running | sequential_finished | first_finished_second_running"""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    ctx: dict = {"calls": [], "killed": [], "questions": [], "lines": []}
    release = asyncio.Event()
    run1 = schedule == "both_running"
    run2 = schedule in {"both_running", "first_finished_second_running"}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        ctx["calls"].append((did, task))
        running = (did == "d1" and run1) or (did == "d2" and run2)
        if did in {"d1", "d2"} and running:
            try:
                await asyncio.wait_for(release.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                ctx["killed"].append(did)
                raise
        return {"d1": "Professor X and Professor Y.", "d2": "The Royal York."}.get(did, "Done."), "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}
    clock = {"at": 20000}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(first, 0, 900))
            p.push(H.delegation("d1", 950))
        elif e["type"] == "session.commentary.append":
            content = str(e.get("content") or "")
            start = clock["at"]
            if _is_which(content) or R4._is_question(content):
                ctx["questions"].append((content, start, start + 1500))
            ctx["lines"].append((content, start, start + 1500))
            spoken = render(content) if render is not None else content
            p.push(H.out_text(spoken, start, start + 1500))
            p.push(H.out_audio("Q"))
            clock["at"] += 6000
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        p = box["p"]
        await _wait(lambda: len(ctx["calls"]) >= 1)
        if schedule in {"sequential_finished", "first_finished_second_running"}:
            await _wait(lambda: R4._terminal(client, "d1"), 4.0)
            await asyncio.sleep(1.2)
        p.push(H.user_delta(second, 2000, 2800))
        p.push(H.delegation("d2", 2850))
        await _wait(lambda: len(ctx["calls"]) >= 2)
        if schedule == "sequential_finished":
            await _wait(lambda: R4._terminal(client, "d2"), 4.0)
            await asyncio.sleep(1.2)
        n_before = len(ctx["questions"])
        p.push(H.user_delta(followup, 5000, 5800))
        p.push(H.delegation("d3", 5850))
        await _wait(lambda: len(ctx["questions"]) > n_before or _frames_for(client, "d3"), 3.0)
        await asyncio.sleep(0.5)
        ctx["before_d3"] = list(_frames_for(client, "d3"))
        await steps(ctx, p, client)
        await asyncio.sleep(0.6)
        release.set()
        await _wait(lambda: all(R4._terminal(client, d) for d in ("d1", "d2")), 5.0)
        await _wait(lambda: R4._terminal(client, "d3") or not _frames_for(client, "d3"), 3.0)
        await asyncio.sleep(0.4)

    client = Phone([H.config(features=features), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, session_id=_psid("jobs", db))
    ctx["client"], ctx["provider"] = client, provider
    await H.run_relay(client, provider, timeout=24, db_session_id=db)
    return ctx


def _reply(text, *, at_offset=2500, delegate=None):
    async def steps(ctx, p, client):
        if not ctx["questions"]:
            await asyncio.sleep(0.3)
            return
        _send(client, R4._idle(R4._epoch_of(client)))
        await asyncio.sleep(0.2)
        _content, _q_start, q_end = ctx["questions"][-1]
        s = q_end + at_offset
        p.push(H.user_delta(text, s, s + 500))
        if delegate:
            p.push(H.delegation(delegate, s + 550))
        await asyncio.sleep(1.0)
    return steps


async def _nothing(ctx, p, client):
    await asyncio.sleep(0.3)


def _input_for(ctx, did) -> str:
    return next((t for d, t in ctx["calls"] if d == did), "")


@pytest.mark.asyncio
@pytest.mark.parametrize(("followup", "fa", "asks_nothing"), [
    # the addendum's NONE controls: no question of any kind, no binding
    ("no, book me a hotel in toronto", False, True),
    ("no whats the weather in toronto right now", False, True),
    ("نه هوای تورنتو الان چطوره", True, True),
    ("نه، هوای تورنتو چطوریه؟", True, True),
    ("no it's fine, book me a taxi home", False, True),
    # shares 'near union station' — 3 of its 4 own words — with the HOTEL
    # search: PARTIAL by the proportion rule, so the relay may ask once
    # whether to stop THAT search; never held, never bound, never cancelled.
    ("no, it's okay, find me a pharmacy near union station", False, False),
])
async def test_r68_a_self_contained_request_with_two_jobs_running_is_dispatched(
    monkeypatch, followup, fa, asks_nothing,
):
    """Old: held (no card) behind a which-question, bound to the job one
    shared word pointed at, or asked about stopping it."""

    kwargs = {"first": A_FA, "second": HOTEL_FA} if fa else {}
    ctx = await _jobs_call(monkeypatch, _nothing, followup=followup, db=_db("r68-two", followup), **kwargs)
    provider = ctx["provider"]
    assert ctx["before_d3"], "the request was held undispatched"
    assert not _which(provider), _which(provider)
    if asks_nothing:
        assert not _stops(provider), _stops(provider)
    assert BINDING not in _input_for(ctx, "d3").split("Current caller request:")[0]
    assert ctx["killed"] == []


@pytest.mark.asyncio
async def test_r68_control_an_elliptical_follow_up_with_two_jobs_asks_which(monkeypatch):
    ctx = await _jobs_call(monkeypatch, _nothing, db=_db("r68-which"))
    assert not ctx["before_d3"] and len(_which(ctx["provider"])) == 1
    assert ctx["killed"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [
    "no, what about the professors at the downtown campus?",
    "no, the professors at the downtown campus",
])
async def test_r68_words_select_the_older_finished_job(monkeypatch, followup):
    """Verifier w4.  Old: the 'moved on' filter dropped the professors job
    before its words were read, and the hotel job was linked by default."""

    ctx = await _jobs_call(monkeypatch, _nothing, followup=followup, schedule="sequential_finished",
                           db=_db("r68-w4", followup))
    rels = _relations(ctx["client"], "d3")
    assert {"kind": "replaces", "task_id": "d2"} not in rels, rels
    assert rels and rels[-1] == {"kind": "replaces", "task_id": "d1"}, rels
    assert "Request the caller is correcting" in _input_for(ctx, "d3")
    assert PROFESSORS in _input_for(ctx, "d3")


@pytest.mark.asyncio
async def test_r68_a_correction_naming_a_finished_job_never_binds_the_running_one(monkeypatch):
    """Verifier w5: professors finished, hotel running.  Old: 'Should I stop
    «find me a hotel …»?' and the correction bound to the hotel job."""

    followup = "no, what about the professors at the downtown campus?"
    ctx = await _jobs_call(monkeypatch, _nothing, followup=followup,
                           schedule="first_finished_second_running", db=_db("r68-w5"))
    provider = ctx["provider"]
    assert not any("hotel" in s for s in _stops(provider)), _stops(provider)
    assert not _which(provider)
    head = _input_for(ctx, "d3").split("Current caller request:")[0]
    assert not (BINDING in head and HOTEL in head), head
    assert _relations(ctx["client"], "d3")[-1] == {"kind": "replaces", "task_id": "d1"}
    assert ctx["killed"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("newreq", [
    "what's the weather in toronto right now?",
    "find me a pharmacy near union station",
    "یه داروخونه نزدیک یونیون استیشن پیدا کن",
])
async def test_r68_a_new_request_while_the_which_question_waits_runs_as_its_own(monkeypatch, newreq):
    """Verifier / supervisor x7.  Old: consumed as an 'unclear answer', the
    model's delegation absorbed with "Do not delegate it again", and the
    relay asked which once more."""

    ctx = await _jobs_call(monkeypatch, _reply(newreq, delegate="d-answer"), db=_db("r68-x7", newreq))
    client, provider = ctx["client"], ctx["provider"]
    assert _which(provider), "precondition: the which-question was asked"
    phases = _phases(client, "d-answer")
    assert phases[:1] == ["created"] and phases[-1] == "completed", phases
    assert len(_which(provider)) == 1, _which(provider)
    assert not any("unclear answer" in n for n in _notes(provider, "d-answer"))
    assert not _phases(client, "d3"), "nothing may be dispatched for the unanswered which-question"


@pytest.mark.asyncio
async def test_r68_control_a_selection_with_a_request_is_applied_and_the_rest_runs(monkeypatch):
    ans = "the professors one, and find me a pharmacy near union station"
    ctx = await _jobs_call(monkeypatch, _reply(ans, delegate="d-answer"), db=_db("r68-mixed"))
    client = ctx["client"]
    assert _phases(client, "d3"), "the selection was not applied"
    assert BINDING in _input_for(ctx, "d3") and PROFESSORS in _input_for(ctx, "d3")
    titles = [f.get("title") for f in _frames_for(client, "d-answer") if f.get("title")]
    assert titles and titles[0] == "find me a pharmacy near union station", titles


@pytest.mark.asyncio
async def test_r68_a_paraphrased_which_question_still_binds_the_selection(monkeypatch):
    """CONTROL (both trees) — R6-2 for the which-question: GPT-Live says it
    with a lead-in and without «»; the selecting answer still binds."""

    def render(content):
        return ("Sure. " + _bare(content)) if _is_which(content) else content

    ctx = await _jobs_call(monkeypatch, _reply("the first one"), render=render, db=_db("r68-para"))
    assert _which(ctx["provider"])
    assert _phases(ctx["client"], "d3"), "the selection did not bind"
    assert BINDING in _input_for(ctx, "d3") and PROFESSORS in _input_for(ctx, "d3")


# ══════════════════════════════════════════════════════════════════════
# R6-8 C12 — the continuation claim is tentative until the caller's input
# for its window has settled (the verifier's output_first, both ways)
# ══════════════════════════════════════════════════════════════════════

def _parents(phone) -> list[str]:
    seen: dict[str, str] = {}
    for frame in phone.of("response_text"):
        epoch = str(frame.get("assistant_turn_id") or frame.get("response_id") or "")
        seen.setdefault(epoch, str(frame.get("parent_user_turn_id") or ""))
    return list(seen.values())


async def _c12(monkeypatch, script_fn, tag):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    def on_send(p, e):
        if e.get("type") == "session.start":
            box["p"] = p
            p.push(H.user_delta(NR.REQUEST, 0, 900))
            p.push(H.out_text("There is one on Front Street.", 1000, 1400))
            p.push(H.out_audio(NR.pcm(0)))
        elif e.get("type") == "session.close":
            p.push(H.closed())

    provider = H.FakeProvider(on_send=on_send, session_id=_psid("c12", tag), auto_ack=True)
    ref: list = []

    async def script():
        await script_fn(ref[0], box)

    phone = NR.Phone([NR._config(NR.APP_FEATURES), script, 0.3, {"type": "stop"}], tape=[])
    ref.append(phone)
    await H.run_relay(phone, provider, timeout=10, db_session_id=_db("c12", tag))
    return phone, provider


@pytest.mark.asyncio
async def test_c12_output_before_the_callers_transcript_is_not_rs(monkeypatch):
    """The reply to 'hello?' reaches the relay 50 ms BEFORE the input
    transcript of 'hello?' (which came first on the provider timeline).  Old
    (fx6/snapF): parented to R — a heard 'Yes, I'm here.' would count as R's
    coverage."""

    async def script(phone, box):
        assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.4)            # the gap timer retires e1
        p = box["p"]
        p.push(H.out_text("Yes, I'm here.", 2050, 2400))
        p.push(H.out_audio(NR.pcm(1)))
        await asyncio.sleep(0.05)
        p.push(H.user_delta("hello?", 1450, 1750))
        await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        await asyncio.sleep(0.5)

    phone, _ = await _c12(monkeypatch, script, "output-first")
    finals = NR._finals(phone)
    assert _parents(phone) == [finals[0]["turn_id"], finals[1]["turn_id"]], _parents(phone)


@pytest.mark.asyncio
async def test_c12_control_a_real_continuation_keeps_r_when_nobody_spoke(monkeypatch):
    """The other way: the same output, and NO caller speech in the window —
    the tentative claim settles and the continuation keeps R."""

    async def script(phone, box):
        assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.4)
        p = box["p"]
        p.push(H.out_text("It opens at nine.", 2050, 2400))
        p.push(H.out_audio(NR.pcm(1)))
        await NR._wait(lambda: len(_parents(phone)) >= 2, 3.0)
        await asyncio.sleep(0.3)

    phone, _ = await _c12(monkeypatch, script, "no-speech")
    rid = NR._finals(phone)[0]["turn_id"]
    assert _parents(phone) == [rid, rid], _parents(phone)
    assert len(NR._finals(phone)) == 1


@pytest.mark.asyncio
async def test_c12_a_caller_turn_found_later_in_the_window_reparents_before_the_reask(monkeypatch):
    """The caller's transcript arrives even AFTER the tentative window was
    released: the relay re-evaluates the continuation before any verdict
    reads it.  R's own answer failed to play; the reply heard was to
    'hello?' — R is never 'answered' by it."""

    async def script(phone, box):
        assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.4)
        e1 = NR._epoch(phone)
        phone.push_frame(NR._failed(e1))       # R's answer never played
        await asyncio.sleep(0.2)
        p = box["p"]
        p.push(H.out_text("Yes, I'm here.", 1500, 1800))
        p.push(H.out_audio(NR.pcm(1)))
        assert await NR._wait(lambda: NR._epoch(phone) != e1)
        await asyncio.sleep(0.5)             # released: nobody (yet) in the window
        e2 = NR._epoch(phone)
        phone.push_frame(NR._idle(e2))        # the reply was heard
        await asyncio.sleep(0.2)
        p.push(H.user_delta("hello?", 1420, 1480))   # …the late transcript
        await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        await asyncio.sleep(0.3)
        phone.reask(NR._finals(phone)[0]["turn_id"])
        await NR._wait(lambda: phone.of("reask_result"))

    phone, _ = await _c12(monkeypatch, script, "late")
    rid = NR._finals(phone)[0]["turn_id"]
    result = phone.of("reask_result")[-1]
    assert result["user_turn_id"] == rid and result["outcome"] != "answered", result
