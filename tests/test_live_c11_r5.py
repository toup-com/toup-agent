"""R2 addendum 5 C11 (supervisor 21:54, target ambiguity 22:1x): a contextual
correction is related to the work it corrects — and never guessed.

Owner requirement 5: intent and the intended target determine the transition;
related refinements keep their causal connection; independent jobs stay
separate; saying something related never cancels running work; ambiguous
cancellation or replacement requires clarification.  Acceptance row: "existing
search + added campus/department constraint".

The supervisor's ORIGINAL probe (/private/tmp/toup-supervisor-r5-campus.vWJfEQ/
selected.xml — the critic's test_r4c_value_question_running.py, parameter
'no, what about the downtown campus?'): running "find iranian computer science
professors at the university of toronto" + "no, what about the downtown
campus?" → both delegations ran, d1 stayed started, NO relation and NO
clarification.  The C7 lexical half-overlap rule explained that and was not
accepted: absent shared words are no proof of independence.

The rule (structural, `contextual_reference`; no campus/department keyword):
under a corrective opener (no / not (that) / actually / instead / «نه» / «نه
بابا» / «ولی») a follow-up is TARGET-DEPENDENT when it is ELLIPTICAL (no
predicate of its own: "what/how about <NP>?", "and <NP>?", a bare NP, «<NP>
چطور؟», «<NP> چی؟», «پس <NP>؟» — not «… چطوره؟») or ANAPHORIC in a corrective
statement ("no, these aren't downtown", «نه، اینا یوآفتی داون‌تاون نیستند»).
Such a follow-up is PARTIAL:

  * ONE running job (or words that select one): it starts as its own task with
    that job's request as BINDING context and the relay asks once "Should I
    stop «A»?" (every §5.3 safety) — never an automatic cancel;
  * several running/queued jobs and nothing selecting one: NOTHING is
    dispatched; the relay asks "Which one do you mean — «A» or «C»?" (a bound
    record: its heard epoch, order, expiry, one re-ask); the answer continues
    exactly as the single-target path; no answer → nothing dispatched;
  * finished work: exactly one plausible finished job → the `replaces` link
    with its request + answer as correcting context; several → ask which;
  * a self-contained question/imperative with its own predicate stays
    independent: no question, no link.

Documented conflict — RESOLVED by R2 addendum 6 R6-8 (pins changed below):
addendum 5 C11's control list names 'no, book me a hotel in toronto' as
independent, while C11 (a) "shares non-value content (existing)".  The
integrator's decision: lexical sharing is evidence only in proportion to what
the follow-up itself asks — a SELF-CONTAINED follow-up is PARTIAL only when
the shared words are at least half of its own.  So that utterance is NONE (no
question, no binding), and the §5.3 suites' trigger R4.PARTIAL is now a
self-contained correction that shares half or more ('no, find a dorm at the
university of toronto').

Old-fail: a copy of THIS file run against fx4-relay-snapE (lvp 670abce5…)
fails every behaviour test by assertion; the controls pass on both.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os

import pytest

import test_live_harness as H
import test_live_round4 as R4
from app.config import settings
from app.services import live_voice_protocol as live


PROFESSORS = R4.PROFESSORS
A_FA = R4.A_FA
HOTEL = "find me a hotel near union station"
HOTEL_FA = "یه هتل نزدیک یونیون استیشن پیدا کن"
CAMPUS = "no, what about the downtown campus?"
DEPARTMENT = "no, what about the engineering department?"
CAMPUS_FA = "نه، پردیس داون‌تاون چطور؟"
V1_FA = "نه، اینا یوآفتی داون‌تاون نیستند"
THESE = "no, these aren't downtown"
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

WHICH_MARKS = ("Which one do you mean", "Sorry — which one", "منظورت کدومه", "ببخشید — کدومش")


def _db(tag: str, *parts) -> str:
    return f"db-c11-{tag}-{abs(hash(parts)) % 10**10}"


def _psid(tag: str, *parts) -> str:
    return f"live-psid-c11-{tag}-{abs(hash(parts)) % 10**8}"


def _record(name: str, what: str, client, relay: dict) -> None:
    """Cross-layer: written ONLY under C11_R5_FRAMES_OUT (never otherwise) —
    the relay's frames to the phone for the scenario, in the ROUND4_FRAMES_OUT
    shape, plus the relay-side facts, for the cross-layer owner's replay
    through the app's real turn watch / reducer."""

    path = os.environ.get("C11_R5_FRAMES_OUT")
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
    data["_what"] = (
        "R2 addendum 5 C11 relay frames: contextual corrections (ellipsis/anaphora) — one running "
        "job (bound refined request + stop question), several jobs (the which-question holds the "
        "follow-up; the answer continues it), finished jobs (replaces link). Written by "
        "backend/tests/test_live_c11_r5.py under C11_R5_FRAMES_OUT."
    )
    data["_relay"] = {
        "tree": os.path.abspath(os.getcwd()),
        "live_voice_protocol": os.path.abspath(live.__file__),
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    data.setdefault("scenarios", {})[name] = {
        "what": what,
        "frames": [f for f in client.frames if f.get("type") in keep],
        "relay": relay,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


def _is_which(content: str) -> bool:
    return any(mark in content for mark in WHICH_MARKS)


def _which(provider) -> list[str]:
    return [c for c in _commentary(provider) if _is_which(c)]


def _stops(provider) -> list[str]:
    return [c for c in _commentary(provider) if R4._is_question(c)]


def _relations(client, did) -> list:
    return [f.get("relation") for f in _lifecycle(_frames_for(client, did))]


def _input_for(calls, did) -> str:
    return next((text for d, text in calls if d == did), "")


# ══════════════════════════════════════════════════════════════════════
# 1. The grade (pure): ellipsis/anaphora are PARTIAL, own predicates stay NONE
# ══════════════════════════════════════════════════════════════════════

PROF_WORDS = live._content_words(PROFESSORS)
FA_WORDS = live._content_words(A_FA)


@pytest.mark.parametrize(("text", "held"), [
    (CAMPUS, PROF_WORDS),                                  # the supervisor's probe
    (DEPARTMENT, PROF_WORDS),
    ("no, how about the downtown campus?", PROF_WORDS),
    ("no, and the downtown campus?", PROF_WORDS),
    ("no, the downtown campus", PROF_WORDS),               # a bare NP
    ("actually, what about the downtown campus?", PROF_WORDS),
    ("no, not that, the engineering department", PROF_WORDS),
    (THESE, PROF_WORDS),                                   # anaphora
    ("no, those are all at the mississauga campus", PROF_WORDS),
    (CAMPUS_FA, FA_WORDS),
    ("نه، پس پردیس داون‌تاون؟", FA_WORDS),
    ("ولی پردیس داون‌تاون چطور؟", FA_WORDS),
    ("نه بابا، دانشکده مهندسی چی؟", FA_WORDS),
    (V1_FA, FA_WORDS),                                     # the V1 anaphoric statement
])
def test_a_contextual_correction_is_partial_not_none(text, held):
    """Old: every one graded 'none' (no shared non-value word)."""

    assert live.negation_evidence(text, held) == "partial", text


@pytest.mark.parametrize(("text", "held"), [
    ("no, what's the weather in toronto right now?", PROF_WORDS),
    ("no, how do I get to the university of toronto from union station?", PROF_WORDS),
    ("no, what time is it in tokyo right now?", PROF_WORDS),
    ("no, find a pharmacy near union station", PROF_WORDS),
    ("no, call mom", PROF_WORDS),                          # a request verb heads it
    ("no, play some jazz", PROF_WORDS),
    ("نه، هوای تورنتو چطوره؟", FA_WORDS),                   # «چطوره» has its own copula
    ("نه بابا، ساعت توکیو چنده؟", FA_WORDS),
    ("نه، یه داروخونه نزدیک یونیون استیشن پیدا کن", FA_WORDS),
])
def test_control_a_self_contained_question_or_imperative_stays_none(text, held):
    """CONTROL (both trees): its own predicate and subject — independent."""

    assert live.negation_evidence(text, held) == "none", text


@pytest.mark.parametrize(("text", "held", "grade"), [
    ("no, thursday?", live._content_words(R4.BOOKING["transcript"]), "full"),        # C7 value-only
    ("no, what about thursday?", live._content_words(R4.BOOKING["transcript"]), "full"),
    ("نه، فردا چطور؟", live._content_words("یه وقت دندونپزشکی برای امروز بگیر"), "full"),
    ("no, what about associate professors?", PROF_WORDS, "partial"),                 # C7 partial
    ("no, only the downtown campus", PROF_WORDS, "full"),                            # explicit
])
def test_control_existing_c7_grades_are_kept(text, held, grade):
    assert live.negation_evidence(text, held) == grade, text


def test_documented_conflict_the_hotel_imperative_keeps_the_addendum4_partial():
    """PIN CHANGED (R2 addendum 6 R6-8 — the integrator RESOLVED the conflict
    this file documented): 'no, book me a hotel in toronto' is SELF-CONTAINED
    and shares one of its three own words with the research — NONE (no
    question, never bound).  R4.PARTIAL is now a self-contained correction
    sharing at least half of its own words: still PARTIAL, still unbound."""

    assert live.negation_evidence(R4.PARTIAL_R5, PROF_WORDS) == "none"
    assert live.contextual_reference(R4.PARTIAL_R5) == ""
    assert live.negation_evidence(R4.PARTIAL, PROF_WORDS) == "partial"
    assert live.contextual_reference(R4.PARTIAL) == ""


@pytest.mark.parametrize(("text", "record"), [
    (CAMPUS, R4.PROF_RECORD),
    (THESE, R4.PROF_RECORD),
    (V1_FA, {"transcript": A_FA, "answer": "دکتر الف و دکتر ب، هر دو در پردیس سنت جورج."}),
    (CAMPUS_FA, {"transcript": A_FA, "answer": "دکتر الف."}),
])
def test_a_contextual_correction_of_the_finished_request_links(text, record):
    """Old: not linked — the V1 «نه، اینا یوآفتی داون‌تاون نیستند» ran as a
    stranger with nothing of the request it corrects."""

    assert live._LiveSession.modifies_record(None, text, record), text


@pytest.mark.parametrize(("text", "record"), [
    ("no, what's the weather in toronto right now?", R4.PROF_RECORD),
    ("no, how do i get to the university of toronto from union station?", R4.PROF_RECORD),
    ("نه، هوای تورنتو چطوره؟", {"transcript": A_FA, "answer": ""}),
])
def test_control_an_independent_question_never_links_a_finished_request(text, record):
    assert not live._LiveSession.modifies_record(None, text, record), text


# ══════════════════════════════════════════════════════════════════════
# 2. ONE running job: the refined request runs bound to it, and the relay
#    asks "Should I stop «A»?" — never an automatic cancel
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "followup"), [
    (PROFESSORS, CAMPUS),          # the ORIGINAL supervisor probe / critic case
    (PROFESSORS, DEPARTMENT),
    (PROFESSORS, THESE),
    (A_FA, CAMPUS_FA),
    (A_FA, V1_FA),                 # the V1 anaphoric statement, running
])
async def test_a_contextual_correction_of_running_work_asks_and_binds(monkeypatch, first, followup):
    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, first, followup, running=True, db=_db("run", first, followup),
    )
    _record(f"one_running[{followup}]", f"A={first!r} running; the caller: {followup!r}", client,
            {"questions": _stops(provider), "killed": killed})
    d1, d2 = _phases(client, "d1"), _phases(client, "d2")
    # never "both run, no relation, no clarification" — and never a cancel
    assert killed == [] and "superseded" not in d1 and "cancelled" not in d1, (d1, killed)
    questions = _stops(provider)
    assert len(questions) == 1, _commentary(provider)
    title = live.request_title(first, 60)
    assert title in questions[0], questions
    if first == A_FA:
        assert "رو متوقف کنم؟" in questions[0]
    # the refined request runs as its own task, BOUND to A's request
    assert d2[:2] == ["created", "started"], d2
    assert all(rel is None for rel in _relations(client, "d2")), "a relation before the answer"
    refined = calls[-1]
    assert BINDING in refined and first in refined.split("Current caller request:")[0], refined
    assert "(non-binding context" not in refined.split("Current caller request:")[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes", "yes stop it"])
async def test_a_yes_to_the_contextual_question_stops_a_and_b_replaces_it(monkeypatch, answer):
    ctx = await R4._confirm_call(monkeypatch, R4._answer(answer), partial=CAMPUS, db=_db("yes", answer))
    client = ctx["client"]
    _record(f"one_running_yes[{answer}]", f"{CAMPUS!r} while A runs; asked; the caller: {answer!r}",
            client, {"questions": _stops(ctx["provider"])})
    assert ctx["asked"]
    assert R4._cancelled_a(ctx), _phases(client, "d1")
    rel = _relations(client, "d2")
    assert rel[0] is None and rel[-1] == {"kind": "replaces", "task_id": "d1"}, rel


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["no", "keep both"])
async def test_a_no_to_the_contextual_question_keeps_both_running(monkeypatch, answer):
    ctx = await R4._confirm_call(monkeypatch, R4._answer(answer), partial=CAMPUS, db=_db("no", answer))
    client = ctx["client"]
    _record(f"one_running_no[{answer}]", f"{CAMPUS!r} while A runs; asked; the caller: {answer!r}",
            client, {"questions": _stops(ctx["provider"])})
    assert ctx["asked"]
    assert not R4._cancelled_a(ctx) and _phases(client, "d1")[-1] == "completed"
    assert _phases(client, "d2")[-1] == "completed"
    assert all(rel is None for rel in _relations(client, "d2"))


# ══════════════════════════════════════════════════════════════════════
# 3. Unrelated controls: no question, no link (en + fa)
# ══════════════════════════════════════════════════════════════════════

UNRELATED = [
    (PROFESSORS, "no, what's the weather in toronto right now?"),
    (A_FA, "نه، هوای تورنتو چطوره؟"),
    (PROFESSORS, "no, how do I get to the university of toronto from union station?"),
    (PROFESSORS, "no, find a pharmacy near union station"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "followup"), UNRELATED)
@pytest.mark.parametrize("running", [True, False], ids=["running", "finished"])
async def test_control_an_independent_follow_up_asks_nothing_and_links_nothing(
    monkeypatch, first, followup, running,
):
    """CONTROL (both trees)."""

    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, first, followup, running=running, db=_db("unrel", first, followup, running),
    )
    assert killed == [] and "superseded" not in _phases(client, "d1")
    assert all(rel is None for rel in _relations(client, "d2")), _relations(client, "d2")
    assert not _stops(provider) and not _which(provider), _commentary(provider)
    assert BINDING not in calls[-1] and "Request the caller is correcting" not in calls[-1]


@pytest.mark.asyncio
async def test_documented_conflict_the_hotel_imperative_still_asks_and_never_binds(monkeypatch):
    """PIN CHANGED (R2 addendum 6 R6-8): the resolved conflict — 'no, book me
    a hotel in toronto' runs normally with NO stop question and no binding;
    the new R4.PARTIAL (≥ half its own words shared) still asks once and is
    never bound."""

    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, PROFESSORS, R4.PARTIAL_R5, running=True, db=_db("hotel"),
    )
    assert killed == [] and "superseded" not in _phases(client, "d1")
    assert _stops(provider) == [] and not _which(provider)
    assert BINDING not in calls[-1]


@pytest.mark.asyncio
async def test_the_new_partial_trigger_asks_once_and_never_binds(monkeypatch):
    """R6-8: R4.PARTIAL shares at least half of its own words — it asks once
    (never cancels) and is never bound as a contextual correction."""

    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, PROFESSORS, R4.PARTIAL, running=True, db=_db("dorm"),
    )
    assert killed == [] and "superseded" not in _phases(client, "d1")
    assert len(_stops(provider)) == 1
    assert BINDING not in calls[-1]


# ══════════════════════════════════════════════════════════════════════
# 4. Finished work: exactly one plausible job → the `replaces` link with its
#    request + answer as correcting context
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "followup"), [
    (PROFESSORS, CAMPUS),
    (PROFESSORS, THESE),
    (A_FA, V1_FA),                 # the V1 anaphoric statement, finished
    (A_FA, CAMPUS_FA),
])
async def test_a_contextual_correction_of_one_finished_job_links_replaces(monkeypatch, first, followup):
    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, first, followup, running=False, db=_db("fin", first, followup),
    )
    _record(f"one_finished[{followup}]", f"A={first!r} finished; the caller: {followup!r}", client, {})
    rel = _relations(client, "d2")
    assert rel and all(r == {"kind": "replaces", "task_id": "d1"} for r in rel), rel
    assert "superseded" not in _phases(client, "d1") and "cancelled" not in _phases(client, "d1")
    refined = calls[-1].split("Current caller request:")[0]
    assert "Request the caller is correcting" in refined and first in refined, refined
    assert "Done." in refined, "the corrected answer is context"
    assert not _which(provider)


# ══════════════════════════════════════════════════════════════════════
# 5. TARGET AMBIGUITY: several plausible jobs, nothing selects one → ask
#    WHICH first; nothing is dispatched until the caller says
# ══════════════════════════════════════════════════════════════════════

async def _two_jobs_call(
    monkeypatch, steps, *, first=PROFESSORS, second=HOTEL, followup=CAMPUS,
    running=True, together=True, features=V03, db="",
):
    """A (d1) and B (d2) run (or both finished); the caller's `followup` is
    delegated as d3; the model says every commentary line (results, relay
    lines, the relay's questions) aloud as its own epoch; `steps(ctx, p,
    client)` plays the phone and the caller."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    ctx: dict = {"calls": [], "killed": [], "questions": [], "lines": []}
    release = asyncio.Event()
    both_asked = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        ctx["calls"].append((did, task))
        if did == "d2":
            both_asked.set()
        if did in {"d1", "d2"} and running:
            try:
                await asyncio.wait_for(release.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                ctx["killed"].append(did)
                raise
        elif did in {"d1", "d2"} and together:
            # Asked together, finished together: neither result was heard
            # before the other request was made (both stay plausible).
            await asyncio.wait_for(both_asked.wait(), timeout=4.0)
            await asyncio.sleep(0.2)
        return {"d1": "Professor X and Professor Y.", "d2": "The Royal York."}.get(did, "Done."), "m"

    H.patch_relay(monkeypatch, think=think)
    ctx["release"] = release
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
            p.push(H.out_text(content, start, start + 1500))
            p.push(H.out_audio("Q"))
            clock["at"] += 6000
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        p = box["p"]
        await _wait(lambda: len(ctx["calls"]) >= 1)
        if not running and not together:
            await _wait(lambda: R4._terminal(client, "d1"), 4.0)
            await asyncio.sleep(1.2)          # A's result was said before B was asked
        p.push(H.user_delta(second, 2000, 2800))
        p.push(H.delegation("d2", 2850))
        await _wait(lambda: len(ctx["calls"]) >= 2)
        if not running:
            await _wait(lambda: R4._terminal(client, "d1") and R4._terminal(client, "d2"), 4.0)
            await asyncio.sleep(1.2)          # the results' deliveries settle
        p.push(H.user_delta(followup, 5000, 5800))
        p.push(H.delegation("d3", 5850))
        if running or together:
            await _wait(lambda: ctx["questions"] and client.of("audio_delta"), 3.0)
        else:
            await _wait(lambda: _frames_for(client, "d3"), 3.0)
        await asyncio.sleep(0.3)
        ctx["before"] = {
            "d3": list(_frames_for(client, "d3")),
            "calls": [d for d, _t in ctx["calls"]],
        }
        await steps(ctx, p, client)
        await asyncio.sleep(0.6)
        release.set()
        await _wait(lambda: all(R4._terminal(client, d) for d in ("d1", "d2")), 5.0)
        await _wait(lambda: R4._terminal(client, "d3") or not _frames_for(client, "d3"), 3.0)
        await asyncio.sleep(0.4)

    client = Phone([H.config(features=features), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, session_id=_psid("two", db))
    ctx["client"], ctx["provider"] = client, provider
    await H.run_relay(client, provider, timeout=24, db_session_id=db or _db("two", repr(steps)))
    return ctx


def _heard_last(client):
    return R4._idle(R4._epoch_of(client))


def _reply(text, *, at_offset=2500, receipt=True, delegate=False):
    async def steps(ctx, p, client):
        if receipt:
            _send(client, _heard_last(client))
            await asyncio.sleep(0.2)
        _content, _start, q_end = ctx["questions"][-1]
        p.push(H.user_delta(text, q_end + at_offset, q_end + at_offset + 500))
        if delegate:
            p.push(H.delegation("d-answer", q_end + at_offset + 550))
        await asyncio.sleep(1.0)
    return steps


def _assert_held_then_asked(ctx, titles):
    before = ctx["before"]
    assert before["d3"] == [], f"dispatched before the target was known: {before['d3']}"
    assert "d3" not in before["calls"], before["calls"]
    which = _which(ctx["provider"])
    assert which, _commentary(ctx["provider"])
    for title in titles:
        assert live.request_title(title, 60) in which[0], which
    # all existing jobs untouched while the question is open
    for did in ("d1", "d2"):
        assert "cancelled" not in _phases(ctx["client"], did)
        assert "superseded" not in _phases(ctx["client"], did)


@pytest.mark.asyncio
async def test_two_running_jobs_and_a_contextual_correction_ask_which_first(monkeypatch):
    """Nothing dispatched, the which-question asked, both jobs untouched; with
    no answer (expiry) nothing is ever dispatched or linked."""

    monkeypatch.setattr(live, "_CONFIRM_EXPIRY_MS", 1000, raising=False)
    ctx = await _two_jobs_call(monkeypatch, _reply("the professors one", at_offset=4000), db=_db("expire"))
    _record("two_running_which_expired", f"A and B run; {CAMPUS!r}; which?; answered after expiry",
            ctx["client"], {"which": _which(ctx["provider"])})
    _assert_held_then_asked(ctx, [PROFESSORS, HOTEL])
    client = ctx["client"]
    assert _frames_for(client, "d3") == [], _frames_for(client, "d3")
    assert "d3" not in [d for d, _t in ctx["calls"]]
    assert [p for p in _phases(client, "d1")][-1] == "completed" and ctx["killed"] == []
    notes = [e for e in _thinking(ctx["provider"]) if e.get("delegation_id") == "d3"]
    assert notes and "Nothing was started" in notes[0]["content"], notes


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "target", "title"), [
    ("the professors one", "d1", PROFESSORS),
    ("the first one", "d1", PROFESSORS),
    ("the hotel search", "d2", HOTEL),
    ("the second one", "d2", HOTEL),
])
async def test_an_answer_selecting_one_job_continues_as_the_single_target_path(
    monkeypatch, answer, target, title,
):
    ctx = await _two_jobs_call(monkeypatch, _reply(answer), db=_db("pick", answer))
    client, provider = ctx["client"], ctx["provider"]
    _record(f"two_running_which[{answer}]", f"A and B run; {CAMPUS!r}; which?; the caller: {answer!r}",
            client, {"which": _which(provider), "stops": _stops(provider)})
    _assert_held_then_asked(ctx, [PROFESSORS, HOTEL])
    # the pure answer turn is covered by the refined request (never re-asked)
    answer_turn = next(f["turn_id"] for f in R4._finals(client) if f["text"] == answer)
    created = _lifecycle(_frames_for(client, "d3"))[0]
    assert created["request_turn_ids"][-1] == answer_turn, created
    # the refined request is created only now, bound to THAT job's request
    phases = _phases(client, "d3")
    assert phases and phases[0] == "created", phases
    refined = _input_for(ctx["calls"], "d3").split("Current caller request:")[0]
    assert BINDING in refined and title in refined, refined
    # …and the bound stop question names that job — never an automatic cancel
    stops = _stops(provider)
    assert len(stops) == 1 and live.request_title(title, 60) in stops[0], stops
    assert ctx["killed"] == []
    for did in ("d1", "d2"):
        assert "cancelled" not in _phases(client, did) and "superseded" not in _phases(client, did)
    assert all(rel is None for rel in _relations(client, "d3"))


@pytest.mark.asyncio
async def test_a_persian_contextual_correction_with_two_jobs_asks_which_in_persian(monkeypatch):
    ctx = await _two_jobs_call(
        monkeypatch, _reply("اولی"), first=A_FA, second=HOTEL_FA, followup=CAMPUS_FA,
        db=_db("fa-pick"),
    )
    _assert_held_then_asked(ctx, [A_FA, HOTEL_FA])
    assert "منظورت کدومه" in _which(ctx["provider"])[0]
    refined = _input_for(ctx["calls"], "d3").split("Current caller request:")[0]
    assert BINDING in refined and A_FA in refined, refined
    stops = _stops(ctx["provider"])
    assert len(stops) == 1 and "رو متوقف کنم؟" in stops[0] and live.request_title(A_FA, 60) in stops[0]


@pytest.mark.asyncio
async def test_an_unclear_answer_is_asked_once_more_then_a_clear_one_selects(monkeypatch):
    async def steps(ctx, p, client):
        await _reply("both")(ctx, p, client)
        await _wait(lambda: len(ctx["questions"]) >= 2, 3.0)
        await _wait(lambda: len(client.of("audio_delta")) >= 2, 2.0)
        await _reply("the hotel one", at_offset=1500)(ctx, p, client)

    ctx = await _two_jobs_call(monkeypatch, steps, db=_db("again"))
    which = _which(ctx["provider"])
    assert len(which) == 2 and which[1].startswith("Sorry — which one"), which
    refined = _input_for(ctx["calls"], "d3").split("Current caller request:")[0]
    assert BINDING in refined and HOTEL in refined, refined


@pytest.mark.asyncio
async def test_an_answer_the_model_delegates_is_absorbed(monkeypatch):
    ctx = await _two_jobs_call(monkeypatch, _reply("the professors one", delegate=True), db=_db("absorb"))
    client = ctx["client"]
    assert _frames_for(client, "d-answer") == [], "a card for the answer"
    assert "d-answer" not in [d for d, _t in ctx["calls"]]
    notes = [e for e in _thinking(ctx["provider"]) if e.get("delegation_id") == "d-answer"]
    assert notes and "handled" in notes[0]["content"], notes
    assert _phases(client, "d3")[:1] == ["created"]


@pytest.mark.asyncio
async def test_an_unheard_which_question_binds_no_answer(monkeypatch):
    """No receipt for the question: the answer is not bound to it (§5.3), the
    held request is released to the model to ask itself — nothing dispatched."""

    ctx = await _two_jobs_call(monkeypatch, _reply("the professors one", receipt=False), db=_db("unheard"))
    client = ctx["client"]
    assert _frames_for(client, "d3") == []
    notes = [e for e in _thinking(ctx["provider"]) if e.get("delegation_id") == "d3"]
    assert notes and "Ask them briefly which one" in notes[0]["content"], notes


@pytest.mark.asyncio
async def test_control_an_independent_request_with_two_jobs_running_runs_normally(monkeypatch):
    """CONTROL (both trees): a self-contained request while two jobs run —
    dispatched at once, no which-question, no stop question, no link."""

    followup = "no, what's the weather in toronto right now?"
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list = []
    killed: list = []
    release = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        calls.append(did)
        if did in {"d1", "d2"}:
            try:
                await asyncio.wait_for(release.wait(), timeout=6.0)
            except asyncio.CancelledError:
                killed.append(did)
                raise
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(PROFESSORS, 0, 900))
            p.push(H.delegation("d1", 950))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        p = box["p"]
        await _wait(lambda: len(calls) >= 1)
        p.push(H.user_delta(HOTEL, 2000, 2800))
        p.push(H.delegation("d2", 2850))
        await _wait(lambda: len(calls) >= 2)
        p.push(H.user_delta(followup, 5000, 5800))
        p.push(H.delegation("d3", 5850))
        await _wait(lambda: _frames_for(client, "d3"), 3.0)
        await asyncio.sleep(0.4)
        release.set()
        await _wait(lambda: R4._terminal(client, "d3"), 5.0)
        await asyncio.sleep(0.3)

    client = Phone([H.config(features=V03), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, session_id=_psid("indep"))
    await H.run_relay(client, provider, timeout=14, db_session_id=_db("indep"))
    assert _phases(client, "d3")[:1] == ["created"], _phases(client, "d3")
    assert not _which(provider) and not _stops(provider), _commentary(provider)
    assert killed == [] and all(rel is None for rel in _relations(client, "d3"))


@pytest.mark.asyncio
async def test_two_finished_jobs_and_a_contextual_correction_ask_which_and_force_no_link(monkeypatch):
    ctx = await _two_jobs_call(
        monkeypatch, _reply("the hotel one"), running=False, db=_db("fin-two"),
    )
    _record("two_finished_which", f"A and B finished; {CAMPUS!r}; which?; the caller: 'the hotel one'",
            ctx["client"], {"which": _which(ctx["provider"])})
    _assert_held_then_asked(ctx, [PROFESSORS, HOTEL])
    client = ctx["client"]
    rel = _relations(client, "d3")
    assert rel and all(r == {"kind": "replaces", "task_id": "d2"} for r in rel), rel
    refined = _input_for(ctx["calls"], "d3").split("Current caller request:")[0]
    assert "Request the caller is correcting" in refined and HOTEL in refined, refined
    assert not _stops(ctx["provider"]), "nothing running: no stop question"


@pytest.mark.asyncio
async def test_two_finished_jobs_and_no_answer_link_nothing(monkeypatch):
    monkeypatch.setattr(live, "_CONFIRM_EXPIRY_MS", 1000, raising=False)
    ctx = await _two_jobs_call(
        monkeypatch, _reply("the hotel one", at_offset=4000), running=False, db=_db("fin-none"),
    )
    _assert_held_then_asked(ctx, [PROFESSORS, HOTEL])
    assert _frames_for(ctx["client"], "d3") == []


@pytest.mark.asyncio
async def test_after_the_caller_moved_on_only_the_newest_result_is_the_target(monkeypatch):
    """A finished, its result SAID; then the caller asked for B, which finished
    too; then "no, what about the downtown campus?".  The caller moved on from
    A when they asked for B, so B is the one plausible target: linked
    `replaces` with no question — the V1 shape (the address asked for after
    the earlier answers; test_live_three_videos.py pins V1 itself)."""

    async def nothing(ctx, p, client):
        await asyncio.sleep(0.3)

    ctx = await _two_jobs_call(monkeypatch, nothing, running=False, together=False, db=_db("moved-on"))
    client = ctx["client"]
    assert not _which(ctx["provider"]), _commentary(ctx["provider"])
    rel = _relations(client, "d3")
    assert rel and all(r == {"kind": "replaces", "task_id": "d2"} for r in rel), rel
