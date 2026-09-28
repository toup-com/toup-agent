"""R2 addendum 6 R6-10 (round-6 classifier verifier ae0bd30b… + supervisor 03:56).

C-1  Answers that were pure stay pure: a residue made only of the confirmation
     lexicon + closed function words / idioms ('Sure, why not.', 'Sure
     thing.', 'Both of them.', "Yes, that's all.", 'Yeah, go for it.') is part
     of the answer — never a card; its reask is 'answered'.
C-2  A same-polarity verb whose complement is A's own action or a reference
     restates the answer ('No, keep looking.', 'No, let it finish.', 'Yes,
     stop searching for the professors.', «بذار تموم شه», «دیگه لازمش
     نداریم», «جستجوی استادها رو ول کن»): A's title words by stem, negated
     need in every person; begin() absorbs a keep restatement like a cancel.
C-3  A which-answer that selects exactly one job is `selected` whatever its
     copula / light verb / repair frame ("it's the hotel one", 'pick the hotel
     one', «اولی رو میگم», «منظورم اولی بود») — scored BEFORE the predicate.
C-4  Tail repair / hedge markers are framing ('…, I mean', 'I guess', «… رو
     میگم»); «هتله» / «استادا» fold to their stems; the remainder is never the
     whole turn or a bare marker.
C-5  Hedges ("I'm not sure", 'whichever', «نمی‌دونم», «فرقی نمی‌کنه») are
     `ambiguous`: one re-ask, consumed when nothing of their own.
C-6  Stacked / punctuated connectors never leave an empty lead.
C-7  A corrective 'that' subject ("no, that's wrong", "no, that's not the right
     campus") is CONTEXT-DEPENDENT → PARTIAL (running: ask + bind, never an
     automatic cancel; finished: link).
'I mean NP' parity: FULL with exactly one determinable running target;
     several → which-question; "I mean, <question>" is no modification.
V1   A candidate's lexical evidence includes the context it was bound/linked
     to (a request that REFERS BACK carries the result it resolved against);
     equal selection + an anaphoric/elliptical correction → the most recently
     delivered result; otherwise the which-question (no subject keywords).
C10  fast_failed_b: a failed B keeps a saved row when the confirmation applies
     (live card == saved history), both schedules; control: no relation → no
     row, as always.
C12  late input: the caller transcript for a continuation window arrives AFTER
     the hold bound: reask, carry and coverage verdicts use the re-evaluated
     parent; the wire parent the app got is recorded (ROUND6B_C12_OUT) for the
     cross-layer replay.

Wire tests run the REAL relay (`test_live_harness`, a distinct provider
session id per socket); pure tests grade the shared helpers, en + fa.
Old-fail: this file against a copy of fx4-relay-snapG (lvp 94a9a1a0…).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os

import pytest

import test_live_harness as H
import test_live_no_receipt_r4 as NR
import test_live_round4 as R4
import test_live_round6 as R6
from app.config import settings
from app.services import live_voice_protocol as live


PROFESSORS = R6.PROFESSORS
A_FA = R6.A_FA
HOTEL = R6.HOTEL
HOTEL_FA = R6.HOTEL_FA
BINDING = R6.BINDING
FA_FOLLOWUP = "نه، پردیس داون‌تاون چطور؟"
PROF_TITLE = frozenset(live._content_words(PROFESSORS))
FA_TITLE = frozenset(live._content_words(A_FA))
EN_WORDS = [live._content_words(PROFESSORS), live._content_words(HOTEL)]
FA_WORDS = [live._content_words(A_FA), live._content_words(HOTEL_FA)]


def _title(text: str) -> frozenset:
    return PROF_TITLE if text.isascii() else FA_TITLE


def _created(client, did="d-answer") -> list:
    return [f.get("title") for f in R6._frames_for(client, did) if f.get("phase") == "created"]


# ══════════════════════════════════════════════════════════════════════
# C-1 / C-2 — the confirmation answer (pure)
# ══════════════════════════════════════════════════════════════════════

PURE = [
    # C-1: closed function words / idioms (pure on snapF, regressed in fx8)
    ("Sure, why not.", "assent"), ("Sure thing.", "assent"), ("Both of them.", "keep"),
    ("Yes, that's all.", "assent"), ("No, that's all, thanks.", "keep"),
    ("Yeah, go for it.", "assent"), ("Yes, of course.", "assent"),
    ("yes please, that'd be great", "assent"), ("آره، همینه", "assent"),
    # C-2: a restated stop / keep of A
    ("No, keep looking.", "keep"), ("No, let it finish.", "keep"), ("No, let it run.", "keep"),
    ("No, keep both of them going.", "keep"), ("No, keep going with it.", "keep"),
    ("No, continue with it.", "keep"), ("no, keep it running", "keep"), ("keep it running", "keep"),
    ("no, keep the professors search running", "keep"),
    ("Yes, stop searching for the professors.", "assent"),
    ("Yes, stop looking for professors.", "assent"),
    ("yes, we don't need the professors anymore", "assent"),
    ("نه، بذار تموم شه.", "keep"), ("نه، بذار کارشو بکنه", "keep"),
    ("آره، دیگه لازمش نداریم.", "assent"), ("آره، لازم ندارن", "assent"),
    ("آره، جستجوی استادها رو ول کن.", "assent"),
    # the same closed classes the relay already holds: the keep verb "leave",
    # the stop idioms, and HOW/WHEN to stop (the cancel-modifier class)
    ("nah, leave it", "keep"), ("no leave it running", "keep"), ("yes stop it immediately", "assent"),
    ("yes, forget it", "assent"), ("yes, get rid of it", "assent"), ("آره، بیخیالش", "assent"),
    ("آره ولش کن بابا", "assent"), ("no, keep it playing", "keep"),
]


@pytest.mark.parametrize(("text", "kind"), PURE)
def test_c1_c2_a_pure_answer_has_no_request(text, kind):
    """Old (snapG): 'why not' / 'thing' / 'of them' / "that's all" and the
    restated verb phrases ('keep looking', «بذار تموم شه») came back as the
    caller's REQUEST — a card of their own, reask 'accepted'."""

    reading = live.read_confirmation_answer(text, _title(text))
    assert (reading.kind, reading.request) == (kind, ""), reading


@pytest.mark.parametrize(("text", "kind", "words"), [
    # CONTROLS: a real request after the answer stays the caller's (R6-1).
    ("yes, what did you find?", "assent", "what did you find?"),
    ("yes, I changed my mind", "assent", "I changed my mind"),
    ("yes please find the dean", "assent", "find the dean"),
    ("stop it please find the dean", "assent", "find the dean"),
    ("yes stop the hotel one", "assent", "stop the hotel one"),
    ("yes, stop searching for hotels", "assent", "stop searching for hotels"),
    ("yes, look for professors at waterloo", "assent", "look for professors at waterloo"),
    ("yes let's find a pharmacy", "assent", "find a pharmacy"),      # "let's" is no permissive keep
    ("yes, let me know when the hotel is booked", "assent", "let me know when the hotel is booked"),
    ("آره، جستجوی هتل رو قطع کن", "assent", "جستجوی هتل رو قطع کن"),
    ("no keep it running and find a pharmacy near union station", "keep",
     "find a pharmacy near union station"),
])
def test_c2_control_a_request_after_the_answer_stays_the_callers(text, kind, words):
    reading = live.read_confirmation_answer(text, _title(text))
    assert (reading.kind, reading.request) == (kind, words), reading


@pytest.mark.parametrize("text", [
    "No, stop it.", "yes but wait", "No, that's fine.",
    "آره، بذار تموم شه",                  # "yes" + "let it finish": opposite polarities
    "yes, stop it and keep looking", "yes, leave it", "نه، ولش کن",
])
def test_c2_control_opposite_polarity_stays_ambiguous(text):
    assert live.read_confirmation_answer(text, _title(text)).kind == "ambiguous", text


# ══════════════════════════════════════════════════════════════════════
# C-1 / C-2 — wire: pure answers are consumed, answered and never run
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "stops"), [
    ("Sure, why not.", True), ("Both of them.", False), ("Yes, that's all.", True),
    ("No, keep looking.", False), ("No, let it finish.", False),
    ("Yes, stop searching for the professors.", True),
    ("نه، بذار تموم شه.", False), ("آره، دیگه لازمش نداریم.", True),
    ("آره، جستجوی استادها رو ول کن.", True),
])
async def test_c1_c2_a_delegated_pure_answer_never_runs_as_a_task(monkeypatch, text, stops):
    ctx = await R6._confirm(monkeypatch, R6._answer(text, delegate=True),
                            db=R6._db("r6b-pure-del", text), **R6._fa(text))
    client = ctx["client"]
    assert len(R6._questions_said(ctx)) == 1, R6._questions_said(ctx)
    assert R6._cancelled_a(ctx) is stops, R6._phases(client, "d1")
    assert _created(client) == [], _created(client)
    assert [did for did, _d in ctx["calls"]] == ["d1", "d2"], ctx["calls"]


def _then_reask(text, marks):
    async def steps(ctx, p, client):
        await R6._answer(text)(ctx, p, client)
        await R6._wait(lambda: any(any(m in c for m in marks) for c, *_ in ctx["lines"]), 3.0)
        await R6._wait(lambda: len(client.of("audio_delta")) >= 2, 2.0)
        await asyncio.sleep(0.2)
        R6._send(client, R4._idle(client.of("audio_delta")[-1]["response_id"]))
        await asyncio.sleep(0.4)
        answer = [f for f in R6._finals(client) if f["text"] == text][0]
        ctx["answer_turn"] = answer["turn_id"]
        R6._send(client, {"type": "inject_text", "text": text, "reason": "no_response",
                          "reask_of_user_turn_id": answer["turn_id"]})
        await asyncio.sleep(0.6)
    return steps


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "stops"), [
    ("Sure thing.", True), ("No, keep both of them going.", False),
    ("آره، دیگه لازمش نداریم.", True),
])
async def test_c1_c2_the_outcome_line_answers_the_answer_turn(monkeypatch, text, stops):
    marks = R6.STOPPED_MARKS if stops else R6.KEPT_MARKS
    ctx = await R6._confirm(monkeypatch, _then_reask(text, marks),
                            db=R6._db("r6b-pure-reask", text), **R6._fa(text))
    client = ctx["client"]
    assert R6._cancelled_a(ctx) is stops, R6._phases(client, "d1")
    outcome = [f for f in client.of("response_text") if any(m in (f.get("text") or "") for m in marks)]
    assert outcome and outcome[0].get("parent_user_turn_id") == ctx["answer_turn"], outcome[:1]
    results = [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")]
    assert results == [(ctx["answer_turn"], "answered")], results


@pytest.mark.asyncio
@pytest.mark.parametrize(("forced", "absorbed"), [
    ("keep looking", True),               # a keep restatement of A
    ("let it finish", True),
    ("keep the professors search going", True),
    ("look for waterloo robotics professors", False),   # CONTROL: a real request runs
])
async def test_c2_begin_absorbs_a_keep_restatement_like_a_cancel(monkeypatch, forced, absorbed):
    """Defense in depth: even if the classifier handed back a remainder that
    restates KEEP, begin() absorbs it with the C5 result — one answer, one
    action, never a second card."""

    text = f"no, {forced}"
    real = live.read_confirmation_answer

    def reading(value, title_words=frozenset()):
        if value == text:
            return live.ConfirmationReading("keep", forced, True)
        return real(value, title_words)

    monkeypatch.setattr(live, "read_confirmation_answer", reading)
    ctx = await R6._confirm(monkeypatch, R6._answer(text, delegate=True), db=R6._db("r6b-keep-dd", forced))
    client, provider = ctx["client"], ctx["provider"]
    assert not R6._cancelled_a(ctx), R6._phases(client, "d1")
    if absorbed:
        assert R6._frames_for(client, "d-answer") == [], R6._frames_for(client, "d-answer")
        assert [did for did, _d in ctx["calls"]] == ["d1", "d2"], ctx["calls"]
        notes = R6._notes(provider, "d-answer")
        assert notes and "handled" in notes[0], notes
    else:
        assert ("d-answer", forced) in ctx["calls"], ctx["calls"]


# ══════════════════════════════════════════════════════════════════════
# C-3 .. C-6 — the which-question answer (pure)
# ══════════════════════════════════════════════════════════════════════

WHICH = [
    # C-3: the selection is scored before the predicate
    ("it's the hotel one", "en", ("selected", 1, "")), ("that's the hotel one", "en", ("selected", 1, "")),
    ("I want the first one", "en", ("selected", 0, "")), ("pick the hotel one", "en", ("selected", 1, "")),
    ("take the first one", "en", ("selected", 0, "")), ("I'd say the hotel one", "en", ("selected", 1, "")),
    ("definitely the hotel one", "en", ("selected", 1, "")),
    ("the hotel one is what I meant", "en", ("selected", 1, "")),
    ("let's go with the hotel one", "en", ("selected", 1, "")),
    ("اولی رو میگم", "fa", ("selected", 0, "")), ("منظورم اولی بود", "fa", ("selected", 0, "")),
    ("هتل رو انتخاب می‌کنم", "fa", ("selected", 1, "")),
    # C-4: tail markers are framing; spoken Persian folds
    ("the hotel one, I mean", "en", ("selected", 1, "")), ("the hotel one I mean", "en", ("selected", 1, "")),
    ("the first one I guess", "en", ("selected", 0, "")), ("the hotel one I think", "en", ("selected", 1, "")),
    ("هتل رو میگم", "fa", ("selected", 1, "")), ("استادها رو میگم", "fa", ("selected", 0, "")),
    ("هتله", "fa", ("selected", 1, "")), ("استادا", "fa", ("selected", 0, "")),
    ("منظورم هتله", "fa", ("selected", 1, "")), ("منظورم دومیه", "fa", ("selected", 1, "")),
    ("the professors one at waterloo", "en", ("selected", 0, "at waterloo")),   # never the whole turn
    # C-5: hedges
    ("I'm not sure", "en", ("ambiguous", -1, "")), ("whichever", "en", ("ambiguous", -1, "")),
    ("I don't know", "en", ("ambiguous", -1, "")), ("نمی‌دونم", "fa", ("ambiguous", -1, "")),
    ("فرقی نمی‌کنه", "fa", ("ambiguous", -1, "")), ("the hotel one, I'm not sure", "en", ("ambiguous", -1, "")),
    # C-6: stacked / punctuated connectors
    ("and also find me a pharmacy near union station", "en", ("none", -1, "")),
    ("also, what's the weather in toronto?", "en", ("none", -1, "")),
    ("و همچنین یه تاکسی برام بگیر", "fa", ("none", -1, "")),
    # CONTROLS (R6-8, unchanged): requests of their own; selections with a rest
    ("book me a hotel in toronto", "en", ("none", -1, "")),
    ("find me a pharmacy near union station", "en", ("none", -1, "")),
    ("how far is union station from the university of toronto?", "en", ("none", -1, "")),
    ("یه داروخونه نزدیک یونیون استیشن پیدا کن", "fa", ("none", -1, "")),
    ("هوای تورنتو چطوره؟", "fa", ("none", -1, "")),
    ("neither, I'm asking about the weather", "en", ("none", -1, "")),
    ("the professors one, and find me a pharmacy near union station", "en",
     ("selected", 0, "find me a pharmacy near union station")),
    ("the hotel one. also find me a pharmacy", "en", ("selected", 1, "find me a pharmacy")),
    ("both", "en", ("ambiguous", -1, "")), ("yes", "en", ("ambiguous", -1, "")),
    ("hmm, which?", "en", ("ambiguous", -1, "")),
]


@pytest.mark.parametrize(("text", "lang", "want"), WHICH)
def test_c3_c6_only_a_selection_clause_answers_the_which_question(text, lang, want):
    words = EN_WORDS if lang == "en" else FA_WORDS
    assert live.parse_target_answer(text, words) == want, text


@pytest.mark.parametrize(("text", "lang"), [
    ("I'm not sure", "en"), ("whichever", "en"), ("نمی‌دونم", "fa"), ("فرقی نمی‌کنه", "fa"),
    ("the hotel one, I mean", "en"),
])
def test_c4_c5_a_hedge_or_a_marker_holds_nothing_of_the_callers_own(text, lang):
    words = EN_WORDS if lang == "en" else FA_WORDS
    assert live.target_answer_residue(text, words) == set(), text


# ══════════════════════════════════════════════════════════════════════
# C-3 .. C-6 — wire: two jobs running, the which-question, the answer
# ══════════════════════════════════════════════════════════════════════

def _fa_jobs(fa: bool) -> dict:
    return {"first": A_FA, "second": HOTEL_FA, "followup": FA_FOLLOWUP} if fa else {}


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "fa", "target"), [
    ("it's the hotel one", False, "d2"), ("I want the first one", False, "d1"),
    ("the hotel one, I mean", False, "d2"),
    ("اولی رو میگم", True, "d1"), ("هتله", True, "d2"), ("استادا", True, "d1"),
])
async def test_c3_c4_a_framed_selection_binds_and_never_runs(monkeypatch, answer, fa, target):
    """Old: 'none' (the predicate test first) — the held correction dropped
    and the answer ran as its own card; or 'selected' with the marker ('I
    mean') / the whole turn as a MIXED remainder that ran."""

    ctx = await R6._jobs_call(monkeypatch, R6._reply(answer, delegate="d-answer"),
                              db=R6._db("r6b-which-sel", answer), **_fa_jobs(fa))
    client, provider = ctx["client"], ctx["provider"]
    assert R6._which(provider), "precondition: the which-question was asked"
    assert R6._phases(client, "d3"), "the selection did not bind; the held correction was dropped"
    title = {("d1", True): A_FA, ("d2", True): HOTEL_FA, ("d1", False): PROFESSORS, ("d2", False): HOTEL}[
        (target, fa)]
    head = R6._input_for(ctx, "d3")
    assert BINDING in head and title in head, head[:400]
    assert _created(client) == [], _created(client)
    assert ctx["killed"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "fa"), [("I'm not sure", False), ("whichever", False), ("نمی‌دونم", True)])
async def test_c5_a_hedge_reasks_once_and_never_runs(monkeypatch, answer, fa):
    ctx = await R6._jobs_call(monkeypatch, R6._reply(answer, delegate="d-answer"),
                              db=R6._db("r6b-which-hedge", answer), **_fa_jobs(fa))
    client, provider = ctx["client"], ctx["provider"]
    assert len(R6._which(provider)) == 2, R6._which(provider)   # one re-ask
    assert _created(client) == [], _created(client)
    assert ctx["killed"] == []


@pytest.mark.asyncio
async def test_c6_a_request_after_stacked_connectors_is_its_own_and_no_reask(monkeypatch):
    newreq = "and also find me a pharmacy near union station"
    ctx = await R6._jobs_call(monkeypatch, R6._reply(newreq, delegate="d-answer"), db=R6._db("r6b-andalso"))
    client, provider = ctx["client"], ctx["provider"]
    assert len(R6._which(provider)) == 1, R6._which(provider)   # the target question expires
    assert R6._phases(client, "d-answer")[:1] == ["created"], R6._phases(client, "d-answer")
    assert not R6._phases(client, "d3"), "nothing may be dispatched for the unanswered which-question"


# ══════════════════════════════════════════════════════════════════════
# C-7 — the corrective 'that' anaphor
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "no, that's wrong", "no, that's not the right campus", "no, that's the wrong campus",
    "no, that is wrong", "no that's not right", "no, it's fine, that's the wrong campus",
    "نه، این درست نیست",
])
def test_c7_a_corrective_that_is_context_dependent(text):
    held = set(PROF_TITLE if text.isascii() else FA_TITLE)
    assert live.contextual_reference(text) == "anaphora", text
    assert live.negation_evidence(text, held) == "partial", text


@pytest.mark.parametrize("text", ["no, that's fine", "no, that's okay, thanks", "no, that's it"])
def test_c7_control_a_pleasantry_or_a_closing_is_no_anaphor(text):
    assert live.contextual_reference(text) == "", text
    assert live.negation_evidence(text, set(PROF_TITLE)) == "none", text


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", ["no, that's wrong", "no, that's not the right campus"])
@pytest.mark.parametrize("running", [False, True], ids=["finished", "running"])
async def test_c7_a_that_correction_asks_or_links_never_cancels(monkeypatch, followup, running):
    """Old (snapG/fx8): NONE — an unrelated stranger job, no binding (snapF
    had FULL: an automatic supersede of the running job)."""

    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, PROFESSORS, followup, running=running, db=R6._db("r6b-c7", followup, running),
    )
    assert killed == [] and "superseded" not in R6._phases(client, "d1")
    rels = R6._relations(client, "d2")
    if running:
        questions = R6._stops(provider)
        assert len(questions) == 1 and live.request_title(PROFESSORS, 60) in questions[0], questions
        assert BINDING in calls[-1].split("Current caller request:")[0]
    else:
        assert rels and rels[-1] == {"kind": "replaces", "task_id": "d1"}, rels
        assert "Request the caller is correcting" in calls[-1]


# ══════════════════════════════════════════════════════════════════════
# 'I mean NP' parity (R6-10): FULL beside 'i meant' / «منظورم»
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "no, I mean the downtown campus", "no, I mean the ones at the downtown campus",
    "I mean the downtown campus", "no, I meant the downtown campus", "نه، منظورم پردیس داون‌تاونه",
])
def test_i_mean_np_is_an_explicit_modification(text):
    held = set(PROF_TITLE if text.isascii() else FA_TITLE)
    assert live.explicit_modification(text) == "correction", text
    assert live.negation_evidence(text, held) == "full", text


@pytest.mark.parametrize("text", [
    "I mean, what's the weather in toronto?", "no, I mean, can you check the weather",
])
def test_i_mean_control_a_filler_before_a_question_modifies_nothing(text):
    assert live.negation_evidence(text, set(PROF_TITLE)) == "none", text


@pytest.mark.parametrize("text", [
    "no, I'm asking about the downtown campus", "no, the downtown campus I mean",
])
def test_i_mean_control_the_other_repairs_stay_partial(text):
    assert live.contextual_reference(text) == "repair", text
    assert live.negation_evidence(text, set(PROF_TITLE)) == "partial", text


@pytest.mark.asyncio
async def test_i_mean_with_one_running_target_supersedes_without_a_question(monkeypatch):
    followup = "no, I mean the downtown campus"
    client, provider, calls, _killed = await R4._correction_call(
        monkeypatch, PROFESSORS, followup, running=True, db=R6._db("r6b-imean-one"),
    )
    assert "superseded" in R6._phases(client, "d1"), R6._phases(client, "d1")
    assert R6._relations(client, "d2")[-1] == {"kind": "replaces", "task_id": "d1"}
    assert not R6._stops(provider) and not R6._which(provider), R6._commentary(provider)
    assert "Request the caller is correcting" in calls[-1]


@pytest.mark.asyncio
async def test_i_mean_with_several_running_targets_asks_which(monkeypatch):
    ctx = await R6._jobs_call(monkeypatch, R6._nothing, followup="no, I mean the downtown campus",
                              db=R6._db("r6b-imean-two"))
    assert not ctx["before_d3"] and len(R6._which(ctx["provider"])) == 1, R6._which(ctx["provider"])
    assert ctx["killed"] == []


@pytest.mark.asyncio
async def test_i_mean_control_a_question_after_the_filler_runs_on_its_own(monkeypatch):
    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, PROFESSORS, "I mean, what's the weather in toronto right now?", running=True,
        db=R6._db("r6b-imean-ctl"),
    )
    assert killed == [] and "superseded" not in R6._phases(client, "d1")
    assert not R6._stops(provider) and not R6._which(provider)
    assert "Request the caller is correcting" not in calls[-1]


# ══════════════════════════════════════════════════════════════════════
# V1 structural rule — evidence includes the context a job was linked to;
# equal selection + anaphora/ellipsis → the most recently delivered result
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(("text", "back"), [
    ("آدرس دفترشون رو بفرست", True), ("کارشون ال‌ال‌امم", True), ("ایمیلشو پیدا کن", True),
    ("آدرسش رو بفرست", True), ("send me their office addresses", True),
    ("can you email it to me", True), ("what are they working on", True),
    # controls: self-contained requests, a dummy "it", «نشون بده» ("show")
    ("find me a hotel near union station", False), (HOTEL_FA, False), (A_FA, False),
    ("is it going to rain tomorrow", False), ("it's late, book me a taxi", False),
    ("نشون بده", False), ("بی‌خیال شو", False),
])
def test_v1_a_request_refers_back_by_structure(text, back):
    assert live.refers_back(text) is back, text


LINKED = {
    "en": (PROFESSORS, "send me their office addresses", HOTEL,
           "no, these aren't at the university of toronto downtown"),
    "fa": (A_FA, "آدرس دفترشون رو بفرست", HOTEL_FA, "نه، اینا دانشگاه تورنتو نیستند"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "fa"])
async def test_v1_an_anaphoric_correction_links_the_most_recent_linked_result(monkeypatch, lang):
    """The V1 shape, generalized (no campus keywords): the professors result,
    then a request that REFERS BACK to it («آدرس دفترشون رو بفرست»), then a
    correction whose words name the professors search.  The second job carries
    the context it resolved against, the words select both EQUALLY, and the
    anaphoric correction links the most recently delivered result — the
    second job (old: the older professors job by its words alone)."""

    first, anaphoric, _self_contained, followup = LINKED[lang]
    ctx = await R6._jobs_call(monkeypatch, R6._nothing, first=first, second=anaphoric, followup=followup,
                              schedule="sequential_finished", db=R6._db("r6b-v1-link", lang))
    rels = R6._relations(ctx["client"], "d3")
    assert rels and rels[-1] == {"kind": "replaces", "task_id": "d2"}, rels
    assert ("Request the caller is correcting (keep its constraints unless the current request "
            "changes them): " + anaphoric) in R6._input_for(ctx, "d3")
    assert not R6._which(ctx["provider"])


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "fa"])
async def test_v1_control_a_self_contained_second_request_carries_no_context(monkeypatch, lang):
    """CONTROL (w4 shape): the second request is self-contained — it resolved
    nothing against the first job's result — so the words select the first."""

    first, _anaphoric, self_contained, followup = LINKED[lang]
    ctx = await R6._jobs_call(monkeypatch, R6._nothing, first=first, second=self_contained,
                              followup=followup, schedule="sequential_finished",
                              db=R6._db("r6b-v1-self", lang))
    rels = R6._relations(ctx["client"], "d3")
    assert rels and rels[-1] == {"kind": "replaces", "task_id": "d1"}, rels


@pytest.mark.asyncio
async def test_v1_control_an_equal_selection_by_a_repair_asks_which(monkeypatch):
    """CONTROL: equal selection but a REPAIR ("I'm asking about …"), not an
    anaphoric/elliptical correction → the which-question, never a guess."""

    first, anaphoric, _self, _followup = LINKED["en"]
    ctx = await R6._jobs_call(
        monkeypatch, R6._nothing, first=first, second=anaphoric,
        followup="no, I'm asking about the university of toronto downtown",
        schedule="sequential_finished", db=R6._db("r6b-v1-repair"),
    )
    assert len(R6._which(ctx["provider"])) == 1, R6._which(ctx["provider"])
    assert not ctx["before_d3"]


# ══════════════════════════════════════════════════════════════════════
# C10 fast_failed_b — a failed B keeps its saved row when "yes" applies
# ══════════════════════════════════════════════════════════════════════

REPLACES = {"kind": "replaces", "task_id": "d1"}


async def _c10(monkeypatch, *, b_s, fail_b, answer, wait_b, db):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    ctx: dict = {"calls": [], "killed": [], "questions": [], "lines": []}
    a_done = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        ctx["calls"].append(did)
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
            await asyncio.sleep(b_s)
            if fail_b:
                raise RuntimeError("tenant down")
            return "Found a dorm.", "m"
        return "Done.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)
    box: dict = {}
    clock = {"at": 20000}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(PROFESSORS, 0, 900))
            p.push(H.delegation("d1", 950))
        elif e["type"] == "session.commentary.append":
            content = str(e.get("content") or "")
            start = clock["at"]
            if R4._is_question(content):
                ctx["questions"].append((start, start + 1500))
            ctx["lines"].append((content, start, start + 1500))
            p.push(H.out_text(content, start, start + 1500))
            p.push(H.out_audio("L"))
            clock["at"] += 6000
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await R6._wait(lambda: len(ctx["calls"]) >= 1)
        p = box["p"]
        p.push(H.user_delta(R6.PARTIAL_EN, 5000, 5800))
        p.push(H.delegation("d2", 5850))
        await R6._wait(lambda: ctx["questions"] and client.of("audio_delta"), 3.0)
        if wait_b:
            await R6._wait(lambda: R4._terminal(client, "d2"), 5.0)
            await asyncio.sleep(0.2)
        R6._send(client, R4._idle(R6._question_epoch(client)))
        await asyncio.sleep(0.2)
        at = max(end for _l, _s, end in ctx["lines"]) + 2500
        p.push(H.user_delta(answer, at, at + 500))
        await R6._wait(lambda: R4._terminal(client, "d2"), 6.0)
        await asyncio.sleep(0.8)
        a_done.set()
        await R6._wait(lambda: R4._terminal(client, "d1"), 3.0)
        await asyncio.sleep(0.5)

    client = R6.Phone([H.config(features=R6.V03), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, session_id=R6._psid("c10", db))
    ctx["client"], ctx["provider"] = client, provider
    await H.run_relay(client, provider, timeout=24, db_session_id=db)
    ref = f"live-delegation:{provider._session_id}:d2"
    ctx["rows"] = [s for s in recorded["saves"] if s.get("assistant_ref") == ref]
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize(("b_s", "wait_b"), [(0.4, True), (3.0, False)], ids=["fast", "slow"])
async def test_c10_a_failed_b_keeps_its_row_when_the_confirmation_applies(monkeypatch, b_s, wait_b):
    """Old (snapG, and the critic's test_fast_failed_b): the live card said
    B `failed` and `replaces` A, the saved history had no row for B at all."""

    ctx = await _c10(monkeypatch, b_s=b_s, fail_b=True, answer="yes", wait_b=wait_b,
                     db=R6._db("r6b-c10-fail", b_s))
    client = ctx["client"]
    assert R6._cancelled_a(ctx), R6._phases(client, "d1")
    life = R6._lifecycle(R6._frames_for(client, "d2"))
    assert life and life[-1].get("relation") == REPLACES, life
    assert {f.get("phase") for f in life if f.get("phase") in {"completed", "failed"}} == {"failed"}
    rows = ctx["rows"]
    assert rows, "B's row was never written"
    assert [(r.get("assistant_voice") or {}).get("related_task_id") for r in rows][-1] == "d1", rows
    voice = rows[-1]["assistant_voice"]
    assert voice.get("source") == "delegated_record", voice          # never proof of delivery
    assert rows[0].get("assistant_occurred_at") is not None          # its place under its request
    assert [d for d in ctx["calls"] if d == "d2"] == ["d2"]          # B ran once


@pytest.mark.asyncio
async def test_c10_control_a_failed_b_with_no_relation_writes_no_row(monkeypatch):
    """CONTROL: 'no' (keep both) — B failed and nothing names it: no row,
    exactly as before (an empty delegation never narrates itself)."""

    ctx = await _c10(monkeypatch, b_s=0.4, fail_b=True, answer="no", wait_b=True,
                     db=R6._db("r6b-c10-keep"))
    assert not R6._cancelled_a(ctx)
    assert ctx["rows"] == [], ctx["rows"]
    assert all(not f.get("relation") for f in R6._lifecycle(R6._frames_for(ctx["client"], "d2")))


# ══════════════════════════════════════════════════════════════════════
# C12 late input — the caller's transcript for the continuation window
# arrives AFTER the hold bound
# ══════════════════════════════════════════════════════════════════════

LATE_REPLY = "Yes, I'm here."
#: R6-13 / R6-15 PIN CHANGE (C12 VERSION-NEGOTIATED REPLAY): these tapes are
#: replayed through the CURRENT app hook, so they are recorded with the
#: features that app sends (its `LIVE_FEATURES`, incl. `media_scope_floor` and
#: the new `output_attribution`) — was the v0.3 list, which no app build
#: sends. The TF132 recording is tests/test_live_round7_c12.py's.
C12_APP_FEATURES = list(NR.APP_FEATURES) + ["media_scope_floor", "output_attribution"]


def _record_c12(name: str, what: str, sockets: list, relay: dict) -> None:
    """Cross-layer: written ONLY under ROUND6B_C12_OUT (never otherwise), in the
    no-receipt tape format — the REAL relay's frames to the phone (in), the
    phone's frames (out), per socket, plus the relay's ACTUAL verdicts and the
    WIRE parent the app got — for the cross-layer owner's replay through the
    app's turn watch."""

    path = os.environ.get("ROUND6B_C12_OUT")
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    data["_what"] = (
        "R2 addendum 6 R6-10 C12 late input: a continuation released at its hold bound "
        "(R6-13: with NO wire parent — its input window had not settled); the caller transcript "
        "for its window arrives after; the "
        "relay re-evaluates the parent before its reask / carry / coverage verdicts. Written "
        "by backend/tests/test_live_round6b.py under ROUND6B_C12_OUT."
    )
    data["_relay"] = {
        "tree": os.path.abspath(os.getcwd()),
        "live_voice_protocol_sha256": NR._sha(os.path.abspath(live.__file__)),
        "test_file_sha256": NR._sha(os.path.abspath(__file__)),
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    data["features"] = list(C12_APP_FEATURES)
    data.setdefault("scenarios", {})[name] = {
        "what": what,
        "sockets": [{"provider_session_id": psid, "tape": list(tape)} for psid, tape in sockets],
        "relay": relay,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


def _wire_parents(phone) -> dict:
    """epoch id → the parent_user_turn_id the phone was sent for it."""

    seen: dict[str, str] = {}
    for frame in phone.of("response_text"):
        epoch = str(frame.get("assistant_turn_id") or frame.get("response_id") or "")
        seen.setdefault(epoch, str(frame.get("parent_user_turn_id") or ""))
    return seen


#: The provider's OUTPUT timeline runs ahead of its input timeline (as in
#: production), so the input clock cannot settle the held window early: the
#: hold is released at its BOUND (the 650 ms window), never by the input.
AHEAD = 3000


async def _late_input(phone, box, facts):
    """R's answer fails to play; the model's next words (a reply to a
    'hello?' the relay has not heard yet) are HELD as a tentative
    continuation and released at the hold BOUND; only then does the
    transcript of 'hello?' (inside the window) arrive."""

    assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
    await asyncio.sleep(0.4)                     # the relay's gap timer retires e1
    e1 = NR._epoch(phone)
    phone.push_frame(NR._failed(e1))             # R's own answer never played
    await asyncio.sleep(0.2)
    p = box["p"]
    released_at = asyncio.get_event_loop().time()
    p.push(H.out_text(LATE_REPLY, AHEAD + 2050, AHEAD + 2400))
    p.push(H.out_audio(NR.pcm(1)))
    assert await NR._wait(lambda: NR._epoch(phone) != e1, 3.0)
    facts["held_s"] = asyncio.get_event_loop().time() - released_at
    await asyncio.sleep(0.4)                     # released: nobody (yet) in the window
    e2 = NR._epoch(phone)
    facts["e1"], facts["e2"] = e1, e2
    facts["wire_parents_before"] = dict(_wire_parents(phone))
    phone.push_frame(NR._idle(e2))               # the reply was heard
    await asyncio.sleep(0.2)
    p.push(H.user_delta("hello?", AHEAD + 1450, AHEAD + 1750))   # …LATE, inside the window
    assert await NR._wait(lambda: len(NR._finals(phone)) >= 2)
    await asyncio.sleep(0.3)
    facts["rid"] = NR._finals(phone)[0]["turn_id"]
    facts["hid"] = NR._finals(phone)[1]["turn_id"]


def _capture_coverage(monkeypatch, facts):
    real = live._LiveSession.stranded_turn_record

    def spy(self, detached):
        rid, hid = facts.get("rid"), facts.get("hid")
        if rid and hid and "coverage" not in facts:
            facts["coverage"] = {
                "R": self.turn_output_evidence(rid), "hello": self.turn_output_evidence(hid),
                "reparented": dict(self.reparented_epochs or {}),
            }
        return real(self, detached)

    monkeypatch.setattr(live._LiveSession, "stranded_turn_record", spy)


async def _c12_socket(monkeypatch, script_fn, *, db, psid, start=True, last=True, ahead=AHEAD):
    box: dict = {}

    def on_send(p, e):
        if e.get("type") == "session.start":
            box["p"] = p
            if start:
                p.push(H.user_delta(NR.REQUEST, 0, 900))
                p.push(H.out_text("There is one on Front Street.", ahead + 1000, ahead + 1400))
                p.push(H.out_audio(NR.pcm(0)))
        elif e.get("type") == "session.close":
            p.push(H.closed())

    provider = H.FakeProvider(on_send=on_send, session_id=psid, auto_ack=True)
    ref: list = []

    async def script():
        await script_fn(ref[0], box)

    items = [NR._config(C12_APP_FEATURES), script]
    if last:
        items += [0.3, {"type": "stop"}]
    phone = NR.Phone(items, tape=[])
    ref.append(phone)
    await H.run_relay(phone, provider, timeout=10, db_session_id=db)
    return phone, provider


@pytest.mark.asyncio
async def test_c12_late_input_reask_and_coverage_use_the_reevaluated_parent(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    _capture_coverage(monkeypatch, facts)

    async def script(phone, box):
        await _late_input(phone, box, facts)
        phone.reask(facts["rid"])
        assert await NR._wait(lambda: phone.of("reask_result"))

    psid = R6._psid("c12-late-reask")
    phone, provider = await _c12_socket(monkeypatch, script, db=R6._db("c12-late-reask"), psid=psid)
    result = phone.of("reask_result")[-1]
    _record_c12(
        "late_input_reask",
        "R's answer failed to play; a reply to an unheard 'hello?' held as a continuation and "
        "released at its hold bound (R6-13: wire parent withheld); the 'hello?' transcript arrives after; R "
        "re-asked by id",
        [(psid, phone.tape)],
        {"R": facts["rid"], "hello": facts["hid"], "epochs": [facts["e1"], facts["e2"]],
         "wire_parent_of_reply": facts["wire_parents_before"].get(facts["e2"]),
         "held_to_bound_s": round(facts["held_s"], 3),
         "relay_parent_of_reply": (facts.get("coverage") or {}).get("reparented", {}),
         "coverage": facts.get("coverage"), "reask": phone.of("reask_result"),
         "instructions": NR._reask_contents(provider)},
    )
    # The precondition: the reply was HELD to the bound (its 650 ms window; the
    # input clock never reached its start) — the transcript was not there yet.
    # R6-13 PIN CHANGE (addendum 6 R6-13, "the relay must not commit R as the
    # wire parent of a continuation whose input window has not settled at
    # release"): it went out with NO wire parent (was: R — the parent the app
    # counted as R's answer and never learned was re-evaluated).
    assert facts["held_s"] >= 0.6, facts["held_s"]
    assert facts["wire_parents_before"].get(facts["e2"]) == "", facts["wire_parents_before"]
    # R6-13: …and the negotiated phone is TOLD — provisionally R, then the
    # relay's re-evaluated verdict ('hello?') — so its watch converges.
    assert [(f["assistant_turn_id"], f["parent_user_turn_id"], f["settled"])
            for f in phone.of("output_attribution")] == [
        (facts["e2"], facts["rid"], False), (facts["e2"], facts["hid"], True),
    ], phone.of("output_attribution")
    # Coverage: the relay's re-evaluated parent — the reply answered 'hello?'.
    assert facts["coverage"]["R"] == live.OUTPUT_UNHEARD, facts["coverage"]
    assert facts["coverage"]["hello"] == live.OUTPUT_HEARD, facts["coverage"]
    # Reask: R is not answered by the reply to 'hello?' — it may have been
    # (conditional), its own answer provably unheard.
    assert (result["user_turn_id"], result["outcome"]) == (facts["rid"], "accepted"), result
    assert result.get("conditional") is True, result
    contents = NR._reask_contents(provider)
    assert len(contents) == 1 and "hello?" in contents[0], contents


@pytest.mark.asyncio
async def test_c12_late_input_the_carry_uses_the_reevaluated_parent(monkeypatch):
    """The same shape, then the socket drops: R is CARRIED (conditional,
    after 'hello?') — on the wire parent alone it would read as heard and
    nothing would be carried."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    _capture_coverage(monkeypatch, facts)
    db = R6._db("c12-late-carry")

    async def s1(phone, box):
        await _late_input(phone, box, facts)
        phone.push_frame(NR.DROP)

    async def s2(phone, box):
        await asyncio.sleep(0.3)
        phone.reask(facts["rid"])
        assert await NR._wait(lambda: phone.of("reask_result"))

    psid1, psid2 = R6._psid("c12-late-c1"), R6._psid("c12-late-c2")
    phone1, _p1 = await _c12_socket(monkeypatch, s1, db=db, psid=psid1, last=False)
    phone2, provider2 = await _c12_socket(monkeypatch, s2, db=db, psid=psid2, start=False)
    result = phone2.of("reask_result")[-1]
    _record_c12(
        "late_input_carried",
        "the late-input shape, then the socket dropped; R asked by id on the repaired socket",
        [(psid1, phone1.tape), (psid2, phone2.tape)],
        {"R": facts["rid"], "hello": facts["hid"], "epochs": [facts["e1"], facts["e2"]],
         "wire_parent_of_reply": facts["wire_parents_before"].get(facts["e2"]),
         "held_to_bound_s": round(facts["held_s"], 3),
         "coverage": facts.get("coverage"), "reask": phone2.of("reask_result"),
         "instructions": NR._reask_contents(provider2)},
    )
    assert facts["held_s"] >= 0.6, facts["held_s"]
    # R6-13 PIN CHANGE: released unsettled → no wire parent (was: R).
    assert facts["wire_parents_before"].get(facts["e2"]) == "", facts["wire_parents_before"]
    assert facts["coverage"]["R"] == live.OUTPUT_UNHEARD, facts["coverage"]
    assert (result["user_turn_id"], result["outcome"], result.get("conditional")) == (
        facts["rid"], "accepted", True,
    ), result
    contents = NR._reask_contents(provider2)
    assert len(contents) == 1 and "hello?" in contents[0], contents


@pytest.mark.asyncio
async def test_c12_control_no_late_input_keeps_r_covered(monkeypatch):
    """CONTROL: the same held continuation with NO caller speech in its window
    — it keeps R's parent, relay-side too; heard, R is answered."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}

    async def script(phone, box):
        assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.4)
        e1 = NR._epoch(phone)
        phone.push_frame(NR._failed(e1))
        await asyncio.sleep(0.2)
        box["p"].push(H.out_text("It opens at nine.", AHEAD + 2050, AHEAD + 2400))
        box["p"].push(H.out_audio(NR.pcm(1)))
        assert await NR._wait(lambda: NR._epoch(phone) != e1, 3.0)
        await asyncio.sleep(0.4)
        e2 = NR._epoch(phone)
        facts["parents"] = dict(_wire_parents(phone))
        facts["e2"] = e2
        phone.push_frame(NR._idle(e2))
        await asyncio.sleep(0.3)
        facts["rid"] = NR._finals(phone)[0]["turn_id"]
        phone.reask(facts["rid"])
        assert await NR._wait(lambda: phone.of("reask_result"))
        # R6-19: the window settles with no caller input → the watch's one
        # retry after `in_flight` is judged on the settled owner.
        assert await NR._wait(
            lambda: any(f.get("settled") for f in phone.of("output_attribution")), 8.0,
        )
        await asyncio.sleep(0.1)
        phone.reask(facts["rid"])
        assert await NR._wait(lambda: len(phone.of("reask_result")) >= 2)

    phone, provider = await _c12_socket(monkeypatch, script, db=R6._db("c12-late-ctl"),
                                        psid=R6._psid("c12-late-ctl"))
    # R6-13 PIN CHANGE: released at its bound, unsettled → no WIRE parent
    # (was: R); relay-side it keeps R's parent and R is answered (below) —
    # the genuine-continuation control of R6-13 / R6-15.
    assert facts["parents"].get(facts["e2"]) == "", facts["parents"]
    # R2 addendum 6 R6-19 PIN CHANGE (was: `answered` at the first reask): a
    # heard continuation whose attribution to R is still PENDING is no firm
    # answer yet — `in_flight` (asks the model nothing), then `answered` once
    # the window settles on R.  Genuine-heard suppression holds: nothing is
    # asked twice (same control as test_live_r619's settled-unspoken case).
    first, settled = phone.of("reask_result")[-2:]
    assert (first["user_turn_id"], first["outcome"]) == (facts["rid"], "in_flight"), first
    assert (settled["user_turn_id"], settled["outcome"]) == (facts["rid"], "answered"), settled
    assert NR._reask_contents(provider) == [], NR._reask_contents(provider)
    assert len(NR._finals(phone)) == 1
