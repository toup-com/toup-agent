"""R2 round 6b — the relay's media follow-up (addendum 6 R6-10 "Relay media":
M1-M6 + the Option A residual).

Binding spec: /private/tmp/toup-voice-r2-spec-addendum-6.md (R6-10, with
R6-1's classifier, R6-4/R6-4b, R6-5, R6-6), contract v0.3 §7, addendum 4 §6.1
("a play NEWER than the anchor is never evidence").  Findings: the media
verifier's r6-round-aa55bcf6f66e035b6.json and the supervisor's consent.xml
(3 FAIL / 1 PASS on fx7).

  M1  the B1 media-tail/continuer class answers ONLY a question that restates
      a relay-owned media command.  Every other question: a consent is the C2
      verdict (+ its closed politeness); a WH/choice answer ('now', 'off',
      'uh, now', «الان», «خاموش») keeps the caller's words — card and
      display_request — and is never told it "agreed to what it proposed".
  M2  an ARMED command whose restating question the caller refuses is
      DISARMED: no tenant stop now, at the grace end or on new evidence; the
      refusal runs with the truthful context that the stop was NOT carried
      out.  The FIRED case stays distinct: one stop (already sent) + the
      truthful "already carried out" context.  An armed ASSENT still continues
      the command (media-2b).  The order gate never fires a halt because of a
      delegation that answers that very command.
  M3  a refusal is the OPPOSITE polarity to the restated command by the ONE
      R6-1 classifier (`read_confirmation_answer`) — no first-word list.
  M4  non-media ledger traffic never evicts media evidence: an announced play
      is still a stop's evidence after 40 unrelated tasks (N = 30/31/32/40).
  M5  a play NEWER than the stop's anchor is never its evidence — also when
      the phone reports it: no tenant call, no line (a §7 tenant and an
      arrival-keyed tenant alike).
  M6  a consent title strips ONLY the closed interrogative frame at the clause
      edge; the proposal's words stay; a question that proposes nothing keeps
      the caller's words.
  Option A  the in-process agent path (`_think` with `_agent_runner`, no HTTP)
      binds the run's ordered media halt mark like the tenant endpoints do.

WIRE LEVEL: the real relay, the real `ws_realtime` request builders, and the
fake §7 TENANT of `test_live_round6_media` (records every request body).  Every
socket has its own FakeProvider session id.  Old-fail: this file run against a
copy of fx4-relay-snapG (lvp 94a9a1a0…).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import test_live_harness as H
import test_live_round6_media as M
from app.api import ws_realtime as rt
from app.services import live_voice_protocol as live


def _fa(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in text)


def _results(provider, delegation_id):
    return [
        str(e.get("content") or "") for e in M._thinking(provider)
        if e.get("delegation_id") == delegation_id
    ]


# ══════════════════════════════════════════════════════════════════════
# PURE — the structural rules
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("answer", [
    "now", "off", "it", "please", "uh, now", "دیگه", "کن", "الان", "خاموش", "no", "نه",
    "yes, what did you find?",
])
def test_m1_a_bare_tail_or_wh_answer_is_not_a_consent(answer):
    assert not live.answer_consents(answer), answer


@pytest.mark.parametrize("answer", [
    "yes", "yes please", "sure, go ahead", "go ahead please", "آره", "آره لطفا", "بله حتما",
    "sounds good", "ادامه بده", "okay",
])
def test_m1_control_real_consents_stay_consents(answer):
    assert live.answer_consents(answer), answer


@pytest.mark.parametrize("answer", ["now", "off", "it", "please", "کن", "دیگه"])
def test_m1_the_media_tail_still_continues_a_restated_command(answer):
    # The B1 tail class is untouched — it is only no longer read as a consent
    # to OTHER questions.
    assert live.media_answer_continues(answer), answer


@pytest.mark.parametrize(("question", "subject"), [
    ("Do you want me to find songs like this one?", "find songs like this one"),
    ("Would you like me to book the table for us?", "book the table for us"),
    ("Should I check what the weather is like tomorrow?", "check what the weather is like tomorrow"),
    ("Do you want me to find a podcast instead?", "find a podcast instead"),
    ("Okay, do you still want me to look for hotels?", "look for hotels"),
    ("Want me to pause it?", "pause it"),
    ("How about some jazz instead?", "some jazz instead"),
    ("قطعش کنم؟", "قطعش"),
    ("آهنگ رو قطع کنم؟", "آهنگ رو قطع"),
    ("می‌خوای یه پادکست برات پخش کنم؟", "یه پادکست برات پخش"),
    ("می‌خوای برات یه آهنگ آروم پیدا کنم؟", "یه آهنگ آروم پیدا"),
])
def test_m6_only_the_clause_edge_frame_is_stripped(question, subject):
    assert live.question_subject(question) == subject


@pytest.mark.parametrize("question", [
    "Anything else?", "Is there anything else?", "Sound good?", "When should I remind you?",
    "What time should I set the alarm for?", "Should I turn the lights on or off?",
    "Can you hear me?", "چیز دیگه‌ای هم هست؟", "چراغ‌ها رو روشن کنم یا خاموش؟",
])
def test_m6_a_question_that_proposes_nothing_has_no_subject(question):
    assert live.question_subject(question) == "", question


@pytest.mark.parametrize(("question", "action"), [
    ("قطعش کنم؟", "stop"), ("آهنگ رو قطع کنم؟", "stop"), ("Should I stop the music?", "stop"),
    ("Do you want me to stop it?", "stop"), ("Do you want me to turn it off?", "stop"),
    ("Would you like me to pause the music?", "pause"), ("می‌خوای قطعش کنم؟", "stop"),
])
def test_m6_control_the_restatement_rule_is_unchanged(question, action):
    assert live.question_restates_command(question, action), question


@pytest.mark.parametrize(("answer", "reading"), [
    # refusal = the OPPOSITE polarity by the ONE R6-1 classifier (no first-word list)
    ("نه", "refuse"), ("no, don't", "refuse"), ("keep it playing", "refuse"),
    ("please don't", "refuse"), ("don't stop it", "refuse"), ("بذار باشه", "refuse"),
    ("قطعش نکن", "refuse"), ("no, keep it on", "refuse"),
    # PIN UPDATED (round 7, cross-fork): 'leave it on' was "unclear" here; the
    # MERGED R6-1 classifier (snapH, R6-10 M3's own example list) reads the
    # English keep verb 'leave' + 'on' as KEEP polarity — the opposite of the
    # restated stop — so it is a REFUSAL.  Intended reading; the relay's
    # behaviour is unchanged (an armed refusal and an unclear answer both
    # disarm: 0 stops, 'NOT carried out').
    ("leave it on", "refuse"),
    # hedge / no polarity — never a confirmation (the relay never acts on it)
    ("wait, don't", "unclear"), ("actually no, leave it on", "unclear"),
    ("yes but wait", "unclear"),
    # PIN UPDATED (round 7, R6-15 F1 BINDING): a new request instead of an
    # answer is NOT an answer ("none", was "unclear"): it gets normal handling
    # and the confirmation expires unconfirmed — still never a confirmation.
    ("play some jazz instead", "none"),
    # continuation (media-2 / media-2b) and confirmation
    ("yes please", "continue"), ("آره", "continue"), ("کن لطفا", "continue"),
    ("yes, stop it", "confirm"), ("آره قطعش کن", "confirm"), ("turn it off", "confirm"),
    # PIN UPDATED (round 7, R6-14 F2): «ولش کن» was "confirm" (R6-1 reads
    # «ولش» as STOP polarity, "drop it").  As an answer to a stop/pause
    # CONFIRMATION a dismissal idiom that doubles as a stop verb is AMBIGUOUS
    # (leave-it vs drop-it): never fires an armed halt, never absorbed.  The
    # classifier's own reading is unchanged (the §5.3 task-stop question keeps
    # it); the ambiguity is the media confirmation's.
    ("ولش کن", "unclear"), ("drop it", "unclear"), ("forget it", "unclear"),
    ("بی‌خیال", "unclear"),
])
def test_m3_the_answer_is_read_by_the_one_classifier(answer, reading):
    assert live.media_answer_reading(answer, "stop", "stop the music") == reading, answer


def test_m3_an_answer_the_classifier_cannot_place_is_never_acted_on():
    """PIN UPDATED (round 7): «نمی‌خوام» was pinned "unclear".  The ONE R6-1
    classifier reads it with no polarity and words of its own — for the §5.3
    question as for this one it is "not an answer" ("none", R6-15: normal
    handling, the confirmation expires unconfirmed); the classifier fork's
    negated-want structure (R6-16 P4) may give it stop polarity, which the
    media confirmation reads as "unclear" (R6-14 F2: no assent said).  Both
    are the pinned behaviour: never a confirmation, never a refusal claim —
    an armed command is disarmed (0 stops, 'did not confirm')."""

    reading = live.media_answer_reading("نمی‌خوام", "stop", "آهنگ رو قطع")
    assert reading in {"none", "unclear"}, reading


def test_m3_there_is_no_first_word_negation_list():
    assert not hasattr(live, "_MEDIA_ANSWER_NEGATIONS")
    # A refusal that does not START with a negation is still one; a request
    # that does is read by the classifier, not by its first word.
    assert live.media_answer_refuses("please don't")
    assert live.media_answer_refuses("keep it playing")


def _ledger(kinds):
    session = live._LiveSession.__new__(live._LiveSession)
    session.media_intents = {}
    for index, (kind, phase) in enumerate(kinds):
        session.media_intents[f"t{index}"] = live.MediaIntent(
            order=index + 1, kind=kind, path="agent", phase=phase, task_id=f"t{index}",
        )
        session.trim_media_intents()
    return session.media_intents


@pytest.mark.parametrize("runs", [30, 31, 32, 40])
def test_m4_runs_never_evict_an_announced_play(runs):
    intents = _ledger([("play", "announced")] + [("run", "done")] * runs)
    assert "t0" in intents and intents["t0"].phase == "announced"
    assert sum(1 for i in intents.values() if i.kind == "run") == min(runs, live._MEDIA_INTENTS_MAX)


def test_m4_halts_never_evict_plays_and_settled_intents_go_first():
    intents = _ledger(
        [("play", "announced")] + [("stop", "answered")] * 40 + [("run", "running")]
        + [("run", "done")] * 40,
    )
    assert "t0" in intents
    assert intents["t41"].kind == "run" and intents["t41"].phase == "running"   # live run kept


# ══════════════════════════════════════════════════════════════════════
# WIRE — M1 / M6: the caller's own answer vs a consent to a proposal
# ══════════════════════════════════════════════════════════════════════

async def _answer(monkeypatch, question, answer, lead="I have a dentist appointment tomorrow"):
    tenant = M.Tenant(agents={"": {"think_s": 0.1}})

    async def steps(p, client):
        p.push(H.user_delta(lead, 1000, 1600))
        p.push(H.out_text(question, 1700, 2400))
        p.push(H.out_audio("Q"))
        await asyncio.sleep(0.6)
        p.push(H.user_delta(answer, 3000, 3300))
        p.push(H.delegation("d-ans", 3350))
        await M._wait(lambda: tenant.runs(), 4.0)
        await asyncio.sleep(0.5)

    client, provider = await M._call(monkeypatch, tenant, steps)
    created = [f for f in M._frames_for(client, "d-ans") if f.get("phase") == "created"]
    return created, tenant.runs()


@pytest.mark.asyncio
@pytest.mark.parametrize(("question", "answer"), [
    ("When should I remind you?", "now"),
    ("Should I turn the lights on or off?", "off"),
    ("What time should I set the alarm for?", "uh, now"),
    ("کی یادت بندازم؟", "دیگه"),               # a B1 tail word only (old: "agreed")
    ("کی یادت بندازم؟", "الان"),                # controls: never tail words
    ("چراغ‌ها رو روشن کنم یا خاموش؟", "خاموش"),
])
async def test_m1_a_wh_or_choice_answer_keeps_the_callers_words(monkeypatch, question, answer):
    created, runs = await _answer(monkeypatch, question, answer)
    assert created and runs, (created, runs)
    assert created[0]["title"].strip() == answer, created[0]["title"]
    assert runs[0]["display_request"].strip() == answer, runs[0]["display_request"]
    assert "agree to what it proposed" not in runs[0]["message"], runs[0]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("question", "answer", "kept"), [
    ("Do you want me to find a podcast instead?", "yes please", "find a podcast instead"),
    ("Do you want me to find songs like this one?", "yes", "find songs like this one"),
    ("Would you like me to book the table for us?", "yes please", "book the table for us"),
    ("Should I check what the weather is like tomorrow?", "yes", "check what the weather is like tomorrow"),
    ("می‌خوای برات یه آهنگ آروم پیدا کنم؟", "آره لطفا", "یه آهنگ آروم پیدا"),
])
async def test_m6_a_consent_is_titled_from_the_whole_proposal(monkeypatch, question, answer, kept):
    created, runs = await _answer(monkeypatch, question, answer)
    assert created and runs
    assert created[0]["title"] == kept, created[0]["title"]
    assert runs[0]["display_request"] == kept, runs[0]["display_request"]
    assert "agree to what it proposed" in runs[0]["message"], runs[0]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("question", "answer"), [
    ("Anything else?", "yes"),
    ("چیز دیگه‌ای هم هست؟", "آره"),
])
async def test_m6_a_consent_to_a_question_that_proposes_nothing_keeps_the_callers_words(
    monkeypatch, question, answer,
):
    created, runs = await _answer(monkeypatch, question, answer, lead="thanks for the list")
    assert created and runs
    assert created[0]["title"].strip() == answer, created[0]["title"]
    assert "agree to what it proposed" not in runs[0]["message"], runs[0]["message"]


# ══════════════════════════════════════════════════════════════════════
# WIRE — M2 / M3: a refusal of the question that restates the command
# ══════════════════════════════════════════════════════════════════════

ARMED_WAIT, ARMED_GRACE = 0.3, 1500     # the answer and its delegation inside the grace
FIRED_WAIT, FIRED_GRACE = 0.9, 300      # the backstop already fired


async def _restated(monkeypatch, command, question, answer, *, wait_s, grace_ms,
                    delegate=True, late_evidence=False, tail_s=2.4, tenant=None):
    tenant = tenant or M.Tenant(device=M._old_song(), agents={"": {"think_s": 0.1}})

    async def steps(p, client):
        p.push(H.user_delta(command, 1000, 1600))
        p.push(H.out_text(question, 1700, 2300))
        p.push(H.out_audio("Q"))
        await asyncio.sleep(wait_s)
        p.push(H.user_delta(answer, 3000, 3300))
        p.push(H.out_text("باشه." if _fa(answer) else "Okay.", 3400, 3600))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.6)
        if delegate:
            p.push(H.delegation("d-late", 3700))
        await asyncio.sleep(tail_s)
        if late_evidence:
            # New playback evidence AFTER the disarm, then well past the grace.
            M._send(client, {"type": "now_playing", "title": "Fadat Sham"})
            await asyncio.sleep(grace_ms / 1000.0 + 1.0)

    client, provider = await M._call(
        monkeypatch, tenant, steps, grace_ms=grace_ms,
        frames=({"type": "now_playing", "title": "Fadat Sham"},), timeout=30,
    )
    return client, provider, tenant


REFUSALS = [
    ("آهنگ رو قطع", "قطعش کنم؟", "نه"),
    ("stop the music", "Should I stop the music?", "no, don't"),
    ("stop the music", "Should I stop the music?", "wait, don't"),
    ("stop the music", "Should I stop the music?", "actually no, leave it on"),
    ("stop the music", "Should I stop the music?", "leave it on"),
    ("stop the music", "Do you want me to turn it off?", "keep it playing"),
    ("آهنگ رو قطع", "قطعش کنم؟", "نمی‌خوام"),
    ("آهنگ رو قطع", "آهنگ رو قطع کنم؟", "نه، بذار باشه"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), REFUSALS)
async def test_m2_an_armed_refusal_disarms_the_stop(monkeypatch, command, question, answer):
    """The supervisor's oracle: armed + refusal of the restating question →
    the refusal DISARMS the halt: 0 tenant stops now, at the grace end and on
    new evidence; it runs as the caller's request with the truthful context
    that the stop was NOT carried out."""

    client, provider, tenant = await _restated(
        monkeypatch, command, question, answer, wait_s=ARMED_WAIT, grace_ms=ARMED_GRACE,
        late_evidence=True,
    )
    assert tenant.controls("stop") == [], tenant.events
    assert tenant.device is not None and tenant.device["title"] == "Fadat Sham", tenant.events
    runs = tenant.runs()
    assert len(runs) == 1, tenant.bodies                  # never absorbed: it runs
    message = runs[0]["message"]
    assert "NOT carried that command out" in message and command in message, message
    assert "already carried out" not in message, message
    assert runs[0]["display_request"].strip() == answer   # the caller's own words
    assert not any("belong to their playback command" in r for r in _results(provider, "d-late"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("آهنگ رو قطع", "قطعش کنم؟", "نه"),
    ("stop the music", "Should I stop the music?", "no, don't"),
])
async def test_m2_an_armed_refusal_disarms_even_with_no_delegation(monkeypatch, command, question, answer):
    """The disarm follows the caller's ANSWER, not the model: with no
    delegation at all, the grace end and new evidence still never stop."""

    client, provider, tenant = await _restated(
        monkeypatch, command, question, answer, wait_s=ARMED_WAIT, grace_ms=ARMED_GRACE,
        delegate=False, late_evidence=True,
    )
    assert tenant.controls("stop") == [], tenant.events
    assert tenant.runs() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), REFUSALS)
async def test_m2_a_fired_refusal_learns_the_stop_already_ran(monkeypatch, command, question, answer):
    """The FIRED case stays distinct: the one stop was already sent; the
    refusal runs (never absorbed, never a second stop) and is told truthfully
    that the command already happened (it may offer to play again)."""

    client, provider, tenant = await _restated(
        monkeypatch, command, question, answer, wait_s=FIRED_WAIT, grace_ms=FIRED_GRACE,
    )
    assert len(tenant.controls("stop")) == 1, tenant.bodies
    assert ("media_stop", "Fadat Sham") in tenant.events
    runs = tenant.runs()
    assert len(runs) == 1, tenant.bodies
    message = runs[0]["message"]
    assert "already carried out" in message and command in message, message
    assert "play it again" in message and "NOT carried" not in message, message


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("stop the music", "Should I stop the music?", "yes please"),
    ("آهنگ رو قطع", "قطعش کنم؟", "آره"),
    ("آهنگ رو قطع", "آهنگ رو قطع کنم؟", "کن لطفا"),
])
async def test_m2_control_an_armed_assent_still_continues_the_command(monkeypatch, command, question, answer):
    """media-2b kept: the assent joins the ARMED unit and its delegation binds
    to it — exactly one stop, no run, no card, the absorbed result names it."""

    client, provider, tenant = await _restated(
        monkeypatch, command, question, answer, wait_s=ARMED_WAIT, grace_ms=ARMED_GRACE,
    )
    assert len(tenant.controls("stop")) == 1, tenant.bodies
    assert tenant.runs() == [], tenant.runs()
    assert M._frames_for(client, "d-late") == []
    results = _results(provider, "d-late")
    assert results and "already carried out" in results[0], results


@pytest.mark.asyncio
async def test_m2_the_order_gate_never_fires_a_halt_for_the_delegation_that_answers_it(monkeypatch):
    """A CONFIRMATION with a request of its own ('yes, and what's the weather
    in Paris?') leaves the command armed; its own delegation does not fire the
    stop early (M2) — the stop runs once, at its grace end, AFTER the run's
    request; the run is told the app carries the command out itself."""

    client, provider, tenant = await _restated(
        monkeypatch, "stop the music", "Should I stop the music?",
        "yes, and what's the weather in Paris?", wait_s=ARMED_WAIT, grace_ms=ARMED_GRACE,
    )
    assert len(tenant.controls("stop")) == 1, tenant.bodies
    paths = [path for path, _body in tenant.bodies]
    run_at = next(i for i, p in enumerate(paths) if p.endswith("/internal/agent-turn/stream"))
    stop_at = next(i for i, p in enumerate(paths) if p.endswith("/internal/media-control"))
    assert run_at < stop_at, paths
    assert "confirm their playback command" in tenant.runs()[0]["message"]


# ══════════════════════════════════════════════════════════════════════
# WIRE — M4: media evidence survives unrelated traffic
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_m4_an_announced_play_is_still_evidence_after_40_unrelated_tasks(monkeypatch):
    tenant = M.Tenant(searches={"jazz": (0.1, True)}, agents={"": {"think_s": 0.02}})
    count = 40

    async def steps(p, client):
        p.push(H.user_delta("play some jazz", 1000, 1500))
        p.push(H.delegation("dJazz", 1520))
        await M._wait(lambda: "media_play" in tenant.kinds(), 4.0)
        await asyncio.sleep(0.3)
        t = 3000
        for i in range(count):
            p.push(H.user_delta(f"what is the capital of country number {i + 1}", t, t + 600))
            p.push(H.delegation(f"dQ{i}", t + 620))
            await M._wait(lambda: len(tenant.runs()) >= i + 1, 3.0)
            await asyncio.sleep(0.12)
            t += 2500
        await asyncio.sleep(0.3)
        p.push(H.user_delta("stop the music", t, t + 500))
        p.push(H.out_text("Okay.", t + 600, t + 800))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.5)

    client, provider = await M._call(monkeypatch, tenant, steps, grace_ms=300, timeout=120)
    assert len(tenant.runs()) == count
    assert len(tenant.controls("stop")) == 1, tenant.events[-6:]
    assert ("media_stop", "some jazz") in tenant.events


# ══════════════════════════════════════════════════════════════════════
# WIRE — M5: a NEWER play is never the stop's evidence
# ══════════════════════════════════════════════════════════════════════

class ArrivalTenant(M.Tenant):
    """A tenant that predates contract §7: arrival marks only."""

    def key(self, body, field):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_cls", [M.Tenant, ArrivalTenant])
@pytest.mark.parametrize(("stop", "ack", "play", "needle"), [
    ("stop the music", "Okay.", "play some jazz", "jazz"),
    ("آهنگ رو قطع کن", "چشم.", "یه آهنگ جاز پخش کن", "جاز"),
])
async def test_m5_the_newer_plays_own_playback_is_never_evidence(
    monkeypatch, tenant_cls, stop, ack, play, needle,
):
    """Nothing plays; the stop is only acknowledged; a NEWER play starts
    inside the stop's grace and the phone reports it.  No tenant stop, no line
    about it — the newest request keeps playing."""

    tenant = tenant_cls(searches={needle: (0.1, True)})

    async def steps(p, client):
        p.push(H.user_delta(stop, 1000, 1500))
        p.push(H.out_text(ack, 1600, 1800))
        p.push(H.out_audio("OK"))
        await M._acknowledged(client)
        p.push(H.user_delta(play, 2000, 2500))
        p.push(H.delegation("dPlay", 2520))
        await M._wait(lambda: "media_play" in tenant.kinds(), 3.0)
        M._send(client, {"type": "now_playing", "title": tenant.device["title"]})
        await asyncio.sleep(2.5)

    client, provider = await M._call(monkeypatch, tenant, steps, grace_ms=1500)
    said = M._said(provider)
    assert tenant.device is not None and needle in tenant.device["title"], tenant.events
    assert tenant.controls("stop") == [], tenant.events
    assert not any("left it on" in s or "Stopped" in s or "قطع" in s for s in said), said


@pytest.mark.asyncio
async def test_m5_control_an_older_item_on_the_phone_is_still_evidence(monkeypatch):
    """The phone reports an item that is NOT a newer relay play: the stop has
    evidence, fires before the newer play's request, and the newer play wins."""

    tenant = M.Tenant(searches={"jazz": (0.1, True)}, device=M._old_song())

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 1000, 1500))
        p.push(H.out_text("Okay.", 1600, 1800))
        p.push(H.out_audio("OK"))
        await M._acknowledged(client)
        p.push(H.user_delta("play some jazz", 2000, 2500))
        p.push(H.delegation("dPlay", 2520))
        await M._wait(lambda: "media_play" in tenant.kinds(), 3.0)
        await asyncio.sleep(2.0)

    client, provider = await M._call(
        monkeypatch, tenant, steps, grace_ms=1500,
        frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    assert len(tenant.controls("stop")) == 1, tenant.events
    assert tenant.kinds().index("control") < tenant.kinds().index("play_requested"), tenant.events
    assert tenant.device is not None and tenant.device["title"] == "some jazz", tenant.events


# ══════════════════════════════════════════════════════════════════════
# WIRE — Option A: the in-process agent run carries the caller's order
# ══════════════════════════════════════════════════════════════════════

class _InProcessRunner:
    """An in-process `_agent_runner`: records the media halt mark bound for
    the run it is asked to make (what its play_media tool would check)."""

    def __init__(self):
        self.marks: list = []

    async def run(self, **kwargs):
        from app.agent.radio import control

        self.marks.append(control.run_halt_mark(kwargs.get("user_id") or ""))
        return SimpleNamespace(
            text="Here is what I found.", model="m", processing_time_ms=1,
            tokens_input=0, tokens_output=0, persisted={},
        )


@pytest.mark.asyncio
async def test_option_a_the_in_process_run_binds_the_callers_order(monkeypatch):
    tenant = M.Tenant()
    runner = _InProcessRunner()

    async def steps(p, client):
        monkeypatch.setattr(rt, "_agent_runner", runner)
        p.push(H.user_delta("what is the capital of france", 1000, 1600))
        p.push(H.delegation("dA", 1620))
        await M._wait(lambda: runner.marks, 4.0)
        await asyncio.sleep(0.3)

    client, provider = await M._call(monkeypatch, tenant, steps)
    assert runner.marks, "the in-process runner never ran"
    mark = runner.marks[0]
    assert mark is not None and mark.ordered, mark
    created = [f for f in M._frames_for(client, "dA") if f.get("phase") == "created"]
    turn = str(created[0].get("parent_user_turn_id") or created[0].get("user_turn_id") or "")
    if turn.rsplit(":", 1)[-1].isdigit():
        assert mark.media_order == int(turn.rsplit(":", 1)[-1]), (mark, turn)
    assert mark.media_scope == M._psid(provider)
    assert mark.media_scope_started_ms == M._stamp(client, provider) > 0
    assert tenant.runs() == []                      # no HTTP run: Option A only


@pytest.mark.asyncio
async def test_option_a_control_outside_the_relay_nothing_is_bound(monkeypatch):
    runner = _InProcessRunner()
    monkeypatch.setattr(rt, "_agent_runner", runner)
    text, _model = await M.REAL_THINK("user-option-a", "what is the capital of france", None)
    assert text == "Here is what I found."
    assert runner.marks == [None]


# ══════════════════════════════════════════════════════════════════════
# WIRE — supervisor 05:24 (app verifier a7ba9d2e): a `newer_playing` halt is
# never "Stopped" on ANY client.  Every shipped client maps a `cancelled`
# completion to "Stopped" and TF132 never gets the correlated verdict, so the
# control step ends `not_run` (both builds drop it).  The paused truth rides on
# the negotiated verdict frame (`paused: true`, only for `media_transport`).
# ══════════════════════════════════════════════════════════════════════

class _FailingTenant(M.Tenant):
    async def media_control(self, body):
        self.seq += 1
        self.events.append(("control", str(body.get("action") or "")))
        return {"ok": False, "reason": "delivery_failed", "acked_devices": 0, "silent_devices": 0}


async def _halt(monkeypatch, words, *, features, device, paused=False, ack="", tenant=None):
    tenant = tenant or M.Tenant()

    async def steps(p, client):
        if device == "newer":
            tenant.first_seen.setdefault("live-psid-newer-scope", 99)
            tenant.device = {
                "title": "Jazz Mix", "video_id": "JAZZMIX0001",
                "key": ("live-psid-newer-scope", 2 ** 52, 1), "paused": paused,
            }
        else:
            tenant.device = {"title": "Fadat Sham", "video_id": "OLDSONG0001", "key": None, "paused": False}
        p.push(H.user_delta(words, 1000, 1600))
        if ack:
            p.push(H.out_text(ack, 1700, 1900))
            p.push(H.out_audio("OK"))
        else:
            p.push(H.delegation("dStop", 1620))
        await M._wait(lambda: tenant.of("/internal/media-control"), 4.0)
        await asyncio.sleep(1.2)

    client, provider = await M._call(
        monkeypatch, tenant, steps, grace_ms=300, features=features,
        frames=({"type": "now_playing", "title": "Jazz Mix"},),
    )
    completions = [f for f in client.of("tool_call.completed") if f.get("name") == "media_control"]
    verdicts = [f for f in client.of("media_control") if f.get("revision") == 2]
    return client, provider, tenant, completions, verdicts


NEWER_HALTS = [
    ("stop the music", False), ("stop the music", True), ("آهنگ رو قطع کن", False),
    ("آهنگ رو قطع کن", True), ("pause the music", False), ("آهنگ رو نگه دار", False),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("words", "paused"), NEWER_HALTS)
async def test_np_tf132_a_newer_playing_halt_is_never_a_stopped_step(monkeypatch, words, paused):
    client, provider, tenant, completions, verdicts = await _halt(
        monkeypatch, words, features=M.TF132, device="newer", paused=paused,
    )
    assert ("newer_playing", "Jazz Mix") in tenant.events, tenant.events
    assert [(f.get("ok"), f.get("outcome")) for f in completions] == [(False, "not_run")], completions
    assert client.of("media_control") == []           # TF132: no stop/pause frames, as before
    assert "completed" in M._phases(client, "dStop")
    said = M._said(provider)
    lang = "fa" if _fa(words) else "en"
    key = "media_newer_paused" if paused else "media_newer_playing"
    assert any(live.relay_line(key, lang, title="Jazz Mix") in s for s in said), said


@pytest.mark.asyncio
@pytest.mark.parametrize(("words", "paused"), NEWER_HALTS)
async def test_np_v03_not_run_and_the_paused_truth_on_the_verdict(monkeypatch, words, paused):
    client, provider, tenant, completions, verdicts = await _halt(
        monkeypatch, words, features=M.V03_FLOOR, device="newer", paused=paused,
    )
    assert [(f.get("ok"), f.get("outcome")) for f in completions] == [(False, "not_run")], completions
    assert len(verdicts) == 1 and verdicts[0]["reason"] == "newer_playing", verdicts
    assert verdicts[0]["status"] == "error"                 # never `executed`
    if paused:
        assert verdicts[0].get("paused") is True, verdicts
    else:
        assert "paused" not in verdicts[0], verdicts


@pytest.mark.asyncio
@pytest.mark.parametrize(("words", "ack", "paused"), [
    ("stop the music", "Okay.", False), ("آهنگ رو قطع کن", "چشم.", True),
])
async def test_np_backstop_verdict_carries_the_paused_truth(monkeypatch, words, ack, paused):
    client, provider, tenant, completions, verdicts = await _halt(
        monkeypatch, words, features=M.V03_FLOOR, device="newer", paused=paused, ack=ack,
    )
    assert completions == []                                 # the backstop has no step
    assert len(verdicts) == 1 and verdicts[0]["reason"] == "newer_playing", verdicts
    assert (verdicts[0].get("paused") is True) == paused, verdicts


@pytest.mark.asyncio
@pytest.mark.parametrize("features", ["TF132", "V03"])
@pytest.mark.parametrize("words", ["stop the music", "آهنگ رو قطع کن"])
async def test_np_control_a_confirmed_stop_is_unchanged(monkeypatch, features, words):
    client, provider, tenant, completions, verdicts = await _halt(
        monkeypatch, words, features=M.TF132 if features == "TF132" else M.V03_FLOOR, device="older",
    )
    assert ("media_stop", "Fadat Sham") in tenant.events
    assert [(f.get("ok"), f.get("outcome")) for f in completions] == [(True, "ok")], completions
    if features == "V03":
        assert [(v["status"], v["reason"]) for v in verdicts] == [("executed", "stopped")]
        assert "paused" not in verdicts[0]
    lang = "fa" if _fa(words) else "en"
    assert any(live.relay_line("media_stopped", lang) in s for s in M._said(provider))


@pytest.mark.asyncio
@pytest.mark.parametrize("features", ["TF132", "V03"])
@pytest.mark.parametrize("words", ["stop the music", "آهنگ رو قطع کن"])
async def test_np_control_a_failed_stop_still_reads_as_a_failure(monkeypatch, features, words):
    client, provider, tenant, completions, verdicts = await _halt(
        monkeypatch, words, features=M.TF132 if features == "TF132" else M.V03_FLOOR,
        device="older", tenant=_FailingTenant(),
    )
    assert [(f.get("ok"), f.get("outcome")) for f in completions] == [(False, "error")], completions
    if features == "V03":
        assert [(v["status"], v["reason"]) for v in verdicts] == [("error", "delivery_failed")]
        assert "paused" not in verdicts[0]


def test_np_media_verdict_extras_only_for_a_paused_newer_item():
    assert live.media_verdict_extras({"paused": True}, newer=True) == {"paused": True}
    assert live.media_verdict_extras({"paused": True}, newer=False) == {}
    assert live.media_verdict_extras({"paused": "yes"}, newer=True) == {}
    assert live.media_verdict_extras({}, newer=True) == {}
