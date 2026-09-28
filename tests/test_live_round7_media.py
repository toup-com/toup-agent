"""R2 round 7 — the relay's media confirmation binding (addendum 6 R6-14 F1,
F2, F3, F5 with the R6-15 F1 BINDING).

Binding spec: /private/tmp/toup-voice-r2-spec-addendum-6.md (R6-14, R6-15;
R6-10 M2/M3/M5 kept), addendum 4 §5.3 (heard / order / expiry / newer
question), §6.1 (a play NEWER than the anchor is never evidence), contract
v0.3 (TF132 gets no new frame, enum or field).  Findings: the fx9 media
verifier (`/private/tmp/toup-voice-r2-review/probes/r6b-verify-media2.Dri1YA`,
results `…/verify-r3.9uCiCM/r6b-media2-*.json`).

  F1  While a stop/pause is ARMED, the agent's CONFIRMATION of it binds the
      caller's answer — through its causal context only (R6-15): the agent's
      REPLY to the command (an epoch parented to the command's turn chain)
      that asks nothing of its own beyond the command ('Are you sure?',
      «مطمئنی؟», «قطع کنم؟», 'Should I go ahead and stop the music?', 'Just to
      confirm, …'), heard (UNKNOWN keeps the legacy rule), in order,
      unexpired, no newer question since.  Refusal → disarm (0 stops, 'NOT
      carried out'); a reply that offers something BESIDE it ('… Or just pause
      it?') makes a bare yes/no AMBIGUOUS → never acted on.  A question with a
      proposal of its own ('Should I search for restaurants?') binds nothing:
      the stop proceeds under its own rules.  Mixed answer → the refusal part
      applies, the request runs as its own card; a new request instead of an
      answer → normal handling, the confirmation expires unconfirmed; a newer
      agent question voids the binding.
  F2  An answer to such a question is never swallowed by the idle-dismissal
      (cancel_idle) branch: 'never mind', 'forget it', «بی‌خیال», «ولش کن» run
      with truthful context (fired → 'already carried out'; armed → 0 stops +
      'NOT carried out').  Dismissal idioms that double as a stop verb
      («ولش کن», 'drop it', 'let it go') are AMBIGUOUS as answers to a
      stop/pause confirmation: never fire an armed halt, never absorbed.
  F3  Evidence identity is sequence-based: an item the phone FIRST reports
      after a newer media request was dispatched to the device path (a fast
      play at dispatch, an agent play at its `play_media` tool start) is
      never an older halt's evidence unless it is known to predate it (an
      older relay play's title or resolved title).
  F5  An answer to a confirmation the caller provably did NOT hear (UNHEARD
      receipt) is not an answer to it: no disarm, never told 'declined'.

WIRE LEVEL: the real relay, the real `ws_realtime` request builders and the
fake §7 TENANT of `test_live_round6_media` (it records every request body).
Every socket has its own FakeProvider session id.  Old-fail: this file run
against a copy of fx4-relay-snapH (lvp 5ce72b7b…).
"""

from __future__ import annotations

import asyncio

import pytest

import test_live_harness as H
import test_live_round6_media as M
from app.config import settings
from app.services import live_voice_protocol as live

ARMED_WAIT, ARMED_GRACE = 0.3, 1500     # the answer and its delegation inside the grace
FIRED_WAIT, FIRED_GRACE = 0.9, 300      # the backstop already fired
NOT_CARRIED = "NOT carried that command out"
ALREADY = "already carried out"


def _fa(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in text)


async def _call(monkeypatch, tenant, steps, *, grace_ms=ARMED_GRACE, features=None,
                frames=(), timeout=30, on_commentary=None):
    """One socket (`M._call`, plus a hook for what the model says when the
    relay hands it a line or a result)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", grace_ms)
    H.patch_relay(monkeypatch)
    tenant.install(monkeypatch)
    box: dict = {}
    tag = M._tag("r7m")

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.commentary.append" and on_commentary is not None:
            on_commentary(p, e)
        if e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await M._wait(lambda: "p" in box and client.of("ready"), 4.0)
        await asyncio.sleep(0.1)
        await steps(box["p"], client)

    client = M.Phone([
        H.config(features=list(features or M.V03_FLOOR)), *frames, script, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, session_id=f"live-psid-{tag}")
    await H.run_relay(
        client, provider, timeout=timeout, db_session_id=f"db-{tag}", user_id=f"user-{tag}",
    )
    return client, provider


def _receipt(client, kind):
    eid = client.of("audio_delta")[-1].get("response_id")
    if kind == "unheard":
        return {"type": "interrupt", "response_id": eid, "item_id": eid, "played_ms": 0}
    return {"type": "playback_idle", "response_id": eid, "item_id": eid}


async def _confirm(monkeypatch, command, question, answer, *, wait_s=ARMED_WAIT,
                   grace_ms=ARMED_GRACE, receipt=None, delegate=True, answer_at=3000,
                   after=None, tenant=None, features=None, tail_s=2.4):
    """The caller's command, the agent's reply to it (`question`), the
    caller's `answer` and (optionally) the model's delegation of it."""

    tenant = tenant or M.Tenant(device=M._old_song(), agents={"": {"think_s": 0.1}})

    async def steps(p, client):
        p.push(H.user_delta(command, 1000, 1600))
        p.push(H.out_text(question, 1700, 2300))
        p.push(H.out_audio("Q"))
        if receipt is not None:
            await M._wait(lambda: client.of("audio_delta"), 2.0)
            await asyncio.sleep(0.1)
            M._send(client, _receipt(client, receipt))
            await asyncio.sleep(max(0.05, wait_s - 0.1))
        else:
            await asyncio.sleep(wait_s)
        p.push(H.user_delta(answer, answer_at, answer_at + 300))
        p.push(H.out_text("باشه." if _fa(answer) else "Okay.", answer_at + 400, answer_at + 600))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.6)
        if delegate:
            p.push(H.delegation("d-ans", answer_at + 700))
        await asyncio.sleep(tail_s)
        if after is not None:
            await after(p, client, tenant)

    client, provider = await _call(
        monkeypatch, tenant, steps, grace_ms=grace_ms, features=features,
        frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    return client, provider, tenant


async def _later_evidence_and_a_newer_request(p, client, tenant):
    """New playback evidence past every grace, then a NEWER request whose own
    delegation passes the order gate — a disarmed command never fires."""

    M._send(client, {"type": "now_playing", "title": "Fadat Sham"})
    await asyncio.sleep(2.0)
    p.push(H.user_delta("what is the capital of france", 9000, 9600))
    p.push(H.delegation("d-new", 9650))
    await M._wait(lambda: len(tenant.runs()) >= 2, 5.0)
    await asyncio.sleep(0.8)


def _halts(tenant) -> int:
    return len(tenant.controls("stop")) + len(tenant.controls("pause"))


def _run(tenant, index=0) -> dict:
    runs = tenant.runs()
    assert len(runs) > index, tenant.bodies
    return runs[index]


def _created(client, did="d-ans"):
    return [f for f in M._frames_for(client, did) if f.get("phase") == "created"]


def _no_task_spoken(provider) -> bool:
    said = M._said(provider)
    return any(
        live.relay_line("no_task", lang) in s for s in said for lang in ("en", "fa")
    )


# ══════════════════════════════════════════════════════════════════════
# PURE — the structural question reader (R6-15)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(("question", "action"), [
    ("Are you sure?", "stop"), ("مطمئنی؟", "stop"), ("قطع کنم؟", "stop"),
    ("Should I go ahead and stop the music?", "stop"),
    ("Just to confirm, should I stop the music?", "stop"),
    ("Do you want me to stop the music for you?", "stop"),
    ("You want me to stop the music?", "stop"), ("Did you say stop the music?", "stop"),
    ("قطعش کنم یا نه؟", "stop"), ("Should I stop the music? Yes or no?", "stop"),
    ("Are you sure you want me to pause it?", "pause"), ("می‌خوای آهنگ رو قطع کنم؟", "stop"),
])
def test_f1_a_question_that_asks_nothing_beyond_the_command_confirms_it(question, action):
    readings = [live.media_question_reading(s, action) for s in live.media_question_sentences(question)]
    assert readings and set(readings) == {"confirm"}, (question, readings)


@pytest.mark.parametrize(("question", "reading"), [
    ("Should I search for restaurants?", "other"),
    ("Want me to find a pharmacy?", "other"),
    ("Should I pause it instead?", "other"),
    ("Should I stop the search?", "other"),
    ("Anything else?", "other"),
    ("می‌خوای یه رستوران پیدا کنم؟", "other"),
    # the command's action AND another media action offered beside it
    ("Should I stop the music or just pause it?", "choice"),
    ("قطعش کنم یا فقط مکث کنم؟", "choice"),
])
def test_f1_a_question_with_a_proposal_of_its_own_is_never_a_confirmation(question, reading):
    readings = [live.media_question_reading(s, "stop") for s in live.media_question_sentences(question)]
    assert readings == [reading], (question, readings)


def test_f1_question_sentences_follow_r6_2():
    assert live.media_question_sentences("Okay. Should I stop the music? Or just pause it?") == [
        "Should I stop the music", "Or just pause it",
    ]
    assert live.media_question_sentences("I stopped «Why?» for you.") == []
    assert live.media_question_sentences("should I stop it") == ["should I stop it"]
    assert live.media_question_sentences("Will do.") == []


@pytest.mark.parametrize(("answer", "reading"), [
    # R6-14 F2: dismissal idioms / other stop verbs — leave-it vs drop-it
    ("ولش کن", "unclear"), ("ولش", "unclear"), ("ول کن", "unclear"), ("drop it", "unclear"),
    ("forget it", "unclear"), ("بی‌خیال", "unclear"), ("let it go", "unclear"),
    ("cancel it", "unclear"),
    # …with an assent said, or the command's own action, they confirm
    ("yes, drop it", "confirm"), ("آره ولش کن", "confirm"), ("stop it", "confirm"),
    # a dismissal the classifier reads as KEEP is a refusal
    ("never mind", "refuse"), ("leave it", "refuse"),
])
def test_f2_dismissal_idioms_as_answers_to_a_stop_confirmation(answer, reading):
    assert live.media_answer_reading(answer, "stop", "stop the music") == reading, answer


@pytest.mark.parametrize(("answer", "reading"), [
    ("no", "unclear"), ("نه", "unclear"), ("yes", "unclear"), ("آره", "unclear"),
    ("no, don't", "unclear"),
    # the answer names the command's own action: it is about THAT command
    ("no, don't stop it", "refuse"), ("نه قطعش نکن", "refuse"), ("yes, stop it", "confirm"),
])
def test_f1_a_bare_answer_to_a_reply_that_offered_something_else_is_unclear(answer, reading):
    assert live.media_answer_reading(answer, "stop", "stop the music", ambiguous=True) == reading


def test_f1_mixed_answer_keeps_the_callers_request():
    assert live.media_answer_parts("no, and find me a pharmacy", "stop", "stop the music") == (
        "refuse", "find me a pharmacy",
    )
    # a dismissal is one idiom, never a request ("never" + "mind")
    assert live.media_answer_parts("never mind", "stop", "stop the music") == ("refuse", "")


# ══════════════════════════════════════════════════════════════════════
# WIRE — F1: every confirmation framing disarms an armed stop on refusal
# ══════════════════════════════════════════════════════════════════════

F1_REFUSALS = [
    ("stop the music", "Are you sure?", "no"),
    ("stop the music", "Should I go ahead and stop the music?", "no"),
    ("stop the music", "Just to confirm, should I stop the music?", "no"),
    ("stop the music", "Do you want me to stop the music for you?", "no"),
    ("آهنگ رو قطع", "مطمئنی؟", "نه"),
    ("آهنگ رو قطع", "قطع کنم؟", "نه"),
    ("pause the music", "Are you sure?", "no, don't"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), F1_REFUSALS)
async def test_f1_an_armed_refusal_of_any_confirmation_never_stops(monkeypatch, command, question, answer):
    client, provider, tenant = await _confirm(
        monkeypatch, command, question, answer, after=_later_evidence_and_a_newer_request,
    )
    assert _halts(tenant) == 0, tenant.events
    assert tenant.device is not None and tenant.device["title"] == "Fadat Sham"
    run = _run(tenant)
    assert run["display_request"].strip() == answer
    assert NOT_CARRIED in run["message"] and "declined it" in run["message"], run["message"]
    assert question.rstrip("?؟") in run["message"], run["message"]
    assert ALREADY not in run["message"]
    assert len(tenant.runs()) == 2, tenant.bodies     # the newer request ran; still no halt


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("stop the music", "Are you sure?", "yes"),
    ("آهنگ رو قطع", "مطمئنی؟", "آره"),
    ("آهنگ رو قطع", "قطع کنم؟", "آره"),
])
async def test_f1_control_an_armed_assent_to_the_confirmation_is_one_stop(monkeypatch, command, question, answer):
    """media-2b: the assent joins the ARMED unit; its delegation binds to it —
    exactly one stop, no run, no card."""

    client, provider, tenant = await _confirm(monkeypatch, command, question, answer)
    assert len(tenant.controls("stop")) == 1, tenant.events
    assert tenant.runs() == [], tenant.runs()
    assert _created(client) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("stop the music", "Are you sure?", "yes"),
    ("آهنگ رو قطع", "مطمئنی؟", "آره"),
])
async def test_f1_control_a_fired_assent_to_the_confirmation_is_absorbed(monkeypatch, command, question, answer):
    """The FIRED case: the assent is the command's continuation (R6-5 media-2)
    — absorbed under the relay-media id, one stop, no run, no card."""

    client, provider, tenant = await _confirm(
        monkeypatch, command, question, answer, wait_s=FIRED_WAIT, grace_ms=FIRED_GRACE,
    )
    assert len(tenant.controls("stop")) == 1, tenant.events
    assert tenant.runs() == [], tenant.runs()
    assert _created(client) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("stop the music", "Are you sure?", "yes, stop it"),
    ("آهنگ رو قطع", "مطمئنی؟", "آره قطعش کن"),
])
async def test_f1_control_an_armed_confirmation_is_exactly_one_stop(monkeypatch, command, question, answer):
    client, provider, tenant = await _confirm(
        monkeypatch, command, question, answer, after=_later_evidence_and_a_newer_request,
    )
    assert len(tenant.controls("stop")) == 1, tenant.events


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("stop the music", "Should I search for restaurants?", "no"),
    ("stop the music", "Want me to find a pharmacy?", "no"),
    ("آهنگ رو قطع", "می‌خوای یه رستوران پیدا کنم؟", "نه"),
])
async def test_f1_an_unrelated_question_binds_nothing_and_the_stop_proceeds(monkeypatch, command, question, answer):
    """R6-15 control: 'no' answers THAT question — the stop is neither refused
    nor confirmed by it and proceeds under its own rules (evidence): one stop;
    the 'no' runs as the caller's own answer with no media context."""

    client, provider, tenant = await _confirm(monkeypatch, command, question, answer)
    assert len(tenant.controls("stop")) == 1, tenant.events
    assert ("media_stop", "Fadat Sham") in tenant.events
    run = _run(tenant)
    assert run["display_request"].strip() == answer
    assert NOT_CARRIED not in run["message"] and "declined" not in run["message"], run["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer", "remainder"), [
    ("stop the music", "Are you sure?", "no, and find me a pharmacy", "find me a pharmacy"),
    ("آهنگ رو قطع", "مطمئنی؟", "نه، یه داروخونه هم پیدا کن", "یه داروخونه هم پیدا کن"),
])
async def test_f1_a_mixed_answer_refuses_and_its_request_runs_as_its_own_card(
    monkeypatch, command, question, answer, remainder,
):
    client, provider, tenant = await _confirm(monkeypatch, command, question, answer)
    assert _halts(tenant) == 0, tenant.events
    assert len(tenant.runs()) == 1, tenant.bodies
    run = _run(tenant)
    assert run["display_request"].strip() == remainder, run["display_request"]
    created = _created(client)
    assert created and created[0]["title"].strip() == remainder, created
    assert NOT_CARRIED in run["message"], run["message"]


@pytest.mark.asyncio
async def test_f1_control_a_mixed_assent_confirms_and_its_request_runs(monkeypatch):
    client, provider, tenant = await _confirm(
        monkeypatch, "stop the music", "Are you sure?", "yes, and find me a pharmacy",
    )
    assert len(tenant.controls("stop")) == 1, tenant.events
    run = _run(tenant)
    assert run["display_request"].strip() == "find me a pharmacy", run["display_request"]
    assert "confirm their playback command" in run["message"], run["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "words", "later"), [
    ("stop the music", "Are you sure?", "what's the weather in Paris?", "no"),
    ("آهنگ رو قطع", "مطمئنی؟", "هوای پاریس چطوره؟", "نه"),
])
async def test_f1_a_new_request_instead_of_an_answer_expires_the_confirmation(
    monkeypatch, command, question, words, later,
):
    """R6-15: normal handling (its own card, words and title — never consumed
    or titled as an answer); the confirmation expires UNCONFIRMED (the
    command it asked about is never acted on); a later 'no' is no answer to
    it (the chain is broken by the request)."""

    async def after(p, client, tenant):
        p.push(H.user_delta(later, 7000, 7300))
        p.push(H.delegation("d-later", 7400))
        await M._wait(lambda: len(tenant.runs()) >= 2, 4.0)
        await asyncio.sleep(0.5)

    client, provider, tenant = await _confirm(
        monkeypatch, command, question, words, after=after,
    )
    assert _halts(tenant) == 0, tenant.events
    first = _run(tenant)
    assert first["display_request"].strip() == words, first["display_request"]
    created = _created(client)
    assert created and created[0]["title"].strip() == words, created
    assert NOT_CARRIED in first["message"] and "declined" not in first["message"], first["message"]
    second = _run(tenant, 1)
    assert second["display_request"].strip() == later
    assert NOT_CARRIED not in second["message"] and ALREADY not in second["message"]


async def _newer_question(monkeypatch, result_text):
    """A research job finishes while the stop is armed and its result is read
    out AFTER the agent's confirmation and BEFORE the caller's answer."""

    tenant = M.Tenant(device=M._old_song(), agents={
        "pharmac": {"think_s": 0.6, "text": "I found three pharmacies."}, "": {"think_s": 0.1},
    })
    spoken = {"n": 0}

    def on_commentary(p, e):
        if "pharmac" in str(e.get("content") or "").lower() and not spoken["n"]:
            spoken["n"] = 1
            p.push(H.out_text(result_text, 3600, 4200))
            p.push(H.out_audio("R"))

    async def steps(p, client):
        p.push(H.user_delta("find me a pharmacy near me", 1000, 1600))
        p.push(H.delegation("dPh", 1620))
        p.push(H.out_text("On it.", 1700, 1900))
        p.push(H.out_audio("A"))
        await asyncio.sleep(0.3)
        p.push(H.user_delta("stop the music", 2500, 3000))
        p.push(H.out_text("Are you sure?", 3100, 3400))
        p.push(H.out_audio("Q"))
        await M._wait(lambda: spoken["n"], 3.0)
        await asyncio.sleep(0.4)
        p.push(H.user_delta("no", 4500, 4800))
        p.push(H.out_text("Okay.", 4900, 5100))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.6)
        p.push(H.delegation("d-no", 5200))
        await asyncio.sleep(3.0)

    client, provider = await _call(
        monkeypatch, tenant, steps, grace_ms=4000, on_commentary=on_commentary,
        frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    assert spoken["n"], "the research result was never read out"
    runs = [r for r in tenant.runs() if r.get("display_request") == "no"]
    return tenant, runs


@pytest.mark.asyncio
async def test_f1_a_newer_agent_question_voids_the_binding(monkeypatch):
    tenant, runs = await _newer_question(
        monkeypatch, "I found three pharmacies. Want their addresses?",
    )
    assert len(tenant.controls("stop")) == 1, tenant.events       # the stop proceeds
    assert runs and NOT_CARRIED not in runs[0]["message"], runs


@pytest.mark.asyncio
async def test_f1_control_a_newer_statement_keeps_the_binding(monkeypatch):
    tenant, runs = await _newer_question(monkeypatch, "I found three pharmacies.")
    assert _halts(tenant) == 0, tenant.events                       # the 'no' refused it
    assert runs and NOT_CARRIED in runs[0]["message"], runs


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("stop the music", "Should I stop the music? Or just pause it?", "no"),
    ("stop the music", "Should I stop the music? Or just pause it?", "yes"),
    ("stop the music", "Should I stop the music or just pause it?", "no"),
    ("stop the music", "Should I stop the music? Or do you want something else?", "no"),
    ("آهنگ رو قطع", "قطعش کنم یا فقط مکث کنم؟", "نه"),
    ("آهنگ رو قطع", "قطعش کنم؟ یا فقط مکث کنم؟", "آره"),
])
async def test_f1_a_bare_answer_to_a_reply_offering_something_else_is_never_acted_on(
    monkeypatch, command, question, answer,
):
    """Ambiguous: no fire from the answer's delegation or at the grace end; the
    answer is never absorbed — it runs, told the command was NOT carried out
    and the answer was unclear."""

    client, provider, tenant = await _confirm(
        monkeypatch, command, question, answer, after=_later_evidence_and_a_newer_request,
    )
    assert _halts(tenant) == 0, tenant.events
    run = _run(tenant)
    assert run["display_request"].strip() == answer
    assert NOT_CARRIED in run["message"] and "unclear" in run["message"], run["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "stops", "context"), [
    ("yes, stop it", 1, None),
    ("no, don't stop it", 0, "declined it"),
])
async def test_f1_control_an_answer_naming_the_command_is_about_that_command(monkeypatch, answer, stops, context):
    client, provider, tenant = await _confirm(
        monkeypatch, "stop the music", "Should I stop the music? Or just pause it?", answer,
    )
    assert len(tenant.controls("stop")) == stops, tenant.events
    if context:
        assert context in _run(tenant)["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("delay", "stops"), [(16000, 1), (3000, 0)])
async def test_f1_an_expired_confirmation_binds_nothing(monkeypatch, delay, stops):
    """Unexpired only: an answer starting long after the question (provider
    timeline: past the chain's request-span gap and the 15 s expiry) is no
    answer to it — the stop proceeds; the control inside the window
    disarms."""

    client, provider, tenant = await _confirm(
        monkeypatch, "stop the music", "Are you sure?", "no", answer_at=2300 + delay,
    )
    assert len(tenant.controls("stop")) == stops, tenant.events


# ══════════════════════════════════════════════════════════════════════
# WIRE — F2: never swallowed by cancel_idle; stop-verb idioms are ambiguous
# ══════════════════════════════════════════════════════════════════════

F2_ANSWERS = [
    ("stop the music", "Should I stop the music?", "never mind"),
    ("stop the music", "Should I stop the music?", "forget it"),
    ("stop the music", "Are you sure?", "drop it"),
    ("آهنگ رو قطع", "قطعش کنم؟", "بی‌خیال"),
    ("آهنگ رو قطع", "قطعش کنم؟", "ولش کن"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), F2_ANSWERS)
async def test_f2_a_fired_dismissal_answer_runs_with_the_truth(monkeypatch, command, question, answer):
    client, provider, tenant = await _confirm(
        monkeypatch, command, question, answer, wait_s=FIRED_WAIT, grace_ms=FIRED_GRACE,
    )
    assert len(tenant.controls("stop")) == 1, tenant.events
    run = _run(tenant)
    assert run["display_request"].strip() == answer
    assert ALREADY in run["message"] and NOT_CARRIED not in run["message"], run["message"]
    assert not _no_task_spoken(provider), M._said(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    *F2_ANSWERS, ("stop the music", "Should I stop the music?", "let it go"),
])
async def test_f2_an_armed_dismissal_answer_never_stops_and_runs(monkeypatch, command, question, answer):
    client, provider, tenant = await _confirm(
        monkeypatch, command, question, answer, after=_later_evidence_and_a_newer_request,
    )
    assert _halts(tenant) == 0, tenant.events
    run = _run(tenant)
    assert run["display_request"].strip() == answer
    assert NOT_CARRIED in run["message"], run["message"]
    assert not _no_task_spoken(provider), M._said(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("words", ["never mind", "بی‌خیال"])
async def test_f2_control_a_dismissal_with_nothing_asked_is_still_cancel_idle(monkeypatch, words):
    """F12 unchanged for a dismissal that answers no media question: nothing
    in flight → no card, no agent turn, the honest 'nothing is running'."""

    tenant = M.Tenant(agents={"": {"think_s": 0.1}})

    async def steps(p, client):
        p.push(H.user_delta(words, 1000, 1500))
        p.push(H.delegation("dN", 1550))
        await asyncio.sleep(1.5)

    client, provider = await _call(monkeypatch, tenant, steps)
    assert tenant.runs() == []
    assert _no_task_spoken(provider), M._said(provider)


@pytest.mark.asyncio
async def test_f2_a_dismissal_answer_never_cancels_other_work(monkeypatch):
    """«ولش کن» answers the stop question — it is never a cancel of the
    research running beside it."""

    tenant = M.Tenant(device=M._old_song(), agents={
        "دندانپزشک": {"think_s": 4.0}, "": {"think_s": 0.1},
    })

    async def steps(p, client):
        p.push(H.user_delta("یه دندانپزشک نزدیک من پیدا کن", 200, 800))
        p.push(H.delegation("dR", 820))
        await M._wait(lambda: tenant.runs(), 3.0)
        p.push(H.user_delta("آهنگ رو قطع", 1000, 1600))
        p.push(H.out_text("قطعش کنم؟", 1700, 2300))
        p.push(H.out_audio("Q"))
        await asyncio.sleep(ARMED_WAIT)
        p.push(H.user_delta("ولش کن", 3000, 3300))
        p.push(H.out_text("باشه.", 3400, 3600))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.6)
        p.push(H.delegation("d-ans", 3700))
        await asyncio.sleep(2.4)

    client, provider = await _call(
        monkeypatch, tenant, steps, frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    assert _halts(tenant) == 0, tenant.events
    assert "cancelled" not in M._phases(client, "dR"), M._phases(client, "dR")
    answers = [r for r in tenant.runs() if r.get("display_request") == "ولش کن"]
    assert answers and NOT_CARRIED in answers[0]["message"], tenant.bodies


# ══════════════════════════════════════════════════════════════════════
# WIRE — F3: sequence-based evidence identity
# ══════════════════════════════════════════════════════════════════════

class ArrivalTenant(M.Tenant):
    """A tenant that predates contract §7: arrival marks only."""

    def key(self, body, field):
        return None


def _resolving(base, resolved, compose_s):
    class T(base):
        def broadcast(self, title, key):
            return super().broadcast(resolved or title, key)

        async def agent_turn(self, body, relay):
            out = await super().agent_turn(body, relay)
            await asyncio.sleep(compose_s)
            return out
    return T


@pytest.mark.asyncio
@pytest.mark.parametrize("base", [M.Tenant, ArrivalTenant], ids=["v07_tenant", "arrival_tenant"])
@pytest.mark.parametrize(("stop", "ack", "play", "resolved"), [
    ("stop the music", "Okay.", "can you play Halo by Beyonce", "Beyoncé - Halo (Official Video)"),
    # the Persian request declines the fast path too, so the newer play is the
    # AGENT's (its resolved title is unknown to the relay until it announces)
    ("آهنگ رو قطع کن", "چشم.", "میشه هالوی بیانسه رو برام پیدا کنی", "Beyoncé - Halo (Official Video)"),
])
async def test_f3_an_item_first_reported_after_a_newer_request_is_never_evidence(
    monkeypatch, base, stop, ack, play, resolved,
):
    """Nothing plays; the stop is only acknowledged; a NEWER agent play
    starts under its RESOLVED title and the phone reports that title INSIDE
    the stop's grace, while the run is still composing.  No tenant stop — the newest request keeps playing.
    (Control, pinned in test_live_round6_media media-4: a first report that
    arrives while the newer agent run is still THINKING — before its play
    tool went out — is still the stop's evidence.)"""

    tenant = _resolving(base, resolved, 2.5)(agents={
        "halo": {"think_s": 0.1, "play": "Halo by Beyonce", "search_s": 0.2},
        "بیانسه": {"think_s": 0.1, "play": "Halo by Beyonce", "search_s": 0.2},
    })

    async def steps(p, client):
        p.push(H.user_delta(stop, 1000, 1500))
        p.push(H.out_text(ack, 1600, 1800))
        p.push(H.out_audio("OK"))
        await M._acknowledged(client)
        p.push(H.user_delta(play, 2000, 2600))
        p.push(H.delegation("dHalo", 2620))
        await M._wait(lambda: "media_play" in tenant.kinds(), 4.0)
        M._send(client, {"type": "now_playing", "title": tenant.device["title"]})
        await asyncio.sleep(3.5)

    client, provider = await _call(monkeypatch, tenant, steps, grace_ms=3000)
    assert ("media_play", resolved) in tenant.events, tenant.events
    assert tenant.controls("stop") == [], tenant.events
    assert tenant.device is not None and tenant.device["title"] == resolved


@pytest.mark.asyncio
async def test_f3_control_an_item_reported_before_the_newer_request_is_evidence(monkeypatch):
    """The old song was reported BEFORE the newer request was dispatched: it
    is known to predate the halt — the stop fires before the newer play."""

    tenant = _resolving(M.Tenant, "Beyoncé - Halo (Official Video)", 0.5)(
        device=M._old_song(),
        agents={"halo": {"think_s": 0.1, "play": "Halo by Beyonce", "search_s": 0.2}},
    )

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 1000, 1500))
        p.push(H.out_text("Okay.", 1600, 1800))
        p.push(H.out_audio("OK"))
        await M._acknowledged(client)
        p.push(H.user_delta("can you play Halo by Beyonce", 2000, 2600))
        p.push(H.delegation("dHalo", 2620))
        await M._wait(lambda: "media_play" in tenant.kinds(), 4.0)
        await asyncio.sleep(2.0)

    client, provider = await _call(
        monkeypatch, tenant, steps, frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    assert len(tenant.controls("stop")) == 1, tenant.events
    assert ("media_stop", "Fadat Sham") in tenant.events, tenant.events
    assert tenant.device is not None and tenant.device["title"].startswith("Beyoncé")


@pytest.mark.asyncio
async def test_f3_control_a_newer_non_media_request_never_hides_evidence(monkeypatch):
    """Only a newer MEDIA request can own a newly reported item: after a
    weather question the phone's first report of the old song is still the
    stop's evidence (one stop at the grace end)."""

    tenant = M.Tenant(device=M._old_song(), agents={"weather": {"think_s": 3.0}})

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 1000, 1500))
        p.push(H.out_text("Okay.", 1600, 1800))
        p.push(H.out_audio("OK"))
        await M._acknowledged(client)
        p.push(H.user_delta("what's the weather in Paris", 2000, 2600))
        p.push(H.delegation("dW", 2620))
        await M._wait(lambda: tenant.runs(), 3.0)
        M._send(client, {"type": "now_playing", "title": "Fadat Sham"})
        await asyncio.sleep(2.5)

    client, provider = await _call(monkeypatch, tenant, steps)
    assert len(tenant.controls("stop")) == 1, tenant.events
    assert ("media_stop", "Fadat Sham") in tenant.events


def test_f1_a_mixed_remainder_is_a_request_only_when_it_asks_for_something():
    # media words, a predicate or a question make the remainder its own card
    assert live.media_request_shaped("just pause it")
    assert live.media_request_shaped("find me a pharmacy")
    assert live.media_request_shaped("یه داروخونه هم پیدا کن")
    assert live.media_request_shaped("what's the weather in Paris?")
    # a bare adverb left by the classifier ('not really' → 'really') is not
    assert not live.media_request_shaped("really")


# ══════════════════════════════════════════════════════════════════════
# WIRE — F5: an UNHEARD confirmation binds nothing; UNKNOWN keeps the legacy rule
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("stop the music", "Should I stop the music?", "no"),
    ("stop the music", "Are you sure?", "no"),
    ("آهنگ رو قطع", "قطعش کنم؟", "نه"),
])
@pytest.mark.parametrize(("receipt", "stops"), [("unheard", 1), ("heard", 0), (None, 0)],
                         ids=["unheard", "heard", "unknown"])
async def test_f5_an_unheard_confirmation_is_not_answered(monkeypatch, command, question, answer, receipt, stops):
    client, provider, tenant = await _confirm(
        monkeypatch, command, question, answer, receipt=receipt,
    )
    assert len(tenant.controls("stop")) == stops, tenant.events
    run = _run(tenant)
    if receipt == "unheard":
        # never told the caller declined a question they never heard
        assert "declined" not in run["message"] and NOT_CARRIED not in run["message"], run["message"]
    else:
        assert NOT_CARRIED in run["message"] and "declined it" in run["message"], run["message"]


# ══════════════════════════════════════════════════════════════════════
# WIRE — TF132: no new frame, enum or field
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("stop the music", "Are you sure?", "no"),
    ("آهنگ رو قطع", "مطمئنی؟", "ولش کن"),
])
async def test_tf132_control_nothing_new_on_the_wire(monkeypatch, command, question, answer):
    client, provider, tenant = await _confirm(
        monkeypatch, command, question, answer, features=M.TF132,
    )
    assert client.of("media_control") == []
    dumped = repr(client.frames)
    for key in ("media_scope", "before_order", "newer_playing", "reask_result", "media_order"):
        assert key not in dumped, key
    # Without `media_transport` the relay never arms a stop (it cannot reach
    # this phone), so there is no relay-owned command to confirm: the answer
    # is the model's business exactly as shipped.
    assert _halts(tenant) == 0, tenant.events
