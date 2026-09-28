"""R2 round 5 (addendum 5, the round-4 critics' findings + supervisor 19:35): the relay half.

Binding spec: /private/tmp/toup-voice-r2-spec-addendum-5.md (with the spec and
addenda 1-4).  Findings and probes: /private/tmp/toup-voice-r2-gates/
verify-r3.9uCiCM/r4-round-results.txt; the no-receipt verifier's
relay_probes/test_verify_nr4.py; the supervisor's C10 receipt
/private/tmp/toup-supervisor-r5-review.XWm8nS/selected.xml.

  A3  an OPEN caller turn with text is UNJUDGED: a reask crossing it is
      `in_flight still_open`, in-socket and carried — never `stale` on its
      first words ("can you", «صدامو», "thank you for").
  A4  a turn the SESSION END cut off is classified structurally: a truncated
      fragment (function-word end, Persian object marker, a proper prefix of
      the asks-nothing grammar) never displaces or is executed as the owed
      request — the older request is carried and its instruction says the call
      dropped mid-sentence; with nothing older, the fragment is carried with
      "ask them to finish".  A COMPLETE request closed by the drop is the owed
      request exactly as before (both ways, and without an older request).
  A5  a tap interrupt that names no epoch: played_ms 0/absent → unheard,
      > 0 → UNKNOWN (never heard).
  A7  the "unconfirmed" reask allows the work that never started (a promise
      only), forbids redoing work already done (en + fa).
  B1  a fired media command absorbs only a TRUE continuation (the light-verb /
      politeness tail, hesitations transparent) — never a consent to a newer
      question, never a new play request.
  B2  the agent's own `play_media` is play evidence: pending while its tool
      runs, announced once it succeeded, cleared when it failed.
  B3  a stop the newer-play fence skipped is deferred, bound to that play: it
      runs if the play starts no media, and is dropped if the play starts.
  C1  agreeing punctuated clauses are one pure answer ("Yes, stop it.").
  C2  §1.7 refuses only greetings/presence/closings/prompts, never a
      delegated assent or continuer.
  C3  a question of the model's own in the confirmation's epoch voids it.
  C4  a MIXED answer's outcome line is parented to the new task's turn and no
      relay confirmation line answers a caller turn: the remainder stays
      recoverable.
  C5  every absorbed confirmation delegation gets a function result.
  C6  asks-nothing turns in the MIDDLE of a span leave its words/title/ids.
  C7  value-only negation questions are FULL; an about-question sharing at
      least half its subject is PARTIAL (ask); fewer stays NONE.
  C8  "get/catch that" is presence only in its understanding sense.
  C9  the Persian connector «و» ends the answer.
  C10 (supervisor 19:35) the confirmation applied after the new task B is
      already TERMINAL: B's terminal observation is re-sent relation-only and
      B's saved row is revised with the relation — fast (1.2 s) AND slow
      schedules AND a mixed answer; B never re-run, its result never twice.

Old-fail is proven by running THIS file against fx4-relay-snapD (the
pre-round-5 relay, lvp af4b1e64…): every behaviour test fails there by
assertion; the controls pass on both.
"""

from __future__ import annotations

import asyncio

import pytest

import test_live_harness as H
import test_live_round4 as R4
from app.config import settings
from app.services import live_voice_protocol as live


V03 = R4.V03
TF132 = R4.TF132
Phone = R4.Phone
_send = R4._send
_wait = R4._wait
_finals = R4._finals
_frames_for = R4._frames_for
_phases = R4._phases
_terminal = R4._terminal
_lifecycle = R4._lifecycle
_reask_appends = R4._reask_appends
_commentary = R4._commentary
_thinking = R4._thinking

REQUEST = "what time does the union station library close"
PHARMACY = "find me a pharmacy near union station"
OFFICE_FA = "آدرس دفترش رو بفرست"


def _db(tag: str, *parts) -> str:
    return f"db-r5-{tag}-{abs(hash(parts)) % 10**10}"


def _psid(tag: str, *parts) -> str:
    return f"live-psid-r5-{tag}-{abs(hash(parts)) % 10**8}"


def _reask(turn_id: str, text: str) -> dict:
    return {
        "type": "inject_text", "text": text, "reason": "no_response",
        "reask_of_user_turn_id": turn_id,
    }


def _contents(provider) -> list[str]:
    return [str(a.get("content") or "") for a in _reask_appends(provider)]


def _closing_provider(psid: str, start=None, box=None) -> H.FakeProvider:
    def on_send(p, e):
        if box is not None:
            box["p"] = p
        if e["type"] == "session.start" and start is not None:
            start(p)
        elif e["type"] == "session.close":
            p.push(H.closed())
    return H.FakeProvider(on_send=on_send, auto_ack=True, session_id=psid)


# ══════════════════════════════════════════════════════════════════════
# A3 — an OPEN turn is unjudged: in_flight, never stale on its first words
# ══════════════════════════════════════════════════════════════════════

OPEN_PREFIXES = [
    ("can you", " still hear me?"),
    ("are you", " still there?"),
    ("did you", " hear me?"),
    ("صدامو", " میشنوی؟"),
    ("thank you for", " your help"),
]


async def _reask_across_an_open_turn(monkeypatch, head, tail, *, db):
    """R unanswered; the caller starts `head` (open); the phone's reask of R
    crosses it; the caller finishes the same utterance with `tail`; R is
    asked again after the final."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=900, voice_live_utterance_hard_gap_ms=1800)
    H.patch_relay(monkeypatch)
    box: dict = {}

    def start(p):
        p.push(H.user_delta(REQUEST, 0, 900))

    async def script():
        await _wait(lambda: len(_finals(client)) >= 1)
        r_id = _finals(client)[0]["turn_id"]
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(head, 3000, 3300))
        await _wait(lambda: any(
            not f.get("final") and f.get("turn_id") != r_id for f in client.of("transcript")
        ))
        client.push_frame(_reask(r_id, REQUEST))
        await _wait(lambda: client.of("reask_result"))
        box["p"].push(H.user_delta(tail, 3350, 3900))
        await _wait(lambda: len(_finals(client)) >= 2, 5.0)
        client.push_frame(_reask(r_id, REQUEST))
        await _wait(lambda: len(client.of("reask_result")) >= 2)
        await asyncio.sleep(0.2)

    client = Phone([H.config(features=V03), script, 0.3, {"type": "stop"}])
    provider = _closing_provider(_psid("a3", db), start, box)
    await H.run_relay(client, provider, timeout=12, db_session_id=db)
    return client, provider


@pytest.mark.asyncio
@pytest.mark.parametrize(("head", "tail"), OPEN_PREFIXES)
async def test_a3_a_reask_crossing_an_open_turn_is_in_flight_whatever_its_first_words(monkeypatch, head, tail):
    """Old: `stale newer_turn` on 'can you' / «صدامو» / 'thank you for' — the
    app covered R for good, then the turn closed as a presence check."""

    client, provider = await _reask_across_an_open_turn(monkeypatch, head, tail, db=_db("a3", head))
    finals = _finals(client)
    assert finals[1].get("phatic") is True, finals
    results = [f["outcome"] for f in client.of("reask_result")]
    assert results == ["in_flight", "accepted"], results
    assert len(_contents(provider)) == 1


@pytest.mark.asyncio
async def test_a3_control_an_open_turn_that_closes_as_a_request_supersedes_after_its_final(monkeypatch):
    """CONTROL (both trees): a turn that closes as a REQUEST supersedes R
    once judged — the second verdict is `stale`, nothing re-asked."""

    client, provider = await _reask_across_an_open_turn(
        monkeypatch, "and what about", " the one on king street?", db=_db("a3c"),
    )
    assert _finals(client)[1].get("phatic") is None
    assert [f["outcome"] for f in client.of("reask_result")][-1] == "stale"
    assert _contents(provider) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("head", "tail"), [OPEN_PREFIXES[0], OPEN_PREFIXES[3]])
async def test_a3_a_carried_reask_crossing_an_open_turn_is_in_flight(monkeypatch, head, tail):
    """The same on the repaired socket: the carried R meets the caller's open
    'can you' / «صدامو» — in_flight (old: stale), then accepted once it
    closed as a presence check."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=900, voice_live_utterance_hard_gap_ms=1800)
    H.patch_relay(monkeypatch)
    db = _db("a3carry", head)
    c1 = Phone([H.config(features=V03), 0.6, R4._drop])
    await H.run_relay(
        c1, _closing_provider(_psid("a3c1", head), lambda p: p.push(H.user_delta(REQUEST, 0, 900))),
        timeout=6, db_session_id=db,
    )
    r_id = _finals(c1)[0]["turn_id"]
    box: dict = {}

    async def script():
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(head, 1000, 1300))
        await _wait(lambda: c2.of("transcript"))
        c2.push_frame(_reask(r_id, REQUEST))
        await _wait(lambda: c2.of("reask_result"))
        box["p"].push(H.user_delta(tail, 1350, 1900))
        await _wait(lambda: len(_finals(c2)) >= 1, 5.0)
        c2.push_frame(_reask(r_id, REQUEST))
        await _wait(lambda: len(c2.of("reask_result")) >= 2)
        await asyncio.sleep(0.2)

    c2 = Phone([H.config(features=V03), script, 0.3, {"type": "stop"}])
    p2 = _closing_provider(_psid("a3c2", head), None, box)
    await H.run_relay(c2, p2, timeout=12, db_session_id=db)
    assert [(f["user_turn_id"], f["outcome"]) for f in c2.of("reask_result")] == [
        (r_id, "in_flight"), (r_id, "accepted"),
    ]


# ══════════════════════════════════════════════════════════════════════
# A4 — a turn the session end cut off: truncated vs complete, both ways
# ══════════════════════════════════════════════════════════════════════

TRUNCATED = ["can you", "are you still", "الو هنوز صدامو", "thank you for"]
COMPLETE = [PHARMACY, OFFICE_FA]


@pytest.mark.parametrize("text", TRUNCATED + [
    "did you", "صدامو", "آهنگ رو", "find me a", "where is the", "can you hear",
    "book it for", "یه دندونپزشک با",
])
def test_a4_a_truncated_turn_is_a_fragment(text):
    fn = getattr(live, "teardown_fragment", None)
    assert fn is not None and fn(text), text


@pytest.mark.parametrize("text", COMPLETE + [
    "and one that is open on sunday", "what time is it in tokyo", "یه دندونپزشک پیدا کن",
    "cancel it", "find it", "call them", "turn it off",
])
def test_a4_control_a_complete_request_is_no_fragment(text):
    fn = getattr(live, "teardown_fragment", None)
    assert fn is None or not fn(text), text


async def _drop_mid(monkeypatch, older, open_text, asks, *, tag, delegate_at=None, think=None):
    """Socket 1: `older` (unanswered, may be None) then `open_text` still OPEN
    when the line dies.  Socket 2 (a NEW provider session): the phone asks
    each (label, words) in `asks` by id, in order.  Returns ids, verdicts,
    socket-2 instructions."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=3000, voice_live_utterance_hard_gap_ms=6000)
    H.patch_relay(monkeypatch, think=think)
    db = _db("a4", tag, older, open_text)

    def start(p):
        at = 0
        if older:
            p.push(H.user_delta(older, 0, 900))
            at = 5000
        p.push(H.user_delta(open_text, at, at + 400))
        if delegate_at is not None:
            p.push(H.delegation("d-open", at + delegate_at))

    c1 = Phone([H.config(features=V03), 0.6, R4._drop])
    await H.run_relay(c1, _closing_provider(_psid("a4a", tag, older, open_text), start),
                      timeout=6, db_session_id=db)
    ids: dict[str, str] = {}
    for f in c1.of("transcript"):
        text = (f.get("text") or "").strip()
        if older and text == older:
            ids["R"] = f["turn_id"]
        elif text == open_text:
            ids["Q"] = f["turn_id"]
    script: list = [H.config(features=V03), 0.5]
    for label, words in asks:
        script += [_reask(ids.get(label, f"live-utt:missing:{label}"), words), 0.4]
    script += [{"type": "stop"}]
    c2 = Phone(script)
    p2 = _closing_provider(_psid("a4b", tag, older, open_text))
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    labels = {v: k for k, v in ids.items()}
    verdicts = [(labels.get(f["user_turn_id"], "?"), f["outcome"]) for f in c2.of("reask_result")]
    return ids, verdicts, _contents(p2), c1, c2


@pytest.mark.asyncio
@pytest.mark.parametrize("fragment", TRUNCATED)
async def test_a4_a_truncated_turn_at_the_drop_never_displaces_the_owed_request(monkeypatch, fragment):
    """Old: the fragment was carried and accepted ("reply to exactly these
    words «can you»") and R was `unknown` — lost."""

    ids, verdicts, contents, _c1, _c2 = await _drop_mid(
        monkeypatch, REQUEST, fragment, [("Q", fragment), ("R", REQUEST)], tag="trunc",
    )
    assert verdicts == [("Q", "unknown"), ("R", "accepted")], verdicts
    assert len(contents) == 1, contents
    content = contents[0]
    assert f"«{REQUEST}»" in content, content
    assert f"«{fragment}…»" in content and "call dropped" in content, content
    assert "finish" in content and "do not act on those unfinished words" in content, content


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", COMPLETE)
async def test_a4_control_a_complete_request_at_the_drop_is_the_owed_request(monkeypatch, complete):
    """CONTROL (both trees; pins drop_in_open_request / the verifier's
    test_verify_drop_in_open_request(_askq)): Q is recovered by id; R is
    superseded exactly as in-socket (contract v0.3 §4)."""

    ids, verdicts, contents, _c1, _c2 = await _drop_mid(
        monkeypatch, REQUEST, complete, [("Q", complete), ("R", REQUEST)], tag="complete",
    )
    assert verdicts == [("Q", "accepted"), ("R", "unknown")], verdicts
    assert len(contents) == 1 and f"«{complete}»" in contents[0], contents
    assert "call dropped" not in contents[0] and "تماس قطع شد" not in contents[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("fragment", ["can you", "الو هنوز صدامو"])
async def test_a4_a_truncated_turn_with_nothing_older_owed_asks_them_to_finish(monkeypatch, fragment):
    """No older request: the fragment is carried, but its instruction is
    "the call dropped while the caller was saying «…». Ask them to finish" —
    never "reply once … to exactly these words" (old)."""

    ids, verdicts, contents, _c1, _c2 = await _drop_mid(
        monkeypatch, None, fragment, [("Q", fragment)], tag="alone",
    )
    assert verdicts == [("Q", "accepted")], verdicts
    assert len(contents) == 1
    content = contents[0]
    assert f"«{fragment}…»" in content, content
    if fragment.isascii():
        assert "The call dropped while the caller was saying" in content
        assert "Ask them to finish what they were saying" in content
    else:
        assert "تماس قطع شد" in content and "تمام کند" in content, content
    assert "exactly these words" not in content and "دقیقاً به همین حرف" not in content, content


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", COMPLETE)
async def test_a4_control_a_complete_request_with_nothing_older_is_recovered_plain(monkeypatch, complete):
    ids, verdicts, contents, _c1, _c2 = await _drop_mid(
        monkeypatch, None, complete, [("Q", complete)], tag="alone-complete",
    )
    assert verdicts == [("Q", "accepted")], verdicts
    assert len(contents) == 1 and f"«{complete}»" in contents[0]
    assert "call dropped" not in contents[0] and "تماس قطع شد" not in contents[0]


@pytest.mark.asyncio
async def test_a4_tf132_text_replays_recover_the_request_never_the_fragment(monkeypatch):
    """TF132 names no turn: its text replay of the fragment matches nothing
    carried (no instruction), its replay of R gets R's instruction with the
    truthful drop clause — and no new frame (old: «can you» was accepted as
    'reply once to exactly these words' and R's replay matched nothing)."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=3000, voice_live_utterance_hard_gap_ms=6000)
    H.patch_relay(monkeypatch)
    db = _db("a4tf")

    def start(p):
        p.push(H.user_delta(REQUEST, 0, 900))
        p.push(H.user_delta("can you", 5000, 5400))

    c1 = Phone([H.config(features=TF132), 0.6, R4._drop])
    await H.run_relay(c1, _closing_provider(_psid("a4tf1"), start), timeout=6, db_session_id=db)
    c2 = Phone([
        H.config(features=TF132), 0.5,
        {"type": "inject_text", "text": "can you"}, 0.4,
        {"type": "inject_text", "text": REQUEST}, 0.4, {"type": "stop"},
    ])
    p2 = _closing_provider(_psid("a4tf2"))
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    contents = _contents(p2)
    assert len(contents) == 1, contents
    assert f"«{REQUEST}»" in contents[0] and "«can you…»" in contents[0], contents
    assert c2.of("reask_result") == [] and not any("relation_only" in f for f in c2.frames)


@pytest.mark.asyncio
async def test_a4_a_fragment_is_never_executed_by_a_delegation_made_while_it_was_said(monkeypatch):
    """The model delegated while the caller was still saying 'can you' and
    the line died: the fragment is never the task's words (old: think ran
    «can you»)."""

    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "Done.", "m"

    ids, verdicts, contents, c1, _c2 = await _drop_mid(
        monkeypatch, REQUEST, "can you", [], tag="deleg", delegate_at=200, think=think,
    )
    await H.drain_detached()
    # Old: the fragment was bound (alone, or swept into R's span as
    # "… library close can you") and run as the task's words.
    assert not any("can you" in t for t in thinks), thinks
    assert not any("can you" in str(f.get("title") or "") for f in c1.of("delegation"))


# ══════════════════════════════════════════════════════════════════════
# A5 — a tap interrupt that names no epoch is no heard evidence
# ══════════════════════════════════════════════════════════════════════

async def _tap_without_identity(monkeypatch, frame, *, db):
    """R1 answered and HEARD (identified drain, played_ms 300); R2 asked; its
    reply's TEXT arrives, none of its audio; the caller taps (`frame`); R2 is
    re-asked by id (critic c6 shape)."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    box: dict = {}
    r1, r2 = "what is the weather in paris", "and in london tomorrow"

    def start(p):
        p.push(H.user_delta(r1, 0, 900))
        p.push(H.out_text("It is sunny in Paris.", 1000, 1600))
        p.push(H.out_audio("A"))

    async def script():
        await _wait(lambda: client.of("audio_delta") and _finals(client))
        e1 = client.of("audio_delta")[-1]["response_id"]
        _send(client, {"type": "playback_idle", "response_id": e1, "item_id": e1, "played_ms": 300})
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(r2, 5000, 5600))
        await _wait(lambda: len(_finals(client)) >= 2)
        box["p"].push(H.out_text("In London tomorrow", 6000, 6400))
        await _wait(lambda: any(
            f.get("parent_user_turn_id") == _finals(client)[1]["turn_id"]
            for f in client.of("response_text")
        ))
        _send(client, frame(client))
        await asyncio.sleep(0.5)
        _send(client, _reask(_finals(client)[1]["turn_id"], r2))
        await _wait(lambda: client.of("reask_result"))

    client = Phone([H.config(features=V03), script, 0.3, {"type": "stop"}])
    provider = _closing_provider(_psid("a5", db), start, box)
    await H.run_relay(client, provider, timeout=8, db_session_id=db)
    r2_id = _finals(client)[1]["turn_id"]
    return [(f["user_turn_id"] == r2_id, f["outcome"], f.get("conditional"))
            for f in client.of("reask_result")], _contents(provider)


@pytest.mark.asyncio
async def test_a5_an_identityless_tap_with_a_cumulative_played_ms_is_no_hearing(monkeypatch):
    """Old: the relay counted 300 ms (the PREVIOUS reply's playback) as heard
    for the next reply, of which nothing played — R2 `answered`, lost."""

    verdicts, contents = await _tap_without_identity(
        monkeypatch, lambda _c: {"type": "interrupt", "played_ms": 300}, db=_db("a5-300"),
    )
    assert verdicts == [(True, "accepted", True)], verdicts
    assert len(contents) == 1 and "never confirmed playing it" in contents[0], contents


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", [
    lambda _c: {"type": "interrupt", "played_ms": 0},
    lambda _c: {"type": "interrupt"},
], ids=["played_0", "played_absent"])
async def test_a5_control_an_identityless_tap_with_no_playback_is_unheard(monkeypatch, frame):
    """CONTROL (both trees): 0/absent → UNHEARD → plain truthful reask."""

    verdicts, contents = await _tap_without_identity(monkeypatch, frame, db=_db("a5-0", repr(frame)))
    assert verdicts == [(True, "accepted", None)], verdicts
    assert len(contents) == 1 and "did not reach them" in contents[0], contents


@pytest.mark.asyncio
async def test_a5_control_an_identified_clip_after_played_audio_answers_it(monkeypatch):
    """CONTROL (both trees): the existing receipt rule for a NAMED epoch."""

    def named(client):
        eid = client.of("response_text")[-1].get("assistant_turn_id")
        return {"type": "interrupt", "response_id": eid, "item_id": eid, "played_ms": 300}

    verdicts, contents = await _tap_without_identity(monkeypatch, named, db=_db("a5-named"))
    assert verdicts == [(True, "answered", None)], verdicts
    assert contents == []


def test_a5_the_receipt_rule_for_an_identityless_clip():
    """Pure: the tri-state for a receipt that named no epoch."""

    closed = live.ClosedEpoch(
        epoch=2, output_id="live:x:2", text="In London tomorrow", interrupted=True,
        retire_reason="interrupt", played_ms=300,
    )
    if hasattr(closed, "identityless_receipt"):
        closed.identityless_receipt = True
    assert live._LiveSession.epoch_evidence(closed) == live.OUTPUT_UNKNOWN
    closed.played_ms = 0
    assert live._LiveSession.epoch_evidence(closed) == live.OUTPUT_UNHEARD


# ══════════════════════════════════════════════════════════════════════
# A7 — a promise-only answer: the unconfirmed reask allows the work
# ══════════════════════════════════════════════════════════════════════

async def _promise_only(monkeypatch, words, promise, *, db):
    H.fast_clocks(monkeypatch)
    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)

    def start(p):
        p.push(H.user_delta(words, 0, 900))
        p.push(H.out_text(promise, 1000, 1600))       # text only: no audio, no receipt

    async def script():
        await _wait(lambda: _finals(client) and client.of("response_text"))
        await asyncio.sleep(0.5)
        _send(client, _reask(_finals(client)[0]["turn_id"], words))
        await _wait(lambda: client.of("reask_result"))

    client = Phone([H.config(features=V03), script, 0.3, {"type": "stop"}])
    provider = _closing_provider(_psid("a7", db), start)
    await H.run_relay(client, provider, timeout=8, db_session_id=db)
    return client, provider, thinks


@pytest.mark.asyncio
@pytest.mark.parametrize(("words", "promise", "lang"), [
    ("what are the library hours on sunday", "Sure, let me check the library hours for you.", "en"),
    ("ساعت کاری کتابخونه رو یکشنبه پیدا کن", "باشه، الان ساعت کاری کتابخونه رو برات پیدا می‌کنم.", "fa"),
])
async def test_a7_a_promise_only_answer_is_recovered_and_may_be_delegated(monkeypatch, words, promise, lang):
    """Old: "Don't redo any action or start new work for it" — the work the
    promise owed was forbidden, R silently lost."""

    client, provider, thinks = await _promise_only(monkeypatch, words, promise, db=_db("a7", words))
    assert client.of("delegation") == [] and thinks == []      # precondition: no work exists
    result = client.of("reask_result")[-1]
    assert (result["outcome"], result.get("conditional")) == ("accepted", True), result
    contents = _contents(provider)
    assert len(contents) == 1
    content = contents[0]
    assert f"«{words}»" in content
    if lang == "en":
        assert "never confirmed playing it" in content
        assert "start new work" not in content
        assert "delegate it if it needs work" in content and "only a promise" in content
        assert "Don't redo any action you already took" in content
    else:
        assert "هیچ‌وقت تأیید نکرد" in content and "دوباره انجام نده" in content
        assert "کار تازه‌ای شروع نکن" not in content
        assert "فقط یک قول" in content and "واگذار کن" in content


@pytest.mark.asyncio
async def test_a7_control_work_that_ran_is_never_reasked(monkeypatch):
    """CONTROL (both trees): the fences decide first — a task consumed the
    turn, so the reask is `answered`/`in_flight`, never the unconfirmed line."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    H.patch_relay(monkeypatch)

    def start(p):
        p.push(H.user_delta(REQUEST, 0, 900))
        p.push(H.delegation("d1", 950))

    async def script():
        await _wait(lambda: _terminal(client, "d1"), 4.0)
        await asyncio.sleep(0.3)
        _send(client, _reask(_finals(client)[0]["turn_id"], REQUEST))
        await _wait(lambda: client.of("reask_result"))

    client = Phone([H.config(features=V03), script, 0.3, {"type": "stop"}])
    provider = _closing_provider(_psid("a7c"), start)
    await H.run_relay(client, provider, timeout=8, db_session_id=_db("a7c"))
    assert [f["outcome"] for f in client.of("reask_result")] in (["answered"], ["in_flight"])
    assert _contents(provider) == []


# ══════════════════════════════════════════════════════════════════════
# B1 — only a TRUE continuation of a fired command is absorbed
# ══════════════════════════════════════════════════════════════════════

async def _fired_then(monkeypatch, stop, ack, steps, *, db, play=None):
    """Music is reported playing; `stop` (unpunctuated, phrase-open) gets only
    `ack`; the backstop fires; then `steps(p, client)`."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    controls, thinks, plays = [], [], []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or task)
        return "Done.", "m"

    async def play_media(_u, query, variety=False):
        plays.append(query)
        return f"Starting {query}.", {"type": "youtube", "video_id": "NEWTRACK001", "title": query}

    H.patch_relay(monkeypatch, think=think, control=control, play=play or play_media)
    box: dict = {}

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        p.push(H.user_delta(stop, 3000, 3600))
        p.push(H.out_text(ack, 3700, 3900))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.9)                    # > grace: the backstop fires
        await steps(p, client)

    client = Phone([
        H.config(features=V03), {"type": "now_playing", "title": "Fadat Sham - Mahasti"},
        script, {"type": "stop"},
    ])
    provider = _closing_provider(_psid("b1", db), None, box)
    await H.run_relay(client, provider, timeout=14, db_session_id=db)
    return client, controls, thinks, plays


@pytest.mark.asyncio
@pytest.mark.parametrize(("stop", "ack", "question", "answer"), [
    ("آهنگ رو قطع کن", "چشم.", "می‌خوای به جاش یه پادکست برات پخش کنم؟", "آره"),
    ("آهنگ رو قطع کن", "چشم.", "می‌خوای به جاش یه پادکست برات پخش کنم؟", "بله"),
    ("stop the music", "Okay.", "Do you want me to find a podcast instead?", "yes"),
])
async def test_b1_a_consent_to_the_agents_newer_question_is_never_absorbed(monkeypatch, stop, ack, question, answer):
    """Old: 'media command tail absorbed' → the consent's delegation was
    retired as relay_media: no card, no run."""

    async def steps(p, client):
        p.push(H.out_text(question, 4300, 5600))
        p.push(H.out_audio("Q"))
        await asyncio.sleep(0.5)
        p.push(H.user_delta(answer, 6200, 6500))
        await asyncio.sleep(0.6)
        p.push(H.delegation("d-consent", 6600))
        await asyncio.sleep(1.6)

    client, controls, thinks, _plays = await _fired_then(
        monkeypatch, stop, ack, steps, db=_db("b1c", stop, answer),
    )
    assert controls == ["stop"], controls
    assert thinks, "the consent never ran"
    assert any(f.get("delegation_id") == "d-consent" for f in client.of("delegation"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("stop", "ack", "play_ask"), [
    ("آهنگ رو قطع کن", "چشم.", "یه آهنگ پخش کن"),
    ("آهنگ رو قطع کن", "چشم.", "آهنگ بذار"),
    ("stop the music", "Okay.", "play the music"),
])
async def test_b1_a_new_play_request_after_a_fired_stop_plays(monkeypatch, stop, ack, play_ask):
    """Old: absorbed as the stop's tail — nothing played, no card."""

    async def steps(p, client):
        p.push(H.user_delta(play_ask, 5200, 5900))
        await asyncio.sleep(0.4)
        p.push(H.delegation("d-play", 6000))
        await asyncio.sleep(1.8)

    client, controls, thinks, plays = await _fired_then(
        monkeypatch, stop, ack, steps, db=_db("b1p", stop, play_ask),
    )
    assert controls == ["stop"], controls
    assert plays or thinks, (plays, thinks)
    assert any(f.get("delegation_id") == "d-play" for f in client.of("delegation"))


@pytest.mark.asyncio
@pytest.mark.parametrize("filler", ["اِ", "خب"])
@pytest.mark.parametrize("tail", ["کن", "کن لطفا"])
async def test_b1_a_hesitation_before_the_tail_is_transparent(monkeypatch, filler, tail):
    """«آهنگ رو قطع» + «اِ» + «کن»: one command — stopped once, the late
    delegation absorbed, no «کن» card (old: «کن» ran as research)."""

    async def steps(p, client):
        p.push(H.user_delta(filler, 4500, 4700))
        await asyncio.sleep(0.5)
        p.push(H.user_delta(tail, 5300, 5600))
        p.push(H.out_text("چشم.", 5700, 5900))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.6)
        p.push(H.delegation("d-late", 6000))
        await asyncio.sleep(1.4)

    client, controls, thinks, _plays = await _fired_then(
        monkeypatch, "آهنگ رو قطع", "چشم.", steps, db=_db("b1h", filler, tail),
    )
    assert controls == ["stop"], controls
    assert thinks == [], thinks
    assert client.of("delegation") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("tail", "absorbed"), [("کن", True), ("کن لطفا", True), ("نکن", False)])
async def test_b1_control_the_adjacent_tail_and_a_negation(monkeypatch, tail, absorbed):
    """CONTROL (both trees): «کن» joins the fired «آهنگ رو قطع»; «نکن» never."""

    async def steps(p, client):
        p.push(H.user_delta(tail, 4900, 5200))
        p.push(H.out_text("چشم.", 5300, 5500))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.6)
        p.push(H.delegation("d-late", 5600))
        await asyncio.sleep(1.4)

    client, controls, thinks, _plays = await _fired_then(
        monkeypatch, "آهنگ رو قطع", "چشم.", steps, db=_db("b1a", tail),
    )
    assert controls == ["stop"], controls
    assert (thinks == []) is absorbed, thinks


def test_b1_the_tail_grammar():
    """Pure: which turns are a true continuation."""

    session = live._LiveSession.__new__(live._LiveSession)
    session.utterances = None
    session.tracker = live.LiveDelegationTracker()
    session.deferred_output = []
    fn = getattr(session, "true_media_tail", None)
    if fn is None:
        pytest.fail("no continuation-only tail rule")

    def turn(text):
        return live.Utterance(turn_id="t:9", ordinal=9, start_ms=9000, end_ms=9400, text=text,
                              closed=True, causal_speech=True)

    for text in ["کن", "کن لطفا", "اِ کن", "please", "off", "بکنید", "اِ"]:
        assert fn(turn(text)), text
    for text in ["آره", "بله", "yes", "okay", "یه آهنگ پخش کن", "play the music", "بعدی", "نکن",
                 "یه پادکست بذار"]:
        assert not fn(turn(text)), text


# ══════════════════════════════════════════════════════════════════════
# B2 — the agent's own play_media is play evidence (tool lifecycle)
# ══════════════════════════════════════════════════════════════════════

class Tenant:
    """The tenant's stale-play guard (as the critic modelled it) — used by the
    fake AGENT's play_media tool, which reports its lifecycle to the relay
    exactly as the real agent's inner tool relay does (tool.start / tool.end
    named `play_media`)."""

    def __init__(self, search_s: float, *, found: bool = True):
        self.search_s = search_s
        self.found = found
        self.halt = 0
        self.timeline: list[tuple[str, str]] = []

    async def play(self, query: str) -> bool:
        mark = self.halt
        self.timeline.append(("play_requested", query))
        await asyncio.sleep(self.search_s)
        if self.halt > mark:
            self.timeline.append(("play_dropped_superseded", query))
            return False
        if not self.found:
            self.timeline.append(("play_not_found", query))
            return False
        self.timeline.append(("play_broadcast", query))
        return True

    async def control(self, _uid, action):
        self.halt += 1
        self.timeline.append(("control_requested", action))
        audible = any(t[0] == "play_broadcast" for t in self.timeline)
        return {"ok": True, "action": action, "reason": "stopped" if audible else "nothing_playing"}


def _agent_with_play_tool(tenant: Tenant, query: str):
    async def agent(user_id, task, session_id, relay=None, out=None, **kw):
        if relay is not None:
            await relay.on_event({"type": "tool.start", "call_id": "c-play", "name": "play_media",
                                  "args": {"query": query}})
        ok = await tenant.play(query)
        if relay is not None:
            await relay.on_event({"type": "tool.end", "call_id": "c-play", "name": "play_media",
                                  "ok": ok, "elapsed_ms": 10})
        if ok and out is not None:
            out["media"] = {"type": "youtube", "video_id": "HALO0000001", "title": query}
        return (f"Starting {query}." if ok else "I couldn't start that."), "m"
    return agent


async def _agent_play_then_stop(monkeypatch, tenant, play_text, *, stop_after_done=False, db):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    H.patch_relay(monkeypatch, think=_agent_with_play_tool(tenant, "Halo"), control=tenant.control)
    box: dict = {}

    def start(p):
        p.push(H.user_delta(play_text, 0, 600))
        p.push(H.delegation("dPlay", 620))

    async def script():
        if stop_after_done:
            await _wait(lambda: _terminal(client, "dPlay"), 6.0)
            await asyncio.sleep(0.4)
        else:
            await _wait(lambda: any(t[0] == "play_requested" for t in tenant.timeline))
            await asyncio.sleep(0.3)
        p = box["p"]
        p.push(H.user_delta("stop the music", 3000, 3600))
        p.push(H.out_text("Okay.", 3700, 3900))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(2.2)

    client = Phone([H.config(features=V03), script, {"type": "stop"}])
    provider = _closing_provider(_psid("b2", db), start, box)
    await H.run_relay(client, provider, timeout=16, db_session_id=db)
    return client, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("play_text", ["can you play Halo by Beyonce", "I want to listen to jazz"])
async def test_b2_an_agent_play_in_flight_is_stop_evidence(monkeypatch, play_text):
    """Old: `media backstop skipped reason=not_playing` — the music started
    after the caller's acknowledged stop."""

    tenant = Tenant(search_s=1.6)
    client, provider = await _agent_play_then_stop(monkeypatch, tenant, play_text, db=_db("b2", play_text))
    assert ("control_requested", "stop") in tenant.timeline, tenant.timeline
    assert not any(t[0] == "play_broadcast" for t in tenant.timeline), tenant.timeline
    said = R4._said(provider)
    assert any("won't start «Halo»" in s for s in said), said
    assert not any("Nothing was playing" in s for s in said)


@pytest.mark.asyncio
async def test_b2_a_completed_agent_play_is_announced_evidence(monkeypatch):
    """The agent's play succeeded (announced) before the phone reported it:
    the acknowledged stop is carried out (old: skipped, the track sounds)."""

    tenant = Tenant(search_s=0.2)
    client, provider = await _agent_play_then_stop(
        monkeypatch, tenant, "can you play Halo by Beyonce", stop_after_done=True, db=_db("b2done"),
    )
    assert ("control_requested", "stop") in tenant.timeline, tenant.timeline


@pytest.mark.asyncio
async def test_b2_control_a_failed_agent_play_is_no_evidence(monkeypatch):
    """CONTROL (both trees): the play tool FAILED — nothing plays, nothing
    was reported: the tenant is never called."""

    tenant = Tenant(search_s=0.2, found=False)
    client, provider = await _agent_play_then_stop(
        monkeypatch, tenant, "can you play Halo by Beyonce", stop_after_done=True, db=_db("b2fail"),
    )
    assert ("control_requested", "stop") not in tenant.timeline, tenant.timeline


# ══════════════════════════════════════════════════════════════════════
# B3 — the newer-play fence defers the stop instead of losing it
# ══════════════════════════════════════════════════════════════════════

async def _stop_then_newer_play(monkeypatch, *, found: bool, db):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    timeline: list[tuple[str, str]] = []
    halt = {"seq": 0}

    async def play(_uid, query, variety=False):
        from app.api import ws_realtime as rt

        mark = halt["seq"]
        timeline.append(("play_requested", query))
        await asyncio.sleep(1.2)
        if halt["seq"] > mark:
            timeline.append(("play_dropped_superseded", query))
            return rt.PLAY_MEDIA_SUPERSEDED, None
        if not found:
            timeline.append(("play_not_found", query))
            return "ERROR: nothing found for that.", None
        timeline.append(("play_broadcast", query))
        return (f"Starting {query}.", {"type": "youtube", "video_id": "HALO0000001", "title": "Halo"})

    async def control(_uid, action):
        halt["seq"] += 1
        timeline.append(("control_requested", action))
        return {"ok": True, "action": action, "reason": "stopped"}

    async def agent(user_id, task, session_id, relay=None, out=None, **kw):
        timeline.append(("agent", "could not find it"))
        return "I couldn't find that track.", "m"

    H.patch_relay(monkeypatch, think=agent, play=play, control=control)
    box: dict = {}

    async def script():
        await asyncio.sleep(0.2)
        p = box["p"]
        p.push(H.user_delta("stop the music", 3000, 3500))
        p.push(H.out_text("Okay.", 3600, 3800))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.5)
        p.push(H.user_delta("play halo by beyonce", 4600, 5400))
        p.push(H.delegation("dPlay2", 5450))
        await _wait(lambda: any(t[0] in ("play_not_found", "play_broadcast", "play_dropped_superseded")
                                for t in timeline), 8.0)
        await asyncio.sleep(2.0)

    client = Phone([
        H.config(features=V03), {"type": "now_playing", "title": "Fadat Sham - Mahasti"},
        script, {"type": "stop"},
    ])
    provider = _closing_provider(_psid("b3", db), None, box)
    await H.run_relay(client, provider, timeout=16, db_session_id=db)
    return client, provider, timeline


@pytest.mark.asyncio
async def test_b3_the_stop_runs_when_the_newer_play_starts_no_media(monkeypatch):
    """Old: `tenant skipped reason=newer_play` for good — the old song plays on.

    PIN CHANGED (R2 addendum 6 R6-4, named there: "the B3 deferral and its
    waiting set are removed"): the stop is no longer deferred until the newer
    play ends — it goes to the tenant at fire time, scoped to the plays asked
    for before it, and BEFORE the newer play's own request (so even this
    arrival-keyed fake tenant never supersedes the newer play).  The old
    assertion "the stop runs after the newer play found nothing" pinned the
    deferral itself; what it protected — the caller's stop is never lost and
    is worded by its verdict — is asserted instead.
    """

    client, provider, timeline = await _stop_then_newer_play(monkeypatch, found=False, db=_db("b3f"))
    assert ("control_requested", "stop") in timeline, timeline
    assert timeline.index(("control_requested", "stop")) < timeline.index(("play_requested", "halo by beyonce"))
    assert ("play_dropped_superseded", "halo by beyonce") not in timeline, timeline
    told = R4._said(provider) + [str(e.get("content") or "") for e in _thinking(provider)]
    assert any("Stopped" in s for s in told), told


@pytest.mark.asyncio
async def test_b3_control_a_newer_play_that_starts_is_never_halted(monkeypatch):
    """CONTROL (both trees; the verifier's v5 invariant): the newer play
    plays, and no stop is ever sent after it.

    PIN CHANGED (R2 addendum 6 R6-4: a stop is sent at fire time, scoped,
    never withheld): the old "no stop is sent at all" pinned the B3 fence
    that withheld it.  Now the caller's stop reaches the tenant first — before
    the newer play's request — and the newer play still plays.
    """

    client, provider, timeline = await _stop_then_newer_play(monkeypatch, found=True, db=_db("b3s"))
    assert ("play_broadcast", "halo by beyonce") in timeline, timeline
    kinds = [kind for kind, _what in timeline]
    assert "control_requested" not in kinds[kinds.index("play_broadcast"):], timeline
    assert kinds.index("control_requested") < kinds.index("play_requested"), timeline
    assert "completed" in _phases(client, "dPlay2")


# ══════════════════════════════════════════════════════════════════════
# §5.3 confirmation harness (the model SAYS every relay line as its own
# epoch, like the real provider; B's duration is a parameter — C10)
# ══════════════════════════════════════════════════════════════════════

PROFESSORS = R4.PROFESSORS
PARTIAL = R4.PARTIAL


async def _confirm(
    monkeypatch, steps, *, d2_s=2.5, features=V03, first=PROFESSORS, partial=PARTIAL, db="",
):
    """A (d1) runs; the PARTIAL correction is delegated (d2, `d2_s` long); the
    model says each relay line aloud as its own epoch; `steps(ctx, p,
    client)` plays the phone and the caller.  ctx carries the calls, the
    lines, the saves and the frames."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    ctx: dict = {"calls": [], "killed": [], "questions": [], "lines": []}
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

    recorded = H.patch_relay(monkeypatch, think=think)
    ctx["saves"] = recorded["saves"]
    ctx["a_done"] = a_done
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
            if R4._is_question(content):
                ctx["questions"].append((start, start + 1500))
            ctx["lines"].append((content, start, start + 1500))
            p.push(H.out_text(content, start, start + 1500))
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
        await _wait(lambda: _terminal(client, "d1") and _terminal(client, "d2"), 6.0)
        await asyncio.sleep(0.5)

    client = Phone([H.config(features=features), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, session_id=_psid("conf", db))
    ctx["client"], ctx["provider"] = client, provider
    await H.run_relay(client, provider, timeout=24, db_session_id=db or _db("conf"))
    return ctx


def _question_epoch(client) -> str:
    """The output epoch that SAID the relay's (latest) stop question."""

    said = [
        f for f in client.of("response_text")
        if R4._is_question(f.get("text") or "") and f.get("assistant_turn_id")
    ]
    if said:
        return said[-1]["assistant_turn_id"]
    return client.of("audio_delta")[-1]["response_id"]


def _answer(text, *, delegate=False, at_offset=2500, heard=True):
    async def steps(ctx, p, client):
        if heard:
            _send(client, R4._idle(_question_epoch(client)))
            await asyncio.sleep(0.2)
        # After everything the model has said so far (a fast B's result may
        # already have been spoken after the question).
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


# ══════════════════════════════════════════════════════════════════════
# C1 — agreeing punctuated clauses are one PURE answer
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(("text", "expected"), [
    ("Yes, stop it.", ("assent", "")), ("Yes, cancel it.", ("assent", "")),
    ("Sure, go ahead.", ("assent", "")), ("آره، قطعش کن.", ("assent", "")),
    ("No, keep both.", ("keep", "")), ("No, keep it.", ("keep", "")),
    ("Yes, sounds good.", ("assent", "")), ("Yes. Stop it.", ("assent", "")),
    ("No, don't stop it.", ("keep", "")),
    ("Yes, stop it, and find the dean of engineering", ("assent", "find the dean of engineering")),
])
def test_c1_agreeing_clauses_are_one_answer(text, expected):
    assert live.parse_confirmation_answer(text) == expected, text


@pytest.mark.parametrize("text", ["No, stop it.", "yes. no, keep it", "yes but wait", "Yes, but wait."])
def test_c1_control_opposite_polarity_or_a_hedge_stays_ambiguous(text):
    assert live.parse_confirmation_answer(text)[0] == "ambiguous", text


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Yes, stop it.", "Sure, go ahead.", "آره، قطعش کن."])
async def test_c1_a_punctuated_yes_stops_a_on_the_first_reply(monkeypatch, answer):
    """Old: re-asked ('Sorry — should I stop …? Yes or no?') and then voided."""

    kwargs = {"first": R4.A_FA, "partial": R4.PARTIAL_FA} if not answer.isascii() else {}
    ctx = await _confirm(monkeypatch, _answer(answer), db=_db("c1", answer), **kwargs)
    assert _cancelled_a(ctx), _phases(ctx["client"], "d1")
    assert len(_questions_said(ctx)) == 1, _questions_said(ctx)


@pytest.mark.asyncio
async def test_c1_a_punctuated_no_keeps_both_without_asking_again(monkeypatch):
    ctx = await _confirm(monkeypatch, _answer("No, keep both."), db=_db("c1no"))
    assert not _cancelled_a(ctx)
    assert len(_questions_said(ctx)) == 1, _questions_said(ctx)
    assert any("keep both" in c for c, *_ in ctx["lines"])


# ══════════════════════════════════════════════════════════════════════
# C2 — a delegated assent/continuer runs; greetings/closings do not
# ══════════════════════════════════════════════════════════════════════

PROPOSAL = "I found three professors. I could keep looking for more professors if you'd like."


async def _after_proposal(monkeypatch, answer, *, db):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def start(p):
        p.push(H.out_text(PROPOSAL, 0, 2500))
        p.push(H.out_audio("P"))

    async def script():
        await _wait(lambda: client.of("audio_delta"))
        _send(client, R4._idle(client.of("audio_delta")[-1]["response_id"]))
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(answer, 4000, 4600))
        box["p"].push(H.delegation("d2", 4650))
        await _wait(lambda: _terminal(client, "d2") or bool(_thinking(provider)), 4.0)
        await asyncio.sleep(0.4)

    client = Phone([H.config(features=V03), script, 0.3, {"type": "stop"}])
    provider = _closing_provider(_psid("c2", db), start, box)
    await H.run_relay(client, provider, timeout=10, db_session_id=db)
    return client, provider, thinks


@pytest.mark.parametrize("text", ["sounds good", "okay go on", "perfect", "ادامه بده", "آره عالیه", "yes, perfect"])
def test_c2_consent_shapes(text):
    fn = getattr(live, "consent_or_continuer", None)
    assert fn is not None and fn(text), text


@pytest.mark.parametrize("text", ["okay thanks", "hello", "are you there?", "so?", "thanks man", "hold on",
                                  "mm hm", "مرسی"])
def test_c2_control_greetings_presence_closings_and_prompts_are_no_consent(text):
    fn = getattr(live, "consent_or_continuer", None)
    assert fn is None or not fn(text), text


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["sounds good", "okay go on", "perfect", "ادامه بده", "آره عالیه"])
async def test_c2_a_delegated_consent_to_a_proposal_without_a_question_mark_runs(monkeypatch, answer):
    """Old: "No task was started … this is conversation … do not delegate it
    again" — the consented work never ran."""

    client, provider, thinks = await _after_proposal(monkeypatch, answer, db=_db("c2", answer))
    assert thinks == [answer], thinks
    assert any(f.get("phase") == "created" for f in _frames_for(client, "d2"))


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["okay thanks", "hello?"])
async def test_c2_control_a_closing_or_greeting_after_the_proposal_starts_no_task(monkeypatch, answer):
    client, provider, thinks = await _after_proposal(monkeypatch, answer, db=_db("c2c", answer))
    assert thinks == []
    assert _frames_for(client, "d2") == []
    notes = [e for e in _thinking(provider) if e.get("delegation_id") == "d2"]
    assert notes and "conversation" in notes[0]["content"]


# ══════════════════════════════════════════════════════════════════════
# C3 — a question of the model's own IN the question's epoch voids it
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [
    " Also, do you want me to include their email addresses?",
    " And would you like hotels near the university?",
    " Or do you want me to keep both running?",
])
async def test_c3_assent_to_the_models_own_question_in_the_same_epoch_never_cancels(monkeypatch, tail):
    """Old: the epoch's LATEST sentence was 'a question', so the relay bound
    the 'yes' to its own — A cancelled although the caller answered the
    model (in the third case, the 'yes' asked to keep both)."""

    async def steps(ctx, p, client):
        _q_start, q_end = ctx["questions"][-1]
        p.push(H.out_text(tail, q_end, q_end + 1500))      # same epoch, no receipt yet
        await asyncio.sleep(0.3)
        _send(client, R4._idle(_question_epoch(client)))
        await asyncio.sleep(0.3)
        p.push(H.user_delta("yes", q_end + 3500, q_end + 3900))
        await asyncio.sleep(0.8)

    ctx = await _confirm(monkeypatch, steps, db=_db("c3", tail))
    assert not _cancelled_a(ctx), _phases(ctx["client"], "d1")
    assert _phases(ctx["client"], "d1")[-1] == "completed"


@pytest.mark.asyncio
async def test_c3_control_the_relays_question_alone_binds_the_yes(monkeypatch):
    ctx = await _confirm(monkeypatch, _answer("yes"), db=_db("c3c"))
    assert _cancelled_a(ctx)


def test_c3_the_epoch_reader():
    fn = getattr(live._LiveSession, "question_beside_relay_line", None)
    line = live.relay_line("confirm_stop_ask", "en", title="find iranian computer science professors…")
    assert fn is not None
    assert fn(line + " Or do you want me to keep both running?", line)
    assert fn("Do you want their emails? " + line, line)
    assert not fn(line, line)
    fa = live.relay_line("confirm_stop_ask", "fa", title=R4.A_FA)
    assert not fn(fa, fa)
    assert fn(fa + " ایمیلشون رو هم می‌خوای؟", fa)


# ══════════════════════════════════════════════════════════════════════
# C4 — a MIXED answer's remainder stays recoverable
# ══════════════════════════════════════════════════════════════════════

MIXED = "yes, and search for waterloo robotics professors"


def _mixed_then_reask(text):
    async def steps(ctx, p, client):
        await _answer(text)(ctx, p, client)
        await _wait(lambda: any("Okay, I stopped" in c for c, *_ in ctx["lines"]), 3.0)
        await _wait(lambda: len(client.of("audio_delta")) >= 2, 2.0)
        await asyncio.sleep(0.2)
        _send(client, R4._idle(client.of("audio_delta")[-1]["response_id"]))   # stop line heard
        await asyncio.sleep(0.4)
        answer = [f for f in _finals(client) if f["text"] == text][0]
        ctx["answer_turn"] = answer["turn_id"]
        _send(client, _reask(answer["turn_id"], text))
        await asyncio.sleep(0.6)
    return steps


@pytest.mark.asyncio
async def test_c4_a_mixed_answers_remainder_is_recoverable_when_the_model_is_silent(monkeypatch):
    """Old: the relay's 'Okay, I stopped «A».' was parented to the mixed turn,
    heard, and the remainder's reask was refused `answered` — lost."""

    ctx = await _confirm(monkeypatch, _mixed_then_reask(MIXED), db=_db("c4"))
    client, provider = ctx["client"], ctx["provider"]
    assert _cancelled_a(ctx)
    d2_turn = next(f["turn_id"] for f in _frames_for(client, "d2") if f.get("phase") == "created")
    stop_parents = {
        f.get("parent_user_turn_id") for f in client.of("response_text")
        if "Okay, I stopped" in (f.get("text") or "")
    }
    assert stop_parents == {d2_turn}, stop_parents
    results = [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")]
    assert results == [(ctx["answer_turn"], "accepted")], results
    contents = _contents(provider)
    assert len(contents) == 1 and "«search for waterloo robotics professors»" in contents[0], contents


@pytest.mark.asyncio
async def test_c4_control_a_pure_answer_is_never_reasked(monkeypatch):
    """CONTROL (both trees): a pure 'yes' is consumed — its reask is
    `answered` and nothing is re-applied."""

    async def steps(ctx, p, client):
        await _answer("yes")(ctx, p, client)
        await asyncio.sleep(1.0)
        answer = [f for f in _finals(client) if f["text"] == "yes"][0]
        _send(client, _reask(answer["turn_id"], "yes"))
        await asyncio.sleep(0.5)

    ctx = await _confirm(monkeypatch, steps, db=_db("c4c"))
    assert _cancelled_a(ctx)
    assert [f["outcome"] for f in ctx["client"].of("reask_result")] == ["answered"]
    assert _contents(ctx["provider"]) == []


# ══════════════════════════════════════════════════════════════════════
# C5 — every absorbed confirmation delegation gets a function result
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("second", [None, "hmm, wait"], ids=["no_second_answer", "still_ambiguous"])
async def test_c5_an_absorbed_ambiguous_answer_gets_a_result(monkeypatch, second):
    """Old: the delegation of 'yes but wait' was absorbed and its function
    call never completed."""

    async def steps(ctx, p, client):
        await _answer("yes but wait", delegate=True)(ctx, p, client)
        if second:
            await _wait(lambda: len(ctx["questions"]) >= 2, 3.0)
            await _wait(lambda: len(client.of("audio_delta")) >= 2, 2.0)
            _send(client, R4._idle(_question_epoch(client)))
            await asyncio.sleep(0.2)
            _q_start, q_end = ctx["questions"][-1]
            p.push(H.user_delta(second, q_end + 1500, q_end + 1900))
            await asyncio.sleep(0.8)

    ctx = await _confirm(monkeypatch, steps, db=_db("c5", second))
    assert not _cancelled_a(ctx)
    notes = [e for e in _thinking(ctx["provider"]) if e.get("delegation_id") == "d-answer"]
    assert notes, "the absorbed delegation never got a function result"
    assert _frames_for(ctx["client"], "d-answer") == []


@pytest.mark.asyncio
async def test_c5_control_a_pure_absorbed_answer_gets_its_result(monkeypatch):
    ctx = await _confirm(monkeypatch, _answer("yes", delegate=True), db=_db("c5c"))
    notes = [e for e in _thinking(ctx["provider"]) if e.get("delegation_id") == "d-answer"]
    assert notes and "handled" in notes[0]["content"]


# ══════════════════════════════════════════════════════════════════════
# C6 — an asks-nothing turn in the MIDDLE of the span
# ══════════════════════════════════════════════════════════════════════

async def _three_turns(monkeypatch, first, middle, last, *, db):
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
        p.push(H.user_delta(middle, 2400, 3000))
        await _wait(lambda: len(_finals(client)) >= 2)
        p.push(H.user_delta(last, 4500, 5200))
        await _wait(lambda: len(_finals(client)) >= 3)
        p.push(H.delegation("d-span", 5300))
        await _wait(lambda: _terminal(client, "d-span"), 4.0)
        await asyncio.sleep(0.3)

    client = Phone([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=R4._silent_after(box, first), auto_ack=True, session_id=_psid("c6", db))
    await H.run_relay(client, provider, timeout=10, db_session_id=db)
    return client, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("middle", ["Are you there?", "هستی؟", "Hello? Anyone?"])
async def test_c6_a_middle_asks_nothing_turn_leaves_the_request(monkeypatch, middle):
    """Old: title 'find me a dentist near union station. Are you there? One
    with parking, please.' and request_turn_ids listed the phatic turn."""

    first, last = "find me a dentist near union station.", "One with parking, please."
    client, calls = await _three_turns(monkeypatch, first, middle, last, db=_db("c6", middle))
    finals = _finals(client)
    created = [f for f in _frames_for(client, "d-span") if f.get("phase") == "created"]
    assert created, client.of("delegation")
    title = created[0]["title"]
    assert middle not in title and title.startswith(first) and title.endswith(last), title
    assert created[0].get("request_turn_ids") == [finals[0]["turn_id"], finals[2]["turn_id"]]
    assert [c["display"] for c in calls] == [title]
    assert calls[0]["lang"] == "en"


@pytest.mark.asyncio
async def test_c6_control_an_unpunctuated_middle_stays_joined(monkeypatch):
    """CONTROL (both trees): no structural evidence — words never dropped."""

    client, calls = await _three_turns(
        monkeypatch, "find me a dentist near union station", "okay thanks", "one with parking",
        db=_db("c6c"),
    )
    created = [f for f in _frames_for(client, "d-span") if f.get("phase") == "created"]
    assert created and "okay thanks" in created[0]["title"], created


# ══════════════════════════════════════════════════════════════════════
# C7 — negation QUESTIONS: value-only FULL, about-question PARTIAL
# ══════════════════════════════════════════════════════════════════════

BOOK = "book a dentist appointment for wednesday at 2pm"
BOOK_FA = "یه وقت دندونپزشکی برای امروز بگیر"


@pytest.mark.parametrize(("target", "text", "grade"), [
    (BOOK, "no, thursday?", "full"),
    (BOOK, "no, what about thursday?", "full"),
    (BOOK, "no, how about thursday at 3pm?", "full"),
    (BOOK_FA, "نه، فردا چطور؟", "full"),
    (PROFESSORS, "no, what about associate professors?", "partial"),
])
def test_c7_negation_question_grades(target, text, grade):
    assert live.negation_evidence(text, live._content_words(target)) == grade, text


@pytest.mark.parametrize(("target", "text"), [
    (PROFESSORS, "no, what's the weather in toronto right now?"),
    # (PROFESSORS, "no, what about the downtown campus?") was pinned NONE here.
    # R2 addendum 5 C11 (supervisor 21:54) names that change: an ELLIPTICAL
    # about-question borrows the work's action and is PARTIAL — absent shared
    # words are no proof of independence.  Pinned in test_live_c11_r5.py.
    (PROFESSORS, "no, what time is it in tokyo"),
    (PROFESSORS, "نه، هوای تورنتو چطوره؟"),
    (BOOK, "no, what's on wednesday?"),      # a question ABOUT the value: its own
])
def test_c7_control_fewer_shared_words_stay_none(target, text):
    assert live.negation_evidence(text, live._content_words(target)) == "none", text


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "followup"), [
    (BOOK, "no, thursday?"), (BOOK, "no, what about thursday?"), (BOOK, "no, how about thursday at 3pm?"),
    (BOOK_FA, "نه، فردا چطور؟"),
])
async def test_c7_a_value_question_supersedes_the_running_booking(monkeypatch, first, followup):
    """Old: graded 'none' — two bookings ran with no question."""

    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, first, followup, running=True, db=_db("c7", first, followup),
    )
    assert "superseded" in _phases(client, "d1"), _phases(client, "d1")
    assert _lifecycle(_frames_for(client, "d2"))[-1].get("relation") == {"kind": "replaces", "task_id": "d1"}


@pytest.mark.asyncio
async def test_c7_an_about_question_sharing_the_subject_asks_first(monkeypatch):
    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, PROFESSORS, "no, what about associate professors?", running=True, db=_db("c7p"),
    )
    assert killed == [] and "superseded" not in _phases(client, "d1")
    assert any(R4._is_question(c) for c in _commentary(provider)), _commentary(provider)


@pytest.mark.asyncio
async def test_c7_control_the_finished_path_links_only_full_evidence(monkeypatch):
    """CONTROL (both trees) — REVISED by R2 addendum 5 C11 (supervisor 21:54 /
    22:1x), which names this change: an elliptical about-question is a
    TARGET-DEPENDENT follow-up, and when exactly one finished job is its
    determinable target it gets the non-destructive `replaces` link with that
    job's request + answer as correcting context (before: separate, with
    non-binding context).  Several plausible finished jobs → the relay asks
    which first (test_live_c11_r5.py).  A self-contained question still never
    links (relay-phatic-6, test_live_round4.py)."""

    client, provider, calls, killed = await R4._correction_call(
        monkeypatch, PROFESSORS, "no, what about associate professors?", running=False, db=_db("c7f"),
    )
    lifecycle = _lifecycle(_frames_for(client, "d2"))
    assert lifecycle and all(
        f.get("relation") == {"kind": "replaces", "task_id": "d1"} for f in lifecycle
    ), lifecycle
    assert "superseded" not in _phases(client, "d1") and "cancelled" not in _phases(client, "d1")


# ══════════════════════════════════════════════════════════════════════
# C8 — "get that" is presence only as understanding
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "Can you get that?", "Okay, can you get that?", "can you get that again?", "can you still get that?",
    "hey, can you get that?", "could you catch that",
])
def test_c8_request_framed_get_that_is_a_request(text):
    assert not live.is_phatic(text), text


@pytest.mark.parametrize("text", ["did you get that?", "you get that?", "got that?", "Did you catch that?",
                                  "you got that?"])
def test_c8_control_understanding_checks_stay_phatic(text):
    assert live.is_phatic(text), text


@pytest.mark.asyncio
@pytest.mark.parametrize("ask", ["Can you get that?", "Okay, can you get that?"])
async def test_c8_a_delegated_can_you_get_that_runs(monkeypatch, ask):
    """Old: final phatic:true, the delegation refused as conversation."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "Ordered.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def start(p):
        p.push(H.out_text("The cheapest good one is the Sony WH-CH720N at 149 dollars on Amazon.", 0, 2500))
        p.push(H.out_audio("S"))

    async def script():
        await _wait(lambda: client.of("audio_delta"))
        _send(client, R4._idle(client.of("audio_delta")[-1]["response_id"]))
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta(ask, 4000, 4600))
        box["p"].push(H.delegation("d-get", 4650))
        await _wait(lambda: _terminal(client, "d-get"), 4.0)
        await asyncio.sleep(0.3)

    client = Phone([H.config(features=V03), script, 0.3, {"type": "stop"}])
    provider = _closing_provider(_psid("c8", ask), start, box)
    await H.run_relay(client, provider, timeout=10, db_session_id=_db("c8", ask))
    assert thinks == [ask], thinks
    assert _finals(client)[-1].get("phatic") is None


# ══════════════════════════════════════════════════════════════════════
# C9 — the Persian connector «و» ends the answer
# ══════════════════════════════════════════════════════════════════════

def test_c9_the_persian_connector_splits_a_mixed_answer():
    assert live.parse_confirmation_answer("آره و یه هتل ارزون هم پیدا کن") == (
        "assent", "یه هتل ارزون هم پیدا کن",
    )


@pytest.mark.asyncio
async def test_c9_a_persian_mixed_answer_stops_a_and_runs_the_rest_under_its_own_title(monkeypatch):
    """Old: ambiguous — A not stopped, re-asked, and the card titled «آره و …»."""

    ctx = await _confirm(
        monkeypatch, _answer("آره و یه هتل ارزون هم پیدا کن", delegate=True),
        first=R4.A_FA, partial=R4.PARTIAL_FA, db=_db("c9"),
    )
    assert _cancelled_a(ctx)
    created = [f for f in _frames_for(ctx["client"], "d-answer") if f.get("phase") == "created"]
    assert created and created[0]["title"] == "یه هتل ارزون هم پیدا کن", created


# ══════════════════════════════════════════════════════════════════════
# C10 — the TERMINAL-B relation race (supervisor 19:35)
# ══════════════════════════════════════════════════════════════════════

def _b_ref(ctx) -> str:
    return f"live-delegation:{ctx['provider']._session_id}:d2"


def _saved_b(ctx) -> list[dict]:
    return [s for s in ctx["saves"] if s.get("assistant_ref") == _b_ref(ctx)]


def _relation_after_confirmation(ctx, answer: str) -> tuple[list, list]:
    """(B's lifecycle frames after the caller's answer was final, all of B's
    lifecycle frames)."""

    client = ctx["client"]
    frames = client.frames
    cut = next(
        (i for i, f in enumerate(frames)
         if f.get("type") == "transcript" and f.get("final") and (f.get("text") or "") == answer),
        len(frames),
    )
    after = [
        f for f in frames[cut:]
        if f.get("type") == "delegation" and f.get("delegation_id") == "d2" and "task_revision" in f
    ]
    return after, _lifecycle(_frames_for(client, "d2"))


async def _c10(monkeypatch, *, d2_s, answer, delegate, features=V03, db):
    async def steps(ctx, p, client):
        if d2_s < 2.0:
            # FAST B: finished (terminal) before the caller answers.
            await _wait(lambda: _terminal(client, "d2"), 4.0)
            await asyncio.sleep(0.2)
        await _answer(answer, delegate=delegate)(ctx, p, client)
        await _wait(lambda: _terminal(client, "d1"), 3.0)
        await asyncio.sleep(0.6)

    return await _confirm(monkeypatch, steps, d2_s=d2_s, features=features, db=db)


@pytest.mark.asyncio
@pytest.mark.parametrize(("d2_s", "answer", "delegate"), [
    (1.2, "yes", False),
    (1.2, MIXED, True),
    (3.0, "yes", False),
    (3.0, MIXED, True),
], ids=["fast_yes", "fast_mixed", "slow_yes", "slow_mixed"])
async def test_c10_the_confirmed_relation_reaches_the_card_and_the_saved_row(monkeypatch, d2_s, answer, delegate):
    """Old (fast): A cancelled (user_cancel) but d2_relations=[None, None, None]
    and B's saved row without `related_task_id` — the card and the history
    never showed that B replaced A.  Required for BOTH schedules and a mixed
    answer: the final relation on B's card and on B's saved row, B never
    re-run, its result never delivered twice."""

    ctx = await _c10(monkeypatch, d2_s=d2_s, answer=answer, delegate=delegate,
                     db=_db("c10", d2_s, answer))
    client = ctx["client"]
    assert _cancelled_a(ctx), _phases(client, "d1")
    after, lifecycle = _relation_after_confirmation(ctx, answer)
    replaces = {"kind": "replaces", "task_id": "d1"}
    assert lifecycle and lifecycle[-1].get("relation") == replaces, [f.get("relation") for f in lifecycle]
    assert after and all(f.get("relation") == replaces for f in after), after
    revisions = [f["task_revision"] for f in lifecycle]
    assert revisions == sorted(set(revisions)), revisions
    # Never re-run, never a second outcome, never the result twice.
    assert [did for did, _d in ctx["calls"]].count("d2") == 1, ctx["calls"]
    terminal = [f for f in lifecycle if f.get("phase") in {"completed", "failed", "cancelled"}]
    assert {f["phase"] for f in terminal} == {"completed"}, terminal
    relation_only = [f for f in lifecycle if f.get("relation_only")]
    if d2_s < 2.0:
        assert len(relation_only) == 1 and relation_only[0]["state"] == "completed", relation_only
        assert "result_text" not in relation_only[0] and "spoken" not in relation_only[0]
    said = [c for c, *_ in ctx["lines"] if "Hotel Ocho" in c]
    assert len(said) <= 1, said
    # The saved history says what the card says.
    saved = _saved_b(ctx)
    assert saved, "B's row was never written"
    assert saved[-1]["assistant_voice"].get("related_task_id") == "d1", saved[-1]["assistant_voice"]
    if delegate:
        created = [f for f in _frames_for(client, "d-answer") if f.get("phase") == "created"]
        assert created and created[0]["title"] == "search for waterloo robotics professors"


@pytest.mark.asyncio
async def test_c10_tf132_gets_no_relation_only_frame_but_the_saved_row_is_related(monkeypatch):
    """Feature negotiation: a TF132 client (no reask_turns) receives no new
    frame shape or key; the saved row still carries the relation."""

    ctx = await _c10(monkeypatch, d2_s=1.2, answer="yes", delegate=False, features=TF132,
                     db=_db("c10tf"))
    assert _cancelled_a(ctx)
    assert not any("relation_only" in f for f in ctx["client"].frames)
    saved = _saved_b(ctx)
    assert saved and saved[-1]["assistant_voice"].get("related_task_id") == "d1"


@pytest.mark.asyncio
async def test_c10_control_a_no_answer_never_relates_b(monkeypatch):
    """CONTROL (both trees): "no" keeps both — B is never related to A, on
    the card or in the saved row."""

    ctx = await _c10(monkeypatch, d2_s=1.2, answer="no", delegate=False, db=_db("c10no"))
    assert not _cancelled_a(ctx)
    assert all("relation" not in f for f in _lifecycle(_frames_for(ctx["client"], "d2")))
    assert all(not s["assistant_voice"].get("related_task_id") for s in _saved_b(ctx))
