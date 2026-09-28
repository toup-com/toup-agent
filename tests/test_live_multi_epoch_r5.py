"""R2 addendum 5 C12 (supervisor 21:54): multi-epoch evidence, tested validly.

The no-receipt verifier's probe `test_verify_nr4.py::test_v_two_epochs_for_r_combine`
failed its OWN precondition on the relay ("both epochs must be parented to R";
the second reply came out parentless) and was waived for that reason.  Not
accepted: this file establishes how the relay REALLY produces more than one
output epoch for one caller turn, and drives the tri-state combine through the
REAL verdict paths at the wire.

How two epochs for R really happen (GPT-Live has no output-complete event):

  1. The SAME answer continues after its first epoch was retired by something
     that says nothing about the MODEL finishing — the relay's own idle timer
     (`gap`, 700 ms of relay-clock silence: a stall in the stream), the phone's
     drain (`playback_idle`) or failure (`playback_failed`), or the phone's
     cut (`interrupt`: the provider's already-generated tail still streams).
     Before this round the continuation went through `consume_direct_turn`,
     which admits only turns no reply fenced yet: R was fenced by the first
     epoch, so the continuation was PARENTLESS and R's coverage depended on an
     epoch that named no turn.  That was a causal-identity DEFECT, fixed in
     `_LiveSession.continuation_parent`: with no NEW cause (no newer caller
     turn — none at all after an interrupt —, no speakable provider append
     since the first epoch began, contiguous on the provider's OUTPUT timeline
     within `_CONTINUATION_MAX_GAP_MS` = 700 ms, the first epoch a direct
     reply), the
     continuation keeps R's parent.  Unrelated and asks-nothing replies are
     never re-parented (addendum 4 §3.1): each has a newer turn or a new cause
     (controls below).
  2. An acknowledgement, then a TOOL CALL: "Let me check." (R's epoch) and the
     delegated result (a claimed epoch parented to R's task turn).  R is then
     consumed by the task, and its coverage is the task's (in_flight /
     answered by task) whatever either epoch's receipt says.
  3. An accepted REASK: the model's answer to the reask instruction is minted
     for R (the held parent, §F).  R is then re-askable no more (`duplicate`,
     once per turn) and never carried (`reasked_turns`) — so that flow has no
     second tri-state verdict to combine.

The tri-state combine (`turn_output_evidence`): HEARD+UNHEARD → HEARD,
UNHEARD+UNKNOWN → UNKNOWN, UNHEARD+UNHEARD → UNHEARD, UNKNOWN+HEARD → HEARD —
here through `resolve_reask` (in socket) and `stranded_turn_record` /
`resolve_carried_reask` (a drop, then a by-id ask on the next socket), with
distinct provider session ids per socket.

Old-fail: run a copy of THIS file against fx4-relay-snapE (lvp 670abce5…): every
combine test fails its parent precondition there (the continuation is
parentless); the controls pass on both.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import test_live_harness as H
import test_live_no_receipt_r4 as NR
from app.services import live_voice_protocol as live


REQ = NR.REQUEST
UNCONF = NR.UNCONFIRMED_MARK          # "never confirmed playing it"
UNHEARD = NR.UNHEARD_MARK             # "did not reach them"
FIRST = "There is one on Front Street."
SECOND = "It opens at nine."

RECEIPTS = {
    None: None,
    "idle": NR._idle,
    "cut0": lambda eid: NR._cut(eid, 0, heard_text=""),
    "failed": NR._failed,
}


def _record(name: str, what: str, sockets: list, relay: dict) -> None:
    """Cross-layer: written ONLY under MULTI_EPOCH_R5_OUT (never otherwise), in
    the no-receipt tape format (test_live_no_receipt_r4.py `_record`): the REAL
    relay's frames to the phone (in), the phone's frames (out), frames sent
    after a drop (undelivered), per socket, plus the relay's ACTUAL verdicts —
    for the cross-layer owner's replay through the app's real hook."""

    path = os.environ.get("MULTI_EPOCH_R5_OUT")
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    data["_what"] = (
        "R2 addendum 5 C12 multi-epoch tapes: two output epochs for one caller turn (the same "
        "answer continuing after the relay's gap timer / the phone's drain, failure or cut), "
        "both parented to R, and the tri-state combine through resolve_reask / "
        "stranded_turn_record / resolve_carried_reask. Written by "
        "backend/tests/test_live_multi_epoch_r5.py under MULTI_EPOCH_R5_OUT."
    )
    data["_relay"] = {
        "tree": os.path.abspath(os.getcwd()),
        "live_voice_protocol_sha256": NR._sha(os.path.abspath(live.__file__)),
        "test_file_sha256": NR._sha(os.path.abspath(__file__)),
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    data["features"] = list(NR.APP_FEATURES)
    data.setdefault("scenarios", {})[name] = {
        "what": what,
        "sockets": [{"provider_session_id": psid, "tape": list(tape)} for psid, tape in sockets],
        "relay": relay,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


def _tag(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _parents(phone) -> list[str]:
    """Each output epoch's parent, in the order the phone got them."""

    seen: dict[str, str] = {}
    for frame in phone.of("response_text"):
        epoch = str(frame.get("assistant_turn_id") or frame.get("response_id") or "")
        seen.setdefault(epoch, str(frame.get("parent_user_turn_id") or ""))
    return list(seen.values())


async def _two_epochs(phone, box, first, second, *, second_start=1500):
    """R, its answer's first epoch, a stall the relay's gap timer retires it
    on, the first receipt, then the SAME answer's next words (contiguous on the
    provider output timeline) and their receipt."""

    assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
    await asyncio.sleep(0.4)                      # the relay's gap timer retires e1
    e1 = NR._epoch(phone)
    if RECEIPTS[first]:
        phone.push_frame(RECEIPTS[first](e1))
        await asyncio.sleep(0.2)
    box["p"].push(H.out_text(SECOND, second_start, second_start + 400))
    box["p"].push(H.out_audio(NR.pcm(1)))
    assert await NR._wait(lambda: NR._epoch(phone) != e1)
    await asyncio.sleep(0.4)                      # …and e2
    e2 = NR._epoch(phone)
    if RECEIPTS[second]:
        phone.push_frame(RECEIPTS[second](e2))
        await asyncio.sleep(0.2)
    return e1, e2


def _start_r(p):
    p.push(H.user_delta(REQ, 0, 900))
    p.push(H.out_text(FIRST, 1000, 1400))
    p.push(H.out_audio(NR.pcm(0)))


async def _socket(monkeypatch, start, script_fn, *, db, psid, features=NR.APP_FEATURES, last=True):
    box: dict = {}

    def on_send(p, e):
        if e.get("type") == "session.start":
            box["p"] = p
            if start is not None:
                start(p)
        elif e.get("type") == "session.close":
            p.push(H.closed())

    provider = H.FakeProvider(on_send=on_send, session_id=psid, auto_ack=True)
    ref: list = []

    async def script():
        await script_fn(ref[0], box)

    items = [NR._config(features), script]
    if last:
        items += [0.3, {"type": "stop"}]
    phone = NR.Phone(items, tape=[])
    ref.append(phone)
    await H.run_relay(phone, provider, timeout=8, db_session_id=db)
    return phone, provider


# ══════════════════════════════════════════════════════════════════════
# 1. The continuation keeps R's parent; the combine through resolve_reask
# ══════════════════════════════════════════════════════════════════════

COMBINE = [
    # (first receipt, second receipt, (outcome, conditional, instruction mark))
    ("idle", "failed", ("answered", None, None)),      # HEARD + UNHEARD → HEARD
    ("failed", None, ("accepted", True, UNCONF)),      # UNHEARD + UNKNOWN → UNKNOWN
    ("cut0", None, ("accepted", True, UNCONF)),        # (the phone's cut, then the tail)
    ("failed", "failed", ("accepted", None, UNHEARD)),  # UNHEARD + UNHEARD → UNHEARD
    ("cut0", "failed", ("accepted", None, UNHEARD)),
    (None, "idle", ("answered", None, None)),          # UNKNOWN + HEARD → HEARD
    (None, None, ("accepted", True, UNCONF)),          # UNKNOWN + UNKNOWN → UNKNOWN
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second", "want"), COMBINE)
async def test_a_continuation_of_rs_answer_keeps_r_and_the_reask_combines(monkeypatch, first, second, want):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}

    async def script(phone, box):
        facts["epochs"] = await _two_epochs(phone, box, first, second)
        phone.reask(NR._finals(phone)[0]["turn_id"])
        assert await NR._wait(lambda: phone.of("reask_result"))

    tag = _tag("in", first, second)
    psid = f"live-psid-me5-in-{tag}"
    phone, provider = await _socket(monkeypatch, _start_r, script, db=f"db-me5-in-{tag}", psid=psid)
    rid = NR._finals(phone)[0]["turn_id"]
    _record(
        f"two_epochs_in_socket[{first},{second}]",
        f"R answered in two epochs (the same answer continuing after its first epoch retired); "
        f"receipts {first}/{second}; R re-asked by id",
        [(psid, phone.tape)],
        {"R": rid, "parents": _parents(phone), "reask": phone.of("reask_result"),
         "instructions": NR._reask_contents(provider)},
    )
    # The precondition the verifier's probe could not meet: BOTH epochs are R's.
    assert _parents(phone) == [rid, rid], _parents(phone)
    result = phone.of("reask_result")[-1]
    outcome, conditional, mark = want
    assert (result["user_turn_id"], result["outcome"], result.get("conditional")) == (
        rid, outcome, conditional,
    ), result
    contents = NR._reask_contents(provider)
    if mark is None:
        assert contents == [], contents
    else:
        assert len(contents) == 1 and mark in contents[0], contents
        if mark == UNHEARD:
            assert UNCONF not in contents[0]


# ══════════════════════════════════════════════════════════════════════
# 2. The same combine carried over a drop (stranded_turn_record /
#    resolve_carried_reask), one provider session per socket
# ══════════════════════════════════════════════════════════════════════

CARRIED = [
    ("idle", "failed", ("unknown", None, None)),               # HEARD: not carried
    ("failed", None, ("accepted", True, UNCONF)),              # UNKNOWN: carried unconfirmed
    ("cut0", "failed", ("accepted", None, UNHEARD)),           # UNHEARD: carried plain
    (None, "idle", ("unknown", None, None)),                   # HEARD: not carried
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second", "want"), CARRIED)
async def test_the_combine_decides_what_a_drop_carries(monkeypatch, first, second, want):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    tag = _tag("carry", first, second)
    db = f"db-me5-carry-{tag}"

    async def s1(phone, box):
        await _two_epochs(phone, box, first, second)
        facts["rid"] = NR._finals(phone)[0]["turn_id"]
        facts["parents"] = _parents(phone)
        phone.push_frame(NR.DROP)

    async def s2(phone, box):
        await asyncio.sleep(0.3)
        phone.reask(facts["rid"])
        assert await NR._wait(lambda: phone.of("reask_result"))

    psid1, psid2 = f"live-psid-me5-c1-{tag}", f"live-psid-me5-c2-{tag}"
    phone1, _provider1 = await _socket(monkeypatch, _start_r, s1, db=db, psid=psid1, last=False)
    phone2, provider2 = await _socket(monkeypatch, None, s2, db=db, psid=psid2)
    _record(
        f"two_epochs_carried[{first},{second}]",
        f"R answered in two epochs (receipts {first}/{second}), the socket dropped, R asked by id "
        "on the repaired socket",
        [(psid1, phone1.tape), (psid2, phone2.tape)],
        {"R": facts["rid"], "parents": facts["parents"], "reask": phone2.of("reask_result"),
         "instructions": NR._reask_contents(provider2)},
    )
    assert facts["parents"] == [facts["rid"], facts["rid"]], facts["parents"]
    result = phone2.of("reask_result")[-1]
    outcome, conditional, mark = want
    assert (result["user_turn_id"], result["outcome"], result.get("conditional")) == (
        facts["rid"], outcome, conditional,
    ), result
    contents = NR._reask_contents(provider2)
    if mark is None:
        assert contents == [], contents
    else:
        assert len(contents) == 1 and mark in contents[0], contents


# ══════════════════════════════════════════════════════════════════════
# 3. Controls: causal identity is never re-parented (addendum 4 §3.1)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("later", "kind"), [
    ("hello?", "asks_nothing"),                       # §3.1: the reply is P's
    ("and is it open on sunday", "request"),          # a newer request's reply is its own
])
async def test_control_a_reply_after_a_newer_turn_is_that_turns(monkeypatch, later, kind):
    """CONTROL (both trees): contiguous on the output timeline, but the caller
    spoke in between — the reply answers THAT turn, never R."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    async def script(phone, box):
        assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.4)
        box["p"].push(H.user_delta(later, 1500, 1900))
        assert await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        box["p"].push(H.out_text("Yes, I'm here." if kind == "asks_nothing" else "It is.", 1500, 1900))
        box["p"].push(H.out_audio(NR.pcm(1)))
        await asyncio.sleep(0.4)

    tag = _tag("newer", later)
    phone, _provider = await _socket(
        monkeypatch, _start_r, script, db=f"db-me5-newer-{tag}", psid=f"live-psid-me5-newer-{tag}",
    )
    finals = NR._finals(phone)
    assert _parents(phone) == [finals[0]["turn_id"], finals[1]["turn_id"]], _parents(phone)


@pytest.mark.asyncio
async def test_control_speech_after_a_real_silence_is_not_a_continuation(monkeypatch):
    """CONTROL (both trees): nothing new was said, but the model's next words
    start 1.4 s after its answer ended on the OUTPUT timeline (the F0 pin's
    "Anything else I can help with?" shape) — a new utterance, not the same
    answer continuing; parentless exactly as before."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    async def script(phone, box):
        await _two_epochs(phone, box, None, None, second_start=1400 + 1400)

    tag = _tag("silence")
    phone, _provider = await _socket(
        monkeypatch, _start_r, script, db=f"db-me5-sil-{tag}", psid=f"live-psid-me5-sil-{tag}",
    )
    assert _parents(phone) == [NR._finals(phone)[0]["turn_id"], ""], _parents(phone)


@pytest.mark.asyncio
async def test_control_speech_after_a_relay_append_is_not_a_continuation(monkeypatch):
    """CONTROL (both trees): the relay told the model something it may answer
    out loud (here the caller's language preference, an instructions append):
    the next words have a new cause and are judged as before (parentless)."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    async def script(phone, box):
        assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.4)
        phone.push_frame({"type": "language_preference", "lang": "fa"})
        await asyncio.sleep(0.3)
        box["p"].push(H.out_text(SECOND, 1500, 1900))
        box["p"].push(H.out_audio(NR.pcm(1)))
        await asyncio.sleep(0.4)

    tag = _tag("append")
    phone, provider = await _socket(
        monkeypatch, _start_r, script, db=f"db-me5-app-{tag}", psid=f"live-psid-me5-app-{tag}",
    )
    assert provider.of("session.instructions.append"), "precondition: the relay appended"
    assert _parents(phone) == [NR._finals(phone)[0]["turn_id"], ""], _parents(phone)


def _stub_session(*, retire_reason="gap", claimed=False, appended=0, turns=None, output_open=False):
    """A `_LiveSession` with just what `continuation_parent` reads."""

    session = live._LiveSession.__new__(live._LiveSession)
    session.utterances = live.UtteranceAssembler("p")
    session.utterances.closed_queue = list(turns or [
        live.Utterance(turn_id="R", ordinal=1, start_ms=0, end_ms=900, text=REQ,
                       causal_speech=True, closed=True, dispatched=True),
    ])
    adapter = live.LiveClientAdapter(session_id="p")
    adapter._history = [live.ClosedEpoch(
        epoch=1, output_id="live:p:1", text=FIRST, interrupted=retire_reason in {"interrupt", "playback_failed"},
        parent_user_turn_id="R", start_ms=1000, end_ms=1400, retire_reason=retire_reason,
    )]
    adapter._epoch = 2 if output_open else 1
    if output_open:
        adapter._output_id = "live:p:2"
    session.adapter = adapter
    session.epoch_births = {1: (0, claimed)}
    session.provider_append_count = appended
    session.relay_confirmation_epochs = {}
    return session


@pytest.mark.parametrize(("kwargs", "offset", "want"), [
    ({}, 1500, "R"),                                                   # gap-retired, contiguous
    ({"retire_reason": "playback_idle"}, 1500, "R"),
    ({"retire_reason": "playback_failed"}, 1500, "R"),
    ({"retire_reason": "interrupt"}, 1500, "R"),
    ({"output_open": True}, 1500, "R"),                                # audio minted e2 first
    ({"retire_reason": "new_output"}, 1500, ""),                       # a new reply closed it
    ({"retire_reason": "causal_user_turn"}, 1500, ""),
    ({"claimed": True}, 1500, ""),                                     # relay line / result epoch
    ({"appended": 1}, 1500, ""),                                       # a new cause since
    ({}, 1400 + 700, "R"),                                             # within the epoch gap
    ({}, 1400 + 701, ""),                                              # a real silence
    ({}, 900, ""),                                                     # before the answer began
])
def test_continuation_parent_rules(kwargs, offset, want):
    """The rule, unit by unit (the wire tests above drive it end to end)."""

    assert _stub_session(**kwargs).continuation_parent(offset) == want


def test_continuation_parent_needs_r_to_be_the_newest_turn():
    r = live.Utterance(turn_id="R", ordinal=1, start_ms=0, end_ms=900, text=REQ,
                       causal_speech=True, closed=True, dispatched=True)
    echo = live.Utterance(turn_id="E", ordinal=2, start_ms=1000, end_ms=1300, text="front street",
                          causal_speech=False, closed=True)
    hello = live.Utterance(turn_id="P", ordinal=2, start_ms=1000, end_ms=1300, text="hello?",
                           causal_speech=True, closed=True)
    # Echo of the answer itself does not end a gap continuation…
    assert _stub_session(turns=[r, echo]).continuation_parent(1500) == "R"
    # …but after an INTERRUPT any caller turn (a barge-in's words) does.
    assert _stub_session(turns=[r, echo], retire_reason="interrupt").continuation_parent(1500) == ""
    # A newer causal turn always does (addendum 4 §3.1: the reply is P's).
    assert _stub_session(turns=[r, hello]).continuation_parent(1500) == ""


# ══════════════════════════════════════════════════════════════════════
# 4. The other real multi-epoch flows, and which verdict governs them
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_an_acknowledgement_then_a_tool_call_is_governed_by_the_task(monkeypatch):
    """"Let me check." (R's own epoch, cut before audio) + the delegation R's
    words became: two epochs for R's turn (the acknowledgement, the claimed
    result), and R's coverage is the TASK's — a reask while it runs is
    `in_flight`, never decided by the acknowledgement's receipt."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    release = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.wait_for(release.wait(), timeout=4.0)
        return "The Front Street office opens at nine.", "m"

    H.patch_relay(monkeypatch, think=think)

    def start(p):
        p.push(H.user_delta(REQ, 0, 900))
        p.push(H.out_text("Let me check.", 1000, 1300))
        p.push(H.out_audio(NR.pcm(0)))
        p.push(H.delegation("d-tool", 1350))

    async def script(phone, box):
        assert await NR._wait(lambda: phone.of("delegation"))
        await asyncio.sleep(0.3)
        phone.push_frame(NR._cut(NR._epoch(phone), 0, heard_text=""))
        await asyncio.sleep(0.2)
        phone.reask(NR._finals(phone)[0]["turn_id"])
        assert await NR._wait(lambda: phone.of("reask_result"))
        release.set()
        await asyncio.sleep(0.6)

    tag = _tag("tool")
    phone, provider = await _socket(
        monkeypatch, start, script, db=f"db-me5-tool-{tag}", psid=f"live-psid-me5-tool-{tag}",
    )
    rid = NR._finals(phone)[0]["turn_id"]
    assert _parents(phone)[0] == rid
    assert [(f["outcome"]) for f in phone.of("reask_result")] == ["in_flight"]
    assert NR._reask_contents(provider) == []


@pytest.mark.asyncio
async def test_a_reask_answer_is_rs_second_epoch_and_r_is_asked_once(monkeypatch):
    """R's answer never reached the caller (cut at 0 ms): the reask is
    accepted plain, and the model's answer to it is minted for R (the held
    parent) — R's second epoch.  R is then never re-asked again (`duplicate`)
    and, after a drop, never carried: that flow ends in no second verdict."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    tag = _tag("reask")
    db = f"db-me5-reask-{tag}"

    async def s1(phone, box):
        assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.4)
        phone.push_frame(NR._cut(NR._epoch(phone), 0, heard_text=""))
        await asyncio.sleep(0.3)
        rid = facts["rid"] = NR._finals(phone)[0]["turn_id"]
        phone.reask(rid)
        assert await NR._wait(lambda: phone.of("reask_result"))
        box["p"].push(H.out_text("There is one on Front Street, open until six.", 6000, 6600))
        box["p"].push(H.out_audio(NR.pcm(2)))
        await asyncio.sleep(0.4)
        phone.reask(rid)
        assert await NR._wait(lambda: len(phone.of("reask_result")) >= 2)
        facts["results"] = [f["outcome"] for f in phone.of("reask_result")]
        facts["parents"] = _parents(phone)
        phone.push_frame(NR.DROP)

    async def s2(phone, box):
        await asyncio.sleep(0.3)
        phone.reask(facts["rid"])
        assert await NR._wait(lambda: phone.of("reask_result"))

    await _socket(monkeypatch, _start_r, s1, db=db, psid=f"live-psid-me5-r1-{tag}", last=False)
    phone2, _p2 = await _socket(monkeypatch, None, s2, db=db, psid=f"live-psid-me5-r2-{tag}")
    assert facts["parents"] == [facts["rid"], facts["rid"]], facts["parents"]
    assert facts["results"] == ["accepted", "duplicate"], facts["results"]
    assert [f["outcome"] for f in phone2.of("reask_result")] == ["unknown"]


def test_the_combiner_itself(monkeypatch):
    """The four combinations the supervisor named, at the combiner (the wire
    tests above reach it through the real verdicts)."""

    def epoch(**kw):
        base = {"epoch": 1, "output_id": "o", "text": "x", "interrupted": False}
        base.update(kw)
        return live.ClosedEpoch(**base)

    heard = dict(retire_reason="playback_idle", played_ms=900)
    unheard = dict(interrupted=True, retire_reason="playback_failed", played_ms=0)
    unknown = dict(retire_reason="gap")
    session = SimpleNamespace(
        adapter=None, playback_receipts=True, epoch_evidence=live._LiveSession.epoch_evidence,
    )
    for pair, want in [
        ((heard, unheard), live.OUTPUT_HEARD),
        ((unheard, unknown), live.OUTPUT_UNKNOWN),
        ((unheard, unheard), live.OUTPUT_UNHEARD),
        ((unknown, heard), live.OUTPUT_HEARD),
    ]:
        epochs = [epoch(**p) for p in pair]
        session.adapter = SimpleNamespace(
            epochs_for=lambda _tid, epochs=epochs: (epochs, False),
            answered_turn=lambda _tid: True,
        )
        assert live._LiveSession.turn_output_evidence(session, "R") == want, pair
