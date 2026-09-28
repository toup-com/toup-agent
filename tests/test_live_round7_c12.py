"""R2 addendum 6 R6-13 + R6-15 — C12 VERSION-NEGOTIATED REPLAY (release blocker).

The defect (supervisor 06:55, reproduced through the REAL hook): R's answer
(epoch 1) is never heard; the model's next words ("Yes, I'm here.", a reply to a
'hello?' the relay has not transcribed yet) are held as a tentative
continuation of R's answer and released at the hold BOUND — the caller's input
for the window has not settled — and went out with wire parent R.  The phone
heard it and counted it as R's answer; the 'hello?' transcript arrived after
and the relay re-parented the epoch RELAY-SIDE ONLY.  The app never learned:
R silently lost.

The fix, by client version:

* EVERY client (TF132 included — no new frame, field or enum value): the relay
  never commits R as the WIRE parent of a continuation whose input window has
  not settled at its release.  The epoch goes out parentless for its whole
  life — exactly the shipped pre-C12 behaviour — while the relay keeps its
  provisional parent for its own verdicts and re-evaluates it when late input
  lands in the window (reask / carry / coverage unchanged).
* A client that negotiated `output_attribution` (the new app) also gets
  `ready.capabilities.output_attribution` and `output_attribution` frames: a
  provisional one when such an epoch is released (`settled: false`), a settled
  one when the input clock passes its window or a caller turn inside the window
  re-evaluates it, and a CORRECTION when a turn lands in the window of a
  continuation whose parent the wire did name (settled at release).  A
  re-attribution follows a chain of continuations.

Wire tests run the REAL relay (`test_live_harness`, a distinct provider session
id per socket) for BOTH client versions.  `ROUND7_C12_OUT` (new-app features)
and `ROUND7_C12_TF132_OUT` (TF132 features) record the tapes the cross-layer
owner replays through the matching app code (the current hook / the TF132
hook).  Old-fail: this file against a copy of fx4-relay-snapH (lvp 5ce72b7b…).
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


#: The new app's `LIVE_FEATURES` (src/shared/voice/liveProtocol.ts), verbatim.
# PIN CHANGED at integration (R2 addendum 6 R6-10 AP1 / R6-13): the recorder list
# NR.APP_FEATURES already carries media_scope_floor + output_attribution (the
# app's LIVE_FEATURES), so appending them again would duplicate the names.
NEW_APP = list(NR.APP_FEATURES)
#: The shipped TestFlight build.
TF132 = list(NR.TF132)
CLIENTS = {"new": NEW_APP, "tf132": TF132}
OUT_ENV = {"new": "ROUND7_C12_OUT", "tf132": "ROUND7_C12_TF132_OUT"}

#: The provider's OUTPUT timeline runs ahead of its input timeline (as in
#: production), so a continuation's input window is NOT settled when the hold
#: releases it at its bound.
AHEAD = 1500
REPLY = "Yes, I'm here."
FIRST = "There is one on Front Street."
NEWER = "and is there parking"


def _tag(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _psid(name: str, client: str, *parts) -> str:
    return f"live-psid-r7c12-{name}-{client}-{_tag(*parts)}"


def _db(name: str, client: str, *parts) -> str:
    return f"db-r7c12-{name}-{client}-{_tag(*parts)}"


def _record(client: str, name: str, what: str, sockets: list, relay: dict) -> None:
    """Cross-layer: written ONLY under ROUND7_C12_OUT / ROUND7_C12_TF132_OUT
    (never otherwise), in the no-receipt tape format the supervisor's
    replay-c12.js reads: the REAL relay's frames to the phone (in), the phone's
    frames (out), per socket, plus the relay's ACTUAL verdicts."""

    path = os.environ.get(OUT_ENV[client])
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    data["_what"] = (
        "R2 addendum 6 R6-13/R6-15 C12 version-negotiated tapes: a continuation released "
        "before the caller's input for its window settled goes out with NO wire parent; a "
        "client that negotiated output_attribution is told the provisional and the settled "
        "attribution (and a correction of a wire parent the relay later re-evaluates). "
        f"Client: {client}. Written by backend/tests/test_live_round7_c12.py under "
        f"{OUT_ENV[client]}."
    )
    data["_relay"] = {
        "tree": os.path.abspath(os.getcwd()),
        "live_voice_protocol_sha256": NR._sha(os.path.abspath(live.__file__)),
        "test_file_sha256": NR._sha(os.path.abspath(__file__)),
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    data["client"] = client
    data["features"] = list(CLIENTS[client])
    data.setdefault("scenarios", {})[name] = {
        "what": what,
        "sockets": [{"provider_session_id": psid, "tape": list(tape)} for psid, tape in sockets],
        "relay": relay,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


# ── readers ─────────────────────────────────────────────────────────

def _wire_parents(phone) -> dict:
    """epoch id → EVERY parent_user_turn_id the phone was sent for it."""

    seen: dict[str, list] = {}
    for frame in phone.of("response_text"):
        epoch = str(frame.get("assistant_turn_id") or frame.get("response_id") or "")
        seen.setdefault(epoch, []).append(frame.get("parent_user_turn_id"))
    return seen


def _attributions(phone) -> list[tuple]:
    return [
        (f["assistant_turn_id"], f.get("parent_user_turn_id"), f.get("settled"))
        for f in phone.of("output_attribution")
    ]


def _index(phone, pred) -> int:
    for i, frame in enumerate(phone.frames):
        if pred(frame):
            return i
    return -1


def _coverage_spy(monkeypatch, facts, *names):
    real = live._LiveSession.stranded_turn_record

    def spy(self, detached):
        ids = {name: facts.get(name) for name in names}
        if all(ids.values()) and "coverage" not in facts:
            facts["coverage"] = {name: self.turn_output_evidence(tid) for name, tid in ids.items()}
            facts["coverage"]["reparented"] = dict(self.reparented_epochs or {})
        return real(self, detached)

    monkeypatch.setattr(live._LiveSession, "stranded_turn_record", spy)


async def _socket(monkeypatch, script_fn, *, client, db, psid, start, last=True, timeout=14):
    box: dict = {}

    def on_send(p, e):
        if e.get("type") == "session.start":
            box["p"] = p
            start(p)
        elif e.get("type") == "session.close":
            p.push(H.closed())

    provider = H.FakeProvider(on_send=on_send, session_id=psid, auto_ack=True)
    ref: list = []

    async def script():
        await script_fn(ref[0], box)

    items = [NR._config(CLIENTS[client]), script]
    if last:
        items += [0.3, {"type": "stop"}]
    phone = NR.Phone(items, tape=[])
    ref.append(phone)
    await H.run_relay(phone, provider, timeout=timeout, db_session_id=db)
    return phone, provider


def _start(ahead: int):
    def start(p):
        p.push(H.user_delta(NR.REQUEST, 0, 900))
        p.push(H.out_text(FIRST, ahead + 1000, ahead + 1400))
        p.push(H.out_audio(NR.pcm(0)))
    return start


async def _answer_fails(phone) -> str:
    """R closes, its answer's first epoch reaches the phone and fails to play
    (the phone's failure retires it — a continuable retirement)."""

    assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
    await asyncio.sleep(0.1)
    e1 = NR._epoch(phone)
    phone.push_frame(NR._failed(e1))
    await asyncio.sleep(0.2)
    return e1


async def _release_held(phone, box, e1, facts, *, text=REPLY, start_ms=AHEAD + 2050):
    """The model's next words, contiguous on the OUTPUT timeline: held as a
    tentative continuation and released at the hold BOUND (input unsettled)."""

    released_at = asyncio.get_event_loop().time()
    box["p"].push(H.out_text(text, start_ms, start_ms + 350))
    box["p"].push(H.out_audio(NR.pcm(1)))
    assert await NR._wait(lambda: NR._epoch(phone) != e1, 3.0)
    facts["held_s"] = asyncio.get_event_loop().time() - released_at
    return NR._epoch(phone)


# ══════════════════════════════════════════════════════════════════════
# 1. The defect's shape, both client versions
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["new", "tf132"])
async def test_late_input_the_wire_never_commits_r_and_a_negotiated_phone_is_told(monkeypatch, client):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    _coverage_spy(monkeypatch, facts, "R", "hello")

    async def script(phone, box):
        e1 = await _answer_fails(phone)
        e2 = await _release_held(phone, box, e1, facts)
        await asyncio.sleep(0.3)                       # released: nobody (yet) in the window
        facts["e1"], facts["e2"] = e1, e2
        phone.push_frame(NR._idle(e2))                 # the reply was HEARD
        await asyncio.sleep(0.2)
        box["p"].push(H.user_delta("hello?", AHEAD + 1450, AHEAD + 1750))   # …LATE, inside the window
        assert await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        await asyncio.sleep(0.3)
        facts["R"] = NR._finals(phone)[0]["turn_id"]
        facts["hello"] = NR._finals(phone)[1]["turn_id"]
        if client == "new":
            phone.reask(facts["R"])                    # what the new app's watch now sends
            assert await NR._wait(lambda: phone.of("reask_result"))

    psid = _psid("late", client)
    phone, provider = await _socket(
        monkeypatch, script, client=client, db=_db("late", client), psid=psid, start=_start(AHEAD),
    )
    rid, hid, e2 = facts["R"], facts["hello"], facts["e2"]
    _record(
        client, "late_input_reask",
        "R's answer failed to play; a reply to an unheard 'hello?' held as a continuation and "
        "released at its hold bound with NO wire parent (R6-13); the 'hello?' transcript "
        "arrives after; the relay re-evaluates; the new app is told (output_attribution)",
        [(psid, phone.tape)],
        {"R": rid, "hello": hid, "epochs": [facts["e1"], e2],
         "wire_parents": _wire_parents(phone), "attributions": _attributions(phone),
         "held_to_bound_s": round(facts["held_s"], 3), "coverage": facts.get("coverage"),
         "reask": phone.of("reask_result"), "instructions": NR._reask_contents(provider)},
    )
    assert facts["held_s"] >= 0.6, facts["held_s"]
    # EVERY client: the unsettled continuation names no parent on the wire.
    assert _wire_parents(phone)[e2] and set(_wire_parents(phone)[e2]) == {None}, _wire_parents(phone)
    assert _wire_parents(phone)[facts["e1"]][0] == rid
    ready = phone.of("ready")[0]
    # The relay keeps its causal verdict: the reply answered 'hello?', not R.
    assert facts["coverage"]["R"] == live.OUTPUT_UNHEARD, facts["coverage"]
    assert facts["coverage"]["hello"] == live.OUTPUT_HEARD, facts["coverage"]
    if client == "tf132":
        # TF132: no new capability key, no new frame type.
        assert "output_attribution" not in ready["capabilities"], ready
        assert not phone.of("output_attribution")
        return
    assert ready["capabilities"].get("output_attribution") is True, ready
    # Provisional (R) BEFORE the epoch's first frame; settled ('hello?')
    # right after 'hello?' closed.
    assert _attributions(phone) == [(e2, rid, False), (e2, hid, True)], _attributions(phone)
    first_e2 = _index(phone, lambda f: f.get("type") in {"response_text", "audio_delta"}
                      and (f.get("assistant_turn_id") or f.get("response_id")) == e2)
    provisional = _index(phone, lambda f: f.get("type") == "output_attribution")
    hello_final = _index(phone, lambda f: f.get("type") == "transcript" and f.get("final")
                         and f.get("turn_id") == hid)
    settled = _index(phone, lambda f: f.get("type") == "output_attribution" and f.get("settled"))
    assert 0 <= provisional < first_e2 < hello_final < settled, (provisional, first_e2, hello_final, settled)
    result = phone.of("reask_result")[-1]
    assert (result["user_turn_id"], result["outcome"], result.get("conditional")) == (rid, "accepted", True)


# ══════════════════════════════════════════════════════════════════════
# 2. The genuine continuation (no caller speech in its window) keeps R's
#    parent; HEARD / UNHEARD / UNKNOWN combine per C12
# ══════════════════════════════════════════════════════════════════════

COMBINE = [
    # (second receipt, R's relay evidence, reask verdict (outcome, conditional))
    ("idle", live.OUTPUT_HEARD, ("answered", None)),     # UNHEARD + HEARD → HEARD
    ("failed", live.OUTPUT_UNHEARD, ("accepted", None)),  # UNHEARD + UNHEARD → UNHEARD
    (None, live.OUTPUT_UNKNOWN, ("accepted", True)),     # UNHEARD + UNKNOWN → UNKNOWN
]


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["new", "tf132"])
@pytest.mark.parametrize(("second", "evidence", "verdict"), COMBINE)
async def test_a_genuine_continuation_settles_on_r_and_combines(monkeypatch, client, second, evidence, verdict):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    _coverage_spy(monkeypatch, facts, "R")

    async def script(phone, box):
        e1 = await _answer_fails(phone)
        e2 = await _release_held(phone, box, e1, facts, text="It opens at nine.")
        facts["e1"], facts["e2"] = e1, e2
        await asyncio.sleep(0.2)
        if second:
            phone.push_frame({"idle": NR._idle, "failed": NR._failed}[second](e2))
        facts["R"] = NR._finals(phone)[0]["turn_id"]
        # Nobody spoke: the input clock passes the window's end → settled.
        if client == "new":
            assert await NR._wait(lambda: any(a[2] for a in _attributions(phone)), 5.0)
        else:
            await asyncio.sleep(AHEAD / 1000.0 + 1.3)
        await asyncio.sleep(0.2)
        if client == "new":
            phone.reask(facts["R"])
            assert await NR._wait(lambda: phone.of("reask_result"))

    psid = _psid("genuine", client, second)
    phone, provider = await _socket(
        monkeypatch, script, client=client, db=_db("genuine", client, second), psid=psid,
        start=_start(AHEAD),
    )
    rid, e2 = facts["R"], facts["e2"]
    _record(
        client, f"genuine_continuation[{second}]",
        "R's answer failed to play; the SAME answer continues (no caller speech in its window), "
        f"released unsettled with no wire parent; settled on R; second receipt {second}",
        [(psid, phone.tape)],
        {"R": rid, "epochs": [facts["e1"], e2], "wire_parents": _wire_parents(phone),
         "attributions": _attributions(phone), "coverage": facts.get("coverage"),
         "reask": phone.of("reask_result"), "instructions": NR._reask_contents(provider)},
    )
    assert set(_wire_parents(phone)[e2]) == {None}, _wire_parents(phone)
    assert facts["coverage"]["R"] == evidence, facts["coverage"]
    assert facts["coverage"]["reparented"] == {}, facts["coverage"]
    if client == "tf132":
        assert not phone.of("output_attribution")
        return
    assert _attributions(phone) == [(e2, rid, False), (e2, rid, True)], _attributions(phone)
    result = phone.of("reask_result")[-1]
    assert (result["user_turn_id"], result["outcome"], result.get("conditional")) == (rid, *verdict), result


# ══════════════════════════════════════════════════════════════════════
# 3. R6-15 pin: a newer request pending while the unsettled continuation
#    keeps streaming — its null parent is coverage for NO request
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["new", "tf132"])
async def test_a_newer_request_is_never_answered_by_the_unattributed_epoch(monkeypatch, client):
    # The relay's own epoch gap long enough that the continuing words stay in
    # the SAME epoch while the caller's newer words are being closed.
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=1500)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    _coverage_spy(monkeypatch, facts, "R", "N")

    async def script(phone, box):
        e1 = await _answer_fails(phone)
        e2 = await _release_held(phone, box, e1, facts, text="It opens at nine,")
        facts["e1"], facts["e2"] = e1, e2
        # The caller asks something new over the reply that keeps going.
        box["p"].push(H.user_delta(NEWER, AHEAD + 2450, AHEAD + 3200))
        box["p"].push(H.out_text(" and closes at six.", AHEAD + 2400, AHEAD + 2800))
        box["p"].push(H.out_audio(NR.pcm(2)))
        assert await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        facts["N"] = NR._finals(phone)[1]["turn_id"]
        n_final = _index(phone, lambda f: f.get("type") == "transcript" and f.get("final")
                         and f.get("turn_id") == facts["N"])
        assert await NR._wait(lambda: any(
            f.get("type") == "response_text" and f.get("assistant_turn_id") == e2
            for f in phone.frames[n_final + 1:]
        ), 3.0)
        await asyncio.sleep(0.3)
        phone.push_frame(NR._idle(e2))                 # the reply was HEARD
        await asyncio.sleep(0.3)
        facts["R"] = NR._finals(phone)[0]["turn_id"]
        if client == "new":
            phone.reask(facts["N"], NEWER)
            assert await NR._wait(lambda: phone.of("reask_result"))

    psid = _psid("newer", client)
    phone, provider = await _socket(
        monkeypatch, script, client=client, db=_db("newer", client), psid=psid, start=_start(AHEAD),
    )
    rid, nid, e2 = facts["R"], facts["N"], facts["e2"]
    _record(
        client, "newer_request_over_unattributed_epoch",
        "R's answer failed; its continuation released unsettled (no wire parent); the caller "
        "asks a NEWER request over it while it keeps streaming — its later frames arrive "
        "after the newer request's final, still with no parent",
        [(psid, phone.tape)],
        {"R": rid, "N": nid, "epochs": [facts["e1"], e2], "wire_parents": _wire_parents(phone),
         "attributions": _attributions(phone), "coverage": facts.get("coverage"),
         "reask": phone.of("reask_result"), "instructions": NR._reask_contents(provider)},
    )
    n_final = _index(phone, lambda f: f.get("type") == "transcript" and f.get("final")
                     and f.get("turn_id") == nid)
    after = [f for f in phone.frames[n_final + 1:]
             if f.get("type") == "response_text" and f.get("assistant_turn_id") == e2]
    assert after and all(f.get("parent_user_turn_id") is None for f in after), after
    # The reply is R's (the newer request came after its start), never N's.
    assert facts["coverage"]["R"] == live.OUTPUT_HEARD, facts["coverage"]
    assert facts["coverage"]["N"] is None, facts["coverage"]
    if client == "tf132":
        assert not phone.of("output_attribution")
        return
    assert (e2, rid, False) in _attributions(phone) and (e2, rid, True) in _attributions(phone)
    assert all(a[1] != nid for a in _attributions(phone)), _attributions(phone)
    result = phone.of("reask_result")[-1]
    assert (result["user_turn_id"], result["outcome"]) == (nid, "accepted"), result


# ══════════════════════════════════════════════════════════════════════
# 4. A CORRECTION: the continuation's window WAS settled at release (wire
#    parent R), and a caller turn still lands in it afterwards
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["new", "tf132"])
async def test_a_wire_parent_the_relay_reevaluates_is_corrected_for_a_negotiated_phone(monkeypatch, client):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    _coverage_spy(monkeypatch, facts, "R", "hello")

    async def script(phone, box):
        e1 = await _answer_fails(phone)
        await asyncio.sleep(0.5)                       # the input clock passes 1500
        box["p"].push(H.out_text(REPLY, 1500, 1850))   # settled: not held, wire parent R
        box["p"].push(H.out_audio(NR.pcm(1)))
        assert await NR._wait(lambda: NR._epoch(phone) != e1, 3.0)
        await asyncio.sleep(0.2)
        e2 = NR._epoch(phone)
        facts["e1"], facts["e2"] = e1, e2
        phone.push_frame(NR._idle(e2))                 # heard
        await asyncio.sleep(0.2)
        box["p"].push(H.user_delta("hello?", 1420, 1480))   # a transcript still lands in the window
        assert await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        await asyncio.sleep(0.3)
        facts["R"] = NR._finals(phone)[0]["turn_id"]
        facts["hello"] = NR._finals(phone)[1]["turn_id"]
        if client == "new":
            phone.reask(facts["R"])
            assert await NR._wait(lambda: phone.of("reask_result"))

    psid = _psid("correct", client)
    phone, provider = await _socket(
        monkeypatch, script, client=client, db=_db("correct", client), psid=psid, start=_start(0),
    )
    rid, hid, e2 = facts["R"], facts["hello"], facts["e2"]
    _record(
        client, "committed_then_corrected",
        "R's answer failed; its continuation's window had SETTLED at release (wire parent R, "
        "heard); a 'hello?' transcript still lands in that window → the relay re-evaluates "
        "and a negotiated phone gets the correction",
        [(psid, phone.tape)],
        {"R": rid, "hello": hid, "epochs": [facts["e1"], e2], "wire_parents": _wire_parents(phone),
         "attributions": _attributions(phone), "coverage": facts.get("coverage"),
         "reask": phone.of("reask_result"), "instructions": NR._reask_contents(provider)},
    )
    assert set(_wire_parents(phone)[e2]) == {rid}, _wire_parents(phone)
    assert facts["coverage"]["R"] == live.OUTPUT_UNHEARD, facts["coverage"]
    assert facts["coverage"]["hello"] == live.OUTPUT_HEARD, facts["coverage"]
    if client == "tf132":
        # Documented shipped-build limitation: TF132 cannot be told (no new
        # frame); it counted R's heard reply as R's answer on arrival anyway.
        assert not phone.of("output_attribution")
        return
    assert _attributions(phone) == [(e2, hid, True)], _attributions(phone)
    result = phone.of("reask_result")[-1]
    assert (result["user_turn_id"], result["outcome"], result.get("conditional")) == (rid, "accepted", True)


# ══════════════════════════════════════════════════════════════════════
# 5. A chain: the continuation of an unsettled continuation follows it
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["new", "tf132"])
async def test_a_reattribution_follows_the_chain_of_continuations(monkeypatch, client):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    _coverage_spy(monkeypatch, facts, "R", "hello")

    async def script(phone, box):
        e1 = await _answer_fails(phone)
        e2 = await _release_held(phone, box, e1, facts)
        await asyncio.sleep(0.3)                       # the relay's gap timer retires e2
        box["p"].push(H.out_text(" How can I help?", AHEAD + 2500, AHEAD + 2800))
        box["p"].push(H.out_audio(NR.pcm(2)))
        assert await NR._wait(lambda: NR._epoch(phone) not in {e1, e2}, 3.0)
        await asyncio.sleep(0.3)
        e3 = NR._epoch(phone)
        facts["e1"], facts["e2"], facts["e3"] = e1, e2, e3
        phone.push_frame(NR._idle(e3))
        await asyncio.sleep(0.2)
        box["p"].push(H.user_delta("hello?", AHEAD + 1450, AHEAD + 1750))   # in e2's window
        assert await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        facts["R"] = NR._finals(phone)[0]["turn_id"]
        facts["hello"] = NR._finals(phone)[1]["turn_id"]
        if client == "new":
            assert await NR._wait(lambda: sum(1 for a in _attributions(phone) if a[2]) >= 2, 4.0)
        else:
            await asyncio.sleep(1.5)
        await asyncio.sleep(0.2)

    psid = _psid("chain", client)
    phone, provider = await _socket(
        monkeypatch, script, client=client, db=_db("chain", client), psid=psid, start=_start(AHEAD),
    )
    rid, hid, e2, e3 = facts["R"], facts["hello"], facts["e2"], facts["e3"]
    _record(
        client, "chain_reattributed",
        "two continuations of R's failed answer, both released unsettled; 'hello?' lands in "
        "the FIRST one's window → both answer 'hello?'",
        [(psid, phone.tape)],
        {"R": rid, "hello": hid, "epochs": [facts["e1"], e2, e3], "wire_parents": _wire_parents(phone),
         "attributions": _attributions(phone), "coverage": facts.get("coverage")},
    )
    assert set(_wire_parents(phone)[e2]) == {None} and set(_wire_parents(phone)[e3]) == {None}
    assert facts["coverage"]["reparented"] == {int(e2.rsplit(":", 1)[1]): hid,
                                               int(e3.rsplit(":", 1)[1]): hid}, facts["coverage"]
    assert facts["coverage"]["R"] == live.OUTPUT_UNHEARD, facts["coverage"]
    assert facts["coverage"]["hello"] == live.OUTPUT_HEARD, facts["coverage"]
    if client == "tf132":
        assert not phone.of("output_attribution")
        return
    got = _attributions(phone)
    assert got[:2] == [(e2, rid, False), (e3, rid, False)], got
    assert (e2, hid, True) in got and (e3, hid, True) in got, got
    assert got[-1] == (e3, hid, True), got


# ══════════════════════════════════════════════════════════════════════
# 6. Units
# ══════════════════════════════════════════════════════════════════════

def _unit_session(features=()):
    session = live._LiveSession.__new__(live._LiveSession)
    session.features = set(features)
    session.adapter = live.LiveClientAdapter(session_id="p")
    session.adapter._epoch = 2
    session.adapter._output_id = "live:p:2"
    session.adapter._parent_user_turn_id = "R"
    session.user_id = "user-1"
    session.epoch_births = {2: (0, False)}
    session.continuation_epochs = {}
    session.continuation_pending = {}
    session.continuation_withheld = set()
    session.continuation_links = {}
    session.reparented_epochs = {}
    return session


@pytest.mark.parametrize("features", [NEW_APP, TF132])
def test_an_unsettled_claim_is_pending_and_its_wire_parent_withheld(features):
    session = _unit_session(features)
    session.continuation_claim = ("R", 1400, 3550, 1, False)
    lead = session.note_continuation_epoch()
    assert session.continuation_pending == {2: ("R", 1400, 3550)}
    assert session.continuation_epochs == {}
    assert session.continuation_withheld == {2}
    frames = [{"type": "response_text", "text": "x", "partial": True, "assistant_turn_id": "live:p:2",
               "epoch": 2, "parent_user_turn_id": "R"},
              {"type": "response_text", "text": "y", "partial": True}]      # no turn timing: untouched
    session.withhold_wire_parent(frames)
    assert frames[0]["parent_user_turn_id"] is None and "parent_user_turn_id" not in frames[1]
    if "output_attribution" in features:
        assert lead == [{"type": "output_attribution", "assistant_turn_id": "live:p:2",
                         "response_id": "live:p:2", "epoch": 2, "parent_user_turn_id": "R",
                         "settled": False}]
    else:
        assert lead == []


def test_a_settled_claim_is_committed_as_before():
    session = _unit_session(NEW_APP)
    session.continuation_claim = ("R", 1400, 1500, 1, True)
    assert session.note_continuation_epoch() == []
    assert session.continuation_epochs == {2: ("R", 1400, 1500)}
    assert session.continuation_pending == {} and session.continuation_withheld == set()


def test_a_claimed_epoch_is_never_a_pending_continuation():
    session = _unit_session(NEW_APP)
    session.epoch_births = {2: (0, True)}
    session.continuation_claim = ("R", 1400, 3550, 1, False)
    assert session.note_continuation_epoch() == []
    assert session.continuation_pending == {} and session.continuation_withheld == set()


def test_reevaluation_moves_pending_and_committed_and_follows_the_chain():
    session = _unit_session(NEW_APP)
    session.continuation_pending = {2: ("R", 1400, 2000), 3: ("R", 2300, 2400)}
    session.continuation_links = {2: 1, 3: 2}
    hello = live.Utterance(turn_id="H", ordinal=2, start_ms=1500, end_ms=1800, text="hello?",
                           causal_speech=True, closed=True)
    frames = session.reevaluate_continuations(hello)
    assert session.reparented_epochs == {2: "H", 3: "H"}
    assert session.continuation_pending == {3: ("H", 2300, 2400)}    # still unsettled: provisional
    assert [(f["epoch"], f["parent_user_turn_id"], f["settled"]) for f in frames] == [
        (2, "H", True), (3, "H", False),
    ]
    echo = SimpleNamespace(turn_id="E", causal_speech=False, start_ms=2300, end_ms=2350)
    assert session.reevaluate_continuations(echo) == []


@pytest.mark.asyncio
async def test_settlement_waits_for_the_input_clock_unless_the_session_ends(monkeypatch):
    session = _unit_session(NEW_APP)
    session.continuation_pending = {2: ("R", 1400, 3550)}
    sent: list = []

    async def emit(frames):
        sent.extend(frames)

    monkeypatch.setattr(session, "emit_frames", emit, raising=False)
    monkeypatch.setattr(session, "input_clock_ms", lambda: 3000, raising=False)
    await session.settle_continuations()
    assert session.continuation_pending and not sent
    monkeypatch.setattr(session, "input_clock_ms", lambda: 3550, raising=False)
    await session.settle_continuations()
    assert session.continuation_pending == {} and session.continuation_epochs == {2: ("R", 1400, 3550)}
    assert [(f["epoch"], f["parent_user_turn_id"], f["settled"]) for f in sent] == [(2, "R", True)]
    session.continuation_pending = {3: ("R", 3600, 9000)}
    await session.settle_continuations(force=True)
    assert session.continuation_pending == {}


@pytest.mark.parametrize(("withheld", "placed"), [({2}, ""), (set(), "R")])
def test_the_saved_row_goes_where_the_phone_placed_the_reply(monkeypatch, withheld, placed):
    """Transcript parity: a withheld-parent epoch is drawn by the phone as a
    parentless reply (at arrival), so its row's `occurred_at` is reserved as a
    parentless one — never under R, where the phone did not put it.  Control:
    an epoch whose wire parent is R is placed under R, as before."""

    session = _unit_session(NEW_APP)
    session.adapter._output_text = "Yes, I'm here."
    session.adapter._output_start_ms = 3550
    session.continuation_withheld = set(withheld)
    session.anchor_wall = _dt.datetime.now(_dt.timezone.utc)
    session.epoch_stamps = {}
    calls: list = []
    monkeypatch.setattr(session, "stamp_child", lambda ref, parent, start: calls.append(parent) or session.anchor_wall,
                        raising=False)
    monkeypatch.setattr(session, "timeline_output", lambda parent, start: None, raising=False)
    session.reserve_epoch_stamp()
    assert calls == [placed]
