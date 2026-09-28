"""R2 addendum 6 R6-19 — C12 pending ownership never grants an irrevocable 'answered'.

The residual (supervisor 10:42, root reproducer
/private/tmp/toup-supervisor-1040.iUS0J1/): R's own answer (E1) is never heard;
the model's next words (E2) are released as a CONTINUATION of R's answer before
the caller's input for the gap before them has settled (R6-13: no wire parent,
provisionally R's relay-side).  E2 is heard.  A by-id reask of R that arrives
while that attribution is still PENDING was judged on the provisional parent:
terminal `answered`.  The late 'hello?' transcript then moved E2 to 'hello?'
(R UNHEARD) — but the app had already closed R for good on that verdict.

The fix, relay side: HEARD evidence that rests ONLY on a pending attribution is
not an answer yet.  The reask fences decide as before, and past them the verdict
is `in_flight` (reason `attribution_pending`) — an existing, non-terminal outcome
(no new frame, field or enum; TF132 gets no frame for it and no firm verdict
either).  Nothing is asked of the model meanwhile (no duplicate work) and the
once-per-turn fence is not spent.  After the window settles the same by-id
reask is judged on the SETTLED owner: `answered` when the continuation was
confirmed (bounded settling, genuine continuation), a (conditional) accept when
it moved.

Wire tests run the REAL relay (`test_live_harness`, fake provider whose output
timeline runs ahead of its input timeline).  `ROUND7_C12_OUT` / `ROUND7_C12_TF132_OUT`
record the tapes in the round-7 C12 format for replay through the app's hooks.
Timing honesty: relay clocks only (the app's clock is exercised separately by
the app guards); the provider offset is a chosen test value, not a measurement.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
import pytest_asyncio

import test_live_harness as H
import test_live_no_receipt_r4 as NR
import test_live_round7_c12 as R
from app.services import live_voice_protocol as live


@pytest_asyncio.fixture(autouse=True)
async def schema():
    from app.db import drop_db, init_db
    from app.db.database import engine
    await init_db()
    yield
    await drop_db()
    await engine.dispose()


#: Output timeline this far ahead of input: the window stays unsettled for the
#: whole scenario (the root reproducer's offset).
FAR = 60000
#: …or only this far: the window settles (no caller input) a few seconds in.
NEAR = 3000


def _record(client: str, name: str, what: str, sockets: list, relay: dict) -> None:
    """The round-7 recorder (same tape format), plus which file wrote it."""

    R._record(client, name, what, sockets, relay)
    path = os.environ.get(R.OUT_ENV[client])
    if not path:
        return
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    data["_relay"]["r619_test_file"] = os.path.abspath(__file__)
    data["_relay"]["r619_test_file_sha256"] = NR._sha(os.path.abspath(__file__))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


def _verdict_spy(monkeypatch) -> list:
    """Every (turn, outcome, reason) the relay's reask resolution returned —
    TF132 is sent no `reask_result`, so this is how its verdict is seen."""

    seen: list = []
    real = live._LiveSession.resolve_reask

    def spy(self, wanted, text):
        target, outcome, why = real(self, wanted, text)
        seen.append((target.turn_id if target is not None else wanted, outcome, why))
        return target, outcome, why

    monkeypatch.setattr(live._LiveSession, "resolve_reask", spy)
    return seen


def _text_replay(phone, text: str = NR.REQUEST) -> None:
    """TF132's reask: the words only, no turn id (resolved by normalized text)."""

    phone.push_frame({"type": "inject_text", "text": text})


# ══════════════════════════════════════════════════════════════════════
# 1. The residual, negotiated client: in_flight while pending; after the
#    late 'hello?' moves E2, R is recoverable — asked once, never twice
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_pending_heard_continuation_is_in_flight_then_r_is_recoverable_once(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    verdicts = _verdict_spy(monkeypatch)
    R._coverage_spy(monkeypatch, facts, "R", "hello")

    async def script(phone, box):
        e1 = await R._answer_fails(phone)
        e2 = await R._release_held(phone, box, e1, facts, start_ms=FAR + 2050)
        await asyncio.sleep(0.2)
        phone.push_frame(NR._idle(e2))                     # E2 was HEARD
        await asyncio.sleep(0.2)
        facts.update(R=NR._finals(phone)[0]["turn_id"], e1=e1, e2=e2)
        phone.reask(facts["R"])                            # the app's hold expired: asked by id
        assert await NR._wait(lambda: len(phone.of("reask_result")) >= 1)
        facts["pending_attributions"] = R._attributions(phone)
        facts["instructions_while_pending"] = list(NR._reask_contents(box["provider"]))
        box["p"].push(H.user_delta("hello?", FAR + 1450, FAR + 1750))    # …late, inside the window
        assert await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        assert await NR._wait(lambda: any(a[2] for a in R._attributions(phone)), 3.0)
        facts["hello"] = NR._finals(phone)[1]["turn_id"]
        phone.reask(facts["R"])                            # re-opened by the correction
        assert await NR._wait(lambda: len(phone.of("reask_result")) >= 2)
        phone.reask(facts["R"])                            # never twice
        assert await NR._wait(lambda: len(phone.of("reask_result")) >= 3)

    psid = R._psid("r619-moved", "new")
    box_ref: dict = {}
    phone, provider = await _socket_with_provider(
        monkeypatch, script, client="new", db=R._db("r619-moved", "new"), psid=psid,
        start=R._start(FAR), box_ref=box_ref,
    )
    rid, hid, e2 = facts["R"], facts["hello"], facts["e2"]
    results = NR._results(phone)
    _record(
        "new", "r619_pending_then_moved",
        "R's answer failed to play; its continuation E2 released unsettled (no wire parent) and "
        "HEARD; a by-id reask of R while E2's attribution is pending → in_flight (never a "
        "terminal answered); the late 'hello?' moves E2; R re-asked → accepted conditional; "
        "a third ask → duplicate",
        [(psid, phone.tape)],
        {"R": rid, "hello": hid, "epochs": [facts["e1"], e2], "results": results,
         "verdicts": verdicts, "attributions": R._attributions(phone),
         "coverage": facts.get("coverage"), "instructions": NR._reask_contents(provider)},
    )
    assert facts["pending_attributions"] == [(e2, rid, False)], facts
    assert results[0] == (rid, "in_flight", None), results
    assert verdicts[0] == (rid, "in_flight", "attribution_pending"), verdicts
    # Not acted on while pending: no instruction, the once-per-turn fence unspent.
    assert facts["instructions_while_pending"] == [], facts
    # After the correction R's own answer is UNHEARD; the reply went to 'hello?'.
    assert facts["coverage"]["R"] == live.OUTPUT_UNHEARD, facts["coverage"]
    assert facts["coverage"]["hello"] == live.OUTPUT_HEARD, facts["coverage"]
    assert R._attributions(phone) == [(e2, rid, False), (e2, hid, True)], R._attributions(phone)
    assert results[1] == (rid, "accepted", True), results
    assert results[2] == (rid, "duplicate", None), results
    contents = NR._reask_contents(provider)
    assert len(contents) == 1 and NR.REQUEST in contents[0], contents
    assert set(R._wire_parents(phone)[e2]) == {None}, R._wire_parents(phone)


async def _socket_with_provider(monkeypatch, script_fn, *, client, db, psid, start, box_ref):
    """`R._socket`, with the provider handed to the script (its instruction
    log is read mid-scenario)."""

    async def script(phone, box):
        box["provider"] = box_ref["provider"]
        await script_fn(phone, box)

    real = H.FakeProvider

    def capture(*args, **kwargs):
        provider = real(*args, **kwargs)
        box_ref["provider"] = provider
        return provider

    monkeypatch.setattr(H, "FakeProvider", capture)
    try:
        return await R._socket(monkeypatch, script, client=client, db=db, psid=psid, start=start)
    finally:
        monkeypatch.setattr(H, "FakeProvider", real)


# ══════════════════════════════════════════════════════════════════════
# 2. Bounded settling: the window settles with NO caller input → the
#    continuation is confirmed and R is answered; nothing asked twice.
#    Control: an UNHEARD continuation is never delayed by R6-19.
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("second", ["idle", "failed"])
async def test_a_pending_window_that_settles_unspoken_confirms_r(monkeypatch, second):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    verdicts = _verdict_spy(monkeypatch)

    async def script(phone, box):
        e1 = await R._answer_fails(phone)
        e2 = await R._release_held(phone, box, e1, facts, text="It opens at nine.", start_ms=NEAR + 2050)
        await asyncio.sleep(0.1)
        phone.push_frame({"idle": NR._idle, "failed": NR._failed}[second](e2))
        await asyncio.sleep(0.15)
        facts.update(R=NR._finals(phone)[0]["turn_id"], e1=e1, e2=e2)
        phone.reask(facts["R"])
        assert await NR._wait(lambda: len(phone.of("reask_result")) >= 1)
        facts["settled_before_first_verdict"] = any(a[2] for a in R._attributions(phone))
        facts["instructions_after_first"] = list(NR._reask_contents(box["provider"]))
        if second == "failed":
            return
        # Nobody spoke: the input clock passes the window → settled on R.
        assert await NR._wait(lambda: any(a[2] for a in R._attributions(phone)), 8.0)
        await asyncio.sleep(0.1)
        phone.reask(facts["R"])                            # the watch's one retry after in_flight
        assert await NR._wait(lambda: len(phone.of("reask_result")) >= 2)

    psid = R._psid("r619-settled", "new", second)
    box_ref: dict = {}
    phone, provider = await _socket_with_provider(
        monkeypatch, script, client="new", db=R._db("r619-settled", "new", second), psid=psid,
        start=R._start(NEAR), box_ref=box_ref,
    )
    rid, e2 = facts["R"], facts["e2"]
    results = NR._results(phone)
    _record(
        "new", f"r619_pending_then_settled[{second}]",
        "R's answer failed; the SAME answer continues (E2, released unsettled, no wire parent); "
        f"second receipt {second}; a by-id reask while pending; the window then settles with no "
        "caller input",
        [(psid, phone.tape)],
        {"R": rid, "epochs": [facts["e1"], e2], "results": results, "verdicts": verdicts,
         "attributions": R._attributions(phone), "instructions": NR._reask_contents(provider)},
    )
    assert not facts["settled_before_first_verdict"], "precondition: the reask came while E2 was pending"
    if second == "failed":
        # Nothing was heard: R6-19 never delays a recovery — the plain truthful accept.
        assert results == [(rid, "accepted", None)], results
        assert verdicts == [(rid, "accepted", "own_unheard")], verdicts
        assert len(facts["instructions_after_first"]) == 1
        return
    assert results[0] == (rid, "in_flight", None), results
    assert verdicts[0] == (rid, "in_flight", "attribution_pending"), verdicts
    assert R._attributions(phone) == [(e2, rid, False), (e2, rid, True)], R._attributions(phone)
    # Settled on R: the heard continuation answers R — firm now, never retried.
    assert results[1] == (rid, "answered", None), results
    assert NR._reask_contents(provider) == [], NR._reask_contents(provider)


# ══════════════════════════════════════════════════════════════════════
# 3. Genuine heard control: R's OWN answer heard → answered at once, even
#    while a continuation of it is still pending
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_control_rs_own_heard_answer_is_answered_while_a_continuation_is_pending(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    verdicts = _verdict_spy(monkeypatch)

    async def script(phone, box):
        assert await NR._wait(lambda: NR._finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.1)
        e1 = NR._epoch(phone)
        phone.push_frame(NR._idle(e1))                     # R's own answer HEARD
        await asyncio.sleep(0.2)
        e2 = await R._release_held(phone, box, e1, facts, start_ms=FAR + 2050)
        await asyncio.sleep(0.2)
        phone.push_frame(NR._idle(e2))
        await asyncio.sleep(0.2)
        facts.update(R=NR._finals(phone)[0]["turn_id"], e1=e1, e2=e2)
        facts["attributions_at_reask"] = R._attributions(phone)
        phone.reask(facts["R"])
        assert await NR._wait(lambda: phone.of("reask_result"))

    psid = R._psid("r619-own-heard", "new")
    phone, provider = await R._socket(
        monkeypatch, script, client="new", db=R._db("r619-own-heard", "new"), psid=psid,
        start=R._start(FAR),
    )
    rid, e2 = facts["R"], facts["e2"]
    assert facts["attributions_at_reask"] == [(e2, rid, False)], facts
    assert NR._results(phone) == [(rid, "answered", None)], NR._results(phone)
    assert verdicts == [(rid, "answered", "epoch")], verdicts
    assert NR._reask_contents(provider) == []


# ══════════════════════════════════════════════════════════════════════
# 4. TF132: no new frame / field / enum, and no firm answered while pending
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tf132_gets_no_new_frame_and_no_firm_answered_while_pending(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    facts: dict = {}
    verdicts = _verdict_spy(monkeypatch)
    R._coverage_spy(monkeypatch, facts, "R", "hello")

    async def script(phone, box):
        e1 = await R._answer_fails(phone)
        e2 = await R._release_held(phone, box, e1, facts, start_ms=FAR + 2050)
        await asyncio.sleep(0.2)
        phone.push_frame(NR._idle(e2))
        await asyncio.sleep(0.2)
        facts.update(R=NR._finals(phone)[0]["turn_id"], e1=e1, e2=e2)
        _text_replay(phone)                                # TF132's replay while E2 is pending
        assert await NR._wait(lambda: len(verdicts) >= 1)
        await asyncio.sleep(0.1)
        facts["instructions_while_pending"] = list(NR._reask_contents(box["provider"]))
        box["p"].push(H.user_delta("hello?", FAR + 1450, FAR + 1750))
        assert await NR._wait(lambda: len(NR._finals(phone)) >= 2)
        await asyncio.sleep(0.3)
        facts["hello"] = NR._finals(phone)[1]["turn_id"]
        _text_replay(phone)                                # …asked again after the correction
        assert await NR._wait(lambda: len(verdicts) >= 2)
        await asyncio.sleep(0.1)

    psid = R._psid("r619-tf132", "tf132")
    box_ref: dict = {}
    phone, provider = await _socket_with_provider(
        monkeypatch, script, client="tf132", db=R._db("r619-tf132", "tf132"), psid=psid,
        start=R._start(FAR), box_ref=box_ref,
    )
    rid, e2 = facts["R"], facts["e2"]
    _record(
        "tf132", "r619_pending_tf132",
        "TF132 features: R's answer failed; E2 released unsettled (null wire parent) and heard; a "
        "text-only replay of R while E2's attribution is pending; the late 'hello?' moves E2; "
        "a second text-only replay",
        [(psid, phone.tape)],
        {"R": rid, "hello": facts["hello"], "epochs": [facts["e1"], e2], "verdicts": verdicts,
         "coverage": facts.get("coverage"), "instructions": NR._reask_contents(provider)},
    )
    ready = phone.of("ready")[0]
    # No new capability, frame type, field or enum value for TF132 (TF132 does
    # not list `reask_turns`, so it is never sent a `reask_result` at all).
    assert "output_attribution" not in ready["capabilities"], ready
    assert not phone.of("output_attribution") and not phone.of("reask_result")
    assert set(R._wire_parents(phone)[e2]) == {None}, R._wire_parents(phone)
    known = {"session_id", "ready", "state", "transcript", "response_text", "audio_delta",
             "speech_segment_complete", "error"}
    assert {f.get("type") for f in phone.frames} <= known, sorted({f.get("type") for f in phone.frames})
    # The relay's own verdict: never a firm answered while pending.
    assert verdicts[0] == (rid, "in_flight", "attribution_pending"), verdicts
    assert facts["instructions_while_pending"] == [], facts
    # After the correction the replay is judged on the settled owner.
    assert facts["coverage"]["R"] == live.OUTPUT_UNHEARD, facts["coverage"]
    assert verdicts[1] == (rid, "accepted", "conditional"), verdicts
    assert len(NR._reask_contents(provider)) == 1


# ══════════════════════════════════════════════════════════════════════
# 5. Units
# ══════════════════════════════════════════════════════════════════════

def test_pending_attributions_name_only_the_pending_owner():
    session = R._unit_session(R.NEW_APP)
    session.continuation_pending = {2: ("R", 1400, 3550), 3: ("H", 3600, 3700)}
    assert session.pending_attributions("R") == frozenset({2})
    assert session.pending_attributions("H") == frozenset({3})
    assert session.pending_attributions("X") == frozenset()
    session.continuation_pending = None
    assert session.pending_attributions("R") == frozenset()


def test_evidence_without_the_pending_epochs():
    session = R._unit_session(R.NEW_APP)
    adapter = session.adapter
    failed = live.ClosedEpoch(epoch=1, output_id="live:p:1", text="a", interrupted=False,
                              parent_user_turn_id="R", retire_reason="playback_failed")
    heard = live.ClosedEpoch(epoch=2, output_id="live:p:2", text="b", interrupted=False,
                             parent_user_turn_id="R", retire_reason="playback_idle")
    adapter._history = [failed, heard]
    adapter._epochs_by_parent = {"R": [failed, heard]}
    adapter._output_id = None                          # nothing streaming
    session.continuation_pending = {2: ("R", 1400, 3550)}
    assert session.turn_output_evidence("R") == live.OUTPUT_HEARD
    assert session.turn_output_evidence("R", exclude=session.pending_attributions("R")) == live.OUTPUT_UNHEARD
    # A heard epoch that is NOT pending keeps the firm verdict.
    session.continuation_pending = {1: ("R", 0, 900)}
    assert session.turn_output_evidence("R", exclude=session.pending_attributions("R")) == live.OUTPUT_HEARD
