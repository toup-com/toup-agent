"""R2 round 4, supervisor 18:05-18:16: a missing playback receipt is UNKNOWN.

The integration blocker (reproduced with actual relay frames + the real app
hook, /private/tmp/toup-supervisor-r4-no-receipt-cross.cjs): R final → an
R-parented text-only answer → no native playback, no receipt → the app's
bounded liveness timer re-asks R once, by id → the relay answered
`answered` → the app cleared its status and its hand-off → R was silently
unrecoverable.  Root cause: `turn_output_heard` returned True for an open
epoch and for an epoch with no receipt, i.e. receipt ABSENCE was read as
proof of hearing — which contradicts addendum 4 §3.2 as the supervisor fixed
it at 14:19 (completion = HEARD playback under the existing receipt rules).

The relay now keeps three answers apart (`_LiveSession.turn_output_evidence`):

  HEARD    a positive receipt: `heard_verified` non-empty, a `playback_idle`
           drain, a cut after played audio > 0            → `answered`
  UNHEARD  positive non-hearing: a validated EMPTY receipt, `playback_failed`,
           a cut before any played audio                  → plain accept,
           truthful "did not reach them" (carried plain)
  UNKNOWN  no receipt: still open (→ `in_flight`, it is on its way now),
           gap-retired, aged out, dropped with the socket → accept with
           `conditional: true` and an instruction that claims NEITHER
           (recap or ask; never redo work) (carried conditional)

after every existing fence (unknown / duplicate / task consumed-or-owed /
answered-by-task / stale / not caller speech / asks nothing), and a client
that never negotiated playback receipts (no `playback_frames`, no
`heard_text`: build 129 / web) keeps today's semantics exactly (reached =
answered).

Old-fail: run THIS file against fx4-relay-snapB (lvp 99e5b23d…): every
behaviour test fails by assertion; the controls pass on both.

Cross-layer: with NO_RECEIPT_R4_OUT=<path> set, the scenarios marked
`_record(...)` write the relay's full wire tape — relay frames IN, phone
frames OUT, frames sent after a drop UNDELIVERED, per socket, in the order the
relay saw them — plus the ACTUAL relay verdicts, for the app's
scripts/check-voice-no-receipt-r4.js, which replays them through the REAL
`useRealtimeVoice` hook.  Nothing is written without the variable.  The tape
is written BEFORE the relay-side assertions, so a run against the old relay
still yields the tape the app guard shows failing end to end.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import hashlib
import json
import math
import os
import struct

import pytest

import test_live_harness as H
from app.services import live_voice_protocol as live


#: The app's `LIVE_FEATURES` (src/shared/voice/liveProtocol.ts), verbatim —
#: the fixture is recorded with exactly what the app negotiates, and the app
#: guard cross-pins it against the module it loads.
APP_FEATURES = [
    "live_turns", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    "heard_text",
    "reask_turns", "language_pref", "media_transport",
    # PIN CHANGED (R2 addendum 6 R6-4b / R6-10 AP1 and R6-13): the app's LIVE_FEATURES
    # (src/shared/voice/liveProtocol.ts) now also lists these two, so the tapes these
    # recorders write must negotiate exactly what the shipped new app sends.
    "media_scope_floor", "output_attribution",
]
#: The shipped TestFlight build: receipts, but no `reask_turns` (text-only
#: replays, no `reask_result`).
TF132 = [
    "delegation_frames", "heard_text", "live_turns", "media_control",
    "playback_frames", "task_lifecycle", "turn_timing",
]
#: Build 129 / web: never negotiated playback receipts.
WEB = ["live_turns", "state_seq"]

REQUEST = "find a dentist near union station"
ANSWER = "There is one on Front Street, open until six."
NEWER = "and book me a table for two at seven"
RESEARCH = "research the toronto robotics faculty"

UNCONFIRMED_MARK = "never confirmed playing it"
UNHEARD_MARK = "did not reach them"
PLAIN_MARK = "No reply to the caller's most recent words reached them"


# ══════════════════════════════════════════════════════════════════════
# the phone, the tape
# ══════════════════════════════════════════════════════════════════════

def pcm(index: int = 0, samples: int = 480, amp: int = 2000) -> str:
    """20 ms of real PCM16LE at 24 kHz — the app decodes and queues it."""

    values = [int(amp * math.sin((index * samples + k) / 7)) for k in range(samples)]
    return base64.b64encode(struct.pack(f"<{samples}h", *values)).decode()


class _Drop:
    """The socket dies here (the phone lost the network)."""


DROP = _Drop()


def _finals(client) -> list[dict]:
    return [f for f in client.of("transcript") if f.get("final")]


async def _wait(pred, timeout=4.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


class Phone(H.FakeClient):
    """The phone as the relay sees it, and a recorder of the exact wire.

    It reports NOTHING about playback on its own: every receipt a scenario
    wants is pushed explicitly (`push_frame`), so "no receipt" is exactly
    that.  The scenario runs in the background, so the phone can send a
    frame NOW while the scenario keeps going."""

    def __init__(self, script, *, tape: list):
        super().__init__(script)
        self.tape = tape
        self.inbox: list = []
        self._bg = None
        self.dead = False

    async def send_json(self, frame):
        await super().send_json(frame)
        self.tape.append({"undelivered" if self.dead else "in": frame})

    def push_frame(self, frame) -> None:
        self.inbox.append(frame)

    def _deliver(self, item):
        if item is DROP:
            from fastapi import WebSocketDisconnect

            self.dead = True
            self.tape.append({"out": {"type": "__drop__"}})
            raise WebSocketDisconnect(code=1006)
        self.tape.append({"out": item})
        return json.dumps(item)

    async def receive_text(self):
        import inspect

        while True:
            if self.dead:
                await asyncio.sleep(3600)
            if self.inbox:
                item = self.inbox.pop(0)
                self.reads += 1
                return self._deliver(item)
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
            return self._deliver(item)

    def reask(self, turn_id: str, text: str = REQUEST) -> None:
        """What the app's turn watch sends: the turn's words, by id."""

        self.push_frame({
            "type": "inject_text", "text": text, "reason": "no_response",
            "reask_of_user_turn_id": turn_id,
        })


def _config(features=None):
    if features is None:
        return H.legacy_config()
    return {"type": "config", "protocol": "live1", "voice": "shimmer", "features": list(features)}


def _provider(start=None, *, psid: str = H.PSID, speak_on_commentary: bool = False):
    def on_send(p, e):
        if e.get("type") == "session.start" and start is not None:
            start(p)
        elif e.get("type") == "session.close":
            p.push(H.closed())
    return H.FakeProvider(
        on_send=on_send, session_id=psid, auto_ack=True, speak_on_commentary=speak_on_commentary,
    )


def _request_answered(*, audio: bool, answer: str = ANSWER):
    """R, then the model's direct answer to it (text, and audio if `audio`)."""

    def start(p):
        p.push(H.user_delta(REQUEST, 0, 900))
        p.push(H.out_text(answer, 1000, 1600))
        if audio:
            p.push(H.out_audio(pcm(0)))
    return start


def _reask_contents(provider) -> list[str]:
    return [
        str(e.get("content") or "") for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-reask-")
    ]


def _results(client) -> list[tuple]:
    return [(f["user_turn_id"], f["outcome"], f.get("conditional")) for f in client.of("reask_result")]


def _epoch(client) -> str:
    """The id of the newest output epoch the phone was sent (text or audio)."""

    ids = [
        str(f.get("assistant_turn_id") or f.get("response_id") or "")
        for f in client.frames if f.get("type") in {"response_text", "audio_delta"}
    ]
    ids = [i for i in ids if i]
    return ids[-1] if ids else ""


def _idle(eid, played_ms=1800):
    return {"type": "playback_idle", "response_id": eid, "item_id": eid, "played_ms": played_ms}


def _cut(eid, played_ms, **extra):
    return {"type": "interrupt", "response_id": eid, "item_id": eid, "played_ms": played_ms, **extra}


def _failed(eid):
    return {"type": "playback_failed", "response_id": eid, "item_id": eid,
            "played_ms": 0, "reason": "decode_error"}


def _sha(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return ""


def _db(name: str) -> str:
    return f"db-nr4-{name}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"


def _record(name: str, what: str, sockets: list[tuple[str, list]], relay: dict,
            *, probe_cut: bool = False) -> None:
    """Cross-layer: written ONLY under NO_RECEIPT_R4_OUT (never otherwise).

    `probe_cut`: the scenario's reask was a PROBE of what the relay would say
    (the app must never ask it — e.g. after a heard answer): the tape stops
    before that reask and the verdict it got is kept as `relay.probe`."""

    path = os.environ.get("NO_RECEIPT_R4_OUT")
    if not path:
        return
    tapes = []
    for psid, tape in sockets:
        events = list(tape)
        if probe_cut:
            for i, ev in enumerate(events):
                out = ev.get("out") or {}
                if out.get("type") == "inject_text":
                    events = events[:i]
                    break
        tapes.append({"provider_session_id": psid, "tape": events})
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    data["_what"] = (
        "Supervisor 18:16 no-receipt cross-layer tapes: the REAL relay's frames to the phone (in), "
        "the phone's frames to the relay (out) and frames sent after a drop (undelivered), per "
        "socket, in relay order, plus the relay's ACTUAL reask verdicts. Written by "
        "backend/tests/test_live_no_receipt_r4.py under NO_RECEIPT_R4_OUT; replayed by "
        "scripts/check-voice-no-receipt-r4.js through the real useRealtimeVoice hook."
    )
    data["_relay"] = {
        "tree": os.path.abspath(os.getcwd()),
        "live_voice_protocol_sha256": _sha(os.path.abspath(live.__file__)),
        "test_file_sha256": _sha(os.path.abspath(__file__)),
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    data["features"] = list(APP_FEATURES)
    data.setdefault("scenarios", {})[name] = {"what": what, "sockets": tapes, "relay": relay}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


async def _answered_then(
    monkeypatch, *, audio: bool, receipt=None, features=APP_FEATURES, reask: bool = True,
    after=None, db: str, gap_ms=None, reask_at_once: bool = False, think=None,
):
    """ONE socket: R, the model's direct answer (text[, audio]); `receipt(eid)`
    is the phone's frame for that epoch (None: no receipt at all); after the
    relay's own gap retirement (unless `reask_at_once`) R is re-asked by id
    (by text for a client without `reask_turns`); `after(phone, eid, rid)`
    runs next.  Returns (phone, provider, tape, facts)."""

    overrides = {} if gap_ms is None else {"voice_live_output_epoch_gap_ms": gap_ms}
    H.fast_clocks(monkeypatch, **overrides)
    H.patch_relay(monkeypatch, **({"think": think} if think else {}))
    tape: list = []
    facts: dict = {}

    async def script():
        assert await _wait(lambda: _finals(phone) and phone.of("response_text"))
        if audio:
            assert await _wait(lambda: phone.of("audio_delta"))
        eid = _epoch(phone)
        rid = _finals(phone)[0]["turn_id"]
        facts.update(eid=eid, rid=rid)
        if receipt is not None:
            phone.push_frame(receipt(eid))
            await asyncio.sleep(0.3)
        elif not reask_at_once:
            await asyncio.sleep(0.5)      # the relay's own gap timer retires it; no receipt
        if reask:
            if features is not None and "reask_turns" in features:
                phone.reask(rid)
                assert await _wait(lambda: phone.of("reask_result"))
            else:
                phone.push_frame({"type": "inject_text", "text": REQUEST})
                await asyncio.sleep(0.3)
        if after is not None:
            await after(phone, eid, rid)

    phone = Phone([_config(features), script, 0.3, {"type": "stop"}], tape=tape)
    provider = _provider(_request_answered(audio=audio))
    await H.run_relay(phone, provider, timeout=8, db_session_id=db)
    return phone, provider, tape, facts


async def _dropped_then_reask(
    monkeypatch, *, audio: bool, receipt=None, features=APP_FEATURES, db: str,
    gap_ms=None, reask: bool = True,
):
    """TWO sockets on one day session: socket 1 — R, the direct answer,
    `receipt(eid)` (or none), then the socket DIES; socket 2 (a new provider
    session) — R re-asked (by id; by text without `reask_turns`)."""

    overrides = {} if gap_ms is None else {"voice_live_output_epoch_gap_ms": gap_ms}
    H.fast_clocks(monkeypatch, **overrides)
    H.patch_relay(monkeypatch)
    tape1: list = []
    tape2: list = []
    facts: dict = {}

    async def first():
        assert await _wait(lambda: _finals(c1) and c1.of("response_text"))
        if audio:
            assert await _wait(lambda: c1.of("audio_delta"))
        eid = _epoch(c1)
        facts.update(eid=eid, rid=_finals(c1)[0]["turn_id"])
        if receipt is not None:
            c1.push_frame(receipt(eid))
            await asyncio.sleep(0.3)
        elif gap_ms is None:
            await asyncio.sleep(0.5)      # gap-retired, no receipt
        c1.push_frame(DROP)

    c1 = Phone([_config(features), first], tape=tape1)
    await H.run_relay(c1, _provider(_request_answered(audio=audio)), timeout=6, db_session_id=db)

    async def second():
        await asyncio.sleep(0.3)
        if not reask:
            return
        if features is not None and "reask_turns" in features:
            c2.reask(facts["rid"])
            assert await _wait(lambda: c2.of("reask_result"))
        else:
            c2.push_frame({"type": "inject_text", "text": REQUEST})
            await asyncio.sleep(0.3)

    c2 = Phone([_config(features), second, 0.3, {"type": "stop"}], tape=tape2)
    p2 = _provider(psid="live-psid-nr4-2")
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    return c1, c2, p2, tape1, tape2, facts


def _assert_unconfirmed(content: str, words: str = REQUEST) -> None:
    """The one instruction for UNKNOWN: claims neither, asks for a recap or a
    question, forbids REDOING work.

    Pin changed by R2 addendum 5 A7 (no-receipt verifier, test_verify_nr4.py::
    test_v_promise_only_answer_instruction_allows_the_work): this line is only
    issued after the task fences, so the relay KNOWS no work was started for
    the turn.  The old "Don't redo any action or start new work for it" forbade
    the work a promise-only answer still owed; the line now forbids redoing
    what was done and allows answering or delegating a promise-only answer."""

    assert f"«{words}»" in content, content
    assert UNCONFIRMED_MARK in content, content
    assert "Don't redo any action you already took for it" in content, content
    assert "start new work" not in content, content
    assert "delegate it if it needs work" in content, content
    assert UNHEARD_MARK not in content and "No reply" not in content, content


def _assert_unheard(content: str, words: str = REQUEST) -> None:
    assert f"«{words}»" in content and UNHEARD_MARK in content, content
    assert UNCONFIRMED_MARK not in content, content


# ══════════════════════════════════════════════════════════════════════
# the receipt rules themselves (controls: unchanged, pass on both trees)
# ══════════════════════════════════════════════════════════════════════

def _closed(**kw) -> "live.ClosedEpoch":
    base = {"epoch": 1, "output_id": "live-out:1", "text": ANSWER, "interrupted": False}
    base.update(kw)
    return live.ClosedEpoch(**base)


@pytest.mark.parametrize(("closed", "expected"), [
    (_closed(retire_reason="playback_idle", played_ms=1800), True),                 # drained
    (_closed(interrupted=True, retire_reason="interrupt", played_ms=1200), True),   # clipped
    (_closed(heard_verified="There is one"), True),                                 # validated words
    (_closed(heard_verified=""), False),                                            # validated EMPTY
    (_closed(interrupted=True, retire_reason="interrupt", played_ms=0), False),      # cut before audio
    (_closed(interrupted=True, retire_reason="interrupt", played_ms=None), False),
    (_closed(interrupted=True, retire_reason="playback_failed", played_ms=400), False),
    (_closed(retire_reason="gap"), None),                                           # no receipt
    (_closed(retire_reason="new_output"), None),
], ids=["drained", "clipped", "heard_words", "empty_receipt", "cut_0", "cut_none",
        "playback_failed", "gap_no_receipt", "new_output_no_receipt"])
def test_control_the_existing_receipt_rules_are_unchanged(closed, expected):
    """CONTROL: no new threshold — `epoch_heard` reads the receipts exactly
    as before (None = no receipt).  What changed is only what a recovery path
    does with None."""

    assert live._LiveSession.epoch_heard(closed) is expected


# ══════════════════════════════════════════════════════════════════════
# UNKNOWN — no receipt: conditional, never answered, never "did not reach"
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_text_only_answer_with_no_receipt_is_a_conditional_accept(monkeypatch):
    """The supervisor's blocker: R-parented text-only answer, no receipt; the
    by-id reask is `accepted` + `conditional: true` (old: `answered`, which
    silently cleared the app's recovery).  One instruction, claiming neither
    hearing nor non-hearing; nothing delegated, nothing redone."""

    calls: list = []

    async def think(*_a, **_k):
        calls.append(1)
        return "An answer.", "m"

    phone, provider, tape, facts = await _answered_then(
        monkeypatch, audio=False, db=_db("text-only"), think=think,
    )
    _record("text_only", "R answered text-only, no playback, no receipt; the app re-asks R by id",
            [(H.PSID, tape)], {"verdicts": phone.of("reask_result"),
                               "reask_instructions": _reask_contents(provider)})
    assert _results(phone) == [(facts["rid"], "accepted", True)]
    contents = _reask_contents(provider)
    assert len(contents) == 1
    _assert_unconfirmed(contents[0])
    assert calls == [] and phone.of("delegation") == []


@pytest.mark.asyncio
async def test_queued_audio_with_no_playback_callback_is_a_conditional_accept(monkeypatch):
    """Audio queued on the phone, native playback never confirmed any of it,
    no receipt: UNKNOWN → conditional (old: `answered`)."""

    phone, provider, tape, facts = await _answered_then(
        monkeypatch, audio=True, db=_db("queued"),
    )
    _record("queued_no_callback", "R answered with audio the phone never confirmed playing; no receipt",
            [(H.PSID, tape)], {"verdicts": phone.of("reask_result"),
                               "reask_instructions": _reask_contents(provider)})
    assert _results(phone) == [(facts["rid"], "accepted", True)]
    contents = _reask_contents(provider)
    assert len(contents) == 1
    _assert_unconfirmed(contents[0])


@pytest.mark.asyncio
async def test_a_persian_unconfirmed_reask_is_in_persian(monkeypatch):
    """The unconfirmed line exists in Persian too, with the same three
    properties (neither claim, recap-or-ask, nothing redone)."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    words = "یه دندونپزشک نزدیک ایستگاه یونیون پیدا کن"
    tape: list = []

    async def script():
        assert await _wait(lambda: _finals(phone) and phone.of("response_text"))
        await asyncio.sleep(0.5)
        phone.reask(_finals(phone)[0]["turn_id"], words)
        assert await _wait(lambda: phone.of("reask_result"))

    def start(p):
        p.push(H.user_delta(words, 0, 900))
        p.push(H.out_text("یکی تو خیابون فرانت هست.", 1000, 1600))

    phone = Phone([_config(APP_FEATURES), script, 0.3, {"type": "stop"}], tape=tape)
    provider = _provider(start)
    await H.run_relay(phone, provider, timeout=8, db_session_id=_db("fa"))
    assert [(o, c) for _t, o, c in _results(phone)] == [("accepted", True)]
    content = _reask_contents(provider)[0]
    assert words in content and "هیچ‌وقت تأیید نکرد" in content and "دوباره" in content
    assert "به او نرسید" not in content


@pytest.mark.asyncio
async def test_a_late_playback_receipt_after_the_conditional_accept_changes_nothing(monkeypatch):
    """The phone confirms playback AFTER the conditional accept (a long
    answer that was still queued): nothing is asked again, nothing redone —
    exactly one reask instruction, no task, no second verdict."""

    async def late(phone, eid, _rid):
        await asyncio.sleep(0.2)
        phone.push_frame(_idle(eid))
        await asyncio.sleep(0.3)

    phone, provider, tape, facts = await _answered_then(
        monkeypatch, audio=True, after=late, db=_db("late"),
    )
    _record("late_playback", "R's audio played back late — after the conditional accept",
            [(H.PSID, tape)], {"verdicts": phone.of("reask_result"),
                               "reask_instructions": _reask_contents(provider)})
    assert _results(phone) == [(facts["rid"], "accepted", True)]
    assert len(_reask_contents(provider)) == 1
    assert phone.of("delegation") == []


@pytest.mark.asyncio
async def test_a_second_ask_after_a_late_receipt_is_a_duplicate(monkeypatch):
    """Once per turn is unchanged: a second by-id ask after the late receipt
    is `duplicate`, never a second instruction."""

    async def late_then_again(phone, eid, rid):
        phone.push_frame(_idle(eid))
        await asyncio.sleep(0.3)
        phone.reask(rid)
        assert await _wait(lambda: len(phone.of("reask_result")) >= 2)

    phone, provider, _tape, facts = await _answered_then(
        monkeypatch, audio=True, after=late_then_again, db=_db("late-dup"),
    )
    assert [(o, c) for _t, o, c in _results(phone)] == [("accepted", True), ("duplicate", None)]
    assert len(_reask_contents(provider)) == 1


@pytest.mark.asyncio
async def test_a_second_ask_is_a_duplicate(monkeypatch):
    async def again(phone, _eid, rid):
        phone.reask(rid)
        assert await _wait(lambda: len(phone.of("reask_result")) >= 2)

    phone, provider, _tape, _facts = await _answered_then(
        monkeypatch, audio=False, after=again, db=_db("dup"),
    )
    assert [(o, c) for _t, o, c in _results(phone)] == [("accepted", True), ("duplicate", None)]
    assert len(_reask_contents(provider)) == 1


@pytest.mark.asyncio
async def test_output_still_streaming_is_in_flight(monkeypatch):
    """UNKNOWN because the answer is still being produced NOW (the epoch is
    open): `in_flight` — it is on its way, never re-asked over itself."""

    phone, provider, _tape, facts = await _answered_then(
        monkeypatch, audio=True, db=_db("streaming"), gap_ms=5000, reask_at_once=True,
    )
    assert _results(phone) == [(facts["rid"], "in_flight", None)]
    assert _reask_contents(provider) == []


@pytest.mark.asyncio
async def test_a_newer_request_makes_it_stale_and_is_itself_plain(monkeypatch):
    """The staleness fence still comes first: after a newer real request, R
    is `stale` whatever its receipt said; the newer request (no output at
    all) is a plain accept."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    tape: list = []
    box: dict = {}

    def start(p):
        box["p"] = p
        _request_answered(audio=True)(p)

    async def script():
        assert await _wait(lambda: _finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.5)
        box["p"].push(H.user_delta(NEWER, 5000, 5800))
        assert await _wait(lambda: len(_finals(phone)) >= 2)
        r, n = _finals(phone)[:2]
        phone.reask(r["turn_id"])
        assert await _wait(lambda: phone.of("reask_result"))
        phone.reask(n["turn_id"], NEWER)
        assert await _wait(lambda: len(phone.of("reask_result")) >= 2)

    phone = Phone([_config(APP_FEATURES), script, 0.3, {"type": "stop"}], tape=tape)
    provider = _provider(start)
    await H.run_relay(phone, provider, timeout=8, db_session_id=_db("stale"))
    assert [(o, c) for _t, o, c in _results(phone)] == [("stale", None), ("accepted", None)]
    contents = _reask_contents(provider)
    assert len(contents) == 1 and contents[0].startswith(PLAIN_MARK) and NEWER in contents[0]


@pytest.mark.asyncio
async def test_task_fences_decide_before_any_receipt(monkeypatch):
    """R consumed by a task: while it runs → `in_flight` (task); once it has
    finished and been delivered → `answered` (task) — the acknowledgement
    before the delegation had no receipt, and that changes nothing: a task
    fence is explicit evidence, and no side effect is ever replayed."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[str] = []

    async def think(_user_id, task, _session_id, **kw):
        calls.append(kw.get("display_request") or task)
        await asyncio.sleep(0.8)
        return "Union Dental on Front Street.", "m"

    H.patch_relay(monkeypatch, think=think)
    tape: list = []

    def start(p):
        p.push(H.user_delta(REQUEST, 0, 900))
        p.push(H.out_text("Let me look that up.", 1000, 1400))
        p.push(H.delegation("d1", 1450))

    async def script():
        assert await _wait(lambda: any(f.get("state") == "running" for f in phone.of("delegation")))
        rid = _finals(phone)[0]["turn_id"]
        phone.reask(rid)
        assert await _wait(lambda: phone.of("reask_result"))
        assert await _wait(
            lambda: any(f.get("phase") == "completed" for f in phone.of("delegation")), 5.0,
        )
        await asyncio.sleep(0.8)
        phone.reask(rid)
        assert await _wait(lambda: len(phone.of("reask_result")) >= 2)

    phone = Phone([_config(APP_FEATURES), script, 0.3, {"type": "stop"}], tape=tape)
    provider = _provider(start, speak_on_commentary=True)
    await H.run_relay(phone, provider, timeout=10, db_session_id=_db("task"))
    assert [(o, c) for _t, o, c in _results(phone)] == [("in_flight", None), ("answered", None)]
    assert _reask_contents(provider) == []
    assert calls == [REQUEST]


# ══════════════════════════════════════════════════════════════════════
# HEARD — a positive receipt answers it (controls)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [
    lambda eid: _idle(eid),
    lambda eid: _cut(eid, 1200),
    lambda eid: _cut(eid, 1200, heard_text="There is one"),
], ids=["drained", "clipped", "clipped_with_heard_text"])
async def test_control_a_heard_answer_is_answered(monkeypatch, receipt):
    phone, provider, tape, facts = await _answered_then(
        monkeypatch, audio=True, receipt=receipt, db=_db(f"heard-{id(receipt)}"),
    )
    assert _results(phone) == [(facts["rid"], "answered", None)]
    assert _reask_contents(provider) == []


@pytest.mark.asyncio
async def test_control_a_drained_answer_is_covered_and_never_asked(monkeypatch):
    """The cross-layer shape of HEARD: the phone drains the answer; the app
    must never ask.  The relay's verdict for a (probe) ask is `answered`."""

    phone, provider, tape, facts = await _answered_then(
        monkeypatch, audio=True, receipt=_idle, db=_db("heard-x"),
    )
    _record("heard_drained", "R's answer played to the end (playback_idle); the app never asks",
            [(H.PSID, tape)], {"probe": phone.of("reask_result")}, probe_cut=True)
    assert _results(phone) == [(facts["rid"], "answered", None)]


@pytest.mark.asyncio
async def test_control_a_clipped_answer_is_covered_and_never_asked(monkeypatch):
    """The cross-layer shape of a CLIPPED answer: some of it played, then the
    caller cut it — heard for what was heard (today's rule); the app never
    asks, and the relay's verdict for a (probe) ask is `answered`."""

    phone, provider, tape, facts = await _answered_then(
        monkeypatch, audio=True, receipt=lambda eid: _cut(eid, 1200), db=_db("clipped-x"),
    )
    _record("heard_clipped", "R's answer cut by the caller after some of it played; the app never asks",
            [(H.PSID, tape)], {"probe": phone.of("reask_result")}, probe_cut=True)
    assert _results(phone) == [(facts["rid"], "answered", None)]


# ══════════════════════════════════════════════════════════════════════
# UNHEARD — positive non-hearing: plain, truthful
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [
    lambda eid: _cut(eid, 900, heard_text=""),
    lambda eid: _cut(eid, 0),
    lambda eid: _cut(eid, 0, heard_text=""),
    _failed,
], ids=["validated_empty_receipt", "cut_before_audio", "cut_before_audio_app_shape", "playback_failed"])
async def test_control_a_positively_unheard_answer_is_a_plain_truthful_accept(monkeypatch, receipt):
    """Positive non-hearing is NOT unknown: a plain accept (no `conditional`)
    with the truthful "your answer … did not reach them" (§3.2 pin)."""

    phone, provider, tape, facts = await _answered_then(
        monkeypatch, audio=True, receipt=receipt, db=_db(f"unheard-{id(receipt)}"),
    )
    assert _results(phone) == [(facts["rid"], "accepted", None)]
    contents = _reask_contents(provider)
    assert len(contents) == 1
    _assert_unheard(contents[0])


@pytest.mark.asyncio
async def test_control_the_app_cut_before_audio_takes_the_plain_path(monkeypatch):
    """The cross-layer shape of UNHEARD: the caller taps before any audio
    (the app's interrupt: played_ms 0, heard_text "")."""

    phone, provider, tape, facts = await _answered_then(
        monkeypatch, audio=True, receipt=lambda eid: _cut(eid, 0, heard_text=""),
        db=_db("unheard-x"),
    )
    _record("unheard_cut", "R's answer cut before any audio; the app re-asks R by id",
            [(H.PSID, tape)], {"verdicts": phone.of("reask_result"),
                               "reask_instructions": _reask_contents(provider)})
    assert _results(phone) == [(facts["rid"], "accepted", None)]
    _assert_unheard(_reask_contents(provider)[0])


# ══════════════════════════════════════════════════════════════════════
# precedence: R's own evidence outranks a reply to a later phatic turn
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("p_receipt", ["idle", "cut"])
async def test_rs_own_unconfirmed_answer_outranks_a_reply_to_hello(monkeypatch, p_receipt):
    """R answered (no receipt), then "hello?" and a reply to it (heard, or
    cut before audio): R is conditional with R's OWN unconfirmed wording —
    explicit evidence about R wins over the §3.3 P-reply path, and the model
    is never told to "delegate it if it needs work" for an answer it gave."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    tape: list = []
    box: dict = {}

    def start(p):
        box["p"] = p
        _request_answered(audio=True)(p)

    async def script():
        assert await _wait(lambda: _finals(phone) and phone.of("audio_delta"))
        await asyncio.sleep(0.5)
        box["p"].push(H.user_delta("hello?", 6000, 6400))
        assert await _wait(lambda: len(_finals(phone)) >= 2)
        box["p"].push(H.out_text("Yes, I'm here.", 6600, 7000))
        box["p"].push(H.out_audio(pcm(1)))
        await asyncio.sleep(0.2)
        eid = _epoch(phone)
        phone.push_frame(_idle(eid) if p_receipt == "idle" else _cut(eid, 0))
        await asyncio.sleep(0.4)
        phone.reask(_finals(phone)[0]["turn_id"])
        assert await _wait(lambda: phone.of("reask_result"))

    phone = Phone([_config(APP_FEATURES), script, 0.3, {"type": "stop"}], tape=tape)
    provider = _provider(start)
    await H.run_relay(phone, provider, timeout=8, db_session_id=_db(f"prec-{p_receipt}"))
    assert [(o, c) for _t, o, c in _results(phone)] == [("accepted", True)]
    _assert_unconfirmed(_reask_contents(provider)[0])


# ══════════════════════════════════════════════════════════════════════
# reconnect — the carried record keeps the three answers apart
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("audio", [True, False], ids=["audio", "text_only"])
async def test_output_dropped_with_the_socket_without_a_receipt_is_carried_conditional(
    monkeypatch, audio,
):
    c1, c2, p2, tape1, tape2, facts = await _dropped_then_reask(
        monkeypatch, audio=audio, db=_db(f"carry-unknown-{audio}"),
    )
    _record(
        "reconnect_no_receipt" if audio else "reconnect_text_only",
        "R answered with no receipt, then the socket dropped; R re-asked by id on the repaired socket",
        [(H.PSID, tape1), ("live-psid-nr4-2", tape2)],
        {"verdicts": c2.of("reask_result"), "reask_instructions": _reask_contents(p2)},
    )
    assert _results(c2) == [(facts["rid"], "accepted", True)]
    contents = _reask_contents(p2)
    assert len(contents) == 1
    _assert_unconfirmed(contents[0])


@pytest.mark.asyncio
async def test_an_epoch_still_open_at_the_drop_is_carried_conditional(monkeypatch):
    """The answer was still streaming when the socket died: no receipt could
    ever come — UNKNOWN, carried conditional (old: presumed heard, dropped)."""

    c1, c2, p2, tape1, tape2, facts = await _dropped_then_reask(
        monkeypatch, audio=True, db=_db("carry-open"), gap_ms=5000,
    )
    _record(
        "reconnect_open_epoch", "the socket died while R's answer was still streaming",
        [(H.PSID, tape1), ("live-psid-nr4-2", tape2)],
        {"verdicts": c2.of("reask_result"), "reask_instructions": _reask_contents(p2)},
    )
    assert _results(c2) == [(facts["rid"], "accepted", True)]
    _assert_unconfirmed(_reask_contents(p2)[0])


@pytest.mark.asyncio
async def test_control_a_positively_unheard_answer_is_carried_plain(monkeypatch):
    c1, c2, p2, tape1, tape2, facts = await _dropped_then_reask(
        monkeypatch, audio=True, receipt=lambda eid: _cut(eid, 0, heard_text=""),
        db=_db("carry-unheard"),
    )
    _record(
        "reconnect_unheard", "R's answer cut before any audio, then the socket dropped",
        [(H.PSID, tape1), ("live-psid-nr4-2", tape2)],
        {"verdicts": c2.of("reask_result"), "reask_instructions": _reask_contents(p2)},
    )
    assert _results(c2) == [(facts["rid"], "accepted", None)]
    _assert_unheard(_reask_contents(p2)[0])


@pytest.mark.asyncio
async def test_control_a_heard_answer_is_not_carried(monkeypatch):
    c1, c2, p2, tape1, tape2, facts = await _dropped_then_reask(
        monkeypatch, audio=True, receipt=_idle, db=_db("carry-heard"),
    )
    _record(
        "reconnect_heard", "R's answer heard to the end, then the socket dropped; the app never asks",
        [(H.PSID, tape1), ("live-psid-nr4-2", tape2)],
        {"probe": c2.of("reask_result")}, probe_cut=True,
    )
    assert _results(c2) == [(facts["rid"], "unknown", None)]
    assert _reask_contents(p2) == []


# ══════════════════════════════════════════════════════════════════════
# feature negotiation: receipts or not
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("features", [None, WEB, WEB + ["reask_turns"]],
                         ids=["build129_no_features", "web", "no_receipts_with_reask_turns"])
async def test_control_a_client_without_receipts_keeps_todays_semantics(monkeypatch, features):
    """The legacy gate: a client that never negotiated playback receipts
    (neither `playback_frames` nor `heard_text`) was never asked for them,
    so their absence proves nothing there — output that reached it answers
    the turn, exactly as before (no instruction; `answered` on the wire
    where a verdict is sent at all)."""

    phone, provider, _tape, facts = await _answered_then(
        monkeypatch, audio=True, features=features, db=_db(f"legacy-{features}"),
    )
    assert _reask_contents(provider) == []
    if features is not None and "reask_turns" in features:
        assert _results(phone) == [(facts["rid"], "answered", None)]
    else:
        assert phone.of("reask_result") == []


@pytest.mark.asyncio
async def test_control_a_client_without_receipts_is_not_carried(monkeypatch):
    c1, c2, p2, _t1, _t2, _facts = await _dropped_then_reask(
        monkeypatch, audio=True, features=None, db=_db("legacy-carry"),
    )
    assert _reask_contents(p2) == []


@pytest.mark.asyncio
async def test_tf132_gets_the_unconfirmed_recovery_and_no_new_frame(monkeypatch):
    """TF132 sends receipts, so the tri-state applies to it; it negotiated no
    `reask_turns`, so it gets no `reask_result` (and no new key) — only the
    instruction for its text-only replay."""

    phone, provider, _tape, _facts = await _answered_then(
        monkeypatch, audio=True, features=TF132, db=_db("tf132"),
    )
    assert phone.of("reask_result") == []
    contents = _reask_contents(provider)
    assert len(contents) == 1
    _assert_unconfirmed(contents[0])


# ══════════════════════════════════════════════════════════════════════
# the status path (`unanswered_before`): UNKNOWN never starts new work
# ══════════════════════════════════════════════════════════════════════

async def _check_in_after(monkeypatch, receipt, db):
    """Research runs; the caller asks R; the model answers R directly;
    `receipt(eid)` (or none); then "what happened?" — does the check-in
    dispatch R as NEW work (only when R's answer was provably unheard)?"""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    displays: list[str] = []
    started = asyncio.Event()
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        if len(displays) == 1:
            started.set()
            await asyncio.sleep(2.0)
            return "Robotics list.", "m"
        return "Union Dental.", "m"

    H.patch_relay(monkeypatch, think=think)

    def start(p):
        box["p"] = p
        p.push(H.user_delta(RESEARCH, 0, 900))
        p.push(H.delegation("d1", 950))

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta(REQUEST, 3000, 3900))
        assert await _wait(lambda: len(_finals(phone)) >= 2)
        p.push(H.out_text(ANSWER, 4000, 4600))
        p.push(H.out_audio(pcm(2)))
        assert await _wait(lambda: phone.of("audio_delta"))
        if receipt is not None:
            phone.push_frame(receipt(_epoch(phone)))
        await asyncio.sleep(0.4)
        p.push(H.user_delta("what happened?", 15000, 15500))
        p.push(H.delegation("d2", 15550))
        await _wait(lambda: any(
            f.get("phase") in {"completed", "failed"} and f.get("delegation_id") == "d1"
            for f in phone.of("delegation")
        ), 5.0)
        await asyncio.sleep(0.3)

    phone = Phone([_config(APP_FEATURES), script, 0.3, {"type": "stop"}], tape=[])
    await H.run_relay(phone, _provider(start), timeout=12, db_session_id=db)
    return displays


@pytest.mark.asyncio
async def test_control_a_check_in_never_dispatches_an_unconfirmed_answer_as_new_work(monkeypatch):
    """Decided explicitly (heard_rule_sites): the status path's §F
    "unanswered" walk DISPATCHES work, so a missing receipt must not
    fabricate non-hearing there — R is not re-run by a check-in; it stays
    recoverable through its by-id reask (conditional).  Unchanged."""

    displays = await _check_in_after(monkeypatch, None, _db("status-unknown"))
    assert displays == [RESEARCH], displays


@pytest.mark.asyncio
async def test_control_a_check_in_dispatches_a_provably_unheard_answer(monkeypatch):
    """…while a provably unheard answer leaves R unanswered there, and the
    check-in dispatches IT (addendum 4 §3.2, unchanged)."""

    displays = await _check_in_after(
        monkeypatch, lambda eid: _cut(eid, 0, heard_text=""), _db("status-unheard"),
    )
    assert displays == [RESEARCH, REQUEST], displays
