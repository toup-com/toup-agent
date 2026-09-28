"""R2 round 4 (addendum 4): CROSS-LAYER evidence, relay half.

Binding spec: /private/tmp/toup-voice-r2-spec-addendum-4.md (§8: "relay
scenarios captured to a JSON fixture (written only under an env flag) and
replayed through the app's real turnWatch/hook in a guard").

Every scenario here runs the REAL relay (`run_live_voice_session`) against the
harness fakes, with a phone that behaves the way the shipped app does:

  * it sends the app's own `config` (the app's `LIVE_FEATURES`, verbatim) and
    `audio_ready`;
  * it PLAYS every output epoch it receives and reports it the way the app's
    native drain does (`playback_idle` with a positive `played_ms`), unless
    the scenario says the caller cut it before any audio was heard (the app's
    tap interrupt: `interrupt`, `played_ms: 0`, `heard_text: ""`);
  * it re-asks exactly the turn the app's turn watch re-asks, by id, and
    nothing else;
  * a drop is a real socket loss: frames the relay sends after it are
    recorded as UNDELIVERED (the phone never saw them), and the repair is a
    NEW provider session (distinct `session_id`) on the same day session.

With `R4_CROSS_LAYER_OUT=<path>` set, each scenario's full wire tape (relay
frames IN, phone frames OUT, undelivered frames, per socket, in the order the
relay saw them) plus the relay-side facts are written to that JSON file for
`scripts/check-voice-cross-layer-r4.js` in the app, which replays them through
the app's REAL `useRealtimeVoice` hook and `turnWatch` and checks that the app
sends the same frames at the same points and ends in the right state.  Without
the variable nothing is written.  The tape is written BEFORE the relay-side
assertions, so a run against the pre-round-4 relay (integrated-snap5) still
yields a tape the app guard can show failing end to end.

R2 round 5 (addendum 5, /private/tmp/toup-voice-r2-spec-addendum-5.md, with
the supervisor's 19:35 corrections) adds the scenarios where the two layers
decided from DIFFERENT evidence (A6): an answer with no playback receipt
(text only / audio never confirmed / dropped with the socket), a reask
crossing an OPEN turn's first words, a caller turn after a PLAIN accept, a
drop in the middle of a presence check (the A4 contract both ways, with and
without an older request), an identity-less tap after a heard reply; the A7
promise-only answer; representative B1/C3/C4 shapes; and C10 — the
confirmation applied after the new task already finished — on the FAST
(1.2 s, fx5's preserved cross-test 70773d3e…) AND the slow schedule and a
mixed answer, each with the card and the saved row compared.

Two things the phone does here that it did not do in round 4, both because
the shipped app does them:

  * ONE playback identity.  The app's `LivePlaybackLifecycle.audio` replaces
    its active identity when a newer epoch's audio arrives, so an epoch whose
    drain had not been reported yet is NEVER acknowledged: its buffer played
    (native SoundChunkPlayed — heard, for the turn watch), and the one drain
    report is the newest epoch's.  The phone records that as a LOCAL event
    (`{"local": {"type": "chunk_played", …}}`) so the replay can drive the
    same native callback; the relay never sees it.
  * A CROSSING is recorded in the phone's order.  When the phone's frame left
    before a relay frame reached it (the app's reask vs the caller's first
    settle), the relay reads it after sending that frame, but the tape puts
    the `out` BEFORE the frames it crossed (`"crossed": n`), because the app
    sent it without having seen them.
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
from app.config import settings


#: The app's `LIVE_FEATURES` (src/shared/voice/liveProtocol.ts), verbatim.
#: The app guard cross-pins this list against the module it loads.
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

UNANSWERED = "find me a dentist near union station"
RESEARCH = "research the toronto robotics faculty"
LIBRARY_Q = "what time does the union station library close"
WEATHER_Q = "what's the weather in toronto right now"
PARIS_Q = "what's the weather in paris"
PROFESSORS = "find iranian computer science professors at the university of toronto"
# PIN CHANGED (R2 addendum 6 R6-8 resolves the C11 spec conflict): 'no, book
# me a hotel in toronto' is a SELF-CONTAINED request sharing one of its three
# own words with the professors search — NONE now (no stop question).  The
# C10/C3/C4 cross-layer scenarios keep a real PARTIAL correction, as
# tests/test_live_round4.py's R4.PARTIAL: it shares university + toronto, at
# least half of its own words.  (Tapes recorded from this file change with it:
# the cross-layer owner re-records them.)
PARTIAL = "no, find a dorm at the university of toronto"
QUESTION = "I found Union Dental on Front Street. Should I book you in for tomorrow at ten?"
STATEMENT = "I found Union Dental on Front Street. It is open until six today."
QUESTION_FA = "دندونپزشکی یونیون رو پیدا کردم. برای فردا ساعت ده وقت بگیرم؟"
STOP_MARKS = ("Should I stop", "Sorry — should I stop", "رو متوقف کنم")
LIBRARY_FA = "ساعت کاری کتابخونه رو یکشنبه پیدا کن"
PHARMACY = "find me a pharmacy near union station"
OFFICE_FA = "آدرس دفترش رو بفرست"
MIXED = "yes, and search for waterloo robotics professors"
REMAINDER = "search for waterloo robotics professors"
LIBRARY_A = "The Union Station library closes at nine tonight."
#: The relay's wording marks (live_voice_protocol.py `_REASK_*_LINES`).
UNCONFIRMED_MARK = "never confirmed playing it"
UNHEARD_MARK = "did not reach them"
FRAGMENT_MARKS = ("call dropped", "تماس قطع شد")


# ══════════════════════════════════════════════════════════════════════
# the phone, the model, the tape
# ══════════════════════════════════════════════════════════════════════

def pcm(index: int = 0, samples: int = 480, amp: int = 2000) -> str:
    """20 ms of real PCM16LE at 24 kHz — the app decodes and queues it."""

    values = [int(amp * math.sin((index * samples + k) / 7)) for k in range(samples)]
    return base64.b64encode(struct.pack(f"<{samples}h", *values)).decode()


class _Drop:
    """The socket dies here (the phone lost the network)."""


DROP = _Drop()


class _Crossing:
    """A phone frame that left BEFORE the relay frames matching `crossed`
    reached the phone (see the module docstring)."""

    def __init__(self, frame: dict, crossed):
        self.frame = frame
        self.crossed = crossed


def _finals(client) -> list[dict]:
    return [f for f in client.of("transcript") if f.get("final")]


async def _wait(pred, timeout=4.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


class Phone(H.FakeClient):
    """The app, as the relay sees it — and a recorder of the exact wire.

    `play(eid, ordinal)` decides what the phone reports for an epoch:
    "idle" (heard to the end), "cut" (tapped away before any audio was
    heard) or None (the audio was queued and native playback never confirmed
    any of it — no callback, no report, ever).  The report goes out ~120 ms
    after the epoch's first audio frame, like the native drain of a short
    reply — unless a NEWER epoch's audio arrives first: the app keeps one
    playback identity, so the older epoch is then never acknowledged (its
    buffer was played — a LOCAL `chunk_played` — and only the newest drains)."""

    def __init__(self, script, *, tape: list, play=None):
        super().__init__(script)
        self.tape = tape
        self.inbox: list = []
        self._bg = None
        self.dead = False
        self.epochs: list[str] = []
        self._play = play or (lambda _eid, _n: "idle")
        #: (epoch, timer) of a drain report not sent yet.
        self._pending = None

    async def send_json(self, frame):
        await super().send_json(frame)
        self.tape.append({"undelivered" if self.dead else "in": frame})
        if self.dead or frame.get("type") != "audio_delta":
            return
        eid = str(frame.get("response_id") or "")
        if not eid or eid in self.epochs:
            return
        if self._pending is not None:
            # The app's LivePlaybackLifecycle.audio: a newer epoch replaces the
            # active identity; the older one's drain is never reported.  Its
            # buffer DID play (FIFO) — native SoundChunkPlayed — before this one.
            prev, handle = self._pending
            handle.cancel()
            self._pending = None
            self.tape.insert(len(self.tape) - 1, {"local": {"type": "chunk_played", "response_id": prev}})
        self.epochs.append(eid)
        what = self._play(eid, len(self.epochs))
        if what == "idle":
            report = {"type": "playback_idle", "response_id": eid, "item_id": eid, "played_ms": 20}
        elif what == "cut":
            report = {"type": "interrupt", "response_id": eid, "item_id": eid,
                      "played_ms": 0, "heard_text": ""}
        else:
            return
        handle = asyncio.get_running_loop().call_later(0.12, self._report, eid, report)
        if what == "idle":
            self._pending = (eid, handle)

    def _report(self, eid: str, report: dict) -> None:
        if self._pending is not None and self._pending[0] == eid:
            self._pending = None
        self.push_frame(report)

    def push_frame(self, frame) -> None:
        self.inbox.append(frame)

    def push_crossing(self, frame: dict, crossed) -> None:
        """`frame` left the phone before the relay frames matching `crossed`
        (a predicate) reached it; the relay reads it after sending them."""

        self.inbox.append(_Crossing(frame, crossed))

    def tap_without_identity(self, played_ms=0) -> None:
        """The app's tap with NO active playback identity (nothing of any
        open epoch is playing on the phone): an `interrupt` that names no
        epoch.  This build sends played_ms 0 (addendum 5 A5); a shipped
        build sent its session-cumulative counter."""

        frame: dict = {"type": "interrupt"}
        if played_ms is not None:
            frame["played_ms"] = played_ms
        self.push_frame(frame)

    def _deliver(self, item):
        if item is DROP:
            from fastapi import WebSocketDisconnect

            self.dead = True
            self.tape.append({"out": {"type": "__drop__"}})
            raise WebSocketDisconnect(code=1006)
        if isinstance(item, _Crossing):
            at = next((i for i, ev in enumerate(self.tape) if "in" in ev and item.crossed(ev["in"])), None)
            if at is None:
                self.tape.append({"out": item.frame})
            else:
                self.tape.insert(at, {"out": item.frame, "crossed": len(self.tape) - at})
            return json.dumps(item.frame)
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
                if callable(item):
                    item()
                    continue
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

    def reask(self, turn: dict) -> None:
        """What the app's turn watch sends: the turn's words, by id."""

        self.push_frame({
            "type": "inject_text", "text": turn["text"], "reason": "no_response",
            "reask_of_user_turn_id": turn["turn_id"],
        })

    def reask_id(self, turn_id: str, text: str) -> None:
        self.push_frame({
            "type": "inject_text", "text": text, "reason": "no_response",
            "reask_of_user_turn_id": turn_id,
        })

    def reask_crossing(self, turn: dict, crossed) -> None:
        """The app's reask of `turn`, sent before the frames matching
        `crossed` reached the phone (addendum 4 §4.1 / addendum 5 A3)."""

        self.push_crossing({
            "type": "inject_text", "text": turn["text"], "reason": "no_response",
            "reask_of_user_turn_id": turn["turn_id"],
        }, crossed)


def _config():
    return {"type": "config", "protocol": "live1", "voice": "shimmer", "features": list(APP_FEATURES)}


class Model:
    """GPT-Live on one socket: a provider clock, scripted caller speech, and
    a model that SAYS what the relay asks it to (commentary it speaks as its
    own epoch, a reask it answers or not).

    `answer_reask(event)` returns what the model does with an accepted
    reask's instruction: words to say, a callable run with the model (to
    delegate, say…), or nothing.  `commentary_tail(content)` returns words
    the model adds IN THE SAME EPOCH after a relay line it speaks."""

    def __init__(self, psid: str, *, start=None, speak_commentary=True, answer_reask=None,
                 commentary_tail=None):
        self.at = 0
        self._start = start
        self._speak_commentary = speak_commentary
        self._answer_reask = answer_reask
        self._commentary_tail = commentary_tail
        self.appends: list[dict] = []
        self.chunk = 0
        self.provider = H.FakeProvider(on_send=self._on_send, session_id=psid, auto_ack=True)

    def user(self, text: str, *, gap: int = 1500, dur: int = 600) -> None:
        self.at += gap
        self.provider.push(H.user_delta(text, self.at, self.at + dur))
        self.at += dur

    def say(self, text: str, *, gap: int = 200, dur: int = 900, audio: bool = True, tail: str = "") -> None:
        self.at += gap
        self.provider.push(H.out_text(text, self.at, self.at + dur))
        if tail:
            # The model goes on in the SAME output epoch (no receipt between).
            self.provider.push(H.out_text(tail, self.at + dur, self.at + dur + 900))
            self.at += 900
        if audio:
            self.provider.push(H.out_audio(pcm(self.chunk)))
            self.chunk += 1
        self.at += dur

    def delegate(self, did: str) -> None:
        self.provider.push(H.delegation(did, self.at + 50))

    def _on_send(self, p, e):
        kind = e.get("type")
        if kind == "session.start":
            if self._start is not None:
                self._start(self)
        elif kind == "session.commentary.append":
            content = str(e.get("content") or "")
            self.appends.append({"kind": "commentary", "content": content})
            if self._speak_commentary:
                tail = self._commentary_tail(content) if self._commentary_tail else ""
                self.say(content, tail=tail or "")
        elif kind == "session.instructions.append":
            event_id = str(e.get("event_id") or "")
            self.appends.append({"kind": "instructions", "id": event_id, "content": str(e.get("content") or "")})
            if event_id.startswith("toup-live-reask-") and self._answer_reask is not None:
                reply = self._answer_reask(e)
                if callable(reply):
                    reply(self)
                elif reply:
                    self.say(reply)
            elif event_id.startswith("toup-live-status-"):
                self.say("Here is what I found earlier.")
        elif kind == "session.close":
            p.push(H.closed())

    def reask_contents(self) -> list[str]:
        return [a["content"] for a in self.appends
                if a["kind"] == "instructions" and a["id"].startswith("toup-live-reask-")]

    def commentary(self) -> list[str]:
        return [a["content"] for a in self.appends if a["kind"] == "commentary"]


def _sha(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return ""


def _record(name: str, what: str, sockets: list[tuple[str, list]], relay: dict) -> None:
    """Cross-layer: written ONLY under R4_CROSS_LAYER_OUT (never otherwise)."""

    path = os.environ.get("R4_CROSS_LAYER_OUT")
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    data["_what"] = (
        "R2 rounds 4-5 (addenda 4-5) cross-layer tapes: the REAL relay's frames to the phone (in), the "
        "phone's frames to the relay (out; a crossing is placed where the phone sent it, 'crossed': n), "
        "frames sent after a drop (undelivered) and the phone's own native playback events (local), per "
        "socket, in relay order. Written by backend/tests/test_live_cross_layer_r4.py under "
        "R4_CROSS_LAYER_OUT; replayed by scripts/check-voice-cross-layer-r4.js."
    )
    data["_relay"] = {
        "tree": os.path.abspath(os.getcwd()),
        "live_voice_protocol_sha256": _sha("app/services/live_voice_protocol.py"),
        "test_file_sha256": _sha(os.path.abspath(__file__)),
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    data["features"] = list(APP_FEATURES)
    data.setdefault("scenarios", {})[name] = {
        "what": what,
        "sockets": [{"provider_session_id": psid, "tape": tape} for psid, tape in sockets],
        "relay": relay,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1, sort_keys=False, default=str)


def _db(name: str) -> str:
    return f"db-x4-{name}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"


def _saved_rows(saves: list[dict], ref: str) -> list[dict]:
    """What the relay PERSISTED for one assistant row (the saved history the
    day thread is built from), in write order: the payload it sent, and
    `stored` — that payload's voice provenance after the messages route's
    own allowlist (`app.api.sessions._clean_voice`), i.e. what the row keeps."""

    try:
        from app.api.sessions import _clean_voice
    except ImportError:          # a tree without the allowlist helper
        def _clean_voice(voice):
            return voice
    return [
        {
            "ref": s.get("assistant_ref"),
            "revision": s.get("assistant_revision"),
            "text": s.get("assistant_text"),
            "voice": dict(s.get("assistant_voice") or {}),
            "stored": _clean_voice(dict(s.get("assistant_voice") or {})) or {},
        }
        for s in saves if s.get("assistant_ref") == ref
    ]


def _results(client) -> list[tuple]:
    return [(f["user_turn_id"], f["outcome"], f.get("conditional")) for f in client.of("reask_result")]


def _frames_for(client, did):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _terminal(client, did):
    return any(
        f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
        for f in _frames_for(client, did)
    )


# ══════════════════════════════════════════════════════════════════════
# S1/S2 — consent after the agent's question vs the same words after a
# statement (§1.1/§1.4, relay-phatic-1)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "line", "booked"), [
    ("yes, sounds good", QUESTION, "Done — you're booked for tomorrow at ten."),
    ("آره عالیه", QUESTION_FA, "باشه، برای فردا ساعت ده وقت گرفتم."),
], ids=["en", "fa"])
async def test_x_consent_after_the_agents_question_is_asked_and_answered(monkeypatch, answer, line, booked):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    name = f"consent_after_question_{'fa' if answer[0] > 'z' else 'en'}"

    def start(m):
        m.user(UNANSWERED, gap=0, dur=900)
        m.say(line, dur=1600)

    model = Model(f"live-psid-x4-{name}", start=start, answer_reask=lambda _e: booked)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1 and phone.of("audio_delta"))
        await asyncio.sleep(0.5)                       # the question is played and heard
        model.user(answer, gap=1400)
        await _wait(lambda: len(_finals(phone)) >= 2)
        await asyncio.sleep(0.5)                       # the model says nothing
        phone.reask(_finals(phone)[1])                 # the watch asks the consent, by id
        await _wait(lambda: phone.of("reask_result"))
        await asyncio.sleep(0.6)                       # the booking is said and heard

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "consent_turn": finals[1]["turn_id"] if len(finals) > 1 else None,
        "consent_phatic": finals[1].get("phatic") if len(finals) > 1 else None,
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
    }
    _record(name, f"R unanswered → the agent asks «{line}» (heard) → the caller consents «{answer}» → "
            "the app re-asks the consent by id → accepted → the booking is said and heard.",
            [(model.provider._session_id, tape)], relay)
    assert relay["consent_phatic"] is not True, "a consent to the agent's question was flagged phatic"
    assert relay["reask_results"] == [(relay["consent_turn"], "accepted", None)], relay
    assert len(relay["reask_instructions"]) == 1


@pytest.mark.asyncio
async def test_x_control_the_same_words_after_a_statement_close_the_exchange(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    name = "closing_after_statement"

    def start(m):
        m.user(UNANSWERED, gap=0, dur=900)
        m.say(STATEMENT, dur=1600)

    model = Model(f"live-psid-x4-{name}", start=start)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1 and phone.of("audio_delta"))
        await asyncio.sleep(0.5)
        model.user("yes, sounds good", gap=1400)
        await _wait(lambda: len(_finals(phone)) >= 2)
        await asyncio.sleep(0.8)                       # the app asks nothing: R was answered and heard

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {"closing_phatic": finals[1].get("phatic") if len(finals) > 1 else None,
             "reask_results": _results(phone)}
    _record(name, "R → the agent answers with a STATEMENT (heard) → «yes, sounds good» closes the exchange: "
            "flagged phatic; the app never asks anything.", [(model.provider._session_id, tape)], relay)
    assert relay["closing_phatic"] is True
    assert relay["reask_results"] == []


# ══════════════════════════════════════════════════════════════════════
# S3 — a presence check between the running research and a check-in
# (§1.3/§1.5/§1.6, relay-phatic-2; owner req 13)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("presence", "check_in", "tag"), [
    ("can you still hear me?", "any update?", "en"),
    ("هستی هنوز؟", "چی شد؟", "fa"),
])
async def test_x_a_presence_check_never_overtakes_the_check_in(monkeypatch, presence, check_in, tag):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    counters = H.capture_counters(monkeypatch)
    displays: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        if len(displays) == 1:
            started.set()
            try:
                await asyncio.wait_for(release.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass
            return "Three robotics labs: Kumar, Rahimi and Chen.", "m"
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)
    name = f"presence_then_check_in_{tag}"

    def start(m):
        m.user(RESEARCH, gap=0, dur=900)
        m.delegate("d1")

    model = Model(f"live-psid-x4-{name}", start=start)

    async def script():
        await started.wait()
        await asyncio.sleep(0.4)
        model.user(presence, gap=11000, dur=500)
        await asyncio.sleep(0.4)
        model.user(check_in, gap=11000, dur=500)
        model.delegate("d2")
        await asyncio.sleep(0.8)                  # the status line is said and heard
        release.set()
        await _wait(lambda: _terminal(phone, "d1"), 4.0)
        await asyncio.sleep(0.8)                  # the result is said and heard

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=14, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "displays": displays,
        "cards": sorted({f.get("delegation_id") for f in phone.of("delegation")}),
        "phatic": {f["text"]: f.get("phatic") for f in finals},
        "status_answered": [c[1] for c in counters if c[0] == "live_status_answered"],
        "status_dispatched": [c[1] for c in counters if c[0] == "live_status_dispatched"],
    }
    _record(name, f"research d1 runs → «{presence}» → «{check_in}» (the model delegates it as d2): the presence "
            "check asks nothing, the check-in is answered from state (said, heard), no second card.",
            [(model.provider._session_id, tape)], relay)
    assert displays == [RESEARCH], displays
    assert relay["cards"] == ["d1"], relay["cards"]
    assert relay["phatic"].get(presence) is True
    assert {"source": "running"} in relay["status_answered"]


# ══════════════════════════════════════════════════════════════════════
# S4 — a bare prompt after the unanswered request (§1.3 prompt-shaped,
# relay-phatic-3)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("prompt", "tag"), [("so?", "en"), ("پس؟", "fa")])
async def test_x_so_after_the_unanswered_request_reasks_the_request(monkeypatch, prompt, tag):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    name = f"prompt_after_request_{tag}"
    answer = "Union Dental on Front Street is the closest one."

    def start(m):
        m.user(UNANSWERED, gap=0, dur=900)

    model = Model(f"live-psid-x4-{name}", start=start, answer_reask=lambda _e: answer)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1)
        await asyncio.sleep(0.4)                       # the model says nothing
        model.user(prompt, gap=7000, dur=400)
        await _wait(lambda: len(_finals(phone)) >= 2)
        await asyncio.sleep(0.5)                       # still nothing
        phone.reask(_finals(phone)[0])                 # the watch asks R, by id — not the prompt
        await _wait(lambda: phone.of("reask_result"))
        await asyncio.sleep(0.6)

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "request_turn": finals[0]["turn_id"],
        "prompt_phatic": finals[1].get("phatic") if len(finals) > 1 else None,
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
    }
    _record(name, f"R unanswered → «{prompt}» (asks nothing) → the app re-asks R by id → accepted (plain) → "
            "the answer is said and heard.", [(model.provider._session_id, tape)], relay)
    assert relay["prompt_phatic"] is True, finals
    assert relay["reask_results"] == [(relay["request_turn"], "accepted", None)], relay


# ══════════════════════════════════════════════════════════════════════
# S5 — a reply to "hello?" was HEARD: R is UNCERTAIN, re-asked
# conditionally, never "No reply" (§3.2-§3.5, app-1; supervisor 14:19)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "ask", "reply", "play_reply", "after"), [
    # §3.6: the model never acts on "let me check" → R is NOT silently cleared.
    ("hello_then_let_me_check", LIBRARY_Q, "Yes, I'm here, let me check the library hours.", True, None),
    # §3.6: a paraphrased real answer → conditional, never "No reply".
    ("hello_then_paraphrase", WEATHER_Q, "Yes, I'm here. It's sunny and twenty-two.", True, None),
    # §3.6: pure "Yes, I'm here." → conditional.
    ("hello_then_pure_presence", LIBRARY_Q, "Yes, I'm here.", True, None),
    # §3.2 pin: the reply to P was cut before any audio → PLAIN; the model
    # answers the plain reask and R is covered.
    ("hello_reply_unheard", LIBRARY_Q, "Yes, I'm here.", False,
     "The Union Station library closes at nine tonight."),
])
async def test_x_a_request_after_a_heard_reply_to_hello_is_uncertain(
    monkeypatch, name, ask, reply, play_reply, after,
):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def start(m):
        m.user(ask, gap=0, dur=900)

    model = Model(f"live-psid-x4-{name}", start=start, answer_reask=lambda _e: after)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1)
        await asyncio.sleep(0.3)
        model.user("hello?", gap=7100, dur=400)
        await _wait(lambda: len(_finals(phone)) >= 2)
        model.say(reply, gap=200, dur=1300)            # parented to "hello?"
        await _wait(lambda: phone.of("audio_delta"))
        await asyncio.sleep(0.6)                       # heard (or cut) and reported
        phone.reask(_finals(phone)[0])                 # the watch asks R, by id, after the reply ended
        await _wait(lambda: phone.of("reask_result"))
        await asyncio.sleep(0.6)

    play = (lambda _eid, _n: "idle") if play_reply else (lambda _eid, n: "cut" if n == 1 else "idle")
    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape, play=play)
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "request_turn": finals[0]["turn_id"],
        "hello_phatic": finals[1].get("phatic") if len(finals) > 1 else None,
        "reply_parent": next((f.get("parent_user_turn_id") for f in phone.of("response_text")), None),
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
    }
    _record(name, f"R «{ask}» unanswered → «hello?» → the model replies «{reply}» (parented to hello?, "
            f"{'HEARD' if play_reply else 'cut before any audio'}) → the app re-asks R by id after the reply "
            f"ended → {'accepted + conditional' if play_reply else 'accepted (plain)'}"
            f"{'; the model answers' if after else '; the model says nothing more'}.",
            [(model.provider._session_id, tape)], relay)
    assert relay["hello_phatic"] is True
    assert relay["reply_parent"] == finals[1]["turn_id"], relay
    content = relay["reask_instructions"][0] if relay["reask_instructions"] else ""
    if play_reply:
        assert relay["reask_results"] == [(relay["request_turn"], "accepted", True)], relay
        assert "Earlier the caller asked" in content and "No reply" not in content, content
    else:
        assert relay["reask_results"] == [(relay["request_turn"], "accepted", None)], relay
        assert "Earlier the caller asked" not in content, content


# ══════════════════════════════════════════════════════════════════════
# S6 — R's own answer was never heard (§3.2: heard, not received)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_x_an_answer_nobody_heard_leaves_the_request_recoverable(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    name = "own_answer_unheard"
    again = "It's sunny in Paris, about twenty degrees."

    def start(m):
        m.user(PARIS_Q, gap=0, dur=900)
        m.say("It is sunny in Paris.", gap=100, dur=600)    # parented to R …

    model = Model(f"live-psid-x4-{name}", start=start, answer_reask=lambda _e: again)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1 and phone.of("audio_delta"))
        await asyncio.sleep(0.6)                       # … and cut before any of it was heard
        phone.reask(_finals(phone)[0])
        await _wait(lambda: phone.of("reask_result"))
        await asyncio.sleep(0.6)

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape,
                  play=lambda _eid, n: "cut" if n == 1 else "idle")
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "request_turn": finals[0]["turn_id"],
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
    }
    _record(name, "R → its answer (parented to R) is cut before any audio was heard → the app re-asks R by "
            "id → accepted (plain, truthful 'did not reach them') → answered again, heard.",
            [(model.provider._session_id, tape)], relay)
    assert relay["reask_results"] == [(relay["request_turn"], "accepted", None)], relay
    assert relay["reask_instructions"] and "did not reach them" in relay["reask_instructions"][0]


# ══════════════════════════════════════════════════════════════════════
# S7/S8 — a drop inside an open turn (§4.2, app-3): a REAL request Q is
# carried; a presence check P is not and R is the one owed
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "open_turn", "carried"), [
    ("drop_in_open_request", "book me a table for two tonight", "open"),
    ("drop_in_open_presence_check", "can you still hear me?", "request"),
])
async def test_x_a_drop_inside_an_open_turn(monkeypatch, name, open_turn, carried):
    H.fast_clocks(
        monkeypatch, voice_live_utterance_gap_ms=3000, voice_live_utterance_hard_gap_ms=6000,
    )
    H.patch_relay(monkeypatch)
    db = _db(name)
    answer = "Done — I'll look for a table for two tonight." if carried == "open" \
        else "Union Dental on Front Street is the closest one."

    def start(m):
        m.user(UNANSWERED, gap=0, dur=900)
        m.user(open_turn, gap=4100, dur=600)          # still open when the network goes

    m1 = Model(f"live-psid-x4-{name}-1", start=start)
    tape1: list = []
    p1 = Phone([_config(), {"type": "audio_ready"}, 0.6, DROP], tape=tape1)
    await H.run_relay(p1, m1.provider, timeout=6, db_session_id=db)
    delivered = [ev["in"] for ev in tape1 if "in" in ev and ev["in"].get("type") == "transcript"]
    request = next(f for f in delivered if f.get("final"))
    settle = [f for f in delivered if f["text"] == open_turn]
    assert settle and not any(f.get("final") for f in settle), "the open turn's final reached the phone"
    open_id = settle[-1]["turn_id"]

    m2 = Model(f"live-psid-x4-{name}-2", answer_reask=lambda _e: answer)

    async def repaired():
        await _wait(lambda: p2.of("ready"))
        await asyncio.sleep(0.2)
        p2.reask_id(open_id, settle[-1]["text"])        # the abandoned turn, by id, first
        await _wait(lambda: p2.of("reask_result"))
        await asyncio.sleep(0.3)
        if carried == "request":                       # unknown → R is due now, by id
            p2.reask(request)
            await _wait(lambda: len(p2.of("reask_result")) >= 2)
        await asyncio.sleep(0.6)

    tape2: list = []
    p2 = Phone([_config(), {"type": "audio_ready"}, repaired, 0.3, {"type": "stop"}], tape=tape2)
    await H.run_relay(p2, m2.provider, timeout=10, db_session_id=db)
    relay = {
        "request_turn": request["turn_id"],
        "open_turn": open_id,
        "reask_results": _results(p2),
        "reask_instructions": m2.reask_contents(),
    }
    _record(name, f"socket 1: R final, «{open_turn}» still open when the network drops (its final never "
            "reaches the phone) → socket 2 (new provider session): the app asks the abandoned turn by id; "
            + ("the relay carried it → accepted → answered, heard; R is never asked."
               if carried == "open" else
               "the relay never carries a turn that asks nothing → unknown; R is asked by id → accepted → "
               "answered, heard."),
            [(m1.provider._session_id, tape1), (m2.provider._session_id, tape2)], relay)
    if carried == "open":
        assert relay["reask_results"] == [(open_id, "accepted", None)], relay
    else:
        assert relay["reask_results"] == [(open_id, "unknown", None), (request["turn_id"], "accepted", None)], relay


# ══════════════════════════════════════════════════════════════════════
# S9 — a phatic tail inside the delegation span (§2, app-5)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_x_a_phatic_tail_never_anchors_or_titles_the_task(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list[dict] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"display": kw.get("display_request") or "", "lang": kw.get("reply_language")})
        await asyncio.sleep(0.3)
        return "Union Dental, Front Street.", "m"

    H.patch_relay(monkeypatch, think=think)
    name = "span_phatic_tail"

    def start(m):
        m.user(UNANSWERED, gap=0, dur=900)

    model = Model(f"live-psid-x4-{name}", start=start)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1)
        model.user("Are you there?", gap=1500, dur=600)
        model.delegate("d-span")
        await _wait(lambda: _terminal(phone, "d-span"), 4.0)
        await asyncio.sleep(0.8)                       # the result is said and heard

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    created = [f for f in _frames_for(phone, "d-span") if f.get("phase") == "created"]
    relay = {
        "request_turn": finals[0]["turn_id"],
        "tail_phatic": finals[1].get("phatic") if len(finals) > 1 else None,
        "created": created[0] if created else None,
        "calls": calls,
    }
    _record(name, "R unanswered → «Are you there?» inside the span → the model delegates: the task anchors "
            "on R, is titled by R, lists only R; its result is said and heard.",
            [(model.provider._session_id, tape)], relay)
    assert created and created[0]["turn_id"] == relay["request_turn"], created
    assert created[0]["title"] == UNANSWERED
    assert created[0].get("request_turn_ids") == [relay["request_turn"]]
    assert [c["display"] for c in calls] == [UNANSWERED]


# ══════════════════════════════════════════════════════════════════════
# S10/S11 — a PARTIAL correction of running work: the relay asks first
# (§5.3, owner req 5); "yes" stops A, "yes, and search B" also runs B.
#
# R2 addendum 5 C10 (supervisor 19:35): the FAST schedule — the new task B
# (d2) finishes 1.2 s after dispatch, i.e. it is TERMINAL before the
# confirmation lands — is fx5's preserved cross-test (test_live_cross_layer_r4
# .py 70773d3e…) and stays; the SLOW schedule (B still running at the answer,
# 3 s) is kept beside it, and each runs with a plain "yes" and with a MIXED
# answer.  Every one must end with B's card AND B's saved row naming the
# relation (replaces A), B run once, its result delivered once, and no task
# resurrected.  The relay's persisted rows are recorded on the tape so the app
# guard compares the card the real reducer ends on with the saved history.
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "answer", "delegate_answer", "b_seconds"), [
    ("clarify_yes", "yes", False, 1.2),
    ("clarify_yes_and_more", MIXED, True, 1.2),
    ("clarify_yes_slow", "yes", False, 3.0),
    ("clarify_yes_and_more_slow", MIXED, True, 3.0),
], ids=["fast_yes", "fast_mixed", "slow_yes", "slow_mixed"])
async def test_x_a_partial_correction_is_confirmed_before_anything_stops(
    monkeypatch, name, answer, delegate_answer, b_seconds,
):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    calls: list[tuple] = []
    killed: list[str] = []
    a_done = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        calls.append((did, kw.get("display_request") or ""))
        if did == "d1":
            try:
                await asyncio.wait_for(a_done.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                killed.append(did)
                raise
            return "Professor X.", "m"
        if did == "d2":
            # FAST (1.2 s, fx5's schedule): B is terminal before the caller's
            # answer is applied.  SLOW (3 s): B is still running then.  B's
            # result has its own words (fx5 used "Done." for every task) only
            # so that its delivery can be counted apart from the remainder's.
            await asyncio.sleep(b_seconds)
            return "Booked the Hotel Ocho.", "m"
        await asyncio.sleep(1.2)
        return "Done.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def start(m):
        m.user(PROFESSORS, gap=0, dur=900)
        m.delegate("d1")

    model = Model(f"live-psid-x4-{name}", start=start)

    def asked():
        return [c for c in model.commentary() if any(mark in c for mark in STOP_MARKS)]

    async def script():
        await _wait(lambda: len(calls) >= 1)
        await asyncio.sleep(0.3)
        model.user(PARTIAL, gap=4000, dur=800)
        model.delegate("d2")
        await _wait(lambda: asked(), 3.0)
        await asyncio.sleep(0.6)                       # the question is said and HEARD
        model.user(answer, gap=2500, dur=600)
        if delegate_answer:
            model.delegate("d-answer")
        await _wait(lambda: _terminal(phone, "d1"), 3.0)
        await _wait(lambda: _terminal(phone, "d2") and (not delegate_answer or _terminal(phone, "d-answer")), 5.0)
        await asyncio.sleep(0.8)
        a_done.set()

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=20, db_session_id=_db(name))
    d1 = _frames_for(phone, "d1")
    d2 = [f for f in _frames_for(phone, "d2") if "task_revision" in f]
    b_ref = f"live-delegation:{model.provider._session_id}:d2"
    relay = {
        "b_seconds": b_seconds,
        "questions": asked(),
        "d1_phases": [f.get("phase") for f in d1],
        "d1_cancel_reason": next((f.get("reason") for f in d1 if f.get("phase") == "cancelled"), None),
        "d2_relations": [f.get("relation") for f in d2],
        "d2_lifecycle": [
            {"rev": f.get("task_revision"), "phase": f.get("phase"), "state": f.get("state"),
             "relation": f.get("relation"), "relation_only": f.get("relation_only", False),
             "result": "result_text" in f, "spoken": f.get("spoken")}
            for f in d2
        ],
        "b_terminal_before_confirmation": _terminal_before_relation(d2),
        "answer_card": [f for f in _frames_for(phone, "d-answer") if f.get("phase") == "created"][:1],
        "calls": calls,
        "killed": killed,
        "said": [c for c in model.commentary() if "Okay, I stopped" in c],
        "b_result_said": [c for c in model.commentary() if "Hotel Ocho" in c],
        "b_saved": _saved_rows(recorded["saves"], b_ref),
    }
    _record(name, f"A (d1) runs → «{PARTIAL}» (d2, partial evidence; B takes {b_seconds} s) → the relay asks "
            f"«Should I stop …?» (said, heard) → «{answer}» → A cancelled (user_cancel), d2's relation becomes "
            "replaces — on the card AND the saved row"
            + ("; the remainder runs as its own task under its own title." if delegate_answer else "."),
            [(model.provider._session_id, tape)], relay)
    assert len(relay["questions"]) == 1, relay["questions"]
    assert relay["d1_cancel_reason"] == "user_cancel", relay["d1_phases"]
    replaces = {"kind": "replaces", "task_id": "d1"}
    assert relay["d2_relations"] and relay["d2_relations"][-1] == replaces, relay["d2_lifecycle"]
    relation_only = [x for x in relay["d2_lifecycle"] if x["relation_only"]]
    if b_seconds < 2.0:
        # The schedule really is the race (supervisor 19:35): B's outcome went
        # out BEFORE the confirmation was applied, so the relation can reach
        # B only as ONE relation-only update of its terminal observation.
        assert relay["b_terminal_before_confirmation"], relay["d2_lifecycle"]
        assert len(relation_only) == 1, relay["d2_lifecycle"]
    else:
        # SLOW: B was still running — the relation rides its own lifecycle.
        assert not relay["b_terminal_before_confirmation"] and not relation_only, relay["d2_lifecycle"]
    # B ran once, reached one outcome, and its result was delivered once.
    assert [did for did, _d in calls].count("d2") == 1, calls
    outcomes = [x for x in relay["d2_lifecycle"]
                if x["phase"] in {"completed", "failed", "cancelled"} and not x["relation_only"]]
    assert [x["phase"] for x in outcomes] == ["completed"], relay["d2_lifecycle"]
    assert all(x["state"] == "completed" and not x["result"] and x["spoken"] is None
               for x in relay["d2_lifecycle"] if x["relation_only"]), relay["d2_lifecycle"]
    assert len(relay["b_result_said"]) == 1, relay["b_result_said"]
    revisions = [x["rev"] for x in relay["d2_lifecycle"]]
    assert revisions == sorted(set(revisions)), revisions
    # The saved history names the same relation as the card.
    assert relay["b_saved"], "B's row was never written"
    assert {r["ref"] for r in relay["b_saved"]} == {b_ref}
    assert relay["b_saved"][-1]["voice"].get("related_task_id") == "d1", relay["b_saved"][-1]
    assert relay["b_saved"][-1]["stored"].get("related_task_id") == "d1", relay["b_saved"][-1]
    if delegate_answer:
        assert relay["answer_card"] and relay["answer_card"][0]["title"] == REMAINDER


def _terminal_before_relation(lifecycle: list[dict]) -> bool:
    """Did the task's OUTCOME frame go out before any frame named a relation
    (the confirmation was applied only after the task was terminal)?"""

    for f in lifecycle:
        if f.get("relation"):
            return False
        if f.get("phase") in {"completed", "failed", "cancelled"}:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# S12 — control: an old task's heard output never makes a new request
# uncertain (the supervisor watch-causality property, relay side)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_x_control_an_old_tasks_output_never_covers_a_new_request(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    started = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        started.set()
        await asyncio.sleep(1.2)
        return "Professor A and Professor B.", "m"

    H.patch_relay(monkeypatch, think=think)
    name = "old_task_new_request"
    request_b = "what's the weather in paris tomorrow"

    def start(m):
        m.user(RESEARCH, gap=0, dur=900)
        m.delegate("d1")

    model = Model(f"live-psid-x4-{name}", start=start,
                  answer_reask=lambda _e: "Tomorrow in Paris: cloudy, fourteen degrees.")

    async def script():
        await started.wait()
        model.user(request_b, gap=2000, dur=700)
        await _wait(lambda: _terminal(phone, "d1"), 4.0)
        await asyncio.sleep(0.8)                       # A's result is said (its own epoch) and heard
        phone.reask(next(f for f in _finals(phone) if f["text"] == request_b))
        await _wait(lambda: phone.of("reask_result"))
        await asyncio.sleep(0.6)

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=12, db_session_id=_db(name))
    b = next(f for f in _finals(phone) if f["text"] == request_b)
    relay = {"request_b": b["turn_id"], "reask_results": _results(phone)}
    _record(name, "A (d1) runs → B unanswered → A's result is said and heard (parented to A) → the app "
            "re-asks B by id → accepted PLAIN (A's output never makes B uncertain) → B answered, heard.",
            [(model.provider._session_id, tape)], relay)
    assert relay["reask_results"] == [(b["turn_id"], "accepted", None)], relay


# ══════════════════════════════════════════════════════════════════════
# S13 — the drop comes after a HEARD reply to "hello?": R is carried
# UNCERTAIN and re-asked conditionally on the repaired socket (§3.4)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_x_a_drop_after_a_heard_reply_to_hello_carries_r_conditionally(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    name = "drop_after_heard_reply_to_hello"
    db = _db(name)

    def start(m):
        m.user(LIBRARY_Q, gap=0, dur=900)

    m1 = Model(f"live-psid-x4-{name}-1", start=start)

    async def first():
        await _wait(lambda: len(_finals(p1)) >= 1)
        await asyncio.sleep(0.3)
        m1.user("hello?", gap=7100, dur=400)
        await _wait(lambda: len(_finals(p1)) >= 2)
        m1.say("Yes, I'm here.", gap=200, dur=700)
        await _wait(lambda: p1.of("audio_delta"))
        await asyncio.sleep(0.5)                       # heard and reported
        p1.push_frame(DROP)

    tape1: list = []
    p1 = Phone([_config(), {"type": "audio_ready"}, first, 3.0], tape=tape1)
    await H.run_relay(p1, m1.provider, timeout=8, db_session_id=db)
    request = _finals(p1)[0]

    m2 = Model(f"live-psid-x4-{name}-2")

    async def repaired():
        await _wait(lambda: p2.of("ready"))
        await asyncio.sleep(0.2)
        p2.reask(request)                              # R, by id, on the repaired socket
        await _wait(lambda: p2.of("reask_result"))
        await asyncio.sleep(0.6)                       # the model says nothing more

    tape2: list = []
    p2 = Phone([_config(), {"type": "audio_ready"}, repaired, 0.3, {"type": "stop"}], tape=tape2)
    await H.run_relay(p2, m2.provider, timeout=10, db_session_id=db)
    relay = {
        "request_turn": request["turn_id"],
        "reask_results": _results(p2),
        "reask_instructions": m2.reask_contents(),
    }
    _record(name, "socket 1: R unanswered → «hello?» → «Yes, I'm here.» (heard) → the network drops → "
            "socket 2 (new provider session): the app asks R by id → the relay carried R UNCERTAIN → "
            "accepted + conditional (never 'No reply') → the model says nothing more.",
            [(m1.provider._session_id, tape1), (m2.provider._session_id, tape2)], relay)
    assert relay["reask_results"] == [(request["turn_id"], "accepted", True)], relay
    content = relay["reask_instructions"][0] if relay["reask_instructions"] else ""
    assert "Earlier the caller asked" in content and "No reply" not in content, content


# ══════════════════════════════════════════════════════════════════════
# ROUND 5 (addendum 5) — the shapes where the two layers decided from
# different evidence (A6), the A4 contract both ways, A7, B1, C3, C4.
# ══════════════════════════════════════════════════════════════════════

def _transcripts_of(tape: list, *, final=None) -> list[dict]:
    out = []
    for ev in tape:
        f = ev.get("in")
        if f and f.get("type") == "transcript" and (final is None or bool(f.get("final")) == final):
            out.append(f)
    return out


# ── A1/A6: an answer with NO playback receipt ──────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("audio", [False, True], ids=["text", "audio"])
async def test_x5_an_answer_with_no_receipt_leaves_the_request_recoverable(monkeypatch, audio):
    """R answered text-only (or with audio the phone queued and native
    playback never confirmed); the app's liveness bound asks R by id; the
    relay must not read receipt ABSENCE as hearing (old fx4 99e5b23d:
    `answered`, the app cleared R silently)."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    name = "no_receipt_audio" if audio else "no_receipt_text"

    def start(m):
        m.user(UNANSWERED, gap=0, dur=900)
        m.say("There is one on Front Street.", gap=100, dur=600, audio=audio)

    model = Model(f"live-psid-x5-{name}", start=start)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1 and phone.of("response_text"))
        if audio:
            await _wait(lambda: phone.of("audio_delta"))
        await asyncio.sleep(0.5)                       # retired by the relay's own gap; no receipt ever
        phone.reask(_finals(phone)[0])                 # the app's liveness bound: R, by id
        await _wait(lambda: phone.of("reask_result"))
        await asyncio.sleep(0.6)                       # the model says nothing more

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape,
                  play=lambda _eid, _n: None)
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "request_turn": finals[0]["turn_id"],
        "answer_parent": next((f.get("parent_user_turn_id") for f in phone.of("response_text")), None),
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
    }
    _record(name, "R → its answer (parented to R) was " + ("audio the phone queued and never confirmed playing"
            if audio else "text only") + " — no receipt at all → the app's liveness bound asks R by id → "
            "accepted + conditional (UNKNOWN, never 'answered') → the model says nothing more.",
            [(model.provider._session_id, tape)], relay)
    assert relay["answer_parent"] == relay["request_turn"], relay
    assert relay["reask_results"] == [(relay["request_turn"], "accepted", True)], relay
    content = relay["reask_instructions"][0] if relay["reask_instructions"] else ""
    assert UNCONFIRMED_MARK in content and UNHEARD_MARK not in content, content


@pytest.mark.asyncio
async def test_x5_an_answer_dropped_with_the_socket_leaves_the_request_recoverable(monkeypatch):
    """R's answer (text + audio) was still streaming, unconfirmed, when the
    socket died; on the repaired socket (a new provider session) the app asks
    R by id (old fx4 99e5b23d: `unknown` — R was not carried, handed off)."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=5000)
    H.patch_relay(monkeypatch)
    name = "drop_mid_answer"
    db = _db(name)

    def start(m):
        m.user(PARIS_Q, gap=0, dur=900)
        m.say("It is sunny in Paris.", gap=100, dur=600)

    m1 = Model(f"live-psid-x5-{name}-1", start=start)

    async def first():
        await _wait(lambda: len(_finals(p1)) >= 1 and p1.of("audio_delta"))
        await asyncio.sleep(0.3)                       # queued, never confirmed; still streaming
        p1.push_frame(DROP)

    tape1: list = []
    p1 = Phone([_config(), {"type": "audio_ready"}, first, 3.0], tape=tape1, play=lambda _eid, _n: None)
    await H.run_relay(p1, m1.provider, timeout=8, db_session_id=db)
    request = _finals(p1)[0]

    m2 = Model(f"live-psid-x5-{name}-2")

    async def repaired():
        await _wait(lambda: p2.of("ready"))
        await asyncio.sleep(0.2)
        p2.reask(request)                              # the app's repair: R, by id
        await _wait(lambda: p2.of("reask_result"))
        await asyncio.sleep(0.6)                       # the model says nothing more

    tape2: list = []
    p2 = Phone([_config(), {"type": "audio_ready"}, repaired, 0.3, {"type": "stop"}], tape=tape2)
    await H.run_relay(p2, m2.provider, timeout=10, db_session_id=db)
    relay = {
        "request_turn": request["turn_id"],
        "reask_results": _results(p2),
        "reask_instructions": m2.reask_contents(),
    }
    _record(name, "socket 1: R → its answer streaming (text + audio), never confirmed → the network drops → "
            "socket 2 (new provider session): the app asks R by id → the relay CARRIED R (UNKNOWN) → "
            "accepted + conditional → the model says nothing more.",
            [(m1.provider._session_id, tape1), (m2.provider._session_id, tape2)], relay)
    assert relay["reask_results"] == [(request["turn_id"], "accepted", True)], relay
    content = relay["reask_instructions"][0] if relay["reask_instructions"] else ""
    assert UNCONFIRMED_MARK in content, content


# ── A3/A6: a reask crossing an OPEN turn's first words ─────────────────

OPEN_PREFIXES = [
    ("can you", " still hear me?", "can_you"),
    ("are you", " still there?", "are_you"),
    ("did you", " hear me?", "did_you"),
    ("صدامو", " میشنوی؟", "sedamo"),
    ("thank you for", " your help", "thank_you_for"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("head", "tail", "tag"), OPEN_PREFIXES, ids=[t for _h, _t, t in OPEN_PREFIXES])
async def test_x5_a_reask_crossing_an_open_turn_is_never_stale_on_its_first_words(monkeypatch, head, tail, tag):
    """The app's reask of R left before the caller's first settle reached it;
    the relay judged the OPEN turn (old: `stale` on «can you» → the app
    covered R, which was then lost when the turn closed as a presence
    check).  Now: in_flight → the app retries R by id once the turn closed
    asking nothing → accepted → answered and heard."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=900, voice_live_utterance_hard_gap_ms=1800)
    H.patch_relay(monkeypatch)
    name = f"stale_on_open_prefix_{tag}"

    def start(m):
        m.user(LIBRARY_Q, gap=0, dur=900)

    model = Model(f"live-psid-x5-{name}", start=start, answer_reask=lambda _e: LIBRARY_A)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1)
        r = _finals(phone)[0]
        await asyncio.sleep(0.3)                       # nothing is said
        model.user(head, gap=2100, dur=300)            # the caller starts … (OPEN)
        await _wait(lambda: any(f.get("turn_id") != r["turn_id"] for f in phone.of("transcript")))
        phone.reask_crossing(r, lambda f: f.get("type") == "transcript" and f.get("turn_id") != r["turn_id"])
        await _wait(lambda: phone.of("reask_result"))
        model.user(tail, gap=50, dur=550)              # … the same utterance goes on and closes
        await _wait(lambda: len(_finals(phone)) >= 2, 5.0)
        await asyncio.sleep(0.3)
        phone.reask(r)                                 # the app retries R once the turn asked nothing
        await _wait(lambda: len(phone.of("reask_result")) >= 2)
        await asyncio.sleep(0.6)                       # the answer is said and heard

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=12, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "request_turn": finals[0]["turn_id"],
        "open_turn_final": finals[1] if len(finals) > 1 else None,
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
        "crossed": [ev.get("crossed") for ev in tape if "out" in ev and ev.get("crossed")],
    }
    _record(name, f"R unanswered → the caller starts «{head}» (open) → the app's reask of R crosses that settle "
            f"on the wire → in_flight (never stale on the first words) → «{head}{tail}» closes asking nothing "
            "→ the app retries R by id → accepted → answered and heard.",
            [(model.provider._session_id, tape)], relay)
    assert relay["crossed"], "the reask never crossed the open turn"
    assert relay["open_turn_final"] and relay["open_turn_final"].get("phatic") is True, relay
    assert [(t, o) for t, o, _c in relay["reask_results"]] == [
        (relay["request_turn"], "in_flight"), (relay["request_turn"], "accepted"),
    ], relay


# ── A2/A6: a heard reply to "hello?" AFTER a plain accept ──────────────

@pytest.mark.asyncio
async def test_x5_a_heard_reply_to_hello_after_a_plain_accept(monkeypatch):
    """R's by-id reask is accepted PLAIN; before the model speaks the caller
    says "hello?"; the model's answer is parented (truthfully) to "hello?"
    and HEARD.  The app must not say "No reply … asking again" over it, and
    any hand-off is the conditional one (addendum 5 A2)."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    name = "hello_after_plain_accept"

    def start(m):
        m.user(LIBRARY_Q, gap=0, dur=900)

    model = Model(f"live-psid-x5-{name}", start=start)

    async def script():
        await _wait(lambda: len(_finals(phone)) >= 1)
        await asyncio.sleep(0.4)                       # nothing is said
        phone.reask(_finals(phone)[0])                 # the app's no-reply timer: R, by id
        await _wait(lambda: phone.of("reask_result"))
        await asyncio.sleep(0.3)
        model.user("hello?", gap=19000, dur=400)       # the caller, before the model speaks
        await _wait(lambda: len(_finals(phone)) >= 2)
        model.say(LIBRARY_A, gap=200, dur=1300)        # the model answers R now
        await _wait(lambda: phone.of("audio_delta"))
        await asyncio.sleep(0.6)                       # heard to the end and reported

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "request_turn": finals[0]["turn_id"],
        "hello_turn": finals[1]["turn_id"] if len(finals) > 1 else None,
        "hello_phatic": finals[1].get("phatic") if len(finals) > 1 else None,
        "reply_parent": next((f.get("parent_user_turn_id") for f in phone.of("response_text")), None),
        "reask_results": _results(phone),
    }
    _record(name, "R unanswered → the app asks R by id → accepted PLAIN → «hello?» → the model's answer "
            "(parented to hello?) is HEARD → no 'asking again' over it; the hand-off, if any, is conditional.",
            [(model.provider._session_id, tape)], relay)
    assert relay["reask_results"] == [(relay["request_turn"], "accepted", None)], relay
    assert relay["hello_phatic"] is True, relay
    assert relay["reply_parent"] == relay["hello_turn"], relay


# ── A4/A6: a drop in the middle of a turn — the contract both ways ─────

A4_DROPS = [
    # (name, older request or None, the words open at the drop, kind)
    ("drop_mid_presence_check_can_you", LIBRARY_Q, "can you", "fragment"),
    ("drop_mid_presence_check_are_you_still", LIBRARY_Q, "are you still", "fragment"),
    ("drop_mid_presence_check_fa", LIBRARY_FA, "الو هنوز صدامو", "fragment"),
    ("drop_mid_presence_check_thank_you_for", LIBRARY_Q, "thank you for", "fragment"),
    ("drop_complete_request_en", LIBRARY_Q, PHARMACY, "complete"),
    ("drop_complete_request_fa", LIBRARY_FA, OFFICE_FA, "complete"),
    ("drop_fragment_alone_en", None, "can you", "fragment"),
    ("drop_fragment_alone_fa", None, "الو هنوز صدامو", "fragment"),
    ("drop_complete_alone_en", None, PHARMACY, "complete"),
    ("drop_complete_alone_fa", None, OFFICE_FA, "complete"),
]
_A4_REPLIES = {
    "fragment_with_r_en": "Sorry, the line dropped. The library closes at nine tonight. You were saying something — could you finish?",
    "fragment_with_r_fa": "ببخشید، تماس قطع شد. کتابخونه یکشنبه از ده تا پنج بازه. داشتی یه چیزی می‌گفتی — میشه حرفت رو تموم کنی؟",
    "fragment_alone_en": "Sorry, the line dropped while you were talking. Could you finish what you were saying?",
    "fragment_alone_fa": "ببخشید، وسط حرفت تماس قطع شد. میشه حرفت رو تموم کنی؟",
    "complete_en": "The closest pharmacy is Union Pharmacy on Front Street.",
    "complete_fa": "آدرس دفترش خیابان فرانت، پلاک ده است.",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "older", "open_text", "kind"), A4_DROPS, ids=[n for n, *_ in A4_DROPS])
async def test_x5_a_drop_mid_turn_keeps_the_owed_request(monkeypatch, name, older, open_text, kind):
    """Addendum 5 A4 (supervisor 19:35), BOTH ways.  A TRUNCATED turn at the
    drop ('can you', 'are you still', «الو هنوز صدامو», 'thank you for')
    never displaces the older request R: its by-id ask is `unknown`, R is
    accepted with the truthful "the call dropped while the caller was
    saying «…»: ask them to finish" clause, and the fragment is never
    executed (old: the fragment was carried as "reply to exactly these
    words" and R was `unknown` — lost).  A COMPLETE new request at the drop
    is recovered by id exactly as today (pins drop_in_open_request); with
    nothing older, a fragment is carried as "ask them to finish"."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=3000, voice_live_utterance_hard_gap_ms=6000)
    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)
    db = _db(name)
    fa = not open_text.isascii()

    def start(m):
        if older:
            m.user(older, gap=0, dur=900)
            m.user(open_text, gap=4100, dur=400)       # still open when the network goes
        else:
            m.user(open_text, gap=0, dur=400)

    m1 = Model(f"live-psid-x5-{name}-1", start=start)
    tape1: list = []
    p1 = Phone([_config(), {"type": "audio_ready"}, 0.6, DROP], tape=tape1)
    await H.run_relay(p1, m1.provider, timeout=6, db_session_id=db)
    delivered = _transcripts_of(tape1)
    request = next((f for f in delivered if f.get("final") and older and f["text"] == older), None)
    settle = [f for f in delivered if f["text"] == open_text]
    assert settle and not any(f.get("final") for f in settle), "the open turn's final reached the phone"
    open_id = settle[-1]["turn_id"]
    if kind == "fragment":
        reply = _A4_REPLIES[f"fragment_{'with_r' if older else 'alone'}_{'fa' if fa else 'en'}"]
    else:
        reply = _A4_REPLIES[f"complete_{'fa' if fa else 'en'}"]
    m2 = Model(f"live-psid-x5-{name}-2", answer_reask=lambda _e: reply)

    async def repaired():
        await _wait(lambda: p2.of("ready"))
        await asyncio.sleep(0.2)
        p2.reask_id(open_id, settle[-1]["text"])       # the abandoned turn, by id, first
        await _wait(lambda: p2.of("reask_result"))
        await asyncio.sleep(0.3)
        if older and p2.of("reask_result")[-1].get("outcome") == "unknown":
            p2.reask(request)                          # unknown → R is due at once, by id
            await _wait(lambda: len(p2.of("reask_result")) >= 2)
        await asyncio.sleep(0.6)                       # the reply is said and heard

    tape2: list = []
    p2 = Phone([_config(), {"type": "audio_ready"}, repaired, 0.3, {"type": "stop"}], tape=tape2)
    await H.run_relay(p2, m2.provider, timeout=10, db_session_id=db)
    await H.drain_detached()
    relay = {
        "kind": kind,
        "request_turn": request["turn_id"] if request else None,
        "open_turn": open_id,
        "reask_results": _results(p2),
        "reask_instructions": m2.reask_contents(),
        "thinks": thinks,
    }
    _record(name, f"socket 1: {'R «' + older + '» unanswered, ' if older else ''}«{open_text}» still open when the "
            "network drops → socket 2 (new provider session): the app asks the abandoned turn by id"
            + (" → unknown (a fragment is never the owed request) → R asked by id → accepted with 'the call "
               "dropped while the caller was saying «…»: ask them to finish' → said, heard."
               if kind == "fragment" and older else
               " → accepted with 'the call dropped … ask them to finish' (never 'exactly these words') → said, heard."
               if kind == "fragment" else
               " → accepted (a complete request is the owed one) → answered, heard"
               + ("; R is never asked (superseded, contract v0.3 §4)." if older else ".")),
            [(m1.provider._session_id, tape1), (m2.provider._session_id, tape2)], relay)
    contents = relay["reask_instructions"]
    assert len(contents) == 1, contents
    assert not any(open_text in t for t in thinks), thinks          # never executed by a task
    if kind == "fragment" and older:
        assert relay["reask_results"] == [(open_id, "unknown", None), (request["turn_id"], "accepted", None)], relay
        assert f"«{older}»" in contents[0] and f"«{open_text}…»" in contents[0], contents
        assert any(mark in contents[0] for mark in FRAGMENT_MARKS), contents
    elif kind == "fragment":
        assert relay["reask_results"] == [(open_id, "accepted", None)], relay
        assert f"«{open_text}…»" in contents[0] and any(mark in contents[0] for mark in FRAGMENT_MARKS), contents
        assert "exactly these words" not in contents[0] and "دقیقاً به همین حرف" not in contents[0], contents
    else:
        assert relay["reask_results"] == [(open_id, "accepted", None)], relay
        assert f"«{open_text}»" in contents[0], contents
        assert not any(mark in contents[0] for mark in FRAGMENT_MARKS), contents


# ── A5/A6: a tap that names no epoch, after a heard reply ──────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "played"), [
    ("tap_without_identity", 0),
    ("tap_without_identity_cumulative", 300),
], ids=["this_build", "shipped_build_value"])
async def test_x5_a_tap_that_names_no_epoch_is_no_hearing(monkeypatch, name, played):
    """R1's reply was heard to the end; R2's reply has shown its TEXT but
    none of its audio; the caller taps — no playback identity is active, so
    the tap names no epoch.  This build sends played_ms 0; a shipped build
    sent its session-cumulative counter (300 here, the previous reply's).
    Either way nothing of R2's reply was heard (old: the relay counted the
    cumulative value as hearing — `answered`, R2 silently cleared)."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    r1, r2 = "what is the weather in paris", "and in london tomorrow"
    again = "Tomorrow in London: light rain, about twelve degrees."

    def start(m):
        m.user(r1, gap=0, dur=900)
        m.say("It is sunny in Paris.", gap=100, dur=600)

    model = Model(f"live-psid-x5-{name}", start=start, answer_reask=lambda _e: again)

    async def script():
        await _wait(lambda: phone.of("audio_delta") and _finals(phone))
        await asyncio.sleep(0.4)                       # R1's reply is heard to the end and reported
        model.user(r2, gap=3400, dur=600)
        await _wait(lambda: len(_finals(phone)) >= 2)
        model.say("In London tomorrow", gap=400, dur=400, audio=False)   # text so far, no audio
        await _wait(lambda: any(f.get("parent_user_turn_id") == _finals(phone)[1]["turn_id"]
                                for f in phone.of("response_text")))
        phone.tap_without_identity(played)
        await asyncio.sleep(0.5)
        phone.reask(_finals(phone)[1])                 # the app: R2 was never heard → by id
        await _wait(lambda: phone.of("reask_result"))
        await asyncio.sleep(0.6)                       # the answer is said again and heard

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=10, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "request_turn": finals[1]["turn_id"],
        "tap_played_ms": played,
        "playback_interrupted": phone.of("playback_interrupted"),
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
    }
    _record(name, f"R1 answered and HEARD → R2 → its reply's text, no audio → a tap that names no epoch "
            f"(played_ms {played}) → the app asks R2 by id → accepted (never 'answered') → answered again, heard.",
            [(model.provider._session_id, tape)], relay)
    assert [(t, o) for t, o, _c in relay["reask_results"]] == [(relay["request_turn"], "accepted")], relay
    if played:
        assert relay["reask_results"][0][2] is True, relay     # UNKNOWN → conditional
    else:
        assert relay["reask_results"][0][2] is None, relay     # UNHEARD → plain, truthful
        assert UNHEARD_MARK in relay["reask_instructions"][0], relay


# ── A7: a promise-only answer ──────────────────────────────────────────

PROMISE_Q = "what are the library hours on sunday"
PROMISE = "Sure, let me check the library hours for you."
PROMISE_FA = "باشه، الان ساعت کاری کتابخونه رو برات پیدا می‌کنم."
_WORK_ALLOWED = ("delegate it if it needs work", "واگذار کن")
_WORK_FORBIDDEN = ("start new work", "کار تازه‌ای شروع نکن")


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "words", "promise", "obey"), [
    ("promise_only_en", PROMISE_Q, PROMISE, False),
    ("promise_only_fa", LIBRARY_FA, PROMISE_FA, False),
    ("promise_only_delegated", PROMISE_Q, PROMISE, True),
], ids=["en_silent", "fa_silent", "en_model_delegates"])
async def test_x5_a_promise_only_answer_is_never_silently_lost(monkeypatch, name, words, promise, obey):
    """Addendum 5 A7: R answered only with a promise (text only, no receipt,
    no delegation).  The app's liveness bound asks R by id; the relay's
    unconfirmed instruction must ALLOW the work that never started.  The
    model here does what the instruction allows: in `promise_only_delegated`
    it delegates the work only when the instruction permits new work (old:
    "Don't redo any action or start new work" — nothing ran, the promise was
    never kept); otherwise it says nothing and R ends in the conditional
    hand-off — never silently lost."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        await asyncio.sleep(0.3)
        return "On Sunday the library is open from ten to five.", "m"

    H.patch_relay(monkeypatch, think=think)

    def start(m):
        m.user(words, gap=0, dur=900)
        m.say(promise, gap=100, dur=600, audio=False)   # a promise, text only: no audio, no receipt

    def answer(e):
        content = str(e.get("content") or "")
        allowed = any(x in content for x in _WORK_ALLOWED) and not any(x in content for x in _WORK_FORBIDDEN)
        if obey and allowed:
            return lambda m: m.delegate("d-promise")
        return None

    model = Model(f"live-psid-x5-{name}", start=start, answer_reask=answer)

    async def script():
        await _wait(lambda: _finals(phone) and phone.of("response_text"))
        await asyncio.sleep(0.5)
        phone.reask(_finals(phone)[0])                 # the app's liveness bound: R, by id
        await _wait(lambda: phone.of("reask_result"))
        if obey:
            await _wait(lambda: _terminal(phone, "d-promise"), 4.0)
            await asyncio.sleep(0.8)                   # the result is said and heard
        else:
            await asyncio.sleep(0.6)

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape,
                  play=lambda _eid, _n: "idle")
    await H.run_relay(phone, model.provider, timeout=12, db_session_id=_db(name))
    finals = _finals(phone)
    relay = {
        "request_turn": finals[0]["turn_id"],
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
        "thinks": thinks,
        "cards": sorted({f.get("delegation_id") for f in phone.of("delegation")}),
    }
    _record(name, f"R «{words}» → the model only PROMISES («{promise}», text only, no receipt, no task) → the "
            "app's liveness bound asks R by id → accepted + conditional with an instruction that allows the "
            "work → " + ("the model delegates it → the task runs, its result is said and heard."
                         if obey else "the model says nothing more → the conditional hand-off."),
            [(model.provider._session_id, tape)], relay)
    assert relay["reask_results"] == [(relay["request_turn"], "accepted", True)], relay
    content = relay["reask_instructions"][0] if relay["reask_instructions"] else ""
    assert f"«{words}»" in content and any(x in content for x in _WORK_ALLOWED), content
    assert not any(x in content for x in _WORK_FORBIDDEN), content
    if obey:
        assert relay["cards"] == ["d-promise"] and len(thinks) == 1, relay


# ── B1: a hesitation before the media command's tail ──────────────────

@pytest.mark.asyncio
async def test_x5_a_hesitation_before_the_media_tail_is_one_command(monkeypatch):
    """«آهنگ رو قطع» + «اِ» + «کن»: ONE stop — the relay stops once, the late
    delegation is absorbed, no «کن» card (old: «کن» ran as research — a card
    titled «کن»)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    controls: list[str] = []
    thinks: list[str] = []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or task)
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think, control=control)
    name = "media_tail_hesitation"
    model = Model(f"live-psid-x5-{name}")

    async def script():
        await _wait(lambda: phone.of("ready"))
        await asyncio.sleep(0.2)
        model.user("آهنگ رو قطع", gap=3000, dur=600)
        model.say("چشم.", gap=100, dur=200)
        await asyncio.sleep(0.9)                       # > grace: the backstop fires the stop
        model.user("اِ", gap=600, dur=200)
        await asyncio.sleep(0.5)
        model.user("کن", gap=600, dur=300)
        model.say("چشم.", gap=100, dur=200)
        await asyncio.sleep(0.6)
        model.delegate("d-late")
        await asyncio.sleep(1.4)

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, {"type": "now_playing", "title": "Fadat Sham - Mahasti"},
                   script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=14, db_session_id=_db(name))
    relay = {
        "controls": controls,
        "thinks": thinks,
        "cards": sorted({f.get("delegation_id") for f in phone.of("delegation")}),
        "finals": [(f["turn_id"], f["text"], f.get("phatic")) for f in _finals(phone)],
        "media_control": phone.of("media_control"),
        "reask_results": _results(phone),
    }
    _record(name, "music playing → «آهنگ رو قطع» (acked «چشم.») → the stop fires → «اِ» → «کن» (acked) → the "
            "model's late delegation: one command — stopped once, no card, nothing asked.",
            [(model.provider._session_id, tape)], relay)
    assert controls == ["stop"], controls
    assert thinks == [] and relay["cards"] == [], relay


# ── C3: assent to the model's OWN question in the stop question's epoch ─

@pytest.mark.asyncio
async def test_x5_assent_to_the_models_own_question_never_cancels(monkeypatch):
    """The relay asked «Should I stop A?» and the model went on, IN THE SAME
    EPOCH, with a question of its own; the caller's "yes" answers the
    model's question — the confirmation is void (newer_question) and A is
    never cancelled (old: the 'yes' cancelled A)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    calls: list[tuple] = []
    a_done = asyncio.Event()
    own_q = " Also, do you want me to include their email addresses?"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        calls.append((did, kw.get("display_request") or ""))
        if did == "d1":
            try:
                await asyncio.wait_for(a_done.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                pass
            return "Professor X.", "m"
        await asyncio.sleep(3.0)
        return "Booked the Hotel Ocho.", "m"

    H.patch_relay(monkeypatch, think=think)
    name = "own_question_same_epoch"

    def start(m):
        m.user(PROFESSORS, gap=0, dur=900)
        m.delegate("d1")

    model = Model(f"live-psid-x5-{name}", start=start,
                  commentary_tail=lambda c: own_q if any(mark in c for mark in STOP_MARKS) else "")

    def asked():
        return [c for c in model.commentary() if any(mark in c for mark in STOP_MARKS)]

    async def script():
        await _wait(lambda: len(calls) >= 1)
        await asyncio.sleep(0.3)
        model.user(PARTIAL, gap=4000, dur=800)
        model.delegate("d2")
        await _wait(lambda: asked(), 3.0)
        await asyncio.sleep(0.6)                       # the epoch (both questions) is said and HEARD
        model.user("yes", gap=2500, dur=400)
        await _wait(lambda: len(_finals(phone)) >= 3)
        await asyncio.sleep(0.2)
        model.say("Okay, I'll include their email addresses.", gap=200, dur=900)
        await asyncio.sleep(0.6)
        a_done.set()
        await _wait(lambda: _terminal(phone, "d1") and _terminal(phone, "d2"), 6.0)
        await asyncio.sleep(0.8)

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=20, db_session_id=_db(name))
    d1 = _frames_for(phone, "d1")
    d2 = [f for f in _frames_for(phone, "d2") if "task_revision" in f]
    relay = {
        "questions": asked(),
        "d1_phases": [f.get("phase") for f in d1],
        "d1_cancel_reason": next((f.get("reason") for f in d1 if f.get("phase") == "cancelled"), None),
        "d2_relations": [f.get("relation") for f in d2],
        "reask_results": _results(phone),
    }
    _record(name, f"A (d1) runs → «{PARTIAL}» (d2) → «Should I stop …?{own_q}» in ONE epoch (heard) → «yes» → "
            "the confirmation is void: A is never cancelled, B relates to nothing; the model acts on its own "
            "question.", [(model.provider._session_id, tape)], relay)
    assert "cancelled" not in relay["d1_phases"], relay
    assert relay["d1_phases"][-1] == "completed", relay
    assert not any(relay["d2_relations"]), relay


# ── C4: a MIXED answer whose remainder the model does not delegate ─────

@pytest.mark.asyncio
async def test_x5_a_mixed_answers_remainder_stays_recoverable(monkeypatch):
    """«yes, and search for waterloo robotics professors» stops A; the model
    delegates nothing.  The relay's "Okay, I stopped «A»." is parented to the
    NEW task's turn, never to the mixed answer, so the app still owes the
    answer turn and asks it by id → accepted with the remainder → the model
    delegates it (old: the stop line covered the mixed turn — the remainder
    was silently lost)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    calls: list[tuple] = []
    a_done = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id") or ""
        calls.append((did, kw.get("display_request") or ""))
        if did == "d1":
            try:
                await asyncio.wait_for(a_done.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            return "Professor X.", "m"
        if did == "d2":
            await asyncio.sleep(3.0)
            return "Booked the Hotel Ocho.", "m"
        await asyncio.sleep(0.3)
        return "Three Waterloo robotics professors: Lee, Patel and Moreno.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)
    name = "mixed_answer_remainder"

    def remainder(e):
        content = str(e.get("content") or "")
        return (lambda m: m.delegate("d-rem")) if f"«{REMAINDER}»" in content else None

    def start(m):
        m.user(PROFESSORS, gap=0, dur=900)
        m.delegate("d1")

    model = Model(f"live-psid-x5-{name}", start=start, answer_reask=remainder)

    def asked():
        return [c for c in model.commentary() if any(mark in c for mark in STOP_MARKS)]

    async def script():
        await _wait(lambda: len(calls) >= 1)
        await asyncio.sleep(0.3)
        model.user(PARTIAL, gap=4000, dur=800)
        model.delegate("d2")
        await _wait(lambda: asked(), 3.0)
        await asyncio.sleep(0.6)                       # the question is said and HEARD
        model.user(MIXED, gap=2500, dur=600)           # … and the model delegates nothing
        await _wait(lambda: _terminal(phone, "d1"), 3.0)
        await asyncio.sleep(0.8)                       # "Okay, I stopped «A»." is said and heard
        answer = next(f for f in _finals(phone) if f["text"] == MIXED)
        phone.reask(answer)                            # the app still owes the mixed turn: by id
        await _wait(lambda: phone.of("reask_result"))
        await _wait(lambda: _terminal(phone, "d-rem"), 4.0)
        await asyncio.sleep(0.8)                       # the remainder's result is said and heard
        a_done.set()
        await _wait(lambda: _terminal(phone, "d2"), 5.0)
        await asyncio.sleep(0.6)

    tape: list = []
    phone = Phone([_config(), {"type": "audio_ready"}, script, 0.3, {"type": "stop"}], tape=tape)
    await H.run_relay(phone, model.provider, timeout=24, db_session_id=_db(name))
    d1 = _frames_for(phone, "d1")
    d2_created = next((f for f in _frames_for(phone, "d2") if f.get("phase") == "created"), {})
    answer = next((f for f in _finals(phone) if f["text"] == MIXED), {})
    relay = {
        "answer_turn": answer.get("turn_id"),
        "d2_turn": d2_created.get("turn_id"),
        "d1_cancel_reason": next((f.get("reason") for f in d1 if f.get("phase") == "cancelled"), None),
        "stop_line_parents": sorted({
            str(f.get("parent_user_turn_id")) for f in phone.of("response_text")
            if "Okay, I stopped" in (f.get("text") or "")
        }),
        "reask_results": _results(phone),
        "reask_instructions": model.reask_contents(),
        "remainder_card": [f for f in _frames_for(phone, "d-rem") if f.get("phase") == "created"][:1],
        "calls": calls,
        "d2_relations": [f.get("relation") for f in _frames_for(phone, "d2") if "task_revision" in f],
        "b_saved": _saved_rows(recorded["saves"], f"live-delegation:{model.provider._session_id}:d2"),
    }
    _record(name, f"A (d1) runs → «{PARTIAL}» (d2) → «Should I stop …?» (heard) → «{MIXED}», the model delegates "
            "nothing → A cancelled; «Okay, I stopped …» is parented to d2's turn (heard) → the app still owes "
            "the mixed turn and asks it by id → accepted with the remainder → the model delegates it → its "
            "result is said and heard.", [(model.provider._session_id, tape)], relay)
    assert relay["d1_cancel_reason"] == "user_cancel", relay
    assert relay["stop_line_parents"] == [relay["d2_turn"]], relay
    assert relay["reask_results"] == [(relay["answer_turn"], "accepted", None)], relay
    content = relay["reask_instructions"][0] if relay["reask_instructions"] else ""
    assert f"«{REMAINDER}»" in content, content
    assert relay["remainder_card"] and relay["remainder_card"][0]["title"] == REMAINDER, relay
    # C10 for this mixed answer too: B's card and B's saved row name the relation.
    assert relay["d2_relations"] and relay["d2_relations"][-1] == {"kind": "replaces", "task_id": "d1"}, relay
    assert relay["b_saved"] and relay["b_saved"][-1]["stored"].get("related_task_id") == "d1", relay["b_saved"]
