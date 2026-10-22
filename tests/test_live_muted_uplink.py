"""P1 (2026-10-02 voice call): a finished result is SAID while the caller's
uplink is muted — and when it cannot be, the relay's verdict is honest.

Production (relay f49c49a6, psid live_u2_EUjm, phone build 138):

    02:29:11.810  progress spoken id=item_EUjmcVJ kind=idle line=1
    02:29:17.445  relay line claim withdrawn reason=no_epoch
    02:29:21.875  progress spoken id=item_EUjmcVJ kind=search line=2
    02:29:27.880  relay line claim withdrawn reason=no_epoch
    02:29:35.397  delegation finished id=item_EUjmcVJ outcome=ok chars=295
    02:29:41.396  result nudge sent id=item_EUjmcVJ reason=no_ack
    02:29:47.397  result NOT spoken id=item_EUjmcVJ reason=no_ack_after_nudge
    02:30:09.366  output epoch opened epoch=7 parent=-   (after the caller unmuted)
    ...
    02:32:02.039  append rejected kind=commentary code=context_injection_incomplete (x2,
                  two job-3 appends sent 22 s and 5 s earlier, answered at hang-up)

What the code did (findings-p1-ladder.md has the line references):

* the phone's mute (`useRealtimeVoice.setMuteTo`) only stops `audio` frames;
  no frame tells the relay, and `client_loop` has no mute branch;
* the relay never originated audio: `client_loop` was the only
  `audio_append` call site, so a muted caller meant NO `session.input_audio`
  reached GPT-Live at all;
* the result is spoken only through `session.commentary.append` (GPT-Live has
  no `response.create`); the ladder waited 6 s for an epoch, nudged with ONE
  `session.instructions.append`, waited 6 s more and sent
  `delegation{phase:completed, spoken:false, result_text}` ("Result in chat").

What the call shows about GPT-Live (inferred from the logs, not documented):
its timeline is its INPUT AUDIO (`offset_ms` advanced 20.6 s over 84.9 s of
wall time across the mute, `skew_ms`=64 810), and with no input it neither
acks nor voices an append — the answer was voiced only once audio resumed.

`InputClockedProvider` models exactly that observed contract and nothing more:
an append is acked (and a commentary voiced) only after >= `consume_ms` of
input audio has arrived since it was sent; at `session.close` every append
still pending is rejected `context_injection_incomplete`.  It encodes the
hypothesis, so these tests prove the RELAY's behaviour under it.
`scripts/live_append_experiment.py` answered it on 2026-10-05 (no input: never
acked; real-time silence: acked ~600 ms), so the keeper now ships ON.

The fix (PR-5), and the test that pins each part — every test below FAILS on
f49c49a6 unless it says "guard" (on f49c49a6 the keeper tests cannot even
set their switch: `voice_live_uplink_keeper_enabled` did not exist):

(a) `derived_state` says `thinking` while an answer is owed, and the adapter's
    epoch-close `listening` is not forwarded while a task runs or an answer
    is owed — `test_derived_state_is_thinking_while_a_delivery_is_pending`,
    `test_no_listening_state_reaches_the_phone_while_work_or_an_answer_is_owed`;
(b) no input since the append ⇒ the deadline is re-armed (`no_input`), bounded
    by `voice_live_speak_no_input_max_s`, and then the usual ladder —
    `test_keeper_off_a_muted_result_is_held_for_input_then_ruled_inside_the_bound`;
(c) a late ack re-files the claim, so the late paraphrase binds its row —
    `test_keeper_off_a_result_taken_late_after_unmute_binds_its_own_row`,
    `test_keeper_off_a_progress_line_taken_late_never_binds_the_results_claim`;
(d) no progress line behind one the provider has not taken, none for ended
    work — `test_keeper_off_no_second_progress_line_is_stacked_behind_a_starved_one`,
    `test_no_progress_line_is_appended_for_a_task_that_already_ended`;
(e) the uplink keeper — the two original repros, now green with it ON, plus
    the production-constants timing, mid-utterance, phone-audio, per-episode
    cap and nothing-owed tests.
"""

import asyncio
import base64
import json
import time

import pytest

import test_live_harness as H
from app.config import settings
from app.services import live_voice_protocol as live


ANSWER = (
    "I couldn't find specific information about the most experienced officer "
    "in the LLM program at U of T."
)
REQUEST = "search who is the most experienced officer in LLM in U of T"

#: `build_session_start`: audio/pcm @ 24 kHz — PCM16 mono, 2 bytes a sample.
PCM_RATE = 24000
FRAME_MS = 50


def pcm_silence_b64(ms: int) -> str:
    return base64.b64encode(b"\x00\x00" * (PCM_RATE * ms // 1000)).decode()


#: One phone mic frame (digital silence is enough: the fake counts samples).
#: 50 ms, so it can never be mistaken for the keeper's 100 ms chunk.
AUDIO = {"type": "audio", "data": pcm_silence_b64(FRAME_MS)}

#: The uplink keeper's chunk, pinned here independently of the module: 100 ms
#: of PCM16 24 kHz digital silence. Any audio append carrying exactly this
#: payload is the RELAY's; the fake phone's frames are 50 ms.
KEEPER_CHUNK = pcm_silence_b64(100)


def test_the_keeper_chunk_is_100_ms_of_pcm16_24k_silence():
    """(e) The shape is load-bearing: GPT-Live's input format is fixed by
    `session.start`, and a chunk of any other length would advance its
    timeline at the wrong pace."""

    assert live._UPLINK_KEEPER_SILENCE_B64 == KEEPER_CHUNK
    raw = base64.b64decode(KEEPER_CHUNK)
    assert raw == b"\x00" * 4800
    assert len(raw) / 2 / PCM_RATE == pytest.approx(0.1)
    # D7 (2026-10-05, measured in a production container): no input → the
    # append is never acked; real-time digital silence → acked ~600 ms. So
    # the keeper ships ON; `fast_clocks` pins it off for every other suite.
    assert settings.voice_live_uplink_keeper_enabled is True, "the keeper ships ON"
    assert settings.voice_live_uplink_keeper_max_s == 45.0
    assert settings.voice_live_speak_no_input_max_s == 24.0, (
        "the verdict must land inside the app's DELIVERY_GRACE_MS (30 s)"
    )


class InputClockedProvider(H.FakeProvider):
    """GPT-Live as the 2026-10-02 call shows it: context injections are only
    consumed while its input-audio timeline advances.

    `voice_gap_s`: with 0 (the original repro) everything consumed in one go
    is voiced back to back; above 0 each voicing waits that long after the
    previous one, so two appends taken together still become two epochs."""

    def __init__(self, on_send=None, *, consume_ms: float = 100.0, voice_gap_s: float = 0.0):
        # auto_ack stays OFF: every ack below is conditional on input audio.
        super().__init__(on_send=on_send)
        self.consume_ms = float(consume_ms)
        self.voice_gap_s = float(voice_gap_s)
        self.audio_ms = 0.0
        #: (monotonic, cumulative input audio ms) after each audio append.
        self.audio_log: list[tuple[float, float]] = []
        #: (monotonic, payload) of every audio append, phone's and relay's.
        self.audio_events: list[tuple[float, str]] = []
        #: [(event, audio_ms when it arrived)]
        self.pending: list[tuple[dict, float]] = []
        #: [(event, monotonic when consumed)]
        self.consumed: list[tuple[dict, float]] = []
        #: event ids answered `context_injection_incomplete` at close.
        self.incomplete: list[str] = []
        self.voiced: list[str] = []
        self._t = 20_000
        self._voice_queue: list[str] = []
        self._voicer = None

    async def send(self, raw):
        event = json.loads(raw)
        kind = str(event.get("type") or "")
        if kind == "session.input_audio.append":
            payload = str(event.get("audio") or "")
            try:
                pcm = base64.b64decode(payload)
            except Exception:  # noqa: BLE001 — a malformed frame advances nothing
                pcm = b""
            self.audio_ms += len(pcm) / 2 / PCM_RATE * 1000.0
            now = time.monotonic()
            self.audio_log.append((now, self.audio_ms))
            self.audio_events.append((now, payload))
        elif kind in H.APPEND_ACKS:
            self.pending.append((event, self.audio_ms))
        elif kind == "session.close":
            # Production 02:32:02.039: what was never consumed is refused at
            # hang-up, BEFORE `session.closed` (pushed by `on_send`).
            for pending, _at in self.pending:
                event_id = str(pending.get("event_id") or "")
                self.incomplete.append(event_id)
                self.push({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "context_injection_incomplete",
                        "client_event_id": event_id,
                    },
                })
            self.pending = []
        await super().send(raw)
        self._consume()

    def _consume(self) -> None:
        still: list[tuple[dict, float]] = []
        for event, at in self.pending:
            if self.audio_ms - at >= self.consume_ms:
                self._take(event)
            else:
                still.append((event, at))
        self.pending = still

    def _take(self, event: dict) -> None:
        kind = str(event["type"])
        event_id = str(event.get("event_id") or "")
        self.consumed.append((event, time.monotonic()))
        self.push({
            "type": H.APPEND_ACKS[kind], "client_event_id": event_id,
            "start_ms": 0, "end_ms": 0,
        })
        speaks = kind == "session.commentary.append" or (
            kind == "session.instructions.append" and event_id.startswith("toup-live-nudge-")
        )
        if speaks:
            # The model paraphrases it aloud: a new output epoch.
            text = "Voiced: " + str(event.get("content") or "")[:60]
            self.voiced.append(text)
            if self.voice_gap_s <= 0:
                self._say(text)
            else:
                self._voice_queue.append(text)
                if self._voicer is None or self._voicer.done():
                    self._voicer = asyncio.get_running_loop().create_task(self._voice_all())

    def _say(self, text: str) -> None:
        start = self._t
        self._t += 2000
        self.push(H.out_text(text, start, start + 700))
        self.push(H.out_audio("SPEAKING"))

    async def _voice_all(self) -> None:
        while self._voice_queue:
            self._say(self._voice_queue.pop(0))
            await asyncio.sleep(self.voice_gap_s)

    def audio_since(self, moment: float) -> float:
        before = max((ms for at, ms in self.audio_log if at <= moment), default=0.0)
        return self.audio_ms - before

    def keeper_times(self) -> list[float]:
        """When each RELAY-originated audio chunk arrived (the keeper's)."""

        return [at for at, payload in self.audio_events if payload == KEEPER_CHUNK]

    def phone_times(self) -> list[float]:
        return [at for at, payload in self.audio_events if payload != KEEPER_CHUNK]


class RecordingProvider(H.FakeProvider):
    """A provider that never acks, recording when each audio append arrived."""

    def __init__(self, on_send=None, **kwargs):
        super().__init__(on_send=on_send, **kwargs)
        self.audio_events: list[tuple[float, str]] = []

    async def send(self, raw):
        event = json.loads(raw)
        if event.get("type") == "session.input_audio.append":
            self.audio_events.append((time.monotonic(), str(event.get("audio") or "")))
        await super().send(raw)

    def keeper_times(self) -> list[float]:
        return [at for at, payload in self.audio_events if payload == KEEPER_CHUNK]

    def phone_times(self) -> list[float]:
        return [at for at, payload in self.audio_events if payload != KEEPER_CHUNK]


def _keeper(monkeypatch, on: bool) -> None:
    if not on and not hasattr(settings, "voice_live_uplink_keeper_enabled"):
        return      # a relay without a keeper is a relay with the keeper off
    monkeypatch.setattr(settings, "voice_live_uplink_keeper_enabled", on)


def _script(box, *, request=REQUEST):
    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 900))
            provider.push(H.delegation("d1", 950))
        elif event["type"] == "session.close":
            provider.push(H.closed())
    return on_send


def _capture(monkeypatch) -> list[tuple[str, dict, float]]:
    """`_vcount` with a timestamp, so a failure prints the ladder's timeline."""

    from app.api import ws_realtime as rt

    seen: list[tuple[str, dict, float]] = []

    def _count(name, _user_id, **fields):
        seen.append((name, fields, time.monotonic()))

    monkeypatch.setattr(rt, "_vcount", _count)
    return seen


async def _wait(pred, timeout: float = 4.0) -> bool:
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def _timeline(counters, origin: float) -> list[str]:
    keep = (
        "live_progress_spoken", "live_relay_claim_withdrawn", "live_result_nudged",
        "live_result_unspoken", "live_commentary_rejected", "live_delegation_done",
        "live_result_rearmed", "live_uplink_keeper", "live_result_late_ack",
        "live_result_spoken_late", "live_relay_line_late_ack",
    )
    return [
        f"+{(at - origin) * 1000:.0f}ms {name} {fields}"
        for name, fields, at in counters if name in keep
    ]


def _first(counters, name: str, **fields) -> float:
    for counted, got, at in counters:
        if counted == name and all(got.get(k) == v for k, v in fields.items()):
            return at
    return 0.0


def _delegated_spoken_flags(saves) -> list:
    return [
        s["assistant_voice"]["spoken"] for s in saves
        if str(s.get("assistant_ref") or "").startswith("live-delegation:")
        and "spoken" in (s.get("assistant_voice") or {})
    ]


def _result_consumed(provider) -> list[tuple[dict, float]]:
    return [
        (e, at) for e, at in provider.consumed
        if e["type"] == "session.commentary.append" and ANSWER[:40] in e["content"]
    ]


def _epochs(client) -> dict:
    """assistant_turn_id → {"parent", "text", "epoch"} from the wire."""

    rows: dict = {}
    for frame in client.of("response_text"):
        row = rows.setdefault(frame.get("assistant_turn_id"), {
            "parent": frame.get("parent_user_turn_id"), "text": "",
            "epoch": frame.get("epoch"),
        })
        row["text"] += frame.get("text") or ""
    return rows


def _request_turn(client) -> str:
    finals = [f for f in client.of("transcript") if f.get("final")]
    assert finals, client.frames
    return finals[0]["turn_id"]


async def _run_result_scenario(
    monkeypatch, *, uplink: str, db: str, keeper: bool, clocks=None, rest=None,
    provider_kwargs=None, box=None,
):
    H.fast_clocks(monkeypatch, **{
        "voice_live_delegation_ttl_s": 3.0, "voice_live_speak_timeout_s": 1.0,
        **(clocks or {}),
    })
    _keeper(monkeypatch, keeper)
    counters = _capture(monkeypatch)
    marks: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.4)          # the job runs while the caller waits
        marks["finished"] = time.monotonic()
        return ANSWER, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)
    request_audio = [AUDIO, FRAME_MS / 1000.0] * 4      # the caller asks…
    if rest is not None:
        tail = list(rest)
    elif uplink == "muted":
        # …and taps mute: build 138 sends NOTHING more (no frame says so).
        tail = [3.5]
    else:
        # …and stays unmuted: the phone keeps streaming its mic.
        tail = [AUDIO, FRAME_MS / 1000.0] * 70
    client = H.FakeClient([H.config(), *request_audio, *tail, {"type": "stop"}])
    box = {} if box is None else box
    box["client"] = client
    provider = InputClockedProvider(on_send=_script(box), **(provider_kwargs or {}))
    started = time.monotonic()
    await H.run_relay(client, provider, timeout=20, db_session_id=db)
    return counters, recorded, client, provider, marks, started


# ══════════════════════════════════════════════════════════════════════
# CONTROL — the fake voices the result while the uplink flows (guard)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_control_the_result_is_spoken_while_the_uplink_flows(monkeypatch):
    counters, recorded, client, provider, marks, started = await _run_result_scenario(
        monkeypatch, uplink="flowing", db="db-p1-control", keeper=False,
    )
    names = [name for name, _fields, _at in counters]
    diag = (
        f"audio after finish={provider.audio_since(marks.get('finished', started)):.0f}ms "
        f"consumed={[e['type'] for e, _ in provider.consumed]} "
        f"timeline={_timeline(counters, marks.get('finished', started))}"
    )
    assert _result_consumed(provider), "control broken: the fake never voiced the result " + diag
    assert not [
        f for f in client.of("delegation")
        if f.get("delegation_id") == "d1" and f.get("spoken") is False
    ], diag
    assert "live_result_unspoken" not in names, diag
    assert "live_result_nudged" not in names, diag
    assert "live_result_rearmed" not in names, "input was flowing: nothing to hold " + diag
    flags = _delegated_spoken_flags(recorded["saves"])
    assert flags and flags[-1] is True, f"flags={flags} " + diag
    assert provider.keeper_times() == [], "keeper off: the relay originates no audio"


# ══════════════════════════════════════════════════════════════════════
# (e) THE REPRO, keeper ON — the result is said while the caller is muted
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_repro_a_result_finishing_while_the_uplink_is_muted_is_spoken(monkeypatch):
    """P1 (a)/(b): the ladder waited for an ack/epoch GPT-Live cannot produce
    without input audio, nudged once into the same void, and handed the answer
    to the chat ("Result in chat") although the caller was still on the call.
    With the uplink keeper on, real-time silence keeps the input moving."""

    counters, recorded, client, provider, marks, started = await _run_result_scenario(
        monkeypatch, uplink="muted", db="db-p1-muted", keeper=True,
    )
    names = [name for name, _fields, _at in counters]
    finished = marks.get("finished", started)
    unspoken_frames = [
        f for f in client.of("delegation")
        if f.get("delegation_id") == "d1" and f.get("spoken") is False
    ]
    diag = (
        f"\n  input audio reaching GPT-Live after the task finished: "
        f"{provider.audio_since(finished):.0f} ms"
        f"\n  appends consumed: {[e['type'] for e, _ in provider.consumed]}"
        f"\n  appends refused at close (context_injection_incomplete): {len(provider.incomplete)}"
        f"\n  session.input_audio.append received by GPT-Live: "
        f"{len(provider.of('session.input_audio.append'))} total, "
        f"{len(provider.keeper_times())} of them the relay's own silence"
        f"\n  ladder timeline (from finish): {_timeline(counters, finished)}"
        f"\n  'Result in chat' frames: {unspoken_frames}"
    )
    assert any(ANSWER[:40] in c["content"] for c in provider.of("session.commentary.append")), (
        "scenario broken: the relay never appended the result" + diag
    )
    assert _result_consumed(provider), (
        "P1 REPRODUCED — GPT-Live never consumed the result commentary while the "
        "uplink was muted:" + diag
    )
    assert not unspoken_frames, (
        "P1 REPRODUCED — the call surface was handed completed{spoken:false} "
        "('Result in chat') for an answer that was never given a chance to be spoken:" + diag
    )
    assert "live_result_unspoken" not in names, diag
    flags = _delegated_spoken_flags(recorded["saves"])
    assert flags and flags[-1] is True, f"flags={flags}" + diag
    # The phone's four request frames are untouched; everything else is the
    # keeper's 100 ms chunk, and none of it before the task finished.
    assert len(provider.phone_times()) == 4, diag
    assert provider.keeper_times() and min(provider.keeper_times()) >= finished - 0.05, diag


@pytest.mark.asyncio
async def test_repro_a_progress_line_while_the_uplink_is_muted_is_voiced(monkeypatch):
    """P1 (c): the same mechanism behind 'relay line claim withdrawn
    reason=no_epoch' — the spoken progress line is a commentary append too.
    With the keeper on it is voiced inside its speak budget."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_speak_timeout_s=0.6,
    )
    monkeypatch.setattr(settings, "voice_live_progress_speak_after_s", 0.2)
    _keeper(monkeypatch, True)
    counters = _capture(monkeypatch)
    box: dict = {}
    marks: dict = {}

    def _progress_appends() -> list[dict]:
        p = box.get("p")
        if p is None:
            return []
        return [
            e for e in p.of("session.commentary.append")
            if "Still working on it" in str(e.get("content") or "")
        ]

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await relay.on_event({
            "type": "tool.start", "call_id": "c1", "name": "web_search",
            "args": {"query": "u of t llm faculty"},
        })
        await _wait(lambda: bool(_progress_appends()), 3.0)
        marks["progress"] = time.monotonic()
        await asyncio.sleep(1.0)        # well past the 0.6 s speak budget
        await relay.on_event({
            "type": "tool.end", "call_id": "c1", "name": "web_search",
            "ok": True, "preview": "found", "elapsed_ms": 20,
        })
        return ANSWER, "test-model"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([
        H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4), 4.5, {"type": "stop"},
    ])
    provider = InputClockedProvider(on_send=_script(box))
    started = time.monotonic()
    await H.run_relay(client, provider, timeout=15, db_session_id="db-p1-progress")

    names = [name for name, _fields, _at in counters]
    origin = marks.get("progress", started)
    withdrawn = [
        fields for name, fields, _at in counters
        if name == "live_relay_claim_withdrawn" and fields.get("reason") == "no_epoch"
    ]
    voiced_progress = [
        e for e, _at in provider.consumed
        if e["type"] == "session.commentary.append" and "Still working on it" in e["content"]
    ]
    diag = (
        f"\n  progress appends sent: {[e['content'] for e in _progress_appends()]}"
        f"\n  input audio after the line: {provider.audio_since(origin):.0f} ms"
        f"\n  timeline (from the line): {_timeline(counters, origin)}"
    )
    assert "live_progress_spoken" in names, "scenario broken: no progress line" + diag
    assert voiced_progress, (
        "P1(c) REPRODUCED — the progress line was never consumed/voiced while muted:" + diag
    )
    assert not withdrawn, (
        "P1(c) REPRODUCED — 'relay line claim withdrawn reason=no_epoch':" + diag
    )
    # …and the result after it is said too, never handed to the chat.
    assert _result_consumed(provider), diag
    assert "live_result_unspoken" not in names, diag


@pytest.mark.asyncio
async def test_with_production_constants_a_muted_result_is_spoken_within_5_s(monkeypatch):
    """(e) The production budgets (6 s speak timeout, 4 s defer, 24 s
    no-input bound): muted, the result is consumed and voiced within 5 s of
    the job finishing — no nudge, no hold, no "Result in chat"."""

    counters, recorded, client, provider, marks, started = await _run_result_scenario(
        monkeypatch, uplink="muted", db="db-p1-prod", keeper=True,
        clocks={
            "voice_live_speak_timeout_s": 6.0,
            "voice_live_result_defer_max_ms": 4000,
            "voice_live_speak_no_input_max_s": 24.0,
        },
    )
    names = [name for name, _fields, _at in counters]
    finished = marks["finished"]
    diag = f"\n  timeline (from finish): {_timeline(counters, finished)}"
    taken = _result_consumed(provider)
    assert taken, "the result was never consumed" + diag
    assert taken[0][1] - finished < 5.0, f"spoken {taken[0][1] - finished:.2f} s after" + diag
    assert "live_result_nudged" not in names, diag
    assert "live_result_rearmed" not in names, "the keeper fed the input: nothing to hold" + diag
    assert "live_result_unspoken" not in names, diag
    flags = _delegated_spoken_flags(recorded["saves"])
    assert flags and flags[-1] is True, flags


# ══════════════════════════════════════════════════════════════════════
# (b) keeper OFF — held for input, then the same honest ladder, in time
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_keeper_off_a_muted_result_is_held_for_input_then_ruled_inside_the_bound(
    monkeypatch,
):
    """With nothing feeding GPT-Live's input, the 1.0 s speak budget is
    re-armed (`no_input`) instead of nudging into the void at 1.0 s; the hold
    is bounded (2.5 s here, 24 s in production) so the verdict still lands
    inside it: the nudge leaves one budget early, and a provider that never
    acks still ends `no_ack_after_nudge` + `completed{spoken:false}`."""

    counters, recorded, client, provider, marks, started = await _run_result_scenario(
        monkeypatch, uplink="muted", db="db-p1-hold", keeper=False,
        clocks={"voice_live_speak_no_input_max_s": 2.5},
    )
    names = [name for name, _fields, _at in counters]
    finished = marks["finished"]
    diag = f"\n  timeline (from finish): {_timeline(counters, finished)}"
    assert ("live_result_rearmed", {"reason": "no_input"}) in [
        (n, f) for n, f, _at in counters
    ], "the no-input hold never happened" + diag
    nudged = _first(counters, "live_result_nudged", reason="no_ack")
    ruled = _first(counters, "live_result_unspoken", reason="no_ack_after_nudge")
    assert nudged and ruled, diag
    assert nudged - finished >= 1.3, (
        f"nudged {nudged - finished:.2f} s after finish — the 1.0 s budget expired "
        "with no input reaching the provider" + diag
    )
    assert ruled - finished <= 2.5 + 0.35, (
        f"verdict {ruled - finished:.2f} s after finish — outside the bound" + diag
    )
    assert names.count("live_result_nudged") == 1, diag
    unspoken = [f for f in client.of("delegation") if f.get("spoken") is False]
    # OC-2: the verdict carries the answer (no card holds it), and with no
    # input since the append no second copy is queued — it would be voiced
    # twice once input moves.
    assert len(unspoken) == 1 and unspoken[0].get("result_text") == ANSWER, unspoken
    assert "live_result_requeued" not in names, diag
    assert _delegated_spoken_flags(recorded["saves"]) == [False]
    assert provider.keeper_times() == [], "keeper off: the relay originates no audio"


# ══════════════════════════════════════════════════════════════════════
# (c) late acks re-file the claim
# ══════════════════════════════════════════════════════════════════════

async def _wait_for_completed(client) -> None:
    await _wait(
        lambda: any(f.get("phase") == "completed" for f in client.of("delegation")), 6.0,
    )


@pytest.mark.asyncio
async def test_keeper_off_a_result_taken_late_after_unmute_binds_its_own_row(monkeypatch):
    """The production tail: the verdict went out while muted, the caller
    unmuted, and GPT-Live voiced the backlog as an epoch with NO parent — a
    second row beside the delegated one. The late ack now re-files the claim:
    the paraphrase is parented on the request, folds onto the delegated row,
    and that row's `spoken` flag is revised to true. Nothing new is sent to
    the phone (the relay never sends `spoken:true`)."""

    box: dict = {}

    async def wait_verdict():
        await _wait(lambda: any(
            f.get("spoken") is False for f in box["client"].of("delegation")
        ), 6.0)
        await asyncio.sleep(0.3)

    rest = [wait_verdict, *([AUDIO, FRAME_MS / 1000.0] * 30), 0.5]
    counters, recorded, client, provider, marks, started = await _run_result_scenario(
        monkeypatch, uplink="muted", db="db-p1-late", keeper=False, rest=rest, box=box,
        clocks={"voice_live_speak_timeout_s": 0.5, "voice_live_speak_no_input_max_s": 1.6},
    )
    names = [name for name, _fields, _at in counters]
    diag = (
        f"\n  timeline: {_timeline(counters, marks.get('finished', started))}"
        f"\n  epochs: {_epochs(client)}"
    )
    assert "live_result_unspoken" in names, "scenario: the verdict precedes the unmute" + diag
    assert "live_result_late_ack" in names, diag
    assert "live_result_spoken_late" in names, diag
    request_turn = _request_turn(client)
    late = [row for row in _epochs(client).values() if ANSWER[:30] in row["text"]]
    assert late, "the late paraphrase never reached the phone" + diag
    assert all(row["parent"] == request_turn for row in late), (
        "the late paraphrase was minted without its request as parent" + diag
    )
    flags = _delegated_spoken_flags(recorded["saves"])
    # The verdict, then the late correction (and the fold of the heard epoch,
    # which carries the corrected flag on the same row).
    assert flags[:2] == [False, True] and set(flags[1:]) == {True}, flags
    loose = [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_spoken"
        and ANSWER[:30] in (s.get("assistant_text") or "")
    ]
    assert loose == [], "the late paraphrase was stored as a row of its own"
    # D-P1: the phone may show an unspoken badge — ONE additive
    # `completed{spoken:true}` revokes it (it used to send nothing).
    revoked = [f for f in client.of("delegation") if f.get("spoken") is True]
    assert len(revoked) == 1 and revoked[0]["phase"] == "completed", revoked


@pytest.mark.asyncio
async def test_keeper_off_a_progress_line_taken_late_never_binds_the_results_claim(
    monkeypatch,
):
    """A progress line withdrawn unvoiced while muted is still waiting at the
    provider. The job finishes, its result is appended (the result's claim is
    now the head of the FIFO) — and the caller unmutes. GPT-Live voices the
    OLD line first: its epoch used to bind the result's claim (the answer
    recorded heard on the strength of "still working on it", the real
    paraphrase left parentless). The line's late ack re-files ITS claim
    first, so each epoch binds its own."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_speak_timeout_s=0.6,
        voice_live_speak_no_input_max_s=4.0,
    )
    monkeypatch.setattr(settings, "voice_live_progress_speak_after_s", 0.2)
    _keeper(monkeypatch, False)
    counters = _capture(monkeypatch)
    box: dict = {}

    def progress_sent() -> bool:
        p = box.get("p")
        return bool(p and any(
            "Still working on it" in str(e.get("content") or "")
            for e in p.of("session.commentary.append")
        ))

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await relay.on_event({
            "type": "tool.start", "call_id": "c1", "name": "web_search",
            "args": {"query": "u of t llm faculty"},
        })
        await _wait(progress_sent, 3.0)
        await asyncio.sleep(1.0)        # the line's claim is withdrawn meanwhile
        await relay.on_event({
            "type": "tool.end", "call_id": "c1", "name": "web_search",
            "ok": True, "preview": "found", "elapsed_ms": 20,
        })
        return ANSWER, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    async def unmute_after_finish():
        await _wait(lambda: any(
            f.get("phase") == "completed" for f in client.of("delegation")
        ), 6.0)
        await asyncio.sleep(0.3)

    async def answer_epoch_retired():
        # PR-7 (F6b) keeps an output epoch whose text does not end a sentence
        # open for `voice_live_output_epoch_hard_gap_ms` (2.5 s) before it
        # retires it, and this fake voices only the first 60 characters of the
        # answer — mid-sentence.  The answer's epoch is folded onto the
        # delegated row when it RETIRES, so the call must outlast that hold
        # before it hangs up (bounded: a relay that never retires it still
        # fails the fold assertion below).
        await _wait(lambda: any(
            ANSWER[:30] in str(f.get("text") or "")
            for f in client.of("speech_segment_complete")
        ), 4.0)

    client = H.FakeClient([
        H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4), unmute_after_finish,
        *([AUDIO, FRAME_MS / 1000.0] * 40), answer_epoch_retired, 0.3,
        {"type": "stop"},
    ])
    provider = InputClockedProvider(on_send=_script(box), voice_gap_s=0.3)
    await H.run_relay(client, provider, timeout=20, db_session_id="db-p1-late-line")

    names = [name for name, _fields, _at in counters]
    epochs = _epochs(client)
    diag = f"\n  timeline: {_timeline(counters, 0)}\n  epochs: {epochs}"
    assert "live_relay_claim_withdrawn" in names, "scenario: the line was withdrawn" + diag
    assert "live_relay_line_late_ack" in names, diag
    request_turn = _request_turn(client)
    line_rows = [r for r in epochs.values() if "Still working on it" in r["text"]]
    answer_rows = [r for r in epochs.values() if ANSWER[:30] in r["text"]]
    assert line_rows and answer_rows, diag
    assert {r["parent"] for r in line_rows} == {request_turn}, diag
    assert {r["parent"] for r in answer_rows} == {request_turn}, diag
    folded = [
        s["assistant_voice"] for s in recorded["saves"]
        if str(s.get("assistant_ref") or "").startswith("live-delegation:")
        and "epoch" in (s.get("assistant_voice") or {})
    ]
    assert folded, "the answer's epoch was never folded onto its row" + diag
    assert {v["epoch"] for v in folded} == {r["epoch"] for r in answer_rows}, (
        "the delegated row recorded the progress line's epoch as the answer being "
        "heard" + diag
    )
    flags = _delegated_spoken_flags(recorded["saves"])
    assert flags and flags[-1] is True, flags


# ══════════════════════════════════════════════════════════════════════
# (d) progress lines: never stacked into a void, never for ended work
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_keeper_off_no_second_progress_line_is_stacked_behind_a_starved_one(
    monkeypatch,
):
    """Production appended a second progress line ten seconds after the first
    one had been withdrawn unvoiced — both still waiting at the provider, so
    the caller later heard them back to back, stale. A real change of tool
    kind still earns no second line while the first was never taken."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_speak_timeout_s=0.4,
    )
    monkeypatch.setattr(settings, "voice_live_progress_speak_after_s", 0.2)
    _keeper(monkeypatch, False)
    counters = _capture(monkeypatch)
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        for index, name in enumerate(("web_search", "web_fetch")):
            await relay.on_event({
                "type": "tool.start", "call_id": f"c{index}", "name": name,
                "args": {"query": "u of t llm faculty"},
            })
            await asyncio.sleep(1.0)
            await relay.on_event({
                "type": "tool.end", "call_id": f"c{index}", "name": name,
                "ok": True, "preview": "found", "elapsed_ms": 20,
            })
        return ANSWER, "test-model"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([
        H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4), 3.0, {"type": "stop"},
    ])
    provider = InputClockedProvider(on_send=_script(box))
    await H.run_relay(client, provider, timeout=15, db_session_id="db-p1-stack")

    lines = [
        e["content"] for e in provider.of("session.commentary.append")
        if "Still working on it" in str(e.get("content") or "")
    ]
    diag = f"\n  lines: {lines}\n  timeline: {_timeline(counters, 0)}"
    assert "live_relay_claim_withdrawn" in [n for n, _f, _a in counters], (
        "scenario: the first line was withdrawn unvoiced" + diag
    )
    assert len(lines) == 1, diag


def test_progress_is_never_due_for_a_task_that_already_ended():
    """(d) unit: the flag the runner sets in its `finally` closes the window
    in which an ended task still sat in `supervisor.running`."""

    from app.services.live_voice_protocol import (
        DelegatedTask, RunningDelegation, _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.settings = settings
    session.adapter = None
    task = DelegatedTask(delegation_id="d1", offset_ms=0, transcript=REQUEST, turn_id="t1")
    entry = RunningDelegation(task=task, started_monotonic=time.monotonic() - 100.0)
    assert session.progress_speech_due(entry) > 0, "control: a quiet toolless task is due"
    entry.ended = True
    assert session.progress_speech_due(entry) == 0.0


@pytest.mark.asyncio
async def test_no_progress_line_is_appended_for_a_task_that_already_ended(monkeypatch):
    """The agent turn returns just before its idle line falls due; the runner
    is then slow to reach `finish_delegation` (here: held 0.6 s). The entry is
    still in `supervisor.running` for that window, and the line used to go
    out — "still working on it" about work that had already ended."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_progress_speak_after_s", 0.1)
    monkeypatch.setattr(settings, "voice_live_progress_idle_after_s", 0.35)
    box: dict = {}
    marks: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.25)
        marks["returned"] = time.monotonic()
        return ANSWER, "test-model"

    original = live._LiveSession.finish_delegation

    async def slow_finish(self, *args, **kwargs):
        await asyncio.sleep(0.6)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(live._LiveSession, "finish_delegation", slow_finish)
    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 1.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_script(box), auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-p1-ended")

    # The toolless line reads "Still working on «title» — one moment."
    lines = [
        e["content"] for e in provider.of("session.commentary.append")
        if str(e.get("content") or "").startswith("Still working")
    ]
    assert "returned" in marks
    assert lines == [], f"a progress line was appended after the task ended: {lines}"
    assert any(ANSWER in e["content"] for e in provider.of("session.commentary.append"))


# ══════════════════════════════════════════════════════════════════════
# (a) the phone is never told `listening` while work or an answer is owed
# ══════════════════════════════════════════════════════════════════════

def test_derived_state_is_thinking_while_a_delivery_is_pending():
    from app.services.live_voice_protocol import (
        DelegatedTask, DelegationSupervisor, LiveClientAdapter, ResultDelivery, _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.adapter = LiveClientAdapter(session_id=H.PSID)
    session.supervisor = DelegationSupervisor()
    session.active_delivery = None
    session.delivery_queue = []
    session.results_owed = set()
    assert session.derived_state() == "listening"

    task = DelegatedTask(delegation_id="d1", offset_ms=0, transcript=REQUEST, turn_id="t1")
    session.results_owed = {"d1"}            # finish decided to say it, not queued yet
    assert session.derived_state() == "thinking"
    session.results_owed = set()
    queued = ResultDelivery(task=task, answer=ANSWER, claim={})
    session.delivery_queue = [queued]        # waiting for a clean boundary
    assert session.derived_state() == "thinking"
    session.delivery_queue = []
    session.active_delivery = queued         # appended, no epoch yet
    assert session.derived_state() == "thinking"
    assert session.listening_withheld()
    queued.resolved = True                   # said (or honestly not)
    assert session.derived_state() == "listening"
    assert not session.listening_withheld()
    assert live.LIVE_STATES == ("listening", "thinking", "tool_use", "speaking")


@pytest.mark.asyncio
async def test_no_listening_state_reaches_the_phone_while_work_or_an_answer_is_owed(
    monkeypatch,
):
    """Two paths sent `listening` mid-job: every epoch close (the model's
    "okay, let me check" while the task runs), and `finish_delegation`'s
    reassert right before the answer is appended. Build 138's silent idle
    starts on `listening` + 6 s of quiet and then stops the uplink — so the
    answer could never be said. The phone now sees `thinking` until the
    answer's verdict."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.2)
        box["p"].push(H.out_text("Okay, let me check.", 1200, 1700))
        box["p"].push(H.out_audio("OK"))
        await asyncio.sleep(0.5)            # the epoch closes on the gap meanwhile
        return ANSWER, "test-model"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 2.5, {"type": "stop"}])
    # Never acks, never speaks: the answer stays owed until its verdict.
    provider = H.FakeProvider(on_send=_script(box))
    await H.run_relay(client, provider, timeout=10, db_session_id="db-p1-state")

    frames = client.frames
    started = next(
        i for i, f in enumerate(frames)
        if f.get("type") == "delegation" and f.get("phase") == "started"
    )
    verdict = next(
        i for i, f in enumerate(frames)
        if f.get("type") == "delegation" and f.get("spoken") is False
    )
    window = [f["state"] for f in frames[started:verdict] if f.get("type") == "state"]
    assert "speaking" in window, f"scenario: the mid-job reply reached the phone {window}"
    assert "thinking" in window, window
    assert "listening" not in window, (
        f"`listening` while a task ran or its answer was owed: {window}"
    )
    after = [f["state"] for f in frames[verdict:] if f.get("type") == "state"]
    assert "listening" in after, f"back to listening once nothing is owed: {after}"


# ══════════════════════════════════════════════════════════════════════
# (e) the keeper's boundaries
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_the_keeper_never_fires_while_the_caller_is_mid_utterance(monkeypatch):
    """An answer is owed and the phone has gone quiet — but transcript of the
    caller's words is still arriving (the phone's frames are late, or the
    words were buffered). Silence spliced into speech would corrupt the
    caller's turn; the keeper waits until the turn is over."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0,
        voice_live_utterance_gap_ms=1500, voice_live_utterance_hard_gap_ms=3000,
        voice_live_speak_timeout_s=3.0,
    )
    _keeper(monkeypatch, True)
    box: dict = {}
    marks: dict = {"deltas": []}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        return ANSWER, "test-model"

    H.patch_relay(monkeypatch, think=think)

    async def talk():
        await _wait(lambda: any(
            f.get("phase") == "completed" for f in client.of("delegation")
        ), 6.0)
        start = 5000
        for word in "and also can you check the other campus please".split():
            box["p"].push(H.user_delta(" " + word, start, start + 80))
            marks["deltas"].append(time.monotonic())
            start += 100
            await asyncio.sleep(0.12)

    client = H.FakeClient([
        H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4), talk, 2.0, {"type": "stop"},
    ])
    provider = RecordingProvider(on_send=_script(box))
    await H.run_relay(client, provider, timeout=12, db_session_id="db-p1-midutt")

    first, last = marks["deltas"][0], marks["deltas"][-1]
    keeper = provider.keeper_times()
    inside = [t for t in keeper if first + 0.05 <= t <= last + 0.9]
    assert keeper, "scenario: the keeper ran once the caller had finished"
    assert inside == [], (
        f"{len(inside)} keeper chunks spliced into the caller's utterance "
        f"(deltas {first:.2f}..{last:.2f}, chunks {[round(t, 2) for t in keeper]})"
    )
    assert max(keeper) > last + 0.9, "and it resumed after the turn was over"


@pytest.mark.asyncio
async def test_the_keeper_stops_the_moment_phone_audio_arrives(monkeypatch):
    """Muted → the keeper feeds the provider; the caller unmutes → not one
    keeper chunk while phone frames flow, or within 500 ms of the last one;
    muted again → it resumes. It is real time: ~10 chunks a second, never a
    burst."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_speak_timeout_s=3.0,
    )
    _keeper(monkeypatch, True)
    box: dict = {}
    marks: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        marks["finished"] = time.monotonic()
        return ANSWER, "test-model"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([
        H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4),
        1.5,                                         # muted: keeper on
        *([AUDIO, FRAME_MS / 1000.0] * 16),          # unmuted 0.8 s
        1.4,                                         # muted again
        {"type": "stop"},
    ])
    # Never acks: the answer stays owed for the whole call.
    provider = RecordingProvider(on_send=_script(box))
    await H.run_relay(client, provider, timeout=12, db_session_id="db-p1-phone")

    phone = provider.phone_times()
    keeper = provider.keeper_times()
    unmuted = [t for t in phone[4:]]
    assert len(unmuted) == 16, phone
    before = [t for t in keeper if t < unmuted[0]]
    after = [t for t in keeper if t > unmuted[-1]]
    assert before, "the keeper never fed a muted caller's provider"
    assert after, "the keeper never resumed after the caller muted again"
    for t in keeper:
        # 10 ms of slack: the relay stamps a phone frame just BEFORE the
        # provider receives it.
        recent = [p for p in phone if t - 0.49 < p <= t]
        assert not recent, f"keeper chunk at {t:.3f} within 500 ms of phone audio {recent}"
    assert min(after) - unmuted[-1] >= 0.49
    # Real-time pace: one chunk per 100 ms slot, never a catch-up burst (a
    # late slot is skipped rather than sent on the heels of the one before).
    gaps = [b - a for a, b in zip(before, before[1:])]
    assert gaps and min(gaps) >= 0.045, gaps
    span = before[-1] - before[0]
    assert len(before) <= span / 0.1 + 1.5, (len(before), span)


@pytest.mark.asyncio
async def test_the_keeper_is_capped_per_episode(monkeypatch):
    """`voice_live_uplink_keeper_max_s` bounds one episode: past it the
    keeper waits, sending nothing, until the episode ends (here: the phone
    is heard); the next episode gets a fresh allowance."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_speak_timeout_s=3.0,
    )
    _keeper(monkeypatch, True)
    monkeypatch.setattr(settings, "voice_live_uplink_keeper_max_s", 0.5)
    counters = _capture(monkeypatch)
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        return ANSWER, "test-model"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([
        H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4),
        2.0,                                         # muted: one capped episode
        *([AUDIO, FRAME_MS / 1000.0] * 4),           # heard: the episode ends
        1.6,                                         # muted: a new episode
        {"type": "stop"},
    ])
    # Never acks: the answer stays owed for the whole call.
    provider = RecordingProvider(on_send=_script(box))
    await H.run_relay(client, provider, timeout=12, db_session_id="db-p1-cap")

    phone = provider.phone_times()
    keeper = provider.keeper_times()
    heard = phone[4:]
    assert len(heard) == 4, phone
    first = [t for t in keeper if t < heard[0]]
    second = [t for t in keeper if t > heard[-1]]
    assert 4 <= len(first) <= 6, f"first episode sent {len(first)} chunks (cap 0.5 s)"
    assert 4 <= len(second) <= 6, f"second episode sent {len(second)} chunks (cap 0.5 s)"
    assert "live_uplink_keeper_capped" in [name for name, _f, _at in counters]


@pytest.mark.asyncio
async def test_with_nothing_owed_the_relay_sends_no_audio_of_its_own(monkeypatch):
    """Guard. Muted and idle, or muted while a task runs with nothing
    appended for the caller: no relay-originated `input_audio.append` — the
    keeper fills gaps only while speech is owed (silent idle's saving is
    kept)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_progress_speak_after_s", 600.0)
    monkeypatch.setattr(settings, "voice_live_progress_idle_after_s", 600.0)
    _keeper(monkeypatch, True)
    release = asyncio.Event()
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await release.wait()
        return ANSWER, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def ask():
        box["p"].push(H.user_delta(REQUEST, 0, 900))
        box["p"].push(H.delegation("d1", 950))
        await _wait(lambda: any(
            f.get("phase") == "started" for f in client.of("delegation")
        ), 4.0)

    client = H.FakeClient([
        H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4),
        1.2,            # muted, nothing running, nothing owed
        ask, 1.5,       # muted, a task running, nothing appended for the caller
        {"type": "stop"},
    ])
    provider = RecordingProvider(on_send=on_send)
    try:
        await H.run_relay(client, provider, timeout=10, db_session_id="db-p1-idle")
    finally:
        release.set()
        await H.drain_detached()

    assert any(f.get("phase") == "started" for f in client.of("delegation"))
    assert provider.keeper_times() == []
    assert len(provider.phone_times()) == 4


# ══════════════════════════════════════════════════════════════════════
# The `client_event` frame — the transport of PR-9's mute hint
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_client_event_frame_never_ends_the_call(monkeypatch):
    """Any `client_event` is counted (`live_client_event{event, detail}`) and
    the call goes on.  Its counter used to pass the event's name as `name=`
    into `_vcount(name, user_id, **fields)`: a TypeError raised at the call
    site, inside `client_loop`, which ends the call — for EVERY client_event
    frame, so the first app to send one (PR-9's mute hint) would have hung up
    on the caller at their first mute."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        return "Union Plumbing on Front Street.", "test-model"

    H.patch_relay(monkeypatch, think=think)
    counters = H.capture_counters(monkeypatch)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    async def ask():
        await _wait(lambda: "p" in box)
        box["p"].push(H.user_delta("find a plumber near union station", 0, 900))
        box["p"].push(H.delegation("d1", 950))
        await _wait(lambda: any(
            f.get("phase") == "completed" for f in client.of("delegation")
        ), 4.0)

    events = [
        {"type": "client_event", "name": "mute", "detail": "on"},
        {"type": "client_event", "name": "some_future_event"},
        {"type": "client_event", "name": "mute", "detail": "off"},
    ]
    client = H.FakeClient([H.config(), *events, ask, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-client-event")

    assert any(f.get("phase") == "completed" for f in client.of("delegation")), (
        "the call ended at the first client_event frame"
    )
    counted = [fields for name, fields in counters if name == "live_client_event"]
    assert counted == [
        {"event": "mute", "detail": "on"},
        {"event": "some_future_event", "detail": None},
        {"event": "mute", "detail": "off"},
    ], counted


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", [True, False], ids=["mute_hint_on", "no_hint"])
async def test_a_mute_hint_counts_as_no_input_for_the_result_hold(monkeypatch, hint):
    """Contract: the phone's `client_event` `mute` "on" means there has been
    no input since the result's append, whatever audio still arrives — PR-9's
    app keeps the uplink clocked with zero-filled frames while muted, so the
    audio clock alone would say input flows.  Here the phone streams and the
    provider never takes the append: with the hint the speak deadline is
    re-armed (`no_input`) and the nudge waits; without it the 1.0 s budget
    runs out into the nudge exactly as before.  The hint is counted
    (`live_client_mute`) and mute itself is unchanged: every phone frame is
    forwarded and the relay sends no audio of its own."""

    mute = [{"type": "client_event", "name": "mute", "detail": "on"}] if hint else []
    counters, _recorded, _client, provider, marks, _started = await _run_result_scenario(
        monkeypatch, uplink="flowing", db=f"db-p1-mute-hint-{hint}", keeper=False,
        clocks={"voice_live_speak_no_input_max_s": 2.5},
        rest=[*mute, *([AUDIO, FRAME_MS / 1000.0] * 70)],
        # Input flows, but this provider never takes the append.
        provider_kwargs={"consume_ms": 1e12},
    )
    named = [(name, fields) for name, fields, _at in counters]
    finished = marks["finished"]
    diag = f"\n  timeline (from finish): {_timeline(counters, finished)}"
    nudged = _first(counters, "live_result_nudged", reason="no_ack")
    assert nudged, "the ladder never nudged" + diag
    if hint:
        assert ("live_client_mute", {"state": "on"}) in named, named
        assert ("live_result_rearmed", {"reason": "no_input"}) in named, (
            "the mute hint did not hold the result" + diag
        )
        assert nudged - finished >= 1.3, diag
    else:
        assert not [n for n, _f in named if n in ("live_client_mute", "live_result_rearmed")], (
            "input was flowing and no hint came: nothing to hold" + diag
        )
        assert nudged - finished < 1.3, diag
    assert len(provider.phone_times()) == 4 + 70, "every phone frame is forwarded"
    assert provider.keeper_times() == [], "keeper off: the relay originates no audio"


def test_a_direct_reply_owed_is_thinking_and_keeps_input_moving():
    """D-P1 (2026-10-05 call, 0:20 / 1:04 / 3:38): a closed request whose
    reply had not begun was `listening` — the phone's silent idle then stopped
    the uplink and GPT-Live (D7, measured) answered nothing until the caller
    made a sound.  Owed now: `thinking`, the epoch-close `listening` withheld,
    and the keeper owes `reply` — until output, a delegation, a newer request
    or the bound.  Same four states."""

    from types import SimpleNamespace

    from app.services.live_voice_protocol import (
        DelegationSupervisor, LiveClientAdapter, PendingDelegation, Utterance,
        UtteranceAssembler, _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.adapter = LiveClientAdapter(session_id=H.PSID)
    session.supervisor = DelegationSupervisor()
    session.active_delivery = None
    session.delivery_queue = []
    session.results_owed = set()
    session.relay_spoken_epoch = -1
    session.utterances = UtteranceAssembler(provider_session_id=H.PSID)

    def closed(turn_id, ordinal, start, end, text, *, asks_nothing=False):
        turn = Utterance(
            turn_id=turn_id, ordinal=ordinal, start_ms=start, end_ms=end, text=text,
            closed=True, causal_speech=True, asks_nothing=asks_nothing,
        )
        session.utterances.closed_queue.append(turn)
        session.open_reply_owed(turn)
        return turn

    hello = closed("t0", 1, 0, 400, "are you there", asks_nothing=True)
    assert session.reply_pending_turn() == "" and session.derived_state() == "listening"
    turn = closed("t1", 2, 1000, 2000, "what is the best book for founders")
    assert hello.turn_id != turn.turn_id
    assert session.reply_pending_turn() == "t1"
    assert session.derived_state() == "thinking"
    assert session.listening_withheld()
    assert session.uplink_owed() == "reply"

    # A provider delegation that may still claim it (binding is one await
    # after the close) is work on its way, not a reply owed.
    session.tracker = SimpleNamespace(pending=[PendingDelegation("d1", 2050, 0.0)])
    assert session.reply_pending_turn() == ""
    session.tracker = SimpleNamespace(pending=[])
    assert session.reply_pending_turn() == "t1"

    # Relay speech (a claim took the epoch) is not the reply; a reply is.
    session.note_reply_output(1, claimed=True)
    assert session.reply_pending_turn() == "t1"
    # RO-3: an epoch that answers ANOTHER turn, or the held tail of a sentence
    # the caller spoke over (`crossed_epochs`), is not this request's reply.
    session.note_reply_output(1, claimed=False, parent="t0")
    assert session.reply_pending_turn() == "t1"
    session.crossed_epochs = {1}
    session.note_reply_output(1, claimed=False)
    assert session.reply_pending_turn() == "t1"
    session.crossed_epochs = set()

    # F8 / T5: the reply's epoch is OPEN — speaking, and the keeper still owes
    # input until it retires (GPT-Live's output freezes ~3.5 s after the last
    # input otherwise).
    session.adapter.provider_event(H.out_audio("AQID"), now=1.0)
    assert session.adapter.output_open
    session.note_reply_output(session.adapter.epoch, claimed=False)
    assert session.reply_pending_turn() == "" and session.derived_state() == "speaking"
    assert session.uplink_owed() == "speaking"
    session.adapter.rotate_on_gap(now=60.0, sentence_hold=False)
    assert not session.adapter.output_open
    assert session.derived_state() == "listening"
    assert session.uplink_owed() == ""

    # Bound, and a delegation that consumed it.
    later = closed("t2", 3, 3000, 4000, "and search it for me")
    assert session.reply_pending_turn() == "t2"
    later.consumed_by = "d2"
    assert session.reply_pending_turn() == ""
    later.consumed_by = None
    session.reply_owed["closed_at"] -= live._REPLY_PENDING_MAX_S + 1
    assert session.reply_pending_turn() == ""

    # An accepted reask not answered yet keeps input moving too.
    session.reask_pending = {"t2": 1.0}
    assert session.uplink_owed() == "reask"
    assert live.LIVE_STATES == ("listening", "thinking", "tool_use", "speaking")


# ── F9 / T6: GPT-Live streams all-zero output audio while input moves ─────
#
# Measured 2026-10-05 (Q5): 34/58 output deltas all-zero, the first before any
# speech. Treated as output, one such delta minted an epoch, sent `speaking`
# over silence, forwarded zeros to the phone, ruled a reply-pending turn
# answered (so the keeper stopped and the phone fell to `listening` — the T1
# frozen timeline) and ruled a delivered result SPOKEN with nothing said.

ZERO_DELTA = base64.b64encode(b"\x00" * 4800).decode()
#: 100 ms of a quiet-but-audible tone (RMS ~70 > OUTPUT_SILENCE_RMS 50).
VOICED_DELTA = base64.b64encode(
    b"".join(int(100 if i % 2 else -100).to_bytes(2, "little", signed=True)
             for i in range(2400))
).decode()


def test_the_silence_gate_reads_the_pcm_not_the_payload():
    assert live.output_audio_is_silent(H.out_audio(ZERO_DELTA))
    assert live.output_audio_is_silent(H.out_audio(""))
    # Below RMS 50 is silence (the experiment's own QUIET_RMS) …
    quiet = base64.b64encode((7).to_bytes(2, "little", signed=True) * 2400).decode()
    assert live.output_audio_is_silent(H.out_audio(quiet))
    # … at or above it is not, and neither is anything it cannot decode.
    assert not live.output_audio_is_silent(H.out_audio(VOICED_DELTA))
    assert not live.output_audio_is_silent(H.out_audio("AQID"))
    assert not live.output_audio_is_silent(H.out_audio("A"))


@pytest.mark.asyncio
async def test_an_all_zero_output_delta_is_not_output(monkeypatch, caplog):
    """A zero delta after the caller's final changes NOTHING: no epoch, no
    `speaking`, no audio on the wire, reply-pending and the keeper stand. The
    reply's own audible delta opens the epoch; a zero BETWEEN two audible
    deltas of it is a pause and plays in order under the same id; a zero
    before it never reaches the phone."""

    import logging
    caplog.set_level(logging.INFO)
    H.fast_clocks(monkeypatch, voice_live_reply_regen_after_s=60.0)
    _keeper(monkeypatch, True)
    H.patch_relay(monkeypatch)
    box: dict = {}
    marks: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def ask():
        p = box["p"]
        p.push(H.user_delta("what is the best book for founders", 0, 900))
        await _wait(lambda: any(f.get("final") for f in client.of("transcript")), 3.0)
        await asyncio.sleep(0.8)
        p.push(H.out_audio(ZERO_DELTA))
        await asyncio.sleep(0.4)
        marks["zero_at"] = time.monotonic()
        marks["states"] = list(client.states())
        marks["audio"] = len(client.of("audio_delta"))
        p.push(H.out_audio(ZERO_DELTA))
        await asyncio.sleep(0.4)
        marks["keeper_after_zero"] = [
            t for t in p.keeper_times() if t > marks["zero_at"] + 0.1
        ]
        p.push(H.out_audio(VOICED_DELTA))
        p.push(H.out_audio(ZERO_DELTA))
        p.push(H.out_audio(VOICED_DELTA))
        await asyncio.sleep(0.1)

    client = H.FakeClient([H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4), ask, 0.3, {"type": "stop"}])
    provider = RecordingProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=12)

    assert marks["states"] == ["listening", "thinking"], marks["states"]
    assert marks["audio"] == 0, "a zero delta reached the phone"
    assert marks["keeper_after_zero"], "the keeper stopped on a zero delta"
    deltas = client.of("audio_delta")
    assert [d["data"] for d in deltas] == [VOICED_DELTA, ZERO_DELTA, VOICED_DELTA], (
        "the pause between two audible deltas plays in order; nothing before them"
    )
    assert len({d["response_id"] for d in deltas}) == 1
    opened = [r.getMessage() for r in caplog.records if "output epoch opened" in r.getMessage()]
    assert len(opened) == 1, opened
    states = client.states()
    assert "speaking" in states and states.index("speaking") > states.index("thinking")


@pytest.mark.asyncio
async def test_a_result_followed_only_by_silence_is_never_ruled_spoken(monkeypatch, caplog):
    """RO-2: the result's commentary claim binds to the next MINTED epoch. A
    zero delta must not be one, or the answer is recorded as said and the
    nudge/requeue ladder never runs."""

    import logging
    caplog.set_level(logging.INFO)
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box: dict = {}

    class _SilentAfterAppend(RecordingProvider):
        def _auto_respond(self, event):
            super()._auto_respond(event)
            if event.get("type") == "session.commentary.append":
                self.push(H.out_audio(ZERO_DELTA))

    client = H.FakeClient([H.config(), 2.5, {"type": "stop"}])
    provider = _SilentAfterAppend(on_send=_script(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=10)

    msgs = [r.getMessage() for r in caplog.records]
    assert not any("result spoken id=d1" in m for m in msgs), msgs
    assert any("result nudge sent id=d1" in m for m in msgs)
    terminal = [f for f in client.of("delegation") if f.get("phase") == "completed"]
    assert terminal and terminal[-1].get("spoken") is False, terminal
    assert not client.of("audio_delta")
    assert "speaking" not in client.states()


def test_a_zero_delta_neither_refreshes_the_gap_nor_outlives_a_caller_final():
    """The gap timer is provider SILENCE; zeros are silence. An epoch with
    zeros streaming behind its last word still retires on the gap, and a
    zero after that retirement mints nothing."""

    adapter = live.LiveClientAdapter("zero-gap")
    t0 = 1000.0
    assert adapter.provider_event(H.out_audio(ZERO_DELTA), now=t0) == []
    assert not adapter.output_open
    frames = adapter.provider_event(H.out_audio(VOICED_DELTA), now=t0 + 0.1)
    assert [f["type"] for f in frames] == ["state", "audio_delta"]
    assert adapter.output_open and adapter.epoch == 1
    for k in range(1, 10):
        assert adapter.provider_event(H.out_audio(ZERO_DELTA), now=t0 + 0.1 + k * 0.1) == []
    gap_s = adapter.required_gap_ms(sentence_hold=False) / 1000.0
    _frames, closed = adapter.rotate_on_gap(now=t0 + 0.1 + gap_s + 0.01, sentence_hold=False)
    assert closed is not None and closed.retire_reason == "gap"
    assert closed.audio_chunks == 1, "held silence that no speech followed is not audio"
    assert adapter.provider_event(H.out_audio(ZERO_DELTA), now=t0 + 5) == []
    assert not adapter.output_open and adapter.epoch == 1


@pytest.mark.asyncio
async def test_a_zero_delta_under_the_callers_turn_never_holds_the_reply(monkeypatch):
    """Silence is not held behind the caller's turn like real output. Held, it
    is untimed audio (no transcript will ever time a zero): the release then
    waits a whole output gap for it, holds the reply behind it, and finally
    drops it — a delay the reply pays for silence."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    counters = _capture(monkeypatch)
    box: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def ask():
        p = box["p"]
        p.push(H.user_delta("what is the best book", 0, 500))
        p.push(H.out_audio(ZERO_DELTA))
        p.push(H.user_delta(" for founders", 500, 900))
        p.push(H.out_audio(ZERO_DELTA))
        await _wait(lambda: any(f.get("final") for f in client.of("transcript")), 3.0)
        p.push(H.out_text("The classic one.", 1500, 1900))
        p.push(H.out_audio(VOICED_DELTA))
        await asyncio.sleep(0.4)

    client = H.FakeClient([H.config(), ask, 0.2, {"type": "stop"}])
    await H.run_relay(client, RecordingProvider(on_send=on_send), timeout=10)

    dropped = [f for n, f, _ in counters if n == "live_deferred_output_dropped"]
    assert not dropped, dropped
    assert [d["data"] for d in client.of("audio_delta")] == [VOICED_DELTA]


@pytest.mark.asyncio
async def test_the_keeper_feeds_a_direct_reply_until_its_epoch_retires(monkeypatch):
    """F8 / T5 (measured 2026-10-05, Q3): with no input GPT-Live's reply
    FROZE ~3.5 s after the last input, mid-sentence. The phone is quiet here
    (muted, or silent-idled after the close), so the keeper is the only input:
    it must keep feeding the reply's epoch while it is open — it used to stop
    at the epoch's birth, the moment reply-pending ended — and owe nothing
    once the epoch has retired."""

    H.fast_clocks(monkeypatch, voice_live_reply_regen_after_s=60.0)
    _keeper(monkeypatch, True)
    H.patch_relay(monkeypatch)
    box: dict = {}
    marks: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def ask():
        p = box["p"]
        p.push(H.user_delta("tell me about the best book for founders", 0, 900))
        await _wait(lambda: any(f.get("final") for f in client.of("transcript")), 3.0)
        p.push(H.out_text("The classic one is Zero to One.", 1200, 1600))
        marks["reply_at"] = time.monotonic()
        while time.monotonic() - marks["reply_at"] < 1.8:
            p.push(H.out_audio(VOICED_DELTA))
            await asyncio.sleep(0.05)
        marks["reply_end"] = time.monotonic()
        await asyncio.sleep(1.2)

    client = H.FakeClient([H.config(), *([AUDIO, FRAME_MS / 1000.0] * 4), ask, 0.1, {"type": "stop"}])
    provider = RecordingProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=12)

    kt = provider.keeper_times()
    during = [t for t in kt if marks["reply_at"] + 0.7 < t < marks["reply_end"]]
    after = [t for t in kt if t > marks["reply_end"] + 0.6]
    assert len(during) >= 5, f"keeper starved the open reply: {len(during)} chunks"
    assert not after, f"keeper kept going after the epoch retired: {len(after)}"


@pytest.mark.asyncio
async def test_an_old_replys_held_tail_does_not_answer_the_new_request(monkeypatch, caplog):
    """RO-3: the caller asks a NEW question over a reply. R1 retires the old
    epoch at their final; the rest of the old sentence then mints an epoch
    whose parent is the OLDER turn (`held tail after a caller turn`). That
    tail is not the reply the new request is owed: `thinking` and the keeper
    must outlast it. It used to end reply-pending — `listening`, no keeper,
    no regeneration, the T1 dead air."""

    import logging
    caplog.set_level(logging.INFO)
    H.fast_clocks(monkeypatch, voice_live_reply_regen_after_s=60.0,
                  voice_live_output_epoch_gap_ms=400)
    _keeper(monkeypatch, True)
    H.patch_relay(monkeypatch)
    box: dict = {}
    marks: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def scene():
        p = box["p"]
        p.push(H.user_delta("tell me about founders", 0, 600))
        await _wait(lambda: any(f.get("final") for f in client.of("transcript")), 3.0)
        p.push(H.out_text("Founders often read many books and", 800, 2000))
        p.push(H.out_audio("AQID"))
        await asyncio.sleep(0.05)
        p.push(H.user_delta("what is the best book for investors", 1800, 2600))
        await asyncio.sleep(0.03)
        p.push(H.out_text(" the classic one is", 2000, 2700))
        p.push(H.out_audio("AQID"))
        await _wait(lambda: sum(1 for f in client.of("transcript") if f.get("final")) >= 2, 3.0)
        marks["closed2"] = time.monotonic()
        await asyncio.sleep(1.6)
        marks["states"] = list(client.states())

    client = H.FakeClient([H.config(), *([AUDIO, FRAME_MS / 1000.0] * 2), scene, 0.1, {"type": "stop"}])
    provider = RecordingProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=12)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("held tail after a caller turn epoch=2" in m for m in msgs), msgs
    assert any("output epoch closed epoch=2 reason=gap" in m for m in msgs), msgs
    assert marks["states"][-1] == "thinking", marks["states"]
    owed_after_tail = [t for t in provider.keeper_times() if t > marks["closed2"] + 1.0]
    assert owed_after_tail, "the tail ended reply-pending: no keeper for the new request"


# ── voice round 2026-10-06 (D-R1): the reply ladder while the phone streams ──


@pytest.mark.asyncio
async def test_a_reply_owed_is_thinking_through_reask_backstop_and_task_with_no_keeper(
    monkeypatch, caplog,
):
    """The 08:38 «فورمولوان» shape: the phone streamed real mic audio the
    whole time (input moving), the relay's accepted note produced nothing.
    The ladder carries it — reask, then the relay's own `relay-reply:` task —
    and the state is `thinking` from the close to the task's end (never the
    `listening` that lets the phone's silent idle close the uplink), while
    the keeper sends NOTHING: GPT-Live has one input timeline, and silence
    spliced into the caller's mic audio is exactly what it must never get."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.fast_clocks(
        monkeypatch, voice_live_uplink_keeper_enabled=True,
        voice_live_reply_regen_after_s=0.3, voice_live_reply_backstop_enabled=True,
        voice_live_delegation_ttl_s=2.0,
    )
    started = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        started.set()
        await asyncio.sleep(0.6)
        return "Verstappen won the Formula One race.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta("who won the formula one race", 0, 900))
        elif e["type"] == "session.close":
            p.push(H.closed())

    mic = []
    for _ in range(40):                      # ~4 s of real phone audio
        mic += [AUDIO, 0.1]
    client = H.FakeClient([H.config(), *mic, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-ladder-streaming")

    rids = [
        f["delegation_id"] for f in client.of("delegation")
        if str(f.get("delegation_id") or "").startswith(live.RELAY_REPLY_PREFIX)
    ]
    assert rids, client.of("delegation")
    rid = rids[0]
    frames = client.frames
    closed_at = next(
        i for i, f in enumerate(frames) if f.get("type") == "transcript" and f.get("final")
    )
    done_at = next(
        i for i, f in enumerate(frames)
        if f.get("type") == "delegation" and f.get("delegation_id") == rid
        and f.get("phase") == "completed"
    )
    between = [f["state"] for f in frames[closed_at:done_at] if f.get("type") == "state"]
    assert between and set(between) <= {"thinking", "tool_use"}, between
    reasks = [
        e for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-reask-")
    ]
    assert len(reasks) == 1, reasks
    messages = [r.getMessage() for r in caplog.records]
    assert not any("[LIVE] uplink keeper on" in m for m in messages), messages
    sent = provider.of("session.input_audio.append")
    assert sent, "the phone's mic audio reached the provider"
    # The phone's frames are 50 ms; the keeper's chunk is 100 ms — none sent.
    assert not [e for e in sent if e.get("audio") == live._UPLINK_KEEPER_SILENCE_B64]
    assert started.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("newer", [True, False])
async def test_a_delegated_turn_is_never_reported_unanswered_by_the_expiry_or_a_newer_request(
    monkeypatch, caplog, newer,
):
    """Phase D (correctness-relay-5): request A is DELEGATED (the model's own
    `d1`), so its claimed epochs never mark the owed reply answered; the 60 s
    expiry then logged `reply pending expired` and sent the phone
    `reply_status{unanswered, max_wait}` for a turn its agent was handling,
    and a newer request B reported A `unanswered / superseded`.  A bound turn
    ends silently; B — never bound — still expires as before."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.fast_clocks(monkeypatch)
    monkeypatch.setattr(live, "_REPLY_PENDING_MAX_S", 1.0)

    async def think(*_a, **_k):
        await asyncio.sleep(0.2)
        return "Zero to One.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        if e["type"] == "session.start":
            p.push(H.user_delta("what is the best book for a founder?", 0, 900))
            p.push(H.delegation("d1", 950))
            if newer:
                p.push(H.user_delta("what is the capital of peru?", 1600, 2400))
        elif e["type"] == "session.close":
            p.push(H.closed())

    features = H.config()["features"] + ["reply_status"]
    client = H.FakeClient([H.config(features=features), 2.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=f"db-reply-bound-{newer}")

    finals = [f for f in client.of("transcript") if f.get("final")]
    a = finals[0]["turn_id"]
    b = finals[1]["turn_id"] if newer else ""
    messages = [r.getMessage() for r in caplog.records]
    expired = [m for m in messages if m.startswith("[LIVE] reply pending expired")]
    assert not [m for m in expired if f"turn_id={a} " in m], expired
    assert not [f for f in client.of("reply_status") if f["user_turn_id"] == a], client.of("reply_status")
    assert not [
        m for m in messages
        if m.startswith("[LIVE] turn timeline") and f"turn_id={a} " in m and "unanswered" in m
    ], messages
    if newer:
        # The unbound newer request still expires and says so.
        assert [(f["outcome"], f.get("reason")) for f in client.of("reply_status")
                if f["user_turn_id"] == b] == [("unanswered", "max_wait")]
    else:
        assert client.of("reply_status") == []


@pytest.mark.asyncio
async def test_an_owed_reply_that_expires_says_so_and_ends_the_keeper_unanswered(
    monkeypatch, caplog,
):
    """`_REPLY_PENDING_MAX_S` used to end an owed reply in SILENCE — state
    `listening`, the keeper `settled`, no line.  Now the expiry logs
    `reply pending expired`, the turn's timeline line reads
    `unanswered_timeout`, the keeper's episode ends `reason=unanswered`, and a
    `reply_status` client is told `unanswered` / `max_wait`."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.fast_clocks(monkeypatch, voice_live_uplink_keeper_enabled=True)
    monkeypatch.setattr(live, "_REPLY_PENDING_MAX_S", 1.0)
    H.patch_relay(monkeypatch)

    def on_send(p, e):
        if e["type"] == "session.start":
            p.push(H.user_delta("what is the capital of australia", 0, 900))
        elif e["type"] == "session.close":
            p.push(H.closed())

    features = H.config()["features"] + ["reply_status"]
    client = H.FakeClient([H.config(features=features), 2.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-reply-expired")

    turn = next(f for f in client.of("transcript") if f.get("final"))["turn_id"]
    messages = [r.getMessage() for r in caplog.records]
    expired = [m for m in messages if m.startswith("[LIVE] reply pending expired")]
    assert len(expired) == 1 and f"turn_id={turn} " in expired[0], messages
    assert any(
        m.startswith("[LIVE] turn timeline") and f"turn_id={turn} " in m
        and "reason=unanswered_timeout" in m
        for m in messages
    ), messages
    assert any("[LIVE] uplink keeper off reason=unanswered owed=reply" in m for m in messages), [
        m for m in messages if "uplink keeper" in m
    ]
    status = client.of("reply_status")
    assert [(f["outcome"], f.get("reason")) for f in status] == [("unanswered", "max_wait")], status
    assert status[0]["user_turn_id"] == turn and status[0]["since_ms"] >= 1000
