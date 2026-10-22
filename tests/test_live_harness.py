"""Fakes for driving the GPT-Live relay offline.

Imported by the other `test_live_*` modules the way the ws-proxy suite imports
its own harness.  It carries one self-test so the file is never silently broken
by a change nobody runs.
"""

import asyncio
import inspect
import json
import time

import pytest

from app.config import settings


PSID = "live-psid"


class FakeClient:
    """The phone.  A script of outbound frames in, every relay frame out."""

    def __init__(self, script=None):
        self.frames: list[dict] = []
        self._script = list(script or [])
        self.reads = 0
        #: frame type → monotonic of the FIRST one of that type.  Startup
        #: latency is a property of WHEN `ready` arrives, and a test that can
        #: only count frames cannot measure it.
        self.stamps: dict[str, float] = {}

    async def send_json(self, frame):
        self.stamps.setdefault(str(frame.get("type") or ""), time.monotonic())
        self.frames.append(frame)

    async def receive_text(self):
        while self._script:
            item = self._script.pop(0)
            self.reads += 1
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
                continue
            if callable(item):
                result = item()
                if inspect.isawaitable(result):
                    await result
                continue
            return json.dumps(item)
        # The relay owns teardown from here (a `stop` frame or the provider).
        await asyncio.sleep(3600)

    # ── readers ────────────────────────────────────────────────────
    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f.get("type") == kind]

    def states(self) -> list[str]:
        return [f["state"] for f in self.of("state")]

    def phases(self) -> list[str]:
        return [f["phase"] for f in self.of("delegation")]


#: The three append kinds and the ack event each one is answered with.  A real
#: GPT-Live acks every accepted append by echoing our `client_event_id`; the
#: fake does it too, because "nothing came back" and "it was refused" have
#: opposite remedies in the relay and a fake that never acks can only exercise
#: one of them.
APPEND_ACKS = {
    "session.commentary.append": "session.commentary.appended",
    "session.instructions.append": "session.instructions.appended",
    "session.thinking.append": "session.thinking.appended",
}


def spoken_for(event: dict) -> str:
    """What the fake model says aloud for an append it was told to voice.

    A commentary is said as its `content`.  An instruction is said as its
    SLOT: the relay's speakable instructions (`result_nudge`,
    `relay_task_say`, `status_result`, the media lines) end in
    "…: {result}", so the text after the first ": " of the lead is what the
    caller hears.  The FIRST separator, not the last, so a result that itself
    contains ": " is said whole.  An instruction with no slot is said whole.
    """

    content = str(event.get("content") or "").strip()
    if str(event.get("type") or "") == "session.instructions.append":
        head, sep, tail = content.partition(": ")
        if sep and tail.strip():
            return tail.strip()
    return content or "Okay."


class FakeProvider:
    """GPT-Live.  `on_send(provider, event)` scripts the responses.

    `auto_ack` makes it behave like the documented provider: every append is
    acked with its own `client_event_id`.  `reject_appends` turns a named set of
    append kinds into an `error` carrying that id instead — the shape the relay
    has to tell apart from a session-level fault.  `speak_on_commentary` emits
    an output epoch after an accepted commentary append, which is the only
    evidence that the model actually said it.
    """

    def __init__(
        self,
        on_send=None,
        session_id: str = PSID,
        *,
        auto_ack: bool = False,
        reject_appends=(),
        reject_first: int = 0,
        speak_on_commentary: bool = False,
        speak_on_instructions: bool = False,
        start_delay: float = 0.0,
    ):
        #: How long GPT-Live takes to answer `session.start` with
        #: `session.started`.  Zero by default; a startup-latency test needs a
        #: handshake with real duration, or "overlapped" and "serialized" are
        #: the same measurement.
        self._start_delay = float(start_delay)
        self.sent: list[dict] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self._on_send = on_send
        self._session_id = session_id
        self._auto_ack = auto_ack
        self._reject = set(reject_appends or ())
        self._reject_first = int(reject_first)
        self._rejected = 0
        self._speak_after = set()
        if speak_on_commentary:
            self._speak_after.add("session.commentary.append")
        if speak_on_instructions:
            # The nudge path: an instruction can make the model start speaking
            # too, and a fake that only ever answers commentary would make the
            # recovery in R12.2 untestable.
            self._speak_after.add("session.instructions.append")
        self.acked: list[str] = []
        self.refused: list[str] = []

    async def recv(self):
        if self._start_delay:
            await asyncio.sleep(self._start_delay)
        return json.dumps({"type": "session.started", "session": {"id": self._session_id}})

    def _auto_respond(self, event):
        kind = str(event.get("type") or "")
        ack = APPEND_ACKS.get(kind)
        if ack is None:
            return
        event_id = str(event.get("event_id") or "")
        refuse = kind in self._reject and (
            self._reject_first <= 0 or self._rejected < self._reject_first
        )
        if refuse:
            self._rejected += 1
            self.refused.append(event_id)
            self.push({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "append_rejected",
                    "client_event_id": event_id,
                },
            })
            return
        self.acked.append(event_id)
        self.push({"type": ack, "client_event_id": event_id, "start_ms": 0, "end_ms": 0})
        if kind in self._speak_after:
            # The model paraphrases it aloud: transcript + audio, i.e. a new
            # output epoch.  Voice round 2026-10-07 (D-R1): the relay proves a
            # delivery by the epoch's WORDS, so the fake says what it was
            # given — a commentary's `content`, an instruction's slot (the
            # text after its "…: " lead, `spoken_for`) — exactly the model the
            # docstring above describes.  The old fixed "Here is what I
            # found." proved nothing and would record every answer unspoken.
            self.push(out_text(spoken_for(event), 900, 1500))
            self.push(out_audio("SPEAKING"))

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if self._auto_ack:
            self._auto_respond(event)
        if self._on_send is not None:
            result = self._on_send(self, event)
            if inspect.isawaitable(result):
                await result

    def push(self, event):
        self.queue.put_nowait(event)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return json.dumps(await self.queue.get())

    # ── readers ────────────────────────────────────────────────────
    def of(self, kind: str) -> list[dict]:
        return [e for e in self.sent if e.get("type") == kind]


def user_delta(text, start_ms, end_ms):
    return {
        "type": "session.input_transcript.delta",
        "delta": text, "start_ms": start_ms, "end_ms": end_ms,
    }


def delegation(delegation_id, offset_ms):
    return {
        "type": "session.delegation.created",
        "offset_ms": offset_ms,
        "delegation": {"id": delegation_id, "target": "client"},
    }


def out_text(text, start_ms=0, end_ms=0):
    return {
        "type": "session.output_transcript.delta",
        "delta": text, "start_ms": start_ms, "end_ms": end_ms,
    }


def out_audio(data="AQID"):
    return {"type": "session.output_audio.delta", "delta": data}


def closed(reason="close_requested", seconds=0):
    return {"type": "session.closed", "reason": reason, "usage": {"seconds": seconds}}


def config(**extra):
    frame = {"type": "config", "protocol": "live1", "features": [
        "live_turns", "state_seq", "outcomes", "delegation_frames",
        "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    ]}
    frame.update(extra)
    return frame


def legacy_config(**extra):
    """Build 129 / web: no `features` list at all."""

    frame = {"type": "config", "protocol": "live1"}
    frame.update(extra)
    return frame


def capture_counters(monkeypatch) -> list[tuple]:
    """Record every fleet counter the relay fires, in order.

    `_vcount` is the only evidence some of R12's outcomes leave behind — a
    result that was never spoken is silent by construction — so a test that
    names a counter has to be able to read it.
    """

    from app.api import ws_realtime as rt

    seen: list[tuple] = []

    def _count(name, _user_id, **fields):
        seen.append((name, fields))

    monkeypatch.setattr(rt, "_vcount", _count)
    return seen


def fast_clocks(monkeypatch, **overrides):
    """Shrink every Live clock so a test is milliseconds, not seconds."""

    defaults = {
        "voice_live_transcript_settle_ms": 10,
        "voice_live_utterance_gap_ms": 60,
        "voice_live_utterance_hard_gap_ms": 120,
        "voice_live_persist_min_interval_ms": 0,
        "voice_live_delegation_ttl_s": 0.4,
        "voice_live_output_epoch_gap_ms": 80,
        "voice_live_interrupt_suppress_ms": 120,
        "voice_live_progress_min_interval_s": 0.0,
        # D-P3: the research backstop is ON in production; the suites written
        # before it script their own delegations and turn it on explicitly
        # (tests/test_live_research_backstop.py).
        "voice_live_research_backstop_enabled": False,
        # D-R3 / D-R1(e) / D-R4 (2026-10-06 call): the PLAY backstop and the
        # promise mode are ON in production; a suite whose fake model leaves a
        # play-shaped turn unanswered would otherwise grow a `relay-play:`
        # delegation.  Their tests opt in (tests/test_live_media_backstop.py).
        # The bare-command grace is pinned at its production value.
        "voice_live_play_backstop_enabled": False,
        "voice_live_claim_grace_ms": 0,
        "voice_live_media_backstop_playing_grace_ms": 400,
        "voice_live_persist_drain_s": 2.0,
        "voice_live_history_rows": 0,
        "voice_live_speak_timeout_s": 0.5,
        "voice_live_result_defer_max_ms": 200,
        # R7 (DESIGN-R2 D-R7 (6)): results go out in finish order in the
        # suites written before the ask-order wait; its own tests opt in.
        "voice_live_result_order_wait_s": 0.0,
        # P1 (b): the no-input hold's bound, scaled like the speak budget it
        # extends — twice the 0.5 s above, so a fake that never acks (and no
        # fake phone here streams audio) gets its nudge and verdict exactly
        # when it always did: the bound is spent by the first deadline.
        "voice_live_speak_no_input_max_s": 1.0,
        # Production runs the keeper (D7); no fake phone here streams audio, so
        # with it on every suite would hear relay silence. Keeper tests opt in.
        "voice_live_uplink_keeper_enabled": False,
        # Voice round 2026-10-06 (D-R1): the reply ladder is ON in production
        # (regen 3.0 s, then the `relay-reply:` backstop).  The suites written
        # before it script replies that may take longer than 3 s to arrive (a
        # fake model that never answers a turn is a scripted fact, not a
        # stuck model), so the relay's own reasks and reply backstop are OFF
        # here; the ladder's tests opt in.
        "voice_live_reply_regen_after_s": 0,
        "voice_live_reply_backstop_enabled": False,
        # Voice round 2026-10-07 (D-R1, relay-delivery §4.10): the two-result
        # merge, the continuation cue and the media fold window are
        # relay-initiated behaviours, ON in production; the suites written
        # before them expect one result per append and their own rows, so
        # they are OFF here and their tests opt in.  The words proof itself
        # cannot be pinned off — it IS the verdict — which is why the fake
        # model above says what it is given.  The verdict and late-words
        # bounds are scaled like the speak budget.
        "voice_live_result_merge_window_s": 0.0,
        "voice_live_result_continue_cue_enabled": False,
        "voice_live_result_verdict_max_s": 3.0,
        "voice_live_result_late_words_s": 2.0,
        "voice_live_media_fold_window_s": 0.0,
        # Voice round 2026-10-07, lane RB: every new relay-initiated turn
        # behaviour is OFF here and its own tests opt in — the endpoint rules
        # (hesitant / verbless spans keep the hard gap), the addendum join and
        # the seam absorber (one card for a question tail), the fragment carry
        # (a cut piece of the agent's sentence is never its own row) and the
        # provider-start retry.  `voice_live_session_start_attempt_s` keeps its
        # production value: with the retry off it bounds nothing.
        "voice_live_endpoint_rules_enabled": False,
        "voice_live_addendum_window_ms": 0,
        "voice_live_seam_onset_ms": 0,
        "voice_live_fragment_carry_s": 0,
        "voice_live_provider_start_retry_enabled": False,
        # Voice round 2026-10-07 (D-R4, relay-search-play): the relay's own
        # now-playing answer (Hooks A/B) is ON in production; a suite whose
        # fake phone reports a track and whose caller asks about it would
        # otherwise have the relay answer before the scripted model does.  The
        # near-finish progress rule is 0 (off) here: the suites written before
        # it run tasks for well under a second, and its tests opt in.
        "voice_live_media_status_enabled": False,
        "voice_live_progress_min_remaining_s": 0.0,
        "auto_extract_memories": False,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        monkeypatch.setattr(settings, key, value)


def patch_relay(
    monkeypatch, *, think=None, save=None, curate=None, play=None,
    control=None, vps=None, summarize=None,
):
    """Stub every seam that leaves the process.  Returns the recorded calls.

    `summarize` is the task summarizer's one model call
    (`live.summarize_task_text`, PR-14).  By default it answers nothing, so a
    test that does not script it sees no summary at all — exactly the frames
    and rows of a relay whose summarizer timed out.
    """

    from app.api import ws_realtime as rt
    import app.services.live_voice_protocol as live

    saves: list[dict] = []
    curates: list[tuple] = []

    async def default_think(*_args, **_kwargs):
        return "An answer.", "test-model"

    async def default_save(user_id, session_id, **kwargs):
        saves.append({"user_id": user_id, "session_id": session_id, **kwargs})
        return 0

    async def default_curate(user_id, user_text, assistant_text):
        curates.append((user_id, user_text, assistant_text))

    async def default_play(_user_id, _query, _variety=False):
        return "ERROR: no media in this test.", None

    async def default_control(_user_id, action):
        return {"ok": False, "action": action, "reason": "no_active_session"}

    async def default_vps(_user_id):
        return None

    async def noop_tz(_user_id, _tz):
        return None

    async def noop_meter(*_args, **_kwargs):
        return None

    async def no_summary(**_kwargs):
        return None

    monkeypatch.setattr(rt, "_think", think or default_think)
    monkeypatch.setattr(rt, "_save_voice_messages", save or default_save)
    monkeypatch.setattr(rt, "_curate_voice_turn", curate or default_curate)
    monkeypatch.setattr(rt, "_play_media_direct", play or default_play)
    monkeypatch.setattr(rt, "_control_media_direct", control or default_control)
    monkeypatch.setattr(rt, "_get_vps_info", vps or default_vps)
    monkeypatch.setattr(rt, "_apply_client_tz", noop_tz)
    monkeypatch.setattr(live, "meter_live_session", noop_meter)
    monkeypatch.setattr(live, "summarize_task_text", summarize or no_summary)
    return {"saves": saves, "curates": curates}


async def drain_detached(timeout: float = 2.0) -> None:
    """Join the relay's own detached tasks before the loop goes away.

    `_DETACHED` holds the shielded delegation runners, the late persistence
    writes and `_finish_detached`.  They outlive `run_live_voice_session` BY
    DESIGN (D4), so a test that simply returned left them pending and the loop
    tore them down with "Task was destroyed but it is pending!" — noise that
    reads like a leak in the code under test.  This is a join, not a sleep: it
    returns as soon as they finish, and cancels only what is still running
    after the bound.
    """

    from app.services.live_voice_protocol import _DETACHED

    tasks = [t for t in list(_DETACHED) if not t.done()]
    if not tasks:
        return
    # A JOIN, and nothing is cancelled. Two things make cancelling wrong here
    # rather than merely blunt: outliving the socket is the whole of D4, so a
    # test that holds a runner open past the bound is asserting exactly the
    # behaviour a cancel would fabricate away; and `live-detached-finisher`
    # awaits those runners through an `asyncio.gather`, which propagates its own
    # cancellation INTO them — so cancelling the bookkeeping task silently kills
    # the work it exists to wait for (measured: the detached play's card stopped
    # being persisted). What is left pending after the bound is a task some test
    # deliberately froze, and that is the test's statement, not a leak.
    await asyncio.wait(tasks, timeout=timeout)


async def run_relay(client, provider, *, timeout=5.0, drain=True, **kwargs):
    from app.services.live_voice_protocol import run_live_voice_session

    params = {
        "websocket": client,
        "provider_ws": provider,
        "user_id": "user-1",
        "using_platform_key": True,
        "db_session_id": "db-session",
        "instructions": "Listen and help.",
    }
    params.update(kwargs)
    try:
        await asyncio.wait_for(run_live_voice_session(**params), timeout=timeout)
    finally:
        if drain:
            await drain_detached()


@pytest.mark.asyncio
async def test_harness_drives_a_minimal_session(monkeypatch):
    fast_clocks(monkeypatch)
    patch_relay(monkeypatch)
    client = FakeClient([config(), 0.02, {"type": "stop"}])
    provider = FakeProvider(
        on_send=lambda p, e: p.push(closed()) if e["type"] == "session.close" else None,
    )
    await run_relay(client, provider)
    assert client.of("ready")
    assert provider.of("session.start")
