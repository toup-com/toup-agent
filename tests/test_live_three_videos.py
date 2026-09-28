"""The three recorded calls of 2026-09-22 (V1, V2, V3), replayed through the REAL relay.

One provider session covered all three recordings (RUNTIME_EVIDENCE: psid
`live_u1_ER7K…`, DB session `ca90792a`, a TF132-shaped app negotiating
`delegation_frames,heard_text,live_turns,media_control,playback_frames,
task_lifecycle,turn_timing`).  Each test below is one continuous call driven
through `run_live_voice_session` by the fake GPT-Live harness
(`test_live_harness`), with the event ORDER and COUNTS taken from
`relay-timeline-0247Z-0303Z.txt` and the provider input deltas taken from the
visible transcript fragments in the three video evidence reports
(`video-{1,2,3}-evidence.en.md`).  The recordings carry no usable speech audio,
so where a report shows only fragments the connecting words are a
reconstruction; every reconstruction keeps the fragments the reports quote
(«ر» + «وبات», «یوآفت» + «ی», «نیست» + «ند», «نمی‌ت» + «ونی», «خب چی» / «شد»,
'What' … 'LLM').

The production pathologies are reproduced, not stubbed out:

* the provider's INPUT timeline lags the relay's wire clock, in plateaus: it
  only advances while the caller is audible (the app's silent idle), so a wall
  silence of seconds can be a few hundred ms of input timeline (V1+04:14 lag
  jumped to −26…−28 s);
* the assistant OUTPUT transcript runs ahead of wall time (every spoken chunk
  advances ~450 ms of provider position in 50 ms of wall time);
* Persian words split below the word («ر»|«وبات»|«یک», «نمی‌ت»|«ونی»,
  «یوآفت»|«ی», «نیست»|«ند»), hesitations shorter than the gap, and a comma
  tail that legitimately waits for the hard gap;
* the model deciding twice for one utterance (a delegation mid-sentence and
  another at its end), the caller barging in over long spoken results
  (client `interrupt` with the recorded played_ms), the agent's own voice
  echoing back through the microphone, and the app's re-ask frames — both the
  TF132 identity-free `inject_text` and the v0.3 `reask_of_user_turn_id`.

Clocks are scaled (settle 40 / gap 600 / hard gap 1300 ms against production
350 / 1200 / 2600) so each call takes seconds, while every relationship the
recordings depend on (pause < gap < real silence, lag > gap) is kept.

The assertions are on what leaves the relay: frames to the phone, calls to the
fake tenant (`_think`, `_play_media_direct`, `_control_media_direct`) and the
appends to the provider.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Optional

import pytest

import test_live_harness as H
from app.services.live_voice_protocol import relay_line, request_title


#: What the recorded app negotiated (relay log, V1+00:08.5).
TF132 = [
    "delegation_frames", "heard_text", "live_turns", "media_control",
    "playback_frames", "task_lifecycle", "turn_timing",
]
#: A v0.3 build: the same, plus the three R2 features.
V03 = TF132 + ["media_transport", "reask_turns", "language_pref"]

CLOCKS = dict(
    voice_live_transcript_settle_ms=40,
    voice_live_utterance_gap_ms=600,
    voice_live_utterance_hard_gap_ms=1300,
    voice_live_output_epoch_gap_ms=300,
    voice_live_delegation_ttl_s=2.5,
    voice_live_interrupt_suppress_ms=400,
    voice_live_result_defer_max_ms=1500,
    voice_live_speak_timeout_s=1.5,
    voice_live_bargein_min_ms=200,
    # §E is covered by test_live_followups; kept out of these replays so a
    # progress line never races the status lines they assert on.
    voice_live_progress_speak_after_s=600.0,
)

#: The identity-free sentence the shipped relay sent for EVERY inject_text.
IDENTITY_FREE = ("has not been answered", "repeated in your notes", "Answer it now")

STEP = 0.1          # wall seconds between two provider deltas of one utterance
SPAN = 80           # input-timeline ms each delta covers (the rest is inter-word gap)
LEAD = 60           # input-timeline ms between a new utterance and whatever preceded it


# ══════════════════════════════════════════════════════════════════════
# The call: phone, provider (input + output timelines) and tenant fakes
# ══════════════════════════════════════════════════════════════════════

class Phone(H.FakeClient):
    """The app.  Frames go out through an inbox the scenario feeds (so the
    relay reads them while the scenario keeps driving the provider); every
    relay frame comes back with its arrival instant."""

    def __init__(self, config: dict):
        super().__init__([])
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.inbox.put_nowait(config)
        self.times: list[float] = []

    async def send_json(self, frame):
        self.times.append(time.monotonic())
        await super().send_json(frame)

    async def receive_text(self):
        return json.dumps(await self.inbox.get())

    def send(self, frame: dict) -> None:
        self.inbox.put_nowait(frame)

    # ── readers ────────────────────────────────────────────────────
    def finals(self) -> list[dict]:
        return [f for f in self.of("transcript") if f.get("final")]

    def final_texts(self) -> list[str]:
        return [f["text"] for f in self.finals()]

    def turn_of(self, text: str) -> str:
        matches = [f["turn_id"] for f in self.finals() if f["text"] == text]
        assert matches, f"no final user turn reads {text!r}: {self.final_texts()}"
        return matches[-1]

    def frames_for(self, did: str) -> list[dict]:
        return [f for f in self.of("delegation") if f.get("delegation_id") == did]

    def phases(self, did: str) -> list[str]:
        return [f.get("phase") for f in self.frames_for(did)]

    def created(self) -> list[dict]:
        return [f for f in self.of("delegation") if f.get("phase") == "created"]

    def created_for(self, did: str) -> dict:
        frames = [f for f in self.frames_for(did) if f.get("phase") == "created"]
        assert len(frames) == 1, (did, self.phases(did))
        return frames[0]

    def terminal(self, did: str, phase: Optional[str] = None) -> bool:
        wanted = {phase} if phase else {
            "completed", "failed", "cancelled", "superseded", "expired",
        }
        return any(f.get("phase") in wanted for f in self.frames_for(did))

    def parents_after(self, mark: int, needle: str = "") -> list[str]:
        """Causal parent of every response_text frame after `mark` (a frame
        index) whose text carries `needle`."""

        return [
            f.get("parent_user_turn_id") or ""
            for f in self.frames[mark:]
            if f.get("type") == "response_text" and needle in (f.get("text") or "")
        ]

    def stop_actions(self) -> list[dict]:
        return [
            f for f in self.frames
            if f.get("action") in {"stop", "pause"}
            or (f.get("type") == "media_control" and f.get("action") not in {"next", "previous"})
        ]


class Tenant:
    """The Toup tenant behind the relay: the delegated agent turn (`_think`),
    the direct play, and the internal media control."""

    def __init__(self):
        self.calls: list[dict] = []
        self.plans: list[tuple[str, dict]] = []
        self.plays: list[str] = []
        self.play_result: tuple[str, Optional[dict]] = ("ERROR: no media", None)
        self.controls: list[str] = []
        self.control_results: list[dict] = []
        self.control_hook: Optional[Callable[[str], Awaitable[None]]] = None
        self.control_returned_at: list[int] = []   # len(provider.sent) at return
        self.provider: Optional[H.FakeProvider] = None

    def plan(self, marker: str, answer: str, *, delay: float = 0.3,
             tool: Optional[str] = "web_search", gate: Optional[asyncio.Event] = None) -> None:
        """Answer the first agent turn whose DISPLAY request contains `marker`."""

        self.plans.append((marker, {"answer": answer, "delay": delay, "tool": tool, "gate": gate}))

    def call(self, did: str) -> dict:
        found = [c for c in self.calls if c["did"] == did]
        assert len(found) == 1, (did, [c["did"] for c in self.calls])
        return found[0]

    async def think(self, _user_id, task, _session_id, relay=None, out=None, **kw):
        display = str(kw.get("display_request") or "")
        record = {
            "did": kw.get("delegation_id"), "task": task, "display": display,
            "lang": kw.get("reply_language"), "cancelled": False, "done": False,
        }
        self.calls.append(record)
        plan = next((p for marker, p in self.plans if marker in display), None)
        assert plan is not None, f"no tenant plan for {display!r}"
        call_id = f"srch-{record['did']}"
        try:
            if plan["tool"] and relay is not None:
                await relay.on_event({
                    "type": "tool.start", "call_id": call_id, "name": plan["tool"],
                    "args": {"query": display[:60]},
                })
                record["tool_started"] = True
            if plan["gate"] is not None:
                await asyncio.wait_for(plan["gate"].wait(), timeout=15)
            await asyncio.sleep(plan["delay"])
            if plan["tool"] and relay is not None:
                await relay.on_event({
                    "type": "tool.end", "call_id": call_id, "name": plan["tool"],
                    "ok": True, "preview": "found", "elapsed_ms": 40,
                })
        except asyncio.CancelledError:
            record["cancelled"] = True
            raise
        record["done"] = True
        return plan["answer"], "test-model"

    async def play(self, _user_id, query, _variety=False):
        self.plays.append(query)
        return self.play_result

    async def control(self, _user_id, action):
        self.controls.append(action)
        if self.control_hook is not None:
            await self.control_hook(action)
        result = dict(self.control_results.pop(0)) if self.control_results else {
            "ok": False, "reason": "no_active_session",
        }
        result.setdefault("action", action)
        self.control_returned_at.append(len(self.provider.sent) if self.provider else 0)
        return result


class Call:
    """One relay session: the phone, GPT-Live (input and output timelines) and
    the tenant, with helpers that speak in provider-shaped events."""

    def __init__(self, monkeypatch, *, features, name: str, clocks=None, language_pin=None):
        H.fast_clocks(monkeypatch, **dict(CLOCKS, **(clocks or {})))
        self.tenant = Tenant()
        H.patch_relay(
            monkeypatch, think=self.tenant.think, play=self.tenant.play,
            control=self.tenant.control,
        )
        #: Every fleet counter the relay fired, in order (name, fields).
        self.counters = H.capture_counters(monkeypatch)
        self.phone = Phone(H.config(features=list(features)))
        self.provider = H.FakeProvider(on_send=self._on_send, auto_ack=True)
        self.tenant.provider = self.provider
        self.db = f"db-three-videos-{name}-{time.monotonic_ns()}"
        self.language_pin = language_pin
        self.started = asyncio.Event()
        #: The provider's INPUT timeline (ms): advances only while the caller is
        #: audible or by an explicit provider-visible silence.
        self.inp = 0
        #: The provider's OUTPUT timeline (ms): runs ~9x faster than wall time.
        self.out = 0
        #: Delegation ids whose commentary the model does NOT voice.
        self.muted: set[str] = set()
        self._speech: list[str] = []
        self._speech_wake = asyncio.Event()
        self._cut = False
        self.speaking = False
        self._speaker: Optional[asyncio.Task] = None
        self.error: Optional[BaseException] = None
        #: Arrival instant of every provider append, aligned with provider.sent.
        self.sent_times: list[float] = []

    # ── the provider's side ────────────────────────────────────────
    def _on_send(self, provider, event):
        self.sent_times.append(time.monotonic())
        kind = event.get("type")
        if kind == "session.start":
            self.started.set()
            if self._speaker is None:
                self._speaker = asyncio.get_running_loop().create_task(self._speak_loop())
        elif kind == "session.close":
            provider.push(H.closed())
        elif kind == "session.commentary.append":
            # The model paraphrases what the relay hands it.
            if event.get("delegation_id") not in self.muted:
                self.voice(str(event.get("content") or ""))
        elif kind == "session.instructions.append":
            event_id = str(event.get("event_id") or "")
            content = str(event.get("content") or "")
            if event_id.startswith("toup-live-interrupt-"):
                self.hush()
            elif event_id.startswith(("toup-live-status-", "toup-live-nudge-", "toup-live-reask-")):
                # Instructions that ask the model to say something now.
                self.voice(content.split(": ", 1)[-1])

    def voice(self, text: str) -> None:
        if text.strip():
            self._speech.append(text)
            self._speech_wake.set()

    def hush(self) -> None:
        """The model stops talking (it heard the caller / was told to)."""

        self._speech.clear()
        self._cut = True

    async def _speak_loop(self) -> None:
        while True:
            await self._speech_wake.wait()
            self._speech_wake.clear()
            while self._speech:
                text = self._speech.pop(0)
                self._cut = False
                self.speaking = True
                words = text.split()
                # A reply starts after the words it answers, and its positions
                # then run ahead of wall time.
                self.out = max(self.out, self.inp + 400)
                for index in range(0, len(words), 2):
                    if self._cut:
                        break
                    start = self.out
                    self.out += 450
                    self.provider.push(H.out_text(" " + " ".join(words[index:index + 2]), start, self.out))
                    self.provider.push(H.out_audio("QUFB"))
                    await asyncio.sleep(0.05)
                self.speaking = False

    # ── the caller's side (provider input deltas) ──────────────────
    async def utter(
        self, pieces: list[str], *, pauses: Optional[dict[int, float]] = None,
        after: Optional[dict[int, Callable[[], Any]]] = None,
    ) -> None:
        """Say `pieces` as provider input deltas, STEP apart on both clocks.

        `pauses[i]` adds a real hesitation (both clocks) before piece i;
        `after[i]` runs once piece i has been pushed (a mid-sentence decision,
        the app's interrupt)."""

        pauses = pauses or {}
        after = after or {}
        for index, piece in enumerate(pieces):
            if index:
                extra = pauses.get(index, 0.0)
                await asyncio.sleep(STEP + extra)
                self.inp += int((STEP + extra) * 1000) - SPAN
            else:
                # An utterance starts after the provider's last decision point
                # (`delegate` offsets sit 40 ms past the input end).
                self.inp += LEAD
            start = self.inp
            self.inp += SPAN
            self.provider.push(H.user_delta(piece, start, self.inp))
            hook = after.get(index)
            if hook is not None:
                result = hook()
                if asyncio.iscoroutine(result):
                    await result

    async def idle(self, wall_s: float, *, input_ms: int) -> None:
        """Silence: `wall_s` of wall time during which the provider's input
        timeline advances only `input_ms` (a plateau when input_ms < wall)."""

        await asyncio.sleep(wall_s)
        self.inp += int(input_ms)

    def delegate(self, did: str) -> None:
        """GPT-Live decides to delegate, at its current input position."""

        self.provider.push(H.delegation(did, self.inp + 40))

    async def barge_in(self, played_ms: int) -> None:
        """The app heard the caller over the agent: the model stops, the app
        sends `interrupt` for the epoch it is playing."""

        self.hush()
        await self.until(lambda: self.provider.queue.empty(), "provider drained", timeout=3)
        await asyncio.sleep(0.03)
        audio = self.phone.of("audio_delta")
        assert audio, "nothing was playing to interrupt"
        rid = audio[-1]["response_id"]
        self.phone.send({
            "type": "interrupt", "response_id": rid, "item_id": rid, "played_ms": played_ms,
        })
        await self.until(
            lambda: any(
                str(e.get("event_id") or "").startswith("toup-live-interrupt-")
                for e in self.provider.sent
            ) or any(f.get("type") == "playback_interrupted" for f in self.phone.frames),
            "interrupt handled",
        )

    # ── waiting ────────────────────────────────────────────────────
    async def until(self, predicate, label: str, *, timeout: float = 6.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate():
            if loop.time() >= deadline:
                raise AssertionError(f"timed out waiting for {label}")
            await asyncio.sleep(0.01)

    async def quiet(self, label: str = "the model finished speaking") -> None:
        await self.until(lambda: not self._speech and not self.speaking, label)
        # …and the output epoch has retired on its gap.
        await asyncio.sleep(0.4)

    async def final(self, text: str, *, timeout: float = 6.0) -> str:
        await self.until(
            lambda: text in self.phone.final_texts(), f"final turn {text!r}", timeout=timeout,
        )
        return self.phone.turn_of(text)

    async def done(self, did: str, *, timeout: float = 8.0) -> None:
        await self.until(lambda: self.phone.terminal(did), f"{did} terminal", timeout=timeout)

    # ── provider-append readers ────────────────────────────────────
    def commentary(self, since: int = 0) -> list[dict]:
        return [e for e in self.provider.sent[since:] if e.get("type") == "session.commentary.append"]

    def spoken(self, since: int = 0) -> str:
        return " ".join(str(e.get("content") or "") for e in self.commentary(since))

    def instructions(self, prefix: str = "", since: int = 0) -> list[dict]:
        return [
            e for e in self.provider.sent[since:]
            if e.get("type") == "session.instructions.append"
            and str(e.get("event_id") or "").startswith(prefix)
        ]

    def counted(self, name: str, since: int = 0, **fields) -> int:
        return sum(
            1 for counter, seen in self.counters[since:]
            if counter == name and all(seen.get(k) == v for k, v in fields.items())
        )

    def no_identity_free_instruction(self) -> None:
        for event in self.provider.sent:
            if event.get("type") in {"session.instructions.append", "session.thinking.append"}:
                content = str(event.get("content") or "")
                for phrase in IDENTITY_FREE:
                    assert phrase not in content, event

    # ── run ────────────────────────────────────────────────────────
    async def run(self, scenario: Callable[["Call"], Awaitable[None]], *, timeout: float = 60.0):
        async def drive():
            try:
                await asyncio.wait_for(self.started.wait(), timeout=5)
                await scenario(self)
            except BaseException as exc:  # surfaced after the relay is joined
                self.error = exc
            finally:
                self.phone.send({"type": "stop"})

        driver = asyncio.create_task(drive())
        try:
            await H.run_relay(
                self.phone, self.provider, timeout=timeout,
                db_session_id=self.db, language_pin=self.language_pin,
            )
        finally:
            if not driver.done():
                driver.cancel()
            await asyncio.gather(driver, return_exceptions=True)
            if self._speaker is not None:
                self._speaker.cancel()
                await asyncio.gather(self._speaker, return_exceptions=True)
        if self.error is not None:
            raise self.error


def _words(pieces: list[str]) -> str:
    return "".join(pieces).strip()


def _stamps_monotonic(phone: Phone) -> None:
    """The WIRE clock stays monotonic whatever the input timeline does
    (shipped clients drop a lifecycle frame whose `updated_ms` goes back)."""

    stamps = [f["updated_ms"] for f in phone.of("delegation") if "updated_ms" in f]
    assert stamps == sorted(stamps), stamps


# ══════════════════════════════════════════════════════════════════════
# V1 — 22:49:16, 5:53: music, next song, the long UofT robotics request,
#      "what happened?", the LLM question, the address and the campus fix
# ══════════════════════════════════════════════════════════════════════

MUSIC = ["یه", " آهنگ", " از", " امیر", " تتلو", " برام", " بذار"]
TATALOO = "Amir Tataloo - Man Avali Naboodam Vali Akharisham"
#: V1 frame f047, relay U:4 chars=22 — went to the play/search fast path.
NEXT = ["عوض", " بشه،", " بزن", " آهنگ", " بعدی"]
BAHANEH = "Bahaneh - Hayedeh"

#: V1 01:00–02:00: «روباتیکن پیدا کن واسم» (U:7, the first card's title),
#: «استادای», «ایرانی که», «توی», «کامپیوتر», «یو آفتی ان», «توی», «دانشکده»,
#: «کامپیوتر کسی», «نیست» (U:17, the second card's title), and at 02:30 the
#: robotics word itself split «ر» | «وبات» | «یک».
ROBOTICS = [
    "استادای", " ایرانی", " که", " توی", " کامپیوتر", " یو", " آفتی", " ان",
    " و", " تو", " کار", " ر", "وبات", "یک", " هستن", " رو", " پیدا", " کن", " واسم،",
    " توی", " دانشکده", " کامپیوتر", " کسی", " نیست",
]
#: Hesitations inside the sentence: shorter than the gap (both clocks), and
#: one after «واسم،» longer than the gap — a comma tail waits for the hard gap.
ROBOTICS_PAUSES = {8: 0.25, 12: 0.2, 19: 0.65, 23: 0.2}
ROBOTICS_ANSWER = (
    "دو نفر پیدا کردم: استاد آرمان رضایی و استاد نسرین کاظمی، هر دو در دانشکده "
    "کامپیوتر یوآفتی روی رباتیک و یادگیری ماشین کار می‌کنن."
)
STATUS = ["چی", " شد", " میگم"]                      # U:37 chars=10
LLM_QUESTION = ["کارشون", " ال‌ال‌امم"]                 # U:35 chars=6, U:36 chars=9
ADDRESS = ["آدرس", " دفترشون", " رو", " بفرست"]          # card «بفرست» (U:40)
ADDRESS_ANSWER = "دفترشون تو پردیس میسیساگاست: ساختمان دیویس، جاده میسیساگا."
#: V1 05:10–05:20 / V2 00:00: «یوآفت», «ی», «داون‌تاون», «نیست», «ند» — the
#: last card was titled «ند».
CAMPUS = ["نه،", " اینا", " یوآفت", "ی", " داون‌تاون", " نیست", "ند"]


def _v1_tenant(call: Call) -> None:
    tenant = call.tenant
    tenant.play_result = (
        f"Starting {TATALOO} on the user's device.",
        {"type": "youtube", "video_id": "v-tataloo", "title": TATALOO},
    )
    tenant.control_results = [{"ok": True, "reason": "executed", "title": BAHANEH}]
    tenant.plan("روباتیک", ROBOTICS_ANSWER, delay=0.9)
    tenant.plan("چی شد", "درباره‌ی کار ال‌ال‌امشون هنوز چیزی پیدا نکردم.", delay=0.2)
    tenant.plan("ال‌ال‌ام", "آره، استاد آرمان رضایی روی مدل‌های زبانی بزرگ هم کار می‌کنه.", delay=0.2)
    tenant.plan("آدرس", ADDRESS_ANSWER, delay=0.4)
    tenant.plan("داون‌تاون", "ببخشید، پردیس سنت جورج: ساختمان بهن، خیابان کالج.", delay=0.3)


async def _v1_until_first_status(call: Call, seen: dict) -> None:
    """V1 00:00 → 04:20: greeting, music, next, the long request, its heard
    result with echo, and «چی شد میگم» answered from that result."""

    phone = call.phone
    # ── greeting (00:09): a direct reply, no task ────────────────────
    await call.idle(0.2, input_ms=200)
    await call.utter(["سلام،", " خوبی؟"])
    await call.final("سلام، خوبی؟")
    call.voice("سلام! خوبم، تو چطوری؟ امروز چی کار می‌تونم برات بکنم؟")
    await call.quiet()

    # ── music (00:30): the play fast path ────────────────────────────
    await call.idle(0.3, input_ms=300)
    await call.utter(MUSIC)
    call.delegate("d-music")
    await call.done("d-music")
    await call.quiet()
    phone.send({"type": "now_playing", "title": TATALOO})

    # ── "next song" (00:46): the model decides before the endpoint ───
    await call.idle(0.3, input_ms=300)
    await call.utter(NEXT)
    call.delegate("d-next")
    await call.done("d-next")
    await call.quiet()
    phone.send({"type": "now_playing", "title": BAHANEH})

    # ── the long request (01:00–02:00), on a lagging input timeline ──
    # The wall clock runs on while the provider's input timeline stays put
    # (the app idles its uplink while music plays): the lag the relay saw.
    await call.idle(1.2, input_ms=150)
    first_decision = ROBOTICS.index(" واسم،")
    await call.utter(
        ROBOTICS, pauses=ROBOTICS_PAUSES,
        # GPT-Live delegated at «…پیدا کن واسم،» (item_ER7LwVC, U:7) while
        # the caller was still talking — and again at «نیست» (item_ER7MZrh).
        after={first_decision: lambda: call.delegate("d-rob")},
    )
    call.delegate("d-rob-again")
    rob_turn = await call.final(_words(ROBOTICS))
    seen["rob_turn"] = rob_turn
    await call.done("d-rob")

    # ── the result is SAID; the agent's voice echoes back (01:49) ────
    # Echo is judged against what the agent has ALREADY said in the epoch
    # that is still playing, so it arrives near the end of the answer.
    await call.until(
        lambda: any("یادگیری" in (f.get("text") or "") for f in phone.of("response_text")),
        "the robotics result is spoken",
    )
    speech_started_before = len(phone.of("speech_started"))
    counters_before = len(call.counters)
    await call.utter([" استاد آرمان رضایی", " دانشکده کامپیوتر"])
    seen["echo_speech_started"] = len(phone.of("speech_started")) - speech_started_before
    seen["echo_judged"] = call.counted("live_bargein_echo_suppressed", counters_before)
    await call.quiet()
    await call.done("d-rob-again", timeout=6)
    # A real silence, on both clocks: the echo turn closes on its own.
    await call.idle(0.9, input_ms=900)

    # ── «چی شد میگم» with nothing newer pending (04:14) ─────────────
    mark_frames = len(phone.frames)
    mark_sent = len(call.provider.sent)
    await call.utter(STATUS)
    call.delegate("d-status")
    status_turn = await call.final(_words(STATUS))
    seen["status_turn"] = status_turn
    await call.until(
        lambda: call.instructions("toup-live-status-", since=mark_sent),
        "the status answer from the result",
    )
    await call.until(
        lambda: call.phone.parents_after(mark_frames, "رضایی"),
        "the status answer is spoken",
    )
    seen["status_mark"] = mark_frames
    seen["status_sent"] = mark_sent


@pytest.mark.asyncio
async def test_v1_one_continuous_call(monkeypatch):
    """V1 end to end.  Production (RUNTIME_EVIDENCE 2–5, 9): the next-song ask
    played a video titled with its own residue words; the robotics request
    became ten turns and two cards titled «روباتیکن پیدا کن واسم» and
    «نیست»; «چی شد میگم» became an empty Done card; the LLM question was never
    worked on; the campus correction became a card titled «ند»."""

    call = Call(monkeypatch, features=TF132, name="v1")
    _v1_tenant(call)
    seen: dict = {}

    async def scenario(call: Call):
        phone = call.phone
        await _v1_until_first_status(call, seen)

        # ── the LLM question barges into the long replay (03:37) ─────
        # interrupt played_ms=19800; GPT-Live does not delegate it.
        seen["llm_mark"] = len(phone.frames)
        await call.utter(LLM_QUESTION, after={0: lambda: call.barge_in(19800)})
        seen["llm_turn"] = await call.final(_words(LLM_QUESTION))
        # ~33 s of wall silence (scaled).  The input timeline advances
        # 12 s: further apart than the request-span bound, as the relay
        # timeline puts U:36 and U:37 (the gap straddles the 24 s plateau).
        await call.idle(1.5, input_ms=12000)
        mark_sent = len(call.provider.sent)
        await call.utter(STATUS)
        call.delegate("d-status-again")
        await call.done("d-status-again")
        seen["status_again_sent"] = mark_sent
        await call.quiet()

        # ── the address (04:25) and the campus correction (05:10) ────
        await call.idle(0.5, input_ms=500)
        await call.utter(ADDRESS)
        call.delegate("d-addr")
        await call.done("d-addr")
        await call.quiet()
        await call.idle(0.4, input_ms=400)
        await call.utter(CAMPUS)
        call.delegate("d-campus")
        await call.done("d-campus")
        await call.quiet()

    await call.run(scenario, timeout=60)
    phone, tenant = call.phone, call.tenant

    # ── music: one play (the request), then a deterministic NEXT ──────
    assert tenant.plays == ["امیر تتلو"], tenant.plays
    assert tenant.controls == ["next"], "«عوض بشه، بزن آهنگ بعدی» is the next control"
    assert all("آهنگ" not in c["display"] for c in tenant.calls), "no agent turn for music"
    next_frames = [f for f in phone.of("media_control") if f.get("task_id") == "d-next"]
    assert [(f["action"], f["status"]) for f in next_frames] == [
        ("next", "requested"), ("next", "executed"),
    ]
    assert relay_line("media_playing", "fa", title=TATALOO) in call.spoken()
    assert relay_line("media_next", "fa", title=BAHANEH) in call.spoken()
    assert phone.phases("d-music")[-1] == "completed"
    assert phone.phases("d-next")[-1] == "completed"

    # ── the long request: ONE turn, ONE task, titled by the request ──
    request = _words(ROBOTICS)
    assert request in phone.final_texts()
    rob_finals = [t for t in phone.final_texts() if any(w in t for w in ("وبات", "کسی نیست"))]
    assert rob_finals == [request], "the request was split into turns"
    created = phone.created_for("d-rob")
    assert created["title"] == request_title(request)
    assert created["turn_id"] == seen["rob_turn"]
    assert created["title"] not in {"نیست", "کن", "واسم،", "روباتیکن پیدا کن واسم"}
    assert tenant.call("d-rob")["display"] == request
    # The second decision on the same words ran nothing and drew no card.
    assert phone.phases("d-rob-again") == ["expired"]
    assert phone.frames_for("d-rob-again")[0].get("reason") == "no_causal_turn"
    assert [c["did"] for c in tenant.calls].count("d-rob") == 1

    # ── echo: the agent's own words never became the caller's ───────
    assert seen["echo_judged"] >= 1, "the agent's own words were not recognised as echo"
    assert seen["echo_speech_started"] == 0, "the agent's echo fired a barge-in"
    assert not any("رضایی" in c["display"] for c in tenant.calls)
    assert _words(STATUS) in phone.final_texts(), "the echo prefixed the status question"

    # ── «چی شد میگم»: no card, answered from the result, parented ───
    assert phone.frames_for("d-status") == [], "a relay-answered status question is not a card"
    status_replays = call.instructions("toup-live-status-")
    assert len(status_replays) == 1
    assert "رضایی" in status_replays[0]["content"]
    # (Up to the LLM question: the answer to THAT question names the professor
    # too, and is parented on the question — asserted below.)
    parents = [
        f.get("parent_user_turn_id") or ""
        for f in phone.frames[seen["status_mark"]:seen["llm_mark"]]
        if f.get("type") == "response_text" and "رضایی" in (f.get("text") or "")
    ]
    assert parents and set(parents) == {seen["status_turn"]}, parents

    assert call.counted("live_status_answered", source="result") == 1
    # ── RT-2: after a HEARD result and a newer unanswered question, the
    #    second «چی شد میگم» is worked on — never re-read from the old result.
    #    Addendum 3: what is dispatched is the QUESTION itself — its words, its
    #    turn, its title — never the bare status words (the recorded V1 card
    #    titled «چی شد میگم» whose agent turn never saw the question).
    assert call.counted("live_status_dispatched", reason="newer_turn") == 1
    assert call.instructions("toup-live-status-", since=seen["status_again_sent"]) == []
    assert "d-status-again" in [c["did"] for c in tenant.calls]
    assert tenant.call("d-status-again")["display"] == _words(LLM_QUESTION)
    assert phone.created_for("d-status-again")["title"] == _words(LLM_QUESTION)
    assert phone.created_for("d-status-again")["turn_id"] == seen["llm_turn"]
    llm_parents = phone.parents_after(seen["llm_mark"], "زبانی")
    assert llm_parents and set(llm_parents) == {seen["llm_turn"]}, llm_parents

    # ── the address, then the campus correction ─────────────────────
    assert phone.created_for("d-addr")["title"] == _words(ADDRESS)
    campus = _words(CAMPUS)
    campus_card = phone.created_for("d-campus")
    assert campus_card["title"] == campus, "the correction is titled by the caller's words"
    lifecycle = [f for f in phone.frames_for("d-campus") if "task_revision" in f]
    # Pin change, R2 addendum 5 C11 (supervisor 21:54 / 22:1x), which names it:
    # "the V1 anaphoric statement (running → question; finished → link)".
    # «نه، اینا یوآفتی داون‌تاون نیستند» shares no word with «آدرس دفترشون رو
    # بفرست» or its answer, and absent shared words are no proof of
    # independence: it is a corrective statement that refers ANAPHORICALLY
    # («اینا», "these") to the result the caller just heard.  The address job
    # is the one plausible finished target (the caller asked for it after the
    # earlier results), so it is linked `replaces` — non-destructive — with the
    # address request (and its answer, in the ledger) as correcting context.
    # (Round 3's pin, no link + non-binding "request made just before", is
    # what C11 replaces.)
    # ORIGINAL PIN RESTORED — R2 addendum 6 R6-10 "V1 ORACLE CHANGE REJECTED":
    # after the address answer showed Mississauga the caller clarified
    # downtown; the correction contradicts the ADDRESS result.  Structural
    # rule (no campus keywords): a candidate's lexical evidence includes the
    # context it was bound/linked to — the address job resolved «دفترشون»
    # against the professors' result (via «کارشون ال‌ال‌امم»), so it carries
    # «یوآفتی» too; the words select the robotics, LLM and address jobs
    # EQUALLY, and an anaphoric correction («اینا») resolves to the most
    # recently delivered result: the address.  (fx8's d-rob pin is withdrawn.)
    assert lifecycle and all(
        f.get("relation") == {"kind": "replaces", "task_id": "d-addr"} for f in lifecycle
    ), lifecycle
    corrected = tenant.call("d-campus")["task"]
    assert (
        "Request the caller is correcting (keep its constraints unless the current "
        "request changes them): " + _words(ADDRESS)
    ) in corrected
    assert "(non-binding context" not in corrected
    assert (
        "Prior caller request: " + _words(ADDRESS) + "\nAccepted backend result: "
        + ADDRESS_ANSWER
    ) in corrected
    # …and the original constraints (UofT, computer science, robotics) ride
    # along with it, as the prior request and its accepted result.
    assert "Prior caller request: " + request in corrected
    assert "Follow-up rules" in corrected
    assert corrected.rstrip().endswith(
        "Current caller request: " + campus + "\n\nReply in the caller's language (Persian/Farsi)."
    ), corrected[-300:]
    assert phone.phases("d-addr")[-1] == "completed", "the finished address task stays finished"

    # ── every card on the call, and nothing else ────────────────────
    assert sorted(f["delegation_id"] for f in phone.created()) == sorted([
        "d-music", "d-next", "d-rob", "d-status-again", "d-addr", "d-campus",
    ])
    for frame in phone.created():
        assert frame["title"] not in {"نیست", "ند", "کن", "بفرست", "شد", "چی شد میگم"}, frame
    _stamps_monotonic(phone)
    call.no_identity_free_instruction()


@pytest.mark.asyncio
async def test_v1_rt2_status_bound_with_the_unanswered_question_is_worked_on(monkeypatch):
    """V1 03:37→04:14 with the input-timeline plateau at its steepest: the app's
    silent idle keeps the provider's input timeline almost still for the 33 s
    wall silence, so «کارشون ال‌ال‌امم» and «چی شد میگم» sit 1.5 s apart on it
    and the delegation binds both (one request span).  The span reads
    «کارشون ال‌ال‌امم چی شد میگم»: a newer question the heard result does not
    answer.  It must be worked on, not answered by re-reading that result."""

    call = Call(monkeypatch, features=TF132, name="v1-rt2")
    _v1_tenant(call)
    seen: dict = {}

    async def scenario(call: Call):
        await _v1_until_first_status(call, seen)
        await call.utter(LLM_QUESTION, after={0: lambda: call.barge_in(19800)})
        seen["llm_turn"] = await call.final(_words(LLM_QUESTION))
        await call.idle(1.5, input_ms=1500)       # the plateau
        seen["mark_sent"] = len(call.provider.sent)
        await call.utter(STATUS)
        call.delegate("d-status-again")
        await call.until(
            lambda: call.instructions("toup-live-status-", since=seen["mark_sent"])
            or any(c["did"] == "d-status-again" for c in call.tenant.calls),
            "the second status question is handled",
        )
        await asyncio.sleep(0.6)

    await call.run(scenario, timeout=60)

    assert call.instructions("toup-live-status-", since=seen["mark_sent"]) == [], (
        "the heard robotics result was re-read over the newer LLM question"
    )
    worked = [c for c in call.tenant.calls if c["did"] == "d-status-again"]
    assert worked, "the newer LLM question was never dispatched"
    assert "ال‌ال‌ام" in worked[0]["display"]
    call.no_identity_free_instruction()


# ══════════════════════════════════════════════════════════════════════
# V2 — 22:55:17, 3:56: «خب چی» / «شد», contact / office / origin /
#      affiliation follow-ups, and «آهنگ رو قطع» while a search runs
# ══════════════════════════════════════════════════════════════════════

PROF = [
    "استادای", " ایرانی", " کامپیوتر", " یوآفتی", " داون‌تاون", " رو", " پیدا",
    " کن", " که", " روی", " روباتیک", " کار", " می‌کنن",
]
PROF_ANSWER = (
    "آره، حق با تو بود! اون‌هایی که گفتم میسیساگا بودن. استاد آرمان رضایی "
    "استاد کامپیوتر یوآفتی داون‌تاونه و روی رباتیک کار می‌کنه."
)
#: V2 00:20: «خب چی» / «شد» — two turns (U:50 chars=5, U:51 chars=2).
STATUS_HEAD, STATUS_TAIL = ["خب", " چی"], [" ش", "د"]
#: V2 01:00: «خب», «شمارشو بگو», «شماره و», «ایمیلشو», «بگو» — card «بگو».
CONTACT = ["خب", " شمارشو بگو", " شماره و", " ایمیلشو", " بگو"]
#: V2 01:35: «اوکی», «اه», «کجای», «استاد» — card «استاد».
OFFICE = ["اوکی،", " اه", " کجای", " استاد"]
#: V2 01:40–02:10: «نه», «، ک», «جایی», «هستش», «استاده», «؟ ک», «دوم»,
#: «شهر», «دنیا... کدوم», «کشور», «دنیا؟» — cards «استاده», «دنیا؟ من».
ORIGIN = [
    "نه", "، ک", "جایی", " هستش", " استاده", "؟ ک", "دوم", " شهر", " دنیا...",
    " کدوم", " کشور", " دنیا؟",
]
ORIGIN_ANSWER = (
    "اهل شیرازه و دکتراش رو از آمریکا گرفته. الان استاد جورجیا تکه و با "
    "یوآفتی و وکتور همکاری می‌کنه، سال‌ها هم تو این زمینه مقاله داده."
)
#: V2 02:35: «خب، پس», «استاد», «یوآفتی», «نیست» — card «نیست».
AFFILIATION = ["خب،", " پس", " استاد", " یوآفتی", " نیست؟"]
#: V2 03:20: «اوکی», «، الان», «آهنگ رو قطع» — the recognizer dropped «کن».
STOP_1 = ["اوکی", "، الان", " آهنگ رو قطع"]
STOP_2 = ["آهنگ رو قطع"]


@pytest.mark.asyncio
async def test_v2_one_continuous_call(monkeypatch):
    """V2 end to end on a v0.3 client (media_transport negotiated).
    Production: «خب چی شد» ran a 13.6 s research turn titled «شد»; every
    follow-up was titled by its last fragment; the stop request produced no
    delegation, no control and no tenant call — the model said «چشم.» while
    the player kept playing."""

    call = Call(monkeypatch, features=V03, name="v2")
    tenant = call.tenant
    prof_gate = asyncio.Event()
    affil_gate = asyncio.Event()
    tenant.plan("روباتیک", PROF_ANSWER, delay=0.2, gate=prof_gate)
    tenant.plan("ایمیلشو", "ایمیلش arman.rezaei@cs.toronto.example است؛ شماره مستقیمی پیدا نکردم.", delay=0.2)
    tenant.plan("کجای استاد", "دفترش تو پردیس داون‌تاونه، ساختمان بهن.", delay=0.2)
    tenant.plan("کشور", ORIGIN_ANSWER, delay=0.2)
    tenant.plan("یوآفتی نیست", "نه، هست — استاد رسمی کامپیوتر یوآفتی داون‌تاونه؛ حرف جورجیا تک درست نبود.",
                delay=0.2, gate=affil_gate)
    # The first stop is acknowledged by the phone; the repeat is not.
    tenant.control_results = [
        {"ok": True, "reason": "stopped"},
        {"ok": False, "reason": "unacknowledged"},
    ]
    stopped_line = relay_line("media_stopped", "fa")
    unconfirmed_line = relay_line("media_stop_unconfirmed", "fa")
    seen: dict = {}

    async def before_the_tenant_answers(action):
        seen.setdefault("spoken_before_verdict", []).append(
            stopped_line in call.spoken() and len(tenant.controls) == 1
        )
        await asyncio.sleep(0.2)

    tenant.control_hook = before_the_tenant_answers

    async def scenario(call: Call):
        phone = call.phone
        phone.send({"type": "now_playing", "title": BAHANEH})
        await call.idle(0.2, input_ms=200)
        await call.utter(PROF)
        call.delegate("d-prof")
        await call.until(
            lambda: any(c.get("tool_started") for c in tenant.calls), "the search started",
        )

        # ── «خب چی» · real pause · «ش»«د» while the search runs ─────
        await call.idle(0.3, input_ms=300)
        await call.utter(STATUS_HEAD)
        await call.final(_words(STATUS_HEAD))
        await call.idle(0.2, input_ms=900)       # > gap on the input timeline
        mark_frames = len(phone.frames)
        await call.utter(STATUS_TAIL)
        call.delegate("d-status")
        seen["status_turn"] = await call.final(_words(STATUS_TAIL))
        await call.until(
            lambda: any("هنوز دارم روش کار می‌کنم" in e["content"] for e in call.commentary()),
            "the status line",
        )
        await call.until(lambda: phone.parents_after(mark_frames, "هنوز"), "status line spoken")
        seen["status_mark"] = mark_frames
        await call.quiet()
        prof_gate.set()
        await call.done("d-prof")
        await call.until(
            lambda: any("رضایی" in (f.get("text") or "") for f in phone.of("response_text")),
            "the professor result is spoken",
        )
        await call.quiet()

        # ── contact, office, origin ─────────────────────────────────
        for pieces, did in ((CONTACT, "d-contact"), (OFFICE, "d-office")):
            await call.idle(0.4, input_ms=400)
            await call.utter(pieces)
            call.delegate(did)
            await call.done(did)
            await call.quiet()
        await call.idle(0.4, input_ms=400)
        await call.utter(ORIGIN)
        call.delegate("d-origin")
        await call.done("d-origin")
        # ── the affiliation doubt barges into the long origin answer ─
        await call.until(
            lambda: any("شیرازه" in (f.get("text") or "") for f in phone.of("response_text")),
            "the origin result is spoken",
        )
        await call.utter(AFFILIATION, after={0: lambda: call.barge_in(27300)})
        call.delegate("d-affil")
        await call.until(
            lambda: any(c["did"] == "d-affil" and c.get("tool_started") for c in tenant.calls),
            "the affiliation search runs",
        )
        await call.quiet()

        # ── «اوکی، الان آهنگ رو قطع» while music plays and it searches ─
        phone.send({"type": "now_playing", "title": "Fadat Sham - Mahasti"})
        await call.idle(0.4, input_ms=400)
        await call.utter(STOP_1)
        call.delegate("d-stop")
        await call.done("d-stop")
        await call.until(lambda: stopped_line in call.spoken(), "the stopped line")
        await call.quiet()
        seen["affil_running_after_stop"] = not phone.terminal("d-affil")
        # …and the caller says it again; this time the phone never confirms.
        await call.idle(0.4, input_ms=400)
        await call.utter(STOP_2)
        call.delegate("d-stop-again")
        await call.done("d-stop-again")
        await call.until(lambda: unconfirmed_line in call.spoken(), "the unconfirmed line")
        await call.quiet()
        affil_gate.set()
        await call.done("d-affil")
        await call.quiet()

    await call.run(scenario, timeout=60)
    phone = call.phone

    # ── «خب چی» + «شد» is ONE status question: no research, no card ──
    assert [c["did"] for c in tenant.calls].count("d-status") == 0, "a status question was researched"
    assert call.counted("live_status_answered", source="running") == 1
    assert phone.frames_for("d-status") == []
    status_lines = [e for e in call.commentary() if "هنوز دارم روش کار می‌کنم" in e["content"]]
    assert len(status_lines) == 1
    assert status_lines[0]["delegation_id"] == "d-prof", "the update is about the running search"
    assert "در حال جستجو در وب" in status_lines[0]["content"], "a localized label of the real step"
    assert "Searching" not in status_lines[0]["content"]
    assert _words(PROF)[:20] not in status_lines[0]["content"], "no raw query aloud"
    parents = phone.parents_after(seen["status_mark"], "هنوز")
    assert parents and set(parents) == {seen["status_turn"]}, parents
    assert phone.phases("d-prof")[-1] == "completed"

    # ── follow-ups: titled by the whole request, same entity, grounded ─
    for did, pieces in (
        ("d-contact", CONTACT), ("d-office", OFFICE), ("d-origin", ORIGIN),
        ("d-affil", AFFILIATION),
    ):
        words = _words(pieces)
        assert phone.created_for(did)["title"] == request_title(words), did
        record = tenant.call(did)
        assert record["display"] == words, did
        message = record["task"]
        assert "Prior caller request: " + _words(PROF) in message, did
        assert "رضایی" in message, f"{did} lost the professor the caller is asking about"
        low = message.lower()
        assert "still apply unless the caller changes them" in low, did
        assert "current position on the organisation's own current page" in low, did
        assert "former role" in low, did
        assert "differs from an accepted backend result" in low, did
    for fragment_title in ("بگو", "استاد", "استاده", "دنیا؟ من", "نیست", "شد"):
        assert fragment_title not in [f["title"] for f in phone.created()], fragment_title
    # «نه، کجایی هستش استاده؟…» corrects the office question it follows.
    origin_frames = [f for f in phone.frames_for("d-origin") if "task_revision" in f]
    assert origin_frames and all(
        f.get("relation") == {"kind": "replaces", "task_id": "d-office"} for f in origin_frames
    ), origin_frames
    assert "Request the caller is correcting" in tenant.call("d-origin")["task"]
    assert _words(OFFICE) in tenant.call("d-origin")["task"]
    # The affiliation doubt carries the accepted Georgia Tech answer it contradicts.
    assert "جورجیا تک" in tenant.call("d-affil")["task"]

    # ── the stop: a deterministic control, worded by the device verdict ─
    assert tenant.controls == ["stop", "stop"]
    assert tenant.plays == []
    assert not any(c["did"] in {"d-stop", "d-stop-again"} for c in tenant.calls), (
        "a stop went to a full agent turn"
    )
    assert phone.created_for("d-stop")["title"] == _words(STOP_1)
    first = [f for f in phone.of("media_control") if f.get("task_id") == "d-stop"]
    assert [(f["action"], f["status"]) for f in first] == [("stop", "requested"), ("stop", "executed")]
    assert first[1]["ok"] is True and first[1]["reason"] == "stopped"
    assert first[0]["control_id"] == first[1]["control_id"]
    assert seen["spoken_before_verdict"][0] is False, "«stopped» was said before the phone acked"
    stopped_at = next(
        i for i, e in enumerate(call.provider.sent)
        if e.get("type") == "session.commentary.append" and e.get("content") == stopped_line
    )
    assert stopped_at >= tenant.control_returned_at[0]
    assert phone.phases("d-stop")[-1] == "completed"
    second = [f for f in phone.of("media_control") if f.get("task_id") == "d-stop-again"]
    assert [(f["action"], f["status"]) for f in second] == [("stop", "requested"), ("stop", "error")]
    assert second[1]["reason"] == "unacknowledged" and second[1]["ok"] is False
    assert phone.phases("d-stop-again")[-1] == "failed"
    stop_again_lines = [e["content"] for e in call.commentary() if e["delegation_id"] == "d-stop-again"]
    assert stop_again_lines == [unconfirmed_line], stop_again_lines
    # The search survived both stops and the barge-in; nothing was cancelled.
    assert seen["affil_running_after_stop"] is True
    assert phone.phases("d-affil")[-1] == "completed"
    assert not any(c["cancelled"] for c in tenant.calls)
    for did in ("d-origin", "d-affil", "d-prof"):
        assert not {"cancelled", "superseded"} & set(phone.phases(did)), did
    _stamps_monotonic(phone)
    call.no_identity_free_instruction()


@pytest.mark.asyncio
@pytest.mark.parametrize(("reason", "line_key", "phase"), [
    ("stopped", "media_stopped", "completed"),
    ("unacknowledged", "media_stop_unconfirmed", "failed"),
])
async def test_v2_stop_on_the_recorded_tf132_client(monkeypatch, reason, line_key, phase):
    """The recorded app did not negotiate `media_transport`: TF132's tracker
    drops a media_control frame whose action it does not know.  The stop
    still runs (task, tool rows, the tenant call, the spoken verdict), but no
    frame carrying action `stop` ever reaches that client."""

    call = Call(monkeypatch, features=TF132, name=f"v2-tf132-{reason}")
    tenant = call.tenant
    gate = asyncio.Event()
    tenant.plan("یوآفتی نیست", "نه، هست — استاد رسمی کامپیوتر یوآفتی داون‌تاونه.", delay=0.1, gate=gate)
    tenant.control_results = [{"ok": reason == "stopped", "reason": reason}]
    line = relay_line(line_key, "fa")

    async def scenario(call: Call):
        call.phone.send({"type": "now_playing", "title": "Fadat Sham - Mahasti"})
        await call.idle(0.2, input_ms=200)
        await call.utter(AFFILIATION)
        call.delegate("d-affil")
        await call.until(lambda: any(c.get("tool_started") for c in tenant.calls), "search runs")
        await call.idle(0.4, input_ms=400)
        await call.utter(["آهنگ", " رو", " قطع", " کن"])
        call.delegate("d-stop")
        await call.done("d-stop")
        await call.until(lambda: line in call.spoken(), "the verdict line")
        await call.quiet()
        gate.set()
        await call.done("d-affil")
        await call.quiet()

    await call.run(scenario, timeout=30)
    phone = call.phone

    assert tenant.controls == ["stop"]
    assert phone.stop_actions() == [], "a TF132 client received a stop/pause action"
    assert [f for f in phone.of("media_control") if f.get("task_id") == "d-stop"] == []
    assert phone.phases("d-stop")[-1] == phase
    started = [f for f in phone.of("tool_call.started") if f.get("name") == "media_control"]
    assert started and started[0]["task_id"] == "d-stop" and started[0]["detail"] == "stop"
    spoken = [e["content"] for e in call.commentary() if e["delegation_id"] == "d-stop"]
    assert spoken == [line], spoken
    assert phone.phases("d-affil")[-1] == "completed"
    assert not any(c["cancelled"] for c in tenant.calls)
    instructions = call.provider.of("session.start")[0]["session"]["instructions"]
    assert "pause or stop playback" not in instructions
    call.no_identity_free_instruction()


# ══════════════════════════════════════════════════════════════════════
# V3 — 22:59:25, 1:40: stop, «…باهام انگلیسی حرف بزنی فقط», the English
#      UofT/LLM question, and the app re-asking the old «قطع»
# ══════════════════════════════════════════════════════════════════════

#: V3 00:10–00:14: «آهنگ رو», «نمی‌ت», «ونی», «قطع» (U:83 chars=7, U:84
#: chars=5, U:85 chars=3; U:86 «قطع» chars=3 three seconds later).
STOP_HEAD = ["آهنگ رو", " نمی‌ت", "ونی"]
STOP_TAIL = [" قطع"]
#: V3 00:31: «الان می‌ت» (U:87 chars=9) + «ونی... می‌تونی باهام انگلیسی حرف
#: بزنی فقط» (U:88 chars=41).
ENGLISH_ONLY = ["الان", " می‌ت", "ونی...", " می‌تونی", " باهام", " انگلیسی", " حرف", " بزنی", " فقط"]
ENGLISH_AGAIN = ["می‌تونی", " انگلیسی", " حرف", " بزنی؟"]
#: V3 01:01–01:15.7 (U:91–U:100): the question no delegation was ever made for.
HESITANT = [
    "What", " the most", ", uh... the best", " professor in UofT",
    " who is", " working on", " LLM",
]
HESITANT_PAUSES = {1: 0.02, 2: 0.15, 3: 0.02, 6: 0.15}
EMAIL = ["ایمیلشو", " پیدا", " کن"]
LLM_ANSWER = "Professor Arman Rezaei at U of T works on large language models for robotics."


@pytest.mark.asyncio
@pytest.mark.parametrize("app", ["tf132", "v03"])
async def test_v3_one_continuous_call(monkeypatch, app):
    """V3 end to end, on the recorded TF132 client and on a v0.3 client.
    Production: the stop never executed; «…باهام انگلیسی حرف بزنی فقط» was
    agreed to in English but nothing held it; the English question became ten
    turns and no delegation; the app replayed the old «قطع» and the model then
    said «الان قطع می‌کنم.» in Persian under the English question."""

    features = TF132 if app == "tf132" else V03
    call = Call(monkeypatch, features=features, name=f"v3-{app}")
    tenant = call.tenant
    tenant.control_results = [{"ok": True, "reason": "stopped"}]
    llm_gate = asyncio.Event()
    tenant.plan("LLM", LLM_ANSWER, delay=0.2, gate=llm_gate)
    tenant.plan("ایمیلشو", "ایمیلش arman.rezaei@cs.toronto.example است.", delay=0.2)
    seen: dict = {}

    async def scenario(call: Call):
        phone = call.phone
        phone.send({"type": "now_playing", "title": "Fadat Sham - Mahasti"})
        # The call opens on the tail of a 44 s spoken answer (played_ms=44400).
        await call.idle(0.2, input_ms=200)
        call.voice(
            "نه ددی، هست — استاد آرمان رضایی استاد رسمی کامپیوتر تو خود یوآفتی "
            "داون‌تاونه، اون حرف جورجیا تک درست نبود و ازت عذر می‌خوام بابتش."
        )
        await call.until(lambda: phone.of("audio_delta"), "the answer is playing")

        # ── the stop, barging in; «قطع» three seconds later ──────────
        await call.utter(STOP_HEAD, after={0: lambda: call.barge_in(44400)})
        await call.final(_words(STOP_HEAD))
        await call.idle(1.0, input_ms=1000)
        await call.utter(STOP_TAIL)
        call.delegate("d-stop")
        seen["qat_turn"] = await call.final(_words(STOP_TAIL))
        await call.done("d-stop")
        await call.quiet()

        # ── «الان می‌ت» «ونی... می‌تونی باهام انگلیسی حرف بزنی فقط» ──
        await call.idle(0.8, input_ms=300)
        await call.utter(ENGLISH_ONLY)
        await call.final(_words(ENGLISH_ONLY))
        # The request is committed once the model has ANSWERED it on its own
        # (fix-round F20: a closed turn is not yet a closed request — "Reply in
        # English" + a pause + "to the email from Sara" is one task), so the
        # note follows the model's reply rather than the turn's close.
        call.voice("Sure. I'll speak only English with you.")
        await call.until(lambda: call.instructions("toup-live-language-"), "the preference note")
        await call.quiet()
        await call.idle(0.4, input_ms=400)
        await call.utter(ENGLISH_AGAIN)
        await call.final(_words(ENGLISH_AGAIN))
        call.voice("I can. Let's keep it in English, and I will answer in English from now on.")
        await call.until(
            lambda: any("keep" in (f.get("text") or "") for f in phone.of("response_text")),
            "the second English reply is playing",
        )

        # ── the English question, barging in (played_ms=7700), on an
        #    input timeline far behind the wire clock (the output
        #    transcript ran ahead; the input idled on its plateau) ─────
        await call.utter(
            HESITANT, pauses=HESITANT_PAUSES,
            after={1: lambda: call.barge_in(7700)},
        )
        call.delegate("d-llm")
        seen["llm_turn"] = await call.final(_words(HESITANT))
        await call.until(lambda: any(c["did"] == "d-llm" for c in tenant.calls), "the LLM search runs")

        # ── the app's watchdog re-asks the OLD fragment «قطع» ─────────
        seen["reask_mark"] = len(call.provider.sent)
        phone.send({"type": "inject_text", "text": "قطع"})
        if app == "v03":
            phone.send({
                "type": "inject_text", "text": "قطع", "reason": "no_response",
                "reask_of_user_turn_id": seen["qat_turn"],
            })
            phone.send({
                "type": "inject_text", "text": "LLM", "reason": "no_response",
                "reask_of_user_turn_id": seen["llm_turn"],
            })
        await asyncio.sleep(0.4)
        seen["after_reask"] = len(call.provider.sent)
        llm_gate.set()
        await call.done("d-llm")
        await call.until(lambda: LLM_ANSWER in call.spoken(), "the LLM answer")
        await call.quiet()

        # ── a Persian-script request after "English only" ───────────
        call.muted.add("d-email")          # the model does not voice it at first
        await call.idle(0.4, input_ms=400)
        await call.utter(EMAIL)
        call.delegate("d-email")
        await call.done("d-email")
        await call.until(lambda: call.instructions("toup-live-nudge-d-email"), "the nudge")
        await call.quiet()

    await call.run(scenario, timeout=60)
    phone = call.phone

    # ── the stop executed; the old «قطع» is its own (consumed) turn ────
    assert tenant.controls == ["stop"]
    assert phone.created_for("d-stop")["title"] == "آهنگ رو نمی‌تونی قطع"
    assert phone.phases("d-stop")[-1] == "completed"
    assert relay_line("media_stopped", "fa") in call.spoken()
    stop_frames = [f for f in phone.of("media_control") if f.get("task_id") == "d-stop"]
    if app == "v03":
        assert [(f["action"], f["status"]) for f in stop_frames] == [
            ("stop", "requested"), ("stop", "executed"),
        ]
    else:
        assert stop_frames == [] and phone.stop_actions() == []

    # ── "English only": one preference, one note, told to the app if asked ─
    notes = call.instructions("toup-live-language-")
    assert len(notes) == 1 and "English" in notes[0]["content"], notes
    if app == "v03":
        assert phone.of("language") == [{"type": "language", "lang": "en", "source": "explicit"}]
    else:
        assert phone.of("language") == []

    # ── the English question: ONE turn, ONE delegation, the whole question ─
    question = _words(HESITANT)
    english_finals = [t for t in phone.final_texts() if any(w in t for w in ("UofT", "LLM", "What"))]
    assert english_finals == [question], english_finals
    llm = phone.created_for("d-llm")
    assert llm["title"] == question
    assert llm["turn_id"] == seen["llm_turn"]
    record = tenant.call("d-llm")
    assert record["display"] == question
    assert record["lang"] == "en"
    assert [c["did"] for c in tenant.calls].count("d-llm") == 1

    # ── the re-asks: the old «قطع» is refused; nothing identity-free ───
    assert call.instructions("toup-live-reask-") == []
    for event in call.provider.sent[seen["reask_mark"]:seen["after_reask"]]:
        if event.get("type") in {"session.instructions.append", "session.thinking.append"}:
            assert "قطع" not in str(event.get("content") or ""), event
    results = [(f["user_turn_id"], f["outcome"]) for f in phone.of("reask_result")]
    if app == "v03":
        assert [r for r in results if r[0] == seen["qat_turn"]] and all(
            outcome in {"answered", "stale"} for tid, outcome in results if tid == seen["qat_turn"]
        ), results
        assert (seen["llm_turn"], "in_flight") in results, results
        assert not any(outcome == "accepted" for _tid, outcome in results), results
    else:
        assert results == []
    call.no_identity_free_instruction()
    # Nothing Persian was said about the stop after the English question.
    assert relay_line("media_stopped", "fa") not in call.spoken(seen["reask_mark"])

    # ── the Persian-script request is delivered in English ─────────────
    email = tenant.call("d-email")
    assert email["lang"] == "en"
    assert "Reply in English" in email["task"] and "Persian/Farsi" not in email["task"]
    nudges = call.instructions("toup-live-nudge-d-email")
    assert len(nudges) == 1
    assert nudges[0]["content"].startswith("The backend finished the caller's earlier request."), nudges
    assert "Reply in English" in record["task"]
    persian_nudge = relay_line("result_nudge", "fa", result="").split(".")[0]
    assert not any(persian_nudge in n["content"] for n in call.instructions("toup-live-nudge-"))
    _stamps_monotonic(phone)


@pytest.mark.asyncio
@pytest.mark.parametrize("app", ["tf132", "v03"])
async def test_v3_the_unanswered_stop_fragment_is_not_reasked_after_the_english_question(
    monkeypatch, app,
):
    """The recorded shape exactly: GPT-Live never delegated the stop (U:83–U:86
    closed with no delegation), so «قطع» is an UNANSWERED turn of its own —
    and ~60 s later, after the caller had asked for English and asked the
    English question, the app's watchdog replayed it.  The newer turns make it
    stale: it is refused, the model is told nothing about it, and no tenant
    control fires on its account."""

    features = TF132 if app == "tf132" else V03
    call = Call(monkeypatch, features=features, name=f"v3-stale-{app}")
    tenant = call.tenant
    # The music is playing (now_playing below): a stop the tenant carries out.
    tenant.control_results = [{"ok": True, "reason": "stopped"}]
    tenant.plan("LLM", LLM_ANSWER, delay=0.2)
    seen: dict = {}

    async def scenario(call: Call):
        phone = call.phone
        phone.send({"type": "now_playing", "title": "Fadat Sham - Mahasti"})
        await call.idle(0.2, input_ms=200)
        call.voice("نه ددی، هست — استاد آرمان رضایی استاد رسمی کامپیوتر تو خود یوآفتی داون‌تاونه.")
        await call.until(lambda: phone.of("audio_delta"), "the answer is playing")
        await call.utter(STOP_HEAD, after={0: lambda: call.barge_in(44400)})
        await call.final(_words(STOP_HEAD))
        await call.idle(1.0, input_ms=1000)
        await call.utter(STOP_TAIL)
        seen["qat_turn"] = await call.final(_words(STOP_TAIL))
        # No delegation, no reply: nothing reached the caller for it.
        await call.idle(0.8, input_ms=300)
        await call.utter(ENGLISH_ONLY)
        await call.final(_words(ENGLISH_ONLY))
        call.voice("Sure. I'll speak only English with you.")
        await call.quiet()
        await call.idle(0.4, input_ms=400)
        await call.utter(HESITANT, pauses=HESITANT_PAUSES)
        call.delegate("d-llm")
        await call.final(_words(HESITANT))
        await call.done("d-llm")
        await call.quiet()
        seen["mark"] = len(call.provider.sent)
        seen["controls_at_mark"] = list(tenant.controls)
        phone.send({"type": "inject_text", "text": "قطع"})
        if app == "v03":
            phone.send({
                "type": "inject_text", "text": "قطع", "reason": "no_response",
                "reask_of_user_turn_id": seen["qat_turn"],
            })
        await asyncio.sleep(0.5)

    await call.run(scenario, timeout=40)
    phone = call.phone

    # Round 3 pin change (addendum 3 item 3): on a v0.3 client the recorded
    # split stop («آهنگ رو نمی‌تونی» + «قطع», the model silent) is the media
    # backstop's authority unit and is executed AT THE TIME — once, before
    # anything else is said.  The later replay of «قطع» is still no command:
    # no tenant control fires on its account.  TF132 negotiated no stop.
    assert tenant.controls == (["stop"] if app == "v03" else [])
    assert tenant.controls == seen["controls_at_mark"], "a stale fragment is not a command"
    assert call.instructions("toup-live-reask-") == []
    for event in call.provider.sent[seen["mark"]:]:
        assert event.get("type") not in {
            "session.instructions.append", "session.thinking.append",
            "session.commentary.append",
        }, event
    call.no_identity_free_instruction()
    results = [(f["user_turn_id"], f["outcome"]) for f in phone.of("reask_result")]
    if app == "v03":
        # The relay carried «قطع» out itself (its turn is consumed by the
        # stop), so the replay is refused as `answered` — never re-asked.
        assert results == [(seen["qat_turn"], "answered"), (seen["qat_turn"], "answered")], results
    else:
        assert results == []


@pytest.mark.asyncio
async def test_v3_the_answer_to_the_question_that_barged_in_is_not_held_by_an_expired_fence(
    monkeypatch,
):
    """V3 01:03.6: the English question barged into «I can. Let's keep it in
    English» (interrupt, played_ms=7700).  The model obeys "stop speaking" and
    says nothing more; the question's delegation finishes well after the
    interrupt fence (here 400 ms) has run out.  Its answer is appended at once —
    not after the result deferral budget."""

    call = Call(
        monkeypatch, features=V03, name="v3-fence",
        clocks={"voice_live_result_defer_max_ms": 3000, "voice_live_speak_timeout_s": 3.0},
    )
    call.tenant.plan("LLM", LLM_ANSWER, delay=0.9)

    async def scenario(call: Call):
        phone = call.phone
        await call.idle(0.2, input_ms=200)
        await call.utter(ENGLISH_AGAIN)
        await call.final(_words(ENGLISH_AGAIN))
        call.voice("I can. Let's keep it in English, and I will answer in English from now on.")
        await call.until(
            lambda: any("keep" in (f.get("text") or "") for f in phone.of("response_text")),
            "the reply is playing",
        )
        await call.utter(HESITANT, pauses=HESITANT_PAUSES, after={1: lambda: call.barge_in(7700)})
        call.delegate("d-llm")
        await call.done("d-llm")
        await call.until(lambda: LLM_ANSWER in call.spoken(), "the answer is appended", timeout=6)
        await call.quiet()

    await call.run(scenario, timeout=40)
    phone = call.phone

    finished = next(
        t for f, t in zip(phone.frames, phone.times)
        if f.get("type") == "delegation" and f.get("delegation_id") == "d-llm"
        and f.get("phase") == "completed"
    )
    interrupted = next(
        t for e, t in zip(call.provider.sent, call.sent_times)
        if str(e.get("event_id") or "").startswith("toup-live-interrupt-")
    )
    appended = next(
        t for e, t in zip(call.provider.sent, call.sent_times)
        if e.get("type") == "session.commentary.append" and e.get("content") == LLM_ANSWER
    )
    assert finished - interrupted > 0.8, "the fence (400 ms) had long run out"
    assert appended - finished < 0.6, (
        f"the answer waited {appended - finished:.2f}s behind an interrupt fence that "
        f"expired {finished - interrupted - 0.4:.2f}s before it finished"
    )
