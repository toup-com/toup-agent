#!/usr/bin/env python3
"""Controlled experiment — the gate for PR-5's uplink keeper (P1, 2026-10-02).

QUESTION: does GPT-Live take (ack) and voice a context append while NO input
audio is reaching it, and does real-time PCM16 digital silence unblock it?

The 2026-10-02 call says "no, and probably yes": while the caller was muted
(build 138 sends no frame at all) the result's `session.commentary.append` and
its one nudge were never acked (`reason=no_ack`), no output epoch began, and the
backlog was voiced the moment audio resumed. That is an inference from logs.
The relay's uplink keeper (`voice_live_uplink_keeper_enabled`, default OFF)
acts on it; this script tests it before anyone turns the keeper on.

Usage — from backend/, in the platform environment (`settings.openai_api_key`
is the platform's OpenAI key; it is never printed):

    python -m scripts.live_append_experiment --mode A --seconds 20
    python -m scripts.live_append_experiment --mode B --seconds 20

  --mode A   no input audio at all after `session.started` (a muted caller).
  --mode B   100 ms of PCM16 24 kHz digital silence (4,800 zero bytes) every
             100 ms of wall time from `session.started` to the end — exactly
             the chunk the relay's keeper sends, through the relay's own
             `audio_append` builder.

Both modes open the session with the relay's own `build_session_start` and
`LIVE_WS_URL`, wait `--lead-in` seconds after `session.started`, append ONE
line, observe for `--seconds`, then close gracefully and print:

  * a timeline of every provider event TYPE (ms after the append; audio is
    counted, never printed),
  * whether and when our append was acked (`session.<kind>.appended` with our
    `client_event_id`, and the ack's `start_ms`/`end_ms` when it carries them)
    or refused (`error` carrying it),
  * when the first output transcript / audio delta arrived, and the words,
  * the usage the provider reports at close.

`--kind commentary` (default) appends with `commentary_append`, which names a
delegation. Outside a real call there is no provider delegation, so GPT-Live
may refuse a synthetic `--delegation-id` — the refusal is printed, and is an
answer too. `--kind instructions` uses `instructions_append` (the relay's
nudge/status path), which needs no delegation; `--kind both` sends the
commentary and then the instruction, the way the relay's ladder does, and
`--kind none` sends no append at all.

Run each mode at least twice; the verdict that gates the keeper is A: no ack
and no output within the window, B: an ack and output deltas soon after the
append. Costs one short Live session per run (mode B bills ~`--seconds` of
input audio).

Q3 (2026-10-05) — does GPT-Live answer a SPOKEN question when no input at all
follows it (a caller who asks, then mutes)?

    python -m scripts.live_append_experiment --speech-file /tmp/q.pcm --kind none --after none --seconds 15
    python -m scripts.live_append_experiment --speech-file /tmp/q.pcm --kind none --after silence --seconds 15

  `--speech-file` is raw PCM16 mono 24 kHz little-endian with no header
  (`scripts/make_speech_pcm.sh` makes one on macOS). After the lead-in it goes
  out on the keeper's schedule — 4,800 bytes (100 ms) per 100 ms slot through
  `audio_append`; a late chunk is delayed, never dropped or burst — then
  `--after none` sends nothing more and `--after silence` sends the keeper's
  silence chunk every 100 ms until the window closes. The window and every
  time printed start at the END of the speech, so what arrived during it is
  negative. With client delegation, a question the model cannot answer itself
  may produce `session.delegation.created` rather than speech; the verdict
  lists either as the reply starting.

Q5 (every mode) — does GPT-Live stream SILENT audio, which would mint phantom
output epochs in the relay? Every `session.output_audio.delta` is decoded and
counted per output epoch — split where no output delta arrived for the relay's
`voice_live_output_epoch_gap_ms` (its ordinary gap; the mid-sentence hard gap
can only merge two of these) — by bytes, deltas, last delta, and the deltas
that are all-zero or below RMS 50 (int16 units); then the total audio seconds
against the wall time from the first audio delta to the last.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import base64
import json
import logging
import math
import sys
import time
import uuid
from typing import Any, Optional

# Never let a debug-level handler print the handshake (it carries the key).
logging.getLogger("websockets").setLevel(logging.WARNING)

ACKS = {
    "session.commentary.appended",
    "session.instructions.appended",
    "session.thinking.appended",
}
DEFAULT_LINE = (
    "The backend finished the caller's earlier request. The library on campus "
    "closes at eleven tonight."
)
DEFAULT_INSTRUCTIONS = (
    "You are a voice assistant on a phone call. When the backend appends a "
    "result or an instruction to tell the caller something, say it to the "
    "caller now, briefly, in English."
)

#: PCM16 mono at 24 kHz — the format `build_session_start` declares both ways.
PCM_BYTES_PER_S = 48_000
CHUNK_BYTES = 4_800                     # 100 ms: the keeper's chunk
#: Below this RMS (int16 units) an output delta, or 10 ms of the speech, is silence.
QUIET_RMS = 50


class Redactor:
    """Every string this script prints goes through here: the key never does."""

    def __init__(self, secret: str):
        self.secret = secret or ""

    def __call__(self, text: Any) -> str:
        text = str(text)
        if self.secret and self.secret in text:
            text = text.replace(self.secret, "***")
        return text


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


def _at(ms: Optional[int]) -> str:
    return "none" if ms is None else f"{ms:+d} ms"


def _span(event: dict) -> str:
    """The provider-timeline fields an event carries, when it carries them."""

    return "".join(f" {k}={event[k]}" for k in ("start_ms", "end_ms") if event.get(k) is not None)


def _rms(pcm: bytes) -> float:
    samples = array.array("h", pcm[: len(pcm) // 2 * 2])
    if sys.byteorder != "little":
        samples.byteswap()              # the wire is little-endian
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


def _quiet_tail_ms(pcm: bytes) -> int:
    """How much of the speech's own end is already quiet (10 ms frames): input
    the model hears after the words even with `--after none`."""

    end = len(pcm)
    while end > 0 and _rms(pcm[max(0, end - 480):end]) < QUIET_RMS:
        end -= 480
    return (len(pcm) - max(0, end)) * 1000 // PCM_BYTES_PER_S


async def _send_silence(ws, stop: asyncio.Event, sent: dict, chunk_b64: str, audio_append) -> None:
    """Real-time pace: one 100 ms chunk per 100 ms of wall time, never a burst."""

    next_at = time.monotonic()
    while not stop.is_set():
        await ws.send(json.dumps(audio_append(chunk_b64)))
        sent["chunks"] += 1
        sent_at = time.monotonic()
        next_at += 0.1
        while next_at - sent_at < 0.05:
            next_at += 0.1          # behind: skip the slot, as the keeper does
        try:
            await asyncio.wait_for(stop.wait(), timeout=next_at - time.monotonic())
        except asyncio.TimeoutError:
            pass


async def _send_speech(
    ws, stop: asyncio.Event, sent: dict, speech: bytes, after: str, silence_b64: str, audio_append,
) -> None:
    """The question on `_send_silence`'s schedule: one 4,800-byte (100 ms) chunk
    per slot, a late slot skipped — so a speech chunk is DELAYED, never dropped
    and never burst.  `sent["speech_end"]` is the last chunk's send time plus
    its length: when the question ends in real time.  `--after silence` keeps
    the schedule with the keeper's chunk until `stop`."""

    chunks = [speech[i:i + CHUNK_BYTES] for i in range(0, len(speech), CHUNK_BYTES)]
    next_at = time.monotonic()
    while not stop.is_set():
        if sent["speech"] < len(chunks):
            chunk = chunks[sent["speech"]]
            await ws.send(json.dumps(audio_append(base64.b64encode(chunk).decode("ascii"))))
            sent["speech"] += 1
            if sent["speech"] == len(chunks):
                sent["speech_end"] = time.monotonic() + len(chunk) / PCM_BYTES_PER_S
        elif after == "silence":
            await ws.send(json.dumps(audio_append(silence_b64)))
            sent["chunks"] += 1
        else:
            return
        sent_at = time.monotonic()
        next_at += 0.1
        while next_at - sent_at < 0.05:
            next_at += 0.1
        try:
            await asyncio.wait_for(stop.wait(), timeout=next_at - time.monotonic())
        except asyncio.TimeoutError:
            pass


async def run(args: argparse.Namespace) -> int:
    from app.config import settings
    from app.services.live_voice_protocol import (
        LIVE_MODEL,
        LIVE_WS_URL,
        _UPLINK_KEEPER_SILENCE_B64,
        audio_append,
        build_session_start,
        close_session,
        commentary_append,
        instructions_append,
    )
    import websockets

    key = getattr(settings, "openai_api_key", None) or ""
    redact = Redactor(key)
    if not key:
        print("settings.openai_api_key is not set in this environment — nothing to do.")
        return 2
    speech: bytes = args.speech or b""
    gap_ms = int(getattr(settings, "voice_live_output_epoch_gap_ms", 700))   # the relay's
    stimulus = (
        f"speech={len(speech) * 1000 // PCM_BYTES_PER_S}ms "
        f"quiet_tail={_quiet_tail_ms(speech)}ms after={args.after}"
        if speech else f"mode={args.mode}"
    )
    print(
        f"{stimulus} kind={args.kind} lead_in={args.lead_in:.1f}s "
        f"observe={args.seconds:.1f}s model={args.model or LIVE_MODEL} "
        f"url={LIVE_WS_URL} key=set epoch_gap={gap_ms}ms"
    )

    start_event = build_session_start(
        instructions=args.instructions, voice=args.voice, model=args.model or LIVE_MODEL,
        event_id=f"toup-experiment-start-{uuid.uuid4().hex[:8]}",
    )
    timeline: list[tuple[int, str, str]] = []
    errors: list[dict] = []
    ours: dict[str, str] = {}            # event id → kind we sent
    acked: dict[str, int] = {}           # event id → ms after t=0
    ack_spans: dict[str, str] = {}       # event id → the ack's start_ms/end_ms
    refused: dict[str, str] = {}         # event id → code
    first_text_ms: Optional[int] = None
    first_audio_ms: Optional[int] = None
    first_voiced_ms: Optional[int] = None    # the first audio delta that is not silence
    last_audio_ms: Optional[int] = None
    first_input_ms: Optional[int] = None
    first_delegation_ms: Optional[int] = None
    words: list[str] = []
    heard: list[str] = []
    audio_bytes = 0
    undecodable = 0
    epochs: list[dict[str, Any]] = []    # Q5: output audio per relay-style epoch
    last_output_at: Optional[float] = None
    usage: Any = None
    sent = {"chunks": 0, "speech": 0}
    stop_audio = asyncio.Event()
    audio_task: Optional[asyncio.Task] = None

    try:
        ws = await websockets.connect(
            LIVE_WS_URL,
            additional_headers={"Authorization": f"Bearer {key}"},
            max_size=10 * 1024 * 1024,
            open_timeout=15,
        )
    except Exception as exc:  # noqa: BLE001 — printed redacted, never raised with the key
        print(f"connect failed: {type(exc).__name__}: {redact(exc)}")
        return 1

    try:
        await ws.send(json.dumps(start_event))
        opened = time.monotonic()
        psid = ""
        while not psid:
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            event = json.loads(raw)
            kind = str(event.get("type") or "")
            if kind == "session.started":
                psid = str((event.get("session") or {}).get("id") or "?")
            elif kind == "error":
                print(f"session.start refused: {redact(json.dumps(event.get('error') or {}))}")
                return 1
            else:
                print(f"  (before session.started) {kind}")
        print(f"session.started psid={psid[:12]} after {_ms(opened)} ms")

        if args.mode == "B":
            audio_task = asyncio.create_task(_send_silence(
                ws, stop_audio, sent, _UPLINK_KEEPER_SILENCE_B64, audio_append,
            ))

        # Lead-in: read (and log) whatever arrives before the append (or the speech).
        lead_until = time.monotonic() + max(0.0, args.lead_in)
        while time.monotonic() < lead_until:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=lead_until - time.monotonic())
            except asyncio.TimeoutError:
                break
            event = json.loads(raw)
            print(f"  (lead-in) {event.get('type')}")

        # Q3: what arrives while the question streams is kept, and handled below
        # once its end — t=0 for everything after — is known.
        backlog: list[tuple[float, dict]] = []
        speech_end: Optional[float] = None
        if speech:
            spoke_at = time.monotonic()
            audio_task = asyncio.create_task(_send_speech(
                ws, stop_audio, sent, speech, args.after, _UPLINK_KEEPER_SILENCE_B64, audio_append,
            ))
            while not audio_task.done() and time.monotonic() < sent.get("speech_end", math.inf):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                backlog.append((time.monotonic(), json.loads(raw)))
            if audio_task.done():
                audio_task.result()          # a sender that died mid-question raises here
            speech_end = sent.get("speech_end") or time.monotonic()
            print(f"speech: {sent['speech']} chunks, {len(speech) * 1000 // PCM_BYTES_PER_S} ms of "
                  f"audio in {int((speech_end - spoke_at) * 1000)} ms of wall time; t=0 is its end")

        appended = time.monotonic()
        if args.kind in ("commentary", "both"):
            event_id = f"toup-experiment-commentary-{uuid.uuid4().hex[:8]}"
            ours[event_id] = "commentary"
            await ws.send(json.dumps(commentary_append(
                args.delegation_id, args.line, event_id=event_id,
            )))
        if args.kind in ("instructions", "both"):
            event_id = f"toup-experiment-instructions-{uuid.uuid4().hex[:8]}"
            ours[event_id] = "instructions"
            await ws.send(json.dumps(instructions_append(
                f"Tell the caller this now: {args.line}", event_id=event_id,
            )))
        t0 = speech_end if speech_end is not None else appended
        print(f"appended {sorted(ours.values()) or 'nothing (--kind none)'} at "
              f"{_at(int((appended - t0) * 1000))} (silence chunks so far: {sent['chunks']})")

        until = t0 + max(1.0, args.seconds)
        while backlog or time.monotonic() < until:
            if backlog:
                arrival, event = backlog.pop(0)
            else:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=until - time.monotonic())
                except asyncio.TimeoutError:
                    break
                arrival, event = time.monotonic(), json.loads(raw)
            kind = str(event.get("type") or "")
            at = int((arrival - t0) * 1000)
            note = ""
            if kind in ("session.output_transcript.delta", "session.output_audio.delta"):
                # Q5: the relay opens an epoch on output after `gap_ms` without any.
                if last_output_at is None or arrival - last_output_at >= gap_ms / 1000:
                    epochs.append({"first": None, "last": None, "deltas": 0, "bytes": 0,
                                   "silent": 0, "zero": 0, "chars": 0})
                last_output_at = arrival
            if kind in ACKS:
                event_id = str(event.get("client_event_id") or "")
                note = _span(event).lstrip()
                if event_id in ours:
                    acked.setdefault(event_id, at)
                    ack_spans.setdefault(event_id, _span(event))
                    note = f"OUR {ours[event_id]} append acked{_span(event)}"
            elif kind == "error":
                error = event.get("error") or {}
                errors.append(error)
                event_id = str(error.get("client_event_id") or "")
                if event_id in ours:
                    refused[event_id] = str(error.get("code") or "?")
                note = redact(json.dumps(error)[:300])
            elif kind == "session.output_transcript.delta":
                if first_text_ms is None:
                    first_text_ms = at
                words.append(str(event.get("delta") or ""))
                epochs[-1]["chars"] += len(words[-1])
                note = repr(words[-1][:80]) + _span(event)
            elif kind == "session.output_audio.delta":
                try:
                    pcm = base64.b64decode(str(event.get("delta") or ""))
                except ValueError:          # binascii.Error: counted apart, never as silence
                    undecodable += 1
                    continue
                zero = not pcm.strip(b"\x00")
                rms = 0.0 if zero else _rms(pcm)
                epoch = epochs[-1]
                if epoch["first"] is None:
                    epoch["first"] = at
                epoch["last"] = last_audio_ms = at
                epoch["deltas"] += 1
                epoch["bytes"] += len(pcm)
                epoch["zero"] += zero
                epoch["silent"] += rms < QUIET_RMS
                if first_audio_ms is None:
                    first_audio_ms = at
                if first_voiced_ms is None and rms >= QUIET_RMS:
                    first_voiced_ms = at
                audio_bytes += len(pcm)
                # Audio is counted, not printed, and one line per delta is noise:
                # one line per run of deltas, and one for each epoch's first.
                if timeline and timeline[-1][1] == kind and epoch["deltas"] > 1:
                    continue
                note = f"epoch {len(epochs)} rms={rms:.0f}"
            elif kind == "session.input_transcript.delta":
                if first_input_ms is None:
                    first_input_ms = at
                heard.append(str(event.get("delta") or ""))
                note = repr(heard[-1][:80]) + _span(event)
            elif kind == "session.delegation.created":
                if first_delegation_ms is None:
                    first_delegation_ms = at
                note = redact(json.dumps(event.get("delegation") or {})[:200])
            elif kind == "session.closed":
                usage = event.get("usage")
                note = f"reason={event.get('reason')}"
            timeline.append((at, kind, note))
            print(redact(f"  {at:>+7} ms  {kind}  {note}"))
            if kind == "session.closed":
                break
    finally:
        stop_audio.set()
        if audio_task is not None:
            await asyncio.gather(audio_task, return_exceptions=True)
        try:
            await ws.send(json.dumps(close_session(f"toup-experiment-close-{uuid.uuid4().hex[:8]}")))
            closing = time.monotonic() + 5.0
            while usage is None and time.monotonic() < closing:
                raw = await asyncio.wait_for(ws.recv(), timeout=closing - time.monotonic())
                event = json.loads(raw)
                if event.get("type") == "session.closed":
                    usage = event.get("usage")
                elif event.get("type") == "error":
                    error = event.get("error") or {}
                    errors.append(error)
                    event_id = str(error.get("client_event_id") or "")
                    if event_id in ours:
                        refused.setdefault(event_id, str(error.get("code") or "?"))
                    print(f"  (at close) error {redact(json.dumps(error)[:300])}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (close) {type(exc).__name__}: {redact(exc)}")
        await ws.close()

    print()
    print("── verdict ─────────────────────────────────────────────")
    if speech:
        print(f"speech: {len(speech) * 1000 // PCM_BYTES_PER_S} ms in {sent['speech']} chunks "
              f"(its own last {_quiet_tail_ms(speech)} ms already quiet), then "
              + ("NOTHING (--after none)" if args.after == "none"
                 else f"{sent['chunks']} x 100 ms of digital silence (--after silence)")
              + " — every time below is ms after the speech ENDED")
    else:
        print(f"mode {args.mode}: input audio sent after session.started = "
              f"{sent['chunks'] * 100} ms ({sent['chunks']} x 100 ms of digital silence)")
    for event_id, what in ours.items():
        if event_id in acked:
            outcome = f"ACKED {_at(acked[event_id])}{ack_spans[event_id]}"
        elif event_id in refused:
            outcome = f"REFUSED code={refused[event_id]}"
        else:
            outcome = "no ack and no refusal within the window"
        print(f"{what:>12} append: {outcome}")
    print(f"first input transcript delta:  {_at(first_input_ms)}")
    if heard:
        print(redact(f"heard: {''.join(heard)[:400]!r}"))
    print(f"first delegation created:      {_at(first_delegation_ms)}")
    print(f"first output transcript delta: {_at(first_text_ms)}")
    print("first output audio delta:      "
          + (f"{_at(first_audio_ms)} ({audio_bytes} bytes)" if first_audio_ms is not None else "none"))
    if words:
        print(redact(f"said: {''.join(words)[:400]!r}"))

    # Q5: the output's pacing, and how much of it is silence.
    deltas = sum(epoch["deltas"] for epoch in epochs)
    if deltas:
        silent = sum(epoch["silent"] for epoch in epochs)
        zero = sum(epoch["zero"] for epoch in epochs)
        audio_s = audio_bytes / PCM_BYTES_PER_S
        wall_s = (last_audio_ms - first_audio_ms) / 1000
        pacing = f"{audio_s / wall_s:.2f}x real time" if wall_s > 0 else "n/a (one delta)"
        print(f"output audio: {audio_s:.2f} s in {deltas} deltas over {wall_s:.2f} s of wall "
              f"time, first to last delta — pacing {pacing}")
        print(f"silent deltas: {silent}/{deltas} = {silent / deltas:.0%} ({zero} all-zero, "
              f"{silent - zero} RMS < {QUIET_RMS}); first voiced delta {_at(first_voiced_ms)}")
    if undecodable:
        print(f"undecodable audio deltas: {undecodable} (not counted above)")
    print(f"output epochs (opened after {gap_ms} ms with no output, as the relay does): {len(epochs)}")
    for n, epoch in enumerate(epochs, 1):
        if not epoch["deltas"]:
            print(f"  epoch {n}: no audio, {epoch['chars']} chars of text")
            continue
        phantom = epoch["silent"] == epoch["deltas"] and not epoch["chars"]
        print(f"  epoch {n}: {epoch['first']:+d}..{epoch['last']:+d} ms  {epoch['deltas']} deltas  "
              f"{epoch['bytes']} bytes ({epoch['bytes'] / PCM_BYTES_PER_S:.2f} s)  silent "
              f"{epoch['silent']}/{epoch['deltas']}  text {epoch['chars']} chars"
              + ("  <- all silent, no text: a PHANTOM epoch in the relay" if phantom else ""))

    if speech:
        trailing = ("with NO trailing input (--after none)" if args.after == "none"
                    else "with trailing real-time silence (--after silence)")
        started = [f"{name} {_at(ms)}" for name, ms in (
            ("delegation", first_delegation_ms),
            ("output transcript", first_text_ms),
            ("voiced audio", first_voiced_ms),
        ) if ms is not None]
        if started:
            print(f"VERDICT Q3: {trailing}, a reply to the SPOKEN turn STARTED within the "
                  f"{args.seconds:g} s window — {', '.join(started)}")
        else:
            print(f"VERDICT Q3: {trailing}, NO reply to the SPOKEN turn started within the "
                  f"{args.seconds:g} s window — no delegation, no output transcript, no voiced audio")
        if ours:
            print("  (an append also went out at the speech's end: the reply may be answering it)")
    if errors:
        print(f"errors: {len(errors)}")
    print(f"usage at close: {redact(json.dumps(usage)) if usage is not None else 'not reported'}")
    return 0


def _pcm_file(path: str) -> bytes:
    """`--speech-file`: the file's bytes, when they can be raw PCM16."""

    try:
        with open(path, "rb") as fh:
            pcm = fh.read()
    except OSError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if not pcm or len(pcm) % 2 or pcm[:4] == b"RIFF":
        raise argparse.ArgumentTypeError(
            f"{path}: want raw PCM16 mono 24 kHz little-endian with no header "
            "(scripts/make_speech_pcm.sh makes one)"
        )
    return pcm


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.live_append_experiment",
        description="Does GPT-Live take an append with no input audio (A) / with real-time silence (B)? "
                    "Does it answer a spoken question that no input follows (--speech-file)?",
    )
    stimulus = parser.add_mutually_exclusive_group(required=True)
    stimulus.add_argument("--mode", choices=("A", "B"),
                          help="A: no input audio; B: real-time PCM16 24 kHz silence")
    stimulus.add_argument("--speech-file", dest="speech", type=_pcm_file, metavar="PCM",
                          help="Q3: stream this raw PCM16 mono 24 kHz LE question in real time "
                               "after the lead-in; times are then ms after it ENDS")
    parser.add_argument("--after", choices=("none", "silence"), default="none",
                        help="with --speech-file: none = send nothing more (default); silence = "
                             "the keeper's 100 ms chunk every 100 ms until the window closes")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to observe after the append, or the speech (default 20)")
    parser.add_argument("--lead-in", type=float, default=1.0,
                        help="seconds between session.started and the append, or the speech (default 1)")
    parser.add_argument("--kind", choices=("commentary", "instructions", "both", "none"),
                        default="commentary",
                        help="which append to send (default commentary; none = no append)")
    parser.add_argument("--line", default=DEFAULT_LINE, help="the text to append")
    parser.add_argument("--delegation-id", default="toup-experiment",
                        help="delegation id for --kind commentary (synthetic by default)")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS,
                        help="session instructions for session.start")
    parser.add_argument("--voice", default="marin")
    parser.add_argument("--model", default="", help="Live model (default: the relay's LIVE_MODEL)")
    args = parser.parse_args(argv)
    if args.after == "silence" and args.speech is None:
        parser.error("--after applies only with --speech-file")
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
