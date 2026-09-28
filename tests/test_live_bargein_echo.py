"""The agent must not interrupt itself with its own voice (R13).

The production regression this file exists for: on build 130 + relay d3ebda17
the agent started every spoken reply and was cut off 1.8-3.4 s in, every time.
Build 130 keeps the mic hot while the agent speaks on Live, residual echo of the
agent's voice reached GPT-Live and was transcribed as INPUT, and the relay
counted any two input words during an output epoch as "the user is talking over
the agent" — so it synthesized `speech_started`, the client obeyed it with an
`interrupt`, and the relay fenced the reply off. `assistant=0 chars` in both
sessions.

`Build130Client` below models the shipped client's half of that chain exactly:
it answers every `speech_started` with an identity-less `interrupt`, consulting
nothing. A scenario that sends it a `speech_started` therefore reproduces the
cut, not merely a frame.

Letters in the test names are R13.3.6's scenario letters.

NOT EXERCISABLE HERE (owner device test only): how much echo the phone's AEC
really leaves on each route, what GPT-Live's ASR makes of it, and whether any
audio reaches the caller's ear. The fake provider emits the documented events;
the words and timings are modelled on the two production calls, whose
transcripts are — correctly — not in any log.
"""

import asyncio
import json
import logging
import re

import pytest

import test_live_harness as H
from app.agent.media_intent import normalize_fa
from app.services.live_voice_protocol import (
    LIVE_STATES,
    LiveClientAdapter,
    _echo_reference_tokens,
    echo_overlap,
)


DB = "db-r13"


def _oid(epoch: int) -> str:
    return f"live:{H.PSID}:{epoch}"


class Build130Client(H.FakeClient):
    """TestFlight build 130: `speech_started` → cut playback + `interrupt`,
    unconditionally. The frame carries no identity (A2-05)."""

    def __init__(self, script=None, interrupt_frame=None):
        super().__init__(script)
        self.injected: asyncio.Queue = asyncio.Queue()
        self.interrupt_frame = interrupt_frame or {"type": "interrupt"}

    async def send_json(self, frame):
        await super().send_json(frame)
        if frame.get("type") == "speech_started":
            self.injected.put_nowait(dict(self.interrupt_frame))

    async def receive_text(self):
        loop = asyncio.get_running_loop()
        while True:
            if not self.injected.empty():
                return json.dumps(self.injected.get_nowait())
            if not self._script:
                return json.dumps(await self.injected.get())
            item = self._script.pop(0)
            if isinstance(item, (int, float)):
                # Sleep in slices so an obeyed `speech_started` goes out at
                # once, as it does on the phone, and the rest of the pause is
                # kept rather than lost.
                deadline = loop.time() + item
                while loop.time() < deadline and self.injected.empty():
                    await asyncio.sleep(0.005)
                remaining = deadline - loop.time()
                if remaining > 0:
                    self._script.insert(0, remaining)
                continue
            if callable(item):
                item()
                continue
            return json.dumps(item)


def _idle(epoch: int, played_ms: int) -> dict:
    return {
        "type": "playback_idle", "response_id": _oid(epoch), "item_id": _oid(epoch),
        "played_ms": played_ms,
    }


def _stop_instructions(provider) -> list[dict]:
    return [
        e for e in provider.of("session.instructions.append")
        if "Stop speaking now" in str(e.get("content") or "")
    ]


def _spoken_rows(saves) -> list[dict]:
    return [s for s in saves if str(s.get("assistant_ref") or "").startswith("live-output:")]


def _delegated_rows(saves) -> list[dict]:
    return [s for s in saves if str(s.get("assistant_ref") or "").startswith("live-delegation:")]


def _speak(provider, chunks, *, echo=(), start_ms=0, step_ms=450, lag_chunks=1):
    """Push one reply as the provider does — transcript + audio per chunk —
    with `echo` (input deltas) arriving `lag_chunks` behind the chunk they
    repeat, which is how echo trails the audio it came from."""

    pending = list(echo)
    t = start_ms
    for index, chunk in enumerate(chunks):
        provider.push(H.out_text(chunk, t, t + step_ms))
        provider.push(H.out_audio())
        if index >= lag_chunks and pending:
            text, s_ms, e_ms = pending.pop(0)
            provider.push(H.user_delta(text, s_ms, e_ms))
        t += step_ms
    for text, s_ms, e_ms in pending:
        provider.push(H.user_delta(text, s_ms, e_ms))


# The reply and its echo, modelled on the owner's calls: a sentence-long
# answer, then input-transcript deltas re-voicing it a chunk behind, each a
# few words over ~0.4-0.6 s. The first echo delta lands ~0.9 s in and the old
# rule fired on it; by played_ms 1850-3400 the client had cut playback.
REPLY = [
    "Sure, I can help with that. ",
    "The weather in Toronto today ",
    "is mostly sunny, with a high ",
    "of twenty two degrees ",
    "and a light breeze this evening.",
]
REPLY_TEXT = "".join(REPLY).strip()
ECHO = [
    ("Sure I can help", 900, 1400),
    ("with that the weather", 1400, 1950),
    ("in Toronto today", 1950, 2400),
    ("is mostly sunny with", 2400, 2900),
    ("a high of twenty two", 2900, 3400),
]


def _repro_on_send(provider, event):
    if event["type"] == "session.start":
        provider.push(H.user_delta("what's the weather like", 0, 700))
        _speak(provider, REPLY, echo=ECHO, start_ms=800)
    elif event["type"] == "session.close":
        provider.push(H.closed())


# ── (a) the observed calls ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_echo_of_the_agents_own_reply_no_longer_cuts_it_off(monkeypatch):
    """(a) REPRO. Against relay d3ebda17 this sends `speech_started` on the
    first echo delta, the build-130 client obeys it, and the reply is fenced
    and persisted `interrupted` — the production symptom. Now the reply
    finishes and persists whole."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    counters = H.capture_counters(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    client = Build130Client([
        H.config(), 0.4, _idle(1, 4200), 0.3, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_repro_on_send)
    await H.run_relay(client, provider, timeout=6, db_session_id=DB)

    assert client.of("speech_started") == [], (
        "the agent's own words, heard back through the speaker, are not the "
        "caller interrupting"
    )
    assert _stop_instructions(provider) == []
    segments = client.of("speech_segment_complete")
    assert len(segments) == 1
    assert segments[0]["interrupted"] is False
    assert segments[0]["text"] == REPLY_TEXT
    rows = _spoken_rows(recorded["saves"])
    assert rows and rows[-1]["assistant_text"] == REPLY_TEXT
    assert rows[-1]["assistant_voice"]["interrupted"] is False
    suppressed = [n for n, _ in counters if n == "live_bargein_echo_suppressed"]
    assert len(suppressed) == len(ECHO)
    assert "live_bargein" not in [n for n, _ in counters]
    # R13.3.4: the input transcript is still recorded, whatever barge-in
    # decided. (Echo reaching the user turn is R13.5 — recorded, not solved.)
    # The question asked BEFORE the reply would satisfy a bare "some user
    # text", so this names words that only the ECHO carries.
    user_text = " ".join(str(s.get("user_text") or "") for s in recorded["saves"])
    for word in ("Toronto", "sunny", "twenty"):
        assert word in user_text, (word, "echo must still reach the user turn")


@pytest.mark.asyncio
async def test_a_the_echo_verdict_never_puts_words_in_a_log(monkeypatch, caplog):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    client = Build130Client([H.config(), 0.4, {"type": "stop"}])
    with caplog.at_level(logging.DEBUG):
        await H.run_relay(
            client, H.FakeProvider(on_send=_repro_on_send), timeout=6, db_session_id=DB,
        )
    lines = [r.getMessage() for r in caplog.records]
    suppressed = [line for line in lines if "barge-in echo" in line]
    assert suppressed, lines
    # The line's whole shape is pinned: any added field is a place words
    # could go.
    for line in suppressed:
        assert re.fullmatch(
            r"\[LIVE\] barge-in echo (suppressed|vetoed) epoch=\d+ ratio=[0-9.]+ total=\d+",
            line,
        ), line
    # And no record anywhere carries a content word of the reply or its echo.
    words = set()
    for text in REPLY + [e[0] for e in ECHO] + ["what's the weather like"]:
        words.update(w for w in re.findall(r"[A-Za-z]+", text.casefold()) if len(w) >= 4)
    for line in lines:
        found = {w for w in words if re.search(rf"\b{w}\b", line.casefold())}
        assert not found, (found, line)


# ── (b) silence ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_b_a_reply_over_silence_plays_to_the_end(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("hello there", 0, 500))
            # What an ASR emits for a quiet room: whitespace and punctuation.
            _speak(provider, REPLY, echo=[(" ", 1200, 1300), ("...", 1300, 1900),
                                          (",", 2000, 2800)], start_ms=600)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(), 0.4, _idle(1, 4000), 0.3, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert client.of("speech_started") == []
    rows = _spoken_rows(recorded["saves"])
    assert rows and rows[-1]["assistant_text"] == REPLY_TEXT
    assert rows[-1]["assistant_voice"]["interrupted"] is False


# ── (c) exact echo ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c_an_exact_echo_is_suppressed_and_counted(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    counters = H.capture_counters(monkeypatch)
    H.patch_relay(monkeypatch)
    exact = [(chunk, 900 + i * 450, 1300 + i * 450) for i, chunk in enumerate(REPLY)]

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, REPLY, echo=exact, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(), 0.4, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert client.of("speech_started") == []
    assert [n for n, _ in counters].count("live_bargein_echo_suppressed") == len(exact)


# ── (d) fuzzy / partial ASR echo ──────────────────────────────────────

@pytest.mark.asyncio
async def test_d_a_misheard_or_truncated_echo_is_still_echo(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    # A voice through a phone speaker, re-heard: respellings, dropped letters,
    # a word cut off at the delta boundary.
    fuzzy = [
        ("shure I can halp with", 900, 1500),
        ("the whether in Toront", 1500, 2100),
        ("tooday its mostly sunnny", 2100, 2700),
        ("with a hi of twenty two", 2700, 3300),
    ]

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, REPLY, echo=fuzzy, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(), 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6, db_session_id=DB)

    assert client.of("speech_started") == []
    assert _stop_instructions(provider) == []


# ── (e) echo lagging into the next epoch ──────────────────────────────

@pytest.mark.asyncio
async def test_e_the_tail_of_one_reply_echoing_into_the_next_is_still_echo(monkeypatch):
    """The 700 ms gap rotation retires an epoch while the phone is still
    playing its buffered audio, so the echo of reply one arrives while reply
    two is open. Only the RETAINED previous-epoch text recognises it."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=80)
    H.patch_relay(monkeypatch)
    first = ["Your meeting with Priya ", "moved to Thursday afternoon ", "at four o'clock."]
    second = ["Should I ", "add a reminder?"]
    holder = {}

    def on_send(provider, event):
        holder["p"] = provider
        if event["type"] == "session.start":
            _speak(provider, first, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def reply_two():
        provider = holder["p"]
        provider.push(H.out_text(second[0], 2000, 2300))
        provider.push(H.out_audio())
        provider.push(H.user_delta("meeting with Priya moved", 2100, 2600))
        provider.push(H.user_delta("to Thursday afternoon", 2600, 3100))
        provider.push(H.out_text(second[1], 2300, 2800))
        provider.push(H.user_delta("at four o'clock", 3100, 3600))

    client = Build130Client([H.config(), 0.3, reply_two, 0.3, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    epochs = {f["response_id"] for f in client.of("audio_delta")}
    assert epochs == {_oid(1), _oid(2)}, "the echo must land in a SECOND epoch"
    assert client.of("speech_started") == []


# ── (f) a genuine interruption ────────────────────────────────────────

@pytest.mark.asyncio
async def test_f_a_real_interruption_fires_exactly_once_and_stops_the_agent(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    counters = H.capture_counters(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    words = [
        ("no stop that", 1200, 1600),
        ("is not what", 1600, 2000),
        ("I asked about", 2000, 2400),
    ]

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, REPLY, echo=words, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(), 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6, db_session_id=DB)

    started = client.of("speech_started")
    assert len(started) == 1 and started[0]["source"] == "transcript"
    assert len(_stop_instructions(provider)) == 1, "the obeyed interrupt stops the model"
    assert "live_bargein_echo_suppressed" not in [n for n, _ in counters]
    rows = _spoken_rows(recorded["saves"])
    assert rows and rows[-1]["assistant_voice"]["interrupted"] is True


CONTRACTION_REPLY = [
    "I don't think it's going ",
    "to rain today, so you ",
    "won't need an umbrella.",
]


@pytest.mark.asyncio
async def test_f_a_real_interruption_with_contractions_is_not_echo(monkeypatch):
    """An apostrophe used to become a space, so "don't" voted as "don" + "t"
    — two tokens carrying no identity that matched any reply containing a
    contraction. "no, don't do that" then read as the agent's own words."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    words = [
        ("no, don't do that", 1200, 1600),
        ("I don't care", 1600, 2000),
        ("stop, it's fine", 2000, 2400),
    ]

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, CONTRACTION_REPLY, echo=words, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(), 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6, db_session_id=DB)

    assert len(client.of("speech_started")) == 1


# ── (g) Persian ───────────────────────────────────────────────────────

FA_REPLY = ["یکی از کیک‌ها ", "۲۲ هزار تومان ", "است."]


@pytest.mark.asyncio
async def test_g_persian_echo_in_other_letter_forms_is_suppressed(monkeypatch):
    """The ASR's Persian is not byte-identical to the model's: Arabic ي/ك for
    ی/ک, Arabic-Indic digits for Persian ones, a space for the ZWNJ. Only the
    `normalize_fa` fold makes these the same words."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    echo = [
        ("يكي از كيك", 700, 1100),
        ("كيك ٢٢", 1100, 1500),
        ("يكي ٢٢ تا", 1500, 1900),
    ]

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, FA_REPLY, echo=echo, start_ms=0, lag_chunks=1)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(language="fa"), 0.4, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert client.of("speech_started") == []


@pytest.mark.asyncio
async def test_g_a_genuine_persian_interruption_still_fires(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    words = [("نه صبر کن", 700, 1100), ("اینو نپرسیدم", 1100, 1500)]

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, FA_REPLY, echo=words, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(language="fa"), 0.4, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert len(client.of("speech_started")) == 1


# The provider's input deltas are FRAGMENTS appended exactly, and nothing
# bounds how fine they are. These are GPT-4o-family (o200k_base) tokens, the
# granularity at which a Persian word breaks into pieces too short to match
# anything on their own ("هو", "فت", "رس"). Hard-coded so the test needs no
# tokenizer download.
FA_WEATHER_REPLY = [
    "هوای تهران امروز ", "آفتابی است و دمای هوا ", "به بیست و پنج درجه ", "می‌رسد.",
]
FA_WEATHER_ECHO_TOKENS = [
    "هو", "ای", " تهران", " امروز", " آ", "فت", "ابی", " است", " و", " د", "مای",
    " هوا", " به", " بی", "ست", " و", " پنج", " درجه", " می", "رس", "د", ".",
]
FA_GENUINE_TOKENS = [
    "نه", " ص", "بر", " کن", "،", " این", "و", " ن", "پ", "رس", "ید", "م", ".",
    " من", " ساعت", " ده", " رو", " گ", "ف", "تم",
]
EN_GENUINE_TOKENS = [
    "no", " wait", ",", " that", " is", " not", " what", " I", " asked", ",",
    " Pri", "y", "anka", " said", " Friday",
]


def _timed(fragments, *, start_ms, ms_per_char):
    out, t = [], start_ms
    for fragment in fragments:
        span = max(1, len(fragment)) * ms_per_char
        out.append((fragment, t, t + span))
        t += span
    return out


def _words(fragments):
    """The same text re-split at word boundaries, one word per delta."""

    return [(" " if i else "") + w for i, w in enumerate("".join(fragments).split())]


async def _fa_weather_run(monkeypatch, fragments, ms_per_char):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    echo = _timed(fragments, start_ms=900, ms_per_char=ms_per_char)

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, FA_WEATHER_REPLY, echo=echo, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(language="fa"), 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6, db_session_id=DB)
    return client, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("ms_per_char", [40, 65, 100, 150])
async def test_g_a_persian_echo_in_token_level_deltas_is_still_echo(monkeypatch, ms_per_char):
    """The verdict must not depend on how finely the provider splits its
    deltas. Judged fragment by fragment, "هو" / "فت" / "رس" are too short to
    match and each counted as a NON-echo word, so a pure echo of an ordinary
    Persian sentence fired barge-in at every timing."""

    client, provider = await _fa_weather_run(monkeypatch, FA_WEATHER_ECHO_TOKENS, ms_per_char)
    assert client.of("speech_started") == []
    assert _stop_instructions(provider) == []


@pytest.mark.asyncio
async def test_g_a_window_veto_is_counted_and_logged_without_words(monkeypatch, caplog):
    # Every fragment is too short to be judged echo on its own, so the only
    # suppression here is the window veto — and it must be as visible in the
    # counters as a per-delta one.
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    counters = H.capture_counters(monkeypatch)
    H.patch_relay(monkeypatch)
    fragments = _timed(["هو", "ای", " آ", "فت", "ابی"], start_ms=900, ms_per_char=150)

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, ["هوای ", "آفتابی."], echo=fragments, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(language="fa"), 0.4, {"type": "stop"}])
    # DEBUG, and every record of the relay's logger — not only the veto line:
    # a new record anywhere on this path must not carry the words either.
    with caplog.at_level(logging.DEBUG), caplog.at_level(
        logging.DEBUG, logger="app.services.live_voice_protocol",
    ):
        await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert client.of("speech_started") == []
    relay_lines = [
        normalize_fa(r.getMessage()) for r in caplog.records
        if r.name == "app.services.live_voice_protocol"
    ]
    assert relay_lines
    words = {normalize_fa(w) for w in ["هو", "ای", "آ", "فت", "ابی", "هوای", "آفتابی"]}
    leaked = [line for line in relay_lines if any(w in line for w in words)]
    assert leaked == [], "Persian text reached a log record"
    vetoed = [r.getMessage() for r in caplog.records if "barge-in echo vetoed" in r.getMessage()]
    assert vetoed and all(
        re.fullmatch(r"\[LIVE\] barge-in echo vetoed epoch=\d+ ratio=[0-9.]+ total=\d+", v)
        for v in vetoed
    ), vetoed
    names = [n for n, _ in counters]
    assert names.count("live_bargein_echo_suppressed") == len(vetoed)
    assert "live_bargein" not in names


@pytest.mark.asyncio
@pytest.mark.parametrize("ms_per_char", [40, 150])
async def test_g_the_same_persian_echo_in_word_level_deltas_is_echo(monkeypatch, ms_per_char):
    client, _provider = await _fa_weather_run(
        monkeypatch, _words(FA_WEATHER_ECHO_TOKENS), ms_per_char,
    )
    assert client.of("speech_started") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("granularity", ["token", "word"])
@pytest.mark.parametrize("language,fragments,reply", [
    ("fa", FA_GENUINE_TOKENS, FA_WEATHER_REPLY),
    ("en", EN_GENUINE_TOKENS, REPLY),
])
async def test_g_a_genuine_interruption_fires_at_any_delta_granularity(
    monkeypatch, granularity, language, fragments, reply,
):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    split = fragments if granularity == "token" else _words(fragments)
    words = _timed(split, start_ms=900, ms_per_char=65)

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, reply, echo=words, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(language=language), 0.4, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert len(client.of("speech_started")) == 1


# A reply dense in present-tense verbs: every one is a می‌/نمی‌ compound, and
# the model's transcript and the ASR's need not spell them the same way —
# ZWNJ, a space, or joined. `normalize_fa` makes the ZWNJ a space, so "می‌رسد"
# was "می" + "رسد", both under the fuzzy floor, and the ASR's joined "میرسد"
# matched neither: three such verbs in a window and pure echo fired.
ZWNJ = "\u200c"
FA_VERB_REPLY_TEXT = (
    f"من نمی{ZWNJ}دانم کی می{ZWNJ}رسد، ولی فکر می{ZWNJ}کنم فردا صبح می{ZWNJ}رسند"
    f" و بعد می{ZWNJ}روند."
)
FA_COLLOQUIAL_REPLY_TEXT = f"نمی{ZWNJ}دونم دقیقاً کی می{ZWNJ}رسه، ولی فکر می{ZWNJ}کنم تا عصر برسه."
# o200k_base tokens of the JOINED spelling, hard-coded (no tokenizer download).
FA_VERB_JOINED_TOKENS = [
    "من", " نم", "ید", "ان", "م", " کی", " می", "رس", "د", "،", " ولی", " فکر",
    " میکن", "م", " فرد", "ا", " صبح", " می", "رس", "ند", " و", " بعد", " میر",
    "وند", ".",
]
# ...and of the ZWNJ spelling, which the ASR may emit against a joined reply.
FA_VERB_ZWNJ_TOKENS = [
    "من", " نمی", ZWNJ, "دان", "م", " کی", " می", ZWNJ, "رس", "د", "،", " ولی",
    " فکر", " می", ZWNJ + "کن", "م", " فرد", "ا", " صبح", " می", ZWNJ, "رس", "ند",
    " و", " بعد", " می", ZWNJ + "ر", "وند", ".",
]
FA_COLLOQUIAL_JOINED_TOKENS = [
    "نم", "ید", "ون", "م", " دقیق", "اً", " کی", " می", "رس", "ه", "،", " ولی",
    " فکر", " میکن", "م", " تا", " عصر", " بر", "سه", ".",
]


def _chunks(text, n=4):
    words = text.split(" ")
    size = -(-len(words) // n)
    return [" ".join(words[i:i + size]) + " " for i in range(0, len(words), size)]


async def _fa_reply_run(monkeypatch, reply_text, fragments, ms_per_char):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    echo = _timed(fragments, start_ms=900, ms_per_char=ms_per_char)

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, _chunks(reply_text), echo=echo, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(language="fa"), 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6, db_session_id=DB)
    return client, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("ms_per_char", [40, 65, 100, 150])
@pytest.mark.parametrize("granularity", ["token", "word"])
@pytest.mark.parametrize("reply_text,fragments", [
    # the reply keeps its ZWNJs, the ASR joins them
    (FA_VERB_REPLY_TEXT, FA_VERB_JOINED_TOKENS),
    # the reverse: the reply joined, the ASR writes the ZWNJ
    (FA_VERB_REPLY_TEXT.replace(ZWNJ, ""), FA_VERB_ZWNJ_TOKENS),
    # ...or a space
    (FA_VERB_REPLY_TEXT.replace(ZWNJ, ""), [f.replace(ZWNJ, " ") for f in FA_VERB_ZWNJ_TOKENS]),
    # the reply written with spaces, the ASR joins
    (FA_VERB_REPLY_TEXT.replace(ZWNJ, " "), FA_VERB_JOINED_TOKENS),
    (FA_COLLOQUIAL_REPLY_TEXT, FA_COLLOQUIAL_JOINED_TOKENS),
], ids=["reply-zwnj/echo-joined", "reply-joined/echo-zwnj", "reply-joined/echo-space",
        "reply-space/echo-joined", "colloquial/echo-joined"])
async def test_g_a_persian_echo_is_echo_however_its_compounds_are_spelled(
    monkeypatch, reply_text, fragments, granularity, ms_per_char,
):
    split = fragments if granularity == "token" else _words(fragments)
    client, provider = await _fa_reply_run(monkeypatch, reply_text, split, ms_per_char)
    assert client.of("speech_started") == []
    assert _stop_instructions(provider) == []


# The model's transcript spells the می/نمی compounds JOINED and the ASR hands
# the echo back SPACED, at real o200k granularity: " می", " رس", "د". A window
# veto spent on a window ending in " می" left a fresh one holding only "رسد" —
# no neighbour to join, too short to fuzz — while "رس" and "د" counted as TWO
# content words, so " و" padded the run and pure echo fired. The older
# "reply-joined/echo-space" case above splits inside ZWNJ tokenization
# (" می", " ", "رس") and never produced that shape. Hard-coded o200k_base
# tokens of the SPACED spelling.
FA_SPACED_ECHOES = {
    "formal-verbs": (
        f"من نمی{ZWNJ}دانم که او کی می{ZWNJ}آید، اما فکر می{ZWNJ}کنم امروز عصر"
        f" می{ZWNJ}رسد و شب برمی{ZWNJ}گردد.",
        ["من", " نمی", " دان", "م", " که", " او", " کی", " می", " آ", "ید", "،",
         " اما", " فکر", " می", " کنم", " امروز", " عصر", " می", " رس", "د", " و",
         " شب", " بر", "می", " گردد", "."],
    ),
    "formal-result": (
        f"نتیجه آماده است: قیمت{ZWNJ}ها از فردا تغییر می{ZWNJ}کنند و سفارش شما هفته"
        f" آینده می{ZWNJ}رسد و ما به شما خبر می{ZWNJ}دهیم.",
        ["نت", "ی", "جه", " آماده", " است", ":", " قیمت", " ها", " از", " فرد", "ا",
         " تغییر", " می", " کنند", " و", " سفارش", " شما", " هفته", " آینده", " می",
         " رس", "د", " و", " ما", " به", " شما", " خبر", " می", " ده", "یم", "."],
    ),
    "colloquial": (
        f"نمی{ZWNJ}دونم کی می{ZWNJ}رسه، ولی فکر می{ZWNJ}کنم عصر می{ZWNJ}رسه و بعدش"
        f" می{ZWNJ}ره خونه.",
        ["ن", "می", " دون", "م", " کی", " می", " رس", "ه", "،", " ولی", " فکر",
         " می", " کنم", " عصر", " می", " رس", "ه", " و", " بعد", "ش", " می", " ره",
         " خ", "ونه", "."],
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("ms_per_char", [65, 100, 150])
@pytest.mark.parametrize("name", sorted(FA_SPACED_ECHOES))
async def test_g_a_spaced_echo_of_a_joined_persian_reply_is_echo(monkeypatch, name, ms_per_char):
    reply_text, fragments = FA_SPACED_ECHOES[name]
    client, provider = await _fa_reply_run(
        monkeypatch, reply_text.replace(ZWNJ, ""), fragments, ms_per_char,
    )
    assert client.of("speech_started") == []
    assert _stop_instructions(provider) == []


def test_a_compound_half_left_alone_by_a_veto_is_not_two_words():
    """The adapter-level shape of the case above: a window holding only the
    second half of a compound ("رس" + "د") and a function word is ONE word the
    agent did not recognisably say — it waits for more, it does not fire."""

    reply_text, fragments = FA_SPACED_ECHOES["formal-verbs"]
    adapter = LiveClientAdapter("spaced-echo")
    adapter.provider_event(
        {"type": "session.output_transcript.delta", "delta": reply_text.replace(ZWNJ, "")},
        now=0.0,
    )
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=0.0)
    assert _feed(adapter, _timed(fragments, start_ms=900, ms_per_char=100)) is False
    assert adapter.echo_vetoed >= 1


def test_the_overlap_joins_a_split_persian_compound_both_ways():
    ref = frozenset(_echo_reference_tokens(FA_VERB_REPLY_TEXT))
    # the ASR joined what the model spelled with a ZWNJ
    assert echo_overlap("نمیدانم میرسد میکنم", ref) == (3, 1.0)
    joined = frozenset(_echo_reference_tokens(FA_VERB_REPLY_TEXT.replace(ZWNJ, "")))
    # the ASR split what the model joined
    assert echo_overlap(f"نمی{ZWNJ}دانم می رسد", joined) == (4, 1.0)
    # the model wrote a SPACE where the ASR joined — the reference must carry
    # the joined spelling itself, since the transcript side has nothing to join
    spaced = frozenset(_echo_reference_tokens(FA_VERB_REPLY_TEXT.replace(ZWNJ, " ")))
    assert echo_overlap("نمیدانم میرسد", spaced) == (2, 1.0)
    # a two-ZWNJ compound, joined by the ASR
    assert echo_overlap(
        "کتابهایمان", frozenset(_echo_reference_tokens(f"کتاب{ZWNJ}های{ZWNJ}مان را آوردم")),
    ) == (1, 1.0)
    # English is never joined: "any" + "one" must not become the agent's "anyone"
    assert echo_overlap("any one", frozenset({"anyone"})) == (2, 0.0)


# Short interruptions built mostly from function words. As one phrase delta
# they always fired; split into words, each bare "that" / "is" / "it" was
# dropped as "evidence of nothing", so the same sentence never reached three
# tokens — whether the caller could interrupt depended on delta granularity.
SHORT_GENUINE = [
    ("en", "no that is not it", REPLY),
    ("en", "stop it you are wrong", REPLY),
    # "can" alone is echo of the reply's "I can help", and must not sink it
    ("en", "can you stop that now", REPLY),
    ("fa", "نه این نیست", FA_WEATHER_REPLY),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("granularity", ["phrase", "word"])
@pytest.mark.parametrize("language,sentence,reply", SHORT_GENUINE,
                         ids=[f"{lang}-{i}" for i, (lang, _s, _r) in enumerate(SHORT_GENUINE)])
async def test_f_a_short_interruption_fires_whatever_the_delta_granularity(
    monkeypatch, granularity, language, sentence, reply,
):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    words = sentence.split()
    split = (
        [(" " if i else "") + " ".join(words[i:i + 2]) for i in range(0, len(words), 2)]
        if granularity == "phrase"
        else _words([sentence])
    )
    deltas = _timed(split, start_ms=900, ms_per_char=75)

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, reply, echo=deltas, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(language=language), 0.4, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert len(client.of("speech_started")) == 1


@pytest.mark.parametrize("sentence", ["helb with that the", "wether is in the"])
def test_function_words_never_carry_an_echo_over_the_line(sentence):
    """The counterweight to the rule above: function words are most of any
    echoed sentence, so they may pad a run but never make one. One misheard
    echo word followed by the agent's own "with that the" is not an
    interruption, even word by word."""

    adapter = LiveClientAdapter("session-pad")
    adapter.provider_event(
        {"type": "session.output_transcript.delta", "delta": REPLY_TEXT}, now=0.0,
    )
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=0.0)
    fired = False
    for text, s_ms, e_ms in _timed(_words([sentence]), start_ms=900, ms_per_char=75):
        adapter.note_user_input(e_ms, speech_ms=e_ms - s_ms, text=text)
        fired = adapter.bargein_due(600, 3, now=1.0, rearm_ms=1500) or fired
    assert fired is False


def test_one_delta_at_the_echo_boundary_is_echo_counted_and_adds_nothing():
    """R13.3.1 per delta: overlap >= voice_live_echo_overlap is ECHO. Pinned on
    its own, because the fire-time window veto shares the 0.6 and would
    otherwise mask a broken per-delta rule."""

    adapter = LiveClientAdapter("session-boundary")
    adapter.provider_event(
        {"type": "session.output_transcript.delta", "delta": REPLY_TEXT}, now=0.0,
    )
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=0.0)
    ratio = adapter.note_user_input(1500, speech_ms=900, text="Toronto sunny tomorrow")
    assert ratio == pytest.approx(2 / 3)
    assert adapter.echo_suppressed == 1
    # ...and it is not evidence: had it counted, this one genuine delta after
    # it would complete a two-delta, three-token, 600 ms run.
    adapter.note_user_input(1900, speech_ms=400, text="stop please")
    assert adapter.bargein_due(600, 3, now=1.0, rearm_ms=1500) is False
    # Control: the same two deltas with the first below the boundary do fire.
    control = LiveClientAdapter("session-boundary-control")
    control.provider_event(
        {"type": "session.output_transcript.delta", "delta": REPLY_TEXT}, now=0.0,
    )
    control.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=0.0)
    assert control.note_user_input(1500, speech_ms=900, text="Toronto rainy yesterday") is None
    control.note_user_input(1900, speech_ms=400, text="stop please")
    assert control.bargein_due(600, 3, now=1.0, rearm_ms=1500) is True


def test_a_window_vetoed_as_echo_is_spent_not_kept():
    """Kept, a vetoed window would only grow with echo and outvote whatever
    the caller says next — so genuine speech after it must still fire."""

    adapter = LiveClientAdapter("session-veto")
    adapter.provider_event(
        {"type": "session.output_transcript.delta", "delta": "".join(FA_WEATHER_REPLY)},
        now=0.0,
    )
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=0.0)
    fired = []
    for fragment, s_ms, e_ms in _timed(FA_WEATHER_ECHO_TOKENS, start_ms=900, ms_per_char=65):
        adapter.note_user_input(e_ms, speech_ms=e_ms - s_ms, text=fragment)
        fired.append(adapter.bargein_due(600, 3, now=1.0, rearm_ms=1500))
    assert not any(fired)
    assert adapter.echo_vetoed >= 1
    t = 900 + 65 * len("".join(FA_WEATHER_ECHO_TOKENS)) + 22 * 65
    for fragment, s_ms, e_ms in _timed(FA_GENUINE_TOKENS, start_ms=t, ms_per_char=65):
        adapter.note_user_input(e_ms, speech_ms=e_ms - s_ms, text=fragment)
        fired.append(adapter.bargein_due(600, 3, now=1.0, rearm_ms=1500))
    assert fired.count(True) == 1


def _fa_adapter(name):
    adapter = LiveClientAdapter(name)
    adapter.provider_event(
        {"type": "session.output_transcript.delta", "delta": "".join(FA_WEATHER_REPLY)},
        now=0.0,
    )
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=0.0)
    return adapter


def _feed(adapter, deltas):
    """(text, start_ms, end_ms) in order; True if any delta made a shot due."""

    fired = False
    for text, s_ms, e_ms in deltas:
        adapter.note_user_input(e_ms, speech_ms=e_ms - s_ms, text=text)
        fired = adapter.bargein_due(600, 3, now=1.0, rearm_ms=1500) or fired
    return fired


def test_the_window_includes_deltas_judged_echo_on_their_own():
    # The middle delta is echo alone and adds no evidence, but it still sits
    # BETWEEN the other two: joined exactly, the last fragment is the start
    # of "دمای", not a word of the caller's.
    adapter = _fa_adapter("window-echo-inside")
    assert _feed(adapter, [
        ("درجه آ", 900, 1300), ("فتابی تهران امروز", 1300, 1900), ("دم", 1900, 2300),
    ]) is False
    assert adapter.echo_vetoed == 1


def test_the_window_starts_at_the_start_of_the_word():
    # The first evidence fragment "بی" continues "آفتا", which arrived in an
    # echo delta: the window is judged from "آفتابی", not from "بی".
    adapter = _fa_adapter("window-word-start")
    assert _feed(adapter, [
        (" تهران امروز آفتا", 900, 1500), ("بی", 1500, 1800), (" د", 1800, 2000),
        ("مای", 2000, 2300),
    ]) is False
    assert adapter.echo_vetoed == 1


def test_a_word_head_does_not_survive_a_gap_in_speech():
    # Three seconds later "بی" begins a NEW word; carrying the stale "آفتا"
    # would make the caller's "بیمزه" read as the agent's "آفتابی".
    adapter = _fa_adapter("window-gap")
    assert _feed(adapter, [
        (" تهران امروز آفتا", 900, 1500),
        ("بی", 4500, 4800), ("مز", 4800, 5100), ("ه", 5100, 5400),
        # A second word: one word split into three tokens is one word, and
        # one word is not an interruption (see the spaced-echo case above).
        (" چرا", 5400, 5700),
    ]) is True
    assert adapter.echo_vetoed == 0


def test_a_word_split_across_an_epoch_rotation_is_judged_whole():
    # Echo lags its audio, so the tail of reply one can still be arriving —
    # mid-word — after the gap rotation has opened reply two.
    adapter = _fa_adapter("window-epoch")
    assert _feed(adapter, [(" تهران امروز آفتا", 900, 1500)]) is False
    adapter.close_for_new_output("test")
    adapter.provider_event(
        {"type": "session.output_transcript.delta", "delta": "چیز دیگری هم هست؟"}, now=0.0,
    )
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=0.0)
    assert adapter.epoch == 2
    assert _feed(adapter, [
        ("بی", 1500, 1800), (" هو", 1800, 2100), ("ای", 2100, 2400),
    ]) is False
    assert adapter.echo_vetoed == 1


def test_a_short_interruption_right_after_vetoed_echo_still_fires():
    """Kept, a vetoed window would only grow with echo and outvote the few
    words a caller needs to say "stop"."""

    adapter = _fa_adapter("window-veto-short")
    echo = _timed(FA_WEATHER_ECHO_TOKENS, start_ms=900, ms_per_char=65)
    assert _feed(adapter, echo) is False
    assert adapter.echo_vetoed >= 1
    t = echo[-1][2] + 200
    assert _feed(adapter, _timed(["نه", " ص", "بر", " کن"], start_ms=t, ms_per_char=150)) is True


# ── (h) noise and single words ────────────────────────────────────────

@pytest.mark.asyncio
async def test_h_noise_fillers_and_single_word_deltas_never_fire(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    noise = [
        ("mm", 600, 900),            # a murmur
        ("uh hmm", 900, 1500),       # a cough, rendered
        ("okay", 1500, 2400),        # ONE word, nearly a second long
        ("fine", 4200, 4400),        # scattered single words, not one
        ("yeah", 6200, 6500),        # person talking over anything
    ]
    # Non-echo by construction: each word, alone, is judged the caller's
    # (a word the reply contains, "right" ~ "light", would be echo instead
    # and prove nothing here).
    ref = frozenset(" ".join(REPLY).casefold().replace(",", " ").replace(".", " ").split())
    assert all(echo_overlap(w, ref)[1] == 0.0 for w, _, _ in noise[2:])

    def on_send(provider, event):
        if event["type"] == "session.start":
            _speak(provider, REPLY, echo=noise, start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = Build130Client([H.config(), 0.4, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert client.of("speech_started") == []


# ── (i) an unsolicited job result, echoed ─────────────────────────────

def _job_think(answer, gate=None):
    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        if gate is not None:
            await gate.wait()
        return answer, "test-model"
    return think


def _result_on_send(after_commentary):
    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when do admissions open", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append":
            # The fake has already pushed the spoken paraphrase ("Here is what
            # I found.") and its audio; what the mic hears next follows it.
            after_commentary(provider)
        elif event["type"] == "session.close":
            provider.push(H.closed())
    return on_send


@pytest.mark.asyncio
async def test_i_the_echo_of_a_proactively_spoken_result_does_not_cut_it(monkeypatch):
    """(i) R12 speaks a finished job through the same output epochs, so the
    same echo used to cut an UNSOLICITED result mid-sentence."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    answer = "Admissions open on the twelfth of September."
    recorded = H.patch_relay(monkeypatch, think=_job_think(answer))

    def echo(provider):
        provider.push(H.user_delta("here is what", 1600, 2000))
        provider.push(H.user_delta("I found", 2000, 2400))

    client = Build130Client([H.config(), 1.2, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_result_on_send(echo), auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id=DB)

    assert any(answer in e["content"] for e in provider.of("session.commentary.append"))
    assert client.of("speech_started") == []
    assert _stop_instructions(provider) == []
    rows = _delegated_rows(recorded["saves"])
    assert rows and rows[0]["assistant_text"] == answer
    assert rows[-1]["assistant_voice"].get("spoken") is True, rows[-1]["assistant_voice"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ms_per_char", [100, 150])
async def test_i_a_persian_result_echoed_back_spaced_is_not_cut(monkeypatch, ms_per_char):
    """(i), in Persian: the result read out with its compounds JOINED, the
    echo transcribed SPACED at o200k granularity — the shape that still cut an
    unsolicited result mid-sentence after the pair-join landed."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    reply_text, fragments = FA_SPACED_ECHOES["formal-result"]
    answer = reply_text.replace(ZWNJ, "")
    recorded = H.patch_relay(monkeypatch, think=_job_think(answer))

    def speak_result(provider):
        _speak(
            provider, _chunks(answer),
            echo=_timed(fragments, start_ms=1600, ms_per_char=ms_per_char), start_ms=1500,
        )

    client = Build130Client([H.config(language="fa"), 1.6, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_result_on_send(speak_result), auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id=DB)

    assert client.of("speech_started") == []
    assert _stop_instructions(provider) == []
    assert "completed" in client.phases()
    rows = _delegated_rows(recorded["saves"])
    assert rows and rows[0]["assistant_text"] == answer
    assert rows[-1]["assistant_voice"].get("interrupted") is not True, rows[-1]["assistant_voice"]


# ── (j) a real barge-in during a proactive result ─────────────────────

@pytest.mark.asyncio
async def test_j_talking_over_a_proactive_result_stops_speech_not_the_job(monkeypatch):
    """(j) An interrupt is "stop speaking" ONLY (A6-2): the delegation is not
    cancelled and its answer is still persisted in full."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    answer = "Admissions open on the twelfth of September."
    recorded = H.patch_relay(monkeypatch, think=_job_think(answer))

    def interrupt(provider):
        provider.push(H.user_delta("wait wait never mind", 1600, 2000))
        provider.push(H.user_delta("tell me tomorrow instead", 2000, 2500))

    client = Build130Client([H.config(), 1.2, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_result_on_send(interrupt), auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id=DB)

    assert len(client.of("speech_started")) == 1
    assert len(_stop_instructions(provider)) == 1
    assert "cancelled" not in client.phases()
    assert "completed" in client.phases()
    rows = _delegated_rows(recorded["saves"])
    assert rows and rows[0]["assistant_text"] == answer
    assert not rows[-1]["assistant_voice"]["cancelled"]


# ── (k) a declined shot is given back ─────────────────────────────────

@pytest.mark.asyncio
async def test_k_an_unanswered_speech_started_re_arms_the_same_reply(monkeypatch):
    """(k) A shot spent on something the relay could not tell from speech —
    echo the ASR turned into unrelated words, a TV — that the client DECLINED
    (build 131 acts only on near-end acoustic evidence) used to leave the rest
    of the reply with no barge-in at all. After `rearm_ms` with no interrupt,
    a genuine interruption in the SAME epoch fires."""

    H.fast_clocks(
        monkeypatch, voice_live_output_epoch_gap_ms=4000, voice_live_bargein_rearm_ms=150,
    )
    H.patch_relay(monkeypatch)
    holder = {}

    def on_send(provider, event):
        holder["p"] = provider
        if event["type"] == "session.start":
            _speak(provider, REPLY, echo=[
                ("channel seven news", 900, 1300),
                ("at eleven tonight", 1300, 1700),
            ], start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def genuine():
        provider = holder["p"]
        provider.push(H.out_text("more words ", 2300, 2700))
        provider.push(H.out_audio())
        provider.push(H.user_delta("no hang on", 2600, 3000))
        provider.push(H.user_delta("that's wrong actually", 3000, 3500))
        provider.push(H.user_delta("I meant Friday", 3500, 3900))

    # A plain client: it never answers `speech_started`, i.e. it declined.
    client = H.FakeClient([H.config(), 0.35, genuine, 0.3, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    epochs = {f["response_id"] for f in client.of("audio_delta")}
    assert epochs == {_oid(1)}, "both shots must be in ONE reply"
    assert len(client.of("speech_started")) == 2


@pytest.mark.asyncio
async def test_k_an_interrupt_naming_the_previous_reply_still_answers_the_shot(monkeypatch):
    """The commonest real interrupt names the epoch the phone is still
    PLAYING, which the relay may already have retired — so the current epoch
    stays open behind the fence. That interrupt answered the shot all the
    same, and re-arming it would fire a second `speech_started` at a client
    that has already stopped."""

    H.fast_clocks(
        monkeypatch, voice_live_output_epoch_gap_ms=4000, voice_live_bargein_rearm_ms=100,
    )
    H.patch_relay(monkeypatch)
    holder = {}

    def on_send(provider, event):
        holder["p"] = provider
        if event["type"] == "session.start":
            _speak(provider, ["First reply, ", "all of it."], start_ms=0)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def reply_two():
        provider = holder["p"]
        provider.push(H.out_text("And another thing ", 2000, 2400))
        provider.push(H.out_audio())
        provider.push(H.user_delta("no stop that", 2000, 2400))
        provider.push(H.user_delta("is not it", 2400, 2800))

    def more_talk():
        provider = holder["p"]
        provider.push(H.user_delta("I said Friday", 2900, 3300))
        provider.push(H.user_delta("not Thursday", 3300, 3700))

    client = Build130Client(
        [H.config(), 0.2, _idle(1, 900), 0.1, reply_two, 0.3, more_talk, 0.3,
         {"type": "stop"}],
        interrupt_frame={"type": "interrupt", "response_id": _oid(1), "item_id": _oid(1)},
    )
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id=DB)

    assert {f["response_id"] for f in client.of("audio_delta")} == {_oid(1), _oid(2)}
    assert len(client.of("speech_started")) == 1


def _talk(adapter, end_ms):
    """Two contiguous deltas, 800 ms and 4 words between them: enough."""

    adapter.note_user_input(end_ms - 400, speech_ms=400, tokens=2)
    adapter.note_user_input(end_ms, speech_ms=400, tokens=2)


def test_k_an_answered_shot_is_not_re_armed_and_a_shot_spends_its_evidence():
    adapter = LiveClientAdapter("session-k")
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=0.0)
    _talk(adapter, 800)
    assert adapter.bargein_due(600, 3, now=1.0, rearm_ms=1500) is True
    # Past the window with nothing new said: given back, but the evidence
    # that fired the first shot does not fire a second one by itself.
    assert adapter.bargein_due(600, 3, now=2.6, rearm_ms=1500) is False
    # Speech inside the window of an unanswered shot waits for the window...
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"}, now=2.6)
    assert adapter.bargein_due(600, 3, now=2.7, rearm_ms=1500) is False
    _talk(adapter, 1700)
    assert adapter.bargein_due(600, 3, now=2.8, rearm_ms=1500) is True
    _talk(adapter, 2600)
    assert adapter.bargein_due(600, 3, now=3.5, rearm_ms=1500) is False
    # ...and counts once it has passed.
    assert adapter.bargein_due(600, 3, now=4.4, rearm_ms=1500) is True
    # An ANSWERED shot is never given back, however long it has been.
    adapter.bargein_answered()
    _talk(adapter, 3500)
    assert adapter.bargein_due(600, 3, now=99.0, rearm_ms=1500) is False


# ── pure-function edges ───────────────────────────────────────────────

def test_the_overlap_is_measured_on_content_words_only():
    ref = frozenset("the weather in toronto is sunny".split())
    # Function words never vote: "the ... in" must not make this echo.
    assert echo_overlap("no the one in london", ref) == (3, 0.0)
    assert echo_overlap("Toronto, sunny!", ref) == (2, 1.0)
    assert echo_overlap("uh... hmm", ref) == (0, 0.0)


def test_a_single_delta_can_never_satisfy_the_rule():
    adapter = LiveClientAdapter("session-one")
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"})
    adapter.note_user_input(2000, speech_ms=2000, tokens=1, text="stop")
    assert adapter.bargein_due(600, 3) is False
    adapter2 = LiveClientAdapter("session-two")
    adapter2.provider_event({"type": "session.output_audio.delta", "delta": "A"})
    adapter2.note_user_input(300, speech_ms=300, tokens=4, text="no wait stop now")
    assert adapter2.bargein_due(600, 3) is False
    # Long AND wordy, but still one ASR emission: not yet an interruption...
    adapter3 = LiveClientAdapter("session-three")
    adapter3.provider_event({"type": "session.output_audio.delta", "delta": "A"})
    adapter3.note_user_input(1200, speech_ms=1200, tokens=5, text="no wait stop that now")
    assert adapter3.bargein_due(600, 3) is False
    # ...until the speech continues into a second one.
    adapter3.note_user_input(1500, speech_ms=300, tokens=1, text="please")
    assert adapter3.bargein_due(600, 3) is True


# ── (l) A7-08's four values ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_l_the_four_load_bearing_values_are_unchanged(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    client = Build130Client([H.config(), 0.4, _idle(1, 4200), 0.3, {"type": "stop"}])
    await H.run_relay(
        client, H.FakeProvider(on_send=_repro_on_send), timeout=6, db_session_id=DB,
    )

    caps = client.of("ready")[0]["capabilities"]
    assert caps["voice_tasks"] is False
    assert caps["voice_provider"] == "live"
    assert caps["playback_ack"] is True
    assert caps["audio"] == {"type": "pcm16le", "rate": 24000}
    complete = client.of("speech_segment_complete")[0]
    assert complete["provider_complete"] is False
    assert complete["durable"] is False
    deltas = client.of("audio_delta")
    assert deltas and all(f["response_id"] == f["item_id"] for f in deltas)
    assert all(s in LIVE_STATES for s in client.states())
