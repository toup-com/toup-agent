"""R2 fix round (w2b): the media backstop and the media findings F13/F25/F28/F29.

Production (RUNTIME_EVIDENCE #4, V2+03:17 and V3+00:09): «آهنگ رو قطع» closed as
a turn, GPT-Live only said «چشم», no delegation was ever made — and the music
played on. Every stop/pause/next path started from a model delegation, so a
turn the model merely acknowledged stopped nothing (critic gap). Addendum 6:
a CLOSED causal turn whose own words are a bare media command and that no
delegation claims within `voice_live_media_backstop_grace_ms` is executed by
the relay through the same correlated media_control path (stop/pause only
with `media_transport`; TF132 gets next/previous), consuming the turn; a later
provider delegation for that turn is absorbed and never rebinds to an older one.

Findings pinned here (each reproduced by an independent verifier's probe under
/private/tmp/toup-voice-r2-review/probes/):

- F13 "That track didn't start" had no causal parent and was recorded as the
  answer to an unrelated newer turn (verify-relay-tasks-6-1).
- F25 "stop that music" read as a task cancel: research killed, music playing
  (verify-media-0-0 / 0-1).
- F28 "back to the music" / «برگرد به آهنگ» skipped BACK a track
  (verify-media-3-0 / 3-1).
- F29 a play asked before a spoken stop still started after the stop was
  confirmed (verify-media-4-0).

Everything drives the real relay through `test_live_harness` with provider-
shaped events; the tenant is faked only at `_control_media_direct` /
`_play_media_direct`.
"""

import ast
import asyncio
import pathlib

import pytest

import test_live_harness as H
from test_live_repair_contract import _terminal, _wait_until
from app.config import settings
from app.services import live_voice_protocol as live
from app.services.live_voice_protocol import detect_media_control, relay_line


BASE = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
]
V03 = BASE + ["media_transport", "reask_turns"]
#: The negotiated set on the recorded call (RUNTIME_EVIDENCE, serving identity).
TF132 = [
    "delegation_frames", "heard_text", "live_turns", "media_control",
    "playback_frames", "task_lifecycle", "turn_timing",
]
STOP = "آهنگ رو قطع کن"
WEATHER = "what's the weather in paris this weekend"


def _turn(n: int) -> str:
    return f"live-utt:{H.PSID}:{n}"


def _line(key, lang, **fmt):
    text = relay_line(key, lang, **fmt)
    assert text, f"relay line {key!r} is missing"
    return text


def _instructions(provider) -> list[str]:
    return [str(e.get("content") or "") for e in provider.of("session.instructions.append")]


def _commentary(provider) -> list[str]:
    return [str(e.get("content") or "") for e in provider.of("session.commentary.append")]


def _epochs(client) -> dict[str, dict]:
    """assistant_turn_id → {parent, text} from the response_text frames."""

    out: dict[str, dict] = {}
    for frame in client.of("response_text"):
        row = out.setdefault(
            frame.get("assistant_turn_id"),
            {"parent": frame.get("parent_user_turn_id"), "text": ""},
        )
        row["text"] += frame.get("text") or ""
    return out


def _parents_of(client, needle) -> list:
    return [row["parent"] for row in _epochs(client).values() if needle in row["text"]]


def _stop_actions(client) -> list[dict]:
    return [
        f for f in client.frames
        if f.get("type") == "media_control" or f.get("action") in {"stop", "pause"}
    ]


def _grace(monkeypatch, ms: int) -> None:
    """Shorten the backstop grace. Tolerant of the setting being absent, so on
    a relay without the backstop a test fails on its ASSERTION, not on setup."""

    try:
        monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", ms)
    except (AttributeError, ValueError):
        pass


def test_the_backstop_is_on_by_default_with_a_short_grace():
    assert settings.voice_live_media_backstop_enabled is True
    assert settings.voice_live_media_backstop_grace_ms == 1500


# ══════════════════════════════════════════════════════════════════════
# Addendum 6 — the model only says «چشم»
# ══════════════════════════════════════════════════════════════════════

def _cheshm_call(box, *, older_unanswered: bool = True, stop_text: str = STOP):
    """«آهنگ رو قطع کن» closes; the model voices «چشم» and delegates nothing.

    The model also voices whatever the relay asks it to say (the backstop's
    verdict), on a provider clock that runs after the input."""

    box.setdefault("voice_at", 6000)

    def on_send(provider, event):
        box["p"] = provider
        kind = event["type"]
        if kind == "session.start":
            if older_unanswered:
                # Round 3 pin change (addendum 3 item 3): the backstop reads the
                # anchor's AUTHORITY UNIT, and an older turn with no terminal
                # punctuation is structurally the same phrase as the stop after
                # it («Don't» + "stop the music").  The older question carries
                # its "?" here, as GPT-Live's English input transcripts do, so
                # it stays a request of its own the late delegation must not
                # rebind to — the property this call pins.
                provider.push(H.user_delta(WEATHER + "?", 0, 900))
        elif kind in {"session.instructions.append", "session.commentary.append"}:
            content = str(event.get("content") or "")
            said = box.get("say", {})
            for needle, spoken in said.items():
                if needle in content:
                    start = box["voice_at"]
                    box["voice_at"] += 1500
                    provider.push(H.out_text(spoken, start, start + 500))
                    provider.push(H.out_audio("SPEAK"))
        elif kind == "session.close":
            provider.push(H.closed())
    return on_send


@pytest.mark.asyncio
async def test_v2_v3_the_model_only_says_cheshm_and_the_relay_stops_the_music(monkeypatch):
    """The recorded V2/V3 shape on a v0.3 client: music playing, «آهنگ رو قطع کن»
    closes, GPT-Live answers «چشم» and never delegates. The relay executes the
    stop itself once the grace passes, words the CONFIRMED outcome, parents it
    on the stop turn, and absorbs the model's late delegation for that turn
    instead of binding it to the older unanswered question."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    counters = H.capture_counters(monkeypatch)
    thinks, controls = [], []
    gate = asyncio.Event()
    called = asyncio.Event()
    stopped = _line("media_stopped", "fa")

    async def think(*a, **k):
        thinks.append(k.get("delegation_id"))
        return "It will be sunny.", "m"

    async def control(_u, action):
        controls.append(action)
        called.set()
        await gate.wait()
        return {"ok": True, "action": action, "reason": "stopped"}

    rec = H.patch_relay(monkeypatch, think=think, control=control)
    box = {"say": {stopped: "موزیک رو قطع کردم."}}
    observed = {}

    async def script():
        p = box["p"]
        await asyncio.sleep(0.3)
        p.push(H.user_delta(STOP, 3000, 3600))
        # GPT-Live acknowledges and does nothing else.
        p.push(H.out_text("چشم.", 3700, 3900))
        p.push(H.out_audio("OK"))
        await _wait_until(
            called.is_set, timeout=4, label="the relay executes the stop nobody delegated",
        )
        await asyncio.sleep(0.2)
        observed["said_before_verdict"] = [c for c in _instructions(p) if stopped in c]
        gate.set()
        await _wait_until(
            lambda: any(stopped in c for c in _instructions(p)), timeout=4, label="verdict asked",
        )
        await _wait_until(lambda: _parents_of(client, "موزیک رو قطع کردم"), label="verdict voiced")
        # …and the model catches up with a delegation for the same words.
        p.push(H.delegation("d-late", 7600))
        await asyncio.sleep(0.6)

    client = H.FakeClient([
        H.config(features=V03),
        {"type": "now_playing", "title": "Fadat Sham - Mahasti"},
        script, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_cheshm_call(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-backstop-v2v3")

    stop_turn = _turn(2)
    assert controls == ["stop"], controls
    assert thinks == [], "the late delegation ran as an agent turn"
    frames = client.of("media_control")
    assert [(f["action"], f["status"]) for f in frames] == [
        ("stop", "requested"), ("stop", "executed"),
    ]
    assert frames[0]["control_id"] == frames[1]["control_id"]
    assert all(f["parent_user_turn_id"] == stop_turn for f in frames)
    assert all(str(f["task_id"]).startswith("relay-media:") for f in frames)
    assert frames[1]["ok"] is True and frames[1]["outcome"] == "ok"
    # Worded only after the tenant's verdict, and only what it confirmed.
    assert observed["said_before_verdict"] == []
    verdicts = [c for c in _instructions(provider) if stopped in c]
    assert len(verdicts) == 1
    assert _parents_of(client, "موزیک رو قطع کردم") == [stop_turn]
    assert _parents_of(client, "چشم") == [stop_turn]
    # No card for a media command, and none for the absorbed delegation.
    assert client.of("delegation") == []
    assert [f for f in client.of("tool_call.started")] == []
    names = [name for name, _fields in counters]
    assert "live_media_backstop" in names
    assert ("live_delegation_absorbed", {"reason": "relay_media"}) in counters
    assert "live_delegation_expired" not in names
    # The verdict row exists once, on the stop turn; the voiced epoch folds onto it.
    rows = [
        s for s in rec["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_media_control"
    ]
    assert rows and {r["assistant_text"] for r in rows} == {stopped}
    assert {r["assistant_voice"]["parent_user_turn_id"] for r in rows} == {stop_turn}
    assert {r["assistant_voice"]["task_id"] for r in rows} == {frames[0]["task_id"]}


@pytest.mark.asyncio
async def test_tf132_gets_no_stop_frame_and_no_relay_stop(monkeypatch):
    """TF132 negotiated `media_control` but not `media_transport`: its tracker
    drops a `stop` action and it has no media_stop handler (the tenant would
    wait ~7 s for an ack that never comes). The backstop never stops for it."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _grace(monkeypatch, 300)
    controls = []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, control=control)
    box = {}

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        p.push(H.user_delta(STOP, 3000, 3600))
        p.push(H.out_text("چشم.", 3700, 3900))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.5)

    client = H.FakeClient([H.config(features=TF132), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_cheshm_call(box, older_unanswered=False), auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-backstop-tf132")

    assert controls == []
    assert _stop_actions(client) == []
    assert not any(_line("media_stopped", "fa") in c for c in _instructions(provider))


@pytest.mark.asyncio
async def test_tf132_next_is_backstopped_with_the_frames_it_understands(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _grace(monkeypatch, 300)
    controls = []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "executed", "title": "Bahaneh"}

    H.patch_relay(monkeypatch, control=control)
    box = {}
    switching = _line("media_next", "fa", title="Bahaneh")

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        p.push(H.user_delta("آهنگ بعدی", 3000, 3500))
        p.push(H.out_text("باشه.", 3600, 3800))
        p.push(H.out_audio("OK"))
        await _wait_until(
            lambda: any(switching in c for c in _instructions(p)), timeout=4, label="verdict",
        )
        await asyncio.sleep(0.2)

    # Round 3 pin change (addendum 3 item 3): the relay executes a media
    # command itself only while the phone reports playback (TF132 sends
    # `now_playing` titles too).
    client = H.FakeClient([
        H.config(features=TF132), {"type": "now_playing", "title": "Fadat Sham - Mahasti"},
        script, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_cheshm_call(box, older_unanswered=False), auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-backstop-tf132-next")

    assert controls == ["next"]
    frames = client.of("media_control")
    assert [(f["action"], f["status"]) for f in frames] == [("next", "requested"), ("next", "executed")]
    assert client.of("delegation") == []


@pytest.mark.asyncio
async def test_a_delegation_in_time_is_the_path_and_the_backstop_stays_quiet(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _grace(monkeypatch, 300)
    controls = []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, control=control)
    box = {}

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        p.push(H.user_delta(STOP, 3000, 3600))
        p.push(H.delegation("d1", 3650))
        await _wait_until(lambda: _terminal(client, "d1"), label="d1 terminal")
        await asyncio.sleep(0.8)   # well past the grace

    client = H.FakeClient([H.config(features=V03), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_cheshm_call(box, older_unanswered=False), auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-backstop-in-time")

    assert controls == ["stop"], "executed twice"
    assert {f["task_id"] for f in client.of("media_control")} == {"d1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["the next song is better", "stop the search", "back to the music"])
async def test_a_turn_that_is_not_a_bare_media_command_is_never_backstopped(monkeypatch, text):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _grace(monkeypatch, 200)
    controls = []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "executed"}

    H.patch_relay(monkeypatch, control=control)
    box = {}

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        p.push(H.user_delta(text, 3000, 3600))
        p.push(H.out_text("Sure.", 3700, 3900))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.0)

    client = H.FakeClient([H.config(features=V03), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_cheshm_call(box, older_unanswered=False), auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-backstop-none")

    assert controls == []
    assert client.of("media_control") == []


# ══════════════════════════════════════════════════════════════════════
# F13 — "That track didn't start" answers the turn that asked for the track
# ══════════════════════════════════════════════════════════════════════

async def _play_then_weather(monkeypatch, *, send_failure: bool):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def play(_u, _q, _variety=False):
        return "Starting Halo on the user's device.", {
            "type": "youtube", "video_id": "v1", "title": "Halo",
        }

    H.patch_relay(monkeypatch, play=play)
    starting = _line("media_playing", "en", title="Halo")
    correction = _line("media_start_failed", "en")
    voice_at = [900, 4600]

    def on_send(provider, event):
        kind = event["type"]
        if kind == "session.start":
            provider.push(H.user_delta("play halo by beyonce", 0, 350))
            provider.push(H.delegation("d1", 400))
        elif kind == "session.commentary.append":
            start = voice_at.pop(0) if voice_at else 6000
            provider.push(H.out_text(event.get("content") or "", start, start + 400))
            provider.push(H.out_audio("SPEAK"))
        elif kind == "session.close":
            provider.push(H.closed())

    provider = H.FakeProvider(on_send=on_send, auto_ack=True)

    async def wait_started():
        await _wait_until(lambda: _terminal(client, "d1"), label="d1 terminal")
        await _wait_until(lambda: _parents_of(client, starting), label="starting voiced")
        await asyncio.sleep(0.3)   # the Starting epoch retires on the output gap

    def weather():
        # Asked, and not answered yet: no delegation, no output.
        provider.push(H.user_delta(WEATHER, 3000, 4000))

    async def wait_correction():
        await _wait_until(lambda: _parents_of(client, correction), label="correction voiced")
        await asyncio.sleep(0.3)

    script = [H.config(features=V03), wait_started, weather, 0.4]
    if send_failure:
        script += [
            {"type": "playback_failed", "video_id": "v1", "reason": "stalled"},
            wait_correction,
        ]
    script += [
        {
            "type": "inject_text", "text": WEATHER,
            "reask_of_user_turn_id": _turn(2), "reason": "no_response",
        },
        0.3, {"type": "stop"},
    ]
    client = H.FakeClient(script)
    await H.run_relay(client, provider, timeout=8, db_session_id=f"db-f13-{send_failure}")
    return client, starting, correction


@pytest.mark.asyncio
async def test_the_track_didnt_start_line_answers_the_play_turn_not_the_newer_question(monkeypatch):
    client, starting, correction = await _play_then_weather(monkeypatch, send_failure=True)

    assert _parents_of(client, starting) == [_turn(1)]
    assert _parents_of(client, correction) == [_turn(1)], (
        "the correction was recorded as the answer to the unrelated weather question"
    )
    # …so the weather question is still unanswered, and a reask of it is taken.
    assert [f.get("outcome") for f in client.of("reask_result")] == ["accepted"]


def test_every_relay_speak_call_names_its_parent_or_its_claim():
    """R2 §C, as a guard: a bare `self.speak(...)` with neither `claim=` nor
    `parent_user_turn_id=` is exactly the F13 defect — its epoch falls to
    `consume_direct_turn` and answers whichever unrelated turn is newest."""

    source = pathlib.Path(live.__file__).read_text(encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute) and func.attr == "speak"
            and isinstance(func.value, ast.Name) and func.value.id == "self"
        ):
            continue
        keywords = {k.arg for k in node.keywords}
        if not keywords & {"claim", "parent_user_turn_id"}:
            offenders.append(node.lineno)
    assert offenders == [], f"self.speak(...) without a parent or a claim at lines {offenders}"


# ══════════════════════════════════════════════════════════════════════
# F25 — "stop that music" is the music, never the running task
# ══════════════════════════════════════════════════════════════════════

RESEARCH = "search the web for the history of jazz"


async def _running_then(monkeypatch, text, db):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    release = asyncio.Event()
    started, controls, cancelled = [], [], []
    box = {}

    async def think(_user_id, _task, _session_id, **kwargs):
        started.append(kwargs["delegation_id"])
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(kwargs["delegation_id"])
            raise
        return "Jazz began in New Orleans.", "m"

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, think=think, control=control)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(box.get("first", RESEARCH), 0, 400))
            provider.push(H.delegation("d1", 450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def script():
        await _wait_until(lambda: "d1" in started, label="d1 running")
        box["p"].push(H.user_delta(text, 2000, 2300))
        box["p"].push(H.delegation("d2", 2350))
        await _wait_until(
            lambda: _terminal(client, "d2") or _terminal(client, "d1", "cancelled")
            or "d2" in started,
            label="d2 handled",
        )
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(features=V03), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    try:
        await H.run_relay(client, provider, timeout=8, db_session_id=db)
    finally:
        release.set()
    return client, provider, controls, cancelled, started, box


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["stop that music", "Stop that song.", "اون آهنگو قطع کن"])
async def test_stop_that_music_stops_the_music_and_keeps_the_research(monkeypatch, text):
    client, provider, controls, cancelled, started, _box = await _running_then(
        monkeypatch, text, f"db-f25-{abs(hash(text)) % 10**8}",
    )
    assert controls == ["stop"]
    assert cancelled == [] and not _terminal(client, "d1", "cancelled"), "the research was killed"
    assert started == ["d1"], "the stop went to a full agent turn"
    spoken = " ".join(_commentary(provider))
    lang = "fa" if "آهنگ" in text else "en"
    assert _line("media_stopped", lang) in spoken
    assert _line("delegation_cancelled", lang) not in spoken


@pytest.mark.asyncio
async def test_a_bare_stop_that_is_still_an_explicit_task_cancel(monkeypatch):
    client, provider, controls, cancelled, _started, _box = await _running_then(
        monkeypatch, "stop that", "db-f25-bare",
    )
    assert controls == []
    assert _terminal(client, "d1", "cancelled")


# ══════════════════════════════════════════════════════════════════════
# F28 — "back to the music" is never the previous track
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "lang"), [
    ("ok, back to the music", "en"), ("برگرد به آهنگ", "fa"),
])
async def test_back_to_the_music_never_skips_back_a_track(monkeypatch, text, lang):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    _grace(monkeypatch, 200)
    controls, thinks = [], []
    box = {}

    async def control(_u, action):
        controls.append(action)
        return {
            "ok": True, "action": action,
            "reason": "paused" if action == "pause" else "executed", "title": "Earlier Song",
        }

    async def think(*_a, **k):
        thinks.append(k.get("delegation_id"))
        return "Okay.", "m"

    H.patch_relay(monkeypatch, control=control, think=think)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("pause the music", 0, 400))
            provider.push(H.delegation("d1", 450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def script():
        await _wait_until(lambda: _terminal(client, "d1"), label="pause done")
        box["p"].push(H.user_delta(text, 3000, 3500))
        box["p"].push(H.delegation("d2", 3550))
        await _wait_until(lambda: _terminal(client, "d2"), label="d2 terminal")
        await asyncio.sleep(0.5)   # past the backstop grace

    client = H.FakeClient([H.config(features=V03), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=f"db-f28-{lang}")

    assert controls == ["pause"], "the resume request moved the station back a track"
    assert detect_media_control(text) is None
    assert not any(f.get("action") == "previous" for f in client.of("media_control"))
    spoken = " ".join(_commentary(provider))
    assert _line("media_previous", lang, title="Earlier Song") not in spoken
    # Declined by the grammar, it reaches the agent — which is not a track skip.
    assert thinks == ["d2"]


# ══════════════════════════════════════════════════════════════════════
# F29 — a stop outranks every play asked for before it
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_play_still_searching_when_the_stop_lands_never_starts_the_music(monkeypatch):
    """«play some jazz» (the tenant search is slow), then «stop the music»: the
    stop is confirmed first, then the older play returns — it had already
    broadcast media_play, which lifts the phone's user-stop fence. The relay
    never announces it, ends it cancelled, and re-asserts the newer stop once
    so the latest intent wins on the device."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    timeline: list[tuple[str, str]] = []
    stop_acked = asyncio.Event()
    box = {}

    async def play(_uid, query, variety=False):
        timeline.append(("play_requested", query))
        await stop_acked.wait()
        await asyncio.sleep(0.03)
        timeline.append(("play_broadcast", "JAZZ0000001"))
        return (
            "Starting Jazz Mix on the user's device.",
            {"type": "youtube", "video_id": "JAZZ0000001", "title": "Jazz Mix"},
        )

    async def control(_uid, action):
        timeline.append(("control_requested", action))
        await asyncio.sleep(0.02)
        timeline.append(("control_acked", action))
        stop_acked.set()
        return {"ok": True, "action": action, "reason": "stopped"}

    rec = H.patch_relay(monkeypatch, play=play, control=control)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta("play some jazz", 0, 500))
            p.push(H.delegation("dPlay", 520))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait_until(
            lambda: any(t[0] == "play_requested" for t in timeline), label="play started",
        )
        box["p"].push(H.user_delta("stop the music", 2000, 2600))
        box["p"].push(H.delegation("dStop", 2620))
        await _wait_until(lambda: _terminal(client, "dStop"), label="stop terminal")
        await _wait_until(
            lambda: any(
                _terminal(client, "dPlay", ph) for ph in ("completed", "failed", "cancelled")
            ),
            label="play terminal",
        )
        await _wait_until(
            lambda: [t for t in timeline if t == ("control_requested", "stop")][1:],
            label="stop re-asserted",
        )
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(features=V03), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-f29-inflight")

    spoken = _commentary(provider)
    assert any("Stopped the music" in s for s in spoken)
    assert not any("Jazz Mix" in s for s in spoken), "the stale play was announced"
    assert _terminal(client, "dPlay", "cancelled")
    assert not _terminal(client, "dPlay", "completed")
    names = [t[0] for t in timeline]
    assert names.index("control_acked") < names.index("play_broadcast")
    # The newer stop is re-asserted AFTER the stale media_play went out.
    assert names[names.index("play_broadcast"):].count("control_requested") == 1
    assert not [
        s for s in rec["saves"]
        if (s.get("media") or {}).get("video_id") == "JAZZ0000001"
    ], "a card row for music the caller stopped"


@pytest.mark.asyncio
async def test_a_queued_play_is_dropped_by_a_later_stop_and_never_searched(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    release = asyncio.Event()
    started, plays, controls = [], [], []
    box = {}

    async def think(_u, _t, _s, **kwargs):
        started.append(kwargs["delegation_id"])
        await release.wait()
        return "A long answer.", "m"

    async def play(_uid, query, variety=False):
        plays.append(query)
        return "Starting Jazz Mix.", {"type": "youtube", "video_id": "J1", "title": "Jazz Mix"}

    async def control(_uid, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, think=think, play=play, control=control)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta("find the history of jazz in new orleans", 0, 400))
            p.push(H.delegation("d1", 450))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        p = box["p"]
        await _wait_until(lambda: "d1" in started, label="d1 running")
        p.push(H.user_delta("what is the weather in paris tomorrow", 1500, 1900))
        p.push(H.delegation("d2", 1950))
        await _wait_until(lambda: "d2" in started, label="d2 running")
        p.push(H.user_delta("play some jazz", 3000, 3400))
        p.push(H.delegation("dPlay", 3450))
        await _wait_until(
            lambda: any(
                f.get("phase") == "pending" and f.get("delegation_id") == "dPlay"
                for f in client.of("delegation")
            ),
            label="play queued",
        )
        p.push(H.user_delta("stop the music", 4500, 4900))
        p.push(H.delegation("dStop", 4950))
        await _wait_until(lambda: _terminal(client, "dStop"), label="stop done")
        await _wait_until(lambda: _terminal(client, "dPlay", "cancelled"), label="play dropped")
        release.set()
        await _wait_until(
            lambda: _terminal(client, "d1") and _terminal(client, "d2"), label="research done",
        )
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(features=V03), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    try:
        await H.run_relay(client, provider, timeout=10, db_session_id="db-f29-queued")
    finally:
        release.set()

    assert controls == ["stop"]
    assert plays == [], "a play the caller had since stopped was searched and started"
    dropped = [
        f for f in client.of("delegation")
        if f.get("delegation_id") == "dPlay" and f.get("phase") == "cancelled"
    ]
    assert dropped and dropped[-1].get("reason") == "media_stopped"


@pytest.mark.asyncio
async def test_a_play_asked_after_the_stop_still_plays(monkeypatch):
    """The newer intent wins either way: an explicit play after a stop is the
    caller turning the music back on, and it is untouched."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    plays, controls = [], []
    box = {}

    async def play(_uid, query, variety=False):
        plays.append(query)
        return "Starting Halo.", {"type": "youtube", "video_id": "v1", "title": "Halo"}

    async def control(_uid, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, play=play, control=control)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta("stop the music", 0, 400))
            p.push(H.delegation("dStop", 450))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait_until(lambda: _terminal(client, "dStop"), label="stop done")
        box["p"].push(H.user_delta("play halo by beyonce", 2000, 2400))
        box["p"].push(H.delegation("dPlay", 2450))
        await _wait_until(lambda: _terminal(client, "dPlay"), label="play done")
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(features=V03), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-f29-after")

    assert controls == ["stop"]
    assert len(plays) == 1
    assert _terminal(client, "dPlay", "completed")
    assert any(_line("media_playing", "en", title="Halo") in c for c in _commentary(provider))
