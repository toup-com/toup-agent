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
import time

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


async def _running_then(monkeypatch, text, db, *, playing: bool = False, delegate: bool = True):
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
        if not delegate:
            # Phase D2: the model delegates nothing — only the relay's own
            # media backstop can act on the words.
            for _ in range(150):
                if controls:
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.8)
            return
        box["p"].push(H.delegation("d2", 2350))
        await _wait_until(
            lambda: _terminal(client, "d2") or _terminal(client, "d1", "cancelled")
            or "d2" in started,
            label="d2 handled",
        )
        await asyncio.sleep(0.3)

    # D-R4: `playing` — the phone reports a song, so a bare playback command
    # would be media were nothing else running (gate term 4 is under test).
    playing_frames = (
        [{"type": "now_playing", "title": "Fadat Sham - Mahasti"}] if playing else []
    )
    client = H.FakeClient([H.config(features=V03), *playing_frames, script, {"type": "stop"}])
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
@pytest.mark.parametrize("playing", [False, True])
async def test_a_bare_stop_that_is_still_an_explicit_task_cancel(monkeypatch, playing):
    """D-R4 extension: with a song playing too, "stop that" while a NON-media
    task runs is still that task's cancel (gate term 4), never the music."""

    client, provider, controls, cancelled, _started, _box = await _running_then(
        monkeypatch, "stop that", f"db-f25-bare-{playing}", playing=playing,
    )
    assert controls == []
    assert _terminal(client, "d1", "cancelled")


@pytest.mark.parametrize(("text", "may_cancel"), [
    ("stop it", True), ("stop that", True), ("بسه", True), ("pause", True),
    ("استاپش کن", True), ("skip", True), ("skip it", True), ("skip that", True),
    ("بعدی", False), ("next", False), ("عوضش", False), ("قبلی", False),
    ("previous", False), ("skip the song", False), ("what is the weather", False),
])
def test_only_a_stop_pause_or_bare_skip_can_read_as_a_task_cancel(text, may_cancel):
    """Phase D2 (acceptance-3): the heads gate term 4 binds."""

    assert live.media_playing_may_cancel(text) is may_cancel


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "expected"), [
    ("بعدی", ["next"]), ("next", ["next"]),
    # F25 stays: these can be the running search's cancel.
    ("skip that", []), ("stop it", []),
])
async def test_a_bare_next_while_a_search_runs_is_the_track_never_a_cancel(
    monkeypatch, text, expected,
):
    """Phase D2 (acceptance-3): a song plays and a research task runs; the
    caller says «بعدی» / "next" and the model delegates nothing.  Gate term 4
    (`non_media_work_in_flight`) declined EVERY bare head, so the media
    backstop never armed, the reply backstop refused it (`not_a_request`) and
    the phone was told `unanswered` for a skip nobody performed.  Only a head
    that can read as the task's cancel — "skip that", "stop it" — stands down
    (F25); the research runs on either way."""

    client, _provider, controls, cancelled, started, _box = await _running_then(
        monkeypatch, text, f"db-d2-next-{abs(hash(text)) % 10**8}", playing=True,
        delegate=False,
    )
    assert controls == expected, controls
    assert started == ["d1"], "a playback command became an agent turn"
    assert cancelled == [] and not _terminal(client, "d1", "cancelled")
    frames = client.of("media_control")
    if expected:
        assert [f["status"] for f in frames] == ["requested", "executed"], frames
        assert all(str(f["task_id"]).startswith("relay-media:") for f in frames), frames
    else:
        assert frames == [], frames


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


# ══════════════════════════════════════════════════════════════════════
# D-R4 (2026-10-06 call): a bare playback command while something plays
# ══════════════════════════════════════════════════════════════════════
#
# 2:55 «استاپش کن» while a song played → «چشم.» → «ببخشید، از اینجا نمی‌تونم
# متوقفش کنم» and the song played on.  The grammar declined the nounless
# clitic form, so neither this backstop nor the control fast path saw it.

PLAYING = {"type": "now_playing", "title": "Hotline Bling - Drake"}


def _verdict_reason(action: str) -> str:
    return {"stop": "stopped", "pause": "paused"}.get(action, "executed")


async def _bare_command_call(
    monkeypatch, text, *, playing=True, over_output=False, late_delegation=False,
    db="db-bare", wait_s=1.6, then="",
):
    """«text» closes while (or not) a song plays; the model says «چشم.» and
    delegates nothing (or, `late_delegation`, catches up 0.2 s after the
    relay fired).  `over_output`: the turn begins while the agent is still
    talking.  Returns (controls, seconds from the words to the control,
    client, provider, thinks)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    controls, thinks, at = [], [], {}

    async def control(_u, action):
        controls.append(action)
        at.setdefault("control", time.monotonic())
        return {"ok": True, "action": action, "reason": _verdict_reason(action), "title": "Bahaneh"}

    async def think(*_a, **k):
        thinks.append(k.get("delegation_id"))
        return "An answer.", "m"

    H.patch_relay(monkeypatch, control=control, think=think)
    box = {}

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        if over_output:
            p.push(H.out_text("Jazz began in New Orleans in the early twentieth century, and", 2000, 5200))
            p.push(H.out_audio("LONG"))
            await asyncio.sleep(0.05)
        at["said"] = time.monotonic()
        p.push(H.user_delta(text, 3000, 3400))
        if then:
            # A request of the caller's own, straight after the command.
            await asyncio.sleep(0.15)
            p.push(H.user_delta(then, 3600, 4400))
        if not over_output:
            p.push(H.out_text("چشم.", 3500, 3700))
            p.push(H.out_audio("OK"))
        if late_delegation:
            await _wait_until(lambda: controls, timeout=4, label="the relay carries it out")
            await asyncio.sleep(0.2)
            p.push(H.delegation("d-late", 3450))
        await asyncio.sleep(wait_s)

    frames = [H.config(features=V03)]
    if playing:
        frames.append(PLAYING)
    client = H.FakeClient([*frames, script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_cheshm_call(box, older_unanswered=False), auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=db)
    elapsed = (at["control"] - at["said"]) if "control" in at else None
    return controls, elapsed, client, provider, thinks


def test_the_playing_grace_is_short_and_the_named_grace_is_unchanged():
    assert settings.voice_live_media_backstop_playing_grace_ms == 400
    assert settings.voice_live_media_backstop_grace_ms == 1500


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "action"), [
    ("استاپش کن", "stop"),      # the recorded words
    ("قطعش کن", "stop"),
    ("بسه", "stop"),
    ("stop it", "stop"),
    ("pause", "pause"),
    ("kill it", "stop"),
    ("نگهش دار", "pause"),
    ("خاموشش کن", "stop"),
    ("next", "next"),
])
async def test_d_r4_the_model_only_says_cheshm_and_the_relay_carries_out_a_bare_command(
    monkeypatch, text, action,
):
    controls, elapsed, client, provider, thinks = await _bare_command_call(
        monkeypatch, text, db=f"db-bare-{action}-{abs(hash(text)) % 10**6}",
    )
    assert controls == [action]
    # Within ~1 s of the words: the bare grace is 0.4 s, not the 1.5 s a
    # NAMED command waits for (the recorded stop never ran at all).
    assert elapsed is not None and elapsed < 1.2, elapsed
    assert thinks == []
    assert client.of("delegation") == [], "a playback command is never a card"
    frames = client.of("media_control")
    assert [f["status"] for f in frames] == ["requested", "executed"]
    assert all(str(f["task_id"]).startswith("relay-media:") for f in frames)
    # The model was told at the command, quietly, never under a provider id.
    lang = "fa" if any("؀" <= ch <= "ۿ" for ch in text) else "en"
    told = [
        e for e in provider.of("session.thinking.append")
        if _line("media_command_started", lang, request=text) in str(e.get("content") or "")
    ]
    assert len(told) == 1 and told[0].get("delegation_id") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["استاپش کن", "stop it", "بسه", "pause", "next"])
async def test_d_r4_with_nothing_playing_a_bare_command_is_left_alone(monkeypatch, text):
    controls, _elapsed, client, _provider, _thinks = await _bare_command_call(
        monkeypatch, text, playing=False, db=f"db-bare-off-{abs(hash(text)) % 10**6}",
    )
    assert controls == []
    assert client.of("media_control") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["قطع شد", "صبر کن", "hold on", "قطع نکن", "stop the search"])
async def test_d_r4_complaints_holds_and_negations_never_touch_the_music(monkeypatch, text):
    controls, _elapsed, client, _provider, _thinks = await _bare_command_call(
        monkeypatch, text, db=f"db-bare-neg-{abs(hash(text)) % 10**6}",
    )
    assert controls == []


@pytest.mark.asyncio
async def test_d_r4_an_ambiguous_stop_over_the_agents_voice_is_a_barge_in(monkeypatch):
    """Gate term 3: "stop it" said while the agent is talking means the agent
    — the phone's hold/cut answers it, the music plays on."""

    controls, _elapsed, _client, _provider, _thinks = await _bare_command_call(
        monkeypatch, "stop it", over_output=True, db="db-bare-over-ambiguous",
    )
    assert controls == []


@pytest.mark.asyncio
async def test_d_r4_an_object_explicit_stop_over_the_agents_voice_still_stops(monkeypatch):
    """…while «استاپش کن» names the music: a command even over the agent's
    voice (DESIGN-R2 D-R4 ruling, critic G4 iii)."""

    controls, _elapsed, _client, _provider, _thinks = await _bare_command_call(
        monkeypatch, "استاپش کن", over_output=True, db="db-bare-over-explicit",
    )
    assert controls == ["stop"]


@pytest.mark.asyncio
async def test_d_r4_a_late_model_delegation_never_skips_twice(monkeypatch):
    """Critic R4: a bare «بعدی» is carried out at 0.4 s; the model's own
    delegation for the same words lands just after — it is absorbed, the
    station moves ONE track, and no agent turn runs."""

    counters = H.capture_counters(monkeypatch)
    controls, _elapsed, client, _provider, thinks = await _bare_command_call(
        monkeypatch, "بعدی", late_delegation=True, db="db-bare-late",
    )
    assert controls == ["next"]
    assert thinks == []
    assert ("live_delegation_absorbed", {"reason": "relay_media"}) in counters
    assert client.of("delegation") == []


# ══════════════════════════════════════════════════════════════════════
# D-R3 (2026-10-06 call): the relay plays what the caller asked for
# ══════════════════════════════════════════════════════════════════════
#
# 1:20 «اوکی برا می آهنگ پخش کن» → «حتماً، دارم پخش می‌کنم.» → nothing: no
# delegation, no card, no music.  2:00 «نه یه آهنگی برام پخش کن» stuck 31 s.
# The shared `media_request` declines BOTH (the ASR split «برا می», the lead
# «نه»), which is why the relay reads plays with its own repairs.

RECORDED_PLAY_1 = "اوکی برا می آهنگ پخش کن"
RECORDED_PLAY_2 = "نه یه آهنگی برام پخش کن"
ACK = "حتماً، دارم پخش می‌کنم."


async def _play_call(
    monkeypatch, text, *, reply=ACK, delegate_at=None, late_delegation=False,
    db="db-play", wait_s=2.2, then=None, clocks=None, features=V03,
):
    """«text» closes; the model voices `reply` and (unless `delegate_at`)
    delegates nothing; `then=(words, after_s)` is a second caller turn spoken
    `after_s` later (provider 3900–4400 ms); `clocks` merges extra settings.
    Returns (plays, thinks, client, provider, counters)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0,
                  voice_live_play_backstop_enabled=True, **(clocks or {}))
    counters = H.capture_counters(monkeypatch)
    plays, thinks = [], []

    async def play(_u, query, variety=False):
        plays.append((query, variety))
        return "Starting Bahaneh on the user's device; more in the same style will follow.", {
            "type": "youtube", "video_id": "v1", "title": "Bahaneh",
        }

    async def think(*_a, **k):
        thinks.append(k.get("delegation_id"))
        return "Starting it.", "m"

    H.patch_relay(monkeypatch, play=play, think=think)
    box = {}

    async def script():
        p = box["p"]
        await asyncio.sleep(0.2)
        p.push(H.user_delta(text, 3000, 3600))
        if delegate_at is not None:
            p.push(H.delegation("d1", delegate_at))
        if reply:
            p.push(H.out_text(reply, 3700, 4200))
            p.push(H.out_audio("OK"))
        if then is not None:
            await asyncio.sleep(then[1])
            p.push(H.user_delta(then[0], 3900, 4400))
        if late_delegation:
            await _wait_until(lambda: plays, timeout=4, label="the relay plays it")
            await asyncio.sleep(0.2)
            p.push(H.delegation("d-late", 3650))
        await asyncio.sleep(wait_s)

    client = H.FakeClient([H.config(features=features), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_cheshm_call(box, older_unanswered=False), auto_ack=True)
    await H.run_relay(client, provider, timeout=9, db_session_id=db)
    return plays, thinks, client, provider, counters


def _relay_play_ids(client) -> set:
    return {
        f.get("delegation_id") or f.get("task_id") for f in client.of("delegation")
        if str(f.get("delegation_id") or f.get("task_id") or "").startswith("relay-play:")
    }


def test_the_reply_ladder_promise_and_order_defaults_are_the_contract():
    """Phase D (tests-guards-7): `fast_clocks` overrides every one of these in
    every suite, so a config edit reverting regen to 8 s, shipping the reply
    backstop off, or zeroing the claim grace / order wait turned a recorded
    D-R1/D-R7 fix off in production with every suite green.  The DEFAULTS
    (DESIGN-R2 §3) are read here, never the patched instance."""
    from app.config import Settings

    S = Settings.model_fields
    assert S["voice_live_reply_regen_after_s"].default == 3.0
    assert S["voice_live_reply_backstop_enabled"].default is True
    assert S["voice_live_reply_backstop_after_s"].default == 8.0
    assert S["voice_live_claim_grace_ms"].default == 1500
    assert S["voice_live_result_order_wait_s"].default == 4.0
    assert live._REPLY_REGEN_ATTEMPTS == 1 and live._REASK_MAX_PER_TURN == 2


def test_the_play_backstop_is_on_by_default_with_its_own_grace():
    assert settings.voice_live_play_backstop_enabled is True
    # Voice round 2026-10-07 (D-R4 (3)): a PARSED play waits 300 ms, not 800.
    assert settings.voice_live_play_backstop_grace_ms == 300
    assert settings.voice_live_media_backstop_grace_ms == 1500
    assert live._PLAY_BACKSTOP_MAX_WAIT_S == 3.0
    assert live.relay_opened_delegation("relay-play:abc")
    assert "relay-play:" in live.RELAY_OPENED_PREFIXES


@pytest.mark.parametrize(("text", "query", "variety"), [
    (RECORDED_PLAY_1, "آهنگ", True),          # the ASR split «برا می» = «برام»
    (RECORDED_PLAY_2, "آهنگ", True),          # one leading «نه» dropped
    ("یه آهنگی برام پخش کن", "آهنگ", True),
    ("یه چیزی پخش کن", "آهنگ", True),          # unparsed, open-ended
    ("play something", "music", True),        # unparsed, open-ended
    ("can you play a song", "song", True),
    # Phase D (acceptance-6): an indefinite bare media noun names nothing — it
    # is open-ended (`variety`), never the top hit for the literal "song".
    ("play a song", "song", True),
    ("play me a song", "song", True),
    ("put on a song", "song", True),
    ("play music", "music", True),
])
def test_play_request_evidence_reads_the_recorded_asks(text, query, variety):
    ask, is_play = live.play_request_evidence(text)
    assert is_play and ask is not None, text
    assert (ask.query, ask.variety) == (query, variety)


@pytest.mark.parametrize("text", [
    "دیشب یه آهنگ قشنگ شنیدم", "what's this song", "آهنگ پخش نکن", "نه آهنگ پخش نکن",
    "play it back", "آهنگ قبلی چی بود؟", "آهنگ پخش کردم", "آهنگ بعدی رو بذار",
    "I played a song yesterday", "music please",
    # A negation anywhere — the unparsed reading would otherwise hand these to
    # the agent as plays.
    "don't play any music", "do not play a song", "never play music again",
    # Phase D (correctness-relay-2): «بذار» alone is "let me", «یه چیزی» is any
    # "something", a reported frame is no ask, and a question / instruction /
    # deferral about a play is none either — each of these opened a
    # `relay-play:` agent turn titled with the caller's words.
    "بذار یه چیزی بگم", "بذار یه چیزی رو چک کنم", "بذار یه چیزی بپرسم",
    "یه چیزی بذار تو تقویمم", "بذار برم یه چیزی بیارم", "دیروز بهم گفت یه آهنگ پخش کن",
    "why did you play that song", "did you play the song",
    "tell me how to play a song on guitar", "you can play a song later",
])
def test_play_request_evidence_declines_what_is_not_a_play(text):
    assert live.play_request_evidence(text) == (None, False), text


@pytest.mark.parametrize("text", [
    "play the song from yesterday", "play something by drake",
    "can you play something by drake?", "یه چیزی از گوگوش پخش کن",
])
def test_an_unparsed_play_with_a_subject_is_the_agents(text):
    assert live.play_request_evidence(text) == (None, True), text


def test_a_definite_media_noun_is_not_made_open_ended():
    ask, is_play = live.play_request_evidence("play the song")
    assert is_play and ask is not None and ask.variety is False


def test_a_model_delegated_bare_song_ask_is_open_ended_on_the_fast_path():
    """The common path: the model delegates "play a song" itself, so the task
    carries no relay `media_ask` and `task_media_ask` reads its words."""

    task = live.DelegatedTask(delegation_id="d1", offset_ms=0, transcript="play a song")
    ask = live.task_media_ask(task)
    assert ask is not None and (ask.query, ask.variety) == ("song", True)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [RECORDED_PLAY_1, RECORDED_PLAY_2, "یه آهنگی برام پخش کن"])
async def test_d_r3_a_play_the_model_only_acknowledges_is_started_by_the_relay(monkeypatch, text):
    plays, thinks, client, provider, counters = await _play_call(
        monkeypatch, text, db=f"db-play-ack-{abs(hash(text)) % 10**6}",
    )
    assert plays == [("آهنگ", True)], plays
    assert thinks == [], "a parsed play is the fast path, never an agent turn"
    ids = _relay_play_ids(client)
    assert len(ids) == 1, client.of("delegation")
    created = [f for f in client.of("delegation") if f.get("phase") == "created"
               or f.get("status") == "created"]
    assert created, client.of("delegation")
    # The caller hears the relay's own line (an instruction: GPT-Live never
    # minted this id), and the model was told, quietly, that the app plays it.
    starting = _line("media_playing", "fa", title="Bahaneh")
    assert any(starting in c for c in _instructions(provider))
    told = [
        e for e in provider.of("session.thinking.append")
        if "«" + " ".join(text.split()) + "»" in str(e.get("content") or "")
    ]
    assert told and all(e.get("delegation_id") is None for e in told)
    assert ("live_play_backstop", {"outcome": "fired_parsed"}) in counters


@pytest.mark.asyncio
async def test_d_r3_a_play_delegation_in_time_is_the_path_and_the_backstop_stays_quiet(monkeypatch):
    plays, thinks, client, _provider, _counters = await _play_call(
        monkeypatch, "یه آهنگ برام پخش کن", reply="", delegate_at=3650, db="db-play-in-time",
    )
    assert plays == [("آهنگ", True)]
    assert _relay_play_ids(client) == set()


@pytest.mark.asyncio
async def test_d_r3_a_late_model_delegation_for_the_same_play_is_absorbed(monkeypatch):
    plays, thinks, client, provider, counters = await _play_call(
        monkeypatch, RECORDED_PLAY_1, late_delegation=True, db="db-play-late",
    )
    assert plays == [("آهنگ", True)], "the music was started twice"
    assert thinks == []
    assert ("live_delegation_absorbed", {"reason": "relay_play"}) in counters
    # One card: the relay's.
    starts = {f.get("delegation_id") for f in client.of("delegation")}
    assert starts == _relay_play_ids(client)


@pytest.mark.asyncio
async def test_d_r3_an_unparsed_play_runs_the_agent_with_the_callers_words(monkeypatch):
    plays, thinks, client, _provider, _counters = await _play_call(
        monkeypatch, "play the song from yesterday", reply="Sure.", db="db-play-agent",
        wait_s=3.0,
    )
    assert plays == []
    assert len(thinks) == 1 and str(thinks[0]).startswith("relay-play:"), thinks


@pytest.mark.asyncio
async def test_d_r3_q_s1_an_unparsed_play_answered_with_a_question_is_left_to_the_caller(monkeypatch):
    plays, thinks, client, _provider, counters = await _play_call(
        monkeypatch, "play the song from yesterday", reply="Which song do you mean?",
        db="db-play-asked", wait_s=5.0,
    )
    assert plays == [] and thinks == []
    assert _relay_play_ids(client) == set()
    assert ("live_play_backstop", {"outcome": "skipped_model_asked"}) in counters


@pytest.mark.asyncio
async def test_d_r3_q_s1_a_parsed_play_plays_even_under_a_clarifying_question(monkeypatch):
    plays, _thinks, _client, _provider, _counters = await _play_call(
        monkeypatch, "یه آهنگ برام پخش کن", reply="چه آهنگی دوست داری؟", db="db-play-parsed-q",
    )
    assert plays == [("آهنگ", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "دیشب یه آهنگ قشنگ شنیدم", "what's this song", "آهنگ پخش نکن", "نه آهنگ پخش نکن",
    "play it back", "آهنگ قبلی چی بود؟",
])
async def test_d_r3_what_is_not_a_play_is_never_played(monkeypatch, text):
    plays, thinks, client, _provider, _counters = await _play_call(
        monkeypatch, text, reply="Okay.", db=f"db-play-neg-{abs(hash(text)) % 10**6}", wait_s=2.0,
    )
    assert plays == [] and thinks == []
    assert _relay_play_ids(client) == set()


@pytest.mark.asyncio
async def test_d_r3_let_me_say_something_never_opens_a_relay_play(monkeypatch):
    """Phase D (correctness-relay-2), end to end: «بذار یه چیزی بگم» answered
    «بفرمایید.» opened a `relay-play:` agent turn ~1.5 s later."""

    plays, thinks, client, _provider, _counters = await _play_call(
        monkeypatch, "بذار یه چیزی بگم", reply="بفرمایید.", db="db-play-let-me", wait_s=2.5,
    )
    assert plays == [] and thinks == []
    assert _relay_play_ids(client) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "tail"), [
    ("play some music", "by Drake"),
    ("یه آهنگ برام پخش کن", "از دریک"),
])
async def test_d_r3_a_play_split_across_two_turns_plays_once_with_the_whole_phrase(
    monkeypatch, first, tail,
):
    """Phase D (correctness-relay-3): each closed turn armed on its own, and
    the fire recomputed the unit — the head played "some music" through the
    fast path, then the tail ("by Drake") opened a SECOND relay-play agent
    turn.  The later arming owns the phrase: one play, carrying both halves."""

    plays, thinks, client, _provider, counters = await _play_call(
        monkeypatch, first, reply="", then=(tail, 0.3),
        db=f"db-play-split-{abs(hash(first)) % 10**6}", wait_s=3.5,
    )
    assert len(plays) + len(thinks) == 1, (plays, thinks)
    assert len(_relay_play_ids(client)) == 1, client.of("delegation")
    titles = [str(f.get("title") or f.get("request") or "") for f in client.of("delegation")
              if f.get("phase") == "created" or f.get("status") == "created"]
    assert any(tail.split()[-1] in t for t in titles), titles
    assert ("live_play_backstop", {"outcome": "skipped_extended"}) in counters


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "retract"), [
    ("play some music", "no wait, never mind"),
    ("یه آهنگ برام پخش کن", "نه ولش کن، پخش نکن"),
    ("یه آهنگ برام پخش کن", "نه ولش کن"),
])
async def test_d_r3_a_play_retracted_in_the_next_turn_is_never_played(monkeypatch, first, retract):
    """Phase D (correctness-relay-4): the retraction is its OWN turn, so the
    armed unit's negation reader never saw it and the music started ~0.8 s
    after the first turn closed."""

    plays, thinks, client, _provider, counters = await _play_call(
        monkeypatch, first, reply="", then=(retract, 0.3),
        db=f"db-play-retract-{abs(hash(retract)) % 10**6}", wait_s=3.0,
    )
    assert plays == [] and thinks == [], (plays, thinks)
    assert _relay_play_ids(client) == set()
    assert ("live_play_backstop", {"outcome": "skipped_retracted"}) in counters


@pytest.mark.asyncio
async def test_d_r3_a_corrected_play_still_plays_once_from_the_newer_turn(monkeypatch):
    """Positive control for the retraction reader: "no, play some Drake" is a
    play of its own, never a retraction."""

    plays, thinks, client, _provider, _counters = await _play_call(
        monkeypatch, "play some music", reply="", then=("no, play some drake", 0.3),
        db="db-play-corrected", wait_s=3.0,
    )
    assert len(plays) + len(thinks) == 1, (plays, thinks)
    assert plays and "drake" in plays[0][0].lower(), plays


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "retract"), [
    ("play some music", "no wait, never mind"),
    ("یه آهنگ بذار", "نه ولش کن"),
    # Reads as a request of its own (`correction`): the reply backstop used to
    # hand «…پخش کن نه ولش کن، پخش نکن» to the agent as a `relay-reply:` card.
    ("یه آهنگ برام پخش کن", "نه ولش کن، پخش نکن"),
])
async def test_d3_a_retracted_play_ends_the_owed_reply_and_nothing_plays_it_later(
    monkeypatch, caplog, first, retract,
):
    """Phase D3 (supervisor ruling 1): the play backstop stood down for the
    retraction, and the REPLY LADDER then asked the model again and handed
    the caller's span — «play some music» + «no wait, never mind» — to the
    agent as a `relay-reply:` card: the agent played what the caller had just
    cancelled.  A retraction ends the owed reply the moment it is judged (at
    the retracting turn's close): no reask, no `relay-reply:`, no play, and
    the caller hears `reply_status unanswered reason=superseded` — the caller
    moved on.  Every rung is on, as in production."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    plays, thinks, client, provider, counters = await _play_call(
        monkeypatch, first, reply="", then=(retract, 0.3), wait_s=3.4,
        db=f"db-d3-retract-{abs(hash(retract)) % 10**6}",
        features=V03 + ["reply_status"],
        clocks=dict(
            voice_live_claim_grace_ms=200, voice_live_reply_regen_after_s=1.0,
            voice_live_reply_backstop_enabled=True, voice_live_uplink_keeper_enabled=True,
        ),
    )
    assert plays == [] and thinks == [], (plays, thinks)
    assert _relay_play_ids(client) == set()
    assert not [f for f in client.of("delegation")
                if str(f.get("delegation_id") or "").startswith("relay-reply:")], (
        client.of("delegation"))
    reasks = [e for e in provider.of("session.instructions.append")
              if str(e.get("event_id") or "").startswith("toup-live-reask-")]
    assert reasks == [], reasks
    finals = [f for f in client.of("transcript") if f.get("final")]
    played = next(f for f in finals if f["text"] == first)["turn_id"]
    cancelled = next(f for f in finals if f["text"] == retract)["turn_id"]
    # The retraction is not phatic, so it is the newest request and the reply
    # owed is ITS — over a span that still carries the play.
    statuses = [(f["user_turn_id"], f["outcome"], f.get("reason")) for f in client.of("reply_status")]
    assert (cancelled, "unanswered", "superseded") in statuses, statuses
    assert not [s for s in statuses if s[1] in ("pending", "reasked", "delegated")], statuses
    logs = [r.getMessage() for r in caplog.records]
    assert f"[LIVE] reply not owed turn_id={cancelled} reason=retracted" in logs, (
        [m for m in logs if "reply" in m])
    for turn_id in (played, cancelled):
        assert any(
            m.startswith(f"[LIVE] turn timeline turn_id={turn_id} reason=unanswered_superseded ")
            for m in logs
        ), [m for m in logs if "turn timeline" in m]
    assert not any("reply backstop fired" in m for m in logs)
    assert ("live_play_backstop", {"outcome": "skipped_retracted"}) in counters, counters


@pytest.mark.asyncio
async def test_d3_a_retraction_ends_the_owed_reply_at_its_close_not_at_the_play_grace(
    monkeypatch, caplog,
):
    """The retraction is judged when the retracting turn CLOSES.  Judged only
    at the play's grace (here 2.5 s, longer than the ladder's 0.6 s regen),
    the ladder re-asked the model about «play some music no wait, never mind»
    first — the reask itself invites the play the caller cancelled."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    plays, thinks, client, provider, counters = await _play_call(
        monkeypatch, "play some music", reply="", then=("no wait, never mind", 0.3), wait_s=3.4,
        db="db-d3-retract-at-close", features=V03 + ["reply_status"],
        clocks=dict(
            voice_live_play_backstop_grace_ms=2500,
            voice_live_claim_grace_ms=200, voice_live_reply_regen_after_s=0.6,
            voice_live_reply_backstop_enabled=True, voice_live_uplink_keeper_enabled=True,
        ),
    )
    assert plays == [] and thinks == [], (plays, thinks)
    reasks = [e for e in provider.of("session.instructions.append")
              if str(e.get("event_id") or "").startswith("toup-live-reask-")]
    assert reasks == [], reasks
    assert not [f for f in client.of("delegation")
                if str(f.get("delegation_id") or "").startswith(("relay-reply:", "relay-play:"))]
    logs = [r.getMessage() for r in caplog.records]
    skipped = next(i for i, m in enumerate(logs)
                   if "play backstop skipped" in m and "reason=retracted" in m)
    ended = next(i for i, m in enumerate(logs) if "reply not owed" in m and "reason=retracted" in m)
    assert skipped < ended, logs
    assert not any("via=relay:regen" in m for m in logs[:ended]), logs
    assert ("live_play_backstop", {"outcome": "skipped_retracted"}) in counters, counters


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["", ACK])
async def test_d_r3_the_play_backstop_owns_the_turn_with_the_reply_ladder_and_promise_mode_on(
    monkeypatch, caplog, reply,
):
    """Phase D (tests-guards-4): every suite runs ONE of the new relay rungs at
    a time (`fast_clocks` pins the others off), so the `play_armed` hold —
    the reply ladder standing down while the play backstop owns the turn —
    was pinned by nothing.  Production runs all of them: the play still plays
    once, never as an agent turn, never as a `relay-reply:` card, and an
    acknowledgement is not read as a promise while the play is armed."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    plays, thinks, client, _provider, counters = await _play_call(
        monkeypatch, RECORDED_PLAY_2, reply=reply, wait_s=3.0,
        db=f"db-play-prod-{bool(reply)}",
        clocks=dict(
            voice_live_claim_grace_ms=200, voice_live_reply_regen_after_s=0.3,
            voice_live_reply_backstop_enabled=True, voice_live_uplink_keeper_enabled=True,
            # 2026-10-07 (D-R4 (3)): the parsed grace is 300 ms in production,
            # which fires the play before this 0.3 s ladder rung can look at
            # it; the HOLD is what is under test, so the play stays armed past
            # the rung here (the old 800 ms).
            voice_live_play_backstop_grace_ms=800,
        ),
    )
    assert plays == [("آهنگ", True)], plays
    assert thinks == []
    assert not [f for f in client.of("delegation")
                if str(f.get("delegation_id") or "").startswith("relay-reply:")]
    assert not [r for r in caplog.records if "reply was a promise" in r.getMessage()]
    if not reply:
        # Phase D2: an armed play is a HOLD the ladder re-decides, never a skip.
        assert ("live_reply_backstop_held", {"reason": "play_armed"}) in counters, counters


@pytest.mark.asyncio
async def test_d_r3_the_harness_pins_the_play_backstop_off(monkeypatch):
    """`fast_clocks` keeps every older suite's play-shaped turn quiet."""

    H.fast_clocks(monkeypatch)
    assert settings.voice_live_play_backstop_enabled is False
    assert settings.voice_live_claim_grace_ms == 0


# ══════════════════════════════════════════════════════════════════════
# D-R1(e), the reader half: an action CLAIM is not an action
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("said", [
    "حتماً، دارم پخش می‌کنم.",          # 12:39:37, recorded
    "دارم راهش می‌ندازم، یه لحظه.",     # 12:39:55, recorded
    "چشم.",                             # 12:41:11, recorded (a stop)
    "playing it now", "I'll put it on", "Sure.", "On it.", "here it comes",
])
def test_action_claim_evidence_reads_the_recorded_claims(said):
    assert live.action_claim_evidence(said) is True


@pytest.mark.parametrize("said", [
    "دارم فکر می‌کنم", "پخش شد؟", "I played that yesterday", "Which song do you want?",
    "حتماً! چه آهنگی دوست داری؟", "The weather in Paris will be sunny this weekend.",
    "Okay, I'm looking that up for you.", "",
    # Claim-shaped, but asked: an offer is not a claim.
    "الان پخشش می‌کنم؟", "I'll play it now, okay?",
])
def test_action_claim_evidence_refuses_questions_thoughts_reports_and_answers(said):
    assert live.action_claim_evidence(said) is False


def test_the_promise_reask_names_what_was_said_in_both_languages():
    for lang in ("en", "fa"):
        line = live._REASK_PROMISE_LINES[lang].format(said="«x»", words="«y»")
        assert "«x»" in line and "«y»" in line


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["استاپش کن", "stop it"])
async def test_d_r4_a_request_right_after_a_whole_command_does_not_cancel_it(monkeypatch, text):
    """«استاپش کن» / "stop it" name their object: the phrase is whole, and the
    weather question the caller asks straight after is a request of its own —
    never a continuation that voids the stop (only an OBJECTLESS "Stop." is
    re-read with what follows it)."""

    controls, _elapsed, _client, _provider, _thinks = await _bare_command_call(
        monkeypatch, text, then="what's the weather in paris this weekend",
        db=f"db-bare-then-{abs(hash(text)) % 10**6}",
    )
    assert controls == ["stop"]


# ══════════════════════════════════════════════════════════════════════
# Voice round 2026-10-07 (D-R4 (3)): «حالا یه دنه آهنگ دریک واسم پخش کن»
# ══════════════════════════════════════════════════════════════════════
# The recorded ask fell out of the grammar on «دنه» alone: `play backstop
# armed … parsed=no grace_ms=1500` → a 12.1 s agent play.  Now it parses, the
# backstop waits the PARSED grace (300 ms), and the fast path plays it.

RECORDED_PLAY_1007 = "حالا یه دنه آهنگ دریک واسم پخش کن"


@pytest.mark.asyncio
async def test_a_parsed_play_fires_after_the_short_grace(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    plays, thinks, client, _provider, counters = await _play_call(
        monkeypatch, RECORDED_PLAY_1007, reply="باشه.", db="db-play-1007",
    )
    assert plays == [("Drake", True)], plays
    assert thinks == [], "a parsed play is the fast path, never an agent turn"
    assert len(_relay_play_ids(client)) == 1
    armed = [r.getMessage() for r in caplog.records if "play backstop armed" in r.getMessage()]
    assert armed and all("parsed=yes grace_ms=300" in line for line in armed), armed
    assert ("live_play_backstop", {"outcome": "fired_parsed"}) in counters


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "query"), [
    ("now play a song by drake", "drake"),
    ("okay now play a song by drake", "drake"),
])
async def test_a_model_delegation_of_now_play_gets_the_relay_repairs(monkeypatch, text, query):
    """GPT-Live delegated it in time: the delegation is the path (no relay
    play), and `task_media_ask` reads the repaired ask — so it is the media
    fast path, never a 12 s agent play."""

    plays, thinks, client, _provider, _counters = await _play_call(
        monkeypatch, text, reply="", delegate_at=3650,
        db=f"db-play-now-{abs(hash(text)) % 10**6}",
    )
    assert [q for q, _variety in plays] == [query], plays
    assert thinks == [], "the model's own play delegation ran an agent turn"
    assert _relay_play_ids(client) == set()
