"""R2 §H, relay side: spoken stop/pause, next that is never a search, and media
wording that claims only what the tenant confirmed.

Production (2026-09-22, RUNTIME_EVIDENCE items 4-5): "آهنگ رو قطع" closed as a
turn and nothing happened — no delegation, no control, the relay had no stop —
and "عوض بشه، بزن آهنگ بعدی" went down the play/search fast path and played a
video whose title happened to contain the residue words. Everything here drives
the real relay through `test_live_harness` with provider-shaped events; the
tenant endpoint is faked at `_control_media_direct`/`_vps_api`, coded against
the spec contract (`action ∈ next|previous|stop|pause`, response
`{ok, reason ∈ stopped|paused|nothing_playing|unacknowledged|delivery_failed|error}`).
"""

import asyncio
import json
import pathlib

import pytest

import test_live_harness as H
from test_live_repair_contract import _terminal, _wait_until
from app.services import live_voice_protocol as live
from app.services.live_voice_protocol import (
    classify_followup,
    detect_media_control,
    relay_line,
)


FIXTURES = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "media-command-fixtures.json")
    .read_text(encoding="utf-8")
)

#: The exact production utterance (V1 frame f047, relay log U:4 chars=22).
PROD_NEXT = "عوض بشه، بزن آهنگ بعدی"
#: The exact production stop text (V2 t-212, U:81/U:82 chars=11): the
#: recognizer dropped «کن».
PROD_STOP = "آهنگ رو قطع"
STOP = "آهنگ رو قطع کن"

_BASE_FEATURES = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
]


def transport_config():
    return H.config(features=_BASE_FEATURES + ["media_transport"])


def _spoken(provider) -> list[dict]:
    return provider.of("session.commentary.append")


def _spoken_text(provider) -> str:
    return " ".join(e["content"] for e in _spoken(provider))


def _line(key, lang, **fmt):
    """`relay_line`, refusing the empty string an unknown key returns — an
    `in` assertion against "" is vacuously true."""

    text = relay_line(key, lang, **fmt)
    assert text, f"relay line {key!r} is missing"
    return text


# ── the closed grammar ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "case", FIXTURES, ids=[f"{i}-{c['expected']}" for i, c in enumerate(FIXTURES)],
)
def test_media_command_grammar_matches_the_shared_fixtures(case):
    assert detect_media_control(case["text"]) == case["expected"], case


def test_the_fixtures_carry_the_production_strings_and_the_spec_negatives():
    table = {c["text"]: c["expected"] for c in FIXTURES}
    assert table[PROD_NEXT] == "next"
    assert table[PROD_STOP] == "stop"
    assert table["آهنگ رو قطع کن"] == "stop"
    for negative in (
        "مرحله بعدی رو بزن", "هفته بعدی", "the next song is better",
        "stop the search", "تماس رو قطع کن",
    ):
        assert table[negative] is None, negative


@pytest.mark.parametrize(
    "text", [c["text"] for c in FIXTURES if c["expected"] in {"stop", "pause"}],
)
def test_a_music_stop_never_classifies_as_a_task_cancel(text):
    """Stopping the music must never be read as "cancel the running task" —
    the two are different objects, and only an explicit cancel may kill work.

    Judged the way `begin()` decides (R2 fix round, F25): the closed media
    grammar is consulted FIRST and a bare media command is `media_control`
    whatever the lexical classifier says. "stop that music" OPENS with the
    anchored cancel phrase "stop that", so `classify_followup` alone says
    cancel for it — which is exactly how it used to kill the running research
    while the song played on. The executed guard is
    test_live_media_backstop.py::test_stop_that_music_stops_the_music_and_keeps_the_research.
    """

    running = "search the web for the history of jazz"
    verdict = (
        "media_control" if detect_media_control(text) else classify_followup(text, running)
    )
    assert verdict != "cancel"


@pytest.mark.parametrize("text", live._CANCEL_PHRASES)
def test_no_task_cancel_phrase_is_a_media_command(text):
    assert detect_media_control(text) is None


# ── wording: progressive, and only what was confirmed ─────────────────

def test_play_and_next_lines_are_progressive_not_claims_of_audio():
    en = relay_line("media_playing", "en", title="Halo")
    fa = relay_line("media_playing", "fa", title="Halo")
    assert en.startswith("Starting") and "Halo" in en
    assert not en.startswith("Playing")
    assert "در حال پخش است" not in fa and "Halo" in fa
    assert relay_line("media_next", "en", title="Halo").startswith("Switching to")
    assert not relay_line("media_next", "en", title="Halo").startswith("Skipped")
    for key in (
        "media_stopped", "media_paused", "media_stop_unconfirmed",
        "media_control_failed", "media_start_failed",
    ):
        assert relay_line(key, "en") and relay_line(key, "fa"), key
        assert relay_line(key, "en") != relay_line(key, "fa"), key
    # The unconfirmed line must not say the music stopped.
    assert "stopped the music" not in relay_line("media_stop_unconfirmed", "en").lower()


@pytest.mark.asyncio
async def test_the_direct_play_result_never_says_the_track_is_already_audible(monkeypatch):
    from app.api import ws_realtime as rt

    async def vps(_user_id):
        return ("http://agent.invalid", "key")

    async def vps_api(*_args, **_kwargs):
        return {"ok": True, "title": "Halo", "video_id": "v1"}

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)
    text, media = await rt._play_media_direct("user-1", "halo")
    assert media and media["video_id"] == "v1"
    assert "already audible" not in text
    assert "Now playing" not in text
    assert "Halo" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["next", "previous", "stop", "pause"])
async def test_the_relay_transport_forwards_every_allowlisted_action(monkeypatch, action):
    from app.api import ws_realtime as rt

    bodies = []

    async def vps(_user_id):
        return ("http://agent.invalid", "key")

    async def vps_api(_url, _key, method, path, json_body=None, **_kwargs):
        bodies.append((method, path, json_body))
        reason = {"stop": "stopped", "pause": "paused"}.get(action, "executed")
        return {"ok": True, "reason": reason}

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)
    out = await rt._control_media_direct("user-1", action)
    assert bodies == [(
        "POST", "/api/v1/internal/media-control",
        {"user_id": "user-1", "action": action, "channel": "app"},
    )]
    assert out["ok"] is True and out["action"] == action
    assert out["reason"] == {"stop": "stopped", "pause": "paused"}.get(action, "executed")


@pytest.mark.asyncio
async def test_the_relay_transport_refuses_anything_outside_the_allowlist(monkeypatch):
    from app.api import ws_realtime as rt

    async def boom(*_args, **_kwargs):
        raise AssertionError("an unknown action must never reach the tenant")

    monkeypatch.setattr(rt, "_get_vps_info", boom)
    out = await rt._control_media_direct("user-1", "rewind")
    assert out == {"ok": False, "action": "rewind", "reason": "unsupported_action"}


# ── negotiation ───────────────────────────────────────────────────────

async def _start_only(config_frame, monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    client = H.FakeClient([config_frame, 0.05, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
    )
    await H.run_relay(client, provider, timeout=5)
    return client, provider


@pytest.mark.asyncio
async def test_ready_advertises_media_transport(monkeypatch):
    client, _provider = await _start_only(H.config(), monkeypatch)
    assert client.of("ready")[0]["capabilities"]["media_transport"] is True


@pytest.mark.asyncio
async def test_the_capability_line_adds_pause_and_stop_only_when_negotiated(monkeypatch):
    _client, provider = await _start_only(transport_config(), monkeypatch)
    negotiated = provider.of("session.start")[0]["session"]["instructions"]
    assert "pause or stop playback" in negotiated

    _client, provider = await _start_only(H.config(), monkeypatch)
    legacy = provider.of("session.start")[0]["session"]["instructions"]
    assert "pause or stop playback" not in legacy
    # The un-negotiated session keeps the reviewed line word for word (C3).
    assert (
        "- Music and audio: start playback, or move to the next/previous track "
        "on the user's device.\n"
    ) in legacy


# ── the harness: stop/pause/next reach the tenant control ─────────────

def _single_turn(text, delegation_id="d1", start=0, end=350):
    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(text, start, end))
            provider.push(H.delegation(delegation_id, end + 50))
        elif event["type"] == "session.close":
            provider.push(H.closed())
    return on_send


def _done(client, delegation_id="d1"):
    return any(
        _terminal(client, delegation_id, phase)
        for phase in ("completed", "failed", "cancelled")
    )


@pytest.mark.asyncio
async def test_the_production_next_utterance_takes_the_next_control_not_a_search(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    plays, controls, thinks = [], [], []

    async def play(_u, query, variety=False):
        plays.append(query)
        return "Starting X.", {"type": "youtube", "video_id": "v1", "title": "X"}

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "executed", "title": "Bahaneh"}

    async def think(*a, **k):
        thinks.append(a)
        return "must not run", "m"

    H.patch_relay(monkeypatch, play=play, control=control, think=think)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([H.config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn(PROD_NEXT), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-media-prod-next")

    assert plays == [], f"fast path searched YouTube for {plays!r}"
    assert thinks == []
    assert controls == ["next"]
    assert _terminal(client, "d1")
    assert _line("media_next", "fa", title="Bahaneh") in _spoken_text(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [PROD_STOP, STOP])
async def test_a_negotiated_stop_is_a_correlated_control_not_an_agent_turn(monkeypatch, text):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks, controls = [], []

    async def think(*a, **k):
        thinks.append(a)
        return "چشم.", "m"

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, think=think, control=control)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([transport_config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn(text), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-media-stop")

    assert thinks == [], "stop went to a full agent turn, which has no stop tool"
    assert controls == ["stop"]
    frames = client.of("media_control")
    assert [f["status"] for f in frames] == ["requested", "executed"]
    assert all(f["action"] == "stop" for f in frames)
    assert frames[0]["control_id"] == frames[1]["control_id"]
    assert frames[1]["ok"] is True and frames[1]["outcome"] == "ok"
    assert frames[1]["reason"] == "stopped"
    assert _terminal(client, "d1")
    assert _line("media_stopped", "fa") in _spoken_text(provider)


@pytest.mark.asyncio
async def test_a_negotiated_pause_is_a_correlated_control(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    controls = []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "paused"}

    H.patch_relay(monkeypatch, control=control)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([transport_config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn("pause the music"), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-media-pause")

    assert controls == ["pause"]
    frames = client.of("media_control")
    assert [f["action"] for f in frames] == ["pause", "pause"]
    assert _line("media_paused", "en") in _spoken_text(provider)
    assert _line("media_stopped", "en") not in _spoken_text(provider)


@pytest.mark.asyncio
async def test_an_unnegotiated_client_never_receives_a_stop_action(monkeypatch):
    """TF132's tracker drops any media_control action but next/previous; the
    contract forbids new enum values to a client that did not negotiate them.
    The lifecycle (task + tool rows + spoken outcome) still runs."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    controls, thinks = [], []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    async def think(*a, **k):
        thinks.append(a)
        return "must not run", "m"

    H.patch_relay(monkeypatch, control=control, think=think)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([H.config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn(STOP), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-media-legacy-stop")

    assert controls == ["stop"] and thinks == []
    assert not [
        f for f in client.frames
        if f.get("type") == "media_control" or f.get("action") in {"stop", "pause"}
    ]
    assert _terminal(client, "d1")
    started = [f for f in client.of("tool_call.started") if f.get("name") == "media_control"]
    assert started and started[0]["task_id"] == "d1"


@pytest.mark.asyncio
async def test_stop_executes_while_two_long_tasks_hold_every_slot(monkeypatch):
    """A stop must not wait 5-16 s behind two research turns in the capacity
    queue, and it must not cancel either of them."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    release = asyncio.Event()
    started: list[str] = []
    cancelled: list[str] = []
    controls: list[str] = []
    provider_box = {}

    async def think(_user_id, _task, _session_id, **kwargs):
        started.append(kwargs["delegation_id"])
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(kwargs["delegation_id"])
            raise
        return "A long answer.", "m"

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, think=think, control=control)

    def on_send(provider, event):
        provider_box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("find the history of jazz in new orleans", 0, 400))
            provider.push(H.delegation("d1", 450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    observed = {}

    async def script():
        provider = provider_box["p"]
        await _wait_until(lambda: "d1" in started, label="d1 running")
        provider.push(H.user_delta("what is the weather in paris tomorrow", 2000, 2400))
        provider.push(H.delegation("d2", 2450))
        await _wait_until(lambda: "d2" in started, label="d2 running")
        provider.push(H.user_delta(STOP, 4000, 4400))
        provider.push(H.delegation("d3", 4450))
        await _wait_until(lambda: _terminal(client, "d3"), label="stop executed")
        observed["controls_before_release"] = list(controls)
        observed["d1_done_before"] = _done(client, "d1")
        observed["d2_done_before"] = _done(client, "d2")
        release.set()
        await _wait_until(
            lambda: _terminal(client, "d1") and _terminal(client, "d2"),
            label="long tasks finish",
        )

    client = H.FakeClient([transport_config(), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=9, db_session_id="db-media-capacity")

    assert observed["controls_before_release"] == ["stop"]
    assert observed["d1_done_before"] is False and observed["d2_done_before"] is False
    assert cancelled == []
    assert not any(
        f.get("phase") == "pending" for f in client.of("delegation")
        if f.get("delegation_id") == "d3"
    ), "the stop waited in the capacity queue"
    assert sorted(started) == ["d1", "d2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_text", ["stop the search", "کنسلش کن"])
async def test_a_task_cancel_still_cancels_and_never_stops_the_music(monkeypatch, cancel_text):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    release = asyncio.Event()
    started: list[str] = []
    controls: list[str] = []
    provider_box = {}

    async def think(_user_id, _task, _session_id, **kwargs):
        started.append(kwargs["delegation_id"])
        await release.wait()
        return "never", "m"

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, think=think, control=control)

    def on_send(provider, event):
        provider_box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("search the web for the history of jazz", 0, 400))
            provider.push(H.delegation("d1", 450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def script():
        await _wait_until(lambda: "d1" in started, label="d1 running")
        provider_box["p"].push(H.user_delta(cancel_text, 2000, 2300))
        provider_box["p"].push(H.delegation("d2", 2350))
        await _wait_until(lambda: _terminal(client, "d1", "cancelled"), label="d1 cancelled")

    client = H.FakeClient([transport_config(), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    try:
        await H.run_relay(client, provider, timeout=8, db_session_id="db-media-cancel")
    finally:
        release.set()

    assert controls == []
    assert client.of("media_control") == []
    assert started == ["d1"]


@pytest.mark.asyncio
async def test_stop_wording_waits_for_the_confirmed_outcome(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    gate = asyncio.Event()
    called = asyncio.Event()

    async def control(_u, action):
        called.set()
        await gate.wait()
        return {"ok": True, "action": action, "reason": "stopped"}

    H.patch_relay(monkeypatch, control=control)
    observed = {}

    async def script():
        await asyncio.wait_for(called.wait(), timeout=3)
        await asyncio.sleep(0.1)
        observed["before"] = _spoken_text(provider)
        observed["executed_before"] = [
            f for f in client.of("media_control") if f.get("status") != "requested"
        ]
        gate.set()
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([transport_config(), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn(STOP), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-media-wording")

    stopped = _line("media_stopped", "fa")
    assert stopped not in observed["before"]
    assert observed["executed_before"] == []
    assert stopped in _spoken_text(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "line_key"),
    [
        ("unacknowledged", "media_stop_unconfirmed"),
        # F27: an idle device answered, another relevant one stayed silent —
        # the station is off, but "nothing was playing" is not proven.
        ("partially_confirmed", "media_stop_partial"),
        ("delivery_failed", "media_control_failed"),
        ("error", "media_control_failed"),
        ("agent_error", "media_control_failed"),
    ],
)
async def test_an_unconfirmed_or_failed_stop_says_so_and_never_claims_success(
    monkeypatch, reason, line_key,
):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def control(_u, action):
        return {"ok": False, "action": action, "reason": reason}

    H.patch_relay(monkeypatch, control=control)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([transport_config(), wait, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn(STOP), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id=f"db-media-{reason}")

    spoken = _spoken_text(provider)
    assert _line(line_key, "fa") in spoken
    assert _line("media_stopped", "fa") not in spoken
    # One outcome line, not the generic retry line on top of it.
    assert _line("delegation_failed", "fa") not in spoken
    assert _terminal(client, "d1", "failed")
    assert not _terminal(client, "d1", "completed")
    frames = client.of("media_control")
    assert [f["status"] for f in frames] == ["requested", "error"]
    assert frames[1]["ok"] is False and frames[1]["outcome"] == "error"
    assert frames[1]["reason"] == reason


def test_a_partially_confirmed_verdict_never_claims_nothing_was_playing():
    """F27: the tenant's `partially_confirmed` (the answering devices were idle,
    another relevant device never answered) is worded as exactly that — never
    "nothing was playing", never "stopped", never a generic failure."""

    from app.services.live_voice_protocol import media_control_line

    for lang in ("en", "fa"):
        key, line = media_control_line("stop", False, "partially_confirmed", lang)
        assert key == "media_stop_partial"
        assert line and line != _line("media_nothing_playing", lang)
        key, line = media_control_line("pause", False, "partially_confirmed", lang)
        assert key == "media_pause_partial"
        assert line and line != _line("media_control_failed", lang)
    # A next/previous never produces this reason; if it did, it is a failure.
    assert media_control_line("next", False, "partially_confirmed", "en")[0] == "media_control_failed"


@pytest.mark.asyncio
async def test_a_failed_next_speaks_the_media_failure_line(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def control(_u, action):
        return {"ok": False, "action": action, "reason": "error"}

    H.patch_relay(monkeypatch, control=control)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([H.config(), wait, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn("next song please"), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-media-next-fail")

    spoken = _spoken_text(provider)
    assert _line("media_control_failed", "en") in spoken
    assert _line("delegation_failed", "en") not in spoken
    assert _terminal(client, "d1", "failed")


# ── what the phone reports back ───────────────────────────────────────

@pytest.mark.asyncio
async def test_now_playing_stopped_tells_the_model_quietly(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    client = H.FakeClient([
        transport_config(), 0.05,
        {"type": "now_playing", "title": "", "state": "stopped", "source": "media_x"},
        0.1, {"type": "stop"},
    ])
    provider = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
    )
    await H.run_relay(client, provider, timeout=5)

    notes = [e["content"] for e in provider.of("session.thinking.append")]
    assert any("stopped the music" in n and "nothing is playing" in n for n in notes), notes
    # A context correction, not a turn: nothing that asks the model to speak.
    assert _spoken(provider) == []
    assert provider.of("response.create") == []


def test_the_realtime_relay_builds_the_same_quiet_notes():
    from app.api.ws_realtime import _now_playing_note

    stopped = _now_playing_note("", "stopped")
    assert stopped and "stopped the music" in stopped and "nothing is playing" in stopped
    moved = _now_playing_note("Halo", "")
    assert moved and "Halo" in moved
    assert _now_playing_note("", "") is None
    assert _now_playing_note("", "playing") is None


@pytest.mark.asyncio
async def test_a_failed_announced_track_gets_one_spoken_correction(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def play(_u, _query, _variety=False):
        return "Starting Halo on the user's device.", {
            "type": "youtube", "video_id": "v1", "title": "Halo",
        }

    H.patch_relay(monkeypatch, play=play)
    starting = _line("media_playing", "en", title="Halo")
    correction = _line("media_start_failed", "en")
    observed = {}

    async def wait_started():
        await _wait_until(lambda: _done(client), label="d1 terminal")
        await _wait_until(lambda: starting in _spoken_text(provider), label="starting line")
        observed["before"] = _spoken_text(provider)

    async def wait_correction():
        await _wait_until(lambda: correction in _spoken_text(provider), label="correction")

    client = H.FakeClient([
        H.config(), wait_started,
        {"type": "playback_failed", "video_id": "v1", "reason": "load_error"},
        wait_correction,
        {"type": "playback_failed", "video_id": "v1", "reason": "load_error"},
        {"type": "playback_failed", "video_id": "v-other", "reason": "load_error"},
        0.3, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_single_turn("play halo by beyonce"), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-media-start-failed")

    assert correction not in observed["before"]
    corrections = [e for e in _spoken(provider) if e["content"] == correction]
    assert len(corrections) == 1, "one correction per announced track, never for others"
    assert corrections[0]["delegation_id"] == "d1"
    # The model-facing fact still lands for every failure.
    notes = [e["content"] for e in provider.of("session.thinking.append")]
    assert sum("Playback did not start" in n for n in notes) == 3


# ══════════════════════════════════════════════════════════════════════
# D-R4 (2026-10-06 call): «استاپش کن» while a song played
# ══════════════════════════════════════════════════════════════════════
#
# The recorded «استاپش کن» named no media noun, so the closed grammar above
# declined it (it is still declined there — the context-free rows keep their
# `null`), it became a delegation to an agent that has no tool that can stop
# anything, and the agent apologised over a song that played on.  While the
# phone reports playback the nounless form IS the command: the shared fixture
# rows with a `while_playing` key are that reading's table.

WHILE_PLAYING = [c for c in FIXTURES if "while_playing" in c]


def test_the_while_playing_table_holds_the_recorded_string_and_its_negatives():
    table = {c["text"]: c["while_playing"] for c in WHILE_PLAYING}
    assert table["استاپش کن"] == "stop"
    for negative in ("قطع شد", "قطع نکن", "صبر کن", "hold on", "stop the search"):
        assert table[negative] is None, negative
    # The context-free grammar is unchanged for every bare form.
    bare = {c["text"]: c["expected"] for c in WHILE_PLAYING}
    for text in ("استاپش کن", "پازش کن", "نگهش دار", "بسه", "kill it", "stop it"):
        assert bare[text] is None, text
    assert bare["آهنگو استاپش کن"] == "stop"


@pytest.mark.parametrize(
    "case", WHILE_PLAYING, ids=[f"{i}-{c['while_playing']}" for i, c in enumerate(WHILE_PLAYING)],
)
def test_media_command_while_playing_matches_the_shared_fixtures(case):
    assert live.media_command_while_playing(case["text"]) == case["while_playing"], case
    # …and the context-free grammar never moves with it.
    assert detect_media_control(case["text"]) == case["expected"], case


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "action", "lang"), [
    ("استاپش کن", "stop", "fa"),
    ("stop it", "stop", "en"),
    ("pause", "pause", "en"),
])
async def test_a_bare_command_while_playing_is_a_correlated_control_not_an_agent_turn(
    monkeypatch, text, action, lang,
):
    """The delegated path: GPT-Live DID delegate «استاپش کن» (299deaad on the
    recording).  The session verdict rides the task (`media_action`), so the
    control fast path executes it — no agent turn, which has nothing to stop
    with — and the R6-4 ledger records a halt, not a run."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks, controls = [], []
    box = {}

    async def think(*a, **k):
        thinks.append(a)
        return "ببخشید، از اینجا نمی‌تونم متوقفش کنم.", "m"

    async def control(_u, act):
        controls.append(act)
        return {"ok": True, "action": act, "reason": "stopped" if act == "stop" else "paused"}

    H.patch_relay(monkeypatch, think=think, control=control)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def speak():
        box["p"].push(H.user_delta(text, 1000, 1400))
        box["p"].push(H.delegation("d1", 1450))
        await _wait_until(lambda: _done(client), label="d1 terminal")
        session = box.get("session")
        intents = getattr(session, "media_intents", None) or {}
        box["kinds"] = sorted({i.kind for i in intents.values()})

    original_begin = live._LiveSession.begin

    async def spy_begin(self, task, utterance):
        box["session"] = self
        return await original_begin(self, task, utterance)

    monkeypatch.setattr(live._LiveSession, "begin", spy_begin)
    client = H.FakeClient([
        transport_config(), {"type": "now_playing", "title": "Hotline Bling - Drake"},
        0.2, speak, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id=f"db-media-playing-{action}")

    assert thinks == [], "the bare command went to a full agent turn, which has no stop tool"
    assert controls == [action]
    frames = client.of("media_control")
    assert [f["action"] for f in frames] == [action, action]
    key = "media_stopped" if action == "stop" else "media_paused"
    assert _line(key, lang) in _spoken_text(provider)
    # R6-4: the ledger reads the task's verdict — a halt, never an agent `run`.
    assert box["kinds"] == [action], box["kinds"]
    # D-R4: the model was told at the command, quietly, that the app does it.
    told = [e for e in provider.of("session.thinking.append")
            if _line("media_command_started", lang, request=text) in str(e.get("content") or "")]
    assert told and all(e.get("delegation_id") is None for e in told)


@pytest.mark.asyncio
async def test_a_bare_command_with_nothing_playing_is_still_the_agents(monkeypatch):
    """Control: without `now_playing` the same «استاپش کن» is not a media
    command (gate term 1) — the agent gets it, as before."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks, controls = [], []

    async def think(*a, **k):
        thinks.append(a)
        return "چیزی پخش نمی‌شه.", "m"

    async def control(_u, act):
        controls.append(act)
        return {"ok": True, "action": act, "reason": "stopped"}

    H.patch_relay(monkeypatch, think=think, control=control)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([transport_config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn("استاپش کن"), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-media-playing-none")

    assert controls == []
    assert len(thinks) == 1


@pytest.mark.parametrize(("text", "expected"), [
    ("resume", True), ("resume the music", True), ("آهنگو ادامه بده", True),
    ("play it again", True), ("unpause", True), ("دوباره پخشش کن", True),
    # Never the bare continuer — that is the caller's backchannel.
    ("ادامه بده", False), ("continue", False), ("go on", False),
    ("resume the search", False), ("play again", False),
])
def test_an_explicit_resume_is_read_and_the_bare_continuer_is_not(text, expected):
    assert live.media_resume_request(text) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "lang"), [
    ("آهنگو ادامه بده", "fa"), ("resume the music", "en"),
])
async def test_an_explicit_resume_over_a_paused_track_is_the_relays_resume(monkeypatch, text, lang):
    """Phase D2 (tests-guards-5): `playing_command`'s explicit-resume branch
    (`media_resume_request` + a titled PAUSED `now_playing`) was pinned by
    nothing.  The closed grammar has no resume head, so without that branch
    a delegated «آهنگو ادامه بده» / "resume the music" is a full agent turn.
    With it the relay runs the tenant's resume itself — never correlated on
    the wire (`resume` is in no shipped client's closed set) — and speaks the
    honest verdict."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    controls, thinks = [], []

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "resumed", "title": "Halo"}

    async def think(*a, **k):
        thinks.append(k.get("delegation_id"))
        return "Resuming it.", "m"

    H.patch_relay(monkeypatch, control=control, think=think)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def ask():
        box["p"].push(H.user_delta(text, 0, 400))
        box["p"].push(H.delegation("d1", 450))
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([
        transport_config(),
        {"type": "now_playing", "title": "Halo - Beyonce", "state": "paused"},
        0.2, ask, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id=f"db-media-resume-{lang}")

    assert controls == ["resume"], controls
    assert thinks == [], "the resume went to a full agent turn"
    started = [f for f in client.of("tool_call.started") if f.get("name") == "media_control"]
    assert [f.get("detail") for f in started] == ["resume"], started
    assert client.of("media_control") == [], "resume is never a correlated frame"
    assert _line("media_resumed", lang, title="Halo") in _spoken_text(provider)
    assert _terminal(client, "d1")


def test_the_resume_verdict_has_its_own_honest_lines():
    key, line = live.media_control_line("resume", True, "resumed", "fa", "Halo")
    assert key == "media_resumed" and "Halo" in line
    key, line = live.media_control_line("resume", False, "nothing_to_resume", "en")
    assert key == "media_nothing_to_resume" and line
    # A tenant image that predates `resume` refuses it: an honest failure.
    key, _line = live.media_control_line("resume", False, "unsupported_action", "en")
    assert key == "media_control_failed"


@pytest.mark.asyncio
async def test_d_t1_the_agents_own_media_control_tool_is_recorded_not_voiced_twice(monkeypatch, caplog):
    """D-R4 / D-T1: a delegated agent turn that ran its own `media_control`
    tool reports `tool.end.media`; the relay records it (log + counter) and the
    spoken outcome stays the agent's own answer — no relay verdict line."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await relay.on_event({"type": "tool.start", "call_id": "m1", "name": "media_control",
                              "args": {"action": "stop"}})
        await relay.on_event({
            "type": "tool.end", "call_id": "m1", "name": "media_control", "ok": True,
            "elapsed_ms": 30,
            "media": {"action": "stop", "reason": "stopped", "ok": True, "title": "Halo"},
        })
        return "Done — the music is off now.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([transport_config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_single_turn("can you deal with the music for me"), auto_ack=True,
    )
    with caplog.at_level(logging.INFO):
        await H.run_relay(client, provider, timeout=7, db_session_id="db-media-tool-end")

    assert "action=stop ok=True reason=stopped source=tool" in caplog.text
    assert ("live_media_control", {"action": "stop", "ok": True, "source": "tool"}) in counters
    # The relay did not run a control of its own, and spoke no verdict line.
    assert client.of("media_control") == []
    assert _line("media_stopped", "en") not in _spoken_text(provider)


# ── Voice round 2026-10-07 (D-R1 §4.8): one media act, one row ─────────

def _rows(saves, prefix):
    return [s for s in saves if str(s.get("assistant_ref") or "").startswith(prefix)]


@pytest.mark.asyncio
@pytest.mark.parametrize("window_s", [0.0, 6.0])
async def test_a_media_lines_claim_cannot_capture_a_later_unrelated_epoch(monkeypatch, window_s):
    """10-07: the stop line «موزیک رو قطع کردم.» was never voiced, its claim
    had no deadline, and 12 s later it bound the model's first word to the
    NEXT ask — the stop row says `epoch: 23`.  A media line's claim is
    bounded like every relay line now: a later reply is its own row.

    Phase D (words-proof-9): ALSO at the production fold window (6 s) — the
    reply lands 1.2 s after the line, inside the window, and the fold used to
    take it (voice fields only: its words in no row of the day).  An epoch
    with no media word, no word of the line and more than three words of its
    own is not about the act."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=2.0, voice_live_media_fold_window_s=window_s,
    )

    async def control(_u, action):
        return {"ok": True, "action": action, "reason": "stopped"}

    recorded = H.patch_relay(monkeypatch, control=control)
    box = {}
    single = _single_turn(STOP)

    def on_send(provider, event):
        box["p"] = provider
        single(provider, event)

    later = "By the way, the weather is lovely today."

    async def later_reply():
        await _wait_until(lambda: _done(client), label="d1 terminal")
        await asyncio.sleep(1.2)      # past the line's speak budget (0.5 s)
        box["p"].push(H.out_text(later, 5000, 5600))
        box["p"].push(H.out_audio("LATER"))
        await asyncio.sleep(0.4)

    client = H.FakeClient([transport_config(), later_reply, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=f"db-media-claim-bound-{window_s}")

    control_rows = _rows(recorded["saves"], "live-media-control:")
    assert control_rows, recorded["saves"]
    assert not [s for s in control_rows if "epoch" in (s.get("assistant_voice") or {})], (
        "a later, unrelated reply was folded onto the stop row"
    )
    assert later in [s.get("assistant_text") for s in _rows(recorded["saves"], "live-output:")]


@pytest.mark.parametrize(("text", "folds"), [
    ("قطعش کردم.", True),                                   # the 10-07 stop, in its own words
    ("دارم یه آهنگ از دریک می‌ذارم", True),                  # a media word
    ("Here it is, enjoy!", True),                           # short
    ("Starting Hotline Bling now, enjoy the track.", True),  # a word of the line
    ("By the way, the weather is lovely today.", False),
    ("Also, your meeting with Sara is at four o'clock.", False),
])
def test_only_words_about_the_media_act_fold_onto_its_row(text, folds):
    """Phase D (words-proof-9): `fold_media_window`'s relatedness test."""

    claim = {"payload": {
        "assistant_text": "Starting Hotline Bling.",
        "media": {"type": "youtube", "video_id": "v1", "title": "Hotline Bling"},
    }}
    assert live.media_fold_relates(text, claim) is folds


@pytest.mark.asyncio
@pytest.mark.parametrize("window_s", [6.0, 0.0])
async def test_a_media_act_is_one_row(monkeypatch, caplog, window_s):
    """10-07: one stop, two rows — the model's own «قطعش کردم.» (born before
    the line's ack) and the relay's line.  An unbound epoch born inside a
    media line's `voice_live_media_fold_window_s` folds onto its row."""

    import logging

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=2.0, voice_live_media_fold_window_s=window_s,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    async def control(_u, action):
        return {"ok": True, "action": action, "reason": "stopped"}

    recorded = H.patch_relay(monkeypatch, control=control)
    single = _single_turn(STOP)

    def on_send(provider, event):
        single(provider, event)
        if event["type"] == "session.commentary.append":
            line = event["content"]

            async def two_epochs():
                # The model's own words first, then the line itself.
                provider.push(H.out_text("قطعش کردم.", 5000, 5300))
                provider.push(H.out_audio("OWN"))
                await asyncio.sleep(0.3)
                provider.push(H.out_text(line, 6000, 6600))
                provider.push(H.out_audio("LINE"))
            asyncio.get_running_loop().create_task(two_epochs())

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")
        await asyncio.sleep(1.0)

    client = H.FakeClient([transport_config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=f"db-media-one-row-{window_s}")

    own_rows = [s.get("assistant_text") for s in _rows(recorded["saves"], "live-output:")]
    folded = [r.getMessage() for r in caplog.records if "media line folded" in r.getMessage()]
    assert _rows(recorded["saves"], "live-media-control:")
    if window_s:
        assert own_rows == [], own_rows
        assert folded and "reason=send_window" in folded[0], folded
    else:
        assert len(own_rows) == 1, own_rows
        assert folded == []


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["told_first", "terminal_first"])
async def test_a_play_line_told_before_its_terminal_frame_is_reported_heard(
    monkeypatch, caplog, order,
):
    """Ruling 35 (F3, pre-existing at 985e611e): a delegated play's relay line
    can be voiced BEFORE the task's `completed` frame — the cross-layer tape
    put `response_text «Starting Hotline Bling.»` at index 17 and `completed`
    at 20.  The line is then inside the card's delivery floor, so the phone's
    F41 hold (`deliveryPending`) kept the play card "writing the answer" for
    45 s over the music.  The relay now says the line was heard
    (`completed{spoken:true}`, the merged group's frame) — in that ORDER: told,
    then the verdict.  A line voiced AFTER the terminal frame needs no verdict:
    the phone sees it arrive after its floor and lets go by itself."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    plays: list = []

    async def play(_u, query, _variety=False):
        plays.append(query)
        return "Starting Hotline Bling.", {
            "type": "youtube", "video_id": "v1", "title": "Hotline Bling",
        }

    H.patch_relay(monkeypatch, play=play)
    if order == "told_first":
        # The fast path's teardown outlives the line's first words (the tape's
        # order): the epoch opens while the delegation is still finishing.
        finish = live._LiveSession.finish_delegation

        async def slow_finish(self, *a, **k):
            await asyncio.sleep(0.3)
            return await finish(self, *a, **k)

        monkeypatch.setattr(live._LiveSession, "finish_delegation", slow_finish)
    single = _single_turn("play hotline bling by drake")

    def on_send(provider, event):
        single(provider, event)
        if event["type"] == "session.commentary.append":
            line = event["content"]

            async def tell():
                if order == "terminal_first":
                    await _wait_until(lambda: _done(client), label="d1 terminal")
                provider.push(H.out_text(line, 5000, 5600))
                provider.push(H.out_audio("LINE"))
            asyncio.get_running_loop().create_task(tell())

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")
        await asyncio.sleep(0.8)

    client = H.FakeClient([transport_config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id=f"db-play-heard-{order}")

    assert plays, "the play never took the fast path"
    line = _line("media_playing", "en", title="Hotline Bling")
    seq = []
    for frame in client.frames:
        if frame.get("type") == "response_text" and line[:12] in str(frame.get("text") or ""):
            seq.append("told")
        elif frame.get("type") == "delegation" and frame.get("delegation_id") == "d1" \
                and frame.get("phase") == "completed":
            seq.append("completed{spoken:true}" if frame.get("spoken") is True else "completed")
    seq = [k for i, k in enumerate(seq) if i == 0 or k != seq[i - 1]]
    heard = [r.getMessage() for r in caplog.records if "media line heard" in r.getMessage()]
    if order == "told_first":
        assert seq == ["told", "completed", "completed{spoken:true}"], seq
        assert heard and "before=terminal" in heard[0], heard
    else:
        assert seq == ["completed", "told"], seq
        assert heard == [], heard


# ══════════════════════════════════════════════════════════════════════
# Voice round 2026-10-07 (D-R4 (4)): "what's playing?" answered by the relay
# ══════════════════════════════════════════════════════════════════════
# «اسم صبحش آهنگ» (4:37 in the recording) opened a card and a 3.9 s agent turn
# and was answered 5.3 s after the turn closed, for a fact the relay already
# held: the phone's own `now_playing` report.  Hook B answers it ~0.4 s after
# the close with no delegation at all; Hook A answers a delegation of it.

#: What the app reports: "{song} — {artist}" (`nowPlayingLabel`, an EM dash).
NP_TITLE = "Hotline Bling — Drake"


@pytest.mark.parametrize("text", [
    "این آهنگ چیه", "اسم این آهنگ چیه", "اسم آهنگ چیه", "اسمش چیه", "اسم صبحش آهنگ",
    "این آهنگ اسمش چیه", "چی داره پخش میشه", "خواننده این آهنگ کیه",
    "what's this song", "what's playing", "what song is this",
    "what is this song called", "who sings this song", "what's the name of this song",
])
def test_whats_playing_questions_are_read(text):
    assert live.media_status_question(text), text


@pytest.mark.parametrize("text", [
    "این آهنگ رو پخش کن", "اسم یه آهنگ از ابی رو بگو", "آهنگ قبلی چی بود", "اسم استادش چیه",
    "چرا این آهنگو گذاشتی", "what's the best song by drake", "what's this song about",
    "این آهنگ چه سالی اومده", "stop this song", "play this song again", "what's the status",
    "چی شد", "what happened",
    # Phase D (search-play-title-5): another track than the one playing —
    # answered with the CURRENT track, and the turn was consumed.
    "اسم آهنگ قبلی", "اسم آهنگ قبلی چیه", "اسم آهنگ بعدی چیه", "آهنگ قبلی اسمش چیه",
    "اسم آهنگ بعدیش چیه", "اسم آهنگ اول", "اسم آهنگ آخر چیه", "اسم آهنگ دیگه چیه",
    "name the next song", "what was the last song",
    # …and the forgiven garble only sits BETWEEN the name and the song noun.
    "اسم آهنگ مدونا چیه", "اسم آهنگ چی بود", "name of madonna song",
    "اسم قبلی آهنگ چیه",    # a referent word is never the forgiven garble
])
def test_what_is_not_a_whats_playing_question(text):
    assert not live.media_status_question(text), text


async def _np_call(
    monkeypatch, text, *, title=NP_TITLE, state=None, delegate=False, reply=None, db,
    enabled=True, before=None, wait_s=1.2,
):
    """`before` = (words, delegation id) asked and finished first; the phone
    reports `title` (and `state`) after it, then the caller asks `text`."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=2.0,
        voice_live_media_status_enabled=enabled,
    )
    thinks: list = []
    plays: list = []

    async def think(*_a, **k):
        thinks.append(k.get("delegation_id"))
        return "پروفسور جفری هینتون.", "m"

    async def play(_u, query, _variety=False):
        plays.append(query)
        return "Starting Hotline Bling.", {
            "type": "youtube", "video_id": "v1", "title": "Hotline Bling",
        }

    H.patch_relay(monkeypatch, think=think, play=play)
    box: dict = {}

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    async def first():
        if before is None:
            return
        p = box["p"]
        p.push(H.user_delta(before[0], 500, 900))
        p.push(H.delegation(before[1], 950))
        await _wait_until(lambda: _terminal(client, before[1], "completed"), label="first task")
        await asyncio.sleep(0.3)

    async def ask():
        p = box["p"]
        p.push(H.user_delta(text, 3000, 3500))
        if delegate:
            p.push(H.delegation("d1", 3550))
        if reply:
            p.push(H.out_text(reply, 3600, 3900))
            p.push(H.out_audio("R"))
        await asyncio.sleep(wait_s)

    frames: list = [transport_config(), 0.2, first]
    if title is not None:
        report = {"type": "now_playing", "title": title}
        if state is not None:
            report["state"] = state
        frames.append(report)
    client = H.FakeClient(frames + [0.2, ask, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=db)
    return client, provider, thinks, plays


def _status_lines(provider, line) -> list[str]:
    texts = [str(e.get("content") or "") for e in provider.of("session.instructions.append")]
    texts += [str(e.get("content") or "") for e in provider.of("session.commentary.append")]
    return [t for t in texts if line in t]


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "lang"), [
    ("اسم صبحش آهنگ", "fa"),              # the recorded garble
    ("what's this song", "en"),
])
async def test_whats_playing_is_answered_from_now_playing_with_no_card(
    monkeypatch, caplog, text, lang,
):
    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    client, provider, thinks, _plays = await _np_call(
        monkeypatch, text, db=f"db-np-close-{lang}",
    )
    line = _line("media_now_playing", lang, song="Hotline Bling", artist="Drake")
    said = [
        str(e.get("content") or "") for e in provider.of("session.instructions.append")
        if line in str(e.get("content") or "")
    ]
    assert said == [_line("relay_task_say", lang, line=line)], said
    assert client.of("delegation") == [], "a card for a fact the relay holds"
    assert thinks == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("media status armed turn_id=" in m and "grace_ms=400" in m for m in messages)
    assert any("media status answered turn_id=" in m and "via=close" in m for m in messages)


@pytest.mark.asyncio
async def test_the_answer_is_said_in_the_language_the_caller_asked_in(monkeypatch):
    """Critic U18: a `relay-status:` id is in no task table, so `speak()`'s
    instruction wrapper used to take the language of the LINE's own script —
    an English question about a Persian title went out wrapped in Persian."""

    _client, provider, _thinks, _plays = await _np_call(
        monkeypatch, "what's this song", title="بهانه — ابی", db="db-np-lang",
    )
    line = _line("media_now_playing", "en", song="بهانه", artist="ابی")
    said = [
        str(e.get("content") or "") for e in provider.of("session.instructions.append")
        if line in str(e.get("content") or "")
    ]
    assert said == [_line("relay_task_say", "en", line=line)], said


@pytest.mark.asyncio
async def test_whats_playing_is_answered_via_a_model_delegation(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    client, provider, thinks, _plays = await _np_call(
        monkeypatch, "این آهنگ چیه", delegate=True, db="db-np-delegation",
    )
    line = _line("media_now_playing", "fa", song="Hotline Bling", artist="Drake")
    spoken = [e for e in provider.of("session.commentary.append") if line in e["content"]]
    assert spoken and spoken[0]["delegation_id"] == "d1", provider.of("session.commentary.append")
    assert thinks == [], "an agent turn for what the phone reported"
    assert [f for f in client.of("delegation") if f.get("phase") == "created"] == []
    assert any(
        "media status answered turn_id=" in r.getMessage() and "via=delegation" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_with_nothing_playing_the_question_goes_where_it_always_went(monkeypatch):
    client, provider, thinks, _plays = await _np_call(
        monkeypatch, "این آهنگ چیه", title=None, delegate=True, db="db-np-nothing",
    )
    assert thinks == ["d1"], thinks
    assert _status_lines(provider, "Hotline Bling") == []


@pytest.mark.asyncio
async def test_a_paused_track_is_not_playing(monkeypatch):
    _client, provider, thinks, _plays = await _np_call(
        monkeypatch, "what's this song", state="paused", delegate=True, db="db-np-paused",
    )
    assert thinks == ["d1"], thinks
    assert _status_lines(provider, "Hotline Bling") == []


@pytest.mark.asyncio
async def test_the_harness_pins_the_relays_now_playing_answer_off(monkeypatch):
    _client, provider, _thinks, _plays = await _np_call(
        monkeypatch, "اسم صبحش آهنگ", enabled=False, db="db-np-off",
    )
    assert _status_lines(provider, "Hotline Bling") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "answered"), [
    ("play hotline bling by drake", True),                       # media in focus
    ("who is the best LLM professor at toronto", False),        # a research answer
], ids=["after_a_play", "after_research"])
async def test_a_bare_pointer_is_about_the_song_only_while_media_is_in_focus(
    monkeypatch, first, answered,
):
    """«اسمش چیه» ("what's it called") names no song: after a play it asks the
    song's name; after a research answer about a professor it does not, and it
    goes to the model/agent as before."""

    _client, provider, thinks, _plays = await _np_call(
        monkeypatch, "اسمش چیه", delegate=True, before=(first, "d0"),
        db=f"db-np-focus-{answered}",
    )
    line = _line("media_now_playing", "fa", song="Hotline Bling", artist="Drake")
    if answered:
        assert _status_lines(provider, line), provider.of("session.commentary.append")
        assert "d1" not in thinks, thinks
    else:
        assert _status_lines(provider, line) == []
        assert "d1" in thinks, thinks


# ── Phase C (ruling 7): Hook B re-checks the model at the moment it SENDS ──

@pytest.mark.asyncio
@pytest.mark.parametrize("model_answered", [True, False], ids=["model_answered", "silent"])
async def test_a_held_now_playing_answer_is_dropped_if_the_model_answered_meanwhile(
    monkeypatch, caplog, model_answered,
):
    """Hook B fires at close + 0.4 s; when a result (or the model's own
    sentence) is being told, `speak` would hold the line and send it later
    with no second look — by then GPT-Live, which has the now-playing note,
    may have answered the question itself.  The re-check is made when the
    line actually goes out."""

    import logging
    from types import SimpleNamespace

    from app.config import settings
    from app.services.live_voice_protocol import OUTPUT_UNKNOWN, _LiveSession

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    session = _LiveSession.__new__(_LiveSession)
    session.settings = settings
    session.user_id = "user-1"
    session.held_speech = []
    session.held_model_since = 0.0
    session.relay_status_langs = {}
    session.media_status_seen = {}
    busy = {"v": True}
    evidence = {"v": None}
    spoken: list = []
    monkeypatch.setattr(session, "may_speak", lambda: True)
    monkeypatch.setattr(session, "delivery_group_busy", lambda: busy["v"])
    monkeypatch.setattr(session, "model_speaking_hold_open", lambda: False)
    monkeypatch.setattr(session, "turn_output_evidence", lambda turn_id, **_k: evidence["v"])
    monkeypatch.setattr(
        session, "media_status_answer", lambda _t: ("Hotline Bling — Drake", "playing"),
    )

    async def speak(delegation_id, text, **kw):
        spoken.append((delegation_id, text, kw.get("parent_user_turn_id")))
        return []

    monkeypatch.setattr(session, "speak", speak)
    session.active_delivery = SimpleNamespace(
        task=SimpleNamespace(delegation_id="d-result"), resolved=False, speaking_epoch=None,
    )
    assert await session.answer_media_status(
        "live-utt:p:7", delegation_id="relay-status:abc", text="what's this song", via="close",
    ) is True
    assert spoken == [] and len(session.held_speech) == 1, "held behind the result"
    # The result is told; meanwhile the model may have answered the question.
    busy["v"] = False
    session.active_delivery = None
    if model_answered:
        evidence["v"] = OUTPUT_UNKNOWN
    await session.flush_held_speech()
    messages = [r.getMessage() for r in caplog.records]
    if model_answered:
        assert spoken == [], spoken
        assert "[LIVE] media status skipped turn_id=live-utt:p:7 reason=answered" in messages
    else:
        assert [(d, p) for d, _t, p in spoken] == [("relay-status:abc", "live-utt:p:7")]
        assert any(m.startswith("[LIVE] media status answered turn_id=live-utt:p:7 via=close")
                   for m in messages), messages
