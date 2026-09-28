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
