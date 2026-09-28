"""R2 second fix round, fx2-misc (spec addendum 2, items 6, 9 and 10 — relay).

* item 6 — a terminal `delegation` failed frame carries the `reason` code the
  relay already knows, so the app's compact failed card can say why;
* item 9 — a play the TENANT reports `superseded` (`{ok: false, reason:
  "superseded"}`: a stop/pause landed after the play began) says no
  "Starting …" line, is never handed to a full agent turn, and ends
  `cancelled`;
* item 10 — the media backstop's verdict row is stamped under the turn it
  answers (`stamp_child`); `tick_loop` backs off after a failing tick instead
  of spinning; `_LANGUAGE_NEG_NEUTRAL` words count only next to a negation;
  the dead `stamp_for` is gone.

Everything that can be is driven through the real relay (`test_live_harness`)
with provider-shaped events; the tenant is faked at its HTTP seam
(`_vps_api`) or at `_control_media_direct`.
"""

import asyncio
import time

import pytest

import test_live_harness as H
from test_live_repair_contract import _delegations, _terminal, _wait_until
from app.config import settings
from app.services import live_voice_protocol as live
from app.services.live_voice_protocol import relay_line


_TRANSPORT = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    "media_transport",
]


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
        for phase in ("completed", "failed", "cancelled", "superseded")
    )


def _terminal_frame(client, delegation_id="d1"):
    frames = [
        f for f in _delegations(client, delegation_id)
        if f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
    ]
    assert frames, f"no terminal frame for {delegation_id}: {client.phases()}"
    return frames[0]


def _spoken_text(provider) -> str:
    return " ".join(
        str(e.get("content") or "")
        for e in provider.sent
        if e.get("type") in {"session.commentary.append", "session.instructions.append"}
    )


# ── item 6: a failed frame says why ───────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, code", [
    (RuntimeError("agent exploded"), "delegation_failed"),
    ("   ", "delegation_empty"),
])
async def test_a_failed_delegation_frame_carries_its_reason_code(monkeypatch, answer, code):
    H.fast_clocks(monkeypatch)

    async def think(*_args, **_kwargs):
        if isinstance(answer, Exception):
            raise answer
        return answer, "m"

    H.patch_relay(monkeypatch, think=think)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([H.config(), wait, {"type": "stop"}])
    await H.run_relay(
        client, H.FakeProvider(on_send=_single_turn("find me a plumber")), timeout=6,
    )

    frame = _terminal_frame(client)
    assert frame["phase"] == "failed" and frame["state"] == "failed", frame
    assert frame.get("reason") == code, frame
    # The separate error frame keeps naming the same kind (unchanged contract).
    assert any(f.get("code") == code for f in client.of("error"))


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_reason, code", [
    ("unacknowledged", "media_stop_unconfirmed"),
    ("error", "media_control_failed"),
])
async def test_a_failed_media_control_frame_carries_the_media_reason(
    monkeypatch, tenant_reason, code,
):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def control(_u, action):
        return {"ok": False, "action": action, "reason": tenant_reason}

    H.patch_relay(monkeypatch, control=control)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")

    client = H.FakeClient([H.config(features=_TRANSPORT), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn("آهنگ رو قطع کن"), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id=f"db-misc-{tenant_reason}")

    frame = _terminal_frame(client)
    assert frame["phase"] == "failed", frame
    assert frame.get("reason") == code, frame


@pytest.mark.asyncio
async def test_every_failed_frame_gets_a_code_even_from_a_bare_call_site():
    """The frame builder backs the rule: a `failed` observation sent without a
    reason still carries one; a call site's own code wins."""

    from datetime import datetime, timezone

    session = live._LiveSession.__new__(live._LiveSession)
    session.features = {"delegation_frames", "task_lifecycle"}
    session.client_alive = True
    session.ws = H.FakeClient()
    session.task_revisions, session.task_states, session.terminal_tasks = {}, {}, set()
    session.provider_now_ms, session.provider_now_at = 0, time.monotonic()
    session.anchor_wall = datetime.now(timezone.utc)

    def task(delegation_id):
        return live.DelegatedTask(
            delegation_id=delegation_id, offset_ms=0, transcript="find a plumber",
            turn_id="live-utt:x:1", created_ms=0,
        )

    await session.send_delegation("failed", task("a"))
    await session.send_delegation("failed", task("b"), reason="anchor_lost")
    await session.send_delegation("completed", task("c"))
    by_id = {f["delegation_id"]: f for f in session.ws.of("delegation")}
    assert by_id["a"]["reason"] == "delegation_failed"
    assert by_id["b"]["reason"] == "anchor_lost"
    assert "reason" not in by_id["c"]


# ── item 9: a play the tenant superseded ──────────────────────────────


def _superseding_tenant(monkeypatch, bodies):
    from app.api import ws_realtime as rt

    async def vps(_user_id):
        return ("http://agent.invalid", "key")

    async def vps_api(_url, _key, method, path, json_body=None, **_kwargs):
        bodies.append((method, path, json_body))
        if path == "/api/v1/internal/play-media":
            return {"ok": False, "reason": "superseded"}
        return None

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)
    return vps


@pytest.mark.asyncio
async def test_the_direct_play_maps_a_superseded_tenant_answer_to_no_start(monkeypatch):
    from app.api import ws_realtime as rt

    bodies: list = []
    _superseding_tenant(monkeypatch, bodies)
    text, media = await rt._play_media_direct("user-1", "Jazz Mix")

    assert [b[1] for b in bodies] == ["/api/v1/internal/play-media"]
    assert media is None, "a dropped play must never persist a card"
    # An ERROR for the model (the realtime relay's tool frame reads ok:false)…
    assert text.strip().upper().startswith("ERROR")
    # …that says what happened, never that the track is starting or failed.
    assert "superseded" in text.lower()
    assert "starting" not in text.lower()
    assert "try again" not in text.lower() and "try a different" not in text.lower()


@pytest.mark.asyncio
async def test_a_play_the_tenant_superseded_ends_cancelled_and_says_nothing(monkeypatch):
    from app.api import ws_realtime as rt

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    real_play = rt._play_media_direct
    bodies: list = []
    thinks: list = []
    controls: list = []

    async def think(*args, **_kwargs):
        thinks.append(args)
        return "Starting Jazz Mix on your phone.", "m"

    async def control(_u, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "stopped"}

    vps = _superseding_tenant(monkeypatch, bodies)
    recorded = H.patch_relay(monkeypatch, think=think, control=control, play=real_play, vps=vps)

    async def wait():
        await _wait_until(lambda: _done(client), label="d1 terminal")
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(features=_TRANSPORT), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_single_turn("play Jazz Mix"), auto_ack=True)
    await H.run_relay(client, provider, timeout=7, db_session_id="db-misc-superseded")

    assert [b[1] for b in bodies if b[1] == "/api/v1/internal/play-media"] == [
        "/api/v1/internal/play-media",
    ]
    assert thinks == [], "a superseded play was handed to a full agent turn"
    frame = _terminal_frame(client)
    assert frame["phase"] == "cancelled", client.phases()
    assert frame["state"] == "canceled"
    assert frame.get("reason") == "media_stopped"
    assert not _terminal(client, "d1", "completed")
    tool = [f for f in client.of("tool_call.completed") if f.get("name") == "play_media"]
    assert [(f["ok"], f["outcome"]) for f in tool] == [(False, "cancelled")]
    spoken = _spoken_text(provider)
    assert "Starting" not in spoken
    assert relay_line("media_playing", "en", title="Jazz Mix") not in spoken
    # The tenant never broadcast it: nothing to re-assert, no card row.
    assert controls == []
    assert not [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_media"
    ]


@pytest.mark.asyncio
async def test_a_spoken_stop_then_the_tenant_dropping_the_older_play_sends_one_stop(monkeypatch):
    """The F29 race with the tenant guard in place: "play some jazz" is still
    searching when "stop the music" is confirmed; the tenant then drops the
    stale play. The relay knows both — the play ends `cancelled`, nothing is
    announced, and the stop is NOT re-asserted (nothing reached the phone)."""

    from app.api import ws_realtime as rt

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    real_play = rt._play_media_direct
    halt_acked = asyncio.Event()
    controls: list = []
    bodies: list = []

    async def vps(_user_id):
        return ("http://agent.invalid", "key")

    async def vps_api(_url, _key, method, path, json_body=None, **_kwargs):
        bodies.append(path)
        if path == "/api/v1/internal/play-media":
            await halt_acked.wait()
            return {"ok": False, "reason": "superseded"}
        return None

    async def control(_u, action):
        controls.append(action)
        halt_acked.set()
        return {"ok": True, "action": action, "reason": "stopped", "changed": True}

    recorded = H.patch_relay(monkeypatch, control=control, play=real_play, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("play some jazz", 0, 500))
            provider.push(H.delegation("dPlay", 520))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def script():
        await _wait_until(lambda: "/api/v1/internal/play-media" in bodies, label="play sent")
        box["p"].push(H.user_delta("stop the music", 2000, 2600))
        box["p"].push(H.delegation("dHalt", 2620))
        await _wait_until(
            lambda: _done(client, "dHalt") and _done(client, "dPlay"), timeout=5,
            label="both terminal",
        )
        await asyncio.sleep(0.4)

    client = H.FakeClient([H.config(features=_TRANSPORT), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-misc-f29-tenant")

    assert controls == ["stop"], controls
    assert _terminal_frame(client, "dPlay")["phase"] == "cancelled"
    assert _terminal(client, "dHalt", "completed")
    spoken = _spoken_text(provider)
    assert relay_line("media_stopped", "en") in spoken
    assert "jazz" not in spoken.lower() and "Starting" not in spoken
    assert not [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_media"
    ]


# ── item 10: the backstop verdict row sits under its turn ─────────────


def _final_rows(saves):
    rows = {}
    for save in saves:
        for side in ("user", "assistant"):
            ref = save.get(f"{side}_ref")
            if not ref:
                continue
            at = save.get(f"{side}_occurred_at")
            prev = rows.get(ref) or {}
            voice = save.get(f"{side}_voice") or {}
            rows[ref] = {
                "at": at if at is not None else prev.get("at"),
                "source": voice.get("source") or prev.get("source"),
                "parent": voice.get("parent_user_turn_id") or prev.get("parent"),
            }
    return rows


@pytest.mark.asyncio
async def test_the_backstop_verdict_row_is_saved_under_the_stop_turn(monkeypatch):
    """The caller says a stop the model never delegates; while the tenant waits
    for the phone's ack they say something else. Live shows the verdict under
    the stop turn, so the saved row must sort there too — not below the later
    turn, where the day chat reads it as that turn's answer (reverify
    w4-persist NEW, F32 class)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", 300)
    gate = asyncio.Event()
    called = asyncio.Event()

    async def control(_u, action):
        called.set()
        await gate.wait()
        return {"ok": True, "action": action, "reason": "stopped"}

    recorded = H.patch_relay(monkeypatch, control=control)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def script():
        p = box["p"]
        await asyncio.sleep(0.3)
        p.push(H.user_delta("آهنگ رو قطع کن", 3000, 3600))
        p.push(H.out_text("چشم.", 3700, 3900))
        p.push(H.out_audio("OK"))
        assert await asyncio.wait_for(called.wait(), 4) is True
        p.push(H.user_delta("what's the weather in paris this weekend", 5000, 6200))
        await asyncio.sleep(1.2)          # the later turn closes and is saved
        gate.set()
        await _wait_until(lambda: any(
            (s.get("assistant_voice") or {}).get("source") == "live_media_control"
            for s in recorded["saves"]), timeout=4, label="verdict row")
        await asyncio.sleep(0.3)

    client = H.FakeClient([
        H.config(features=_TRANSPORT),
        {"type": "now_playing", "title": "Fadat Sham - Mahasti"},
        script, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=14, db_session_id="db-misc-backstop")

    rows = _final_rows(recorded["saves"])
    ordered = [r for r, d in sorted(
        ((r, d) for r, d in rows.items() if d["at"]), key=lambda kv: kv[1]["at"],
    )]
    control_row = next(r for r in ordered if r.startswith("live-media-control:"))
    stop_turn = f"live-utt:{H.PSID}:1"
    later = [r for r in ordered if r.startswith("live-utt:")][-1]
    assert rows[control_row]["parent"] == stop_turn
    assert later != stop_turn
    assert ordered.index(stop_turn) < ordered.index(control_row) < ordered.index(later), ordered


# ── item 10: a failing tick backs off ─────────────────────────────────


def _tick_session():
    session = live._LiveSession.__new__(live._LiveSession)
    session.closed = asyncio.Event()
    session.tick_wake = asyncio.Event()
    # The deadline is always past: the loop has nothing to await but the tick.
    session.next_deadline = lambda: time.monotonic() - 1.0
    return session


@pytest.mark.asyncio
async def test_a_tick_that_always_fails_backs_off_and_never_starves_the_loop():
    session = _tick_session()
    calls = 0
    beats = 0

    async def failing_tick():
        nonlocal calls
        calls += 1
        if calls >= 400:
            # A bound for the unfixed loop, which never yields: without it
            # this test would hang instead of failing.
            session.closed.set()
        raise RuntimeError("tick bug")

    async def heartbeat():
        nonlocal beats
        while not session.closed.is_set():
            beats += 1
            await asyncio.sleep(0.01)

    session.tick = failing_tick
    loop_task = asyncio.create_task(session.tick_loop())
    beat_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.4)
    session.closed.set()
    loop_task.cancel()
    await asyncio.gather(loop_task, beat_task, return_exceptions=True)

    # 50 ms, 100 ms, 200 ms…: a handful of retries in 0.4 s, not hundreds.
    assert 2 <= calls <= 8, calls
    assert beats >= 10, f"the event loop starved: heartbeat ran {beats} times"


@pytest.mark.asyncio
async def test_a_tick_that_recovers_resumes_at_full_speed():
    session = _tick_session()
    calls = 0

    async def flaky_tick():
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise RuntimeError("transient")
        await asyncio.sleep(0.005)

    session.tick = flaky_tick
    loop_task = asyncio.create_task(session.tick_loop())
    await asyncio.sleep(0.8)
    session.closed.set()
    loop_task.cancel()
    await asyncio.gather(loop_task, return_exceptions=True)

    # Three failures cost 50+100+200 ms; after the first success the backoff
    # is gone and the loop ticks every few milliseconds again.
    assert calls > 20, calls


# ── item 10: negation-neutral words need a negation next to them ─────


@pytest.mark.parametrize("text", [
    "Do you speak English?",
    "do you speak Persian",
    "you do speak English",
    "speak English any longer",
    "speak more English",
    "can't speak English",
    "you can't speak English",
])
def test_a_neutral_word_away_from_a_negation_is_not_a_request(text):
    assert live.detect_language_request(text) is None
    assert live.language_verdict(text, "") is None


@pytest.mark.parametrize("text, current, expected", [
    # Still a switch or a release: the neutral word touches its negation.
    ("instead of English, answer in Farsi", "", "fa"),
    ("rather than English, speak Persian", "", "fa"),
    ("don't speak Persian, speak English", "", "en"),
    ("speak Persian, not English any more", "", "fa"),
    ("no more English", "en", "auto"),
    ("no longer English", "en", "auto"),
    ("do not speak English", "en", "auto"),
    ("don't speak English", "en", "auto"),
    ("don't speak English anymore", "en", "auto"),
    ("not English any longer", "en", "auto"),
    # …and a capability question releases nothing either.
    ("do you speak English", "en", None),
])
def test_a_neutral_word_next_to_a_negation_still_counts(text, current, expected):
    assert live.language_verdict(text, current) == expected


# ── item 10: dead code ────────────────────────────────────────────────


def test_the_dead_stamp_for_is_gone():
    """Every row is stamped by `stamp_user` / `stamp_child` /
    `reserve_epoch_stamp` now; a leftover `stamp_for` only let probes patch
    something nothing calls (reverify w4-persist)."""

    assert not hasattr(live._LiveSession, "stamp_for")
