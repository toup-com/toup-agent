"""Focused contract tests for the tenant-side radio control endpoint."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.agent.radio import control
from app.agent.radio.control import MediaControlOutcome
from app.agent.radio.playlist import StationTrack
from app.agent.radio.session import RadioSession, RadioSessionManager, SeedTrack
from app.api import api_v1, ws_chat
from app.api.api_v1 import MediaControlRequest
from app.config import settings


class _FakeRequest:
    def __init__(self, agent_key: str | None):
        self.headers = {} if agent_key is None else {"X-Agent-Key": agent_key}


class _Manager:
    def __init__(self, session):
        self.session = session

    def get(self, user_id, channel):
        return self.session


def _session(video_id: str = "before"):
    return SimpleNamespace(
        enabled=True,
        current_track_id=video_id,
        current_station_track=SimpleNamespace(title=f"Title {video_id}"),
    )


@pytest.fixture
def agent_mode(monkeypatch):
    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "secret-agent-key")
    monkeypatch.setattr(settings, "user_id", "owner-1")


@pytest.mark.asyncio
async def test_service_waits_for_next_and_returns_the_correlated_track(monkeypatch):
    session = _session()
    started = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(control, "get_radio_manager", lambda: _Manager(session))

    async def next_track(user_id, msg):
        assert user_id == "owner-1"
        assert msg == {
            "channel": "app", "reason": "user", "require_delivery": True,
        }
        started.set()
        await release.wait()
        session.current_track_id = "after"
        session.current_station_track = SimpleNamespace(title="Title after")
        return True

    monkeypatch.setattr(ws_chat, "_handle_radio_skip_next", next_track)

    pending = asyncio.create_task(control.execute_media_control("owner-1", "next"))
    await started.wait()
    assert not pending.done(), "the result must not acknowledge before the handler completes"
    release.set()
    outcome = await pending

    assert outcome == MediaControlOutcome(
        ok=True,
        user_id="owner-1",
        action="next",
        channel="app",
        changed=True,
        reason="advanced",
        previous_video_id="before",
        video_id="after",
        title="Title after",
    )


@pytest.mark.asyncio
async def test_service_reports_no_session_without_claiming_success(monkeypatch):
    called = False
    monkeypatch.setattr(control, "get_radio_manager", lambda: _Manager(None))

    async def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(ws_chat, "_handle_radio_skip_next", should_not_run)
    outcome = await control.execute_media_control("owner-1", "next", "app")

    assert outcome.ok is False
    assert outcome.changed is False
    assert outcome.reason == "no_active_session"
    assert called is False


@pytest.mark.asyncio
async def test_session_lost_after_dispatch_is_superseded_not_no_session(monkeypatch):
    session = _session()
    manager = _Manager(session)
    monkeypatch.setattr(control, "get_radio_manager", lambda: manager)

    async def turned_off_before_handler_lock(user_id, msg):
        session.enabled = False
        msg["_media_control_result"] = {
            "reason": "no_active_session",
            "delivered": False,
            "changed": False,
        }
        return False

    monkeypatch.setattr(
        ws_chat, "_handle_radio_skip_next", turned_off_before_handler_lock,
    )
    outcome = await control.execute_media_control("owner-1", "next", "app")

    assert outcome.ok is False
    assert outcome.reason == "superseded"


@pytest.mark.asyncio
async def test_service_reports_unchanged_and_error_as_false(monkeypatch):
    session = _session()
    monkeypatch.setattr(control, "get_radio_manager", lambda: _Manager(session))

    async def no_previous(user_id, msg):
        return False

    monkeypatch.setattr(ws_chat, "_handle_radio_skip_prev", no_previous)
    unchanged = await control.execute_media_control("owner-1", "previous", "app")
    assert (unchanged.ok, unchanged.changed, unchanged.reason) == (False, False, "unchanged")

    async def failed_after_mutation(user_id, msg):
        session.current_track_id = "mutated-before-error"
        raise RuntimeError("broadcast failed")

    monkeypatch.setattr(ws_chat, "_handle_radio_skip_prev", failed_after_mutation)
    failed = await control.execute_media_control("owner-1", "previous", "app")
    assert failed.ok is False
    assert failed.changed is True
    assert failed.reason == "error"
    assert failed.video_id == "mutated-before-error"


@pytest.mark.asyncio
async def test_service_rejects_a_concurrent_reseed_even_if_handler_returns_true(monkeypatch):
    session = _session()
    session.seed_track = SimpleNamespace(video_id="seed-before")
    monkeypatch.setattr(control, "get_radio_manager", lambda: _Manager(session))

    async def reseeded(user_id, msg):
        session.seed_track = SimpleNamespace(video_id="seed-after")
        session.current_track_id = "after"
        return True

    monkeypatch.setattr(ws_chat, "_handle_radio_skip_next", reseeded)
    outcome = await control.execute_media_control("owner-1", "next", "app")

    assert outcome.ok is False
    assert outcome.changed is True
    assert outcome.reason == "superseded"


@pytest.mark.asyncio
async def test_radio_broadcast_false_is_not_reported_as_delivery(monkeypatch):
    import app.agent.radio.player as player

    session = RadioSession(user_id="owner-1", channel="app", enabled=True)
    track = StationTrack(video_id="after", title="After")

    async def no_delivery(**kwargs):
        return False

    async def state_delivery(user_id, payload):
        return 1

    async def no_resolve(*args, **kwargs):
        return None

    monkeypatch.setattr(player, "broadcast_radio_track", no_delivery)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", state_delivery)
    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", no_resolve)

    delivered = await ws_chat._broadcast_track_for_mode(
        "owner-1", "app", session, track, "skip_prev", record=False,
    )
    assert delivered is False


@pytest.mark.asyncio
async def test_radio_player_requires_at_least_one_live_recipient(monkeypatch):
    import app.agent.radio.player as player

    async def no_recipient(user_id, payload):
        return 0

    async def no_age_check(*args, **kwargs):
        return None

    monkeypatch.setattr(ws_chat, "broadcast_to_user", no_recipient)
    monkeypatch.setattr(ws_chat, "_check_age_and_swap", no_age_check)
    monkeypatch.setattr(player, "warm_audio_cache", lambda *args, **kwargs: None)

    delivered = await player.broadcast_radio_track(
        user_id="owner-1", video_id="track-1", title="Track 1", channel="app",
    )
    await asyncio.sleep(0)
    assert delivered is False


@pytest.mark.asyncio
async def test_transactional_next_rolls_back_when_delivery_fails(monkeypatch):
    import app.agent.radio as radio

    manager = RadioSessionManager()
    session = RadioSession(user_id="rollback-owner", channel="app", enabled=True)
    before = StationTrack(video_id="before", title="Before")
    after = StationTrack(video_id="after", title="After")
    session.seed_track = SeedTrack(video_id="before", title="Before")
    session.mark_current_track("before")
    session.current_station_track = before
    session.playlist = [after]
    session.playlist_cursor = 0
    session.played_track_ids = {"before"}
    session.played_history = [before]
    session.history_cursor = 0
    manager._sessions[(session.user_id, session.channel)] = session

    async def no_refill(*args, **kwargs):
        return None, []

    async def failed_delivery(*args, **kwargs):
        return False

    monkeypatch.setattr(radio, "get_radio_manager", lambda: manager)
    monkeypatch.setattr(radio, "build_station", no_refill)
    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", failed_delivery)

    delivered = await ws_chat._handle_radio_skip_next(
        session.user_id,
        {"channel": "app", "reason": "user", "require_delivery": True},
    )

    assert delivered is False
    assert session.current_track_id == "before"
    assert session.playlist_cursor == 0
    assert [track.video_id for track in session.played_history] == ["before"]
    assert session.history_cursor == 0


def _active_radio_session(
    user_id: str, *, queue_size: int = 5, history_size: int = 1,
) -> RadioSession:
    session = RadioSession(user_id=user_id, channel="app", enabled=True)
    current = StationTrack(video_id="current-track", title="Current")
    session.seed_track = SeedTrack(video_id="seed-track", title="Seed")
    session.mark_current_track(current.video_id)
    session.current_station_track = current
    session.playlist = [
        StationTrack(video_id=f"next-{index}", title=f"Next {index}")
        for index in range(queue_size)
    ]
    history = [
        StationTrack(video_id=f"history-{index}", title=f"History {index}")
        for index in range(max(0, history_size - 1))
    ] + [current]
    session.played_history = history
    session.history_cursor = len(history) - 1
    session.played_track_ids = {track.video_id for track in history}
    return session


def _patch_radio_manager(monkeypatch, manager: RadioSessionManager) -> None:
    import app.agent.radio as radio

    monkeypatch.setattr(radio, "get_radio_manager", lambda: manager)
    monkeypatch.setattr(control, "get_radio_manager", lambda: manager)
    monkeypatch.setattr(ws_chat, "_media_ended_locks", {})
    monkeypatch.setattr(ws_chat, "_radio_toggle_locks", {})
    monkeypatch.setattr(ws_chat, "_radio_off_generations", {})
    monkeypatch.setattr(ws_chat, "_display_mode_locks", {})


@pytest.mark.asyncio
async def test_real_toggle_off_preempts_a_blocked_refill(monkeypatch):
    import app.agent.radio as radio

    session = _active_radio_session("off-owner", queue_size=1)
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    refill_started = asyncio.Event()
    release_refill = asyncio.Event()
    played: list[str] = []
    sent: list[dict] = []

    async def blocked_refill(*args, **kwargs):
        refill_started.set()
        await release_refill.wait()
        return None, [StationTrack(video_id="refilled", title="Refilled")]

    async def capture_play(user_id, channel, sess, track, trigger, record):
        played.append(track.video_id)
        return True

    async def capture_state(user_id, payload):
        sent.append(payload)
        return 1

    monkeypatch.setattr(radio, "build_station", blocked_refill)
    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", capture_play)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", capture_state)

    advance = asyncio.create_task(
        ws_chat._handle_radio_skip_next(session.user_id, {"channel": "app"}),
    )
    await refill_started.wait()
    await asyncio.wait_for(
        ws_chat._handle_radio_toggle(
            session.user_id, {"channel": "app", "enabled": False},
        ),
        timeout=0.25,
    )
    assert session.enabled is False

    release_refill.set()
    assert await advance is False
    assert played == []
    assert any(frame.get("type") == "radio_state" and not frame["enabled"] for frame in sent)


@pytest.mark.asyncio
async def test_toggle_off_invalidates_on_requests_already_queued_behind_a_build(
    monkeypatch,
):
    import app.agent.radio as radio
    import app.agent.radio.player as player
    import app.agent.radio.playlist as playlist

    session = _active_radio_session("queued-off-owner")
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    build_started = asyncio.Event()
    release_build = asyncio.Event()
    build_calls = 0
    played: list[str] = []

    async def no_topic_version(*args, **kwargs):
        return None

    async def blocked_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            build_started.set()
            await release_build.wait()
        return None, [StationTrack(video_id="replacement-next", title="Replacement")]

    async def capture_play(**kwargs):
        played.append(kwargs["video_id"])
        return True

    async def capture_state(*args, **kwargs):
        return 1

    monkeypatch.setattr(playlist, "find_topic_version", no_topic_version)
    monkeypatch.setattr(radio, "build_station", blocked_build)
    monkeypatch.setattr(player, "broadcast_radio_track", capture_play)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", capture_state)

    first_on = asyncio.create_task(ws_chat._handle_radio_toggle(
        session.user_id,
        {"channel": "app", "enabled": True, "video_id": "replacement-seed"},
    ))
    await build_started.wait()
    queued_on = asyncio.create_task(ws_chat._handle_radio_toggle(
        session.user_id,
        {"channel": "app", "enabled": True, "video_id": "replacement-seed"},
    ))
    await asyncio.sleep(0)

    await asyncio.wait_for(ws_chat._handle_radio_toggle(
        session.user_id, {"channel": "app", "enabled": False},
    ), timeout=0.25)
    release_build.set()
    await asyncio.gather(first_on, queued_on)

    assert session.enabled is False
    assert build_calls == 1, "the ON queued before OFF must not start a new build"
    assert played == []


@pytest.mark.asyncio
async def test_display_mode_flip_serializes_with_navigation(monkeypatch):
    import app.api.media_playlists as media_playlists

    session = _active_radio_session("mode-lock-owner")
    session.display_mode = "song"
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()

    async def blocked_delivery(*args, **kwargs):
        delivery_started.set()
        await release_delivery.wait()
        return True

    async def no_resolve(*args, **kwargs):
        return None

    async def capture_state(*args, **kwargs):
        return 1

    async def no_autosave(*args, **kwargs):
        return None

    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", blocked_delivery)
    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", no_resolve)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", capture_state)
    monkeypatch.setattr(media_playlists, "autosave_station", no_autosave)

    navigation = asyncio.create_task(ws_chat._handle_radio_skip_next(
        session.user_id, {"channel": "app", "reason": "user"},
    ))
    await delivery_started.wait()
    flip = asyncio.create_task(ws_chat._handle_radio_display_mode(
        session.user_id, {"channel": "app", "mode": "video"},
    ))
    await asyncio.sleep(0)

    assert not flip.done()
    assert session.display_mode == "song"

    release_delivery.set()
    assert await navigation is True
    await flip
    await asyncio.sleep(0)

    assert session.current_track_id == "next-0"
    assert session.display_mode == "video"


@pytest.mark.asyncio
async def test_timeout_after_media_delivery_returns_committed_success(
    agent_mode, monkeypatch,
):
    import app.agent.radio.player as player
    import app.api.media_playlists as media_playlists

    session = _active_radio_session("owner-1")
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    media_delivered = asyncio.Event()
    state_cancelled = asyncio.Event()

    async def deliver_media(**kwargs):
        media_delivered.set()
        return True

    async def blocked_state(*args, **kwargs):
        try:
            await asyncio.Future()
        finally:
            state_cancelled.set()

    async def no_resolve(*args, **kwargs):
        return None

    async def no_autosave(*args, **kwargs):
        return None

    monkeypatch.setattr(player, "broadcast_radio_track", deliver_media)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", blocked_state)
    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", no_resolve)
    monkeypatch.setattr(media_playlists, "autosave_station", no_autosave)
    monkeypatch.setattr(settings, "voice_live_media_control_timeout_s", 0.05)

    response = await api_v1.internal_media_control(
        MediaControlRequest(user_id="owner-1", action="next"),
        _FakeRequest("secret-agent-key"),
    )
    await asyncio.sleep(0)

    assert media_delivered.is_set() and state_cancelled.is_set()
    assert (response["ok"], response["reason"]) == (True, "advanced")
    assert response["video_id"] == "next-0"
    assert session.current_track_id == "next-0"


@pytest.mark.asyncio
async def test_endpoint_timeout_during_refill_rolls_back_without_late_play(
    agent_mode, monkeypatch,
):
    import app.agent.radio as radio

    session = _active_radio_session("owner-1", queue_size=1)
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    refill_started = asyncio.Event()
    refill_cancelled = asyncio.Event()
    played: list[str] = []

    async def blocked_refill(*args, **kwargs):
        refill_started.set()
        try:
            await asyncio.Future()
        finally:
            refill_cancelled.set()

    async def capture_play(user_id, channel, sess, track, trigger, record):
        played.append(track.video_id)
        return True

    monkeypatch.setattr(radio, "build_station", blocked_refill)
    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", capture_play)
    monkeypatch.setattr(settings, "voice_live_media_control_timeout_s", 0.05)

    response = await api_v1.internal_media_control(
        MediaControlRequest(user_id="owner-1", action="next"),
        _FakeRequest("secret-agent-key"),
    )

    assert refill_started.is_set() and refill_cancelled.is_set()
    assert (response["ok"], response["reason"]) == (False, "timeout")
    assert session.current_track_id == "current-track"
    assert session.playlist_cursor == 0
    assert [track.video_id for track in session.played_history] == ["current-track"]
    assert played == []
    await asyncio.sleep(0)
    assert played == []


@pytest.mark.asyncio
async def test_failed_delivery_retry_selects_the_same_track(monkeypatch):
    import app.api.media_playlists as media_playlists

    session = _active_radio_session("retry-owner")
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    attempted: list[str] = []
    deliveries = iter((False, True))

    async def deliver(user_id, channel, sess, track, trigger, record):
        attempted.append(track.video_id)
        return next(deliveries)

    async def no_autosave(*args, **kwargs):
        return None

    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", deliver)
    monkeypatch.setattr(media_playlists, "autosave_station", no_autosave)

    first_msg = {"channel": "app", "reason": "user", "require_delivery": True}
    second_msg = {"channel": "app", "reason": "user", "require_delivery": True}
    assert await ws_chat._handle_radio_skip_next(session.user_id, first_msg) is False
    assert session.current_track_id == "current-track"
    assert session.playlist_cursor == 0
    assert first_msg["_media_control_result"]["reason"] == "delivery_failed"

    assert await ws_chat._handle_radio_skip_next(session.user_id, second_msg) is True
    await asyncio.sleep(0)
    assert attempted == ["next-0", "next-0"]
    assert session.current_track_id == "next-0"
    assert second_msg["_media_control_result"]["reason"] == "advanced"


@pytest.mark.asyncio
async def test_failed_previous_delivery_restores_history_cursor(monkeypatch):
    session = _active_radio_session("prev-owner", history_size=2)
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    async def fail_delivery(*args, **kwargs):
        return False

    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", fail_delivery)
    msg = {"channel": "app", "require_delivery": True}

    assert await ws_chat._handle_radio_skip_prev(session.user_id, msg) is False
    assert session.current_track_id == "current-track"
    assert session.history_cursor == 1
    assert msg["_media_control_result"]["reason"] == "delivery_failed"


@pytest.mark.asyncio
async def test_mode_flip_is_not_erased_by_transaction_cleanup(monkeypatch):
    session = _active_radio_session("mode-owner")
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()

    async def delayed_failure(*args, **kwargs):
        delivery_started.set()
        await release_delivery.wait()
        return False

    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", delayed_failure)
    msg = {"channel": "app", "reason": "user", "require_delivery": True}
    pending = asyncio.create_task(
        ws_chat._handle_radio_skip_next(session.user_id, msg),
    )
    await delivery_started.wait()
    manager.set_display_mode(
        session, "video", user_initiated=True, source="test_concurrent_flip",
    )
    release_delivery.set()

    assert await pending is False
    assert session.display_mode == "video"
    assert session.display_mode_user_override is True
    assert msg["_media_control_result"]["reason"] == "superseded"


@pytest.mark.asyncio
async def test_same_seed_rebuild_supersedes_a_blocked_advance(monkeypatch):
    import app.agent.radio as radio

    session = _active_radio_session("reseed-owner", queue_size=1)
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    refill_started = asyncio.Event()
    release_refill = asyncio.Event()
    played: list[str] = []

    async def blocked_refill(*args, **kwargs):
        refill_started.set()
        await release_refill.wait()
        return None, [StationTrack(video_id="stale-refill", title="Stale")]

    async def capture_play(user_id, channel, sess, track, trigger, record):
        played.append(track.video_id)
        return True

    monkeypatch.setattr(radio, "build_station", blocked_refill)
    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", capture_play)
    msg = {"channel": "app", "reason": "user", "require_delivery": True}
    pending = asyncio.create_task(
        ws_chat._handle_radio_skip_next(session.user_id, msg),
    )
    await refill_started.wait()
    manager.enable(
        user_id=session.user_id,
        channel="app",
        seed_intent="same seed, new station",
        seed_track=SeedTrack(video_id="seed-track", title="Seed"),
        station=[StationTrack(video_id="fresh-next", title="Fresh")],
        source="test_same_seed_rebuild",
    )
    release_refill.set()

    assert await pending is False
    assert played == []
    assert session.current_track_id == "seed-track"
    assert [track.video_id for track in session.playlist] == ["fresh-next"]
    assert msg["_media_control_result"]["reason"] == "superseded"


@pytest.mark.asyncio
async def test_exhaustion_is_error_detail_and_keeps_failure_accounting(monkeypatch):
    import app.agent.radio as radio

    session = _active_radio_session("exhaust-owner", queue_size=0)
    manager = RadioSessionManager()
    manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)

    async def empty_refill(*args, **kwargs):
        return None, []

    async def capture_state(*args, **kwargs):
        return 1

    monkeypatch.setattr(radio, "build_station", empty_refill)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", capture_state)

    outcome = await control.execute_media_control(session.user_id, "next", "app")

    assert outcome.ok is False
    assert outcome.changed is False
    assert outcome.reason == "exhausted"
    assert session.consecutive_failures == 1
    assert session.enabled is True


@pytest.mark.asyncio
async def test_endpoint_timeout_cancels_the_tenant_operation(agent_mode, monkeypatch):
    cancelled = asyncio.Event()

    async def never_finishes(*args, **kwargs):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(control, "execute_media_control", never_finishes)
    monkeypatch.setattr(settings, "voice_live_media_control_timeout_s", 0.1)

    response = await api_v1.internal_media_control(
        MediaControlRequest(user_id="owner-1", action="next"),
        _FakeRequest("secret-agent-key"),
    )

    assert response["ok"] is False
    assert response["reason"] == "timeout"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_endpoint_uses_agent_auth_and_passes_the_default_channel(
    agent_mode, monkeypatch,
):
    seen = {}

    async def execute(user_id, action, channel):
        seen.update(user_id=user_id, action=action, channel=channel)
        return MediaControlOutcome(
            ok=False,
            user_id=user_id,
            action=action,
            channel=channel or "app",
            changed=False,
            reason="unchanged",
        )

    monkeypatch.setattr(control, "execute_media_control", execute)
    response = await api_v1.internal_media_control(
        MediaControlRequest(user_id="owner-1", action="previous"),
        _FakeRequest("secret-agent-key"),
    )

    assert seen == {"user_id": "owner-1", "action": "previous", "channel": "app"}
    assert response["ok"] is False
    assert response["reason"] == "unchanged"
    assert response["action"] == "previous"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run_mode", "key", "user_id", "status_code"),
    [
        ("platform", "secret-agent-key", "owner-1", 404),
        ("agent", None, "owner-1", 401),
        ("agent", "wrong", "owner-1", 401),
        ("agent", "secret-agent-key", "other-owner", 401),
    ],
)
async def test_endpoint_rejects_non_tenant_callers(
    agent_mode, monkeypatch, run_mode, key, user_id, status_code,
):
    monkeypatch.setattr(settings, "run_mode", run_mode)
    with pytest.raises(HTTPException) as exc:
        await api_v1.internal_media_control(
            MediaControlRequest(user_id=user_id, action="next"),
            _FakeRequest(key),
        )
    assert exc.value.status_code == status_code


# ── Device-confirmed stop / pause (spec v0.3 §H, tenant side) ────────────
#
# Recording V2+03:17 and V3+00:09: the caller asked the agent to stop the
# music twice and it kept playing. No layer had a stop: the request model
# accepted only next/previous and `execute_media_control` answered
# `unsupported_action` for anything else. A stop now turns the station OFF
# with the existing radio_toggle semantics and then asks the phone itself to
# stop; `ok` is true only on the phone's acknowledgement.


class _Wire:
    """Every frame the tenant broadcasts to the user's sockets."""

    def __init__(self, recipients: int = 1):
        self.recipients = recipients
        self.frames: list[dict] = []

    async def broadcast(self, user_id, event, exclude=None):
        self.frames.append(dict(event))
        return self.recipients

    def of(self, frame_type: str) -> list[dict]:
        return [frame for frame in self.frames if frame.get("type") == frame_type]


async def _until(predicate, timeout: float = 2.0, label: str = "condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.005)


def _stop_rig(monkeypatch, *, session: RadioSession | None, recipients: int = 1):
    manager = RadioSessionManager()
    if session is not None:
        manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)
    wire = _Wire(recipients)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", wire.broadcast)
    monkeypatch.setattr(control, "_pending_acks", type(control._pending_acks)())
    return manager, wire


def test_the_endpoint_accepts_stop_and_pause_and_nothing_else_new():
    assert MediaControlRequest(user_id="u", action="stop").action == "stop"
    assert MediaControlRequest(user_id="u", action="pause").action == "pause"
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MediaControlRequest(user_id="u", action="shuffle")


@pytest.mark.asyncio
async def test_stop_turns_the_station_off_then_waits_for_the_phone(monkeypatch):
    import uuid

    session = _active_radio_session("owner-1")
    epoch_before = session.station_epoch
    _, wire = _stop_rig(monkeypatch, session=session)

    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=2.0),
    )
    await _until(lambda: wire.of("media_stop"), label="media_stop frame")

    # radio_toggle enabled:false semantics, before the phone is asked.
    assert session.enabled is False
    assert session.station_epoch == epoch_before + 1
    assert ws_chat._radio_off_generations[("owner-1", "app")] == 1
    types = [frame["type"] for frame in wire.frames]
    assert types.index("radio_state") < types.index("media_stop")
    assert wire.of("radio_state")[0]["enabled"] is False

    frame = wire.of("media_stop")[0]
    assert frame["channel"] == "app" and frame["reason"] == "voice"
    uuid.UUID(frame["command_id"])

    await asyncio.sleep(0.05)
    assert not pending.done(), "a stop must not report success before the device acks"

    await ws_chat._dispatch_radio_frame("owner-1", {
        "type": "media_stop_ack", "command_id": frame["command_id"],
        "stopped": True, "was_playing": True,
    })
    outcome = await asyncio.wait_for(pending, timeout=1.0)

    assert (outcome.ok, outcome.reason, outcome.action) == (True, "stopped", "stop")
    assert outcome.command_id == frame["command_id"]
    assert outcome.was_playing is True
    assert outcome.changed is True
    assert outcome.previous_video_id == "current-track"
    assert not control._pending_acks


@pytest.mark.asyncio
async def test_stop_works_with_no_station_and_reports_nothing_playing(monkeypatch):
    # The YT-Music proxy 402 in production means a one-off track can be
    # playing with no station built at all; the stop must still reach it.
    _, wire = _stop_rig(monkeypatch, session=None)

    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=2.0),
    )
    await _until(lambda: wire.of("media_stop"), label="media_stop frame")
    assert ws_chat._radio_off_generations[("owner-1", "app")] == 1
    assert wire.of("radio_state") == [
        {"type": "radio_state", "channel": "app", "enabled": False},
    ]

    command_id = wire.of("media_stop")[0]["command_id"]
    assert control.deliver_media_ack("owner-1", {
        "type": "media_stop_ack", "command_id": command_id,
        "stopped": True, "was_playing": False,
    }) is True
    outcome = await asyncio.wait_for(pending, timeout=1.0)

    assert (outcome.ok, outcome.reason) == (True, "nothing_playing")
    assert outcome.was_playing is False
    assert outcome.changed is False


@pytest.mark.asyncio
async def test_stop_without_an_ack_is_unacknowledged_and_the_station_stays_off(
    monkeypatch,
):
    session = _active_radio_session("owner-1")
    _, wire = _stop_rig(monkeypatch, session=session)

    outcome = await asyncio.wait_for(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=0.05),
        timeout=1.0,
    )

    assert (outcome.ok, outcome.reason) == (False, "unacknowledged")
    assert outcome.was_playing is None
    assert session.enabled is False
    assert len(wire.of("media_stop")) == 1
    assert not control._pending_acks


@pytest.mark.asyncio
async def test_stop_with_no_live_socket_is_delivery_failed_without_waiting(
    monkeypatch,
):
    session = _active_radio_session("owner-1")
    _stop_rig(monkeypatch, session=session, recipients=0)

    outcome = await asyncio.wait_for(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=30.0),
        timeout=1.0,
    )

    assert (outcome.ok, outcome.reason) == (False, "delivery_failed")
    assert session.enabled is False, "the station is off even when no phone is connected"
    assert not control._pending_acks


@pytest.mark.asyncio
async def test_a_device_that_kept_playing_is_an_error_not_a_stop(monkeypatch):
    _, wire = _stop_rig(monkeypatch, session=_active_radio_session("owner-1"))

    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=2.0),
    )
    await _until(lambda: wire.of("media_stop"), label="media_stop frame")
    control.deliver_media_ack("owner-1", {
        "type": "media_stop_ack", "command_id": wire.of("media_stop")[0]["command_id"],
        "stopped": False, "was_playing": True,
    })
    outcome = await asyncio.wait_for(pending, timeout=1.0)

    assert (outcome.ok, outcome.reason) == (False, "error")


@pytest.mark.asyncio
async def test_pause_leaves_the_station_alone_and_needs_its_own_ack(monkeypatch):
    session = _active_radio_session("owner-1")
    epoch_before = session.station_epoch
    _, wire = _stop_rig(monkeypatch, session=session)

    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "pause", "app", ack_timeout_s=2.0),
    )
    await _until(lambda: wire.of("media_pause"), label="media_pause frame")

    assert session.enabled is True
    assert session.station_epoch == epoch_before
    assert wire.of("radio_state") == [] and wire.of("media_stop") == []
    frame = wire.of("media_pause")[0]
    assert frame["channel"] == "app" and frame["reason"] == "voice"

    # A stop ack cannot settle a pause command, even with its id.
    assert control.deliver_media_ack("owner-1", {
        "type": "media_stop_ack", "command_id": frame["command_id"],
        "stopped": True, "was_playing": True,
    }) is False
    await asyncio.sleep(0.02)
    assert not pending.done()

    assert control.deliver_media_ack("owner-1", {
        "type": "media_pause_ack", "command_id": frame["command_id"],
        "paused": True, "was_playing": True,
    }) is True
    outcome = await asyncio.wait_for(pending, timeout=1.0)
    assert (outcome.ok, outcome.reason, outcome.action) == (True, "paused", "pause")


@pytest.mark.asyncio
async def test_an_ack_from_another_user_or_for_another_command_settles_nothing(
    monkeypatch,
):
    _, wire = _stop_rig(monkeypatch, session=None)
    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=2.0),
    )
    await _until(lambda: wire.of("media_stop"), label="media_stop frame")
    command_id = wire.of("media_stop")[0]["command_id"]
    ack = {"type": "media_stop_ack", "stopped": True, "was_playing": True}

    assert control.deliver_media_ack("intruder", {**ack, "command_id": command_id}) is False
    assert control.deliver_media_ack("owner-1", {**ack, "command_id": "not-it"}) is False
    assert control.deliver_media_ack("owner-1", {**ack, "command_id": None}) is False
    assert control.deliver_media_ack("owner-1", {**ack, "command_id": "x" * 500}) is False
    await asyncio.sleep(0.02)
    assert not pending.done()

    assert control.deliver_media_ack("owner-1", {**ack, "command_id": command_id}) is True
    assert (await pending).reason == "stopped"


@pytest.mark.asyncio
async def test_with_two_sockets_an_idle_one_cannot_hide_the_one_that_stopped(
    monkeypatch,
):
    _, wire = _stop_rig(monkeypatch, session=None, recipients=2)
    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=2.0),
    )
    await _until(lambda: wire.of("media_stop"), label="media_stop frame")
    command_id = wire.of("media_stop")[0]["command_id"]

    control.deliver_media_ack("owner-1", {
        "type": "media_stop_ack", "command_id": command_id,
        "stopped": True, "was_playing": False,
    })
    await asyncio.sleep(0.02)
    assert not pending.done(), "one idle socket answered; the other may be the player"

    control.deliver_media_ack("owner-1", {
        "type": "media_stop_ack", "command_id": command_id,
        "stopped": True, "was_playing": True,
    })
    assert (await asyncio.wait_for(pending, timeout=1.0)).reason == "stopped"


@pytest.mark.asyncio
async def test_the_ack_registry_is_bounded(monkeypatch):
    _, wire = _stop_rig(monkeypatch, session=None)
    cap = control._MAX_PENDING_ACKS
    tasks = [
        asyncio.create_task(
            control.execute_media_control("owner-1", "pause", "app", ack_timeout_s=5.0),
        )
        for _ in range(cap + 2)
    ]
    await _until(lambda: len(wire.of("media_pause")) == cap + 2, label="all pauses sent")

    assert len(control._pending_acks) <= cap
    # The two oldest were evicted and answered honestly, not left hanging.
    evicted = await asyncio.wait_for(asyncio.gather(*tasks[:2]), timeout=1.0)
    assert [outcome.reason for outcome in evicted] == ["unacknowledged"] * 2

    for frame in wire.of("media_pause")[2:]:
        control.deliver_media_ack("owner-1", {
            "type": "media_pause_ack", "command_id": frame["command_id"],
            "paused": True, "was_playing": True,
        })
    rest = await asyncio.wait_for(asyncio.gather(*tasks[2:]), timeout=1.0)
    assert {outcome.reason for outcome in rest} == {"paused"}
    assert not control._pending_acks


@pytest.mark.asyncio
async def test_endpoint_stop_confirmed_by_the_device(agent_mode, monkeypatch):
    _, wire = _stop_rig(monkeypatch, session=_active_radio_session("owner-1"))
    monkeypatch.setattr(settings, "voice_live_media_control_timeout_s", 2.0)

    pending = asyncio.create_task(api_v1.internal_media_control(
        MediaControlRequest(user_id="owner-1", action="stop"),
        _FakeRequest("secret-agent-key"),
    ))
    await _until(lambda: wire.of("media_stop"), label="media_stop frame")
    command_id = wire.of("media_stop")[0]["command_id"]
    control.deliver_media_ack("owner-1", {
        "type": "media_stop_ack", "command_id": command_id,
        "stopped": True, "was_playing": True,
    })
    response = await asyncio.wait_for(pending, timeout=1.0)

    assert (response["ok"], response["reason"]) == (True, "stopped")
    assert response["command_id"] == command_id
    assert response["action"] == "stop"


@pytest.mark.asyncio
async def test_endpoint_stop_without_an_ack_answers_inside_the_budget(
    agent_mode, monkeypatch,
):
    """The ack wait is the endpoint budget MINUS a margin: an unanswered stop
    comes back as `unacknowledged` (station off, phone silent), never as the
    endpoint's own `timeout`, which would tell the relay nothing."""
    session = _active_radio_session("owner-1")
    _stop_rig(monkeypatch, session=session)
    monkeypatch.setattr(settings, "voice_live_media_control_timeout_s", 1.0)

    started = asyncio.get_running_loop().time()
    response = await api_v1.internal_media_control(
        MediaControlRequest(user_id="owner-1", action="stop"),
        _FakeRequest("secret-agent-key"),
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert (response["ok"], response["reason"]) == (False, "unacknowledged")
    assert elapsed < 1.0
    assert session.enabled is False
    assert not control._pending_acks


@pytest.mark.asyncio
async def test_endpoint_timeout_for_a_stop_is_an_error_from_the_closed_set(
    agent_mode, monkeypatch,
):
    async def never_finishes(*args, **kwargs):
        await asyncio.Future()

    monkeypatch.setattr(control, "execute_media_control", never_finishes)
    monkeypatch.setattr(settings, "voice_live_media_control_timeout_s", 0.05)

    response = await api_v1.internal_media_control(
        MediaControlRequest(user_id="owner-1", action="stop"),
        _FakeRequest("secret-agent-key"),
    )
    assert (response["ok"], response["reason"]) == (False, "error")


@pytest.mark.asyncio
async def test_a_stop_on_a_channel_the_station_cannot_have_is_an_error(monkeypatch):
    _, wire = _stop_rig(monkeypatch, session=None)

    outcome = await control.execute_media_control("owner-1", "stop", "voice")

    assert (outcome.ok, outcome.reason) == (False, "error")
    assert wire.frames == []


# ── A socket that never answers must not hold the verdict (review F27) ───
# `broadcast_to_user` counts EVERY chat socket of the user, and most of them can
# never answer a media_stop: the web ChatPage (channel 'web', drops channel-'app'
# frames), the desktop bridge, the browser extension, a half-open phone queue
# left behind by a network switch. The phone's "nothing was playing here" then
# waited for them until the whole ack budget ran out — 7 s of silence on the
# call for a stop whose answer arrived in 50 ms.
#
# What is bounded is how long the other sockets get once one device has
# answered — the same round trip any device that CAN answer needs.
#
# The verdict is scoped to who answered (review F27 residual, supervisor
# follow-up): `nothing_playing` is the answering devices' report, so it needs
# an answer from every reached socket that could be playing the phone's
# channel. A socket proven unable to (it declared channel 'web' — the web
# ChatPage drops every channel-'app' frame) is left out; a socket that has
# shown nothing may be the player (an old build), so an idle answer beside it
# is `partially_confirmed` (ok false), never "Nothing was playing". It is not
# `unacknowledged` either: the phone did answer. See test_media_verdict_scope.

#: An answer the call can wait for: the peer grace (`control._PEER_ACK_GRACE_S`,
#: 1.5 s) plus scheduling slack, and far inside the 7 s ack budget.
_PROMPT_S = 2.5

def _real_sockets(monkeypatch, *, session: RadioSession | None = None):
    manager = RadioSessionManager()
    if session is not None:
        manager._sessions[(session.user_id, session.channel)] = session
    _patch_radio_manager(monkeypatch, manager)
    monkeypatch.setattr(ws_chat, "_user_ws_queues", {})
    monkeypatch.setattr(ws_chat, "_recent_broadcasts", {})
    monkeypatch.setattr(control, "_pending_acks", type(control._pending_acks)())
    return manager


def _chat_socket(user_id: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    ws_chat._register_ws_queue(user_id, queue)
    return queue


async def _phone(user_id: str, queue: asyncio.Queue, *, was_playing: bool,
                 done: bool = True, delay: float = 0.05):
    """The phone (`api.ts _runMediaCommand`): it acts on channel-'app'
    media_stop / media_pause and answers over its socket's real ack path."""
    while True:
        frame = await queue.get()
        if frame.get("channel") not in (None, "app"):
            continue
        if frame.get("type") in ("media_stop", "media_pause"):
            await asyncio.sleep(delay)  # native pause + the round trip
            key = "stopped" if frame["type"] == "media_stop" else "paused"
            await ws_chat._dispatch_radio_frame(user_id, {
                "type": f"{frame['type']}_ack", "command_id": frame["command_id"],
                key: done, "was_playing": was_playing, "channel": "app",
            })


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "pause"])
async def test_an_idle_phone_beside_a_socket_that_cannot_answer_is_not_held_to_the_budget(
    agent_mode, monkeypatch, action,
):
    """The endpoint over real chat queues at the production budget: the phone
    answers "nothing was playing", a web tab that received the command never
    will. The answer is the phone's, and it comes back promptly, not after
    the 7 s ack budget.

    PIN MOVED (F27 residual): the web tab now shows what it is the way the web
    ChatPage does — an inbound frame on channel 'web' — so it is known not to
    be the player and the idle answer is the whole truth. A socket that has
    shown nothing is the next test."""
    _real_sockets(monkeypatch, session=_active_radio_session("owner-1"))
    monkeypatch.setattr(settings, "voice_live_media_control_timeout_s", 8.0)
    assert control._default_ack_timeout_s() > 6.0

    phone_q = _chat_socket("owner-1")
    web_q = _chat_socket("owner-1")
    ws_chat._note_socket_frame(web_q, {"type": "message", "channel": "web", "text": "hi"})
    phone = asyncio.create_task(_phone("owner-1", phone_q, was_playing=False))
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        response = await asyncio.wait_for(api_v1.internal_media_control(
            MediaControlRequest(user_id="owner-1", action=action),
            _FakeRequest("secret-agent-key"),
        ), timeout=10.0)
    finally:
        phone.cancel()
    waited = loop.time() - started

    web_saw = [web_q.get_nowait()["type"] for _ in range(web_q.qsize())]
    assert f"media_{action}" in web_saw, "the silent socket was counted: it got the command"
    assert (response["ok"], response["reason"], response["was_playing"]) == (
        True, "nothing_playing", False,
    )
    assert waited < _PROMPT_S, (
        f"an idle phone's answer waited {waited:.2f}s for a socket that cannot answer"
    )
    assert not control._pending_acks


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "pause"])
async def test_an_idle_phone_beside_a_socket_that_has_shown_nothing_is_partial_promptly(
    agent_mode, monkeypatch, action,
):
    """The same, but the second socket has sent nothing that says what it is
    (a web tab never typed in, the desktop bridge, an old build that may be
    the player). Still prompt — and never "Nothing was playing"."""
    _real_sockets(monkeypatch, session=_active_radio_session("owner-1"))
    monkeypatch.setattr(settings, "voice_live_media_control_timeout_s", 8.0)

    phone_q = _chat_socket("owner-1")
    _chat_socket("owner-1")
    phone = asyncio.create_task(_phone("owner-1", phone_q, was_playing=False))
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        response = await asyncio.wait_for(api_v1.internal_media_control(
            MediaControlRequest(user_id="owner-1", action=action),
            _FakeRequest("secret-agent-key"),
        ), timeout=10.0)
    finally:
        phone.cancel()
    waited = loop.time() - started

    assert (response["ok"], response["reason"], response["was_playing"]) == (
        False, "partially_confirmed", None,
    )
    assert (response["acked_devices"], response["silent_devices"]) == (1, 1)
    assert waited < _PROMPT_S
    assert not control._pending_acks


@pytest.mark.asyncio
async def test_a_device_that_kept_playing_beside_a_silent_socket_is_an_error_promptly(
    monkeypatch,
):
    _real_sockets(monkeypatch, session=None)
    phone_q = _chat_socket("owner-1")
    _chat_socket("owner-1")
    phone = asyncio.create_task(
        _phone("owner-1", phone_q, was_playing=True, done=False),
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        outcome = await asyncio.wait_for(
            control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=6.0),
            timeout=8.0,
        )
    finally:
        phone.cancel()

    assert (outcome.ok, outcome.reason, outcome.was_playing) == (False, "error", True)
    assert loop.time() - started < _PROMPT_S


@pytest.mark.asyncio
async def test_the_player_answering_inside_the_grace_still_wins(monkeypatch):
    """The grace is for sockets that cannot answer, not a race against ones
    that can: a second device that really stopped the music, answering a
    moment after the idle one, still makes it `stopped`."""
    _, wire = _stop_rig(monkeypatch, session=None, recipients=2)
    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=6.0),
    )
    await _until(lambda: wire.of("media_stop"), label="media_stop frame")
    command_id = wire.of("media_stop")[0]["command_id"]

    control.deliver_media_ack("owner-1", {
        "type": "media_stop_ack", "command_id": command_id,
        "stopped": True, "was_playing": False,
    })
    await asyncio.sleep(0.5)  # well inside the 1.5 s grace
    assert not pending.done(), "the idle answer must not settle before the grace"
    control.deliver_media_ack("owner-1", {
        "type": "media_stop_ack", "command_id": command_id,
        "stopped": True, "was_playing": True,
    })
    outcome = await asyncio.wait_for(pending, timeout=1.0)
    assert (outcome.ok, outcome.reason, outcome.was_playing) == (True, "stopped", True)


@pytest.mark.asyncio
async def test_the_grace_never_outlives_the_ack_budget(monkeypatch):
    _, wire = _stop_rig(monkeypatch, session=None, recipients=2)
    loop = asyncio.get_running_loop()
    started = loop.time()
    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "pause", "app", ack_timeout_s=0.3),
    )
    await _until(lambda: wire.of("media_pause"), label="media_pause frame")
    control.deliver_media_ack("owner-1", {
        "type": "media_pause_ack", "command_id": wire.of("media_pause")[0]["command_id"],
        "paused": True, "was_playing": False,
    })
    outcome = await asyncio.wait_for(pending, timeout=1.0)

    # PIN MOVED (F27 residual): the second recipient never answered, so the
    # idle answer covers only its own device — partial, not nothing_playing.
    assert (outcome.ok, outcome.reason) == (False, "partially_confirmed")
    assert loop.time() - started < 0.3 + 0.2
    assert not control._pending_acks


@pytest.mark.asyncio
async def test_no_answer_from_any_socket_is_still_unacknowledged(monkeypatch):
    """Nothing answered at all: `nothing_playing` needs a real ack saying so."""
    _real_sockets(monkeypatch, session=None)
    _chat_socket("owner-1")
    _chat_socket("owner-1")

    outcome = await asyncio.wait_for(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=0.2),
        timeout=1.0,
    )
    assert (outcome.ok, outcome.reason, outcome.was_playing) == (False, "unacknowledged", None)
    assert not control._pending_acks


@pytest.mark.asyncio
async def test_a_late_ack_after_the_grace_settles_nothing(monkeypatch):
    _, wire = _stop_rig(monkeypatch, session=None, recipients=2)
    monkeypatch.setattr(control, "_PEER_ACK_GRACE_S", 0.05)
    pending = asyncio.create_task(
        control.execute_media_control("owner-1", "stop", "app", ack_timeout_s=5.0),
    )
    await _until(lambda: wire.of("media_stop"), label="media_stop frame")
    command_id = wire.of("media_stop")[0]["command_id"]
    ack = {"type": "media_stop_ack", "command_id": command_id}

    control.deliver_media_ack("owner-1", {**ack, "stopped": True, "was_playing": False})
    outcome = await asyncio.wait_for(pending, timeout=1.0)
    # PIN MOVED (F27 residual): one of two recipients answered idle.
    assert (outcome.ok, outcome.reason) == (False, "partially_confirmed")
    assert not control._pending_acks
    # The command is over; a straggler is matched to nothing.
    assert control.deliver_media_ack("owner-1", {**ack, "stopped": True, "was_playing": True}) is False
