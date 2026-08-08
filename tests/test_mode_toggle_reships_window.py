"""A mode toggle must re-ship the upcoming window in BOTH directions.

The window is resolved per mode: `_resolve_upcoming_variants` rewrites each
upcoming slot to the ATV or the OMV depending on `sess.display_mode`. The phone
skips into that window OPTIMISTICALLY — `handleSkipNext` reads `upcoming[0]`
and advances the card to it without waiting for the backend — so a window that
describes the wrong mode is not a stale cache, it is a wrong card on screen.

Only the song direction re-shipped it. A Video tap swapped the CURRENT track to
its music video and left the phone holding the SONG-side window for the rest of
the station, so every later ⏭ jumped to an ATV id the station would never play.
An ATV plays in the iframe as static album art with no chrome, which is exactly
the founder's 2026-08-06 report: a song-looking card under a lit Video pill,
healing a second or two later when the pop's own resolved media_play arrived.

Reproduced in the simulator the same day: flip to Video, ⏭ within ~1s, and the
client played YDswuo2dIvY while the backend went on to play TyHvyGVs42U.

Hermetic: the lookups and the broadcast are monkeypatched, so this asserts the
BOOKKEEPING and the frame, which is where the bug lived.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from app.agent.radio.playlist import StationTrack           # noqa: E402
from app.agent.radio import RadioSession, SeedTrack         # noqa: E402
import app.api.ws_chat as ws                                # noqa: E402


def _atv(vid: str) -> StationTrack:
    return StationTrack(video_id=vid, title="Song", artist="Artist",
                        video_type="MUSIC_VIDEO_TYPE_ATV")


def _omv(vid: str) -> StationTrack:
    return StationTrack(video_id=vid, title="Song", artist="Artist",
                        video_type="MUSIC_VIDEO_TYPE_OMV")


@pytest.fixture
def sent(monkeypatch):
    """Collect every frame the handler broadcasts."""
    frames = []

    async def _capture(_user_id, payload):
        frames.append(payload)
    monkeypatch.setattr(ws, "broadcast_to_user", _capture)

    async def _track(**kw):
        frames.append({"type": "media_play", **kw})
    monkeypatch.setattr("app.agent.radio.player.broadcast_radio_track", _track)
    return frames


def _install(monkeypatch, sess):
    """Pin the manager's lookup to our session (the handler imports it locally)."""
    from app.agent.radio import get_radio_manager
    mgr = get_radio_manager()
    monkeypatch.setattr(mgr, "get", lambda *a, **k: sess)
    monkeypatch.setattr(mgr, "get_or_create", lambda *a, **k: sess)


def _session(current: StationTrack, upcoming: list[StationTrack]) -> RadioSession:
    s = RadioSession(user_id="u" * 32, channel="app")
    s.enabled = True
    s.seed_track = SeedTrack(video_id=current.video_id, title="seed")
    s.current_track_id = current.video_id
    s.current_station_track = current
    s.playlist = [current] + upcoming
    s.playlist_cursor = 1
    return s


@pytest.mark.asyncio
async def test_video_toggle_reships_the_window(monkeypatch, sent):
    """A Video tap must leave the phone holding a VIDEO-side window."""
    sess = _session(_atv("cur00000000"), [_atv("up100000000"), _atv("up200000000")])
    _install(monkeypatch, sess)

    async def _finds_mv(track):
        return _omv("mv_" + track.video_id[3:])
    monkeypatch.setattr("app.agent.radio.playlist.find_music_video", _finds_mv)

    await ws._handle_radio_display_mode("u" * 32, {"channel": "app", "mode": "video"})

    wins = [f for f in sent if f.get("type") == "radio_upcoming"]
    assert wins, (
        "a Video toggle shipped NO radio_upcoming — the phone keeps the "
        "song-side window and its optimistic ⏭ jumps to an ATV the station "
        "will never play"
    )
    ids = [t["video_id"] for t in wins[-1]["upcoming"]]
    assert all(i.startswith("mv_") for i in ids), (
        f"window still holds song-side ids after a Video toggle: {ids}"
    )


@pytest.mark.asyncio
async def test_song_toggle_still_reships_the_window(monkeypatch, sent):
    """The direction that already worked keeps working (no regression)."""
    sess = _session(_omv("cur00000000"), [_omv("up100000000")])
    sess.display_mode = "video"
    _install(monkeypatch, sess)

    async def _finds_atv(track):
        return _atv("tv_" + track.video_id[3:])
    monkeypatch.setattr("app.agent.radio.playlist.find_topic_version", _finds_atv)

    await ws._handle_radio_display_mode("u" * 32, {"channel": "app", "mode": "song"})

    wins = [f for f in sent if f.get("type") == "radio_upcoming"]
    assert wins, "the song direction stopped re-shipping its window"
    ids = [t["video_id"] for t in wins[-1]["upcoming"]]
    assert all(i.startswith("tv_") for i in ids), (
        f"window still holds video-side ids after a Song toggle: {ids}"
    )


@pytest.mark.asyncio
async def test_no_window_frame_when_radio_is_off(monkeypatch, sent):
    """No station, nothing to re-ship — the toggle must not invent a window."""
    sess = _session(_atv("cur00000000"), [_atv("up100000000")])
    sess.enabled = False
    _install(monkeypatch, sess)

    await ws._handle_radio_display_mode("u" * 32, {"channel": "app", "mode": "video"})

    assert not [f for f in sent if f.get("type") == "radio_upcoming"]


@pytest.mark.asyncio
async def test_window_is_stamped_with_the_mode_it_was_resolved_for(monkeypatch, sent):
    """Every radio_upcoming says which mode its ids are for.

    Re-shipping on both directions closes the common case, but not the race:
    the flip's own round-trip can deliver a window that was resolved BEFORE we
    learned the new mode. The phone cannot tell such a window from a good one
    without being told, and it skips into it optimistically — so the frame
    carries the mode and the phone drops a mismatch. Measured 2026-08-06: after
    pill→video the phone adopted a window whose head was YDswuo2dIvY while the
    station went on to play TyHvyGVs42U.
    """
    sess = _session(_atv("cur00000000"), [_atv("up100000000")])
    _install(monkeypatch, sess)

    async def _finds_mv(track):
        return _omv("mv_" + track.video_id[3:])
    monkeypatch.setattr("app.agent.radio.playlist.find_music_video", _finds_mv)

    await ws._handle_radio_display_mode("u" * 32, {"channel": "app", "mode": "video"})

    wins = [f for f in sent if f.get("type") == "radio_upcoming"]
    assert wins, "no window shipped"
    for w in wins:
        assert w.get("resolved_mode") == "video", (
            "radio_upcoming must name the mode its ids were resolved for, or "
            "the phone cannot reject a window from the other mode: "
            f"{w.get('resolved_mode')!r}"
        )


@pytest.mark.asyncio
async def test_window_stops_at_the_first_unsettled_slot(monkeypatch, sent):
    """The window may only promise slots the pop will not swap.

    `_resolve_upcoming_variants` works under a wall-clock budget, so it
    routinely leaves later slots unsettled — and the pop re-resolves those
    inline, playing a different id than the window advertised. Since the phone
    skips into the window optimistically, that difference IS the stale card:
    measured 2026-08-06, it skipped to KAROL G - Topic (album art) while the
    station played the official video, same song, ~1s apart.
    """
    settled, unsettled = _omv("aaaaaaaaaaa"), _omv("bbbbbbbbbbb")
    settled.variant_resolved_mode = "video"          # pre-resolver reached it
    unsettled.variant_resolved_mode = ""             # budget ran out here
    sess = _session(_omv("cur00000000"), [settled, unsettled])
    sess.display_mode = "video"
    sess.display_mode_user_override = True

    win = ws._upcoming_tracks(sess)
    assert [t["video_id"] for t in win] == ["aaaaaaaaaaa"], (
        "the window advertised a slot the pop would still swap — the phone "
        f"skips straight into it: {[t['video_id'] for t in win]}"
    )


@pytest.mark.asyncio
async def test_window_is_not_truncated_without_an_override(monkeypatch, sent):
    """No override → no pop-time swap → every slot is already true.

    Truncating here would cost the prefetch queue for nothing.
    """
    a, b = _atv("aaaaaaaaaaa"), _atv("bbbbbbbbbbb")
    a.variant_resolved_mode = b.variant_resolved_mode = ""
    sess = _session(_atv("cur00000000"), [a, b])
    sess.display_mode_user_override = False

    win = ws._upcoming_tracks(sess)
    assert [t["video_id"] for t in win] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
