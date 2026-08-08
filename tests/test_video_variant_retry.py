"""A failed variant lookup must not masquerade as a resolved one.

`StationTrack.variant_resolved_mode == mode` has exactly one meaning to every
reader downstream — "this track is already the right variant for that mode" —
and the pop-time swap gate in `_advance_and_broadcast_next` is one of those
readers:

    if sess.display_mode_user_override and next_track.variant_resolved_mode != effective_mode:
        ... find_music_video(next_track) ...

So stamping the field after a lookup that FAILED does not merely skip a
re-search, which is what its old comment claimed ("No swap needed/available —
accept the current variant and mark it so we don't search it again every
frame"). It tells the pop, and every later frame, that a Topic/ATV upload IS
the music video. One 6s `find_music_video` timeout, one anti-bot-throttled YT
Music search, or one artist-name mismatch pinned that track to album art for
the rest of the session — which is what "Video mode shows a song card" looked
like from the outside (founder report, 2026-08-06: an "Armani White - Topic"
upload playing inside the video embed).

These tests are hermetic: `find_music_video` is monkeypatched, so they assert
the BOOKKEEPING, which is where the bug lived.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from app.agent.radio.playlist import StationTrack           # noqa: E402
from app.agent.radio import RadioSession, SeedTrack         # noqa: E402
import app.api.ws_chat as ws                                # noqa: E402


def _sess_in_video_mode(*tracks: StationTrack) -> RadioSession:
    s = RadioSession(user_id="u" * 32, channel="app")
    s.enabled = True
    s.display_mode = "video"
    s.display_mode_user_override = True          # the pre-resolver's own gate
    s.seed_track = SeedTrack(video_id="seed12345ab", title="seed")
    s.playlist = list(tracks)
    s.playlist_cursor = 0
    return s


def _atv(vid: str) -> StationTrack:
    """A Topic/audio upload — the variant Video mode must swap AWAY from."""
    return StationTrack(video_id=vid, title="Song", artist="Artist",
                        video_type="MUSIC_VIDEO_TYPE_ATV")


@pytest.mark.asyncio
async def test_failed_lookup_leaves_the_track_retryable(monkeypatch):
    """A lookup that comes back empty must NOT settle the track.

    This is the whole bug: if the track is marked resolved, the pop-time swap
    is skipped and the ATV id reaches the iframe as if it were the video.
    """
    monkeypatch.setattr(ws, "_VARIANT_RESOLVE_MAX_ATTEMPTS", 3, raising=False)

    async def _always_fails(_track):
        return None
    monkeypatch.setattr("app.agent.radio.playlist.find_music_video", _always_fails)

    t = _atv("aaaaaaaaaaa")
    sess = _sess_in_video_mode(t)
    await ws._resolve_upcoming_variants(sess)

    assert t.variant_attempts == 1, "the failure must be counted"
    assert t.variant_resolved_mode != "video", (
        "a FAILED lookup marked the track resolved — the pop-time swap gate "
        "reads that as 'already the music video' and never retries"
    )


@pytest.mark.asyncio
async def test_repeated_failures_eventually_settle(monkeypatch):
    """...but not forever: a song with no music video must stop costing searches.

    The retry budget is what separates 'transient' from 'genuinely absent', and
    without an upper bound the fix for the pin would be an unbounded search loop
    on every frame.
    """
    monkeypatch.setattr(ws, "_VARIANT_RESOLVE_MAX_ATTEMPTS", 3, raising=False)

    async def _always_fails(_track):
        return None
    monkeypatch.setattr("app.agent.radio.playlist.find_music_video", _always_fails)

    t = _atv("bbbbbbbbbbb")
    sess = _sess_in_video_mode(t)
    for _ in range(3):
        t.variant_resolved_mode = ""          # a later frame re-targets it
        await ws._resolve_upcoming_variants(sess)

    assert t.variant_attempts == 3
    assert t.variant_resolved_mode == "video", (
        "after the retry budget the track must settle, or every frame pays for "
        "a search that will never succeed"
    )


@pytest.mark.asyncio
async def test_no_swap_needed_settles_immediately(monkeypatch):
    """A track already in the right variant is settled on sight, not retried.

    The counter exists for FAILURES only. An OMV in video mode, or a type we
    have no counterpart lookup for at all, is a fact — not an attempt.
    """
    called = False

    async def _should_not_be_called(_track):
        nonlocal called
        called = True
        return None
    monkeypatch.setattr("app.agent.radio.playlist.find_music_video", _should_not_be_called)

    omv = StationTrack(video_id="ccccccccccc", title="Song", artist="Artist",
                       video_type="MUSIC_VIDEO_TYPE_OMV")
    ugc = StationTrack(video_id="ddddddddddd", title="Song", artist="Artist",
                       video_type="")
    sess = _sess_in_video_mode(omv, ugc)
    await ws._resolve_upcoming_variants(sess)

    assert not called, "no lookup should be issued for a track needing no swap"
    assert omv.variant_resolved_mode == "video"
    assert ugc.variant_resolved_mode == "video"
    assert omv.variant_attempts == 0 and ugc.variant_attempts == 0


@pytest.mark.asyncio
async def test_successful_lookup_swaps_and_settles(monkeypatch):
    """The happy path still works: the ATV slot is replaced by the OMV id."""
    async def _finds(track):
        return StationTrack(video_id="eeeeeeeeeee", title=track.title,
                            artist=track.artist, video_type="MUSIC_VIDEO_TYPE_OMV")
    monkeypatch.setattr("app.agent.radio.playlist.find_music_video", _finds)

    t = _atv("fffffffffff")
    sess = _sess_in_video_mode(t)
    await ws._resolve_upcoming_variants(sess)

    assert sess.playlist[0].video_id == "eeeeeeeeeee"
    assert sess.playlist[0].variant_resolved_mode == "video"
