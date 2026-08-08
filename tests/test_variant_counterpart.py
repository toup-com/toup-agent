"""A Song<->Video flip must not cost a network search, and must not shrink the
prebuffer window.

BACKGROUND (2026-08-07, founder device trail):
every track that was not already downloaded took 5.44s +/- 0.03 to become
audible, against <1s for one the phone had prebuffered. The phone can only
prebuffer what `radio_upcoming` tells it about, and that window had collapsed
to ONE track.

Why it collapsed: `_upcoming_tracks` truncates at the first slot the pop would
still swap (that truncation is itself a fix — a window promising ids the pop
will replace IS the stale-card bug). Settling a slot needs a variant lookup, the
forward swap DISCARDED the track it replaced, and so flipping back re-searched
YT Music for tracks it had held moments earlier — under a 2s budget, over 8
slots. One or two settled; the rest truncated the window away.

So the window depth after a flip is a function of how many lookups fit in the
budget, which is the wrong thing for it to depend on. Remembering the
counterpart makes a flip back free and the window full-depth by construction.

These tests are hermetic: no network, no DB, no event loop beyond asyncio.run.
"""
import asyncio
import sys
import types

import pytest

from app.agent.radio.playlist import StationTrack


# ── helpers ───────────────────────────────────────────────────────────────
ATV = "MUSIC_VIDEO_TYPE_ATV"
OMV = "MUSIC_VIDEO_TYPE_OMV"


def _atv(i):
    return StationTrack(video_id=f"atv{i}", title=f"Song {i}", artist="A", video_type=ATV)


def _omv(i):
    return StationTrack(video_id=f"omv{i}", title=f"Song {i}", artist="A", video_type=OMV)


class _Sess:
    """The slice of RadioSession the resolver touches."""

    def __init__(self, playlist, mode="song"):
        self.channel = "app"
        self.display_mode = mode
        self.display_mode_user_override = True
        self.playlist = playlist
        self.playlist_cursor = 0
        self.played_track_ids = set()


@pytest.fixture
def ws(monkeypatch):
    """Import ws_chat with the two variant lookups stubbed, counting calls."""
    import app.api.ws_chat as ws_chat
    import app.agent.radio.playlist as pl

    calls = {"topic": 0, "mv": 0}

    async def fake_topic(tr):
        calls["topic"] += 1
        return StationTrack(video_id=tr.video_id.replace("omv", "atv"),
                            title=tr.title, artist=tr.artist, video_type=ATV)

    async def fake_mv(tr):
        calls["mv"] += 1
        return StationTrack(video_id=tr.video_id.replace("atv", "omv"),
                            title=tr.title, artist=tr.artist, video_type=OMV)

    monkeypatch.setattr(pl, "find_topic_version", fake_topic, raising=False)
    monkeypatch.setattr(pl, "find_music_video", fake_mv, raising=False)
    return ws_chat, calls


# ── is_right_variant_for ──────────────────────────────────────────────────
def test_right_variant_predicate():
    assert _atv(1).is_right_variant_for("song") is True
    assert _atv(1).is_right_variant_for("video") is False
    assert _omv(1).is_right_variant_for("video") is True
    assert _omv(1).is_right_variant_for("song") is False


def test_unknown_video_type_is_right_in_both_modes():
    """UGC / unknown has no counterpart lookup, so it is as right as it gets.
    Answering False would make it truncate the window in BOTH directions, for
    a track nothing can ever settle."""
    ugc = StationTrack(video_id="u1", title="t", video_type="")
    assert ugc.is_right_variant_for("song") is True
    assert ugc.is_right_variant_for("video") is True


# ── the round trip ────────────────────────────────────────────────────────
def test_flip_back_costs_no_lookup_and_keeps_window_full(ws):
    ws_chat, calls = ws
    sess = _Sess([_atv(i) for i in range(5)], mode="video")

    # Flip to VIDEO: five ATVs, five music-video lookups.
    asyncio.run(ws_chat._resolve_upcoming_variants(sess, n=5, budget=30.0))
    assert calls["mv"] == 5
    assert [t.video_id for t in sess.playlist] == [f"omv{i}" for i in range(5)]
    assert ws_chat._upcoming_tracks(sess, n=5) != []
    assert len(ws_chat._upcoming_tracks(sess, n=5)) == 5

    # Flip BACK to song. This is the whole point: zero lookups.
    sess.display_mode = "song"
    asyncio.run(ws_chat._resolve_upcoming_variants(sess, n=5, budget=30.0))
    assert calls["topic"] == 0, "flipping back must not hit the network"
    assert [t.video_id for t in sess.playlist] == [f"atv{i}" for i in range(5)]

    # …and the window is still FULL DEPTH, which is what feeds the prebuffer.
    win = ws_chat._upcoming_tracks(sess, n=5)
    assert len(win) == 5, f"window collapsed to {len(win)}"


def test_flip_back_is_free_even_with_a_zero_budget(ws):
    """The budget is the mechanism that used to shrink the window. With the
    counterpart in hand the resolve does no awaiting work at all, so even a
    budget too small to permit a single search settles every slot."""
    ws_chat, calls = ws
    sess = _Sess([_atv(i) for i in range(5)], mode="video")
    asyncio.run(ws_chat._resolve_upcoming_variants(sess, n=5, budget=30.0))

    sess.display_mode = "song"
    asyncio.run(ws_chat._resolve_upcoming_variants(sess, n=5, budget=0.001))
    assert calls["topic"] == 0
    assert len(ws_chat._upcoming_tracks(sess, n=5)) == 5


def test_counterpart_survives_repeated_flips(ws):
    """Song -> Video -> Song -> Video. The link is re-established on every
    swap, so the third and fourth flips are free too."""
    ws_chat, calls = ws
    sess = _Sess([_atv(i) for i in range(3)], mode="video")
    asyncio.run(ws_chat._resolve_upcoming_variants(sess, n=3, budget=30.0))
    first = calls["mv"]

    for mode in ("song", "video", "song"):
        sess.display_mode = mode
        asyncio.run(ws_chat._resolve_upcoming_variants(sess, n=3, budget=30.0))

    assert calls["mv"] == first, "a repeat flip to video re-searched"
    assert calls["topic"] == 0, "a repeat flip to song re-searched"
    assert len(ws_chat._upcoming_tracks(sess, n=3)) == 3


def test_counterpart_is_not_used_for_the_wrong_mode(ws):
    """A stashed counterpart must only satisfy the mode it actually matches.
    If this ever loosened, video mode would serve the ATV — the album-art bug
    the variant machinery exists to prevent."""
    ws_chat, calls = ws
    sess = _Sess([_atv(0)], mode="video")
    asyncio.run(ws_chat._resolve_upcoming_variants(sess, n=1, budget=30.0))
    assert sess.playlist[0].video_id == "omv0"

    # The OMV's counterpart is the ATV. Resolving for VIDEO again must NOT
    # adopt it just because a counterpart exists.
    asyncio.run(ws_chat._resolve_upcoming_variants(sess, n=1, budget=30.0))
    assert sess.playlist[0].video_id == "omv0"


def test_repr_does_not_recurse_on_a_linked_pair():
    """The two tracks point at each other. A dataclass repr that included the
    field would recurse until the stack blew — in a debug print, in prod."""
    a, b = _atv(1), _omv(1)
    a.counterpart, b.counterpart = b, a
    assert "counterpart" not in repr(a)
    repr(b)  # must not raise RecursionError
