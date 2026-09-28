"""Round-6 verifier findings, tenant half (addendum 6 R6-10 TA1-TA7;
supervisor 03:56 reproduction `/private/tmp/toup-supervisor-0351.W0zQc6/
reseed.xml`; verifier probes `/private/tmp/toup-voice-r2-review/probes/
r6-verify-tenant2.41CoyE`).

Owner requirement 7 / pattern B: the caller's newest media request wins; a
stop halts exactly the plays requested before it; the spoken outcome matches
the device.

TA1  A same-item station re-announcement — the toggle seed the tenant sends
     when the app reseeds from the song it was just told to play (ChatScreen
     onMediaPlay -> requestReseed -> radio_toggle ON with that video_id), a
     re-anchor to the station's current track, a variant swap of the current
     track — keeps the recorded item's causal identity (order, scope, stamp).
     A genuinely different station track (an advance, another card's seed) is
     a new unordered item.
TA2  The English trailing-tag strip: at the clause edge, never inside
     quotes, never after a title marker, never leaving a head of class words
     (R6-11 TB1: every closed tail tag at the end, no longer ONE).
TA3  A quoted or title-marked Persian span is the query, never dropped.
TA4  An English retraction in a clause after the request declines.
     [R6-17, writer:r7-tenant: a SINGLE quote no longer shields a negation;
     the changed pins say so where they stand — tests/test_media_r7_tenant.py.]
TA5  The media noun carries its plural/possessive suffix in every spelling;
     a referential «آهنگ‌هاش» with no named subject declines.
TA6  «آهنگ رو بذار کنار» / «کنار بذار» ("put aside") is not a play.
TA7  ws_browser's media_play is not a device of the media-control plane.

Everything drives the REAL endpoints, play tool, radio toggle / media_ended /
display-mode handlers, radio player, control and the typed-chat fast path
through the rig of `tests/test_media_order_scope_r6.py`; only the YouTube
search, the YT Music station build, the ATV/MV lookups, the variant resolve,
the autosave and the socket fan-out are stubbed.
"""
# The shared `rig` fixture is imported and then requested by name, which ruff
# reads as a redefinition.
# ruff: noqa: F811
from __future__ import annotations

import asyncio
import inspect
import pathlib

import pytest

from app.agent import media_intent
from app.agent.media_intent import media_request
from app.agent.radio import control
from app.agent.radio.playlist import StationTrack
from app.api import api_v1, ws_chat
from test_media_order_scope_r6 import (  # noqa: F401  (rig is a fixture)
    SCOPE,
    USER,
    _collect,
    _halt,
    _play,
    _played,
    _Req,
    _Runner,
    _turn,
    rig,
)

JAZZ, JAZZ_MV, HALO = "JAZZ0000001", "JAZZMV00001", "HALO0000001"
A, B = "psid-r6c-scope-A", "psid-r6c-scope-B"
T_A, T_B = 30_000, 30_001


def _media(frames):
    return [(f["type"], f.get("video_id", ""), f.get("reason", "")) for f in frames
            if f["type"].startswith("media_")]


async def _settle(task):
    return await asyncio.wait_for(task, timeout=3.0)


def _recorded_order():
    state = control._user_state(USER)
    item = state.playing if state is not None else None
    if item is None:
        return None
    ev = item.event
    return (item.video_id, None if ev is None else (ev.scope, ev.order, ev.stamp))


@pytest.fixture
def station(monkeypatch, rig):
    """The YT Music half of the station: a seed whose own entry is the song
    (ATV) side, six station tracks, and a music-video counterpart."""
    import app.agent.radio as radio_pkg
    import app.agent.radio.playlist as playlist_mod

    async def fake_station(seed, limit=50, **kw):
        meta = StationTrack(video_id=seed, title="Jazz Mix", artist="Various",
                            video_type="MUSIC_VIDEO_TYPE_ATV")
        tracks = [StationTrack(video_id=f"STN{i:08d}", title=f"Station {i}", artist="X")
                  for i in range(6)]
        return meta, tracks

    async def none(*a, **kw):
        return None

    async def music_video(track, *a, **kw):
        return StationTrack(video_id=JAZZ_MV, title="Jazz Mix (Official Video)",
                            artist="Various", video_type="MUSIC_VIDEO_TYPE_OMV")

    monkeypatch.setattr(radio_pkg, "build_station", fake_station)
    monkeypatch.setattr(playlist_mod, "find_topic_version", none)
    monkeypatch.setattr(playlist_mod, "find_music_video", music_video)
    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", none)
    import app.api.media_playlists as mp
    monkeypatch.setattr(mp, "autosave_station", none)
    return rig


async def _phone_reseed(video_id: str, title: str = "Jazz Mix"):
    """What the app sends for every user-initiated media_play (ChatScreen
    onMediaPlay: `_reseed = !radioAuto && (!radioOn || new seed)` ->
    requestReseed -> toggleRadio(true, m)): a radio_toggle ON, no `_auto`."""
    await ws_chat._handle_radio_toggle(USER, {
        "type": "radio_toggle", "channel": "app", "enabled": True,
        "video_id": video_id, "title": title,
    })


async def _jazz_then_reseed(rig, **order):
    out = await _settle(_play("some jazz", **order))
    assert out["ok"] is True and out["video_id"] == JAZZ, out
    await _phone_reseed(JAZZ)
    assert _media(rig.frames)[-1] == ("media_play", JAZZ, "toggle_seed"), _media(rig.frames)


# ══ TA1. A same-item re-announcement keeps the item's causal identity ═════

async def test_ta1_reseed_keeps_the_order_a_late_older_stop_is_newer_playing(station):
    """The supervisor's reseed.xml case (a), through the real toggle path."""
    rig = station
    await _jazz_then_reseed(rig, order=3, scope=SCOPE)
    assert _recorded_order() == (JAZZ, (SCOPE, 3, None))
    before = list(rig.frames)
    late = await _halt(rig, before=2, scope=SCOPE)
    assert (late["ok"], late["reason"], late["video_id"]) == (False, "newer_playing", JAZZ), late
    assert rig.frames == before, "no station OFF, no media_stop: jazz keeps playing"


async def test_ta1_reseed_keeps_the_order_an_older_play_resolving_late_is_fenced(station):
    """Case (b): 'play Halo' (1) still searching, jazz (3) on the device and
    re-announced by the reseed — Halo must not replace it (R6-9 T1)."""
    rig = station
    gate = rig.hold("Halo")
    halo = _play("Halo", order=1, scope=SCOPE)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    await _jazz_then_reseed(rig, order=3, scope=SCOPE)
    gate.set()
    out = await _settle(halo)
    assert (out["ok"], out.get("reason")) == (False, "superseded"), out
    assert HALO not in _played(rig.frames)


async def test_ta1_the_other_direction_a_newer_stop_still_stops_it(station):
    rig = station
    await _jazz_then_reseed(rig, order=3, scope=SCOPE)
    stop = await _halt(rig, before=4, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop
    assert _media(rig.frames)[-1][0] == "media_stop"


async def test_ta1_the_other_direction_a_newer_play_still_replaces_it(station):
    rig = station
    await _jazz_then_reseed(rig, order=3, scope=SCOPE)
    out = await _settle(_play("Halo", order=5, scope=SCOPE))
    assert out["ok"] is True and out["video_id"] == HALO, out
    assert _recorded_order() == (HALO, (SCOPE, 5, None))


async def test_ta1_reseed_keeps_the_scope_and_stamp_across_a_reconnect(station):
    """The identity is the whole key: B's item (stamped) re-announced by the
    reseed still beats A's late stop (older stamp), and A's late play."""
    rig = station
    await _jazz_then_reseed(rig, order=1, scope=B, started=T_B)
    assert _recorded_order() == (JAZZ, (B, 1, T_B))
    late = await _halt(rig, before=9, scope=A, started=T_A)
    assert (late["ok"], late["reason"]) == (False, "newer_playing"), late
    old = await _settle(_play("Halo", order=8, scope=A, started=T_A))
    assert (old["ok"], old["reason"]) == (False, "superseded"), old


async def test_ta1_the_agent_path_item_keeps_its_order_through_the_reseed(station, monkeypatch):
    """The delegated run (the stream route voice takes) plays jazz; the app
    reseeds; a late older stop leaves jazz playing."""
    rig = station
    runner = _Runner(rig.tools)
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    stream = await api_v1.internal_agent_turn_stream(_turn("some jazz", 3), _Req())
    frames = asyncio.create_task(_collect(stream))
    runner.release["some jazz"].set()
    await asyncio.wait_for(frames, timeout=5.0)
    assert runner.results["some jazz"].startswith("Now playing"), runner.results
    await _phone_reseed(JAZZ)
    late = await _halt(rig, before=2, scope=SCOPE)
    assert (late["ok"], late["reason"]) == (False, "newer_playing"), late


async def test_ta1_a_variant_swap_and_a_reanchor_keep_the_order(station):
    """A Video tap swaps the current (seed) track to its music video; a stale
    end for the pre-swap id then re-anchors to the current track. Both are
    re-announcements of the caller's jazz: its order survives both, and the
    verdict names what the device now holds."""
    rig = station
    await _jazz_then_reseed(rig, order=3, scope=SCOPE)
    await ws_chat._handle_radio_display_mode(USER, {"channel": "app", "mode": "video"})
    assert _media(rig.frames)[-1] == ("media_play", JAZZ_MV, "mv_swap"), _media(rig.frames)
    assert _recorded_order() == (JAZZ_MV, (SCOPE, 3, None))
    await ws_chat._handle_media_ended(USER, {"channel": "app", "video_id": JAZZ})
    assert _media(rig.frames)[-1] == ("media_play", JAZZ_MV, "reanchor"), _media(rig.frames)
    assert _recorded_order() == (JAZZ_MV, (SCOPE, 3, None))
    gate = rig.hold("Halo")
    halo = _play("Halo", order=1, scope=SCOPE)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    late = await _halt(rig, before=2, scope=SCOPE)
    assert (late["ok"], late["reason"], late["video_id"]) == (False, "newer_playing", JAZZ_MV), late
    gate.set()
    out = await _settle(halo)
    assert (out["ok"], out.get("reason")) == (False, "superseded"), out


# ── TA1 controls: a genuinely different station item is new and unordered ──

async def test_ta1_control_a_real_advance_is_a_new_unordered_item(station):
    """The seed ends; the station advances. The advance is not the caller's
    jazz: an ordered stop asked before jazz still reaches it."""
    rig = station
    await _jazz_then_reseed(rig, order=3, scope=SCOPE)
    # The phone proves the end (playhead at the track's duration), so the
    # station's early-end pin lets it advance.
    await ws_chat._handle_media_ended(USER, {
        "channel": "app", "video_id": JAZZ, "position": 240, "duration": 240,
    })
    last = _media(rig.frames)[-1]
    assert last[0] == "media_play" and last[1].startswith("STN") and last[2] == "auto_advance", last
    assert _recorded_order() == (last[1], None)
    stop = await _halt(rig, before=2, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop


async def test_ta1_control_a_reseed_from_another_card_is_a_new_unordered_item(station):
    """The radio pill on an OLDER card seeds a different song: that is a new
    station item, not the caller's jazz."""
    rig = station
    await _jazz_then_reseed(rig, order=3, scope=SCOPE)
    await _phone_reseed("OLDCARD0001", "An older card")
    assert _media(rig.frames)[-1] == ("media_play", "OLDCARD0001", "toggle_seed")
    assert _recorded_order() == ("OLDCARD0001", None)
    stop = await _halt(rig, before=2, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop


async def test_ta1_control_after_the_phone_x_a_reseed_is_a_new_item(station):
    """The phone's X stopped jazz (an unordered OFF): nothing is live, so a
    later reseed of the same song is a new unordered item, not the old one."""
    rig = station
    await _jazz_then_reseed(rig, order=3, scope=SCOPE)
    await ws_chat._handle_radio_toggle(USER, {"type": "radio_toggle", "channel": "app", "enabled": False})
    await _phone_reseed(JAZZ)
    assert _recorded_order() == (JAZZ, None)
    stop = await _halt(rig, before=2, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop


async def test_ta1_the_supervisor_controls_hold(station):
    """reseed.xml's two PASS controls: without the reseed the late stop is
    newer_playing; a DIFFERENT station track is stopped."""
    from app.agent.radio import player

    rig = station
    assert (await _settle(_play("some jazz", order=3, scope=SCOPE)))["ok"] is True
    late = await _halt(rig, before=2, scope=SCOPE)
    assert (late["ok"], late["reason"]) == (False, "newer_playing"), late
    assert await player.broadcast_radio_track(
        user_id=USER, video_id="STN00000001", title="Station 1", channel="app",
    )
    stop = await _halt(rig, before=2, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop


def test_ta1_only_the_live_item_is_kept_and_a_mark_wins(monkeypatch):
    """The chokepoint's rule, directly: `reannounces` keeps the LIVE item's
    event only when it names that item; an ordered mark is its own order."""
    monkeypatch.setattr(control, "_latest_halts", type(control._pending_acks)())
    monkeypatch.setattr(ws_chat, "_radio_off_generations", {})
    mark = control.media_halt_mark(USER, media_order=4, media_scope=SCOPE)
    control.note_media_broadcast(USER, mark, video_id=JAZZ, title="Jazz")
    control.note_media_broadcast(USER, video_id=JAZZ, title="Jazz (seed)", reannounces=JAZZ)
    assert _recorded_order() == (JAZZ, (SCOPE, 4, None))
    assert control.recorded_media_item(USER)["title"] == "Jazz (seed)"
    control.note_media_broadcast(USER, video_id="OTHER000001", reannounces="SOMETHING01")
    assert _recorded_order() == ("OTHER000001", None)
    # A re-announcement of an unordered item stays unordered.
    control.note_media_broadcast(USER, video_id="OTHER000001", reannounces="OTHER000001")
    assert _recorded_order() == ("OTHER000001", None)
    # An ordered mark is its own order, whatever it re-announces.
    mark = control.media_halt_mark(USER, media_order=7, media_scope=SCOPE)
    control.note_media_broadcast(USER, mark, video_id="OTHER000001", reannounces="OTHER000001")
    assert _recorded_order() == ("OTHER000001", (SCOPE, 7, None))


def test_ta1_every_same_item_re_announcement_names_the_item_it_re_announces():
    """The three station re-announcements say which item they re-announce;
    advances, steps and saved playlists do not."""
    from app.agent.radio import player

    assert "reannounces" in inspect.signature(player.broadcast_radio_track).parameters
    assert "reannounces=reannounces" in inspect.getsource(player.broadcast_radio_track)
    toggle = inspect.getsource(ws_chat._handle_radio_toggle_locked)
    assert "reannounces=seed_video_id" in toggle
    anchor = inspect.getsource(ws_chat._broadcast_track_for_mode)
    assert 'reannounces=track.video_id if trigger == "reanchor" else ""' in anchor
    swap = inspect.getsource(ws_chat._handle_radio_display_mode_locked)
    assert "reannounces=current_track.video_id" in swap
    import app.api.media_playlists as mp

    assert "reannounces" not in inspect.getsource(mp)


# ══ TA2. The English trailing tag: at the clause edge, never the title ═════

@pytest.mark.parametrize(("text", "query"), [
    # the verifier's three (test_r6v2_residue_typed.py, T6 regression)
    ('play "Well, Well, Well"', '"Well, Well, Well"'),
    ('play the song called "Well, okay"', 'the song called "Well, okay"'),
    ("play Well, Well, Well", "Well, Well, Well"),
    # pure_r6v2_media_intent_breadth / quoted_and_retract
    ("play Hey, Hey, Hey", "Hey, Hey, Hey"),
    ("play Okay, Okay", "Okay, Okay"),
    ("play Oh, Yeah", "Oh, Yeah"),
    ('play "Hello, okay"', '"Hello, okay"'),
    ("play the song called Well, okay", "the song called Well, okay"),
    ("play the song called hello, would you?", "the song called hello, would you?"),
    # PIN CHANGED (addendum 6 R6-11 TB1, writer:r6d-tenant; was "some jazz,
    # please" — "ONE tag only"): every closed tail tag at the end leaves the
    # query; the cases above stay whole by structure (quotes, a title marker,
    # a head of class words only), not by a count.
    ("play some jazz, please, okay?", "some jazz"),
])
def test_ta2_the_tail_strip_never_truncates_a_title(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


@pytest.mark.parametrize(("text", "query"), [
    ("play some jazz, would you?", "some jazz"),
    ("play Halo by Beyonce, could you?", "Halo by Beyonce"),
    ("play some jazz, okay?", "some jazz"),
    ("play some jazz, would you please?", "some jazz"),
    ('play "Halo", would you?', '"Halo"'),
    # PIN CHANGED (addendum 6 R6-17, writer:r7-tenant; was 'play Thank U,
    # Next'): 'next' is a word of the closed temporal/sequence class, so that
    # title takes the agent path (tests/test_media_r7_tenant.py); the control
    # — a comma inside a title is not a tag — stays pinned on a title with no
    # trouble word.
    ("play Well, Well, Well", "Well, Well, Well"),
])
def test_ta2_controls_a_real_trailing_tag_still_goes(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


# ══ TA3. A quoted or title-marked Persian span is the query ═══════════════

@pytest.mark.parametrize(("text", "query"), [
    ("بذار آهنگ «باشه»", "باشه"),
    ("آهنگ «خوب» رو بذار", "خوب"),
    ("آهنگ «آره» رو پخش کن", "آره"),
    ("آهنگ «میشه» رو بذار", "میشه"),
    ("یه آهنگ به اسم «باشه» پخش کن", "اسم باشه"),
    ("یه آهنگ به اسم باشه پخش کن", "اسم باشه"),
    # PIN CHANGED (addendum 6 R6-17, writer:r7-tenant; was «یه آهنگ از ابی
    # بذار به اسم باشه» → 'ابی اسم باشه'): the «بذار» let-rule now covers the
    # WHOLE complement, «به اسم» spans included, and «باشه» has the
    # subjunctive shape ('let it be'), so after «بذار» that sentence declines
    # (tests/test_media_r7_tenant.py). The postposed marked title itself is
    # pinned after the light verb, where no let-reading exists:
    ("یه آهنگ از ابی پخش کن به اسم باشه", "ابی اسم باشه"),
])
def test_ta3_a_named_title_is_never_dropped_and_never_a_variety_play(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)
    assert ask.variety is False, ("a named title is that song, not a random one", ask)


@pytest.mark.parametrize(("text", "query", "variety"), [
    ("آهنگ «سلام» رو پخش کن", "سلام", False),
    ("آهنگ سلام آخر رو بذار", "سلام آخر", False),
    ("یه آهنگ از ابی پخش کن", "ابی", True),
    ("باشه یه آهنگ بذار مرسی", "آهنگ", True),
])
def test_ta3_controls(text, query, variety):
    ask = media_request(text)
    assert ask is not None and (ask.query, ask.variety) == (query, variety), (text, ask)


def test_ta3_a_quoted_title_is_not_the_callers_negation():
    """The title span is the caller's title, not the caller's own clause:
    a quoted «نده» is searched; the same word unquoted still declines (T5),
    and a negated verb after a marked title still declines."""
    ask = media_request("آهنگ «نده» رو پخش کن")
    assert ask is not None and ask.query == "نده", ask
    assert media_request("آهنگ نده رو پخش کن") is None
    assert media_request("یه آهنگ به اسم باشه پخش نکن") is None


# ══ TA4. An English retraction declines ════════════════════════════════════

@pytest.mark.parametrize("text", [
    # the verifier's three
    "Play Halo. Actually, no.",
    "play Halo, never mind",
    "Play some jazz. Never mind.",
    # the same shape, other spellings
    "play Halo, actually no",
    "play Halo, no wait",
    "play Halo, or actually don't",
    "play some jazz, dont",
    "play some jazz, nevermind",
    "play jazz, not Halo",
    # the Persian arm reads an English retraction too
    "یه آهنگ از ابی پخش کن. Actually, no.",
    # and the Persian negation guards the English arm («play …» mixed script)
    "play آهنگ ابی، نه",
])
def test_ta4_a_retracted_request_never_becomes_a_play(text):
    assert media_request(text) is None, text


@pytest.mark.parametrize(("text", "query"), [
    # a negation inside a CLOSED DOUBLE quote is the title. PIN CHANGED
    # (addendum 6 R6-17, writer:r7-tenant): "play 'No Tears Left to Cry'"
    # moved to the agent-path list below — a single quote never shields.
    ('play "No, No, No"', '"No, No, No"'),
    ('play "No Tears Left to Cry"', '"No Tears Left to Cry"'),
    ("play «No Tears Left to Cry»", "«No Tears Left to Cry»"),
])
def test_ta4_controls_a_title_with_a_negation_still_plays(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


@pytest.mark.parametrize("text", [
    # PINS CHANGED (addendum 6 R6-11 TB1, writer:r6d-tenant; each was a
    # fast-path play of the whole text): an English negation lexeme anywhere
    # outside a closed quote declines — these titles take the agent path and
    # still play, one round trip later ("titles containing one — 'No Tears
    # Left to Cry' — take the agent path").
    "play No Tears Left to Cry",
    "play Don't Stop Me Now",
    "play I Won't Back Down",
    # …and a comma followed by a lowercase non-tag is a second clause, not a
    # title part: the agent answers.
    "play Halo, by Beyonce",
    # PIN CHANGED (addendum 6 R6-17, writer:r7-tenant; was a fast-path play of
    # "'No Tears Left to Cry'"): a SINGLE quote never shields — an elision
    # apostrophe pair ('80s … Guns N') read as one shielded the caller's own
    # 'not' (r6d verifier finding a) — so only double / guillemet /
    # curly-double quotes keep a trouble word out of the scan.
    "play 'No Tears Left to Cry'",
])
def test_ta4_tb1_a_negated_or_two_clause_title_takes_the_agent_path(text):
    assert media_request(text) is None, text


# ══ TA5. The media noun carries its suffix in every spelling ═══════════════

@pytest.mark.parametrize("text", [
    "آهنگ‌های ابی رو پخش کن", "آهنگ های ابی رو پخش کن", "آهنگهای ابی رو پخش کن",
    "آهنگای ابی رو پخش کن", "یه آهنگ از آهنگ‌های ابی بذار", "می‌شه آهنگ‌های ابی رو پخش کنی؟",
])
def test_ta5_every_spelling_of_the_plural_searches_the_subject(text):
    ask = media_request(text)
    assert ask is not None and ask.query == "ابی", (text, ask)


@pytest.mark.parametrize(("text", "query"), [
    ("پادکست‌های جدید رو پخش کن", "جدید"),
    ("آهنگ‌های شاد پخش کن", "شاد"),
    ("ترانه‌ای از ابی بذار", "ابی"),
    ("ویدیوی ابی رو بذار", "ابی"),
    # a subject-less plural is the user's own noun, never the suffix
    ("آهنگ‌ها رو پخش کن", "آهنگ"),
])
def test_ta5_the_suffix_is_never_the_query(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


@pytest.mark.parametrize("text", [
    "آهنگ‌هاش رو پخش کن",      # his/its songs
    "آهنگامو بذار",             # my songs
    "آهنگه رو بذار",            # THE song (the one we talked about)
    "پلی‌لیستمو بذار",          # my playlist
    "فیلمش رو بذار",            # his film
    # the suffixed noun no longer hides «another» / «next»
    "آهنگ‌های دیگه بذار",
    "آهنگ‌های بعدی رو بذار",
])
def test_ta5_a_referential_noun_with_no_named_subject_declines(text):
    assert media_request(text) is None, text


def test_ta5_the_verifier_suffix_probe_reads_one_word():
    got = {t: media_request(t) for t in (
        "آهنگ‌های ابی رو پخش کن", "آهنگ های ابی رو پخش کن", "آهنگهای ابی رو پخش کن")}
    assert {a.query for a in got.values()} == {"ابی"}, got
    assert all("های" not in a.query.split() for a in got.values()), got


# ══ TA6. «بذار کنار» is "put aside" ════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "آهنگ رو بذار کنار", "کنار بذار", "آهنگ رو کنار بذار", "بذارش کنار",
    "آهنگ رو بزن کنار", "این آهنگ رو بذار کنار لطفا",
])
def test_ta6_put_aside_is_not_a_play(text):
    assert media_request(text) is None, text


@pytest.mark.parametrize(("text", "query"), [
    # «کنار» that is not the placing verb's particle is content
    ("آهنگ کنار دریا رو بذار", "کنار دریا"),
    ("یه آهنگ بذار", "آهنگ"),
])
def test_ta6_controls(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


# ══ Both fast paths read the same verdict (relay + typed chat) ════════════

async def test_both_fast_paths_read_the_same_r6c_residue(rig):
    from app.services import live_voice_protocol as lvp

    for text, query in (
        ("آهنگ‌های ابی رو پخش کن", "ابی"),
        ("بذار آهنگ «باشه»", "باشه"),
        ("play Well, Well, Well", "Well, Well, Well"),
    ):
        assert lvp._media_request(text) == media_request(text), text
        q: asyncio.Queue = asyncio.Queue()
        before = len(rig.searches)
        result = await asyncio.wait_for(ws_chat._fast_media_check(text, USER, q), timeout=2.0)
        assert result is not None and rig.searches[before:] == [query], (text, rig.searches)
    for text in ("play Halo, never mind", "آهنگ‌هاش رو پخش کن", "آهنگ رو بذار کنار"):
        assert lvp._media_request(text) is None, text
        q = asyncio.Queue()
        before = len(rig.searches)
        assert await asyncio.wait_for(ws_chat._fast_media_check(text, USER, q), timeout=2.0) is None
        assert rig.searches[before:] == [], (text, rig.searches)


def test_the_one_class_lexicon_is_untouched():
    """TA2-TA6 add structure, never words to the relay's shared class."""
    import hashlib

    from test_media_r6b_tenant import _SNAPF_LEXICON_SHA256

    digest = hashlib.sha256("\x00".join((
        media_intent.ASSENT_WORDS_TEXT,
        "\x01".join(media_intent.ASSENT_PHRASES_TEXT),
        media_intent.CONTENT_FREE_WORDS_TEXT,
    )).encode("utf-8")).hexdigest()
    assert digest == _SNAPF_LEXICON_SHA256


# ══ TA7. ws_browser's media_play is not a media-control device ═════════════

def test_ta7_ws_browser_media_play_is_outside_the_media_control_plane():
    """Code evidence (the decision is: leave it, do not record it). The
    browser agent's play_media sends ONE frame on its own /ws/browser socket
    (`websocket.send_json`, the socket `_run_browser_agent_inner` hands to
    `_exec_browser_tool`). That socket is never registered in the chat fan-out
    (`ws_chat._register_ws_queue`, called only by the chat socket itself), so
    the stop/pause frames `control._execute_transport` sends through
    `broadcast_to_user` never reach it and `media_command_audience` never
    counts it: nothing the gate decides (a media_stop, `newer_playing`) can
    act on it. Recording it would make a phone-side verdict describe a pane
    the stop cannot touch, and overwrite the causal identity of what plays on
    the chat audience — the TA1 bug class."""
    from app.api import ws_browser

    src = inspect.getsource(ws_browser)
    for chat_plane in ("broadcast_to_user", "_user_ws_queues", "_register_ws_queue",
                       "send_media_play", "media_stop", "media_pause"):
        assert chat_plane not in src, chat_plane
    tool = inspect.getsource(ws_browser._exec_browser_tool)
    play = tool[tool.index('elif name == "play_media"'):tool.index('elif name == "done"')]
    assert "await websocket.send_json({" in play and '"type": "media_play"' in play
    assert play.count("send_json(") == 1
    app_dir = pathlib.Path(control.__file__).resolve().parents[2]
    registrars = sorted(
        str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py")
        if "_register_ws_queue(" in p.read_text(encoding="utf-8")
    )
    assert registrars == ["api/ws_chat.py"], registrars
    transport = inspect.getsource(control._execute_transport)
    assert "ws_chat.broadcast_to_user(" in transport
