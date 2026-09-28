"""R6-11: the tenant fast path is a WHITELIST with a safe default (addendum 6
R6-11 TB1-TB4, TA7 amended; r6b verifier probes `/private/tmp/toup-voice-r2-
review/probes/r6b-verify-tenant.QEUWAj`, results `r6b-a80eb9cd8dee99e1a.json`).

TB1  The fast path accepts a request only when the whole sentence parses as
     [assent/framing] + play verb + media noun and/or a NAMED title + [closed
     tail tags]; anything else declines to the agent (which plays it one round
     trip later). The caller's own negation / retraction anywhere outside a
     CLOSED quote declines; an unclosed quote declines; a relative clause
     declines; a second clause that is not a closed tail declines (comma,
     dash, double hyphen, parentheses, sentence break); a non-play complement
     of «بذار» declines («تموم شه», «بره», «بمونه», «کنار», «زمین»); a title
     span ends at the caller's framing words («برام», «لطفا», 'for me',
     'please'). An English negation lexeme anywhere outside a closed quote
     declines. The whitelist must NOT collapse to "decline everything": the
     ordinary asks still play on the fast path (pinned below, en + fa).
     [R6-17, writer:r7-tenant — tests/test_media_r7_tenant.py: the closed-
     class TROUBLE scan runs on the whole utterance; a SINGLE quote never
     shields; the «بذار» let-rule covers the whole complement; a MARKED title
     keeps its framing words (never truncate). The pins below that R6-17
     changed say so where they stand.]
TB2  The suffix join takes only the closed free-standing plural forms; never a
     bare «ای» / «ی» / «ا», never a split of a whole word; an ambiguous
     reading declines; a title is never truncated.
TB3  Single quotes (' ‘ ’) are quotes; an apostrophe is not.
TB4  A same-item re-announcement never resumes an item the caller paused by a
     newer halt: no media_play, the record stays paused.
TA7  (amended) ws_browser's media_play stays outside the media-control plane.

Everything drives the REAL typed fast path (`ws_chat._fast_media_check`),
endpoints, play tool, radio toggle / media_ended / display-mode handlers,
radio player and control through the rig of `tests/test_media_order_scope_r6.py`;
only the YouTube search, the YT Music station build, the variant lookups, the
autosave and the socket fan-out are stubbed. The relay's fast path calls the
same `media_intent.media_request` (pinned equal below).
"""
# The shared `rig` fixture is imported and then requested by name, which ruff
# reads as a redefinition.
# ruff: noqa: F811
from __future__ import annotations

import asyncio
import hashlib
import inspect
import pathlib

import pytest

from app.agent import media_intent
from app.agent.media_intent import media_request
from app.agent.radio import control
from app.agent.radio.playlist import StationTrack
from app.api import ws_chat
from test_media_order_scope_r6 import SCOPE, USER, _halt, _play, rig  # noqa: F401

JAZZ, JAZZ_MV, HALO = "JAZZ0000001", "JAZZMV00001", "HALO0000001"


async def _typed(rig, text):
    """(result, searched, media_play ids) of the typed fast path."""
    before = len(rig.searches)
    q: asyncio.Queue = asyncio.Queue()
    result = await asyncio.wait_for(ws_chat._fast_media_check(text, USER, q), timeout=2.0)
    frames = []
    while not q.empty():
        frames.append(q.get_nowait())
    plays = [f.get("video_id") for f in frames if f.get("type") == "media_play"]
    return result, rig.searches[before:], plays


async def _declines(rig, text):
    result, searched, plays = await _typed(rig, text)
    assert result is None and not searched and not plays, (
        f"{text!r}: searched {searched!r}, media_play {plays!r}")


def _relay_agrees(text):
    from app.services import live_voice_protocol as lvp

    assert lvp._media_request(text) == media_request(text), text


# ══ TB1 — the verifier's r6b cases: every one declines, both fast paths ════

VERIFIER_DECLINES = [
    # V-TA3: a title span never hides the caller's own negation
    "آهنگی به اسم سلام نباید پخش بشه",
    "آهنگ «باشه رو پخش نکن",
    "«آهنگ ابی رو پخش نکن",
    "آهنگ به اسم سلام نه ابی بذار",
    "آهنگی که اسمش یادم نمیاد رو بذار",
    "یه آهنگ پخش کن که اسمش نمیدونم چیه",
    # V-TA4a: a retraction after a dash / in parentheses
    "Play Halo — never mind.",
    "Play Halo — no, wait.",
    "play Halo -- never mind",
    "play Halo (actually no)",
    # V-TA4b: a retraction with no negation particle (en + fa)
    "یه آهنگ از ابی پخش کن، بی‌خیال",
    "یه آهنگ از ابی پخش کن، ولش کن",
    "play Halo, forget it",
    "play Halo. Cancel that.",
    "play Halo, scratch that",
    # V-TA6: the permissive «بذار» ("let it …")
    "آهنگ رو بذار تموم شه",
    "بذار آهنگ تموم بشه",
    "بذار آهنگ بره",
    "آهنگ رو بذار بمونه",
]


@pytest.mark.parametrize("text", VERIFIER_DECLINES)
async def test_tb1_the_verifier_cases_decline_on_the_typed_path(rig, text):
    await _declines(rig, text)


@pytest.mark.parametrize("text", VERIFIER_DECLINES)
def test_tb1_the_verifier_cases_decline_on_the_relay_path(text):
    assert media_request(text) is None, text
    _relay_agrees(text)


# ══ TB1 — the structural classes, beyond the verifier's phrases ═══════════

@pytest.mark.parametrize("text", [
    # the caller's negation / retraction inside a MARKED title span (never shielded)
    "یه آهنگ به اسم سلام نکن پخش کن",
    "یه آهنگ به اسم سلام بی‌خیال پخش کن",
    "play the song called Halo, never mind",
    "play the song called forget it",
    # an English negation lexeme anywhere outside a closed quote
    "play No Tears Left to Cry",
    "play Don't Stop Me Now",
    "play Halo no wait",
    # an unclosed quote (any style), or a closer with no opener
    'play "Halo, never mind',
    "play 'Halo",
    "play “Halo",
    "آهنگ باشه» رو پخش کن",
    # a relative clause
    "play the song whose name I forgot",
    "play the song that goes la la la",
    "یه آهنگ بذار که برقصیم",
    # a second clause that is not a closed tail
    "play Halo; then stop the music",
    "play Halo: the live version",
    "Play Halo. Play some jazz.",
    "play Halo, wait",
    "play Halo, by Beyonce",
    # PIN CHANGED (addendum 6 R6-17 whitelist collapse, writer:r7-tenant):
    # "play Halo (live)" now plays — a parenthesised VERSION TAG is part of
    # the title (tests/test_media_r7_tenant.py); a parenthesis holding
    # anything else is still a second clause:
    "play Halo (the one from the concert)",
    "آهنگ ابی، داریوش رو بذار",
    "یه آهنگ پخش کن. آهنگ ابی",
    "یه آهنگ پخش کن و صداشو زیاد کن",
    # a second verb of the grammar's own class inside the object: two requests
    "play halo stop",
    "play Halo, Stop",
    "play Stop and Stare",
    # the caller's Persian framing after an English verb is never a title word
    "play Halo لطفا",
    # a non-play complement of «بذار» / a word outside the grammar
    "بذار آهنگ پخش بشه",
    "آهنگ رو بذار کنار",
    "آهنگ رو کنار بذار",
    "این آهنگ رو پخش کن",
    "به این آهنگ گوش بده",
])
async def test_tb1_anything_outside_the_grammar_declines(rig, text):
    await _declines(rig, text)
    _relay_agrees(text)


# ══ TB1 — the whitelist does NOT collapse: ordinary asks still play ═══════

WHITELIST_PLAYS = [
    # the eight named in the task, en + fa
    ("یه آهنگ از ابی پخش کن", "ابی"),
    ("play some jazz", "some jazz"),
    ("آهنگ «باشه» رو بذار", "باشه"),
    ('play "Well, Well, Well"', '"Well, Well, Well"'),
    ("آهنگ‌های ابی رو پخش کن", "ابی"),
    ("play some jazz, would you?", "some jazz"),
    ("آهنگ سلام آخر رو بذار", "سلام آخر"),
    ("play Halo by Beyonce", "Halo by Beyonce"),
    # the r6b verifier's controls
    ("یه آهنگ به اسم باشه پخش کن", "اسم باشه"),
    ("آهنگ «سلام» رو پخش کن", "سلام"),
    ("Play Beyonce - Halo", "Beyonce - Halo"),
    # framing around the request (lead / pre-verb / tail)
    ("می‌شه آهنگ هالو از بیانسه رو برام پخش کنی؟", "هالو بیانسه"),
    ("آهنگ ابی رو می‌تونی بذاری؟", "ابی"),
    ("بله یه آهنگ از شجریان بذار لطفا", "شجریان"),
    ("یه آهنگ از امیر تتلو برام بذار", "امیر تتلو"),
    ("play Halo by Beyonce, could you?", "Halo by Beyonce"),
    # verb-first, postposed title, the 'let me listen' purpose clause
    ("پخش کن آهنگ سلام", "سلام"),
    ("بذار آهنگ hello", "hello"),
    ("یه آهنگ از ادل پخش کن به اسم hello", "ادل اسم hello"),
    ("بذار یه چیزی گوش بدم", "آهنگ"),
    # titles made of commas / class words, written as titles. PIN CHANGED
    # (addendum 6 R6-17, writer:r7-tenant): 'play Thank U, Next' left this
    # list — 'next' is a word of the closed temporal/sequence class, so the
    # title takes the agent path (tests/test_media_r7_tenant.py proves it
    # reaches the agent with the full words on both paths).
    ("play Well, Well, Well", "Well, Well, Well"),
    ("play Okay, Okay", "Okay, Okay"),
    # a negation INSIDE a closed DOUBLE quote is the title. PIN CHANGED
    # (addendum 6 R6-17, writer:r7-tenant): "play 'No Tears Left to Cry'"
    # left this list — a single quote never shields (it declines; r7).
    ('play "No, No, No"', '"No, No, No"'),
    ('play "No Tears Left to Cry"', '"No Tears Left to Cry"'),
    ("آهنگ «نده» رو پخش کن", "نده"),
    # R6-17 whitelist collapse: a parenthesised version tag is the title
    ("play Halo (live)", "Halo (live)"),
    # «کنار» that is not the placing particle
    ("آهنگ کنار دریا رو بذار", "کنار دریا"),
    # a Persian title after the English verb (no Persian grammar in it)
    ("play بعد از تو", "بعد از تو"),
]


@pytest.mark.parametrize(("text", "query"), WHITELIST_PLAYS)
async def test_tb1_the_ordinary_asks_still_play_on_the_fast_path(rig, text, query):
    result, searched, plays = await _typed(rig, text)
    assert result is not None and searched == [query] and plays, (text, searched, plays)
    _relay_agrees(text)


def test_tb1_the_whitelist_keeps_its_latency_win():
    """Declining is the safe default, not the goal: every ordinary ask above
    takes the fast path."""
    accepted = [text for text, _q in WHITELIST_PLAYS if media_request(text) is not None]
    assert len(accepted) == len(WHITELIST_PLAYS)


# ══ TB1 — «بذار» first: "put on" its object, never 'let' its clause ═══════

@pytest.mark.parametrize("text", [
    # the subjunctive 'let' clause: «ب» + stem + person ending, or «شه»
    "بذار آهنگ بره",
    "بذار آهنگ بمونه",
    "بذار آهنگ تموم شه",
    "بذار آهنگ تموم بشه",
    "بذار یه آهنگ تموم شه",
    "بذار آهنگ یه کم دیگه بمونه",
    # the shape is the rule: a name that has it costs one round trip
    "بذار آهنگ بیانسه",
])
def test_tb1_a_let_clause_after_bezar_declines(text):
    assert media_request(text) is None, text


@pytest.mark.parametrize(("text", "query"), [
    ("بذار آهنگ ابی، مرسی", "ابی"),
    ("بذار آهنگ سلام", "سلام"),
    ("بذار آهنگ بهار", "بهار"),
    # PINS CHANGED (addendum 6 R6-17 let-rule over the WHOLE complement,
    # writer:r7-tenant): «بذار یه آهنگ از بیانسه» ('بیانسه') and «بذار آهنگ
    # به اسم بره» ('اسم بره') now DECLINE — a subjunctive-shaped word anywhere
    # after «بذار», inside an «از» argument or a «به اسم» span too, reads as
    # 'let …' (tests/test_media_r7_tenant.py). The artist after the light verb
    # still plays:
    ("یه آهنگ از بیانسه پخش کن", "بیانسه"),
    ("بذار آهنگ «بمونه»", "بمونه"),
])
def test_tb1_bezar_with_its_object_still_plays(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


# ══ TB1 — a title span ends at the caller's framing words ═════════════════

# PINS CHANGED (addendum 6 R6-17 NEVER TRUNCATE, writer:r7-tenant): the two
# MARKED spans «یه آهنگ به اسم سلام برام / لطفا پخش کن» were pinned here with
# the framing cut out of the title; R6-17 resolves the TB1-vs-T3 conflict the
# r6d verifier raised the other way — a marked title keeps its words, framing
# leaves only as a separate tail (tests/test_media_r7_tenant.py pins them
# whole). The unmarked argument still ends at the preverbal framing slot:
@pytest.mark.parametrize(("text", "title", "framing"), [
    ("یه آهنگ از ابی برام پخش کن", "ابی", "برام"),
    ("یه آهنگ از ابی لطفا پخش کن", "ابی", "لطفا"),
])
async def test_tb1_the_title_span_is_the_title_not_the_framing(rig, text, title, framing):
    result, searched, plays = await _typed(rig, text)
    assert searched and title in searched[0].split() and framing not in searched[0].split(), (
        text, searched)


@pytest.mark.parametrize(("text", "query"), [
    # PIN CHANGED (addendum 6 R6-17 NEVER TRUNCATE, writer:r7-tenant; was
    # 'Halo'): a marked English title keeps its words ('Pray for Me').
    ("play a song called Halo for me", "Halo for me"),
    ("play the song called Halo, please", "the song called Halo"),
    # an assent / modal tag after a title marker may be the title (TA2)
    ("play the song called Well, okay", "the song called Well, okay"),
])
def test_tb1_an_english_marked_title_ends_at_the_framing(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


def test_tb1_framing_inside_a_marked_title_with_more_after_it_declines():
    # R6-17 (writer:r7-tenant): still None, now because 'Me' is a word of the
    # closed pronoun class — no longer because the framing cut the title.
    assert media_request("play the song called Please Please Me") is None


# ══ TB2 — the suffix join never truncates, never splits a whole word ══════

@pytest.mark.parametrize(("text", "title"), [
    ("آهنگ ای ایران رو پخش کن", "ای ایران"),
    ("آهنگ ای کاش رو بذار", "ای کاش"),
    ("آهنگ ای یار رو پخش کن", "ای یار"),
    ("فیلم ای وای رو پخش کن", "ای وای"),
    ("آهنگ ایمان رو پخش کن", "ایمان"),
    ("آهنگ‌های هایده رو بذار", "هایده"),
])
async def test_tb2_a_title_is_searched_whole(rig, text, title):
    result, searched, plays = await _typed(rig, text)
    assert searched == [title] and plays, (text, searched)


@pytest.mark.parametrize("text", [
    "آهنگ‌های ابی رو پخش کن", "آهنگ های ابی رو پخش کن", "آهنگهای ابی رو پخش کن",
    "آهنگای ابی رو پخش کن", "ترانه‌ای از ابی بذار",
])
def test_tb2_the_noun_suffix_in_every_spelling_is_still_the_noun(text):
    ask = media_request(text)
    assert ask is not None and ask.query == "ابی", (text, ask)


@pytest.mark.parametrize("text", [
    # a bare «ای» typed apart, alone after the noun: suffix or one-word title?
    "ترانه ای از ابی بذار",
    "آهنگ ای بذار",
    # «هامون»: the film Hamoun, or «ها»+«مون» "our films" — ambiguous
    "فیلم هامون رو پخش کن",
    # a plural + possessive written apart points back
    "آهنگ هاش رو پخش کن",
    "آهنگ‌هاش رو پخش کن",
])
def test_tb2_an_ambiguous_reading_declines(text):
    assert media_request(text) is None, text


def test_tb2_the_suffix_join_is_the_closed_plural_set_only():
    fold = media_intent._fa_fold
    assert fold("آهنگ های ابی") == fold("آهنگ‌های ابی") == "آهنگهای ابی"
    assert fold("آهنگ هایی از ابی") == "آهنگهایی از ابی"
    # never a bare «ای» / «ی» / «ا», never a whole word of another shape
    for text in ("آهنگ ای ایران", "آهنگ ی ابی", "آهنگ ا ابی", "آهنگ ایمان", "آهنگ هایده"):
        assert fold(text) == media_intent.normalize_fa(text), text


# ══ TB3 — single quotes are quotes; an apostrophe is not ══════════════════

# PINS CHANGED (addendum 6 R6-17, writer:r7-tenant): 'Talk to Me, Please' in
# straight / curly single quotes now DECLINES — a single quote never shields
# and 'Me' is a word of the closed pronoun class (tests/test_media_r7_tenant.py).
# What a single quote still does — keep its title whole — is pinned on titles
# with no trouble word:
@pytest.mark.parametrize(("text", "title"), [
    ("play the track 'Hello, Goodbye, Hello'", "'Hello, Goodbye, Hello'"),
    ("play ‘Hello, Goodbye, Hello’", "‘Hello, Goodbye, Hello’"),
    ("play the song 'Hello, okay'", "'Hello, okay'"),
])
async def test_tb3_a_single_quoted_title_is_searched_whole(rig, text, title):
    result, searched, plays = await _typed(rig, text)
    assert searched and title in searched[0] and plays, (text, searched)


@pytest.mark.parametrize(("text", "query"), [
    ("play Beyoncé's Halo", "Beyoncé's Halo"),
    # PIN CHANGED (addendum 6 R6-17, writer:r7-tenant; was "play the Beatles'
    # Yesterday"): 'yesterday' is a word of the closed temporal class, so that
    # title takes the agent path; the apostrophe fact is pinned on 'Help'.
    ("play the Beatles' Help", "the Beatles' Help"),
    ("play rock 'n' roll", "rock 'n' roll"),
])
def test_tb3_an_apostrophe_is_not_a_quote(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


def test_tb3_the_quote_reader():
    spans = media_intent._quote_spans
    assert spans("play 'Talk to Me, Please'") == [(5, 25)]
    assert spans("play ‘a’ and “b” and «c»") == [(5, 8), (13, 16), (21, 24)]
    assert spans("don't it’s the Beatles' song") == []
    assert spans("play 'Halo") is None and spans('play "Halo') is None
    assert spans("play Halo»") is None and spans("«آهنگ ابی") is None


# ══ TB4 — a re-announcement never resumes what the caller paused ══════════

def _media(frames):
    return [(f["type"], f.get("video_id", ""), f.get("reason", "")) for f in frames
            if f["type"].startswith("media_")]


def _recorded():
    state = control._user_state(USER)
    item = state.playing if state is not None else None
    if item is None:
        return None
    ev = item.event
    return (item.video_id, None if ev is None else (ev.scope, ev.order, ev.stamp), item.paused)


@pytest.fixture
def station(monkeypatch, rig):
    import app.agent.radio as radio_pkg
    import app.agent.radio.playlist as playlist_mod
    import app.api.media_playlists as mp

    rig.build_gate = None

    async def fake_station(seed, limit=50, **kw):
        if rig.build_gate is not None:
            await rig.build_gate.wait()
        meta = StationTrack(video_id=seed, title="Jazz Mix", artist="Various",
                            video_type="MUSIC_VIDEO_TYPE_ATV")
        return meta, [StationTrack(video_id=f"STN{i:08d}", title=f"Station {i}", artist="X")
                      for i in range(6)]

    async def none(*a, **kw):
        return None

    async def music_video(track, *a, **kw):
        return StationTrack(video_id=JAZZ_MV, title="Jazz Mix (Official Video)",
                            artist="Various", video_type="MUSIC_VIDEO_TYPE_OMV")

    monkeypatch.setattr(radio_pkg, "build_station", fake_station)
    monkeypatch.setattr(playlist_mod, "find_topic_version", none)
    monkeypatch.setattr(playlist_mod, "find_music_video", music_video)
    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", none)
    monkeypatch.setattr(mp, "autosave_station", none)
    return rig


async def _reseed(video_id=JAZZ, title="Jazz Mix"):
    await ws_chat._handle_radio_toggle(USER, {
        "type": "radio_toggle", "channel": "app", "enabled": True,
        "video_id": video_id, "title": title,
    })


async def _jazz(order=3):
    out = await asyncio.wait_for(_play("some jazz", order=order, scope=SCOPE), 3)
    assert out["ok"] is True and out["video_id"] == JAZZ, out


async def test_tb4_a_newer_pause_during_the_reseed_build_is_not_undone(station):
    """The verifier's (i): the reseed's toggle seed lands after the caller's
    newer pause — no media_play, the window and the station state still
    ship, the record stays the caller's paused jazz."""
    rig = station
    await _jazz(3)
    rig.build_gate = asyncio.Event()
    toggle = asyncio.create_task(_reseed())
    await asyncio.sleep(0.05)
    paused = await _halt(rig, action="pause", before=4, scope=SCOPE)
    assert (paused["ok"], paused["reason"]) == (True, "paused"), paused
    n = len(rig.frames)
    rig.build_gate.set()
    await asyncio.wait_for(toggle, 3)
    after = rig.frames[n:]
    assert not [f for f in after if f["type"] == "media_play"], _media(after)
    assert any(f["type"] == "radio_upcoming" and f["upcoming"] for f in after), after
    assert any(f["type"] == "radio_state" for f in after), after
    assert _recorded() == (JAZZ, (SCOPE, 3, None), True), _recorded()


async def test_tb4_a_reanchor_after_a_newer_pause_does_not_resume(station):
    """The verifier's (j): swap to the MV, pause (newer), a stale end for the
    pre-swap id re-anchors — no media_play; the record stays paused."""
    rig = station
    await _jazz(3)
    await _reseed()
    await ws_chat._handle_radio_display_mode(USER, {"channel": "app", "mode": "video"})
    assert _media(rig.frames)[-1] == ("media_play", JAZZ_MV, "mv_swap"), _media(rig.frames)
    paused = await _halt(rig, action="pause", before=4, scope=SCOPE)
    assert (paused["ok"], paused["reason"]) == (True, "paused"), paused
    n = len(rig.frames)
    await ws_chat._handle_media_ended(USER, {"channel": "app", "video_id": JAZZ})
    after = rig.frames[n:]
    assert not [f for f in after if f["type"] == "media_play"], _media(after)
    assert any(f["type"] == "radio_state" for f in after), after
    assert _recorded() == (JAZZ_MV, (SCOPE, 3, None), True), _recorded()


async def test_tb4_a_variant_swap_of_a_paused_item_is_held(station):
    """Pause first, then the Video tap: the swap would re-announce the paused
    item (and resume it). Held — the station keeps the variant the device
    holds, the mode flip ships as state, the record stays paused."""
    rig = station
    await _jazz(3)
    await _reseed()
    paused = await _halt(rig, action="pause", before=4, scope=SCOPE)
    assert (paused["ok"], paused["reason"]) == (True, "paused"), paused
    n = len(rig.frames)
    await ws_chat._handle_radio_display_mode(USER, {"channel": "app", "mode": "video"})
    after = rig.frames[n:]
    assert not [f for f in after if f["type"] == "media_play"], _media(after)
    sess = rig.manager.get(USER, "app")
    assert sess.current_track_id == JAZZ and sess.display_mode == "video"
    assert any(f["type"] == "radio_state" for f in after), after
    assert _recorded() == (JAZZ, (SCOPE, 3, None), True), _recorded()


async def test_tb4_control_without_a_pause_the_reseed_still_re_announces(station):
    """TA1 unchanged: no pause, the toggle seed is sent and keeps the order."""
    rig = station
    await _jazz(3)
    await _reseed()
    assert _media(rig.frames)[-1] == ("media_play", JAZZ, "toggle_seed"), _media(rig.frames)
    assert _recorded() == (JAZZ, (SCOPE, 3, None), False), _recorded()


async def test_tb4_control_an_older_pause_is_not_applied_so_nothing_is_held(station):
    """A pause asked BEFORE jazz answers newer_playing (nothing paused); the
    reseed's re-announcement is sent as before."""
    rig = station
    await _jazz(3)
    rig.build_gate = asyncio.Event()
    toggle = asyncio.create_task(_reseed())
    await asyncio.sleep(0.05)
    late = await _halt(rig, action="pause", before=2, scope=SCOPE)
    assert (late["ok"], late["reason"]) == (False, "newer_playing"), late
    rig.build_gate.set()
    await asyncio.wait_for(toggle, 3)
    assert _media(rig.frames)[-1] == ("media_play", JAZZ, "toggle_seed"), _media(rig.frames)
    assert _recorded() == (JAZZ, (SCOPE, 3, None), False), _recorded()


async def test_tb4_control_a_newer_play_after_the_pause_still_plays(station):
    """The caller's own newer request is never held: it plays and replaces."""
    rig = station
    await _jazz(3)
    await _reseed()
    assert (await _halt(rig, action="pause", before=4, scope=SCOPE))["reason"] == "paused"
    out = await asyncio.wait_for(_play("Halo", order=5, scope=SCOPE), 3)
    assert out["ok"] is True and out["video_id"] == HALO, out
    assert _media(rig.frames)[-1][:2] == ("media_play", HALO)
    assert _recorded() == (HALO, (SCOPE, 5, None), False), _recorded()


def test_tb4_the_hold_rule(monkeypatch):
    monkeypatch.setattr(control, "_latest_halts", type(control._pending_acks)())
    monkeypatch.setattr(ws_chat, "_radio_off_generations", {})
    held = control.holds_paused_reannouncement
    mark = control.media_halt_mark(USER, media_order=3, media_scope=SCOPE)
    control.note_media_broadcast(USER, mark, video_id=JAZZ, title="Jazz")
    assert not held(USER, JAZZ), "not paused: nothing to hold"
    state = control._user_state(USER)
    state.playing = state.playing.__class__(**{**state.playing.__dict__, "paused": True})
    assert held(USER, JAZZ)
    assert not held(USER, "OTHER000001"), "a different item is not a re-announcement"
    assert not held(USER, ""), "no re-announcement"
    newer = control.media_halt_mark(USER, media_order=5, media_scope=SCOPE)
    assert not held(USER, JAZZ, newer), "an ordered request of its own is never held"
    # the phone's X (an unordered OFF) stopped it: nothing is live
    ws_chat._radio_off_generations[(USER, "app")] = 1
    assert not held(USER, JAZZ)


def test_tb4_the_three_re_announcement_sites_are_guarded():
    from app.agent.radio import player

    body = inspect.getsource(player.broadcast_radio_track)
    assert "holds_paused_reannouncement(user_id, reannounces)" in body
    assert '"type": "radio_upcoming"' in body
    assert "holds_paused_reannouncement(user_id, reannounces, mark)" in inspect.getsource(
        control.send_media_play)
    swap = inspect.getsource(ws_chat._handle_radio_display_mode_locked)
    assert "holds_paused_reannouncement(user_id, current_track.video_id)" in swap


# ══ TA7 (amended) — ws_browser's media_play stays outside the plane ═══════

def test_ta7_amended_the_browser_pane_is_not_in_the_media_control_plane():
    """Integrator decision (R6-11 TA7 AMENDED): the media-control plane (stop /
    pause, the device gate, newer_playing) is defined over the chat / phone
    audience (`broadcast_to_user`). The web hub's browser-agent pane plays
    YouTube on its own /ws/browser socket and cannot receive media_stop /
    media_pause; recording its plays would make the gate claim control it
    does not have. So ws_browser is left as it is — this pin supersedes the
    r6b verifier's `test_ws_browser_play_media_goes_through_the_chokepoint`
    (kept unmodified in its probe dir)."""
    from app.api import ws_browser

    src = inspect.getsource(ws_browser)
    assert '"type": "media_play"' in src
    for plane in ("send_media_play", "note_media_broadcast", "broadcast_to_user",
                  "_user_ws_queues", "media_stop", "media_pause"):
        assert plane not in src, plane
    app_dir = pathlib.Path(control.__file__).resolve().parents[2]
    recorders = sorted(
        str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py")
        if "note_media_broadcast(" in p.read_text(encoding="utf-8")
    )
    assert recorders == ["agent/radio/control.py"], recorders


# ══ The ONE CLASS lexicon stays the relay's, byte for byte ════════════════

def test_the_one_class_lexicon_and_modal_set_are_untouched():
    from test_media_r6b_tenant import _SNAPF_LEXICON_SHA256

    digest = hashlib.sha256("\x00".join((
        media_intent.ASSENT_WORDS_TEXT,
        "\x01".join(media_intent.ASSENT_PHRASES_TEXT),
        media_intent.CONTENT_FREE_WORDS_TEXT,
    )).encode("utf-8")).hexdigest()
    assert digest == _SNAPF_LEXICON_SHA256
    modal = hashlib.sha256("\x01".join(media_intent.MODAL_FRAMING_TEXT).encode("utf-8")).hexdigest()
    assert modal == _MODAL_FRAMING_SHA256, modal


# sha256 of MODAL_FRAMING_TEXT as shipped before R6-11 (unchanged).
_MODAL_FRAMING_SHA256 = "7e7afd8db392d013737ded0ab72713962aacd4de86f41bd0b08a0cff19e87b19"
