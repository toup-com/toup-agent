"""A navigation word is never a search query (media repair M2).

Recording V1, 2026-09-22 (+00:46): the caller said «عوض بشه، بزن آهنگ بعدی»
("change it, play the next song"). `media_request` scored it 3 (noun آهنگ +
verb بزن), kept the residue «عوض بشه، بعدی» as the search query, and the relay's
play fast path searched YouTube for it — which is why the phone started a video
titled «شاید بعد از این ویدیو نظرت عوض بشه!…»: both residue words are in that
title. Typed chat shares `media_request`, so it had the identical bug.

"Next"/"previous" are navigation over what is ALREADY playing. They are
answered by the deterministic control (tenant `/internal/media-control`) or by
the agent, never by a search for the word itself. Declining is the module's
safe direction: the turn falls through to the control grammar or the agent.

What must NOT change: an English title that merely contains "Next" is still a
title ("Thank U, Next", "Next To Me"), so only the navigation SHAPES decline in
English. In Persian «بعدی» is an adjective ("the next one"); left over after the
verb, the media noun and the fillers are removed, it is always a reference to
the queue, never a subject.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.media_intent import media_request


# The exact production utterance (V1 frame f047; relay log U:4 chars=22).
PROD_NEXT = "عوض بشه، بزن آهنگ بعدی"


@pytest.mark.parametrize("message", [
    PROD_NEXT,
    "آهنگ بعدی رو بزن",
    "آهنگ بعدی رو پخش کن",
    "آهنگ بعدیو بذار",
    "آهنگ بعدیشو پخش کن",
    "آهنگ رو بزن بعدی",
    # Mixed script: when the Persian arm declines, the English arm must not
    # pick the sentence up and search for «آهنگ بعدی» instead.
    "play آهنگ بعدی",
    "play the next song",
    "play next song",
    "play the next track",
    "play the next one",
    "play next",
    "play the previous song",
    "play previous",
    "play skip",
])
def test_a_navigation_word_is_never_a_search_query(message):
    assert media_request(message) is None


@pytest.mark.parametrize("message,query", [
    # "Next" inside a title is a title, not a command. Only the navigation
    # shapes (next + media noun, or nothing but navigation) decline.
    ("play Next To Me", "Next To Me"),
    ("play Thank U, Next", "Thank U, Next"),
    ("play The Next Episode", "The Next Episode"),
])
def test_an_english_title_that_contains_next_still_plays(message, query):
    # PIN CHANGED (R2 addendum 6 R6-17): 'next' is in the closed temporal/sequence
    # class, so a title containing it takes the AGENT path (never a fast-path
    # guess); the agent receives the full words (R6-12/R6-18, test_media_r7_tenant).
    # The name is kept for continuity; `query` documents the title the agent gets.
    assert media_request(message) is None


@pytest.mark.parametrize("message,query", [
    # The ordinary asks around it are untouched.
    ("یه آهنگ از ابی پخش کن", "ابی"),
    ("یه آهنگ پخش کن", "آهنگ"),
    ("play Bohemian Rhapsody", "Bohemian Rhapsody"),
])
def test_ordinary_play_asks_are_unchanged(message, query):
    ask = media_request(message)
    assert ask is not None and ask.query == query


def test_typed_chat_fast_path_does_not_search_for_the_next_song(monkeypatch):
    """Typed chat reads the same predicate: the production sentence must reach
    the agent, not a YouTube scrape for «عوض بشه، بعدی»."""
    import httpx

    from app.api import ws_chat

    searched: list = []

    class _Recorder:
        def __init__(self, *a, **kw):
            searched.append(True)

        async def __aenter__(self):
            raise RuntimeError("no network in tests")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Recorder)
    queue: asyncio.Queue = asyncio.Queue()
    result = asyncio.run(ws_chat._fast_media_check(PROD_NEXT, "u1", queue))

    assert result is None
    assert searched == [], "the typed-chat fast path searched YouTube for a navigation word"
    assert queue.empty()


@pytest.mark.asyncio
async def test_relay_fast_path_does_not_play_a_search_for_next(monkeypatch):
    """The Live relay's play fast path, driven with provider-shaped events:
    the production "next" utterance must not reach `_play_media_direct`."""
    import test_live_harness as H

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    plays: list = []
    thinks: list = []

    async def play(_user_id, query, variety=False):
        plays.append(query)
        return "Now playing: X.", {"type": "youtube", "video_id": "v1", "title": "X"}

    async def think(*args, **kwargs):
        thinks.append(args)
        return "باشه.", "test-model"

    async def control(_user_id, action):
        return {"ok": True, "action": action, "reason": "advanced", "title": "Next"}

    H.patch_relay(monkeypatch, play=play, think=think, control=control)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(PROD_NEXT, 0, 350))
            provider.push(H.delegation("d1", 400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def settled():
        return any(
            frame.get("delegation_id") == "d1"
            and frame.get("phase") in {"completed", "failed"}
            for frame in client.of("delegation")
        )

    async def wait():
        deadline = asyncio.get_running_loop().time() + 5.0
        while not settled():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("delegation d1 never settled")
            await asyncio.sleep(0.01)

    client = H.FakeClient([H.config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-m2-next")

    assert plays == [], f"the play fast path searched YouTube for {plays!r}"


# ── «آهنگ بعد»: the ezafe, or «بعدی» with its ی swallowed (review F30) ──
# «بزن آهنگ بعد» / «آهنگ بعد رو بزن» is ordinary spoken Persian for "play the
# next song" (آهنگِ بعد), and a recogniser dropping the final ی of «بعدی» gives
# the same string. The token rule above only knew «بعدی»/«قبلی», so the residue
# «بعد» was searched as a title — the V1 failure again, one letter shorter.
# Bare «بعد» is an ordinary word ("then", "after", and a title word: «بعد از
# تو»), so it is navigation only directly ON a media noun, which has to be read
# on the whole sentence: the residue has already lost the noun.
FA_NEXT_WITHOUT_YEH = [
    "بزن آهنگ بعد",
    "آهنگ بعد رو بزن",
    "آهنگ بعد رو بذار",
    "آهنگ بعد رو پخش کن",
    "آهنگ بعدو بزن",
    "آهنگ بعدش رو بزن",
    "آهنگ قبل رو بزن",
    "موزیک بعد رو بذار",
    "ترانه بعد رو پخش کن",
    "ویدیوی بعد رو بذار",
    "پلی‌لیست بعد رو بذار",
    # The V1 production sentence with the final ی dropped.
    "عوض بشه، بزن آهنگ بعد",
    # Mixed script, both arms.
    "play آهنگ بعد",
    "play بعد",
]


@pytest.mark.parametrize("message", FA_NEXT_WITHOUT_YEH)
def test_the_ezafe_next_song_is_never_a_search_query(message):
    assert media_request(message) is None


@pytest.mark.parametrize("message", [
    # "next one" in English was already navigation; pinned beside its Persian
    # twin so the two cannot drift apart.
    "play the next one",
    "play the next one please",
    "put on the next one",
    "play next one",
])
def test_the_english_next_one_is_never_a_search_query(message):
    assert media_request(message) is None


@pytest.mark.parametrize("message,query", [
    # «بعد» that is NOT on the media noun is not navigation: a leading "then"
    # does not decline a real request.
    ("بعد یه آهنگ از ابی بذار", "ابی"),
    ("آهنگ ساری گلین رو پخش کن", "ساری گلین"),
    ("یه آهنگ از ابی پخش کن", "ابی"),
])
def test_a_bare_bad_elsewhere_does_not_decline_a_real_request(message, query):
    ask = media_request(message)
    assert ask is not None and query in ask.query, ask


def test_typed_chat_does_not_search_for_the_ezafe_next_song(monkeypatch):
    import httpx

    from app.api import ws_chat

    searched: list = []

    class _Recorder:
        def __init__(self, *a, **kw):
            searched.append(True)

        async def __aenter__(self):
            raise RuntimeError("no network in tests")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Recorder)
    for message in ("آهنگ بعد رو بزن", "عوض بشه، بزن آهنگ بعد"):
        queue: asyncio.Queue = asyncio.Queue()
        assert asyncio.run(ws_chat._fast_media_check(message, "u1", queue)) is None
        assert queue.empty()
    assert searched == [], "the typed-chat fast path searched YouTube for «بعد»"


@pytest.mark.asyncio
@pytest.mark.parametrize("utterance", ["آهنگ بعد رو بزن", "بزن آهنگ بعد"])
async def test_relay_never_plays_a_search_for_the_ezafe_next_song(monkeypatch, utterance):
    """The Live relay with provider-shaped events: the spoken «آهنگ بعد» must
    not reach `_play_media_direct` as a search for «بعد». It is answered by
    the deterministic control when the relay grammar knows the shape, and by
    the agent otherwise — never by a stranger's video."""
    import test_live_harness as H

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    plays: list = []
    controls: list = []
    thinks: list = []

    async def play(_user_id, query, variety=False):
        plays.append(query)
        return "Starting X.", {"type": "youtube", "video_id": "v1", "title": "X"}

    async def think(*args, **kwargs):
        thinks.append(args)
        return "باشه.", "test-model"

    async def control(_user_id, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "advanced", "title": "Next"}

    H.patch_relay(monkeypatch, play=play, think=think, control=control)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(utterance, 0, 350))
            provider.push(H.delegation("d1", 400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def settled():
        return any(
            frame.get("delegation_id") == "d1"
            and frame.get("phase") in {"completed", "failed", "cancelled"}
            for frame in client.of("delegation")
        ) or bool(controls)

    async def wait():
        deadline = asyncio.get_running_loop().time() + 5.0
        while not settled():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("delegation d1 never settled")
            await asyncio.sleep(0.01)

    client = H.FakeClient([H.config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-f30-next")

    assert plays == [], f"the play fast path searched YouTube for {plays!r}"
    assert controls == ["next"] or thinks, "the request was dropped instead of handed on"


# ── "Another song" and a leading «بعد» (reverify w5, new) ────────────────
# "play another song" was a YouTube search for "another song", «یه آهنگ دیگه
# بذار» a search for «آهنگ», «آهنگ دیگه‌ای بذار» a search for «ای», and
# «بعد یه آهنگ از ابی بذار» a search for «بعد ابی». "Another" is relative to
# what is already playing (the queue's next, or the agent's choice), and a
# sentence-opening «بعد» ("then") names nothing. None of them is a subject.

ANOTHER = [
    "play another song",
    "put on another song",
    "play another one",
    "play another",
    "play me another",
    "play something else",
    "play a different song",
    "play a different one",
    "play some other music",
    "play another track please",
    "play one more song",
    "play some more",
    "play a different version",
    # A subject beside "another" is still relative to what is on: the fast
    # path cannot promise it is a DIFFERENT song by them.
    "play another song by Ebi",
    "play other songs by ebi",
    "یه آهنگ دیگه بذار",
    "یه آهنگ دیگه پخش کن",
    "آهنگ دیگه‌ای بذار",
    "آهنگ دیگه ای بذار",
    "یه آهنگ دیگری پخش کن",
    "یه موزیک دیگه بذار",
    "یه آهنگ دیگه از ابی بذار",
    "آهنگای دیگه ابی رو بذار",
    "play یه آهنگ دیگه",
]


@pytest.mark.parametrize("message", ANOTHER)
def test_another_song_is_never_a_search_query(message):
    assert media_request(message) is None


@pytest.mark.parametrize("message,query", [
    # Titles that merely contain a relative word keep their subject.
    ("play Another Love", "Another Love"),
    ("play Another One Bites the Dust", "Another One Bites the Dust"),
    ("play Another Brick in the Wall", "Another Brick in the Wall"),
    ("play More Than Words", "More Than Words"),
    ("play One More Time", "One More Time"),
    ("play The Other Side", "The Other Side"),
    ("play more jazz", "more jazz"),
    # «دیگه» after the verb is the emphatic particle ("come on, play a song").
    ("یه آهنگ بذار دیگه", "آهنگ"),
])
def test_a_title_or_particle_with_a_relative_word_still_plays(message, query):
    ask = media_request(message)
    assert ask is not None and ask.query == query, ask


@pytest.mark.parametrize("message", [
    "بعد یه آهنگ از ابی بذار",
    "بعدش یه آهنگ از ابی بذار",
    "بعدا یه آهنگ از ابی بذار",
    "لطفا بعد یه آهنگ از ابی بذار",
    "بعد، یه آهنگ از ابی بذار",
])
def test_a_leading_then_is_never_part_of_the_query(message):
    ask = media_request(message)
    if message.startswith("بعدا"):
        # PIN CHANGED (R2 addendum 6 R6-17; r6d verifier finding h): «بعدا» means
        # 'later', a temporal/deferral word — the request waits for the agent.
        assert ask is None, ask
        return
    assert ask is not None, "a real request after «بعد» still plays"
    assert ask.query == "ابی", ask
    assert "بعد" not in ask.query


@pytest.mark.parametrize("message", [
    # Nothing is left once «بعد» is gone: declined, as a bare «بعد» already was.
    "بعد یه آهنگ بذار",
    "بعدا یه آهنگ بذار",
    "بعد آهنگ رو بزن",
    # "after this (song), play …" asks for an order this path cannot keep.
    "بعد از این یه آهنگ از ابی بذار",
])
def test_a_leading_then_with_nothing_else_or_an_after_clause_declines(message):
    assert media_request(message) is None


def test_play_titles_that_start_with_bad_keep_the_english_arm():
    # 'play بعد از تو' is the title «بعد از تو», spelled after an English verb.
    ask = media_request("play بعد از تو")
    assert ask is not None and ask.query == "بعد از تو"


def test_typed_chat_does_not_search_for_another_song_or_a_leading_then(monkeypatch):
    import httpx

    from app.api import ws_chat

    searched: list = []

    class _Recorder:
        def __init__(self, *a, **kw):
            searched.append(True)

        async def __aenter__(self):
            raise RuntimeError("no network in tests")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Recorder)
    for message in (
        "play another song", "یه آهنگ دیگه بذار", "آهنگ دیگه‌ای بذار",
        "play something else", "بعد از این یه آهنگ از ابی بذار",
    ):
        queue: asyncio.Queue = asyncio.Queue()
        assert asyncio.run(ws_chat._fast_media_check(message, "u1", queue)) is None, message
        assert queue.empty()
    assert searched == [], "the typed-chat fast path searched YouTube for a relative ask"


@pytest.mark.asyncio
@pytest.mark.parametrize("utterance", ["آهنگ دیگه‌ای بذار", "play something else"])
async def test_relay_never_plays_a_search_for_another_song(monkeypatch, utterance):
    """The Live relay: its own grammar does not read these as `next`, and the
    shared predicate used to hand them to `_play_media_direct` as a search
    («ای», "something else"). They go to the agent instead."""
    import test_live_harness as H

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    plays: list = []
    controls: list = []
    thinks: list = []

    async def play(_user_id, query, variety=False):
        plays.append(query)
        return "Starting X.", {"type": "youtube", "video_id": "v1", "title": "X"}

    async def think(*args, **kwargs):
        thinks.append(args)
        return "OK.", "test-model"

    async def control(_user_id, action):
        controls.append(action)
        return {"ok": True, "action": action, "reason": "advanced", "title": "Next"}

    H.patch_relay(monkeypatch, play=play, think=think, control=control)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(utterance, 0, 350))
            provider.push(H.delegation("d1", 400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def settled():
        return any(
            frame.get("delegation_id") == "d1"
            and frame.get("phase") in {"completed", "failed", "cancelled"}
            for frame in client.of("delegation")
        ) or bool(controls)

    async def wait():
        deadline = asyncio.get_running_loop().time() + 5.0
        while not settled():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("delegation d1 never settled")
            await asyncio.sleep(0.01)

    client = H.FakeClient([H.config(), wait, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-another-song")

    assert plays == [], f"the play fast path searched YouTube for {plays!r}"
    assert controls == ["next"] or thinks, "the request was dropped instead of handed on"
