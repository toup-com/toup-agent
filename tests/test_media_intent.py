"""The shared play predicate: one verdict for three fast paths.

`ws_chat._fast_media_check`, `query_intent.classify_query_intent` and the
GPT-Live relay each have to answer "is this a request to PLAY something", and
before `media_intent` they answered it two different ways and not at all.

The part that needs pinning is not the happy path — it is DECLINING.
`media_request` returning None means "hand this to the agent", which is always
safe; returning a MediaAsk means "search YouTube for this and start it", which
is irreversible from the listener's point of view. The 2026-07-31 recording is
a requested song replaced eighteen seconds in by a stranger's track, so every
case below where the subject is a back-reference is a case where guessing
produces exactly that.
"""
from __future__ import annotations

import pytest

from app.agent.media_intent import (
    fa_media_score,
    is_error_result,
    media_request,
    normalize_fa,
    requested_mode_fa,
)


# ── Normalisation ────────────────────────────────────────────────────────

def test_arabic_kaf_and_yeh_fold_to_their_persian_forms():
    assert normalize_fa("كتاب يك") == normalize_fa("کتاب یک")


def test_zwnj_becomes_a_space_not_nothing():
    """`پلی‌لیست` and `پلی لیست` are one word typed two ways. Joining them
    into `پلیلیست` would match neither spelling."""
    assert normalize_fa("پلی‌لیست") == "پلی لیست"


def test_persian_and_arabic_digits_become_ascii():
    assert normalize_fa("۱۲۳") == "123"
    assert normalize_fa("٤٥٦") == "456"


def test_normalisation_is_idempotent():
    for s in ("یه آهنگ پخش كن", "Play Some Music", "۱۲ ايميل"):
        assert normalize_fa(normalize_fa(s)) == normalize_fa(s)


def test_normalisation_leaves_ascii_alone_beyond_case_and_spacing():
    assert normalize_fa("  Play   Bohemian  Rhapsody ") == "play bohemian rhapsody"
    assert normalize_fa("https://youtu.be/x") == "https://youtu.be/x"


# ── Scoring ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("message,at_least", [
    ("یه آهنگ پخش کن", 3),
    ("موسیقی بذار", 3),
    ("یه ویدیو بذار", 3),
    ("بذار یه چیزی گوش بدم", 3),
    ("پلی کن", 2),
])
def test_a_persian_play_ask_scores(message, at_least):
    assert fa_media_score(message) >= at_least


@pytest.mark.parametrize("message", [
    # The stems are prefixes of ordinary words. `پلی` is inside پلیس
    # (police) and `گوش` is inside گوشی (phone); a right boundary is the
    # only thing keeping those out, and `\b` cannot help — every Persian
    # letter is a word character, so `\b` sits at the space either way.
    "پلیس رو خبر کن",
    "گوشی رو بده",
    # A consume verb on its own is a request for attention, not for music.
    "به حرفم گوش بده",
    # Ordinary non-media asks.
    "برام چک کن که ایونتش کیه",
    "آخرین اخبار رو بگو",
    "ایمیل‌هام رو چک کن",
])
def test_a_non_media_persian_ask_scores_zero(message):
    assert fa_media_score(message) == 0


def test_english_scores_zero_here_on_purpose():
    """query_intent keeps its own English media families — they also cover
    image generation, send_photo and tts. Scoring English twice would
    double-count and change which category wins a tie."""
    assert fa_media_score("play some music") == 0
    assert fa_media_score("generate an image of a cat") == 0


# ── Extraction: what the fast paths act on ───────────────────────────────

@pytest.mark.parametrize("message,query,variety", [
    ("یه آهنگ از ابی پخش کن", "ابی", True),
    ("یه موزیک ویدیو از شادمهر بذار", "شادمهر", True),
    ("یه آهنگ خوب از ابی بذار", "ابی", True),
])
def test_a_named_subject_becomes_the_query(message, query, variety):
    ask = media_request(message)
    assert ask is not None and ask.query == query
    assert ask.variety is variety


@pytest.mark.parametrize("message", [
    "اوکی بعد به چیز دیگه میخوام برام یه دونه آهنگ دریکو پلی کن",
    "باشه، بعد یه چیز دیگه می خوام، برام یه دونه آهنگ دریکو پلی کن",
])
def test_topic_transition_before_complete_play_request(message):
    """A conversational lead must not send an otherwise complete play ask to a slow agent turn."""
    ask = media_request(message)
    assert ask is not None
    assert ask.query == "Drake"
    assert ask.variety is True


@pytest.mark.parametrize("message", [
    # Visible final bubble in the owner's 2026-09-24 TestFlight recording.
    "برنامه... یه... یه آهنگ... آهنگ دری... دریک پ... پلی کن",
    "برنامه ... یه ... یه آهنگ ... آهنگ دری ... دریک پ ... پلی کن",
    "یه… یه آهنگ… آهنگ دری… دریک پ… پلی کن",
    "یه یه آهنگ آهنگ دریک پلی کن",
    "یه آهنگ دریک پلی کن",
    "دریک پلی کن",
    "دریک پخش کن",
    "دریک رو پلی کن",
    "برام دریکو پلی کن",
])
def test_observed_drake_false_starts_and_short_artist_asks(message):
    ask = media_request(message)
    assert ask is not None
    assert (ask.query, ask.variety, ask.mode, ask.lang) == ("Drake", True, None, "fa")


def test_observed_drake_false_start_keeps_the_agent_media_tool_as_a_fallback():
    from app.agent.query_intent import classify_query_intent

    intent = classify_query_intent(
        "برنامه... یه... یه آهنگ... آهنگ دری... دریک پ... پلی کن"
    )
    assert "play_media" in intent.tool_names
    assert intent.category == "media" and intent.include_media_section


def test_clean_drake_song_ask_keeps_its_existing_variety_semantics():
    ask = media_request("آهنگ دریک پلی کن")
    assert ask is not None
    assert (ask.query, ask.variety) == ("Drake", False)


@pytest.mark.parametrize("message,query,variety", [
    ("یه یه آهنگ آهنگ ابی پلی کن", "ابی", True),
    ("یه... یه آهنگ... آهنگ ابی پلی کن", "ابی", True),
    ("یه...یه آهنگ...آهنگ ابی پلی کن", "ابی", True),
    ("یک یک ترانه ترانه شادمهر پخش کن", "شادمهر", True),
    ("آهنگ... آهنگ ابی پلی کن", "ابی", False),
])
def test_repeated_persian_request_head_repairs_for_other_artists(message, query, variety):
    ask = media_request(message)
    assert ask is not None and (ask.query, ask.variety) == (query, variety)


@pytest.mark.parametrize("message", [
    "برنامه یه آهنگ دریک پلی کن",  # no pause: «برنامه» may be the subject
    "برنامه بساز... یه آهنگ دریک پلی کن",  # another request
    "یه آهنگ دری دریک پلی کن",  # unmarked partial word could be a title
    "یه آهنگ دریک پ پلی کن",
    "یه آهنگ دری... دریک پلی نکن",
    "همون دریک رو پلی کن",
    "دریک رو دوباره پلی کن",
    "دریک پلی کن بعد هوا رو بگو",
    "دریک رو تو نتفلیکس پلی کن",
    "دریک پخش کن و ایمیل بفرست",
    "بزن دریک",  # no explicit, known-artist play verb
    "یه یه ایمیل ایمیل بفرست",  # repeated non-media noun
    "یه یه آهنگ آهنگ ابی پلی نکن",  # original negation survives repair
    "یه یه آهنگ آهنگ ابی پلی کن و هوا رو بگو",  # still two requests
    "یه یه آهنگ آهنگ همون رو پلی کن",  # original reference survives repair
    "یه یه آهنگ آهنگ بعدی رو پلی کن",  # original queue navigation survives
    "یه یه آهنگ آهنگ قبلی بذار",
])
def test_drake_repair_does_not_guess_through_ambiguous_or_unsafe_words(message):
    assert media_request(message) is None


def test_known_artist_object_marker_does_not_change_an_explicit_title():
    ask = media_request("آهنگ «دریکو» پخش کن")
    assert ask is not None
    assert ask.query == "دریکو"


def test_partial_drake_words_are_preserved_when_explicitly_named_as_a_title():
    ask = media_request("آهنگ «دری دریک» پلی کن")
    assert ask is not None
    assert ask.query == "دری دریک"


def test_unfragmented_quoted_drake_title_is_not_the_artist_alias():
    ask = media_request("آهنگ «دریک» پلی کن")
    assert ask is not None and ask.query == "دریک"


@pytest.mark.parametrize("message", [
    "اوکی بعد به چیز دیگه میخوام اول هوا رو بگو بعد برام یه آهنگ دریک پلی کن",
    "اوکی بعد به چیز دیگه میخوام آهنگ دریکو پخش نکن",
    "اوکی بعد به چیز دیگه میخوام آهنگ بعدی رو پلی کن",
    "اوکی بعد به چیز دیگه میخوام درباره موسیقی توضیح بده",
])
def test_topic_transition_does_not_bypass_media_safety_grammar(message):
    assert media_request(message) is None


@pytest.mark.parametrize("message,query", [
    ("یه آهنگ پخش کن", "آهنگ"),
    ("موسیقی بذار", "موسیقی"),
    ("برام یه پادکست بذار", "پادکست"),
])
def test_a_subject_less_ask_uses_the_users_own_word(message, query):
    """Searching "music" for a Persian sentence returns an English catalogue,
    which is the wrong answer to a question asked in Persian. Subject-less is
    open-ended by construction, so `variety` is set."""
    ask = media_request(message)
    assert ask is not None
    assert ask.query == query
    assert ask.variety is True


@pytest.mark.parametrize("message", [
    "برای خودم یه آهنگ بذار",     # "play ME a song" — a possessive, not a reference
    "همین الان یه آهنگ بذار",      # "play a song RIGHT NOW" — an adverb
])
def test_ordinary_asks_are_not_mistaken_for_back_references(message):
    """The decline list is the dangerous half in the other direction too:
    over-declining costs the whole latency win this predicate exists for, and
    both of these read as "the usual one" to a naive possessive/deictic
    match."""
    ask = media_request(message)
    assert ask is not None and ask.query == "آهنگ"


def test_something_to_listen_to_reaches_a_generic_query():
    ask = media_request("بذار یه چیزی گوش بدم")
    assert ask is not None and ask.variety is True and ask.query


# ── Declining, which is the half that prevents an incident ───────────────

@pytest.mark.parametrize("message", [
    # "the usual one" / "that one again" — only the agent knows which.
    "آهنگ همیشگی رو بذار",
    "همون آهنگ رو دوباره بذار",
    # The user's OWN saved list. A YouTube search answers it with a
    # stranger's, which is the 2026-07-31 class.
    "پلی‌لیست منو بذار",
    # A bare imperative with nothing to play.
    "پلی کن",
    "بذار",
])
def test_a_back_reference_is_handed_to_the_agent(message):
    assert media_request(message) is None


@pytest.mark.parametrize("message", [
    "play X on netflix",
    "یه فیلم از نتفلیکس بذار",
    # Spelled with a ZWNJ, which is how a Persian keyboard writes it. The
    # guard runs over the FOLDED text as well as the raw one: ZWNJ is not
    # `\s` to Python's `re`, so on the raw string the `نت\s*فلیکس` arm misses
    # and the ask fell through to a YouTube search.
    "یه فیلم از نت‌فلیکس بذار",
])
def test_another_catalogue_is_handed_to_the_agent(message):
    """A YouTube search cannot answer a Netflix ask, and the agent is the one
    that should say so."""
    assert media_request(message) is None


@pytest.mark.parametrize("message", [
    "حال",
    "برام",
    "",
    "   ",
    "what is the weather today",
])
def test_a_non_play_message_is_declined(message):
    assert media_request(message) is None


@pytest.mark.parametrize("message", [
    # Questions ABOUT music. The old gate let a bare strong token through, so
    # "what is music?" became a YouTube search for "چیه؟" and played whatever
    # came back — `media_resolve.pick_best` has no relevance floor, it returns
    # `scored[0]` for any non-empty pool, and both consumers of this predicate
    # act irreversibly (ws_chat broadcasts `media_play`; the relay calls
    # `_play_media_direct`).
    "موسیقی چیه؟",
    "پادکست چیه اصلا؟",
    "آهنگ مورد علاقه تو چیه؟",
    "نظرت درباره موسیقی سنتی چیه",
    # Statements that merely mention music.
    "درباره موسیقی کلاسیک برام توضیح بده",
    "دیشب یه آهنگ قشنگ شنیدم",
    "موزیک برای تمرکز خوبه یا نه",
    # The leak reached English: one Persian codepoint routes the whole
    # sentence to the Persian arm, so a translation question scored 2 and
    # extracted "what does mean in english?" as a search query.
    "what does آهنگ mean in english?",
])
def test_a_sentence_that_only_MENTIONS_music_is_not_a_play_request(message):
    """The classifier and the fast paths want different bars and this is why.
    `fa_media_score` still scores these 2, which widens the tool set so the
    agent is handed `play_media` — free and reversible. Acting on them is not:
    nothing downstream asks whether the result answers the question."""
    assert media_request(message) is None


def test_mentioning_music_still_scores_for_the_CLASSIFIER():
    """The inverse of the test above, and the reason the fix is free. If the
    score moved too, a Persian question about music would lose the media tool
    set and the agent could not answer it by playing something when asked to."""
    for message in ("موسیقی چیه؟", "دیشب یه آهنگ قشنگ شنیدم"):
        assert fa_media_score(message) >= 2


# ── The English arm is the old ws_chat behaviour, byte for byte ──────────

@pytest.mark.parametrize("message,query", [
    ("play Bohemian Rhapsody", "Bohemian Rhapsody"),
    ("play me a song by Radiohead", "Radiohead"),
    ("put on Daft Punk", "Daft Punk"),
    ("play Hotel California on youtube", "Hotel California"),
    # Quotes survive, because the first pattern matches before the
    # quote-stripping one gets a look. Pinned as-is: this is the behaviour
    # typed chat has shipped, and "moved verbatim" has to mean verbatim.
    ('play "Blue Monday"', '"Blue Monday"'),
])
def test_the_two_moved_english_patterns_extract_what_they_always_did(message, query):
    ask = media_request(message)
    assert ask is not None and ask.query == query
    assert ask.lang == "en"


def test_a_mixed_script_ask_takes_the_persian_arm():
    ask = media_request("play کن یه آهنگ")
    assert ask is not None and ask.lang == "fa" and ask.query == "آهنگ"


# ── Which surface ────────────────────────────────────────────────────────

@pytest.mark.parametrize("message", [
    "یه موزیک ویدیو از شادمهر بذار",
    "یه ویدیو بذار",
    "یه فیلم بذار ببینیم",
    "برام یه پادکست بذار",
    "یه مستند درباره کوسه بذار",
])
def test_a_watch_ask_answers_video(message):
    assert requested_mode_fa(message) == "video"
    ask = media_request(message)
    assert ask is None or ask.mode == "video"


@pytest.mark.parametrize("message", [
    "یه آهنگ از ابی پخش کن",
    "موسیقی بذار",
])
def test_an_audio_ask_answers_none_never_song(message):
    """Absence of a video signal is not an audio request — only the caller
    knows its channel's default surface. Same contract as
    `radio.player.infer_requested_mode`, which this is half of."""
    assert requested_mode_fa(message) is None
    ask = media_request(message)
    assert ask is not None and ask.mode is None


def test_watch_is_intent_only_at_the_start_in_persian_too():
    """`radio.player` is deliberately conservative about "watch": mid-sentence
    it is far more likely to be part of a title, and mistaking one for the
    other puts a song on a screen nobody is looking at. The Persian half
    mirrors it — start-anchored verbs, unanchored compound nouns."""
    assert requested_mode_fa("ببین چی میگم") == "video"
    # …and a mid-sentence watch verb with no video noun does not flip it.
    assert requested_mode_fa("آهنگی که دیشب گفتم رو بذار") is None


def test_the_english_half_still_answers_through_the_shared_entry_point():
    from app.agent.radio.player import infer_requested_mode
    assert infer_requested_mode("play the music video for HUMBLE") == "video"
    assert infer_requested_mode("play Bohemian Rhapsody") is None
    # …and the Persian half is now reachable from the SAME call, which is what
    # makes the tool and the typed-chat fast path agree.
    assert infer_requested_mode("یه موزیک ویدیو از شادمهر بذار") == "video"


# ── The failure contract the three paths share ───────────────────────────

def test_the_failure_test_is_ONE_function_and_it_is_the_loose_one():
    """There is no `ERROR_PREFIX` constant to reach for instead. One sat beside
    this function with no consumer and disagreed with it on three of five
    realistic strings, so which of the two a caller picked decided whether a
    failure read as a success — the exact drift this module exists to end.
    The loose test is the one that matches every producer: `tool_executor`'s
    `result.startswith("ERROR")` and `api_v1`'s upper-cased test both carry no
    colon."""
    import app.agent.media_intent as _mi

    assert not hasattr(_mi, "ERROR_PREFIX")
    assert "ERROR_PREFIX" not in _mi.__all__
    # The three strings a colon-anchored constant would have called successes.
    assert is_error_result("ERROR - no track")
    assert is_error_result("ERRORS were found")
    assert is_error_result("error: nope")

    assert is_error_result("ERROR: could not start that track.")
    assert is_error_result("  error: nope")
    assert not is_error_result("Now playing: Bohemian Rhapsody.")
    assert not is_error_result("")


# ── `MediaAsk.lang` has a consumer ───────────────────────────────────────

@pytest.mark.parametrize("message,expected", [
    ("play Bohemian Rhapsody", "lang=en"),
    ("یه آهنگ پخش کن", "lang=fa"),
])
def test_the_language_tag_reaches_the_log_line(message, expected, caplog,
                                               monkeypatch):
    """`lang` was produced on every call and read by nothing — the round's own
    declared-but-unconsulted class. It is now on `[FAST-MEDIA]`, which is the
    only way to tell the NEW Persian arm apart from the English one in Loki
    after this ships: both arms now come down the same line.

    It is an enum, so it is inside the logging rule — the assertions below
    also re-prove that the user's words still are not."""
    import asyncio
    import logging

    import httpx

    from app.api import ws_chat

    class _Boom:
        async def __aenter__(self):
            raise RuntimeError("no network in tests")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Boom())

    with caplog.at_level(logging.DEBUG):
        asyncio.run(ws_chat._fast_media_check(message, "u1", asyncio.Queue()))

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "[FAST-MEDIA]" in blob, "the path declined — the test proves nothing"
    assert expected in blob, blob
    # The query and the message are still lengths, never content.
    assert message not in blob, blob
