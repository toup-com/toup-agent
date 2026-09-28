"""A Persian "play a song" must reach play_media. Recording 1 is why.

2026-09-20, GPT-Live, founder tenant: the caller asked for music in Persian,
`[PERF] query_intent: category=question, tools=3`, `tool_filter: 172 total →
11 for intent=question`, `tool_calls=0`, and an 89-second call ended with a
172-character spoken promise and silence. `TOOLS_QUESTION` is `{web_search,
web_fetch, recall_day}`; `play_media` lives only in `TOOLS_MEDIA`. Under the
prefix-stable layout the intent set becomes an OpenAI `allowed_tools`
restriction over the FULL array, so the model could SEE the tool, was told it
existed, and was forbidden to call it. The only honest output left was a
sentence.

Every natural Persian play phrasing of ≤8 words did this, because
`_MEDIA_KEYWORDS` and `_MEDIA_PATTERNS_RE` contained no non-Latin token at
all. The same request in English worked, which is the pair the owner reported
as "Persian requests need four repetitions".

Three layers are pinned here, and the third is the one that must not be
dropped in a later cleanup: a zero score in a script this classifier has no
vocabulary for means "I did not understand", not "no tools needed", and those
two have opposite correct answers.
"""
from __future__ import annotations

import pytest

from app.agent.query_intent import (
    _ALWAYS_INCLUDED_TOOLS,
    classify_query_intent,
    filter_tools_by_intent,
)


TOOL_DEFS = [
    {"name": name}
    for name in (
        "play_media", "tts", "send_file", "canvas", "web_search", "web_fetch",
        "browser", "recall_day", "create_job", "update_job", "start_mission",
        "write_file", "read_file", "generate_pdf", "memory_search",
        "memory_store", "memory_read_file", "navigate_to", "spawn",
        "routines__remind", "routines__create", "routines__list",
        "gmail__search_emails", "calendar__list_events",
    )
]


def _exposed(message: str) -> set:
    intent = classify_query_intent(message)
    return {t["name"] for t in filter_tools_by_intent(TOOL_DEFS, intent)}


def _can_play(message: str) -> bool:
    """The property that actually matters: could this turn call play_media?

    `full` sends everything (`filter_tools_by_intent` short-circuits), so it
    qualifies as surely as `media` does."""
    intent = classify_query_intent(message)
    return intent.category == "full" or "play_media" in _exposed(message)


# ── The case table, straight off the recording and the owner's script ────

PLAY_ASKS = [
    "یه آهنگ پخش کن",
    "موسیقی بذار",
    "آهنگ برام پلی کن",
    "بذار یه چیزی گوش بدم",
    "یه ویدیو بذار",
    "پلی کن",
    "یه آهنگ از ابی پخش کن",
    "یه موزیک ویدیو از شادمهر بذار",
    "آهنگ همیشگی رو بذار",
    "برام یه پادکست بذار",
    "play some music",
    "play کن یه آهنگ",
]


@pytest.mark.parametrize("message", PLAY_ASKS)
def test_a_play_request_can_actually_play(message):
    assert _can_play(message), (
        f"{message!r} classified {classify_query_intent(message).category!r} "
        f"with no play_media — the agent can only promise and fall silent"
    )


@pytest.mark.parametrize("message", PLAY_ASKS)
def test_a_play_request_classifies_as_media_not_merely_as_full(message):
    """`media` specifically, and this is the assertion that has teeth.

    The non-Latin backstop below ALSO rescues every one of these — a Persian
    play ask scoring zero would land in `full`, which exposes play_media too —
    so a test that only asked "can it play?" passed with the Persian grammar
    deleted entirely (verified by mutation). `full` is not the right answer
    either: it is the widest prompt, the widest tool array, and on voice it is
    the intent whose 170-name allow-list produced the 400. The grammar has to
    be what routes these, and the backstop has to be the net under it."""
    assert classify_query_intent(message).category == "media"


@pytest.mark.parametrize("message", PLAY_ASKS)
def test_a_play_request_is_not_the_tool_less_question_intent(message):
    """The specific landing zone from the recording. `question` carries two
    web tools and recall; nothing in it can start a song."""
    assert classify_query_intent(message).category != "question"


# ── Spelling is not meaning ──────────────────────────────────────────────
# A soft keyboard emits ARABIC kaf/yeh (ك U+0643 / ي U+064A) where Persian
# wants ک/ی, and ZWNJ where a space would do. Before `normalize_fa` every
# Persian regex in the classifier matched raw text, so the same sentence
# classified two different ways depending on which keyboard was in the hand.

_ARABIC_SPELLING = str.maketrans({"ک": "ك", "ی": "ي"})


@pytest.mark.parametrize("message", PLAY_ASKS)
def test_the_arabic_codepoint_spelling_classifies_identically(message):
    twin = message.translate(_ARABIC_SPELLING)
    assert classify_query_intent(twin).category == \
        classify_query_intent(message).category


@pytest.mark.parametrize("message", [
    "پلی‌لیست رو پخش کن",
    "یه آهنگ‌ خوب بذار",
])
def test_zwnj_and_its_absence_classify_identically(message):
    plain = message.replace("‌", " ")
    joined = message.replace("‌", "")
    assert classify_query_intent(plain).category == \
        classify_query_intent(message).category
    # …and the ZWNJ folds to a SPACE, not to nothing: `پلی‌لیست` and
    # `پلی لیست` are one word; `پلیلیست` is a spelling nobody types and
    # matching it is not the point — the point is that removing the joiner
    # must not silently change the verdict for the two forms that ARE typed.
    assert isinstance(classify_query_intent(joined).category, str)


# ── Negatives: the classifier must not have become a media classifier ────

@pytest.mark.parametrize("message,forbidden", [
    # An event lookup is not a song.
    ("برام چک کن که ایونتش کیه", "media"),
    # The Persian web route (test_query_intent_farsi_web) must be untouched.
    ("آخرین اخبار رو بگو", "media"),
    # Owned data still wins outright — it returns FULL before scoring is read.
    ("ایمیل‌هام رو چک کن", "media"),
])
def test_a_non_media_persian_ask_does_not_become_media(message, forbidden):
    assert classify_query_intent(message).category != forbidden


def test_the_persian_web_route_is_still_web():
    assert classify_query_intent("آخرین اخبار رو بگو").category == "web"


def test_owned_data_still_opens_the_full_surface():
    assert classify_query_intent("ایمیل‌هام رو چک کن").category == "full"


def test_the_owned_data_family_reads_the_folded_text_too():
    """The fold is not only about media. Every Persian family in this file
    matched RAW text, so `ايميل` (Arabic yeh) missed the owned-data family
    that `ایمیل` (Persian yeh) hits — the same sentence classified two
    different ways depending on which keyboard was in the hand.

    Pinned on a MIXED-script sentence on purpose: for an all-Persian ask the
    non-Latin backstop rescues the mistake and the fold is invisible, so a
    test written there would pass with the fold reverted (verified by
    mutation). Above 30% Latin the backstop does not fire, and the fold is
    the only thing standing between "check my email" and the tool-less
    question intent."""
    folded = classify_query_intent("check my ايميل ها please").category
    persian = classify_query_intent("check my ایمیل ها please").category
    assert folded == persian == "full"


@pytest.mark.parametrize("fragment", ["حال", "بذار", "برام"])
def test_a_one_token_transcript_fragment_creates_no_media_turn(fragment):
    """Recording 1's captions: the relay settles an utterance every 350 ms, so
    a sentence arrives as fragments. A single Persian word must not open the
    full tool surface on its own — the backstop's minimum token count is the
    only thing standing between "a script we do not know" and "every partial
    transcript is a full-tool turn"."""
    intent = classify_query_intent(fragment)
    assert intent.category == "question"
    assert "play_media" not in _exposed(fragment)


# ── The backstop ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("message", [
    "मुझे यह चाहिए",            # Devanagari, 3 words, no vocabulary here
    "هذا اختبار جديد",           # Arabic, 3 words
    "это очень важно",           # Cyrillic, 3 words
])
def test_a_zero_scoring_non_latin_ask_opens_the_full_surface(message):
    """A new-script user must never default into the tool-less question
    intent. `full` is merely WIDER — every tool, and the model chooses."""
    assert classify_query_intent(message).category == "full"


def test_a_zero_scoring_english_ask_is_still_a_question():
    """The backstop may not widen the common path: English scoring zero is
    genuinely "no tools needed", and that is the whole reason `question`
    exists."""
    assert classify_query_intent("random english words here").category == "question"


def test_a_mixed_script_ask_is_judged_by_its_majority():
    """A Persian sentence routinely carries an English proper noun. Demanding
    ZERO Latin would exempt exactly the asks a bilingual user types."""
    assert classify_query_intent("یه آهنگ از Coldplay بذار").category == "media"


def test_the_always_included_affordances_are_not_what_plays_music():
    """Guards the shape of the fix: `play_media` must arrive via the INTENT,
    not by being quietly added to the always-included set — that set is one of
    ten legal moves on iteration 0 of every short question, and putting a
    player in it is the ND-18 mistake."""
    assert "play_media" not in _ALWAYS_INCLUDED_TOOLS
