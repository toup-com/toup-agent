"""One verdict on "is this a request to PLAY something", for all three sites.

Three independent places have to answer that question and they used to answer
it three different ways:

  * ``ws_chat._fast_media_check`` — the typed-chat fast path, two anchored
    English regexes;
  * ``query_intent.classify_query_intent`` — the tool gate, an English keyword
    list plus an English pattern;
  * the GPT-Live relay's media fast path — which had no classifier at all.

Recording 1 (2026-09-20) is what that costs: ``یه آهنگ پخش کن`` scored **zero**
in every category of the intent classifier, fell to ``INTENT_QUESTION`` — whose
tool set is two web tools and recall — and the agent spent 89 s producing a
spoken promise and no music. Every natural Persian play phrasing of ≤8 words
did the same. In English the identical request worked, which is the shape the
owner reported as "Persian requests need four repetitions".

So the grammar lives here, once, and the three sites read it:

  * ``fa_media_score(text)`` — what a Persian play ask contributes to
    ``query_intent``'s ``media`` score. English scoring stays in
    ``query_intent`` untouched: its media family also covers image
    generation, ``send_photo`` and ``tts``, which are media INTENT but are
    emphatically not a request to start playback, and this module must never
    hand one of those to a fast path that would search YouTube for it.
  * ``media_request(text)`` — the extraction the two fast paths need:
    what to search for, whether the ask was open-ended, and which surface.
    Returns ``None`` when the ask is not a play request **or** when its
    subject is something only the agent can resolve ("the usual song"), so
    declining is always the safe answer: the turn goes to the agent, which
    has the context this module deliberately does not.
  * ``requested_mode(text)`` — 'video' | None, the audio-first question, whose
    English half stays in ``radio.player`` (its own contract, its own history)
    and whose Persian half is added here.

Nothing in this module reads settings, the DB or the network, and it holds no
state: it is a pure text predicate that both platform-api and the agent image
import.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "MediaAsk",
    "is_error_result",
    "normalize_fa",
    "fa_media_score",
    "media_request",
    "requested_mode",
    "requested_mode_fa",
]


# ── The failure contract the fast paths share ────────────────────────────
# The FUNCTION is the whole contract. An `ERROR_PREFIX = "ERROR:"` constant
# sat here beside it with no consumer and disagreed with it on 'ERROR - no
# track', 'ERRORS were found' and 'error: nope' — so whichever of the two a
# future caller reached for decided whether a failure read as a success. The
# loose test is the correct one: it has to agree with `tool_executor`'s
# `result.startswith("ERROR")` and `api_v1`'s `text.upper().startswith("ERROR")`,
# neither of which carries a colon.


def is_error_result(text: str) -> bool:
    """True when a play result string reports a FAILURE.

    A5-08/A3-9: the relay's direct play path prefixes failures with ERROR and
    the agent's tool returns prose, so the same no-video outcome renders as a
    green completed step down one path and a failed one down the other. One
    test, so a caller cannot accidentally believe a failure."""
    return (text or "").strip().upper().startswith("ERROR")


# ── Normalisation ────────────────────────────────────────────────────────
# A soft keyboard produces ARABIC kaf/yeh (ك U+0643, ي U+064A) where Persian
# wants ک U+06A9 / ی U+06CC, and every Persian regex in query_intent matched
# raw text — so `ایمیل` typed on an Arabic keyboard missed the owned-data
# family entirely and the same sentence classified two different ways
# depending on which keyboard the user happened to be holding.

_ZWNJ = "‌"

# ZWNJ becomes a SPACE, not nothing: `پلی‌لیست` and `پلی لیست` must read the
# same, and joining them into `پلیلیست` would match neither spelling.
_FOLD = {
    ord("ك"): "ک",   # ARABIC KAF          → ک
    ord("ي"): "ی",   # ARABIC YEH          → ی
    ord("ى"): "ی",   # ALEF MAKSURA        → ی
    ord("ة"): "ه",   # TEH MARBUTA         → ه
    ord("ە"): "ه",   # AE                  → ه
    ord(_ZWNJ): " ",
    ord("‍"): " ",        # ZWJ
    ord("‎"): " ",        # LRM
    ord("‏"): " ",        # RLM
    ord("ـ"): None,       # tatweel — pure decoration
}
# Harakat (fatha…sukun) and the superscript alef: optional vowel marks a user
# may or may not type, never part of a stem.
for _cp in list(range(0x064B, 0x0653)) + [0x0670, 0x0654, 0x0655]:
    _FOLD[_cp] = None

# Arabic-Indic ٠-٩ and Eastern Arabic (Persian) ۰-۹ digits → ASCII, so a
# number in a Persian sentence reads the same as one in an English sentence.
for _i in range(10):
    _FOLD[0x0660 + _i] = str(_i)
    _FOLD[0x06F0 + _i] = str(_i)

_WS_RE = re.compile(r"\s+")


def normalize_fa(s: str) -> str:
    """Codepoint-fold a message so one spelling of a Persian word is one
    string: NFKC, Arabic→Persian letter forms, digits to ASCII, ZWNJ to a
    space, harakat dropped, whitespace collapsed, lowercased.

    Idempotent, and a no-op on pure ASCII beyond `lower()` and whitespace
    collapsing — so it is safe to run an English pattern over the result."""
    if not s:
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", s).translate(_FOLD)).strip().lower()


# ── Persian play grammar ─────────────────────────────────────────────────
# Stems, not words. Persian verbs inflect for person and tense and attach
# clitics, so enumerating surface forms is how the English list got written
# and is why it covers nothing. Everything below is matched against
# `normalize_fa` output.

# A Persian letter — used as an explicit right boundary where a stem is a
# prefix of an unrelated common word. `\b` cannot help: every Persian letter
# is a word character, so `\b` sits at the SPACE either way.
_FA = "؀-ۿﭐ-﷿ﹰ-﻿"

# The thing being played. Suffixes are deliberately allowed (آهنگی، آهنگها،
# ویدیوی) — Persian pluralises and marks indefiniteness by suffix, and a
# right boundary here would reject the ordinary spoken forms.
_FA_MEDIA_NOUN_RE = re.compile(
    r"(?:[آا]هنگ|ترانه|موزیک|موسیقی|نماهنگ|کلیپ|ویدیو|ویدئو|فیلم|سریال"
    rf"|پادکست|[آا]لبوم|پلی\s*لیست|لیست\s*پخش|رادیو)"
)

# "put it on" — the verb half. `پلی` is the Latin word in Persian script and
# needs a right boundary or it eats پلیس (police); `گوش` needs one or it eats
# گوشی (phone). `play` is here too, for the mixed-script asks a bilingual user
# actually types ("play کن یه آهنگ").
_FA_PLAY_VERB_RE = re.compile(
    rf"(?:پخش|ب[ذز]ار|بگذار|بنداز|راه\s*بنداز|پلی(?![{_FA}])|بزن"
    rf"|گوش(?![{_FA}])|play)"
)

# Tokens strong enough to mean "media" with no verb at all.
_FA_STRONG_RE = re.compile(
    rf"(?:پلی(?![{_FA}])|موزیک|موسیقی|[آا]هنگ|ترانه|پادکست|نماهنگ)"
)

# "let me listen to something" — a consume verb. On its own it is ambiguous
# ("به حرفم گوش بده" is a request for attention, not for music), so it only
# scores when the ask is also explicitly open-ended.
_FA_CONSUME_RE = re.compile(
    rf"(?:گوش(?![{_FA}])\s*(?:بد|کن|می\s*کن|میکن)|بشنو|بشنویم)"
)

# "a"/"some"/"something" — the marker that turns a request into an open-ended
# one, which is exactly what `variety` means downstream.
_FA_OPEN_RE = re.compile(
    rf"(?:(?<![{_FA}])یه(?![{_FA}])|(?<![{_FA}])یک(?![{_FA}])|چیزی|هرچی|یکی"
    r"|\bsome\b|\bsomething\b|\banything\b|\bany\b)"
)

# The subject is a back-reference to something said earlier ("the usual one",
# "that one again"). The fast paths cannot resolve it and must not guess: a
# guess here plays a stranger's track, which is the 2026-07-31 incident class.
_FA_REFERENTIAL_RE = re.compile(
    # Deliberately NOT `همین` or `خودم`: "همین الان یه آهنگ بذار" is "play a
    # song RIGHT NOW" and "برای خودم یه آهنگ بذار" is "play me a song" —
    # ordinary asks with no back-reference at all. Over-declining costs the
    # whole latency win this predicate exists for.
    rf"(?:همیشگی|همون|همان|قبلی|دوباره|مجدد|باز\s*هم|(?<![{_FA}])[آا]ن(?![{_FA}])"
    rf"|(?<![{_FA}])اون(?![{_FA}])"
    # Possessives, and only the ones that name a SAVED thing. "پلی‌لیست منو
    # بذار" is the user's OWN list, which a YouTube search answers with a
    # stranger's — the 2026-07-31 class.
    rf"|(?<![{_FA}])منو(?![{_FA}])|مال\s*من|ذخیره|سیو)"
)

# ── Which surface? ───────────────────────────────────────────────────────
# The English half of this question lives in `radio.player.infer_requested_mode`
# and keeps its own contract — "watch" is a verb of intent only at the START of
# a request, mid-sentence it is a title ("Watch Me"). The Persian half mirrors
# it: the watch verbs are start-anchored, the compound nouns are not.
_FA_VIDEO_INTENT_RE = re.compile(
    r"^\s*(?:ببین|ببینیم|بیا\s+ببین|تماشا|نشون\s*(?:م|ش)?\s*بده|نشانم\s+بده)"
    r"|موزیک\s*(?:ویدیو|ویدئو)|ویدیو\s*کلیپ"
    r"|(?:رو[ی]?|سر)\s*(?:صفحه|تلویزیون|تی\s*وی|تلوزیون)"
)

# Content that is not music at all — audio-only is plainly wrong for a
# documentary, and the user should not have to say "video" to watch one.
# `ویدیو` is here rather than in the intent list because asking for "a video"
# IS asking to watch, which is the parity English already has via
# `play (the|a) video`.
_FA_NON_MUSIC_RE = re.compile(
    r"(?:مستند|تریلر|فیلم|سریال|قسمت|اپیزود|پادکست|مصاحبه|سخنرانی|کنسرت"
    r"|[آا]موزش|گیم\s*پلی|استند\s*[آا]پ|تد\s*تاک|خلاصه\s*بازی|ویدیو|ویدئو|کلیپ)"
)

# Someone else's catalogue. A "play X on Netflix" ask is not something a
# YouTube search can answer, and the agent is the one that should say so.
# Deliberately WIDER than the `_NETFLIX_KEYWORDS` it replaces in ws_chat:
# `apple tv`, a flexible `prime\s*video`, and the Persian services are new.
# The old list was English-only on a path this round is opening to Persian,
# and every addition can only make a fast path DECLINE — the safe direction.
_STREAMING_SERVICE_RE = re.compile(
    r"\b(?:netflix|disney|hulu|prime\s*video|hbo|apple\s*tv)\b"
    r"|(?:نتفلیکس|نت\s*فلیکس|فیلیمو|نماوا)",
    re.I,
)


def requested_mode_fa(text: str) -> Optional[str]:
    """'video' when a PERSIAN request explicitly asks to watch, or names
    content that has nothing to listen to. Never 'song' — absence of a video
    signal is not an audio request, and only the caller knows its channel's
    default surface (the same contract `radio.player.infer_requested_mode`
    documents)."""
    if not text:
        return None
    n = normalize_fa(text)
    if _FA_VIDEO_INTENT_RE.search(n) or _FA_NON_MUSIC_RE.search(n):
        return "video"
    return None


def requested_mode(text: str) -> Optional[str]:
    """Both halves of the audio-first question. Delegates to
    `radio.player.infer_requested_mode`, which is the site the tool and the
    typed-chat fast path already call — so there is one answer, not two."""
    if not text:
        return None
    # Lazy: `radio.player` reaches into ws_chat/media_proxy at call time, and
    # this module is imported by the intent classifier on the hot path.
    from app.agent.radio.player import infer_requested_mode
    return infer_requested_mode(text)


# ── Scoring, for the intent classifier ───────────────────────────────────

def fa_media_score(text: str) -> int:
    """What a Persian play ask adds to ``query_intent``'s ``media`` score.

    Weights mirror the English families so the priority-order ties break
    identically: a pattern is 3, a strong keyword is 2.

    Returns 0 for English text — the English media families in
    ``query_intent`` are untouched and already score it, and adding a second
    source would double-count and change which category wins a tie."""
    if not text:
        return 0
    n = normalize_fa(text)
    noun = bool(_FA_MEDIA_NOUN_RE.search(n))
    verb = bool(_FA_PLAY_VERB_RE.search(n))
    score = 0
    if noun and verb:
        score = 3
    if _FA_CONSUME_RE.search(n) and _FA_OPEN_RE.search(n):
        score = max(score, 3)
    if _FA_STRONG_RE.search(n):
        score = max(score, 2)
    return score


# ── Extraction, for the two fast paths ───────────────────────────────────

@dataclass(frozen=True)
class MediaAsk:
    """A play request a fast path can act on without an agent turn.

    ``query`` — what to search for.
    ``variety`` — the ask was open-ended ("a song", "some music"), so the
      resolver may randomise among good hits instead of pinning the top one.
    ``mode`` — 'video' when they asked to WATCH, else None (the caller
      applies its channel default; on a voice call that is always 'song').
    ``lang`` — 'fa' | 'en', for the log line only: `ws_chat._fast_media_check`
      prints it on `[FAST-MEDIA]`. It is an enum, never the user's words, and
      it is the only way to tell the Persian arm from the English one in Loki
      once both ship down the same path.
    """
    query: str
    variety: bool = False
    mode: Optional[str] = None
    lang: str = "en"


# The two English patterns, moved verbatim from `ws_chat._PLAY_PATTERNS` so
# the typed-chat fast path keeps behaving byte-identically while the relay
# gains the same grammar. Anchored, and tried in order.
_EN_PLAY_PATTERNS = (
    re.compile(
        r'^\s*play\s+(?:me\s+)?(?:a\s+)?(?:song\s+(?:of\s+|by\s+|called\s+)?'
        r'|video\s+(?:of\s+|by\s+|called\s+)?|music\s+(?:of\s+|by\s+)?)?'
        r'(.+?)(?:\s+(?:on|from|in)\s+(?:youtube|yt))?\s*$', re.I),
    re.compile(r'^\s*(?:put on|play me|play)\s+["“]?(.+?)["”]?\s*$', re.I),
)

# Everything a Persian play sentence is made of EXCEPT its subject. What is
# left after removing these is what the user actually wants to hear.
_FA_STOPWORDS = frozenset("""
یه یک لطفا لطفن برام برایم برای من ما رو را از با تو در به بهم که هم این
الان حالا دیگه دیگر یکم کمی چند تا و یا هست باشه بابا جان میشه می خوام
چیزی چیزایی هرچی هرچیزی یکی خوب قشنگ خودم خودت خودمون همین
ببین ببینم ببینیم ببینید بینیم تماشا نگاه نشون نشونم بده شده شد بشه
میخوام بخوام بزار بذار بگذار بگذارید بندازید بنداز راه پخش پلی گوش بده بدم
بدیم بدید کن کنم کنی کنید بکن بزن بزنم بزنید بشنوم بشنویم play put on some
a an the me my for us something anything any song music video track please
""".split())

# The media nouns, as bare tokens, so the residue does not keep them — and so
# a subject-less ask can be answered with the user's OWN word for what they
# want, which is what keeps a Persian "play a song" from searching an English
# catalogue.
_FA_NOUN_TOKENS = frozenset("""
آهنگ اهنگ آهنگی اهنگی آهنگا اهنگا آهنگها اهنگها ترانه موزیک موسیقی نماهنگ
کلیپ ویدیو ویدئو ویدیویی فیلم فیلمی سریال پادکست آلبوم البوم لیست رادیو
""".split())


# ── The media noun carries its suffix in every spelling (R6-10 TA5, R6-11 TB2) ─
# «آهنگ‌های ابی» folds (ZWNJ → space) to «آهنگ های ابی», and the split-off
# plural «های» was read as content: three spellings of one word («آهنگ‌های»,
# «آهنگ های», «آهنگهای») searched three different things. A media noun is a
# TOKEN SHAPE — the stem plus its plural (ها / colloquial ا), its
# indefinite/ezafe (ی / یی / ای), its possessive clitic (م ت ش مون تون شون مان
# تان شان) or the colloquial definite «ه» ("THE song"), and a colloquial object
# «و» — and each spelling of it is ONE token:
#
#   * a suffix joined to the noun by a ZWNJ («آهنگ‌های», «ترانه‌ای», «آهنگ‌هاش»)
#     is part of that one written word: the ZWNJ goes before the fold would
#     turn it into a space (`_join_zwnj_suffix`);
#   * a suffix written as its OWN token (a typed space) joins the noun only when
#     it is one of the closed free-standing plural forms (TB2): «ها» / «های» /
#     «هایی», optionally with a possessive clitic («آهنگ های ابی», «آهنگ هاش»).
#     Never a bare «ای» / «ی» / «ا» («آهنگ ای ایران» is the title "Ey Iran"; a
#     bare one standing alone right after the noun is ambiguous and declines —
#     `_fa_object`), and never a whole word of another shape («ایمان» stays the
#     title it is). A plural+clitic token that is also a whole word («فیلم
#     هامون»: the film Hamoun, or "our films") has two readings: joined, it is
#     the possessive, which points back (below) — ambiguous, so it declines;
#     it is never split into a query.
#
# A possessive or definite clitic points back at something only the
# conversation knows («آهنگ‌هاش», «آهنگامو», «آهنگه»): with no named subject
# the agent answers; the fragment is never a query.
_FA_NOUN_STEM = (
    r"(?:[آا]هنگ|ترانه|موزیک|موسیقی|نماهنگ|کلیپ|ویدیو|ویدئو|فیلم|سریال"
    r"|پادکست|[آا]لبوم|لیست|رادیو)"
)
_FA_NOUN_CLITIC = r"(?:مان|تان|شان|مون|تون|شون|م|ت|ش)"
_FA_NOUN_TOKEN_RE = re.compile(
    rf"^{_FA_NOUN_STEM}(?:ها|ا)?(?:ای|یی|ی)?(?P<ref>{_FA_NOUN_CLITIC}|ه)?(?:و)?$"
)
# TB2: the closed free-standing plural forms, with an optional possessive clitic.
_FA_BARE_SUFFIX_RE = re.compile(rf"^(?:ها|های|هایی){_FA_NOUN_CLITIC}?(?:و)?$")
# A suffix joined to its noun by a ZWNJ: one written word.
_FA_ZWNJ_SUFFIX_RE = re.compile(
    rf"(?<![{_FA}]){_FA_NOUN_STEM}{_ZWNJ}+(?:ها|ا)?(?:ای|یی|ی)?{_ZWNJ}*"
    rf"(?:{_FA_NOUN_CLITIC}|ه)?(?:و)?(?![{_FA}{_ZWNJ}])"
)
_FOLD_KEEP_ZWNJ = {k: v for k, v in _FOLD.items() if k != ord(_ZWNJ)}


def _is_fa_noun(token: str) -> bool:
    return token in _FA_NOUN_TOKENS or bool(_FA_NOUN_TOKEN_RE.match(token))


def _join_zwnj_suffix(raw: str) -> str:
    """`raw` with every ZWNJ between a media noun and its own suffix dropped
    («آهنگ‌های» → «آهنگهای»): the letter forms are folded first, so an Arabic
    keyboard's spelling joins the same way; every other ZWNJ is left for the
    fold («پلی‌لیست» → «پلی لیست»)."""
    text = unicodedata.normalize("NFKC", raw or "").translate(_FOLD_KEEP_ZWNJ)
    return _FA_ZWNJ_SUFFIX_RE.sub(lambda m: m.group(0).replace(_ZWNJ, ""), text)


def _join_noun_suffixes(folded: str) -> str:
    """`folded` with every closed free-standing plural token joined back onto
    the media noun before it («آهنگ های ابی» → «آهنگهای ابی»), so each
    spelling of the noun is one token. Only onto a media noun, and never across
    a clause mark (the noun's token carries none at its end)."""
    tokens = folded.split()
    out: list = []
    for tok in tokens:
        if out:
            prev = out[-1]
            core = tok.strip(_RESIDUE_EDGE_PUNCT)
            prev_core = prev.strip(_RESIDUE_EDGE_PUNCT)
            if (
                core and tok.startswith(core)
                and prev_core and prev.endswith(prev_core)
                and _FA_BARE_SUFFIX_RE.match(core)
                and _FA_NOUN_TOKEN_RE.match(prev_core)
                and _FA_NOUN_TOKEN_RE.match(prev_core + core)
            ):
                out[-1] = prev + tok
                continue
        out.append(tok)
    return " ".join(out)


def _fa_fold(raw: str) -> str:
    """The ONE normalized form both arms read: the codepoint fold, an
    Arabic-script mark inside a token as a boundary (residual 5), and the media
    noun's suffix joined back onto it (TA5, TB2)."""
    return _join_noun_suffixes(_split_inner_punct(normalize_fa(_join_zwnj_suffix(raw))))


def _refers_back(tokens: list) -> bool:
    """A media noun carrying a possessive or definite clitic («آهنگ‌هاش»)."""
    for tok in tokens:
        m = _FA_NOUN_TOKEN_RE.match(tok.strip(_RESIDUE_EDGE_PUNCT))
        if m and m.group("ref"):
            return True
    return False


# ── «بذار کنار» is "put aside", not "put on" (R6-10 TA6) ─────────────────
# The placing verbs this grammar reads as "play" («بذار», «بگذار», «بزن»,
# «بنداز») are other verbs with a directional particle as their complement:
# «کنار گذاشتن» / «بذار کنار» (put aside), «کنار زدن» (push aside), «زمین
# گذاشتن» (put down). The particle right beside the verb (either side, with or
# without the verb's clitic) makes the clause not a play request. (R6-11 TB1's
# grammar declines every such clause on its own; this stays as the named case.)
_FA_PLACE_VERB_RE = re.compile(
    r"^(?:ب[ذز]ار|بگذار|بنداز|بزن)(?:ی|ید|ین|یم|م|ن|ه)?(?:ش|شو|شون)?$"
)
_FA_PLACE_PARTICLE_RE = re.compile(r"^(?:کنار|زمین)(?:ش|شو|شون)?$")


def _puts_aside(tokens: list) -> bool:
    cores = [t.strip(_RESIDUE_EDGE_PUNCT) for t in tokens]
    for i, core in enumerate(cores):
        if not _FA_PLACE_PARTICLE_RE.match(core):
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(cores) and _FA_PLACE_VERB_RE.match(cores[j]):
                return True
    return False


# A Persian ask with no subject: fall back to the user's own word. Searching
# "music" for a Persian sentence returns an English catalogue, which is the
# wrong answer to a question asked in Persian.
_FA_GENERIC_QUERY = "آهنگ"

_FA_SCRIPT_RE = re.compile(f"[{_FA}]")


# ── The caller's own framing is not a subject (addendum 6 R6-6, media-x1) ─
# «آره یه پادکست پخش کن» ("yes, play a podcast") searched «آره»; «آره، …»
# searched «آره،»; «می‌تونی آهنگ هالو از بیانسه رو برام پخش کنی» ("can you
# play Halo by Beyonce for me") searched «تونی هالو بیانسه» — the ZWNJ fold
# splits «می‌تونی» and its «تونی» half survived. The residue kept the answer
# word, the modal framing and the punctuation, because this grammar did not
# know the words the relay's grammar already classifies as saying nothing.
#
# ONE class, two grammars: these are the relay's closed assent and
# content-free sets (`live_voice_protocol._ASSENT_TOKENS` + `_ASSENT_PHRASES`
# and `_DISCOURSE_FILLERS`), spelled once here — the module the relay already
# imports its normaliser from — and pinned equal to the relay's by
# `tests/test_media_order_scope_r6.py`. Nothing is added to them here.
ASSENT_WORDS_TEXT = "yes yeah yep yup sure okay ok alright right آره بله باشه اوکی حتما"
ASSENT_PHRASES_TEXT = (
    "all right", "sounds good", "perfect", "great", "that's fine", "fine", "go on",
    "go ahead", "عالیه", "خوبه", "بگو", "بفرمایید", "ادامه بده",
)
CONTENT_FREE_WORDS_TEXT = (
    "uh um umm uhm hm hmm mm mmm mhm mhmm ah ahh oh eh er erm huh ha "
    "ok okay alright right yeah yes yep yup sure well so hey hi hello wait "
    "look listen thanks thank you "
    "اه ام هوم اوم آه اِ خب خوب اوکی باشه آره آهان ببین میگم پس حالا "
    "مرسی ممنون الو سلام بله"
)
# The closed modal framing around a play ask, ONE entry per modal (and
# person) — R6-9 T2. Every spelling of an entry is the same entry: matching
# runs on one normalized token form, where the ZWNJ fold, a plain space and
# the joined verbal prefix «می»/«نمی» all read the same («میشه», «می‌شه»,
# «می شه» → «میشه»; `_indexed_words`). The polite/plural «می‌تونید/
# می‌تونین» is the same modal in its other person, and the written-register
# forms («می‌توانی», «می‌شود», «می‌خواهم») are those modals in full spelling.
MODAL_FRAMING_TEXT = (
    "can you", "could you", "would you", "i want to listen to",
    "می‌تونی", "می‌تونید", "می‌تونین", "می‌شه", "می‌خوام",
    "می‌توانی", "می‌توانید", "می‌شود", "می‌خواهم",
)

# Words for class comparison: the fold plus alef-madda → alef (the relay's
# NFKD fold drops the madda, so «آره» and «اره» are one word to it), split at
# every non-word character — Arabic comma/semicolon/question mark included.
_CLASS_SPLIT_RE = re.compile(r"[^\w؀-ۿ]+|[،؛؟٫٬۔]+")
# The verbal prefix, written apart from its verb (ZWNJ folds to a space).
_FA_VERB_PREFIXES = frozenset({"می", "نمی"})
# Punctuation that closes a clause inside a sentence (T3's boundary).
_CLAUSE_PUNCT = frozenset(",،;؛:.!?؟…۔")


def _token_words(token: str) -> list:
    return [w for w in _CLASS_SPLIT_RE.split(normalize_fa(token).replace("آ", "ا")) if w]


def _indexed_words(tokens: list) -> list:
    """[(word, token indices)] of whitespace `tokens`, on the ONE normalized
    form (T2): a bare verbal prefix («می», «نمی») that is a whole token is
    joined to the next token's first word, so «می‌شه», «می شه» and «میشه» are
    one word, «میشه»."""
    words: list = []
    pending = None  # (prefix, index) waiting for its verb
    for idx, tok in enumerate(tokens):
        parts = _token_words(tok)
        if pending is not None:
            prefix, pidx = pending
            pending = None
            if parts and _FA_SCRIPT_RE.match(parts[0]):
                words.append((prefix + parts[0], frozenset((pidx, idx))))
                parts = parts[1:]
            else:
                words.append((prefix, frozenset((pidx,))))
        if (
            len(parts) == 1 and parts[0] in _FA_VERB_PREFIXES
            and not any(ch in _CLAUSE_PUNCT for ch in tok)
            and idx + 1 < len(tokens)
        ):
            pending = (parts[0], idx)
            continue
        words.extend((w, frozenset((idx,))) for w in parts)
    if pending is not None:
        words.append((pending[0], frozenset((pending[1],))))
    return words


def _class_words(text: str) -> tuple:
    return tuple(w for w, _ in _indexed_words(normalize_fa(text).split()))


def _phrase_set(*phrases: str) -> frozenset:
    return frozenset(p for p in (_class_words(x) for x in phrases) if p)


_ASSENT_CLASS = _phrase_set(*ASSENT_WORDS_TEXT.split(), *ASSENT_PHRASES_TEXT)
_CONTENT_FREE_CLASS = _ASSENT_CLASS | _phrase_set(*CONTENT_FREE_WORDS_TEXT.split())
_MODAL_FRAMING = _phrase_set(*MODAL_FRAMING_TEXT)

# Punctuation a residue token loses at its edges — the Arabic-script marks
# (، ؛ ؟ ۔ « ») with the Latin ones.
_RESIDUE_EDGE_PUNCT = "\"'«»“”‘’.,،;؛:!?؟()[]-…۔٫٬"

# Residual 5: an Arabic-script mark INSIDE a token («آره،ابی») is a token
# boundary, like the space the typist left out. The mark stays on the word
# before it, where it closes that clause.
_FA_INNER_PUNCT_RE = re.compile(r"\s*([،؛؟۔])\s*")


def _split_inner_punct(folded: str) -> str:
    return _FA_INNER_PUNCT_RE.sub(r"\1 ", folded).strip()


# The play/light verb carrying its person ending or object clitic: «بذاری»
# ("(that) you put on") is what the modal framing governs, «بذارش» is "put it
# on". The stems are `_FA_PLAY_VERB_RE`'s and the light verb «کن»; a verb is
# never the subject.
_FA_VERB_FORM_RE = re.compile(
    r"^(?:ب[ذز]ار|بگذار|بنداز|بزن|بکن|کن|پخش)(?:ی|ید|ین|یم|م|ن|ه)?(?:ش|شو|شون)?$"
)
# The light verbs that also END a play clause («گوش بده», «پخش بشه»).
_FA_CLAUSE_END_VERB_RE = re.compile(r"^(?:بد|بش)(?:ه|م|ی|یم|ید|ین|ن)$")

# Negation is content, never a subject (R6-9 T5): a NEGATED verb form anywhere
# in the clause declines — the agent answers. Structural, not a phrase list:
# the particle «نه», «نباید» ("must not"), the negative progressive «نمی» on
# any verb, and the negative prefix «ن» on the play verbs and the light/modal
# verbs (present and past stems): کردن «نکن/نکرد», ذاشتن/گذاشتن «نذار/نگذار/
# نذاشت», زدن «نزن», انداختن «ننداز/نینداز», خواستن «نخوا», شدن «نشه/نشود/نشد»,
# دادن (the «گوش دادن» play verb) «نده/نداد». The short stems (ش, د) take only
# their person endings, so «نشون» ("show"), «ندا», «نشان» are never read as
# negations; «نزد»/«نزدیک» (a preposition, "near") are left out on purpose.
_FA_NEGATION_RE = re.compile(
    r"^(?:نه|نباید\w*|نمی\w*"
    r"|ن(?:کن|کرد|[ذز]ار|گذار|[ذز]اشت|گذاشت|زن|خوا)\w*"
    r"|نی?(?:ا)?نداز\w*|نی?(?:ا)?نداخت\w*"
    r"|نش(?:ه|ی|م|ن|یم|ید|ین)|نشد\w*|نشو(?:د|م|ی|یم|ید|ند)"
    r"|ند(?:ه|م|ی|ید|ین|یم|ن)|نداد\w*"
    r")$"
)



def _width(table: frozenset) -> int:
    return max((len(p) for p in table), default=0)


def _run_length(words: list, table: frozenset) -> int:
    """How many of `words`, from the start, form a run of `table` phrases."""
    width = _width(table)
    i = 0
    while i < len(words):
        for w in range(min(width, len(words) - i), 0, -1):
            if tuple(words[i:i + w]) in table:
                i += w
                break
        else:
            break
    return i


# ══ R6-11 TB1: the fast path is a WHITELIST with a safe default ══════════
# Every round of phrase handling on this grammar surfaced new edge cases: a
# title span that swallowed the caller's own «نباید», an unclosed quote that
# shielded «پخش نکن», a «بذار» that meant "let" («بذار آهنگ تموم بشه»), a
# retraction after a dash. The fast path is an ACCELERATOR for unambiguous
# requests; the agent path answers every request correctly, one round trip
# later. So the fast path now accepts a request only when the WHOLE sentence
# parses as the closed grammar
#
#     [assent / framing] + play verb + media noun and/or a NAMED title
#                                                  + [closed tail tags]
#
# and everything else declines to the agent. A false decline costs one
# agent round trip — and R6-12 requires that path to be PROVEN (the full words
# reach the agent, once, with no fast-path play; tests/test_media_r7_tenant.py)
# rather than assumed; a false accept plays a stranger's track. Before either
# arm reads the sentence, two things decline it outright, in both languages:
#
#   * R6-17: a word of the CLOSED function-word classes (`_trouble`, below)
#     anywhere outside a double / guillemet / curly-double quote — the
#     caller's own negation or retraction («نکن», «نباید», «نه», 'never mind',
#     «بی‌خیال», «ولش کن»), a relativizer («که», «ک», 'whose', 'where'), a
#     pronoun / possessive / demonstrative, an indefinite, an exclusion, a
#     temporal or sequence word, a repair. A title span never shields one
#     ('No Tears Left to Cry' takes the agent path and still plays); a SINGLE
#     quote never shields either;
#   * a quote left open (or closed without its opener): an unclosed quote
#     shields nothing, it declines.
#
# The grammar itself (Persian: `_fa_parse`; English: `_en_parse`) then rejects
# every second clause that is not a closed tail (a comma, dash, double hyphen,
# parenthesis — unless it holds a version tag, R6-17 — or sentence break
# followed by anything else) and a non-play complement of «بذار» / «بزار»
# (R6-17: a subjunctive-shaped word ANYWHERE after it — «تموم شه», «بره»,
# «بمونه», «بخونم» — and the particles «کنار» / «زمین»). The caller's framing
# words («برام», «لطفا», 'for me', 'please') end an UNMARKED Persian subject
# and leave the query as a separate tail; a MARKED title keeps them (R6-17
# NEVER TRUNCATE).
#
# New structures only: the relay's ONE class (ASSENT_* / CONTENT_FREE_* /
# MODAL_FRAMING_TEXT above) is read, never extended.
_POLITE = _phrase_set("لطفا", "لطفن", "please")
_THANKS = _phrase_set("مرسی", "ممنون", "ممنونم", "thanks", "thank you")
_BENEFACTIVE = _phrase_set(
    "برام", "برامون", "برایم", "برایمان", "واسم", "واسمون", "برای من", "برای ما",
    "برای خودم", "برای خودمون", "واسه من", "واسه ما", "واسه خودم", "for me", "for us",
)
_TIME = _phrase_set("الان", "همین الان", "حالا", "همین حالا", "now", "right now")
# «بعد یه آهنگ از ابی بذار» — "then, play …": sequences the request, names nothing.
# (R6-17: «بعدا» / «بعدن» mean "LATER", not this "then" — they are temporal
# words of the trouble scan, `_TROUBLE_WORDS`, and decline.)
_SEQUENCE = _phrase_set("بعد", "بعدش")
# «یه آهنگ بذار دیگه» — the emphatic particle after the verb.
_EMPHATIC = _phrase_set("دیگه", "دیگر")
_FA_DET = _phrase_set(
    "یه", "یک", "یکم", "کمی", "یه کم", "یه سری", "یه دونه", "چند", "چندتا", "چند تا",
)
# "something" — the head of an open «بذار یه چیزی گوش بدم».
_FA_OPEN_HEAD = frozenset({"چیزی", "چیز"})
_FA_PREP = _phrase_set("از", "درباره", "راجع به", "راجب", "در مورد", "در باره")
_TITLE_MARK = _phrase_set(
    "اسم", "اسمش", "به اسم", "با اسم", "به نام", "با نام", "called", "named", "titled",
)
_FA_OBJECT_MARKERS = frozenset({"رو", "را"})
# A bare suffix written as its own token right after the noun (TB2).
_FA_BARE_EZAFE = frozenset({"ای", "ی", "ا"})
# The caller's framing: a subject or a title span ends here, never includes it.
_FRAMING_STOP = _POLITE | _BENEFACTIVE | _MODAL_FRAMING
# What may stand between the object and its verb («آهنگ ابی رو می‌تونی
# بذاری؟», «یه پادکست، آره، پخش کن», «… از بیانسه رو برام پخش کنی»).
_PREVERB = _CONTENT_FREE_CLASS | _MODAL_FRAMING | _POLITE | _BENEFACTIVE | _TIME
# What may open the sentence.
_LEAD = _PREVERB | _SEQUENCE
# R6-17 whitelist collapse: the closed discourse tails that frame a request
# after its verb ("…, then", "let's see", "if it's no trouble", "dear") —
# «یه آهنگ از ابی پخش کن خب / ببینم / بی زحمت / عزیزم».
_FA_DISCOURSE_TAIL = _phrase_set("خب", "ببینم", "بی زحمت", "بیزحمت", "عزیزم")
# The closed tail tags after the verb.
_FA_TAIL = _ASSENT_CLASS | _POLITE | _THANKS | _BENEFACTIVE | _TIME | _EMPHATIC | _FA_DISCOURSE_TAIL
# English: a closed tail tag after a clause edge (T6); the framing a marked
# title span ends at; a head made only of class words (TA2: that IS the title).
_EN_TAG = _ASSENT_CLASS | _MODAL_FRAMING | _POLITE | _THANKS | _BENEFACTIVE
_EN_FRAMING = _POLITE | _BENEFACTIVE
_EN_HEAD_CLASS = _CONTENT_FREE_CLASS | _MODAL_FRAMING | _POLITE

# The verbs of the grammar, on a unit's comparison word (alef-madda folded).
# The light verb of «پخش کن» / «پلی کن» / «play کن»: imperative or 2nd person —
# and (R6-17 whitelist collapse) its present/progressive question form in the
# 2nd person, «پخش می‌کنی؟» / «پخش میکنین» ("will you play …?"; `_fa_units`
# has already joined the «می» written apart). Never the 1st person «میکنم».
_FA_LIGHT_RE = re.compile(r"^(?:ب?کن(?:ی|ید|ین|یم)?|میکن(?:ی|ید|ین))$")
# «بذار» — also the verb of 'let' (see `_lets`).
_FA_PLACE_RE = re.compile(r"^ب(?:ذ|ز|گذ)ار(?:ی|ید|ین|یم)?(?:ش|شو|شون)?$")
_FA_ZAN_RE = re.compile(r"^بزن(?:ی|ید|ین|یم)?(?:ش|شو)?$")
_FA_ANDAZ_RE = re.compile(r"^بی?نداز(?:ی|ید|ین|یم)?(?:ش|شو)?$")
# «گوش بدم / بدیم / بده / بدهم / کنیم» — the consume verb's light half.
_FA_CONSUME_TAIL_RE = re.compile(
    r"^(?:بد(?:م|یم|ه|ی|ید|ین)|بده(?:م|یم|ی|ید|ین)?|ب?کن(?:م|یم|ی|ید|ین)?)$"
)
_FA_HEAR_RE = re.compile(r"^بشنو(?:م|یم|ید|ین)?$")
# «یه فیلم بذار ببینیم» — "put on a film (so) we watch".
_FA_WATCH_RE = re.compile(r"^ببین(?:م|یم|ید|ین)$")
# The verb of a 'let' clause: a subjunctive — the «ب» prefix on a present stem
# with its person ending («بره», «بمونه», «بیاد», «بخونه») — or the light verb
# «شه» / «بشه» of a compound («تموم شه»). A shape, not a word list; a name
# that happens to have it («بیانسه») only costs a round trip.
_FA_SUBJUNCTIVE_RE = re.compile(r"^(?:ب\w+(?:ه|م|ی|یم|ید|ین|ن|د)|ب?ش(?:ه|م|ی|یم|ید|ین|ن))$")

# The caller's own "no" in the contracted English forms ("don't", "can't",
# "won't"), read on a unit's whole core — the word split takes the apostrophe
# apart. The bare negation words are members of `_TROUBLE_WORDS` below.
_EN_NEGATION_RE = re.compile(
    r"\b(?:no|not|nope|nah|never|nevermind"
    r"|(?:do|does|did|is|are|was|were|should|would|could|ca|wo|must|need|have|has|had)n['’]?t)\b",
    re.I,
)
# «بی‌خیال» ("never mind"), «ولش» ("let it go"), «کنسلش» ("cancel it").
_FA_RETRACTION_RE = re.compile(r"^(?:بیخیال\w*|ولش\w*|کنسل\w*)$")
# The negated copula («نیست», spoken «نیس», and its persons; «نبود…») — R6-17
# "with the existing negation shapes": exact person endings, so «نیسان» is a
# name, not a negation.
_FA_COPULA_NEG_RE = re.compile(
    r"^(?:نیس|نیست(?:م|ی|یم|ید|ین|ن|ند|ه)?|نبود(?:م|ی|یم|ید|ین|ن|ند|ه)?)$"
)


# ══ R6-17: ONE closed-class TROUBLE scan ══════════════════════════════════
# The Persian arm is a closed grammar, but the English object slot searched
# whatever followed 'play', and both arms admitted FUNCTION words that change
# who or what is meant: 'play it' (which?), 'play anything but Halo' (the
# excluded name was searched), 'play the song you played yesterday', «آهنگی ک
# اسمش یادم نیس», «بعدا یه آهنگ بذار» (played NOW). Growing phrase exceptions
# around each one is what kept surfacing new ones, so the fix is one scan over
# the WHOLE utterance, both arms, for the words of a handful of CLOSED
# function-word classes. Function-word classes are closed — English and
# Persian do not grow new pronouns or relativizers — so the scan is complete
# where a sentence list never is.
#
# A word of any class, anywhere outside a DOUBLE / guillemet / curly-double
# quote, declines to the agent. A single quote never shields (an elision
# apostrophe is a letter, and 'x' is too easily a pair of them); a title that
# holds such a word ('No Tears Left to Cry', 'Beat It', 'Thank U, Next')
# takes the agent path, which plays it one round trip later with the
# caller's full words (R6-12). The cost asymmetry is the reason: a false
# DECLINE costs one round trip, a false ACCEPT plays a stranger's track.
#
# Exempt are only the grammar's OWN framing slots, which hold a pronoun by
# construction: 'play me …', 'for me' / «برام» / «برای خودم», the modal
# framing ('can you', «می‌تونی»), 'thank you', the assent class ("that's
# fine"), «همین الان» (RIGHT now) and the determiner «چند تا».
_TROUBLE_CLASSES = {
    # pronouns / possessives / demonstratives — the object or its determiner,
    # and a first/second-person subject clause ('I …', 'you …') or a reduced
    # relative ('the song he sang').
    "pronoun": """
        i me my mine myself you your yours yourself yourselves he him his himself
        she her hers herself it its itself we us our ours ourselves they them
        their theirs themselves this that these those
        این اینو اینا اینها اینی اون اونو اونا اونی ان انها او ایشون ایشان وی
        همون همونو همونی همونا همان همین همینو همینی
        خودم خودت خودش خودمون خودتون خودشون خودمان خودتان خودشان
        برات براتون واست واستون بهت بهتون
        م ت ش مون تون شون مان تان شان
    """,
    # indefinite / universal pronouns and determiners
    "indefinite": """
        any anything anyone anybody anywhere something someone somebody somewhere
        nothing nobody none nowhere everything everyone everybody everywhere
        every each all whatever whoever whichever wherever whenever
        هر هرچی هرچیزی هرکدوم هرکی هیچ هیچی هیچکدوم همه همش همشو
    """,
    # exclusion / negation
    "exclusion": """
        but except excepting excluding without besides instead unless
        no not nope nah never nevermind nor neither cannot
        غیر جز بجز الا بدون سوای مگه مگر بجای نخیر نچ
    """,
    # relativizers and the wh-words
    "relative": """
        where who whom whose which what why how
        که ک کی کیه چی چیه چه کجا کدوم کدام کدومش چرا چطور چطوری چجوری
    """,
    # temporal / sequence (bare «بعد» / «بعدش» "then" stays a LEAD — `_SEQUENCE`)
    "temporal": """
        again later next after afterwards afterward before earlier when while
        until till til then tomorrow yesterday tonight today soon last previous
        previously
        بعدا بعدن دیروز فردا امشب امروز دیشب پریشب پریروز پسفردا قبلا قبلن وقتی
        موقعی هروقت تا
    """,
    # repair / retraction
    "repair": """
        nvm jk sorry oops whoops actually wait kidding joking forget cancel
        scratch disregard correction
        منظورم منظور منظورمه یعنی ببخشید ببخشین ببخش شوخی اشتباه اشتباهی
    """,
}
# The members that are phrases of words none of which is a member alone.
_TROUBLE_PHRASES = _phrase_set(
    "other than", "rather than", "apart from", "aside from", "never mind",
    "on second thought", "second thought", "second thoughts", "j k", "به جای",
)
_TROUBLE_WORDS = frozenset(
    w for words in _TROUBLE_CLASSES.values() for w in _class_words(" ".join(words.split()))
)
# «من / تو / ما / شما» are trouble as the object's EZAFE («آهنگ من», «پلی
# لیست تو») or a subject — never as a preposition's object («بعد از تو» is a
# title, «برای من» framing).
_FA_PERSONAL = frozenset({"من", "تو", "ما", "شما"})
_FA_PREPOSITION = frozenset({
    "از", "به", "با", "برای", "برا", "واسه", "بی", "درباره", "مثل", "پیش",
})
# The grammar's own framing slots: a pronoun there is part of the grammar.
_TROUBLE_EXEMPT = (
    _ASSENT_CLASS | _MODAL_FRAMING | _POLITE | _THANKS | _BENEFACTIVE | _TIME | _FA_DET
    | _phrase_set("play me")
)


def _flat_phrase(words: list, i: int, table: frozenset) -> int:
    """How many of `words` from `i` spell one phrase of `table` (longest
    first; a shielded slot — None — is never part of one), or 0."""
    for w in range(min(_width(table), len(words) - i), 0, -1):
        seq = words[i:i + w]
        if None not in seq and tuple(seq) in table:
            return w
    return 0


def _trouble(units: list) -> bool:
    """R6-17: ONE scan of the whole utterance, both arms — a word of the
    closed classes (`_TROUBLE_CLASSES`), the negation shapes (R6-9 T5: a
    negated verb or copula anywhere), a retraction («بی‌خیال», «ولش کن»),
    anywhere outside a double / guillemet / curly-double quote, outside the
    grammar's own framing slots (`_TROUBLE_EXEMPT`)."""

    words: list = []  # the comparison words in order; None = shielded
    for u in units:
        if u.shielded:
            words.append(None)
            continue
        if _EN_NEGATION_RE.search(u.core):
            return True
        words.extend(u.words)
    i = 0
    while i < len(words):
        w = words[i]
        if w is None:
            i += 1
            continue
        exempt = _flat_phrase(words, i, _TROUBLE_EXEMPT)
        if exempt:
            i += exempt
            continue
        if _flat_phrase(words, i, _TROUBLE_PHRASES) or w in _TROUBLE_WORDS:
            return True
        if (
            _FA_NEGATION_RE.match(w) or _FA_COPULA_NEG_RE.match(w)
            or _FA_RETRACTION_RE.match(w)
        ):
            return True
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is not None and (
            (w == "بی" and nxt.startswith("خیال")) or (w == "ول" and _FA_LIGHT_RE.match(nxt))
        ):
            return True
        if w in _FA_PERSONAL and not (i > 0 and words[i - 1] in _FA_PREPOSITION):
            return True
        i += 1
    return False


# ── Quotes (TB1: only a CLOSED quote counts; TB3/R6-17: single quotes) ──
_QUOTE_OPENERS = {"«": "»", "“": "”", "‘": "’"}
_QUOTE_OPENER_OF = {v: k for k, v in _QUOTE_OPENERS.items()}
_QUOTE_GLYPHS = frozenset("«»“”‘’\"'")
# R6-17: the quotes that SHIELD their words from the trouble scan — double,
# guillemet and curly-double. A single-quoted span keeps its words whole (a
# comma inside it is no clause edge, a tag inside it is not stripped), but a
# trouble word inside it still declines.
_SHIELD_OPENERS = frozenset("«\"“")
# R6-17: an ELISION apostrophe is a letter, never a quote mark — the closed
# class of clipped words it opens ('Til, 'em, 'n', 'cause, 'bout, 'round,
# 'tis, 'twas, 'nuff, 'kay) and a clipped number ('80s, ‘90s). A word-final
# apostrophe that closes no open single quote (Guns N', the Beatles',
# Stayin') is a letter too.
_ELIDED_HEADS = frozenset({
    "til", "till", "em", "n", "cause", "cos", "coz", "cuz", "bout", "round",
    "tis", "twas", "nuff", "kay", "sup",
})
_WORD_AHEAD_RE = re.compile(r"[^\W_]+")


def _elided(text: str, i: int) -> bool:
    """Whether the apostrophe at `i` opens a clipped word (an elision)."""
    m = _WORD_AHEAD_RE.match(text, i + 1)
    word = m.group(0).lower() if m else ""
    return bool(word) and (word[0].isdigit() or word in _ELIDED_HEADS)


def _quote_spans(text: str) -> Optional[list]:
    """[(start, end)] of every CLOSED quoted span of `text` (its quote marks
    included), or None when a quote is left open or closed without its opener.

    « », “ ”, " and — TB3 — the single quotes ' ‘ ’ are quotes. A ' or ’
    between two letters («don't», «it’s»), an elision (R6-17: '80s, 'Til,
    rock 'n' roll), or a word-final one with no single quote open ('the
    Beatles' song', 'Guns N' Roses') is an apostrophe."""

    spans: list = []
    stack: list = []  # [(opener, index)]
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in _QUOTE_GLYPHS:
            continue
        prev = text[i - 1] if i > 0 else " "
        nxt = text[i + 1] if i + 1 < n else " "
        top = stack[-1][0] if stack else ""
        if ch in ("'", "‘") and not prev.isalnum() and nxt.isalnum() and _elided(text, i):
            continue  # R6-17: a leading elision apostrophe is a letter
        if ch in _QUOTE_OPENERS:
            stack.append((ch, i))
        elif ch == '"':
            if top == '"':
                spans.append((stack.pop()[1], i + 1))
            else:
                stack.append((ch, i))
        elif ch in ("'", "’"):
            if prev.isalnum() and nxt.isalnum():
                continue  # an apostrophe inside a word
            if top in ("'", "‘") and not nxt.isalnum():
                spans.append((stack.pop()[1], i + 1))
            elif ch == "'" and not prev.isalnum() and nxt.isalnum():
                stack.append((ch, i))
            # otherwise an apostrophe at a word's edge
        elif top == _QUOTE_OPENER_OF[ch]:
            spans.append((stack.pop()[1], i + 1))
        else:
            return None  # » or ” with no opener
    return None if stack else spans


def _shield_spans(text: str, spans: list) -> list:
    """The spans of `spans` that SHIELD (R6-17): double / guillemet /
    curly-double quotes only."""
    return [(a, b) for a, b in spans if text[a] in _SHIELD_OPENERS]


def _inside(spans: list, pos: int) -> bool:
    return any(a <= pos < b for a, b in spans)


def _masked(text: str, spans: list) -> str:
    """`text` with every closed quoted span blanked out: what the caller said
    in their own voice, never a title they named."""
    if not spans:
        return text
    chars = list(text)
    for a, b in spans:
        for k in range(a, b):
            chars[k] = " "
    return "".join(chars)


# ── Persian units: one token, its comparison word, its clause breaks ─────
# A clause break: the Latin and Arabic-script clause marks, dashes and
# brackets (a quote is never a break).
_BREAK_CHARS = frozenset(",،;؛:.!?؟…۔—–()[]")
_UNIT_EDGE = _RESIDUE_EDGE_PUNCT + "—–"


class _Unit:
    """One whitespace token of the folded sentence: its `core` (the token
    without edge punctuation, as folded), its comparison `words` (alef-madda
    folded, lowercased), whether it sits inside a CLOSED quote (`quoted`: any
    quote, a title — the grammar never cuts it) and inside a SHIELDING one
    (`shielded`, R6-17: double / guillemet / curly-double — the trouble scan
    never reads it), and whether a clause break stands right before/after it
    (outside quotes)."""

    __slots__ = ("core", "words", "word", "quoted", "shielded", "brk_before", "brk_after")

    def __init__(self, core: str, words: tuple, quoted: bool,
                 brk_before: bool, brk_after: bool, shielded: bool = False) -> None:
        self.core = core
        self.words = words
        self.word = words[0] if len(words) == 1 else ""
        self.quoted = quoted
        self.shielded = shielded
        self.brk_before = brk_before
        self.brk_after = brk_after


def _fa_units(folded: str) -> Optional[list]:
    """The units of `folded`, or None when a quote is left open. A verbal
    prefix «می» / «نمی» written apart from its verb is one unit with it
    (R6-9 T2's one normalized form: «می شه» → «میشه»)."""

    spans = _quote_spans(folded)
    if spans is None:
        return None
    shield = _shield_spans(folded, spans)
    units: list = []
    carry = False
    for m in re.finditer(r"\S+", folded):
        tok, start = m.group(0), m.start()
        left = len(tok) - len(tok.lstrip(_UNIT_EDGE))
        outside = [ch for k, ch in enumerate(tok) if not _inside(spans, start + k)]
        if left >= len(tok):
            # Punctuation alone («—», «-», «(»): a break when it is one.
            if any(ch in _BREAK_CHARS or ch == "-" for ch in outside):
                if units:
                    units[-1].brk_after = True
                carry = True
            continue
        right = len(tok) - len(tok.rstrip(_UNIT_EDGE))
        core = tok[left:len(tok) - right]
        lead = [ch for k, ch in enumerate(tok[:left]) if not _inside(spans, start + k)]
        tail_at = start + len(tok) - right
        trail = [ch for k, ch in enumerate(tok[len(tok) - right:]) if not _inside(spans, tail_at + k)]
        units.append(_Unit(
            core=core,
            words=tuple(_token_words(core)),
            quoted=_inside(spans, start + left),
            brk_before=carry or any(ch in _BREAK_CHARS for ch in lead),
            brk_after=any(ch in _BREAK_CHARS for ch in trail),
            shielded=_inside(shield, start + left),
        ))
        carry = False
    merged: list = []
    for u in units:
        p = merged[-1] if merged else None
        if (
            p is not None and p.word in _FA_VERB_PREFIXES and not p.quoted and not u.quoted
            and not p.brk_after and not u.brk_before and u.words
            and _FA_SCRIPT_RE.match(u.words[0])
        ):
            merged[-1] = _Unit(
                core=f"{p.core} {u.core}", words=(p.word + u.words[0],) + u.words[1:],
                quoted=False, brk_before=p.brk_before, brk_after=u.brk_after,
                shielded=False,
            )
            continue
        merged.append(u)
    return merged


def _brk(units: list, a: int, b: int) -> bool:
    """A clause break between unit `a` and the next unit `b`."""
    return units[a].brk_after or units[b].brk_before


def _phrase_at(units: list, i: int, table: frozenset) -> int:
    """How many units from `i` spell one phrase of `table` (longest first; no
    quoted unit, no break inside the phrase), or 0."""
    for w in range(min(_width(table), len(units) - i), 0, -1):
        seq = units[i:i + w]
        if any(u.quoted or not u.word for u in seq):
            continue
        if any(seq[k].brk_after or seq[k + 1].brk_before for k in range(w - 1)):
            continue
        if tuple(u.word for u in seq) in table:
            return w
    return 0


# ── The Persian grammar (TB1) ──────────────────────────────────────────────

class _FaParse:
    """A complete parse: the object's words with their roles, in order —
    'noun' (the media noun), 'open' («چیزی»), 'prep' («از»), 'arg' (what a
    preposition governs: the artist, the topic), 'marker' («به اسم»),
    'content' (right on the noun: an adjective, an unmarked title) and
    'title' (a quoted span, or the span a title marker introduces)."""

    def __init__(self, seq_lead: bool) -> None:
        self.seq_lead = seq_lead
        self.items: list = []
        self.marked = False
        self.ambiguous = False


def _lets(units: list, start: int, exempt: set) -> bool:
    """«بذار» is also 'let' («بذار آهنگ بره», «بذار آهنگ تموم شه», «بذار یه
    آهنگ از ابی برات بخونم»: let the song go on / finish, let me sing …) — a
    clause whose verb is a SUBJUNCTIVE: the «ب» prefix on a stem with its
    person ending, or the light verb «شه». R6-17: the let-rule covers the
    WHOLE complement — a subjunctive-shaped word anywhere after «بذار»
    (inside an «از» / «درباره» argument or a «به اسم» span too) and the
    sentence reads both ways: the agent answers. Exempt are only the
    grammar's own words at `exempt` (the purpose complement «گوش بدم» /
    «ببینیم», the closed tail tags) and a shielded title («بذار آهنگ
    «بمونه»»). A name that has the shape («بیانسه») costs one round trip."""
    return any(
        k not in exempt and not units[k].shielded
        and any(_FA_SUBJUNCTIVE_RE.match(w) for w in units[k].words)
        for k in range(start, len(units))
    )


def _fa_noun_at(units: list, j: int) -> bool:
    u = units[j]
    if u.quoted:
        return False
    if _is_fa_noun(u.core):
        return True
    # «پلی لیست» (the ZWNJ fold of «پلی‌لیست»).
    return (
        u.word == "پلی" and j + 1 < len(units) and not _brk(units, j, j + 1)
        and not units[j + 1].quoted and units[j + 1].core.startswith("لیست")
        and _is_fa_noun(units[j + 1].core)
    )


def _fa_verb_at(units: list, j: int) -> Optional[tuple]:
    """(end, kind) when a play verb starts at unit `j`, else None."""
    n = len(units)
    if j >= n or units[j].quoted:
        return None
    w = units[j].word
    nxt = units[j + 1] if j + 1 < n and not units[j + 1].quoted and not _brk(units, j, j + 1) else None
    nw = nxt.word if nxt is not None else ""
    if w in ("پخش", "پخشش", "پلی", "play") and nw and _FA_LIGHT_RE.match(nw):
        return j + 2, "light"
    if w == "play":
        return j + 1, "light"
    if w == "راه" and nw and _FA_ANDAZ_RE.match(nw):
        return j + 2, "put"
    if w == "گوش" and nw and _FA_CONSUME_TAIL_RE.match(nw):
        return j + 2, "consume"
    if _FA_HEAR_RE.match(w):
        return j + 1, "consume"
    if _FA_PLACE_RE.match(w):
        return j + 1, "place"
    if _FA_ZAN_RE.match(w) or _FA_ANDAZ_RE.match(w):
        return j + 1, "put"
    return None


def _fa_verb_like(u: _Unit) -> bool:
    """A verb form this grammar knows, in any position — never a subject."""
    if u.quoted:
        return False
    return any(
        w in ("پخش", "پخشش") or _FA_LIGHT_RE.match(w) or _FA_PLACE_RE.match(w)
        or _FA_ZAN_RE.match(w) or _FA_ANDAZ_RE.match(w) or _FA_CONSUME_TAIL_RE.match(w)
        or _FA_HEAR_RE.match(w) or _FA_WATCH_RE.match(w)
        or _FA_CLAUSE_END_VERB_RE.match(w) or _FA_VERB_FORM_RE.match(w)
        for w in u.words
    )


def _fa_ends_content(units: list, j: int, role: str) -> bool:
    """Whether unit `j` ends a content run. A subject ends at the object
    marker, a preposition or title marker, the caller's framing and any verb;
    a MARKED title span (after «به اسم» / «اسمش») only at the object marker
    and a verb — «به اسم بعد از تو» is one title. R6-17 NEVER TRUNCATE: the
    framing words («برام», «لطفا», «الان», the modal framing) inside a marked
    title stay in it (the caller's marker says the words are the title); the
    framing that leaves the query is the separate tail after the verb or a
    clause break (`_FA_TAIL`, and a content run never crosses a break)."""
    u = units[j]
    if u.word in _FA_OBJECT_MARKERS:
        return True
    if role != "title" and _phrase_at(units, j, _FRAMING_STOP):
        return True
    if _fa_verb_at(units, j) is not None or _fa_verb_like(u):
        return True
    return role != "title" and bool(
        _phrase_at(units, j, _FA_PREP) or _phrase_at(units, j, _TITLE_MARK)
    )


def _fa_content(units: list, j: int, p: _FaParse, role: str) -> int:
    """Consume one content run from `j` (never across a clause break); a
    quoted unit inside it is a title. Returns where it ended."""
    start = j
    while j < len(units):
        u = units[j]
        if j > start and _brk(units, j - 1, j):
            break
        if u.quoted:
            p.items.append((u, "title"))
        elif _fa_ends_content(units, j, role):
            break
        else:
            p.items.append((u, role))
        j += 1
    return j


def _fa_modifier(units: list, j: int, p: _FaParse) -> Optional[int]:
    """A preposition phrase («از ابی») or a title-marker phrase («به اسم
    باشه») at `j`: its end, j itself when there is none, None when the marker
    has nothing after it."""
    for table, marker_role, role in ((_FA_PREP, "prep", "arg"), (_TITLE_MARK, "marker", "title")):
        w = _phrase_at(units, j, table)
        if not w:
            continue
        p.items.extend((u, marker_role) for u in units[j:j + w])
        end = _fa_content(units, j + w, p, role)
        return end if end > j + w else None
    return j


def _fa_object(units: list, j: int, p: _FaParse) -> Optional[int]:
    """[DET] media-noun (or «چیزی») [modifiers] [«رو»] at `j`: its end, or
    None. A clause break ends the object: a content run never resumes after
    one («آهنگ ابی، داریوش رو بذار» is a correction, not one title)."""
    n = len(units)
    j += _phrase_at(units, j, _FA_DET)
    head = j
    while j < n and _fa_noun_at(units, j) and (j == head or not _brk(units, j - 1, j)):
        p.items.append((units[j], "noun"))
        j += 1
    if j == head:
        if j < n and not units[j].quoted and units[j].word in _FA_OPEN_HEAD:
            p.items.append((units[j], "open"))
            j += 1
        else:
            return None
    first = True
    while j < n and not _brk(units, j - 1, j):
        u = units[j]
        if not u.quoted and u.word in _FA_OBJECT_MARKERS:
            p.marked = True
            return j + 1
        end = _fa_modifier(units, j, p)
        if end is None:
            return None
        if end == j:
            end = _fa_content(units, j, p, "content")
            if end == j:
                return j
            # TB2: a bare «ای» / «ی» / «ا» that is the WHOLE run right after
            # the noun — the indefinite suffix typed apart («ترانه ای از ابی»),
            # or a one-word title? Ambiguous: the agent answers.
            if first and end == j + 1 and not u.quoted and u.word in _FA_BARE_EZAFE:
                p.ambiguous = True
        first = False
        j = end
    return j


def _fa_complement(units: list, j: int) -> int:
    """After «بذار»: its purpose clause «گوش بدم» / «ببینیم» ("put something on
    (so) I listen / we watch"), when present."""
    if j >= len(units) or j == 0 or _brk(units, j - 1, j):
        return j
    v = _fa_verb_at(units, j)
    if v is not None and v[1] == "consume":
        return v[0]
    if not units[j].quoted and _FA_WATCH_RE.match(units[j].word):
        return j + 1
    return j


def _fa_skip(units: list, j: int, table: frozenset) -> int:
    while j < len(units):
        w = _phrase_at(units, j, table)
        if not w:
            break
        j += w
    return j


def _fa_parse(units: list) -> Optional[_FaParse]:
    """The complete parse of the sentence, or None (decline):

        LEAD*  OBJECT  PREVERB*  VERB [complement]  POSTMOD*  TAIL*
        LEAD*  VERB  OBJECT [complement]  POSTMOD*  TAIL*

    POSTMOD is a «از …» / «به اسم …» phrase after the verb («یه آهنگ از ادل
    پخش کن به اسم hello»). Nothing else may stand anywhere."""

    n = len(units)
    lead, seq_lead = 0, False
    while lead < n:
        w = _phrase_at(units, lead, _LEAD)
        if not w:
            break
        if tuple(u.word for u in units[lead:lead + w]) in _SEQUENCE:
            seq_lead = True
        lead += w
    for verb_first in (False, True):
        p = _FaParse(seq_lead)
        if verb_first:
            v = _fa_verb_at(units, lead)
            if v is None:
                continue
            verb_end, kind = v
            j = _fa_skip(units, verb_end, _BENEFACTIVE | _POLITE)
            j = _fa_object(units, j, p)
            if j is None:
                continue
        else:
            j = _fa_object(units, lead, p)
            if j is None:
                continue
            v = _fa_verb_at(units, _fa_skip(units, j, _PREVERB))
            if v is None:
                continue
            verb_end, kind = v
            j = verb_end
        exempt: set = set()
        if kind == "place":
            c = _fa_complement(units, j)
            exempt.update(range(j, c))
            j = c
        complete = True
        while j < n:
            end = _fa_modifier(units, j, p)
            if end is None:
                complete = False
            if end is None or end == j:
                break
            j = end
        if not complete:
            continue
        tail = j
        j = _fa_skip(units, j, _FA_TAIL)
        exempt.update(range(tail, j))
        if j != n or p.ambiguous:
            continue
        if kind == "place" and _lets(units, verb_end, exempt):
            continue
        return p
    return None


def _fa_residue(p: _FaParse) -> list:
    """[(text, titled)] the parse names: every title word as it is, and the
    other object words that are content (no stopword, noun, back-reference or
    verb form)."""
    out: list = []
    for u, role in p.items:
        if role == "title":
            out.append((u.core, True))
        elif role in ("content", "arg", "prep", "marker") and (
            u.core not in _FA_STOPWORDS and not _is_fa_noun(u.core)
            and not _FA_REFERENTIAL_RE.fullmatch(u.core) and not _FA_VERB_FORM_RE.match(u.core)
        ):
            out.append((u.core, False))
    return out


# ── The English grammar (TB1) ─────────────────────────────────────────────
# The anchored verb, then ONE request clause. What follows a clause edge
# outside quotes must be a closed tail tag (', would you?', ', please',
# '. Thanks!'), or — after a comma or a spaced hyphen only — the next part
# of a title written as a title ('Thank U, Next', 'Well, Well, Well',
# 'Beyonce - Halo'). A dash, double hyphen, parenthesis, colon, semicolon or
# sentence break followed by anything else is a second clause: decline.
_EN_VERB_RE = re.compile(r"^\s*(?:play|put\s+on)\s+", re.I)
_EN_SOFT = frozenset(",،")
_EN_HARD = frozenset(";؛:—–()[]")
_EN_STOP = frozenset(".!?…؟۔")
_EN_MARKER_RE = re.compile(rf"\b(?:called|named|titled)\b|(?<![{_FA}])(?:اسم|اسمش)(?![{_FA}])", re.I)
# A second verb of the grammar's own class inside the object — another play,
# or a transport command ('play halo stop', 'play Halo, Stop'): two requests,
# not a title. A title that holds one ('Stop and Stare') takes the agent path.
_EN_SECOND_VERB_RE = re.compile(r"\b(?:play|stop|pause|skip|resume)\b|\bput\s+on\b", re.I)


# R6-17 whitelist collapse: a parenthesised / bracketed VERSION TAG is part
# of the title ('Halo (Live)', 'Halo [Acoustic]', 'Yesterday (2009 Remaster)'),
# not a second clause: a closed class of tag words, optionally with a year.
_EN_VERSION_TAGS = frozenset("""
live acoustic remix remixed remaster remastered official music video audio lyric
lyrics visualizer version edit radio extended mix instrumental unplugged demo
cover explicit clean original deluxe mono stereo karaoke slowed reverb sped up
hd hq session sessions performance studio single edition anniversary bonus
""".split())
_EN_BRACKETS = {"(": ")", "[": "]"}


def _en_version_tag_end(raw: str, i: int, end: int) -> int:
    """The index after the bracket group opened at `i` when it is a version
    tag, else -1."""
    close = raw.find(_EN_BRACKETS[raw[i]], i + 1, end)
    if close < 0:
        return -1
    words = [w for w in re.split(r"[^A-Za-z0-9]+", raw[i + 1:close].lower()) if w]
    if words and all(w in _EN_VERSION_TAGS or w.isdigit() for w in words) and any(
        not w.isdigit() for w in words
    ):
        return close + 1
    return -1


def _en_segments(raw: str, start: int, spans: list) -> list:
    """[(kind, sep, start, end, after_hard)] of the object text from `start`,
    split at clause edges outside quotes; `kind` is '' for the head, else
    'soft' (comma, spaced hyphen) or 'hard'; terminal punctuation is left out
    and empty segments dropped. A version-tag bracket group is title text."""
    end = len(raw)
    while end > start and not _inside(spans, end - 1) and (raw[end - 1].isspace() or raw[end - 1] in _EN_STOP):
        end -= 1
    found: list = []
    kind, sep, seg_start, hard = "", start, start, False
    i = start
    while i < end:
        ch = raw[i]
        k, width = "", 1
        if _inside(spans, i):
            pass
        elif ch in _EN_BRACKETS and _en_version_tag_end(raw, i, end) > 0:
            i = _en_version_tag_end(raw, i, end)
            continue
        elif ch in _EN_SOFT:
            k = "soft"
        elif ch == "-" and raw.startswith("--", i):
            k, width = "hard", len(raw[i:end]) - len(raw[i:end].lstrip("-"))
        elif ch == "-" and raw[i - 1:i].isspace() and raw[i + 1:i + 2].isspace():
            k = "soft"
        elif ch in _EN_HARD:
            k = "hard"
        elif ch in _EN_STOP:
            j = i
            while j < end and raw[j] in _EN_STOP:
                j += 1
            if j < end and raw[j].isspace():
                k, width = "hard", j - i
        if not k:
            i += 1
            continue
        found.append((kind, sep, seg_start, i, hard))
        hard = hard or k == "hard"
        kind, sep = k, i
        i += width
        seg_start = i
    found.append((kind, sep, seg_start, end, hard))
    return [s for s in found if raw[s[2]:s[3]].strip()]


def _en_words_in(text: str, table: frozenset) -> bool:
    words = list(_class_words(text))
    return bool(words) and _run_length(words, table) == len(words)


def _en_title_part(raw: str, start: int, end: int, spans: list) -> bool:
    """A segment written as the next part of a title: it opens with a capital
    letter, or with a closed quote."""
    text = raw[start:end]
    at = start + len(text) - len(text.lstrip())
    ch = raw[at:at + 1]
    return bool(ch) and ((ch in _QUOTE_GLYPHS and _inside(spans, at)) or ch.isupper())


def _en_holds_persian_grammar(obj: str) -> bool:
    """A Persian media noun, play verb or framing word after 'play': the
    Persian arm's sentence, which it declined — the English arm never takes it
    up, and the caller's «لطفا» / «برام» is never searched as a title word."""
    units = _fa_units(_fa_fold(obj)) or []
    return any(
        not u.quoted and (
            _fa_noun_at(units, k) or _fa_verb_at(units, k) is not None or _fa_verb_like(u)
            or _phrase_at(units, k, _FRAMING_STOP | _FA_TAIL)
        )
        for k, u in enumerate(units)
    )


def _en_parse(raw: str, spans: list) -> Optional[str]:
    """The text the anchored English patterns read — `raw` without its
    closed tail tags — or None when `raw` is not ONE English play clause."""

    verb = _EN_VERB_RE.match(raw)
    if verb is None:
        return None
    if _EN_SECOND_VERB_RE.search(_masked(raw, spans), verb.end()):
        return None
    segs = _en_segments(raw, verb.end(), spans)
    if not segs or segs[0][0]:
        return None
    in_tail = False
    for kind, _sep, start, end, after_hard in segs[1:]:
        if _en_words_in(raw[start:end], _EN_TAG):
            in_tail = True
            continue
        if not in_tail and not after_hard and kind == "soft" and _en_title_part(raw, start, end, spans):
            continue
        return None
    head_end = segs[0][3]
    marker = next(
        (m.end() for m in _EN_MARKER_RE.finditer(raw, verb.end(), head_end) if not _inside(spans, m.start())),
        -1,
    )
    # The closed tail tags at the end are the caller's framing, never the
    # query (T6, TB1) — unless they are the title: never inside quotes (the
    # segmenting skipped them), never an assent/modal tag after a title marker
    # ('the song called Well, okay'), never leaving a head made only of class
    # words ('Okay, Okay', 'Oh, Yeah': the whole thing IS the title — TA2).
    cut = len(raw)
    for _kind, sep, start, end, _hard in reversed(segs[1:]):
        tag = raw[start:end]
        if not _en_words_in(tag, _EN_TAG):
            break
        if marker >= 0 and not _en_words_in(tag, _EN_FRAMING):
            break
        head = _en_anchored_query(raw[:sep])
        if head is None or _en_words_in(head, _EN_HEAD_CLASS) or not _class_words(head):
            break
        cut = sep
    # R6-17 NEVER TRUNCATE: a MARKED title ('the song called Pray for Me',
    # 'Pretty Please') and an unpunctuated object ('Halo please') keep their
    # words; the framing that leaves the query is only the separate tail after
    # a clause edge, stripped above.
    kept = raw[:cut].rstrip() if cut < len(raw) else raw
    obj = kept[verb.end():]
    if _FA_SCRIPT_RE.search(obj) and _en_holds_persian_grammar(obj):
        return None
    return kept


def _en_anchored_query(text: str) -> Optional[str]:
    """The subject the anchored English grammar reads from `text` (the first
    pattern that matches), or None."""
    for pat in _EN_PLAY_PATTERNS:
        m = pat.match(text)
        if m:
            return (m.group(1) or "").strip()
    return None


# A closed conversational lead, not a second request.  In the 2026-09-24
# phone trace the caller said «اوکی بعد به چیز دیگه میخوام برام یه دونه آهنگ
# دریکو پلی کن».  The play clause is complete, but the lead made the shared
# fast-path grammar decline it; the agent then answered in prose without
# invoking play_media.  Strip only this specific kind of topic transition.
# The remainder still has to pass every ordinary trouble, negation, reference,
# navigation and full-clause check below before anything can start playing.
_FA_NEW_TOPIC_LEAD_RE = re.compile(
    r"^\s*(?:اوکی|باشه|خب)\s*[,،]?\s*(?:بعد(?:ش)?\s+)?"
    r"(?:به|یه|یک)\s+چیز\s+دیگه\s+می[\s\u200c]*خوام\s*[,،]?\s*"
)


def _strip_fa_new_topic_lead(text: str) -> str:
    match = _FA_NEW_TOPIC_LEAD_RE.match(text)
    return text[match.end():].strip() if match else text


# A recorded Live turn contains ASR false starts, including a partial artist
# name and a partial «پلی»: «برنامه... یه... یه آهنگ... آهنگ دری...
# دریک پ... پلی کن».  Only this complete, bounded Drake play shape is
# repaired.  In particular, «دری» and «پ» require an audible pause marker;
# without one they could be real title words.  The original utterance still
# passes the quote, streaming-service and trouble checks before this repair.
_FA_ASR_PAUSE = r"(?:\.{3,}|…)"
_FA_REPEAT_DET_RE = re.compile(
    rf"^(?P<det>یه|یک)(?:\s*{_FA_ASR_PAUSE}\s*|\s+)(?P=det)(?=\s)"
)
_FA_REPEAT_NOUN_RE = re.compile(
    rf"^(?P<lead>(?:(?:یه|یک)\s+)?)"
    rf"(?P<noun>آهنگ|موزیک|ترانه)(?:\s*{_FA_ASR_PAUSE}\s*|\s+)"
    rf"(?P=noun)(?=\s|$)"
)
_FA_DRAKE_FALSE_START_RE = re.compile(
    rf"^(?:برنامه\s*{_FA_ASR_PAUSE}\s*)?"
    rf"(?:یه(?:\s*{_FA_ASR_PAUSE})?\s+){{0,2}}"
    rf"(?:آهنگ(?:\s*{_FA_ASR_PAUSE})?\s+){{1,2}}"
    rf"(?:دری\s*{_FA_ASR_PAUSE}\s*)?"
    rf"دریک(?:\s+پ\s*{_FA_ASR_PAUSE})?\s+پلی\s+کن[.!?؟]*$"
)
_FA_DRAKE_ARTIST_ONLY_RE = re.compile(
    r"^(?:(?:برام|لطفا)\s+)?دریک(?:و|\s+رو)?\s+(?:پلی|پخش)\s+کن[.!?؟]*$"
)


def _repair_repeated_fa_media_head(text: str) -> str:
    """Collapse only adjacent repeats of request framing at the sentence head.

    The ordinary complete-clause grammar must still accept the result.  This
    handles disfluency for any named artist after «یه یه آهنگ آهنگ», without
    guessing what an unfinished artist name meant or editing a quoted title.
    """
    det = _FA_REPEAT_DET_RE.match(text)
    if det:
        text = det.group("det") + text[det.end():]
    noun = _FA_REPEAT_NOUN_RE.match(text)
    if noun:
        text = noun.group("lead") + noun.group("noun") + text[noun.end():]
    return text


def media_request(text: str) -> Optional[MediaAsk]:
    """Extract a play request a fast path can execute, or None.

    None means "do not take the fast path" — it is not a play request, its
    subject is a back-reference this module cannot resolve, (R6-11 TB1) the
    sentence does not parse completely as the closed play grammar, or (R6-17)
    it holds a word of the closed trouble classes. All of these fall through
    to the agent, which is the only component that holds the conversation;
    both callers hand it the caller's full words (R6-12, proven in
    tests/test_media_r7_tenant.py on the typed and the voice path)."""
    raw = (text or "").strip()
    if not raw or len(raw) > 500:
        return None
    raw = _strip_fa_new_topic_lead(raw)
    if not raw:
        return None
    # TB1: an unclosed quote shields nothing — it declines.
    spans = _quote_spans(raw)
    if spans is None:
        return None
    # The ONE normalized form (residual 5: an Arabic-script mark inside a
    # token is a token boundary; TA5/TB2: the media noun keeps its suffix).
    folded = _fa_fold(raw)
    # Another catalogue entirely — a YouTube search cannot answer it and the
    # agent is the one that should say so. Tested against the FOLDED text as
    # well as the raw one: `نت‌فلیکس` carries a ZWNJ, which is not `\s` to
    # Python's `re`, so the raw form misses the pattern's `نت\s*فلیکس` arm.
    if _STREAMING_SERVICE_RE.search(raw) or _STREAMING_SERVICE_RE.search(folded):
        return None
    units = _fa_units(folded)
    if units is None:
        return None
    # R6-17: ONE closed-class trouble scan of the whole utterance, whichever
    # arm would read it — the caller's own negation / retraction / relative
    # clause (TB1), a pronoun / possessive / demonstrative, an indefinite, an
    # exclusion, a temporal or sequence word, a repair — anywhere outside a
    # double / guillemet / curly-double quote.
    if _trouble(units):
        return None
    # «آهنگ بعد رو بزن» — "the next song". Read on the whole sentence, before
    # either arm strips the noun that makes «بعد» navigation (see below).
    if _FA_NAV_AFTER_NOUN_RE.search(folded):
        return None

    # `mode` is resolved LAST, by the arm that matched: `requested_mode` pulls
    # in `radio.player`, and this function runs on every typed chat message.
    ask = None
    if _FA_SCRIPT_RE.search(raw):
        # Read the original words for every safety gate above.  The repaired
        # text is used only as the closed grammar's input, never to erase a
        # negation, reference, title, or second request from those gates.
        if _FA_DRAKE_FALSE_START_RE.fullmatch(folded) and (
            re.search(_FA_ASR_PAUSE, folded)
            or folded.count("یه ") > 1
            or folded.count("آهنگ ") > 1
        ):
            repaired = "یه آهنگ دریک پلی کن"
            ask = _fa_request(repaired, repaired, _fa_units(repaired))
        else:
            repaired = _repair_repeated_fa_media_head(folded)
            if repaired != folded:
                ask = _fa_request(raw, repaired, _fa_units(repaired))
            else:
                ask = _fa_request(raw, folded, units)
        if ask is None and _FA_DRAKE_ARTIST_ONLY_RE.fullmatch(folded):
            # A bare artist plus an explicit play verb is unambiguous only for
            # this observed, known artist.  Do not make arbitrary bare words
            # fast-play queries: the catalogue resolver can pick a near hit.
            ask = MediaAsk(query="Drake", variety=True, mode=requested_mode(raw), lang="fa")
    if ask is None:
        ask = _en_request(raw, spans)
    # Checked on whichever arm produced the ask: when the Persian arm declines
    # «play آهنگ بعدی», the English arm would otherwise search «آهنگ بعدی».
    if ask is not None and _names_navigation(ask.query):
        return None
    if ask is not None and _asks_for_another(folded, ask.query):
        return None
    return ask


# ── Navigation is not a subject ──────────────────────────────────────────
# "Next"/"previous" move through what is ALREADY playing; the deterministic
# control (`/internal/media-control`) or the agent answers them. Recording
# 2026-09-22 V1: «عوض بشه، بزن آهنگ بعدی» ("change it, play the next song")
# became a YouTube search for the residue «عوض بشه، بعدی» and started a stranger's
# video whose title happened to contain both words. Declining is this module's
# safe direction, so a navigation word left in the query hands the turn on.
#
# The two scripts get different rules on purpose. In Persian «بعدی»/«قبلی» are
# adjectives ("the next/previous one"): one still standing after the play verb,
# the media noun and the fillers are gone refers to the queue, never to a
# title. In English "Next" is an ordinary title word ("Thank U, Next", "Next
# To Me"), so only the navigation SHAPES decline: next/previous directly on a
# media noun ("the next song"), or a query that is nothing but navigation.
_FA_NAV_TOKEN_RE = re.compile(r"^(?:بعدی|قبلی)(?:شو|و|ش|ه)?$")
# «آهنگ بعد» / «آهنگ قبل» — the ezafe ("the song after/before"), and also what
# «بعدی» becomes when the recogniser swallows its final ی. Bare «بعد» is an
# ordinary word ("then", "after", and a title word: «بعد از تو»), so it is
# navigation only directly ON a media noun — and that is decided on the whole
# sentence, because the residue `_names_navigation` sees has lost the noun.
# The price of the safe direction: a title that starts with «بعد»/«قبل» right
# after the noun («آهنگ بعد از تو رو بذار») goes to the agent, which can
# resolve it, instead of the fast path (review F30).
_FA_NAV_AFTER_NOUN_RE = re.compile(
    rf"(?<![{_FA}]){_FA_MEDIA_NOUN_RE.pattern}[{_FA}]*\s+(?:بعد|قبل)[یوشه]*(?![{_FA}])"
)
# A query that is nothing but «بعد»/«قبل» ("play بعد") names no title either.
_FA_NAV_BARE_RE = re.compile(r"^(?:بعد|قبل)[یوشه]*$")
_EN_NAV_TOKENS = frozenset({"next", "previous", "prev", "skip"})
_EN_NAV_NOUNS = frozenset({
    "song", "songs", "track", "tracks", "tune", "video", "clip", "one", "music",
})
_EN_NAV_FILLER = frozenset({"the", "a", "an", "this", "that", "it", "please"}) | _EN_NAV_NOUNS
_TOKEN_EDGE_PUNCT = "\"'«»“”‘’.,،;؛:!?؟()[]-…"


def _names_navigation(query: str) -> bool:
    tokens = [t.strip(_TOKEN_EDGE_PUNCT) for t in normalize_fa(query).split()]
    tokens = [t for t in tokens if t]
    if any(_FA_NAV_TOKEN_RE.match(t) for t in tokens):
        return True
    for i, tok in enumerate(tokens[:-1]):
        if tok in _EN_NAV_TOKENS and tokens[i + 1] in _EN_NAV_NOUNS:
            return True
    content = [t for t in tokens if t not in _EN_NAV_FILLER]
    return bool(content) and all(
        t in _EN_NAV_TOKENS or _FA_NAV_BARE_RE.match(t) for t in content
    )


# ── "Another one" is not a subject either ────────────────────────────────
# "play another song", «یه آهنگ دیگه بذار», «آهنگ دیگه‌ای بذار» are relative to
# what is ALREADY playing: the queue's next, or a choice only the agent can
# make, because only it knows what is on. Searched, they became YouTube queries
# for "another song", «آهنگ» or «ای», and a stranger's video played (reverify
# w5). The relay's grammar reads the subject-less ones as `next` before this
# predicate runs; here they decline, and the agent answers.
#
# Persian: «دیگه/دیگر/دیگری» directly ON a media noun, «یکی» or «چیز». After the
# verb it is the emphatic particle («یه آهنگ بذار دیگه», "come on, play a
# song"), which stays a play.
_FA_ANOTHER_RE = re.compile(
    rf"(?:(?<![{_FA}]){_FA_MEDIA_NOUN_RE.pattern}[{_FA}]*|(?<![{_FA}])یکی"
    rf"|(?<![{_FA}])چیز[{_FA}]*)\s+(?:دیگه|دیگر|دیگری)(?![{_FA}])"
)
# English: a relative word directly on a media noun ("another song by Ebi", "a
# different version"), or a query with nothing left but relative words and
# fillers ("another", "another one", "something else", "some more"). Titles
# that merely contain one ("Another Love", "More Than Words", "One More Time",
# "Another One Bites the Dust") keep their subject and keep the fast path.
_EN_ANOTHER_TOKENS = frozenset({"another", "other", "different", "else", "more"})
_EN_ANOTHER_NOUNS = frozenset({
    "song", "songs", "track", "tracks", "tune", "tunes", "video", "videos",
    "clip", "clips", "music", "version", "versions",
})
_EN_ANOTHER_FILLER = _EN_ANOTHER_NOUNS | frozenset({
    "the", "a", "an", "some", "something", "anything", "one", "ones", "please",
    "me", "us", "for", "just", "this", "that", "it", "like", "of",
})


def _asks_for_another(folded: str, query: str) -> bool:
    if _FA_ANOTHER_RE.search(folded):
        return True
    tokens = [t.strip(_TOKEN_EDGE_PUNCT) for t in normalize_fa(query).split()]
    tokens = [t for t in tokens if t]
    if not any(t in _EN_ANOTHER_TOKENS for t in tokens):
        return False
    for i, tok in enumerate(tokens[:-1]):
        if tok in _EN_ANOTHER_TOKENS and tokens[i + 1] in _EN_ANOTHER_NOUNS:
            return True
    return all(t in _EN_ANOTHER_TOKENS or t in _EN_ANOTHER_FILLER for t in tokens)



def _en_request(raw: str, spans: list) -> Optional[MediaAsk]:
    # R6-11 TB1: ONE English play clause (`_en_parse`); its closed tail tags
    # (", would you?", ", okay?", ", please") frame the request and are not
    # searched, unless they are part of the title (T6/TA2). Nothing else
    # moves: the anchored grammar reads what is left byte for byte.
    kept = _en_parse(raw, spans)
    if kept is None:
        return None
    query = _en_anchored_query(kept)
    if query is None or len(query) < 2:
        return None
    return MediaAsk(
        query=query,
        variety=bool(_FA_OPEN_RE.search(query.lower())),
        mode=requested_mode(raw),
        lang="en",
    )


#: What a Persian sentence has to score before a fast path may ACT on it.
#:
#: `fa_media_score` returns 2 for a bare strong token — a sentence that merely
#: MENTIONS music — and 3 only for noun+verb or consume-verb+open-marker, i.e.
#: an actual request to put something on. The classifier wants the lower bar
#: (2 is enough to widen the tool set so the agent gets `play_media`, which is
#: free and reversible); the two fast paths must not, because both act
#: irreversibly with no relevance floor underneath them: `ws_chat`'s scrapes
#: YouTube and broadcasts `media_play`, and the relay's calls
#: `_play_media_direct`, while `media_resolve.pick_best` returns `scored[0]`
#: for any non-empty pool. At the score-2 bar "موسیقی چیه؟" (what is music?)
#: became a search for "چیه؟" and started playing a stranger's video in answer
#: to a question. Declining costs a round trip through the agent, which is the
#: component that holds the context anyway.
#:
#: The price is real and was measured, so the Persian acceptance run has
#: something to check "it felt slow" against. Six verb-less asks that NAME what
#: to play score exactly 2 and now decline — they play one AgentRunner round
#: trip later instead of in ~1 s: «یه آهنگ از ابی», «آهنگ ساری گلین»,
#: «آهنگ جدید ابی», «یه موزیک ویدیو از شادمهر», «موزیک آروم»,
#: «یه پادکست درباره تاریخ». Lowering the bar back to 2 is NOT the fix:
#: «دیشب یه آهنگ قشنگ شنیدم» (I heard a nice song last night) also scores 2
#: with a non-empty residue, so "has a named subject" cannot separate the two.
#: A narrow score-2 ACT arm (media noun + residue + no interrogative + no
#: report/past marker) is the candidate, and it is a behaviour change on a
#: platform-deploying path — it needs its own evidence round, not this one.
_FA_ACT_MIN_SCORE = 3


def _fa_request(raw: str, n: str, units: list) -> Optional[MediaAsk]:
    # `n` is the ONE folded form (residual 5: «آره،ابی» is two tokens, as if
    # the space had been typed; TA5/TB2: «آهنگ‌های» / «آهنگ های» /
    # «آهنگهای» are one token); `units` its units.
    if fa_media_score(n) < _FA_ACT_MIN_SCORE:
        return None
    words = n.split()
    # R6-10 TA6: «آهنگ رو بذار کنار» — the placing verb with its directional
    # particle is "put aside", not a play request.
    if _puts_aside(words):
        return None
    # R6-11 TB1: the whole sentence parses as the closed grammar, or the
    # agent answers.
    parse = _fa_parse(units)
    if parse is None:
        return None
    # The caller's framing is not a subject (R6-6): the grammar has already
    # set the lead, the modal framing, the tail tags and the punctuation
    # aside; a named title (quoted, or after a title marker, up to the
    # caller's framing words) is never filtered: its words ARE the query.
    residue = _fa_residue(parse)
    if parse.seq_lead and not residue:
        # «بعد یه آهنگ بذار» — nothing named once the "then" is set aside.
        return None
    named_title = any(title for _t, title in residue)
    residue_words = [w for t, _title in residue for w in _class_words(t)]
    if (
        not named_title and residue_words
        and _run_length(residue_words, _ASSENT_CLASS) == len(residue_words)
    ):
        # Only an assent is left («یه پادکست آره پخش کن»): it names
        # nothing, so the named noun below is the query.
        residue = []
    query = " ".join(t for t, _title in residue).strip()

    if query:
        # A real subject was named: play it. `variety` only when the ask also
        # says "a"/"some" — "یه آهنگ از ابی" is one of Ebi's songs, any of
        # them; "آهنگ ساری گلین" is that one — and never for a named title
        # (TA3): «یه آهنگ به اسم «باشه»» is that song, not a random one.
        # In the observed request «دریکو» is the artist Drake plus colloquial
        # object marker «رو». Searching that surface form returns Draco Malfoy
        # videos. Keep this alias exact: stripping a final «و» generically
        # would damage legitimate artist names and song titles.
        if not named_title and query in {"دری دریک", "دریک پ", "دری دریک پ"}:
            # Unmarked partial ASR words are also plausible title words; do
            # not send the combined phrase to a fuzzy catalogue search.
            return None
        catalog_query = "Drake" if query in {"دریک", "دریکو"} and not named_title else query
        return MediaAsk(
            query=catalog_query,
            variety=bool(_FA_OPEN_RE.search(n)) and not named_title,
            mode=requested_mode(raw),
            lang="fa",
        )

    # TA5: a media noun with a possessive/definite clitic points back too.
    if _FA_REFERENTIAL_RE.search(n) or _refers_back(words):
        # "the usual one", "that one again" — only the agent knows which.
        return None

    noun_m = _FA_MEDIA_NOUN_RE.search(n)
    if noun_m:
        # No subject at all, but they named what KIND: "یه آهنگ پخش کن".
        # Their own word is the query, so a Persian ask reaches a Persian
        # catalogue. Subject-less is open-ended by construction.
        return MediaAsk(query=noun_m.group(0), variety=True,
                        mode=requested_mode(raw), lang="fa")

    if _FA_CONSUME_RE.search(n) and _FA_OPEN_RE.search(n):
        # "بذار یه چیزی گوش بدم" — no noun and no subject, but explicitly
        # "something to listen to".
        return MediaAsk(query=_FA_GENERIC_QUERY, variety=True,
                        mode=requested_mode(raw), lang="fa")

    # A bare "پلی کن" / "بزن" with nothing to play: the agent has the context
    # this module does not.
    return None
