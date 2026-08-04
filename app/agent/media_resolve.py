"""Shared media resolution: search, pick a result, and prove it answers the ask.

WHY THIS MODULE EXISTS. Two producers resolve "play me X" independently — the
fast path in `api/ws_chat.py` and the `play_media` tool in
`agent/tool_executor.py` — and they drifted apart in three ways that all reached
the user as the same symptom, a card playing one song while the agent announced
another:

  1. The fast path anchors its title regex to the video id it chose; the tool
     did not, so it paired the FIRST id on the results page with the FIRST title
     on the page — different results whenever a shelf, ad or Shorts row sits
     between them.
  2. Neither checked that the resolved track has anything to do with the
     request. `variety` then made it worse by picking a UNIFORMLY RANDOM track
     out of the top eight hits, which is right for "play me some Drake" and
     completely wrong for "play Daman Zardoo" — the 2026-08-03 17:05 screenshot,
     where the agent said "Playing Daman Zardoo now" over ONEDAM — Del Tanha.
  3. When the fast path could not find a title it used the RAW USER QUERY as
     one, which is how a card came to be titled lowercase "ebi" — and that
     fake title then seeded the station, so the whole queue drifted to other
     artists.

The rules live here once, and both callers use them.
"""
from __future__ import annotations

import json
import logging
import random
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Words that carry no identifying information about WHICH track is wanted.
# Stripped before deciding whether a request names a specific song.
_FILLER = {
    "play", "playing", "put", "on", "me", "some", "a", "an", "the", "song",
    "songs", "music", "track", "by", "from", "please", "now", "video", "watch",
    "listen", "to", "for", "of", "my", "i", "want", "wanna", "lets", "let",
    "us", "hey", "toup", "and", "with", "something", "anything", "good",
    "best", "top", "new", "old", "next", "another", "more", "yt", "youtube",
}

_WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


# Characters NFKD does not decompose — they are distinct letters, not accented
# ones — but which people routinely type as their ASCII lookalike.
_FOLD_MAP = str.maketrans({
    "ø": "o", "œ": "oe", "æ": "ae", "ß": "ss", "đ": "d", "ð": "d",
    "þ": "th", "ł": "l", "ı": "i", "ʻ": "", "ʼ": "", "’": "", "'": "",
})


def normalize(text: str) -> str:
    """Lowercase, FOLD DIACRITICS, strip bracketed noise and production tags.
    Used for matching only — never for anything the user sees.

    The folding is load-bearing, not cosmetic. People type "beyonce halo",
    "titi me pregunto", "celine dion" — without the accents the catalogue
    actually uses. Byte-comparing those against "Beyoncé - Halo" scores them a
    MISS on every accented word, so a random re-upload that happens to spell
    the name in ASCII outranks the official recording, and the mismatch warning
    fires on a result that is perfectly correct (the agent then apologises for
    failing to find the song it is playing). Verified against live YouTube
    results for all three of those queries.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKD", text.casefold())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(_FOLD_MAP)
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(
        r"\b(official|music|video|audio|lyrics?|hd|4k|remaster(ed)?|mv|full|"
        r"visualizer|live|version|feat|ft)\b",
        " ",
        s,
    )
    return re.sub(r"\s+", " ", s).strip()


def tokens(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(normalize(text)) if len(w) > 1]


def content_tokens(query: str) -> List[str]:
    """Query words that actually identify a track (filler removed)."""
    return [w for w in tokens(query) if w not in _FILLER]


def looks_specific(query: str) -> bool:
    """Does this request NAME something, or is it open-ended?

    Two or more identifying words means the user had a particular track (or a
    track-by-artist) in mind: "daman zardoo", "ebi gheseh eshgh". One word is an
    artist, a genre or a mood: "ebi", "drake", "jazz" — those are the requests
    variety exists for. Getting this wrong in the permissive direction plays a
    stranger's song; in the strict direction it merely replays a favourite, so
    the bias is deliberate.
    """
    return len(content_tokens(query)) >= 2


def relevance(query: str, title: str, artist: str = "") -> float:
    """Fraction of the request's identifying words present in this candidate.
    1.0 means every word the user said appears in the title/artist."""
    want = content_tokens(query)
    if not want:
        return 1.0  # nothing was specified, so nothing can mismatch
    hay = f"{normalize(artist)} {normalize(title)}"
    hay_tokens = set(_WORD_RE.findall(hay))
    hit = 0
    for w in want:
        # Substring as well as token equality: "zardoo" inside "zardooo",
        # transliterations, and joined words are common in this catalogue.
        if w in hay_tokens or w in hay:
            hit += 1
    return hit / len(want)


# A candidate must carry at least this share of the request's words to be
# considered an answer to it. Two-word asks therefore need both words; a
# four-word ask tolerates one miss (titles carry extra words legitimately).
MATCH_THRESHOLD = 0.6


def matches_request(query: str, title: str, artist: str = "") -> bool:
    return relevance(query, title, artist) >= MATCH_THRESHOLD


# Download-farm / SEO-spam markers.
#
# Pure word-overlap cannot see these: a download site's whole strategy is to
# stuff the user's exact words into the title, so it scores at or ABOVE the
# legitimate upload, which often has a shorter, cleaner title. On 2026-08-03 a
# request for a Lori song resolved to "دانلود آهنگ لری ار نبینمش خر…" — a
# download-site rip with a dead thumbnail — over the real track.
#
# A DEMERIT, never a filter. Regional catalogues are full of these and
# sometimes they are genuinely all that exists; declining to play anything
# would be worse than playing the least-bad one. This only ever reorders
# candidates that already answer the request.
_JUNK_MARKERS = (
    "دانلود",          # "download" — the Persian download farms
    "دانلو",
    "دریافت آهنگ",     # "get the song"
    "download",
    "free mp3",
    "mp3 download",
    "320kbps",
    "128kbps",
    "full album zip",
    "www.",
)


def junk_score(title: str) -> float:
    """How much download-farm signal a title carries. 0.0 is clean.

    Deliberately NOT `normalize()`: that strips bracketed segments and
    production tags, so a "[دانلود]" tag — exactly where these markers like to
    sit — would be erased before it could be counted.
    """
    low = (title or "").casefold()
    return float(sum(1 for m in _JUNK_MARKERS if m in low))


def pick_best(
    query: str,
    candidates: Sequence[Tuple[str, str, str]],
    variety: bool = False,
) -> Optional[Tuple[str, str, str]]:
    """Choose from (video_id, title, artist) candidates in search-rank order.

    Specific request  → the most relevant candidate, ties broken by rank. Never
                        random, whatever `variety` says: the model sets that
                        flag from phrasing and it is wrong often enough that
                        trusting it blind is what played the wrong song.
    Open-ended + variety → a random pick among candidates that still match
                        (for a one-word artist ask every song by that artist
                        matches, which is exactly the intended pool).
    """
    pool = [c for c in candidates if c and c[0] and c[1]]
    if not pool:
        return None
    if variety and not looks_specific(query):
        relevant = [c for c in pool if matches_request(query, c[1], c[2])] or pool
        # Keep the download farms out of the random pool when anything else is
        # available — a random pick is the one place a demerit cannot help.
        clean = [c for c in relevant if not junk_score(c[1])] or relevant
        return random.choice(clean[:8])
    # Relevance first, then quality, then search rank. Quality is a TIE-BREAK
    # below relevance on purpose: it reorders answers to the request, it never
    # promotes a cleaner-looking result that answers something else.
    scored = sorted(
        (
            (relevance(query, t, a), -junk_score(t), -i, (v, t, a))
            for i, (v, t, a) in enumerate(pool)
        ),
        key=lambda x: (x[0], x[1], x[2]),
        reverse=True,
    )
    return scored[0][3]


# ── YouTube results-page scraping ─────────────────────────────────────────
# ONE implementation. The title regex is anchored to the id it belongs to and
# refuses to cross into the next result — without that guard a page whose chosen
# block carries no title silently borrows the following video's title, which is
# how HUMBLE once played under a Rakim reaction video's name (2026-07-31).
_ID_RE = re.compile(r'"videoId":"([a-zA-Z0-9_-]{11})"')


def _title_for_id(html: str, video_id: str) -> str:
    m = re.search(
        re.escape(f'"videoId":"{video_id}"')
        + r'(?:(?!"videoId":")[\s\S]){0,3000}?'
        + r'"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"',
        html,
    )
    if not m:
        return ""
    raw = m.group(1)
    try:
        return json.loads(f'"{raw}"')  # YouTube JSON-escapes &, quotes, unicode
    except Exception:
        return raw


def scrape_results(html: str, limit: int = 6) -> List[Tuple[str, str, str]]:
    """(video_id, title, artist='') for the first `limit` DISTINCT results that
    have a title of their own. Results without one are dropped, never
    back-filled from a neighbour or from the user's query."""
    out: List[Tuple[str, str, str]] = []
    seen: set = set()
    for vid in _ID_RE.findall(html or ""):
        if vid in seen:
            continue
        seen.add(vid)
        title = _title_for_id(html, vid)
        if not title:
            continue
        out.append((vid, title, ""))
        if len(out) >= limit:
            break
    return out


def split_artist_title(label: str) -> Tuple[str, str]:
    """Best-effort "Artist - Title" split, for callers that only have the
    combined label (the fast path's scrape, the variety pick's display name).
    Returns ("", label) when there is no separator — an empty artist is honest;
    a guessed one disables the artist guard downstream."""
    if not label:
        return "", ""
    for sep in (" - ", " – ", " — ", " | "):
        if sep in label:
            left, right = label.split(sep, 1)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    return "", label.strip()


def describe_mismatch(query: str, title: str, artist: str = "") -> Optional[str]:
    """A sentence for the MODEL when the resolved track does not answer the
    request, or None when it does. The agent must not be left to discover this
    on its own — it cannot see the card."""
    if not looks_specific(query):
        return None
    if matches_request(query, title, artist):
        return None
    named = f"{artist} - {title}" if artist else title
    return (
        f'WARNING: the user asked for "{query}" but the only thing that could be '
        f'found and started is "{named}", which does not match. Tell them plainly '
        f'that you could not find "{query}", say what is playing instead, and ask '
        f'whether they want it or something else. Do NOT claim you are playing '
        f'"{query}".'
    )
