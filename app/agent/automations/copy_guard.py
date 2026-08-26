"""The copy contract, enforced in code (R30 §5.7, D-08).

One list, two repos: `fixtures/automations/banned-copy.json` is the
contract document both repos carry byte-identical; this module embeds
the same data so the guard works inside a deployed image that ships no
fixtures directory. `test_automation_copy_guard.py` pins the embedded
data to the JSON — drift fails CI, so there is still exactly one list.

Scanned surfaces: app string tables (B's scanner), server-rendered
sentence templates and agent output (this one). Never scanned:
identifiers, wire keys, component names, or quoted vendor content —
`Item.title`, `Item.sub`, `msg.text` are the vendor's own words
(a Jira status like "In progress → Reopened" is not ours to rewrite).

Matching is whole-word and case-sensitive, per the contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Keep byte-identical with fixtures/automations/banned-copy.json.
BANNED_WORDS: tuple[str, ...] = (
    "JQL",
    "poll",
    "polls",
    "polling",
    "trigger",
    "routine",
    "job",
    "workflow",
    "executed",
    "fetch",
    "Partial",
)

BANNED_PHRASES: tuple[str, ...] = (
    "Mission Control",
    "Temporarily unavailable",
    "re-authentication",
    "In progress",
    # R31 §4.3(7) — each one was read off a founder's screen on 26 August.
    # Both cases: matching is case-SENSITIVE, so banning only the
    # lower-case form left `"An account refused the token."` — the home
    # card's own needs-you sentence — passing the guard that existed to
    # stop exactly it. A sentence-initial noun phrase is the likeliest
    # place for this defect, not the least.
    "an account",
    "An account",
    "Reasoned it through",
    "Reasoned over gathered sources",
    "TEST RUN",
    "undo window",
    "Migrated from",
    "Already running",
    "Nothing recorded for this task yet",
)

#: Exact strings sanctioned despite containing banned words: the shipped
#: sidebar sub-label, and the two canvas-mandated workflow strings
#: (§3.8 menu row, and the canvas header renamed to "Workflow" by
#: R31-23) — the canvas wins over the list for its own drawn copy.
#: "The whole workflow" is RETIRED and absent on purpose: the scanner is
#: what fails if the rename is missed.
WHITELIST_EXACT: tuple[str, ...] = (
    "automation · mobile · routine",
    "Edit workflow",
    "Workflow",
)

#: UI glyphs that are not emoji.
WHITELIST_GLYPHS: frozenset[str] = frozenset("✓✕●⋯‹›")

_ISO_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_PERCENT_RE = re.compile(r"\d{1,3}\s?%")
_RAW_TOOL_ID_RE = re.compile(r"\b[a-z]+__[a-z_]+\b")

#: R31-26. Naming the zone is not the fix for a UTC stamp; rendering in
#: the user's own zone is. A sentence that says "UTC" has already lost.
_UTC_MARKER_RE = re.compile(r"\b(?:UTC|GMT)\b")

#: R31-24. Markdown reaches a thread bubble as literal characters: the
#: thread renders agent turns as plain text, so "**paused**" is what the
#: user reads. These three cover what the app/web formatting section
#: teaches the model to type.
_MD_EMPHASIS_RE = re.compile(r"\*\*")
_MD_CODE_SPAN_RE = re.compile(r"`")
_MD_LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*[-*+][ \t]+")

#: R31-02 / R31-04 — our own internal prefixes, deleted at their source.
_BRACKET_TAG_RE = re.compile(r"\[(?:test|automation)\]")

#: R31-25 — rendered mode only; see `scan`.
_PLACEHOLDER_RE = re.compile(r"[{}]")

#: Emoji detection without a Unicode-property regex engine: the blocks
#: that cover Emoji_Presentation / Extended_Pictographic in practice,
#: minus the whitelisted UI glyphs.
_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F000, 0x1FAFF),  # pictographs, emoticons, symbols, extended-A
    (0x2600, 0x27BF),    # misc symbols + dingbats (⚠ ✅ live here)
    (0x2B00, 0x2BFF),    # arrows/symbols incl. ⭐
    (0xFE00, 0xFE0F),    # variation selectors
    (0x1F1E6, 0x1F1FF),  # regional indicators
    (0x2190, 0x21FF),    # arrows — glyph noise the design never uses
)


#: Every regex rule `scan` applies, in order, as (name, pattern). The
#: list is data rather than a run of `if`s so `contract_dict` can
#: report what is ACTUALLY enforced: the parity test compared four keys
#: and never asked whether a `banned_patterns` entry in the fixture had
#: an implementation, so a rule could be written down, published in the
#: contract, and enforce nothing — green either way.
_PATTERN_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("iso_timestamp", _ISO_TIMESTAMP_RE),
    ("percent_complete", _PERCENT_RE),
    ("raw_tool_id", _RAW_TOOL_ID_RE),
    ("utc_marker", _UTC_MARKER_RE),
    ("markdown_emphasis", _MD_EMPHASIS_RE),
    ("markdown_code_span", _MD_CODE_SPAN_RE),
    ("markdown_list_marker", _MD_LIST_MARKER_RE),
    ("bracket_tag", _BRACKET_TAG_RE),
)

#: The two rules that are not a plain regex sweep: the brace rule runs
#: in rendered mode only, and emoji is a codepoint scan.
_SPECIAL_RULES: tuple[str, ...] = ("unrendered_placeholder", "emoji")


@dataclass(frozen=True)
class Violation:
    rule: str
    needle: str

    def __str__(self) -> str:  # pragma: no cover — debugging nicety
        return f"{self.rule}: {self.needle!r}"


def _mask_whitelist(text: str) -> str:
    for allowed in WHITELIST_EXACT:
        if allowed in text:
            text = text.replace(allowed, " " * len(allowed))
    return text


def _is_emoji(ch: str) -> bool:
    if ch in WHITELIST_GLYPHS:
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES)


def scan(text: str, *, rendered: bool = True) -> list[Violation]:
    """Scan one authored string. Returns every violation found (empty
    list = clean). Callers exempt vendor-quoted content by simply not
    scanning it.

    ``rendered`` (R31-25): a RENDERED string is one a user can read, and
    it may not contain a brace — a brace there means a slot nobody
    filled. A TEMPLATE still carries its slots, so the brace rule is off
    for it. Everything else applies to both.

    The distinction is the whole defect. `"{count} issues moved ·
    {need_count} needs you"` passed every check this repo had, because
    the pin rendered it with `str.format(count=3, need_count=1, …)` — a
    kwargs bag generous enough to fill a slot the production renderer
    (`automation_verbs._n`, which substitutes `{n}` and `{count}` and
    nothing else) could not. The template was clean, the render was
    clean, and the screen read `0 issues moved · {need_count} needs
    you`. A template scan can never catch that; only scanning what the
    production path actually produced can.
    """
    if not text:
        return []
    haystack = _mask_whitelist(text)
    found: list[Violation] = []
    if rendered:
        placeholder = _PLACEHOLDER_RE.search(haystack)
        if placeholder:
            found.append(Violation("unrendered_placeholder", placeholder.group(0)))
    for phrase in BANNED_PHRASES:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", haystack):
            found.append(Violation("banned_phrase", phrase))
    for word in BANNED_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", haystack):
            found.append(Violation("banned_word", word))
    if _ISO_TIMESTAMP_RE.search(haystack):
        found.append(Violation("iso_timestamp", _ISO_TIMESTAMP_RE.search(haystack).group(0)))
    if _PERCENT_RE.search(haystack):
        found.append(Violation("percent_complete", _PERCENT_RE.search(haystack).group(0)))
    raw_tool = _RAW_TOOL_ID_RE.search(haystack)
    if raw_tool:
        found.append(Violation("raw_tool_id", raw_tool.group(0)))
    for rule, pattern in _PATTERN_RULES:
        if rule in ("iso_timestamp", "percent_complete", "raw_tool_id"):
            continue  # reported above, with their own messages
        hit = pattern.search(haystack)
        if hit:
            found.append(Violation(rule, hit.group(0)))
    for ch in haystack:
        if _is_emoji(ch):
            found.append(Violation("emoji", ch))
            break
    return found


def clean(text: str, *, rendered: bool = True) -> bool:
    """True when the authored string passes the contract."""
    return not scan(text, rendered=rendered)


def contract_dict() -> dict:
    """The embedded contract in the JSON fixture's shape — the parity
    test compares this against fixtures/automations/banned-copy.json."""
    return {
        "banned_words": list(BANNED_WORDS),
        "banned_phrases": list(BANNED_PHRASES),
        "whitelist_exact": list(WHITELIST_EXACT),
        "whitelist_glyphs": sorted(WHITELIST_GLYPHS),
        # What is ACTUALLY enforced, derived from the rules `scan`
        # runs — not a hand-kept list that could agree with the
        # fixture while the scanner ignores half of it.
        "banned_pattern_names": sorted(
            [name for name, _ in _PATTERN_RULES] + list(_SPECIAL_RULES)
        ),
    }
