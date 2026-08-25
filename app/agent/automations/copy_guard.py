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
)

#: Exact strings sanctioned despite containing banned words: the shipped
#: sidebar sub-label, and the two canvas-mandated workflow strings
#: (§3.8 menu row, §3.9 canvas header) — the canvas wins over the list
#: for its own drawn copy.
WHITELIST_EXACT: tuple[str, ...] = (
    "automation · mobile · routine",
    "Edit workflow",
    "The whole workflow",
)

#: UI glyphs that are not emoji.
WHITELIST_GLYPHS: frozenset[str] = frozenset("✓✕●⋯‹›")

_ISO_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_PERCENT_RE = re.compile(r"\d{1,3}\s?%")
_RAW_TOOL_ID_RE = re.compile(r"\b[a-z]+__[a-z_]+\b")

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


def scan(text: str) -> list[Violation]:
    """Scan one authored string. Returns every violation found (empty
    list = clean). Callers exempt vendor-quoted content by simply not
    scanning it."""
    if not text:
        return []
    haystack = _mask_whitelist(text)
    found: list[Violation] = []
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
    for ch in haystack:
        if _is_emoji(ch):
            found.append(Violation("emoji", ch))
            break
    return found


def clean(text: str) -> bool:
    """True when the authored string passes the contract."""
    return not scan(text)


def contract_dict() -> dict:
    """The embedded contract in the JSON fixture's shape — the parity
    test compares this against fixtures/automations/banned-copy.json."""
    return {
        "banned_words": list(BANNED_WORDS),
        "banned_phrases": list(BANNED_PHRASES),
        "whitelist_exact": list(WHITELIST_EXACT),
        "whitelist_glyphs": sorted(WHITELIST_GLYPHS),
    }
