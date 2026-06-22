"""Cheap, model-agnostic token budgeting for web tool output.

We avoid a real tokenizer (no tiktoken dependency, no per-call cost, works for
both OpenAI and Anthropic models) and use a ~4-chars-per-token heuristic, which
is accurate enough to bound context growth. Truncation is single-pass and trims
to a word boundary when one is close.
"""
from __future__ import annotations

_CHARS_PER_TOKEN = 4
_TRUNC_MARKER = "\n... (truncated to fit token budget)"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return ``text`` trimmed to at most ``max_tokens`` (estimated). No-op when
    it already fits; trims back to a nearby word boundary and appends a marker."""
    text = text or ""  # total over None/empty — never return non-str
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    char_budget = max_tokens * _CHARS_PER_TOKEN
    cut = text[:char_budget]
    sp = cut.rfind(" ")
    if sp > char_budget - 200 and sp > 0:  # snap to a word boundary if one is close
        cut = cut[:sp]
    return cut.rstrip() + _TRUNC_MARKER
