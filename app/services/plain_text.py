"""Markdown → plain text for notification surfaces (Round 4, item 4).

APNs alert bodies, Live Activity content-state strings (``title``,
``subtitle``, ``stepName``, ``preview``) and Expo push bodies render text
verbatim — a model answer that opens with ``**Gemini 3.7 Flash**`` puts
literal asterisks on the lock screen. This is the ONE stripper every
notification-surface writer goes through, on both sides of the wire:

* the platform (``apns_push``, ``expo_push``) strips before capping, so
  any producer — including old agent images — is covered;
* the agent-side producers strip BEFORE they slice a preview, so a
  ``**`` pair can't be cut in half by ``[:120]`` and survive the strip.

Deliberately conservative and stdlib-only: this must import cleanly in
the platform image (no bs4/markdown packages) and it must never make
text LONGER or raise. Anything it does not recognise is left alone.

Not applied to Telegram/WhatsApp fallback delivery — those lanes render
markdown on purpose.
"""

from __future__ import annotations

import re
from typing import Optional

# Fenced code blocks: keep the code, drop the fences + language tag.
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]*)`")
# Images before links so ![alt](src) → alt (not "!alt").
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)\s]+)(?:\s+\"[^\"]*\")?\)")
# Product chips the web/mobile clients render as buttons: [[navigate:/x]],
# [[open_app:slug]], [[Play TITLE on Netflix]] — meaningless as text.
_CHIP_RE = re.compile(r"\[\[[^\]]*\]\]")
# Round 15: the HOLE a chip leaves. Removing "[[navigate:/jobs]]" from
# "You can watch it in [[navigate:/jobs]]." shipped "You can watch it in ."
# to a real user; this surface (push bodies, Telegram, SMS) would have sent
# the same sentence, because it strips chips with the same blunt sub. Two
# rules, matching `frontend/src/modules/chat/chipDirectives.ts`:
#   1. a chip at the END takes the clause that introduced it;
#   2. punctuation and whitespace close back up around the gap.
# Rule 1 is anchored to a chip actually having been there — the same trailing
# word in ordinary prose ("the toggle is on.") must survive.
_CHIP_TAIL_RE = re.compile(
    r"[ \t]*\b(?:in|at|on|to|from|into|under|via|inside|over|here|below|above)\b"
    r"[ \t]*[.:;,!?…]*[ \t]*(?=\n|$)",
    re.IGNORECASE,
)
_CHIP_AT_END_RE = re.compile(r"\[\[[^\]]*\]\][ \t.,;:!?…]*(?=\n|$)")
_GAP_SPACE_RE = re.compile(r"[ \t]{2,}")
_GAP_PUNCT_RE = re.compile(r"[ \t]+([.,;:!?…])")


_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_HEADING_TRAIL_RE = re.compile(r"[ \t]+#{1,6}[ \t]*$", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)
# Bullets: "- ", "* ", "+ ", "1. ", "1) " at line start.
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,3}[.)])[ \t]+", re.MULTILINE)
_HR_RE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE)
# Emphasis. Order matters: *** → ** → * / ___ → __ → _. Underscore emphasis
# only when it wraps a word boundary, so snake_case identifiers survive.
_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|___)(?=\S)(.+?)(?<=\S)\1")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_STAR_RE = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_ITALIC_UNDER_RE = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
# HTML tags the model occasionally emits (<br>, <b>, <sub>…). Only real
# tag shapes — "a < b" and "<3" are left alone.
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s+[^<>]*?)?/?>")
# Table rows: drop the pipes; separator rows (|---|---|) go entirely.
_TABLE_SEP_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", re.MULTILINE)
_TABLE_PIPE_RE = re.compile(r"[ \t]*\|[ \t]*")
# Escaped markdown chars: \* \_ \# \` \[ \] \( \) \> \- \! \|  — parked
# behind private-use sentinels BEFORE any syntax pass runs (so "\*not
# bold\*" is never seen as emphasis) and restored as the bare char at the
# end.
_ESCAPABLE = "\\`*_{}[]()#+-.!|>~"
_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")
_SENTINEL_BASE = 0xE100
_SENTINEL_RE = re.compile("[%s-%s]" % (chr(_SENTINEL_BASE), chr(_SENTINEL_BASE + len(_ESCAPABLE))))
# LaTeX inline math left as its body: $x^2$ → x^2 ; \( … \) → …
_LATEX_INLINE_RE = re.compile(r"\\\((.+?)\\\)|\$(?!\$)([^$\n]+?)\$")
_LATEX_BLOCK_RE = re.compile(r"\\\[(.+?)\\\]|\$\$(.+?)\$\$", re.DOTALL)


def _latex_inline_repl(m: "re.Match[str]") -> str:
    r"""``\( … \)`` is always math. ``$…$`` is math only when the body looks
    like TeX (``\``, ``^``, ``{``, ``}``) — "$5 and $10 today" is prices, and
    prices are far more common on a lock screen than inline LaTeX."""
    if m.group(1) is not None:
        return m.group(1).strip()
    body = m.group(2) or ""
    if any(ch in body for ch in "\\^{}"):
        return body.strip()
    return m.group(0)


def _strip_chips(s: str) -> str:
    """Remove chips AND the residue they leave, line by line."""
    out = []
    for line in s.split("\n"):
        if "[[" not in line:
            out.append(line)
            continue
        at_end = bool(_CHIP_AT_END_RE.search(line))
        line = _CHIP_RE.sub("", line)
        line = _GAP_SPACE_RE.sub(" ", line)
        line = _GAP_PUNCT_RE.sub(r"\1", line)
        if at_end:
            trimmed = _CHIP_TAIL_RE.sub("", line).rstrip()
            if trimmed != line.rstrip():
                line = f"{trimmed}." if trimmed.strip() and trimmed[-1] not in ".!?…:" else trimmed
        # A line reduced to punctuation is residue, not content.
        out.append("" if line.strip() and not re.sub(r"[.,;:!?…\s]", "", line) else line)
    return "\n".join(out)


def strip_markdown(text: Optional[str], *, collapse_whitespace: bool = False) -> str:
    """Return ``text`` with markdown syntax removed and its words intact.

    ``collapse_whitespace=True`` additionally folds all runs of whitespace
    (including newlines) into single spaces — the shape single-line
    surfaces (LA ``stepName``/``preview``, alert titles) want. Multi-line
    surfaces (alert bodies) keep their line breaks.

    Idempotent; never raises; returns ``""`` for None/empty.
    """
    if not text:
        return ""
    s = str(text)
    try:
        s = _ESCAPE_RE.sub(
            lambda m: chr(_SENTINEL_BASE + _ESCAPABLE.index(m.group(1))), s,
        )
        s = _FENCE_RE.sub(lambda m: m.group(1), s)
        s = _INLINE_CODE_RE.sub(lambda m: m.group(1), s)
        s = _LATEX_BLOCK_RE.sub(lambda m: (m.group(1) or m.group(2) or "").strip(), s)
        s = _LATEX_INLINE_RE.sub(_latex_inline_repl, s)
        s = _IMAGE_RE.sub(lambda m: m.group(1), s)
        s = _LINK_RE.sub(lambda m: m.group(1), s)
        s = _strip_chips(s)
        s = _HTML_TAG_RE.sub("", s)
        s = _TABLE_SEP_RE.sub("", s)
        if "|" in s:
            s = "\n".join(
                _TABLE_PIPE_RE.sub(" ", ln).strip() if ln.lstrip().startswith("|") else ln
                for ln in s.split("\n")
            )
        s = _HEADING_RE.sub("", s)
        s = _HEADING_TRAIL_RE.sub("", s)
        s = _BLOCKQUOTE_RE.sub("", s)
        s = _HR_RE.sub("", s)
        s = _BULLET_RE.sub("", s)
        s = _BOLD_ITALIC_RE.sub(lambda m: m.group(2), s)
        s = _BOLD_RE.sub(lambda m: m.group(2), s)
        s = _STRIKE_RE.sub(lambda m: m.group(1), s)
        s = _ITALIC_STAR_RE.sub(lambda m: m.group(1), s)
        s = _ITALIC_UNDER_RE.sub(lambda m: m.group(1), s)
        # A stray unmatched "**" / "__" left by a truncated pair.
        s = s.replace("**", "").replace("__", "")
        s = _SENTINEL_RE.sub(lambda m: _ESCAPABLE[ord(m.group(0)) - _SENTINEL_BASE], s)
    except Exception:  # noqa: BLE001 — a stripper must never break a push
        s = str(text)
    if collapse_whitespace:
        return " ".join(s.split())
    # Trim trailing spaces per line and collapse 3+ blank lines.
    s = "\n".join(ln.rstrip() for ln in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def plain_preview(text: Optional[str], limit: int) -> str:
    """Strip THEN slice — the order matters (a ``**`` pair cut in half by
    the slice would survive the strip). Single-line."""
    return strip_markdown(text, collapse_whitespace=True)[:limit]


# ── Push copy gates (2026-08-19) ─────────────────────────────────────────────
# The founder's lock screen read "Dispatch background conversation / On it —
# two tasks running:". Both halves were leaks with the same shape: model- or
# turn-authored text reaching a user-facing notification surface verbatim.
# These two helpers are the seam every producer routes through.

# A LABEL is internal jargon when it talks about the machinery instead of the
# errand. Patterns, not single words: "check my Telegram channel" is a real
# ask and must survive; "Dispatch background conversation" must not.
_INTERNAL_LABEL_RES = [
    re.compile(r"\b(dispatch(?:ed|ing)?|spawn(?:ed|ing)?|orchestrat\w*)\b.{0,24}\b(background|sub-?agent|mission|conversation|session|job|task)\b", re.IGNORECASE),
    re.compile(r"\bbackground\s+(conversation|sub-?agent|session|turn|job|task|work)\b", re.IGNORECASE),
    re.compile(r"\b(sub-?agent|chatturn|agent[_ ]task|autopilot tick)\b", re.IGNORECASE),
    re.compile(r"^(job|task|mission|dispatch|background)$", re.IGNORECASE),
]


def humanize_label(label: Optional[str], fallback: str = "Working on it") -> str:
    """A model-authored work label, made safe for a lock screen.

    The label rides every job push as the card's HEADLINE (``mission_title``)
    and several banner titles. The model names it from the vocabulary its own
    prompts use — job, dispatch, background, sub-agent — which is meaningless
    or alarming on a user's phone. A label that talks about the machinery is
    replaced by ``fallback``; a label that names the errand passes untouched.
    """
    s = " ".join((label or "").split())
    if not s:
        return fallback
    for pat in _INTERNAL_LABEL_RES:
        if pat.search(s):
            return fallback
    return s


# The agent's opening ack is process narration, not an answer. A first line
# that ends in ':' is a list intro; a line matching these stems is a filler
# ("On it — two tasks running:" was the founder's entire push body).
_NARRATION_LINE_RE = re.compile(
    r"^(on it|got it|sure|okay|ok|right away|will do|working on|absolutely|of course|one (sec|moment)|let me|here('s| is) what)\b",
    re.IGNORECASE,
)


def answer_preview(
    text: Optional[str], limit: int,
    fallback: str = "Your answer is ready — tap to read it.",
) -> str:
    """The push body for a finished turn: the first CONTENT line of the
    answer — never the agent's own process narration, never a colon-hanging
    list intro.

    The LAST remaining line is dropped only when it hangs on a colon: a
    single-line answer that merely OPENS with a narration word ("Sure — your
    flight is booked for June 3.") carries the content in the same line, and
    discarding it would replace a real answer with the fallback.
    """
    plain = strip_markdown(text, collapse_whitespace=False)
    lines = [ln.strip() for ln in plain.split("\n") if ln.strip()]
    while len(lines) > 1 and (_NARRATION_LINE_RE.match(lines[0]) or lines[0].endswith(":")):
        lines.pop(0)
    if lines and len(lines) == 1 and lines[0].endswith(":"):
        lines = []
    body = " ".join(" ".join(lines).split())[:limit]
    return body or fallback
