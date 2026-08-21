"""The title on a job card — one shape, whichever path produced it.

A chat turn titles its job through the model: ``create_job(title="Find the
strongest image-generation model")``. Voice cannot — ``create_job`` is removed
from the voice loadout on purpose (``prompt_profile.VOICE_DISABLED_TOOLS``,
and the founder's 2026-08-01 session is why) — so the RUNNER titles a voice
job from the request the user actually made (``voice_jobs.py``).

Two producers is fine. Two *normalisers* is not: it is the same card on the
same screen, and a voice job that kept its trailing full stop, its "can you
please", or its 300 spoken characters would read as a different product than
the chat job beside it. Everything that becomes a job title goes through
:func:`normalize_job_title`; a raw utterance goes through
:func:`derive_job_title` first, which is that same normaliser with the spoken
scaffolding taken off in front of it.

Pure, dependency-free, never raises, well under a millisecond — it runs on the
turn's own thread while the user is holding a live audio session open.
"""
from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "JOB_TITLE_MAX",
    "derive_job_title",
    "is_rtl_text",
    "normalize_job_title",
]

#: Card budget, shared by both paths. Two comfortable lines on a phone. The
#: model writes titles far shorter than this, so in practice only a spoken
#: sentence is ever clipped — which is exactly the input that needs clipping.
JOB_TITLE_MAX = 80

#: Below this a "title" is noise ("ok", "yes"), not a job name.
_MIN_TITLE_CHARS = 3

#: Zero-width non-joiner — a real Persian word boundary, and the one the app's
#: own clamp breaks on (`VoiceModeOverlay.clampWords`).
_ZWNJ = "‌"

#: Whitespace that is not whitespace to ``str.split``: NBSP, narrow NBSP, and
#: the zero-width space a transcript can carry. Folded to a plain space first
#: so every rule below sees one uniform separator.
_ODD_SPACE_RE = re.compile("[   ​﻿]")

# Quote-ish characters a title can arrive wrapped in — ASCII, typographic, and
# the guillemets Persian text uses. Peeled only in matched pairs.
_QUOTE_PAIRS = (
    ('"', '"'), ("'", "'"), ("`", "`"),
    ("“", "”"), ("‘", "’"),
    ("«", "»"),
)

#: Trailing punctuation a card never wants, including the Arabic comma and
#: semicolon. "?" is deliberately kept — a job that IS a question reads
#: correctly as one ("Which model is strongest?").
_TRAILING_PUNCT = ".,;:!…،؛"

_WS_RE = re.compile(r"\s+")

#: Markdown scaffolding a model occasionally wraps a title in. Stripped, not
#: rendered: every client prints a job title as plain text.
_MD_RE = re.compile(r"[*_`#]+")

# ── Spoken scaffolding ────────────────────────────────────────────────────
# A voice `task` string is a REQUEST, not a title: "hey, can you please find
# the strongest image-generation model for me". These take the framing off so
# the verb leads, which is what makes a derived title read like a written one.
#
# Applied repeatedly (a real utterance stacks them: "ok so can you please …"),
# and only ever from the FRONT — nothing here may touch the subject matter.
_EN_LEAD_RES = (
    re.compile(r"^(?:ok(?:ay)?|so|well|um+|uh+|hey|hi|hello|yo)\b[\s,.]*", re.I),
    re.compile(r"^(?:can|could|would|will)\s+you\s*(?:please|pls)?\b[\s,]*", re.I),
    re.compile(r"^(?:please|pls)\b[\s,]*", re.I),
    re.compile(r"^i\s*(?:'d|\s+would)\s+like\s+(?:you\s+to|to)\b\s*", re.I),
    re.compile(r"^i\s+(?:want|need)\s+you\s+to\b\s*", re.I),
    re.compile(r"^(?:let's|lets)\b\s*", re.I),
    re.compile(r"^(?:help\s+me|go\s+ahead\s+and|try\s+to)\b\s*", re.I),
)

# The Persian half. Deliberately tiny, and only the unambiguous politeness
# openers — the founder is Persian-primary and a wrong strip takes MEANING off
# the card, which is worse than leaving a filler word on it. "ببین" (look /
# check) and "بگرد" (search) are VERBS and are never touched.
#
# ``\b`` is not used: Python's word boundary is defined on ASCII-ish word
# characters and does not behave on Arabic script, so each opener is anchored
# on the separator that must follow it instead.
_FA_LEAD_RES = (
    re.compile("^(?:لطفاً?|خواهشاً?)(?=$|[\\s،,])[\\s،,]*"),
    re.compile("^می" + _ZWNJ + "?(?:شه|تونی)(?=$|[\\s،,])[\\s،,]*"),
    re.compile("^بی" + _ZWNJ + "?زحمت(?=$|[\\s،,])[\\s،,]*"),
)

_EN_TRAIL_RE = re.compile(r"[\s,]*(?:please|pls|thanks|thank\s+you)\s*$", re.I)
_FA_TRAIL_RE = re.compile(
    "[\\s،,]*(?:لطفاً?|ممنون)\\s*$"
)

#: An utterance that is only an acknowledgement is not a job name, however
#: many characters it happens to have ("yes" clears any length threshold).
#: Checked in :func:`derive_job_title` ONLY — the ``create_job`` tool's own
#: contract is "any non-empty title is legal", and a deliberately terse
#: model-written title must stay legal there.
_ACK_ONLY = frozenset({
    "yes", "yeah", "yep", "yup", "no", "nope", "ok", "okay", "sure",
    "thanks", "thank you", "cool", "nice", "great", "done", "go", "go ahead",
    "بله", "آره", "نه", "باشه", "ممنون", "مرسی", "خوبه", "اوکی",
})

#: Right-to-left scripts: Hebrew (U+0590–U+05FF), Arabic and its supplements
#: (U+0600–U+06FF, U+0750–U+077F, U+08A0–U+08FF), and the presentation forms
#: (U+FB1D–U+FDFF, U+FE70–U+FEFF). Used only to pick a rule set — never to
#: reorder anything.
_RTL_RE = re.compile(
    "[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
    "\ufb1d-\ufdff\ufe70-\ufefc]"
)


def is_rtl_text(text: Optional[str]) -> bool:
    """True when ``text`` carries right-to-left script (Persian, Arabic,
    Hebrew)."""
    return bool(text) and bool(_RTL_RE.search(text))


def _fold(raw: str) -> str:
    """Markdown off, odd spaces folded, whitespace collapsed, quotes peeled."""
    text = _MD_RE.sub("", _ODD_SPACE_RE.sub(" ", raw))
    text = _WS_RE.sub(" ", text).strip()
    for _ in range(2):
        for open_q, close_q in _QUOTE_PAIRS:
            if len(text) > 1 and text.startswith(open_q) and text.endswith(close_q):
                text = text[len(open_q):-len(close_q)].strip()
                break
        else:
            break
    return text


def _clip(text: str, limit: int) -> str:
    """Clip to ``limit`` on a word boundary, with an ellipsis.

    The boundary search only accepts a break in the back of the budget — a
    single very long token would otherwise clip to two characters plus an
    ellipsis. ZWNJ counts as a boundary, the same rule the app's own clamp
    uses, because RN will not break a Persian word on it.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit]
    brk = max(cut.rfind(" "), cut.rfind(_ZWNJ), cut.rfind("-"))
    if brk > limit * 0.6:
        cut = cut[:brk]
    return cut.rstrip(_TRAILING_PUNCT + " " + _ZWNJ + "-") + "…"


def normalize_job_title(raw: Optional[str], *, limit: int = JOB_TITLE_MAX) -> str:
    """The card shape: one line, no wrapping quotes, no markdown, no trailing
    full stop, bounded length. Returns ``""`` for anything that is not a title.

    Idempotent — normalising an already-normalised title is a no-op, which is
    what lets both paths call it without knowing whether the other already did.
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = _fold(raw).rstrip(_TRAILING_PUNCT + " ").strip()
    if len(text) < _MIN_TITLE_CHARS:
        return ""
    return _clip(text, max(limit, _MIN_TITLE_CHARS + 1))


def derive_job_title(
    request: Optional[str], *, fallback: str = "", limit: int = JOB_TITLE_MAX,
) -> str:
    """Title a job from the REQUEST that opened it — the voice path's input.

    Takes the spoken scaffolding off the front and back ("hey, can you please
    … thanks") so the verb leads, then hands the rest to
    :func:`normalize_job_title`. Returns ``fallback`` (itself normalised) when
    nothing usable survives — an "ok" is not a job.
    """
    text = ""
    if isinstance(request, str) and request:
        text = _fold(request)
        rtl = is_rtl_text(text)
        # An utterance stacks openers ("ok so can you please …"). Peel until
        # nothing matches, bounded so no input can spin here.
        for _ in range(4):
            before = text
            for rx in (_FA_LEAD_RES if rtl else _EN_LEAD_RES):
                text = rx.sub("", text, count=1)
            text = text.strip()
            if text == before:
                break
        text = (_FA_TRAIL_RE if rtl else _EN_TRAIL_RE).sub("", text).strip()
        if text.strip(_TRAILING_PUNCT + " ?").lower() in _ACK_ONLY:
            text = ""
        if text and not rtl:
            # Sentence case for Latin only: a lowercased utterance becomes a
            # card title. Never `.title()` — that mangles acronyms and product
            # names ("GPT-5.6" -> "Gpt-5.6").
            text = text[0].upper() + text[1:]
    return normalize_job_title(text, limit=limit) or normalize_job_title(
        fallback, limit=limit
    )
