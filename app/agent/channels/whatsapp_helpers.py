"""WhatsApp-specific outbound helpers — pure functions only.

This module is intentionally dependency-free so it can be imported and
unit-tested without spinning up the agent runtime.

Three concerns live here:

* :func:`markdown_to_whatsapp` — converts the agent's GitHub-Flavored
  Markdown into WhatsApp's bold/italic/strike syntax. WhatsApp uses
  single-character markers (``*bold*``, ``_italic_``, ``~strike~``) and
  silently drops the GFM doubles (``**bold**`` displays as ``**bold**``
  literally). Code blocks must be preserved verbatim — a ``**`` inside
  a fenced block is not a marker.

* :func:`chunk_for_whatsapp` — splits a reply into ≤ ``limit`` chunks
  along the cleanest available break (paragraph → sentence → word →
  hard cut). WhatsApp's per-message text limit is 4096 characters; the
  current outbound path truncates at that limit, silently dropping
  anything longer.

* :func:`retry_status_codes` — the small set of HTTP status codes that
  warrant an automatic retry on outbound calls to the WhatsApp Cloud
  API. The actual retry loop lives in the channel adapter (see
  :py:meth:`WhatsAppChannel.send_text`); this constant just centralises
  the policy so other senders pick up the same behaviour.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# WhatsApp's per-message text cap. Exposed as a module constant so
# callers and tests can reference it without magic numbers.
WHATSAPP_TEXT_LIMIT = 4096

# Status codes worth a retry. 429 = rate-limit, 5xx = transient Meta
# failure. 4xx (other than 429) means the request is bad — no retry.
RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Sentinels used to lift fenced code and inline code out of the text
# before transforming markdown. They use NULs so they cannot collide
# with anything a user (or the agent) could legitimately type.
_FENCE_SENTINEL = "\x00FENCE\x00"
_INLINE_SENTINEL = "\x00INLINE\x00"

# Capture fenced code blocks, including the optional language tag.
_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# Capture inline code spans. WhatsApp's syntax for inline code is the
# same single backtick, so we keep the backticks in the output.
_INLINE_RE = re.compile(r"`[^`\n]+`")

# **bold** and __bold__ collapse to *bold*. Greedy ``.+?`` so they don't
# slurp trailing markers from neighbouring spans.
_BOLD_DOUBLE_STAR_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_DOUBLE_UNDER_RE = re.compile(r"__(.+?)__", re.DOTALL)

# ~~strike~~ collapses to ~strike~.
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)


def markdown_to_whatsapp(text: str) -> str:
    """Convert GFM-ish markdown into WhatsApp formatting.

    The transform is intentionally minimal — only constructs that
    differ between the two dialects are touched:

    +----------------+--------------------+
    | GFM source     | WhatsApp output    |
    +================+====================+
    | ``**bold**``   | ``*bold*``         |
    | ``__bold__``   | ``*bold*``         |
    | ``~~strike~~`` | ``~strike~``       |
    +----------------+--------------------+

    Anything inside fenced or inline code is preserved literally so
    that source code containing ``**`` or ``~~`` does not get rewritten
    by accident.

    GFM italics (``*italic*`` / ``_italic_``) are already valid
    WhatsApp italics — we leave them alone.
    """
    if not text:
        return text

    fences: List[str] = []
    inlines: List[str] = []

    def _take_fence(match: "re.Match[str]") -> str:
        fences.append(match.group(0))
        return _FENCE_SENTINEL

    def _take_inline(match: "re.Match[str]") -> str:
        inlines.append(match.group(0))
        return _INLINE_SENTINEL

    # Order matters: fences first (which may contain backticks), then
    # surviving inline spans.
    stripped = _FENCE_RE.sub(_take_fence, text)
    stripped = _INLINE_RE.sub(_take_inline, stripped)

    stripped = _BOLD_DOUBLE_STAR_RE.sub(r"*\1*", stripped)
    stripped = _BOLD_DOUBLE_UNDER_RE.sub(r"*\1*", stripped)
    stripped = _STRIKE_RE.sub(r"~\1~", stripped)

    # Restore the protected spans in original order.
    if fences:
        for fence in fences:
            stripped = stripped.replace(_FENCE_SENTINEL, fence, 1)
    if inlines:
        for inline in inlines:
            stripped = stripped.replace(_INLINE_SENTINEL, inline, 1)

    return stripped


def chunk_for_whatsapp(text: str, limit: int = WHATSAPP_TEXT_LIMIT) -> List[str]:
    """Split *text* into chunks of at most *limit* characters.

    The splitter walks from coarse to fine break candidates so the seam
    lands at a natural boundary whenever one is available within the
    last ``limit`` characters of the prefix:

    1. **Paragraph break** (``\\n\\n``).
    2. **Newline** (``\\n``).
    3. **Sentence end** (``. ``, ``! ``, ``? ``).
    4. **Whitespace** (any).
    5. **Hard cut** at exactly ``limit`` if no break is found — last
       resort, only triggers for solid blobs of non-whitespace.

    Empty / whitespace-only input returns ``[]`` so callers can iterate
    blindly without sending stray blank messages.
    """
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = _find_break(window)
        chunk = remaining[:cut].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _find_break(window: str) -> int:
    """Find the best split index inside *window*.

    Returns an index in ``[1, len(window)]``. Never zero — that would
    make :func:`chunk_for_whatsapp` loop forever on a chunk that starts
    with whitespace.
    """
    # 1. Paragraph break — strongest signal.
    idx = window.rfind("\n\n")
    if idx > 0:
        return idx + 2  # consume the blank line

    # 2. Single newline.
    idx = window.rfind("\n")
    if idx > 0:
        return idx + 1

    # 3. Sentence end. Search each terminator and take the rightmost.
    best = -1
    for terminator in (". ", "! ", "? "):
        candidate = window.rfind(terminator)
        if candidate > best:
            best = candidate
    if best > 0:
        return best + 2  # past the trailing space

    # 4. Any whitespace — falls back to mid-line word boundary.
    idx = max(window.rfind(" "), window.rfind("\t"))
    if idx > 0:
        return idx + 1

    # 5. Hard cut. Solid blob with no break candidate.
    return len(window)


def retry_status_codes() -> Iterable[int]:
    """Status codes warranting an outbound retry. Module-level constant
    re-exposed via a function so call sites read naturally:

        if response.status_code in retry_status_codes(): ...
    """
    return RETRY_STATUS_CODES


def redact_phone(phone: str) -> str:
    """Mask all but the leading and trailing two digits of an E.164.

    >>> redact_phone("+14155552671")
    '+1***71'
    >>> redact_phone("")
    ''
    """
    if not phone:
        return phone
    if len(phone) <= 4:
        return "***"
    return f"{phone[:2]}***{phone[-2:]}"
