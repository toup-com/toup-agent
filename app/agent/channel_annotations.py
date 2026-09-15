"""The history annotation tag, and its inverse.

``day_context_loader.annotate_message`` labels every history row the model
reads with ``[<channel> <h:mm><am|pm>] `` so the model can tell a WhatsApp
line from a mobile one. Nothing in the prompt ever said those labels are
the SYSTEM's and not part of anyone's words, and on 2026-09-14 the model
began a reply with ``[mobile 9:41pm]`` — persisted verbatim, shown to the
user, and then read back as history, which taught it the pattern again.

This module is the ONE definition of that tag shape and the only place it
is matched. Four consumers share it: the streaming filter and the final
text sanitizer in ``agent_runner``, the history loader, and ``public_text``
in the client-facing serializers (plus the ops backfill).

The regex is deliberately strict — anchored at the start, the channel must
be a real ``KNOWN_CHANNELS`` member, and the clock is MANDATORY. That is
what keeps a markdown link head (``[label](url)``), a list marker (``[1]``)
and a user's own prose (``[note 2:00pm] ...``) intact. The clockless
``[web]`` shape that ``annotate_message`` emits when ``created_at`` is None
is deliberately NOT stripped: the false-positive risk on ordinary prose
outweighs it, so ``LOOSE_TAG_RE`` counts those instead of removing them.
"""

from __future__ import annotations

import inspect
import re
from typing import Callable, Optional, Tuple

from app.agent.channel_util import KNOWN_CHANNELS

# `unknown` is `resolve_channel`'s own default and therefore a tag the
# history loader can EMIT (`[unknown 9:41pm] …`) — it has to be strippable
# even though it is not a channel anyone can send from.
STRIPPABLE_CHANNELS = frozenset(KNOWN_CHANNELS) | {"unknown"}

# Longest-first so a channel that prefixes another ("web" vs a future
# "webhook") cannot shadow it inside the alternation.
KNOWN_CHANNEL_ALT = "|".join(
    re.escape(c) for c in sorted(STRIPPABLE_CHANNELS, key=len, reverse=True)
)

# STRICT — what we actually remove. One or more consecutive tags at the very
# start, tolerant of leading/intervening whitespace (including newlines), of
# a space before am/pm, and of case.
LEAKED_TAG_RE = re.compile(
    r"\A\s*(?:\[(?:" + KNOWN_CHANNEL_ALT + r")[ \t]+\d{1,2}:\d{2}(?:[ \t]?(?:am|pm))?\]\s*)+",
    re.IGNORECASE,
)

# LOOSE — clock optional. COUNT ONLY, never used to strip: it would eat
# "[api] returns 404" out of a perfectly good answer. Its job is to make the
# "how often does the clockless shape leak?" question answerable with a
# number instead of an argument.
LOOSE_TAG_RE = re.compile(
    r"\A\s*(?:\[(?:" + KNOWN_CHANNEL_ALT + r")(?:[ \t]+\d{1,2}:\d{2}[ \t]?(?:am|pm))?\]\s*)+",
    re.IGNORECASE,
)

# How many characters the stream filter may hold back while it decides
# whether a reply opens with a tag. Longer than the longest possible tag
# run we care about, short enough that a held prefix is never perceptible.
MAX_TAG_HOLD = 64


def strip_leaked_tags(text: str) -> Tuple[str, str]:
    """Return ``(clean, stripped_prefix)``.

    ``stripped_prefix`` is the removed text byte-for-byte, so the ops
    backfill can restore it. Never raises — every caller is on a path where
    an exception would cost the user their reply.
    """
    if not text:
        return "", ""
    try:
        m = LEAKED_TAG_RE.match(text)
        if not m:
            return text, ""
        return text[m.end():], m.group(0)
    except Exception:  # noqa: BLE001 — output hygiene must never break a turn
        return text, ""


def make_stream_tag_filter(on_text_chunk: Optional[Callable]) -> Optional[Callable]:
    """Wrap ``on_text_chunk`` so a leaked tag never reaches the live bubble.

    Armed ONCE per turn, not per text block: the final-text sanitizer only
    strips the leading tag of the WHOLE reply, so per-block arming would make
    the stream and the persisted text disagree.

    Fast path (the overwhelming majority of turns): the first non-whitespace
    character is not ``[`` → forward verbatim and disarm forever, one
    downstream call, no buffering. Otherwise hold at most ``MAX_TAG_HOLD``
    characters, until a ``]`` or a newline settles it.

    The returned callable carries ``.flush()`` — the turn teardown MUST call
    it, or a reply whose entire text is ``[`` is swallowed.
    """
    if on_text_chunk is None:
        return None

    state = {"armed": True, "buf": "", "trim": False}

    async def _send(text: str) -> None:
        # Every production sink is a coroutine function, but a plain callable
        # (a test recorder, a future sync consumer) must not turn into
        # `await None`.
        r = on_text_chunk(text)
        if inspect.isawaitable(r):
            await r

    async def _flush() -> None:
        if not state["armed"]:
            return
        state["armed"] = False
        buf = state["buf"]
        state["buf"] = ""
        if not buf:
            return
        clean, stripped = strip_leaked_tags(buf)
        if clean:
            await _send(clean)
        elif stripped:
            # The tag ended exactly at a chunk boundary: the whitespace that
            # follows it arrives in the NEXT chunk and belongs to the tag, not
            # to the reply (the final sanitizer's regex eats it in one piece).
            state["trim"] = True

    async def _emit(chunk: str) -> None:
        if not state["armed"]:
            if state["trim"]:
                chunk = (chunk or "").lstrip()
                if not chunk:
                    return
                state["trim"] = False
            await _send(chunk)
            return
        buf = state["buf"] + (chunk or "")
        head = buf.lstrip()
        if head and head[0] != "[":
            # Cannot be a tag. Disarm and forward everything held so far in
            # one call — holding whitespace and then emitting it separately
            # would split the first word across two chunks on the wire.
            state["armed"] = False
            state["buf"] = ""
            await _send(buf)
            return
        state["buf"] = buf
        if len(buf) >= MAX_TAG_HOLD or "]" in buf or "\n" in buf:
            await _flush()

    _emit.flush = _flush  # type: ignore[attr-defined]
    return _emit
