"""What a tool says to the MODEL and what it says to the USER are different strings.

Until now they were the same one. `agent_runner` set `summary = result[:200]` —
the tool's own return value, written for the model — and `ws_chat` shipped it
verbatim as `tool_end.summary`, which the phone and the web both render inside
the expanded actions card. So expanding a finished image edit showed the user:

    Image edited and delivered to the user. Generated infinity-pool-position.png
    (2519998 bytes, image/png) at workspace path
    generated/871bac24-c366-42b5-b224-8802c73aef3a/282250e8…_infinity-pool-position.png.
    File will appear in the document pane; agent_runner will emit the attachment
    event after this tool call completes.

— a tenant UUID, the storage layout, an internal component name and the event
bus's semantics, rendered as product copy.

Two layers, deliberately, because either alone is insufficient:

1. `ToolResult` lets a tool state its own user-facing sentence (`display`). It
   subclasses `str` and IS the model-facing text, so every existing consumer —
   `len(result)`, `result[:200]`, `result.startswith("ERROR:")`, the transcript
   the model reads — keeps working untouched. Only code that asks for `.display`
   sees the difference.

2. `sanitize_for_client` runs at the boundary on EVERY summary, whether or not
   the tool has been converted. There are ~100 tools; a convention that depends
   on each one being updated is a convention that leaks the day someone adds
   the 101st. This is the guarantee, `display` is the quality.
"""

from __future__ import annotations

import json
import re
from typing import Optional


class ToolResult(str):
    """A tool's return value, carrying a separate user-facing summary.

    `ToolResult("...for the model...", display="Edited image")` is a `str`
    everywhere a plain return value was, so returning one from a tool is a
    no-op for the agent loop.
    """

    display: Optional[str]

    def __new__(cls, internal: str, display: Optional[str] = None) -> "ToolResult":
        obj = super().__new__(cls, internal)
        obj.display = display
        return obj


def display_of(result: object) -> Optional[str]:
    """The tool's own user-facing sentence, if it declared one."""
    d = getattr(result, "display", None)
    return d if isinstance(d, str) and d.strip() else None


# ── Redaction ────────────────────────────────────────────────────────────
# Ordered: the most specific shapes first, so a workspace path is collapsed to
# its filename before the bare-UUID rule can chew a hole in the middle of it.

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_HEX32 = r"[0-9a-fA-F]{32}"

# "at workspace path generated/<uuid>/<hex32>_name.png." — the whole clause.
_WORKSPACE_CLAUSE = re.compile(
    r"\s*(?:at|in|to|from)?\s*(?:workspace|storage)\s+(?:path|key|location)\s+\S+\.?",
    re.IGNORECASE,
)

# A bare storage path anywhere else: keep the human filename, drop the layout.
_STORAGE_PATH = re.compile(
    rf"(?:[\w./-]*/)?(?:{_UUID}|{_HEX32})(?:/[\w.-]*)*(?:/|_)?([\w.-]+\.[A-Za-z0-9]{{1,6}})"
)

# Absolute server paths.
_ABS_PATH = re.compile(r"(?:/(?:app|home|root|srv|var|tmp|workspace|data)(?:/[\w.\-]+)+)")

_BARE_UUID = re.compile(_UUID)
_BARE_HEX32 = re.compile(_HEX32)

# Internal component / module names. Word-boundary anchored so "the agent" and
# "a runner" survive; only the identifiers do not.
_INTERNAL_NAMES = re.compile(
    r"\b(?:agent_runner|tool_executor|doc_generators|media_resolve|ws_chat|"
    r"ws_realtime|agent_loop|file_storage|storage_backend|celery|alembic|"
    r"sqlalchemy|fastapi|uvicorn)\b",
    re.IGNORECASE,
)

# Sentences that only describe our own plumbing. Dropped whole — redacting the
# identifier inside them leaves "File will appear in the document pane; will
# emit the attachment event after this tool call completes."
_PLUMBING_SENTENCE = re.compile(
    r"[^.]*\b(?:will emit|event bus|after this tool call|tool call completes|"
    r"document pane|attachment event|internal(?:ly)?)\b[^.]*\.\s*",
    re.IGNORECASE,
)


def sanitize_for_client(text: Optional[str]) -> str:
    """Strip anything that describes how this system is built.

    Never raises and never returns None: this runs on the frame-emit path, and
    a redactor that can throw is a redactor that gets wrapped in a bare
    `except` and stops redacting.
    """
    if not text:
        return ""
    try:
        out = str(text)
        # A JSON payload is a MACHINE result, not prose, and the client parses
        # it structurally — `create_job` answers {"job_id": "<uuid>"} and
        # `activity.ts::jobIdsFromTools` reads that id to bind a job card to
        # the turn that made it. A job id is a client-facing identifier (it is
        # what `toup://chat?mission=` carries), unlike a tenant or storage
        # uuid, so stripping it would break the job card and protect nothing.
        # Prose is what leaks; give a JSON tool a `display` instead.
        stripped = out.strip()
        if stripped[:1] in "{[" and _is_json(stripped):
            return stripped
        out = _PLUMBING_SENTENCE.sub("", out)
        out = _WORKSPACE_CLAUSE.sub("", out)
        out = _STORAGE_PATH.sub(r"\1", out)
        out = _ABS_PATH.sub("", out)
        out = _BARE_UUID.sub("", out)
        out = _BARE_HEX32.sub("", out)
        out = _INTERNAL_NAMES.sub("", out)
        # Tidy the punctuation the removals left behind.
        out = re.sub(r"\(\s*[,;]?\s*\)", "", out)
        out = re.sub(r"\s+([,.;:])", r"\1", out)
        out = re.sub(r"([,;:])\s*([.;])", r"\2", out)
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\.{2,}", ".", out)
        return out.strip(" \t\n;,")
    except Exception:  # noqa: BLE001 - see docstring
        return ""


def client_summary(result: object, cap: int = 200) -> str:
    """The one function the emit path calls: the tool's own sentence when it
    has one, otherwise its redacted return value, capped."""
    chosen = display_of(result)
    if chosen is None:
        chosen = sanitize_for_client(str(result))
    else:
        # A declared display string is still redacted. A tool author writing
        # `display=f"Saved to {path}"` should not be able to reopen this hole.
        chosen = sanitize_for_client(chosen)
    if len(chosen) > cap:
        chosen = chosen[:cap].rstrip() + "…"
    return chosen


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:  # noqa: BLE001
        return False
