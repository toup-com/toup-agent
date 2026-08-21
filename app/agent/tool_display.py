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

Round 18 proved the second half of that claim by breaking it. The app builder
shipped without `display` on any of its five tools, and its results were
written to be read by a model, so a person who asked for a snake game was
shown, as the agent's own words: `app_html__bash_app`, the route
`/api/artifacts/nokia-snake`, the directive `[[open_app:nokia-snake]]`, and the
sentence "Tell the user what it does in one or two sentences". Every one of
those is a shape the redactor had no rule for — it knew about storage paths and
UUIDs, which is what had leaked the time before.

So this round adds `display` to those five tools (the quality layer) AND
teaches the redactor the three shapes it did not know (the guarantee layer):
directive tokens, `<skill>__<tool>` identifiers, and the YAML frontmatter that
opens a skill file.
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

# `[[open_app:nokia-snake]]`, `[[navigate:/files]]`, `[[Try again]]`. A chip
# directive is a token the CHAT renderer understands; a tool summary is
# rendered as plain text, so here it is only ever the literal brackets on
# screen. Bounded length so a stray `[[` cannot eat the rest of the string.
_CHIP_DIRECTIVE = re.compile(r"\[\[[^\[\]\n]{0,120}\]\]")

# `app_html__bash_app`, `routines__create`, `gmail__list_messages`. The wire
# name of a tool, which is an identifier by construction — a user cannot act
# on it and it is exactly the thing round 16 established must not be printed.
_TOOL_IDENTIFIER = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*__[a-z0-9_]+\b")

# The head of a skill file: `---\nname: …\ndescription: …\n---`. Reading a
# skill's markdown yields its frontmatter as the first thing in the result, so
# this is the first thing a dump of one shows.
_FRONTMATTER = re.compile(r"\A\s*---\s*\n.*?\n---\s*\n", re.DOTALL)

# Stage direction. A tool result that tells the model how to talk to the user
# reads, when shown to the user, as the agent being coached mid-sentence.
_STAGE_DIRECTION = re.compile(
    r"[^.\n]*\b(?:tell the user|do not paste|say to the user|"
    r"offer the \[\[|inform the user)\b[^.\n]*\.?\s*",
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
        out = _FRONTMATTER.sub("", out)
        out = _STAGE_DIRECTION.sub("", out)
        out = _PLUMBING_SENTENCE.sub("", out)
        out = _CHIP_DIRECTIVE.sub("", out)
        out = _TOOL_IDENTIFIER.sub("", out)
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


# ── Live status line ─────────────────────────────────────────────────────
# What a client shows WHILE a call is in flight. It has no result to describe
# yet, only a name, so without a label it can do nothing but humanise the
# identifier — and `app_html__create_app_file` humanises to "App html", which
# then sat on a lock screen for nine seconds and became "App html — still
# going". The tool knows what it is doing; it should say so.
_LIVE_LABELS = {
    "app_html__create_app_file": "Building your app",
    "app_html__view_app_file": "Reading your app",
    "app_html__edit_app_file": "Updating your app",
    "app_html__bash_app": "Checking your app",
    "app_html__present_app": "Finishing your app",
}


def public_label(tool_name: str) -> Optional[str]:
    """A human status line for a tool, or None when we have none.

    Deliberately a lookup and not a derivation: an absent label leaves the
    client on its own existing fallback, which is a known quantity. A DERIVED
    label is how an identifier reaches a screen, and this function exists
    because that happened.
    """
    return _LIVE_LABELS.get(tool_name or "")
