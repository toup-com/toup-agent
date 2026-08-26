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

# R30 (D-03): emoji are machinery grammar on these surfaces. "Overall: ✅ OK"
# and "⚠ That step didn't finish." reached a founder's job sheet verbatim —
# a tool result written for a terminal (the doctor report) persisted as a
# ToolPillRow summary. A served string states its status in words; the glyph
# is the terminal's dialect, not the product's. Ranges cover the pictograph
# planes plus the two symbol blocks the recordings actually showed (⚠ ✅ ❌
# live in 2600–27BF; ⏰ ⏹ in the media-control run of Misc Technical), and
# the joiners/selectors that ride them. Arrows and box-drawing stay — "old →
# new" is prose.
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictographs, emoticons, transport, supplemental
    "\u2600-\u27BF"          # misc symbols + dingbats (warning, check, cross, sparkle)
    "\u2B00-\u2BFF"          # misc symbols and arrows (up-arrow, star)
    "\u23E9-\u23FA"          # media controls + clocks (alarm, hourglass, stop)
    "\u2139"                  # information source
    "\uFE0F\u200D\u20E3"    # variation selector, ZWJ, keycap combiner
    "]+"
)


def strip_emoji(text: Optional[str]) -> str:
    """``text`` with emoji removed and the spacing closed back up.

    For strings a MACHINE authored that a client will render (step labels,
    summaries, card titles) — never for the model's own prose to the user.
    Never raises; returns ``""`` for None/empty.
    """
    if not text:
        return ""
    try:
        s = str(text)
        out = _EMOJI.sub("", s)
        if out == s:
            # Untouched text passes through byte-identical — the spacing
            # tidy-up below is only for the holes a removal leaves.
            return s
        out = re.sub(r" {2,}", " ", out)
        return out.strip()
    except Exception:  # noqa: BLE001 — a stripper must never break a serve
        return str(text)


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
        # R30 (D-03): status glyphs out of every served summary. The doctor
        # report's "Overall: ✅ OK" is the proven leak; the rule is general
        # because the next terminal-flavoured tool result will pick a
        # different glyph.
        out = _EMOJI.sub("", out)
        # Tidy the punctuation the removals left behind.
        out = re.sub(r"\(\s*[,;]?\s*\)", "", out)
        out = re.sub(r"\s+([,.;:])", r"\1", out)
        out = re.sub(r"([,;:])\s*([.;])", r"\2", out)
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\.{2,}", ".", out)
        return out.strip(" \t\n;,")
    except Exception:  # noqa: BLE001 - see docstring
        return ""


def client_summary(result: object, cap: int = 200,
                   tool_name: Optional[str] = None,
                   is_skill_tool: bool = False) -> str:
    """The one function the emit path calls: the tool's own sentence when it
    has one, otherwise its redacted return value, capped.

    ``tool_name`` (R30, D-17): the JSON pass-through above exists for
    first-party tools whose payload the client parses structurally
    (`create_job`'s job_id binds the card to its turn). A CONNECTOR tool's
    JSON is a different animal — a vendor API response the client renders as
    text, which is how ``{"site": "Toup", "is_last": true}`` became the
    detail line "Site: Toup · Is last: true" on a founder's job sheet. When
    the caller names a connector tool (``__`` in the wire name) and the tool
    declared no ``display`` sentence, a JSON result serves as EMPTY — the
    step's label carries the row, and the model still reads the full payload.

    ``is_skill_tool``: "connector tool" was originally TESTED as ``"__" in
    tool_name``, and SKILL tools are prefixed too — so a rule written for
    vendor payloads silently swallowed every skill tool returning JSON
    without a declared ``display``: `routines__remind`, `routines__create`,
    `routines__list`, `triggers__list`. Not a cosmetic loss: the reminder
    CARD is built client-side from this exact string
    (`reminders.ts::remindersFromTools` parses
    `{"status":"created","reminder":{…}}` out of it), and `routines__remind`
    sits in the client's HIDDEN_TOOLS *because* the card is meant to BE its
    rendering. Empty summary ⇒ no chip and no card, so a reminder the user
    had just created rendered as a bare text bubble. `app_html__*` survived
    only because it declares a `display` sentence and never reaches here.

    A skill tool is first-party by construction — its payload is a shape
    this repo defines, not a vendor's — so it keeps the JSON pass-through.
    """
BUILTIN_SKILL_PREFIXES: frozenset = frozenset({
    "app_builder", "app_html", "automations", "routines", "toup",
    "triggers",
})


def is_first_party_tool(tool_name: str, is_skill_tool: bool = False) -> bool:
    """Is this a SKILL tool (first-party) rather than a connector's?

    The distinction decides whether a JSON tool result is served or
    blanked, and getting it wrong is not cosmetic: the reminder CARD is
    built client-side out of `routines__remind`'s JSON, so a blanked
    summary is a reminder that renders as a bare text bubble.

    Two sources, deliberately. `is_skill_tool` is the loader's live
    registration index and is authoritative WHEN THERE IS A LOADER —
    the write path always has one. The history READ path does not: it
    is a serializer, it may run in a request that never built a runner,
    and a lookup that answers "no" because nothing was registered blanks
    every skill tool in the user's history. So the prefix set is the
    floor. It is a static list of the builtin skill directories, which
    is a fact about this repo's layout rather than about runtime state.

    Verified live on 2026-08-26 (R31-C, confirmed here): the read path
    in `day_chats.py::_with_public_copy` carried its own copy of the old
    `"__" in tool` predicate with no `is_skill_tool` at all, wrapped in
    a bare except — so a fix applied only at the write path left history
    broken, and a fix applied there would have failed silently.
    """
    if is_skill_tool:
        return True
    name = str(tool_name or "")
    if "__" not in name:
        return False
    return name.split("__", 1)[0] in BUILTIN_SKILL_PREFIXES


    chosen = display_of(result)
    if chosen is None:
        chosen = sanitize_for_client(str(result))
        if tool_name and "__" in tool_name and not is_first_party_tool(
                tool_name, is_skill_tool):
            head = (chosen or "").strip()[:1]
            if head in "{[":
                return ""
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


# ── Total step labels (R30, D-01/D-17) ───────────────────────────────────
# `public_label` is Optional by design: a live surface with its own known
# fallback may prefer it. A PERSISTED surface has no such luxury — a
# `tool_events` record served without a label leaves every client to
# humanise the wire id, and the founder's job sheet read "List events",
# "Search issues", "List repos": raw connector ids with the underscores
# swapped out. `public_step_label` is the total form: every input yields a
# human sentence, never the identifier, never a bare verb.
#
# Same discipline as `automation_verbs.step_verb` (the R29 dictionary the
# automation ledger passes): a lookup first, then a closed connector rule,
# then the honest generic.

#: The agent's own (non-connector) work. One vocabulary with the live
#: Dynamic-Island subtitles (`turn_progress._TOOL_SUBTITLES`), minus the
#: ellipsis — these label settled rows, not in-flight ones.
_STEP_LABELS = {
    "web_search": "Searching the web",
    "extension_search": "Searching the web",
    "extension_research": "Searching the web",
    "web_fetch": "Reading a page",
    "extension_read": "Reading a page",
    "smart_fetch": "Reading a page",
    "exec": "Running a command",
    "pty_exec": "Running a command",
    "process": "Running a command",
    "generate_image": "Creating an image",
    "edit_image": "Editing an image",
    "analyze_image": "Looking at an image",
    "canvas": "Creating an image",
    "write_file": "Writing a file",
    "edit_file": "Editing a file",
    "apply_patch": "Editing a file",
    "read_file": "Reading a file",
    "list_files": "Looking at files",
    "ls": "Looking at files",
    "grep": "Searching files",
    "find": "Searching files",
    "generate_pdf": "Building a document",
    "generate_docx": "Building a document",
    "generate_xlsx": "Building a document",
    "generate_pptx": "Building a document",
    "generate_markdown": "Building a document",
    "generate_html_to_pdf": "Building a document",
    "convert_document": "Building a document",
    "memory_store": "Saving notes",
    "memory_delete": "Updating notes",
    "memory_search": "Checking notes",
    "recall_day": "Checking notes",
    "sessions_list": "Reviewing conversations",
    "sessions_history": "Reviewing conversations",
    "session_status": "Reviewing conversations",
    "thread": "Reviewing conversations",
    "doctor": "Running a health check",
    "spawn": "Starting a helper",
    "start_mission": "Starting a mission",
    "create_job": "Planning the work",
    "update_job": "Tracking progress",
    "send_file": "Sending a file",
    "send_photo": "Sending a photo",
}

#: The recorded defects, pinned by name: the exact tools the founder's job
#: sheet showed as raw ids get a first-class sentence rather than the
#: connector rule's generic.
_CONNECTOR_STEP_LABELS = {
    "calendar__list_events": "Checking your calendar",
    "gcal__list_events": "Checking your calendar",
    "teams__list_chats": "Checking Teams",
    "teams__read_chat_messages": "Checking Teams",
    "jira__search_issues": "Searching Jira",
    "github__list_repos": "Checking GitHub",
    "github__list_issues": "Checking GitHub",
    "gmail__list_messages": "Checking Gmail",
    "gmail__search_threads": "Searching Gmail",
    "outlook__list_messages": "Checking Outlook",
    "drive__list_files": "Checking Drive",
    "notion__search": "Searching Notion",
    "slack__send_message": "Posting to Slack",
    "teams__send_chat_message": "Posting to Teams",
}

#: Brand casing for a connector prefix. Mirrors `automation_verbs.
#: _CONNECTOR_NAMES` (which the platform image owns) — kept local because
#: this module ships in the agent image and must import with no siblings.
_CONNECTOR_BRANDS = {
    "gmail": "Gmail", "outlook": "Outlook", "jira": "Jira",
    "github": "GitHub", "slack": "Slack", "teams": "Teams",
    "notion": "Notion", "drive": "Drive", "gdrive": "Drive",
    "docs": "Docs", "sheets": "Sheets", "slides": "Slides",
    "calendar": "Calendar", "gcal": "Calendar", "linkedin": "LinkedIn",
    "figma": "Figma", "linear": "Linear", "stripe": "Stripe",
}

#: Connector actions that only read. Anything else labels as work IN the
#: connector, which is the safe direction — "Checking Slack" beside a message
#: that was sent is the bigger lie.
_CONNECTOR_READ_VERBS = (
    "list", "get", "search", "read", "find", "fetch", "query", "lookup",
    "describe", "download", "export", "check",
)

#: `<skill>__<tool>` prefixes that are OUR machinery, not a vendor brand.
#: The connector rule must not title-case these into "App Html" — the exact
#: humanised-identifier failure this module exists to end.
_INTERNAL_TOOL_PREFIXES = frozenset({
    "app_html", "app", "routines", "triggers", "memory", "session",
})


def public_step_label(tool_name: str) -> str:
    """A human label for a settled tool row. TOTAL: any input — an unmapped
    connector action, a tool literally named ``list`` or ``create``, an empty
    string — yields a sentence a person can read, never the wire id and never
    a bare verb. Dictionary first; the connector rule is the only derivation
    and it derives a BRAND (a single-word vendor prefix, title-cased — the
    `voice_jobs._brand` precedent), never the action's words.
    """
    name = str(tool_name or "")
    lbl = _LIVE_LABELS.get(name) or _CONNECTOR_STEP_LABELS.get(name) \
        or _STEP_LABELS.get(name)
    if lbl:
        return lbl
    if name.startswith("browser"):
        return "Browsing"
    if "__" in name:
        prefix, _, action = name.partition("__")
        p = prefix.lower()
        brand = _CONNECTOR_BRANDS.get(p)
        if brand is None and p not in _INTERNAL_TOOL_PREFIXES \
                and re.fullmatch(r"[a-z][a-z0-9]{1,23}", p):
            brand = p.title()
        if not brand:
            return "Working"
        if any(action.lower().startswith(v) for v in _CONNECTOR_READ_VERBS):
            return f"Checking {brand}"
        return f"Updating {brand}"
    return "Working"
