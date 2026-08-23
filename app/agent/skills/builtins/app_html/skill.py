"""AppHtmlSkill — build an app by writing ONE self-contained HTML file.

Five tools, one file, no build step::

    create_app_file(slug, title, html)          "Creating file: {title}"
    view_app_file(slug)                         "Reading current app before editing"
    edit_app_file(slug, old_string, new_string) "Editing file: {reason}"
    bash_app(slug, command)                     "Verifying changes"
    present_app(slug)                           "Presented app"

The loader enforces a ``<skill>__`` prefix on every tool name
(``SkillLoader._register``), so on the wire these are
``app_html__create_app_file`` … ``app_html__present_app``.

Steps land in the existing job/activity UI via :mod:`.steps` — no new
frontend contract. Metering rides the existing flat-fee chokepoint
(``ToolExecutor._meter_flat_tool``), which also runs on the skill dispatch
branch, which is why every failure path here returns a string starting with
``ERROR:`` — that prefix is what stops a failed call being billed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta
from app.agent.tool_display import ToolResult

# ABSOLUTE imports, not `from . import store`. SkillLoader discovers builtins
# by path and loads them with
# ``importlib.util.spec_from_file_location("toup_skill_app_html", …)`` — a
# TOP-LEVEL module name with no parent package — so a relative import here
# raises "attempted relative import with no known parent package", the loader
# logs it and moves on, and the skill is simply absent at runtime with no
# other symptom. (Caught by tests/test_app_pipeline_gate.py, which loads
# through the real SkillLoader rather than importing the class directly.)
from app.agent.skills.builtins.app_html import appskill
from app.agent.skills.builtins.app_html import logo as logo_mod
from app.agent.skills.builtins.app_html import steps as steps_mod
from app.agent.skills.builtins.app_html import store
from app.agent.skills.builtins.app_html import verify as verify_mod
from app.agent.skills.builtins.app_html import vision as vision_mod
from app.agent.skills.builtins.app_html.shell import (
    DEFAULT_TIMEOUT_SECONDS,
    ShellRefusal,
    run_in_app_dir,
)
from app.agent.skills.builtins.app_html.store import AppStoreError

logger = logging.getLogger(__name__)

DESIGN_SKILL_FILENAME = "toup-frontend-design.md"

#: How many times the publish gate may refuse an app over POLISH findings
#: (layout, audio, the visual review) before it publishes with the remaining
#: notes named instead of refusing again. Breakage — syntax, runtime, load,
#: behaviour (a crash, a start screen that never dismisses) — is never subject
#: to this budget. Round 24: the recorded tennis build cycled open→judge→fix
#: for 9 minutes and 40 actions, then resigned with "maximum tool iterations";
#: three refusals is enough to fix a real mess and few enough to converge.
GATE_MAX_REFUSALS = 3

#: `Finding.kind`s that always refuse a publish, budget or no budget. A crash
#: or an app that cannot be started is not a rough edge.
_HARD_FINDING_KINDS = frozenset({"syntax", "runtime", "load", "behaviour"})

#: Wall clock for the whole logo phase. `ensure_icon`'s own budgets sum to
#: ~140s (a 20s subject call plus three 40s draws) and the app is already
#: built by then — every extra second is a row spinning on the user's card.
#: 45s covers a healthy draw with room for one retry; past that the app keeps
#: its letter mark and the step says so.
LOGO_PHASE_BUDGET_S = 45

#: Beyond this, `view_app_file` returns head+tail rather than the whole file.
#: A single-file app is normally 10–60 KB; 120 KB means embedded data URIs,
#: and pasting those back through the model is pure waste.
VIEW_INLINE_LIMIT = 120_000
VIEW_HEAD_CHARS = 80_000
VIEW_TAIL_CHARS = 20_000


def _packaged_design_skill() -> str:
    path = os.path.join(os.path.dirname(__file__), "DESIGN_SKILL.md")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _design_guidance() -> str:
    """The design skill's body, ready to be part of the system prompt.

    Round 18. This used to be delivered by telling the model, in a tool
    description AND in the prompt, to call
    ``read_file(path='/app/skills/toup-frontend-design.md')`` first. Two
    things came of that, and they were both bad.

    The visible one: a tool result is shown to the user, and this tool result
    is 8 KB of markdown. A person who asked for a snake game was shown the
    document's YAML frontmatter, its anti-slop checklist and its CSS token
    block, inside their own chat, as though the agent were talking to them.

    The quiet one: it made a mandatory step out of a file read that can fail.
    ``on_load`` writes the file to whichever of two directories is writable;
    if neither was, the first tool call of a fresh chat got "File not found"
    for a path the prompt had just insisted on — the cold-start failure this
    round was also asked to fix.

    Both go away by putting the guidance where guidance belongs. It costs
    tokens in the stable prefix, where they are cached, instead of a round
    trip and a rendered dump on every single build.

    The frontmatter is stripped: ``name:``/``description:`` are how a skill
    FILE is indexed, and mean nothing inside a prompt.
    """
    body = _packaged_design_skill()
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end != -1:
            body = body[end + 4:]
    # The instruction to "read this before you write any UI" is addressed to
    # a reader who had to go and fetch it. In the prompt it is already read.
    body = body.replace(
        "Read this **before** you write any UI. It is the difference between "
        "an app\nsomeone keeps and an app that looks like every other "
        "generated page.",
        "This is the difference between an app someone keeps and an app that "
        "looks\nlike every other generated page.",
    )
    return body.strip()


#: How many in-flight tool calls may be tracked at once (round 25). A turn
#: issues one tool call at a time on every wire we serve, so this is slack for
#: abandoned streams rather than a real concurrency bound.
_MAX_LIVE_CALLS = 8

#: A COMPLETE JSON string value for a top-level key, inside a prefix.
#:
#: The trailing quote is the point. A tool's arguments arrive as a growing,
#: usually-invalid JSON prefix, and half a slug is worse than no slug: it
#: names a different app. Requiring the closing quote means this returns
#: nothing until the value has actually finished arriving, so a caller can
#: poll it on every delta and act exactly once. `[^"\\]|\\.` keeps an escaped
#: quote inside the value from ending it early.
_JSON_STR_RE_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _partial_arg(partial_json: str, key: str) -> str:
    """The value of ``key``, or ``""`` if it has not fully arrived yet."""
    pattern = _JSON_STR_RE_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(
            r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"'
        )
        _JSON_STR_RE_CACHE[key] = pattern
    m = pattern.search(partial_json or "")
    if not m:
        return ""
    try:
        return json.loads(f'"{m.group(1)}"')
    except ValueError:
        return ""


def _streamed_body_len(partial_json: str) -> int:
    """How much of the app's own document has streamed so far, in bytes.

    Measured from the opening quote of the ``html`` value to the end of what
    has arrived — deliberately not the length of the whole buffer, which also
    carries the slug, the title and the brief and would show a build as
    already several KB in before a line of the app existed.
    """
    marker = _HTML_KEY_RE.search(partial_json or "")
    if not marker:
        return 0
    return max(0, len(partial_json) - marker.end())


_HTML_KEY_RE = re.compile(r'"html"\s*:\s*"')


def _short(exc: Exception) -> str:
    """A failure, in one clause, for a step's detail line.

    Tool-result prose and step details have different readers and different
    lengths. `store.edit_app`'s "old_string appears 3 times in snake.html — it
    must match exactly once. Include more surrounding lines to make it
    unique." is exactly right for the model and far too long, and about the
    wrong subject, for a line under a progress bar. First sentence only, and
    never the shell's allowlist, which alone runs to 300 characters of binary
    names.
    """
    text = " ".join(str(exc).split())
    text = text.split(" bash_app is a verification shell")[0]
    for stop in (". ", " — "):
        head = text.split(stop)[0]
        if len(head) >= 12:
            text = head
            break
    return text[:120]


def _writable(directory: str) -> bool:
    try:
        os.makedirs(directory, exist_ok=True)
        return os.access(directory, os.W_OK)
    except OSError:
        return False


def design_skill_path() -> str:
    """Where the design skill lives inside the container.

    ``settings.skills_dir`` (``/app/skills``) is the natural home: it is an
    rw volume AND one of the roots ``ToolExecutor._allowed_path_roots``
    permits, so the agent can ``read_file`` it. The apps root is the fallback
    for a box where that directory does not exist (a dev checkout, a test).

    **Resolved here, not in ``on_load``, and the choice must be settled
    before ``get_tools()`` runs.** ``SkillLoader._register`` calls
    ``get_tools()`` BEFORE ``on_load()``, and this path is embedded in both a
    tool description and the system-prompt section — a value that changed
    between those two moments would put two different byte sequences into the
    wire array over a container's life, which is a forked provider cache
    lineage. Both candidates are fixed for the process, so this function is
    deterministic.
    """
    try:
        from app.config import settings
        base = getattr(settings, "skills_dir", "") or "/app/skills"
    except Exception:  # pragma: no cover
        base = "/app/skills"
    if not _writable(base):
        base = store.apps_root()
    return os.path.join(base, DESIGN_SKILL_FILENAME)


class InternalRefusal(AppStoreError):
    """A refusal whose reason the user must never be told.

    `execute_tool` turns an `AppStoreError` into an `ERROR:` string for the
    model AND a one-clause `display` for the user, built from the same
    sentence. That is right for almost every refusal here — "old_string
    appears 3 times" shortens to something a person can live with.

    It is wrong for exactly one class: a refusal that is ABOUT the app's
    internal brief. "legacy has no brief yet, so it was not published" is a
    perfectly good instruction to a model and, rendered in a chat, tells
    someone that a file exists which describes them, which they cannot open,
    and which they were never meant to know about. So this subclass carries
    its own user-facing sentence and the detail never reaches it.
    """

    def __init__(self, message: str, *, display: str) -> None:
        super().__init__(message)
        self.display = display


class AppHtmlSkill(Skill):
    """Single-file HTML artifact pipeline."""

    meta = SkillMeta(
        name="app_html",
        version="1.0.0",
        description="Build apps as one self-contained HTML file (no build step)",
        author="toup",
    )

    def __init__(self) -> None:
        #: Edit reasons since this app was last published, per slug. Handed to
        #: the visual review so it can answer "is the change you just made on
        #: screen" rather than only "does this look broken". In-process on
        #: purpose: it is context for a review, not a fact about the app, and
        #: losing it across a restart costs one sentence of prompt, not
        #: correctness. The durable record of what changed is the brief's own
        #: history, which `edit_app_file` writes.
        self._pending_changes: Dict[str, List[str]] = {}
        #: Gate refusals since the last successful publish, per slug. Round 24
        #: budget discipline: the recorded tennis build burned 40 actions and 9
        #: minutes cycling open→judge→fix and then resigned mid-sentence. The
        #: gate now refuses POLISH findings (layout/audio/visual) at most
        #: `_GATE_MAX_REFUSALS` times per app; after that the app publishes
        #: with the remaining notes named honestly. Breakage (syntax/runtime/
        #: load/behaviour — a crash, an app that never starts) refuses forever:
        #: no budget ships a broken app. In-process on purpose, like
        #: `_pending_changes`: losing the count across a restart costs at most
        #: one extra polish loop.
        self._gate_refusals: Dict[str, int] = {}
        #: Round 25, items 1 and 5. One entry per tool call whose arguments
        #: are still arriving, keyed by the provider's call id: the phase it
        #: opened, the last frame it broadcast (so the card can be kept moving
        #: without a database write per tick) and how much of the app has
        #: streamed so far. Dropped when the call executes.
        self._live_calls: Dict[str, Dict[str, Any]] = {}
        self._design_path: str = design_skill_path()
        # Read ONCE, from the image, in __init__ — before `get_tools()` and
        # before `on_load()`, both of which the loader calls later. The bytes
        # are therefore fixed for the life of the process, which is the cache
        # invariant `get_system_prompt_section` has to hold.
        self._design_guidance: str = _design_guidance()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def on_load(self) -> None:
        """Create the app root and materialise the design skill on disk.

        The packaged copy is authoritative — it is rewritten on every boot so
        an image upgrade actually ships the new guidance instead of leaving a
        stale copy on the volume forever.
        """
        # AppStoreError as well as OSError: `ensure_root` now REFUSES an
        # unwritable root instead of returning it, and boot is the one caller
        # that must not care. A skill whose on_load raises does not load, and
        # a workspace that is unwritable at boot is very often writable by the
        # time someone asks for an app (`_create` calls this again, and the
        # repair runs there too). Losing the whole pipeline over it would turn
        # a transient into a permanent.
        try:
            store.ensure_root()
        except (OSError, AppStoreError):
            logger.warning("[app_html] could not create the app root", exc_info=True)

        content = _packaged_design_skill()
        if not content:
            logger.warning("[app_html] packaged design skill missing")
            return
        # Write to the path already advertised by __init__ — never re-resolve
        # it here (see design_skill_path's docstring). Overwritten on every
        # boot so an image upgrade actually ships the new guidance instead of
        # leaving last quarter's copy on the volume forever.
        try:
            os.makedirs(os.path.dirname(self._design_path), exist_ok=True)
            with open(self._design_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            logger.info("[app_html] design skill at %s", self._design_path)
        except OSError:
            # The prompt still names this path. That is the right trade: the
            # agent gets "File not found" from read_file — a visible, fixable
            # failure — instead of a silently missing design pass.
            logger.warning(
                "[app_html] could not write the design skill to %s",
                self._design_path, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------
    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "app_html__create_app_file",
                "description": (
                    "Build an app by writing ONE complete, self-contained HTML file. "
                    "This is how apps are built — there is no scaffold, no bundler, no "
                    "package install.\n"
                    "RULES: inline every line of CSS in a <style> and every line of JS "
                    "in a <script> — the file must run with no other file. External "
                    "libraries ONLY from https://cdnjs.cloudflare.com (React needs "
                    "react, react-dom and babel-standalone from cdnjs, with "
                    "<script type=\"text/babel\">). Mobile-first: design 360px first, "
                    "then 768 and 1280.\n"
                    "The JavaScript is parsed the moment you write it and the result "
                    "comes back in this tool's reply, so a truncated or unbalanced "
                    "script is caught here rather than by the user opening a dead "
                    "page.\n"
                    "WORKFLOW: one big write here, then surgical edit_app_file changes, "
                    "then present_app. Calling this again for "
                    "the same slug replaces the file (the previous revision is kept)."
                ),
                "input_schema": {
                    "type": "object",
                    # `brief` deliberately PRECEDES `html`. Arguments are
                    # generated in schema order, so this is what makes the
                    # plan get written before the code that implements it —
                    # palette, layout and control placement are committed in
                    # prose first, and the file is written to that plan
                    # rather than the plan reverse-engineered from the file.
                    "properties": {
                        "slug": {
                            "type": "string",
                            "description": "URL-safe id: 1-60 lowercase letters, digits and hyphens (e.g. 'budget-tracker').",
                        },
                        "title": {
                            "type": "string",
                            "description": "Human name shown on the app card, e.g. 'Budget Tracker'.",
                        },
                        "brief": {
                            "type": "string",
                            "description": (
                                "Your PLAN, in markdown, written BEFORE the html — "
                                "stored beside the app, NEVER shown to the user, and "
                                "the first thing you will read before every future "
                                "edit. Cover: what it is and who it is for; the "
                                "problem it solves; its STATE DIAGRAM — every screen/"
                                "state and the arrow between each pair (start → "
                                "playing → ended → playing again), including what "
                                "dismisses each overlay and how every terminal state "
                                "leads back; every feature, "
                                "state and control and what each does; the layout — "
                                "where each control cluster sits and how it aligns; "
                                "and the design decisions (palette, type, spacing, "
                                "the signature element) WITH your reasoning. The html "
                                "you write next implements THIS — the publish gate "
                                "presses your start control and refuses the app if "
                                "the screen does not change, so the transitions you "
                                "draw here are the ones that must work. Write it now, "
                                "while you still know why you chose what you chose — "
                                "in three turns' time this file is all that is left "
                                "of it."
                            ),
                        },
                        "html": {
                            "type": "string",
                            "description": (
                                "The COMPLETE document implementing the brief above: "
                                "<!doctype html> through </html>, with all CSS and JS "
                                "inlined. Not a fragment, not a placeholder."
                            ),
                        },
                    },
                    "required": ["slug", "title", "brief", "html"],
                },
            },
            {
                "name": "app_html__view_app_file",
                "description": (
                    "Read an app's current HTML exactly as it is on disk, with the "
                    "app's own brief above it. Call this before every edit_app_file — "
                    "old_string must match the file byte-for-byte, and you cannot "
                    "match text you have not looked at in this turn.\n"
                    "The brief is what this app is FOR. Read it before you decide what "
                    "to change: a request is about the app, not about its own wording."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string", "description": "The app's slug."},
                    },
                    "required": ["slug"],
                },
            },
            {
                "name": "app_html__edit_app_file",
                "description": (
                    "Change part of an app by exact string replacement. old_string must "
                    "appear EXACTLY ONCE — if it is missing or ambiguous the edit is "
                    "refused and nothing is written, so include enough surrounding "
                    "lines to be unique. Prefer several small edits over rewriting the "
                    "file. Copy old_string from view_app_file output, including "
                    "indentation and line breaks.\n"
                    "The file is read back from disk afterwards and the reply says "
                    "whether the change is actually in it, and whether the script still "
                    "parses. Do not describe a change to the user on weaker evidence "
                    "than that line — and not before present_app has published it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string", "description": "The app's slug."},
                        "old_string": {
                            "type": "string",
                            "description": "Exact text to replace. Must occur exactly once in the file.",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text. Use an empty string to delete.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Short phrase for the progress card, e.g. 'add dark mode toggle'. Also recorded in the app's brief as this revision's history line.",
                        },
                        "brief": {
                            "type": "string",
                            "description": (
                                "Only when this change alters what the app IS — a new "
                                "feature, a changed audience, a different look. Pass "
                                "the app's brief REWRITTEN in full (it replaces the "
                                "previous one); omit it for an ordinary tweak, whose "
                                "history line comes from `reason`."
                            ),
                        },
                    },
                    "required": ["slug", "old_string", "new_string"],
                },
            },
            {
                "name": "app_html__bash_app",
                "description": (
                    "Run a read-only shell command in the app directory to verify your "
                    "work — grep for a handler you just added, wc -c the file, "
                    "node --check an extracted script. Allowed commands only "
                    "(ls cat head tail wc grep sed awk diff find node python3 jq …); "
                    "no network, no package managers, no paths outside the app "
                    "directory. Use the main `exec` tool for anything else."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string", "description": "The app's slug."},
                        "command": {
                            "type": "string",
                            "description": "Shell command, run with the app directory as CWD. The app is ./<slug>.html.",
                        },
                    },
                    "required": ["slug", "command"],
                },
            },
            {
                "name": "app_html__present_app",
                "description": (
                    "Publish the app to the user: registers it, refreshes the preview "
                    "and posts the app card in chat. Call this once the app is "
                    "finished — and again after a round of edits so the user sees the "
                    "new version.\n"
                    "The app is opened in a real browser at 390x844 first, played "
                    "with, PHOTOGRAPHED and looked at. It is refused if anything "
                    "throws, if a control renders under 44x44, if text is under 12px, "
                    "if the page scrolls sideways, if it builds sound that never "
                    "plays, or if the screenshot shows something broken — unreadable "
                    "text, a clipped or empty panel, a collapsed layout. If it comes "
                    "back with problems, fix them with edit_app_file and call this "
                    "again. It also needs the app's brief to exist; write one with "
                    "the `brief` argument on create_app_file or edit_app_file."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string", "description": "The app's slug."},
                    },
                    "required": ["slug"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    def get_system_prompt_section(self) -> Optional[str]:
        """Constant for the life of the container — see the cache invariant in
        ``app/agent/tool_entitlements.py``. Nothing here varies per turn."""
        return (
            "# Building apps\n"
            "An app is ONE self-contained .html file — no scaffold, no bundler, no "
            "package install. Inline all CSS and JS. External libraries ONLY from "
            "https://cdnjs.cloudflare.com (React = react + react-dom + "
            "babel-standalone from cdnjs). Mobile-first: 360px, then 768, then 1280. "
            "It runs in a sandboxed frame on an opaque origin: there is no network, "
            "no navigation and no parent page. Storage cannot throw (the runner "
            "replaces it), but it is not durable within a first paint either, so "
            "seed the UI from in-memory defaults and reconcile after.\n"
            "`classList.add`/`remove` THROW on an empty string, so never pass a "
            "conditional that can produce one — `add('snake', i===0?'head':'')` "
            "kills the whole render loop on the second segment. Use "
            "`classList.toggle('head', i===0)` or add conditionally.\n"
            "BUILD IT IMMEDIATELY. Do NOT ask the user questions first, do not "
            "offer directions to choose between, do not present a plan and wait, "
            "and do not announce what you are about to do — go straight to "
            "app_html__create_app_file on the very first turn. \"Build me a snake "
            "game\" is a complete brief: pick the obvious interpretation, make "
            "every remaining decision yourself, and show the user something they "
            "can play. Ask only if the request is genuinely unintelligible — "
            "never to narrow scope, confirm taste or gather requirements. When "
            "the app is up, say in one line what you built and what you can "
            "change; refinement happens on a working app, not on a questionnaire.\n"
            "You have the internet: your ordinary web_search works mid-build "
            "like in any other turn — never ask the user for facts you can "
            "look up. Use it BEFORE you build whenever it would genuinely "
            "improve your understanding of the request: a named real thing "
            "(classic Windows Minesweeper, a Nokia 3310, a known board game — "
            "its colours, its chrome, its rules), an unfamiliar subject, or "
            "real-world facts the app itself needs to be correct (rules, "
            "formulas, categories, conventions). Research is for accuracy, "
            "never for delay: one or two quick searches, no questions, "
            "straight into the build.\n"
            "The `brief` argument is your PLAN and you write it before the "
            "html: subject grounding, palette with its reasoning, the layout "
            "with exact control placement and alignment, the input model, the "
            "STATE DIAGRAM — every screen/state with the arrow between each "
            "pair (start → playing → ended → playing again), what dismisses "
            "each overlay, and how every terminal state leads back — and the "
            "one signature element. The file implements the plan — not the "
            "other way round. The publish gate executes the diagram: it "
            "presses your start control and refuses the app if the screen "
            "does not change, so a start handler's first job is hiding its "
            "own overlay.\n"
            "Loop: app_html__create_app_file (one big write) → app_html__edit_app_file "
            "(small exact-string edits; view_app_file first) → app_html__present_app "
            "(show the user). app_html__bash_app is there when you want to look "
            "something up in the file; it is not a required step, because the "
            "write and the publish check the app themselves. Never hand-scaffold "
            "an app with exec/write_file.\n"
            "## Sound\n"
            "An app may make noise, and if it is a game it usually should. Build "
            "the AudioContext INSIDE the first input handler, never at the top of "
            "the script, and call `ctx.resume()` at the top of every handler that "
            "plays something — a context made while the page is loading starts "
            "suspended, and nothing anywhere reports that the app went silent. "
            "Synthesise sounds with oscillators; a data: URI works too. Never "
            "fetch one, never use autoplay, and give anything that loops a way to "
            "turn it off.\n"
            "## The app's brief\n"
            "Every app carries a brief: what it is, who it is for, the problem it "
            "solves, its flows, what each control does, and why it looks the way "
            "it does. You write it with the `brief` argument to "
            "app_html__create_app_file, and app_html__view_app_file hands it back "
            "to you before every edit. It is INTERNAL — never quote it, never "
            "paste it into chat, never tell the user it exists, and never mention "
            "writing one.\n"
            "Read it before you decide what to change. A request is about the app "
            "the brief describes, not about its own wording: work out what the "
            "person is trying to solve and answer THAT. If what you have been "
            "asked for would break what the app is for, say so in one sentence "
            "and propose the change that serves the purpose — then do it. Pass a "
            "rewritten `brief` on the edit whenever the change alters what the "
            "app is.\n"
            "The brief is VERSIONED: every write to the app bumps it and adds "
            "one line saying why the change was made and what the file actually "
            "did. Those lines are the app's history — read the last few before "
            "you change anything, because they are where you find out that the "
            "thing you are about to 'fix' was done deliberately two versions "
            "ago. Always pass `reason` to app_html__edit_app_file: it is what "
            "the next editor reads, and 'edited' tells them nothing.\n"
            "If a check comes back with problems, fix ALL of them in ONE edit "
            "pass and call present_app again — never fix one finding per "
            "round. The polish budget is three checks per app: cosmetic "
            "findings that survive three rounds ship with the app and are "
            "named to you honestly (tell the user in one plain sentence, no "
            "internals); a crash or an app that will not start never ships at "
            "all. Do not report a failure to the user and stop, and never "
            "present an app that has not come back clean or been honestly "
            "closed out this way.\n"
            "## Changing an app that is already open\n"
            "\"Make the button bigger\" is about the control the person was USING "
            "when they said it. In a game that is the thing they play with — the "
            "D-pad, the fire button, the paddle — not the PLAY button on the "
            "start screen, which they pressed once. Ask yourself which element "
            "the complaint could have come from, and if more than one answer is "
            "reasonable, CHANGE THEM ALL: every control of that kind, in the same "
            "way, in one round of edits. A person who says a control is too small "
            "and gets a bigger menu button has been answered with the wrong "
            "object, and has to ask again. Widening the change is nearly free; "
            "guessing wrong costs a whole turn. Never ask which button they "
            "meant.\n"
            "When the request names no element at all (\"make it nicer\", \"too "
            "cramped\"), read the file first and change the thing the words are "
            "actually about, not the first match for them.\n"
            "## Never say it is done until it is\n"
            "app_html__edit_app_file reads the file back and tells you whether "
            "your change is in it. app_html__present_app opens the app in a real "
            "browser, plays with it, photographs it and LOOKS at the picture — so "
            "it can tell you the app runs and still refuse it for text nobody can "
            "read or a panel that came out empty. Both of those come back "
            "BEFORE you write a word to the user about the change. An edit that "
            "was written is not an edit the person can see: publishing is what "
            "puts it in front of them, so \"done\" means present_app returned "
            "clean, not that a write returned. If a read-back or a publish "
            "reports a problem, fix it and go round again — do not narrate the "
            "problem and do not claim the change.\n"
            "Do NOT call create_job for a build. This pipeline opens its own job "
            "and reports every phase into it, so a second one is the same build "
            "announced twice — the user sees two progress cards for one app.\n"
            "## What you say when it is done\n"
            "One or two plain sentences: what it is, and what you can change next. "
            "Nothing else goes in that message — no link, no file path, no "
            "`[[open_app:…]]` chip and no `[[navigate:…]]` chip. The app card is "
            "already in the chat with its own Open button, and a second one beside "
            "it is a second button for the same thing. Never paste a tool's "
            "output, an exit code, a byte count or an error dump into your reply; "
            "if something went wrong, say what happened in one sentence and what "
            "you are doing about it.\n"
            "\n"
            f"{self._design_guidance}"
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    #: Which phase each tool opens the moment the model starts calling it.
    #: `present_app` is absent deliberately — its arguments are one slug, so
    #: it is executing within a token of starting and an extra pre-open would
    #: be two writes for one transition.
    _LIVE_PHASE: Dict[str, str] = {
        "app_html__create_app_file": "create",
        "app_html__view_app_file": "review",
        "app_html__edit_app_file": "edit",
    }

    async def on_tool_input(
        self, tool_name: str, call_id: str, partial_json: str,
        ctx: SkillContext,
    ) -> None:
        """Open — and then keep moving — the build card while the model is
        still writing the tool call.

        Round 25, items 1 and 5. The card used to be minted inside `_create`,
        which runs only once the whole call has arrived; and the whole call is
        the app. So a build looked like this: a generic "Building your app"
        spinner, alone, for as long as the document took to write — commonly
        two minutes — and then, all at once, a finished-looking steps card.
        Every phase the pipeline reports was reported truthfully; none of it
        was reported while there was anything to watch.

        Nothing here is new information. `slug` and `title` are the first two
        properties of the tool's schema and arguments are generated in schema
        order — the same fact the schema already relies on to get `brief`
        written before `html`. They are on the wire within a second or so of
        the model committing to the call; they were simply being buffered in
        the provider adapter and thrown at us at the end.

        Cheap and idempotent by contract: `ensure_job` is find-or-create, the
        phase transition is emitted at most once per call, and every tick
        after that re-sends the frame the transition already produced with a
        fresh detail (`steps.retouch`) rather than writing the row again.
        """
        phase = self._LIVE_PHASE.get(tool_name)
        if not phase or not call_id:
            return
        state = self._live_calls.get(call_id)
        if state is None:
            slug = _partial_arg(partial_json, "slug")
            if not slug:
                # The identifying field has not finished arriving. There is
                # nothing to open a card ON yet, and guessing one would be a
                # card for the wrong app.
                return
            try:
                slug = store.normalise_slug(slug)
            except (AppStoreError, ValueError):
                return
            title = _partial_arg(partial_json, "title") or slug
            job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
            if not job_id:
                return
            frame = await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type=phase,
                status="running",
            )
            state = {"slug": slug, "job_id": job_id, "phase": phase,
                     "frame": frame, "written": -1}
            self._live_calls[call_id] = state
            # Bounded, and evicted by age rather than by the call completing:
            # there is no seam that tells a skill a call it watched has
            # finished (`execute_tool` gets no call id), so an entry left
            # behind by an abandoned stream would otherwise live for the life
            # of the process. Each entry holds one frame, and only the most
            # recent handful can still be receiving deltas.
            while len(self._live_calls) > _MAX_LIVE_CALLS:
                self._live_calls.pop(next(iter(self._live_calls)), None)

        if phase != "create":
            # Only the app's own body is worth a byte counter. An edit's
            # arguments are a few lines and a view's is a slug.
            return
        written = _streamed_body_len(partial_json)
        # Whole kilobytes only: this fires about once a second and a number
        # that twitches is harder to read than one that ticks.
        if written // 1024 == state["written"] // 1024:
            return
        state["written"] = written
        if written <= 0:
            return
        state["frame"] = await steps_mod.retouch(
            ctx.user_id, state["frame"], step_type="create",
            detail=f"{written // 1024:,} KB written so far",
        )

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext,
    ) -> str:
        handlers = {
            "app_html__create_app_file": self._create,
            "app_html__view_app_file": self._view,
            "app_html__edit_app_file": self._edit,
            "app_html__bash_app": self._bash,
            "app_html__present_app": self._present,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"ERROR: Unknown tool '{tool_name}'"
        try:
            return await handler(args, ctx)
        except (AppStoreError, ShellRefusal) as exc:
            # Expected, model-fixable. No traceback — the message IS the fix.
            #
            # `display` is what the user is shown instead. The model's copy of
            # a refusal is a paragraph: what was wrong, why, and the exact
            # call to make next, sometimes with the shell allowlist attached.
            # None of that is the user's problem and all of it was rendering
            # in their chat, so they get the first clause of it and nothing
            # else. Failures are never silently prettified into success —
            # `_FAILED_DISPLAY` says plainly that something did not work.
            return ToolResult(
                f"ERROR: {exc}",
                display=self._failure_display(tool_name, exc),
            )
        except OSError as exc:
            logger.warning("[app_html] %s filesystem error", tool_name, exc_info=True)
            return ToolResult(
                f"ERROR: filesystem error: {exc}",
                display="Couldn't save that — the workspace didn't accept the write.",
            )

    #: What a user is told when a phase of the build fails. One sentence per
    #: tool, because "what went wrong" is a different sentence depending on
    #: what was being attempted, and "Error" is not an answer.
    _FAILED_DISPLAY = {
        "app_html__create_app_file": "Couldn't write the app",
        "app_html__view_app_file": "Couldn't read the app",
        "app_html__edit_app_file": "Couldn't update the app",
        "app_html__bash_app": "That check couldn't run",
        "app_html__present_app": "The app isn't ready yet",
    }

    @staticmethod
    def _failure_display(tool_name: str, exc: Exception) -> str:
        # An InternalRefusal brought its own sentence precisely so that this
        # function never gets to shorten one that must not be shown at all.
        if isinstance(exc, InternalRefusal):
            return exc.display
        head = AppHtmlSkill._FAILED_DISPLAY.get(tool_name, "That didn't work")
        detail = _short(exc)
        return f"{head} — {detail}" if detail else f"{head}."

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    @staticmethod
    def _existing_title(slug: str) -> str:
        """Title from the manifest, after proving the app is really there.

        Every tool except `create` opens (or re-opens) the app's job card
        before doing its work — so each one must first establish that the app
        EXISTS. Without this, `view_app_file('budget-trakcer')` would mint a
        job and post a "Build: budget-trakcer" card in the user's chat for a
        typo, and the card would never resolve because nothing is building.
        """
        record = store.read_manifest().get(slug)
        # A record with no file is a repairable state, not a missing app —
        # the last kept revision IS the app. Try that before telling the
        # model (and through it the user) that the thing does not exist.
        if record is not None and not store.exists(slug):
            store.restore_latest_version(slug)
        if record is None or not store.exists(slug):
            known = ", ".join(sorted(store.read_manifest())) or "none yet"
            raise AppStoreError(
                f"no app named {slug!r}. Apps that exist: {known}. "
                f"Use create_app_file to make a new one."
            )
        return record.title or slug

    @staticmethod
    def _size_detail(size_bytes: int) -> str:
        """A size a person reads, for the step's detail line.

        `9299 bytes` was appearing as a step's own label. It is not a label,
        and at four significant figures it is not even a useful number — the
        only question it answers is "did something get written", which
        `9.3 KB` answers just as well and does not read as a memory dump.
        """
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        return f"{size_bytes / 1024:.1f} KB"

    async def _create(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        title = (args.get("title") or slug).strip()
        html = args.get("html")

        # Validate BEFORE opening the job. A stub call or a blocked CDN
        # reference is a call that never should have happened; minting a job
        # for it would leave a permanently failed "Build: …" card in chat
        # for an app that was never started. `ensure_root` joins it for the
        # same reason and one more: it is the cold-start repair (create the
        # directory, fix its mode), and a workspace that cannot be written to
        # should stop the build before there is a card to strand.
        #
        # The brief is checked here too, and for exactly the same reason. It
        # is not paperwork that can be caught up later: the moment to write
        # down why this app is the way it is, is the moment the model still
        # knows, and that moment is this call.
        store.validate_html(html)
        narrative = self._check_brief(args.get("brief"), required=True)
        store.ensure_root()

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="create",
            status="running",
        )
        try:
            record, warnings = store.write_app(slug, title, html)
        except AppStoreError as exc:
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="create",
                status="failed", detail=_short(exc),
            )
            raise
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="create",
            status="done", detail=self._size_detail(record.size_bytes),
        )
        # No step of its own, ever. The brief is internal: a row reading
        # "Wrote the app's brief" would tell the user a file exists that they
        # cannot open and were never meant to know about. It rides the write
        # that produced it, and a failure to store it is logged, not raised —
        # losing the note must never lose the app.
        self._save_brief(slug, title=title, html=html, revision=record.revision,
                         narrative=narrative, history_line="built")

        # Parse the script NOW. A generation that ran out of room mid-function
        # produces a file that is a perfectly valid HTML document with a dead
        # script inside it — `validate_html` passes it, the card goes green,
        # and the first person to find out is the user, looking at a board
        # that will not move. Syntax only here: it is milliseconds and offline,
        # and the expensive real-browser pass belongs at the publish gate
        # where it runs once instead of once per edit.
        report = await verify_mod.verify_app(html, deep=False)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="verify",
            status="failed" if report.findings else "done",
            detail=report.summary(),
            # The tool result below hands the model the parse errors and tells
            # it to fix them first, so this is the loop working, not a dead
            # build. Round 25, item 2 — see `emit_step`'s `recoverable`.
            recoverable=True,
        )

        out = (
            f"Wrote {slug}.html ({record.size_bytes:,} bytes, revision "
            f"{record.revision}).\n"
        )
        if warnings:
            out += "Warnings:\n" + "".join(f"  - {w}\n" for w in warnings)
        if report.findings:
            out += (
                "The script does not parse, so the app will render but do "
                "nothing:\n" + report.as_error() + "\n"
                "Fix that before anything else.\n"
            )
        return ToolResult(
            out.rstrip(),
            display=("Wrote the app — the script needs a fix."
                     if report.findings else "Wrote the app."),
        )

    @staticmethod
    def _check_brief(raw: Any, *, required: bool) -> str:
        """`appskill.validate_narrative`, with the user kept out of it.

        The validator's messages are instructions to a model and name the
        thing they are about — "brief is required", "that is a label, not a
        brief". `_short` would put the first clause of one of those under a
        progress bar, which tells the user about the internal file as surely
        as printing it would. So every refusal from this validator is
        re-raised with a sentence written for a person; the model still gets
        the full rule.
        """
        try:
            return appskill.validate_narrative(raw, required=required)
        except AppStoreError as exc:
            raise InternalRefusal(
                str(exc), display="Nearly there — I'm just finishing this off.",
            ) from None

    @staticmethod
    def _purpose_line(brief: Optional["appskill.Brief"]) -> str:
        """One sentence of what the app is, for the reviewer and the icon.

        The first real prose line of the narrative. Both consumers need to
        know what they are looking at — an empty board is correct for a chess
        app and a defect for a dashboard, and "Nokia Snake Classic" tells an
        icon model considerably less than the sentence under *What it is*.

        Prose is PREFERRED, not required. A bulleted line is skipped on the
        first pass because in a markdown brief those are usually the feature
        list, and a feature is not the purpose. But returning "" when the whole
        narrative happens to be bulleted costs more than it saves now: round 22
        made this string the gate on the reviewer's palette judgement (an empty
        purpose means the reviewer is told nothing and says nothing about
        colour), and §1a asks the model to open the brief with three answers
        that it may well write as a list. So the bullets are a fallback rather
        than a blind spot — still second, still only when there is no prose.
        """
        if brief is None or not brief.narrative:
            return ""
        fallback = ""
        for raw in brief.narrative.splitlines():
            line = raw.strip().lstrip("#").strip()
            if not line or len(line) <= 30:
                continue
            if not line.startswith(("-", "*", "#")):
                return line[:400]
            if not fallback:
                fallback = line.lstrip("-*").strip()[:400]
        return fallback

    @staticmethod
    def _save_brief(slug: str, *, title: str, html: str, revision: int,
                    narrative: Optional[str], history_line: str) -> None:
        """Write the app's note. Never raises — see the call sites."""
        try:
            brief = appskill.save(slug, title=title, html=html,
                                  revision=revision, narrative=narrative,
                                  history_line=history_line)
        except Exception:  # noqa: BLE001 - a note is never worth an app
            logger.warning("[app_html] could not write the brief for %s", slug,
                           exc_info=True)
            return
        logger.info("[app_html] brief for %s is at version %d", slug,
                    brief.version)

    async def _view(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        title = self._existing_title(slug)

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="review",
            status="running",
        )
        content = store.read_app(slug)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="review",
            status="done", detail=self._size_detail(len(content.encode("utf-8"))),
        )

        # `display` matters more here than anywhere else in this file: the
        # model-facing result is the ENTIRE app, and the client renders the
        # first couple of kilobytes of whatever it is given. Without it, a
        # routine read-before-edit puts the user's own doctype, meta tags and
        # CSS custom properties in their chat as the agent's reply.
        # The brief goes ABOVE the file, in the model-facing half only.
        # Round 20: this is the mandatory read, and it is mandatory by being
        # unavoidable rather than by being instructed. The prompt already
        # requires view_app_file before every edit; putting the app's purpose
        # in that same result means an edit cannot be made by a model that
        # has not been told what the app is for. A separate "read the brief"
        # tool would be a step that can be skipped, and one more tool name in
        # the user's actions rail for a file they must never learn about.
        record = store.read_manifest().get(slug)
        brief = appskill.context_block(
            slug, current_revision=record.revision if record else 1,
        )

        if len(content) <= VIEW_INLINE_LIMIT:
            # Raw, no line numbers: edit_app_file matches byte-for-byte, and
            # a "  42\t" gutter is exactly the kind of thing that ends up
            # copied into old_string and never matches.
            return ToolResult(f"{brief}\n\n{content}", display="Read the app.")
        head = content[:VIEW_HEAD_CHARS]
        tail = content[-VIEW_TAIL_CHARS:]
        omitted = len(content) - VIEW_HEAD_CHARS - VIEW_TAIL_CHARS
        return ToolResult(
            f"{brief}\n\n"
            f"{head}\n\n"
            f"[… {omitted:,} characters omitted — {slug}.html is {len(content):,} "
            f"chars. Use app_html__bash_app with grep -n to locate the exact text "
            f"you want to edit in the omitted region …]\n\n"
            f"{tail}",
            display="Read the app.",
        )

    async def _edit(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        reason = (args.get("reason") or "").strip()
        title = self._existing_title(slug)
        # Optional here, required on create: an ordinary tweak does not
        # change what the app is, and demanding a rewritten brief for every
        # padding change would guarantee the brief stops being read.
        narrative = self._check_brief(args.get("brief"), required=False)

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="edit",
            status="running", detail=reason,
        )
        try:
            rec, delta = store.edit_app(
                slug, args.get("old_string", ""), args.get("new_string", ""),
            )
        except AppStoreError as exc:
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="edit",
                status="failed", detail=_short(exc),
            )
            raise
        # ── Read it back ──────────────────────────────────────────────
        # An edit used to report success on the strength of `_atomic_write`
        # not having raised. That is a claim about a syscall, and the thing
        # the model then tells the user — "I made the button bigger" — is a
        # claim about the file. They are not the same claim, and the whole
        # class of bug where one is offered as the other is what this round is
        # about. So the file is opened again, from disk, and asked.
        new_string = args.get("new_string", "")
        try:
            on_disk = store.read_app(slug)
        except AppStoreError:
            on_disk = ""
        landed = bool(new_string) and new_string in on_disk
        if new_string and not landed:
            # The write returned, the file does not contain it. Something
            # else is writing this path, or the volume lied about the flush.
            # Never report this as an edit: the model would go on to tell the
            # user about a change that is not in the app they will open.
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="edit",
                status="failed", detail="the change did not land on disk",
            )
            raise AppStoreError(
                f"the edit was written but {slug}.html does not contain the "
                f"new text when read back — the change did NOT take effect. "
                f"Do not tell the user it did. Call view_app_file to see the "
                f"current file and try again."
            )

        # Syntax, on every edit, not only on create. `create_app_file` has
        # parsed its own JavaScript since round 18; an edit could break the
        # same script and nothing looked until `present_app` — so a model that
        # edited and then answered the user shipped a dead script with a
        # success message on top of it.
        report = await verify_mod.verify_app(on_disk or "", deep=False)

        # The note follows the file. The structure section is re-derived from
        # the bytes that just landed (a map of where things were two edits ago
        # sends the next edit to the wrong place) and `reason` becomes this
        # revision's history line — the durable record of what has been
        # happening to this app, which is what the next editor reads.
        self._save_brief(
            slug, title=title, html=on_disk or "", revision=rec.revision,
            narrative=narrative or None,
            history_line=reason or "edited",
        )
        if reason:
            pending = self._pending_changes.setdefault(slug, [])
            if reason not in pending:
                pending.append(reason)
            del pending[:-4]

        # `+0 bytes` was the old detail for a same-length edit — a change the
        # user could see in the app, described to them as nothing having
        # happened. The reason the model gave for the edit is both true and
        # worth reading; the revision is the fallback when it gave none.
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="edit",
            status="failed" if report.findings else "done",
            detail=(report.summary() if report.findings
                    else (reason or f"revision {rec.revision}")),
            # Same as `_create`: a parse error found on an edit is handed
            # straight back for repair (round 25, item 2).
            recoverable=True,
        )

        # A deletion has no new text to find, so the proof is the other half:
        # the old text is gone. Both are read from the file, not inferred.
        confirmed = ("the new text is in the file" if landed
                     else "the old text is gone from the file")
        out = (
            f"Edited {slug}.html — {delta:+,} bytes, now {rec.size_bytes:,} "
            f"(revision {rec.revision}).\n"
            f"Read back from disk: {confirmed}.\n"
        )
        for w in getattr(rec, "warnings", []) or []:
            out += f"  - warning: {w}\n"
        if report.findings:
            out += (
                "But the script no longer parses, so the app will render and "
                "do nothing:\n" + report.as_error() + "\n"
                "Fix that before you say anything to the user about this "
                "change.\n"
            )
        else:
            out += (
                "The change is on disk and the script still parses. It is NOT "
                "in front of the user until app_html__present_app publishes "
                "it — call that before you describe the change.\n"
            )
        # Round 21, item 4. The memory has just been written and is a version
        # ahead of anything the model has been shown, so the new version comes
        # back WITH the result. That is what makes "read it before you edit"
        # true for every change after the first in a chain, whether or not the
        # model remembered to call view_app_file — and the copy is never a
        # duplicate, because a write is what produced it.
        #
        # Model-facing only; `display` says nothing about it, which is the
        # same rule `_view` follows.
        out += (
            "\nYour working memory of this app, at the version this edit just "
            "wrote. Read it before the next change.\n"
            f"{appskill.context_block(slug, current_revision=rec.revision)}\n"
        )
        return ToolResult(
            out.rstrip(),
            display=("Updated the app — the script needs a fix."
                     if report.findings
                     else (f"Updated the app: {reason}" if reason
                           else "Updated the app.")),
        )

    async def _bash(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        command = args.get("command", "")
        title = self._existing_title(slug)

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        # No `detail=command`. The command is the model's working note —
        # `wc -c nokia-snake.html` was appearing where the user is told what
        # is happening to their app.
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="verify",
            status="running",
        )
        try:
            code, output = await run_in_app_dir(
                command, timeout=DEFAULT_TIMEOUT_SECONDS,
            )
        except ShellRefusal:
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="verify",
                status="failed", detail="that check couldn't run",
            )
            raise
        # The step is "did this inspection happen", and it did. A non-zero
        # exit means grep found nothing — a normal, useful answer — and
        # marking the step FAILED for it is how a card came to carry a red ✗
        # for a check that worked perfectly. The exit code still goes to the
        # model, which is the only reader that can interpret it.
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="verify",
            status="done", detail="looked inside the file",
        )
        body = output.rstrip() or "(no output)"
        # The model gets the exit code and the raw output — that is the whole
        # point of the tool. The user gets neither: "exit 1" and a grep dump
        # are not a status, and both were appearing as one.
        return ToolResult(f"exit {code}\n{body}", display="Checked the app.")

    async def _present(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        title = self._existing_title(slug)
        html_path = store.app_path(slug)

        # ── The brief must exist ──────────────────────────────────────
        # Checked FIRST, before a browser is launched, because it is the one
        # requirement that costs nothing to check and cannot be satisfied by
        # this call. `create_app_file` requires a brief, so only two apps can
        # reach here without one: an app built before round 20 whose backfill
        # could not reach a model, and one whose brief write failed. Both are
        # one `edit_app_file(brief=…)` away, and the message says so.
        #
        # This is a gate rather than an instruction on purpose. An app with no
        # brief is an app whose next edit is a guess, and the round exists to
        # end that — a rule the model is merely asked to follow is a rule that
        # holds until the first busy turn.
        existing_brief = appskill.read(slug)
        if existing_brief is None or not existing_brief.has_narrative:
            raise InternalRefusal(
                f"{slug} has no brief yet, so it was not published. Call "
                f"edit_app_file with the `brief` argument (any small edit will "
                f"do, or re-send the file with create_app_file) and describe "
                f"the app: what it is, who it is for, the problem it solves, "
                f"its flows, what each control does, and why it looks the way "
                f"it does. It is internal — the user never sees it — and it is "
                f"what you will read before every future change.",
                display="Nearly there — I'm just finishing this off.",
            )

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)

        # ── The gate ──────────────────────────────────────────────────
        # Publishing is the moment the app stops being the model's problem
        # and becomes the user's, so it is the right and only place to
        # insist that it works. Round 18's whole complaint — a card reading
        # "100% · App built" over a game that could not be played — was
        # possible because nothing between generation and this line had ever
        # opened the file in a browser.
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="verify",
            status="running",
        )
        report = await verify_mod.verify_app(store.read_app(slug), deep=True)
        # Budget discipline (round 24): breakage always refuses; polish
        # refuses at most GATE_MAX_REFUSALS times per app, then ships with
        # the remaining notes named. `remaining_notes` accumulates what the
        # gate waved through so the publish can say so honestly.
        hard = [f for f in report.findings if f.kind in _HARD_FINDING_KINDS]
        soft = [f for f in report.findings if f.kind not in _HARD_FINDING_KINDS]
        refusals = self._gate_refusals.get(slug, 0)
        over_budget = refusals >= GATE_MAX_REFUSALS
        remaining_notes: List[str] = []
        if report.findings and (hard or not over_budget):
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="verify",
                status="failed", detail=report.summary(),
                # The gate refusing is the designed loop: the model is handed
                # the errors with real line numbers and told to come back.
                # Before round 25 this held the WHOLE job at `failed` — through
                # the model's read, its edit and the write-back — so the card
                # was a red "Couldn't build" pill for the entire repair.
                recoverable=True,
            )
            self._gate_refusals[slug] = refusals + 1
            # Not presented, so no app card, no completed job and no "Built"
            # badge. The model gets the errors with their real line numbers
            # and the instruction to come back.
            checks_left = max(0, GATE_MAX_REFUSALS - refusals - 1)
            raise AppStoreError(
                "the app is not working yet, so it was not published:\n"
                + report.as_error()
                + "\nFix these with edit_app_file and call present_app again."
                + (
                    f"\nFix ALL of them in ONE edit pass — {checks_left} more "
                    "check(s) before the app ships with its remaining rough "
                    "edges listed instead of being held back."
                    if checks_left and not hard else ""
                )
            )
        if soft and over_budget:
            remaining_notes.extend(f.as_text() for f in soft)
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="verify",
                status="done",
                detail=f"opened it — {len(soft)} rough edge"
                f"{'' if len(soft) == 1 else 's'} noted",
            )
        else:
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="verify",
                status="done", detail=report.summary(),
            )

        # ── Somebody looks at it ──────────────────────────────────────
        # Round 20, item 3. Everything above this line has established that
        # the app RUNS. None of it can see that the score is white on white,
        # that the board is clipped at the bottom, or that the panel the model
        # was proud of rendered as an empty grey rectangle. Those need an eye,
        # so the screenshot taken during the run goes to one.
        #
        # The findings are the model's repair list and are never shown to the
        # user; the step's detail says only how many there are. And a look
        # that could not happen says so — `Look.ran` is false, the step
        # reports "couldn't look at it here", and the publish is NOT refused
        # on the strength of a review that never ran.
        change = "; ".join(self._pending_changes.get(slug, []))
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="look",
            status="running",
        )
        look = await vision_mod.review_screenshot(
            report.screenshot,
            user_id=ctx.user_id,
            title=title,
            purpose=self._purpose_line(existing_brief),
            change=change,
            # The screenshot is taken AFTER the gate pressed the start
            # control (when it found one). Told nothing, the reviewer's
            # "a start screen is a normal state" rule blanket-approved a
            # start overlay that had failed to dismiss — the recorded
            # Ping-Pong. The label makes the shot's timing part of the ask.
            pressed_start=report.pressed_start or "",
        )
        # A look that did not happen resolves to SKIPPED here, at the moment
        # it is known — never `done` ("Checked the app looks right" on an app
        # nobody looked at is the overclaim this pipeline exists to end), and
        # never left `running`: the recorded card had this row spinning while
        # icon and publish completed BENEATH it, because the skip used to be
        # deferred to `finish_job` at build end. A skipped verification is an
        # infrastructure failure, not a shrug — with the browser shipped in
        # the image it must simply never happen, so when it does it logs at
        # error level.
        if look.ran:
            look_refuses = bool(look.problems) and not (
                self._gate_refusals.get(slug, 0) >= GATE_MAX_REFUSALS
            )
            if look.problems and not look_refuses:
                remaining_notes.extend(look.problems)
                await steps_mod.emit_step(
                    user_id=ctx.user_id, job_id=job_id, step_type="look",
                    status="done",
                    detail=f"looked at it — {len(look.problems)} thing"
                    f"{'' if len(look.problems) == 1 else 's'} left to polish",
                )
            else:
                await steps_mod.emit_step(
                    user_id=ctx.user_id, job_id=job_id, step_type="look",
                    status="failed" if look.problems else "done",
                    detail=look.summary(),
                    # The visual review finding something to fix is the loop
                    # too — the raise below hands the model the repair list
                    # (round 25, item 2).
                    recoverable=True,
                )
        else:
            skip_reason = look.reason or "couldn't look at it here"
            if report.downgrade_reason and skip_reason == "no screenshot was captured":
                # The screenshot is missing BECAUSE the browser pass was
                # downgraded — name the real cause, not the symptom.
                skip_reason = report.downgrade_reason
            logger.error(
                "[app_html] LOOK SKIPPED for %s — %s (verification did not "
                "run; infrastructure defect if the renderer is unavailable)",
                slug, skip_reason,
            )
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="look",
                status="skipped", detail=skip_reason,
            )
        if look.problems and not remaining_notes:
            refusals = self._gate_refusals.get(slug, 0)
            self._gate_refusals[slug] = refusals + 1
            checks_left = max(0, GATE_MAX_REFUSALS - refusals - 1)
            raise AppStoreError(
                "the app runs, but it does not look right, so it was not "
                "published:\n" + look.as_error()
                + "\nFix these with edit_app_file and call present_app again."
                + (
                    f"\nFix ALL of them in ONE edit pass — {checks_left} more "
                    "check(s) before the app ships with its remaining rough "
                    "edges listed instead of being held back."
                    if checks_left else ""
                )
            )

        # ── The preview, from the same browser run (round 23) ─────────
        # The gate has just opened the app at the phone viewport; the frame
        # it took BEFORE pressing anything — the start screen — is the app's
        # face, and it becomes the card's preview. This replaces the client
        # re-rendering the app under its own (mistimed) capture conditions,
        # which is what produced the recorded full-width-header /
        # collapsed-left-column cards. Best-effort by contract: a preview is
        # never worth a publish, and a card with no picture shows its
        # placeholder rather than a broken frame.
        try:
            snap = report.cover or report.screenshot
            if snap and store.write_preview(slug, snap):
                logger.info("[app_html] preview for %s: %d bytes", slug, len(snap))
        except Exception:  # noqa: BLE001 - a picture is never worth a publish
            logger.warning("[app_html] preview persist failed for %s", slug,
                           exc_info=True)

        size = os.path.getsize(html_path)
        record = store.upsert_record(slug, title, size, bump_revision=False,
                                     presented=True)

        # ── The mark, BEFORE anything is handed over ──────────────────
        # Round 21, item 1. This used to be a side effect tucked inside the
        # `present` step: no phase of its own, so a slow or failed drawing was
        # invisible, and — the part that showed — the app card and the Files
        # row could be published while the icon route still had nothing to
        # answer with. Whatever asked first got the deterministic monogram,
        # which the route then STORED, and the app wore a holding mark it had
        # not earned.
        #
        # So it is its own step and it happens first. `ensure_icon` reads the
        # app's palette out of the file and redraws when the app's identity
        # (title, purpose, colours) has moved — which is what makes an edited
        # app get an updated mark rather than last week's. It still cannot
        # fail a publish: a container that cannot reach a model gets the
        # monogram, recorded as provisional so a later run upgrades it.
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="logo",
            status="running",
        )
        icon_source = ""
        timed_out = False
        try:
            # A PHASE BUDGET, not a hang guard (round 24): `ensure_icon`'s own
            # internal budgets sum to ~140s (a 20s subject call plus three 40s
            # draw attempts), and every second of it is a row spinning on the
            # user's card with the app already built. The recorded 1:26 build
            # showed a step spinning ~2 minutes and the user could not tell
            # whether anything was happening. No mark is worth that, so the
            # phase is cut off and SAYS it was.
            _svg, icon_source = await asyncio.wait_for(
                logo_mod.ensure_icon(
                    slug, title=title, purpose=self._purpose_line(existing_brief),
                    user_id=ctx.user_id, html=store.read_app(slug),
                ),
                timeout=LOGO_PHASE_BUDGET_S,
            )
            logger.info("[app_html] icon for %s: %s", slug, icon_source)
        except asyncio.TimeoutError:
            timed_out = True
            logger.warning(
                "[app_html] icon step for %s exceeded its %ss phase budget",
                slug, LOGO_PHASE_BUDGET_S,
            )
        except Exception:  # noqa: BLE001 - an icon is never worth a publish
            logger.warning("[app_html] icon step failed for %s", slug, exc_info=True)
        if icon_source:
            # `done` for a monogram as well as a designed mark: the app HAS a
            # mark either way, and `failed` here would flip the whole build
            # card to failed (see `emit_step`) over something cosmetic that
            # the next publish upgrades by itself.
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="logo",
                status="done",
                detail=("drew a fresh mark in the app's colours"
                        if icon_source == "model"
                        else "kept the mark it already had"
                        if icon_source == "kept"
                        else "gave it a simple mark for now"),
            )
        else:
            # The drawing raised. Resolve NOW — a row left `running` sits
            # spinning while publish completes beneath it (the recorded
            # bottom-up card); a phase that never reported back is not work
            # outstanding and not work done, so it is skipped, with words.
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="logo",
                status="skipped",
                detail=("took too long, so the app kept its letter mark"
                        if timed_out else "couldn't draw one this time"),
            )

        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="present",
            status="running",
        )
        app_id = await steps_mod.upsert_app_row(
            user_id=ctx.user_id, slug=slug, title=title,
            html_path=html_path, size_bytes=size, job_id=job_id,
        )
        # …and into Files, now, rather than whenever the next listing happens
        # to fall outside the library sync's throttle window. See
        # `library_service.register_app_file`.
        await steps_mod.register_in_library(user_id=ctx.user_id, slug=slug)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="present",
            status="done",
            detail=f"revision {record.revision} is live — card and viewer updated",
        )
        # The memory follows the app here too, not only on write. Round 21,
        # item 4: a publish is the moment the user's copy of this app changes,
        # so it is exactly the marker the next editor needs in order to tell
        # "what I changed" from "what they have actually seen". It costs one
        # version and one line.
        self._save_brief(
            slug, title=title, html=store.read_app(slug),
            revision=record.revision, narrative=None,
            history_line=f"published revision {record.revision}"
                         + (f" — {change}" if change else ""),
        )
        await steps_mod.finish_job(ctx.user_id, job_id)
        await steps_mod.announce_ready(
            user_id=ctx.user_id, job_id=job_id, app_id=app_id, title=title, slug=slug,
        )
        # Published, so the edits it contains are no longer pending. Cleared
        # here and not in `_edit`, because "what changed since the user last
        # saw this app" is exactly the window between two publishes.
        self._pending_changes.pop(slug, None)
        self._gate_refusals.pop(slug, None)

        # Round 18, items 2 and 6. This used to hand back an internal URL
        # path, an `[[open_app:…]]` chip and two sentences of stage direction
        # ("Tell the user what it does…", "Do not paste the HTML…"). All
        # three were rendered to the user as the agent's own words: an
        # internal route, a token nobody outside this codebase can read, and
        # instructions addressed to a model. The chip also produced a SECOND
        # open button under the card that already has one.
        #
        # What replaces it is one sentence of fact. Everything the model is
        # supposed to do next lives in the system prompt, where the user
        # cannot be shown it.
        if remaining_notes:
            # Over the polish budget, published anyway — but never silently.
            # The model is told what it shipped with so it can (a) tell the
            # user honestly and (b) fix the list from this report in ONE edit
            # pass without another round of the gate.
            notes = "\n".join(f"  - {n}" for n in remaining_notes)
            return ToolResult(
                f"Published '{title}'. The app card is in the chat, ready to "
                f"open. It shipped with {len(remaining_notes)} known rough "
                f"edge{'' if len(remaining_notes) == 1 else 's'} — tell the "
                f"user, and if you can fix them from this list alone, do it "
                f"in one edit pass and present again:\n{notes}",
                display=f"{title} is ready.",
            )
        return ToolResult(
            f"Published '{title}'. The app card is in the chat, ready to open.",
            display=f"{title} is ready.",
        )
