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

import logging
import os
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
from app.agent.skills.builtins.app_html import steps as steps_mod
from app.agent.skills.builtins.app_html import store
from app.agent.skills.builtins.app_html import verify as verify_mod
from app.agent.skills.builtins.app_html.shell import (
    DEFAULT_TIMEOUT_SECONDS,
    ShellRefusal,
    run_in_app_dir,
)
from app.agent.skills.builtins.app_html.store import AppStoreError

logger = logging.getLogger(__name__)

DESIGN_SKILL_FILENAME = "toup-frontend-design.md"

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


class AppHtmlSkill(Skill):
    """Single-file HTML artifact pipeline."""

    meta = SkillMeta(
        name="app_html",
        version="1.0.0",
        description="Build apps as one self-contained HTML file (no build step)",
        author="toup",
    )

    def __init__(self) -> None:
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
                    "properties": {
                        "slug": {
                            "type": "string",
                            "description": "URL-safe id: 1-60 lowercase letters, digits and hyphens (e.g. 'budget-tracker').",
                        },
                        "title": {
                            "type": "string",
                            "description": "Human name shown on the app card, e.g. 'Budget Tracker'.",
                        },
                        "html": {
                            "type": "string",
                            "description": "The COMPLETE document: <!doctype html> through </html>, with all CSS and JS inlined. Not a fragment, not a placeholder.",
                        },
                    },
                    "required": ["slug", "title", "html"],
                },
            },
            {
                "name": "app_html__view_app_file",
                "description": (
                    "Read an app's current HTML exactly as it is on disk. Call this "
                    "before every edit_app_file — old_string must match the file "
                    "byte-for-byte, and you cannot match text you have not looked at "
                    "in this turn."
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
                            "description": "Short phrase for the progress card, e.g. 'add dark mode toggle'.",
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
                    "The app is opened in a real browser at 390x844 first and refused "
                    "if anything throws OR if a control renders under 44x44, text "
                    "under 12px, or the page scrolls sideways — so this is also the "
                    "check that it works and can be used with a thumb. If it comes "
                    "back with problems, fix them with edit_app_file and call this "
                    "again."
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
            "Loop: app_html__create_app_file (one big write) → app_html__edit_app_file "
            "(small exact-string edits; view_app_file first) → app_html__present_app "
            "(show the user). app_html__bash_app is there when you want to look "
            "something up in the file; it is not a required step, because the "
            "write and the publish check the app themselves. Never hand-scaffold "
            "an app with exec/write_file.\n"
            "If a check comes back with problems, FIX THEM AND CALL AGAIN. That is "
            "the whole recovery procedure — do not report a failure to the user "
            "and stop, and never present an app that has not come back clean.\n"
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
            "browser and tells you whether it still runs. Both of those come back "
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
        store.validate_html(html)
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
        if len(content) <= VIEW_INLINE_LIMIT:
            # Raw, no line numbers: edit_app_file matches byte-for-byte, and
            # a "  42\t" gutter is exactly the kind of thing that ends up
            # copied into old_string and never matches.
            return ToolResult(content, display="Read the app.")
        head = content[:VIEW_HEAD_CHARS]
        tail = content[-VIEW_TAIL_CHARS:]
        omitted = len(content) - VIEW_HEAD_CHARS - VIEW_TAIL_CHARS
        return ToolResult(
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

        # `+0 bytes` was the old detail for a same-length edit — a change the
        # user could see in the app, described to them as nothing having
        # happened. The reason the model gave for the edit is both true and
        # worth reading; the revision is the fallback when it gave none.
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="edit",
            status="failed" if report.findings else "done",
            detail=(report.summary() if report.findings
                    else (reason or f"revision {rec.revision}")),
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
            status="done",
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
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="verify",
            status="failed" if report.findings else "done",
            detail=report.summary(),
        )
        if report.findings:
            # Not presented, so no app card, no completed job and no "Built"
            # badge. The model gets the errors with their real line numbers
            # and the instruction to come back.
            raise AppStoreError(
                "the app is not working yet, so it was not published:\n"
                + report.as_error()
                + "\nFix these with edit_app_file and call present_app again."
            )

        size = os.path.getsize(html_path)
        record = store.upsert_record(slug, title, size, bump_revision=False,
                                     presented=True)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="present",
            status="running",
        )
        app_id = await steps_mod.upsert_app_row(
            user_id=ctx.user_id, slug=slug, title=title,
            html_path=html_path, size_bytes=size, job_id=job_id,
        )
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="present",
            status="done", detail=f"revision {record.revision}",
        )
        await steps_mod.finish_job(ctx.user_id, job_id)
        await steps_mod.announce_ready(
            user_id=ctx.user_id, job_id=job_id, app_id=app_id, title=title, slug=slug,
        )

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
        return ToolResult(
            f"Published '{title}'. The app card is in the chat, ready to open.",
            display=f"{title} is ready.",
        )
