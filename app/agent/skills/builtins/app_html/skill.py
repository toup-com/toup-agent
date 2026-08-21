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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def on_load(self) -> None:
        """Create the app root and materialise the design skill on disk.

        The packaged copy is authoritative — it is rewritten on every boot so
        an image upgrade actually ships the new guidance instead of leaving a
        stale copy on the volume forever.
        """
        try:
            store.ensure_root()
        except OSError:
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
        design = self._design_path
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
                    "then 768 and 1280. NEVER touch localStorage/cookies/fetch without "
                    "a try/catch — the app runs in a sandboxed opaque origin where they "
                    "throw.\n"
                    f"BEFORE writing any UI, read the design skill: "
                    f"read_file(path='{design}'). Do not skip it — it is what stops the "
                    "app looking generated.\n"
                    "WORKFLOW: one big write here, then surgical edit_app_file changes, "
                    "then bash_app to verify, then present_app. Calling this again for "
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
                    "indentation and line breaks."
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
                    "and posts the app card in chat. Call this once the app is finished "
                    "and verified — and again after a round of edits so the user sees "
                    "the new version."
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
            "It runs in a sandboxed frame on an opaque origin, so localStorage, "
            "cookies and network calls THROW — guard every one and keep state in "
            "memory.\n"
            f"ALWAYS read the design skill before writing UI: "
            f"read_file(path='{self._design_path}'). It is what keeps an app from "
            "looking generated.\n"
            "Loop: app_html__create_app_file (one big write) → app_html__edit_app_file "
            "(small exact-string edits; view_app_file first) → app_html__bash_app "
            "(verify) → app_html__present_app (show the user). Never hand-scaffold an "
            "app with exec/write_file."
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
            return f"ERROR: {exc}"
        except OSError as exc:
            logger.warning("[app_html] %s filesystem error", tool_name, exc_info=True)
            return f"ERROR: filesystem error: {exc}"

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
        if record is None or not store.exists(slug):
            known = ", ".join(sorted(store.read_manifest())) or "none yet"
            raise AppStoreError(
                f"no app named {slug!r}. Apps that exist: {known}. "
                f"Use create_app_file to make a new one."
            )
        return record.title or slug

    async def _create(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        title = (args.get("title") or slug).strip()
        html = args.get("html")

        # Validate BEFORE opening the job. A stub call or a blocked CDN
        # reference is a call that never should have happened; minting a job
        # for it would leave a permanently failed "Build: …" card in chat
        # for an app that was never started.
        store.validate_html(html)

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="create",
            label=f"Creating file: {title}", status="running",
        )
        try:
            record, warnings = store.write_app(slug, title, html)
        except AppStoreError as exc:
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="create",
                label=f"Creating file: {title}", status="failed", detail=str(exc),
            )
            raise
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="create",
            label=f"Creating file: {title}", status="done",
            detail=f"{record.size_bytes} bytes",
        )

        out = (
            f"Wrote {slug}.html ({record.size_bytes:,} bytes, revision "
            f"{record.revision}) to the app directory.\n"
        )
        if warnings:
            out += "Warnings:\n" + "".join(f"  - {w}\n" for w in warnings)
        out += (
            "Next: verify with app_html__bash_app, make any fixes with "
            "app_html__edit_app_file, then call app_html__present_app to show "
            "the user."
        )
        return out

    async def _view(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        title = self._existing_title(slug)

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="review",
            label="Reading current app before editing", status="running",
        )
        content = store.read_app(slug)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="review",
            label="Reading current app before editing", status="done",
            detail=f"{len(content)} chars",
        )

        if len(content) <= VIEW_INLINE_LIMIT:
            # Raw, no line numbers: edit_app_file matches byte-for-byte, and
            # a "  42\t" gutter is exactly the kind of thing that ends up
            # copied into old_string and never matches.
            return content
        head = content[:VIEW_HEAD_CHARS]
        tail = content[-VIEW_TAIL_CHARS:]
        omitted = len(content) - VIEW_HEAD_CHARS - VIEW_TAIL_CHARS
        return (
            f"{head}\n\n"
            f"[… {omitted:,} characters omitted — {slug}.html is {len(content):,} "
            f"chars. Use app_html__bash_app with grep -n to locate the exact text "
            f"you want to edit in the omitted region …]\n\n"
            f"{tail}"
        )

    async def _edit(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        reason = (args.get("reason") or "").strip()
        title = self._existing_title(slug)
        label = f"Editing file: {reason}" if reason else "Editing file"

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="edit",
            label=label, status="running",
        )
        try:
            rec, delta = store.edit_app(
                slug, args.get("old_string", ""), args.get("new_string", ""),
            )
        except AppStoreError as exc:
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="edit",
                label=label, status="failed", detail=str(exc),
            )
            raise
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="edit",
            label=label, status="done", detail=f"{delta:+d} bytes",
        )

        out = (
            f"Edited {slug}.html — {delta:+,} bytes, now {rec.size_bytes:,} "
            f"(revision {rec.revision}).\n"
        )
        for w in getattr(rec, "warnings", []) or []:
            out += f"  - warning: {w}\n"
        out += "Call app_html__present_app when the change is ready to show."
        return out

    async def _bash(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        command = args.get("command", "")
        title = self._existing_title(slug)

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="verify",
            label="Verifying changes", status="running", detail=command[:120],
        )
        try:
            code, output = await run_in_app_dir(
                command, timeout=DEFAULT_TIMEOUT_SECONDS,
            )
        except ShellRefusal as exc:
            await steps_mod.emit_step(
                user_id=ctx.user_id, job_id=job_id, step_type="verify",
                label="Verifying changes", status="failed", detail=str(exc)[:200],
            )
            raise
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="verify",
            label="Verifying changes", status="done" if code == 0 else "failed",
            detail=f"exit {code}",
        )
        body = output.rstrip() or "(no output)"
        # A non-zero exit is INFORMATION (grep found nothing), not a tool
        # failure — do not prefix it with ERROR:, or the model reads a
        # successful negative check as a broken tool.
        return f"exit {code}\n{body}"

    async def _present(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        slug = store.normalise_slug(args.get("slug", ""))
        title = self._existing_title(slug)
        html_path = store.app_path(slug)
        size = os.path.getsize(html_path)
        record = store.upsert_record(slug, title, size, bump_revision=False,
                                     presented=True)

        job_id = await steps_mod.ensure_job(ctx.user_id, slug, title)
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="present",
            label="Presented app", status="running",
        )
        app_id = await steps_mod.upsert_app_row(
            user_id=ctx.user_id, slug=slug, title=title,
            html_path=html_path, size_bytes=size, job_id=job_id,
        )
        await steps_mod.emit_step(
            user_id=ctx.user_id, job_id=job_id, step_type="present",
            label="Presented app", status="done", detail=f"revision {record.revision}",
        )
        await steps_mod.finish_job(ctx.user_id, job_id)
        await steps_mod.announce_ready(
            user_id=ctx.user_id, job_id=job_id, app_id=app_id, title=title,
        )

        return (
            f"Presented '{title}' — the app card is now in the chat and the app "
            f"is live at /api/artifacts/{slug} ({size:,} bytes, revision "
            f"{record.revision}).\n"
            f"Tell the user what it does in one or two sentences and offer the "
            f"[[open_app:{slug}]] chip. Do not paste the HTML or the file path."
        )
