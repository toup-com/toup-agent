"""
Layer 2 context builder — consolidated context injection for app-channel messages.

Replaces the duplicated inline context-building in ws_chat.py and apps_proxy.py
(Risk 5 from the Checkpoint 2c HTTP endpoint audit).

Both the WS path (ChatPage → agent) and the HTTP bridge path (iframe Orb → platform
proxy → agent) now call build_layer2_context() to get structured context, then
call .render() to produce the final string prepended to the user's message.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class Layer2Context:
    """Structured context for app-channel messages.

    base fields (always populated): app identity, tools, behavior rules.
    layer2 fields (only when is_layer2=True): L1 history, STEP 1-4 instructions.
    """
    app_name: str
    slug_safe: str        # slug with hyphens → underscores (for tool name prefix)
    app_slug: str         # original slug (for [[open_app:slug]] chips)
    preview_url: str
    available_tools: List[str]
    behavior_rules: List[str]
    layer1_history: Optional[Dict[str, Any]] = None  # prompt, checkpoint, plan, description
    layer2_instructions: Optional[str] = None         # STEP 1-4 block
    recent_changes: List[Dict[str, Any]] = field(default_factory=list)

    def render(self, is_layer2: bool = False) -> str:
        """Produce the final context string to prepend to the user message."""
        parts = [self._render_base_context()]
        if is_layer2 and self.layer2_instructions:
            parts.append(self._render_layer2_block())
        return "[" + "\n".join(parts) + "]"

    def _render_base_context(self) -> str:
        lines = [
            f"CONTEXT: The user is chatting from inside their '{self.app_name}' app. "
            f"You are their in-app assistant.",
        ]
        for rule in self.behavior_rules:
            lines.append(f"- {rule}")
        lines.append("- You have these app tools:")
        for tool in self.available_tools:
            lines.append(f"  {tool}")
        return "\n".join(lines)

    def _render_layer2_block(self) -> str:
        parts = ["\n- LAYER 2 CUSTOMIZATION MODE activated."]

        if self.layer1_history:
            l1_lines = []
            if self.layer1_history.get("prompt"):
                l1_lines.append(f'  LAYER 1 BUILD REQUEST: "{self.layer1_history["prompt"]}"')
            if self.layer1_history.get("choices"):
                l1_lines.append(f'  LAYER 1 CHOICES: {self.layer1_history["choices"]}')
            if self.layer1_history.get("extra_context"):
                l1_lines.append(f'  LAYER 1 EXTRA CONTEXT: {self.layer1_history["extra_context"][:500]}')
            if self.layer1_history.get("plan_summary"):
                l1_lines.append(f'  LAYER 1 APP PLAN: {self.layer1_history["plan_summary"][:500]}')
            if self.layer1_history.get("description"):
                l1_lines.append(f'  APP DESCRIPTION: {self.layer1_history["description"][:300]}')

            if l1_lines:
                parts.append(
                    "\n  ── WHAT LAYER 1 ALREADY ESTABLISHED (DO NOT RE-ASK ANY OF THIS) ──\n"
                    + "\n".join(l1_lines) + "\n"
                    "  ── END LAYER 1 CONTEXT ──\n\n"
                    "  The above was ALREADY asked and answered during Layer 1. The app was ALREADY built with these parameters.\n"
                    "  You MUST NOT ask about any of the above topics again. They are settled.\n"
                )

        parts.append(self.layer2_instructions or "")
        return "\n".join(parts)


def _build_behavior_rules(app_name: str, slug_safe: str, app_slug: str, preview_url: str) -> List[str]:
    """Standard behavior rules for app-channel messages."""
    return [
        "Be conversational and helpful. Greet naturally when they say hi.",
        "NEVER mention internal details (SQLite, bridges, connections, file paths, agent infrastructure).",
        f"NEVER give localhost URLs to the user. The app preview URL is: {preview_url}",
        f"When the user asks to change something in the app (UI, content, settings), "
        f"use write_file/edit_file to make the change, then call restart to apply it.",
        f"After fixing or restarting, give the user a clickable [[open_app:{app_slug}]] chip.",
        "Suggest helpful actions as [[Button Label]] chips.",
    ]


def _build_available_tools(slug_safe: str) -> List[str]:
    """List of app-specific tools available to the agent."""
    return [
        f"app_{slug_safe}__navigate (change screens),",
        f"app_{slug_safe}__read_file / app_{slug_safe}__write_file (edit the app),",
        f"app_{slug_safe}__edit_file (search/replace edits),",
        f"app_{slug_safe}__query_db (read/write app data),",
        f"app_{slug_safe}__restart (restart the app after code changes — ALWAYS call this after editing files).",
    ]


def _build_layer2_instructions(slug_safe: str) -> str:
    """STEP 1-4 instructions for Layer 2 customization mode (richer version per D2)."""
    return (
        f"  CRITICAL: Layer 2 is an EDIT LAYER on top of Layer 1. You are NOT rebuilding the app.\n"
        f"  Layer 1 already created a functional app. Your job is to ENHANCE, FIX, and EXTEND it.\n"
        f"  NEVER propose replacing or rebuilding what Layer 1 created. Only improve it.\n\n"
        f"  STEP 1 (SILENT): Use app_{slug_safe}__read_file to read the app's key files — "
        f"App.tsx, main screen components, database/seed data, config/constants. "
        f"Be EFFICIENT: read 3-5 key files, not every single file. "
        f"Do NOT tell the user you are reading files. Do NOT expose paths or technical details.\n"
        f"  As you read, identify ONLY things Layer 1 did poorly or left incomplete:\n"
        f"  - Placeholder/demo data that should be real content\n"
        f"  - Shallow features that need deeper implementation\n"
        f"  - Generic defaults that should be personalized\n"
        f"  - Missing functionality that would make the app truly useful\n"
        f"  - Hardcoded content that should be dynamic\n\n"
        f"  STEP 2: Ask 10+ questions that reference SPECIFIC things you found in the code.\n"
        f"  Each question MUST cite something concrete from the code (a number, a file, a feature).\n"
        f"  Example: 'I found 500 vocabulary words but they are all general English — should I focus them on academic passages for your field?'\n"
        f"  Example: 'The study plan is a fixed 90-day schedule — want me to make it adaptive based on your quiz performance?'\n"
        f"  Example: 'The reading section has 10 passages but they are placeholder text — should I generate real IELTS-style passages?'\n"
        f"  FORBIDDEN TOPICS (Layer 1 already handled these): target score, test date, study hours, color theme, "
        f"app name, which test type (academic/general), basic preferences, weekly availability.\n"
        f"  Every question MUST have [[option]] buttons on the NEXT LINE — buttons must be inline with their question, "
        f"NOT collected at the end.\n\n"
        f"  STEP 3 (AFTER user answers): IMMEDIATELY begin editing the app using write_file/query_db.\n"
        f"  Do NOT just acknowledge the answers or offer action buttons — you MUST apply actual code changes.\n"
        f"  Do NOT use memory_store to save preferences. Do NOT say 'let me store your preferences'.\n"
        f"  INSTEAD: Use write_file to rewrite app files with the user's choices applied.\n"
        f"  Replace placeholder data with real content, upgrade features, add algorithms.\n"
        f"  Be EFFICIENT — use write_file to write complete files, batch related changes together.\n"
        f"  Show brief progress after each edit.\n"
        f"  IMPORTANT: You have a limited number of tool calls. Be efficient — don't waste iterations "
        f"on unnecessary reads. Combine related edits into single write_file calls when possible.\n\n"
        f"  STEP 4 (COMPLETION): Give a BRIEF human-friendly summary of what you customized.\n"
        f"  NEVER expose internal operations (memory storage, file reading, database queries) to the user.\n"
        f"  NEVER say 'let me store' or 'let me save to memory' — the user does not care about internals.\n"
    )


async def _load_layer1_history(app_id: str, db: AsyncSession, app: Any) -> Optional[Dict[str, Any]]:
    """Load Layer 1 build history for the app."""
    from app.db.models import BuildJob

    history: Dict[str, Any] = {}

    try:
        result = await db.execute(
            select(BuildJob)
            .where(BuildJob.app_id == app_id, BuildJob.layer == 1)
            .order_by(BuildJob.created_at.desc())
            .limit(1)
        )
        l1_job = result.scalar_one_or_none()
        if l1_job:
            history["prompt"] = l1_job.prompt
            if l1_job.checkpoint_json:
                try:
                    ckpt = json.loads(l1_job.checkpoint_json)
                    pc = ckpt.get("plan_context") or {}
                    if pc:
                        parts = []
                        for k in ("screens", "features"):
                            if pc.get(k):
                                parts.append(f"{k}: {', '.join(pc[k])}")
                        if pc.get("db_type"):
                            parts.append(f"database: {pc['db_type']}")
                        if pc.get("design_notes"):
                            parts.append(f"design: {pc['design_notes']}")
                        if parts:
                            history["choices"] = "; ".join(parts)
                    ec = ckpt.get("extra_context")
                    if ec:
                        history["extra_context"] = ec
                except (json.JSONDecodeError, TypeError):
                    pass
    except Exception as e:
        logger.debug("Could not load Layer 1 build job: %s", e)

    # App plan
    if getattr(app, 'plan_json', None):
        try:
            plan = json.loads(app.plan_json)
            if plan.get("summary"):
                history["plan_summary"] = plan["summary"]
        except (json.JSONDecodeError, TypeError):
            pass

    # App description
    if getattr(app, 'description', None):
        history["description"] = app.description

    return history if history else None


async def _load_recent_changes(app_id: str, db: AsyncSession) -> List[Dict[str, Any]]:
    """Load recent customization changes from the most recent build job."""
    from app.db.models import BuildJob

    try:
        result = await db.execute(
            select(BuildJob)
            .where(BuildJob.app_id == app_id)
            .order_by(BuildJob.created_at.desc())
            .limit(1)
        )
        job = result.scalar_one_or_none()
        if job and job.layer2_changes_json:
            changes = json.loads(job.layer2_changes_json)
            if isinstance(changes, list):
                return changes
    except Exception as e:
        logger.debug("Could not load recent changes: %s", e)

    return []


async def build_layer2_context(
    app_id: str,
    db: AsyncSession,
    is_layer2: bool = False,
) -> Optional[Layer2Context]:
    """Build structured context for an app-channel message.

    Always returns base context (tools, behavior rules) for app messages.
    Adds Layer 2 context (L1 history, STEP 1-4 instructions) when is_layer2=True.

    Args:
        app_id: The app's UUID.
        db: Active async DB session (caller-provided, not self-created).
        is_layer2: Whether Layer 2 customization mode is active for this message.

    Returns:
        Layer2Context with structured fields and a render() method, or None if
        the app can't be found.
    """
    from app.db.models import App

    try:
        app = await db.get(App, app_id)
    except Exception as e:
        logger.warning("build_layer2_context: failed to load app %s: %s", app_id[:8], e)
        return None

    if not app:
        return None

    slug_safe = app.slug.replace('-', '_')
    preview_url = f"https://toup.ai/workspace/apps/{app.slug}"

    ctx = Layer2Context(
        app_name=app.name,
        slug_safe=slug_safe,
        app_slug=app.slug,
        preview_url=preview_url,
        available_tools=_build_available_tools(slug_safe),
        behavior_rules=_build_behavior_rules(app.name, slug_safe, app.slug, preview_url),
    )

    if is_layer2:
        ctx.layer1_history = await _load_layer1_history(app_id, db, app)
        ctx.layer2_instructions = _build_layer2_instructions(slug_safe)
        ctx.recent_changes = await _load_recent_changes(app_id, db)

    return ctx
