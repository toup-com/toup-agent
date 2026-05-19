"""
Agent Runner — Core orchestration loop.

Flow:
  1. Build system prompt (identity + memories + runtime context)
  2. Apply context window management (compact if needed)
  3. Call LLM API with tool definitions
  4. If LLM requests tools → execute → feed results back → repeat
  5. Collect final text response
  6. Save conversation + extract memories

Features:
  - Context window management with auto-compaction
  - Error recovery with retry on transient failures
  - Image/vision support via OpenAI image_url content blocks
  - Detailed [AGENT] logging throughout
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone as _dt_timezone
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover — VPS Python 3.12 has it
    ZoneInfo = None  # type: ignore
from typing import Any, Callable, Coroutine, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context_manager import (
    needs_compaction,
    compact_messages,
    estimate_tokens,
    estimate_messages_tokens,
)
from app.agent.tool_definitions import get_agent_tools, get_extended_tools, get_doc_generation_tools
from app.agent.tool_executor import ToolExecutor
from app.agent.skills.loader import SkillLoader
from app.agent.query_intent import (
    classify_query_intent, filter_tools_by_intent, QueryIntent, INTENT_FULL,
)
from app.config import settings
from app.services.openai_agent_service import OpenAIAgentService, StreamEvent
from app.services.anthropic_service import AnthropicService
from app.services.model_router import classify_request, RoutingDecision
from app.agent.hooks import get_hook_bus, HookEvent

logger = logging.getLogger(__name__)

# Max retries on transient LLM errors
MAX_RETRIES = 2
RETRY_DELAY = 2.0  # seconds


def _is_claude_model(model: str) -> bool:
    """Check if a model name refers to an Anthropic Claude model."""
    return model.startswith("claude-")


# Vault CP4.1: channels that cannot render the CredentialConfirmCard today.
# `telegram` and `voice` are permanently excluded (their retention model
# makes chat-save the wrong UX). `mobile` is excluded until the RN
# renderer ships; CP4.4 removes `mobile` from this set.
VAULT_TOOL_CHANNEL_BLOCK = frozenset({"telegram", "voice", "mobile"})
VAULT_TOOL_NAME = "save_streaming_credential"


def strip_vault_tool_for_channel(tools, channel):
    if not channel or channel.strip().lower() not in VAULT_TOOL_CHANNEL_BLOCK:
        return tools
    return [
        t for t in tools
        if (t.get("name", "") or t.get("function", {}).get("name", "")) != VAULT_TOOL_NAME
    ]


@dataclass
class AgentResponse:
    """Final response from a single agent run."""
    text: str
    session_id: str
    day_chat_id: str = ""  # Day-as-Chat: parent day container (empty when feature flag off)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    model: str = ""
    processing_time_ms: int = 0
    memories_extracted: int = 0
    # Server-side UUID of the saved assistant Message row. Threaded
    # through the WS `done` payload so the frontend can stamp it on
    # the live-rendered bubble — without it, the bubble carries a
    # client-generated `msg-<timestamp>` id that doesn't match the
    # DB id, and the day-chat reload renders the same content twice
    # (once from the live append, once from the history fetch).
    asst_message_id: str = ""


OnTextChunk = Callable[[str], Coroutine[Any, Any, None]]
OnToolStart = Callable[[str], Coroutine[Any, Any, None]]
OnToolEnd = Callable[[str, str], Coroutine[Any, Any, None]]
OnToolProgress = Callable[[str, str], Coroutine[Any, Any, None]]
# Emitted per generated attachment. (message_id, attachment_dict)
OnAttachment = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, None]]
# Vault CP4: emitted when save_streaming_credential is invoked; carries
# the full frame payload (already shaped for the WS wire).
OnCredentialConfirmRequest = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


class AgentRunner:
    """
    Runs the agentic loop:  user message → (LLM ↔ tools)* → final response.
    """

    def __init__(
        self,
        llm_service: OpenAIAgentService,
        tool_executor: ToolExecutor,
        skill_loader: Optional["SkillLoader"] = None,
    ):
        self.llm = llm_service
        self.anthropic = AnthropicService()
        self.tools = tool_executor
        self.skill_loader = skill_loader
        # Core tools (static) — skill tools are added dynamically via property.
        # Document-generation tools (generate_pdf/docx/xlsx/pptx/md/html_to_pdf)
        # are only registered when the feature flag is on.
        self._core_tool_defs = get_agent_tools() + get_extended_tools()
        if getattr(settings, "feature_doc_generation", False):
            self._core_tool_defs += get_doc_generation_tools()
        self.max_iterations = settings.agent_max_tool_iterations
        self._session_model_override: Optional[str] = None  # Per-session model
        self._current_lane: str = 'main'  # Active execution lane
        self._idempotency_key: Optional[str] = None  # Current run idempotency key
        self._disabled_tool_names: set = set()  # Per-session disabled tools
        # Phase 5: Track retrieved memories for feedback loop
        self._last_retrieved_memories: List[Dict[str, Any]] = []
        # F6: per-turn memory health state captured during system_prompt
        # assembly. One structured log line is emitted from run() after
        # the prompt is built so operators have a single grep target for
        # "is memory working for user X right now?". Resets per turn.
        self._memory_health: Dict[str, Any] = {}

    @property
    def tool_defs(self) -> list:
        """Dynamically combine core tools + skill tools (picks up new app skills).

        T1g: when `use_connector_dispatch=True`, append connector tool
        definitions stored on the executor at boot
        (`tool_executor.mcp_tool_defs`, populated by agent_main.py from
        the platform's per-user filtered `tools/list`). Each connector
        tool is already user-scoped — the platform's T1f filter
        middleware only returns tools for connectors the user has
        actively connected. Skills win over connector tools by ordering
        (skills first → MCP tools won't shadow them in the LLM's view
        if a name happens to collide; the dispatcher's tool-name
        registry lints this at platform boot).
        """
        defs = list(self._core_tool_defs)
        if self.skill_loader:
            defs = defs + self.skill_loader.get_all_tool_definitions()
        # Defensive read — `use_connector_dispatch` is a newer setting and
        # this code can run on agent images deployed before the Settings
        # class added the attribute. Without getattr, any incoming
        # message (Telegram/WhatsApp/web) crashes with AttributeError on
        # such an agent. Default False keeps connector dispatch off until
        # the field is genuinely present.
        if getattr(settings, "use_connector_dispatch", False):
            mcp_tool_defs = getattr(self.tools, "mcp_tool_defs", None) or []
            if mcp_tool_defs:
                # Tools come from the platform as {name, description,
                # input_schema} — already in Anthropic shape (the OpenAI
                # adapter elsewhere normalises if needed). Skill+connector
                # collisions: registry lints at platform boot, but if
                # one slips through, the executor's dispatch order
                # (skills first) means the skill wins on call.
                defs = defs + list(mcp_tool_defs)
        # Apply per-session disabled filter (tools can be Anthropic or OpenAI format)
        if self._disabled_tool_names:
            defs = [
                t for t in defs
                if (t.get("name") or t.get("function", {}).get("name")) not in self._disabled_tool_names
            ]
        return defs

    # ------------------------------------------------------------------
    # Vibecoding DB registration
    # ------------------------------------------------------------------
    async def _register_vibecoding_session(
        self, user_id: str, prompt: str, session_id: Optional[str] = None,
    ) -> tuple:
        """Create App + BuildJob records for a vibecoding session.

        Returns (job_id, app_id) tuple. Idempotent per session — follow-up
        messages in the same session reuse the existing job (session continuity).
        """
        import uuid as _uuid
        import re as _re
        import os as _os
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob
        from app.agent.app_manager import APPS_DIR

        workspace = getattr(settings, 'agent_workspace_dir', None) or './workspace'
        vibecoding_dir = _os.path.join(_os.path.abspath(workspace), 'vibecoding')

        async with async_session_maker() as db:
            # Session continuity: reuse existing vibe_code job for this session
            if session_id:
                from sqlalchemy import select, and_
                existing = await db.execute(
                    select(BuildJob).where(
                        and_(
                            BuildJob.user_id == user_id,
                            BuildJob.status.in_(["running", "completed"]),
                        )
                    ).order_by(BuildJob.created_at.desc()).limit(10)
                )
                recent_jobs = existing.scalars().all()
                for j in recent_jobs:
                    if getattr(j, 'job_type', '') == 'vibe_code' and j.status in ("running", "completed"):
                        # Reuse: reopen if completed, append to running
                        if j.status == "completed":
                            j.status = "running"
                            await db.commit()
                        # Look up app_dir for the existing app
                        _existing_app = await db.get(App, j.app_id) if j.app_id else None
                        _existing_dir = _existing_app.app_dir if _existing_app else ""
                        return j.id, j.app_id, _existing_dir

            # Generate slug from prompt
            words = _re.sub(r'[^a-zA-Z0-9\s]', '', prompt).split()[:4]
            slug_base = '-'.join(w.lower() for w in words) or 'vibecoding-project'
            slug = slug_base[:50]

            # Deduplicate slug
            from sqlalchemy import select as sa_select
            result = await db.execute(sa_select(App.slug).where(App.slug.like(f"{slug}%")))
            existing_slugs = {row[0] for row in result.fetchall()}
            if slug in existing_slugs:
                counter = 1
                while f"{slug}-{counter}" in existing_slugs:
                    counter += 1
                slug = f"{slug}-{counter}"

            app_id = str(_uuid.uuid4())
            job_id = str(_uuid.uuid4())
            app_dir = _os.path.join(vibecoding_dir, slug)

            # Create the directory
            _os.makedirs(app_dir, exist_ok=True)

            name = slug.replace('-', ' ').title()

            app = App(
                id=app_id,
                user_id=user_id,
                name=name,
                description=prompt[:500] if prompt else "",
                slug=slug,
                status="building",
                source="vibecoding",
                app_dir=app_dir,
                platforms="web",
            )
            db.add(app)

            job = BuildJob(
                id=job_id,
                user_id=user_id,
                app_id=app_id,
                job_type="vibe_code",
                title=f"Vibecoding: {name}",
                prompt=prompt[:2000] if prompt else "",
                status="running",
                steps_json="[]",
                model="",
                layer=0,
                created_at=datetime.utcnow(),
            )
            db.add(job)
            await db.commit()

            logger.info(f"[VIBE] Registered vibecoding session: app={app_id[:8]} job={job_id[:8]} slug={slug} dir={app_dir}")
            return job_id, app_id, app_dir

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None,
        telegram_chat_id: Optional[int] = None,
        channel: Optional[str] = None,
        on_text_chunk: Optional[OnTextChunk] = None,
        on_tool_start: Optional[OnToolStart] = None,
        on_tool_end: Optional[OnToolEnd] = None,
        on_tool_progress: Optional[OnToolProgress] = None,
        on_attachment: Optional[OnAttachment] = None,
        on_credential_confirm_request: Optional[OnCredentialConfirmRequest] = None,
        media_paths: Optional[List[str]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        model_override: Optional[str] = None,
        thinking_budget: int = 0,
        idempotency_key: Optional[str] = None,
        save_user_message: bool = True,
        display_user_message: Optional[str] = None,
        client_tz: Optional[str] = None,
        app_id: Optional[str] = None,
        force_new_session: bool = False,
    ) -> AgentResponse:
        """
        Run the full agent loop for a single user message.
        """
        start = time.time()
        logger.info(f"[AGENT] === New agent run for user_id={user_id} ===")

        # F6: zero per-turn memory_health dict so a previous turn's counts
        # can't leak into this turn's [memory_health] log line.
        self._memory_health = {
            "retrieved": 0,
            "active_tasks": 0,
            "recent_days": 0,
            "summary_status": None,
            "summary_failure_reason": None,
            "today_summary_present": False,
        }

        # Pre-generate the assistant message ID so generate_* tools can emit
        # attachment WS events with a stable message_id before the message
        # is persisted. Used at line ~1649 when creating the assistant Message.
        asst_message_id = str(uuid.uuid4())
        # Reset pending attachments — belongs to this run only.
        self.tools.pending_attachments = []
        _attachments_emitted_count = 0

        # ── Classify query intent (lightweight, <1ms) ─────────────────
        t_classify = time.perf_counter()
        query_intent = classify_query_intent(user_message)
        logger.info(
            f"[PERF] query_intent: {(time.perf_counter() - t_classify) * 1000:.1f}ms → "
            f"category={query_intent.category}, "
            f"tools={len(query_intent.tool_names) or 'all'}"
        )

        # Set user context for memory tools and current chat
        self.tools.set_user_id(user_id)
        self.tools.set_chat_id(telegram_chat_id)
        self.tools.set_channel(channel)
        self.tools._on_tool_progress = on_tool_progress
        # Vault CP4: handler uses this to emit credential_confirm_request frames.
        self.tools._on_credential_confirm_request = on_credential_confirm_request

        # Hook: agent run starting
        _hb = get_hook_bus()
        await _hb.emit(HookEvent.BEFORE_AGENT_START, {
            "user_id": user_id, "session_id": session_id,
            "message": user_message[:200],
        })

        # Idempotency dedup — skip if same key already processed recently
        if idempotency_key:
            self._idempotency_key = idempotency_key
            from app.agent.lanes import get_lane_manager
            lm = get_lane_manager()
            if idempotency_key in lm._idempotency_cache:
                logger.info(f"[AGENT] Idempotency hit — key={idempotency_key}, skipping")
                return AgentResponse(
                    text="(duplicate request — skipped)",
                    session_id=session_id or "",
                    tool_calls=[],
                    model_used="",
                    input_tokens=0,
                    output_tokens=0,
                )
            lm._idempotency_cache[idempotency_key] = True

        # Store thinking budget for this run
        self._thinking_budget = thinking_budget

        from app.db.database import async_session_maker

        # ── Phase 1: Load from DB (short-lived session) ──────────
        t_phase1 = time.perf_counter()
        async with async_session_maker() as db:
            t_db = time.perf_counter()
            session, is_new = await self._get_or_create_session(db, user_id, session_id, telegram_chat_id, channel=channel, app_id=app_id, force_new=force_new_session)
            session_id = session.id
            logger.info(f"[PERF] get_or_create_session: {(time.perf_counter() - t_db) * 1000:.0f}ms")

            # Load user's disabled tools from AgentConfig
            # AgentConfig is platform-only — may not exist in agent DBs
            t_db = time.perf_counter()
            try:
                from sqlalchemy import select as _select
                from app.db import AgentConfig
                async with db.begin_nested():
                    _ac_result = await db.execute(
                        _select(AgentConfig).where(AgentConfig.user_id == user_id)
                    )
                    _ac = _ac_result.scalars().first()
                if _ac and getattr(_ac, 'disabled_tools', None):
                    import json as _json
                    _user_disabled = set(_json.loads(_ac.disabled_tools))
                    self.tools.user_disabled_tools = _user_disabled
                    self._disabled_tool_names = _user_disabled
                else:
                    self.tools.user_disabled_tools = set()
                    self._disabled_tool_names = set()
            except Exception:
                self.tools.user_disabled_tools = set()
                self._disabled_tool_names = set()
            logger.info(f"[PERF] load_agent_config: {(time.perf_counter() - t_db) * 1000:.0f}ms")

            # Resolve effective timezone here (not just inside
            # _build_system_prompt) so day-chat resolution + history
            # rendering both use the user's local time. Without this,
            # Telegram/WhatsApp/voice channels — which never pass
            # `client_tz` — got `tz_name=None` in
            # `resolve_day_chat_id_for_now` / `load_day_context`,
            # which then defaulted to UTC. Result: history messages
            # the agent saw were stamped in UTC, so a 9:58 PM EDT
            # message rendered as "1:58am" in the LLM context and
            # the agent confidently told the user "you sent that at
            # 1:58 AM". Single source of truth: User.timezone, with
            # client_tz override when the surface supplies one.
            if not client_tz:
                try:
                    from sqlalchemy import select as _select_for_tz
                    from app.db.models import User as _User_for_tz
                    _u_tz_row = (
                        await db.execute(
                            _select_for_tz(_User_for_tz).where(_User_for_tz.id == user_id)
                        )
                    ).scalar_one_or_none()
                    _profile_tz = (
                        getattr(_u_tz_row, "timezone", None) if _u_tz_row else None
                    )
                    if _profile_tz:
                        client_tz = _profile_tz
                        logger.info(
                            "[AGENT] tz_seeded_from_profile user=%s channel=%s tz=%s",
                            user_id[:8], channel, _profile_tz,
                        )
                except Exception as _tz_seed_err:
                    logger.debug(
                        "[AGENT] tz_seed_failed user=%s err=%s",
                        user_id[:8], _tz_seed_err,
                    )

            t_db = time.perf_counter()
            # ── Day-Chat context path (feature-flagged) ──
            _day_chat_id = None
            _day_context = None
            _use_day_ctx = False
            try:
                from app.agent.day_chat_resolver import should_use_day_chat_context
                _use_day_ctx = await should_use_day_chat_context()
            except Exception:
                pass

            if _use_day_ctx:
                try:
                    from app.agent.day_context_loader import load_day_context
                    from app.agent.context_manager import get_context_window
                    from app.db.message_helpers import resolve_day_chat_id_for_now
                    _day_chat_id = await resolve_day_chat_id_for_now(db, user_id, tz_override=client_tz)
                    _ctx_window = get_context_window(settings.agent_model)
                    # Pass client_tz so history annotations render in the
                    # user's LOCAL time (e.g. "[mobile 2:41pm]"), never UTC.
                    # Same tz the Runtime Context uses — single source.
                    _day_context = await load_day_context(db, _day_chat_id, model=settings.agent_model, model_context_tokens=_ctx_window, calling_channel=channel, tz_name=client_tz)
                    history = _day_context["messages"]
                    logger.info(f"[PERF] load_day_context: {(time.perf_counter() - t_db) * 1000:.0f}ms — {len(history)} messages (day-chat)")
                except Exception as _dce:
                    # ERROR, not WARNING — this silently degraded mobile day-chat
                    # recall for who knows how long. If this fires in prod, the
                    # agent just lost access to the full day's history and is
                    # running on the local session's last-N messages instead.
                    # Make it visible per Rule 12 / time-channel-fix PR.
                    logger.error(
                        "[AGENT] day_context_load_failed — falling back to session history. "
                        "user=%s channel=%s session_id=%s client_tz=%r attempted_day_chat_id=%s err=%s: %s",
                        user_id[:8], channel, session_id, client_tz,
                        locals().get("_day_chat_id"),
                        type(_dce).__name__, _dce,
                        exc_info=True,
                    )
                    _use_day_ctx = False
                    _day_context = None
                    history = await self._load_history(db, session_id)
                    logger.info(f"[PERF] load_history: {(time.perf_counter() - t_db) * 1000:.0f}ms — {len(history)} messages (fallback)")
            else:
                history = await self._load_history(db, session_id)
                logger.info(f"[PERF] load_history: {(time.perf_counter() - t_db) * 1000:.0f}ms — {len(history)} messages")

            # If conversation has active app_builder context (direction cards,
            # tool calls), override intent so tools and skill prompts stay available
            if query_intent.category != "full" and self._has_builder_context(history):
                query_intent = INTENT_FULL
                logger.info("[AGENT] Overriding intent to FULL — app_builder context detected in history")

            t_prompt = time.perf_counter()
            system_prompt = await self._build_system_prompt(db, user_id, user_message, channel=channel, intent=query_intent, client_tz=client_tz)

            # Inject <today_so_far> block when using day-chat context with a summary
            if _use_day_ctx and _day_context and _day_context.get("summary"):
                from app.agent.day_context_loader import build_today_so_far_block
                system_prompt += build_today_so_far_block(_day_context["summary"])
                self._memory_health["today_summary_present"] = True

            # Reply-to directive: if the current user turn carries a <reply_to>
            # block, mirror the block into the system prompt so even if day
            # history dominates the model's attention, the quoted content is
            # also present in the system context. Belt-and-suspenders: the
            # preamble lives in BOTH the system prompt and the user message.
            _stripped_um = user_message.lstrip()
            if _stripped_um.startswith("<reply_to>"):
                import re as _re
                _block_match = _re.search(
                    r"<reply_to>.*?</reply_to>", _stripped_um, _re.DOTALL,
                )
                _reply_block = _block_match.group(0) if _block_match else ""
                system_prompt += (
                    "\n<reply_to_directive>\n"
                    "CRITICAL FOR THIS TURN ONLY:\n"
                    "The user's latest message begins with the following "
                    "reply-to block, which identifies the specific earlier "
                    "message they are responding to:\n\n"
                    f"{_reply_block}\n\n"
                    "Answer the user's question STRICTLY about the quoted "
                    "message above. Do NOT substitute another topic from the "
                    "day's history, even if other topics (like prior emails, "
                    "routines, or earlier conversations) are more prominent "
                    "in the recent context. If the quoted message is unclear, "
                    "ask a clarifying question about IT, not about anything "
                    "else.\n"
                    "</reply_to_directive>\n"
                )

            # F8: Inject <recent_days> recap on day-boundary warm starts.
            # Only fires when today's day-chat is fresh (no rolling summary
            # yet, low message count). Once the day grows its own context,
            # the active conversation IS the continuity — don't double up.
            #
            # F6 telemetry: capture summary_status / failure_reason from
            # today's day_chat row even when we don't inject recent_days,
            # so the [memory_health] line below can surface summarizer
            # health (e.g. "summary=failed reason=auth_error") at every
            # turn, not just on warm starts.
            if _use_day_ctx and _day_context and _day_chat_id:
                try:
                    from app.services.recent_days_service import (
                        should_inject_recent_days,
                        get_recent_day_summaries,
                        build_recent_days_block,
                    )
                    from sqlalchemy import select as _sel_dc
                    from app.db.models.day_chat import DayChat
                    _today_dc = (await db.execute(
                        _sel_dc(DayChat).where(DayChat.id == _day_chat_id)
                    )).scalar_one_or_none()
                    if _today_dc:
                        # F6: always capture summary state for memory_health
                        self._memory_health["summary_status"] = _today_dc.summary_status
                        self._memory_health["summary_failure_reason"] = _today_dc.summary_last_failure_reason
                        # F8: gated injection
                        if should_inject_recent_days(_day_context):
                            _recent = await get_recent_day_summaries(
                                db, user_id, _today_dc.local_date,
                            )
                            if _recent:
                                system_prompt += build_recent_days_block(
                                    _recent, today_local_date=_today_dc.local_date,
                                )
                                self._memory_health["recent_days"] = len(_recent)
                                logger.info(
                                    "[AGENT] recent_days injected: %d day(s) "
                                    "(fresh day-chat msg_count=%d)",
                                    len(_recent),
                                    _day_context.get("message_count", 0),
                                )
                except Exception as _rd_err:
                    # Non-fatal: continuity recap is a quality-of-life
                    # surface, not a correctness gate. Log loudly so a
                    # silent regression is visible.
                    logger.warning(
                        "[AGENT] recent_days_block_failed user=%s err=%s: %s",
                        user_id[:8], type(_rd_err).__name__, _rd_err,
                    )

            logger.info(f"[PERF] build_system_prompt: {(time.perf_counter() - t_prompt) * 1000:.0f}ms — {len(system_prompt)} chars (~{estimate_tokens(system_prompt)} tokens)")

            # F6: Single structured per-turn memory_health line.
            # ONE grep target for "is memory working for user X right now?".
            # Fields are stable so an operator can pipe through awk/jq.
            #   retrieved=N        → user memories from hybrid_search
            #   active_tasks=N     → open threads in <active_tasks> block
            #   recent_days=N      → days surfaced in <recent_days> block (F8)
            #   today_summary=Y/N  → was <today_so_far> injected this turn
            #   summary=<status>   → today's day_chat summary lifecycle
            #   reason=<reason>    → FF-B.2 failure taxonomy if last attempt failed
            #   intent=<cat>       → query intent classification
            #   tokens=N           → estimated system prompt tokens
            try:
                _mh = self._memory_health
                logger.info(
                    "[memory_health] user=%s channel=%s retrieved=%d active_tasks=%d "
                    "recent_days=%d today_summary=%s summary=%s reason=%s "
                    "intent=%s tokens=%d",
                    user_id[:8], channel,
                    _mh.get("retrieved", 0),
                    _mh.get("active_tasks", 0),
                    _mh.get("recent_days", 0),
                    "Y" if _mh.get("today_summary_present") else "N",
                    _mh.get("summary_status") or "-",
                    _mh.get("summary_failure_reason") or "-",
                    getattr(query_intent, "category", "-"),
                    estimate_tokens(system_prompt),
                )

                # F-final: WARN-level alert on degraded memory state.
                # Two signals chosen because they're high-precision and
                # actionable: a credentialed-failure streak means the
                # summarizer hasn't worked in 3+ days (key/network ops
                # issue), and an empty retrieval on a non-greeting turn
                # means hybrid_search returned nothing for a real query
                # (corpus gap or retrieval bug). Greeting/social intent
                # legitimately retrieves nothing — those are excluded.
                _alerts: List[str] = []
                _intent_cat = getattr(query_intent, "category", "")
                _summary_status = _mh.get("summary_status")
                _failure_reason = _mh.get("summary_failure_reason")
                # Heuristic: persistent failure → recent failures + a
                # diagnostic reason. The summarizer service writes the
                # reason every time it fails, so presence + status='failed'
                # is the operator-actionable signal.
                if _summary_status == "failed" and _failure_reason:
                    _alerts.append(f"summarizer_persistent_fail:{_failure_reason}")
                if (
                    _mh.get("retrieved", 0) == 0
                    and _intent_cat not in ("", "-", "greeting", "social", "casual")
                    and len(user_message.strip()) > 12  # skip yes/no/hi
                ):
                    _alerts.append("retrieval_empty_on_substantive_turn")

                if _alerts:
                    logger.warning(
                        "[memory_health_alert] user=%s channel=%s reasons=%s "
                        "summary=%s retrieved=%d intent=%s",
                        user_id[:8], channel, ",".join(_alerts),
                        _summary_status or "-",
                        _mh.get("retrieved", 0),
                        _intent_cat or "-",
                    )
            except Exception as _mh_err:
                # Never let telemetry break the turn.
                logger.debug(f"[memory_health] log build failed: {_mh_err}")

            await db.commit()
        logger.info(f"[PERF] phase1_total: {(time.perf_counter() - t_phase1) * 1000:.0f}ms")
        # DB session closed — no connection held during LLM calls

        # Prepare messages
        messages = list(history)
        if media_paths:
            content_blocks = self._build_media_content(user_message, media_paths)
            messages.append({"role": "user", "content": content_blocks})
        else:
            messages.append({"role": "user", "content": user_message})

        # ── ContextBudgetLog telemetry (day-chat path only) ──
        if _use_day_ctx and _day_context and _day_chat_id:
            try:
                async with async_session_maker() as _cbl_db:
                    from app.agent.context_budget import log_context_budget
                    await log_context_budget(
                        db=_cbl_db,
                        day_chat_id=_day_chat_id,
                        conversation_id=session_id,
                        user_id=user_id,
                        turn_id=None,  # Set after user message is saved
                        system_tokens=estimate_tokens(system_prompt),
                        summary_tokens=estimate_tokens(_day_context.get("summary") or ""),
                        history_tokens=sum(estimate_tokens(m.get("content", "")) for m in messages),
                        tool_tokens=0,  # Counted after tool loop
                        memory_tokens=0,  # Already in system_tokens
                        total_tokens=estimate_tokens(system_prompt) + sum(estimate_tokens(m.get("content", "")) for m in messages),
                        model=settings.agent_model,
                        summary_was_stale=_day_context.get("summary_was_stale", False),
                    )
                    await _cbl_db.commit()
            except Exception as _cbl_err:
                logger.warning("[context_budget] Log failed (non-fatal): %s", _cbl_err)

        # Context window management — initial check
        if needs_compaction(system_prompt, messages, settings.agent_model):
            logger.info(f"[AGENT] Context compaction triggered ({len(messages)} messages)")
            messages = await compact_messages(messages, settings.agent_model)
            logger.info(f"[AGENT] After compaction: {len(messages)} messages, ~{estimate_messages_tokens(messages)} tokens")

        # Context tracking helper
        from app.agent.context_manager import get_context_window
        _context_window = get_context_window(settings.agent_model)
        _compaction_count = 0

        # ── Phase 2: Agent loop (no DB connection held) ──────────
        total_input = 0
        total_output = 0
        all_tool_calls: List[Dict[str, Any]] = []
        # Per-call records persisted alongside the assistant message so the
        # ToolPillRow chrome (frontend) can re-render days later when the
        # user scrolls back. Each entry: {tool, started_at_ms,
        # completed_at_ms, summary}. Kept thin (summary capped at 2KB
        # per record) — the message column is Text-backed JSON, no point
        # paying a few hundred KB just so a click-to-expand pill can
        # show a giant raw blob.
        tool_event_records: List[Dict[str, Any]] = []
        final_text = ""
        model_used = ""

        # Determine which model to use
        routing_decision: Optional[RoutingDecision] = None
        if self._session_model_override and model_override is None:
            model_override = self._session_model_override

        if model_override == "auto" or model_override is None:
            # Read user's preferred provider from agent_config (bundle mode).
            # Defaults to anthropic if not set or BYOK.
            preferred = None
            try:
                from app.db import AgentConfig as _AC
                from sqlalchemy import select as _sel
                _pref = (await db.execute(
                    _sel(_AC.preferred_provider).where(_AC.user_id == user_id)
                )).scalar_one_or_none()
                preferred = _pref
            except Exception:
                pass
            routing_decision = classify_request(
                user_message=user_message,
                conversation_history=messages[:-1],
                has_media=bool(media_paths),
                preferred_provider=preferred,
            )
            active_model = routing_decision.model
            logger.info(f"[AGENT] Auto-routed: {routing_decision.reason}")
        else:
            active_model = model_override

        active_llm = self.anthropic if _is_claude_model(active_model) else self.llm

        # ── Filter tools by query intent ──────────────────────────────
        # First iteration uses intent-filtered tools. If the LLM requests
        # tools and we loop back, escalate to full toolset so the agent
        # isn't artificially constrained mid-conversation.
        all_tools = self.tool_defs
        filtered_tools = filter_tools_by_intent(all_tools, query_intent)
        current_tools = filtered_tools

        # Vault CP4.1: strip save_streaming_credential on channels that
        # cannot render the confirmation card today (see module-level
        # VAULT_TOOL_CHANNEL_BLOCK).
        current_tools = strip_vault_tool_for_channel(current_tools, channel)

        # In vibecoding mode, strip all app_builder tools — agent should code directly
        _vibe_job_id: Optional[str] = None
        _vibe_app_id: Optional[str] = None
        if channel == "vibecoding":
            current_tools = [
                t for t in current_tools
                if not (t.get("name", "") or t.get("function", {}).get("name", "") or "").startswith("app_builder__")
            ]
            logger.info(f"[VIBE] Stripped app_builder tools for vibecoding channel, {len(current_tools)} tools remaining")

            # Register vibecoding session in DB (App + BuildJob) for visibility
            _vibe_logger = None
            _vibe_app_dir = None
            try:
                _vibe_job_id, _vibe_app_id, _vibe_app_dir = await self._register_vibecoding_session(
                    user_id=user_id,
                    prompt=user_message,
                    session_id=session_id,
                )
                if _vibe_job_id:
                    from app.agent.job_logger import JobLogger
                    _vibe_logger = JobLogger(_vibe_job_id, user_id)
                    await _vibe_logger.info(f"Vibe coding: {user_message[:80]}")
                # Set session workspace so write_file targets vibecoding/{slug}/
                if _vibe_app_dir and hasattr(self.tools, 'set_session_workspace'):
                    self.tools.set_session_workspace(_vibe_app_dir)
                    logger.info(f"[VIBE] Set session workspace to {_vibe_app_dir}")
            except Exception as e:
                logger.warning(f"[VIBE] Failed to register vibecoding session: {e}")

        # In app-channel mode, strip app_builder tools AND core mutation tools —
        # customization must use app__write_file / app__edit_file so edits get logged
        # via _record_layer2_change. Core write_file/edit_file/exec bypass the audit trail.
        # Agent proved it uses exec with Python scripts as a workaround when write_file is stripped.
        if channel == "app":
            current_tools = [
                t for t in current_tools
                if not (t.get("name", "") or t.get("function", {}).get("name", "") or "").startswith("app_builder__")
                and (t.get("name", "") or t.get("function", {}).get("name", "") or "") not in ("write_file", "edit_file", "exec", "pty_exec", "apply_patch")
            ]
            logger.info(f"[APP] Stripped app_builder + core mutation tools for app channel, {len(current_tools)} tools remaining")

        logger.info(
            f"[PERF] tool_filter: {len(all_tools)} total → {len(current_tools)} for intent={query_intent.category}"
        )
        logger.info(f"[AGENT] Using {active_model} via {'Anthropic' if _is_claude_model(active_model) else 'OpenAI'} with {len(messages)} messages")

        for iteration in range(self.max_iterations):
            logger.info(f"[AGENT] Iteration {iteration + 1}/{self.max_iterations}")

            text_buf = ""
            pending_tool_calls: List[Dict[str, Any]] = []
            stop_reason = ""

            for attempt in range(MAX_RETRIES + 1):
                try:
                    text_buf = ""
                    pending_tool_calls = []
                    stop_reason = ""
                    _t_llm_start = time.perf_counter()
                    _t_first_token = None

                    # In vibecoding mode, force tool use on first iteration
                    _tool_choice = "required" if (channel == "vibecoding" and iteration == 0 and current_tools) else None

                    async for event in active_llm.create_message_stream(
                        messages=messages,
                        system=system_prompt,
                        tools=current_tools or None,
                        model=active_model,
                        thinking_budget=thinking_budget if _is_claude_model(active_model) else 0,
                        tool_choice=_tool_choice,
                    ):
                        if cancel_check and cancel_check():
                            logger.info("[AGENT] Cancelled during streaming")
                            raise asyncio.CancelledError("Generation cancelled by user")

                        if event.type == "text":
                            if _t_first_token is None:
                                _t_first_token = time.perf_counter()
                                logger.info(f"[PERF] llm_ttft: {(_t_first_token - _t_llm_start) * 1000:.0f}ms (iteration {iteration + 1})")
                            text_buf += event.text
                            if on_text_chunk:
                                await on_text_chunk(event.text)

                        elif event.type in ("thinking_start", "thinking"):
                            pass

                        elif event.type == "tool_use_start":
                            if on_tool_start:
                                await on_tool_start(event.tool_name)

                        elif event.type == "tool_use_end":
                            pending_tool_calls.append({
                                "id": event.tool_id,
                                "name": event.tool_name,
                                "input": event.tool_input,
                            })

                        elif event.type == "message_end":
                            stop_reason = event.stop_reason
                            total_input += event.usage.get("input_tokens", 0)
                            total_output += event.usage.get("output_tokens", 0)
                            model_used = active_model

                    logger.info(
                        f"[PERF] llm_total: {(time.perf_counter() - _t_llm_start) * 1000:.0f}ms "
                        f"(iteration {iteration + 1}, in={event.usage.get('input_tokens', 0)}, "
                        f"out={event.usage.get('output_tokens', 0)}, stop={stop_reason})"
                    )
                    break  # Success

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    # Detect errors that warrant immediate cross-provider fallback:
                    # - 401 auth errors: broken credentials, skip retries entirely
                    # - 429 rate limits: after exhausting retries, cross to other provider
                    _err_str = str(e).lower()
                    _is_auth_error = "401" in str(e) or "authentication" in _err_str or "AuthenticationError" in type(e).__name__
                    _is_rate_limit = "429" in str(e) or "rate_limit" in _err_str or "RateLimitError" in type(e).__name__
                    _should_cross_provider = _is_auth_error or _is_rate_limit
                    if _is_auth_error:
                        attempt = MAX_RETRIES  # skip remaining retries

                    if attempt < MAX_RETRIES:
                        logger.warning(f"[AGENT] LLM call failed (attempt {attempt + 1}), retrying: {e}")
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        # Pick fallback: on auth/rate-limit errors, cross to the OTHER
                        # provider so broken or throttled keys fall back gracefully.
                        # If the only configured provider is the one that just failed,
                        # don't pretend we can cross over — re-raise the original error
                        # so the user sees the real cause, not a spurious "key missing"
                        # from the other provider.
                        from app.services.key_provider import keys as _keys
                        from app.services.model_resolver import (
                            default_anthropic_model as _default_anthropic_model,
                            default_openai_model as _default_openai_model,
                        )
                        primary_is_claude = _is_claude_model(active_model)
                        fallback = None
                        if _should_cross_provider and primary_is_claude and _keys.has_openai:
                            fallback = _default_openai_model()
                            logger.warning(f"[AGENT] {type(e).__name__} on {active_model}, crossing to OpenAI fallback {fallback}")
                        elif _should_cross_provider and not primary_is_claude and _keys.has_anthropic:
                            fallback = _default_anthropic_model()
                            logger.warning(f"[AGENT] {type(e).__name__} on {active_model}, crossing to Anthropic fallback {fallback}")
                        else:
                            configured = settings.agent_fallback_model
                            fallback_is_claude = _is_claude_model(configured)
                            if (fallback_is_claude and _keys.has_anthropic) or (not fallback_is_claude and _keys.has_openai):
                                fallback = configured
                                logger.warning(f"[AGENT] Primary model {active_model} failed, trying fallback {fallback}")
                            else:
                                logger.error(f"[AGENT] Primary model {active_model} failed and no other provider configured — re-raising")
                                raise

                        if active_model != fallback:
                            fallback_llm = self.anthropic if _is_claude_model(fallback) else self.llm
                            try:
                                text_buf = ""
                                pending_tool_calls = []
                                stop_reason = ""
                                async for event in fallback_llm.create_message_stream(
                                    messages=messages,
                                    system=system_prompt,
                                    tools=current_tools or None,
                                    model=fallback,
                                    thinking_budget=thinking_budget if _is_claude_model(fallback) else 0,
                                ):
                                    if cancel_check and cancel_check():
                                        raise asyncio.CancelledError("Cancelled")
                                    if event.type == "text":
                                        text_buf += event.text
                                        if on_text_chunk:
                                            await on_text_chunk(event.text)
                                    elif event.type == "tool_use_start":
                                        if on_tool_start:
                                            await on_tool_start(event.tool_name)
                                    elif event.type == "tool_use_end":
                                        pending_tool_calls.append({
                                            "id": event.tool_id,
                                            "name": event.tool_name,
                                            "input": event.tool_input,
                                        })
                                    elif event.type == "message_end":
                                        stop_reason = event.stop_reason
                                        total_input += event.usage.get("input_tokens", 0)
                                        total_output += event.usage.get("output_tokens", 0)
                                        model_used = fallback
                                break  # Fallback succeeded
                            except asyncio.CancelledError:
                                raise
                            except Exception as fallback_err:
                                logger.error(f"[AGENT] Fallback model {fallback} also failed: {fallback_err}")
                                await self._log_error(
                                    user_id=user_id,
                                    session_id=session_id,
                                    error_type="llm_error",
                                    error_message=f"Primary ({active_model}): {e}\nFallback ({fallback}): {fallback_err}",
                                    context={"iteration": iteration, "messages_count": len(messages)},
                                )
                                raise fallback_err
                        else:
                            await self._log_error(
                                user_id=user_id,
                                session_id=session_id,
                                error_type="llm_error",
                                error_message=str(e),
                                context={"iteration": iteration, "model": active_model},
                            )
                            logger.error(f"[AGENT] LLM call failed after {MAX_RETRIES + 1} attempts: {e}")
                            raise

            # Build the assistant message for conversation continuity
            assistant_content: List[Dict[str, Any]] = []
            if text_buf:
                assistant_content.append({"type": "text", "text": text_buf})
            for tc in pending_tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })

            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            # If no tool calls, we're done
            if stop_reason != "tool_use" or not pending_tool_calls:
                final_text = text_buf
                break

            # Execute tool calls
            tool_results: List[Dict[str, Any]] = []
            for tc in pending_tool_calls:
                if cancel_check and cancel_check():
                    logger.info("[AGENT] Cancelled before tool execution")
                    raise asyncio.CancelledError("Generation cancelled by user")

                logger.info(f"[AGENT] Tool called: {tc['name']}({json.dumps(tc['input'])[:200]})")
                all_tool_calls.append(tc)
                await _hb.emit(HookEvent.BEFORE_TOOL_CALL, {"tool": tc["name"], "input": tc["input"]})

                _t_tool = time.perf_counter()
                _t_tool_started_ms = int(time.time() * 1000)
                try:
                    result = await self.tools.execute(tc["name"], tc["input"])
                except Exception as e:
                    logger.exception(f"[AGENT] Tool {tc['name']} crashed")
                    result = f"ERROR: Tool crashed: {type(e).__name__}: {e}"

                logger.info(f"[PERF] tool_exec({tc['name']}): {(time.perf_counter() - _t_tool) * 1000:.0f}ms — {len(result)} chars")
                logger.info(f"[AGENT] Tool result: {result[:200]}")
                await _hb.emit(HookEvent.AFTER_TOOL_CALL, {"tool": tc["name"], "result_len": len(result)})
                # Capture for the persisted ToolPillRow re-render. We
                # cap summary at 2KB per record — the click-to-expand
                # pill UI shows a popover, not a code editor; if you
                # need the full payload for debugging, pull it from
                # logs instead of bloating every message row.
                _record_summary = result if len(result) <= 2048 else (result[:2048] + "…")
                tool_event_records.append({
                    "tool": tc["name"],
                    "started_at_ms": _t_tool_started_ms,
                    "completed_at_ms": int(time.time() * 1000),
                    "summary": _record_summary,
                })
                if on_tool_end:
                    summary = result[:200] + "..." if len(result) > 200 else result
                    await on_tool_end(tc["name"], summary, tc.get("input"))

                # Drain newly-registered attachments (from generate_* tools) and
                # emit them over the WS. We emit per-attachment so the frontend
                # can open the DocumentSplit pane as soon as the first file is ready.
                if on_attachment and len(self.tools.pending_attachments) > _attachments_emitted_count:
                    for att in self.tools.pending_attachments[_attachments_emitted_count:]:
                        try:
                            await on_attachment(asst_message_id, att)
                        except Exception:
                            logger.exception("on_attachment callback raised")
                    _attachments_emitted_count = len(self.tools.pending_attachments)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

            # After first tool use, escalate to full toolset for subsequent iterations
            # so the agent isn't constrained if it discovers it needs more tools
            if current_tools is not all_tools and query_intent.category != "full":
                current_tools = all_tools
                # Re-apply vibecoding filter after escalation
                if channel == "vibecoding":
                    current_tools = [
                        t for t in current_tools
                        if not (t.get("name", "") or t.get("function", {}).get("name", "") or "").startswith("app_builder__")
                    ]
                # Vault CP4.1: re-strip save_streaming_credential on blocked channels.
                current_tools = strip_vault_tool_for_channel(current_tools, channel)
                logger.info(f"[AGENT] Escalated to full toolset ({len(current_tools)} tools) after tool use")

            # ── Mid-loop context compaction ──────────────────────
            # Check if context is getting large and compact if needed.
            # This prevents the agent from hard-stopping mid-conversation.
            _msg_tokens = estimate_messages_tokens(messages)
            _sys_tokens = estimate_tokens(system_prompt)
            _total_ctx = _msg_tokens + _sys_tokens
            _usage_ratio = _total_ctx / _context_window if _context_window > 0 else 0

            # Broadcast context usage when above 50%
            if _usage_ratio >= 0.50 and on_tool_progress:
                _pct = round(_usage_ratio * 100)
                try:
                    await on_tool_progress("__context__", f"Session: {_pct}%")
                except Exception:
                    pass

            # Auto-compact at 80% to keep the conversation going
            if _usage_ratio >= 0.80:
                _compaction_count += 1
                logger.info(f"[AGENT] Mid-loop compaction #{_compaction_count} at {_usage_ratio:.0%} ({_total_ctx:,} tokens)")
                messages = await compact_messages(messages, settings.agent_model)
                _after = estimate_messages_tokens(messages)
                logger.info(f"[AGENT] After compaction: {len(messages)} messages, ~{_after:,} tokens")
                # Inject continuation marker so the agent picks up seamlessly
                messages.insert(0, {
                    "role": "user",
                    "content": (
                        "[This session was auto-compacted to free up context space. "
                        "Earlier conversation has been summarized above. "
                        "Continue seamlessly from where you left off.]"
                    ),
                })

        else:
            # Max iterations reached
            if not final_text:
                final_text = text_buf or "I've reached the maximum number of tool iterations. Here's what I have so far."

        # ── Phase 3: Save to DB (short-lived session) ────────────
        t_phase3 = time.perf_counter()
        # Save messages synchronously (fast, needed for conversation continuity)
        async with async_session_maker() as db:
            await self._save_messages(
                db=db,
                session_id=session_id,
                user_id=user_id,
                user_message=display_user_message or user_message,
                assistant_response=final_text,
                tokens_input=total_input,
                tokens_output=total_output,
                model=model_used,
                processing_time_ms=int((time.time() - start) * 1000),
                save_user_message=save_user_message,
                client_tz=client_tz,
                asst_message_id=asst_message_id,
                channel=channel,
                tool_event_records=tool_event_records,
            )
            await db.commit()
        logger.info(f"[PERF] phase3_save: {(time.perf_counter() - t_phase3) * 1000:.0f}ms")

        # ── Phase 3b: Background tasks (memory extraction, feedback) ──
        # These are slow (LLM calls) — run in background so response returns immediately
        async def _background_post_processing():
            try:
                async with async_session_maker() as bg_db:
                    try:
                        if settings.auto_extract_memories and final_text:
                            mem_count = await self._extract_memories(
                                db=bg_db,
                                user_id=user_id,
                                user_message=user_message,
                                assistant_response=final_text,
                            )
                            logger.info(f"[AGENT] Background: extracted {mem_count} memories")

                        # Active task extraction (pattern-based, no LLM cost)
                        try:
                            from app.services.active_task_service import detect_active_tasks, store_active_task, decay_expired_tasks
                            tasks_found = detect_active_tasks(user_message, final_text)
                            for task_text in tasks_found:
                                await store_active_task(bg_db, user_id, task_text)
                            # Also decay expired tasks periodically
                            archived = await decay_expired_tasks(bg_db, user_id)
                            if tasks_found or archived:
                                await bg_db.commit()
                                logger.info(f"[AGENT] Active tasks: {len(tasks_found)} found, {archived} archived")
                        except Exception as _at_err:
                            logger.debug(f"[AGENT] Active task extraction skipped: {_at_err}")

                        try:
                            from app.services.retrieval_feedback import get_retrieval_feedback
                            feedback_svc = get_retrieval_feedback(bg_db)
                            await feedback_svc.log_retrieval_feedback(
                                user_id=user_id,
                                query=user_message,
                                retrieved_memories=self._last_retrieved_memories,
                                response=final_text,
                                conversation_id=session_id,
                                strategies_used=["vector", "keyword", "graph"],
                            )
                        except Exception as e:
                            logger.warning(f"[AGENT] Feedback logging failed (non-fatal): {e}")

                        await bg_db.commit()
                    except Exception as e:
                        await bg_db.rollback()
                        logger.warning(f"[AGENT] Background post-processing failed (non-fatal): {e}")
            except Exception as e:
                logger.warning(f"[AGENT] Background session error (non-fatal): {e}")

        asyncio.create_task(_background_post_processing())

        # ── Day-Chat summarizer (async, debounced, never blocks) ──
        if _use_day_ctx and _day_chat_id:
            try:
                from app.services.day_summarizer import run_summarizer_if_needed
                asyncio.create_task(run_summarizer_if_needed(async_session_maker, _day_chat_id))
            except Exception as _sum_err:
                logger.warning("[AGENT] Summarizer scheduling failed (non-fatal): %s", _sum_err)

        elapsed = int((time.time() - start) * 1000)
        logger.info(f"[AGENT] Response: {final_text[:100]}...")
        logger.info(
            f"[PERF] agent_run_total: {elapsed}ms | intent={query_intent.category} "
            f"| tools_sent={len(current_tools)} | in={total_input} out={total_output} "
            f"| tool_calls={len(all_tool_calls)}"
        )

        # Hook: agent run complete
        await _hb.emit(HookEvent.AGENT_END, {
            "user_id": user_id, "session_id": session_id,
            "tool_count": len(all_tool_calls),
            "tokens": total_input + total_output,
            "elapsed_ms": elapsed,
        })

        # Finalize vibecoding session: log events, mark job completed/failed
        if _vibe_job_id and _vibe_app_id:
            try:
                # Log tool calls and file edits
                if _vibe_logger:
                    for tc in all_tool_calls:
                        name = tc.get("name", "")
                        if name in ("write_file", "edit_file"):
                            path = tc.get("arguments", {}).get("path", "") if isinstance(tc.get("arguments"), dict) else ""
                            await _vibe_logger.edit(f"Edited {path or 'file'}", meta={"path": path, "tool": name})
                        elif name:
                            await _vibe_logger.tool(f"Used {name}", meta={"tool": name})
                    _vibe_logger._total_tokens = total_input + total_output
                    await _vibe_logger.persist()

                async with async_session_maker() as vdb:
                    from app.db.models import App as _App, BuildJob as _BJ
                    vj = await vdb.get(_BJ, _vibe_job_id)
                    va = await vdb.get(_App, _vibe_app_id)
                    if vj and vj.status == "running":
                        wrote_files = any(
                            tc.get("name", "") in ("write_file", "edit_file", "exec")
                            for tc in all_tool_calls
                        )
                        vj.status = "completed" if wrote_files else "failed"
                        vj.completed_at = datetime.utcnow()
                        vj.total_tokens = total_input + total_output
                        vj.model = model_used
                        if not wrote_files:
                            vj.error_message = "No files were written"
                    if va:
                        va.status = "ready" if (vj and vj.status == "completed") else "error"
                    await vdb.commit()
            except Exception as e:
                logger.warning(f"[VIBE] Failed to finalize vibecoding session: {e}")
            finally:
                # Reset session workspace so next non-vibe run uses the default
                if hasattr(self.tools, 'set_session_workspace'):
                    self.tools.set_session_workspace(None)

        return AgentResponse(
            text=final_text,
            session_id=session_id,
            day_chat_id=_day_chat_id or "",
            tool_calls=all_tool_calls,
            tokens_input=total_input,
            tokens_output=total_output,
            tokens_total=total_input + total_output,
            model=model_used,
            processing_time_ms=elapsed,
            memories_extracted=0,  # extracted in background
            asst_message_id=asst_message_id,
        )
    
    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    async def _get_or_create_session(
        self,
        db: AsyncSession,
        user_id: str,
        session_id: Optional[str],
        telegram_chat_id: Optional[int],
        channel: Optional[str] = None,
        app_id: Optional[str] = None,
        force_new: bool = False,
    ):
        from sqlalchemy import select, and_
        from app.db.models import Conversation

        # If Telegram, try to find an active session for this chat
        if telegram_chat_id and not session_id:
            result = await db.execute(
                select(Conversation).where(
                    and_(
                        Conversation.user_id == user_id,
                        Conversation.channel == "telegram",
                        Conversation.is_active == True,
                        Conversation.metadata_json.contains(str(telegram_chat_id)),
                    )
                ).order_by(Conversation.updated_at.desc()).limit(1)
            )
            session = result.scalar_one_or_none()
            if session:
                return session, False

        # ── 2c Risk 4 compat shim: bridge sends literal "app-{id}" session_id ──
        # The injected bridge script (apps_proxy.py) hardcodes session_id: "app-{app_id}"
        # in every message. This bypasses the 2b metadata_json lookup that ChatPage uses,
        # creating duplicate Conversations for the same app on the same day.
        # Shim: if session_id looks like a bridge-generated value, normalize it to None
        # so it falls through to the 2b one-per-app-per-day reuse path below.
        # Session IDs in this system are always UUIDs (never start with "app-").
        # See: docs/checkpoint-2c-http-endpoint-audit.md, Risk 4.
        if (
            session_id
            and session_id.startswith("app-")
            and channel == "app"
            and app_id
        ):
            logger.info(
                "bridge_session_shim_applied user=%s app=%s original_session_id=%s",
                user_id[:8] if user_id else "?",
                app_id[:8] if app_id else "?",
                session_id,
            )
            session_id = None  # fall through to 2b resolution path

        if session_id:
            from sqlalchemy import select
            result = await db.execute(
                select(Conversation).where(
                    and_(
                        Conversation.id == session_id,
                        Conversation.user_id == user_id,
                    )
                )
            )
            session = result.scalar_one_or_none()
            if session:
                from datetime import datetime, timezone
                now_utc = datetime.now(timezone.utc)
                channel_switched = channel and session.channel and channel != session.channel

                if session.channel == "telegram" and not channel_switched:
                    # Telegram → Telegram: always reuse (long-lived sessions)
                    return session, False

                if channel_switched:
                    # Channel switch (e.g. telegram → web, web → vibecoding):
                    # create a new session so messages are tagged with the correct channel
                    logger.info(f"[AGENT] Channel switched {session.channel} → {channel}, creating new session")
                elif session.started_at:
                    started = session.started_at.replace(tzinfo=timezone.utc) if session.started_at.tzinfo is None else session.started_at
                    if started.date() != now_utc.date():
                        logger.info(f"[AGENT] Session {session_id} is from {started.date()}, creating new session for today")
                    else:
                        return session, False
                else:
                    return session, False

        # Resolve DayChat parent (if day-chat feature is active or backfill has run)
        # Prefer client_tz from the WS message over User.timezone from DB
        _day_chat_id = None
        try:
            from app.agent.day_chat_resolver import get_or_create_day_chat
            _tz_for_day = client_tz
            if not _tz_for_day:
                from app.db.models import User
                _user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
                _tz_for_day = getattr(_user, 'timezone', None) if _user else None
            _dc = await get_or_create_day_chat(db, user_id, tz_name=_tz_for_day)
            _day_chat_id = _dc.id
        except Exception as _dce:
            logger.debug("[AGENT] DayChat resolution skipped: %s", _dce)

        # ── App channel: one Conversation per app per day (reuse pattern) ──
        # When channel="app" and session_id is null, look for an existing
        # Conversation for this app in today's day chat. Reuse it so the user
        # sees a single continuous thread per app per day.
        # Uses SELECT ... FOR UPDATE on the DayChat row to prevent concurrent
        # first-messages from creating duplicate Conversations.
        if channel == "app" and app_id and not session_id and _day_chat_id and not force_new:
            try:
                from app.db.models.day_chat import DayChat
                # Lock the DayChat row — serializes concurrent app-conversation creation
                await db.execute(
                    select(DayChat).where(DayChat.id == _day_chat_id).with_for_update()
                )
                # Look for existing app conversation in today's day chat
                candidates = (await db.execute(
                    select(Conversation).where(and_(
                        Conversation.user_id == user_id,
                        Conversation.day_chat_id == _day_chat_id,
                        Conversation.channel == "app",
                    ))
                )).scalars().all()
                for conv in candidates:
                    try:
                        meta = json.loads(conv.metadata_json or "{}")
                        if meta.get("app_id") == app_id:
                            logger.info("[AGENT] Reusing app conversation %s for app %s", conv.id[:8], app_id[:8])
                            return conv, False
                    except (json.JSONDecodeError, TypeError):
                        continue
            except Exception as _app_err:
                logger.debug("[AGENT] App conversation lookup failed: %s", _app_err)

        # Create new session
        # Channel resolution — never silently default to "agent". If channel
        # is missing, resolve_channel logs a warning and returns "unknown".
        # Telegram is special-cased because telegram_chat_id is a structural
        # signal that we're in a Telegram session even if the caller forgot
        # the channel kwarg.
        from app.agent.channel_util import resolve_channel as _resolve_channel
        if telegram_chat_id:
            _channel = "telegram"
        else:
            _channel = _resolve_channel(
                explicit=channel,
                user_id=user_id,
                site="session_create",
            )
        _meta = None
        if telegram_chat_id:
            _meta = json.dumps({"telegram_chat_id": telegram_chat_id})
        elif channel == "app" and app_id:
            _meta = json.dumps({"app_id": app_id})

        session = Conversation(
            user_id=user_id,
            channel=_channel,
            is_active=True,
            day_chat_id=_day_chat_id,
            metadata_json=_meta,
        )
        db.add(session)
        await db.flush()
        return session, True
    
    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    async def _build_system_prompt(
        self,
        db: AsyncSession,
        user_id: str,
        user_message: str,
        channel: Optional[str] = None,
        intent: Optional[QueryIntent] = None,
        client_tz: Optional[str] = None,
    ) -> str:
        """Build a rich system prompt from identities + memories + runtime context.

        The `intent` parameter controls which sections are included:
        - Greetings/questions: skip memory retrieval, skills, environment, media
        - Code/full: include everything
        This reduces system prompt token count and thus LLM TTFT.

        Section order (most → least behavioral influence):
          1. Core Identity (soul)
          2. User Brain (portrait + facts + memories + entities)
          3. Skills
          4. Environment & Capabilities
          5. Runtime Context
          6. Formatting Rules (channel-specific)
          7. Onboarding (CONDITIONAL — only if not completed)
        """
        if intent is None:
            intent = INTENT_FULL
        from sqlalchemy import select, and_
        from app.db.models import Identity, IdentityType

        # Named section buckets — assembled in order at the end
        section_parts: Dict[str, str] = {}

        logger.info(f"[AGENT] Building system prompt for user: {user_id}")

        # ── 1. Load identities ──────────────────────────────────────
        result = await db.execute(
            select(Identity).where(
                and_(
                    Identity.user_id == user_id,
                    Identity.is_active == True,
                )
            ).order_by(Identity.priority.desc())
        )
        identities = result.scalars().all()
        logger.info(f"[AGENT] Found {len(identities)} identities")

        has_soul_identity = False
        identity_parts = []
        for identity in identities:
            if identity.identity_type == IdentityType.SOUL.value:
                identity_parts.insert(0, f"# Core Identity\n{identity.content}")
                has_soul_identity = True
            elif identity.identity_type == IdentityType.AGENT_INSTRUCTIONS.value:
                identity_parts.append(f"# Behavioral Guidelines\n{identity.content}")
            elif identity.identity_type == IdentityType.USER_PROFILE.value:
                identity_parts.append(f"# About the User\n{identity.content}")
            elif identity.identity_type == IdentityType.TOOLS.value:
                identity_parts.append(f"# Tool Guidelines\n{identity.content}")

        # Default identity if no Soul exists. Pre-Soul users see this on every
        # turn until they configure tone — so it must already feel like a
        # person, not a corporate assistant. The agent name is left to the
        # identity_anchor block below; here we set voice and posture.
        if not has_soul_identity:
            identity_parts.insert(0, (
                "# Core Identity\n"
                "You are the user's personal agent — present in their life as a "
                "real person who happens to be exceptionally capable. Not a "
                "chatbot, not an FAQ, not a help desk. Be warm, direct, curious. "
                "A trusted friend who can also actually get things done."
            ))
            logger.warning(f"No soul config found for user {user_id}, using default")

        section_parts["identity"] = "\n\n".join(identity_parts)

        # ── 1b. Identity anchor — kill the "I'm Claude / GPT" hallucination ──
        # LLMs have stale self-identity baked into their training data
        # ("I'm Claude 3.5 Sonnet"). Without this anchor, when a user asks
        # "what model are you?" the agent breaks the white-label illusion
        # by naming the underlying provider. We override that explicitly:
        # the agent identifies as the user's agent name (or generic "your
        # agent" until they've named it), period.
        try:
            from app.db.models import AgentConfig as _AC
            async with db.begin_nested():
                _name_cfg = (await db.execute(
                    select(_AC.agent_name).where(_AC.user_id == user_id)
                )).scalar_one_or_none()
        except Exception:
            _name_cfg = None
        _agent_label = (_name_cfg or "").strip()
        # Two cases: agent has a real name (use it everywhere) vs no name set
        # yet (don't fake one). The previous wording "you are <name> — a Toup
        # agent" was ambiguous in English: the model parsed "a Toup agent" as
        # "an agent named Toup" and self-IDed as Toup on greetings. The voice
        # channel had a hand-rolled guard for exactly this (see
        # ws_realtime.py:368). We hoist it to every channel and disambiguate
        # the sentence: Toup is the *platform*, the agent's name is separate.
        if _agent_label:
            section_parts["identity_anchor"] = (
                "# Identity\n"
                f"Your name is **{_agent_label}**. That is your name — use it "
                "when self-identifying.\n\n"
                "Toup is the **platform** you run on (toup.ai), NOT your name. "
                f"Never introduce yourself as 'Toup' or call yourself 'Toup'. "
                f"When asked your name, say **{_agent_label}**, not Toup.\n\n"
                "You are NOT Claude, NOT GPT, NOT Sonnet, NOT Opus, NOT any "
                "specific provider model. When the user asks what you are, "
                "who built you, or what model is powering you, answer as "
                f"**{_agent_label}** — never name the underlying LLM provider "
                "or version. The provider may change without notice; your "
                "identity to the user is stable."
            )
        else:
            section_parts["identity_anchor"] = (
                "# Identity\n"
                "You don't have a name yet — the user hasn't picked one. "
                "Don't introduce yourself with a made-up name, and especially "
                "do NOT call yourself 'Toup'. Toup is the platform you run on "
                "(toup.ai), not your name. If naming comes up naturally, ask "
                "what they'd like to call you.\n\n"
                "You are NOT Claude, NOT GPT, NOT Sonnet, NOT Opus, NOT any "
                "specific provider model. When the user asks what model is "
                "powering you, answer as the agent — never name the underlying "
                "LLM provider or version."
            )

        # ── 1d. Platform knowledge — what Toup is, every page, every capability ──
        # Always-on. The agent's job is to be the user's ultimate assistant on
        # this product — they should be able to say "I want X" and the agent
        # already knows where X lives, what tool gets there, and whether to
        # navigate, build, or recall. Without this section, the agent treats
        # each turn like a generic chat and forgets the platform exists.
        #
        # When you add a new top-level page or capability to Toup, update this
        # block — it is the agent's only authoritative product map. Routes
        # are verified against frontend/src/App.tsx; capability names match
        # tool_definitions.py.
        section_parts["platform_knowledge"] = (
            "# Platform Knowledge — How Toup Works\n"
            "You run on **Toup** — a personal-agent platform. The user lives "
            "here. You don't need to ask what features exist; you know. Below "
            "is the complete map. Use it: when the user expresses a goal, "
            "translate it into the right action without making them learn the "
            "product.\n\n"
            "## How tools work — read this carefully\n"
            "Throughout this section you'll see references to tools by name. "
            "When you need to use a tool, **invoke it through your function-"
            "calling API** — don't write the tool name as text in your "
            "message.\n\n"
            "WRONG (do NOT produce output like this):\n"
            "- `<navigate_to path=\"/brain\" />`\n"
            "- `navigate_to('/brain')` as a literal text line\n"
            "- `{tool: \"navigate_to\", args: {path: \"/brain\"}}`\n"
            "- Any XML/JSX/JSON-shaped tool-call syntax in the message body.\n\n"
            "RIGHT:\n"
            "- Call the actual tool. The user sees no tool syntax — only your "
            "natural-language reply (\"Done — there you go.\" / \"Pulled it up.\").\n"
            "- For chips that the user clicks, write `[[navigate:/brain]]` "
            "directly in your message text. Chips ARE part of your message; "
            "tool calls are NOT.\n\n"
            "If a tool you want to use isn't available on this turn (you'd see "
            "it in your tool list), explain in plain words instead and offer a "
            "[[chip]] for the user to take the action themselves. Never "
            "fake a tool call as text.\n\n"
            "## Pages — where things live\n"
            "Take the user anywhere via the navigate_to tool (explicit "
            "request) or a `[[navigate:/path]]` chip (suggestion).\n\n"
            "- `/` — **Hub**. Home. Entry cards.\n"
            "- `/chat` — **Chat**. This conversation. Day-grouped at "
            "`/chat/<date>` (e.g. `/chat/2026-05-08`).\n"
            "- `/brain` — **Brain**. Memories you've stored about the user, "
            "organized by category (identity, work, goals, preferences, "
            "knowledge, projects). Same data as the `# User Brain` section "
            "above — they see what you see.\n"
            "- `/browser` — **Live Browser**. The user watches your headless "
            "browser in real time. Use the `browser` tool while they're here "
            "and they see every click.\n"
            "- `/workspace` — **Workspace**. Apps you've built for the user.\n"
            "- `/workspace/apps/<slug>` — preview of a specific app. Reach "
            "via the `[[open_app:<slug>]]` chip after you build/restart one.\n"
            "- `/jobs` — **Jobs**. Long-running tasks with status and logs.\n"
            "- `/dashboard` — **Dashboard**. Metrics, inbox, daily summary.\n"
            "- `/agent` — **Agent home**. Soul, channels, LLM keys.\n"
            "- `/agent/soul` — **Soul**. Your personality config — name, "
            "style (casual/professional/mentor/creative), traits (humor, "
            "direct, proactive, references_past, etc.), pronouns, custom "
            "instructions.\n"
            "- `/agent/settings` — **Channels & Settings**. WhatsApp "
            "(BYOA paste OR QR-link mode), Telegram, voice wiring.\n"
            "- `/agent/tools` — **Tools catalog** with descriptions of every "
            "tool you have.\n"
            "- `/agent/skills` — **Skills catalog**. Domain-specific skill "
            "packs (some installed, some marketplace).\n"
            "- `/account` — **Account**. Profile, password, billing.\n"
            "- `/movies` — **Movies**. Netflix integration when relevant.\n\n"
            "## Capabilities — what you can actually do\n\n"
            "### Memory (lives at `/brain`)\n"
            "You have permanent memory across conversations. The user expects "
            "you to **remember**. When they share something about themselves, "
            "their projects, preferences, or facts about people they work "
            "with — store it with `memory_store`. When they ask 'what do you "
            "know about X' — `memory_search`. Relevant memories for the "
            "current turn are already loaded into the `# User Brain` section "
            "above; use them before searching again.\n\n"
            "### Apps you build (live at `/workspace`)\n"
            "You can BUILD real React apps for the user via the app_builder "
            "skill. Apps are deployed and previewable at "
            "`/workspace/apps/<slug>`. Use this when:\n"
            "- They ask for a tool ('make me a habit tracker', 'build a "
            "calorie counter', 'I need a quote generator').\n"
            "- A repeating workflow would be cleaner as an app than a "
            "conversation.\n"
            "After building or restarting, offer `[[open_app:<slug>]]` so "
            "they can tap to see it.\n\n"
            "### Live Browser (lives at `/browser`)\n"
            "You have a real headless Chromium via the `browser` tool. You "
            "can search, navigate, click, type, scroll, screenshot. The user "
            "watches you do it live at `/browser`. Use it for: real-time "
            "research, shopping comparisons, bookings, sign-ups, form "
            "filling — anything that needs the actual current web. If the "
            "user isn't already on `/browser`, consider suggesting "
            "`[[navigate:/browser]]` so they can follow along.\n\n"
            "### Day-Chat — one continuous thread per day\n"
            "The user can talk to you on web, mobile app, WhatsApp, "
            "Telegram, or voice. **All of those share the same thread for a "
            "given day.** Replying on WhatsApp shows on web. Don't "
            "reintroduce yourself when channels switch — it's the same "
            "conversation. Past days are recoverable via `recall_day`.\n\n"
            "### Voice\n"
            "The user can hit the voice button for real-time spoken "
            "conversation (OpenAI Realtime API). All your tools work in "
            "voice including `navigate_to`. Voice picks up the same memory, "
            "soul, and identity.\n\n"
            "### Channels (WhatsApp / Telegram / Mobile)\n"
            "If the user wants to text you on WhatsApp, point them to "
            "`/agent/settings` — they paste a token (BYOA mode) or scan a "
            "QR code (link mode, ~30s setup). Telegram is similar. Once "
            "wired, you receive their texts in the same Day-Chat thread.\n\n"
            "### Documents\n"
            "You can produce PDF (`generate_pdf`), Word "
            "(`generate_docx`), Excel (`generate_xlsx`), PowerPoint "
            "(`generate_pptx`), or Markdown (`generate_markdown`). Use "
            "these when the user wants something to **keep, share, or "
            "edit** — not for inline conversational answers.\n\n"
            "### Jobs & schedules\n"
            "Multi-step work → `create_job` then `update_job` as you "
            "progress. Live job card visible at `/jobs`. For reminders "
            "(text delivered at a scheduled time) → `routines__remind` "
            "with `when=once|daily|every`. For recurring agent tasks "
            "('every morning summarise my email') → `routines__create` "
            "with `kind=agent_task`. The legacy `cron` tool was removed "
            "— calling it returns an ERROR redirecting here.\n\n"
            "### Media\n"
            "`play_media` plays YouTube and Netflix in chat. For Netflix "
            "specifically, every title you mention should have a "
            "`[[Play TITLE on Netflix]]` chip on the line after it — see "
            "the Media section.\n\n"
            "### Past-day recall\n"
            "`recall_day` loads any past day's conversation across all "
            "channels (web, telegram, voice, app). Accepts natural dates: "
            "'yesterday', 'last Monday', '3 days ago', '2026-04-15'. NEVER "
            "tell the user you can't remember a past day — call this.\n\n"
            "## Decision rules — when user says X, do Y\n\n"
            "Use these as guidance for tool selection. Always invoke tools via "
            "the function-calling API (never as text — see above).\n\n"
            "- 'remember <fact>' / 'I'm working on <project>' → call `memory_store`, give a one-line confirmation\n"
            "- 'what do you know about <X>' → call `memory_search`\n"
            "- 'show me my memories' / 'take me to my brain' → call `navigate_to` with path `/brain`\n"
            "- 'make me a <tool/app>' / 'I need a <thing>' → use the app_builder skill\n"
            "- 'search the web' / 'find <X> for me' / 'book <X>' → call `browser`. If they should watch, also drop `[[navigate:/browser]]`\n"
            "- 'remind me at <Y>' / 'in N minutes remind me' / 'every morning at 7 nudge me' → call `routines__remind` (NEVER `cron` — it's removed and ERRORs)\n"
            "- 'schedule <agent task X>' / 'every morning summarise my email' → call `routines__create` with kind=`agent_task` or `email_briefing`\n"
            "- 'make this a PDF/doc/spreadsheet/deck' → call `generate_pdf` / `generate_docx` / `generate_xlsx` / `generate_pptx`\n"
            "- 'play <song/movie/show>' → call `play_media`\n"
            "- 'open settings' / 'wire up whatsapp' / 'change my phone' → call `navigate_to` with path `/agent/settings`\n"
            "- 'change your name' / 'change your personality' → call `navigate_to` with path `/agent/soul`\n"
            "- 'what tools do you have' → call `navigate_to` with path `/agent/tools`\n"
            "- 'yesterday's chat' / 'what did we talk about Monday' → call `recall_day`\n"
            "- 'where's my account' / 'change password' / 'billing' → call `navigate_to` with path `/account`\n"
            "- 'show me the dashboard' / 'metrics' → call `navigate_to` with path `/dashboard`\n"
            "- 'build me an app' (then they want to see it) → app_builder, then offer `[[open_app:<slug>]]` chip\n\n"
            "## What you should NEVER make the user do\n"
            "- Hunt through menus to find a feature you can navigate them to. Just take them.\n"
            "- Repeat themselves between channels — it's all one thread.\n"
            "- Manually copy data between Brain / Apps / Chat — you can do it.\n"
            "- Ask 'would you like me to do X?' when they explicitly asked you to do X. Just do it.\n\n"
            "The whole point of you is: the user says what they want, you make it happen."
        )

        # ── 1d. Self-knowledge — how YOUR memory actually works ──────
        # F7 (2026-05-08): pre-F7 the agent had no integrated picture of
        # its memory system in the prompt. When a user asked "how does
        # your memory work?", the model improvised. This block gives an
        # accurate, in-voice description that distinguishes the layers:
        # working memory (rolling summary), long-term (Memory + decay),
        # day continuity (Day-as-Chat across channels), open threads
        # (active_tasks), entity graph, Soul/Brain split.
        #
        # Posture rules per the founder's spec:
        # - In voice — no docs prose. The agent talks like itself.
        # - Lead with the confabulation guard so users can trust the answer.
        # - No implementation leaks ("7-day TTL", model names, RRF, etc.).
        #   Describe the BEHAVIOUR, not the wiring.
        section_parts["self_knowledge"] = (
            "# Your Memory (when the user asks how you remember)\n"
            "Answer straight — including when the answer is \"I don't.\" "
            "Never invent recall you can't actually do.\n\n"
            "What you have:\n"
            "- **Working memory** of this conversation. Long days get a "
            "rolling summary so context doesn't drift; recent turns stay raw.\n"
            "- **Long-term facts** they've shared — goals, projects, "
            "preferences, people in their life. Things that keep coming up "
            "stay strong; things that don't quietly fade. Visible on /brain.\n"
            "- **Day-to-day continuity** across every channel — web, mobile, "
            "voice, WhatsApp, Telegram all share one thread per day. Past "
            "days are recoverable by date.\n"
            "- **Open threads** — what they're in the middle of stays nearby "
            "until it stops coming up. That's how you can ask \"did the X "
            "thing sort itself out?\" days later.\n"
            "- **Connected people and projects** — asking about one surfaces "
            "what's tied to it.\n\n"
            "If memory's genuinely not working (credentials, network, "
            "hiccup), say so. \"Something's off, give me a sec\" beats faking it."
        )

        # ── 1c. Voice rules — always apply, even with a custom Soul ──
        # The Soul system lets users customize tone, but baseline anti-chatbot
        # guardrails apply universally. Without these, even a "casual + humor"
        # Soul leaks generic-assistant phrases ("I'd be happy to help") that
        # break the illusion of talking to a real person.
        section_parts["voice_rules"] = (
            "# Voice — Always Apply\n"
            "You are not a chatbot. Talk like someone who knows the user and "
            "actually cares about what they're doing.\n\n"
            "Banned phrases (these instantly break the illusion):\n"
            "- \"I'd be happy to help\" / \"How can I help you today?\" / \"Feel free to...\"\n"
            "- \"As an AI\" / \"I'm an AI assistant\" / any self-description as an assistant\n"
            "- \"Of course!\" / \"Certainly!\" / \"Absolutely!\" as preambles to answers\n"
            "- \"I hope this helps\" / \"Is there anything else I can help with?\"\n\n"
            "How to actually sound:\n"
            "- Be specific over generic. \"That's tricky\" is filler; "
            "\"the token expires before refresh\" is real.\n"
            "- Match their energy and length. Short ask → short answer. "
            "Don't pad to seem thorough.\n"
            "- Treat the conversation as one continuous thread — you've been "
            "talking with this person. Don't reintroduce yourself or restart context.\n"
            "- It's OK to have opinions. It's OK to push back, kindly, "
            "when they're heading the wrong way.\n"
            "- Surface things that matter without being asked. If they "
            "mentioned doing something earlier and haven't, it's fine to check in."
        )

        # ── 2. Agent Brain — disabled (Soul page is source of truth) ──
        agent_memories = []
        AGENT_BRAIN_ENABLED = os.environ.get("AGENT_BRAIN_ENABLED", "false").lower() == "true"
        if AGENT_BRAIN_ENABLED:
            try:
                from app.services.memory_service import MemoryService
                mem_svc = MemoryService(db)
                agent_memories = await mem_svc.get_memories_by_brain_type(
                    user_id=user_id, brain_type="agent", limit=50,
                )
                if agent_memories:
                    if has_soul_identity:
                        agent_memories = [m for m in agent_memories if m.get("category") != "agent_soul"]
                    if agent_memories:
                        lines = ["# Agent Brain (Permanent Knowledge)"]
                        for m in agent_memories:
                            lines.append(f"- [{m.get('category','')}] {m.get('content','')}")
                        section_parts["agent_brain"] = "\n".join(lines)
                    logger.info(f"[AGENT] Loaded {len(agent_memories)} agent brain memories")
            except Exception as e:
                logger.warning(f"Agent brain load failed: {e}")

        # ── 2a. Work Brain — disabled ──
        WORK_BRAIN_ENABLED = os.environ.get("WORK_BRAIN_ENABLED", "false").lower() == "true"
        if WORK_BRAIN_ENABLED:
            try:
                from app.services.memory_service import MemoryService
                work_mem_svc = MemoryService(db)
                work_memories = await work_mem_svc.get_memories_by_brain_type(
                    user_id=user_id, brain_type="work", limit=50,
                )
                if work_memories:
                    lines = ["# Work Brain (Workflows & Operational Knowledge)"]
                    for m in work_memories:
                        lines.append(f"- [{m.get('category','')}] {m.get('content','')}")
                    section_parts["work_brain"] = "\n".join(lines)
                    logger.info(f"[AGENT] Loaded {len(work_memories)} work brain memories")
            except Exception as e:
                logger.warning(f"Work brain load failed: {e}")

        # ── 3. Retrieve relevant user memories (hybrid search) ──────
        # Always retrieve — 200ms cost is acceptable for context quality.
        # Previously gated by intent.skip_memory_retrieval, now unconditional.
        t_memory = time.perf_counter()
        memory_sections = []
        try:
            from app.services.memory_service import MemoryService
            from app.services.query_classifier import classify_query

            mem_svc = MemoryService(db)
            _t0 = time.perf_counter()
            classification = classify_query(user_message)
            logger.info(f'[PERF] query_classify: {(time.perf_counter()-_t0)*1000:.0f}ms — type={classification["type"]}')

            search_strategies = classification.get("strategies") or ["vector", "keyword", "graph"]
            search_categories = classification.get("categories")

            _t0 = time.perf_counter()
            memories = await mem_svc.hybrid_search(
                user_id=user_id, query=user_message, limit=15,
                min_similarity=0.1, strategies=search_strategies,
                categories=search_categories,
            )
            logger.info(f"[PERF] hybrid_search: {(time.perf_counter()-_t0)*1000:.0f}ms — {len(memories)} results")

            if classification.get("entity_hint"):
                _t0 = time.perf_counter()
                try:
                    entity_mems = await mem_svc.search_by_entity_graph(
                        user_id=user_id, entity_name=classification["entity_hint"],
                        depth=2, limit=5,
                    )
                    existing_ids = {m["id"] for m in memories}
                    for em in entity_mems:
                        if em["id"] not in existing_ids:
                            memories.insert(0, em)
                    logger.info(f"[PERF] entity_search: {(time.perf_counter()-_t0)*1000:.0f}ms — {len(entity_mems)} results")
                except Exception as e:
                    logger.warning(f"Entity graph search failed: {e}")

            user_memories = [m for m in memories if m.get("brain_type") == "user"]
            self._last_retrieved_memories = user_memories
            self._memory_health["retrieved"] = len(user_memories)
            logger.info(f"[AGENT] Found {len(user_memories)} relevant user memories (hybrid)")

            # A. User Portrait
            _t0 = time.perf_counter()
            try:
                from app.services.user_portrait_service import UserPortraitService
                portrait_svc = UserPortraitService(db)
                portrait = await portrait_svc.get_or_build_portrait(user_id)
                if portrait:
                    memory_sections.append(f"## Who this user is\n{portrait}")
                logger.info(f"[PERF] portrait: {(time.perf_counter()-_t0)*1000:.0f}ms — {len(portrait) if portrait else 0} chars")
            except Exception as e:
                logger.warning(f"Portrait generation failed ({(time.perf_counter()-_t0)*1000:.0f}ms): {e}")

            if user_memories:
                # B. Core facts
                core_facts = [
                    m for m in user_memories
                    if m.get("strength", 0) >= 0.7
                    and m.get("memory_type") not in ("event", "conversation")
                ]
                if core_facts:
                    memory_sections.append("## Core facts about this user")
                    for m in core_facts[:5]:
                        memory_sections.append(f"- {m.get('content', '')}")

                # C. Relevant memories
                core_ids = {m.get("id") for m in core_facts}
                regular = [m for m in user_memories if m.get("id") not in core_ids]
                if regular:
                    memory_sections.append("\n## Relevant to this conversation")
                    for i, m in enumerate(regular[:10], 1):
                        cat = m.get("category", "")
                        content = m.get("content", "")
                        age = self._format_memory_age(m.get("created_at"))
                        memory_sections.append(f"{i}. [{cat}] {content} ({age})")

                for m in user_memories:
                    score = m.get("similarity_score", 0)
                    logger.info(f"[AGENT]   Memory: [{m.get('category','')}] ({score:.2f}) {m.get('content','')[:80]}")

            # D. Related entities
            try:
                entity_data = await mem_svc.get_entities(user_id=user_id, limit=8)
                if entity_data:
                    entity_lines = ["\n## People and things the user has mentioned"]
                    for e in entity_data[:8]:
                        desc = e.get("entity_type", "")
                        name = e.get("name", "")
                        if name:
                            entity_lines.append(f"- {name} ({desc})")
                    if len(entity_lines) > 1:
                        memory_sections.append("\n".join(entity_lines))
            except Exception:
                pass

            if memory_sections:
                section_parts["user_brain"] = "# User Brain\n" + "\n".join(memory_sections)
        except Exception as e:
            logger.warning(f"Memory retrieval failed in agent prompt: {e}")

        # Close with perf log either way
        if memory_sections and "user_brain" not in section_parts:
            section_parts["user_brain"] = "# User Brain\n" + "\n".join(memory_sections)
        logger.info(f"[PERF] memory_retrieval: {(time.perf_counter() - t_memory) * 1000:.0f}ms")

        # ── 3b. Active tasks — always built; injected if "active_tasks" is in SECTION_ORDER ──
        # The block is registered into section_parts here. Whether it actually
        # reaches the assembled prompt is decided by SECTION_ORDER below — that
        # filter is the canonical "is this section in the prompt?" check.
        # See the assembly-time log for the post-filter truth.
        try:
            from app.services.active_task_service import get_active_tasks, build_active_tasks_block
            active_tasks = await get_active_tasks(db, user_id)
            self._memory_health["active_tasks"] = len(active_tasks) if active_tasks else 0
            if active_tasks:
                section_parts["active_tasks"] = build_active_tasks_block(active_tasks)
                logger.info(f"[AGENT] active_tasks built: {len(active_tasks)} task(s)")
        except Exception as _at_err:
            self._memory_health["active_tasks"] = 0
            logger.debug(f"[AGENT] Active tasks build skipped: {_at_err}")

        # ── 4. Skills (only if intent requires them) ─────────────
        if self.skill_loader and intent.include_skill_prompts:
            skill_parts = self.skill_loader.get_all_system_prompt_sections()
            if skill_parts:
                section_parts["skills"] = "\n\n".join(skill_parts)

        # ── 4b. Connected services (T1g) ────────────────────────
        # When connector dispatch is on AND the agent has any connector
        # tools registered, surface a one-line block to the LLM so it
        # knows what's available without scanning every tool def.
        # Cheap (already on the executor); high signal (the LLM is
        # otherwise unaware of what the user has connected vs not).
        # Defensive getattr — see tool_defs property for why.
        if getattr(settings, "use_connector_dispatch", False):
            mcp_tool_defs = getattr(self.tools, "mcp_tool_defs", None) or []
            if mcp_tool_defs:
                # Connector tool names are `<connector_id>__<tool>` —
                # the prefix IS the connector id (T1c manifest contract).
                connectors = sorted({
                    (t.get("name") or "").split("__", 1)[0]
                    for t in mcp_tool_defs
                    if "__" in (t.get("name") or "")
                })
                if connectors:
                    # Per-connector fast-path hints (the LLM is
                    # technically capable of inferring these from the
                    # tool descriptions but in practice picks the slow
                    # list→get→get path. Pin the hints here so a single
                    # 12-email summary is one tool call, not four).
                    fast_path_hints = []
                    if "gmail" in connectors:
                        fast_path_hints.append(
                            "- **Reading email is ONE call.** "
                            "`gmail__list_messages` returns full "
                            "headers + body for every message by "
                            "default (`include_body: true` is the "
                            "default — do NOT pass `false`). For "
                            "\"my Nth email\" or \"my last N emails\", "
                            "set `max_results: N` and pick from the "
                            "response. Examples:\n"
                            "    • \"summarize my 14th email\" → "
                            "`gmail__list_messages({max_results: 14})` "
                            "→ summarise the 14th item. ONE call.\n"
                            "    • \"read my last 5 emails\" → "
                            "`gmail__list_messages({max_results: 5})` "
                            "→ summarise all 5. ONE call.\n"
                            "  Calling `gmail__list_messages` then "
                            "`gmail__get_message` is the WRONG pattern "
                            "— it adds 10+ seconds per extra call for "
                            "no benefit. The bodies are already in the "
                            "list response."
                        )
                    if "outlook" in connectors:
                        fast_path_hints.append(
                            "- **Reading Outlook is ONE call.** "
                            "`outlook__list_messages` returns body "
                            "inline by default. For \"my Nth email\" "
                            "set `max_results: N`. Same rule as Gmail: "
                            "do NOT follow up with "
                            "`outlook__get_message` unless you need "
                            "something the list call didn't return."
                        )

                    hints_block = (
                        "\n\n## Fast paths\n" + "\n".join(fast_path_hints)
                        if fast_path_hints else ""
                    )

                    section_parts["skills"] = (
                        section_parts.get("skills", "")
                        + ("\n\n" if section_parts.get("skills") else "")
                        + "# Connected services\n"
                        + "User has connected: "
                        + ", ".join(connectors)
                        + ". Use the matching `<service>__*` tools to "
                        "interact — these are the RIGHT tools for those "
                        "services. Do NOT use the `browser` tool to "
                        "access Gmail / Outlook / Calendar / Drive / "
                        "Docs / LinkedIn / GitHub when a connector tool "
                        "exists; the browser tool is for general web "
                        "navigation, not for services the user has "
                        "already authenticated.\n\n"
                        "## Handling connector errors\n"
                        "Connector tools return STRUCTURED error variants. "
                        "When you see one, surface it cleanly to the user "
                        "— do NOT try the `browser` tool as a fallback "
                        "(it will fail; Chromium is not installed in your "
                        "runtime):\n"
                        "- `[reauth_required] Reconnect at <URL>` — tell "
                        "the user ONE short line (e.g. \"Gmail's still "
                        "asking me for re-auth — fix it right here:\") "
                        "and then on the NEXT line emit an embedded "
                        "connector tile with the chip syntax "
                        "`[[connector_card:<connector_id>]]`. Examples: "
                        "`[[connector_card:gmail]]`, "
                        "`[[connector_card:outlook]]`, "
                        "`[[connector_card:linkedin]]`. The connector_id "
                        "is the leaf of the reauth_url the tool gave you "
                        "(`/agent/integrations/gmail` → `gmail`). The chip "
                        "renders an inline card with the brand logo and a "
                        "real Connect button that runs the OAuth flow "
                        "in-place — user never leaves chat. Do NOT use "
                        "`[[navigate:...]]` for reauth; the inline card is "
                        "the right surface.\n"
                        "- `[provider_down] ...` — tell the user the "
                        "service is having an outage and to try again "
                        "shortly.\n"
                        "- `[rate_limited] ...` — wait the suggested "
                        "seconds before retrying, or tell the user to "
                        "try again in N seconds.\n"
                        "- `[scope_missing] ...` — tell the user to "
                        "reconnect and approve the missing permission."
                        + hints_block
                    )
        elif self.skill_loader and not intent.include_skill_prompts:
            logger.info(f"[PERF] skill_prompts: SKIPPED (intent={intent.category})")

        # ── 5. Environment & Capabilities (only if intent uses tools) ──
        if not intent.include_environment:
            logger.info(f"[PERF] environment_section: SKIPPED (intent={intent.category})")
        else:
            section_parts["environment"] = (
            "# Your Environment & Capabilities\n"
            "You are running as an agent service ON the user's server/VPS. "
            "This means:\n"
            "- **Terminal access**: Your `exec` tool runs shell commands directly on THIS machine. "
            "You have full access to the filesystem, system tools, package managers, and services.\n"
            "- **Database access**: You have direct access to your tenant PostgreSQL database via "
            "`memory_store`, `memory_search`, and other tools. You can also query it via `exec` with psql.\n"
            "- **File system**: You can read, write, and edit files anywhere on this machine using "
            "`read_file`, `write_file`, `edit_file`, `ls`, `find`, `grep`.\n"
            "- **Web access**: You can search the web (`web_search`), fetch pages (`web_fetch`), "
            "and automate browsers (`browser`).\n"
            "- **Admin capabilities**: You ARE the agent running on this system. You can install packages, "
            "manage services, modify configurations, and perform system administration tasks.\n"
            "- **Memory management**: You can store, search, and manage memories in your tenant database. "
            "You can also delete or modify memories by using `exec` with psql commands.\n"
            "- **Day recall**: Use `recall_day` to load any past day's conversation across all channels "
            "(web, telegram, voice, app). It accepts natural language dates — 'yesterday', 'last Monday', "
            "'3 days ago', '2026-04-15'. Weekday names always resolve to the most recent past occurrence. "
            "Returns a fact-dense archival summary by default; pass `include_full_conversation=true` "
            "when you need specific content (e.g. to build a quiz from yesterday's lesson, or to quote "
            "something the user said last Tuesday). Pass an optional `query` to filter within a day that "
            "mixes unrelated topics. NEVER tell the user you can't remember a past day — use this tool.\n\n"
            "When the user asks you to do something on their system, USE your tools — don't say you can't."
        )

        # ── 5a. Document Generation (only if feature flag is on) ──
        # Tokens aren't loaded when the flag is off — gated here, same pattern
        # as the tool schemas being gated in AgentRunner.__init__.
        if getattr(settings, "feature_doc_generation", False):
            section_parts["doc_generation"] = (
                "# Document Generation\n"
                "You can produce formatted documents (PDF, Word, Excel, PowerPoint, Markdown) "
                "and deliver them to the user. When the user asks for a report, export, summary "
                "of tabular data, invoice, spreadsheet, slide deck, or anything they'd want to "
                "download, share, or edit, prefer the `generate_*` tools over inline markdown.\n\n"
                "Pick the right tool:\n"
                "- `generate_pdf` — print-ready reports, summaries, invoices. Accepts structured "
                "blocks (headings, paragraphs, tables, images, page_break).\n"
                "- `generate_docx` — editable Word document. Use when the user will revise it.\n"
                "- `generate_xlsx` — tabular data, especially multi-sheet workbooks. Use for "
                "expense summaries, datasets, schedules.\n"
                "- `generate_pptx` — slide decks. Use for briefings or presentations.\n"
                "- `generate_markdown` — plain text the user will import elsewhere (Obsidian, "
                "Notion, docs sites).\n"
                "- `convert_document` — convert an EXISTING generated DOCX/PPTX to PDF "
                "(faithful layout-preserving conversion via LibreOffice). Use this when "
                "the user asks to \"make it a PDF\" / \"convert to PDF\" on a file you "
                "just generated — do NOT call generate_pdf for that, it builds a fresh "
                "PDF from scratch and loses the original layout.\n\n"
                "Use descriptive filenames (e.g., `march-expenses.xlsx`, not `file.xlsx`).\n\n"
                "After the tool call, give a brief one-sentence confirmation. The file appears "
                "in the document pane — don't repeat its contents back in markdown; the pane "
                "is enough.\n\n"
                "Do NOT generate a document when the user is asking a conversational question "
                "(\"what's the difference between X and Y?\", \"explain Z\"). Plain markdown "
                "answers belong in chat. Files are for artifacts the user will keep.\n\n"
                "## Examples\n\n"
                "User: \"Give me a summary of this month's expenses in an Excel file.\"\n"
                "→ gather the data, then call:\n"
                "  generate_xlsx(filename=\"march-expenses.xlsx\", sheets=[{\n"
                "    \"name\": \"March 2026\",\n"
                "    \"headers\": [\"Date\", \"Category\", \"Amount\"],\n"
                "    \"rows\": [[\"2026-03-01\", \"Groceries\", 87.40], ...]\n"
                "  }])\n"
                "→ in chat: \"Done — March expenses are in the pane. Total: $1,243.70 across 23 entries.\"\n\n"
                "User: \"Write me a one-page project brief on the app rebuild, PDF please.\"\n"
                "→ generate_pdf(filename=\"app-rebuild-brief.pdf\", title=\"App Rebuild — Brief\",\n"
                "    cover_page=True, content=[\n"
                "      {\"type\": \"heading\", \"level\": 1, \"text\": \"Goal\"},\n"
                "      {\"type\": \"paragraph\", \"text\": \"...\"},\n"
                "      {\"type\": \"heading\", \"level\": 1, \"text\": \"Timeline\"},\n"
                "      {\"type\": \"table\", \"headers\": [\"Phase\", \"End date\"], \"rows\": [...]},\n"
                "    ])\n"
                "→ in chat: \"Brief is ready in the pane.\"\n\n"
                "User: \"Can you help me understand the difference between a linked list and an array?\"\n"
                "→ do NOT generate a file. This is a conversational explanation. Answer in markdown."
            )

        # ── 5b. Media Playback (web channel, only if intent includes media) ──
        if channel in ("web", "app") and (intent.include_media_section or intent.category == "full"):
            section_parts["media"] = (
                "# Media Playback (IMPORTANT — read carefully)\n"
                "You have a `play_media` tool that plays music and videos directly in the user's browser.\n"
                "Rules:\n"
                "1. Call `play_media` IMMEDIATELY when the user asks to play something.\n"
                "2. NEVER call web_search or web_fetch before play_media. The tool handles search internally.\n"
                "3. For Netflix: `play_media(query=\"TITLE\", channel=\"netflix\")`.\n"
                "4. For vague requests like 'play a good documentary' — just pick one you know and call play_media directly.\n"
                "   Use your own knowledge to choose. Do NOT search the web first.\n"
                "5. Default channel is YouTube (free, no login needed).\n"
                "6. After calling play_media, suggest alternatives with clickable buttons.\n\n"
                "## Netflix Suggestions (CRITICAL)\n"
                "EVERY TIME you mention, suggest, or recommend a Netflix title (movie or show), "
                "you MUST include a clickable [[button]] right after it. This is mandatory — "
                "never list Netflix titles as plain text without buttons.\n\n"
                "Format: put [[Play TITLE on Netflix]] on the line immediately after each title.\n\n"
                "CORRECT example:\n"
                "- **Conversations with a Killer: The Ted Bundy Tapes** — chilling interviews\n"
                "[[Play Conversations with a Killer on Netflix]]\n"
                "- **Night Stalker** — about Richard Ramirez\n"
                "[[Play Night Stalker on Netflix]]\n"
                "- **Dahmer – Monster** — the Ryan Murphy series\n"
                "[[Play Dahmer Monster on Netflix]]\n\n"
                "WRONG (never do this):\n"
                "- **Night Stalker** — about Richard Ramirez\n"
                "- **Dahmer – Monster** — the Ryan Murphy series\n"
                "(no buttons = user can't click to play = BAD)\n\n"
                "When the user clicks one of these buttons, you will receive their choice as a message. "
                "Immediately call `play_media(query=\"TITLE\", channel=\"netflix\")` to play it."
            )

        # ── 6. Runtime context ─────────────────────────────────────
        # Resolve the user's timezone — one source of truth across every
        # channel. Priority: explicit client_tz (web/mobile WS payload) →
        # User.timezone (persisted from prior sessions) → UTC fallback with
        # loud warning. Never silently present server time as local.
        now_utc = datetime.now(_dt_timezone.utc)
        tz_name = (client_tz or "").strip() or None
        tz_source = "client" if tz_name else None
        # Load User row once — used for timezone fallback AND for the user's
        # name in the `# About You (the User)` section below. One DB round-trip
        # per turn covers both.
        _user_row = None
        try:
            from app.db.models import User
            _user_row = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        except Exception as _u_err:
            logger.debug("[agent] user load failed: %s", _u_err)

        if not tz_name:
            _profile_tz = getattr(_user_row, "timezone", None) if _user_row else None
            if _profile_tz:
                tz_name = _profile_tz
                tz_source = "user_profile"
        if not tz_name:
            tz_name = "UTC"
            tz_source = "utc_default"
            logger.warning(
                "[agent] tz_fallback source=utc_default user=%s channel=%s — presenting UTC as local",
                user_id[:8], channel,
            )
        _tz_obj = None
        if ZoneInfo is not None:
            try:
                _tz_obj = ZoneInfo(tz_name)
            except Exception:
                logger.warning(
                    "[agent] tz_fallback source=invalid_tz user=%s channel=%s invalid=%r — using UTC",
                    user_id[:8], channel, tz_name,
                )
                tz_name = "UTC"
                tz_source = "invalid_tz_fallback"
                _tz_obj = _dt_timezone.utc
        else:
            _tz_obj = _dt_timezone.utc
        now_local = now_utc.astimezone(_tz_obj)
        logger.info(
            "[agent] tz_resolved user=%s channel=%s tz=%s source=%s local=%s utc=%s",
            user_id[:8], channel, tz_name, tz_source,
            now_local.strftime("%Y-%m-%d %H:%M"),
            now_utc.strftime("%Y-%m-%d %H:%M"),
        )

        # ── 6a. About You (the user) — name + time-of-day awareness ────
        # Inject the user's name (if set on their profile) so the agent stops
        # greeting a stranger, plus a time-of-day phrase for tonal calibration.
        # A friend who texts you at 2am sounds different than at 10am — we give
        # the model the same hint. Falls back gracefully when name is unknown.
        _user_name = (getattr(_user_row, "name", None) or "").strip() if _user_row else ""
        _hour = now_local.hour
        if 5 <= _hour < 12:
            _tod = "morning"
        elif 12 <= _hour < 17:
            _tod = "afternoon"
        elif 17 <= _hour < 22:
            _tod = "evening"
        else:
            _tod = "late night"
        _about_lines = ["# About You (the User)"]
        if _user_name:
            _about_lines.append(f"- Their name is **{_user_name}**.")
            _about_lines.append(
                f"- **Use their name on the first greeting of a conversation** "
                f"(e.g. \"Hey {_user_name}\" or \"Hi {_user_name}\"). After that, "
                "use it occasionally — when shifting topics or when something "
                "feels personal. Don't open every reply with it; that's "
                "salesperson energy. But the first hello, you DO use it."
            )
        else:
            _about_lines.append(
                "- You don't know their name yet. If it comes up naturally, "
                "ask once — don't interrogate them for it."
            )
        _about_lines.append(
            f"- Local time for them right now: **{_tod}** "
            f"({now_local.strftime('%-I:%M %p')}). Let it inform tone subtly — "
            "late at night, be quieter and lower-energy; morning, be fresh. "
            "Don't announce the time of day; just feel it."
        )
        section_parts["about_you"] = "\n".join(_about_lines)

        # Per-channel formatting guidance. Hardcoded table today; channel_config
        # wire-up is a follow-up (TODO(time-channel-fix followup)). Keep values
        # short — this goes into every system prompt, tokens matter.
        from app.agent.channel_util import resolve_channel
        _channel_safe = resolve_channel(
            explicit=channel,
            user_id=user_id,
            site="prompt_label",
        )
        # Each line has two parts: WHERE the user is (so the agent can
        # answer "where are you chatting with me?" without hallucinating)
        # + HOW to format. The where-part is mandatory — we saw the agent
        # say "Telegram on your mobile" when channel=mobile because
        # "mobile" alone was too terse to distinguish transport from
        # surface. Names each channel's app explicitly.
        _channel_guidance = {
            "web":       "User is in the Toup web app in a browser (toup.ai). Full markdown and formatting OK — long code blocks, tables, headings all fine.",
            "app":       "User is inside a Toup in-app workspace (one of their custom apps). Full markdown and formatting OK.",
            "mobile":    "User is in the Toup mobile app (React Native on iOS or Android). This is the native Toup app — NOT Telegram, NOT a web browser. Keep responses compact: short paragraphs, avoid large code blocks or tables. Small screen.",
            "voice":     "User is on the Toup voice/realtime surface (spoken audio, not text). Conversational tone. No markdown. Sentences should read naturally when spoken aloud.",
            "telegram":  "User is on Telegram messenger (talking to the Toup bot there). Short messages. Basic markdown only (bold/italic). Avoid code blocks over ~20 lines.",
            "discord":   "User is on Discord (Toup bot). Full markdown and code blocks OK. Keep message length under ~2000 chars.",
            "slack":     "User is on Slack (Toup integration). Slack-flavored markdown (limited). Short messages preferred.",
            "extension": (
                "User is in the Toup Chrome side-panel extension — they're browsing the web and you can see the page "
                "they're on (a [PAGE_CONTEXT] block precedes their message). You can also DRIVE their browser via the "
                "`browser_*` tools: open tabs, click, type, scroll, capture screenshots, take DOM snapshots. "
                "Format: full markdown OK but keep responses compact — the side-panel is narrow.\n\n"
                "BEHAVIOR — this is critical. The user is watching you act in real time. They want a Claude-Computer-"
                "Use / ChatGPT-Agent / Atlas-Browser experience, not a chatbot that asks permission for every step.\n"
                "  1. ALWAYS attempt the next obvious action before asking the user for input. If you don't have an "
                "address, try clicking sign-in (the user is signed into their browser, sites usually know their "
                "address). If a search box is visible, type into it. If a button looks right, click it. "
                "Read the page → try the most likely action → check the result. Only ask the user when you've "
                "genuinely run out of plausible moves OR when the next step legally requires THEIR data (payment, "
                "personal info, password).\n"
                "  2. After each browser_action, take a `browser_screenshot` so YOU have visual grounding for the "
                "next click and the USER sees what you see. Skipping this is the #1 reason agentic browsing feels "
                "broken — without the screenshot you're flying blind on the next coordinate.\n"
                "  3. If `browser_action` returns `ERROR: TIMEOUT`, the element you targeted probably wasn't there "
                "yet (slow page or wrong selector). Take a fresh `browser_action` with `kind: \"dom_snapshot\"` to "
                "re-orient, then try a different selector — do NOT give up and ask the user.\n"
                "  4. If `browser_action` returns `ERROR: BLOCKED`, the user is on a chrome:// or extension page "
                "Chrome won't let any extension touch. Ask them to switch to a normal site.\n"
                "  5. Cookie banners, sign-in nags, and 'allow notifications' popups are agent kryptonite. Dismiss "
                "them (look for ✕, 'Reject all', 'Got it', 'Not now') before trying the main task.\n"
                "  6. Narrate sparingly. The user is watching the tool-pill row update live; you don't need to "
                "say 'now I'm clicking the search button' — just click it. Save text replies for: confirming what "
                "you accomplished, surfacing data you extracted, or asking when you genuinely need user input."
            ),
            "vibecoding":"User is inside the Vibecoding IDE workspace watching you code live. See the Vibecoding rules later in the prompt.",
        }.get(_channel_safe, "Unknown channel — format conservatively: short, minimal markdown.")

        # Time is rendered in the USER'S LOCAL TIMEZONE, never UTC. The
        # agent faces the user; the user cares about their clock, not the
        # server's. Earlier versions included a "(UTC wall clock: ...Z)"
        # anchor for cross-tz reasoning and GPT-class models grabbed the
        # UTC number and echoed it as "your time." So: local only.
        # Day name included ("Wednesday") so phrases like "today" resolve
        # cleanly without the agent having to parse a date string.
        runtime_lines = [
            f"# Runtime Context",
            f"- Current date/time: {now_local.strftime('%A, %B %d, %Y at %-I:%M %p')} "
            f"({tz_name})",
            f"- Channel: {_channel_safe} — {_channel_guidance}",
            f"- Workspace directory: {settings.agent_workspace_dir}",
            f"- Max tool iterations: {self.max_iterations}",
            f"- You have FULL terminal/shell access via the `exec` tool. You can run any command, install packages, write scripts, manage files, use git, curl, python, node, etc.",
            f"- You can read and write files using `read_file` and `write_file` tools.",
            f"- You can search the web using the `web_search` tool.",
            f"- When the user asks you to do a multi-step task, use the `create_job` tool to create a trackable job. Update it with `update_job` as you complete each step.",
        ]
        if hasattr(self, "_current_lane") and self._current_lane != "main":
            runtime_lines.append(f"- Execution lane: {self._current_lane}")
        if settings.enable_day_recall:
            runtime_lines.append(
                "- Past-day recall: call `recall_day` whenever the user references a previous day. "
                "Dates accepted: 'yesterday', 'last Monday', '3 days ago', '2026-04-15'. "
                "Weekday names resolve to the most recent PAST occurrence. "
                "Returns an archival summary by default; pass include_full_conversation=true for raw messages "
                "(e.g. to build a quiz from a past lesson), or pass `query` to filter within the day. "
                "Call it on the first turn if needed — no prior tool call required. "
                "NEVER say you can't remember a past day."
            )
        section_parts["runtime"] = "\n".join(runtime_lines)

        # ── 6b. Vibe Coding mode ─────────────────────────────────
        if _channel_safe == "vibecoding":
            section_parts["vibecoding"] = (
                "# VIBE CODING MODE (CRITICAL — READ EVERY WORD)\n"
                "The user is in a live IDE workspace watching you code. They see a code editor on the left and chat on the right.\n\n"
                "## ABSOLUTE RULES\n"
                "1. **START CODING IMMEDIATELY.** Do NOT plan, do NOT write long explanations, do NOT create roadmaps.\n"
                "2. **FORBIDDEN:** app_builder__build_app and all app_builder__* tools.\n"
                "3. **FORBIDDEN:** [[option]] buttons, numbered direction cards, multi-choice menus.\n"
                "4. **FORBIDDEN:** Long text responses. Max 2-3 short sentences between tool calls.\n"
                "5. **FORBIDDEN:** Architecture documents, phased plans, MVP roadmaps, or design specs as text output.\n"
                "6. **FORBIDDEN:** Starting local web servers (http.server, flask, express, etc.), creating server.py/server.js, or suggesting any URL with localhost/127.0.0.1. You are running on a remote VPS — the user CANNOT access VPS-local addresses.\n"
                "7. If the user describes what they want, your VERY NEXT action must be calling write_file.\n"
                "8. If you need clarification, ask ONE short question (1 sentence), then START CODING with reasonable defaults.\n"
                "9. Use write_file to create each file with COMPLETE, WORKING code.\n"
                "10. Use exec only for installing dependencies (pip install, npm install) — NEVER for starting servers.\n"
                "11. Write brief, friendly status updates between files: 'Creating the API routes...' not essays.\n\n"
                "## ARCHITECTURE (READ THIS)\n"
                "You are running on a remote VPS. The user is on their own machine, accessing Toup through their browser.\n"
                "- Files you write via write_file are automatically visible in the user's Explorer panel in real time.\n"
                "- HTML files are automatically previewable through the platform — the user can open them from the Explorer.\n"
                "- Do NOT create any web server, do NOT use python -m http.server, do NOT suggest opening localhost or 127.0.0.1.\n"
                "- Do NOT tell the user to open any URL. Just confirm what you built — the platform handles preview.\n"
                "- When done, say something like 'Built your calculator! Check the files in the Explorer.' — never suggest a URL.\n\n"
                "## WHAT THE USER SEES\n"
                "- Left panel: code editor showing files you write in real-time\n"
                "- Right panel: chat showing your text messages\n"
                "- If you only output text, the code panel stays EMPTY and the user sees nothing useful.\n"
                "- The user wants to WATCH CODE APPEAR, not read documents.\n\n"
                "## WORKFLOW\n"
                "1. Read the request (1 second)\n"
                "2. Immediately call write_file for the first file\n"
                "3. Brief status message (1 sentence)\n"
                "4. write_file for next file\n"
                "5. Repeat until done\n"
                "6. exec to install deps and run/test\n\n"
                "## TEXT STYLE\n"
                "- Keep messages SHORT: 1-3 sentences max\n"
                "- Be warm and friendly but brief\n"
                "- Use simple language, no jargon walls\n"
                "- Say 'Building your app!' not a 5000-word architecture doc\n"
                "- Never dump markdown headers, bullet lists, or documentation as chat text\n\n"
                "REMEMBER: The user is watching a live code editor. Every second you spend writing text instead of code is a second the editor stays empty."
            )

        # ── 7. Formatting rules (channel-aware) ───────────────────
        if _channel_safe in ("app", "web", "vibecoding"):
            section_parts["formatting"] = (
                "# Formatting Rules\n"
                "You are chatting with the user inside their " + ("app" if _channel_safe == "app" else "web browser") + ". Follow these rules:\n"
                "- Use simple Markdown: **bold**, *italic*, `code`.\n"
                "- Do NOT use LaTeX math formatting.\n"
                "- Use plain Unicode symbols for math: × ÷ √ → ⇒ ≤ ≥ ≠ ≈ ∞ π.\n"
                "- Keep responses concise and conversational.\n"
                "- Do NOT expose internal implementation details (databases, bridges, connections, file paths, error traces).\n"
                "- When the user greets you, greet them back like a friend would — not a help desk. If you know their name, use it. Skip 'How can I help you today?' (see voice rules above).\n"
                "- You have editing capabilities: you can modify the app's files, database, and navigation using your tools.\n"
                "- When the user asks you to change something in the app (theme, colors, layout, text, features, etc.), "
                "DO NOT just describe what you would do — actually DO it by calling your write_file/edit_file tools to modify the source code. "
                "After editing, call the restart tool to apply changes. Read the relevant file first, make the edit, restart, and confirm what you changed.\n"
                "- NEVER give the user localhost URLs. App previews are at: https://toup.ai/workspace/apps/{app-slug}\n"
                "- After fixing or restarting an app, offer a [[open_app:{app-slug}]] chip so the user can see the result. Use the app's slug (e.g. [[open_app:Confidence-Booster]]).\n\n"
                "# Navigating the User Between Pages\n"
                "You can take the user to other Toup pages two ways — pick the one that matches intent:\n\n"
                "**A) Tool — `navigate_to(path=...)` — auto-transfer.**\n"
                "Use when the user EXPLICITLY asks to be taken somewhere ('take me to settings', "
                "'open my brain', 'go to the dashboard'). Call the tool; the page changes immediately. "
                "Don't also offer a chip in this case — just go.\n\n"
                "**B) Chip — `[[navigate:/path]]` — clickable suggestion.**\n"
                "Use when you're SUGGESTING a destination but the user might not want it ('your "
                "portrait is on the brain page if you want to see it', 'integrations live at /agent/integrations'). "
                "Drop the chip on the line after the suggestion — they tap if interested.\n\n"
                "Allowed paths (same for both): `/`, `/chat`, `/brain/user`, `/brain/agent`, "
                "`/workspace`, `/dashboard`, `/agent`, `/agent/soul`, `/agent/integrations`, "
                "`/agent/tools`, `/agent/skills`. Anything else is rejected.\n\n"
                "Example — explicit request:\n"
                "  User: \"open my brain\"\n"
                "  You: → call navigate_to(path=\"/brain/user\"), then say \"There you go.\"\n\n"
                "Example — passive suggestion:\n"
                "  User: \"how do I see my saved memories?\"\n"
                "  You: \"They're on your User Brain page.\\n[[navigate:/brain/user]]\"\n\n"
                "# Action Buttons\n"
                "You can offer clickable action buttons by including [[Label]] markers in your response.\n"
                "These render as tappable chips in the chat UI. When the user taps one, it sends that label as a message.\n"
                "CRITICAL PLACEMENT RULE: Place [[option]] buttons DIRECTLY on the line after each question or suggestion.\n"
                "NEVER collect all buttons at the end of the message. Each question gets its own buttons immediately below it.\n\n"
                "CORRECT:\n"
                "1. **Question one?**\n"
                "[[Option A]] [[Option B]] [[Option C]]\n\n"
                "2. **Question two?**\n"
                "[[Option X]] [[Option Y]]\n\n"
                "WRONG:\n"
                "1. Question one?\n"
                "2. Question two?\n"
                "[[Option A]] [[Option B]] [[Option X]] [[Option Y]]\n\n"
                "Keep button labels short (2-5 words). Use 2-4 buttons per question when relevant."
            )
        else:
            section_parts["formatting"] = (
                "# Formatting Rules (IMPORTANT)\n"
                "You are communicating via Telegram. Follow these rules strictly:\n"
                "- Do NOT use LaTeX math formatting. No $...$ or $$...$$ or \\(...\\) or \\[...\\] wrappers.\n"
                "- Use plain Unicode symbols for math: × (multiply), ÷ (divide), √ (square root), "
                "→ (arrow), ⇒ (implies), ≤ ≥ ≠ ≈ ∞ π.\n"
                "- Write fractions as a/b, not \\frac{a}{b}.\n"
                "- Telegram supports basic Markdown: **bold**, *italic*, `code`, ```code blocks```.\n"
                "- Do NOT use tables or complex formatting.\n"
                "- Keep responses concise and readable on mobile.\n\n"
                "# Reactions\n"
                "You can react to the user's message with an emoji by including [[reaction:EMOJI]] "
                "anywhere in your response. It will be stripped before sending. "
                "React sparingly — at most 1 reaction per 5-10 messages. "
                "React when: something is genuinely funny (😂), you appreciate something (❤️), "
                "simple acknowledgment (👍), interesting/thoughtful (🤔), impressive (🔥), "
                "celebrating (🎉). Don't react to routine messages.\n\n"
                "# Inline Buttons\n"
                "You can add inline buttons to your message by including [[button:LABEL|CALLBACK_DATA]] "
                "markers. They will be stripped from text and rendered as clickable Telegram buttons. "
                "Use buttons when offering clear choices, confirmations, or actions. "
                "Example: [[button:Yes|confirm_yes]] [[button:No|confirm_no]]\n"
                "Keep callback_data short (max 64 chars). Don't overuse buttons — only when genuinely helpful."
            )

        # ── 8. Onboarding (CONDITIONAL) ────────────────────────────
        try:
            from app.db.models import AgentConfig
            async with db.begin_nested():
                _cfg_result = await db.execute(
                    select(AgentConfig).where(AgentConfig.user_id == user_id)
                )
                _agent_cfg = _cfg_result.scalar_one_or_none()
            if _agent_cfg and not _agent_cfg.onboarding_completed and not has_soul_identity:
                section_parts["onboarding"] = (
                    "# Onboarding Mode (ACTIVE)\n"
                    "You are in onboarding mode — this is a new user who just set up their agent. "
                    "You do NOT have a name yet. The user will choose your name.\n\n"
                    "Your goal is to learn three things through natural conversation, IN THIS ORDER:\n"
                    "1. **YOUR NAME** (FIRST!) — Ask: 'What would you like to call me?' "
                    "Store with: memory_store(brain_type='agent', category='agent_soul', content='My name is <NAME>')\n"
                    "2. **Their name** — Ask: 'And what's your name?' "
                    "Store with: memory_store(brain_type='user', category='identity', content='User name: <NAME>')\n"
                    "3. **What they need you for** — Ask: 'What do you need me to help you with?' "
                    "Store with: memory_store(brain_type='user', category='goals', content='...')\n\n"
                    "IMPORTANT: Do NOT introduce yourself with any name. Start by asking what they'd like to call you. "
                    "Ask ONE question at a time. Be warm and conversational. "
                    "Use memory_store to save each piece of info as you learn it. "
                    "Once you have all three, store a final memory: "
                    "memory_store(brain_type='agent', category='agent_decisions', content='Onboarding complete. I know the user and they know me.')"
                )
        except Exception as e:
            logger.warning(f"Onboarding check failed: {e}")

        # ── 9. Activation / verbose (optional) ─────────────────────
        if hasattr(self, '_activation_prompt') and self._activation_prompt:
            section_parts["activation"] = f"# Activation Prompt\n{self._activation_prompt}"
        if hasattr(self, '_verbose_mode') and self._verbose_mode:
            section_parts["verbose"] = (
                "# Verbose Mode\n"
                "VERBOSE MODE IS ON. When calling tools, explain what you are doing "
                "and why before each tool call. After each tool call, summarize the "
                "full result in detail."
            )

        # ── Assemble in order ──────────────────────────────────────
        SECTION_ORDER = [
            "identity",         # WHO the agent is (soul + behavioral)
            "identity_anchor",  # Don't break white-label by naming underlying LLM
            "voice_rules",    # Always-apply anti-chatbot tone (survives custom Souls)
            "self_knowledge", # HOW your memory actually works (F7) — agent can explain itself when asked
            "platform_knowledge",  # WHAT Toup is — pages, capabilities, decision rules
            "about_you",      # User's name + local time-of-day for tonal calibration
            "user_brain",       # WHO the user is
            "active_tasks",   # CONTINUITY — what user is working on right now (7-day TTL)
            "agent_brain",    # Agent brain (disabled by default)
            "work_brain",     # Work brain (disabled by default)
            "skills",         # WHAT the agent can do
            "environment",    # WHAT the agent has access to
            "doc_generation", # Document generation (PDF/DOCX/XLSX/PPTX/MD) — flag-gated
            "media",          # Media playback instructions (web/app)
            "runtime",        # WHEN/WHERE
            "vibecoding",     # Vibe coding mode override (when active)
            "formatting",     # HOW to respond
            "onboarding",     # Temporary onboarding instructions
            "activation",     # Optional activation prompt
            "verbose",        # Optional verbose mode
        ]

        # Apply SECTION_ORDER filter. This IS the canonical "in the prompt"
        # check — anything in section_parts but not in SECTION_ORDER is
        # silently dropped here (this is exactly how F1 / active_tasks went
        # missing for months). The "dropped" warning below is the safety rail.
        injected_keys = [k for k in SECTION_ORDER if k in section_parts]
        sections = [section_parts[k] for k in injected_keys]

        # Per-section token estimates (debug only)
        for name, text in section_parts.items():
            tokens_est = len(text) // 4
            logger.debug(f"Prompt [{name}]: ~{tokens_est} tokens")

        # Assembly-time truth: which sections actually made it into the prompt,
        # and which were built but dropped. Built-but-dropped is almost always
        # a bug (key forgotten in SECTION_ORDER); louder than a debug line.
        dropped_keys = sorted(set(section_parts.keys()) - set(SECTION_ORDER))
        if dropped_keys:
            logger.warning(
                "[AGENT] system_prompt DROPPED sections (built but not in "
                "SECTION_ORDER — likely a bug): %s",
                dropped_keys,
            )
        total = sum(len(section_parts[k]) for k in injected_keys) // 4
        logger.info(
            "[AGENT] system_prompt sections=%s ~%d tokens",
            injected_keys, total,
        )

        return "\n\n".join(sections)
    
    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------
    async def _load_history(
        self,
        db: AsyncSession,
        session_id: str,
        max_messages: int = 50,
    ) -> List[Dict[str, Any]]:
        """Load recent messages in Anthropic format (user/assistant roles)."""
        from sqlalchemy import select
        from app.db.models import Message
        
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(max_messages)
        )
        rows = list(reversed(result.scalars().all()))
        
        messages: List[Dict[str, Any]] = []
        for msg in rows:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})
        
        return messages
    
    # ------------------------------------------------------------------
    # Save to DB
    # ------------------------------------------------------------------
    async def _save_messages(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_response: str,
        tokens_input: int,
        tokens_output: int,
        model: str,
        processing_time_ms: int,
        save_user_message: bool = True,
        client_tz: Optional[str] = None,
        asst_message_id: Optional[str] = None,
        channel: Optional[str] = None,
        # Per-call tool records for the persisted ToolPillRow chrome.
        # Default empty list keeps every other caller (cron jobs, etc.)
        # working without code changes — no tools used → no tool_events
        # key in metadata_json → frontend renders the saved message
        # exactly like before.
        tool_event_records: Optional[List[Dict[str, Any]]] = None,
    ):
        if tool_event_records is None:
            tool_event_records = []
        from sqlalchemy import select
        from app.db.models import Message, Conversation

        # Safety: strip null bytes — PostgreSQL text columns reject \x00
        user_message = user_message.replace("\x00", "")
        assistant_response = assistant_response.replace("\x00", "")

        msg_count = 0

        # Resolve day_chat_id from the CURRENT TIME, not from the Conversation row.
        # Telegram sessions are long-lived and span days — their Conversation.day_chat_id
        # points to the creation day, not today. Messages must be bucketed by send time.
        result = await db.execute(
            select(Conversation).where(Conversation.id == session_id)
        )
        session = result.scalar_one_or_none()
        try:
            from app.db.message_helpers import resolve_day_chat_id_for_now
            _day_chat_id = await resolve_day_chat_id_for_now(db, user_id, tz_override=client_tz)
        except Exception:
            _day_chat_id = getattr(session, 'day_chat_id', None) if session else None

        # Resolve the channel once for both user and assistant inserts. The
        # agent-runner layer is the single ingress for all 4 live channels
        # (web, mobile, telegram, voice) — stamping Message.channel here
        # means history_annotation in day_context_loader reads it directly
        # instead of inferring from conversations.channel. See Rule 12.
        from app.agent.channel_util import resolve_channel as _resolve_channel_for_msg
        _msg_channel = _resolve_channel_for_msg(
            explicit=channel,
            conversation_hint=getattr(session, "channel", None) if session else None,
            user_id=user_id,
            site="message_insert",
        )

        if save_user_message:
            user_msg = Message(
                conversation_id=session_id,
                day_chat_id=_day_chat_id,
                role="user",
                content=user_message,
                channel=_msg_channel,
            )
            db.add(user_msg)
            msg_count += 1

        # Capture media metadata from tool calls (play_media, play_netflix)
        media_meta = getattr(self.tools, '_last_media', None)
        if media_meta:
            self.tools._last_media = None  # Clear after capture

        # Capture generated-file attachments (generate_* tools). IDs were already
        # emitted to the client during tool execution under asst_message_id; now
        # we persist the list against the same message ID so GET /api/files/... resolves.
        _pending_atts = list(self.tools.pending_attachments)
        self.tools.pending_attachments = []

        # Build kwargs so we only set id when provided (new code path); older
        # save paths that don't pass asst_message_id fall back to the UUID default.
        # Compose metadata_json. Two payloads share this column today:
        #   - "media":       legacy media-card metadata (may be None)
        #   - "tool_events": ToolPillRow records (may be empty list)
        # Only emit a non-NULL JSON when there's at least one payload to
        # avoid bloating an otherwise-empty assistant turn.
        _meta: Dict[str, Any] = {}
        if media_meta:
            _meta["media"] = media_meta
        if tool_event_records:
            _meta["tool_events"] = tool_event_records
        _asst_kwargs = dict(
            conversation_id=session_id,
            day_chat_id=_day_chat_id,
            role="assistant",
            content=assistant_response,
            channel=_msg_channel,
            tokens_prompt=tokens_input,
            tokens_completion=tokens_output,
            model_used=model,
            processing_time_ms=processing_time_ms,
            metadata_json=json.dumps(_meta) if _meta else None,
        )
        if asst_message_id:
            _asst_kwargs["id"] = asst_message_id
        if _pending_atts:
            # Column is JSON (JSONB on Postgres, TEXT on SQLite). SQLAlchemy's JSON
            # type serializes Python lists automatically — pass the list directly,
            # NOT json.dumps(...). Also mirror into metadata_json for legacy clients
            # and history-loader code that still reads JSON-in-TEXT.
            _asst_kwargs["attachments"] = _pending_atts
        asst_msg = Message(**_asst_kwargs)
        db.add(asst_msg)
        msg_count += 1

        # Update conversation counters
        if session:
            session.message_count = (session.message_count or 0) + msg_count
            session.total_tokens = (session.total_tokens or 0) + tokens_input + tokens_output
            session.updated_at = datetime.utcnow()

        # Update DayChat counters (if linked)
        if _day_chat_id:
            try:
                from app.db.models.day_chat import DayChat
                dc = (await db.execute(select(DayChat).where(DayChat.id == _day_chat_id))).scalar_one_or_none()
                if dc:
                    dc.message_count = (dc.message_count or 0) + msg_count
                    dc.total_tokens = (dc.total_tokens or 0) + tokens_input + tokens_output
                    dc.last_message_at = datetime.utcnow()
            except Exception:
                pass  # Non-fatal — DayChat stats are advisory

        await db.flush()
    
    # ------------------------------------------------------------------
    # Memory extraction
    # ------------------------------------------------------------------
    async def _extract_memories(
        self,
        db: AsyncSession,
        user_id: str,
        user_message: str,
        assistant_response: str,
    ) -> int:
        """Extract and store memories from the conversation. Returns count."""
        try:
            from app.services.memory_extractor import get_memory_extractor
            from app.services.memory_dedup_service import MemoryDedupService
            from app.schemas import MemoryCreate, BrainType, MemoryType, MemoryLevel
            
            # Use user's own API key if available
            user_api_key = None
            try:
                from app.db import AgentConfig
                from sqlalchemy import select as _sel
                async with db.begin_nested():
                    result = await db.execute(
                        _sel(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
                    )
                    user_api_key = result.scalar_one_or_none()
            except Exception:
                pass

            extractor = get_memory_extractor()
            extracted = await extractor.extract_memories_with_llm(
                user_message=user_message,
                assistant_response=assistant_response,
                brain_type="user",
                max_memories=15,
                api_key=user_api_key,
            )
            
            dedup = MemoryDedupService(db, api_key=user_api_key)
            count = 0
            for mem in extracted:
                memory_data = MemoryCreate(
                    content=mem.content,
                    summary=mem.summary,
                    brain_type=BrainType.USER,
                    category=mem.category.value if hasattr(mem.category, 'value') else mem.category,
                    memory_type=mem.memory_type,
                    importance=mem.importance,
                    confidence=mem.confidence,
                    memory_level=MemoryLevel.EPISODIC,
                    emotional_salience=0.5,
                    tags=mem.tags,
                    metadata=mem.metadata,
                    source_type="conversation",
                )
                stored, action = await dedup.smart_create_memory(
                    new_memory=memory_data,
                    user_id=user_id,
                )
                logger.info(f"Memory {action}: {stored.content[:50]}...")
                count += 1
                
                # Phase 4: Upsert entities with schema-enforced data + create EntityLinks
                if mem.entities:
                    from app.services.memory_service import MemoryService as _MemSvc
                    from app.db.models import EntityLink as _EL
                    _ms = _MemSvc(db)
                    for ent in mem.entities:
                        ent_name = ent.get("name", "").strip()
                        if not ent_name or len(ent_name) < 2:
                            continue
                        entity_obj = await _ms._upsert_entity(
                            user_id=user_id,
                            name=ent_name,
                            entity_type=ent.get("type", "unknown"),
                            schema_type=ent.get("schema_type"),
                            attributes=ent.get("data"),
                        )
                        # Create EntityLink connecting this memory to the entity
                        if entity_obj and stored:
                            import uuid as _uuid
                            from sqlalchemy import select as _sel, and_ as _and
                            existing_link = await db.execute(
                                _sel(_EL).where(_and(
                                    _EL.memory_id == stored.id,
                                    _EL.entity_id == entity_obj.id,
                                ))
                            )
                            if not existing_link.scalar_one_or_none():
                                db.add(_EL(
                                    id=str(_uuid.uuid4()),
                                    memory_id=stored.id,
                                    entity_id=entity_obj.id,
                                    role=ent.get("role", "mentioned"),
                                ))
                    await db.flush()
            
            # P3: Extract entity relationships and store them
            try:
                relationships = await extractor.extract_relationships_with_llm(
                    user_message=user_message,
                    assistant_response=assistant_response,
                )
                if relationships:
                    from app.services.memory_service import MemoryService
                    mem_service = MemoryService(db)
                    for rel in relationships:
                        await mem_service.store_entity_relationship(
                            user_id=user_id,
                            source_name=rel["source"],
                            source_type=rel["source_type"],
                            target_name=rel["target"],
                            target_type=rel["target_type"],
                            relationship=rel["relationship"],
                            confidence=rel["confidence"],
                            properties=rel.get("properties"),
                        )
                    logger.info(f"Extracted {len(relationships)} entity relationships")
            except Exception as e:
                logger.warning(f"Entity relationship extraction failed (non-fatal): {e}")

            # Invalidate portrait cache if memories were created
            if count >= 3:
                try:
                    from app.services.user_portrait_service import UserPortraitService
                    UserPortraitService(db).invalidate_cache(user_id)
                    logger.info(f"Portrait cache invalidated after {count} new memories")
                except Exception as e:
                    logger.debug(f"Portrait cache invalidation failed (non-fatal): {e}")

            return count
        except Exception as e:
            logger.warning(f"Agent memory extraction failed: {e}")
            return 0
    
    @staticmethod
    def _format_memory_age(created_at_str) -> str:
        """Format memory age as human-readable string."""
        if not created_at_str:
            return ""
        try:
            if isinstance(created_at_str, str):
                dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            else:
                dt = created_at_str
            age = datetime.utcnow() - dt.replace(tzinfo=None)
            days = age.days
            if days == 0:
                return "today"
            elif days == 1:
                return "yesterday"
            elif days < 7:
                return f"{days}d ago"
            elif days < 30:
                return f"{days // 7}w ago"
            elif days < 365:
                return f"{days // 30}mo ago"
            else:
                return f"{days // 365}y ago"
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Media handling (OpenAI vision format)
    # ------------------------------------------------------------------
    @staticmethod
    def _has_builder_context(messages: List[Dict[str, Any]]) -> bool:
        """Check if conversation history contains app_builder interactions.

        Detects both tool_use blocks (app_builder__*) and the [[button]] syntax
        that the builder's direction cards / question cards use.
        """
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                if "[[" in content and "]]" in content:
                    return True
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use" and block.get("name", "").startswith("app_builder__"):
                        return True
                    if block.get("type") == "text":
                        t = block.get("text", "")
                        if "[[" in t and "]]" in t:
                            return True
        return False

    def _build_media_content(
        self,
        text: str,
        media_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Build OpenAI content blocks with images (image_url) and document text."""
        import base64
        import mimetypes

        blocks: List[Dict[str, Any]] = []
        doc_texts: List[str] = []

        # Extensions that should be extracted as text via document parsers
        _DOC_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.zip', '.txt', '.md', '.json', '.yaml', '.yml', '.csv'}

        for path in media_paths:
            mime, _ = mimetypes.guess_type(path)
            ext = os.path.splitext(path.lower())[1]

            if mime and mime.startswith("image/"):
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                    data = base64.standard_b64encode(raw).decode("ascii")
                    logger.info(f"[AGENT] Image loaded: {path} ({len(raw)} bytes, {mime}, base64={len(data)} chars)")
                    blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{data}",
                            "detail": "auto",
                        },
                    })
                except Exception as e:
                    logger.warning(f"Failed to read image {path}: {e}")

            elif ext in _DOC_EXTENSIONS:
                fname = os.path.basename(path)
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                    extracted = self._extract_document_text(raw, fname, ext)
                    if extracted:
                        doc_texts.append(f"[Attached document: {fname}]\n{extracted}")
                        logger.info(f"[AGENT] Document extracted: {path} ({len(raw)} bytes, {len(extracted)} chars text)")
                    else:
                        doc_texts.append(f"[Attached document: {fname} — could not extract text (unsupported or empty)]")
                        logger.warning(f"[AGENT] Document extraction returned empty for {path}")
                except Exception as e:
                    doc_texts.append(f"[Attached document: {fname} — extraction failed: {e}]")
                    logger.warning(f"Failed to extract document {path}: {e}")
            else:
                # Unknown extension — still tell the agent a file was attached
                fname = os.path.basename(path)
                doc_texts.append(f"[Attached file: {fname} — unsupported format, contents not available]")
                logger.warning(f"[AGENT] Unsupported media: {path} (mime={mime}, ext={ext})")

        # Combine user text with any extracted document content
        combined_text = text or ""
        if doc_texts:
            combined_text = combined_text + "\n\n" + "\n\n".join(doc_texts) if combined_text else "\n\n".join(doc_texts)

        if combined_text:
            blocks.append({"type": "text", "text": combined_text})

        return blocks if blocks else [{"type": "text", "text": text or ""}]

    def _extract_document_text(self, content: bytes, filename: str, ext: str) -> Optional[str]:
        """Synchronously extract text from a document file for inline chat context."""
        import io as _io

        try:
            if ext == '.pdf':
                from pypdf import PdfReader
                reader = PdfReader(_io.BytesIO(content))
                return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

            elif ext == '.docx':
                from docx import Document as DocxDocument
                doc = DocxDocument(_io.BytesIO(content))
                return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

            elif ext == '.pptx':
                from pptx import Presentation as PptxPresentation
                prs = PptxPresentation(_io.BytesIO(content))
                parts = []
                for i, slide in enumerate(prs.slides, 1):
                    slide_parts = [f"--- Slide {i} ---"]
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            t = shape.text_frame.text.strip()
                            if t:
                                slide_parts.append(t)
                        if shape.has_table:
                            for row in shape.table.rows:
                                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                                if row_text.strip(" |"):
                                    slide_parts.append(row_text)
                    if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                        notes = slide.notes_slide.notes_text_frame.text.strip()
                        if notes:
                            slide_parts.append(f"[Speaker Notes] {notes}")
                    parts.append("\n".join(slide_parts))
                return "\n\n".join(parts)

            elif ext == '.zip':
                import zipfile
                if not zipfile.is_zipfile(_io.BytesIO(content)):
                    return "[Invalid ZIP file]"
                zf = zipfile.ZipFile(_io.BytesIO(content))
                parts = []
                for i, info in enumerate(zf.infolist()):
                    if info.is_dir() or i >= 50:
                        continue
                    inner_ext = os.path.splitext(info.filename.lower())[1]
                    if inner_ext in {'.txt', '.md', '.json', '.csv', '.yaml', '.yml', '.py', '.js', '.ts'}:
                        try:
                            raw = zf.read(info)
                            parts.append(f"=== {info.filename} ===\n{raw.decode('utf-8', errors='replace')}")
                        except Exception:
                            pass
                    elif inner_ext in {'.pdf', '.docx', '.pptx'}:
                        try:
                            raw = zf.read(info)
                            inner_text = self._extract_document_text(raw, info.filename, inner_ext)
                            if inner_text:
                                parts.append(f"=== {info.filename} ===\n{inner_text}")
                        except Exception:
                            pass
                zf.close()
                return "\n\n".join(parts) if parts else "[Empty or no supported files in ZIP]"

            else:
                # Plain text types (.txt, .md, .json, .csv, .yaml, .yml)
                return content.decode("utf-8", errors="replace").strip()

        except Exception as e:
            logger.warning(f"[AGENT] Document extraction failed for {filename}: {e}")
            return None

    # ------------------------------------------------------------------
    # Error logging
    # ------------------------------------------------------------------
    async def _log_error(
        self,
        user_id: str,
        session_id: Optional[str],
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Log an agent error to the database for monitoring."""
        import traceback
        try:
            from app.db.database import async_session_maker
            from app.db.models import AgentError

            async with async_session_maker() as db:
                err = AgentError(
                    user_id=user_id,
                    session_id=session_id,
                    error_type=error_type,
                    error_message=error_message[:2000],
                    error_traceback=traceback.format_exc()[:5000],
                    context_json=json.dumps(context) if context else None,
                )
                db.add(err)
                await db.commit()
        except Exception as e:
            logger.warning(f"Failed to log agent error to DB: {e}")
