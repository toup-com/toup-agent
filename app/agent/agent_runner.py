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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context_manager import (
    needs_compaction,
    compact_messages,
    estimate_tokens,
    estimate_messages_tokens,
)
from app.agent.tool_definitions import get_agent_tools, get_extended_tools
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


OnTextChunk = Callable[[str], Coroutine[Any, Any, None]]
OnToolStart = Callable[[str], Coroutine[Any, Any, None]]
OnToolEnd = Callable[[str, str], Coroutine[Any, Any, None]]
OnToolProgress = Callable[[str, str], Coroutine[Any, Any, None]]


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
        # Core tools (static) — skill tools are added dynamically via property
        self._core_tool_defs = get_agent_tools() + get_extended_tools()
        self.max_iterations = settings.agent_max_tool_iterations
        self._session_model_override: Optional[str] = None  # Per-session model
        self._current_lane: str = 'main'  # Active execution lane
        self._idempotency_key: Optional[str] = None  # Current run idempotency key
        self._disabled_tool_names: set = set()  # Per-session disabled tools
        # Phase 5: Track retrieved memories for feedback loop
        self._last_retrieved_memories: List[Dict[str, Any]] = []

    @property
    def tool_defs(self) -> list:
        """Dynamically combine core tools + skill tools (picks up new app skills)."""
        defs = list(self._core_tool_defs)
        if self.skill_loader:
            defs = defs + self.skill_loader.get_all_tool_definitions()
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
        self.tools._on_tool_progress = on_tool_progress

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
                    _day_context = await load_day_context(db, _day_chat_id, model=settings.agent_model, model_context_tokens=_ctx_window)
                    history = _day_context["messages"]
                    logger.info(f"[PERF] load_day_context: {(time.perf_counter() - t_db) * 1000:.0f}ms — {len(history)} messages (day-chat)")
                except Exception as _dce:
                    logger.warning("[AGENT] Day context load failed, falling back to session history: %s", _dce)
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
            system_prompt = await self._build_system_prompt(db, user_id, user_message, channel=channel, intent=query_intent)

            # Inject <today_so_far> block when using day-chat context with a summary
            if _use_day_ctx and _day_context and _day_context.get("summary"):
                from app.agent.day_context_loader import build_today_so_far_block
                system_prompt += build_today_so_far_block(_day_context["summary"])

            logger.info(f"[PERF] build_system_prompt: {(time.perf_counter() - t_prompt) * 1000:.0f}ms — {len(system_prompt)} chars (~{estimate_tokens(system_prompt)} tokens)")
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
        final_text = ""
        model_used = ""

        # Determine which model to use
        routing_decision: Optional[RoutingDecision] = None
        if self._session_model_override and model_override is None:
            model_override = self._session_model_override

        if model_override == "auto" or model_override is None:
            routing_decision = classify_request(
                user_message=user_message,
                conversation_history=messages[:-1],
                has_media=bool(media_paths),
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
                    if attempt < MAX_RETRIES:
                        logger.warning(f"[AGENT] LLM call failed (attempt {attempt + 1}), retrying: {e}")
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        fallback = settings.agent_fallback_model
                        if active_model != fallback:
                            fallback_llm = self.anthropic if _is_claude_model(fallback) else self.llm
                            logger.warning(f"[AGENT] Primary model {active_model} failed, trying fallback {fallback}")
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
                try:
                    result = await self.tools.execute(tc["name"], tc["input"])
                except Exception as e:
                    logger.exception(f"[AGENT] Tool {tc['name']} crashed")
                    result = f"ERROR: Tool crashed: {type(e).__name__}: {e}"

                logger.info(f"[PERF] tool_exec({tc['name']}): {(time.perf_counter() - _t_tool) * 1000:.0f}ms — {len(result)} chars")
                logger.info(f"[AGENT] Tool result: {result[:200]}")
                await _hb.emit(HookEvent.AFTER_TOOL_CALL, {"tool": tc["name"], "result_len": len(result)})
                if on_tool_end:
                    summary = result[:200] + "..." if len(result) > 200 else result
                    await on_tool_end(tc["name"], summary, tc.get("input"))

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
        _channel = "telegram" if telegram_chat_id else (channel or "agent")
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

        # Default identity if no Soul exists
        if not has_soul_identity:
            identity_parts.insert(0, (
                "# Core Identity\n"
                "Your name is Toup. You are an intelligent AI assistant with persistent memory.\n"
                "Communicate in a friendly, helpful manner. Be concise but thorough. "
                "Ask clarifying questions when needed."
            ))
            logger.warning(f"No soul config found for user {user_id}, using default")

        section_parts["identity"] = "\n\n".join(identity_parts)

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

        # ── 3b. Active tasks — always injected, regardless of intent ──
        try:
            from app.services.active_task_service import get_active_tasks, build_active_tasks_block
            active_tasks = await get_active_tasks(db, user_id)
            if active_tasks:
                section_parts["active_tasks"] = build_active_tasks_block(active_tasks)
                logger.info(f"[AGENT] Injected {len(active_tasks)} active task(s)")
        except Exception as _at_err:
            logger.debug(f"[AGENT] Active tasks injection skipped: {_at_err}")

        # ── 4. Skills (only if intent requires them) ─────────────
        if self.skill_loader and intent.include_skill_prompts:
            skill_parts = self.skill_loader.get_all_system_prompt_sections()
            if skill_parts:
                section_parts["skills"] = "\n\n".join(skill_parts)
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
            "- **Database access**: You have direct access to the `toup_brain` PostgreSQL database via "
            "`memory_store`, `memory_search`, and other tools. You can also query it via `exec` with psql.\n"
            "- **File system**: You can read, write, and edit files anywhere on this machine using "
            "`read_file`, `write_file`, `edit_file`, `ls`, `find`, `grep`.\n"
            "- **Web access**: You can search the web (`web_search`), fetch pages (`web_fetch`), "
            "and automate browsers (`browser`).\n"
            "- **Admin capabilities**: You ARE the agent running on this system. You can install packages, "
            "manage services, modify configurations, and perform system administration tasks.\n"
            "- **Memory management**: You can store, search, and manage memories in the brain database. "
            "You can also delete or modify memories by using `exec` with psql commands.\n\n"
            "When the user asks you to do something on their system, USE your tools — don't say you can't."
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
        now = datetime.utcnow()
        _channel_label = channel or "telegram"
        runtime_lines = [
            f"# Runtime Context",
            f"- Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"- Channel: {_channel_label}",
            f"- Workspace directory: {settings.agent_workspace_dir}",
            f"- Max tool iterations: {self.max_iterations}",
            f"- You have FULL terminal/shell access via the `exec` tool. You can run any command, install packages, write scripts, manage files, use git, curl, python, node, etc.",
            f"- You can read and write files using `read_file` and `write_file` tools.",
            f"- You can search the web using the `web_search` tool.",
            f"- When the user asks you to do a multi-step task, use the `create_job` tool to create a trackable job. Update it with `update_job` as you complete each step.",
        ]
        if hasattr(self, "_current_lane") and self._current_lane != "main":
            runtime_lines.append(f"- Execution lane: {self._current_lane}")
        section_parts["runtime"] = "\n".join(runtime_lines)

        # ── 6b. Vibe Coding mode ─────────────────────────────────
        if _channel_label == "vibecoding":
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
        if _channel_label in ("app", "web", "vibecoding"):
            section_parts["formatting"] = (
                "# Formatting Rules\n"
                "You are chatting with the user inside their " + ("app" if _channel_label == "app" else "web browser") + ". Follow these rules:\n"
                "- Use simple Markdown: **bold**, *italic*, `code`.\n"
                "- Do NOT use LaTeX math formatting.\n"
                "- Use plain Unicode symbols for math: × ÷ √ → ⇒ ≤ ≥ ≠ ≈ ∞ π.\n"
                "- Keep responses concise and conversational.\n"
                "- Do NOT expose internal implementation details (databases, bridges, connections, file paths, error traces).\n"
                "- When the user greets you, greet them back warmly and offer to help.\n"
                "- You have editing capabilities: you can modify the app's files, database, and navigation using your tools.\n"
                "- When the user asks you to change something in the app (theme, colors, layout, text, features, etc.), "
                "DO NOT just describe what you would do — actually DO it by calling your write_file/edit_file tools to modify the source code. "
                "After editing, call the restart tool to apply changes. Read the relevant file first, make the edit, restart, and confirm what you changed.\n"
                "- NEVER give the user localhost URLs. App previews are at: https://toup.ai/workspace/apps/{app-slug}\n"
                "- After fixing or restarting an app, offer a [[open_app:{app-slug}]] chip so the user can see the result. Use the app's slug (e.g. [[open_app:Confidence-Booster]]).\n\n"
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
            "identity",       # WHO the agent is
            "user_brain",     # WHO the user is
            "agent_brain",    # Agent brain (disabled by default)
            "work_brain",     # Work brain (disabled by default)
            "skills",         # WHAT the agent can do
            "environment",    # WHAT the agent has access to
            "media",          # Media playback instructions (web/app)
            "runtime",        # WHEN/WHERE
            "vibecoding",     # Vibe coding mode override (when active)
            "formatting",     # HOW to respond
            "onboarding",     # Temporary onboarding instructions
            "activation",     # Optional activation prompt
            "verbose",        # Optional verbose mode
        ]

        sections = [section_parts[k] for k in SECTION_ORDER if k in section_parts]

        # Log section sizes for debugging
        for name, text in section_parts.items():
            tokens_est = len(text) // 4
            logger.debug(f"Prompt [{name}]: ~{tokens_est} tokens")
        total = sum(len(v) for v in section_parts.values()) // 4
        logger.info(f"[AGENT] System prompt: {len(section_parts)} sections, ~{total} tokens est.")

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
    ):
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

        if save_user_message:
            user_msg = Message(
                conversation_id=session_id,
                day_chat_id=_day_chat_id,
                role="user",
                content=user_message,
            )
            db.add(user_msg)
            msg_count += 1

        # Capture media metadata from tool calls (play_media, play_netflix)
        media_meta = getattr(self.tools, '_last_media', None)
        if media_meta:
            self.tools._last_media = None  # Clear after capture

        asst_msg = Message(
            conversation_id=session_id,
            day_chat_id=_day_chat_id,
            role="assistant",
            content=assistant_response,
            tokens_prompt=tokens_input,
            tokens_completion=tokens_output,
            model_used=model,
            processing_time_ms=processing_time_ms,
            metadata_json=json.dumps({"media": media_meta}) if media_meta else None,
        )
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
