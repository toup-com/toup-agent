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
        # Combine core tools + skill tools
        self.tool_defs = get_agent_tools() + get_extended_tools()
        if self.skill_loader:
            self.tool_defs = self.tool_defs + self.skill_loader.get_all_tool_definitions()
        self.max_iterations = settings.agent_max_tool_iterations
        self._session_model_override: Optional[str] = None  # Per-session model
        self._current_lane: str = 'main'  # Active execution lane
        self._idempotency_key: Optional[str] = None  # Current run idempotency key
        # Phase 5: Track retrieved memories for feedback loop
        self._last_retrieved_memories: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None,
        telegram_chat_id: Optional[int] = None,
        on_text_chunk: Optional[OnTextChunk] = None,
        on_tool_start: Optional[OnToolStart] = None,
        on_tool_end: Optional[OnToolEnd] = None,
        on_tool_progress: Optional[OnToolProgress] = None,
        media_paths: Optional[List[str]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        model_override: Optional[str] = None,
        thinking_budget: int = 0,
        idempotency_key: Optional[str] = None,
    ) -> AgentResponse:
        """
        Run the full agent loop for a single user message.
        """
        start = time.time()
        logger.info(f"[AGENT] === New agent run for user_id={user_id} ===")

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
        async with async_session_maker() as db:
            session, is_new = await self._get_or_create_session(db, user_id, session_id, telegram_chat_id)
            session_id = session.id

            system_prompt = await self._build_system_prompt(db, user_id, user_message)
            logger.info(f"[AGENT] System prompt length: {len(system_prompt)} chars (~{estimate_tokens(system_prompt)} tokens)")

            history = await self._load_history(db, session_id)
            await db.commit()
        # DB session closed — no connection held during LLM calls

        # Prepare messages
        messages = list(history)
        if media_paths:
            content_blocks = self._build_media_content(user_message, media_paths)
            messages.append({"role": "user", "content": content_blocks})
        else:
            messages.append({"role": "user", "content": user_message})

        # Context window management
        if needs_compaction(system_prompt, messages, settings.agent_model):
            logger.info(f"[AGENT] Context compaction triggered ({len(messages)} messages)")
            messages = await compact_messages(messages, settings.agent_model)
            logger.info(f"[AGENT] After compaction: {len(messages)} messages, ~{estimate_messages_tokens(messages)} tokens")

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

                    async for event in active_llm.create_message_stream(
                        messages=messages,
                        system=system_prompt,
                        tools=self.tool_defs,
                        model=active_model,
                        thinking_budget=thinking_budget if _is_claude_model(active_model) else 0,
                    ):
                        if cancel_check and cancel_check():
                            logger.info("[AGENT] Cancelled during streaming")
                            raise asyncio.CancelledError("Generation cancelled by user")

                        if event.type == "text":
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
                                    tools=self.tool_defs,
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

                try:
                    result = await self.tools.execute(tc["name"], tc["input"])
                except Exception as e:
                    logger.exception(f"[AGENT] Tool {tc['name']} crashed")
                    result = f"ERROR: Tool crashed: {type(e).__name__}: {e}"

                logger.info(f"[AGENT] Tool result: {result[:200]}")
                await _hb.emit(HookEvent.AFTER_TOOL_CALL, {"tool": tc["name"], "result_len": len(result)})
                if on_tool_end:
                    summary = result[:200] + "..." if len(result) > 200 else result
                    await on_tool_end(tc["name"], summary)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            # Max iterations reached
            if not final_text:
                final_text = text_buf or "I've reached the maximum number of tool iterations. Here's what I have so far."

        # ── Phase 3: Save to DB (short-lived session) ────────────
        memories_count = 0
        async with async_session_maker() as db:
            await self._save_messages(
                db=db,
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_response=final_text,
                tokens_input=total_input,
                tokens_output=total_output,
                model=model_used,
                processing_time_ms=int((time.time() - start) * 1000),
            )

            if settings.auto_extract_memories and final_text:
                memories_count = await self._extract_memories(
                    db=db,
                    user_id=user_id,
                    user_message=user_message,
                    assistant_response=final_text,
                )

            try:
                from app.services.retrieval_feedback import get_retrieval_feedback
                feedback_svc = get_retrieval_feedback(db)
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

            await db.commit()

        elapsed = int((time.time() - start) * 1000)
        logger.info(f"[AGENT] Response: {final_text[:100]}...")
        logger.info(f"[AGENT] Tokens: in={total_input} out={total_output} | Tools: {len(all_tool_calls)} | Memories: {memories_count} | Time: {elapsed}ms")

        # Hook: agent run complete
        await _hb.emit(HookEvent.AGENT_END, {
            "user_id": user_id, "session_id": session_id,
            "tool_count": len(all_tool_calls),
            "tokens": total_input + total_output,
            "elapsed_ms": elapsed,
        })

        return AgentResponse(
            text=final_text,
            session_id=session_id,
            tool_calls=all_tool_calls,
            tokens_input=total_input,
            tokens_output=total_output,
            tokens_total=total_input + total_output,
            model=model_used,
            processing_time_ms=elapsed,
            memories_extracted=memories_count,
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
                return session, False
        
        # Create new session
        session = Conversation(
            user_id=user_id,
            channel="telegram" if telegram_chat_id else "agent",
            is_active=True,
            metadata_json=json.dumps({"telegram_chat_id": telegram_chat_id}) if telegram_chat_id else None,
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
    ) -> str:
        """Build a rich system prompt from identities + memories + runtime context."""
        from sqlalchemy import select, and_
        from app.db.models import Identity, IdentityType
        
        sections: List[str] = []
        
        # 1. Load identities
        result = await db.execute(
            select(Identity).where(
                and_(
                    Identity.user_id == user_id,
                    Identity.is_active == True,
                )
            ).order_by(Identity.priority.desc())
        )
        identities = result.scalars().all()
        logger.info(f"[AGENT] Building system prompt for user: {user_id}")
        logger.info(f"[AGENT] Found {len(identities)} identities")
        
        for identity in identities:
            if identity.identity_type == IdentityType.SOUL.value:
                sections.append(f"# Core Identity\n{identity.content}")
            elif identity.identity_type == IdentityType.AGENT_INSTRUCTIONS.value:
                sections.append(f"# Behavioral Guidelines\n{identity.content}")
            elif identity.identity_type == IdentityType.USER_PROFILE.value:
                sections.append(f"# About the User\n{identity.content}")
            elif identity.identity_type == IdentityType.TOOLS.value:
                sections.append(f"# Tool Guidelines\n{identity.content}")
        
        # 2. Load agent brain memories (permanent identity/skills/patterns)
        try:
            from app.services.memory_service import MemoryService
            mem_svc = MemoryService(db)
            
            # Agent brain: load ALL active agent memories (soul, skills, tools, patterns, procedures, decisions)
            agent_memories = await mem_svc.get_memories_by_brain_type(
                user_id=user_id,
                brain_type="agent",
                limit=50,
            )
            if agent_memories:
                agent_lines = ["# Agent Brain (Permanent Knowledge)"]
                for m in agent_memories:
                    cat = m.get("category", "")
                    content = m.get("content", "")
                    agent_lines.append(f"- [{cat}] {content}")
                sections.append("\n".join(agent_lines))
                logger.info(f"[AGENT] Loaded {len(agent_memories)} agent brain memories")
        except Exception as e:
            logger.warning(f"Agent brain load failed: {e}")
        
        # 3. Retrieve relevant user memories (hybrid search: vector + keyword + graph)
        try:
            from app.services.memory_service import MemoryService
            
            mem_svc = MemoryService(db)
            logger.info(f'[AGENT] Hybrid searching user memories for: "{user_message[:80]}"')
            
            memories = await mem_svc.hybrid_search(
                user_id=user_id,
                query=user_message,
                limit=15,
                min_similarity=0.1,
                strategies=["vector", "keyword", "graph"],
            )
            # Filter to user brain only (agent brain already loaded above)
            user_memories = [m for m in memories if m.get("brain_type") == "user"]
            # Phase 5: Store retrieved memories for feedback loop
            self._last_retrieved_memories = user_memories
            logger.info(f"[AGENT] Found {len(user_memories)} relevant user memories (hybrid)")
            
            if user_memories:
                mem_lines = ["# User Brain (Relevant Memories)"]
                for i, m in enumerate(user_memories, 1):
                    cat = m.get("category", "")
                    content = m.get("content", "")
                    score = m.get("similarity_score", 0)
                    logger.info(f"[AGENT]   Memory {i}: [{cat}] ({score:.2f}) {content[:80]}")
                    mem_lines.append(f"{i}. [{cat}] (relevance: {score:.2f}) {content}")
                sections.append("\n".join(mem_lines))
        except Exception as e:
            logger.warning(f"Memory retrieval failed in agent prompt: {e}")
        
        # 3. Default identity if none exists
        if not identities:
            sections.insert(0, (
                "# Core Identity\n"
                "You are HexBrain, an intelligent AI assistant with persistent memory. "
                "You can execute shell commands, read/write files, search the web, "
                "and remember information about the user across conversations. "
                "Be helpful, proactive, and concise. Use tools when they help answer the question."
            ))
        
        # 4. Runtime context
        now = datetime.utcnow()
        sections.append(
            f"# Runtime Context\n"
            f"- Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"- Channel: Telegram\n"
            f"- Workspace directory: {settings.agent_workspace_dir}\n"
            f"- Max tool iterations: {self.max_iterations}"
        )

        # 4b. Lane context
        if hasattr(self, "_current_lane") and self._current_lane != "main":
            sections.append(
                f"# Execution Lane\n"
                f"You are running in the **{self._current_lane}** lane."
            )

        # 4c. Activation prompt (set via /activation command)
        if hasattr(self, '_activation_prompt') and self._activation_prompt:
            sections.append(f"# Activation Prompt\n{self._activation_prompt}")

        # 4d. Verbose mode flag
        if hasattr(self, '_verbose_mode') and self._verbose_mode:
            sections.append(
                "# Verbose Mode\n"
                "VERBOSE MODE IS ON. When calling tools, explain what you are doing "
                "and why before each tool call. After each tool call, summarize the "
                "full result in detail."
            )

        # 5. Telegram formatting rules
        sections.append(
            "# Formatting Rules (IMPORTANT)\n"
            "You are communicating via Telegram. Follow these rules strictly:\n"
            "- Do NOT use LaTeX math formatting. No $...$ or $$...$$ or \\(...\\) or \\[...\\] wrappers.\n"
            "- Use plain Unicode symbols for math: × (multiply), ÷ (divide), √ (square root), "
            "→ (arrow), ⇒ (implies), ≤ ≥ ≠ ≈ ∞ π.\n"
            "- Write fractions as a/b, not \\frac{a}{b}.\n"
            "- Telegram supports basic Markdown: **bold**, *italic*, `code`, ```code blocks```.\n"
            "- Do NOT use tables or complex formatting.\n"
            "- Keep responses concise and readable on mobile."
        )

        # 6. Reactions
        sections.append(
            "# Reactions\n"
            "You can react to the user's message with an emoji by including [[reaction:EMOJI]] "
            "anywhere in your response. It will be stripped before sending. "
            "React sparingly — at most 1 reaction per 5-10 messages. "
            "React when: something is genuinely funny (😂), you appreciate something (❤️), "
            "simple acknowledgment (👍), interesting/thoughtful (🤔), impressive (🔥), "
            "celebrating (🎉). Don't react to routine messages."
        )

        # 7. Inline buttons
        sections.append(
            "# Inline Buttons\n"
            "You can add inline buttons to your message by including [[button:LABEL|CALLBACK_DATA]] "
            "markers. They will be stripped from text and rendered as clickable Telegram buttons. "
            "Use buttons when offering clear choices, confirmations, or actions. "
            "Example: [[button:Yes|confirm_yes]] [[button:No|confirm_no]]\n"
            "Keep callback_data short (max 64 chars). Don't overuse buttons — only when genuinely helpful."
        )

        # 8. Skill prompt sections
        if self.skill_loader:
            for skill_section in self.skill_loader.get_all_system_prompt_sections():
                sections.append(skill_section)
        
        return "\n\n".join(sections)
    
    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------
    async def _load_history(
        self,
        db: AsyncSession,
        session_id: str,
        max_messages: int = 20,
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
    ):
        from sqlalchemy import select
        from app.db.models import Message, Conversation

        # Safety: strip null bytes — PostgreSQL text columns reject \x00
        user_message = user_message.replace("\x00", "")
        assistant_response = assistant_response.replace("\x00", "")

        user_msg = Message(
            conversation_id=session_id,
            role="user",
            content=user_message,
        )
        db.add(user_msg)

        asst_msg = Message(
            conversation_id=session_id,
            role="assistant",
            content=assistant_response,
            tokens_prompt=tokens_input,
            tokens_completion=tokens_output,
            model_used=model,
            processing_time_ms=processing_time_ms,
        )
        db.add(asst_msg)

        # Update conversation counters (load by ID in this short-lived session)
        result = await db.execute(
            select(Conversation).where(Conversation.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            session.message_count = (session.message_count or 0) + 2
            session.total_tokens = (session.total_tokens or 0) + tokens_input + tokens_output
            session.updated_at = datetime.utcnow()

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
            
            extractor = get_memory_extractor()
            extracted = await extractor.extract_memories_with_llm(
                user_message=user_message,
                assistant_response=assistant_response,
                brain_type="user",
                max_memories=15,
            )
            
            dedup = MemoryDedupService(db)
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
                
                # Phase 4: Upsert entities with schema-enforced data
                if mem.entities:
                    from app.services.memory_service import MemoryService as _MemSvc
                    _ms = _MemSvc(db)
                    for ent in mem.entities:
                        ent_name = ent.get("name", "").strip()
                        if not ent_name or len(ent_name) < 2:
                            continue
                        await _ms._upsert_entity(
                            user_id=user_id,
                            name=ent_name,
                            entity_type=ent.get("type", "unknown"),
                            schema_type=ent.get("schema_type"),
                            attributes=ent.get("data"),
                        )
            
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
            
            return count
        except Exception as e:
            logger.warning(f"Agent memory extraction failed: {e}")
            return 0
    
    # ------------------------------------------------------------------
    # Media handling (OpenAI vision format)
    # ------------------------------------------------------------------
    def _build_media_content(
        self,
        text: str,
        media_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Build OpenAI content blocks with images (image_url) and text."""
        import base64
        import mimetypes

        blocks: List[Dict[str, Any]] = []

        for path in media_paths:
            mime, _ = mimetypes.guess_type(path)
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
            else:
                logger.warning(f"[AGENT] Skipping non-image media: {path} (mime={mime})")

        if text:
            blocks.append({"type": "text", "text": text})
        
        return blocks if blocks else [{"type": "text", "text": text or ""}]

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
