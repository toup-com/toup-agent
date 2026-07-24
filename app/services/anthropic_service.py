"""
Anthropic Claude API Service — Wrapper for chat completions with tool use and streaming.

Provides:
- Streaming support (yield chunks as they arrive)
- Tool use support (handle tool_use content blocks)
- Multi-turn tool calling (LLM calls tool → get result → LLM continues)
- Token counting from response usage
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Dict, Any, Optional, Union

import anthropic

from app.config import settings

logger = logging.getLogger(__name__)


def _model_rejects_sampling(model: str) -> bool:
    """Opus 4.7+ returns 400 on temperature/top_p/top_k and on extended-thinking
    budgets. Detect models that require the new calling convention."""
    m = (model or "").lower()
    return "opus-4-7" in m


def _mark_tools_cacheable(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Add ephemeral cache_control to the last tool so the whole tools block
    gets cached by Anthropic. Saves ~2–3k prompt tokens per turn on a 70+ tool
    loadout — big deal for agents that iterate many turns per user message."""
    if not tools:
        return tools
    marked = [dict(t) for t in tools]
    marked[-1] = {**marked[-1], "cache_control": {"type": "ephemeral"}}
    return marked


def _mark_messages_cacheable(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add ephemeral cache_control to the final content block of the last
    message so the entire history prefix becomes cacheable. Subsequent turns
    that share the same prefix (system + tools + earlier turns) get a cache
    hit and pay only ~10% of the cached input tokens.

    Handles both content shapes:
      - string content → wrap into a single text block with cache_control
      - list-of-blocks content → set cache_control on the last block

    Cache control is incompatible with tool_result blocks containing only
    image content for some models, but for our agent loop the last message
    is always either user text or a tool_result with text — both supported.

    PR-1 stable layout: the runner may append a per-turn ``<turn_context>``
    message (volatile clock/memory/day blocks) between history and the
    current user message. Its bytes differ every turn and it is never
    persisted, so a breakpoint at the very end would cover a span that can
    never re-match — every turn would re-WRITE the whole day history
    (1.25x) and never read it back. Place the breakpoint on the last
    message BEFORE the first trailing turn-context/user pair instead, so
    the append-only history span stays cacheable and the volatile tail
    sits after the breakpoint.
    """
    if not messages:
        return messages
    out = [dict(m) for m in messages]
    # Find a trailing <turn_context> message (at -1 or -2 — the runner puts
    # it immediately before the current user message). Mark the message
    # preceding it so the cached span ends at end-of-history.
    mark_idx = len(out) - 1
    for probe in (len(out) - 2, len(out) - 1):
        if probe >= 1:
            c = out[probe].get("content")
            if isinstance(c, str) and c.startswith("<turn_context>"):
                mark_idx = probe - 1
                break
    last = dict(out[mark_idx])
    content = last.get("content")
    if isinstance(content, str):
        if not content:
            return out
        last["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    elif isinstance(content, list) and content:
        new_blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        # Only mark cache_control on dict blocks; skip non-dict (defensive).
        for i in range(len(new_blocks) - 1, -1, -1):
            block = new_blocks[i]
            if isinstance(block, dict):
                new_blocks[i] = {**block, "cache_control": {"type": "ephemeral"}}
                break
        last["content"] = new_blocks
    else:
        return out
    out[mark_idx] = last
    return out


def _convert_messages_for_anthropic(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format messages to Anthropic format.

    The main difference is image handling:
      OpenAI:    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    """
    converted = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            converted.append(msg)
            continue

        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                # Convert OpenAI image_url → Anthropic image
                url = (block.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    # Parse data URI: data:image/png;base64,iVBOR...
                    header, _, b64data = url.partition(",")
                    media_type = header.split(":", 1)[-1].split(";")[0] if ":" in header else "image/png"
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    # External URL — use Anthropic's URL source
                    new_content.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
            else:
                new_content.append(block)

        converted.append({**msg, "content": new_content})
    return converted


@dataclass
class AnthropicResponse:
    """Non-streaming response from Anthropic."""
    content: str                          # Concatenated text content
    tool_calls: List[Dict[str, Any]]      # List of tool_use blocks
    model: str
    tokens_input: int
    tokens_output: int
    tokens_total: int
    stop_reason: str                      # "end_turn" | "tool_use" | "max_tokens"


@dataclass
class StreamEvent:
    """Single event from a streaming response."""
    type: str           # "text", "tool_use_start", "tool_use_input", "tool_use_end", "message_end"
    text: str = ""
    tool_name: str = ""
    tool_id: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


class AnthropicService:
    """
    Anthropic Claude API wrapper.
    
    Supports:
    - Non-streaming completions
    - Streaming completions with tool use
    - Automatic retry on transient errors
    """
    
    def __init__(self):
        from app.services.key_provider import keys
        self._key_version = -1  # Force initial build
        self._keys = keys
        self.client = None
        self.is_oauth = False
        self.default_model = settings.anthropic_model
        self.default_max_tokens = settings.anthropic_max_tokens
        self._ensure_client()

    def _ensure_client(self):
        """Rebuild the Anthropic client if the API key has changed."""
        if self._key_version == self._keys.version and self.client is not None:
            return
        self._key_version = self._keys.version

        from app.services.bundle_client import make_anthropic_client

        api_key = self._keys.anthropic or ""
        # OAuth detection only matters for BYOK; bundle mode never uses OAuth.
        self.is_oauth = bool(
            settings.llm_mode != "bundle"
            and api_key
            and "sk-ant-oat" in api_key
        )
        # bundle_client.make_anthropic_client returns None when there's no
        # way to talk to a provider (no bundle, no BYOK key). Leaving
        # self.client = None and surfacing an actionable error from the
        # entry-point methods is cleaner than the previous sentinel that
        # masked the cause as a generic "Incorrect API key provided: missing"
        # 401 from Anthropic.
        self.client = make_anthropic_client(byok_key=api_key or None)
        if self.client is None:
            logger.warning(
                "Anthropic client could not be built (no key, not in bundle mode)"
            )
            return
        logger.info("[ANTHROPIC] Client rebuilt (mode=%s, oauth=%s, v%d)",
                    settings.llm_mode, self.is_oauth, self._key_version)
    
    def _prepare_system(self, system: str) -> any:
        """
        Prepare system prompt.

        Always wraps the system prompt in a text block with ephemeral
        ``cache_control`` so Anthropic's 5-minute prompt cache covers the
        (large, mostly-stable) system block across consecutive turns. OAuth
        tokens additionally prepend the Claude Code identity (Anthropic's
        credential check) — also cached.

        Returns:
            The raw string only when there's no system prompt to cache; a
            list-of-blocks otherwise.
        """
        if not self.is_oauth:
            if not system:
                return system
            return [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # OAuth tokens MUST include Claude Code identity prefix
        cc_identity = "You are Claude Code, Anthropic's official CLI for Claude."
        parts = [
            {"type": "text", "text": cc_identity, "cache_control": {"type": "ephemeral"}},
        ]
        if system:
            parts.append({"type": "text", "text": system, "cache_control": {"type": "ephemeral"}})
        return parts
    
    # ------------------------------------------------------------------
    # Non-streaming completion
    # ------------------------------------------------------------------
    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        thinking_budget: int = 0,
    ) -> AnthropicResponse:
        """
        Create a non-streaming message.

        Args:
            messages: Conversation history in Anthropic format.
            system: System prompt.
            tools: Tool definitions.
            model: Model override.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            thinking_budget: Extended thinking token budget (0 = disabled).

        Returns:
            AnthropicResponse with text content and any tool calls.
        """
        model = model or self.default_model
        self._ensure_client()
        if self.client is None:
            raise RuntimeError(
                "Anthropic client is not configured. Add ANTHROPIC_API_KEY "
                "(or activate bundle mode with TOUP_TOKEN) and try again."
            )
        max_tokens = max_tokens or self.default_max_tokens
        # Pre-flight: gate on the last known credit state. See
        # services/credit_reporter.py for the rationale.
        from app.services.credit_reporter import raise_if_exhausted
        raise_if_exhausted()

        messages = _convert_messages_for_anthropic(messages)
        # TKT-LAT-001: mark the final message block cacheable so the
        # conversation prefix participates in Anthropic's 5-minute cache.
        messages = _mark_messages_cacheable(messages)

        kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        rejects_sampling = _model_rejects_sampling(model)
        if not rejects_sampling:
            kwargs["temperature"] = temperature
        # Extended thinking: if budget > 0, enable thinking with budget control
        # (Opus 4.7+ removed budget-based thinking — use adaptive thinking instead)
        if thinking_budget > 0 and not rejects_sampling:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Thinking requires temperature=1 per Anthropic docs
            kwargs["temperature"] = 1
        # Prepare system prompt (adds Claude Code identity for OAuth tokens;
        # also wraps every system block in ephemeral cache_control — TKT-LAT-001).
        prepared_system = self._prepare_system(system)
        if prepared_system:
            kwargs["system"] = prepared_system
        if tools:
            kwargs["tools"] = _mark_tools_cacheable(tools)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(**kwargs)

                text_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []

                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                # TKT-LAT-001: surface cache hit/write counts so prod can
                # observe the cache working without scraping the SDK object.
                usage = response.usage
                logger.info(
                    "[PERF] cache_read=%s cache_creation=%s input=%s output=%s model=%s",
                    getattr(usage, "cache_read_input_tokens", 0) or 0,
                    getattr(usage, "cache_creation_input_tokens", 0) or 0,
                    usage.input_tokens,
                    usage.output_tokens,
                    response.model,
                )

                # Credit metering for direct (non-proxy) LLM calls.
                # Fire-and-forget — failures never block the chat response.
                # See services/credit_reporter.py for the full design.
                try:
                    from app.services.credit_reporter import report_llm_usage
                    user_id = getattr(settings, "user_id", "") or ""
                    if user_id:
                        await report_llm_usage(
                            user_id=user_id,
                            model=response.model,
                            provider="anthropic",
                            input_tokens=int(response.usage.input_tokens or 0),
                            output_tokens=int(response.usage.output_tokens or 0),
                            idempotency_key=getattr(response, "id", None),
                            event_id=getattr(response, "id", None),
                        )
                except Exception:
                    logger.exception("[credits] anthropic non-stream report failed")

                return AnthropicResponse(
                    content="\n".join(text_parts),
                    tool_calls=tool_calls,
                    model=response.model,
                    tokens_input=response.usage.input_tokens,
                    tokens_output=response.usage.output_tokens,
                    tokens_total=response.usage.input_tokens + response.usage.output_tokens,
                    stop_reason=response.stop_reason,
                )
            
            except anthropic.APIStatusError as e:
                # Bundle 401 self-heal (see openai_agent_service for rationale):
                # the first message can race /admin/bind so the cached client
                # carries the stale GENERIC toup_token → proxy 401. Reload
                # identity+token, rebuild client, retry once. Bundle-only,
                # first-attempt-only so a real BYOK bad key fails fast.
                if e.status_code == 401 and attempt == 0 and (getattr(settings, "llm_mode", "") == "bundle"):
                    try:
                        from app.services import runtime_identity as _ri
                        _ri.reload()
                        _ri.apply_to_settings(_ri.all_runtime_fields())
                    except Exception as _re:
                        logger.warning(f"[ANTHROPIC] identity reload on 401 failed: {_re}")
                    self._keys.refresh()
                    self._ensure_client()
                    logger.info("[ANTHROPIC] bundle 401 — reloaded identity+token, retrying once")
                    continue
                # Retry on 429 (rate limit) and 529 (overloaded)
                if e.status_code in (429, 529) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # 2s, 4s
                    logger.warning(f"Anthropic {e.status_code} {type(e).__name__}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise
            except anthropic.APIConnectionError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Anthropic connection error, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise
    
    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------
    async def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        thinking_budget: int = 0,
        tool_choice: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,  # noqa: ARG002 — TKT-LAT-018: accepted for cross-provider signature parity, unused on Anthropic (cache_control already wires the prompt cache)
        safety_identifier: Optional[str] = None,  # noqa: ARG002 — PR-1 signature parity; OpenAI-only param
        idempotency_key: Optional[str] = None,  # noqa: ARG002 — PR-1 signature parity; Anthropic metering is unchanged
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream a message. Yields StreamEvent objects for text chunks,
        tool use blocks, and the final message_end event.
        """
        self._ensure_client()
        if self.client is None:
            raise RuntimeError(
                "Anthropic client is not configured. Add ANTHROPIC_API_KEY "
                "(or activate bundle mode with TOUP_TOKEN) and try again."
            )
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens

        # Pre-flight: gate on the last known credit state.
        from app.services.credit_reporter import raise_if_exhausted
        raise_if_exhausted()

        messages = _convert_messages_for_anthropic(messages)
        # TKT-LAT-001: mark the final message block cacheable so the
        # conversation prefix participates in Anthropic's 5-minute cache.
        messages = _mark_messages_cacheable(messages)

        kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        rejects_sampling = _model_rejects_sampling(model)
        if not rejects_sampling:
            kwargs["temperature"] = temperature
        # Extended thinking: if budget > 0, enable thinking with budget control
        # (Opus 4.7+ removed budget-based thinking — use adaptive thinking instead)
        if thinking_budget > 0 and not rejects_sampling:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Thinking requires temperature=1 per Anthropic docs
            kwargs["temperature"] = 1
        # Prepare system prompt (adds Claude Code identity for OAuth tokens;
        # also wraps every system block in ephemeral cache_control — TKT-LAT-001).
        prepared_system = self._prepare_system(system)
        if prepared_system:
            kwargs["system"] = prepared_system
        if tools:
            kwargs["tools"] = _mark_tools_cacheable(tools)
            if tool_choice == "required":
                kwargs["tool_choice"] = {"type": "any"}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                current_tool_name = ""
                current_tool_id = ""
                input_json_buf = ""

                async with self.client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        # --- text delta ---
                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "thinking":
                                yield StreamEvent(type="thinking_start", text="")
                            elif block.type == "tool_use":
                                # Convert CC tool name back to original
                                current_tool_name = block.name
                                current_tool_id = block.id
                                input_json_buf = ""
                                yield StreamEvent(
                                    type="tool_use_start",
                                    tool_name=current_tool_name,
                                    tool_id=current_tool_id,
                                )

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                yield StreamEvent(type="text", text=delta.text)
                            elif delta.type == "thinking_delta":
                                yield StreamEvent(type="thinking", text=delta.thinking)
                            elif delta.type == "input_json_delta":
                                input_json_buf += delta.partial_json

                        elif event.type == "content_block_stop":
                            if current_tool_name:
                                import json as _json
                                try:
                                    tool_input = _json.loads(input_json_buf) if input_json_buf else {}
                                except _json.JSONDecodeError:
                                    tool_input = {"raw": input_json_buf}
                                yield StreamEvent(
                                    type="tool_use_end",
                                    tool_name=current_tool_name,
                                    tool_id=current_tool_id,
                                    tool_input=tool_input,
                                )
                                current_tool_name = ""
                                current_tool_id = ""
                                input_json_buf = ""

                        elif event.type == "message_stop":
                            pass

                    # After stream ends, get final message for usage/stop_reason
                    final = await stream.get_final_message()
                    # TKT-LAT-001: surface cache hit/write counts so prod can
                    # observe the cache working without scraping the SDK object.
                    final_usage = final.usage
                    cache_read = getattr(final_usage, "cache_read_input_tokens", 0) or 0
                    cache_creation = getattr(final_usage, "cache_creation_input_tokens", 0) or 0
                    logger.info(
                        "[PERF] cache_read=%s cache_creation=%s input=%s output=%s model=%s",
                        cache_read,
                        cache_creation,
                        final_usage.input_tokens,
                        final_usage.output_tokens,
                        final.model,
                    )
                    # Credit metering for streaming direct (non-proxy) calls.
                    # Fire-and-forget — never blocks the stream end.
                    try:
                        from app.services.credit_reporter import report_llm_usage
                        user_id = getattr(settings, "user_id", "") or ""
                        if user_id:
                            await report_llm_usage(
                                user_id=user_id,
                                model=final.model,
                                provider="anthropic",
                                input_tokens=int(final_usage.input_tokens or 0),
                                output_tokens=int(final_usage.output_tokens or 0),
                                idempotency_key=getattr(final, "id", None),
                                event_id=getattr(final, "id", None),
                            )
                    except Exception:
                        logger.exception("[credits] anthropic stream report failed")

                    yield StreamEvent(
                        type="message_end",
                        stop_reason=final.stop_reason,
                        usage={
                            "input_tokens": final_usage.input_tokens,
                            "output_tokens": final_usage.output_tokens,
                            "cache_read_input_tokens": cache_read,
                            "cache_creation_input_tokens": cache_creation,
                        },
                    )
                return  # Success — exit retry loop

            except anthropic.APIStatusError as e:
                # Bundle 401 self-heal — see create_message above.
                if e.status_code == 401 and attempt == 0 and (getattr(settings, "llm_mode", "") == "bundle"):
                    try:
                        from app.services import runtime_identity as _ri
                        _ri.reload()
                        _ri.apply_to_settings(_ri.all_runtime_fields())
                    except Exception as _re:
                        logger.warning(f"[ANTHROPIC] identity reload on 401 failed: {_re}")
                    self._keys.refresh()
                    self._ensure_client()
                    logger.info("[ANTHROPIC] bundle stream 401 — reloaded identity+token, retrying once")
                    continue
                if e.status_code in (429, 529) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Anthropic stream {e.status_code} {type(e).__name__}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise
            except anthropic.APIConnectionError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Anthropic stream connection error, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_anthropic_service: Optional[AnthropicService] = None


def get_anthropic_service() -> AnthropicService:
    global _anthropic_service
    if _anthropic_service is None:
        _anthropic_service = AnthropicService()
    return _anthropic_service
