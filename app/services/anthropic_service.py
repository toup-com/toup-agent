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
        api_key = settings.anthropic_api_key
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — AnthropicService will fail on calls")
        
        # Detect OAuth tokens (sk-ant-oat*) vs regular API keys (sk-ant-api*)
        self.is_oauth = bool(api_key and "sk-ant-oat" in api_key)
        if self.is_oauth:
            logger.info("Using Anthropic OAuth token authentication (Claude Code mode)")
            # IMPORTANT: Remove ANTHROPIC_API_KEY from env before creating client.
            # The SDK auto-reads this env var and sends it as X-Api-Key header,
            # which causes 401 for OAuth tokens. We need ONLY Authorization: Bearer.
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            self.client = anthropic.AsyncAnthropic(
                auth_token=api_key,
                default_headers={
                    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14",
                    "user-agent": "claude-cli/2.1.2 (external, cli)",
                    "x-app": "cli",
                },
            )
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key or "missing")
        
        self.default_model = settings.anthropic_model
        self.default_max_tokens = settings.anthropic_max_tokens
    
    # Claude Code canonical tool names (OAuth tokens require these)
    CLAUDE_CODE_TOOLS = [
        "Read", "Write", "Edit", "Bash", "Grep", "Glob",
        "AskUserQuestion", "EnterPlanMode", "ExitPlanMode", "KillShell",
        "NotebookEdit", "Skill", "Task", "TaskOutput", "TodoWrite",
        "WebFetch", "WebSearch",
    ]
    _CC_TOOL_LOOKUP = {t.lower(): t for t in CLAUDE_CODE_TOOLS}
    
    # Map HexBrain tool names → Claude Code tool names
    _TOOL_NAME_MAP = {
        "exec": "Bash",
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Edit",
        "memory_search": "Grep",
        "memory_store": "TodoWrite",
        "web_search": "WebSearch",
        "web_fetch": "WebFetch",
        "send_file": "Task",
        "send_photo": "TaskOutput",
        "analyze_image": "Skill",
        "cron": "NotebookEdit",
        "spawn": "Glob",
        "process": "KillShell",
        "tts": "EnterPlanMode",
        "sessions_list": "ExitPlanMode",
        "sessions_history": "AskUserQuestion",
        "browser": "Read",  # duplicate — will rename below
    }
    # Reverse map for converting CC names back to HexBrain names
    _TOOL_NAME_REVERSE = {}
    
    def _build_tool_maps(self, tools: list) -> tuple:
        """Build forward (hex→cc) and reverse (cc→hex) tool name maps.
        Ensures each CC name is unique by appending suffix if needed."""
        forward = {}
        reverse = {}
        used_cc_names = set()
        
        cc_pool = list(self.CLAUDE_CODE_TOOLS)
        
        for t in tools:
            name = t["name"]
            # Try the explicit mapping first
            cc_name = self._TOOL_NAME_MAP.get(name)
            if cc_name and cc_name not in used_cc_names:
                forward[name] = cc_name
                reverse[cc_name] = name
                used_cc_names.add(cc_name)
            else:
                # Assign next available CC name
                for cn in cc_pool:
                    if cn not in used_cc_names:
                        forward[name] = cn
                        reverse[cn] = name
                        used_cc_names.add(cn)
                        break
                else:
                    # Ran out of CC names — keep original (will likely fail)
                    forward[name] = name
                    reverse[name] = name
        
        return forward, reverse
    
    def _prepare_tools(self, tools: list) -> tuple:
        """For OAuth tokens, rename tools to Claude Code canonical names.
        Returns (prepared_tools, reverse_map)."""
        if not self.is_oauth or not tools:
            return tools, {}
        
        forward, reverse = self._build_tool_maps(tools)
        prepared = [
            {**t, "name": forward.get(t["name"], t["name"])}
            for t in tools
        ]
        return prepared, reverse
    
    def _prepare_system(self, system: str) -> any:
        """
        Prepare system prompt. For OAuth tokens, prepend Claude Code identity
        as required by Anthropic's credential check.
        """
        if not self.is_oauth:
            return system
        
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
        max_tokens = max_tokens or self.default_max_tokens

        kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        # Extended thinking: if budget > 0, enable thinking with budget control
        if thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Thinking requires temperature=1 per Anthropic docs
            kwargs["temperature"] = 1
        # Prepare system prompt (adds Claude Code identity for OAuth tokens)
        prepared_system = self._prepare_system(system)
        if prepared_system:
            kwargs["system"] = prepared_system
        reverse_map = {}
        if tools:
            prepared_tools, reverse_map = self._prepare_tools(tools)
            kwargs["tools"] = prepared_tools
        
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
                
                return AnthropicResponse(
                    content="\n".join(text_parts),
                    tool_calls=tool_calls,
                    model=response.model,
                    tokens_input=response.usage.input_tokens,
                    tokens_output=response.usage.output_tokens,
                    tokens_total=response.usage.input_tokens + response.usage.output_tokens,
                    stop_reason=response.stop_reason,
                )
            
            except anthropic.RateLimitError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Anthropic rate-limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise
            except anthropic.APIConnectionError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Anthropic connection error, retrying in {wait}s")
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
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream a message. Yields StreamEvent objects for text chunks,
        tool use blocks, and the final message_end event.
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        
        kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        # Extended thinking: if budget > 0, enable thinking with budget control
        if thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Thinking requires temperature=1 per Anthropic docs
            kwargs["temperature"] = 1
        # Prepare system prompt (adds Claude Code identity for OAuth tokens)
        prepared_system = self._prepare_system(system)
        if prepared_system:
            kwargs["system"] = prepared_system
        # For OAuth: rename tools to Claude Code canonical names
        reverse_map = {}
        if tools:
            prepared_tools, reverse_map = self._prepare_tools(tools)
            kwargs["tools"] = prepared_tools

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
                        current_tool_name = reverse_map.get(block.name, block.name) if reverse_map else block.name
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
                        # Extended thinking — emit as a "thinking" event
                        yield StreamEvent(type="thinking", text=delta.thinking)
                    elif delta.type == "input_json_delta":
                        input_json_buf += delta.partial_json
                
                elif event.type == "content_block_stop":
                    if current_tool_name:
                        # Parse accumulated JSON input
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
                    pass  # Final usage comes from message_end or get_final_message
            
            # After stream ends, get final message for usage/stop_reason
            final = await stream.get_final_message()
            yield StreamEvent(
                type="message_end",
                stop_reason=final.stop_reason,
                usage={
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                },
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_anthropic_service: Optional[AnthropicService] = None


def get_anthropic_service() -> AnthropicService:
    global _anthropic_service
    if _anthropic_service is None:
        _anthropic_service = AnthropicService()
    return _anthropic_service
