"""
OpenAI Agent Service — Drop-in replacement for AnthropicService.

Uses OpenAI's chat completion API with tool calling.
Emits the same StreamEvent interface so the agent runner works unchanged.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, Any, List, Optional

from openai import AsyncOpenAI, RateLimitError, APIConnectionError

from app.config import settings

logger = logging.getLogger(__name__)


# Re-use the same event dataclasses so agent_runner doesn't change
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


def _anthropic_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Anthropic-format tool definitions to OpenAI function-calling format.
    
    Anthropic:  { name, description, input_schema: {...} }
    OpenAI:     { type: "function", function: { name, description, parameters: {...} } }
    """
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


class OpenAIAgentService:
    """
    OpenAI-based agent LLM service.
    Same streaming interface as AnthropicService so the agent runner is compatible.
    """

    def __init__(self):
        api_key = settings.openai_api_key
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — OpenAIAgentService will fail on calls")
        self.client = AsyncOpenAI(api_key=api_key or "missing")
        self.default_model = settings.agent_model
        self.default_max_tokens = settings.agent_max_tokens

    # ------------------------------------------------------------------
    # Streaming completion  (main interface used by agent_runner)
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
        Stream a chat completion. Yields StreamEvent objects matching the
        same contract as AnthropicService so the agent loop is unchanged.
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens

        # Build OpenAI messages list
        oai_messages = self._build_openai_messages(system, messages)

        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=oai_messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        # Convert Anthropic-format tools to OpenAI format
        if tools:
            kwargs["tools"] = _anthropic_tools_to_openai(tools)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                stream = await self.client.chat.completions.create(**kwargs)

                # Track tool calls being built across chunks
                tool_calls_in_progress: Dict[int, Dict[str, Any]] = {}
                usage_data: Dict[str, int] = {}
                finish_reason = ""

                async for chunk in stream:
                    # Usage comes in the final chunk
                    if chunk.usage:
                        usage_data = {
                            "input_tokens": chunk.usage.prompt_tokens or 0,
                            "output_tokens": chunk.usage.completion_tokens or 0,
                        }

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    # --- text delta ---
                    if delta and delta.content:
                        yield StreamEvent(type="text", text=delta.content)

                    # --- tool call deltas ---
                    if delta and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_in_progress:
                                tool_calls_in_progress[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }

                            tc = tool_calls_in_progress[idx]

                            if tc_delta.id:
                                tc["id"] = tc_delta.id

                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tc["name"] = tc_delta.function.name
                                    # Emit tool_use_start
                                    yield StreamEvent(
                                        type="tool_use_start",
                                        tool_name=tc["name"],
                                        tool_id=tc["id"],
                                    )
                                if tc_delta.function.arguments:
                                    tc["arguments"] += tc_delta.function.arguments

                # After stream ends, emit tool_use_end for each completed tool call
                for idx in sorted(tool_calls_in_progress):
                    tc = tool_calls_in_progress[idx]
                    try:
                        tool_input = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        tool_input = {"raw": tc["arguments"]}

                    yield StreamEvent(
                        type="tool_use_end",
                        tool_name=tc["name"],
                        tool_id=tc["id"],
                        tool_input=tool_input,
                    )

                # Map OpenAI finish reasons to Anthropic-style
                stop_reason_map = {
                    "stop": "end_turn",
                    "tool_calls": "tool_use",
                    "length": "max_tokens",
                    "content_filter": "end_turn",
                }
                mapped_stop = stop_reason_map.get(finish_reason, finish_reason)

                yield StreamEvent(
                    type="message_end",
                    stop_reason=mapped_stop,
                    usage=usage_data,
                )
                return  # Success, no retry

            except RateLimitError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"OpenAI rate-limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise
            except APIConnectionError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"OpenAI connection error, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise

    # ------------------------------------------------------------------
    # Message format conversion
    # ------------------------------------------------------------------
    def _convert_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert agent_runner's Anthropic-style messages to OpenAI format.
        
        Handles:
        - Simple string content → pass through
        - List content with text/tool_use blocks → OpenAI assistant message with tool_calls
        - List content with tool_result blocks → multiple "tool" role messages
        """
        role = msg["role"]
        content = msg["content"]

        # Simple string message
        if isinstance(content, str):
            return {"role": role, "content": content}

        # List of content blocks
        if isinstance(content, list):
            # Check if these are tool results (from user role)
            if content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                return {
                    "_multi": True,
                    "messages": [
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                        }
                        for block in content
                        if block.get("type") == "tool_result"
                    ],
                }

            # User message with multi-modal content (text + image_url blocks)
            # OpenAI natively supports this format — pass through directly
            if role == "user":
                has_image = any(
                    isinstance(b, dict) and b.get("type") in ("image_url", "image")
                    for b in content
                )
                if has_image:
                    return {"role": "user", "content": content}

            # Assistant message with text + tool_use blocks
            text_parts = []
            tool_calls = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })

            result: Dict[str, Any] = {"role": "assistant"}
            if text_parts:
                result["content"] = "\n".join(text_parts)
            else:
                result["content"] = None
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result

        return {"role": role, "content": str(content)}

    def _build_openai_messages(self, system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert a full message list, expanding multi-messages."""
        oai: List[Dict[str, Any]] = []
        if system:
            oai.append({"role": "system", "content": system})
        for msg in messages:
            converted = self._convert_message(msg)
            if converted.get("_multi"):
                oai.extend(converted["messages"])
            else:
                oai.append(converted)
        return oai


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_service: Optional[OpenAIAgentService] = None


def get_openai_agent_service() -> OpenAIAgentService:
    global _service
    if _service is None:
        _service = OpenAIAgentService()
    return _service
