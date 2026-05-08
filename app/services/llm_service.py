"""
LLM Service - Multi-provider wrapper for chat completions.

Supports:
- Anthropic Claude (preferred when ANTHROPIC_API_KEY is set)
- OpenAI (fallback)
- Token tracking, retries, streaming
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, List, Dict, Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM completion"""
    content: str
    model: str
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    finish_reason: str


@dataclass
class StreamChunk:
    """Chunk from streaming response"""
    content: str
    is_final: bool
    finish_reason: Optional[str] = None


def _should_use_anthropic(api_key: Optional[str] = None) -> bool:
    """Decide whether to use Anthropic backend.
    If caller passes an explicit OpenAI key, use OpenAI.
    Otherwise prefer Anthropic when its key is available."""
    if api_key:
        return False  # Explicit per-user OpenAI key
    from app.services.key_provider import keys
    return keys.has_anthropic


class LLMService:
    """
    Multi-provider LLM Service for chat completions.

    Automatically uses Anthropic Claude when ANTHROPIC_API_KEY is set,
    falls back to OpenAI otherwise. This ensures memory extraction,
    deduplication, and consolidation work without a valid OpenAI key.
    """

    def __init__(self, api_key: Optional[str] = None):
        from app.services.key_provider import keys
        self._use_anthropic = _should_use_anthropic(api_key)
        self._anthropic_svc = None
        self._openai_client = None

        if self._use_anthropic:
            logger.info("LLMService: using Anthropic backend")
            from app.services.anthropic_service import AnthropicService
            from app.services.model_resolver import default_anthropic_model
            self._anthropic_svc = AnthropicService()
            self.default_model = settings.agent_model or default_anthropic_model()
        else:
            from app.services.bundle_client import make_openai_client
            from openai import AsyncOpenAI
            client = make_openai_client(byok_key=api_key or keys.openai or None)
            self._openai_client = client or AsyncOpenAI(api_key="missing")
            self.default_model = settings.default_model

        self.default_temperature = settings.temperature
        self.default_max_tokens = settings.max_tokens

        # Initialize tokenizer
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(self.default_model)
        except Exception:
            try:
                import tiktoken
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if self._encoding:
            return len(self._encoding.encode(text))
        return len(text) // 4  # rough estimate

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for message in messages:
            total += 4
            total += self.count_tokens(message.get("content", ""))
            total += self.count_tokens(message.get("role", ""))
        total += 2
        return total
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion (Anthropic or OpenAI)."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        if self._use_anthropic:
            return await self._complete_anthropic(messages, model, temperature, max_tokens)
        return await self._complete_openai(messages, model, temperature, max_tokens, **kwargs)

    async def _complete_anthropic(
        self, messages, model, temperature, max_tokens
    ) -> LLMResponse:
        """Use AnthropicService for completion."""
        # Separate system message from conversation
        system = ""
        conv_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                conv_messages.append(msg)

        # Ensure we have at least one user message
        if not conv_messages:
            conv_messages = [{"role": "user", "content": ""}]

        resp = await self._anthropic_svc.create_message(
            messages=conv_messages,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(
            content=resp.content,
            model=resp.model,
            tokens_prompt=resp.tokens_input,
            tokens_completion=resp.tokens_output,
            tokens_total=resp.tokens_total,
            finish_reason=resp.stop_reason,
        )

    async def _complete_openai(
        self, messages, model, temperature, max_tokens, **kwargs
    ) -> LLMResponse:
        """Use OpenAI for completion."""
        from openai import APIError, RateLimitError, APIConnectionError

        max_retries = 3
        retry_delay = 1.0

        from app.services.model_resolver import supports_custom_temperature

        for attempt in range(max_retries):
            try:
                call_kwargs = dict(kwargs)
                if supports_custom_temperature(model):
                    call_kwargs["temperature"] = temperature
                response = await self._openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                choice = response.choices[0]
                usage = response.usage
                return LLMResponse(
                    content=choice.message.content or "",
                    model=response.model,
                    tokens_prompt=usage.prompt_tokens,
                    tokens_completion=usage.completion_tokens,
                    tokens_total=usage.total_tokens,
                    finish_reason=choice.finish_reason
                )
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    raise
            except APIConnectionError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    raise
            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion (OpenAI only for now)."""
        if self._use_anthropic:
            # For Anthropic, fall back to non-streaming and yield result at once
            resp = await self.complete(messages, model, temperature, max_tokens)
            yield StreamChunk(content=resp.content, is_final=True, finish_reason=resp.finish_reason)
            return

        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        from app.services.model_resolver import supports_custom_temperature

        try:
            call_kwargs = dict(kwargs)
            if supports_custom_temperature(model):
                call_kwargs["temperature"] = temperature
            stream = await self._openai_client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, stream=True, **call_kwargs,
            )
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta.content:
                        yield StreamChunk(content=choice.delta.content, is_final=False)
                    if choice.finish_reason:
                        yield StreamChunk(content="", is_final=True, finish_reason=choice.finish_reason)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    async def complete_with_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion expecting JSON response."""
        if self._use_anthropic:
            # Anthropic doesn't have response_format — the prompt already asks for JSON
            return await self.complete(messages, model, temperature, max_tokens)
        return await self.complete(
            messages=messages, model=model, temperature=temperature,
            max_tokens=max_tokens, response_format={"type": "json_object"}, **kwargs
        )


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
