"""
LLM Service - OpenAI API wrapper for chat completions

Provides:
- Chat completion with token tracking
- Streaming support
- Error handling and retries
- Model selection and configuration
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, List, Dict, Any, Optional
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
import tiktoken

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


class LLMService:
    """
    OpenAI LLM Service for chat completions.
    
    Handles:
    - Chat completions (streaming and non-streaming)
    - Token counting and tracking
    - Retry logic for transient errors
    - Model configuration
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.default_model = settings.default_model
        self.default_temperature = settings.temperature
        self.default_max_tokens = settings.max_tokens
        
        # Initialize tokenizer for the default model
        try:
            self._encoding = tiktoken.encoding_for_model(self.default_model)
        except KeyError:
            # Fall back to cl100k_base for newer models
            self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._encoding.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.
        
        Accounts for message formatting overhead.
        """
        total = 0
        for message in messages:
            # Each message has overhead: role + content wrapper
            total += 4  # Approximate overhead per message
            total += self.count_tokens(message.get("content", ""))
            total += self.count_tokens(message.get("role", ""))
        total += 2  # Priming tokens
        return total
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to settings.default_chat_model)
            temperature: Temperature for sampling (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters passed to OpenAI
        
        Returns:
            LLMResponse with content and token usage
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        
        # Retry logic for transient errors
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                # Extract response data
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
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            
            except APIConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Connection error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
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
        """
        Generate a streaming chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Yields:
            StreamChunk objects with content and metadata
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if delta.content:
                        yield StreamChunk(
                            content=delta.content,
                            is_final=False,
                            finish_reason=None
                        )
                    
                    if choice.finish_reason:
                        yield StreamChunk(
                            content="",
                            is_final=True,
                            finish_reason=choice.finish_reason
                        )
        
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
        """
        Generate a completion with JSON response format.
        
        Useful for structured extraction tasks.
        """
        return await self.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs
        )


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
