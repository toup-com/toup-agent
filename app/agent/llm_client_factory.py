"""
LLM Client Factory — routes agent LLM calls based on LLM_MODE.

Usage:
    client = get_chat_client()
    # client is either an Anthropic client (direct or proxied) or OpenAI

    client = get_embedding_client()
    # Always returns OpenAI-compatible client (direct or proxied)

When LLM_MODE=bundle:
    - Chat → Anthropic via Toup proxy (https://toup.ai/api/llm)
    - Embeddings → OpenAI via Toup proxy
    - Voice/TTS → OpenAI via Toup proxy (future, v2)

When LLM_MODE=manual (BYOK):
    - Everything uses the user's own API keys directly
"""

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


def get_chat_client():
    """
    Return an Anthropic client configured for the active LLM mode.

    In bundle mode: points at the Toup LLM proxy with TOUP_LLM_TOKEN auth.
    In BYOK mode: points at Anthropic directly with the user's key.

    Returns None if no valid configuration exists.
    """
    if settings.llm_mode == "bundle" and settings.toup_token:
        try:
            import anthropic
            return anthropic.Anthropic(
                api_key=settings.toup_token,  # TOUP_LLM_TOKEN used as api_key
                base_url=f"{settings.platform_api_url}/llm",
            )
        except ImportError:
            logger.error("anthropic package not installed")
            return None

    # BYOK mode — use user's own Anthropic key
    if settings.anthropic_api_key:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        except ImportError:
            logger.error("anthropic package not installed")
            return None

    return None


def get_openai_client():
    """
    Return an OpenAI client configured for the active LLM mode.

    In bundle mode: points at the Toup LLM proxy for chat/embeddings.
    In BYOK mode: points at OpenAI directly with the user's key.
    """
    if settings.llm_mode == "bundle" and settings.toup_token:
        try:
            import openai
            return openai.OpenAI(
                api_key=settings.toup_token,
                base_url=f"{settings.platform_api_url}/llm/openai/v1",
            )
        except ImportError:
            logger.error("openai package not installed")
            return None

    # BYOK mode
    if settings.openai_api_key:
        try:
            import openai
            return openai.OpenAI(api_key=settings.openai_api_key)
        except ImportError:
            logger.error("openai package not installed")
            return None

    return None


def get_embedding_client():
    """
    Return an OpenAI client for embeddings.

    Bundle mode: proxied through Toup platform.
    BYOK mode: direct to OpenAI.
    """
    return get_openai_client()
