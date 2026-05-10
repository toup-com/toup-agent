"""
Internal LLM helper — platform-side system operations (archival summaries,
background jobs, etc.) that need to call Claude/GPT but can't route through
the HTTP LLM proxy (no agent auth token available).

Logs usage into llm_proxy_events with operation_type="system.*" so costs show
up in operational dashboards but do NOT count toward the user's monthly/daily
caps. See _get_spend() in app/api/llm_proxy.py for the exemption logic.

Usage:
    text = await call_anthropic_system(
        user_id=user_id,
        operation_type="system.day_archival",
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _calc_cost_cents(model: str, input_tokens: int, output_tokens: int) -> int:
    """Compute the cost of a system-op LLM call in cents.

    Tries the platform pricing table first (same source the proxy uses so
    cost dashboards compare apples-to-apples). Falls back to model-family
    heuristics so an Opus call isn't logged as Haiku-priced.
    """
    pricing = None
    if hasattr(settings, "pricing_per_1k"):
        pricing = settings.pricing_per_1k.get(model)
    if not pricing:
        pricing = _fallback_pricing_for_model(model)
    cost_usd = (input_tokens * pricing["input"] / 1000) + (output_tokens * pricing["output"] / 1000)
    return max(1, int(cost_usd * 100))


def _fallback_pricing_for_model(model: str) -> Dict[str, float]:
    """USD per 1k tokens. Matches Anthropic/OpenAI published pricing within ~10%.

    Used only when settings.pricing_per_1k is missing. Conservative (tends to
    overestimate slightly) so cost dashboards don't under-report.
    """
    m = (model or "").lower()
    # Claude family
    if "opus" in m:
        return {"input": 0.015, "output": 0.075}
    if "sonnet" in m:
        return {"input": 0.003, "output": 0.015}
    if "haiku" in m:
        return {"input": 0.0008, "output": 0.004}
    # OpenAI family
    if "gpt-5" in m or "gpt5" in m:
        return {"input": 0.005, "output": 0.015}
    if "gpt-4o-mini" in m:
        return {"input": 0.00015, "output": 0.0006}
    if "gpt-4o" in m or "gpt-4" in m:
        return {"input": 0.0025, "output": 0.01}
    # Unknown — assume mid-tier Claude pricing
    return {"input": 0.003, "output": 0.015}


async def _log_system_event(
    user_id: str,
    provider: str,
    model: str,
    operation_type: str,
    input_tokens: int,
    output_tokens: int,
    cost_cents: int,
    latency_ms: int,
    status: str = "ok",
):
    """Log a system-tagged LLMProxyEvent. Safe to call with no DB session available."""
    try:
        from app.db.database import async_session_maker
        from app.db.models import LLMProxyEvent

        async with async_session_maker() as db:
            event = LLMProxyEvent(
                id=str(uuid.uuid4()),
                user_id=user_id,
                provider=provider,
                model=model,
                endpoint="chat",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_cents=cost_cents,
                was_fallback=False,
                latency_ms=latency_ms,
                status=status,
                operation_type=operation_type,
            )
            db.add(event)
            await db.commit()
    except Exception as e:
        logger.warning("[internal_llm] Failed to log system event: %s", e)

    logger.info(
        "internal_llm user=%s provider=%s model=%s op=%s tokens_in=%d tokens_out=%d cost_cents=%d latency=%dms status=%s",
        (user_id or "-")[:8], provider, model, operation_type,
        input_tokens, output_tokens, cost_cents, latency_ms, status,
    )


async def call_anthropic_system(
    user_id: str,
    operation_type: str,
    model: str,
    max_tokens: int,
    system: str,
    messages: List[Dict[str, Any]],
    timeout: int = 45,
) -> Optional[str]:
    """Call Anthropic directly with platform credentials for a system operation.

    Returns the text response on success, None on failure. Always logs usage
    with operation_type so it's tracked but exempt from the user's cap.

    `operation_type` must start with "system." to be exempt (enforced by caller
    contract — the exemption check is in llm_proxy._get_spend).
    """
    if not operation_type.startswith("system."):
        raise ValueError(
            f"operation_type must start with 'system.' for internal calls (got {operation_type!r})"
        )

    api_key = (
        getattr(settings, "platform_anthropic_api_key", None)
        or settings.anthropic_api_key
    )
    if not api_key:
        logger.warning("[internal_llm] No Anthropic key for op=%s", operation_type)
        return None

    start = time.time()
    status = "ok"
    input_tokens = 0
    output_tokens = 0
    text: Optional[str] = None

    # Anthropic accepts two auth shapes:
    #   - API keys (sk-ant-api03-…) → `x-api-key`
    #   - OAuth tokens (sk-ant-oat01-…) → `Authorization: Bearer …` + beta
    # Mismatch → 401 invalid x-api-key, which silently broke every internal
    # Anthropic call (memory extraction, portrait, intent) for OAuth-mode
    # tenants. Detection now matches the rest of the codebase.
    is_oauth = isinstance(api_key, str) and api_key.startswith("sk-ant-oat")
    if is_oauth:
        ant_headers = {
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14",
            "content-type": "application/json",
        }
    else:
        ant_headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                _ANTHROPIC_URL,
                headers=ant_headers,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": messages,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                blocks = data.get("content", [])
                if blocks and isinstance(blocks, list):
                    text = blocks[0].get("text", "") or None
            else:
                status = "error"
                logger.warning(
                    "[internal_llm] Anthropic %d for op=%s: %s",
                    resp.status_code, operation_type, resp.text[:300],
                )
    except Exception as e:
        status = "error"
        logger.warning("[internal_llm] Anthropic call failed for op=%s: %s", operation_type, e)

    latency_ms = int((time.time() - start) * 1000)
    cost_cents = _calc_cost_cents(model, input_tokens, output_tokens)
    await _log_system_event(
        user_id=user_id,
        provider="anthropic",
        model=model,
        operation_type=operation_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_cents=cost_cents,
        latency_ms=latency_ms,
        status=status,
    )
    return text


async def call_openai_system(
    user_id: str,
    operation_type: str,
    model: str,
    max_tokens: int,
    system: str,
    messages: List[Dict[str, Any]],
    timeout: int = 45,
) -> Optional[str]:
    """OpenAI fallback for system operations (used when Anthropic key missing)."""
    if not operation_type.startswith("system."):
        raise ValueError(
            f"operation_type must start with 'system.' for internal calls (got {operation_type!r})"
        )

    api_key = (
        getattr(settings, "platform_openai_api_key", None)
        or settings.openai_api_key
    )
    if not api_key:
        logger.warning("[internal_llm] No OpenAI key for op=%s", operation_type)
        return None

    start = time.time()
    status = "ok"
    input_tokens = 0
    output_tokens = 0
    text: Optional[str] = None

    chat_messages = [{"role": "system", "content": system}] + list(messages)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                _OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": chat_messages,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content") or None
            else:
                status = "error"
                logger.warning(
                    "[internal_llm] OpenAI %d for op=%s: %s",
                    resp.status_code, operation_type, resp.text[:300],
                )
    except Exception as e:
        status = "error"
        logger.warning("[internal_llm] OpenAI call failed for op=%s: %s", operation_type, e)

    latency_ms = int((time.time() - start) * 1000)
    cost_cents = _calc_cost_cents(model, input_tokens, output_tokens)
    await _log_system_event(
        user_id=user_id,
        provider="openai",
        model=model,
        operation_type=operation_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_cents=cost_cents,
        latency_ms=latency_ms,
        status=status,
    )
    return text
