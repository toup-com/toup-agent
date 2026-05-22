"""Agent → platform credit deduction reporter.

When the tenant agent makes a DIRECT LLM call (manual mode, own API
key) it bypasses the platform's `/api/llm/chat` proxy and its
deduction hook. This module closes that gap by POSTing a deduction
record to `{platform_api_url}/api/credits/agent-deduct` after every
LLM call, authenticated with the agent's existing `X-Agent-Key`.

The platform's `agent_deduct` endpoint validates the key, routes
through `credit_service.try_charge`, and returns the new balance +
enforcement state. The agent should honor `enforcement_enabled=true
+ success=false` by short-circuiting subsequent LLM calls, but the
deduction is recorded either way so the operator can observe true
cost via /api/credits/ledger.

Fail-open by design — a credit-reporting outage must NEVER break the
agent's primary chat loop. All exceptions are caught and logged.

Used by:
    * services/anthropic_service.py — both streaming and non-streaming
      message paths, after `usage` is known.
    * services/openai_agent_service.py — after final stream event with
      `usage.prompt_tokens` / `usage.completion_tokens`.
    * services/internal_llm.py — system.* calls are explicitly
      skipped on the platform side, so reporting them is harmless.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import settings


logger = logging.getLogger(__name__)


# Bound the per-request timeout so the reporter never adds noticeable
# latency to the user's chat response. The deduction is fire-and-forget
# from the agent's perspective — when the platform is slow, we'd rather
# under-meter for one event than make the user wait.
_REPORT_TIMEOUT_S = 3.0


async def report_llm_usage(
    *,
    user_id: str,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    underlying_cost_cents: Optional[float] = None,
    operation_type: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    event_id: Optional[str] = None,
) -> Optional[dict]:
    """POST a deduction record to the platform.

    Returns the platform's AgentDeductResponse dict on success, None
    on any failure (network, auth, 4xx, 5xx). The agent's caller can
    inspect the response to short-circuit a future call when
    enforcement_enabled is True AND success is False — i.e. the user
    is out of credits. When the response is None the agent should
    proceed (fail-open) so a platform outage doesn't lock users out.
    """
    # Fast bail when the agent isn't configured to reach the platform.
    # Avoids spamming logs in dev where settings may be empty.
    platform_url = (getattr(settings, "platform_api_url", "") or "").strip()
    agent_key = (getattr(settings, "agent_api_key", "") or "").strip()
    if not platform_url or not agent_key:
        return None

    if input_tokens <= 0 and output_tokens <= 0:
        return None

    payload = {
        "user_id": user_id,
        "model": model,
        "provider": provider,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
    }
    if underlying_cost_cents is not None:
        payload["underlying_cost_cents"] = float(underlying_cost_cents)
    if operation_type:
        payload["operation_type"] = operation_type
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    if event_id:
        payload["event_id"] = event_id

    url = f"{platform_url.rstrip('/')}/credits/agent-deduct"
    try:
        async with httpx.AsyncClient(timeout=_REPORT_TIMEOUT_S) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"X-Agent-Key": agent_key},
            )
        if resp.status_code != 200:
            logger.warning(
                "[credits] agent-deduct non-200 status=%d body=%s",
                resp.status_code, resp.text[:300],
            )
            return None
        result = resp.json()
        if not result.get("idempotent_hit"):
            logger.info(
                "[credits] reported user=%s model=%s tokens=%d/%d "
                "balance_after=%s success=%s reason=%s",
                user_id[:8] if user_id else "?", model,
                input_tokens, output_tokens,
                result.get("balance_after"),
                result.get("success"),
                result.get("reason"),
            )
        return result
    except httpx.TimeoutException:
        logger.warning(
            "[credits] agent-deduct timed out (>%ss) user=%s model=%s",
            _REPORT_TIMEOUT_S, user_id[:8] if user_id else "?", model,
        )
        return None
    except Exception:
        logger.exception(
            "[credits] agent-deduct failed user=%s model=%s",
            user_id[:8] if user_id else "?", model,
        )
        return None


def is_out_of_credits(deduct_response: Optional[dict]) -> bool:
    """True iff the platform reports enforcement_enabled AND success=False.

    Agent callers should check this BEFORE the next LLM call to
    short-circuit out-of-credits users with a clear "upgrade" message
    rather than burning another roundtrip's worth of tokens.
    """
    if not deduct_response:
        return False
    return bool(
        deduct_response.get("enforcement_enabled")
        and not deduct_response.get("success")
    )
