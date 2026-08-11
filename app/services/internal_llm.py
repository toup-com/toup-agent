"""
Internal LLM helper — platform-side system operations (archival summaries,
background jobs, etc.) that need to call an LLM directly (no agent auth
token available in this call site).

Two-layer API:

  • call_system_llm()  — UNIFIED entry point. Resolves the active model
    via model_resolver, picks the SDK + auth path automatically: bundle
    proxy when the tenant is in bundle mode, direct API otherwise.
    This is what new code should call.

  • call_anthropic_system() / call_openai_system() — legacy direct-API
    callers. Kept for backwards compatibility with day_summarizer and
    any pre-existing caller that needed to pin a provider. New code
    should NOT use these — they bypass bundle mode and silently break
    on tenants whose chat works through the platform proxy.

Logs usage into llm_proxy_events with operation_type="system.*" so costs
show up in operational dashboards but do NOT count toward the user's
monthly/daily caps. See _get_spend() in app/api/llm_proxy.py for the
exemption logic.

Usage (preferred):
    text = await call_system_llm(
        user_id=user_id,
        operation_type="system.routine.email_briefing",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        # model: optional — defaults to model_resolver.default_model()
        # (i.e. settings.agent_model / AGENT_MODEL env var)
    )
"""

import logging
import sys
import time
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# ── Unified entry point ──────────────────────────────────────────────


async def call_system_llm(
    *,
    user_id: str,
    operation_type: str,
    max_tokens: int,
    system: str,
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    timeout: int = 45,
    failure_out: Optional[Dict[str, str]] = None,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
    safety_identifier: Optional[str] = None,
) -> Optional[str]:
    """Call the active LLM for a system operation.

    Resolves provider + auth automatically:
      1. `model` argument wins (caller pin — e.g. per-routine override).
      2. Else `model_resolver.default_model()` (settings.agent_model /
         AGENT_MODEL env). This is what the user's main chat uses, so a
         GPT-5.5 user gets GPT-5.5 in their routine too — no surprise
         provider switch.
      3. Provider is detected from the model id
         (`is_claude_model` / `is_openai_model`).
      4. If bundle mode is active (LLM_MODE=bundle + TOUP_TOKEN), the
         call routes through `bundle_client.make_*_client()` →
         {platform_api_url}/llm — same path the chat uses. Otherwise
         falls back to the legacy direct-API helper.

    Returns the text completion on success, None on failure. Failures
    are logged but never raised — caller decides how to react.

    `failure_out`: optional caller-supplied dict. When the call returns
    None, `failure_out["reason"]` is set to a taxonomy value from
    `_classify_failure_reason` so callers that persist structured
    failure stubs (day_summarizer M3) can record WHY. Untouched on
    success.
    W1.0: `prompt_cache_key` / `prompt_cache_retention` / `safety_identifier`
    are optional OpenAI prompt-cache params, forwarded only when set (no
    request-shape change for callers that don't opt in). The Anthropic
    dispatch accepts-and-ignores them (its cache is cache_control-based),
    but still forwards them to its transparent OpenAI fallback.
    """
    if not (operation_type.startswith("system.") or operation_type.startswith("user.")):
        raise ValueError(
            f"operation_type must start with 'system.' or 'user.' for internal calls (got {operation_type!r})"
        )

    from app.services.model_resolver import (
        default_model,
        is_claude_model,
        is_openai_model,
    )

    # Guardrail (2026-05-13 cost audit): `model=None` here resolves
    # to the user's chat model (`settings.agent_model`). That's
    # appropriate for one-shot user-initiated operations but a known
    # leak for any high-frequency / background call site. Log a WARN
    # with the operation_type so a regression shows up in the
    # dashboard within the first burst of calls — operator can grep
    # `[internal_llm] model=None` and see exactly which op is
    # falling through. Does NOT block the call; this is detection,
    # not enforcement.
    if not (model or "").strip():
        # Caller frame name — valid pre-first-await (direct awaits chain
        # frames normally); a Task-wrapped call just yields the runner name.
        try:
            caller = sys._getframe(1).f_code.co_name
        except Exception:
            caller = "?"
        logger.warning(
            "[internal_llm] model=None for operation_type=%s caller=%s — resolving to user agent_model. "
            "If this is a high-frequency or background call site, pin an explicit model "
            "(e.g. 'claude-haiku-4-5-20251001') to keep cost predictable.",
            operation_type,
            caller,
        )

    resolved_model = (model or "").strip() or default_model()

    # Classify provider. Unknown model id → treat as Claude (back-compat
    # with legacy callers that passed dated Anthropic model strings).
    is_openai = is_openai_model(resolved_model)
    is_claude = is_claude_model(resolved_model) or not is_openai

    if is_openai:
        return await _system_call_openai(
            user_id=user_id,
            operation_type=operation_type,
            model=resolved_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            timeout=timeout,
            failure_out=failure_out,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            safety_identifier=safety_identifier,
        )
    return await _system_call_anthropic(
        user_id=user_id,
        operation_type=operation_type,
        model=resolved_model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        timeout=timeout,
        failure_out=failure_out,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        safety_identifier=safety_identifier,
    )


def _bundle_active() -> bool:
    """Re-evaluated each call (settings may flip at runtime when the
    platform pushes new env via the bridge bind path)."""
    return (
        getattr(settings, "llm_mode", None) == "bundle"
        and bool(getattr(settings, "toup_token", None))
    )


def _classify_failure_reason(
    *,
    status_code: Optional[int] = None,
    exception: Optional[BaseException] = None,
) -> str:
    """Map a failed dispatch onto the M3 reason taxonomy used by the day
    summarizer ({auth_error, rate_limit, timeout, server_error,
    parse_error, no_keys, other}).

    Surfaced to callers via the `failure_out` param — values end up in
    day_chats.summary_last_failure_reason and metric labels, so keep the
    set small and stable (matches day_summarizer._classify_failure).
    """
    if exception is not None:
        # SDK errors (anthropic/openai APIStatusError) carry .status_code
        code = getattr(exception, "status_code", None)
        if isinstance(code, int):
            return _classify_failure_reason(status_code=code)
        name = type(exception).__name__.lower()
        if "timeout" in name or "timeout" in str(exception).lower():
            return "timeout"
        return "other"
    if status_code is None:
        return "other"
    if status_code in (401, 403):
        return "auth_error"
    if status_code == 429:
        return "rate_limit"
    if 500 <= status_code < 600:
        return "server_error"
    return "other"


# Production fallback model when an Anthropic dispatch can't complete.
# Cheapest production-grade option ($0.15/1M input, $0.60/1M output) on
# the platform's OpenAI key, vision-capable. Background callers that
# pin a `claude-…` model don't have to know about this — the fallback
# is transparent below.
_ANTHROPIC_FALLBACK_OPENAI_MODEL = "gpt-4o-mini"


async def _system_call_anthropic(
    *,
    user_id: str,
    operation_type: str,
    model: str,
    max_tokens: int,
    system: str,
    messages: List[Dict[str, Any]],
    timeout: int,
    failure_out: Optional[Dict[str, str]] = None,
    prompt_cache_key: Optional[str] = None,  # noqa: ARG001 — W1.0 signature parity; OpenAI-only, forwarded to the fallback below
    prompt_cache_retention: Optional[str] = None,  # noqa: ARG001 — W1.0 signature parity; OpenAI-only
    safety_identifier: Optional[str] = None,  # noqa: ARG001 — W1.0 signature parity; OpenAI-only
) -> Optional[str]:
    """Anthropic dispatch — bundle SDK when active, else direct API.

    Bundle path uses the same `make_anthropic_client()` the chat does,
    so credentials + WAF-bypass headers stay in lockstep with the rest
    of the agent. Single source of truth for "how does this agent reach
    Anthropic" — we don't reimplement auth here.

    Production fallback (2026-05-13): if the Anthropic path can't
    succeed (no key, bundle client returns None, SDK raises, direct
    call 4xx/5xx), we fall through to gpt-4o-mini on the platform's
    OpenAI key instead of silently returning None. Background features
    that pin a Claude model are decoupled from "is Anthropic reachable
    right now" — the call always either returns text or raises a
    transport-level error, never a "succeeded with None" zombie."""
    start = time.time()
    status = "ok"
    input_tokens = 0
    output_tokens = 0
    text: Optional[str] = None
    fallback_reason: Optional[str] = None  # set if we should fall through to OpenAI
    anthropic_reason = ""  # taxonomy reason for the Anthropic-side failure

    if _bundle_active():
        # SDK path — let bundle_client handle proxy URL, TOUP_TOKEN auth,
        # WAF header rewriting. We just call .messages.create().
        try:
            from app.services.bundle_client import make_anthropic_client
            client = make_anthropic_client()
            if client is None:
                logger.warning(
                    "[internal_llm] bundle mode but make_anthropic_client returned None for op=%s — falling back to %s",
                    operation_type, _ANTHROPIC_FALLBACK_OPENAI_MODEL,
                )
                fallback_reason = "bundle_client_none"
                anthropic_reason = "no_keys"
            else:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    timeout=timeout,
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                    # W1.0 telemetry parity — same [PERF] shape as the
                    # agent services so system calls show up in cache
                    # dashboards.
                    logger.info(
                        "[PERF] system_llm cache_read=%s input=%s output=%s model=%s provider=anthropic",
                        int(getattr(usage, "cache_read_input_tokens", 0) or 0),
                        input_tokens, output_tokens, model,
                    )
                blocks = getattr(resp, "content", None) or []
                if blocks:
                    first = blocks[0]
                    # SDK returns TextBlock objects; .text is the attr
                    text = getattr(first, "text", None) or None
        except Exception as e:
            status = "error"
            logger.warning(
                "[internal_llm] bundle Anthropic call failed for op=%s: %s: %s — falling back to %s",
                operation_type, type(e).__name__, str(e)[:300],
                _ANTHROPIC_FALLBACK_OPENAI_MODEL,
            )
            fallback_reason = f"bundle_exc_{type(e).__name__}"
            anthropic_reason = _classify_failure_reason(exception=e)
    else:
        # Legacy direct-API path. Same code as `call_anthropic_system`
        # below — inlined here to keep one logging call per dispatch.
        text, input_tokens, output_tokens, status, anthropic_reason = await _direct_anthropic(
            operation_type=operation_type,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            timeout=timeout,
        )
        if status == "error" or text is None:
            fallback_reason = "direct_failed_or_empty"
            logger.warning(
                "[internal_llm] direct Anthropic returned status=%s text=%s for op=%s — falling back to %s",
                status, "None" if text is None else "<text>",
                operation_type, _ANTHROPIC_FALLBACK_OPENAI_MODEL,
            )

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

    # Credit deduction for direct-API mode. Bundle mode already
    # deducts via llm_proxy._log_event so we'd double-charge if we
    # called report_llm_usage in that case. Only fires for user.*
    # ops — system.* is exempt server-side and skipping the round-
    # trip saves a few ms per turn.
    if (
        not _bundle_active()
        and status == "ok"
        and (input_tokens > 0 or output_tokens > 0)
        and not operation_type.startswith("system.")
    ):
        try:
            from app.services.credit_reporter import report_llm_usage
            await report_llm_usage(
                user_id=user_id,
                model=model,
                provider="anthropic",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                underlying_cost_cents=float(cost_cents),
                operation_type=operation_type,
            )
        except Exception:
            logger.exception("[credits] internal_llm anthropic direct deduct failed")

    # Production fallback: Anthropic dispatch couldn't complete cleanly
    # — call gpt-4o-mini through the OpenAI path and return its result.
    # Both calls get logged separately to llm_proxy_events so the dashboard
    # shows the failed attempt + the recovery in chronological order.
    if fallback_reason is not None:
        logger.info(
            "[internal_llm] Anthropic fallback engaged for op=%s reason=%s → %s",
            operation_type, fallback_reason, _ANTHROPIC_FALLBACK_OPENAI_MODEL,
        )
        fb_text = await _system_call_openai(
            user_id=user_id,
            operation_type=operation_type + ".anthropic_fallback",
            model=_ANTHROPIC_FALLBACK_OPENAI_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            timeout=timeout,
            failure_out=failure_out,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            safety_identifier=safety_identifier,
        )
        # Both providers failed: surface the more diagnostic reason —
        # the Anthropic one wins unless it was generic (mirrors the M1
        # composition the day summarizer used before this path existed).
        # "no_keys" counts as generic here: with Anthropic disabled
        # platform-wide the anthropic leg ALWAYS fails no_keys, and letting
        # it stomp the OpenAI leg's real diagnostic (rate_limit, timeout…)
        # would mislabel every dual failure on the prod-default config.
        if fb_text is None and failure_out is not None and anthropic_reason not in ("", "other", "no_keys"):
            failure_out["reason"] = anthropic_reason
        return fb_text
    if text is None and failure_out is not None:
        # 200-with-empty-content (no fallback engaged) — closest taxonomy fit.
        failure_out["reason"] = anthropic_reason or "parse_error"
    return text


async def _system_call_openai(
    *,
    user_id: str,
    operation_type: str,
    model: str,
    max_tokens: int,
    system: str,
    messages: List[Dict[str, Any]],
    timeout: int,
    failure_out: Optional[Dict[str, str]] = None,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
    safety_identifier: Optional[str] = None,
) -> Optional[str]:
    """OpenAI dispatch — bundle SDK when active, else direct API.

    Handles the gpt-5.x / o-series quirks centrally (max_completion_tokens
    instead of max_tokens, no custom temperature). System prompt is
    prepended to messages — OpenAI uses an in-band 'system' role, not a
    separate field like Anthropic."""
    from app.services.model_resolver import (
        is_reasoning_model,
        supports_custom_temperature,
        uses_max_completion_tokens,
    )

    start = time.time()
    status = "ok"
    input_tokens = 0
    output_tokens = 0
    text: Optional[str] = None
    openai_reason = ""  # taxonomy reason for a failed dispatch

    chat_messages = [{"role": "system", "content": system}] + list(messages)
    token_kwarg = "max_completion_tokens" if uses_max_completion_tokens(model) else "max_tokens"
    is_reasoning = is_reasoning_model(model)

    if _bundle_active():
        try:
            from app.services.bundle_client import make_openai_client
            client = make_openai_client()
            if client is None:
                logger.warning(
                    "[internal_llm] bundle mode but make_openai_client returned None for op=%s",
                    operation_type,
                )
                if failure_out is not None:
                    failure_out["reason"] = "no_keys"
                return None
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": chat_messages,
                token_kwarg: max_tokens,
                "timeout": timeout,
            }
            # W1.0: prompt-cache params — only set when the caller opted
            # in, so the request shape is unchanged otherwise.
            if prompt_cache_key:
                kwargs["prompt_cache_key"] = prompt_cache_key
            if prompt_cache_retention:
                kwargs["prompt_cache_retention"] = prompt_cache_retention
            if safety_identifier:
                kwargs["safety_identifier"] = safety_identifier
            # Reasoning models (gpt-5.x, o-series) silently spend output
            # budget on hidden chain-of-thought BEFORE the visible answer.
            # For a summarisation job there's nothing to think hard about
            # — pin `reasoning_effort="low"` so the budget is reserved
            # for the actual summary. Without this, a 13k-token email
            # input on gpt-5.5 with max_completion_tokens=1200 spent the
            # entire 1200 on reasoning and returned empty content — caller
            # saw `llm_returned_none` even though the HTTP call succeeded.
            # ("low" not "minimal" — gpt-5.5 only accepts
            # {none, low, medium, high, xhigh}; "low" is the smallest
            # universally-supported value.)
            if is_reasoning:
                kwargs["reasoning_effort"] = "low"
            # gpt-5.x / reasoning models reject explicit temperature.
            # We never need a non-default temperature for system ops, so
            # we just omit it entirely (default is 1.0) — safer than
            # branching per-model.
            resp = await client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None)
            reasoning_tokens = 0
            if usage is not None:
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                # Surface reasoning portion if the SDK exposed it — lets
                # ops spot future truncation early.
                details = getattr(usage, "completion_tokens_details", None)
                if details is not None:
                    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
                # W1.0 telemetry parity — same [PERF] shape as the agent
                # services so system calls show up in cache dashboards.
                _prompt_details = getattr(usage, "prompt_tokens_details", None)
                logger.info(
                    "[PERF] system_llm cache_read=%s input=%s output=%s model=%s provider=openai",
                    int(getattr(_prompt_details, "cached_tokens", 0) or 0) if _prompt_details is not None else 0,
                    input_tokens, output_tokens, model,
                )
            choices = getattr(resp, "choices", None) or []
            finish_reason = None
            if choices:
                finish_reason = getattr(choices[0], "finish_reason", None)
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    text = (getattr(msg, "content", None) or "").strip() or None
            # Empty content + non-zero output_tokens almost always means
            # the reasoning budget ate everything. Surface that loudly so
            # we don't return a misleading None up to the caller without
            # an obvious log signature for triage.
            if text is None and output_tokens > 0:
                logger.warning(
                    "[internal_llm] OpenAI returned empty text op=%s model=%s "
                    "finish=%s out_tokens=%d reasoning_tokens=%d — likely "
                    "reasoning consumed the budget; raise max_tokens or set "
                    "reasoning_effort='minimal'",
                    operation_type, model, finish_reason,
                    output_tokens, reasoning_tokens,
                )
                status = "empty_output"
                openai_reason = "parse_error"
        except Exception as e:
            status = "error"
            openai_reason = _classify_failure_reason(exception=e)
            logger.warning(
                "[internal_llm] bundle OpenAI call failed for op=%s: %s: %s",
                operation_type, type(e).__name__, str(e)[:300],
            )
    else:
        text, input_tokens, output_tokens, status, openai_reason = await _direct_openai(
            operation_type=operation_type,
            model=model,
            max_tokens=max_tokens,
            chat_messages=chat_messages,
            token_kwarg=token_kwarg,
            timeout=timeout,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            safety_identifier=safety_identifier,
        )

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

    # Credit deduction for direct-API mode (see _system_call_anthropic).
    if (
        not _bundle_active()
        and status == "ok"
        and (input_tokens > 0 or output_tokens > 0)
        and not operation_type.startswith("system.")
    ):
        try:
            from app.services.credit_reporter import report_llm_usage
            await report_llm_usage(
                user_id=user_id,
                model=model,
                provider="openai",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                underlying_cost_cents=float(cost_cents),
                operation_type=operation_type,
            )
        except Exception:
            logger.exception("[credits] internal_llm openai direct deduct failed")
    if text is None and failure_out is not None:
        failure_out["reason"] = openai_reason or "parse_error"
    return text


async def _direct_anthropic(
    *,
    operation_type: str,
    model: str,
    max_tokens: int,
    system: str,
    messages: List[Dict[str, Any]],
    timeout: int,
) -> tuple[Optional[str], int, int, str, str]:
    """BYOK direct-API Anthropic call. Returns (text, in_tokens, out_tokens, status, reason).

    `reason` is "" on success, else a `_classify_failure_reason` value.
    Same auth + OAuth detection as the legacy `call_anthropic_system`,
    factored out so the unified dispatcher can share it."""
    api_key = (
        getattr(settings, "platform_anthropic_api_key", None)
        or settings.anthropic_api_key
    )
    if not api_key:
        logger.warning("[internal_llm] No Anthropic key for op=%s", operation_type)
        return None, 0, 0, "error", "no_keys"

    is_oauth = isinstance(api_key, str) and api_key.startswith("sk-ant-oat")
    if is_oauth:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14",
            "content-type": "application/json",
        }
    else:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                _ANTHROPIC_URL,
                headers=headers,
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
                blocks = data.get("content", [])
                text = blocks[0].get("text") if blocks else None
                return (
                    text or None,
                    int(usage.get("input_tokens", 0) or 0),
                    int(usage.get("output_tokens", 0) or 0),
                    "ok",
                    "" if text else "parse_error",
                )
            logger.warning(
                "[internal_llm] Anthropic %d for op=%s: %s",
                resp.status_code, operation_type, resp.text[:300],
            )
            return None, 0, 0, "error", _classify_failure_reason(status_code=resp.status_code)
    except Exception as e:
        logger.warning("[internal_llm] Anthropic call failed for op=%s: %s", operation_type, e)
        return None, 0, 0, "error", _classify_failure_reason(exception=e)


async def _direct_openai(
    *,
    operation_type: str,
    model: str,
    max_tokens: int,
    chat_messages: List[Dict[str, Any]],
    token_kwarg: str,
    timeout: int,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
    safety_identifier: Optional[str] = None,
) -> tuple[Optional[str], int, int, str, str]:
    """BYOK direct-API OpenAI call. Returns (text, in_tokens, out_tokens, status, reason).

    `reason` is "" on success, else a `_classify_failure_reason` value."""
    api_key = (
        getattr(settings, "platform_openai_api_key", None)
        or settings.openai_api_key
    )
    if not api_key:
        logger.warning("[internal_llm] No OpenAI key for op=%s", operation_type)
        return None, 0, 0, "error", "no_keys"

    body: Dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        token_kwarg: max_tokens,
    }
    # W1.0: prompt-cache params — only set when the caller opted in.
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_retention:
        body["prompt_cache_retention"] = prompt_cache_retention
    if safety_identifier:
        body["safety_identifier"] = safety_identifier

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                _OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
                json=body,
            )
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                choices = data.get("choices", [])
                text = (
                    choices[0].get("message", {}).get("content")
                    if choices else None
                )
                # W1.0 telemetry parity (see _system_call_openai).
                logger.info(
                    "[PERF] system_llm cache_read=%s input=%s output=%s model=%s provider=openai",
                    int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0),
                    int(usage.get("prompt_tokens", 0) or 0),
                    int(usage.get("completion_tokens", 0) or 0),
                    model,
                )
                return (
                    text or None,
                    int(usage.get("prompt_tokens", 0) or 0),
                    int(usage.get("completion_tokens", 0) or 0),
                    "ok",
                    "" if text else "parse_error",
                )
            logger.warning(
                "[internal_llm] OpenAI %d for op=%s: %s",
                resp.status_code, operation_type, resp.text[:300],
            )
            return None, 0, 0, "error", _classify_failure_reason(status_code=resp.status_code)
    except Exception as e:
        logger.warning("[internal_llm] OpenAI call failed for op=%s: %s", operation_type, e)
        return None, 0, 0, "error", _classify_failure_reason(exception=e)


def _calc_cost_cents(model: str, input_tokens: int, output_tokens: int) -> Decimal:
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
    # R-3: exact to 4dp, capped at the legacy floored value — recorded cost
    # may only ever go down (see llm_proxy._never_higher_cents, same rule).
    exact = (Decimal(str(cost_usd)) * 100).quantize(Decimal("0.0001"))
    if exact <= 0:
        return Decimal("0")
    return min(exact, Decimal(max(1, int(cost_usd * 100))))


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
        # cost_cents is a Decimal since R-3; %d truncates sub-cent costs to 0.
        "internal_llm user=%s provider=%s model=%s op=%s tokens_in=%d tokens_out=%d cost_cents=%s latency=%dms status=%s",
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
    if not (operation_type.startswith("system.") or operation_type.startswith("user.")):
        raise ValueError(
            f"operation_type must start with 'system.' or 'user.' for internal calls (got {operation_type!r})"
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
    if not (operation_type.startswith("system.") or operation_type.startswith("user.")):
        raise ValueError(
            f"operation_type must start with 'system.' or 'user.' for internal calls (got {operation_type!r})"
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
