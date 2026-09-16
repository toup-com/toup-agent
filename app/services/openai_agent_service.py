"""
OpenAI Agent Service — Drop-in replacement for AnthropicService.

Uses OpenAI's chat completion API with tool calling.
Emits the same StreamEvent interface so the agent runner works unchanged.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, Any, List, Optional

from openai import AsyncOpenAI, RateLimitError, APIConnectionError, AuthenticationError

from app.config import settings

logger = logging.getLogger(__name__)

# A canned `response.completed` used ONLY by `OpenAIAgentService.warm()` to
# drive the SDK's stream parser once. Minimal but structurally real: the
# expensive part is the union/discriminator resolution for the Response and
# its output items, which this exercises.
_WARM_RESPONSE_COMPLETED: Dict[str, Any] = {
    "type": "response.completed",
    "sequence_number": 0,
    "response": {
        "id": "resp_warm",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "warm",
        "output": [
            {
                "id": "msg_warm",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
        "metadata": {},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "text": {"format": {"type": "text"}},
        "truncation": "disabled",
        "user": None,
    },
}


def _abort_rather_than_replay(emitted_any: bool, exc: Exception, attempt: int) -> bool:
    """F-12. True when a mid-stream failure must be raised instead of retried.

    Both streaming paths put their retry loop AROUND the `async for`, so a
    connection error at chunk 500 restarts the request and yields a second,
    independently-generated answer into the SAME consumer, appended to the
    text the caller already received. The user reads a doubled reply, that
    doubled text is what gets persisted and what memory extraction later
    reads, and the whole prompt is re-billed.

    Resuming is not available to us: the model is regenerating from scratch
    and the new text does not continue the old prefix, so no amount of
    de-duplication reconstructs one coherent answer. Failing the turn is the
    honest outcome — the streaming-recover path already handles a raised
    error, and a visible failure beats silent corruption.

    Retries BEFORE any output (the common case: connect/rate-limit at
    request time) are untouched.

    Kill switch: settings.llm_stream_duplicate_guard = False restores the
    pre-2026-07-31 replay behaviour.
    """
    if not emitted_any:
        return False
    if not getattr(settings, "llm_stream_duplicate_guard", True):
        return False
    logger.warning(
        "[OPENAI] %s after partial output on attempt %d — aborting instead of "
        "replaying the stream (F-12)", type(exc).__name__, attempt,
    )
    return True


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


def _metering_idempotency_key(
    idempotency_key: Optional[str],
    prompt_cache_key: Optional[str],
    completion_id: Optional[str],
) -> Optional[str]:
    """Billing dedupe key for the direct (manual-mode/BYOK) OpenAI path.

    Legacy (``metering_correctness_v2=False``, default): the per-session
    constant agent_runner passes as ``idempotency_key``, falling back to
    ``prompt_cache_key`` — byte-identical to the historical behavior.
    Because that key is a per-SESSION constant (and the fallback is now
    DAY-scoped after PR-1), every call after the first idempotent-hits
    the credit ledger and is never metered (assessment A9-3 / F-11).

    ``metering_correctness_v2=True`` (W4.2 / gate G2 preparation): a
    per-REQUEST key — the OpenAI completion id when the stream carried
    one, else a fresh UUID — so every completed stream meters exactly
    once. Retries inside create_message_stream get a NEW completion id
    per attempt, which is correct: OpenAI bills each request. Forward-
    only; the flag flip is gated on written approval — see
    docs/audits/2026-07-g2-billing-gate.md.
    """
    if getattr(settings, "metering_correctness_v2", False):
        return f"oaireq:{completion_id or uuid.uuid4()}"
    return idempotency_key or prompt_cache_key


def _responses_cache_key(prompt_cache_key: str) -> str:
    """Fit ``prompt_cache_key`` into the Responses API's 64-char limit.

    ``/v1/responses`` rejects keys over 64 chars (``string_above_max_length``
    — hit on the 2026-07-29 canary parity soak with a 73-char day-scoped
    key); chat completions never enforced a limit, so the chat wire passes
    keys through untouched. Keys within the limit are returned unchanged.
    Longer keys keep a readable 31-char prefix and append a sha256 fragment
    of the FULL key — deterministic (same key → same routing hint, which is
    all the cache cares about) and collision-safe across scopes that share
    a prefix.
    """
    if len(prompt_cache_key) <= 64:
        return prompt_cache_key
    digest = hashlib.sha256(prompt_cache_key.encode("utf-8")).hexdigest()[:32]
    return f"{prompt_cache_key[:31]}-{digest}"


# Families that accept the Responses `reasoning` object. Derived from the
# model id for the same reason `wire_api_for` is (model_resolver ~l.100): a
# container can be pointed at any model by env or per-tenant config, and an
# unsupported `reasoning` key is a 400 on every turn — a whole-fleet outage
# that looks like a model problem. gpt-4o (the non-reasoning fallback) is
# deliberately excluded and gets no parameter.
#: Header the platform proxy reads to tag a request as platform overhead
#: rather than the user's turn (llm_proxy.proxy_responses). Only values the
#: proxy allowlists are honoured there; the agent sending it is necessary but
#: never sufficient.
OPERATION_TYPE_HEADER = "X-Toup-Operation-Type"


def is_system_operation(operation_type: Optional[str]) -> bool:
    """True for platform overhead that must never be billed to the user.

    Same "system." rule the platform enforces in three places already
    (llm_proxy._log_event, llm_proxy._get_spend, credits.agent-deduct), named
    once here so the agent side cannot drift from it.
    """
    return bool(operation_type and operation_type.startswith("system."))


_REASONING_EFFORT_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def supports_reasoning_effort(model: str | None) -> bool:
    """True when `model` accepts the Responses `reasoning: {effort}` param."""
    if not model:
        return False
    return model.lower().strip().startswith(_REASONING_EFFORT_PREFIXES)


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


def _anthropic_tools_to_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Anthropic-format tool definitions to Responses-API function format.

    Anthropic:  { name, description, input_schema: {...} }
    Responses:  { type: "function", name, description, parameters: {...}, strict }

    Unlike chat completions there is NO nested "function" wrapper. `strict`
    defaults to true on the Responses API — our tool schemas are not
    strict-clean (missing additionalProperties:false etc.), so we send an
    explicit False to keep today's chat-wire validation semantics.
    """
    responses_tools = []
    for tool in tools:
        responses_tools.append({
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            "strict": False,
        })
    return responses_tools


def _chat_tool_choice_to_responses(tool_choice: Any) -> Any:
    """
    Translate a chat-completions ``tool_choice`` into the Responses shape.

    Strings ("auto" / "required" / "none") pass through unchanged. The two
    dict shapes agent_runner / callers can produce need flattening:

      chat allowed_tools:  {"type": "allowed_tools", "allowed_tools":
                            {"mode": m, "tools": [{"type": "function",
                             "function": {"name": n}}, ...]}}
      responses:           {"type": "allowed_tools", "mode": m, "tools":
                            [{"type": "function", "name": n}, ...]}

      chat named function: {"type": "function", "function": {"name": n}}
      responses:           {"type": "function", "name": n}

    Unknown dict shapes pass through unchanged (forward-compat: if a caller
    already sends the Responses shape, we must not double-translate).
    """
    if not isinstance(tool_choice, dict):
        return tool_choice
    tc_type = tool_choice.get("type")
    if tc_type == "allowed_tools" and isinstance(tool_choice.get("allowed_tools"), dict):
        allowed = tool_choice["allowed_tools"]
        flat_tools: List[Dict[str, Any]] = []
        for t in allowed.get("tools", []) or []:
            if isinstance(t, dict) and isinstance(t.get("function"), dict):
                flat_tools.append({"type": "function", "name": t["function"].get("name", "")})
            else:
                flat_tools.append(t)
        return {
            "type": "allowed_tools",
            "mode": allowed.get("mode", "auto"),
            "tools": flat_tools,
        }
    if tc_type == "function" and isinstance(tool_choice.get("function"), dict):
        return {"type": "function", "name": tool_choice["function"].get("name", "")}
    return tool_choice


# Cap on the per-process call_id → reasoning-item cache used by the
# Responses wire path (§ _create_responses_stream). Bounded so a
# long-lived agent process can't grow it without limit, but sized to
# make FIFO eviction impossible WITHIN a live run loop: history
# rehydration is string-only (_load_history / day_context_loader), so
# function_call resubmission only ever replays tool_use blocks from the
# current in-process run() loop — worst case agent_max_tool_iterations
# (40) × a parallel tool batch per iteration, times concurrent runs
# (chat + routines/autopilot share the process). Evicting a live entry
# would resubmit a function_call without its reasoning item, which
# reasoning models reject (400). 2048 ≫ any realistic in-flight volume.
_REASONING_CACHE_MAX = 2048


class OpenAIAgentService:
    """
    OpenAI-based agent LLM service.
    Same streaming interface as AnthropicService so the agent runner is compatible.
    """

    def __init__(self):
        from app.services.key_provider import keys
        self._keys = keys
        self._key_version = -1  # Force initial build
        self.client = None
        self.default_model = settings.agent_model
        self.default_max_tokens = settings.agent_max_tokens
        # Responses wire path (openai_wire_api="responses"): call_id →
        # reasoning output item (id/summary/encrypted_content) captured
        # during streaming. Reasoning models reject a resubmitted
        # function_call item without its reasoning item, and store=false
        # means the only way to round-trip it is echoing the encrypted
        # content back on the next turn's input. Bounded FIFO.
        self._responses_reasoning: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        # `warm()` reaches `_ensure_client` from a worker thread (round 46);
        # everything else reaches it from the loop. See `_ensure_client`.
        import threading as _threading
        self._client_lock = _threading.RLock()
        self._ensure_client()

    def _ensure_client(self):
        """Rebuild the OpenAI client if the API key has changed.

        Locked, and the version stamp is written LAST. Until round 46 this ran
        only on the event-loop thread, so the two-step update was atomic by
        construction; `warm()` now calls it from `asyncio.to_thread`, and a
        turn landing inside that ~500 ms bind window could otherwise read the
        NEW version beside the OLD client and early-return — on a freshly
        claimed container the old client is `AsyncOpenAI(api_key="missing")`,
        i.e. the first turn 401s. Uncontended in the normal case."""
        if self._key_version == self._keys.version and self.client is not None:
            return
        with self._client_lock:
            if self._key_version == self._keys.version and self.client is not None:
                return
            version = self._keys.version

            from app.services.bundle_client import make_openai_client
            client = make_openai_client(byok_key=self._keys.openai or None)
            if client is None:
                logger.warning("OpenAI client could not be built (no key, not in bundle mode)")
                self.client = AsyncOpenAI(api_key="missing")
                self._key_version = version
                return
            self.client = client
            self._key_version = version
            logger.info("[OPENAI] Client rebuilt (mode=%s, v%d)",
                        settings.llm_mode, self._key_version)

    # ------------------------------------------------------------------
    # First-use warm  (round 46, A1)
    # ------------------------------------------------------------------
    def warm(self) -> float:
        """Pay the SDK's once-per-process lazy work here instead of inside
        the user's first turn. Returns milliseconds spent. Never raises.

        Two costs, both measured on 2026-09-15 with openai==2.53.0 /
        pydantic==2.13.5 (fresh process, 3 runs, unthrottled M-series):

          * `client.responses` is a `@cached_property` whose body is
            `from .resources.responses import AsyncResponses` — 134-141 ms
            on the first touch, 0.0003 ms after.
          * the first `construct_type(ResponseStreamEvent, …)` of a
            `response.completed` — the SDK's own stream parse path — is
            202-214 ms, and 0.42-0.54 ms on every subsequent call. A ~400x
            ratio: that is pydantic building the core schema for the
            Response/output-item union, not the payload.

        343 ms total, unthrottled. It does NOT account for the 9.47 s
        whole-process freeze on pool-82, which is why `loop_health` ships
        alongside; this removes a measured third of a second of on-loop
        first-use work from the user's first turn and claims nothing more.

        MUST be called off the loop (`asyncio.to_thread`): both halves hold
        the import lock / GIL, and running them on the loop is the defect,
        not the fix.
        """
        t0 = time.perf_counter()
        try:
            self._ensure_client()
            client = self.client
            if client is None:
                return 0.0
            # (a) the lazy resource imports
            _ = client.responses
            _ = client.chat.completions
            # (b) the pydantic core schemas for the Responses stream union,
            #     driven through the SDK's own parser so we warm exactly the
            #     path `AsyncStream.__stream__` takes and not a lookalike.
            from openai._models import construct_type
            from openai.types.responses import ResponseStreamEvent

            construct_type(type_=ResponseStreamEvent, value=_WARM_RESPONSE_COMPLETED)
        except Exception:
            # A warm that can fail a bind is worse than a cold wire.
            logger.debug("[OPENAI] warm skipped", exc_info=True)
            return 0.0
        finally:
            # In a `finally`: the fact `llm_wire_warm` reports is "the
            # once-per-process work has been ATTEMPTED here", and this is its
            # only producer. Marked on the success path alone, one swallowed
            # failure — a private import the SDK moves, a client that could
            # not be built — left the flag false for the life of the process.
            # It is a `turn_ready_detail` DIAGNOSTIC and never a term of
            # `serving` (D1), precisely because a single silent producer must
            # not be able to refuse a container that answers turns.
            mark_wire_warm()
        return (time.perf_counter() - t0) * 1000.0

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
        tool_choice: Optional[Any] = None,
        prompt_cache_key: Optional[str] = None,
        safety_identifier: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        stable_prefix_active: bool = False,
        channel: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        operation_type: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream a chat completion. Yields StreamEvent objects matching the
        same contract as AnthropicService so the agent loop is unchanged.

        ``reasoning_effort`` is honoured on the Responses wire only (the chat
        wire here exists for the gpt-4o fallback, which has no reasoning to
        budget); it is a request parameter, never prompt input.
        """
        self._ensure_client()
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens

        # Pre-flight: short-circuit if the last known platform state
        # says the user is out of credits. Raises OutOfCreditsError
        # which agent_runner converts into a credit_exhausted stream
        # event. Cheap, no extra HTTP — relies on state updated by
        # the previous call's deduct response. The first call after a
        # cold start fail-opens; subsequent calls are gated.
        from app.services.credit_reporter import raise_if_exhausted_async
        await raise_if_exhausted_async()

        # G1 (docs/audits/2026-08-g1-cost-and-latency.md): gpt-5.6-*
        # rejects /v1/chat/completions when function tools are present —
        # the Responses API is the tools+reasoning wire for that family.
        #
        # Derived from the resolved model, not read straight from settings:
        # the wire is a hard property of the model family, and treating it
        # as an independent setting let a container be configured with a
        # 5.6 model on the chat wire, which 400s every single turn while
        # looking perfectly healthy. `wire_api_for` still honours
        # settings.openai_wire_api for every other family, so the gpt-4o
        # fallback keeps the chat wire it has always used.
        from app.services.model_resolver import wire_api_for

        if wire_api_for(model) == "responses":
            async for _ev in self._create_responses_stream(
                messages=messages,
                system=system,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tool_choice=tool_choice,
                prompt_cache_key=prompt_cache_key,
                safety_identifier=safety_identifier,
                idempotency_key=idempotency_key,
                stable_prefix_active=stable_prefix_active,
                channel=channel,
                reasoning_effort=reasoning_effort,
                operation_type=operation_type,
            ):
                yield _ev
            return

        # Build OpenAI messages list
        oai_messages = self._build_openai_messages(system, messages)

        from app.services.model_resolver import supports_custom_temperature

        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=oai_messages,
            max_completion_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if supports_custom_temperature(model):
            kwargs["temperature"] = temperature
        # TKT-LAT-018: stable per-session cache key — improves OpenAI
        # prompt-cache hit rate by routing the request to a replica
        # that already holds the session's prefix in its local cache.
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key

        # Channel attribution for cache telemetry (alembic 082). Sent as a
        # header, NOT a body field: the body is forwarded verbatim to OpenAI
        # and an unknown key there is a 400 on the whole turn. The platform
        # proxy reads it (llm_proxy._sanitize_channel) and records it on the
        # llm_proxy_events row; OpenAI ignores an unrecognised header. Absent
        # when the caller reports no channel, which keeps every non-agent
        # caller and the BYOK direct-to-OpenAI path unchanged.
        if channel:
            kwargs["extra_headers"] = {"X-Toup-Channel": str(channel)[:20]}
        # Token-efficiency PR-1 (flag-gated: new request params are a
        # request-shape change; keep them off the legacy path).
        # prompt_cache_retention="24h" keeps the day prefix warm across
        # Day-as-Chat's natural gaps (default cache evicts after 5-10min
        # idle); safety_identifier carries the per-user abuse-detection
        # signal the deprecated `user` param used to (cache routing is
        # prompt_cache_key's job).
        #
        # Gated on the EFFECTIVE per-turn flag passed by agent_runner, NOT
        # the global setting — otherwise a per-tenant CANARY (global flag
        # off, user in stable_prefix_canary_user_ids) would get the stable
        # layout but miss retention="24h", and measured out at ~0.67
        # cached/prompt (intermittent replica routing) instead of the 0.89
        # that retention makes reliable. Measured on prod, 2026-07-24.
        if stable_prefix_active:
            kwargs["prompt_cache_retention"] = "24h"
            if safety_identifier:
                kwargs["safety_identifier"] = safety_identifier

        # Convert Anthropic-format tools to OpenAI format
        if tools:
            kwargs["tools"] = _anthropic_tools_to_openai(tools)
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        max_retries = 3
        for attempt in range(max_retries):
            try:
                stream = await self.client.chat.completions.create(**kwargs)

                # Track tool calls being built across chunks
                tool_calls_in_progress: Dict[int, Dict[str, Any]] = {}
                usage_data: Dict[str, int] = {}
                finish_reason = ""
                # W4.2: per-request OpenAI completion id ("chatcmpl-…"),
                # identical across all chunks of one stream and fresh on
                # each retry attempt. Only consumed when
                # metering_correctness_v2 is on (per-request billing
                # dedupe key); harmless capture otherwise.
                completion_id = ""
                # F-12: has this attempt already handed output to the
                # consumer? The retry wraps the whole `async for`, so a
                # restart after partial output replays text the caller
                # already has. See _abort_or_retry.
                emitted_any = False

                async for chunk in stream:
                    if not completion_id and getattr(chunk, "id", None):
                        completion_id = chunk.id
                    # Usage comes in the final chunk
                    if chunk.usage:
                        # TKT-LAT-018: capture cached_tokens for the
                        # cross-provider [PERF] log line. OpenAI nests
                        # it under prompt_tokens_details; older API
                        # versions may omit the field entirely.
                        _cached = 0
                        _details = getattr(chunk.usage, "prompt_tokens_details", None)
                        if _details is not None:
                            _cached = getattr(_details, "cached_tokens", 0) or 0
                        usage_data = {
                            "input_tokens": chunk.usage.prompt_tokens or 0,
                            "output_tokens": chunk.usage.completion_tokens or 0,
                            "cache_read_input_tokens": _cached,
                            "cache_creation_input_tokens": 0,
                        }

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    # --- text delta ---
                    if delta and delta.content:
                        emitted_any = True
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
                                    emitted_any = True
                                    yield StreamEvent(
                                        type="tool_use_start",
                                        tool_name=tc["name"],
                                        tool_id=tc["id"],
                                    )
                                if tc_delta.function.arguments:
                                    tc["arguments"] += tc_delta.function.arguments
                                    # Round 25, same as the Responses wire and
                                    # Anthropic: pass the increment on so a
                                    # skill can react while a long argument is
                                    # still arriving. Kept in step across all
                                    # three so a channel's behaviour does not
                                    # depend on which wire served the turn.
                                    yield StreamEvent(
                                        type="tool_use_input",
                                        tool_name=tc["name"],
                                        tool_id=tc["id"],
                                        text=tc_delta.function.arguments,
                                    )

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

                # TKT-LAT-018: cross-provider [PERF] log shape so a
                # single dashboard query aggregates Anthropic + OpenAI
                # cache hit rate without branching on provider.
                logger.info(
                    "[PERF] cache_read=%s cache_creation=0 input=%s output=%s model=%s provider=openai",
                    usage_data.get("cache_read_input_tokens", 0),
                    usage_data.get("input_tokens", 0),
                    usage_data.get("output_tokens", 0),
                    model,
                )

                # Credit metering for direct (non-proxy) OpenAI calls.
                # Updates the in-process CreditState so the NEXT call's
                # pre-flight short-circuits if this one took us to zero.
                # Network failures fail-open (DeductOutcome.network_ok=False
                # leaves state untouched).
                try:
                    from app.services.credit_reporter import report_llm_usage_bg
                    from app.config import settings as _cr_settings
                    user_id = getattr(_cr_settings, "user_id", "") or ""
                    # R44 F1 — same exemption as the Responses wire. This path
                    # is the gpt-4o fallback; a recorded head on that family
                    # would warm through here.
                    if user_id and not is_system_operation(operation_type):
                        report_llm_usage_bg(
                            user_id=user_id,
                            model=model,
                            provider="openai",
                            input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                            output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                            # Billing dedupe key. Flag OFF (default): the
                            # per-session key, byte-identical to before —
                            # which under-meters (~one ledger row/session,
                            # A9-3/F-11). Flag ON (metering_correctness_v2,
                            # gate G2): per-request completion id so every
                            # completed stream meters exactly once.
                            idempotency_key=_metering_idempotency_key(
                                idempotency_key,
                                prompt_cache_key,
                                completion_id or None,
                            ),
                            # F-7 / A9-1: cached hits were captured above but
                            # dropped at report time — forward them so the
                            # platform can persist cache telemetry.
                            cached_tokens=int(usage_data.get("cache_read_input_tokens", 0) or 0),
                        )
                except Exception:
                    logger.exception("[credits] openai stream report failed")

                yield StreamEvent(
                    type="message_end",
                    stop_reason=mapped_stop,
                    usage=usage_data,
                )
                return  # Success, no retry

            except AuthenticationError:
                # Bundle 401 self-heal. A pool container's first message can
                # race /admin/bind: the cached client still carries the GENERIC
                # lobby toup_token, so the platform proxy 401s ("Your API key is
                # invalid"). Reload runtime identity (re-read runtime.json the
                # bind wrote), remap connect_token→settings.toup_token, bump the
                # key version (Change 1 makes refresh() detect it), rebuild the
                # client with the fresh token, and retry ONCE. Bundle-only and
                # first-attempt-only so a genuine BYOK bad key fails fast and
                # surfaces the "finishing setup" friendly message instead.
                if attempt == 0 and (getattr(settings, "llm_mode", "") == "bundle"):
                    try:
                        from app.services import runtime_identity as _ri
                        _ri.reload()
                        _ri.apply_to_settings(_ri.all_runtime_fields())
                    except Exception as _re:
                        logger.warning(f"[OPENAI] identity reload on 401 failed: {_re}")
                    self._keys.refresh()
                    self._ensure_client()
                    logger.info("[OPENAI] bundle 401 — reloaded identity+token, retrying once")
                    continue
                raise
            except RateLimitError as e:
                if _abort_rather_than_replay(emitted_any, e, attempt):
                    raise
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"OpenAI rate-limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise
            except APIConnectionError as e:
                if _abort_rather_than_replay(emitted_any, e, attempt):
                    raise
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"OpenAI connection error, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise

    # ------------------------------------------------------------------
    # Responses API wire path (openai_wire_api="responses")
    # ------------------------------------------------------------------
    def _remember_reasoning(self, call_id: str, reasoning_item: Dict[str, Any]) -> None:
        """Cache a completed reasoning output item under the function call_id
        that followed it, so the next turn's input build can echo it back
        (stateless mode requires it — see _create_responses_stream). FIFO
        capped so a long-lived process stays bounded."""
        if not call_id:
            return
        self._responses_reasoning[call_id] = reasoning_item
        while len(self._responses_reasoning) > _REASONING_CACHE_MAX:
            self._responses_reasoning.popitem(last=False)

    @staticmethod
    def _reasoning_item_to_input(item: Any) -> Dict[str, Any]:
        """Normalize a streamed reasoning output item (SDK model or dict)
        into a plain input-item dict for resubmission."""
        def _get(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        summary_out: List[Dict[str, Any]] = []
        for s in _get(item, "summary", None) or []:
            if isinstance(s, dict):
                summary_out.append(s)
            elif hasattr(s, "model_dump"):
                summary_out.append(s.model_dump(exclude_none=True))
        result: Dict[str, Any] = {
            "type": "reasoning",
            "id": _get(item, "id", "") or "",
            "summary": summary_out,
        }
        encrypted = _get(item, "encrypted_content", None)
        if encrypted:
            result["encrypted_content"] = encrypted
        return result

    @staticmethod
    def _responses_usage_dict(usage: Any) -> Dict[str, int]:
        """Map a Responses usage object to the chat-path usage_data shape
        (the dict agent_runner reads off message_end)."""
        if usage is None:
            return {}
        _details = getattr(usage, "input_tokens_details", None)
        _cached = getattr(_details, "cached_tokens", 0) or 0
        return {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "cache_read_input_tokens": int(_cached),
            "cache_creation_input_tokens": 0,
        }

    def _convert_message_responses(
        self, msg: Dict[str, Any], seen_reasoning_ids: set
    ) -> List[Dict[str, Any]]:
        """
        Convert ONE agent_runner Anthropic-style message into Responses
        input items. Sibling of _convert_message (chat wire) — same input
        shapes, flattened-item output:

        - plain string → {"role", "content"} message
        - tool_result blocks → {"type": "function_call_output", ...} items
        - user multi-modal parts → input_text / input_image parts
        - assistant text + tool_use → assistant message (text, if any) then
          one function_call item per tool_use, each preceded by its cached
          reasoning item when we hold one (deduped per build by item id)
        """
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            return [{"role": role, "content": content}]

        if isinstance(content, list):
            # Tool results (from user role) → function_call_output items
            if content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                return [
                    {
                        "type": "function_call_output",
                        "call_id": block["tool_use_id"],
                        "output": block.get("content", ""),
                    }
                    for block in content
                    if block.get("type") == "tool_result"
                ]

            # User message with multi-modal content: chat-format parts are
            # passed through untouched on the chat wire; Responses renames
            # them (text→input_text, image_url→input_image w/ string URL).
            if role == "user":
                has_image = any(
                    isinstance(b, dict) and b.get("type") in ("image_url", "image")
                    for b in content
                )
                if has_image:
                    parts: List[Dict[str, Any]] = []
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        btype = b.get("type")
                        if btype == "text":
                            parts.append({"type": "input_text", "text": b.get("text", "")})
                        elif btype == "image_url":
                            url = b.get("image_url")
                            detail = None
                            if isinstance(url, dict):
                                detail = url.get("detail")
                                url = url.get("url", "")
                            part: Dict[str, Any] = {
                                "type": "input_image",
                                "image_url": url or "",
                            }
                            if detail:
                                part["detail"] = detail
                            parts.append(part)
                        elif btype == "image":
                            # Anthropic base64 spelling (browser screenshots)
                            source = b.get("source") or {}
                            if isinstance(source, dict) and source.get("data"):
                                media = source.get("media_type", "image/png")
                                parts.append({
                                    "type": "input_image",
                                    "image_url": f"data:{media};base64,{source['data']}",
                                })
                    return [{"role": "user", "content": parts}]

            # Assistant message with text + tool_use blocks
            items: List[Dict[str, Any]] = []
            text_parts: List[str] = []
            tool_use_blocks: List[Dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_use_blocks.append(block)

            text = "\n".join(text_parts)
            if text:
                items.append({"role": "assistant", "content": text})
            for block in tool_use_blocks:
                call_id = block["id"]
                reasoning = self._responses_reasoning.get(call_id)
                if reasoning is not None:
                    r_id = reasoning.get("id", "")
                    if r_id and r_id not in seen_reasoning_ids:
                        seen_reasoning_ids.add(r_id)
                        items.append(reasoning)
                items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                })
            return items

        return [{"role": role, "content": str(content)}]

    def _build_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert the full message list into a flat Responses input array.
        The system prompt is NOT an input item — it rides the top-level
        `instructions` param (equivalent placement to chat's system role)."""
        items: List[Dict[str, Any]] = []
        seen_reasoning_ids: set = set()
        for msg in messages:
            items.extend(self._convert_message_responses(msg, seen_reasoning_ids))
        return items

    async def _create_responses_stream(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        model: str = "",
        max_tokens: int = 0,
        temperature: float = 0.7,
        tool_choice: Optional[Any] = None,
        prompt_cache_key: Optional[str] = None,
        safety_identifier: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        stable_prefix_active: bool = False,
        channel: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        operation_type: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream a completion over the Responses API (/v1/responses), yielding
        the EXACT StreamEvent sequence create_message_stream's chat path
        produces: interleaved text + tool_use_start, all tool_use_end after
        the wire stream ends (deterministic output_index order, partial args
        never yielded), one trailing message_end with the same usage shape.

        Stateless multi-turn: store=false + the full input array each turn —
        no previous_response_id, no server-side conversation state, matching
        today's stateless chat usage. include=reasoning.encrypted_content so
        reasoning items can be echoed back next turn (the API rejects a
        resubmitted function_call without its reasoning item on reasoning
        models).

        `include=["reasoning.encrypted_content"]` stays unconditional. It was
        re-examined in R44 as a cache suspect and cleared: it is a RESPONSE
        directive — it asks for an extra field on the way back and adds
        nothing to the request's cached prefix (tools + instructions + input).
        Dropping it would only be safe on a build whose input carries no
        `function_call` item, and the input builder emits one for every
        tool_use in history, so any turn after a tool call would start
        400-ing. Left on.

        `reasoning: {effort}` IS sent when the caller asks for one (R44) —
        also a request parameter, outside the prefix, so it can vary per turn
        at no cache cost. Absent → the model's server-side default effort.
        """
        from app.services.model_resolver import supports_custom_temperature

        kwargs: Dict[str, Any] = dict(
            model=model,
            input=self._build_responses_input(messages),
            max_output_tokens=max_tokens,
            stream=True,
            store=False,
            include=["reasoning.encrypted_content"],
        )
        if system:
            kwargs["instructions"] = system
        if supports_custom_temperature(model):
            kwargs["temperature"] = temperature
        # R44: effort rides the request, never the prompt. Gated on the model
        # family rather than trusted from the caller — agent_runner decides
        # policy, this decides whether the wire can carry it.
        if reasoning_effort and supports_reasoning_effort(model):
            kwargs["reasoning"] = {"effort": reasoning_effort}
        # Same cache/abuse params as the chat wire (first-class Responses
        # params, verified against SDK types — prompt_cache_retention is
        # Literal["in-memory","24h"] on responses.create). Same effective
        # per-turn gate as the chat path (see comment there). Responses
        # streams always deliver usage in response.completed, so there is
        # no stream_options={"include_usage": True} equivalent to send.
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = _responses_cache_key(prompt_cache_key)
        if stable_prefix_active:
            kwargs["prompt_cache_retention"] = "24h"
            if safety_identifier:
                kwargs["safety_identifier"] = safety_identifier


        # Channel attribution for cache telemetry (alembic 082). Sent as a
        # header, NOT a body field: the body is forwarded verbatim to OpenAI
        # and an unknown key there is a 400 on the whole turn. The platform
        # proxy reads it (llm_proxy._sanitize_channel) and records it on the
        # llm_proxy_events row; OpenAI ignores an unrecognised header. Absent
        # when the caller reports no channel, which keeps every non-agent
        # caller and the BYOK direct-to-OpenAI path unchanged.
        if channel:
            kwargs["extra_headers"] = {"X-Toup-Channel": str(channel)[:20]}
        # R44 F1: platform-overhead marker. On the bundle/proxy path this is
        # the ONLY way the platform can tell a cache warm from a turn — an
        # untagged /responses call is logged with operation_type=None, which
        # llm_proxy treats as user-attributable and charges. The proxy
        # allowlists the value and checks the body still looks like a warm,
        # so sending it is necessary and not sufficient.
        if is_system_operation(operation_type):
            kwargs.setdefault("extra_headers", {})[OPERATION_TYPE_HEADER] = (
                str(operation_type)[:40]
            )

        if tools:
            kwargs["tools"] = _anthropic_tools_to_responses(tools)
            if tool_choice:
                kwargs["tool_choice"] = _chat_tool_choice_to_responses(tool_choice)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                stream = await self.client.responses.create(**kwargs)

                # item_id ("fc_…") → in-flight function_call tracker. The
                # yielded tool_id is ALWAYS the call_id ("call_…") — that's
                # what agent_runner stores as the tool_use id and echoes as
                # tool_result tool_use_id, and what the input builder reuses
                # verbatim as function_call_output call_id.
                fn_calls: Dict[str, Dict[str, Any]] = {}
                usage_data: Dict[str, int] = {}
                terminal_status = ""
                incomplete_reason = ""
                completion_id = ""  # "resp_…" — metering key (v2) only
                pending_reasoning: Optional[Dict[str, Any]] = None
                # F-12: see _abort_rather_than_replay.
                emitted_any = False

                async for ev in stream:
                    ev_type = getattr(ev, "type", "")

                    if ev_type == "response.created":
                        _resp = getattr(ev, "response", None)
                        if not completion_id and getattr(_resp, "id", None):
                            completion_id = _resp.id

                    elif ev_type == "response.output_text.delta":
                        delta = getattr(ev, "delta", "") or ""
                        if delta:
                            emitted_any = True
                            yield StreamEvent(type="text", text=delta)

                    elif ev_type == "response.output_item.added":
                        item = getattr(ev, "item", None)
                        if getattr(item, "type", "") == "function_call":
                            tracker = {
                                "call_id": getattr(item, "call_id", "") or "",
                                "name": getattr(item, "name", "") or "",
                                "arguments": "",
                                "order": getattr(ev, "output_index", len(fn_calls)),
                            }
                            fn_calls[getattr(item, "id", "") or ""] = tracker
                            # Mirrors the chat path's name-arrival emit:
                            # tool_use_start is live, args are buffered.
                            emitted_any = True
                            yield StreamEvent(
                                type="tool_use_start",
                                tool_name=tracker["name"],
                                tool_id=tracker["call_id"],
                            )

                    elif ev_type == "response.function_call_arguments.delta":
                        tracker = fn_calls.get(getattr(ev, "item_id", "") or "")
                        if tracker is not None:
                            _delta = getattr(ev, "delta", "") or ""
                            tracker["arguments"] += _delta
                            # Round 25. THIS is the live path — gpt-5.6-* is
                            # the fleet default and it is Responses-only, so
                            # every real app build streams its arguments right
                            # here and they were accumulated in silence. The
                            # wire has always carried them; we simply never
                            # passed them on. See `Skill.on_tool_input`.
                            if _delta:
                                emitted_any = True
                                yield StreamEvent(
                                    type="tool_use_input",
                                    tool_name=tracker["name"],
                                    tool_id=tracker["call_id"],
                                    text=_delta,
                                )

                    elif ev_type == "response.function_call_arguments.done":
                        tracker = fn_calls.get(getattr(ev, "item_id", "") or "")
                        if tracker is not None:
                            # Authoritative overwrite of the accumulated deltas
                            tracker["arguments"] = getattr(ev, "arguments", "") or ""

                    elif ev_type == "response.output_item.done":
                        item = getattr(ev, "item", None)
                        item_type = getattr(item, "type", "")
                        if item_type == "reasoning":
                            pending_reasoning = self._reasoning_item_to_input(item)
                        elif item_type == "function_call":
                            tracker = fn_calls.get(getattr(item, "id", "") or "")
                            if tracker is not None:
                                if getattr(item, "call_id", None):
                                    tracker["call_id"] = item.call_id
                                if getattr(item, "name", None):
                                    tracker["name"] = item.name
                                if getattr(item, "arguments", None):
                                    tracker["arguments"] = item.arguments
                                if pending_reasoning is not None and tracker["call_id"]:
                                    self._remember_reasoning(
                                        tracker["call_id"], pending_reasoning
                                    )
                                    pending_reasoning = None

                    elif ev_type == "response.completed":
                        _resp = getattr(ev, "response", None)
                        usage_data = self._responses_usage_dict(
                            getattr(_resp, "usage", None)
                        )
                        terminal_status = "completed"

                    elif ev_type == "response.incomplete":
                        _resp = getattr(ev, "response", None)
                        _usage = getattr(_resp, "usage", None)
                        if _usage is not None:
                            usage_data = self._responses_usage_dict(_usage)
                        terminal_status = "incomplete"
                        _details = getattr(_resp, "incomplete_details", None)
                        incomplete_reason = getattr(_details, "reason", "") or ""

                    elif ev_type in ("response.failed", "error"):
                        if ev_type == "error":
                            _code = getattr(ev, "code", "") or ""
                            _message = getattr(ev, "message", "") or ""
                        else:
                            _err = getattr(getattr(ev, "response", None), "error", None)
                            _code = getattr(_err, "code", "") or ""
                            _message = getattr(_err, "message", "") or ""
                        # Propagates like a generic APIError so agent_runner's
                        # existing fallback logic engages.
                        raise RuntimeError(
                            f"openai responses stream failed: {_code}: {_message}"
                        )

                    # Everything else (response.in_progress, content_part.*,
                    # output_text.done, reasoning summary deltas, obfuscation,
                    # future event types) is ignored silently.

                # After stream ends, emit tool_use_end per call in
                # output_index order (chat parity: ends after the wire
                # stream, deterministic order, parsed args).
                for tracker in sorted(fn_calls.values(), key=lambda t: t["order"]):
                    try:
                        tool_input = (
                            json.loads(tracker["arguments"])
                            if tracker["arguments"] else {}
                        )
                    except json.JSONDecodeError:
                        tool_input = {"raw": tracker["arguments"]}

                    yield StreamEvent(
                        type="tool_use_end",
                        tool_name=tracker["name"],
                        tool_id=tracker["call_id"],
                        tool_input=tool_input,
                    )

                # Responses has no finish_reason; infer the chat-parity stop
                # reason. The TERMINAL EVENT always wins — on the chat wire
                # the latched finish_reason is authoritative, so a stream
                # truncated mid-function-call maps "length"→"max_tokens"
                # even though tool_calls were partially buffered, and
                # agent_runner's exec gate (stop_reason == "tool_use") stays
                # closed instead of executing a call with truncated args.
                # Only a COMPLETED response with function_call items maps to
                # "tool_use" (chat: finish_reason="tool_calls");
                # incomplete/other (e.g. content_filter) maps to "end_turn"
                # like the chat table; a stream that died without a terminal
                # event maps to "" (chat: unlatched finish_reason) — in both
                # cases buffered fn_calls are surfaced but never executed.
                if terminal_status == "completed":
                    mapped_stop = "tool_use" if fn_calls else "end_turn"
                elif terminal_status == "incomplete":
                    mapped_stop = (
                        "max_tokens"
                        if incomplete_reason == "max_output_tokens"
                        else "end_turn"
                    )
                else:
                    mapped_stop = ""

                # Identical [PERF] shape to the chat path so the single
                # cross-provider dashboard query keeps working.
                logger.info(
                    "[PERF] cache_read=%s cache_creation=0 input=%s output=%s model=%s provider=openai",
                    usage_data.get("cache_read_input_tokens", 0),
                    usage_data.get("input_tokens", 0),
                    usage_data.get("output_tokens", 0),
                    model,
                )

                # Credit metering — same contract as the chat path (same
                # helper-derived idempotency key, same cached_tokens forward).
                #
                # R44 F1: a system operation is NOT reported at all on this
                # path. /credits/agent-deduct would exempt it by
                # operation_type anyway, but skipping is what makes the warm
                # free against a platform build that predates the exemption —
                # and the manual/BYOK path mints a unique
                # `oaireq:{completion_id}` key per call, so a single missed
                # exemption is a real charge to a user who asked for nothing.
                try:
                    from app.services.credit_reporter import report_llm_usage_bg
                    from app.config import settings as _cr_settings
                    user_id = getattr(_cr_settings, "user_id", "") or ""
                    if user_id and not is_system_operation(operation_type):
                        report_llm_usage_bg(
                            user_id=user_id,
                            model=model,
                            provider="openai",
                            input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                            output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                            idempotency_key=_metering_idempotency_key(
                                idempotency_key,
                                prompt_cache_key,
                                completion_id or None,
                            ),
                            cached_tokens=int(usage_data.get("cache_read_input_tokens", 0) or 0),
                        )
                except Exception:
                    logger.exception("[credits] openai stream report failed")

                yield StreamEvent(
                    type="message_end",
                    stop_reason=mapped_stop,
                    usage=usage_data,
                )
                return  # Success, no retry

            except AuthenticationError:
                # Bundle 401 self-heal — identical to the chat path (the
                # pool-bind race is wire-agnostic; see the comment there).
                if attempt == 0 and (getattr(settings, "llm_mode", "") == "bundle"):
                    try:
                        from app.services import runtime_identity as _ri
                        _ri.reload()
                        _ri.apply_to_settings(_ri.all_runtime_fields())
                    except Exception as _re:
                        logger.warning(f"[OPENAI] identity reload on 401 failed: {_re}")
                    self._keys.refresh()
                    self._ensure_client()
                    logger.info("[OPENAI] bundle 401 — reloaded identity+token, retrying once")
                    continue
                raise
            except RateLimitError as e:
                if _abort_rather_than_replay(emitted_any, e, attempt):
                    raise
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"OpenAI rate-limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise
            except APIConnectionError as e:
                if _abort_rather_than_replay(emitted_any, e, attempt):
                    raise
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
# LLM-wire warm state (round 46, A1)
# ---------------------------------------------------------------------------
# Read by `/agent/health` as `turn_ready_detail.llm_wire_warm`. Lives here
# rather than in a new module so the health handler reads it from the service
# that owns the fact — and so `warm()` cannot be called without setting it.
_wire_warm_at: Optional[float] = None


def mark_wire_warm() -> None:
    global _wire_warm_at
    if _wire_warm_at is None:
        _wire_warm_at = time.time()


def wire_warm() -> bool:
    return _wire_warm_at is not None


def wire_warm_at() -> Optional[float]:
    return _wire_warm_at


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_service: Optional[OpenAIAgentService] = None


def get_openai_agent_service() -> OpenAIAgentService:
    global _service
    if _service is None:
        _service = OpenAIAgentService()
    return _service
