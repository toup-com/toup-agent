"""Warm the provider's prompt-cache HEAD when a chat socket attaches.

Measured 2026-09-13/14 over 76 iteration-1 LLM calls (the ones that do the
cross-turn cache lookup): 26 missed. Split by comparing each miss against the
previous call from the same container —

  * 15  the container's first call in the window (cold process / first of day)
  *  7  IDENTICAL head hash, 4-6 hours idle  ← this module's target
  *  4  head changed, and every one of those also changed its TOOLS hash
        (a different tenant bound to that pool slot), or sat behind a 6 h gap

So the misses are TTL, not layout and not key rotation: the longest observed
HIT gap is 1,616 s and the shortest same-head MISS gap is 14,419 s, despite
`prompt_cache_retention="24h"`. The cacheable head is instructions + tools
(~40,192 tokens in the 2026-09-14 sample) and needs no day chat and no
history — so a socket attaching after an idle stretch can pay for the head
BEFORE the user finishes typing, and the turn that follows reads it.

Why this replays a RECORDED head instead of rebuilding one
----------------------------------------------------------
A warm whose bytes differ from the next turn's by one character is a paid
no-op that reports itself as a success. Rebuilding the head at connect time
would have to re-derive the system prompt (whose Runtime Context carries the
user's local DATE) and the wire tools array (channel converge, the surface
disable set, connector scope) in a SECOND place, and the two would drift
silently — the failure mode is invisible because the model still answers.

`record_head()` therefore stores the exact bytes `agent_runner.run()` last
handed the wire, and the warm replays them. Byte-equality is true by
construction rather than by review. The cost is scope: a process that has
served no turn for this user has nothing to replay, and refuses.

An earlier revision of this module ALSO refused across local midnight,
because the system prompt carried a `Today's date:` line and yesterday's head
was a different head. That line has since moved to the per-turn
`<turn_context>` tail (prefix_stability.render_time_lines), so the recorded
head no longer expires at midnight and the refusal is gone — which is what
turns the first message of a day, the single largest miss bucket, into
something a warm can actually serve. `local_date` and `tz_name` are still
recorded: they cost nothing, and they are what a future check would need if
anything date-shaped ever re-enters the head.

Shipped dark
------------
`llm_cache_warm_on_connect` defaults to **False**. The warm is only free
because BOTH halves of the no-charge path are live: this module skips the
agent's own `report_llm_usage_bg` for a system operation, and the platform's
`llm_proxy.proxy_responses` honours the `X-Toup-Operation-Type` header so the
`llm_proxy_events` row is filed with `operation_type="system.cache_warm"` —
which `_log_event`, `_get_spend` and `/cache/daily` all already exempt.
Enable only when the platform build carrying `_system_operation_for` is
deployed everywhere, the per-day cap is in place, and one canary tenant has
been watched for a day with its warm rows carrying that operation_type and no
matching `credit_ledger` entry. Sending a warm against a platform that
predates the exemption is a real ~10-credit charge per warm to a user who
asked for nothing.

Cost, when it is on: one warm is the head at the uncached input rate —
40,192 tokens x $2.50/1M on gpt-5.6-terra = $0.100, all of it Toup's. When it
lands, the turn behind it reads that head cached ($0.010) instead of paying
$0.100, so the net is ~$0.010 for ttft 5,586 -> 2,611 ms p50. Capped at
`llm_cache_warm_max_per_day` (12) per user per process: $1.21/user/day worst
case, and $0 to the user in every case.

The hook
--------
`schedule_warm` has exactly ONE caller: `app/api/ws_chat.py::ws_chat`,
immediately after the `[WS] Authenticated user:` line and above the
drain-state increment. That is the live agent-side chat accept point —
`app/modules/chat/ws_router.py` looks like a candidate and is NOT one, because
`app/main.py:31` imports `ws_chat_router` from `app.api.ws_chat` and the
modules copy is mounted by nothing.

`schedule_warm` returns before any I/O and swallows everything, so a socket
can never be delayed or failed by a cache optimisation.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Channels that may trigger a warm. The chat WebSocket is by construction the
#: app/web/mobile socket — WhatsApp, Telegram, triggers and routines arrive
#: over HTTP, and live voice has its own relay — but the deny set is explicit
#: anyway so a future caller cannot quietly widen it.
#: The platform's marker for "this is Toup's overhead, not the user's turn".
#: Allowlisted by llm_proxy._system_operation_for and exempted from the
#: charge, the daily cap and /cache/daily. Changing this string without
#: changing the platform's allowlist bills every warm to a user.
SYSTEM_OPERATION = "system.cache_warm"

WARM_DENY_CHANNELS = frozenset({
    "voice", "whatsapp", "telegram", "trigger", "routine", "automation",
    "agent_task", "subagent", "health_probe", "vibecoding",
})

#: user_id -> the exact head bytes of that user's last real LLM call.
_HEADS: Dict[str, Dict[str, Any]] = {}
#: user_id -> monotonic seconds of that user's last real LLM call.
_LAST_CALL: Dict[str, float] = {}
#: user_id -> monotonic seconds of the last warm this process issued.
_LAST_WARM: Dict[str, float] = {}
#: user_id -> (UTC date, warms issued today). Per PROCESS, so a restart
#: refills it; the cap is a blast-radius bound, not an accounting record.
_WARMS_TODAY: Dict[str, tuple] = {}
#: user_id -> monotonic seconds at which an LLM call STARTED and has not yet
#: been reported finished. A turn whose single iteration outruns the idle
#: window would otherwise look idle while it is still streaming.
_IN_FLIGHT: Dict[str, float] = {}

#: Ceiling on how long an in-flight stamp is believed. A turn that raises
#: between start and finish never clears its stamp, and a permanently
#: "in-flight" user is a feature silently switched off for the life of the
#: process — the same class of stuck flag as speakerActiveRef on the app.
_IN_FLIGHT_MAX_S = 900.0

#: In-flight warm tasks. A bare `create_task` reference can be collected
#: mid-flight, which is a warm that is billed and never observed.
_PENDING: set = set()

#: Bound on the three dicts above. An agent container is single-tenant, but a
#: pool slot rebinds and the runner is a process singleton — the same
#: unbounded-growth guard `tools_array_change` already carries.
_MAX_TRACKED = 256


def _trim(store: Dict[str, Any]) -> None:
    if len(store) > _MAX_TRACKED:
        store.clear()


def record_head(
    user_id: str,
    *,
    llm: Any,
    system_prompt: str,
    tools: Optional[List[Dict[str, Any]]],
    model: str,
    prompt_cache_key: Optional[str],
    safety_identifier: Optional[str],
    stable_prefix_active: bool,
    channel: Optional[str],
    local_date: Optional[date],
    tz_name: Optional[str],
    is_main_turn: bool = True,
) -> None:
    """Remember the head `run()` is about to send. Never raises.

    Called once per turn, beside the `[PERF] prefix_head` line — i.e. from the
    ONE place that has already finished deciding what the wire array and the
    instructions are.

    `is_main_turn` is False for anything that is not the user's own chat turn
    — a SUBAGENT run above all. A child's system prompt is the sub-agent
    prompt and its cache scope is `subagent:{job_id}` (agent_runner ~3260), so
    recording it would have the warm pay to heat a prefix the next chat turn
    will never ask for: a miss that costs money and looks like a success.
    """
    try:
        if not user_id or not is_main_turn:
            return
        _trim(_HEADS)
        _HEADS[user_id] = {
            "llm": llm,
            "system_prompt": system_prompt,
            # Copied: `run()` keeps mutating names into the executor's
            # disabled set, and a shared list would let a later turn edit
            # the bytes we promised to replay.
            "tools": list(tools or []),
            "model": model,
            "prompt_cache_key": prompt_cache_key,
            "safety_identifier": safety_identifier,
            "stable_prefix_active": bool(stable_prefix_active),
            "channel": channel,
            "local_date": local_date,
            "tz_name": tz_name,
        }
    except Exception:  # noqa: BLE001 — bookkeeping may never cost a turn
        logger.debug("[cache_warm] record_head failed", exc_info=True)


def set_cache_key(user_id: str, prompt_cache_key: Optional[str]) -> None:
    """Complete the record with the turn's routing hint.

    `run()` mints `prompt_cache_key` inside the retry loop, after the head is
    already decided. Completing the record there — rather than re-deriving the
    key in this module — keeps exactly one place that knows how the key is
    built, which is the whole reason the head is recorded and not rebuilt.
    """
    try:
        head = _HEADS.get(user_id)
        if head is not None:
            head["prompt_cache_key"] = prompt_cache_key
    except Exception:  # noqa: BLE001
        pass


def note_llm_call(user_id: str) -> None:
    """A real LLM call is STARTING. The idle gate reads this."""
    try:
        if user_id:
            _trim(_LAST_CALL)
            now = time.monotonic()
            _LAST_CALL[user_id] = now
            _IN_FLIGHT[user_id] = now
    except Exception:  # noqa: BLE001
        pass


def note_llm_done(user_id: str) -> None:
    """That call has finished. Re-stamps the idle clock from the END.

    The start stamp alone is not enough: `llm_cache_warm_idle_s` is 240 s and
    a single tool-heavy iteration can outrun it, so a socket attaching
    mid-turn would have seen an "idle" user and fired a warm concurrently
    with the very turn that is about to read the cache. Both ends, plus the
    in-flight stamp the gate refuses on outright.
    """
    try:
        if user_id:
            _LAST_CALL[user_id] = time.monotonic()
            _IN_FLIGHT.pop(user_id, None)
    except Exception:  # noqa: BLE001
        pass


def _utc_day() -> date:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date()


def _today_in(tz_name: Optional[str]) -> Optional[date]:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            return now.astimezone(ZoneInfo(tz_name)).date()
        except Exception:  # noqa: BLE001
            pass
    return now.date()


def gate(user_id: str, channel: Optional[str], *, now: Optional[float] = None) -> str:
    """"" to proceed, otherwise the reason this warm is declined.

    Pure and side-effect free so the decision is unit-testable without a
    provider; `warm_prefix` is the only thing that acts on it.
    """
    from app.config import settings

    if not getattr(settings, "llm_cache_warm_on_connect", False):
        return "flag_off"
    if not user_id:
        return "no_user"
    head = _HEADS.get(user_id)
    if not head:
        return "no_recorded_head"
    if not head.get("system_prompt"):
        return "empty_head"
    # R44 F5: the channel that matters is the one the user's last REAL turn
    # came in on, not the literal the socket handler passes. At accept time
    # the chat WS knows no channel (it arrives per message), and the browser
    # extension dials the same endpoint — so a caller-supplied label made
    # WARM_DENY_CHANNELS inert. Derived from traffic, the deny list means
    # something again: a user whose last turn was voice or WhatsApp is not
    # warmed on a socket that may not be theirs to warm.
    ch = (channel or head.get("channel") or "").strip().lower()
    if ch in WARM_DENY_CHANNELS:
        return f"denied_channel:{ch}"
    # Anthropic prompt caching is cache_control-based and priced differently;
    # a warm there is a cost with no matching mechanism.
    from app.agent.agent_runner import _is_claude_model

    model = head.get("model") or ""
    if _is_claude_model(model):
        return "not_openai"
    # R44 F1: the Responses wire is the ONLY one where the no-charge path is
    # complete end to end — the agent skips its own metering there AND
    # llm_proxy.proxy_responses honours the system marker. proxy_chat does
    # not read it, so a gpt-4o-family recorded head would be billed.
    from app.services.model_resolver import wire_api_for

    if wire_api_for(model) != "responses":
        return f"not_responses_wire:{model}"

    mono = time.monotonic() if now is None else now
    started = _IN_FLIGHT.get(user_id)
    if started is not None and (mono - started) < _IN_FLIGHT_MAX_S:
        return "turn_in_flight"
    day, used = _WARMS_TODAY.get(user_id, (None, 0))
    if day == _utc_day() and used >= int(
        getattr(settings, "llm_cache_warm_max_per_day", 0) or 0
    ):
        return f"daily_cap:{used}"
    last_call = _LAST_CALL.get(user_id)
    idle_s = float(getattr(settings, "llm_cache_warm_idle_s", 240) or 0)
    if last_call is not None and (mono - last_call) < idle_s:
        # Still warm on its own — a warm here would pay for a hit we already
        # have, which is the one way this feature can cost more than it saves.
        return f"not_idle:{int(mono - last_call)}s"
    last_warm = _LAST_WARM.get(user_id)
    gap_s = float(getattr(settings, "llm_cache_warm_min_gap_s", 300) or 0)
    if last_warm is not None and (mono - last_warm) < gap_s:
        return f"min_gap:{int(mono - last_warm)}s"
    return ""


async def warm_prefix(user_id: str, channel: Optional[str] = None) -> None:
    """Issue ONE minimal Responses call that carries the recorded head.

    Awaited body of `schedule_warm`. Writes nothing: no messages, no
    `_last_media`, no `turns_completed`. Credit exhaustion is enforced by
    `create_message_stream` itself (`raise_if_exhausted_async`), which is the
    same gate the turn path uses — so there is one check, not two that drift.
    """
    reason = gate(user_id, channel)
    if reason:
        logger.debug("[PERF] cache_warm skipped reason=%s user=%s",
                     reason, (user_id or "")[:8])
        return

    head = _HEADS.get(user_id) or {}
    _LAST_WARM[user_id] = time.monotonic()
    _trim(_WARMS_TODAY)
    _day, _used = _WARMS_TODAY.get(user_id, (None, 0))
    _WARMS_TODAY[user_id] = (
        _utc_day(), (_used + 1) if _day == _utc_day() else 1,
    )
    t0 = time.perf_counter()
    cached = inp = 0
    try:
        llm = head["llm"]
        async for event in llm.create_message_stream(
            # ONE minimal input item. It is the only part of the request that
            # is not the head, and it sits AFTER it, so it cannot disturb the
            # prefix the next turn will read.
            messages=[{"role": "user", "content": "."}],
            system=head["system_prompt"],
            tools=head["tools"] or None,
            model=head["model"],
            max_tokens=16,
            tool_choice="none",
            prompt_cache_key=head.get("prompt_cache_key"),
            safety_identifier=head.get("safety_identifier"),
            # No idempotency_key: that key is the BILLING dedupe for a turn,
            # and handing the warm the turn's key would make one of them
            # vanish from metering.
            stable_prefix_active=bool(head.get("stable_prefix_active")),
            # R44 F5: the wire channel is "system", not the socket's. This is
            # what `llm_proxy_events.channel` records, and a warm filed as
            # "app" makes /cache/daily count one turn twice — the warm's miss
            # and the turn's hit — which would read as the feature failing.
            channel="system",
            # R44 F1: NOT the user's. Skips the agent's own metering outright
            # and tells the proxy to file a non-user-attributable row.
            operation_type=SYSTEM_OPERATION,
        ):
            if event.type == "message_end":
                cached = int(event.usage.get("cache_read_input_tokens", 0) or 0)
                inp = int(event.usage.get("input_tokens", 0) or 0)
    except Exception as err:  # noqa: BLE001 — a warm may never surface to a user
        logger.info(
            "[PERF] cache_warm failed channel=%s user=%s ms=%d err=%s: %s",
            channel, (user_id or "")[:8],
            int((time.perf_counter() - t0) * 1000),
            type(err).__name__, err,
        )
        return

    logger.info(
        "[PERF] cache_warm channel=%s user=%s cached=%d input=%d ms=%d",
        channel, (user_id or "")[:8], cached, inp,
        int((time.perf_counter() - t0) * 1000),
    )
    try:
        from app.services import health_signals as _hs

        _hs.incr("llm_cache_warms")
    except Exception:  # noqa: BLE001
        pass


def schedule_warm(user_id: str, channel: Optional[str] = None) -> None:
    """Fire-and-forget entry point — the ONE thing a WS accept path calls.

    Returns before any I/O happens, so it can never delay a connection. Also
    never raises: a socket must not fail because a cache optimisation did.
    """
    try:
        import asyncio

        from app.config import settings

        if not getattr(settings, "llm_cache_warm_on_connect", False):
            return
        # Cheap pre-check so an ineligible connect does not even cost a task.
        if gate(user_id, channel):
            return
        task = asyncio.get_running_loop().create_task(warm_prefix(user_id, channel))
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)
    except Exception:  # noqa: BLE001
        logger.debug("[cache_warm] schedule failed", exc_info=True)
