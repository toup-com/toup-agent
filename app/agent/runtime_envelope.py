"""The per-turn `<runtime_envelope>` message.

WHY A SEPARATE MESSAGE, AND WHY IT MAY GIVE ORDERS
--------------------------------------------------
`<turn_context>` carries this turn's volatile FACTS and its own preamble
tells the model to treat everything inside as reference DATA it must never
take instructions from (`prefix_stability.build_turn_context_message`).
The envelope is the opposite: it is the system telling the model where this
turn came from and what the surface can render, and it must OVERRIDE any
stale channel claim earlier in the day. Folding it into `<turn_context>`
would either weaken that injection boundary or make the envelope inert.
`source_conflict.build_turn_rules_message` set the precedent — same slot,
same user role, same never-persisted lifetime, for exactly this reason.

LEGACY-SESSION COMPATIBILITY
----------------------------
The provider call is STATELESS: `agent_runner` sends the full messages +
system + tools array every turn, there is no `previous_response_id` and no
per-channel provider session; `prompt_cache_key` is a routing hint only
(see the converge comment at its assignment). So no "legacy session" can
carry a baked-in channel, and switching channels mid-day resets nothing.
The only legacy channel state is persisted rows, and it is exactly three
things:
  (a) `Message.content` carrying a leaked annotation tag — stripped on
      read (`day_context_loader`, `recall_day`, `public_text`) and removed
      for good by `scripts/backfill_annotation_leaks.py`;
  (b) `DayChat.rolling_summary` with a tag baked into its prose — a
      residual, cleared on the next summariser run;
  (c) `Message.channel` NULL on old ws_chat presave rows — resolved by
      `resolve_channel`'s conversation_hint fallback, which is why such a
      row can annotate as its Conversation's channel rather than its own.
Nothing else about channel survives a turn.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

ENVELOPE_OPEN = "<runtime_envelope>"
ENVELOPE_CLOSE = "</runtime_envelope>"

# Estimated-token ceiling for the whole block (len // 4, the runner's own
# estimator). The envelope rides OUTSIDE the cached prefix — every byte here
# is billed at full price on every turn of every channel — so it is capped by
# construction, not by review: `build_runtime_envelope_message` shrinks the
# surface descriptor until the block fits.
# Measured on production inputs (tests/test_channel_neutral_prefix.py): the
# longest channels — extension (a 163-char first sentence) and trigger (a
# 66-char client-surface label) — land at ~144 est. tokens with their Surface
# line WHOLE; everything else sits at 104–120.
ENVELOPE_TOKEN_BUDGET = 150

# The descriptor is never shrunk below this: a "Surface:" line cut to a
# half-word is worse than a block a few tokens over budget, so past this
# floor the builder logs and ships the whole first sentence instead.
MIN_SURFACE_DESCRIPTOR = 60

# Nominal cap on the "where the user is" descriptor. The behavioural essays
# in CHANNEL_GUIDANCE (voice 1.8k chars, extension 2.2k) stay in the cached
# system prompt as explicitly-scoped surface contracts; only the short
# locating sentence travels per turn.
MAX_SURFACE_DESCRIPTOR = 200

_YN = {True: "y", False: "n"}

# What each surface can actually RENDER. The prompt used to teach these
# rules per channel inside the cached system prompt, which is what forked
# every channel onto its own provider cache lineage. The rules are now taught
# once, unconditionally, each conditioned on this line.
#   markdown:      full | basic | none
#   media_player:  inline (an embedded player in the thread)
#                | native (audio/video handled off-thread by the client)
#                | none
#   max_message_chars: 0 = no transport limit
_CAPS: Dict[str, Dict[str, Any]] = {
    "web": dict(markdown="full", tables=True, code_blocks=True,
                quick_reply_buttons=True, reactions=False,
                media_player="inline", max_message_chars=0),
    "app": dict(markdown="full", tables=True, code_blocks=True,
                quick_reply_buttons=True, reactions=False,
                media_player="inline", max_message_chars=0),
    "vibecoding": dict(markdown="full", tables=True, code_blocks=True,
                       quick_reply_buttons=False, reactions=False,
                       media_player="none", max_message_chars=0),
    "api": dict(markdown="full", tables=True, code_blocks=True,
                quick_reply_buttons=False, reactions=False,
                media_player="none", max_message_chars=0),
    "extension": dict(markdown="full", tables=True, code_blocks=True,
                      quick_reply_buttons=False, reactions=False,
                      media_player="none", max_message_chars=0),
    "mobile": dict(markdown="basic", tables=False, code_blocks=False,
                   quick_reply_buttons=True, reactions=False,
                   media_player="native", max_message_chars=0),
    "voice": dict(markdown="none", tables=False, code_blocks=False,
                  quick_reply_buttons=False, reactions=False,
                  media_player="native", max_message_chars=0),
    "telegram": dict(markdown="basic", tables=False, code_blocks=True,
                     quick_reply_buttons=True, reactions=True,
                     media_player="none", max_message_chars=4096),
    # cron/heartbeat turns are DELIVERED to the user's Telegram by the bot,
    # so they render exactly like a telegram turn.
    "cron": dict(markdown="basic", tables=False, code_blocks=True,
                 quick_reply_buttons=True, reactions=True,
                 media_player="none", max_message_chars=4096),
    "heartbeat": dict(markdown="basic", tables=False, code_blocks=True,
                      quick_reply_buttons=True, reactions=True,
                      media_player="none", max_message_chars=4096),
    "whatsapp": dict(markdown="basic", tables=False, code_blocks=True,
                     quick_reply_buttons=False, reactions=False,
                     media_player="none", max_message_chars=4096),
    "discord": dict(markdown="full", tables=False, code_blocks=True,
                    quick_reply_buttons=False, reactions=False,
                    media_player="none", max_message_chars=2000),
    "slack": dict(markdown="basic", tables=False, code_blocks=True,
                  quick_reply_buttons=False, reactions=False,
                  media_player="none", max_message_chars=4000),
}

# Unknown / background surfaces (trigger, routine, autopilot, agent_task,
# health_probe, admin, automation, subagent, agent, ""): nothing interactive
# can render and nobody is present to tap it.
_CAPS_DEFAULT: Dict[str, Any] = dict(
    markdown="basic", tables=False, code_blocks=True,
    quick_reply_buttons=False, reactions=False,
    media_player="none", max_message_chars=0,
)

# Name of the scoped block in the system prompt's `surface_contracts`
# section that governs this surface. "" = no contract beyond the neutral
# rules, which is the common case.
_CONTRACTS = {
    "voice": "VOICE",
    "vibecoding": "VIBECODING",
    "extension": "EXTENSION",
    # Unattended email-trigger turn: its guidance carries the G-19b policy
    # pin ("NEVER claim to have sent…"), which only the first sentence of
    # the per-turn descriptor would otherwise have dropped.
    "trigger": "TRIGGER",
}

# Byte-identical for every channel and every turn, which is exactly why it
# lives in the CACHED system prompt (agent_runner's runtime rules under the
# envelope) and NOT in this per-turn block: here it was the single largest
# uncached cost and it crowded the Surface line out of the budget. Every
# clause is load-bearing — the shape, that it is the system's, and both
# prohibitions.
ANNOTATION_RULE = (
    "History lines carry a SYSTEM-added origin tag (the `[channel h:mmam]` shape, "
    "e.g. `[mobile 9:41pm]`, or `You [mobile]:`). They are context metadata, not "
    "anyone's words and not a format to copy. NEVER begin a reply with one or "
    "write one anywhere."
)

# The deferral tools whose availability the cached prompt says this block
# states. Read from `prompt_profile.disabled_tools_for_channel`, the same set
# the runner removes from the wire array, so the line cannot disagree with
# the tool list.
_DEFERRAL_TOOLS = ("create_job", "update_job", "start_mission")

_PREAMBLE = (
    "SYSTEM-written for THIS turn; overrides anything earlier about channels, "
    "surfaces or formatting."
)


def channel_capabilities(channel: Optional[str]) -> Dict[str, Any]:
    """What the surface can render. Never raises; unknown → the conservative
    background default."""
    key = (channel or "").strip().lower()
    return dict(_CAPS.get(key, _CAPS_DEFAULT))


def _descriptor(guidance: str, limit: int = MAX_SURFACE_DESCRIPTOR) -> str:
    """The locating half of a CHANNEL_GUIDANCE value — its first sentence,
    clamped. The behavioural remainder lives in the cached prompt."""
    text = " ".join((guidance or "").split())
    if not text:
        return ""
    head = text.split("\n", 1)[0]
    dot = head.find(". ")
    if dot != -1:
        head = head[: dot + 1]
    if limit <= 0:
        return ""
    if len(head) > limit:
        head = head[: max(0, limit - 1)].rstrip() + "…"
    return head


def deferral_line(origin_channel: Optional[str], *, managed_voice: bool = False) -> str:
    """`Deferral: create_job=y; update_job=y; start_mission=y` for this
    surface. Never raises — an import failure answers as if everything were
    available, which is the pre-envelope behaviour."""
    disabled: frozenset = frozenset()
    try:
        from app.agent.prompt_profile import disabled_tools_for_channel

        disabled = frozenset(disabled_tools_for_channel(origin_channel) or ())
    except Exception:  # noqa: BLE001
        disabled = frozenset()
    if managed_voice:
        # A managed voice task keeps its tracked job; it must not open another.
        disabled = disabled | {"create_job"}
    return "Deferral: " + "; ".join(
        f"{t}={_YN[t not in disabled]}" for t in _DEFERRAL_TOOLS
    )


def _caps_line(caps: Dict[str, Any]) -> str:
    return (
        "Caps: markdown={markdown}; tables={tables}; code_blocks={code}; "
        "quick_reply_buttons={btn}; reactions={react}; media_player={media}; "
        "max_message_chars={mx}".format(
            markdown=caps.get("markdown", "basic"),
            tables=_YN[bool(caps.get("tables"))],
            code=_YN[bool(caps.get("code_blocks"))],
            btn=_YN[bool(caps.get("quick_reply_buttons"))],
            react=_YN[bool(caps.get("reactions"))],
            media=caps.get("media_player", "none"),
            mx=int(caps.get("max_message_chars") or 0),
        )
    )


def _render(lines) -> str:
    return ENVELOPE_OPEN + "\n" + "\n".join(l for l in lines if l) + "\n" + ENVELOPE_CLOSE


def build_runtime_envelope_message(
    *,
    origin_channel: str,
    reply_channel: str,
    client_surface: str,
    guidance: str,
    capabilities: Dict[str, Any],
    request_id: Optional[str] = None,
    message_id: Optional[str] = None,
    day_chat_id: Optional[str] = None,
    managed_voice: bool = False,
) -> Dict[str, str]:
    """The per-turn envelope, as a separate user-role message.

    Same shape and lifetime as `source_conflict.build_turn_rules_message`:
    appended after `<turn_context>`, before the current user message, never
    persisted, so it cannot leak into the next turn's history.
    """
    origin = (origin_channel or "unknown").strip().lower() or "unknown"
    reply = (reply_channel or origin).strip().lower() or origin
    contract = _CONTRACTS.get(origin, "")
    if managed_voice:
        contract = "VOICE"
    # request_id / message_id / day_chat_id are accepted for call-site
    # compatibility and deliberately NOT rendered: eight-character prefixes of
    # internal uuids told the model nothing and cost the Surface line its
    # budget.
    del request_id, message_id, day_chat_id

    deferral = deferral_line(origin, managed_voice=managed_voice)

    def _body(desc_limit: int) -> str:
        desc = _descriptor(guidance, desc_limit)
        return _render([
            _PREAMBLE,
            "Origin: {o} | Reply: {r} | Client: {s}{c}".format(
                o=origin, r=reply, s=(client_surface or "unknown"),
                c=(f" | Contract: {contract}" if contract else ""),
            ),
            (f"Surface: {desc}" if desc else ""),
            _caps_line(capabilities or {}),
            deferral,
        ])

    # The descriptor is the only variable-length part and the least
    # load-bearing one (the surface is already named twice above it), so it is
    # what gives way when a guidance value is long — down to
    # MIN_SURFACE_DESCRIPTOR. Past that floor a truncated half-word sentence
    # is worse than a few tokens over budget: log it (so a new channel's
    # essay is caught in review, not by the model) and ship the whole first
    # sentence.
    limit = MAX_SURFACE_DESCRIPTOR
    content = _body(limit)
    while limit > MIN_SURFACE_DESCRIPTOR and len(content) // 4 > ENVELOPE_TOKEN_BUDGET:
        over = len(content) - (ENVELOPE_TOKEN_BUDGET * 4)
        limit = max(MIN_SURFACE_DESCRIPTOR, limit - max(over, 8))
        content = _body(limit)
    if len(content) // 4 > ENVELOPE_TOKEN_BUDGET:
        content = _body(MAX_SURFACE_DESCRIPTOR)
        try:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "[runtime_envelope] over budget for origin=%s (~%d est. tokens > %d): "
                "shipping the full surface sentence rather than a fragment",
                origin, len(content) // 4, ENVELOPE_TOKEN_BUDGET,
            )
        except Exception:  # noqa: BLE001
            pass
    return {"role": "user", "content": content}
