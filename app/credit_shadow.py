"""Shadow mode for the credit admission gate: decide, log, deny nothing.

WHY THIS MODULE EXISTS
----------------------
``credit_cap_admission_control`` and ``credit_enforcement_enabled`` are both
default-False, and #471 closed the three ways the gate would have misbehaved
the moment they were switched on. What nobody can answer is the question that
actually decides whether to switch them on: **what would enforcement deny?**

The only number on record — "274 calls, $17.17, 59% of all real-user LLM cost
served free with reason=daily_cap_exceeded" (2026-08-03) — was measured against
the BROKEN gate, the one that zeroed costs already incurred. It says nothing
about what the fixed gate would refuse, because the fixed gate refuses at
ADMISSION and exempts an incurred cost from the cap entirely.

Shadow mode answers it without flipping anything: on every MESSAGE-bucket
charge, ask the real gate what it would have said, write the answer to the log,
and throw the answer away. Nothing is denied, no balance moves, no user sees a
difference. ``backend/scripts/credit_shadow_rollup.py`` turns the resulting
lines into the two numbers that matter.

WHAT IS AND IS NOT IN A LINE
----------------------------
User ids only. No email, no API key, no message content, no model prompt.

Every free-form field is ALLOW-LISTED BY SHAPE and REJECTED outright when it
does not fit — replaced by the constant ``rejected``, not shortened. Scrubbing
``victim@example.com`` leaves ``victim_example.com`` and truncating an API key
leaves most of an API key: truncation is not redaction (2026-08-03, #419). The
shapes admit a UUID, a snake_case ledger event type and an ISO date, and admit
nothing containing ``@``, a space, a dot or an uppercase letter.

ONE COPY OF THE SCHEMA
----------------------
:func:`format_line` and :func:`parse_line` live in the same file on purpose.
A rollup that reimplements the field names is a rollup that can read
``usage["cache_read"]`` forever and print a plausible, uniform, fictional zero.
``tests/test_credit_shadow_mode.py`` round-trips real emitter output through
the real parser so the two cannot drift apart.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

logger = logging.getLogger(__name__)

# Stable, greppable prefix — same shape as the existing `[CACHE]` / `[PERF]`
# lines. `grep -F '[CREDIT-SHADOW]'` is the whole ingestion story.
SHADOW_PREFIX = "[CREDIT-SHADOW]"

# Bump when a field changes meaning. `parse_line` REFUSES an unknown version
# rather than reading a field that no longer means what it used to.
SHADOW_SCHEMA_VERSION = 1

# What the live pre-flight actually asks for. `llm_proxy.proxy_chat`,
# `llm_proxy.proxy_responses`, `ws_realtime` and `/credits/agent-deduct` all
# call `check_balance(..., BUCKET_MESSAGE, Decimal("0.1"))` — a nominal probe,
# not the turn's real cost. The shadow verdict must ask the same question the
# production gate would ask, so it uses the same quote. The turn's REAL cost is
# carried separately in `amount=`, which is what makes "share of spend"
# computable at all.
PREFLIGHT_QUOTE_CREDITS = Decimal("0.1")

DECISION_ALLOW = "allow"
DECISION_DENY = "deny"

EVENT_ADMISSION = "admission"

# Values are emitted with 4 decimal places; `_AMOUNT_QUANTUM` in credit_service
# is the same precision.
_DP = 4

_LINE_RE = re.compile(re.escape(SHADOW_PREFIX) + r"\s+(?P<fields>.*)$")
_FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>\S+)")

# Free-form values are ALLOW-LISTED BY SHAPE and REJECTED when they do not fit,
# never scrubbed into something shorter. Scrubbing an email leaves
# `victim_example.com`, and truncating a key leaves the first 40 characters of
# a key: truncation is not redaction. A value that does not match its shape is
# replaced wholesale by a constant.
#
# `_ID_SHAPE` admits a UUID and nothing with an `@`, a space, or a `/`.
# `_TOKEN_SHAPE` admits the snake_case ledger event types and denial reasons
# ("chat_message", "image_generation", "daily_cap_exceeded") and rejects
# anything with an uppercase letter, a dot, a dash or a space — i.e. every
# email, every `sk-…` key, every JWT, and any sentence of user content.
_ID_SHAPE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")
_TOKEN_SHAPE = re.compile(r"\A[a-z0-9_]{1,32}\Z")
_DAY_SHAPE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")

REJECTED = "rejected"          # the value did not match its allowed shape

NUMERIC_FIELDS = (
    "amount", "quote", "used_today", "plan_remaining", "purchased_remaining",
)
# `cap` is numeric OR the literal "none" (no daily cap configured).
OPTIONAL_NUMERIC_FIELDS = ("cap",)
REQUIRED_FIELDS = (
    "v", "event", "user", "decision", "reason", "amount", "quote", "used_today",
    "cap", "plan_remaining", "purchased_remaining", "day", "type", "unlimited",
    "enforcement", "cap_admission",
)


def _shaped(value: object, shape: re.Pattern[str], *, empty: str = "-") -> str:
    """Return ``value`` if it matches ``shape``, else the constant ``REJECTED``.

    Reject, never repair. A repaired value is still the original secret minus a
    character or two; a rejected one is the word "rejected".
    """
    text = str(value).strip() if value is not None else ""
    if not text:
        return empty
    return text if shape.fullmatch(text) else REJECTED


def _num(value: Decimal | float | int) -> str:
    return f"{Decimal(str(value)):.{_DP}f}"


@dataclass(frozen=True)
class ShadowAdmission:
    """One observed MESSAGE-bucket charge and the verdict the gate WOULD give.

    ``decision``/``reason`` are counterfactual — what a fully-enabled gate
    (``credit_enforcement_enabled`` AND ``credit_cap_admission_control``) would
    have answered for this turn's pre-flight, judged on the state as it stood
    immediately before the charge landed. ``amount`` is not counterfactual: it
    is the credits this turn really cost.
    """

    user_id: str
    decision: str
    reason: Optional[str]
    amount: Decimal
    quote: Decimal
    used_today: Decimal
    cap: Optional[Decimal]
    plan_remaining: Decimal
    purchased_remaining: Decimal
    day: str
    event_type: str
    unlimited: bool
    # The LIVE flag values at the moment of observation. Without these a reader
    # cannot tell whether a line describes a counterfactual (flags off — the
    # spend really happened) or live behaviour (flags on — it did not).
    enforcement_enabled: bool
    cap_admission_control: bool


def format_line(obs: ShadowAdmission) -> str:
    """Render one observation as a single greppable log line."""
    fields = [
        f"v={SHADOW_SCHEMA_VERSION}",
        f"event={EVENT_ADMISSION}",
        f"user={_shaped(obs.user_id, _ID_SHAPE)}",
        f"decision={obs.decision if obs.decision in (DECISION_ALLOW, DECISION_DENY) else REJECTED}",
        f"reason={_shaped(obs.reason, _TOKEN_SHAPE)}",
        f"amount={_num(obs.amount)}",
        f"quote={_num(obs.quote)}",
        f"used_today={_num(obs.used_today)}",
        f"cap={_num(obs.cap) if obs.cap is not None else 'none'}",
        f"plan_remaining={_num(obs.plan_remaining)}",
        f"purchased_remaining={_num(obs.purchased_remaining)}",
        f"day={_shaped(obs.day, _DAY_SHAPE)}",
        f"type={_shaped(obs.event_type, _TOKEN_SHAPE)}",
        f"unlimited={int(bool(obs.unlimited))}",
        f"enforcement={int(bool(obs.enforcement_enabled))}",
        f"cap_admission={int(bool(obs.cap_admission_control))}",
    ]
    return f"{SHADOW_PREFIX} " + " ".join(fields)


def emit(obs: ShadowAdmission) -> None:
    """Write one observation to the log. This is the entire side effect of
    shadow mode — there is deliberately no return value to act on."""
    logger.info("%s", format_line(obs))


def parse_line(line: str) -> Optional[dict]:
    """Inverse of :func:`format_line`.

    Returns a dict with Decimals for the numeric fields, or ``None`` when the
    line is not a shadow line, carries an unknown schema version, or is missing
    a field the rollup depends on. Returning ``None`` (rather than a partial
    dict) is the point: a rollup that silently skips a malformed line prints a
    smaller number, which is visible; one that reads a missing field as zero
    prints a wrong number, which is not.
    """
    match = _LINE_RE.search(line)
    if match is None:
        return None
    fields = {m.group("key"): m.group("value")
              for m in _FIELD_RE.finditer(match.group("fields"))}
    if fields.get("v") != str(SHADOW_SCHEMA_VERSION):
        return None
    if fields.get("event") != EVENT_ADMISSION:
        return None
    if any(name not in fields for name in REQUIRED_FIELDS):
        return None
    out: dict = dict(fields)
    try:
        for name in NUMERIC_FIELDS:
            out[name] = Decimal(fields[name])
        for name in OPTIONAL_NUMERIC_FIELDS:
            out[name] = None if fields[name] == "none" else Decimal(fields[name])
    except (InvalidOperation, ValueError):
        return None
    if out["decision"] not in (DECISION_ALLOW, DECISION_DENY):
        return None
    for name in ("unlimited", "enforcement", "cap_admission"):
        if fields[name] not in ("0", "1"):
            return None
        out[name] = fields[name] == "1"
    if out["reason"] == "-":
        out["reason"] = None
    return out
