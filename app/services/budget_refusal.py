"""The monthly model-budget refusal, in one vocabulary for platform and agent.

The platform LLM proxy refuses a call with HTTP 429 once a tenant's monthly
provider budget (``AgentConfig.bundle_*_budget_cents``) is spent. Before
2026-09-29 that refusal was a bare ``{"detail": "Monthly openai budget
exceeded"}`` and every consumer classified it by substring — the analysis job
called it "service access", the chat turn treated it as a transient rate
limit (and then crashed on ``emitted_any``), and job cards printed the raw
exception. This module is the single place that

* names the refusal (``REASON`` — the ``X-Toup-Reason`` header value and the
  ``detail.error`` value the proxy sends),
* recognises it on a 429 the SDKs hand us, or on the stored text of one
  (``is_budget_refusal``) — never on arbitrary text that merely quotes it,
* lifts the structured detail back out of an exception or its ``str()``
  (``budget_refusal_detail``), and
* renders every user-facing sentence about it (``chat_sentence``,
  ``job_sentence``, ``push_body``, ``reset_when_phrase``).

Stdlib only, on purpose: it is imported by the platform image (llm_proxy,
ws_chat, job runners) and by the agent image (attachment_analysis,
openai_agent_service, agent_runner).

Time contract: the proxy's DateTime columns are naive UTC. Every helper here
treats a naive value or an ISO string without an offset as UTC, returns aware
UTC datetimes from ``parse_utc``, and serialises with an explicit ``+00:00``
and no microseconds (``iso_utc``). Compare aware to aware only.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: The typed reason. Header value AND ``detail.error`` value on the proxy 429.
REASON = "monthly_model_budget_exceeded"
#: The header the proxy sets on every budget refusal (case-insensitive on the wire).
REASON_HEADER = "X-Toup-Reason"
#: ``error_class`` written on a job that a budget refusal stopped. Clients
#: render ``user_message`` verbatim whenever ``error_class`` is set.
ERROR_CLASS = "model_budget"
#: The chat ``error`` frame code (``{"type": "error", "code": ...}``).
FRAME_CODE = REASON

_LEGACY_RE = re.compile(r"monthly (openai|anthropic) budget exceeded", re.I)
_MARKER_RE = re.compile(re.escape(REASON), re.I)
#: The prefix the openai and anthropic SDKs give every status error's message
#: (``f"Error code: {status} - {body}"``). ``repr(exc)`` and the job rows that
#: store ``str(e)``/``repr(e)`` keep it. Text is only trusted to describe the
#: refusal when it carries this (or the exception itself says 429): anything
#: else — a summariser error quoting an email, a customer's own words — is
#: third-party text, and must never pass for a budget stop or name its reset.
_SDK_429_RE = re.compile(r"\bError code: 429\b", re.I)
#: The providers the proxy keeps a monthly budget for — the only values its
#: refusal's ``detail.provider`` can carry.
_PROXY_PROVIDERS = frozenset({"openai", "anthropic"})
_DIGITS_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December")
_MONTHS_FA = ("ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت",
              "سپتامبر", "اکتبر", "نوامبر", "دسامبر")
_SOON = timedelta(hours=48)


# ── time helpers ─────────────────────────────────────────────────────


def parse_utc(value: Any) -> Optional[datetime]:
    """An aware UTC datetime, or None. Naive input and offset-less ISO text
    are UTC. Never raises."""
    try:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z") or text.endswith("z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        else:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001 — a malformed timestamp must never raise
        return None


def to_naive_utc(value: Any) -> Optional[datetime]:
    """The naive-UTC form the proxy's DateTime columns and asyncpg expect."""
    dt = parse_utc(value)
    return dt.replace(tzinfo=None) if dt is not None else None


def iso_utc(value: Any) -> Optional[str]:
    """``2026-10-24T19:10:40+00:00`` — explicit offset, no microseconds, so a
    substring classifier (``'401' in str(e)``, ``\\b5\\d\\d\\b``) never trips
    on a microsecond field."""
    dt = parse_utc(value)
    return dt.replace(microsecond=0).isoformat() if dt is not None else None


def _now_utc(now: Any = None) -> datetime:
    return parse_utc(now) or datetime.now(timezone.utc)


# ── recognition ──────────────────────────────────────────────────────


def _header_value(headers: Any) -> Optional[str]:
    if headers is None:
        return None
    try:
        value = headers.get(REASON_HEADER)
        if value is None:
            value = headers.get(REASON_HEADER.lower())
        if value is None and hasattr(headers, "items"):
            for key, val in headers.items():
                if str(key).lower() == REASON_HEADER.lower():
                    value = val
                    break
        return str(value) if value is not None else None
    except Exception:  # noqa: BLE001
        return None


def _detail_of(body: Any) -> Optional[dict]:
    """The proxy's ``detail`` dict when ``body`` carries the typed refusal."""
    if not isinstance(body, dict):
        return None
    detail = body.get("detail", body)
    if isinstance(detail, dict) and detail.get("error") == REASON:
        return detail
    return None


def _proxy_shaped(parsed: Any) -> Optional[dict]:
    """``_detail_of`` for a dict lifted out of TEXT, which must also have the
    proxy's shape — a ``provider`` the proxy budgets — so a dict quoted inside
    someone else's message is not taken for the refusal (or its reset date)."""
    found = _detail_of(parsed)
    provider = found.get("provider") if found is not None else None
    if isinstance(provider, str) and provider in _PROXY_PROVIDERS:
        return found
    return None


def _says_429(obj: Any) -> bool:
    """The exception itself carries HTTP 429 (the SDKs' ``APIStatusError``,
    Starlette's ``HTTPException``)."""
    return getattr(obj, "status_code", None) == 429


def _trusted_text(obj: Any) -> Optional[str]:
    """``str(obj)`` when it may be read for the refusal, else None.

    An exception's text is trusted when the exception says 429 or the text
    carries the SDKs' ``Error code: 429`` prefix; any other object (a ``str``
    of stored error text) only with that prefix. Everything else is text we
    did not produce."""
    text = str(obj)
    if isinstance(obj, BaseException) and _says_429(obj):
        return text
    return text if _SDK_429_RE.search(text) else None


def is_budget_refusal(obj: Any) -> bool:
    """True for the proxy's monthly budget refusal, however it reaches us:
    the typed header, the typed body, or — on a 429 only (the exception's
    ``status_code``, or the SDKs' ``Error code: 429`` prefix in its text or
    in stored error text) — the legacy sentence or the reason marker. Never
    raises. False for ordinary rate limits, the G-20 per-minute limiter,
    credit (402) refusals, upstream ``insufficient_quota``, and third-party
    text that merely quotes the marker or the sentence."""
    try:
        if obj is None:
            return False
        if isinstance(obj, BaseException):
            response = getattr(obj, "response", None)
            header = _header_value(getattr(response, "headers", None))
            if header is not None and header.strip().lower() == REASON:
                return True
            if _detail_of(getattr(obj, "body", None)) is not None:
                return True
        elif isinstance(obj, dict):
            return _detail_of(obj) is not None
        text = _trusted_text(obj)
        if text is None:
            return False
        return bool(_MARKER_RE.search(text) or _LEGACY_RE.search(text))
    except Exception:  # noqa: BLE001
        return False


def _scan_text(text: str) -> Optional[dict]:
    """Lift the detail dict out of ``str(exc)`` (a python repr, single quotes)
    or JSON. Same brace-walk as ``ws_chat._extract_out_of_credits_detail``.
    Only a proxy-shaped dict counts (``_proxy_shaped``); callers pass only
    text ``_trusted_text`` accepted."""
    idx = text.find(REASON)
    if idx == -1:
        return None
    start = text.rfind("{", 0, idx)
    while start != -1:
        depth = 0
        for end in range(start, min(len(text), start + 4000)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start:end + 1]
                    for loads in (json.loads, ast.literal_eval):
                        try:
                            parsed = loads(blob)
                        except Exception:  # noqa: BLE001
                            continue
                        found = _proxy_shaped(parsed)
                        if found is not None:
                            return found
                    break
        start = text.rfind("{", 0, start)
    return None


def budget_refusal_detail(obj: Any) -> Optional[dict]:
    """The proxy's structured detail (``error, message, provider,
    period_start, period_end``) or None when only the legacy sentence is
    available — or when the only copy is text we do not trust (see
    ``is_budget_refusal``). Never raises."""
    try:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return _detail_of(obj)
        if isinstance(obj, BaseException):
            found = _detail_of(getattr(obj, "body", None))
            if found is not None:
                return found
            response = getattr(obj, "response", None)
            if response is not None:
                try:
                    found = _detail_of(response.json())
                except Exception:  # noqa: BLE001
                    found = None
                if found is not None:
                    return found
        text = _trusted_text(obj)
        return _scan_text(text) if text is not None else None
    except Exception:  # noqa: BLE001
        return None


def resets_at(detail: Optional[dict], now: Any = None) -> Optional[datetime]:
    """Aware UTC ``period_end`` from a detail dict, or None when absent or
    already past (a past reset is never promised)."""
    if not isinstance(detail, dict):
        return None
    end = parse_utc(detail.get("period_end"))
    if end is None or end <= _now_utc(now):
        return None
    return end


# ── rendering ────────────────────────────────────────────────────────


def _zone(tz_name: Optional[str]) -> tuple[tzinfo, bool]:
    """(zone, fell_back_to_utc)."""
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(str(tz_name)), False
        except Exception:  # noqa: BLE001
            pass
    return timezone.utc, True


def reset_when_phrase(period_end: Any, tz_name: Optional[str] = None,
                      now: Any = None, lang: str = "en") -> Optional[str]:
    """An ABSOLUTE date in the user's zone — ``October 24``, ``October 24 at
    3:10 PM`` inside 48 h, ``January 3, 2027`` across a year boundary. Never
    relative ("in 26 days" is false the next day inside a saved message).
    When no zone is known the date is UTC, marked " UTC" only inside 48 h;
    beyond that a bare UTC date can be a day off for a user near midnight
    UTC (every web/app frame and warm channel turn carries a zone).
    None when there is no future reset to name."""
    end = parse_utc(period_end)
    current = _now_utc(now)
    if end is None or end <= current:
        return None
    zone, fell_back = _zone(tz_name)
    local = end.astimezone(zone)
    local_now = current.astimezone(zone)
    soon = (end - current) < _SOON
    if lang == "fa":
        text = f"{str(local.day).translate(_DIGITS_FA)} {_MONTHS_FA[local.month - 1]}"
        if local.year != local_now.year:
            text += f" {str(local.year).translate(_DIGITS_FA)}"
        if soon:
            text += f" ساعت {local.strftime('%H:%M').translate(_DIGITS_FA)}"
            if fell_back:
                text += " UTC"
        return text
    text = f"{_MONTHS_EN[local.month - 1]} {local.day}"
    if local.year != local_now.year:
        text += f", {local.year}"
    if soon:
        hour = local.hour % 12 or 12
        text += f" at {hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"
        if fell_back:
            text += " UTC"
    return text


def chat_sentence(detail: Optional[dict], tz_name: Optional[str] = None,
                  now: Any = None) -> str:
    """The chat ``error`` bubble. No 'rate limit', no 'out of Toup credits'
    (the paywall trigger phrase), no cents, no timestamps."""
    when = reset_when_phrase((detail or {}).get("period_end"), tz_name, now)
    if when:
        return (
            "Your agent’s monthly AI budget is used up, so it can’t reply until "
            f"the budget resets on {when}. Your credits aren’t affected."
        )
    return (
        "Your agent’s AI budget is used up, so it can’t reply right now. "
        "Your credits aren’t affected."
    )


def job_sentence(detail: Optional[dict], tz_name: Optional[str] = None,
                 now: Any = None) -> str:
    """``user_message`` for a job/build/mission the refusal stopped. "couldn’t
    finish", not "couldn’t run": the refused job has often done work first.
    The undated sentence is also ``job_status``'s static ``model_budget``
    copy and the web/app fallback (pinned against this function)."""
    when = reset_when_phrase((detail or {}).get("period_end"), tz_name, now)
    if when:
        return (
            "This task couldn’t finish because your agent’s monthly AI budget "
            f"is used up. It resets on {when}. Your credits aren’t affected."
        )
    return (
        "This task couldn’t finish because your agent’s AI budget is used up. "
        "Your credits aren’t affected."
    )


def push_body(detail: Optional[dict], tz_name: Optional[str] = None,
              now: Any = None) -> str:
    """A push-notification body (≤ 300 chars everywhere it is used)."""
    when = reset_when_phrase((detail or {}).get("period_end"), tz_name, now)
    if when:
        return f"Your agent’s monthly AI budget is used up. It resets on {when}."
    return "Your agent’s AI budget is used up."


def frame_fields(detail: Optional[dict]) -> dict:
    """The extra keys a chat ``error`` frame carries for this refusal."""
    end = resets_at(detail)
    return {"code": FRAME_CODE, "retryable": False,
            "resets_at": iso_utc(end) if end else None}
