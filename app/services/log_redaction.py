"""Keep bearer tokens out of the access log.

Every WebSocket connect is logged by uvicorn's access logger with the full
request line, and the mobile client still authenticates with `?token=<JWT>`
(the subprotocol migration is mid-bake — see `_ws_auth_helpers`). So each
connect wrote a live, unexpired access token to the platform log:

    "WebSocket /api/ws/chat?token=eyJhbGciOiJIUzI1NiIs..." [accepted]

Those logs are retained, shipped to Railway, and readable by anyone with
deploy access — a credential store nobody intended to create. The token is a
full-privilege session credential: with it you can open that user's chat
socket, read their threads and drive their agent, until it expires.

This does not wait for the subprotocol migration to finish, because the leak
is live now and the migration is gated on client rollout. Redaction is
independent of it and stays correct afterwards (a request with no `token=`
is untouched).

The filter is deliberately conservative:
  * it only rewrites the ARG uvicorn formats into `%s`, never the message;
  * anything unexpected (missing args, odd shapes) is passed through — a log
    filter must never be able to swallow a line;
  * it always returns True, so nothing is dropped, only rewritten.
"""
from __future__ import annotations

import logging
import re

# `token=` / `access_token=` / `auth=` up to the next & or whitespace or quote.
_SECRET_QS = re.compile(
    r"((?:access_|id_|auth_)?token|auth|api_key|key)=([^&\s\"']+)",
    re.IGNORECASE,
)
_REDACTED = r"\1=<redacted>"


def scrub(text: str) -> str:
    """Replace secret-looking query params in a URL/……/request line."""
    if not text or "=" not in text:
        return text
    return _SECRET_QS.sub(_REDACTED, text)


class RedactQueryTokens(logging.Filter):
    """Strip credentials from uvicorn.access records before they are emitted."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        try:
            if record.args and isinstance(record.args, tuple):
                scrubbed = tuple(
                    scrub(a) if isinstance(a, str) else a for a in record.args
                )
                if scrubbed != record.args:
                    record.args = scrubbed
            elif isinstance(record.msg, str) and "token=" in record.msg:
                # Some emitters pre-format. Cheap guard, same rule.
                record.msg = scrub(record.msg)
        except Exception:  # pragma: no cover - a filter must never raise
            pass
        return True


# ── Identifier redaction (round 46, A15) ───────────────────────────────────
#
# The agent container's log is shipped to Loki and read by operators and by
# agents. On 2026-09-15 it held, on ordinary code paths: a full Telegram bot
# token inside an httpx request line (`POST https://api.telegram.org/bot<TOKEN>
# /getUpdates` — no httpx logger level was set anywhere in backend/), E.164
# phone numbers and WhatsApp JIDs from the sidecar, and a user's full name
# from admin_pool. None of those are diagnostics; all of them are the user.
#
# These patterns are applied to every record on the root handler, because the
# leaks are spread across modules nobody enumerates in advance. The rule for
# each: keep enough to tell two things apart, keep nothing that identifies.

# A Telegram bot token in a URL path. Also matches when the token has already
# been split across `bot` and the rest by a formatter.
_BOT_TOKEN = re.compile(r"/bot\d{5,}:[A-Za-z0-9_\-]{20,}", re.IGNORECASE)

# E.164, 8-15 digits after the +. Kept: the leading `+` and country-ish first
# digit, plus the LAST TWO digits — enough to tell two test numbers apart in a
# trail, not enough to dial or to identify.
_E164 = re.compile(r"\+(\d)(\d{5,12})(\d{2})\b")

# WhatsApp JIDs: `<digits>@s.whatsapp.net`, `<digits>@lid`, `<digits>-<ts>@g.us`.
# `:` is in the class because Baileys' MOST COMMON shape is device-suffixed
# (`<number>:<device>@s.whatsapp.net`); without it the whole JID failed to
# match and the phone number inside went to Loki in the clear, while the
# `_E164` rule could not catch the remainder either (a JID carries no `+`).
# The JS twin (whatsapp_sidecar/sidecar.mjs `_JID_RE`) has always had it.
_JID = re.compile(r"\b([0-9][0-9\-.:]{4,24})@(s\.whatsapp\.net|lid|g\.us|c\.us)\b")

# A phone number written WITHOUT its `+`, or with separators — `phone=1415…`,
# `+1 415 555 …`, `+1-415-…`. `_E164` needs a literal `+` followed by unbroken
# digits, which the config/pairing paths do not always produce. Bounded on
# both sides by a non-word character so it cannot eat an identifier; the
# substitution decides (see `_mask_loose_phone`) and leaves dates, IPv4s,
# versions and short counters alone. Known residual: a BARE 10-15 digit run
# is masked, so an epoch-in-milliseconds logged as a bare integer comes out
# as `+***00(13)`. Accepted — that shape is also exactly a phone number, and
# the mask still carries the digit count.
_PHONE_LOOSE = re.compile(r"(?<![\w.+])(\+?\d[\d\-. ()]{6,18}\d)(?![\w.])")


def _jid_hash(local: str) -> str:
    import hashlib

    # Hash the NUMBER, not the number-plus-device: `1415…:12@` and `1415…:27@`
    # are the same person on two devices and must produce the same token, or
    # the trail reads as two contacts.
    local = local.split(":", 1)[0]
    return hashlib.sha256(local.encode("utf-8", "ignore")).hexdigest()[:8]


def _mask_loose_phone(m: "re.Match") -> str:
    """Mask only what is really a phone number. Keeps the last two digits and
    the digit count, so two numbers stay distinguishable in a trail.

    This rule runs over EVERY log line the process emits, so its false
    positives are as much a defect as its false negatives — a masked byte
    count or a masked date is a diagnostic destroyed. Hence the three narrow
    shapes below rather than "8-15 digits anywhere": a bare 8- or 9-digit run
    is far more often a size or a counter than a phone number, and a
    dash/dot-separated run with no `+` is far more often a date, an IPv4 or a
    version. Those keep their `_E164` / `_JID` coverage, which is exact."""
    raw = m.group(1)
    digits = "".join(c for c in raw if c.isdigit())
    if not (8 <= len(digits) <= 15):
        return raw
    masked = f"+***{digits[-2:]}({len(digits)})"
    if raw.startswith("+"):
        return masked
    if raw.isdigit():
        # Bare run: a phone number written without its `+` (`phone=1415…`).
        return masked if len(digits) >= 10 else raw
    if " " in raw or "(" in raw:
        return masked
    return raw


def scrub_identifiers(text: str) -> str:
    """Mask phone numbers, chat-app ids and bot tokens in one log line."""
    if not text:
        return text
    out = _BOT_TOKEN.sub("/bot***", text)
    out = _JID.sub(lambda m: f"jid:{_jid_hash(m.group(1))}@{m.group(2)}", out)
    out = _E164.sub(lambda m: f"+{m.group(1)}***{m.group(3)}", out)
    # Last, and deliberately after the three shape-specific rules: they keep
    # more of the value (the country digit, the JID's domain) and a
    # generic pass first would swallow the shapes they are tuned for.
    out = _PHONE_LOOSE.sub(_mask_loose_phone, out)
    # Last, and deliberately after the three shape-specific rules: they keep
    # more of the value (the country digit, the JID's domain) and a
    # generic pass first would swallow the shapes they are tuned for.
    return out


class RedactUserIdentifiers(logging.Filter):
    """Mask identifiers in the rendered message of every record.

    Renders with `record.getMessage()` and replaces msg/args wholesale when
    anything changed, because the identifier can live in either half (the
    sidecar formats its own strings; httpx logs a URL as an arg). Never
    raises and never drops a record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        try:
            rendered = record.getMessage()
            masked = scrub_identifiers(rendered)
            if masked != rendered:
                record.msg = masked
                record.args = None
        except Exception:  # pragma: no cover - a filter must never raise
            pass
        return True


class RedactingFormatter(logging.Formatter):
    """Wrap another formatter and mask identifiers in its ENTIRE output.

    A Filter is not enough and never was. `Filter.filter` can only rewrite
    `record.msg`/`record.args`; `logging.Formatter.format` appends
    `formatException(record.exc_info)` and `record.stack_info` AFTERWARDS. So
    every one of the ~500 `logger.exception(...)` / `exc_info=True` sites in
    this codebase bypassed the filter entirely — including the two classes it
    exists for: `httpx.HTTPStatusError.__str__` embeds the request URL (a live
    Telegram bot token), and a raised WhatsApp error carries the E.164 and the
    JID. A Formatter sees message, exc_text and stack_info as one string, so
    nothing that is added later can escape it."""

    def __init__(self, inner: "logging.Formatter | None" = None) -> None:
        super().__init__()
        self._inner = inner if inner is not None else logging.Formatter()

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        rendered = self._inner.format(record)
        try:
            return scrub_identifiers(rendered)
        except Exception:  # pragma: no cover - a formatter must never raise
            return rendered


# uvicorn installs its own handlers with `propagate=False`, so a record logged
# through `uvicorn.access` never reaches a root handler and is never redacted
# by one. Named here rather than discovered, because the point is that they do
# NOT propagate.
_OWN_HANDLER_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "gunicorn",
    "gunicorn.error",
    "gunicorn.access",
)


def _redact_handlers_of(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if not any(isinstance(f, RedactUserIdentifiers) for f in handler.filters):
            handler.addFilter(RedactUserIdentifiers())
        if not isinstance(handler.formatter, RedactingFormatter):
            handler.setFormatter(RedactingFormatter(handler.formatter))


def install_content_redaction() -> None:
    """Mask identifiers on the ROOT HANDLERS (and uvicorn's own) and quiet httpx.

    Handlers, not the root logger: a filter on a logger runs only for records
    logged THROUGH that logger, so a root-logger filter never sees a record
    that propagated up from `app.agent.channels…`. Handlers see everything
    that reaches them. Call after `logging.basicConfig`.

    Idempotent, and meant to be called AGAIN after the server has configured
    its own logging — uvicorn replaces its handlers when it starts, which is
    long after this module is imported.

    httpx logs every request at INFO with the full URL. That is how a live
    Telegram bot token reached Loki; WARNING keeps the failures and drops the
    URLs."""
    root = logging.getLogger()
    _redact_handlers_of(root)
    if not any(isinstance(f, RedactUserIdentifiers) for f in root.filters):
        root.addFilter(RedactUserIdentifiers())
    for name in _OWN_HANDLER_LOGGERS:
        _redact_handlers_of(logging.getLogger(name))
    for noisy in ("httpx", "httpcore", "hpack", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def install_log_redaction() -> None:
    """Attach the filter to every logger that renders a request line.

    Attached to the LOGGER (not a handler): uvicorn installs its own handlers
    and Railway/gunicorn setups replace them, so a handler-scoped filter
    silently stops applying the moment the process is run a different way.
    """
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn", "gunicorn.access"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactQueryTokens) for f in logger.filters):
            logger.addFilter(RedactQueryTokens())
