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
