"""A live session token must never survive into the access log.

Observed in production on 2026-08-06 and again 2026-08-07: every WebSocket
connect logged the full request line, including the `?token=<JWT>` the mobile
client still authenticates with. That token is a full-privilege session
credential and the log is retained and widely readable.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("ENVIRONMENT", "test")

from app.services.log_redaction import (            # noqa: E402
    RedactQueryTokens, install_log_redaction, scrub,
)

_JWT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3MTMyNTU2NS04MjQ0LTRi"
        "NGEtYmJjNi0yZDVlZTRjOTM0MDAiLCJleHAiOjE3ODY2Mzk1NzN9.HNpIytewZ50TnzTO")


def _record(*args):
    return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                             '%s - "%s %s HTTP/%s" %d', args, None)


def test_the_real_production_line_is_redacted():
    rec = _record("1.2.3.4:5", "WebSocket", f"/api/ws/chat?token={_JWT}", "1.1", 200)
    assert RedactQueryTokens().filter(rec) is True
    line = rec.getMessage()
    assert _JWT not in line, "the JWT survived into the formatted log line"
    assert "token=<redacted>" in line
    assert "/api/ws/chat" in line, "the path itself must still be readable"


def test_a_filter_never_drops_a_line():
    """Redaction is not a reason to lose a log record."""
    for args in ((), None, ("no query here",), (object(),)):
        rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                                "%s", args if args else None, None)
        assert RedactQueryTokens().filter(rec) is True


def test_requests_without_credentials_are_untouched():
    rec = _record("1.2.3.4:5", "GET", "/api/health?verbose=1", "1.1", 200)
    RedactQueryTokens().filter(rec)
    assert "verbose=1" in rec.getMessage()


def test_other_credential_shapes():
    assert "<redacted>" in scrub("/x?access_token=abc123")
    assert "<redacted>" in scrub("/x?api_key=sk-live-xyz")
    assert "<redacted>" in scrub("/x?auth=Bearer%20abc")
    # ...and only the value, never the whole query string
    out = scrub("/x?token=abc&page=2")
    assert "page=2" in out and "abc" not in out


def test_install_is_idempotent():
    """Lifespan can run more than once per process (reload, tests, workers)."""
    install_log_redaction()
    install_log_redaction()
    lg = logging.getLogger("uvicorn.access")
    assert sum(isinstance(f, RedactQueryTokens) for f in lg.filters) == 1


def test_filter_is_attached_to_the_logger_not_a_handler():
    """uvicorn/gunicorn replace handlers; a handler-scoped filter would stop
    applying the moment the process is started a different way."""
    install_log_redaction()
    lg = logging.getLogger("uvicorn.access")
    assert any(isinstance(f, RedactQueryTokens) for f in lg.filters)
