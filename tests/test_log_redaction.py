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


# ══════════════════════════════════════════════════════════════════════
# Round 46 (A15) — identifier redaction.
#
# The 2026-09-15 fleet survey found, on ORDINARY code paths: a live Telegram
# bot token in full inside an httpx request line (httpx logs every request URL
# at INFO and no logger level was set for it anywhere in `backend/`),
# subscriber phone numbers in E.164, WhatsApp JIDs and LIDs, and a user's real
# name. None of those is the diagnostic — what an operator needs is whether an
# identifier arrived, whether it is the SAME one as last time, and its shape,
# all of which a mask preserves.
#
# The new filter is attached to the root HANDLERS, not to the root logger: a
# filter on a logger runs only for records logged THROUGH that logger, so a
# root-logger filter never sees a record that propagated up from
# `app.agent.channels…` — which is every leaking line. That distinction is
# pinned below, and it is the difference between this working and looking like
# it works. (The token filter above is logger-scoped for the opposite and
# equally deliberate reason, documented in its own test.)
# ══════════════════════════════════════════════════════════════════════

import io                                            # noqa: E402

from app.services.log_redaction import (             # noqa: E402
    RedactUserIdentifiers, install_content_redaction, scrub_identifiers,
)


# A fabricated number, a fabricated JID and a fabricated token. Nothing in
# this file is a real identifier — the round's own privacy rule applies to
# its tests.
PHONE = "+12045550137"
JID = "12045550137@s.whatsapp.net"
LID = "98765432100@lid"
BOT_URL = "https://api.telegram.org/bot8123456789:AAH_fake_token_value_0123456789x/getUpdates"


def test_e164_keeps_only_a_shape():
    out = scrub_identifiers(f"connection.open self={PHONE}")
    assert PHONE not in out
    assert "5550137" not in out
    # Enough to tell two numbers apart in a trail, not enough to dial.
    assert out.endswith("37")
    assert "+1***37" in out


def test_two_numbers_stay_distinguishable():
    a = scrub_identifiers("+12045550137")
    b = scrub_identifiers("+12045550199")
    assert a != b


def test_jid_becomes_a_stable_hash_keeping_the_domain():
    out = scrub_identifiers(f"inbound from {JID}")
    assert "12045550137" not in out
    assert out.endswith("@s.whatsapp.net")   # the SHAPE is the diagnostic
    assert out == scrub_identifiers(f"inbound from {JID}")  # stable
    assert scrub_identifiers(JID) != scrub_identifiers("12045550199@s.whatsapp.net")


def test_lid_is_masked_too():
    out = scrub_identifiers(f"lid seen {LID}")
    assert "98765432100" not in out
    assert out.endswith("@lid")


def test_bot_token_never_survives():
    out = scrub_identifiers(f"HTTP Request: POST {BOT_URL}")
    assert "AAH_fake_token_value" not in out
    assert "8123456789" not in out
    assert "/bot***" in out
    # The rest of the line — which endpoint was called — is kept.
    assert "getUpdates" in out


def test_ordinary_lines_are_untouched():
    for line in (
        "[PERF] phase3_save: 34505ms",
        "agent_run_total 53537ms",
        "[WS] Authenticated user: feed0082",
        "GET /agent/health HTTP/1.1 200 OK",
    ):
        assert scrub_identifiers(line) == line


def test_the_identifier_filter_never_drops_a_record():
    """A log filter that can swallow a line is worse than the leak it fixes."""
    f = RedactUserIdentifiers()
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    assert f.filter(rec) is True
    assert rec.getMessage() == "hello world"


def test_a_filter_never_raises_on_a_malformed_record():
    f = RedactUserIdentifiers()
    # Arg count mismatch: getMessage() raises. The filter must still pass it.
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "a %s %s", ("one",), None)
    assert f.filter(rec) is True


def test_the_identifier_in_an_ARG_is_masked_not_only_in_the_message():
    """httpx logs the URL as an argument, not as part of the format string."""
    f = RedactUserIdentifiers()
    rec = logging.LogRecord("httpx", logging.INFO, __file__, 1,
                            "HTTP Request: %s %s", ("POST", BOT_URL), None)
    f.filter(rec)
    assert "AAH_fake_token_value" not in rec.getMessage()


def test_install_masks_records_that_PROPAGATE_from_a_child_logger():
    """The load-bearing one. Every leaking line in the survey came from a
    module logger (`app.agent.channels…`, `httpx`), i.e. from a record that
    propagates to the root's handler. A filter attached to the root LOGGER
    would not see any of them, and the whole fix would be cosmetic."""
    root = logging.getLogger()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    old_handlers = list(root.handlers)
    old_level = root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    try:
        install_content_redaction()
        logging.getLogger("app.agent.channels.whatsapp_baileys").info(
            "phone_to_jid.updated_from_inbound phone=%s jid=%s", PHONE, JID,
        )
        handler.flush()
        out = stream.getvalue()
        assert PHONE not in out, out
        assert "12045550137" not in out, out
        assert "+1***37" in out, out
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)


def test_install_quiets_httpx_so_request_urls_stop_being_logged():
    """Masking the token is the belt; not logging every URL at INFO is the
    braces. The token leak existed because nothing had ever set this level."""
    old = logging.getLogger("httpx").level
    try:
        install_content_redaction()
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING
    finally:
        logging.getLogger("httpx").setLevel(old)


def test_install_content_redaction_is_idempotent():
    """It runs at import time in agent_main and may be called again by a
    test or a re-entrant boot; N calls must not stack N filters."""
    root = logging.getLogger()
    handler = logging.StreamHandler(io.StringIO())
    old_handlers = list(root.handlers)
    root.handlers = [handler]
    try:
        install_content_redaction()
        install_content_redaction()
        install_content_redaction()
        assert sum(isinstance(f, RedactUserIdentifiers) for f in handler.filters) == 1
    finally:
        root.handlers = old_handlers
        for f in list(root.filters):
            if isinstance(f, RedactUserIdentifiers):
                root.removeFilter(f)


def test_agent_main_installs_it_before_anything_can_log():
    """Ordering, not presence: a redaction installed after the first module
    logs its first line is a redaction with a hole in it."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "agent_main.py").read_text("utf-8")
    assert "install_content_redaction()" in src
    basic = src.index("logging.basicConfig(")
    install = src.index("install_content_redaction()")
    first_app_import = src.index("from app.config import settings")
    assert basic < install < first_app_import, (
        "install_content_redaction must sit between basicConfig and the first "
        "application import"
    )


# ── Tracebacks: the half a Filter structurally cannot reach ────────────────

def test_an_exception_traceback_is_masked():
    """`Filter.filter` can only touch `record.msg`/`record.args`;
    `logging.Formatter.format` appends `formatException(record.exc_info)`
    AFTERWARDS. So every `logger.exception(...)` / `exc_info=True` site — ~500
    of them, 40 on the WhatsApp/Telegram paths — bypassed the redaction
    entirely, including for the two classes it was built for. Verified before
    the fix: both the phone and the JID appeared in the formatted output."""
    root = logging.getLogger()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    old_handlers = list(root.handlers)
    old_level = root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    try:
        install_content_redaction()
        try:
            raise ValueError(f"allowlist rejected {PHONE} ({JID})")
        except ValueError:
            logging.getLogger("app.agent.channels.whatsapp_baileys").exception(
                "[RESTART] whatsapp allowlist hot-apply failed"
            )
        handler.flush()
        out = stream.getvalue()
        assert "Traceback" in out, "the path did not run — the test proves nothing"
        assert PHONE not in out, out
        assert JID.split("@")[0] not in out, out
        assert "+1***37" in out, out
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)


def test_a_bot_token_inside_a_raised_httpx_error_is_masked():
    """`httpx.HTTPStatusError.__str__` embeds the request URL, so a
    raised-and-logged Telegram error reproduces the exact `/bot<TOKEN>/` leak
    `_BOT_TOKEN` exists to stop — through the traceback, not the message."""
    root = logging.getLogger()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    old_handlers = list(root.handlers)
    old_level = root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    try:
        install_content_redaction()
        try:
            raise RuntimeError(
                "Client error '401 Unauthorized' for url "
                "'https://api.telegram.org/bot123456789:AAHfitzZZZZZZZZZZZZZZZZZZZZZZZZZZZZ/sendMessage'"
            )
        except RuntimeError:
            logging.getLogger("app.agent.telegram_bot").exception("[TG] send failed")
        handler.flush()
        out = stream.getvalue()
        assert "AAHfitz" not in out, out
        assert "/bot***" in out, out
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)


def test_uvicorns_own_handlers_are_redacted_too():
    """uvicorn configures `uvicorn.access` with its own handler and
    `propagate=False`, so its records never reach a root handler. Attaching
    only to the root left the one logger that renders every request line
    untouched."""
    import io as _io

    from app.services.log_redaction import RedactingFormatter

    acc = logging.getLogger("uvicorn.access")
    stream = _io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    old_handlers = list(acc.handlers)
    old_prop = acc.propagate
    old_level = acc.level
    acc.handlers = [handler]
    acc.propagate = False
    acc.setLevel(logging.INFO)
    try:
        install_content_redaction()
        assert isinstance(handler.formatter, RedactingFormatter)
        acc.info("connected %s", PHONE)
        handler.flush()
        out = stream.getvalue()
        assert PHONE not in out, out
        assert "+1***37" in out, out
    finally:
        acc.handlers = old_handlers
        acc.propagate = old_prop
        acc.setLevel(old_level)


def test_the_formatter_is_installed_once_and_keeps_the_inner_one():
    """Wrapping must not stack, and must not throw away the format the
    operator reads (uvicorn's access formatter, Railway's JSON one)."""
    from app.services.log_redaction import RedactingFormatter

    root = logging.getLogger()
    handler = logging.StreamHandler(io.StringIO())
    inner = logging.Formatter("PREFIX %(message)s")
    handler.setFormatter(inner)
    old_handlers = list(root.handlers)
    root.handlers = [handler]
    try:
        install_content_redaction()
        install_content_redaction()
        fmt = handler.formatter
        assert isinstance(fmt, RedactingFormatter)
        assert fmt._inner is inner, "the original formatter was discarded"
        rec = logging.LogRecord("x", logging.INFO, __file__, 1, "hi %s", (PHONE,), None)
        assert fmt.format(rec).startswith("PREFIX ")
        assert PHONE not in fmt.format(rec)
    finally:
        root.handlers = old_handlers
        for f in list(root.filters):
            if isinstance(f, RedactUserIdentifiers):
                root.removeFilter(f)


def test_the_formatter_never_raises():
    from app.services.log_redaction import RedactingFormatter

    class _Boom(logging.Formatter):
        def format(self, record):
            return "ok " + PHONE

    fmt = RedactingFormatter(_Boom())
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "m", None, None)
    assert PHONE not in fmt.format(rec)


def test_a_device_suffixed_jid_is_masked():
    """Baileys' most common JID shape is `<number>:<device>@s.whatsapp.net`.
    The Python rule's local-part class had no `:` — so the whole JID failed to
    match, the number inside went to Loki in the clear, and `_E164` could not
    catch the remainder either (a JID carries no `+`). The JS twin
    (`whatsapp_sidecar/sidecar.mjs` `_JID_RE`) has always had it."""
    out = scrub_identifiers(f"send to 12045550137:12@s.whatsapp.net ok")
    assert "12045550137" not in out
    assert "jid:" in out and out.endswith("@s.whatsapp.net ok")


def test_two_devices_of_one_person_hash_alike():
    """`…:12@` and `…:27@` are one contact on two devices. Hashing the device
    suffix too would read as two people in the trail."""
    a = scrub_identifiers("12045550137:12@s.whatsapp.net")
    b = scrub_identifiers("12045550137:27@s.whatsapp.net")
    assert a == b
    assert a != scrub_identifiers("12045550199:12@s.whatsapp.net")


def test_a_phone_without_its_plus_is_masked():
    """`_E164` fires only on a literal `+` followed by unbroken digits, so a
    number logged from a config field or split by a formatter passed through
    untouched. The loose rule is the backstop the root-handler install needs."""
    for raw in ("phone=12045550137", "+1 204 555 0137", "+1-204-555-0137",
                "wa 442045550137"):
        out = scrub_identifiers(raw)
        assert "5550137" not in out, raw
        assert "***" in out, raw


def test_the_loose_rule_leaves_ordinary_numbers_alone():
    """It runs over EVERY line this process emits, so a false positive is a
    destroyed diagnostic. A date, an IPv4, a version, a byte count and a
    uuid must survive it byte for byte."""
    for raw in ("day 2026-09-15", "ip 192.168.100.101", "v 1.2.3",
                "size_bytes=10485760", "lag_ms=9470.5 top=app.api.ws_chat:4215",
                "user 00000000-0000-4000-8000-000000000001",
                "image 1cd801aacb11"):
        assert scrub_identifiers(raw) == raw, raw
