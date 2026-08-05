"""The platform's own logs must be readable, or nothing else can be diagnosed.

Two independent defects made production effectively unobservable, and they
compound: one silenced the application, the other flooded what was left.

  1. `platform_main` never configured logging. The root logger therefore sat at
     Python's default WARNING and every `logger.info` in the service was
     discarded — including `media_proxy`'s per-stage `audio_stream TIMING` line,
     which exists *specifically* so a delivery regression can be told apart from
     a cold track. Measured against production on 2026-08-05: 1000 consecutive
     log lines, ZERO `INFO:app.*` records. `agent_main.py` has configured
     logging since it was written; this entry point simply never did.

  2. 823 of those same 1000 lines were `POST /api/mcp/mcp` access records
     arriving at 2.3/s. They held the entire retained log window to SIX
     MINUTES, so any incident older than that was unrecoverable regardless of
     level.

Fixing only (1) makes it worse — a louder application spends the six-minute
window faster. Both halves are the fix, which is why they are tested together.
"""

from __future__ import annotations

import ast
import importlib
import logging
import os
from pathlib import Path

import pytest

import platform_main


def _access_record(path: str, status: int) -> logging.LogRecord:
    """A record shaped exactly like uvicorn's access logger emits.

    uvicorn passes a 5-tuple as `args` and lets its own formatter render it:
    (client_addr, method, full_path, http_version, status_code). Matching on
    that tuple rather than on the formatted string is the whole reason the
    filter keeps working when uvicorn's format changes.
    """
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("1.2.3.4:5678", "POST", path, "1.1", status),
        exc_info=None,
    )


# ── The application must actually be audible ──────────────────────────


def test_the_entry_point_configures_logging_at_all():
    """The bug was an ABSENCE. Pin that the call exists in this module."""
    src = Path(platform_main.__file__).read_text()
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "basicConfig"
    ]
    assert calls, "platform_main must configure logging — without it every logger.info is dropped"


def test_an_earlier_basic_config_does_not_win():
    """`force=True` is load-bearing, and its absence would be invisible.

    Something in the import graph already calls `basicConfig()` — production
    lines arrive in logging's BASIC_FORMAT (`WARNING:app.mcp_auth:…`), which
    only a configured handler produces; the bare last-resort handler emits the
    message alone. A second plain `basicConfig` is a documented NO-OP once the
    root logger has handlers, so without `force` this fix would change nothing
    while looking completely correct.

    So reproduce that exact situation — a prior `basicConfig` that leaves root
    at WARNING — and require the reload to still win. Asserting the keyword is
    in the source would pass against a `basicConfig` that had been reordered
    behind something else; this asserts the outcome.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        for h in list(root.handlers):
            root.removeHandler(h)
        logging.basicConfig()  # the competing call: installs a handler, leaves WARNING
        root.setLevel(logging.WARNING)
        assert root.level == logging.WARNING, "precondition"

        importlib.reload(platform_main)

        assert logging.getLogger().level == logging.INFO, (
            "an earlier basicConfig won — every logger.info in the service stays discarded"
        )
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        importlib.reload(platform_main)


def test_an_app_logger_emits_at_info(caplog):
    """The end the user cares about: an `app.*` INFO record survives.

    `caplog` installs its own handler, so this asserts the LEVEL is permissive
    — which is the half that was broken — not the handler wiring.
    """
    with caplog.at_level(logging.INFO, logger="app.api.media_proxy"):
        logging.getLogger("app.api.media_proxy").info(
            "[media_proxy] audio_stream TIMING video_id=abc tier=MISS extract_ms=3400"
        )
    assert any("audio_stream TIMING" in r.getMessage() for r in caplog.records)


def test_the_root_level_is_info_not_warning():
    assert logging.getLogger().level <= logging.INFO, (
        "root above INFO means every logger.info in the service is discarded"
    )


def test_noisy_third_party_loggers_are_pinned_to_warning():
    """Raising the app to INFO without lowering these trades one blind spot for
    another — botocore and httpx alone would out-talk the application."""
    for name in ("httpx", "botocore", "boto3", "urllib3", "sqlalchemy.engine"):
        assert logging.getLogger(name).level >= logging.WARNING, f"{name} left chatty at INFO"


def test_log_level_is_env_overridable(monkeypatch):
    """An operator must be able to turn it down without a code change."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    try:
        reloaded = importlib.reload(platform_main)
        assert logging.getLogger().level == logging.WARNING
    finally:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        importlib.reload(platform_main)


def test_a_bogus_log_level_falls_back_to_info_instead_of_crashing():
    """A typo in an env var must not stop the service from booting."""
    os.environ["LOG_LEVEL"] = "VERBOSE"  # not a logging level
    try:
        importlib.reload(platform_main)
        assert logging.getLogger().level == logging.INFO
    finally:
        os.environ.pop("LOG_LEVEL", None)
        importlib.reload(platform_main)


# ── …and the flood must stop drowning it ──────────────────────────────


def test_successful_mcp_access_lines_are_dropped():
    f = platform_main.QuietAccessFilter()
    assert f.filter(_access_record("/api/mcp/mcp", 200)) is False
    assert f.filter(_access_record("/api/mcp/mcp", 202)) is False


def test_a_failing_mcp_request_is_still_logged():
    """The one access line on a quiet path that is worth having."""
    f = platform_main.QuietAccessFilter()
    assert f.filter(_access_record("/api/mcp/mcp", 500)) is True
    assert f.filter(_access_record("/api/mcp/mcp", 401)) is True


def test_ordinary_traffic_is_untouched():
    f = platform_main.QuietAccessFilter()
    assert f.filter(_access_record("/api/media/xyz/audio_stream", 206)) is True
    assert f.filter(_access_record("/api/auth/login", 200)) is True


def test_a_query_string_does_not_defeat_the_match():
    """uvicorn logs `full_path`, which carries the query string."""
    f = platform_main.QuietAccessFilter()
    assert f.filter(_access_record("/api/mcp/mcp?session_id=abc", 200)) is False


def test_a_record_that_is_not_an_access_line_is_kept():
    """The filter sits on one logger today, but a non-access record reaching it
    must pass through rather than be silently eaten."""
    f = platform_main.QuietAccessFilter()
    plain = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg="something entirely different", args=None, exc_info=None,
    )
    assert f.filter(plain) is True


def test_a_malformed_status_is_kept_rather_than_dropped():
    """Fail OPEN. Dropping a line we failed to parse is how a filter turns into
    a second blind spot."""
    f = platform_main.QuietAccessFilter()
    rec = _access_record("/api/mcp/mcp", 200)
    rec.args = ("1.2.3.4", "POST", "/api/mcp/mcp", "1.1", "not-a-status")
    assert f.filter(rec) is True


def test_installing_the_filter_is_idempotent():
    """The lifespan can run more than once in a process (tests, reloads); two
    copies of the filter is a latent double-drop."""
    access = logging.getLogger("uvicorn.access")
    before = list(access.filters)
    try:
        access.filters = []
        assert platform_main.install_access_log_filter() is True
        assert platform_main.install_access_log_filter() is True
        installed = [f for f in access.filters if isinstance(f, platform_main.QuietAccessFilter)]
        assert len(installed) == 1
    finally:
        access.filters = before


def test_the_quiet_list_can_be_emptied_by_an_operator(monkeypatch):
    """An empty override must disable the filter entirely — the escape hatch for
    debugging the very endpoint it hides."""
    monkeypatch.setenv("LOG_QUIET_ACCESS_PATHS", "")
    try:
        reloaded = importlib.reload(platform_main)
        assert reloaded._QUIET_ACCESS_PATHS == ()
        access = logging.getLogger("uvicorn.access")
        before = list(access.filters)
        try:
            access.filters = []
            assert reloaded.install_access_log_filter() is False
            assert access.filters == []
        finally:
            access.filters = before
    finally:
        monkeypatch.delenv("LOG_QUIET_ACCESS_PATHS", raising=False)
        importlib.reload(platform_main)


def test_the_lifespan_installs_the_filter():
    """A filter nobody attaches is decoration. Pin the call site."""
    src = Path(platform_main.__file__).read_text()
    tree = ast.parse(src)
    lifespan = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan"
    )
    called = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "install_access_log_filter"
        for n in ast.walk(lifespan)
    )
    assert called, "lifespan must call install_access_log_filter()"


# ── The dead import that warned on every boot ─────────────────────────


def test_no_import_of_a_media_metrics_task_that_never_existed():
    """`start_metrics_flush_task` was imported from `app.api.media_proxy` and
    has never existed there — `git log -S 'def start_metrics_flush_task'`
    returns nothing. Every replica printed "⚠️ Could not start media metrics
    flush task" on every boot, which is noise that reads like a real fault.
    """
    src = Path(platform_main.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "start_metrics_flush_task", (
                    "platform_main imports a function media_proxy does not define"
                )
