"""SQL statement echo must be OFF unless explicitly asked for.

Regression guard for a live production defect (2026-08-01): every engine
was built with `echo=settings.debug`, `debug` defaults to True, and the
agent containers never set DEBUG — so all 55 tenant agents ran with full
statement echo. Echo logs BOUND PARAMETERS, so `INSERT INTO messages
(... content ...)` wrote the user's message body verbatim into the
container log. Confirmed on a live tenant before the fix.

The load-bearing assertion is `test_echo_is_not_tied_to_debug`: it sets
`debug=True` (the shipped default) and requires echo to stay False. That
is exactly the state production was in, so it fails on the old code.

Direct invocation works:
    cd backend && ENVIRONMENT=development python tests/test_sql_echo_default_off.py
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.db import database as db_module  # noqa: E402


# Every URL/run_mode combination that reaches a distinct create_async_engine
# call in _build_engine. All three carried the same defect.
ENGINE_CASES = [
    ("sqlite branch", "sqlite+aiosqlite:///./test.db", "agent"),
    ("agent postgres branch", "postgresql+asyncpg://u:p@h:5432/d", "agent"),
    ("platform postgres branch", "postgresql+asyncpg://u:p@h:5432/d", "platform"),
    ("fallback branch", "postgresql+asyncpg://u:p@h:5432/d", "other"),
]


def _echo_for(database_url: str, run_mode: str, **setting_overrides) -> bool:
    """Build an engine with patched settings and report the echo kwarg
    that _build_engine actually passed to SQLAlchemy."""
    captured = {}

    def fake_create_async_engine(url, **kwargs):
        captured.update(kwargs)
        return object()

    # _env_file=None keeps this hermetic. Without it, Settings() reads the
    # developer's backend/.env (which sets DEBUG=false locally) and the
    # default-case test would pass for the wrong reason on a machine whose
    # .env happens to disagree with the container's environment — which is
    # precisely the mismatch that hid this bug in the first place.
    fake_settings = Settings(_env_file=None, run_mode=run_mode, **setting_overrides)
    with patch.object(db_module, "create_async_engine", fake_create_async_engine), \
         patch.object(db_module, "settings", fake_settings):
        db_module._build_engine(database_url)

    assert "echo" in captured, "engine was built without an explicit echo kwarg"
    return captured["echo"]


def test_echo_is_off_by_default_on_every_branch():
    for name, url, run_mode in ENGINE_CASES:
        echo = _echo_for(url, run_mode)
        assert echo is False, f"{name}: echo defaulted to {echo!r}, expected False"


def test_echo_is_not_tied_to_debug():
    """The actual production state: debug=True (the shipped default) and
    SQL_ECHO unset. Echo must still be off. Fails on the old code."""
    for name, url, run_mode in ENGINE_CASES:
        echo = _echo_for(url, run_mode, debug=True)
        assert echo is False, (
            f"{name}: debug=True re-enabled statement echo (echo={echo!r}). "
            "Echo logs bound parameters, i.e. user message content, into "
            "container logs — it must not ride on a general dev flag."
        )


def test_sql_echo_can_still_be_turned_on():
    """The knob has to work, or the next person debugging will just put
    echo=settings.debug back."""
    for name, url, run_mode in ENGINE_CASES:
        echo = _echo_for(url, run_mode, sql_echo=True)
        assert echo is True, f"{name}: SQL_ECHO=true did not enable echo (echo={echo!r})"


def test_shipped_default_is_off():
    assert Settings(_env_file=None).sql_echo is False, (
        "sql_echo must ship defaulting to False"
    )


if __name__ == "__main__":
    tests = [
        test_echo_is_off_by_default_on_every_branch,
        test_echo_is_not_tied_to_debug,
        test_sql_echo_can_still_be_turned_on,
        test_shipped_default_is_off,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
