"""What counts as "the store was unreachable", and what does not.

`app/api/_infra_errors.py` is the single definition every swept read path now
shares. It has to draw the line in exactly one place, because both sides of it
are load-bearing:

  * too NARROW and 2026-09-15 repeats — `GET /api/routines` answered 200 `[]`
    for eleven minutes while `ConnectionRefusedError` was raised on every DB
    access, and the user read that as "you have no automations";
  * too WIDE and one drifted column takes a whole tenant's panel offline,
    which is the degrade those broad handlers were written for in the first
    place.

The second endpoint here, `extension._list_devices`, feeds both
`GET /api/extension/status` and `GET /api/extension/devices`. Its swallow had
the same shape: a dead database rendered as "you have paired no devices".

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. \
        pytest tests/test_infra_error_honesty.py -q
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import DataError, IntegrityError, OperationalError, ProgrammingError

from app.api import _infra_errors as infra


# ── the predicate ────────────────────────────────────────────────


@pytest.mark.parametrize("exc", [
    ConnectionRefusedError(111, "Connection refused"),
    ConnectionResetError("peer reset"),
    asyncio.TimeoutError(),
    OperationalError("SELECT 1", {}, Exception("server closed the connection")),
])
def test_unreachable_is_infrastructure(exc):
    assert infra.is_infrastructure_error(exc) is True


@pytest.mark.parametrize("exc", [
    ProgrammingError("SELECT x", {}, Exception('column "x" does not exist')),
    IntegrityError("INSERT", {}, Exception("duplicate key")),
    DataError("SELECT", {}, Exception("invalid input syntax")),
    ValueError("a row would not serialise"),
    AttributeError("Routine has no attribute 'phase'"),
])
def test_data_and_schema_faults_are_not_infrastructure(exc):
    assert infra.is_infrastructure_error(exc) is False, (
        "widening this turns one stale column into a whole panel going dark"
    )


def test_a_wrapped_connection_error_is_still_infrastructure():
    """asyncpg's refusal can arrive inside a generic wrapper; the wrapper's
    own class would misread it as a query fault."""
    wrapper = RuntimeError("db wipe failed")
    wrapper.__cause__ = ConnectionRefusedError(111, "Connection refused")
    assert infra.is_infrastructure_error(wrapper) is True


def test_the_503_carries_a_code_and_never_blames_the_user():
    e = infra.backend_unavailable()
    assert e.status_code == 503
    assert e.detail["code"] == "backend_unavailable"
    body = e.detail["message"].lower()
    assert "internet" not in body and "connection" not in body, (
        "the app told a user to check their internet during a server-side "
        "outage; the server's own copy must not repeat the mistake"
    )


def test_raise_if_infrastructure_lets_everything_else_through():
    infra.raise_if_infrastructure(ValueError("bad row"))  # must not raise
    with pytest.raises(HTTPException):
        infra.raise_if_infrastructure(ConnectionRefusedError(111, "refused"))


# ── the swept endpoint ───────────────────────────────────────────


def _broken(exc):
    class _Maker:
        def __call__(self, *a, **k):
            return self

        async def __aenter__(self):
            raise exc

        async def __aexit__(self, *a):
            return False
    return _Maker()


@pytest.mark.asyncio
async def test_extension_devices_does_not_report_zero_devices_on_a_dead_db(monkeypatch):
    from app.api import extension

    monkeypatch.setattr(
        extension, "async_session_maker",
        _broken(ConnectionRefusedError(111, "Connection refused")),
    )
    with pytest.raises(HTTPException) as ei:
        await extension._list_devices("u" * 36)
    assert ei.value.status_code == 503
    assert ei.value.detail["code"] == "backend_unavailable"


@pytest.mark.asyncio
async def test_extension_devices_still_degrades_on_a_missing_table(monkeypatch):
    """The pre-migration branch stays: a replica mid-rollout must not break
    pairing."""
    from app.api import extension

    monkeypatch.setattr(
        extension, "async_session_maker",
        _broken(ProgrammingError("SELECT", {}, Exception("no such table"))),
    )
    assert await extension._list_devices("u" * 36) == []
