"""One definition of "this failure is infrastructure, not data".

2026-09-15, 15:08:19.140Z: `GET /api/routines` answered **200 with `[]`**
while the host's pgbouncer was dead and every DB access in the container was
raising `ConnectionRefusedError`. A total database outage reached the user as
"you have no automations". The same shape — a broad `except Exception` whose
handler returns an empty collection — exists on several read paths, and it is
indistinguishable at the edge from a truthful empty list.

The rule this module encodes (already law for the app: an unreachable agent
returns 503, never an empty list):

* a PER-ROW failure still degrades — one unrenderable row must not brick a list;
* an INFRASTRUCTURE failure (the database is unreachable, the pool timed out,
  the connection died mid-statement) is a 503 with `code="backend_unavailable"`,
  because the honest answer to "what do I have?" is "I cannot tell you", never
  "nothing".

Deliberately NOT infrastructure: `ProgrammingError` / `IntegrityError` /
`DataError`. Those are schema drift or bad data — real bugs, but the row-level
degrade is the established behaviour for them and turning them into 503s would
take a whole tenant's list offline for one stale column.
"""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, status

try:  # SQLAlchemy is always present in this service; stay importable without it.
    from sqlalchemy.exc import (
        DBAPIError,
        DisconnectionError,
        InterfaceError,
        OperationalError,
    )
    from sqlalchemy.exc import TimeoutError as SATimeoutError
    _SA_INFRA: tuple = (OperationalError, InterfaceError, DisconnectionError, SATimeoutError)
except Exception:  # pragma: no cover - defensive
    DBAPIError = ()  # type: ignore[assignment]
    _SA_INFRA = ()

#: Exceptions that mean "the backing store could not be reached or spoke to".
#: `ConnectionRefusedError`/`ConnectionResetError` are `ConnectionError`
#: subclasses; asyncpg raised the first of those RAW (not wrapped by
#: SQLAlchemy) during the 15:08Z outage, which is why the builtin is listed.
INFRASTRUCTURE_ERRORS: tuple = _SA_INFRA + (
    ConnectionError,
    asyncio.TimeoutError,
    TimeoutError,
)

BACKEND_UNAVAILABLE = "backend_unavailable"

_DEFAULT_MESSAGE = (
    "We could not reach the service behind this. This is on our side — "
    "your data is still there. Please try again in a moment."
)


def is_infrastructure_error(exc: BaseException, *, _depth: int = 4) -> bool:
    """True when `exc` (or something it wraps) means "the store was unreachable".

    Walks `__cause__` because a driver-level `ConnectionRefusedError` can
    arrive wrapped in a generic `DBAPIError`, and the wrapper's class alone
    would misread it as a query fault.
    """
    seen = 0
    cur: BaseException | None = exc
    while cur is not None and seen <= _depth:
        if isinstance(cur, INFRASTRUCTURE_ERRORS):
            return True
        if DBAPIError and isinstance(cur, DBAPIError) and getattr(cur, "connection_invalidated", False):
            return True
        cur = cur.__cause__
        seen += 1
    return False


def backend_unavailable(message: str = "") -> HTTPException:
    """The 503 every swept read path raises. Body shape matches the existing
    `detail={"code": ..., "message": ...}` convention (see `app/api/automations.py`)."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": BACKEND_UNAVAILABLE, "message": message or _DEFAULT_MESSAGE},
    )


def raise_if_infrastructure(exc: BaseException, message: str = "") -> None:
    """`raise_if_infrastructure(e)` inside a broad handler: infra escapes as a
    503, everything else falls through to the caller's existing degrade."""
    if is_infrastructure_error(exc):
        raise backend_unavailable(message) from exc
