"""The chat transport's fault vocabulary — one table, three hops.

Round 46, incident 3 (2026-09-15): PgBouncer died, the agent's ws_chat
handler raised `ConnectionRefusedError` on its first DB access, uvicorn
closed the transport with no close frame (peer sees 1006), the platform
proxy's `_safe_close_code` laundered 1006 → 1000, and the app — having
no code to class on — matched its own English error text against
`/websocket.*(closed|failed|error)/` and told the user to check her
internet. A server-side database outage was rendered as the user's 5G.

The root design problem is that no fault CLASS crossed agent → proxy →
client: every hop re-guessed from a close code or a sentence, and the
last hop's guess blamed the user. This module is the single typed
vocabulary all three hops share. Its app-side twin is
`src/shared/wsFaults.ts` in the mobile repo — the two tables must stay
byte-identical in codes and numbers.

Close-code space: 4404/4502/4503/4504 are already taken by
ws_chat_proxy's warm-up vocabulary, so the new classes start at 4505.
Everything here is ADDITIVE on the CLOSE CODE: an old client that does
not know 4505-4508 treats them as it treated 1000/1006 (a bounded silent
retry, then an error).

The error FRAME is not additive in the same sense, and that asymmetry is
load-bearing. In builds 123-126 `case 'error'` with an unrecognised code
calls `onError` and settles the turn, and a settled turn makes
`onclose`'s `if (settled) return` skip the bounded pre-reply resend —
the resend that eventually got incident 2's turn through. Those builds
also have no `turnErrorCopy` branch for any code here, so the user gets
the generic sentence either way. A fault frame is therefore sent only
where an immediate, terminal error IS the intent (the agent-authored
4505/4507/4508); a merely-unexplained upstream end travels as the close
code alone. See ws_chat_proxy's relay end.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# --- machine codes ---------------------------------------------------------

AGENT_DB_UNAVAILABLE = "agent_db_unavailable"
PLATFORM_UPSTREAM_LOST = "platform_upstream_lost"
AGENT_INTERNAL = "agent_internal"
AGENT_NOT_READY = "agent_not_ready"
ATTACHMENT_REJECTED = "attachment_rejected"

# --- the table -------------------------------------------------------------
#
# `retryable` means "re-sending this exact turn, with the SAME
# client_msg_id, cannot double-deliver": the fault is raised BEFORE the
# message was read off the socket and turned into a run, so the
# exactly-once ledger never saw it. `retry_after_ms` is advisory; the
# client may back off further but must not retry sooner.
FAULTS: Dict[str, Dict[str, Any]] = {
    AGENT_DB_UNAVAILABLE: {
        "close_code": 4505,
        "retryable": True,
        "retry_after_ms": 5000,
        # User-facing text lives in the app (activity.ts turnErrorCopy);
        # this is the fallback for a client that does not know the code.
        "message": "Your agent's storage is unreachable. Your message was not sent.",
    },
    PLATFORM_UPSTREAM_LOST: {
        "close_code": 4506,
        "retryable": True,
        "retry_after_ms": 2000,
        "message": "The connection to your agent ended before it replied.",
    },
    AGENT_INTERNAL: {
        "close_code": 4507,
        "retryable": False,
        "retry_after_ms": None,
        "message": "Your agent hit an internal error handling this message.",
    },
    AGENT_NOT_READY: {
        "close_code": 4503,
        "retryable": True,
        "retry_after_ms": 1500,
        "message": "Your agent is still starting up.",
    },
    ATTACHMENT_REJECTED: {
        "close_code": 4508,
        "retryable": False,
        "retry_after_ms": None,
        "message": "That attachment could not be accepted.",
    },
}

# Codes whose close is an INFRASTRUCTURE verdict rather than a warm-up
# hint — the client must settle the turn with this code instead of
# looping the warm-up budget. Kept beside the table so the app-side
# `INFRA_CLOSE_CODES` has one source of truth to mirror.
INFRA_CLOSE_CODES = frozenset(
    {
        FAULTS[AGENT_DB_UNAVAILABLE]["close_code"],
        FAULTS[PLATFORM_UPSTREAM_LOST]["close_code"],
        FAULTS[AGENT_INTERNAL]["close_code"],
        # A refused attachment is an infra VERDICT for the client too: the
        # turn is settled on the code, never re-sent (the bytes were the
        # problem). The app's set carried it from the start; the two halves
        # of this table must be equal or the mirror is a lie.
        FAULTS[ATTACHMENT_REJECTED]["close_code"],
    }
)


class WsInfraUnavailable(Exception):
    """An infrastructure dependency (the tenant DB, today) is unreachable.

    Raised instead of returning "not authenticated", because those are
    opposite verdicts with opposite recoveries: incident 3's DB outage
    reached `_authenticate_ws`'s `except Exception: return None`, was closed
    as 4001 "Authentication required", and the proxy mapped 4001 → 4503
    ("agent starting") — so the app re-sent silently up to fifteen times
    into a database that was down for eleven minutes.

    `code` is the fault to answer with; the default is the only case that
    has ever raised this."""

    def __init__(self, code: str = AGENT_DB_UNAVAILABLE, detail: Optional[str] = None):
        super().__init__(code)
        self.code = code
        self.detail = detail


# Exception classes that mean "the dependency is down", not "the caller is
# wrong". asyncpg/SQLAlchemy raise these through several layers, and every
# one of them arrived at incident 3's `except Exception` as an auth failure.
#
# ONE definition, shared with the REST read paths (`app.api._infra_errors`).
# Two predicates shipped in round 46 and they disagreed on three SQLAlchemy
# families: this one tested `DBAPIError`, the superclass of ProgrammingError /
# DataError / IntegrityError, so schema drift or an over-long client-supplied
# id came out as `agent_db_unavailable` — "your agent's storage is
# unreachable", `retryable: true`, `retry_after_ms: 5000` — and the app
# re-sent the same client_msg_id forever against a permanent server bug. A
# deterministic failure is `agent_internal` (4507, not retryable), which is
# the honest verdict and the one that stops the loop. `is_infrastructure_error`
# also walks `__cause__`, which this never did: a driver-level
# ConnectionRefusedError wrapped in a generic DBAPIError used to be read by its
# wrapper alone.
def is_infra_error(exc: BaseException) -> bool:
    if isinstance(exc, WsInfraUnavailable):
        return True
    try:
        from app.api._infra_errors import is_infrastructure_error

        if is_infrastructure_error(exc):
            return True
    except Exception:  # pragma: no cover — importable without the app package
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
    # asyncpg wraps the socket error; the class name is the stable signal
    # across its versions.
    name = type(exc).__name__
    return name in {
        "ConnectionDoesNotExistError",
        "CannotConnectNowError",
        "TooManyConnectionsError",
        "ConnectionFailureError",
        "PostgresConnectionError",
    }


def close_code_for(code: str) -> int:
    """The WS close code carrying `code`. Unknown → 4507 (agent_internal):
    an unclassified failure is an internal one, never a normal close."""
    entry = FAULTS.get(code)
    return int(entry["close_code"]) if entry else FAULTS[AGENT_INTERNAL]["close_code"]


def code_for_close(close_code: int) -> Optional[str]:
    """Inverse lookup: the machine code a close code carries, or None."""
    for name, entry in FAULTS.items():
        if entry["close_code"] == close_code:
            return name
    return None


def fault_frame(
    code: str,
    *,
    retry_after_ms: Optional[int] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """The error frame that PRECEDES the close.

    Sent as a frame as well as a close code because the close code alone
    is lost on several paths (a browser that never surfaces it, a proxy
    hop that rewrites it, an app build that does not know the number) —
    and because both clients already class an error frame by `code`.

    `detail` is machine-oriented and must never carry user content: a
    reason token, a stage name, a count. It is not shown to the user.
    """
    entry = FAULTS.get(code) or FAULTS[AGENT_INTERNAL]
    frame: Dict[str, Any] = {
        "type": "error",
        "code": code if code in FAULTS else AGENT_INTERNAL,
        "retryable": bool(entry["retryable"]),
        "message": entry["message"],
    }
    after = retry_after_ms if retry_after_ms is not None else entry["retry_after_ms"]
    if after is not None:
        frame["retry_after_ms"] = int(after)
    if detail:
        frame["detail"] = str(detail)[:200]
    return frame
