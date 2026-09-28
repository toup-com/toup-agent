"""`cmid_hash` — the one correlation hash for a chat turn, in a shared home.

This function was born in `app/api/_turn_trace.py` and still lives there for
every existing caller. It moved HERE because R48 needs it in the AGENT layer
(`agent_runner`'s `[TURN_WATERFALL]` line and `app/db/db_span.py` both stamp
`cmid_h`), and `from app.api import ...` inside `app/agent/` is the wrong
direction: the platform image ships no `app/agent/`, the agent image ships
both, and an api→agent→api import loop is one refactor away from a boot-time
cycle. `app/services/` is the layer both already import (health_signals,
loop_health), so it is the only place the two halves can share a function
without either one importing the other.

ONE COST, RECORDED. `app/api/_turn_trace.py` was a leaf module with no app
imports at all; re-exporting from here puts `app/services/__init__.py` — and
therefore embedding_service, memory_extractor, auth_service, memory_service
and document_service — on the import graph of everything that imports it,
including `ws_chat_proxy` in the PLATFORM image. Both entrypoints were
verified to import clean in both lanes (R48), so this is a layering note and
not a break; but a future leaf-module claim about `_turn_trace` is now false.

There is exactly ONE implementation. `_turn_trace.cmid_hash` re-exports this
one; the app-parity fixtures in `tests/test_log_privacy_user_content.py` test
that name and therefore test this code.
"""

from __future__ import annotations

from typing import Optional

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193


def cmid_hash(client_msg_id: Optional[str]) -> str:
    """FNV-1a/32 over the id, hex — the SAME function as `cmidHash` in the
    app's `src/shared/turnTrace.ts`.

    Deliberately not a cryptographic hash: its job is to join logs cheaply and
    identically on both sides, over ids that are always ASCII uuid4s. It is a
    hash rather than the id itself because raw correlation ids are banned from
    telemetry.
    """
    s = client_msg_id or ""
    if not s:
        return "00000000"
    h = _FNV_OFFSET
    # The low byte of each UTF-16 CODE UNIT — exactly what the app's
    # `charCodeAt(i) & 0xff` yields, surrogate pairs included. NOT
    # `encode("ascii", "ignore")`, which DROPS a non-ASCII character where the
    # app keeps its low byte: ids are ASCII uuid4s today, so the two agreed by
    # accident, and the first non-ASCII id would have silently broken the
    # three-hop join this hash exists for.
    for b in s.encode("utf-16-le")[0::2]:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return f"{h:08x}"
