"""`[TURNTRACE]` — one correlation id for a chat turn, across three hops.

Round 46 contract C10. The app mints the `client_msg_id` it already mints;
every hop logs ONE line per stage keyed by `cmid_h`, so "what happened to
this message" is a single Loki query instead of a reconstruction from
WebSocket accept counts.

The hash is FNV-1a/32 over the ASCII id, hex — the SAME function as
`cmidHash` in the app's `src/shared/turnTrace.ts`. Deliberately not a
cryptographic hash: its job is to join logs cheaply and identically on both
sides, over ids that are always ASCII uuid4s. It is a hash rather than the
id itself because raw correlation ids are banned from telemetry.

Never log message text, filenames, or a raw id through here. `outcome` is a
short machine token (a reason code, a count, a kind).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193

# The C10 stage list. Not enforced at runtime (a typo must not break a turn),
# but the set a grep/dashboard can rely on.
STAGES = (
    "client_send",
    "ws_open",
    "ws_close",
    "platform_receipt",
    "agent_ack",
    "ledger_claim",
    "dispatch",
    "tool_start",
    "tool_end",
    "artifact_created",
    "done",
    "reconnect",
    "resume",
    "replay_served",
    "render",
)


def cmid_hash(client_msg_id: Optional[str]) -> str:
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


def turntrace(
    stage: str,
    client_msg_id: Optional[str],
    *,
    t_ms: Optional[int] = None,
    outcome: Optional[str] = None,
    **extra: Any,
) -> None:
    """Emit one `[TURNTRACE] {...}` line. Never raises."""
    try:
        payload = {"cmid_h": cmid_hash(client_msg_id), "stage": stage}
        if t_ms is not None:
            payload["t_ms"] = int(t_ms)
        if outcome:
            payload["outcome"] = str(outcome)[:32]
        for k, v in extra.items():
            if v is None:
                continue
            payload[k] = v if isinstance(v, (int, float, bool)) else str(v)[:32]
        logger.info("[TURNTRACE] %s", json.dumps(payload, separators=(",", ":")))
    except Exception:  # telemetry must never break a turn
        pass
