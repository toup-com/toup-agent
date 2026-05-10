"""
T5b — In-process connector quarantine list.

Operator break-glass primitive consulted by the dispatcher BEFORE
any other rule (vault, channel, manifest). When a connector id is
quarantined, every dispatch returns ConnectorReauthRequired with a
"Service paused by operator" reason, regardless of whether the
identity is otherwise valid.

State: in-process set + the API endpoint also writes to a DB table
for cross-replica + cross-restart durability. v1 reads from the
in-process set only — restarts repopulate from the DB at lifespan
boot. Cross-replica fanout (multiple platform processes) is a v2
problem; today the platform runs single-process on Railway, and the
DB write doubles as the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class QuarantineEntry:
    connector_id: str
    reason: str
    actor_user_id: Optional[str]
    quarantined_at: datetime


_quarantined: dict[str, QuarantineEntry] = {}


def is_quarantined(connector_id: str) -> Optional[QuarantineEntry]:
    """Returns the entry if quarantined, None otherwise."""
    return _quarantined.get(connector_id)


def add(entry: QuarantineEntry) -> None:
    _quarantined[entry.connector_id] = entry


def remove(connector_id: str) -> None:
    _quarantined.pop(connector_id, None)


def list_active() -> list[QuarantineEntry]:
    return sorted(_quarantined.values(), key=lambda e: e.quarantined_at)


def reset_for_tests() -> None:
    _quarantined.clear()
