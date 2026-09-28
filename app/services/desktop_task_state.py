"""The one function that decides a remote Mac task's status.

`desktop/docs/CONNECTIONS.md` §8.3. Every writer calls `derive()` — staging,
approve, reject, action-result, task-status, cancel, revoke and every read —
so the status a phone sees can only ever be the answer to one question asked
one way. Pure: no clock, no database, no I/O. The caller supplies `now`.

The rules apply in order and the first match wins. The order is the design:
a cancellation outranks everything, a Mac that went away outranks work that
looks in progress, and a card waiting for the phone outranks a turn that is
still technically running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, List, Optional, Tuple

TASK_DEADLINE_S = 30 * 60
MAX_OPEN_TASKS_PER_MAC = 3

TASK_STATUSES = (
    "queued", "dispatched", "running", "waiting_on_you", "waiting_on_mac",
    "done", "failed", "cancelled", "mac_offline",
)
TERMINAL = frozenset({"done", "failed", "cancelled", "mac_offline"})

OUTCOME_CANCELLED_BY_YOU = "You cancelled this."
OUTCOME_DISCONNECTED = "This Mac was disconnected."
OUTCOME_DEADLINE = "This didn't finish within 30 minutes."
OUTCOME_NEEDS_YOU = "Needs your OK."
OUTCOME_ERRORED = "Your agent hit a problem and stopped."
OUTCOME_CARD_EXPIRED = (
    "The confirmation ran out before you answered, so nothing changed on "
    "your Mac."
)
OUTCOME_DECLINED = "You declined, so nothing changed on your Mac."
OUTCOME_DONE = "Done. Open the conversation for the answer."
OUTCOME_UNREACHABLE = (
    "Your agent couldn't be reached, so nothing was sent to your Mac."
)


def outcome_offline(device_name: str) -> str:
    return f"{device_name} went offline before this finished."[:240]


def outcome_waiting_on_mac(device_name: str) -> str:
    return (
        f"Waiting for someone at {device_name} to confirm. If nobody answers "
        "within 2 minutes, your Mac says no."
    )[:240]


@dataclass
class TaskFacts:
    status: str
    turn_state: str
    created_at: datetime
    outcome: Optional[str] = None
    cancelled_at: Optional[datetime] = None
    cancel_reason: Optional[str] = None


@dataclass
class ActionFacts:
    status: str
    expires_at: datetime
    mac_prompt_at: Optional[datetime] = None
    #: The Mac's one-line summary, when it answered. Never its data.
    summary: str = ""
    created_at: Optional[datetime] = None
    has_result: bool = False


@dataclass
class DeviceFacts:
    name: str = "Your Mac"
    online: bool = False
    revoked: bool = False


def action_facts_from_row(row: Any) -> ActionFacts:
    summary = ""
    has_result = False
    raw = getattr(row, "result_json", None)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                summary = str(parsed.get("summary") or "")[:240]
                has_result = True
        except (ValueError, TypeError):
            pass
    return ActionFacts(
        status=row.status,
        expires_at=row.expires_at,
        mac_prompt_at=getattr(row, "mac_prompt_at", None),
        summary=summary,
        created_at=getattr(row, "created_at", None),
        has_result=has_result,
    )


def _in_flight(a: ActionFacts) -> bool:
    return a.status in ("approved", "dispatched") and not a.has_result


def derive(
    task: TaskFacts,
    actions: Iterable[ActionFacts],
    device: DeviceFacts,
    now: datetime,
) -> Tuple[str, Optional[str]]:
    """Return `(status, outcome)`. A terminal task is returned unchanged."""
    if task.status in TERMINAL:
        return task.status, task.outcome

    acts: List[ActionFacts] = sorted(
        actions, key=lambda a: a.created_at or datetime.min,
    )

    # 1
    if task.cancelled_at is not None:
        return "cancelled", (
            OUTCOME_DISCONNECTED if task.cancel_reason == "revoked"
            else OUTCOME_CANCELLED_BY_YOU
        )
    # 2
    if device.revoked:
        return "cancelled", OUTCOME_DISCONNECTED
    # 3
    if task.turn_state == "mac_offline":
        return "mac_offline", outcome_offline(device.name)
    # 4
    if (now - task.created_at).total_seconds() > TASK_DEADLINE_S:
        return "failed", OUTCOME_DEADLINE
    # 5
    if task.turn_state == "queued":
        return "queued", None
    if task.turn_state == "accepted":
        return "dispatched", None
    # 6
    if any(a.status == "pending" and a.expires_at > now for a in acts):
        return "waiting_on_you", OUTCOME_NEEDS_YOU
    busy = any(_in_flight(a) for a in acts) or task.turn_state == "running"
    # 7
    if busy and not device.online:
        return "mac_offline", outcome_offline(device.name)
    # 8
    if any(_in_flight(a) and a.mac_prompt_at is not None for a in acts):
        return "waiting_on_mac", outcome_waiting_on_mac(device.name)
    # 9
    if busy:
        return "running", None
    # 10
    if task.turn_state == "errored":
        return "failed", OUTCOME_ERRORED
    # 11
    failed = [a for a in acts if a.status == "failed"]
    if failed:
        return "failed", (failed[-1].summary or "Your Mac didn't finish this.")[:240]
    # 12
    lapsed = [
        a for a in acts
        if a.status == "expired" or (a.status == "pending" and a.expires_at <= now)
    ]
    if lapsed and not any(a.status == "executed" for a in acts):
        return "failed", OUTCOME_CARD_EXPIRED
    # 13
    if acts and all(a.status == "rejected" for a in acts):
        return "cancelled", OUTCOME_DECLINED
    # 14
    executed = [a for a in acts if a.status == "executed"]
    if executed and executed[-1].summary:
        return "done", executed[-1].summary[:240]
    return "done", OUTCOME_DONE
