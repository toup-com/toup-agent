"""Desktop relay bridge — the agent's end of the socket to a user's Mac.

Modelled on `extension_bridge.py` (same shape: per-user endpoint registry,
newest wins; `dispatch()` returning an awaited future keyed by task id;
per-endpoint rate limit; unsolicited event fan-out), and deliberately a
SEPARATE module rather than an extension of it — `agent-tool-relay.md` §5.1:
"a Chrome tab session and a filesystem grant are different objects and the
module's `_Session` is entirely about tabs."

The wire contract is `desktop/docs/RELAY_PROTOCOL.md`. Section numbers below
refer to it.

WHICH PROCESS OWNS WHAT — read this before adding a reader
──────────────────────────────────────────────────────────────────────────
This registry is **in-process, and every one of its consumers is in the same
process by construction.** The only caller of `dispatch()` is the `desktop`
skill, which runs inside `tool_executor` inside the tenant agent container,
which is also the process that terminates `/api/ws/desktop`. So the hot path
never crosses a process boundary.

That is not luck, it is the one thing `agent-tool-relay.md` §1.9 says to get
right. The extension shipped its *session* routes reading this same kind of
dict from the PLATFORM, which serves `numReplicas: 2` and has its own empty
copy of the module — so on production `GET /extension/sessions` returns `[]`,
`pause`/`resume` 404, and the "take back control" story does not work. The
rule that follows: **nothing the web UI reads may be read out of here.**
Presence goes to the platform DB via the heartbeat in `app/api/desktop.py`
(the half of the extension's split that *was* fixed), and device activity
goes to the same place. `is_connected()` is for the agent's own
execution-time check and for the WS handler — not for a settings page.

WHY THE PENDING MAP IS PER-USER AND NOT PER-ENDPOINT
──────────────────────────────────────────────────────────────────────────
`extension_bridge` keys `pending` on the endpoint and fails every future on
disconnect (`extension_bridge.py:145-148`). A Mac is not a browser tab: it
sleeps, its socket half-opens, and §5.4 requires the device to keep results
for tasks whose socket died in a bounded outbox and flush them after `hello`
on the next connection — "because the agent is `await`ing a future keyed by
that id, and a dropped result is a hung turn, not a lost log line."

A per-endpoint map cannot accept that flush: the future it belongs to died
with the old endpoint. So futures live per USER, keyed by task id, and a
disconnect leaves them pending. The cost is honest and bounded: a task whose
Mac never comes back waits out its own `timeout_s` instead of failing in
milliseconds. `_TASK_DEADLINE_SLACK_S` is the same "+ a couple of seconds"
`extension_bridge.py:488` uses, so the device's own `timeout` result nearly
always wins the race and the agent gets the device's words rather than a
generic timeout.

An id is resolved AT MOST ONCE (§5.4), which the `pop` gives for free.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# ─── Limits (§5.7) — one place, matching the client's own table ─────────
MAX_INBOUND_FRAME_BYTES = 1024 * 1024        # 1 MiB; dropped and counted above
MAX_TASKS_IN_FLIGHT = 4                       # `busy` beyond this
RATE_LIMIT_PER_WINDOW = 60                    # 60 tasks / 60 s
RATE_LIMIT_WINDOW_S = 60.0
ACTION_MAX_LEN = 96
REASON_MAX_LEN = 240
#: Added to the device-side `timeout_ms` when waiting on the future, so the
#: device's own `timeout` status (which carries a sentence) beats ours.
_TASK_DEADLINE_SLACK_S = 2.0

#: `[A-Za-z0-9_.-]{1,96}` per §5.3, as a membership test rather than a regex
#: so a pathological action string cannot cost backtracking.
_ACTION_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)

#: Terminal result statuses (§5.4). `denied` is NOT an error: the user (or a
#: standing local policy) refused, and the agent must stop asking — which is
#: why it gets its own exception type below rather than being folded into
#: `DesktopError`.
RESULT_STATUSES = frozenset({
    "ok", "denied", "unavailable", "invalid_arguments",
    "timeout", "error", "cancelled",
})

#: Event kinds the device may send (§5.6). Metadata only, by construction:
#: there is no field on that frame that can carry arguments, paths or output,
#: so the activity feed cannot become a second copy of the user's files on
#: the wire. Anything else is dropped and counted.
EVENT_KINDS = frozenset({
    "tool_started", "tool_finished", "approval_pending", "approval_resolved",
})


# ─── Exceptions ────────────────────────────────────────────────────────
class DesktopUnavailable(Exception):
    """No Mac is connected for this user, or the socket died mid-task."""


class DesktopDenied(Exception):
    """The user, or standing local policy, refused.

    Separate from `DesktopError` on purpose (§5.4): a refusal is not a
    fault, and the tool handler turns it into a sentence that tells the
    model to stop rather than one that invites a retry.
    """

    def __init__(self, summary: str = "", code: str = ""):
        super().__init__(summary or "refused on the device")
        self.summary = summary
        self.code = code


class DesktopError(Exception):
    def __init__(self, status: str, summary: str = "", code: str = ""):
        super().__init__(f"{status}: {summary}" if summary else status)
        self.status = status
        self.summary = summary
        self.code = code


@dataclass
class _Endpoint:
    """One live relay socket."""

    user_id: str
    ws: WebSocket
    device_id: Optional[str] = None
    token_jti: Optional[str] = None
    connected_at: float = field(default_factory=time.time)
    protocol: int = 1
    #: Device self-report from `hello` (§5.1) — name, os_version,
    #: app_version, flavor. Display metadata for the activity panel.
    device_info: Dict[str, Any] = field(default_factory=dict)
    #: The device's current tool list (§5.2): what this Mac is *willing* to
    #: do right now — capability on, grant present, macOS permission
    #: granted. Re-read on every connect because a grant may have been
    #: revoked while the Mac was asleep.
    #:
    #: ⚠️ USED FOR: the Settings UI, and refusing a call early with a good
    #: sentence. NEVER for the agent's wire tools array — per-turn variation
    #: there forks the provider cache lineage (§5.2's own warning, and
    #: `tool_entitlements.py:21-39`).
    tools: List[Dict[str, Any]] = field(default_factory=list)
    recent_tasks: deque = field(default_factory=deque)
    counters: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    #: Last frame of ANY kind from the Mac. The Mac pings every 25 s, so a
    #: socket that has been silent for a minute is half-open, whatever the
    #: TCP state says (CONNECTIONS.md §7, F-I).
    last_inbound_at: float = field(default_factory=time.time)


# Newest wins, and a LIST rather than a single slot for the reason
# `extension_bridge.py:89-104` records: keying strictly by user made a second
# client forcibly close the first, which reconnected, which closed the
# second, forever — and the agent never got a stable socket to dispatch
# through. Two Macs on one account is the ordinary case here, not an edge.
_endpoints_by_user: Dict[str, List[_Endpoint]] = defaultdict(list)
#: user_id → {task_id: Future}. Per-user, see the module docstring.
_pending: Dict[str, Dict[str, asyncio.Future]] = defaultdict(dict)
#: Bounded, in-memory, per-process — a debugging aid and nothing more. The
#: durable activity trail is the platform DB (§8). Explicitly NOT the
#: extension's 100-entry ring masquerading as an audit log (§1.8).
_recent_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
_lock = asyncio.Lock()

#: relay task id → (pending-action id, remote task id). Bounded, because an
#: approval event for an id nobody remembers is simply not forwarded.
_RELAY_ID_CAP = 512
_relay_ids: "OrderedDict[str, Tuple[str, Optional[str], str]]" = OrderedDict()
#: remote task id → relay ids dispatched for it, so cancelling the task can
#: cancel what is actually running on the Mac.
_task_relay_ids: Dict[str, List[str]] = defaultdict(list)
#: Remote tasks the phone cancelled. The skill refuses any further desktop
#: call for them and the turn's `cancel_check` reads this.
_cancelled_tasks: "OrderedDict[str, float]" = OrderedDict()

#: Set by `app.api.desktop` at import: called with
#: (user_id, action_id, remote_task_id, kind) when the Mac reports that it is
#: asking someone at the Mac about an approved action. A hook rather than an
#: import because that module imports this one.
approval_event_hook: Optional[
    Callable[[str, str, Optional[str], str], Awaitable[None]]
] = None


@dataclass
class DesktopTarget:
    """The Mac a remote task's turn is pinned to (CONNECTIONS.md M3).

    `flags` is a mutable holder on purpose: a `ContextVar.set()` made inside
    a child task is lost to its parent, a change to an inherited object is
    not, and the turn runner reads `mac_unavailable` after the turn ends.
    """

    task_id: str
    device_id: str
    device_name: str = "your Mac"
    flags: Dict[str, Any] = field(default_factory=dict)


DESKTOP_TARGET: contextvars.ContextVar[Optional[DesktopTarget]] = (
    contextvars.ContextVar("desktop_target", default=None)
)


def _active_eps(user_id: str) -> List[_Endpoint]:
    """Endpoints for the user, oldest first / newest last."""
    return list(_endpoints_by_user.get(user_id) or [])


# ─── Endpoint lifecycle ────────────────────────────────────────────────
async def register(
    user_id: str,
    ws: WebSocket,
    *,
    device_id: Optional[str] = None,
    token_jti: Optional[str] = None,
) -> _Endpoint:
    async with _lock:
        ep = _Endpoint(
            user_id=user_id, ws=ws, device_id=device_id, token_jti=token_jti,
        )
        _endpoints_by_user[user_id].append(ep)
        logger.info(
            "[desktop-bridge] registered user=%s device=%s (conns=%d)",
            user_id[:8], (device_id or "?")[:8],
            len(_endpoints_by_user[user_id]),
        )
        return ep


async def unregister(user_id: str, ep: Optional[_Endpoint] = None) -> None:
    """Drop an endpoint. **Pending futures are deliberately left alone.**

    See the module docstring: the device keeps a bounded outbox and flushes
    it after `hello` on the next connection, so failing the futures here
    would throw away a result that is about to arrive. They time out on
    their own deadline if the Mac really is gone.
    """
    async with _lock:
        bucket = _endpoints_by_user.get(user_id)
        if not bucket:
            return
        if ep is None:
            _endpoints_by_user.pop(user_id, None)
        else:
            remaining = [e for e in bucket if e is not ep]
            if remaining:
                _endpoints_by_user[user_id] = remaining
            else:
                _endpoints_by_user.pop(user_id, None)
        logger.info(
            "[desktop-bridge] unregistered user=%s (remaining=%d, pending=%d)",
            user_id[:8], len(_endpoints_by_user.get(user_id) or []),
            len(_pending.get(user_id) or {}),
        )


def is_connected(user_id: str) -> bool:
    """Whether a Mac is on the socket RIGHT NOW, in THIS process.

    The execution-time availability check (`agent-tool-relay.md` §5.8, and
    precedent A in §2.6). Never used to build the tools array.
    """
    return bool(_endpoints_by_user.get(user_id))


def connected_device_ids(user_id: str) -> List[str]:
    return [e.device_id for e in _active_eps(user_id) if e.device_id]


def is_device_connected(user_id: str, device_id: Optional[str]) -> bool:
    """Whether THIS Mac, not merely some Mac of this user, is on a socket."""
    if not device_id:
        return False
    return any(e.device_id == device_id for e in _active_eps(user_id))


def _endpoint_for(user_id: str, device_id: Optional[str]) -> Optional[_Endpoint]:
    """Newest endpoint for exactly `device_id`, or None. Never another Mac."""
    for ep in reversed(_active_eps(user_id)):
        if ep.device_id == device_id:
            return ep
    return None


def advertised_tool_names(user_id: str, device_id: Optional[str]) -> List[str]:
    """What this Mac said it is willing to do right now (§5.2).

    The Mac filters that list by its owner's grants, so a tool missing from
    it is a family the owner did not switch on. Used to refuse a pinned
    remote task's call before anything is staged or sent.
    """
    ep = _endpoint_for(user_id, device_id)
    if ep is None:
        return []
    return [
        str(t.get("name")) for t in ep.tools
        if isinstance(t, dict) and t.get("name")
    ]


def mark_inbound(ep: _Endpoint) -> None:
    ep.last_inbound_at = time.time()


def seconds_since_inbound(ep: _Endpoint) -> float:
    return max(0.0, time.time() - ep.last_inbound_at)


def track_relay_id(
    user_id: str, relay_id: str, action_id: str, remote_task_id: Optional[str],
) -> None:
    _relay_ids[relay_id] = (action_id, remote_task_id, user_id)
    _relay_ids.move_to_end(relay_id)
    while len(_relay_ids) > _RELAY_ID_CAP:
        _relay_ids.popitem(last=False)
    if remote_task_id:
        ids = _task_relay_ids[remote_task_id]
        if relay_id not in ids:
            ids.append(relay_id)
            del ids[:-16]


def mark_task_cancelled(task_id: str) -> None:
    _cancelled_tasks[task_id] = time.time()
    _cancelled_tasks.move_to_end(task_id)
    while len(_cancelled_tasks) > _RELAY_ID_CAP:
        _cancelled_tasks.popitem(last=False)


def is_task_cancelled(task_id: Optional[str]) -> bool:
    return bool(task_id) and task_id in _cancelled_tasks


async def cancel_task_relays(user_id: str, task_id: str) -> int:
    """Send a targeted `cancel` for every in-flight relay id of one task."""
    sent = 0
    pending = _pending.get(user_id) or {}
    for relay_id in list(_task_relay_ids.get(task_id) or []):
        if relay_id in pending and await cancel(user_id, relay_id):
            sent += 1
    return sent


def device_tools(user_id: str) -> List[Dict[str, Any]]:
    """What the newest connected Mac says it is currently willing to do.

    §5.2. For the Settings UI and for early refusal with a good sentence.
    Not for the prompt.
    """
    eps = _active_eps(user_id)
    return list(eps[-1].tools) if eps else []


def device_summary(user_id: str) -> Optional[Dict[str, Any]]:
    """A small display record for the newest endpoint, or None."""
    eps = _active_eps(user_id)
    if not eps:
        return None
    ep = eps[-1]
    return {
        "device_id": ep.device_id,
        "device": dict(ep.device_info),
        "protocol": ep.protocol,
        "connected_for_s": int(time.time() - ep.connected_at),
        "tool_names": [
            t.get("name") for t in ep.tools if isinstance(t, dict) and t.get("name")
        ],
        "in_flight": len(_pending.get(user_id) or {}),
    }


def recent_events(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    return list(_recent_events.get(user_id) or [])[-limit:]


def stats() -> Dict[str, Any]:
    flat = [ep for eps in _endpoints_by_user.values() for ep in eps]
    return {
        "active_users": len(_endpoints_by_user),
        "active_connections": len(flat),
        "pending_tasks": sum(len(p) for p in _pending.values()),
        "connections": [
            {
                "user_id": ep.user_id[:8],
                "device_id": (ep.device_id or "?")[:8],
                "connected_for_s": int(time.time() - ep.connected_at),
                "tools": len(ep.tools),
                "counters": dict(ep.counters),
            }
            for ep in flat
        ],
    }


# ─── Inbound routing ───────────────────────────────────────────────────
def handle_hello(ep: _Endpoint, msg: Dict[str, Any]) -> None:
    """§5.1. Records the device self-report and the initial tool list."""
    mark_inbound(ep)
    try:
        ep.protocol = int(msg.get("protocol") or 1)
    except (TypeError, ValueError):
        ep.protocol = 1
    dev = msg.get("device")
    if isinstance(dev, dict):
        ep.device_info = {
            k: str(v)[:80]
            for k, v in dev.items()
            if k in ("name", "os_version", "app_version", "flavor")
            and isinstance(v, (str, int, float))
        }
    handle_tools(ep, msg)


def handle_tools(ep: _Endpoint, msg: Dict[str, Any]) -> None:
    """§5.2. `inputSchema` is camelCase on the wire and stays that way.

    The type is frozen in `ToupSeam` with no `CodingKeys`, so it is the one
    camelCase key in an otherwise snake_case protocol. Normalising it here
    would put this module's guess in the middle of a contract the client
    already implements — the descriptors are stored verbatim.
    """
    mark_inbound(ep)
    raw = msg.get("tools")
    if not isinstance(raw, list):
        return
    ep.tools = [t for t in raw if isinstance(t, dict) and t.get("name")][:64]


def deliver_inbound(user_id: str, msg: Dict[str, Any]) -> None:
    """Route one parsed inbound frame.

    Unknown `type` values are dropped and counted, never fatal (§5) — a
    newer app build may send frames this server has never heard of.
    """
    for ep in _active_eps(user_id):
        mark_inbound(ep)
    mtype = msg.get("type")
    if mtype == "result":
        _deliver_result(user_id, msg)
    elif mtype == "event":
        _deliver_event(user_id, msg)
    else:
        for ep in _active_eps(user_id):
            ep.counters["unknown_frames"] += 1


def _deliver_result(user_id: str, msg: Dict[str, Any]) -> None:
    task_id = msg.get("id")
    if not task_id or not isinstance(task_id, str):
        return
    fut = (_pending.get(user_id) or {}).pop(task_id, None)
    if fut is None:
        # Either the agent already gave up, or this is a replay from the
        # device's outbox for a task that has since been answered. Both are
        # expected (§5.4) and neither is an error — an id resolves at most
        # once.
        for ep in _active_eps(user_id):
            ep.counters["orphan_results"] += 1
        return
    if fut.done():
        return

    status = msg.get("status")
    if status not in RESULT_STATUSES:
        # `ok` is documented as DERIVED from `status`, never set apart from
        # it (§5.4). A frame with no legible status is malformed, and
        # guessing from `ok` would be exactly that inversion.
        fut.set_exception(DesktopError(
            "error", "the Mac sent a result this server could not read",
            code="bad_result",
        ))
        return

    summary = _clean_text(msg.get("summary"), 400)
    data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
    code = str((data or {}).get("code") or "")[:40]

    if status == "ok":
        fut.set_result({
            "status": "ok",
            "summary": summary,
            "data": data,
            "took_ms": msg.get("took_ms"),
        })
    elif status == "denied":
        fut.set_exception(DesktopDenied(summary, code))
    else:
        fut.set_exception(DesktopError(status, summary, code))


def _deliver_event(user_id: str, msg: Dict[str, Any]) -> None:
    """§5.6. Metadata only — and this function is where that is enforced.

    Only four keys are kept, so a device (or a compromised one) cannot use
    the activity channel to stream file contents: there is nowhere to put
    them. Nothing awaits an event, so events are not kept across a
    reconnect.
    """
    kind = msg.get("kind")
    if kind not in EVENT_KINDS:
        for ep in _active_eps(user_id):
            ep.counters["unknown_events"] += 1
        return
    _recent_events[user_id].append({
        "kind": kind,
        "id": str(msg.get("id") or "")[:64],
        "tool": str(msg.get("tool") or "")[:96],
        "status": str(msg.get("status") or "")[:24],
        "at": int(time.time() * 1000),
    })
    if kind in ("approval_pending", "approval_resolved"):
        _forward_approval_event(user_id, str(msg.get("id") or ""), kind)


def _forward_approval_event(user_id: str, relay_id: str, kind: str) -> None:
    """Tell the platform the Mac is asking locally about an approved action.

    Only for ids this process dispatched for a card, and only for the user
    that dispatched it: an event naming another tenant's id, or an id we
    never sent, is dropped.
    """
    known = _relay_ids.get(relay_id)
    hook = approval_event_hook
    if not known or hook is None or known[2] != user_id:
        return
    action_id, remote_task_id, _ = known
    try:
        from app.services.background_tasks import spawn

        spawn(hook(user_id, action_id, remote_task_id, kind),
              name=f"desktop-approval-event-{action_id[:8]}")
    except RuntimeError:
        pass


def _clean_text(value: Any, limit: int) -> str:
    """Flatten a device-supplied display string.

    §5.3 requires this of the *device* for `reason`; the same treatment is
    owed in this direction. A summary is rendered in a chat card and read
    aloud, and newlines let one sentence impersonate three lines of
    app-authored UI.
    """
    if not isinstance(value, str):
        return ""
    flat = " ".join(value.split())
    return flat[:limit]


# ─── Outbound: dispatch ────────────────────────────────────────────────
async def dispatch(
    user_id: str,
    action: str,
    params: Dict[str, Any],
    timeout_s: float = 30.0,
    *,
    task_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    reason: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send one task and await the device's result.

    `task_id` is the seam's idempotency key (§5.3): pass one to REPLAY a
    task (an approved pending action reuses the id it was staged with, so a
    redelivery cannot run `rm` twice), and leave it None for a fresh call.

    `device_id` BINDS the task to that Mac (CONNECTIONS.md M2): if it is not
    on a socket this raises `DesktopUnavailable` rather than using another
    of the user's Macs. With no `device_id` the newest socket is used, which
    is what an unpinned chat read has always meant.

    Raises `DesktopUnavailable`, `DesktopDenied`, `DesktopError`, or
    `asyncio.TimeoutError`.
    """
    if not _valid_action(action):
        raise DesktopError(
            "invalid_arguments", f"action {action!r} is not a legal tool name",
            code="invalid_task",
        )
    if not isinstance(params, dict):
        # §5.3: `params` must be a JSON object. Refused here rather than on
        # the device so a coding error never reaches the Mac at all.
        raise DesktopError(
            "invalid_arguments", "params must be an object", code="invalid_task",
        )

    eps = _active_eps(user_id)
    if not eps:
        raise DesktopUnavailable("no Mac connected for this user")

    in_flight = len(_pending.get(user_id) or {})
    if in_flight >= MAX_TASKS_IN_FLIGHT:
        # Refused HERE as well as device-side: the device answers `busy`
        # past 4 in flight, and asking it to do so costs a round trip and a
        # queue slot on a machine we already know is saturated.
        raise DesktopError(
            "error",
            f"{in_flight} local tasks are already running on this Mac",
            code="busy",
        )

    if device_id is not None:
        ep = _endpoint_for(user_id, device_id)
        if ep is None:
            raise DesktopUnavailable("that Mac is not connected")
    else:
        ep = eps[-1]
    _check_rate_limit(ep)

    tid = task_id or uuid.uuid4().hex
    frame: Dict[str, Any] = {
        "type": "task",
        "id": tid,
        "action": action,
        "params": params,
        "timeout_ms": int(max(1.0, timeout_s) * 1000),
    }
    if turn_id:
        frame["turn_id"] = str(turn_id)[:64]
    if reason:
        frame["reason"] = _clean_text(reason, REASON_MAX_LEN)
    # NOTE: no account field, deliberately (§5.3). The device stamps the
    # request with the account it is paired to. If the account travelled in
    # the frame, a relay pointed at the wrong tenant could aim a tool call
    # at another user's grants, and the device's "must equal the paired
    # account" check would be comparing a value to itself.

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    _pending[user_id][tid] = fut
    try:
        await ep.ws.send_text(json.dumps(frame))
        ep.counters["tasks_sent"] += 1
    except Exception as exc:
        _pending[user_id].pop(tid, None)
        logger.warning("[desktop-bridge] send failed user=%s: %s", user_id[:8], exc)
        raise DesktopUnavailable(f"send failed: {exc}")

    try:
        return await asyncio.wait_for(fut, timeout=timeout_s + _TASK_DEADLINE_SLACK_S)
    finally:
        # Resolve-at-most-once is the `pop` in `_deliver_result`; this is
        # the leak guard for the timeout path.
        _pending.get(user_id, {}).pop(tid, None)


def _valid_action(action: Any) -> bool:
    return (
        isinstance(action, str)
        and 1 <= len(action) <= ACTION_MAX_LEN
        and set(action) <= _ACTION_CHARS
    )


async def cancel(user_id: str, task_id: Optional[str] = None) -> bool:
    """§5.5. A targeted cancel is never widened into 'cancel everything'.

    So `task_id=None` is only honoured when the caller passed None: it is
    not a fallback for an id that failed to resolve.
    """
    eps = _active_eps(user_id)
    if not eps:
        return False
    frame: Dict[str, Any] = {"type": "cancel"}
    if task_id:
        frame["id"] = str(task_id)[:64]
    for ep in reversed(eps):
        try:
            await ep.ws.send_text(json.dumps(frame))
            return True
        except Exception:
            continue
    return False


async def close_for_device(
    user_id: str, *, device_id: Optional[str] = None,
    token_jti: Optional[str] = None, code: int = 4003,
) -> int:
    """Close live socket(s) for one revoked device. Returns how many closed.

    This is the third of the three things `RELAY_PROTOCOL.md` §1.4 requires
    of a revoke, and the one the extension never did: "mark the row revoked"
    and "denylist the jti" both leave a socket that is already open still
    reading tasks. `4003` is documented as never retried, so the app deletes
    its token and stops rather than reconnecting into a refusal loop.

    Matching is by device id OR jti; with neither, every socket for the user
    closes — which is what "revoke all my Macs" means.
    """
    closed = 0
    for ep in _active_eps(user_id):
        if device_id and ep.device_id and ep.device_id != device_id:
            continue
        if token_jti and ep.token_jti and ep.token_jti != token_jti:
            continue
        try:
            await ep.ws.close(code=code, reason="device revoked")
            closed += 1
        except Exception as exc:
            logger.debug("[desktop-bridge] close failed: %s", exc)
        await unregister(user_id, ep)
    if closed:
        logger.info(
            "[desktop-bridge] closed %d socket(s) for revoked device user=%s",
            closed, user_id[:8],
        )
    return closed


def _check_rate_limit(ep: _Endpoint) -> None:
    now = time.time()
    while ep.recent_tasks and ep.recent_tasks[0] < now - RATE_LIMIT_WINDOW_S:
        ep.recent_tasks.popleft()
    if len(ep.recent_tasks) >= RATE_LIMIT_PER_WINDOW:
        raise DesktopError(
            "error",
            f"more than {RATE_LIMIT_PER_WINDOW} local tasks in "
            f"{int(RATE_LIMIT_WINDOW_S)}s",
            code="rate_limited",
        )
    ep.recent_tasks.append(now)


def reset_for_tests() -> None:
    """Drop all registry state. TEST ONLY."""
    _endpoints_by_user.clear()
    _pending.clear()
    _recent_events.clear()
    _relay_ids.clear()
    _task_relay_ids.clear()
    _cancelled_tasks.clear()
