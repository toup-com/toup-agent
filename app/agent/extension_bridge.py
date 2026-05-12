"""
Extension bridge — v2.

Adds on top of v1:
  - Session registry: session_id → owner user_id (so SSE can authorize
    by current user without round-tripping to the extension).
  - Event router: extension's unsolicited "event" frames are fanned out
    to local asyncio Queue subscribers (one per open SSE connection).
  - Per-session ring buffer for SSE Last-Event-ID replay.
  - Control-frame sender (pause/resume) used by the /settings/extensions
    "Take back control" button.
  - send_control() helper used by FastAPI routes.

v1 dispatch / register / unregister surface is preserved.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# ─── Limits ─────────────────────────────────────────────────────────
RATE_LIMIT_PER_MIN = 60
EVENT_BUFFER_PER_SESSION = 100
TASK_TRACE_RETAIN = 6                  # how many recent traces to keep per session

# Side-effect action kinds — these require user confirmation in the
# sidepanel before the extension actually executes them. Read-only
# kinds run silently.
SIDE_EFFECT_KINDS = {"navigate", "click", "type", "scroll", "select", "evaluate", "extract"}


# ─── Exceptions ─────────────────────────────────────────────────────
class ExtensionUnavailable(Exception):
    """No extension is currently connected for this user."""


class ExtensionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ─── Per-connection endpoint state ──────────────────────────────────
@dataclass
class _Endpoint:
    user_id: str
    ws: WebSocket
    connected_at: float = field(default_factory=time.time)
    schema_version: int = 2
    pending: Dict[str, asyncio.Future] = field(default_factory=dict)
    recent_tasks: deque = field(default_factory=deque)


# ─── Session metadata mirrored on the backend ────────────────────────
@dataclass
class _Session:
    session_id: str
    user_id: str
    name: str
    created_at: float
    last_event_at: float
    status: str = "ready"               # ready | paused | ended
    last_url: Optional[str] = None
    last_title: Optional[str] = None
    last_screenshot_b64: Optional[str] = None
    last_screenshot_meta: Dict[str, Any] = field(default_factory=dict)
    # Ring buffer of (seq, event_dict) for SSE replay.
    events: deque = field(default_factory=lambda: deque(maxlen=EVENT_BUFFER_PER_SESSION))
    next_seq: int = 0
    # Phase 5 — per-task accumulation. We bucket action_started /
    # navigation / action_complete events under their task_id and emit
    # a final `task_summary` event when the task ends.
    task_steps: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    task_started: Dict[str, float] = field(default_factory=dict)
    recent_traces: deque = field(default_factory=lambda: deque(maxlen=TASK_TRACE_RETAIN))


_endpoints: Dict[str, _Endpoint] = {}                  # user_id → endpoint
_sessions:  Dict[str, _Session]  = {}                  # session_id → session
_sessions_by_user: Dict[str, set] = defaultdict(set)   # user_id → {session_id}
_subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)  # session_id → [queue, ...]
_lock = asyncio.Lock()


# ─── Endpoint lifecycle ─────────────────────────────────────────────
async def register(user_id: str, ws: WebSocket) -> _Endpoint:
    async with _lock:
        prior = _endpoints.pop(user_id, None)
        if prior:
            logger.info("[ext-bridge] replacing prior connection for %s", user_id)
            for fut in prior.pending.values():
                if not fut.done():
                    fut.set_exception(ExtensionUnavailable("connection replaced"))
            try: await prior.ws.close(code=4002, reason="replaced by newer pairing")
            except Exception: pass
        ep = _Endpoint(user_id=user_id, ws=ws)
        _endpoints[user_id] = ep
        logger.info("[ext-bridge] registered %s (total=%d)", user_id, len(_endpoints))
        return ep


async def unregister(user_id: str, ep: Optional[_Endpoint] = None) -> None:
    async with _lock:
        existing = _endpoints.get(user_id)
        if existing and (ep is None or existing is ep):
            del _endpoints[user_id]
            for fut in existing.pending.values():
                if not fut.done():
                    fut.set_exception(ExtensionUnavailable("extension disconnected"))
            # Mark all of user's sessions as "ended" but keep them visible to
            # subscribers briefly so they see the closed event.
            for sid in list(_sessions_by_user.get(user_id, [])):
                sess = _sessions.get(sid)
                if sess:
                    sess.status = "ended"
                    _push_event_locked(sess, {"kind": "closed", "data": {"reason": "extension_disconnected"}})
            logger.info("[ext-bridge] unregistered %s", user_id)


def is_connected(user_id: str) -> bool:
    return user_id in _endpoints


def stats() -> Dict[str, Any]:
    return {
        "active_users": len(_endpoints),
        "active_sessions": len(_sessions),
        "users": [
            {
                "user_id": ep.user_id,
                "connected_for_s": int(time.time() - ep.connected_at),
                "pending_tasks": len(ep.pending),
                "sessions": list(_sessions_by_user.get(ep.user_id, [])),
            }
            for ep in _endpoints.values()
        ],
    }


# ─── Inbound message routing ─────────────────────────────────────────
def deliver_inbound(user_id: str, msg: Dict[str, Any]) -> None:
    """Route inbound messages to whoever's waiting."""
    mtype = msg.get("type")
    if mtype == "result":
        _deliver_result(user_id, msg)
    elif mtype == "event":
        _deliver_event(user_id, msg)
    elif mtype == "progress":
        sid = (msg.get("data") or {}).get("session_id")
        if sid:
            sess = _sessions.get(sid)
            if sess: _push_event(sess, {"kind": "progress", "data": msg.get("data") or {}})


def _deliver_result(user_id: str, msg: Dict[str, Any]) -> None:
    ep = _endpoints.get(user_id)
    if not ep: return
    task_id = msg.get("id")
    fut = ep.pending.pop(task_id, None) if task_id else None
    if not fut or fut.done(): return
    if msg.get("ok"):
        data = msg.get("data") or {}
        _maybe_update_session_from_result(user_id, data)
        fut.set_result(data)
    else:
        err = msg.get("error") or {}
        fut.set_exception(ExtensionError(
            code=err.get("code", "INTERNAL"),
            message=err.get("message", "unspecified extension error"),
        ))


def _maybe_update_session_from_result(user_id: str, data: Dict[str, Any]) -> None:
    """Side-effect: keep session mirror fresh from action results."""
    sid = data.get("session_id")
    if not sid and isinstance(data.get("tab_state"), dict):
        # action result; session_id is in the original task — we infer
        # from URL match.
        return
    # session_start path: data has session_id, url, title at top.
    if sid:
        sess = _sessions.get(sid)
        if not sess:
            sess = _Session(
                session_id=sid,
                user_id=user_id,
                name=data.get("name") or "session",
                created_at=time.time(),
                last_event_at=time.time(),
            )
            _sessions[sid] = sess
            _sessions_by_user[user_id].add(sid)
        sess.last_url   = data.get("url",   sess.last_url)
        sess.last_title = data.get("title", sess.last_title)
        if "image_b64" in data:
            sess.last_screenshot_b64  = data["image_b64"]
            sess.last_screenshot_meta = {
                "format": data.get("format"), "w": data.get("width"),
                "h": data.get("height"), "quality": data.get("quality"),
                "captured_at": data.get("captured_at"),
            }
    # action result with tab_state: update url/title.
    if isinstance(data.get("tab_state"), dict):
        # walk through all sessions for the user to find the one currently active
        ts = data["tab_state"]
        for s in (_sessions.get(x) for x in _sessions_by_user.get(user_id, [])):
            if not s: continue
            # Heuristic: latest_event session = the one we're updating.
            s.last_url   = ts.get("url",   s.last_url)
            s.last_title = ts.get("title", s.last_title)
        if "screenshot" in data:
            for s in (_sessions.get(x) for x in _sessions_by_user.get(user_id, [])):
                if s:
                    s.last_screenshot_b64  = data["screenshot"]
                    s.last_screenshot_meta = data.get("screenshot_meta") or {}


def _deliver_event(user_id: str, msg: Dict[str, Any]) -> None:
    sid = msg.get("session_id")
    kind = msg.get("kind")
    data = msg.get("data") or {}
    if not sid or not kind: return

    sess = _sessions.get(sid)
    if not sess:
        # session_started arrives before result for session_start in some
        # races — synthesize the record.
        if kind in ("session_started", "tab_state"):
            sess = _Session(
                session_id=sid, user_id=user_id, name=data.get("name") or "session",
                created_at=time.time(), last_event_at=time.time(),
            )
            _sessions[sid] = sess
            _sessions_by_user[user_id].add(sid)
        else:
            return

    sess.last_event_at = time.time()
    if kind == "navigation":
        sess.last_url   = data.get("url",   sess.last_url)
        sess.last_title = data.get("title", sess.last_title)
    elif kind == "paused":
        sess.status = "paused"
    elif kind == "resumed":
        sess.status = "ready"
    elif kind == "closed":
        sess.status = "ended"

    _push_event(sess, {"kind": kind, "data": data})
    _accumulate_trace(sess, kind, data)

    # If the session is ended, schedule eviction after a short delay so
    # SSE clients can flush.
    if sess.status == "ended":
        asyncio.create_task(_evict_after(sid, 5.0))


def _push_event(sess: _Session, event: Dict[str, Any]) -> None:
    _push_event_locked(sess, event)


# ─── Phase 5: per-task trace assembly ───────────────────────────────
def _accumulate_trace(sess: _Session, kind: str, data: Dict[str, Any]) -> None:
    """
    Group inbound events by task_id. When a task finishes, emit a
    consolidated `task_summary` event so the sidepanel can render
    a "what I did" trace.
    """
    if kind == "action_started":
        task_id = data.get("task_id")
        if not task_id: return
        sess.task_started.setdefault(task_id, time.time())
        sess.task_steps.setdefault(task_id, []).append({
            "phase":   "started",
            "kind":    data.get("kind"),
            "ts":      int(time.time() * 1000),
            "url":     sess.last_url,
            "title":   sess.last_title,
        })
        return

    if kind == "navigation":
        # Attach to the most recent open task, if any.
        if not sess.task_started: return
        last_task = next(reversed(sess.task_started))
        sess.task_steps.setdefault(last_task, []).append({
            "phase":   "navigation",
            "url":     data.get("url"),
            "title":   data.get("title"),
            "ts":      int(time.time() * 1000),
        })
        return

    if kind == "action_complete":
        task_id = data.get("task_id")
        if not task_id: return
        steps = sess.task_steps.get(task_id) or []
        started = sess.task_started.pop(task_id, None)
        steps.append({
            "phase":    "complete",
            "kind":     data.get("kind"),
            "ok":       data.get("ok"),
            "took_ms":  data.get("took_ms"),
            "ts":       int(time.time() * 1000),
        })
        sess.task_steps.pop(task_id, None)
        # Build a compact trace.
        trace = {
            "task_id":    task_id,
            "started_at": int((started or time.time()) * 1000),
            "ended_at":   int(time.time() * 1000),
            "duration_ms": int(((time.time() - (started or time.time())) * 1000)),
            "steps":      steps,
            "page_count": len({s.get("url") for s in steps if s.get("url")}),
        }
        sess.recent_traces.append(trace)
        _push_event(sess, {"kind": "task_summary", "data": trace})
        # Phase 6: persist into the browsing memory (fire-and-forget).
        asyncio.create_task(_persist_browsing_memory(sess.user_id, sess.session_id, trace))


async def _persist_browsing_memory(user_id: str, session_id: str, trace: Dict[str, Any]) -> None:
    try:
        from app.services.browsing_memory import record_task, derive_primary_from_trace
        primary = derive_primary_from_trace(trace)
        # Compact assistant-facing summary line for the suggest prompt.
        kinds = [s.get("kind") for s in (trace.get("steps") or []) if s.get("kind")]
        summary = ", ".join([k for k in kinds if k]) or None
        await record_task(
            user_id=user_id,
            session_id=session_id,
            task_id=trace.get("task_id"),
            primary_url=primary["primary_url"],
            primary_title=primary["primary_title"],
            page_count=primary["page_count"],
            duration_ms=primary["duration_ms"],
            agent_summary=summary,
            visits=primary["visits"],
        )
    except Exception as exc:
        logger.warning("[ext-bridge] browsing-memory write failed: %s", exc)


def latest_traces(user_id: str, session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
    sess = _sessions.get(session_id)
    if not sess or sess.user_id != user_id: return []
    return list(sess.recent_traces)[-limit:]


# ─── Phase 5: notify the user out-of-band when a long task ends ────
async def notify_task_complete(
    *,
    user_id: str,
    delivery_channels: List[str],
    routine_name: str,
    summary: str,
) -> None:
    """
    Fire the existing routine delivery fan-out (telegram / whatsapp /
    website) when a long-running agent browser task finishes while the
    user is away from the sidepanel. Best-effort — exceptions logged
    and swallowed.
    """
    if not delivery_channels:
        return
    try:
        from app.agent.routines.channel_dispatcher import deliver_to_extra_channels
        from app.db.database import async_session_maker
        await deliver_to_extra_channels(
            user_id=user_id,
            delivery_channels=delivery_channels,
            routine_name=routine_name or "Browser task",
            content=summary,
            db_session_maker=async_session_maker,
        )
    except Exception as exc:
        logger.warning("[ext-bridge] notify_task_complete failed: %s", exc)


def _push_event_locked(sess: _Session, event: Dict[str, Any]) -> None:
    sess.next_seq += 1
    seq = sess.next_seq
    item = {"seq": seq, "ts": int(time.time() * 1000), **event}
    sess.events.append(item)
    for q in list(_subscribers.get(sess.session_id, [])):
        try: q.put_nowait(item)
        except asyncio.QueueFull: pass


async def _evict_after(sid: str, delay_s: float) -> None:
    try: await asyncio.sleep(delay_s)
    except asyncio.CancelledError: return
    sess = _sessions.pop(sid, None)
    if sess:
        _sessions_by_user.get(sess.user_id, set()).discard(sid)
    for q in list(_subscribers.get(sid, [])):
        try: q.put_nowait(None)  # sentinel → close stream
        except asyncio.QueueFull: pass
    _subscribers.pop(sid, None)


# ─── Outbound: task dispatch ────────────────────────────────────────
async def dispatch(
    user_id: str,
    action: str,
    params: Dict[str, Any],
    timeout_s: int = 30,
) -> Dict[str, Any]:
    ep = _endpoints.get(user_id)
    if not ep:
        raise ExtensionUnavailable("no connected extension for user")
    _check_rate_limit(ep)

    task_id = uuid.uuid4().hex
    payload = {
        "type":       "task",
        "id":         task_id,
        "action":     action,
        "params":     params,
        "timeout_ms": int(timeout_s * 1000),
    }
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    ep.pending[task_id] = fut
    try:
        await ep.ws.send_text(json.dumps(payload))
    except Exception as exc:
        ep.pending.pop(task_id, None)
        logger.warning("[ext-bridge] send failed for %s: %s", user_id, exc)
        raise ExtensionUnavailable(f"send failed: {exc}")
    try:
        return await asyncio.wait_for(fut, timeout=timeout_s + 2)
    finally:
        ep.pending.pop(task_id, None)


# ─── Outbound: fire-and-forget control frames ───────────────────────
async def send_control(user_id: str, session_id: str, control: str) -> None:
    ep = _endpoints.get(user_id)
    if not ep:
        raise ExtensionUnavailable("extension not connected")
    payload = {"type": "control", "session_id": session_id, "control": control}
    try:
        await ep.ws.send_text(json.dumps(payload))
    except Exception as exc:
        raise ExtensionUnavailable(f"send_control failed: {exc}")


def _check_rate_limit(ep: _Endpoint) -> None:
    window = 60.0
    now = time.time()
    while ep.recent_tasks and ep.recent_tasks[0] < now - window:
        ep.recent_tasks.popleft()
    if len(ep.recent_tasks) >= RATE_LIMIT_PER_MIN:
        raise ExtensionError("RATE_LIMITED", f"per-minute limit ({RATE_LIMIT_PER_MIN}) exceeded")
    ep.recent_tasks.append(now)


# ─── Session inspection (for UI) ────────────────────────────────────
def list_sessions_for_user(user_id: str) -> List[Dict[str, Any]]:
    out = []
    for sid in _sessions_by_user.get(user_id, set()):
        sess = _sessions.get(sid)
        if not sess: continue
        out.append({
            "session_id":    sess.session_id,
            "name":          sess.name,
            "status":        sess.status,
            "url":           sess.last_url,
            "title":         sess.last_title,
            "created_at":    sess.created_at,
            "last_event_at": sess.last_event_at,
        })
    return out


def get_session(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    sess = _sessions.get(session_id)
    if not sess or sess.user_id != user_id:
        return None
    return {
        "session_id":    sess.session_id,
        "name":          sess.name,
        "status":        sess.status,
        "url":           sess.last_url,
        "title":         sess.last_title,
        "created_at":    sess.created_at,
        "last_event_at": sess.last_event_at,
        "screenshot": {
            "b64":  sess.last_screenshot_b64,
            "meta": sess.last_screenshot_meta,
        } if sess.last_screenshot_b64 else None,
    }


def latest_screenshot(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    sess = _sessions.get(session_id)
    if not sess or sess.user_id != user_id or not sess.last_screenshot_b64:
        return None
    return {"b64": sess.last_screenshot_b64, **(sess.last_screenshot_meta or {})}


# ─── SSE subscription ───────────────────────────────────────────────
async def subscribe(user_id: str, session_id: str, last_event_id: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]:
    """
    Async iterator of events for a session. First yield is the current
    state snapshot, then any buffered events newer than last_event_id,
    then live events as they arrive. Yields None to signal close.
    """
    sess = _sessions.get(session_id)
    if not sess or sess.user_id != user_id:
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers[session_id].append(queue)
    try:
        # 1. Snapshot frame
        yield {
            "_event": "state",
            "data": get_session(user_id, session_id),
        }
        # 2. Replay buffered events
        if last_event_id is not None:
            for item in list(sess.events):
                if item["seq"] > last_event_id:
                    yield {"_event": "event", "data": item}
        # 3. Live forward
        while True:
            item = await queue.get()
            if item is None:
                yield {"_event": "end", "data": {}}
                return
            yield {"_event": "event", "data": item}
    finally:
        try: _subscribers[session_id].remove(queue)
        except (KeyError, ValueError): pass
