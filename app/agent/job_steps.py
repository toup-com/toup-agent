"""The step list of a ``create_job`` job — state transitions and timing.

``BuildJob.steps_json`` is a list of ``{id, type, label, status}`` dicts the
model declares up front. Round 8 adds honest per-step timing to the same
dicts, written by the ONE set of rules below and read by every client
(the app renders ``durationMs`` when the server provides it — before this the
clients inferred a step's window from ``update_job`` tick timestamps, and a
tick that closed two steps at once printed the skipped one as "0ms").

Keys added per step (all optional — older rows and older clients are fine):

* ``started_at`` / ``completed_at`` — ISO-8601 UTC, when the step became
  the running step / was marked done.
* ``duration_ms`` (and its camelCase twin ``durationMs``, which is the key
  the mobile ``JobStep`` type reads) — ``completed_at - started_at``.

Rules
-----
* ``create``: step 0 is RUNNING from the moment the job exists (it was
  ``pending`` before, and the clients had to guess which step was live).
* ``advance(k)`` (``update_job(current_step=k)``): steps 0..k are done,
  k+1 becomes running. A step that was running gets its real window. A
  step closed by the same tick WITHOUT ever having been the running step
  (the model closed two steps at once) SHARES the window it was closed in
  — the honest statement "this step completed inside this window", never
  a fabricated split and never zero.
* ``finish_all`` (turn-end finalizer / reconciler / ``update_job(completed)``):
  every step not yet done becomes done at ``now``; the running step gets
  its real window, never-run steps share the last window.
* A step's window can only be stamped once; re-running a rule is idempotent.

Pure functions over plain dicts; no DB, no clock beyond the ``now`` passed in.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "parse_steps", "dump_steps", "open_first_step", "advance_steps",
    "finish_all_steps", "counts", "current_label",
]


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=(dt.microsecond // 1000) * 1000).isoformat()


def _parse_iso(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def parse_steps(steps_json: Optional[str]) -> List[Dict[str, Any]]:
    """The step list, or ``[]`` for a missing / malformed column."""
    if not steps_json:
        return []
    try:
        steps = json.loads(steps_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict)]


def dump_steps(steps: List[Dict[str, Any]]) -> str:
    return json.dumps(steps)


def _stamp_close(
    step: Dict[str, Any], started: Optional[datetime], now: datetime,
    measured: bool = True,
) -> None:
    """Close ``step`` at ``now``, opening it at ``started`` when it never
    opened. Idempotent: an already-closed step keeps its window.

    ``measured=False`` closes the step WITHOUT a duration: the step is
    done, and how long it took is not known. Writing a number there is
    what put the job's whole elapsed time on every one of its steps.
    """
    if step.get("completed_at"):
        return
    if not measured:
        step["completed_at"] = _iso(now)
        return
    if not step.get("started_at"):
        step["started_at"] = _iso(started or now)
    step["completed_at"] = _iso(now)
    s = _parse_iso(step["started_at"])
    dur = int(max(0.0, (now - s).total_seconds() * 1000)) if s else 0
    step["duration_ms"] = dur
    step["durationMs"] = dur


def open_first_step(steps: List[Dict[str, Any]], now: datetime) -> List[Dict[str, Any]]:
    """``create_job``: step 0 is running from creation."""
    if steps:
        first = steps[0]
        if first.get("status") in (None, "pending"):
            first["status"] = "running"
        first.setdefault("started_at", _iso(now))
    return steps


def _last_window_start(
    steps: List[Dict[str, Any]], upto: int, now: datetime,
    fallback: Optional[datetime] = None,
) -> datetime:
    """When the window that is closing right now opened: the running step's
    ``started_at`` if one is running among 0..upto, else the newest
    ``completed_at`` before it, else ``fallback`` (the job's own
    ``created_at`` — rows written before timing existed carry no stamps at
    all, and closing them at a zero-width window would print the very 0ms
    this module removes), else ``now``."""
    for s in steps[: upto + 1]:
        if s.get("status") == "running":
            st = _parse_iso(s.get("started_at"))
            if st:
                return st
    latest: Optional[datetime] = None
    for s in steps[: upto + 1]:
        c = _parse_iso(s.get("completed_at"))
        if c and (latest is None or c > latest):
            latest = c
    if latest:
        return latest
    if fallback and fallback <= now:
        return fallback
    return now


def advance_steps(
    steps: List[Dict[str, Any]], current_step: int, now: datetime,
    fallback_start: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """``update_job(current_step=k)``: 0..k done, k+1 running (see module
    docstring for the timing rules). ``fallback_start`` = the job's
    ``created_at``, the window start for rows that carry no stamps yet."""
    if not steps or current_step is None or current_step < 0:
        return steps
    k = min(int(current_step), len(steps) - 1)
    window_start = _last_window_start(steps, k, now, fallback_start)
    for i, s in enumerate(steps):
        if i <= k:
            if s.get("status") != "done":
                # MEASURED, always — deliberately NOT `finish_all_steps`'s
                # owner rule. An `update_job` tick closes inside a REAL,
                # observed window (the running step's own run, typically a
                # second or two), so a step the tick jumped over sharing it
                # is an honest approximation and the round-8 contract this
                # path was built for: "never zero, never unknown". The
                # window at TURN END is a different animal — there it is
                # the whole job, which is why only that one has an owner.
                s["status"] = "done"
                _stamp_close(s, window_start, now)
        elif i == k + 1:
            if s.get("status") != "done":
                s["status"] = "running"
                s.setdefault("started_at", _iso(now))
    return steps


def finish_all_steps(
    steps: List[Dict[str, Any]], now: datetime,
    fallback_start: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Every remaining step done at ``now`` — the contract's "the system marks
    every remaining step done ... the moment your reply is delivered".
    ``fallback_start`` = the job's ``created_at`` (legacy rows, see
    ``_last_window_start``)."""
    if not steps:
        return steps
    window_start = _last_window_start(steps, len(steps) - 1, now, fallback_start)
    # ── One window belongs to ONE step (round 33, item 1) ───────────────
    # This closed every outstanding step inside the SAME window, on the
    # rule "a step closed in this window shares it". For a turn where the
    # model never called `update_job` — the ordinary case, because the
    # speed contract forbids calling one alone — that window is the whole
    # job, so a three-step plan reported 16.2s, 16.2s and 16.2s: the job's
    # own total, printed three times as if each step had taken it.
    #
    # `_last_window_start` answers when the window now closing OPENED, and
    # exactly one step can honestly claim it:
    #   • the step that was RUNNING, if one was — it is the step that has
    #     been in that window; nothing after it ever started; or
    #   • when none was running, the FIRST step being closed — it began
    #     when its predecessor finished, which is a real edge.
    # Every other step is closed WITHOUT a duration. A duration nobody
    # measured is unknown, not the enclosing window, and every client
    # already renders nothing for an absent one (`formatDurationMs`
    # answers null).
    running_idx = next(
        (i for i, s in enumerate(steps) if s.get("status") == "running"), None,
    )
    owner_idx = running_idx if running_idx is not None else next(
        (i for i, s in enumerate(steps) if s.get("status") != "done"), None,
    )
    for i, s in enumerate(steps):
        if s.get("status") != "done":
            owns = i == owner_idx
            s["status"] = "done"
            _stamp_close(s, window_start if owns else None, now,
                         measured=owns)
    return steps


def counts(steps: List[Dict[str, Any]]) -> Tuple[int, int]:
    """(done, total)."""
    return sum(1 for s in steps if s.get("status") == "done"), len(steps)


def current_label(steps: List[Dict[str, Any]]) -> str:
    """The step happening now: the running one, else the last."""
    for s in steps:
        if s.get("status") == "running":
            return str(s.get("label") or "")
    return str(steps[-1].get("label") or "") if steps else ""
