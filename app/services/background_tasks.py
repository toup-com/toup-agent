"""One place to launch fire-and-forget work so it cannot silently vanish.

WHY THIS EXISTS

`asyncio.create_task(...)` returns a task the event loop holds only a WEAK
reference to. If nobody stores the result, the task can be garbage-collected
mid-await — the hazard CPython's own docs call out. Two things follow, and
both have cost us production incidents:

1. **A dropped task abandons whatever it was holding.** These call sites open
   DB sessions; a task collected inside `async with async_session_maker()`
   never runs `__aexit__`, so the connection is never checked in. That is the
   `SAWarning: The garbage collector is trying to clean up non-checked-in
   connection` seen on the canary, and left unchecked it ends as
   `PendingRollbackError` and HTTP 500 on every chat turn (2026-07-04,
   2026-08-01).
2. **The work silently does not happen.** A vanished post-processing task is a
   turn whose memories were never extracted, with nothing in the logs saying
   so. That is the quieter and arguably worse failure.

This module also fixes a third gap the ad-hoc helpers shared: none of them
looked at the task's exception, so a background task that RAISED died in
silence too. `spawn` logs it.

USE THIS instead of a bare `asyncio.create_task` whenever the coroutine owns a
DB session, performs writes, or its not-running would be a bug. A task that is
genuinely disposable (a stream pump owned by a process object that already
holds the reference) does not need it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, Optional, Set

logger = logging.getLogger(__name__)

# Strong references to in-flight background tasks. Entries are discarded in the
# done-callback, so this never grows without bound — it holds exactly the tasks
# that are currently running.
_background_tasks: Set[asyncio.Task] = set()


def _on_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        # Cancellation is a normal shutdown path; not an error.
        return
    exc = task.exception()
    if exc is not None:
        # Previously this went nowhere: a background task that raised failed
        # silently. Log it loudly enough to be greppable.
        logger.error(
            "[background] task %r failed: %s: %s",
            task.get_name(), type(exc).__name__, exc,
            exc_info=exc,
        )


def spawn(coro: Coroutine[Any, Any, Any], *, name: Optional[str] = None) -> asyncio.Task:
    """Launch `coro` as a background task that will not be garbage-collected.

    Returns the task so callers that DO want to await or cancel it still can;
    callers that ignore the return value are safe, which is the whole point.
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_on_done)
    return task


def pending_count() -> int:
    """How many background tasks are in flight. For health/diagnostics."""
    return len(_background_tasks)
