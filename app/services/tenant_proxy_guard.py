"""Load shedding for the platform's read proxies to a stalled tenant (PR-15).

THE INCIDENT (2026-10-02, founder tenant 871bac24, FINDINGS §3)
    The tenant's event loop stalled for long stretches. Every platform read
    proxy kept dialling it anyway: the app's job board polled `GET
    /api/apps/jobs/` and the autopilot missions list every 5 s, and each of
    those tried the tunnel (15 s) and then re-dialled over HTTPS — even right
    after a tunnel forward to the same tenant had just timed out. The
    day-chats and sessions proxies ran their own 2 x 6 s ladders on top.
    Between 02:28:34 and 02:32:00 that was 78 jobs and 78 missions requests,
    each one more queued work for a process that could not keep up, and five
    queued tunnel forwards came back in the same millisecond once it did.

WHAT THIS MODULE DOES
    Two per-tenant, in-process guards, consulted by the guarded read proxies
    only (`apps_proxy`'s job/app reads, `autopilot_proxy` GETs,
    `day_chats._proxy_day_chats`, `sessions._proxy_sessions`):

    * A "recently timed out" memo. When a tunnel forward
      (`ws_agent_tunnel.send_http_forward`) or a guarded proxied read to a
      tenant times out, the tenant is marked for
      `settings.tenant_proxy_timeout_memo_s` seconds. While it is marked, a
      guarded read answers 503 `agent_unreachable` at once, with a
      `Retry-After` of the seconds left — no tunnel, no HTTPS re-dial. The
      app honours a `Retry-After` of up to 10 s, so its next try lands after
      the window. A guarded read that the tenant answers clears the mark.
    * A cap on concurrent guarded reads in flight to one tenant
      (`settings.tenant_proxy_max_inflight`). Past it the read answers 503
      `agent_unreachable` at once rather than queueing behind the others.

WHAT IT NEVER TOUCHES
    Health probes (`agent_setup`'s `/agent/health` calls, `apps_proxy`'s
    `/server` status probe), writes (a refused write would lose the user's
    intent; writes are rare and are not the pile-up) and the voice relay's own
    tenant calls (`ws_realtime._vps_api*`, `live_voice_protocol`) never consult
    either guard: the guard is opt-in per proxy helper, and those paths do not
    opt in. `tests/test_tenant_proxy_guard.py` pins all three.

    The state is per process, like `ws_agent_tunnel._tunnels`: each platform
    replica learns about a stalled tenant on its own. It is a load-shedding
    hint, never an authority on whether the tenant is up.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

logger = logging.getLogger(__name__)

#: The client-facing reason. Both clients already treat `agent_unreachable`
#: as "come back in a moment" (the app's retry ladder honours Retry-After).
SHED_REASON = "agent_unreachable"
SHED_DETAIL = "Your agent is not answering right now. Please try again in a moment."
#: Retry-After for an in-flight-cap refusal: the requests ahead are already
#: running, so the next slot is usually free within a second or two.
INFLIGHT_RETRY_AFTER_S = 2

def _clock() -> float:
    """Monotonic seconds; a seam so tests can move time without touching the
    global `time` module the event loop also reads."""
    return time.monotonic()


#: user_id -> monotonic deadline of the "recently timed out" window.
_timed_out_until: Dict[str, float] = {}
#: user_id -> guarded reads currently in flight from this process.
_inflight: Dict[str, int] = {}


class TenantProxyShed(Exception):
    """A guarded read refused before dialling the tenant.

    `why` is `recent_timeout` or `inflight_cap` (for the trail and the
    `X-Toup-Shed` header); `retry_after` is whole seconds, at least 1.
    """

    def __init__(self, why: str, retry_after: int):
        super().__init__(f"tenant proxy shed: {why} (retry after {retry_after}s)")
        self.why = why
        self.retry_after = max(1, int(retry_after))
        self.reason = SHED_REASON
        self.detail = SHED_DETAIL

    def headers(self) -> Dict[str, str]:
        return {
            "Retry-After": str(self.retry_after),
            "X-Toup-Reason": self.reason,
            "X-Toup-Shed": self.why,
        }

    def json_response(self):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"detail": self.detail, "code": self.reason},
            headers=self.headers(),
        )


def _memo_window_s() -> float:
    try:
        from app.config import settings

        value = float(getattr(settings, "tenant_proxy_timeout_memo_s", 10.0))
    except Exception:  # noqa: BLE001 — a guard must never break a proxy
        return 10.0
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return value


def _max_inflight() -> int:
    try:
        from app.config import settings

        value = int(getattr(settings, "tenant_proxy_max_inflight", 8))
    except Exception:  # noqa: BLE001
        return 8
    return max(0, value)


def is_timeout_error(exc: BaseException) -> bool:
    """A transport timeout that says the TENANT is slow.

    `httpx.PoolTimeout` is excluded: that is this process's own connection
    pool being full, not the tenant failing to answer.
    """
    import asyncio

    try:
        import httpx
    except Exception:  # noqa: BLE001
        httpx = None  # type: ignore[assignment]
    if httpx is not None:
        if isinstance(exc, httpx.PoolTimeout):
            return False
        if isinstance(exc, httpx.TimeoutException):
            return True
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError))


def note_timeout(user_id: str) -> None:
    """Mark the tenant as recently timed out (extends, never shortens)."""
    if not user_id:
        return
    window = _memo_window_s()
    if window <= 0:
        return
    deadline = _clock() + window
    if _timed_out_until.get(user_id, 0.0) < deadline:
        _timed_out_until[user_id] = deadline
    logger.info(
        "[TENANT_SHED] user=%s marked recently-timed-out for %.1fs",
        user_id[:8], window,
    )


def note_answered(user_id: str) -> None:
    """The tenant answered a guarded read: drop its timeout mark."""
    if user_id:
        _timed_out_until.pop(user_id, None)


def recent_timeout_remaining(user_id: str) -> float:
    """Seconds left in the tenant's timeout window, 0.0 when not marked."""
    deadline = _timed_out_until.get(user_id)
    if deadline is None:
        return 0.0
    left = deadline - _clock()
    if left <= 0:
        _timed_out_until.pop(user_id, None)
        return 0.0
    return left


def check_recent_timeout(user_id: str) -> None:
    """Raise `TenantProxyShed` while the tenant is in its timeout window."""
    left = recent_timeout_remaining(user_id)
    if left > 0:
        logger.info(
            "[TENANT_SHED] user=%s read refused: recently timed out (%.1fs left)",
            user_id[:8], left,
        )
        raise TenantProxyShed("recent_timeout", math.ceil(left))


def inflight(user_id: str) -> int:
    return _inflight.get(user_id, 0)


@asynccontextmanager
async def guarded_read(user_id: str) -> AsyncIterator[None]:
    """Admit one guarded read to `user_id`'s tenant, or raise TenantProxyShed.

    Checks the timeout memo first, then the in-flight cap; holds one in-flight
    slot for the duration of the block. The caller still calls
    `check_recent_timeout` between a tunnel attempt and its HTTPS fallback —
    that is the re-dial this module exists to skip.
    """
    check_recent_timeout(user_id)
    cap = _max_inflight()
    current = _inflight.get(user_id, 0)
    if cap and current >= cap:
        logger.warning(
            "[TENANT_SHED] user=%s read refused: %d reads already in flight (cap %d)",
            user_id[:8], current, cap,
        )
        raise TenantProxyShed("inflight_cap", INFLIGHT_RETRY_AFTER_S)
    _inflight[user_id] = current + 1
    try:
        yield
    finally:
        left = _inflight.get(user_id, 1) - 1
        if left > 0:
            _inflight[user_id] = left
        else:
            _inflight.pop(user_id, None)


def reset_for_test() -> None:
    _timed_out_until.clear()
    _inflight.clear()
