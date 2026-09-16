"""Event-loop stall sampler — the instrument the 15 Sep incident had none of.

Round 46, incident 1/2: container `toup-agent-pool-82` stopped logging
ENTIRELY twice inside one turn (9.47 s at 14:49:37.705, right after
`[OPENAI] Client rebuilt`; 20.86 s at 14:49:55.059, right after
`llm_total`) while ~40 sibling containers on the same host logged
normally throughout. The only evidence either block left was an ABSENCE
of lines — including the ~10 s `/agent/health` poll — because nothing in
this codebase measures whether the loop is running. Every remedy
proposed for those seconds is guesswork until a block names itself.

Two halves, and both are necessary:

  * the SAMPLER task measures the block. It runs on the event loop, so
    by construction it cannot observe itself while blocked — it only
    learns the lag afterwards, from its own overshoot.

  * the WATCHER thread tries to name it. It is a plain daemon thread
    that reads the sampler's heartbeat, and when the heartbeat goes
    stale it dumps the loop thread's stack.

WHAT `top=` IS, AND WHAT IT IS NOT — measured, not assumed. The watcher
calls `faulthandler.dump_traceback()` FROM PYTHON, so it must hold the
GIL to make that call. Two cases, both reproduced on the repo's own
interpreter (py3.12):

  * a blocker that RELEASES the GIL (a blocking socket read, a `sleep`,
    a C call that drops it): the dump runs during the block and `top=`
    names the blocker. A 3.0 s `time.sleep` block was named correctly.
  * a blocker that HOLDS the GIL (a lazy import, pydantic core-schema
    construction, a long pure-Python/bigint computation — i.e. the
    leading hypothesis for 15 Sep): the watcher is frozen behind it too.
    The dump completes only AFTER the block ends, and `top=` names
    whatever the loop RESUMED INTO. A 27.5 s GIL-holding block reported
    the watcher's own post-block frames.

`cap_ms` on the [LOOP_STALL] line is the discriminator: it is the wall
time the capture itself took. Small ⇒ the watcher was not starved and
`top=` is the blocker. Large (comparable to `lag_ms`) ⇒ the watcher was
GIL-starved, so the stall is a GIL-HOLDING one and `top=` is the
resume point, not the cause. Either way `lag_ms` — the DURATION — is
correct: the sampler measures it from its own overshoot and nothing
about the GIL affects it.

An earlier version of this docstring claimed faulthandler "walks the
frames without the GIL". It does so only when armed as a C-level
watchdog (`dump_traceback_later`), never when called from Python; the
`sys._current_frames()` rationale was backwards for the same reason.

Privacy (IMPL_RULES §5): the raw faulthandler text carries absolute
paths and function names. It is parsed down to `module:lineno` and
DISCARDED; nothing else is ever logged, returned or stored.

This is DIAGNOSTIC ONLY. `serving` does not branch on it and
`container_monitor` must never be pointed at it: the normal
distribution of loop lag on a 1.0-CPU cgroup has never been measured,
and a moving-window threshold wired into a restart decision is how a
sampler becomes an outage.
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import re
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How often the loop proves it is alive. 250 ms is small enough to put a
# tight bound on the detection delay and far too cheap to matter: one
# `asyncio.sleep` wake-up per quarter second.
SAMPLE_INTERVAL_S = 0.25

# Samples are kept for this long; `snapshot()` reports over the window.
WINDOW_S = 30.0
_MAX_SAMPLES = int(WINDOW_S / SAMPLE_INTERVAL_S) + 8

# Frames rendered per stall. Enough to see the caller chain, short
# enough that one wedged process cannot fill a log line.
_STACK_FRAMES = 6

# faulthandler's own cap. A 200-thread dump of a wedged process must not
# be able to allocate without bound in the watcher thread.
_DUMP_MAX_BYTES = 256 * 1024

_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+) in (?P<func>.+)$')
_THREAD_RE = re.compile(r"^(?:Current thread|Thread) (?P<ident>0x[0-9a-fA-F]+)")


def _setting(name: str, default: float) -> float:
    """Threshold lookup: settings first (L1 may append them to config.py),
    then the environment, then the default. Deliberately not a hard import
    of `settings` — this module is started from the agent lifespan and must
    not be able to fail a boot over a missing attribute."""
    try:
        from app.config import settings  # local: keep the import graph shallow

        val = getattr(settings, name, None)
        if val is not None:
            return float(val)
    except Exception:
        pass
    raw = os.getenv(name.upper())
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


# ---------------------------------------------------------------------------
# module state
# ---------------------------------------------------------------------------

_samples: Deque[Tuple[float, float]] = deque(maxlen=_MAX_SAMPLES)  # (monotonic, lag_ms)
_task: Optional["asyncio.Task[None]"] = None
_thread: Optional[threading.Thread] = None
_stop = threading.Event()

_last_beat: float = 0.0          # monotonic, written by the sampler ONLY
_loop_thread_ident: int = 0      # the thread the event loop runs on
_stalls: int = 0
_last_stall: Optional[Dict[str, Any]] = None
_last_report_at: float = 0.0     # monotonic, rate limiter
_reported_this_stall: bool = False


# ---------------------------------------------------------------------------
# stack capture (watcher thread only)
# ---------------------------------------------------------------------------

def _module_lineno(path: str, lineno: str) -> str:
    """`/app/app/agent/agent_runner.py`, `7412` -> `agent_runner:7412`.

    Basename without extension only: a full path is not PII but it is not
    information either, and keeping the rendering to one shape is what
    makes these lines greppable across the fleet."""
    base = path.rsplit("/", 1)[-1]
    if base.endswith(".py"):
        base = base[:-3]
    return f"{base}:{lineno}"


def _capture_frames(ident: int) -> List[str]:
    """Top frames of the thread with this ident, as `module:lineno`.

    Runs on the watcher thread while the loop is blocked. Never raises:
    a sampler that can crash the process it is watching is worse than no
    sampler."""
    try:
        # faulthandler writes to a FILE DESCRIPTOR, not a Python file object:
        # an io.StringIO raises `io.UnsupportedOperation: fileno` and the
        # capture silently returns nothing (every stall reads `top=unknown`).
        # A real temp fd is the only sink it accepts; a pipe would deadlock
        # on a dump larger than the pipe buffer with nobody reading.
        with tempfile.TemporaryFile(buffering=0) as fh:
            faulthandler.dump_traceback(file=fh, all_threads=True)
            fh.seek(0)
            text = fh.read(_DUMP_MAX_BYTES).decode("utf-8", "replace")
    except Exception:
        return []

    in_section = False
    frames: List[str] = []
    for line in text.splitlines():
        m = _THREAD_RE.match(line)
        if m:
            if in_section:
                break  # the next thread's header ends ours
            # Compare NUMERICALLY: faulthandler zero-pads the id to the
            # platform's pointer width (`0x0000000205f922c0`), so a string
            # compare against `hex(get_ident())` never matches and every
            # stall reports `top=unknown`.
            try:
                in_section = int(m.group("ident"), 16) == ident
            except ValueError:
                in_section = False
            continue
        if not in_section:
            continue
        fm = _FRAME_RE.match(line)
        if not fm:
            continue
        # Never attribute a stall to the sampler that is reporting it.
        if fm.group("file").rsplit("/", 1)[-1] == "loop_health.py":
            continue
        frames.append(_module_lineno(fm.group("file"), fm.group("line")))
        if len(frames) >= _STACK_FRAMES:
            break
    return frames


def _watch() -> None:
    """Daemon thread: detect a stale heartbeat and photograph the loop."""
    global _stalls, _last_stall, _last_report_at, _reported_this_stall

    warn_s = _setting("loop_stall_warn_s", 2.0)
    report_interval_s = _setting("loop_stall_report_interval_s", 30.0)

    while not _stop.wait(SAMPLE_INTERVAL_S):
        beat = _last_beat
        if not beat:
            continue
        lag = time.monotonic() - beat
        if lag < warn_s:
            _reported_this_stall = False
            continue
        if _reported_this_stall:
            continue  # one report per stall, not one per poll
        _reported_this_stall = True
        _stalls += 1

        # Time the capture. `dump_traceback` is a Python call, so the watcher
        # must take the GIL to make it: if the blocker HOLDS the GIL the
        # capture is itself blocked, and the frames it eventually returns are
        # the loop's RESUME point rather than the cause. `cap_ms` is the only
        # thing that tells those two stalls apart, and it costs one clock read.
        _cap_t0 = time.monotonic()
        frames = _capture_frames(_loop_thread_ident)
        cap_ms = (time.monotonic() - _cap_t0) * 1000.0
        top = frames[0] if frames else "unknown"
        _last_stall = {
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "lag_ms": round(lag * 1000.0, 1),
            "top": top,
            "cap_ms": round(cap_ms, 1),
            # False ⇒ `top` names the loop's resume point, not the blocker.
            "top_is_blocker": cap_ms < max(250.0, warn_s * 1000.0 * 0.25),
        }

        now = time.monotonic()
        if now - _last_report_at < report_interval_s:
            continue  # counted, not logged: a wedged process must not flood
        _last_report_at = now
        try:
            logger.warning(
                "[LOOP_STALL] lag_ms=%d threshold_ms=%d cap_ms=%d "
                "top_is_blocker=%d top=%s stack=%s threads=%d stalls=%d",
                int(lag * 1000.0),
                int(warn_s * 1000.0),
                int(cap_ms),
                1 if _last_stall.get("top_is_blocker") else 0,
                top,
                ">".join(frames[:_STACK_FRAMES]) or "-",
                threading.active_count(),
                _stalls,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# sampler (event loop)
# ---------------------------------------------------------------------------

async def _sample() -> None:
    global _last_beat
    _last_beat = time.monotonic()
    while not _stop.is_set():
        before = time.monotonic()
        try:
            await asyncio.sleep(SAMPLE_INTERVAL_S)
        except asyncio.CancelledError:
            raise
        now = time.monotonic()
        _last_beat = now
        # Overshoot beyond the sleep we asked for IS the block: the loop
        # owed us a wake-up at `before + SAMPLE_INTERVAL_S` and did not
        # deliver it until `now`.
        lag_ms = max(0.0, (now - before - SAMPLE_INTERVAL_S) * 1000.0)
        _samples.append((now, lag_ms))


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------

def start(loop: Optional[asyncio.AbstractEventLoop] = None) -> bool:
    """Idempotent. Returns True if this call started the sampler.

    Never raises: it is called from the agent lifespan, and a diagnostic
    that can fail a boot is a liability, not an instrument."""
    global _task, _thread, _loop_thread_ident

    try:
        if _task is not None and not _task.done():
            return False
        _stop.clear()
        _loop_thread_ident = threading.get_ident()
        loop = loop or asyncio.get_running_loop()
        _task = loop.create_task(_sample(), name="loop_health.sampler")
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(
                target=_watch, name="loop-health-watch", daemon=True
            )
            _thread.start()
        logger.info(
            "[LOOP_STALL] sampler started interval_ms=%d threshold_ms=%d",
            int(SAMPLE_INTERVAL_S * 1000),
            int(_setting("loop_stall_warn_s", 2.0) * 1000),
        )
        return True
    except Exception:
        logger.debug("loop_health.start failed", exc_info=True)
        return False


async def stop() -> None:
    """Idempotent."""
    global _task, _thread
    _stop.set()
    task, _task = _task, None
    if task is not None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    thread, _thread = _thread, None
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)


def snapshot() -> Dict[str, Any]:
    """Cheap, allocation-light, safe from any thread. Shape is a contract —
    `/agent/health`'s `loop` block and lane L1 both read it."""
    now = time.monotonic()
    window = [lag for (ts, lag) in list(_samples) if now - ts <= WINDOW_S]
    n = len(window)
    if n:
        window.sort()
        idx = min(n - 1, int(round(0.99 * (n - 1))))
        p99 = window[idx]
        mx = window[-1]
    else:
        p99 = 0.0
        mx = 0.0
    # R46/L8: mirror the window max into the shared gauge so the anomaly sweep
    # reads one number per container. Recorded, never "due" — health_signals
    # will not try to refresh it. A gauge must never fail a snapshot.
    try:
        from app.services import health_signals as _hs
        _hs.set_gauge("loop_lag_ms_max_30s", int(mx))
    except Exception:  # noqa: BLE001
        pass
    return {
        "running": _task is not None and not _task.done(),
        "samples": n,
        "blocked_ms_max_30s": round(mx, 1),
        "blocked_ms_p99": round(p99, 1),
        "stalls": _stalls,
        "last_stall": dict(_last_stall) if _last_stall else None,
    }


def _reset_for_tests() -> None:
    """Test-only: the module keeps process-global state by design."""
    global _stalls, _last_stall, _last_report_at, _reported_this_stall, _last_beat
    _samples.clear()
    _stalls = 0
    _last_stall = None
    _last_report_at = 0.0
    _reported_this_stall = False
    _last_beat = 0.0
