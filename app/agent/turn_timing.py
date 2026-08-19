"""Per-turn latency waterfall (Round 4, item 7a).

One :class:`TurnWaterfall` per agent turn. Stages are recorded as they
happen (offset from the turn's t0, plus duration) and rendered ONCE at
turn end as a single structured log line::

    [TURN_WATERFALL] {"turn_ms": 17515, "phase1_ms": 1590, ...,
                      "stages": [{"name": "llm", "i": 1, "t0": 1592, "ms": 3366, ...}, ...]}

Why one line and not the scattered ``[PERF]`` lines (which stay): the
question "where did 45 seconds go on THIS turn?" needs every stage of one
turn together, in order, with the same clock. The scattered lines are the
per-stage detail; this is the map. Cheap: a list of small dicts, no I/O
until the final ``logger.info``.

Also keeps the two counters that decide whether the round-4 structural
fixes are working in prod:

* ``bookkeeping_only_rounds`` — LLM rounds whose ONLY tool calls were
  ``create_job`` / ``update_job``. Every one of those is a full LLM
  round-trip (2.5–8 s measured) spent on narration; the prompt now asks
  the model to fold them into the same response as real tool calls, and
  this counter is how we know it complied.
* ``tool_rounds`` — LLM rounds that executed at least one tool.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Tools that are pure progress narration — a round made only of these
#: did no work the user asked for.
BOOKKEEPING_TOOLS: frozenset = frozenset({"create_job", "update_job"})


class TurnWaterfall:
    __slots__ = (
        "t0", "stages", "meta", "_open", "bookkeeping_only_rounds",
        "tool_rounds", "llm_rounds", "_emitted",
    )

    def __init__(self, t0: Optional[float] = None) -> None:
        self.t0 = time.perf_counter() if t0 is None else t0
        self.stages: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {}
        self._open: Dict[str, float] = {}
        self.bookkeeping_only_rounds = 0
        self.tool_rounds = 0
        self.llm_rounds = 0
        self._emitted = False

    # ── clock helpers ─────────────────────────────────────────────
    def now_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def start(self, name: str) -> None:
        """Open a stage; pair with :meth:`end`."""
        self._open[name] = time.perf_counter()

    def end(self, name: str, **extra: Any) -> int:
        """Close a stage opened by :meth:`start`; returns its duration (ms)."""
        t_start = self._open.pop(name, None)
        if t_start is None:
            return 0
        ms = int((time.perf_counter() - t_start) * 1000)
        self.stages.append({
            "name": name,
            "t0": int((t_start - self.t0) * 1000),
            "ms": ms,
            **{k: v for k, v in extra.items() if v is not None},
        })
        return ms

    def mark(self, stage: str, ms: int = 0, *, t0_ms: Optional[int] = None,
             **extra: Any) -> None:
        """Record a stage whose timing was measured elsewhere.

        ``t0_ms`` is the stage's start offset from the turn's t0; when it is
        omitted the stage is assumed to have just ended (start = now - ms).
        Extra keys ride the stage dict; ``name``/``t0``/``ms`` are reserved.
        """
        if t0_ms is None:
            t0_ms = max(0, self.now_ms() - int(ms))
        self.stages.append({
            "name": stage, "t0": int(t0_ms), "ms": int(ms),
            **{k: v for k, v in extra.items() if v is not None and k not in ("name", "t0", "ms")},
        })

    # ── round accounting ──────────────────────────────────────────
    def note_round(self, tool_names: List[str]) -> None:
        """Called once per LLM round that returned tool calls."""
        if not tool_names:
            return
        self.tool_rounds += 1
        if all(n in BOOKKEEPING_TOOLS for n in tool_names):
            self.bookkeeping_only_rounds += 1

    # ── render ────────────────────────────────────────────────────
    def summary(self) -> Dict[str, Any]:
        by_name: Dict[str, int] = {}
        for s in self.stages:
            by_name[s["name"]] = by_name.get(s["name"], 0) + int(s.get("ms", 0))
        llm_ms = by_name.get("llm", 0)
        tool_ms = sum(int(s.get("ms", 0)) for s in self.stages if s["name"] == "tool")
        out: Dict[str, Any] = {
            "turn_ms": self.now_ms(),
            "llm_ms": llm_ms,
            "llm_rounds": self.llm_rounds,
            "tool_ms_sum": tool_ms,
            "tool_rounds": self.tool_rounds,
            "bookkeeping_only_rounds": self.bookkeeping_only_rounds,
        }
        for k in ("phase1", "memory_retrieval", "build_system_prompt", "save"):
            if k in by_name:
                out[f"{k}_ms"] = by_name[k]
        out.update(self.meta)
        out["stages"] = self.stages
        return out

    def emit(self, level: int = logging.INFO) -> Optional[Dict[str, Any]]:
        """Log the waterfall once. Safe to call twice (second is a no-op)."""
        if self._emitted:
            return None
        self._emitted = True
        try:
            data = self.summary()
            logger.log(level, "[TURN_WATERFALL] %s", json.dumps(data, separators=(",", ":"), default=str))
            return data
        except Exception:  # noqa: BLE001 — telemetry must never break a turn
            logger.debug("[TURN_WATERFALL] render failed", exc_info=True)
            return None
