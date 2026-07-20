"""Interim Live Activity progress during a single agent turn.

Founder bug 2026-07-16: the mission progress bar jumped 0 → 45 → 100
because the ONLY progress source was the AUTOPILOT_PROGRESS marker
parsed after each ~300s tick. During the turn itself zero events fire,
so the lock screen / Dynamic Island sit frozen for minutes.

``TurnProgressEmitter`` hooks the runner's existing per-tool-call
seams (``on_tool_start`` / ``on_tool_end``) and emits throttled
``event_kind='progress'`` rows through the durable notify outbox. The
Live Activity lane turns those into content-state updates (bar +
step subtitle); the dispatcher never renders ``progress`` as an alert
push, so emitting mid-turn is safe. Rows carry ``update_only`` — they
refresh an existing card and never silently START one, so emitting on
every turn (Claude-parity live status) cannot grow lock-screen cards
on ordinary foreground turns.

Interpolation: ``p(k) = base + (ceiling - base) * (1 - 0.85**k)`` —
strictly monotonic in the number of tool calls ``k``, asymptotic to
``ceiling``, never reaching it. Only a real terminal event (done /
answer-delivered) sets 100; the bar must never LIE about being nearly
finished, which is why the ceiling stays below the authoritative
end-of-turn value.

Throttling lives HERE (producer side): every emission costs an outbox
row + platform POST + notification_queue row + N APNs sends, and the
LA lane has no throttle of its own for progress rows.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# Step subtitles — mirrors the client ToolBlock vocabulary so the
# Dynamic Island says the same thing the in-app action row would.
_TOOL_SUBTITLES = {
    "web_search": "Searching the web…",
    "extension_search": "Searching the web…",
    "web_fetch": "Reading a page…",
    "exec": "Running a command…",
    "pty_exec": "Running a command…",
    "generate_image": "Creating an image…",
    "edit_image": "Editing an image…",
    "write_file": "Writing a file…",
    "edit_file": "Editing a file…",
    "read_file": "Reading a file…",
    "list_files": "Looking at files…",
    "spawn": "Spawning a helper…",
    "start_mission": "Starting a mission…",
}


def _subtitle_for(tool_name: str) -> str:
    if tool_name in _TOOL_SUBTITLES:
        return _TOOL_SUBTITLES[tool_name]
    if tool_name.startswith("browser_"):
        return "Browsing…"
    if tool_name.startswith("memory_") or tool_name in ("recall_day", "search_memory"):
        return "Checking notes…"
    return (tool_name or "working").replace("_", " ") + "…"


class TurnProgressEmitter:
    """Per-turn progress beacon. Wire ``on_tool_start``/``on_tool_end``
    into ``AgentRunner.run``'s callbacks (compose with existing ones).

    ``gate``: optional callable; when it returns False, nothing is
    emitted. With ``update_only`` rows (the default) most producers
    need no gate — the row can only refresh an existing card.
    """

    def __init__(
        self,
        *,
        mission_id: str,
        mission_title: str,
        base_progress: int = 0,
        ceiling: int = 90,
        route: str = "mission-control",
        min_interval_s: float = 8.0,
        gate: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.mission_id = mission_id
        self.mission_title = mission_title[:200]
        self.base = max(0, min(100, int(base_progress)))
        self.ceiling = max(self.base, min(99, int(ceiling)))
        self.route = route
        self.min_interval_s = min_interval_s
        self.gate = gate
        self.tool_count = 0
        self.last_emitted_progress = self.base
        self._last_emit_ts = 0.0

    def _p(self) -> int:
        span = self.ceiling - self.base
        if span <= 0:
            return self.base
        p = int(round(self.base + span * (1 - 0.85 ** self.tool_count)))
        # Rounding may touch the ceiling after many tool calls — interim
        # progress must stay strictly below it (only a real terminal
        # event closes the bar).
        return min(p, self.ceiling - 1)

    def force_next(self) -> None:
        """Reset the throttle so the next tool boundary emits
        immediately — used when the card appears mid-turn (chat
        force-quit after several tools already ran)."""
        self._last_emit_ts = 0.0

    async def on_tool_start(self, tool_name: str) -> None:
        self.tool_count += 1
        if self.gate is not None and not self.gate():
            return
        now = time.monotonic()
        if self.tool_count > 1 and (now - self._last_emit_ts) < self.min_interval_s:
            return
        progress = self._p()
        try:
            from app.services.agent_notify_client import notify

            await notify(
                event_kind="progress",
                title=self.mission_title,
                body=_subtitle_for(tool_name)[:300],
                data={
                    "mission_id": self.mission_id,
                    "mission_title": self.mission_title[:80],
                    "route": self.route,
                    "progress": progress,
                    # Refresh an existing card only — never start one.
                    "update_only": True,
                },
                priority="low",
                dedup_key=f"{self.mission_id}:progress",
            )
            self._last_emit_ts = now
            self.last_emitted_progress = max(self.last_emitted_progress, progress)
        except Exception as e:  # noqa: BLE001 — cosmetic beacon, never breaks a turn
            logger.debug(
                "[turn_progress] emit failed mission=%s: %s", self.mission_id, e,
            )

    async def on_tool_end(self, tool_name: str, summary: str, tool_input=None) -> None:
        # Advance the interpolation without doubling push volume — the
        # emission happens on the NEXT tool_start, which is when the
        # subtitle changes anyway.
        return None
