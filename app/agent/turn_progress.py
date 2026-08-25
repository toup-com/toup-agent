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
    # `tool_display` owns the human names for tools that have one, so the
    # Dynamic Island and the chat row cannot drift apart. Round 18 made the
    # unmapped-connector case an honest generic; R30 (D-01) makes the WHOLE
    # tail total — `public_step_label` never yields the wire id, so the last
    # derivation (`tool_name.replace("_", " ")` for plain tools) is gone too.
    from app.agent.tool_display import public_step_label
    return f"{public_step_label(tool_name)}…"


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
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> None:
        self.mission_id = mission_id
        self.mission_title = mission_title[:200]
        self.base = max(0, min(100, int(base_progress)))
        self.ceiling = max(self.base, min(99, int(ceiling)))
        self.route = route
        self.min_interval_s = min_interval_s
        self.gate = gate
        # Round 3 (item 3): deep-link ids for the card's content-state.
        # Both may be set later (`set_deep_link`) — ws_chat learns the
        # session id after the emitter exists on first-message turns.
        self.chat_id = chat_id
        self.message_id = message_id
        self.tool_count = 0
        self.last_emitted_progress = self.base
        self._last_emit_ts = 0.0
        # Round 4 (item 3): job step context, fed by step_change events, so
        # the interim beacon carries the same stepName / stepsDone /
        # stepsTotal / jobType the job's own pushes do — the Dynamic Island
        # segmented bar then never blanks between update_job calls.
        self.job_id: Optional[str] = None
        self.job_type: Optional[str] = None
        self.step_index: Optional[int] = None
        self.step_name: Optional[str] = None
        self.steps_total: Optional[int] = None

    def _p(self) -> int:
        span = self.ceiling - self.base
        if span <= 0:
            return self.base
        p = int(round(self.base + span * (1 - 0.85 ** self.tool_count)))
        # Rounding may touch the ceiling after many tool calls — interim
        # progress must stay strictly below it (only a real terminal
        # event closes the bar).
        return min(p, self.ceiling - 1)

    def set_deep_link(self, chat_id: Optional[str] = None,
                      message_id: Optional[str] = None) -> None:
        """Attach/refresh the deep-link ids subsequent beacons carry."""
        if chat_id:
            self.chat_id = str(chat_id)[:64]
        if message_id:
            self.message_id = str(message_id)[:64]

    def force_next(self) -> None:
        """Reset the throttle so the next tool boundary emits
        immediately — used when the card appears mid-turn (chat
        force-quit after several tools already ran)."""
        self._last_emit_ts = 0.0

    async def on_step_change(self, payload: dict) -> None:
        """Round 4 (item 3): remember the active job step (from the runner's
        step_change event) and refresh the card right away — a step
        transition is exactly the moment the island should move."""
        try:
            self.job_id = payload.get("job_id") or self.job_id
            if payload.get("step_index") is not None:
                self.step_index = int(payload["step_index"])
            self.step_name = payload.get("step_name") or self.step_name
            if payload.get("steps_total"):
                self.steps_total = int(payload["steps_total"])
            if payload.get("job_type"):
                self.job_type = payload.get("job_type")
        except Exception:  # noqa: BLE001
            return
        self.force_next()

    def _job_fields(self) -> dict:
        out: dict = {}
        if self.job_type:
            out["job_type"] = str(self.job_type)[:16]
        if self.steps_total:
            out["steps_total"] = int(self.steps_total)
            if self.step_index is not None:
                out["steps_done"] = max(0, min(int(self.step_index), int(self.steps_total)))
        return out

    async def on_tool_start(self, tool_name: str, meta: Optional[dict] = None) -> None:
        self.tool_count += 1
        if self.gate is not None and not self.gate():
            return
        if meta:
            # Provisional step context from the runner (see agent_runner
            # OnToolStart docs); step_change refines it. Round 8: the
            # provisional fields are a SNAPSHOT taken before the round's
            # bookkeeping ran (the web batch is scheduled with them and its
            # tool_start frames arrive after update_job advanced the step),
            # so they may name an OLDER step than the last step_change did.
            # They can only move the beacon FORWARD — a beacon that stepped
            # back would push a lower stepsDone onto the phone card and the
            # island would read "1/3" after the app said "2/3". A different
            # job id is a new job and resets.
            new_job = bool(meta.get("job_id")) and meta.get("job_id") != self.job_id
            if meta.get("job_id"):
                self.job_id = meta.get("job_id")
            if meta.get("step_index") is not None:
                try:
                    _idx = int(meta["step_index"])
                    if new_job or self.step_index is None or _idx >= self.step_index:
                        self.step_index = _idx
                        if meta.get("step_name"):
                            self.step_name = meta.get("step_name")
                except (TypeError, ValueError):
                    pass
            elif meta.get("step_name") and (new_job or not self.step_name):
                self.step_name = meta.get("step_name")
            if meta.get("steps_total"):
                self.steps_total = int(meta["steps_total"])
        # Round 4: create_job / update_job are bookkeeping, not actions — the
        # job's own pushes carry the card, and a beacon reading "update job…"
        # is exactly the raw-tool-name subtitle this surface should not show.
        if tool_name in ("create_job", "update_job"):
            return
        now = time.monotonic()
        if self.tool_count > 1 and (now - self._last_emit_ts) < self.min_interval_s:
            return
        progress = self._p()
        try:
            from app.services.agent_notify_client import notify

            # Round 4 (item 3): when a job step is active, the card's step
            # line is the STEP ("Read the top results"), not the tool verb —
            # the tool verb rides as the alert body. Discrete step progress
            # wins over the interpolated curve when we have it (the LA lane
            # never moves a bar backwards, so it can only tighten).
            _step_line = (self.step_name or _subtitle_for(tool_name))[:80]
            if self.steps_total and self.step_index is not None:
                progress = max(progress, int(self.base + (self.ceiling - self.base)
                                             * (self.step_index / max(1, self.steps_total))))
            data = {
                "mission_id": self.mission_id,
                "mission_title": self.mission_title[:80],
                "route": self.route,
                "progress": progress,
                # Refresh an existing card only — never start one.
                "update_only": True,
                "step_name": _step_line,
                **self._job_fields(),
            }
            if self.chat_id:
                data["chat_id"] = self.chat_id
            if self.message_id:
                data["message_id"] = self.message_id
            await notify(
                event_kind="progress",
                title=self.mission_title,
                body=_subtitle_for(tool_name)[:300],
                data=data,
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
