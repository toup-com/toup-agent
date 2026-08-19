"""Per-turn job/step attribution (Round 4, item 8).

The clients render a turn as *steps* (declared by ``create_job``) with the
*actions* (tool calls) that served each step underneath. Until now every
tool frame carried only a tool name, so the iOS run view bucketed all
actions under step 1 and printed "No actions recorded" for the rest.

:class:`StepTracker` is the runner-side source of truth for "which step is
active right now". It is fed the ``create_job`` / ``update_job`` calls as
they execute (inputs + results) and answers :attr:`step_index` /
:attr:`job_id` for every tool event emitted afterwards. Rules:

* ``create_job(steps=[...])`` → job active, step 0 running.
* ``update_job(current_step=k)`` → steps 0..k done, step ``k+1`` running
  (capped at the last step). This is the tool's documented semantics —
  "index of the step to mark as done" — so a batch that reads
  ``[update_job(current_step=0), web_fetch, web_fetch]`` attributes both
  fetches to step 1, which is what the model meant.
* ``update_job(status=completed|failed|waiting_on_user)`` → all steps
  resolved; the active index stays on the last step.
* An ``update_job`` for a job this tracker has never seen (a job created
  by the regex intake, or in an earlier turn) adopts it: step index from
  the input, total from the tool's result JSON.

The final answer (an LLM round with no tool calls) is attributed to the
LAST declared step when the job has more than one — the writing/summary
step is by construction the last one, and the turn-end finalizer marks
every remaining step done anyway. :meth:`final_step_index` encodes that.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class StepTracker:
    __slots__ = ("job_id", "steps", "active", "steps_total", "_seen_update", "job_type")

    def __init__(self) -> None:
        self.job_id: Optional[str] = None
        self.steps: List[str] = []
        self.steps_total: int = 0
        self.active: int = 0
        self.job_type: Optional[str] = None
        self._seen_update = False

    # ── state ─────────────────────────────────────────────────────
    @property
    def has_job(self) -> bool:
        return bool(self.job_id)

    @property
    def step_index(self) -> Optional[int]:
        """0-based index of the step actions should attach to; None when
        no job has been declared this turn."""
        return self.active if self.has_job else None

    def step_name(self, index: Optional[int] = None) -> Optional[str]:
        i = self.active if index is None else index
        if 0 <= i < len(self.steps):
            return self.steps[i]
        return None

    def final_step_index(self) -> Optional[int]:
        """Step for the closing answer round (no tools). The last declared
        step when there is more than one — see module docstring."""
        if not self.has_job:
            return None
        if self.steps_total > 1:
            return self.steps_total - 1
        return self.active

    def event_fields(self, *, step_index: Optional[int] = None) -> Dict[str, Any]:
        """The attribution keys added to every tool/step frame. Absent keys
        (not nulls) when there is no job, so old clients see no change."""
        if not self.has_job:
            return {}
        idx = self.active if step_index is None else step_index
        out: Dict[str, Any] = {"job_id": self.job_id, "step_index": idx}
        name = self.step_name(idx)
        if name:
            out["step_name"] = name
        if self.steps_total:
            out["steps_total"] = self.steps_total
        if self.job_type:
            out["job_type"] = self.job_type
        return out

    # ── feeds ─────────────────────────────────────────────────────
    def observe(self, tool_name: str, tool_input: Any, result: Any) -> bool:
        """Feed one executed bookkeeping call. Returns True when the active
        step (or job) changed — the runner emits a ``step_change`` event."""
        if tool_name == "create_job":
            return self._on_create(tool_input, result)
        if tool_name == "update_job":
            return self._on_update(tool_input, result)
        return False

    def _on_create(self, tool_input: Any, result: Any) -> bool:
        data = _as_dict(result)
        job_id = data.get("job_id")
        if not job_id:
            return False  # ERROR: … — nothing to track
        steps = tool_input.get("steps") if isinstance(tool_input, dict) else None
        self.job_id = str(job_id)
        self.steps = [str(s) for s in steps] if isinstance(steps, list) else []
        self.steps_total = len(self.steps) or int(data.get("steps") or 0)
        self.job_type = data.get("job_type") or None
        self.active = 0
        return True

    def _on_update(self, tool_input: Any, result: Any) -> bool:
        if not isinstance(tool_input, dict):
            return False
        data = _as_dict(result)
        if not data.get("ok"):
            return False  # unknown job / error — leave state alone
        job_id = str(tool_input.get("job_id") or "").strip()
        changed = False
        if job_id and job_id != self.job_id:
            # Adopt a job we did not create (regex intake / earlier turn).
            self.job_id = job_id
            self.steps = []
            self.steps_total = 0
            self.active = 0
            changed = True
        total = data.get("total_steps")
        if isinstance(total, int) and total > 0:
            if total != self.steps_total:
                changed = True
            self.steps_total = total
        if data.get("job_type"):
            self.job_type = data.get("job_type")
        cur = tool_input.get("current_step")
        status = tool_input.get("status")
        new_active = self.active
        if isinstance(cur, int) and cur >= 0:
            last = max(0, (self.steps_total or (cur + 1)) - 1)
            new_active = min(cur + 1, last)
        if status in ("completed", "failed", "waiting_on_user") and self.steps_total:
            new_active = self.steps_total - 1
        if new_active != self.active:
            self.active = new_active
            changed = True
        return changed


def _as_dict(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        s = result.strip()
        if s.startswith("{"):
            try:
                d = json.loads(s)
                return d if isinstance(d, dict) else {}
            except ValueError:
                return {}
    return {}
