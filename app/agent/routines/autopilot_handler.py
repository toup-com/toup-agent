"""Autopilot mission handler — the autonomous heartbeat engine core
(Autopilot arc PR6, docs/autopilot/PLAN.md D1-D4).

A MISSION is a ``Routine`` row with ``kind='autopilot'``:

- ``config_json`` (user-owned, set at hand-off): ``goal`` (required),
  ``budget_credits``, ``max_ticks``, ``tick_timeout_s``, ``urgent``.
- ``last_state_json`` (engine-owned working state, persisted through
  the runner's existing watermark mechanism): status, next_due_at,
  ticks_run, spent_credits, no_progress_streak, note (the agent's
  scratchpad for its next tick), last_summary, platform_fail_streak.

The tick is TWO-TIER (the OpenClaw lesson — no-op ticks must cost $0):
tier 1 is deterministic — status/due/budget/credit gates, zero LLM
calls, returns ``success_empty``; tier 2 runs one bounded agent turn
through the proven background-turn seam (``channel='routine'``,
``save_user_message=False``, ``current_job_id`` linkage — the
agent_task_handler call shape) wrapped in ``asyncio.wait_for``.

CONTRACT WITH THE RUNNER: this handler NEVER raises and always returns
``status='success'`` — mission-level failure is a *state* (the
watermark only advances on success, runner.py, and a raised exception
would burn the runner's 3-attempt retry on a turn that already spent
tokens). The runner's retry loop is for infrastructure, not missions.

Tick output contract (v1, superseded by real tools in PR7): the turn
prompt instructs the model to end with::

    AUTOPILOT_STATUS: working | done | blocked
    AUTOPILOT_SUMMARY: <one user-visible line of progress>
    AUTOPILOT_NOTE: <private note to your next tick>

A missing status line counts as ``working`` with a no-progress strike
— the engine treats the string contract as a hint, and the strike
system (not the model's honesty) is what terminates a drifting loop.

Terminal/blocked transitions disable the routine row (``enabled=False``
— ``_fire`` re-checks enabled at fire start, so job minting stops
immediately even before the 10-min reconcile drops the trigger) and
fire a notification through the durable outbox (PR4).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from .base_handler import RoutineResult

logger = logging.getLogger(__name__)


# Mission statuses (last_state_json["status"]).
MISSION_ACTIVE = "active"
MISSION_WAITING_INPUT = "waiting_input"        # PR7 wires the resume path
MISSION_WAITING_APPROVAL = "waiting_approval"  # PR7
MISSION_BLOCKED = "blocked"
MISSION_COMPLETED = "completed"
MISSION_FAILED = "failed"

_PAUSED_STATUSES = {
    MISSION_WAITING_INPUT, MISSION_WAITING_APPROVAL, MISSION_BLOCKED,
}
_TERMINAL_STATUSES = {MISSION_COMPLETED, MISSION_FAILED}

_STATUS_MARKER = "AUTOPILOT_STATUS:"
_SUMMARY_MARKER = "AUTOPILOT_SUMMARY:"
_NOTE_MARKER = "AUTOPILOT_NOTE:"
_PROGRESS_MARKER = "AUTOPILOT_PROGRESS:"


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _parse_iso(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def parse_tick_markers(text: str) -> Dict[str, str]:
    """Extract the AUTOPILOT_* marker lines from the turn's final text.
    Tolerant: markers anywhere in the text, last occurrence wins."""
    out: Dict[str, str] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        for marker, key in (
            (_STATUS_MARKER, "status"),
            (_SUMMARY_MARKER, "summary"),
            (_NOTE_MARKER, "note"),
            (_PROGRESS_MARKER, "progress"),
        ):
            if stripped.upper().startswith(marker):
                out[key] = stripped[len(marker):].strip()
    if "status" in out:
        out["status"] = out["status"].split()[0].lower() if out["status"] else ""
    return out


def parse_progress_value(raw: Optional[str]) -> Optional[int]:
    """'45', '45%', '45 percent' → 45 (clamped 0-100); junk → None.
    The marker is a hint like the others — never let a malformed line
    crash a tick."""
    if not raw:
        return None
    token = raw.strip().split()[0].rstrip("%")
    try:
        return max(0, min(100, int(float(token))))
    except ValueError:
        return None


class AutopilotHandler:
    """kind='autopilot' — one instance shared across fires (stateless;
    all state lives on the Routine row)."""

    kind = "autopilot"

    def __init__(self, agent_runner: Any = None):
        # Injected by RoutineRunner at boot (same seam as agent_task).
        self._agent_runner = agent_runner

    # ── entry point ────────────────────────────────────────────────
    async def execute(self, routine: Any, run: Any, db: AsyncSession) -> RoutineResult:
        now = datetime.utcnow()
        cfg = routine.config_json or {}
        state = dict(routine.last_state_json or {})
        goal = (cfg.get("goal") or routine.prompt_text or "").strip()

        if not goal:
            return await self._transition(
                routine, db, state, now, MISSION_FAILED, "no_goal",
                notify_kind="mission_failed",
                notify_title=f"Autopilot mission “{routine.name}” failed",
                notify_body="The mission has no goal text.",
            )

        status = state.get("status", MISSION_ACTIVE)

        # ── Tier 1: deterministic gates (zero-cost no-op ticks) ─────
        if status in _TERMINAL_STATUSES or status in _PAUSED_STATUSES:
            # Paused/terminal missions shouldn't even be firing —
            # transitions disable the routine — but a fire racing the
            # disable lands here. Cheap exit.
            return RoutineResult(status="success", outcome="success_empty",
                                 metrics={"gate": f"status_{status}"})

        next_due = _parse_iso(state.get("next_due_at"))
        if next_due is not None and now < next_due:
            return RoutineResult(status="success", outcome="success_empty",
                                 metrics={"gate": "not_due"})

        max_ticks = int(cfg.get("max_ticks") or settings.autopilot_max_ticks)
        if int(state.get("ticks_run", 0)) >= max_ticks:
            return await self._transition(
                routine, db, state, now, MISSION_FAILED, "max_ticks_exceeded",
                notify_kind="mission_failed",
                notify_title=f"Autopilot mission “{routine.name}” stopped",
                notify_body=(
                    f"Hit the {max_ticks}-tick safety cap without finishing. "
                    "Review progress in Mission Control and restart with a "
                    "narrower goal if needed."
                ),
            )

        budget = float(cfg.get("budget_credits") or settings.autopilot_default_budget_credits)
        spent = float(state.get("spent_credits", 0.0))
        if budget and spent >= budget:
            return await self._transition(
                routine, db, state, now, MISSION_BLOCKED, "budget_exhausted",
                notify_kind="mission_failed",
                notify_title=f"Autopilot mission “{routine.name}” hit its budget",
                notify_body=(
                    f"Spent {spent:.1f} of {budget:.0f} credits. Raise the "
                    "budget in Mission Control to continue."
                ),
                notify_priority="high",
            )

        # Credit + platform-reachability preflight. Fail-CLOSED after a
        # streak of unreachable ticks: an autonomous loop must not spend
        # a whole night blind (docs/autopilot/PLAN.md D4) — the default
        # agent-side credit machinery is fail-open by design and would
        # never stop us.
        gate = await self._credit_gate(routine, db, state, now)
        if gate is not None:
            return gate

        # ── Tier 2: one bounded agent turn ──────────────────────────
        return await self._run_tick(routine, run, db, cfg, state, goal, now)

    # ── credit / reachability gate ─────────────────────────────────
    async def _credit_gate(
        self, routine: Any, db: AsyncSession, state: Dict[str, Any], now: datetime,
    ) -> Optional[RoutineResult]:
        try:
            from app.services.credit_reporter import (
                _agent_key, _platform_endpoint, check_balance_remote,
                raise_if_exhausted,
            )
            from app.services.credit_exhausted import OutOfCreditsError
        except ImportError:  # pragma: no cover — stripped test envs
            return None

        try:
            raise_if_exhausted()
        except OutOfCreditsError:
            return await self._transition(
                routine, db, state, now, MISSION_BLOCKED, "insufficient_credits",
                notify_kind="mission_failed",
                notify_title=f"Autopilot mission “{routine.name}” paused",
                notify_body="You're out of credits — the mission will stay "
                            "paused until your balance recovers.",
                notify_priority="high",
            )

        # check_balance_remote returns None for BOTH "platform not
        # configured" (dev/monolith) AND "network failed" — fail-open
        # by design. Fail-closed only applies when we KNOW a platform
        # is configured and we still can't reach it.
        configured = bool(
            _platform_endpoint("/credits/preflight")
            and _agent_key()
            and routine.user_id
        )
        if not configured:
            return None

        try:
            preflight = await check_balance_remote(
                user_id=routine.user_id, required=1,
            )
            network_ok = preflight is not None
        except Exception:  # noqa: BLE001 — treat as unreachable
            network_ok = False

        if network_ok:
            if state.get("platform_fail_streak"):
                state["platform_fail_streak"] = 0
            return None

        streak = int(state.get("platform_fail_streak", 0)) + 1
        if streak >= settings.autopilot_platform_failclosed_threshold:
            return await self._transition(
                routine, db, state, now, MISSION_BLOCKED, "platform_unreachable",
                notify_kind="mission_failed",
                notify_title=f"Autopilot mission “{routine.name}” paused",
                notify_body="The platform has been unreachable for several "
                            "ticks, so the mission paused itself rather than "
                            "run unmetered.",
            )
        state["platform_fail_streak"] = streak
        state["next_due_at"] = _iso(now + timedelta(seconds=60))
        return RoutineResult(
            status="success", outcome="success_empty",
            new_watermark=state,
            metrics={"gate": "platform_unreachable", "streak": streak},
        )

    # ── the actual tick ────────────────────────────────────────────
    async def _run_tick(
        self, routine: Any, run: Any, db: AsyncSession,
        cfg: Dict[str, Any], state: Dict[str, Any], goal: str, now: datetime,
    ) -> RoutineResult:
        runner = self._agent_runner
        if runner is None or not hasattr(runner, "run"):
            # No agent runner wired (boot ordering) — try again shortly,
            # don't burn a tick.
            state["next_due_at"] = _iso(now + timedelta(seconds=60))
            return RoutineResult(
                status="success", outcome="success_empty",
                new_watermark=state, metrics={"gate": "runner_not_wired"},
            )

        ticks_run = int(state.get("ticks_run", 0)) + 1
        budget = float(cfg.get("budget_credits") or settings.autopilot_default_budget_credits)
        spent = float(state.get("spent_credits", 0.0))
        timeout_s = int(cfg.get("tick_timeout_s") or settings.autopilot_tick_timeout_seconds)
        job_id = getattr(run, "job_id", None) or getattr(run, "id", None)

        # One-shot: answers the user gave since the last tick (written
        # by the decision endpoint) are injected into THIS tick's
        # prompt, then cleared from state.
        user_answers = state.pop("user_answers", None) or []

        prompt = self._tick_prompt(
            routine, goal, state, ticks_run, budget, spent, user_answers,
        )

        # PR7 policy: channel='autopilot' (proven arbitrary-channel
        # path — sub-agents run as channel='subagent') keys THREE
        # policy layers keyed off the channel string: the connector
        # dispatcher's mutating-tool default-deny, the vault-tool
        # strip, and MCP pending_channel gating (tool_executor sets it
        # from the run channel). PromptProfile.AUTOPILOT adds the
        # profile deny set + the stripped prompt sections.
        try:
            from app.agent.prompt_profile import PromptProfile
            profile = PromptProfile.AUTOPILOT
        except (ImportError, AttributeError):  # pragma: no cover
            profile = None

        run_kwargs: Dict[str, Any] = dict(
            user_message=prompt,
            user_id=routine.user_id,
            channel="autopilot",
            current_job_id=job_id,
            telegram_chat_id=None,
            # Synthetic turn — the tick prompt must never render
            # as if the user typed it (2026-05-12 bug class).
            save_user_message=False,
        )
        if profile is not None:
            run_kwargs["prompt_profile"] = profile

        try:
            response = await asyncio.wait_for(
                runner.run(**run_kwargs),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return self._after_failed_turn(
                state, now, ticks_run, "tick_timeout",
                f"turn exceeded {timeout_s}s",
            )
        except Exception as e:  # noqa: BLE001 — mission failure is state
            return self._after_failed_turn(
                state, now, ticks_run, type(e).__name__, str(e)[:300],
            )

        # Meter the tick (self-imposed — works in shadow mode).
        tokens_in = int(getattr(response, "tokens_input", 0) or 0)
        tokens_out = int(getattr(response, "tokens_output", 0) or 0)
        model = getattr(response, "model", "") or ""
        spent += self._estimate_credits(model, tokens_in, tokens_out)

        markers = parse_tick_markers(getattr(response, "text", "") or "")
        tick_status = markers.get("status", "")
        summary = markers.get("summary") or ""
        note = markers.get("note") or ""

        # Monotonic mission progress (0-100): the model's self-estimate,
        # never allowed to move backwards — this value drives the
        # progress bar on the phone's Live Activity.
        reported = parse_progress_value(markers.get("progress"))
        progress = int(state.get("progress", 0) or 0)
        if reported is not None:
            progress = max(progress, reported)

        state.update({
            "ticks_run": ticks_run,
            "spent_credits": round(spent, 3),
            "last_tick_at": _iso(now),
            "last_summary": summary[:300] or state.get("last_summary", ""),
            "note": note[:2000] or state.get("note", ""),
            "progress": progress,
            "platform_fail_streak": 0,
        })

        if tick_status == "done":
            state["progress"] = 100
            return await self._transition(
                routine, db, state, now, MISSION_COMPLETED, "done",
                notify_kind="mission_completed",
                notify_title=f"✅ “{routine.name}” is done",
                notify_body=summary[:500] or "Mission completed.",
            )

        if tick_status == "blocked":
            question = summary[:300] or "Autopilot is blocked and needs your input to continue."
            approval_id = await self._create_approval(
                routine, db, question, {"tick": ticks_run, "note": note[:500]},
            )
            state["pending_approval_id"] = approval_id
            return await self._transition(
                routine, db, state, now, MISSION_WAITING_INPUT, "agent_blocked",
                notify_kind="needs_input",
                notify_title=f"“{routine.name}” needs you",
                notify_body=question,
                notify_priority="high",
                keep_enabled=True,
                notify_data_extra={"approval_id": approval_id},
            )

        # working (or missing marker — counts as a strike: the contract
        # is a hint, the strike system is the enforcement).
        prev_summary = (routine.last_state_json or {}).get("last_summary", "")
        made_progress = bool(tick_status == "working" and summary and summary != prev_summary)
        streak = 0 if made_progress else int(state.get("no_progress_streak", 0)) + 1
        state["no_progress_streak"] = streak

        if streak >= settings.autopilot_no_progress_blocker_threshold:
            return await self._transition(
                routine, db, state, now, MISSION_WAITING_INPUT, "no_progress",
                notify_kind="needs_input",
                notify_title=f"“{routine.name}” is stuck",
                notify_body=(
                    "Several ticks in a row made no visible progress. "
                    "Tell Autopilot how to proceed (or cancel the mission)."
                ),
                notify_priority="high",
                keep_enabled=True,
            )

        # Adaptive cadence: base interval, doubled per no-progress tick,
        # capped. Progress resets to base (docs/autopilot/PLAN.md D2).
        base_s = int(routine.schedule_interval_seconds
                     or settings.autopilot_base_interval_seconds)
        delay_s = min(
            base_s * (2 ** streak), settings.autopilot_backoff_cap_seconds,
        )
        state["next_due_at"] = _iso(now + timedelta(seconds=delay_s))

        # Progress heartbeat for the phone's Live Activity. The
        # dispatcher never turns this kind into an alert push — it only
        # moves the lock-screen/Dynamic-Island bar — so emitting every
        # working tick is safe. Best-effort, same stance as _transition.
        try:
            from app.services.agent_notify_client import notify
            await notify(
                event_kind="progress",
                title=f"Autopilot: {routine.name}"[:200],
                body=(summary or state.get("last_summary") or "Working…")[:300],
                data={
                    "route": "mission-control",
                    "mission_id": routine.id,
                    "mission_title": routine.name,
                    "progress": progress,
                    "tick": ticks_run,
                },
                priority="low",
                dedup_key=f"{routine.id}:progress",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("autopilot: progress notify failed for %s: %s",
                         routine.id, e)

        return RoutineResult(
            status="success",
            new_watermark=state,
            summary_message_id=getattr(response, "asst_message_id", None) or None,
            metrics={
                "tick": ticks_run, "tick_status": tick_status or "missing",
                "spent_credits": state["spent_credits"], "next_delay_s": delay_s,
                "tokens_in": tokens_in, "tokens_out": tokens_out,
            },
        )

    # ── helpers ────────────────────────────────────────────────────
    def _after_failed_turn(
        self, state: Dict[str, Any], now: datetime, ticks_run: int,
        error_class: str, detail: str,
    ) -> RoutineResult:
        """A crashed/timed-out turn is a mission-state event, not a
        handler failure: count the tick (it may have spent tokens),
        strike no-progress, back off, and keep going."""
        streak = int(state.get("no_progress_streak", 0)) + 1
        state.update({
            "ticks_run": ticks_run,
            "no_progress_streak": streak,
            "last_error": f"{error_class}: {detail}"[:300],
            "next_due_at": _iso(now + timedelta(seconds=min(
                settings.autopilot_base_interval_seconds * (2 ** streak),
                settings.autopilot_backoff_cap_seconds,
            ))),
        })
        return RoutineResult(
            status="success", new_watermark=state,
            metrics={"tick": ticks_run, "tick_status": "turn_error",
                     "error_class": error_class},
        )

    def _estimate_credits(self, model: str, tokens_in: int, tokens_out: int) -> float:
        try:
            from app.services.credit_service import tokens_to_credits
            return float(tokens_to_credits(model, tokens_in, tokens_out))
        except Exception:  # noqa: BLE001 — pricing table unavailable
            # Rough fallback: gpt-class blended ≈ $3/$15 per 1M.
            return (tokens_in * 0.0003 + tokens_out * 0.0015) / 10.0

    async def _create_approval(
        self, routine: Any, db: AsyncSession, title: str,
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Durable ask-the-user row (PR7) — what Mission Control lists
        and the decision endpoint resolves."""
        try:
            import uuid as _uuid
            from app.db.models import AutopilotApproval, APPROVAL_KIND_QUESTION

            approval_id = str(_uuid.uuid4())
            db.add(AutopilotApproval(
                id=approval_id,
                mission_id=routine.id,
                user_id=routine.user_id,
                kind=APPROVAL_KIND_QUESTION,
                title=title[:300],
                payload_json=payload,
            ))
            await db.commit()
            return approval_id
        except Exception as e:  # noqa: BLE001 — approval row is additive
            logger.warning("autopilot: approval row failed for %s: %s",
                           routine.id, e)
            return None

    def _tick_prompt(
        self, routine: Any, goal: str, state: Dict[str, Any],
        ticks_run: int, budget: float, spent: float,
        user_answers: Optional[list] = None,
    ) -> str:
        note = state.get("note") or "(none — this is your first tick)"
        last_summary = state.get("last_summary") or "(none yet)"
        answers_block = ""
        if user_answers:
            lines = "\n".join(
                f"- Q: {a.get('question', '?')}\n  A: {a.get('answer', '')}"
                for a in user_answers[-5:]
            )
            answers_block = (
                "\nThe user answered your earlier question(s) — act on "
                f"these now:\n{lines}\n"
            )
        return (
            "[Autopilot tick — autonomous background turn, the user is "
            "away and CANNOT reply right now]\n"
            f"Mission: {routine.name}\n"
            f"Goal: {goal}\n"
            f"Tick {ticks_run}; credits spent {spent:.1f} of {budget:.0f} "
            "budgeted.\n"
            f"Previous progress summary: {last_summary}\n"
            f"Your note from last tick: {note}\n"
            f"{answers_block}\n"
            "Advance the mission as far as you can THIS turn using your "
            "tools. Do not ask conversational questions — nobody will "
            "answer until the user returns. Do not send messages or take "
            "irreversible external actions; draft and prepare instead, "
            "and mark yourself blocked if the next step truly needs the "
            "user.\n\n"
            "End your reply with exactly these four lines:\n"
            "AUTOPILOT_STATUS: working | done | blocked\n"
            "AUTOPILOT_SUMMARY: <one line of user-visible progress — "
            "change it every tick you actually advance>\n"
            "AUTOPILOT_PROGRESS: <0-100 — honest percent of the WHOLE "
            "mission that is complete so far; never lower than last "
            "tick's number>\n"
            "AUTOPILOT_NOTE: <private note to your next tick: what's "
            "done, what's next>"
        )

    async def _transition(
        self, routine: Any, db: AsyncSession, state: Dict[str, Any],
        now: datetime, new_status: str, reason: str,
        *, notify_kind: str, notify_title: str, notify_body: str,
        notify_priority: str = "default", keep_enabled: bool = False,
        notify_data_extra: Optional[Dict[str, Any]] = None,
    ) -> RoutineResult:
        """Terminal/paused transition: stamp state, stop the tick loop
        (disable the routine unless a resume path needs it firing),
        and notify through the durable outbox."""
        state.update({
            "status": new_status,
            "status_reason": reason,
            "status_changed_at": _iso(now),
            "next_due_at": None,
        })

        if not keep_enabled:
            try:
                from app.db.models import Routine
                await db.execute(
                    update(Routine)
                    .where(Routine.id == routine.id)
                    .values(enabled=False, updated_at=now)
                )
                # The runner owns run-row commits, but the enabled flag
                # is mission-level and must stick even if the fire's
                # finalize path fails — _fire re-checks enabled at fire
                # start, so this is what stops job minting immediately.
                await db.commit()
            except Exception as e:  # noqa: BLE001
                logger.warning("autopilot: disable routine %s failed: %s",
                               routine.id, e)

        try:
            from app.services.agent_notify_client import notify
            await notify(
                event_kind=notify_kind,
                title=notify_title,
                body=notify_body,
                data={
                    "route": "mission-control",
                    "mission_id": routine.id,
                    "mission_title": routine.name,
                    "progress": int(state.get("progress", 0) or 0),
                    "status": new_status,
                    "reason": reason,
                    "urgent": bool((routine.config_json or {}).get("urgent")),
                    **(notify_data_extra or {}),
                },
                priority=notify_priority,
                dedup_key=f"{routine.id}:{new_status}:{reason}",
            )
        except Exception as e:  # noqa: BLE001 — notify is best-effort here
            logger.warning("autopilot: notify failed for %s: %s", routine.id, e)

        logger.info(
            "[autopilot] mission=%s → %s (%s) ticks=%s spent=%.2f",
            routine.id, new_status, reason,
            state.get("ticks_run", 0), float(state.get("spent_credits", 0.0)),
        )
        return RoutineResult(
            status="success",
            new_watermark=state,
            metrics={"transition": new_status, "reason": reason},
        )
