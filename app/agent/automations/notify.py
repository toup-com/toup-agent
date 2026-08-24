"""Notification composition for automation outcomes (Round 28).

One composer instead of call-site f-strings. Everything rides
`agent_notify_client.notify` (tenant outbox → platform queue, the
outbox row id is the idempotency key), so a lost flush can never
double-push.

What is noteworthy — and what is not:
  - a terminal run that EXECUTED a write (`sent`, or `partial` when a
    skip-tolerant read was skipped but the writes landed) → push;
  - approval parking already pushes from the outbox
    (`_notify_needs_approval`) and auto-pause already pushes from the
    sweep — this module does not double-cover them;
  - a run that wrote nothing (no fresh events, filtered, undone) or a
    single failure below the auto-pause streak is NOT noteworthy: the
    session thread and the Activity page carry it, the lock screen
    does not.

The dedup key for completions is stable per automation (not per run):
the queue's default-priority window then collapses a chatty poll
automation to at most one push per ~30 minutes, while distinct days
and quiet automations always get theirs through.

Deep link contract (shared with the R28 app session): the push `data`
carries `route:"automation"`, `automation_id`, `run_id` (== the
BuildJob id, also stamped as `mission_id`), plus `chat_id`/`message_id`
when a run-card message exists. The platform's Live Activity link
builder maps `route:"automation"` to
`toup://automation?session=<automation_id>&run=<run_id>&mission=<id>`
server-side — producer-supplied URLs stay ignored for agent rows.

iOS routing reality (R28 app round, 2026-08-24): the app has NO
expo-notifications listener — a push routes only as a Live Activity
tap through the deepLinkUrl. A `mission_completed` with no prior
`mission_started` never mints an LA card, so THIS push is
informational on iOS (banner text, no in-app routing) until either a
run-lifecycle LA ships or the app grows a push listener. Do not gate
anything on the Expo `data` fields being consumed.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Outcomes that mean "the automation actually did something".
_NOTEWORTHY_OUTCOMES = frozenset({"sent", "partial"})


def outcome_summary(outcome: Optional[str], wrote_count: int = 0) -> str:
    """One human sentence for a terminal outcome — shared by the push
    body and the run card's session framing so the two never drift."""
    if outcome == "partial":
        return "Ran with some sources unavailable — the write still went out."
    if outcome == "sent":
        if wrote_count > 1:
            return f"Completed — {wrote_count} actions taken."
        return "Completed — action taken."
    if outcome == "undone":
        return "You undid this run before it executed."
    if outcome == "forbidden_tool":
        return "Blocked: automations never send mail — use a draft action."
    return "Finished."


async def notify_run_outcome(
    *,
    user_id: str,
    automation_id: str,
    automation_name: str,
    job_id: str,
    outcome: Optional[str],
    wrote_count: int = 0,
    chat_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> bool:
    """Push a noteworthy terminal outcome. Returns True when a push was
    enqueued, False when the outcome wasn't noteworthy or the enqueue
    failed (best-effort — never raises into the finalizer)."""
    if outcome not in _NOTEWORTHY_OUTCOMES:
        return False
    data: dict = {
        "kind": "automation",
        "route": "automation",
        "automation_id": automation_id,
        "run_id": job_id,
        "mission_id": job_id,
        # The session thread + Activity page are the durable record; a
        # push that reaches no device must not re-route through
        # Telegram/WhatsApp.
        "no_agent_fallback": True,
    }
    if chat_id:
        data["chat_id"] = chat_id
    if message_id:
        data["message_id"] = message_id
    try:
        from app.services.agent_notify_client import notify

        await notify(
            event_kind="mission_completed",
            title=f"{automation_name} ran",
            body=outcome_summary(outcome, wrote_count),
            data=data,
            priority="default",
            dedup_key=f"automation:{automation_id}:run_done",
        )
        return True
    except Exception as e:  # noqa: BLE001 — a push must never fail a run
        logger.warning(
            "[automations] outcome notify failed automation=%s job=%s: %s",
            automation_id[:8], job_id[:8], e,
        )
        return False
