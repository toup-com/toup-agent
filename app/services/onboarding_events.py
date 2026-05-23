"""
Structured-log events for the onboarding-v2 funnel.

Emits single-line key=value records that Promtail picks up and ships to
Loki; the admin observability surface queries them as
`{logger="onboarding.events"}`. Event names are namespaced with the
`onboarding.` prefix so they don't collide with the existing
`agent.`, `auth.`, `billing.` keys.

Schema is intentionally narrow — just what the brief asks for:
  * step_completed{step, user_id, duration_ms}
  * plan_selected{plan, user_id, was_default}
  * agent_ready{user_id, duration_ms, cold_start}

If you find yourself wanting to add a new event, also add it to the
allow-list in `app/api/onboarding_events.py::OnboardingEventName`
(the POST endpoint validates against it) and update the admin docs.
"""

from __future__ import annotations

import logging
from typing import Optional


# Dedicated logger so admin dashboards can filter on the logger name
# without pattern-matching every message body.
event_log = logging.getLogger("onboarding.events")


def _fmt_kv(payload: dict) -> str:
    # Compact key=value string; quoted values are avoided unless the
    # value contains whitespace, to keep Grafana queries readable.
    parts: list[str] = []
    for k, v in payload.items():
        if v is None:
            continue
        s = str(v)
        if any(c.isspace() for c in s):
            s = f'"{s}"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


def emit_step_completed(*, step: str, user_id: str, duration_ms: int) -> None:
    event_log.info(
        "onboarding.step_completed %s",
        _fmt_kv({"step": step, "user_id": user_id, "duration_ms": int(duration_ms)}),
    )


def emit_plan_selected(*, plan: str, user_id: str, was_default: bool) -> None:
    event_log.info(
        "onboarding.plan_selected %s",
        _fmt_kv({
            "plan": plan, "user_id": user_id,
            "was_default": "true" if was_default else "false",
        }),
    )


def emit_agent_ready(
    *, user_id: str, duration_ms: int, cold_start: bool,
    stage: Optional[str] = None,
) -> None:
    # Used by the PR 3 follow-up (`/api/agent-setup/ready` poll) to log
    # one record per onboarding when the container reaches `running`.
    # `cold_start=true` means we did not hit the warm pool.
    event_log.info(
        "onboarding.agent_ready %s",
        _fmt_kv({
            "user_id": user_id,
            "duration_ms": int(duration_ms),
            "cold_start": "true" if cold_start else "false",
            "stage": stage,
        }),
    )
