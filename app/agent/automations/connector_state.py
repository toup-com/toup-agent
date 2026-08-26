"""Connector state — the `connector.state` frame + §4.7 auto-resume.

CONTRACTS-R30 §4.7/§6: no connector state change was visible to clients
before this round. When the platform's OAuth callback (or an expiry
flip in the vault) lands on the tenant's `_connector_connected` hook,
this module:

  - broadcasts ONE `connector.state` frame
    `{account_id, connector_id, state, reconnected_at}` — B's in-app
    OAuth sheet dismisses on it, so the frame goes out FIRST and is
    best-effort like every other frame (no live socket is normal);
  - on a reconnect, auto-resumes: every automation of this user
    blocked on that connector gets its checkpointed stopped run
    resumed (`run_v3.resume_run` — "It picked up where it stopped"
    must be literally true) or, when it was paused `connector_reauth`,
    re-armed through `service.arm_automation` (whose post-commit
    routine nudge fires the catch-up), with the RECONNECTED note
    (`stamp: reconnected`) appended to its thread;
  - on an expiry, emits the `expired` frame and pauses NOTHING — the
    run-level failure path owns pausing, and flipping automations here
    would double-report the same outage.

Constraints this module lives under:
  - Frames carry NO `channel` key (the app's frame filter drops
    channeled frames) and ride `ws_chat.broadcast_to_user` — the same
    idiom as `cards.broadcast_card`.
  - `account_id == connector_id` verbatim today (§1); both keys ride
    the frame so the app never parses one out of the other.
  - Every per-automation sub-step wears its OWN try/except: one
    automation's failed recovery must never block a sibling's (the
    separate-try/except-per-fallback repo convention).
  - AGENT_ONLY: reads `automations` + `build_jobs` + the v3 ledger
    tables, none of which exist in the platform image.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation, BuildJob

logger = logging.getLogger(__name__)


# Substrings that mark a failed run's `user_message` as reauth-shaped.
# Deliberately loose (§4.7 dispatch: "keep the query simple") — a false
# positive costs one extra RECONNECTED note on a thread whose connector
# just reconnected, which reads as honest rather than wrong.
_REAUTH_HINTS = ("access", "token", "reconnect", "expired", "reauth")


async def emit_state_frame(
    user_id: str,
    *,
    connector_id: str,
    state: str,
    reconnected_at: Optional[object] = None,
    reason_code: str = "",
    fix: str = "",
) -> None:
    """Broadcast one `connector.state` frame. Best-effort; the durable
    record is the identity row on the platform, not this frame.

    `reconnected_at` accepts a datetime (serialized ISO + "Z") or an
    already-formatted string; None rides as null on the expiry leg.
    NO `channel` key — same try/except idiom as `cards.broadcast_card`.
    """
    at = reconnected_at
    if isinstance(at, datetime):
        at = at.isoformat() + "Z"
    frame = {
        "type": "connector.state",
        "account_id": connector_id,
        "connector_id": connector_id,
        "state": state,
        "reconnected_at": at,
        # R31 §4.4 — additive. The R30 frame carried a two-value `state`
        # and nothing else, so a client learned that something was wrong
        # and could not say what or offer the fix; the platform hook's
        # own `error` field ("reauth_required" vs "provider_down") was
        # accepted at the boundary and discarded, which made a dead
        # token and an outage the same event.
        "reason_code": reason_code or "",
        "fix": fix or ("retry" if state == "connected" else "reconnect"),
    }
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(user_id, frame)
    except Exception as e:  # noqa: BLE001 — no live socket is normal
        logger.debug("[connector_state] frame broadcast skipped: %s", e)


def _references_connector(automation: Automation, connector_id: str) -> bool:
    """Does this automation's spec touch the connector?

    The denormalised `connector_id` column covers the trigger side; the
    quoted-JSON substring covers read/write steps in either spec
    version without parsing. Quoted, so `"git"` can never match
    `"github"`.
    """
    if (automation.connector_id or "") == connector_id:
        return True
    return f'"{connector_id}"' in (automation.spec_json or "")


def _reauth_shaped_failure(job: BuildJob) -> bool:
    """A failed run whose error reads like lost access (§4.7)."""
    if (job.status or "").lower() != "failed":
        return False
    if (job.error_class or "") != "tool_error":
        return False
    msg = (job.user_message or "").lower()
    return any(h in msg for h in _REAUTH_HINTS)


async def _latest_run(
    db: AsyncSession, automation_id: str,
) -> Optional[BuildJob]:
    return (
        await db.execute(
            select(BuildJob)
            .where(BuildJob.source_id == automation_id)
            .where(BuildJob.job_type == "automation_run")
            .order_by(BuildJob.created_at.desc())
            .limit(1)
        )
    ).scalars().first()


async def on_connector_connected(
    db: AsyncSession, *, user_id: str, connector_id: str,
) -> dict:
    """The §4.7 auto-resume, called when a (re)connect lands.

    Returns `{"resumed": [run ids], "rearmed": [automation ids],
    "noted": [automation ids]}`. Every sub-step is best-effort with its
    own try/except so one automation's failure never blocks another's
    recovery; the caller treats the whole thing as a hook, not a
    transaction.
    """
    now = datetime.utcnow()
    result: dict = {"resumed": [], "rearmed": [], "noted": []}

    # (a) The frame first — B's OAuth sheet dismisses on it, and the
    # recovery below can take seconds (a resumed run executes inline).
    await emit_state_frame(
        user_id, connector_id=connector_id, state="connected",
        reconnected_at=now,
    )

    # (b) Every automation of this user blocked on this connector.
    try:
        automations = (
            await db.execute(
                select(Automation).where(Automation.user_id == user_id)
            )
        ).scalars().all()
    except Exception as e:  # noqa: BLE001 — recovery is best-effort
        logger.warning(
            "[connector_state] automation scan failed for user=%s: %s",
            user_id[:8], e,
        )
        return result

    from app.agent.automations import ledger, run_v3

    for automation in automations:
        if not _references_connector(automation, connector_id):
            continue

        paused_on_reauth = (
            automation.status in ("paused", "error")
            and automation.paused_reason == "connector_reauth"
        )

        stopped_job: Optional[BuildJob] = None
        reauth_failed = False
        try:
            latest = await _latest_run(db, automation.id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[connector_state] latest-run lookup failed for %s: %s",
                automation.id, e,
            )
            latest = None
        if latest is not None:
            if (
                (latest.status or "").lower() == "cancelled"
                and (latest.outcome or "").lower() == "stopped"
            ):
                stopped_job = latest
            elif _reauth_shaped_failure(latest):
                reauth_failed = True

        # A PARTIAL run holding this account in `accounts_failed` is
        # reason enough on its own — the automation is neither paused
        # nor stopped nor "reauth-shaped failed", it simply came back
        # missing one source, which is the whole point of §4.2a.
        has_failed_source = await _has_failed_source(
            db, automation_id=automation.id, connector_id=connector_id,
        )
        if not (paused_on_reauth or stopped_job is not None
                or reauth_failed or has_failed_source):
            continue

        # (c1) RECONNECTED note. Own try/except: the note is honesty,
        # not a precondition for the resume. Silently skipped when the
        # automation has no thread (pre-v3 rows).
        try:
            thread = await ledger.thread_for(db, automation.id)
            if thread is not None:
                await ledger.append_turn(
                    db, user_id=user_id, thread=thread,
                    run_id=stopped_job.id if stopped_job is not None else None,
                    kind="note",
                    payload={
                        "stamp": "reconnected",
                        "at": now.isoformat() + "Z",
                    },
                )
                result["noted"].append(automation.id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[connector_state] RECONNECTED note failed for %s: %s",
                automation.id, e,
            )

        # (c0) §4.2a — the PER-SOURCE resume, and the first thing tried.
        #
        # The R30 auto-resume below is whole-run: it restarts a stopped
        # run or re-arms a paused automation. Neither describes what
        # actually happens when one of five accounts is fixed — the
        # other four already read successfully, and re-reading them is
        # both wasted work and a different brief.
        try:
            resumed_source = await _resume_failed_source(
                db, user_id=user_id, automation=automation,
                connector_id=connector_id,
            )
            if resumed_source:
                result["resumed"].append(resumed_source)
                continue
        except Exception as e:  # noqa: BLE001 — one automation's merge
            # never blocks another's recovery
            logger.warning(
                "[connector_state] per-source resume failed for %s: %s",
                automation.id, e,
            )

        # (c2) A checkpointed stopped run exists → resume it under the
        # same run id (§4.3). resume_run re-reads anything the dedupe
        # window moved past, so "picked up where it stopped" is honest.
        if stopped_job is not None:
            try:
                res = await run_v3.resume_run(db, job_id=stopped_job.id)
                if res.get("resumed"):
                    result["resumed"].append(stopped_job.id)
                else:
                    logger.info(
                        "[connector_state] resume declined for run %s: %s",
                        stopped_job.id, res.get("error"),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[connector_state] resume failed for run %s: %s",
                    stopped_job.id, e,
                )
            continue

        # (c3) …no stopped run, but the automation sits paused on this
        # connector → re-arm it. arm_automation keeps the post-commit
        # routine nudge, so the catch-up fire needs nothing extra here.
        if paused_on_reauth:
            try:
                from app.agent.automations import service
                await service.arm_automation(
                    db, automation_id=automation.id, user_id=user_id,
                )
                result["rearmed"].append(automation.id)
            except Exception as e:  # noqa: BLE001 — logged, not raised
                logger.warning(
                    "[connector_state] re-arm failed for %s: %s",
                    automation.id, e,
                )

    return result


async def on_connector_expired(
    db: AsyncSession, *, user_id: str, connector_id: str,
    error: str = "",
) -> None:
    """An identity flipped to reauth_required/provider_down (§4.7).

    Emits the `expired` frame and deliberately pauses NOTHING: the
    run-level failure path owns pausing (a run that actually hits the
    dead token pauses with `connector_reauth` and its own honest turn),
    and the summary/workflow readers derive `needs_attention` from the
    identity state itself. Flipping automations here would mark
    automations that might never have fired during the outage.

    `db` is accepted for signature parity with the connected leg (and
    for the day this hook grows an honest read); it is not written.
    """
    # R31 §4.4: the hook's `error` reaches the frame now. It was
    # accepted at the route boundary and never read, so a revoked
    # connection, an expired token and a provider outage all arrived as
    # the single word "expired" — and the last of those is transient,
    # so the account should not have moved off `connected` at all.
    from . import account_health
    code = account_health.classify(error or "reauth_required", error or "")
    state, fix = account_health.state_for_reason(code)
    try:
        await account_health.record_use(
            db, user_id=user_id, account_id=connector_id, ok=False,
            reason_code=code, message=error or "",
        )
        await db.commit()
    except Exception as e:  # noqa: BLE001 — the frame still goes out
        logger.debug("[connector_state] expiry health write skipped: %s", e)
    await emit_state_frame(
        user_id, connector_id=connector_id, state=state,
        reconnected_at=None, reason_code=code, fix=fix,
    )


# ─── WIRING (coordinator) ────────────────────────────────────────────
# app/api/automations.py::connector_connected_hook (the
# `POST /_connector_connected` route, ~line 554) — after the card
# updates, inside the `async with async_session_maker() as db:` block,
# add exactly:
#
#     from app.agent.automations import connector_state
#     if body.ok:
#         await connector_state.on_connector_connected(
#             db, user_id=_user_id(), connector_id=body.connector_id)
#     else:
#         await connector_state.on_connector_expired(
#             db, user_id=_user_id(), connector_id=body.connector_id)
#
# The platform now also POSTs `{"connector_id", "ok": false, "error":
# "reauth_required"}` from `connector_vault.mark_reauth_required`
# (fire-and-forget), so the `ok=False` leg is live traffic, not just
# a failed-OAuth signal. `connector_health_probe.py:219` flips
# identities to provider_down and is NOT hooked yet (file owned
# elsewhere) — `connector_vault.notify_agent_connector_state` is
# importable for exactly that call site.


async def _has_failed_source(
    db: AsyncSession, *, automation_id: str, connector_id: str,
) -> bool:
    """Does this automation's last run hold this account as failed?"""
    from app.agent.automations import ledger
    job = await _latest_run(db, automation_id)
    if job is None:
        return False
    failed = (ledger._cfg_of(job).get("accounts_failed") or [])
    return connector_id in failed


async def _resume_failed_source(
    db: AsyncSession, *, user_id: str, automation, connector_id: str,
) -> Optional[str]:
    """§4.2a. Returns the run id merged into, or None.

    A run older than a day gets a fresh `run_now` instead — merging
    yesterday's reads into yesterday's brief is not what the user
    expects from fixing an account today.
    """
    from app.agent.automations import executor_v2, ledger
    job = await _latest_run(db, automation.id)
    if job is None:
        return None
    if connector_id not in (ledger._cfg_of(job).get("accounts_failed") or []):
        return None
    out = await executor_v2.resume_source(
        db, automation=automation, job_id=job.id, account_id=connector_id,
    )
    if out.get("resumed"):
        return job.id
    if out.get("reason") == "too_old":
        try:
            from app.agent.automations.service import parse_spec_live
            vspec = await parse_spec_live(automation)
            from app.agent.automations.spec_v2 import ValidatedSpecV2
            if isinstance(vspec, ValidatedSpecV2):
                import uuid as _uuid
                source = vspec.schedule_source() or (
                    vspec.sources[0] if vspec.sources else None)
                if source is not None:
                    await executor_v2.run_schedule_fire_v2(
                        db, automation, vspec, source,
                        fire_key=f"reconnect:{_uuid.uuid4()}",
                        run_kind="run_now",
                    )
                    return job.id
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[connector_state] catch-up fire failed for %s: %s",
                automation.id, e,
            )
    return None
