"""Pipeline orchestrator — drives the support issue state machine.

Two off-request-path entry points, each opening its OWN DB session (the
``_bg_finalize_signup`` pattern):

* ``run_diagnosis_pipeline``  — intake → classify → triage(route) →
                                diagnose → awaiting_approval | parked | failed.
* ``run_implementation``      — (approved) → implementing → verifying →
                                pr_open | failed.

Guardrails enforced here: only BUG proceeds past classification; a coverage
gap parks the issue with a "propose a new skill" note instead of fixing
blind; implementation runs only from the APPROVED state. The admin approval
gate lives between the two functions — nothing in run_implementation runs
until an admin calls the decision endpoint.
"""

from __future__ import annotations

import asyncio
import logging

from app.db import async_session_maker
from app.support import classifier, router as triage, diagnoser, implementer, repository as repo
from app.support import skills_index
from app.support.enums import (
    SupportIssueStatus as S,
    SupportClassification,
    SupportEventType as E,
)

logger = logging.getLogger("support.pipeline")

_background_tasks: set = set()


def spawn(coro) -> None:
    """Fire-and-forget with a strong ref (dodges GC), mirroring auth._spawn_background."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _llm_user(issue) -> str:
    return issue.reporter_user_id or "system-support"


async def run_diagnosis_pipeline(issue_id: str, *, actor_user_id: str | None = None) -> None:
    """Classify → route → diagnose. Idempotent enough to be re-run by /diagnose."""
    async with async_session_maker() as db:
        issue = await repo.get_issue(db, issue_id)
        if not issue:
            logger.warning("[support.pipeline] issue %s not found", issue_id)
            return
        try:
            # 1) CLASSIFY ---------------------------------------------------
            await repo.set_status(db, issue, S.CLASSIFYING)
            cls = await classifier.classify(
                user_id=_llm_user(issue),
                raw_report=issue.raw_report,
                repro_info=issue.repro_info or "",
            )
            issue.classification = cls.classification.value
            issue.classification_rationale = cls.rationale
            db.add(issue)
            await repo.set_status(
                db, issue, S.CLASSIFIED, event_type=E.CLASSIFIED,
                message=f"classified={cls.classification.value} (conf={cls.confidence:.2f})",
                detail={"classification": cls.classification.value, "confidence": cls.confidence,
                        "rationale": cls.rationale},
            )

            # Only BUG proceeds. Park everything else for the admin.
            if cls.classification is not SupportClassification.BUG:
                await repo.set_status(
                    db, issue, S.PARKED, event_type=E.PARKED,
                    message=f"Parked: {cls.classification.value} is not an auto-fixable bug.",
                    detail={"reason": cls.classification.value,
                            "surfaced_to_admin": True,
                            "note": ("Missing-feature / unclear reports are routed to the admin "
                                     "as feature requests or for clarification — never auto-fixed.")},
                )
                return

            # 2) TRIAGE / ROUTE --------------------------------------------
            await repo.set_status(db, issue, S.TRIAGING)
            issue.skills_baseline_sha = skills_index.baseline_sha()
            routing = await triage.route(
                user_id=_llm_user(issue),
                symptom=issue.symptom or issue.raw_report[:200],
                raw_report=issue.raw_report,
            )
            issue.routed_subsystems = routing.subsystems
            issue.routing_rationale = routing.rationale
            db.add(issue)

            if routing.coverage_gap or not routing.subsystems:
                # No skill covers it → propose a new skill, do NOT fix blind.
                await repo.set_status(
                    db, issue, S.PARKED, event_type=E.PARKED,
                    message="Parked: no subsystem skill covers this area (coverage gap).",
                    detail={"coverage_gap": True,
                            "gap_note": routing.gap_note,
                            "candidates": routing.candidates,
                            "action": "Propose a NEW skill for this area before fixing."},
                )
                return

            await repo.set_status(
                db, issue, S.DIAGNOSING, event_type=E.ROUTED,
                message=f"routed → {', '.join(routing.subsystems)}",
                detail={"subsystems": routing.subsystems, "rationale": routing.rationale,
                        "candidates": routing.candidates},
            )

            # 3) DIAGNOSE --------------------------------------------------
            prior_notes = issue.decision_notes if issue.decision == "request_changes" else ""
            dx = await diagnoser.diagnose(
                user_id=_llm_user(issue),
                symptom=issue.symptom or issue.raw_report[:200],
                raw_report=issue.raw_report,
                repro_info=issue.repro_info or "",
                subsystems=routing.subsystems,
                prior_decision_notes=prior_notes or "",
            )
            issue.diagnosis_report = dx.report
            issue.diagnosis_summary = dx.summary
            db.add(issue)

            if not dx.ok:
                await repo.set_status(
                    db, issue, S.FAILED, event_type=E.FAILED,
                    message="Diagnosis could not be produced (retry via /diagnose).",
                    detail={"skills_referenced": dx.skills_referenced},
                )
                return

            await repo.set_status(
                db, issue, S.AWAITING_APPROVAL, event_type=E.DIAGNOSED,
                message="Diagnosis ready — awaiting admin approval.",
                detail={"skills_referenced": dx.skills_referenced,
                        "confidence": dx.report.get("confidence"),
                        "affected_files": dx.report.get("affected_files")},
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("[support.pipeline] diagnosis failed for %s", issue_id)
            try:
                await repo.set_status(db, issue, S.FAILED, event_type=E.FAILED,
                                      message="Pipeline exception", detail={"error": str(e)})
            except Exception:
                pass


async def run_implementation(issue_id: str, *, admin_user_id: str | None = None) -> None:
    """Implement → verify → PR. Only valid from the APPROVED state."""
    async with async_session_maker() as db:
        issue = await repo.get_issue(db, issue_id)
        if not issue:
            return
        if issue.status != S.APPROVED.value:
            logger.warning("[support.pipeline] implement called on %s in status=%s (need approved)",
                           issue_id, issue.status)
            return
        if issue.classification != SupportClassification.BUG.value:
            await repo.set_status(db, issue, S.FAILED, event_type=E.FAILED,
                                  message="Refusing to implement a non-bug.")
            return

        await repo.set_status(db, issue, S.IMPLEMENTING, event_type=E.IMPLEMENT_STARTED,
                              actor="system",
                              message="Implementing approved fix on a fresh branch.")
        try:
            result = await implementer.implement(
                user_id=_llm_user(issue),
                issue_id=issue.id,
                symptom=issue.symptom or issue.raw_report[:200],
                diagnosis=issue.diagnosis_report or {},
            )
            issue.branch_name = result.branch
            issue.pr_url = result.pr_url
            issue.verification = result.verification
            db.add(issue)

            if not result.ok:
                issue.error = result.error or result.reason
                db.add(issue)
                await repo.set_status(
                    db, issue, S.FAILED, event_type=E.FAILED,
                    message=f"Implementation not completed: {result.reason}",
                    detail={"reason": result.reason, "error": result.error,
                            "edits_failed": result.edits_failed},
                )
                return

            await repo.set_status(
                db, issue, S.VERIFYING, event_type=E.VERIFIED,
                message="Verification passed.",
                detail={"verification": result.verification,
                        "edits_applied": result.edits_applied},
            )
            await repo.set_status(
                db, issue, S.PR_OPEN, event_type=E.PR_OPENED,
                message=f"PR opened: {result.pr_url or '(branch pushed; PR not opened)'}",
                detail={"branch": result.branch, "pr_url": result.pr_url},
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("[support.pipeline] implementation failed for %s", issue_id)
            issue.error = str(e)
            db.add(issue)
            await repo.set_status(db, issue, S.FAILED, event_type=E.FAILED,
                                  message="Implementation exception", detail={"error": str(e)})
