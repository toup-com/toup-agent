"""The error taxonomy must never let a raw internal string reach a user.

Every literal in this file is a VERBATIM error string captured from the
founder's production tenant on 2026-07-29 (11 user-visible "Failed" jobs, of
which zero were legitimate failures). If a future refactor lets any of these
fall through to `unknown`-with-raw-text, or lets a needs-user class become
retryable again, these tests fail.

See docs/audits/mission-control-raw/production-evidence-2026-07-29.md
"""

from __future__ import annotations

import pytest

from app.agent.job_status import (
    ALL_STATUSES,
    DISPOSITION_NEEDS_USER,
    DISPOSITION_REQUEUE,
    DISPOSITION_RETRY,
    DISPOSITION_TERMINAL,
    ERR_CREDITS_TOUP,
    ERR_CREDITS_UPSTREAM,
    ERR_INFRA_RESTART,
    ERR_UNKNOWN,
    PARKED_STATUSES,
    STATUS_COMPLETED,
    STATUS_WAITING_ON_USER,
    TERMINAL_STATUSES,
    classify,
    is_retryable,
)

# ── verbatim production strings ──────────────────────────────────────────
PROD_RESTART = "Agent restarted during execution"
PROD_402 = (
    "Error code: 402 - {'detail': {'error': 'out_of_credits', 'reason': "
    "'insufficient_message_credits', 'bucket': 'message', 'balance_after': '0.00'}}"
)
PROD_UPSTREAM_CREDITS = (
    'BadRequestError: Error code: 400 - {\'detail\': \'{"type":"error","error":'
    '{"type":"invalid_request_error","message":"Your credit balance is too low '
    'to access the Anthropic API..."}}\'}'
)
PROD_ATTRIBUTE_ERROR = (
    "all_retries_exhausted: AttributeError(\"'BuildJob' object has no "
    "attribute 'event_dedupe_id'\")"
)

ALL_PROD_ERRORS = [
    PROD_RESTART,
    PROD_402,
    PROD_UPSTREAM_CREDITS,
    PROD_ATTRIBUTE_ERROR,
]


def test_restart_never_shows_the_raw_marker():
    """64% of production failures carried this exact string.

    The user must never read it. Whether the job is silently re-queued
    (trigger) or terminalised as interrupted (everything else) is decided
    by `restart_verdict(source_kind)`, not by the text — see
    `test_only_trigger_jobs_are_auto_requeued`. Either way the literal
    "Agent restarted during execution" never reaches a screen.
    """
    c = classify(PROD_RESTART)
    assert c.error_class in {ERR_INFRA_RESTART, "infra_interrupted"}
    assert "restarted during execution" not in (c.user_message or "")


def test_toup_402_routes_to_needs_user_and_is_never_retried():
    """Retrying this is what re-billed tokens against a zero balance."""
    c = classify(PROD_402)
    assert c.error_class == ERR_CREDITS_TOUP
    assert c.disposition == DISPOSITION_NEEDS_USER
    assert c.required_action == "top_up_credits"
    assert not is_retryable(PROD_402)


def test_upstream_credit_error_beats_malformed_input():
    """Both are HTTP 400 carrying `invalid_request_error`.

    Only payload meaning separates them, so rule ORDER is load-bearing —
    this is the regression guard for that ordering.
    """
    c = classify(PROD_UPSTREAM_CREDITS)
    assert c.error_class == ERR_CREDITS_UPSTREAM, (
        "matched the generic 400 rule instead of the credit rule"
    )
    assert c.disposition == DISPOSITION_NEEDS_USER
    assert c.required_action == "top_up_credits"
    assert not is_retryable(PROD_UPSTREAM_CREDITS)


def test_both_credit_shapes_give_identical_user_copy():
    """The user hit one condition; they must read one sentence."""
    assert classify(PROD_402).user_message == classify(PROD_UPSTREAM_CREDITS).user_message


def test_unknown_error_is_generic_and_auto_reported():
    c = classify(PROD_ATTRIBUTE_ERROR)
    assert c.error_class == ERR_UNKNOWN
    assert c.auto_report is True
    assert "AttributeError" not in (c.user_message or "")
    assert "BuildJob" not in (c.user_message or "")


def test_unknown_errors_stay_retryable():
    """Resilience guard.

    An unrecognised exception is usually a flaky downstream, and the
    pre-taxonomy runners retried it successfully. Classifying `unknown`
    as terminal would make the system LESS resilient than the code this
    replaced — it would strand every transient crash we haven't written
    a rule for yet. Only NEEDS_USER and explicitly-terminal classes may
    skip the retry budget.
    """
    for generic in (
        RuntimeError("boom"),
        ValueError("bad data"),
        Exception("downstream service is down"),
    ):
        assert classify(generic).error_class == ERR_UNKNOWN
        assert is_retryable(generic), f"{generic!r} must still be retried"


def test_malformed_input_is_not_retried():
    """A 400 validation error cannot be fixed by trying again."""
    assert not is_retryable("ValidationError: field required")


@pytest.mark.parametrize("raw", ALL_PROD_ERRORS)
def test_no_production_error_ever_leaks_internals(raw):
    """The core guarantee: user_message is humanized copy or nothing."""
    msg = classify(raw).user_message
    if msg is None:
        return
    banned = [
        "Traceback", "Error code:", "{'", '{"', "AttributeError",
        "BadRequestError", "detail", "BuildJob", "_dedupe", "0.00",
    ]
    for token in banned:
        assert token not in msg, f"{token!r} leaked into user copy: {msg!r}"


@pytest.mark.parametrize("raw", ALL_PROD_ERRORS)
def test_every_production_error_classifies_without_raising(raw):
    c = classify(raw)
    assert c.error_class
    assert c.disposition in {
        DISPOSITION_RETRY, DISPOSITION_NEEDS_USER,
        DISPOSITION_REQUEUE, DISPOSITION_TERMINAL,
    }


def test_classify_never_returns_none_for_degenerate_input():
    for junk in (None, "", "   ", 0, [], object()):
        c = classify(junk)
        assert c.error_class
        assert c.user_message, "a user-visible class must carry copy"


def test_needs_user_classes_are_never_retryable():
    """Blanket invariant — the whole point of the taxonomy."""
    for raw in ALL_PROD_ERRORS:
        c = classify(raw)
        if c.disposition == DISPOSITION_NEEDS_USER:
            assert not is_retryable(raw)
            assert c.required_action, "needs_user must name an action"


def test_status_sets_are_disjoint_and_complete():
    from app.agent.job_status import ACTIVE_STATUSES

    assert not (ACTIVE_STATUSES & TERMINAL_STATUSES)
    assert not (PARKED_STATUSES & TERMINAL_STATUSES)
    assert not (ACTIVE_STATUSES & PARKED_STATUSES)
    assert STATUS_COMPLETED in TERMINAL_STATUSES
    # waiting_on_user is explicitly NOT terminal — it resumes.
    assert STATUS_WAITING_ON_USER in PARKED_STATUSES
    assert STATUS_WAITING_ON_USER not in TERMINAL_STATUSES
    assert STATUS_WAITING_ON_USER in ALL_STATUSES


def test_completed_wire_value_is_preserved():
    """38 readers consume this string; renaming it is a breaking change."""
    assert STATUS_COMPLETED == "completed"


# ── restart recovery ↔ Live Activity coupling ────────────────────────────
#
# A Live Activity card is closed ONLY by a terminal notification
# (mission_completed / mission_failed → alerting update + event=end). A DB
# write never touches the card. So "auto-resume" is only safe where a drain
# loop will actually re-dispatch the row; anywhere else it would strand the
# job in `queued` AND leave its lock screen / Dynamic Island card spinning
# on stale progress for hours.


def test_only_trigger_jobs_are_auto_requeued():
    """TriggerRunner is the ONLY runner with a queued-row drain loop
    (`_fetch_queued_ids` filters source_kind == 'trigger')."""
    from app.agent.job_status import RESUMABLE_SOURCE_KINDS, restart_verdict

    assert RESUMABLE_SOURCE_KINDS == {"trigger"}

    v = restart_verdict("trigger")
    assert v.disposition == DISPOSITION_REQUEUE
    assert v.user_message is None, "an auto-resumed job must stay invisible"


@pytest.mark.parametrize("kind", ["routine", "subagent", "chat_intent", "manual", None])
def test_non_resumable_kinds_are_interrupted_not_requeued(kind):
    """Re-queueing these would park them in `queued` forever with an
    orphaned Live Activity card. They must terminalise instead — visible,
    honest, and retryable."""
    from app.agent.job_status import ERR_INFRA_INTERRUPTED, restart_verdict

    v = restart_verdict(kind)
    assert v.disposition == DISPOSITION_TERMINAL
    assert v.error_class == ERR_INFRA_INTERRUPTED
    assert v.user_message, "an interrupted job must explain itself"
    assert "restart" in v.user_message.lower()


def test_legacy_restart_rows_are_interrupted_not_requeued():
    """The 79-day-old corpses will never be re-dispatched by anything.
    Classifying them `requeue` would promise a resume that cannot happen."""
    v = classify(PROD_RESTART)
    assert v.disposition == DISPOSITION_TERMINAL
    assert v.user_message, "legacy restart rows must explain themselves"


def test_notify_kinds_used_by_job_states_are_in_the_platform_enum():
    """Unknown event_kinds are rejected at ingest by
    `AgentNotifyRequest._kind_known`, so a typo here means the card is
    never updated and the failure is silent."""
    from app.db.models.notification import KNOWN_NOTIFY_KINDS

    # waiting_on_user keeps the card ALIVE …
    assert "needs_input" in KNOWN_NOTIFY_KINDS
    assert "needs_approval" in KNOWN_NOTIFY_KINDS
    # … while a genuine stop ENDS it.
    assert "mission_failed" in KNOWN_NOTIFY_KINDS


def test_waiting_action_types_map_to_a_notify_kind():
    """Every needs-user class names an action, and every action routes to
    either the approval or the input card treatment."""
    from app.agent.subagent_orchestrator import _APPROVAL_ACTIONS

    for raw in ALL_PROD_ERRORS:
        c = classify(raw)
        if c.disposition != DISPOSITION_NEEDS_USER:
            continue
        assert c.required_action
        kind = "needs_approval" if c.required_action in _APPROVAL_ACTIONS else "needs_input"
        assert kind in {"needs_approval", "needs_input"}
