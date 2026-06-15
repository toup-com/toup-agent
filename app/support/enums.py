"""Canonical enums for the maintenance / support agent state machine.

Stdlib-only (no model/service imports) so ``app.db.models.support`` can
import these without an import cycle. DB columns store the ``.value``
strings; these enums are the single source of truth for them.
"""

from enum import Enum


class SupportIssueStatus(str, Enum):
    """The issue lifecycle. Forward-only except the request_changes loop.

    intake → classifying → classified → triaging → diagnosing →
    awaiting_approval → (approved → implementing → verifying → pr_open → done)
                       | (rejected) | (changes_requested → diagnosing ...)
    Terminal: done, rejected, parked, failed.
    """

    INTAKE = "intake"
    CLASSIFYING = "classifying"
    CLASSIFIED = "classified"
    TRIAGING = "triaging"
    DIAGNOSING = "diagnosing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    PR_OPEN = "pr_open"
    DONE = "done"
    # Not a BUG (missing-feature / unclear) — surfaced to admin, never fixed.
    PARKED = "parked"
    FAILED = "failed"


# Terminal states: no further automated transition happens from here.
TERMINAL_STATUSES = frozenset({
    SupportIssueStatus.DONE.value,
    SupportIssueStatus.REJECTED.value,
    SupportIssueStatus.PARKED.value,
    SupportIssueStatus.FAILED.value,
})


class SupportClassification(str, Enum):
    """Only BUG proceeds to the fix loop."""

    BUG = "bug"                      # broken behavior in shipped code
    MISSING_FEATURE = "missing_feature"  # works-as-built; never implemented
    UNCLEAR = "unclear"              # not enough info / ambiguous


class AdminDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupportEventType(str, Enum):
    CREATED = "created"
    CLASSIFIED = "classified"
    ROUTED = "routed"
    DIAGNOSED = "diagnosed"
    PARKED = "parked"
    DECISION = "decision"
    IMPLEMENT_STARTED = "implement_started"
    VERIFIED = "verified"
    PR_OPENED = "pr_opened"
    FAILED = "failed"
    NOTE = "note"
