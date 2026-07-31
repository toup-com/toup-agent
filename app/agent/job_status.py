"""Canonical job state machine and error taxonomy.

One source of truth for (a) the set of legal ``BuildJob.status`` values and
(b) how a raw exception becomes something a human can act on. Every retry
layer, every sweep, and every API validator must consult this module rather
than hard-coding a tuple — the four-value tuple in ``apps.py`` and the
no-default status map in the web ``BuildJobCard`` are exactly the drift this
module exists to prevent.

Why a taxonomy at all
---------------------
Audited on 2026-07-29 against the founder's live tenant: of 11 user-visible
"Failed" jobs, **zero** were legitimate failures. 7 were an agent restart, 3
were credit exhaustion arriving in three different raw payload shapes, and 1
was a fossil of a bug fixed months earlier. The three independent retry
layers (trigger, routine, LLM ladder) each retried blind, so a 402
``out_of_credits`` — which no amount of retrying can fix — was retried to
exhaustion, re-billing tokens each attempt.

The rule this encodes: an error's *meaning* decides its disposition, not its
transport status code. ``credits_upstream`` and ``malformed_input`` are both
HTTP 400; only payload inspection separates them, so ordering in
``classify`` is load-bearing.

Three persisted fields
----------------------
``error_class``      closed enum below — the routing key
``user_message``     humanized copy; **the only field the client may render**
``technical_detail`` internal-only, never serialized to the app

See docs/audits/mission-control-design.md §3.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Status values ────────────────────────────────────────────────────────
# `completed` is deliberately NOT renamed to `succeeded`: 38 readers across
# mobile, web, backend and tests consume this string, two of which hard-fail
# on an unknown value. The product word ("Done") is a display concern.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_WAITING_ON_USER = "waiting_on_user"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

# Legacy/adjacent values written elsewhere in the codebase. Listed so that
# validators accept them and sweeps can reason about them, NOT because new
# code should write them.
STATUS_PAUSED = "paused"            # app-builder token-limit pause/resume
STATUS_TIMEOUT = "timeout"          # sub-agent arc
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"  # sub-agent arc

#: Statuses a job can hold while still expected to make progress.
ACTIVE_STATUSES = frozenset({STATUS_QUEUED, STATUS_RUNNING})

#: Statuses that are not active but are not terminal either — the job is
#: parked pending something external. A restart must NOT disturb these.
PARKED_STATUSES = frozenset({STATUS_WAITING_ON_USER, STATUS_PAUSED})

#: Nothing further will happen without explicit user action.
TERMINAL_STATUSES = frozenset({
    STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED,
    STATUS_TIMEOUT, STATUS_BUDGET_EXHAUSTED,
})

ALL_STATUSES = ACTIVE_STATUSES | PARKED_STATUSES | TERMINAL_STATUSES


# ── Dispositions ─────────────────────────────────────────────────────────
#: Transient — retry with backoff + jitter.
DISPOSITION_RETRY = "retry"
#: Only a human can clear this. Route to `waiting_on_user`, NEVER retry:
#: retrying is what re-billed the founder's tokens against a zero balance.
DISPOSITION_NEEDS_USER = "needs_user"
#: Infrastructure. Re-queue idempotently; never user-visible.
#: ONLY valid where a drain loop will actually pick the row back up.
DISPOSITION_REQUEUE = "requeue"
#: Retrying cannot help and no user action would fix it.
DISPOSITION_TERMINAL = "terminal"


# ── Error classes ────────────────────────────────────────────────────────
#: Orphaned by a restart AND automatically re-queued. Invisible to the
#: user by contract — the job is still going to run.
ERR_INFRA_RESTART = "infra_restart"
#: Orphaned by a restart and NOT auto-resumable, because only
#: `source_kind='trigger'` has a queued-row drain loop
#: (triggers/runner.py::_fetch_queued_ids). Re-queueing anything else
#: strands it in `queued` forever with a Live Activity card that never
#: ends. User-visible with honest copy and a working Retry instead of a
#: silent loss.
ERR_INFRA_INTERRUPTED = "infra_interrupted"
#: The conversation turn that created this job ended before the job did.
#: Distinct from a restart: the container is healthy, the TURN went away —
#: the voice relay cancels its SSE generator the moment the caller hangs up
#: (api_v1.internal_agent_turn_stream), which takes the agent loop with it.
#: A job created by the `create_job` tool is narration for work the model
#: does inline, so when the turn dies nothing else will ever advance it.
ERR_TURN_INTERRUPTED = "turn_interrupted"
ERR_INFRA_UNRECOVERABLE = "infra_unrecoverable"
ERR_CREDITS_TOUP = "credits_toup"
ERR_CREDITS_UPSTREAM = "credits_upstream"
ERR_CONNECTOR_AUTH = "connector_auth"
ERR_RATE_LIMITED = "rate_limited"
ERR_UPSTREAM_UNAVAILABLE = "upstream_unavailable"
ERR_TIMEOUT = "timeout"
ERR_TOOL_FAILURE = "tool_failure"
ERR_MALFORMED_INPUT = "malformed_input"
ERR_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorClassification:
    """The complete verdict on one failure."""

    error_class: str
    #: Humanized copy. ``None`` means "never show this to the user at all"
    #: (infrastructure the user should never learn about).
    user_message: Optional[str]
    disposition: str
    #: For DISPOSITION_NEEDS_USER — the `required_action.type` to raise.
    required_action: Optional[str] = None
    #: Report to internal telemetry for root-causing.
    auto_report: bool = False

    @property
    def is_user_visible_failure(self) -> bool:
        return self.disposition == DISPOSITION_TERMINAL


# Ordered most-specific first. `classify` returns on the first match, so
# ordering IS the disambiguation for overlapping status codes.
_RULES: list[tuple[str, re.Pattern[str], ErrorClassification]] = []


def _rule(name: str, pattern: str, c: ErrorClassification) -> None:
    _RULES.append((name, re.compile(pattern, re.I | re.S), c))


#: Only these source kinds have a loop that re-dispatches `queued` rows
#: after a restart (``TriggerRunner._drain_loop`` → ``_fetch_queued_ids``,
#: which filters ``source_kind == 'trigger'``). Re-queueing any other kind
#: leaves the row parked forever, because every other dispatch path is
#: in-process and push-based: the routine scheduler mints a NEW job on its
#: next fire, sub-agents are spawned inline from a chat turn, and
#: app-builder resumption goes through `status='paused'` + checkpoint_json.
RESUMABLE_SOURCE_KINDS = frozenset({"trigger"})


def turn_interrupted() -> ErrorClassification:
    """The verdict for a job whose creating turn died before it finished.

    An accessor rather than a `classify()` call: classify() matches error
    TEXT against the rule table, so passing the bare class name silently
    falls through to the `unknown` copy — "Something went wrong. We've been
    notified" — on a task the user just watched the agent answer out loud.
    """
    return ErrorClassification(
        ERR_TURN_INTERRUPTED,
        "This stopped when the conversation ended before it finished. "
        "You can run it again.",
        DISPOSITION_TERMINAL,
    )


def restart_verdict(source_kind: Optional[str]) -> ErrorClassification:
    """How to treat a job orphaned by an agent restart.

    Auto-resume is only honest where something will actually pick the row
    back up. Everywhere else the user gets a truthful "interrupted" row
    with a Retry that works, rather than a job silently stuck in `queued`.
    """
    if source_kind in RESUMABLE_SOURCE_KINDS:
        return ErrorClassification(ERR_INFRA_RESTART, None, DISPOSITION_REQUEUE)
    return ErrorClassification(
        ERR_INFRA_INTERRUPTED,
        "This task stopped when your agent restarted. Nothing was lost — "
        "you can run it again.",
        DISPOSITION_TERMINAL,
    )


# -- infrastructure ------------------------------------------------------
# Matches the legacy free-text marker written before this module existed.
# Legacy rows are historical corpses that will never be re-dispatched, so
# they classify as interrupted (visible + retryable), NOT requeue — a
# requeue verdict on a 79-day-old row would be a lie.
_rule(
    "restart", r"restarted during execution|agent restart|orphaned by agent restart",
    ErrorClassification(
        ERR_INFRA_INTERRUPTED,
        "This task stopped when your agent restarted. Nothing was lost — "
        "you can run it again.",
        DISPOSITION_TERMINAL,
    ),
)

# The turn died, not the container. Matched ahead of the generic rules so
# the reaper's own stall text lands here rather than in `unknown` — a job
# the user watched the agent answer out loud must never read "Something
# went wrong. We've been notified."
_rule(
    "turn_interrupted",
    r"conversation ended|turn (was )?cancell?ed|no progress for \d+ minutes|stalled",
    ErrorClassification(
        ERR_TURN_INTERRUPTED,
        "This stopped when the conversation ended before it finished. "
        "You can run it again.",
        DISPOSITION_TERMINAL,
    ),
)

# -- needs the user ------------------------------------------------------
# Toup's own 402. Must precede the generic 4xx rules.
_rule(
    "credits_toup",
    r"out_of_credits|insufficient_message_credits|insufficient_integration_credits",
    ErrorClassification(
        ERR_CREDITS_TOUP,
        "Your agent ran out of Toup credits partway through this task.",
        DISPOSITION_NEEDS_USER, required_action="top_up_credits",
    ),
)
# The upstream provider's own balance error. Arrives as HTTP 400 with an
# `invalid_request_error` type, so it MUST be matched before malformed_input.
_rule(
    "credits_upstream",
    r"credit balance is too low|billing.{0,40}hard limit|quota.{0,20}exceeded|insufficient_quota",
    ErrorClassification(
        ERR_CREDITS_UPSTREAM,
        "Your agent ran out of Toup credits partway through this task.",
        DISPOSITION_NEEDS_USER, required_action="top_up_credits",
    ),
)
_rule(
    "connector_auth",
    r"skipped_reauth|invalid_grant|token (has been )?expired|reauth|unauthoriz|401",
    ErrorClassification(
        ERR_CONNECTOR_AUTH,
        "Toup needs you to reconnect an app before it can finish this.",
        DISPOSITION_NEEDS_USER, required_action="connector_auth",
    ),
)

# -- transient -----------------------------------------------------------
_rule(
    "rate_limited", r"rate.?limit|429|too many requests",
    ErrorClassification(
        ERR_RATE_LIMITED,
        "A service was busy — your agent is retrying.",
        DISPOSITION_RETRY,
    ),
)
_rule(
    "upstream", r"\b5\d\d\b|bad gateway|service unavailable|connection (reset|refused|error)|overloaded",
    ErrorClassification(
        ERR_UPSTREAM_UNAVAILABLE,
        "A service your agent relies on is having trouble.",
        DISPOSITION_RETRY,
    ),
)
_rule(
    "timeout", r"timed? ?out|timeouterror|deadline exceeded",
    ErrorClassification(
        ERR_TIMEOUT,
        "This took longer than expected and was stopped.",
        DISPOSITION_RETRY,
    ),
)

# -- terminal ------------------------------------------------------------
_rule(
    "malformed", r"validationerror|invalid_request_error|\b400\b|unprocessable",
    ErrorClassification(
        ERR_MALFORMED_INPUT,
        "Your agent couldn't understand this request.",
        DISPOSITION_TERMINAL,
    ),
)
_rule(
    "tool", r"toolerror|tool .{0,20}fail|mcp .{0,20}error",
    ErrorClassification(
        ERR_TOOL_FAILURE,
        "Your agent couldn't finish one of its steps.",
        DISPOSITION_RETRY,
    ),
)

# Unknown errors are RETRYABLE, deliberately.
#
# The narrow harm this module exists to prevent is retrying a NEEDS_USER
# error (credits, auth) — those are now classified and short-circuit. An
# unrecognised exception, by contrast, is most often a flaky downstream,
# and the pre-taxonomy runners retried it successfully. Classifying
# unknown as terminal would make the system LESS resilient than the code
# it replaces, and would strand every transient crash we haven't written
# a rule for yet.
#
# "Retryable" only means "spend the retry budget". Once the loop
# exhausts, the finaliser writes `failed` with this generic copy — the
# user never sees the raw exception either way.
_UNKNOWN = ErrorClassification(
    ERR_UNKNOWN,
    "Something went wrong. We've been notified and are looking into it.",
    DISPOSITION_RETRY, auto_report=True,
)


def classify(error: object) -> ErrorClassification:
    """Map any exception or error string to its taxonomy entry.

    Never raises and never returns ``None`` — an unmatched error becomes
    :data:`ERR_UNKNOWN` with generic copy and ``auto_report=True``, because
    the alternative (what shipped before this module) was rendering
    ``repr(exc)`` to the user.
    """
    if error is None:
        return _UNKNOWN
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
    if not text.strip():
        return _UNKNOWN
    for _name, pattern, verdict in _RULES:
        if pattern.search(text):
            return verdict
    return _UNKNOWN


def technical_detail(error: object, *, limit: int = 2000) -> str:
    """Internal-only diagnostic string.

    Never serialized to the client. Deliberately excludes message bodies and
    connector payloads — callers pass the exception, not the user's data.
    """
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"[:limit]
    return str(error)[:limit]


def is_retryable(error: object) -> bool:
    """True only for genuinely transient classes.

    The bug this replaces: all three retry layers retried *any* exception,
    so a 402 out-of-credits burned 3 routine attempts x the LLM ladder.
    """
    return classify(error).disposition == DISPOSITION_RETRY
