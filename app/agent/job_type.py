"""Job-type tag for the phone/web icon system (Round 3, item 2).

A `create_job` call carries a title (and usually a description) but no
machine-readable *kind* — every card wore the same generic mark. This
module derives one string tag from the intent words in the title so the
clients can pick an icon without re-implementing the heuristic:

    verify   — verify / fact-check / check / double-check / validate
    search   — search / find / look up / lookup / research
    write    — write / summarize / draft / compose
    compare  — compare / vs / versus / comparison
    generic  — anything else

Precedence is POSITIONAL, not by class: the earliest matching phrase in
the title wins ("Compare A vs B and write a summary" → compare; "Write a
comparison of A vs B" → write) — the leading verb is the job's intent.
The description is consulted only when the title yields nothing.

Deliberately NOT stored on ``BuildJob.job_type`` — that column is the
JobRunner HANDLER discriminator (``agent_task`` / ``auto_builder`` /
``vibe_code`` / ``subagent`` / ``autopilot_tick``) and dispatch keys on
it. The tag lives in ``BuildJob.config_json["job_type"]`` (JSONB, no
schema change) and rides the wire as ``job_type`` on the create_job
response, every ``job_update`` WS frame the create/update tools emit,
and the Live Activity content-state (``jobType``).

Pure, dependency-free, never raises.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

JOB_TYPE_VERIFY = "verify"
JOB_TYPE_SEARCH = "search"
JOB_TYPE_WRITE = "write"
JOB_TYPE_COMPARE = "compare"
JOB_TYPE_GENERIC = "generic"

JOB_TYPES: Tuple[str, ...] = (
    JOB_TYPE_VERIFY, JOB_TYPE_SEARCH, JOB_TYPE_WRITE, JOB_TYPE_COMPARE,
    JOB_TYPE_GENERIC,
)

_W = r"(?<![\w-])"
_E = r"(?![\w-])"

# (tag, regex). Word-ish boundaries that also stop at a hyphen so
# "double-check" and "fact-check" match as phrases and "checkout" /
# "research" (contains "search" — deliberately allowed, see below) are
# handled explicitly rather than by accident.
_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (JOB_TYPE_VERIFY, re.compile(
        _W + r"(?:verify|verifying|verification|fact[- ]?check(?:ing)?|"
        r"double[- ]?check(?:ing)?|cross[- ]?check(?:ing)?|check(?:ing)?|"
        r"validate|validating|confirm(?:ing)?)" + _E,
        re.IGNORECASE,
    )),
    (JOB_TYPE_SEARCH, re.compile(
        _W + r"(?:search(?:ing)?|find(?:ing)?|look(?:ing)? ?up|lookup|"
        r"research(?:ing)?|locate|locating|scan(?:ning)? for)" + _E,
        re.IGNORECASE,
    )),
    (JOB_TYPE_WRITE, re.compile(
        _W + r"(?:write|writing|summari[sz]e|summari[sz]ing|summary|"
        r"draft(?:ing)?|compose|composing|rewrite|rewriting)" + _E,
        re.IGNORECASE,
    )),
    (JOB_TYPE_COMPARE, re.compile(
        _W + r"(?:compare|comparing|comparison|vs\.?|versus)" + _E,
        re.IGNORECASE,
    )),
)


def _first_match(text: str) -> Optional[Tuple[int, str]]:
    """(position, tag) of the earliest matching phrase, or None."""
    best: Optional[Tuple[int, str]] = None
    for tag, rx in _RULES:
        m = rx.search(text)
        if m is None:
            continue
        if best is None or m.start() < best[0]:
            best = (m.start(), tag)
    return best


def classify_job_type(title: Optional[str], description: Optional[str] = None) -> str:
    """One of ``JOB_TYPES``. Title first; description only as fallback."""
    try:
        for text in (title, description):
            if not text:
                continue
            hit = _first_match(" ".join(str(text).split())[:400])
            if hit is not None:
                return hit[1]
        return JOB_TYPE_GENERIC
    except Exception:  # pragma: no cover — a tag must never break a job
        return JOB_TYPE_GENERIC


def normalize_job_type(value: Optional[str]) -> Optional[str]:
    """Accept a stored/wire value only if it is one of ours."""
    if not value:
        return None
    v = str(value).strip().lower()
    return v if v in JOB_TYPES else None
