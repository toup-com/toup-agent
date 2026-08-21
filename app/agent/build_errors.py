"""User-facing text for a failed app build.

A build failure reached the user as the tool's own words. On 2026-08-21 that
was, verbatim, in chat::

    ❌ Build failed: Nokia Snake Classic — npm install failed (stale):
    npm install failed (exit 243): npm error code EACCES

Four layers of machinery quoting each other, an exit code, a POSIX errno and
a misleading classification, for a person who asked for a snake game. None of
it is actionable by them and none of it is theirs to read.

This module is the single translation point: raw text in, ONE plain sentence
out, plus a retry affordance. The raw text keeps going to the log and to
``BuildJob.config_json['raw_error']`` — the operator surface — so nothing is
lost; it just stops being the user's problem.

Rules this file exists to enforce:

* never an exit code, errno, command line, file path, container path, job id
  or app id;
* never the word "npm", "Metro", "Expo" or any other machinery the user did
  not choose;
* always something the reader can DO, which for every recoverable class is
  "try again" — and for the unrecoverable ones is still "try again", because
  the alternative is a dead end.
"""

from __future__ import annotations

import re
from typing import Optional

#: The chip the chat surfaces render as a button. Tapping it sends the label
#: as an ordinary message, which the agent handles as a rebuild request —
#: the same path any "try again" in chat already takes.
RETRY_CHIP = "[[Try again]]"

#: One sentence per failure class. Present tense, no jargon, ends in a next
#: step. Kept short: this renders inside a chat bubble and a job card banner.
_CLASS_MESSAGES: dict[str, str] = {
    # "permissions" is deliberately absent from the sentence: it is a word
    # from the machine's side of the wall, and it is also this module's own
    # class name, which the leak test bans on sight.
    "permissions": (
        "The workspace for this app wasn't set up correctly, so the build "
        "couldn't save its files. I've repaired it — try again."
    ),
    "transient": (
        "The build couldn't reach the library server. That's usually "
        "temporary — try again."
    ),
    "bad_dep": (
        "One of the building blocks this app asked for isn't available in the "
        "version it wanted. Try again and I'll pick a different one."
    ),
    "disk": (
        "The server ran out of space while building this app. Try again in a "
        "few minutes."
    ),
    "stale": (
        "Leftovers from an earlier build got in the way. I've cleared them — "
        "try again."
    ),
    "timeout": (
        "The build took longer than it's allowed to and I stopped it. Try "
        "again — a smaller first version usually gets through."
    ),
    "codegen": (
        "I couldn't finish writing the app's code this time. Try again."
    ),
    "unknown": (
        "Something went wrong while building this app. Try again."
    ),
}

#: Every sentence this module can emit, for the idempotence check below.
_TRANSLATED: frozenset = frozenset(_CLASS_MESSAGES.values())

#: Signals for classes the install classifier does not see (they happen in
#: other phases). Ordered — first match wins.
_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("timeout", ("TIMED OUT", "TIMEOUT")),
    ("permissions", ("EACCES", "EPERM", "PERMISSION DENIED", "ROOT-OWNED FILES")),
    ("disk", ("ENOSPC", "ENOMEM", "NO SPACE LEFT")),
    ("bad_dep", ("ERESOLVE", "ETARGET", "NO MATCHING VERSION", "404 NOT FOUND",
                 "PEER DEP", "UNABLE TO RESOLVE")),
    ("transient", ("ETIMEDOUT", "ECONNRESET", "EAI_AGAIN", "FETCH_ERROR",
                   "NETWORK", "503")),
    ("stale", ("ENOENT", "LOCK")),
    ("codegen", ("CODE GENERATION", "PLANNING FAILED", "BUNDLE COMPILATION",
                 "NO FILES GENERATED")),
)


def classify_build_failure(raw: str) -> str:
    """Best-effort class for any build failure string. Never raises."""
    upper = (raw or "").upper()
    if not upper.strip():
        return "unknown"
    # EVIDENCE FIRST. An "(<class>)" marker from AppManager's own classifier
    # is useful — it saw the whole npm output where this sees a truncated
    # tail — but it can also be WRONG, and it was: the observed failure was
    # labelled "(stale)" while the very same string said EACCES. Trusting the
    # label over the errno would reproduce the original misdiagnosis, so the
    # marker is only consulted when nothing in the text speaks for itself.
    for label, needles in _PATTERNS:
        if any(n in upper for n in needles):
            return label
    m = re.search(r"\((permissions|transient|bad_dep|disk|stale|unknown)\)", raw or "")
    if m and m.group(1) != "unknown":
        return m.group(1)
    return "unknown"


def friendly_build_error(raw: str, classification: Optional[str] = None) -> str:
    """One plain sentence for the user. Raw text NEVER passes through.

    Unrecognised input degrades to the generic sentence rather than leaking
    the original — the failure mode this function exists to remove is exactly
    "we didn't have a message for that one, so we showed them the traceback".

    Idempotent by construction: an already-translated sentence comes back
    unchanged rather than being re-classified as "unknown" and flattened to
    the generic one. The failure path has two translation points (the job row
    and the chat post) and the second usually receives the first's output.
    """
    text = (raw or "").strip()
    if text in _TRANSLATED:
        return text
    label = classification if classification in _CLASS_MESSAGES else None
    if label is None:
        label = classify_build_failure(raw)
    return _CLASS_MESSAGES.get(label, _CLASS_MESSAGES["unknown"])


def friendly_build_error_with_retry(raw: str, classification: Optional[str] = None) -> str:
    """:func:`friendly_build_error` plus the retry chip on its own line."""
    return f"{friendly_build_error(raw, classification)}\n{RETRY_CHIP}"
