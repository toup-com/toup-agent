"""Round 15 — a build failure must never reach the user as machinery.

What a user was shown when their snake game failed to build::

    ❌ Build failed: Nokia Snake Classic — npm install failed (stale):
    npm install failed (exit 243): npm error code EACCES

Four layers quoting each other, an exit code, a POSIX errno and a
classification that was wrong. This suite pins the property, not the wording:
whatever comes out of `build_errors`, it contains no exit code, no errno, no
path, no command and no package-manager name — and it always ends in
something the reader can do.
"""

from __future__ import annotations

import re

import pytest

from app.agent import build_errors as be


# Every raw string a build has actually produced, plus the ones it can.
RAW_FAILURES = [
    "npm install failed (permissions): npm install failed (exit 243): "
    "npm error code EACCES\nnpm error syscall mkdir\n"
    "npm error path /opt/toup-agent/apps/nokia-snake-classic/node_modules",
    "npm install failed (stale): npm install failed (exit 243): npm error code EACCES",
    "npm install failed (bad_dep): npm error code ERESOLVE unable to resolve "
    "dependency tree while resolving: expo-router@3.4.8",
    "npm install failed (disk): npm error code ENOSPC no space left on device",
    "npm install failed (transient): npm error network ETIMEDOUT registry.npmjs.org",
    "npm install timed out after 300s in /opt/toup-agent/apps/x",
    "Bundle compilation failed after resume repair",
    "Planning failed on resume",
    "Traceback (most recent call last):\n  File \"/app/app/agent/skills/x.py\", "
    "line 42, in _build_app\n    raise RuntimeError('boom')\nRuntimeError: boom",
    "",
    None,
]

# Words and shapes that must never survive the translation.
_FORBIDDEN = [
    re.compile(r"\bexit\s*\d+", re.I),
    re.compile(r"\bE[A-Z]{3,}\b"),          # EACCES / ENOSPC / ETIMEDOUT / ERESOLVE
    re.compile(r"\bnpm\b|\bmetro\b|\bexpo\b|\byarn\b|\bnode_modules\b", re.I),
    re.compile(r"(^|\s)/[\w./-]+"),          # any absolute path
    re.compile(r"\bTraceback\b|\bRuntimeError\b|\bexception\b", re.I),
    re.compile(r"\bsyscall\b|\berrno\b", re.I),
    re.compile(r"\bstale\b|\bbad_dep\b|\bpermissions\b"),  # our own class names
]


@pytest.mark.parametrize("raw", RAW_FAILURES)
def test_no_machinery_ever_reaches_the_user(raw):
    msg = be.friendly_build_error(raw)
    for pattern in _FORBIDDEN:
        assert not pattern.search(msg), f"{pattern.pattern!r} leaked into: {msg!r}"
    assert msg and msg[0].isupper() and msg.rstrip().endswith(".")


@pytest.mark.parametrize("raw", RAW_FAILURES)
def test_every_failure_offers_a_next_step(raw):
    """"Try again" is the action for every recoverable class, and for the
    unrecoverable ones too — the alternative is a dead end."""
    assert "try again" in be.friendly_build_error(raw).lower()


def test_the_retry_chip_is_the_platform_chip():
    out = be.friendly_build_error_with_retry("npm error code EACCES")
    assert out.endswith("\n[[Try again]]")
    # One chip, on its own line: the renderer turns a line's chips into a
    # button row, and a chip buried mid-sentence leaves residue behind.
    assert out.count("[[") == 1 and "\n[[" in out


@pytest.mark.parametrize("raw,expected", [
    ("npm install failed (permissions): exit 243 EACCES", "permissions"),
    ("npm install failed (stale): npm error code EACCES", "permissions"),
    ("npm error code ENOSPC", "disk"),
    ("npm error code ERESOLVE", "bad_dep"),
    ("npm install timed out after 300s", "timeout"),
    ("Bundle compilation failed", "codegen"),
    ("something nobody has seen before", "unknown"),
])
def test_classification(raw, expected):
    assert be.classify_build_failure(raw) == expected


def test_a_stale_label_does_not_override_the_evidence():
    """The observed failure was mis-labelled `(stale)` by the old classifier.
    A wrong label in the text must not beat EACCES in the same string."""
    raw = "npm install failed (stale): npm error code EACCES"
    assert be.classify_build_failure(raw) == "permissions"
    assert "permissions" not in be.friendly_build_error(raw)
    assert "wasn't set up correctly" in be.friendly_build_error(raw)


def test_translation_is_idempotent():
    """The failure path translates twice — once onto the job row, once into
    the chat post, and the second usually receives the first's output. A
    second pass must not flatten a specific message to the generic one."""
    for raw in RAW_FAILURES:
        once = be.friendly_build_error(raw)
        assert be.friendly_build_error(once) == once, once


def test_distinct_classes_say_distinct_things():
    """A translation layer that answered every failure with one sentence
    would pass every test above and tell the user nothing."""
    msgs = {be.friendly_build_error(r) for r in [
        "EACCES", "ENOSPC", "ERESOLVE", "ETIMEDOUT", "timed out after 300s",
    ]}
    assert len(msgs) == 5
