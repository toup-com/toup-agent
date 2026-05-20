"""Regression pin — the three runner/intake files MUST NOT reference
the legacy ``TriggerEvent`` / ``RoutineRun`` ORM classes in code.

PR #49 of the unified-jobs cutover arc removed the legacy
``trigger_events`` and ``routine_runs`` dual-writes from:

  - ``app/agent/triggers/runner.py``
  - ``app/api/triggers_inbound.py``
  - ``app/agent/routines/runner.py``

PR #50 drops the tables (gated by ``ALLOW_LEGACY_JOB_TABLES_DROP=
true``). Between PR #49 landing and PR #50 deploying, a stray re-add
of a TriggerEvent/RoutineRun reference in any of these three files
would either resurrect a dual-write (defeating the cutover) or
crash at runtime once the tables are dropped.

This test grep-pins the three files for the regression. It tolerates
the strings appearing in comments and docstrings (deliberate
historical references explaining the removal) — only code-level
references (import, class instantiation, ``db.get(<class>, …)``,
``select(<class>)``, ``update(<class>)``) would actually re-introduce
the dependency, and those all require the class name to appear as
an identifier in non-comment/docstring source.

The check is a heuristic source-grep (cheap, fast, no full AST parse)
that rejects the symbols if any line that mentions them is NOT
inside a docstring/comment context. We use a small state machine
to track triple-quoted blocks so multi-line docstrings are skipped
correctly.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


# Paths are resolved relative to the backend/ root, which is where
# pytest is invoked from in CI.
_TARGET_FILES = [
    "app/agent/triggers/runner.py",
    "app/api/triggers_inbound.py",
    "app/agent/routines/runner.py",
]

_BANNED_SYMBOLS = ("TriggerEvent", "RoutineRun")


def _code_only_lines(text: str) -> list[tuple[int, str]]:
    """Yield (1-based lineno, source) for lines that are NOT inside a
    triple-quoted block and NOT pure comments. Inline-comment tails
    are stripped — only the code prefix on each line is returned.

    Not a full Python parser; sufficient for the source-grep
    regression check because legitimate code references to
    ``TriggerEvent`` / ``RoutineRun`` would appear as bare
    identifiers on a line that isn't a comment or docstring.
    """
    out: list[tuple[int, str]] = []
    in_block: str | None = None  # '"""' or "'''" when inside one
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw
        while line:
            if in_block is None:
                # Find the start of a triple-quoted string OR a #
                # comment. The earliest hit wins.
                tq_pos = -1
                tq_kind: str | None = None
                for q in ('"""', "'''"):
                    p = line.find(q)
                    if p >= 0 and (tq_pos < 0 or p < tq_pos):
                        tq_pos = p
                        tq_kind = q
                hash_pos = line.find("#")
                # ``#`` inside a string literal would be a false
                # positive, but the runner files don't use bare
                # ``#`` inside single-line strings other than format
                # specifiers (handled fine because the # check is
                # only used to drop the *tail*).
                if tq_pos >= 0 and (hash_pos < 0 or tq_pos < hash_pos):
                    # Emit the code prefix up to the opening triple
                    # quote, then enter the block.
                    prefix = line[:tq_pos]
                    if prefix.strip():
                        out.append((lineno, prefix))
                    rest = line[tq_pos + 3:]
                    # Did the docstring close on the same line?
                    close_pos = rest.find(tq_kind)  # type: ignore[arg-type]
                    if close_pos >= 0:
                        line = rest[close_pos + 3:]
                        continue
                    in_block = tq_kind
                    line = ""
                else:
                    # No triple quote on this line — strip the # tail.
                    if hash_pos >= 0:
                        line = line[:hash_pos]
                    if line.strip():
                        out.append((lineno, line))
                    line = ""
            else:
                # Inside a docstring block. Look for the close.
                close_pos = line.find(in_block)
                if close_pos >= 0:
                    in_block = None
                    line = line[close_pos + 3:]
                else:
                    line = ""
    return out


def _find_banned_references(file_path: Path) -> list[tuple[int, str, str]]:
    """Return (lineno, symbol, code) tuples for each banned symbol
    that appears as a bare identifier in non-comment / non-docstring
    source. Empty list when the file is clean."""
    text = file_path.read_text()
    code_lines = _code_only_lines(text)
    out: list[tuple[int, str, str]] = []
    for symbol in _BANNED_SYMBOLS:
        # Word-boundary match — ``RoutineRunner`` does NOT match
        # ``RoutineRun`` because the suffix ``ner`` extends the
        # identifier. Same for ``TriggerEventResponse`` (but that
        # type only lives in app/api/triggers.py — not in scope).
        pattern = re.compile(rf"\b{re.escape(symbol)}\b")
        for lineno, code in code_lines:
            if pattern.search(code):
                out.append((lineno, symbol, code.rstrip()))
    return out


def _backend_root() -> Path:
    # The test is invoked from backend/, so resolve relative to cwd.
    return Path(os.getcwd())


def test_triggers_runner_no_legacy_refs():
    """``app/agent/triggers/runner.py`` must contain zero code-level
    references to ``TriggerEvent`` or ``RoutineRun`` after PR #49."""
    file_path = _backend_root() / "app" / "agent" / "triggers" / "runner.py"
    hits = _find_banned_references(file_path)
    assert hits == [], (
        f"{file_path} contains banned legacy table references after PR #49 "
        f"cutover:\n" + "\n".join(
            f"  line {ln}: {sym} → {code}" for (ln, sym, code) in hits
        )
    )


def test_triggers_inbound_no_legacy_refs():
    """``app/api/triggers_inbound.py`` must contain zero code-level
    references to ``TriggerEvent`` or ``RoutineRun`` after PR #49."""
    file_path = _backend_root() / "app" / "api" / "triggers_inbound.py"
    hits = _find_banned_references(file_path)
    assert hits == [], (
        f"{file_path} contains banned legacy table references after PR #49 "
        f"cutover:\n" + "\n".join(
            f"  line {ln}: {sym} → {code}" for (ln, sym, code) in hits
        )
    )


def test_routines_runner_no_legacy_refs():
    """``app/agent/routines/runner.py`` must contain zero code-level
    references to ``TriggerEvent`` or ``RoutineRun`` after PR #49.

    NB: ``RoutineRunner`` is the class name in this file. The
    ``RoutineRun`` substring search MUST be word-boundary-aware so
    ``RoutineRunner`` doesn't trip the check (``Runner`` extends the
    identifier past ``RoutineRun``)."""
    file_path = _backend_root() / "app" / "agent" / "routines" / "runner.py"
    hits = _find_banned_references(file_path)
    assert hits == [], (
        f"{file_path} contains banned legacy table references after PR #49 "
        f"cutover:\n" + "\n".join(
            f"  line {ln}: {sym} → {code}" for (ln, sym, code) in hits
        )
    )
