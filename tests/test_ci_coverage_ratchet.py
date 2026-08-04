"""Every test file must be RUN by CI, or explicitly excused in writing.

On 2026-08-03 CI ran **49 of 305** backend test files. Nothing collected the
`tests/` directory — every invocation named files explicitly, so a file was
covered only if someone remembered to add it. The 256 nobody remembered had
rotted quietly: 27 were red, including tests asserting the existence of
functions that had been refactored away months earlier, and two stale pins
broken by my own #408 that CI could not see.

That is the failure this file exists to prevent. The platform-mode sweep now
runs `tests/` MINUS `COVERAGE_DEBT.txt`, so a new test file is covered the
moment it lands, and the only way to escape coverage is to write the file's
name and a reason into the debt list — where it is visible and can be paid off.

    cd backend && ENVIRONMENT=development python -m pytest tests/test_ci_coverage_ratchet.py
"""
from __future__ import annotations

import os
import pathlib
import re

os.environ.setdefault("ENVIRONMENT", "development")

BACKEND = pathlib.Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
DEBT = BACKEND / "tests" / "COVERAGE_DEBT.txt"


WORKFLOW = REPO / ".github" / "workflows" / "test-backend.yml"


def _debt_entries() -> list[str]:
    out = []
    for raw in DEBT.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _named_by_a_step() -> set[str]:
    """Test files a named step in the workflow already runs.

    Mirrors the sweep's own derivation exactly, including dropping YAML
    comment lines first — several comments in that file name test files in
    prose, and counting those as "already covered" would silently remove them
    from the sweep, which is the failure this whole mechanism exists to end.
    """
    out = set()
    for raw in WORKFLOW.read_text().splitlines():
        if raw.lstrip().startswith("#"):
            continue
        out.update(
            m.group(1)
            for m in re.finditer(r"tests/(test_[a-z0-9_]+\.py)", raw)
        )
    return out


def _excluded_dirs() -> list[str]:
    """Debt entries ending in `/` excuse a whole directory."""
    return [e for e in _debt_entries() if e.endswith("/")]


def _test_files() -> set[str]:
    """Every test file, INCLUDING ones in subdirectories.

    `glob("test_*.py")` would miss `tests/agent/test_system_prompt_assembly.py`
    entirely — which is exactly what happened when I first surveyed coverage,
    and that file turned out to be red. Paths are relative to tests/ so a
    subdirectory file is nameable in COVERAGE_DEBT.txt.
    """
    root = BACKEND / "tests"
    dirs = tuple(_excluded_dirs())
    return {
        rel for rel in (str(p.relative_to(root)) for p in root.rglob("test_*.py"))
        if not (dirs and rel.startswith(dirs))
    }


def test_every_debt_entry_still_exists():
    """A deleted test must be removed from the list, not left to rot in it."""
    dirs = _excluded_dirs()
    for d in dirs:
        assert (BACKEND / "tests" / d).is_dir(), (
            f"COVERAGE_DEBT.txt excuses the directory {d!r}, which does not exist"
        )
    missing = sorted(set(_debt_entries()) - set(dirs) - _test_files())
    assert not missing, (
        "COVERAGE_DEBT.txt names files that no longer exist — delete these "
        f"lines: {missing}"
    )


def test_debt_entries_are_unique():
    entries = _debt_entries()
    dupes = sorted({e for e in entries if entries.count(e) > 1})
    assert not dupes, f"duplicated in COVERAGE_DEBT.txt: {dupes}"


def test_every_debt_entry_carries_a_reason():
    """'Excused' has to mean something. A bare filename is not a reason."""
    bare = []
    for raw in DEBT.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition("#")
        if not reason.strip():
            bare.append(name.strip())
    assert not bare, (
        "these are excused from CI with no reason given — say why, so the "
        f"debt can be paid off: {bare}"
    )


def test_the_sweep_actually_runs_the_rest():
    """The whole point: the workflow must run tests/ minus the debt list.

    If someone replaces the sweep with another hand-maintained list of
    filenames, coverage silently reverts to opt-in and this file stops
    meaning anything.
    """
    text = WORKFLOW.read_text()
    assert "COVERAGE_DEBT.txt" in text, (
        "the workflow no longer reads COVERAGE_DEBT.txt — coverage has "
        "reverted to opt-in, which is how 256 files rotted unnoticed"
    )
    # The swept set must be DERIVED from the directory listing, not typed out.
    # (The sweep runs one process per file — see the workflow comment — so it
    # is not a single `pytest tests/` invocation.)
    #
    # `find`, not `ls`: `ls test_*.py` cannot see tests/agent/, and a test file
    # in a subdirectory that nothing runs is exactly how
    # agent/test_system_prompt_assembly.py rotted without anyone noticing.
    assert "find . -name 'test_*.py'" in text, (
        "the sweep no longer derives its file list from the tests/ directory "
        "with find. If it has become another hand-maintained enumeration — or "
        "gone back to a glob that cannot see subdirectories — coverage is "
        "opt-in again and this file stops meaning anything."
    )


def test_the_sweep_skips_only_what_a_named_step_already_runs():
    """The sweep must subtract named steps by DERIVING them from this file.

    Two things go wrong without it. Nineteen files ran twice; and eight of
    those ran the second time under the wrong RUN_MODE, because the
    unified-jobs step runs them with RUN_MODE=agent (they need AGENT_ONLY
    tables) and the sweep then re-ran them under platform, where
    "no such table: build_jobs" is guaranteed.

    A second hand-typed list would drift, so the sweep greps this workflow.
    """
    text = WORKFLOW.read_text()
    assert "tests/test_[a-z0-9_]+" in text, (
        "the sweep no longer derives the already-run set from this workflow "
        "file; if that became a second hand-maintained list it will drift"
    )
    assert re.search(r"grep -v '\^\[\[:space:\]\]\*#'.*test-backend\.yml", text, re.S), (
        "the named-step derivation must drop YAML comment lines first. "
        "Comments in this workflow name test files in prose; counting those "
        "as already-covered would remove real files from the sweep silently."
    )


def test_no_debt_entry_is_already_run_by_a_named_step():
    """An excuse for a file that CI already runs is a false excuse.

    It inflates the debt count, and it hides that the file is in fact covered
    — so nobody ever removes the line. Three entries were in exactly this
    state (test_ddl_plan, test_behavioral_suite_eval,
    test_tenant_schema_drift_selfheal): listed as agent-mode debt AND run by
    name in the schema-drift step.
    """
    both = sorted(set(_debt_entries()) & _named_by_a_step())
    assert not both, (
        "these are excused from the sweep but a named step already runs them "
        f"— delete the debt lines, they are covered: {both}"
    )


def test_the_sweep_still_has_most_of_the_suite_in_it():
    """A subtraction bug that empties the sweep must fail loudly, not quietly.

    `comm -23` on unsorted input, a `grep -c` that counts lines in an empty
    string as 1, a stray character in the debt list — every one of those
    shrinks the swept set silently, and a green build would then mean almost
    nothing was run. So put a floor under it.

    This is deliberately a floor, not an equality: files legitimately move
    between the sweep, the named steps and the debt list. 2026-08-03: 305
    files, 51 named, 50 excused, 204 swept.
    """
    named = _named_by_a_step()
    excused = set(_debt_entries())
    swept = _test_files() - named - excused
    assert len(swept) >= 180, (
        f"the sweep would run only {len(swept)} of {len(_test_files())} files. "
        "Something in the subtraction is wrong — a green build here would "
        "mean almost nothing ran."
    )


def test_debt_is_not_growing_silently():
    """A ceiling that must be lowered deliberately, never raised casually.

    2026-08-03 baseline: 43 (15 agent-mode, 28 quarantined). That number was
    measured by a classifier that silently stopped partway through the
    alphabet, so it was never the truth — CI then went red on 41 files it had
    never reached.

    2026-08-04: 72, after re-measuring all 310 files under the exact CI
    environment (`env -i`, no local `.env` leaking in). The rise is honesty,
    not decay: 23 of the additions were ALWAYS broken and simply unmeasured,
    and every one now carries a diagnosis instead of a shrug. Eight files were
    fixed outright in the same pass rather than excused.

    73 counts one DIRECTORY entry (`memverify/`, 15 files) as a single line —
    it is a live-Postgres package with its own runner, not unit-sweep work.

    74 added `test_toup_media_and_audio_first.py` from the media PRs
    (#432/#436), which landed on main while #423 was open. It is not debt in
    the "broken" sense: 27/27 green under RUN_MODE=agent, 7 red under platform
    because `media_playlists` is AGENT_ONLY. That entry ROUTES it to the right
    lane rather than excusing it.

    2026-08-04 (INFRA): 67. Three more PAID OFF, not excused —
    test_migration_021_roundtrip (had no collectible test at all, so pytest
    exited 5 and "no tests ran" looked like a pass), test_sqlite_test_infra
    (asserted the ambient DATABASE_URL rather than the default it means to
    protect), and test_search_fetch_cache (its stub had fallen behind both the
    SSRF guard's DNS lookup and the hand-rolled redirect loop).

    2026-08-04 (partitioning): 70. `test_table_partitioning` is PAID OFF, not
    excused — the 7 uncategorized tables it flagged are now categorized, so the
    test passes and its debt line is gone. That is the direction this number is
    supposed to move.

    2026-08-04 (follow-up): 71. Three more source-text assertions replaced with
    behavioural ones — each mutation-checked, i.e. shown to go red when the
    behaviour it claims to protect is actually broken. 71 and not 70: this
    change removes three entries from a list that had meanwhile grown by one,
    and taking the originally-planned 70 would silently re-forbid the media
    routing entry that #423 deliberately added.

    2026-08-04 (Baileys): 66. `test_whatsapp_baileys` is PAID OFF, not excused.
    Commit 056eaf25 replaced neonize with the Baileys Node sidecar and deleted
    the constructor-time disk probe the file pinned, so its session_status test
    asserted on a mechanism that no longer exists, and its ACL tests asserted
    set membership on `whatsapp_baileys_session` — a module with zero production
    importers — instead of the real gate. It is now 32 behavioural tests driving
    `start()`, the SSE state machine, the ACL/dedupe inbound path and the
    pairing controls through recorded HTTP calls and a spawn spy, all 25
    mutations confirmed red. The entry is deleted rather than re-laned: the
    file runs clean under RUN_MODE=platform, so the sweep picks it up.

    Lower this when you fix one. Raising it means shipping a test nobody runs,
    and should have to be argued for out loud in review.
    """
    CEILING = 66
    n = len(_debt_entries())
    assert n <= CEILING, (
        f"{n} files are now excused from the sweep, up from {CEILING}. "
        "Adding to COVERAGE_DEBT.txt means shipping a test nobody runs — "
        "fix the test, or raise this ceiling deliberately and say why."
    )


if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
