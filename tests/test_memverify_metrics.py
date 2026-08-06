"""Unit tests for the memory-verification CI gate and its headline metrics.

Why these live outside tests/memverify/
--------------------------------------
`tests/memverify/` refuses to run without a live Postgres+pgvector, RUN_MODE=agent
and a real OPENAI_API_KEY — deliberately, so its assertions can never pass
vacuously. It is excused from CI's sweep for that reason
(tests/COVERAGE_DEBT.txt). Everything asserted here needs none of that: the CI
gate is a shell script, and the metric definitions are pure functions over
plain data. So this file runs in the ordinary sweep and is the thing that
actually keeps the gate honest.

The defect being pinned
-----------------------
The workflow step that runs the memory suite began:

    if [ -z "$OPENAI_API_KEY" ]; then
      echo "::warning title=memory verification skipped::..."
      exit 0
    fi

and `gh secret list` shows this repository has no OPENAI_API_KEY. That branch
was therefore taken on every build the step has ever had: the suite has never
run, and the step reported green each time, because an annotation that exits 0
is indistinguishable from a pass. These tests execute the real guard script
under every (secret x trigger) combination and assert the exit codes.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
GUARD = BACKEND / "scripts" / "memverify_ci_guard.sh"
WORKFLOW = REPO / ".github" / "workflows" / "test-backend.yml"
BASELINE = BACKEND / "tests" / "memverify" / "baseline.json"

# Obviously fake. Never a real key, and the guard must not echo it either way.
FAKE_KEY = "sk-NOT-A-REAL-KEY-memverify-guard-fixture"


# ── The CI gate ──────────────────────────────────────────────────────────


def run_guard(tmp_path, **env_overrides):
    """Execute the real guard script and return (returncode, stdout, decision)."""
    env = {
        k: v
        for k, v in os.environ.items()
        # Start from a clean slate for the two variables under test so a
        # developer's own exported key cannot make this pass.
        if k not in ("OPENAI_API_KEY", "MEMVERIFY_IS_FORK_PR", "GITHUB_OUTPUT")
    }
    out_file = tmp_path / "github_output"
    out_file.touch()
    env["GITHUB_OUTPUT"] = str(out_file)
    env.update({k: v for k, v in env_overrides.items() if v is not None})

    proc = subprocess.run(
        ["bash", str(GUARD)], env=env, capture_output=True, text=True, timeout=30
    )
    written = out_file.read_text()
    decision = None
    for line in written.splitlines():
        if line.startswith("decision="):
            decision = line.split("=", 1)[1]
    return proc.returncode, proc.stdout + proc.stderr, decision


def test_guard_script_exists_and_is_executable():
    assert GUARD.is_file(), f"{GUARD} is missing"
    assert os.access(GUARD, os.X_OK), f"{GUARD} is not executable"


def test_guard_requires_the_suite_when_the_secret_is_present(tmp_path):
    rc, out, decision = run_guard(tmp_path, OPENAI_API_KEY=FAKE_KEY)
    assert rc == 0, out
    assert decision == "run", out


def test_guard_never_echoes_the_key(tmp_path):
    _, out, _ = run_guard(tmp_path, OPENAI_API_KEY=FAKE_KEY)
    assert FAKE_KEY not in out, "the guard leaked the secret value into its output"


@pytest.mark.parametrize(
    "case,env",
    [
        # push to main: the expression renders "false"
        ("push_to_main", {"MEMVERIFY_IS_FORK_PR": "false"}),
        # same-repo pull request: head repo == this repo, so also "false"
        ("same_repo_pr", {"MEMVERIFY_IS_FORK_PR": "false"}),
        # workflow_dispatch / anything that never set the variable at all.
        # Fail CLOSED: an unset variable must not be read as "probably a fork".
        ("variable_absent", {}),
        # A secret that exists but is empty is the same as no secret.
        ("empty_secret", {"OPENAI_API_KEY": "", "MEMVERIFY_IS_FORK_PR": "false"}),
        # Defensive: only the exact string "true" buys an exemption.
        ("truthy_lookalike", {"MEMVERIFY_IS_FORK_PR": "TRUE"}),
        ("numeric_lookalike", {"MEMVERIFY_IS_FORK_PR": "1"}),
    ],
)
def test_guard_fails_when_the_secret_is_missing_off_a_fork(tmp_path, case, env):
    """This is the whole fix. Before it, every one of these exited 0."""
    rc, out, decision = run_guard(tmp_path, **env)
    assert rc != 0, (
        f"[{case}] the guard exited 0 with no OPENAI_API_KEY — the memory suite "
        f"did not run and the build would still be green.\n{out}"
    )
    assert decision != "run"
    assert "::error" in out, f"[{case}] no error annotation emitted:\n{out}"
    assert "OPENAI_API_KEY" in out


def test_guard_tolerates_a_fork_pull_request(tmp_path):
    """Forks cannot be given the secret by GitHub, so they must not be blocked
    — but the outcome has to be visibly a skip, not a pass, which is what
    `decision=skip` plus the `if:` on the suite step buys."""
    rc, out, decision = run_guard(tmp_path, MEMVERIFY_IS_FORK_PR="true")
    assert rc == 0, out
    assert decision == "skip", out
    assert "::notice" in out
    assert "::error" not in out


# ── The workflow wiring ──────────────────────────────────────────────────


def _all_steps():
    """Every step of every job.

    Deliberately not scoped to one job: the memory verification steps live in
    `pytest-postgres`, and an earlier draft of this file looked only at
    `pytest` — so `test_workflow_no_longer_swallows_a_missing_secret` passed by
    inspecting a job that never mentions OPENAI_API_KEY. A guard that reads the
    wrong half of the file is the same defect as a CI step that exits 0.
    """
    wf = yaml.safe_load(WORKFLOW.read_text())
    return [(job, step) for job, spec in wf["jobs"].items() for step in spec.get("steps", [])]


def _step_named(fragment: str):
    for _job, step in _all_steps():
        if fragment.lower() in str(step.get("name", "")).lower():
            return step
    return None


def test_the_workflow_scan_actually_sees_the_memory_steps():
    """Anti-vacuity for the two scans below. If the job structure moves and
    `_all_steps` stops reaching the memory verification steps, every workflow
    assertion in this file becomes a check on an empty set."""
    scripts = [str(s.get("run", "")) for _j, s in _all_steps()]
    assert any("OPENAI_API_KEY" in s for s in scripts) or any(
        "OPENAI_API_KEY" in str(s.get("env", {})) for _j, s in _all_steps()
    ), "the workflow scan reached no step that mentions OPENAI_API_KEY"


def test_workflow_gate_step_invokes_the_guard_and_forwards_the_secret():
    gate = _step_named("is the memory verification suite required")
    assert gate is not None, "the memverify gate step is gone from the workflow"
    assert gate.get("id") == "memverify_gate"
    assert "memverify_ci_guard.sh" in gate["run"]

    env = gate.get("env", {})
    assert "secrets.OPENAI_API_KEY" in env.get("OPENAI_API_KEY", "")
    fork_expr = env.get("MEMVERIFY_IS_FORK_PR", "")
    # The fork test must compare the PR's head repository against this one.
    # Without that comparison the exemption would apply to every event.
    assert "github.event.pull_request.head.repo.full_name" in fork_expr
    assert "github.repository" in fork_expr
    assert "github.event_name == 'pull_request'" in fork_expr


def test_workflow_suite_step_is_gated_on_the_guard_decision():
    step = _step_named("Run agent memory verification suite")
    assert step is not None, "the memory verification step is gone from the workflow"
    assert step.get("if") == "steps.memverify_gate.outputs.decision == 'run'", (
        "the suite step must be `if:`-gated on the guard, so a missing secret "
        "renders as a SKIPPED step rather than a green one"
    )


def test_workflow_no_longer_swallows_a_missing_secret():
    """The regression guard proper.

    The old shape was an inline `exit 0` on an empty OPENAI_API_KEY. Assert no
    step in the job can still reach `exit 0` from a `-z "$OPENAI_API_KEY"`
    test, in any step, however it is reworded.
    """
    offenders = []
    for job, step in _all_steps():
        script = str(step.get("run", ""))
        if "OPENAI_API_KEY" not in script:
            continue
        lines = [ln.strip() for ln in script.splitlines()]
        for i, line in enumerate(lines):
            if "-z" in line and "OPENAI_API_KEY" in line:
                window = " ".join(lines[i : i + 6])
                if "exit 0" in window:
                    offenders.append(f"{job}/{step.get('name')!r}: {window[:160]}")
    assert not offenders, (
        "a step still exits 0 when OPENAI_API_KEY is missing — that reads as a "
        "pass in the checks UI for a suite that did not run:\n"
        + "\n".join(offenders)
    )


# ── The metric definitions ───────────────────────────────────────────────


def _metrics():
    from tests.memverify.metrics import (  # noqa: PLC0415 - import cost is real
        CATCH_ALL,
        CATCH_ALL_BY_BRAIN,
        GATED_METRICS,
        LABELED,
        NON_CANONICAL,
        ScenarioCounts,
        check,
        label_verdict,
        load_baseline,
        summarize,
    )

    return dict(
        CATCH_ALL=CATCH_ALL,
        CATCH_ALL_BY_BRAIN=CATCH_ALL_BY_BRAIN,
        GATED_METRICS=GATED_METRICS,
        LABELED=LABELED,
        NON_CANONICAL=NON_CANONICAL,
        ScenarioCounts=ScenarioCounts,
        check=check,
        label_verdict=label_verdict,
        load_baseline=load_baseline,
        summarize=summarize,
    )


@pytest.mark.parametrize(
    "category,brain,expected",
    [
        # Canonical, discriminating -> labeled.
        ("identity", "user", "labeled"),
        ("health", "user", "labeled"),
        ("active_task", "user", "labeled"),
        ("corrections", "agent", "labeled"),
        # The catch-all bucket: stored fine, says nothing.
        ("other", "user", "catch_all"),
        ("domain_knowledge", "agent", "catch_all"),
        # WorkCategory has exactly one member, so `process` is not evidence of
        # a missing label — see metrics.py's docstring.
        ("process", "work", "labeled"),
        # Nothing at all.
        (None, "user", "non_canonical"),
        ("", "user", "non_canonical"),
        ("   ", "user", "non_canonical"),
        # The exact 2026-07-29 shape: pre-unification values the app has no
        # label for. They ALIAS to something canonical, but the value sitting
        # in the column is not canonical, so the app still renders "Other".
        ("schedule", "user", "non_canonical"),
        ("projects", "user", "non_canonical"),
        ("family", "user", "non_canonical"),
        # An outright unknown value.
        ("quux", "user", "non_canonical"),
        # A canonical USER value is meaningless on the AGENT brain.
        ("identity", "agent", "non_canonical"),
        # Blank brain_type defaults to the user brain.
        ("identity", "", "labeled"),
        ("other", None, "catch_all"),
    ],
)
def test_label_verdict(category, brain, expected):
    assert _metrics()["label_verdict"](category, brain) == expected


def test_catch_all_table_still_matches_the_taxonomy():
    """`CATCH_ALL_BY_BRAIN` is a copy of what `normalize_category` falls back
    to. If the taxonomy changes its fallback and this table does not, the
    unlabeled rate silently starts measuring the wrong thing."""
    from app.memory_taxonomy import normalize_category

    m = _metrics()
    for brain, expected in m["CATCH_ALL_BY_BRAIN"].items():
        assert normalize_category("a-value-no-taxonomy-will-ever-have", brain) == expected


def _scenario(SC, **kw):
    return SC(id=kw.pop("id", "X"), **kw)


def test_summarize_computes_all_three_headline_numbers():
    m = _metrics()
    SC = m["ScenarioCounts"]
    scenarios = [
        # A capture scenario: 4 facts wanted, 3 found. 4 rows stored, one of
        # them in the catch-all bucket.
        _scenario(
            SC,
            id="A01",
            must_store_total=4,
            must_store_found=3,
            labels=(
                ("identity", "user"),
                ("health", "user"),
                ("work", "user"),
                ("other", "user"),
            ),
            counts_toward_recall=True,
        ),
        # A junk scenario: 5 markers must stay out, 1 got in. It also left one
        # non-canonical row behind.
        _scenario(
            SC,
            id="B01",
            must_not_store_total=5,
            must_not_store_hit=1,
            labels=(("schedule", "user"),),
        ),
        # A scenario with must_store markers that does NOT count toward recall
        # (SENSITIVE K05 is exactly this shape). It must not move recall_pct.
        _scenario(
            SC,
            id="K05",
            must_store_total=10,
            must_store_found=0,
            must_not_store_total=1,
            must_not_store_hit=0,
            labels=(("health", "user"),),
            counts_toward_recall=False,
        ),
    ]
    got = m["summarize"](scenarios)

    assert got["recall_facts_total"] == 4
    assert got["recall_facts_found"] == 3
    assert got["recall_pct"] == 75.0

    assert got["junk_markers_total"] == 6
    assert got["junk_stored_count"] == 1
    assert got["precision_pct"] == round(100 * 5 / 6, 2)

    assert got["stored_rows_total"] == 6
    assert got["non_canonical_rows"] == 1
    assert got["catch_all_rows"] == 1
    assert got["unlabeled_rows"] == 2
    assert got["unlabeled_rate_pct"] == round(100 * 2 / 6, 2)
    assert got["non_canonical_rate_pct"] == round(100 * 1 / 6, 2)
    assert got["catch_all_rate_pct"] == round(100 * 1 / 6, 2)


def test_summarize_on_a_perfect_run():
    m = _metrics()
    SC = m["ScenarioCounts"]
    got = m["summarize"](
        [
            _scenario(
                SC,
                must_store_total=2,
                must_store_found=2,
                must_not_store_total=3,
                must_not_store_hit=0,
                labels=(("identity", "user"), ("goals", "user")),
                counts_toward_recall=True,
            )
        ]
    )
    assert got["recall_pct"] == 100.0
    assert got["precision_pct"] == 100.0
    assert got["unlabeled_rate_pct"] == 0.0
    # A perfect run clears both `contract` bounds in the REAL committed
    # baseline. It does not clear unlabeled_rate_pct, whose bound is null —
    # that is the point of the null, and it is asserted separately below.
    offenders = {
        v.metric for v in m["check"](got, m["load_baseline"]()) if v.metric != "unlabeled_rate_pct"
    }
    assert offenders == set(), offenders


# ── The baseline comparison ──────────────────────────────────────────────


def test_check_passes_a_run_that_meets_every_bound():
    m = _metrics()
    measured = {
        "recall_pct": 100.0,
        "precision_pct": 100.0,
        "unlabeled_rate_pct": 12.0,
        "non_canonical_rate_pct": 0.0,
        "recall_facts_total": 30,
        "junk_markers_total": 36,
        "stored_rows_total": 90,
    }
    baseline = {
        "recall_pct": {"min": 100.0},
        "precision_pct": {"min": 100.0},
        "unlabeled_rate_pct": {"max": 15.0},
        "non_canonical_rate_pct": {"max": 0.0},
    }
    assert m["check"](measured, baseline) == []


@pytest.mark.parametrize(
    "metric,value,kind",
    [
        ("recall_pct", 96.7, "min"),
        ("precision_pct", 97.2, "min"),
        ("unlabeled_rate_pct", 15.01, "max"),
        ("non_canonical_rate_pct", 1.1, "max"),
    ],
)
def test_check_reports_a_regression_on_each_gated_metric(metric, value, kind):
    m = _metrics()
    measured = {
        "recall_pct": 100.0,
        "precision_pct": 100.0,
        "unlabeled_rate_pct": 12.0,
        "non_canonical_rate_pct": 0.0,
        "recall_facts_total": 30,
        "junk_markers_total": 36,
        "stored_rows_total": 90,
    }
    measured[metric] = value
    baseline = {
        "recall_pct": {"min": 100.0},
        "precision_pct": {"min": 100.0},
        "unlabeled_rate_pct": {"max": 15.0},
        "non_canonical_rate_pct": {"max": 0.0},
    }
    violations = m["check"](measured, baseline)
    assert [v.metric for v in violations] == [metric]
    assert violations[0].kind == kind
    assert str(value) in str(violations[0])


def test_a_null_bound_is_a_failure_not_a_pass():
    """The same class of defect as the CI step itself: an unmeasured number
    must not read as a pass. The message has to carry the value to record."""
    m = _metrics()
    measured = {
        "recall_pct": 100.0,
        "precision_pct": 100.0,
        "unlabeled_rate_pct": 8.5,
        "non_canonical_rate_pct": 0.0,
        "recall_facts_total": 30,
        "junk_markers_total": 36,
        "stored_rows_total": 90,
    }
    baseline = {
        "recall_pct": {"min": 100.0},
        "precision_pct": {"min": 100.0},
        "unlabeled_rate_pct": {"max": None},
        "non_canonical_rate_pct": {"max": 0.0},
    }
    violations = m["check"](measured, baseline)
    assert [v.metric for v in violations] == ["unlabeled_rate_pct"]
    assert violations[0].kind == "unrecorded"
    assert "8.5" in str(violations[0])


@pytest.mark.parametrize(
    "denominator,metric",
    [
        ("recall_facts_total", "recall_pct"),
        ("junk_markers_total", "precision_pct"),
        ("stored_rows_total", "unlabeled_rate_pct"),
    ],
)
def test_check_refuses_a_vacuous_rate(denominator, metric):
    """ANTI-VACUITY. `summarize` returns 0.0 for an empty denominator, which
    would sail past `min: 100`? No — but `max: 15` on an empty store, and
    `precision 100%` over zero markers, both read as clean passes on a corpus
    that did nothing. A zero denominator is a failure."""
    m = _metrics()
    measured = {
        "recall_pct": 100.0,
        "precision_pct": 100.0,
        "unlabeled_rate_pct": 0.0,
        "non_canonical_rate_pct": 0.0,
        "recall_facts_total": 30,
        "junk_markers_total": 36,
        "stored_rows_total": 90,
    }
    measured[denominator] = 0
    baseline = {
        "recall_pct": {"min": 100.0},
        "precision_pct": {"min": 100.0},
        "unlabeled_rate_pct": {"max": 15.0},
        "non_canonical_rate_pct": {"max": 0.0},
    }
    violations = m["check"](measured, baseline)
    assert metric in [v.metric for v in violations]
    assert all(v.kind == "vacuous" for v in violations if v.metric == metric)


def test_check_reports_a_gated_metric_with_no_baseline_entry():
    m = _metrics()
    measured = {
        "recall_pct": 100.0,
        "precision_pct": 100.0,
        "unlabeled_rate_pct": 5.0,
        "non_canonical_rate_pct": 0.0,
        "recall_facts_total": 30,
        "junk_markers_total": 36,
        "stored_rows_total": 90,
    }
    baseline = {"recall_pct": {"min": 100.0}}
    violations = m["check"](measured, baseline)
    assert {v.metric for v in violations} == {
        "precision_pct",
        "unlabeled_rate_pct",
        "non_canonical_rate_pct",
    }
    assert all(v.kind == "missing" for v in violations)


# ── The committed baseline file ──────────────────────────────────────────


def test_baseline_file_is_valid_and_covers_every_gated_metric():
    m = _metrics()
    data = json.loads(BASELINE.read_text())
    metrics = data["metrics"]
    for name in m["GATED_METRICS"]:
        entry = metrics.get(name)
        assert isinstance(entry, dict), f"baseline.json has no entry for {name}"
        assert ("min" in entry) ^ ("max" in entry), f"{name}: exactly one of min/max"
        assert entry.get("source") in {
            "contract",
            "invariant",
            "measured",
            "unrecorded",
        }, f"{name}: `source` must say where the bound came from"
        assert entry.get("note"), f"{name}: a bound without a justification is a guess"


def test_baseline_marks_an_unrecorded_bound_as_unrecorded():
    """A null bound and a `source` of anything but `unrecorded` would be a
    modelled figure dressed as a measured one."""
    metrics = json.loads(BASELINE.read_text())["metrics"]
    for name, entry in metrics.items():
        bound = entry.get("min", entry.get("max"))
        if bound is None:
            assert entry["source"] == "unrecorded", name
        if entry.get("source") == "measured":
            assert entry.get("recorded_at"), f"{name}: `measured` needs a run id"


def test_the_repo_baseline_still_gates_a_regressed_run():
    """End-to-end over the REAL committed file: a run that misses one fact and
    stores one junk row must not pass."""
    m = _metrics()
    baseline = m["load_baseline"]()
    bad = {
        "recall_pct": 96.67,
        "precision_pct": 97.22,
        "unlabeled_rate_pct": 4.0,
        "non_canonical_rate_pct": 0.0,
        "recall_facts_total": 30,
        "junk_markers_total": 36,
        "stored_rows_total": 90,
    }
    violations = m["check"](bad, baseline)
    assert {v.metric for v in violations} >= {"recall_pct", "precision_pct"}
