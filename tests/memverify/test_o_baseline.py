"""O. The headline numbers, gated against a committed baseline.

Categories A, B, G, J and K each assert their own half of the picture.
Nothing compared those to anything, so a regression showed up as a different
number in a JSON artifact nobody opened. This file turns all four headline
metrics into one gate against `baseline.json`:

    capture_pct      of the facts that must be in a file, how many are
    precision_pct    of the markers that must NOT be stored, how many stayed out
    lint_clean_pct   of the bullets and descriptions written, how many pass lint
    misroute_pct     of the facts captured, how many landed in the wrong file

The definitions live in `metrics.py` and are unit-tested with fixtures by
`tests/test_memverify_metrics.py`, which runs in CI's ordinary sqlite sweep.
Only the measurement needs the live stack; the arithmetic does not.
"""

from __future__ import annotations

from .conftest import record_metric
from .corpus import ALL_LABELED, CAPTURE_IDS
from .metrics import (
    GATED_METRICS,
    ScenarioCounts,
    check,
    format_report,
    load_baseline,
    summarize,
)


def counts_from(labeled_run) -> list:
    """Every scenario's contribution, including the BELT runs.

    A `[dirty]` entry is the same scenario driven with ws_chat's rewritten
    string. It counts for PRECISION and LINT — those are the properties the
    belt exists to test — and its captures are ignored for the capture rate,
    because a run whose input production never produces cannot be evidence
    that production captures anything.
    """
    out = []
    for sc in ALL_LABELED:
        for key in (sc.id, sc.id + "[dirty]"):
            entry = labeled_run.get(key)
            if entry is None:
                continue
            res = entry["result"]
            is_belt = key.endswith("[dirty]")
            out.append(
                ScenarioCounts(
                    id=key,
                    capture_total=len(sc.must_capture),
                    capture_found=len(res.captured),
                    misrouted=len(res.misrouted),
                    reject_total=len(sc.must_reject),
                    reject_hit=len(res.junk),
                    bullets_total=res.lint.bullets_total,
                    bullet_problems=len(res.lint.bullet_problems),
                    descriptions_total=res.lint.descriptions_total,
                    description_problems=len(res.lint.description_problems),
                    forbidden_slugs=len(res.forbidden_slugs),
                    cardinality_violations=len(res.cardinality),
                    counts_toward_capture=(sc.id in CAPTURE_IDS) and not is_belt,
                )
            )
    return out


def test_headline_metrics_meet_the_committed_baseline(labeled_run):
    measured = summarize(counts_from(labeled_run))
    for key, value in measured.items():
        record_metric(key, value)

    # Always printed, pass or fail — the artifact is machine-readable but the
    # console line is what a human actually sees.
    print("\n" + format_report(measured))

    violations = check(measured, load_baseline())
    assert not violations, (
        "memory verification headline numbers regressed against "
        "tests/memverify/baseline.json:\n\n"
        + "\n\n".join(f"  - {v}" for v in violations)
        + "\n\n"
        + format_report(measured)
    )


def test_every_gated_metric_has_a_baseline_entry():
    """The gate cannot be disarmed by deleting a line from baseline.json."""
    baseline = load_baseline()
    missing = [m for m in GATED_METRICS if m not in baseline]
    assert not missing, f"baseline.json has no entry for {missing}"


def test_the_writer_actually_wrote_something(labeled_run):
    """Anti-vacuity. Three of the four rates divide by something the WRITER
    produced. If the labeled run wrote no bullets at all, `lint_clean_pct`
    and `misroute_pct` are 0/0 and the gate above would read as a clean pass
    on a completely dead pipeline.

    `check()` already refuses a zero denominator, but asserting it here names
    the failure ("nothing was written") instead of ("a rate was vacuous").
    """
    bullets = sum(
        e["result"].lint.bullets_total for e in labeled_run.values()
    )
    record_metric("labeled_run_bullets", bullets)
    assert bullets > 0, (
        f"the labeled corpus wrote ZERO bullets across {len(labeled_run)} "
        "runs — every rate in this file would be vacuous. Check the writer, "
        "not the metric."
    )


def test_no_scenario_errored(labeled_run):
    """An exception inside a scenario yields an empty result, which reads as
    "nothing was stored" — indistinguishable from a correct rejection."""
    errored = {k: e["error"] for k, e in labeled_run.items() if e["error"]}
    assert not errored, f"scenarios raised: {errored}"
