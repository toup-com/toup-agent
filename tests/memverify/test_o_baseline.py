"""O. The headline numbers, gated against a committed baseline.

Categories A and B each assert their own half of the picture and record a
metric on the way past. Nothing compared those metrics to anything, so a
regression showed up as a different number in a JSON artifact nobody opened.
This file turns all three into one gate against `baseline.json`:

    recall_pct              of the facts that must be stored, how many were
    precision_pct           of the markers that must NOT be stored, how many stayed out
    unlabeled_rate_pct      of the rows that WERE stored, how many carry no usable label

The definitions live in `metrics.py` and are unit-tested with fixtures by
`tests/test_memverify_metrics.py`, which runs in CI's ordinary sweep. Only the
measurement needs the live stack; the arithmetic does not.
"""

from __future__ import annotations

from .conftest import record_metric
from .corpus import ALL_LABELED, CAPTURE
from .metrics import (
    GATED_METRICS,
    ScenarioCounts,
    check,
    format_report,
    load_baseline,
    summarize,
)


def _counts(labeled_run) -> list:
    capture_ids = {s.id for s in CAPTURE}
    out = []
    for sc in ALL_LABELED:
        entry = labeled_run[sc.id]
        res = entry["result"]
        out.append(
            ScenarioCounts(
                id=sc.id,
                must_store_total=len(sc.must_store),
                must_store_found=len(res.stored),
                must_not_store_total=len(sc.must_not_store),
                must_not_store_hit=len(res.junk),
                labels=tuple(tuple(pair) for pair in res.labels),
                counts_toward_recall=sc.id in capture_ids,
            )
        )
    return out


def test_headline_metrics_meet_the_committed_baseline(labeled_run):
    measured = summarize(_counts(labeled_run))
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


def test_the_label_sample_is_not_empty(labeled_run):
    """Anti-vacuity. `unlabeled_rate` is a ratio over stored rows; if the
    labeled run captured no rows at all the ratio is 0/0 and the gate above
    would read as a clean pass on a completely dead pipeline.

    `check()` already refuses a zero denominator, but asserting it here names
    the failure ("nothing was stored") instead of ("a rate was vacuous").
    """
    rows = sum(len(labeled_run[s.id]["result"].labels) for s in ALL_LABELED)
    record_metric("labeled_run_stored_rows", rows)
    assert rows > 0, (
        "the labeled corpus stored ZERO rows across all "
        f"{len(ALL_LABELED)} scenarios — every rate in this file would be "
        "vacuous. Check the extractor, not the metric."
    )
