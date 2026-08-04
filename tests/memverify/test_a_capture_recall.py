"""A. Capture — recall. Nothing important may be silently dropped.

Target: 100% recall on the labeled ground-truth set. Any miss is a failure.
"""

from __future__ import annotations

import pytest

from .conftest import record_metric
from .corpus import CAPTURE


@pytest.mark.parametrize("scenario", CAPTURE, ids=lambda s: s.id)
def test_capture_recall(labeled_run, scenario):
    entry = labeled_run[scenario.id]
    assert entry["error"] is None, f"pipeline raised: {entry['error']}"
    res = entry["result"]
    assert not res.missed, (
        f"\n{res.describe()}\n"
        f"note: {scenario.note or '-'}\n"
        f"turns: {[t.user[:70] for t in scenario.turns]}"
    )


def test_recall_aggregate(labeled_run):
    """The headline number, computed over every capture scenario."""
    total_facts = sum(len(s.must_store) for s in CAPTURE)
    found = sum(
        len(labeled_run[s.id]["result"].stored) for s in CAPTURE
    )
    recall = found / total_facts if total_facts else 1.0
    record_metric("recall_facts_total", total_facts)
    record_metric("recall_facts_found", found)
    record_metric("recall_pct", round(recall * 100, 2))

    misses = [
        f"{s.id}:{m}"
        for s in CAPTURE
        for m in labeled_run[s.id]["result"].missed
    ]
    assert recall == 1.0, (
        f"recall {recall:.1%} ({found}/{total_facts}); missed: {misses}"
    )


def test_recall_farsi_and_voice_subset(labeled_run):
    """Called out separately because these are the paths that historically
    regressed silently — a cross-lingual memory overlaps neither the user's nor
    the assistant's token set, which is exactly where the echo gate's margin
    rule has to abstain rather than delete."""
    subset = [s for s in CAPTURE if s.lang != "en" or "voice" in s.id]
    assert subset, "subset guard: corpus lost its Farsi/voice scenarios"
    misses = [
        f"{s.id}:{m}" for s in subset for m in labeled_run[s.id]["result"].missed
    ]
    record_metric("recall_farsi_voice_scenarios", len(subset))
    assert not misses, f"Farsi/voice recall misses: {misses}"
