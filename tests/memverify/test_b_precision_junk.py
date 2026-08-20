"""B. Precision — every bad memory from the dispatch, refused.

Each fixture in `corpus.JUNK` is a memory that actually reached the
founder's brain, quoted. The bar is zero: not "fewer than before".

Two runs per injected scenario. The CLEAN run is what production does — the
runner hands the writer `display_user_message`. The `[dirty]` run hands it
the string ws_chat actually builds, `[SYSTEM: The track "…"]` and all. Both
must refuse, and they are separable failures: clean-fails means the
durability rules are wrong, dirty-fails-only means the rules depend on
someone else having stripped the injection first.
"""

from __future__ import annotations

import pytest

from .conftest import record_metric
from .corpus import ALL_LABELED, JUNK
from .pipeline import lint_files


@pytest.mark.parametrize("sc", JUNK, ids=[s.id for s in JUNK])
def test_the_junk_is_not_stored(sc, labeled_run):
    res = labeled_run[sc.id]["result"]
    assert not labeled_run[sc.id]["error"], labeled_run[sc.id]["error"]
    assert not res.junk, "\n" + res.describe()


_INJECTED = [s for s in JUNK if any(t.injected for t in s.turns)]


@pytest.mark.parametrize("sc", _INJECTED, ids=[s.id for s in _INJECTED])
def test_the_junk_is_not_stored_even_when_the_injection_reaches_the_writer(
    sc, labeled_run
):
    """The BELT.

    Root cause #1 was structural — ws_chat rewrote `user_message` and the
    writer was handed the rewritten copy. v3 fixes that by passing
    `display_user_message`, pinned in tests/test_curator_producers.py. This
    asserts the OTHER half: the durability rules refuse a scraped title on
    their own merits, so the product is not one argument away from the
    original defect.
    """
    res = labeled_run[sc.id + "[dirty]"]["result"]
    assert not res.junk, "\n" + res.describe()


def test_junk_rate_is_zero_across_the_WHOLE_corpus(labeled_run):
    """Counted over every scenario, not just the JUNK ones: a capture
    scenario that also stores garbage is still a precision failure."""
    total = sum(len(s.must_reject) for s in ALL_LABELED)
    hit = sum(
        len(labeled_run[k]["result"].junk)
        for k in labeled_run
    )
    record_metric("b_junk_markers_total", total)
    record_metric("b_junk_stored", hit)
    assert total > 0, "the junk corpus is empty — this gate is vacuous"
    assert hit == 0, "\n".join(
        e["result"].describe() for e in labeled_run.values() if e["result"].junk
    )


def test_a_trivial_turn_writes_nothing_at_all(labeled_run):
    """B15. The pre-gate must decline BEFORE the model call: this is the
    single most common turn shape and paying for it is the difference
    between one LLM call per conversation and one per message."""
    res = labeled_run["B15-a-greeting-changes-nothing"]["result"]
    assert res.applied == 0, "\n" + res.describe()
    assert all(not b.strip() for b in res.bodies.values()), "\n" + res.describe()


def test_no_junk_scenario_invents_a_file(labeled_run):
    """A one-off must not mint a file either. An empty `topics/moo-meshki`
    with a real description is still the writer having believed a scraped
    title was a subject in this person's life."""
    from app.memory_files import SYSTEM_FILES

    offenders = {}
    for sc in JUNK:
        for key in (sc.id, sc.id + "[dirty]"):
            entry = labeled_run.get(key)
            if entry is None:
                continue
            extra = [s for s in entry["result"].slugs if s not in SYSTEM_FILES]
            if extra:
                offenders[key] = extra
    assert not offenders, offenders


def test_every_bullet_the_writer_produced_passes_the_voice_lint(labeled_run):
    """The lint is deterministic and runs INSIDE `validate_ops`, so a
    non-zero rate here does not mean the model wrote badly — it means
    something reached a body without going through the ops engine."""
    problems = []
    for entry in labeled_run.values():
        problems += entry["result"].lint.bullet_problems
    record_metric("b_lint_problems", len(problems))
    assert not problems, problems
