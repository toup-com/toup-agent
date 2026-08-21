"""A. Capture — the durable facts the corpus says must be in a file, are.

The unit is a BULLET IN A NAMED FILE. Round 8's corpus could only ask "is
this text stored somewhere", which cannot see the failure that motivated
half this rebuild: a fact filed under the wrong subject (a `people/` file
about the account owner, a play request filed as a preference).
"""

from __future__ import annotations

import pytest

from .conftest import record_metric
from .corpus import CAPTURE, SENSITIVE
from .pipeline import bullets_of

_POSITIVE = CAPTURE + SENSITIVE


@pytest.mark.parametrize("sc", _POSITIVE, ids=[s.id for s in _POSITIVE])
def test_the_labeled_fact_is_in_the_right_file(sc, labeled_run):
    res = labeled_run[sc.id]["result"]
    assert not labeled_run[sc.id]["error"], labeled_run[sc.id]["error"]
    assert not res.missed, "\n" + res.describe()
    assert not res.misrouted, "\n" + res.describe()


@pytest.mark.parametrize("sc", _POSITIVE, ids=[s.id for s in _POSITIVE])
def test_a_positive_scenario_stores_no_junk_of_its_own(sc, labeled_run):
    """Precision is not only the JUNK scenarios' business. A turn that
    carries a real fact AND a one-off request must store the first and
    refuse the second — which is the harder case, because the model has
    already decided the turn is worth writing about."""
    res = labeled_run[sc.id]["result"]
    assert not res.junk, "\n" + res.describe()


def test_capture_rate(labeled_run):
    total = sum(len(s.must_capture) for s in _POSITIVE)
    found = sum(len(labeled_run[s.id]["result"].captured) for s in _POSITIVE)
    record_metric("a_capture_markers_total", total)
    record_metric("a_capture_markers_found", found)
    assert total > 0, "the capture corpus is empty — this gate is vacuous"
    assert found == total, "\n".join(
        labeled_run[s.id]["result"].describe()
        for s in _POSITIVE if labeled_run[s.id]["result"].missed
    )


def test_a_restated_fact_is_rewritten_not_accumulated(labeled_run):
    """P03: three phrasings of "I'm vegetarian" over three turns.

    §2.2's whole merge rule in one number. Round 8 stored five rows for five
    phrasings and called the dedup pass a success because none of them was a
    byte-for-byte duplicate; the file model's answer is that the writer is
    SHOWN the body and chooses `rewrite` over `add`.
    """
    res = labeled_run["P03-merge-does-not-append"]["result"]
    matching = [
        b for body in res.bodies.values() for b in bullets_of(body)
        if "vegetarian" in b.lower() or "meat" in b.lower()
    ]
    record_metric("a_vegetarian_bullets", len(matching))
    assert len(matching) == 1, (
        "three phrasings of one fact left "
        f"{len(matching)} bullets:\n" + res.describe()
    )


def test_a_contradiction_resolves_newest_wins_or_says_it_is_unresolved(labeled_run):
    """P04: Toronto, then Vancouver.

    §1.3 allows exactly two correct answers — the new value replaces the old
    one, or the file records the conflict explicitly. What is NOT allowed is
    both values sitting there as peers, which is what the row corpus left
    behind and what makes a memory system worse than no memory system.
    """
    res = labeled_run["P04-contradiction-newest-wins"]["result"]
    text = " ".join(res.bodies.values()).lower()
    assert "vancouver" in text, "\n" + res.describe()
    if "toronto" in text:
        stale = [
            b for body in res.bodies.values() for b in bullets_of(body)
            if "toronto" in b.lower()
        ]
        assert all(
            any(w in b.lower() for w in
                ("moved", "used to", "previously", "until", "no longer",
                 "former", "unresolved", "discrepancy", "was "))
            for b in stale
        ), (
            "both cities are stored as live peers, with nothing to tell the "
            "model which is current:\n" + res.describe()
        )


def test_the_owner_never_gets_a_people_file(labeled_run):
    """Root cause #3, across every positive scenario.

    `people/nariman`, `people/nariman-hosseini` and `people/user` were three
    separate People files about the account owner, and nothing could merge
    them: the only self-check in the whole tree was `name == "user"`.
    """
    offenders = {
        s.id: labeled_run[s.id]["result"].forbidden_slugs
        for s in _POSITIVE
        if labeled_run[s.id]["result"].forbidden_slugs
    }
    assert not offenders, offenders


def test_a_second_person_gets_exactly_one_file(labeled_run):
    res = labeled_run["P06-a-second-person-gets-exactly-one-file"]["result"]
    assert not res.cardinality, "\n" + res.describe()


def test_farsi_is_stored_byte_exact(labeled_run):
    """Storage stays raw; bidi isolation is applied at RENDER by both
    clients. A writer that "cleans" Persian on the way in is unrecoverable."""
    res = labeled_run["P02-farsi-is-byte-exact"]["result"]
    hit = res.captured.get("P02")
    assert hit, "\n" + res.describe()
    slug, bullet = hit
    # The STEM, not "می‌دوم". That spelling is first person ("I run") and the
    # contract requires subjectless third person, which Persian carries in the
    # verb ending — the correct bullet says "می‌دود". This assertion demanded
    # the one form the house voice forbids, the same way the corpus marker
    # did; CI 32436100481 produced
    #   هر روز صبح ساعت ۷ می‌دود و بعدش صبحانه می‌خورد
    # and was failed for obeying the voice rule. The stem survives any
    # conjugation.
    assert "می‌دو" in bullet, bullet
    # Persian digits must not have been rewritten as ASCII on the way in.
    assert "۷" in bullet, bullet
    # No bidi control characters, no HTML entities, no mojibake.
    assert "‏" not in bullet and "‫" not in bullet, repr(bullet)
    assert "&#" not in bullet, bullet


def test_every_mutating_op_left_a_change_line(labeled_run):
    """§2.2: `change` is required on add/rewrite/remove/delete_file, and it
    is what the user reads in the Memory log. A write with no line is a
    change the user cannot see or reconstruct."""
    for sc in _POSITIVE:
        res = labeled_run[sc.id]["result"]
        if res.applied and any(b.strip() for b in res.bodies.values()):
            assert res.changes, f"{sc.id} wrote a body and no change line"
            assert all(c.strip() for c in res.changes), res.changes


def test_every_file_has_a_real_description(labeled_run):
    """§1.4: a file is born with a real description or the create op is
    rejected. `default_purpose` templating is deleted, so an empty
    description means something created a row outside the ops engine."""
    from app.memory_files import description_problem

    problems = []
    for sc in _POSITIVE:
        res = labeled_run[sc.id]["result"]
        for slug in res.slugs:
            if not res.bodies.get(slug, "").strip():
                continue  # an empty system file has nothing to describe yet
        problems += res.lint.description_problems
    assert not problems, problems
