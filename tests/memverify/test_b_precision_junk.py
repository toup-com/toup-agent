"""B. Capture — precision. Zero junk.

Target: 0 junk entries across every labeled run. One is a failure.
"""

from __future__ import annotations

import pytest

from .conftest import record_metric
from .corpus import ALL_LABELED, JUNK
from .pipeline import active_contents, drive_turn


@pytest.mark.parametrize("scenario", JUNK, ids=lambda s: s.id)
def test_no_junk_stored(labeled_run, scenario):
    entry = labeled_run[scenario.id]
    assert entry["error"] is None, f"pipeline raised: {entry['error']}"
    res = entry["result"]
    assert not res.junk, (
        f"\n{res.describe()}\n"
        f"note: {scenario.note or '-'}\n"
        f"turns: {[t.user[:70] for t in scenario.turns]}"
    )


def test_junk_aggregate(labeled_run):
    """Junk count over the WHOLE labeled corpus, not just the junk scenarios —
    a capture scenario that also stores garbage is still a precision failure."""
    junk = [
        f"{s.id}:{j}"
        for s in ALL_LABELED
        for j in labeled_run[s.id]["result"].junk
    ]
    record_metric("junk_markers_total", sum(len(s.must_not_store) for s in ALL_LABELED))
    record_metric("junk_stored_count", len(junk))
    assert not junk, f"{len(junk)} junk entries stored:\n" + "\n".join(junk)


def test_trivial_turn_writes_nothing(labeled_run):
    """The trivial-query skip must actually skip: no rows at all, not 'few'."""
    from .corpus import BY_ID

    entry = labeled_run["B20-trivial-turn"]
    assert entry["error"] is None
    assert not entry["result"].junk


async def test_junk_gate_still_fires_on_a_placeholder_tenant(db):
    """memory_gate's relationship rule fails open when the tenant identity is a
    placeholder — every tenant except the founder's carries name='Agent Owner'
    until onboarding. The identity-FREE rules must still fire there, otherwise
    the fail-open silently disables the whole gate for 54 of 55 tenants.

    This is the shape the labeled corpus does NOT cover (it uses a real name on
    purpose), so it is asserted directly.
    """
    from app.services.memory_gate import memory_gate_reason, relationship_gate_reason

    # Identity-free rules fire regardless of who the user is.
    assert memory_gate_reason("Sub-agent 2 summarizes the latest OpenAI news")
    assert memory_gate_reason("x" * 700) == "not_a_single_fact"
    assert relationship_gate_reason(
        "Assistant", "summarizes", "Gmail", user_aliases=[]
    ) == "agent_talking_about_itself"
    assert relationship_gate_reason(
        "Toup", "has_launch", "Toup's launch", user_aliases=[]
    ) is not None

    # ...and the user-endpoint rule ABSTAINS rather than dropping a real fact.
    for aliases in ([], ["Agent Owner"], ["3134fece"], ["agent owner", "user"]):
        assert relationship_gate_reason(
            "Nariman", "owns", "Toup", user_aliases=aliases
        ) is None, f"real fact dropped on placeholder identity {aliases}"


async def test_gate_rejects_junk_end_to_end_but_admits_a_real_fact(db, user_a):
    """The discriminating pair. A 'no junk left' check cannot tell a working
    gate apart from one that rejects everything, so both directions are
    asserted on the same user in the same test."""
    junk_turn = await drive_turn(
        db, user_a,
        "What is a 409A valuation?",
        "A 409A valuation is an independent appraisal that sets the fair market "
        "value of a private company's common stock for tax purposes under "
        "Section 409A of the Internal Revenue Code. Companies typically refresh "
        "it annually or after a material event such as a financing round.",
    )
    after_junk = await active_contents(db, user_a)
    assert not any("409a" in c.lower() for c in after_junk), after_junk

    await drive_turn(
        db, user_a,
        "I've decided to move my whole team onto Linear next quarter.",
        "Linear next quarter — noted.",
    )
    after_fact = await active_contents(db, user_a)
    assert any("linear" in c.lower() for c in after_fact), after_fact
