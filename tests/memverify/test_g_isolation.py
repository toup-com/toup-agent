"""G. Two users' files never cross, and an unknown owner fails SOFT.

Isolation is scoping, and scoping in the file model is one predicate:
`memory_files.user_id`. That is easy to get right and catastrophic to get
wrong, so it is asserted end to end rather than by reading a WHERE clause.

The second half is subtler and is the reason this file also owns the
placeholder case. `user_identity.known` is False on a fresh tenant
("Agent Owner", "<hex>@agent.local"), and the user-not-in-people rule fails
SOFT there — treating "unknown" as "definitely not the owner" would drop
real facts on every new tenant, and treating a placeholder as a name
attaches every fact to a person file called "Agent Owner". Both directions
are wrong; the corpus runs with a real name and this file covers the other
state.
"""

from __future__ import annotations

from .pipeline import bodies_by_slug, drive_turn


async def test_two_users_never_see_each_others_files(db, user_a, user_b):
    await drive_turn(
        db, user_a,
        "I'm a marine biologist and I live in Lisbon with two greyhounds.",
        "Noted.",
    )
    await drive_turn(
        db, user_b,
        "I'm a tax accountant in Winnipeg and I'm allergic to cats.",
        "Noted.",
    )

    a = " ".join((await bodies_by_slug(db, user_a)).values()).lower()
    b = " ".join((await bodies_by_slug(db, user_b)).values()).lower()

    assert "lisbon" in a or "biolog" in a, a
    assert "winnipeg" in b or "accountant" in b, b
    for leaked in ("winnipeg", "accountant", "cats"):
        assert leaked not in a, f"user_b's {leaked!r} reached user_a: {a}"
    for leaked in ("lisbon", "greyhound", "biolog"):
        assert leaked not in b, f"user_a's {leaked!r} reached user_b: {b}"


async def test_a_placeholder_identity_still_captures_real_facts(db):
    """A fresh tenant boots with `users.name = "Agent Owner"`.

    `resolve_user_identity` reports `known=False`, and the writer is told
    "unknown — do not create a people/ file for anyone who might be the
    owner". It must still write the ordinary facts: a new user's first
    conversation is exactly when memory matters most.
    """
    from .conftest import make_user

    uid = await make_user(db, name="Agent Owner", email="deadbeef@agent.local")
    await drive_turn(
        db, uid,
        "I work as a structural engineer and I've been at Arup for six years.",
        "Noted.",
    )
    bodies = await bodies_by_slug(db, uid)
    joined = " ".join(bodies.values()).lower()
    assert "arup" in joined or "engineer" in joined, bodies
    assert not [s for s in bodies if s.startswith("people/")], (
        "an unknown owner produced a people/ file — the fail-soft branch "
        f"attached the owner's own facts to a stranger: {sorted(bodies)}"
    )


async def test_forget_everything_is_scoped_to_one_user(db, user_a, user_b):
    from app.services import memory_file_ops as ops

    await drive_turn(db, user_a, "I play the santur and I've had it 12 years.", "Noted.")
    await drive_turn(db, user_b, "I row competitively, six mornings a week.", "Noted.")

    removed = await ops.forget_everything(db, user_a)
    assert removed > 0

    assert not await bodies_by_slug(db, user_a)
    survivors = " ".join((await bodies_by_slug(db, user_b)).values()).lower()
    assert "row" in survivors or "morning" in survivors, survivors
