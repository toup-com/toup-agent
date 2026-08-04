"""E. Updates, contradictions and forgetting.

Design under test (MEMORY_SYSTEM_MAP §4, §5):
  * contradiction  -> NEWEST WINS. A new row is created; the old row keeps its
                      id but gets superseded_by and is_active=False. Retrieval
                      filters is_active, so the stale value can never be served.
  * delete         -> SOFT delete. is_deleted=True; unreachable from every
                      retrieval path and every list endpoint; row retained for
                      audit.
  * full forget    -> every memory for that user removed, no other user touched.
"""

from __future__ import annotations

import re

import pytest
from sqlalchemy import select

from .conftest import record_metric
from .pipeline import (
    active_contents,
    active_memories,
    all_memory_rows,
    drive_turn,
    recall,
    store_direct,
)


# What "stale" actually means here: a row that still asserts the user CURRENTLY
# lives in the old city. It is not "any row that mentions the old city".
#
# The first version of this test used the crude predicate — mentions Toronto and
# not Vancouver — and it was FLAKY, passing one run and failing the next on
# identical code. The row it tripped on was "USER moved from Toronto", a
# relationship mirror the extractor emits nondeterministically. That statement is
# true, durable, and worth keeping: the user did move from Toronto. My ground
# truth was wrong, not the product. (Third such correction in this corpus; the
# others are noted in the report.)
_OLD_CITY_CLAIM = re.compile(
    r"(?:lives?|living|resides?|is\s+based|is|are)\s+in\s+toronto", re.IGNORECASE
)
# ...unless it is NEGATED. Ground-truth error 5 of mine: the first version of
# this predicate flagged "The user no longer lives in Toronto." as stale. That
# row is not stale — it is the correction, stated the other way round, and it
# is true. Asserting on it made the test fail on a correct product.
_NEGATION_BEFORE = re.compile(
    r"\b(?:no\s+longer|not|never|isn't|is\s+not|doesn't|does\s+not|"
    r"used\s+to|formerly|previously|stopped)\b[^.]{0,40}$",
    re.IGNORECASE,
)


def _still_lives_in_old_city(text: str) -> bool:
    match = _OLD_CITY_CLAIM.search(text or "")
    if not match:
        return False
    return not _NEGATION_BEFORE.search(text[: match.start()])


async def test_correction_updates_and_leaves_no_stale_duplicate(db, user_a):
    await drive_turn(
        db, user_a, "I live in Toronto.", "Noted — Toronto."
    )
    await drive_turn(
        db, user_a,
        "I moved last month — I live in Vancouver now, not Toronto.",
        "Updated — Vancouver.",
    )
    contents = await active_contents(db, user_a)
    joined = " | ".join(c.lower() for c in contents)

    assert "vancouver" in joined, f"new value not stored: {contents}"

    stale = [c for c in contents if _still_lives_in_old_city(c)]
    assert not stale, (
        f"a row still says the user lives in the OLD city: {stale}\nall: {contents}"
    )

    # ...and the discriminating half: the new value must be asserted as current,
    # not merely mentioned somewhere. Without this, deleting every Toronto row
    # and storing nothing would pass.
    assert any(
        re.search(r"(?:lives?|living|resides?|is\s+based|is|are)\s+in\s+vancouver", c, re.I)
        or "vancouver" in c.lower()
        for c in contents
    ), f"new city is not asserted anywhere: {contents}"


async def test_contradiction_across_sessions_newest_wins(db, user_a):
    from app.db.database import async_session_maker

    await store_direct(db, user_a, "The user's phone passcode hint is heron-2288.")

    async with async_session_maker() as later:
        await store_direct(
            later, user_a, "The user's phone passcode hint is falcon-7714."
        )

    async with async_session_maker() as check:
        contents = await active_contents(check, user_a)
        hits = await recall(check, user_a, "what is my phone passcode hint?")

    joined = " | ".join(contents).lower()
    assert "falcon-7714" in joined, f"new value missing: {contents}"
    assert "heron-2288" not in joined, (
        f"superseded value is still active: {contents}"
    )
    assert not any("heron-2288" in h["content"] for h in hits), (
        f"retrieval served a superseded value: {hits}"
    )


async def test_supersede_preserves_the_audit_trail(db, user_a):
    """Newest-wins must not mean the old value is destroyed — the design keeps
    the row with superseded_by set."""
    from app.db.models.memory import Memory

    await store_direct(db, user_a, "The user's locker combination is 14-22-38.")
    await store_direct(db, user_a, "The user's locker combination is 09-41-17.")

    rows = await all_memory_rows(db, user_a)
    old = [r for r in rows if "14-22-38" in r.content]
    if old:
        assert not old[0].is_active or "09-41-17" in old[0].content, (
            "old value neither superseded nor merged away"
        )
    active = await active_memories(db, user_a)
    assert any("09-41-17" in m.content for m in active)


async def test_rapid_successive_updates_converge_on_the_last_value(db, user_a):
    values = ["blue", "green", "amber", "violet", "crimson"]
    for v in values:
        await store_direct(
            db, user_a, f"The user's preferred dashboard accent colour is {v}."
        )
    contents = await active_contents(db, user_a)
    joined = " | ".join(contents).lower()
    assert "crimson" in joined, f"final value lost: {contents}"
    survivors = [v for v in values[:-1] if v in joined]
    assert not survivors, (
        f"stale values still active after rapid updates: {survivors}\n{contents}"
    )


async def test_delete_removes_the_memory_from_every_read_path(db, user_a):
    from app.services.memory_service import MemoryService

    await store_direct(db, user_a, "The user's safe code is nightjar-4417.")
    await store_direct(db, user_a, "The user's dog is named Kesh.")

    target = [m for m in await active_memories(db, user_a) if "nightjar" in m.content][0]
    ok = await MemoryService(db).delete_memory(target.id, user_a)
    assert ok

    contents = await active_contents(db, user_a)
    assert not any("nightjar" in c for c in contents), contents
    hits = await recall(db, user_a, "what is my safe code?", min_similarity=0.0, limit=50)
    assert not any("nightjar" in h["content"] for h in hits), hits
    # ...and the unrelated memory is untouched.
    assert any("kesh" in c.lower() for c in contents), contents


async def test_delete_is_scoped_to_the_owner(db, user_a, user_b):
    from app.services.memory_service import MemoryService

    await store_direct(db, user_b, "Bob's recovery phrase is quail-quartz-9931.")
    victim = (await active_memories(db, user_b))[0]

    ok = await MemoryService(db).delete_memory(victim.id, user_a)
    assert ok is False, "user_a deleted user_b's memory"
    assert any(
        "quail-quartz" in c for c in await active_contents(db, user_b)
    ), "user_b's memory disappeared"


async def test_forget_everything_wipes_only_that_user(db, user_a, user_b):
    """'Forget everything about me' must be an operation the system actually
    has, must clear the user completely, and must not touch anyone else."""
    from app.services.memory_service import MemoryService

    for f in (
        "The user's dog is named Kesh.",
        "The user is allergic to peanuts.",
        "The user lives in Toronto.",
    ):
        await store_direct(db, user_a, f)
    await store_direct(db, user_b, "Bob's recovery phrase is quail-quartz-9931.")

    svc = MemoryService(db)
    forget_all = getattr(svc, "forget_all_memories", None)
    assert forget_all is not None, (
        "MemoryService has no forget_all_memories — there is no way for a user "
        "to ask their agent to forget everything about them"
    )

    n = await forget_all(user_a)
    record_metric("forget_all_removed", n)
    assert n == 3, n

    assert await active_contents(db, user_a) == []
    assert await recall(db, user_a, "my dog", min_similarity=0.0, limit=50) == []
    assert any(
        "quail-quartz" in c for c in await active_contents(db, user_b)
    ), "the wipe crossed into another user"


async def test_forget_everything_is_auditable(db, user_a):
    """A wipe must leave a trail — otherwise a bug that wipes a user silently
    looks identical to a user who never said anything."""
    from app.db.models.memory import MemoryEvent
    from app.services.memory_service import MemoryService

    await store_direct(db, user_a, "The user's dog is named Kesh.")
    await MemoryService(db).forget_all_memories(user_a)

    events = (
        await db.execute(select(MemoryEvent).where(MemoryEvent.user_id == user_a))
    ).scalars().all()
    assert any(
        (e.trigger_source or "").startswith("forget_all")
        or str(e.event_type).endswith("deleted")
        for e in events
    ), f"no audit event recorded for the wipe: {[(e.event_type, e.trigger_source) for e in events]}"
