"""K. Privacy — sensitive data policy, and memory contents in logs.

Policy under test (documented in MEMORY_SYSTEM_MAP.md §2.1 and enforced by
app/services/memory_gate.py):

  NEVER stored verbatim: passwords, full payment card numbers, card CVVs,
  government identity numbers (SIN/SSN/passport), and API keys / bearer tokens.

  STILL stored: health details, medications, dietary needs, employer, location
  — these are durable facts about the user and are the entire point of the
  system. A privacy control that rejects them is a recall bug, not a fix.

K05 exists so that a change which passes this file by rejecting everything
sensitive-adjacent fails instead.
"""

from __future__ import annotations

import logging

import pytest

from .conftest import record_metric
from .corpus import SENSITIVE
from .pipeline import active_contents, drive_turn


@pytest.mark.parametrize("scenario", SENSITIVE, ids=lambda s: s.id)
def test_sensitive_policy(labeled_run, scenario):
    entry = labeled_run[scenario.id]
    assert entry["error"] is None, f"pipeline raised: {entry['error']}"
    res = entry["result"]
    assert not res.junk, f"sensitive value stored verbatim:\n{res.describe()}"
    assert not res.missed, (
        f"legitimate durable fact dropped:\n{res.describe()}\n"
        f"note: {scenario.note}"
    )


def test_sensitive_aggregate(labeled_run):
    stored = sum(len(labeled_run[s.id]["result"].junk) for s in SENSITIVE)
    record_metric("sensitive_values_stored", stored)
    assert stored == 0


async def test_memory_content_not_written_to_logs_verbatim(db, user_a, caplog):
    """Memory contents must not appear in plaintext in logs.

    This is the test that found BUG-25, and it is also the reason the bug
    survived four green runs: the capture path logged `content[:50]`, so
    whether this secret leaked depended on where the extractor's phrasing put
    it inside that window. It passed run after run and then failed one.

    Keep it — it is the only check that covers the whole real turn — but the
    deterministic guards at the bottom of this file are what actually hold the
    invariant. A test that needs the model to phrase things unluckily before
    it fails is a smoke alarm with a coin flip in it.
    """
    caplog.set_level(logging.INFO)
    secret = "wolverine-plinth-6620"
    await drive_turn(
        db, user_a,
        f"Please remember my locker phrase: {secret}",
        "Saved.",
    )
    stored = await active_contents(db, user_a)
    assert any(secret in c for c in stored), "explicit-save fact was not stored"

    leaked = [r.getMessage() for r in caplog.records if secret in r.getMessage()]
    record_metric("log_lines_containing_memory_content", len(leaked))
    assert not leaked, (
        f"{len(leaked)} log line(s) contain memory content verbatim:\n"
        + "\n".join(leaked[:5])
    )


# ── The storage backstop ─────────────────────────────────────────────────
#
# The full gate lives at the dedup boundary, but EIGHT callers reach
# MemoryService.create_memory directly and bypass it — including
# mcp_server.create_memory, which is model-controlled, and the REST endpoint.
# create_memory is the last function before the row exists, so the policy that
# must hold on every path is enforced there.
#
# Deliberately only the never-store tier. Applying the whole gate here would
# break document ingestion, which legitimately stores long RAG chunks.

NEVER_STORE_ON_ANY_PATH = [
    ("payment card", "The user's card number is 4539 8712 3344 9020."),
    ("CVV", "The user's card CVV is 412."),
    ("social insurance number", "The user's SIN is 046 454 286."),
    ("API key", "The user's API key is sk-proj-abcdefghijklmnopqrstuvwxyz012345."),
    ("AWS key", "The user's AWS key is AKIAIOSFODNN7EXAMPLE."),
]


@pytest.mark.parametrize(
    "label,content", NEVER_STORE_ON_ANY_PATH,
    ids=[c[0] for c in NEVER_STORE_ON_ANY_PATH],
)
async def test_create_memory_refuses_never_store_secrets(db, user_a, label, content):
    """Straight at the storage function, not through the dedup boundary —
    this is the path MCP and REST take."""
    from app.schemas import BrainType, MemoryCreate
    from app.services.memory_gate import MemoryRejected
    from app.services.memory_service import MemoryService

    svc = MemoryService(db)
    with pytest.raises(MemoryRejected):
        await svc.create_memory(
            user_a,
            MemoryCreate(
                content=content,
                brain_type=BrainType.USER,
                category="finance",
                memory_type="fact",
                importance=0.9,
                confidence=1.0,
            ),
        )

    assert not any(
        content[:20].lower() in c.lower() for c in await active_contents(db, user_a)
    )


async def test_the_backstop_does_not_break_document_ingestion(db, user_a):
    """The paired must-KEEP, and the reason the backstop is only one tier.

    A document chunk is long, is not a single fact, and would fail the length
    and scaffolding rules — applying the whole gate here would silently break
    RAG ingestion for every uploaded file."""
    from app.schemas import BrainType, MemoryCreate
    from app.services.memory_service import MemoryService

    chunk = (
        "Section 4.2 of the tenancy agreement. " + ("The lessee shall maintain "
        "the premises in good repair throughout the term of this agreement. " * 12)
    )
    assert len(chunk) > 600, "this test is meaningless unless the chunk is long"

    svc = MemoryService(db)
    memory = await svc.create_memory(
        user_a,
        MemoryCreate(
            content=chunk,
            brain_type=BrainType.USER,
            category="knowledge",
            memory_type="fact",
            importance=0.5,
            confidence=0.9,
        ),
    )
    assert memory is not None and memory.id


# ── BUG-25: memory content was logged verbatim ───────────────────────────
#
# The end-to-end test above is the one that FOUND this, and it is also the
# reason it took so long: it passed four consecutive runs and failed the
# fifth. Whether the secret leaked depended on where the extractor's phrasing
# put it relative to a 50-character truncation —
# "The user's locker phrase is wolverine-plinth-6620" is 49 characters, so
# that phrasing leaked the whole thing, and a slightly longer one did not.
#
# A test whose verdict depends on the model's word choice is not a guard.
# These put the marker at index 0, so ANY truncating log line contains it.
# All three fail on the pre-fix code every single time.

LEAK_MARKER = "zzqxmarker7731plinth"


def _leaked(caplog):
    return [r.getMessage() for r in caplog.records if LEAK_MARKER in r.getMessage()]


async def test_active_task_path_does_not_log_content(db, user_a, caplog):
    from app.services.active_task_service import store_active_task

    caplog.set_level(logging.INFO)
    # Marker first, so `content[:60]` cannot miss it.
    await store_active_task(
        db, user_a, f"{LEAK_MARKER} is the project the user is working on",
    )
    assert not _leaked(caplog), (
        "active-task write logged memory content verbatim:\n"
        + "\n".join(_leaked(caplog))
    )


async def test_reflection_path_does_not_log_content(db, user_a, caplog):
    from app.services.agent_reflection import store_agent_reflections

    caplog.set_level(logging.INFO)
    await store_agent_reflections(
        db, user_a,
        [{"content": f"{LEAK_MARKER} — always reply to this user in short sentences",
          "category": "behavior", "importance": 0.6, "confidence": 0.9}],
    )
    assert not _leaked(caplog), (
        "reflection write logged memory content verbatim:\n"
        + "\n".join(_leaked(caplog))
    )


async def test_dedup_candidate_line_does_not_log_content(db, user_a, caplog):
    """The 'Found candidate with similarity' line quoted the EXISTING row —
    so this one leaked a fact the user stated on an earlier turn, on a later
    turn's write."""
    from app.schemas import BrainType, MemoryCreate
    from app.services.memory_dedup_service import MemoryDedupService

    dedup = MemoryDedupService(db)
    first = MemoryCreate(
        content=f"{LEAK_MARKER} is the user's climbing gym in Toronto",
        brain_type=BrainType.USER, category="preferences",
        memory_type="fact", importance=0.6, confidence=0.9,
    )
    await dedup.smart_create_memory(new_memory=first, user_id=user_a)
    await db.commit()

    caplog.set_level(logging.INFO)
    second = MemoryCreate(
        content=f"{LEAK_MARKER} is where the user climbs in Toronto",
        brain_type=BrainType.USER, category="preferences",
        memory_type="fact", importance=0.6, confidence=0.9,
    )
    await dedup.smart_create_memory(new_memory=second, user_id=user_a)

    assert not _leaked(caplog), (
        "dedup logged an existing memory's content verbatim:\n"
        + "\n".join(_leaked(caplog))
    )


async def test_the_gate_does_not_log_the_content_it_refused(db, user_a, caplog):
    """The worst of the eleven, and one I wrote in this job.

    `[memory_gate] refused at write (%s): %s` printed the refused content —
    which is, by construction, the most sensitive text the system ever sees.
    The gate that exists to keep a card number out of the database was
    writing that card number to the application log."""
    from app.schemas import BrainType, MemoryCreate
    from app.services.memory_dedup_service import MemoryDedupService

    caplog.set_level(logging.INFO)
    dedup = MemoryDedupService(db)
    memory, action = await dedup.smart_create_memory(
        new_memory=MemoryCreate(
            content=f"{LEAK_MARKER} card number 4539 8712 3344 9020",
            brain_type=BrainType.USER, category="finance",
            memory_type="fact", importance=0.9, confidence=1.0,
        ),
        user_id=user_a,
    )
    assert memory is None, "this test is meaningless unless the gate refuses"
    assert not _leaked(caplog), (
        "the gate logged the content it refused:\n" + "\n".join(_leaked(caplog))
    )
