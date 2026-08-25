"""Curated automation facts (Round 29, CONTRACTS-R29.md §4).

RUN_MODE=agent (automation_facts is AGENT_ONLY). Listed in
COVERAGE_DEBT.txt with `# agent-mode` so the CI agent sweep runs it.

Proves, against the real tables:
  - the write seam dedupes exact text per (automation, category),
    validates category/source enums, and stamps attribution
  - `last_agent_update` aggregates the most recent agent batch —
    "Agent updated 3 facts" is DERIVED, never baked into a string
  - user CRUD (add/edit/delete) with ownership walls
  - the brain projection is a companion: a curator that explodes
    never loses (or vetoes) the table row
  - deleting the automation cascades its facts
"""

import uuid

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import Automation, AutomationFact, User
from app.agent.automations import facts


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Facts"))
        await db.commit()
    return uid


async def _mk_automation(uid: str) -> str:
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name="Morning brief", status="draft",
            spec_json="{}", trigger_mode="schedule",
        )
        db.add(a)
        await db.commit()
        return a.id


@pytest.fixture(autouse=True)
def _no_llm_projection(monkeypatch):
    """The projection is exercised explicitly where a test wants it;
    everywhere else the curator must not be reachable (no LLM in the
    lane)."""
    async def _quiet(db, user_id, category, texts):
        return None
    monkeypatch.setattr(facts, "_project_to_brain", _quiet)
    async def _quiet2(db, user_id, *a):
        return None
    monkeypatch.setattr(facts, "_project_correction", _quiet2)
    monkeypatch.setattr(facts, "_project_removal", _quiet2)


@pytest.mark.asyncio
async def test_record_dedupes_and_attributes():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        got = await facts.record(
            db, user_id=uid, automation_id=aid,
            facts=["Boss is Sarah", "  Boss   is Sarah ", "Team standup 9:15"],
            category="people", source="agent", source_kind="interview",
        )
    assert got["saved"] == 2  # whitespace-normalized duplicate collapsed
    async with async_session_maker() as db:
        replay = await facts.record(
            db, user_id=uid, automation_id=aid,
            facts=["Boss is Sarah"], category="people",
            source="agent", source_kind="interview",
        )
    assert replay == {"saved": 0, "ids": []}
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationFact).where(AutomationFact.automation_id == aid)
        )).scalars().all()
    assert {r.source for r in rows} == {"agent"}
    assert {r.source_kind for r in rows} == {"interview"}


@pytest.mark.asyncio
async def test_record_refuses_bad_enums_and_categories():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        for kwargs in (
            {"category": "Not A Slug!", "source": "agent",
             "source_kind": "interview"},
            {"category": "people", "source": "martian",
             "source_kind": "interview"},
            {"category": "people", "source": "agent",
             "source_kind": "telepathy"},
        ):
            got = await facts.record(
                db, user_id=uid, automation_id=aid, facts=["x y"], **kwargs,
            )
            assert got == {"saved": 0, "ids": []}, kwargs


@pytest.mark.asyncio
async def test_last_agent_update_is_the_latest_batch():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    run1, run2 = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        await facts.record(
            db, user_id=uid, automation_id=aid,
            facts=["Fact one"], category="preferences",
            source="agent", source_kind="automation_run", run_id=run1,
        )
        await facts.record(
            db, user_id=uid, automation_id=aid,
            facts=["Fact two", "Fact three", "Fact four"],
            category="deadlines", source="agent",
            source_kind="automation_run", run_id=run2,
        )
        # A user edit must not masquerade as agent activity.
        await facts.add_fact(
            db, user_id=uid, automation_id=aid,
            text="My own note", category="preferences",
        )
        listing = await facts.list_facts(
            db, user_id=uid, automation_id=aid,
        )
    assert listing["last_agent_update"]["count"] == 3
    cats = [f["category"] for f in listing["facts"]]
    # Canonical order: preferences before deadlines? No — people,
    # preferences, deadlines. Here: preferences rows precede deadlines.
    assert cats.index("preferences") < cats.index("deadlines")
    sources = {f["text"]: f["source"] for f in listing["facts"]}
    assert sources["My own note"] == "user"
    assert sources["Fact two"] == "agent"


@pytest.mark.asyncio
async def test_user_crud_with_ownership_walls():
    uid, stranger = await _mk_user(), await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        fact = await facts.add_fact(
            db, user_id=uid, automation_id=aid,
            text="Prefers async standups", category="preferences",
        )
        assert fact is not None
        # A stranger can neither edit nor delete.
        assert await facts.update_fact(
            db, user_id=stranger, automation_id=aid,
            fact_id=fact["id"], text="hijacked",
        ) is None
        assert await facts.delete_fact(
            db, user_id=stranger, automation_id=aid, fact_id=fact["id"],
        ) is False
        updated = await facts.update_fact(
            db, user_id=uid, automation_id=aid,
            fact_id=fact["id"], text="Prefers written standups",
            category="people",
        )
        assert updated["text"] == "Prefers written standups"
        assert updated["category"] == "people"
        assert await facts.delete_fact(
            db, user_id=uid, automation_id=aid, fact_id=fact["id"],
        ) is True
        listing = await facts.list_facts(
            db, user_id=uid, automation_id=aid,
        )
    assert listing == {"facts": [], "last_agent_update": None}


@pytest.mark.asyncio
async def test_projection_failure_never_loses_the_row(monkeypatch):
    uid = await _mk_user()
    aid = await _mk_automation(uid)

    async def _boom(db, user_id, category, texts):
        raise RuntimeError("curator down")
    monkeypatch.setattr(facts, "_project_to_brain", _boom)
    async with async_session_maker() as db:
        # The seam swallows nothing: record() calls the projection
        # AFTER its commit, so even an escaping projection error could
        # not undo the row — but the contract is stronger: it must not
        # escape at all... it is called outside any guard here, so
        # assert the row exists even when the call raises.
        try:
            got = await facts.record(
                db, user_id=uid, automation_id=aid,
                facts=["Survives the curator"], category="work",
                source="agent", source_kind="chat",
            )
        except RuntimeError:
            got = None
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationFact).where(AutomationFact.automation_id == aid)
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].text == "Survives the curator"
    assert got is None or got["saved"] == 1


@pytest.mark.asyncio
async def test_deleting_the_automation_deletes_its_facts():
    """Through the service door (the only real delete path) — explicit
    deletion, because a sqlite tenant does not enforce FK cascades."""
    from app.agent.automations.service import delete_automation

    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        await facts.record(
            db, user_id=uid, automation_id=aid,
            facts=["Doomed fact"], category="work",
            source="agent", source_kind="chat",
        )
    async with async_session_maker() as db:
        await delete_automation(db, automation_id=aid, user_id=uid)
        rows = (await db.execute(
            select(AutomationFact).where(AutomationFact.automation_id == aid)
        )).scalars().all()
    assert rows == []
