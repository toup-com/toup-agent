# agent-mode
"""Memory v2 service (Round 30, CONTRACTS-R30.md §4.5).

RUN_MODE=agent (memory_facts / memory_episodes / memory_entities /
memory_forgets / automation_facts / build_jobs are all AGENT_ONLY).
Listed in COVERAGE_DEBT.txt with `# agent-mode` so the CI agent sweep
runs it.

Proves, against the real tables:
  - dedupe across scopes: a globally-held fact is CONFIRMED, never
    echoed into an automation scope
  - the §3.10 sheet payload always carries all five categories
  - forget → suppression: the re-add inside the 30-day window is
    refused, whoever asks
  - recall by entity / query / scope widening (own scope + global,
    never a sibling's)
  - the §4.5 migration drops definition-facts (D-20) and run-status
    leakage (ND-2) — using the two live examples GROUND-TRUTH found —
    and is idempotent
  - the episode back-fill writes one episode per pre-v3 terminal run,
    and never doubles a run the ledger already covered
  - a brain projection that explodes never loses the row
"""

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationFact, BuildJob, MemoryEntity, MemoryEpisode,
    MemoryFact, MemoryForget, User, MEMORY_V2_CATEGORIES,
)
from app.services import memory_v2_service as mv2


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Memory"))
        await db.commit()
    return uid


async def _mk_automation(uid: str, name: str = "Morning work brief",
                         domain: str | None = "work") -> str:
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name=name, status="draft",
            spec_json="{}", trigger_mode="schedule", domain=domain,
        )
        db.add(a)
        await db.commit()
        return a.id


@pytest.fixture(autouse=True)
def _no_llm_projection(monkeypatch):
    """The projection seam is exercised explicitly where a test wants
    it; everywhere else the curator must not be reachable (no LLM in
    the lane)."""
    async def _quiet(db, user_id, **kw):
        return None
    async def _quiet2(db, user_id, text):
        return None
    monkeypatch.setattr(mv2, "_project_fact", _quiet)
    monkeypatch.setattr(mv2, "_project_removal", _quiet2)


# ── add_fact: dedupe across scopes ───────────────────────────────────


@pytest.mark.asyncio
async def test_a_global_fact_is_never_duplicated_into_an_automation_scope():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        first = await mv2.add_fact(
            db, user_id=uid, text="Boss is Sarah Chen",
            category="people", scope="global",
        )
        assert first["scope"] == "global"
        assert first["last_confirmed_at"] is None
        # Same belief, learned again inside an automation: the GLOBAL
        # row is confirmed; no scoped echo is minted.
        echo = await mv2.add_fact(
            db, user_id=uid, text="boss  is Sarah Chen.",
            category="people", scope=aid,
        )
    assert echo["id"] == first["id"]
    assert echo["scope"] == "global"
    assert echo["last_confirmed_at"] is not None
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].scope == "global"


@pytest.mark.asyncio
async def test_a_scoped_re_add_confirms_instead_of_duplicating():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        first = await mv2.add_fact(
            db, user_id=uid, text="Standup moved to 9:15",
            category="your_time", scope=aid,
        )
        again = await mv2.add_fact(
            db, user_id=uid, text="Standup moved to 9:15",
            category="your_time", scope=aid,
        )
    assert again["id"] == first["id"]
    assert again["last_confirmed_at"] is not None
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_unknown_category_maps_to_work_you_own_not_an_error():
    """Total functions: old data must save, not raise."""
    uid = await _mk_user()
    async with async_session_maker() as db:
        got = await mv2.add_fact(
            db, user_id=uid, text="Ships on Fridays",
            category="not-a-category",
        )
    assert got["category"] == "work_you_own"


# ── The §3.10 sheet payload ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_sheet_always_serves_all_five_categories():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        await mv2.add_fact(
            db, user_id=uid, text="Marcus escalates on Mondays",
            category="people", scope=aid,
            subject_entity={"kind": "person", "name": "Marcus Reid"},
        )
        await mv2.add_fact(
            db, user_id=uid, text="Skips vendor newsletters",
            category="noise_filters", scope=aid,
        )
        sheet = await mv2.list_facts_for_scope(db, user_id=uid, scope=aid)
    assert sheet["count"] == 2
    keys = [c["key"] for c in sheet["categories"]]
    assert keys == list(MEMORY_V2_CATEGORIES)
    by_key = {c["key"]: c for c in sheet["categories"]}
    assert len(by_key["people"]["items"]) == 1
    assert len(by_key["noise_filters"]["items"]) == 1
    # Empty categories are present with empty items, never absent.
    assert by_key["your_time"]["items"] == []
    assert by_key["team_workspace"]["items"] == []
    assert by_key["work_you_own"]["items"] == []
    for c in sheet["categories"]:
        assert c["label"] and c["tone"]


# ── Forget → suppression ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_suppresses_the_re_add_for_thirty_days():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        fact = await mv2.add_fact(
            db, user_id=uid, text="Prefers async standups",
            category="team_workspace", scope="global",
        )
        assert await mv2.forget_fact(
            db, user_id=uid, fact_id=fact["id"],
        ) is True
    async with async_session_maker() as db:
        assert (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars().all() == []
        signals = (await db.execute(
            select(MemoryForget).where(MemoryForget.user_id == uid)
        )).scalars().all()
        assert len(signals) == 1
        assert signals[0].until > datetime.utcnow() + timedelta(days=29)
        # The re-add inside the window is refused — in ANY scope, with
        # any normalization of the same text.
        refused = await mv2.add_fact(
            db, user_id=uid, text="prefers async  standups.",
            category="team_workspace", scope=aid,
        )
        assert refused == {"suppressed": True}
        rows = (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_forget_walls_off_strangers():
    uid, stranger = await _mk_user(), await _mk_user()
    async with async_session_maker() as db:
        fact = await mv2.add_fact(
            db, user_id=uid, text="Owns the deploy checklist",
            category="work_you_own",
        )
        assert await mv2.forget_fact(
            db, user_id=stranger, fact_id=fact["id"],
        ) is False
        assert await db.get(MemoryFact, fact["id"]) is not None


# ── Recall ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_by_entity_query_and_scope_widening():
    uid = await _mk_user()
    aid = await _mk_automation(uid, name="Jira watcher")
    bid = await _mk_automation(uid, name="Mail sorter")
    async with async_session_maker() as db:
        await mv2.add_fact(
            db, user_id=uid, text="Marcus Reid escalates on Mondays",
            category="people", scope="global",
            subject_entity={"kind": "person", "name": "Marcus Reid"},
        )
        await mv2.add_fact(
            db, user_id=uid, text="SCRUM board triage happens at 9",
            category="your_time", scope=aid,
        )
        await mv2.add_fact(
            db, user_id=uid, text="Newsletters are noise",
            category="noise_filters", scope=bid,
        )
        db.add(MemoryEpisode(
            user_id=uid, automation_id=aid, run_id=str(uuid.uuid4()),
            text="Jira watcher — ran", outcome="completed",
        ))
        await db.commit()

    async with async_session_maker() as db:
        # Entity: case-insensitive substring over the entity NAME.
        got = await mv2.recall(db, user_id=uid, entity="marcus")
        assert [f["text"] for f in got["facts"]] == \
            ["Marcus Reid escalates on Mondays"]
        # Query: ILIKE over fact text.
        got = await mv2.recall(db, user_id=uid, query="triage")
        assert [f["text"] for f in got["facts"]] == \
            ["SCRUM board triage happens at 9"]
        # Scope widening: automation A sees its own scope + global,
        # never automation B's.
        got = await mv2.recall(db, user_id=uid, scope=aid)
        texts = {f["text"] for f in got["facts"]}
        assert "SCRUM board triage happens at 9" in texts
        assert "Marcus Reid escalates on Mondays" in texts
        assert "Newsletters are noise" not in texts
        # ...and the run episode rides along with its links.
        assert len(got["episodes"]) == 1
        assert got["episodes"][0]["automation_id"] == aid
        assert got["episodes"][0]["outcome"] == "completed"
        # An entity nobody knows recalls nothing, honestly.
        got = await mv2.recall(db, user_id=uid, entity="zebra")
        assert got == {"facts": [], "episodes": []}


# ── The §4.5 migration ───────────────────────────────────────────────


async def _seed_r29_facts(uid: str, aid: str) -> None:
    """The R29 ledger as GROUND-TRUTH found it: real beliefs mixed
    with a definition-fact and a run-status sentence (the two live
    examples), across the old category vocabulary."""
    async with async_session_maker() as db:
        for text, category, source in (
            # D-20 definition-fact — the live example, verbatim shape.
            ("Has an automation 'Morning work brief': Every day at "
             "22:52, check Jira for updates and summarize them.",
             "work", "agent"),
            # ND-2 run-status leakage — the live example.
            ("The Morning work brief is currently paused.",
             "work", "agent"),
            ("Boss is Sarah Chen", "people", "agent"),
            ("Prefers meetings before noon", "preferences", "agent"),
            ("Prefers dark roast coffee", "preferences", "user"),
            ("Q3 report due September 12", "deadlines", "agent"),
            ("Owns the vendor renewal", "work", "user"),
        ):
            db.add(AutomationFact(
                user_id=uid, automation_id=aid, category=category,
                text=text, source=source,
                source_kind="edit" if source == "user" else "automation_run",
            ))
        await db.commit()


@pytest.mark.asyncio
async def test_migration_drops_definition_and_status_facts_and_maps_the_rest():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    await _seed_r29_facts(uid, aid)
    async with async_session_maker() as db:
        counts = await mv2.migrate_user(db, user_id=uid)
    assert counts == {"migrated": 5, "dropped_definition": 1,
                      "dropped_status": 1, "skipped_existing": 0}
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars().all()
        entities = (await db.execute(
            select(MemoryEntity).where(MemoryEntity.user_id == uid)
        )).scalars().all()
    by_text = {r.text: r for r in rows}
    # Nothing that describes the tool or its run state survived.
    assert not any("Has an automation" in t for t in by_text)
    assert not any("currently paused" in t for t in by_text)
    # Category mapping: people→people; time-flavoured preferences→
    # your_time; taste preferences→work_you_own; deadlines→your_time;
    # everything else→work_you_own.
    assert by_text["Boss is Sarah Chen"].category == "people"
    assert by_text["Prefers meetings before noon"].category == "your_time"
    assert by_text["Prefers dark roast coffee"].category == "work_you_own"
    assert by_text["Q3 report due September 12"].category == "your_time"
    assert by_text["Owns the vendor renewal"].category == "work_you_own"
    # Scope, provenance, source mapping.
    assert all(r.scope == aid for r in rows)
    assert all(r.why == "Saved from an earlier version" for r in rows)
    assert by_text["Owns the vendor renewal"].source == "told"
    assert by_text["Boss is Sarah Chen"].source == "agent"
    # Best-effort person entity from the First-Last shape.
    names = {e.name for e in entities}
    assert "Sarah Chen" in names
    assert by_text["Boss is Sarah Chen"].subject_entity_id is not None


@pytest.mark.asyncio
async def test_migration_is_idempotent():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    await _seed_r29_facts(uid, aid)
    async with async_session_maker() as db:
        first = await mv2.migrate_user(db, user_id=uid)
    async with async_session_maker() as db:
        second = await mv2.migrate_user(db, user_id=uid)
    assert first["migrated"] == 5
    assert second["migrated"] == 0
    assert second["skipped_existing"] == 5
    # The drops are re-counted (they are still dropped), never migrated.
    assert second["dropped_definition"] == 1
    assert second["dropped_status"] == 1
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars().all()
    assert len(rows) == 5


# ── Episode back-fill ────────────────────────────────────────────────


async def _mk_run(uid: str, aid: str, *, completed: bool = True,
                  user_message: str | None = None,
                  outcome: str | None = None,
                  status: str = "completed") -> str:
    async with async_session_maker() as db:
        job = BuildJob(
            user_id=uid, title="Morning work brief",
            prompt="run", job_type="automation_run", status=status,
            source_kind="automation", source_id=aid,
            user_message=user_message, outcome=outcome,
            completed_at=datetime.utcnow() if completed else None,
        )
        db.add(job)
        await db.commit()
        return job.id


@pytest.mark.asyncio
async def test_backfill_writes_one_episode_per_terminal_run_and_is_idempotent():
    uid = await _mk_user()
    aid = await _mk_automation(uid)
    done = await _mk_run(uid, aid, user_message="Checked Jira — 3 updates")
    failed = await _mk_run(uid, aid, outcome="partial", status="completed")
    running = await _mk_run(uid, aid, completed=False, status="running")
    async with async_session_maker() as db:
        wrote = await mv2.backfill_episodes(db, user_id=uid)
    assert wrote == 2
    async with async_session_maker() as db:
        eps = (await db.execute(
            select(MemoryEpisode).where(MemoryEpisode.user_id == uid)
        )).scalars().all()
    by_run = {e.run_id: e for e in eps}
    assert set(by_run) == {done, failed}
    assert running not in by_run
    # Text = automation name + em-dash + the humanized outcome, with
    # user_message winning over the raw outcome/status vocabulary.
    assert by_run[done].text == "Morning work brief — Checked Jira — 3 updates"
    assert by_run[failed].text == "Morning work brief — partial"
    assert by_run[failed].outcome == "partial"
    assert by_run[done].automation_id == aid
    # A second pass writes nothing — runs already covered stay covered.
    async with async_session_maker() as db:
        assert await mv2.backfill_episodes(db, user_id=uid) == 0
        eps = (await db.execute(
            select(MemoryEpisode).where(MemoryEpisode.user_id == uid)
        )).scalars().all()
    assert len(eps) == 2


# ── Projection is a companion, never a veto ──────────────────────────


@pytest.mark.asyncio
async def test_projection_failure_never_loses_the_row(monkeypatch):
    uid = await _mk_user()

    async def _boom(db, user_id, **kw):
        raise RuntimeError("curator down")
    monkeypatch.setattr(mv2, "_project_fact", _boom)
    async with async_session_maker() as db:
        # The real projection swallows internally; this patched raiser
        # bypasses that guard, so even an ESCAPING error must find the
        # row already committed.
        try:
            await mv2.add_fact(
                db, user_id=uid, text="Survives the curator",
                category="work_you_own",
            )
        except RuntimeError:
            pass
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].text == "Survives the curator"
