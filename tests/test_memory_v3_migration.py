"""The round-8 → v3 migration (rebuild-2026-08-v3 §7).

WHAT IS AND IS NOT PROVEN HERE. The writer is faked — a scripted mini-model
that reads the prompt it was handed and replays a fixed table of verdicts.
So every assertion below is about MECHANISM: what the writer is allowed to
see, what the deterministic validator refuses to let it do, what reaches the
database, what the report says, and what a second/interrupted/dry run does.
The QUALITY of the routing (would a real model actually drop the sleep log?)
is NOT proven here; that is the eval set's job and it needs a live key.

Three of the assertions are stronger than "the script said so", and they are
the ones that matter most, because they hold whatever a model proposes:

* a row the USER deleted is never even shown to the writer (asserted against
  the recorded prompt, not against the output);
* an op that would file the owner's facts in `people/*` is refused by the
  validator, and the report says so in words;
* a bullet carrying a UUID or a tool parameter is refused by the lint.

The seed corpus is the founder's REAL production memory, captured from the
live tenant on 2026-08-20, plus one row per failure class named in the WS-5
spec. The four real entries are quoted verbatim: they are the reason the
rebuild exists.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.base import Base
from app.db.models.day_chat import MigrationStatus
from app.db.models.memory import Memory, MemoryEvent, MemoryFile, MemoryFileChange
from app.db.models.user import User
from app.memory_files import PROFILE_SLUG, parse_bullets
from app.services import memory_curator as curator
from app.services import memory_file_ops as ops
from app.services import memory_v3_migration as mig
from app.services.user_identity import forget_cached_identity


# ── The founder's real before-state (production, 2026-08-20) ─────────

FOUNDER_TELEGRAM = "User has an account on Telegram"
FOUNDER_JOJO = (
    "The user goes by the nickname 'jojo' (on Telegram or in personal "
    "contexts) and communicates with the AI assistant via WhatsApp. They "
    "also have Telegram available as a communication channel, with a "
    "Telegram token/account that they expected the AI assistant to have "
    "access to."
)
FOUNDER_MUSIC = (
    "You understand and communicate in Persian (Farsi) and enjoy a diverse "
    "range of music, including Drake and Persian music. You were listening "
    "to 'Gol-e Aftabgardoon' (Sunflower) by Arian Band, a classic Iranian "
    "pop group, and have discussed Iranian music in Farsi. You were also "
    "researching a tool called 'OpenClaw Moltbot' and its potential "
    "WhatsApp integration, but could not find solid information about it "
    "online."
)
FOUNDER_SLEEP = (
    "User logged their sleep at 1:41 AM on April 22, sending a message "
    "about it at 5:41 AM from mobile."
)

FORGOTTEN = "Home address is 12 Maple Street, apartment 4B"
SCRAPED_TITLE = "Arash Ft Helena - Broken Angel (OFFICIAL VIDEO) 4K"
JOB_PROMPT = (
    "Every morning at 11:49 AM run a Gmail briefing with max_results=1 and "
    "summarize the unread mail"
)
SNOOZE = "Snoozed the 7:00 AM alarm for 9 more minutes"
UUID_ROW = (
    "The active workspace app is 7f3c2b1a-9d4e-4c8f-a1b2-c3d4e5f60718"
)
QUOTE_ROUTINE = "Receives a quote every day at 5:06 PM"
IELTS_A = "The user is preparing for the IELTS exam on Aug 30, 2026"
IELTS_B = "User has an IELTS exam booked for August 30 2026"
OWN_NAME = "Nariman Hosseini is the user's name"
ANDROID = "Uses an Android phone"
DOC_CHUNK = "Section 4.2 of the lease agreement covers the security deposit"
CARD = "My card number is 4111 1111 1111 1111, expiry 03/29"
SUPERSEDED = "The user likes listening to Drake's music."


def _mem(**kw) -> Memory:
    base = dict(
        id=str(uuid.uuid4()),
        brain_type="user",
        category="knowledge",
        memory_type="fact",
        source_type="conversation",
        created_at=datetime(2026, 5, 1, 3, 0, 0),
        is_active=True,
        is_deleted=False,
    )
    base.update(kw)
    return Memory(**base)


async def _seed(db: AsyncSession, user_id: str) -> dict:
    """The founder's four active entries plus one row per failure class."""
    rows = {
        # ── the four real ones, verbatim ──
        "telegram": _mem(content=FOUNDER_TELEGRAM, category="people",
                         file_slug="people/user", source_type="entity_extraction"),
        "jojo": _mem(content=FOUNDER_JOJO, category="possessions",
                     file_slug="knowledge"),
        "music": _mem(content=FOUNDER_MUSIC, category="media",
                      memory_type="preference", file_slug="knowledge"),
        "sleep": _mem(content=FOUNDER_SLEEP, category="health",
                      memory_type="event", file_slug="profile"),
        # ── one per failure class from the spec ──
        "forgotten": _mem(content=FORGOTTEN, category="identity",
                          file_slug="profile", is_deleted=True, is_active=False,
                          deleted_at=datetime(2026, 6, 1)),
        "scraped": _mem(content=SCRAPED_TITLE, category="media",
                        file_slug="knowledge"),
        "job": _mem(content=JOB_PROMPT, category="other", file_slug="working"),
        "snooze": _mem(content=SNOOZE, category="other", file_slug="working"),
        "uuid": _mem(content=UUID_ROW, category="possessions",
                     file_slug="knowledge"),
        "routine": _mem(content=QUOTE_ROUTINE, category="other",
                        file_slug="working", ref_kind="routine",
                        ref_id="r-1"),
        "ielts_a": _mem(content=IELTS_A, category="goals",
                        file_slug="areas/work"),
        "ielts_b": _mem(content=IELTS_B, category="goals",
                        file_slug="areas/work"),
        "own_name": _mem(content=OWN_NAME, category="people",
                         file_slug="people/nariman-hosseini"),
        # is_active=False WITHOUT a delete: round 8 archived it. This is the
        # half the migration is supposed to recover.
        "android": _mem(content=ANDROID, category="possessions",
                        file_slug="knowledge", is_active=False),
        "doc": _mem(content=DOC_CHUNK, category="knowledge",
                    source_type="document"),
        "card": _mem(content=CARD, category="identity", file_slug="profile"),
    }
    survivor = rows["music"]
    rows["superseded"] = _mem(
        content=SUPERSEDED, category="media", file_slug="knowledge",
        is_active=False, superseded_by=survivor.id,
    )
    # Distinct, ordered timestamps so `plan_batches` (which sorts by legacy
    # file, then created_at, then id) is deterministic. Identical timestamps
    # fall through to a random uuid and batch membership stops being a fact
    # a test can rely on.
    for i, row in enumerate(rows.values()):
        row.user_id = user_id
        row.created_at = datetime(2026, 5, 1, 3, 0, 0) + timedelta(minutes=i)
        db.add(row)
    # The forget leaves an audit event, exactly as `delete_memory` does.
    db.add(MemoryEvent(
        id=str(uuid.uuid4()), memory_id=rows["forgotten"].id, user_id=user_id,
        event_type="deleted", trigger_source="api",
        timestamp=datetime(2026, 6, 1),
    ))
    await db.commit()
    return rows


async def _session(name="Nariman Hosseini"):
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=[
            User.__table__, Memory.__table__, MemoryEvent.__table__,
            MemoryFile.__table__, MemoryFileChange.__table__,
            MigrationStatus.__table__,
        ])
    db = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()
    user_id = str(uuid.uuid4())
    db.add(User(id=user_id, email="nariman@toup.ai", hashed_password="x", name=name))
    await db.commit()
    forget_cached_identity()
    return db, user_id


# ── The scripted writer ──────────────────────────────────────────────
#
# It reads the prompt the migration built, finds the entries it was shown,
# and replays a fixed verdict per entry. Deliberately literal: three of its
# rules propose ops the deterministic validator MUST refuse, because that
# refusal is the property under test.

_ENTRY_RE = re.compile(r"^(L\d+)\) \[[^\]]*\] (.*)$", re.MULTILINE)

_MUSIC_DESC = "Music taste — artists and styles they return to; read when music comes up."
_IELTS_DESC = "IELTS preparation — the exam date and how it is going; read when IELTS comes up."
_DEV_DESC = "Devices and apps — what they run day to day; read when a device comes up."
_PERSON_DESC = "About this person — who they are to them; read when their name comes up."

#: substring → (verdict, [(slug, bullet, section, description)], reason)
_SCRIPT = [
    (FOUNDER_TELEGRAM, "merged",
     [(PROFILE_SLUG, "uses Telegram and WhatsApp", None, None)],
     "folded into the owner's profile; it was filed as if the owner were "
     "somebody else"),
    ("nickname 'jojo'", "rewritten",
     [(PROFILE_SLUG, "goes by the nickname jojo", None, None)],
     "kept the nickname; dropped which app the messages arrive on"),
    ("Gol-e Aftabgardoon", "rewritten",
     [(PROFILE_SLUG, "speaks Persian (Farsi)", None, None),
      ("topics/music", "listens to Drake and to Persian pop", "topics", _MUSIC_DESC)],
     "kept the language and the taste; dropped the track that happened to "
     "be playing and the tool that was being researched"),
    ("logged their sleep", "dropped", [],
     "a single timestamped event, not a standing fact"),
    ("OFFICIAL VIDEO", "dropped", [],
     "a title a media tool returned about a one-off play"),
    # Proposes the job prompt verbatim — `max_results=1` must be refused.
    ("Gmail briefing", "kept",
     [(PROFILE_SLUG,
       "receives a daily Gmail briefing at 11:49 AM with max_results=1",
       None, None)],
     "a standing arrangement"),
    ("Snoozed the 7:00 AM", "dropped", [], "a snooze"),
    # Proposes the UUID verbatim — the lint must refuse it.
    ("workspace app is 7f3c2b1a", "kept",
     [("topics/devices",
       "runs the workspace app 7f3c2b1a-9d4e-4c8f-a1b2-c3d4e5f60718",
       "topics", _DEV_DESC)],
     "the app they work in"),
    (IELTS_A, "kept",
     [("areas/ielts", "IELTS exam booked for Aug 30, 2026", "areas", _IELTS_DESC)],
     "a commitment with a real date"),
    (IELTS_B, "merged", [], "the same exam as the entry above; one bullet, "
     "not two", "areas/ielts"),
    # Proposes a people/ file for the OWNER — the resolver must refuse it.
    ("is the user's name", "kept",
     [("people/nariman-hosseini", "goes by Nariman", "people", _PERSON_DESC)],
     "who this person is"),
    (ANDROID, "kept",
     [(PROFILE_SLUG, "uses an Android phone", None, None)],
     "a standing fact about their setup; the old system had archived it"),
]


#: What the writer proposes on the SECOND round, once the validator has told
#: it why the first shape was refused. A competent model does exactly this:
#: keeps the fact, drops the refused part — or says plainly that there was
#: nothing left underneath it. All three cases are represented, because a
#: retry that could only ever succeed would prove nothing about the third.
_RETRY_SCRIPT = [
    # The owner's own name belongs in Profile, never in a people/ file.
    ("is the user's name", "kept",
     [(PROFILE_SLUG, "goes by Nariman Hosseini", None, None)],
     "the owner's own name — it belongs in their profile, not in a file "
     "about somebody else"),
    # The standing arrangement WITHOUT the tool parameter. The durability
    # rules allow exactly this line.
    ("Gmail briefing", "kept",
     [(PROFILE_SLUG, "receives a daily Gmail briefing at 11:49 AM", None, None)],
     "a standing arrangement, without the job's parameters"),
    # Nothing durable survives removing the id, and saying so IS the answer.
    ("workspace app is 7f3c2b1a", "dropped", [],
     "the entry was only an internal id; with the id removed there is no "
     "fact about this person left"),
]


class _Resp:
    def __init__(self, content):
        self.content = content


class _ScriptedWriter:
    """Replays `_SCRIPT` against whatever entries the prompt actually holds."""

    #: Present only in the orphan-retry prompt. Used to tell the two rounds
    #: apart, both here and in the tests.
    RETRY_MARK = "THESE ENTRIES WERE NOT WRITTEN"

    def __init__(self, fail_on_first_round_batch: int | None = None):
        self.prompts: list[str] = []
        self.fail_on = fail_on_first_round_batch
        self.raised = 0

    @property
    def first_round_prompts(self) -> list[str]:
        return [p for p in self.prompts if self.RETRY_MARK not in p]

    @property
    def retry_prompts(self) -> list[str]:
        return [p for p in self.prompts if self.RETRY_MARK in p]

    async def complete_with_json(self, messages, model=None, temperature=None, **kw):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        # Keyed to the Nth FIRST-ROUND prompt, never to a raw call index:
        # the orphan retry adds calls, so a count-based injection quietly
        # migrated onto the retry and stopped testing the batch-failure
        # path it was aimed at.
        # TWICE, not once: `_complete_json_with_retry` absorbs a single
        # transient blip.
        if (
            self.fail_on is not None
            and self.RETRY_MARK not in prompt
            and len(self.first_round_prompts) >= self.fail_on
            and self.raised < 2
        ):
            self.raised += 1
            raise RuntimeError("injected provider failure")

        index_block = prompt.split("FILE INDEX:", 1)[1].split("FILE BODIES:", 1)[0]
        entries_block = prompt.split("THE OLD ENTRIES:", 1)[1]
        for tail in (self.RETRY_MARK, "\n\nWHAT IS"):
            entries_block = entries_block.split(tail, 1)[0]
        existing = set(re.findall(r"^- ([\w/-]+) —", index_block, re.MULTILINE))
        script = _RETRY_SCRIPT if self.RETRY_MARK in prompt else _SCRIPT

        ops_out: list[dict] = []
        dispositions: list[dict] = []
        created: set[str] = set()
        for handle, content in _ENTRY_RE.findall(entries_block):
            rule = next((r for r in script if r[0] in content), None)
            if rule is None:
                dispositions.append({
                    "ref": handle, "verdict": "dropped", "slug": None,
                    "reason": "nothing durable in it",
                })
                continue
            verdict, writes, reason = rule[1], rule[2], rule[3]
            slug = rule[4] if len(rule) > 4 else (writes[0][0] if writes else None)
            for target, bullet, section, description in writes:
                if target not in existing and target not in created:
                    created.add(target)
                    ops_out.append({
                        "op": "create_file", "section": section, "slug": target,
                        "title": target.split("/")[-1].replace("-", " ").title(),
                        "description": description,
                    })
                ops_out.append({
                    "op": "add", "slug": target, "bullet": bullet,
                    "change": f"Added {target}: {bullet[:40]}.",
                    "refs": [handle],
                })
            dispositions.append({
                "ref": handle, "verdict": verdict, "slug": slug, "reason": reason,
            })
        return _Resp(json.dumps({"ops": ops_out, "dispositions": dispositions}))


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(curator, "EXTRACTION_RETRY_BACKOFF_S", 0)

    def install(fail_on_first_round_batch=None):
        writer = _ScriptedWriter(fail_on_first_round_batch=fail_on_first_round_batch)
        monkeypatch.setattr(curator, "_llm", lambda api_key: writer)
        return writer
    return install


@pytest.fixture(autouse=True)
def _small_batches(monkeypatch):
    """Four rows per batch, so batching, resume and the cross-batch view of
    the file bodies are all exercised by the seed corpus."""
    monkeypatch.setattr(mig, "BATCH_MAX_ROWS", 4)


# ── Helpers ──────────────────────────────────────────────────────────

async def _bodies(db, user_id) -> dict:
    return {f.slug: (f.body_md or "") for f in await ops._all_files(db, user_id)}


async def _legacy_fingerprint(db, user_id) -> list:
    rows = (await db.execute(
        select(Memory).where(Memory.user_id == user_id).order_by(Memory.id)
    )).scalars().all()
    return [
        (r.id, r.content, r.is_active, r.is_deleted, r.file_slug,
         r.file_position, r.superseded_by, r.category)
        for r in rows
    ]


def _by_id(report, row) -> dict:
    return next(d for d in report["dispositions"] if d["id"] == row.id)


# ══ The acceptance criteria ══════════════════════════════════════════

async def test_the_founders_corpus_becomes_files(scripted):
    """The end-to-end shape, on the real before-state.

    MECHANISM, not model quality: the verdicts come from `_SCRIPT`. What is
    proven is that a corpus of round-8 rows leaves as file BODIES, that the
    acceptance criteria hold over those bodies, and that the parts the
    validator owns hold regardless of what the writer proposed.
    """
    db, user_id = await _session()
    rows = await _seed(db, user_id)
    scripted()

    report = await mig.migrate_user(db, user_id, dry_run=False)
    bodies = await _bodies(db, user_id)

    # The user is never a People file — the writer TRIED and was refused —
    # but their NAME still lands, in the right file. The refusal is about
    # the shape, never about the fact.
    assert not any(s.startswith("people/") for s in bodies), bodies.keys()
    own = _by_id(report, rows["own_name"])
    assert own["disposition"] == "kept"
    assert own["slug"] == PROFILE_SLUG
    assert "goes by Nariman Hosseini" in bodies[PROFILE_SLUG]

    # Zero facts from tool output or media titles.
    joined = "\n".join(bodies.values())
    assert "OFFICIAL VIDEO" not in joined
    assert "Gol-e Aftabgardoon" not in joined
    assert "Arian Band" not in joined
    assert "OpenClaw" not in joined

    # Zero UUIDs, zero tool parameters — the lint, not the script.
    assert "7f3c2b1a" not in joined
    assert "max_results" not in joined

    # The standing arrangement survives the refusal that removed its
    # parameter. `TURN_DURABILITY_RULES` allows exactly this one line, and
    # before the orphan retry the whole bullet went with the parameter.
    job = _by_id(report, rows["job"])
    assert job["disposition"] == "kept"
    assert job["slug"] == PROFILE_SLUG
    assert "receives a daily Gmail briefing at 11:49 AM" in bodies[PROFILE_SLUG]

    # And a row with nothing under the refused part is a CONSIDERED drop:
    # re-asked, and dropped for a stated reason rather than by silence.
    uuid_row = _by_id(report, rows["uuid"])
    assert uuid_row["disposition"] == "dropped"
    assert "no fact about this person left" in uuid_row["reason"]

    # The routine appears at most once, and never as its own text.
    assert joined.count("5:06 PM") == 0
    assert _by_id(report, rows["routine"])["disposition"] == mig.SKIP_SCHEDULER

    # The one-off event and the snooze are gone.
    assert "1:41 AM" not in joined
    assert "Snoozed" not in joined

    # What SHOULD survive, did.
    profile = bodies[PROFILE_SLUG]
    assert "goes by the nickname jojo" in profile
    assert "speaks Persian (Farsi)" in profile
    assert "uses Telegram and WhatsApp" in profile
    assert "uses an Android phone" in profile, "an archived row was not recovered"
    assert "listens to Drake and to Persian pop" in bodies["topics/music"]
    assert parse_bullets(bodies["areas/ielts"]) == [
        "IELTS exam booked for Aug 30, 2026"
    ], "the two IELTS entries did not become one bullet"


async def test_a_row_the_user_deleted_is_never_shown_to_the_writer(scripted):
    """The worst possible outcome of this migration, guarded at the source.

    Asserted against the PROMPTS rather than the output: a rule that only
    checks the resulting bodies passes for a writer that was shown the
    address and happened not to use it, and the next model would.
    """
    db, user_id = await _session()
    rows = await _seed(db, user_id)
    writer = scripted()

    report = await mig.migrate_user(db, user_id, dry_run=False)

    everything_the_writer_saw = "\n".join(writer.prompts)
    assert "Maple Street" not in everything_the_writer_saw
    assert FORGOTTEN not in everything_the_writer_saw
    assert "4111 1111 1111 1111" not in everything_the_writer_saw
    assert DOC_CHUNK not in everything_the_writer_saw

    # …and each one has an answer in the report, not silence.
    assert _by_id(report, rows["forgotten"])["disposition"] == mig.SKIP_DELETED
    assert _by_id(report, rows["card"])["disposition"] == mig.SKIP_SECRET
    assert _by_id(report, rows["doc"])["disposition"] == mig.SKIP_DOCUMENT
    assert _by_id(report, rows["superseded"])["disposition"] == mig.SKIP_SUPERSEDED
    for key in ("forgotten", "card", "doc", "superseded"):
        assert len(_by_id(report, rows[key])["reason"]) > 20


async def test_a_soft_deleted_row_with_no_audit_event_is_still_untouched(scripted):
    """`memory_events` is corroboration, not the gate. A tenant whose audit
    table is empty or unreadable must still keep its deletions deleted."""
    db, user_id = await _session()
    rows = await _seed(db, user_id)
    for event in (await db.execute(select(MemoryEvent))).scalars().all():
        await db.delete(event)
    await db.commit()
    writer = scripted()

    report = await mig.migrate_user(db, user_id, dry_run=False)
    assert "Maple Street" not in "\n".join(writer.prompts)
    assert _by_id(report, rows["forgotten"])["disposition"] == mig.SKIP_DELETED


async def test_the_legacy_table_is_never_written_to(scripted):
    """The rollback IS the legacy table. No supersede stamps, no deletes,
    and no `file_slug` backfill — not even on the rows that have none."""
    db, user_id = await _session()
    await _seed(db, user_id)
    # A row that predates the last organize pass: no file_slug at all.
    db.add(_mem(user_id=user_id, content="Plays five-a-side on Thursdays",
                category="habits", file_slug=None))
    await db.commit()
    before = await _legacy_fingerprint(db, user_id)
    scripted()

    await mig.migrate_user(db, user_id, dry_run=False)

    assert await _legacy_fingerprint(db, user_id) == before


async def test_a_row_with_no_file_slug_still_routes(scripted):
    """`legacy_default_slug_for` is read-only input, and it is what makes a
    row written after the last round-8 organize pass migratable at all."""
    db, user_id = await _session()
    row = _mem(user_id=user_id, content=ANDROID, category="possessions",
               file_slug=None)
    db.add(row)
    await db.commit()
    writer = scripted()

    report = await mig.migrate_user(db, user_id, dry_run=False)
    assert "old file: knowledge" in writer.prompts[0]
    assert _by_id(report, row)["disposition"] == "kept"
    assert "uses an Android phone" in (await _bodies(db, user_id))[PROFILE_SLUG]


# ── The report ───────────────────────────────────────────────────────

async def test_every_source_row_has_a_disposition_and_a_reason(scripted):
    """"Where did this specific memory go?" answerable from the report
    alone, for every id, including the ones nothing was allowed to touch."""
    db, user_id = await _session()
    rows = await _seed(db, user_id)
    scripted()

    report = await mig.migrate_user(db, user_id, dry_run=False)

    reported = {d["id"] for d in report["dispositions"]}
    assert reported == {r.id for r in rows.values()}
    for d in report["dispositions"]:
        assert d["disposition"], d
        assert (d["reason"] or "").strip(), d
        assert d["disposition"] != mig.UNACCOUNTED, (
            "the writer was shown a row and said nothing about it"
        )
        if d["disposition"] in ("kept", "merged", "rewritten"):
            assert d["slug"], d

    assert report["before"]["legacy"]["rows_total"] == len(rows)
    assert report["before"]["legacy"]["by_legacy_file"]["knowledge"] >= 4
    assert sum(report["tallies"].values()) == len(rows)
    after = {f["slug"]: f for f in report["after"]["files"]}
    assert after[PROFILE_SLUG]["bullets"] == len(
        parse_bullets((await _bodies(db, user_id))[PROFILE_SLUG])
    )


async def test_a_row_the_writer_ignores_is_named_not_dropped_quietly(monkeypatch):
    """`unaccounted` exists so a writer defect cannot look like a decision."""
    db, user_id = await _session()
    row = _mem(user_id=user_id, content="Grew up in Tehran", category="identity")
    db.add(row)
    await db.commit()

    class _Silent:
        async def complete_with_json(self, messages, **kw):
            return _Resp(json.dumps({"ops": [], "dispositions": []}))

    monkeypatch.setattr(curator, "_llm", lambda api_key: _Silent())
    report = await mig.migrate_user(db, user_id, dry_run=False)

    entry = next(d for d in report["dispositions"] if d["id"] == row.id)
    assert entry["disposition"] == mig.UNACCOUNTED
    assert "did not account for it" in entry["reason"]


# ── The change log ───────────────────────────────────────────────────

async def test_the_memory_log_gets_one_line_per_file_not_one_per_op(scripted):
    """A migration replaying a corpus through the writer would otherwise
    open the user's log on upgrade day with dozens of lines about ops they
    never asked for."""
    db, user_id = await _session()
    await _seed(db, user_id)
    scripted()

    report = await mig.migrate_user(db, user_id, dry_run=False)
    changes = (await db.execute(
        select(MemoryFileChange).where(MemoryFileChange.user_id == user_id)
    )).scalars().all()

    assert len(changes) == len(report["files_filled"])
    assert {c.file_slug for c in changes} == set(report["files_filled"])
    assert {c.summary for c in changes} == {mig.MIGRATED_SUMMARY}
    kinds = {c.file_slug: c.kind for c in changes}
    assert kinds["topics/music"] == "created"
    assert kinds[PROFILE_SLUG] == "updated", (
        "Profile already existed — calling that a creation is a lie in the log"
    )
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", c.day_key) for c in changes)


# ── Idempotence, resume, dry run ─────────────────────────────────────

async def test_running_it_twice_changes_nothing(scripted):
    db, user_id = await _session()
    await _seed(db, user_id)
    scripted()

    first = await mig.migrate_user(db, user_id, dry_run=False)
    bodies_after_first = await _bodies(db, user_id)
    changes_after_first = len((await db.execute(select(MemoryFileChange))).scalars().all())

    second = await mig.migrate_user(db, user_id, dry_run=False)

    assert second["status"] == "already_completed"
    assert await _bodies(db, user_id) == bodies_after_first
    assert len((await db.execute(select(MemoryFileChange))).scalars().all()) \
        == changes_after_first
    assert second["dispositions"] == first["dispositions"]


async def test_an_interrupted_run_resumes_without_duplicating(scripted):
    """Killed mid-corpus, the next run continues from the ledger.

    The ledger is the report's own dispositions — a row already accounted
    for is not fed again — and the batch that was in flight IS fed again,
    because losing a fact is unrecoverable and a duplicate bullet is one
    instruct away. `validate_ops` refuses the exact-duplicate add, so the
    ordinary case of that re-feed is a no-op.
    """
    db, user_id = await _session()
    rows = await _seed(db, user_id)

    writer = scripted(fail_on_first_round_batch=2)
    partial = await mig.migrate_user(db, user_id, dry_run=False)
    assert partial["batches"]["failed"] == 1
    failed_ids = [d["id"] for d in partial["dispositions"]
                  if d["disposition"] == mig.FAILED]
    assert failed_ids, "the injected failure did not land"

    # A failed batch leaves the marker FAILED, not completed: rows nothing
    # judged mean the run is not finished, and the ordinary retry policy is
    # what re-feeds them. Simulate the grace having elapsed rather than
    # sleeping an hour.
    marker = await mig._marker(db)
    assert marker.status == "failed"
    progress = json.loads(marker.progress_json)
    progress["report"]["last_failed_at"] = "2020-01-01T00:00:00"
    marker.progress_json = json.dumps(progress)
    await db.commit()

    scripted()
    resumed = await mig.migrate_user(db, user_id, dry_run=False)

    bodies = await _bodies(db, user_id)
    for body in bodies.values():
        bullets = parse_bullets(body)
        assert len(bullets) == len(set(bullets)), f"duplicate bullet in {body!r}"
    assert {d["id"] for d in resumed["dispositions"]} == {r.id for r in rows.values()}
    assert not [d for d in resumed["dispositions"]
                if d["disposition"] == mig.FAILED]
    assert (await mig._marker(db)).status == "completed"
    changes = (await db.execute(select(MemoryFileChange))).scalars().all()
    assert len(changes) == len(set(c.file_slug for c in changes)), (
        "a resumed run filed a second migration line for the same file"
    )


async def test_a_dry_run_writes_absolutely_nothing(scripted):
    """Report produced, database unchanged — asserted on a full snapshot of
    every table this migration could possibly touch."""
    db, user_id = await _session()
    await _seed(db, user_id)
    scripted()

    async def snapshot_everything():
        return {
            "legacy": await _legacy_fingerprint(db, user_id),
            "files": [
                (f.slug, f.section, f.title, f.description, f.body_md, f.links_json)
                for f in (await db.execute(
                    select(MemoryFile).order_by(MemoryFile.slug)
                )).scalars().all()
            ],
            "changes": [
                (c.file_slug, c.kind, c.summary)
                for c in (await db.execute(select(MemoryFileChange))).scalars().all()
            ],
            "markers": [
                (m.migration_name, m.status, m.progress_json)
                for m in (await db.execute(select(MigrationStatus))).scalars().all()
            ],
        }

    before = await snapshot_everything()
    report = await mig.migrate_user(db, user_id, dry_run=True)
    assert await snapshot_everything() == before

    assert report["dry_run"] is True
    assert report["status"] == "completed"
    assert report["dispositions"], "a dry run still owes a full report"
    after = {f["slug"]: f for f in report["after"]["files"]}
    assert "goes by the nickname jojo" in after[PROFILE_SLUG]["body_md"]
    assert "topics/music" in after, "the dry run must project created files too"


async def test_a_dry_run_projects_across_batches_not_just_the_first(scripted):
    """Batch 2 must see what batch 1 would have written, or the IELTS pair
    lands twice on paper and the report's "after" is a fiction."""
    db, user_id = await _session()
    await _seed(db, user_id)
    writer = scripted()

    report = await mig.migrate_user(db, user_id, dry_run=True)

    assert len(writer.prompts) > 1, "the corpus did not batch"
    later = writer.prompts[-1]
    assert "goes by the nickname jojo" in later, (
        "the last batch was shown the ORIGINAL bodies, so it could not merge"
    )
    after = {f["slug"]: f for f in report["after"]["files"]}
    bullets = parse_bullets(after[PROFILE_SLUG]["body_md"])
    assert len(bullets) == len(set(bullets))


async def test_a_tenant_with_no_rows_completes_and_still_gets_its_files(scripted):
    db, user_id = await _session()
    scripted()

    report = await mig.migrate_user(db, user_id, dry_run=False)

    assert report["status"] == "completed"
    assert report["dispositions"] == []
    assert set((await _bodies(db, user_id))) >= {
        "you/profile", "you/current-context", "learned",
    }


# ── The marker lifecycle ─────────────────────────────────────────────

async def test_a_failure_records_attempts_and_the_grace_defers_the_retry(scripted):
    db, user_id = await _session()
    await _seed(db, user_id)

    async def boom(*a, **kw):
        raise RuntimeError("db went away")

    original = mig.read_corpus
    mig.read_corpus = boom
    try:
        with pytest.raises(RuntimeError):
            await mig.migrate_user_guarded(db, user_id, dry_run=False)
    finally:
        mig.read_corpus = original

    marker = await mig._marker(db)
    assert marker.status == "failed"
    assert "db went away" in (marker.error_message or "")

    scripted()
    deferred = await mig.migrate_user(db, user_id, dry_run=False)
    assert deferred.get("retry_deferred") is True, (
        "a failure that retries immediately retries at one model call per row"
    )


async def test_the_force_env_resets_a_completed_marker(scripted, monkeypatch):
    db, user_id = await _session()
    await _seed(db, user_id)
    scripted()
    await mig.migrate_user(db, user_id, dry_run=False)
    assert (await mig._marker(db)).status == "completed"

    monkeypatch.setenv(mig.FORCE_ENV, "true")
    scripted()
    again = await mig.migrate_user(db, user_id, dry_run=False)
    assert again["status"] != "already_completed"


# ── Rollback ─────────────────────────────────────────────────────────

async def test_rollback_returns_the_tenant_to_its_round_8_state(scripted):
    """Report-driven: created files are dropped, filled files are restored
    to the exact body they had, the change log goes, and the legacy rows —
    which is where round 8's content actually lives — were never touched."""
    db, user_id = await _session()
    await _seed(db, user_id)
    scripted()

    legacy_before = await _legacy_fingerprint(db, user_id)
    await mig.migrate_user(db, user_id, dry_run=False)
    assert "topics/music" in await _bodies(db, user_id)

    result = await mig.rollback(db, user_id)

    assert result["used_report"] is True
    bodies = await _bodies(db, user_id)
    assert "topics/music" not in bodies
    assert "areas/ielts" not in bodies
    assert bodies[PROFILE_SLUG] == "", "a filled system file was not emptied"
    assert (await db.execute(select(MemoryFileChange))).scalars().all() == []
    assert (await mig._marker(db)).status == "not_started"
    assert await _legacy_fingerprint(db, user_id) == legacy_before


async def test_a_hard_rollback_needs_no_report(scripted):
    db, user_id = await _session()
    await _seed(db, user_id)
    scripted()
    await mig.migrate_user(db, user_id, dry_run=False)
    marker = await mig._marker(db)
    marker.progress_json = None
    await db.commit()

    result = await mig.rollback(db, user_id, hard=True)

    assert result["used_report"] is False
    bodies = await _bodies(db, user_id)
    assert not [s for s in bodies if s.startswith(("topics/", "areas/", "people/"))]
    assert bodies["you/profile"] == ""
    assert "you/profile" in bodies, "a system file is emptied, never dropped"


# ── The scheduler slot ───────────────────────────────────────────────

def test_the_migration_hangs_off_the_one_maintenance_slot():
    """Not a second registration in `agent_main`. The boot one-shot
    (T+180s DateTrigger) and the nightly cron both call
    `run_memory_maintenance`, so hooking there is what keeps one slot to
    keep on a CronTrigger instead of two."""
    from pathlib import Path

    ops_src = Path(ops.__file__).read_text()
    assert "run_scheduled_migration" in ops_src
    at = ops_src.index("async def run_memory_maintenance")
    body = ops_src[at:]
    assert body.index("ensure_system_files") < body.index("run_scheduled_migration"), (
        "the writer routes the owner's facts into you/profile and cannot add "
        "to a file that does not exist yet"
    )

    agent_main = Path(ops.__file__).resolve().parents[2] / "agent_main.py"
    src = agent_main.read_text()
    assert 'id="memory_file_migration_boot"' in src
    assert "seconds=180" in src
    assert "memory_v3_migration" not in src, (
        "the migration must not become a second scheduler registration"
    )


# ══ The fleet driver (platform side) ═════════════════════════════════
#
# Loaded by path rather than imported: `backend/scripts/` is not a package
# on the test path, and the driver's own `if __name__ == "__main__"` guard
# means importing it is free.

def _load_driver(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(ops.__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeClient:
    """Stands in for httpx.AsyncClient. Records what the driver asked for."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def _answer(self, method, url, kwargs):
        self.calls.append((method, url, kwargs.get("params"),
                           kwargs.get("headers", {}).get("X-Agent-Key")))
        for pattern, answer in self.routes:
            if pattern in url:
                if isinstance(answer, Exception):
                    raise answer
                return _FakeResponse(*answer)
        raise AssertionError(f"no stub for {url}")

    async def get(self, url, **kw):
        return self._answer("GET", url, kw)

    async def post(self, url, **kw):
        return self._answer("POST", url, kw)


def _install_fleet(monkeypatch, driver, routes, tenants):
    async def candidates(user_ids):
        return tenants

    client = _FakeClient(routes)
    monkeypatch.setattr(driver, "_candidates", candidates)
    monkeypatch.setattr(driver.httpx, "AsyncClient", lambda **kw: client)
    return client


_TENANT = [("u" * 36, "founder@toup.ai", "https://agent.example", "SECRET-KEY")]


async def test_the_driver_records_the_before_state_before_it_writes(
    monkeypatch, tmp_path, capsys,
):
    """The in-band backup. The nightly pg_dump is the fleet snapshot; this
    is the per-tenant record that makes ONE tenant's rollback checkable
    without restoring a database — so it has to be fetched BEFORE the POST,
    not after."""
    driver = _load_driver("memory_v3_migrate_fleet")
    before = {"status": "not_started", "report": None,
              "snapshot": {"legacy": {"rows_total": 4}, "files": []}}
    result = {"status": "completed", "tallies": {"kept": 2, "dropped": 2},
              "files_filled": ["you/profile"]}
    client = _install_fleet(monkeypatch, driver, [
        ("migrate-v3/report", (200, before)),
        ("migrate-v3", (200, result)),
    ], _TENANT)
    out = tmp_path / "fleet.json"
    monkeypatch.setattr("sys.argv", [
        "x", "--apply", "--out", str(out),
    ])

    assert await driver.main() == 0

    methods = [(m, "report" in u) for m, u, *_ in client.calls]
    assert methods[0] == ("GET", True), "the before-state was not taken first"
    assert methods[1] == ("POST", False)
    artifact = json.loads(out.read_text())
    assert artifact["results"][0]["before"] == before
    assert artifact["results"][0]["result"] == result
    assert artifact["dry_run"] is False
    # The key is used and never printed.
    assert all(c[3] == "SECRET-KEY" for c in client.calls)
    assert "SECRET-KEY" not in capsys.readouterr().out
    assert "SECRET-KEY" not in out.read_text()


async def test_the_driver_is_dry_by_default_and_says_so(
    monkeypatch, tmp_path, capsys,
):
    driver = _load_driver("memory_v3_migrate_fleet")
    client = _install_fleet(monkeypatch, driver, [
        ("migrate-v3/report", (200, {"status": "not_started", "snapshot": {}})),
        ("migrate-v3", (200, {"status": "completed", "dry_run": True,
                              "tallies": {}, "files_filled": []})),
    ], _TENANT)
    monkeypatch.setattr("sys.argv", ["x", "--out", str(tmp_path / "d.json")])

    assert await driver.main() == 0

    post = next(c for c in client.calls if c[0] == "POST")
    assert post[2]["dry_run"] == "true"
    assert "DRY RUN" in capsys.readouterr().out


async def test_an_unreachable_tenant_is_reported_not_retried(
    monkeypatch, tmp_path, capsys,
):
    """A container that is down stays down for the length of the pass. A
    retry loop here turns one bad tenant into a stalled fleet."""
    driver = _load_driver("memory_v3_migrate_fleet")
    client = _install_fleet(monkeypatch, driver, [
        ("migrate-v3/report", (503, {"detail": "nope"})),
    ], _TENANT * 2)
    monkeypatch.setattr("sys.argv", [
        "x", "--apply", "--continue-on-error", "--out", str(tmp_path / "u.json"),
    ])

    assert await driver.main() == 0
    assert len(client.calls) == 2, "an unreachable tenant was retried"
    assert "UNREACHABLE" in capsys.readouterr().out


async def test_a_completed_tenant_is_skipped_on_apply(monkeypatch, tmp_path):
    """Idempotence is the tenant's job, but the driver must not spend a
    round trip re-asking — and must not report the skip as a migration."""
    driver = _load_driver("memory_v3_migrate_fleet")
    client = _install_fleet(monkeypatch, driver, [
        ("migrate-v3/report", (200, {"status": "completed", "snapshot": {}})),
    ], _TENANT)
    monkeypatch.setattr("sys.argv", ["x", "--apply", "--out", str(tmp_path / "s.json")])

    assert await driver.main() == 0
    assert not [c for c in client.calls if c[0] == "POST"]


def test_the_rollback_driver_refuses_to_run_unscoped(monkeypatch, capsys):
    """`--all` has to be typed. A rollback that defaults to the whole fleet
    is one flag away from being the incident."""
    driver = _load_driver("memory_v3_rollback")
    monkeypatch.setattr("sys.argv", ["x", "--apply"])
    import asyncio as _aio

    assert _aio.get_event_loop_policy().new_event_loop().run_until_complete(
        driver.main()
    ) == 2
    assert "refusing to run with no scope" in capsys.readouterr().out


async def test_the_rollback_driver_sends_the_exact_confirmation(
    monkeypatch, tmp_path,
):
    driver = _load_driver("memory_v3_rollback")
    client = _install_fleet(monkeypatch, driver, [
        ("migrate-v3/report", (200, {"status": "completed", "report": {
            "files_created": ["topics/music"], "files_filled": [
                "topics/music", "you/profile"]}})),
        ("rollback", (200, {"rolled_back": True, "files_deleted": ["topics/music"]})),
    ], _TENANT)
    monkeypatch.setattr("sys.argv", [
        "x", "--user", "u" * 36, "--apply", "--out", str(tmp_path / "r.json"),
    ])

    assert await driver.main() == 0
    post = next(c for c in client.calls if c[0] == "POST")
    assert post[2]["confirm"] == "ROLLBACK MEMORY V3"
    assert post[2]["confirm"] == driver.CONFIRM == mig.ROLLBACK_CONFIRM


async def test_the_maintenance_slot_actually_runs_the_migration(monkeypatch, scripted):
    """The wiring, driven rather than grepped.

    `run_memory_maintenance` is what both APScheduler registrations point at
    (boot T+180s DateTrigger, nightly cron). Calling it must create the
    system files AND migrate, and the second call must cost the marker check
    and nothing else.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=[
            User.__table__, Memory.__table__, MemoryEvent.__table__,
            MemoryFile.__table__, MemoryFileChange.__table__,
            MigrationStatus.__table__,
        ])
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    async with maker() as db:
        db.add(User(id=user_id, email="nariman@toup.ai", hashed_password="x",
                    name="Nariman Hosseini"))
        await db.commit()
        await _seed(db, user_id)
    forget_cached_identity()

    import app.db.database as database
    from app.config import settings

    monkeypatch.setattr(database, "async_session_maker", maker)
    monkeypatch.setattr(settings, "user_id", user_id, raising=False)
    writer = scripted()

    first = await ops.run_memory_maintenance()
    assert first["system_files_created"] == 3
    assert first["memory_v3_migration"]["status"] == "completed"
    async with maker() as db:
        bodies = await _bodies(db, user_id)
    assert "goes by the nickname jojo" in bodies[PROFILE_SLUG]

    calls = len(writer.prompts)
    second = await ops.run_memory_maintenance()
    assert second["memory_v3_migration"] == {"skipped": "already_completed"}
    assert len(writer.prompts) == calls, "the nightly slot re-ran the migration"


async def test_the_maintenance_slot_survives_a_migration_that_raises(monkeypatch):
    """A failing migration must not take the system-file check down with it —
    that check is what keeps `you/profile` existing for the injection."""
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=[
            User.__table__, MemoryFile.__table__, MemoryFileChange.__table__,
        ])
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    async with maker() as db:
        db.add(User(id=user_id, email="a@b.c", hashed_password="x", name="A"))
        await db.commit()

    import app.db.database as database
    from app.config import settings

    monkeypatch.setattr(database, "async_session_maker", maker)
    monkeypatch.setattr(settings, "user_id", user_id, raising=False)

    async def boom():
        raise RuntimeError("migration_status table is missing")

    monkeypatch.setattr(mig, "run_scheduled_migration", boom)

    result = await ops.run_memory_maintenance()
    assert result["system_files_created"] == 3
    assert result["memory_v3_migration"]["status"] == "failed"


async def test_an_early_failure_still_counts_toward_giving_up(monkeypatch, scripted):
    """The attempt counter is persisted when the run STARTS, not at the
    first batch boundary.

    A failure before the first batch — an unreadable corpus, a dead pool —
    never reaches the per-batch progress write. A counter written only there
    stays at 1 forever, the "give up after 3" cap is unreachable, and the
    nightly slot retries a broken tenant with a model call in it for the
    length of the deploy.
    """
    db, user_id = await _session()
    await _seed(db, user_id)

    async def boom(*a, **kw):
        raise RuntimeError("pool is gone")

    monkeypatch.setattr(mig, "read_corpus", boom)

    for attempt in range(mig.MAX_AUTO_RETRY_ATTEMPTS):
        with pytest.raises(RuntimeError):
            await mig.migrate_user_guarded(db, user_id, dry_run=False)
        marker = await mig._marker(db)
        progress = json.loads(marker.progress_json)
        assert progress["attempts"] == attempt + 1, progress
        # Age the failure past the grace so the next call actually retries.
        progress["report"]["last_failed_at"] = "2020-01-01T00:00:00"
        marker.progress_json = json.dumps(progress)
        await db.commit()

    gave_up = await mig.migrate_user(db, user_id, dry_run=False)
    assert gave_up.get("gave_up") is True
    assert gave_up["status"] == "failed"


# ══ The orphan retry (a PARTIAL refusal) ═════════════════════════════
#
# The defect these pin: `validate_ops` refuses OPS, and the writer only
# re-asked when the WHOLE proposal was refused (`plan.complaints and not
# plan.accepted`). One accepted op made `plan.accepted` truthy, so a batch
# that lost one bullet to the lint never re-asked and the refused ROW was
# simply gone — a decision about an op becoming, silently, a decision about
# a row that nobody made.

class _PartialWriter:
    """One accepted op and one refused op in the SAME batch.

    Round 1 proposes a clean bullet for L1 and a UUID-carrying bullet for
    L2. Round 2 — if it happens at all — proposes L2 without the id.
    """

    RETRY_MARK = _ScriptedWriter.RETRY_MARK

    def __init__(self, fix_on_retry=True):
        self.prompts: list[str] = []
        self.fix_on_retry = fix_on_retry

    @property
    def retry_prompts(self):
        return [p for p in self.prompts if self.RETRY_MARK in p]

    async def complete_with_json(self, messages, model=None, temperature=None, **kw):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        entries = prompt.split("THE OLD ENTRIES:", 1)[1]
        for tail in (self.RETRY_MARK, "\n\nWHAT IS"):
            entries = entries.split(tail, 1)[0]
        found = dict(_ENTRY_RE.findall(entries))

        ops_out, dispositions = [], []
        for handle, content in found.items():
            if "Android" in content:
                ops_out.append({
                    "op": "add", "slug": PROFILE_SLUG,
                    "bullet": "uses an Android phone",
                    "change": "Added Profile: uses an Android phone.",
                    "refs": [handle],
                })
                dispositions.append({"ref": handle, "verdict": "kept",
                                     "slug": PROFILE_SLUG, "reason": "their phone"})
            else:
                bullet = (
                    "runs the workspace app"
                    if (self.RETRY_MARK in prompt and self.fix_on_retry)
                    else "runs the workspace app "
                         "7f3c2b1a-9d4e-4c8f-a1b2-c3d4e5f60718"
                )
                ops_out.append({
                    "op": "add", "slug": PROFILE_SLUG, "bullet": bullet,
                    "change": "Added Profile: the workspace app.",
                    "refs": [handle],
                })
                dispositions.append({"ref": handle, "verdict": "kept",
                                     "slug": PROFILE_SLUG,
                                     "reason": "the app they work in"})
        return _Resp(json.dumps({"ops": ops_out, "dispositions": dispositions}))


async def _partial_tenant(monkeypatch, writer):
    monkeypatch.setattr(curator, "EXTRACTION_RETRY_BACKOFF_S", 0)
    db, user_id = await _session()
    rows = {
        "android": _mem(user_id=user_id, content=ANDROID, category="possessions",
                        created_at=datetime(2026, 5, 1, 3, 0)),
        "uuid": _mem(user_id=user_id, content=UUID_ROW, category="possessions",
                     created_at=datetime(2026, 5, 1, 3, 1)),
    }
    for row in rows.values():
        db.add(row)
    await db.commit()
    monkeypatch.setattr(curator, "_llm", lambda api_key: writer)
    return db, user_id, rows


async def test_a_partial_refusal_re_asks_and_the_refused_row_lands(monkeypatch):
    """One accepted, one refused, one batch. The refused row must come back.

    Before the fix this batch wrote "uses an Android phone" and dropped the
    other row on the floor with no re-ask — `plan.accepted` was truthy, so
    the only retry in the writer never fired.
    """
    writer = _PartialWriter()
    db, user_id, rows = await _partial_tenant(monkeypatch, writer)

    report = await mig.migrate_user(db, user_id, dry_run=False)
    bodies = await _bodies(db, user_id)

    assert writer.retry_prompts, "a partial refusal did not re-ask"
    assert _by_id(report, rows["android"])["disposition"] == "kept"
    refused = _by_id(report, rows["uuid"])
    assert refused["disposition"] == "kept", "the refused row never came back"
    assert refused["slug"] == PROFILE_SLUG
    assert "runs the workspace app" in bodies[PROFILE_SLUG]
    # The rule itself never bent: the id is still nowhere.
    assert "7f3c2b1a" not in bodies[PROFILE_SLUG]


async def test_the_retry_is_told_why_and_shown_what_already_landed(monkeypatch):
    """Two constraints in one prompt: the validator's own words (which is
    what makes the re-ask actionable rather than a coin flip), and the state
    AFTER the accepted ops — or the retry re-proposes what already landed
    and the batch double-writes."""
    writer = _PartialWriter()
    db, user_id, _ = await _partial_tenant(monkeypatch, writer)

    await mig.migrate_user(db, user_id, dry_run=False)

    retry = writer.retry_prompts[0]
    assert "internal ids are never stored in a memory file" in retry
    assert "not negotiable and this is not an appeal" in retry
    assert "uses an Android phone" in retry, (
        "the retry was shown the ORIGINAL bodies, so it can re-propose what "
        "already landed"
    )
    # Scoped to the orphan: the accepted row is not re-fed.
    entries = retry.split("THE OLD ENTRIES:", 1)[1].split(writer.RETRY_MARK, 1)[0]
    assert UUID_ROW in entries
    assert ANDROID not in entries


async def test_exactly_one_extra_round_and_no_loop(monkeypatch):
    """A writer that never fixes anything must cost ONE extra call, not a
    retry storm — and the row must end as a drop, not as an unaccounted
    silence."""
    writer = _PartialWriter(fix_on_retry=False)
    db, user_id, rows = await _partial_tenant(monkeypatch, writer)

    report = await mig.migrate_user(db, user_id, dry_run=False)

    assert len(writer.prompts) == 2, f"{len(writer.prompts)} model calls"
    assert len(writer.retry_prompts) == 1
    stubborn = _by_id(report, rows["uuid"])
    assert stubborn["disposition"] == "dropped"
    # Refused twice, and BOTH refusals are in the answer.
    assert stubborn["reason"].count("internal ids are never stored") == 2, (
        stubborn["reason"]
    )
    assert "7f3c2b1a" not in "".join((await _bodies(db, user_id)).values())


async def test_a_retry_that_raises_leaves_round_one_s_answer_standing(monkeypatch):
    """Losing the re-ask is not losing the record. The row stays dropped
    with the complaint that refused it, and the batch is NOT marked failed —
    the accepted half of it really did land."""
    writer = _PartialWriter()

    class _Exploding(_PartialWriter):
        async def complete_with_json(self, messages, **kw):
            if self.RETRY_MARK in messages[0]["content"]:
                self.prompts.append(messages[0]["content"])
                raise RuntimeError("provider died on the re-ask")
            return await _PartialWriter.complete_with_json(self, messages, **kw)

    writer = _Exploding()
    db, user_id, rows = await _partial_tenant(monkeypatch, writer)

    report = await mig.migrate_user(db, user_id, dry_run=False)

    assert report["batches"]["failed"] == 0
    assert _by_id(report, rows["android"])["disposition"] == "kept"
    refused = _by_id(report, rows["uuid"])
    assert refused["disposition"] == "dropped"
    assert "internal ids are never stored" in refused["reason"]
    assert any("retry" in e for e in report["errors"]), report["errors"]


async def test_the_retry_never_lets_a_refused_SHAPE_through(monkeypatch):
    """The retry exists so the FACT survives, never so the refused shape
    gets in on the second try. A writer that simply repeats itself gets the
    same refusal, and the three invariants are untouched by all of this."""
    writer = _PartialWriter(fix_on_retry=False)
    db, user_id, _ = await _partial_tenant(monkeypatch, writer)

    await mig.migrate_user(db, user_id, dry_run=False)

    bodies = await _bodies(db, user_id)
    assert "7f3c2b1a-9d4e-4c8f-a1b2-c3d4e5f60718" not in "".join(bodies.values())
    assert not [s for s in bodies if s.startswith("people/")]


async def test_a_dry_run_re_asks_too_and_still_writes_nothing(monkeypatch):
    """The dry run's whole value is that its report is what a real run would
    produce. A retry that only happened on the wet path would make the dry
    report pessimistic in exactly the cases that matter."""
    writer = _PartialWriter()
    db, user_id, rows = await _partial_tenant(monkeypatch, writer)

    before = [
        (f.slug, f.body_md) for f in
        (await db.execute(select(MemoryFile).order_by(MemoryFile.slug))).scalars().all()
    ]
    report = await mig.migrate_user(db, user_id, dry_run=True)
    after = [
        (f.slug, f.body_md) for f in
        (await db.execute(select(MemoryFile).order_by(MemoryFile.slug))).scalars().all()
    ]

    assert writer.retry_prompts, "the dry run did not re-ask"
    assert before == after, "the dry run wrote something"
    assert _by_id(report, rows["uuid"])["disposition"] == "kept"
    projected = {f["slug"]: f["body_md"] for f in report["after"]["files"]}
    assert "runs the workspace app" in projected[PROFILE_SLUG]
    assert "uses an Android phone" in projected[PROFILE_SLUG]
