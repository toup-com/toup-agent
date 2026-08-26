"""The v3 ops engine and the file store (rebuild-2026-08-v3 §1.3–§2.2).

Round 8's version of this file pinned an ops contract over `memories` ROWS
addressed by positional handles (`e1..eN`). v3 addresses BULLETS by their
exact text, writes whole bodies, and logs every mutation — so this is a
rewrite, not an edit.

Self-contained sqlite (its own engine), so RUN_MODE is irrelevant here.
"""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.base import Base
from app.db.models.memory import MemoryFile, MemoryFileChange
from app.db.models.user import User
from app.memory_files import (
    CURRENT_CONTEXT_SLUG,
    LEARNED_SLUG,
    MAX_BODY_CHARS,
    PROFILE_SLUG,
    SYSTEM_FILES,
    description_problem,
)
from app.services import memory_file_ops as ops
from app.services.user_identity import forget_cached_identity, resolve_user_identity

GOOD_DESC = (
    "Your IELTS preparation — tutor, dates and band targets; "
    "read when IELTS or the exam comes up."
)


async def _session(name: str = "Nariman Hosseini") -> tuple[AsyncSession, str]:
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[User.__table__, MemoryFile.__table__, MemoryFileChange.__table__],
        )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = maker()
    user_id = str(uuid.uuid4())
    db.add(User(
        id=user_id, email=f"{user_id[:8]}@test.local", hashed_password="x", name=name,
    ))
    await db.commit()
    forget_cached_identity()
    return db, user_id


async def _apply(db, user_id, raw_ops, *, identity=None):
    if identity is None:
        identity = await resolve_user_identity(db, user_id)
    await ops.ensure_system_files(db, user_id)
    await db.commit()
    plan = ops.validate_ops(raw_ops, await ops._all_files(db, user_id), identity=identity)
    result = await ops.apply_ops(db, user_id, plan)
    return plan, result


def _create_ielts(**over):
    op = {
        "op": "create_file", "section": "areas", "slug": "areas/ielts",
        "title": "IELTS", "description": GOOD_DESC,
    }
    op.update(over)
    return op


# ── The store ─────────────────────────────────────────────────────────

async def test_system_files_are_created_once_and_listed_in_order():
    db, user_id = await _session()
    assert len(await ops.ensure_system_files(db, user_id)) == 3
    await db.commit()
    assert await ops.ensure_system_files(db, user_id) == []   # idempotent

    listing = await ops.list_files(db, user_id)
    assert [s["section"] for s in listing["sections"]] == [
        "you", "people", "topics", "areas", "learned"
    ]
    by_slug = {f["slug"]: f for s in listing["sections"] for f in s["files"]}
    assert set(by_slug) == set(SYSTEM_FILES)
    # The listing payload is exactly five keys. Not one engine field.
    assert set(by_slug[PROFILE_SLUG]) == {
        "slug", "section", "title", "description", "updated_at"
    }


async def test_an_adopted_round_8_learned_row_gets_a_VALID_description():
    """The path EVERY already-live tenant takes, and it was one-sided.

    Round 8's `learned` row carries the same slug and the same meaning, so
    `ensure_system_files` adopts it rather than colliding with the unique
    index. But its description is a `purpose`-era sentence — "Corrections
    and lessons." — which fails DESCRIPTION_RE, and the repair only ran when
    the field was EMPTY. So every existing tenant kept an invalid
    description forever, in a string that is BOTH served to the client and
    injected into the prompt as this file's index line ("the description IS
    the signal", memory_files.index_line).

    It also poisons the eval set's lint metric with a defect no writer
    caused, which is how a measured invariant stops meaning anything.
    """
    db, user_id = await _session()
    db.add(MemoryFile(
        user_id=user_id, slug=LEARNED_SLUG, section="learned",
        title="Learned", description="Corrections and lessons.",
        body_md="- send reminders to this chat, not Telegram", position=5,
    ))
    await db.commit()

    created = await ops.ensure_system_files(db, user_id)
    await db.commit()

    # Adopted, not duplicated.
    assert LEARNED_SLUG not in [r.slug for r in created]
    rows = [r for r in await ops._all_files(db, user_id) if r.slug == LEARNED_SLUG]
    assert len(rows) == 1
    row = rows[0]

    assert description_problem(row.description) is None, row.description
    assert row.description == SYSTEM_FILES[LEARNED_SLUG]["description"]
    assert row.is_system is True
    # The BODY is never touched: the user's corrections survive the repair.
    assert row.body_md == "- send reminders to this chat, not Telegram"


async def test_a_valid_description_is_left_alone():
    """ANTI-VACUITY for the test above: "always overwrite" would pass it and
    would clobber a description the writer had legitimately regenerated."""
    db, user_id = await _session()
    await ops.ensure_system_files(db, user_id)
    await db.commit()
    rows = {r.slug: r for r in await ops._all_files(db, user_id)}
    custom = (
        "How to work with Dara — tone, tools and standing corrections; "
        "read when you are about to act on their behalf."
    )
    assert description_problem(custom) is None
    rows[LEARNED_SLUG].description = custom
    await db.commit()

    await ops.ensure_system_files(db, user_id)
    await db.commit()
    rows = {r.slug: r for r in await ops._all_files(db, user_id)}
    assert rows[LEARNED_SLUG].description == custom


async def test_round_8_rows_are_invisible_rather_than_mis_sectioned():
    """A tenant that booted a round-8 image has `knowledge`/`working` file
    rows. They stay on disk untouched — that is the rollback — but they
    must not surface inside a v3 section they were never written for."""
    db, user_id = await _session()
    db.add(MemoryFile(
        user_id=user_id, slug="knowledge", section="knowledge",
        title="Knowledge", description="Facts about your world.",
    ))
    await db.commit()

    listing = await ops.list_files(db, user_id)
    slugs = {f["slug"] for s in listing["sections"] for f in s["files"]}
    assert "knowledge" not in slugs
    assert await ops.get_file(db, user_id, "knowledge") is None
    # …and the row is still there.
    assert (await db.execute(
        select(MemoryFile).where(MemoryFile.slug == "knowledge")
    )).scalar_one_or_none() is not None


async def test_a_round_8_learned_row_is_adopted_not_duplicated():
    """`learned` is the one slug both models use, and (user_id, slug) is
    unique — a second INSERT would raise rather than degrade."""
    db, user_id = await _session()
    db.add(MemoryFile(
        user_id=user_id, slug=LEARNED_SLUG, section="learned",
        title="Learned", description=None, is_system=False, body_md="- keep this",
    ))
    await db.commit()

    await ops.ensure_system_files(db, user_id)
    await db.commit()
    rows = (await db.execute(
        select(MemoryFile).where(MemoryFile.slug == LEARNED_SLUG)
    )).scalars().all()
    assert len(rows) == 1
    assert rows[0].is_system is True
    assert rows[0].body_md == "- keep this"      # the body is never touched
    assert rows[0].description == SYSTEM_FILES[LEARNED_SLUG]["description"]


# ── Ops: the happy path ───────────────────────────────────────────────

async def test_create_then_add_in_one_batch_validates_and_persists():
    db, user_id = await _session()
    plan, result = await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts",
         "bullet": "IELTS exam booked for Aug 30, 2026",
         "change": "Added IELTS: exam booked for Aug 30."},
        {"op": "add", "slug": "areas/ielts",
         "bullet": "targeting band 7.5 overall",
         "change": "Added IELTS: band target."},
    ])
    assert plan.complaints == []
    assert result["applied"] == 3

    file = await ops.get_file(db, user_id, "areas/ielts")
    assert file["title"] == "IELTS"
    assert file["description"] == GOOD_DESC
    assert file["body_md"] == (
        "- IELTS exam booked for Aug 30, 2026\n- targeting band 7.5 overall"
    )
    assert file["section"] == "areas"
    # The payload a client sees carries no engine field.
    assert set(file) == {
        "slug", "section", "title", "description", "body_md", "links", "updated_at"
    }


async def test_rewrite_and_remove_address_a_bullet_by_its_exact_text():
    """No index arithmetic: a model that miscounts by one silently rewrites
    the wrong fact, and any concurrent write makes an index stale."""
    db, user_id = await _session()
    await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts", "bullet": "exam booked for Aug 30, 2026",
         "change": "Added IELTS: exam date."},
        {"op": "add", "slug": "areas/ielts", "bullet": "targeting band 7.5 overall",
         "change": "Added IELTS: band target."},
    ])

    plan, _ = await _apply(db, user_id, [
        {"op": "rewrite", "slug": "areas/ielts",
         "match": "exam booked for Aug 30, 2026",
         "bullet": "exam moved to Sep 13, 2026",
         "change": "Updated IELTS: exam moved to Sep 13."},
        {"op": "remove", "slug": "areas/ielts", "match": "targeting band 7.5 overall",
         "change": "Removed the old band target from IELTS."},
    ])
    assert plan.complaints == []
    file = await ops.get_file(db, user_id, "areas/ielts")
    assert file["body_md"] == "- exam moved to Sep 13, 2026"

    # A `match` that is not present is refused, not fuzzily matched.
    plan, result = await _apply(db, user_id, [
        {"op": "rewrite", "slug": "areas/ielts", "match": "exam moved to Sep 13",
         "bullet": "exam moved again", "change": "x"},
    ])
    assert result["applied"] == 0
    assert any("no bullet reads exactly" in c for c in plan.complaints)


async def test_every_mutation_writes_one_change_line_on_the_user_local_day():
    db, user_id = await _session()
    # A zone west of UTC, where "now" is often yesterday locally.
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one()
    user.timezone = "America/Toronto"
    await db.commit()

    await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts", "bullet": "exam booked for Aug 30, 2026",
         "change": "Added IELTS: exam booked for Aug 30."},
        {"op": "link", "slug": "areas/ielts", "links": [PROFILE_SLUG]},
    ])

    changes = (await db.execute(
        select(MemoryFileChange).where(MemoryFileChange.user_id == user_id)
    )).scalars().all()
    kinds = sorted(c.kind for c in changes)
    assert kinds == ["created", "updated"], "link is not a user-visible change"
    from zoneinfo import ZoneInfo
    expected = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d")
    assert {c.day_key for c in changes} == {expected}
    assert any(c.summary == "Added IELTS: exam booked for Aug 30." for c in changes)
    assert all(c.file_title == "IELTS" for c in changes)


async def test_the_log_groups_by_day_within_a_month():
    db, user_id = await _session()
    await _apply(db, user_id, [_create_ielts()])
    month = datetime.utcnow().strftime("%Y-%m")

    log = await ops.read_log(db, user_id, month)
    assert len(log["days"]) == 1
    entries = log["days"][0]["entries"]
    assert entries[0]["file_slug"] == "areas/ielts"
    assert entries[0]["kind"] == "created"
    assert set(entries[0]) == {"file_slug", "file_title", "kind", "summary", "at"}
    assert (await ops.read_log(db, user_id, "1999-01"))["days"] == []
    with pytest.raises(ValueError):
        await ops.read_log(db, user_id, "not-a-month")


# ── Ops: what must be refused ─────────────────────────────────────────

async def test_a_mutation_without_a_change_line_is_refused():
    """The change line is the memory log. An op that skips it writes a fact
    the user can never find out about."""
    db, user_id = await _session()
    plan, result = await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts", "bullet": "exam booked for Aug 30, 2026"},
    ])
    assert result["applied"] == 1                       # the create only
    assert any("`change` line is required" in c for c in plan.complaints)


async def test_the_owner_never_gets_a_people_file():
    """The v3 answer to "why is there a People file about me": the resolver
    is asked before the slug is minted."""
    db, user_id = await _session(name="Nariman Hosseini")
    plan, result = await _apply(db, user_id, [
        {"op": "create_file", "section": "people", "slug": "people/nariman",
         "title": "Nariman",
         "description": "The owner — his facts; read when he comes up."},
        {"op": "create_file", "section": "people", "slug": "people/nariman-hosseini",
         "title": "Nariman Hosseini",
         "description": "The owner again — facts; read when he comes up."},
        {"op": "create_file", "section": "people", "slug": "people/majid-tajik",
         "title": "Majid Tajik",
         "description": "The IELTS tutor — how he teaches; read when Majid comes up."},
    ])
    assert result["changed_files"] == ["people/majid-tajik"]
    assert sum("people/nariman" in c for c in plan.complaints) == 2


async def test_a_short_email_local_part_does_not_establish_an_identity():
    """`n@toup.ai` on a tenant whose users.name is still "Agent Owner":
    the alias set is {agent, agent owner, n}, and a one-character alias
    would flip `known` on for exactly the placeholder tenant the flag
    exists to report. Short aliases stay MATCHABLE, they just cannot be
    the sole evidence."""
    db, user_id = await _session(name="Agent Owner")
    from sqlalchemy import update

    await db.execute(update(User).where(User.id == user_id).values(email="n@toup.ai"))
    await db.commit()
    forget_cached_identity()
    identity = await resolve_user_identity(db, user_id)
    assert "n" in identity.aliases
    assert identity.known is False


async def test_an_unknown_identity_does_not_block_person_files():
    """A fresh tenant is 'Agent Owner' with an <hex>@agent.local email. The
    resolver must report `known=False` and the writer must fail OPEN —
    failing closed drops real facts on every new tenant."""
    db, user_id = await _session(name="Agent Owner")
    identity = await resolve_user_identity(db, user_id)
    assert identity.known is False
    _, result = await _apply(db, user_id, [
        {"op": "create_file", "section": "people", "slug": "people/sara",
         "title": "Sara", "description": "A colleague — where they work; read when Sara comes up."},
    ], identity=identity)
    assert result["applied"] == 1


async def test_the_voice_lint_runs_on_every_written_bullet():
    db, user_id = await _session()
    plan, result = await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts", "bullet": "You are targeting band 7.5",
         "change": "x"},
        {"op": "add", "slug": "areas/ielts",
         "bullet": "the tutor set max_results=1 on the practice search", "change": "x"},
    ])
    assert result["applied"] == 1
    assert any("subject is implied" in c for c in plan.complaints)
    assert any("tool parameters" in c for c in plan.complaints)


async def test_a_never_store_value_is_refused_at_the_bullet():
    db, user_id = await _session()
    plan, result = await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts",
         "bullet": "pays the tutor with card 4111 1111 1111 1111 every month",
         "change": "Added IELTS: payment card."},
    ])
    assert result["applied"] == 1
    assert any("never stored" in c for c in plan.complaints)


async def test_a_description_that_is_not_the_pattern_kills_the_create():
    db, user_id = await _session()
    plan, result = await _apply(db, user_id, [
        {"op": "create_file", "section": "topics", "slug": "topics/music",
         "title": "Music", "description": "Music stuff"},
    ])
    assert result["applied"] == 0
    assert any("read when" in c for c in plan.complaints)


async def test_a_slug_that_disagrees_with_the_declared_section_is_refused():
    """When the two CONFLICT, the slug decides — and the op is refused rather
    than silently re-filed, because the writer has said two different things
    about where a fact belongs and only it knows which it meant.

    This used to also assert that a BARE slug ("music", section "topics") was
    refused for having no section. That half is deliberately gone: there is no
    conflict there, the declared section is the only information in the op,
    and refusing it cost eight ops and a whole turn on CI run 32433614861.
    See `test_a_bare_slug_is_repaired_from_the_declared_section`."""
    db, user_id = await _session()
    plan, _ = await _apply(db, user_id, [
        {"op": "create_file", "section": "areas", "slug": "topics/music",
         "title": "Music",
         "description": "Music — taste and artists; read when music comes up."},
    ])
    assert any("is a topics file, not 'areas'" in c for c in plan.complaints)
    assert plan.accepted == []


async def test_the_per_file_cap_rejects_the_op_that_would_overflow_it():
    db, user_id = await _session()
    filler = "keeps a very long standing arrangement noted in detail " * 6
    ops_batch = [_create_ielts()]
    for i in range(200):
        ops_batch.append({
            "op": "add", "slug": "areas/ielts", "bullet": f"{filler} number {i}",
            "change": f"Added IELTS: note {i}.",
        })
    plan, _ = await _apply(db, user_id[:], ops_batch[:41])   # MAX_OPS is 40
    assert plan.complaints == ["too many ops (41 > 40)"]

    plan, _ = await _apply(db, user_id, ops_batch[:40])
    file = await ops.get_file(db, user_id, "areas/ielts")
    assert len(file["body_md"]) <= MAX_BODY_CHARS
    assert any("consolidate it first" in c for c in plan.complaints)


async def test_an_op_naming_a_file_that_does_not_exist_is_refused():
    db, user_id = await _session()
    plan, result = await _apply(db, user_id, [
        {"op": "add", "slug": "areas/nope", "bullet": "something true", "change": "x"},
        {"op": "add", "slug": "not a slug", "bullet": "something true", "change": "x"},
        {"op": "frobnicate", "slug": PROFILE_SLUG},
    ])
    assert result["applied"] == 0
    assert any("create it first" in c for c in plan.complaints)
    assert any("not a valid slug" in c for c in plan.complaints)
    assert any("unknown op" in c for c in plan.complaints)


async def test_links_must_resolve_and_a_dangling_one_is_dropped():
    """The writer is the last line of defence for `[[slug]]`: a link the
    client renders as tappable and that opens nothing is worse than none."""
    db, user_id = await _session()
    plan, _ = await _apply(db, user_id, [
        _create_ielts(),
        {"op": "create_file", "section": "people", "slug": "people/majid-tajik",
         "title": "Majid Tajik",
         "description": "The IELTS tutor — how he teaches; read when Majid comes up."},
        # Forward reference to a file created in the SAME batch: legal.
        {"op": "add", "slug": "areas/ielts",
         "bullet": "taught by [[people/majid-tajik]] over Teams",
         "change": "Added IELTS: tutor."},
        {"op": "link", "slug": "areas/ielts",
         "links": ["people/majid-tajik", "people/ghost"]},
    ])
    assert any("do not exist and were dropped" in c for c in plan.complaints)
    file = await ops.get_file(db, user_id, "areas/ielts")
    assert [l["slug"] for l in file["links"]] == ["people/majid-tajik"]
    assert file["links"][0]["title"] == "Majid Tajik"


async def test_a_bullet_op_on_a_prose_body_is_refused_not_flattened():
    """Current context is `##` layer headings with prose beneath (§6), and
    `parse_bullets` cannot see any of it — so re-rendering that body from
    its bullets DELETES the layers. An `add` there is refused rather than
    applied, and a `link` (which writes its own column) must not reflow the
    body either."""
    db, user_id = await _session()
    await ops.ensure_system_files(db, user_id)
    row = (await db.execute(
        select(MemoryFile).where(MemoryFile.slug == CURRENT_CONTEXT_SLUG)
    )).scalar_one()
    layered = "## Today\nPreparing for the IELTS exam on Aug 30.\n## Yesterday\nRested."
    row.body_md = layered
    await db.commit()

    plan, result = await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": CURRENT_CONTEXT_SLUG,
         "bullet": "has soccer at 5:20 PM today", "change": "x"},
        {"op": "link", "slug": CURRENT_CONTEXT_SLUG, "links": ["areas/ielts"]},
    ])
    assert any("not a bullet list" in c for c in plan.complaints)
    file = await ops.get_file(db, user_id, CURRENT_CONTEXT_SLUG)
    assert file["body_md"] == layered, "the layers were flattened"
    assert [l["slug"] for l in file["links"]] == ["areas/ielts"], (
        "a link must still apply — it writes its own column"
    )


async def test_current_context_prose_survives_a_batch_that_does_not_touch_it():
    """WS-3 owns that body's `##` layer shape. A curator pass that reads
    every file must not flatten it into bullets on the way past."""
    db, user_id = await _session()
    await ops.ensure_system_files(db, user_id)
    row = (await db.execute(
        select(MemoryFile).where(MemoryFile.slug == CURRENT_CONTEXT_SLUG)
    )).scalar_one()
    row.body_md = "## Today\nPreparing for the IELTS exam on Aug 30.\n## Yesterday\nRested."
    await db.commit()

    await _apply(db, user_id, [_create_ielts()])
    file = await ops.get_file(db, user_id, CURRENT_CONTEXT_SLUG)
    assert file["body_md"].startswith("## Today\nPreparing")


# ── Delete ────────────────────────────────────────────────────────────

async def test_delete_drops_a_normal_file_and_only_empties_a_system_one():
    db, user_id = await _session()
    await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts", "bullet": "exam booked for Aug 30, 2026",
         "change": "Added IELTS: exam date."},
    ])
    assert await ops.delete_file(db, user_id, "areas/ielts") is True
    assert await ops.get_file(db, user_id, "areas/ielts") is None

    profile = (await db.execute(
        select(MemoryFile).where(MemoryFile.slug == PROFILE_SLUG)
    )).scalar_one()
    profile.body_md = "- uses an Android phone"
    await db.commit()
    assert await ops.delete_file(db, user_id, PROFILE_SLUG) is True
    # The row survives — the injection depends on it existing.
    assert (await ops.get_file(db, user_id, PROFILE_SLUG))["body_md"] == ""
    assert await ops.delete_file(db, user_id, "areas/never-existed") is False

    kinds = [c.kind for c in (await db.execute(
        select(MemoryFileChange).where(MemoryFileChange.kind == "file_deleted")
    )).scalars().all()]
    assert kinds == ["file_deleted", "file_deleted"]


async def test_forget_everything_clears_files_and_the_log():
    db, user_id = await _session()
    await _apply(db, user_id, [_create_ielts()])
    removed = await ops.forget_everything(db, user_id)
    assert removed == 4                                  # 3 system + IELTS
    assert (await ops.list_files(db, user_id))["sections"][1]["files"] == []
    month = datetime.utcnow().strftime("%Y-%m")
    assert (await ops.read_log(db, user_id, month))["days"] == []


# ── Search + the injection load ───────────────────────────────────────

async def test_search_is_file_attributed_over_bodies_and_titles():
    db, user_id = await _session()
    await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts", "bullet": "taught by Majid over Teams",
         "change": "Added IELTS: tutor."},
    ])
    hits = await ops.search_files(db, user_id, "teams")
    assert [h["slug"] for h in hits] == ["areas/ielts"]
    assert set(hits[0]) == {"slug", "title", "snippet"}
    assert "Teams" in hits[0]["snippet"]
    assert await ops.search_files(db, user_id, "") == []
    assert await ops.search_files(db, user_id, "nothing matches this") == []


async def test_load_brain_reads_the_three_files_plus_a_ranked_index():
    db, user_id = await _session()
    await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts", "bullet": "exam booked for Aug 30, 2026",
         "change": "Added IELTS: exam date."},
        {"op": "create_file", "section": "topics", "slug": "topics/music",
         "title": "Music",
         "description": "Music taste — artists; read when music comes up."},
        # NOTE: three words minimum. "likes Googoosh" — the contract's own
        # change-line example — would be REJECTED by its own bullet lint.
        {"op": "add", "slug": "topics/music", "bullet": "likes Googoosh and Ebi",
         "change": "Added Music: likes Googoosh."},
    ])
    profile = (await db.execute(
        select(MemoryFile).where(MemoryFile.slug == PROFILE_SLUG)
    )).scalar_one()
    profile.body_md = "- uses an Android phone"
    await db.commit()

    brain = await ops.load_brain(db, user_id, "how is the ielts exam prep going")
    assert brain.profile == "- uses an Android phone"
    assert brain.current_context == ""
    assert [t for t, _ in brain.index] == ["Music", "IELTS"]     # section order
    assert all(d for _, d in brain.index), "the index carries descriptions"
    assert [t for t, _ in brain.relevant] == ["IELTS"]           # query-ranked
    # The three always-injected files are never ALSO in the index.
    assert "Profile" not in [t for t, _ in brain.index]


async def test_an_empty_file_is_not_advertised_in_the_index():
    """An index line for a file with nothing in it costs tokens every turn
    and can only mislead the model into opening it."""
    db, user_id = await _session()
    await _apply(db, user_id, [_create_ielts()])
    brain = await ops.load_brain(db, user_id, "ielts")
    assert brain.index == []
    assert brain.file_count == 4


# ── Retiring the shells the migration empties ─────────────────────────
#
# Round 8 stored FILES as well as rows. The v3 migration moves the rows into
# the new file set and leaves the old file records alone, so the generic
# catch-alls survive their own contents. Found in production on the founder's
# tenant right after the first real run (2026-08-20 23:41): you/profile had
# migrated correctly and the People section still listed an empty "User" —
# acceptance criterion #1 failing while every batch reported success.


async def _file(db, user_id, slug, section, title, body=""):
    row = MemoryFile(
        user_id=user_id, slug=slug, section=section, title=title,
        description=GOOD_DESC, body_md=body,
    )
    db.add(row)
    await db.commit()
    return row


async def test_the_empty_round_8_shells_are_retired():
    db, user_id = await _session()
    await ops.ensure_system_files(db, user_id)
    await db.commit()
    await _file(db, user_id, "people/user", "people", "User")
    await _file(db, user_id, "areas/work", "areas", "Work & goals")

    removed = await ops.prune_empty_files(db, user_id)
    await db.commit()

    assert sorted(removed) == ["areas/work", "people/user"]
    slugs = {f.slug for f in await ops._all_files(db, user_id)}
    assert "people/user" not in slugs
    assert "areas/work" not in slugs


async def test_a_file_with_content_is_never_touched():
    """Emptiness is the entire test — this must not be able to lose a fact."""
    db, user_id = await _session()
    await _file(db, user_id, "topics/music", "topics", "Music",
                body="- listens to Persian pop")

    assert await ops.prune_empty_files(db, user_id) == []
    slugs = {f.slug for f in await ops._all_files(db, user_id)}
    assert "topics/music" in slugs


async def test_the_three_system_files_are_exempt_while_still_empty():
    """They are created empty ON PURPOSE, before the writer has ever run; the
    curator and the rollover fill them later. Pruning them would delete the
    two files injected into every single reply."""
    db, user_id = await _session()
    await ops.ensure_system_files(db, user_id)
    await db.commit()

    assert await ops.prune_empty_files(db, user_id) == []
    slugs = {f.slug for f in await ops._all_files(db, user_id)}
    assert {PROFILE_SLUG, CURRENT_CONTEXT_SLUG, LEARNED_SLUG} <= slugs


async def test_the_prune_is_idempotent():
    db, user_id = await _session()
    await _file(db, user_id, "people/user", "people", "User")

    assert await ops.prune_empty_files(db, user_id) == ["people/user"]
    await db.commit()
    assert await ops.prune_empty_files(db, user_id) == []


async def test_whitespace_is_not_content():
    db, user_id = await _session()
    await _file(db, user_id, "knowledge", "areas", "Knowledge", body="\n  \n")

    assert await ops.prune_empty_files(db, user_id) == ["knowledge"]


# ── The fixed files' descriptions are canon ───────────────────────────


async def test_a_system_file_description_cannot_be_rewritten():
    """CI run 32430971208: "I switched to an Android phone last month, a Pixel
    9" produced applied=1, every body empty, and the one op was "Updated what
    Profile is for." Four scenarios failed that way in a single run. A turn has
    a small op budget, and a description rewrite is the one op that can spend
    it while recording nothing — so this is a capture bug wearing a drift bug's
    clothes. `SYSTEM_FILES` declares these three descriptions and
    `ensure_system_files` repairs them; the writer has no business restating
    them."""
    db, user_id = await _session()
    await ops.ensure_system_files(db, user_id)
    await db.commit()
    identity = await resolve_user_identity(db, user_id)
    files = await ops._all_files(db, user_id)

    plan = ops.validate_ops(
        [{"op": "update_description", "slug": PROFILE_SLUG,
          "description": GOOD_DESC}],
        files, identity=identity,
    )

    assert plan.accepted == []
    assert any("fixed" in c for c in plan.complaints), plan.complaints
    row = next(f for f in await ops._all_files(db, user_id)
               if f.slug == PROFILE_SLUG)
    assert row.description == SYSTEM_FILES[PROFILE_SLUG]["description"]


async def test_an_ordinary_file_description_is_still_editable():
    """The refusal is scoped to the three fixed files — every other file's
    description is generated and must stay regenerable."""
    db, user_id = await _session()
    await _file(db, user_id, "areas/ielts", "areas", "Ielts",
                body="- exam booked for Aug 30, 2026")
    identity = await resolve_user_identity(db, user_id)
    files = await ops._all_files(db, user_id)

    plan = ops.validate_ops(
        [{"op": "update_description", "slug": "areas/ielts",
          "description": GOOD_DESC}],
        files, identity=identity,
    )

    assert len(plan.accepted) == 1
    assert plan.accepted[0]["description"] == GOOD_DESC


async def test_the_stored_bullet_is_the_normalised_one():
    """Normalisation happens BEFORE the lint, so what is linted is what is
    stored — and a later `rewrite`, which must match CHARACTER FOR CHARACTER
    against the rendered body, is matching the same string."""
    db, user_id = await _session()
    await _file(db, user_id, "topics/music", "topics", "Music",
                body="- listens to Persian pop")
    identity = await resolve_user_identity(db, user_id)
    files = await ops._all_files(db, user_id)

    plan = ops.validate_ops(
        [{"op": "add", "slug": "topics/music",
          "bullet": "listens to Googoosh constantly.",
          "change": "Added Music: a favourite singer."}],
        files, identity=identity,
    )

    assert plan.accepted[0]["bullet"] == "listens to Googoosh constantly"


# ── A bare slug whose section the op already declares ─────────────────


async def test_a_bare_slug_is_repaired_from_the_declared_section():
    """CI run 32433614861. The writer finally proposed the person file P06 had
    been missing — as `majid-tajik`, without the `people/`. `create_file`
    carries `section` as its own field, so that is not ambiguous; it is
    `people/majid-tajik` with the prefix left off. Refusing it cost the whole
    batch: the following `add` and `link` named `areas/ielts`, whose own
    `create_file` was refused the same way, so all EIGHT ops died and the turn
    stored nothing at all."""
    db, user_id = await _session()
    identity = await resolve_user_identity(db, user_id)

    plan = ops.validate_ops(
        [{"op": "create_file", "section": "people", "slug": "majid-tajik",
          "title": "Majid Tajik", "description": GOOD_DESC}],
        await ops._all_files(db, user_id), identity=identity,
    )

    assert plan.accepted, plan.complaints
    assert plan.accepted[0]["slug"] == "people/majid-tajik"
    assert plan.accepted[0]["section"] == "people"


async def test_the_repair_unblocks_the_ops_that_named_the_full_slug():
    """The reason the repair is safe rather than a guess: the model's OWN
    later ops already use the namespaced form."""
    db, user_id = await _session()
    identity = await resolve_user_identity(db, user_id)

    plan = ops.validate_ops(
        [
            {"op": "create_file", "section": "areas", "slug": "ielts",
             "title": "Ielts", "description": GOOD_DESC},
            {"op": "add", "slug": "areas/ielts",
             "bullet": "exam booked for Aug 30, 2026",
             "change": "Added Ielts: the exam date."},
        ],
        await ops._all_files(db, user_id), identity=identity,
    )

    assert [o["op"] for o in plan.accepted] == ["create_file", "add"]
    assert plan.complaints == []


async def test_a_bare_slug_with_no_declared_section_is_still_refused():
    """The repair reads the op's own `section`; it does not invent one."""
    db, user_id = await _session()
    identity = await resolve_user_identity(db, user_id)

    plan = ops.validate_ops(
        [{"op": "create_file", "slug": "majid-tajik", "title": "Majid Tajik",
          "description": GOOD_DESC}],
        await ops._all_files(db, user_id), identity=identity,
    )

    assert plan.accepted == []
    assert any("no section" in c for c in plan.complaints), plan.complaints


async def test_a_bare_slug_cannot_be_repaired_into_a_system_section():
    """`you` and `learned` are not repairable prefixes. The section holds
    exactly two files, both system files, and repairing a bare slug into it
    would let the writer open a THIRD — a file injected into every reply that
    nothing in `SYSTEM_FILES` declares.

    Deliberately not `profile`: that slug would be caught by the already-exists
    check instead, so the test would pass while the prefix restriction did
    nothing. A mutation proved exactly that."""
    db, user_id = await _session()
    identity = await resolve_user_identity(db, user_id)

    plan = ops.validate_ops(
        [{"op": "create_file", "section": "you", "slug": "hobbies",
          "title": "Hobbies", "description": GOOD_DESC}],
        await ops._all_files(db, user_id), identity=identity,
    )

    assert plan.accepted == [], "a third you/ file was created"
    assert any("no section" in c for c in plan.complaints), plan.complaints


async def test_the_repair_still_refuses_the_owner_as_a_person():
    """A bare slug must not become a way around the self-person guard."""
    db, user_id = await _session("Nariman Hosseini")
    identity = await resolve_user_identity(db, user_id)

    plan = ops.validate_ops(
        [{"op": "create_file", "section": "people", "slug": "nariman-hosseini",
          "title": "Nariman Hosseini", "description": GOOD_DESC}],
        await ops._all_files(db, user_id), identity=identity,
    )

    assert plan.accepted == []
    assert any("whose memory this is" in c for c in plan.complaints), plan.complaints


async def test_nd8_a_match_that_echoes_the_bullet_marker_still_lands():
    """ND-8 (found in CI memverify P04, root-caused 2026-08-25): the model
    is SHOWN the file body, so it echoes the line as it reads there —
    "- lives in Toronto". Bullets are stored marker-less, `match` was only
    `.strip()`ed, so every such rewrite was refused with "no bullet reads
    exactly …" and the correction silently lost: the corpus still said
    Toronto after the user moved to Vancouver. A correction that never
    lands makes "one memory" dishonest.

    The tolerance must NOT gut the guard — an absent bullet is still
    refused, and a mid-sentence dash is not a marker.
    """
    db, user_id = await _session()
    await _apply(db, user_id, [
        _create_ielts(),
        {"op": "add", "slug": "areas/ielts", "bullet": "lives in Toronto",
         "change": "Added IELTS: city."},
    ])

    # The ND-8 case: `match` carries the bullet marker.
    plan, result = await _apply(db, user_id, [
        {"op": "rewrite", "slug": "areas/ielts",
         "match": "- lives in Toronto",
         "bullet": "lives in Vancouver",
         "change": "Updated: moved to Vancouver."},
    ])
    assert plan.complaints == [], plan.complaints
    assert result["applied"] == 1
    file = await ops.get_file(db, user_id, "areas/ielts")
    assert file["body_md"] == "- lives in Vancouver", file["body_md"]

    # An indented / asterisk marker is tolerated the same way (remove leg).
    await _apply(db, user_id, [
        {"op": "add", "slug": "areas/ielts", "bullet": "targeting band 7.5",
         "change": "Added band target."},
    ])
    plan, result = await _apply(db, user_id, [
        {"op": "remove", "slug": "areas/ielts", "match": "  * targeting band 7.5",
         "change": "Removed the band target."},
    ])
    assert plan.complaints == [], plan.complaints
    assert result["applied"] == 1

    # THE GUARD IS INTACT: a bullet that genuinely is not there is still
    # refused — with or without a marker — and never fuzzily matched.
    for absent in ("- lives in Montreal", "lives in Montreal"):
        plan, result = await _apply(db, user_id, [
            {"op": "rewrite", "slug": "areas/ielts", "match": absent,
             "bullet": "lives in Ottawa", "change": "x"},
        ])
        assert result["applied"] == 0, absent
        assert any("no bullet reads exactly" in c for c in plan.complaints)

    # A dash INSIDE a bullet is not a marker: the text is addressed whole.
    await _apply(db, user_id, [
        {"op": "add", "slug": "areas/ielts",
         "bullet": "reading - writing gap is the weak spot",
         "change": "Added: the gap."},
    ])
    plan, result = await _apply(db, user_id, [
        {"op": "rewrite", "slug": "areas/ielts",
         "match": "reading - writing gap is the weak spot",
         "bullet": "reading and writing are now even",
         "change": "Updated: gap closed."},
    ])
    assert plan.complaints == [], plan.complaints
    assert result["applied"] == 1
