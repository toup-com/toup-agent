"""The curator's instruct paths (rebuild-2026-08-v3 §2).

The LLM is faked — what is under test is everything AROUND the model call,
which is where round 8's write path failed: the prompt it is handed, the
scoping of an edit box to one file, the single retry with the validator's
own complaints, and the fact that nothing the model says is applied until
the deterministic engine agrees.

`curate_turn` (post-turn extraction) is exercised at the bottom: its
deterministic pre-gates run with no model at all, and the model call itself
is faked the same way.
"""

import json
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.base import Base
from app.db.models.memory import MemoryFile, MemoryFileChange
from app.db.models.user import User
from app.memory_files import PROFILE_SLUG
from app.services import memory_curator as curator
from app.services import memory_file_ops as ops
from app.services.user_identity import forget_cached_identity

MUSIC_OPS = json.dumps({"ops": [
    {"op": "create_file", "section": "topics", "slug": "topics/music",
     "title": "Music",
     "description": "Music taste — artists and albums; read when music comes up."},
    {"op": "add", "slug": "topics/music", "bullet": "likes Googoosh and Ebi",
     "change": "Added Music: likes Googoosh."},
]})


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Replays canned replies and records the prompts it was given."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.temperatures = []

    async def complete_with_json(self, messages, model=None, temperature=None, **kw):
        self.prompts.append(messages[0]["content"])
        self.temperatures.append(temperature)
        return _Resp(self.replies.pop(0))


@pytest.fixture
def fake_llm(monkeypatch):
    holder = {}

    def install(*replies):
        llm = _FakeLLM(*replies)
        holder["llm"] = llm
        monkeypatch.setattr(curator, "_llm", lambda api_key: llm)
        return llm

    return install


async def _session(name="Nariman Hosseini"):
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[User.__table__, MemoryFile.__table__, MemoryFileChange.__table__],
        )
    db = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()
    user_id = str(uuid.uuid4())
    db.add(User(id=user_id, email="nariman@toup.ai", hashed_password="x", name=name))
    await db.commit()
    forget_cached_identity()
    return db, user_id


# ── The prompt ────────────────────────────────────────────────────────

async def test_the_prompt_carries_the_identity_the_router_needs(fake_llm):
    """§2.2: the writer is told who the owner is, because a `people/*` file
    about the owner is the defect the resolver exists to prevent — and the
    model cannot avoid it if it does not know the name."""
    db, user_id = await _session()
    llm = fake_llm(MUSIC_OPS)
    await curator.instruct_global(db, user_id, "remember I like Googoosh")

    prompt = llm.prompts[0]
    assert "WHOSE MEMORY THIS IS: Nariman Hosseini" in prompt
    assert "NEVER in a people/ file" in prompt
    assert "Today is" in prompt
    # The index, so an existing file is reused rather than duplicated…
    assert "FILE INDEX:" in prompt and PROFILE_SLUG in prompt
    # …and the bodies, so the writer can tell "already known" from "new".
    assert "FILE BODIES:" in prompt
    # The op contract and the voice rules, verbatim.
    assert '"op":"rewrite"' in prompt
    assert '(never "the user")' in prompt
    assert "read when <trigger>" in prompt


async def test_an_unknown_identity_says_so_rather_than_inventing_one(fake_llm):
    """A fresh tenant is "Agent Owner" with an <hex>@agent.local email.
    Naming that as the owner would attach every fact to a person file
    called Agent Owner; the writer is told the identity is unknown and
    fails OPEN instead."""
    db, user_id = await _session(name="Agent Owner")
    from sqlalchemy import update
    from app.db.models.user import User as _U
    await db.execute(update(_U).where(_U.id == user_id).values(
        email=f"{user_id[:8]}@agent.local"))
    await db.commit()
    from app.services.user_identity import forget_cached_identity as _f
    _f()
    llm = fake_llm(MUSIC_OPS)
    await curator.instruct_global(db, user_id, "remember I like Googoosh")
    assert "WHOSE MEMORY THIS IS: unknown" in llm.prompts[0]


async def test_the_file_box_names_the_file_it_is_editing(fake_llm):
    db, user_id = await _session()
    fake_llm(MUSIC_OPS)
    await curator.instruct_global(db, user_id, "remember I like Googoosh")
    llm = fake_llm(json.dumps({"ops": []}))
    await curator.instruct_file(db, user_id, "topics/music", "drop the Ebi bit")
    assert 'editing the file "topics/music"' in llm.prompts[0]
    assert "touching NO other file" in llm.prompts[0]


# ── Applying ─────────────────────────────────────────────────────────

async def test_a_global_instruction_creates_and_fills_a_file(fake_llm):
    db, user_id = await _session()
    fake_llm(f"```json\n{MUSIC_OPS}\n```")   # fenced — the model does this
    result = await curator.instruct_global(db, user_id, "remember I like Googoosh")

    assert result == {
        "applied": 2, "rejected": [], "changed_files": ["topics/music"],
        # The note names the TITLE. A slug is an internal id and no client
        # renders one anywhere else; this sentence is the only place one
        # ever reached a user.
        "changed_titles": ["Music"],
        "note": "Saved 2 changes in Music.",
    }
    file = await ops.get_file(db, user_id, "topics/music")
    assert file["body_md"] == "- likes Googoosh and Ebi"
    log = await ops.read_log(
        db, user_id, __import__("datetime").datetime.utcnow().strftime("%Y-%m")
    )
    assert [e["summary"] for e in log["days"][0]["entries"]] == [
        "Added Music: likes Googoosh.", "Created Music.",
    ]


async def test_an_op_naming_another_file_is_dropped_AND_reported(fake_llm):
    """The scoping itself is the safety property — "remove the IELTS dates"
    typed on the Music page must not edit IELTS. Reporting it is the other
    half: a silent drop is indistinguishable from a change that landed."""
    db, user_id = await _session()
    fake_llm(MUSIC_OPS)
    await curator.instruct_global(db, user_id, "remember I like Googoosh")

    fake_llm(json.dumps({"ops": [
        {"op": "remove", "slug": "topics/music", "match": "likes Googoosh and Ebi",
         "change": "Removed Googoosh from Music."},
        {"op": "add", "slug": PROFILE_SLUG, "bullet": "lives in Toronto now",
         "change": "Added Profile: Toronto."},
    ]}))
    result = await curator.instruct_file(
        db, user_id, "topics/music", "forget the googoosh thing",
    )
    assert result["applied"] == 1
    assert any("named another file" in r for r in result["rejected"])
    assert (await ops.get_file(db, user_id, PROFILE_SLUG))["body_md"] == ""


async def test_a_wholly_rejected_proposal_retries_once_with_the_complaints(fake_llm):
    """Round 8's pattern, kept: the validator's complaints are precise
    enough to fix ("no bullet reads exactly X"), and a second full rejection
    is a real answer rather than a transient."""
    db, user_id = await _session()
    fake_llm(MUSIC_OPS)
    await curator.instruct_global(db, user_id, "remember I like Googoosh")

    llm = fake_llm(
        json.dumps({"ops": [{"op": "add", "slug": "topics/music",
                             "bullet": "You like Ebi", "change": "x"}]}),
        json.dumps({"ops": [{"op": "add", "slug": "topics/music",
                             "bullet": "listens to Ebi on repeat",
                             "change": "Added Music: Ebi on repeat."}]}),
    )
    result = await curator.instruct_file(db, user_id, "topics/music", "I like Ebi")
    assert len(llm.prompts) == 2
    assert "subject is implied" in llm.prompts[1]
    assert result["applied"] == 1
    assert "- listens to Ebi on repeat" in (
        await ops.get_file(db, user_id, "topics/music")
    )["body_md"]


async def test_a_proposal_rejected_twice_changes_nothing_and_says_why(fake_llm):
    db, user_id = await _session()
    bad = json.dumps({"ops": [{"op": "add", "slug": "topics/ghost",
                               "bullet": "something true here", "change": "x"}]})
    fake_llm(bad, bad)
    result = await curator.instruct_global(db, user_id, "remember something")
    assert result["applied"] == 0
    assert result["note"] == "Nothing changed — that didn't pass the memory rules."
    assert any("create it first" in r for r in result["rejected"])


async def test_nothing_to_change_is_a_clean_answer(fake_llm):
    db, user_id = await _session()
    fake_llm(json.dumps({"ops": []}))
    result = await curator.instruct_global(db, user_id, "nothing here")
    assert result == {
        "applied": 0, "rejected": [], "changed_files": [], "changed_titles": [],
        "note": "Nothing to change.",
    }


async def test_an_unknown_file_is_a_valueerror_not_a_new_file(fake_llm):
    """The route turns this into a 404. Creating the file instead would let
    a typo'd slug mint an empty file with no description."""
    db, user_id = await _session()
    fake_llm(json.dumps({"ops": []}))
    with pytest.raises(ValueError):
        await curator.instruct_file(db, user_id, "topics/ghost", "change something")


async def test_system_files_exist_before_the_first_instruction(fake_llm):
    db, user_id = await _session()
    fake_llm(json.dumps({"ops": []}))
    await curator.instruct_global(db, user_id, "anything")
    listing = await ops.list_files(db, user_id)
    slugs = {f["slug"] for s in listing["sections"] for f in s["files"]}
    assert slugs == {"you/profile", "you/current-context", "learned"}


async def test_the_writer_is_never_sampled(fake_llm):
    """A memory decision must not be a coin flip.

    Inherited from `test_memory_gate_cross_script`'s pin on the three dedup
    adjudication call sites, which retired with that service. There is one
    decision point now, so there is one place to pin.
    """
    db, user_id = await _session()
    llm = fake_llm(MUSIC_OPS)
    await curator.instruct_global(db, user_id, "remember I like Googoosh")
    assert llm.temperatures == [0.0]


async def test_the_confirmation_names_the_TITLE_never_the_slug(fake_llm):
    """A slug is an internal id. No client renders one — the file page shows
    the title, the log shows the title, the index shows the title. The note
    under the instruct box was the ONE place a user ever saw
    "people/majid-tajik", and it is the sentence they read right after
    typing.
    """
    db, user_id = await _session()
    fake_llm(MUSIC_OPS)
    result = await curator.instruct_global(db, user_id, "I like Googoosh")

    assert result["note"] == "Saved 2 changes in Music."
    assert "topics/" not in result["note"] and "/" not in result["note"]
    # The slug is still in the payload — clients navigate by it.
    assert result["changed_files"] == ["topics/music"]
    assert result["changed_titles"] == ["Music"]


async def test_the_ops_contract_shows_a_description_the_VALIDATOR_accepts(fake_llm):
    """The shape line and the worked example printed an ASCII hyphen while
    `DESCRIPTION_RE` demands U+2014, so a model copying the example got
    rejected on EVERY description it ever wrote — and the retry showed it
    the same broken example again.

    Asserted by running the example through the real validator, not by
    grepping for a character: that is the only version of this test that
    cannot pass while the contract is still wrong.
    """
    import re as _re

    from app.memory_files import description_problem

    # Only the CONCRETE example, not the `<placeholder>` shape line — the
    # shape is asserted literally below.
    examples = [
        e for e in _re.findall(r'"([^"]*read when[^"]*)"', curator.OPS_CONTRACT)
        if "<" not in e and len(e) > 30
    ]
    assert examples, "the ops contract no longer shows a description example"
    for example in examples:
        assert description_problem(example) is None, (
            f"the contract shows a description the validator refuses: "
            f"{example!r} -> {description_problem(example)}"
        )
    # And the SHAPE line, which is what a model actually pattern-matches.
    assert "<what this is> — <scope>; read when <trigger>." in curator.OPS_CONTRACT
    assert "use an em dash" not in curator.OPS_CONTRACT, (
        "the parenthetical workaround is redundant once the example is right, "
        "and a rule stated twice in two forms is a rule that will drift"
    )


# ── The turn curator (§2.2) ───────────────────────────────────────────

async def test_the_turn_prompt_labels_its_two_slots_asymmetrically(fake_llm):
    """Root cause #1, in the prompt half.

    `user_text` is the only source of facts and the assistant reply is
    context; the model has to be TOLD that, because nothing in the two
    strings distinguishes them. The structural half of the fix (handing it
    `display_user_message` rather than the rewritten `user_message`) is
    pinned in test_curator_producers.py.
    """
    db, user_id = await _session()
    llm = fake_llm(MUSIC_OPS)
    await curator.curate_turn(
        db, user_id,
        user_text="I have been listening to Googoosh all week",
        assistant_text="Great — I found three albums for you.",
    )
    prompt = llm.prompts[0]
    assert "WHAT THE USER SAID (the ONLY source of facts)" in prompt
    assert "CONTEXT ONLY — never a source of facts" in prompt
    assert "Facts may come only from what the USER said or confirmed" in prompt
    # The durability rules and the ops contract both ride the turn prompt.
    assert "WHAT IS DURABLE" in prompt
    assert "BULLET VOICE" in prompt
    assert "Today is" in prompt


async def test_the_durability_rules_name_the_dispatch_s_own_bad_memories():
    """Every class in §2 of the dispatch is a NAMED rule, not a vibe.

    A prompt that says "only store durable facts" is what round 8 shipped;
    what it stored was a scraped YouTube title, a two-minute reminder and a
    Gmail-briefing prompt. The rules have to name the actual shapes.
    """
    rules = curator.TURN_DURABILITY_RULES
    for needle in (
        "One-off requests",          # "play X"
        "wake me 1 minute later",    # the snooze
        "reminder to go to soccer",  # the two-minute reminder
        "currently hungry",          # a transient state
        "playback OUTCOMES",         # what played / what a tool returned
        "You have Gmail messages",   # a tool result read as a fact
        "scheduled job, routine or trigger",
        "daily Gmail briefing at 11:49 AM",   # the ONE allowed standing line
        "UUIDs",
        "Advice, tips or explanations",       # round 8.5's guard
        "should\navoid shellfish",            # …with its own example
        "Hypotheticals",                      # round 8.5's other guard
        "World knowledge",
    ):
        assert needle.replace("\n", " ") in " ".join(rules.split()), needle


@pytest.mark.parametrize("text,reason", [
    ("ok thanks", "too_short"),
    ("what is the weather in Toronto today?", "question_only"),
    ("play the new Googoosh album for me", "one_off_request"),
    ("wake me up 1 minute later", "one_off_request"),
    ("remind me to go to soccer in 2 minutes", "one_off_request"),
    ("my visa card is 4111 1111 1111 1111 remember it", "sensitive_card_number"),
])
def test_the_pre_gate_skips_a_turn_with_nothing_to_remember(text, reason):
    assert curator.turn_skip_reason(text) == reason


@pytest.mark.parametrize("text", [
    "I'm allergic to peanuts, what should I eat?",
    "my IELTS exam is booked for Aug 30 2026",
    "remind me that my sister's birthday is March 3",
    "I moved to Toronto in 2019 for the UofT program",
    "می‌خواهم هر روز صبح خلاصه ایمیل‌ها را بگیرم",
])
def test_the_pre_gate_never_eats_a_turn_that_states_something(text):
    """The cheap gate is the cheapest place a real fact can be lost.

    Note the third case: "remind me that …" matches the one-off regex AND
    carries a durable fact. The possessive-copula rule is what keeps it.
    """
    assert curator.turn_skip_reason(text) is None


async def test_a_trivial_turn_costs_no_model_call(fake_llm):
    db, user_id = await _session()
    llm = fake_llm(MUSIC_OPS)
    result = await curator.curate_turn(
        db, user_id, user_text="thanks!", query_was_trivial=True,
    )
    assert result == {
        "applied": 0, "rejected": [], "changed_files": [], "skipped": "trivial",
    }
    assert llm.prompts == []


async def test_the_pre_gate_runs_before_the_model(fake_llm):
    db, user_id = await _session()
    llm = fake_llm(MUSIC_OPS)
    result = await curator.curate_turn(
        db, user_id, user_text="play me some quiet jazz please",
    )
    assert result["skipped"] == "one_off_request"
    assert result["applied"] == 0
    assert llm.prompts == []


async def test_a_turn_that_carries_a_fact_writes_the_file_and_the_change_line(fake_llm):
    db, user_id = await _session()
    fake_llm(MUSIC_OPS)
    result = await curator.curate_turn(
        db, user_id,
        user_text="I have been listening to Googoosh and Ebi all week",
        assistant_text="Noted.",
    )
    assert result["skipped"] is None
    assert result["applied"] == 2
    file = await ops.get_file(db, user_id, "topics/music")
    assert "likes Googoosh and Ebi" in file["body_md"]
    log = await ops.read_log(db, user_id, __import__("datetime").datetime.utcnow().strftime("%Y-%m"))
    summaries = [e["summary"] for d in log["days"] for e in d["entries"]]
    assert "Added Music: likes Googoosh." in summaries


async def test_empty_ops_is_the_ordinary_answer(fake_llm):
    db, user_id = await _session()
    fake_llm(json.dumps({"ops": []}))
    result = await curator.curate_turn(
        db, user_id, user_text="I was thinking about the weather again today",
    )
    assert result["applied"] == 0 and result["skipped"] is None
    assert result["changed_files"] == []


async def test_a_model_failure_propagates_so_the_caller_can_park_the_turn(monkeypatch):
    """The outbox is the only thing between a transient blip and losing
    everything the user said that turn — so a real failure must RAISE, not
    be swallowed into `applied: 0`, which is indistinguishable from
    "nothing was worth storing"."""
    db, user_id = await _session()

    class _Dead:
        async def complete_with_json(self, **kw):
            raise RuntimeError("provider down")

    monkeypatch.setattr(curator, "_llm", lambda api_key: _Dead())
    monkeypatch.setattr(curator, "EXTRACTION_RETRY_BACKOFF_S", 0)
    with pytest.raises(RuntimeError):
        await curator.curate_turn(
            db, user_id, user_text="I moved to Toronto in 2019 for UofT",
        )


async def test_one_transient_blip_is_retried_before_it_becomes_a_failure(monkeypatch):
    """A6-2, moved here with the model call it protected."""
    db, user_id = await _session()
    calls = {"n": 0}

    class _Flaky:
        async def complete_with_json(self, messages, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("blip")
            return _Resp(MUSIC_OPS)

    monkeypatch.setattr(curator, "_llm", lambda api_key: _Flaky())
    monkeypatch.setattr(curator, "EXTRACTION_RETRY_BACKOFF_S", 0)
    result = await curator.curate_turn(
        db, user_id, user_text="I have been listening to Googoosh all week",
    )
    assert calls["n"] == 2
    assert result["applied"] == 2


async def test_today_is_resolved_in_the_users_own_timezone(fake_llm):
    """A bullet says "Aug 30, 2026" forever. At 23:40 in Toronto a UTC clock
    is already tomorrow, so the writer would resolve "tomorrow" to the wrong
    absolute date — permanently, in the voice the product presents as fact."""
    import datetime as _dt

    db, user_id = await _session()
    from sqlalchemy import update as _update

    from app.db.models.user import User as _User
    await db.execute(
        _update(_User).where(_User.id == user_id).values(timezone="Pacific/Kiritimati")
    )
    await db.commit()

    llm = fake_llm(json.dumps({"ops": []}))
    await curator.curate_turn(
        db, user_id, user_text="I moved to Toronto in 2019 for the UofT program",
    )
    # UTC+14 — the one zone that is reliably a different calendar day from
    # UTC for most of the day.
    from zoneinfo import ZoneInfo
    local = _dt.datetime.now(ZoneInfo("Pacific/Kiritimati")).strftime("%A, %B %d, %Y")
    assert f"Today is {local}" in llm.prompts[0]
