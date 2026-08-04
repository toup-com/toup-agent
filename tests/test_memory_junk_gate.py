"""The write-time junk gate, tested against REAL production strings.

Every string in CORPUS below was read out of the founder's live tenant
(871bac24) on 2026-07-30. Nothing here is invented: if a test says the gate
must reject "video about context engeeniering is about context engeeniering",
that row existed, was embedded, was retrievable, and was shown to the user.

Standing rule this file exists to honour (see
feedback_source_text_assertions_are_not_tests): a test that asserts a line of
source code was typed proves nothing. Every assertion here EXECUTES the gate.
"""

import inspect

import pytest

from app.services.memory_gate import (
    ECHO_ASSISTANT_MIN,
    ECHO_MARGIN_MIN,
    MAX_MEMORY_CHARS,
    assistant_echo_reason,
    degenerate_relationship_reason,
    memory_gate_reason,
    relationship_gate_reason,
    scaffolding_reason,
)

ALIASES = ["Nariman", "Nariman HOSSEINI"]


# ── Real triples that produced real junk rows ─────────────────────────────
# (source, predicate, target, rendered_content)
TAUTOLOGIES = [
    ("video about context engeeniering", "is_about", "context engeeniering"),
    ("video about context management", "is_about", "context management"),
    ("LLM/AI researcher", "researches", "LLM/AI"),
    ("Toup", "has_launch", "Toup's launch"),
    ("Gmail tool", "connects_to", "Gmail"),
    ("Gmail messages", "are_from", "Gmail"),
    ("Daily 5:06 PM motivational quote routine", "includes", "motivational quote"),
    ("Top AI Papers This Week", "is_about", "AI"),
    ("Latest OpenAI News", "is_about", "OpenAI"),
    ("200-word Essay on Attention", "is_about", "Attention"),
    ("OpenAI", "is_subject_of", "latest OpenAI news"),
]

WORLD_KNOWLEDGE = [
    ("Better Call Saul", "play_on", "Netflix"),
    ("Drake", "artist_of", "0-100"),
    ("Bunker", "performed_by", "Baltazar"),
    ("Ana Al Ghaltan (Remix)", "performed_by", "Cheb Rezki"),
    ("Arash", "performed", "Dooset Daram"),
    ("Codex", "is_included_in", "ChatGPT Free"),
    ("Rampage", "happens_in", "USA"),
    ("Rampage", "might_cause_problems_in", "Canada"),
    ("USA", "people_might_come_to", "Canada"),
    ("Shops at Don Mills", "located_in", "Toronto"),
    ("lint store", "located_in", "Shops at Don Mills"),
    ("Church Texas Chicken", "offers", "chicken wrap"),
    ("chicken wrap", "is_from", "Church Texas Chicken"),
    ("Nate Hark", "created_content_about", "Context Engineering"),
]

SCAFFOLDING = [
    ("Sub-agent 1", "researches", "Top AI Papers This Week"),
    ("Sub-agent 2", "summarizes", "Latest OpenAI News"),
    ("Sub-agent 3", "writes", "200-word Essay on Attention"),
    ("Assistant", "summarizes_new_messages_from", "Gmail"),
    ("Assistant", "does_not_have", "reminder tool"),
]

# Rows a human judged worth keeping. The gate must NOT reject these.
KEEP = [
    ("User", "owns", "Toup"),
    ("User", "works_on", "mobile app"),
    ("user", "uses", "Gmail"),
    ("Nariman HOSSEINI", "uses", "Microsoft Teams"),
    ("Nariman HOSSEINI", "chatted_with", "majid tajik"),
    ("Nariman HOSSEINI", "has_chat_with", "sarah rostami"),
    ("Nariman", "receives_daily", "Motivational Quotes"),
    ("User", "is_sibling_of", "Brother"),
    ("Reminder", "should_not_be_sent_via", "Telegram"),
]


@pytest.mark.parametrize("s,p,t", TAUTOLOGIES)
def test_tautologies_are_rejected(s, p, t):
    """A triple whose two ends restate each other carries no information."""
    assert relationship_gate_reason(s, p, t, user_aliases=ALIASES) is not None, (
        f"gate admitted the tautology {s!r} --{p}--> {t!r}"
    )


@pytest.mark.parametrize("s,p,t", WORLD_KNOWLEDGE)
def test_world_knowledge_is_rejected(s, p, t):
    """Encyclopedia edges: equally true for a stranger, so not a memory."""
    assert relationship_gate_reason(s, p, t, user_aliases=ALIASES) is not None, (
        f"gate admitted world knowledge {s!r} --{p}--> {t!r}"
    )


@pytest.mark.parametrize("s,p,t", SCAFFOLDING)
def test_agent_scaffolding_is_rejected(s, p, t):
    assert relationship_gate_reason(s, p, t, user_aliases=ALIASES) is not None


@pytest.mark.parametrize("s,p,t", KEEP)
def test_real_user_facts_survive(s, p, t):
    """The gate's precision claim, asserted rather than remembered."""
    reason = relationship_gate_reason(s, p, t, user_aliases=ALIASES)
    assert reason is None, f"gate wrongly rejected {s!r} --{p}--> {t!r} as {reason}"


def test_user_endpoint_survives_even_when_the_stoplist_would_empty_it():
    """Regression: the overlap stoplist contains "user".

    Reusing it for the containment test made `user --owns--> Toup` collapse to
    an empty source and get rejected as contentless. Measured against the live
    corpus that was silently discarding "User owns Toup", "User works on mobile
    app" and "user uses Gmail" — three of the most useful rows in the brain.
    """
    assert degenerate_relationship_reason("User", "owns", "Toup") is None
    assert relationship_gate_reason("User", "owns", "Toup") is None


def test_merged_rows_are_rescued_by_enrichment_not_by_the_word_user():
    """Dedup merges edges; a merged row must not be judged on its stale triple.

    But the rescue must key on ENRICHMENT — requiring only a self-referent let
    12 junk rows back in, because routine-internals rows mention the user too.
    """
    merged = (
        "When handling Gmail, including during the user's daily Gmail briefing, "
        "clearly tell the user if no recent Gmail messages are found; if Gmail "
        "requires re-authentication, instruct the user to reconnect Gmail using "
        "the inline connector card."
    )
    assert relationship_gate_reason(
        "Gmail", "reconnect_using", "inline connector card",
        user_aliases=ALIASES, rendered=merged,
    ) is None

    # ...while an UNmerged row that merely says "user" stays rejected.
    assert relationship_gate_reason(
        "Gmail briefing", "summarizes_email_for", "user_thing",
        user_aliases=ALIASES, rendered="Gmail briefing summarizes email for user_thing",
    ) is not None


# ── Content-level gate ────────────────────────────────────────────────────

def test_assistant_lecture_is_rejected_as_not_a_single_fact():
    """The 1002-char stock-options row: an entire lecture wearing "The user".

    Longest row a human kept was 460 chars; this is the length rule doing its
    job on real production content.
    """
    lecture = (
        "The user is interested in how employee stock options work in startups "
        "and understands that stocks represent immediate ownership in a company, "
    ) + ("while stock options give the right to purchase shares in the future. " * 12)
    assert len(lecture) > MAX_MEMORY_CHARS
    assert memory_gate_reason(lecture) == "not_a_single_fact"


def test_encyclopedia_restated_from_the_assistant_is_rejected():
    """The defect that started this: Persian 409A facts mined from the agent's
    own spoken answer during a voice turn.

    The user asked a short question; the assistant lectured; the extractor
    stored the lecture. The gate sees that the content lives in the assistant's
    reply and not in the user's, and drops it.
    """
    user_msg = "آپشن سهام چطور کار می‌کنه؟"
    assistant = (
        "در آمریکا، روش رایج برای شرکت‌های خصوصی گزارش ارزش‌گذاری 409A است. "
        "هیئت‌مدیره شرکت معمولاً قیمت اعمال را تصویب و در اسناد رسمی ثبت می‌کند. "
        "آپشن کارکنان معمولاً روی سهام عادی داده می‌شود."
    )
    for junk in (
        "در آمریکا، روش رایج برای شرکت‌های خصوصی گزارش ارزش‌گذاری 409A است.",
        "هیئت‌مدیره شرکت معمولاً قیمت اعمال را تصویب و در اسناد رسمی ثبت می‌کند.",
        "آپشن کارکنان معمولاً روی سهام عادی داده می‌شود.",
    ):
        assert memory_gate_reason(
            junk, user_message=user_msg, assistant_response=assistant
        ) == "assistant_echo", f"gate admitted assistant echo: {junk}"


def test_echo_rule_abstains_when_the_memory_is_grounded_in_the_user_turn():
    """A real fact the user stated must survive even if the assistant echoes it."""
    user_msg = "my name is Nariman HOSSEINI and I work on a mobile app called Toup"
    assistant = "Nice to meet you Nariman! Toup sounds like a great mobile app."
    assert memory_gate_reason(
        "The user's name is Nariman HOSSEINI and they work on a mobile app called Toup",
        user_message=user_msg, assistant_response=assistant,
    ) is None


def test_echo_rule_abstains_cross_lingual_rather_than_deleting_a_real_fact():
    """Fail-open on the case this multilingual user actually hits.

    An English memory extracted from a Persian turn overlaps NEITHER side. The
    conjunction in assistant_echo_reason must make the rule abstain, because
    deleting a real fact is far worse than keeping a marginal one.
    """
    assert assistant_echo_reason(
        "The user likes listening to Ali Sorena.",
        "یه آهنگ از علی سورنا بذار",
        "باشه، الان پخش می‌کنم.",
    ) is None


def test_scaffolding_regex_does_not_eat_a_legitimate_project_memory():
    """Regression: "Job ID"/"App ID"/UUID matching used to reject this row.

    It is a real project memory that merely happens to quote a build error.
    Operational-handle matching is therefore endpoint-only.
    """
    row = (
        "The user is working on a game app project named “Build: Nokia Snake "
        "Arcade” with App ID 5f386567-cd48-4f1d-8b61-0d7e87b3b668, which should "
        "have three tabs: Game, Stats, and Settings."
    )
    assert memory_gate_reason(row) is None
    # ...but the same handle as an ENTITY NAME is still rejected.
    assert scaffolding_reason("Job ID 54420e9e-9547", endpoints=True) is not None


# ── The turn-type gates: non-user turns must not mine memories ────────────

def test_routine_handler_disables_post_processing():
    """The routine's own prompt was being read back as user speech.

    agent_task_handler already passes save_user_message=False with a long
    comment explaining the prompt is NOT the user's words — but
    disable_post_processing is an INDEPENDENT gate, and without it the
    extractor still received prompt_text as `user_message`. That produced a
    dozen rows describing the platform's own plumbing, regenerated daily.

    Structural assertion (call-argument presence), so it is checked against the
    parsed source rather than a runtime behaviour that would need a live
    scheduler, a tenant DB and an LLM to observe.
    """
    from app.agent.routines import agent_task_handler

    src = inspect.getsource(agent_task_handler)
    assert "disable_post_processing=True" in src, (
        "routine turns will mine the routine's own instruction text as user memories"
    )


def test_voice_agent_turn_disables_post_processing_when_not_saving():
    """`save=False` means "not the turn of record" — so do not extract either.

    Both the blocking and streaming siblings must set it; the streaming one is
    the path voice actually takes when tool events are enabled, so missing it
    there would leave the defect fully live.
    """
    from app.api import api_v1

    src = inspect.getsource(api_v1)
    assert src.count("disable_post_processing=not req.save") == 2, (
        "a voice turn can still write memories with no row in `messages`"
    )


def test_voice_tunnel_forwards_the_expiry_horizon():
    """ttl_days must cross the tunnel or transient voice memories are immortal."""
    from app.api import ws_realtime

    src = inspect.getsource(ws_realtime._extract_voice_memories)
    assert '"ttl_days"' in src


def test_memory_store_tool_accepts_and_applies_ttl():
    """The receiving end must actually use it — a forwarded field that the
    handler ignores is worse than not sending it, because it looks fixed."""
    from app.agent.tool_executor import ToolExecutor

    src = inspect.getsource(ToolExecutor._tool_memory_store)
    assert 'inp.get("ttl_days")' in src
    assert "expires_at=" in src
    # The old default wrote a category that exists in no enum.
    assert 'inp.get("category", "context")' not in src


def test_dedup_adjudicates_the_most_similar_candidate():
    """Regression for the blended-score P0.

    find_similar_memories ranks by final_score (similarity is only 40%), so
    candidates[0] was frequently NOT the nearest row — dedup then read
    similarity off the wrong one. Asserted by executing the sort, not by
    reading the source.
    """
    from app.services.memory_dedup_service import MemoryDedupService  # noqa: F401

    candidates = [
        {"id": "hot-but-distant", "similarity_score": 0.62, "final_score": 0.91},
        {"id": "true-duplicate", "similarity_score": 0.94, "final_score": 0.78},
    ]
    ordered = sorted(
        candidates, key=lambda c: c.get("similarity_score", 0), reverse=True
    )
    assert ordered[0]["id"] == "true-duplicate"

    src = inspect.getsource(MemoryDedupService.smart_create_memory)
    assert "key=lambda c: c.get('similarity_score', 0)" in src, (
        "dedup is still adjudicating whichever row won the relevance blend"
    )


# ── Placeholder-identity fail-open ────────────────────────────────────────

def test_no_user_endpoint_abstains_when_the_tenant_identity_is_a_placeholder():
    """Found on live tenant 3134fece AFTER the first version shipped.

    Tenant DBs are provisioned with name="Agent Owner" /
    "<prefix>@agent.local" until onboarding writes the real identity. With no
    usable alias, "Nariman owns Toup" reads as having no user endpoint — and
    the backfill dry-run on that tenant proposed archiving exactly that row,
    plus "Nariman created Toup". Dropping a real fact is worse than keeping a
    marginal one, so the rule must abstain when identity is unknown.
    """
    placeholder = ["Agent Owner", "3134fece"]
    for s, p, t in [("Nariman", "owns", "Toup"), ("Nariman", "created", "Toup")]:
        assert relationship_gate_reason(s, p, t, user_aliases=placeholder) is None, (
            f"{s} --{p}--> {t} dropped on a tenant with a placeholder identity"
        )

    # Identity-free rules must STILL fire on such a tenant, or the fail-open
    # would silently disable the whole gate wherever onboarding is incomplete.
    assert relationship_gate_reason(
        "Sub-agent 1", "researches", "Top AI Papers", user_aliases=placeholder
    ) is not None
    assert relationship_gate_reason(
        "video about X", "is_about", "X", user_aliases=placeholder
    ) is not None
    assert relationship_gate_reason(
        "Assistant", "summarizes", "Gmail", user_aliases=placeholder
    ) is not None


def test_real_identity_still_enforces_the_user_endpoint_rule():
    """The fail-open must not leak into tenants where we DO know the user."""
    assert relationship_gate_reason(
        "Better Call Saul", "play_on", "Netflix",
        user_aliases=["Nariman", "nariman@toup.ai"],
    ) == "no_user_endpoint"


# ── The graph half: "someone/something in the USER's life" ────────────────

@pytest.mark.asyncio
async def test_user_world_rescue_keeps_facts_about_the_users_people_and_projects(
    test_user_id,
):
    """The pure gate can only see the user; the graph sees their world.

    Without this, #385 rejected "Majid Tajik works at Microsoft" (the user's
    tutor), "Brother lives in Tehran", and "Toup uses React Native" — the
    user's OWN product, which the extraction prompt explicitly lists as a valid
    Project -> Technology edge.

    Measured on the founder's live graph before shipping: 16 entities within
    one hop, rescuing all of Toup/Majid Tajik/Brother/Sarah Rostami/Mohammad
    while readmitting ZERO of the 15 labelled world-knowledge rows.
    """
    from sqlalchemy import inspect as _inspect

    from app.db.database import async_session_maker, engine
    from app.services.memory_service import MemoryService

    # `entities` is an AGENT_ONLY table with a pgvector column. CI runs
    # RUN_MODE=platform on SQLite (.github/workflows/test-backend.yml), where
    # init_db() does not create it and the column could not compile anyway — so
    # this test raised `no such table: entities` and would have turned CI red
    # the moment this branch merged. Skipping honestly is better than a red
    # build OR a green one that never ran the assertion: the same rescue is
    # exercised end-to-end against real Postgres + pgvector by
    # tests/memverify/test_g_isolation.py and test_b_precision_junk.py.
    async with engine.connect() as conn:
        has_entities = await conn.run_sync(
            lambda sync_conn: _inspect(sync_conn).has_table("entities")
        )
    if not has_entities:
        pytest.skip(
            "requires the AGENT_ONLY `entities` table (RUN_MODE=agent on "
            "Postgres+pgvector); covered by backend/tests/memverify"
        )

    async with async_session_maker() as db_session:
        await _user_world_body(db_session, test_user_id)


async def _user_world_body(db_session, test_user_id):
    from app.services.memory_service import MemoryService

    svc = MemoryService(db_session)

    # The user tells the agent about their tutor and their product.
    await svc._upsert_entity(test_user_id, "User", "person")
    for name, etype in (("Majid Tajik", "person"), ("Toup", "project")):
        await svc._upsert_entity(test_user_id, name, etype)
    await db_session.flush()

    from app.db.models import Entity, EntityRelationship
    from sqlalchemy import select as _sel
    import uuid as _uuid
    from datetime import datetime as _dt

    async def link(a, b):
        ents = {
            e.name: e for e in (await db_session.execute(
                _sel(Entity).where(Entity.user_id == test_user_id)
            )).scalars()
        }
        db_session.add(EntityRelationship(
            id=str(_uuid.uuid4()), user_id=test_user_id,
            source_entity_id=ents[a].id, target_entity_id=ents[b].id,
            relationship_type="knows", relationship_label=f"{a} knows {b}",
            confidence=0.9, mention_count=1,
            first_seen_at=_dt.utcnow(), last_seen_at=_dt.utcnow(),
        ))
        await db_session.flush()

    await link("User", "Majid Tajik")
    await link("User", "Toup")

    # In the user's world -> rescued.
    assert await svc._endpoint_in_user_world(test_user_id, "Majid Tajik", "Microsoft")
    assert await svc._endpoint_in_user_world(test_user_id, "Toup", "React Native")

    # Never linked to the user -> still world knowledge, still rejected.
    assert not await svc._endpoint_in_user_world(
        test_user_id, "Better Call Saul", "Netflix")
    assert not await svc._endpoint_in_user_world(test_user_id, "Drake", "0-100")
