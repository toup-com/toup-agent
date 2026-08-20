"""The unrequested x.pdf (founder, iOS, 2026-08-18) — layer (a): the loadout.

User: "Can you search for me the latest ai news and give me the 5 most
important one in this mounth". The agent's FIRST action, before any
search, was `generate_pdf({"filename": "x", "content": []})` — a 952-byte
empty PDF attached to the reply. Prod log (toup-agent-871bac24):

    [PERF] query_intent: 0.8ms → category=web, tools=14
    [PERF] stable_tools: wire=143 allowed=19 intent=web
    [AGENT] Tool called: generate_pdf({"filename": "x", "content": []})
    [AGENT] Tool called: create_job({"title": "August AI news roundup", …})
    [AGENT] Tool called: web_search(…)

Two facts made the first line possible, and both are pinned here:

1. `create_job` was in NO intent's allowed set, while the platform-
   knowledge decision rules say "research … → call `create_job` FIRST".
   Under the prefix-stable layout that gap is enforced as an OpenAI
   `tool_choice: allowed_tools` restriction over the FULL wire array, so
   the model — told to open with create_job, forbidden to name it — had
   its call snapped onto the next allowed name in wire order:
   create_job → update_job → save_streaming_credential → **generate_pdf**,
   with placeholder args. Iteration 2 (no restriction) then called
   create_job exactly as instructed.
2. TOOLS_DOCGEN was merged into EVERY work intent, so the exporters were
   allowed on a turn that never asked for a file.

Fix: TOOLS_WORK_TRACKING rides on every work intent; TOOLS_DOCGEN rides
on `has_document_intent` (an explicit format/artifact word) instead of
the category. A genuine "make me a PDF" still exposes the exporters —
every phrasing test_docgen_path_normalize.py pinned for the #371
incident is re-asserted below so the two incidents cannot trade places.
"""
from app.agent.query_intent import (
    TOOLS_DOCGEN,
    TOOLS_WORK_TRACKING,
    INTENT_AGENT, INTENT_CODE, INTENT_MEDIA, INTENT_MEMORY,
    INTENT_SCHEDULING, INTENT_WEB, INTENT_GREETING, INTENT_QUESTION,
    _ALWAYS_INCLUDED_TOOLS,
    classify_query_intent, filter_tools_by_intent, has_document_intent,
    with_document_tools,
)
from app.agent.tool_definitions import (
    get_agent_tools, get_extended_tools, get_doc_generation_tools,
    get_navigation_tools,
)


INCIDENT_MESSAGE = (
    "Can you search for me the latest ai news and give me the 5 most "
    "important one in this mounth"
)


def _wire():
    """The real wire array, in the order AgentRunner assembles it."""
    return (
        get_agent_tools() + get_extended_tools()
        + get_doc_generation_tools() + get_navigation_tools()
    )


def _allowed(msg: str) -> set:
    """Turn-1 allowed set the runner derives: intent tools ∪ always-included,
    intersected with the wire array (mirrors _gated_names)."""
    intent = classify_query_intent(msg)
    return {t["name"] for t in filter_tools_by_intent(_wire(), intent)}


# ── The incident, byte-for-byte ────────────────────────────────────


def test_incident_message_is_web_intent_with_jobs_and_without_exporters():
    intent = classify_query_intent(INCIDENT_MESSAGE)
    assert intent.category == "web"
    allowed = _allowed(INCIDENT_MESSAGE)
    # The tool the prompt mandates first is now allowed on turn 1 …
    assert "create_job" in allowed
    assert "update_job" in allowed
    # … and not one exporter is offered to a search-and-summarize ask.
    assert not (TOOLS_DOCGEN & allowed), sorted(TOOLS_DOCGEN & allowed)
    # The search tools it actually needs are still there.
    assert {"web_search", "web_fetch"} <= allowed


def test_snap_target_no_longer_exists_in_wire_order():
    """The mechanism: walk the wire array from create_job forward and take
    the first ALLOWED name. On the incident build that was generate_pdf.
    Now create_job itself is allowed, so the walk stops where the model
    aimed — no snap, no stub."""
    names = [t["name"] for t in _wire()]
    allowed = _allowed(INCIDENT_MESSAGE)
    i = names.index("create_job")
    first_allowed_from_create_job = next(n for n in names[i:] if n in allowed)
    assert first_allowed_from_create_job == "create_job"


def test_prod_log_counts_reproduce_before_the_fix_shape():
    """The prod line said `tools=14` / `allowed=19`. Documented, not pinned:
    the new numbers are what the fix produces (14 - 7 exporters + 3 work
    tracking = 10 intent tools, +1 for memory v3's `memory_read_file`; the
    runner adds the always-included set on top). If someone re-merges
    TOOLS_DOCGEN into TOOLS_WEB this fails."""
    intent = classify_query_intent(INCIDENT_MESSAGE)
    # 11 since memory v3: `memory_read_file` joined TOOLS_MEMORY. The index
    # in `# User Brain` names the user's files on every non-trivial turn,
    # so any intent can end up holding a slug it needs to open.
    assert len(intent.tool_names) == 11, sorted(intent.tool_names)
    assert not (TOOLS_DOCGEN & intent.tool_names)


# ── Work tracking is allowed on every work intent ─────────────────


def test_work_tracking_in_every_work_intent():
    for intent in (INTENT_CODE, INTENT_WEB, INTENT_MEDIA, INTENT_MEMORY,
                   INTENT_SCHEDULING, INTENT_AGENT):
        assert TOOLS_WORK_TRACKING <= intent.tool_names, intent.category


def test_work_tracking_not_on_greeting_or_question():
    """A greeting/question turn has no do-something ask; the always-
    included affordances are the only tools it carries."""
    for intent in (INTENT_GREETING, INTENT_QUESTION):
        assert not intent.tool_names
    for msg in ("hi", "thanks", "what's the weather in toronto"):
        allowed = _allowed(msg)
        assert "create_job" not in allowed, msg
        assert allowed <= _ALWAYS_INCLUDED_TOOLS, msg


def test_research_asks_across_categories_can_open_with_create_job():
    """Every phrasing here is a do-something ask that lands in a non-full
    category; each must be able to call create_job on turn 1."""
    for msg in (
        "search the web for the best pizza in toronto and rank the top 3",
        "look up the latest news on the toronto housing market",
        "what did we discuss yesterday about the stripe migration",
        "play some drake",
        "remind me in 20 minutes to call mom",
        "spawn a sub-agent to research LLM professors at UofT",
    ):
        allowed = _allowed(msg)
        intent = classify_query_intent(msg)
        assert intent.category != "full", (msg, intent.category)
        assert "create_job" in allowed, (msg, intent.category)


# ── Document intent: exporters ride on the ask, not the category ──


def test_no_document_intent_means_no_exporters_on_turn_one():
    for msg in (
        INCIDENT_MESSAGE,
        "search the web for the best pizza in toronto",
        "what's the latest news on AI regulation",
        "give me a brief overview of quantum computing",
        "summarize what we discussed yesterday",
        "play some drake",
        "remind me in 20 minutes to call mom",
        "the report said revenue was up 12%",
        "hi", "thanks", "yes", "ok",
    ):
        assert not has_document_intent(msg), msg
        assert not (TOOLS_DOCGEN & _allowed(msg)), msg


def test_explicit_document_asks_expose_exporters():
    """The #371 pins plus follow-up phrasings — a genuine document ask must
    still see the exporters on turn 1, whatever category the subject
    vocabulary picks."""
    for msg, tool in (
        ("Make me a one-page PDF summarizing the water cycle. Keep it brief - a few short sections is fine.", "generate_pdf"),
        ("Make me a PDF please", "generate_pdf"),
        ("Create a spreadsheet of my expenses this month", "generate_xlsx"),
        ("Can you make me a slide deck about Mars?", "generate_pptx"),
        ("Give me a summary of this month's expenses in an Excel file.", "generate_xlsx"),
        ("Write me a one-page project brief on the app rebuild, PDF please.", "generate_pdf"),
        ("write me an invoice for 3 hours of consulting", "generate_pdf"),
        ("put together a report on the top 5 AI news this month", "generate_pdf"),
        ("search the latest AI news and give me a report as a downloadable file", "generate_pdf"),
        ("export my expenses to excel", "generate_xlsx"),
        ("draft a memo to the team about the new policy", "generate_docx"),
        ("I need a cover letter for this job", "generate_docx"),
        ("save the notes as notes.md", "generate_markdown"),
        ("as a word doc please", "generate_docx"),
    ):
        assert has_document_intent(msg), msg
        allowed = _allowed(msg)
        assert tool in allowed, f"{msg!r} → {classify_query_intent(msg).category} hides {tool}"
        assert "convert_document" in allowed, msg


def test_follow_up_accepting_an_offered_export_is_not_a_greeting():
    """'yes pdf please' — first word a greeting, ≤4 words — used to short-
    circuit to INTENT_GREETING with no work tools. It is the user saying
    yes to a file."""
    for msg in ("yes pdf please", "ok make it a pdf", "sure, as a PDF",
                "yes, the excel version", "convert to PDF", "give me the PDF version"):
        intent = classify_query_intent(msg)
        assert intent.category != "greeting", (msg, intent.category)
        assert TOOLS_DOCGEN <= intent.tool_names, (msg, intent.category)


def test_with_document_tools_is_a_no_op_on_full():
    from app.agent.query_intent import INTENT_FULL
    assert with_document_tools(INTENT_FULL) is INTENT_FULL


def test_document_intent_merges_onto_question_intent():
    """A short ask that names a file keeps the tool-less question category
    (its prompt sections) but carries the exporters — the runner's filter
    branches on tool_names being non-empty, not on the category."""
    intent = classify_query_intent("as a word doc please")
    assert intent.category == "question"
    assert TOOLS_DOCGEN <= intent.tool_names
    exposed = {t["name"] for t in filter_tools_by_intent(_wire(), intent)}
    assert "generate_docx" in exposed
    assert "navigate_to" in exposed  # always-included survives the merge


def test_document_intent_is_cheap():
    import time
    t0 = time.perf_counter()
    for _ in range(2000):
        has_document_intent(INCIDENT_MESSAGE)
    per_call_ms = (time.perf_counter() - t0) * 1000 / 2000
    assert per_call_ms < 0.2, per_call_ms
