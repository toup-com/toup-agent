"""The model-facing memory tools (rebuild-2026-08-v3 §3.2).

Round 8 shipped an index into the prompt that named the user's files and NO
tool that could open one — `memory_search` had no slug parameter and
`memory_service.py` contained zero references to `file_slug`, so the map
led nowhere. `memory_read_file` is the load-bearing half of dropping
sentence retrieval; these tests pin that it exists, is registered
everywhere a tool has to be registered, and returns a file.
"""

import pathlib
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.tool_definitions import get_agent_tools
from app.db.models.base import Base
from app.db.models.memory import MemoryFile, MemoryFileChange
from app.db.models.user import User
from app.memory_files import PROFILE_SLUG


def _defs():
    return {t["name"]: t for t in get_agent_tools()}


# ── Registration ──────────────────────────────────────────────────────

def test_memory_read_file_is_defined_with_a_slug_parameter():
    tool = _defs()["memory_read_file"]
    assert tool["input_schema"]["required"] == ["slug"]
    assert "slug" in tool["input_schema"]["properties"]
    # The description has to teach where a slug comes from, or the model
    # invents one.
    assert "# User Brain" in tool["description"]


def test_memory_search_no_longer_offers_row_era_filters():
    """`brain_type` was a row concept (user/agent/work brains). The agent
    brain is the `learned` FILE now, so the parameter can only mislead."""
    tool = _defs()["memory_search"]
    assert set(tool["input_schema"]["properties"]) == {"query", "limit"}
    assert "memory_read_file" in tool["description"]


def test_memory_store_is_described_as_the_explicit_ask_only():
    tool = _defs()["memory_store"]
    text = tool["description"].lower()
    assert "explicitly" in text or "asked you to remember" in text
    # It must warn off exactly what round 8 filled the brain with.
    assert "reminder" in text and "one-off" in text


def test_every_memory_tool_has_a_handler():
    """Dispatch is `getattr(self, f"_tool_{name}")` — a defined tool with no
    handler is a runtime AttributeError the model triggers, not a test
    failure."""
    from app.agent.tool_executor import ToolExecutor

    for name in ("memory_search", "memory_read_file", "memory_store"):
        assert hasattr(ToolExecutor, f"_tool_{name}"), name


def test_the_new_tool_is_registered_in_the_intent_gate():
    """A tool the prompt advertises and the intent gate hides is the A6-3
    failure class: the model hallucinates the content, or truthfully says
    it cannot read its own memory."""
    from app.agent.query_intent import TOOLS_MEMORY, _ALWAYS_INCLUDED_TOOLS

    assert "memory_read_file" in TOOLS_MEMORY
    assert "memory_read_file" in _ALWAYS_INCLUDED_TOOLS


def test_a_subagent_cannot_open_a_memory_file():
    """SUBAGENT gets no `user_brain` section; a tool that returns Profile in
    full would reopen exactly that isolation."""
    from app.agent.prompt_profile import SUBAGENT_DISABLED_TOOLS

    assert "memory_read_file" in SUBAGENT_DISABLED_TOOLS
    assert "memory_store" in SUBAGENT_DISABLED_TOOLS
    # memory_search keeps its documented exemption — it answers a scoped
    # question rather than handing over a whole file.
    assert "memory_search" not in SUBAGENT_DISABLED_TOOLS


def test_the_output_cap_covers_a_whole_file():
    from app.agent.tool_executor import TOOL_OUTPUT_LIMITS
    from app.memory_files import MAX_BODY_CHARS

    assert TOOL_OUTPUT_LIMITS["memory_read_file"] >= MAX_BODY_CHARS


# ── Behaviour ─────────────────────────────────────────────────────────

async def _seeded_executor(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[User.__table__, MemoryFile.__table__, MemoryFileChange.__table__],
        )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    async with maker() as db:
        db.add(User(id=user_id, email="t@t.local", hashed_password="x", name="Nariman"))
        db.add(MemoryFile(
            user_id=user_id, slug=PROFILE_SLUG, section="you", title="Profile",
            description="Who this person is — setup; read when it matters.",
            body_md="- uses an Android phone", is_system=True,
        ))
        db.add(MemoryFile(
            user_id=user_id, slug="topics/music", section="topics", title="Music",
            description="Music taste — artists; read when music comes up.",
            body_md="- likes Googoosh and Ebi",
            links_json='["you/profile"]',
        ))
        await db.commit()

    import app.db.database as database

    monkeypatch.setattr(database, "async_session_maker", maker)

    from app.agent.tool_executor import ToolExecutor

    # Phase 8: per-call state is a ContextVar, not an attribute — writing
    # the attribute raises, which is the point of the property.
    executor = ToolExecutor.__new__(ToolExecutor)
    executor.set_user_id(user_id)
    return executor, engine


async def test_memory_read_file_returns_the_body_and_the_links(monkeypatch):
    executor, engine = await _seeded_executor(monkeypatch)
    out = await executor._tool_memory_read_file({"slug": "topics/music"})
    assert "# Music (topics/music)" in out
    assert "Music taste — artists" in out
    assert "- likes Googoosh and Ebi" in out
    assert "See also: Profile (you/profile)" in out

    missing = await executor._tool_memory_read_file({"slug": "topics/ghost"})
    assert "No memory file" in missing
    assert "ERROR" not in missing, "an absent file is an answer, not a failure"
    assert "ERROR" in await executor._tool_memory_read_file({"slug": ""})
    await engine.dispose()


async def test_memory_search_answers_with_file_attribution(monkeypatch):
    executor, engine = await _seeded_executor(monkeypatch)
    out = await executor._tool_memory_search({"query": "googoosh"})
    assert out.startswith("[topics/music] Music — ")
    assert "likes Googoosh" in out
    # No id, no similarity score, no category — the row-era output format
    # is what made the tool's results unusable as an answer.
    assert "sim=" not in out and "id=" not in out
    await engine.dispose()


async def test_the_document_leg_failing_does_not_take_the_file_half_down(monkeypatch):
    """The embedding call needs a key. Files are the memory; documents are
    the extra — a missing key must cost the extra only."""
    executor, engine = await _seeded_executor(monkeypatch)

    async def boom(*a, **kw):
        raise RuntimeError("no embedding key")

    monkeypatch.setattr(executor, "_search_documents", boom)
    out = await executor._tool_memory_search({"query": "googoosh"})
    assert "ERROR" in out or "[topics/music]" in out
    await engine.dispose()


def test_the_document_media_consumer_set_is_pinned():
    """§3.4: `documents.py` / `ingest.py` rows (source_type document/media)
    keep their pipeline and their embedding search, exposed ONLY through
    `memory_search`'s document leg. This enumerates BOTH ends so a future
    change has to come back here:

    * the producers — the only two places that mint such a row;
    * the consumer — the one caller that scopes a search to them.

    Everything else calling `search_memories_by_embedding` is row-era
    machinery WS-2 retires (dedup adjudication, the legacy REST /chat
    paths, the public v1 search). None of it reaches the memory PRODUCT,
    and none of it may start scoping to document/media to sneak back in."""
    backend = pathlib.Path(__file__).resolve().parents[1]
    # Code, not commentary — documents.py explains its own history in a
    # comment that also names the value.
    docs = "\n".join(
        line for line in (backend / "app" / "api" / "documents.py").read_text().splitlines()
        if not line.strip().startswith("#")
    )
    assert docs.count('source_type="document"') == 1
    assert docs.count('source_type="media"') == 1

    executor = (backend / "app" / "agent" / "tool_executor.py").read_text()
    assert executor.count('source_types=["document", "media"]') == 1, (
        "the document leg moved or multiplied — it is meant to be the ONE "
        "reader of memory rows left in the product"
    )
    # And it lives on the search tool's helper, not on a memory-file path.
    body = executor.split("async def _search_documents")[1]
    body = body.split("\n    # ---")[0]
    assert 'source_types=["document", "media"]' in body


async def test_the_document_leg_is_scoped_to_document_and_media(monkeypatch):
    """§3.4: uploads and transcripts keep their embedding pipeline and stay
    reachable HERE. A retired conversation row must never come back through
    this tool."""
    executor, engine = await _seeded_executor(monkeypatch)
    seen = {}

    class _Svc:
        api_key = None

        def __init__(self, db):
            pass

        class embedding_service:
            @staticmethod
            async def embed_async(text, api_key=None):
                return [0.0] * 8

        async def search_memories_by_embedding(self, **kw):
            seen.update(kw)
            return [{"content": "a transcript line"}]

    import app.services.memory_service as ms

    monkeypatch.setattr(ms, "MemoryService", _Svc)
    out = await executor._tool_memory_search({"query": "transcript", "limit": 5})
    assert seen["source_types"] == ["document", "media"]
    assert "[document] a transcript line" in out
    await engine.dispose()
