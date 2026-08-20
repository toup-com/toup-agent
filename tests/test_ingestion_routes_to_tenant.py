"""Document / media / conversation ingestion must land in the TENANT store.

THE DEFECT THESE TESTS LOCK
    `app/api/documents.py` and `app/api/ingest.py` were mounted only by
    `platform_main.py` (:894 and :893). Every table they touch — `documents`,
    `document_chunks`, `media`, `conversations`, `messages`, `entities`,
    `memories` — is in `AGENT_ONLY_TABLES`, excluded by platform-mode
    `init_db` and created by no alembic revision. Measured against a
    platform-mode Postgres on this branch's parent commit:

        POST /api/ingest/document  -> UndefinedTableError: relation "documents"
        GET  /api/ingest/documents -> UndefinedTableError: relation "documents"
        GET  /api/ingest/media     -> UndefinedTableError: relation "media"
        POST /api/ingest/message   -> UndefinedTableError: relation "conversations"

    On an older platform DB that still carries those tables as monolith
    leftovers the failure is quieter and worse: the rows land in a database
    the tenant agent never reads.

WHAT "WHICH DATABASE" MEANS HERE
    Every table the handler touches must EXIST locally, so that "zero local
    rows" is a real measurement rather than a table that happens to be
    missing. When a tenant is configured and the request still leaves zero
    rows behind while an X-Agent-Key request goes out to the agent's URL, the
    write demonstrably went to the tenant and not to the local session.

    This docstring used to say those tables were present because the suite
    ran in the conftest's default monolith profile, where `init_db` creates
    everything. That was true locally and FALSE in CI: the backend sweep runs
    every file under **RUN_MODE=platform**
    (`.github/workflows/test-backend.yml`), which excludes AGENT_ONLY tables
    by design. So all 11 DB-backed tests here failed with `no such table:
    documents` on main — an assumption about the environment, written down
    confidently, that the environment did not honour.

    The `_tenant_side_tables` fixture below now creates them explicitly from
    `AGENT_ONLY_TABLES` instead of inheriting them from a profile, so the file
    no longer depends on which RUN_MODE it happens to be invoked under.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

# Obviously fake — never a real credential.
FAKE_AGENT_URL = "http://tenant-agent.invalid:9100"
FAKE_AGENT_KEY = "fake-agent-key-for-tests-0000"


# ── fake tenant agent ────────────────────────────────────────────────────

class _FakeAgentHTTP:
    """Stand-in for the shared platform→agent httpx client."""

    def __init__(self):
        self.calls: list[dict] = []
        self.status_code = 200
        self.payload: object = {}
        self.raise_exc: Exception | None = None

    async def request(self, method, url, headers=None, params=None,
                      json=None, data=None, files=None, timeout=None):
        self.calls.append({
            "method": method,
            "url": url,
            "headers": dict(headers or {}),
            "params": dict(params or {}),
            "json": json,
            "data": dict(data or {}),
            "files": files,
        })
        if self.raise_exc is not None:
            raise self.raise_exc
        return httpx.Response(self.status_code, json=self.payload)


class _FakeEmbeddings:
    """Deterministic vectors — no network, no sentence-transformers download."""

    def __init__(self):
        self.vector = [0.01] * _column_dimension()

    async def embed_async(self, text, api_key=None):
        return list(self.vector)

    def embed(self, text, api_key=None):
        return list(self.vector)

    def embed_to_json(self, text):
        import json as _json
        return _json.dumps(list(self.vector))

    @staticmethod
    def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        return dot / (n1 * n2) if n1 and n2 else 0.0


def _column_dimension() -> int:
    from app.db.models import Memory
    col = Memory.__table__.columns.get("embedding")
    dim = getattr(col.type, "dim", None) if col is not None else None
    return int(dim) if dim else 1536


# ── fixtures ─────────────────────────────────────────────────────────────

class _User:
    def __init__(self, uid: str):
        self.id = uid


async def _seed_user(uid: str) -> None:
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as s:
        s.add(User(id=uid, email=f"{uid[:12]}@example.test",
                   hashed_password="x", name="Ingest Test"))
        await s.commit()


async def _seed_tenant(uid: str) -> None:
    """Give the user an ACTIVE agent — i.e. a tenant store to route to."""
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig

    async with async_session_maker() as s:
        s.add(AgentConfig(
            user_id=uid,
            agent_url=FAKE_AGENT_URL,
            agent_api_key=FAKE_AGENT_KEY,
            deploy_status="active",
        ))
        await s.commit()


async def _count(model) -> int:
    from app.db.database import async_session_maker

    async with async_session_maker() as s:
        return int((await s.execute(
            select(func.count()).select_from(model.__table__)
        )).scalar_one())


@pytest_asyncio.fixture(autouse=True)
async def _tenant_side_tables():
    """Create the tenant-side tables this file asserts emptiness on.

    `documents`, `document_chunks`, `media`, `conversations` and `memories`
    are AGENT_ONLY: `init_db()` creates them under RUN_MODE=agent, and the CI
    sweep runs this suite under **RUN_MODE=platform**
    (`.github/workflows/test-backend.yml`). Absent, every assertion here died
    with `no such table: documents` — 11 red on main, which is neither a pass
    nor a useful failure.

    Created rather than SKIPPED (the `requires_agent_tables` route in
    conftest) on purpose. The entire claim of this file is that ingestion
    writes **nothing** to the platform DB, and you cannot assert a table is
    empty if the table does not exist — skipping would retire the assertion
    along with the error. It would also mean these tests run nowhere at all:
    the sqlite sweep would skip them and `pytest-postgres` runs only four
    hand-listed files.

    Driven off `AGENT_ONLY_TABLES` — the same set `init_db()` itself uses
    (`database.py:266`) — rather than a hand-written list, which would drift
    the first time the ingestion path touched one more table. It already
    would have: fixing `documents`/`media`/`conversations` merely moved the
    error to `memory_events`.

    Safe on SQLite: the model layer skips pgvector columns when the extension
    is absent (`database.py:286`). Any table that still refuses to compile is
    left out rather than failing the run — `test_the_tenant_side_tables_really
    _exist` below asserts the ones THIS file needs, so an over-broad skip
    cannot pass silently.
    """
    from app.db.database import engine
    from app.db.models.base import AGENT_ONLY_TABLES, Base

    targets = [t for name, t in Base.metadata.tables.items()
               if name in AGENT_ONLY_TABLES]
    for table in targets:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(
                    lambda sync_conn, t=table: t.create(sync_conn, checkfirst=True))
        except Exception:
            # Dialect cannot express this table (e.g. a pgvector index). Not
            # fatal: the guard test below fails loudly if one this file
            # actually needs went missing.
            continue
    yield


@pytest.mark.asyncio
async def test_the_tenant_side_tables_really_exist():
    """Anti-vacuity guard for the fixture above.

    Every "writes nothing locally" assertion in this file is a count against
    one of these tables. If the fixture stopped creating them the counts would
    raise, and a future refactor that turned those raises into skips or
    tolerated errors would leave the file passing while asserting nothing.
    """
    from sqlalchemy import inspect as _inspect
    from app.db.database import engine

    async with engine.connect() as conn:
        present = await conn.run_sync(
            lambda s: {n: _inspect(s).has_table(n) for n in
                       ("documents", "document_chunks", "media",
                        "conversations", "memories", "memory_events")})
    missing = sorted(n for n, ok in present.items() if not ok)
    assert not missing, f"fixture failed to create: {missing}"


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    """App mounting the real ingestion routers, plus a fake tenant agent.

    Starts in PLATFORM mode (`run_mode="platform"`), which is what
    platform_main.py runs and the only mode where routing is even a question.
    The anti-vacuity control flips it back to monolith.
    """
    from app.api.auth import get_current_user
    from app.api.documents import router as documents_router
    from app.api.ingest import router as ingest_router
    from app.config import settings

    uid = uuid.uuid4().hex
    await _seed_user(uid)

    monkeypatch.setattr(settings, "run_mode", "platform")

    agent = _FakeAgentHTTP()
    monkeypatch.setattr(
        "app.services.agent_http.get_agent_http_client", lambda: agent
    )

    emb = _FakeEmbeddings()
    import app.services.memory_service as ms_mod
    import app.api.ingest as ingest_mod
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)
    monkeypatch.setattr(ingest_mod, "get_embedding_service", lambda: emb)

    app = FastAPI()
    app.include_router(ingest_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")
    app.dependency_overrides[get_current_user] = lambda: _User(uid)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {"client": client, "uid": uid, "agent": agent, "settings": settings}


# ── documents.py — WRITE routes to the tenant ────────────────────────────

@pytest.mark.asyncio
async def test_document_upload_goes_to_the_tenant_and_writes_nothing_locally(ctx):
    from app.db.models import Document, DocumentChunk, Memory

    await _seed_tenant(ctx["uid"])
    agent = ctx["agent"]
    agent.payload = {
        "document_id": "tenant-doc-1", "filename": "notes.txt",
        "file_type": "text", "chunks_processed": 1, "memories_created": 1,
        "entities_extracted": 0, "summary": None, "key_topics": None,
        "processing_time_ms": 7,
    }

    r = await ctx["client"].post(
        "/api/ingest/document",
        files={"file": ("notes.txt", b"The rocket is called Falcon.", "text/plain")},
        data={"category": "knowledge", "generate_summary": "false"},
    )

    assert r.status_code == 200, r.text
    # The id in the response was minted by the TENANT, not locally.
    assert r.json()["document_id"] == "tenant-doc-1"

    assert len(agent.calls) == 1
    call = agent.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"{FAKE_AGENT_URL}/api/ingest/document"
    assert call["headers"]["X-Agent-Key"] == FAKE_AGENT_KEY
    assert call["files"]["file"][1] == b"The rocket is called Falcon."
    assert call["data"]["category"] == "knowledge"

    # The local tables EXIST in this profile and are still empty: the write
    # went to the tenant, not to the session the handler was holding.
    assert await _count(Document) == 0
    assert await _count(DocumentChunk) == 0
    assert await _count(Memory) == 0


@pytest.mark.asyncio
async def test_document_upload_502s_and_writes_nothing_when_tenant_unreachable(ctx):
    from app.db.models import Document, DocumentChunk, Memory

    await _seed_tenant(ctx["uid"])
    ctx["agent"].raise_exc = httpx.ConnectError("tenant unreachable")

    r = await ctx["client"].post(
        "/api/ingest/document",
        files={"file": ("notes.txt", b"The rocket is called Falcon.", "text/plain")},
        data={"category": "knowledge", "generate_summary": "false"},
    )

    assert r.status_code == 502, r.text
    # No silent local write — that is how this class of bug survives.
    assert await _count(Document) == 0
    assert await _count(DocumentChunk) == 0
    assert await _count(Memory) == 0


@pytest.mark.asyncio
async def test_media_upload_goes_to_the_tenant(ctx):
    from app.db.models import Media, Memory

    await _seed_tenant(ctx["uid"])
    agent = ctx["agent"]
    agent.payload = {
        "media_id": "tenant-media-1", "filename": "shot.png",
        "media_type": "image", "memories_created": 1,
        "ai_description": None, "ai_transcript": None, "processing_time_ms": 4,
    }

    r = await ctx["client"].post(
        "/api/ingest/media",
        files={"file": ("shot.png", b"\x89PNG\r\n\x1a\n fake", "image/png")},
        data={"category": "knowledge"},
    )

    assert r.status_code == 200, r.text
    assert r.json()["media_id"] == "tenant-media-1"
    assert agent.calls[0]["url"] == f"{FAKE_AGENT_URL}/api/ingest/media"
    assert await _count(Media) == 0
    assert await _count(Memory) == 0


@pytest.mark.asyncio
async def test_agent_4xx_is_propagated_not_masked_as_502(ctx):
    """A duplicate upload must still read as 409, not "your agent is down"."""
    await _seed_tenant(ctx["uid"])
    ctx["agent"].status_code = 409
    ctx["agent"].payload = {"detail": "This document has already been uploaded"}

    r = await ctx["client"].post(
        "/api/ingest/document",
        files={"file": ("notes.txt", b"dupe", "text/plain")},
        data={"category": "knowledge", "generate_summary": "false"},
    )
    assert r.status_code == 409, r.text
    assert "already been uploaded" in r.json()["detail"]


# ── documents.py — READ routes to the tenant ─────────────────────────────

@pytest.mark.asyncio
async def test_list_documents_reads_the_tenant_not_the_platform_db(ctx):
    """A fixed write path with reads on the wrong DB is half a fix."""
    from app.db.database import async_session_maker
    from app.db.models import Document

    await _seed_tenant(ctx["uid"])

    # A decoy row in the PLATFORM db. If the read is still local it will show
    # up; if the read reaches the tenant it cannot.
    async with async_session_maker() as s:
        s.add(Document(
            id="platform-decoy", user_id=ctx["uid"], brain_type="user",
            category="knowledge", filename="decoy.txt",
            original_filename="decoy.txt", file_type="text",
            mime_type="text/plain", file_size=3, file_path="/tmp/decoy.txt",
            file_hash="deadbeef", title="Decoy",
        ))
        await s.commit()

    ctx["agent"].payload = [{
        "id": "tenant-doc-9", "filename": "from-tenant.txt", "file_type": "text",
        "file_size": 12, "brain_type": "user", "category": "knowledge",
        "title": "From tenant", "description": None, "chunk_count": 1,
        "memories_created": 1, "entities_extracted": 0, "summary": None,
        "key_topics": None, "created_at": "2026-08-06T00:00:00",
        "processed_at": None,
    }]

    r = await ctx["client"].get("/api/ingest/documents?limit=5")

    assert r.status_code == 200, r.text
    ids = [d["id"] for d in r.json()]
    assert ids == ["tenant-doc-9"]
    assert "platform-decoy" not in ids
    assert ctx["agent"].calls[0]["url"] == f"{FAKE_AGENT_URL}/api/ingest/documents"
    assert ctx["agent"].calls[0]["headers"]["X-Agent-Key"] == FAKE_AGENT_KEY


@pytest.mark.asyncio
async def test_list_documents_502s_when_tenant_unreachable(ctx):
    """No mirror table exists to fall back to, so an empty list would be a lie."""
    await _seed_tenant(ctx["uid"])
    ctx["agent"].raise_exc = httpx.ConnectError("tenant unreachable")

    r = await ctx["client"].get("/api/ingest/documents")
    assert r.status_code == 502, r.text


@pytest.mark.asyncio
async def test_delete_document_goes_to_the_tenant_and_leaves_the_platform_row_alone(ctx):
    from app.db.database import async_session_maker
    from app.db.models import Document

    await _seed_tenant(ctx["uid"])

    async with async_session_maker() as s:
        s.add(Document(
            id="platform-decoy", user_id=ctx["uid"], brain_type="user",
            category="knowledge", filename="decoy.txt",
            original_filename="decoy.txt", file_type="text",
            mime_type="text/plain", file_size=3, file_path="/tmp/decoy.txt",
            file_hash="deadbeef", title="Decoy",
        ))
        await s.commit()

    ctx["agent"].payload = {"status": "deleted", "document_id": "platform-decoy"}

    r = await ctx["client"].delete("/api/ingest/documents/platform-decoy")
    assert r.status_code == 200, r.text
    assert ctx["agent"].calls[0]["method"] == "DELETE"
    assert ctx["agent"].calls[0]["url"] == (
        f"{FAKE_AGENT_URL}/api/ingest/documents/platform-decoy"
    )

    # The delete was executed by the tenant; the platform row was never touched
    # (proving the handler did not silently operate on the wrong database).
    async with async_session_maker() as s:
        row = (await s.execute(
            select(Document).where(Document.id == "platform-decoy")
        )).scalar_one()
        assert row.is_deleted is False


@pytest.mark.asyncio
async def test_delete_document_502s_when_tenant_unreachable(ctx):
    await _seed_tenant(ctx["uid"])
    ctx["agent"].raise_exc = httpx.ConnectError("tenant unreachable")

    r = await ctx["client"].delete("/api/ingest/documents/whatever")
    assert r.status_code == 502, r.text


# ── ingest.py — the sibling with the identical defect ────────────────────

@pytest.mark.asyncio
async def test_message_ingest_goes_to_the_tenant_and_writes_nothing_locally(ctx):
    from app.db.models import Conversation, Memory, Message

    await _seed_tenant(ctx["uid"])
    ctx["agent"].payload = {
        "conversation_id": "tenant-conv-1", "messages_ingested": 2,
        "memories_extracted": 1, "entities_extracted": 0, "memories": [],
    }

    r = await ctx["client"].post("/api/ingest/message", json={
        "user_message": "My rocket is called Falcon.",
        "assistant_response": "Noted.",
        "extract_memories": True,
    })

    assert r.status_code == 200, r.text
    assert r.json()["conversation_id"] == "tenant-conv-1"
    assert ctx["agent"].calls[0]["url"] == f"{FAKE_AGENT_URL}/api/ingest/message"
    assert ctx["agent"].calls[0]["headers"]["X-Agent-Key"] == FAKE_AGENT_KEY
    assert ctx["agent"].calls[0]["json"]["user_message"] == "My rocket is called Falcon."

    assert await _count(Conversation) == 0
    assert await _count(Message) == 0
    assert await _count(Memory) == 0


@pytest.mark.asyncio
async def test_message_ingest_502s_and_writes_nothing_when_tenant_unreachable(ctx):
    from app.db.models import Conversation, Memory, Message

    await _seed_tenant(ctx["uid"])
    ctx["agent"].raise_exc = httpx.ConnectError("tenant unreachable")

    r = await ctx["client"].post("/api/ingest/message", json={
        "user_message": "My rocket is called Falcon.",
        "assistant_response": "Noted.",
        "extract_memories": True,
    })

    assert r.status_code == 502, r.text
    assert await _count(Conversation) == 0
    assert await _count(Message) == 0
    assert await _count(Memory) == 0


@pytest.mark.asyncio
async def test_conversation_ingest_goes_to_the_tenant(ctx):
    from app.db.models import Conversation, Message

    await _seed_tenant(ctx["uid"])
    ctx["agent"].payload = {
        "conversation_id": "tenant-conv-2", "messages_ingested": 2,
        "memories_extracted": 0, "entities_extracted": 0, "memories": [],
    }

    r = await ctx["client"].post("/api/ingest/conversation", json={
        "messages": [
            {"role": "user", "content": "Where do I keep the keys?"},
            {"role": "assistant", "content": "In the blue bowl."},
        ],
        "extract_memories": False,
    })

    assert r.status_code == 200, r.text
    assert r.json()["conversation_id"] == "tenant-conv-2"
    assert ctx["agent"].calls[0]["url"] == f"{FAKE_AGENT_URL}/api/ingest/conversation"
    assert await _count(Conversation) == 0
    assert await _count(Message) == 0


# ── ANTI-VACUITY CONTROLS ────────────────────────────────────────────────
#
# These must stay GREEN when the routing is reverted. They prove the tests
# above measure WHERE the write went, not merely that ingestion is broken.

@pytest.mark.asyncio
async def test_control_document_ingest_still_works_and_is_retrievable(ctx, monkeypatch):
    """Agent-side / monolith: the store IS this database, so ingest locally.

    This is the happy path the whole feature exists for — a document goes in,
    a document comes back out of the listing, and its text is retrievable as a
    memory.
    """
    from app.db.database import async_session_maker
    from app.db.models import Document, DocumentChunk, Memory

    # No tenant configured AND serving locally — the agent's own situation.
    monkeypatch.setattr(ctx["settings"], "run_mode", "monolith")

    r = await ctx["client"].post(
        "/api/ingest/document",
        files={"file": ("falcon.txt", b"The rocket is called Falcon.", "text/plain")},
        data={"category": "knowledge", "generate_summary": "false"},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["memories_created"] >= 1
    doc_id = body["document_id"]

    # Nothing was proxied — the store is right here.
    assert ctx["agent"].calls == []

    # The rows are real.
    assert await _count(Document) == 1
    assert await _count(DocumentChunk) >= 1
    assert await _count(Memory) >= 1

    # Retrievable through the listing endpoint...
    listed = await ctx["client"].get("/api/ingest/documents")
    assert listed.status_code == 200, listed.text
    assert [d["id"] for d in listed.json()] == [doc_id]

    # ...and the fact itself is in the memory store.
    async with async_session_maker() as s:
        contents = (await s.execute(select(Memory.content))).scalars().all()
    assert any("Falcon" in c for c in contents), contents


@pytest.mark.asyncio
@pytest.mark.parametrize("extract_flag", [True, False])
async def test_control_message_ingest_stores_the_turn_and_writes_no_memory(
    ctx, extract_flag
):
    """The control for ingest.py, half rewritten and half kept.

    KEPT — the routing control this test exists for: a turn goes in, the
    conversation and both messages are stored, and the tenant was NOT called
    (`run_mode=monolith`, so the route serves locally). That half is exactly
    what a control is for and it still ships.

    REWRITTEN — the memory half. This used to monkeypatch
    `ingest_mod.get_memory_extractor` with a stub returning one
    `ExtractedMemory` and assert `memories_extracted == 1` plus a `Memory`
    row containing "Falcon". `/ingest` was the SIXTH memory writer — the
    rule-based extractor plus a direct `create_memory`, with
    `extract_memories` defaulting to True on a router mounted by both
    entrypoints — and v3 severed it (rebuild-2026-08-v3 §2.1). The symbol it
    patched no longer exists, so the stub and the monkeypatch went with it.

    PARAMETRIZED over `extract_memories` ON PURPOSE. The request field is
    still in the schema (`IngestMessageRequest.extract_memories`), so a test
    that only ever passed `False` would not notice extraction coming back —
    it would read as "the caller asked for nothing and got nothing". Passing
    True is the case that matters: the caller asks, and the answer is still
    zero rows.
    """
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Memory, Message

    monkeypatch_settings = ctx["settings"]
    original_mode = monkeypatch_settings.run_mode
    monkeypatch_settings.run_mode = "monolith"
    try:
        r = await ctx["client"].post("/api/ingest/message", json={
            "user_message": "My rocket is called Falcon.",
            "assistant_response": "Noted.",
            "extract_memories": extract_flag,
        })
    finally:
        monkeypatch_settings.run_mode = original_mode

    assert r.status_code == 200, r.text
    body = r.json()

    # The half that still ships.
    assert body["messages_ingested"] == 2
    assert ctx["agent"].calls == []
    assert await _count(Conversation) == 1
    assert await _count(Message) == 2

    # The half that v3 severed — whatever the caller asked for.
    assert body["memories_extracted"] == 0, (
        f"ingest reported extracting memories with extract_memories="
        f"{extract_flag} — the sixth writer is back"
    )
    assert body["entities_extracted"] == 0
    assert body["memories"] == []

    async with async_session_maker() as s:
        contents = (await s.execute(select(Memory.content))).scalars().all()
    assert contents == [], f"ingest wrote a memory row: {contents}"
    assert not any("Falcon" in c for c in contents), contents


@pytest.mark.asyncio
async def test_control_conversation_ingest_stores_every_message_and_no_memory(ctx):
    """The sibling control, which never existed.

    `/ingest/conversation` had only a PROXY test — nothing checked that the
    local handler stores what it is given. v3 rewrote that handler's body
    (the extraction block came out, and the message loop was restructured
    around it), so a mistake there — an off-by-one in the pairing loop, a
    lost `flush`, a message_count that no longer matches — would have been
    invisible. Three messages in, three rows out, zero memory.
    """
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Memory, Message

    original_mode = ctx["settings"].run_mode
    ctx["settings"].run_mode = "monolith"
    try:
        r = await ctx["client"].post("/api/ingest/conversation", json={
            "messages": [
                {"role": "user", "content": "Where do I keep the keys?"},
                {"role": "assistant", "content": "In the blue bowl."},
                {"role": "user", "content": "My rocket is called Falcon."},
            ],
            "title": "Keys",
            # Asked for, and still refused — same reasoning as the message
            # control: the schema field survives, so True is the case that
            # would notice extraction coming back.
            "extract_memories": True,
        })
    finally:
        ctx["settings"].run_mode = original_mode

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["messages_ingested"] == 3
    assert ctx["agent"].calls == []

    assert await _count(Conversation) == 1
    assert await _count(Message) == 3
    assert await _count(Memory) == 0

    assert body["memories_extracted"] == 0
    assert body["entities_extracted"] == 0
    assert body["memories"] == []

    # An odd message count must not lose the trailing message — the old
    # handler paired messages two at a time and only the PAIRS were mined,
    # so a dangling third was easy to drop when the loop was rewritten.
    async with async_session_maker() as s:
        stored = (await s.execute(select(Message.content))).scalars().all()
    assert any("Falcon" in c for c in stored), stored


# ── end-to-end: platform → the REAL agent-side handler ──────────────────

@pytest.mark.asyncio
async def test_platform_upload_is_ingested_by_the_real_agent_handler(ctx, monkeypatch):
    """Close the loop with the actual agent router, not a canned response.

    A fake that always answers 200 cannot catch a shape mismatch — a renamed
    form field or a wrong path would still "pass". Here the platform's outbound
    request is fed straight into a second app mounting the SAME routers, which
    is exactly what agent_main.py now does. The tenant-side identity is a
    different user (on a real agent `get_current_user` resolves X-Agent-Key to
    `settings.user_id`), so the owner of the resulting rows tells us
    unambiguously which handler wrote them.
    """
    from app.api.auth import get_current_user
    from app.api.documents import router as documents_router
    from app.api.ingest import router as ingest_router
    from app.db.database import async_session_maker
    from app.db.models import Document, Memory

    await _seed_tenant(ctx["uid"])

    tenant_uid = uuid.uuid4().hex
    await _seed_user(tenant_uid)  # the agent's owner — has NO agent_config row

    agent_app = FastAPI()
    agent_app.include_router(ingest_router, prefix="/api")
    agent_app.include_router(documents_router, prefix="/api")
    agent_app.dependency_overrides[get_current_user] = lambda: _User(tenant_uid)

    class _Bridge:
        def __init__(self, app):
            self.inner = AsyncClient(
                transport=ASGITransport(app=app), base_url=FAKE_AGENT_URL
            )
            self.calls: list[dict] = []

        async def request(self, method, url, headers=None, params=None,
                          json=None, data=None, files=None, timeout=None):
            self.calls.append({"method": method, "url": url})
            return await self.inner.request(
                method, url, headers=headers, params=params,
                json=json, data=data, files=files,
            )

    bridge = _Bridge(agent_app)
    monkeypatch.setattr(
        "app.services.agent_http.get_agent_http_client", lambda: bridge
    )

    r = await ctx["client"].post(
        "/api/ingest/document",
        files={"file": ("falcon.txt", b"The rocket is called Falcon.", "text/plain")},
        data={"category": "knowledge", "generate_summary": "false"},
    )

    assert r.status_code == 200, r.text
    assert r.json()["memories_created"] >= 1

    # The row belongs to the TENANT identity — written by the agent-side
    # handler. A platform-local write would have carried ctx["uid"].
    async with async_session_maker() as s:
        owners = (await s.execute(select(Document.user_id))).scalars().all()
    assert owners == [tenant_uid]
    assert ctx["uid"] not in owners

    # And it comes back out through the platform's own listing, which proxies
    # to the agent's list handler.
    listed = await ctx["client"].get("/api/ingest/documents")
    assert listed.status_code == 200, listed.text
    assert [d["id"] for d in listed.json()] == [r.json()["document_id"]]

    async with async_session_maker() as s:
        contents = (await s.execute(select(Memory.content))).scalars().all()
    assert any("Falcon" in c for c in contents), contents

    await bridge.inner.aclose()


# ── the mount half of the defect ─────────────────────────────────────────

def test_agent_main_mounts_the_ingestion_routers():
    """`grep -c documents_router backend/agent_main.py` used to return 0.

    Proxying is only half the fix: if the agent does not mount these routers
    the proxy has nowhere to land and every upload becomes a 404 → 502.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "agent_main.py").read_text()
    assert "app.include_router(documents_router" in src
    assert "app.include_router(ingest_router" in src
