"""Per-automation session threads + outcome notify + memory domains
(Round 28-C, agent lane).

Listed in COVERAGE_DEBT.txt `# agent-mode` (conversations/messages/
build_jobs/automations/agent_notify_outbox are AGENT_ONLY).

Proves:
  - "automation" is a KNOWN channel for chat plumbing but deliberately
    NOT an MCP channel (the background clamp stays the mutating-tool
    fence) and NOT in the per-day unique index set (N automations share
    one day)
  - the session resolver converges concurrent-style repeat writes onto
    ONE conversation per (user, day, automation), and isolates two
    automations into two rows
  - a run card is a real role="job" marker the existing card pipeline
    can hydrate, written best-effort in its own session
  - GET /{id}/thread mints today's session lazily, returns the
    /api/sessions serialization, spills over days, and 404s on strangers
  - GET /{id}/memory serves the engine's working-state row and 404s
    before the first write
  - notify_run_outcome pushes ONLY noteworthy outcomes, with the deep
    link contract fields and a stable per-automation dedupe key
  - domains normalize/refuse correctly and fact-writing creates the
    areas/<domain> file deterministically, then hands the curator a
    clean instruction (never provider payloads)
"""

import json
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    Automation, BuildJob, Conversation, Memory, Message, User,
)


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Sessions"))
        await db.commit()
    return uid


async def _mk_automation(uid: str, name: str = "Watcher") -> str:
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name=name, status="draft",
            spec_json=json.dumps({"name": name}),
            trigger_mode="poll", connector_id="stub",
        )
        db.add(a)
        await db.commit()
        return a.id


# ── Channel registration ─────────────────────────────────────────────


def test_automation_channel_is_known_but_fenced():
    from app.agent.channel_util import KNOWN_CHANNELS
    from app.agent.conversation_resolver import (
        INDEXED_SYSTEM_CHANNELS, SYSTEM_CHANNELS,
    )
    assert "automation" in KNOWN_CHANNELS
    # N automations share one day — the (user, day, channel) unique
    # index would make the second one an IntegrityError.
    assert "automation" not in INDEXED_SYSTEM_CHANNELS
    assert "automation" not in SYSTEM_CHANNELS
    # The MCP background clamp is a mutating-tool fence: an MCP turn
    # claiming channel=automation must keep clamping to background.
    from app.mcp_auth import _KNOWN_CHANNELS as MCP_CHANNELS
    assert "automation" not in MCP_CHANNELS


# ── Session resolver ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_converges_per_automation_and_isolates():
    uid = await _mk_user()
    a1 = await _mk_automation(uid, "First")
    a2 = await _mk_automation(uid, "Second")
    from app.agent.automations.session import write_session_message

    ids = set()
    async with async_session_maker() as db:
        for _ in range(2):
            mid, day_id = await write_session_message(
                db, user_id=uid, automation_id=a1,
                content="notice", title="First",
            )
            assert mid and day_id
    async with async_session_maker() as db:
        mid2, _ = await write_session_message(
            db, user_id=uid, automation_id=a2, content="other",
            title="Second",
        )
        assert mid2

    async with async_session_maker() as db:
        convs = (await db.execute(
            select(Conversation).where(
                Conversation.user_id == uid,
                Conversation.channel == "automation",
            )
        )).scalars().all()
        assert len(convs) == 2  # one per automation, not per message
        by_auto = {}
        for c in convs:
            meta = json.loads(c.metadata_json or "{}")
            by_auto[meta.get("automation_id")] = c
        assert set(by_auto) == {a1, a2}
        assert by_auto[a1].message_count == 2
        msgs = (await db.execute(
            select(Message).where(
                Message.conversation_id == by_auto[a1].id)
        )).scalars().all()
        assert len(msgs) == 2
        assert all(m.channel == "automation" for m in msgs)
        assert all(m.source == "automation" for m in msgs)
        assert all(m.day_chat_id for m in msgs)  # in the agent's day


@pytest.mark.asyncio
async def test_run_card_is_a_hydratable_job_marker():
    uid = await _mk_user()
    aid = await _mk_automation(uid, "Card maker")
    job_id = str(uuid.uuid4())
    from app.agent.automations.session import write_run_card
    from app.api.message_cards import parse_job_marker

    msg_id, _day = await write_run_card(
        user_id=uid, automation_id=aid,
        automation_name="Card maker", job_id=job_id,
    )
    assert msg_id
    async with async_session_maker() as db:
        m = await db.get(Message, msg_id)
        assert m is not None and m.role == "job"
        marker = parse_job_marker(m.content)
        assert marker and marker["job_id"] == job_id
        assert marker["job_name"] == "Card maker"
        assert marker["job_type"] == "automation_run"


# ── HTTP surface ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_endpoint_mints_serializes_and_404s(monkeypatch):
    uid = await _mk_user()
    aid = await _mk_automation(uid, "Threaded")
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)
    from app.api.automations import automation_thread
    from fastapi import HTTPException

    out = await automation_thread(aid, limit=100)
    assert out["session_id"]
    assert out["messages"] == []

    # A run marker with a real job row hydrates through the pipeline.
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=uid, title="Threaded",
            prompt="(automation)", job_type="automation_run",
            status="completed", outcome="sent",
            source_kind="automation", source_id=aid,
        ))
        await db.commit()
    from app.agent.automations.session import write_run_card
    msg_id, _ = await write_run_card(
        user_id=uid, automation_id=aid,
        automation_name="Threaded", job_id=job_id,
    )
    assert msg_id

    out2 = await automation_thread(aid, limit=100)
    assert out2["session_id"] == out["session_id"]  # same day → same row
    rows = out2["messages"]
    assert len(rows) == 1
    row = rows[0]
    assert row.role == "job"
    assert row.job_id == job_id
    assert row.job_status == "completed"

    with pytest.raises(HTTPException) as e:
        await automation_thread(str(uuid.uuid4()), limit=100)
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_memory_endpoint_serves_state_row(monkeypatch):
    uid = await _mk_user()
    aid = await _mk_automation(uid, "Remembers")
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)
    from app.api.automations import automation_memory
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        await automation_memory(aid)
    assert e.value.status_code == 404

    async with async_session_maker() as db:
        db.add(Memory(
            id=str(uuid.uuid4()), user_id=uid,
            content="Last ran and posted 3 items.",
            ref_kind="automation", ref_id=aid,
            source_type="automation", brain_type="agent",
            category="automation", memory_type="state",
            metadata_json=json.dumps(
                {"last_outcome": "sent", "last_counts": {"jira": 3}}),
        ))
        await db.commit()

    out = await automation_memory(aid)
    assert out["content"] == "Last ran and posted 3 items."
    assert out["metadata"]["last_outcome"] == "sent"
    assert out["updated_at"]


@pytest.mark.asyncio
async def test_flag_off_thread_404s(monkeypatch):
    monkeypatch.setattr(settings, "automations_enabled", False)
    from app.api.automations import automation_thread
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        await automation_thread(str(uuid.uuid4()), limit=100)
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_composer_send_does_not_fork_the_session():
    """An explicit session_id aimed at the automation thread survives
    the channel-switch fork (app/web composers always declare a
    different channel), and a stale id from a previous day rolls to
    TODAY's row for the SAME automation — never a plain-channel fork."""
    from types import SimpleNamespace
    from datetime import timedelta

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Composer")
    from app.agent.automations.session import write_session_message
    from app.agent.agent_runner import AgentRunner

    async with async_session_maker() as db:
        mid, _ = await write_session_message(
            db, user_id=uid, automation_id=aid, content="card",
            title="Composer",
        )
        assert mid

    async with async_session_maker() as db:
        conv = (await db.execute(
            select(Conversation).where(
                Conversation.user_id == uid,
                Conversation.channel == "automation",
            )
        )).scalar_one()
        sess, is_new = await AgentRunner._get_or_create_session(
            SimpleNamespace(), db, uid, conv.id, None, channel="app",
        )
        assert sess.id == conv.id
        assert is_new is False

    # A stale id: move the row to a previous day (own DayChat, backdated
    # start) → the send rolls to a fresh row for the same automation.
    from app.db.models.day_chat import DayChat
    async with async_session_maker() as db:
        old_day = DayChat(
            id=str(uuid.uuid4()), user_id=uid,
            local_date=(datetime.utcnow() - timedelta(days=3)).date(),
            timezone="UTC",
        )
        db.add(old_day)
        row = await db.get(Conversation, conv.id)
        row.day_chat_id = old_day.id
        row.started_at = datetime.utcnow() - timedelta(days=3)
        await db.commit()

    async with async_session_maker() as db:
        sess2, is_new2 = await AgentRunner._get_or_create_session(
            SimpleNamespace(), db, uid, conv.id, None, channel="web",
        )
        assert sess2.id != conv.id
        assert sess2.channel == "automation"
        assert json.loads(sess2.metadata_json)["automation_id"] == aid
        assert is_new2 is False  # continuous thread, resolver semantics


# ── Outcome notify ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_pushes_only_noteworthy_outcomes():
    uid = await _mk_user()
    aid = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    from app.agent.automations.notify import notify_run_outcome
    from app.db.models import AgentNotifyOutbox

    # A quiet outcome never pushes.
    assert await notify_run_outcome(
        user_id=uid, automation_id=aid, automation_name="Quiet",
        job_id=job_id, outcome="failed",
    ) is False
    assert await notify_run_outcome(
        user_id=uid, automation_id=aid, automation_name="Quiet",
        job_id=job_id, outcome=None,
    ) is False

    assert await notify_run_outcome(
        user_id=uid, automation_id=aid, automation_name="Loud",
        job_id=job_id, outcome="sent", wrote_count=1,
        chat_id="conv-1", message_id="msg-1",
    ) is True

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AgentNotifyOutbox).where(
                AgentNotifyOutbox.dedup_key
                == f"automation:{aid}:run_done")
        )).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.event_kind == "mission_completed"
        assert row.title == "Loud ran"
        data = row.data_json or {}
        assert data["route"] == "automation"
        assert data["automation_id"] == aid
        assert data["run_id"] == job_id
        assert data["mission_id"] == job_id
        assert data["chat_id"] == "conv-1"
        assert data["message_id"] == "msg-1"
        assert data["no_agent_fallback"] is True


@pytest.mark.asyncio
async def test_create_automation_normalizes_domain(monkeypatch):
    """The skill's `domain` arg reaches the column normalized; junk
    becomes NULL (no facts filed), never an error."""
    uid = await _mk_user()
    from app.agent.automations import registry as reg
    from app.agent.automations.service import (
        automation_payload, create_automation,
    )

    stub_registry = {
        "jira": {
            "connector_id": "jira", "push": False, "poll": True,
            "floor_s": 300, "rate_budget": {}, "scopes_read": [],
            "scopes_write_by_action": {}, "target_param_by_action": {},
            "events": [{
                "key": "issue_created", "description": "",
                "source_tool": "jira__search_issues", "poll_args": {},
                "items_path": "issues", "dedupe_field": "key",
                "fields": {"key": "key", "summary": "summary"},
            }],
        },
        "slack": {
            "connector_id": "slack", "push": False, "poll": False,
            "floor_s": 300, "rate_budget": {}, "scopes_read": [],
            "scopes_write_by_action": {"slack__send_message": ["w"]},
            "target_param_by_action": {"slack__send_message": "channel"},
            "events": [],
        },
    }

    async def fake_registry(_uid, force=False):
        return stub_registry

    monkeypatch.setattr(reg, "fetch_registry", fake_registry)
    spec = {
        "name": "Jira watch",
        "trigger": {"mode": "poll", "connector_id": "jira",
                    "event": "issue_created", "poll_interval_s": 300},
        "action": {"connector_id": "slack",
                   "tool": "slack__send_message",
                   "params_template": {"channel": "{{grant.target.id}}",
                                       "text": "{{event.summary}}"},
                   "grant_id": "g-1"},
        "dedupe_key": "event.key",
        "mode": "auto",
    }
    async with async_session_maker() as db:
        a1, _ = await create_automation(
            db, user_id=uid, spec=spec, domain=" Work ",
        )
        assert a1.domain == "work"
        assert automation_payload(a1)["domain"] == "work"
        a2, _ = await create_automation(
            db, user_id=uid, spec=dict(spec, name="Second"),
            domain="Not A Domain!",
        )
        assert a2.domain is None


# ── Memory domains ───────────────────────────────────────────────────


def test_normalize_domain_matrix():
    from app.agent.automations.memory_notes import normalize_domain
    assert normalize_domain("work") == "work"
    assert normalize_domain(" University ") == "university"
    assert normalize_domain("personal") == "personal"
    assert normalize_domain("side-projects") == "side-projects"
    assert normalize_domain("") is None
    assert normalize_domain(None) is None
    assert normalize_domain("Bad Slug!") is None
    assert normalize_domain("x" * 40) is None
    assert normalize_domain(42) is None


@pytest.mark.asyncio
async def test_record_fact_creates_domain_file_and_stays_clean(monkeypatch):
    uid = await _mk_user()
    from app.agent.automations import memory_notes

    seen: dict = {}

    async def fake_instruct(db, user_id, slug, instruction, **kw):
        seen["slug"] = slug
        seen["instruction"] = instruction
        return {"applied": 1, "rejected": [], "changed_files": [slug]}

    from app.services import memory_curator
    monkeypatch.setattr(memory_curator, "instruct_file", fake_instruct)

    async with async_session_maker() as db:
        ok = await memory_notes.record_automation_fact(
            db, user_id=uid, domain="work",
            fact=memory_notes.setup_fact(
                automation_name="Morning brief",
                trigger_summary="runs weekday mornings",
                action_summary="posts a summary to Slack",
            ),
        )
    assert ok is True
    assert seen["slug"] == "areas/work"
    assert "Morning brief" in seen["instruction"]
    assert "\n" not in seen["instruction"]  # single clean line
    # The domain file was created deterministically before the curator ran.
    from app.db.models import MemoryFile
    async with async_session_maker() as db:
        row = (await db.execute(
            select(MemoryFile).where(
                MemoryFile.user_id == uid,
                MemoryFile.slug == "areas/work")
        )).scalar_one_or_none()
        assert row is not None
        assert row.section == "areas"

    # Bad domain: nothing written, curator never called.
    seen.clear()
    async with async_session_maker() as db:
        ok2 = await memory_notes.record_automation_fact(
            db, user_id=uid, domain="Not A Domain!", fact="x",
        )
    assert ok2 is False and not seen


def test_setup_fact_is_one_clean_line():
    from app.agent.automations.memory_notes import setup_fact
    s = setup_fact(
        automation_name="A\nB",
        trigger_summary="watches   mail",
        action_summary="drafts\treplies",
    )
    assert "\n" not in s and "\t" not in s
    assert "A B" in s and "watches mail" in s and "drafts replies" in s
