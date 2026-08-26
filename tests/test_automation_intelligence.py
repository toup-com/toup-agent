"""Round 29-C — confirm mode, session cards/chips, memory-filtered
events, and the interview seam (agent lane).

Listed in COVERAGE_DEBT.txt `# agent-mode` (build_jobs/automations/
conversations/messages are AGENT_ONLY).

Proves:
  - a confirmation_required flush parks the run VISIBLY: error_class
    stamped (the class every TTL backstop matches — R28 omitted it),
    a pending_action card in the session thread (the chat path's key,
    so clients reuse their renderer), and the card ids in config
  - resolutions ride the finalize gate with the run vocabulary:
    executed→sent, rejected/expired→cancelled+"skipped" (a user
    decision, not a failure — the health streak must not move),
    failed→write_failed (streak moves)
  - the confirm sweep closes ONLY expired parks and tells the
    "nothing was sent" truth; the generic reaper leaves automation
    runs alone
  - a `{{facts.<category>}}` filter needle matches events against the
    fact ledger reverse-containment style, and an empty/unavailable
    ledger matches NOTHING (an allowlist, not a pass-through)
  - an executed draft write leaves a draft_card in the session that
    says nothing was sent; raw tool names appear on none of these
    surfaces
  - the auto-pause notice carries a fix chip (label+prompt), not a
    navigate directive into a page the web app doesn't have
  - the serializers pass the four R29 metadata keys through, and the
    memory-update marker carries count+at plus its WS frame shape
  - the interview: an automation-channel conversation yields a prompt
    section (name + posture, no tool names), and the post-turn
    extractor records via the §4 seam and emits the chip
"""

import json
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationOutbox, BuildJob, Conversation, Message, User,
)

pytestmark = pytest.mark.asyncio


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="R29C"))
        await db.commit()
    return uid


async def _mk_automation(uid: str, name: str = "Draft replies") -> str:
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name=name, status="armed",
            spec_json=json.dumps({"name": name}),
            trigger_mode="push", connector_id="gmail",
        )
        db.add(a)
        await db.commit()
        return a.id


async def _mk_run(uid: str, aid: str, *, status: str = "running") -> str:
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=uid, title="Draft replies",
            prompt="(automation)", job_type="automation_run",
            status=status, source_kind="automation", source_id=aid,
        ))
        await db.commit()
    return job_id


async def _stage_row(uid: str, aid: str, job_id: str,
                     *, tool: str = "gmail__create_draft",
                     connector: str = "gmail",
                     payload: dict | None = None) -> str:
    async with async_session_maker() as db:
        row = AutomationOutbox(
            user_id=uid, automation_id=aid, job_id=job_id,
            connector_id=connector, tool_name=tool,
            payload_json=json.dumps(payload or {
                "to": "sarah@acme.com", "subject": "Re: contract",
                "body": "Draft body here.",
            }),
            grant_id="g-1",
            idempotency_key=f"t:{uuid.uuid4()}",
            execute_after=datetime.utcnow() - timedelta(seconds=1),
        )
        db.add(row)
        await db.commit()
        return row.id


def _dispatch_returning(result: dict):
    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        grant_id=None, automation_id=None, request_id=None,
                        timeout_s=60.0):
        return result
    return _dispatch


async def _thread_turns(aid: str, kind: str = "") -> list[dict]:
    """The automation's TURNS — where R31 moved everything the session
    used to write into the day chat (CONTRACTS-R31 §4.1).

    `_session_messages` below still exists because two tests assert the
    day chat is EMPTY, which needs a reader pointed at it.
    """
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, aid)
        if thread is None:
            return []
        turns, _more = await ledger.list_turns(
            db, thread_id=thread.id, limit=200,
        )
    return [t for t in turns if not kind or t.get("kind") == kind]


async def _session_messages(uid: str, aid: str) -> list[Message]:
    async with async_session_maker() as db:
        convs = (await db.execute(
            select(Conversation).where(
                Conversation.user_id == uid,
                Conversation.channel == "automation",
            )
        )).scalars().all()
        conv_ids = [
            c.id for c in convs
            if json.loads(c.metadata_json or "{}").get("automation_id") == aid
        ]
        if not conv_ids:
            return []
        return list((await db.execute(
            select(Message).where(Message.conversation_id.in_(conv_ids))
            .order_by(Message.created_at.asc())
        )).scalars().all())


# ── Confirm park ─────────────────────────────────────────────────────


async def test_confirm_park_stamps_class_and_writes_card(monkeypatch):
    from app.agent.automations import outbox
    from app.agent.job_status import ERR_AWAITING_CONFIRMATION

    uid = await _mk_user()
    aid = await _mk_automation(uid)
    job_id = await _mk_run(uid, aid)
    oid = await _stage_row(uid, aid, job_id)
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _dispatch_returning({
            "kind": "confirmation_required",
            "action_id": "act-1",
            "summary": "Draft a reply to sarah@acme.com",
            "payload": {"to": "sarah@acme.com"},
            "expires_at": (datetime.utcnow()
                           + timedelta(hours=24)).isoformat() + "Z",
        }))

    async with async_session_maker() as db:
        assert await outbox._claim(db, oid) is True
        assert await outbox._execute_claimed(db, oid) == "executed"

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "waiting_on_user"
        # THE R28 gap: without this stamp no TTL backstop can see the
        # park (job_reaper matches on exactly this class).
        assert job.error_class == ERR_AWAITING_CONFIRMATION
        assert "approval" in (job.user_message or "").lower()
        cfg = job.config_json or {}
        assert cfg.get("pending_action_id") == "act-1"
        assert cfg.get("pending_action_expires_at")

    # R31 §4.1: the park's DURABLE half is the `waiting` turn in the
    # automation's thread. It used to also write a day-chat Message
    # carrying `**{name}** staged an action…` — raw markdown bold, in
    # the main chat, for an automation the main chat should not be
    # narrating. The live `pending_action` frame still goes out.
    waiting = await _thread_turns(aid, "waiting")
    assert len(waiting) == 1
    assert waiting[0]["pending_action_id"] == "act-1"
    assert waiting[0]["text"]
    # No raw tool name in the human copy.
    assert "__" not in waiting[0]["text"]
    # And nothing reached the day chat.
    assert await _session_messages(uid, aid) == []


async def test_resolutions_use_run_vocabulary_and_spare_health(monkeypatch):
    from app.agent.automations import confirm, outbox

    uid = await _mk_user()
    aid = await _mk_automation(uid)

    async def _park(action_id: str) -> str:
        job_id = await _mk_run(uid, aid)
        oid = await _stage_row(uid, aid, job_id)
        monkeypatch.setattr(
            "app.agent.automations.registry.dispatch_via_platform",
            _dispatch_returning({
                "kind": "confirmation_required", "action_id": action_id,
                "summary": "s",
            }))
        async with async_session_maker() as db:
            assert await outbox._claim(db, oid)
            await outbox._execute_claimed(db, oid)
        return job_id

    # rejected → cancelled + skipped, card flipped, streak untouched
    j1 = await _park("act-r")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, j1)
        assert await confirm.resolve_parked_run(db, job, outcome="rejected")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, j1)
        assert (job.status, job.outcome) == ("cancelled", "skipped")
        assert "nothing was sent" in (job.user_message or "").lower()
        assert job.error_class is None  # the park stamp must not linger
        a = await db.get(Automation, aid)
        assert (a.consecutive_failures or 0) == 0

    # executed → completed + sent, streak reset stays 0
    j2 = await _park("act-e")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, j2)
        assert await confirm.resolve_parked_run(db, job, outcome="executed")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, j2)
        assert (job.status, job.outcome) == ("completed", "sent")

    # failed → failed + write_failed, streak moves
    j3 = await _park("act-f")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, j3)
        assert await confirm.resolve_parked_run(
            db, job, outcome="failed", detail="provider said no")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, j3)
        assert (job.status, job.outcome) == ("failed", "write_failed")
        a = await db.get(Automation, aid)
        assert (a.consecutive_failures or 0) == 1

    # A replay is a no-op: the finalize gate's rowcount already flipped.
    async with async_session_maker() as db:
        job = await db.get(BuildJob, j1)
        await confirm.resolve_parked_run(db, job, outcome="executed")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, j1)
        assert (job.status, job.outcome) == ("cancelled", "skipped")

    # R31 §4.1: the THREAD followed the resolutions.
    #
    # This read the day-chat cards' metadata. There are no such cards
    # now, and the surface that asked "Nothing happens until you
    # approve." has to be the one that reports what was decided — a
    # card that cannot answer the question it asked is worse than no
    # card. The `waiting` turn is REPLACED by the decision, so the
    # thread never carries a standing approval request underneath its
    # own answer.
    assert await _session_messages(uid, aid) == []
    assert await _thread_turns(aid, "waiting") == []
    said = " ".join(t.get("text") or ""
                    for t in await _thread_turns(aid, "agent"))
    assert "You said no, so nothing was sent." in said
    assert "You approved it, so I went ahead." in said
    assert "the change did not go through" in said


async def test_expiry_sweep_closes_only_expired_and_reaper_abstains(
        monkeypatch):
    from app.agent.automations import confirm, outbox
    from app.agent import job_reaper

    uid = await _mk_user()
    aid = await _mk_automation(uid)

    async def _park(action_id: str, expires_delta: timedelta) -> str:
        job_id = await _mk_run(uid, aid)
        oid = await _stage_row(uid, aid, job_id)
        monkeypatch.setattr(
            "app.agent.automations.registry.dispatch_via_platform",
            _dispatch_returning({
                "kind": "confirmation_required", "action_id": action_id,
                "summary": "s",
                "expires_at": (datetime.utcnow()
                               + expires_delta).isoformat() + "Z",
            }))
        async with async_session_maker() as db:
            assert await outbox._claim(db, oid)
            await outbox._execute_claimed(db, oid)
        return job_id

    dead = await _park("act-dead", timedelta(minutes=-30))  # past + grace
    live = await _park("act-live", timedelta(hours=23))

    # The GENERIC reaper must leave automation parks to the engine —
    # backdate past its 25h cutoff and prove it abstains.
    async with async_session_maker() as db:
        job = await db.get(BuildJob, dead)
        job.created_at = datetime.utcnow() - timedelta(hours=30)
        await db.commit()
    await job_reaper.sweep_expired_card_parks()
    async with async_session_maker() as db:
        job = await db.get(BuildJob, dead)
        assert job.status == "waiting_on_user"

    closed = await confirm.sweep_expired_confirm_parks()
    assert closed == 1
    async with async_session_maker() as db:
        job = await db.get(BuildJob, dead)
        assert (job.status, job.outcome) == ("cancelled", "skipped")
        assert "nothing was sent" in (job.user_message or "").lower()
        job = await db.get(BuildJob, live)
        assert job.status == "waiting_on_user"
        # The user's silence is not a failure.
        a = await db.get(Automation, aid)
        assert (a.consecutive_failures or 0) == 0


# ── Memory-filtered events ───────────────────────────────────────────


def test_facts_needle_matches_ledger_not_substring():
    from types import SimpleNamespace

    from app.agent.automations.executor import _passes_filter

    # _passes_filter reads only .filter_rules; ValidatedSpec is frozen.
    vspec = SimpleNamespace(filter_rules={"from": ["{{facts.people}}"]})

    facts = {"people": [
        "boss: sarah <sarah@acme.com> — replies urgent",
    ]}
    # The from-header form: token (the email) found inside a fact.
    assert _passes_filter(
        vspec, {"from": "Sarah X <sarah@acme.com>"}, facts)
    assert not _passes_filter(
        vspec, {"from": "spam@nowhere.io"}, facts)
    # Empty ledger = allowlist that admits nobody.
    assert not _passes_filter(
        vspec, {"from": "sarah@acme.com"}, {"people": []})
    assert not _passes_filter(vspec, {"from": "sarah@acme.com"}, None)
    # Literal needles beside a facts needle keep v1 semantics (OR).
    vspec.filter_rules = {"from": ["{{facts.people}}", "@corp.example"]}
    assert _passes_filter(
        vspec, {"from": "who@corp.example"}, {"people": []})


async def test_v2_memory_filter_end_to_end():
    """The whole leg on real rows: a fact recorded through the R29-A
    seam admits its sender through a v2 `{{facts.people}}` needle and
    refuses a stranger — and the needle is intercepted BEFORE the var
    renderer (which would blank the unknown template into a
    match-nothing literal)."""
    from types import SimpleNamespace

    from app.agent.automations import facts as facts_seam
    from app.agent.automations.executor_v2 import _passes_filter_v2
    from app.agent.automations.facts_context import load_facts_context

    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        out = await facts_seam.record(
            db, user_id=uid, automation_id=aid,
            facts=["Boss: Sarah <sarah@acme.com> — replies urgent"],
            category="people", source="user", source_kind="edit",
        )
        assert out["saved"] == 1

    source = SimpleNamespace(filter_rules={"from": ["{{facts.people}}"]})
    async with async_session_maker() as db:
        ctx = await load_facts_context(db, aid, source.filter_rules)
    assert ctx == {"people": [
        "boss: sarah <sarah@acme.com> — replies urgent"]}
    assert _passes_filter_v2(
        source, {"from": "Sarah X <sarah@acme.com>"}, {}, ctx)
    assert not _passes_filter_v2(source, {"from": "spam@x.io"}, {}, ctx)
    # No ctx (or an unreferenced load) still refuses — allowlist rule.
    assert not _passes_filter_v2(
        source, {"from": "sarah@acme.com"}, {}, None)


def test_facts_needle_shapes():
    from app.agent.automations.facts_context import (
        facts_needle_category, referenced_categories,
    )
    assert facts_needle_category("{{facts.people}}") == "people"
    assert facts_needle_category("{{ facts.work }}") == "work"
    assert facts_needle_category("{{var.boss}}") is None
    assert facts_needle_category("plain") is None
    assert referenced_categories(
        {"from": ["{{facts.people}}"], "subject": ["urgent"]}
    ) == {"people"}
    assert referenced_categories(None) == set()


# ── Draft card ───────────────────────────────────────────────────────


async def test_executed_draft_leaves_truthful_card(monkeypatch):
    from app.agent.automations import outbox

    uid = await _mk_user()
    aid = await _mk_automation(uid)
    job_id = await _mk_run(uid, aid)
    oid = await _stage_row(uid, aid, job_id)
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _dispatch_returning({
            "kind": "ok",
            "content": json.dumps({"draft_id": "d-1", "id": "m-77"}),
        }))

    async with async_session_maker() as db:
        assert await outbox._claim(db, oid)
        assert await outbox._execute_claimed(db, oid) == "executed"

    # R31 §4.1: the draft's durable half is its `draft` turn in the
    # thread; the card rides the live frame. It used to ALSO write a
    # day-chat message, so a draft announced itself twice — once where
    # the work happened and once in the main chat.
    assert await _session_messages(uid, aid) == []
    # The honest record is the write's TOOL turn — the verb dictionary's
    # own "Drafted a reply · waiting in Gmail — nothing sent". (The
    # `draft` turn beside it carries the drafted BODY, and this staged
    # row has none.)
    writes = [t for t in await _thread_turns(aid, "tool")
              if t.get("tool_kind") == "write"]
    assert len(writes) == 1
    line = f"{writes[0]['action']} {writes[0]['detail']}".lower()
    assert "draft" in line
    assert "nothing sent" in line or "nothing was sent" in line
    assert "__" not in line

    # The card payload itself is still built correctly — it is what the
    # frame carries.
    from app.agent.automations.draft_card import draft_card_payload
    card = draft_card_payload(
        tool_name="gmail__create_draft", connector_id="gmail",
        staged_payload={"to": "sarah@acme.com", "subject": "Re: contract"},
        result_content={"draft_id": "d-1", "id": "m-77"},
    )
    assert card["provider"] == "gmail"
    assert card["sender"] == "sarah@acme.com"
    assert card["subject"] == "Re: contract"
    assert card["open_url"].endswith("/m-77")


async def test_non_draft_write_gets_no_card(monkeypatch):
    from app.agent.automations import outbox

    uid = await _mk_user()
    aid = await _mk_automation(uid)
    job_id = await _mk_run(uid, aid)
    oid = await _stage_row(
        uid, aid, job_id, tool="slack__send_message", connector="slack",
        payload={"channel": "C1", "text": "hi"})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _dispatch_returning({"kind": "ok", "content": "{}"}))
    async with async_session_maker() as db:
        assert await outbox._claim(db, oid)
        await outbox._execute_claimed(db, oid)
    msgs = await _session_messages(uid, aid)
    assert not any("draft_card" in (m.metadata_json or "") for m in msgs)


# ── Notices + chips ──────────────────────────────────────────────────


async def test_auto_pause_notice_carries_fix_chip():
    from app.agent.automations.sweep import _post_error_notice

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Flaky one")
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        a.last_error = "tool_error: upstream 500"
        await _post_error_notice(db, a)

    # R31 §4.1: the notice lands in the automation's own THREAD. It
    # used to be a day-chat Message plus a `type: "message"` chat frame
    # — an automation speaking in the main chat's own voice. The push
    # is untouched: a paused automation must still reach a closed app.
    assert await _session_messages(uid, aid) == []
    notices = await _thread_turns(aid, "agent")
    assert len(notices) == 1
    text = notices[0]["text"] or ""
    assert text
    # C's sentence, not A's fallback. `auto_pause_body` is ZERO-arg and
    # was called with two, so the bare except swallowed the TypeError
    # and C's copy — written to replace a live string wearing an emoji
    # and markdown bold — had never reached a user.
    from app.agent.automations.notification_templates import auto_pause_body
    assert text == auto_pause_body()
    assert "[[navigate" not in text
    assert "__" not in text


async def test_memory_update_marker_shape():
    from app.agent.automations.session import emit_memory_update

    uid = await _mk_user()
    aid = await _mk_automation(uid)
    async with async_session_maker() as db:
        assert await emit_memory_update(
            db, user_id=uid, automation_id=aid, count=0) is None
        mid = await emit_memory_update(
            db, user_id=uid, automation_id=aid, count=2, title="X")
        assert mid

    # R31 §4.1: `Memory updated · N facts` belongs to the conversation
    # that LEARNED the facts. It appeared in the founder's main chat at
    # 11:17 on 26 August, directly under a thread answer that had leaked
    # there too — two rows, one cause.
    assert await _session_messages(uid, aid) == []
    markers = await _thread_turns(aid, "memory")
    assert len(markers) == 1
    assert markers[0]["count"] == 2
    assert markers[0]["sheet"] == "memory"


def test_serializers_pass_r29_keys_through():
    from app.api.day_chats import _serialize_meta_card
    from app.api.sessions import _message_to_response

    msg = Message(
        id=str(uuid.uuid4()), conversation_id="c", role="assistant",
        content="Memory updated · 2 facts",
        created_at=datetime.utcnow(),
        metadata_json=json.dumps({
            "memory_update": {"count": 2, "at": "2026-08-24T00:00:00Z"},
            "draft_card": {"provider": "gmail", "sender": "s@x.co",
                           "subject": "s", "preview": "p",
                           "open_url": None},
            "fix_chip": {"label": "Fix", "prompt": "Fix it"},
            "pending_action": {"action_id": "a-1", "status": "pending"},
        }),
    )
    for key in ("memory_update", "draft_card", "fix_chip",
                "pending_action"):
        assert _serialize_meta_card(msg, key) is not None, key
    resp = _message_to_response(msg)
    assert resp.memory_update == {"count": 2, "at": "2026-08-24T00:00:00Z"}
    assert resp.draft_card["provider"] == "gmail"
    assert resp.fix_chip["label"] == "Fix"
    assert resp.pending_action["action_id"] == "a-1"


# ── Interview ────────────────────────────────────────────────────────


async def test_session_conversation_yields_prompt_section():
    from app.agent.automations.interview import (
        build_automation_context, prompt_section,
    )
    # The conversation row, without the retired writer: this test is
    # about the prompt section, not about who put a message in it.
    from app.agent.automations.session import resolve_session_conversation

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Morning brief")
    async with async_session_maker() as db:
        await resolve_session_conversation(
            db, user_id=uid, automation_id=aid, title="Morning brief")
        await db.commit()
    async with async_session_maker() as db:
        conv = (await db.execute(
            select(Conversation).where(
                Conversation.user_id == uid,
                Conversation.channel == "automation",
            )
        )).scalars().first()
        ctx = await build_automation_context(db, conv)
    assert ctx and ctx["automation_id"] == aid
    assert ctx["name"] == "Morning brief"
    section = prompt_section(ctx)
    assert "Morning brief" in section
    assert "ONE question at a time" in section
    assert "NEVER sends mail" in section
    assert "__" not in section

    # A plain chat conversation yields nothing.
    async with async_session_maker() as db:
        plain = Conversation(id=str(uuid.uuid4()), user_id=uid,
                             channel="web", is_active=True)
        assert await build_automation_context(db, plain) is None


def _capture_memory_writes(monkeypatch, recorded):
    """Capture fact writes on EITHER seam, so these pins hold on this
    branch (legacy fallback) AND on the merged tree (curator_v2 finds
    the v2 store and writes memory_facts instead — the merge-seam class
    A's integration run caught). Entries: {"facts", "category",
    "source", "source_kind"} with the category in whichever vocabulary
    the firing seam speaks."""
    import sys as _sys
    import types as _types

    async def _add_fact(db, **kw):
        recorded.append({"facts": [kw["text"]], "category": kw["category"],
                         "source": kw.get("source"),
                         "source_kind": "interview"})
        return {"saved": True}

    try:
        from app.services import memory_v2_service as _v2
        monkeypatch.setattr(_v2, "add_fact", _add_fact)
    except ImportError:
        _fake = _types.ModuleType("app.services.memory_v2_service")
        _fake.add_fact = _add_fact
        monkeypatch.setitem(
            _sys.modules, "app.services.memory_v2_service", _fake)
        import app.services as _services_pkg
        monkeypatch.setattr(
            _services_pkg, "memory_v2_service", _fake, raising=False)

    from app.agent.automations import facts as _facts_seam

    async def _record(db, *, user_id, automation_id, facts, category,
                      source, source_kind, run_id=None):
        recorded.append({"facts": list(facts), "category": category,
                         "source": source, "source_kind": source_kind})
        return {"saved": len(facts), "ids": [str(uuid.uuid4())]}

    monkeypatch.setattr(_facts_seam, "record", _record)


async def test_interview_extractor_records_via_seam_and_emits_chip(
        monkeypatch):
    from app.agent.automations import interview

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Draft replies")

    recorded = []
    _capture_memory_writes(monkeypatch, recorded)

    class _Resp:
        content = json.dumps({"facts": [
            {"text": "Boss is Sarah (sarah@acme.com).",
             "category": "people"},
            {"text": "Keep draft replies under three sentences.",
             "category": "preferences"},
            {"text": "Something uncategorizable.", "category": "nope!"},
        ]})

    class _LLM:
        async def complete_with_json(self, **kwargs):
            return _Resp()

    monkeypatch.setattr(
        "app.services.llm_service.get_llm_service", lambda: _LLM())

    ctx = {"automation_id": aid, "name": "Draft replies",
           "rule_text": "watches mail and drafts replies",
           "domain": None, "facts": {}}
    async with async_session_maker() as db:
        n = await interview.extract_and_record_facts(
            db, user_id=uid, ctx=ctx,
            user_text="My boss is Sarah, sarah@acme.com — keep replies short",
            assistant_text="Got it.",
        )
    assert n == 2
    texts = sorted(t for r in recorded for t in r["facts"])
    assert texts == ["Boss is Sarah (sarah@acme.com).",
                     "Keep draft replies under three sentences."]
    # Category vocabulary depends on the firing seam: legacy fallback
    # speaks {people, preferences}; the v2 store {people, your_time}.
    cats = {r["category"] for r in recorded}
    assert cats in ({"people", "preferences"}, {"people", "your_time"})
    assert all(r["source"] == "agent" and r["source_kind"] == "interview"
               for r in recorded)

    # R31 §4.1: the chip is a `memory` turn in the thread that learned
    # the facts, not a row in the main chat.
    assert await _session_messages(uid, aid) == []
    markers = await _thread_turns(aid, "memory")
    assert len(markers) == 1
    assert markers[0]["count"] == 2


async def test_interview_extractor_refuses_status_and_definition_facts(
        monkeypatch):
    """R30 §5.6 (ND-2/ND-3): the two classes GROUND-TRUTH found live in
    prod are refused at the extraction gate — not merely dropped later
    by A's migration — so they can never re-enter."""
    from app.agent.automations import interview

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Morning work brief")

    recorded = []
    _capture_memory_writes(monkeypatch, recorded)

    class _Resp:
        content = json.dumps({"facts": [
            {"text": "Marcus Webb gets same-day answers.",
             "category": "people"},
            {"text": "The Morning work brief is currently paused.",
             "category": "preferences"},
            {"text": ("Has an automation 'Morning work brief': Every day "
                      "at 22:52, check Jira, GitHub, Teams, Gmail and "
                      "Outlook and post to Slack."),
             "category": "preferences"},
        ]})

    class _LLM:
        async def complete_with_json(self, **kwargs):
            return _Resp()

    monkeypatch.setattr(
        "app.services.llm_service.get_llm_service", lambda: _LLM())

    ctx = {"automation_id": aid, "name": "Morning work brief",
           "rule_text": "reads overnight mail and briefs you",
           "domain": None, "facts": {}}
    async with async_session_maker() as db:
        n = await interview.extract_and_record_facts(
            db, user_id=uid, ctx=ctx,
            user_text="Marcus always gets same-day answers from me",
            assistant_text="Noted.",
        )
    assert n == 1
    assert [t for r in recorded for t in r["facts"]] == [
        "Marcus Webb gets same-day answers."]

    # R31 §4.1: the chip is a thread turn.
    assert await _session_messages(uid, aid) == []
    markers = await _thread_turns(aid, "memory")
    assert len(markers) == 1
    assert markers[0]["count"] == 1


async def test_prompt_section_states_status_once_and_hides_raw_errors():
    """R30 §5.4 (D-19): one engine-derived status claim, stated once;
    the raw last_error string never reaches the prompt."""
    from app.agent.automations.interview import prompt_section

    raw_error = "forbidden tool gmail__send_message at step 2"
    section = prompt_section({
        "automation_id": "a1", "name": "Morning work brief",
        "rule_text": "reads overnight mail and briefs you",
        "status": "error", "last_error": raw_error, "facts": {},
    })
    assert raw_error not in section
    assert "gmail__send_message" not in section
    assert "__" not in section
    assert "it tried something automations never do" in section
    assert "ONCE per reply" in section
    assert "never quietly re-run" in section
    assert "check memory first" in section
    assert "never a memory" in section

    paused = prompt_section({
        "automation_id": "a1", "name": "Morning work brief",
        "rule_text": "x", "status": "paused", "facts": {},
    })
    assert "paused by the user" in paused
    assert paused.count("stated once") == 1
