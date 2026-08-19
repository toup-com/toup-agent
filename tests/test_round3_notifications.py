"""Round 3 backend — job-type tag, Live Activity step/completion pushes,
deep-link payload, one card per conversation, the seen signal, and the
F4/F6 source-conflict rules (item 7).

Contract doc: docs/design/round3-notifications.md. Every APNs send is
captured at ``live_activity_service.apns_push.send_live_activity`` — no
network. Behavioural where the seam allows it (the LA lane against a real
sqlite queue), structural only for the loop-integration shape that needs a
live LLM turn to reach.
"""
from __future__ import annotations

import inspect
import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from app.config import settings
from app.db.models import (
    LA_ENDED, LA_STARTED, LiveActivity, LiveActivityDevice,
    NotificationQueue, User, NQ_QUEUED, NQ_SUPPRESSED,
)
from app.services import apns_push
from app.services import live_activity_service as las
from app.services import notification_dispatcher as nd

_BACKEND = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════
# 1. Job-type tag
# ═══════════════════════════════════════════════════════════════════

from app.agent.job_type import classify_job_type, normalize_job_type, JOB_TYPES  # noqa: E402


@pytest.mark.parametrize("title,expected", [
    ("Verify Anthropic's newest model release", "verify"),
    ("Fact-check the claim about GDP", "verify"),
    ("Check whether the flight is delayed", "verify"),
    ("Double-check the totals", "verify"),
    ("Search for a good CRM", "search"),
    ("Find the cheapest flight to Lisbon", "search"),
    ("Look up the capital of Mongolia", "search"),
    ("Lookup DNS for toup.ai", "search"),
    ("Write a cover letter", "write"),
    ("Summarize this article", "write"),
    ("Summarise the meeting notes", "write"),
    ("Draft a reply to Sam", "write"),
    ("Compare React and Vue", "compare"),
    ("iPhone vs Pixel", "compare"),
    ("Claude versus GPT for coding", "compare"),
    ("Book a table for two", "generic"),
    ("", "generic"),
    (None, "generic"),
])
def test_classifier_rules(title, expected):
    assert classify_job_type(title) == expected


def test_classifier_earliest_phrase_wins():
    """Positional precedence — the leading verb is the intent."""
    assert classify_job_type("Compare A vs B and write a summary") == "compare"
    assert classify_job_type("Write a comparison of A vs B") == "write"
    assert classify_job_type("Search the web, then verify the top result") == "search"


def test_classifier_falls_back_to_description_only_when_title_is_generic():
    assert classify_job_type("Task 3", "compare the two vendors") == "compare"
    assert classify_job_type("Verify totals", "compare the two vendors") == "verify"


def test_classifier_output_is_closed_and_normalizer_rejects_junk():
    for t in ("verify", "search", "write", "compare", "generic"):
        assert t in JOB_TYPES
    assert normalize_job_type("VERIFY ") == "verify"
    assert normalize_job_type("agent_task") is None
    assert normalize_job_type(None) is None


def test_create_job_persists_and_returns_the_tag_without_touching_the_column():
    """The tag lives in config_json — BuildJob.job_type is the JobRunner
    HANDLER discriminator and dispatch keys on it."""
    src = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    body = _fn_body(src, "async def _tool_create_job(")
    # Round 8: config_json also carries the turn's answer id for the
    # reconciler; the tag is still the first key.
    assert '_job_cfg: Dict[str, Any] = {"job_type": job_type}' in body
    assert "config_json=_job_cfg" in body
    assert 'job_type="agent_task"' in body, "the handler discriminator is untouched"
    assert '"job_type": job_type' in body, "create_job's response carries the tag"
    assert '"job_type": job_type,' in _fn_body(src, "async def _tool_update_job(")


def _fn_body(src: str, header: str) -> str:
    idx = src.index(header)
    nxt = re.search(r"\n    (?:async )?def ", src[idx + 10:])
    return src[idx: idx + 10 + nxt.start()] if nxt else src[idx:]


# ═══════════════════════════════════════════════════════════════════
# 2/3/4. Producer contract — _notify_job_event
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_notify_job_event_carries_the_round3_fields(monkeypatch):
    from app.agent import subagent_orchestrator as so
    from app.services import agent_notify_client as anc

    calls = []

    async def fake_notify(**kw):
        calls.append(kw)
        return "row"

    monkeypatch.setattr(anc, "notify", fake_notify)
    await so._notify_job_event(
        job_id="job-1", label="Verify the release", kind="progress",
        title="Working on: Verify the release", body="Reading a page…",
        progress=40, priority="low", dedup_suffix="progress",
        chat_id="sess-1", message_id="msg-1", job_type="verify",
        step_name="Reading a page…", steps_done=2, steps_total=5,
    )
    d = calls[0]["data"]
    # ONE CARD PER CONVERSATION — the mission is the conversation.
    assert d["mission_id"] == "chatjob:sess-1"
    assert d["route"] == "chat"
    assert d["job_id"] == "job-1"
    # Deep link.
    assert d["chat_id"] == "sess-1" and d["message_id"] == "msg-1"
    # Card content.
    assert d["job_type"] == "verify" and d["step_name"] == "Reading a page…"
    assert d["steps_done"] == 2 and d["steps_total"] == 5 and d["progress"] == 40
    # Flat scalars only — the ingest validator rejects anything else.
    assert all(isinstance(v, (str, int, float, bool)) for v in d.values())
    assert json.dumps(d).__len__() < 2048
    # Dedup stays PER JOB even though the mission is per conversation.
    assert calls[0]["dedup_key"] == "job-1:progress"


@pytest.mark.asyncio
async def test_notify_job_event_without_a_chat_keeps_the_per_job_card(monkeypatch):
    from app.agent import subagent_orchestrator as so
    from app.services import agent_notify_client as anc

    calls = []

    async def fake_notify(**kw):
        calls.append(kw)

    monkeypatch.setattr(anc, "notify", fake_notify)
    await so._notify_job_event(
        job_id="job-2", label="Nightly digest", kind="mission_started",
        title="t", dedup_suffix="started",
    )
    d = calls[0]["data"]
    assert d["mission_id"] == "job-2" and d["route"] == "mission-control"
    assert "chat_id" not in d and "refresh_if_started" not in d


@pytest.mark.asyncio
async def test_completion_push_carries_preview_and_deferred_end(monkeypatch):
    from app.agent import subagent_orchestrator as so
    from app.services import agent_notify_client as anc

    calls = []

    async def fake_notify(**kw):
        calls.append(kw)

    monkeypatch.setattr(anc, "notify", fake_notify)
    await so._notify_job_event(
        job_id="job-3", label="Compare CRMs", kind="mission_completed",
        title="✅ Done", body="HubSpot wins on price…", progress=100,
        dismiss_after_s=900, dedup_suffix="completed",
        chat_id="s", message_id="m", job_type="compare", step_name="Done",
        steps_done=3, steps_total=3,
        preview="HubSpot wins on price; Salesforce on depth. " * 5,
        end_after_s=so.JOB_CARD_END_AFTER_S,
    )
    d = calls[0]["data"]
    assert len(d["preview"]) <= 120
    assert d["end_after_s"] == so.JOB_CARD_END_AFTER_S
    assert 0 < so.JOB_CARD_END_AFTER_S <= 30, "a BRIEF delay"


def test_update_job_defers_the_terminal_push_for_this_turns_jobs():
    """`update_job(status=completed)` runs BEFORE the model writes the
    answer, so it cannot carry the response preview. For this turn's
    jobs it sends a 100% 'finishing' progress update; the runner's
    finalizer — which has final_text + the message id — sends the
    mission_completed with the preview + the deferred end. Older jobs
    (not this turn's) keep the immediate terminal push."""
    src = (_BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    body = _fn_body(src, "async def _tool_update_job(")
    assert 'if job.status == "completed" and _this_turns_job:' in body
    branch = body[body.index('if job.status == "completed" and _this_turns_job:'):]
    branch = branch[: branch.index('elif job.status == "completed":')]
    assert 'kind="progress"' in branch and "progress=100" in branch
    assert 'kind="mission_completed"' not in branch
    assert "peek_created_job_ids()" in body


def test_runner_finalizer_sends_preview_message_id_and_deferred_end():
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    idx = src.index("_created_job_ids = self.tools.take_created_job_ids()")
    blk = src[idx: src.index("return AgentResponse(", idx)]
    assert "_preview = " in blk and "final_text" in blk
    assert "message_id=asst_message_id" in blk
    assert "preview=_preview" in blk
    assert "end_after_s=JOB_CARD_END_AFTER_S" in blk
    assert 'kind="mission_completed"' in blk
    # It closes cards of jobs the model already marked completed too
    # (their tool-level push was deferred here).
    assert '_r[2] != "completed"' in blk


def test_runner_hands_the_answer_id_to_the_tool_context():
    """The deep link's message_id is pre-minted; the executor must get
    it with the session id, not learn it at the end."""
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    # Round 4 added the turn's receipt time as a third argument (relative
    # reminders count from it); 2026-08-19 anchored it to the CHANNEL's
    # receipt stamp (received_at) with run() entry as the fallback — the
    # answer id still rides with the session.
    assert "turn_started_at=(received_at if received_at else start)" in src
    assert "self.tools.set_session_id(" in src


def test_cancel_path_ends_cards_of_jobs_completed_mid_turn():
    """The deferral moved the terminal push to the happy path; the
    interrupted-turn sweep must cover the jobs the model completed
    before the turn died or their card sits at 100% until stale."""
    from app.agent.agent_runner import AgentRunner
    src = inspect.getsource(AgentRunner._close_interrupted_jobs)
    assert '_BJ.status == "completed"' in src
    assert 'kind="mission_completed"' in src


# ═══════════════════════════════════════════════════════════════════
# apns_push — content-state extras
# ═══════════════════════════════════════════════════════════════════


def test_content_state_extras_whitelisted_and_capped():
    cs = apns_push._content_state(
        "T", "S", 0.4, extra={
            "jobType": "verify", "stepName": "  Reading   a page ", "stepsDone": 2,
            "stepsTotal": 5, "percent": 40, "preview": "x" * 500,
            "chatId": "sess", "messageId": "msg", "evil": "no", "fired": True,
            "stepsDone_bool": True,
        },
    )
    assert cs["jobType"] == "verify" and cs["stepName"] == "Reading a page"
    assert cs["stepsDone"] == 2 and cs["stepsTotal"] == 5 and cs["percent"] == 40
    assert len(cs["preview"]) == 120
    assert cs["chatId"] == "sess" and cs["messageId"] == "msg"
    assert "evil" not in cs and "stepsDone_bool" not in cs
    assert cs.get("fired") is None, "extras cannot set fired"
    assert cs["progress"] == 0.4


def test_content_state_percent_follows_the_clamped_fraction():
    # The lane's never-backwards clamp raises the fraction; percent must
    # not contradict it on the same card.
    cs = apns_push._content_state("T", None, 0.6, extra={"percent": 40})
    assert cs["progress"] == 0.6 and cs["percent"] == 60
    # bool is not an int for our purposes
    cs = apns_push._content_state("T", None, None, extra={"percent": True})
    assert "percent" not in cs


def test_content_state_without_extras_is_byte_identical_to_before():
    """The pre-Round-3 Swift contract test pins the exact key set; no
    extras → no new keys, so widgets in the field decode as before."""
    cs = apns_push._content_state("T", "S", 0.2)
    assert set(cs) == {"title", "subtitle", "progress"}


def test_builders_accept_extra_on_start_update_and_end():
    x = {"jobType": "search", "chatId": "c", "messageId": "m", "percent": 100}
    s = apns_push.build_start_payload(mission_id="chatjob:c", title="T", timestamp=1, extra=x)
    u = apns_push.build_update_payload(title="T", progress=1.0, timestamp=1, extra=x)
    e = apns_push.build_end_payload(title="T", progress=1.0, timestamp=1, extra=x)
    for p in (s, u, e):
        cs = p["aps"]["content-state"]
        assert cs["jobType"] == "search" and cs["chatId"] == "c" and cs["messageId"] == "m"
    assert len(json.dumps(s)) < 4096


# ═══════════════════════════════════════════════════════════════════
# LA lane — behavioural (sqlite queue, captured APNs)
# ═══════════════════════════════════════════════════════════════════


async def _mk_user() -> str:
    from app.db import async_session_maker
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"r3-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="R3",
            timezone="America/Toronto",
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
            },
        ))
        await db.commit()
    return user_id


async def _mk_device(user_id: str) -> str:
    from app.db import async_session_maker

    device_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(LiveActivityDevice(
            id=device_id, user_id=user_id,
            push_to_start_token=uuid.uuid4().hex + uuid.uuid4().hex,
            apns_environment="development", created_at=datetime.utcnow(),
        ))
        await db.commit()
    return device_id


async def _enqueue(user_id: str, **overrides) -> str:
    from app.db import async_session_maker

    row_id = str(uuid.uuid4())
    fields = dict(
        id=row_id, user_id=user_id, source="agent",
        event_kind="progress", title="Working on: T", body="Reading a page…",
        priority="low", idempotency_key=f"idem-{row_id}",
        status=NQ_QUEUED, created_at=datetime.utcnow(),
        data_json={"mission_id": "chatjob:s1", "mission_title": "T", "progress": 40},
    )
    fields.update(overrides)
    async with async_session_maker() as db:
        db.add(NotificationQueue(**fields))
        await db.commit()
    return row_id


async def _dispatch(row_id: str, now: datetime = None) -> str:
    from app.db import async_session_maker

    now = now or datetime.utcnow()
    async with async_session_maker() as db:
        claimed = await nd._claim_batch(db, now)
        assert row_id in claimed, "row not claimable"
        return await nd._dispatch_row(db, row_id, now)


def _patch_apns(monkeypatch, sent: list, status: int = 200, reason: str = ""):
    async def fake_send(token, payload, *, environment="development", priority=10):
        sent.append({"token": token, "payload": payload,
                     "environment": environment, "priority": priority})
        return status, reason

    monkeypatch.setattr(las.apns_push, "send_live_activity", fake_send)
    monkeypatch.setattr(settings, "apns_key_b64", "eA==")
    monkeypatch.setattr(settings, "apns_key_id", "KEY123")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123")


async def _la_row(user_id: str, mission_id: str) -> LiveActivity:
    from app.db import async_session_maker
    async with async_session_maker() as db:
        return (await db.execute(
            select(LiveActivity).where(
                LiveActivity.user_id == user_id, LiveActivity.mission_id == mission_id,
            )
        )).scalar_one()


_JOB_DATA = {
    "mission_id": "chatjob:s1", "mission_title": "Verify the release",
    "kind": "job", "job_id": "job-A", "route": "chat", "urgent": True,
    "chat_id": "s1", "message_id": "m1", "job_type": "verify",
    "step_name": "Searching…", "steps_done": 0, "steps_total": 3,
    "timer_end_ms": 1_900_000_000_000, "refresh_if_started": True,
}


@pytest.mark.asyncio
async def test_start_carries_deep_link_params_and_extras(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    row = await _enqueue(
        user_id, event_kind="mission_started", title="🛠 Working on: Verify",
        priority="default", data_json=dict(_JOB_DATA),
    )
    assert await _dispatch(row) == "sent"
    aps = sent[0]["payload"]["aps"]
    assert aps["event"] == "start"
    assert aps["attributes"]["deepLinkUrl"] == (
        "toup://chat?mission=chatjob:s1&chat_id=s1&message_id=m1"
    )
    cs = aps["content-state"]
    assert cs["jobType"] == "verify" and cs["chatId"] == "s1" and cs["messageId"] == "m1"
    assert cs["stepName"] == "Searching…" and cs["stepsDone"] == 0 and cs["stepsTotal"] == 3
    assert "timerEndDateInMilliseconds" in cs


@pytest.mark.asyncio
async def test_deep_link_ids_are_validated_not_passed_through(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    bad = dict(_JOB_DATA, chat_id="s1 evil", message_id="x" * 65)
    row = await _enqueue(user_id, event_kind="mission_started", priority="default", data_json=bad)
    await _dispatch(row)
    aps = sent[0]["payload"]["aps"]
    assert aps["attributes"]["deepLinkUrl"] == "toup://chat?mission=chatjob:s1"
    assert "chatId" not in aps["content-state"] and "messageId" not in aps["content-state"]


@pytest.mark.asyncio
async def test_progress_update_carries_step_and_percent(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    start = await _enqueue(user_id, event_kind="mission_started", priority="default",
                           data_json=dict(_JOB_DATA))
    await _dispatch(start)
    sent.clear()
    prog = await _enqueue(user_id, data_json=dict(
        _JOB_DATA, progress=67, step_name="Reading a page…", steps_done=2,
        timer_end_ms=None, refresh_if_started=None,
    ))
    assert (await _dispatch(prog)).startswith("suppressed:progress_in_app_only")
    aps = sent[0]["payload"]["aps"]
    assert aps["event"] == "update"
    cs = aps["content-state"]
    assert cs["progress"] == 0.67 and cs["percent"] == 67
    assert cs["stepName"] == "Reading a page…" and cs["stepsDone"] == 2 and cs["stepsTotal"] == 3
    assert cs["chatId"] == "s1" and cs["messageId"] == "m1" and cs["jobType"] == "verify"


@pytest.mark.asyncio
async def test_completed_shows_state_then_ends_after_a_delay(monkeypatch):
    """Item 2: alerting update with ✓ + preview NOW; the end is a chained
    queue row `end_after_s` out — no in-process sleep, no immediate end."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    start = await _enqueue(user_id, event_kind="mission_started", priority="default",
                           data_json=dict(_JOB_DATA))
    await _dispatch(start)
    sent.clear()

    now = datetime.utcnow()
    done = await _enqueue(
        user_id, event_kind="mission_completed", title="✅ Done: Verify",
        body="Fable 5 is the most capable…", priority="default",
        data_json=dict(
            _JOB_DATA, progress=100, step_name="Done", steps_done=3,
            preview="Fable 5 is Anthropic's most capable GA model.",
            dismiss_after_s=900, end_after_s=8, timer_end_ms=None,
            refresh_if_started=None,
        ),
    )
    out = await _dispatch(done, now)
    assert out == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update"], f"expected ONE alerting update, no end yet: {events}"
    upd = sent[0]["payload"]["aps"]
    assert upd["alert"]["title"] == "✅ Done: Verify"
    cs = upd["content-state"]
    assert cs["subtitle"] == "Completed ✓" and cs["progress"] == 1.0 and cs["percent"] == 100
    assert cs["preview"].startswith("Fable 5 is Anthropic")
    assert cs["stepName"] == "Done" and cs["stepsDone"] == 3

    # Card still alive (not ended) — the end is booked, not sent.
    la = await _la_row(user_id, "chatjob:s1")
    assert la.status == LA_STARTED
    from app.db import async_session_maker
    async with async_session_maker() as db:
        end_row = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key == "la-end:chatjob:s1:job-A",
            )
        )).scalar_one()
    assert end_row.status == NQ_QUEUED and end_row.source == "platform"
    assert end_row.data_json["end_only"] is True and end_row.data_json["la_only"] is True
    assert end_row.data_json["silent"] is True and end_row.data_json["urgent"] is True
    assert timedelta(seconds=7) <= (end_row.scheduled_for - now) <= timedelta(seconds=9)
    assert end_row.dedup_key is None, "the dedup window must not swallow the end"

    # Not claimable before its time…
    async with async_session_maker() as db:
        assert end_row.id not in await nd._claim_batch(db, now + timedelta(seconds=3))
    # …then it fires: ONE bannerless end, card ended, final state intact.
    sent.clear()
    out2 = await _dispatch(end_row.id, now + timedelta(seconds=9))
    assert out2 == "sent"
    assert [s["payload"]["aps"]["event"] for s in sent] == ["end"]
    end = sent[0]["payload"]["aps"]
    assert "alert" not in end
    assert end["content-state"]["subtitle"] == "Completed ✓"
    assert end["content-state"]["preview"].startswith("Fable 5")
    assert end["dismissal-date"] >= int((now + timedelta(seconds=9)).timestamp()) + 890
    la = await _la_row(user_id, "chatjob:s1")
    assert la.status == LA_ENDED

    # Retrying the original completed row books nothing twice.
    async with async_session_maker() as db:
        r = await db.get(NotificationQueue, done)
        n = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key.like("la-end:chatjob:s1:%"),
            )
        )).scalars().all()
        assert len(n) == 1


@pytest.mark.asyncio
async def test_completed_without_end_after_s_still_ends_immediately(monkeypatch):
    """Control: producers that did not ask for a linger keep the
    pre-Round-3 update+end pair (missions, reminders, sub-agents)."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    start = await _enqueue(user_id, event_kind="mission_started", priority="default",
                           data_json={"mission_id": "job-Z", "mission_title": "Z"})
    await _dispatch(start)
    sent.clear()
    done = await _enqueue(user_id, event_kind="mission_completed", priority="default",
                          title="Done", data_json={"mission_id": "job-Z", "progress": 100})
    await _dispatch(done)
    assert [s["payload"]["aps"]["event"] for s in sent] == ["update", "end"]


@pytest.mark.asyncio
async def test_second_job_in_same_chat_refreshes_the_card_not_a_new_one(monkeypatch):
    """Item 4: same conversation → same activity. The second job's start
    UPDATES the live card (event=update, new title, alert, progress reset,
    started_at bumped) — never a second push-to-start, never 'Superseded'."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    t0 = datetime.utcnow() - timedelta(seconds=60)
    a = await _enqueue(user_id, event_kind="mission_started", priority="default",
                       title="🛠 Working on: A", created_at=t0, data_json=dict(_JOB_DATA))
    await _dispatch(a, t0)
    la = await _la_row(user_id, "chatjob:s1")
    assert la.status == LA_STARTED
    # A progressed to 100%.
    from app.db import async_session_maker
    async with async_session_maker() as db:
        r = await db.get(LiveActivity, la.id)
        r.last_progress = 100
        await db.commit()
    sent.clear()

    t1 = datetime.utcnow()
    b = await _enqueue(
        user_id, event_kind="mission_started", priority="default",
        title="🛠 Working on: B", body="second job", created_at=t1,
        data_json=dict(_JOB_DATA, job_id="job-B", mission_title="Compare CRMs",
                       job_type="compare", message_id="m2", steps_total=2),
    )
    assert await _dispatch(b, t1) == "sent"
    assert len(sent) == 1
    aps = sent[0]["payload"]["aps"]
    assert aps["event"] == "update", "a refresh, not a second start"
    assert aps["content-state"]["title"] == "Compare CRMs"
    assert aps["content-state"]["jobType"] == "compare"
    assert aps["content-state"]["messageId"] == "m2"
    assert aps["alert"]["title"] == "🛠 Working on: B"
    la2 = await _la_row(user_id, "chatjob:s1")
    assert la2.id == la.id, "same activity row"
    assert la2.status == LA_STARTED
    assert la2.last_progress == 0, "the new job starts at zero — never pinned at A's 100%"
    assert la2.started_at >= t1 - timedelta(seconds=1)
    # No 'Superseded' end, no second start.
    assert not any(s["payload"]["aps"]["event"] in ("start", "end") for s in sent)


@pytest.mark.asyncio
async def test_a_retry_of_the_same_start_does_not_refresh(monkeypatch):
    """At-least-once retries of the start that opened the card still
    dedup — only a row NEWER than the card's start refreshes it."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    t0 = datetime.utcnow()
    a = await _enqueue(user_id, event_kind="mission_started", priority="default",
                       created_at=t0 - timedelta(seconds=5), data_json=dict(_JOB_DATA))
    await _dispatch(a, t0)
    sent.clear()
    # Same row re-queued (simulate a partial-failure retry): created BEFORE
    # the card started.
    from app.db import async_session_maker
    async with async_session_maker() as db:
        r = await db.get(NotificationQueue, a)
        r.status = NQ_QUEUED
        r.claimed_at = None
        r.scheduled_for = None
        await db.commit()
    out = await _dispatch(a, t0 + timedelta(seconds=10))
    assert sent == []
    assert out.startswith("suppressed") or out == "sent" or out == "retry_scheduled"


@pytest.mark.asyncio
async def test_new_job_cancels_the_previous_jobs_pending_end(monkeypatch):
    """A's deferred end must not close B's card out from under it."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    t0 = datetime.utcnow() - timedelta(seconds=30)
    a = await _enqueue(user_id, event_kind="mission_started", priority="default",
                       created_at=t0, data_json=dict(_JOB_DATA))
    await _dispatch(a, t0)
    done = await _enqueue(user_id, event_kind="mission_completed", priority="default",
                          title="Done", created_at=t0 + timedelta(seconds=5),
                          data_json=dict(_JOB_DATA, progress=100, end_after_s=8,
                                         timer_end_ms=None, refresh_if_started=None))
    await _dispatch(done, t0 + timedelta(seconds=5))
    from app.db import async_session_maker
    async with async_session_maker() as db:
        end_row = (await db.execute(select(NotificationQueue).where(
            NotificationQueue.idempotency_key == "la-end:chatjob:s1:job-A"))).scalar_one()
        assert end_row.status == NQ_QUEUED
    t1 = datetime.utcnow()
    b = await _enqueue(user_id, event_kind="mission_started", priority="default",
                       created_at=t1, data_json=dict(_JOB_DATA, job_id="job-B"))
    await _dispatch(b, t1)
    async with async_session_maker() as db:
        end_row = await db.get(NotificationQueue, end_row.id)
        assert end_row.status == NQ_SUPPRESSED
        assert end_row.channels_json["policy"]["suppressed"] == "end_superseded"


@pytest.mark.asyncio
async def test_chatjob_cards_yield_to_a_live_reminder_countdown(monkeypatch):
    """REMINDER WINS extends to the conversation job card."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_device(user_id)
    from app.db import async_session_maker
    async with async_session_maker() as db:
        db.add(LiveActivity(id=str(uuid.uuid4()), user_id=user_id,
                            mission_id="reminder:r1", device_id=device_id,
                            status=LA_STARTED, started_at=datetime.utcnow()))
        await db.commit()
    row = await _enqueue(user_id, event_kind="mission_started", priority="default",
                         data_json=dict(_JOB_DATA))
    await _dispatch(row)
    assert sent == [], "no start, no preempt end"
    assert "chatjob:" in las._NEVER_PREEMPT_REMINDER_PREFIXES


@pytest.mark.asyncio
async def test_stale_sweep_ages_chatjob_rows_like_chat_turns():
    user_id = await _mk_user()
    device_id = await _mk_device(user_id)
    from app.db import async_session_maker
    old = datetime.utcnow() - timedelta(minutes=45)
    async with async_session_maker() as db:
        db.add(LiveActivity(id=str(uuid.uuid4()), user_id=user_id,
                            mission_id="chatjob:old", device_id=device_id,
                            status=LA_STARTED, started_at=old))
        await db.commit()
        swept = await las.sweep_stale_activities(db, datetime.utcnow(), user_id=user_id)
    assert swept == 1


# ═══════════════════════════════════════════════════════════════════
# 5. Seen signal
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_seen_endpoint_ends_the_conversation_card_and_pending_rows(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_device(user_id)
    start = await _enqueue(user_id, event_kind="mission_started", priority="default",
                           data_json=dict(_JOB_DATA))
    await _dispatch(start)
    # A pending deferred end + a pending progress row for the mission.
    end_id = await _enqueue(user_id, event_kind="mission_completed", source="platform",
                            idempotency_key="la-end:chatjob:s1:job-A",
                            scheduled_for=datetime.utcnow() + timedelta(seconds=8),
                            data_json=dict(_JOB_DATA, end_only=True, la_only=True))
    prog_id = await _enqueue(user_id, data_json=dict(_JOB_DATA, progress=50))
    sent.clear()

    from app.api.live_activity_devices import LiveActivitySeen, seen_live_activity
    from app.db import async_session_maker

    class _U:  # get_current_user stand-in
        id = user_id

    async with async_session_maker() as db:
        out = await seen_live_activity(
            LiveActivitySeen(chat_id="s1", message_id="m1"), current_user=_U(), db=db,
        )
    assert out["ok"] and out["ended"] == 1 and out["missions"] == ["chatjob:s1"]
    assert out["suppressed"] == 2
    assert [s["payload"]["aps"]["event"] for s in sent] == ["end"]
    assert sent[0]["payload"]["aps"]["dismissal-date"] < int(datetime.utcnow().timestamp())
    la = await _la_row(user_id, "chatjob:s1")
    assert la.status == LA_ENDED
    async with async_session_maker() as db:
        for rid in (end_id, prog_id):
            r = await db.get(NotificationQueue, rid)
            assert r.status == NQ_SUPPRESSED
            assert r.channels_json["policy"]["suppressed"] == "seen"
    # Idempotent.
    async with async_session_maker() as db:
        out2 = await seen_live_activity(
            LiveActivitySeen(chat_id="s1"), current_user=_U(), db=db,
        )
    assert out2["ended"] == 0


@pytest.mark.asyncio
async def test_seen_may_end_the_turn_card_but_never_a_reminder(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_device(user_id)
    from app.db import async_session_maker
    async with async_session_maker() as db:
        for mid in ("chatturn:abc", "reminder:r1"):
            db.add(LiveActivity(id=str(uuid.uuid4()), user_id=user_id, mission_id=mid,
                                device_id=device_id, status=LA_STARTED,
                                started_at=datetime.utcnow()))
        await db.commit()
    from app.api.live_activity_devices import LiveActivitySeen, seen_live_activity

    class _U:
        id = user_id

    async with async_session_maker() as db:
        out = await seen_live_activity(
            LiveActivitySeen(chat_id="s9", mission_id="reminder:r1"), current_user=_U(), db=db,
        )
    assert out["missions"] == ["chatjob:s9"] and out["ended"] == 0
    async with async_session_maker() as db:
        out = await seen_live_activity(
            LiveActivitySeen(chat_id="s9", mission_id="chatturn:abc"), current_user=_U(), db=db,
        )
    assert out["ended"] == 1 and "chatturn:abc" in out["missions"]
    assert (await _la_row(user_id, "reminder:r1")).status == LA_STARTED


def test_seen_route_is_mounted_beside_ack():
    from app.api.live_activity_devices import router
    paths = {r.path for r in router.routes}
    assert "/devices/live-activity/seen" in paths
    assert "/devices/live-activity/ack" in paths


# ═══════════════════════════════════════════════════════════════════
# Ingest fast lane — the job-card start rides it
# ═══════════════════════════════════════════════════════════════════


def test_fast_lane_covers_conversation_job_starts():
    src = inspect.getsource(__import__("app.api.agent_notify", fromlist=["x"]))
    assert 'body.event_kind == "mission_started"' in src
    assert 'get("refresh_if_started")' in src
    assert "notification_progress_fastlane_enabled" in src


# ═══════════════════════════════════════════════════════════════════
# ws_chat chat-turn pushes carry the deep link
# ═══════════════════════════════════════════════════════════════════


def test_chat_turn_pushes_carry_chat_id():
    src = (_BACKEND / "app" / "api" / "ws_chat.py").read_text()
    assert '_answer_data["chat_id"] = response.session_id' in src
    assert '_wc_data["chat_id"] = session_id' in src
    assert '_fail_data["chat_id"] = session_id' in src
    assert 'chat_id=session_id,' in src  # the progress emitter


# ═══════════════════════════════════════════════════════════════════
# 7. Item 7 — source-conflict rules (the Fable 5 / Opus 5 flip-flop)
# ═══════════════════════════════════════════════════════════════════

from app.agent.source_conflict import (  # noqa: E402
    SOURCE_CONFLICT_RULES, build_turn_rules_message, wants_source_conflict_rules,
)

_INCIDENT_HISTORY = [
    {"role": "user", "content": "What is the most capable AI model right now?"},
    {"role": "assistant", "content": (
        "Anthropic's most capable generally available model is Claude Fable 5 "
        "(per Anthropic's models overview); OpenAI's flagship is GPT-5.6."
    )},
]


@pytest.mark.parametrize("msg", [
    # the 12:40 turn — the same question again
    "What is the most capable model?",
    "which is the strongest anthropic model",
    # a challenge to the earlier answer
    "Are you sure? I read Opus 5 is the strongest.",
    "Check again, that's wrong",
    "verify that against official sources",
    "which is it — Fable 5 or Opus 5?",
    # recency / release shaped
    "what's the newest claude model",
    "latest gpt version?",
    # who-is shaped
    "who is the current CEO of OpenAI",
])
def test_gate_fires_on_the_incident_shapes(msg):
    assert wants_source_conflict_rules(msg, [])


def test_gate_fires_on_a_follow_up_to_a_prior_superlative_claim():
    # "what about anthropic" carries no superlative itself; the topic is
    # live because the assistant just made the claim.
    assert wants_source_conflict_rules("what about anthropic", _INCIDENT_HISTORY)
    assert wants_source_conflict_rules("and google?", _INCIDENT_HISTORY)


@pytest.mark.parametrize("msg", [
    "what's the weather in Lisbon",
    "remind me to stretch at 5",
    "write a haiku about rain",
    "how much is 17 * 23",
    "thanks!",
])
def test_gate_stays_quiet_on_ordinary_turns(msg):
    assert not wants_source_conflict_rules(msg, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hey! What can I do for you?"},
    ])


def test_gate_lookback_is_bounded_and_never_raises():
    old = [{"role": "assistant", "content": "the most capable model is X"}]
    recent = [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "sure"},
              {"role": "user", "content": "ok"}, {"role": "assistant", "content": "sure"}]
    assert not wants_source_conflict_rules("thanks", old + recent, lookback=2)
    assert wants_source_conflict_rules("thanks", old + recent, lookback=3)
    assert wants_source_conflict_rules(None, None) is False
    assert wants_source_conflict_rules("most capable", [{"role": "assistant", "content": None}])


def test_rules_encode_the_three_fixes():
    r = SOURCE_CONFLICT_RULES
    # 1. vendor positioning wins over third-party benchmarks
    assert "VENDOR POSITIONING WINS" in r
    assert "NEVER overrides it" in r
    for idx in ("Artificial Analysis", "LMArena"):
        assert idx in r
    # 2. third-party = benchmark result, never "official"
    assert "THIRD-PARTY RESULT IS A BENCHMARK RESULT" in r
    assert 'never as "official"' in r
    # 3. no retraction without an official basis
    assert "NO RETRACTION WITHOUT AN OFFICIAL BASIS" in r
    assert "explicitly contradicts it" in r
    # consistency across the conversation
    assert "repeat that answer unless rule 3 is satisfied" in r
    assert r.startswith("<turn_rules>") and r.rstrip().endswith("</turn_rules>")


def test_rules_ride_the_non_cached_slot_not_the_system_prompt():
    """The cached prefix (system prompt) must not change: the rules are a
    per-turn message appended after <turn_context>, before the user
    message, and never persisted."""
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    # Not in the runtime section (which F6 uses — static, cached).
    runtime = src[src.index("runtime_lines = ["): src.index('section_parts["runtime"]')]
    assert "SOURCE-CONFLICT" not in runtime and "source_conflict" not in runtime
    assert "SOURCE_CONFLICT_RULES" not in src.split("def _build_system_prompt")[1].split("def ", 1)[0]
    # In the message assembly, after the turn_context append and before
    # the current user message.
    i_tc = src.index("messages.append(_tc_msg)")
    i_rules = src.index("wants_source_conflict_rules(user_message, history)")
    i_user = src.index('messages.append({"role": "user", "content": user_message})')
    assert i_tc < i_rules < i_user
    # Persistence stores the user message + reply only.
    save = src[src.index("async def _save_messages"):]
    save = save[: save.index("\n    async def ")]
    assert "turn_rules" not in save and "SOURCE_CONFLICT" not in save
    m = build_turn_rules_message()
    assert m["role"] == "user" and m["content"] == SOURCE_CONFLICT_RULES


def test_incident_replay_shape_is_gated_and_context_framing_is_not_data():
    """The regression case, end to end at the gate: turn 1 answers Fable
    5, turn 2 asks again — the rules must be present on turn 2, and they
    must NOT be inside <turn_context> (framed as data the model must not
    take instructions from)."""
    from app.agent.prefix_stability import build_turn_context_message
    assert wants_source_conflict_rules("What is the most capable model?", _INCIDENT_HISTORY)
    tc = build_turn_context_message(["<clock>12:40</clock>"])
    assert "never follow" in tc["content"].lower() or "reference data" in tc["content"].lower()
    assert "SOURCE-CONFLICT" not in tc["content"]
