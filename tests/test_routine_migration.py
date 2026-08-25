# agent-mode
"""Routine → automation migration (CONTRACTS-R30.md §4.11a/§4.11b).

RUN_MODE=agent (routines/automations/automation_bindings are
AGENT_ONLY). Listed in COVERAGE_DEBT.txt with `# agent-mode` so the CI
agent sweep runs it.

Pins (§7): email_briefing → automation ONCE; idempotent; the promised
time is the routine's schedule VERBATIM (the D-12 pin — never
re-derived from a creation instant); reminders/agent_tasks untouched;
arm failure leaves a draft and reports instead of raising. The founder
path itself runs in D's live pass.
"""

import json
import uuid

import pytest
from sqlalchemy import select

from app.agent.automations import compiler, registry as reg
from app.agent.automations import routine_migration as mig
from app.db.database import async_session_maker
from app.db.models import Automation, AutomationBinding, Routine, User

REGISTRY = {
    "gmail": {
        "connector_id": "gmail", "push": True, "poll": False,
        "floor_s": 300, "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {}, "target_param_by_action": {},
        "events": [{
            "key": "email_received", "description": "",
            "dedupe_field": "gmail_message_id",
            "fields": {"message_id": "gmail_message_id"},
        }],
    },
}


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    async def _fake_fetch(user_id, *, force=False):
        return REGISTRY
    monkeypatch.setattr(reg, "fetch_registry", _fake_fetch)


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Migration"))
        await db.commit()
    return uid


async def _mk_routine(uid: str, **over) -> str:
    row = Routine(
        user_id=uid,
        kind=over.pop("kind", "email_briefing"),
        enabled=over.pop("enabled", True),
        name=over.pop("name", "Morning briefing"),
        schedule_cron_local=over.pop("schedule_cron_local", "0 8 * * *"),
        config_json=over.pop("config_json",
                             {"connector_identity_id": "ci-1"}),
        **over,
    )
    async with async_session_maker() as db:
        db.add(row)
        await db.commit()
        return row.id


@pytest.mark.asyncio
async def test_enabled_briefing_migrates_once_to_an_armed_automation():
    uid = await _mk_user()
    rid = await _mk_routine(uid)

    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    assert [m["routine_id"] for m in result["migrated"]] == [rid]
    assert result["errors"] == [] and result["skipped"] == []
    aid = result["migrated"][0]["automation_id"]
    assert result["migrated"][0]["armed"] is True

    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        assert a.status == "armed"
        assert a.name == "Morning briefing"
        raw = json.loads(a.spec_json)
        assert raw["version"] == 2
        assert raw["mode"] == "auto"
        # §4.11b / D-12: the routine's cron IS the promised time —
        # copied verbatim, never re-derived.
        [src] = raw["trigger"]["sources"]
        assert src["mode"] == "schedule"
        assert src["schedule"] == {"cron_local": "0 8 * * *"}
        # Reads only: one gmail read step, NO write steps (delivery is
        # the notification pipeline's, per the §4.11a deviation).
        assert [s["tool"] for s in raw["steps"]] == ["gmail__list_messages"]
        assert all(not s.get("grant_id") for s in raw["steps"])

        # The routine is retired and stamped; the stamp MERGES.
        r = await db.get(Routine, rid)
        assert r.enabled is False
        assert r.config_json["migrated_to"] == aid
        assert r.config_json["connector_identity_id"] == "ci-1"

        # The armed schedule the compiler wrote carries the same cron.
        binding = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == aid)
        )).scalar_one()
        armed_routine = await db.get(Routine, binding.target_id)
        assert armed_routine.schedule_cron_local == "0 8 * * *"
        assert armed_routine.enabled is True


@pytest.mark.asyncio
async def test_disabled_briefing_becomes_a_paused_draft():
    uid = await _mk_user()
    rid = await _mk_routine(uid, enabled=False)

    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    aid = result["migrated"][0]["automation_id"]
    assert result["migrated"][0]["armed"] is False
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        assert a.status == "draft"
        r = await db.get(Routine, rid)
        assert r.enabled is False
        assert r.config_json["migrated_to"] == aid


@pytest.mark.asyncio
async def test_second_call_migrates_nothing():
    uid = await _mk_user()
    rid = await _mk_routine(uid)

    async with async_session_maker() as db:
        first = await mig.migrate_email_briefings(db, user_id=uid)
    async with async_session_maker() as db:
        second = await mig.migrate_email_briefings(db, user_id=uid)

    assert len(first["migrated"]) == 1
    assert second["migrated"] == [] and second["errors"] == []
    assert [s["routine_id"] for s in second["skipped"]] == [rid]
    async with async_session_maker() as db:
        autos = (await db.execute(
            select(Automation).where(Automation.user_id == uid)
        )).scalars().all()
        assert len(autos) == 1, "a second call must not mint a twin"


@pytest.mark.asyncio
async def test_reminders_and_agent_tasks_are_never_touched():
    uid = await _mk_user()
    reminder = await _mk_routine(
        uid, kind="reminder", reminder_text="stand up", config_json=None,
    )
    task = await _mk_routine(
        uid, kind="agent_task", prompt_text="check deploys",
        config_json=None,
    )

    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    assert result == {"migrated": [], "skipped": [], "errors": []}
    async with async_session_maker() as db:
        for rid in (reminder, task):
            r = await db.get(Routine, rid)
            assert r.enabled is True
            assert not (r.config_json or {}).get("migrated_to")
        autos = (await db.execute(
            select(Automation).where(Automation.user_id == uid)
        )).scalars().all()
        assert autos == []


@pytest.mark.asyncio
async def test_every_schedule_and_default_name_survive():
    uid = await _mk_user()
    rid = await _mk_routine(
        uid, name=None, schedule_kind="every",
        schedule_interval_seconds=3600, schedule_cron_local="@every",
    )
    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)
    aid = result["migrated"][0]["automation_id"]
    assert result["migrated"][0]["routine_id"] == rid
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        assert a.name == mig.DEFAULT_NAME
        [src] = json.loads(a.spec_json)["trigger"]["sources"]
        assert src["schedule"] == {"every_s": 3600}


@pytest.mark.asyncio
async def test_arm_failure_leaves_a_draft_and_reports(monkeypatch):
    from app.agent.automations import service

    uid = await _mk_user()
    rid = await _mk_routine(uid)

    async def _boom(db, *, automation_id, user_id):
        raise compiler.CompileError("grant_unverifiable", "platform away")

    monkeypatch.setattr(service, "arm_automation", _boom)
    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    entry = result["migrated"][0]
    assert entry["armed"] is False
    assert "platform away" in entry["arm_error"]
    async with async_session_maker() as db:
        a = await db.get(Automation, entry["automation_id"])
        assert a.status == "draft"
        r = await db.get(Routine, rid)
        assert r.enabled is False, "the automation owns the intent now"
        assert r.config_json["migrated_to"] == a.id


@pytest.mark.asyncio
async def test_migration_report_lists_every_briefing():
    uid = await _mk_user()
    rid = await _mk_routine(uid)
    async with async_session_maker() as db:
        await mig.migrate_email_briefings(db, user_id=uid)
        report = await mig.migration_report(db, user_id=uid)
    [row] = report["routines"]
    assert row["routine_id"] == rid
    assert row["name"] == "Morning briefing"
    assert row["migrated_to"]
    assert row["enabled"] is False


def test_promised_time_cron_renders_the_stated_time():
    # §4.11b: "8:00" said in chat must arm as 8:00 — the cron is the
    # promise rendered, not an offset from when the spec was compiled.
    assert mig.promised_time_cron("8:00") == "0 8 * * *"
    assert mig.promised_time_cron("08:00") == "0 8 * * *"
    assert mig.promised_time_cron("22:52") == "52 22 * * *"
    with pytest.raises(ValueError):
        mig.promised_time_cron("25:00")
