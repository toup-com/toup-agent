"""R30-C follow-up train: the §11 seams, the recall tool, told facts,
and the ND-1 flow-order fix — every LLM stubbed, every write captured.

Pure monkeypatch tests, platform sweep. The seams under test:
  recompiler.recompile_steps      (§4.4 Steps-sheet recompile)
  describe_compile.compile_describe (§4.6 Describe your own)
  told_facts.file_told_facts       (§5.6 day-chat told facts)
  AutomationsSkill automations__memory_recall  (§4.5 recall from the main chat)
  AutomationsSkill._create         (ND-1: grants bound at create)
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from app.agent.automations import describe_compile, recompiler, told_facts
from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.automations.skill import AutomationsSkill


class _FakeAutomation:
    def __init__(self, spec: dict, status: str = "armed"):
        self.id = "auto-x"
        self.name = "Morning work brief"
        self.status = status
        self.spec_json = json.dumps(spec)
        self.connector_id = "gmail"


class _FakeDb:
    async def refresh(self, obj):  # noqa: D401 — seam double
        pass


CURRENT_SPEC = {
    "version": 2, "name": "Morning work brief", "mode": "auto",
    "trigger": {"sources": [{"id": "when", "mode": "schedule",
                             "schedule": {"cron_local": "0 8 * * 1-5"}}]},
    "steps": [
        {"id": "read_mail", "connector_id": "gmail",
         "tool": "gmail__list_messages", "collect": {"limit": 10}},
        {"id": "draft", "connector_id": "gmail",
         "tool": "gmail__create_draft", "grant_id": "grant-1",
         "params": {"body": "{{steps.read_mail.text}}"}},
    ],
}

STEPS = [{"text": "Reads what came in overnight", "sub": ""},
         {"text": "Drafts a reply where an answer is owed", "sub": ""}]


# ------------------------------------------------------------- recompiler

def test_recompile_applies_and_confirms(monkeypatch):
    applied = {}

    async def update_automation(db, *, automation_id, user_id, spec):
        applied.update(spec)
        return None, None

    monkeypatch.setattr("app.agent.automations.service.update_automation",
                        update_automation)

    async def complete(prompt):
        assert "CURRENT SPEC" in prompt and "read_mail" in prompt
        return CURRENT_SPEC

    automation = _FakeAutomation(CURRENT_SPEC, status="armed")
    out = asyncio.run(recompiler.recompile_steps(
        _FakeDb(), automation=automation, user_id="u1", steps=STEPS,
        complete=complete))
    assert out["recompiled"] is True
    assert out["sentence"] == ("Changed the plan — it now does what your "
                               "2 steps say.")
    assert applied["version"] == 2


def test_recompile_singularises_and_flags_a_dropped_arm(monkeypatch):
    async def update_automation(db, *, automation_id, user_id, spec):
        return None, None

    monkeypatch.setattr("app.agent.automations.service.update_automation",
                        update_automation)

    async def complete(prompt):
        return CURRENT_SPEC

    automation = _FakeAutomation(CURRENT_SPEC, status="draft")
    out = asyncio.run(recompiler.recompile_steps(
        _FakeDb(), automation=automation, user_id="u1",
        steps=STEPS[:1], complete=complete))
    assert "your 1 step say" in out["sentence"]
    assert "another look before it runs again" in out["sentence"]


def test_recompile_new_write_needs_consent(monkeypatch):
    async def update_automation(db, **kw):  # pragma: no cover
        raise AssertionError("a consent refusal must never apply")

    monkeypatch.setattr("app.agent.automations.service.update_automation",
                        update_automation)

    with_new_write = json.loads(json.dumps(CURRENT_SPEC))
    with_new_write["steps"].append(
        {"id": "post", "connector_id": "slack",
         "tool": "slack__send_message", "grant_id": None})

    async def complete(prompt):
        return with_new_write

    out = asyncio.run(recompiler.recompile_steps(
        _FakeDb(), automation=_FakeAutomation(CURRENT_SPEC),
        user_id="u1", steps=STEPS, complete=complete))
    assert out["recompiled"] is False
    assert out["code"] == "needs_consent"
    assert out["extra"] == {"account_id": "slack"}
    assert "your yes" in out["sentence"]


def test_recompile_retries_then_degrades_honestly(monkeypatch):
    from app.agent.automations.spec import SpecError

    calls = []

    async def update_automation(db, **kw):
        raise SpecError([{"code": "unknown_tool", "field": "steps[0]", "message": "unknown tool"}])

    monkeypatch.setattr("app.agent.automations.service.update_automation",
                        update_automation)

    async def complete(prompt):
        calls.append(prompt)
        return CURRENT_SPEC

    out = asyncio.run(recompiler.recompile_steps(
        _FakeDb(), automation=_FakeAutomation(CURRENT_SPEC),
        user_id="u1", steps=STEPS, complete=complete))
    assert len(calls) == 2
    assert "unknown tool" in calls[1]
    assert out["recompiled"] is False and "code" not in out
    assert "the wording is saved" in out["sentence"]


def test_recompile_survives_a_dead_model():
    async def complete(prompt):
        raise RuntimeError("boom")

    out = asyncio.run(recompiler.recompile_steps(
        _FakeDb(), automation=_FakeAutomation(CURRENT_SPEC),
        user_id="u1", steps=STEPS, complete=complete))
    assert out["recompiled"] is False and "code" not in out


# -------------------------------------------------------- describe_compile

def _describe_doubles(monkeypatch, created_spec):
    turns = []

    async def fetch_registry(user_id):
        return {"gmail": {"connected": True,
                          "events": {"message_received": {}},
                          "writes": {"gmail__create_draft": {}}}}

    async def create_automation(db, *, user_id, spec, template_slug=None,
                                domain=None, template_mode=False):
        assert template_mode is True
        created_spec.update(spec)
        return _FakeAutomation(spec, status="draft"), None

    class _Thread:
        id = "th-x"

    async def ensure_thread(db, *, user_id, automation_id):
        return _Thread()

    async def append_turn(db, *, user_id, thread, run_id, kind, payload):
        turns.append((kind, payload))

    def automation_payload(a):
        return {"id": a.id, "name": a.name, "status": a.status}

    monkeypatch.setattr("app.agent.automations.registry.fetch_registry",
                        fetch_registry)
    monkeypatch.setattr("app.agent.automations.service.create_automation",
                        create_automation)
    monkeypatch.setattr("app.agent.automations.service.automation_payload",
                        automation_payload)
    monkeypatch.setattr("app.agent.automations.ledger.ensure_thread",
                        ensure_thread)
    monkeypatch.setattr("app.agent.automations.ledger.append_turn",
                        append_turn)
    monkeypatch.setattr("app.agent.automations.workflow.mode_of",
                        lambda a, raw: ("drafts_only", "drafts only"))
    monkeypatch.setattr("app.agent.automations.workflow.schedule_block",
                        lambda a, raw: {"sentence": "Every Friday at 17:00"})
    return turns


def test_describe_creates_and_seeds_the_thread(monkeypatch):
    created = {}
    turns = _describe_doubles(monkeypatch, created)

    async def complete(prompt):
        assert "REGISTRY" in prompt and "every Friday, summarise" in prompt
        return {"spec": CURRENT_SPEC, "domain": "work"}

    out = asyncio.run(describe_compile.compile_describe(
        None, user_id="u1", text="every Friday, summarise my week",
        complete=complete))
    # R38 added `build_history` — the automation the app just made,
    # with the record of how it was built, so the sheet can draw it
    # without a second round trip. This stub's fake automation has no
    # spec columns, so the recorder degrades to None rather than
    # failing the create, which is the contract.
    assert out == {"automation": {"id": "auto-x",
                                  "name": "Morning work brief",
                                  "status": "draft"},
                   "thread_id": "th-x",
                   "build_history": None}
    kinds = [k for k, _ in turns]
    assert kinds[:3] == ["note", "user", "agent"]
    assert turns[0][1]["stamp"] == "added"
    assert turns[1][1]["text"] == "every Friday, summarise my week"
    assert "Nothing runs until you say so." in turns[2][1]["text"]
    # The §5.3 script follows: agent, tool, think, agent.
    assert kinds[3:] == ["agent", "tool", "think", "agent"]
    tool_payload = dict(turns[4][1])
    assert tool_payload["action"] == "Checked what I can do"
    # A's positional call semantics land the schedule sentence in the
    # close, lowered to read as one sentence.
    assert "First run is every Friday at 17:00." in turns[6][1]["text"]


def test_describe_retries_spec_errors_then_refuses(monkeypatch):
    from app.agent.automations.spec import SpecError

    created = {}
    _describe_doubles(monkeypatch, created)

    async def create_automation(db, **kw):
        raise SpecError([{"code": "no_source", "field": "trigger", "message": "no source"}])

    monkeypatch.setattr("app.agent.automations.service.create_automation",
                        create_automation)

    async def complete(prompt):
        return {"spec": {"version": 2}}

    with pytest.raises(describe_compile.DescribeError) as err:
        asyncio.run(describe_compile.compile_describe(
            None, user_id="u1", text="do something", complete=complete))
    assert err.value.code == "cannot_compile"
    # R31-22: it offers another try, not a trip to the main chat. The
    # sentence used to end "tell me in chat and I will build it with
    # you" — which is how a setup request became a chat conversation
    # that created no card at all.
    assert "Try saying it another way" in err.value.sentence
    assert "chat" not in err.value.sentence


def test_describe_dead_model_maps_to_compiler_unavailable(monkeypatch):
    _describe_doubles(monkeypatch, {})

    async def complete(prompt):
        raise RuntimeError("boom")

    with pytest.raises(describe_compile.DescribeError) as err:
        asyncio.run(describe_compile.compile_describe(
            None, user_id="u1", text="do something", complete=complete))
    assert err.value.code == "compiler_unavailable"


# ------------------------------------------------------------- told facts

def test_told_facts_files_global_with_source_told(monkeypatch):
    calls = []

    async def add_fact(db, **kw):
        calls.append(kw)
        return {"saved": True}

    fake = types.ModuleType("app.services.memory_v2_service")
    fake.add_fact = add_fact
    monkeypatch.setitem(sys.modules, "app.services.memory_v2_service", fake)
    import app.services as services_pkg
    monkeypatch.setattr(services_pkg, "memory_v2_service", fake,
                        raising=False)

    async def complete(prompt):
        assert "main chat" in prompt
        return {"facts": [
            {"text": "Sarah is the user's boss", "category": "people",
             "subject": "Sarah", "why": "You said Sarah is your boss."},
            {"text": "The Morning work brief is currently paused.",
             "category": "people"},
        ]}

    n = asyncio.run(told_facts.file_told_facts(
        None, user_id="u1", user_text="My boss is Sarah",
        assistant_text="Noted.", complete=complete))
    assert n == 1
    [kw] = calls
    assert kw["scope"] == "global" and kw["source"] == "told"
    assert kw["text"] == "Sarah is the user's boss"


def test_told_facts_never_raises():
    async def complete(prompt):
        raise RuntimeError("boom")

    assert asyncio.run(told_facts.file_told_facts(
        None, user_id="u1", user_text="hello", assistant_text="",
        complete=complete)) == 0


# ------------------------------------------------------------ skill tools

def test_memory_recall_is_the_last_tool():
    # R31-04 appended `automations__run_now` after it — the array only
    # ever grows at the end, so recall is now second from last.
    tools = AutomationsSkill().get_tools()
    # R37 appended set_destination; R38 appended the five edit tools after
    # it. The array only ever GROWS at the end — that is the whole rule this
    # test protects — so the assertion is on the relative order of the block
    # it names, not on a fixed distance from the tail. Pinning `tools[-3]`
    # made every future append a red test with nothing wrong behind it, and
    # this round is the second time it has come due.
    names = [t["name"] for t in tools]
    assert names.index("automations__memory_recall") \
        < names.index("automations__run_now") \
        < names.index("automations__set_destination")
    assert names[-5:] == [
        "automations__edit_schedule", "automations__edit_rules",
        "automations__edit_steps", "automations__edit_permissions",
        "automations__edit_accounts",
    ]
    description = tools[names.index("automations__memory_recall")]["description"].lower()
    # It must say WHEN it is relevant...
    assert "when the user asks" in description
    # ...but never carry a flow posture: "use it before answering
    # anything" in a description every automations turn sees competes
    # with the setup conversation for its early iterations
    # (CONTRACTS-R30 §14 rule 1). This assertion previously required
    # the opposite — it was pinning the defect.
    assert "before answering" not in description


def test_memory_recall_dispatch_reads_the_v2_store(monkeypatch):
    async def recall(db, *, user_id, query=None, entity=None, category=None,
                     scope=None, since=None):
        assert user_id == "u1" and entity == "Marcus"
        return {"facts": [{"text": "Marcus Webb gets same-day answers"}],
                "episodes": []}

    fake = types.ModuleType("app.services.memory_v2_service")
    fake.recall = recall
    monkeypatch.setitem(sys.modules, "app.services.memory_v2_service", fake)
    import app.services as services_pkg
    monkeypatch.setattr(services_pkg, "memory_v2_service", fake,
                        raising=False)

    from app.config import settings
    monkeypatch.setattr(settings, "automations_enabled", True,
                        raising=False)
    skill = AutomationsSkill()
    out = asyncio.run(skill.execute_tool(
        "automations__memory_recall", {"entity": "Marcus"},
        SkillContext(user_id="u1")))
    assert "same-day answers" in out


def test_create_binds_staged_grants(monkeypatch):
    async def create_automation(db, *, user_id, spec, template_slug=None,
                                domain=None):
        return _FakeAutomation(spec, status="draft"), None

    bound = []

    async def bind_grant(user_id, *, grant_id, automation_id):
        bound.append((grant_id, automation_id))
        return {"ok": True}

    monkeypatch.setattr("app.agent.automations.service.create_automation",
                        create_automation)
    monkeypatch.setattr("app.agent.automations.registry.bind_grant",
                        bind_grant)

    from app.config import settings
    monkeypatch.setattr(settings, "automations_enabled", True,
                        raising=False)
    skill = AutomationsSkill()
    out = asyncio.run(skill.execute_tool(
        "automations__create",
        {"spec": {"steps": [{"tool": "x", "grant_id": "g-1"}],
                  "action": {"grant_id": "g-2"}}},
        SkillContext(user_id="u1")))
    assert "Created automation" in out
    assert bound == [("g-1", "auto-x"), ("g-2", "auto-x")]


def test_every_followup_sentence_passes_the_copy_guard(monkeypatch):
    from app.agent.automations import copy_guard

    sentences = [
        "Changed the plan — it now does what your 2 steps say.",
        "Changed the plan — it now does what your 1 step say.",
        " It needs another look before it runs again.",
        ("Rewrote the steps — the wording is saved, but I could not "
         "recompile the plan to match. Tell me in the thread what "
         "should change and I will do it there."),
        ("That plan needs your yes before Slack can make changes — "
         "approve it and I will finish the change."),
        # R31-22: both of these used to send the user to the main chat.
        "I could not set that up just now. Try again in a moment.",
        ("I could not turn that into a plan. Try saying it another "
         "way — name what it should watch, and what it should do."),
        ("Here is the plan I built from your sentence. "
         "Nothing runs until you say so."),
        "Looked in memory · nothing matched",
        "Looked in memory · 1 fact and 2 runs",
    ]
    for sentence in sentences:
        assert copy_guard.clean(sentence), (sentence,
                                            copy_guard.scan(sentence))


def test_the_recall_coaching_line_is_not_the_user_facing_summary():
    """R31-28. `Nothing in memory matches that. Say so plainly rather
    than inventing an answer.` was the memory tool's whole return value,
    so a line written to steer the model was rendered on a founder's job
    sheet as the thing the tool had done (E-15).

    The instruction is not deleted — without it the model fills the
    silence — it is moved into the half only the model reads.
    """
    from app.agent.automations import copy_guard
    from app.agent.skills.builtins.automations import skill as sk

    for result, expected in (
        ({}, "Looked in memory · nothing matched"),
        ({"facts": [1], "episodes": []}, "Looked in memory · 1 fact"),
        ({"facts": [1, 2], "episodes": [3]},
         "Looked in memory · 2 facts and 1 run"),
    ):
        summary = sk._recall_summary(result)
        assert summary == expected, (result, summary)
        assert copy_guard.clean(summary), copy_guard.scan(summary)
