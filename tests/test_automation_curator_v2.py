"""Curator v2 write path — gate in front, one store behind (R30 §5.6).

`normalize_candidate` is the writer-side ban (ND-2/ND-3/D-20) plus the
category/scope normalization; `file_facts` targets A's
`memory_v2_service.add_fact` seam and falls back to the R29 seam with
the legacy mapping until the integration merge. Both paths tested with
stubs; no DB — platform sweep.
"""

from __future__ import annotations

import asyncio
import sys
import types

from app.agent.automations import curator_v2


def test_normalize_accepts_v2_and_legacy_categories():
    v2 = curator_v2.normalize_candidate(
        {"text": "Marcus Webb gets same-day answers", "category": "people",
         "scope": "automation", "subject": "Marcus Webb",
         "why": "You replied within the hour four times running."})
    assert v2 and v2["category"] == "people" and v2["scope"] == "automation"
    legacy = curator_v2.normalize_candidate(
        {"text": "Keep replies under three sentences",
         "category": "preferences"})
    assert legacy and legacy["category"] == "your_time"
    assert legacy["scope"] == "global"  # unstated scope defaults global


def test_normalize_refuses_the_banned_classes_and_junk():
    assert curator_v2.normalize_candidate(
        {"text": "The Morning work brief is currently paused.",
         "category": "people"}) is None
    assert curator_v2.normalize_candidate(
        {"text": "Has an automation 'X': Every day at 8:00, check mail "
                 "and post to Slack.", "category": "your_time"}) is None
    assert curator_v2.normalize_candidate(
        {"text": "A real fact", "category": "vibes"}) is None
    assert curator_v2.normalize_candidate("not a dict") is None
    assert curator_v2.normalize_candidate({"category": "people"}) is None


def test_file_facts_uses_the_v2_seam_when_present(monkeypatch):
    calls = []

    async def add_fact(db, **kw):
        calls.append(kw)
        return {"saved": True}

    fake = types.ModuleType("app.services.memory_v2_service")
    fake.add_fact = add_fact
    monkeypatch.setitem(sys.modules, "app.services.memory_v2_service", fake)

    facts = [
        {"text": "Marcus Webb gets same-day answers", "category": "people",
         "scope": "global", "subject": "Marcus Webb",
         "why": "You replied within the hour."},
        {"text": "Thursday afternoons stay free", "category": "your_time",
         "scope": "automation", "subject": None, "why": None},
    ]
    saved = asyncio.run(curator_v2.file_facts(
        None, user_id="u1", facts=facts, automation_id="auto-1",
        domain="work", source="reaction", run_id="r1"))
    assert saved == 2
    assert calls[0]["scope"] == "global"
    assert calls[0]["subject_entity"] == {"kind": "person",
                                           "name": "Marcus Webb"}
    assert calls[1]["scope"] == "auto-1"   # automation scope = its id
    assert all(c["source"] == "reaction" and c["run_id"] == "r1"
               for c in calls)


def test_file_facts_counts_neither_suppressed_nor_failed(monkeypatch):
    async def add_fact(db, **kw):
        if "suppressed" in kw["text"]:
            return {"suppressed": True}
        raise RuntimeError("boom")

    fake = types.ModuleType("app.services.memory_v2_service")
    fake.add_fact = add_fact
    monkeypatch.setitem(sys.modules, "app.services.memory_v2_service", fake)

    saved = asyncio.run(curator_v2.file_facts(
        None, user_id="u1",
        facts=[{"text": "a suppressed one", "category": "people",
                "scope": "global", "subject": None, "why": None},
               {"text": "a failing one", "category": "people",
                "scope": "global", "subject": None, "why": None}],
        automation_id="auto-1"))
    assert saved == 0


def test_fallback_files_through_the_legacy_seam(monkeypatch):
    monkeypatch.setitem(sys.modules, "app.services.memory_v2_service", None)
    # sys.modules[name] = None makes `import` raise ImportError.

    from app.agent.automations import facts as facts_seam

    recorded = []

    async def record(db, *, user_id, automation_id, facts, category,
                     source, source_kind, run_id=None):
        recorded.append((category, tuple(facts)))
        return {"saved": len(facts), "ids": []}

    monkeypatch.setattr(facts_seam, "record", record)

    saved = asyncio.run(curator_v2.file_facts(
        None, user_id="u1",
        facts=[
            {"text": "Marcus gets same-day answers", "category": "people",
             "scope": "global", "subject": None, "why": None},
            {"text": "#platform is the channel that matters",
             "category": "team_workspace", "scope": "automation",
             "subject": None, "why": None},
        ],
        automation_id="auto-1"))
    assert saved == 2
    assert dict(recorded) == {
        "people": ("Marcus gets same-day answers",),
        "preferences": ("#platform is the channel that matters",),
    }


def test_fallback_without_an_automation_scope_files_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "app.services.memory_v2_service", None)
    saved = asyncio.run(curator_v2.file_facts(
        None, user_id="u1",
        facts=[{"text": "x", "category": "people", "scope": "global",
                "subject": None, "why": None}],
        automation_id=None))
    assert saved == 0
