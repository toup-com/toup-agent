"""Scheduling intent + reminder-tool reachability (founder bug 2026-07-16).

"Temind me teo means later" and "Every 11:10 pm send me a joke" both
scored zero in every category, fell to 'question', and the routines__
tools were filtered out of the LLM's tool list — the agent truthfully
replied "the reminder tool isn't available to me in this turn". Two
layers fixed here: a real scheduling intent for clean phrasings, and
reminder tools in the always-included set so typos can't strip them.
"""

from __future__ import annotations

import pytest

from app.agent.query_intent import (
    _ALWAYS_INCLUDED_TOOLS,
    INTENT_SCHEDULING,
    classify_query_intent,
    filter_tools_by_intent,
)


# ── classifier: clean phrasings land on scheduling ─────────────────


@pytest.mark.parametrize("msg", [
    "Remind me two minutes later",
    "remind me at 7 pm to call mom",
    "set a reminder for tomorrow morning",
    "Every 11:10 pm send me a joke",
    "every morning at 7 send me the news",
    "schedule a daily briefing",
    "wake me at 6:30 am every day",
    "don't let me forget the dentist",
    "in 20 minutes remind me to stretch",
])
def test_scheduling_phrasings_classify_scheduling(msg):
    assert classify_query_intent(msg).category == "scheduling", msg


def test_scheduling_beats_web_for_recurring_news():
    """'every morning send me the latest news' has web signals too —
    the routine to CREATE must win over the one-off search."""
    intent = classify_query_intent("every morning send me the latest news")
    assert intent.category == "scheduling"


def test_scheduling_intent_carries_routines_tools_and_clock():
    assert "routines__remind" in INTENT_SCHEDULING.tool_names
    assert "routines__create" in INTENT_SCHEDULING.tool_names
    assert "start_mission" in INTENT_SCHEDULING.tool_names
    # Reminders are meaningless without the current time in context.
    assert INTENT_SCHEDULING.include_environment is True


# ── existing categories must not shift ─────────────────────────────


@pytest.mark.parametrize("msg,cat", [
    ("hi", "greeting"),
    ("search the web for rust tutorials", "web"),
    ("fix the bug in api.ts", "code"),
    ("what did we discuss yesterday", "memory"),
    ("play some jazz music", "media"),
])
def test_existing_categories_unchanged(msg, cat):
    assert classify_query_intent(msg).category == cat, msg


# ── typo safety net: question intent still ships reminder tools ────


def _fake_tools(names):
    return [{"name": n} for n in names]


def test_typoed_reminder_still_reaches_the_tools():
    """The founder's literal message. Zero keyword hits → question is
    acceptable — but the filtered tool list must still contain the
    reminder tools so the model can act."""
    intent = classify_query_intent("Temind me teo means later")
    assert intent.category == "question"

    all_tools = _fake_tools([
        "routines__remind", "routines__create", "routines__list",
        "routines__delete", "web_search", "exec", "navigate_to", "spawn",
    ])
    kept = {t["name"] for t in filter_tools_by_intent(all_tools, intent)}
    assert "routines__remind" in kept
    assert "routines__create" in kept
    # Round 33: a question turn DOES carry the two tools that answer one —
    # it carried none, and the model spent its first iteration on whatever
    # the always-included set happened to hold. What it must not carry is
    # the do-something set, which `test_work_tracking_not_on_greeting_or_
    # question` pins.
    assert "exec" not in kept, "question stays lean otherwise"


def test_always_included_set_pins():
    for name in ("routines__remind", "routines__create", "routines__list",
                 "spawn", "navigate_to"):
        assert name in _ALWAYS_INCLUDED_TOOLS
