"""The curator may not file its own reach as a fact about the user.

Round 33, item 6. Six rows reached the founder's Memory page from ONE
automation run — every one the AGENT's connector failure, re-voiced in
the second person by the prompt's own "why" rule:

    "You have access to the Slack channels #all-toup and #social, but you
     don't have message-reading access there."   why: "You mentioned that…"
    "You cannot read messages in GitHub because the org blocks Toup's
     GitHub access…"                             why: "You stated that…"

The user said none of it. Two layers now stop it: the prompts label the
assistant block CONTEXT ONLY and import the v3 durability rules, and this
deterministic gate refuses the class outright. Prompts are advisory; this
file pins the half that is not.

The positive controls matter as much as the negative ones — a gate that
refuses a real fact is worse than the one it replaces.
"""

import pytest

from app.agent.automations.curation_rules import refuse_reason

#: The six observed rows, verbatim from the founder's Memory page, plus
#: the two classes the gate already knew.
JUNK = [
    ("You have access to the Slack channels #all-toup and #social, but you "
     "don't have message-reading access there.", "agent_capability"),
    ("You cannot read messages in GitHub because the org blocks Toup's GitHub "
     "access and an org owner needs to approve it in GitHub's OAuth app "
     "policy.", "agent_capability"),
    ("You cannot read messages in Teams because its connection needs "
     "re-authentication.", "agent_capability"),
    ("You cannot read inbox messages through the currently available Outlook "
     "connection.", "agent_capability"),
    ("You have an open Jira item: SCRUM-1: R29-D live loop test — Jira to "
     "Slack, Medium priority, unassigned, still To Do; last updated Aug 24.",
     "item_status"),
    ("The Morning work brief is currently paused.", "run_status"),
    ("Has an automation 'Morning work brief': Every day at 22:52, check Jira, "
     "read GitHub and post to Slack.", "definition"),
]

#: Real facts about a person. Every one MUST survive — this is the half a
#: stricter gate breaks, and the reason the round did not simply ban the
#: word "cannot".
REAL = [
    "Sarah Chen is your manager.",
    "You are allergic to shellfish.",
    "You block 9–11am for deep work every weekday.",
    "You own the billing surface at Toup.",
    "You never want marketing email surfaced.",
    "You reply to Jira mentions within the hour.",
    "Your father is Nariman Hosseini.",
    "You prefer Persian pop, especially Googoosh.",
    "You are learning to sail and take lessons on Saturdays.",
    "You moved to Toronto in March 2026.",
]


@pytest.mark.parametrize("text,expected", JUNK)
def test_the_agents_own_report_is_refused(text, expected):
    assert refuse_reason(text) == expected, text


@pytest.mark.parametrize("text", REAL)
def test_a_real_fact_about_the_user_is_kept(text):
    assert refuse_reason(text) is None, text


def test_an_empty_candidate_is_refused():
    assert refuse_reason("") == "empty"
    assert refuse_reason("   ") == "empty"
