"""Curator v2 rules — what may become a memory, and where it files (R30 §4.5, §5.6).

Two prod incidents drive the refusal gate (GROUND-TRUTH-R30 ND-2/ND-3,
dispatch D-20): the curated store contained the automation's own
definition ("Has an automation 'Morning work brief': Every day at
22:52, …") and run-status sentences ("The Morning work brief is
currently paused."). Neither is a memory: status lives in the engine,
the definition lives in the spec. The gate refuses both classes at
write time; A's migration drops the existing rows; the canvas's own
fifteen memory facts are the negative controls — a pattern that
refuses one of them is wrong, not strict.

Also here: the five §4.5 category keys, the scope rule, the
normalized dedupe key that forget-suppression and cross-scope dedupe
share, and the classification prompt for `told` facts.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

CATEGORY_KEYS = (
    "people", "team_workspace", "your_time", "work_you_own", "noise_filters",
)

CATEGORY_LABELS = {
    "people": "PEOPLE",
    "team_workspace": "TEAM & WORKSPACE",
    "your_time": "YOUR TIME",
    "work_you_own": "WORK YOU OWN",
    "noise_filters": "NOISE IT FILTERS",
}

#: R29 → v2 category mapping the migration and the extractor share.
LEGACY_CATEGORY_MAP = {
    "people": "people",
    "preferences": "your_time",
    "deadlines": "work_you_own",
}

_DEFINITION_PATTERNS = (
    # "Has an automation 'Morning work brief': Every day at 22:52, …"
    re.compile(r"\bhas an automation\b", re.IGNORECASE),
    # "Every day at 22:52, check Jira, … and post to Slack."
    re.compile(r"\bcheck .{0,120}\band post\b", re.IGNORECASE),
    re.compile(r"\bevery (?:day|morning|evening|weekday)s? at \d{1,2}[:.]\d{2}\b.{0,80}\b(?:check|read|watch|scan)(?:s|es)?\b",
               re.IGNORECASE),
    # A spec restated as a sentence: "it watches X and posts to Y"
    re.compile(r"\bwatch(?:es)? .{0,80}\band posts? to\b", re.IGNORECASE),
)

_STATUS_PATTERNS = (
    # "The Morning work brief is currently paused." (16:19Z today, prod)
    re.compile(r"\b(?:is|are|was|it's)\s+currently\s+(?:paused|active|running|enabled|disabled|armed)\b",
               re.IGNORECASE),
    re.compile(r"\blast run\b.{0,40}\b(?:partial|failed|completed|succeeded|error)\b",
               re.IGNORECASE),
    re.compile(r"\bpaused after\b.{0,40}\bfailures?\b", re.IGNORECASE),
    re.compile(r"\b(?:carried forward|working state)\b", re.IGNORECASE),
    # Raw engine timestamps have no business in a memory either.
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"),
)


# ── The agent's own reach is not a fact about the user (round 33, item 6)
#
# Six rows reached the founder's Memory page from ONE automation run, each
# one the agent's own connector failure re-voiced in the second person:
#
#   "You have access to the Slack channels #all-toup and #social, but you
#    don't have message-reading access there."
#   "You cannot read messages in GitHub because the org blocks Toup's
#    GitHub access and an org owner needs to approve it in GitHub's OAuth
#    app policy."
#   "You cannot read messages in Teams because its connection needs
#    re-authentication."
#   "You cannot read inbox messages through the currently available
#    Outlook connection."
#
# None of them matched `definition` or `run_status`, which are the only two
# classes this gate knew. They are all the same class: what the AGENT can
# or cannot reach. That belongs to the connection, it changes the moment
# someone reconnects, and it is read back into every later run as if the
# user had said it.
_CAPABILITY_PATTERNS = (
    re.compile(r"\b(?:you|it)\s+(?:cannot|can't|can not)\b.{0,80}"
               r"\b(?:read|access|reach|see|view|send|post|write)\b",
               re.IGNORECASE),
    re.compile(r"\b(?:you|it)\s+(?:do(?:n't| not)\s+have|lacks?|is missing)\b"
               r".{0,80}\b(?:access|permission|scope|rights?)\b",
               re.IGNORECASE),
    re.compile(r"\bneeds?\s+re-?authenticat", re.IGNORECASE),
    re.compile(r"\borg(?:anisation|anization)?\s+(?:owner|admin)\b.{0,60}"
               r"\bapprov", re.IGNORECASE),
    re.compile(r"\bOAuth app polic", re.IGNORECASE),
    re.compile(r"\bmessage-reading access\b", re.IGNORECASE),
    re.compile(r"\bblocks?\s+Toup\b", re.IGNORECASE),
    re.compile(r"\bconnection (?:needs|is|has)\b.{0,40}"
               r"\b(?:expired|re-?auth|broken|unavailable)\b", re.IGNORECASE),
)

# ── A ticket's state is the ticket's, and it is stale by tomorrow ────────
# "You have an open Jira item: SCRUM-1: R29-D live loop test — Jira to
# Slack, Medium priority, unassigned, still To Do; last updated Aug 24."
# was stored as a PERMANENT fact about the user. It is a row in someone
# else's system, read once; the automation reads it fresh every morning.
_ITEM_STATUS_PATTERNS = (
    re.compile(r"\b(?:still|currently)\s+"
               r"(?:to ?do|in progress|done|blocked|open|closed)\b",
               re.IGNORECASE),
    re.compile(r"\blast updated\b", re.IGNORECASE),
    re.compile(r"\b(?:unassigned|assigned to)\b.{0,40}"
               r"\b(?:priority|status|ticket|issue)\b", re.IGNORECASE),
    re.compile(r"\b(?:medium|high|low|highest|lowest)\s+priority\b",
               re.IGNORECASE),
    re.compile(r"\b(?:open|closed)\s+"
               r"(?:jira|linear|github|asana|trello)\s+"
               r"(?:item|issue|ticket|pr|pull request)\b", re.IGNORECASE),
    re.compile(r"\b(?:unread|new)\s+(?:messages?|emails?|mentions?)\b"
               r".{0,30}\bin\b", re.IGNORECASE),
)


def refuse_reason(text: str) -> Optional[str]:
    """Why this text may not become a fact — or None when it may.

    `definition` = the automation's own rule restated (D-20/ND-3);
    `run_status` = engine state wearing a sentence (ND-2);
    `agent_capability` = what the AGENT can or cannot reach (round 33);
    `item_status` = the state of a row in a third-party system (round 33).
    """
    candidate = " ".join((text or "").split())
    if not candidate:
        return "empty"
    for pattern in _DEFINITION_PATTERNS:
        if pattern.search(candidate):
            return "definition"
    for pattern in _STATUS_PATTERNS:
        if pattern.search(candidate):
            return "run_status"
    for pattern in _CAPABILITY_PATTERNS:
        if pattern.search(candidate):
            return "agent_capability"
    for pattern in _ITEM_STATUS_PATTERNS:
        if pattern.search(candidate):
            return "item_status"
    return None


def dedupe_key(text: str) -> str:
    """One normalized hash for cross-scope dedupe and the 30-day forget
    suppression (`memory_forgets.text_hash`): case-, spacing- and
    trailing-punctuation-insensitive, so a relearned fact with a
    cosmetic difference still honours the forget."""
    normalized = re.sub(r"[\s]+", " ", (text or "").strip().lower())
    normalized = normalized.rstrip(".!,;: ")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


#: Reaction signals the curator learns from (§5.6) — named here so the
#: prompt and the evals share one list.
REACTION_SIGNALS = (
    "replied within the hour",
    "ignored it repeatedly",
    "deleted the invite but kept the holds",
    "archived it unread",
    "moved meetings out of the block",
    "forwarded it without comment",
)


def classification_prompt(
    *,
    automation_name: Optional[str],
    candidate_facts: list[str],
    existing_by_category: dict[str, list[str]],
) -> str:
    """The v2 classification instruction for extracted or told facts:
    one category from the five keys, one scope, one entity where one
    resolves — and the refusal classes restated so the model drops them
    before the deterministic gate has to."""
    import json as _json

    where = (
        f'inside the automation "{automation_name}"' if automation_name
        else "in the main chat"
    )
    return (
        "You file durable facts about the user into one platform "
        f"memory. The exchange happened {where}.\n"
        f"Candidate facts: {_json.dumps(candidate_facts, ensure_ascii=False)}\n"
        f"Already known: {_json.dumps(existing_by_category, ensure_ascii=False)}\n\n"
        "For each candidate worth keeping, reply with:\n"
        '- "category": one of people (who matters and how), '
        "team_workspace (channels, ownership, team habits), your_time "
        "(blocks, holds, when things reach the user), work_you_own "
        "(surfaces, tickets, priorities), noise_filters (what never "
        "surfaces).\n"
        '- "scope": "automation" when the fact only matters to this '
        'automation\'s work; "global" when it is about the person or '
        "the user themselves.\n"
        '- "subject": the person, channel, ticket, repo or project the '
        "fact is about, or null.\n"
        '- "why": the evidence in one sentence, in the second person '
        '("You replied within the hour four times running.").\n\n'
        "NEVER file: what an automation is or does, its schedule, its "
        "status, run outcomes, engine counters, or anything the store "
        "already says (across BOTH scopes — a global fact is never "
        "duplicated into an automation scope). Dates become absolute. "
        "An empty list is the right answer for small talk.\n\n"
        'Reply as JSON: {"facts": [{"text", "category", "scope", '
        '"subject", "why"}]}'
    )
