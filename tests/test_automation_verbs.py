"""The verb dictionary (Round 29, CONTRACTS-R29.md §1) — totality and
copy pins.

The module's one hard promise: NO input — unknown tool, unknown
connector, garbage cron, hostile step id — may yield a raw tool name,
a `__`-shaped identifier, or a cron string. R29-C builds session copy
on that promise; the fuzz below runs over every REAL registry tool
plus adversarial junk.
"""

import pytest

from app.services import automation_verbs as verbs
from app.services.connector_registry import ConnectorRegistry


@pytest.fixture(scope="module")
def registry():
    r = ConnectorRegistry()
    r.load_all(include_experimental=True)
    return r


@pytest.fixture(scope="module")
def all_real_tools(registry):
    out = []
    for entry in registry.automation_registry():
        cid = entry["connector_id"]
        for tool in (entry.get("scopes_write_by_action") or {}):
            out.append((tool, cid))
        for ev in entry.get("events") or []:
            if ev.get("source_tool"):
                out.append((ev["source_tool"], cid))
    assert out
    return out


# ── Totality ─────────────────────────────────────────────────────────


_GARBAGE = [
    (None, None), ("", ""), ("wat__evil_tool", "wat"),
    ("gmail__send_message", "gmail"),      # rail-forbidden but total
    ("DROP TABLE users;--", "gmail"), ("x" * 300, None),
    ("florb__do_things", None), (None, "florb"),
    ("jira__search_issues", None),          # connector omitted
]


def test_every_real_tool_and_all_garbage_yield_safe_labels(all_real_tools):
    for tool, cid in list(all_real_tools) + _GARBAGE:
        for status in ("pending", "running", "done", "failed"):
            for count in (None, 0, 6):
                v = verbs.step_verb(tool, cid, status=status, count=count)
                label = v["label"]
                assert isinstance(label, str) and label, (tool, status)
                assert "__" not in label, (tool, label)
                assert label != tool, (tool, label)
                assert "cron" not in label.lower()


def test_every_real_tool_has_a_dictionary_entry_not_a_fallback(all_real_tools):
    """A registry tool falling through to "Working with X" means the
    dictionary lagged a manifest — the lint that keeps copy curated."""
    for tool, cid in all_real_tools:
        v = verbs.step_verb(tool, cid)
        assert not v["label"].startswith("Working"), (
            f"{tool} has no _TOOL_VERBS entry"
        )


def test_engine_phases_brand_as_the_orb():
    for phase in ("evaluate", "prepare", "record", "compose", "deliver"):
        v = verbs.step_verb(None, None, phase=phase)
        assert v["brand"] is None
        assert v["label"]
    # Unknown phase: still total, still the orb.
    v = verbs.step_verb(None, None, phase="frobnicate")
    assert v == {"label": "Working", "brand": None}


def test_connector_steps_brand_as_their_connector():
    assert verbs.step_verb("jira__search_issues", "jira")["brand"] == "jira"
    # Brand derivable from the tool prefix when connector_id is absent.
    assert verbs.step_verb("slack__send_message", None)["brand"] == "slack"


def test_count_interpolation_only_where_declared():
    done = verbs.step_verb("jira__search_issues", "jira",
                           status="done", count=6)
    assert done["label"] == "Read 6 Jira issues"
    # No count form on writes — the count is ignored, never rendered raw.
    posted = verbs.step_verb("slack__send_message", "slack",
                             status="done", count=6)
    assert posted["label"] == "Posted to Slack"
    # Doing-form ignores the count too.
    doing = verbs.step_verb("jira__search_issues", "jira", count=6)
    assert doing["label"] == "Checking Jira"


# ── Schedules ────────────────────────────────────────────────────────


@pytest.mark.parametrize("sched,short", [
    ({"cron_local": "0 8 * * 1-5"}, "weekdays 8:00"),
    ({"cron_local": "30 17 * * *"}, "daily 17:30"),
    ({"cron_local": "0 9 * * 1"}, "Mondays 9:00"),
    ({"cron_local": "0 10 * * 0,6"}, "weekends 10:00"),
    ({"cron_local": "*/5 * * * *"}, "on a custom schedule"),
    ({"cron_local": "not a cron at all"}, "on a custom schedule"),
    ({"every_s": 7200}, "every 2 hours"),
    ({"every_s": 3600}, "hourly"),
    ({"every_s": 300}, "every 5 minutes"),
    ({"every_s": 5}, "every 5 seconds"),
    ({"at": "9:30"}, "daily 9:30"),
])
def test_schedule_human_table(sched, short):
    assert verbs.schedule_human(sched) == short
    if "cron_local" in sched:
        assert sched["cron_local"] not in (verbs.schedule_human(sched) or "")


def test_schedule_human_finds_the_schedule_in_spec_shapes():
    v2 = {"version": 2, "trigger": {"sources": [
        {"id": "m", "mode": "poll", "connector_id": "gmail"},
        {"id": "s", "mode": "schedule",
         "schedule": {"cron_local": "0 8 * * 1-5"}},
    ]}}
    assert verbs.schedule_human(v2) == "weekdays 8:00"
    v1 = {"trigger": {"mode": "schedule", "schedule": {"every_s": 3600}}}
    assert verbs.schedule_human(v1) == "hourly"
    assert verbs.schedule_human({"trigger": {"mode": "poll"}}) is None
    assert verbs.schedule_human(None) is None
    assert verbs.schedule_human("garbage") is None


# ── Rule sentences ───────────────────────────────────────────────────


def test_rule_sentence_reads_like_the_canvas():
    spec = {
        "version": 2,
        "trigger": {"sources": [
            {"id": "s", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ]},
        "steps": [
            {"id": "jira", "connector_id": "jira",
             "tool": "jira__search_issues", "params": {}},
            {"id": "gh", "connector_id": "github",
             "tool": "github__list_issues", "params": {}},
            {"id": "post", "connector_id": "slack",
             "tool": "slack__send_message", "params": {},
             "grant_id": "g1"},
        ],
    }
    sent = verbs.rule_sentence(spec)
    assert sent == ("Every weekday at 8:00, check Jira and GitHub "
                    "and post to Slack.")


def test_rule_sentence_v1_and_junk():
    v1 = {
        "trigger": {"mode": "push", "connector_id": "gmail",
                    "event": "email_received"},
        "action": {"connector_id": "gmail", "tool": "gmail__create_draft",
                   "grant_id": "g"},
    }
    sent = verbs.rule_sentence(v1)
    assert "when a new email arrives" in sent.lower()
    assert "draft an email for you" in sent
    assert "__" not in sent
    assert verbs.rule_sentence({}) is None
    assert verbs.rule_sentence(None) is None


# ── Outcomes and chips ───────────────────────────────────────────────


def test_tone_vocabulary():
    assert verbs.tone_for("sent") == "ok"
    for o in ("partial", "undone", "skipped"):
        assert verbs.tone_for(o) == "warn"
    for o in ("write_failed", "step_failed", "run_cap", "lost",
              "forbidden_tool", "junk", None):
        assert verbs.tone_for(o) == "err"


def test_outcome_sentences_carry_counts_and_extra_writes():
    got = verbs.outcome_sentence(
        "sent", write_tool="slack__send_message", connector_id="slack",
        counts={"jira": 4, "github": 2}, wrote_count=1,
    )
    assert got == {"sentence": "Posted to Slack — Jira 4, GitHub 2.",
                   "tone": "ok"}
    multi = verbs.outcome_sentence(
        "sent", write_tool="slack__send_message", connector_id="slack",
        wrote_count=3,
    )
    assert "(+2 more)" in multi["sentence"]
    skipped = verbs.outcome_sentence("skipped")
    assert skipped["tone"] == "warn"
    assert "confirmation expired" in skipped["sentence"]
    junk = verbs.outcome_sentence("wat__wat", write_tool="wat__wat")
    assert junk["tone"] == "err"
    assert "__" not in junk["sentence"]


def test_fix_chip_shape():
    chip = verbs.fix_chip('Morning "brief"', "step_failed", "boom")
    assert chip["label"] == "Fix this"
    assert "__" not in chip["prompt"]
    assert "Morning" in chip["prompt"]
    assert "failed" in chip["prompt"]


def test_event_tags():
    assert verbs.event_tag("issue_opened") == "on new GitHub issues"
    assert verbs.event_tag("mystery_event") == "on new activity"
    assert verbs.event_tag(None) is None
