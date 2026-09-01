"""Spec v2 validator (Round 28) — multi-source triggers, steps[],
variables, per-step on_error, and the fast-lane floors, each rejection
proven to actually reject.

Pure-function tests: no DB, no network. The registry snapshot mirrors
`ConnectorRegistry.automation_registry()` including the Round-28
`params_required` event field.
"""

import pytest

from app.agent.automations.spec import (
    SpecError, validate_spec,
    unanswered_variables as spec_unanswered,
)
from app.agent.automations.spec_v2 import (
    MAX_SOURCES, ValidatedSpecV2, unanswered_variables,
)


REGISTRY = {
    "jira": {
        "connector_id": "jira",
        "push": False, "poll": True, "floor_s": 300,
        "scopes_write_by_action": {
            "jira__create_issue": ["write:jira-work"],
        },
        "target_param_by_action": {"jira__create_issue": "project_key"},
        "events": [{
            "key": "issue_created", "description": "",
            "source_tool": "jira__search_issues",
            "poll_args": {"jql": "created >= -1d"},
            "params_required": [],
            "items_path": "issues",
            "dedupe_field": "key",
            "fields": {"key": "key", "summary": "summary"},
        }],
    },
    "teams": {
        "connector_id": "teams",
        "push": False, "poll": True, "floor_s": 300,
        "scopes_write_by_action": {"teams__send_chat_message": ["send"]},
        "target_param_by_action": {"teams__send_chat_message": "chat_id"},
        "events": [{
            "key": "chat_message_received", "description": "",
            "source_tool": "teams__read_chat_messages",
            "poll_args": {"max_results": 25},
            "params_required": ["chat_id"],
            "items_path": "messages",
            "dedupe_field": "id",
            "fields": {"id": "id", "body": "body"},
        }],
    },
    "slack": {
        "connector_id": "slack",
        "push": False, "poll": False, "floor_s": 300,
        "scopes_write_by_action": {"slack__send_message": ["chat:write"]},
        "target_param_by_action": {"slack__send_message": "channel"},
        "events": [],
    },
    "gmail": {
        "connector_id": "gmail",
        "push": True, "poll": False, "floor_s": 300,
        "scopes_write_by_action": {"gmail__create_draft": ["compose"]},
        "target_param_by_action": {"gmail__create_draft": "to"},
        "events": [{
            "key": "email_received", "description": "",
            "params_required": [],
            "dedupe_field": "gmail_message_id",
            "fields": {"message_id": "gmail_message_id",
                       "subject": "subject"},
        }],
    },
}


def good_spec(**over):
    spec = {
        "version": 2,
        "name": "Brief",
        "mode": "auto",
        "variables": {"jql": "assignee = currentUser()"},
        "trigger": {
            "sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "0 8 * * 1-5"}},
            ],
        },
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues",
             "params": {"jql": "{{var.jql}}"},
             "collect": {"items_path": "issues",
                         "fields": {"key": "key"},
                         "format": "• {{item.key}}",
                         "empty_text": "none"},
             "on_error": "skip"},
            {"id": "post", "connector_id": "slack",
             "tool": "slack__send_message",
             "params": {"channel": "{{grant.target.id}}",
                        "text": "{{steps.issues.text}}"},
             "grant_id": "g-1"},
        ],
    }
    spec.update(over)
    return spec


def codes(exc_info):
    return {e["code"] for e in exc_info.value.errors}


# ── dispatch ─────────────────────────────────────────────────────────


def test_version_2_returns_v2_and_v1_path_is_untouched():
    v = validate_spec(good_spec(), REGISTRY)
    assert isinstance(v, ValidatedSpecV2)
    assert v.raw["version"] == 2
    # v1 spec still takes the v1 path and its canonical dict carries
    # no version key — byte-identical persistence for existing rows.
    v1 = validate_spec({
        "name": "Old", "mode": "auto",
        "trigger": {"mode": "schedule", "schedule": {"cron_local": "0 8 * * *"}},
        "action": {"connector_id": "slack", "tool": "slack__send_message",
                   "params_template": {"channel": "{{grant.target.id}}",
                                       "text": "hi"},
                   "grant_id": "g-1"},
    }, REGISTRY)
    assert not isinstance(v1, ValidatedSpecV2)
    assert "version" not in v1.raw


def test_bad_version_is_rejected():
    with pytest.raises(SpecError) as ei:
        validate_spec(good_spec(version=3), REGISTRY)
    assert "bad_version" in codes(ei)


# ── sources ──────────────────────────────────────────────────────────


def test_multi_source_validates_and_each_lane_keeps_its_dedupe():
    spec = good_spec(trigger={"sources": [
        {"id": "sched", "mode": "schedule",
         "schedule": {"cron_local": "0 8 * * 1-5"}},
        {"id": "tickets", "mode": "poll", "connector_id": "jira",
         "event": "issue_created", "poll_interval_s": 300,
         "dedupe_key": "event.key"},
        {"id": "mail", "mode": "push", "connector_id": "gmail",
         "event": "email_received", "dedupe_key": "event.message_id"},
    ]})
    v = validate_spec(spec, REGISTRY)
    assert [s.id for s in v.sources] == ["sched", "tickets", "mail"]
    assert v.source_by_id("tickets").dedupe_key_field == "key"
    assert v.source_by_id("mail").dedupe_key_field == "message_id"
    assert v.trigger_mode == "multi"


def test_source_limits_and_duplicates():
    # R43 raised MAX_SOURCES 4 -> 12 (one schedule plus up to eleven instant
    # lanes), so the count has to be read from the constant or the test stops
    # testing the cap the moment it moves again.
    many = [{"id": f"s{i}", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * *"}}
            for i in range(MAX_SOURCES + 1)]
    with pytest.raises(SpecError) as ei:
        validate_spec(good_spec(trigger={"sources": many}), REGISTRY)
    assert {"too_many_sources", "duplicate_schedule_source"} <= codes(ei)

    dup = [{"id": "x", "mode": "schedule",
            "schedule": {"cron_local": "0 8 * * *"}},
           {"id": "x", "mode": "poll", "connector_id": "jira",
            "event": "issue_created", "dedupe_key": "event.key"}]
    with pytest.raises(SpecError) as ei:
        validate_spec(good_spec(trigger={"sources": dup}), REGISTRY)
    assert "duplicate_source_id" in codes(ei)


def test_poll_source_without_dedupe_key_is_rejected():
    spec = good_spec(trigger={"sources": [
        {"id": "t", "mode": "poll", "connector_id": "jira",
         "event": "issue_created"},
    ]})
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "missing_dedupe_key" in codes(ei)


def test_missing_event_param_is_rejected():
    spec = good_spec(trigger={"sources": [
        {"id": "chat", "mode": "poll", "connector_id": "teams",
         "event": "chat_message_received", "dedupe_key": "event.id"},
    ]})
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "missing_event_param" in codes(ei)
    # Supplying the required param (even as a variable ref) passes.
    spec = good_spec(
        variables={"jql": "x", "chat": "19:abc"},
        trigger={"sources": [
            {"id": "chat", "mode": "poll", "connector_id": "teams",
             "event": "chat_message_received",
             "params": {"chat_id": "{{var.chat}}"},
             "dedupe_key": "event.id"},
        ]})
    validate_spec(spec, REGISTRY)


def test_push_only_where_the_connector_declares_it():
    spec = good_spec(trigger={"sources": [
        {"id": "t", "mode": "push", "connector_id": "jira",
         "event": "issue_created", "dedupe_key": "event.key"},
    ]})
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "push_unavailable" in codes(ei)


# ── steps ────────────────────────────────────────────────────────────


def test_write_without_grant_per_step():
    spec = good_spec()
    del spec["steps"][1]["grant_id"]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "write_without_grant" in codes(ei)


def test_reads_only_spec_is_accepted():
    # Round 30 (§4.11a): a spec with read steps only is legal — migrated
    # email briefings deliver via the notification pipeline, not a write
    # step, and §4.1 derives mode "reads_only" from this shape. (Until
    # R30 this was the `no_write_step` rejection.)
    spec = good_spec()
    spec["steps"] = spec["steps"][:1]
    v = validate_spec(spec, REGISTRY)
    assert isinstance(v, ValidatedSpecV2)
    assert v.write_steps == ()


def test_read_after_write_is_rejected():
    spec = good_spec()
    spec["steps"] = [spec["steps"][1], spec["steps"][0]]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "write_before_read" in codes(ei)


def test_too_many_writes_is_rejected():
    spec = good_spec()
    spec["steps"] = [spec["steps"][0]] + [
        {"id": f"w{i}", "connector_id": "slack",
         "tool": "slack__send_message",
         "params": {"channel": "{{grant.target.id}}", "text": "x"},
         "grant_id": f"g-{i}"}
        for i in range(4)
    ]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "too_many_writes" in codes(ei)


def test_reserved_and_duplicate_step_ids():
    spec = good_spec()
    spec["steps"][0]["id"] = "event"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "reserved_step_id" in codes(ei)

    spec = good_spec()
    spec["steps"][0]["id"] = "post"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "duplicate_step_id" in codes(ei)


def test_collect_on_write_step_is_rejected():
    spec = good_spec()
    spec["steps"][1]["collect"] = {"items_path": "x"}
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "bad_collect" in codes(ei)


def test_bad_on_error_is_rejected():
    spec = good_spec()
    spec["steps"][0]["on_error"] = "retry"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "bad_on_error" in codes(ei)


# ── variables ────────────────────────────────────────────────────────


def test_undeclared_variable_reference_is_rejected():
    spec = good_spec(variables={})
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "unknown_variable" in codes(ei)


def test_template_mode_treats_template_vars_as_declared_and_waives_grants():
    spec = good_spec(variables={})
    del spec["steps"][1]["grant_id"]
    v = validate_spec(spec, REGISTRY, template_mode=True,
                      template_vars={"jql"})
    assert isinstance(v, ValidatedSpecV2)
    # The create/run path (template_mode off) still rejects both.
    with pytest.raises(SpecError):
        validate_spec(spec, REGISTRY)


# ── fast-lane floors ─────────────────────────────────────────────────


def _fast_spec(interval):
    return good_spec(trigger={"sources": [
        {"id": "t", "mode": "poll", "connector_id": "jira",
         "event": "issue_created", "poll_interval_s": interval,
         "dedupe_key": "event.key"},
    ]})


def test_poll_floor_holds_without_the_fast_lane(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "automations_dev_fast_lane", False,
                        raising=False)
    with pytest.raises(SpecError) as ei:
        validate_spec(_fast_spec(30), REGISTRY)
    assert "interval_below_floor" in codes(ei)


def test_fast_lane_lowers_the_floor_off_production(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "automations_dev_fast_lane", True,
                        raising=False)
    monkeypatch.setattr(settings, "environment", "development")
    v = validate_spec(_fast_spec(5), REGISTRY)
    assert v.sources[0].poll_interval_s == 5
    # v1 gets the same relief through the shared helper.
    v1 = validate_spec({
        "name": "Old",
        "trigger": {"mode": "poll", "connector_id": "jira",
                    "event": "issue_created", "poll_interval_s": 5},
        "action": {"connector_id": "slack", "tool": "slack__send_message",
                   "params_template": {"channel": "c", "text": "x"},
                   "grant_id": "g"},
        "dedupe_key": "event.key",
    }, REGISTRY)
    assert v1.poll_interval_s == 5


def test_fast_lane_is_refused_in_production(monkeypatch):
    """The two-sided refusal: the env flag alone must change nothing on
    a production tenant."""
    from app.config import settings
    monkeypatch.setattr(settings, "automations_dev_fast_lane", True,
                        raising=False)
    monkeypatch.setattr(settings, "environment", "production")
    with pytest.raises(SpecError) as ei:
        validate_spec(_fast_spec(5), REGISTRY)
    assert "interval_below_floor" in codes(ei)


# ── canonical shape ──────────────────────────────────────────────────


def test_canonical_round_trips_through_the_validator():
    v = validate_spec(good_spec(), REGISTRY)
    again = validate_spec(v.raw, REGISTRY)
    assert again.raw == v.raw


def test_grant_target_is_accepted_on_revalidation_only_as_object():
    spec = good_spec()
    spec["steps"][1]["grant_target"] = {"kind": "channel", "id": "C1"}
    v = validate_spec(spec, REGISTRY)
    assert v.steps[1].grant_target == {"kind": "channel", "id": "C1"}
    spec["steps"][1]["grant_target"] = "C1"
    with pytest.raises(SpecError):
        validate_spec(spec, REGISTRY)


def test_every_error_reported_at_once():
    spec = good_spec(variables={})
    spec["trigger"]["sources"].append(
        {"id": "t", "mode": "poll", "connector_id": "nope",
         "event": "x"})
    del spec["steps"][1]["grant_id"]
    spec["steps"][0]["on_error"] = "explode"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert len(codes(ei)) >= 4


# ── R38: the agent step ──────────────────────────────────────────────
#
# A step whose work is a model call. It has no connector, carries a
# prompt and an `output_var`, and binds what it worked out to
# `{{var.<output_var>}}` for later steps and the narration.


def agent_spec(**over):
    """`good_spec` with a thinking step between the read and the write,
    and the write reading what it produced."""
    spec = good_spec()
    spec["steps"] = [
        spec["steps"][0],
        {"id": "rank", "kind": "agent",
         "prompt": "Rank {{steps.issues.text}} by what blocks someone.",
         "output_var": "ranked"},
        {**spec["steps"][1],
         "params": {"channel": "{{grant.target.id}}",
                    "text": "{{var.ranked}}"}},
    ]
    spec.update(over)
    return spec


def test_an_agent_step_validates_and_declares_the_name_it_writes():
    v = validate_spec(agent_spec(), REGISTRY)
    assert [s.kind for s in v.steps] == ["tool", "agent", "tool"]
    step = v.steps[1]
    assert step.output_var == "ranked"
    assert step.connector_id == "" and step.tool == ""
    assert step.mutates is False
    assert [s.id for s in v.agent_steps] == ["rank"]
    # `{{var.ranked}}` in the write is declared BY the agent step — no
    # `variables` entry is needed and none is invented.
    assert "ranked" not in v.raw.get("variables", {})


def test_an_agent_step_defaults_to_fail_not_continue():
    """Its answer is interpolated into a later template, and a missing
    value renders as an empty string — a swallowed failure posts a hole
    rather than omitting a section."""
    v = validate_spec(agent_spec(), REGISTRY)
    assert v.steps[1].on_error == "fail"
    # …and the canonical form omits the default, exactly like a step
    # that declares nothing anywhere else.
    assert "on_error" not in v.raw["steps"][1]
    explicit = agent_spec()
    explicit["steps"][1]["on_error"] = "skip"
    assert validate_spec(explicit, REGISTRY).raw["steps"][1]["on_error"] \
        == "skip"


def test_a_spec_with_no_agent_step_canonicalizes_exactly_as_before():
    """Backwards compatibility, at the byte level: `kind` appears on an
    agent step and NOWHERE else, so every persisted v2 spec re-validates
    to the identical dict it did before this round."""
    v = validate_spec(good_spec(), REGISTRY)
    assert all("kind" not in s for s in v.raw["steps"])
    assert v.raw["steps"] == [
        {"id": "issues", "connector_id": "jira",
         "tool": "jira__search_issues", "params": {"jql": "{{var.jql}}"},
         "collect": {"items_path": "issues", "fields": {"key": "key"},
                     "format": "• {{item.key}}", "limit": 10,
                     "empty_text": "none"},
         "on_error": "skip"},
        {"id": "post", "connector_id": "slack",
         "tool": "slack__send_message",
         "params": {"channel": "{{grant.target.id}}",
                    "text": "{{steps.issues.text}}"},
         "grant_id": "g-1"},
    ]
    assert validate_spec(v.raw, REGISTRY).raw == v.raw


def test_agent_step_round_trips():
    v = validate_spec(agent_spec(), REGISTRY)
    assert validate_spec(v.raw, REGISTRY).raw == v.raw


def test_an_agent_step_needs_a_prompt_and_a_readable_name():
    spec = agent_spec()
    spec["steps"][1] = {"id": "rank", "kind": "agent"}
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert {"missing_agent_prompt", "bad_output_var"} <= codes(ei)
    # Every message is a sentence a person can act on, not a code.
    for e in ei.value.errors:
        assert len(e["message"]) > 25 and " " in e["message"]


def test_an_agent_step_prompt_is_bounded():
    from app.agent.automations.spec_v2 import AGENT_PROMPT_MAX_CHARS
    spec = agent_spec()
    spec["steps"][1]["prompt"] = "x" * (AGENT_PROMPT_MAX_CHARS + 1)
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "agent_prompt_too_long" in codes(ei)


@pytest.mark.parametrize("name", ["Ranked", "1st", "steps", "var", ""])
def test_output_var_must_be_a_usable_template_name(name):
    spec = agent_spec()
    spec["steps"][1]["output_var"] = name
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "bad_output_var" in codes(ei)


@pytest.mark.parametrize("field,value", [
    ("connector_id", "jira"),
    ("tool", "jira__search_issues"),
    ("params", {"jql": "x"}),
    ("collect", {"items_path": "issues", "fields": {"key": "key"}}),
    ("grant_id", "g-2"),
    ("grant_target", {"id": "C1"}),
])
def test_an_agent_step_calls_nothing(field, value):
    spec = agent_spec()
    spec["steps"][1][field] = value
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "agent_step_calls_nothing" in codes(ei)


def test_a_tool_step_may_not_carry_agent_fields():
    spec = good_spec()
    spec["steps"][0]["prompt"] = "think about it"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "tool_step_is_not_an_agent_step" in codes(ei)


def test_an_unknown_step_kind_is_refused():
    spec = agent_spec()
    spec["steps"][1]["kind"] = "oracle"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "bad_step_kind" in codes(ei)


def test_two_agent_steps_may_not_write_the_same_name():
    spec = agent_spec()
    spec["steps"].insert(2, {"id": "rank2", "kind": "agent",
                             "prompt": "again", "output_var": "ranked"})
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "duplicate_output_var" in codes(ei)


def test_output_var_may_not_shadow_a_declared_variable():
    spec = agent_spec(variables={"jql": "x", "ranked": "seed"})
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "output_var_shadows_variable" in codes(ei)


def test_agent_steps_are_capped():
    from app.agent.automations.spec_v2 import MAX_AGENT_STEPS
    spec = agent_spec()
    for i in range(MAX_AGENT_STEPS):
        spec["steps"].insert(2, {"id": f"extra{i}", "kind": "agent",
                                 "prompt": "x", "output_var": f"v{i}"})
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "too_many_agent_steps" in codes(ei)


def test_an_agent_step_may_not_follow_a_write():
    """Same rule and same reason as a read: a write is staged
    asynchronously, so nothing after it could see it happen."""
    spec = agent_spec()
    spec["steps"] = [spec["steps"][0], spec["steps"][2],
                     {"id": "late", "kind": "agent", "prompt": "x",
                      "output_var": "v"}]
    spec["steps"][1] = {**spec["steps"][1],
                        "params": {"channel": "{{grant.target.id}}",
                                   "text": "hi"}}
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "write_before_read" in codes(ei)


def test_an_undeclared_variable_in_a_prompt_is_still_undeclared():
    """The prompt is a template like any params value, so a reference
    that would silently render as an empty string mid-sentence fails
    the same way it does everywhere else."""
    spec = agent_spec()
    spec["steps"][1]["prompt"] = "Rank it for {{var.nobody}}"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "unknown_variable" in codes(ei)


def test_an_agent_step_is_never_the_automations_acting_connector():
    """`action_connector_id` answers "which account does this act
    through" — an agent step has none, and must not answer ''."""
    spec = agent_spec()
    spec["steps"] = [spec["steps"][1], spec["steps"][0]]   # agent first
    spec["steps"][1] = {**spec["steps"][1], "on_error": "skip"}
    spec["steps"][0] = {**spec["steps"][0], "prompt": "Rank the day."}
    v = validate_spec(spec, REGISTRY)
    assert v.steps[0].kind == "agent"
    assert v.action_connector_id == "jira"


# ── unanswered settings (R42) ────────────────────────────────────────
#
# `validate_spec(template_mode=True, template_vars=None)` deliberately
# waives the undeclared-variable rule so a mid-setup draft can still be
# read and edited. `unanswered_variables` is what the arm gate and the
# fire path ask instead, and it must agree with what `render_value`
# would actually produce.

def test_a_referenced_variable_with_no_value_is_unanswered():
    """The founder's chain, from the top: `{{var.github_owner}}` with
    nothing behind it renders as "" and reaches GitHub as an empty
    owner. It is named here so nothing downstream has to guess."""
    spec = good_spec(variables={})
    spec["trigger"]["sources"] = [{
        "id": "gh", "mode": "poll", "connector_id": "github",
        "event": "issue_opened",
        "params": {"owner": "{{var.github_owner}}", "repo": "x"},
        "poll_interval_s": 600, "dedupe_key": "event.number",
    }]
    assert unanswered_variables(spec) == ["github_owner", "jql"]
    # Reference order, deduped, and an EMPTY string is not an answer.
    spec["variables"] = {"jql": "  ", "github_owner": "toup-com"}
    assert unanswered_variables(spec) == ["jql"]


def test_an_agent_steps_own_output_is_never_unanswered():
    """`{{var.ranked}}` is written DURING the run by the step that
    declares it, so a spec whose only reference is one is complete —
    this is the shipped Morning work brief's exact shape."""
    assert unanswered_variables(agent_spec()) == []


def test_a_v1_spec_has_no_settings_to_leave_unanswered():
    """Variables are a v2 grammar; the dispatch must not read a v1
    spec's `action` as if it had them."""
    assert spec_unanswered({"trigger": {}, "action": {}}) == []
