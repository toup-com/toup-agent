"""Spec v2 validator (Round 28) — multi-source triggers, steps[],
variables, per-step on_error, and the fast-lane floors, each rejection
proven to actually reject.

Pure-function tests: no DB, no network. The registry snapshot mirrors
`ConnectorRegistry.automation_registry()` including the Round-28
`params_required` event field.
"""

import pytest

from app.agent.automations.spec import SpecError, validate_spec
from app.agent.automations.spec_v2 import ValidatedSpecV2


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
    many = [{"id": f"s{i}", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * *"}} for i in range(5)]
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


def test_no_write_step_is_rejected():
    spec = good_spec()
    spec["steps"] = spec["steps"][:1]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "no_write_step" in codes(ei)


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
