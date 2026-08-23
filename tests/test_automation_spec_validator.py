"""AutomationSpec validator — every rejection the round brief names,
proven to actually reject (a validator with teeth, not a longer prompt).

Pure-function tests: no DB, no network. The registry snapshot mirrors
the real shape `ConnectorRegistry.automation_registry()` emits.
"""

import pytest

from app.agent.automations.spec import (
    SpecError, render_params, resolve_path, validate_spec,
)


REGISTRY = {
    "jira": {
        "connector_id": "jira",
        "push": False, "poll": True, "floor_s": 300,
        "rate_budget": {"per_hour": 12},
        "scopes_read": ["read:jira-work"],
        "scopes_write_by_action": {
            "jira__add_comment": ["write:jira-work"],
            "jira__create_issue": ["write:jira-work"],
        },
        "target_param_by_action": {
            "jira__add_comment": "issue_key",
            "jira__create_issue": "project_key",
        },
        "events": [{
            "key": "issue_created",
            "description": "",
            "source_tool": "jira__search_issues",
            "poll_args": {"jql": "created >= -1d"},
            "items_path": "issues",
            "dedupe_field": "key",
            "fields": {"key": "key", "summary": "summary", "url": "url"},
        }],
    },
    "slack": {
        "connector_id": "slack",
        "push": False, "poll": False, "floor_s": 300,
        "rate_budget": {},
        "scopes_read": ["channels:read"],
        "scopes_write_by_action": {"slack__send_message": ["chat:write"]},
        "target_param_by_action": {"slack__send_message": "channel"},
        "events": [],
    },
    "gmail": {
        "connector_id": "gmail",
        "push": True, "poll": False, "floor_s": 300,
        "rate_budget": {},
        "scopes_read": [],
        "scopes_write_by_action": {"gmail__create_draft": ["compose"]},
        "target_param_by_action": {"gmail__create_draft": "to"},
        "events": [{
            "key": "email_received", "description": "",
            "dedupe_field": "gmail_message_id",
            "fields": {"message_id": "gmail_message_id",
                       "subject": "subject"},
        }],
    },
}


def good_spec(**over):
    spec = {
        "name": "Jira → Slack",
        "trigger": {
            "mode": "poll",
            "connector_id": "jira",
            "event": "issue_created",
            "poll_interval_s": 300,
        },
        "action": {
            "connector_id": "slack",
            "tool": "slack__send_message",
            "params_template": {"channel": "{{grant.target.id}}",
                                "text": "New: {{event.summary}}"},
            "grant_id": "g-1",
        },
        "dedupe_key": "event.key",
        "mode": "auto",
    }
    spec.update(over)
    return spec


def codes(exc_info):
    return {e["code"] for e in exc_info.value.errors}


def test_good_spec_passes_and_canonicalizes():
    v = validate_spec(good_spec(), REGISTRY)
    assert v.trigger_mode == "poll"
    assert v.action_mutates is True
    assert v.grant_id == "g-1"
    assert v.dedupe_key_field == "key"
    assert v.raw["name"] == "Jira → Slack"


def test_write_without_grant_is_rejected():
    spec = good_spec()
    del spec["action"]["grant_id"]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "write_without_grant" in codes(ei)


def test_unknown_tool_is_rejected():
    # A tool belonging to a DIFFERENT connector than action.connector_id
    # must reject — the prefix check is the ownership check.
    spec = good_spec()
    spec["action"]["tool"] = "jira__create_issue"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "unknown_tool" in codes(ei)


def test_missing_dedupe_key_is_rejected_for_poll_and_push():
    spec = good_spec()
    del spec["dedupe_key"]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "missing_dedupe_key" in codes(ei)

    push = good_spec(trigger={
        "mode": "push", "connector_id": "gmail", "event": "email_received",
    })
    del push["dedupe_key"]
    with pytest.raises(SpecError) as ei:
        validate_spec(push, REGISTRY)
    assert "missing_dedupe_key" in codes(ei)


def test_dedupe_key_must_reference_a_declared_field():
    spec = good_spec(dedupe_key="event.nonexistent")
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "bad_dedupe_key" in codes(ei)


def test_interval_below_floor_is_rejected():
    spec = good_spec()
    spec["trigger"]["poll_interval_s"] = 60
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "interval_below_floor" in codes(ei)


def test_connector_floor_overrides_global_when_higher():
    reg = {**REGISTRY, "jira": {**REGISTRY["jira"], "floor_s": 900}}
    spec = good_spec()
    spec["trigger"]["poll_interval_s"] = 600
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, reg)
    assert "interval_below_floor" in codes(ei)


def test_push_on_pushless_connector_is_rejected():
    spec = good_spec(trigger={
        "mode": "push", "connector_id": "jira", "event": "issue_created",
    })
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "push_unavailable" in codes(ei)


def test_unknown_event_and_connector_are_rejected():
    spec = good_spec()
    spec["trigger"]["event"] = "issue_teleported"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "unknown_event" in codes(ei)

    spec = good_spec()
    spec["trigger"]["connector_id"] = "gitlab"
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "unknown_connector" in codes(ei)


def test_schedule_mode_needs_exactly_one_shape():
    spec = good_spec(trigger={"mode": "schedule", "schedule": {}})
    del spec["dedupe_key"]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "missing_schedule" in codes(ei) or "bad_schedule" in codes(ei)

    spec = good_spec(trigger={
        "mode": "schedule",
        "schedule": {"cron_local": "0 9 * * 1-5", "every_s": 600},
    })
    del spec["dedupe_key"]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    assert "bad_schedule" in codes(ei)

    spec = good_spec(trigger={
        "mode": "schedule", "schedule": {"cron_local": "0 9 * * 1-5"},
    })
    del spec["dedupe_key"]
    v = validate_spec(spec, REGISTRY)
    assert v.trigger_mode == "schedule"


def test_every_error_reported_at_once_not_one_per_round_trip():
    spec = good_spec()
    del spec["action"]["grant_id"]
    spec["trigger"]["poll_interval_s"] = 10
    del spec["dedupe_key"]
    with pytest.raises(SpecError) as ei:
        validate_spec(spec, REGISTRY)
    got = codes(ei)
    assert {"write_without_grant", "interval_below_floor",
            "missing_dedupe_key"} <= got


def test_unknown_top_level_field_is_rejected():
    with pytest.raises(SpecError) as ei:
        validate_spec(good_spec(surprise=1), REGISTRY)
    assert "unknown_field" in codes(ei)


def test_render_params_fills_event_and_grant_target():
    out = render_params(
        {"channel": "{{grant.target.id}}",
         "text": "New: {{event.summary}} ({{event.url}})",
         "count": 3},
        event={"summary": "Fix login", "url": "https://x/ENG-1"},
        grant_target={"id": "C123", "label": "#eng"},
    )
    assert out == {"channel": "C123",
                   "text": "New: Fix login (https://x/ENG-1)",
                   "count": 3}


def test_resolve_path_misses_return_none():
    assert resolve_path({"a": {"b": 1}}, "a.b") == 1
    assert resolve_path({"a": {"b": 1}}, "a.c") is None
    assert resolve_path({"a": 1}, "a.b") is None
