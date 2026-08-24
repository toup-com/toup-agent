"""Capability metadata honesty (Round 28) — the REAL manifest-built
registry, pinned.

These tests load the actual connector manifests (experimental included,
for the stub) and assert the §3 table of CONTRACTS-R28.md: all ten
product connectors are automatable, `push` is true ONLY where a real
webhook path exists (Gmail Pub/Sub — still the only one), floors are at
or above the 5-minute rail, every declared write action is a mutating
tool whose pinned-target parameter exists in its own input schema, and
every poll event's source_tool resolves. A capability block that
flatters a connector cannot merge.
"""

import pytest

from app.services.connector_registry import ConnectorRegistry

ALL_TEN = {
    "jira", "slack", "github", "calendar", "gmail",
    "outlook", "teams", "notion", "drive", "docs",
}


@pytest.fixture(scope="module")
def registry():
    r = ConnectorRegistry()
    r.load_all(include_experimental=True)
    assert not r.alarms(), [a.reason for a in r.alarms()]
    return r


@pytest.fixture(scope="module")
def entries(registry):
    return {e["connector_id"]: e for e in registry.automation_registry()}


def test_all_ten_connectors_are_automatable(entries):
    assert ALL_TEN <= set(entries), sorted(ALL_TEN - set(entries))


def test_push_is_true_only_for_gmail(entries):
    """Honesty pin: no other connector has an inbound webhook path in
    this codebase. A new push:true entry must arrive WITH the webhook,
    not before it."""
    push = {cid for cid, e in entries.items() if e["push"]}
    assert push == {"gmail"}


def test_every_floor_is_at_or_above_the_rail(entries):
    for cid, e in entries.items():
        assert e["floor_s"] >= 300, f"{cid} floor {e['floor_s']}"


def test_write_actions_are_mutating_tools_with_real_target_params(
    registry, entries,
):
    for cid, e in entries.items():
        for tool_name in e["scopes_write_by_action"]:
            spec = registry.get_tool_spec(tool_name)
            assert spec is not None, f"{cid}: {tool_name} not a tool"
            assert spec.mutates, f"{cid}: {tool_name} is not mutating"
            target = e["target_param_by_action"].get(tool_name)
            assert target, f"{cid}: {tool_name} has no pinned target param"
            props = (spec.input_schema or {}).get("properties") or {}
            assert target in props, (
                f"{cid}: {tool_name} target {target!r} not in its schema"
            )


def test_poll_events_resolve_and_declare_their_requirements(
    registry, entries,
):
    for cid, e in entries.items():
        for ev in e["events"]:
            if not e["poll"]:
                continue
            tool = ev.get("source_tool")
            assert tool, f"{cid}: poll event {ev['key']} has no source_tool"
            spec = registry.get_tool_spec(tool)
            assert spec is not None and not spec.mutates, (
                f"{cid}: {ev['key']} source {tool} missing or mutating"
            )
            props = (spec.input_schema or {}).get("properties") or {}
            required = set((spec.input_schema or {}).get("required") or [])
            supplied = set(ev.get("poll_args") or {}) | set(
                ev.get("params_required") or []
            )
            missing = required - supplied
            assert not missing, (
                f"{cid}: {ev['key']} would poll {tool} without its "
                f"required params {sorted(missing)} — declare them in "
                f"poll_args or params_required"
            )
            for p in ev.get("params_required") or []:
                assert p in props, (
                    f"{cid}: {ev['key']} params_required {p!r} not in "
                    f"{tool}'s schema"
                )


def test_specific_round28_shape_pins(entries):
    """The §3 table, literally — so a drive-by edit shows up in review
    as a contract change, not a silent drift."""
    # Outlook: poll source only, NO writes (send is rail-forbidden and
    # there is no draft tool).
    assert entries["outlook"]["poll"] is True
    assert entries["outlook"]["scopes_write_by_action"] == {}
    assert [e["key"] for e in entries["outlook"]["events"]] == [
        "email_received"]
    # Teams: chat polling requires the chat id.
    teams_ev = entries["teams"]["events"][0]
    assert teams_ev["params_required"] == ["chat_id"]
    assert entries["teams"]["target_param_by_action"][
        "teams__send_chat_message"] == "chat_id"
    # Notion: create_page pinned to a parent page.
    assert entries["notion"]["target_param_by_action"][
        "notion__create_page"] == "parent_page_id"
    # Drive: source-only (create_doc has no pinnable stable target).
    assert entries["drive"]["scopes_write_by_action"] == {}
    assert [e["key"] for e in entries["drive"]["events"]] == ["file_added"]
    # Docs: action-only (nothing to poll), append pinned to a document.
    assert entries["docs"]["events"] == []
    assert entries["docs"]["poll"] is False
    assert entries["docs"]["target_param_by_action"][
        "docs__append_text"] == "document_id"
    # GitHub honesty retrofit: the list tool needs a repo.
    gh_ev = entries["github"]["events"][0]
    assert gh_ev["params_required"] == ["owner", "repo"]


def test_params_required_survives_the_wire_serialization(entries):
    """R26's e2e caught the registry wire payload silently dropping
    event fields — pin the new one on every event."""
    for e in entries.values():
        for ev in e["events"]:
            assert "params_required" in ev
