"""Verb dictionary v2 entries — total, clean, and never a false done-form.

Three pins:
  1. totality — every tool the v1 dictionary knows has a v2 entry, so
     the overlay can never REDUCE coverage; every connector table
     carries a `*` read fallback;
  2. the copy contract — every phrase the tables can serve passes the
     guard (D-01/D-08: this is the vocabulary that replaces raw tool
     ids);
  3. ND-4 — a failure phrasing never collides with a write's done form
     and never opens with a past-tense success verb: a refused write
     must not read "Posted to Slack".

`app/services` module, no DB — platform sweep (the platform image ships
this file: templates read the dictionary platform-side).
"""

from __future__ import annotations

import pytest

from app.agent.automations import copy_guard
from app.services import automation_verb_entries as entries
from app.services.automation_verbs import _TOOL_VERBS

_DONE_OPENERS = (
    "Posted", "Sent", "Created", "Held", "Told", "Commented",
    "Added", "Filed", "Wrote", "Updated", "Checked", "Read",
)


def test_every_v1_tool_has_a_v2_entry():
    covered = set(entries.V2_READ) | set(entries.V2_WRITE)
    missing = set(_TOOL_VERBS) - covered
    assert not missing, f"v1 tools without a v2 entry: {sorted(missing)}"


def test_every_connector_has_a_read_fallback_and_a_trigger_sub():
    for cid, entry in entries.ENTRIES.items():
        assert "*" in entry.get("reads", {}), f"{cid}: no `*` read fallback"
        assert entry.get("trigger_sub"), f"{cid}: no trigger sub"
        assert entry.get("permission_labels"), f"{cid}: no permission labels"


def test_every_phrase_passes_the_copy_guard():
    for phrase in entries.every_phrase():
        # Slots are engine-filled; scan with them filled innocuously so
        # the braces themselves never mask a violation.
        rendered = phrase.format(
            count=3, need_count=1, when="15:30 on Thursday",
            channel="#platform", target="TP-482", name="GitHub",
            scope_summary="two repositories",
        )
        assert copy_guard.clean(rendered), (phrase, copy_guard.scan(rendered))


def test_no_failure_phrasing_wears_a_done_form():
    done_forms = set(entries.write_done_forms())
    for reason, failure in entries.V2_FAILURE.items():
        action = failure["action"]
        assert action not in done_forms, (reason, action)
        first_word = action.split()[0].rstrip(",")
        assert first_word not in _DONE_OPENERS, (
            f"ND-4: failure {reason!r} opens like a success: {action!r}"
        )


def test_progressive_forms_exist_for_every_tool():
    for tool in set(entries.V2_READ) | set(entries.V2_WRITE):
        assert tool in entries.V2_READ_LIVE, f"{tool}: no progressive form"
        assert entries.V2_READ_LIVE[tool][0].islower(), (
            f"progressive forms are mid-sentence: {entries.V2_READ_LIVE[tool]!r}"
        )


def test_the_engine_actions_are_exactly_the_two_dispatchless_ones():
    assert entries.V2_ENGINE_ACTIONS == (
        "Checked what I can do", "Connected again",
    )


def test_the_canvas_write_phrases_are_verbatim():
    slack = entries.V2_WRITE["slack__send_message"]
    assert slack["action"] == "Told you in Slack"
    assert slack["detail"] == "one line, no thread"
    cal = entries.V2_WRITE["calendar__create_event"]
    assert cal["action"] == "Held {when}"
    assert cal["detail"] == "only you can see it"


def test_the_canvas_rails_are_in_the_connectors_own_words():
    assert entries.V2_RAILS["gmail"] == ("Send anything", "Delete mail")
    assert entries.V2_RAILS["github"] == ("Push or merge",)
    assert entries.V2_RAILS["slack"] == ("Read private DMs",)
    assert entries.V2_RAILS["calendar"][0] == "Invite other people"


def test_flat_views_carry_no_raw_identifier_values():
    for view in (entries.V2_READ, entries.V2_WRITE, entries.V2_FAILURE):
        for verb in view.values():
            for value in verb.values():
                if isinstance(value, str):
                    assert "__" not in value, value
