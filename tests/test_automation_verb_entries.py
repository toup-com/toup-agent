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
    """Templates, scanned as templates — slots are legal here."""
    for phrase in entries.every_phrase():
        assert copy_guard.clean(phrase, rendered=False), (
            phrase, copy_guard.scan(phrase, rendered=False)
        )


def test_every_phrase_renders_through_production_without_a_brace():
    """R31-25. The pin that could not fail, made able to fail.

    This test used to render each template with
    `str.format(count=3, need_count=1, …)` — a kwargs bag more generous
    than any renderer in the product. `{need_count}` was filled here and
    nowhere in production, so the suite stayed green while a founder's
    job sheet read `0 issues moved · {need_count} needs you`.

    A template is now rendered by THE FUNCTIONS THAT SERVE IT and the
    result is scanned in rendered mode, where a brace is a violation.
    Declaring a slot no renderer fills fails this test.
    """
    from app.services import automation_verbs as verbs

    checked = 0
    for cid, entry in entries.ENTRIES.items():
        for tool in entry.get("reads", {}):
            real = None if tool == "*" else tool
            for count in (None, 0, 1, 3):
                got = verbs.turn_action(cid, real, kind="read", count=count)
                for field in ("action", "detail"):
                    text = got[field]
                    assert copy_guard.clean(text), (
                        cid, tool, count, field, text, copy_guard.scan(text)
                    )
                live = verbs.live_sentence(cid, real, count=count)
                assert copy_guard.clean(live), (cid, tool, count, live)
                checked += 1
        for tool in entry.get("writes", {}):
            for audience in ("you", "others"):
                got = verbs.turn_action(cid, tool, kind="write",
                                        target="#all-toup", audience=audience)
                for field in ("action", "detail"):
                    text = got[field]
                    assert copy_guard.clean(text), (
                        cid, tool, audience, field, text, copy_guard.scan(text)
                    )
                checked += 1
        for reason in entries.V2_FAILURE:
            got = verbs.failure_action(cid, reason)
            for field in ("action", "detail"):
                text = got[field]
                assert copy_guard.clean(text), (
                    cid, reason, field, text, copy_guard.scan(text)
                )
            checked += 1
    assert checked > 200, f"the sweep collapsed to {checked} renders"


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
    assert cal["detail"] == "only you can see it"
    # R31-25 parity correction. This used to assert `cal["action"] ==
    # "Held {when}"` and call it canvas-verbatim. The canvas does not
    # say that: `Automations.dc.html` draws "Held 15:30 on Thursday" —
    # a RENDERED example. Someone turned the example into a template
    # with a `{when}` slot, and `turn_action` fills `{target}` and
    # `{channel}` and nothing else, so every held slot shipped the
    # brace. The pin then froze the defect in place as design parity.
    #
    # The action is slot-free until A gives the renderer a `when`
    # filler (CONTRACTS-R31 §2a); the canvas's own wording is the
    # target once it can be rendered.
    assert "{" not in cal["action"], cal["action"]
    assert cal["action"] == "Held time on your calendar"


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
