"""The copy contract has teeth (R30 §5.7/§5.8, D-08).

Three claims, each mutation-honest:
  1. the guard catches every class the contract bans and passes the
     sanctioned strings — asserted needle by needle, so weakening a
     regex or dropping a word fails a named case;
  2. the embedded contract in `copy_guard.py` is byte-equal to
     `fixtures/automations/banned-copy.json` — the file both repos
     carry; drift on either side fails here;
  3. every string the C-owned template modules can emit passes the
     guard — the surfaces D-08 exists to protect.

Pure functions, no DB — runs in the platform sweep.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from app.agent.automations import copy_guard, notification_templates, setup_script

REPO_ROOT = Path(__file__).resolve().parents[2]
BANNED_COPY_JSON = REPO_ROOT / "fixtures" / "automations" / "banned-copy.json"


# ---------------------------------------------------------------- 1. behavior

@pytest.mark.parametrize("needle", copy_guard.BANNED_WORDS)
def test_every_banned_word_is_caught_whole_word(needle):
    hits = copy_guard.scan(f"The agent will {needle} now.")
    assert any(v.needle == needle for v in hits), needle


@pytest.mark.parametrize("needle", copy_guard.BANNED_PHRASES)
def test_every_banned_phrase_is_caught(needle):
    hits = copy_guard.scan(f"Open {needle} to continue.")
    assert any(v.needle == needle for v in hits), needle


def test_whole_word_means_substrings_pass():
    # "polling" is banned on its own, but "unpolluted" and "jobs" (a
    # plural the contract deliberately does not list) contain banned
    # words only as substrings.
    assert copy_guard.clean("The unpolluted stream of pollen.")
    assert copy_guard.clean("Two jobs finished.")  # "job" whole-word only


def test_case_sensitive_means_partial_lowercase_passes():
    # The banned word is "Partial" (the retired status label); the
    # ordinary adjective stays usable.
    assert copy_guard.clean("Ran with a partial view of the inbox.")
    assert not copy_guard.clean("Status: Partial")


def test_the_recorded_jargon_from_the_recordings_is_banned():
    # D-08 verbatim: recording 10:08/12:18 strings must all be caught.
    for sentence in (
        "you can tweak or pause it later in Mission Control",
        "it polls every minute with the 90-second bounded JQL window",
        "The workflow is live",
        "Microsoft Teams is Temporarily unavailable",
        "the Teams connection needs re-authentication",
    ):
        assert not copy_guard.clean(sentence), sentence


def test_iso_timestamps_percent_and_raw_tool_ids_fail():
    assert not copy_guard.clean("last run partial at 2026-08-25T02:52:14Z")
    assert not copy_guard.clean("In progress · 2/3 steps · 66%")
    assert not copy_guard.clean("gmail__list_threads returned 18 rows")


def test_emoji_fails_but_ui_glyphs_pass():
    assert not copy_guard.clean("Overall: ✅ OK")
    assert not copy_guard.clean("⚠ That step didn't finish.")
    assert copy_guard.clean("✓ Read new mail · ✕ Send anything · ● WHY ⋯ ‹ ›")


def test_canvas_mandated_workflow_strings_are_the_only_sanctioned_uses():
    assert copy_guard.clean("Edit workflow")
    assert copy_guard.clean("The whole workflow")
    assert copy_guard.clean("automation · mobile · routine")
    # The same word outside the sanctioned strings still fails.
    assert not copy_guard.clean("The workflow is live")
    assert not copy_guard.clean("Edit workflow settings for the workflow")


# ------------------------------------------------------- 2. one list, pinned

def test_embedded_contract_matches_the_fixture_bytes():
    fixture = json.loads(BANNED_COPY_JSON.read_text())
    embedded = copy_guard.contract_dict()
    assert embedded["banned_words"] == fixture["banned_words"]
    assert embedded["banned_phrases"] == fixture["banned_phrases"]
    assert embedded["whitelist_exact"] == fixture["whitelist_exact"]
    emoji_pattern = next(
        p for p in fixture["banned_patterns"] if p["name"] == "emoji"
    )
    assert embedded["whitelist_glyphs"] == sorted(emoji_pattern["whitelist_glyphs"])
    assert fixture["matching"] == {"whole_word": True, "case_sensitive": True}


# ------------------------------------------------- 3. the C surfaces are clean

def _every_notification_body() -> list[str]:
    bodies = [
        notification_templates.draft_staged_body(),
        notification_templates.auto_pause_body(),
        notification_templates.setup_card("Morning work brief")["title"],
        notification_templates.setup_card("Morning work brief")["body"],
    ]
    for run_kind, vocabulary, status, n in itertools.product(
        ("scheduled", "run_now"),
        ("brief", "changes"),
        ("completed", "failed", "waiting_on_user"),
        (0, 1, 2, 9, 10, 128),
    ):
        bodies.append(notification_templates.notification_body(
            "automation_run",
            {"run_kind": run_kind, "status": status, "vocabulary": vocabulary,
             "needs_count": n, "writes_count": n,
             "failed_connector_name": "GitHub" if status == "failed" else ""},
        ))
    for summary in (
        {"status": "waiting_on_user"},
        {"status": "failed", "failed_connector_name": "GitHub"},
        {"status": "failed"},
        {},
    ):
        bodies.append(notification_templates.notification_body(
            "automation_needs_you", summary))
    return bodies


def test_every_notification_body_passes_the_guard():
    for body in _every_notification_body():
        assert copy_guard.clean(body), (body, copy_guard.scan(body))


def test_every_setup_script_string_passes_the_guard():
    for mode in setup_script.MODES:
        for label in ("tonight", "tomorrow morning", "in a few minutes"):
            for turn in setup_script.setup_turns(
                mode, channel_label="#platform", first_run_label=label,
                scope_lines=[{"text": "Read new mail", "ok": True}],
            ):
                for value in (turn.get("text"), turn.get("action"), turn.get("detail")):
                    if value:
                        assert copy_guard.clean(value), (mode, value)
