"""R31-C — the agent surface: the real re-run, and nothing raw.

Pins for the defects R31-C owns. Each fails on the pre-fix tree:

  R31-04  `automations__test_run` was the only run-shaped tool the model
          could reach, so "Run all of them again" became a synthetic
          fire that answered "TEST RUN STAGED (the write goes out after
          the normal undo window)" and reported the automation's status
          as paused instead of running it.
  R31-25  a verb-dictionary entry declared `{need_count}`, a slot no
          renderer fills, and the brace reached a founder's job sheet.
  R31-28  a line of coaching addressed to the model was a tool's whole
          return value, so it was rendered as what the tool had done.
  R31-07  a failure named no account: "an account refused" on a run
          that had no refusing account at all.

Pure functions and prompt text — no DB, platform lane.
"""

from __future__ import annotations

import pytest

from app.agent.automations import copy_guard, notification_templates
from app.agent.skills.builtins.automations.skill import AutomationsSkill
from app.config import settings


def _tool_names(skill: AutomationsSkill) -> list[str]:
    return [t["name"] for t in skill.get_tools()]


# ---------------------------------------------------------------- R31-04

def test_run_now_is_reachable_and_the_test_path_is_not():
    names = _tool_names(AutomationsSkill())
    assert "automations__run_now" in names
    assert "automations__test_run" not in names


def test_the_test_path_returns_under_the_dev_flag_only(monkeypatch):
    monkeypatch.setattr(settings, "automations_dev_fast_lane", True)
    monkeypatch.setattr(settings, "environment", "development")
    assert "automations__test_run" in _tool_names(AutomationsSkill())

    # A stray env var on a production tenant changes nothing — the
    # dev gate is ANDed with the environment at its use site.
    monkeypatch.setattr(settings, "environment", "production")
    assert "automations__test_run" not in _tool_names(AutomationsSkill())


@pytest.mark.asyncio
async def test_the_test_path_is_refused_at_the_door_too(monkeypatch):
    """Unregistering a tool does not un-teach its name.

    A model can emit a name it saw earlier in the conversation or in
    its own history, so the array is not the only gate that has to
    hold. The refusal names the replacement rather than just saying no.
    """
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "automations_dev_fast_lane", False)
    out = await AutomationsSkill().execute_tool(
        "automations__test_run", {"automation_id": "a-1"},
        _ctx(),
    )
    assert "automations__run_now" in out
    assert "STAGED" not in out


@pytest.mark.asyncio
@pytest.mark.parametrize("status,detail,expect_display,forbid", [
    (404, "Not found", None, "STAGED"),
    (409, {"code": "already_running",
           "sentence": "It is running now — step 2 of 5."},
     "It is already running", "Already running"),
    (409, {"code": "v1_not_supported",
           "sentence": "This automation predates run-now."},
     "Could not start the run", "paused"),
])
async def test_every_run_refusal_is_reported_as_itself(
    monkeypatch, status, detail, expect_display, forbid,
):
    """The handler calls the same route the Run it now button calls, so
    every refusal the UI can meet reaches the model here. None of them
    may come back as a STATUS — reporting one instead of running is the
    whole of R31-04.
    """
    from fastapi import HTTPException

    import app.api.automations as api

    monkeypatch.setattr(settings, "automations_enabled", True)

    async def refuse(_automation_id):
        raise HTTPException(status_code=status, detail=detail)

    monkeypatch.setattr(api, "run_now", refuse)
    out = await AutomationsSkill().execute_tool(
        "automations__run_now", {"automation_id": "a-1"}, _ctx(),
    )
    assert getattr(out, "display", None) == expect_display
    assert forbid not in str(out)


@pytest.mark.asyncio
async def test_a_started_run_tells_the_model_to_say_one_line(monkeypatch):
    """"I ran the work brief again. Its status is **paused**. This
    re-run is staged to post…" — the model narrating a run it did not
    watch. The run reports itself in its own thread; a summary here is
    a second account of the same run that will disagree with the first.
    """
    import app.api.automations as api

    monkeypatch.setattr(settings, "automations_enabled", True)

    async def fired(_automation_id):
        return {"fired": True, "status": "running"}

    monkeypatch.setattr(api, "run_now", fired)
    out = await AutomationsSkill().execute_tool(
        "automations__run_now", {"automation_id": "a-1"}, _ctx(),
    )
    assert out.display == "Started the run"
    assert "one short line" in str(out)
    assert "what it found" in str(out)


def test_the_prompt_routes_every_run_phrasing_to_the_engine():
    prompt = AutomationsSkill().get_system_prompt_section()
    assert prompt
    for phrasing in ("run it again", "run all of them again", "try again"):
        assert phrasing in prompt, phrasing
    assert "automations__run_now" in prompt
    # The build order must no longer send the model through the
    # synthetic path — that instruction is what put "TEST RUN STAGED"
    # in front of a user.
    assert "automations__test_run" not in prompt


def test_the_prompt_forbids_answering_a_run_request_with_a_status():
    """The founder asked for a run and was told a status.

    `Its status is **paused**` is not a wrong answer to "run all of
    them again" — it is a way of not answering. The rule has to be in
    the prompt, because nothing downstream can tell a status sentence
    from a run.
    """
    prompt = AutomationsSkill().get_system_prompt_section()
    assert "status is paused" in prompt
    assert "never with a status" in prompt


def test_no_prompt_surface_teaches_the_words_the_user_must_not_read():
    prompt = AutomationsSkill().get_system_prompt_section() or ""
    for banned in ("undo window", "TEST RUN"):
        assert banned not in prompt, banned


def test_the_hard_rules_promise_only_what_the_engine_actually_does():
    """These are promises made to a user about the engine's behaviour,
    so each must be true of the engine as it IS.

    A draft of this section deleted "3 failures in a row pauses the
    automation" — which is true today (`sweep._sweep_auto_pause`,
    AUTOMATION_AUTO_PAUSE_FAILURES = 3) — and replaced it with "one
    broken account never pauses an automation and never stops a run",
    which is R31 §4.2a's intent and not yet the code: `on_error` still
    defaults to "fail" and a failed read step finalizes the run as
    failed. Telling a user their automation cannot be paused by a
    broken account, and having it paused that afternoon, is worse than
    saying nothing.
    """
    prompt = AutomationsSkill().get_system_prompt_section() or ""
    assert "3 failed runs in a row pause the automation" in prompt
    assert "never pauses an automation" not in prompt

    # And the claim is still true of the code that makes it true.
    from app.config import settings as _s
    assert int(getattr(_s, "AUTOMATION_AUTO_PAUSE_FAILURES", 0) or
               getattr(_s, "automation_auto_pause_failures", 3)) == 3


# ---------------------------------------------------------------- R31-28

def test_the_memory_summary_is_a_sentence_not_a_prompt_fragment():
    from app.agent.skills.builtins.automations import skill as sk

    for result, expected in (
        ({}, "Looked in memory · nothing matched"),
        ({"facts": [1], "episodes": []}, "Looked in memory · 1 fact"),
        ({"facts": [1, 2], "episodes": [3]},
         "Looked in memory · 2 facts and 1 run"),
    ):
        summary = sk._recall_summary(result)
        assert summary == expected, (result, summary)
        assert copy_guard.clean(summary), copy_guard.scan(summary)
        # The thing that shipped: an imperative addressed to the model.
        assert "Say so" not in summary


# ---------------------------------------------------------------- R31-07

def test_a_failure_with_no_account_blames_no_account():
    body = notification_templates.notification_body(
        "automation_run",
        {"run_kind": "scheduled", "status": "failed", "vocabulary": "brief"},
    )
    assert "an account" not in body
    assert "refused" not in body
    assert copy_guard.clean(body), copy_guard.scan(body)


def test_a_failure_with_an_account_names_it():
    body = notification_templates.notification_body(
        "automation_run",
        {"run_kind": "scheduled", "status": "failed", "vocabulary": "brief",
         "failed_connector_name": "GitHub"},
    )
    assert "GitHub refused" in body
    assert copy_guard.clean(body), copy_guard.scan(body)


# ---------------------------------------------------------------- R31-25

def test_no_dictionary_entry_declares_a_slot_nobody_fills():
    """The pin that could not fail, made able to fail.

    The old sweep rendered every template with
    `str.format(count=3, need_count=1, …)`. Production renders with
    `automation_verbs._n`, which substitutes `{n}` and `{count}` and
    nothing else. So the suite proved a string that never existed.
    """
    from app.services import automation_verb_entries as entries
    from app.services import automation_verbs as verbs

    for cid, entry in entries.ENTRIES.items():
        for tool in entry.get("reads", {}):
            real = None if tool == "*" else tool
            for count in (None, 0, 1, 7):
                got = verbs.turn_action(cid, real, kind="read", count=count)
                assert "{" not in got["action"] + got["detail"], (cid, tool, got)
        for tool in entry.get("writes", {}):
            for audience in ("you", "others"):
                got = verbs.turn_action(cid, tool, kind="write",
                                        target="#all-toup", audience=audience)
                assert "{" not in got["action"] + got["detail"], (
                    cid, tool, audience, got
                )


def test_a_read_verb_reads_like_a_read():
    """`0 issues moved` on a turn that moved nothing (E-09)."""
    from app.services import automation_verbs as verbs

    got = verbs.turn_action("jira", "jira__search_issues", kind="read", count=3)
    assert got == {"action": "Checked your board", "detail": "3 open issues"}


# ---------------------------------------------------------------- R31-24

def _thread_prompt(**over) -> str:
    from app.agent.automations.interview import prompt_section
    ctx = {"automation_id": "a-1", "name": "Morning work brief",
           "rule_text": "", "status": "active", "facts": {}}
    ctx.update(over)
    return prompt_section(ctx)


def test_the_thread_persona_forbids_markdown_and_says_why():
    """A thread bubble renders plain text, so markdown is DISPLAYED.

    The founder read `Its status is **paused**` with the asterisks
    (E-42), and `- **Gmail:** …` as a literal dash and stars (E-34).
    The persona has to say the reason, not just the rule: a model told
    "no markdown" with no reason will reach for it again the moment a
    list feels clearer.
    """
    section = _thread_prompt()
    assert "plain text" in section
    for forbidden in ("No bold", "no italics", "no backticks"):
        assert forbidden in section, forbidden
    # It must beat the formatting section, which is assembled AFTER
    # this one (`prompt_profile._FULL_SECTIONS`) and tells the model to
    # use simple Markdown. Being right is not enough; it has to say it
    # outranks the later instruction.
    assert "overrides any formatting guidance later" in section


def test_the_thread_persona_forbids_quick_reply_syntax():
    """`[[Label]]` becomes a tappable button that speaks AS the user.

    The main chat binds that syntax to its quick-reply handler, which
    is the standing candidate for the `Run all of them again` turns
    that appeared as USER messages nobody typed (E-01, E-40).
    """
    section = _thread_prompt()
    assert "double square brackets" in section
    assert "never said" in section


def test_the_thread_persona_carries_no_raw_tool_id():
    """A persona that names tool ids invites them into the copy."""
    assert "__" not in _thread_prompt()


def test_the_thread_persona_orders_a_failure_account_reason_fix():
    section = _thread_prompt()
    assert "which account, the real reason, and what fixes it" in section
    assert "the organisation has not approved Toup yet" in section
    assert 'Never "an account"' in section
    # An honest unknown beats an inferred cause.
    assert "I could not tell why" in section
    # The agent's OWN reading continues past a broken account — but the
    # persona must not promise anything about what a SCHEDULED run
    # does, because `on_error` still defaults to "fail" and three failed
    # runs auto-pause. A draft of this said "one broken account never
    # pauses an automation"; that is §4.2a's intent, not the code.
    assert "That is about your own reading" in section
    assert "never pauses an automation" not in section


def test_the_thread_persona_makes_a_fresh_question_a_run():
    """R31-08. 40 seconds of "Looking at that now…", then prose with no
    job card and no per-account rows (E-38)."""
    section = _thread_prompt()
    assert "is a run, not a paragraph you compose" in section
    assert "one short paragraph per account" in section
    # Teams was named in an answer though it is not on the canvas.
    assert "Consult ONLY the accounts this automation has" in section


def test_the_thread_persona_never_answers_a_run_with_a_status():
    section = _thread_prompt()
    assert "never with its status" in section
    assert "a way of not doing it" in section


def test_the_thread_persona_scopes_memory_by_subject_not_by_place():
    """The re-scope rule, in the words the model reads.

    `fixtures/automations/memory-scope.json` is the rule of record;
    this is the half that has to be true at write time.
    """
    section = _thread_prompt()
    assert "follows what it is ABOUT, not where it was said" in section
    assert "it belongs to the person" in section
    # Status is never a memory (ND-2/ND-3, D-20).
    assert "never a memory" in section


def test_a_zero_is_never_a_failed_read():
    """Gmail's run row said `0 new threads` with 7 unread on screen
    (E-41). Whatever the cause, a failed read may not wear a count."""
    section = _thread_prompt()
    assert "A zero is a fact, not a shrug" in section


# ------------------------------------------------- R31-13 · the string table

def _reason_strings() -> dict:
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "fixtures" / "automations" / "reason-strings.json"
    return json.loads(path.read_text())


def test_the_string_table_covers_every_state_on_every_surface():
    """A wires this file mechanically and edits no string, so a missing
    row is not a blank on screen — it is a KeyError in the renderer, or
    a fallback that says something nobody wrote."""
    table = _reason_strings()
    surfaces = set(table["surfaces"])
    expected_states = {"connected", "expired", "revoked", "scope_missing",
                       "org_approval_needed", "not_connected"}
    assert set(table["states"]) == expected_states
    for state, row in table["states"].items():
        assert surfaces <= set(row), (state, sorted(surfaces - set(row)))


def test_no_failing_state_can_read_connected():
    """R31-13, the whole of it. Outlook's own sheet showed the pill
    `Connected` directly above `Last use · Could not connect · access
    expired` (E-27)."""
    table = _reason_strings()
    for state, row in table["states"].items():
        if state == "connected":
            continue
        assert row["pill"] != "Connected", state
        assert row["button_label"], f"{state} offers no way out"
        assert row["fix"] in table["fix_buttons"], state


def test_a_transient_reason_keeps_connected_and_offers_try_again():
    """A rate limit is not a broken connection. Moving the account off
    `connected` for it would send the user to reconnect something that
    is working."""
    table = _reason_strings()
    for code, row in table["reason_codes"].items():
        if code.startswith("$"):
            continue
        if row["transient"]:
            assert row["state"] == "connected", code
            assert row["fix"] == "retry", code
        else:
            assert row["state"] != "connected", code
            assert row["fix"] != "retry", code


def test_no_status_string_in_the_table_says_an_account():
    """R31-07. `Could not reach an account` names nobody."""
    table = _reason_strings()
    for state, row in table["states"].items():
        for surface, text in row.items():
            if not isinstance(text, str) or surface.startswith("$"):
                continue
            assert "an account" not in text, (state, surface, text)
    for name, form in table["forms"].items():
        if isinstance(form, str):
            assert "an account" not in form, (name, form)


def test_every_form_the_dispatch_names_lives_here_once():
    forms = _reason_strings()["forms"]
    for key, text in (
        ("tried_names_need_you", "Tried {t} · {names} need you"),
        ("tried_did_not_finish", "Tried {t} · it did not finish"),
        ("ran_n_of_m", "Ran {t} · {n} of {m} accounts · needs you"),
        ("ran_brief_not_posted", "Ran {t} · brief not posted · needs you"),
        ("paused_ran_once", "Paused · ran once {t}"),
        ("could_not_reach_2", "Could not reach {A} and {B}"),
        ("start_it_over", "Start it over — the stopped run is dropped."),
        ("waiting_org_full", "Waiting for the organisation to approve Toup"),
        ("waiting_org_short", "Waiting for the organisation"),
        ("needs_reconnecting", "Needs reconnecting"),
        ("reconnected_just_now", "Reconnected just now"),
        ("needs_you_stamp", "NEEDS YOU"),
    ):
        assert forms[key] == text, (key, forms.get(key))


def test_every_table_string_passes_the_copy_guard_as_a_template():
    """Slots are legal in a template; everything else is not. The
    RENDERED scan is A's, at the serializer — this half proves the
    words themselves are clean before anything fills them."""
    table = _reason_strings()
    for block in ("states", "reason_codes"):
        for key, row in table[block].items():
            if key.startswith("$"):
                continue
            for surface, text in row.items():
                if not isinstance(text, str) or surface.startswith("$"):
                    continue
                assert copy_guard.clean(text, rendered=False), (
                    key, surface, text, copy_guard.scan(text, rendered=False)
                )
    for name, form in table["forms"].items():
        if isinstance(form, str) and not name.startswith("$"):
            assert copy_guard.clean(form, rendered=False), (
                name, form, copy_guard.scan(form, rendered=False)
            )


# ------------------------------------------- R31-07 · failure narration

_ORG = ("I could not read GitHub — the organisation has not approved Toup "
        "yet. An owner can approve it from the button below, and I will "
        "pick up from there.")


def _record(status: str, **over) -> dict:
    rec = {
        "automation": {"title": "Morning work brief"},
        "run_kind": "scheduled", "vocabulary": "brief", "status": status,
        "steps": [
            {"step_ref": "s1", "connector_name": "Gmail", "ok": True,
             "action": "Read your unread mail", "items": []},
            {"step_ref": "s2", "connector_name": "GitHub", "ok": False,
             "failure_reason": _ORG, "items": []},
        ],
    }
    rec.update(over)
    return rec


@pytest.mark.parametrize("status", ["partial", "failed", "completed"])
def test_any_failed_source_gets_failure_rules_whatever_the_status(status):
    """A PARTIAL run is the ordinary failure shape, and it had none.

    `_FAILED_RULES` fired only on `status == "failed"`, so the common
    case — one account broken, the rest read — reached the model with
    the brief's ranking rules and nothing about failure. R31-D fired a
    real run on 26 August: GitHub failed, the run finished `partial`,
    and the narration said "GitHub did not respond" for an account
    whose actual problem was that the organisation had never approved
    Toup. The agent knew the true reason — it gave it correctly when
    asked in the thread minutes later — so nothing was missing except
    an instruction to use the sentence it was handed.
    """
    from app.agent.automations import narrator

    prompt = narrator.build_prompt(_record(status))
    assert "SOME SOURCES FAILED" in prompt, status
    assert _ORG in prompt, status


def test_a_question_run_is_answered_not_ranked():
    """R31-08 / §4.9. The founder asked for "everything latest in all
    chanels" and got 40 s of a loading pill and then prose — no job
    card, no per-account rows, and Teams named although it is not on
    that automation's canvas (E-38).

    A question run carries no result turn: the answer IS the result,
    and a ranked five-tier brief for "what is the latest in Gmail"
    would be the changes-vocabulary mistake in another costume.
    `_validate_result` already exempts this run kind; it had no shape
    to be narrated into.
    """
    from app.agent.automations import narrator

    prompt = narrator.build_prompt(_record("completed", run_kind="question"))
    assert "NO result turn" in prompt
    assert "ONE short paragraph per account" in prompt
    assert "DO FIRST" not in prompt          # the brief's tiers
    assert "CHANGED YOUR WEEK" not in prompt  # the changes tiers
    # A count the user cannot check is the thing being fixed.
    assert "must be a count you actually got back" in prompt
    # And an unread account is named, not omitted.
    assert "a silent omission reads as" in prompt

    ranked = narrator.build_prompt(_record("completed"))
    assert "DO FIRST" in ranked, "a scheduled brief is still ranked"


def test_a_run_with_no_failed_source_gets_no_failure_rules():
    from app.agent.automations import narrator

    clean = _record("completed")
    clean["steps"] = [s for s in clean["steps"] if s["ok"]]
    prompt = narrator.build_prompt(clean)
    assert "SOME SOURCES FAILED" not in prompt


def test_a_real_reason_is_quoted_verbatim():
    """§4.4: the narration quotes `thread_sentence` verbatim and adds
    at most one sentence. A paraphrase moves the fix — "it did not
    answer" sends the user to wait, when what they must do is ask an
    owner to approve an OAuth app."""
    from app.agent.automations import narrator

    prompt = narrator.build_prompt(_record("partial"))
    assert "EXACTLY AS GIVEN" in prompt
    assert "do not reword it" in prompt
    assert 'never "an account"' in prompt
    # And the zero that is not a zero (E-41).
    assert "never report a failed read as a count of zero" in prompt


def test_a_fragment_is_never_mandated_as_the_cause():
    """The fix that would have made things worse.

    `failure_reason` is DOCUMENTED as the string table's
    `thread_sentence`, but its only producer today is `executor_v2`
    writing `turn["detail"]` — the verb dictionary's short fragment —
    and `_failure_reason` does not recognise `org_approval_needed` at
    all, so the exact GitHub case this was written for arrives as "it
    did not answer".

    An unconditional "quote it verbatim" would therefore MANDATE the
    vague answer the block exists to prevent: the model could
    previously improvise its way to the true reason, and would now be
    forbidden from doing anything but repeating the wrong one. So the
    instruction is conditional on what was actually supplied.
    """
    from app.agent.automations import narrator

    record = _record("partial")
    record["steps"][1]["failure_reason"] = "it did not answer"
    prompt = narrator.build_prompt(record)

    assert "EXACTLY AS GIVEN" not in prompt
    assert "not diagnoses" in prompt
    assert "do NOT supply a cause of your own" in prompt
    assert "say you do not know why" in prompt


def test_an_unknown_failure_says_so_instead_of_naming_a_cause():
    """`_failure_reason` returns "unreachable" for anything it does not
    recognise, which fell through to the default ("Could not connect",
    "it did not answer") — so "we do not know why" was rendered as a
    specific diagnosis, and the user was sent to wait for a service
    that was answering perfectly well."""
    from app.services import automation_verbs as verbs

    assert verbs.failure_action("github", "unreachable") == {
        "action": "Could not reach GitHub",
        "detail": "I could not tell why",
    }


# ---------------------------------------------------- R31-18 · rules

def test_rules_are_extracted_from_the_constraints_already_stated():
    """`LINES IT WILL NOT CROSS 0` on an automation whose own step read
    `Told you in Slack · one line, no thread` (E-20, E-21)."""
    from app.agent.automations import rule_extraction as rx

    rows = rx.extract_rules(
        description="Post one line in #all-toup, one line, no thread.",
        setup_text=["Only unread mail please, and never post anywhere else."],
        steps=["Told you in Slack · one line, no thread"],
    )
    texts = {r["text"] for r in rows}
    assert "One line only." in texts
    assert "No thread." in texts
    assert "Only unread mail please." in texts
    assert any(t.startswith("Never post anywhere") for t in texts), texts
    # Every row is auditable back to the words it came from.
    for row in rows:
        assert row["origin"] in rx.ORIGINS
        assert row["source"]


def test_the_same_constraint_from_two_routes_is_one_rule():
    """`one line, no thread` arrives from the description AND from the
    step that renders it. Two rows would show the user the same line
    twice in the one place they look to see what it will not do."""
    from app.agent.automations import rule_extraction as rx

    rows = rx.extract_rules(
        description="one line, no thread",
        steps=["Told you in Slack · one line, no thread"],
    )
    assert sorted(r["text"] for r in rows) == ["No thread.", "One line only."]


def test_the_back_fill_is_idempotent():
    """It runs as a migration over automations nobody is watching, and
    a half-finished pass has to be safe to re-run."""
    from app.agent.automations import rule_extraction as rx

    first = rx.extract_rules(description="Never post anywhere.")
    again = rx.extract_rules(description="Never post anywhere.",
                             existing=first)
    assert first and again == []


def test_extraction_invents_nothing_from_an_ordinary_description():
    """The safe direction is asymmetric but neither side is free: an
    invented rule lists a line the user never drew, in the one place
    that claims to show exactly that."""
    from app.agent.automations import rule_extraction as rx

    assert rx.extract_rules(
        description="Open Jira issues, unread Gmail and Outlook, "
                    "then post a summary.",
        steps=["Read your Jira board", "Read new mail in Gmail"],
    ) == []


@pytest.mark.parametrize("rule", [
    "Never post in a channel — DM me instead.",
    "Leave anything finance owns alone.",
    "Hold time on my calendar rather than booking meetings.",
])
def test_the_canvas_rules_round_trip_byte_for_byte(rule):
    """The approved canvas draws exactly these three under LINES IT
    WILL NOT CROSS. Extraction must return the user's own sentence,
    not a tidied version of it: a rule is the line the agent will not
    cross, and rewording it moves the line where nobody can see it
    move. The em-dash case is why a clause is not cut at punctuation —
    splitting there would keep "Never post in a channel." and silently
    drop "DM me instead", which is the half that says what to do.
    """
    from app.agent.automations import rule_extraction as rx

    rows = rx.extract_rules(description=rule)
    assert [r["text"] for r in rows] == [rule]


def test_a_fragment_is_not_a_rule():
    from app.agent.automations import rule_extraction as rx

    assert rx.extract_rules(description="only it") == []


# ---------------------------------------------------------------- R31-37

@pytest.mark.parametrize("tools,expected", [
    ([], "brief"),
    (["slack__send_message"], "brief"),
    (["gmail__create_draft"], "brief"),
    (["slack__send_message", "gmail__create_draft"], "brief"),
    (["calendar__create_event"], "changes"),
    (["jira__add_comment"], "changes"),
    (["slack__send_message", "calendar__create_event"], "changes"),
])
def test_a_brief_that_posts_is_still_a_brief(tools, expected):
    """The founder's Morning work brief read five accounts and posted
    one line, and was rendered `CHANGED YOUR WEEK · 1 item` (E-35).

    The old derivation asked "does it write anything that is not a
    draft?", so delivering the brief made it a change-making run.
    Posting is how a brief is delivered — `workflow.output_block`
    already leads a posting automation with "A brief on your phone".
    """
    from app.agent.automations.narrator import vocabulary_for

    assert vocabulary_for(tools) == expected


# ---------------------------------------------------------------- R31-22

def test_no_failure_sends_the_user_to_the_main_chat_to_set_one_up():
    """`Set up the "Replies drafted before you wake" automation for me.`
    appeared as a USER message in the main chat, the agent asked its
    setup questions there, and no card was ever created (E-45, E-46).
    Two of the three sentences that produced it were ours."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in (root / "app" / "agent" / "automations" / "describe_compile.py",
                 root / "app" / "api" / "automations.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # the note recording what it used to say
            for needle in ("in chat and I", "to me in chat"):
                if needle in line:
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, offenders


def test_no_automations_tool_returns_a_bare_payload():
    """R31-28, structurally.

    A skill tool's JSON passes STRAIGHT THROUGH `client_summary` as its
    user-facing summary — that is deliberate for tools whose payload a
    client parses, and it is why `automations__memory_recall` shipped a
    line of prompt coaching as its summary on a founder's job sheet.
    Every payload-returning handler must declare a `display` sentence,
    so what a person reads is never what the model reads.

    Source probe rather than a call-through: these handlers each open
    their own DB session, and a mock deep enough to reach the return
    statement would be a second model of the file rather than a test
    of it.
    """
    import pathlib
    import re

    path = (pathlib.Path(__file__).resolve().parents[1] / "app" / "agent" /
            "skills" / "builtins" / "automations" / "skill.py")
    src = path.read_text()
    offenders = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(src.splitlines(), 1)
        if re.search(r"return\s+.*_as_json\(", line)
    ]
    assert not offenders, offenders

    # And every declared display is a sentence, not a payload or a
    # fragment addressed to the model.
    for display in re.findall(r'display=(".*?"|f".*?")', src):
        text = display.strip('f"')
        assert not text.startswith(("{", "[")), display
        assert "Say so" not in text, display
        assert copy_guard.clean(text.replace("{automation.name}", "X")), display


# ------------------------------------------- R31-22 · the setup script

def test_the_setup_opening_names_the_channel_not_the_mode():
    """Both call sites do `mode, label = mode_of(...)` and pass `label`,
    which for `posts` is already "posts to #all-toup". The opening then
    read "post one line in posts to #all-toup, nothing else." and the
    capability check's detail read "posts to posts to #all-toup" — in
    the first four turns of a new automation's thread. The module's own
    docstring said passing the mode label there was harmless.
    """
    from app.agent.automations.setup_script import setup_turns

    for label in ("posts to #all-toup", "#all-toup"):
        turns = setup_turns(mode="posts", channel_label=label,
                            first_run_label="tomorrow 8:00",
                            scope_lines=[])
        assert turns[0]["text"] == (
            "Here is what I will be able to do — post one line in "
            "#all-toup, nothing else."
        ), label
        assert turns[1]["detail"] == "posts to #all-toup", label


def test_the_capability_check_says_what_it_can_and_cannot_do():
    """It was called with `scope_lines=[]` from both creation paths, so
    the one turn whose whole job is to say what an automation will and
    will not be able to do was empty on every automation ever made."""
    from app.agent.automations.setup_script import (
        scope_lines_from, setup_turns,
    )

    lines = scope_lines_from({
        "can": [{"id": "read_new_mail", "label": "Read new mail"},
                {"id": "write_drafts", "label": "Write drafts"}],
        "cant": [{"id": "send", "label": "Send anything", "kind": "rail"}],
    })
    assert lines == [
        {"text": "Read new mail", "ok": True},
        {"text": "Write drafts", "ok": True},
        {"text": "Send anything", "ok": False},
    ]
    turns = setup_turns(mode="drafts_only", channel_label="",
                        first_run_label="tonight", scope_lines=lines)
    assert turns[1]["steps"] == lines


def test_capability_lines_name_the_account_when_there_are_several():
    """"Read new mail" twice, unlabelled, is the workflow's duplicate
    step problem in a different turn (E-20)."""
    from app.agent.automations.setup_script import scope_lines_from

    lines = scope_lines_from(
        {"can": [{"id": "read_new_mail", "label": "Read new mail"}],
         "cant": []},
        connector_name="Outlook",
    )
    assert lines == [{"text": "Read new mail · Outlook", "ok": True}]


def test_write_time_scoping_uses_the_same_rule_as_the_back_fill():
    """The migration and the classifier have to agree, or the Memory
    tab means one thing for facts written before it ran and another
    for facts written after."""
    import json
    import pathlib

    from app.agent.automations.interview import _extraction_prompt

    prompt = _extraction_prompt(
        {"name": "Morning work brief", "rule_text": "", "facts": {}},
        "Marcus is on holiday until Friday.", "Noted.",
    )
    assert "deleted tomorrow" in prompt
    assert 'answer "global"' in prompt
    # Never a memory (ND-2/ND-3, D-20).
    assert "NEVER file what an automation is or does" in prompt

    root = pathlib.Path(__file__).resolve().parents[2]
    rule = json.loads(
        (root / "fixtures" / "automations" / "memory-scope.json").read_text()
    )
    assert rule["ambiguity"]["resolve_to"] == "global"
    assert "deleted tomorrow" in rule["rule"]["1_survival_test"]["question"]


def _ctx():
    from types import SimpleNamespace
    return SimpleNamespace(user_id="u-1", conversation_id=None,
                           message_id=None, job_id=None)
