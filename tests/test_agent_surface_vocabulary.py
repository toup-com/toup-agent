"""What the agent's own prompt teaches it to say (R30 §5.7, §14, ND-18).

Three live defects this round traced to strings the model READS rather
than to code it runs, and none of them were catchable by a test that
calls a tool function directly:

  ND-18 — `routines__list`'s description told the model to answer
  "what automations do I have set up?" from the routines list, so the
  agent read the founder's reminders ("Eat tea", "Do the dishes") back
  as automations: 19 instead of 4.

  D-08 — the recordings caught the agent saying "you can tweak or pause
  it later in Mission Control". The copy contract bans that phrase in
  user-facing output; the routines prompt was *instructing* it.

  §14 rule 1 — a tool description carrying a "do this BEFORE answering
  anything" posture competes for the early iterations a setup
  conversation needs. Flow postures belong in the prompt section that
  owns the flow, never in a description every turn sees.

These are invariants over the assembled surface, deliberately not
exact-string pins: they stay true through rewording and fail the moment
a surface starts claiming something that is not its to claim.

Pure string reads — no DB, no model. Platform sweep.
"""

from __future__ import annotations

import re

import pytest

from app.agent.automations.copy_guard import BANNED_PHRASES
from app.agent.skills.builtins.automations.skill import AutomationsSkill
from app.agent.skills.builtins.routines.skill import RoutinesSkill
from app.agent.skills.builtins.triggers.skill import TriggersSkill

AUTOMATIONS = AutomationsSkill()
ROUTINES = RoutinesSkill()

#: EVERY builtin whose strings ride a real turn. The first version of
#: this file swept two skills and an assembly probe immediately found
#: the same banned phrase in a third — a scope that is not "all of them"
#: is a scope that misses one.
ALL_SKILLS = [AUTOMATIONS, ROUTINES, TriggersSkill()]


def _descriptions(skill) -> list[tuple[str, str]]:
    """(tool name, every human string in its schema) for one skill."""
    out: list[tuple[str, str]] = []
    for tool in skill.get_tools():
        name = tool["name"]
        parts = [tool.get("description") or ""]
        schema = tool.get("input_schema") or {}
        for prop in (schema.get("properties") or {}).values():
            if isinstance(prop, dict) and prop.get("description"):
                parts.append(str(prop["description"]))
        out.append((name, "\n".join(parts)))
    return out


def _surfaces(skill) -> list[tuple[str, str]]:
    """Everything of a skill the model reads: its tool strings plus its
    system-prompt section."""
    out = _descriptions(skill)
    section = skill.get_system_prompt_section() or ""
    out.append((f"{skill.meta.name}:prompt_section", section))
    return out


ALL_SURFACES = [row for skill in ALL_SKILLS for row in _surfaces(skill)]


# ------------------------------------------------------------ ND-18

#: Skills whose rows are NOT the user's automations. Anything here that
#: offers itself as the answer to an inventory question is ND-18's shape.
#: Triggers earned its place the hard way: a payload trace of the live
#: turn showed the model calling `triggers__list` for an automations
#: question, because its description invited "what triggers do I have?"
#: — a noun the user never says and the copy contract bans.
NON_AUTOMATION_SKILLS = [ROUTINES, TriggersSkill()]


@pytest.mark.parametrize(
    "skill", NON_AUTOMATION_SKILLS, ids=lambda s: s.meta.name)
def test_no_other_surface_claims_to_answer_an_automations_question(skill):
    """The ND-18 root: a non-automations surface offering itself as the
    answer to "what automations do I have". Reminders, scheduled tasks
    and triggers are not automations to a user, whatever is reachable
    from the tool."""
    offenders = []
    for name, text in _surfaces(skill):
        for match in re.finditer(r"[^.\n]*automations?[^.\n]*", text, re.I):
            clause = match.group(0)
            # A DISCLAIMER is the fix, not the defect: a clause that
            # denies the claim, or redirects to the automations surface,
            # is exactly what we want this file to say.
            if re.search(r"\b(not|never|separate|instead)\b|automations__",
                         clause, re.I):
                continue
            # What is left is an offer to answer for automations.
            if re.search(r"(what|which|how many|list|show|all)\b[^.]{0,40}"
                         r"automations?", clause, re.I):
                offenders.append(f"{name}: {clause.strip()}")
    assert not offenders, (
        "a routines surface claims the automations question:\n"
        + "\n".join(offenders)
    )


def test_the_routines_surface_does_not_call_its_own_rows_automations():
    """The reinforcing half: if the section headlines routines AS
    automations, the model merges the two lists even without the
    explicit instruction."""
    section = ROUTINES.get_system_prompt_section() or ""
    headline = section.split("\n", 1)[0]
    assert "automation" not in headline.lower(), (
        f"the routines section headlines itself as automations: {headline!r}"
    )
    # The self-description sentence, wherever it sits.
    assert not re.search(
        r"you can (create|build|author)[^.]{0,40}automations?[^.]{0,40}"
        r"routines__",
        section, re.I,
    ), "the routines section still offers routines__* as the automation builder"


def test_an_automations_surface_states_where_the_count_comes_from():
    """The positive half: the model must be told, in the section that
    owns the surface, that the automation list IS automations__list."""
    section = AUTOMATIONS.get_system_prompt_section() or ""
    assert re.search(r"automations__list", section), (
        "the automations section never names automations__list as the "
        "source of truth for what the user has"
    )
    assert re.search(r"reminder", section, re.I), (
        "the automations section never distinguishes reminders, so the "
        "model has nothing to separate them by"
    )



def _teaches(text: str, phrase: str) -> bool:
    """True when `text` puts `phrase` in the model's mouth. A phrase the
    text is BANNING ("never engine jargon (no 'Mission Control', ...)")
    is the copy contract being enforced, not taught — the automations
    section names the forbidden words on purpose."""
    for match in re.finditer(re.escape(phrase), text):
        before = text[max(0, match.start() - 40):match.start()]
        if re.search(r"\b(no|never|not|avoid|don't|banned|forbidden)\b"
                     r"[\s\'\"“‘(]*$", before, re.I):
            continue
        return True
    return False


# ------------------------------------------------------------- D-08

@pytest.mark.parametrize("phrase", sorted(BANNED_PHRASES))
def test_no_prompt_surface_teaches_a_banned_phrase(phrase):
    """The copy contract bans these in user-facing output; a prompt that
    instructs one produces it (D-08's "Mission Control" came from here,
    not from the model's imagination)."""
    offenders = [
        f"{name}: ...{text[max(0, text.find(phrase) - 60):text.find(phrase) + 60]}..."
        for name, text in ALL_SURFACES
        if _teaches(text, phrase)
    ]
    assert not offenders, (
        f"a prompt surface teaches the banned phrase {phrase!r}:\n"
        + "\n".join(offenders)
    )


def _model_visible_literals(module) -> list[tuple[int, str]]:
    """Every string literal in a skill module that is NOT a docstring or
    comment — i.e. every string that can reach the model, whether it
    rides a description, a prompt section, or a tool RESULT. D-08's
    live source was the last kind: `routines__remind` returned a `hint`
    telling the user where to change their reminder, and the model
    dutifully paraphrased the internal surface name."""
    import ast
    import inspect

    source = inspect.getsource(module)
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value not in docstrings
    ]


@pytest.mark.parametrize("phrase", sorted(BANNED_PHRASES))
def test_no_skill_string_the_model_can_read_carries_a_banned_phrase(phrase):
    """Wider than the surfaces above: a tool RESULT is read by the model
    too, and a `hint` field is copy the user hears almost verbatim."""
    from app.agent.skills.builtins.automations import skill as auto_mod
    from app.agent.skills.builtins.routines import skill as routines_mod
    from app.agent.skills.builtins.triggers import skill as triggers_mod

    offenders = []
    for module in (auto_mod, routines_mod, triggers_mod):
        for lineno, text in _model_visible_literals(module):
            if _teaches(text, phrase):
                offenders.append(
                    f"{module.__name__}:{lineno}: {text.strip()[:120]}")
    assert not offenders, (
        f"a model-readable string carries the banned phrase {phrase!r} "
        f"(docstrings and comments are exempt — these are not):\n"
        + "\n".join(offenders)
    )


# ------------------------------------------------------------- §14

# Round 33, item 2: the first alternative is the original — a posture that
# fires on ORDINARY conversation ("before answering anything"). The two
# added ones are the UNCONDITIONAL shape, which "ALWAYS call this before
# proposing an automation" had in `automations__get_registry` while the
# guard stayed green.
#
# Deliberately NOT widened to `before (creating|building|starting)`:
# `routines__list` and `triggers__list` both say "check the list BEFORE
# creating one", which is a scoped, correct instruction — it cannot fire
# on a turn that is not already creating something. The rule is about
# imperatives with no scope, not about the word "before".
_PREEMPTIVE = re.compile(
    r"\b(use (it|this) )?before (answering|doing|replying|responding)\b"
    r"|\balways call (this|it)\b"
    r"|\bcall (this|it) first\b",
    re.I,
)


def test_no_tool_description_carries_a_flow_posture():
    """§14 rule 1 — tool descriptions are instructions the model competes
    over. "Call this before answering anything" in a description every
    automations turn sees can starve the setup conversation of the early
    iterations it needs; the posture belongs in the owning section."""
    offenders = [
        f"{name}: {_PREEMPTIVE.search(text).group(0)!r}"
        for name, text in [r for sk in ALL_SKILLS for r in _descriptions(sk)]
        if _PREEMPTIVE.search(text)
    ]
    assert not offenders, (
        "a tool description carries a do-this-first posture (§14 rule 1):\n"
        + "\n".join(offenders)
    )


def test_the_recall_posture_still_exists_where_it_belongs():
    """Removing the posture from the description must not delete it —
    §5.4 keeps recall-first in the sections that own those answers."""
    from app.agent.automations.interview import prompt_section

    thread = prompt_section({
        "automation_id": "a1", "name": "Morning work brief",
        "rule_text": "reads overnight mail", "status": "active", "facts": {},
    })
    assert re.search(r"memory", thread, re.I)
    section = AUTOMATIONS.get_system_prompt_section() or ""
    assert re.search(r"memory", section, re.I), (
        "the automations section lost the answer-from-memory rule"
    )


# ------------------------------------------------- the assembled surface

def _assembled_surface() -> str:
    """Everything a real turn puts in front of the model, through the
    REAL loader — not a hand-listed set of skills.

    This exists because the file-scoped guards above were written over
    two skills, passed, and an assembly probe then found the same banned
    phrase living in a third. A guard that enumerates its own scope can
    only ever be as complete as the enumeration; this one asks the
    loader what actually loaded.
    """
    import asyncio

    from app.config import settings
    from app.agent.skills.loader import SkillLoader

    previous = getattr(settings, "automations_enabled", False)
    settings.automations_enabled = True
    try:
        loader = SkillLoader()
        asyncio.run(loader.load_all())
        parts = []
        for skill in loader.skills.values():
            for tool in skill.get_tools():
                parts.append(tool.get("description") or "")
                schema = tool.get("input_schema") or {}
                for prop in (schema.get("properties") or {}).values():
                    if isinstance(prop, dict) and prop.get("description"):
                        parts.append(str(prop["description"]))
        sections = loader.get_all_system_prompt_sections()
        values = sections.values() if isinstance(sections, dict) else sections
        parts.extend(str(v) for v in values)
        return "\n".join(parts)
    finally:
        settings.automations_enabled = previous


@pytest.mark.parametrize("phrase", sorted(BANNED_PHRASES))
def test_the_whole_assembled_surface_teaches_no_banned_phrase(phrase):
    assert not _teaches(_assembled_surface(), phrase), (
        f"the assembled agent surface teaches {phrase!r} — find which "
        "skill and fix it there; every builtin counts, not just the ones "
        "this file names"
    )


def test_the_assembled_surface_answers_the_automations_question_once():
    """End to end: the ND-18 instruction is gone from every loaded skill,
    and the surface that owns the answer states it."""
    whole = _assembled_surface()
    assert "what automations do I have set up?" not in whole
    assert "recurring agent automations" not in whole
    assert "automations__list`, and only that" in whole


def test_the_assembled_surface_keeps_the_reminder_flow_intact():
    """The guard on the guard: this round's vocabulary fix must not have
    taken any behavioural instruction of the reminder path with it."""
    whole = _assembled_surface()
    for rule in ("routines__remind", "in_seconds", "duplicates"):
        assert rule in whole, f"the reminder flow lost {rule!r}"
    assert "never ask where" in whole.lower()


# ------------------------------------------- the tool has to be ON the turn

def test_an_automations_inventory_ask_can_reach_the_automations_list():
    """The structural half of ND-18, and the half no prompt can fix.

    "what automations do I have?" classifies `question`, which filters
    the tools array down to the intent's own names plus a small
    always-included set. `routines__list` was in that set and
    `automations__list` was not — so on exactly the turns a user asks
    the inventory question in the short way, the only list the model
    could reach was the wrong one. It answered from reminders because
    reminders were all it had.
    """
    from app.agent.query_intent import (
        classify_query_intent, filter_tools_by_intent, has_automation_intent,
    )

    # Round 33: the MECHANISM moved, the invariant did not. ND-18 put
    # `automations__list` in `_ALWAYS_INCLUDED_TOOLS`, which made an
    # inventory tool reachable on every short question in the product —
    # "King Charles vs reza pahlavi" called it before it searched anything.
    # It is classified now, off the noun the user actually says, so the
    # behavioural loop below is the whole test.
    assert has_automation_intent("what automations do i have?")
    assert not has_automation_intent("king charles vs reza pahlavi")

    tools = [{"name": n, "input_schema": {"type": "object"}}
             for n in ("automations__list", "routines__list",
                       "automations__create", "web_search")]
    for ask in ("what automations do I have?",
                "how many automations do I have?",
                "list my automations",
                "what are my automations"):
        intent = classify_query_intent(ask)
        assert intent.category == "question", (ask, intent.category)
        survivors = {t["name"] for t in filter_tools_by_intent(tools, intent)}
        assert "automations__list" in survivors, (
            f"{ask!r} cannot reach the automations inventory: {survivors}"
        )


def test_the_automations_list_claims_the_inventory_question():
    """The vocabulary half: the tool that owns the answer has to say so.
    The production sentence classifies `full` (every tool present), so
    that turn was lost at tool CHOICE — the description is what decides
    it."""
    listing = next(t for t in AUTOMATIONS.get_tools()
                   if t["name"] == "automations__list")
    text = listing["description"].lower()
    assert "what automations they have" in text
    assert "only" in text
    assert "reminders" in text


# ------------------------------- the set that grows one incident at a time

#: Zero-argument list tools that are NOT a user-facing inventory, with the
#: reason. A catalog of what COULD be built is not a list of what the user
#: HAS, and nobody asks "how many templates do I have".
_NOT_AN_INVENTORY = {
    "automations__list_templates": "a catalog of what could be built, "
                                   "not what this user has",
    "automations__get_registry": "connector capability metadata, not the "
                                 "user's own things",
    "triggers__list": "triggers are internal plumbing behind automations; "
                      "'trigger' is banned as a user-facing noun (§5.7), "
                      "so no user asks what triggers they have",
}


def test_every_user_inventory_tool_is_reachable_on_a_question_turn():
    """ND-18's structural cause, generalised.

    `_ALWAYS_INCLUDED_TOOLS` has grown at least three times, each time
    after a live failure: `memory_read_file`, `memory_store`,
    `routines__remind` (the 2026-07-16 typo repro), and now
    `automations__list`. A set that only grows after incidents is a set
    that is still incomplete — so assert the rule instead of the
    membership: if a tool answers "what do I have" for a noun the user
    actually says, a short ask must be able to reach it.

    A new inventory tool therefore fails here on the day it is added,
    not on the day a user asks about it in four words.
    """
    from app.agent.query_intent import (
        _ALWAYS_INCLUDED_TOOLS, classify_query_intent, filter_tools_by_intent,
    )

    # Round 33: reachability is the rule, membership was only ever one way
    # of getting it. A tool may be always-included OR classified off its own
    # noun — the automations family is the latter now, because an inventory
    # tool that is always on the turn gets CALLED on turns that are not
    # about it. So the assertion is the one that matters: ask for the thing
    # in four words, and the tool that answers must be reachable.
    def _reachable(name: str, noun: str) -> bool:
        if name in _ALWAYS_INCLUDED_TOOLS:
            return True
        defs = [{"name": name, "input_schema": {"type": "object"}}]
        for ask in (f"what {noun} do i have?", f"list my {noun}",
                    f"how many {noun} do i have?"):
            intent = classify_query_intent(ask)
            survivors = {t["name"] for t in filter_tools_by_intent(defs, intent)}
            if name not in survivors:
                return False
        return True

    missing = []
    for skill in ALL_SKILLS:
        for tool in skill.get_tools():
            name = tool["name"]
            schema = tool.get("input_schema") or {}
            if not name.endswith("__list") and "__list" not in name:
                continue
            if schema.get("required"):
                continue  # takes an argument: not a bare inventory ask
            description = (tool.get("description") or "").lower()
            if "user's" not in description and "your " not in description:
                continue  # not a list of the user's own things
            if name in _NOT_AN_INVENTORY:
                continue
            noun = name.split("__", 1)[0]
            if not _reachable(name, noun):
                missing.append(name)
    assert not missing, (
        "these tools answer 'what do I have' but a short question-intent "
        f"ask cannot reach them: {missing}. Add them to "
        "_ALWAYS_INCLUDED_TOOLS, or add them to _NOT_AN_INVENTORY here "
        "with the reason they are not a user-facing inventory."
    )


def test_the_inventory_exemptions_still_exist():
    """An exemption for a tool that no longer exists is dead weight that
    hides the next one."""
    names = {t["name"] for skill in ALL_SKILLS for t in skill.get_tools()}
    stale = sorted(set(_NOT_AN_INVENTORY) - names)
    assert not stale, f"exemptions for tools that no longer exist: {stale}"


# ------------------------------------------------ ND-19: never invent one

def test_the_boundary_names_the_tool_only_when_the_tool_exists():
    """ND-19's first half. Automations are flag-gated: with the skill
    unloaded, `automations__list` is not in the array and the
    automations section is absent — but the routines boundary still
    told the model to answer from that tool. Instructed to use
    something it did not have, the agent said "I can't verify your full
    automation list from here right now" and then invented one.
    """
    from app.config import settings

    previous = getattr(settings, "automations_enabled", False)
    try:
        settings.automations_enabled = False
        off = RoutinesSkill().get_system_prompt_section() or ""
        settings.automations_enabled = True
        on = RoutinesSkill().get_system_prompt_section() or ""
    finally:
        settings.automations_enabled = previous

    # The half that stops the ND-18 conflation ships in both states.
    for section in (off, on):
        assert "not the user's automations" in section

    # The half that points at a tool ships only with the tool.
    assert "automations__list" not in off, (
        "the routines prompt names a tool that is not on this tenant"
    )
    assert "automations__list" in on


@pytest.mark.parametrize("flag", [True, False])
def test_no_surface_ever_points_at_a_tool_the_turn_does_not_have(flag):
    """The general rule behind ND-19: a prompt may only name a tool that
    is actually in the array for the same configuration. Checked over
    the real loader, in both flag states."""
    import asyncio

    from app.config import settings
    from app.agent.skills.loader import SkillLoader

    previous = getattr(settings, "automations_enabled", False)
    settings.automations_enabled = flag
    try:
        loader = SkillLoader()
        asyncio.run(loader.load_all())
        available = {t["name"] for sk in loader.skills.values()
                     for t in sk.get_tools()}
        sections = loader.get_all_system_prompt_sections()
        values = sections.values() if isinstance(sections, dict) else sections
        prose = "\n".join(str(v) for v in values)
    finally:
        settings.automations_enabled = previous

    named = set(re.findall(r"`(automations__\w+|routines__\w+)`", prose))
    missing = sorted(n for n in named if n not in available)
    assert not missing, (
        f"with automations_enabled={flag} the prompt names tools that are "
        f"not in the array: {missing}"
    )


def test_the_agent_is_told_never_to_invent_an_automation():
    """ND-19's second half, and the worse one. With no list to read the
    agent named "Teams chat reader" — which does not exist — and gave it
    a plausible status. Nothing had ever told it not to."""
    section = AUTOMATIONS.get_system_prompt_section() or ""
    assert "Never name an automation you have not just read" in section
    assert "name none" in section
    # The rule has to cover the invented STATUS too, not just the name.
    assert "guessed status" in section


def test_the_boundary_follows_the_BOOT_gate_not_a_live_reload():
    """The lifetime trap under ND-19.

    `skill_enabled` is resolved ONCE AT BOOT by the loader, but
    `settings.automations_enabled` flips live on a settings reload. A
    prompt that reads the setting at render time would therefore start
    naming `automations__list` after a reload turned the flag on, in a
    process where the loader never registered the skill — a prompt
    pointing at a tool that does not exist, which is exactly the defect.
    So the section must follow the boot snapshot, not the live value.
    """
    import asyncio

    from app.config import settings

    previous = getattr(settings, "automations_enabled", False)
    try:
        # Booted WITH automations, flag later turned off by a reload:
        # the tools are still registered, so the prompt keeps naming them.
        settings.automations_enabled = True
        booted_on = RoutinesSkill()
        asyncio.run(booted_on.on_load())
        settings.automations_enabled = False
        assert "automations__list" in (booted_on.get_system_prompt_section() or "")

        # Booted WITHOUT, flag later turned on: the skill was never
        # registered this process, so the prompt must stay silent.
        settings.automations_enabled = False
        booted_off = RoutinesSkill()
        asyncio.run(booted_off.on_load())
        settings.automations_enabled = True
        section = booted_off.get_system_prompt_section() or ""
        assert "automations__list" not in section, (
            "a live reload made the prompt name a tool the loader never "
            "registered in this process"
        )
        assert "not the user's automations" in section
    finally:
        settings.automations_enabled = previous


# ------------------------------- the probe that measures the honesty path

def test_the_toolless_honesty_probe_can_actually_fail():
    """A probe that cannot fail is decoration, and this one nearly was.

    The eval scenario for ND-19's honest-refusal path started with two
    checks — "does the reply sound honest" and "does it claim a count".
    Run against the REAL production failure, both passed: *"I can't
    verify your full automation list from here right now. The two I know
    about are: Jira -> Slack ... Teams chat reader - currently paused"*
    carries an honest marker and never says "you have N", so the probe
    would have PASSED the exact reply that failed. The naming check is
    what closes it — with no list to read, any enumeration is invented.

    Pinned here against verbatim production strings so the classifier
    cannot drift back into a false pass.
    """
    import importlib.util
    from pathlib import Path

    suite_path = (Path(__file__).resolve().parents[2]
                  / "scripts" / "eval" / "automation_agent_suite.py")
    spec = importlib.util.spec_from_file_location("_eval_suite", suite_path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    def verdict(reply: str) -> bool:
        clean = suite._normalise(reply)
        return (bool(suite._HONEST.search(clean))
                and not suite._COUNT_CLAIM.search(clean)
                and not suite._enumerates(clean))

    # Verbatim from the live run that failed (ND-19).
    fabricated = (
        "I can’t verify your full automation list from here right now. "
        "The two I know about are:\n"
        "- **Jira → Slack**\n"
        "- **Teams chat reader** — currently paused because Teams needs "
        "re-authentication."
    )
    counted = "From what I have on hand, you have **2 automations**."
    # Verbatim from the run that behaved correctly, curly quotes and all.
    honest = (
        "I can’t access your automation list from here, so I can’t "
        "see how many you have or their names. Reminders and scheduled "
        "tasks are separate from automations."
    )

    assert not verdict(fabricated), "the probe passes a fabricated list"
    assert not verdict(counted), "the probe passes an invented count"
    assert verdict(honest), "the probe fails an honest refusal"

    # And the other direction, also from a real run: pointing the user at
    # the dashboard's **Automations** section is a helpful honest reply.
    # Emphasis is not enumeration; failing it would train the fix toward
    # terser, less useful answers.
    ui_reference = (
        "I can’t access your account’s automation list from this chat, so "
        "I can’t see how many you have or their names. Check your "
        "dashboard’s **Automations** section for the current list."
    )
    assert verdict(ui_reference), (
        "the probe fails an honest reply that merely names a UI section"
    )


def test_the_honesty_probe_survives_curly_punctuation():
    """The first matcher was a list of phrasings with straight
    apostrophes and reported FAIL on three perfect replies, because the
    model writes "can’t". A probe that emits a confident false FAIL
    wastes the same trust as one that emits a false PASS."""
    import importlib.util
    from pathlib import Path

    suite_path = (Path(__file__).resolve().parents[2]
                  / "scripts" / "eval" / "automation_agent_suite.py")
    spec = importlib.util.spec_from_file_location("_eval_suite2", suite_path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    for apostrophe in ("'", "’"):
        reply = f"I can{apostrophe}t access your automation list from here."
        assert suite._HONEST.search(suite._normalise(reply)), apostrophe


def test_the_snapshot_is_retaken_when_registration_reopens():
    """The lifetime can now EXTEND, and the prompt has to hear about it.

    `on_load` alone was correct only while registration was resolved once
    per process. `SkillLoader.refresh_entitlements` lets a container that
    booted dark register automations later — at which point a boot-time
    snapshot under-claims a tool the model is actually holding. That is
    the safe direction, and therefore the kind of stale nobody notices,
    so it gets a pin: silent while dark, naming the tool once the loader
    reports the entitlement set changed.
    """
    import asyncio

    from app.config import settings

    previous = getattr(settings, "automations_enabled", False)
    try:
        settings.automations_enabled = False
        skill = RoutinesSkill()
        asyncio.run(skill.on_load())
        assert "automations__list" not in (skill.get_system_prompt_section() or "")

        # The flag flipping is NOT the signal — registration is.
        settings.automations_enabled = True
        assert "automations__list" not in (skill.get_system_prompt_section() or ""), (
            "the prompt followed the raw setting instead of registration"
        )

        # The loader says the entitlement set changed: resample.
        asyncio.run(skill.on_entitlements_changed())
        assert "automations__list" in (skill.get_system_prompt_section() or ""), (
            "registration reopened and the prompt never noticed — it now "
            "under-claims a tool the model is holding"
        )
    finally:
        settings.automations_enabled = previous
