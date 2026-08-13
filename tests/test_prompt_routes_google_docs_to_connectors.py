""""Create me a Google Doc" must not produce a local .docx.

Found 2026-08-12 from a founder recording. The user typed "create me
google doc and put hi on top of it" on an account with docs, drive,
sheets, gmail and calendar all `active`, and got `hi.docx` — a Word
file attached to the chat, which is not a Google Doc, does not appear
in their Drive, and cannot be opened as one.

Nothing was broken. The tools were registered and the identities were
live. The model was simply never told the Google tools were an option,
while THREE separate prompt sections steered it the other way:

    'make this a PDF/doc/spreadsheet/deck' -> generate_pdf / generate_docx / ...
    '- `generate_docx` - editable Word document. Use when the user will revise it.'
    (and the same list again in the diet profile)

`grep -r "Google Doc" app/agent/` returned NOTHING. "google doc"
contains "doc", so the one loud rule won every time.

Two guards:

  * `test_generate_docx_guidance_always_names_the_google_alternative`
    is the general one — anywhere a prompt steers to `generate_*`, the
    Google discriminator has to be within reach. A fourth section added
    later without it fails here.
  * `test_prompts_only_name_connector_tools_that_exist` stops the
    opposite drift: a prompt naming a tool that no manifest defines.
    `sheets__list_spreadsheets` was deleted on 2026-08-11 for needing a
    scope we never request; a prompt still advertising it would send the
    model at a tool that cannot be called.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

AGENT = pathlib.Path("app/agent")
CONNECTORS = pathlib.Path("app/connectors")

# Files that build the model's system prompt and mention the local
# document generators. Both are checked; adding a third is the case the
# proximity test below is designed to catch.
PROMPT_FILES = ["agent_runner.py", "prompt_diet.py"]

# How far from a `generate_docx` mention the discriminator may sit.
#
# Measured 2026-08-12, not guessed. Every real rule sits within 434
# chars of the steer it qualifies (434 / 207 / 419 in agent_runner,
# 158 in prompt_diet), while the two CLOSEST distinct prompt sections
# are 3309 chars apart. 1000 clears every real distance with 2.3x
# headroom and still leaves one section unable to vouch for the next.
#
# The first draft used 2500 and two mutations survived it: deleting the
# rule from the routing list passed, because the ### Documents rule
# 3309 chars away was inside the window. A guard that one section can
# satisfy on another's behalf is not measuring what it claims to.
WINDOW = 1000

# Any ONE of these, near the steer, tells the model the Google tools
# exist at all.
GOOGLE_TOOLS = ("docs__create", "sheets__create_spreadsheet", "drive__create_doc")

# ...and any one of THESE says why it matters. Naming the alternative is
# not enough on its own: the competing rule ('make this a doc ->
# generate_docx') is imperative and specific, so a neutral "you could
# also use docs__create" loses to it. What breaks the tie is stating
# that the generated file is a different object.
#
# A mutation that stripped exactly this sentence, leaving the tool names
# in place, survived the first version of this guard — which checked the
# phrase once at file level and so could not notice one of two
# occurrences going missing.
CONSEQUENCE = re.compile(
    r"not a google doc"
    r"|never reaches their drive"
    r"|cannot be opened as one"
    r"|lands in this chat"
    r"|their own google account",
    re.I,
)


def _text(name: str) -> str:
    return (AGENT / name).read_text()


def _manifest_tool_names() -> set[str]:
    names: set[str] = set()
    for d in CONNECTORS.iterdir():
        mf = d / "manifest.yaml"
        if not mf.exists():
            continue
        m = yaml.safe_load(mf.read_text()) or {}
        for t in (m.get("tools") or []):
            if t.get("name"):
                names.add(t["name"])
    return names


@pytest.mark.parametrize("fname", PROMPT_FILES)
def test_generate_docx_guidance_always_names_the_google_alternative(fname):
    """Every steer toward the local generators sits near the Google rule.

    Proximity, not mere presence: a file can mention `docs__create` once
    in an unrelated section while a second `generate_docx` list five
    thousand lines away still teaches the wrong thing.
    """
    src = _text(fname)
    hits = [m.start() for m in re.finditer(r"generate_docx", src)]
    assert hits, f"{fname} no longer mentions generate_docx — update this guard"

    no_tool, no_why = [], []
    for pos in hits:
        window = src[max(0, pos - WINDOW):pos + WINDOW]
        line = src.count("\n", 0, pos) + 1
        if not any(tool in window for tool in GOOGLE_TOOLS):
            no_tool.append(line)
        elif not CONSEQUENCE.search(window):
            no_why.append(line)

    assert not no_tool, (
        f"{fname} steers to generate_docx at line(s) {no_tool} with no Google "
        f"connector alternative within {WINDOW} chars. A user asking for 'a "
        f"Google Doc' will get a .docx attachment. Name one of {GOOGLE_TOOLS} "
        f"in that section."
    )
    assert not no_why, (
        f"{fname} names the Google tools near line(s) {no_why} but never says "
        f"a generated file is a DIFFERENT object. The competing rule is "
        f"imperative; a neutral mention loses to it. State the consequence — "
        f"that a .docx is not a Google Doc and never reaches their Drive."
    )


@pytest.mark.parametrize("fname", PROMPT_FILES)
def test_prompts_only_name_connector_tools_that_exist(fname):
    """A prompt may not advertise a connector tool no manifest defines."""
    src = _text(fname)
    referenced = set(re.findall(r"\b([a-z_]+__[a-z_]+)\b", src))
    if not referenced:
        pytest.skip(f"{fname} names no connector tools")

    # Only judge prefixes that are real connectors — `routines__remind`
    # and friends use the same double-underscore shape but are skills,
    # registered elsewhere and out of scope here.
    connector_ids = {
        d.name for d in CONNECTORS.iterdir() if (d / "manifest.yaml").exists()
    }
    known = _manifest_tool_names()
    ghosts = sorted(
        t for t in referenced
        if t.split("__", 1)[0] in connector_ids and t not in known
    )
    assert not ghosts, (
        f"{fname} tells the model to call {ghosts}, which no connector "
        f"manifest defines. Either the tool was removed and the prompt "
        f"wasn't, or the name is a typo — both send the model at a tool "
        f"that cannot run."
    )

