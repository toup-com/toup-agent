"""W2.1a prefix diet — compact prompt sections served behind ``settings.prompt_diet``.

Audit context: docs/audits/2026-07-sota-assessment.md measured a 27.2k-token
wire prefix; ~5,750 of it is restatement (the app_builder essay, three fat
tool-schema essays, platform_knowledge blurbs that re-describe tool schemas,
and the doc_generation section triple-carrying guidance the generate_*
schemas already carry). This module holds the compact replacements plus the
tiny helpers the touched call-sites share.

Contract (regression-pinned in tests/test_prompt_diet.py):
  * flag OFF (default) → every touched section and tool schema is
    byte-identical to the pre-diet output;
  * flag ON → tool ARG SHAPES (properties/enums/required) never change —
    only description strings shrink.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def prompt_diet_enabled() -> bool:
    """True when the PROMPT_DIET env flag is set (default off)."""
    from app.config import settings
    return bool(getattr(settings, "prompt_diet", False))


def apply_tool_description_diet(
    tools: List[Dict[str, Any]],
    tool_descriptions: Dict[str, str],
    property_descriptions: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Swap description strings on freshly-built tool defs, in place.

    Only ``description`` values are touched — never property names, types,
    enums, or ``required`` lists, so the wire arg shapes stay identical to
    the full schemas. Unknown tool/property names are ignored (a rename in
    the full schema can never make the diet path KeyError).
    """
    props_by_tool = property_descriptions or {}
    for tool in tools:
        name = tool.get("name", "")
        if name in tool_descriptions:
            tool["description"] = tool_descriptions[name]
        for prop, desc in props_by_tool.get(name, {}).items():
            prop_schema = (
                tool.get("input_schema", {}).get("properties", {}).get(prop)
            )
            if isinstance(prop_schema, dict):
                prop_schema["description"] = desc
    return tools


# ── (c) platform_knowledge diet ─────────────────────────────────────────
# Keeps (verbatim from the legacy section): the intro, the "How tools work"
# anti-fake-tool-call rules, the full Pages map, the Day-Chat/Voice/Channels
# platform behaviors, and the NEVER list. Drops the seven capability blurbs
# that restate tool schemas (Memory/Apps/Browser/Documents/Jobs/Media/
# Past-day-recall) and compresses Decision rules to one-line hints. The
# owner fact + injection-fencing blocks are appended by the caller in both
# paths, unchanged.
# The two rules that MUST differ on voice. Voice has no create_job/update_job/
# spawn (prompt_profile.VOICE_DISABLED_TOOLS), and naming a tool the model
# cannot call is worse than naming none — it answers "I can't do that from
# here" instead of doing the thing it can. The legacy literal in agent_runner
# is already voice-aware; this is the same fix for the diet path, which
# REPLACES that literal wholesale when PROMPT_DIET is on.
def _DIET_SEARCH_RULE(voice: bool) -> str:
    if voice:
        return ("- web search / find / look up / real-time web → `web_search`, then "
                "`web_fetch` on the best hits. `browser` only for pages you must "
                "OPERATE (sign in, fill a form) — it costs tens of seconds a step\n")
    return ("- web search / find / book / real-time web → `browser` (suggest "
            "`[[navigate:/browser]]` so they can watch)\n")


def _DIET_JOB_RULE(voice: bool) -> str:
    if voice:
        return ("- the user is SPEAKING and waiting: do the work in this turn and say "
                "the answer. Never promise a report or anything arriving later. Only "
                "`start_mission` defers, and only when they ask for work that outlives "
                "the call\n")
    return ("- multi-step work you'll finish THIS turn → `create_job` IN THE SAME "
            "response as the first step's tool calls (never alone), `update_job` in "
            "the same response as the next step's tools; never mark completed — the "
            "system does when you reply; work that continues after the conversation → "
            "`start_mission` (not create_job)\n")


def platform_knowledge_diet(voice: bool = False) -> str:
    """The compact platform map. `voice` swaps the two rules above."""
    return (
        "# Platform Knowledge — How Toup Works\n"
        "You run on **Toup** — a personal-agent platform. The user lives "
        "here. You don't need to ask what features exist; you know. Below "
        "is the complete map. Use it: when the user expresses a goal, "
        "translate it into the right action without making them learn the "
        "product.\n\n"
        "## How tools work — read this carefully\n"
        "Throughout this section you'll see references to tools by name. "
        "When you need to use a tool, **invoke it through your function-"
        "calling API** — don't write the tool name as text in your "
        "message.\n\n"
        "WRONG (do NOT produce output like this):\n"
        "- `<navigate_to path=\"/brain\" />`\n"
        "- `navigate_to('/brain')` as a literal text line\n"
        "- `{tool: \"navigate_to\", args: {path: \"/brain\"}}`\n"
        "- Any XML/JSX/JSON-shaped tool-call syntax in the message body.\n\n"
        "RIGHT:\n"
        "- Call the actual tool. The user sees no tool syntax — only your "
        "natural-language reply (\"Done — there you go.\" / \"Pulled it up.\").\n"
        "- For chips that the user clicks, write `[[navigate:/brain]]` "
        "directly in your message text. Chips ARE part of your message; "
        "tool calls are NOT.\n\n"
        "If a tool you want to use isn't available on this turn (you'd see "
        "it in your tool list), explain in plain words instead and offer a "
        "[[chip]] for the user to take the action themselves. Never "
        "fake a tool call as text.\n\n"
        "## Pages — where things live\n"
        "Take the user anywhere via the navigate_to tool (explicit "
        "request) or a `[[navigate:/path]]` chip (suggestion).\n\n"
        "- `/` — **Hub**. Home. Entry cards.\n"
        "- `/chat` — **Chat**. This conversation. Day-grouped at "
        "`/chat/<date>` (e.g. `/chat/2026-05-08`).\n"
        "- `/brain` — **Brain**. Memories you've stored about the user, "
        "organized by category (identity, work, goals, preferences, "
        "knowledge, projects). Same data as the `# User Brain` section "
        "above — they see what you see.\n"
        "- `/browser` — **Live Browser**. The user watches your headless "
        "browser in real time. Use the `browser` tool while they're here "
        "and they see every click.\n"
        "- `/workspace` — **Workspace**. Apps you've built for the user.\n"
        "- `/workspace/apps/<slug>` — preview of a specific app. Reach "
        "via the `[[open_app:<slug>]]` chip after you build/restart one.\n"
        "- `/jobs` — **Jobs**. Long-running tasks with status and logs.\n"
        "- `/dashboard` — **Dashboard**. Metrics, inbox, daily summary.\n"
        "- `/agent` — **Agent home**. Soul, channels, LLM keys.\n"
        "- `/agent/soul` — **Soul**. Your personality config — name, "
        "style (casual/professional/mentor/creative), traits (humor, "
        "direct, proactive, references_past, etc.), pronouns, custom "
        "instructions.\n"
        "- `/agent/settings` — **Channels & Settings**. WhatsApp "
        "(BYOA paste OR QR-link mode), Telegram, voice wiring.\n"
        "- `/agent/tools` — **Tools catalog** with descriptions of every "
        "tool you have.\n"
        "- `/agent/skills` — **Skills catalog**. Domain-specific skill "
        "packs (some installed, some marketplace).\n"
        "- `/account` — **Account**. Profile, password, billing.\n"
        "- `/movies` — **Movies**. Netflix integration when relevant.\n\n"
        "## Platform behaviors\n\n"
        "### Day-Chat — one continuous thread per day\n"
        "The user can talk to you on web, mobile app, WhatsApp, "
        "Telegram, or voice. **All of those share the same thread for a "
        "given day.** Replying on WhatsApp shows on web. Don't "
        "reintroduce yourself when channels switch — it's the same "
        "conversation. Past days are recoverable via `recall_day`.\n\n"
        "### Voice\n"
        "The user can hit the voice button for real-time spoken "
        "conversation. All your tools work in "
        "voice including `navigate_to`. Voice picks up the same memory, "
        "soul, and identity.\n\n"
        "### Channels (WhatsApp / Telegram / Mobile)\n"
        "If the user wants to text you on WhatsApp, point them to "
        "`/agent/settings` — they paste a token (BYOA mode) or scan a "
        "QR code (link mode, ~30s setup). Telegram is similar. Once "
        "wired, you receive their texts in the same Day-Chat thread.\n\n"
        "## Decision rules — quick hints\n"
        "Your tool schemas carry the detail; these map intent → tool.\n"
        "- 'remember <fact>' → `memory_store`; 'what do you know about X' → "
        "open the file the `# User Brain` index names with `memory_read_file`, "
        "else `memory_search` (Profile, Current context and Learned are "
        "already in `# User Brain`)\n"
        "- 'make me a <tool/app>' → app_builder skill; afterwards offer "
        "`[[open_app:<slug>]]`\n"
        + _DIET_SEARCH_RULE(voice) +
        "- 'remind me' → `routines__remind`; recurring agent task / briefing → "
        "`routines__create`\n"
        "- 'make this a PDF/doc/spreadsheet/deck' → the `generate_*` tools; "
        "'play <song/movie/show>' → `play_media` (Netflix titles get a "
        "`[[Play TITLE on Netflix]]` chip)\n"
        "- past days → `recall_day` (any natural date, all channels — NEVER "
        "tell the user you can't remember a past day)\n"
        + _DIET_JOB_RULE(voice) +
        "- navigation asks → `navigate_to`: settings/channels "
        "`/agent/settings`, personality `/agent/soul`, tools `/agent/tools`, "
        "memories `/brain`, account/billing `/account`, metrics `/dashboard`\n\n"
        "## What you should NEVER make the user do\n"
        "- Hunt through menus to find a feature you can navigate them to. Just take them.\n"
        "- Repeat themselves between channels — it's all one thread.\n"
        "- Manually copy data between Brain / Apps / Chat — you can do it.\n"
        "- Ask 'would you like me to do X?' when they explicitly asked you to do X. Just do it.\n"
        "- Answer 'where should I send it?' for reminders/routines — delivery is automatic to chat + every connected channel.\n\n"
        "The whole point of you is: the user says what they want, you make it happen."
    )


# Back-compat: the web variant, byte-identical to the pinned literal.
PLATFORM_KNOWLEDGE_DIET = platform_knowledge_diet(False)


# ── (d) doc_generation diet ─────────────────────────────────────────────
# Keeps the tool-choice rules + the convert-vs-regenerate rule; drops the
# worked examples and pane prose the generate_* schemas already carry.
DOC_GENERATION_DIET = (
    "# Document Generation\n"
    "Use the `generate_*` tools when the user wants a file to **keep, "
    "share, or edit** — never for conversational answers (those stay "
    "inline markdown in chat).\n\n"
    "Google Doc/Sheet/Drive → `docs__create` / `sheets__create_spreadsheet` "
    "/ `drive__create_doc`: a generated file lands in this chat and is not "
    "a Google Doc.\n\n"
    "Pick the right tool:\n"
    "- `generate_pdf` — print-ready reports, summaries, invoices.\n"
    "- `generate_docx` — editable Word document the user will revise.\n"
    "- `generate_xlsx` — tabular data, multi-sheet workbooks.\n"
    "- `generate_pptx` — slide decks.\n"
    "- `generate_markdown` — plain text for import elsewhere.\n"
    "- `generate_data_file` — CSV, JSON, txt, code.\n"
    "- `generate_audio` — text read aloud.\n"
    "- `convert_document` — convert an EXISTING generated DOCX/PPTX to "
    "PDF (layout-preserving). Use it when the user asks to \"make it a "
    "PDF\" on a file you just generated — do NOT call generate_pdf for "
    "that; it rebuilds from scratch and loses the original layout.\n\n"
    "Use descriptive filenames (e.g., `march-expenses.xlsx`). After the "
    "call, confirm in one sentence — the file is attached to your reply; "
    "don't repeat its contents back in markdown."
)


# ── (i) skill prose diet ────────────────────────────────────────────────
# R48 patch I. A SEPARATE flag from `prompt_diet` above, default OFF, and a
# separate module section because it compacts a different thing: not a
# literal this repo wrote into agent_runner, but the system-prompt sections
# the loaded SKILLS render.
#
# Why it is worth a seam (LOCAL, o200k, this tree, agent lane): the joined
# skill sections are 15,092 tokens — app_html 11,603 (9,874 of it the
# packaged DESIGN_SKILL.md, 1,729 hand-written prose), automations 2,340,
# routines 1,149, triggers 0. The stable layout includes them on EVERY turn
# of EVERY intent (`agent_runner.py`: `intent.include_skill_prompts or
# _stable`), so this is the largest always-on prose block the agent sends.
#
# What this section does NOT do, deliberately: it cuts no rule. The one entry
# (`app_html`, below) removes only text the retained text restates and
# whitespace that carries nothing — LOCAL, o200k: 11,603 → 11,156 tokens. Which
# sentences of a design document that drives a publish gate may be SHORTENED
# (memo D4b: B1/B2″) is a product judgment and is not here: app_html §8 and
# §11 are EXECUTED by the publish gate and pass through byte-identical.

def skill_prose_diet_enabled(
    user_id: str | None = None, channel: str | None = None,
) -> bool:
    """True for the global flag or an exact-user mobile canary.

    Separate from `prompt_diet_enabled()` on purpose — `prompt_diet` ships
    ON with text that was reviewed alongside it, and this flag would carry
    whatever a future mapping holds.
    """
    from app.config import settings, _exact_user_canary_enabled
    return bool(getattr(settings, "skill_prose_diet", False)) or (
        channel == "mobile" and _exact_user_canary_enabled(
            user_id, getattr(settings, "skill_prose_diet_canary_user_ids", "")
        )
    )


# ── the one entry: app_html, REDUNDANCY ONLY (R48 patch I, round 2) ──────
# Everything below removes text whose content the RETAINED text still states,
# or whitespace/separators that carry no content. It is NOT the memo's B1/B2″
# (those cut the only statement of some rules and are owner decisions); the
# classification table and the R2 list live in patches/i-skill-prose-diet.NOTES.md.
#
# Four walls, each pinned by its own tests in test_prompt_diet.py:
#   * §8 (motion/sound/state machine) and §11 (before `present_app`) are
#     passed through BYTE-IDENTICAL — the publish gate executes them (the
#     §8/§11 byte test).
#   * every fenced code block and every table row is passed through verbatim
#     (a block can be the only source of an exact value the model reproduces;
#     the atom gate, which also covers headings, modal sentences, backticked
#     and bold spans, names, URLs, paths and numbers).
#   * any anchor that is not found exactly once raises, and
#     `skill_section_diet` then serves the FULL body (the drift tests).
#   * every cut names its RETAINED TWINS (`_APP_HTML_TWINS`): the sentences,
#     elsewhere in the section, that still state what the cut removed. Each
#     twin must be present, whitespace-normalised, in its own region of the
#     DIETED output, or the diet raises and the full body is served. The
#     anchors alone only cover drift of the text being CUT; a cut deletes one
#     copy of a rule and relies on another copy, often in a different file
#     (`skill.py`'s head vs the packaged DESIGN_SKILL.md), and editing THAT
#     copy away must not leave the rule stated nowhere (the twin tests).
#   And the cut set itself is pinned by an independent copy of what review
#   round 2 read (`test_the_shipped_cuts_are_the_reviewed_ones`,
#   `test_the_diet_changes_exactly_the_reviewed_words`), so widening this
#   entry is red even where the atom gate is blind.
#   What is and is not proven: anchor drift and twin drift for the listed
#   twins fall back to the full body; the atom gate is a NECESSARY condition
#   over its atom classes (spans, names, numbers, modal sentences, fences,
#   headings), not a proof that no rule was lost. A twin is a substring
#   test, so an edit that NEGATES one by adding a prefix would still pass.
#   The reflow would also join a GFM table row without a leading `|`, a
#   setext underline or a `1)` list; none occurs today (review round 2, N7).

_APP_HTML_DOC_MARK = "\n# Toup frontend design\n"

#: (old, new) exact replacements in the HAND-WRITTEN head of the section.
#: Cut: the head's one-sentence summary of §7. Retained, verbatim, in §7:
#: "The app runs in a sandboxed frame with an **opaque origin**", "The runner
#: replaces all three before your code runs, with objects that cannot throw",
#: "Seed the UI from defaults immediately and reconcile when the data lands",
#: "Still genuinely unavailable in the sandbox: network requests (…), top-level
#: navigation, popups, and the parent page."
_APP_HTML_HEAD_CUTS = (
    (
        " It runs in a sandboxed frame on an opaque origin: there is no network, "
        "no navigation and no parent page. Storage cannot throw (the runner "
        "replaces it), but it is not durable within a first paint either, so "
        "seed the UI from in-memory defaults and reconcile after.",
        "",
    ),
)

#: (old, new) exact replacements in the design document's §10. Cut: the
#: PLAY-button anecdote and its lead-in, and the half of item 3 that the head's
#: "## Changing an app that is already open" states verbatim ("CHANGE THEM
#: ALL: every control of that kind, in the same way, in one round of edits",
#: "Widening the change is nearly free; guessing wrong costs a whole turn",
#: "A person who says a control is too small and gets a bigger menu button has
#: been answered with the wrong object", "change the thing the words are
#: actually about, not the first match for them"). Item 3's "Do not ask them
#: which one they meant." is kept word for word.
_APP_HTML_SEC10_CUTS = (
    (
        "A change request is about the thing the person was *using*, and they will\n"
        "describe it with the shortest word that fits. \"Make the button bigger\", said\n"
        "about a game, means the buttons they were pressing to play it.\n"
        "\n"
        "This went wrong exactly that way: asked to make the button bigger on a Snake\n"
        "with a D-pad, the edit landed on the start screen's `PLAY` button — pressed\n"
        "once, already large enough, and the element in the file that most literally\n"
        "answers to the word \"button\". The D-pad, pressed hundreds of times and\n"
        "genuinely too small, was untouched. The app came back with the same defect and\n"
        "a message saying it had been fixed.\n"
        "\n"
        "So, before you edit:",
        "Before you edit:",
    ),
    (
        "3. **If more than one answer is reasonable, change them all** — every control\n"
        "   of that kind, the same way, in one round of edits. Widening the change is\n"
        "   nearly free; guessing wrong costs the person another turn. Do not ask them\n"
        "   which one they meant.",
        "3. **If more than one answer is reasonable, change them all**. Do not ask them\n"
        "   which one they meant.",
    ),
)

#: (region, twin) — the retained statements every cut above relies on. The
#: region is where the twin must sit in the DIETED output: "head" (the
#: hand-written prose before the design document), "sec7" (§7 up to §8) or
#: "sec10" (§10 up to §11). Compared whitespace-normalised.
_APP_HTML_TWINS = (
    # the head cut (sandbox/storage) is carried by §7
    ("sec7", "The app runs in a sandboxed frame with an **opaque origin**"),
    ("sec7", "The runner replaces all three before your code runs, with objects that cannot throw"),
    ("sec7", "a read taken during first paint returns `null` even when a value exists."),
    ("sec7", "Seed the UI from defaults immediately and reconcile when the data lands"),
    ("sec7", "Still genuinely unavailable in the sandbox: network requests"),
    ("sec7", "top-level navigation, popups, and the parent page."),
    # §10's lead-in + anecdote are carried by the head and by §10 item 2
    ("head", "\"Make the button bigger\" is about the control the person was USING when they said it."),
    ("head", "not the PLAY button on the start screen, which they pressed once."),
    ("head", "A person who says a control is too small and gets a bigger menu button has been answered with the wrong object, and has to ask again."),
    ("head", "change the thing the words are actually about, not the first match for them."),
    ("head", "do not narrate the problem and do not claim the change."),
    ("sec10", "In a game the controls are the D-pad / paddle / fire button; `PLAY`, `RESTART` and menu items are chrome."),
    # §10 item 3's second half is carried by the head
    ("head", "CHANGE THEM ALL: every control of that kind, in the same way, in one round of edits."),
    ("head", "Widening the change is nearly free; guessing wrong costs a whole turn."),
)

#: Lines that START a block and so never continue the line above them.
_BLOCK_STARTS = ("#", "|", "```", "- ", "* ", "+ ", "> ", "---")


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("skill prose diet anchor not found exactly once")
    return text.replace(old, new)


def _reflow_markdown(md: str) -> str:
    """Join soft-wrapped lines of a paragraph or list item into one line.

    Markdown renders a single newline inside a paragraph as a space, so this
    is whitespace only: no word, span or number moves. Fenced blocks and table
    rows are copied verbatim; a line that starts a block (heading, list item,
    numbered item, table row, fence, rule, quote) is never joined upward and a
    heading/table row/rule never absorbs the line below it.
    """
    import re

    out: List[str] = []
    in_fence = False
    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or not out:
            out.append(line)
            continue
        if s.startswith("~~~") or (
            line.startswith("    ") and s and not out[-1].strip()
        ):
            # A `~~~` fence or a four-space indented code block: neither exists
            # in the document these rules were written against, and joining
            # either would rewrite code. Serve the section whole.
            raise ValueError("unrecognised code block")
        ps = out[-1].strip()
        joins = (
            bool(s) and bool(ps)
            and not ps.startswith(("#", "|", "```")) and ps != "---"
            and not s.startswith(_BLOCK_STARTS)
            and not re.match(r"\d+\.\s", s)
        )
        if joins:
            out[-1] = out[-1].rstrip() + " " + s
        else:
            out.append(line)
    if in_fence:
        # An unbalanced fence means the document is not what these rules were
        # written against. Serve it whole.
        raise ValueError("unbalanced code fence")
    return "\n".join(out)


def _drop_rules(text: str) -> str:
    """Remove a bare `---` line that sits between two blank lines, outside
    any fenced block (a `---` inside a fence is code, not a separator)."""
    lines = text.split("\n")
    out: List[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_fence = not in_fence
        elif (
            not in_fence and line == "---"
            and len(out) >= 2 and out[-1] == ""
            and i + 2 < len(lines) and lines[i + 1] == ""
        ):
            i += 2  # the rule and the blank line after it
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _diet_unprotected(text: str) -> str:
    return _reflow_markdown(_drop_rules(text))


def _check_twins(head: str, before8: str, sec9_10: str) -> None:
    import re

    def _n(x: str) -> str:
        return " ".join(x.split())

    def _from(chunk: str, heading: str) -> str:
        if chunk.count(heading) != 1:
            raise ValueError("twin region heading not found exactly once")
        return chunk[chunk.index(heading):]

    regions = {
        "head": _n(head),
        "sec7": _n(_from(before8, "\n## 7. ")),
        "sec10": _n(_from(sec9_10, "\n## 10. ")),
    }
    for region, twin in _APP_HTML_TWINS:
        if _n(twin) not in regions[region]:
            raise ValueError("a retained twin is missing: " + re.sub(r"\s+", " ", twin)[:60])


def _diet_app_html(body: str) -> str:
    if body.count(_APP_HTML_DOC_MARK) != 1:
        raise ValueError("design document marker not found exactly once")
    k = body.index(_APP_HTML_DOC_MARK)
    head, doc = body[:k], body[k:]
    for old, new in _APP_HTML_HEAD_CUTS:
        head = _replace_once(head, old, new)

    # §8 runs from its heading to §9's; §11 from its heading to the end.
    anchors = ("\n## 8. ", "\n## 9. ", "\n## 11. ")
    if any(doc.count(a) != 1 for a in anchors):
        raise ValueError("design document sections moved")
    i8, i9, i11 = (doc.index(a) for a in anchors)
    if not i8 < i9 < i11:
        raise ValueError("design document sections reordered")
    before8, sec8, sec9_10, sec11 = doc[:i8], doc[i8:i9], doc[i9:i11], doc[i11:]

    for old, new in _APP_HTML_SEC10_CUTS:
        sec9_10 = _replace_once(sec9_10, old, new)

    # A rule that closes an unprotected chunk sits right before a protected
    # heading ("…\n\n---\n" + "\n## 8. …"); it goes too, the heading stays.
    def _strip_trailing_rule(chunk: str) -> str:
        return chunk[: -len("\n---\n")] if chunk.endswith("\n\n---\n") else chunk

    before8 = _strip_trailing_rule(_diet_unprotected(before8))
    sec9_10 = _strip_trailing_rule(_diet_unprotected(sec9_10))
    # Checked on the OUTPUT, after every cut: a cut is only redundancy while
    # the statement it relies on is still in what the model is sent.
    _check_twins(head, before8, sec9_10)
    return head + before8 + sec8 + sec9_10 + sec11


#: skill name (``skill.meta.name``) → a compact replacement for that skill's
#: rendered system-prompt section. Each value takes the full body and returns
#: the compact one. Only entries that remove REDUNDANCY belong here; a cut
#: that changes what the model is told is an owner decision (NOTES, R2).
#: `automations` and `routines` have no entry: no cut in them survives the
#: atom gate (NOTES, round 2).
_SKILL_SECTION_DIETS: Dict[str, Callable[[str], str]] = {
    "app_html": _diet_app_html,
}


def skill_section_diet(name: str, body: str) -> str:
    """The compact section for ``name``, or ``body`` unchanged.

    Unknown names are returned untouched — the same discipline
    ``apply_tool_description_diet`` follows, and for the same reason: a
    skill added, renamed or retired in the loader can never make this path
    raise, and can never silently DROP a section it does not recognise. A
    replacement that raises is also swallowed back to the full body: a
    broken compaction must cost tokens, never capability.

    This function does not read the flag. The caller decides whether the
    diet applies, exactly as the two ``prompt_diet`` call sites in
    ``agent_runner`` do.
    """
    fn = _SKILL_SECTION_DIETS.get(name)
    if fn is None:
        return body
    try:
        compact = fn(body)
    except Exception:  # noqa: BLE001 — see the docstring: fall back to full
        return body
    return compact if isinstance(compact, str) and compact else body


def skill_sections_diet(
    skill_loader: Any, sections: List[str],
    user_id: str | None = None, channel: str | None = None,
) -> List[str]:
    """Route each already-rendered section through ``skill_section_diet``.

    ``SkillLoader.get_all_system_prompt_sections()`` returns bodies with no
    names attached, and ``skills/loader.py`` is deliberately out of scope for
    this patch (it is shared with the Voice programme and with patch H's
    neighbourhood). So the names are recovered HERE, by re-rendering each
    loaded skill's section and matching it to the body the loader produced.

    Two consequences worth stating rather than discovering:

    * ``get_system_prompt_section()`` is called a second time per skill —
      ONLY when the flag is on. The renders are pure and cheap (app_html's
      reads ``self._design_guidance``, fixed in ``__init__``), but it is a
      second call and a future non-deterministic renderer would simply fail
      to match.
    * A body that matches no skill is passed through UNCHANGED. A section is
      never dropped, never reordered, and never attributed to the wrong
      skill; the list that comes back is the same length, in the same order.

    Anything that goes wrong — a loader without ``.skills``, a renderer that
    raises — leaves ``sections`` exactly as it arrived.
    """
    if not skill_prose_diet_enabled(user_id, channel):
        # The byte-identity guarantee, made EXECUTABLE. The call site gates
        # too, and that is the gate that earns its keep — it skips the
        # re-render below entirely. This one exists because a public
        # "apply the diet" helper that ignores its own flag is a footgun the
        # next caller will step on, and because a guarantee only a source-pin
        # can check is a guarantee no test stands on. Deleting EITHER is a
        # defect: without this one the guarantee becomes untestable, without
        # the caller's it stops being free.
        return list(sections)

    try:
        by_body: Dict[str, str] = {}
        for skill_name, skill in getattr(skill_loader, "skills", {}).items():
            rendered = skill.get_system_prompt_section()
            if rendered:
                # First writer wins: if two skills somehow render identical
                # bodies, attributing both to the first is still a no-op for
                # an unknown name and stays deterministic.
                by_body.setdefault(rendered, skill_name)
    except Exception:  # noqa: BLE001 — a diet must never break the prompt
        return sections

    return [
        skill_section_diet(by_body.get(body, ""), body)
        for body in sections
    ]
