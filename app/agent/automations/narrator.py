"""The run narrator (R30 §4.2, §5.1–§5.2) — the run must read like the canvas.

Until this round an automation run produced no prose at all: the ledger
was template output. v3 makes the run a readable record — an opening
line, a reason for every item (ignored noise included), a ranked result
where everything appears exactly once, WHY wells, a draft when a reply
is owed, a closing line. This module is that pass.

The ENGINE owns the facts: it executes steps, mints item ids, records
writes, then hands this module its dispatch record. The narrator owns
the judgement and the voice: it returns TurnDrafts (the
`automation_emit_turns` payload) that the engine validates again,
persists with ids/seq, and serves. C never authors an `action` string —
tool-turn facts come from the verb dictionary on the engine side.

Dispatch record contract (CONTRACTS-R30 §5.1):

    record = {
      "automation": {"title": str, "mode": str},
      "run_kind":   "scheduled"|"run_now"|"setup"|"question",
      "vocabulary": "brief"|"changes",
      "status":     "completed"|"partial"|"failed",
      "rules":      [str],                      # "Rules you added", verbatim
      "memory_facts": [{"category": str, "text": str}],   # via memory.recall
      "steps": [
        {"step_ref": str, "connector_name": str, "account_id": str,
         "tool_kind": "read"|"write", "action": str, "detail": str,
         # R31-07: `failure_reason` is the string table's
         # `thread_sentence` for this account's reason code
         # (fixtures/automations/reason-strings.json, §4.4) — the
         # SENTENCE, already written, not a code and not a provider
         # error. The narrator quotes it; it does not compose one.
         "ok": bool, "failure_reason": str|None,
         "items": [{"id": str, "title": str, "sub": str,
                    "msgs": [{"who": str, "at": str, "text": str}]}],
         "write": {"what","target","audience","reversible"}|None,
      }],
    }

Everything free-text passes the copy guard; a violation, an uncovered
item, or a broken enum is a named problem — the narrator retries once
with the problems quoted, and the engine's completeness net (§4.2)
remains the final backstop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from . import copy_guard

logger = logging.getLogger(__name__)

#: Fixed §3.6 vocabularies — rank, label, tone. Anything else is rejected.
BRIEF_GROUPS: tuple[tuple[int, str, str], ...] = (
    (1, "DO FIRST · BLOCKS OTHERS", "danger"),
    (2, "ANSWER TODAY", "warning"),
    (3, "THIS WEEK", "slate"),
    (4, "NO ACTION — FOR AWARENESS", "success"),
    (5, "IGNORED — NOTHING NEEDED YOU", "ghost"),
)
CHANGES_GROUPS: tuple[tuple[int, str, str], ...] = (
    (1, "CHANGED YOUR WEEK", "warning"),
    (2, "TOLD YOU ONLY", "slate"),
    (3, "LEFT ALONE ON PURPOSE", "success"),
)
RESULT_TITLES = {"brief": "Your morning, in order", "changes": "What this run changed"}

#: R36-7 — the tones a free-form ("digest") result may use. The app's
#: tier chrome falls back to neutral for anything else, so this set is
#: a wire contract, not a style preference.
RESULT_TONES: tuple[str, ...] = ("danger", "warning", "slate", "success",
                                 "ghost")


def expected_result_title(vocabulary: str, record: dict) -> str:
    """The one title this run's result turn must carry.

    `brief`/`changes` are the fixed §3.6 titles. `digest` (R36-7) is
    the automation's OWN title — the whole point of the vocabulary is
    that a Newsletter roundup's result says "This week's newsletters",
    not "Your morning, in order".
    """
    if vocabulary == "digest":
        hint = record.get("narration") or {}
        return str(
            hint.get("title")
            or (record.get("automation") or {}).get("title")
            or "What this run found"
        ).strip()
    return RESULT_TITLES[vocabulary]

#: R31-37 — which vocabulary a run's result wears. [C's judgement; A
#: calls this at `executor_v2` where the run closes.]
#:
#: The founder's Morning work brief — five accounts read, one line
#: posted to Slack — was rendered as `CHANGED YOUR WEEK · 1 item`
#: (E-35). It changed nothing about his week. It read five places and
#: told him what was in them.
#:
#: The old derivation asked "does this automation write anything that
#: is not a draft?", so posting the brief made the run a change-making
#: run. But posting is how a brief is DELIVERED — `workflow.output_block`
#: already leads a `posts` automation with "A brief on your phone". The
#: question that separates the two vocabularies is not whether the run
#: wrote, it is whether it changed something the user owns:
#:
#:   `changes` = it modified the user's world and they may want some of
#:               it undone (a calendar hold, a comment on their ticket,
#:               a page in their workspace).
#:   `brief`   = it read, and it told them — including when the telling
#:               is a Slack line or a staged draft.
#:
#: A draft is telling, not changing, for the same reason its own card
#: says so: "It is a draft — nothing has been sent."
TELLING_TOOLS: frozenset[str] = frozenset({
    "slack__send_message",
    "teams__send_chat_message",
    "gmail__create_draft",
    "outlook__create_draft",
})


def vocabulary_for(write_tools) -> str:
    """`brief` unless the run changed something the user owns."""
    return "changes" if any(
        tool not in TELLING_TOOLS for tool in (write_tools or ())
    ) else "brief"

_MAX_WHY_CHARS = 400
_MAX_TEXT_CHARS = 1200
_NARRATION_MAX_TOKENS = 8000


# ---------------------------------------------------------------------------
# Validation — deterministic, engine-independent
# ---------------------------------------------------------------------------

def _guard(problems: list[str], where: str, text: Any) -> None:
    if not isinstance(text, str) or not text.strip():
        problems.append(f"{where}: empty or not a string")
        return
    if len(text) > _MAX_TEXT_CHARS:
        problems.append(f"{where}: longer than {_MAX_TEXT_CHARS} chars")
    for violation in copy_guard.scan(text):
        problems.append(f"{where}: {violation.rule} {violation.needle!r}")


def validate_drafts(drafts: list[dict], record: dict) -> list[str]:
    """Every §5.1 rule as a named problem. Empty list = accepted."""
    problems: list[str] = []
    if not isinstance(drafts, list) or not drafts:
        return ["no turns emitted"]

    vocabulary = record.get("vocabulary") or "brief"
    status = record.get("status") or "completed"
    steps = {s["step_ref"]: s for s in record.get("steps") or []}
    items_by_step = {
        ref: {i["id"]: i for i in (s.get("items") or [])} for ref, s in steps.items()
    }
    all_item_ids = {i for ids in items_by_step.values() for i in ids}
    read_anything = any(s.get("items") for s in steps.values())

    annotated_steps: set[str] = set()
    results = [d for d in drafts if d.get("kind") == "result"]
    agent_turns = [d for d in drafts if d.get("kind") == "agent"]

    if not agent_turns:
        problems.append("no agent turn: a run opens and closes in the agent's voice")
    if drafts[0].get("kind") != "agent":
        problems.append("the first turn must be the opening agent line")

    for n, d in enumerate(drafts):
        where = f"turn[{n}]"
        kind = d.get("kind")
        if kind in ("agent", "think"):
            _guard(problems, f"{where}.text", d.get("text"))
        elif kind == "draft":
            _guard(problems, f"{where}.text", d.get("text"))
            if not d.get("target_account_id"):
                problems.append(f"{where}: draft without target_account_id")
        elif kind == "annotate":
            ref = d.get("step_ref")
            if ref not in steps:
                problems.append(f"{where}: unknown step_ref {ref!r}")
                continue
            if ref in annotated_steps:
                problems.append(f"{where}: step {ref!r} annotated twice")
            annotated_steps.add(ref)
            wanted = dict(items_by_step[ref])
            for j, item in enumerate(d.get("items") or []):
                iid = item.get("id")
                src = wanted.pop(iid, None)
                if src is None:
                    problems.append(f"{where}.items[{j}]: unknown item id {iid!r}")
                    continue
                _guard(problems, f"{where}.items[{j}].why", item.get("why"))
                if isinstance(item.get("why"), str) and len(item["why"]) > _MAX_WHY_CHARS:
                    problems.append(f"{where}.items[{j}].why: longer than {_MAX_WHY_CHARS}")
                msgs = src.get("msgs") or []
                given = {m.get("idx"): m for m in (item.get("msgs") or [])}
                for idx in range(len(msgs)):
                    if idx not in given:
                        problems.append(
                            f"{where}.items[{j}]: msg {idx} has no why "
                            f"(every message gets the agent's read on it)"
                        )
                    else:
                        _guard(problems, f"{where}.items[{j}].msgs[{idx}].why",
                               given[idx].get("why"))
            for iid in wanted:
                problems.append(
                    f"{where}: item {iid!r} left without a why — "
                    "ignored noise gets a reason too"
                )
            if d.get("rest"):
                _guard(problems, f"{where}.rest", d.get("rest"))
        elif kind == "result":
            problems.extend(_validate_result(
                d, vocabulary, all_item_ids, where,
                expected_title=expected_result_title(vocabulary, record),
            ))
        else:
            problems.append(f"{where}: unknown kind {kind!r}")

    for ref, s in steps.items():
        if s.get("items") and ref not in annotated_steps:
            problems.append(f"step {ref!r} was never annotated")

    if status in ("completed", "partial") and read_anything \
            and record.get("run_kind") in (None, "scheduled", "run_now"):
        if len(results) != 1:
            problems.append(
                f"expected exactly one result turn, got {len(results)}"
            )
    elif status == "failed" and results:
        problems.append("a failed run carries no result turn")

    return problems


def _validate_result(
    d: dict, vocabulary: str, all_item_ids: set, where: str,
    *, expected_title: Optional[str] = None,
) -> list[str]:
    problems: list[str] = []
    if d.get("vocabulary") != vocabulary:
        problems.append(f"{where}: vocabulary must be {vocabulary!r}")
    # Whitespace-canonical title compare: a model copying a wrapped
    # prompt literal emits the newline too — canonicalize in place, the
    # title is prescribed anyway.
    want_title = expected_title if expected_title is not None \
        else RESULT_TITLES.get(vocabulary, "")
    title = " ".join(str(d.get("title") or "").split())
    if title == " ".join(str(want_title).split()):
        d["title"] = title
    else:
        problems.append(f"{where}: title must be {want_title!r}")
    groups = d.get("groups") or []
    if vocabulary == "digest":
        # R36-7: free-form groups — the automation's own organisation of
        # its own material. The rails that survive: sequential ranks,
        # short non-empty CAPS-ish labels, tones from the app's tone
        # set, and (below, shared) every item exactly once.
        if not (1 <= len(groups) <= 6):
            problems.append(f"{where}: digest needs 1-6 groups, "
                            f"got {len(groups)}")
        for gi, g in enumerate(groups, start=1):
            if int(g.get("rank", gi)) != gi:
                problems.append(f"{where}: group {gi} rank mismatch")
            label = str(g.get("label") or "").strip()
            if not (1 <= len(label) <= 48):
                problems.append(f"{where}: group {gi} label must be "
                                "1-48 characters")
            if g.get("tone") not in RESULT_TONES:
                problems.append(f"{where}: group {gi} tone must be one "
                                f"of {list(RESULT_TONES)!r}")
    else:
        expected = BRIEF_GROUPS if vocabulary == "brief" else CHANGES_GROUPS
        got = [(g.get("rank"), g.get("label"), g.get("tone"))
               for g in groups]
        if got != list(expected):
            problems.append(
                f"{where}: groups must be exactly {list(expected)!r} "
                f"in order; got {got!r}"
            )
    seen_refs: set[str] = set()
    for gi, g in enumerate(groups):
        for ri, row in enumerate(g.get("rows") or []):
            rw = f"{where}.groups[{gi}].rows[{ri}]"
            _guard(problems, f"{rw}.text", row.get("text"))
            _guard(problems, f"{rw}.sub", row.get("sub"))
            if row.get("tag") is not None:
                _guard(problems, f"{rw}.tag", row.get("tag"))
            for ref in row.get("item_refs") or []:
                if ref not in all_item_ids:
                    problems.append(f"{rw}: unknown item ref {ref!r}")
                elif ref in seen_refs:
                    problems.append(
                        f"{rw}: item {ref!r} referenced twice — "
                        "everything appears exactly once"
                    )
                seen_refs.add(ref)
    missing = sorted(all_item_ids - seen_refs)
    if missing and vocabulary in ("brief", "digest"):
        problems.append(
            "unaccounted items (every item the run touched appears exactly "
            f"once): {missing}"
        )
    return problems


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

_VOICE_RULES = """\
Voice and copy rules (violations are rejected and cost a retry):
- First person, plain language. You are the user's own agent reporting \
what you just did, to the one person it concerns.
- Every item gets a why — one or two sentences, including everything you \
ignored ("Finance owns spend and you are only cc'd. No action for you."). \
Every message inside an item gets your read on it.
- Real names, real keys, real channels. Never invent detail; never repeat \
the item's title back as its why.
- Admit limits. Drafts are drafts ("nothing has been sent" is rendered by \
the app — do not write it into the draft text). Never promise to send, \
merge, invite, or delete anything.
- Never use: Mission Control, JQL, poll/polls/polling, trigger, routine, \
job, workflow, executed, fetch, Partial, "Temporarily unavailable", \
"re-authentication", "In progress", ISO timestamps, percent figures, raw \
tool identifiers, emoji of any kind.
- Times are local and bare (8:02, 21:44) — never with a leading zero on \
the hour, never ISO.
- One status, once. Never two contradictory status claims in one reply."""

_RANKING_RULES = """\
The result turn: title EXACTLY "Your morning, in order", vocabulary
"brief", exactly five groups with these ranks/labels/tones in order.
Every item id appears in EXACTLY ONE row's item_refs — never two rows,
never zero. Ranking ("what breaks if you ignore it"):
tier 1 DO FIRST · BLOCKS OTHERS — blocks someone else, or a hard date today.
tier 2 ANSWER TODAY — a person is waiting on a REPLY today; a late answer \
costs them. A deadline later this week is NOT tier 2, even when the asker \
is a person — what they need is the work, not a reply.
tier 3 THIS WEEK — dated this week: exports due Friday, reviews you owe, \
slots to answer once the calendar is clearer.
tier 4 NO ACTION — FOR AWARENESS — worth knowing; nothing to do.
tier 5 IGNORED — NOTHING NEEDED YOU — noise, aggregated as count rows with \
named categories.
Every item id must appear in exactly one row's item_refs. A row's text is \
a bold one-line statement; its sub is one sentence of consequence; its tag \
is short ("P1 · due Thu", "Waiting since 21:44", "128")."""

def _digest_rules(record: dict) -> str:
    """R36-7 — the rules for a run whose automation has ONE named job
    that is not a task triage. Item 7 of the founder's automationbugs3
    list: a Newsletter roundup ran the morning-triage prompt and
    produced DO FIRST / ANSWER TODAY about newsletters, because nothing
    below `automation.name` ever reached this module."""
    title = expected_result_title("digest", record)
    hint = record.get("narration") or {}
    goal = str(hint.get("goal") or "").strip()
    desc = str((record.get("automation") or {}).get("description")
               or "").strip()
    job = goal or desc or "deliver exactly what its name promises"
    return (
        f'This automation has ONE job, and it is NOT a task triage: '
        f'{job}\n'
        f'The result turn: title EXACTLY "{title}", vocabulary "digest".\n'
        "Invent 1 to 6 groups that organise THIS run's material the way "
        "its job calls for — by theme, by sender, by project, by day: "
        "whatever serves the reader of this particular digest. Labels "
        "are SHORT CAPS phrases (never the triage tiers — no DO FIRST, "
        "no ANSWER TODAY); each group's tone is one of danger | warning "
        "| slate | success | ghost, and calm material is slate or ghost. "
        "Every item id appears in EXACTLY ONE row's item_refs — never "
        "two rows, never zero. A row's text is a bold one-line "
        "statement; its sub is one sentence of substance from the item "
        "itself; its tag is short. Do not rank by urgency, do not "
        "invent tasks, do not tell the user what to do first — this is "
        "a digest of material, not a to-do list."
    )


_CHANGES_RULES = """\
This run CHANGED things. The result turn:
title EXACTLY "What this run changed" (one line), vocabulary "changes",
exactly three groups with these ranks/labels/tones in order; every item
id in EXACTLY ONE row's item_refs. Rank by what the user may want to undo:
tier 1 CHANGED YOUR WEEK — time and anything undoable (say how to undo).
tier 2 TOLD YOU ONLY — messages that reached the user alone.
tier 3 LEFT ALONE ON PURPOSE — deliberate non-actions, each named.
Write-step annotations use rest to say what you deliberately did not do \
("I did not tag the channel or the reporter.")."""

_FAILED_RULES = """\
This run FAILED and produced no result. No result turn. Open with the \
honest first line: what did not happen, and — if the record names a \
cause — the account and the fix. A run can fail without any account \
failing (it was stopped, it ran out of time, it broke), so do not \
assert that a source was unreachable unless a step below says so. One \
think turn: how many times you retried and why nothing is left \
half-done. Close by offering the fix where there is one to offer."""

#: Appended whenever ANY step failed, whatever the run's status.
#:
#: R31-07, and the reason the founder was told the wrong thing twice.
#: `_FAILED_RULES` only fired on `status == "failed"`, so a PARTIAL run
#: — the ordinary shape, where one account breaks and the rest are read
#: — reached the model with the brief's ranking rules and no failure
#: guidance at all. The model improvised from `failure_reason` and
#: produced "GitHub did not respond" for an account whose actual
#: problem was that the organisation had never approved Toup. The agent
#: KNEW the real reason — asked in the thread minutes later it gave it
#: correctly — so this is not a knowledge gap, it is a narration gap:
#: nothing told it that the supplied sentence was the answer.
#:
#: `failure_reason` carries the string table's `thread_sentence` for
#: that account's reason code (`fixtures/automations/reason-strings.json`,
#: CONTRACTS-R31 §4.4). It is quoted, not paraphrased, because a
#: paraphrase moves the fix: "GitHub did not respond" sends the user to
#: wait and try again, when what they must do is ask an owner to
#: approve an OAuth app.
_FAILED_SOURCE_RULES = """\
SOME SOURCES FAILED. The user is ALREADY seeing a card for each one, \
with its name, its reason and the button that fixes it — do not write \
that card again as prose. Write at most ONE sentence for the whole \
group, saying what is missing from this run because of them, and name \
the account every time — never "an account". A source that failed produced \
NO items: never report a failed read as a count of zero, and never let \
a tier imply the account was read and empty."""

#: Appended only when the supplied reasons are full sentences.
_QUOTE_THE_REASON = """\
Each reason below is the sentence the product uses everywhere else for \
that problem — the user will see the same words on the account's own \
card. Use it EXACTLY AS GIVEN: do not reword it, do not shorten it, do \
not soften it. You may add at most one sentence of your own around it."""

#: Appended when they are not.
#:
#: This distinction is the whole of the fix, and getting it wrong makes
#: things WORSE than before. `failure_reason` is documented as the
#: string table's `thread_sentence`, but its only producer today is
#: `executor_v2` writing `turn["detail"]` — the verb dictionary's short
#: fragment — and `_failure_reason` does not recognise
#: `org_approval_needed` at all, so the exact GitHub case this was
#: written for arrives as "it did not answer". An unconditional "quote
#: it verbatim" would therefore MANDATE the vague answer the block
#: exists to prevent: the model could previously improvise its way to
#: the true reason, and would now be forbidden from doing anything but
#: repeating the wrong one. So the instruction is conditional on what
#: was actually supplied, and when the reason is a fragment the model
#: is told to be honest about not knowing rather than to dress it up.
_DO_NOT_INVENT_A_REASON = """\
The reasons below are short fragments, not full sentences, and some of \
them mean only "this did not work" — they are not diagnoses. Say which \
account failed and that you could not read it. Do NOT present a \
fragment as the cause, and do NOT supply a cause of your own: saying \
"the access expired" or "the service was down" when you were not told \
that is a guess the user will act on. If you do not know why, say you \
do not know why and that you will find out."""


def _looks_like_a_sentence(text) -> bool:
    """A supplied reason is quotable only if it IS a sentence.

    The fragments the dictionary produces ("it did not answer",
    "access expired") are lower-case clauses; a `thread_sentence`
    starts with a capital and ends in a full stop.
    """
    s = str(text or "").strip()
    return len(s) > 25 and s[:1].isupper() and s.endswith(".")


#: R31-08 / §4.9 — the shape of a QUESTION run's narration.
#:
#: A question the user asks in the thread that needs fresh reading is a
#: run, not a paragraph the model composes from memory. On 26 August the
#: founder asked for "everything latest in all chanels" and got forty
#: seconds of a loading pill and then prose: no job card, no per-account
#: rows, counts nobody could check, and Teams named although it is not
#: on that automation's canvas.
#:
#: There is no result turn — the answer IS the result, and a ranked
#: five-tier brief for "what is the latest in Gmail" would be the
#: `changes`-vocabulary mistake in another costume. `_validate_result`
#: already exempts this run kind from requiring one.
_QUESTION_RULES = """\
The user asked a question in the thread and you read to answer it. NO \
result turn — the answer itself is the result, and a ranked brief here \
would be an answer to a question nobody asked.

Open with one line saying how many accounts you are about to look at. \
Then answer: ONE short paragraph per account, naming the account, in \
the order you read them. Every count you state must be a count you \
actually got back from that account's step — if a step returned three \
threads, do not write "a few". Nothing you did not read may appear: not \
an account the automation does not have, not a name the user mentioned \
once, not a source that would round the answer out. If an account was \
not read, say so in its own paragraph rather than leaving it out — a \
silent omission reads as "nothing there"."""


def build_prompt(record: dict) -> str:
    vocabulary = record.get("vocabulary") or "brief"
    status = record.get("status") or "completed"
    title = (record.get("automation") or {}).get("title") or "this automation"

    if record.get("run_kind") == "question":
        shape = _QUESTION_RULES
    elif status == "failed":
        shape = _FAILED_RULES
    elif vocabulary == "digest":
        shape = _digest_rules(record)
    elif vocabulary == "changes":
        shape = _CHANGES_RULES
    else:
        shape = _RANKING_RULES

    # A partial run is the ORDINARY failure shape — one account breaks,
    # the rest are read — and it took the brief's rules with no failure
    # guidance at all. Keyed on the steps, not on the status, so it
    # cannot be missed by a status the table does not know.
    failed_steps = [s for s in (record.get("steps") or [])
                    if isinstance(s, dict) and not s.get("ok", True)]

    # R36-7: the automation's own task statement, when it has one. The
    # record used to carry nothing but the name, so every run of every
    # automation was narrated as if its job were the same.
    desc = str((record.get("automation") or {}).get("description")
               or "").strip()
    job_line = f' Its job, in the user\'s words: "{desc}"' if desc else ""
    parts = [
        f'You are narrating one run of the automation "{title}" into its '
        f"thread.{job_line} The engine already did the work — the "
        "dispatch record "
        "below is everything it read and changed, with minted item ids. "
        "You supply the judgement and the voice: the opening line, a why "
        "for every item, the ranked result, think turns for judgement "
        "calls, a draft when an answer is owed, the closing line.",
        _VOICE_RULES,
        shape,
    ]
    if failed_steps:
        parts.append(_FAILED_SOURCE_RULES)
        reasons = [
            (s.get("connector_name") or s.get("account_id") or "that account",
             s.get("failure_reason"))
            for s in failed_steps if s.get("failure_reason")
        ]
        if reasons:
            quotable = all(_looks_like_a_sentence(r) for _, r in reasons)
            parts.append(
                _QUOTE_THE_REASON if quotable else _DO_NOT_INVENT_A_REASON
            )
            parts.append(
                ("The reasons, to use as they are: " if quotable
                 else "What came back, which is all you know: ")
                + " · ".join(f"{who}: {reason}" for who, reason in reasons)
            )
    parts += [
        "Emit turns in §order: agent opening → one annotate per step (whys "
        "for every item and message) → result (unless failed, or unless "
        "this is a question run) → think per "
        "judgement call → draft if an answer is owed → agent close. A "
        "draft turn MUST carry target_account_id (the account the draft "
        "waits in) and target_ref; emit a draft only when the run's work "
        "actually owes someone an answer.",
    ]
    rules = record.get("rules") or []
    if rules:
        parts.append("Rules you added (obey them; they outrank everything "
                     "below): " + " · ".join(rules))
    facts = record.get("memory_facts") or []
    if facts:
        parts.append("What you remember about this user (use it to judge, "
                     "never recite it): " + " · ".join(
                         f"[{f.get('category')}] {f.get('text')}" for f in facts[:24]))
    parts.append("DISPATCH RECORD:\n" + json.dumps(
        {k: record.get(k) for k in
         ("run_kind", "vocabulary", "status", "steps")},
        ensure_ascii=False, indent=1))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

def _emit_turns_tool(vocabulary: str,
                     expected_title: Optional[str] = None) -> dict:
    """The `automation_emit_turns` structured-output tool schema (§4.2)."""
    if vocabulary == "digest":
        # R36-7: labels are the automation's own; tones stay enumerated.
        label_schema: dict = {"type": "string", "maxLength": 48}
        tone_schema: dict = {"enum": list(RESULT_TONES)}
    else:
        groups = BRIEF_GROUPS if vocabulary == "brief" else CHANGES_GROUPS
        label_schema = {"enum": [g[1] for g in groups]}
        tone_schema = {"enum": [g[2] for g in groups]}
    return {
        "name": "automation_emit_turns",
        "description": "Emit the run's turns, in order, exactly once.",
        "input_schema": {
            "type": "object",
            "required": ["turns"],
            "properties": {"turns": {"type": "array", "items": {
                "type": "object",
                "required": ["kind"],
                "properties": {
                    "kind": {"enum": ["agent", "think", "annotate",
                                       "result", "draft"]},
                    "text": {"type": "string"},
                    "step_ref": {"type": "string"},
                    "items": {"type": "array", "items": {
                        "type": "object",
                        "required": ["id", "why"],
                        "properties": {
                            "id": {"type": "string"},
                            "why": {"type": "string"},
                            "msgs": {"type": "array", "items": {
                                "type": "object",
                                "required": ["idx", "why"],
                                "properties": {"idx": {"type": "integer"},
                                                "why": {"type": "string"}}}},
                        }}},
                    "rest": {"type": "string"},
                    "title": {"type": "string"},
                    "vocabulary": {"enum": ["brief", "changes", "digest"]},
                    "groups": {"type": "array", "items": {
                        "type": "object",
                        "required": ["rank", "label", "tone", "rows"],
                        "properties": {
                            "rank": {"type": "integer"},
                            "label": label_schema,
                            "tone": tone_schema,
                            "rows": {"type": "array", "items": {
                                "type": "object",
                                "required": ["text", "sub", "item_refs"],
                                "properties": {
                                    "text": {"type": "string"},
                                    "sub": {"type": "string"},
                                    "tag": {"type": ["string", "null"]},
                                    "item_refs": {"type": "array",
                                                   "items": {"type": "string"}},
                                }}},
                        }}},
                    "target_account_id": {"type": "string"},
                    "target_ref": {"type": "string"},
                },
            }}},
        },
    }


async def narrate_run(record: dict, *, complete=None) -> dict:
    """Run the narration pass. Returns
    `{"turns": [TurnDraft], "problems": [str], "attempts": int}` —
    `problems` non-empty means the engine should apply its completeness
    net (§4.2) to whatever survived validation; it never means silence.

    `complete` — awaitable `(prompt, tool_schema) -> dict` (the tool
    input). Defaults to the pinned-model LLM call; injectable so the
    protocol is testable without a model.
    """
    if complete is None:
        complete = _default_complete
    vocab = record.get("vocabulary") or "brief"
    tool = _emit_turns_tool(
        vocab, expected_title=expected_result_title(vocab, record),
    )
    prompt = build_prompt(record)
    drafts: list[dict] = []
    problems: list[str] = ["not attempted"]
    attempts = 0
    for attempts in (1, 2):
        try:
            payload = await complete(prompt, tool)
        except Exception as e:  # noqa: BLE001 — narration must not kill the run
            logger.warning("[automations] narrator LLM call failed: %s: %s",
                           type(e).__name__, str(e)[:200])
            return {"turns": drafts, "problems": [f"llm: {type(e).__name__}"],
                    "attempts": attempts}
        drafts = payload.get("turns") if isinstance(payload, dict) else None
        drafts = drafts if isinstance(drafts, list) else []
        problems = validate_drafts(drafts, record)
        if not problems:
            break
        prompt = (
            build_prompt(record)
            + "\n\nYour previous emission was rejected. Fix EVERY problem "
              "and emit the full turn list again:\n- "
            + "\n- ".join(problems[:40])
        )
    return {"turns": drafts, "problems": problems, "attempts": attempts}


async def _default_complete(prompt: str, tool: dict) -> dict:
    """The pinned-model call. Never `model=None` on a background path.
    `llm_service` has no tool-forcing method, so the tool's input schema
    rides the prompt and the reply is parsed as JSON (the repo's
    established `complete_with_json` pattern)."""
    import os
    import re

    from app.config import settings
    from app.services.llm_service import get_llm_service

    model = getattr(settings, "automation_narrator_model", None) \
        or os.environ.get("AUTOMATION_NARRATOR_MODEL") \
        or getattr(settings, "memory_extraction_model", None)
    response = await get_llm_service().complete_with_json(
        messages=[{
            "role": "user",
            "content": (
                prompt
                + "\n\nReply ONLY as JSON matching this schema (the "
                  f"`{tool['name']}` input):\n"
                + json.dumps(tool["input_schema"], ensure_ascii=False)
            ),
        }],
        model=model,
        temperature=0.2,
        max_tokens=_NARRATION_MAX_TOKENS,
    )
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, str):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        return json.loads(raw)
    return raw if isinstance(raw, dict) else {}
