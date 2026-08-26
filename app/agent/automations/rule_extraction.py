"""Rule extraction — R31-18 [C].

The founder's Morning work brief showed `LINES IT WILL NOT CROSS 0`
(E-21) for an automation whose own steps ended `Told you in Slack ·
one line, no thread`. The constraint was there, in the automation's own
description, and it had never become a rule — so the Rules tab was
empty, the workflow could not show what it would not do, and nothing
downstream could obey it.

This module turns constraints already stated — in the description, in
the setup conversation, in the steps — into `rules[]` rows. A owns the
seam that calls it (`compiler.py` / `describe_compile.py`) and the
back-fill migration that runs it over every existing automation; this
file owns what counts as a rule and what it says.

Two rules about the rules:

**Quote, do not paraphrase.** A rule is a line the agent will not
cross. Paraphrasing changes where the line is, and the user cannot see
that it moved. Every row carries `source` — the exact phrase it came
from — so a rule can always be read back against what was actually
said. The canvas's own examples are the user's voice, imperative and
first-person ("Never post in a channel — DM me instead.", "Leave
anything finance owns alone."), and that is the register extraction
preserves.

**Conservative, and it says how conservative.** This runs as a bulk
migration over automations nobody is watching, and the two failure
directions are not symmetric. A MISSED rule means the agent crosses a
line the user drew. An INVENTED rule means the workflow lists a
constraint the user never set and the automation quietly does less. So
extraction fires only on explicit constraint language, never on
inference from a description's general shape, and the migration
reports its counts — a back-fill that silently found nothing is
indistinguishable from one that never ran.

Rule text is the USER's own words, like a rule typed into "+ Add a
rule". It is not scanned by the copy guard for the same reason that
surface is not: refusing to record a constraint because the user
phrased it with one of our banned words would drop a line they drew.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Iterable, Optional

#: Where a rule came from. Kept on the row so the Rules tab can say so
#: and so the back-fill is auditable after the fact.
ORIGINS = ("description", "setup", "steps", "user")

#: A clause ends at a sentence break, or where the NEXT constraint
#: begins. Em dashes are kept INSIDE the clause — "Never post in a
#: channel — DM me instead." is one rule, and cutting it at the dash
#: would keep the prohibition and drop the instruction that makes it
#: usable. A bare comma is kept too ("Leave finance, legal and HR
#: alone"), so the only commas that end a clause are the ones followed
#: by a conjunction or by another constraint word: without that,
#: "Only unread mail, and never post anywhere" came out as a single
#: rule reading "Only unread mail, and never post anywhere." — which
#: is two lines the user drew, recorded as one they cannot edit apart.
#: An apostrophe is `'` on a keyboard and `’` on a phone — iOS
#: substitutes the typographic one by default, and the founder types on
#: a phone. `don'?t` matched "dont" and "don't" and silently dropped
#: "don’t post anywhere", which is a constraint the user drew and the
#: workflow would then not show. A dropped rule is the expensive
#: direction: the agent crosses a line nobody recorded.
_APOS = r"['’]"

_CONSTRAINT_OPENERS = (
    rf"and|or|but|never|only|skip|ignore|do\s+not|don{_APOS}?t"
)
_CLAUSE_END = rf"(?=[.;\n]|,\s*(?:{_CONSTRAINT_OPENERS})\b|$)"

#: A constraint word only starts a constraint when it starts a CLAUSE.
#: Mid-sentence, "only" is an ordinary adjective: "It reads the only
#: inbox you use." produced the rule "Only inbox you use." — a line the
#: user never drew, in the one place that claims to list exactly the
#: lines they did.
_CLAUSE_START = r"(?:^|(?<=[.;:,])\s*|(?<=\s)(?:and|or|but|then)\s+)"

#: Explicit constraint language only. Each pattern captures the clause
#: and `render` turns the match into the row's text.
#:
#: Ordered: the more specific shapes first, so "leave X alone" is not
#: also matched as a bare verb phrase.
_PATTERNS: tuple[tuple[str, str], ...] = (
    # "leave anything finance owns alone" — the canvas's own example.
    (rf"\bleave\s+(?P<body>.+?)\s+alone\b", "Leave {body} alone."),
    # "never post anywhere", "never reply to recruiters"
    (rf"\bnever\s+(?P<body>.+?){_CLAUSE_END}", "Never {body}."),
    # "don't post in #general", "do not send anything"
    (rf"\b(?:don{_APOS}?t|do not)\s+(?P<body>.+?){_CLAUSE_END}",
     "Never {body}."),
    # "skip anything from recruiters", "ignore newsletters"
    (rf"\bskip\s+(?P<body>.+?){_CLAUSE_END}", "Skip {body}."),
    (rf"\bignore\s+(?P<body>.+?){_CLAUSE_END}", "Ignore {body}."),
    # "only unread", "only from my team" — clause-initial only, or
    # every "the only X" in a description becomes a rule.
    (rf"{_CLAUSE_START}only\s+(?P<body>.+?){_CLAUSE_END}", "Only {body}."),
    # "hold time on my calendar rather than booking meetings" — the
    # canvas's third example. A constraint stated as a preference
    # rather than a prohibition: the line it will not cross is the
    # second half. Kept whole and in the user's words, because
    # rewriting it as "Never book meetings." drops the alternative
    # that tells the agent what to do instead. Bounded by the same
    # clause end as everything else — unbounded it ran through the
    # comma and swallowed the NEXT constraint, so two lines the user
    # drew separately became one row they cannot edit apart.
    (rf"(?P<body>[^.;\n]*?\brather than\b.+?){_CLAUSE_END}", "{body}."),
    # "one line, no thread" — two constraints, matched separately.
    (r"\bone line\b", "One line only."),
    (r"\bno thread\b", "No thread."),
    (r"\bnothing else\b", "Nothing else."),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), template)
    for pattern, template in _PATTERNS
)

#: A captured clause this short is a fragment, not a constraint
#: ("only it", "never x"). A rule nobody can act on is noise in the one
#: place the user looks to see what the agent will not do.
_MIN_BODY = 3

#: `add_rule` caps stored rule text at 300; match it here so extraction
#: can never produce a row the manual path would refuse.
_MAX_TEXT = 300


def _clean(text: Optional[str]) -> str:
    return " ".join(str(text or "").split())


def _tidy_body(body: str) -> str:
    body = _clean(body).rstrip(" ,;:")
    # A trailing conjunction means the clause ran into the next one.
    body = re.sub(r"\s+(?:and|or|but)$", "", body, flags=re.IGNORECASE)
    return body


def _sentence(text: str) -> str:
    """A rule reads as a sentence: it is drawn on a card of its own."""
    text = _clean(text)
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _row(text: str, *, origin: str, source: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "text": _sentence(text)[:_MAX_TEXT],
        "origin": origin,
        "source": source[:_MAX_TEXT],
        "added_at": datetime.utcnow().isoformat() + "Z",
    }


def rules_from_text(text: str, *, origin: str) -> list[dict]:
    """Every constraint stated in one piece of text.

    Matches are taken from the ORIGINAL string, so two different
    constraints in one sentence ("only unread, and never reply") each
    produce a row.
    """
    source_text = _clean(text)
    if not source_text:
        return []
    rows: list[dict] = []
    for pattern, template in _COMPILED:
        for match in pattern.finditer(source_text):
            groups = match.groupdict()
            if "body" in groups:
                body = _tidy_body(groups.get("body") or "")
                if len(body) < _MIN_BODY:
                    continue
                rendered = template.format(body=body)
            else:
                rendered = template
            rows.append(_row(rendered, origin=origin,
                             source=match.group(0)))
    return rows


def extract_rules(
    *,
    description: Optional[str] = None,
    setup_text: Optional[Iterable[str]] = None,
    steps: Optional[Iterable[str]] = None,
    existing: Optional[Iterable[dict]] = None,
) -> list[dict]:
    """The rules an automation already states but has never recorded.

    `existing` is the automation's current `rules[]`; anything already
    there is left alone and never duplicated. The back-fill is
    idempotent on that basis — running it twice adds nothing the second
    time, which is what makes it safe to re-run over a tenant whose
    first pass half-finished.
    """
    found: list[dict] = []
    found += rules_from_text(description or "", origin="description")
    for line in setup_text or ():
        found += rules_from_text(line, origin="setup")
    for line in steps or ():
        found += rules_from_text(line, origin="steps")

    seen = {_key(r.get("text")) for r in (existing or []) if r.get("text")}
    out: list[dict] = []
    for row in found:
        key = _key(row["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _key(text: Optional[str]) -> str:
    """Dedupe key: the words, lower-cased, punctuation dropped.

    `one line, no thread` reaches this module from the description AND
    from the step that renders it, so the same constraint arrives twice
    by two routes. Keying on the rendered text rather than the source
    phrase is what collapses them.
    """
    return re.sub(r"[^a-z0-9 ]+", "", _clean(text).lower())
