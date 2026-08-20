"""The curator — the ONE writer of memory files (v3 §2).

Every memory write converges here:

* the ops SCHEMA and its prompt (the contract the model must satisfy),
* the two `/instruct` entry points — one file, or "tell your agent what to
  remember" with no file named,
* `curate_turn`, the post-turn writer that replaces the sentence extractor,
  with its pre-gates,
* validation + apply, delegated to `memory_file_ops` so the deterministic
  half is testable without an LLM.

Why one service: round 8 had five independent writers (the extractor, the
active-task detector, the relationship mirror, `memory_store`, the voice
tunnel), each with its own idea of the gate. The gate can only be as strong
as the weakest producer, so v3 has exactly one.

Every producer now routes here, and `tests/test_curator_producers.py` is the
structural guard that they still do.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import MemoryFile
from app.memory_files import (
    ALWAYS_INJECTED_SLUGS,
    CURRENT_CONTEXT_SLUG,
    LEARNED_SLUG,
    PROFILE_SLUG,
    SECTION_ORDER,
    title_from_slug,
)
from app.services import memory_file_ops as ops
from app.services.user_identity import UserIdentity, resolve_user_identity

logger = logging.getLogger(__name__)

#: Bodies are shipped whole while the whole corpus fits; past that the
#: prompt carries the index plus the files the instruction actually names.
FULL_BODY_BUDGET_CHARS = 24 * 1024
TOP_K_BODIES = 6


# ── The contract the model is held to ─────────────────────────────────

OPS_CONTRACT = """Reply with ONLY valid JSON: {"ops": [...]}. Allowed ops:

- {"op":"create_file","section":"people|topics|areas","slug":"people/quill-marsden","title":"Quill Marsden","description":"..."}
- {"op":"update_description","slug":"areas/ielts","description":"..."}
- {"op":"add","slug":"areas/ielts","bullet":"...","change":"..."}
- {"op":"rewrite","slug":"areas/ielts","match":"<the existing bullet, EXACTLY>","bullet":"...","change":"..."}
- {"op":"remove","slug":"areas/ielts","match":"<the existing bullet, EXACTLY>","change":"..."}
- {"op":"link","slug":"areas/ielts","links":["people/quill-marsden"]}
- {"op":"delete_file","slug":"topics/old-thing","change":"..."}

BULLET VOICE — subjectless telegraphic third person, one complete fact per bullet:
  GOOD  "uses an Android phone"
  BAD   "You use an Android phone"        (the subject is implied, never restated)
  BAD   "The user prefers dark mode"      (never "the user")
  BAD   "wants to"                        (a truncation, not a fact)
- A longer bullet still holds ONE fact: the thing itself, anyone in it by full
  name, every date absolute, and the modality in parentheses — clauses joined
  with a semicolon rather than split into fragments.

EVERY SPECIFIC IN A BULLET COMES FROM THE CONVERSATION, AND NOWHERE ELSE.
The slugs, titles and phrasings printed anywhere in this prompt show SHAPE.
They are not evidence. Never copy a name, date, number, place, course or
institution out of them into a bullet, a title or a description. If you did
not read it in the conversation above, you may not write it down.

- The file's subject is implied. In a people/ file the subject is THAT PERSON.
- Name other people normally, in full. Resolve relative time to absolute dates
  ("Aug 19, 2026"). Preserve every name, number, date and time EXACTLY.
- WRITE IN THE LANGUAGE THE PERSON USED. A Persian conversation produces
  Persian bullets, an Arabic one Arabic bullets. Their memory is read back to
  them in their own words; translating it into English loses the person. The
  voice rule above still holds — subjectless, telegraphic, one fact.
- Record an unresolved contradiction explicitly ("calendar says X while the app
  says Y - discrepancy unresolved") rather than picking a side.
- Never store internal ids, tool names, tool parameters, or UUIDs.
- Text in another script (Farsi, Arabic, ...) is preserved exactly as written.
- Cross-reference another file inline as [[slug]]; it must be a file that exists
  or that this same batch creates.
- No trailing period unless the bullet holds more than one sentence; join
  related clauses with a semicolon.

DESCRIPTION — every file has one, in exactly this shape, with an EM DASH (—)
as the first separator, a semicolon before "read when", and a trailing period:
  "<what this is> — <scope>; read when <trigger>."
  e.g. "Your IELTS preparation — tutor, dates and band targets; read when IELTS or the exam comes up."

`change` is one short sentence for the user's memory log, naming the file and
what moved: "Added Cycling: rides to work on Tuesdays.", "Removed the old job
from Work." It is REQUIRED on add, rewrite, remove and delete_file.

WHERE A FACT GOES — pick the one most specific file, then commit to it:
- you/profile — who they ARE: name, languages, where they live, health, family,
  the devices and accounts they use, and standing arrangements as one line each.
  Identity, not a catch-all. A subject with a life of its own leaves profile.
- people/<person> — a fact about ANOTHER person, and everything about how they
  relate to this one. Create the file on the first durable fact about them, and
  put the facts about them IN it; an area file may reference them as [[slug]],
  but it is not where they live.
- areas/<subject> — an ongoing commitment with a state that moves: a study goal,
  an exam, a job hunt, a health programme, a project.
- topics/<subject> — a taste, interest or domain they return to: music, film,
  food, a language, a technology.
- Open the area or topic file as soon as its subject has ONE durable fact and
  will plainly gather more. A taste in music is `topics/music`, not a line in
  profile. Parking a real subject in profile because it is small today is the
  single most common routing mistake — the generic catch-all files this system
  replaced were built exactly one small fact at a time.

RULES:
- Prefer rewrite over add when a bullet already covers the fact. Merge, do not
  accumulate.
- `match` must be the existing bullet text CHARACTER FOR CHARACTER.
- Never create a people/ file for the person whose memory this is.
- If nothing should change, reply {"ops": []}.
"""

#: Specifics that appear in this module's prompts as TEACHING EXHIBITS and
#: nowhere in any real conversation. The first live model run (2026-08-20, CI
#: run 32429017640) copied the tutor exhibit into a user's memory verbatim —
#: the name AND the date range invented to illustrate the shape — because that
#: conversation happened to mention a tutor teaching over Teams. An exhibit is
#: the one kind of prompt text a model can mistake for evidence, so provenance
#: is enforced structurally and not only asked for: `leaked_exhibit` refuses a
#: bullet carrying one of these unless the turn itself contains it. Root cause
#: #1 was the writer being unable to tell the user's words from machine text;
#: this is that same defect arriving from the other side of the prompt.
#:
#: DATES ARE DELIBERATELY NOT IN THIS LIST, and must never be added. The
#: contract REQUIRES the writer to resolve time absolutely ("August 30th 2026"
#: -> "Aug 30, 2026"), so a correct bullet's date is normalised and by
#: construction does not appear verbatim in the evidence. Listing one here
#: refuses the very rewriting the contract demands — measured while building
#: this guard: "Aug 30, 2026" tripped against a source reading "August 30th
#: 2026". Only spellings a writer has no licence to change belong here.
#: Every entry must be a spelling whose ONLY plausible origin is this prompt.
#: A real name fails that test in both directions: the leak is real, but so is
#: the false positive — a Persian speaker writing "گوگوش" would have the Latin
#: bullet refused for having no source, and a silently dropped true fact is
#: worse than the leak. So the exhibits themselves were made synthetic instead,
#: and the guard covers those.
CONTRACT_EXHIBIT_SPECIFICS: tuple[str, ...] = (
    "quill marsden",
    "quill-marsden",
    "old-thing",
    "rides to work on tuesdays",
)


def leaked_exhibit(text: str, *, source_text: str) -> Optional[str]:
    """The exhibit specific this text carries and its source does not, if any.

    `source_text` is everything the ops were legitimately derived from — the
    turn, or the batch of rows a migration is re-curating. A specific present
    there is the user's, however much it looks like an exhibit.
    """
    hay = (text or "").casefold()
    src = (source_text or "").casefold()
    for specific in CONTRACT_EXHIBIT_SPECIFICS:
        if specific in hay and specific not in src:
            return specific
    return None


def _identity_block(identity: UserIdentity) -> str:
    if not identity.known:
        return (
            "WHOSE MEMORY THIS IS: unknown (the account has no real name yet). "
            "Do not create a people/ file for anyone who might be the owner."
        )
    names = ", ".join(sorted(identity.aliases))
    return (
        f"WHOSE MEMORY THIS IS: {identity.display_name} (also referred to as: "
        f"{names}). Facts about this person go in you/profile — NEVER in a "
        "people/ file."
    )


def _index_block(rows: Sequence[MemoryFile]) -> str:
    if not rows:
        return "(no files yet)"
    order = [s.value for s in SECTION_ORDER]
    ordered = sorted(
        rows,
        key=lambda r: (
            order.index(r.section) if r.section in order else len(order),
            (r.title or "").lower(),
        ),
    )
    return "\n".join(
        f"- {r.slug} — {r.title}: {r.description or '(no description)'}"
        for r in ordered
    )


def _bodies_block(rows: Sequence[MemoryFile]) -> str:
    parts: List[str] = []
    for row in rows:
        body = (row.body_md or "").strip() or "(empty)"
        parts.append(f"### {row.slug} — {row.title}\n{body}")
    return "\n\n".join(parts) if parts else "(no bodies loaded)"


def _pick_bodies(
    rows: Sequence[MemoryFile], focus: str, *, always: Sequence[str] = ()
) -> List[MemoryFile]:
    """Every body while the corpus is small, else the index plus top-K.

    The always-list is loaded regardless: the writer cannot decide whether a
    fact is already in Profile without seeing Profile.
    """
    total = sum(len(r.body_md or "") for r in rows)
    if total <= FULL_BODY_BUDGET_CHARS:
        return list(rows)
    pinned = [r for r in rows if r.slug in set(always)]
    ranked = [r for r in ops.lexical_rank(focus, rows) if r.slug not in set(always)]
    return pinned + ranked[:TOP_K_BODIES]


def build_instruct_prompt(
    *,
    instruction: str,
    identity: UserIdentity,
    index: str,
    bodies: str,
    today: str,
    scope_slug: Optional[str] = None,
) -> str:
    if scope_slug:
        task = (
            f'The user is editing the file "{scope_slug}" and asked for this '
            f'change:\n"{instruction}"\n\n'
            "Apply it with the fewest ops that fully honor it, touching NO "
            "other file. If they ask to remove something that is not there, "
            "reply {\"ops\": []}."
        )
    else:
        task = (
            "The user typed this into \"Tell your agent what to remember\":\n"
            f'"{instruction}"\n\n'
            "Route it to the RIGHT file — an existing one when there is a fit, "
            "a new topics/ or areas/ or people/ file when there is not. Facts "
            "about the owner belong in you/profile. Corrections about how to "
            "work with them belong in learned. Do not create a file for a "
            "one-off; do not store reminders, tasks or momentary states."
        )
    return (
        f"Today is {today}.\n"
        f"{_identity_block(identity)}\n\n"
        "You maintain this person's memory files.\n\n"
        f"FILE INDEX:\n{index}\n\n"
        f"FILE BODIES:\n{bodies}\n\n"
        f"{task}\n\n"
        f"{OPS_CONTRACT}"
    )


# ── The model call ────────────────────────────────────────────────────

def _llm(api_key: Optional[str]):
    if api_key:
        from app.services.llm_service import LLMService

        return LLMService(api_key=api_key)
    from app.services.llm_service import get_llm_service

    return get_llm_service()


#: Seconds before the single transient retry. Module-level so tests zero it.
EXTRACTION_RETRY_BACKOFF_S = 1.5


async def _complete_json_with_retry(llm_service, prompt: str):
    """One JSON call with ONE retry after a short backoff.

    A6-2, moved here from the retired extractor with the LLM call it
    protected. The writer is a single un-fallbacked model call per turn, so
    a transient provider or network blip silently loses everything the user
    stated that turn — and it runs fire-and-forget after the reply, where no
    caller is watching. One retry covers the transient class; a persistent
    failure (bad key, dead proxy) still raises on the second attempt, which
    is what parks the turn in the outbox.

    Distinct from the VALIDATION retry in `_run_ops`: that one re-asks a
    model that answered, with the validator's complaints. This one re-asks a
    model that never answered at all.
    """
    import asyncio

    try:
        return await llm_service.complete_with_json(
            messages=[{"role": "user", "content": prompt}],
            model=settings.memory_extraction_model,
            temperature=0.0,
        )
    except Exception as first_err:
        logger.warning(
            "[curator] model call attempt 1 failed, retrying once in %.1fs: "
            "%s: %s",
            EXTRACTION_RETRY_BACKOFF_S,
            type(first_err).__name__, str(first_err)[:200],
        )
        await asyncio.sleep(EXTRACTION_RETRY_BACKOFF_S)
        return await llm_service.complete_with_json(
            messages=[{"role": "user", "content": prompt}],
            model=settings.memory_extraction_model,
            temperature=0.0,
        )


async def propose_plan(llm_service, prompt: str) -> Dict[str, Any]:
    """One JSON call, returning the WHOLE object the model replied with.

    `propose_ops` is the ordinary caller and wants only `ops`. The
    migration additionally needs the model's `dispositions` array — the
    per-source-row accounting that makes "where did this memory go?"
    answerable — so the parse lives here and both callers share it.
    """
    response = await _complete_json_with_retry(llm_service, prompt)
    parsed: Any = response
    if hasattr(response, "content"):
        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("curator response is not an object")
    return parsed


async def propose_ops(llm_service, prompt: str) -> List[Dict]:
    """One JSON call. Raises on a response that is not an ops object."""
    return (await propose_plan(llm_service, prompt)).get("ops", [])


#: The op fields that end up in front of the user — three stored, one in the
#: memory log. A leaked exhibit in any of them is a fabricated fact.
_LEAK_CHECKED_FIELDS = ("bullet", "title", "description", "change")


def _refuse_leaks(raw_ops: Any, source_text: str) -> Tuple[Any, List[str]]:
    """Drop ops that carry a prompt exhibit the evidence never mentioned.

    `source_text` is the evidence, NOT the prompt: the prompt contains the
    exhibits by construction, so checking against it would pass everything.
    """
    if not isinstance(raw_ops, list) or not source_text:
        return raw_ops, []
    kept: List[Any] = []
    complaints: List[str] = []
    for op in raw_ops:
        if not isinstance(op, dict):
            kept.append(op)
            continue
        hit = next(
            (
                leak for leak in (
                    leaked_exhibit(str(op.get(f) or ""), source_text=source_text)
                    for f in _LEAK_CHECKED_FIELDS
                ) if leak
            ),
            None,
        )
        if hit:
            complaints.append(
                f"dropped a change carrying \"{hit}\" — that is an example from "
                "the instructions, not something this conversation said"
            )
            logger.warning(
                "[memory_curator] refused a prompt-exhibit leak: %r", hit,
            )
            continue
        kept.append(op)
    return kept, complaints


async def _run_ops(
    db: AsyncSession,
    user_id: str,
    prompt: str,
    *,
    api_key: Optional[str],
    identity: UserIdentity,
    scope_slug: Optional[str] = None,
    source_text: str = "",
) -> Dict[str, Any]:
    """Propose → validate → (retry once with the complaints) → apply."""
    llm_service = _llm(api_key)
    raw_ops = await propose_ops(llm_service, prompt)

    existing = await ops._all_files(db, user_id)
    raw_ops, leak_complaints = _refuse_leaks(raw_ops, source_text)
    scoped, out_of_scope = _scope(raw_ops, scope_slug)
    plan = ops.validate_ops(scoped, existing, identity=identity)
    plan.complaints.extend(leak_complaints)
    if out_of_scope:
        # Say so. Dropping silently is how a user types two things, sees
        # "Saved 1 change", and never learns the other one went nowhere.
        plan.complaints.append(
            f"{out_of_scope} change(s) named another file and were not "
            "applied — this box edits one file"
        )
    if plan.complaints and not plan.accepted and raw_ops:
        # Whole proposal rejected — one retry, with the validator's own
        # words. Round 8's pattern, kept: the complaints are precise enough
        # to fix ("no bullet reads exactly X"), and a second full rejection
        # is a real answer, not a transient.
        retry = (
            prompt
            + "\n\nYour previous reply was rejected for these reasons — fix "
              "them and reply again:\n- " + "\n- ".join(plan.complaints)
        )
        raw_ops = await propose_ops(llm_service, retry)
        # The retry is a fresh proposal from the same prompt, so it can leak
        # the same way the first one did. A guard the retry path walks past
        # is not a guard.
        raw_ops, retry_leaks = _refuse_leaks(raw_ops, source_text)
        scoped, _ = _scope(raw_ops, scope_slug)
        plan = ops.validate_ops(scoped, existing, identity=identity)
        plan.complaints.extend(retry_leaks)

    result = await ops.apply_ops(db, user_id, plan)
    return {
        "applied": result["applied"],
        "rejected": plan.complaints,
        "changed_files": result["changed_files"],
        # For the sentence a human reads. See `_note`.
        "changed_titles": result.get("changed_titles") or [],
    }


def _scope(raw_ops: Any, scope_slug: Optional[str]) -> Tuple[Any, int]:
    """Drop ops naming another file when the box edits exactly one.

    "remove the IELTS dates" typed on the Music page must not edit IELTS —
    the user is looking at one file and expects to be editing it. Returns
    (kept, dropped_count); the caller reports the count, because a silent
    drop is indistinguishable from a change that was applied.
    """
    if scope_slug is None or not isinstance(raw_ops, list):
        return raw_ops, 0
    kept = [
        op for op in raw_ops
        if not isinstance(op, dict) or op.get("slug") in (None, scope_slug)
    ]
    return kept, len(raw_ops) - len(kept)


def _today(tz_name: Optional[str] = None) -> str:
    now = datetime.now(timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            now = now.astimezone(ZoneInfo(tz_name))
        except Exception:
            pass
    return now.strftime("%A, %B %d, %Y")


async def _today_for(db: AsyncSession, user_id: str) -> str:
    """Today in the USER's zone, on every path.

    "tomorrow at 9" is resolved to an absolute date by the writer, and at
    23:40 in Toronto a UTC clock is already on the next day — so the bullet
    would name the wrong date, permanently, in the one voice the product
    presents as fact. Same resolver `_resolve_day_key` uses, so the change
    line and the prompt can never disagree about which day it is.
    """
    return _today(await ops.resolve_user_tz(db, user_id))


# ── Entry points ──────────────────────────────────────────────────────

async def instruct_file(
    db: AsyncSession,
    user_id: str,
    slug: str,
    instruction: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """"Tell your agent what to change or remove", scoped to one file.

    Raises ValueError when the file does not exist (the route → 404).
    """
    identity = await resolve_user_identity(db, user_id)
    await ops.ensure_system_files(db, user_id)
    await db.commit()
    rows = await ops._all_files(db, user_id)
    target = next((r for r in rows if r.slug == slug), None)
    if target is None:
        raise ValueError(f"memory file {slug!r} not found")

    prompt = build_instruct_prompt(
        instruction=instruction,
        identity=identity,
        index=_index_block(rows),
        bodies=_bodies_block([target]),
        today=await _today_for(db, user_id),
        scope_slug=slug,
    )
    result = await _run_ops(
        db, user_id, prompt, api_key=api_key, identity=identity, scope_slug=slug,
        # The evidence is what the user typed plus the body they are editing —
        # a bullet already in the file is theirs, whatever it resembles.
        source_text=f"{instruction}\n{target.body_md or ''}",
    )
    result["note"] = _note(result)
    logger.info(
        "[curator] instruct file=%s user=%s applied=%d rejected=%d",
        slug, str(user_id)[:8], result["applied"], len(result["rejected"]),
    )
    return result


async def instruct_global(
    db: AsyncSession,
    user_id: str,
    instruction: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """"Tell your agent what to remember" — the curator picks the file."""
    identity = await resolve_user_identity(db, user_id)
    await ops.ensure_system_files(db, user_id)
    await db.commit()
    rows = await ops._all_files(db, user_id)
    bodies = _pick_bodies(rows, instruction, always=ALWAYS_INJECTED_SLUGS)

    prompt = build_instruct_prompt(
        instruction=instruction,
        identity=identity,
        index=_index_block(rows),
        bodies=_bodies_block(bodies),
        today=await _today_for(db, user_id),
    )
    result = await _run_ops(
        db, user_id, prompt, api_key=api_key, identity=identity,
        source_text="\n".join(
            [instruction] + [(r.body_md or "") for r in bodies]
        ),
    )
    result["note"] = _note(result)
    logger.info(
        "[curator] instruct global user=%s applied=%d rejected=%d files=%s",
        str(user_id)[:8], result["applied"], len(result["rejected"]),
        result["changed_files"],
    )
    return result


def _note(result: Dict[str, Any]) -> str:
    """One honest sentence for the client to show under the input box.

    Names the file by its TITLE ("Majid Tajik"), never by its slug. A slug
    is an internal id: no client renders one anywhere else in the product,
    and this string is the only place one ever reached a user.
    """
    n = result["applied"]
    if n:
        titles = result.get("changed_titles") or []
        if not titles:
            titles = [
                title_from_slug(s) for s in (result.get("changed_files") or [])
            ]
        where = f" in {titles[0]}" if len(titles) == 1 else ""
        return f"Saved {n} change{'' if n == 1 else 's'}{where}."
    if result["rejected"]:
        return "Nothing changed — that didn't pass the memory rules."
    return "Nothing to change."




# ══ The turn curator (§2.2) ═══════════════════════════════════════════
#
# One LLM call per non-trivial turn, at most. Three deterministic gates run
# FIRST and cost nothing:
#
#   1. the trivial-turn signal the runner already computed,
#   2. a "no memorable content" pre-gate (too short, pure question, pure
#      one-off command),
#   3. the never-store secret tier, on the RAW turn text.
#
# Gate 3 is the only one that is about safety rather than cost, and it is
# deliberately applied to the whole turn rather than only to the bullets the
# model proposes: a turn that contains a card number has nothing durable in
# it worth the risk of a model paraphrasing the number into a bullet that
# `_secret_problem` happens not to match. `memory_file_ops` still checks
# every bullet — this is the coarse first cut, not the only one.

#: Under this and there is nothing to remember. Two words is a legitimate
#: BULLET ("likes Googoosh"), but a two-word TURN is "ok thanks".
_TURN_MIN_CHARS = 12
_TURN_MIN_WORDS = 4

# A turn that only asks. "what's the weather" states nothing about the user.
# A first-person declarative anywhere disqualifies the whole turn from this
# rule — "I'm allergic to peanuts, what should I eat?" is one sentence
# ending in a question mark and carries a durable fact.
_QUESTION_END_RE = re.compile(r"[?؟]\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟。])\s+")
_ASSERT_RE = re.compile(
    r"\b(?:i'?m|i\s+am|i'?ve|i\s+have|i\s+had|i\s+live|i\s+work|i\s+like|"
    r"i\s+love|i\s+hate|i\s+prefer|i\s+need|i\s+want|i\s+use|i\s+own|"
    r"i\s+study|i\s+play|i\s+moved|i\s+started|i\s+got|i\s+booked|"
    r"i'?ll|i\s+will|we'?re|we\s+are|"
    # A possessive with a copula anywhere after it — "my sister's birthday is
    # March 3". Up to three words in between, because "remind me that my
    # sister's birthday is March 3" is a reminder AND a durable fact, and the
    # one-off rule below must not swallow the second half.
    r"my\s+(?:\S+\s+){0,3}(?:is|are|was|were)\b|"
    r"remember\b|note\s+that\b)",
    re.IGNORECASE,
)

# The one-off command classes from the dispatch's Section 2, as a cheap
# pre-gate. Every one of these is ALSO a prompt rule and an eval fixture —
# this regex only saves the call, it is not the control. It is anchored and
# word-capped precisely so it can never eat a sentence that also states a
# fact ("play Googoosh, she's my favourite singer" is 8 words but carries
# a comma-joined declarative, so `_ASSERT_RE`… does not match it either —
# hence the length cap doing the work).
_ONE_OFF_RE = re.compile(
    r"^\s*(?:hey\s+\w+[,\s]+)?(?:can\s+you\s+|could\s+you\s+|please\s+)?"
    r"(?:play|put\s+on|queue|skip|next|previous|pause|resume|stop|mute|unmute|"
    r"turn\s+(?:it|the\s+\w+)\s+(?:up|down|off|on)|volume|"
    r"wake\s+me|snooze|set\s+(?:a|an)\s+(?:timer|alarm)|remind\s+me|"
    r"open|close|show\s+me|search\s+for|google|look\s+up)\b",
    re.IGNORECASE,
)
_ONE_OFF_MAX_WORDS = 14


def turn_skip_reason(user_text: str) -> Optional[str]:
    """Why this turn needs no writer call, or None to run it.

    Deterministic, no LLM, no DB. Exposed for the eval set's CI lane and for
    `check`-style probes: the pre-gate is the cheapest place a real fact can
    be lost, so it has to be testable on its own.
    """
    text = (user_text or "").strip()
    if len(text) < _TURN_MIN_CHARS or len(text.split()) < _TURN_MIN_WORDS:
        return "too_short"

    from app.services.memory_secrets import sensitive_content_reason

    secret = sensitive_content_reason(text)
    if secret:
        return secret

    asserts = bool(_ASSERT_RE.search(text))
    if not asserts:
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if sentences and all(_QUESTION_END_RE.search(s) for s in sentences):
            return "question_only"
        if (
            len(text.split()) <= _ONE_OFF_MAX_WORDS
            and _ONE_OFF_RE.match(text)
        ):
            return "one_off_request"
    return None


TURN_DURABILITY_RULES = """WHAT IS DURABLE — the only test that matters:
will this still be worth knowing about this person in six weeks?

NEVER store (these are the exact classes that polluted the last version):
- One-off requests. "play Moo Meshki", "پخش کن …", "put on some jazz" — the
  request, the track, and anything a media tool returned about it.
- Reminders, alarms, snoozes, timers, tasks for right now: "wake me 1 minute
  later", "reminder to go to soccer in 2 minutes", "remind me at 5".
- Transient states: "currently hungry", "feeling tired today", "on the train".
- Tool, search or playback OUTCOMES: what played, what failed, what an
  assistant fetched. "You have Gmail messages" is a tool result, not a fact
  about a life. A title the user did not themselves assert is not a fact
  about them.
- The text of a scheduled job, routine or trigger, and any of its parameters.
  A STANDING arrangement may be ONE line in you/profile — "receives a daily
  Gmail briefing at 11:49 AM" — never the job's prompt.
- Internal ids, UUIDs, app ids, slugs, model names, file paths.
- Advice, tips or explanations the assistant gave, in ANY voice. "should
  avoid shellfish at restaurants" is the assistant talking; "allergic to
  shellfish" is the fact underneath it. Store only the fact.
- Hypotheticals and questions about "someone", "anyone", "a friend", "a
  person" — an imagined situation is never a fact about THIS person.
- World knowledge. If it would be true for everybody, it is not a memory.
- Anything the user only ASKED about. A question is not an assertion.

DO store: identity, relationships and who people are to them, health,
commitments with real dates, ongoing projects and their state, standing
preferences and tastes, tools and devices they use, decisions they made.

Empty is the ordinary answer. Most turns change nothing. Reply {"ops": []}
rather than reaching for something."""


def build_turn_prompt(
    *,
    user_text: str,
    assistant_text: str,
    identity: UserIdentity,
    index: str,
    bodies: str,
    today: str,
) -> str:
    """The post-turn task block, over the shared ops contract.

    The two slots are LABELLED asymmetrically on purpose. Root cause #1 of
    the rebuild was that the writer could not tell the user's words from a
    machine's, because ws_chat handed it a rewritten string; the structural
    half of the fix is that `user_text` is now `display_user_message`, and
    this is the other half — the model is told, in the prompt, that only one
    of these two blocks is a source of facts.
    """
    assistant_block = (
        f"WHAT YOU REPLIED (CONTEXT ONLY — never a source of facts):\n"
        f"{assistant_text.strip()}\n\n"
        if (assistant_text or "").strip() else ""
    )
    return (
        f"Today is {today}.\n"
        f"{_identity_block(identity)}\n\n"
        "You maintain this person's memory files. A conversation turn just "
        "finished. Update the files ONLY if it contained something durable.\n\n"
        f"FILE INDEX:\n{index}\n\n"
        f"FILE BODIES:\n{bodies}\n\n"
        "WHAT THE USER SAID (the ONLY source of facts):\n"
        f"{(user_text or '').strip()}\n\n"
        f"{assistant_block}"
        "Facts may come only from what the USER said or confirmed. Anything "
        "that appears only in your reply is yours, not theirs.\n\n"
        f"{TURN_DURABILITY_RULES}\n\n"
        f"{OPS_CONTRACT}"
    )


async def curate_turn(
    db: AsyncSession,
    user_id: str,
    *,
    user_text: str,
    assistant_text: str = "",
    channel: str = "app",
    api_key: Optional[str] = None,
    query_was_trivial: bool = False,
) -> Dict[str, Any]:
    """Post-turn writing into memory files — the replacement for the
    sentence extractor, the active-task detector, the relationship mirror
    and the voice tunnel, all at once (v3 §2.1–§2.3).

    `user_text` MUST be `display_user_message`, the CLEAN text. The
    `user_message` ws_chat hands the runner carries scraped YouTube titles,
    Chrome page context and reply quotes; every round-8 provenance rule
    measured overlap against that string, so the injection disarmed them
    all. Passing `_agent_text` here re-opens root cause #1 of the rebuild.

    Returns the `_run_ops` result plus `skipped` — the pre-gate's reason, or
    None when the model was actually called. Never raises for a gate; a
    genuine failure (no key, DB down, bad JSON) propagates so the caller can
    park the turn in the outbox.
    """
    text = (user_text or "").strip()

    # 1. The runner already classified this turn. Reuse the signal rather
    #    than re-deriving it: `is_trivial_query` is cheap but the runner's
    #    answer is the one the rest of the turn was built on, and two
    #    classifiers that can disagree are a bug waiting for a phrasing.
    if query_was_trivial:
        return {"applied": 0, "rejected": [], "changed_files": [], "skipped": "trivial"}

    # 2/3. Cheap content pre-gate + the never-store secret tier.
    reason = turn_skip_reason(text)
    if reason:
        logger.info(
            "[curator] turn user=%s SKIPPED (%s)", str(user_id)[:8], reason,
        )
        return {"applied": 0, "rejected": [], "changed_files": [], "skipped": reason}

    identity = await resolve_user_identity(db, user_id)
    await ops.ensure_system_files(db, user_id)
    today = await _today_for(db, user_id)
    # Release the pooled connection before the model round-trip. This runs
    # fire-and-forget after the reply on a path that dies routinely (a voice
    # caller hanging up cancels the parent 1.5 s later); a death mid-call
    # used to leave the connection checked out forever and degrade the pool.
    # Same defect #407 fixed in day_summarizer.
    await db.commit()

    rows = await ops._all_files(db, user_id)
    bodies = _pick_bodies(rows, text, always=ALWAYS_INJECTED_SLUGS)
    prompt = build_turn_prompt(
        user_text=text,
        assistant_text=assistant_text or "",
        identity=identity,
        index=_index_block(rows),
        bodies=_bodies_block(bodies),
        today=today,
    )
    result = await _run_ops(
        db, user_id, prompt, api_key=api_key, identity=identity,
        # The turn, plus the bodies already on file: a fact the corpus already
        # holds is the user's, and a rewrite must be free to restate it.
        source_text="\n".join(
            [text, assistant_text or ""] + [(r.body_md or "") for r in bodies]
        ),
    )
    result["skipped"] = None
    logger.info(
        "[curator] turn user=%s channel=%s ops=%d rejected=%d files=%s",
        str(user_id)[:8], channel, result["applied"], len(result["rejected"]),
        result["changed_files"],
    )
    return result


# ══ Migration mode (§7, WS-5) ═════════════════════════════════════════
#
# The one mode the curator did not have. It is NOT a fork of the writer:
# same ops contract, same durability rules, same deterministic validator,
# same apply. Three things differ, and only these three:
#
#   1. the SOURCE is a batch of round-8 `memories` rows rather than a turn,
#   2. the model must ACCOUNT for every row it was shown — a `dispositions`
#      array keyed by the handle each row was given — because the migration
#      report has to answer "where did this specific memory go?" and
#      "why was this dropped?" with something better than silence,
#   3. the per-op change lines are suppressed (see `apply_ops`); the
#      migration files one line per file it filled.
#
# A row the model does not mention is NOT quietly dropped: the caller marks
# it `unaccounted`, which reads as a defect in the report rather than as a
# decision.

MIGRATION_DISPOSITION_CONTRACT = """Put a "refs" key on every add and every
rewrite naming the entries that bullet came from:

  {"op":"add","slug":"you/profile","bullet":"...","change":"...","refs":["L2","L5"]}

Also reply with a "dispositions" array accounting for EVERY numbered entry
you were shown, one object each:

  {"ref":"L3","verdict":"kept|merged|rewritten|dropped","slug":"you/profile","reason":"..."}

- kept      — carried over essentially as written (name the file in `slug`)
- merged    — folded into a bullet that already existed or into another
              entry from this batch (name the file in `slug`)
- rewritten — the durable fact underneath it was kept, in the new voice,
              with the junk around it removed (name the file in `slug`)
- dropped   — nothing durable in it. `slug` is null and `reason` must say
              WHICH rule ("a one-off play request", "what a media tool
              returned", "a reminder", "channel plumbing, not a fact about
              a life", "a single timestamped event, not a standing fact").

`reason` is one short clause, always. "not durable" is not a reason.
Every ref you were shown appears exactly once."""


MIGRATION_TASK = """These are this person's memories as an OLDER system
recorded them. Rewrite them into the memory files.

That system had no durability test and no notion of whose facts these are,
so the corpus below contains real facts mixed with tool output, media
titles, one-off requests and plumbing. Your job is not to move them — it is
to decide what a person would still want known about them, and write only
that, in the file voice, merged rather than accumulated.

- The date beside each entry is when it was RECORDED, not a fact about the
  person. Use it only to resolve a relative date inside the entry.
- The old file name beside each entry is a HINT about where it lived, not
  where it belongs. Those files ("Knowledge", "Preferences", "Working on",
  "People / User") are exactly the junk drawers this rewrite retires.
- An entry filed under the OWNER as if they were a third party is a fact
  about the owner. It goes in you/profile, never in a people/ file.
- Two entries saying the same thing are ONE bullet. Prefer `rewrite` on a
  bullet that already exists over `add`.
- An entry that is only a record of one moment ("logged their sleep at 1:41
  AM on April 22") is not a standing fact. Drop it.
- Say nothing about which channel or app a message arrived on. That is
  plumbing, not a life."""


def build_migration_prompt(
    *,
    entries: str,
    identity: UserIdentity,
    index: str,
    bodies: str,
    today: str,
    extra: str = "",
) -> str:
    """`extra` is the SECOND-ROUND block (WS-5): the validator's own words
    about entries whose ops were refused, plus the instruction to re-express
    the fact in an allowed shape. Empty on the first round."""
    return (
        f"Today is {today}.\n"
        f"{_identity_block(identity)}\n\n"
        "You maintain this person's memory files.\n\n"
        f"FILE INDEX:\n{index}\n\n"
        f"FILE BODIES:\n{bodies}\n\n"
        f"{MIGRATION_TASK}\n\n"
        f"THE OLD ENTRIES:\n{entries}\n\n"
        + (f"{extra}\n\n" if extra else "")
        + f"{TURN_DURABILITY_RULES}\n\n"
        f"{OPS_CONTRACT}\n"
        f"{MIGRATION_DISPOSITION_CONTRACT}\n"
    )


async def curate_migration_batch(
    db: AsyncSession,
    user_id: str,
    *,
    entries: str,
    existing: Sequence[Any],
    identity: UserIdentity,
    today: str,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    extra_instructions: Optional[str] = None,
    allow_retry: bool = True,
) -> Dict[str, Any]:
    """One batch of legacy rows → validated ops (+ the model's accounting).

    `existing` is the file set the batch is written against. On a real run
    that is `ops._all_files(db, user_id)`; on a dry run the caller threads
    its own SIMULATED files through, so batch 2 sees what batch 1 would
    have written and the report's "after" is the state the real run would
    reach rather than batch 1's state repeated.

    `extra_instructions` + `allow_retry=False` are the migration's SECOND
    ROUND over the rows whose ops a partial refusal orphaned. The caller
    owns that orchestration (`memory_v3_migration._retry_orphans`) because
    only it knows which source row ended up with nothing; this function
    stays a single-round primitive, and `allow_retry=False` is what keeps
    "exactly one extra round per batch" true rather than approximately
    true.

    Never commits on a dry run. Never writes a per-op change line: the
    migration owns its own single line per file (§7).
    """
    llm_service = _llm(api_key)
    prompt = build_migration_prompt(
        entries=entries,
        identity=identity,
        index=_index_block(existing),
        bodies=_bodies_block(existing),
        today=today,
        extra=extra_instructions or "",
    )
    # The evidence for a migration batch is the rows being re-curated plus the
    # bodies already written by earlier batches — nothing else.
    evidence = entries + "\n" + "\n".join(
        (getattr(f, "body_md", "") or "") for f in existing
    )
    reply = await propose_plan(llm_service, prompt)
    raw_ops, leak_complaints = _refuse_leaks(reply.get("ops", []), evidence)
    plan = ops.validate_ops(raw_ops, existing, identity=identity)
    plan.complaints.extend(leak_complaints)

    if allow_retry and plan.complaints and not plan.accepted and raw_ops:
        # Same single retry the conversational paths take, with the
        # validator's own words. A migration batch is the one place where
        # losing the whole proposal loses a slice of the user's history
        # rather than one turn's worth, so it is worth the second call.
        retry = (
            prompt
            + "\n\nYour previous reply was rejected for these reasons — fix "
              "them and reply again:\n- " + "\n- ".join(plan.complaints)
        )
        reply = await propose_plan(llm_service, retry)
        raw_ops, retry_leaks = _refuse_leaks(reply.get("ops", []), evidence)
        plan = ops.validate_ops(raw_ops, existing, identity=identity)
        plan.complaints.extend(retry_leaks)

    applied = 0
    changed_files: List[str] = []
    if not dry_run:
        result = await ops.apply_ops(db, user_id, plan, change_lines=False)
        applied = result["applied"]
        changed_files = result["changed_files"]
    else:
        applied = len(plan.accepted)
        changed_files = sorted({op["slug"] for op in plan.accepted})

    dispositions = reply.get("dispositions")
    return {
        "applied": applied,
        "changed_files": changed_files,
        "rejected": plan.complaints,
        "dispositions": dispositions if isinstance(dispositions, list) else [],
        # The RAW ops, because `validate_ops` keeps only the keys it knows
        # and `refs` is not one of them. The caller matches accepted ops
        # back to their refs to work out which SOURCE ROW actually landed —
        # slug-level evidence is not enough when one file receives bullets
        # from several entries and only some of them survive the lint.
        "raw_ops": raw_ops if isinstance(raw_ops, list) else [],
        "plan": plan,
    }
