"""The v2 run pipeline — fire → evaluate → steps → writes → record.

Round 28. v1 runs keep executor.py untouched; this module owns specs
with `version: 2`. Same rails, same primitives:

  - runs ARE BuildJobs (`job_type='automation_run'`), idempotency keys
    unchanged; the job's steps_json is dynamic: `evaluate`, one entry
    per spec step, `record`.
  - events: insert-or-skip on UNIQUE (automation_id, dedupe_key), with
    the v2 key namespaced per source: "<source_id>:<value>".
  - read steps run inline through the platform dispatch RPC (non-
    mutating, no grant — the dispatcher still fails closed on unknown
    tools); `collect` folds their items into {{steps.<id>.text}} /
    {{steps.<id>.count}}.
  - write steps stage to automation_outbox with keys
    "<prefix>:w<n>" (v1's single write is w0 — the same scheme), then
    flush after the normal undo window. One grant per write step.
  - mail rail unchanged: send tools are refused outright.
  - memory: the run reads its namespace once at fire time
    ({{memory.<key>}}) and writes it back after the run, in its own
    session (memory.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError

from app.db.models import (
    Automation, AutomationEvent, AutomationOutbox, BuildJob,
    AUTOMATION_OUTBOX_UNDO_WINDOW_S, AUTOMATION_RUN_CAP_S,
)
from app.agent.job_runner import JobRunner, TaskSpec
from app.agent import job_steps
from .spec import (
    filter_compile as _filter_compile,
    filter_options as _filter_options,
    filter_tools as _filter_tools,
    focus_render_ctx as _focus_render_ctx,
    render_value, render_with_ctx, resolve_path,
)
from .spec_v2 import ValidatedSpecV2, ValidatedSource, ValidatedStep
from .contents import container_of
from .executor import _FORBIDDEN_TOOLS, _finalize_job, _record_health
from .session import on_run_created
from . import agent_step as _agent_step
from . import memory as engine_memory
from . import registry as reg
from app.connectors.textclean import unescape_once

logger = logging.getLogger(__name__)


async def merge_job_config(db, job_id: str, **extras) -> None:
    """Merge keys into a run job's config_json (read-modify-write,
    commits). The one seam for run-scoped extras — used here for
    steps_partial; sibling rounds stamp their own keys through it
    rather than growing second implementations."""
    from app.db.models import BuildJob
    job = await db.get(BuildJob, job_id)
    if job is None:
        return
    cfg = dict(job.config_json or {})
    cfg.update(extras)
    job.config_json = cfg
    await db.commit()


# ── Job-steps plumbing (dynamic step list) ───────────────────────────


def _step_order(vspec: ValidatedSpecV2) -> list[str]:
    return ["evaluate", *[st.id for st in vspec.steps], "record"]


def _new_steps_v2(vspec: ValidatedSpecV2) -> str:
    """Humanized labels at mint (R29): steps_json is the shared
    substrate (runs API, job cards, web) — spec steps wear their
    tool's verb + connector brand, engine phases the orb (brand
    None). The verb dictionary is the only composer."""
    from app.services.automation_verbs import step_verb

    by_id = {st.id: st for st in vspec.steps}
    now = datetime.utcnow()
    steps = []
    for s in _step_order(vspec):
        st = by_id.get(s)
        if st is not None and st.kind == "agent":
            # R38: the agent's own work, branded as the orb — the same
            # treatment the engine phases get, because that is what it
            # is. It counts as a step everywhere a step is counted.
            v = step_verb(None, None, phase="think")
        elif st is not None:
            v = step_verb(st.tool, st.connector_id)
        else:
            v = step_verb(None, None, phase=s)
        steps.append({
            "id": s, "type": "generic", "label": v["label"],
            "brand": v["brand"],
            "status": "pending", "started_at": None, "completed_at": None,
        })
    return job_steps.dump_steps(job_steps.open_first_step(steps, now))


async def _advance_v2(db, job_id: str, vspec: ValidatedSpecV2,
                      done_step: str,
                      count: Optional[int] = None) -> None:
    from app.db.models import BuildJob
    job = await db.get(BuildJob, job_id)
    if job is None:
        return
    order = _step_order(vspec)
    if done_step not in order:
        return
    steps = job_steps.parse_steps(job.steps_json)
    if count is not None:
        # A collected read's count rides the step dict (R29): the runs
        # API's done-form verbs and the last-outcome sentence both read
        # it back — steps_json is the one substrate.
        for s in steps:
            if s.get("id") == done_step:
                s["count"] = count
                break
    job.steps_json = job_steps.dump_steps(
        job_steps.advance_steps(
            steps, order.index(done_step), datetime.utcnow(),
            fallback_start=job.created_at,
        )
    )
    await db.commit()


# ── Event intake (per-source dedupe namespace) ───────────────────────


async def ingest_items_v2(
    db,
    automation: Automation,
    source: ValidatedSource,
    items: list[dict],
    *,
    baseline: bool = False,
) -> list[AutomationEvent]:
    """Insert-or-skip each observed item, dedupe-keyed as
    "<source_id>:<value>" so two sources can never collide in the
    per-automation UNIQUE gate. Payloads are trimmed to the declared
    event fields plus `_source`.

    `baseline` is the first tick of a source that has never observed
    anything (R42, B5): the rows are recorded as `discarded` — seen,
    deliberately not run — and NOTHING comes back as fresh. Without it
    the dedupe table is empty on the first poll, so every item a
    provider hands back is "new" and a 50-item window mints 50 runs at
    once. The status is written at INSERT, never as a second UPDATE: a
    row that lands as `new` and dies before the flip is a run this
    engine still owes.

    The caller decides, because only the caller knows the leg: a poll
    window is history (the last 25-50 items the provider holds), while
    a pushed event arrived because it just happened, and baselining
    THAT drops real mail on the day the automation is armed.
    """
    ev_spec = source.event_spec or {}
    fields: dict[str, str] = dict(ev_spec.get("fields") or {})
    dedupe_field = source.dedupe_key_field or ev_spec.get("dedupe_field") or "id"

    fresh: list[AutomationEvent] = []
    for item in items:
        key = resolve_path(item, fields.get(dedupe_field, dedupe_field))
        if key is None:
            key = item.get(dedupe_field)
        if key is None:
            continue
        payload = {
            name: resolve_path(item, path) for name, path in fields.items()
        }
        payload.setdefault(dedupe_field, key)
        payload["_source"] = source.id
        event = AutomationEvent(
            automation_id=automation.id,
            user_id=automation.user_id,
            dedupe_key=f"{source.id}:{key}"[:255],
            payload_json=json.dumps(payload, default=str)[:8000],
            status="discarded" if baseline else "new",
        )
        try:
            async with db.begin_nested():
                db.add(event)
                await db.flush()
            if not baseline:
                fresh.append(event)
        except IntegrityError:
            pass
    await db.commit()
    return fresh


def _like_prefix(value: str) -> str:
    """Escape a literal for a LIKE prefix match (backslash escape)."""
    return (value.replace("\\", "\\\\")
            .replace("%", "\\%").replace("_", "\\_"))


async def _source_has_history(db, automation_id: str, source_id: str) -> bool:
    """Has this source ever ingested an item?

    The honest "has it polled before" marker, and it is the dedupe
    namespace itself — the same "<source_id>:<value>" gate the intake
    claims — so it needs no column and no second store, it is per
    SOURCE (a second lane armed later baselines on its own first tick),
    and it survives a restart the way the runs it guards do.

    One hole, and it is the cheap side of the trade: a first poll that
    observes NOTHING leaves the source unbaselined, so the next
    non-empty poll baselines items that were genuinely new. That costs
    one batch, recorded as `discarded` rows an audit can read. Getting
    it wrong the other way costs 50 runs in one tick.
    """
    from sqlalchemy import select as _select
    row = await db.execute(
        _select(AutomationEvent.id)
        .where(AutomationEvent.automation_id == automation_id)
        .where(AutomationEvent.dedupe_key.like(
            f"{_like_prefix(source_id)}:%", escape="\\"))
        .limit(1)
    )
    return row.scalar() is not None


def _passes_filter_v2(source: ValidatedSource, payload: dict,
                      variables: dict,
                      facts_ctx: Optional[dict] = None) -> bool:
    """v1 filter semantics per source, with {{var.*}} rendered in the
    needles so templates can parameterize filters.

    R29: a `{{facts.<category>}}` needle matches against the fact
    ledger (facts_context, the "memory-filtered" leg) and is
    intercepted BEFORE render_value — the var renderer would blank the
    unknown template and turn the needle into a match-nothing literal
    instead of a ledger lookup."""
    from .facts_context import facts_needle_category, needle_matches

    ctx = {"var": variables or {}}
    for fld, needles in (source.filter_rules or {}).items():
        if not needles:
            continue
        if not isinstance(needles, list):
            needles = [needles]
        value = str(payload.get(fld) or "").lower()
        ok = False
        for n in needles:
            if facts_needle_category(n) is not None:
                if needle_matches(n, value, facts_ctx):
                    ok = True
                    break
                continue
            rendered = str(render_value(str(n), ctx)).lower()
            if rendered and rendered in value:
                ok = True
                break
        if not ok:
            return False
    return True


# ── Read-step execution ──────────────────────────────────────────────


def _collect_result(step: ValidatedStep, content: dict,
                    variables: dict) -> dict:
    """Fold a read step's JSON result into {text, count, ok} per the
    step's collect declaration."""
    collect = step.collect
    if not collect:
        return {"ok": True, "text": "", "count": 0}
    items = resolve_path(content, collect["items_path"])
    if not isinstance(items, list):
        items = []
    count = len(items)
    limit = int(collect.get("limit") or 10)
    fmt = collect.get("format")
    # R38 — the ingestion seam. Provider strings arrive carrying HTML
    # ("<b>Google Workspace product notifications.</b>" reached a
    # founder's screen, and Gmail snippets ship entities routinely).
    # Sanitized HERE because `lines` feeds BOTH the ledger's item
    # titles and, via {{steps.<id>.text}}, the text a write step posts
    # to Slack or folds into a draft. Known-tag allowlist + entity
    # unescape; an email's angle brackets survive (ledger.strip_html).
    from .ledger import strip_html
    lines: list[str] = []
    raw_fields: list[dict] = []
    if fmt:
        for item in items[:limit]:
            # ONE decode, at ingestion (R43). These raw values are the OTHER
            # derivation of the same provider text: the rendered line below
            # becomes the item's `title`, and these become its `head`/`lede`
            # through `ledger.item_slots`. The line was unescaped here and the
            # fields were not, so `mint_item_ids` had to unescape a second
            # time to reach them — which decoded the line TWICE and turned a
            # sender's literal `&amp;` into `&`. Cleaning both here is what
            # lets every layer downstream strip tags only.
            item_fields = {
                name: (unescape_once(v) if isinstance(v, str) else v)
                for name, v in (
                    (name, resolve_path(item, path))
                    for name, path in (collect.get("fields") or {}).items()
                )
            }
            raw_fields.append(item_fields)
            lines.append(strip_html(str(render_value(
                fmt, {"item": item_fields, "var": variables or {}},
            ))))
    text = (collect.get("join") or "\n").join(lines)
    if count == 0:
        text = collect.get("empty_text") or ""
    # `lines`/`raw_fields` are R30 ledger inputs (mechanical item titles
    # + the narrator's raw material); templates keep reading only
    # text/count via resolve_path — extra keys are inert there.
    return {"ok": True, "text": text, "count": count,
            "lines": lines, "raw_fields": raw_fields}


def _skipped_result(step: ValidatedStep, *, silent: bool = True) -> dict:
    """The placeholder a failed read leaves behind in the context.

    `silent` is `on_error: "skip"` — the mode whose whole point is that an
    optional source says nothing when it is down (the Teams provider_down
    precedent). It keeps its `empty_text`.

    `on_error: "continue"` is the mode that owes the user a named account
    and a reason, and its section must SAY the read did not happen. Round
    33, item 4: both modes used to return `empty_text`, so a Gmail read
    that failed was interpolated into the brief as "Gmail inbox is clear."
    and posted to Slack as a fact about the user's morning. The flagship
    template asked for `skip` on all five reads, which is what made the
    silent mode the one every user met; it asks for `continue` now.
    """
    empty = (step.collect or {}).get("empty_text") or ""
    if silent:
        return {"ok": False, "text": empty, "count": 0}
    name = _display_name(step.connector_id or "") or (step.connector_id or "it")
    return {
        "ok": False,
        "failed": True,
        "text": f"Could not read {name}.",
        "count": 0,
    }


# tool → (the param that names its target, the pin kinds that can fill
# it, in order of preference). A read in this table CANNOT run without a
# target, so an empty one is a hole a pin may fill — never a narrowing.
_PIN_TARGET_PARAM: dict[str, tuple[str, tuple[str, ...]]] = {
    "slack__read_messages": ("channel", ("channel", "thread")),
    "teams__read_chat_messages": ("chat_id", ("thread", "channel")),
}

# github__list_issues takes its target as two params, so it is filled
# separately — the kinds, in preference order, are here beside the rest.
_PIN_TARGET_GITHUB_KINDS: tuple[str, ...] = ("repo", "ticket")


def _apply_focus_scope(
    connector_id: str, tool: str, params: dict, pins: list,
) -> dict:
    """Fill a read's EMPTY target from a pin. Pins rank; they never
    filter (R42, founder P6 — stated twice).

    R39 read the pins as a scope and narrowed the provider call itself:
    a pinned person became `from:dana@x.com` on the Gmail query, a
    pinned project became `project in ("ENG") AND (<the step's own
    jql>)`, and a pinned Slack channel OVERRODE the channel the step
    already named. Three separate wrongs, one mistake. Material the
    user did not pin was never FETCHED, so no ranking step could see
    it and nothing downstream could put the pinned item first — which
    is the whole thing a pin is for. A second pin was silently dropped
    (only `[0]` was used). And every shipped Jira template's JQL ends
    in `ORDER BY`, which JQL forbids inside parentheses, so the
    composed clause 400'd, `on_error: continue` swallowed it, and the
    brief blamed a healthy board.

    So the only case left is the honest one: a tool that REQUIRES a
    target it does not have. A query language that is already broad
    (gmail/outlook `query`, jira `jql`) is left exactly as the spec
    wrote it; the pins reach the RANKING step instead, through
    `focus_render_ctx` — `agent_step.build_prompt` renders them as
    `starts_at` (labels + the user's notes) and the narrator's record
    carries them under `focus`.

    A step that genuinely wants a scoped read can already say so in the
    spec: `{{focus.<connector_id>.first.id}}` renders the pinned target
    into any param. That is authored intent, which a pin is not.

    Pure and total: no pins, unknown tool, a target the spec already
    set, a pin that names no place, or malformed params → the params
    come back untouched.

    A pin id is NOT a target id. R42 gave a preview ROW its own pin,
    whose id is `<container id>#<row id>` and whose kind is the one a
    CONTAINER pin also uses (a Slack message in a channel and a Teams
    message in a chat are both `thread`) — so a run aimed at the
    channel id `C0ALL#1712345.678`, which exists nowhere, and a GitHub
    ticket pin `acme/api#42` would have split into owner `acme` and
    repo `api#42`. `contents.container_of` is the one place that format
    is taken apart, and every fill below goes through it.
    """
    if not pins or not isinstance(pins, list) or not isinstance(params, dict):
        return params
    by_kind: dict[str, list] = {}
    for p in pins:
        if isinstance(p, dict) and p.get("id"):
            by_kind.setdefault(str(p.get("kind") or ""), []).append(p)

    def _targets(kinds: tuple[str, ...]):
        """The container id of each pin of those kinds, in preference
        order. A pin whose container cannot be resolved is skipped —
        never guessed at."""
        for kind in kinds:
            for p in by_kind.get(kind, []):
                got = container_of(connector_id, p)
                if got is None or not got[0]:
                    continue
                yield got[0]

    if tool == "github__list_issues":
        # Two params, one pin: a repo pin is "owner/repo", a ticket pin
        # sits INSIDE one, and either half already set means the spec
        # named the repository.
        if params.get("owner") or params.get("repo"):
            return params
        for target in _targets(_PIN_TARGET_GITHUB_KINDS):
            if "/" in target:
                out = dict(params)
                out["owner"], out["repo"] = target.split("/", 1)
                return out
        return params

    entry = _PIN_TARGET_PARAM.get(tool)
    if entry is None:
        return params
    field, kinds = entry
    if params.get(field):
        return params
    for target in _targets(kinds):
        out = dict(params)
        out[field] = target
        return out
    return params


# ── What each account may OPEN (R43 §2.2) ────────────────────────────
#
# The third and last params pass, and the one that was missing: R43
# shipped a picker ("what it may open here"), a writer (`workflow.
# set_sources`), a validator (`spec_v2.validate_account_sources`) and a
# canvas chip that all agreed about a set of ids NOTHING then read. The
# sheet said "Nothing picked — I will skip Gmail on the next run" and
# the next run read Gmail exactly as before. This is the consumer.
#
# It is NOT `_apply_focus_scope`, and the distinction is the same one
# that separates a pin from a filter. A pin RANKS — it may never stop
# material being fetched, so it only ever fills a target the spec left
# empty. A picked source is the user answering "which places", which is
# a narrowing they asked for out loud, so it composes into the call and
# OVERRIDES a target a pin filled.
#
# Two shapes, because two kinds of read exist and only one of them can
# express a SET:
#
#   query languages (gmail's `query`, jira's `jql`) take an OR-group
#   ANDed into whatever the step already asked for, so any number of
#   picks is exact.
#
#   a target param (slack's `channel`, teams' `chat_id`, outlook's
#   `folder`, calendar's `calendar_id`, github's `owner`/`repo`) names
#   ONE place per call, so N picks are N calls, merged at the step's
#   own `collect.items_path`. Capped at `_SOURCE_FANOUT_MAX`, and the
#   writer refuses a pick past that cap (`source_scope_max`) — a set
#   the run cannot read is the picker that writes nowhere again, one
#   layer down.
#
# A connector absent from the table cannot narrow by place at all, and
# `workflow.set_sources` refuses a pick for it rather than storing one.
# Under-offering is the correct error here, exactly as it is for
# `spec.CONNECTOR_FILTERS`: notion's read is `notion__search`, whose
# `query` is free text and cannot be aimed at a page id; drive, docs,
# sheets and linkedin have no read this vocabulary reaches.

#: How many places one account's read may be fanned out to in a run.
#: Four, the same bound the per-account route cap uses: it is the
#: number of extra provider calls a single step may cost, and it is
#: also the cap the WRITER enforces, so "picked" and "read" cannot
#: differ.
_SOURCE_FANOUT_MAX = 4

#: Gmail's own operators for the three system labels the enumeration
#: ships (`contents._GMAIL_LABELS`, which names them by their API ids).
#: `label:INBOX` is NOT one of them — `label:` takes a label NAME, and a
#: term that matches nothing would empty the brief in silence.
_GMAIL_SYSTEM_TERMS = {
    "inbox": "in:inbox",
    "important": "is:important",
    "starred": "is:starred",
}


def _gmail_source_term(source_id: str) -> str:
    """One picked Gmail source → the query term that selects it, or `""`
    when nothing does.

    Two id shapes reach here and both are real: the live enumeration's
    system-label ids (`INBOX`), and the contract's `label:<name>` form
    for a user label — which is already a Gmail term, quoted here
    because a label name may carry spaces.
    """
    sid = str(source_id or "").strip()
    if not sid:
        return ""
    plain = _GMAIL_SYSTEM_TERMS.get(sid.lower())
    if plain:
        return plain
    if sid.lower().startswith("label:"):
        name = sid.split(":", 1)[1].strip()
        # A quote inside a label name has no escape in Gmail's grammar,
        # so such a label is simply not scopable rather than guessed at.
        if name and '"' not in name:
            return f'label:"{name}"'
    return ""

#: A Jira project key, and nothing that could close the quote it is
#: about to be wrapped in.
_JIRA_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")

_SOURCE_SCOPE: dict[str, dict] = {
    "gmail": {"kind": "query_or", "term": _gmail_source_term,
              "tools": {"gmail__list_messages": "query",
                        "gmail__search_threads": "query"}},
    "jira": {"kind": "jql_in", "field": "project",
             "tools": {"jira__search_issues": "jql"}},
    "slack": {"kind": "target",
              "tools": {"slack__read_messages": ("channel",)}},
    "teams": {"kind": "target",
              "tools": {"teams__read_chat_messages": ("chat_id",)}},
    "outlook": {"kind": "target",
                "tools": {"outlook__list_messages": ("folder",)}},
    "calendar": {"kind": "target",
                 "tools": {"calendar__list_events": ("calendar_id",)}},
    # Two params, one id: a repo source is "owner/repo", which is what
    # `_apply_focus_scope` already splits a repo PIN into.
    "github": {"kind": "target",
               "tools": {"github__list_issues": ("owner", "repo")}},
}


def source_scope_kind(connector_id: str) -> str:
    """How this connector's reads narrow to a picked place, or `""` for
    "they cannot". Public because the WRITER has to ask the same
    question the run answers, and a second copy of this table is how
    the two drift."""
    entry = _SOURCE_SCOPE.get(str(connector_id or ""))
    return str(entry["kind"]) if entry else ""


def source_scope_max(connector_id: str) -> int:
    """How many places this account's read can really open. 0 when the
    connector has no scopable read at all."""
    kind = source_scope_kind(connector_id)
    if not kind:
        return 0
    if kind == "target":
        return _SOURCE_FANOUT_MAX
    from .spec_v2 import MAX_ACCOUNT_SOURCES
    return MAX_ACCOUNT_SOURCES


def source_scope_supports(connector_id: str, source_id: str) -> bool:
    """Can a run really aim a read at THIS id?

    Per-id rather than per-connector because Gmail's answer is per-id:
    the three system labels compile to real operators and a label id
    compiles to nothing.
    """
    entry = _SOURCE_SCOPE.get(str(connector_id or ""))
    sid = str(source_id or "").strip()
    if entry is None or not sid:
        return False
    kind = entry["kind"]
    if kind == "query_or":
        return bool(entry["term"](sid))
    if kind == "jql_in":
        return _JIRA_KEY_RE.match(sid) is not None
    # A target is whatever the provider calls a place; github's is the
    # only one with a shape, because it is two params.
    if str(connector_id) == "github":
        return "/" in sid and all(p.strip() for p in sid.split("/", 1))
    return True


def _fill_target(params: dict, fields: tuple, source_id: str):
    """One params dict aimed at one place, or None when the id does not
    name one this tool can take."""
    out = dict(params)
    if len(fields) == 1:
        out[fields[0]] = source_id
        return out
    if len(fields) == 2 and "/" in source_id:
        owner, repo = source_id.split("/", 1)
        if owner.strip() and repo.strip():
            out[fields[0]], out[fields[1]] = owner.strip(), repo.strip()
            return out
    return None


def _apply_source_scope(
    connector_id: str, tool: str, params: dict, sources: list,
) -> list[dict]:
    """The picked places, compiled into the calls this step will make.

    Returns a LIST because a target-shaped read needs one call per
    place. Pure and total, like the two passes above it: no picks, a
    connector or a tool the table does not reach, malformed params, or
    ids that name nothing this tool can take → `[params]`, untouched.
    """
    if not isinstance(params, dict) or not isinstance(sources, list):
        return [params]
    entry = _SOURCE_SCOPE.get(str(connector_id or ""))
    if entry is None:
        return [params]
    picked = [s.strip() for s in sources
              if isinstance(s, str) and s.strip()
              and source_scope_supports(connector_id, s)]
    if not picked:
        return [params]
    field = (entry.get("tools") or {}).get(tool)
    if not field:
        return [params]

    kind = entry["kind"]
    if kind == "query_or":
        terms = [entry["term"](sid) for sid in picked]
        # Parenthesised only when there is a choice inside it: a single
        # term needs no group, and `_and_terms` already knows how to AND
        # one onto a query that has its own `OR`.
        group = terms[0] if len(terms) == 1 else f"({' OR '.join(terms)})"
        out = dict(params)
        out[field] = _and_terms(out.get(field), group)
        return [out]

    if kind == "jql_in":
        vals = ", ".join(f'"{sid}"' for sid in picked)
        out = dict(params)
        out[field] = _and_jql(out.get(field),
                              [f'{entry["field"]} in ({vals})'])
        return [out]

    sets = [p for p in (_fill_target(params, field, sid)
                        for sid in picked[:_SOURCE_FANOUT_MAX])
            if p is not None]
    return sets or [params]


def _merge_read_contents(contents: list, items_path: str) -> dict:
    """Fold one account's fanned-out reads back into one result.

    Only the step's own `collect.items_path` is merged; every other key
    comes from the first call. A provider's `next_cursor` or
    `result_size` describes ONE of the calls and there is no honest way
    to add them up, while `items_path` is the only thing
    `_collect_result` and `_apply_read_drops` read.
    """
    kept = [c for c in contents if isinstance(c, dict)]
    if not kept:
        return {}
    parts = [p for p in str(items_path or "").split(".") if p]
    if len(kept) == 1 or not parts:
        return kept[0]
    items: list = []
    for c in kept:
        got = resolve_path(c, items_path)
        if isinstance(got, list):
            items.extend(got)
    merged = dict(kept[0])
    cur = merged
    for p in parts[:-1]:
        nxt = cur.get(p)
        cur[p] = dict(nxt) if isinstance(nxt, dict) else {}
        cur = cur[p]
    cur[parts[-1]] = items
    return merged


# ── Per-account read filters (R42, design §5.2) ──────────────────────
#
# The mirror image of `_apply_focus_scope` above, and the distinction
# is the whole point: a PIN ranks, so it must never stop material being
# fetched; a FILTER is the user explicitly asking for less, so it DOES
# compose into the provider call. `spec.CONNECTOR_FILTERS` owns which
# filters exist and which tools each one composes into — this file owns
# only HOW.
#
# Two composition rules, and they differ because the substrates differ:
#
#   query languages (gmail's `query`, jira's `jql`, slack search's
#   `query`) AND their terms, so a filter APPENDS and the step's own
#   query still applies — see `_already_narrowed` for the one case it
#   does not repeat itself.
#
#   PARAMS (outlook `is_read`/`since`, slack `oldest`) are set. A bound
#   only ever moves later — a filter may not widen a read the spec
#   already narrowed — while read state REPLACES, because it is a
#   partition rather than a bound: leaving `is_read: true` under a lit
#   "Unread only" is the chip that lies.
#
# Pure and total, like every other params pass here: unknown connector,
# unknown tool, malformed params, or a filter that needs a clock the
# run did not supply → the params come back untouched.


def _split_jql_order_by(jql: str) -> tuple[str, str]:
    """`("<where>", "ORDER BY …")` — the trailing sort, kept trailing.

    JQL forbids `ORDER BY` inside parentheses, so a composed clause
    must go into the WHERE half and the sort must stay at the very end
    (R42's B2 finding, where a wrapped query 400'd and `on_error:
    continue` blamed a healthy board). The scan skips quoted regions so
    an issue searched for the words "order by" is not mistaken for the
    sort.
    """
    text = str(jql or "")
    low = text.lower()
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif low.startswith("order by", i) and (i == 0 or not text[i - 1].isalnum()):
            return text[:i].strip(), text[i:].strip()
        i += 1
    return text.strip(), ""


def _already_narrowed(text: str, term: str) -> bool:
    """Is `term` already ANDed into `text`?

    A query with no `OR` in it is a plain conjunction, so a term that
    appears is already narrowing and repeating it would be noise. The
    moment an `OR` appears, the same term may sit inside a group where
    it narrows NOTHING — the shipped Jira brief's
    "… OR priority in (Highest, High) OR …" is exactly that — and no
    substring test can tell the two apart. So it is appended there: a
    redundant AND is idempotent, while a skipped one is a lit chip over
    a read that ignores it.
    """
    low = str(text or "").lower()
    if re.search(r"\bor\b", low) is not None:
        return False
    return re.search(rf"(?<![\w:@.-]){re.escape(term.lower())}(?![\w.@-])",
                     low) is not None


def _and_jql(jql: str, clauses: list) -> str:
    """AND every clause into the WHERE half, in ONE pass.

    One pass rather than one call per clause: re-splitting for each
    would wrap the query again each time, and four filters produced
    four nested layers of parentheses around a query a person has to
    be able to read in the ledger.
    """
    where, order = _split_jql_order_by(jql)
    extra = [c for c in clauses if not _already_narrowed(where, c)]
    if not extra:
        return str(jql)
    joined = " AND ".join(extra)
    where = f"({where}) AND {joined}" if where else joined
    return f"{where} {order}".strip() if order else where


def _and_terms(query: str, term: str) -> str:
    """Append one search term so it narrows the WHOLE query.

    Parenthesised, for the same reason `_and_jql` wraps a WHERE clause: in
    Gmail's grammar (and Slack's) `OR` binds looser than the implicit AND
    between adjacent terms, so a bare append attaches to the LAST branch
    only. The shipped "Newsletter roundup" reads
    `category:promotions OR category:updates newer_than:7d`, and appending
    `-category:promotions` to that left the first branch returning every
    promotion — the chip lit, the mail still there.
    """
    q = str(query or "").strip()
    if not q:
        return term
    if _already_narrowed(q, term):
        return q
    # Only where it changes the meaning: adjacent terms are already ANDed, so
    # wrapping every append would stack `((a) b) c` for nothing.
    return f"({q}) {term}" if _HAS_OR.search(q) else f"{q} {term}"


#: A top-level `OR` is the only thing that makes an appended term bind
#: narrower than the whole query; word-bounded so `ORDER`, `WORD` and an
#: address containing "or" are not mistaken for it.
_HAS_OR = re.compile(r"\bOR\b")

def _apply_read_filters(
    connector_id: str, tool: str, params: dict, filters: list,
    clock: dict,
) -> dict:
    """The user's picked filters, compiled into ONE provider call.

    R43 §6: the table declares WHAT each chip does (`spec.CONNECTOR_
    FILTERS[…]["compile"]`, a closed five-kind vocabulary) and this
    owns HOW. It used to own both — a per-connector `if` ladder with its
    own copy of Gmail's terms and Jira's clauses — so a chip could be
    added to the table, drawn in the popup, saved on the account, and
    change nothing about the read. Reading the compile list here is what
    makes "offered" and "narrows" the same fact.

    Pure and total: unknown connector, unknown tool, malformed params,
    or a filter that needs a clock the run did not supply → the params
    come back untouched.
    """
    if not filters or not isinstance(params, dict):
        return params
    # Table order, never the caller's: the same filter set has to
    # compose the same query byte for byte, or one automation's read
    # reads differently on the run after a re-tap.
    on = {f for f in filters if isinstance(f, str)}
    wanted = [f["id"] for f in _filter_options(connector_id)
              if f["id"] in on and tool in _filter_tools(connector_id, f["id"])]
    if not wanted:
        return params
    now = clock.get("now") if isinstance(clock, dict) else None
    if not isinstance(now, datetime):
        now = None
    out = dict(params)
    jql_clauses: list[str] = []

    for fid in wanted:
        for m in _filter_compile(connector_id, fid, tool):
            kind = m.get("kind")
            if kind == "query_term":
                field = str(m.get("param") or "query")
                term = str(m.get("value") or "")
                if not term:
                    continue
                # A filter is per ACCOUNT and composes into every read
                # step on it, and one shipped template deliberately runs
                # two Gmail windows: the Morning brief's `mail` is
                # `newer_than:1d` and its `waiting` is `older_than:1d
                # newer_than:7d` — the separate window IS the age. ANDing
                # `newer_than:1d` onto the second makes a query that can
                # never match, which `empty_text` then states as the fact
                # "Nothing waiting on you." A step that already carries
                # its own lower age bound keeps it.
                if (term.startswith("newer_than:")
                        and "older_than:" in str(out.get(field) or "")):
                    continue
                out[field] = _and_terms(out.get(field), term)
            elif kind == "jql_and":
                clause = str(m.get("value") or "")
                if clause:
                    jql_clauses.append(clause)
            elif kind == "param":
                name = str(m.get("name") or "")
                if name:
                    # SET, not merged: read state is a partition rather
                    # than a bound, and leaving `is_read: true` under a
                    # lit "Unread only" is the chip that lies.
                    out[name] = m.get("value")
            elif kind == "time_window":
                _apply_time_filter(out, m, now)
            elif kind == "drop":
                # The one kind that runs AFTER the call — no provider
                # query expresses it. `_apply_read_drops` owns it.
                continue

    if jql_clauses:
        # ONE pass: re-splitting per clause would wrap the query again
        # each time, and four filters produced four nested layers of
        # parentheses around a query a person has to be able to read.
        out["jql"] = _and_jql(out.get("jql"), jql_clauses)
    return out


def _apply_time_filter(out: dict, m: dict, now: Optional[datetime]) -> None:
    """One `time_window` mutation, in place.

    A bound only ever moves INWARD — a filter narrows a read, it may
    never widen one the spec already narrowed — so "back" takes the
    LATER of the two lower bounds and "ahead" additionally takes the
    EARLIER of the two upper ones.
    """
    if now is None:
        return
    hours = m.get("hours")
    try:
        hours = int(hours)
    except (TypeError, ValueError):
        return
    field = str(m.get("param") or "")
    if not field:
        return
    unit = str(m.get("unit") or "iso")
    if str(m.get("direction") or "back") == "ahead":
        lo = now.replace(microsecond=0)
        hi = (now + timedelta(hours=hours)).replace(microsecond=0)
        out[field] = _later_iso(out.get(field), lo.isoformat())
        max_field = str(m.get("max_param") or "")
        if max_field:
            out[max_field] = _earlier_iso(out.get(max_field), hi.isoformat())
        return
    since = now - timedelta(hours=hours)
    if unit == "unix":
        # Seconds, the unit conversations.history takes.
        out[field] = _later_ts(out.get(field), f"{since.timestamp():.0f}")
    elif unit == "slack_after":
        # Slack search takes ONE `after:`; a second is not an extra AND,
        # so this is the one place a present term is left alone.
        # `after:` is EXCLUSIVE of the day it names, so the day BEFORE
        # the window's start is what makes the search a superset of the
        # hours `oldest` gives the other tool — one chip, one meaning,
        # whichever Slack read it lands on.
        if "after:" in str(out.get(field) or "").lower():
            return
        out[field] = _and_terms(
            out.get(field),
            f"after:{(since - timedelta(days=1)).date().isoformat()}")
    else:
        out[field] = _later_iso(
            out.get(field), since.replace(microsecond=0).isoformat())


def _drop_field(item: Any, field: str) -> Any:
    """One field off a raw provider row. `resolve_path` so a `drop` can
    name a nested one (`from.emailAddress.address`) without this file
    knowing any provider's shape."""
    if not isinstance(item, (dict, list)):
        return None
    return resolve_path(item, field) if "." in field else (
        item.get(field) if isinstance(item, dict) else None)


def _drops_item(item: Any, m: dict, now: Optional[datetime]) -> bool:
    """Does this `drop` mutation remove this row?

    Fails SAFE in every unreadable case: a row whose field cannot be
    read, a bound that will not parse, a `when` this file does not know
    — the row STAYS. A filter that silently deleted material it could
    not judge would be the same lie as a chip that narrows nothing,
    pointed the other way.
    """
    when = str(m.get("when") or "")
    value = _drop_field(item, str(m.get("field") or ""))
    if when == "present":
        return value not in (None, "", [], {})
    if when == "absent":
        return value in (None, "", [], {})
    if when == "contains":
        text = str(value or "").lower()
        if not text:
            return False
        return any(str(v).lower() in text
                   for v in (m.get("values") or ()) if v)
    if when == "older_than":
        if now is None:
            return False
        try:
            hours = int(m.get("hours"))
            at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
        if at.tzinfo is None:
            at = at.replace(tzinfo=now.tzinfo)
        return at < now - timedelta(hours=hours)
    return False


def _apply_read_drops(
    connector_id: str, tool: str, content: dict, filters: list,
    collect: Optional[dict], clock: dict,
) -> dict:
    """The `drop` half of the compile vocabulary — the narrowings no
    provider query can express (R43 §6).

    Outlook has no KQL predicate for automated mail: `NOT from:noreply`
    inside a `$search` narrows only that one literal address. So "Skip
    automated mail" removes the rows AFTER the call, which is honest
    about what it is — the count the step reports is the count the user
    is shown, because it is the same list.

    Pure and total: no drop filters, no collect, or an items path that
    is not a list → the content comes back unchanged, never emptied.
    """
    if not isinstance(content, dict) or not filters:
        return content
    on = {f for f in filters if isinstance(f, str)}
    mutations = [
        m
        for f in _filter_options(connector_id) if f["id"] in on
        and tool in _filter_tools(connector_id, f["id"])
        for m in _filter_compile(connector_id, f["id"], tool)
        if m.get("kind") == "drop"
    ]
    if not mutations:
        return content
    path = str((collect or {}).get("items_path") or "")
    if not path:
        return content
    items = resolve_path(content, path)
    if not isinstance(items, list):
        return content
    now = clock.get("now") if isinstance(clock, dict) else None
    if not isinstance(now, datetime):
        now = None
    kept = [it for it in items
            if not any(_drops_item(it, m, now) for m in mutations)]
    if len(kept) == len(items):
        return content
    out = dict(content)
    # One level only: every shipped `items_path` is a top-level key and
    # rebuilding an arbitrary nesting would be a second path language.
    if "." in path:
        return content
    out[path] = kept
    return out


def _cmp_iso(a: str, b: str) -> Optional[bool]:
    """`a < b`, or None when either will not parse.

    Naive and AWARE are compared as though both were UTC. The run clock
    is tz-aware (`_run_steps`' `_clock`) and a spec's own bound is a
    string an author typed, so mixing the two is the ordinary case —
    and `datetime.__lt__` raises TypeError across it. That exception
    landed in the `except` below, which returns the spec's bound: the
    lit "Last 24 hours" chip narrowed nothing at all whenever the step
    already carried a naive `since`.
    """
    try:
        x, y = datetime.fromisoformat(a), datetime.fromisoformat(b)
    except (TypeError, ValueError):
        return None
    if (x.tzinfo is None) != (y.tzinfo is None):
        x, y = x.replace(tzinfo=None), y.replace(tzinfo=None)
    return x < y


def _later_iso(current: Any, candidate: str) -> str:
    """The later of two ISO bounds — a filter narrows, never widens.

    A bound the spec wrote that this cannot parse is left exactly as it
    is: authored intent outranks a filter that has no way to compare
    itself against it.
    """
    cur = str(current or "").strip()
    if not cur:
        return candidate
    less = _cmp_iso(cur, candidate)
    return candidate if less else cur


def _earlier_iso(current: Any, candidate: str) -> str:
    """The earlier of two ISO upper bounds — the "ahead" half of the
    same rule `_later_iso` enforces for a lower one."""
    cur = str(current or "").strip()
    if not cur:
        return candidate
    less = _cmp_iso(candidate, cur)
    return candidate if less else cur


def _later_ts(current: Any, candidate: str) -> str:
    """Same rule, in the unix seconds Slack speaks."""
    try:
        return candidate if float(current) < float(candidate) else str(current)
    except (TypeError, ValueError):
        return candidate


# tool → (lower-bound param, upper-bound param). A provider that orders
# ASCENDING and windows only when asked answers an unwindowed read with
# the OLDEST rows it holds.
_TIME_WINDOW_PARAMS: dict[str, tuple[str, str]] = {
    "calendar__list_events": ("time_min", "time_max"),
}
_DEFAULT_WINDOW_DAYS = 1
_MAX_WINDOW_DAYS = 366


def _window_days(raw: Any) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_WINDOW_DAYS
    return min(max(n, 1), _MAX_WINDOW_DAYS)


def _apply_time_window(tool: str, params: dict, clock: dict) -> dict:
    """Give a time-ordered read the window the spec cannot compute
    (R42, B1).

    `calendar__list_events` sets `timeMin`/`timeMax` only when the
    caller passes them and orders by start time ASCENDING, and no
    shipped template passes them — so "your day's calendar" posted the
    oldest events in the account's history, every morning. A spec
    cannot fix that from its own text: there is no `{{now}}` render
    root, and deliberately so (a render root is a string the model
    composes; a clock is a fact of the run). The DISPATCHER supplies
    it, from `ctx["_clock"]`.

    A step names its own horizon with `window_days` — SPEC vocabulary,
    popped for every tool so a provider is never handed a key its
    schema does not declare. Default one day: on a daily automation,
    "what is coming up" means today.

    Each bound is filled INDEPENDENTLY and only when empty, so a spec
    that pins one end keeps it and gains the other, and a spec that
    pinned both is untouched. Pure and total: unknown tool, malformed
    params or a clockless ctx → nothing but `window_days` is removed.
    """
    if not isinstance(params, dict):
        return params
    out = dict(params)
    days = _window_days(out.pop("window_days", None))
    fields = _TIME_WINDOW_PARAMS.get(tool)
    now = clock.get("now") if isinstance(clock, dict) else None
    if fields is None or not isinstance(now, datetime):
        return out
    lo, hi = fields
    if not out.get(lo):
        out[lo] = now.replace(microsecond=0).isoformat()
    if not out.get(hi):
        out[hi] = (now + timedelta(days=days)).replace(
            microsecond=0).isoformat()
    return out


async def _execute_read_step(
    automation: Automation,
    step: ValidatedStep,
    ctx: dict,
) -> dict:
    """One inline read via the platform RPC. Raises RuntimeError on a
    non-ok result — the caller applies on_error."""
    params = render_with_ctx(step.params_template, ctx)
    params = _apply_focus_scope(
        step.connector_id, step.tool, params,
        (ctx.get("_focus_pins") or {}).get(step.connector_id) or [],
    )
    # Then the user's own narrowing, which is what a filter is and a
    # pin is not (§5.2). After the pins so a filter composes into the
    # target the pin just filled.
    params = _apply_read_filters(
        step.connector_id, step.tool, params,
        (ctx.get("_filters") or {}).get(step.connector_id) or [],
        ctx.get("_clock") or {},
    )
    # After both, never before: a pin or a filter that could re-widen
    # the window would put the run back where B1 found it.
    params = _apply_time_window(step.tool, params, ctx.get("_clock") or {})
    # LAST, so the place the user picked is the place this call is
    # aimed at — a filter narrows what comes back from it and may not
    # move it, and a pin only ever fills a target nobody chose (§2.2).
    param_sets = _apply_source_scope(
        step.connector_id, step.tool, params,
        (ctx.get("_account_sources") or {}).get(step.connector_id) or [],
    )
    results = await asyncio.gather(*(
        reg.dispatch_via_platform(
            automation.user_id,
            connector_id=step.connector_id,
            tool_name=step.tool,
            tool_input=p,
            automation_id=automation.id,
        ) for p in param_sets
    ))
    contents = []
    for result in results:
        # One dead place fails the step, exactly as one dead account
        # always has: `on_error` then names the account and the run
        # says so. A partial read served as a whole one is the brief
        # that quietly did not arrive.
        if result.get("kind") != "ok":
            raise RuntimeError(
                f"step {step.id!r} failed: {result.get('kind')}: "
                f"{str(result.get('message') or '')[:200]}"
            )
        try:
            got = json.loads(result.get("content") or "{}")
        except (ValueError, TypeError):
            got = {}
        contents.append(got if isinstance(got, dict) else {})
    content = _merge_read_contents(
        contents, (step.collect or {}).get("items_path") or "")
    # The `drop` half of the filter vocabulary, on the way back: the
    # narrowings no provider query can express. Before `_collect_result`
    # so the step's COUNT is the count of what survived — a filtered
    # read that still reported the provider's total would put a number
    # in the brief that nothing in it accounts for.
    content = _apply_read_drops(
        step.connector_id, step.tool, content,
        (ctx.get("_filters") or {}).get(step.connector_id) or [],
        step.collect, ctx.get("_clock") or {},
    )
    return _collect_result(step, content, ctx.get("var") or {})


# ── The run pipeline ─────────────────────────────────────────────────


async def _run_steps(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    job_id: str,
    event_payload: dict,
    source: Optional[ValidatedSource],
    *,
    idem_prefix: str,
    resume: bool = False,
) -> str:
    """steps (reads) → stage writes → flush → record + memory.

    There is deliberately no "stage but do not flush" mode. One existed
    for the test run, and a staged row is not a held row: `flush_loop`
    sweeps every staged row whose undo window has closed, so the mode
    that promised a rehearsal delivered a send. `rehearse_v2` stages
    nothing at all instead.

    `resume` (R30 §4.3): the run is a reopened stopped run — reads
    re-execute (the honest answer to a moved dedupe window), already-
    staged outbox rows are REUSED instead of conflicting, and the
    narration does not repeat its opening line."""
    mem_ctx = await engine_memory.read_context(db, automation)
    ctx: dict[str, Any] = {
        "event": event_payload or {},
        "source": {
            "id": source.id if source else "",
            "connector_id": (source.connector_id or "") if source else "",
            "event": (source.event or "") if source else "",
        },
        # R38: a COPY. An agent step binds its answer into this dict,
        # and `vspec.variables` is the validated spec's own — mutating
        # it would leak one run's answer into the next run parsed from
        # the same object, and into anything that re-reads the spec.
        "var": dict(vspec.variables or {}),
        "steps": {},
        "memory": mem_ctx,
        # R38 — where the user pinned this automation to START, per
        # account: `{{focus.slack.first.id}}` is the channel they
        # chose. Flat leaves only (`spec.focus_render_ctx`), because
        # `render_value` `str()`s anything that is not a leaf and a
        # Python repr in a connector's params is not a target.
        "focus": _focus_render_ctx(vspec.focus),
        # R39 — the raw pins for `_apply_focus_scope`. Underscored: not
        # a render root, never reachable from a `{{…}}` template.
        "_focus_pins": dict(vspec.focus or {}),
        # R42 — the per-account read filters, for `_apply_read_filters`.
        # Underscored for the same reason; there is deliberately no
        # `{{filters.…}}` root, because a filter narrows the CALL and
        # has nothing to say inside a template.
        "_filters": dict(vspec.filters or {}),
        # R43 §2.2 — the places inside each account the user said it
        # may open, for `_apply_source_scope`. Underscored for the same
        # reason as the two above: it narrows the CALL, so it has
        # nothing to say inside a `{{…}}` template.
        "_account_sources": dict(vspec.account_sources or {}),
        # R42 — the run's clock, for `_apply_time_window`. Underscored
        # for exactly the reason the pins are: `render_value` must
        # never reach it. tz-aware because it leaves this process as an
        # ISO string in a provider's params (the DB columns beside it
        # stay naive UTC, as everywhere else in this engine).
        "_clock": {"now": datetime.now(timezone.utc)},
    }

    # Mail rail first — checked before any step runs, exactly like v1.
    for st in vspec.write_steps:
        if st.tool in _FORBIDDEN_TOOLS:
            await _finalize_job(
                db, job_id, status="failed", outcome="forbidden_tool",
                error_class="policy",
                user_message="Automations never send mail — use a draft "
                             "action.",
            )
            await _record_health(db, automation.id, ok=False,
                                 error="forbidden tool " + st.tool)
            return "failed"

    # R30 v3 ledger context — best-effort throughout: a ledger failure
    # never changes the run's outcome (the typed record degrades, the
    # work does not).
    from . import ledger as _ledger
    from . import run_v3 as _rv3
    from app.services import automation_verbs as _verbs
    import time as _time
    thread = None
    job_row = await db.get(BuildJob, job_id)
    try:
        thread = await _ledger.thread_for(db, automation.id)
    except Exception:  # noqa: BLE001
        thread = None
    total_steps = len(vspec.steps)
    step_no = 0
    tool_turn_by_step: dict[str, dict] = {}

    async def _stop_boundary(at_step: int) -> bool:
        """§4.3: the stop takes effect at the next step boundary."""
        if job_row is None:
            return False
        try:
            if await _rv3.stop_requested(db, job_id):
                await _rv3.handle_stop(
                    db, automation=automation, job=job_row,
                    step_index=at_step,
                )
                return True
        except Exception as e:  # noqa: BLE001
            logger.debug("[automations] stop boundary check failed: %s", e)
        return False

    # CONTRACTS-R31 §4.2a — a failing account never stops the run.
    #
    # `partial` alone used to carry two very different facts: "a step was
    # skipped" and nothing else. It could not say WHICH account, WHY, or
    # what would fix it, so the run's own record could not answer the
    # question the user was about to ask. `failed_sources` is that
    # record, and it is what the `needs_you` turns, the honest line, the
    # notification flip and the per-source resume are all built from.
    partial = False
    failed_sources: list[dict] = []
    read_ok: list[str] = []
    # R38: what each agent step worked out, for the narration record.
    # It is deliberately NOT the ledger turn's `detail` — that is
    # copy-guarded engine prose, and this is model output.
    agent_outputs: dict[str, str] = {}
    for st in vspec.steps:
        if st.mutates:
            break
        if await _stop_boundary(step_no):
            return "stopped"
        step_no += 1
        is_agent = st.kind == "agent"
        sentence = (_verbs.live_sentence(None, None, phase="think")
                    if is_agent
                    else _verbs.live_sentence(st.connector_id, st.tool))
        try:
            await _ledger.emit_progress(
                automation.user_id, run_id=job_id,
                automation_id=automation.id, step=step_no,
                total=total_steps, sentence=sentence,
                fraction=(step_no - 1) / max(total_steps, 1),
                status="running",
            )
            if job_row is not None:
                await _rv3.notify_progress(
                    db, automation=automation, job=job_row, step=step_no,
                    total=total_steps, sentence=sentence,
                    fraction=(step_no - 1) / max(total_steps, 1),
                )
        except Exception:  # noqa: BLE001
            pass
        # R31-30, second half: `progress_step` / `progress_total` have
        # FOUR readers in this engine — the terminal frame, the park,
        # the home card's fraction, and run-now's 409 sentence — and had
        # no writer anywhere in the repo. So `Already running — step 0
        # of 5` was not a stale number, it was a column nobody had ever
        # filled, and a running automation's card always drew 0%.
        # Progress lived only in the ephemeral WS frame.
        try:
            if job_row is not None:
                job_row.progress_step = step_no
                job_row.progress_total = total_steps
                await db.flush()
        except Exception as e:  # noqa: BLE001 — progress never fails a run
            logger.debug("[automations] progress stamp skipped: %s", e)

        # §4.5: the same phase change the main chat has always had. An
        # agent step is `thinking` — there is no connector to name and
        # the tool glyph would be a lie about what is happening.
        try:
            await _ledger.emit_activity(
                automation.user_id, automation_id=automation.id,
                thread_id=thread.id if thread is not None else None,
                run_id=job_id,
                phase="thinking" if is_agent else "tool",
                tool=None if is_agent else {
                    "account_id": st.connector_id or "",
                    "label": sentence,
                },
                detail=sentence if is_agent else None,
            )
        except Exception:  # noqa: BLE001 — a frame never fails a run
            pass

        _t0 = _time.monotonic()
        step_failed_reason = None
        try:
            if is_agent:
                answer = await _agent_step.run_agent_step(
                    automation=automation, step=st, ctx=ctx,
                )
                agent_outputs[st.id] = answer
                # BOTH namespaces, because both are already how a step's
                # output is read: `{{var.<output_var>}}` is the name the
                # step declares, `{{steps.<id>.text}}` is what every
                # other step exposes and what a spec author will reach
                # for out of habit.
                ctx["steps"][st.id] = {"ok": True, "text": answer}
                ctx["var"][st.output_var] = answer
            else:
                ctx["steps"][st.id] = await _execute_read_step(
                    automation, st, ctx)
        except Exception as e:  # noqa: BLE001 — transport/shape errors
            step_failed_reason = _failure_reason(e)
            if st.on_error == "fail":
                # Still reachable, and still correct for a step whose
                # absence makes the rest of the run meaningless. It is
                # no longer the DEFAULT for a read (spec_v2) — it IS
                # the default for an agent step, whose answer later
                # templates interpolate.
                await _append_step_turn(
                    db, thread=thread, automation=automation, job_id=job_id,
                    step=st, result=None, ms=int((_time.monotonic() - _t0) * 1000),
                    ok=False, reason=step_failed_reason,
                    turn_index=tool_turn_by_step,
                )
                await _finalize_job(
                    db, job_id, status="failed", outcome="step_failed",
                    error_class="tool_error",
                    user_message=f"Step {st.id!r} failed: {str(e)[:200]}",
                )
                await _record_health(db, automation.id, ok=False,
                                     error=str(e)[:500])
                return "failed"
            logger.info("[automations] step %s continued past %s on %s",
                        st.id, e, automation.id)
            ctx["steps"][st.id] = (
                {"ok": False, "failed": True, "text": ""} if is_agent
                else _skipped_result(st, silent=st.on_error != "continue")
            )
            if is_agent:
                # The name still resolves, to nothing. Leaving it
                # unbound would make `{{var.x}}` render as the literal
                # empty string anyway; binding it says so in one place.
                ctx["var"][st.output_var] = ""
            partial = True
            if st.on_error == "continue" and not is_agent:
                # `skip` stays SILENT (the Teams provider_down
                # precedent); `continue` owes the user a named account,
                # a real reason and a button.
                #
                # An AGENT step is excluded on purpose: `failed_sources`
                # is a list of ACCOUNTS, and every consumer treats it as
                # one — the needs-you cards, `accounts_failed`, the
                # reconnect buttons, the per-source resume. A thinking
                # step has no account to name and no connector to
                # reconnect; its failure is recorded as its own turn,
                # in the thread, next to the run that hit it.
                failed_sources.append({
                    "account_id": st.connector_id or "",
                    "reason_code": _reason_code_of(e, step_failed_reason),
                    "step_id": st.id,
                    "at": datetime.utcnow().isoformat() + "Z",
                    # The provider's own words. `classify` reads them to
                    # tell an expired token from an org policy from a
                    # rate limit, and `record_use` could not fire either
                    # of its regexes without them (round 33, item 4).
                    "message": str(e)[:300],
                })
        else:
            if st.connector_id and st.connector_id not in read_ok:
                read_ok.append(st.connector_id)
        step_result = ctx["steps"].get(st.id) or {}
        await _append_step_turn(
            db, thread=thread, automation=automation, job_id=job_id,
            step=st, result=step_result,
            ms=int((_time.monotonic() - _t0) * 1000),
            ok=step_failed_reason is None, reason=step_failed_reason,
            turn_index=tool_turn_by_step,
        )
        await _advance_v2(
            db, job_id, vspec, st.id,
            count=step_result.get("count")
            if isinstance(step_result.get("count"), int) else None,
        )

    if partial:
        # The aggregate finalizer reads this to report `partial`
        # honestly when the writes themselves succeed.
        await merge_job_config(db, job_id, steps_partial=True)

    if failed_sources:
        # §4.2a. Stamped on the RUN, so `accounts_failed` is answerable
        # before the ledger closes — the notification flip, the home
        # card's meta and the per-source resume all read it, and two of
        # those happen while the run is still going.
        await merge_job_config(
            db, job_id,
            accounts_failed=[f["account_id"] for f in failed_sources
                             if f.get("account_id")],
            failed_sources=failed_sources,
        )
        await _append_needs_you_turns(
            db, thread=thread, automation=automation, job_id=job_id,
            failed_sources=failed_sources,
        )

    if failed_sources and not read_ok:
        # EVERY source failed. There is nothing to post and nothing to
        # rank — a brief assembled from nothing is a lie with a nice
        # layout. The run is `failed`, and the thread already carries one
        # named card per account with the button that fixes it.
        await _finalize_job(
            db, job_id, status="failed", outcome="all_sources_failed",
            error_class="tool_error",
            user_message=_all_failed_message(failed_sources),
        )
        await _record_health(db, automation.id, ok=False,
                             error=_all_failed_message(failed_sources))
        return "failed"

    if failed_sources:
        # Some read, some did not: the brief goes out, and it SAYS so.
        # "GitHub and Outlook are missing from this — I could not read
        # them" is the difference between a brief the user can trust and
        # one they have to audit.
        await _append_honest_line(
            db, thread=thread, automation=automation, job_id=job_id,
            failed_sources=failed_sources,
        )

    # §4.3: the last boundary before writes — a stop that arrived during
    # the reads must land HERE; no write step may start after it.
    if vspec.write_steps and await _stop_boundary(step_no):
        return "stopped"

    # Stage every write in one transaction: a replayed run conflicts on
    # w0 and rolls the whole batch back — all-or-nothing, never a
    # half-staged retry.
    rows: list[AutomationOutbox] = []
    execute_after = datetime.utcnow() + timedelta(
        seconds=AUTOMATION_OUTBOX_UNDO_WINDOW_S)
    for n, st in enumerate(vspec.write_steps):
        step_ctx = dict(ctx)
        step_ctx["grant"] = {"target": st.grant_target or {}}
        params = render_with_ctx(st.params_template, step_ctx)
        rows.append(AutomationOutbox(
            user_id=automation.user_id,
            automation_id=automation.id,
            job_id=job_id,
            connector_id=st.connector_id,
            tool_name=st.tool,
            payload_json=json.dumps(params, sort_keys=True, default=str),
            grant_id=st.grant_id,
            idempotency_key=f"{idem_prefix}:w{n}"[:128],
            execute_after=execute_after,
            display_json=json.dumps(
                _write_display(st), sort_keys=True, default=str,
            ),
        ))
    db.add_all(rows)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        if not resume:
            logger.info("[automations] outbox idempotency hit %s",
                        idem_prefix)
            return "run"
        # Resume: the stop landed after staging — reuse the surviving
        # staged rows (an executed/cancelled one stays terminal; the
        # claim gate keeps a double-send impossible either way).
        from sqlalchemy import select as _select
        rows = list((await db.execute(
            _select(AutomationOutbox)
            .where(AutomationOutbox.job_id == job_id)
            .where(AutomationOutbox.status == "staged")
        )).scalars())
        now2 = datetime.utcnow()
        for row in rows:
            if row.execute_after and row.execute_after < now2:
                row.execute_after = now2 + timedelta(
                    seconds=AUTOMATION_OUTBOX_UNDO_WINDOW_S)
        await db.commit()
    else:
        await db.commit()
    # ND-4: a write step's done-form verb lands when the write actually
    # EXECUTES (outbox ok branch, keyed by display_json.step_id) — never
    # at staging. A refused write must not wear "Posted to Slack".

    # R42 — the WRITE goes first, and the run's terminal with it.
    #
    # Narration used to sit here, ahead of the flush, so that the
    # thread's opening line could precede the write turn. That promise
    # was never kept: the undo window is 6 s and `flush_loop` sweeps
    # every 5 s, so a narration taking anything like an LLM's time was
    # overtaken by the background loop, which sent the write, then
    # blocked on `_mark_write_step`'s UPDATE of `build_jobs` behind the
    # progress stamp this session had flushed and not committed. At a
    # 30 s `statement_timeout` that UPDATE is CANCELLED — so the post
    # had landed, `_finalize_job` never ran, and the card sat at 99%
    # "Working now" until the 360 s stuck-run reaper called it failed.
    #
    # Flushing here fixes both halves. The write no longer waits on
    # prose, and the send + terminal happen on THIS session, where a
    # lock cannot be contended, instead of racing a background loop
    # that had become the de-facto decider of the ordering.
    if rows:
        # …and the LEDGER CLOSE is what waits for the prose now.
        #
        # `close_ledger` reads "the run is terminal and has no result
        # turn" as "narration failed outright" and appends a mechanical
        # one ("N items read — I could not rank them this time"). That
        # test was true only while narration preceded the terminal. With
        # the flush first it fires on every healthy write run, and the
        # narrator's real result lands behind the fabricated one — two
        # results in one thread. The rest of the close judges the
        # narrated record too: the missing-item reconciliation, the
        # vocabulary tripwire and the result-row episodes all read turns
        # that do not exist yet.
        #
        # So the flag defers the whole close rather than one branch of
        # it, `run_v3.on_terminal` honours it, and `_close_ledger_after_
        # narration` below both closes the ledger and clears the flag.
        # Stamped WITH a time, because the flag is a claim on someone
        # else's work and a claim needs an expiry. The `finally` below
        # releases it on any exception, but not on the class this whole
        # reordering exists because of — a pod eviction between the stamp
        # and the release, which would leave every future terminal for
        # this run (including the stuck-run reaper's) skipping its close
        # forever. `on_terminal` honours the claim only while it is fresh.
        await merge_job_config(
            db, job_id,
            narration_pending=True,
            narration_pending_at=datetime.utcnow().isoformat() + "Z",
        )

    from .outbox import flush_row_when_due
    statuses = []
    try:
        for row in rows:
            statuses.append(await flush_row_when_due(db, row.id))

        # Phase 1 is still the opening line + the item whys, and phase 2
        # still holds the result until the writes are known: a run whose
        # write FAILED must not be narrated as one that landed.
        narration = await _narrate_phase1(
            db, automation=automation, vspec=vspec, job_id=job_id,
            thread=thread, tool_turn_by_step=tool_turn_by_step,
            partial=partial, failed_sources=failed_sources,
            agent_outputs=agent_outputs,
            # A run with an outbox row does not own its own terminal:
            # the outbox does, in the flush above or in a later retry.
            # Either way this pass must not announce a step on its
            # behalf.
            terminal_landed=bool(rows),
        )

        await _narrate_phase2(
            db, automation=automation, job_id=job_id, thread=thread,
            narration=narration,
            writes_ok=not any(s == "failed" for s in statuses),
        )
    finally:
        # In a `finally` because the flag is a CLAIM: while it is set
        # `on_terminal` declines to close, so an exception escaping the
        # inline flush (nothing here is the blanket handler `flush_loop`
        # gives the background path) would leave the run's ledger to be
        # skipped by whichever sweep terminalizes it, forever.
        if rows:
            await _close_ledger_after_narration(
                db, automation=automation, job_id=job_id,
            )

    outcome = "sent"
    if any(s == "failed" for s in statuses):
        outcome = "failed"
    elif partial:
        outcome = "partial"

    if not rows:
        # A READS-ONLY run has no outbox row, and the outbox flush is
        # this path's ONLY route to `_finalize_job` — so nothing
        # terminalized it. The job sat `running` for the full 360 s
        # stuck-run window and was then reaped as `failed/lost` with a
        # "Fix this" chip, which is precisely the founder's `Morning
        # new-email briefing`: a thread ending "Your inbox is clear for
        # now." under a home card reading `Tried 1:20 · it did not
        # finish` (R31-31, F10).
        #
        # Reads-only specs became legal in R30 §4.11a — the migrated
        # email briefings are exactly that shape — and this terminal
        # was never added with them. `_finalize_job`'s guarded UPDATE
        # keeps it exactly-once, so it is safe beside every other
        # terminal, and going through it (never a raw UPDATE) is what
        # keeps `_stamp_last_outcome`, the outcome notification and the
        # v3 ledger close coupled (CONTRACTS-R30 §12).
        await _finalize_job(
            db, job_id,
            status="completed",
            outcome="partial" if partial else "sent",
        )
        await _record_health(db, automation.id, ok=True, error=None,
                             ran=True, clean=not partial)

    # R43 §1.2 — and only now. Delivery fans the BRIEF out, so it has to
    # come after the result turn exists AND after the ledger close that
    # may still be repairing it (the mechanical fallback, the missing-
    # item row, the "nothing came back" row). Delivering from inside
    # narration would post whatever the narrator happened to emit, which
    # is exactly the half of the run the completeness net exists for.
    #
    # It is also after the terminal, deliberately: a delivery row is an
    # outbox row like any other, and `outbox._finalize_run` aggregates
    # siblings — so a Slack DM that failed would otherwise flip a run
    # that had already told the user it was done. `_finalize_job`'s
    # guarded UPDATE makes the aggregate a no-op on a terminal row.
    if outcome != "failed":
        await _deliver_run_brief(
            db, automation=automation, vspec=vspec, job_id=job_id,
            thread=thread, source=source, idem_prefix=idem_prefix,
        )

    counts = {
        sid: res.get("count")
        for sid, res in ctx["steps"].items()
        if isinstance(res, dict) and res.get("count") is not None
    }
    await engine_memory.write_after_run(
        user_id=automation.user_id,
        automation_id=automation.id,
        automation_name=automation.name,
        outcome=outcome,
        counts=counts,
    )
    try:
        await _ledger.emit_activity(
            automation.user_id, automation_id=automation.id,
            thread_id=thread.id if thread is not None else None,
            run_id=job_id, phase="done",
        )
        await _ledger.emit_updated(
            db, automation.user_id, automation_id=automation.id,
        )
    except Exception:  # noqa: BLE001 — a frame never fails a run
        pass
    return "run"


# ── Entry points (cap-bounded, mirroring executor.py) ────────────────


def _refuse_during_drain(automation: Automation, kind: str) -> bool:
    """§4.8: a deploy "never starts a run it will kill".

    R31-42. The drain gate blocks new WEBSOCKETS and deliberately lets
    HTTP through — which is exactly how an inbound push starts a run
    during a deploy. That run has, at best, `drain_timeout_s` to do
    three minutes of work; at worst it is killed before its first step,
    and a killed run is not quiet: it is reaped `failed/lost` and the
    user is told their automation broke.

    Skipping is the honest outcome. A scheduled fire comes round again;
    a push event stays in its dedupe namespace and the next poll picks
    it up. Neither is worse than a run that dies at step two and
    reports a connector problem that never happened.
    """
    try:
        from app.services import drain_state as _drain
        if not _drain.should_refuse_new_run():
            return False
    except Exception:  # noqa: BLE001 — no drain module ⇒ never refuse
        return False
    logger.warning(
        "[automations] run refused during drain automation=%s kind=%s "
        "— it would be killed mid-flight",
        automation.id, kind,
    )
    return True


async def run_event_v2(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    source: ValidatedSource,
    event: AutomationEvent,
) -> str:
    if _refuse_during_drain(automation, "event"):
        return "drained"
    try:
        return await asyncio.wait_for(
            _run_event_inner(db, automation, vspec, source, event),
            timeout=AUTOMATION_RUN_CAP_S,
        )
    except asyncio.TimeoutError:
        logger.warning("[automations] run cap hit automation=%s event=%s",
                       automation.id, event.id)
        if event.job_id:
            await _finalize_job(
                db, event.job_id, status="failed", outcome="run_cap",
                error_class="timeout",
                user_message="The run exceeded the 3-minute cap and was "
                             "stopped.",
            )
        await _record_health(db, automation.id, ok=False,
                             error="run cap exceeded")
        return "failed"


async def _run_event_inner(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    source: ValidatedSource,
    event: AutomationEvent,
) -> str:
    try:
        payload = json.loads(event.payload_json) if event.payload_json else {}
    except (ValueError, TypeError):
        payload = {}

    from .facts_context import load_facts_context
    facts_ctx = await load_facts_context(
        db, automation.id, source.filter_rules,
    )
    if not _passes_filter_v2(source, payload, vspec.variables, facts_ctx):
        event.status = "skipped_filter"
        await db.commit()
        return event.status

    job = await JobRunner().create_job(
        job_type="automation_run",
        spec=TaskSpec(
            user_id=automation.user_id,
            channel="automation",
            source_kind="automation",
            source_id=automation.id,
            config_json={"automation_event_id": event.id,
                         "source_id": source.id},
        ),
        title=f"{automation.name}"[:100],
        idempotency_key=f"evt:{event.id}",
        status="running",
        steps_json=_new_steps_v2(vspec),
        layer=0,
    )
    event.status = "run"
    event.job_id = job.id
    await db.commit()
    await on_run_created(db, job=job, automation=automation)
    await _advance_v2(db, job.id, vspec, "evaluate")
    from . import run_v3 as _rv3_open
    await _rv3_open.open_run(db, automation=automation, job=job,
                             kind="scheduled", total_steps=len(vspec.steps))

    return await _run_steps(db, automation, vspec, job.id, payload, source,
                            idem_prefix=f"evt:{event.id}")


async def _finalize_on_cap(job_id: str) -> None:
    """ND-7b: terminalize a cap-hit run on a FRESH session — the run's
    own session was cancelled mid-flight by wait_for and may be wedged
    in an open transaction; reusing it can hang or raise, which is how
    the row stayed `running` forever (the R27 zombie class reborn)."""
    from app.db.database import async_session_maker
    try:
        async with async_session_maker() as fresh:
            await _finalize_job(
                fresh, job_id, status="failed", outcome="run_cap",
                error_class="timeout",
                user_message="The run exceeded the 3-minute cap and was "
                             "stopped.",
            )
    except Exception as e:  # noqa: BLE001 — the sweep is the backstop
        logger.warning("[automations] cap finalize failed job=%s: %s",
                       job_id[:8], e)


async def run_schedule_fire_v2(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    source: ValidatedSource,
    fire_key: str,
    run_kind: str = "scheduled",
) -> str:
    if _refuse_during_drain(automation, run_kind):
        return "drained"
    # ND-7b: the job is minted INSIDE the wait_for — the ref carries its
    # id out so the cap handler can finalize (the old handler could not
    # even name the job and returned with the row still `running`).
    job_ref: dict = {}

    async def _inner() -> str:
        # §4.3: the next scheduled fire supersedes a still-stopped run —
        # its stop note was already written; no new turn.
        try:
            from . import run_v3 as _rv3_sup
            await _rv3_sup.supersede_stopped_run(
                db, automation_id=automation.id,
            )
        except Exception:  # noqa: BLE001
            pass
        job = await JobRunner().create_job(
            job_type="automation_run",
            spec=TaskSpec(
                user_id=automation.user_id,
                channel="automation",
                source_kind="automation",
                source_id=automation.id,
                config_json={"fire_key": fire_key, "source_id": source.id},
            ),
            title=f"{automation.name}"[:100],
            idempotency_key=f"fire:{fire_key}"[:120],
            status="running",
            steps_json=_new_steps_v2(vspec),
            layer=0,
        )
        job_ref["id"] = job.id
        await on_run_created(db, job=job, automation=automation)
        await _advance_v2(db, job.id, vspec, "evaluate")
        from . import run_v3 as _rv3_open
        await _rv3_open.open_run(db, automation=automation, job=job,
                                 kind=run_kind,
                                 total_steps=len(vspec.steps))
        return await _run_steps(db, automation, vspec, job.id, {}, source,
                                idem_prefix=f"fire:{fire_key}")

    try:
        return await asyncio.wait_for(_inner(), timeout=AUTOMATION_RUN_CAP_S)
    except asyncio.TimeoutError:
        logger.warning("[automations] run cap hit automation=%s fire=%s",
                       automation.id, fire_key[:40])
        if job_ref.get("id"):
            await _finalize_on_cap(job_ref["id"])
        try:
            await _record_health(db, automation.id, ok=False,
                                 error="run cap exceeded")
        except Exception:  # noqa: BLE001 — the session may be wedged too
            logger.warning("[automations] cap health record failed "
                           "automation=%s", automation.id)
        return "failed"


# ── Poll leg ─────────────────────────────────────────────────────────


async def _poll_once_v2(
    automation: Automation, source: ValidatedSource,
) -> list[dict]:
    ev_spec = source.event_spec or {}
    source_tool = ev_spec.get("source_tool")
    if not source_tool:
        return []
    args = dict(ev_spec.get("poll_args") or {})
    args.update(source.params or {})
    result = await reg.dispatch_via_platform(
        automation.user_id,
        connector_id=source.connector_id or "",
        tool_name=source_tool,
        tool_input=args,
        automation_id=automation.id,
    )
    if result.get("kind") != "ok":
        raise RuntimeError(
            f"poll failed: {result.get('kind')}: "
            f"{str(result.get('message') or '')[:200]}"
        )
    try:
        content = json.loads(result.get("content") or "{}")
    except (ValueError, TypeError):
        content = {}
    items_path = ev_spec.get("items_path")
    items = resolve_path(content, items_path) if items_path else content
    return items if isinstance(items, list) else []


# The two windows a manifest may declare. Order is immaterial — the
# most restrictive of them decides.
_RATE_WINDOWS: tuple[tuple[str, timedelta], ...] = (
    ("per_hour", timedelta(hours=1)),
    ("per_day", timedelta(days=1)),
)


async def _rate_budget(automation: Automation, connector_id: str) -> dict:
    """The connector's declared `automation.rate_budget`, or {}.

    Fails OPEN: `fetch_registry` answers {} when the platform is
    unreachable, and a poll that cannot read the budget must still run
    the user's automation. The cap is a fan-out guard, not a rail — the
    rails (mail, grants, the outbox) all fail closed platform-side.
    """
    if not connector_id:
        return {}
    entry = (await reg.fetch_registry(automation.user_id)).get(connector_id)
    budget = (entry or {}).get("rate_budget")
    return budget if isinstance(budget, dict) else {}


async def _rate_allowance(
    db, automation: Automation, budget: dict,
) -> Optional[int]:
    """How many more runs this automation may start now, or None when
    nothing caps it (R42, B5).

    Every connector manifest has declared `automation.rate_budget`
    since R26 — `AutomationCapability`'s own field comment says "the
    rate budget the executor enforces" — and nothing ever enforced it,
    while `AUTOMATION_EVENT_STATUSES` has carried `skipped_rate` with
    no writer. A poll window is up to 50 items, so one tick could mint
    50 runs.

    The window is counted off `automation_events` rather than kept in
    memory. `triggers/rate_limiter.py` is the working sliding window
    for the OLD triggers engine and its shape is the one copied here —
    a count of fires inside each window, most restrictive wins — but it
    is a per-container token bucket, and importing it would couple two
    engines this repo keeps apart and would still lose the count on a
    restart that the runs it guards survive. The event rows ARE the
    ledger; counting them is one indexed query per tick
    (`ix_automation_events_auto_received`).

    `received_at` is the intake stamp, which on the poll leg is the
    same tick the run starts on — the only leg this gate sits on.
    """
    limits: list[tuple[int, timedelta]] = []
    for key, span in _RATE_WINDOWS:
        try:
            n = int(budget.get(key))
        except (TypeError, ValueError):
            continue
        if n > 0:
            limits.append((n, span))
    if not limits:
        return None
    from sqlalchemy import func as _func, select as _select
    now = datetime.utcnow()
    allowance: Optional[int] = None
    for n, span in limits:
        used = int((await db.execute(
            _select(_func.count(AutomationEvent.id))
            .where(AutomationEvent.automation_id == automation.id)
            .where(AutomationEvent.status == "run")
            .where(AutomationEvent.received_at >= now - span)
        )).scalar() or 0)
        left = max(n - used, 0)
        allowance = left if allowance is None else min(allowance, left)
    return allowance


# How many carried-over events one tick may pick up, whatever the
# budget allows. The same order of magnitude as a poll window: a tick
# that starts fifty runs is the fan-out this gate exists to stop, and a
# backlog longer than that is drained a tick at a time.
_PENDING_DRAIN_MAX = 50


async def _pending_events(
    db, automation_id: str, source_id: str,
) -> list[AutomationEvent]:
    """This source's fresh events that no tick has run yet, oldest
    first.

    An event the rate budget could not afford is DEFERRED, not dropped:
    it keeps its `new` status and the next tick spends its allowance on
    it before anything the provider has just handed back. Nothing else
    drains a `new` row — the dedupe gate makes `ingest_items_v2` skip
    that item on every later poll — so stamping it `skipped_rate` here
    was the last time the engine would ever see it, and the user could
    not tell that from an automation that had simply not fired.

    The same query picks up an event a deploy drain refused (`run_event_v2`
    returns "drained" and leaves the row `new`), which had exactly the
    same one-way exit.
    """
    from sqlalchemy import select as _select
    rows = (await db.execute(
        _select(AutomationEvent)
        .where(AutomationEvent.automation_id == automation_id)
        .where(AutomationEvent.status == "new")
        .where(AutomationEvent.dedupe_key.like(
            f"{_like_prefix(source_id)}:%", escape="\\"))
        .order_by(AutomationEvent.received_at.asc())
        .limit(_PENDING_DRAIN_MAX)
    )).scalars().all()
    return list(rows)


async def poll_and_run_v2(
    db, automation: Automation, vspec: ValidatedSpecV2,
    source: ValidatedSource,
) -> dict:
    items = await _poll_once_v2(automation, source)
    # B5, first half: the first poll of a source records what it saw
    # and runs nothing. Before this, the dedupe table was empty on that
    # tick, so every item in a 25-50 item window was "new" — arming a
    # calendar automation fired one run per event already on the
    # calendar.
    baselined = not await _source_has_history(db, automation.id, source.id)
    # Read BEFORE the intake, or this tick's own fresh rows come back in
    # it and every event is queued twice.
    pending = await _pending_events(db, automation.id, source.id)
    fresh = await ingest_items_v2(db, automation, source, items,
                                  baseline=baselined)
    if baselined and items:
        logger.info(
            "[automations] first poll baselined automation=%s source=%s "
            "observed=%d — recorded as seen, nothing ran",
            automation.id, source.id, len(items),
        )

    # B5, second half: the declared budget, enforced. One log line per
    # tick — a line per skipped event is the same fan-out in the log.
    allowance = await _rate_allowance(
        db, automation,
        await _rate_budget(automation, source.connector_id or ""),
    )
    ran = 0
    failed = 0
    started = 0
    deferred = 0
    # The carry-over spends the budget first: an event the last tick
    # could not afford is older than anything this poll returned, and
    # letting fresh items overtake it is how a backlog starves.
    for event in pending + fresh:
        if allowance is not None and started >= allowance:
            # Left `new` on purpose — see `_pending_events`. Not
            # `skipped_rate`, which reads as a verdict and is a one-way
            # exit; this event has not been judged, only postponed.
            deferred += 1
            continue
        status = await run_event_v2(db, automation, vspec, source, event)
        # A filtered or drained event minted no job, so it spends
        # nothing: the budget counts RUNS.
        if status not in ("skipped_filter", "drained"):
            started += 1
        if status == "failed":
            failed += 1
        elif status == "run":
            ran += 1
    if deferred:
        logger.info(
            "[automations] rate budget spent automation=%s connector=%s "
            "allowance=%d deferred=%d — the next tick runs them",
            automation.id, source.connector_id or "", allowance, deferred,
        )

    if failed == 0:
        # See `_record_health`: a poll with nothing fresh is connector
        # health, not a run. Stamping it as a run made the health object
        # claim runs the ledger had never heard of (ND-25).
        await _record_health(db, automation.id, ok=True, error=None,
                             ran=ran > 0)
    return {"observed": len(items), "fresh": len(fresh),
            "ran": ran, "failed": failed,
            "baselined": len(items) if baselined else 0,
            "carried_in": len(pending), "deferred_rate": deferred}


# ── Rehearsal ────────────────────────────────────────────────────────
#
# R38 — this used to be `execute_test_run_v2`, and it was DEV-only for
# two reasons that were both true.
#
# It was not a rehearsal. `_run_steps(stage_only=True)` returned before
# narration, but the outbox row it had just committed was `staged` with
# an `execute_after` in the past six seconds — and `outbox.flush_loop`
# sweeps EVERY staged row whose window has closed, every 5 s. So a
# "test" posted to the user's real Slack channel, seconds later, from a
# background loop, with the caller already gone. And it short-circuited
# the ledger's phase-2 close, so the run it opened produced no result
# turn, no notification and a job row that the stuck-run reaper later
# marked `failed/lost`.
#
# The fix is not a flag. A rehearsal STAGES NOTHING: there is no outbox
# row to sweep, so there is no code path — no loop, no restart, no
# retry — that can turn one into a send. The reads run for real (that
# is the whole point of rehearsing against live data, and a read
# changes nothing), the write params are RENDERED exactly as the run
# would render them, and they are reported instead of persisted.
#
# It also opens no run: a rehearsal changes nothing, so there is
# nothing for the thread, the home card or a notification to record,
# and a run row claiming otherwise was half of the old defect.


async def rehearse_v2(
    db, automation: Automation, vspec: ValidatedSpecV2,
) -> dict:
    """Run the reads for real; report what WOULD be written.

    Returns `{"rehearsal": True, "sample_event", "reads", "writes"}`.
    Each `writes` entry carries the rendered params and, when the
    engine would refuse the write, a `blocked` reason — a rehearsal
    that says "it would post to #platform" about a write no grant backs
    is a rehearsal of something that cannot happen.
    """
    sample: dict[str, Any] = {}
    source: Optional[ValidatedSource] = None
    for s in vspec.sources:
        if s.mode in ("push", "poll"):
            source = s
            break
    if source is not None and source.mode == "poll":
        try:
            items = await _poll_once_v2(automation, source)
        except Exception:  # noqa: BLE001 — sample is best-effort
            items = []
        if items:
            ev_spec = source.event_spec or {}
            fields = dict(ev_spec.get("fields") or {})
            sample = {
                name: resolve_path(items[0], path)
                for name, path in fields.items()
            }
    if not sample and source is not None:
        fields = dict((source.event_spec or {}).get("fields") or {})
        sample = {name: f"<{name}>" for name in fields} or {"sample": "<test>"}
    if source is None:
        source = vspec.schedule_source()

    ctx: dict[str, Any] = {
        "event": sample,
        "source": {
            "id": getattr(source, "id", ""),
            "connector_id": getattr(source, "connector_id", "") or "",
            "event": getattr(source, "event", "") or "",
        },
        # A copy, for the same reason the run path takes one: an agent
        # step binds into this dict.
        "var": dict(vspec.variables or {}),
        "steps": {},
        "memory": await engine_memory.read_context(db, automation),
        # R38. The rehearsal must render the SAME write the run would.
        # Without this key `{{focus.slack.first.id}}` — the pattern the
        # authoring guide documents for a pinned sub-node — resolved to ""
        # (render_value answers "" for a missing path), so a rehearsal of a
        # pinned automation read the wrong channel and then reported
        # "it would write" params built from it, under a prompt that says
        # "say what it WOULD write". An agent step reads it too
        # (agent_step.build_prompt's `starts_at`).
        "focus": _focus_render_ctx(vspec.focus),
        # R42. The same run-only roots `_run_steps` builds, for the
        # same reason: a rehearsal that reads a different window, a
        # different pinned channel or an unfiltered inbox is a
        # rehearsal of a different run.
        "_focus_pins": dict(vspec.focus or {}),
        "_filters": dict(vspec.filters or {}),
        "_account_sources": dict(vspec.account_sources or {}),
        "_clock": {"now": datetime.now(timezone.utc)},
    }

    reads: list[dict] = []
    for st in vspec.steps:
        if st.mutates:
            break
        if st.kind == "agent":
            # A rehearsal runs the thinking for real, exactly like the
            # reads: it changes nothing, and the whole point of
            # rehearsing is to see what the writes below would actually
            # say — which is this step's answer, interpolated.
            try:
                answer = await _agent_step.run_agent_step(
                    automation=automation, step=st, ctx=ctx,
                )
            except Exception as e:  # noqa: BLE001 — report, never hide
                ctx["steps"][st.id] = {"ok": False, "failed": True,
                                       "text": ""}
                ctx["var"][st.output_var] = ""
                reads.append({
                    "step_id": st.id, "account_id": None, "kind": "agent",
                    "ok": False, "on_error": st.on_error,
                    "error": str(e)[:200],
                })
            else:
                ctx["steps"][st.id] = {"ok": True, "text": answer}
                ctx["var"][st.output_var] = answer
                reads.append({
                    "step_id": st.id, "account_id": None, "kind": "agent",
                    "ok": True, "output_var": st.output_var,
                    "text": answer[:1000],
                })
            continue
        try:
            result = await _execute_read_step(automation, st, ctx)
        except Exception as e:  # noqa: BLE001 — one broken read is a
            # fact about the rehearsal, not the end of it. The run
            # itself applies `on_error`; here the report says which
            # read failed and the writes downstream show the hole it
            # left, which is the thing worth seeing before arming.
            result = _skipped_result(st, silent=st.on_error == "skip")
            reads.append({
                "step_id": st.id, "account_id": st.connector_id,
                "ok": False, "on_error": st.on_error,
                "error": str(e)[:200],
            })
        else:
            reads.append({
                "step_id": st.id, "account_id": st.connector_id,
                "ok": True, "count": result.get("count"),
                "text": str(result.get("text") or "")[:1000],
            })
        ctx["steps"][st.id] = result

    writes: list[dict] = []
    for st in vspec.write_steps:
        step_ctx = dict(ctx)
        step_ctx["grant"] = {"target": st.grant_target or {}}
        display = _write_display(st)
        entry = {
            "step_id": st.id,
            "account_id": st.connector_id,
            "what": display.get("what"),
            "target": display.get("target"),
            "params": render_with_ctx(st.params_template, step_ctx),
            "blocked": None,
        }
        if st.tool in _FORBIDDEN_TOOLS:
            # The mail rail. `_run_steps` refuses the whole run here;
            # the rehearsal reports it against the step that carries it.
            entry["blocked"] = ("Automations never send mail — this "
                                "step can only draft.")
        elif not st.grant_id:
            entry["blocked"] = "No permission has been asked for yet."
        elif "{{grant.target." in json.dumps(st.params_template or {}) \
                and not (st.grant_target or {}).get("id"):
            entry["blocked"] = "Nothing says where this should go yet."
        else:
            grant = await reg.fetch_grant(automation.user_id, st.grant_id)
            if grant is not None and grant.get("status") != "approved":
                entry["blocked"] = (
                    f"The permission it needs is "
                    f"{grant.get('status') or 'not approved'}.")
        writes.append(entry)

    return {"rehearsal": True, "sample_event": sample,
            "reads": reads, "writes": writes}


# ── R30: the v3 ledger + narration seams ─────────────────────────────


def _failure_reason(e: Exception) -> str:
    """Map a read-step exception onto the verb dictionary's failure
    reasons. The dispatch error message carries the RPC envelope kind."""
    msg = str(e)
    for token in ("reauth_required", "scope_missing", "provider_down",
                  "rate_limited", "timeout"):
        if token in msg:
            return token
    return "unreachable"


def _reason_code_of(e: Exception, token: str) -> str:
    """The R31 reason code for a failed read (§4.4).

    `_failure_reason` gives a loose token from a substring match on the
    exception text; `account_health.classify` turns that plus the
    provider's own message into the code the string table is keyed on —
    which is where `org_approval_needed` comes from, since a GitHub org
    policy announces itself only in the message body.
    """
    from . import account_health
    return account_health.classify(token, str(e))


def _failure_sentence(connector_id: str, reason_code: str) -> str:
    """The string table's `thread_sentence` for a failed source."""
    if not reason_code:
        return ""
    from . import account_health
    state, _fix = account_health.state_for_reason(reason_code)
    return account_health.sentence_for(
        account_state=state, reason_code=reason_code,
        connector_id=connector_id,
        name=account_health.display_of(connector_id),
    )


def _display_name(connector_id: str) -> str:
    from app.services import automation_verbs as _verbs
    return _verbs.display_name(connector_id) or connector_id or "an account"


def _all_failed_message(failed_sources: list[dict]) -> str:
    """The job row's `user_message` when nothing could be read.

    Names every account, because `Could not reach an account` is the
    string this round exists to delete.
    """
    from . import account_health
    names = [_display_name(f.get("account_id") or "") for f in failed_sources]
    return account_health.names_sentence(names, prefix="could_not_reach") \
        or "Could not reach the accounts it needs."


async def _append_needs_you_turns(
    db, *, thread, automation: Automation, job_id: str,
    failed_sources: list[dict],
) -> None:
    """One `needs_you` turn per failed source (§4.4/§4.5).

    This is R31-05: the only route from a failed run to a fix used to be
    job card → row → act page → About GitHub → sheet, five taps away
    from the sentence that named the problem. The card now sits in the
    thread, next to the run that hit it, carrying the button.

    Each turn is its own try/except: one account whose card cannot be
    written must not cost the others theirs.
    """
    if thread is None:
        return
    from . import account_health, ledger as _ledger
    for src in failed_sources:
        account_id = src.get("account_id") or ""
        if not account_id:
            continue
        try:
            payload = account_health.needs_you_payload(
                account_id=account_id,
                connector_id=account_id,
                name=_display_name(account_id),
                # `timeout` is a TRANSIENT state — it kept the account
                # reading "Connected" with a "Try again" button for a
                # failure nobody had classified. An unclassified failure
                # is `unknown_error`, which offers the probe instead.
                reason_code=src.get("reason_code") or "unknown_error",
            )
            await _ledger.append_turn(
                db, user_id=automation.user_id, thread=thread,
                run_id=job_id, kind="needs_you", payload=payload,
            )
            await account_health.record_use(
                db, user_id=automation.user_id, account_id=account_id,
                ok=False, reason_code=src.get("reason_code") or "",
                # Without this `classify`'s org-approval and scope
                # regexes can never fire here, so the projection every
                # other surface reads said "connected" for an account
                # the run had just failed to read.
                message=str(src.get("message") or ""),
            )
        except Exception as e:  # noqa: BLE001 — see docstring
            logger.warning(
                "[automations] needs_you turn skipped account=%s: %s",
                account_id, e,
            )


async def _append_honest_line(
    db, *, thread, automation: Automation, job_id: str,
    failed_sources: list[dict],
) -> None:
    """"GitHub and Outlook are missing from this — I could not read
    them." (§4.2a)

    Written by the ENGINE, not the narrator: it is a fact about the run
    and it must be there whether or not the narration pass succeeded. A
    brief that silently omits two of five accounts is the failure mode
    the whole partial-run design exists to prevent.
    """
    if thread is None:
        return
    from . import account_health, ledger as _ledger
    names = [_display_name(f.get("account_id") or "")
             for f in failed_sources if f.get("account_id")]
    # C's purpose-written form when it exists; otherwise C's OWN
    # `could_not_reach_*`, which says the same true thing in the same
    # voice. A composes from the table and authors no sentence (§4.4).
    text = (account_health.names_sentence(names, prefix="missing_from_this")
            or account_health.names_sentence(names, prefix="could_not_reach"))
    if not text:
        return
    try:
        await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job_id,
            kind="agent", payload={"text": text},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] honest line skipped: %s", e)


def _write_display(st: ValidatedStep) -> dict:
    """The write's honest display form, snapshotted at staging
    (CONTRACTS-R30 §4.8) — the flush path cannot read platform grants,
    so the pinned target label rides here."""
    from app.services.automation_verbs import turn_action
    from .draft_card import DRAFT_TOOLS
    target = (st.grant_target or {})
    label = target.get("label") or target.get("id")
    is_draft = st.tool in DRAFT_TOOLS
    audience = "you" if is_draft or str(label or "").lower().startswith("dm") \
        else "others"
    act = turn_action(
        st.connector_id, st.tool, kind="write", ok=True,
        target=label, audience=audience,
    )
    return {
        "what": act["action"],
        "target": label,
        "audience": audience,
        "reversible": is_draft,
        # ND-4: lets the outbox flip THIS step's verb to its done form
        # only when the write actually executed (or to failed when not).
        "step_id": st.id,
    }


def _action_record(
    tool: Optional[str], *, ok: bool, ms: int,
    summary: Optional[str] = None,
) -> dict:
    """One executed tool call, in the main chat's vocabulary.

    `label` comes from the SAME table the main chat's job card reads
    (`tool_display.public_step_label`), so "Retrieve latest messages"
    is one string on both surfaces rather than two translations of one
    call. Total: a missing label degrades to the raw tool id's verb
    half, never to an exception — this runs inside ledger appends.
    """
    label = ""
    try:
        from app.agent.tool_display import public_step_label
        label = public_step_label(tool or "") or ""
    except Exception:  # noqa: BLE001 — display only
        label = ""
    return {
        "tool": str(tool or ""),
        "label": label,
        "ok": bool(ok),
        "ms": max(int(ms), 0),
        "summary": (str(summary)[:200] if summary else ""),
    }


async def _append_step_turn(
    db, *, thread, automation: Automation, job_id: str,
    step: ValidatedStep, result: Optional[dict], ms: int, ok: bool,
    reason: Optional[str], turn_index: dict,
) -> None:
    """One step's mechanical turn, whichever kind of step it is."""
    if step.kind == "agent":
        await _append_agent_turn(
            db, thread=thread, automation=automation, job_id=job_id,
            step=step, ms=ms, ok=ok, turn_index=turn_index,
        )
        return
    await _append_read_turn(
        db, thread=thread, automation=automation, job_id=job_id,
        step=step, result=result, ms=ms, ok=ok, reason=reason,
        turn_index=turn_index,
    )


async def _append_agent_turn(
    db, *, thread, automation: Automation, job_id: str,
    step: ValidatedStep, ms: int, ok: bool, turn_index: dict,
) -> None:
    """The mechanical turn for one AGENT step (R38).

    Same `tool` turn as every other step, so the job card counts it and
    the act page renders it with the rest — a step the user cannot see
    is a step that cannot be trusted, and the whole reason this kind
    exists is that a run's judgement should be inspectable.

    Two deliberate choices:

    `tool_kind` is "read" because that field answers exactly one
    question — did this change something the user owns — and an agent
    step changes nothing. It is not a claim that a connector was
    called; `account_id` is empty, which is what every consumer keyed
    on an account already reads (`ledger._write_episodes`,
    `automation_verbs.job_card_label`, the home card's touched/failed
    lists).

    The step's OUTPUT is not here. It is model prose that has passed no
    copy guard, and the thread's voice belongs to the narrator — which
    does receive it (`_narrate_phase1`'s record) and may build on it.
    """
    if thread is None:
        return
    try:
        from . import ledger as _ledger
        from app.services import automation_verbs as _verbs
        act = _verbs.engine_action("think", ok=ok)
        turn = await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job_id,
            kind="tool",
            payload={
                "account_id": "",
                "tool_kind": "read",
                "action": act["action"], "detail": act["detail"],
                "ok": ok, "ms": max(int(ms), 0),
                "steps": [], "items": [], "actions": [],
                "write_ids": [], "rest": "",
            },
        )
        turn_index[step.id] = turn
    except Exception as e:  # noqa: BLE001 — the ledger degrades, the run does not
        logger.debug("[automations] agent turn skipped step=%s: %s",
                     step.id, e)


async def _append_read_turn(
    db, *, thread, automation: Automation, job_id: str,
    step: ValidatedStep, result: Optional[dict], ms: int, ok: bool,
    reason: Optional[str], turn_index: dict,
) -> None:
    """The mechanical tool turn for one read step — engine facts only
    (action/detail/ms/ok/items); the narrator fills whys in place."""
    if thread is None:
        return
    try:
        from . import ledger as _ledger
        from .ledger import item_slots as _ledger_item_slots
        from app.services import automation_verbs as _verbs
        if ok:
            extra = {}
            count = (result or {}).get("count")
            act = _verbs.turn_action(
                step.connector_id, step.tool, kind="read", ok=True,
                count=count if isinstance(count, int) else None,
            )
            # R43 §9 — the display slots beside the ranked line. The
            # line stays the item's `title` (the narrator ranks on it,
            # and the mechanical fallback prints it); `head`/`lede` and
            # who/at/where/hot are what the brief's card opens into, and
            # they come from the step's OWN collect fields, which the
            # rendered line already flattened past recovery.
            _raw = (result or {}).get("raw_fields") or []
            items = []
            for _i, line in enumerate((result or {}).get("lines") or []):
                slots = _ledger_item_slots(
                    _raw[_i] if _i < len(_raw) else None,
                    source=step.connector_id or "",
                )
                head, lede = slots.pop("_title", ""), slots.pop("_sub", "")
                items.append({
                    "title": str(line)[:200], "sub": "", "why": "",
                    **{k: v for k, v in slots.items() if v},
                    **({"head": head} if head else {}),
                    **({"lede": lede} if lede else {}),
                })
            steps_lines = []
        else:
            act = _verbs.failure_action(step.connector_id, reason)
            name = _verbs.display_name(step.connector_id) or "the account"
            items = []
            steps_lines = [
                {"text": f"Asked {name} for what changed", "ok": True},
                {"text": act["detail"].capitalize() or "It did not answer",
                 "ok": False},
            ]
            # ── The line and the button, ON the job card (round 33) ──────
            # The app's `AccountLines` renders one tone-coloured line per
            # account with its own fix button — the closer of the two
            # affordances this round shipped — and it filters on `t.line`,
            # which nothing has ever set. It is the same reason and the
            # same fix the needs-you card carries; one derivation.
            from . import account_health as _ah
            code = _ah.classify(reason or "", "")
            state, fix = _ah.state_for_reason(code)
            extra = {
                "line": _ah.sentence_for(
                    account_state=state, reason_code=code,
                    connector_id=step.connector_id or "",
                    name=_ah.display_of(step.connector_id or ""),
                ) or act["detail"] or f"Could not read {name}.",
                "tone": "success" if state == "connected" else "warning",
                "fix": fix or None,
                "reason_code": code,
            }
        turn = await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job_id,
            kind="tool",
            payload={
                "account_id": step.connector_id or "",
                "tool_kind": "read",
                "action": act["action"], "detail": act["detail"],
                "ok": ok, "ms": max(int(ms), 0),
                "steps": steps_lines, "items": items,
                "write_ids": [], "rest": "",
                # R35: the REAL call, under the sentence. The app's act
                # page had a slot for this and nothing ever filled it —
                # every step read as a claim with no work behind it,
                # which the founder read (reasonably) as "the run never
                # actually calls the connector tools".
                "actions": [_action_record(
                    step.tool, ok=ok, ms=ms,
                    summary=act["detail"] or None,
                )],
                **{k: v for k, v in extra.items() if v},
            },
        )
        turn_index[step.id] = turn
    except Exception as e:  # noqa: BLE001 — the ledger degrades, the run does not
        logger.debug("[automations] read turn skipped step=%s: %s",
                     step.id, e)


async def _recall_facts(db, automation: Automation) -> list[dict]:
    """What the narrator may use to judge: this automation's scoped
    facts + globals (memory v2), falling back to the R29 ledger."""
    try:
        from sqlalchemy import select as _select
        from app.db.models import MemoryFact
        # R38: `id` as tiebreak — two facts learned in the same second
        # used to order arbitrarily, so the same run over the same data
        # could build two different prompts (the narrator is
        # temperature-0 now; the prompt must be stable too).
        rows = list((await db.execute(
            _select(MemoryFact)
            .where(MemoryFact.user_id == automation.user_id)
            .where(MemoryFact.scope.in_((automation.id, "global")))
            .order_by(MemoryFact.learned_at.desc(), MemoryFact.id)
            .limit(24)
        )).scalars())
        if rows:
            return [{"category": r.category, "text": r.text} for r in rows]
        from app.db.models import AutomationFact
        legacy = list((await db.execute(
            _select(AutomationFact)
            .where(AutomationFact.automation_id == automation.id)
            .order_by(AutomationFact.updated_at.desc(), AutomationFact.id)
            .limit(24)
        )).scalars())
        return [{"category": r.category, "text": r.text} for r in legacy]
    except Exception:  # noqa: BLE001
        return []


def _rules_of(automation: Automation) -> list[str]:
    try:
        rules = json.loads(automation.rules_json or "[]")
    except (ValueError, TypeError):
        return []
    return [str(r.get("text") or "").strip()
            for r in rules if isinstance(r, dict) and r.get("text")]


async def _narrate_phase1(
    db, *, automation: Automation, vspec: ValidatedSpecV2, job_id: str,
    thread, tool_turn_by_step: dict, partial: bool,
    failed_sources: Optional[list] = None,
    agent_outputs: Optional[dict] = None,
    terminal_landed: bool = False,
) -> Optional[dict]:
    """Run the narrator and persist the opening agent line + the
    per-item whys (in-place annotates). The result/thinks/draft/close
    land in phase 2, gated on the writes, so a run whose write failed
    is never narrated as one that landed.

    `terminal_landed` says the run's terminal transition has already
    happened (the outbox owns it for any run with a write step) — see
    the progress block below."""
    if thread is None:
        return None
    _reason_by_step = {
        str(f.get("step_id") or ""): str(f.get("reason_code") or "")
        for f in (failed_sources or [])
    }
    try:
        job = await db.get(BuildJob, job_id)
        from . import ledger as _ledger
        if job is None or _ledger.run_kind_of(job) not in ("scheduled",
                                                           "run_now"):
            return None
        # R31-37. This asked "does it write anything that is not a
        # draft?", so posting the brief to Slack made a reads-only
        # brief a change-making run and the founder's morning read
        # "CHANGED YOUR WEEK · 1 item". The question is whether the run
        # changed something the user OWNS — `narrator.vocabulary_for`
        # holds that judgement.
        from .narrator import vocabulary_for
        vocabulary = vocabulary_for(st.tool for st in vspec.write_steps)
        # R36-7: an automation whose spec names its own narration style
        # speaks it — a Newsletter roundup is a digest of newsletters,
        # not a morning triage, however its delivery tool classifies.
        narration_hint = dict((vspec.raw or {}).get("narration") or {})
        if str(narration_hint.get("style") or "") in ("digest", "brief",
                                                      "changes"):
            vocabulary = str(narration_hint["style"])

        # R38 — narration is a VISIBLE step. Both LLM phases together
        # can take most of a minute, and the last progress frame was the
        # final read — so the card sat at "step N of N · In progress"
        # with every step terminal while the brief was being written
        # (the founder's stuck-99% window, all three runs on 29 August).
        # The run's visible total extends by ONE step (one, not two:
        # phase 2 is the same writing, after the flush) and announces
        # itself in the automation's own narration vocabulary.
        #
        # R42 — the FRAME is what a landed terminal suppresses, never
        # the columns. A `running` progress frame for a job that has
        # already reported `Done` walks its own card backwards, so a run
        # with a write step (which terminalizes in the outbox flush
        # above) is announced to nobody. But `progress_step` /
        # `progress_total` are the run's durable record, read back by
        # the terminal frame, the park, the home card's fraction and
        # run-now's 409 sentence — and a finished run that stops one
        # short of its own total draws a card that never fills.
        try:
            from . import run_v3 as _rv3
            from .narrator import writing_sentence
            _n_total = len(vspec.steps) + 1
            if not terminal_landed:
                _sentence = writing_sentence(vocabulary)
                await _ledger.emit_progress(
                    automation.user_id, run_id=job_id,
                    automation_id=automation.id, step=_n_total,
                    total=_n_total, sentence=_sentence,
                    fraction=(_n_total - 1) / _n_total,
                    status="running",
                )
                await _rv3.notify_progress(
                    db, automation=automation, job=job, step=_n_total,
                    total=_n_total, sentence=_sentence,
                    fraction=(_n_total - 1) / _n_total,
                )
            # COMMITTED, not flushed. These columns are a published
            # fact, not part of the narration; left open, the UPDATE
            # holds the `build_jobs` tuple lock across two 8000-token
            # completions, and the outbox's own `_mark_write_step`
            # UPDATE on that row blocks behind it until the 30 s
            # `statement_timeout` cancels it — a run that posted and
            # could never be finalized.
            job.progress_step = _n_total
            job.progress_total = _n_total
            await db.commit()
            # The thread IS being written either way, terminal or not,
            # and the `done` phase lands at the end of the run.
            await _ledger.emit_activity(
                automation.user_id, automation_id=automation.id,
                thread_id=thread.id, run_id=job_id, phase="writing",
            )
        except Exception as e:  # noqa: BLE001 — a frame never fails a run
            logger.debug("[automations] narration step frame skipped: %s", e)
            # The narration continues on THIS session — `_recall_facts`,
            # `narrate_run`, `_apply_annotate` and `append_turn` all run
            # below — so a failed commit must not leave it to inherit a
            # transaction that can no longer be used.
            #
            # And a rollback alone is not enough: it EXPIRES every instance
            # the session holds, so the very next read of `automation.user_id`
            # or `thread.id` raises MissingGreenlet in async SQLAlchemy. The
            # rollback swaps a poisoned transaction for expired objects unless
            # what the rest of this pass reads is reloaded — the same rule
            # `outbox._reopen(db, row)` follows.
            try:
                await db.rollback()
                job = await db.get(BuildJob, job_id)
                automation = await db.get(Automation, automation.id) or automation
                if thread is not None:
                    thread = await db.get(type(thread), thread.id) or thread
                if job is None:
                    return None
            except Exception:  # noqa: BLE001 — the session is unusable; the
                # run's own terminal and the outbox are on other sessions.
                return None

        steps_record = []
        from app.services import automation_verbs as _verbs
        for st in vspec.steps:
            if st.kind == "agent":
                # R38 — the run's own earlier conclusion, handed back to
                # the pass that writes the thread. `detail` carries what
                # it worked out, so the narration can build on it rather
                # than re-deriving the same judgement from the raw items
                # (and reaching a different one, at temperature 0, from a
                # different starting point).
                turn = tool_turn_by_step.get(st.id) or {}
                steps_record.append({
                    "step_ref": st.id,
                    "connector_name": "",
                    "account_id": "",
                    "tool_kind": "agent",
                    "action": turn.get("action") or "",
                    "detail": str(
                        (agent_outputs or {}).get(st.id) or "")[:2000],
                    "ok": bool(turn.get("ok", True)),
                    "failure_reason": None,
                    "items": [],
                    "write": None,
                })
                continue
            if st.mutates:
                d = _write_display(st)
                steps_record.append({
                    "step_ref": st.id,
                    "connector_name": _verbs.display_name(st.connector_id)
                    or st.connector_id,
                    "account_id": st.connector_id,
                    "tool_kind": "write",
                    "action": d["what"], "detail": d.get("target") or "",
                    "ok": True, "failure_reason": None, "items": [],
                    "write": d,
                })
                continue
            turn = tool_turn_by_step.get(st.id) or {}
            steps_record.append({
                "step_ref": st.id,
                "connector_name": _verbs.display_name(st.connector_id)
                or st.connector_id,
                "account_id": st.connector_id or "",
                "tool_kind": "read",
                "action": turn.get("action") or "",
                "detail": turn.get("detail") or "",
                "ok": bool(turn.get("ok", True)),
                # C's narrator contract: `failure_reason` IS the string
                # table's `thread_sentence`, quoted verbatim into the
                # prose. It used to be the tool turn's `detail` — "it
                # did not answer" — a run-row fragment with no account
                # and no fix, which is how a GitHub org-approval refusal
                # was narrated as "GitHub did not respond" and sent the
                # user to fix the wrong thing. Passing a reason CODE
                # here would be just as wrong: the model would have to
                # invent the sentence, which is the improvising this
                # field exists to stop.
                "failure_reason": None if turn.get("ok", True)
                else _failure_sentence(st.connector_id or "",
                                       _reason_by_step.get(st.id, ""))
                or (turn.get("detail") or ""),
                "items": [
                    {"id": it["id"], "title": it.get("title") or "",
                     "sub": it.get("sub") or "",
                     "msgs": it.get("msgs") or []}
                    for it in (turn.get("items") or [])
                ],
                "write": None,
            })
        record = {
            "automation": {"title": automation.name,
                           "mode": (vspec.raw or {}).get("mode") or "auto",
                           # R36-7: the task statement. Every automation
                           # was narrated with nothing but its name, so
                           # every one of them got the same treatment.
                           "description": (
                               (vspec.raw or {}).get("description")
                               or automation.description or ""
                           )},
            "narration": narration_hint,
            "run_kind": _ledger.run_kind_of(job),
            "vocabulary": vocabulary,
            "status": "partial" if partial else "completed",
            "rules": _rules_of(automation),
            "memory_facts": await _recall_facts(db, automation),
            # R42 — the pins, and the user's notes on them, reach the
            # pass that RANKS. `_apply_focus_scope` used to narrow the
            # provider call instead, which is the one thing a pin must
            # never do; the ranking step is where "put Dana first"
            # belongs, and this record was the only surface that could
            # not see it. Same shape `agent_step` already reads:
            # {connector_id: {ids, labels, notes, count, first{...}}}.
            "focus": _focus_render_ctx(vspec.focus),
            "steps": steps_record,
        }
        from .narrator import narrate_run
        outcome = await narrate_run(record)
        drafts = outcome.get("turns") or []
        # R43 §9 — the verdict, not just the log line. A result whose
        # every row was rejected used to be persisted anyway and served
        # as the user's brief: five headings, "0 items" under each,
        # nothing to open. Dropping it hands the run to
        # `close_ledger`'s completeness net, which owes the thread one
        # result turn and writes the honest mechanical one instead.
        unservable = _recheck_unservable(
            drafts, record, set(outcome.get("unservable") or ()))
        if outcome.get("problems"):
            logger.log(
                logging.WARNING if unservable else logging.INFO,
                "[automations] narration problems on %s (dropping %d "
                "result draft(s)): %s",
                job_id[:8], len(unservable), outcome["problems"][:5],
            )
        held: list[dict] = []
        # A resumed run already has its opening line — never repeat it.
        existing = await _ledger.run_turns(db, run_id=job_id)
        opened = any(t["kind"] == "agent" for t in existing)
        for n, d in enumerate(drafts):
            kind = d.get("kind")
            if kind == "result" and n in unservable:
                continue
            if kind == "annotate":
                await _apply_annotate(db, automation, tool_turn_by_step, d)
            elif kind == "agent" and not opened:
                opened = True
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="agent",
                    payload={"text": d.get("text") or ""},
                )
            else:
                held.append(d)
        return {"held": held, "vocabulary": vocabulary}
    except Exception as e:  # noqa: BLE001 — narration must not kill the run
        logger.warning("[automations] narration phase 1 skipped: %s", e)
        return None


def _recheck_unservable(drafts: list, record: dict, reported: set) -> set:
    """Which result drafts really cannot be served, judged on the drafts
    in hand rather than on how the narrator stopped.

    R43 repair (finding 15). `narrate_run` reports EVERY result
    unservable when its RETRY throws — a ReadTimeout against the model
    is the ordinary case — but the drafts it hands back are then
    attempt ONE's, which it validated on the previous pass and which
    `unservable_results` may well have KEPT: its own rule is that a
    result with some good rows survives, because a bad `tag` on one
    line is not a reason to replace a real ranking with "I could not
    rank them". So a complete five-tier brief was thrown away and the
    mechanical fallback served over it, and nothing else in the round
    could tell "the retry never happened" from "the retry rejected it".

    Idempotent on every other path: the same two pure functions over
    the same drafts and the same record already produced the number.
    """
    if not drafts or not reported:
        return set(reported)
    from .narrator import unservable_results, validate_drafts
    return unservable_results(drafts, validate_drafts(drafts, record))


async def _apply_annotate(
    db, automation: Automation, tool_turn_by_step: dict, draft: dict,
) -> None:
    """Fill item whys / msg whys / rest into the persisted tool turn
    the annotate addresses (matched by minted item ids)."""
    from app.db.models import AutomationTurn
    from . import ledger as _ledger
    step_ref = draft.get("step_ref")
    turn = None
    for sid, t in tool_turn_by_step.items():
        if sid == step_ref or t.get("id") == step_ref:
            turn = t
            break
    if turn is None:
        return
    row = await db.get(AutomationTurn, turn["id"])
    if row is None:
        return
    body = json.loads(row.payload_json)
    by_id = {it.get("id"): it for it in body.get("items") or []}
    for ann in draft.get("items") or []:
        it = by_id.get(ann.get("id"))
        if it is None:
            continue
        if ann.get("why"):
            it["why"] = str(ann["why"])[:400]
        for m in ann.get("msgs") or []:
            idx = m.get("idx")
            msgs = it.get("msgs") or []
            if isinstance(idx, int) and 0 <= idx < len(msgs) and m.get("why"):
                msgs[idx]["why"] = str(m["why"])[:400]
    if draft.get("rest"):
        body["rest"] = str(draft["rest"])[:400]
    row.payload_json = json.dumps(body, default=str)
    await db.commit()
    turn.update(body)
    # R31 §4.1: every automation frame carries automation_id — the
    # app-level bridge routes on it (R38: this one was missing it).
    await _ledger._broadcast(automation.user_id, {
        "type": "automation.turn",
        "automation_id": automation.id,
        "thread_id": row.thread_id,
        "run_id": row.run_id,
        "turn": _ledger._serialize_row(row),
    })


async def _narrate_phase2(
    db, *, automation: Automation, job_id: str, thread,
    narration: Optional[dict], writes_ok: bool,
) -> None:
    """Post-write narration: result (only when every write landed),
    thinks, the draft (unless the outbox already appended one), the
    closing line."""
    if thread is None or not narration:
        return
    try:
        from . import ledger as _ledger
        existing = await _ledger.run_turns(db, run_id=job_id)
        has_draft = any(t["kind"] == "draft" for t in existing)
        # R43 §9 — the items the ranked rows point at ride the result
        # turn itself, so the BROADCAST carries them too. Attaching them
        # only in `close_ledger` would leave the live card counting rows
        # until the thread was re-read.
        tool_turns = [t for t in existing if t["kind"] == "tool"]
        item_ix = _ledger.item_index(tool_turns)
        reason = _ledger.empty_reason(_ledger._failed_account_names(
            [t for t in tool_turns
             if t.get("tool_kind") == "read" and t.get("account_id")]))
        for d in narration.get("held") or []:
            kind = d.get("kind")
            if kind == "result":
                if not writes_ok:
                    continue
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="result",
                    payload={
                        "title": d.get("title") or "",
                        "vocabulary": d.get("vocabulary")
                        or narration.get("vocabulary") or "brief",
                        "groups": _ledger.attach_items(
                            d.get("groups") or [], item_ix, reason=reason),
                    },
                )
            elif kind == "think":
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="think",
                    payload={"text": d.get("text") or ""},
                )
            elif kind == "draft":
                if has_draft:
                    continue
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="draft",
                    payload={
                        "text": d.get("text") or "",
                        "target": {
                            "account_id": d.get("target_account_id") or "",
                            "ref": d.get("target_ref"),
                        },
                        "sent_at": None,
                    },
                )
                has_draft = True
            elif kind == "agent":
                if not writes_ok:
                    continue
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="agent",
                    payload={"text": d.get("text") or ""},
                )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] narration phase 2 skipped: %s", e)


async def _deliver_run_brief(
    db, *, automation: Automation, vspec: ValidatedSpecV2, job_id: str,
    thread, source: Optional[ValidatedSource], idem_prefix: str,
) -> None:
    """Hand the run's finished brief to `deliver`.

    Best-effort by contract: the run is terminal, its thread already
    carries the brief, and nothing about fanning it out may unwind that.
    A run with no result turn delivers nothing — there is no brief, and
    posting the run's raw steps to Slack under the automation's name is
    the disagreement between the post and the card this round removes.
    """
    if thread is None:
        return
    try:
        from . import deliver as _deliver
        from . import ledger as _ledger
        job = await db.get(BuildJob, job_id)
        if job is None or _ledger.run_kind_of(job) not in ("scheduled",
                                                           "run_now"):
            return
        results = [t for t in await _ledger.run_turns(db, run_id=job_id)
                   if t["kind"] == "result"]
        if not results:
            return
        await _deliver.deliver_brief(
            db, automation=automation, job_id=job_id, thread=thread,
            groups=results[-1].get("groups") or [],
            title=results[-1].get("title") or "",
            delivery=vspec.delivery, source=source,
            idem_prefix=idem_prefix,
        )
    except Exception as e:  # noqa: BLE001 — delivery never fails a run
        logger.warning("[automations] delivery skipped job=%s: %s",
                       job_id[:8], e)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def _close_ledger_after_narration(
    db, *, automation: Automation, job_id: str,
) -> None:
    """Close the v3 ledger the run's terminal deliberately left open,
    and release the claim that made it wait.

    ORDER, and it is the whole function: the job's status is read while
    `narration_pending` is still set, so a terminal that landed at any
    point since the flush — ours, or the background loop's, mid-
    narration — is one `on_terminal` declined to close and this run
    still owes. The flag is cleared LAST, and unconditionally: a run
    whose write is still staged (an undo took the claim, the send went
    to backoff) has no terminal yet, and a flag left set would make the
    terminal that eventually lands skip its close forever.

    That leaves exactly one closer in every case, and the run that
    closes is the one that has already narrated. Best-effort by the same
    contract `close_ledger` itself carries: the run is terminal and
    nothing here may un-finalize it.
    """
    from . import ledger as _ledger
    try:
        job = await db.get(BuildJob, job_id)
        # From the DATABASE, not the identity map. `db.get` answers from the
        # session's own cache, and the terminal we are deciding about is
        # written by ANOTHER session — the outbox's. A retryable write returns
        # `staged` with no terminal and `flush_loop` retries ~10s later, i.e.
        # while these two LLM calls are running: without the refresh this
        # reads a stale `running`, declines, and clears the flag — and
        # `on_terminal` had already declined because the flag was set. Nobody
        # closes the ledger and nothing ever comes back for it.
        if job is not None:
            await db.refresh(job)
        if job is not None and job.status not in (
                "queued", "running", "waiting_on_user"):
            await _ledger.close_ledger(
                db, user_id=automation.user_id, job=job,
                automation=automation,
            )
    except Exception as e:  # noqa: BLE001 — a close never fails a run
        logger.warning("[automations] deferred ledger close skipped "
                       "job=%s: %s", job_id[:8], e)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
    try:
        await merge_job_config(db, job_id, narration_pending=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] narration claim not released "
                       "job=%s: %s", job_id[:8], e)


# ── §4.2a — per-source resume ────────────────────────────────────────

async def resume_source(
    db, *, automation: Automation, job_id: str, account_id: str,
) -> dict:
    """Re-run ONE source's step of an existing run and merge the result.

    CONTRACTS-R31 §4.2a, and the difference between a fix that works
    and a fix that means "start over". On 26 August the only route from
    a broken account to a repaired brief was to fix the connector and
    wait for tomorrow's run — the four accounts that HAD answered were
    re-read from scratch, or not at all.

    What happens instead: the failed step runs again, alone; a
    `RECONNECTED` note and its catch-up tool turn are appended; the
    run's result turn is REPLACED IN PLACE (`ledger.replace_turn`, so
    `GET /thread` returns the merged version and no second brief
    appears under the first); and the run's status is recomputed —
    `partial` becomes `completed` when nothing is left failing.

    Returns `{"resumed": bool, "status": str, "reason": str}`. Never
    raises: this runs from a connector callback and a hook that throws
    loses the reconnect the user just performed.
    """
    from . import account_health, ledger as _ledger, run_v3
    from .service import parse_spec_live
    from app.services import automation_verbs as _verbs
    import time as _time

    job = await db.get(BuildJob, job_id)
    if job is None:
        return {"resumed": False, "reason": "no_run"}
    cfg = _ledger._cfg_of(job)
    failed = list(cfg.get("accounts_failed") or [])
    if account_id not in failed:
        return {"resumed": False, "reason": "not_failed"}

    # §4.2a: a run older than its own cadence is not worth merging into
    # — the reads would be about a day that has passed. The caller
    # fires a fresh `run_now` instead.
    started = job.created_at or datetime.utcnow()
    if (datetime.utcnow() - started) > timedelta(hours=24):
        return {"resumed": False, "reason": "too_old"}

    vspec = await parse_spec_live(automation)
    from .spec_v2 import ValidatedSpecV2
    if not isinstance(vspec, ValidatedSpecV2):
        return {"resumed": False, "reason": "v1_not_supported"}
    step = next(
        (st for st in vspec.steps
         if not st.mutates and st.connector_id == account_id),
        None,
    )
    if step is None:
        # R35: the founder's Slack "Try again". A connector whose only
        # step is the WRITE has no read to re-run, and this returned
        # `no_step` — a 409 the app rendered as "Nothing ran" while the
        # failed outbox row sat there holding the exact payload, tool
        # and grant the retry needs. Re-stage that row and flush it
        # inline; the outbox's own machinery appends the write turn,
        # flips the step and aggregates the terminal.
        return await _resume_write_source(
            db, automation=automation, job=job, job_id=job_id,
            account_id=account_id, cfg=cfg, failed=failed,
        )

    thread = await _ledger.thread_for(db, automation.id)
    if thread is not None:
        await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job_id,
            kind="note",
            payload={"stamp": "reconnected",
                     "at": datetime.utcnow().isoformat() + "Z"},
        )

    ctx: dict = {"steps": {}, "var": {}, "event": {}}
    t0 = _time.monotonic()
    ok, reason = True, ""
    try:
        result = await _execute_read_step(automation, step, ctx)
    except Exception as e:  # noqa: BLE001
        ok, result = False, None
        reason = _reason_code_of(e, _failure_reason(e))

    await _append_read_turn(
        db, thread=thread, automation=automation, job_id=job_id,
        step=step, result=result,
        ms=int((_time.monotonic() - t0) * 1000), ok=ok,
        reason=_failure_reason(Exception(reason)) if not ok else None,
        turn_index={},
    )
    await account_health.record_use(
        db, user_id=automation.user_id, account_id=account_id, ok=ok,
        reason_code=reason,
    )

    if not ok:
        await db.commit()
        return {"resumed": True, "status": "partial", "reason": reason}

    # The source is fixed: drop it from the run's failed list and
    # recompute the terminal.
    still_failed = [a for a in failed if a != account_id]
    sources = [
        dict(f) for f in (cfg.get("failed_sources") or [])
        if f.get("account_id") != account_id
    ]
    touched = list(cfg.get("accounts_touched") or [])
    if account_id not in touched:
        touched.append(account_id)
    await merge_job_config(
        db, job_id, accounts_failed=still_failed,
        failed_sources=sources, accounts_touched=touched,
    )

    await _replace_result_turn(
        db, automation=automation, thread=thread, job_id=job_id,
        still_failed=still_failed,
    )

    if not still_failed and (job.outcome or "") == "partial":
        # Every source is in now. The run is what it would have been.
        row = await db.get(BuildJob, job_id)
        if row is not None:
            row.outcome = "sent"
            await db.commit()
        await _record_health(db, automation.id, ok=True, error=None,
                             ran=True, clean=True)

    try:
        await run_v3.notify_resume(db, job_id=job_id)
    except Exception as e:  # noqa: BLE001
        logger.debug("[automations] resume notify skipped: %s", e)
    await _ledger.emit_updated(
        db, automation.user_id, automation_id=automation.id,
    )
    return {
        "resumed": True,
        "status": "completed" if not still_failed else "partial",
        "reason": "",
    }


async def _resume_write_source(
    db, *, automation: Automation, job, job_id: str,
    account_id: str, cfg: dict, failed: list,
) -> dict:
    """Re-run ONE failed write of an existing run (R35).

    The outbox row is the durable intent — payload, tool, grant — so a
    retry is re-staging that exact row and flushing it inline. Its own
    machinery then appends the write turn (with the real call under
    it), flips the step, records health and aggregates the terminal;
    the only thing left to this function is the run's failed-list
    bookkeeping and the terminal flip `_finalize_job`'s guard refuses
    on an already-terminal row.
    """
    from sqlalchemy import select as _select
    from app.db.models import AutomationOutbox
    from . import ledger as _ledger, run_v3
    from .outbox import flush_row_when_due

    rows = (await db.execute(
        _select(AutomationOutbox)
        .where(AutomationOutbox.job_id == job_id)
        .where(AutomationOutbox.connector_id == account_id)
        .where(AutomationOutbox.status == "failed")
        .order_by(AutomationOutbox.created_at.desc())
    )).scalars().all()
    row = rows[0] if rows else None
    if row is None:
        return {"resumed": False, "reason": "no_step"}

    thread = await _ledger.thread_for(db, automation.id)
    if thread is not None:
        await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job_id,
            kind="note",
            payload={"stamp": "tried",
                     "at": datetime.utcnow().isoformat() + "Z"},
        )

    # A fresh 3-attempt budget: the user asked, and the old count is
    # about a different moment at the provider.
    row.status = "staged"
    row.attempts = 0
    row.execute_after = datetime.utcnow()
    row.next_attempt_at = datetime.utcnow()
    await db.commit()
    status = await flush_row_when_due(db, row.id)

    if status != "executed":
        # Failed again (a fresh turn + card were appended by the flush
        # path) or went back to backoff — either way the run is still
        # owed this write, and saying so is the honest answer.
        return {"resumed": True, "status": "partial", "reason": ""}

    still_failed = [a for a in failed if a != account_id]
    sources = [
        dict(f) for f in (cfg.get("failed_sources") or [])
        if f.get("account_id") != account_id
    ]
    await merge_job_config(
        db, job_id, accounts_failed=still_failed, failed_sources=sources,
    )
    if not still_failed:
        # `_finalize_run` already tried to close the run as sent, and
        # `_finalize_job`'s guard rightly refused to move a terminal
        # row — the flip is this function's to make, exactly like the
        # read path's partial→sent.
        jrow = await db.get(BuildJob, job_id)
        if jrow is not None and jrow.status == "failed" \
                and (jrow.outcome or "") == "write_failed":
            partial = bool((jrow.config_json or {}).get("steps_partial"))
            jrow.status = "completed"
            jrow.outcome = "partial" if partial else "sent"
            jrow.error_class = None
            jrow.user_message = None
            await db.commit()
            await _record_health(db, automation.id, ok=True, error=None,
                                 ran=True, clean=not partial)

    try:
        await run_v3.notify_resume(db, job_id=job_id)
    except Exception as e:  # noqa: BLE001
        logger.debug("[automations] resume notify skipped: %s", e)
    await _ledger.emit_updated(
        db, automation.user_id, automation_id=automation.id,
    )
    return {
        "resumed": True,
        "status": "completed" if not still_failed else "partial",
        "reason": "",
    }


async def _replace_result_turn(
    db, *, automation: Automation, thread, job_id: str,
    still_failed: list,
) -> None:
    """Rewrite the run's honest line where it already sits (§4.5).

    The RESULT turn itself is the narrator's and is not re-narrated
    here — re-running the ranking for one extra account would rewrite
    judgements the user has already read. What is replaced is the line
    that says what is MISSING, because that is the sentence the merge
    makes false.
    """
    if thread is None:
        return
    from . import account_health, ledger as _ledger
    turns = await _ledger.run_turns(db, run_id=job_id)
    target = None
    for t in turns:
        if t.get("kind") != "agent":
            continue
        text = t.get("text") or ""
        if "missing from this" in text or "Could not reach" in text:
            target = t
    if target is None:
        return
    names = [_display_name(a) for a in still_failed if a]
    if names:
        text = (account_health.names_sentence(
                    names, prefix="missing_from_this")
                or account_health.names_sentence(
                    names, prefix="could_not_reach"))
    else:
        text = account_health.form("reconnected_just_now") \
            or "Everything it needed is in this now."
    try:
        await _ledger.replace_turn(
            db, user_id=automation.user_id, thread=thread,
            turn_id=target["id"], kind="agent",
            payload={"text": text}, run_id=job_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] result merge skipped: %s", e)
