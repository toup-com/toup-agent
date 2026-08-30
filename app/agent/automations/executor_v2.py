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
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError

from app.db.models import (
    Automation, AutomationEvent, AutomationOutbox, BuildJob,
    AUTOMATION_OUTBOX_UNDO_WINDOW_S, AUTOMATION_RUN_CAP_S,
)
from app.agent.job_runner import JobRunner, TaskSpec
from app.agent import job_steps
from .spec import (
    focus_render_ctx as _focus_render_ctx,
    render_value, render_with_ctx, resolve_path,
)
from .spec_v2 import ValidatedSpecV2, ValidatedSource, ValidatedStep
from .executor import _FORBIDDEN_TOOLS, _finalize_job, _record_health
from .session import on_run_created
from . import agent_step as _agent_step
from . import memory as engine_memory
from . import registry as reg

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
) -> list[AutomationEvent]:
    """Insert-or-skip each observed item, dedupe-keyed as
    "<source_id>:<value>" so two sources can never collide in the
    per-automation UNIQUE gate. Payloads are trimmed to the declared
    event fields plus `_source`."""
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
        )
        try:
            async with db.begin_nested():
                db.add(event)
                await db.flush()
            fresh.append(event)
        except IntegrityError:
            pass
    await db.commit()
    return fresh


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
            item_fields = {
                name: resolve_path(item, path)
                for name, path in (collect.get("fields") or {}).items()
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


def _apply_focus_scope(
    connector_id: str, tool: str, params: dict, pins: list,
) -> dict:
    """Make the user's pins steer the read (R39, founder P10).

    Until now an accepted pin reached a run only as a `{{focus.*}}`
    render root that no compiled step ever referenced — the canvas
    said "it starts at #all-toup" and the run read the whole account
    anyway. This narrows a read step's params by the pins on its own
    connector, the same vocabulary `contents.py` reads by:

      - query languages COMPOSE (gmail/outlook person+label terms,
        jira project scope) — the spec's own filter still applies;
      - single-target reads FILL an empty target and OVERRIDE a set
        one (slack channel, teams chat, github repo) — the pin is the
        user's later word than the compiled spec.

    Pure and total: no pins, unknown tool, or malformed params → the
    params come back untouched.
    """
    if not pins or not isinstance(params, dict):
        return params
    out = dict(params)
    by_kind: dict[str, list] = {}
    for p in pins:
        if isinstance(p, dict) and p.get("id"):
            by_kind.setdefault(str(p.get("kind") or ""), []).append(p)
    try:
        if tool == "gmail__list_messages":
            terms = [f"from:{p['id']}" for p in by_kind.get("person", [])]
            terms += [f"label:{p['id']}" for p in by_kind.get("label", [])
                      + by_kind.get("folder", [])]
            if terms:
                scoped = terms[0] if len(terms) == 1 \
                    else "(" + " OR ".join(terms) + ")"
                q = str(out.get("query") or "").strip()
                out["query"] = f"{q} {scoped}".strip()
        elif tool == "outlook__list_messages":
            froms = [f"from:{p['id']}" for p in by_kind.get("person", [])]
            if froms:
                scoped = " OR ".join(froms)
                q = str(out.get("query") or "").strip()
                out["query"] = f"({q}) AND ({scoped})" if q else scoped
        elif tool == "slack__read_messages":
            chans = by_kind.get("channel", []) + by_kind.get("thread", [])
            if chans:
                out["channel"] = str(chans[0]["id"])
        elif tool == "teams__read_chat_messages":
            chats = by_kind.get("thread", []) + by_kind.get("channel", [])
            if chats:
                out["chat_id"] = str(chats[0]["id"])
        elif tool == "jira__search_issues":
            projects = [str(p["id"]) for p in by_kind.get("project", [])]
            if projects:
                keys = ", ".join(f'"{k}"' for k in projects)
                jql = str(out.get("jql") or "").strip()
                out["jql"] = (f"project in ({keys}) AND ({jql})"
                              if jql else f"project in ({keys})")
        elif tool == "github__list_issues":
            repos = [str(p["id"]) for p in by_kind.get("repo", [])
                     if "/" in str(p.get("id") or "")]
            if repos:
                owner, repo = repos[0].split("/", 1)
                out["owner"], out["repo"] = owner, repo
    except Exception:  # noqa: BLE001 — a pin must never break a read
        return params
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
    result = await reg.dispatch_via_platform(
        automation.user_id,
        connector_id=step.connector_id,
        tool_name=step.tool,
        tool_input=params,
        automation_id=automation.id,
    )
    if result.get("kind") != "ok":
        raise RuntimeError(
            f"step {step.id!r} failed: {result.get('kind')}: "
            f"{str(result.get('message') or '')[:200]}"
        )
    try:
        content = json.loads(result.get("content") or "{}")
    except (ValueError, TypeError):
        content = {}
    if not isinstance(content, dict):
        content = {}
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

    # R30 narration phase 1 (pre-write): the opening line, item whys,
    # think turns, the draft's text. Result + close land in phase 2 so
    # the thread never claims a change before the write executed.
    narration = await _narrate_phase1(
        db, automation=automation, vspec=vspec, job_id=job_id,
        thread=thread, tool_turn_by_step=tool_turn_by_step,
        partial=partial, failed_sources=failed_sources,
        agent_outputs=agent_outputs,
    )

    from .outbox import flush_row_when_due
    statuses = []
    for row in rows:
        statuses.append(await flush_row_when_due(db, row.id))

    await _narrate_phase2(
        db, automation=automation, job_id=job_id, thread=thread,
        narration=narration,
        writes_ok=not any(s == "failed" for s in statuses),
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


async def poll_and_run_v2(
    db, automation: Automation, vspec: ValidatedSpecV2,
    source: ValidatedSource,
) -> dict:
    items = await _poll_once_v2(automation, source)
    fresh = await ingest_items_v2(db, automation, source, items)
    ran = 0
    failed = 0
    for event in fresh:
        status = await run_event_v2(db, automation, vspec, source, event)
        if status == "failed":
            failed += 1
        elif status == "run":
            ran += 1
    if failed == 0:
        # See `_record_health`: a poll with nothing fresh is connector
        # health, not a run. Stamping it as a run made the health object
        # claim runs the ledger had never heard of (ND-25).
        await _record_health(db, automation.id, ok=True, error=None,
                             ran=ran > 0)
    return {"observed": len(items), "fresh": len(fresh),
            "ran": ran, "failed": failed}


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
        from app.services import automation_verbs as _verbs
        if ok:
            extra = {}
            count = (result or {}).get("count")
            act = _verbs.turn_action(
                step.connector_id, step.tool, kind="read", ok=True,
                count=count if isinstance(count, int) else None,
            )
            items = [
                {"title": str(line)[:200], "sub": "", "why": ""}
                for line in ((result or {}).get("lines") or [])
            ]
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
) -> Optional[dict]:
    """Run the narrator and persist what may precede the writes: the
    opening agent line + the per-item whys (in-place annotates). The
    result/thinks/draft/close land in phase 2, after the writes, so
    the thread never claims a change before it happened."""
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
        try:
            from . import run_v3 as _rv3
            from .narrator import writing_sentence
            _n_total = len(vspec.steps) + 1
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
            # The same columns the read loop stamps — the terminal
            # frame, the park, the home card's fraction and run-now's
            # 409 sentence all read them back.
            job.progress_step = _n_total
            job.progress_total = _n_total
            await db.flush()
            await _ledger.emit_activity(
                automation.user_id, automation_id=automation.id,
                thread_id=thread.id, run_id=job_id, phase="writing",
            )
        except Exception as e:  # noqa: BLE001 — a frame never fails a run
            logger.debug("[automations] narration step frame skipped: %s", e)

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
            "steps": steps_record,
        }
        from .narrator import narrate_run
        outcome = await narrate_run(record)
        drafts = outcome.get("turns") or []
        if outcome.get("problems"):
            logger.info("[automations] narration problems on %s: %s",
                        job_id[:8], outcome["problems"][:5])
        held: list[dict] = []
        # A resumed run already has its opening line — never repeat it.
        existing = await _ledger.run_turns(db, run_id=job_id)
        opened = any(t["kind"] == "agent" for t in existing)
        for d in drafts:
            kind = d.get("kind")
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
                        "groups": d.get("groups") or [],
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
