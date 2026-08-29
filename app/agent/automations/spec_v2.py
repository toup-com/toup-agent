"""AutomationSpec v2 — multi-source triggers, multi-connector steps.

Round 28. `spec.validate_spec` dispatches here on `version: 2`; v1
specs never enter this module. The v2 shape (CONTRACTS-R28.md §1):

  - `trigger.sources[]` — up to 4 independent firing lanes (push /
    poll / schedule), each push/poll lane with its OWN dedupe key.
    Any lane firing runs the SAME steps once per fresh event.
  - `steps[]` — up to 8 connector tool calls. Reads run inline via
    the platform dispatch RPC and can `collect` their items into
    `{{steps.<id>.text}}` / `{{steps.<id>.count}}`; writes are staged
    through the outbox exactly like v1, one grant per write step,
    pinned target re-verified by the dispatcher at call time.
  - `variables` — `{{var.<name>}}` placeholders; every reference must
    be declared (templates fill them at setup).

Hand-rolled validation for the same reason as v1: the security
boundary must not rest on a transitively-present jsonschema package.
`SpecError` carries every problem at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from app.db.models.automation import AUTOMATION_TRIGGER_MODES

from .spec import (
    SpecError, _err, effective_every_floor, effective_poll_floor,
)

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")

# Context roots the renderer owns — a step id equal to one of these
# would shadow the namespace it renders from.
RESERVED_IDS = frozenset({
    "event", "source", "var", "steps", "grant", "memory", "item",
})

MAX_SOURCES = 4
MAX_STEPS = 8
MAX_WRITE_STEPS = 3
COLLECT_LIMIT_DEFAULT = 10
COLLECT_LIMIT_MAX = 25

_TOP_KEYS = {"version", "name", "description", "mode", "variables",
             "trigger", "steps", "narration"}
_SOURCE_KEYS = {"id", "mode", "connector_id", "event", "params",
                "poll_interval_s", "schedule", "filter", "dedupe_key"}
# `grant_target` is SYSTEM-written per write step (arm snapshots the
# approved grant's pinned target) — accepted on re-validation, never
# authored.
_STEP_KEYS = {"id", "connector_id", "tool", "params", "collect",
              "on_error", "grant_id", "grant_target"}
_SCHEDULE_KEYS = {"cron_local", "at", "every_s"}
_COLLECT_KEYS = {"items_path", "fields", "format", "limit",
                 "empty_text", "join"}

_VAR_REF_RE = re.compile(r"\{\{\s*var\.([a-zA-Z0-9_]+)\s*\}\}")


@dataclass(frozen=True)
class ValidatedSource:
    id: str
    mode: str                       # push | poll | schedule
    connector_id: Optional[str]
    event: Optional[str]
    params: dict
    poll_interval_s: Optional[int]
    schedule: Optional[dict]
    filter_rules: dict
    dedupe_key_field: Optional[str]
    event_spec: Optional[dict]


@dataclass(frozen=True)
class ValidatedStep:
    id: str
    connector_id: str
    tool: str
    params_template: dict
    mutates: bool
    grant_id: Optional[str]
    grant_target: dict
    collect: Optional[dict]
    on_error: str                   # fail | skip | continue


@dataclass(frozen=True)
class ValidatedSpecV2:
    """The parsed, validated v2 spec. `raw` is the canonical dict to
    persist. Exposes the v1 attribute names the lifecycle service
    reads (trigger_mode / connector ids), so callers branch only where
    behavior actually differs."""

    raw: dict
    name: str
    mode: str                       # auto | confirm
    variables: dict
    sources: tuple
    steps: tuple

    version: int = 2

    @property
    def trigger_mode(self) -> str:
        """Denormalized column value: the single source's mode, or
        'multi' when several lanes exist."""
        if len(self.sources) == 1:
            return self.sources[0].mode
        return "multi"

    @property
    def trigger_connector_id(self) -> Optional[str]:
        for s in self.sources:
            if s.connector_id:
                return s.connector_id
        return None

    @property
    def action_connector_id(self) -> str:
        for st in self.steps:
            if st.mutates:
                return st.connector_id
        return self.steps[0].connector_id if self.steps else ""

    @property
    def write_steps(self) -> tuple:
        return tuple(st for st in self.steps if st.mutates)

    def source_by_id(self, source_id: Optional[str]) -> Optional[ValidatedSource]:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def schedule_source(self) -> Optional[ValidatedSource]:
        for s in self.sources:
            if s.mode == "schedule":
                return s
        return None


def _validate_source(
    idx: int,
    src: Any,
    registry: dict[str, dict],
    errors: list[dict],
) -> Optional[ValidatedSource]:
    fld = f"trigger.sources[{idx}]"
    if not isinstance(src, dict):
        _err(errors, "bad_source", fld, "each source must be an object")
        return None
    for k in src:
        if k not in _SOURCE_KEYS:
            _err(errors, "unknown_field", f"{fld}.{k}",
                 f"unknown source field {k!r}")

    sid = src.get("id")
    if not isinstance(sid, str) or not _ID_RE.match(sid):
        _err(errors, "bad_source_id", f"{fld}.id",
             "source id must match ^[a-z][a-z0-9_]{0,23}$")
        sid = f"s{idx}"

    mode = src.get("mode")
    if mode not in AUTOMATION_TRIGGER_MODES:
        _err(errors, "bad_trigger_mode", f"{fld}.mode",
             f"source mode must be one of {sorted(AUTOMATION_TRIGGER_MODES)}")
        mode = "schedule"

    params = src.get("params") or {}
    if not isinstance(params, dict):
        _err(errors, "bad_params", f"{fld}.params", "params must be an object")
        params = {}
    filter_rules = src.get("filter") or {}
    if not isinstance(filter_rules, dict):
        _err(errors, "bad_filter", f"{fld}.filter", "filter must be an object")
        filter_rules = {}

    connector = src.get("connector_id")
    event_key = src.get("event")
    poll_interval: Optional[int] = None
    schedule: Optional[dict] = None
    event_spec: Optional[dict] = None
    dedupe_field: Optional[str] = None

    if mode in ("push", "poll"):
        cap = registry.get(connector) if isinstance(connector, str) else None
        if cap is None:
            _err(errors, "unknown_connector", f"{fld}.connector_id",
                 f"connector {connector!r} is not automatable "
                 f"(known: {sorted(registry)})")
        else:
            if mode == "push" and not cap.get("push"):
                _err(errors, "push_unavailable", f"{fld}.mode",
                     f"{connector} has no push path — use poll")
            if mode == "poll" and not cap.get("poll"):
                _err(errors, "poll_unavailable", f"{fld}.mode",
                     f"{connector} does not support polling")
            events = {e["key"]: e for e in cap.get("events", [])}
            if not isinstance(event_key, str) or event_key not in events:
                _err(errors, "unknown_event", f"{fld}.event",
                     f"event {event_key!r} not declared by {connector} "
                     f"(known: {sorted(events)})")
            else:
                event_spec = events[event_key]
                missing = [
                    p for p in (event_spec.get("params_required") or [])
                    if p not in params
                ]
                if missing:
                    _err(errors, "missing_event_param", f"{fld}.params",
                         f"event {event_key!r} requires params "
                         f"{sorted(missing)}")
        if mode == "poll":
            floor = effective_poll_floor((cap or {}).get("floor_s"))
            raw_iv = src.get("poll_interval_s", floor)
            if not isinstance(raw_iv, int) or isinstance(raw_iv, bool):
                _err(errors, "bad_interval", f"{fld}.poll_interval_s",
                     "poll_interval_s must be an integer number of seconds")
            elif raw_iv < floor:
                _err(errors, "interval_below_floor", f"{fld}.poll_interval_s",
                     f"poll_interval_s={raw_iv} is below the floor of "
                     f"{floor}s for {connector}")
            else:
                poll_interval = raw_iv

        dk = src.get("dedupe_key")
        if not isinstance(dk, str) or not dk.strip():
            _err(errors, "missing_dedupe_key", f"{fld}.dedupe_key",
                 "push/poll sources require a dedupe_key "
                 "(\"event.<field>\")")
        else:
            dk = dk.strip()
            if not dk.startswith("event."):
                _err(errors, "bad_dedupe_key", f"{fld}.dedupe_key",
                     "dedupe_key must be an \"event.<field>\" reference")
            else:
                fname = dk[len("event."):]
                if event_spec is not None:
                    known = set((event_spec.get("fields") or {}).keys())
                    known.add(event_spec.get("dedupe_field") or "")
                    if fname not in known:
                        _err(errors, "bad_dedupe_key", f"{fld}.dedupe_key",
                             f"{fname!r} is not a field of event "
                             f"{event_key!r} (known: {sorted(known - {''})})")
                dedupe_field = fname
    elif mode == "schedule":
        schedule = src.get("schedule")
        if not isinstance(schedule, dict) or not schedule:
            _err(errors, "missing_schedule", f"{fld}.schedule",
                 "schedule mode requires a schedule object")
            schedule = None
        else:
            unknown = set(schedule) - _SCHEDULE_KEYS
            if unknown:
                _err(errors, "unknown_field", f"{fld}.schedule",
                     f"unknown schedule fields {sorted(unknown)}")
            shape_keys = [k for k in _SCHEDULE_KEYS if schedule.get(k)]
            if len(shape_keys) != 1:
                _err(errors, "bad_schedule", f"{fld}.schedule",
                     "schedule must set exactly one of cron_local / at / every_s")
            ev = schedule.get("every_s")
            floor = effective_every_floor()
            if ev is not None and (
                not isinstance(ev, int) or isinstance(ev, bool) or ev < floor
            ):
                _err(errors, "bad_schedule", f"{fld}.schedule.every_s",
                     f"every_s must be an integer >= {floor}")

    return ValidatedSource(
        id=sid,
        mode=mode,
        connector_id=connector if isinstance(connector, str) else None,
        event=event_key if isinstance(event_key, str) else None,
        params=params,
        poll_interval_s=poll_interval,
        schedule=schedule,
        filter_rules=filter_rules,
        dedupe_key_field=dedupe_field,
        event_spec=event_spec,
    )


def _validate_collect(
    idx: int, collect: Any, errors: list[dict],
) -> Optional[dict]:
    fld = f"steps[{idx}].collect"
    if collect is None:
        return None
    if not isinstance(collect, dict):
        _err(errors, "bad_collect", fld, "collect must be an object")
        return None
    for k in collect:
        if k not in _COLLECT_KEYS:
            _err(errors, "unknown_field", f"{fld}.{k}",
                 f"unknown collect field {k!r}")
    out: dict = {}
    items_path = collect.get("items_path")
    if not isinstance(items_path, str) or not items_path.strip():
        _err(errors, "bad_collect", f"{fld}.items_path",
             "collect.items_path is required (dot-path to the item list)")
    else:
        out["items_path"] = items_path.strip()
    fields = collect.get("fields") or {}
    if not isinstance(fields, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in fields.items()
    ):
        _err(errors, "bad_collect", f"{fld}.fields",
             "collect.fields must map template names to item dot-paths")
        fields = {}
    out["fields"] = fields
    fmt = collect.get("format")
    if fmt is not None and not isinstance(fmt, str):
        _err(errors, "bad_collect", f"{fld}.format",
             "collect.format must be a string template over {{item.<field>}}")
        fmt = None
    if fmt:
        out["format"] = fmt
    limit = collect.get("limit", COLLECT_LIMIT_DEFAULT)
    if (not isinstance(limit, int) or isinstance(limit, bool)
            or not 1 <= limit <= COLLECT_LIMIT_MAX):
        _err(errors, "bad_collect", f"{fld}.limit",
             f"collect.limit must be an integer 1..{COLLECT_LIMIT_MAX}")
        limit = COLLECT_LIMIT_DEFAULT
    out["limit"] = limit
    for key in ("empty_text", "join"):
        val = collect.get(key)
        if val is not None:
            if not isinstance(val, str):
                _err(errors, "bad_collect", f"{fld}.{key}",
                     f"collect.{key} must be a string")
            else:
                out[key] = val
    return out


def _validate_step(
    idx: int,
    step: Any,
    registry: dict[str, dict],
    errors: list[dict],
    *,
    template_mode: bool,
) -> Optional[ValidatedStep]:
    fld = f"steps[{idx}]"
    if not isinstance(step, dict):
        _err(errors, "bad_step", fld, "each step must be an object")
        return None
    for k in step:
        if k not in _STEP_KEYS:
            _err(errors, "unknown_field", f"{fld}.{k}",
                 f"unknown step field {k!r}")

    sid = step.get("id")
    if not isinstance(sid, str) or not _ID_RE.match(sid):
        _err(errors, "bad_step_id", f"{fld}.id",
             "step id must match ^[a-z][a-z0-9_]{0,23}$")
        sid = f"step{idx}"
    elif sid in RESERVED_IDS:
        _err(errors, "reserved_step_id", f"{fld}.id",
             f"step id {sid!r} is a reserved template root "
             f"({sorted(RESERVED_IDS)})")

    params = step.get("params") or {}
    if not isinstance(params, dict):
        _err(errors, "bad_params", f"{fld}.params", "params must be an object")
        params = {}

    connector = step.get("connector_id")
    tool = step.get("tool")
    grant_id = step.get("grant_id")

    # CONTRACTS-R31 §4.2a. A READ step defaults to `continue`: one
    # unreachable account must not end the run. On 26 August Jira and
    # Gmail both answered and GitHub did not, and the whole Morning work
    # brief stopped at "Stopped before it finished" — Slack was never
    # posted, so the two accounts that DID answer bought the user
    # nothing.
    #
    # `continue` and `skip` do the same thing to control flow. They
    # differ in what the user is told: `continue` names the account and
    # offers the fix (a `needs_you` turn), `skip` is silent, which is
    # what the Teams `provider_down` precedent wants. `fail` still
    # exists and is still the default for WRITES — a write that fails is
    # not a source that is missing, it is a change that did not happen.
    _is_write = bool(step.get("grant_id") or step.get("grant_target"))
    _default_on_error = "fail" if _is_write else "continue"
    on_error = step.get("on_error", _default_on_error)
    if on_error not in ("fail", "skip", "continue"):
        _err(errors, "bad_on_error", f"{fld}.on_error",
             "on_error must be 'fail', 'skip' or 'continue'")
        on_error = _default_on_error
    mutates = False

    cap = registry.get(connector) if isinstance(connector, str) else None
    if cap is None:
        _err(errors, "unknown_connector", f"{fld}.connector_id",
             f"connector {connector!r} is not automatable")
    elif not isinstance(tool, str) or not tool:
        _err(errors, "unknown_tool", f"{fld}.tool", "step.tool is required")
    else:
        writes = cap.get("scopes_write_by_action") or {}
        if tool in writes:
            mutates = True
            if (not template_mode
                    and (not isinstance(grant_id, str) or not grant_id.strip())):
                _err(errors, "write_without_grant", f"{fld}.grant_id",
                     f"{tool} is a write action — a grant reference is "
                     f"required before it can be part of a spec")
        elif not tool.startswith(f"{connector}__"):
            _err(errors, "unknown_tool", f"{fld}.tool",
                 f"{tool!r} does not belong to connector {connector!r}")
        # Non-write tools past the prefix check are validated against
        # the live manifest at execute time by the dispatcher (unknown
        # tool ⇒ tool_error, fail closed) — same rule as v1.

    collect = _validate_collect(idx, step.get("collect"), errors)
    if collect is not None and mutates:
        _err(errors, "bad_collect", f"{fld}.collect",
             "collect only applies to read steps")
        collect = None

    grant_target = step.get("grant_target")
    if grant_target is not None and not isinstance(grant_target, dict):
        _err(errors, "bad_params", f"{fld}.grant_target",
             "grant_target is system-written and must be an object")
        grant_target = None

    return ValidatedStep(
        id=sid,
        connector_id=connector if isinstance(connector, str) else "",
        tool=tool if isinstance(tool, str) else "",
        params_template=params,
        mutates=mutates,
        grant_id=grant_id if isinstance(grant_id, str) else None,
        grant_target=grant_target or {},
        collect=collect,
        on_error=on_error,
    )


def _iter_var_refs(spec: dict):
    """Yield (field, var_name) for every {{var.<name>}} reference in
    the places variables are allowed."""
    def _scan(obj: Any, where: str):
        if isinstance(obj, str):
            for m in _VAR_REF_RE.finditer(obj):
                yield where, m.group(1)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from _scan(v, where)
        elif isinstance(obj, list):
            for v in obj:
                yield from _scan(v, where)

    trig = spec.get("trigger") or {}
    for i, src in enumerate(trig.get("sources") or []):
        if isinstance(src, dict):
            yield from _scan(src.get("params") or {},
                             f"trigger.sources[{i}].params")
            yield from _scan(src.get("filter") or {},
                             f"trigger.sources[{i}].filter")
    for i, step in enumerate(spec.get("steps") or []):
        if isinstance(step, dict):
            yield from _scan(step.get("params") or {}, f"steps[{i}].params")
            collect = step.get("collect")
            if isinstance(collect, dict):
                yield from _scan(collect.get("format"),
                                 f"steps[{i}].collect.format")
                yield from _scan(collect.get("empty_text"),
                                 f"steps[{i}].collect.empty_text")


def validate_spec_v2(
    spec: dict,
    registry: dict[str, dict],
    *,
    template_mode: bool = False,
    template_vars: Optional[set] = None,
) -> ValidatedSpecV2:
    """Validate one v2 AutomationSpec. Raises SpecError with EVERY
    problem found; returns ValidatedSpecV2 on success."""
    errors: list[dict] = []

    for k in spec:
        if k not in _TOP_KEYS:
            _err(errors, "unknown_field", k, f"unknown top-level field {k!r}")

    name = spec.get("name")
    if not isinstance(name, str) or not (1 <= len(name.strip()) <= 120):
        _err(errors, "bad_name", "name", "name must be 1-120 characters")
        name = ""
    else:
        name = name.strip()

    # R36-7: how this automation's result should read. Optional; a
    # template stamps it so a Newsletter roundup narrates a newsletter
    # digest instead of the morning triage.
    narration = spec.get("narration")
    if narration is not None:
        if not isinstance(narration, dict):
            _err(errors, "bad_narration", "narration",
                 "narration must be an object")
            narration = None
        else:
            unknown_n = set(narration) - {"style", "title", "goal"}
            if unknown_n:
                _err(errors, "unknown_field", "narration",
                     f"unknown narration fields {sorted(unknown_n)}")
            style = narration.get("style")
            if style not in ("digest", "brief", "changes"):
                _err(errors, "bad_narration", "narration.style",
                     "narration.style must be 'digest', 'brief' or "
                     "'changes'")
            n_title = narration.get("title")
            if n_title is not None and not (
                isinstance(n_title, str) and 1 <= len(n_title.strip()) <= 80
            ):
                _err(errors, "bad_narration", "narration.title",
                     "narration.title must be 1-80 characters")
            n_goal = narration.get("goal")
            if n_goal is not None and not (
                isinstance(n_goal, str) and 1 <= len(n_goal.strip()) <= 300
            ):
                _err(errors, "bad_narration", "narration.goal",
                     "narration.goal must be 1-300 characters")

    mode = spec.get("mode", "confirm")
    if mode not in ("auto", "confirm"):
        _err(errors, "bad_mode", "mode", "mode must be 'auto' or 'confirm'")
        mode = "confirm"

    variables = spec.get("variables") or {}
    if not isinstance(variables, dict) or not all(
        isinstance(k, str) and _ID_RE.match(k) and isinstance(v, str)
        for k, v in variables.items()
    ):
        _err(errors, "bad_variables", "variables",
             "variables must map ^[a-z][a-z0-9_]{0,23}$ names to strings")
        variables = {}

    # ── sources ──────────────────────────────────────────────────────
    trig = spec.get("trigger")
    if not isinstance(trig, dict):
        _err(errors, "missing_trigger", "trigger", "trigger object is required")
        trig = {}
    for k in trig:
        if k != "sources":
            _err(errors, "unknown_field", f"trigger.{k}",
                 f"v2 triggers have exactly one field: sources[]")
    raw_sources = trig.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        _err(errors, "missing_sources", "trigger.sources",
             "trigger.sources must be a non-empty list")
        raw_sources = []
    if len(raw_sources) > MAX_SOURCES:
        _err(errors, "too_many_sources", "trigger.sources",
             f"at most {MAX_SOURCES} sources")
        raw_sources = raw_sources[:MAX_SOURCES]

    sources: list[ValidatedSource] = []
    for i, raw_src in enumerate(raw_sources):
        vs = _validate_source(i, raw_src, registry, errors)
        if vs is not None:
            sources.append(vs)

    seen_ids: set = set()
    schedule_count = 0
    push_count = 0
    for i, s in enumerate(sources):
        if s.id in seen_ids:
            _err(errors, "duplicate_source_id", f"trigger.sources[{i}].id",
                 f"duplicate source id {s.id!r}")
        seen_ids.add(s.id)
        if s.mode == "schedule":
            schedule_count += 1
        if s.mode == "push":
            push_count += 1
    if schedule_count > 1:
        _err(errors, "duplicate_schedule_source", "trigger.sources",
             "at most one schedule source")
    if push_count > 1:
        _err(errors, "duplicate_push_source", "trigger.sources",
             "at most one push source")

    # ── steps ────────────────────────────────────────────────────────
    raw_steps = spec.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        _err(errors, "no_steps", "steps", "steps must be a non-empty list")
        raw_steps = []
    if len(raw_steps) > MAX_STEPS:
        _err(errors, "too_many_steps", "steps",
             f"at most {MAX_STEPS} steps")
        raw_steps = raw_steps[:MAX_STEPS]

    steps: list[ValidatedStep] = []
    for i, raw_step in enumerate(raw_steps):
        st = _validate_step(i, raw_step, registry, errors,
                            template_mode=template_mode)
        if st is not None:
            steps.append(st)

    step_ids: set = set()
    write_seen = False
    write_count = 0
    for i, st in enumerate(steps):
        if st.id in step_ids or st.id in seen_ids:
            _err(errors, "duplicate_step_id", f"steps[{i}].id",
                 f"duplicate step id {st.id!r}")
        step_ids.add(st.id)
        if st.mutates:
            write_seen = True
            write_count += 1
        elif write_seen:
            _err(errors, "write_before_read", f"steps[{i}]",
                 "read steps must come before write steps — a write is "
                 "staged asynchronously, so a later read could never "
                 "see it")
    # Round 30 (§4.11a): reads-only specs are legal — migrated email
    # briefings deliver through the notification pipeline, not a write
    # step, and §4.1 derives mode "reads_only" from exactly this shape.
    # (Until R30 this was the `no_write_step` rejection.)
    if write_count > MAX_WRITE_STEPS:
        _err(errors, "too_many_writes", "steps",
             f"at most {MAX_WRITE_STEPS} write steps")

    # ── variables actually referenced must exist ─────────────────────
    declared = set(variables) | set(template_vars or ())
    for where, var_name in _iter_var_refs(spec):
        if var_name not in declared:
            _err(errors, "unknown_variable", where,
                 f"{{{{var.{var_name}}}}} is not declared in variables")

    if errors:
        raise SpecError(errors)

    canonical = {
        "version": 2,
        "name": name,
        "description": spec.get("description") or None,
        "mode": mode,
        **({"narration": narration} if narration else {}),
        **({"variables": variables} if variables else {}),
        "trigger": {
            "sources": [
                {
                    "id": s.id,
                    "mode": s.mode,
                    **({"connector_id": s.connector_id} if s.connector_id else {}),
                    **({"event": s.event} if s.event else {}),
                    **({"params": s.params} if s.params else {}),
                    **({"poll_interval_s": s.poll_interval_s}
                       if s.poll_interval_s else {}),
                    **({"schedule": s.schedule} if s.schedule else {}),
                    **({"filter": s.filter_rules} if s.filter_rules else {}),
                    **({"dedupe_key": f"event.{s.dedupe_key_field}"}
                       if s.dedupe_key_field else {}),
                }
                for s in sources
            ],
        },
        "steps": [
            {
                "id": st.id,
                "connector_id": st.connector_id,
                "tool": st.tool,
                "params": st.params_template,
                **({"collect": st.collect} if st.collect else {}),
                **({"on_error": st.on_error}
                   if st.on_error != ("fail" if st.mutates else "continue")
                   else {}),
                **({"grant_id": st.grant_id} if st.grant_id else {}),
                **({"grant_target": st.grant_target}
                   if st.grant_target else {}),
            }
            for st in steps
        ],
    }
    return ValidatedSpecV2(
        raw=canonical,
        name=name,
        mode=mode,
        variables=variables,
        sources=tuple(sources),
        steps=tuple(steps),
    )
