"""AutomationSpec — the canonical spec shape and its validator.

Hand-rolled validation, deliberately: the repo's precedent is that a
security boundary must not rest on the transitively-present `jsonschema`
package (`connector_pending_actions.py` documents the call). Every
rejection the round brief names is a distinct, tested error code:

    write_without_grant       mutating action with no grant ref
    grant_target_mismatch     grant pins a different target (arm-time)
    unknown_tool              action tool not in the connector registry
    unknown_event             trigger event not declared by the connector
    missing_dedupe_key        push/poll spec without a dedupe key
    interval_below_floor      poll interval under the connector/global floor

`validate_spec` is pure — it takes the spec dict plus the capability
registry snapshot and returns a `ValidatedSpec` or raises `SpecError`
carrying every problem at once (a setup agent that gets one error per
round-trip burns a turn per field).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.db.models.automation import (
    AUTOMATION_POLL_FLOOR_S, AUTOMATION_TRIGGER_MODES,
)


class SpecError(ValueError):
    """Validation failed. `errors` is a list of {code, field, message}."""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__("; ".join(
            f"{e['code']}: {e['message']}" for e in errors
        ))


# ── Dev fast-lane (Round 28) ─────────────────────────────────────────
#
# An env-gated override for dev/e2e tenants that lowers the poll and
# every_s floors to seconds so a full trigger→fire loop is watchable in
# one sitting. Two-sided refusal, same as the e2e metering marker: the
# flag alone is not enough — a production ENVIRONMENT ignores it, so a
# stray env var on a prod tenant changes nothing. The manifest load
# lint and the platform registry keep the honest production floors
# either way; this bends only spec validation/compile on this tenant.

AUTOMATION_DEV_POLL_FLOOR_S = 5


def dev_fast_lane_active() -> bool:
    from app.config import settings
    return (
        bool(getattr(settings, "automations_dev_fast_lane", False))
        and (settings.environment or "").strip().lower() != "production"
    )


def effective_poll_floor(cap_floor: Any) -> int:
    """The poll floor for spec validation: the connector's declared
    floor, never below the global rail — unless the dev fast-lane is
    active, in which case seconds are allowed."""
    if dev_fast_lane_active():
        return AUTOMATION_DEV_POLL_FLOOR_S
    try:
        declared = int(cap_floor or AUTOMATION_POLL_FLOOR_S)
    except (TypeError, ValueError):
        declared = AUTOMATION_POLL_FLOOR_S
    return max(declared, AUTOMATION_POLL_FLOOR_S)


def effective_every_floor() -> int:
    """Floor for schedule {every_s}."""
    return AUTOMATION_DEV_POLL_FLOOR_S if dev_fast_lane_active() else 60


@dataclass(frozen=True)
class ValidatedSpec:
    """The parsed, validated spec. `raw` is the canonical dict to
    persist (unknown keys already rejected, defaults filled)."""

    raw: dict
    name: str
    mode: str                       # "auto" | "confirm"
    trigger_mode: str               # push | poll | schedule
    trigger_connector_id: Optional[str]
    trigger_event: Optional[str]
    trigger_params: dict
    poll_interval_s: Optional[int]
    schedule: Optional[dict]
    filter_rules: dict
    action_connector_id: str
    action_tool: str
    action_params_template: dict
    grant_id: Optional[str]
    action_mutates: bool
    dedupe_key_field: Optional[str]
    event_spec: Optional[dict] = field(default=None)


def _err(errors: list, code: str, fld: str, message: str) -> None:
    errors.append({"code": code, "field": fld, "message": message})


_TOP_KEYS = {"name", "description", "trigger", "action", "dedupe_key", "mode",
             "version"}
_TRIGGER_KEYS = {
    "mode", "connector_id", "event", "params", "poll_interval_s",
    "schedule", "filter",
}
# `grant_target` is SYSTEM-written (the arm step snapshots the approved
# grant's pinned target for template rendering) — accepted on
# re-validation, never authored by the user or the model.
_ACTION_KEYS = {"connector_id", "tool", "params_template", "grant_id",
                "grant_target"}
_SCHEDULE_KEYS = {"cron_local", "at", "every_s"}


def validate_spec(
    spec: Any,
    registry: dict[str, dict],
    *,
    template_mode: bool = False,
    template_vars: Optional[set] = None,
):
    """Validate one AutomationSpec against the capability registry.

    `registry` maps connector_id → the automation_registry() entry
    (push/poll/floor_s/events/scopes_write_by_action/...). Raises
    SpecError with EVERY problem found.

    Dispatch (Round 28): a spec with `version: 2` returns a
    `ValidatedSpecV2` from spec_v2.py; anything else takes the v1 path
    below, unchanged. `template_mode` (catalog lint only — nothing on
    the create/run path sets it) waives grant references and treats
    `template_vars` as declared.
    """
    errors: list[dict] = []

    if not isinstance(spec, dict):
        raise SpecError([{
            "code": "not_an_object", "field": "",
            "message": f"spec must be an object, got {type(spec).__name__}",
        }])

    version = spec.get("version", 1)
    if version == 2:
        from .spec_v2 import validate_spec_v2
        return validate_spec_v2(
            spec, registry,
            template_mode=template_mode, template_vars=template_vars,
        )
    if version != 1:
        raise SpecError([{
            "code": "bad_version", "field": "version",
            "message": f"spec version must be 1 or 2, got {version!r}",
        }])

    for k in spec:
        if k not in _TOP_KEYS:
            _err(errors, "unknown_field", k, f"unknown top-level field {k!r}")

    name = spec.get("name")
    if not isinstance(name, str) or not (1 <= len(name.strip()) <= 120):
        _err(errors, "bad_name", "name", "name must be 1-120 characters")
        name = ""
    else:
        name = name.strip()

    mode = spec.get("mode", "confirm")
    if mode not in ("auto", "confirm"):
        _err(errors, "bad_mode", "mode", "mode must be 'auto' or 'confirm'")
        mode = "confirm"

    # ── trigger ──────────────────────────────────────────────────────
    trig = spec.get("trigger")
    if not isinstance(trig, dict):
        _err(errors, "missing_trigger", "trigger", "trigger object is required")
        trig = {}
    for k in trig:
        if k not in _TRIGGER_KEYS:
            _err(errors, "unknown_field", f"trigger.{k}",
                 f"unknown trigger field {k!r}")

    t_mode = trig.get("mode")
    if t_mode not in AUTOMATION_TRIGGER_MODES:
        _err(errors, "bad_trigger_mode", "trigger.mode",
             f"trigger.mode must be one of {sorted(AUTOMATION_TRIGGER_MODES)}")
        t_mode = "schedule"

    t_connector = trig.get("connector_id")
    t_event = trig.get("event")
    t_params = trig.get("params") or {}
    poll_interval: Optional[int] = None
    schedule: Optional[dict] = None
    event_spec: Optional[dict] = None
    filter_rules = trig.get("filter") or {}
    if not isinstance(filter_rules, dict):
        _err(errors, "bad_filter", "trigger.filter", "filter must be an object")
        filter_rules = {}
    if not isinstance(t_params, dict):
        _err(errors, "bad_params", "trigger.params", "params must be an object")
        t_params = {}

    if t_mode in ("push", "poll"):
        cap = registry.get(t_connector) if isinstance(t_connector, str) else None
        if cap is None:
            _err(errors, "unknown_connector", "trigger.connector_id",
                 f"connector {t_connector!r} is not automatable "
                 f"(known: {sorted(registry)})")
        else:
            if t_mode == "push" and not cap.get("push"):
                _err(errors, "push_unavailable", "trigger.mode",
                     f"{t_connector} has no push path — use poll")
            if t_mode == "poll" and not cap.get("poll"):
                _err(errors, "poll_unavailable", "trigger.mode",
                     f"{t_connector} does not support polling")
            events = {e["key"]: e for e in cap.get("events", [])}
            if not isinstance(t_event, str) or t_event not in events:
                _err(errors, "unknown_event", "trigger.event",
                     f"event {t_event!r} not declared by {t_connector} "
                     f"(known: {sorted(events)})")
            else:
                event_spec = events[t_event]
        if t_mode == "poll":
            # Identical to the pre-R28 max() when the fast lane is off
            # (the default); seconds only for env-gated dev tenants.
            floor = effective_poll_floor((cap or {}).get("floor_s"))
            raw_iv = trig.get("poll_interval_s", floor)
            if not isinstance(raw_iv, int) or isinstance(raw_iv, bool):
                _err(errors, "bad_interval", "trigger.poll_interval_s",
                     "poll_interval_s must be an integer number of seconds")
            elif raw_iv < floor:
                _err(errors, "interval_below_floor", "trigger.poll_interval_s",
                     f"poll_interval_s={raw_iv} is below the floor of "
                     f"{floor}s for {t_connector}")
            else:
                poll_interval = raw_iv
    elif t_mode == "schedule":
        schedule = trig.get("schedule")
        if not isinstance(schedule, dict) or not schedule:
            _err(errors, "missing_schedule", "trigger.schedule",
                 "schedule mode requires trigger.schedule")
            schedule = None
        else:
            unknown = set(schedule) - _SCHEDULE_KEYS
            if unknown:
                _err(errors, "unknown_field", "trigger.schedule",
                     f"unknown schedule fields {sorted(unknown)}")
            shape_keys = [k for k in _SCHEDULE_KEYS if schedule.get(k)]
            if len(shape_keys) != 1:
                _err(errors, "bad_schedule", "trigger.schedule",
                     "schedule must set exactly one of cron_local / at / every_s")
            ev = schedule.get("every_s")
            ev_floor = effective_every_floor()
            if ev is not None and (
                not isinstance(ev, int) or isinstance(ev, bool)
                or ev < ev_floor
            ):
                _err(errors, "bad_schedule", "trigger.schedule.every_s",
                     f"every_s must be an integer >= {ev_floor}")

    # ── dedupe key ───────────────────────────────────────────────────
    dedupe_field: Optional[str] = None
    dk = spec.get("dedupe_key")
    if t_mode in ("push", "poll"):
        if not isinstance(dk, str) or not dk.strip():
            _err(errors, "missing_dedupe_key", "dedupe_key",
                 "push/poll automations require a dedupe_key "
                 "(\"event.<field>\")")
        else:
            dk = dk.strip()
            if not dk.startswith("event."):
                _err(errors, "bad_dedupe_key", "dedupe_key",
                     "dedupe_key must be an \"event.<field>\" reference")
            else:
                fname = dk[len("event."):]
                if event_spec is not None:
                    known = set((event_spec.get("fields") or {}).keys())
                    known.add(event_spec.get("dedupe_field") or "")
                    if fname not in known:
                        _err(errors, "bad_dedupe_key", "dedupe_key",
                             f"{fname!r} is not a field of event "
                             f"{t_event!r} (known: {sorted(known - {''})})")
                dedupe_field = fname

    # ── action ───────────────────────────────────────────────────────
    act = spec.get("action")
    if not isinstance(act, dict):
        _err(errors, "missing_action", "action", "action object is required")
        act = {}
    for k in act:
        if k not in _ACTION_KEYS:
            _err(errors, "unknown_field", f"action.{k}",
                 f"unknown action field {k!r}")

    a_connector = act.get("connector_id")
    a_tool = act.get("tool")
    a_params = act.get("params_template") or {}
    grant_id = act.get("grant_id")
    mutates = False
    if not isinstance(a_params, dict):
        _err(errors, "bad_params", "action.params_template",
             "params_template must be an object")
        a_params = {}

    a_cap = registry.get(a_connector) if isinstance(a_connector, str) else None
    if a_cap is None:
        _err(errors, "unknown_connector", "action.connector_id",
             f"connector {a_connector!r} is not automatable")
    elif not isinstance(a_tool, str) or not a_tool:
        _err(errors, "unknown_tool", "action.tool", "action.tool is required")
    else:
        writes = a_cap.get("scopes_write_by_action") or {}
        if a_tool in writes:
            mutates = True
            if (not template_mode
                    and (not isinstance(grant_id, str) or not grant_id.strip())):
                _err(errors, "write_without_grant", "action.grant_id",
                     f"{a_tool} is a write action — a grant reference is "
                     f"required before it can be part of a spec")
        elif not a_tool.startswith(f"{a_connector}__"):
            _err(errors, "unknown_tool", "action.tool",
                 f"{a_tool!r} does not belong to connector {a_connector!r}")
        # Non-write tools that pass the prefix check are validated
        # against the live manifest at arm/execute time by the
        # dispatcher (unknown tool ⇒ tool_error, fail closed).

    if errors:
        raise SpecError(errors)

    canonical = {
        "name": name,
        "description": spec.get("description") or None,
        "trigger": {
            "mode": t_mode,
            **({"connector_id": t_connector} if t_connector else {}),
            **({"event": t_event} if t_event else {}),
            **({"params": t_params} if t_params else {}),
            **({"poll_interval_s": poll_interval} if poll_interval else {}),
            **({"schedule": schedule} if schedule else {}),
            **({"filter": filter_rules} if filter_rules else {}),
        },
        "action": {
            "connector_id": a_connector,
            "tool": a_tool,
            "params_template": a_params,
            **({"grant_id": grant_id} if grant_id else {}),
            **({"grant_target": act.get("grant_target")}
               if isinstance(act.get("grant_target"), dict) else {}),
        },
        **({"dedupe_key": dk} if t_mode in ("push", "poll") else {}),
        "mode": mode,
    }
    return ValidatedSpec(
        raw=canonical,
        name=name,
        mode=mode,
        trigger_mode=t_mode,
        trigger_connector_id=t_connector if isinstance(t_connector, str) else None,
        trigger_event=t_event if isinstance(t_event, str) else None,
        trigger_params=t_params,
        poll_interval_s=poll_interval,
        schedule=schedule,
        filter_rules=filter_rules,
        action_connector_id=a_connector or "",
        action_tool=a_tool or "",
        action_params_template=a_params,
        grant_id=grant_id if isinstance(grant_id, str) else None,
        action_mutates=mutates,
        dedupe_key_field=dedupe_field,
        event_spec=event_spec,
    )


# ── Template rendering (executor's prepare step) ─────────────────────


def resolve_path(obj: Any, path: str) -> Any:
    """Dot-path lookup into nested dicts. Returns None on any miss."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def render_value(val: Any, ctx: dict) -> Any:
    """Render one value against a context dict. Only strings are
    templated; a placeholder that resolves to None renders as an empty
    string. Shared by the v1 and v2 render paths."""
    if not isinstance(val, str):
        return val
    out = val
    # Simple, deliberate: find {{...}} spans, resolve, substitute.
    while "{{" in out and "}}" in out:
        start = out.index("{{")
        end = out.index("}}", start)
        expr = out[start + 2:end].strip()
        resolved = resolve_path(ctx, expr)
        out = out[:start] + ("" if resolved is None else str(resolved)) + out[end + 2:]
    return out


def render_with_ctx(template: dict, ctx: dict) -> dict:
    return {k: render_value(v, ctx) for k, v in template.items()}


def render_params(
    template: dict,
    *,
    event: Optional[dict] = None,
    grant_target: Optional[dict] = None,
) -> dict:
    """Fill a params_template's {{event.x}} / {{grant.target.x}}
    placeholders. Only string values are templated; a placeholder that
    resolves to None renders as an empty string (the validator upstream
    keeps required fields from being empty at execute time)."""
    ctx = {
        "event": event or {},
        "grant": {"target": grant_target or {}},
    }
    return render_with_ctx(template, ctx)
