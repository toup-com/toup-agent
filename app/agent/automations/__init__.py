"""Automations engine (Round 26) — agent-side package.

Composes the four existing primitives into chat-built automations:

  spec.py      — AutomationSpec parsing + validation (hand-rolled, with
                 teeth; see docs/automations/MAPPING.md §3.3 for why not
                 the jsonschema package)
  registry.py  — capability metadata client (platform-fetched, cached)
  compiler.py  — spec → primitive bindings (Trigger / Routine rows);
                 arm / pause / resume / delete lifecycle
  executor.py  — fire → BuildJob run: evaluate, prepare, stage writes
  outbox.py    — durable write outbox flush (undo window, grant refs)
  handlers.py  — the routine-kind handlers (automation_poll /
                 automation_schedule) and the trigger action hook
  sweep.py     — reconciler sweeps (stuck runs, stale bindings,
                 auto-pause at 3 consecutive failures)

Everything is inert unless `settings.automations_enabled` is true.
"""
