"""The build history — how this automation came to exist (R38).

An automation gets one run ledger per firing and, until now, nothing at
all about the minutes before the first one. The canvas could say what
it does; it could not say what was DECIDED, in what order, or what was
deliberately not asked for. So "why can it only draft?" was answerable
by reading the permission sheet and inferring — and a user who wanted
to know what happened while the spinner turned had nothing to read.

`build_history` is that record: an ordered list of phases, each with a
title, a sub-line, its MEASURED duration, and the lines of what it
actually did.

    [{"id": "trigger", "title": "Set when it runs",
      "sub": "weekdays at 8:00", "ms": 1800,
      "did": ["Set it to run weekdays at 8:00.", …]}, …]

Two rules the shape depends on, and both are the reason this is a
column rather than a derivation:

1. **The `ms` is measured, never scripted.** `BuildRecorder.phase()`
   wraps the real work — validation and persistence, the arm attempt,
   the destination resolution, one segment per account — and stamps the
   wall clock it took. A build history whose durations were invented
   would be a progress bar with no process behind it, which is the one
   thing worse than no history.

2. **The `did` lines are derived from the FINISHED automation, once, at
   creation.** Not re-derived per read: the build is a history, and an
   account removed on Tuesday must not unwrite Monday's "Connected
   Gmail". Not authored by the recorder's caller either — the caller
   supplies timings and nothing else, so a phase cannot claim work the
   spec does not show.

Copy note: the design sketch titled the first phase "Set the trigger".
`trigger` is a banned word in `fixtures/automations/banned-copy.json`
(and `copy_guard.BANNED_WORDS`), so every title and every line here is
written to pass `copy_guard.clean` — the shape is the sketch's, the
words are the contract's.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation
from app.db.models.automation import (
    AUTOMATION_AUTO_PAUSE_FAILURES, AUTOMATION_RUN_CAP_S,
)
from app.services import automation_verbs as verbs

logger = logging.getLogger(__name__)

#: How the automation was entered. `chat` is the agent's own
#: `automations__create`; the two canvas doors are their own sources so
#: the history can say which one this was.
BUILD_SOURCES = ("template", "described", "chat")

#: Phase ids the composer knows how to title. An account phase is
#: `account:<connector_id>` and is titled from the connector.
_PHASE_TITLES = {
    "trigger": "Set when it runs",
    "agent": "Wired the agent in",
    "output": "Pointed its output at you",
}


class BuildRecorder:
    """Times the phases of one build. Timings only — never words.

    Usage:

        rec = BuildRecorder("template")
        with rec.phase("trigger"):
            automation, vspec = await create_automation(...)
        ...
        await record(db, automation=automation, recorder=rec)
    """

    def __init__(self, source: str) -> None:
        self.source = source if source in BUILD_SOURCES else "chat"
        self._order: list[str] = []
        self._ms: dict[str, int] = {}
        self._t0 = time.monotonic()

    @contextlib.contextmanager
    def phase(self, phase_id: str):
        """Time one segment. Re-entering an id ADDS to it — an account
        touched in two places spent time in both, and reporting only
        the last would under-report the build."""
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = int((time.monotonic() - started) * 1000)
            if phase_id not in self._ms:
                self._order.append(phase_id)
                self._ms[phase_id] = 0
            self._ms[phase_id] += elapsed

    @property
    def total_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def timings(self) -> list[tuple[str, int]]:
        """Phases in READING order, not measurement order.

        The two differ and only one of them is a narrative: the
        template path resolves the destination before it arms, the
        described path arms before it resolves. Serving measurement
        order would make two automations built from the same spec tell
        their story in different sequences, which is the one thing a
        history must not do. Accounts keep their own relative order
        (the spec's member order), after the three fixed phases.
        """
        rank = {"trigger": 0, "agent": 1, "output": 2}
        return sorted(
            ((pid, self._ms[pid]) for pid in self._order),
            key=lambda pair: (rank.get(pair[0], 3),
                              self._order.index(pair[0])),
        )


# ── the derived lines ────────────────────────────────────────────────

def _sources(raw: dict) -> list[dict]:
    if raw.get("version") == 2:
        return [s for s in ((raw.get("trigger") or {}).get("sources") or [])
                if isinstance(s, dict)]
    trig = raw.get("trigger") or {}
    return [trig] if isinstance(trig, dict) and trig else []


def _minutes(seconds) -> str:
    try:
        n = max(1, round(int(seconds) / 60))
    except (TypeError, ValueError):
        return "every few minutes"
    return "every minute" if n == 1 else f"every {n} minutes"


def _event_clause(src: dict) -> str:
    """"when a new email arrives" — the verb dictionary's own clause for
    the source's event, never the raw event key."""
    clause = (getattr(verbs, "_EVENT_CLAUSES", {}) or {}).get(
        src.get("event") or "")
    return clause or "the moment something changes"


def trigger_lines(automation: Automation, raw: dict) -> tuple[str, list[str]]:
    """(sub, did) for the "Set when it runs" phase."""
    did: list[str] = []
    sub = ""
    for src in _sources(raw):
        cid = src.get("connector_id") or ""
        name = verbs.display_name(cid) or cid
        mode = src.get("mode") or ("schedule" if src.get("schedule") else "")
        if mode == "schedule" or src.get("schedule"):
            human = verbs.schedule_human(src) or "on its own schedule"
            sub = sub or human
            did.append(f"Set it to run {human}.")
        elif mode == "push":
            sub = sub or f"when {name} says so"
            did.append(f"{name} tells it {_event_clause(src)}.")
        elif mode == "poll":
            sub = sub or f"looks at {name}"
            did.append(
                f"Looks at {name} {_minutes(src.get('poll_interval_s'))}, "
                f"{_event_clause(src)}."
            )
        if src.get("dedupe_key"):
            did.append("The same one is never picked up twice.")
    if not did:
        sub = "on request"
        did.append("It starts when you ask for it, and at no other time.")
    else:
        did.append("Nothing else can start it.")
    return sub, did


def agent_lines(automation: Automation, raw: dict) -> tuple[str, list[str]]:
    """(sub, did) for the "Wired the agent in" phase.

    The step count comes from `workflow._steps_human`, the same
    derivation the canvas's Steps sheet renders — so a v1 automation
    (whose steps live in `trigger` + `action`, not a `steps` array)
    counts the same way it reads.
    """
    from . import workflow
    from .setup_script import writer_connectors

    human = workflow._steps_human(automation, raw)
    steps = [s for s in (raw.get("steps") or []) if isinstance(s, dict)]
    thinks = [s for s in steps if s.get("kind") == "agent"]
    writers = writer_connectors(raw)
    readers = sorted({
        verbs.display_name(cid) or cid
        for cid in workflow._member_connectors(raw) if cid not in writers
    })
    did: list[str] = []
    if human:
        n = len(human)
        did.append(
            "Laid out 1 step." if n == 1
            else f"Laid out {n} steps, in order."
        )
    if readers:
        did.append(f"It reads {', '.join(readers)} before it says a word.")
    for t in thinks:
        ask = str(t.get("prompt") or "").strip()
        did.append(
            f"One step is the agent working it out: {ask[:100]}"
            if ask else "One step is the agent working it out."
        )
    did.append("Anything you tell it to remember is read on every run.")
    did.append(
        f"Every run stops at {AUTOMATION_RUN_CAP_S // 60} minutes, "
        f"finished or not."
    )
    did.append(
        f"{AUTOMATION_AUTO_PAUSE_FAILURES} failures in a row and it "
        f"stops itself and tells you."
    )
    return "rules, memory and limits", did


def output_lines(automation: Automation, raw: dict) -> tuple[str, list[str]]:
    """(sub, did) for the "Pointed its output at you" phase.

    The lines come from `workflow.output_block`, which is the same
    derivation the canvas's output node draws — so the history and the
    node can never disagree about where this automation delivers.
    """
    from . import workflow
    block = workflow.output_block(automation, raw)
    mode, _label = workflow.mode_of(automation, raw)
    sub = {
        "reads_only": "left on a branch to read",
        "drafts_only": block.get("node_sub") or "drafts wait for you",
        "posts": block.get("node_sub") or "where you allowed it",
        "asks_first": "staged, and it waits for your yes",
    }.get(mode, block.get("node_sub") or "")
    did: list[str] = []
    for line in block.get("lines") or []:
        title = str(line.get("title") or "").strip()
        body = str(line.get("body") or "").strip()
        if not title:
            continue
        did.append(f"{title} — {body}" if body else title)
    return sub, did


#: The design's four bands (spec §8: "Setup narration is one step per
#: GROUP, not per account"). This mirrors `src/automations/canvas/bands.ts`
#: in the app, deliberately: the app owns how the panel is DRAWN, and this
#: owns how the build is TOLD — and the build history is persisted once, at
#: creation, so it cannot be re-grouped later by a client that disagrees.
#: Move one and move the other.
_BANDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("MAIL", ("gmail", "outlook"), "your mail"),
    ("CHAT", ("slack", "teams"), "your channels"),
    ("WORK", ("jira", "github"), "your tickets and reviews"),
    ("PLANS", ("notion", "calendar"), "your plans"),
    ("CODE", ("claude", "codex"), "the repository you point it at"),
)


def band_of(connector_id: str) -> str:
    for key, members, _phrase in _BANDS:
        if connector_id in members:
            return key
    return "MORE"


def _join(names: list[str]) -> str:
    """"a", "a and b", "a, b and c" — the design's own list voice."""
    kept = [n for n in names if n]
    if not kept:
        return ""
    if len(kept) == 1:
        return kept[0]
    return f"{', '.join(kept[:-1])} and {kept[-1]}"


def band_lines(
    band_key: str, connector_ids: list[str], per_account: dict,
) -> tuple[str, str, list[str]]:
    """(title, sub, did) for one BAND — "Connected Gmail and Outlook".

    `per_account` is `{connector_id: (title, sub, did)}` from
    `account_lines`; the band keeps every account's own `did` lines
    underneath, so nothing the per-account pass established is lost —
    the change is which grain the STEP is at, not how much it says.
    """
    names = [verbs.display_name(cid) or cid for cid in connector_ids]
    title = f"Connected {_join(names)}"
    subs = {per_account[cid][1] for cid in connector_ids if cid in per_account}
    # One sub only when every account in the band got the same access;
    # a band where one is read-only and one can draft must not claim
    # either for both.
    sub = subs.pop() if len(subs) == 1 else "read only, scoped to you"
    did: list[str] = []
    for cid in connector_ids:
        did.extend(per_account.get(cid, ("", "", []))[2])
    return title, sub, did


def closing_sentence(
    band_order: list[tuple[str, list[str]]], format_noun: str,
) -> str:
    """The sentence that ends the setup narration (spec §8).

    `band_order` is `[(band_key, [connector_id, …]), …]` in the panel's
    own order; `format_noun` is the delivery format's noun ("a ranked
    list"), so what the narration promises and what the delivery node
    says are one value.
    """
    phrases = []
    for key, cids in band_order:
        phrase = next((p for k, _m, p in _BANDS if k == key), "your other accounts")
        names = _join([verbs.display_name(c) or c for c in cids])
        phrases.append(f"{phrase} ({names})" if names else phrase)
    read = _join(phrases)
    return (
        f"That is the shape of it. I read {read}, keep what is addressed to "
        f"you, and bring it back as {format_noun} in this chat. They are "
        f"grouped on the canvas — tap any account to see what is in there."
    )


def account_lines(
    connector_id: str, perms: dict, raw: dict,
) -> tuple[str, str, list[str]]:
    """(title, sub, did) for one `account:<cid>` phase.

    `perms` is `permissions.resolve`'s `{can, cant}` — the SAME source
    the connector sheet and the canvas captions read, so a line here
    cannot claim access the permission sheet denies.
    """
    name = verbs.display_name(connector_id) or connector_id
    can = [p for p in (perms or {}).get("can") or [] if p.get("label")]
    cant = [p for p in (perms or {}).get("cant") or [] if p.get("label")]
    writes = [p for p in can if not str(p.get("id") or "").startswith(
        f"{connector_id}.read")]
    rails = [p for p in cant if p.get("kind") == "rail"]

    did: list[str] = []
    if writes:
        did.append(
            f"Asked {name} for {writes[0]['label'].lower()} — and read "
            f"access to work from."
        )
    else:
        did.append(f"Asked {name} for read access — not write.")

    pins = (raw.get("focus") or {}).get(connector_id) or []
    targets = [
        s.get("grant_target") or {}
        for s in (raw.get("steps") or [])
        if isinstance(s, dict) and s.get("connector_id") == connector_id
        and (s.get("grant_target") or {}).get("label")
    ]
    if pins:
        labels = ", ".join(str(p.get("label") or p.get("id")) for p in pins)
        did.append(f"Started it at {labels}.")
    elif targets:
        did.append(f"Pinned it to {targets[0].get('label')}.")
    else:
        did.append(f"Scoped it to what this automation asks {name} for.")

    if rails:
        labels = [str(r["label"]).lower() for r in rails[:3]]
        joined = (labels[0] if len(labels) == 1
                  else ", ".join(labels[:-1]) + f" or {labels[-1]}")
        did.append(f"Confirmed it cannot {joined}.")

    sub = ("read only, scoped to this automation" if not writes
           else f"{writes[0]['label'].lower()}, and nothing else")
    return f"Connected {name}", sub, did


# ── write + read ─────────────────────────────────────────────────────

async def record(
    db: AsyncSession, *, automation: Automation, recorder: BuildRecorder,
) -> Optional[dict]:
    """Compose the history from the FINISHED automation and persist it.

    Best-effort by construction: a build history that fails to write
    must never fail the build that produced it. Returns the payload it
    wrote, or None.
    """
    try:
        raw = json.loads(automation.spec_json or "{}")
    except (ValueError, TypeError):
        raw = {}
    try:
        from . import permissions as _perms
        steps: list[dict] = []
        # The account phases are TIMED per account and TOLD per band
        # (spec §8): "Connected Gmail and Outlook — read only, scoped to
        # you". The band's own step carries the sum of its accounts'
        # measured time, so the durations still add up to `total_ms`.
        per_account: dict[str, tuple[str, str, list[str]]] = {}
        band_members: dict[str, list[str]] = {}
        band_ms: dict[str, int] = {}
        band_first: dict[str, int] = {}
        for i, (phase_id, ms) in enumerate(recorder.timings()):
            if not phase_id.startswith("account:"):
                continue
            cid = phase_id.split(":", 1)[1]
            resolved = await _perms.resolve(
                db, automation=automation, account_id=cid,
            )
            per_account[cid] = account_lines(cid, resolved, raw)
            key = band_of(cid)
            band_members.setdefault(key, []).append(cid)
            band_ms[key] = band_ms.get(key, 0) + max(0, int(ms))
            band_first.setdefault(key, i)
        emitted_bands: set[str] = set()

        for phase_id, ms in recorder.timings():
            if phase_id.startswith("account:"):
                cid = phase_id.split(":", 1)[1]
                key = band_of(cid)
                if key in emitted_bands:
                    continue
                emitted_bands.add(key)
                phase_id = f"band:{key.lower()}"
                ms = band_ms.get(key, 0)
                title, sub, did = band_lines(key, band_members[key], per_account)
            elif phase_id == "trigger":
                title = _PHASE_TITLES["trigger"]
                sub, did = trigger_lines(automation, raw)
            elif phase_id == "agent":
                title = _PHASE_TITLES["agent"]
                sub, did = agent_lines(automation, raw)
            elif phase_id == "output":
                title = _PHASE_TITLES["output"]
                sub, did = output_lines(automation, raw)
            else:
                continue
            steps.append({"id": phase_id, "title": title, "sub": sub,
                          "ms": max(0, int(ms)), "did": did})
        if not steps:
            return None
        payload = {
            "source": recorder.source,
            "at": datetime.utcnow().isoformat() + "Z",
            "total_ms": recorder.total_ms,
            "steps": steps,
        }
        row = await db.get(Automation, automation.id)
        if row is None:
            return None
        row.build_history_json = json.dumps(payload)
        await db.commit()
        automation.build_history_json = row.build_history_json
        return payload
    except Exception as e:  # noqa: BLE001 — a history never fails a build
        logger.warning("[automations] build history skipped: %s", e)
        return None


def read(automation: Automation) -> Optional[dict]:
    """The stored history, or None when this automation predates it.

    None and an empty list are different answers and the wire keeps
    them apart: `build_history: null` means "built before this was
    recorded", which is not the same claim as "built in no steps".
    """
    blob = getattr(automation, "build_history_json", None)
    if not blob:
        return None
    try:
        payload = json.loads(blob)
    except (ValueError, TypeError):
        logger.warning("[automations] build history unreadable on %s",
                       automation.id)
        return None
    if not isinstance(payload, dict) or not payload.get("steps"):
        return None
    return payload
