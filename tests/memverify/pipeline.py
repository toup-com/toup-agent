"""Drivers that exercise the REAL production writer.

Nothing here re-implements capture. `drive_turn` calls
`memory_curator.curate_turn` — the exact function `_background_post_processing`
calls in turn post-processing — so the pre-gates, the prompt, the model call,
the deterministic validator and the atomic apply all run exactly as they do
in production, against a real Postgres.

The one deliberate difference from production is the ARGUMENT, and it is the
point of the suite: the runner passes `display_user_message`, so `drive_turn`
passes the clean text too. A scenario that carries `injected` is additionally
driven a second time with the DIRTY string (`drive_turn_dirty`) — belt as
well as braces. If the durability rules only work because the injection was
stripped upstream, the belt run says so.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select

from .corpus import Capture, Reject, Scenario, Turn


# ── Write path ───────────────────────────────────────────────────────────

async def drive_turn(
    db,
    user_id: str,
    user_text: str,
    assistant_text: str = "",
    *,
    trivial: bool = False,
    channel: str = "app",
) -> Dict[str, Any]:
    """Run one turn through the production writer. Returns its result dict."""
    from app.services import memory_curator

    result = await memory_curator.curate_turn(
        db, user_id,
        user_text=user_text,
        assistant_text=assistant_text,
        channel=channel,
        query_was_trivial=trivial,
    )
    await db.commit()
    return result


async def drive_conversation(
    db, user_id: str, turns: Sequence[Turn], sink: Optional[List[str]] = None,
) -> int:
    applied = 0
    for t in turns:
        res = await drive_turn(
            db, user_id, t.user, t.assistant, trivial=t.trivial,
        )
        applied += int(res.get("applied", 0))
        _collect(res, sink)
    return applied


def _collect(res: Dict[str, Any], sink: Optional[List[str]]) -> None:
    """Keep the writer's own account of what it refused.

    Without this a scenario that stores nothing is indistinguishable from one
    whose every op was rejected, and the two have opposite fixes. Three
    rounds of this rebuild were diagnosed by guessing between them.
    """
    if sink is None:
        return
    if res.get("skipped"):
        sink.append(f"pre-gate skipped the turn: {res['skipped']}")
    for complaint in (res.get("rejected") or []):
        sink.append(str(complaint))


async def drive_conversation_dirty(
    db, user_id: str, turns: Sequence[Turn], sink: Optional[List[str]] = None,
) -> int:
    """The BELT: hand the writer the string ws_chat actually builds.

    Production never does this — `agent_runner` passes
    `display_user_message`, pinned structurally in
    tests/test_curator_producers.py. This run asks the other half of the
    question: if the injection DID reach the writer, do the durability rules
    still refuse it? A suite that only tests the clean path cannot tell
    "the rules work" from "the strip works".
    """
    applied = 0
    for t in turns:
        res = await drive_turn(
            db, user_id, t.user + (t.injected or ""), t.assistant,
            trivial=t.trivial,
        )
        applied += int(res.get("applied", 0))
        _collect(res, sink)
    return applied


# ── Store inspection: FILES, not rows ────────────────────────────────────

async def all_files(db, user_id: str) -> List[Any]:
    from app.db.models import MemoryFile

    rows = await db.execute(
        select(MemoryFile)
        .where(MemoryFile.user_id == user_id)
        .order_by(MemoryFile.slug)
    )
    return list(rows.scalars())


async def bodies_by_slug(db, user_id: str) -> Dict[str, str]:
    return {f.slug: (f.body_md or "") for f in await all_files(db, user_id)}


async def change_summaries(db, user_id: str) -> List[str]:
    from app.db.models import MemoryFileChange

    rows = await db.execute(
        select(MemoryFileChange.summary).where(MemoryFileChange.user_id == user_id)
    )
    return [r[0] for r in rows.all()]


def bullets_of(body: str) -> List[str]:
    from app.memory_files import parse_bullets

    return parse_bullets(body)


# ── Ground-truth matching ────────────────────────────────────────────────

def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", text)


def find_capture(
    marker: Capture, bodies: Dict[str, str]
) -> Optional[Tuple[str, str]]:
    """(slug, bullet) where all of `all_of` appear in ONE bullet, else None.

    One BULLET, not one file: §1.3 says one complete self-contained fact per
    bullet, so a marker whose tokens are scattered across two bullets has
    not been captured as a fact — it has been captured as two fragments.

    A CORRECTLY ROUTED hit wins over any other. Cross-referencing is the
    reference behaviour, not a defect — the Ielts file naming the tutor and a
    Majid Tajik file existing are both right — so scanning `bodies` in dict
    order and reporting the first hit made a well-curated corpus read as
    misrouted purely on slug alphabetisation. Measured on CI run 32429017640:
    P06 was scored MISROUTED to `areas/ielts` while `people/majid-tajik`
    existed in the very same run. Misroute means "the fact is NOT in the file
    the label names", so the routed file has to be looked for before the
    verdict is written.
    """
    fallback: Optional[Tuple[str, str]] = None
    for slug, body in bodies.items():
        for bullet in bullets_of(body):
            n = norm(bullet)
            if all(norm(tok) in n for tok in marker.all_of):
                if capture_is_routed(marker, slug):
                    return slug, bullet
                if fallback is None:
                    fallback = (slug, bullet)
    return fallback


def capture_is_routed(marker: Capture, slug: str) -> bool:
    """Did it land where the label says it should?"""
    from app.memory_files import section_of_slug

    if marker.file:
        return slug == marker.file
    if marker.section:
        section = section_of_slug(slug)
        return section is not None and section.value == marker.section
    return True


def find_reject(marker: Reject, bodies: Dict[str, str]) -> Optional[str]:
    """The body text where all of `all_of` appear TOGETHER, else None.

    Per-file rather than per-bullet on purpose: junk that got split across
    two bullets of the same file is still junk that was stored.
    """
    for slug, body in bodies.items():
        n = norm(body)
        if all(norm(tok) in n for tok in marker.all_of):
            return f"{slug} :: {body.strip()[:200]}"
    return None


# ── Lint over everything the writer left behind ──────────────────────────

@dataclass
class LintReport:
    bullets_total: int = 0
    bullet_problems: List[str] = field(default_factory=list)
    descriptions_total: int = 0
    description_problems: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.bullet_problems and not self.description_problems


def lint_files(files: Sequence[Any]) -> LintReport:
    """Every bullet through `bullet_problem`, every description through
    `DESCRIPTION_RE`. This is the deterministic half of §1.3/§1.4, applied
    to what the REAL writer produced rather than to a hand-written fixture.
    """
    from app.memory_files import (
        CURRENT_CONTEXT_SLUG, bullet_problem, description_problem, is_bullet_list,
    )

    report = LintReport()
    for f in files:
        body = f.body_md or ""
        # Current context is `##` layer headings with prose (§6), not a
        # bullet list — WS-3 owns its shape and the bullet lint does not
        # apply to it.
        if f.slug != CURRENT_CONTEXT_SLUG and body.strip():
            if not is_bullet_list(body):
                report.bullet_problems.append(
                    f"{f.slug}: body is not a bullet list"
                )
            for bullet in bullets_of(body):
                report.bullets_total += 1
                problem = bullet_problem(bullet)
                if problem:
                    report.bullet_problems.append(f"{f.slug}: {problem} :: {bullet}")
        if f.description:
            report.descriptions_total += 1
            problem = description_problem(f.description)
            if problem:
                report.description_problems.append(f"{f.slug}: {problem}")
    return report


# ── One scenario's verdict, as plain data ────────────────────────────────

@dataclass
class ScenarioResult:
    scenario_id: str
    #: Capture markers found, as id -> (slug, bullet).
    captured: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    missed: List[str] = field(default_factory=list)
    #: Captured but in the WRONG file.
    misrouted: List[str] = field(default_factory=list)
    #: Reject markers that were stored anyway.
    junk: List[str] = field(default_factory=list)
    #: Slugs the label forbids that exist anyway.
    forbidden_slugs: List[str] = field(default_factory=list)
    #: Section-cardinality violations ("exactly one people file").
    cardinality: List[str] = field(default_factory=list)
    lint: LintReport = field(default_factory=LintReport)
    slugs: List[str] = field(default_factory=list)
    bodies: Dict[str, str] = field(default_factory=dict)
    changes: List[str] = field(default_factory=list)
    #: What the writer itself said it refused, plus any pre-gate skip.
    rejected: List[str] = field(default_factory=list)
    applied: int = 0

    def describe(self) -> str:
        lines = [
            f"scenario={self.scenario_id} applied={self.applied} "
            f"files={self.slugs}"
        ]
        if self.missed:
            lines.append("  MISSED (must be captured, was not):")
            lines += [f"    - {m}" for m in self.missed]
        if self.misrouted:
            lines.append("  MISROUTED (captured, wrong file):")
            lines += [f"    - {m}" for m in self.misrouted]
        if self.rejected:
            lines.append("  THE WRITER REFUSED:")
            lines += [f"    - {r}" for r in self.rejected]
        if self.junk:
            lines.append("  JUNK (must NOT be stored, was):")
            lines += [f"    - {j}" for j in self.junk]
        if self.forbidden_slugs:
            lines.append(f"  FORBIDDEN FILES: {self.forbidden_slugs}")
        if self.cardinality:
            lines += [f"  CARDINALITY: {c}" for c in self.cardinality]
        if not self.lint.clean:
            lines.append("  LINT:")
            lines += [f"    - {p}" for p in self.lint.bullet_problems]
            lines += [f"    - {p}" for p in self.lint.description_problems]
        for slug, body in sorted(self.bodies.items()):
            if body.strip():
                lines.append(f"  [{slug}]")
                lines += [f"      {ln}" for ln in body.strip().splitlines()]
        return "\n".join(lines)


async def run_scenario(db, user_id: str, sc: Scenario, *, dirty: bool = False):
    from app.memory_files import SYSTEM_FILES, section_of_slug

    rejected: List[str] = []
    applied = (
        await drive_conversation_dirty(db, user_id, sc.turns, rejected) if dirty
        else await drive_conversation(db, user_id, sc.turns, rejected)
    )

    files = await all_files(db, user_id)
    bodies = {f.slug: (f.body_md or "") for f in files}
    res = ScenarioResult(
        scenario_id=sc.id + ("[dirty]" if dirty else ""),
        applied=applied,
        slugs=sorted(bodies),
        bodies=bodies,
        changes=await change_summaries(db, user_id),
        rejected=rejected,
        lint=lint_files(files),
    )

    for marker in sc.must_capture:
        hit = find_capture(marker, bodies)
        if hit is None:
            res.missed.append(marker.id)
            continue
        slug, bullet = hit
        res.captured[marker.id] = (slug, bullet)
        if not capture_is_routed(marker, slug):
            want = marker.file or f"section={marker.section}"
            res.misrouted.append(f"{marker.id}: wanted {want}, got {slug}")

    for marker in sc.must_reject:
        hit = find_reject(marker, bodies)
        if hit:
            res.junk.append(f"{marker.id} :: {hit}")

    res.forbidden_slugs = [s for s in sc.forbid_slugs if s in bodies]

    if sc.exactly_one_in_section:
        want = sc.exactly_one_in_section
        in_section = [
            s for s in bodies
            if s not in SYSTEM_FILES
            and (section_of_slug(s).value if section_of_slug(s) else None) == want
        ]
        if len(in_section) != 1:
            res.cardinality.append(
                f"expected exactly 1 file in section {want!r}, got {in_section}"
            )

    return res
