"""Triage & route — map a symptom to the subsystem skill(s) that own it.

Reads the master router skill (docs/skills/toup/SKILL.md) via the skills
index, generates deterministic candidates by keyword, then asks the LLM to
pick the most relevant subsystem(s) FROM THOSE CANDIDATES (constrained — it
cannot invent a skill that doesn't exist). If nothing covers the area, this
flags a COVERAGE GAP so the caller proposes a new skill rather than fixing
blind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.support import skills_index
from app.support.llm import ask_json

_SYSTEM = (
    "You route a Toup platform problem report to the subsystem SKILL(s) that "
    "own it. You are given the candidate subsystems (name + one-line scope) "
    "drawn from the platform's master router. Choose the 1-3 MOST relevant "
    "candidates, best first. You may ONLY choose from the provided candidate "
    "names — never invent a name. If NONE of the candidates plausibly own the "
    "problem, return an empty list and set coverage_gap=true with a short note "
    "on what subsystem a new skill should cover. Output JSON: "
    "{\"subsystems\": [\"name\", ...], \"coverage_gap\": false, "
    "\"rationale\": \"why\", \"gap_note\": \"\"}."
)


@dataclass
class RoutingResult:
    subsystems: list = field(default_factory=list)   # ordered skill names
    coverage_gap: bool = False
    rationale: str = ""
    gap_note: str = ""
    candidates: list = field(default_factory=list)   # [(name, score)] for transparency


async def route(*, user_id: str, symptom: str, raw_report: str = "") -> RoutingResult:
    text = f"{symptom}\n{raw_report}".strip()
    ranked = skills_index.rank_subsystems(text, top_n=6)
    valid = skills_index.skill_names()
    candidates = [(r.name, r.score) for r in ranked]

    if not ranked:
        # Nothing matched any router symptom or description.
        return RoutingResult(
            subsystems=[], coverage_gap=True,
            rationale="No subsystem skill matched the report by keyword.",
            gap_note="Report did not match any documented subsystem; a new skill may be needed.",
            candidates=candidates,
        )

    cand_block = "\n".join(
        f"- {r.name}: {next((s.description for s in skills_index.list_skills() if s.name == r.name), '')[:200]}"
        for r in ranked
    )
    data = await ask_json(
        user_id=user_id,
        operation_type="system.support.route",
        system=_SYSTEM,
        prompt=f"REPORT:\n{text}\n\nCANDIDATE SUBSYSTEMS:\n{cand_block}",
        max_tokens=500,
    )

    if not data:
        # LLM unavailable → fall back to deterministic top candidate(s).
        top = [r.name for r in ranked[:2]]
        return RoutingResult(
            subsystems=top, coverage_gap=False,
            rationale="LLM router unavailable; used deterministic keyword ranking.",
            candidates=candidates,
        )

    chosen = [s for s in (data.get("subsystems") or []) if s in valid]
    coverage_gap = bool(data.get("coverage_gap")) or not chosen
    return RoutingResult(
        subsystems=chosen,
        coverage_gap=coverage_gap,
        rationale=str(data.get("rationale", "")).strip(),
        gap_note=str(data.get("gap_note", "")).strip(),
        candidates=candidates,
    )
