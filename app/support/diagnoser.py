"""Diagnose — root-cause a BUG grounded in the routed subsystem skill(s).

The skill is the source of truth: its section 6 maps symptom→file:line and
its section 5 lists the invariants/footguns the fix must NOT violate. We
load the routed skill(s), pull REAL excerpts of the files they cite (bounded,
concurrency-capped to stay under provider/IO limits), and ask the LLM for a
structured diagnosis report: root cause, blast radius, proposed fix, affected
files, test plan, rollback path, and the invariants to preserve.

Every diagnosis references the subsystem skill(s) it relied on. If routing
found no covering skill, the caller parks the issue instead of guessing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import settings
from app.support import skills_index
from app.support.llm import ask_json

logger = logging.getLogger("support.diagnoser")

# repo-relative source paths cited inside a skill
_PATH_RE = re.compile(
    r"(?<![\w./-])((?:backend|frontend|bridge|extensions|cloudflare|\.github)/[\w./-]+\.[A-Za-z0-9]+)"
)

_SYSTEM = (
    "You are a senior engineer diagnosing a BUG in the Toup platform. You are "
    "given (1) the problem report, (2) the authoritative SKILL doc(s) for the "
    "implicated subsystem(s) — treat these as ground truth for architecture, "
    "file locations, and invariants — and (3) real excerpts of the cited source "
    "files. Produce a precise, actionable diagnosis.\n\n"
    "RULES:\n"
    "- Cite EXACT files (repo-relative) and, where the skill or code gives them, "
    "line numbers / symbols. Do not invent files.\n"
    "- The fix MUST NOT violate any invariant/footgun the skill lists — copy the "
    "ones the fix must respect into invariants_to_preserve.\n"
    "- If the evidence is insufficient to root-cause, say so (low confidence) "
    "rather than guessing.\n\n"
    "Output JSON with keys: root_cause (str), blast_radius (str), "
    "proposed_fix (str), affected_files (list[str], repo-relative), "
    "test_plan (str), rollback_path (str), invariants_to_preserve (list[str]), "
    "skills_referenced (list[str]), reproducible (bool), repro_steps (str), "
    "confidence (0.0-1.0)."
)


@dataclass
class DiagnosisResult:
    report: dict
    summary: str
    skills_referenced: list = field(default_factory=list)
    ok: bool = True


def _read_excerpt(rel_path: str, max_lines: int) -> Optional[str]:
    try:
        p = (skills_index.repo_root() / rel_path).resolve()
        # Stay inside the repo.
        if not str(p).startswith(str(skills_index.repo_root().resolve())):
            return None
        if not p.is_file():
            return None
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        head = lines[:max_lines]
        body = "\n".join(f"{i+1}: {ln}" for i, ln in enumerate(head))
        suffix = "" if len(lines) <= max_lines else f"\n… (+{len(lines) - max_lines} more lines)"
        return f"// {rel_path} (first {min(len(lines), max_lines)} lines)\n{body}{suffix}"
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[support.diagnoser] excerpt read failed %s: %s", rel_path, e)
        return None


async def _gather_code_context(skill_md_list: list) -> str:
    """Pull bounded, concurrency-capped excerpts of files the skills cite."""
    max_files = int(getattr(settings, "support_diagnose_max_files", 6))
    max_lines = int(getattr(settings, "support_diagnose_max_file_lines", 160))
    cap = max(1, int(getattr(settings, "support_diagnose_concurrency", 3)))

    seen: list = []
    for md in skill_md_list:
        for m in _PATH_RE.finditer(md or ""):
            path = m.group(1)
            if path not in seen:
                seen.append(path)
    targets = seen[:max_files]
    if not targets:
        return ""

    sem = asyncio.Semaphore(cap)

    async def _one(path: str) -> Optional[str]:
        async with sem:  # cap concurrent file IO (mirrors the lane-cap pattern)
            return await asyncio.to_thread(_read_excerpt, path, max_lines)

    excerpts = [e for e in await asyncio.gather(*[_one(p) for p in targets]) if e]
    if len(seen) > max_files:
        excerpts.append(f"// … {len(seen) - max_files} more cited files omitted for budget")
    return "\n\n".join(excerpts)


async def diagnose(
    *,
    user_id: str,
    symptom: str,
    raw_report: str,
    repro_info: str,
    subsystems: list,
    prior_decision_notes: str = "",
) -> DiagnosisResult:
    skill_blocks = []
    loaded = []
    for name in subsystems:
        md = skills_index.load_skill(name)
        if not md:
            continue
        loaded.append(name)
        fm = skills_index.failure_modes(name) or ""
        # Lead with section 6 (the router) then the full skill, capped.
        skill_blocks.append(
            f"===== SKILL: {name} =====\n"
            f"--- Failure modes (section 6) ---\n{fm}\n\n"
            f"--- Full skill ---\n{md[:12000]}"
        )

    if not loaded:
        return DiagnosisResult(
            report={
                "root_cause": "No covering subsystem skill could be loaded.",
                "confidence": 0.0,
            },
            summary="No subsystem skill available to ground a diagnosis.",
            skills_referenced=[],
            ok=False,
        )

    code_ctx = await _gather_code_context([skills_index.load_skill(n) or "" for n in loaded])

    prompt = (
        f"PROBLEM REPORT:\n{raw_report}\n\n"
        f"NORMALIZED SYMPTOM: {symptom}\n"
        f"REPRO / EXTRA INFO: {repro_info or '(none provided)'}\n"
    )
    if prior_decision_notes:
        prompt += f"\nADMIN REQUESTED CHANGES (address these):\n{prior_decision_notes}\n"
    prompt += (
        f"\nIMPLICATED SUBSYSTEM SKILL(S) — GROUND TRUTH:\n\n"
        + "\n\n".join(skill_blocks)
        + f"\n\nREAL SOURCE EXCERPTS OF CITED FILES:\n{code_ctx or '(none extracted)'}"
    )

    data = await ask_json(
        user_id=user_id,
        operation_type="system.support.diagnose",
        system=_SYSTEM,
        prompt=prompt,
        max_tokens=2200,
        timeout=120,
    )

    if not data:
        return DiagnosisResult(
            report={
                "root_cause": "Diagnosis LLM unavailable or returned no parseable result.",
                "confidence": 0.0,
                "skills_referenced": loaded,
            },
            summary="Diagnosis could not be produced (LLM unavailable).",
            skills_referenced=loaded,
            ok=False,
        )

    # Ensure provenance: the skills we actually grounded on are referenced.
    refs = data.get("skills_referenced") or []
    for n in loaded:
        if n not in refs:
            refs.append(n)
    data["skills_referenced"] = refs

    summary = (data.get("root_cause") or "").strip()[:600]
    return DiagnosisResult(report=data, summary=summary, skills_referenced=loaded, ok=True)
