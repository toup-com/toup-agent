"""Skills index — the maintenance agent's SINGLE SOURCE OF TRUTH.

Loads the platform skills under ``docs/skills/``: the master router
(``toup/SKILL.md``) and one ``SKILL.md`` per subsystem. Provides:

* ``list_skills()``            — every subsystem skill + frontmatter.
* ``parse_router_table()``     — the master router's symptom→subsystem rows.
* ``rank_subsystems(text)``    — deterministic keyword routing (LLM-free
                                 fallback + candidate generation).
* ``load_skill(name)``         — full markdown for one skill.
* ``failure_modes(name)``      — the skill's "## 6. Common failure modes"
                                 section (the per-symptom router the agent
                                 greps to locate files/lines).
* ``baseline_sha()``           — git SHA the skills currently reflect
                                 (provenance recorded on every issue).

If no skill covers an area, callers must FLAG THE GAP (propose a new
skill) rather than fix blind — see ``router.py`` / ``diagnoser.py``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.config import settings


# ── Locating the skills dir ──────────────────────────────────────────

def repo_root() -> Path:
    """Repo root: settings.support_repo_dir override, else inferred from this file."""
    override = getattr(settings, "support_repo_dir", None)
    if override:
        return Path(override).resolve()
    # backend/app/support/skills_index.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def skills_dir() -> Path:
    return repo_root() / "docs" / "skills"


def master_router_path() -> Path:
    return skills_dir() / "toup" / "SKILL.md"


# ── Data shapes ──────────────────────────────────────────────────────

@dataclass
class SkillMeta:
    name: str
    path: str            # relative to repo root
    description: str
    exists: bool = True


@dataclass
class RouterRow:
    symptom: str
    skill: str           # primary subsystem skill name
    note: str = ""


@dataclass
class RankedSkill:
    name: str
    score: float
    reasons: list = field(default_factory=list)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Markdown links like [credits-and-enforcement](../credits-and-enforcement/SKILL.md)
_SKILL_LINK_RE = re.compile(r"\[([a-z0-9-]+)\]\(\.\./([a-z0-9-]+)/SKILL\.md\)")
_STOPWORDS = frozenset(
    "the a an and or of to in on for is are was were be been with without "
    "my your our it its this that these those i you we they not no yes can "
    "could should would will shall do does did has have had at by from as "
    "but if then than into out up down over under again app agent user toup "
    "when while after before what why how where which who whom".split()
)


def _parse_frontmatter(text: str) -> dict:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    out: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def _tokens(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z][a-z0-9_]{2,}", (text or "").lower())
        if w not in _STOPWORDS
    }


# ── Public API ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def list_skills() -> list:
    """Every subsystem skill under docs/skills/ with its frontmatter.

    Excludes the master router itself (``toup``). Cached; call
    ``refresh()`` after the skills change on disk.
    """
    out: list = []
    base = skills_dir()
    if not base.is_dir():
        return out
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name == "toup":
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            continue
        fm = _parse_frontmatter(skill_md.read_text(encoding="utf-8", errors="replace"))
        out.append(SkillMeta(
            name=fm.get("name", child.name),
            path=str(skill_md.relative_to(repo_root())),
            description=fm.get("description", ""),
        ))
    return out


def skill_names() -> set:
    return {s.name for s in list_skills()}


@lru_cache(maxsize=1)
def parse_router_table() -> list:
    """Parse the master router's symptom→subsystem rows (section 3).

    The router uses markdown tables ``| Symptom | Skill |`` where Skill is
    a ``[name](../name/SKILL.md)`` link (plus an optional parenthetical
    note). Returns every (symptom, primary-skill, note) row found.
    """
    rows: list = []
    p = master_router_path()
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or "---" in line.split("|")[1:2]:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        symptom, skill_cell = cells[0], cells[1]
        # Header row / non-symptom rows have no skill link.
        links = _SKILL_LINK_RE.findall(skill_cell)
        if not links or symptom.lower() in ("symptom", ""):
            continue
        primary = links[0][0]
        note = skill_cell  # keep the full cell (carries the parenthetical hint)
        rows.append(RouterRow(symptom=symptom, skill=primary, note=note))
    return rows


def rank_subsystems(text: str, top_n: int = 5) -> list:
    """Deterministic keyword routing: score each subsystem skill against
    the report text using (a) the router's symptom rows and (b) the skill
    descriptions. Returns the top_n RankedSkill, best first.

    This is the LLM-free fallback AND the candidate set the LLM router
    chooses from — so routing degrades gracefully if the LLM is down.
    """
    q = _tokens(text)
    if not q:
        return []
    scores: dict = {}
    reasons: dict = {}

    # (a) router symptom rows — strongest signal (curated symptom→skill map)
    for row in parse_router_table():
        overlap = q & _tokens(row.symptom)
        if overlap:
            s = 2.0 * len(overlap)
            scores[row.skill] = scores.get(row.skill, 0.0) + s
            reasons.setdefault(row.skill, []).append(
                f"router symptom “{row.symptom[:60]}” (+{len(overlap)} kw)"
            )

    # (b) skill descriptions — secondary signal
    for sk in list_skills():
        overlap = q & _tokens(sk.description)
        if overlap:
            scores[sk.name] = scores.get(sk.name, 0.0) + 1.0 * len(overlap)
            reasons.setdefault(sk.name, []).append(f"description (+{len(overlap)} kw)")

    ranked = [
        RankedSkill(name=n, score=round(sc, 2), reasons=reasons.get(n, []))
        for n, sc in scores.items()
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_n]


def load_skill(name: str) -> Optional[str]:
    """Full markdown for one subsystem skill (or None if absent)."""
    p = skills_dir() / name / "SKILL.md"
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def failure_modes(name: str) -> Optional[str]:
    """The skill's '## 6. Common failure modes …' section (the router the
    agent greps for file/line pointers). Falls back to None if not found.
    """
    md = load_skill(name)
    if not md:
        return None
    lines = md.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^##\s*6\.", ln):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^##\s*7\.", lines[j]):
            end = j
            break
    return "\n".join(lines[start:end]).strip()


def baseline_sha() -> Optional[str]:
    """Short git SHA the skills currently reflect (provenance)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root()), capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def refresh() -> None:
    """Drop caches (call after the skills change on disk)."""
    list_skills.cache_clear()
    parse_router_table.cache_clear()
