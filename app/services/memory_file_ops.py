"""The memory-file store and its strict ops engine (v3 §1.3–§2.2).

Everything that reads or writes `memory_files.body_md` goes through here.
The LLM half — prompts, the model call, the retry — lives in
`memory_curator.py`; this module is deterministic and testable without a
key, which is what lets `make memory-verify-ci` keep teeth without an LLM.

The op set (v3 §2.2):

    create_file        {section, slug, title, description}
    update_description {slug, description}
    add                {slug, bullet, change}
    rewrite            {slug, match, bullet, change}
    remove             {slug, match, change}
    link               {slug, links: [slug]}
    delete_file        {slug, change}

`match` is the EXACT existing bullet text. No index arithmetic: a model
that miscounts by one silently rewrites the wrong fact, and a body that
changed between the read and the write makes every index stale.

Validation is a SIMULATION. Ops are applied to an in-memory copy of the
touched files in order, so `create_file` followed by `add` to that slug
validates, `rewrite` sees the bullet a previous `add` produced, and the
8 KB per-file cap is measured against the body the batch would actually
leave behind. `apply_ops` then persists exactly the state validation
computed — the two can never disagree, because there is only one walk.

An invalid op skips THAT op and appends a complaint; the caller may retry
once with the complaints appended (memory_curator does). Nothing is
partially applied: the persist step writes whole bodies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryFile, MemoryFileChange
from app.memory_files import (
    ALWAYS_INJECTED_SLUGS,
    CURRENT_CONTEXT_SLUG,
    LEARNED_SLUG,
    MAX_BODY_CHARS,
    PROFILE_SLUG,
    SECTION_ORDER,
    SYSTEM_FILES,
    FileSection,
    bullet_problem,
    description_problem,
    extract_links,
    is_bullet_list,
    is_valid_slug,
    parse_bullets,
    render_bullets,
    section_of_slug,
    slugify,
    title_from_slug,
)
from app.services.user_identity import UserIdentity

logger = logging.getLogger(__name__)

MAX_OPS = 40
MAX_LINKS_PER_FILE = 8
MAX_CHANGE_CHARS = 300

#: kind written to memory_file_changes, per op.
_CHANGE_KIND = {
    "create_file": "created",
    "update_description": "updated",
    "add": "updated",
    "rewrite": "updated",
    "remove": "removed",
    "delete_file": "file_deleted",
}


# ── In-memory file state (the simulation) ─────────────────────────────

@dataclass
class FileState:
    slug: str
    section: str
    title: str
    description: Optional[str] = None
    bullets: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    is_system: bool = False
    raw_body: Optional[str] = None
    #: True when the file has no row yet (a create_file in this batch).
    is_new: bool = False
    #: True when a delete_file op in this batch removed it.
    deleted: bool = False
    #: True when a BULLET op changed the body. Deliberately NOT set by
    #: `link` or `update_description`: those write their own columns, and
    #: re-rendering the body for them would reflow it for no reason.
    dirty: bool = False

    @property
    def has_prose(self) -> bool:
        """True when the stored body holds anything that is not a bullet.

        Current context is `##` layer headings with prose beneath (v3 §6),
        and `parse_bullets` cannot see any of it. Re-rendering such a body
        from `bullets` would DELETE the layers — so a bullet op on a prose
        body is refused rather than applied. WS-3's updater owns that file's
        shape; the curator must not flatten it on the way past.
        """
        return not is_bullet_list(self.raw_body)

    def body(self) -> str:
        """The body this state renders to. Byte-exact unless a bullet op
        touched it."""
        if not self.dirty and self.raw_body is not None:
            return self.raw_body
        return render_bullets(self.bullets)


def _state_from_row(row: MemoryFile) -> FileState:
    return FileState(
        slug=row.slug,
        section=row.section,
        title=row.title,
        description=row.description,
        bullets=parse_bullets(row.body_md),
        links=_load_links(row),
        is_system=bool(row.is_system),
        raw_body=row.body_md,
    )


def _load_links(row: MemoryFile) -> List[str]:
    try:
        parsed = json.loads(row.links_json) if row.links_json else []
        return [s for s in parsed if isinstance(s, str)]
    except Exception:
        return []


# ── Reads ─────────────────────────────────────────────────────────────

async def _all_files(db: AsyncSession, user_id: str) -> List[MemoryFile]:
    """Every v3 file row for a user, one query.

    Round-8 rows (sections `knowledge`, `working`, `preferences`, the bare
    `profile`) are filtered OUT rather than shown in a v3 section they were
    never written for. They stay on disk untouched — that is the rollback —
    until WS-5's migration folds them into v3 files.
    """
    rows = (await db.execute(
        select(MemoryFile).where(MemoryFile.user_id == user_id)
    )).scalars().all()
    valid = {s.value for s in SECTION_ORDER}
    return [r for r in rows if r.section in valid and section_of_slug(r.slug) is not None]


def _sort_key(row: MemoryFile) -> Tuple[int, int, str]:
    order = [s.value for s in SECTION_ORDER]
    return (
        order.index(row.section) if row.section in order else len(order),
        0 if row.is_system else 1,
        (row.title or "").lower(),
    )


async def ensure_system_files(db: AsyncSession, user_id: str) -> List[MemoryFile]:
    """Create the three fixed files if they are missing. Agent-side only.

    Idempotent and cheap. A file born here has an EMPTY body and the canon
    description from `SYSTEM_FILES` — the one place a description is not
    generated, because these three exist before the writer has ever run.
    """
    existing = {
        r.slug: r for r in (await db.execute(
            select(MemoryFile).where(
                and_(MemoryFile.user_id == user_id,
                     MemoryFile.slug.in_(list(SYSTEM_FILES)))
            )
        )).scalars().all()
    }
    created: List[MemoryFile] = []
    for position, (slug, spec) in enumerate(SYSTEM_FILES.items()):
        row = existing.get(slug)
        if row is None:
            row = MemoryFile(
                user_id=user_id,
                slug=slug,
                section=spec["section"],
                title=spec["title"],
                description=spec["description"],
                body_md="",
                is_system=True,
                position=position,
            )
            db.add(row)
            created.append(row)
        else:
            # ADOPTION, and it is the path EVERY already-live tenant takes.
            #
            # A round-8 `learned` row carries the same slug and the same
            # meaning, so it is adopted rather than colliding with the unique
            # index. The trap: round 8's rows carry a `purpose`-era
            # description ("Corrections and lessons.") that fails
            # DESCRIPTION_RE — and that string is BOTH served to the client
            # and injected into the prompt as the index line for this file.
            # Repairing it only when it is EMPTY (the old condition) left
            # every existing tenant with an invalid one forever, which is
            # also a lint failure the eval set would report against a writer
            # that never wrote it.
            if row.section != spec["section"]:
                row.section = spec["section"]
            if not row.is_system:
                row.is_system = True
            if not row.title:
                row.title = spec["title"]
            if description_problem(row.description):
                row.description = spec["description"]
    if created:
        await db.flush()
    return created


def file_summary(row: MemoryFile) -> Dict[str, Any]:
    """The listing payload for one file. No engine metadata, by construction:
    this dict is the complete set of keys a client ever sees for a file row."""
    return {
        "slug": row.slug,
        "section": row.section,
        "title": row.title,
        "description": row.description,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def list_files(db: AsyncSession, user_id: str) -> Dict[str, Any]:
    """`{sections: [{section, files: [...]}]}` in canonical section order."""
    await ensure_system_files(db, user_id)
    await db.commit()
    rows = sorted(await _all_files(db, user_id), key=_sort_key)
    by_section: Dict[str, List[Dict[str, Any]]] = {s.value: [] for s in SECTION_ORDER}
    for row in rows:
        by_section[row.section].append(file_summary(row))
    return {
        "sections": [
            {"section": s.value, "files": by_section[s.value]} for s in SECTION_ORDER
        ]
    }


async def get_file(
    db: AsyncSession, user_id: str, slug: str
) -> Optional[Dict[str, Any]]:
    """One file with its body and its links resolved to titles.

    None when the slug is malformed or unknown. A system slug always
    resolves — `ensure_system_files` makes it exist first.
    """
    if not is_valid_slug(slug):
        return None
    if slug in SYSTEM_FILES:
        await ensure_system_files(db, user_id)
        await db.commit()
    row = (await db.execute(
        select(MemoryFile).where(
            and_(MemoryFile.user_id == user_id, MemoryFile.slug == slug)
        )
    )).scalar_one_or_none()
    if row is None or section_of_slug(row.slug) is None:
        return None

    body = row.body_md or ""
    # Links are the union of the stored list and whatever `[[slug]]` the
    # body actually mentions — the body is the thing the user reads, so a
    # link they can see must be tappable even if a `link` op never ran.
    wanted: List[str] = []
    for s in list(_load_links(row)) + extract_links(body):
        if s not in wanted:
            wanted.append(s)
    titles: Dict[str, str] = {}
    if wanted:
        for other in (await db.execute(
            select(MemoryFile.slug, MemoryFile.title).where(
                and_(MemoryFile.user_id == user_id, MemoryFile.slug.in_(wanted))
            )
        )).all():
            titles[other[0]] = other[1]
    return {
        "slug": row.slug,
        "section": row.section,
        "title": row.title,
        "description": row.description,
        "body_md": body,
        "links": [
            {"slug": s, "title": titles.get(s) or title_from_slug(s)} for s in wanted
        ],
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


_SNIPPET_RADIUS = 90


def _snippet(body: str, needle: str) -> str:
    """The line the match landed on, trimmed around it."""
    low = body.lower()
    at = low.find(needle)
    if at < 0:
        first = next((ln for ln in body.splitlines() if ln.strip()), "")
        return first.lstrip("-* ").strip()[: _SNIPPET_RADIUS * 2]
    start = body.rfind("\n", 0, at) + 1
    end = body.find("\n", at)
    line = body[start: end if end > 0 else len(body)].lstrip("-* ").strip()
    if len(line) <= _SNIPPET_RADIUS * 2:
        return line
    rel = max(0, line.lower().find(needle) - _SNIPPET_RADIUS)
    return ("…" if rel else "") + line[rel: rel + _SNIPPET_RADIUS * 2] + "…"


async def search_files(
    db: AsyncSession, user_id: str, query: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Title + body substring search, file-attributed.

    LIKE rather than tsvector on purpose: `memory_files` is tens of rows per
    user, the bodies are small, and the same statement has to run on sqlite
    (every test lane) and Postgres (production) with identical results.
    """
    q = (query or "").strip()
    if not q:
        return []
    needle = f"%{q.lower()}%"
    rows = (await db.execute(
        select(MemoryFile).where(
            and_(
                MemoryFile.user_id == user_id,
                or_(
                    MemoryFile.title.ilike(needle),
                    MemoryFile.body_md.ilike(needle),
                    MemoryFile.description.ilike(needle),
                ),
            )
        )
    )).scalars().all()
    rows = [r for r in rows if section_of_slug(r.slug) is not None]
    rows.sort(key=_sort_key)
    out: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        out.append({
            "slug": row.slug,
            "title": row.title,
            "snippet": _snippet(row.body_md or row.description or "", q.lower()),
        })
    return out


async def read_log(db: AsyncSession, user_id: str, month: str) -> Dict[str, Any]:
    """`{days: [{day, entries: [...]}]}` for one YYYY-MM, newest day first."""
    if not re.fullmatch(r"\d{4}-\d{2}", month or ""):
        raise ValueError("month must be YYYY-MM")
    rows = (await db.execute(
        select(MemoryFileChange).where(
            and_(
                MemoryFileChange.user_id == user_id,
                MemoryFileChange.day_key.like(f"{month}-%"),
            )
        ).order_by(MemoryFileChange.created_at.desc())
    )).scalars().all()
    days: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        days.setdefault(row.day_key, []).append({
            "file_slug": row.file_slug,
            "file_title": row.file_title,
            "kind": row.kind,
            "summary": row.summary,
            "at": row.created_at.isoformat() if row.created_at else None,
        })
    return {
        "days": [
            {"day": day, "entries": entries}
            for day, entries in sorted(days.items(), reverse=True)
        ]
    }


async def delete_file(db: AsyncSession, user_id: str, slug: str) -> bool:
    """Delete a file, or empty it when it is a system file.

    False when the slug resolves to nothing (the route turns that into 404).
    """
    if not is_valid_slug(slug):
        return False
    row = (await db.execute(
        select(MemoryFile).where(
            and_(MemoryFile.user_id == user_id, MemoryFile.slug == slug)
        )
    )).scalar_one_or_none()
    if row is None:
        return False
    title = row.title
    if slug in SYSTEM_FILES or row.is_system:
        row.body_md = ""
        row.links_json = None
        row.updated_at = datetime.utcnow()
    else:
        await db.delete(row)
    await _write_change(
        db, user_id, slug, title, "file_deleted",
        f"Deleted {title}." if slug not in SYSTEM_FILES else f"Cleared {title}.",
    )
    await db.commit()
    return True


async def forget_everything(db: AsyncSession, user_id: str) -> int:
    """Drop every file and every change row. Returns the files removed."""
    rows = (await db.execute(
        select(MemoryFile).where(MemoryFile.user_id == user_id)
    )).scalars().all()
    for row in rows:
        await db.delete(row)
    changes = (await db.execute(
        select(MemoryFileChange).where(MemoryFileChange.user_id == user_id)
    )).scalars().all()
    for change in changes:
        await db.delete(change)
    await db.commit()
    return len(rows)


# ── The injection load (§3.1) ─────────────────────────────────────────

@dataclass
class BrainFiles:
    profile: str = ""
    current_context: str = ""
    learned: str = ""
    #: (title, description) for every other non-empty file.
    index: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    #: (title, body) for up to N lexically relevant files.
    relevant: List[Tuple[str, str]] = field(default_factory=list)
    file_count: int = 0


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "my", "me", "i", "you", "your", "it",
    "that", "this", "what", "do", "does", "did", "can", "how", "about",
})


def _tokens(text: str) -> Set[str]:
    return {
        t for t in (w.lower() for w in _WORD_RE.findall(text or ""))
        if len(t) > 2 and t not in _STOPWORDS
    }


def lexical_rank(query: str, rows: Sequence[MemoryFile]) -> List[MemoryFile]:
    """Files ordered by overlap with the query, best first, zeroes dropped.

    Deliberately not an embedding: this runs on every non-trivial turn, the
    corpus is tens of files, and a title/description match is the signal the
    index already advertises to the model.
    """
    q = _tokens(query)
    if not q:
        return []
    scored: List[Tuple[float, str, MemoryFile]] = []
    for row in rows:
        title_hits = len(q & _tokens(row.title or ""))
        desc_hits = len(q & _tokens(row.description or ""))
        body_hits = len(q & _tokens(row.body_md or ""))
        score = title_hits * 3 + desc_hits * 2 + body_hits
        if score:
            scored.append((-score, row.slug, row))
    scored.sort(key=lambda s: (s[0], s[1]))
    return [row for _, _, row in scored]


async def load_brain(
    db: AsyncSession,
    user_id: str,
    query: str = "",
    *,
    max_relevant: int = 2,
    max_index: int = 40,
) -> BrainFiles:
    """Everything the prompt needs, in ONE query.

    Round 8's index called `list_files`, which ran a full per-user scan of
    `memories`, and injecting two files in full would have run it three
    times per turn. `memory_files` is tens of rows: read them once.
    """
    rows = await _all_files(db, user_id)
    by_slug = {r.slug: r for r in rows}
    brain = BrainFiles(file_count=len(rows))
    brain.profile = (by_slug.get(PROFILE_SLUG).body_md or "") if PROFILE_SLUG in by_slug else ""
    brain.current_context = (
        (by_slug.get(CURRENT_CONTEXT_SLUG).body_md or "")
        if CURRENT_CONTEXT_SLUG in by_slug else ""
    )
    brain.learned = (by_slug.get(LEARNED_SLUG).body_md or "") if LEARNED_SLUG in by_slug else ""

    others = [
        r for r in sorted(rows, key=_sort_key)
        if r.slug not in ALWAYS_INJECTED_SLUGS and (r.body_md or "").strip()
    ]
    brain.index = [(r.title, r.description) for r in others][:max_index]
    brain.relevant = [
        (r.title, r.body_md or "") for r in lexical_rank(query, others)[:max_relevant]
    ]
    return brain


# ── Validation (the simulation) ───────────────────────────────────────

@dataclass
class OpsPlan:
    accepted: List[Dict[str, Any]] = field(default_factory=list)
    complaints: List[str] = field(default_factory=list)
    states: Dict[str, FileState] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.accepted)


def _clean_change(op_name: str, raw: Any, complaints: List[str]) -> Optional[str]:
    text = (raw or "").strip() if isinstance(raw, str) else ""
    if not text:
        complaints.append(
            f"{op_name}: a `change` line is required — one short sentence for "
            "the memory log, e.g. \"Added Music: likes Googoosh.\""
        )
        return None
    return text[:MAX_CHANGE_CHARS]


def _secret_problem(text: str) -> Optional[str]:
    """The never-store tier (cards, government ids, API keys) — the one gate
    rule v3 keeps verbatim (§2.3), from the v3-owned `memory_secrets` module
    rather than from the row-era gate that is being cut."""
    try:
        from app.services.memory_secrets import sensitive_content_reason

        reason = sensitive_content_reason(text, explicit_save=True)
    except Exception:  # pragma: no cover - the gate module is always present
        return None
    if reason:
        return f"refused: {reason.removeprefix('sensitive_')} is never stored"
    return None


def validate_ops(
    raw_ops: Any,
    existing: Sequence[MemoryFile],
    *,
    identity: Optional[UserIdentity] = None,
) -> OpsPlan:
    """Walk the ops against a copy of the files and report what survives."""
    plan = OpsPlan()
    if not isinstance(raw_ops, list):
        plan.complaints.append("ops must be a list")
        return plan
    if len(raw_ops) > MAX_OPS:
        plan.complaints.append(f"too many ops ({len(raw_ops)} > {MAX_OPS})")
        return plan

    states: Dict[str, FileState] = {r.slug: _state_from_row(r) for r in existing}
    plan.states = states
    complaints = plan.complaints
    linked_slugs: List[Tuple[str, str]] = []  # (source op label, target slug)

    def _target(op_name: str, raw_slug: Any) -> Optional[FileState]:
        slug = (raw_slug or "").strip() if isinstance(raw_slug, str) else ""
        if not is_valid_slug(slug):
            complaints.append(f"{op_name}: {slug!r} is not a valid slug")
            return None
        state = states.get(slug)
        if state is None or state.deleted:
            complaints.append(f"{op_name}: no file {slug!r} — create it first")
            return None
        return state

    def _bullet_ops_allowed(state: FileState, op_name: str) -> bool:
        if not state.has_prose:
            return True
        complaints.append(
            f"{op_name} on {state.slug}: this file's body is not a bullet "
            "list (it has headings and prose) — it is rewritten by its own "
            "updater, not by ops"
        )
        return False

    def _check_bullet(op_name: str, raw: Any, state: FileState) -> Optional[str]:
        text = (raw or "").strip() if isinstance(raw, str) else ""
        problem = bullet_problem(text)
        if problem:
            complaints.append(f"{op_name} on {state.slug}: {problem}")
            return None
        secret = _secret_problem(text)
        if secret:
            complaints.append(f"{op_name} on {state.slug}: {secret}")
            return None
        for target in extract_links(text):
            linked_slugs.append((f"{op_name} on {state.slug}", target))
        return text

    def _fits(state: FileState, op_name: str) -> bool:
        if len(state.body()) <= MAX_BODY_CHARS:
            return True
        complaints.append(
            f"{op_name} on {state.slug}: file would exceed {MAX_BODY_CHARS} "
            "chars — consolidate it first (rewrite two bullets into one)"
        )
        return False

    def _self_person(slug: str, title: str) -> bool:
        if not slug.startswith("people/") or identity is None or not identity.known:
            return False
        segment = slug.split("/", 1)[1]
        return (
            identity.is_self(title)
            or any(segment == slugify(a) for a in identity.aliases)
        )

    for op in raw_ops:
        if not isinstance(op, dict):
            complaints.append("op is not an object")
            continue
        kind = op.get("op")

        if kind == "create_file":
            slug = (op.get("slug") or "").strip() if isinstance(op.get("slug"), str) else ""
            title = (op.get("title") or "").strip() if isinstance(op.get("title"), str) else ""
            declared = (op.get("section") or "").strip() if isinstance(op.get("section"), str) else ""
            if not is_valid_slug(slug):
                complaints.append(f"create_file: {slug!r} is not a valid slug")
                continue
            section = section_of_slug(slug)
            if section is None:
                complaints.append(
                    f"create_file: {slug!r} has no section — namespace it "
                    "people/, topics/ or areas/"
                )
                continue
            if declared and declared != section.value:
                complaints.append(
                    f"create_file: {slug!r} is a {section.value} file, not "
                    f"{declared!r} — the slug decides the section"
                )
                continue
            if slug in SYSTEM_FILES or (slug in states and not states[slug].deleted):
                complaints.append(f"create_file: {slug!r} already exists")
                continue
            if not title:
                complaints.append(f"create_file: {slug!r} needs a title")
                continue
            problem = description_problem(op.get("description"))
            if problem:
                complaints.append(f"create_file {slug!r}: {problem}")
                continue
            if _self_person(slug, title):
                complaints.append(
                    f"create_file: {slug!r} is the person whose memory this "
                    "is — their facts belong in you/profile, never in people/"
                )
                continue
            states[slug] = FileState(
                slug=slug, section=section.value, title=title,
                description=(op.get("description") or "").strip(),
                is_new=True, dirty=True, raw_body="",
            )
            plan.accepted.append({
                "op": "create_file", "slug": slug, "title": title,
                "section": section.value,
                "description": states[slug].description,
                "change": f"Created {title}.",
            })

        elif kind == "update_description":
            state = _target("update_description", op.get("slug"))
            if state is None:
                continue
            problem = description_problem(op.get("description"))
            if problem:
                complaints.append(f"update_description {state.slug}: {problem}")
                continue
            state.description = (op.get("description") or "").strip()
            plan.accepted.append({
                "op": "update_description", "slug": state.slug,
                "description": state.description,
                "change": f"Updated what {state.title} is for.",
            })

        elif kind == "add":
            state = _target("add", op.get("slug"))
            if state is None or not _bullet_ops_allowed(state, "add"):
                continue
            if _self_person(state.slug, state.title):
                complaints.append(
                    f"add: {state.slug!r} is the person whose memory this is "
                    "— put it in you/profile"
                )
                continue
            bullet = _check_bullet("add", op.get("bullet"), state)
            if bullet is None:
                continue
            change = _clean_change("add", op.get("change"), complaints)
            if change is None:
                continue
            if bullet in state.bullets:
                complaints.append(f"add on {state.slug}: that bullet is already there")
                continue
            state.bullets.append(bullet)
            state.dirty = True
            if not _fits(state, "add"):
                state.bullets.pop()
                continue
            plan.accepted.append({
                "op": "add", "slug": state.slug, "bullet": bullet, "change": change,
            })

        elif kind == "rewrite":
            state = _target("rewrite", op.get("slug"))
            if state is None or not _bullet_ops_allowed(state, "rewrite"):
                continue
            match = (op.get("match") or "").strip() if isinstance(op.get("match"), str) else ""
            if match not in state.bullets:
                complaints.append(
                    f"rewrite on {state.slug}: no bullet reads exactly "
                    f"{match!r} — `match` must be the existing text"
                )
                continue
            bullet = _check_bullet("rewrite", op.get("bullet"), state)
            if bullet is None:
                continue
            change = _clean_change("rewrite", op.get("change"), complaints)
            if change is None:
                continue
            at = state.bullets.index(match)
            state.bullets[at] = bullet
            state.dirty = True
            if not _fits(state, "rewrite"):
                state.bullets[at] = match
                continue
            plan.accepted.append({
                "op": "rewrite", "slug": state.slug, "match": match,
                "bullet": bullet, "change": change,
            })

        elif kind == "remove":
            state = _target("remove", op.get("slug"))
            if state is None or not _bullet_ops_allowed(state, "remove"):
                continue
            match = (op.get("match") or "").strip() if isinstance(op.get("match"), str) else ""
            if match not in state.bullets:
                complaints.append(
                    f"remove on {state.slug}: no bullet reads exactly {match!r}"
                )
                continue
            change = _clean_change("remove", op.get("change"), complaints)
            if change is None:
                continue
            state.bullets.remove(match)
            state.dirty = True
            plan.accepted.append({
                "op": "remove", "slug": state.slug, "match": match, "change": change,
            })

        elif kind == "link":
            state = _target("link", op.get("slug"))
            if state is None:
                continue
            raw_links = op.get("links")
            if not isinstance(raw_links, list):
                complaints.append(f"link on {state.slug}: links must be a list of slugs")
                continue
            wanted = [s.strip() for s in raw_links if isinstance(s, str) and s.strip()]
            bad = [s for s in wanted if not is_valid_slug(s)]
            if bad:
                complaints.append(f"link on {state.slug}: invalid slugs {bad}")
                continue
            for target in wanted:
                linked_slugs.append((f"link on {state.slug}", target))
            state.links = wanted[:MAX_LINKS_PER_FILE]
            plan.accepted.append({
                "op": "link", "slug": state.slug, "links": state.links,
            })

        elif kind == "delete_file":
            state = _target("delete_file", op.get("slug"))
            if state is None:
                continue
            change = _clean_change("delete_file", op.get("change"), complaints)
            if change is None:
                continue
            if state.slug in SYSTEM_FILES:
                # A system file is emptied, never dropped — the same rule the
                # DELETE route follows, stated here so the writer cannot
                # delete Profile out from under the injection.
                state.bullets = []
                state.dirty = True
                plan.accepted.append({
                    "op": "delete_file", "slug": state.slug, "change": change,
                    "system": True,
                })
            else:
                state.deleted = True
                state.dirty = True
                plan.accepted.append({
                    "op": "delete_file", "slug": state.slug, "change": change,
                })

        else:
            complaints.append(f"unknown op {kind!r}")

    # Links are resolved LAST: a link may legitimately point at a file the
    # same batch created. A dangling one is dropped rather than failing the
    # batch — the writer is the last line of defence for `[[slug]]`, and a
    # broken link on a page is worse than a missing one.
    live = {s for s, st in states.items() if not st.deleted}
    dangling = sorted({t for _, t in linked_slugs if t not in live})
    if dangling:
        complaints.append(
            "links point at files that do not exist and were dropped: "
            + ", ".join(dangling)
        )
        for state in states.values():
            if state.links:
                kept = [s for s in state.links if s in live]
                if kept != state.links:
                    state.links = kept
        for accepted in plan.accepted:
            if accepted["op"] == "link":
                accepted["links"] = [s for s in accepted["links"] if s in live]

    return plan


# ── Apply ─────────────────────────────────────────────────────────────

async def resolve_user_tz(db: AsyncSession, user_id: str) -> Optional[str]:
    """The user's IANA zone, VALIDATED, or None.

    One resolver for both day-bucketing (`_resolve_day_key`) and the writer's
    "today is …" line. They have to agree: a change line filed under the
    user's local Tuesday while the prompt told the model it was Monday makes
    every relative date the writer resolves off by a day.
    """
    try:
        from zoneinfo import ZoneInfo

        from app.db.models.user import User

        tz_name = (await db.execute(
            select(User.timezone).where(User.id == user_id)
        )).scalar_one_or_none()
        if tz_name:
            ZoneInfo(tz_name)  # an unparseable zone is not trusted
            return tz_name
    except Exception as exc:
        logger.warning(
            "[memory_files] timezone falling back to UTC for user=%s: %s",
            str(user_id)[:8], exc,
        )
    return None


async def _resolve_day_key(db: AsyncSession, user_id: str) -> str:
    """The user's LOCAL YYYY-MM-DD, UTC when their zone is unknown.

    Same resolution chain every other day-bucketing caller uses:
    `User.timezone`, VALIDATED (an unparseable zone is not trusted), then
    UTC. A calendar that files a 23:40-local write under tomorrow is a
    calendar the user cannot reconcile with their own day.
    """
    now = datetime.now(timezone.utc)
    tz_name = await resolve_user_tz(db, user_id)
    if tz_name:
        from zoneinfo import ZoneInfo

        return now.astimezone(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d")


async def _write_change(
    db: AsyncSession,
    user_id: str,
    slug: str,
    title: str,
    kind: str,
    summary: str,
    day_key: Optional[str] = None,
) -> None:
    db.add(MemoryFileChange(
        user_id=user_id,
        file_slug=slug,
        file_title=title,
        kind=kind,
        summary=summary[:MAX_CHANGE_CHARS],
        day_key=day_key or await _resolve_day_key(db, user_id),
        created_at=datetime.utcnow(),
    ))


async def write_change_line(
    db: AsyncSession, user_id: str, slug: str, title: str, kind: str, summary: str,
) -> None:
    """One audit line from a writer that is not the ops engine.

    The v3 migration is the only such writer: it applies its ops with
    `change_lines=False` and then files its own single line per filled
    file. Same table, same user-local day resolution, so the Memory log
    cannot end up with two notions of what day it is.
    """
    await _write_change(db, user_id, slug, title, kind, summary)


async def apply_ops(
    db: AsyncSession, user_id: str, plan: OpsPlan, *, change_lines: bool = True,
) -> Dict[str, Any]:
    """Persist exactly the state the validation walk computed.

    `change_lines=False` is the MIGRATION's mode and nothing else's (WS-5).
    A migration replays a whole legacy corpus through the writer in batches,
    so the per-op trail this function normally writes would open the user's
    Memory log on the day they upgraded with dozens of lines describing ops
    they never asked for. `memory_v3_migration` writes exactly one line per
    file it filled instead — "Migrated from your earlier memory." — which is
    the one sentence that is true about all of them. Default True, so every
    conversational writer is byte-identical to before.
    """
    if not plan.accepted:
        return {"applied": 0, "changed_files": []}

    touched = {op["slug"] for op in plan.accepted}
    rows = {
        r.slug: r for r in (await db.execute(
            select(MemoryFile).where(
                and_(MemoryFile.user_id == user_id, MemoryFile.slug.in_(list(touched)))
            )
        )).scalars().all()
    }
    day_key = await _resolve_day_key(db, user_id)
    now = datetime.utcnow()

    for slug in sorted(touched):
        state = plan.states.get(slug)
        if state is None:
            continue
        row = rows.get(slug)
        if state.deleted:
            if row is not None:
                await db.delete(row)
            continue
        if row is None:
            section = section_of_slug(slug)
            row = MemoryFile(
                user_id=user_id,
                slug=slug,
                section=state.section or (section.value if section else FileSection.TOPICS.value),
                title=state.title,
                is_system=slug in SYSTEM_FILES,
                position=100,
            )
            db.add(row)
            rows[slug] = row
        row.title = state.title
        row.description = state.description
        row.body_md = state.body()
        row.links_json = json.dumps(state.links) if state.links else None
        row.updated_at = now

    if change_lines:
        for op in plan.accepted:
            slug = op["slug"]
            state = plan.states.get(slug)
            title = state.title if state else title_from_slug(slug)
            kind = _CHANGE_KIND.get(op["op"])
            if kind is None or op["op"] == "link":
                continue
            await _write_change(db, user_id, slug, title, kind, op["change"], day_key)

    await db.commit()
    return {
        "applied": len(plan.accepted),
        "changed_files": sorted(touched),
        # TITLES, in the same order, for anything a person reads. A slug is
        # an internal id and no client renders one — the confirmation under
        # the input box saying "Saved 1 change in people/majid-tajik" is the
        # only place one ever leaked to a user. `plan.states` is already in
        # hand here, so the lookup costs nothing; `title_from_slug` covers a
        # slug whose state was dropped.
        "changed_titles": [
            (plan.states[s].title if s in plan.states else title_from_slug(s))
            for s in sorted(touched)
        ],
    }



# ── The scheduled slot (agent-side) ───────────────────────────────────

async def run_memory_maintenance() -> Dict[str, Any]:
    """The agent scheduler's nightly + boot memory job.

    Cheap and idempotent by design. Round 8's occupant of this slot was
    `memory_file_migration.run_memory_file_maintenance` (DELETED by WS-5,
    which was its last reader): organize every row → repair working leases
    → run an LLM curation pass over every file that needed one. All three
    operate on `memories` rows, which v3 retires. Today the slot does two
    things: make sure the three system files exist, and run the round-8 →
    v3 migration once.

    The REGISTRATION SHAPE is kept deliberately (boot one-shot at T+180s
    plus a nightly cron, both marker-cheap): WS-5's v3 migration hooks in
    HERE rather than adding a second pair of registrations in `agent_main`,
    so there is one slot to keep on a CronTrigger instead of two.

    Single-tenant: an agent container serves exactly one user
    (`settings.user_id`). On the platform this returns immediately.
    """
    from app.config import settings

    user_id = getattr(settings, "user_id", "") or ""
    if not user_id:
        return {"skipped": "no tenant user"}

    from app.db.database import async_session_maker

    async with async_session_maker() as db:
        created = await ensure_system_files(db, user_id)
        await db.commit()
    logger.info(
        "[memory_files] maintenance: %d system file(s) created for user=%s",
        len(created), str(user_id)[:8],
    )

    # The round-8 → v3 migration (§7). Marker-guarded: every fire after the
    # first one is a single indexed SELECT. It runs AFTER the system files
    # exist because the writer routes the owner's facts into `you/profile`
    # and cannot add to a file that is not there. A failure here must not
    # take the maintenance slot down with it — the migration owns its own
    # retry policy through the marker.
    migration: Dict[str, Any] = {"skipped": "not attempted"}
    try:
        from app.services.memory_v3_migration import run_scheduled_migration

        migration = await run_scheduled_migration()
    except Exception as exc:  # pragma: no cover - defence for the scheduler
        logger.error("[memory_files] v3 migration slot raised: %s", exc)
        migration = {"status": "failed", "error": str(exc)[:400]}

    # Current context's day rollover (v3 §6). It has its own HOURLY cron —
    # this is the boot pass, so a container that came up after midnight ages
    # the file at T+180s instead of waiting for the next hour, and a fleet
    # recreated every 0.3 h never leaves a stale `Today` in the prompt. It is
    # idempotent per user-local day, so calling it from two slots is free.
    rollover: Dict[str, Any] = {"skipped": "not_run"}
    try:
        from app.services.current_context import run_context_rollover

        rollover = await run_context_rollover()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[memory_files] context rollover skipped: %s", exc)
    return {
        "system_files_created": len(created),
        "memory_v3_migration": migration,
        "context_rollover": rollover,
    }
