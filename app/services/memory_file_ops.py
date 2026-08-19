"""LLM-mediated curation of ONE memory file, under a strict ops contract.

Two callers, one engine (docs/memory/rebuild-2026-08.md §3.6–3.7):

- the natural-language edit box on a file page ("Tell your agent what to
  change or remove") → `curate_file(..., instruction=...)`;
- the one-time migration's per-file consolidation pass → `curate_file(...)`
  with no instruction.

The model proposes operations; nothing it says is applied until it survives
validation, and every applied op rides the EXISTING evolution primitives
(merge_memory / supersede_memory / soft delete / smart create), so history,
audit events, embeddings and the recall surfaces stay honest. Sources of a
merge are soft-archived with `superseded_by` pointers — nothing this module
does hard-deletes a row, which is what makes the whole pass reversible.

Validation is the contract:
- every referenced id must be an ACTIVE entry of THIS file;
- an id may be touched by at most one op;
- text must be non-empty, ≤ MAX_ENTRY_CHARS, and in the user's second
  person — any "the user" phrasing rejects that op;
- `delete` is capped (noise removal, not clearance — the file-delete button
  owns wholesale removal), and consolidation may delete less than instruct;
- an invalid op skips THAT op, never the batch; a malformed response
  retries once with the validator's complaints appended.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Memory, MemoryFile
from app.db.models.enums import MemoryEventType
from app.memory_files import SECTION_DEFAULT_CATEGORY, SYSTEM_FILES, FileSection, is_valid_slug
from app.memory_taxonomy import normalize_category
from app.services.memory_file_service import MemoryFileService
from app.services.memory_log import describe_memory
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

MAX_OPS = 40
MAX_ENTRY_CHARS = 600
# Voice guard: storage speaks to the user. Any third-person "the user"
# phrasing is the exact defect this rebuild removes — never let an op
# reintroduce it.
_THIRD_PERSON_RE = re.compile(r"\bthe\s+user\b", re.IGNORECASE)


def _entry_text(m: Memory) -> str:
    return (m.canonical_content or m.content or "").strip()


def validate_ops(
    raw_ops: Any,
    entries: List[Memory],
    *,
    instruction_mode: bool,
) -> Tuple[List[Dict], List[str]]:
    """Split a model response into applicable ops and per-op rejections."""
    problems: List[str] = []
    if not isinstance(raw_ops, list):
        return [], ["ops must be a list"]
    if len(raw_ops) > MAX_OPS:
        return [], [f"too many ops ({len(raw_ops)} > {MAX_OPS})"]

    by_handle = {f"e{i + 1}": m for i, m in enumerate(entries)}
    touched: set = set()
    valid: List[Dict] = []
    delete_count = 0

    def _claim(handles: List[str], op_name: str) -> bool:
        for h in handles:
            if h not in by_handle:
                problems.append(f"{op_name}: unknown entry id {h!r}")
                return False
            if h in touched:
                problems.append(f"{op_name}: entry {h} already touched by an earlier op")
                return False
        touched.update(handles)
        return True

    def _clean_text(op_name: str, text: Any) -> Optional[str]:
        text = (text or "").strip() if isinstance(text, str) else ""
        if not text:
            problems.append(f"{op_name}: empty text")
            return None
        if len(text) > MAX_ENTRY_CHARS:
            problems.append(f"{op_name}: text over {MAX_ENTRY_CHARS} chars")
            return None
        if _THIRD_PERSON_RE.search(text):
            problems.append(f"{op_name}: third-person voice ('the user') is not allowed")
            return None
        return text

    for op in raw_ops:
        if not isinstance(op, dict):
            problems.append("op is not an object")
            continue
        kind = op.get("op")
        if kind == "merge":
            ids = op.get("ids") or []
            if not isinstance(ids, list) or len(ids) < 2:
                problems.append("merge: needs at least two ids")
                continue
            text = _clean_text("merge", op.get("text"))
            if text is None or not _claim([str(i) for i in ids], "merge"):
                continue
            valid.append({"op": "merge", "ids": [str(i) for i in ids], "text": text})
        elif kind == "update":
            handle = str(op.get("id") or "")
            text = _clean_text("update", op.get("text"))
            if text is None or not _claim([handle], "update"):
                continue
            valid.append({"op": "update", "id": handle, "text": text})
        elif kind == "delete":
            handle = str(op.get("id") or "")
            if not _claim([handle], "delete"):
                continue
            delete_count += 1
            valid.append({"op": "delete", "id": handle, "reason": str(op.get("reason") or "")[:200]})
        elif kind == "add":
            text = _clean_text("add", op.get("text"))
            if text is None:
                continue
            valid.append({"op": "add", "text": text, "category": op.get("category")})
        elif kind == "set_purpose":
            text = _clean_text("set_purpose", op.get("text"))
            if text is None:
                continue
            valid.append({"op": "set_purpose", "text": text[:300]})
        elif kind == "set_related":
            slugs = [s for s in (op.get("slugs") or []) if isinstance(s, str) and is_valid_slug(s)]
            valid.append({"op": "set_related", "slugs": slugs[:8]})
        else:
            problems.append(f"unknown op {kind!r}")

    # Deletion cap. Consolidation exists to merge, not to clear; the user's
    # own instruction gets more room, and wholesale removal has its own
    # button (delete file).
    if entries:
        ceiling = max(3, len(entries) // 2) if instruction_mode else max(1, len(entries) // 5)
        if delete_count > ceiling:
            problems.append(
                f"delete cap exceeded ({delete_count} > {ceiling}) — every delete op dropped; "
                "use the file's delete action for wholesale removal"
            )
            valid = [v for v in valid if v["op"] != "delete"]

    return valid, problems


def render_entries_for_prompt(entries: List[Memory]) -> str:
    lines = []
    for i, m in enumerate(entries):
        bits = []
        if m.created_at:
            bits.append(f"saved {m.created_at.date().isoformat()}")
        if m.expires_at:
            bits.append(f"fades {m.expires_at.date().isoformat()}")
        try:
            if m.tags_json and "standing" in json.loads(m.tags_json):
                bits.append("standing arrangement")
        except Exception:
            pass
        meta = f" ({', '.join(bits)})" if bits else ""
        lines.append(f"[e{i + 1}]{meta} {_entry_text(m)}")
    return "\n".join(lines)


def build_curation_prompt(
    title: str,
    purpose: Optional[str],
    entries_block: str,
    *,
    instruction: Optional[str],
) -> str:
    head = (
        f'You are curating one file of a user\'s memory: "{title}".\n'
        f"Current purpose line: {purpose or '(none)'}\n\n"
        "Entries (each has an id):\n"
        f"{entries_block or '(no entries)'}\n\n"
    )
    contract = """Reply with ONLY valid JSON: {"ops": [...]}. Allowed ops:
- {"op": "merge", "ids": ["e1","e4"], "text": "..."} — fold entries that describe the SAME fact, task or arrangement (variants, restatements, partial updates) into one concise entry carrying every distinct detail. The first id survives.
- {"op": "update", "id": "e2", "text": "..."} — rewrite one entry in place: fix voice, fold in a correction, trim bloat.
- {"op": "delete", "id": "e3", "reason": "..."} — remove true noise only: one-off tool or playback outcomes, momentary states, meaningless fragments.
- {"op": "add", "text": "...", "category": "..."} — a genuinely new entry (rare).
- {"op": "set_purpose", "text": "..."} — the file's one-line header: what it holds and when to read it, e.g. "Your IELTS prep — tutor, dates and targets; read when IELTS comes up."
- {"op": "set_related", "slugs": ["people/majid-tajik"]} — sibling files this one should point at.

Hard rules:
- Every entry text is written in the SECOND PERSON, to the user: "You…", "Your…". Other people by name. NEVER "The user".
- One or two short sentences per entry. Preserve every name, number, date and time EXACTLY. Never invent or embellish facts.
- Prefer merge over delete; when unsure, keep.
- If the file is already clean, reply {"ops": []}.
"""
    if instruction:
        task = (
            "The user asked for this change to the file — apply it with the fewest ops "
            f"that fully honor it, touching nothing else:\n\"{instruction}\"\n\n"
        )
    else:
        task = (
            "Consolidate this file: merge duplicates and restatements, normalize voice, "
            "remove noise, and give it a purpose line if the current one is missing or wrong. "
            "Produce the MINIMAL op set.\n\n"
        )
    return head + task + contract


async def _propose_ops(
    llm_service,
    prompt: str,
) -> List[Dict]:
    response = await llm_service.complete_with_json(
        messages=[{"role": "user", "content": prompt}],
        model=settings.memory_extraction_model,
        temperature=0.0,
    )
    parsed = response
    if hasattr(response, "content"):
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("curation response is not an object")
    return parsed.get("ops", [])


async def apply_ops(
    db: AsyncSession,
    user_id: str,
    slug: str,
    ops: List[Dict],
    entries: List[Memory],
    *,
    api_key: Optional[str] = None,
    trigger_source: str = "file_curation",
) -> Dict:
    """Apply validated ops through the existing evolution primitives."""
    memory_service = MemoryService(db, api_key=api_key)
    file_service = MemoryFileService(db)
    by_handle = {f"e{i + 1}": m for i, m in enumerate(entries)}
    applied: List[Dict] = []

    file_row = (await db.execute(
        select(MemoryFile).where(and_(MemoryFile.user_id == user_id, MemoryFile.slug == slug))
    )).scalar_one_or_none()

    section = FileSection(file_row.section) if file_row else FileSection.KNOWLEDGE

    for op in ops:
        kind = op["op"]
        try:
            if kind == "merge":
                survivor = by_handle[op["ids"][0]]
                await memory_service.merge_memory(
                    memory_id=survivor.id,
                    new_content=op["text"],
                    change_summary=f"Folded {len(op['ids']) - 1} duplicate entr"
                                   f"{'y' if len(op['ids']) == 2 else 'ies'} into this one",
                    source_type=trigger_source,
                    user_id=user_id,
                )
                for handle in op["ids"][1:]:
                    source = by_handle[handle]
                    await memory_service.supersede_memory(
                        old_memory_id=source.id, new_memory_id=survivor.id, user_id=user_id,
                    )
                    await memory_service._log_memory_event(
                        memory_id=source.id,
                        user_id=user_id,
                        event_type=MemoryEventType.CONSOLIDATED,
                        event_data={"into": survivor.id, "via": trigger_source},
                        trigger_source=trigger_source,
                    )
                applied.append({"op": "merge", "survivor": survivor.id, "folded": len(op["ids"]) - 1})
            elif kind == "update":
                target = by_handle[op["id"]]
                if _entry_text(target) != op["text"]:
                    await memory_service.merge_memory(
                        memory_id=target.id,
                        new_content=op["text"],
                        change_summary="Rewritten during file curation",
                        source_type=trigger_source,
                        user_id=user_id,
                    )
                    applied.append({"op": "update", "id": target.id})
            elif kind == "delete":
                target = by_handle[op["id"]]
                await memory_service.delete_memory(target.id, user_id)
                applied.append({"op": "delete", "id": target.id, "reason": op.get("reason", "")})
            elif kind == "add":
                from app.schemas import BrainType, MemoryCreate
                category = normalize_category(
                    (op.get("category") or SECTION_DEFAULT_CATEGORY[section]),
                    brain_type="agent" if section == FileSection.LEARNED else "user",
                )
                created = await memory_service.create_memory(
                    user_id=user_id,
                    memory_data=MemoryCreate(
                        content=op["text"],
                        brain_type=BrainType.AGENT if section == FileSection.LEARNED else BrainType.USER,
                        category=category,
                        memory_type="task" if section == FileSection.WORKING else "fact",
                        importance=0.6,
                        source_type=trigger_source,
                    ),
                    deduplicate=False,
                )
                await file_service.assign_to_file(created, slug)
                await db.commit()
                applied.append({"op": "add", "id": created.id})
            elif kind == "set_purpose" and file_row is not None:
                file_row.purpose = op["text"]
                applied.append({"op": "set_purpose"})
            elif kind == "set_related" and file_row is not None:
                file_row.related_json = json.dumps(op["slugs"])
                applied.append({"op": "set_related", "slugs": op["slugs"]})
        except Exception as e:
            logger.warning("[file_ops] %s failed on %s: %s", kind, slug, e)

    if file_row is not None:
        file_row.updated_at = datetime.utcnow()
    await db.commit()
    return {"applied": applied}


async def curate_file(
    db: AsyncSession,
    user_id: str,
    slug: str,
    *,
    instruction: Optional[str] = None,
    api_key: Optional[str] = None,
    mark_consolidated: bool = False,
) -> Dict:
    """Propose + validate + apply curation ops for one file.

    Returns {"applied": [...], "rejected": [...], "entry_count": n}.
    Raises ValueError for an unknown file. LLM or parse failures surface as
    an exception for the caller to translate (the API returns 502; the
    migration retries on its next slot).
    """
    file_service = MemoryFileService(db)
    resolved = await file_service.get_file(user_id, slug)
    if resolved is None:
        raise ValueError(f"memory file {slug!r} not found")
    info, entries = resolved

    if api_key:
        from app.services.llm_service import LLMService
        llm_service = LLMService(api_key=api_key)
    else:
        from app.services.llm_service import get_llm_service
        llm_service = get_llm_service()

    prompt = build_curation_prompt(
        info["title"], info.get("purpose"),
        render_entries_for_prompt(entries),
        instruction=instruction,
    )
    raw_ops = await _propose_ops(llm_service, prompt)
    ops, problems = validate_ops(raw_ops, entries, instruction_mode=bool(instruction))
    if problems and not ops and raw_ops:
        # Whole proposal rejected — one retry with the complaints appended.
        retry_prompt = (
            prompt
            + "\n\nYour previous reply was rejected for these reasons — fix them:\n- "
            + "\n- ".join(problems)
        )
        raw_ops = await _propose_ops(llm_service, retry_prompt)
        ops, problems = validate_ops(raw_ops, entries, instruction_mode=bool(instruction))

    result = await apply_ops(
        db, user_id, slug, ops, entries,
        api_key=api_key,
        trigger_source="file_instruction" if instruction else "file_consolidation",
    )

    if mark_consolidated:
        file_row = (await db.execute(
            select(MemoryFile).where(and_(MemoryFile.user_id == user_id, MemoryFile.slug == slug))
        )).scalar_one_or_none()
        if file_row is not None:
            file_row.consolidated_at = datetime.utcnow()
            await db.commit()

    logger.info(
        "[file_ops] curated %s for user=%s: %d applied, %d rejected%s",
        slug, user_id, len(result["applied"]), len(problems),
        " (instruction)" if instruction else "",
    )
    result["rejected"] = problems
    result["entry_count"] = len(entries)
    return result
