"""Round 8 → v3: every tenant's old rows become curated files (§7).

The one-shot that runs once per tenant, agent-side, and turns a `memories`
corpus into `memory_files.body_md`. It is the last consumer of the row
table in the memory product; after it runs, nothing in the conversation
path reads or writes a row again.

Five properties, each of which is a test in
`tests/test_memory_v3_migration.py`:

* **The legacy table is never touched.** No supersede stamps, no deletes,
  no `file_slug` writes. That table IS the backup and the rollback: if v3
  is reverted to the previous image tag, round 8 comes back intact because
  nothing under it moved. The round-8 file assignment this migration needs
  is READ (`memories.file_slug`, or `legacy_default_slug_for` when a row
  predates the last organize pass) — never written.
* **What the USER deleted stays deleted.** This is the single worst
  possible outcome of the migration and it is guarded twice: `is_deleted`
  (every forget path in `memory_service` sets it) and, corroborating, a
  DELETED row in `memory_events`. See `_ELIGIBILITY` below for why
  `trigger_source` cannot be the discriminator the contract assumed.
* **Idempotent.** The `migration_status` marker is the gate; a second run
  reports "already_completed" and changes nothing. Pinned by running it
  twice and comparing every body byte for byte.
* **Resumable.** The report's per-row dispositions ARE the resume ledger:
  a row already accounted for is not fed again. A batch interrupted after
  its ops committed but before its ledger did is RE-processed rather than
  assumed done — losing a fact is worse than a duplicate, and the
  reprocessed ids are named in the report so a reviewer can look.
* **Dry-runnable, and dry means dry.** No marker row, no file write, no
  change line, no legacy write. The full report is still produced, because
  the dry run threads a SIMULATED file set from batch to batch — so
  "after" is the state a real run would reach, not batch 1's repeated.

The writer is `memory_curator.curate_migration_batch`. This module decides
what the writer is shown and what happens to what it says; it does not
have opinions about bullets.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory_files import (
    SYSTEM_FILES,
    legacy_default_slug_for,
    parse_bullets,
)
from app.services import memory_curator, memory_file_ops as ops
from app.services.user_identity import resolve_user_identity

logger = logging.getLogger(__name__)

MIGRATION_NAME = "memory_v3"
FORCE_ENV = "FORCE_MEMORY_V3_MIGRATION"

#: Same auto-retry policy as `backfill_day_chats`: a transient failure must
#: not strand a tenant forever, and a genuinely broken migration must not
#: loop. The grace is an hour rather than five minutes because the retry
#: slot here is the nightly cron, not every boot.
MAX_AUTO_RETRY_ATTEMPTS = 3
RETRY_GRACE_SEC = 3600

#: One model call per batch. Small, because merging only works between rows
#: the writer can see at once, and a batch is also the resume granularity.
BATCH_MAX_ROWS = 12
BATCH_MAX_CHARS = 6000

#: §3.4 — document and media rows keep their own pipeline and their own
#: embedding search. They were never the user's memory product, and feeding
#: a PDF chunk to the writer is how a file fills up with somebody's invoice.
DOCUMENT_SOURCE_TYPES = frozenset({"document", "media"})

#: A row minted ABOUT a scheduler object. §2.3: the text of a routine or
#: trigger is never stored; a standing arrangement is one line in Profile,
#: which the writer may still produce from other evidence. `ref_kind` is
#: hard evidence rather than a guess about the content.
SCHEDULER_REF_KINDS = frozenset({"routine", "trigger", "job", "reminder"})

MIGRATED_SUMMARY = "Migrated from your earlier memory."

#: Dispositions decided HERE, deterministically, before the writer is asked.
SKIP_DELETED = "skipped_user_deleted"
SKIP_SUPERSEDED = "skipped_superseded"
SKIP_DOCUMENT = "skipped_document_or_media"
SKIP_SECRET = "skipped_never_store"
SKIP_SCHEDULER = "skipped_scheduler_object"
SKIP_EMPTY = "skipped_empty"
#: Decided by the writer.
MODEL_VERDICTS = frozenset({"kept", "merged", "rewritten", "dropped"})
#: The writer was shown the row and said nothing about it. A defect, not a
#: decision — named so it reads as one in the report.
UNACCOUNTED = "unaccounted"
#: The batch carrying it failed. The row was neither written nor judged.
FAILED = "batch_failed"


# ── Eligibility ───────────────────────────────────────────────────────
#
# _ELIGIBILITY, and the one place this module departs from the contract.
#
# §7 (and the WS-5 spec) say to tell curation-deletes from user forgets via
# `memory_events.trigger_source == 'file_consolidation'`. That value does
# not exist on any DELETED event. Reading round 8's own code (commit
# cd24717b, `memory_file_ops.apply_ops`):
#
#   * a curation MERGE stamps the folded-away row with a CONSOLIDATED event
#     carrying `trigger_source='file_consolidation'` and sets
#     `superseded_by`;
#   * a curation DELETE calls `MemoryService.delete_memory`, which hardcodes
#     `trigger_source="api"` — the SAME value a user's own "Forget this"
#     produces, through the same function.
#
# So the two are indistinguishable at the point the contract wanted to
# distinguish them, and the failure is asymmetric: resurrecting something a
# person chose to erase is unrecoverable, while leaving behind something
# round-8 curation threw away costs a fact that curation had already judged
# junk (and that v3's durability rules would reject again). The rule is
# therefore: **`is_deleted` is never migrated, whoever set it.**
#
# What IS recovered is the large half the contract was actually after: rows
# round 8 deactivated WITHOUT deleting — expiry-archived and
# dedup-deactivated rows, `is_active=False, is_deleted=False`. Those are fed
# to the writer, flagged as archived so it knows what it is looking at.
# Superseded rows are skipped because their content lives on in the
# survivor, which IS fed; migrating both would double every merged fact.


@dataclass
class SourceRow:
    id: str
    content: str
    legacy_slug: str
    category: str
    brain_type: str
    source_type: str
    created_at: Optional[datetime]
    is_active: bool

    @property
    def preview(self) -> str:
        return (self.content or "")[:120]


@dataclass
class _VirtualFile:
    """A `MemoryFile`-shaped value with no session behind it.

    The dry run needs to hand batch 2 the files batch 1 WOULD have written.
    Everything the writer and the validator read off a file row is here and
    nothing else is, so a virtual file can never be persisted by accident.
    """

    slug: str
    section: str
    title: str
    description: Optional[str] = None
    body_md: str = ""
    links_json: Optional[str] = None
    is_system: bool = False


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _now() -> datetime:
    return datetime.utcnow()


async def _marker(db: AsyncSession):
    from app.db.models.day_chat import MigrationStatus

    return (await db.execute(
        select(MigrationStatus).where(
            MigrationStatus.migration_name == MIGRATION_NAME
        )
    )).scalar_one_or_none()


def _forced() -> bool:
    return os.environ.get(FORCE_ENV, "").lower() in ("true", "1", "yes")


# ── Reading the legacy corpus ─────────────────────────────────────────

async def _deleted_row_ids(db: AsyncSession, user_id: str) -> Tuple[Set[str], bool]:
    """Row ids carrying a DELETED audit event, and whether we could look.

    Corroboration for `is_deleted`, not a replacement: every forget path
    writes the event and the flag in one transaction. When the audit table
    cannot be read (it is AGENT_ONLY and a very old tenant may predate it)
    the caller records `events_readable: false` in the report rather than
    pretending the check ran.
    """
    try:
        from app.db.models.memory import MemoryEvent

        rows = (await db.execute(
            select(MemoryEvent.memory_id).where(
                MemoryEvent.user_id == user_id,
                MemoryEvent.event_type == "deleted",
            )
        )).scalars().all()
        return {str(r) for r in rows}, True
    except Exception as exc:
        logger.warning(
            "[memory_v3] memory_events unreadable for user=%s (%s) — falling "
            "back to is_deleted alone", str(user_id)[:8], exc,
        )
        return set(), False


async def read_corpus(
    db: AsyncSession, user_id: str
) -> Tuple[List[SourceRow], List[Dict[str, Any]], Dict[str, Any]]:
    """Every legacy row, split into what the writer sees and what it does not.

    Returns (eligible, predecided, stats). `predecided` rows already carry
    their final disposition and their reason; they still appear in the
    report, because "where did this memory go?" has to be answerable for
    every id, including the ones nothing was allowed to do anything with.
    """
    from app.db.models.memory import Memory
    from app.services.memory_secrets import sensitive_content_reason

    rows = (await db.execute(
        select(Memory)
        .where(Memory.user_id == user_id)
        .order_by(Memory.created_at, Memory.id)
    )).scalars().all()

    deleted_ids, events_readable = await _deleted_row_ids(db, user_id)

    eligible: List[SourceRow] = []
    predecided: List[Dict[str, Any]] = []
    stats = {
        "rows_total": len(rows),
        "rows_active": 0,
        "rows_archived": 0,
        "rows_deleted": 0,
        "by_category": {},
        "by_legacy_file": {},
        "events_readable": events_readable,
    }

    for row in rows:
        rid = str(row.id)
        content = (row.content or "").strip()
        legacy_slug = (
            row.file_slug
            or legacy_default_slug_for(row.category, getattr(row, "brain_type", "user"))
        )
        is_deleted = bool(getattr(row, "is_deleted", False))
        is_active = bool(getattr(row, "is_active", True))

        if is_deleted or rid in deleted_ids:
            stats["rows_deleted"] += 1
        elif is_active:
            stats["rows_active"] += 1
        else:
            stats["rows_archived"] += 1
        stats["by_category"][row.category] = stats["by_category"].get(row.category, 0) + 1
        stats["by_legacy_file"][legacy_slug] = (
            stats["by_legacy_file"].get(legacy_slug, 0) + 1
        )

        def _skip(disposition: str, reason: str) -> None:
            predecided.append({
                "id": rid,
                "preview": content[:120],
                "legacy_file": legacy_slug,
                "category": row.category,
                "disposition": disposition,
                "slug": None,
                "reason": reason,
            })

        if is_deleted or rid in deleted_ids:
            _skip(
                SKIP_DELETED,
                "the person deleted it — a delete is never undone by a "
                "migration, whatever set it",
            )
            continue
        if getattr(row, "superseded_by", None):
            _skip(
                SKIP_SUPERSEDED,
                f"folded into {str(row.superseded_by)[:8]} by the old system; "
                "the surviving entry carries its content",
            )
            continue
        if (row.source_type or "") in DOCUMENT_SOURCE_TYPES:
            _skip(
                SKIP_DOCUMENT,
                f"a {row.source_type} chunk — document and media recall keeps "
                "its own pipeline and was never the memory product",
            )
            continue
        if (getattr(row, "ref_kind", None) or "") in SCHEDULER_REF_KINDS:
            _skip(
                SKIP_SCHEDULER,
                f"minted about a {row.ref_kind} — a scheduler object is not a "
                "fact about a life",
            )
            continue
        if not content:
            _skip(SKIP_EMPTY, "the entry has no text")
            continue
        secret = sensitive_content_reason(content)
        if secret:
            _skip(
                SKIP_SECRET,
                f"{secret.removeprefix('sensitive_')} is never stored — it is "
                "not shown to the writer either",
            )
            continue

        eligible.append(SourceRow(
            id=rid,
            content=content,
            legacy_slug=legacy_slug,
            category=row.category or "",
            brain_type=getattr(row, "brain_type", "user") or "user",
            source_type=row.source_type or "",
            created_at=row.created_at,
            is_active=is_active,
        ))

    return eligible, predecided, stats


# ── Batching ──────────────────────────────────────────────────────────

def plan_batches(rows: Sequence[SourceRow]) -> List[List[SourceRow]]:
    """Rows grouped so duplicates are visible to one another.

    Sorted by legacy file first: "merge, do not accumulate" can only happen
    between entries the writer is shown together, and round 8's own file
    assignment is the best available signal for which entries are about the
    same thing. A single row longer than the char budget still gets its own
    batch rather than being truncated — a half-fact is worse than a big
    prompt.
    """
    ordered = sorted(
        rows,
        key=lambda r: (
            r.legacy_slug,
            r.created_at or datetime.min,
            r.id,
        ),
    )
    batches: List[List[SourceRow]] = []
    current: List[SourceRow] = []
    used = 0
    for row in ordered:
        size = len(row.content)
        if current and (len(current) >= BATCH_MAX_ROWS or used + size > BATCH_MAX_CHARS):
            batches.append(current)
            current, used = [], 0
        current.append(row)
        used += size
    if current:
        batches.append(current)
    return batches


def render_entries(batch: Sequence[SourceRow]) -> Tuple[str, Dict[str, SourceRow]]:
    """The batch as the writer sees it, plus the handle → row map.

    Handles rather than ids on purpose: a UUID in the prompt is a UUID the
    model can copy into a bullet, and `bullet_problem` rejects those — the
    op would be thrown away and the fact with it.
    """
    by_handle: Dict[str, SourceRow] = {}
    lines: List[str] = []
    for i, row in enumerate(batch, start=1):
        handle = f"L{i}"
        by_handle[handle] = row
        meta = [f"old file: {row.legacy_slug}"]
        if row.category:
            meta.append(f"category: {row.category}")
        if row.created_at:
            meta.append(f"recorded {row.created_at.strftime('%b %d, %Y')}")
        if not row.is_active:
            meta.append("archived by the old system")
        lines.append(f"{handle}) [{' · '.join(meta)}] {row.content}")
    return "\n".join(lines), by_handle


# ── Snapshots ─────────────────────────────────────────────────────────

async def _file_snapshot(db: AsyncSession, user_id: str) -> List[Dict[str, Any]]:
    rows = await ops._all_files(db, user_id)
    return [
        {
            "slug": r.slug,
            "title": r.title,
            "section": r.section,
            "description": r.description,
            "bullets": len(parse_bullets(r.body_md)),
            "chars": len(r.body_md or ""),
            "body_md": r.body_md or "",
        }
        for r in sorted(rows, key=lambda r: r.slug)
    ]


def _virtual_snapshot(files: Sequence[_VirtualFile]) -> List[Dict[str, Any]]:
    return [
        {
            "slug": f.slug,
            "title": f.title,
            "section": f.section,
            "description": f.description,
            "bullets": len(parse_bullets(f.body_md)),
            "chars": len(f.body_md or ""),
            "body_md": f.body_md or "",
        }
        for f in sorted(files, key=lambda f: f.slug)
    ]


async def snapshot(db: AsyncSession, user_id: str) -> Dict[str, Any]:
    """The tenant's state right now — the report route's "before" dump.

    The fleet driver calls this BEFORE a real run and keeps it: the nightly
    `pg_dump` is the fleet snapshot, but a per-tenant record in the run's
    own artifact is what makes a single tenant's rollback checkable without
    restoring a database.
    """
    _, predecided, stats = await read_corpus(db, user_id)
    return {
        "legacy": stats,
        "not_eligible": len(predecided),
        "files": await _file_snapshot(db, user_id),
    }


# ── The migration ─────────────────────────────────────────────────────

@dataclass
class _Run:
    dispositions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_filled: List[str] = field(default_factory=list)
    #: Every slug an accepted `create_file` minted, filled or not. The
    #: difference against `files_filled` is the empty-shell set that gets
    #: pruned at the end of the run.
    created_ops: List[str] = field(default_factory=list)
    batches_done: int = 0
    batches_failed: int = 0


def _tallies(dispositions: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dispositions:
        out[d["disposition"]] = out.get(d["disposition"], 0) + 1
    return out


def _virtuals_from(files: Sequence[Any]) -> List[_VirtualFile]:
    return [
        _VirtualFile(
            slug=f.slug,
            section=f.section,
            title=f.title,
            description=f.description,
            body_md=f.body_md or "",
            links_json=getattr(f, "links_json", None),
            is_system=bool(getattr(f, "is_system", False)),
        )
        for f in files
    ]


def _apply_to_virtuals(
    previous: Sequence[_VirtualFile], plan: ops.OpsPlan
) -> List[_VirtualFile]:
    """The file set a dry run would leave behind, for the next batch.

    `plan.states` holds every file the batch was validated against, so an
    untouched file round-trips byte-exact and a touched one carries exactly
    the body `apply_ops` would have persisted.
    """
    out = {f.slug: f for f in previous}
    for slug, state in plan.states.items():
        if state.deleted:
            out.pop(slug, None)
            continue
        out[slug] = _VirtualFile(
            slug=state.slug,
            section=state.section,
            title=state.title,
            description=state.description,
            body_md=state.body(),
            links_json=json.dumps(state.links) if state.links else None,
            is_system=state.is_system,
        )
    return sorted(out.values(), key=lambda f: f.slug)


def _claims(
    raw: Sequence[Any], by_handle: Dict[str, SourceRow]
) -> Dict[str, Dict[str, Any]]:
    """The writer's `dispositions` array, keyed by handle and normalized.

    A handle that was not in this batch is ignored; a verdict that is not
    one of the four is read as `dropped`, because an unrecognized verdict
    is not evidence that anything was kept.
    """
    claimed: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        if ref not in by_handle:
            continue
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in MODEL_VERDICTS:
            verdict = "dropped"
        claimed[ref] = {
            "verdict": verdict,
            "slug": item.get("slug") or None,
            "reason": (str(item.get("reason") or "").strip() or None),
        }
    return claimed


def _did_land(
    handle: str,
    said: Dict[str, Any],
    bullet_slugs: Set[str],
    live_slugs: Set[str],
    landed_refs: Optional[Set[str]],
) -> bool:
    """Did a SURVIVING op actually carry this row's content into a file?

    One rule, two readers — the report's reconciliation and the orphan
    detector that decides who gets re-asked. Two rules would let a row be
    reported as dropped and never re-asked, or re-asked and reported as
    kept, and both are the kind of disagreement nobody notices until a
    person's name is missing from their own memory.
    """
    slug = said["slug"]
    if said["verdict"] == "merged":
        return slug in live_slugs
    if landed_refs is not None:
        return handle in landed_refs
    return slug in bullet_slugs


def orphaned_rows(
    by_handle: Dict[str, SourceRow],
    claimed: Dict[str, Dict[str, Any]],
    bullet_slugs: Set[str],
    live_slugs: Set[str],
    landed_refs: Optional[Set[str]],
) -> List[SourceRow]:
    """Rows the writer meant to KEEP whose op the validator refused.

    The defect this exists for: `validate_ops` refuses ops, and
    `memory_curator._run_ops` only re-asks when the WHOLE proposal was
    refused (`plan.complaints and not plan.accepted`). A batch with four
    good bullets and one refused one therefore never re-asks, and the
    refused row is gone — a rejection that was a decision about an OP
    becomes, silently, a decision about a ROW that nobody made. In a turn
    that costs a fact you will say again tomorrow. In the migration it is
    the one and only pass over a person's whole history, and it is how
    "Nariman Hosseini is the user's name" ended up nowhere.

    NOT included: a row the writer explicitly `dropped` (that IS a
    decision) and a row it never mentioned (`unaccounted` — a different
    defect, and re-asking a writer that ignored a row once tends to produce
    a writer that ignores it twice at the cost of a model call).
    """
    out: List[SourceRow] = []
    for handle, row in by_handle.items():
        said = claimed.get(handle)
        if not said or said["verdict"] == "dropped":
            continue
        if not _did_land(handle, said, bullet_slugs, live_slugs, landed_refs):
            out.append(row)
    return out


def build_retry_block(complaints: Sequence[str]) -> str:
    """The second round's instruction, in the validator's own words.

    The complaints are precise and actionable by construction ("their facts
    belong in you/profile, never in people/"; "tool parameters are never
    stored in a memory file") — which is the same property that justified
    the whole-proposal retry this one generalizes.

    It says explicitly that the RULE is not negotiable, only the shape.
    The retry exists so the FACT survives in an allowed form; it must never
    read as an invitation to get the refused form past the validator on a
    second attempt.
    """
    reasons = "\n".join(f"- {c}" for c in complaints[:12]) or "- (no reason given)"
    return (
        "THESE ENTRIES WERE NOT WRITTEN. Everything else from this batch is "
        "already in the file bodies above — do not add any of it again.\n\n"
        "Your previous ops for the entries below were REFUSED:\n"
        f"{reasons}\n\n"
        "Those rules are not negotiable and this is not an appeal. Each entry "
        "may still hold a durable fact; write it in a shape the rules allow — "
        "the fact WITHOUT the id, the standing arrangement WITHOUT the tool "
        "parameter or the job's text, a fact about the OWNER in you/profile "
        "instead of a people/ file. If there is no durable fact left once the "
        "refused part is removed, propose no op for it and say so in its "
        "disposition."
    )


def _record_dispositions(
    run: _Run,
    by_handle: Dict[str, SourceRow],
    raw: Sequence[Any],
    complaints: Sequence[str],
    bullet_slugs: Set[str],
    live_slugs: Set[str],
    landed_refs: Optional[Set[str]],
) -> None:
    """Reconcile what the writer SAID with what the validator ALLOWED.

    Three ways a row can end up without an honest answer, all of them
    handled here rather than left blank:

    * the writer never mentioned it → `unaccounted`, which reads as the
      defect it is;
    * the writer claims a row was kept and no SURVIVING op carries its
      handle → the claim is downgraded to `dropped` and the validator's
      complaints become the reason. `landed_refs` is per-ROW evidence
      (`refs` on each add/rewrite, matched back after validation), which
      slug-level evidence cannot be: one file receives bullets from several
      entries, and a batch where `topics/devices` was created and its only
      bullet refused for carrying a UUID, or where the Gmail job prompt's
      bullet was refused for `max_results=1` while four other bullets
      landed in the same file, would both read as "kept". When the writer
      omits `refs` entirely the check falls back to slug level and the run
      records that its precision is lower;
    * the writer names a handle that was not in this batch → ignored.

    A `merged` verdict is judged against `live_slugs`: merging INTO a
    bullet that already existed legitimately writes no new one.
    """
    claimed = _claims(raw, by_handle)

    def _why(slug: Optional[str]) -> str:
        """The complaints that are ABOUT this file, not the whole batch.

        A batch's complaint list mixes unrelated refusals, and pasting all
        of them under one row's disposition is how a reviewer reads "a UUID
        is never stored" as the reason a person's name was dropped.
        """
        relevant = [c for c in complaints if slug and slug in c] or list(complaints)
        return "; ".join(relevant)[:280]

    for handle, row in by_handle.items():
        entry = {
            "id": row.id,
            "preview": row.preview,
            "legacy_file": row.legacy_slug,
            "category": row.category,
            "disposition": UNACCOUNTED,
            "slug": None,
            "reason": (
                "the writer was shown this entry and did not account for it"
            ),
        }
        said = claimed.get(handle)
        if said:
            verdict, slug = said["verdict"], said["slug"]
            reason = said["reason"] or "no reason given"
            landed = _did_land(handle, said, bullet_slugs, live_slugs, landed_refs)
            if verdict != "dropped" and not landed:
                note = _why(slug)
                entry["disposition"] = "dropped"
                entry["slug"] = None
                entry["reason"] = (
                    (
                        f"the writer said it was {verdict} but named no file, "
                        "so nothing can be shown to have kept it"
                        if not slug else
                        f"the writer meant to keep it in {slug} but the op did "
                        "not survive validation"
                    )
                    + (f": {note}" if note else "")
                )
            else:
                entry["disposition"] = verdict
                entry["slug"] = slug if verdict != "dropped" else None
                entry["reason"] = reason
        run.dispositions[row.id] = entry


def _landed_refs(
    raw_ops: Sequence[Any], plan: ops.OpsPlan
) -> Optional[Set[str]]:
    """Handles named by ops that SURVIVED validation, or None.

    `validate_ops` keeps only the keys it knows, so `refs` has to be matched
    back from the raw proposal on (op, slug, bullet) — the same three values
    the validator copies through unchanged. None means the writer supplied
    no refs at all and the caller must fall back to file-level evidence.
    """
    by_key: Dict[Tuple[str, str, str], Set[str]] = {}
    saw_any = False
    for op in raw_ops:
        if not isinstance(op, dict) or op.get("op") not in ("add", "rewrite"):
            continue
        refs = op.get("refs")
        if not isinstance(refs, list):
            continue
        saw_any = True
        key = (
            str(op.get("op")),
            str(op.get("slug") or "").strip(),
            str(op.get("bullet") or "").strip(),
        )
        by_key.setdefault(key, set()).update(
            str(r).strip() for r in refs if isinstance(r, str)
        )
    if not saw_any:
        return None
    landed: Set[str] = set()
    for accepted in plan.accepted:
        if accepted["op"] not in ("add", "rewrite"):
            continue
        key = (accepted["op"], accepted["slug"], accepted.get("bullet") or "")
        landed |= by_key.get(key, set())
    return landed


def _absorb(
    run: _Run,
    by_handle: Dict[str, SourceRow],
    result: Dict[str, Any],
    pre_existing_slugs: Set[str],
    label: str,
    *,
    prior_complaints: Sequence[str] = (),
) -> Dict[str, Any]:
    """Fold ONE round's result into the run, and return its evidence.

    Shared by the first round and the retry so the two cannot drift: a row
    judged by one rule and re-asked by another is how a fact gets reported
    as kept and written nowhere.
    """
    plan: ops.OpsPlan = result["plan"]
    bullet_slugs = {
        op["slug"] for op in plan.accepted if op["op"] in ("add", "rewrite")
    }
    live_slugs = {s for s, st in plan.states.items() if not st.deleted}
    landed_refs = _landed_refs(result["raw_ops"], plan)
    if landed_refs is None and "`refs`" not in " ".join(run.notes):
        run.notes.append(
            "the writer omitted `refs` on its ops, so a row's disposition is "
            "evidenced at FILE level rather than per bullet"
        )
    claimed = _claims(result["dispositions"], by_handle)
    _record_dispositions(
        run, by_handle, result["dispositions"],
        list(prior_complaints) + list(result["rejected"]),
        bullet_slugs, live_slugs, landed_refs,
    )
    # A file counts as FILLED only when a bullet landed in it. A round can
    # legitimately create a file and then have its only bullet refused; the
    # empty shell is pruned at the end rather than announced in the memory
    # log as something that was migrated.
    for op in plan.accepted:
        if op["op"] == "create_file" and op["slug"] not in run.created_ops:
            run.created_ops.append(op["slug"])
    for slug in sorted(bullet_slugs):
        if slug not in run.files_filled:
            run.files_filled.append(slug)
        if slug not in pre_existing_slugs and slug not in run.files_created:
            run.files_created.append(slug)
    if result["rejected"]:
        run.errors.extend(f"{label}: {c}" for c in result["rejected"][:6])
    return {
        "bullet_slugs": bullet_slugs,
        "live_slugs": live_slugs,
        "landed_refs": landed_refs,
        "claimed": claimed,
    }


async def _advance(
    db: AsyncSession,
    user_id: str,
    virtuals: List[_VirtualFile],
    plan: ops.OpsPlan,
    dry_run: bool,
) -> List[_VirtualFile]:
    """The file set the NEXT round is written against.

    Re-read from the database on a real run (the ops committed), projected
    from the plan on a dry one. Both matter for the retry specifically: it
    must see what already landed, or it re-proposes it.
    """
    if dry_run:
        return _apply_to_virtuals(virtuals, plan)
    return _virtuals_from(await ops._all_files(db, user_id))


async def _persist_progress(
    db: AsyncSession, run: _Run, report: Dict[str, Any], *, in_flight: Sequence[str] = (),
) -> None:
    marker = await _marker(db)
    if marker is None:
        return
    marker.progress_json = json.dumps({
        "attempts": report.get("attempts", 1),
        "batches_done": run.batches_done,
        "in_flight": list(in_flight),
        "report": report,
    })
    await db.commit()


@dataclass
class _Gate:
    """What the marker says about whether this run may proceed.

    Extracted from `migrate_user` because it is the part with the early
    returns in it, and a lifecycle whose exits are buried 80 lines inside a
    350-line function is a lifecycle nobody re-reads before changing the
    thing above it.
    """

    attempts: int = 1
    resume_dispositions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    in_flight: List[str] = field(default_factory=list)
    #: Non-None means "do not run"; the caller returns it verbatim.
    early_return: Optional[Dict[str, Any]] = None


async def _claim_run(
    db: AsyncSession, user_id: str, started: datetime, *, dry_run: bool
) -> _Gate:
    """Marker lifecycle: force-reset, already-done, give-up, grace, resume.

    A dry run never touches the marker, so it never claims anything and
    always proceeds.
    """
    from app.db.models.day_chat import MigrationStatus

    gate = _Gate()
    if dry_run:
        return gate

    marker = await _marker(db)
    if _forced() and marker is not None and marker.status in ("completed", "failed"):
        logger.info("[memory_v3] %s set — resetting the marker", FORCE_ENV)
        marker.status = "not_started"
        marker.started_at = None
        marker.completed_at = None
        marker.progress_json = None
        marker.error_message = None
        await db.commit()
    if marker is None:
        db.add(MigrationStatus(migration_name=MIGRATION_NAME, status="not_started"))
        await db.commit()
        marker = await _marker(db)

    progress = json.loads(marker.progress_json) if marker.progress_json else {}
    gate.attempts = int(progress.get("attempts", 1))

    if marker.status == "completed":
        gate.early_return = {
            **(progress.get("report") or _empty_report(user_id, started)),
            "status": "already_completed",
        }
        return gate

    if marker.status == "failed":
        prev = progress.get("report") or {}
        grace_elapsed = True
        last_failed = prev.get("last_failed_at")
        if last_failed:
            try:
                grace_elapsed = (
                    (_now() - datetime.fromisoformat(last_failed)).total_seconds()
                    >= RETRY_GRACE_SEC
                )
            except Exception:
                grace_elapsed = True
        if gate.attempts >= MAX_AUTO_RETRY_ATTEMPTS:
            logger.error(
                "[memory_v3] giving up after %d attempts (%s). Set %s=true to "
                "re-run.", gate.attempts, marker.error_message, FORCE_ENV,
            )
            gate.early_return = {**prev, "status": "failed", "gave_up": True}
            return gate
        if not grace_elapsed:
            gate.early_return = {**prev, "status": "failed", "retry_deferred": True}
            return gate
        gate.attempts += 1
        logger.info("[memory_v3] auto-retry attempt %d/%d",
                    gate.attempts, MAX_AUTO_RETRY_ATTEMPTS)

    # Resume ledger: the dispositions already written ARE the cursor — except
    # the two that are not outcomes. A row whose batch FAILED was never
    # judged, and a row the writer left UNACCOUNTED was judged by nobody;
    # carrying either forward as "done" turns a partial run into a silently
    # permanent loss on the retry that was supposed to fix it.
    for entry in (progress.get("report") or {}).get("dispositions", []):
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        if entry.get("disposition") in (FAILED, UNACCOUNTED):
            continue
        gate.resume_dispositions[entry["id"]] = entry
    gate.in_flight = [str(i) for i in (progress.get("in_flight") or [])]
    for rid in gate.in_flight:
        # Its ops may or may not have committed. Feed it again: a fact lost
        # is unrecoverable, a duplicate bullet is one instruct away, and
        # `validate_ops` rejects the exact-match case outright.
        gate.resume_dispositions.pop(rid, None)

    marker.status = "in_progress"
    marker.started_at = marker.started_at or started
    marker.error_message = None
    # Persist the bumped attempt counter NOW, not at the first batch
    # boundary. A failure BEFORE the first batch (an unreadable corpus, a
    # dead pool) never reaches `_persist_progress`, so a counter written only
    # there would stay at 1 forever and the "give up after 3" cap would be
    # unreachable — an hourly retry loop with a model call in it, for the
    # length of the deploy.
    marker.progress_json = json.dumps({
        "attempts": gate.attempts,
        "batches_done": progress.get("batches_done", 0),
        "in_flight": gate.in_flight,
        "report": progress.get("report"),
    })
    await db.commit()
    return gate


async def migrate_user(
    db: AsyncSession,
    user_id: str,
    *,
    dry_run: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Migrate one tenant. The whole of §7's per-user half.

    Returns the report. On a dry run nothing at all is written — not the
    marker, not a file, not a change line, not a legacy row.
    """
    t0 = time.monotonic()
    started = _now()
    run = _Run()

    # ── The marker, and whether we are allowed to run ────────────────
    gate = await _claim_run(db, user_id, started, dry_run=dry_run)
    if gate.early_return is not None:
        return gate.early_return
    attempts = gate.attempts
    resume_dispositions = gate.resume_dispositions
    in_flight = gate.in_flight

    identity = await resolve_user_identity(db, user_id)
    today = await memory_curator._today_for(db, user_id)

    # ── The before-state ─────────────────────────────────────────────
    eligible_all, predecided, stats = await read_corpus(db, user_id)
    if not dry_run:
        await ops.ensure_system_files(db, user_id)
        await db.commit()
        existing_rows = await ops._all_files(db, user_id)
        before_files = await _file_snapshot(db, user_id)
        virtuals = _virtuals_from(existing_rows)
    else:
        existing_rows = await ops._all_files(db, user_id)
        virtuals = _virtuals_from(existing_rows)
        have = {f.slug for f in virtuals}
        for slug, spec in SYSTEM_FILES.items():
            if slug not in have:
                virtuals.append(_VirtualFile(
                    slug=slug, section=spec["section"], title=spec["title"],
                    description=spec["description"], body_md="", is_system=True,
                ))
        virtuals.sort(key=lambda f: f.slug)
        # Includes the three system files even when the tenant has none yet:
        # a real run creates them through `ensure_system_files` before the
        # first batch, so a dry run that excluded them would report Profile
        # as a file the MIGRATION created and hand the rollback a slug it
        # would then delete.
        before_files = _virtual_snapshot(virtuals)

    pre_existing_slugs = {f["slug"] for f in before_files} | set(SYSTEM_FILES)
    before = {
        "legacy": stats,
        "files": before_files,
        "eligible_rows": len(eligible_all),
        "not_eligible": len(predecided),
    }

    for entry in predecided:
        run.dispositions[entry["id"]] = entry
    run.dispositions.update(resume_dispositions)
    if in_flight:
        run.notes.append(
            f"{len(in_flight)} row(s) were in flight when a previous attempt "
            "stopped and were processed again; look for a duplicate bullet in "
            "the files they name"
        )

    eligible = [r for r in eligible_all if r.id not in resume_dispositions]
    batches = plan_batches(eligible)

    report: Dict[str, Any] = {
        "migration": MIGRATION_NAME,
        "user_id": user_id,
        "dry_run": dry_run,
        "status": "in_progress",
        "attempts": attempts,
        "started_at": _iso(started),
        "completed_at": None,
        "before": before,
        "after": {"files": []},
        "tallies": {},
        "dispositions": [],
        "files_created": [],
        "files_filled": [],
        "batches": {"total": len(batches), "done": 0, "failed": 0},
        "errors": [],
        "notes": run.notes,
    }

    # ── The batches ──────────────────────────────────────────────────
    for index, batch in enumerate(batches):
        entries, by_handle = render_entries(batch)
        if not dry_run:
            await _persist_progress(
                db, run,
                _finish(report, run, _virtual_snapshot(virtuals), started, "in_progress"),
                in_flight=[r.id for r in batch],
            )
        try:
            result = await memory_curator.curate_migration_batch(
                db, user_id,
                entries=entries,
                existing=virtuals,
                identity=identity,
                today=today,
                api_key=api_key,
                dry_run=dry_run,
            )
        except Exception as exc:
            # One tenant's bad batch must not end the tenant's migration,
            # and one tenant must not end the fleet pass. The rows are
            # recorded as unjudged, which is what makes a partial run
            # readable instead of merely incomplete.
            run.batches_failed += 1
            run.errors.append(f"batch {index + 1}: {type(exc).__name__}: {exc}"[:400])
            logger.warning("[memory_v3] batch %d failed for user=%s: %s",
                           index + 1, str(user_id)[:8], exc)
            for row in batch:
                run.dispositions[row.id] = {
                    "id": row.id,
                    "preview": row.preview,
                    "legacy_file": row.legacy_slug,
                    "category": row.category,
                    "disposition": FAILED,
                    "slug": None,
                    "reason": f"the batch carrying it failed: {type(exc).__name__}",
                }
            continue

        evidence = _absorb(
            run, by_handle, result, pre_existing_slugs, f"batch {index + 1}",
        )
        virtuals = await _advance(db, user_id, virtuals, result["plan"], dry_run)

        # ── ONE extra round, for the rows a partial refusal orphaned ──
        # `validate_ops` refuses ops, and the writer only re-asks when the
        # WHOLE proposal was refused. A batch with four good bullets and one
        # refused one therefore never re-asked, and the refused ROW was gone
        # — which is how the owner's own name, and a standing arrangement the
        # durability rules explicitly allow, ended up nowhere. Exactly one
        # extra round: `allow_retry=False` on the call, no loop here, and the
        # retry is fed the state AFTER the accepted ops so it cannot
        # double-write what already landed.
        orphans = orphaned_rows(
            by_handle, evidence["claimed"], evidence["bullet_slugs"],
            evidence["live_slugs"], evidence["landed_refs"],
        )
        if orphans:
            entries2, by_handle2 = render_entries(orphans)
            try:
                retry = await memory_curator.curate_migration_batch(
                    db, user_id,
                    entries=entries2,
                    existing=virtuals,
                    identity=identity,
                    today=today,
                    api_key=api_key,
                    dry_run=dry_run,
                    extra_instructions=build_retry_block(result["rejected"]),
                    allow_retry=False,
                )
            except Exception as exc:
                # Round 1's verdict stands: those rows are already recorded
                # as dropped WITH the complaint that refused them, which is
                # a real answer. Losing the retry is not losing the record.
                run.errors.append(
                    f"batch {index + 1} retry: {type(exc).__name__}: {exc}"[:400]
                )
                logger.warning(
                    "[memory_v3] batch %d retry failed for user=%s: %s",
                    index + 1, str(user_id)[:8], exc,
                )
            else:
                # Overwrites exactly the orphaned rows (the ledger is keyed
                # by row id), carrying BOTH rounds' complaints so a row
                # refused twice still says why, twice.
                _absorb(
                    run, by_handle2, retry, pre_existing_slugs,
                    f"batch {index + 1} retry",
                    prior_complaints=result["rejected"],
                )
                virtuals = await _advance(
                    db, user_id, virtuals, retry["plan"], dry_run
                )
                run.notes.append(
                    f"batch {index + 1}: re-asked for {len(orphans)} entr"
                    f"{'y' if len(orphans) == 1 else 'ies'} whose ops were "
                    "refused"
                )

        run.batches_done += 1
        if not dry_run:
            await _persist_progress(
                db, run,
                _finish(report, run, _virtual_snapshot(virtuals), started, "in_progress"),
            )

    # ── Prune the empty shells ───────────────────────────────────────
    # A batch may create a file and then lose its only bullet to the lint.
    # Left behind, that is a file the user opens to find nothing in, with a
    # "Migrated from your earlier memory" line pointing at it.
    orphans = [s for s in run.created_ops if s not in run.files_filled]
    if orphans:
        run.notes.append(
            "dropped " + ", ".join(sorted(orphans))
            + " — created by the writer but every bullet it proposed for them "
              "was refused"
        )
        if dry_run:
            virtuals = [f for f in virtuals if f.slug not in set(orphans)]
        else:
            from app.db.models.memory import MemoryFile as _MF

            for row in (await db.execute(
                select(_MF).where(
                    _MF.user_id == user_id, _MF.slug.in_(orphans)
                )
            )).scalars().all():
                if not (row.body_md or "").strip() and row.slug not in SYSTEM_FILES:
                    await db.delete(row)
            await db.commit()

    # ── The change log, one line per file this actually filled ───────
    if not dry_run and run.files_filled:
        from app.db.models.memory import MemoryFileChange as _MFC

        # A resumed run must not file the same line twice. The log is a
        # calendar the user reads; two identical "Migrated from your earlier
        # memory" entries against one file describe an event that happened
        # once.
        already = {
            row[0] for row in (await db.execute(
                select(_MFC.file_slug).where(
                    _MFC.user_id == user_id, _MFC.summary == MIGRATED_SUMMARY,
                )
            )).all()
        }
        titles = {f.slug: f.title for f in virtuals}
        for slug in run.files_filled:
            if slug in already:
                continue
            await ops.write_change_line(
                db, user_id, slug, titles.get(slug) or slug,
                "created" if slug in run.files_created else "updated",
                MIGRATED_SUMMARY,
            )
        await db.commit()

    after_files = (
        await _file_snapshot(db, user_id) if not dry_run
        else _virtual_snapshot(virtuals)
    )
    status = "completed" if not run.batches_failed else "completed_with_errors"
    final = _finish(report, run, after_files, started, status)
    final["completed_at"] = _iso(_now())
    final["elapsed_sec"] = round(time.monotonic() - t0, 2)

    if not dry_run:
        marker = await _marker(db)
        if marker is not None:
            # A batch that FAILED left rows nothing judged, so the run is not
            # finished even though it returned. The marker goes `failed` so
            # the ordinary retry policy (attempts + grace) picks it up on the
            # nightly slot and re-feeds exactly those rows; `completed` here
            # would freeze a transient provider blip into a permanent hole.
            # Validator complaints are NOT a failure — those are decisions,
            # and retrying a refusal costs one model call per attempt to
            # arrive at the same refusal.
            failed_batches = run.batches_failed > 0
            marker.status = "failed" if failed_batches else "completed"
            marker.completed_at = None if failed_batches else _now()
            marker.error_message = (
                "; ".join(run.errors)[:2000] if run.errors else None
            )
            if failed_batches:
                final["last_failed_at"] = _iso(_now())
            marker.progress_json = json.dumps({
                "attempts": attempts, "batches_done": run.batches_done,
                "in_flight": [], "report": final,
            })
            await db.commit()

    logger.info(
        "[memory_v3] user=%s dry_run=%s rows=%d batches=%d/%d files=%s %s",
        str(user_id)[:8], dry_run, len(eligible_all), run.batches_done,
        len(batches), run.files_filled, final["tallies"],
    )
    return final


def _finish(
    report: Dict[str, Any],
    run: _Run,
    after_files: List[Dict[str, Any]],
    started: datetime,
    status: str,
) -> Dict[str, Any]:
    dispositions = sorted(run.dispositions.values(), key=lambda d: d["id"])
    out = dict(report)
    out.update({
        "status": status,
        "after": {"files": after_files},
        "tallies": _tallies(dispositions),
        "dispositions": dispositions,
        "files_created": list(run.files_created),
        "files_filled": list(run.files_filled),
        "batches": {
            "total": report["batches"]["total"],
            "done": run.batches_done,
            "failed": run.batches_failed,
        },
        "errors": list(run.errors),
        "notes": list(run.notes),
    })
    return out


def _empty_report(user_id: str, started: datetime) -> Dict[str, Any]:
    return {
        "migration": MIGRATION_NAME,
        "user_id": user_id,
        "dry_run": False,
        "status": "completed",
        "attempts": 1,
        "started_at": _iso(started),
        "completed_at": _iso(started),
        "before": {}, "after": {"files": []}, "tallies": {},
        "dispositions": [], "files_created": [], "files_filled": [],
        "batches": {"total": 0, "done": 0, "failed": 0},
        "errors": [], "notes": ["no stored report — the marker predates it"],
    }


async def migrate_user_guarded(
    db: AsyncSession,
    user_id: str,
    *,
    dry_run: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """`migrate_user` with the failure half of the marker lifecycle.

    A raise anywhere inside leaves the marker `failed` with the attempt
    counter and the timestamp the auto-retry policy reads, and re-raises so
    the caller (route or scheduler) sees a real error rather than a report
    claiming success.
    """
    try:
        return await migrate_user(db, user_id, dry_run=dry_run, api_key=api_key)
    except Exception as exc:
        if dry_run:
            raise
        try:
            marker = await _marker(db)
            if marker is not None:
                progress = json.loads(marker.progress_json) if marker.progress_json else {}
                stored = progress.get("report") or {}
                stored["last_failed_at"] = _iso(_now())
                stored["status"] = "failed"
                marker.status = "failed"
                marker.error_message = str(exc)[:2000]
                marker.progress_json = json.dumps({
                    "attempts": int(progress.get("attempts", 1)),
                    "batches_done": progress.get("batches_done", 0),
                    "in_flight": progress.get("in_flight", []),
                    "report": stored,
                })
                await db.commit()
        except Exception:  # pragma: no cover - the marker is best effort here
            logger.exception("[memory_v3] could not record the failure")
        raise


# ── The report route's body ───────────────────────────────────────────

async def read_report(db: AsyncSession, user_id: str) -> Dict[str, Any]:
    """Marker state + the stored report + a LIVE snapshot.

    The live half is what makes this endpoint usable as the fleet driver's
    in-band backup: called before a real run it is the tenant's "before"
    state, and called after it is the proof of what changed.
    """
    marker = await _marker(db)
    progress = {}
    if marker is not None and marker.progress_json:
        try:
            progress = json.loads(marker.progress_json)
        except Exception:
            progress = {}
    return {
        "migration": MIGRATION_NAME,
        "user_id": user_id,
        "status": marker.status if marker is not None else "not_started",
        "attempts": int(progress.get("attempts", 0) or 0),
        "started_at": _iso(marker.started_at) if marker is not None else None,
        "completed_at": _iso(marker.completed_at) if marker is not None else None,
        "error": marker.error_message if marker is not None else None,
        "report": progress.get("report"),
        "snapshot": await snapshot(db, user_id),
    }


# ── Rollback ──────────────────────────────────────────────────────────

ROLLBACK_CONFIRM = "ROLLBACK MEMORY V3"


async def rollback(
    db: AsyncSession, user_id: str, *, hard: bool = False
) -> Dict[str, Any]:
    """Undo the migration for one tenant. See docs/memory/migration-v3-runbook.md.

    Report-driven by default: the stored report names every file the
    migration CREATED (deleted) and every file it FILLED (body restored to
    the exact `before` body). `hard=True` is the escape hatch for a tenant
    whose report is gone — it drops every v3-sectioned file and the whole
    change log.

    What this does NOT restore, in either mode: anything the v3 writer
    wrote AFTER the migration (those bullets are dropped with the files),
    the Memory log, and `you/current-context`. It does not need to restore
    the legacy `memories` rows because the migration never touched them —
    that is the whole reason the rollback is small.

    Run this BEFORE redeploying the previous image tag: the route that
    reaches this function does not exist in that image.
    """
    from app.db.models.memory import MemoryFileChange

    marker = await _marker(db)
    report = None
    if marker is not None and marker.progress_json:
        try:
            report = (json.loads(marker.progress_json) or {}).get("report")
        except Exception:
            report = None

    rows = {r.slug: r for r in await ops._all_files(db, user_id)}
    deleted: List[str] = []
    restored: List[str] = []

    if report and not hard:
        before_bodies = {
            f["slug"]: f.get("body_md", "")
            for f in (report.get("before") or {}).get("files", [])
        }
        created = set(report.get("files_created") or [])
        filled = set(report.get("files_filled") or [])
        for slug in sorted(created):
            row = rows.get(slug)
            if row is not None:
                await db.delete(row)
                deleted.append(slug)
        for slug in sorted(filled - created):
            row = rows.get(slug)
            if row is None:
                continue
            row.body_md = before_bodies.get(slug, "")
            row.updated_at = _now()
            restored.append(slug)
    else:
        for slug, row in sorted(rows.items()):
            if slug in SYSTEM_FILES:
                row.body_md = ""
                row.updated_at = _now()
                restored.append(slug)
            else:
                await db.delete(row)
                deleted.append(slug)

    changes = (await db.execute(
        select(MemoryFileChange).where(MemoryFileChange.user_id == user_id)
    )).scalars().all()
    for change in changes:
        await db.delete(change)

    if marker is not None:
        marker.status = "not_started"
        marker.started_at = None
        marker.completed_at = None
        marker.progress_json = None
        marker.error_message = None
    await db.commit()

    logger.warning(
        "[memory_v3] ROLLBACK user=%s deleted=%s restored=%s changes=%d hard=%s",
        str(user_id)[:8], deleted, restored, len(changes), hard,
    )
    return {
        "rolled_back": True,
        "hard": hard,
        "used_report": bool(report and not hard),
        "files_deleted": deleted,
        "files_restored": restored,
        "change_rows_removed": len(changes),
        "legacy_rows_touched": 0,
    }


# ── The scheduler slot ────────────────────────────────────────────────

async def run_scheduled_migration() -> Dict[str, Any]:
    """The boot one-shot (T+180s) and the nightly retry, in one function.

    Called from `memory_file_ops.run_memory_maintenance`, which is what
    both APScheduler registrations in `agent_main.py` already point at —
    deliberately, so this does not become a second registration to keep in
    sync with the first. Marker-guarded, so every fire after the first is a
    single indexed SELECT.

    Single-tenant: an agent container serves exactly one user. On the
    platform there is no `settings.user_id` and this returns immediately.
    """
    from app.config import settings

    user_id = getattr(settings, "user_id", "") or ""
    if not user_id:
        return {"skipped": "no tenant user"}

    from app.db.database import async_session_maker

    async with async_session_maker() as db:
        marker = await _marker(db)
        if marker is not None and marker.status == "completed" and not _forced():
            return {"skipped": "already_completed"}
        api_key = await _tenant_api_key(db, user_id)
        try:
            report = await migrate_user_guarded(
                db, user_id, dry_run=False, api_key=api_key,
            )
        except Exception as exc:
            logger.error("[memory_v3] scheduled migration failed: %s", exc)
            return {"status": "failed", "error": str(exc)[:400]}
    return {
        "status": report.get("status"),
        "files": report.get("files_filled"),
        "tallies": report.get("tallies"),
    }


async def _tenant_api_key(db: AsyncSession, user_id: str) -> Optional[str]:
    try:
        from app.db.models import AgentConfig

        return (await db.execute(
            select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
        )).scalar_one_or_none()
    except Exception:
        return None
