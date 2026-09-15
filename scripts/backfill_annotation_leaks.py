#!/usr/bin/env python3
"""Remove leaked `[channel h:mmam]` prefixes from stored assistant rows.

The read-side strip in `api/message_cards.public_text` makes the leak
invisible to every client with no client update. This makes it GONE, which is
what stops it reappearing through a future reader that forgets `public_text`.

Runs inside an agent container:

    docker exec -it <container> python scripts/backfill_annotation_leaks.py
    docker exec -it <container> python scripts/backfill_annotation_leaks.py --apply
    docker exec -it <container> python scripts/backfill_annotation_leaks.py --revert --apply

ONE rule for writes: nothing is written without `--apply`, in either
direction. `--revert` alone prints what it would restore.

REVERSIBILITY is what makes this approvable against a tenant DB: the original
prefix is stored under `metadata_json.leaked_prefix`, and `--revert` puts it
back. That marker is ALSO the idempotency check — a cleaned row no longer
matches the pattern, so "have I already done this row?" cannot be answered by
re-matching.

`messages.metadata_json` is a TEXT column holding a JSON string
(db/models/conversation.py) — never JSONB. Read it with json.loads, write it
with json.dumps, and preserve every key that was already there.
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("annotation_leak")

_BATCH = 500
_MARKER = "leaked_prefix"


def _pg_prefilter() -> str:
    """A Postgres-side `~*` prefilter built from the ONE channel list.

    Cheap and deliberately loose — every hit is re-verified in Python with the
    real `LEAKED_TAG_RE` before anything is written, so a prefilter that is
    slightly too generous costs a comparison and a prefilter that is too
    strict would silently skip rows.
    """
    from app.agent.channel_annotations import STRIPPABLE_CHANNELS

    alt = "|".join(sorted(STRIPPABLE_CHANNELS))
    return r"^\s*\[(" + alt + r")\s+\d{1,2}:\d{2}\s?(am|pm)\]"


async def _iter_candidates(session_maker, user_id, limit, revert):
    """Yield batches of (id, content, metadata_json, channel) rows.

    Paged on `id` rather than OFFSET: `--apply` rewrites the rows it reads, so
    an offset would step over the rows that shifted under it.
    """
    from sqlalchemy import and_, select

    from app.db.models import Conversation, Message

    last_id = ""
    seen = 0
    while True:
        conds = [Message.role != "user", Message.id > last_id]
        if user_id:
            conds.append(Conversation.user_id == user_id)
        if revert:
            conds.append(Message.metadata_json.like('%"' + _MARKER + '"%'))
        else:
            conds.append(Message.content.op("~*")(_pg_prefilter()))
        stmt = (
            select(Message.id, Message.content, Message.metadata_json,
                   Message.channel, Conversation.channel)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(and_(*conds))
            .order_by(Message.id.asc())
            .limit(_BATCH)
        )
        async with session_maker() as db:
            rows = (await db.execute(stmt)).all()
        if not rows:
            return
        last_id = rows[-1][0]
        if limit and seen + len(rows) > limit:
            rows = rows[: limit - seen]
        seen += len(rows)
        yield rows
        if limit and seen >= limit:
            return


async def _run(args) -> int:
    from sqlalchemy import update

    from app.agent.channel_annotations import LEAKED_TAG_RE, strip_leaked_tags
    from app.db.database import async_session_maker
    from app.db.models import Message

    hist: Counter = Counter()
    touched = 0
    skipped = 0

    async for rows in _iter_candidates(
        async_session_maker, args.user_id, args.limit, args.revert
    ):
        writes = []
        for mid, content, meta_raw, msg_channel, conv_channel in rows:
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except (TypeError, ValueError):
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            channel = (msg_channel or conv_channel or "web")

            if args.revert:
                prefix = meta.get(_MARKER)
                if not prefix:
                    skipped += 1
                    continue
                already = (content or "").startswith(prefix)
                meta.pop(_MARKER, None)
                # An already-restored row still has its marker dropped, so a
                # second --revert pass finds nothing and is a true no-op.
                new_content = content if already else prefix + (content or "")
                writes.append((mid, new_content, json.dumps(meta) if meta else None))
                if already:
                    skipped += 1
                else:
                    hist[channel] += 1
                continue

            if _MARKER in meta:
                # Already done. A cleaned row no longer matches the pattern,
                # so the STORED MARKER is the only honest idempotency check.
                skipped += 1
                continue
            if not content or not LEAKED_TAG_RE.match(content):
                skipped += 1
                continue
            cleaned, prefix = strip_leaked_tags(content)
            if not prefix or cleaned == content:
                skipped += 1
                continue
            meta[_MARKER] = prefix
            writes.append((mid, cleaned, json.dumps(meta)))
            hist[channel] += 1

        if not writes:
            continue
        touched += len(writes)
        if args.apply:
            async with async_session_maker() as db:
                for mid, new_content, new_meta in writes:
                    await db.execute(
                        update(Message)
                        .where(Message.id == mid)
                        .values(content=new_content, metadata_json=new_meta)
                    )
                await db.commit()
            logger.info("[annotation_leak] committed batch of %d", len(writes))
        else:
            for mid, new_content, _m in writes[:5]:
                print(f"  {mid[:12]}  -> {new_content[:80]!r}")

    direction = "revert" if args.revert else "strip"
    verb = "applied" if args.apply else "would apply"
    print(f"\n[annotation_leak] {verb} {direction} to {touched} row(s); "
          f"skipped {skipped}")
    if hist:
        print("  per channel:")
        for ch, n in hist.most_common():
            print(f"    {ch:<12} {n}")
    if not args.apply:
        print("\n  DRY RUN — nothing was written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Strip leaked [channel h:mmam] prefixes from assistant rows"
    )
    ap.add_argument("--apply", action="store_true", help="Write the changes")
    ap.add_argument("--revert", action="store_true",
                    help="Restore metadata_json.leaked_prefix onto content "
                         "(still needs --apply to write)")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N rows")
    ap.add_argument("--user-id", default=None, help="Only this user's rows")
    sys.exit(asyncio.run(_run(ap.parse_args())))
