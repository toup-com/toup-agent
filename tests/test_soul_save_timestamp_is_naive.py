"""An aware datetime into a naive column failed the whole soul save.

Found 2026-08-13 from prod logs. "Save & Apply" on /soul returned Internal
Server Error and saved nothing:

    INSERT INTO soul_configs (..., vps_soul_synced_at, ...)
      VALUES (..., $10::TIMESTAMP WITHOUT TIME ZONE, ...)
    asyncpg.DataError: invalid input for query argument $10:
      datetime(2026, 8, 13, 23, 40, 4, tzinfo=timezone.utc)
      (can't subtract offset-naive and offset-aware datetimes)

`soul_configs.vps_soul_synced_at` is `mapped_column(DateTime)` — no
`timezone=True`, so TIMESTAMP WITHOUT TIME ZONE. One line assigned
`datetime.now(timezone.utc)` to it. Every other timestamp written in that
file is naive `datetime.utcnow()`; this was the lone outlier.

It was not a lost timestamp. That assignment sits immediately above the
ONLY `db.commit()` on the path, so the bad value failed the INSERT of the
entire row — name, colour, pronouns, traits, compiled prompt. The user's
soul never saved at all.

Second defect, stacked on top: the `except` block reported the failure with
`current_user.id`. After `db.rollback()` that attribute is expired, so
reading it attempted lazy IO and raised MissingGreenlet — the handler
written to surface the cause destroyed it, leaving a bare 500.
"""

from __future__ import annotations

import pathlib
import re

SOUL = pathlib.Path("app/api/soul.py")
MODEL = pathlib.Path("app/db/models/soul_config.py")


def _src() -> str:
    return SOUL.read_text()


def test_the_sync_timestamp_is_naive():
    """The assignment must not be tz-aware while the column is naive."""
    m = re.search(r"config\.vps_soul_synced_at\s*=\s*(.+)", _src())
    assert m, "vps_soul_synced_at is no longer assigned — did the path move?"
    expr = m.group(1).strip()
    assert "timezone.utc" not in expr or "replace(tzinfo=None)" in expr, (
        f"vps_soul_synced_at is assigned an AWARE datetime ({expr}) but the "
        f"column is TIMESTAMP WITHOUT TIME ZONE. asyncpg rejects that, and "
        f"because the only commit on this path follows immediately, it fails "
        f"the INSERT of the whole soul row — the save silently does nothing."
    )


def test_the_column_is_still_naive():
    """The test above is only correct while the column stays naive. If someone
    migrates it to `timezone=True`, this fails and forces both to move."""
    src = MODEL.read_text()
    m = re.search(r"vps_soul_synced_at.*?mapped_column\((.*?)\)", src, re.S)
    assert m, "vps_soul_synced_at column not found"
    assert "timezone=True" not in m.group(1), (
        "the column became timezone-aware — the assignment in soul.py must "
        "become aware in the same change, or the mismatch simply inverts"
    )


def test_every_timestamp_written_in_soul_is_consistent():
    """One outlier caused this. Catch the next one at the file level."""
    bad = [
        ln.strip()
        for ln in _src().splitlines()
        if re.search(r"^\s*\w[\w.]*\.\w*_at\s*=\s*datetime\.now\(timezone\.utc\)\s*$", ln)
    ]
    assert not bad, (
        f"aware datetime assigned to a *_at column: {bad}. Every timestamp "
        f"column in this module is naive UTC."
    )


def test_the_error_handler_does_not_touch_an_expired_orm_attribute():
    """`current_user.id` after `db.rollback()` raises MissingGreenlet, so the
    handler meant to REPORT the failure raises instead and the real cause is
    lost behind a bare 500. The id must be captured before the try."""
    src = _src()
    start = src.index("async def save_soul")
    end = src.index("async def _save_soul_impl", start)
    block = src[start:end]

    assert re.search(r"user_id\s*=\s*current_user\.id", block), (
        "save_soul no longer captures current_user.id before the try"
    )
    capture = block.index("user_id = current_user.id")
    try_at = block.index("try:", capture)
    assert capture < try_at, "the id is captured inside the try, not before it"

    rollback = block.index("await db.rollback()")
    tail = block[rollback:]
    assert "current_user.id" not in tail, (
        "the except block reads current_user.id AFTER rollback — that is the "
        "MissingGreenlet that masked the real error"
    )
