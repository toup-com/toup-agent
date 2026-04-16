"""
Natural-language date resolution for the recall_day tool.

Resolves phrases like "yesterday", "last Monday", "April 10", "3 days ago",
or ISO dates ("2026-04-15") to a concrete `date` in the user's timezone.

Rules:
  - Weekday names ("Monday") always resolve to the most recent PAST occurrence.
    If today is Monday and the user says "Monday", resolve to 7 days ago —
    never to today, never to next Monday. "last Monday" is a synonym.
  - Future references are rejected — recall is about the past.
  - Ambiguous references (e.g. "May 3" in December could be this year or last)
    resolve to the most recent past occurrence within 1 year.

Standalone module — no `app.config` import so tests can exercise it without
pulling the whole settings stack.
"""

from __future__ import annotations

import re
from datetime import date as Date, datetime, timedelta, timezone
from typing import Optional


_WEEKDAY_MAP = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tues": 1, "tue": 1,
    "wednesday": 2, "weds": 2, "wed": 2,
    "thursday": 3, "thurs": 3, "thur": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _today_in_tz(tz_name: Optional[str]) -> Date:
    """Return today's calendar date in the user's timezone, or UTC if no tz.

    DST-safe: uses IANA-aware `zoneinfo` and timezone-aware "now" — so on the
    morning after a spring-forward the local date is correctly reported, and
    "yesterday" resolves to the calendar day that just ended in the user's
    wall clock (not the UTC calendar day).
    """
    if tz_name:
        try:
            import zoneinfo
            return datetime.now(zoneinfo.ZoneInfo(tz_name)).date()
        except Exception:
            pass
    return datetime.now(timezone.utc).date()


def _most_recent_past_weekday(today: Date, target_weekday: int) -> Date:
    """Return the most recent past occurrence of target_weekday, never today.

    If today IS target_weekday, we return today - 7 (a week ago), not today.
    This matches how people use weekday names when referring to the past.
    """
    days_since = (today.weekday() - target_weekday) % 7
    if days_since == 0:
        days_since = 7
    return today - timedelta(days=days_since)


def _most_recent_past_month_day(today: Date, month: int, day: int) -> Optional[Date]:
    """Resolve "April 10" to the most recent past April 10.

    If the date this year is in the future, roll back to last year.
    """
    for year in (today.year, today.year - 1):
        try:
            candidate = Date(year, month, day)
        except ValueError:
            return None
        if candidate <= today:
            return candidate
    return None


def resolve_date_reference(ref: str, tz_name: Optional[str] = None) -> Optional[Date]:
    """Resolve a natural-language date reference to a concrete local date.

    Args:
        ref: User-supplied date reference. Examples:
             "yesterday", "today", "Monday", "last Tuesday", "3 days ago",
             "April 10", "Apr 10", "10 April", "2026-04-15".
        tz_name: IANA timezone for the user (e.g. "America/Toronto").
                 Falls back to UTC if missing or invalid.

    Returns:
        A `datetime.date` in the user's local timezone, or None if unresolvable
        or in the future.
    """
    if not ref or not isinstance(ref, str):
        return None

    s = ref.strip().lower()
    if not s:
        return None

    today = _today_in_tz(tz_name)

    # 1. ISO date — unambiguous.
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            candidate = Date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if candidate > today:
                return None
            return candidate
        except ValueError:
            return None

    # 2. Literal keywords.
    if s == "today":
        return today
    if s == "yesterday":
        return today - timedelta(days=1)
    if s in ("day before yesterday", "the day before yesterday"):
        return today - timedelta(days=2)

    # 3. "N days/weeks/months ago".
    m = re.fullmatch(r"(\d+)\s+days?\s+ago", s)
    if m:
        return today - timedelta(days=int(m.group(1)))
    m = re.fullmatch(r"(\d+)\s+weeks?\s+ago", s)
    if m:
        return today - timedelta(weeks=int(m.group(1)))
    m = re.fullmatch(r"(\d+)\s+months?\s+ago", s)
    if m:
        months_ago = int(m.group(1))
        year = today.year
        month = today.month - months_ago
        while month <= 0:
            month += 12
            year -= 1
        # Clamp day to last valid day of target month
        day = min(today.day, _last_day_of_month(year, month))
        return Date(year, month, day)

    # 4. "last week" / "last month" — resolve to 7 days / ~30 days ago.
    if s in ("last week", "a week ago"):
        return today - timedelta(days=7)
    if s in ("last month", "a month ago"):
        year = today.year
        month = today.month - 1
        if month == 0:
            month = 12
            year -= 1
        day = min(today.day, _last_day_of_month(year, month))
        return Date(year, month, day)

    # 5. Weekday references — always past.
    # Accept: "Monday", "last Monday", "this past Monday", "on Monday".
    weekday_match = re.fullmatch(
        r"(?:last\s+|this\s+past\s+|past\s+|on\s+)?(monday|mon|tuesday|tues|tue|wednesday|weds|wed|thursday|thurs|thur|thu|friday|fri|saturday|sat|sunday|sun)",
        s,
    )
    if weekday_match:
        wd = _WEEKDAY_MAP[weekday_match.group(1)]
        return _most_recent_past_weekday(today, wd)

    # 6. Month-day references: "April 10", "Apr 10", "April 10th", "10 April".
    #    Month + day in either order, optional ordinal suffix, optional year.
    m = re.fullmatch(
        r"(?:on\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?",
        s,
    )
    if m:
        month = _MONTH_MAP[m.group(1)]
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else None
        if year is not None:
            try:
                candidate = Date(year, month, day)
                if candidate > today:
                    return None
                return candidate
            except ValueError:
                return None
        return _most_recent_past_month_day(today, month, day)

    m = re.fullmatch(
        r"(?:on\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:,?\s+(\d{4}))?",
        s,
    )
    if m:
        day = int(m.group(1))
        month = _MONTH_MAP[m.group(2)]
        year = int(m.group(3)) if m.group(3) else None
        if year is not None:
            try:
                candidate = Date(year, month, day)
                if candidate > today:
                    return None
                return candidate
            except ValueError:
                return None
        return _most_recent_past_month_day(today, month, day)

    return None


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        next_first = Date(year + 1, 1, 1)
    else:
        next_first = Date(year, month + 1, 1)
    return (next_first - timedelta(days=1)).day
