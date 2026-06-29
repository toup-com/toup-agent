"""Disposable / temp-mail domain blocklist.

A cheap, no-PII Sybil control: reject signups on known throwaway email
domains so one human can't spin up unlimited free-grant accounts via
mailinator/guerrillamail/etc.

Design:
* The domain set is loaded ONCE from a bundled, swappable text file
  (``data/disposable_email_domains.txt``) and extended by an optional
  external list at ``settings.disposable_email_domains_path``.
* FAIL-OPEN: any load error falls back to a small built-in core set, and
  ``is_disposable_email`` returns False on any unexpected error. A broken
  list must never block a legitimate signup.
* Gated at the call site by ``settings.disposable_email_blocklist_enabled``
  so false-positives can be monitored before enforcement.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# Built-in fallback if the bundled file can't be read. Kept tiny on
# purpose — the file is the real list; this just guarantees the feature
# degrades to "still catches the worst offenders" rather than "off".
_CORE_DOMAINS = frozenset({
    "mailinator.com", "guerrillamail.com", "10minutemail.com",
    "tempmail.com", "temp-mail.org", "yopmail.com", "sharklasers.com",
    "trashmail.com", "maildrop.cc", "getnada.com", "dispostable.com",
})

_BUNDLED_PATH = os.path.join(os.path.dirname(__file__), "data", "disposable_email_domains.txt")


def _read_domains(path: str) -> set[str]:
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip().lower()
            if s and not s.startswith("#"):
                out.add(s)
    return out


@lru_cache(maxsize=1)
def _domains() -> frozenset[str]:
    """Load + cache the blocklist. Fail-open to the core set."""
    domains: set[str] = set(_CORE_DOMAINS)
    try:
        domains |= _read_domains(_BUNDLED_PATH)
    except Exception as e:
        logger.warning("[disposable_email] bundled list unreadable (%s); using core set", e)

    # Optional external override/extension (e.g. a fuller maintained list).
    try:
        from app.config import settings
        extra_path = (getattr(settings, "disposable_email_domains_path", "") or "").strip()
        if extra_path and os.path.exists(extra_path):
            domains |= _read_domains(extra_path)
    except Exception as e:
        logger.warning("[disposable_email] external list load failed (%s); ignoring", e)

    return frozenset(domains)


def is_disposable_email(email: str) -> bool:
    """True iff the email's domain is a known disposable/temp-mail domain.

    Fail-open: returns False on any parse/lookup error so a list problem
    never rejects a real user.
    """
    try:
        e = (email or "").strip().lower()
        if "@" not in e:
            return False
        domain = e.rpartition("@")[2]
        return domain in _domains()
    except Exception:
        return False
