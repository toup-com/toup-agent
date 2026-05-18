"""Filter rules for `email_received` triggers.

User-authored filters live in `Trigger.filter_json`. The runner
evaluates them AFTER fetching the full message (so filters can match
on body content), BEFORE invoking the action handler.

Filter schema (all fields optional):
    {
      "from_contains":   [str, ...],   # match if ANY substring matches `From`
      "subject_contains":[str, ...],   # match if ANY substring matches Subject
      "labels":          [str, ...],   # match if message has ANY of these labels
      "exclude_categories": [str, ...] # drop if message in ANY of these inbox tabs
    }

Match semantics:
  - Empty filter_json → match everything.
  - All present fields are AND'ed: from must match AND subject must
    match AND labels must overlap AND categories must NOT overlap.
  - Within a list, items are OR'd: `from_contains=["foo","bar"]`
    matches if either substring appears.
  - Case-insensitive substring on string fields; exact label match
    on labels (Gmail labels are case-sensitive IDs).

The categories field maps to Gmail's tabs: `CATEGORY_PROMOTIONS`,
`CATEGORY_SOCIAL`, `CATEGORY_UPDATES`, `CATEGORY_FORUMS`. Users
configure them via short names (`"promotions"`, etc.) — we translate.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# User-facing short name → Gmail label id. We translate both ways so
# the API accepts either form (forgiving on input, canonical on
# storage).
_CATEGORY_ALIASES = {
    "promotions": "CATEGORY_PROMOTIONS",
    "social":     "CATEGORY_SOCIAL",
    "updates":    "CATEGORY_UPDATES",
    "forums":     "CATEGORY_FORUMS",
}


def _normalise_categories(raw: Iterable[str]) -> set[str]:
    """Map user-facing category names to Gmail label ids. Anything
    that doesn't map is passed through unchanged (advanced users can
    paste a raw label id and it just works)."""
    out: set[str] = set()
    for item in raw or []:
        if not item:
            continue
        s = str(item).strip()
        if not s:
            continue
        out.add(_CATEGORY_ALIASES.get(s.lower(), s))
    return out


def matches_filter(
    message: dict[str, Any],
    filter_json: Optional[dict[str, Any]],
) -> tuple[bool, str]:
    """Return `(matches, reason)`.

    `message` is the Gmail message dict as returned by
    `users.messages.get(format=metadata|full)` — we read From/Subject
    from `payload.headers`, labels from `labelIds`, and the body
    snippet from `snippet` for from/subject substring matching
    (snippet alone is enough for the v1 filter set).

    `reason` is a short opcode for the structured log:
      - "ok": all rules matched (or no rules to evaluate).
      - "no_from_match" / "no_subject_match" / "no_label_match":
        AND-rule failed; which one tells the operator + the user
        why their filter dropped the event.
      - "excluded_category": message was in an inbox tab the user
        excluded.
    """
    f = filter_json or {}
    if not isinstance(f, dict):
        # Bad data — fail open so a corrupted filter doesn't silently
        # drop every event. Logged for the operator to fix.
        logger.warning("[trigger_filter] filter_json not a dict: %r", type(f).__name__)
        return True, "ok"

    headers = _flatten_headers(message)
    labels = set((message.get("labelIds") or []))

    # From contains (substring, case-insensitive). Sample against the
    # header literally so "Acme <noreply@acme.io>" matches both
    # 'acme' and 'noreply'.
    from_terms = _strip_list(f.get("from_contains"))
    if from_terms:
        sender = (headers.get("from") or "").lower()
        if not any(t.lower() in sender for t in from_terms):
            return False, "no_from_match"

    # Subject contains. Snippet sometimes contains subject quoted text
    # — match against the actual Subject header, not snippet.
    subject_terms = _strip_list(f.get("subject_contains"))
    if subject_terms:
        subj = (headers.get("subject") or "").lower()
        if not any(t.lower() in subj for t in subject_terms):
            return False, "no_subject_match"

    # Label OR — match if any user-specified label is present on the
    # message. Gmail's INBOX is always present so users rarely want to
    # match it; non-empty label list = restrict to messages with those
    # labels.
    required_labels = _strip_list(f.get("labels"))
    if required_labels:
        if not labels.intersection(set(required_labels)):
            return False, "no_label_match"

    # Category exclude. The category labels are mutually exclusive on
    # Gmail's side, so a single one is enough to drop the message.
    excluded = _normalise_categories(f.get("exclude_categories") or [])
    if excluded and labels.intersection(excluded):
        return False, "excluded_category"

    return True, "ok"


# ── Helpers ──────────────────────────────────────────────────────────


def _flatten_headers(message: dict[str, Any]) -> dict[str, str]:
    """Return a lowercased name→value header dict from either of the
    two production shapes:

      A. Connector envelope — `message["headers"]` is already a flat
         dict with capitalised RFC 5322 names. Hot path for live
         trigger fires (gmail/provider.py:gmail__get_message).
      B. Raw Gmail REST response — `message["payload"]["headers"]` is
         a list of `{name, value}` dicts. Synthesised test events.

    Lowercase the keys so callers can use `headers["from"]` /
    `headers["subject"]` without per-call case-juggling.
    """
    out: dict[str, str] = {}
    raw = message.get("headers")
    if isinstance(raw, dict):
        for k, v in raw.items():
            if not k:
                continue
            lk = str(k).lower()
            if lk not in out:
                out[lk] = v or ""
        return out
    payload = message.get("payload") or {}
    for h in payload.get("headers") or []:
        n = (h.get("name") or "").lower()
        v = h.get("value") or ""
        if n and n not in out:
            out[n] = v
    return out


def _strip_list(raw: Any) -> list[str]:
    """Normalize a list-of-strings field. Drops blanks; accepts a
    bare string as a one-element list for forgiveness."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out
