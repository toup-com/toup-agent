"""Log-safe descriptors for memory rows.

Memory content is the most sensitive text this system holds. It is not
incidental user chatter — it is, by construction, the set of durable facts a
person has told the agent about themselves, kept deliberately and forever.
Writing it to the application log copies it out of the tenant database (which
is per-user and access-controlled) into a stream that is aggregated, shipped,
and read by operators.

Until BUG-25 every memory write logged `content[:50]` at INFO, on eleven call
sites. It was found by a privacy test that had passed four consecutive runs
and then failed the fifth: whether a secret leaked depended on where the
extractor's phrasing happened to place it relative to the 50-character
truncation. `"The user's locker phrase is wolverine-plinth-6620"` is 49
characters, so that one leaked whole. The measurement that said zero was true
of the rows it looked at and false about the system.

Truncation is not redaction. It is a coin flip weighted by sentence length.

So: never the content. What an operator actually needs from these lines is to
tell rows apart and to follow one row across several lines, and a fingerprint
does both without carrying the text. It is a one-way hash — it correlates, and
it does not reverse.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

__all__ = ["memory_fingerprint", "describe_memory"]


def memory_fingerprint(content: Optional[str]) -> str:
    """A short, stable, one-way id for a piece of memory content.

    Same content -> same fingerprint, so two log lines about one row can be
    tied together. It cannot be reversed into the content, which is the
    entire point.
    """
    if not content:
        return "-"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]


def describe_memory(
    content: Optional[str] = None,
    *,
    action: Optional[str] = None,
    category: Optional[Any] = None,
    memory_id: Optional[Any] = None,
    extra: Optional[str] = None,
) -> str:
    """Everything a log line may say about a memory. Never the content.

    Shaped as `created cat=health len=48 fp=a3f19c2b` — greppable, stable, and
    carrying no fact about the user.
    """
    parts = []
    if action:
        parts.append(str(action))
    if memory_id:
        parts.append(f"id={str(memory_id)[:8]}")
    if category is not None:
        parts.append(f"cat={getattr(category, 'value', category)}")
    if extra:
        parts.append(extra)
    parts.append(f"len={len(content or '')}")
    parts.append(f"fp={memory_fingerprint(content)}")
    return " ".join(parts)
