"""Citation-integrity gate.

Incident (2026-08-18, turn 3): the model answered "most capable = Claude Opus
5" and cited anthropic.com URLs that appeared in NONE of the four searches it
ran that turn. Nothing checked. This module makes the check mechanical:

  * ``extract_urls`` — every http(s) URL in a piece of text (tool output, user
    message, model answer), markdown-aware, trailing punctuation trimmed.
  * ``allowed_from_tool_outputs`` — the grounded set for one turn.
  * ``gate`` — rewrite an answer so every URL it contains is either grounded
    or visibly marked as unverified (or stripped), and report the violations.

Grounded means: the canonical form of the URL appears in this turn's tool
outputs (or in the other sources the caller chose to trust — the user's own
message, prior conversation, the URLs the model itself passed to web_fetch).
One deliberate exemption: a bare origin (``https://anthropic.com/``) is allowed
when any URL from that host was grounded — pointing at a site's front door is
not the failure this gate exists for.

Stdlib only; used by the agent runner on every turn, so it must be cheap and
must never raise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

from .freshness import canonical_url

# A URL runs until whitespace or a character that cannot be part of one in
# prose/markdown. Trailing sentence punctuation is trimmed afterwards.
_URL = re.compile(r"https?://[^\s<>\"'`\]\}]+", re.IGNORECASE)
_MD_LINK = re.compile(r"(!?)\[([^\]]*)\]\((https?://[^)\s]+)([^)]*)\)", re.IGNORECASE)
_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>", re.IGNORECASE)
_TRAILING = ".,;:!?'\""

MODE_MARK = "mark"      # keep the URL as plain text, append "(unverified)"
MODE_STRIP = "strip"    # remove the URL, leave the label + "(unverified)"
MODES = (MODE_MARK, MODE_STRIP)


def _trim(url: str) -> str:
    u = url.rstrip(_TRAILING)
    # A trailing ')' belongs to prose unless the URL itself opened a paren
    # (Wikipedia-style ``/Foo_(bar)``).
    while u.endswith(")") and u.count("(") < u.count(")"):
        u = u[:-1]
    return u


def extract_urls(text: str) -> List[str]:
    """All http(s) URLs in ``text`` in order of appearance (raw, trimmed)."""
    if not text:
        return []
    out: List[str] = []
    for m in _URL.finditer(text):
        u = _trim(m.group(0))
        if u:
            out.append(u)
    return out


def canonical_set(urls: Iterable[str]) -> Set[str]:
    return {canonical_url(u) for u in urls if u}


def _host_of(url: str) -> str:
    try:
        h = (urlsplit(url).netloc or "").lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def _is_bare_origin(url: str) -> bool:
    try:
        p = urlsplit(url)
        return (p.path or "").strip("/") == "" and not p.query and not p.fragment
    except Exception:
        return False


@dataclass
class GateResult:
    text: str
    violations: List[str] = field(default_factory=list)   # raw URLs rewritten
    checked: int = 0                                       # URLs seen in the answer

    @property
    def clean(self) -> bool:
        return not self.violations


class CitationGate:
    """Holds the grounded set for one turn. Feed it tool outputs as they
    arrive; call ``apply`` on the final answer."""

    def __init__(self, *, trusted_text: Optional[Sequence[str]] = None) -> None:
        self._allowed: Set[str] = set()
        self._hosts: Set[str] = set()
        for t in trusted_text or ():
            self.add_text(t)

    # ── grounding ────────────────────────────────────────────────
    def add_text(self, text: str) -> int:
        """Add every URL found in ``text`` to the grounded set. Returns how
        many were added."""
        n = 0
        for u in extract_urls(text or ""):
            key = canonical_url(u)
            if key and key not in self._allowed:
                self._allowed.add(key)
                n += 1
            h = _host_of(u)
            if h:
                self._hosts.add(h)
        return n

    def add_url(self, url: str) -> None:
        if url:
            self.add_text(url)

    @property
    def size(self) -> int:
        return len(self._allowed)

    def is_grounded(self, url: str) -> bool:
        key = canonical_url(url)
        if key in self._allowed:
            return True
        # Front-door exemption: the host itself was seen this turn.
        if _is_bare_origin(url) and _host_of(url) in self._hosts:
            return True
        return False

    # ── enforcement ──────────────────────────────────────────────
    def apply(self, answer: str, *, mode: str = MODE_MARK) -> GateResult:
        """Rewrite ``answer`` so ungrounded URLs are marked/stripped.

        Markdown links ``[label](url)`` become ``label (unverified: url)`` in
        mark mode or ``label (unverified)`` in strip mode. Bare URLs and
        ``<autolinks>`` become ``url (unverified)`` / ``(unverified link
        removed)``. Grounded URLs are untouched byte-for-byte.
        """
        if not answer:
            return GateResult(answer or "")
        mode = mode if mode in MODES else MODE_MARK
        violations: List[str] = []
        checked = 0
        protected: List[Tuple[int, int]] = []   # spans already rewritten/kept

        def _md(m: "re.Match[str]") -> str:
            nonlocal checked
            bang, label, url, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            checked += 1
            if self.is_grounded(_trim(url)):
                return m.group(0)
            violations.append(_trim(url))
            label = label.strip() or _trim(url)
            if mode == MODE_STRIP:
                return f"{label} (unverified)"
            return f"{label} (unverified: {_trim(url)})"

        text = _MD_LINK.sub(_md, answer)

        def _auto(m: "re.Match[str]") -> str:
            nonlocal checked
            url = _trim(m.group(1))
            checked += 1
            if self.is_grounded(url):
                return m.group(0)
            violations.append(url)
            return "(unverified link removed)" if mode == MODE_STRIP else f"{url} (unverified)"

        text = _AUTOLINK.sub(_auto, text)

        # Bare URLs: everything left that still looks like a URL and is not
        # already inside a rewritten "(unverified: …)" or a grounded markdown
        # link. Walk matches right-to-left so offsets stay valid.
        md_spans = [(m.start(), m.end()) for m in _MD_LINK.finditer(text)]
        unverified_spans = [(m.start(), m.end()) for m in re.finditer(r"\(unverified: [^)]*\)|https?://\S+ \(unverified\)", text)]
        keep = md_spans + unverified_spans

        def _inside(a: int, b: int) -> bool:
            return any(s <= a and b <= e for s, e in keep)

        pieces: List[str] = []
        last = 0
        for m in _URL.finditer(text):
            s = m.start()
            raw = _trim(m.group(0))
            e = s + len(raw)
            if _inside(s, e):
                continue
            checked += 1
            if self.is_grounded(raw):
                continue
            violations.append(raw)
            pieces.append(text[last:s])
            pieces.append("(unverified link removed)" if mode == MODE_STRIP else f"{raw} (unverified)")
            last = e
        pieces.append(text[last:])
        text = "".join(pieces)
        return GateResult(text=text, violations=violations, checked=checked)
