"""Lightweight, dependency-free rerank + filter for search results.

Pure Python (no network, no LLM, no heavy deps) so it adds negligible latency:
  1. drop useless results (no title or no url),
  2. dedup near-duplicate URLs (collapse http/https, www., trailing slash,
     fragment — query is kept so distinct pages survive),
  3. score each surviving result with a compact BM25 over title+snippet vs the
     query terms (title double-weighted) and order by score, ties stable.

Operates on any objects exposing ``.title``, ``.url``, ``.snippet`` so it does
not import the SearchResult type (avoids a circular import with search.py).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List
from urllib.parse import urlsplit

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 2]


def _norm_url(url: str):
    """Normalization key that collapses near-duplicate URLs (scheme, www.,
    trailing slash, fragment) while keeping the query so distinct pages differ."""
    try:
        p = urlsplit((url or "").strip())
        host = (p.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = p.path.rstrip("/") or "/"
        return (host, path, p.query)
    except Exception:
        return ((url or "").strip().lower(),)


def rerank_filter(results: list, query: str) -> list:
    """Dedup + drop-empty + BM25-rerank ``results`` for ``query``. Returns a new
    list; never raises (best-effort — falls back to the cleaned order)."""
    # 1) drop useless + 2) dedup near-duplicate URLs (first occurrence wins)
    seen = set()
    cleaned = []
    for r in results:
        title = (getattr(r, "title", "") or "").strip()
        url = (getattr(r, "url", "") or "").strip()
        if not title or not url:
            continue
        key = _norm_url(url)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(r)

    if len(cleaned) <= 1:
        return cleaned

    q_terms = _tokenize(query)
    if not q_terms:
        return cleaned

    # 3) compact BM25 over (title*2 + snippet) for the small result set
    docs = [_tokenize(f"{r.title} {r.title} {getattr(r, 'snippet', '')}") for r in cleaned]
    n_docs = len(docs)
    avgdl = (sum(len(d) for d in docs) / n_docs) or 1.0
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))

    def _score(doc: List[str]) -> float:
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = len(doc)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * (f * (_BM25_K1 + 1)) / (f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl))
        return s

    scores = [_score(d) for d in docs]
    # stable: higher score first, original order on ties
    order = sorted(range(n_docs), key=lambda i: (-scores[i], i))
    return [cleaned[i] for i in order]
