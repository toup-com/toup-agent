"""The re-rank hop is bounded: budget expiry degrades to hybrid order.

Pre-fix, `RerankerService.rerank` awaited its backend with only the 10s httpx
client timeout between a slow hop and the turn path — G-18 measured retrieval
p95 at 2535-3440 ms against memverify's 1500 ms ceiling, entirely in the
external re-rank call. The fix gives BOTH backends one shared wall-clock
budget (`settings.reranker_timeout_ms`); on expiry the search returns the
caller's hybrid (RRF+weighted) order — degraded precision, never a slow or
failed search.

The three budget tests fail on the pre-fix service (the slow-backend stubs
sleep past the budget and pre-fix code waits them out); the two
unchanged-behavior controls pass on both — verified by running this file
against the parent commit before the fix landed.
"""

import asyncio
import time

from app.config import settings
from app.services.reranker_service import RerankerService


def _cands(n: int):
    # Descending final_score == the hybrid order the fallback must preserve.
    return [
        {"id": str(i), "content": f"doc {i}", "final_score": round(1.0 - i * 0.01, 4)}
        for i in range(n)
    ]


async def test_slow_cohere_degrades_to_hybrid_order_within_budget(monkeypatch):
    monkeypatch.setattr(settings, "reranker_timeout_ms", 300, raising=False)
    svc = RerankerService(cohere_api_key="test-key")

    async def _slow(*a, **k):
        await asyncio.sleep(3.0)
        return []

    monkeypatch.setattr(svc, "_cohere_rerank", _slow)

    t0 = time.monotonic()
    out = await svc.rerank("query", _cands(30), top_k=10)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.5, (
        f"rerank blocked the turn for {elapsed:.2f}s — the budget did not "
        "bound the slow backend"
    )
    assert [c["id"] for c in out] == [str(i) for i in range(10)], (
        "fallback must return the caller's hybrid order, truncated to top_k"
    )
    assert all(c["rerank_score"] == c["final_score"] for c in out)


async def test_budget_is_shared_across_both_backends(monkeypatch):
    """A slow Cohere attempt must eat the LLM fallback's time too.

    Pre-fix there was no budget at all; a naive per-backend timeout would
    still let the two hops stack (2 x budget). The shared deadline caps the
    TOTAL, so with Cohere burning the whole budget the LLM fallback is
    skipped and the search still returns promptly in hybrid order.
    """
    monkeypatch.setattr(settings, "reranker_timeout_ms", 300, raising=False)
    svc = RerankerService(cohere_api_key="test-key", openai_api_key="test-key")
    llm_called = {"n": 0}

    async def _slow_cohere(*a, **k):
        await asyncio.sleep(3.0)
        return []

    async def _slow_llm(*a, **k):
        llm_called["n"] += 1
        await asyncio.sleep(3.0)
        return []

    monkeypatch.setattr(svc, "_cohere_rerank", _slow_cohere)
    monkeypatch.setattr(svc, "_llm_rerank", _slow_llm)

    t0 = time.monotonic()
    out = await svc.rerank("query", _cands(30), top_k=10)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.5, f"backends stacked to {elapsed:.2f}s — budget not shared"
    assert llm_called["n"] == 0, (
        "Cohere consumed the whole budget; the LLM fallback must be skipped, "
        "not started with a fresh clock"
    )
    assert [c["id"] for c in out] == [str(i) for i in range(10)]


async def test_fast_backend_result_is_used_unchanged(monkeypatch):
    """The budget must not degrade a backend that answers in time."""
    svc = RerankerService(cohere_api_key="test-key")
    cands = _cands(30)
    reranked = list(reversed(cands[:10]))

    async def _fast(*a, **k):
        return reranked

    monkeypatch.setattr(svc, "_cohere_rerank", _fast)
    out = await svc.rerank("query", cands, top_k=10)
    assert out is reranked


async def test_timeout_logs_degraded_line_without_content(monkeypatch, caplog):
    """The degrade is observable (counts only — no memory content, no query)."""
    monkeypatch.setattr(settings, "reranker_timeout_ms", 300, raising=False)
    svc = RerankerService(cohere_api_key="test-key")

    async def _slow(*a, **k):
        await asyncio.sleep(3.0)
        return []

    monkeypatch.setattr(svc, "_cohere_rerank", _slow)
    with caplog.at_level("WARNING", logger="app.services.reranker_service"):
        await svc.rerank("a-secret-query-string", _cands(30), top_k=10)

    degraded = [r for r in caplog.records if "degraded" in r.getMessage()]
    assert degraded, "timeout must emit a [RERANKER] degraded log line"
    msg = degraded[0].getMessage()
    assert "reason=timeout" in msg and "candidates=30" in msg
    assert "a-secret-query-string" not in msg, "query text must never be logged"


async def test_small_candidate_set_short_circuits_without_budget(monkeypatch):
    """candidates <= top_k never touches a backend — unchanged behavior."""
    svc = RerankerService(cohere_api_key="test-key")

    async def _explodes(*a, **k):
        raise AssertionError("backend must not be called")

    monkeypatch.setattr(svc, "_cohere_rerank", _explodes)
    out = await svc.rerank("query", _cands(5), top_k=10)
    assert len(out) == 5
    assert all(c["rerank_score"] == c["final_score"] for c in out)
