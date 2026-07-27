"""Unit tests for the A6-2 extraction retry wrapper.

Per-turn background fact extraction is a single un-fallbacked LLM call —
before PR-4 any transient provider/network failure silently lost every
fact stated that turn. The wrapper adds exactly ONE retry with a short
backoff and reports the outcome (ok / retried / failed) so the
[memory_health] log line can surface it.
"""

from __future__ import annotations

import pytest

from app.services import memory_extractor as me
from app.services.memory_extractor import MemoryExtractor, _complete_json_with_retry


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """complete_with_json that fails the first `failures` calls."""

    def __init__(self, failures: int = 0, content: str = '{"memories": []}'):
        self.calls = 0
        self.failures = failures
        self._content = content
        self.models: list = []  # model kwarg per call (W1.4a pin)

    async def complete_with_json(self, messages, temperature, max_tokens, model=None):
        self.calls += 1
        self.models.append(model)
        if self.calls <= self.failures:
            raise RuntimeError("transient boom")
        return _FakeResponse(self._content)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Zero the retry backoff so tests don't sleep."""
    monkeypatch.setattr(me, "_EXTRACTION_RETRY_BACKOFF_S", 0)


async def test_success_first_try_no_retry():
    llm = _FakeLLM(failures=0)
    response, retried = await _complete_json_with_retry(
        llm, messages=[{"role": "user", "content": "x"}],
        temperature=0.3, max_tokens=100,
    )
    assert retried is False
    assert llm.calls == 1
    assert response.content == '{"memories": []}'


async def test_single_retry_recovers_transient_failure():
    llm = _FakeLLM(failures=1)
    response, retried = await _complete_json_with_retry(
        llm, messages=[{"role": "user", "content": "x"}],
        temperature=0.3, max_tokens=100,
    )
    assert retried is True
    assert llm.calls == 2
    assert response.content == '{"memories": []}'


async def test_second_failure_propagates_unchanged():
    """Exactly ONE retry — persistent failures (bad key) fail fast so the
    caller's no-regex-fallback policy stays intact."""
    llm = _FakeLLM(failures=2)
    with pytest.raises(RuntimeError, match="transient boom"):
        await _complete_json_with_retry(
            llm, messages=[{"role": "user", "content": "x"}],
            temperature=0.3, max_tokens=100,
        )
    assert llm.calls == 2


# ── outcome reporting through extract_memories_with_llm ─────────────


def _patch_llm(monkeypatch, llm: _FakeLLM):
    import app.services.llm_service as llm_service

    monkeypatch.setattr(llm_service, "get_llm_service", lambda: llm)


async def test_outcome_ok(monkeypatch):
    llm = _FakeLLM(failures=0)
    _patch_llm(monkeypatch, llm)
    extractor = MemoryExtractor()
    result = await extractor.extract_memories_with_llm(
        user_message="hi", assistant_response="hello",
    )
    assert result == []
    assert extractor.last_extraction_outcome == "ok"


async def test_outcome_retried(monkeypatch):
    llm = _FakeLLM(failures=1)
    _patch_llm(monkeypatch, llm)
    extractor = MemoryExtractor()
    result = await extractor.extract_memories_with_llm(
        user_message="hi", assistant_response="hello",
    )
    assert result == []
    assert llm.calls == 2
    assert extractor.last_extraction_outcome == "retried"


async def test_outcome_failed_returns_empty_not_regex(monkeypatch):
    """Both attempts failing must keep the 2026-05-10 hardening: return
    NOTHING (no regex fallback), and report the failure outcome."""
    llm = _FakeLLM(failures=2)
    _patch_llm(monkeypatch, llm)
    extractor = MemoryExtractor()
    result = await extractor.extract_memories_with_llm(
        user_message="my sister is called Ana", assistant_response="noted",
    )
    assert result == []
    assert extractor.last_extraction_outcome == "failed"


async def test_extracted_memories_survive_retry(monkeypatch):
    """A real memory payload parses after the retry path."""
    content = (
        '{"memories": [{"content": "User\'s sister is called Ana and lives '
        'in Lisbon", "summary": "Sister Ana", "category": "family", '
        '"memory_type": "fact", "importance": 0.8}]}'
    )
    llm = _FakeLLM(failures=1, content=content)
    _patch_llm(monkeypatch, llm)
    extractor = MemoryExtractor()
    result = await extractor.extract_memories_with_llm(
        user_message="my sister Ana lives in Lisbon", assistant_response="noted",
    )
    assert len(result) == 1
    assert "Ana" in result[0].content
    assert extractor.last_extraction_outcome == "retried"
