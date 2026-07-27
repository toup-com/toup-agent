"""Unit W1.9 — system-caller model hygiene.

Covers:
  (a) support_agent_model config default pins gpt-4o-mini (not None →
      platform chat model) so flipping support_agent_enabled on doesn't
      route 4+ calls/ticket to gpt-5.5.
  (b) reranker LLM (gpt-4o-mini) fallback caps its scoring input at
      LLM_RERANK_MAX_DOCS, slicing highest-fused-score first.
"""

import json
from types import SimpleNamespace

import pytest

from app.services.reranker_service import LLM_RERANK_MAX_DOCS, RerankerService


# ── (a) Support agent model default ──────────────────────────────────


def test_support_agent_model_default_is_gpt_4o_mini():
    """Config-class default (env-independent) must pin gpt-4o-mini."""
    from app.config import Settings

    assert Settings.model_fields["support_agent_model"].default == "gpt-4o-mini"


def test_support_model_helper_respects_setting(monkeypatch):
    from app.config import settings
    from app.support.llm import _support_model

    monkeypatch.setattr(settings, "support_agent_model", "gpt-4o-mini")
    assert _support_model() == "gpt-4o-mini"

    # Empty string / None ⇒ platform default (call_system_llm resolves it).
    monkeypatch.setattr(settings, "support_agent_model", "")
    assert _support_model() is None
    monkeypatch.setattr(settings, "support_agent_model", None)
    assert _support_model() is None


# ── (b) LLM-fallback rerank input cap ────────────────────────────────


class _FakeOpenAIClient:
    """Captures the chat.completions.create kwargs and returns ranked scores."""

    def __init__(self, capture: dict):
        create = self._create
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
        self._capture = capture

    async def _create(self, **kwargs):
        self._capture.update(kwargs)
        prompt = kwargs["messages"][0]["content"]
        n_docs = prompt.count("\n[")  # numbered "[i] (cat) content" lines
        scores = [{"index": i, "score": 10.0 - i * 0.1} for i in range(n_docs)]
        msg = SimpleNamespace(content=json.dumps({"scores": scores}))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _make_candidates(n: int):
    # final_score ascending: doc-0 is worst-fused, doc-{n-1} is best.
    return [
        {"id": f"m{i}", "content": f"doc-{i}", "category": "fact", "final_score": float(i)}
        for i in range(n)
    ]


@pytest.fixture
def fake_openai(monkeypatch):
    capture: dict = {}
    import app.services.bundle_client as bundle_client

    monkeypatch.setattr(
        bundle_client,
        "make_openai_client",
        lambda byok_key=None: _FakeOpenAIClient(capture),
    )
    return capture


async def test_llm_rerank_caps_input_at_max_docs(fake_openai):
    svc = RerankerService(openai_api_key="sk-test-not-a-real-key")
    candidates = _make_candidates(60)

    result = await svc._llm_rerank("query", candidates, top_k=15)

    prompt = fake_openai["messages"][0]["content"]
    # Exactly LLM_RERANK_MAX_DOCS numbered document lines in the prompt.
    n_doc_lines = prompt.count("\n[")
    assert n_doc_lines == LLM_RERANK_MAX_DOCS
    # Highest-fused-score docs survive the slice; lowest are dropped.
    assert "doc-59" in prompt
    assert "doc-30" in prompt
    assert "doc-29\n" not in prompt
    assert "doc-0\n" not in prompt
    # Output still honours top_k and only contains sliced candidates.
    assert len(result) == 15
    surviving = {f"doc-{i}" for i in range(30, 60)}
    assert all(r["content"] in surviving for r in result)
    assert all("rerank_score" in r for r in result)


async def test_llm_rerank_under_cap_is_untouched(fake_openai):
    svc = RerankerService(openai_api_key="sk-test-not-a-real-key")
    candidates = _make_candidates(20)

    result = await svc._llm_rerank("query", candidates, top_k=15)

    prompt = fake_openai["messages"][0]["content"]
    assert prompt.count("\n[") == 20
    assert "doc-0" in prompt  # nothing sliced below the cap
    assert len(result) == 15
