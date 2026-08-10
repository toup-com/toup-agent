"""W4.2 / gate G2 preparation — metering_correctness_v2 (A9-3 / F-11).

The direct (manual-mode/BYOK) OpenAI metering path historically deduped
credit deductions on a per-SESSION constant idempotency key
(``{user_id}:{session_id}``), with a fallback to ``prompt_cache_key`` —
which PR-1 made DAY-scoped. Either way, every call after the first
idempotent-hits the ledger and is never metered.

Covers:
* the flag defaults OFF;
* flag OFF → the key is byte-identical to the legacy expression
  ``idempotency_key or prompt_cache_key`` across all input shapes;
* flag ON → per-request unique keys (completion id, UUID fallback);
* the collision scenario reproduced end-to-end through
  ``credit_service.try_charge``: two DISTINCT turns carrying the same
  day-scoped key → old path 1 deduction, new path 2;
* source pin: the report call in openai_agent_service routes through
  ``_metering_idempotency_key`` (no drive-by re-inline).

See docs/audits/2026-07-g2-billing-gate.md. The flag flip is gated on
written approval — these tests only prove the OFF path is unchanged and
the ON path is correct.
"""
from __future__ import annotations

import inspect
from decimal import Decimal

import pytest
import pytest_asyncio

from app.config import settings
from app.services.openai_agent_service import _metering_idempotency_key


pytestmark = pytest.mark.asyncio


# ──────────────────────────────────────────────────────────────────────
# Flag default + flag-off byte-identity
# ──────────────────────────────────────────────────────────────────────


def test_flag_defaults_on_under_the_g2_approval():
    """The G2 gate was written approval, and it was given on 2026-08-10
    (docs/audits/2026-07-g2-billing-gate.md; finish-run W-3).

    This assertion is kept — inverted, not deleted — because its job never
    was "OFF"; its job is that the value is a DELIBERATE one. A silent
    flip back to legacy per-session keys would restore the
    meter-once-per-conversation bug, and that must fail here rather than
    show up as a billing discrepancy weeks later.
    """
    from app.config import Settings
    assert Settings.model_fields["metering_correctness_v2"].default is True


@pytest.mark.parametrize(
    "idem, cache, cid",
    [
        # main chat path: per-session idem key + day-scoped cache key
        ("user-1:sess-abc", "user-1:day-42", "chatcmpl-xyz"),
        # fallback shape: no idem key → legacy falls back to the cache key
        (None, "user-1:day-42", "chatcmpl-xyz"),
        # neither key (no user/session context) → legacy sends None
        (None, None, "chatcmpl-xyz"),
        # no completion id captured either
        ("user-1:sess-abc", "user-1:day-42", None),
        (None, None, None),
        # empty-string idem key falls through `or` exactly like before
        ("", "user-1:day-42", None),
    ],
)
def test_flag_off_key_byte_identical_to_legacy(monkeypatch, idem, cache, cid):
    """Flag OFF: the deduped key must be EXACTLY the legacy expression
    ``idempotency_key or prompt_cache_key`` — same bytes, same None-ness —
    so the flag-off deduction path is unchanged."""
    monkeypatch.setattr(settings, "metering_correctness_v2", False, raising=False)
    legacy = idem or cache
    assert _metering_idempotency_key(idem, cache, cid) == legacy


def test_flag_off_ignores_completion_id(monkeypatch):
    """The completion id capture is new code — flag OFF must never let it
    leak into the billing key."""
    monkeypatch.setattr(settings, "metering_correctness_v2", False, raising=False)
    key = _metering_idempotency_key("u:sess", "u:day", "chatcmpl-should-not-appear")
    assert key == "u:sess"
    assert "chatcmpl" not in key


# ──────────────────────────────────────────────────────────────────────
# Flag-on: per-request uniqueness
# ──────────────────────────────────────────────────────────────────────


def test_flag_on_uses_completion_id(monkeypatch):
    monkeypatch.setattr(settings, "metering_correctness_v2", True, raising=False)
    key = _metering_idempotency_key("u:sess", "u:day", "chatcmpl-abc123")
    assert key == "oaireq:chatcmpl-abc123"
    # fits the credit_ledger.idempotency_key VARCHAR(120) with headroom
    assert len(key) <= 120


def test_flag_on_same_session_distinct_requests_get_distinct_keys(monkeypatch):
    """Two calls in the SAME session (same legacy constants) with distinct
    completion ids → distinct billing keys → both meter."""
    monkeypatch.setattr(settings, "metering_correctness_v2", True, raising=False)
    k1 = _metering_idempotency_key("u:sess", "u:day", "chatcmpl-turn1")
    k2 = _metering_idempotency_key("u:sess", "u:day", "chatcmpl-turn2")
    assert k1 != k2


def test_flag_on_uuid_fallback_is_unique_per_call(monkeypatch):
    """No completion id (defensive: proxy stripped it) → fresh UUID per
    call, never a shared constant."""
    monkeypatch.setattr(settings, "metering_correctness_v2", True, raising=False)
    k1 = _metering_idempotency_key("u:sess", "u:day", None)
    k2 = _metering_idempotency_key("u:sess", "u:day", None)
    assert k1 is not None and k2 is not None
    assert k1 != k2
    assert k1.startswith("oaireq:") and k2.startswith("oaireq:")
    assert len(k1) <= 120


def test_flag_on_same_completion_id_dedupes(monkeypatch):
    """Same completion id (a genuine replay of ONE OpenAI response) maps
    to the same key — replays stay deduped, only distinct requests meter."""
    monkeypatch.setattr(settings, "metering_correctness_v2", True, raising=False)
    k1 = _metering_idempotency_key(None, None, "chatcmpl-same")
    k2 = _metering_idempotency_key("u:sess", "u:day", "chatcmpl-same")
    assert k1 == k2 == "oaireq:chatcmpl-same"


# ──────────────────────────────────────────────────────────────────────
# Collision scenario end-to-end through try_charge
# ──────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def credit_user(test_user_id: str):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, test_user_id)
        await db.commit()
    return test_user_id


async def _charge(user_id: str, key, amount="1.0"):
    from app.db import async_session_maker
    from app.db.models import LEDGER_CHAT_MESSAGE
    from app.services.credit_service import credit_service, BUCKET_MESSAGE

    async with async_session_maker() as db:
        result = await credit_service.try_charge(
            db, user_id, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal(amount),
            idempotency_key=key, model="gpt-5.5", provider="openai",
            input_tokens=1000, output_tokens=100,
            metadata={"surface": "agent_direct", "operation_type": "user"},
        )
        await db.commit()
        return result


async def test_collision_old_path_collapses_two_turns_into_one_deduction(
    monkeypatch, credit_user,
):
    """THE BUG (A9-3/F-11): two DISTINCT turns whose reports carry the same
    day-scoped fallback key collapse into ONE deduction — the second is an
    idempotent hit and is never metered."""
    monkeypatch.setattr(settings, "metering_correctness_v2", False, raising=False)
    day_key = f"{credit_user}:day-chat-42"

    # Turn 1 and turn 2 — distinct real LLM requests, same legacy key.
    key1 = _metering_idempotency_key(None, day_key, "chatcmpl-turn1")
    key2 = _metering_idempotency_key(None, day_key, "chatcmpl-turn2")
    assert key1 == key2 == day_key  # the collision

    r1 = await _charge(credit_user, key1)
    r2 = await _charge(credit_user, key2)
    assert r1.success and not r1.idempotent_hit
    assert r2.idempotent_hit  # turn 2 never metered
    assert r2.balance_after == r1.balance_after  # balance moved ONCE


async def test_collision_new_path_meters_both_turns(monkeypatch, credit_user):
    """Flag ON: the same two turns carry per-request keys → two deductions."""
    monkeypatch.setattr(settings, "metering_correctness_v2", True, raising=False)
    day_key = f"{credit_user}:day-chat-42"

    key1 = _metering_idempotency_key(None, day_key, "chatcmpl-turn1")
    key2 = _metering_idempotency_key(None, day_key, "chatcmpl-turn2")
    assert key1 != key2

    r1 = await _charge(credit_user, key1)
    r2 = await _charge(credit_user, key2)
    assert r1.success and not r1.idempotent_hit
    assert r2.success and not r2.idempotent_hit
    assert r2.balance_after == r1.balance_after - Decimal("1.0")


# ──────────────────────────────────────────────────────────────────────
# Source pins — the wiring stays in place
# ──────────────────────────────────────────────────────────────────────


def test_report_call_routes_through_metering_key_helper():
    """create_message_stream's report_llm_usage call must derive its
    idempotency_key via _metering_idempotency_key — re-inlining the legacy
    expression would silently drop the flag."""
    from app.services.openai_agent_service import OpenAIAgentService
    src = inspect.getsource(OpenAIAgentService.create_message_stream)
    assert "_metering_idempotency_key(" in src
    # the raw legacy expression must not reappear as the passed value
    assert "idempotency_key=idempotency_key or prompt_cache_key" not in src


def test_helper_flag_off_return_is_the_legacy_expression():
    """The flag-off branch is pinned to the exact legacy expression so the
    default path stays byte-identical."""
    src = inspect.getsource(_metering_idempotency_key)
    assert "return idempotency_key or prompt_cache_key" in src
