"""Round 24: moving the rollout canary is a CHECKED operation, not raw SQL.

`users.is_canary` had no route, no script and one raw-SQL snippet in a
checklist — which is why the founder's own live account was still the canary
through the 2026-08-23 outage. The two eligibility traps both fail SILENTLY
(a pool-bound canary aborts every future rollout with no obvious cause; a
canary with no agent_url/agent_api_key WEAKENS the gate instead of failing),
so the route refuses them and says which.

Run: cd backend && pytest tests/test_r24_canary_route.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.admin.rollouts import _canary_blockers  # noqa: E402


def _container(**kw):
    base = dict(status="running", pin_image_tag=None, container_name="toup-agent-3134fece")
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_healthy_named_tenant_is_eligible():
    assert _canary_blockers(_container(), "https://agent-x.agents.toup.ai", "k") == []


def test_no_container_is_refused():
    why = _canary_blockers(None, "https://a", "k")
    assert why and "no managed container" in why[0]


def test_a_pool_bound_tenant_is_refused_by_name():
    # The silent killer: pool members are excluded from the rollout's candidate
    # set, so flagging one aborts every rollout with "no canary in the running
    # set" and nothing points back to this choice.
    why = _canary_blockers(
        _container(container_name="toup-agent-pool-07"), "https://a", "k")
    assert any("pool-bound" in w for w in why)
    assert any("no canary in the running set" in w for w in why)


def test_a_stopped_or_pinned_container_is_refused():
    assert any("not running" in w for w in _canary_blockers(
        _container(status="stopped"), "https://a", "k"))
    assert any("pinned" in w for w in _canary_blockers(
        _container(pin_image_tag="ghcr.io/toup-com/toup-agent:abc"), "https://a", "k"))


def test_a_canary_that_would_WEAKEN_the_gate_is_refused():
    # Both of these pass the rollout's own checks by doing nothing at all —
    # the whole reason they must be refused up front.
    assert any("agent_url" in w for w in _canary_blockers(_container(), None, "k"))
    assert any("agent_api_key" in w for w in _canary_blockers(_container(), "https://a", None))


def test_the_blockers_are_operator_words_not_internals():
    why = _canary_blockers(_container(container_name="toup-agent-pool-07"), None, None)
    joined = " ".join(why).lower()
    for banned in ("traceback", "sqlalchemy", "select(", "none)", "attributeerror"):
        assert banned not in joined


def test_the_move_is_one_transaction_and_refuses_a_live_rollout():
    # The partial unique index refuses two flagged rows, so clear-then-set must
    # share a transaction with a flush between; and moving the canary while a
    # rollout is in flight would change its subject mid-run.
    import inspect
    from app.api.admin import rollouts as R
    src = inspect.getsource(R.set_canary)
    i_active = src.index("active_rollout(db)")
    i_clear = src.index("u.is_canary = False")
    i_flush = src.index("await db.flush()")
    i_set = src.index("target.is_canary = True")
    i_commit = src.index("await db.commit()")
    assert i_active < i_clear < i_flush < i_set < i_commit
    assert "409" in src  # the in-flight refusal
