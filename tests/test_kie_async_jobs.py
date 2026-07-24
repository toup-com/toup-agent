"""Kie image jobs run start+poll so a slow render is collected, not abandoned.

Founder repro 2026-07-23 (20:08): an edit ran 200s, the `edit_image` tool
wrapper killed it, and the user got nothing — while Kie's own dashboard still
showed that task `running` with **18 credits consumed**. They paid for an image
that was thrown away. Kie's measured latency in that account: 25s, 36s, 39s,
41s, 74s … and a **399s** success. No single synchronous budget covers that:
too short abandons paid work, too long outlives the HTTP hop.

These pin the invariants of the start+poll design.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402


def test_tool_timeout_outlasts_the_job_deadline():
    """The wrapper must not kill a job the poller would still have collected."""
    job = float(getattr(settings, "kie_job_timeout_s"))
    for tool in ("generate_image", "edit_image"):
        wrapper = float(settings.tool_timeout_overrides[tool])
        assert wrapper > job, (
            f"{tool} wrapper timeout {wrapper}s must exceed kie_job_timeout_s "
            f"{job}s, else the tool is cut off while the render is still live "
            "(the 200s-vs-399s bug)"
        )


def test_job_deadline_covers_the_observed_slow_tail():
    # 399s was a real success in production logs; the budget must clear it.
    assert float(getattr(settings, "kie_job_timeout_s")) >= 400.0


def test_poll_task_shapes(monkeypatch):
    """poll_task maps Kie's states onto the contract the proxy relies on."""
    import asyncio
    import json as _json
    from app.services import kie_client

    class _R:
        status_code = 200

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    class _Client:
        def __init__(self, payload):
            self._p = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _R(self._p)

    def _run(payload):
        monkeypatch.setattr(kie_client.httpx, "AsyncClient",
                            lambda *a, **k: _Client(payload))
        monkeypatch.setattr(kie_client.settings, "kie_api_key", "test-key")
        return asyncio.run(kie_client.poll_task("t1"))

    # pending
    assert _run({"data": {"state": "waiting"}})["state"] == "pending"

    # success → carries the result url + credits
    ok = _run({"data": {"state": "success", "creditsConsumed": 18,
                        "resultJson": _json.dumps({"resultUrls": ["https://x/y.png"]})}})
    assert ok["state"] == "success" and ok["result_url"] == "https://x/y.png"
    assert ok["credits"] == 18.0

    # fail → surfaces the message AND flags a content refusal distinctly, so the
    # caller can skip a pointless OpenAI retry (see test_kie_moderation_refusal).
    bad = _run({"data": {"state": "fail",
                         "failMsg": "The input or output was flagged as sensitive."}})
    assert bad["state"] == "fail" and bad["moderation"] is True

    tech = _run({"data": {"state": "fail", "failMsg": "upstream connection reset"}})
    assert tech["state"] == "fail" and tech["moderation"] is False


def test_charge_is_idempotent_per_task():
    """Repeat polls after a dropped response must never double-bill.

    The poll endpoint keys try_charge on the Kie taskId; assert the key shape is
    stable and task-derived (not random per request).
    """
    src = Path(__file__).resolve().parents[1].joinpath(
        "app/api/llm_proxy.py").read_text()
    assert 'charge_key = f"kie_task:{task_id}"' in src
    assert "idempotency_key=charge_key" in src


def test_start_releases_the_reserved_slot_when_nothing_started():
    """A job that never launched must not eat a free-tier image slot."""
    src = Path(__file__).resolve().parents[1].joinpath(
        "app/api/llm_proxy.py").read_text()
    start = src.split("async def proxy_kie_image_start", 1)[1].split(
        "async def proxy_kie_image_poll", 1)[0]
    # every failure path inside start frees the reservation
    assert start.count("release_free_image_slot") >= 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
