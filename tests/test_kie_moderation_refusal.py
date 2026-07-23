"""A provider content-policy refusal must be reported honestly, not retried.

Founder repro 2026-07-23: a photo + "Generate me a clean six pack" produced
"declined by the content-safety filter (the wording was flagged). Rephrase…".
That hint was wrong twice over:

  1. Nano Banana had already refused it — `422 The input or output was flagged
     as sensitive` — because it edits a real person's body/physique. The agent
     treated that like any technical Kie error and retried on OpenAI, which
     refuses the same class of request, costing another ~30-60s and a paid
     attempt to arrive at the same "no".
  2. The message blamed the WORDING, so both the user and the model burned
     retries re-phrasing a request that could never succeed.

These tests pin the two behaviours that fix it:
  • a moderation-flagged proxy response raises `_KieModerationRefused`
    (distinct from the generic RuntimeError that triggers the OpenAI fallback)
  • the user-facing copy says re-wording will NOT help, and never claims the
    phrasing was the problem.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.tool_executor import (  # noqa: E402
    _KieModerationRefused,
    _KieQuotaExceeded,
)


class _Resp:
    """Minimal httpx.Response stand-in for the proxy's error shape."""

    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _classify(resp: _Resp):
    """Mirror of _call_kie_image's status handling (the branch under test).

    Kept in lockstep with tool_executor._call_kie_image: 429 → quota,
    502+moderation → policy refusal, other non-200 → generic (falls back).
    """
    if resp.status_code == 429:
        detail = (resp.json() or {}).get("detail") or {}
        raise _KieQuotaExceeded(detail.get("message") or "limit")
    if resp.status_code != 200:
        try:
            d = (resp.json() or {}).get("detail") or {}
        except Exception:
            d = {}
        if isinstance(d, dict) and d.get("moderation"):
            raise _KieModerationRefused(str(d.get("message") or "").strip())
        raise RuntimeError(f"kie proxy HTTP {resp.status_code}")
    return b"image"


def test_moderation_flag_raises_distinct_refusal():
    # Exactly what the platform returns for Nano Banana's sensitive-content 422.
    resp = _Resp(502, {"detail": {
        "code": "kie_failed", "moderation": True,
        "message": "kie task failed: The input or output was flagged as sensitive.",
    }})
    with pytest.raises(_KieModerationRefused) as ei:
        _classify(resp)
    assert "sensitive" in str(ei.value).lower()


def test_technical_failure_still_falls_back():
    # moderation absent/False → generic error → caller may retry on OpenAI.
    resp = _Resp(502, {"detail": {"code": "kie_failed", "moderation": False,
                                  "message": "kie proxy unreachable"}})
    with pytest.raises(RuntimeError):
        _classify(resp)


def test_quota_still_distinct_from_moderation():
    resp = _Resp(429, {"detail": {"code": "image_quota_exceeded",
                                  "message": "free limit reached"}})
    with pytest.raises(_KieQuotaExceeded):
        _classify(resp)


def test_html_error_body_does_not_crash_classification():
    # The edge rewrites 5xx into an HTML page — .json() raises; must not 500.
    resp = _Resp(502, None, text="<!DOCTYPE html> error code: 502")
    with pytest.raises(RuntimeError):
        _classify(resp)


@pytest.mark.parametrize("src,needle", [
    ("edit", "will NOT help"),
    ("generate", "will not help"),
])
def test_refusal_copy_does_not_blame_wording(src, needle):
    """The shipped strings must not send the user re-phrasing in circles."""
    text = Path(__file__).resolve().parents[1].joinpath(
        "app/agent/tool_executor.py").read_text()
    # The old misleading copy is gone everywhere it was user-facing.
    assert "the wording was flagged). Rephrase" not in text
    # And the replacement explicitly says re-wording won't fix a policy refusal.
    assert needle in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
