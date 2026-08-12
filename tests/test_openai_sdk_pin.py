"""The OpenAI SDK version CI runs must be the one production runs.

Red-proof was live CI itself: `openai>=2.14.0` resolved to the brand-new
3.0.0 in every fresh per-job venv on 2026-08-12, whose transport stack
(httpx2) bypassed the suite's httpx.AsyncHTTPTransport mocks — dummy
keys went to the real api.openai.com and every sweep failed on a 401
while the fleet quietly kept running 2.53.0. Same class as the
apscheduler fold-semantics split: CI was testing a library production
has never run.

If this test fails after a deliberate SDK upgrade: bump the pin in all
FIVE requirements files + both install lists in test-backend.yml, roll
the fleet, and verify the deployed images resolve the same version —
then update the range here.
"""

import openai


def test_openai_sdk_is_the_production_major_minor():
    parts = tuple(int(p) for p in openai.__version__.split(".")[:2])
    assert (2, 53) <= parts < (3, 0), (
        f"openai SDK {openai.__version__} is not the production series "
        "(fleet images run 2.53.x). An unpinned resolve broke every CI "
        "sweep on 2026-08-12 — see this file's docstring before changing."
    )
