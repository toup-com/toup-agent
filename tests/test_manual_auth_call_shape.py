"""GA run R-5: three routes called ``get_current_user(request, db)`` by hand.

The signature is ``get_current_user(request, credentials, db)`` — the manual
two-arg call binds the **AsyncSession** to ``credentials``, so
``credentials.credentials`` raises AttributeError before any auth strategy
runs. Every such call site fails for every caller:

- ``/api/admin/llm/stats`` (llm_proxy.py): the blanket ``except`` turned the
  AttributeError into 403 "Admin only" — the endpoint has rejected every
  admin since it shipped. Found live on 2026-08-10: a correctly-signed
  admin JWT for a role=admin user got 403.
- ``apps_proxy.py`` preview/bridge routes (x2): the Bearer path silently
  never authenticated; only the query-param/cookie fallbacks carried it.

Red-first: this file fails on the pre-fix tree with all three sites listed.
"""
from __future__ import annotations

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"

_BAD = re.compile(r"get_current_user\(\s*request\s*,\s*db\s*\)")


def test_no_manual_two_arg_get_current_user_calls():
    hits = []
    for py in APP.rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if _BAD.search(line):
                hits.append(f"{py.relative_to(APP.parent)}:{i}")
    assert not hits, (
        "get_current_user(request, db) binds the SESSION to the credentials "
        "parameter — AttributeError before any auth runs. Call "
        "get_current_user(request, None, db) or use Depends. Sites: "
        + ", ".join(hits)
    )
