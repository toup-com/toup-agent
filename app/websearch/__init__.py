"""Shared, dependency-free web-search policy used by BOTH the platform search
gateway (``app/api/search_proxy.py``, Railway) and the per-tenant agent
(``app/agent/tool_executor.py``, Contabo containers).

Why its own package and not ``app.agent.smart_fetch``: importing that package
runs ``smart_fetch/__init__.py`` → ``search.py`` → ``bs4``, and the platform
image is built from ``requirements.platform.txt`` which does not carry the
scraping stack. A gateway that crashed on import would take fleet-wide search
down. Everything here is stdlib only, on purpose.

Modules
  freshness  — recency-intent classifier, Brave param builder, page-age
               parsing, the 18-month stale filter and web/news merge.
  render     — the numbered result block the model reads (rank / title / URL /
               date / verbatim snippet), budget-aware so the tail is never cut.
  citations  — URL extraction + the citation-integrity gate.
"""
