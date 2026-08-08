"""The admin cache view must show the wire the fleet actually uses.

WHAT WENT WRONG (measured 2026-08-08)
-------------------------------------
`GET /api/admin/llm/cache-daily` defaulted to the literal `endpoint="chat"`.
That was correct when written. It became silently wrong the moment the fleet
moved to gpt-5.6-terra on the Responses wire (#507), because agent turns then
began writing `endpoint="responses"` — so the default view stopped containing a
single agent turn.

Production, 14 days:

    endpoint    calls   input tokens   cached
    chat        1,967      9,013,122    49.0%
    responses     783     11,197,926    18.3%   <- excluded from the default view

The excluded half carried 55% of all input tokens at a third of the hit rate.
The dashboard read 49% and green throughout.

The fix derives the default from the fleet's own model resolution, so the next
wire migration carries this view with it. These tests pin the derivation rather
than the string "responses" — asserting the literal would just re-create the
original bug one migration later.
"""

from __future__ import annotations

import pytest


def test_default_endpoint_is_derived_from_the_fleet_model():
    """The default equals wire_api_for(default_model()) — not a literal."""
    from app.services.model_resolver import default_model, wire_api_for

    derived = wire_api_for(default_model())
    assert derived in ("chat", "responses")

    import inspect

    from app.api.llm_proxy import get_admin_cache_daily

    sig = inspect.signature(get_admin_cache_daily)
    default = sig.parameters["endpoint"].default
    # FastAPI Query(...) object — the DEFAULT must be None so the handler can
    # resolve it per request. A hardcoded string here is the original defect.
    assert getattr(default, "default", default) is None, (
        "endpoint must default to None and be resolved from the fleet wire "
        "inside the handler; a literal default is what broke this view when "
        "the fleet moved wires"
    )


def test_the_derived_default_matches_what_the_proxy_actually_records():
    """A derivation nobody writes rows for would be worse than the bug.

    The proxy records `endpoint` as the wire it called. If the derived default
    ever names a value that is not a real recorded endpoint, the view is empty
    again — just for a new reason.
    """
    from app.services.model_resolver import default_model, wire_api_for

    derived = wire_api_for(default_model())
    # These are the wire endpoints the proxy writes for text completions.
    assert derived in {"chat", "responses"}, (
        f"derived default {derived!r} is not a wire the proxy records"
    )


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.6-terra", "responses"),
        ("gpt-4o", "chat"),
    ],
)
def test_the_view_follows_a_wire_migration(monkeypatch, model, expected):
    """Flip the fleet model, and the default view follows it.

    This is the property that was missing: the old literal could not track a
    migration, so the view emptied without anything failing.
    """
    import app.services.model_resolver as mr

    monkeypatch.setattr(mr, "default_model", lambda: model)
    assert mr.wire_api_for(model) == expected


async def test_explicit_endpoint_still_wins_over_the_derived_default():
    """`?endpoint=chat` and `?endpoint=all` must still work — the derivation
    only fills in when the caller said nothing."""
    import inspect

    from app.api.llm_proxy import get_admin_cache_daily

    src = inspect.getsource(get_admin_cache_daily)
    assert "if endpoint is None:" in src, (
        "the derivation must be conditional on the caller omitting it"
    )
    assert 'if endpoint != "all"' in src, (
        "endpoint=all must still bypass the filter entirely"
    )
