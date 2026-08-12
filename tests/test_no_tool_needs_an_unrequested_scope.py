"""A shipped tool must be reachable with the scopes we actually request.

Two tools shipped that could not once have succeeded, both found on
2026-08-11 from a single user report:

  * `sheets__list_spreadsheets` needed `drive.readonly`
  * `calendar__check_availability` POSTed `/freeBusy`, which Google's
    reference accepts only for calendar, calendar.readonly,
    calendar.freebusy and calendar.events.freebusy

Both scopes live in `scopes_optional`, and `_build_authorize_url` sends
`oauth.scopes` ONLY — so neither has ever been granted to anyone, and
neither ever will be while the policy stands (`drive.readonly` is
restricted; requesting it means an annual CASA assessment).

The failure was invisible from inside: the manifests were valid, the
registry lint passed, every unit test passed, and the connectors showed
healthy. It only surfaced when a user asked for a sheet by name and the
agent told them to grant a permission the consent screen will never
offer — advice no user could follow, on a loop.

The generic guard is `test_no_provider_depends_on_an_optional_scope`.
The rest pin the two specific regressions.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import yaml

CONNECTORS = pathlib.Path("app/connectors")


def _manifest(name: str) -> dict:
    return yaml.safe_load((CONNECTORS / name / "manifest.yaml").read_text())


def _connector_dirs() -> list[str]:
    return sorted(
        p.name for p in CONNECTORS.iterdir()
        if (p / "manifest.yaml").exists() and (p / "provider.py").exists()
    )


def _code_strings(path: pathlib.Path) -> set[str]:
    """Every string literal in the module EXCEPT docstrings.

    Docstrings are excluded on purpose: the whole point of this fix was
    to write down *why* `drive.readonly` is unreachable, and a guard
    that forbids naming it in prose would forbid the explanation.
    Comments never reach the AST, so they're excluded for free.
    """
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return {
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
    }


@pytest.mark.parametrize("connector", _connector_dirs())
def test_no_provider_depends_on_an_optional_scope(connector):
    """Provider CODE may not reference a scope we never request.

    `scopes_optional` is documentation — `oauth.py` reads `oauth.scopes`
    and nothing else. So a scope string appearing in executable code is
    a capability gated on a grant that cannot arrive.
    """
    m = _manifest(connector)
    optional = set((m.get("oauth") or {}).get("scopes_optional") or [])
    if not optional:
        pytest.skip(f"{connector} declares no optional scopes")
    used = _code_strings(CONNECTORS / connector / "provider.py")
    leaked = sorted(optional & used)
    assert not leaked, (
        f"{connector}/provider.py depends on scope(s) that are never "
        f"requested: {leaked}. Either move them into oauth.scopes, or "
        f"remove the code path that needs them."
    )


@pytest.mark.parametrize("connector", _connector_dirs())
def test_health_probe_names_a_tool_that_exists(connector):
    """The manifest's `health.probe` is registry-lint only, but it still
    has to name a real tool — removing a tool that a probe pointed at
    would break boot for that connector."""
    m = _manifest(connector)
    probe = (m.get("health") or {}).get("probe")
    if not probe:
        pytest.skip(f"{connector} declares no health probe")
    names = {t["name"] for t in (m.get("tools") or [])}
    assert probe in names, (
        f"{connector} health.probe={probe!r} is not one of {sorted(names)}"
    )


def test_sheets_no_longer_offers_a_tool_it_cannot_run():
    m = _manifest("sheets")
    names = {t["name"] for t in m["tools"]}
    assert "sheets__list_spreadsheets" not in names, (
        "listing spreadsheets is a Drive capability needing drive.readonly, "
        "which is restricted and deliberately never requested"
    )
    # The four that work on `spreadsheets` alone must survive.
    assert names == {
        "sheets__read_range", "sheets__append_rows",
        "sheets__update_range", "sheets__create_spreadsheet",
    }


def test_calendar_availability_does_not_call_freebusy():
    """freebusy.query rejects `calendar.events`, the only scope this
    connector requests. Verified against Google's reference on
    2026-08-11: accepted scopes are calendar, calendar.readonly,
    calendar.freebusy, calendar.events.freebusy."""
    # Code literals only. The provider explains in a comment WHY freeBusy
    # is unusable, and a guard that banned the word would delete the
    # explanation — which is the part that stops someone re-adding it.
    used = _code_strings(CONNECTORS / "calendar" / "provider.py")
    assert not any("freebusy" in s.lower() for s in used), (
        "check_availability must derive busy blocks from events.list; "
        "freeBusy 403s for every user this connector can authorise"
    )


def test_calendar_availability_is_still_offered_and_derives_from_events():
    """The fix must keep the capability, not delete it — availability is
    the point of the tool, and events.list on `calendar.events` carries
    exactly the same information once recurrences are expanded."""
    m = _manifest("calendar")
    names = {t["name"] for t in m["tools"]}
    assert "calendar__check_availability" in names

    src = (CONNECTORS / "calendar" / "provider.py").read_text()
    block = src[src.index('tool_name == "calendar__check_availability"'):]
    block = block[:block.index('tool_name == "calendar__delete_event"')]
    assert "calendars/primary/events" in block
    assert '"singleEvents": "true"' in block, (
        "recurring events must be expanded or a weekly meeting reads as "
        "one busy block on its first occurrence only"
    )
    # Match the EXPRESSION, not the word. An earlier version asserted
    # `"transparent" in block` and survived its mutation, because the
    # comment above the check also contains the word.
    assert 'ev.get("transparency") == "transparent"' in block, (
        "events marked free ('transparency': 'transparent') do not block "
        "time and must not be reported busy"
    )


def test_calendar_availability_no_longer_accepts_a_calendars_argument():
    """Reading a non-primary calendar needs `calendar.readonly`. Keeping
    the argument would advertise a capability the connector cannot
    deliver — the same defect one layer up."""
    m = _manifest("calendar")
    tool = next(
        t for t in m["tools"] if t["name"] == "calendar__check_availability"
    )
    props = tool["input_schema"]["properties"]
    assert "calendars" not in props
    assert set(tool["input_schema"]["required"]) == {"time_min", "time_max"}
