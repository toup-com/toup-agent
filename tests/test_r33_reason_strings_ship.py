"""The reason-string table has to EXIST in the agent image (round 33, item 4).

`account_health._table_path()` used to resolve four directories above
`backend/app/agent/automations` — the repo root when you run from a
checkout, and `/` inside the container, because the agent image is built
with `context: ./backend` (.github/workflows/build-agent.yml) and
`fixtures/` lives at the repo root. So `strings()` returned `{}` in
production and every connector card said "I could not read {X}." with a
"Try again" button, whatever had actually gone wrong.

Two assertions, and the second is the one that would have caught it:
the table loads, and it loads from inside the PACKAGE — not from a path
that only exists in a checkout.
"""

import json
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent


def test_the_table_loads_and_is_not_empty():
    from app.agent.automations import account_health
    account_health._TABLE = None
    table = account_health.strings()
    assert table, "the reason-string table is empty — every card falls back"
    for key in ("states", "forms", "reason_codes", "fix_buttons"):
        assert key in table, key


def test_the_table_ships_inside_the_package():
    """The path must be under `backend/app/`, i.e. inside the image's
    build context. A path outside it loads in CI and not in production,
    which is exactly how this shipped."""
    from app.agent.automations import account_health
    path = Path(account_health._table_path()).resolve()
    assert path.exists(), path
    assert _BACKEND / "app" in path.parents, (
        f"{path} is outside backend/app — it will not exist in the agent image"
    )


def test_the_contract_document_and_the_shipped_copy_are_identical():
    """`fixtures/automations/reason-strings.json` stays the contract
    document both repos carry; the package copy is what runs. One list,
    two files — so drift fails here rather than in production."""
    from app.agent.automations import account_health
    shipped = json.loads(Path(account_health._table_path()).read_text())
    contract_path = _REPO / "fixtures" / "automations" / "reason-strings.json"
    if not contract_path.exists():          # not checked out (image build)
        return
    assert shipped == json.loads(contract_path.read_text()), (
        "the shipped table and fixtures/automations/reason-strings.json "
        "have drifted — copy the fixture over the package file"
    )
