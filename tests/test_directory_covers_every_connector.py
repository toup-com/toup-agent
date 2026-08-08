"""Every loadable connector needs a directory row keyed by `backendId`.

The frontend builds `DIRECTORY_BY_BACKEND_ID` by filtering on that field.
A connector whose row omits it matches nothing, so the live connector
renders unbranded — WHILE any same-named coming-soon placeholder keeps
rendering its own permanently-disabled card. Two tiles, and the
connectable one is the plain one.

This has now happened twice (Google Sheets, then Microsoft Teams) from
the identical cause, which is why it is a guard rather than a third
one-off fix. Nothing else in the repo compares the two sides: the
manifests are Python-loaded YAML and the directory is a TypeScript
literal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services.connector_registry import get_registry, reset_registry_for_tests

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "frontend/src/components/integrations/connectorDirectory.ts"
)


def _backend_ids_in_directory() -> set[str]:
    src = DIRECTORY.read_text()
    return set(re.findall(r"backendId:\s*'([^']+)'", src))


@pytest.fixture(scope="module")
def loaded_ids() -> set[str]:
    reset_registry_for_tests()
    reg = get_registry()
    reg.load_all(include_experimental=True)
    return {e.manifest.id for e in reg.all_entries()} if hasattr(reg, "all_entries") else set(
        getattr(reg, "_entries", {}).keys()
    )


@pytest.mark.skipif(not DIRECTORY.exists(), reason="frontend not present in this checkout")
def test_every_connector_has_a_directory_row(loaded_ids):
    # `stub` is test-only and deliberately has no tile.
    expected = loaded_ids - {"stub"}
    missing = sorted(expected - _backend_ids_in_directory())
    assert not missing, (
        f"connectors with no directory row: {missing}. Each renders unbranded, and any "
        f"coming-soon placeholder of the same product still renders its disabled card "
        f"alongside it."
    )


@pytest.mark.skipif(not DIRECTORY.exists(), reason="frontend not present in this checkout")
def test_no_backend_id_is_claimed_twice(loaded_ids):
    src = DIRECTORY.read_text()
    ids = re.findall(r"backendId:\s*'([^']+)'", src)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"two directory rows claim the same backendId: {dupes}"
