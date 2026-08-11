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


# ----------------------------------------------------------------------
# Brand marks: a three-tier fallback whose first tier fails SILENTLY
#
# `BrandLogo` tried Iconify, then simple-icons, then a letter mark. But
# `<IconifyIcon>` renders an empty placeholder for a name the API does not
# carry — no throw, no console warning, nothing a caller can catch — and
# the component returned it unconditionally. So a wrong Iconify name did
# not fall through to the next tier; it rendered a BLANK TILE, and the two
# tiers below were unreachable.
#
# Three names were wrong: `logos:google-docs`, `logos:google-sheets` and
# `logos:microsoft-outlook`. The `logos` set carries no Docs, Sheets or
# Outlook icon at all — verified against its complete 1,861-icon index on
# 2026-08-10. All three had a perfectly good bundled simple-icons mark one
# layer below, and all three shipped blank.
#
# The runtime fix makes the chain real (`loadIcon` rejection → fall
# through). These guard the data: an entry that opts out of Iconify must
# have somewhere to fall TO.
# ----------------------------------------------------------------------

BRAND_LOGO = (
    Path(__file__).resolve().parents[2]
    / "frontend/src/components/integrations/BrandLogo.tsx"
)


def _entries() -> list[dict]:
    """Every directory row as {id, iconifyName, slug}, parsed from the TS."""
    src = DIRECTORY.read_text()
    rows = []
    for m in re.finditer(r"\bid:\s*'([a-z0-9_]+)'", src):
        # The record runs to the next `id:` or the end of the array literal.
        start = m.start()
        nxt = src.find("id: '", m.end())
        chunk = src[start: nxt if nxt != -1 else start + 600]
        icon = re.search(r"iconifyName:\s*(?:'([^']+)'|(null))", chunk)
        slug = re.search(r"slug:\s*(?:'([^']+)'|(null))", chunk)
        rows.append({
            "id": m.group(1),
            "iconify": icon.group(1) if icon and icon.group(1) else None,
            "slug": slug.group(1) if slug and slug.group(1) else None,
        })
    return rows


@pytest.mark.skipif(not DIRECTORY.exists(), reason="frontend not present in this checkout")
def test_no_connector_tile_can_render_blank():
    """Every row must reach a mark: an Iconify name, or a slug with a
    bundled simple-icons path, or (deliberately) a letter mark.

    A row with neither an Iconify name nor a slug is fine — that is an
    explicit letter mark. What must never happen is a slug that names an
    import BrandLogo does not have, because the middle tier then produces
    an empty <svg> and the letter mark is never reached either.
    """
    assert BRAND_LOGO.exists(), BRAND_LOGO
    brand_src = BRAND_LOGO.read_text()
    bundled = set(re.findall(r"import\s+(\w+)Svg\s+from\s+'simple-icons/icons/([a-z0-9]+)\.svg",
                             brand_src))
    bundled_slugs = {slug for _, slug in bundled}
    assert bundled_slugs, "no simple-icons imports found — did BrandLogo change shape?"

    # A slug may also key INLINE_SVG — the tier for brands neither Iconify
    # nor simple-icons carries (Outlook). Those are legitimate marks, not
    # dangling references, so they satisfy this guard too.
    inline_block = re.search(r"const INLINE_SVG[^{]*\{(.*?)\n\};", brand_src, re.S)
    inline_keys = set(re.findall(r"^\s*([a-z0-9_]+):", inline_block.group(1), re.M)) if inline_block else set()
    bundled_slugs |= inline_keys

    broken = [
        e for e in _entries()
        if e["slug"] and e["slug"] not in bundled_slugs
    ]
    assert not broken, (
        f"these rows name a simple-icons slug BrandLogo does not import: "
        f"{[(e['id'], e['slug']) for e in broken]}. The middle fallback tier "
        f"renders an empty <svg> for them and the letter mark is unreachable."
    )


@pytest.mark.skipif(not DIRECTORY.exists(), reason="frontend not present in this checkout")
def test_the_three_known_dead_iconify_names_stay_dead():
    """Regression pin, by exact name.

    These are not typos of real icons — the `logos` set has no Docs,
    Sheets or Outlook mark to correct them TO. Anyone reintroducing one is
    guessing, and the guess renders blank rather than failing.
    """
    src = DIRECTORY.read_text()
    for dead in ("logos:google-docs", "logos:google-sheets", "logos:microsoft-outlook"):
        assert dead not in src, (
            f"{dead} does not exist in the Iconify `logos` set. It renders an "
            f"empty placeholder, not an error. Leave iconifyName null and let "
            f"the bundled simple-icons mark render."
        )


@pytest.mark.skipif(not DIRECTORY.exists(), reason="frontend not present in this checkout")
def test_brand_logo_falls_through_when_iconify_has_nothing():
    """The runtime half, asserted structurally.

    Without this the data guards above are cosmetic: they stop the three
    KNOWN dead names, and the fourth one — added next month — is blank
    again. `loadIcon` rejecting is the only signal Iconify gives that a
    name is unknown, so the component must consume it.
    """
    src = BRAND_LOGO.read_text()
    # Assert the CALL, not the import. `"loadIcon" in src` is satisfied by
    # the import line alone — mutation-proven: swapping the call for
    # `Promise.resolve().then` left that check green while disarming the
    # probe entirely.
    assert re.search(r"loadIcon\s*\(\s*iconifyName\s*\)\s*\.catch", src), (
        "BrandLogo no longer probes Iconify with loadIcon(...).catch. An "
        "unknown icon name renders an empty placeholder with no error, so "
        "without this probe a wrong name is a blank tile that no test and no "
        "console warning will surface."
    )
    assert re.search(r"!missing|missing\s*===\s*false", src), (
        "the Iconify branch does not consult the probe result, so it still "
        "returns unconditionally and the fallback chain is unreachable"
    )


@pytest.mark.skipif(not DIRECTORY.exists(), reason="frontend not present in this checkout")
def test_tool_pills_derive_their_brands_from_the_directory():
    """The second hand-maintained brand table, now deleted.

    `ToolPillRow.tsx` kept its own `CONNECTOR_BRANDS` literal whose comment
    called it the "single source of truth" while this directory was a
    second one. They drifted exactly as two hand-kept tables do: the pill
    copy never gained `jira` or `teams`, so every Jira and Teams tool call
    rendered a generic wrench beside Gmail's envelope — reported by the
    founder the first time he used the Jira connector. It also carried its
    own copies of the three nonexistent Iconify names and three unimported
    slugs, so fixing the directory alone left the pills broken.

    Deriving is the fix; this pins it, because re-introducing a literal
    would compile, render, and silently drift again.
    """
    pill = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/modules/chat/ToolPillRow.tsx"
    )
    assert pill.exists(), pill
    src = pill.read_text()

    assert "from '../../components/integrations/connectorDirectory'" in src, (
        "ToolPillRow no longer imports the connector directory — its brand "
        "table has been re-hardcoded and will drift from the grid again"
    )
    assert re.search(r"CONNECTOR_BRANDS[^=]*=\s*Object\.fromEntries\(\s*DIRECTORY", src), (
        "CONNECTOR_BRANDS is no longer derived from DIRECTORY"
    )
    # No brand literals may creep back in alongside the derived map.
    assert "iconifyName: 'logos:" not in src, (
        "a hardcoded Iconify name reappeared in ToolPillRow — brands come "
        "from the directory now, so this is drift waiting to happen"
    )


@pytest.mark.skipif(not DIRECTORY.exists(), reason="frontend not present in this checkout")
def test_brand_dedupe_is_keyed_on_identity_not_slug():
    """`slug` worked as a dedupe key only by accident.

    It was non-null and unique for every brand the old literal carried. The
    directory legitimately has `slug: null` for Slack, Teams and LinkedIn
    (simple-icons dropped all three on legal request), so keying on it
    collapses them to a single `null` entry and a turn touching two of them
    renders one mark.
    """
    for rel in ("frontend/src/modules/chat/ToolPillRow.tsx",
                "frontend/src/modules/chat/ChatPage.tsx"):
        p = Path(__file__).resolve().parents[2] / rel
        src = p.read_text()
        offenders = re.findall(r"(?:has|add|set)\(\s*\w+\.slug", src)
        assert not offenders, (
            f"{rel} still dedupes brands by `.slug`: {offenders}. Slack, Teams "
            f"and LinkedIn all have slug=null, so they collide."
        )
