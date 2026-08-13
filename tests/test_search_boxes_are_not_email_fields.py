"""A search box Chrome mistakes for an email field empties the page.

Found 2026-08-13. Opening /connectors showed "No integrations match your
search" with the user's own email address sitting in the search box. React
never put it there — `search` state was empty. Chrome's heuristic autofill
saw an unnamed `<input type="text">` on an account-ish page, decided it was
an email field, and filled it. The page then filtered every integration
against a string the user had not typed.

The symptoms are what make it expensive to diagnose:
  - it survives a reload, because the reload re-triggers autofill;
  - clearing the box fixes it, so it reads as "sticky state" rather than
    as something being injected;
  - the page looks like the product is EMPTY, not like a filter is active.

All nine search inputs in the app had the same hole. This asserts the fix
stays on every one of them — the next search box added by copy-paste
inherits it, and one that doesn't is caught here rather than by a user
staring at an empty Connectors page.
"""

from __future__ import annotations

import pathlib
import re

import pytest

FRONTEND = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "src"

# Chrome ignores autoComplete="off" on some inputs, so the fix is layered:
# `off` for the standards path, and the two vendor opt-outs for 1Password /
# LastPass which autofill independently of the browser.
REQUIRED = ('autoComplete="off"', "data-1p-ignore", 'data-lpignore="true"')


def _search_inputs():
    """(file, line_no, the attribute block around each Search placeholder)."""
    if not FRONTEND.is_dir():
        pytest.skip("frontend/ not present in this checkout")
    found = []
    for p in sorted(FRONTEND.rglob("*.tsx")):
        lines = p.read_text().splitlines()
        for i, line in enumerate(lines):
            if 'placeholder="Search' in line:
                # The JSX attribute list this placeholder belongs to: walk back
                # to `<input` and forward a little past it.
                start = i
                while start > 0 and "<input" not in lines[start]:
                    start -= 1
                    if i - start > 25:  # not an <input> at all
                        break
                found.append((p, i + 1, "\n".join(lines[start:i + 12])))
    return found


def test_every_search_input_opts_out_of_autofill():
    inputs = _search_inputs()
    assert inputs, "no search inputs found — has the selector changed?"
    missing = []
    for path, line, block in inputs:
        absent = [a for a in REQUIRED if a not in block]
        if absent:
            rel = path.relative_to(FRONTEND.parent.parent)
            missing.append(f"{rel}:{line} missing {absent}")
    assert not missing, (
        "search inputs Chrome can autofill with a saved email:\n  "
        + "\n  ".join(missing)
        + "\n\nA filled search box filters the page to nothing and reads as "
          "an empty product, not as an active filter."
    )


def test_the_connectors_search_is_covered():
    """The page where this was found, named explicitly — a generic sweep that
    silently stops matching would otherwise pass with zero coverage."""
    page = FRONTEND / "pages" / "AgentIntegrationsPage.tsx"
    src = page.read_text()
    assert 'placeholder="Search' in src, "the connectors search box is gone"
    block = src[src.index('placeholder="Search') - 600: src.index('placeholder="Search') + 600]
    for attr in REQUIRED:
        assert attr in block, f"connectors search lost {attr}"


def test_no_search_input_declares_an_email_or_name_type():
    """`type="email"` or a name like `email`/`user` invites autofill no matter
    what else is set — the attributes above are a suppressant, not a licence."""
    for path, line, block in _search_inputs():
        head = block[: block.find("placeholder")]
        assert 'type="email"' not in head, f"{path}:{line} is type=email"
        assert not re.search(r'name="(email|username|user)"', head), (
            f"{path}:{line} carries an autofill-magnet name attribute"
        )
