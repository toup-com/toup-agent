"""GA run R-4: no secret-shaped value may enter the repo unacknowledged.

This is the recurrence guard the credential item required. It runs in the
normal pytest lane — not only in a GitHub Action — because the whole
reason this file exists is that guards fail exactly when the machinery
you assumed would run them does not. (At the time it was written every
Actions run in this org was ending in `startup_failure` on an exhausted
spending limit.)

What it protects against, concretely:

* ``RESTIC_PASSWORD`` sat in ``docs/new-vps/DECISIONS.md`` in cleartext
  from 2026-04-23 until the GA run found it — visible to every org member
  and to the public mirror.
* Audit subagents printed live production credentials into session
  transcripts on 2026-08-09/10.

The scanner's own control is exercised below: `test_scanner_detects_a_real
_committed_secret` reconstructs the exact DECISIONS.md shape and asserts
the scanner flags it. A guard whose detector is never proven able to fire
is decoration — the first version of this scanner returned "0 findings"
on precisely that line, because it only understood ``NAME = "value"`` and
the leak was a markdown table cell.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
SCANNER = BACKEND / "scripts" / "scan_secrets.py"
ALLOWLIST = BACKEND / "tests" / "secret_scan_allowlist.txt"

#: Trees that ship to GitHub. `backend/` and `docs/` are where the two
#: real incidents happened; `.github/` holds workflow files that carry
#: secret NAMES and must keep carrying only names.
_SCAN_ROOTS = ["backend", "docs", ".github", "bridge", "scripts"]

sys.path.insert(0, str(BACKEND))


def _scan(paths: list[str], allowlist: Path | None = ALLOWLIST):
    cmd = [sys.executable, str(SCANNER), *paths]
    if allowlist:
        cmd += ["--allowlist", str(allowlist)]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def test_repo_has_no_unacknowledged_secret_shaped_values():
    roots = [r for r in _SCAN_ROOTS if (REPO / r).exists()]
    result = _scan(roots)
    assert result.returncode == 0, (
        "secret-shaped value(s) found that are not in the allowlist.\n\n"
        f"{result.stdout}\n"
        "Each finding is reported as path:line, pattern, length and a "
        "SALTED digest — never the value itself.\n\n"
        "If it is a real credential: remove it, then ROTATE it (removing "
        "from HEAD does not purge git history or the public mirror).\n"
        "If it is a synthetic fixture: add its digest to "
        "backend/tests/secret_scan_allowlist.txt with a comment saying "
        "why it is not a credential."
    )


def test_scanner_detects_a_real_committed_secret(tmp_path):
    """The control. Rebuilds the exact shape of the 2026-04-23 leak.

    The value below is generated here, has never been a credential
    anywhere, and is not in the allowlist — so a passing assertion proves
    the detector fires on this shape rather than proving anything about
    the string.
    """
    fake = "Dh" + "W0o1r6olFibqbcI700N0CaxtEUktpGsx1UnEz1ZoTUbGEs"
    leak = tmp_path / "DECISIONS.md"
    leak.write_text(
        "| Credential | Captured to |\n"
        "|---|---|\n"
        f"| `RESTIC_PASSWORD` (`{fake}`) | In transcript only |\n"
    )

    result = _scan([str(leak)], allowlist=None)
    assert result.returncode == 1, (
        "the scanner did NOT flag a password sitting in a markdown table "
        "cell — this is the exact shape that went unnoticed for 3.5 "
        "months, and a guard that misses it is decoration"
    )
    assert "secret-name-adjacent-token" in result.stdout


def test_scanner_never_prints_the_value(tmp_path):
    """Findings must carry a digest, never key material — truncation is
    not redaction (#419)."""
    fake = "Ab3" + "xY9zQ2w8Er5tY7uI1oP4aS6dF0gH2jK"
    leak = tmp_path / "conf.env"
    leak.write_text(f"MY_API_KEY={fake}\n")

    result = _scan([str(leak)], allowlist=None)
    assert result.returncode == 1
    assert fake not in result.stdout, "the scanner printed the secret"
    for n in (8, 12, 16, 20):
        assert fake[:n] not in result.stdout, (
            f"the scanner printed a {n}-char prefix of the secret — a "
            "prefix of a live key is still key material"
        )


def test_allowlist_entries_are_digests_with_a_reason():
    """An allowlist line without a comment is an unexplained exemption."""
    assert ALLOWLIST.exists()
    for i, raw in enumerate(ALLOWLIST.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        dg = parts[0]
        assert len(dg) == 12 and all(c in "0123456789abcdef" for c in dg), (
            f"line {i}: {dg!r} is not a 12-hex-char digest"
        )
        assert len(parts) > 1 and parts[1].lstrip().startswith("#"), (
            f"line {i}: allowlisted without a reason — say why this value "
            "is not a credential"
        )


def test_allowlist_cannot_hide_a_new_secret_in_an_allowlisted_file(tmp_path):
    """Keying on the value, not the path: a fixture file may carry an
    acknowledged shape AND still fail on a new one."""
    sys.path.insert(0, str(BACKEND))
    from scripts.scan_secrets import digest

    known = "Kn0wn" + "SyntheticFixtureValue123456"
    fresh = "Fr3sh" + "RealLookingCredential987654"
    f = tmp_path / "fixtures.py"
    f.write_text(f'API_KEY = "{known}"\nOTHER_TOKEN = "{fresh}"\n')

    allow = tmp_path / "allow.txt"
    allow.write_text(f"{digest(known)}  # synthetic fixture\n")

    result = _scan([str(f)], allowlist=allow)
    assert result.returncode == 1, (
        "allowlisting one value in a file silenced the whole file"
    )
    assert digest(fresh) in result.stdout
    assert digest(known) not in result.stdout
