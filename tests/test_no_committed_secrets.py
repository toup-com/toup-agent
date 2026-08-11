"""GA run R-4: no secret-shaped value may enter the repo unacknowledged.

This is the recurrence guard the credential item required. It runs in the
normal pytest lane — not only in a GitHub Action — because the whole
reason this file exists is that guards fail exactly when the machinery
you assumed would run them does not. (At the time it was written every
Actions run in this org was ending in `startup_failure` on an exhausted
spending limit.)

What it protects against, concretely:

* ``RESTIC_PASSWORD`` sat in ``docs/new-vps/DECISIONS.md`` in cleartext
  from 2026-04-23 until the GA run found it — visible to every org
  member. (NOT on the public mirror: the sync workflow ships ``backend/*``
  only, verified by a full-history grep of both repos.)
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

sys.path.insert(0, str(BACKEND))


def _scan(paths: list[str], allowlist: Path | None = ALLOWLIST,
          git_tracked: bool = False, cwd: Path = REPO):
    cmd = [sys.executable, str(SCANNER), *paths]
    if git_tracked:
        cmd += ["--git-tracked"]
    if allowlist:
        cmd += ["--allowlist", str(allowlist)]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def test_repo_has_no_unacknowledged_secret_shaped_values():
    """Every git-tracked file, not a hand-picked subtree list.

    The first version scanned 5 named top-level dirs of the repo's 14 —
    frontend/, extensions/, new-vps/ (the provisioning scripts that
    install the credentials this guard was written about) and every root
    file were never scanned: 20.5% of tracked files, proven by planting
    probes there that the gate passed. A hand-picked list also rots as
    trees are added; `git ls-files` cannot. Tracked-only is deliberate
    the other way too — a developer's local untracked backend/.env holds
    a real local DSN and must not fail the gate.
    """
    result = _scan(["."], git_tracked=True)
    assert result.returncode == 0, (
        "secret-shaped value(s) found that are not in the allowlist.\n\n"
        f"{result.stdout}\n"
        "Each finding is reported as path:line, pattern, length and a "
        "SALTED digest — never the value itself.\n\n"
        "If it is a real credential: remove it, then ROTATE it (removing "
        "from HEAD does not purge git history).\n"
        "If it is a synthetic fixture: add its digest to "
        "backend/tests/secret_scan_allowlist.txt with a comment saying "
        "why it is not a credential."
    )


def test_gate_scope_is_the_whole_tracked_tree(tmp_path):
    """The scope control: a secret in a tree the OLD gate never scanned
    must fail the gate. Builds a minimal git repo shaped like ours."""
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    repo = tmp_path / "repo"
    (repo / "frontend" / "src").mkdir(parents=True)
    fake = "sk-ant-api03-" + "Qw3rTy" * 8
    (repo / "frontend" / "src" / "config.ts").write_text(
        f'const KEY = "{fake}"\n'
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)

    result = _scan(["."], allowlist=ALLOWLIST, git_tracked=True, cwd=repo)
    assert result.returncode == 1, (
        "a synthetic anthropic-key in frontend/ scanned clean — the gate "
        "is back to scanning a subtree list"
    )
    assert "frontend/src/config.ts" in result.stdout


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


# Assembled at runtime so THIS file does not carry the banner's shape —
# the scanner scans source text, and a contiguous banner here would flag
# the gate's own test file (it did, on the first run of the whole-repo
# scope).
_PEM_BANNER = "-----BEGIN " + "PRIVATE KEY-----"


def test_acknowledged_pem_fixtures_do_not_exempt_real_private_keys(tmp_path):
    """The class control. The PEM banner is a CONSTANT: when its digest
    was the bare banner, acknowledging one synthetic fixture allowlisted
    every private key in the repo — three planted real keys passed the
    gate. The digest now binds banner + first line of key material, so
    this key (generated here, never a credential) must fail against the
    COMMITTED allowlist, which acknowledges four PEM fixtures."""
    fake_body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ" + "Xy4"
    leak = tmp_path / "restore-runbook.md"
    leak.write_text(
        "Paste the backup key here:\n\n"
        f"{_PEM_BANNER}\n"
        f"{fake_body}\n"
        f"{_PEM_BANNER.replace('BEGIN', 'END')}\n"
    )

    result = _scan([str(leak)], allowlist=ALLOWLIST)
    assert result.returncode == 1, (
        "a private key with an unacknowledged BODY scanned clean — the "
        "banner acknowledgements are exempting the class again"
    )
    assert "private-key-block" in result.stdout


def test_encrypted_private_keys_are_flagged(tmp_path):
    """The round-2 audit committed a passphrase-protected key with ZERO
    findings: `openssl genpkey ... -aes-256-cbc` emits the ENCRYPTED
    banner, which the alternation lacked. An encrypted key is still a
    credential — the passphrase is usually weaker than the key, and the
    file names exactly what to brute-force."""
    leak = tmp_path / "backup.pem"
    leak.write_text(
        f"{_PEM_BANNER.replace('BEGIN ', 'BEGIN ENCRYPTED ')}\n"
        "MIIFHDBOBgkqhkiG9w0BBQ0wQTApBgkqhkiG9w0B\n"
    )

    result = _scan([str(leak)], allowlist=ALLOWLIST)
    assert result.returncode == 1, (
        "a passphrase-protected private key scanned clean — the ENCRYPTED "
        "banner escaped the private-key pattern"
    )
    assert "private-key-block" in result.stdout


def test_blank_line_after_banner_still_binds_per_key(tmp_path):
    """A banner followed by a blank line (or at EOF) must not collapse
    every such key to one digest — that would recreate the class
    exemption in an edge: one acknowledged degenerate fixture would
    silence every blank-after-banner key."""
    from scripts.scan_secrets import scan_text

    a = scan_text(f"{_PEM_BANNER}\n\nMIIBodyAaaa111\n", "a.pem")
    b = scan_text(f"{_PEM_BANNER}\n\nMIIBodyBbbb222\n", "b.pem")
    assert a and b
    assert a[0].digest != b[0].digest, (
        "two different keys with a blank line after the banner share one "
        "digest — the binding skipped to the empty string, not the body"
    )


def test_all_uppercase_and_digit_secrets_are_caught():
    """Round 2 pinned BOTH opacity lookaheads case-sensitive, which forced
    the value to contain a real LOWERCASE letter — so all-uppercase+digit
    secrets (base32 TOTP/HOTP seeds, uppercase-hex HMAC keys) silently
    escaped both assignment rules. They are secrets and must be caught."""
    from scripts.scan_secrets import scan_text

    totp = "TOTP_" + "SECRET=JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP"
    assert scan_text(totp, "x.env"), "base32 all-caps secret escaped the gate"

    hexkey = "HMAC_" + "SECRET=A1B2C3D4E5F6A7B8C9D0E1F2A3B4C5D6"
    assert scan_text(hexkey, "x.env"), "uppercase-hex secret escaped the gate"

    # And the round-2 false positive it must NOT reintroduce:
    prose = "JWT_" + "SECRET=change-this-to-a-long-random-secret-value-now"
    assert scan_text(prose, "x.env") == []


def test_real_pgp_private_key_armor_is_caught():
    """GnuPG emits a PGP private-key BLOCK armor. The pattern used to list a
    `PGP ` qualifier that built a banner no tool produces — a dead branch —
    so real PGP secret keys escaped. Assemble the real banner at runtime so
    this file does not carry it."""
    from scripts.scan_secrets import scan_text

    banner = "-----BEGIN PGP PRIVATE KEY " + "BLOCK-----"
    key = f"{banner}\nlQVYBGXFsomekeymaterial\n"
    result = scan_text(key, "backup.asc")
    assert any(f.pattern == "private-key-block" for f in result), (
        "a real GnuPG private-key armor was not flagged"
    )


def test_encrypted_pem_with_proc_type_header_binds_per_key():
    """A traditional OpenSSL encrypted PEM's first non-empty line is the
    constant `Proc-Type: 4,ENCRYPTED` header. Binding the digest there
    would collide every such key — the class exemption again. The binding
    must skip armor headers and reach the base64 body."""
    from scripts.scan_secrets import scan_text

    rsa_banner = _PEM_BANNER.replace("BEGIN ", "BEGIN RSA ")

    def enc(iv, body):
        return (
            f"{rsa_banner}\n"
            "Proc-Type: 4,ENCRYPTED\n"
            f"DEK-Info: AES-256-CBC,{iv}\n\n{body}\n"
        )

    a = scan_text(enc("AAAA1111", "MIIEncBodyA"), "a.pem")
    b = scan_text(enc("BBBB2222", "MIIEncBodyB"), "b.pem")
    da = [f for f in a if f.pattern == "private-key-block"][0].digest
    db = [f for f in b if f.pattern == "private-key-block"][0].digest
    assert da != db, (
        "two different encrypted keys share a digest — the binding stopped "
        "at the constant Proc-Type header instead of the body"
    )

    # Body-less banners bind to a path-scoped sentinel: an acknowledgement
    # can never reach beyond its own file.
    e1 = scan_text(f"{_PEM_BANNER}\n", "one.md")
    e2 = scan_text(f"{_PEM_BANNER}\n", "two.md")
    assert e1[0].digest != e2[0].digest


def test_opacity_lookahead_is_case_sensitive(tmp_path):
    """(?i) on the NAME must not leak into the value's [A-Z0-9] opacity
    check — under a bare (?i) it matched lowercase too, flagging
    all-lowercase placeholder prose (`change-this-to-a-...`) as key
    material, which is how allowlist noise gets manufactured."""
    from scripts.scan_secrets import scan_text

    # Assembled at runtime so THIS file does not carry the shapes.
    name = "JWT_" + "SECRET"
    prose = name + "=change-this-to-a-long-random-secret-value-now"
    assert scan_text(prose, "x.env") == [], (
        "an all-lowercase dictionary-word value was flagged as a secret"
    )
    # The control: a value with real opacity still fires.
    opaque_value = "aB3dEf9hIjKlMnOpQrStUvWx" + "Yz01234567890abcdef"
    opaque = name + "=" + opaque_value
    assert any(f.pattern == "env-assignment" for f in scan_text(opaque, "x.env"))


def test_pem_digest_is_per_key_not_per_banner(tmp_path):
    """Two keys, same banner, different bodies → different digests, and
    acknowledging one must not silence the other."""
    from scripts.scan_secrets import digest

    a = tmp_path / "a.pem"
    b = tmp_path / "b.pem"
    a.write_text(f"{_PEM_BANNER}\nMIIBodyAaaa111\n")
    b.write_text(f"{_PEM_BANNER}\nMIIBodyBbbb222\n")

    allow = tmp_path / "allow.txt"
    allow.write_text(
        f"{digest(_PEM_BANNER + chr(10) + 'MIIBodyAaaa111')}"
        "  # synthetic fixture A\n"
    )

    result = _scan([str(a), str(b)], allowlist=allow)
    assert result.returncode == 1, "acknowledging key A silenced key B"
    assert "b.pem" in result.stdout
    assert "a.pem" not in result.stdout


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
