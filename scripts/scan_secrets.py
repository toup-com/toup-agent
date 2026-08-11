#!/usr/bin/env python3
"""Scan a tree for secret-shaped values. Never prints a secret.

GA run R-4. On 2026-08-09/10 audit subagents ran production ``psql`` and
printed live credentials into session transcripts, and a repo sweep found
a real ``RESTIC_PASSWORD`` committed to a docs table since 2026-04-23.
This scanner is the recurrence guard: it runs in the normal pytest lane
(see ``tests/test_no_committed_secrets.py``) so a secret-shaped value
cannot be committed unnoticed even while GitHub Actions is unavailable.

Two design rules, both learned the hard way:

1. **Output is never the secret.** Findings carry path, line, the
   pattern name, and a SALTED hash prefix. Truncation is not redaction
   (#419: the leaked secret was 49 chars and the truncation showed 50);
   a prefix of a live key is still key material.

2. **The allowlist is keyed on the VALUE, not the file.** Detector
   fixtures legitimately carry a key's *shape* — that is what broke the
   public mirror for two days in #519, because push protection matches
   shape, not realness. So known-synthetic values are acknowledged one
   at a time by salted hash. Allowlisting a path would mean a real
   secret added to an allowlisted file scans clean forever.

Usage:
    python scripts/scan_secrets.py [PATH ...] [--allowlist FILE] [--json]

Exit 0 = clean, 1 = findings, 2 = usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Stable salt: makes the digest useless as a rainbow-table lookup for a
# short/low-entropy value while staying reproducible across runs, so the
# allowlist is committable. NOT a security boundary — the digest is only
# ever compared against itself.
_SALT = b"toup-secret-scan-v1"

#: Files/dirs that never contain source we control.
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", "coverage", ".ruff_cache",
    "ios", "android", "Pods", ".expo", "site-packages",
})

_SKIP_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".svg", ".pdf",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".wav", ".zip",
    ".gz", ".tgz", ".bz2", ".xz", ".jar", ".so", ".dylib", ".bin",
    ".lock", ".pyc", ".map",
})

_MAX_BYTES = 4_000_000


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    pattern: str
    digest: str          # salted, 12 hex chars — NEVER the value
    length: int

    def render(self) -> str:
        return (
            f"{self.path}:{self.line}: {self.pattern} "
            f"(len={self.length}, h={self.digest})"
        )


def digest(value: str) -> str:
    return hashlib.sha256(_SALT + value.encode("utf-8", "replace")).hexdigest()[:12]


# ── Patterns ─────────────────────────────────────────────────────────
#
# Each entry is (name, compiled regex). Group 1, when present, is the
# value to hash; otherwise the whole match is used. Patterns are
# deliberately shape-based: a scanner that tried to decide whether a key
# is "real" would be the #519 mistake in reverse.

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai-key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_\-]{20,}")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
    ("stripe-key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("google-key", re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b")),
    # ENCRYPTED is openssl's DEFAULT output for passphrase-protected keys
    # (`genpkey -aes-256-cbc`, `pkcs8 -topk8`) — round 2 committed one with
    # ZERO findings while this alternation lacked it.
    #
    # PGP is its OWN alternative, not a `PGP ` qualifier on the PKCS banner:
    # GnuPG's private-key armor puts the word BLOCK between KEY and the
    # closing dashes, so the qualifier form `PGP <space> PRIVATE KEY----`
    # matched a string no tool ever produces — a dead branch that gave
    # false confidence of PGP coverage while real PGP secret keys escaped
    # (round 3). The literals below are split across adjacent string parts
    # so this scanner's own source does not carry a contiguous banner and
    # flag itself. Every banner is one a real tool emits.
    ("private-key-block", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE " + "KEY-----"
        r"|-----BEGIN PGP PRIVATE KEY " + "BLOCK-----"
    )),
    # A JWT with a payload — two base64url segments joined by a dot.
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    # DSN carrying an inline password. Group 1 = the password only.
    (
        "dsn-password",
        re.compile(
            r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)"
            r"(?:\+\w+)?://[^\s:/@]+:([^\s@/]{6,})@"
        ),
    ),
    # NAME = "long-opaque-literal" where the name says secret. The value
    # must be a QUOTED literal — an unquoted right-hand side is an
    # expression (`_token = _ctx_token.set(...)`, `cached = _extract(u)`),
    # which is code, not key material. Quotes are what separate "this
    # program computes a token" from "this file contains one".
    # The (?i) is for the NAME (jwt_secret= and JWT_SECRET= both count).
    # Value opacity is two lookaheads, and ONLY the second is pinned
    # case-sensitive:
    #   1. `(?=[^'"\s]*[a-zA-Z])` — contains a letter (excludes pure-numeric
    #      IDs), case-insensitive on purpose.
    #   2. `(?-i:(?=[^'"\s]*[A-Z0-9]))` — contains an UPPERCASE or digit,
    #      case-sensitive, so all-lowercase placeholder prose is not opaque.
    # Round 2 pinned BOTH under (?-i:), which also forced the first to
    # require a real lowercase — so all-uppercase secrets (base32 TOTP
    # seeds, uppercase-hex HMAC keys) silently escaped (round 3). Scoping
    # only the second keeps the prose filter and restores that coverage.
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(?:[A-Z0-9_]*(?:SECRET|PASSWORD|PASSWD|API_?KEY|TOKEN|PRIVATE_?KEY)"
            r"[A-Z0-9_]*)\s*[:=]\s*"
            r"['\"]((?=[^'\"\s]*[a-zA-Z])(?-i:(?=[^'\"\s]*[A-Z0-9]))[A-Za-z0-9+/_\-=]{20,})['\"]"
        ),
    ),
    # Env-file shape: NAME=value on its own line, unquoted, no spaces.
    # `.env`-style files have no expressions, so the same value here IS
    # the credential.
    (
        "env-assignment",
        re.compile(
            r"(?im)^\s*(?:export\s+)?[A-Z0-9_]*(?:SECRET|PASSWORD|PASSWD|API_?KEY|TOKEN|PRIVATE_?KEY)"
            r"[A-Z0-9_]*\s*=\s*"
            r"((?=[^'\"\s]*[a-zA-Z])(?-i:(?=[^'\"\s]*[A-Z0-9]))[A-Za-z0-9+/_\-=]{20,})\s*$"
        ),
    ),
    # PROXIMITY: a secret-ish NAME and an opaque high-entropy token on the
    # same line, with ANY separator between them.
    #
    # This rule exists because the first version of this scanner did not
    # have it and therefore MISSED the one real secret in the repo:
    #
    #   | `RESTIC_PASSWORD` (`<48-char value>`) | In transcript only |
    #
    # — a markdown table cell, so no `=` and no `:`. The scanner returned
    # "0 findings" on the file that motivated writing it. A guard that
    # only catches the shapes you thought of is decoration; the shapes
    # that leak are the ones you did not think of, which is why this rule
    # keys on adjacency rather than on syntax.
    # The NAME must be IDENTIFIER-shaped (ALL_CAPS, or containing an
    # underscore) — the bare English words "token"/"key" appear in prose
    # constantly ("a 15-token call"), and matching those turns the guard
    # into noise, which is how guards get switched off.
    (
        "secret-name-adjacent-token",
        re.compile(
            r"\b(?:[A-Z][A-Z0-9]*_)*"
            r"(?:SECRET|PASSWORD|PASSWD|API_?KEY|TOKEN|PRIVATE_?KEY|CREDENTIAL)"
            r"(?:_[A-Z0-9]+)*\b[^A-Za-z0-9\n]{1,12}"
            r"((?=[A-Za-z0-9+/_\-=]*[a-z])(?=[A-Za-z0-9+/_\-=]*[A-Z])"
            r"(?=[A-Za-z0-9+/_\-=]*[0-9])[A-Za-z0-9+/_\-=]{20,})"
        ),
    ),
]

#: Template/indirection expressions — the value is a reference, not a
#: credential. `${{ secrets.X }}` (GitHub Actions), `${VAR}`, `%(name)s`.
_INDIRECTION = re.compile(r"(?:\$\{|\{\{|secrets\.|vars\.|env\.|%\()")

#: Values that are obviously not credentials, matched against group 1 of
#: the generic `assigned-secret` rule. These are shape-level exclusions
#: (placeholders and env indirection), not judgments about realness.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:"
    r"your[_\-]?|xxx+|placeholder|example|changeme|redacted|<[^>]+>|"
    r"\$\{?[A-Z_]+\}?|process\.env|os\.environ|settings\.|config\.|"
    r"true|false|none|null|undefined"
    r")"
)

#: Names whose "value" is a config knob, not a credential.
_NOT_A_SECRET_NAME = re.compile(
    r"(?i)(?:token(?:s)?_(?:per|limit|budget|count|max|min|window|used|in|out)"
    r"|max_tokens|token_count|api_key_header|token_type|secret_name"
    r"|password_field|token_field|api_key_env|_token_id|token_urlsafe)"
)


def _iter_files(roots: list[Path]):
    for root in roots:
        if root.is_file():
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in _SKIP_SUFFIXES:
                    continue
                yield p


def _git_tracked_files(root: Path):
    """Every file git tracks under ``root`` — the honest scope for a rule
    whose text is "no secret may enter the REPO".

    The original gate scanned a hand-picked list of 5 top-level dirs; the
    repo has 14, so frontend/, extensions/ and every root file were never
    scanned (20.5% of tracked files). A hand-picked list also rots as
    trees are added. Tracked-files-only is deliberate the other way too:
    a developer's local, untracked ``backend/.env`` holds a real local
    DSN and MUST NOT fail the gate — untracked files are not "in the
    repo".
    """
    import subprocess

    res = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git ls-files failed in {root}: {res.stderr.strip()}")
    for rel in res.stdout.split("\0"):
        if not rel:
            continue
        p = root / rel
        if p.suffix.lower() in _SKIP_SUFFIXES or not p.is_file():
            continue
        yield p


#: Patterns whose match is a CONSTANT, not key material. Digesting the
#: bare match would mean ONE allowlist acknowledgement exempts the whole
#: class — which happened: acknowledging the synthetic PEM banner in
#: PRE-LAUNCH-CHECKLIST.md silently allowlisted every private key ever
#: pasted anywhere in the repo. For these, the digest binds the banner
#: to the first NON-EMPTY line that follows it (the first line of key
#: material), so each acknowledgement covers one specific key and a real
#: key — whose body cannot equal a fixture's — always produces an
#: unacknowledged digest.
_CONSTANT_MATCH_PATTERNS = frozenset({"private-key-block"})


#: PEM / PGP-armor structural header lines — ``Name: value``. These
#: (``Proc-Type: 4,ENCRYPTED``, ``Version: GnuPG``, ``Comment:`` …) precede
#: the base64 body and are constant or near-constant across keys, so
#: binding a digest to one collapses two different keys to the same digest
#: — the very class exemption the binding exists to prevent (round 3).
_PEM_HEADER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*:\s")


def _constant_match_binding(lines: list[str], lineno: int, tail: str, path: str) -> str:
    """What a constant-match banner's digest binds to.

    The goal is a digest that DIFFERS between two different keys, so one
    acknowledged fixture cannot exempt the class. That means binding to
    the base64 key material, skipping over anything constant:

    * blank lines — else a banner followed by a blank line (or at EOF)
      bound to the empty string, one digest for every such key;
    * ``Name: value`` armor headers — for a traditional encrypted PEM the
      first non-empty line is ``Proc-Type: 4,ENCRYPTED``, and for PGP it is
      ``Version:``/``Comment:``; binding there collides all keys of that
      shape.

    A banner with no body at all binds to a path-scoped sentinel, so an
    acknowledgement of a body-less fixture can never reach beyond its own
    file. The walk stops at an ``-----END`` line for the same reason: a
    truncated ``BEGIN``+``END`` block used to bind to the constant END
    line, one digest for the shape repo-wide (round 4). No key material
    can hide in that class — any real body line binds per-key first —
    but the sentinel keeps even the degenerate acknowledgement file-local.
    """
    if tail and not _PEM_HEADER_LINE.match(tail):
        return tail
    for follow in lines[lineno:]:
        stripped = follow.strip()
        if not stripped or _PEM_HEADER_LINE.match(stripped):
            continue
        if stripped.startswith("-----END"):
            break
        return stripped
    return f"<no-body:{path}>"


def scan_text(text: str, path: str) -> list[Finding]:
    out: list[Finding] = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, 1):
        if len(line) > 20_000:          # minified bundle — not source we own
            continue
        for name, rx in _PATTERNS:
            for m in rx.finditer(line):
                value = m.group(1) if m.groups() else m.group(0)
                if not value:
                    continue
                if name in ("assigned-secret", "env-assignment", "secret-name-adjacent-token"):
                    if (
                        _PLACEHOLDER.match(value)
                        or _NOT_A_SECRET_NAME.search(m.group(0))
                        or _INDIRECTION.search(m.group(0))
                    ):
                        continue
                if name in _CONSTANT_MATCH_PATTERNS:
                    binding = _constant_match_binding(
                        lines, lineno, line[m.end():].strip(), path
                    )
                    digest_input = f"{value}\n{binding}"
                else:
                    digest_input = value
                out.append(
                    Finding(path, lineno, name, digest(digest_input), len(value))
                )
    return out


def scan_paths(roots: list[Path], allow: set[str], git_tracked: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    if git_tracked:
        files = (p for root in roots for p in _git_tracked_files(root))
    else:
        files = _iter_files(roots)
    for p in files:
        try:
            if p.stat().st_size > _MAX_BYTES:
                continue
            text = p.read_text("utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        findings.extend(f for f in scan_text(text, str(p)) if f.digest not in allow)
    return findings


def load_allowlist(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    allow: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        allow.add(line.split()[0])
    return allow


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", default=["."])
    ap.add_argument("--allowlist", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--git-tracked", action="store_true",
        help="scan every git-tracked file under PATH (default '.', the "
        "repo root) instead of walking the directory — the gate's mode",
    )
    args = ap.parse_args(argv)

    roots = [Path(p) for p in (args.paths or ["."])]
    for r in roots:
        if not r.exists():
            print(f"no such path: {r}", file=sys.stderr)
            return 2

    try:
        findings = scan_paths(roots, load_allowlist(args.allowlist), args.git_tracked)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=2))
    else:
        for f in findings:
            print(f.render())
        print(f"\n{len(findings)} finding(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
