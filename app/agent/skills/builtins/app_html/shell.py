"""The jail behind ``bash_app`` — a verification shell, not a general shell.

``bash_app`` exists for one job: let the model check its own work after a
write ("does the file contain the handler I just added?", "how many bytes is
it?", "does node parse the inline script?"). The agent already has a full
``exec`` tool for everything else, so this one is allowlisted rather than
denylisted: a denylist over a shell is a guessing game, and the set of
commands that verify an HTML file is genuinely small.

Three independent gates, each of which alone would be defeatable:

1.  **Binary allowlist** — the first bare word of every pipeline segment must
    be in :data:`ALLOWED_COMMANDS`. This is what removes ``curl``/``wget``/
    ``ssh``/``nc`` (exfiltration), ``npm``/``pip``/``apt`` (arbitrary code),
    ``sudo``/``chmod``/``systemctl`` (privilege), and ``env``/``printenv``
    (the container's injected secrets).
2.  **Substitution ban** — backticks and ``$(...)`` are refused outright,
    because they hide a binary from gate 1. Both this gate and the segmenter
    read a QUOTE-AWARE mask of the command (:func:`_masks`), so shell syntax
    is only recognised where the shell would actually act on it.
3.  **Path jail** — every token that looks like a path must resolve inside
    the app root, so ``cat ../../etc/shadow`` and ``> /etc/cron.d/x`` are
    refused even though ``cat`` and ``>`` are fine in principle.

The child then runs with ``cwd=<app root>``, ``sandbox_environ()`` and
``sandbox_preexec()`` — the same hardening ``AppManager`` applies to npm
children — under a wall-clock timeout, with capped output.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from typing import List, Optional, Tuple

# Absolute, for the same reason as skill.py: the loader imports these
# modules by path under a top-level name, where a relative import fails.
from app.agent.skills.builtins.app_html.store import (
    AppStoreError,
    apps_root,
    repair_file_modes,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 16_000

#: Everything a model needs to inspect one HTML file, and nothing else.
ALLOWED_COMMANDS = frozenset({
    # listing / metadata
    "ls", "stat", "file", "du", "wc", "basename", "dirname", "pwd", "find",
    # reading
    "cat", "head", "tail", "cut", "tr", "sort", "uniq", "rev", "nl",
    # searching
    "grep", "egrep", "fgrep", "rg", "awk", "sed",
    # comparing / hashing
    "diff", "cmp", "md5sum", "sha1sum", "sha256sum",
    # trivial output
    "echo", "printf", "true", "false", "test",
    # real validators — an HTML app's inline JS is checked with these
    "node", "python3", "python", "jq", "tidy", "xmllint",
})

#: Refused outright — each hides the real command from the allowlist, or
#: detaches it from our timeout. Scanned against the EXPANSION mask (see
#: :func:`_masks`), so a literal ``$(`` inside single quotes — ``awk
#: '{print $(NF)}'``, ``grep '${VAR}'`` — is not mistaken for one.
_FORBIDDEN_SYNTAX: Tuple[Tuple[str, str], ...] = (
    ("`", "backtick command substitution"),
    ("$(", "$( ) command substitution"),
    ("<(", "process substitution"),
    (">(", "process substitution"),
    ("${", "parameter expansion"),
)

#: A leading `VAR=value` on a segment — skipped when finding the binary, and
#: refused, because it is how you re-point PATH/LD_PRELOAD at your own code.
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

#: Tokens that name a filesystem location rather than a flag or a pattern.
_PATHISH_RE = re.compile(r"(^[~/])|(^\.{1,2}/)|(/)")


class ShellRefusal(AppStoreError):
    """The command is not something this tool will run. Message is user-facing."""


def _refuse(reason: str) -> "ShellRefusal":
    return ShellRefusal(
        f"{reason}. bash_app is a verification shell scoped to the app "
        f"directory — it runs only: {', '.join(sorted(ALLOWED_COMMANDS))}. "
        f"Use the main `exec` tool for anything broader."
    )


class UnbalancedQuote(ShellRefusal):
    """The command's quoting does not close. Message is user-facing."""


def _masks(cmd: str) -> Tuple[str, str]:
    """Two blanked copies of ``cmd``, for two different scans.

    Round 18. The segmenter used to be ``re.split(r"\\|\\||&&|[;|\\n]", cmd)``,
    which reads a shell command as if quotes did not exist. The single most
    common verification command a model writes is a grep alternation::

        grep -c "requestAnimationFrame\\|setInterval" snake.html
        grep -nE "gameOver|restart" snake.html

    The regex split that on the ``|`` INSIDE the pattern, handed the front
    half — ``grep -nE "gameOver`` — to :func:`shlex.split`, and got back
    ``ValueError: No closing quotation``. The model saw
    ``could not parse the command (No closing quotation)`` for a command that
    is perfectly well formed, could not tell what was wrong with it, and the
    build stalled on a check that was never going to run.

    Both masks replace characters with spaces so every index in the result
    still lines up with ``cmd``:

    * ``ops`` blanks everything inside quotes of EITHER kind, plus every
      backslash-escaped character. Operators (``| ; && || &``) are inert in
      all three positions, so this is what the segmenter and the
      backgrounding check must look at.
    * ``expand`` blanks single-quoted runs and escapes only. Parameter and
      command substitution ARE live inside double quotes, so those must stay
      visible to :data:`_FORBIDDEN_SYNTAX`, while ``'$(NF)'`` must not.

    Raises :class:`UnbalancedQuote` if a quote never closes — the one case
    where the model really did write something unrunnable, and now gets a
    sentence that says so instead of a :mod:`shlex` exception.
    """
    ops: List[str] = []
    expand: List[str] = []
    quote: Optional[str] = None
    i, n = 0, len(cmd)
    while i < n:
        ch = cmd[i]
        # A backslash escape hides the NEXT character from every scan. Inside
        # single quotes it is literal (POSIX), so it escapes nothing there.
        if ch == "\\" and quote != "'" and i + 1 < n:
            ops.append("  ")
            expand.append("  ")
            i += 2
            continue
        if quote:
            closing = ch == quote
            ops.append(" ")
            expand.append(ch if quote == '"' else " ")
            if closing:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            ops.append(" ")
            expand.append(" ")
            i += 1
            continue
        ops.append(ch)
        expand.append(ch)
        i += 1
    if quote:
        raise UnbalancedQuote(
            f"the command has an unclosed {quote} quote. Close it, or drop the "
            f"quotes if the pattern has no spaces — e.g. "
            f"grep -n \"function move\" app.html"
        )
    return "".join(ops), "".join(expand)


def split_segments(cmd: str, ops_mask: str) -> List[str]:
    """Split ``cmd`` on UNQUOTED ``| || && ; newline`` using ``ops_mask``.

    Slices come from ``cmd``, so each segment keeps its original quoting and
    can be handed to :func:`shlex.split` intact.
    """
    out: List[str] = []
    start = 0
    i, n = 0, len(cmd)
    while i < n:
        two = ops_mask[i:i + 2]
        if two in ("||", "&&"):
            out.append(cmd[start:i])
            i += 2
            start = i
            continue
        if ops_mask[i] in ";|\n":
            out.append(cmd[start:i])
            i += 1
            start = i
            continue
        i += 1
    out.append(cmd[start:])
    return out


def _tokens_of(segment: str) -> List[str]:
    try:
        return shlex.split(segment, comments=False, posix=True)
    except ValueError:
        # Unreachable for a segment produced by split_segments (quotes are
        # balanced by construction) — kept because validate_command is not the
        # only possible caller and a raw ValueError here would reach the model
        # as a traceback rather than as an instruction.
        raise UnbalancedQuote(
            "the command has an unclosed quote. Close it, or drop the quotes "
            "if the pattern has no spaces."
        ) from None


def _check_paths(tokens: List[str], root: str) -> None:
    """Every path-shaped token must land inside ``root``."""
    real_root = os.path.realpath(root)
    for tok in tokens:
        if not tok or tok.startswith("-"):
            continue
        if not _PATHISH_RE.search(tok):
            continue
        candidate = os.path.expanduser(tok)
        if not os.path.isabs(candidate):
            candidate = os.path.join(real_root, candidate)
        # normpath, not realpath: the target may not exist yet (a redirect),
        # and realpath on a missing path silently resolves the parent only.
        resolved = os.path.normpath(candidate)
        if resolved != real_root and not resolved.startswith(real_root + os.sep):
            raise _refuse(
                f"the path {tok!r} resolves outside the app directory "
                f"({resolved})"
            )


def validate_command(command: str) -> str:
    """Return the command if it may run; raise :class:`ShellRefusal` if not."""
    cmd = (command or "").strip()
    if not cmd:
        raise ShellRefusal("command is required")
    if len(cmd) > 4000:
        raise _refuse("the command is over 4000 characters")

    ops_mask, expand_mask = _masks(cmd)

    for needle, what in _FORBIDDEN_SYNTAX:
        if needle in expand_mask:
            raise _refuse(f"{what} is not allowed")
    # A trailing/embedded `&` backgrounds the child out from under the
    # timeout. `&&` is legitimate, so only refuse a lone one — and only an
    # UNQUOTED one, or `grep -o "a&b"` would be refused as backgrounding.
    if re.search(r"(?<!&)&(?!&)", ops_mask):
        raise _refuse("backgrounding with & is not allowed")

    root = os.path.realpath(apps_root())
    saw_segment = False
    for segment in split_segments(cmd, ops_mask):
        segment = segment.strip()
        if not segment:
            continue
        tokens = _tokens_of(segment)
        if not tokens:
            continue
        head = tokens[0]
        if _ASSIGNMENT_RE.match(head):
            raise _refuse(f"environment assignment {head!r} is not allowed")
        # `/usr/bin/grep` and `./grep` both dodge a bare-name allowlist.
        if "/" in head:
            raise _refuse(
                f"call {os.path.basename(head)!r} by name, not by path ({head!r})"
            )
        if head not in ALLOWED_COMMANDS:
            raise _refuse(f"{head!r} is not an allowed command")
        _check_paths(tokens[1:], root)
        saw_segment = True

    if not saw_segment:
        raise ShellRefusal("command is required")
    return cmd


async def run_in_app_dir(
    command: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Tuple[int, str]:
    """Validate, then run ``command`` with CWD at the app root.

    Returns ``(returncode, combined_output)``. Raises :class:`ShellRefusal`
    before spawning anything if the command does not pass the jail.

    The timeout is enforced HERE, not by the tool dispatcher: skill tools
    bypass ``ToolExecutor``'s ``asyncio.wait_for(tool_timeout)`` wrapper
    (they take the ``skill_loader`` branch in ``ToolExecutor.execute``), so a
    hung child would otherwise hang the whole turn.
    """
    cmd = validate_command(command)
    root = os.path.realpath(apps_root())
    os.makedirs(root, exist_ok=True)
    # The child runs DROPPED to an unprivileged uid, and every file written
    # before the write-time mode fix is still 0600 root-only — so on an
    # upgraded container the model can read the app it built today and gets
    # "Permission denied" for the one it built last week. Repaired here, not
    # only at `ensure_root`, because `bash_app` is the one tool in this skill
    # that reaches a container which may never create another app.
    #
    # READ, not write: the sweep sets 0644. It cannot give the dropped uid —
    # or the sandboxed page, which has no filesystem at all — any new power
    # over an app beyond looking at it.
    repair_file_modes(root)

    try:
        from app.services.exec_env import sandbox_environ, sandbox_preexec
        # sandbox_environ, not scrubbed_environ: with an explicit ``env=`` the
        # preexec's HOME fix never reaches the child (round 15).
        env = sandbox_environ()
        preexec = sandbox_preexec()
    except Exception:  # pragma: no cover - never block on hardening import
        env, preexec = dict(os.environ), None

    proc = await asyncio.create_subprocess_shell(
        cmd,
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        preexec_fn=preexec,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        raise ShellRefusal(
            f"the command did not finish within {timeout}s and was killed"
        ) from None

    text = (out or b"").decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + f"\n… [truncated at {MAX_OUTPUT_CHARS} chars]"
    return proc.returncode or 0, text
