"""Keep the tenant workspace writable by the exec sandbox.

The agent process runs as ROOT inside the container; every `exec`, PTY and
process child is dropped to uid 1000 (`toup`) by EXEC_SANDBOX_USER
(Dockerfile.agent, EXF-3 hardening). Anything the root agent writes into the
bind-mounted workspace therefore lands root-owned 0644/0755, and the dropped
uid cannot rewrite or delete it — a generated document can be created by the
agent and then be un-editable by the very shell the agent uses to work on it.

Since 2026-06 this was patched by hand after every recreate/rollout:

    docker run --rm -v /data/agents/$p/workspace:/w --entrypoint sh $IMAGE \\
      -c "mkdir -p /w/generated && chmod -R a+rwX /w/generated"

on each hardened tenant. A manual step that must be repeated after every
deploy is a defect, not a procedure: it was forgotten at least twice, and it
does not exist at all for tenants nobody remembers to run it on.

`chmod` (unlike `chown`) needs no capability when the caller owns the file,
and root owns everything it just wrote — so the container can do this for
itself under CapDrop=ALL. Two enforcement points:

  * boot sweep (`sweep_workspace_perms`) — one bounded pass over the shared
    subtrees, which is exactly what the manual ritual did;
  * write time (`share_path` / `shared_makedirs`) — so files created later in
    the container's life never regress.

Permissions here are deliberately permissive INSIDE a single tenant's private
volume: /data/agents/<prefix>/workspace is bind-mounted into exactly one
container, and root + the sandbox uid are the only writers. This grants no
cross-tenant access — it makes the two uids of the SAME tenant agree, which
is the state the manual ritual already established.
"""

from __future__ import annotations

import logging
import os
import stat

logger = logging.getLogger(__name__)

# Subtrees the dropped-uid exec must be able to write. `generated` is the
# document-output root (file_storage + tool_executor's write_file redirect);
# `apps` and `vibecoding` are build workspaces where exec runs compilers.
SHARED_SUBDIRS = ("generated", "apps", "vibecoding")

# Bound the boot sweep so a tenant with a huge apps/ tree cannot delay boot.
_SWEEP_MAX_ENTRIES = 20000

_DIR_MODE = 0o777
_FILE_MODE = 0o666


def _chmod(path: str, mode: int) -> bool:
    """chmod, tolerating races and foreign ownership. Returns True if applied."""
    try:
        current = stat.S_IMODE(os.lstat(path).st_mode)
        if current == mode:
            return False
        os.chmod(path, mode)
        return True
    except FileNotFoundError:
        return False
    except PermissionError:
        # Not ours (e.g. a file the sandbox uid created and root's umask
        # cannot widen). Harmless: that uid can already write it.
        return False
    except OSError as e:
        logger.debug("[workspace-perms] chmod %s failed: %s", path, e)
        return False


def share_path(path: str) -> None:
    """Make one file or directory writable by the sandbox uid."""
    try:
        mode = _DIR_MODE if os.path.isdir(path) else _FILE_MODE
    except OSError:
        return
    _chmod(path, mode)


def shared_makedirs(path: str) -> None:
    """os.makedirs(exist_ok=True) that leaves every directory it creates
    writable by the sandbox uid — including intermediate parents."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)
    # Walk up from `path` widening directories we own. Stop at the workspace
    # root (or filesystem root) rather than climbing into /app.
    root = os.path.abspath(workspace_root())
    cur = os.path.abspath(path)
    while cur.startswith(root) and cur != os.path.dirname(cur):
        _chmod(cur, _DIR_MODE)
        if cur == root:
            break
        cur = os.path.dirname(cur)


def share_tree(root: str, *, max_entries: int = _SWEEP_MAX_ENTRIES,
               skip_dirs: tuple = ("node_modules", ".git")) -> dict:
    """Make one arbitrary subtree writable by the sandbox uid. Bounded.

    ``sweep_workspace_perms`` only covers subtrees of the tenant workspace,
    but the Expo app root is ``$TOUP_APPS_DIR`` (``/opt/toup-agent/apps`` by
    default), outside it. Round 15: the agent writes an app's files as ROOT
    and then runs ``npm`` dropped to uid 1000, so without this the install
    fails ``EACCES``/exit 243 on ``mkdir node_modules``.

    ``node_modules`` is skipped by default — it is npm's own output, already
    owned by the uid that will rewrite it, and walking 27k files to re-chmod
    them is the exact cost this pipeline exists to avoid.

    chmod, never chown: the container runs with CapDrop=ALL, so chown needs a
    capability it does not have, while chmod only needs ownership — and root
    owns everything it just wrote.
    """
    changed = 0
    seen = 0
    truncated = False
    if not root or not os.path.isdir(root):
        return {"root": root, "entries": 0, "changed": 0, "truncated": False}
    if _chmod(root, _DIR_MODE):
        changed += 1
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in dirnames:
            seen += 1
            if _chmod(os.path.join(dirpath, name), _DIR_MODE):
                changed += 1
        for name in filenames:
            seen += 1
            if _chmod(os.path.join(dirpath, name), _FILE_MODE):
                changed += 1
        if seen >= max_entries:
            truncated = True
            break
    return {"root": root, "entries": seen, "changed": changed, "truncated": truncated}


def workspace_root() -> str:
    from app.config import settings

    return getattr(settings, "agent_workspace_dir", None) or "./workspace"


def sweep_workspace_perms() -> dict:
    """One bounded pass over the shared subtrees. Returns a summary dict.

    Replaces the post-rollout manual chmod. Safe to call repeatedly; only
    entries whose mode actually differs are touched.
    """
    from app.config import settings

    if not getattr(settings, "workspace_shared_perms_enabled", True):
        return {"skipped": "disabled"}

    root = workspace_root()
    changed = 0
    seen = 0
    truncated = False
    for sub in SHARED_SUBDIRS:
        base = os.path.join(root, sub)
        try:
            os.makedirs(base, exist_ok=True)
        except OSError as e:
            logger.warning("[workspace-perms] cannot create %s: %s", base, e)
            continue
        if _chmod(base, _DIR_MODE):
            changed += 1
        for dirpath, dirnames, filenames in os.walk(base):
            for name in dirnames:
                seen += 1
                if _chmod(os.path.join(dirpath, name), _DIR_MODE):
                    changed += 1
            for name in filenames:
                seen += 1
                if _chmod(os.path.join(dirpath, name), _FILE_MODE):
                    changed += 1
            if seen >= _SWEEP_MAX_ENTRIES:
                truncated = True
                break
        if truncated:
            break

    summary = {"root": root, "entries": seen, "changed": changed, "truncated": truncated}
    if changed or truncated:
        logger.info("[workspace-perms] boot sweep: %s", summary)
    return summary
