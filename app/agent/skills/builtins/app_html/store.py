"""Filesystem + manifest layer for single-file HTML apps.

Layout, inside the tenant's agent container::

    <APPS_ROOT>/
      <slug>.html                       one app = one self-contained file
      manifest.json                     {"version":1,"apps":{slug: {...}}}
      .versions/<slug>/<rev>-<ts>.html  Phase 3 history, newest last

``APPS_ROOT`` resolves to ``$TOUP_HTML_APPS_DIR``, else
``{settings.agent_workspace_dir}/apps`` (``/app/workspace/apps``).

**Why not literally ``~/apps``.** The brief says ``~/apps/<slug>.html``, and
``~/apps`` IS a working path to these files — :func:`ensure_root` drops a
symlink there. But it cannot be the *real* storage: the provisioning bridge
mounts exactly two rw volumes into an agent container
(``new-vps/08-provisioning-bridge.sh``)::

    /data/agents/<prefix>/workspace -> /app/workspace
    /data/agents/<prefix>/skills    -> /app/skills

``$HOME`` (``/root``) is container-local, so every blue-green upgrade — which
recreates the container from the same image + volumes — would silently drop
every app the user ever built. (The Expo tree at ``/opt/toup-agent/apps`` has
this exact defect today; it is out of scope here, and noted in
``MIGRATION_INVENTORY.md`` §4.) Storage goes on the volume; ``~/apps`` points
at it.

Everything in this module is deliberately synchronous and free of DB/network
imports so it can be unit-tested against a tmpdir with no fixtures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Naming ────────────────────────────────────────────────────────────
# Lowercase kebab. Bounded at 60 to match `apps.slug` (String(60)) so a slug
# that is legal here can always be persisted. No leading/trailing hyphen, so
# a slug can never be confused for a flag by anything that shells out.
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,58}[a-z0-9])?$")

# Names that would collide with the store's own files or with a route
# segment on the serving side.
_RESERVED_SLUGS = frozenset({
    "manifest", "index", "versions", "assets", "static", "api",
    "health", "favicon", "robots", "sitemap", "null", "undefined",
})

MANIFEST_NAME = "manifest.json"
VERSIONS_DIR = ".versions"
HTML_SUFFIX = ".html"

# Keep the last N revisions per app. Each is a whole file, but a whole file
# here is tens of KB, so 20 revisions of 20 apps is single-digit MB — three
# orders of magnitude under one Expo `node_modules`.
MAX_VERSIONS_PER_APP = 20

# A create/edit that produces less than this is a stub, not an app. The
# 952-byte `x.pdf` incident (2026-08-18) is the same failure in the document
# lane: a prompt-mandated tool called with nothing, producing a real,
# persisted, billed artifact. Refuse it at the tool boundary instead.
MIN_HTML_BYTES = 400

# Hard ceiling per file. Generous — a rich single-file app with inlined SVG
# and base64 sprites lands well under this — but bounded so one runaway
# generation cannot fill the workspace volume.
MAX_HTML_BYTES = 4 * 1024 * 1024

# The ONLY external origin an artifact may pull code or styles from.
ALLOWED_CDN_HOSTS = frozenset({"cdnjs.cloudflare.com"})

# Guards the manifest read-modify-write. Tool calls within one turn can run
# concurrently (`agent_runner._run_parallel_safe`), so two creates in the
# same turn would otherwise race the whole file away.
_MANIFEST_LOCK = threading.RLock()


class AppStoreError(Exception):
    """Anything the model can fix by calling again with different arguments."""


# ── Root resolution ───────────────────────────────────────────────────

def apps_root() -> str:
    """Absolute path of the HTML-app root.

    Read fresh every call (not memoized) so a test can point it at a tmpdir
    with ``monkeypatch.setenv`` and so an operator can move it with a
    container restart.
    """
    env = (os.environ.get("TOUP_HTML_APPS_DIR") or "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    try:
        from app.config import settings
        workspace = getattr(settings, "agent_workspace_dir", "") or "/app/workspace"
    except Exception:  # pragma: no cover - config must never break the store
        workspace = "/app/workspace"
    return os.path.abspath(os.path.join(workspace, "apps"))


#: Mode the app root and its subdirectories are repaired to. rwx for owner
#: and group, rx for other. Round 15's lesson applies here in miniature: the
#: fix for "a directory the process cannot write" is **chmod, not chown** —
#: the container drops every capability including CAP_CHOWN, so a chown
#: repair raises EPERM and leaves the directory exactly as unusable as it
#: found it, having reported that it fixed something.
_DIR_MODE = 0o775

#: Mode every file this store writes is repaired to before it is published.
#:
#: `tempfile.mkstemp` creates its file 0600 — owner only, by design — and
#: `os.replace` preserves the temp file's mode, so the app file inherited it.
#: The skill runs as root and `bash_app` runs DROPPED to an unprivileged uid
#: (`sandbox_preexec`), so the verification shell could not read the app it had
#: just written. Observed end to end on a device: the model was asked to grep
#: its own app and answered "the app file is permission-blocked in the app
#: sandbox (Permission denied)". Every `bash_app` check on the file — `wc -c`,
#: `grep`, `node --check` — has been failing that way.
#:
#: The directory got this care (`_DIR_MODE` above) and the file did not, which
#: is the whole bug: a readable directory full of unreadable files.
_FILE_MODE = 0o644


#: Ceiling on how many files one sweep will look at. The root holds one file
#: per app plus a manifest, and `.versions/` holds at most
#: MAX_VERSIONS_PER_APP per app — a couple of hundred entries for a heavy
#: user. The cap exists so a volume that has been used for something else
#: entirely cannot turn a tool call into a directory walk.
_REPAIR_SCAN_LIMIT = 2000


def repair_file_modes(root: Optional[str] = None) -> int:
    """chmod every stored file to :data:`_FILE_MODE`. Returns how many moved.

    ``_FILE_MODE`` is applied to the temp file inside :func:`_atomic_write`,
    which fixes every file written from that commit onward and **no file
    written before it**. Every app already on a tenant's volume is still 0600,
    so on an upgraded container the model can build a new app and grep it, and
    cannot grep the app it built last week — the same "Permission denied", now
    with a shape that reads like a per-app fault.

    So the repair has to be a sweep, and it has to run somewhere a container
    that never builds anything new still reaches it (see
    :func:`shell.run_in_app_dir`, which calls this before it spawns).

    Deliberately narrow:

    * **chmod, not chown.** The container drops CAP_CHOWN, so a chown repair
      raises EPERM and reports that it fixed something (round 15).
    * **0644, not 0666.** The dropped uid needs to READ what it is verifying.
      Making these files writable by other would hand the sandboxed shell —
      and anything else running as that uid — the ability to rewrite a
      published app behind the store's back. Read is the whole requirement.
    * **Regular files inside the root only.** Symlinks are skipped rather than
      followed: `chmod` follows a link to its target, and the one link this
      layout has (``~/apps`` → the root) points at a directory.
    """
    root = os.path.realpath(root or apps_root())
    moved = 0
    scanned = 0
    try:
        directories = [root, os.path.join(root, VERSIONS_DIR),
                       os.path.join(root, STATE_DIR),
                       os.path.join(root, PREVIEWS_DIR),
                       os.path.join(root, JUDGED_DIR)]
        # One level under `.versions/` is a directory per slug.
        try:
            vroot = os.path.join(root, VERSIONS_DIR)
            directories.extend(
                os.path.join(vroot, name) for name in os.listdir(vroot)
                if os.path.isdir(os.path.join(vroot, name))
            )
        except OSError:
            pass
        for directory in directories:
            try:
                names = os.listdir(directory)
            except OSError:
                continue
            for name in names:
                if scanned >= _REPAIR_SCAN_LIMIT:
                    logger.warning(
                        "[app_html] permission sweep stopped at %d files",
                        _REPAIR_SCAN_LIMIT,
                    )
                    return moved
                path = os.path.join(directory, name)
                scanned += 1
                try:
                    if os.path.islink(path) or not os.path.isfile(path):
                        continue
                    mode = os.stat(path).st_mode & 0o7777
                    if mode == _FILE_MODE:
                        continue
                    os.chmod(path, _FILE_MODE)
                    moved += 1
                except OSError as exc:
                    # A file owned by another uid cannot be chmodded by us and
                    # never will be; log it once at debug and keep going, so
                    # one bad entry does not cost the other forty.
                    logger.debug("[app_html] could not chmod %s: %s", path, exc)
    except OSError:  # pragma: no cover - defensive; the loops swallow their own
        logger.debug("[app_html] permission sweep failed", exc_info=True)
    if moved:
        logger.warning(
            "[app_html] repaired the mode on %d app file(s) under %s — they "
            "were written before the mode was set at write time", moved, root,
        )
    return moved


def repair_permissions(path: str) -> bool:
    """Try to make ``path`` writable by this process. True if it now is.

    Called before a write and again after one fails with EACCES. Both are
    needed: the first turns a directory created by an earlier image's root
    process into a usable one before the build starts, and the second catches
    the case where the directory was fine and a stale file inside it was not.
    """
    try:
        if not os.path.isdir(path):
            return False
        if os.access(path, os.W_OK | os.X_OK):
            return True
        mode = os.stat(path).st_mode & 0o7777
        os.chmod(path, mode | _DIR_MODE)
        return os.access(path, os.W_OK | os.X_OK)
    except OSError as exc:
        logger.warning("[app_html] could not repair %s: %s", path, exc)
        return False


def _unwritable(path: str) -> "AppStoreError":
    """The one sentence a caller gets when the volume will not take a write.

    Names no errno, no uid and no container path — the model cannot fix any
    of those, and the string ends up in a step detail. It says the one true
    thing: this is not something the model did wrong.
    """
    logger.error("[app_html] app root is not writable: %s", path)
    return AppStoreError(
        "the app folder can't be written to right now, so nothing was saved. "
        "This is a workspace problem, not a problem with your file — say so "
        "plainly and offer to try again."
    )


def ensure_root() -> str:
    """Create the root (and ``.versions/``) and point ``~/apps`` at it.

    The symlink is best-effort: on a box where ``$HOME`` is read-only, or
    where ``~/apps`` is already a real directory belonging to something else,
    we leave it alone and keep working off the real root.

    Raises :class:`AppStoreError` if the root cannot be made writable, rather
    than returning a path that every subsequent write will fail on — a build
    that cannot store anything should stop at its first step with a reason,
    not four steps later with an errno.
    """
    root = apps_root()
    try:
        os.makedirs(root, exist_ok=True)
    except OSError:
        logger.warning("[app_html] could not create %s", root, exc_info=True)
    if not os.path.isdir(root) or not repair_permissions(root):
        raise _unwritable(root)
    versions = os.path.join(root, VERSIONS_DIR)
    try:
        os.makedirs(versions, exist_ok=True)
        repair_permissions(versions)
    except OSError:
        logger.warning("[app_html] could not create %s", versions, exc_info=True)
    # The directory was given this care and the files were not, which left a
    # readable directory full of unreadable files on every volume that
    # predates the write-time fix. Cheap and idempotent: a root whose modes
    # are already right does one stat per file and changes nothing.
    repair_file_modes(root)

    home_apps = os.path.abspath(os.path.expanduser("~/apps"))
    if home_apps != root:
        try:
            if os.path.islink(home_apps):
                if os.path.realpath(home_apps) != os.path.realpath(root):
                    os.unlink(home_apps)
                    os.symlink(root, home_apps)
            elif not os.path.exists(home_apps):
                os.makedirs(os.path.dirname(home_apps), exist_ok=True)
                os.symlink(root, home_apps)
        except OSError as exc:
            logger.debug("[app_html] ~/apps symlink not created: %s", exc)
    return root


# ── Slug + path ───────────────────────────────────────────────────────

def normalise_slug(raw: str) -> str:
    """Validate a slug, or raise :class:`AppStoreError` explaining the rule.

    Case is the ONE thing normalised: ``Budget`` and ``budget`` are the same
    app, because a slug is a URL path segment and asking a model to remember
    the casing it chose three turns ago is a needless failure mode.

    Nothing else is silently rewritten. An underscore, a space or a ``..``
    raises instead of being massaged into something adjacent — the slug is
    the app's identity in the manifest, the DB row, the URL and the chat
    chip, so a quiet fixup would hand back a name the caller did not choose
    and its next ``edit_app_file`` would miss.
    """
    slug = (raw or "").strip().lower()
    if not slug:
        raise AppStoreError("slug is required (lowercase letters, digits and hyphens)")
    if not _SLUG_RE.match(slug):
        raise AppStoreError(
            f"invalid slug {raw!r} — use 1-60 lowercase letters, digits and "
            f"hyphens, starting and ending with a letter or digit "
            f"(e.g. 'budget-tracker')"
        )
    if slug in _RESERVED_SLUGS:
        raise AppStoreError(f"slug {slug!r} is reserved — pick another name")
    return slug


def app_path(slug: str) -> str:
    """Absolute path of ``<slug>.html``, jailed to the root.

    The slug regex already excludes ``/``, ``.`` and ``..``, so this is
    belt-and-braces — but the store runs as root inside the container and the
    slug is model-supplied, so the jail is asserted rather than assumed
    (same reasoning as ``AppManager.write_app_files``' round-10 path jail).
    """
    slug = normalise_slug(slug)
    root = os.path.realpath(apps_root())
    full = os.path.realpath(os.path.join(root, slug + HTML_SUFFIX))
    if os.path.dirname(full) != root:
        raise AppStoreError(f"refusing path outside the app root: {slug!r}")
    return full


def exists(slug: str) -> bool:
    try:
        return os.path.isfile(app_path(slug))
    except AppStoreError:
        return False


# ── HTML validation ───────────────────────────────────────────────────

_SCRIPT_SRC_RE = re.compile(
    r"""<script\b[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE
)
_LINK_HREF_RE = re.compile(
    r"""<link\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE
)
_CSS_IMPORT_RE = re.compile(
    r"""@import\s+(?:url\()?\s*["']?([^"')\s]+)""", re.IGNORECASE
)
_ABSOLUTE_URL_RE = re.compile(r"^(?:https?:)?//([^/?#]+)", re.IGNORECASE)


def _external_host(url: str) -> Optional[str]:
    """Host of an absolute/protocol-relative URL, else None (relative/inline)."""
    m = _ABSOLUTE_URL_RE.match(url.strip())
    if not m:
        return None
    return m.group(1).split("@")[-1].split(":")[0].lower()


def validate_html(html: str) -> List[str]:
    """Raise on anything fatal; return a list of non-fatal warnings.

    Fatal (raises :class:`AppStoreError`):
      * empty / stub-sized / oversized content
      * not actually an HTML document
      * a ``<script src>``, stylesheet ``<link href>`` or CSS ``@import``
        pointing at an origin the artifact CSP will not load

    That last one is the important case. Under the Phase-2 CSP those
    requests are *blocked*, so shipping them produces a blank page and a
    console error the model never sees — it would report success on a
    broken app. Failing at write time turns a silent runtime break into an
    error string the model can act on.
    """
    if html is None:
        raise AppStoreError("html is required")
    text = html if isinstance(html, str) else str(html)
    raw = text.encode("utf-8")

    if not text.strip():
        raise AppStoreError(
            "html is empty — write the complete single-file app, not a placeholder"
        )
    if len(raw) < MIN_HTML_BYTES:
        raise AppStoreError(
            f"html is only {len(raw)} bytes — that is a stub, not an app. "
            f"Write the complete document (a real app is thousands of bytes: "
            f"doctype, head with inline <style>, body markup, inline <script>)."
        )
    if len(raw) > MAX_HTML_BYTES:
        raise AppStoreError(
            f"html is {len(raw)} bytes, over the {MAX_HTML_BYTES} byte limit. "
            f"Trim embedded data URIs or split the feature set."
        )

    low = text.lower()
    if "<!doctype" not in low and "<html" not in low:
        raise AppStoreError(
            "html does not look like a document — it must start with "
            "<!doctype html> and contain <html>, <head> and <body>"
        )
    if "<body" not in low:
        raise AppStoreError("html has no <body> element")

    warnings: List[str] = []
    for regex, what in (
        (_SCRIPT_SRC_RE, "<script src>"),
        (_CSS_IMPORT_RE, "CSS @import"),
    ):
        for url in regex.findall(text):
            host = _external_host(url)
            if host and host not in ALLOWED_CDN_HOSTS:
                raise AppStoreError(
                    f"{what} loads from {host!r}, which the artifact sandbox "
                    f"blocks. Only https://cdnjs.cloudflare.com is allowed — "
                    f"inline the code or switch to the cdnjs copy."
                )

    for url in _LINK_HREF_RE.findall(text):
        host = _external_host(url)
        if not host or host in ALLOWED_CDN_HOSTS:
            continue
        # Only stylesheet links are fatal; <link rel="canonical"> and friends
        # are inert metadata, not a blocked fetch.
        if re.search(
            r"""<link\b[^>]*href\s*=\s*["']%s["'][^>]*rel\s*=\s*["'][^"']*stylesheet"""
            % re.escape(url),
            text, re.IGNORECASE,
        ) or re.search(
            r"""<link\b[^>]*rel\s*=\s*["'][^"']*stylesheet[^"']*["'][^>]*href\s*=\s*["']%s["']"""
            % re.escape(url),
            text, re.IGNORECASE,
        ):
            raise AppStoreError(
                f"stylesheet <link> loads from {host!r}, which the artifact "
                f"sandbox blocks. Only https://cdnjs.cloudflare.com is allowed."
            )
        warnings.append(f"<link> to {host} will not load in the sandbox")

    # There is deliberately no "you used localStorage without a try/catch"
    # warning any more. The old one tested `"try" not in html` — one `try`
    # anywhere in a 40 KB document silenced it, so in practice it fired never,
    # and when it did fire it asked the model to hand-write a guard on every
    # storage call forever. Round 18 closes the class at the other end
    # instead: `runtime.storage_shim` replaces localStorage, sessionStorage
    # and document.cookie with objects that cannot throw, injected at serve
    # time, so unguarded storage in generated code is no longer a way to kill
    # a page. A warning about a hazard that no longer exists is noise in a
    # tool result the model has to read.
    return warnings


# ── Manifest ──────────────────────────────────────────────────────────

@dataclass
class AppRecord:
    slug: str
    title: str
    created_at: str
    updated_at: str
    revision: int = 1
    size_bytes: int = 0
    presented_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
            "size_bytes": self.size_bytes,
            "presented_at": self.presented_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppRecord":
        return cls(
            slug=str(d.get("slug") or ""),
            title=str(d.get("title") or d.get("slug") or ""),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
            revision=int(d.get("revision") or 1),
            size_bytes=int(d.get("size_bytes") or 0),
            presented_at=d.get("presented_at"),
        )


def _atomic_write(path: str, data: bytes, *, prefix: str = ".tmp-") -> None:
    """Write ``data`` to ``path`` via a temp file in the same directory.

    On EACCES/EPERM the directory's mode is repaired and the write is retried
    ONCE — the self-heal for a workspace whose apps folder was created by a
    different uid (an older image, a restored volume). If it still fails the
    caller gets :func:`_unwritable`'s sentence, never an errno.

    A partially written file is never left behind: the temp file is unlinked
    on any failure, and ``os.replace`` is atomic, so a reader either sees the
    whole previous revision or the whole new one.
    """
    directory = os.path.dirname(path)
    for attempt in (0, 1):
        fd = tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                fd = None  # now owned by the file object
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            # BEFORE the replace, while the name is still ours: `os.replace`
            # carries the temp file's 0600 over, and a file nobody but root can
            # read is not a published app.
            try:
                os.chmod(tmp, _FILE_MODE)
            except OSError as exc:  # pragma: no cover - never worth failing a write
                logger.warning("[app_html] could not set the mode on %s: %s", tmp, exc)
            os.replace(tmp, path)
            return
        except PermissionError:
            _cleanup(fd, tmp)
            if attempt == 0 and repair_permissions(directory):
                logger.warning(
                    "[app_html] repaired permissions on %s and retrying the write",
                    directory,
                )
                continue
            raise _unwritable(directory) from None
        except OSError:
            _cleanup(fd, tmp)
            raise


def _cleanup(fd: Optional[int], tmp: Optional[str]) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    if tmp:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _manifest_path() -> str:
    return os.path.join(apps_root(), MANIFEST_NAME)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest() -> Dict[str, AppRecord]:
    """Every registered app, keyed by slug. Missing/corrupt → empty."""
    path = _manifest_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # A corrupt manifest must not brick the pipeline: the .html files ARE
        # the durable artifacts, the manifest is an index over them.
        logger.warning("[app_html] manifest unreadable (%s) — treating as empty", exc)
        return {}
    apps = data.get("apps") if isinstance(data, dict) else None
    if not isinstance(apps, dict):
        return {}
    out: Dict[str, AppRecord] = {}
    for slug, rec in apps.items():
        if isinstance(rec, dict):
            out[str(slug)] = AppRecord.from_dict({**rec, "slug": slug})
    return out


def _write_manifest(records: Dict[str, AppRecord]) -> None:
    """Atomic replace — a torn manifest is worse than a stale one."""
    ensure_root()
    payload = {
        "version": 1,
        "updated_at": _now(),
        "apps": {slug: rec.to_dict() for slug, rec in sorted(records.items())},
    }
    _atomic_write(
        _manifest_path(),
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        prefix=".manifest-",
    )


def upsert_record(slug: str, title: str, size_bytes: int, *,
                  bump_revision: bool, presented: bool = False) -> AppRecord:
    with _MANIFEST_LOCK:
        records = read_manifest()
        now = _now()
        rec = records.get(slug)
        if rec is None:
            rec = AppRecord(slug=slug, title=title, created_at=now,
                            updated_at=now, revision=1, size_bytes=size_bytes)
        else:
            rec.title = title or rec.title
            rec.updated_at = now
            rec.size_bytes = size_bytes
            if bump_revision:
                rec.revision += 1
        if presented:
            rec.presented_at = now
        records[slug] = rec
        _write_manifest(records)
        return rec


def retitle_record(slug: str, title: str) -> Optional[AppRecord]:
    """Rename an app WITHOUT touching its file, its revision or its slug.

    The title is a display name; the slug is the identity every chat card,
    every runner and the ``apps`` row key on. Renaming must therefore move
    exactly one field — a rename that bumped the revision would make every
    open runner reload, and one that moved the slug would orphan every card
    already in the thread.
    """
    clean = " ".join((title or "").split())[:120]
    if not clean:
        raise AppStoreError("a name is required")
    with _MANIFEST_LOCK:
        records = read_manifest()
        rec = records.get(slug)
        if rec is None:
            return None
        rec.title = clean
        rec.updated_at = _now()
        records[slug] = rec
        _write_manifest(records)
        return rec


def forget_record(slug: str) -> bool:
    with _MANIFEST_LOCK:
        records = read_manifest()
        if slug not in records:
            return False
        del records[slug]
        _write_manifest(records)
        return True


# ── Versions (Phase 3) ────────────────────────────────────────────────

def _versions_dir(slug: str) -> str:
    return os.path.join(apps_root(), VERSIONS_DIR, normalise_slug(slug))


def snapshot(slug: str, revision: int) -> Optional[str]:
    """Copy the CURRENT file into ``.versions/`` before it is overwritten.

    Best-effort: history is a convenience, and failing to keep it must never
    block the edit the user asked for.
    """
    try:
        src = app_path(slug)
        if not os.path.isfile(src):
            return None
        vdir = _versions_dir(slug)
        os.makedirs(vdir, exist_ok=True)
        dst = os.path.join(vdir, f"{revision:04d}-{int(time.time())}{HTML_SUFFIX}")
        shutil.copy2(src, dst)
        _prune_versions(vdir)
        return dst
    except (OSError, AppStoreError) as exc:
        logger.warning("[app_html] snapshot failed for %s: %s", slug, exc)
        return None


def _prune_versions(vdir: str) -> None:
    try:
        entries = sorted(
            e for e in os.listdir(vdir) if e.endswith(HTML_SUFFIX)
        )
        for stale in entries[:-MAX_VERSIONS_PER_APP]:
            os.unlink(os.path.join(vdir, stale))
    except OSError:
        pass


def restore_latest_version(slug: str) -> bool:
    """Put the newest kept revision back at ``<slug>.html``. True if it worked.

    The self-heal for a manifest that knows about an app whose file is not
    there: an interrupted write, a volume restored from under a running
    container, a cleanup script that took the .html and left the index. The
    alternative is telling someone their app does not exist while a copy of it
    sits in ``.versions/`` — which is not a missing app, it is a missing
    filename.

    Snapshots are named ``<revision>-<epoch>.html``, zero-padded, so the last
    entry in sorted order is the newest. Best-effort by design: a failure here
    leaves exactly the state we found.
    """
    try:
        versions = list_versions(slug)
        if not versions:
            return False
        newest = os.path.join(_versions_dir(slug), versions[-1])
        with open(newest, "rb") as fh:
            data = fh.read()
        if not data:
            return False
        ensure_root()
        _atomic_write(app_path(slug), data, prefix=f".{slug}-")
        logger.warning(
            "[app_html] %s.html was missing — restored it from %s",
            slug, versions[-1],
        )
        return True
    except (OSError, AppStoreError) as exc:
        logger.warning("[app_html] could not restore %s: %s", slug, exc)
        return False


def reconcile(slug: str, *, allow_purge: bool = True) -> str:
    """Make one manifest record agree with the disk. ``ok``/``restored``/``purged``.

    The Files page lists the MANIFEST, and every other entry point in this
    skill self-heals a missing file (`_existing_title` calls
    :func:`restore_latest_version`) while the list route did not. So a record
    whose ``.html`` went away — an interrupted write, a volume restored under a
    running container, a cleanup script that took the file and left the index —
    stayed in the list forever as a row with a name, a stale size, and nothing
    behind it. On a device that renders as "Couldn't show this item", sitting
    between two apps that work.

    Two outcomes, in this order:

    * **restored** — a snapshot exists in ``.versions/``. That is not a missing
      app, it is a missing filename, and it is the same repair the skill path
      already performs.
    * **purged** — no file and no snapshot. The record is a tombstone for bytes
      that are never coming back, and keeping it costs the user a row they can
      never open. Its state blob goes with it: slugs are reusable, so leaving
      it behind means the next app of the same name opens holding a stranger's
      saved game.
    """
    if exists(slug):
        return "ok"
    # Restore is attempted unconditionally: it only ever puts a file BACK, so
    # it cannot be the wrong call on a volume that is merely late.
    if restore_latest_version(slug):
        return "restored"
    if not allow_purge:
        return "ok"
    logger.warning(
        "[app_html] %s is in the manifest with no file and no history — "
        "removing the row", slug,
    )
    try:
        os.unlink(_state_path(slug))
    except (OSError, AppStoreError):
        pass
    try:
        os.unlink(_preview_path(slug))
    except (OSError, AppStoreError):
        pass
    try:
        from app.agent.skills.builtins.app_html import appskill, logo
        appskill.delete(slug)
        logo.delete_icon(slug)
    except Exception:  # noqa: BLE001 - cleanup must never fail the repair
        logger.debug("[app_html] companion purge failed for %s", slug, exc_info=True)
    forget_record(slug)
    return "purged"


def reconcile_all() -> Dict[str, str]:
    """Reconcile every record. Returns only the slugs that MOVED.

    Called from the list route, so the repair reaches every user the first time
    they open Files — there is no migration to run and no fleet to sweep.

    **The wholesale-absence brake is the whole safety story, and it guards the
    PURGE only.** If a volume is late mounting or half restored, every record
    looks fileless at once, and an unguarded purge would take a user's entire
    library on a transient. So: a library of several apps in which NOT ONE file
    is on disk is a storage fault, not a verdict about the apps, and nothing is
    dropped. One app missing beside others that are present is genuinely
    missing — and a single-app library is not evidence of anything either way,
    so it is treated normally rather than left permanently unrepairable.

    Restores are never gated. Putting a file back from history cannot lose
    data, so it is the right call even on a volume that is merely late.
    """
    records = read_manifest()
    if not records:
        return {}
    present = [s for s in records if exists(s)]
    wholesale = len(records) > 1 and not present
    if wholesale:
        logger.warning(
            "[app_html] all %d apps are missing their files — treating this as "
            "a storage fault: restoring what has history, dropping nothing",
            len(records),
        )
    moved: Dict[str, str] = {}
    for slug in list(records):
        try:
            outcome = reconcile(slug, allow_purge=not wholesale)
        except (OSError, AppStoreError):
            logger.debug("[app_html] reconcile failed for %s", slug, exc_info=True)
            continue
        if outcome != "ok":
            moved[slug] = outcome
    return moved


def list_versions(slug: str) -> List[str]:
    try:
        vdir = _versions_dir(slug)
        return sorted(e for e in os.listdir(vdir) if e.endswith(HTML_SUFFIX))
    except (OSError, AppStoreError):
        return []


# ── Read / write ──────────────────────────────────────────────────────

def write_app(slug: str, title: str, html: str) -> Tuple[AppRecord, List[str]]:
    """Create or replace ``<slug>.html``. Returns (record, warnings)."""
    slug = normalise_slug(slug)
    warnings = validate_html(html)
    ensure_root()

    existing = read_manifest().get(slug)
    if existing is not None:
        snapshot(slug, existing.revision)

    path = app_path(slug)
    data = html.encode("utf-8")
    _atomic_write(path, data, prefix=f".{slug}-")

    rec = upsert_record(
        slug,
        (title or slug).strip()[:200],
        len(data),
        bump_revision=existing is not None,
    )
    return rec, warnings


def read_app(slug: str) -> str:
    path = app_path(slug)
    if not os.path.isfile(path):
        known = ", ".join(sorted(read_manifest())) or "none yet"
        raise AppStoreError(
            f"no app named {slug!r}. Apps that exist: {known}. "
            f"Use create_app_file to make a new one."
        )
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def delete_app(slug: str) -> bool:
    """Remove the file, its history, its saved state and its manifest row.

    The state file is not optional cleanup. Slugs are reusable — the model
    picks them from the app's name — so a `snake` deleted today and a `snake`
    built next week are the same path, and state left behind means the new app
    opens holding a stranger's saved game. It is also the user's data for an
    app they just asked to be rid of.
    """
    removed = False
    try:
        path = app_path(slug)
        if os.path.isfile(path):
            os.unlink(path)
            removed = True
    except (OSError, AppStoreError):
        pass
    try:
        shutil.rmtree(_versions_dir(slug), ignore_errors=True)
    except AppStoreError:
        pass
    try:
        os.unlink(_state_path(slug))
    except (OSError, AppStoreError):
        pass
    # The preview is a picture of an app that no longer exists — and slugs
    # are reusable, so leaving it behind would hand next week's `snake` the
    # dead app's face until its first publish overwrites it.
    try:
        os.unlink(_preview_path(slug))
    except (OSError, AppStoreError):
        pass
    # The brief and the icon go too, and for the same reason the state does:
    # slugs are reusable. A `snake` deleted today and a `snake` built next
    # week share every path here, so a brief left behind would be read as the
    # NEW app's purpose by the first edit — the worst possible way for this
    # feature to fail, because it is confidently wrong rather than absent.
    #
    # Imported inside the function: `appskill` and `logo` both import this
    # module, and doing it at module scope would be a cycle.
    try:
        from app.agent.skills.builtins.app_html import appskill, logo
        appskill.delete(slug)
        logo.delete_icon(slug)
    except Exception:  # noqa: BLE001 - cleanup must never fail a delete
        logger.debug("[app_html] companion cleanup failed for %s", slug, exc_info=True)
    forget_record(normalise_slug(slug))
    return removed


# ── Storage bridge (Phase 3) ──────────────────────────────────────────
# An artifact runs in a sandboxed frame with an OPAQUE origin, where
# localStorage/sessionStorage/indexedDB/cookies all throw and every network
# call is blocked. So state that must survive a reload cannot be persisted by
# the app itself — it postMessages the Toup shell, and the shell (which has a
# real origin and a real credential) writes it here.
#
# JSON only, per-app, hard-capped. This is app state, not a database: an app
# that needs more than this is asking for a backend it does not have.
STATE_DIR = ".state"
MAX_STATE_BYTES = 256 * 1024


def _state_path(slug: str) -> str:
    slug = normalise_slug(slug)
    root = os.path.realpath(apps_root())
    full = os.path.realpath(os.path.join(root, STATE_DIR, slug + ".json"))
    expected = os.path.join(root, STATE_DIR)
    if os.path.dirname(full) != expected:
        raise AppStoreError(f"refusing state path outside the app root: {slug!r}")
    return full


def read_state(slug: str) -> Dict[str, Any]:
    try:
        with open(_state_path(slug), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("[app_html] state unreadable for %s: %s", slug, exc)
        return {}
    return data if isinstance(data, dict) else {}


def write_state(slug: str, state: Dict[str, Any]) -> int:
    """Replace an app's whole state blob. Returns bytes written."""
    if not isinstance(state, dict):
        raise AppStoreError("state must be a JSON object")
    path = _state_path(slug)
    payload = json.dumps(state, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_STATE_BYTES:
        raise AppStoreError(
            f"state is {len(payload)} bytes, over the {MAX_STATE_BYTES} byte "
            f"limit for one app"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write(path, payload, prefix=".state-")
    return len(payload)


def merge_state(slug: str, updates: Dict[str, Any]) -> int:
    """Set/remove individual keys. A ``None`` value deletes the key."""
    if not isinstance(updates, dict):
        raise AppStoreError("state updates must be a JSON object")
    with _MANIFEST_LOCK:
        state = read_state(slug)
        for key, value in updates.items():
            if value is None:
                state.pop(str(key), None)
            else:
                state[str(key)] = value
        return write_state(slug, state)


# ── Publish-time preview (round 23) ───────────────────────────────────
# The card's preview used to be a client-side re-render of the app under
# capture conditions that don't match the runner's timing (a WebView shot
# 900 ms after load-end photographs a React app mid-hydration — the recorded
# full-width-header / collapsed-left-column card). The publish gate already
# opens the app in a real browser at the phone viewport; the START-SCREEN
# frame from that run is the app's face, so it is persisted here and served
# as the card's preview. One mechanism, two consumers: the same browser pass
# that judges the app also photographs it for the card.
PREVIEWS_DIR = ".previews"
#: A 390×844 PNG runs 40–120 KB; anything past this is a full-bleed
#: photograph and not worth storing per app.
MAX_PREVIEW_BYTES = 3 * 1024 * 1024


def _preview_path(slug: str) -> str:
    slug = normalise_slug(slug)
    root = os.path.realpath(apps_root())
    full = os.path.realpath(os.path.join(root, PREVIEWS_DIR, slug + ".png"))
    expected = os.path.join(root, PREVIEWS_DIR)
    if os.path.dirname(full) != expected:
        raise AppStoreError(f"refusing preview path outside the app root: {slug!r}")
    return full


def write_preview(slug: str, png: bytes) -> bool:
    """Store the publish-time snapshot. Never worth failing a publish over."""
    if not png or len(png) > MAX_PREVIEW_BYTES:
        return False
    try:
        path = _preview_path(slug)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_write(path, png, prefix=".preview-")
        return True
    except (OSError, AppStoreError):
        logger.warning("[app_html] could not write the preview for %s", slug,
                       exc_info=True)
        return False


def read_preview(slug: str) -> Optional[bytes]:
    try:
        with open(_preview_path(slug), "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        return None
    except (OSError, AppStoreError) as exc:
        logger.warning("[app_html] preview unreadable for %s: %s", slug, exc)
        return None


#: (mtime, size) → etag per preview path. The LISTING calls `preview_etag`
#: once per app per request, and hashing twenty 100 KB PNGs per Files open
#: would be a self-inflicted tax; a preview only changes by being rewritten,
#: which moves its stat.
_preview_etag_cache: Dict[str, Tuple[float, int, str]] = {}


def preview_etag(slug: str) -> str:
    """Content hash of the stored preview, '' when there is none.

    The same value the preview route serves as its ETag, so a client can
    cache the picture under (slug, etag) and refetch only when a publish
    actually moved it — the icon's `icon_etag` contract, applied to the
    snapshot.
    """
    try:
        path = _preview_path(slug)
        st = os.stat(path)
    except (OSError, AppStoreError):
        return ""
    hit = _preview_etag_cache.get(path)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    png = read_preview(slug)
    if not png:
        return ""
    etag = hashlib.sha256(png).hexdigest()[:32]
    _preview_etag_cache[path] = (st.st_mtime, st.st_size, etag)
    return etag


# ── The judged screenshot (round 25) ──────────────────────────────────
# The preview above is the app's FACE — the untouched first paint, taken
# before the gate pressed anything. It is NOT the picture the visual reviewer
# was shown: that one (`verify.Report.screenshot`) is taken after the start
# control was pressed, and until this round it existed only in memory for the
# length of one publish. So a "looks right" verdict could not be checked
# afterwards by anybody, which is the one property a screenshot-backed verdict
# is supposed to have.
#
# Keyed by the SHA-256 of the exact HTML that was judged, not by slug: the
# verdict belongs to those bytes, and an app edited twice since would
# otherwise overwrite the evidence for the verdict that let it ship. An
# auditor holding the app file can find its picture with one command —
#
#     shasum -a 256 ~/apps/<slug>.html      # first 16 hex chars = the name
#     open ~/apps/.judged/<those 16>.png
#
JUDGED_DIR = ".judged"
#: Same ceiling as the preview: a 390×844 PNG is 40–120 KB.
MAX_JUDGED_BYTES = 3 * 1024 * 1024
#: How many judged screenshots the volume keeps, oldest pruned first. Twenty
#: apps rebuilt a few times each stays under ~10 MB, which is three orders of
#: magnitude below one `node_modules` — the same budget `MAX_VERSIONS_PER_APP`
#: is set against. Bounded because this directory is written by every publish
#: and read by almost none of them.
MAX_JUDGED_FILES = 60

_JUDGED_DIGEST_RE = re.compile(r"^[0-9a-f]{8,64}$")


def judged_digest(html: str) -> str:
    """The name a judged screenshot is filed under, for `html` as verified."""
    return hashlib.sha256((html or "").encode("utf-8")).hexdigest()[:16]


def judged_path(digest: str) -> str:
    digest = (digest or "").strip().lower()
    if not _JUDGED_DIGEST_RE.match(digest):
        raise AppStoreError(f"invalid judged-screenshot digest: {digest!r}")
    root = os.path.realpath(apps_root())
    full = os.path.realpath(os.path.join(root, JUDGED_DIR, digest + ".png"))
    expected = os.path.join(root, JUDGED_DIR)
    if os.path.dirname(full) != expected:
        raise AppStoreError(f"refusing judged path outside the app root: {digest!r}")
    return full


def _prune_judged(directory: str) -> None:
    """Keep the newest :data:`MAX_JUDGED_FILES`. Best-effort, never raises."""
    try:
        entries = []
        for name in os.listdir(directory):
            if not name.endswith(".png"):
                continue
            path = os.path.join(directory, name)
            try:
                entries.append((os.stat(path).st_mtime, path))
            except OSError:
                continue
        for _mtime, path in sorted(entries, reverse=True)[MAX_JUDGED_FILES:]:
            try:
                os.unlink(path)
            except OSError:
                pass
    except OSError:
        logger.debug("[app_html] judged-screenshot prune skipped", exc_info=True)


def write_judged(digest: str, png: bytes) -> Optional[str]:
    """Retain the picture the reviewer was shown. Returns the path, or None.

    Fail-open by contract, exactly like :func:`write_preview`: this is
    evidence, and evidence is never worth failing a publish over. It also does
    NOT create the app root — a process verifying HTML that was never stored
    (a test, a dry run) must not conjure a workspace tree as a side effect.
    """
    if not png or len(png) > MAX_JUDGED_BYTES:
        return None
    try:
        path = judged_path(digest)
        directory = os.path.dirname(path)
        if not os.path.isdir(os.path.dirname(directory)):
            return None  # no app root here; nothing to be beside
        os.makedirs(directory, exist_ok=True)
        _atomic_write(path, png, prefix=".judged-")
        _prune_judged(directory)
        return path
    except (OSError, AppStoreError):
        logger.warning("[app_html] could not retain the judged screenshot (%s)",
                       digest, exc_info=True)
        return None


def read_judged(digest: str) -> Optional[bytes]:
    try:
        with open(judged_path(digest), "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        return None
    except (OSError, AppStoreError) as exc:
        logger.warning("[app_html] judged screenshot unreadable (%s): %s",
                       digest, exc)
        return None


def edit_app(slug: str, old_string: str, new_string: str) -> Tuple[AppRecord, int]:
    """Exact-string replace. Returns (record, byte delta).

    Fails loudly when ``old_string`` is absent OR appears more than once —
    the two ways an exact-string edit silently does the wrong thing.
    """
    if old_string is None or old_string == "":
        raise AppStoreError(
            "old_string is required and must be non-empty — to replace the "
            "whole file, call create_app_file again instead"
        )
    if old_string == new_string:
        raise AppStoreError("old_string and new_string are identical — nothing to do")

    current = read_app(slug)
    count = current.count(old_string)
    if count == 0:
        raise AppStoreError(
            f"old_string was not found in {slug}.html. Call view_app_file "
            f"first and copy the exact text, including indentation and "
            f"line breaks."
        )
    if count > 1:
        raise AppStoreError(
            f"old_string appears {count} times in {slug}.html — it must match "
            f"exactly once. Include more surrounding lines to make it unique."
        )

    updated = current.replace(old_string, new_string, 1)
    warnings = validate_html(updated)  # raises if the edit broke the document

    rec_before = read_manifest().get(slug)
    snapshot(slug, rec_before.revision if rec_before else 1)

    path = app_path(slug)
    data = updated.encode("utf-8")
    _atomic_write(path, data, prefix=f".{slug}-")

    title = rec_before.title if rec_before else slug
    rec = upsert_record(slug, title, len(data), bump_revision=True)
    rec.warnings = warnings  # type: ignore[attr-defined]
    return rec, len(data) - len(current.encode("utf-8"))
