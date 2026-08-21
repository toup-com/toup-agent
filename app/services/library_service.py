"""The user file library — a virtual tree over the tenant's physical files.

This is the ONLY module that knows where bytes live. Everything above it
(``app/api/library.py`` on the agent, ``app/api/workspace_proxy.py`` on the
platform) speaks ids, display names and virtual folder paths.

Physical roots (all under ``settings.agent_workspace_dir`` = WS) and the
storage-key tag each maps to::

    gen:<rel>   WS/generated/<rel>      the storage backend (attachments:
                                        generate_* docs, image tools, chat
                                        uploads) + write_file's document
                                        redirect when no user scope is set
    uws:<rel>   WS/<user_id>/<rel>      the per-user workspace root (write_file
                                        redirect target, model-written docs)
    ws:<rel>    WS/<rel>                the top-level workspace (legacy
                                        model-written docs and image copies)

Everything else in WS is DENIED BY DEFAULT and never scanned: ``apps/``,
``vibecoding/``, ``toup-code/``, dotdirs (``.whatsapp_auth``, ``.dashboard``,
``.cache`` …), the bare ``<uuid>/`` dir itself, other users' storage scopes,
build trees. The allow-list is ``iter_physical_candidates``; the deny rules
that apply INSIDE an allowed root are ``is_junk`` (test artefacts, stub
PDFs, preview caches, boilerplate).

Sync (``sync_user_library``) is idempotent and cheap: it reads the tenant's
``messages.attachments`` (a few dozen rows even for heavy users) and stats a
handful of directories, inserting manifest rows for anything new and
tombstoning rows whose bytes are gone. It is throttled per user and forced
after every write through the API, so the manifest can never drift far from
disk without anyone hooking every writer.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional, Sequence

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Conversation, Message
from app.db.models.user_file import (
    FILE_ORIGINS,
    ORIGIN_AGENT,
    ORIGIN_UPLOAD,
    SYSTEM_FOLDER_APPS,
    SYSTEM_FOLDER_DOCUMENTS,
    SYSTEM_FOLDER_IMAGES,
    SYSTEM_FOLDER_UPLOADS,
    SYSTEM_FOLDERS,
    UserFile,
    UserFolder,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Classification
# ═════════════════════════════════════════════════════════════════════

KIND_DOCUMENT = "document"
KIND_IMAGE = "image"
KIND_AUDIO = "audio"
KIND_VIDEO = "video"
KIND_ARCHIVE = "archive"
#: An app the agent built — one self-contained HTML file. NOT a document,
#: even though it is .html: a document is opened to be read, an app is opened
#: to be RUN, and it can only be run inside the sandboxed artifact frame. The
#: kind is what tells a client which of the two to do.
KIND_APP = "app"
KIND_OTHER = "other"
KINDS = (KIND_DOCUMENT, KIND_IMAGE, KIND_AUDIO, KIND_VIDEO, KIND_ARCHIVE,
         KIND_APP, KIND_OTHER)

_KIND_BY_EXT: dict[str, str] = {}
for _ext in ("pdf", "doc", "docx", "odt", "rtf", "txt", "md", "markdown", "csv", "tsv",
             "xls", "xlsx", "ods", "ppt", "pptx", "odp", "key", "numbers", "pages",
             "epub", "html", "htm", "json", "xml", "yaml", "yml"):
    _KIND_BY_EXT[_ext] = KIND_DOCUMENT
for _ext in ("png", "jpg", "jpeg", "gif", "webp", "svg", "heic", "heif", "bmp",
             "tif", "tiff", "avif"):
    _KIND_BY_EXT[_ext] = KIND_IMAGE
for _ext in ("mp3", "wav", "m4a", "aac", "ogg", "oga", "flac", "opus"):
    _KIND_BY_EXT[_ext] = KIND_AUDIO
for _ext in ("mp4", "mov", "webm", "mkv", "avi", "m4v"):
    _KIND_BY_EXT[_ext] = KIND_VIDEO
for _ext in ("zip", "tar", "gz", "tgz", "bz2", "xz", "7z", "rar"):
    _KIND_BY_EXT[_ext] = KIND_ARCHIVE

# Extensions a file at a WORKSPACE ROOT (uws:/ws:) must have to count as a
# deliverable. Under generated/ everything non-junk counts; at the roots the
# model also drops scratch code, data dumps and configs, which are not
# something a user asked for.
_ROOT_DELIVERABLE_EXTS = {
    e for e, k in _KIND_BY_EXT.items()
    if k in (KIND_DOCUMENT, KIND_IMAGE, KIND_AUDIO, KIND_VIDEO, KIND_ARCHIVE)
} - {"json", "xml", "yaml", "yml"}

_MIME_OVERRIDES = {
    "md": "text/markdown", "markdown": "text/markdown",
    "webp": "image/webp", "heic": "image/heic", "heif": "image/heif", "avif": "image/avif",
    "yaml": "application/x-yaml", "yml": "application/x-yaml",
    "csv": "text/csv", "tsv": "text/tab-separated-values",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "opus": "audio/opus", "m4a": "audio/mp4", "mkv": "video/x-matroska",
}

# Text types the content routes may return in a JSON body.
_TEXT_MIMES = {
    "application/json", "application/javascript", "application/xml",
    "application/x-sh", "application/x-yaml", "application/toml", "image/svg+xml",
}


def ext_of(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    if "." not in base or base.startswith(".") and base.count(".") == 1:
        return ""
    return base.rsplit(".", 1)[-1].lower()


def kind_of(name: str, mime: str = "") -> str:
    k = _KIND_BY_EXT.get(ext_of(name))
    if k:
        return k
    m = (mime or "").lower()
    if m.startswith("image/"):
        return KIND_IMAGE
    if m.startswith("audio/"):
        return KIND_AUDIO
    if m.startswith("video/"):
        return KIND_VIDEO
    if m.startswith("text/") or m == "application/pdf":
        return KIND_DOCUMENT
    return KIND_OTHER


def kind_of_row(f) -> str:
    """Kind for a manifest ROW, which knows where its bytes live.

    An app's display name has no extension ("Nokia Snake Classic") and its
    mime is text/html, so `kind_of` alone would call it a document and every
    client would offer to READ it. The storage root is the only thing that
    still knows it is an app.
    """
    key = getattr(f, "storage_key", "") or ""
    if key.startswith(ROOT_APP + ":"):
        return KIND_APP
    return kind_of(getattr(f, "name", "") or "", getattr(f, "mime_type", "") or "")


def guess_mime(name: str, fallback: str = "") -> str:
    e = ext_of(name)
    if e in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[e]
    if fallback and fallback != "application/octet-stream":
        return fallback
    guessed, _ = mimetypes.guess_type(name)
    return guessed or fallback or "application/octet-stream"


def is_text_mime(mime: str) -> bool:
    return bool(mime) and (mime.startswith("text/") or mime in _TEXT_MIMES)


def human_size(n: int) -> str:
    n = int(n or 0)
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


# ═════════════════════════════════════════════════════════════════════
# Names — display names and validation
# ═════════════════════════════════════════════════════════════════════

# `<32 hex>_name` (uuid4().hex, what doc_generators._persist writes) and
# `<uuid>_name`. Stripped for display; the storage key keeps the prefix.
_STORAGE_PREFIX_RE = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(?=.)"
)
_HEX32_PREFIX_RE = re.compile(r"^[0-9a-f]{32}_")
_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

MAX_NAME_LEN = 200
_BAD_NAME_CHARS_RE = re.compile(r"[\x00-\x1f\x7f/\\]")


def display_name(storage_filename: str) -> str:
    """Storage filename → what the user sees ("045a…_IMG_3145.jpg" → "IMG_3145.jpg")."""
    base = storage_filename.rsplit("/", 1)[-1]
    stripped = _STORAGE_PREFIX_RE.sub("", base, count=1)
    return stripped or base


class InvalidName(ValueError):
    pass


def validate_name(name: str) -> str:
    """A display/folder name: no path separators, no control chars, no
    leading dot, no `.`/`..`, bounded length. Returns the trimmed name."""
    n = (name or "").strip()
    if not n:
        raise InvalidName("Name is required")
    if len(n) > MAX_NAME_LEN:
        raise InvalidName(f"Name is too long (max {MAX_NAME_LEN} characters)")
    if n in (".", "..") or n.startswith("."):
        raise InvalidName("Name cannot start with a dot")
    if _BAD_NAME_CHARS_RE.search(n):
        raise InvalidName("Name cannot contain slashes or control characters")
    return n


def safe_storage_filename(name: str) -> str:
    """Filename for NEW bytes we persist (uploads, file-write). Keeps the
    extension, drops anything a filesystem or a Content-Disposition
    header would trip on."""
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    base = _BAD_NAME_CHARS_RE.sub("", base)
    base = re.sub(r"[^\w.\- ()\[\]+,&@]", "_", base, flags=re.UNICODE)
    base = base.strip(" .")
    if not base:
        base = "file"
    if len(base) > 150:
        stem, dot, ext = base.rpartition(".")
        if dot and len(ext) <= 10:
            base = stem[:150 - len(ext) - 1] + "." + ext
        else:
            base = base[:150]
    return base


# ═════════════════════════════════════════════════════════════════════
# Junk — deny rules INSIDE an allowed root
# ═════════════════════════════════════════════════════════════════════

# Names harnesses and incidents have actually produced (2026-07 image e2e
# runs, the 2026-08-18 x.pdf stub, docgen smoke tests). Two tiers:
#   ALWAYS  — incident-specific artefact names; never a user's document.
#   PLACEHOLDER — bare stems the model (or a harness) uses when it has no
#     name: test.docx, x.pdf, tmp.md. A name alone cannot condemn a real
#     document (a customer tenant had fourteen 36 KB `x.docx`/`test.docx`
#     with real content, 2026-08-19 dry run), so this tier only hides a
#     file that is ALSO small or content-empty (see is_junk).
# Anchored and specific on purpose: "Q3 test results.pdf" must survive.
_JUNK_NAME_RES_ALWAYS = [
    re.compile(r"^e2e[_\-]", re.I),
    re.compile(r"[_\-]e2e([_\-.]|$)", re.I),
    re.compile(r"^hq_test", re.I),
    re.compile(r"^edited_[a-z]\.[a-z0-9]{1,5}$", re.I),
    re.compile(r"^src_apple\.", re.I),
    re.compile(r"^inbound_test", re.I),
    re.compile(r"^roundtrip", re.I),
    re.compile(r"^user_upload\.[a-z0-9]{1,5}$", re.I),
]
_JUNK_NAME_RES_PLACEHOLDER = [
    re.compile(r"^test_[a-z0-9_\-]*\.[a-z0-9]{1,5}$", re.I),
    re.compile(r"_test\.[a-z0-9]{1,5}$", re.I),
    re.compile(
        r"^(test|tests|tmp|temp|dummy|sample|foo|bar|baz|x|y|z|xx|xxx|untitled|out|output|"
        r"placeholder|example)\d*\.[a-z0-9]{1,5}$",
        re.I,
    ),
]
_PREVIEW_CACHE_SUFFIX = ".preview.pdf"
# reportlab with an empty story emits a 952-byte PDF (the x.pdf incident);
# anything a user could want is bigger.
_STUB_PDF_MAX_BYTES = 1000
# A placeholder-named file this small is a smoke test, not a document.
_PLACEHOLDER_SMALL_BYTES = 8 * 1024
# The per-user README that _ensure_workspace drops.
_BOILERPLATE_README_PREFIX = b"# Toup Agent Workspace"
_OFFICE_EXTS = ("docx", "pptx", "xlsx")
_TAG_RE = re.compile(rb"<[^>]+>")


def office_is_empty(path: str, ext: str) -> Optional[bool]:
    """True when an OOXML document carries no text at all — python-docx's
    empty skeleton is ~36 KB, so size cannot tell a stub from a report.
    Reads only the text-bearing zip entries; None = could not tell (treat
    as NOT empty — never hide what we cannot read)."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if ext == "docx":
                entries = [n for n in ("word/document.xml",) if n in names]
            elif ext == "pptx":
                entries = [n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            else:  # xlsx
                entries = [n for n in names if n == "xl/sharedStrings.xml"
                           or (n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))]
            if not entries:
                return True
            text = 0
            for n in entries[:50]:
                raw = z.read(n)
                if ext == "xlsx" and n.startswith("xl/worksheets/"):
                    # a sheet with any cell value is content
                    if b"<v>" in raw or b"<is>" in raw:
                        return False
                    continue
                text += len(_TAG_RE.sub(b"", raw).strip())
                if text >= 20:
                    return False
            return text < 20
    except Exception:
        return None


def is_junk(name: str, size: int, *, path_for_sniff: Optional[str] = None) -> bool:
    """True for a file the library must never show, even inside an allowed
    root. ``path_for_sniff`` lets the content checks (boilerplate README,
    empty office documents) look at the bytes; without it only name/size
    rules apply."""
    base = name.rsplit("/", 1)[-1]
    if not base or base.startswith("."):
        return True
    if size <= 0:
        return True
    low = base.lower()
    if low.endswith(_PREVIEW_CACHE_SUFFIX):
        return True
    shown = display_name(base)
    stem = shown.rsplit(".", 1)[0]
    ext = ext_of(shown)
    if _UUID_RE.match(stem) or _HEX32_RE.match(stem):
        return True  # a bare id is not a name
    if ext == "pdf" and size <= _STUB_PDF_MAX_BYTES:
        return True
    for rx in _JUNK_NAME_RES_ALWAYS:
        if rx.search(shown):
            return True
    if low == "readme.md" and path_for_sniff:
        try:
            with open(path_for_sniff, "rb") as f:
                if f.read(len(_BOILERPLATE_README_PREFIX)) == _BOILERPLATE_README_PREFIX:
                    return True
        except OSError:
            return True
    placeholder = any(rx.search(shown) for rx in _JUNK_NAME_RES_PLACEHOLDER)
    if placeholder and size < _PLACEHOLDER_SMALL_BYTES:
        return True
    if ext in _OFFICE_EXTS and path_for_sniff:
        # An office file with no text is a failed artefact whatever its
        # name (a docx skeleton is 36 KB). Checked for every office file,
        # not only placeholder names: the model names stubs anything.
        if office_is_empty(path_for_sniff, ext) is True:
            return True
    return False


# ═════════════════════════════════════════════════════════════════════
# Physical roots — the allow-list
# ═════════════════════════════════════════════════════════════════════

ROOT_GEN = "gen"
ROOT_UWS = "uws"
ROOT_WS = "ws"
#: The HTML-app root (`app_html.store.apps_root()`). A separate tag rather
#: than a path under `ws:` because `apps/` is in `_DENIED_DIRS` — it holds a
#: build tree, not deliverables — and must stay denied for the generic walk.
#: What belongs in the library is the ONE presented `<slug>.html` per app,
#: never `.versions/`, never `manifest.json`.
ROOT_APP = "app"
_ROOT_TAGS = (ROOT_GEN, ROOT_UWS, ROOT_WS, ROOT_APP)

# Directory names that are never descended into, wherever they appear.
_DENIED_DIRS = {"apps", "vibecoding", "toup-code", "node_modules", "__pycache__",
                "generated_previews", ".git"}
_MAX_SCAN_DEPTH = 3
_MAX_SCAN_ENTRIES = 5000


def workspace_root() -> str:
    return os.path.realpath(getattr(settings, "agent_workspace_dir", None) or "./workspace")


def apps_root() -> str:
    """The HTML-app root, asked of the store that owns it.

    Imported lazily and with a fallback: the library must keep listing
    documents on a container where the app pipeline is switched off and the
    skill was never loaded.
    """
    try:
        from app.agent.skills.builtins.app_html.store import apps_root as _r
        return os.path.realpath(_r())
    except Exception:
        return os.path.realpath(os.path.join(workspace_root(), "apps"))


def _root_dir(tag: str, user_id: str) -> str:
    ws = workspace_root()
    if tag == ROOT_GEN:
        return os.path.join(ws, "generated")
    if tag == ROOT_UWS:
        return os.path.join(ws, user_id)
    if tag == ROOT_WS:
        return ws
    if tag == ROOT_APP:
        return apps_root()
    raise ValueError(f"unknown storage root {tag!r}")


def make_key(tag: str, rel: str) -> str:
    return f"{tag}:{rel}"


def split_key(key: str) -> tuple[str, str]:
    tag, sep, rel = (key or "").partition(":")
    if not sep or tag not in _ROOT_TAGS or not rel:
        raise ValueError("malformed storage key")
    return tag, rel


def physical_path(user_id: str, key: str) -> str:
    """Absolute path for a storage key, or ValueError when the key would
    escape its root (``..``, absolute, symlink out). Never trust the key
    blindly even though we wrote it — defence in depth is the whole point."""
    tag, rel = split_key(key)
    if rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/").split("/"):
        raise ValueError("storage key escapes its root")
    base = os.path.realpath(_root_dir(tag, user_id))
    full = os.path.realpath(os.path.join(base, rel))
    if full != base and not full.startswith(base + os.sep):
        raise ValueError("storage key escapes its root")
    return full


def storage_backend_key(key: str) -> Optional[str]:
    """The file_storage backend key (relative to generated/) for a gen: key,
    else None. Used for the LibreOffice preview cache."""
    tag, rel = split_key(key)
    return rel if tag == ROOT_GEN else None


@dataclass
class _Candidate:
    key: str
    path: str
    size: int
    mtime: float
    hex_prefixed: bool
    # True = inside an allowed root but denied by scope (a hex-prefixed
    # storage file in a scope that is not ours: a harness or another
    # tenant). Yielded so a workspace COPY of it can be recognised and
    # skipped too; never imported.
    denied: bool = False


def _stat_file(path: str):
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    if not os.path.isfile(path) or os.path.islink(path):
        return None
    return st


def _iter_dir_files(dirpath: str, tag: str, rel_prefix: str, *, recursive_depth: int,
                    budget: list[int]) -> Iterator[_Candidate]:
    """Files directly in dirpath (and, if recursive_depth > 0, in non-denied
    subdirs down to that depth). Dotfiles/dotdirs skipped."""
    try:
        entries = sorted(os.scandir(dirpath), key=lambda e: e.name)
    except OSError:
        return
    for e in entries:
        if budget[0] <= 0:
            return
        if e.name.startswith("."):
            continue
        rel = f"{rel_prefix}{e.name}" if rel_prefix else e.name
        try:
            if e.is_dir(follow_symlinks=False):
                if recursive_depth > 0 and e.name not in _DENIED_DIRS:
                    yield from _iter_dir_files(e.path, tag, rel + "/",
                                               recursive_depth=recursive_depth - 1, budget=budget)
                continue
            if not e.is_file(follow_symlinks=False):
                continue
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        budget[0] -= 1
        yield _Candidate(key=make_key(tag, rel), path=e.path, size=st.st_size,
                         mtime=st.st_mtime, hex_prefixed=bool(_HEX32_PREFIX_RE.match(e.name)))


def _is_tenant_owner(user_id: str) -> bool:
    """The container's one owner (settings.user_id) owns the un-scoped roots
    too: generated/shared, generated/*, and the workspace root — a docgen
    call that ran without a user scope still produced THEIR document. Any
    other user id (dev/monolith mode) sees only its own scopes."""
    try:
        from app.services.runtime_identity import get_user_id
        owner = get_user_id() or ""
    except Exception:
        owner = ""
    owner = owner or (getattr(settings, "user_id", "") or "")
    return not owner or owner == user_id


def app_manifest() -> dict:
    """``{slug: record}`` from the HTML-app store, or ``{}``. Never raises."""
    try:
        from app.agent.skills.builtins.app_html import store as _store
        return _store.read_manifest() or {}
    except Exception:
        return {}


def app_slug_of(key: str) -> Optional[str]:
    """The app slug behind an ``app:`` storage key, else None."""
    try:
        tag, rel = split_key(key)
    except ValueError:
        return None
    if tag != ROOT_APP or not rel.endswith(".html"):
        return None
    return rel[: -len(".html")]


def app_title(slug: str) -> str:
    """The human name for an app: the title the model gave it, falling back
    to a de-slugged version. Never the slug and never the filename — those
    are addresses, not names.

    The equality check is not paranoia. ``AppRecord.from_dict`` substitutes
    the slug for a missing title, so a record can arrive here carrying
    "nokia-snake-classic" in the `title` field and look perfectly valid. The
    slug is disqualified as a name wherever it turns up.
    """
    rec = app_manifest().get(slug)
    title = (getattr(rec, "title", "") or "") if rec is not None else ""
    if not title and isinstance(rec, dict):
        title = rec.get("title") or ""
    title = title.strip()
    if not title or title == slug:
        return slug.replace("-", " ").title()
    return title


def _iter_app_candidates(budget: list[int]) -> Iterator[_Candidate]:
    """One candidate per PRESENTED app: ``<apps_root>/<slug>.html``.

    Depth 0 only, `.html` only. That excludes `manifest.json` (bookkeeping)
    and `.versions/` (history, and a dot-directory anyway) without relying on
    the junk rules to catch them later.

    Only apps the model actually called `present_app` on are listed: a
    half-written file the model abandoned mid-turn is not something the user
    asked to keep, and it would appear in Files before it appeared anywhere
    else.
    """
    root = apps_root()
    if not os.path.isdir(root):
        return
    manifest = app_manifest()
    for c in _iter_dir_files(root, ROOT_APP, "", recursive_depth=0, budget=budget):
        slug = app_slug_of(c.key)
        if not slug:
            continue
        rec = manifest.get(slug)
        presented = getattr(rec, "presented_at", None) if rec is not None else None
        if presented is None and isinstance(rec, dict):
            presented = rec.get("presented_at")
        if not presented:
            continue
        yield c


def iter_physical_candidates(user_id: str) -> Iterator[_Candidate]:
    """The allow-list. Yields every file the library MAY show (junk rules
    still apply), most-authoritative root first so dedupe keeps the
    attachment copy over a workspace copy."""
    ws = workspace_root()
    budget = [_MAX_SCAN_ENTRIES]
    owner = _is_tenant_owner(user_id)

    # 0. app:<slug>.html — the apps the agent has PRESENTED. Yielded first so
    #    that if a copy of the same bytes exists anywhere else, the app entry
    #    is the one that survives dedupe.
    if owner:
        yield from _iter_app_candidates(budget)

    # 1. gen:<user_id>/… and gen:shared/… — the storage backend scopes that
    #    are ours. Everything non-junk here is a deliverable or an upload.
    for scope in ((user_id, "shared") if owner else (user_id,)):
        d = os.path.join(ws, "generated", scope)
        if os.path.isdir(d):
            yield from _iter_dir_files(d, ROOT_GEN, f"{scope}/", recursive_depth=1, budget=budget)

    if not owner:
        # A non-owner still walks its own per-user workspace root below;
        # never the shared/top-level roots.
        uws = os.path.join(ws, user_id)
        if os.path.isdir(uws):
            gen_sub = os.path.join(uws, "generated")
            if os.path.isdir(gen_sub):
                yield from _iter_dir_files(gen_sub, ROOT_UWS, "generated/",
                                           recursive_depth=_MAX_SCAN_DEPTH - 1, budget=budget)
            for c in _iter_dir_files(uws, ROOT_UWS, "", recursive_depth=0, budget=budget):
                if ext_of(c.path) in _ROOT_DELIVERABLE_EXTS:
                    yield c
        return

    # 2. The rest of gen:. Top-level files (exec-sweep placement) and
    #    write_file subdirs (no user scope set) are real; a hex-prefixed file
    #    in any OTHER scope is another _persist scope — a harness or another
    #    tenant — and is yielded as `denied` (never imported, but remembered
    #    so its workspace copy is not shown either).
    gen_root = os.path.join(ws, "generated")
    if os.path.isdir(gen_root):
        try:
            entries = sorted(os.scandir(gen_root), key=lambda e: e.name)
        except OSError:
            entries = []
        for e in entries:
            if budget[0] <= 0:
                break
            if e.name.startswith(".") or e.name in (user_id, "shared") or e.name in _DENIED_DIRS:
                continue
            try:
                is_dir = e.is_dir(follow_symlinks=False)
                is_file = e.is_file(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                foreign = bool(_UUID_RE.match(e.name))  # another user's scope
                for c in _iter_dir_files(e.path, ROOT_GEN, f"{e.name}/",
                                         recursive_depth=_MAX_SCAN_DEPTH - 1, budget=budget):
                    if foreign or c.hex_prefixed:
                        c.denied = True
                    yield c
            elif is_file:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                budget[0] -= 1
                yield _Candidate(key=make_key(ROOT_GEN, e.name), path=e.path, size=st.st_size,
                                 mtime=st.st_mtime, hex_prefixed=bool(_HEX32_PREFIX_RE.match(e.name)))

    # 3. uws: the per-user workspace root — write_file's document redirect
    #    (generated/**) and model-written documents at its top level.
    uws = os.path.join(ws, user_id)
    if os.path.isdir(uws):
        gen_sub = os.path.join(uws, "generated")
        if os.path.isdir(gen_sub):
            yield from _iter_dir_files(gen_sub, ROOT_UWS, "generated/",
                                       recursive_depth=_MAX_SCAN_DEPTH - 1, budget=budget)
        for c in _iter_dir_files(uws, ROOT_UWS, "", recursive_depth=0, budget=budget):
            if ext_of(c.path) in _ROOT_DELIVERABLE_EXTS:
                yield c

    # 4. ws: top-level workspace files (legacy placement when no user scope
    #    was set: model-written .md reports, image-tool workspace copies).
    for c in _iter_dir_files(ws, ROOT_WS, "", recursive_depth=0, budget=budget):
        if ext_of(c.path) in _ROOT_DELIVERABLE_EXTS:
            yield c


# ═════════════════════════════════════════════════════════════════════
# Manifest sync
# ═════════════════════════════════════════════════════════════════════

# A sync is one SELECT over the handful of messages that carry attachments,
# one over the manifest, and a scandir of ~6 directories — milliseconds. The
# interval only collapses bursts (the phone lists folders + files back to
# back); a file the agent just wrote is visible on the next listing.
def _sha256_file(path: str) -> Optional[str]:
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _is_duplicate(user_id: str, c: "_Candidate", name: str, by_key: dict, hashes: dict) -> bool:
    """A candidate whose (display name, size) matches something known is a
    duplicate only if the BYTES match a live twin. A twin we cannot compare
    against (tombstoned, denied, unreadable) counts as a duplicate — the
    conservative answer, since it means "we already decided about these
    bytes". Hashing happens once per candidate, at first import only."""
    twins = [r for r in by_key.values()
             if r.name.lower() == name.lower() and int(r.size_bytes or 0) == int(c.size)]
    live = [r for r in twins if r.deleted_at is None]
    if not live:
        return True
    mine = hashes.get(c.key) or _sha256_file(c.path)
    if mine is None:
        return True
    hashes[c.key] = mine
    for r in live:
        h = hashes.get(r.storage_key)
        if h is None:
            try:
                h = _sha256_file(physical_path(user_id, r.storage_key))
            except ValueError:
                h = None
            if h is None:
                # a live row whose bytes are unreadable right now: treat as
                # the same content rather than importing a possible twin
                return True
            hashes[r.storage_key] = h
        if h == mine:
            return True
    return False


SYNC_MIN_INTERVAL_S = 2.0
_last_sync_at: dict[str, float] = {}
_sync_locks: dict[str, asyncio.Lock] = {}


def _utcnow() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def _dt_from_ts(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None, microsecond=0)


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0)


def default_system_key(origin: str, kind: str) -> str:
    # Apps first: an app is always agent-made and always an app, whatever
    # the origin heuristics would otherwise say about an .html file.
    if kind == KIND_APP:
        return SYSTEM_FOLDER_APPS
    if origin == ORIGIN_UPLOAD:
        return SYSTEM_FOLDER_UPLOADS
    if kind == KIND_IMAGE:
        return SYSTEM_FOLDER_IMAGES
    return SYSTEM_FOLDER_DOCUMENTS


def _uniquify(name: str, folder_id: Optional[str], taken: set[tuple[Optional[str], str]]) -> str:
    """"report.pdf" → "report (2).pdf" while the name is taken in that folder."""
    if (folder_id, name.lower()) not in taken:
        return name
    stem, dot, ext = name.rpartition(".")
    if not dot or not stem or len(ext) > 10:
        stem, ext = name, ""
    n = 2
    while True:
        cand = f"{stem} ({n}).{ext}" if ext else f"{stem} ({n})"
        if (folder_id, cand.lower()) not in taken:
            return cand
        n += 1


async def ensure_system_folders(db: AsyncSession, user_id: str) -> dict[str, UserFolder]:
    rows = (await db.execute(
        select(UserFolder).where(UserFolder.user_id == user_id, UserFolder.system_key.isnot(None))
    )).scalars().all()
    by_key = {r.system_key: r for r in rows if r.system_key in SYSTEM_FOLDERS}
    created = False
    for key, name in SYSTEM_FOLDERS.items():
        if key in by_key:
            continue
        # A user folder already using the name at root adopts the role
        # rather than producing a duplicate name.
        existing = (await db.execute(
            select(UserFolder).where(
                UserFolder.user_id == user_id, UserFolder.parent_id.is_(None),
                func.lower(UserFolder.name) == name.lower(),
            )
        )).scalars().first()
        if existing:
            existing.system_key = key
            by_key[key] = existing
        else:
            f = UserFolder(user_id=user_id, parent_id=None, name=name, system_key=key)
            db.add(f)
            by_key[key] = f
        created = True
    if created:
        await db.flush()
    return by_key


async def _attachment_index(db: AsyncSession, user_id: str) -> dict[str, dict]:
    """storage_path → {role, message_id, attachment_id, created_at, filename,
    mime_type, size_bytes} for every attachment on this user's messages.
    Cheap: attachments is NULL on all but the handful of rows that carry one."""
    rows = (await db.execute(
        select(Message.id, Message.role, Message.created_at, Message.attachments)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(Conversation.user_id == user_id, Message.attachments.isnot(None))
        .order_by(Message.created_at.asc())
    )).all()
    out: dict[str, dict] = {}
    for mid, role, created_at, atts in rows:
        if isinstance(atts, str):
            try:
                import json
                atts = json.loads(atts)
            except ValueError:
                continue
        if not isinstance(atts, list):
            continue
        for a in atts:
            if not isinstance(a, dict):
                continue
            sp = a.get("storage_path")
            if not sp or not isinstance(sp, str):
                continue
            # First message wins (the original turn, not a later re-attach).
            out.setdefault(sp, {
                "role": role or "", "message_id": mid, "attachment_id": a.get("id"),
                "created_at": _parse_iso(a.get("created_at")) or created_at,
                "filename": a.get("filename") or "", "mime_type": a.get("mime_type") or "",
                "size_bytes": a.get("size_bytes"),
            })
    return out


async def sync_user_library(db: AsyncSession, user_id: str, *, force: bool = False) -> dict:
    """Bring the manifest in line with disk + chat attachments. Idempotent.

    Returns counts {imported, tombstoned, restored, skipped_junk}. Throttled
    to once per SYNC_MIN_INTERVAL_S per user unless force=True; serialised
    per user so two concurrent listings cannot double-insert."""
    now = time.monotonic()
    if not force and now - _last_sync_at.get(user_id, 0.0) < SYNC_MIN_INTERVAL_S:
        return {"imported": 0, "tombstoned": 0, "restored": 0, "skipped_junk": 0, "throttled": True}
    lock = _sync_locks.setdefault(user_id, asyncio.Lock())
    async with lock:
        if not force and time.monotonic() - _last_sync_at.get(user_id, 0.0) < SYNC_MIN_INTERVAL_S:
            return {"imported": 0, "tombstoned": 0, "restored": 0, "skipped_junk": 0, "throttled": True}
        try:
            counts = await _sync_locked(db, user_id)
        finally:
            _last_sync_at[user_id] = time.monotonic()
        return counts


def invalidate_sync(user_id: str) -> None:
    _last_sync_at.pop(user_id, None)


async def _sync_locked(db: AsyncSession, user_id: str) -> dict:
    counts = {"imported": 0, "tombstoned": 0, "restored": 0, "skipped_junk": 0, "throttled": False}
    folders = await ensure_system_folders(db, user_id)
    folder_ids = set((await db.execute(
        select(UserFolder.id).where(UserFolder.user_id == user_id)
    )).scalars().all())

    existing_rows = (await db.execute(
        select(UserFile).where(UserFile.user_id == user_id)
    )).scalars().all()
    by_key: dict[str, UserFile] = {r.storage_key: r for r in existing_rows}
    # (display name lower, size) of EVERY known row, tombstoned included —
    # the image tools persist an attachment AND drop a copy at the workspace
    # root, and a copy of something deleted (or denied) must not resurface.
    known_sig: set[tuple[str, int]] = {
        (r.name.lower(), int(r.size_bytes or 0)) for r in existing_rows
    }
    # (folder_id, name lower) of live rows — names stay unique per folder so
    # the path surface can address every file.
    taken: set[tuple[Optional[str], str]] = {
        (r.folder_id, r.name.lower()) for r in existing_rows if r.deleted_at is None
    }

    att_index, candidates = await asyncio.gather(
        _attachment_index(db, user_id),
        asyncio.to_thread(lambda: list(iter_physical_candidates(user_id))),
    )

    def _placement(origin: str, kind: str, previous_folder: Optional[str]) -> Optional[str]:
        if previous_folder and previous_folder in folder_ids:
            return previous_folder
        f = folders.get(default_system_key(origin, kind))
        return f.id if f else None

    hashes: dict[str, str] = {}
    seen_keys: set[str] = set()
    for c in candidates:
        if c.key in seen_keys:
            continue
        seen_keys.add(c.key)
        tag, rel = split_key(c.key)
        base = rel.rsplit("/", 1)[-1]
        att = att_index.get(rel) if tag == ROOT_GEN else None
        if c.denied or is_junk(base, c.size, path_for_sniff=c.path):
            counts["skipped_junk"] += 1
            known_sig.add((display_name(base).lower(), int(c.size)))
            row = by_key.get(c.key)
            if row is not None and row.deleted_at is None:
                # A rule tightened after this row was imported (or a file
                # shrank to a stub): hide it from now on.
                row.deleted_at = _utcnow()
                counts["tombstoned"] += 1
            continue
        row = by_key.get(c.key)
        if row is not None:
            if row.deleted_at is not None:
                # Bytes are back (or never left) after a tombstone: only a
                # NEWER file resurrects, so a library delete stays deleted.
                if _dt_from_ts(c.mtime) > row.deleted_at:
                    row.deleted_at = None
                    row.size_bytes = c.size
                    row.modified_at = _dt_from_ts(c.mtime)
                    row.created_at = row.created_at or _dt_from_ts(c.mtime)
                    row.folder_id = _placement(row.origin, kind_of_row(row), row.folder_id)
                    row.name = _uniquify(row.name, row.folder_id, taken)
                    taken.add((row.folder_id, row.name.lower()))
                    counts["restored"] += 1
                    known_sig.add((row.name.lower(), int(c.size)))
            else:
                # Keep size/mtime honest for files edited in place (write_file
                # over an existing report, an appended log).
                mt = _dt_from_ts(c.mtime)
                if row.modified_at != mt or int(row.size_bytes or 0) != int(c.size):
                    row.modified_at = mt
                    row.size_bytes = c.size
            continue

        # New file.
        if tag == ROOT_APP:
            # An app's display name is the TITLE the model gave it ("Nokia
            # Snake Classic"), not `nokia-snake-classic.html`. The slug lives
            # on in the storage key, where the user never sees it.
            name = app_title(app_slug_of(c.key) or base)
            mime = "text/html"
            origin = ORIGIN_AGENT
            created = _dt_from_ts(c.mtime)
        elif att is not None:
            name = display_name(att.get("filename") or base)
            mime = guess_mime(name, att.get("mime_type") or "")
            origin = ORIGIN_UPLOAD if att.get("role") == "user" else ORIGIN_AGENT
            created = att.get("created_at") or _dt_from_ts(c.mtime)
        else:
            name = display_name(base)
            mime = guess_mime(name)
            origin = ORIGIN_AGENT
            created = _dt_from_ts(c.mtime)
        kind = KIND_APP if tag == ROOT_APP else kind_of(name, mime)
        if tag != ROOT_APP and (name.lower(), int(c.size)) in known_sig and _is_duplicate(user_id, c, name, by_key, hashes):
            # The same bytes again — the image tools drop a workspace copy
            # next to the attachment, Telegram/routine paths persist the same
            # generation repeatedly, a deleted or denied file has a twin at
            # the root. One library entry per distinct content.
            counts["skipped_junk"] += 1
            continue
        folder_id = _placement(origin, kind, None)
        name = _uniquify(name, folder_id, taken)
        row = UserFile(
            user_id=user_id,
            folder_id=folder_id,
            name=name,
            mime_type=mime,
            size_bytes=int(c.size),
            storage_key=c.key,
            origin=origin,
            source_message_id=(att or {}).get("message_id"),
            source_attachment_id=(att or {}).get("attachment_id"),
            created_at=created if isinstance(created, datetime) else _dt_from_ts(c.mtime),
            modified_at=_dt_from_ts(c.mtime),
        )
        db.add(row)
        by_key[c.key] = row
        known_sig.add((name.lower(), int(c.size)))
        taken.add((folder_id, name.lower()))
        counts["imported"] += 1

    # Tombstone live rows whose bytes are gone (deleted by the agent's own
    # shell, a workspace wipe, …).
    for key, row in by_key.items():
        if row.deleted_at is not None:
            continue
        if key in seen_keys:
            continue
        try:
            p = physical_path(user_id, key)
        except ValueError:
            p = ""
        if not p or _stat_file(p) is None:
            row.deleted_at = _utcnow()
            counts["tombstoned"] += 1

    try:
        await db.commit()
    except IntegrityError:
        # A concurrent writer (another process) inserted the same key first.
        await db.rollback()
        logger.info("[library] sync lost an insert race for user=%s; will converge next pass", user_id)
        counts["imported"] = 0
    return counts


# ═════════════════════════════════════════════════════════════════════
# Folders
# ═════════════════════════════════════════════════════════════════════

class NameConflict(ValueError):
    pass


class NotEmpty(ValueError):
    pass


class SystemFolderError(ValueError):
    pass


async def get_folder(db: AsyncSession, user_id: str, folder_id: str) -> Optional[UserFolder]:
    if not folder_id:
        return None
    return (await db.execute(
        select(UserFolder).where(UserFolder.user_id == user_id, UserFolder.id == folder_id)
    )).scalars().first()


async def _child_folder_by_name(db: AsyncSession, user_id: str, parent_id: Optional[str],
                                name: str) -> Optional[UserFolder]:
    q = select(UserFolder).where(UserFolder.user_id == user_id, func.lower(UserFolder.name) == name.lower())
    q = q.where(UserFolder.parent_id.is_(None)) if parent_id is None else q.where(UserFolder.parent_id == parent_id)
    return (await db.execute(q)).scalars().first()


async def _child_file_by_name(db: AsyncSession, user_id: str, folder_id: Optional[str],
                              name: str) -> Optional[UserFile]:
    q = select(UserFile).where(
        UserFile.user_id == user_id, UserFile.deleted_at.is_(None),
        func.lower(UserFile.name) == name.lower(),
    )
    q = q.where(UserFile.folder_id.is_(None)) if folder_id is None else q.where(UserFile.folder_id == folder_id)
    return (await db.execute(q)).scalars().first()


async def _assert_free_name(db: AsyncSession, user_id: str, parent_id: Optional[str], name: str,
                            *, ignore_folder: Optional[str] = None, ignore_file: Optional[str] = None):
    f = await _child_folder_by_name(db, user_id, parent_id, name)
    if f and f.id != ignore_folder:
        raise NameConflict(f"A folder named '{name}' already exists here")
    x = await _child_file_by_name(db, user_id, parent_id, name)
    if x and x.id != ignore_file:
        raise NameConflict(f"A file named '{name}' already exists here")


async def list_folders(db: AsyncSession, user_id: str, parent_id: Optional[str]) -> list[UserFolder]:
    q = select(UserFolder).where(UserFolder.user_id == user_id)
    q = q.where(UserFolder.parent_id.is_(None)) if parent_id is None else q.where(UserFolder.parent_id == parent_id)
    rows = (await db.execute(q)).scalars().all()
    # System folders first, in SYSTEM_FOLDERS order, then user folders A→Z.
    order = {k: i for i, k in enumerate(SYSTEM_FOLDERS)}
    return sorted(rows, key=lambda f: (0 if f.system_key in order else 1,
                                       order.get(f.system_key, 99), f.name.lower()))


async def all_folders(db: AsyncSession, user_id: str) -> dict[str, UserFolder]:
    rows = (await db.execute(select(UserFolder).where(UserFolder.user_id == user_id))).scalars().all()
    return {r.id: r for r in rows}


def folder_path(folder: Optional[UserFolder], folders: dict[str, UserFolder]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    cur = folder
    while cur is not None and cur.id not in seen:
        seen.add(cur.id)
        parts.append(cur.name)
        cur = folders.get(cur.parent_id) if cur.parent_id else None
    return "/".join(reversed(parts))


async def create_folder(db: AsyncSession, user_id: str, name: str,
                        parent_id: Optional[str]) -> UserFolder:
    name = validate_name(name)
    if parent_id:
        parent = await get_folder(db, user_id, parent_id)
        if parent is None:
            raise LookupError("Parent folder not found")
    await _assert_free_name(db, user_id, parent_id or None, name)
    f = UserFolder(user_id=user_id, parent_id=parent_id or None, name=name)
    db.add(f)
    await db.commit()
    await db.refresh(f)
    return f


async def _is_descendant(db: AsyncSession, user_id: str, folder_id: str, ancestor_id: str) -> bool:
    folders = await all_folders(db, user_id)
    cur = folders.get(folder_id)
    hops = 0
    while cur is not None and hops < 1000:
        if cur.id == ancestor_id:
            return True
        cur = folders.get(cur.parent_id) if cur.parent_id else None
        hops += 1
    return False


async def update_folder(db: AsyncSession, user_id: str, folder: UserFolder, *,
                        name: Optional[str] = None, parent_id: Optional[str] = None,
                        move: bool = False) -> UserFolder:
    """Rename and/or move (move=True with parent_id=None means "to root")."""
    new_name = validate_name(name) if name is not None else folder.name
    new_parent = folder.parent_id
    if move:
        if folder.system_key:
            raise SystemFolderError("System folders stay at the top level")
        new_parent = parent_id or None
        if new_parent:
            if new_parent == folder.id or await _is_descendant(db, user_id, new_parent, folder.id):
                raise ValueError("Cannot move a folder into itself")
            if await get_folder(db, user_id, new_parent) is None:
                raise LookupError("Destination folder not found")
    if new_name.lower() != folder.name.lower() or new_parent != folder.parent_id:
        await _assert_free_name(db, user_id, new_parent, new_name, ignore_folder=folder.id)
    folder.name = new_name
    folder.parent_id = new_parent
    folder.updated_at = _utcnow()
    await db.commit()
    await db.refresh(folder)
    return folder


async def folder_counts(db: AsyncSession, user_id: str) -> tuple[dict[Optional[str], int], dict[Optional[str], int]]:
    """(files per folder_id, folders per parent_id) for live rows."""
    files = dict((await db.execute(
        select(UserFile.folder_id, func.count()).where(
            UserFile.user_id == user_id, UserFile.deleted_at.is_(None)
        ).group_by(UserFile.folder_id)
    )).all())
    subs = dict((await db.execute(
        select(UserFolder.parent_id, func.count()).where(UserFolder.user_id == user_id)
        .group_by(UserFolder.parent_id)
    )).all())
    return files, subs


async def delete_folder(db: AsyncSession, user_id: str, folder: UserFolder, *, recursive: bool) -> int:
    """Delete a folder. Non-empty needs recursive=True (bytes of every file
    inside are removed). Returns the number of files deleted."""
    if folder.system_key:
        raise SystemFolderError("System folders cannot be deleted")
    folders = await all_folders(db, user_id)
    subtree = [folder.id]
    i = 0
    while i < len(subtree):
        fid = subtree[i]
        i += 1
        subtree.extend(f.id for f in folders.values() if f.parent_id == fid)
    files = (await db.execute(
        select(UserFile).where(UserFile.user_id == user_id, UserFile.deleted_at.is_(None),
                               UserFile.folder_id.in_(subtree))
    )).scalars().all()
    if (files or len(subtree) > 1) and not recursive:
        raise NotEmpty("Folder is not empty")
    n = 0
    for f in files:
        _unlink_quiet(user_id, f.storage_key)
        f.deleted_at = _utcnow()
        n += 1
    for fid in subtree:
        row = folders.get(fid)
        if row is not None:
            await db.delete(row)
    await db.commit()
    return n


# ═════════════════════════════════════════════════════════════════════
# Files
# ═════════════════════════════════════════════════════════════════════

FOLDER_ROOT = "root"
FOLDER_ALL = "all"
SORT_FIELDS = {"name", "modified", "size", "created"}


async def get_file(db: AsyncSession, user_id: str, file_id: str) -> Optional[UserFile]:
    if not file_id:
        return None
    return (await db.execute(
        select(UserFile).where(UserFile.user_id == user_id, UserFile.id == file_id,
                               UserFile.deleted_at.is_(None))
    )).scalars().first()


async def list_files(db: AsyncSession, user_id: str, *, folder: Optional[str] = FOLDER_ROOT,
                     q: str = "", kind: str = "", origin: str = "", sort: str = "name",
                     order: str = "asc", offset: int = 0, limit: int = 50) -> tuple[list[UserFile], int]:
    """(rows, total). folder = an id, "root" (folder_id IS NULL) or "all"."""
    base = select(UserFile).where(UserFile.user_id == user_id, UserFile.deleted_at.is_(None))
    if q:
        base = base.where(UserFile.name.ilike(f"%{_escape_like(q)}%", escape="\\"))
    if folder in (FOLDER_ROOT, None, ""):
        if not q:
            base = base.where(UserFile.folder_id.is_(None))
    elif folder != FOLDER_ALL:
        base = base.where(UserFile.folder_id == folder)
    if origin in FILE_ORIGINS:
        base = base.where(UserFile.origin == origin)
    if kind in KINDS:
        base = base.where(_kind_filter(kind))

    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0
    col = {
        "name": func.lower(UserFile.name), "modified": UserFile.modified_at,
        "size": UserFile.size_bytes, "created": UserFile.created_at,
    }.get(sort, func.lower(UserFile.name))
    ordered = base.order_by(col.desc() if order == "desc" else col.asc(), UserFile.id.asc())
    rows = (await db.execute(ordered.offset(max(0, offset)).limit(max(1, min(limit, 200))))).scalars().all()
    return list(rows), int(total)


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


#: An app row in SQL. Mirrors `kind_of_row`: the storage ROOT is the only
#: thing that knows an app is an app, because its display name has no
#: extension and its mime is the same text/html a saved web page would have.
_IS_APP_SQL = UserFile.storage_key.like(f"{ROOT_APP}:%")


def _kind_filter(kind: str):
    """SQL predicate mirroring kind_of_row(): storage root first, then
    extension, then MIME family."""
    if kind == KIND_APP:
        return _IS_APP_SQL
    exts = [e for e, k in _KIND_BY_EXT.items() if k == kind]
    conds = [func.lower(UserFile.name).like(f"%.{e}") for e in exts]
    prefix = {KIND_IMAGE: "image/", KIND_AUDIO: "audio/", KIND_VIDEO: "video/"}.get(kind)
    if prefix:
        conds.append(UserFile.mime_type.like(f"{prefix}%"))
    if kind == KIND_OTHER:
        # Without the app exclusion this bucket SWALLOWS every app: a name
        # with no extension and a text/html mime is neither a known ext nor
        # an image/audio/video, so "Other" would quietly be where the user's
        # apps went.
        known = [func.lower(UserFile.name).like(f"%.{e}") for e in _KIND_BY_EXT]
        return and_(~or_(*known), ~UserFile.mime_type.like("image/%"),
                    ~UserFile.mime_type.like("audio/%"), ~UserFile.mime_type.like("video/%"),
                    ~_IS_APP_SQL)
    # Every other kind is extension- or MIME-derived, and an app matches
    # neither — but `document` would catch one if its title ever ended in a
    # known extension ("Budget.xlsx" as an app NAME is legal). Exclude
    # explicitly rather than relying on names.
    return and_(or_(*conds), ~_IS_APP_SQL)


async def _rename_app_everywhere(db: AsyncSession, slug: str, title: str) -> None:
    """An app renamed in Files is renamed in the chat card and the Apps list
    too — one app, one name. Best-effort: a rename that lands in the library
    and nowhere else is still better than a rename that fails."""
    try:
        from app.agent.skills.builtins.app_html import store as _store
        rec = _store.read_manifest().get(slug)
        if rec is not None:
            _store.upsert_record(slug, title, int(getattr(rec, "size_bytes", 0) or 0),
                                 bump_revision=False)
    except Exception:
        logger.warning("[library] app manifest rename failed for %s", slug, exc_info=True)
    try:
        from app.db.models import App
        row = (await db.execute(select(App).where(App.slug == slug))).scalar_one_or_none()
        if row is not None:
            row.name = title
    except Exception:
        logger.warning("[library] app row rename failed for %s", slug, exc_info=True)


async def update_file(db: AsyncSession, user_id: str, f: UserFile, *, name: Optional[str] = None,
                      folder_id: Optional[str] = None, move: bool = False) -> UserFile:
    """Rename and/or move. The bytes never move: only the manifest changes,
    so chat attachment pointers stay valid."""
    new_name = validate_name(name) if name is not None else f.name
    new_folder = f.folder_id
    if move:
        new_folder = folder_id or None
        if new_folder and await get_folder(db, user_id, new_folder) is None:
            raise LookupError("Destination folder not found")
    if new_name.lower() != f.name.lower() or new_folder != f.folder_id:
        await _assert_free_name(db, user_id, new_folder, new_name, ignore_file=f.id)
    renamed = new_name != f.name
    f.name = new_name
    f.folder_id = new_folder
    slug = app_slug_of(f.storage_key)
    if renamed and slug:
        await _rename_app_everywhere(db, slug, new_name)
    await db.commit()
    await db.refresh(f)
    return f


def _unlink_quiet(user_id: str, key: str) -> bool:
    try:
        p = physical_path(user_id, key)
    except ValueError:
        return False
    try:
        os.unlink(p)
    except FileNotFoundError:
        return True
    except OSError as e:
        logger.warning("[library] could not delete bytes for %s: %s", key, e)
        return False
    # Drop the LibreOffice preview cache that sits next to a DOCX/PPTX.
    try:
        cache = p + _PREVIEW_CACHE_SUFFIX
        if os.path.isfile(cache):
            os.unlink(cache)
    except OSError:
        pass
    return True


async def _forget_app(db: AsyncSession, slug: str) -> None:
    """Retire an app everywhere the moment its file is deleted from Files.

    A half-delete is worse than no delete: the bytes would be gone while the
    Apps list still offered it and `/api/artifacts/<slug>` still answered
    from the manifest, so "open" would produce an error the user cannot
    explain. Best-effort, and the library row is tombstoned either way.
    """
    try:
        from app.agent.skills.builtins.app_html import store as _store
        # Takes the revision history with it. `delete_file` has already
        # unlinked the current file; this call is idempotent about that.
        _store.delete_app(slug)
    except Exception:
        logger.warning("[library] app manifest delete failed for %s", slug, exc_info=True)
    try:
        from app.db.models import App
        row = (await db.execute(select(App).where(App.slug == slug))).scalar_one_or_none()
        if row is not None:
            row.status = "deleted"
    except Exception:
        logger.warning("[library] app row retire failed for %s", slug, exc_info=True)


async def delete_file(db: AsyncSession, user_id: str, f: UserFile) -> None:
    """Remove the bytes, then tombstone. If the bytes cannot be removed the
    row stays live and the caller sees an error — never a ghost row that
    hides a file still on disk."""
    try:
        p = physical_path(user_id, f.storage_key)
    except ValueError:
        p = ""
    if p and os.path.isfile(p):
        os.unlink(p)
        try:
            cache = p + _PREVIEW_CACHE_SUFFIX
            if os.path.isfile(cache):
                os.unlink(cache)
        except OSError:
            pass
    slug = app_slug_of(f.storage_key)
    if slug:
        await _forget_app(db, slug)
    f.deleted_at = _utcnow()
    await db.commit()


def file_abspath(user_id: str, f: UserFile) -> Optional[str]:
    """Absolute path of a live file's bytes, or None when unresolvable/missing."""
    try:
        p = physical_path(user_id, f.storage_key)
    except ValueError:
        return None
    return p if os.path.isfile(p) else None


async def store_new_file(db: AsyncSession, user_id: str, *, name: str, data: bytes,
                         mime: str = "", origin: str = ORIGIN_UPLOAD,
                         folder_id: Optional[str] = None) -> UserFile:
    """Persist NEW bytes under the storage backend (gen:<user_id>/<hex>_<name>)
    and register them. Used by upload and file-write-create."""
    if origin not in FILE_ORIGINS:
        origin = ORIGIN_UPLOAD
    shown = validate_name(name)
    if folder_id and await get_folder(db, user_id, folder_id) is None:
        raise LookupError("Destination folder not found")
    await _assert_free_name(db, user_id, folder_id or None, shown)
    from app.services.file_storage import get_storage_backend
    backend = get_storage_backend()
    stored = f"{user_id}/{uuid.uuid4().hex}_{safe_storage_filename(shown)}"
    await backend.put(stored, data)
    now = _utcnow()
    row = UserFile(
        user_id=user_id, folder_id=folder_id or None, name=shown,
        mime_type=guess_mime(shown, mime), size_bytes=len(data),
        storage_key=make_key(ROOT_GEN, stored), origin=origin,
        created_at=now, modified_at=now,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def overwrite_file_bytes(db: AsyncSession, user_id: str, f: UserFile, data: bytes) -> UserFile:
    p = None
    try:
        p = physical_path(user_id, f.storage_key)
    except ValueError:
        pass
    if not p:
        raise LookupError("File is not available")
    from app.services.workspace_perms import share_path, shared_makedirs
    shared_makedirs(os.path.dirname(p))
    with open(p, "wb") as fh:
        fh.write(data)
    share_path(p)
    f.size_bytes = len(data)
    f.modified_at = _utcnow()
    await db.commit()
    await db.refresh(f)
    return f


# ═════════════════════════════════════════════════════════════════════
# Virtual paths (compat surface: /api/workspace/*)
# ═════════════════════════════════════════════════════════════════════

class PathResolution:
    __slots__ = ("kind", "folder", "file", "path")

    def __init__(self, kind: str, folder: Optional[UserFolder] = None,
                 file: Optional[UserFile] = None, path: str = ""):
        self.kind = kind  # "root" | "folder" | "file"
        self.folder = folder
        self.file = file
        self.path = path


def split_virtual_path(path: str) -> list[str]:
    """Normalise a client path into name segments; ValueError on anything
    that smells like traversal. Empty list = root."""
    raw = (path or "").replace("\\", "/").strip()
    parts = [p.strip() for p in raw.split("/") if p.strip()]
    for p in parts:
        if p in (".", "..") or "\x00" in p:
            raise ValueError("invalid path")
    return parts


async def resolve_virtual_path(db: AsyncSession, user_id: str, path: str) -> Optional[PathResolution]:
    """Name path → root / folder / file (all scoped to user). None if not found."""
    try:
        parts = split_virtual_path(path)
    except ValueError:
        return None
    if not parts:
        return PathResolution("root", path="")
    parent: Optional[UserFolder] = None
    for i, seg in enumerate(parts):
        last = i == len(parts) - 1
        folder = await _child_folder_by_name(db, user_id, parent.id if parent else None, seg)
        if folder is not None:
            parent = folder
            if last:
                return PathResolution("folder", folder=folder, path="/".join(parts))
            continue
        if last:
            f = await _child_file_by_name(db, user_id, parent.id if parent else None, seg)
            if f is not None:
                return PathResolution("file", folder=parent, file=f, path="/".join(parts))
        return None
    return None


async def ensure_folder_path(db: AsyncSession, user_id: str, parts: Sequence[str]) -> Optional[UserFolder]:
    """mkdir -p over virtual folders. Returns the leaf (None for root).
    Raises NameConflict if a segment is an existing FILE."""
    parent: Optional[UserFolder] = None
    for seg in parts:
        seg = validate_name(seg)
        folder = await _child_folder_by_name(db, user_id, parent.id if parent else None, seg)
        if folder is None:
            if await _child_file_by_name(db, user_id, parent.id if parent else None, seg) is not None:
                raise NameConflict(f"'{seg}' is a file, not a folder")
            folder = UserFolder(user_id=user_id, parent_id=parent.id if parent else None, name=seg)
            db.add(folder)
            await db.flush()
        parent = folder
    return parent


async def resolve_legacy_physical_path(db: AsyncSession, user_id: str, path: str) -> Optional[UserFile]:
    """Old ``toup://report?path=<workspace-relative>`` links named PHYSICAL
    locations (``generated/shared/<hex>_x.md``, ``notes.md``, …). Map one
    onto a manifest row without touching the filesystem: the path is only
    ever compared against keys we already own."""
    raw = (path or "").replace("\\", "/").strip().strip("/")
    if not raw:
        return None
    parts = raw.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    candidates: list[str] = []
    if parts[0] == "generated" and len(parts) > 1:
        candidates.append(make_key(ROOT_GEN, "/".join(parts[1:])))
    if parts[0] == user_id and len(parts) > 1:
        candidates.append(make_key(ROOT_UWS, "/".join(parts[1:])))
    candidates.append(make_key(ROOT_UWS, raw))
    candidates.append(make_key(ROOT_WS, raw))
    rows = (await db.execute(
        select(UserFile).where(UserFile.user_id == user_id, UserFile.deleted_at.is_(None),
                               UserFile.storage_key.in_(candidates))
    )).scalars().all()
    if not rows:
        return None
    rank = {k: i for i, k in enumerate(candidates)}
    return sorted(rows, key=lambda r: rank.get(r.storage_key, 99))[0]


# ═════════════════════════════════════════════════════════════════════
# Serialisation — the ONLY shapes that leave this layer
# ═════════════════════════════════════════════════════════════════════

def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def file_entry(f: UserFile, path: str, *, api_prefix: str = "/api") -> dict:
    kind = kind_of_row(f)
    entry = {
        "id": f.id,
        "name": f.name,
        "ext": ext_of(f.name),
        "kind": kind,
        "mime": f.mime_type,
        "size": int(f.size_bytes or 0),
        "size_label": human_size(f.size_bytes or 0),
        "modified": _iso(f.modified_at),
        "created": _iso(f.created_at),
        "origin": f.origin,
        "folder_id": f.folder_id,
        "path": path,
        "download_url": f"{api_prefix}/library/files/{f.id}/download",
        "preview_url": f"{api_prefix}/library/files/{f.id}/preview",
    }
    if kind == KIND_APP:
        # `app_slug` is the handle the client exchanges for a slug-scoped
        # artifact token (`POST /api/artifacts/{slug}/token`) and renders in
        # the sandboxed frame. It is NOT for display — `name` is the name,
        # and no client shows a path, an id or a slug for an app.
        #
        # It has to be the slug specifically: the artifact route is the ONLY
        # way these bytes may be opened. Serving them through the ordinary
        # preview/download routes would put model-authored script on an
        # origin that holds the account's session.
        slug = app_slug_of(f.storage_key)
        if slug:
            entry["app_slug"] = slug
            entry["preview_url"] = None
    return entry


def folder_entry(fo: UserFolder, path: str, *, file_count: int = 0, folder_count: int = 0) -> dict:
    return {
        "id": fo.id,
        "name": fo.name,
        "parent_id": fo.parent_id,
        "system": fo.system_key,
        "path": path,
        "file_count": int(file_count),
        "folder_count": int(folder_count),
        "modified": _iso(fo.updated_at),
        "created": _iso(fo.created_at),
    }


def compat_dir_entry(fo: UserFolder, path: str) -> dict:
    """The /api/workspace/files row shape the shipped mobile app renders."""
    return {
        "name": fo.name, "type": "dir", "size": 0, "modified": _iso(fo.updated_at),
        "id": fo.id, "mime": None, "kind": "folder", "size_label": "", "origin": None,
        "path": path, "system": fo.system_key,
    }


def compat_file_entry(f: UserFile, path: str) -> dict:
    e = file_entry(f, path)
    row = {
        "name": f.name, "type": "file", "size": e["size"], "modified": e["modified"],
        "id": f.id, "mime": f.mime_type, "kind": e["kind"], "size_label": e["size_label"],
        "origin": f.origin, "path": path, "system": None,
    }
    # An app is openable only through the artifact frame, so the handle for
    # that has to reach the path-based surface too — the shipped mobile
    # client reads this shape, not `file_entry`.
    if e.get("app_slug"):
        row["app_slug"] = e["app_slug"]
    return row
