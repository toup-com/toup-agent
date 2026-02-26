"""
Find Tool — Dedicated file/directory finder for the workspace.

Provides structured file finding with glob patterns, type filtering,
size limits, modification time, and sorting.

Usage:
    from app.agent.find_tool import FindTool

    tool = FindTool(workspace_root="/path/to/workspace")
    results = tool.find("*.py")
    results = tool.find("*.log", min_size=1024, sort_by="size")
"""

import fnmatch
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FileType(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    ANY = "any"


class SortBy(str, Enum):
    NAME = "name"
    SIZE = "size"
    MODIFIED = "modified"
    PATH = "path"


@dataclass
class FileEntry:
    """A found file/directory entry."""
    path: str
    name: str
    file_type: str  # "file" or "directory"
    size: int = 0
    modified: float = 0.0
    extension: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "type": self.file_type,
            "size": self.size,
            "extension": self.extension,
        }

    def format(self) -> str:
        if self.file_type == "directory":
            return f"  {self.path}/"
        size_str = self._human_size(self.size)
        return f"  {self.path} ({size_str})"

    @staticmethod
    def _human_size(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


@dataclass
class FindResult:
    """Result of a find operation."""
    pattern: str
    entries: List[FileEntry] = field(default_factory=list)
    total_found: int = 0
    total_size: int = 0
    duration_ms: float = 0.0
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "total_found": self.total_found,
            "total_size": self.total_size,
            "truncated": self.truncated,
            "entries": [e.to_dict() for e in self.entries],
        }

    def format(self) -> str:
        lines = [f"Found {self.total_found} items matching '{self.pattern}'"]
        for e in self.entries:
            lines.append(e.format())
        if self.truncated:
            lines.append(f"... (truncated)")
        return "\n".join(lines)


DEFAULT_IGNORE = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".eggs", ".DS_Store",
}


class FindTool:
    """
    File and directory finder for the workspace.

    Searches for files by name pattern with filtering and sorting.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        max_results: int = 500,
    ):
        self._root = os.path.abspath(workspace_root)
        self._max_results = max_results

    def find(
        self,
        pattern: str = "*",
        *,
        file_type: FileType = FileType.ANY,
        search_path: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        extension: Optional[str] = None,
        sort_by: SortBy = SortBy.PATH,
        reverse: bool = False,
        max_results: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> FindResult:
        """
        Find files and directories matching a pattern.

        Args:
            pattern: Glob pattern to match file names.
            file_type: Filter by file or directory.
            search_path: Subdirectory to search in.
            min_size: Minimum file size in bytes.
            max_size: Maximum file size in bytes.
            extension: Filter by file extension (e.g., ".py").
            sort_by: Sort results by name, size, modified, or path.
            reverse: Reverse sort order.
            max_results: Maximum results to return.
            max_depth: Maximum directory depth.
        """
        t0 = time.time()
        limit = max_results or self._max_results
        root = os.path.join(self._root, search_path) if search_path else self._root

        entries: List[FileEntry] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip ignored dirs
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORE]

            # Check depth
            if max_depth is not None:
                depth = dirpath[len(root):].count(os.sep)
                if depth >= max_depth:
                    dirnames.clear()
                    continue

            rel_dir = os.path.relpath(dirpath, self._root)

            # Check directories
            if file_type in (FileType.DIRECTORY, FileType.ANY):
                for dname in dirnames:
                    if fnmatch.fnmatch(dname, pattern):
                        full = os.path.join(dirpath, dname)
                        rel = os.path.relpath(full, self._root)
                        try:
                            stat = os.stat(full)
                            entries.append(FileEntry(
                                path=rel,
                                name=dname,
                                file_type="directory",
                                modified=stat.st_mtime,
                            ))
                        except OSError:
                            pass

            # Check files
            if file_type in (FileType.FILE, FileType.ANY):
                for fname in filenames:
                    if not fnmatch.fnmatch(fname, pattern):
                        continue

                    if extension and not fname.endswith(extension):
                        continue

                    full = os.path.join(dirpath, fname)
                    try:
                        stat = os.stat(full)
                    except OSError:
                        continue

                    if min_size is not None and stat.st_size < min_size:
                        continue
                    if max_size is not None and stat.st_size > max_size:
                        continue

                    rel = os.path.relpath(full, self._root)
                    ext = os.path.splitext(fname)[1]

                    entries.append(FileEntry(
                        path=rel,
                        name=fname,
                        file_type="file",
                        size=stat.st_size,
                        modified=stat.st_mtime,
                        extension=ext,
                    ))

        # Sort
        sort_keys = {
            SortBy.NAME: lambda e: e.name.lower(),
            SortBy.SIZE: lambda e: e.size,
            SortBy.MODIFIED: lambda e: e.modified,
            SortBy.PATH: lambda e: e.path.lower(),
        }
        entries.sort(key=sort_keys.get(sort_by, sort_keys[SortBy.PATH]), reverse=reverse)

        result = FindResult(
            pattern=pattern,
            total_found=len(entries),
            total_size=sum(e.size for e in entries),
            duration_ms=(time.time() - t0) * 1000,
        )

        if len(entries) > limit:
            result.truncated = True
            result.entries = entries[:limit]
        else:
            result.entries = entries

        return result

    def find_by_content(
        self,
        text: str,
        file_pattern: str = "*",
    ) -> FindResult:
        """Find files that contain specific text."""
        t0 = time.time()
        entries: List[FileEntry] = []

        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORE]

            for fname in filenames:
                if not fnmatch.fnmatch(fname, file_pattern):
                    continue

                full = os.path.join(dirpath, fname)
                try:
                    with open(full, "r", errors="replace") as f:
                        content = f.read(self._max_results * 100)
                    if text in content:
                        stat = os.stat(full)
                        rel = os.path.relpath(full, self._root)
                        entries.append(FileEntry(
                            path=rel,
                            name=fname,
                            file_type="file",
                            size=stat.st_size,
                            extension=os.path.splitext(fname)[1],
                        ))
                except (OSError, UnicodeDecodeError):
                    pass

        return FindResult(
            pattern=f"content:{text}",
            entries=entries,
            total_found=len(entries),
            duration_ms=(time.time() - t0) * 1000,
        )

    def tree(
        self,
        path: str = "",
        max_depth: int = 3,
    ) -> str:
        """Generate a tree view of the directory structure."""
        root = os.path.join(self._root, path) if path else self._root
        lines = [os.path.basename(root) + "/"]
        self._tree_recurse(root, "", 0, max_depth, lines)
        return "\n".join(lines)

    def _tree_recurse(
        self,
        path: str,
        prefix: str,
        depth: int,
        max_depth: int,
        lines: List[str],
    ):
        if depth >= max_depth:
            return

        try:
            entries = sorted(os.listdir(path))
        except OSError:
            return

        entries = [e for e in entries if e not in DEFAULT_IGNORE and not e.startswith(".")]

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            full = os.path.join(path, entry)

            if os.path.isdir(full):
                lines.append(f"{prefix}{connector}{entry}/")
                ext_prefix = prefix + ("    " if is_last else "│   ")
                self._tree_recurse(full, ext_prefix, depth + 1, max_depth, lines)
            else:
                lines.append(f"{prefix}{connector}{entry}")

    def stats(self) -> Dict[str, Any]:
        return {
            "workspace_root": self._root,
            "max_results": self._max_results,
        }
