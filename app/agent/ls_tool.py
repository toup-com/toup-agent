"""
LS Tool — Dedicated directory listing for the workspace.

Provides structured directory listings with file metadata,
filtering, sorting, and human-readable formatting.

Usage:
    from app.agent.ls_tool import LsTool

    tool = LsTool(workspace_root="/path/to/workspace")
    result = tool.ls("src/")
    result = tool.ls(".", show_hidden=True, sort_by="size")
"""

import logging
import os
import stat
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LsEntry:
    """A directory listing entry."""
    name: str
    path: str
    is_dir: bool
    size: int = 0
    modified: float = 0.0
    permissions: str = ""
    extension: str = ""
    is_symlink: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "type": "directory" if self.is_dir else "file",
            "size": self.size,
            "extension": self.extension,
            "is_symlink": self.is_symlink,
        }

    def format_long(self) -> str:
        """Format as ls -l style output."""
        type_char = "d" if self.is_dir else "-"
        size_str = self._human_size(self.size) if not self.is_dir else "-"
        name = f"{self.name}/" if self.is_dir else self.name
        return f"{type_char}{self.permissions}  {size_str:>8s}  {name}"

    @staticmethod
    def _human_size(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


@dataclass
class LsResult:
    """Result of a directory listing."""
    directory: str
    entries: List[LsEntry] = field(default_factory=list)
    total_files: int = 0
    total_dirs: int = 0
    total_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directory": self.directory,
            "total_files": self.total_files,
            "total_dirs": self.total_dirs,
            "total_size": self.total_size,
            "entries": [e.to_dict() for e in self.entries],
        }

    def format(self, long_format: bool = False) -> str:
        lines = [f"Directory: {self.directory}"]
        lines.append(f"  {self.total_dirs} directories, {self.total_files} files")
        lines.append("")
        for e in self.entries:
            if long_format:
                lines.append(f"  {e.format_long()}")
            else:
                name = f"{e.name}/" if e.is_dir else e.name
                lines.append(f"  {name}")
        return "\n".join(lines)


class LsTool:
    """
    Directory listing tool for the workspace.

    Lists directory contents with metadata and formatting options.
    """

    def __init__(self, workspace_root: str = "."):
        self._root = os.path.abspath(workspace_root)

    def ls(
        self,
        path: str = ".",
        *,
        show_hidden: bool = False,
        sort_by: str = "name",
        reverse: bool = False,
        dirs_first: bool = True,
        filter_ext: Optional[str] = None,
    ) -> LsResult:
        """
        List directory contents.

        Args:
            path: Directory to list (relative to workspace root).
            show_hidden: Include hidden files (starting with .).
            sort_by: Sort by 'name', 'size', or 'modified'.
            reverse: Reverse sort order.
            dirs_first: List directories before files.
            filter_ext: Filter by file extension.
        """
        full_path = os.path.join(self._root, path)
        if not os.path.isdir(full_path):
            return LsResult(directory=path)

        entries: List[LsEntry] = []
        rel_dir = os.path.relpath(full_path, self._root)

        try:
            items = os.listdir(full_path)
        except OSError:
            return LsResult(directory=path)

        for item in items:
            if not show_hidden and item.startswith("."):
                continue

            item_path = os.path.join(full_path, item)
            try:
                st = os.lstat(item_path)
            except OSError:
                continue

            is_dir = os.path.isdir(item_path)
            is_symlink = os.path.islink(item_path)
            ext = os.path.splitext(item)[1] if not is_dir else ""

            if filter_ext and not is_dir and ext != filter_ext:
                continue

            # Permission string
            mode = st.st_mode
            perms = ""
            for who in (stat.S_IRUSR, stat.S_IWUSR, stat.S_IXUSR,
                        stat.S_IRGRP, stat.S_IWGRP, stat.S_IXGRP,
                        stat.S_IROTH, stat.S_IWOTH, stat.S_IXOTH):
                perms += "r" if (who in (stat.S_IRUSR, stat.S_IRGRP, stat.S_IROTH) and mode & who) else \
                         "w" if (who in (stat.S_IWUSR, stat.S_IWGRP, stat.S_IWOTH) and mode & who) else \
                         "x" if (who in (stat.S_IXUSR, stat.S_IXGRP, stat.S_IXOTH) and mode & who) else \
                         "-"

            rel_item = os.path.join(rel_dir, item) if rel_dir != "." else item

            entries.append(LsEntry(
                name=item,
                path=rel_item,
                is_dir=is_dir,
                size=st.st_size if not is_dir else 0,
                modified=st.st_mtime,
                permissions=perms,
                extension=ext,
                is_symlink=is_symlink,
            ))

        # Sort
        sort_keys = {
            "name": lambda e: e.name.lower(),
            "size": lambda e: e.size,
            "modified": lambda e: e.modified,
        }
        key_fn = sort_keys.get(sort_by, sort_keys["name"])

        if dirs_first:
            dirs = sorted([e for e in entries if e.is_dir], key=key_fn, reverse=reverse)
            files = sorted([e for e in entries if not e.is_dir], key=key_fn, reverse=reverse)
            entries = dirs + files
        else:
            entries.sort(key=key_fn, reverse=reverse)

        result = LsResult(
            directory=rel_dir if rel_dir != "." else path,
            entries=entries,
            total_files=sum(1 for e in entries if not e.is_dir),
            total_dirs=sum(1 for e in entries if e.is_dir),
            total_size=sum(e.size for e in entries),
        )

        return result

    def exists(self, path: str) -> bool:
        """Check if a path exists in the workspace."""
        return os.path.exists(os.path.join(self._root, path))

    def is_dir(self, path: str) -> bool:
        """Check if a path is a directory."""
        return os.path.isdir(os.path.join(self._root, path))

    def file_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a single file."""
        full = os.path.join(self._root, path)
        if not os.path.exists(full):
            return None

        st = os.stat(full)
        return {
            "path": path,
            "name": os.path.basename(path),
            "is_dir": os.path.isdir(full),
            "size": st.st_size,
            "modified": st.st_mtime,
            "extension": os.path.splitext(path)[1],
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "workspace_root": self._root,
        }
