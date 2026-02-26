"""
Grep Tool — Dedicated ripgrep-style file content search.

Provides a structured search interface for finding text patterns
in files within the workspace, with support for regex, file type
filtering, context lines, and result limiting.

Usage:
    from app.agent.grep_tool import GrepTool

    tool = GrepTool(workspace_root="/path/to/workspace")
    results = tool.search("def main", file_pattern="*.py")
    results = tool.search(r"TODO|FIXME", is_regex=True, context_lines=2)
"""

import fnmatch
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GrepMatch:
    """A single grep match."""
    file_path: str
    line_number: int
    line_content: str
    column: int = 0
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "file": self.file_path,
            "line": self.line_number,
            "column": self.column,
            "content": self.line_content.rstrip(),
        }
        if self.context_before:
            d["context_before"] = [l.rstrip() for l in self.context_before]
        if self.context_after:
            d["context_after"] = [l.rstrip() for l in self.context_after]
        return d

    def format(self) -> str:
        return f"{self.file_path}:{self.line_number}: {self.line_content.rstrip()}"


@dataclass
class GrepResult:
    """Result of a grep search."""
    query: str
    matches: List[GrepMatch] = field(default_factory=list)
    files_searched: int = 0
    files_matched: int = 0
    total_matches: int = 0
    duration_ms: float = 0.0
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_matches": self.total_matches,
            "files_searched": self.files_searched,
            "files_matched": self.files_matched,
            "truncated": self.truncated,
            "duration_ms": round(self.duration_ms, 1),
            "matches": [m.to_dict() for m in self.matches],
        }

    def format(self) -> str:
        lines = [f"Found {self.total_matches} matches in {self.files_matched} files"]
        for m in self.matches:
            lines.append(m.format())
        if self.truncated:
            lines.append(f"... (truncated, showing {len(self.matches)} of {self.total_matches})")
        return "\n".join(lines)


# Default ignore patterns
DEFAULT_IGNORE = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".eggs", "*.egg-info", ".DS_Store",
}


class GrepTool:
    """
    Ripgrep-style file content search for the workspace.

    Searches file contents with support for plain text and regex,
    file type filtering, context lines, and result limits.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        max_results: int = 200,
        max_file_size: int = 1_000_000,  # 1MB
    ):
        self._root = os.path.abspath(workspace_root)
        self._max_results = max_results
        self._max_file_size = max_file_size

    def search(
        self,
        pattern: str,
        *,
        is_regex: bool = False,
        case_sensitive: bool = True,
        file_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        context_lines: int = 0,
        max_results: Optional[int] = None,
        search_path: Optional[str] = None,
    ) -> GrepResult:
        """
        Search for a pattern in workspace files.

        Args:
            pattern: Search pattern (plain text or regex).
            is_regex: Whether pattern is a regex.
            case_sensitive: Whether to match case.
            file_pattern: Glob pattern to filter files (e.g., "*.py").
            exclude_pattern: Glob pattern to exclude files.
            context_lines: Number of context lines before/after match.
            max_results: Maximum matches to return.
            search_path: Subdirectory to search in.
        """
        t0 = time.time()
        limit = max_results or self._max_results

        # Compile pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            if is_regex:
                regex = re.compile(pattern, flags)
            else:
                regex = re.compile(re.escape(pattern), flags)
        except re.error as e:
            return GrepResult(query=pattern, duration_ms=0)

        root = os.path.join(self._root, search_path) if search_path else self._root
        result = GrepResult(query=pattern)
        all_matches: List[GrepMatch] = []

        for fpath in self._iter_files(root, file_pattern, exclude_pattern):
            result.files_searched += 1
            file_matches = self._search_file(fpath, regex, context_lines)

            if file_matches:
                result.files_matched += 1
                all_matches.extend(file_matches)

                if len(all_matches) >= limit:
                    result.truncated = True
                    break

        result.total_matches = len(all_matches)
        result.matches = all_matches[:limit]
        result.duration_ms = (time.time() - t0) * 1000
        return result

    def count(
        self,
        pattern: str,
        *,
        is_regex: bool = False,
        file_pattern: Optional[str] = None,
    ) -> int:
        """Count matches without returning details."""
        result = self.search(
            pattern,
            is_regex=is_regex,
            file_pattern=file_pattern,
            max_results=100000,
        )
        return result.total_matches

    def _iter_files(
        self,
        root: str,
        file_pattern: Optional[str],
        exclude_pattern: Optional[str],
    ):
        """Iterate over searchable files."""
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip ignored directories
            dirnames[:] = [
                d for d in dirnames
                if d not in DEFAULT_IGNORE
            ]

            for fname in filenames:
                if file_pattern and not fnmatch.fnmatch(fname, file_pattern):
                    continue
                if exclude_pattern and fnmatch.fnmatch(fname, exclude_pattern):
                    continue

                full = os.path.join(dirpath, fname)
                try:
                    if os.path.getsize(full) > self._max_file_size:
                        continue
                except OSError:
                    continue

                rel = os.path.relpath(full, self._root)
                yield full

    def _search_file(
        self,
        filepath: str,
        regex: re.Pattern,
        context_lines: int,
    ) -> List[GrepMatch]:
        """Search a single file."""
        try:
            with open(filepath, "r", errors="replace") as f:
                lines = f.readlines()
        except (OSError, UnicodeDecodeError):
            return []

        matches = []
        rel_path = os.path.relpath(filepath, self._root)

        for i, line in enumerate(lines):
            m = regex.search(line)
            if m:
                ctx_before = []
                ctx_after = []

                if context_lines > 0:
                    start = max(0, i - context_lines)
                    ctx_before = [lines[j] for j in range(start, i)]
                    end = min(len(lines), i + context_lines + 1)
                    ctx_after = [lines[j] for j in range(i + 1, end)]

                matches.append(GrepMatch(
                    file_path=rel_path,
                    line_number=i + 1,
                    line_content=line,
                    column=m.start() + 1,
                    context_before=ctx_before,
                    context_after=ctx_after,
                ))

        return matches

    def stats(self) -> Dict[str, Any]:
        return {
            "workspace_root": self._root,
            "max_results": self._max_results,
            "max_file_size": self._max_file_size,
        }
