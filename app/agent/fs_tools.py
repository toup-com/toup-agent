"""
File System Tools — apply_patch, grep, find, ls as dedicated agent tools.
Provides structured file operations instead of raw exec wrappers.
"""

import fnmatch
import logging
import os
import re
import stat
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GrepMatch:
    """A single grep match."""
    file: str
    line_number: int
    line: str
    match_start: int = 0
    match_end: int = 0

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line_number": self.line_number,
            "line": self.line,
        }


@dataclass
class FileInfo:
    """File information for ls."""
    name: str
    path: str
    is_dir: bool
    size: int
    modified: float
    permissions: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified": self.modified,
            "permissions": self.permissions,
        }


def apply_patch(file_path: str, patch: str) -> dict:
    """Apply a unified diff patch to a file.
    
    Args:
        file_path: Path to the file to patch
        patch: Unified diff content
        
    Returns:
        Dict with success status and details
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        with open(file_path, "r") as f:
            original_lines = f.readlines()

        hunks = _parse_unified_diff(patch)
        if not hunks:
            return {"success": False, "error": "No valid hunks found in patch"}

        result_lines = list(original_lines)
        offset = 0

        for hunk in hunks:
            start = hunk["old_start"] - 1 + offset
            old_count = hunk["old_count"]
            new_lines = hunk["new_lines"]

            result_lines[start:start + old_count] = new_lines
            offset += len(new_lines) - old_count

        with open(file_path, "w") as f:
            f.writelines(result_lines)

        return {
            "success": True,
            "hunks_applied": len(hunks),
            "lines_before": len(original_lines),
            "lines_after": len(result_lines),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _parse_unified_diff(patch: str) -> List[dict]:
    """Parse unified diff format into hunks."""
    hunks = []
    lines = patch.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        # Look for @@ -old_start,old_count +new_start,new_count @@
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if match:
            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)
            new_start = int(match.group(3))
            new_count = int(match.group(4) or 1)

            new_lines = []
            removed = 0
            added = 0
            i += 1

            while i < len(lines) and not lines[i].startswith("@@"):
                if lines[i].startswith("-"):
                    removed += 1
                elif lines[i].startswith("+"):
                    new_lines.append(lines[i][1:] + "\n")
                    added += 1
                elif lines[i].startswith(" ") or lines[i] == "":
                    content = lines[i][1:] if lines[i].startswith(" ") else lines[i]
                    new_lines.append(content + "\n")
                else:
                    break
                i += 1

            hunks.append({
                "old_start": old_start,
                "old_count": old_count,
                "new_start": new_start,
                "new_count": new_count,
                "new_lines": new_lines,
            })
        else:
            i += 1

    return hunks


def grep(
    pattern: str,
    path: str,
    recursive: bool = True,
    ignore_case: bool = False,
    max_results: int = 100,
    include: Optional[str] = None,
    exclude: Optional[List[str]] = None,
) -> List[GrepMatch]:
    """Search for pattern in files (ripgrep-style).
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search
        recursive: Search recursively
        ignore_case: Case-insensitive search
        max_results: Maximum number of matches
        include: Glob pattern for files to include
        exclude: Glob patterns for files to exclude
    """
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        logger.error(f"Invalid regex: {e}")
        return []

    default_exclude = exclude or [
        "*.pyc", "__pycache__", ".git", "node_modules", ".venv", "*.egg-info"
    ]
    matches = []

    if os.path.isfile(path):
        matches.extend(_grep_file(path, regex, max_results))
    elif os.path.isdir(path):
        for root, dirs, filenames in os.walk(path):
            # Apply exclusions to dirs
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, ex) for ex in default_exclude)]
            
            for fname in filenames:
                if any(fnmatch.fnmatch(fname, ex) for ex in default_exclude):
                    continue
                if include and not fnmatch.fnmatch(fname, include):
                    continue
                
                fpath = os.path.join(root, fname)
                matches.extend(_grep_file(fpath, regex, max_results - len(matches)))
                
                if len(matches) >= max_results:
                    return matches[:max_results]
            
            if not recursive:
                break

    return matches[:max_results]


def _grep_file(file_path: str, regex: re.Pattern, max_results: int) -> List[GrepMatch]:
    """Search for regex in a single file."""
    matches = []
    try:
        with open(file_path, "r", errors="replace") as f:
            for i, line in enumerate(f, 1):
                m = regex.search(line)
                if m:
                    matches.append(GrepMatch(
                        file=file_path,
                        line_number=i,
                        line=line.rstrip(),
                        match_start=m.start(),
                        match_end=m.end(),
                    ))
                    if len(matches) >= max_results:
                        break
    except (IOError, UnicodeDecodeError):
        pass
    return matches


def find(
    path: str,
    name: Optional[str] = None,
    pattern: Optional[str] = None,
    file_type: Optional[str] = None,
    max_depth: Optional[int] = None,
    max_results: int = 200,
) -> List[str]:
    """Find files matching criteria.
    
    Args:
        path: Root directory to search
        name: Exact filename to match
        pattern: Glob pattern for filenames
        file_type: 'f' for files, 'd' for directories
        max_depth: Maximum directory depth
        max_results: Maximum results
    """
    results = []
    base_depth = path.rstrip("/").count("/")

    for root, dirs, files in os.walk(path):
        current_depth = root.rstrip("/").count("/") - base_depth
        
        if max_depth is not None and current_depth > max_depth:
            dirs.clear()
            continue

        # Skip common ignored dirs
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv"}]

        entries = []
        if file_type != "f":
            entries.extend([(d, True) for d in dirs])
        if file_type != "d":
            entries.extend([(f, False) for f in files])

        for entry_name, is_dir in entries:
            if name and entry_name != name:
                continue
            if pattern and not fnmatch.fnmatch(entry_name, pattern):
                continue
            results.append(os.path.join(root, entry_name))
            if len(results) >= max_results:
                return results

    return results


def ls(
    path: str,
    all_files: bool = False,
    long_format: bool = True,
) -> List[FileInfo]:
    """List directory contents.
    
    Args:
        path: Directory to list
        all_files: Include hidden files
        long_format: Include detailed info
    """
    if not os.path.isdir(path):
        return []

    results = []
    try:
        entries = sorted(os.listdir(path))
        for entry in entries:
            if not all_files and entry.startswith("."):
                continue
            full_path = os.path.join(path, entry)
            try:
                st = os.stat(full_path)
                results.append(FileInfo(
                    name=entry,
                    path=full_path,
                    is_dir=os.path.isdir(full_path),
                    size=st.st_size,
                    modified=st.st_mtime,
                    permissions=stat.filemode(st.st_mode),
                ))
            except OSError:
                results.append(FileInfo(
                    name=entry, path=full_path, is_dir=False,
                    size=0, modified=0, permissions="?",
                ))
    except OSError as e:
        logger.error(f"ls failed on {path}: {e}")

    return results
