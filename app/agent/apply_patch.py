"""
Apply Patch — Unified diff patch application tool.

Parses and applies unified diff patches to files. Supports
standard git diff format with context lines, additions,
and deletions.

Usage:
    from app.agent.apply_patch import PatchTool

    tool = PatchTool(workspace_root="/path/to/workspace")
    result = tool.apply(patch_text)
    result = tool.dry_run(patch_text)
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HunkLine:
    """A single line in a patch hunk."""
    type: str  # 'context', 'add', 'remove'
    content: str
    old_lineno: Optional[int] = None
    new_lineno: Optional[int] = None


@dataclass
class Hunk:
    """A patch hunk with line changes."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[HunkLine] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_range": f"{self.old_start},{self.old_count}",
            "new_range": f"{self.new_start},{self.new_count}",
            "lines": len(self.lines),
            "additions": sum(1 for l in self.lines if l.type == "add"),
            "deletions": sum(1 for l in self.lines if l.type == "remove"),
        }


@dataclass
class PatchFile:
    """A single file in a patch."""
    old_path: str
    new_path: str
    hunks: List[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_path": self.old_path,
            "new_path": self.new_path,
            "hunks": len(self.hunks),
            "is_new": self.is_new,
            "is_deleted": self.is_deleted,
            "additions": sum(
                sum(1 for l in h.lines if l.type == "add")
                for h in self.hunks
            ),
            "deletions": sum(
                sum(1 for l in h.lines if l.type == "remove")
                for h in self.hunks
            ),
        }


@dataclass
class PatchResult:
    """Result of applying a patch."""
    success: bool
    files_modified: int = 0
    files_created: int = 0
    files_deleted: int = 0
    total_additions: int = 0
    total_deletions: int = 0
    errors: List[str] = field(default_factory=list)
    file_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "files_deleted": self.files_deleted,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "errors": self.errors,
        }


# Regex for hunk headers
HUNK_HEADER_RE = re.compile(
    r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'
)


class PatchParser:
    """Parses unified diff format patches."""

    def parse(self, patch_text: str) -> List[PatchFile]:
        """Parse a unified diff patch into structured data."""
        files: List[PatchFile] = []
        lines = patch_text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for --- a/path
            if line.startswith('--- '):
                old_path = self._extract_path(line[4:])
                i += 1
                if i < len(lines) and lines[i].startswith('+++ '):
                    new_path = self._extract_path(lines[i][4:])
                    i += 1

                    pf = PatchFile(
                        old_path=old_path,
                        new_path=new_path,
                        is_new=(old_path == '/dev/null'),
                        is_deleted=(new_path == '/dev/null'),
                    )

                    # Parse hunks
                    while i < len(lines):
                        match = HUNK_HEADER_RE.match(lines[i])
                        if match:
                            hunk = self._parse_hunk(lines, i, match)
                            pf.hunks.append(hunk[0])
                            i = hunk[1]
                        elif lines[i].startswith('--- ') or lines[i].startswith('diff '):
                            break
                        else:
                            i += 1

                    files.append(pf)
                    continue

            i += 1

        return files

    def _extract_path(self, path: str) -> str:
        """Extract clean file path from diff header."""
        path = path.strip()
        if path.startswith('a/') or path.startswith('b/'):
            path = path[2:]
        # Remove timestamp if present
        if '\t' in path:
            path = path.split('\t')[0]
        return path

    def _parse_hunk(
        self,
        lines: List[str],
        start: int,
        match: re.Match,
    ) -> Tuple[Hunk, int]:
        """Parse a single hunk starting at the header line."""
        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1

        hunk = Hunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
        )

        i = start + 1
        old_line = old_start
        new_line = new_start

        while i < len(lines):
            line = lines[i]

            if line.startswith('@@') or line.startswith('--- ') or line.startswith('diff '):
                break

            if line.startswith('+'):
                hunk.lines.append(HunkLine(
                    type='add',
                    content=line[1:],
                    new_lineno=new_line,
                ))
                new_line += 1
            elif line.startswith('-'):
                hunk.lines.append(HunkLine(
                    type='remove',
                    content=line[1:],
                    old_lineno=old_line,
                ))
                old_line += 1
            elif line.startswith(' ') or line == '':
                content = line[1:] if line.startswith(' ') else ''
                hunk.lines.append(HunkLine(
                    type='context',
                    content=content,
                    old_lineno=old_line,
                    new_lineno=new_line,
                ))
                old_line += 1
                new_line += 1
            elif line.startswith('\\'):
                # "\ No newline at end of file"
                pass
            else:
                break

            i += 1

        return hunk, i


class PatchTool:
    """
    Applies unified diff patches to files.

    Supports creating new files, modifying existing files,
    and deleting files through patch application.
    """

    def __init__(self, workspace_root: str = "."):
        self._root = workspace_root
        self._parser = PatchParser()

    def parse(self, patch_text: str) -> List[PatchFile]:
        """Parse a patch without applying it."""
        return self._parser.parse(patch_text)

    def dry_run(self, patch_text: str) -> PatchResult:
        """
        Validate a patch without modifying files.

        Returns a result indicating what would change.
        """
        patch_files = self._parser.parse(patch_text)
        result = PatchResult(success=True)

        for pf in patch_files:
            additions = sum(
                sum(1 for l in h.lines if l.type == "add")
                for h in pf.hunks
            )
            deletions = sum(
                sum(1 for l in h.lines if l.type == "remove")
                for h in pf.hunks
            )

            result.total_additions += additions
            result.total_deletions += deletions

            if pf.is_new:
                result.files_created += 1
            elif pf.is_deleted:
                result.files_deleted += 1
            else:
                result.files_modified += 1

            result.file_results.append(pf.to_dict())

        return result

    def apply(self, patch_text: str) -> PatchResult:
        """
        Apply a unified diff patch to the workspace.

        Modifies files on disk according to the patch.
        """
        patch_files = self._parser.parse(patch_text)
        result = PatchResult(success=True)

        for pf in patch_files:
            try:
                file_result = self._apply_file(pf)
                result.file_results.append(file_result)

                if pf.is_new:
                    result.files_created += 1
                elif pf.is_deleted:
                    result.files_deleted += 1
                else:
                    result.files_modified += 1

                result.total_additions += file_result.get("additions", 0)
                result.total_deletions += file_result.get("deletions", 0)

            except Exception as e:
                result.success = False
                result.errors.append(f"{pf.new_path}: {e}")

        return result

    def _apply_file(self, pf: PatchFile) -> Dict[str, Any]:
        """Apply a single file patch."""
        target = pf.new_path if not pf.is_deleted else pf.old_path
        full_path = os.path.join(self._root, target)
        info = pf.to_dict()

        if pf.is_new:
            # Create new file
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            content_lines = []
            for hunk in pf.hunks:
                for line in hunk.lines:
                    if line.type in ('add', 'context'):
                        content_lines.append(line.content)
            with open(full_path, 'w') as f:
                f.write('\n'.join(content_lines))
            info["action"] = "created"

        elif pf.is_deleted:
            if os.path.exists(full_path):
                os.remove(full_path)
            info["action"] = "deleted"

        else:
            # Modify existing file
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File not found: {full_path}")

            with open(full_path, 'r') as f:
                original_lines = f.read().split('\n')

            new_lines = self._apply_hunks(original_lines, pf.hunks)

            with open(full_path, 'w') as f:
                f.write('\n'.join(new_lines))
            info["action"] = "modified"

        return info

    def _apply_hunks(
        self,
        original: List[str],
        hunks: List[Hunk],
    ) -> List[str]:
        """Apply hunks to file content."""
        result = list(original)
        offset = 0

        for hunk in hunks:
            pos = hunk.old_start - 1 + offset  # 0-indexed

            # Remove old lines, add new lines
            remove_count = sum(1 for l in hunk.lines if l.type == 'remove')
            add_lines = [l.content for l in hunk.lines if l.type == 'add']
            context_and_remove = [l for l in hunk.lines if l.type in ('remove', 'context')]

            # Calculate the actual range to replace
            old_end = pos + len(context_and_remove)
            new_content = []
            for line in hunk.lines:
                if line.type in ('add', 'context'):
                    new_content.append(line.content)

            result[pos:old_end] = new_content
            offset += len(new_content) - len(context_and_remove)

        return result

    def stats(self) -> Dict[str, Any]:
        """Get patch tool info."""
        return {
            "workspace_root": self._root,
        }
