"""
File storage backend for generated document attachments.

v1: local disk only, rooted at {agent_workspace_dir}/generated/.
v2 (follow-up): S3 backend for multi-node deploys and larger files.

Storage keys are backend-agnostic relative paths (e.g. "abc123_report.pdf").
The backend resolves them — local prepends the workspace root, S3 prepends
a bucket/prefix. DB stores only the relative key in Message.attachments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, BinaryIO, AsyncIterator

from app.config import settings


class StorageBackend(Protocol):
    """Minimal interface for file persistence.

    Implementations MUST be safe to call from async handlers (file I/O
    should be offloaded if it becomes a bottleneck, but for typical
    document sizes synchronous I/O is fine).
    """

    async def put(self, key: str, data: bytes) -> str:
        """Persist bytes under `key`. Returns the canonical key to store in DB."""
        ...

    def open(self, key: str) -> BinaryIO:
        """Open a file handle for streaming. Caller is responsible for closing."""
        ...

    def path(self, key: str) -> str:
        """Absolute filesystem path for `key` (LocalDiskBackend only).

        S3Backend will raise NotImplementedError here — callers that need
        a path must fall back to streaming via `open()`.
        """
        ...

    def exists(self, key: str) -> bool:
        ...

    def size(self, key: str) -> int:
        ...

    async def delete(self, key: str) -> None:
        ...


@dataclass
class LocalDiskBackend:
    """Stores files under {root}/generated/."""

    root: str

    def _full(self, key: str) -> str:
        # Reject path traversal — keys must be simple names, no ".." or absolute paths.
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"Invalid storage key: {key!r}")
        return os.path.join(self.root, "generated", key)

    async def put(self, key: str, data: bytes) -> str:
        full = self._full(key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)
        return key

    def open(self, key: str) -> BinaryIO:
        return open(self._full(key), "rb")

    def path(self, key: str) -> str:
        return self._full(key)

    def exists(self, key: str) -> bool:
        return os.path.isfile(self._full(key))

    def size(self, key: str) -> int:
        return os.path.getsize(self._full(key))

    async def delete(self, key: str) -> None:
        full = self._full(key)
        if os.path.isfile(full):
            os.unlink(full)


class S3Backend:
    """Stub — implement in the follow-up PR using existing boto3 dep."""

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix

    async def put(self, key: str, data: bytes) -> str:
        raise NotImplementedError("S3 backend is a stub — set FILES_STORAGE_BACKEND=local")

    def open(self, key: str) -> BinaryIO:
        raise NotImplementedError

    def path(self, key: str) -> str:
        raise NotImplementedError("S3 keys have no local path — stream via open()")

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def size(self, key: str) -> int:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError


_backend: StorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    """Return the configured storage backend (cached).

    Env: FILES_STORAGE_BACKEND = "local" | "s3" (default "local").
    Local root is settings.agent_workspace_dir.
    """
    global _backend
    if _backend is not None:
        return _backend
    kind = getattr(settings, "files_storage_backend", "local")
    if kind == "local":
        _backend = LocalDiskBackend(root=settings.agent_workspace_dir)
    elif kind == "s3":
        _backend = S3Backend(bucket=os.environ.get("FILES_S3_BUCKET", ""))
    else:
        raise ValueError(f"Unknown FILES_STORAGE_BACKEND: {kind}")
    return _backend


async def stream_file(key: str, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
    """Async generator yielding file chunks — for FastAPI StreamingResponse."""
    backend = get_storage_backend()
    f = backend.open(key)
    try:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        f.close()
