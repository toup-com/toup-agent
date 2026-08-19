"""The user-facing file library — a VIRTUAL tree over the tenant's files.

Before this manifest existed the "Files" surface listed the agent's
container workspace as-is: UUID-named directories, ``apps/`` /
``vibecoding/`` build trees, ``generated/<scope>/<32hex>_name.pdf`` storage
keys, dotdirs, and every test artefact a harness ever wrote. Users saw
container internals; that is both a trust problem and a leak.

These two tables are what users see instead. Physical placement (the
storage backend under ``generated/``, the per-user workspace root, …) is
an implementation detail recorded in ``storage_key`` and NEVER serialised;
the API speaks ids, display names and virtual folder paths only.

AGENT_ONLY — the bytes live in the tenant container, the manifest lives
next to them in the tenant DB, and the platform is a pass-through proxy
(``app/api/workspace_proxy.py``). ``init_db``'s ``create_all`` builds
these tables on every tenant at boot; no alembic revision — the platform
never reads them (an unreachable agent is a 503, not an empty library).

Invariants (enforced in ``app/services/library_service.py``):
  * A physical file is never moved or renamed by a library operation —
    a rename changes ``name`` only, a move changes ``folder_id`` only. Chat
    attachment pointers (``Message.attachments[].storage_path``) therefore
    keep resolving. Delete removes the bytes AND tombstones the row.
  * Folder names are unique per parent (case-insensitive); system folders
    (``system_key`` set) can be renamed but not deleted.
  * Every read is scoped by ``user_id``; an id that belongs to another
    user is a 404, never a 403.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# System folders — the tree's stable skeleton. New agent output lands in
# one of these by (origin, kind); the user may rename them, never delete.
SYSTEM_FOLDER_DOCUMENTS = "documents"
SYSTEM_FOLDER_IMAGES = "images"
SYSTEM_FOLDER_UPLOADS = "uploads"
SYSTEM_FOLDERS: dict[str, str] = {
    SYSTEM_FOLDER_DOCUMENTS: "Documents",
    SYSTEM_FOLDER_IMAGES: "Images",
    SYSTEM_FOLDER_UPLOADS: "Uploads",
}

# Where a file came from. "upload" = the user sent it (chat attachment on a
# user turn, or the library upload endpoint); "agent" = the agent produced
# it (generate_* tools, image tools, write_file into a deliverable root).
ORIGIN_UPLOAD = "upload"
ORIGIN_AGENT = "agent"
FILE_ORIGINS = (ORIGIN_UPLOAD, ORIGIN_AGENT)


def _uuid() -> str:
    return str(uuid.uuid4())


class UserFolder(Base):
    __tablename__ = "user_folders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    # NULL = a root-level folder.
    parent_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # One of SYSTEM_FOLDERS' keys, or NULL for a user-created folder.
    system_key: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False,
    )

    __table_args__ = (
        Index("ix_user_folders_user_parent", "user_id", "parent_id"),
    )


class UserFile(Base):
    __tablename__ = "user_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    # NULL = at the library root.
    folder_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)

    # Display name — what the user sees and renames. Never a storage key.
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False, default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    # INTERNAL. ``<root>:<relative path>`` where root ∈ {gen, uws, ws} — see
    # library_service.physical_path(). Resolved server-side only; never in
    # a response body.
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)

    origin: Mapped[str] = mapped_column(String(10), nullable=False, default=ORIGIN_AGENT)
    # Provenance when the file is a chat attachment (lets the library and
    # the chat card agree on one physical file).
    source_message_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    source_attachment_id: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    # created_at = when the file came into existence for the user (the
    # message timestamp for attachments, the mtime for imported files);
    # modified_at = the bytes' last-modified time. Both UTC.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    # Set when the bytes were deleted (via the API, or found missing on a
    # sync). Kept so a re-scan does not resurrect a deleted attachment.
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("user_id", "storage_key", name="uq_user_files_user_key"),
        Index("ix_user_files_user_folder", "user_id", "folder_id"),
    )
