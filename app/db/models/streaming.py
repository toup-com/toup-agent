"""Streaming service credentials (Netflix, Prime, etc.) + plaintext-access audit log."""

from datetime import datetime
from .base import Base, Column, String, DateTime, Text


class StreamingCredential(Base):
    __tablename__ = "streaming_credentials"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    channel = Column(String(50), nullable=False)  # netflix, prime, disney, etc.
    email = Column(String(255), nullable=False)
    password = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CredentialAccessLog(Base):
    # Immutable audit row written BEFORE every plaintext fetch via
    # get_credential_internal. Fail-closed: if this insert can't be committed,
    # the fetch returns 503 and plaintext is never emitted.
    __tablename__ = "credential_access_log"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    channel = Column(String(50), nullable=False)
    agent_key_fingerprint = Column(String(16), nullable=False)  # first 8 chars of X-Agent-Key
    source = Column(String(16), nullable=False, default="http")  # http | direct_db (legacy)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
