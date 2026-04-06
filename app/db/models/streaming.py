"""Streaming service credentials (Netflix, Prime, etc.)."""

from datetime import datetime
from .base import Base, Column, String, DateTime, Text


class StreamingCredential(Base):
    __tablename__ = "streaming_credentials"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    channel = Column(String(50), nullable=False)  # netflix, prime, disney, etc.
    email = Column(String(255), nullable=False)
    password = Column(Text, nullable=False)
    # Session persistence: cold=never logged in, warm=cookies exist, expired=needs re-auth, blocked=captcha/2fa
    session_status = Column(String(20), default="cold")
    session_checked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
