"""Shared SQLAlchemy base and imports for all model files."""

from datetime import datetime
from typing import Optional, List
from enum import Enum
import uuid

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, BigInteger, Boolean,
    ForeignKey, Table, Enum as SQLEnum, JSON, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR

try:
    from pgvector.sqlalchemy import Vector as _PgVector
except ImportError:
    _PgVector = None  # Fallback for environments without pgvector


def _get_embedding_dim() -> int:
    """Get configured embedding dimension (avoids circular import at class-body time)."""
    try:
        from app.config import settings
        return settings.embedding_dimension
    except Exception:
        return 1536  # safe default


def Vector(dim: int = 0):
    """Return a pgvector Vector column type with the configured dimension."""
    if _PgVector is None:
        return None
    if dim <= 0:
        dim = _get_embedding_dim()
    return _PgVector(dim)


Base = declarative_base()
