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
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # Fallback for environments without pgvector

Base = declarative_base()
