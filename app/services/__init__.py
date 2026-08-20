from app.services.embedding_service import EmbeddingService, get_embedding_service
# `memory_extractor` keeps only its RULE-BASED half in v3 — the LLM
# extractor and the relationship extractor are deleted with the row corpus
# they produced. Its two remaining consumers are app/api/ingest.py and
# app/scripts/seed_data.py; the conversation path writes through
# `memory_curator` instead.
from app.services.memory_extractor import MemoryExtractor, get_memory_extractor, ExtractedMemory, ExtractedEntity
from app.services.auth_service import (
    verify_password, get_password_hash, create_access_token,
    decode_access_token, decode_platform_jwt, authenticate_user, create_user,
    get_user_by_id, get_user_by_email, change_user_password,
)
from app.services.memory_service import MemoryService
from app.services.document_service import DocumentService, get_document_service

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "MemoryExtractor",
    "get_memory_extractor",
    "ExtractedMemory",
    "ExtractedEntity",
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "decode_access_token",
    "decode_platform_jwt",
    "authenticate_user",
    "create_user",
    "get_user_by_id",
    "get_user_by_email",
    "MemoryService",
    # NEW: Document processing
    "DocumentService",
    "get_document_service",
]
