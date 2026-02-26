from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.memory_extractor import MemoryExtractor, get_memory_extractor, ExtractedMemory, ExtractedEntity
from app.services.auth_service import (
    verify_password, get_password_hash, create_access_token,
    decode_access_token, authenticate_user, create_user,
    get_user_by_id, get_user_by_email
)
from app.services.memory_service import MemoryService
from app.services.decay_service import DecayService, get_decay_service
from app.services.consolidation_service import ConsolidationService, get_consolidation_service
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
    "authenticate_user",
    "create_user",
    "get_user_by_id",
    "get_user_by_email",
    "MemoryService",
    # NEW: Decay and Consolidation services
    "DecayService",
    "get_decay_service",
    "ConsolidationService",
    "get_consolidation_service",
    # NEW: Document processing
    "DocumentService",
    "get_document_service",
]
