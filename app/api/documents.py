"""
Document & Media Ingestion API

Upload and process documents and media files for the HexBrain memory system.
Supports:
- Documents: PDF, Markdown, Text, Code, DOCX, JSON, YAML, CSV
- Media: Images, Videos, Audio
"""

import json
import uuid
import time
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_db, Document, DocumentChunk, Media, Memory
from app.schemas import (
    BrainType, MemoryType, MemoryCreate,
    DocumentUploadRequest, DocumentResponse, IngestDocumentResponse,
    MediaUploadRequest, MediaResponse, IngestMediaResponse,
)
from app.api.auth import get_current_user
from app.services import get_embedding_service, get_document_service
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/ingest", tags=["Document Ingestion"])


# Maximum file sizes
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024     # 20 MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024    # 500 MB


@router.post("/document", response_model=IngestDocumentResponse)
async def ingest_document(
    file: UploadFile = File(...),
    brain_type: str = Form("user"),
    category: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON array as string
    importance: float = Form(0.5),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    extract_entities: bool = Form(True),
    generate_summary: bool = Form(True),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process a document file.
    
    Supported formats:
    - PDF (.pdf)
    - Markdown (.md, .markdown)
    - Text (.txt)
    - Code files (.py, .js, .ts, .java, .go, .rs, .cpp, .c, etc.)
    - Word documents (.docx)
    - JSON (.json)
    - YAML (.yaml, .yml)
    - CSV (.csv)
    
    The document will be:
    1. Chunked into smaller pieces for embedding
    2. Each chunk becomes a searchable memory
    3. Optionally summarized and entities extracted
    """
    start_time = time.time()
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_DOCUMENT_SIZE // (1024*1024)} MB"
        )
    
    document_service = get_document_service()
    embedding_service = get_embedding_service()
    memory_service = MemoryService(db)
    
    # Detect file type
    file_type, mime_type = document_service.detect_file_type(file.filename, content)
    
    # Validate file type
    if file_type not in ['pdf', 'markdown', 'text', 'code', 'docx', 'json', 'yaml', 'csv']:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file_type}. Supported: pdf, markdown, text, code, docx, json, yaml, csv"
        )
    
    # Compute file hash for deduplication
    file_hash = document_service.compute_file_hash(content)
    
    # Check for duplicate
    existing = await db.execute(
        select(Document).where(
            Document.user_id == current_user.id,
            Document.file_hash == file_hash,
            Document.is_deleted == False
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This document has already been uploaded"
        )
    
    # Save file to storage
    file_path = document_service.save_file(content, file.filename, current_user.id)
    
    # Parse tags
    tags_list = []
    if tags:
        try:
            tags_list = json.loads(tags)
        except json.JSONDecodeError:
            tags_list = [t.strip() for t in tags.split(',') if t.strip()]
    
    # Create document record
    doc_id = str(uuid.uuid4())
    document = Document(
        id=doc_id,
        user_id=current_user.id,
        brain_type=brain_type,
        category=category,
        filename=f"{doc_id}_{file.filename}",
        original_filename=file.filename,
        file_type=file_type,
        mime_type=mime_type,
        file_size=len(content),
        file_path=file_path,
        file_hash=file_hash,
        title=title or file.filename,
        description=description,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        importance=importance,
        tags_json=json.dumps(tags_list) if tags_list else None,
        processing_status="processing"
    )
    db.add(document)
    await db.flush()
    
    try:
        # Extract document content and chunks
        extraction = await document_service.extract_document(
            content, file.filename, file_type, chunk_size, chunk_overlap
        )
        
        # Update document metadata
        document.page_count = extraction.page_count
        document.word_count = extraction.word_count
        document.language = extraction.language
        document.encoding = extraction.encoding
        document.programming_language = extraction.programming_language
        document.chunk_count = len(extraction.chunks)
        
        # Generate summary if requested
        if generate_summary and extraction.text:
            summary = await document_service.generate_summary(extraction.text)
            document.summary = summary
            
            # Extract key topics
            key_topics = await document_service.extract_key_topics(extraction.text)
            document.key_topics_json = json.dumps(key_topics) if key_topics else None
        
        memories_created = 0
        entities_extracted = 0
        
        # Create memories from chunks
        for chunk in extraction.chunks:
            # Create memory for this chunk
            memory_content = chunk.content
            
            # Add context info
            if document.title:
                memory_content = f"[From: {document.title}]\n\n{memory_content}"
            
            memory_data = MemoryCreate(
                content=memory_content,
                summary=f"Chunk {chunk.chunk_index + 1} from {document.original_filename}",
                brain_type=BrainType(brain_type),
                category=category,
                memory_type=MemoryType.FILE,
                importance=importance,
                tags=tags_list,
                metadata={
                    "document_id": doc_id,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "source_file": document.original_filename,
                    "file_type": file_type
                }
            )
            
            memory = await memory_service.create_memory(
                current_user.id,
                memory_data,
                source_type="document"
            )
            memories_created += 1
            
            # Create document chunk record
            chunk_record = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                memory_id=memory.id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                page_number=chunk.page_number,
                embedding_json=memory.embedding_json,
                metadata_json=json.dumps(chunk.metadata) if chunk.metadata else None
            )
            db.add(chunk_record)
        
        # Update document with results
        document.memories_created = memories_created
        document.entities_extracted = entities_extracted
        document.processing_status = "completed"
        document.processed_at = datetime.utcnow()
        
        await db.commit()
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return IngestDocumentResponse(
            document_id=doc_id,
            filename=document.original_filename,
            file_type=file_type,
            chunks_processed=len(extraction.chunks),
            memories_created=memories_created,
            entities_extracted=entities_extracted,
            summary=document.summary,
            key_topics=json.loads(document.key_topics_json) if document.key_topics_json else None,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        document.processing_status = "failed"
        document.processing_error = str(e)
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.post("/media", response_model=IngestMediaResponse)
async def ingest_media(
    file: UploadFile = File(...),
    brain_type: str = Form("user"),
    category: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    importance: float = Form(0.5),
    generate_description: bool = Form(True),
    transcribe_audio: bool = Form(True),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process a media file (image, video, audio).
    
    Supported formats:
    - Images: .jpg, .jpeg, .png, .gif, .webp, .bmp, .svg
    - Videos: .mp4, .mov, .avi, .webm, .mkv
    - Audio: .mp3, .wav, .ogg, .flac, .m4a
    
    Processing:
    - Images: AI description generation
    - Videos/Audio: Transcription (when available)
    """
    start_time = time.time()
    
    content = await file.read()
    
    document_service = get_document_service()
    memory_service = MemoryService(db)
    
    # Detect file type
    file_type, mime_type = document_service.detect_file_type(file.filename, content)
    
    # Validate media type
    if file_type not in ['image', 'video', 'audio']:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Not a media file. Detected type: {file_type}"
        )
    
    # Check file size
    max_size = MAX_VIDEO_SIZE if file_type in ['video', 'audio'] else MAX_IMAGE_SIZE
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size for {file_type} is {max_size // (1024*1024)} MB"
        )
    
    # Compute file hash
    file_hash = document_service.compute_file_hash(content)
    
    # Check for duplicate
    existing = await db.execute(
        select(Media).where(
            Media.user_id == current_user.id,
            Media.file_hash == file_hash,
            Media.is_deleted == False
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This media file has already been uploaded"
        )
    
    # Save file
    file_path = document_service.save_file(content, file.filename, current_user.id)
    
    # Parse tags
    tags_list = []
    if tags:
        try:
            tags_list = json.loads(tags)
        except json.JSONDecodeError:
            tags_list = [t.strip() for t in tags.split(',') if t.strip()]
    
    # Create media record
    media_id = str(uuid.uuid4())
    media = Media(
        id=media_id,
        user_id=current_user.id,
        brain_type=brain_type,
        category=category,
        filename=f"{media_id}_{file.filename}",
        original_filename=file.filename,
        media_type=file_type,
        mime_type=mime_type,
        file_size=len(content),
        file_path=file_path,
        file_hash=file_hash,
        title=title or file.filename,
        description=description,
        importance=importance,
        tags_json=json.dumps(tags_list) if tags_list else None,
        processing_status="processing"
    )
    db.add(media)
    await db.flush()
    
    try:
        # Extract media metadata and AI content
        extraction = await document_service.extract_media(
            content, file.filename, file_type,
            generate_description=generate_description,
            transcribe_audio=transcribe_audio
        )
        
        # Update media metadata
        media.width = extraction.width
        media.height = extraction.height
        media.duration = extraction.duration
        media.format = extraction.format
        media.ai_description = extraction.ai_description
        media.ai_transcript = extraction.ai_transcript
        
        memories_created = 0
        
        # Create memory from media
        memory_content_parts = [f"Media file: {media.original_filename}"]
        
        if extraction.ai_description:
            memory_content_parts.append(f"\nDescription: {extraction.ai_description}")
        
        if extraction.ai_transcript:
            memory_content_parts.append(f"\nTranscript: {extraction.ai_transcript}")
        
        if description:
            memory_content_parts.append(f"\nUser description: {description}")
        
        memory_content = "\n".join(memory_content_parts)
        
        memory_data = MemoryCreate(
            content=memory_content,
            summary=f"Media: {media.title or media.original_filename}",
            brain_type=BrainType(brain_type),
            category=category,
            memory_type=MemoryType.FILE,
            importance=importance,
            tags=tags_list,
            metadata={
                "media_id": media_id,
                "media_type": file_type,
                "source_file": media.original_filename,
                "width": extraction.width,
                "height": extraction.height,
                "duration": extraction.duration
            }
        )
        
        memory = await memory_service.create_memory(
            current_user.id,
            memory_data,
            source_type="media"
        )
        media.memory_id = memory.id
        memories_created = 1
        
        # Update media status
        media.memories_created = memories_created
        media.processing_status = "completed"
        media.processed_at = datetime.utcnow()
        
        await db.commit()
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return IngestMediaResponse(
            media_id=media_id,
            filename=media.original_filename,
            media_type=file_type,
            memories_created=memories_created,
            ai_description=extraction.ai_description,
            ai_transcript=extraction.ai_transcript,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        media.processing_status = "failed"
        media.processing_error = str(e)
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process media: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    brain_type: Optional[str] = None,
    category: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List uploaded documents"""
    query = select(Document).where(
        Document.user_id == current_user.id,
        Document.is_deleted == False
    )
    
    if brain_type:
        query = query.where(Document.brain_type == brain_type)
    if category:
        query = query.where(Document.category == category)
    if file_type:
        query = query.where(Document.file_type == file_type)
    
    query = query.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.original_filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            brain_type=doc.brain_type,
            category=doc.category,
            title=doc.title,
            description=doc.description,
            chunk_count=doc.chunk_count,
            memories_created=doc.memories_created,
            entities_extracted=doc.entities_extracted,
            summary=doc.summary,
            key_topics=json.loads(doc.key_topics_json) if doc.key_topics_json else None,
            created_at=doc.created_at,
            processed_at=doc.processed_at
        )
        for doc in documents
    ]


@router.get("/media", response_model=List[MediaResponse])
async def list_media(
    brain_type: Optional[str] = None,
    category: Optional[str] = None,
    media_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List uploaded media files"""
    query = select(Media).where(
        Media.user_id == current_user.id,
        Media.is_deleted == False
    )
    
    if brain_type:
        query = query.where(Media.brain_type == brain_type)
    if category:
        query = query.where(Media.category == category)
    if media_type:
        query = query.where(Media.media_type == media_type)
    
    query = query.order_by(Media.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    media_files = result.scalars().all()
    
    return [
        MediaResponse(
            id=m.id,
            filename=m.original_filename,
            media_type=m.media_type,
            file_size=m.file_size,
            brain_type=m.brain_type,
            category=m.category,
            title=m.title,
            description=m.description,
            width=m.width,
            height=m.height,
            duration=m.duration,
            ai_description=m.ai_description,
            ai_transcript=m.ai_transcript,
            memories_created=m.memories_created,
            created_at=m.created_at,
            processed_at=m.processed_at,
            file_url=f"/api/files/{m.id}" if m.file_path else None,
            thumbnail_url=f"/api/files/{m.id}/thumbnail" if m.thumbnail_path else None
        )
        for m in media_files
    ]


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Soft delete a document and its associated memories"""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id
        )
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    document.is_deleted = True
    document.deleted_at = datetime.utcnow()
    
    # Also soft delete associated memories
    await db.execute(
        Memory.__table__.update().where(
            Memory.id.in_(
                select(DocumentChunk.memory_id).where(
                    DocumentChunk.document_id == document_id
                )
            )
        ).values(is_deleted=True, deleted_at=datetime.utcnow())
    )
    
    await db.commit()
    
    return {"status": "deleted", "document_id": document_id}


@router.delete("/media/{media_id}")
async def delete_media(
    media_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Soft delete a media file and its associated memory"""
    result = await db.execute(
        select(Media).where(
            Media.id == media_id,
            Media.user_id == current_user.id
        )
    )
    media = result.scalar_one_or_none()
    
    if not media:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found"
        )
    
    media.is_deleted = True
    media.deleted_at = datetime.utcnow()
    
    # Also soft delete associated memory
    if media.memory_id:
        result = await db.execute(
            select(Memory).where(Memory.id == media.memory_id)
        )
        memory = result.scalar_one_or_none()
        if memory:
            memory.is_deleted = True
            memory.deleted_at = datetime.utcnow()
    
    await db.commit()
    
    return {"status": "deleted", "media_id": media_id}
